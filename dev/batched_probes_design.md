# Batched probes — design note

*2026-09-02 (Nick + Claude). Status: **design settled and independently reviewed; implemented in all three hooks
(§11 steps 0–3) and adopted by T3Polynomial's call sites**. Remaining follow-ups: §10. Decisions in §9; the review record — two independent
reviewers, same brief, no shared context, findings folded in below — is §12. Measurements behind
it: T3Polynomial's `scripts/x03_batched_probe_bench.py` and
`dev/deq_regime_and_probe_batching_2026_09_02.md` §5, the PETSc checks in §1/§6, and the reviewers'
scratch experiments (§12).*

## 1. Why

At one expansion point the D `(direction set, omega)` probes are independent, while the point itself
costs a sequential nonlinear solve. `probe` handles ONE probe per call and walks the lattice
sequentially (`driver.py`), so every caller loops in Python — T3Polynomial's `datagen` and `n01`, the
gradient sketch in its `reduction.py`, and `examples/jax_deq_input.py` all do. Two measurements say
the loop is the wrong shape:

- **JAX, dispatch-bound at small state dim.** Trained DEQ, state dim 256, input 16, output 8, J = 4,
  laptop CPU: the loop costs ~92 ms per probe on ~2 Mflop of arithmetic. `jax.vmap` over the
  single-direction `probe` call, no library change, results equal to the loop **to round-off**
  (max rel diff 5e-16 … 1e-15 — not bit-identical):

  | D at one point | loop, ms/probe | jit(vmap), ms/probe | plain vmap, ms/probe | speedup (jit) |
  |---|---|---|---|---|
  | 64 | 92.7 | 2.5–2.7 | 12.0 | 37× |
  | 256 | 89.4 | 1.4 | — | 62× |
  | 1024 | 92.3 | 0.9 | 2.5 | 104× |

  Still falling at D = 1024 on 8 cores. The point (Picard to 1e-12 + Jacobian + LU) is 9 ms.
  (Reviewer B's rerun of the same script under load: loop 189 / plain vmap 15.1 / jit(vmap) 3.1 ms
  at D = 64 — same ratios.)
- **FEniCSx / PETSc, solve-bound at PDE scale.** T3Polynomial's Darcy at nx = 60: 0.53 s per direction
  probe of which 7 × 0.06 s are linear solves (~80%). Multi-right-hand-side solves through the same
  factorization, 40k-dof nonsymmetric sparse operator, B = 64, petsc4py 3.25.2 (fenicsx env):

  | KSP configuration | 64 single solves | `KSP.matSolve` | `KSP.matSolveTranspose` |
  |---|---|---|---|
  | LU via MUMPS | 934 ms | 89 ms (10.5×) | 83 ms (11.3×) |
  | LU, PETSc native | 148 ms | 119 ms (1.2×) | 350 ms (1.1× vs its own transpose loop) |
  | GMRES + ILU | 14.9 s | 14.7 s (1.0×) | 16.5 s (1.0×) |

  Residuals 1e-12 … 4e-15 throughout; iterative KSPs loop internally and stay correct. **The MUMPS row
  is reachable only if the hook can be handed a MUMPS KSP** — today it cannot (§6, KSP factory): the
  hook's default is native LU, and Darcy supplies MUMPS as opaque `Vec -> Vec` closures.

Mechanically both hooks have the same structure — one nonlinear solve per point, `2J+1` linear solves
per direction against a shared factorization — and the direction axis is the one that batches.

## 2. The design in one paragraph

**A batch is a vector type.** The driver is vector-type agnostic by construction (it does no
arithmetic on physics vectors), so it is **untouched**: `probe(problem, directions, omega)` keeps its
signature and its lattice walk, and performs one `solve_operator` / `solve_operator_adjoint` call per
node whether or not the vectors are batched. Each hook defines what a *batched vector* is and
implements the three protocol methods for it. There is no new function and no flag.

## 3. The contract (all hooks)

- **Batched vector.** Hook-specific: JAX and numpy — a leading batch axis, directions `(B, p)`,
  `omega` `(B, n_q)`, incrementals `(B, n_u)`; FEniCSx — a Python `list` of B Functions (directions in
  `V_theta`, `omega` in `V_q`), lists of Functions in `V_u` for incrementals, lists of Vecs for outputs.
  **2-D is always a batch, even B = 1** (a `(1, n_q)` row covector is a batch of one). Today every 2-D
  input to the JAX and numpy hooks raises, so no working code changes meaning. Hooks **raise** on
  `ndim > 2` / non-list sequences rather than guess.
- **Broadcasting.** An unbatched vector in a request that also carries batched vectors is shared by
  every batch member. So one call batches over directions at fixed `omega`, over `omega` at a fixed
  direction, or over both, and a mixed request `((V, 2), (b, 1))` with `V` batched and `b` single is
  legal.
- **Batch size.** Inferred per `assemble_partial_sum` call from the batched inputs *the request uses*:
  the `theta_dirs` and `u_vecs` vectors of its terms, an adjoint `pairing` vector by its own shape,
  and **`omega` only through terms whose `pairing is OMEGA`** — the driver passes `omega` to every
  call, including forward and state-RHS requests that never use it, and reading its shape there would
  wrongly batch the forward chain under `omega`-only batching. Batched inputs with different B
  **raise** (structural error, house philosophy); the driver cannot validate up front, so the raise
  surfaces mid-walk after the first batched solves, and the message names both sizes. A hook holds no
  batch state. A request in which every form vanishes (FEniCSx `fenics.py:114-115`) still returns a
  batch of B zero vectors, so B must be inferred even when nothing is assembled.
- **Output batching — per key, not per family.** A result is batched iff its request had a batched
  input. Traced on the real driver walk (both reviewers, six direction/omega patterns), that reads:

  | probe | batched iff |
  |---|---|
  | `forward[mu]` | some batched direction `k` has `mu_k ≥ 1` |
  | `reverse[mu]` | `omega` is batched, or some batched direction `k` has `mu_k ≥ 1` |

  Consequences: `forward[(0, …)] = q(theta0)` is never batched; under **direction-only** batching the
  gradient `reverse[(0, …)]` is *also* unbatched (its request holds only `omega` and `vhat_{}`); in a
  **mixed** request every key with power 0 on the batched axis is unbatched, forward and reverse;
  under **`omega`-only** batching every forward probe is single and every reverse probe batched. All
  of this is mathematically right (those probes do not depend on the batched vectors) but it makes
  the returned dicts **ragged across keys**, and the driver cannot broadcast (it would have to know a
  batch is in flight for a request with no batched input). numpy consumers writing
  `Y[:, t] = forward[(t,)]` get the broadcast for free; `jnp.stack` over keys raises; **for FEniCSx a
  single Vec where a list was expected indexes silently by entry** (`fwd[(0,)][b]` is entry b). Stated
  loudly in every hook's docs and asserted per key in the tests. An optional hook-level
  `broadcast(vec, B)` helper for callers may be added; a driver-level `Batch` marker (which would let
  the driver validate B up front and broadcast) is held in reserve, since it conflicts with the
  driver-untouched ruling.
- **Solve counts.** One batched solve per lattice node — the efficiency invariant the driver's
  `TestSolveCounts` protects, extended to batched calls.
- **Memory.** Every node's incremental (and adjoint) stays alive for the walk: roughly
  `2 · (#nodes) · B · n_state · 8` bytes (single direction: `#nodes = J+1`), e.g. ~200 MB at nx = 60,
  J = 4, B = 64, plus the dense RHS/solution blocks. **Chunking is the caller's job**: fixed chunk
  size, **pad the last chunk** (every distinct B recompiles every kernel — §4); no automatic chunking,
  no bucketing of B (Nick, 2026-09-02 — "maybe later").
- **Nomenclature (settled with Nick, 2026-09-02).** "Batch" currently means *many expansion points*
  in exactly one place — the `refreeze` docstring and its CHANGELOG entry — while ARCHITECTURE.md
  already calls that "multi-point gathering", and T3Polynomial uses "minibatch" for the optimizer's
  row subsets and "batched" in the numpy-vectorization sense. Adopted:

  | term | meaning |
  |---|---|
  | **batch**, batched probes, batch size B, batch axis | B independent `(direction set, omega)` probes at **one** expansion point, sharing its factorization, handled in one `probe` call; all members share the same `max_power`s, so one lattice serves the batch. The standard numerics/jax meaning (multi-RHS = "a batch of right-hand sides"; vmap's batch axis). |
  | **multi-point** probing, the multi-point loop | probing at many expansion points, sequential, one `refreeze` per point. Adopts ARCHITECTURE.md's existing "multi-point gathering". |
  | **minibatch** (T3Polynomial only) | the optimizer-side row subset of the fitting data (d07, open-issues 22); never shortened to "batch" in the fitting context. |
  | *not used* | "sweep" (already a verification loop over probe cases here, the TT sweep in T3Toolbox, an experiment grid in T3Polynomial) and "vectorized" (collides with "vector" as the physics vector the driver is agnostic to). |

  In T3Polynomial's vocabulary an S × D dataset cell is S multi-point iterations, each a batch of D
  probes. The two existing phrases "probing a batch of expansion points" (`refreeze` docstring) and
  "batch-probing many expansion points" (CHANGELOG, Unreleased) are reworded to multi-point (§8).

## 4. JAX hook (`implicit_probing/jax.py`)

- **Batched kernel.** `assemble_partial_sum` infers B (§3), lifts every direction/incremental to the
  stacked space, and calls a batched twin of `_term_value`. **Construction matters** (both reviewers
  verified the naive form fails): `jax.vmap` maps keyword arguments over axis 0, so the static
  structure `(F, mults, open_slot, p)` cannot pass *through* a vmap. The working shape is jit
  **outside** with `static_argnames`, bind the statics into a `functools.partial` **inside** the jitted
  body, then `vmap` with the expansion point `w` shared (`in_axes=None`) — cache key
  `(F, mults, open_slot, p, shapes incl. B)`, exactly what is wanted. Prefer **per-leaf `in_axes`**
  (`0` for batched leaves, `None` for shared ones, chosen from `ndim` inside the wrapper) over
  materializing `(B, p+n_u)` copies of shared vectors: it avoids B× traffic per term for shared
  directions and lets XLA skip replicated work when only the pairing is batched; the cost is one cache
  entry per batched-leaf pattern (a `single in_axes=0` over a mixed tuple raises). The `pairing=None`
  case, an empty `dir_vecs` with a batched pairing, and `jax.grad` of the pairing-contracted scalar
  under vmap all verified fine. **Cache the lifts per request by `id`** (the `ComposedProblem.C`
  pattern): `_lift_theta`/`_lift_u` are per term today and would otherwise copy `(B, p+n_u)` twice per
  term evaluation (~1 GB of traffic per J = 4 batch at B = 1024, n_u = 256).
- **Single path unchanged.** Requests with no batched input call the *literal* existing kernel
  (round-off-identical to today — "bit-identical" is not claimed anywhere; the batched twin differs from
  the looped kernel by ≤ 1e-15 in the reviewers' checks).
- **Solves.** Always the rows convention: `lu_solve(self._lu, b.T).T` (`trans=1` for the adjoint), or
  equivalently `lu_solve(lu, b[..., None])[..., 0]`; **never infer orientation from shape** — a `(B, n)`
  right-hand side with `B == n` is indistinguishable by shape and the wrong orientation is silently
  wrong. User-supplied `forward_solver` / `adjoint_solver` callables keep their vector contract and
  are applied per row.
- `composition.MatrixOperator`: 2-D input → `v @ M.T` / `w @ M` (verified for dense and CSR; scipy's
  `__rmatmul__` handles a JAX `v`). This also fixes a latent hazard: today `M @ V` with a square `V`
  is silently transposed. Custom `LinearOperator`s used with batched problems must accept a leading
  batch axis themselves (documented). `ComposedProblem`'s `id(d)`-keyed cache and the driver's
  `id(vector)` distinct-axis check are unaffected by batched arrays.
- **Recompilation facts** (Nick's question): point change via `refreeze` → **no** recompile (the point
  is traced, as today; verified: cached call after refreeze); new `JaxImplicitProblem` instance →
  recompile (closures are the static key, as today); **new B → every kernel structure recompiles**,
  and there are many: a single-direction J = 4 probe has **58 distinct kernel structures** (J = 2: 22,
  J = 6: 118, `a²b`: 34) at 0.1–1.3 s each at DEQ scale — tens of seconds per new B, hence "fixed
  chunk size, pad the last chunk" (§3).
- **The outer-jit trap, to be documented** (`docs/jax_hook.md`, `refreeze` docstring): wrapping a
  hand-rolled vmap in an outer `jax.jit` that closes over the problem bakes the frozen point into the
  compiled program as a constant, and after `refreeze` it silently returns the OLD point's jets
  (measured rel. error 0.5–1.0). The in-hook batching is immune because `w` is an argument.
- **What the JAX half buys — stated honestly.** The first landing of in-hook batching is the *plain
  vmap* band of §1 (batched kernels, eager lattice walk): ~12–15 ms/probe at B = 64 on this laptop,
  which a three-line user-side `jax.vmap` over `probe` already delivers with zero library change — and
  user-side vmap additionally returns *uniform* batched outputs (`forward[(0,)]` comes back `(B, n_q)`)
  and batches `ComposedProblem` / any JAX-traceable `LinearOperator` for free. Reviewer A expects the
  in-hook path to do better (a `jit(vmap(k))` kernel dispatches ~3× faster than `vmap(jit(k))`, and the
  eager ops between kernels run on the C++ fast path instead of the batching interpreter, so plain
  vmap's ~0.6 s per-call floor should drop toward the sequential walk's ~90 ms); reviewer B's
  per-kernel measurement (0.21 ms/member batched vs 0.45 single at B = 64) puts it in the same band.
  **Step 0 of implementation was that measurement** (`x03`, library arm added; same run, same
  machine, ms/probe; every arm agrees with the loop to ≤ 1e-15):

  | D | loop | user-side vmap | **in-hook batch** | outer jit(vmap) |
  |---|---|---|---|---|
  | 8 | 61 | 44 | **25** | 4.9 |
  | 64 | 63 | 12 | **11** | 2.8 |
  | 1024 | 63 | 5.0 | **2.8** | 0.96 |

  So the in-hook path is 1–1.8× better than user-side vmap (reviewer A's expectation, modestly) and
  2.4–22× better than the loop; the outer jit stays 3–5× ahead and is exactly the stale-point trap.
  The shape contract is adopted for **API uniformity across the three hooks** (Nick's ruling) and
  because for FEniCSx it is the only option. The *jit(vmap)* column of §1 (0.9 ms at B = 1024) needs the whole
  lattice walk compiled as a function of the point — the "compiled probe" follow-up, whose enabling
  change is small and worth naming: a **point-as-argument view** (e.g. `problem.at(theta0, u0)`
  returning a lightweight object carrying `(w0, lu)` as a pytree argument instead of closing over
  `self`), which is also what makes a user's own outer `jit` refreeze-safe.

## 5. numpy reference hook (`implicit_probing/reference_problems.py`)

Same leading-axis contract, so the reference implementation of the protocol demonstrates batching
and serves as the exact oracle for the JAX tests. **Implementation: loop over members with the
existing single-member code and stack** ("the toy is tiny; clarity over speed") — both reviewers
found the obvious vectorized ports silently wrong: `tensordot` against a `(B, in_dim)` direction leaves
the batch axis trailing so the *next* direction contracts it (a B × B outer product for two batched
directions), and `pairing @ block` with `(B, out)` × `(B, out, slot)` returns `(B, B, slot)`. The
oracle must itself be tested against the unbatched path per member, including two batched directions
and the open-slot + adjoint-pairing case. Solves: `np.linalg.solve(A, b.T).T` — `solve(A, b)` with a
`(B, n_u)` right-hand side is silently wrong when `B == n_u`.

## 6. FEniCSx hook (`implicit_probing/fenics.py`)

- **KSP factory injection (Nick, 2026-09-02 — phase 1, not optional).** Reviewer A's blocking finding:
  `_lu_solver` (`fenics.py:40-46`) builds `preonly` + `lu` with PETSc's *default* factor solver (native
  LU, verified), never calls `setFromOptions`, and there is no way to hand the hook a KSP; Darcy
  (`t3polynomial/darcy.py:333-338`) gets MUMPS by passing `forward_solver`/`adjoint_solver` closures
  around its own KSP (`_mumps_lu`). As first drafted, the batched path would therefore deliver the
  1.2× native-LU row for the default and **nothing for Darcy** (closures take the loop fallback), and
  native LU does not run under MPI at all (PETSc error 92). Fix: `FenicsImplicitProblem(...,
  ksp_factory=None)` — a callable `PETSc.Mat -> PETSc.KSP` invoked on the assembled `A` (exactly the
  shape of Darcy's `_mumps_lu`). *First shipped with the default unchanged (native LU); then, on
  Nick's ruling, the default became `direct_lu()`: MUMPS where `PETSc.Sys.hasExternalPackage('mumps')`,
  native LU otherwise — pivoting and MPI over the serial single-solve speed of native LU.* No
  `setFromOptions` on the default — users wanting options pass a factory. Precedence: explicit `forward_solver`/`adjoint_solver` callables win (per-member
  loop, `Vec -> Vec` contract kept); else the factory's KSP; else the default KSP. Darcy switches to
  `ksp_factory=_mumps_lu` and gets the 10× row; cubic Poisson may do the same for MPI. The §6 draft
  sentence "options set through the PETSc options database keep applying" was false and is withdrawn.
- **Solves** — solver-agnostic on the hook's KSP: stack the batch's right-hand sides into a dense
  PETSc `Mat` with `A`'s row layout, call **`ksp.matSolve(B, X)` / `matSolveTranspose`** (both bound in
  petsc4py 3.25; verified against MUMPS, native LU and GMRES+ILU serially, and against MUMPS on two
  ranks — §1 table; `B` and `X` must be different matrices), then wrap the solution columns as
  Functions in `V_u` (copy + `scatter_forward`, as `_wrap` does today). Direct solvers get the
  factor-traversal win; iterative KSPs loop internally and stay correct. Doc note: a MUMPS multi-RHS
  solve is a different MUMPS code path from NRHS = 1 (per-solve options such as iterative refinement
  may not apply); agreement with looped solves measured at 1e-15. (Checked and NOT needed: flipping
  MUMPS `ICNTL(9)` — works, MUMPS-only; factoring `A^T` separately — works, 10.9×, second
  factorization.)
- **Assembly into the batch — no column views.** Both reviewers found the drafted "assemble into a
  `getDenseColumnVec` view" **fails under MPI**: the view has no ghost region while `assemble_vector`
  writes through `b.localForm()` (2 ranks: PETSc error 62 then SEGV), and PETSc permits one
  outstanding column view per Mat. Fix: keep the current per-member recipe on a **ghosted scratch
  Vec** (`assemble_vector` → `ghostUpdate(ADD, REVERSE)` → `set_bc` → `ghostUpdate(INSERT, FORWARD)`),
  then copy the owned block into the dense matrix's local array (`Bm.getDenseArray()[:, j] =
  vec.array`; `Vec.array` on a ghosted Vec is the owned part). Column views are never handed out as
  driver vectors.
- **Assembly cost** — two parts, treated differently:
  1. *Form construction* (nesting `ufl.derivative`, the UFL passes inside `fem.form`) — grows with term
     complexity at high order and is batchable: build the combined form **once per request** with
     *slot* coefficient Functions (one per distinct vector position in the request's terms), `fem.form`
     once, then per member copy that member's dofs into the slots (`x.array[:]` + `scatter_forward`)
     and assemble. Verified correct by reviewer B (exact serially, ≤ 2e-17 on two ranks, including two
     nested derivatives on one slot): `fem.form` binds coefficient *objects*, the standard DOLFINx
     pattern. **Not amortized: coefficient packing** — `assemble_vector` re-packs on every call and
     must, since the slot values change per member. Cost of the trick: `#slots × B` `scatter_forward`
     collectives per request. Unbatched requests take the unchanged path.
  2. *Numeric integration* over the mesh — inherently B×. Parallelism for it is **MPI mesh
     partitioning**, which composes with batching. A blocked-function-space "vectorized" form was
     considered and rejected: the kernel and its ffcx compile time scale with B, arithmetic does not
     shrink.
  **Measured (2026-09-02, laptop, Darcy nx = 60 / 18k state dofs / MUMPS via `ksp_factory`, J = 4,
  B = 64; one batched probe vs a loop of 64 single probes, agreement 2e-13):**

  | | wall | form construction + other | integration (`_assemble` calls) | solves |
  |---|---|---|---|---|
  | loop of 64 | 44.3 s, 692 ms/probe | 31.6 s | 8.6 s (1216) | 4.1 s (576) |
  | batched | **14.1 s, 221 ms/probe** | 5.3 s | 8.4 s (1153) | 0.5 s (9 multi-RHS) |

  So on this machine the loop was **71% form construction, 20% integration, 9% solves** — not the
  80%-solves split the HANDOFF numbers suggested (those solves were 60 ms each on another box; here
  7 ms). The slot-form construction removes 83% of the first, `matSolve` 88% of the third, and the
  integration is B× in both, as predicted. 3.1× per probe at B = 64, amortizing further with B.
- **Order ≤ 1 matrix path (phase 2, optional).** A request whose every term is degree ≤ 1 in the
  batched vectors is a matrix product: assemble `d_theta R`, `d_u Q`, `d_theta Q` once per point and
  the whole first-order lattice — including the gradient — is sparse matrix × dense batch. This is
  exactly the many-`omega` gradient sketch. Changes assembly counts, not solve counts. Deferred.
- **MPI.** With the fixes above nothing is serial-only: Functions live on the partitioned mesh, the
  dense RHS/solution matrices use `A`'s row layout (`MPIDENSE`), assembly stays on ghosted vectors.
  Verified on two ranks: `matSolve`/`matSolveTranspose` (MUMPS) and the slot-form trick; the full
  batched walk is not yet run in parallel, so the `mpirun -n 2` smoke test on the cubic-Poisson test
  problem **must exercise assembly, not just the solve**. Caveat: MUMPS centralizes a dense
  right-hand side on rank 0, so a batch costs `n_u × B` memory on one rank (~20 MB at 40k × 64) on top
  of the §3 incrementals — another reason chunking is the caller's knob.
- **Composition.** `ComposedProblem` passes lists through. A FEniCS-type `LinearOperator` (e.g.
  T3Polynomial's covariance whitening) must accept lists to batch; downstream work.

## 7. Tests

All batched-vs-looped comparisons use tolerances (`assert_allclose`), not equality (§1, round-off).

- **JAX** (`tests/test_jax.py`): batched probes vs looped single probes vs the numpy reference at every
  order, symmetric and asymmetric direction patterns; broadcasting (shared direction, shared `omega`);
  **per-key shapes**: direction-only batching asserting `forward[(0,)]` and `reverse[(0,)]` single, the
  mixed request `((V, 2), (b, 1))` asserting every key, `omega`-only batching asserting forward single
  / reverse batched (and the `omega=None` path untouched); correctness after `refreeze`; B = 1;
  `ndim > 2` raises; batch-size mismatch raises (late, message names both sizes); **unbatched path
  unchanged** (a kernel-count or golden-value regression at 1e-15); custom-solver per-row fallback;
  `ComposedProblem` + `MatrixOperator` with 2-D inputs, dense and CSR, incl. `omega`-only through a
  composed problem (`probe(composed, [], OM)`, the `sketch_gradients` shape); solve counts unchanged;
  `validation.reverse_forward_adjointness` applied per member (its pairing lambdas take floats).
- **numpy** (`tests/test_reference_problems.py`, `tests/test_driver.py`): the batched reference vs the
  unbatched reference per member, incl. two batched directions and the open-slot + adjoint-pairing
  case; the driver's batched probe on the toy vs loop; `TestSolveCounts` gains a batched case (one
  solve per node regardless of B).
- **FEniCSx** (`tests/test_fenics.py`): the existing cubic-Poisson problem, batched vs looped forward
  and reverse at every order; solve counts; the vanished-form fallback returns a list of B zero Vecs;
  `forward[(0,)]` is a single Vec (asserted, documented); mixed-B raise; custom callables take the loop
  fallback; the `ksp_factory` path with MUMPS; serial vs `mpirun -n 2` agreement **including
  assembly**.

## 8. Docs and changelog

`docs/jax_hook.md` "Batched probes" (contract; **2-D is always a batch, even B = 1**; the per-key
batchedness table and the ragged-output consequence; broadcasting; compile behaviour incl. the
kernel count per new B; the outer-jit trap; chunking: fixed size, pad the last chunk; memory formula);
`docs/fenics_hook.md` (list contract; the **single-Vec-where-a-list-was-expected trap** stated loudly;
`ksp_factory`; the `matSolve` path and the MUMPS NRHS > 1 note; MPI/memory caveat; what batching does
and does not accelerate); `docs/overview.md` protocol section (one paragraph: a hook may accept
batched vectors, the driver is agnostic); `docs/composition.md` (2-D `MatrixOperator`; custom
operators must handle batches); `CHANGELOG.md` Unreleased / Added; `probe` docstring: one sentence
pointing at the hook docs; `docs/jax_hook.md:55-56` shape sentences updated. **Nomenclature edits
(§3)**: `refreeze` docstring "probing a batch of expansion points" → "multi-point probing (many
expansion points)", plus the outer-jit trap note; CHANGELOG Unreleased "batch-probing many expansion
points" → "multi-point probing"; T3Polynomial `dev/open_issues_2026_08_27.md` item 2 gains one line:
"minibatch" is the optimizer's row subset, "batch" is reserved for batched probes.
`examples/jax_deq_input.py` gains a batched call. Downstream: T3Polynomial `datagen`'s "resume is
bit-identical to a fresh run" promise holds only within one probing mode (batched vs looped differ at
1e-16) — record the mode in `_CONFIG_KEYS` or relax the wording.

## 9. Decisions (Nick, 2026-09-02)

| decision | ruling |
|---|---|
| API | **shape contract** — batched vectors in, batched probes out; no `probe_batch`, no flag |
| broadcasting of unbatched inputs | **allowed** |
| hooks | **all three**: JAX, numpy reference, FEniCSx |
| batch-size changes | recompile accepted; **no bucketing** for now |
| solver generality (FEniCSx) | via `KSP.matSolve` / `matSolveTranspose` on the hook's KSP — any solver, no option removed |
| FEniCSx solver injection | **`ksp_factory`** (`PETSc.Mat -> PETSc.KSP`), phase 1; explicit callables keep the loop fallback |
| FEniCSx default solver (Nick, 2026-09-02, after implementation) | **direct LU, MUMPS where PETSc has it, native LU fallback** (`direct_lu()`); "KSP" means PETSc's solver object, direct or iterative -- the name stays, the docs say so |
| chunking | caller's responsibility, documented (fixed size, pad the last chunk) |
| driver | untouched |

## 10. Out of scope / follow-ups

- **Compiled probe** (JAX): jit the whole lattice walk with the point as an argument — the last ~3–5×
  (0.96 vs 2.4–2.8 ms/probe at B = 1024). Enabling change: the point-as-argument view (§4).
  **Where that gap lives (measured 2026-09-02, sync after every kernel, DEQ J = 4):** Python +
  dispatch is a flat 62–96 ms per call — 7.7 ms/probe at D = 8 but 0.09 ms/probe at D = 1024 — while
  the 205 kernel calls per probe (195 batched term kernels, 1 single, 9 solves) alone take 2.4–4.2×
  the outer-jit wall. So the gap is NOT Python overhead: at small D it is XLA's per-executable launch
  cost (0.85 ms per call on tiny arrays), at large D it is the work itself — each term kernel
  recomputes its own Taylor-mode jets through R/Q and materializes its result, whereas the fused
  program shares the point-side evaluations across all terms and nodes and fuses the elementwise
  chains. Fewer dispatches will not close it; a larger compiled unit will.
  **Compile-time scaling of the whole-probe jit (measured 2026-09-02, DEQ, D = 64, laptop; per-term
  = today's kernels; break-even = probes after which the whole-probe jit has repaid its extra
  compile):**

  | pattern | nodes | kernel structures | per-term: first call / steady | whole probe: first call / steady | speedup | break-even |
  |---|---|---|---|---|---|---|
  | (4,) | 5 | 58 | 12.6 s / 8.9 ms | **10.0 s** / 2.5 ms | 3.6× | none (compiles faster) |
  | (6,) | 7 | 117 | 43 s / 42 ms | 56 s / 9.0 ms | 4.7× | ~400 probes |
  | (2, 2) | 9 | 51 | 17.5 s / 42.5 ms | 27.5 s / 6.1 ms | 7.0× | ~275 probes |
  | (3, 3) | 16 | 111 | 87 s / 275 ms | **472 s** / 36.6 ms | 7.5× | ~1,600 probes |

  The whole-program compile grows superlinearly with lattice size (XLA's passes are superlinear in
  program size; (3,3) tripped XLA's "very slow compile" alarm), while the run-time win *grows* with
  pattern complexity. A middle option — one kernel per `assemble_partial_sum` request, the analogue
  of the FEniCSx combined form — was prototyped and rejected: 2× at D = 8 (launch count) but only 4%
  at D = 1024, because the redundant work is across requests, not within them. **jax's persistent
  compilation cache** (`jax_compilation_cache_dir`) cuts the (4,) first call from 10.3 s to 4.6 s in a
  second process (the remainder is tracing, which no cache skips); 1.4 MB on disk.
  **Composability with custom solvers** (Nick's concern): inside the jit the point is traced, so a
  solver must be built from the traced operator — a factory `A -> (solve, solve_adjoint)` of
  jax-traceable functions (the JAX twin of `ksp_factory`; the default LU and `jax.scipy.sparse.linalg`
  solvers qualify); host solvers (scipy/PETSc) bridge through `jax.pure_callback` with a
  call-time-resolved holder for the per-point state; today's point-bound closures must be refused.
- Bucketing of B; automatic chunking; a hook-level `broadcast(vec, B)` helper; a driver-level `Batch`
  marker (held in reserve, §3).
- FEniCSx order-≤1 matrix path (§6, phase 2); batched custom-solver callables (`Mat -> Mat`, an
  optional attribute) — less pressing now that `ksp_factory` covers the KSP case.
- ~~Downstream in T3Polynomial~~ — **done 2026-09-02**: `n01.generate_jets`,
  `datagen._generate_one_point` and `reduction.sketch_gradients` issue one batched probe per point;
  `darcy.py` passes `ksp_factory=_mumps_lu`; the covariance, `ReducedInputMap` and both output maps
  accept `(B, ·)` blocks / lists; the datagen resume wording qualified. Agreement with the old loops
  at round-off; T3Polynomial's PDE-side and jets tests green.

## 11. Implementation order

0. ~~JAX prototype measurement~~ — **done** (§4 table): in-hook batching 1–1.8× over user-side vmap.
1. JAX + numpy reference + `MatrixOperator` + tests + docs, as one unit — **implemented 2026-09-02**
   (`implicit_probing/batching.py`, `jax.py`, `reference_problems.py`, `composition.py`; tests in
   `test_jax.py` / `test_driver.py` / `test_reference_problems.py` / `test_composition.py`;
   `docs/jax_hook.md` "Batched probes", `docs/overview.md`, `docs/composition.md`, CHANGELOG).
2. FEniCSx — **implemented 2026-09-02** (`fenics.py`: `ksp_factory`, list batches, `matSolve` /
   `matSolveTranspose` on dense blocks with A's row layout filled from ghosted scratch vectors, the
   combined form built once per request with slot coefficients; tests in `test_fenics.py` and a
   list-aware composed case in `test_fenics_composition.py`; Darcy passes `ksp_factory=_mumps_lu`).
   The slot forms went in with 2a rather than after it, because batched assembly needs per-member
   assembly anyway and the measurement (§6) shows form construction was the dominant cost.
3. MPI: the batched FEniCSx tests pass under `mpirun -n 2` (assembly included) — **done**.
   Changelog, docs (`docs/fenics_hook.md`) — **done**. Downstream T3Polynomial call sites — **done**.

## 12. Review record (2026-09-02)

Two independent reviewers (general-purpose agents, identical briefs, no shared context, no access to
the authors' reasoning), each running its own experiments in the t3toolbox and fenicsx envs including
`mpirun -n 2`; scratch under the session scratchpad `reviewA/`, `reviewB/`. No repository file was
modified by either.

**Found by both, independently** (all folded in above): the MPI failure of column-view assembly (§6);
the per-key output-batching rule and the ragged-output consequence (§3); `omega` counted only via
`OMEGA` pairings and the vanished-form batch (§3); the jit-outside/vmap-inside construction (§4);
round-off, not bit-identity (§1, §7); the numpy vectorized ports being silently wrong (§5); kernel
count per new B and the padded-chunk rule (§3, §4); solves-then-measure-then-slot-forms ordering
(§11); lift caching (§4); in-hook JAX batching landing near plain vmap (§4).
**Reviewer A only**: the KSP default / Darcy-closure hole (§6, blocking); coefficient packing not
amortized (§6); the datagen resume promise (§8).
**Reviewer B only**: the FEniCSx single-Vec-vs-list trap (§3); the simplicity challenge and the
point-as-argument view (§4); the 58-structure kernel count (§4); the memory formula (§3).
**Verified fine by both**: the driver trace under every batching pattern (one consistent B per
request, one solve per node); `lu_solve` on `b.T` and `trans=1`; the batched kernel twin vs the looped
kernel (≤ 1e-15); `refreeze` hits the cache; the outer-jit trap reproduces; `KSP.matSolve` /
`matSolveTranspose` present and correct (serial and two ranks, MUMPS); the slot-Function trick; the
`id()`-based checks; `MatrixOperator` 2-D forms; backward compatibility (no 2-D input works today);
the nomenclature table and the proposed rewordings.
