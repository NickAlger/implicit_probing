# implicit_probing — handoff / current state

## Where we are

**Slice 1 complete: the symbolic differentiation engine (Algorithm 1, t4s.pdf Section 4).** Pure
Python, no numpy.

- `implicit_probing/multiset.py` — immutable, hashable `Multiset` (elements may themselves be
  `Multiset`s) + `subset_lattice` (every sub-multiset of `alpha`, nondecreasing cardinality).
- `implicit_probing/symbolic.py` — `Pairing` (`ID` / `OMEGA` / `adjoint(delta)`), `Term`
  `(rho, tau, mu, Gamma)`, the three seeds (`seed_forward_q`, `seed_residual_r`, `seed_reverse`),
  `differentiate_term` (eqs 19-20), and `differentiate_over_lattice` (Algorithm 1). Plus
  `format_term` / `format_expansion` for human-readable debugging output.
- Tests: `tests/test_multiset.py`, `tests/test_symbolic.py` — validated against the
  paper's worked example `D^2 q theta_1^2` (Section 4.2), the reverse seed + adjoint-order raise (eq
  20), the step-(19c) multiplicity factor, the isolated `A uhat_beta` residual term, and lattice
  sizes `prod(alpha_i + 1)`.

**Reference problem + finite-difference ground truth (the test fixture for Algorithm 2).**

- `implicit_probing/reference_problems.py` — a reusable module of toy implicitly-defined polynomial
  maps with *exact* derivatives:
  - `Polynomial` — vector-valued polynomial in symmetric Taylor-coefficient form; exact directional
    mixed partials of all orders by tensor contraction (the homogenization view).
  - `ImplicitPolynomialProblem` — `q(theta) = Q(theta, u(theta))`, `R(theta,u)=0`; exposes exactly
    what the driver needs: `solve_state`, `A`/`solve_A`/`solve_A_adjoint`, `partial_R`/`partial_Q`
    (directional mixed partials), `omega`, and the end-to-end `q(theta)`. **This already pins down the
    slice-2 problem interface.**
  - `make_toy_problem` — deterministic total-degree-3 toy (theta in R^2, u in R^3, q in R^2), with a
    dominant well-conditioned u-linear block (A cond ~1.2) and theta-dependent u/u^2 coefficients.
  - `forward_probe_by_finite_difference` — **independent** ground truth for a forward probe of ANY
    symmetry (distinct / repeated / mixed directions), by tensor-product central differences of
    `q(theta0 + sum_k s_k d_k)` with Richardson extrapolation. Touches only R, Q, and the solve.
- Tests: `tests/test_reference_problems.py` — Polynomial vs FD + algebraic properties (symmetry,
  vanishing above degree, linearity, jacobian); the toy's solve/operator/partials; and the FD ground
  truth matched to Q's exact partials on an explicit map for symmetric/partial/asymmetric order-3
  probes, plus a first-order end-to-end check on the implicit map (agrees to ~1e-14).

**Slice 2 complete: the numeric driver (Algorithm 2) — the core method now runs end-to-end.**

- `implicit_probing/symbolic.py` (added) — `extract_state_rhs` / `extract_adjoint_rhs`:
  isolate the operator term (`A uhat_beta` / `A* vhat_beta`) and return the rest (eqs 10-11, 17-18).
- `implicit_probing/driver.py` — the **vector-type-agnostic** driver:
  - `PartialTerm` — the lowered request DTO: `(coefficient, function R/Q, theta_dirs, u_vecs,
    open_slot in {None,'theta','u'}, pairing in {None, OMEGA, vhat-vector})`. `theta_dirs`/`u_vecs` are
    `(vector, multiplicity)` pairs (the partial is symmetric in each block, so the directions are a
    multiset, not a sequence — encoding multiplicity rather than flattening lets an AD backend push one
    order-`m` jet per direction; the reference + FEniCS hooks just expand it back to a flat list).
  - `OMEGA` — sentinel meaning "pair the output with the problem's omega functional".
  - `ImplicitProblem` — the 3-method `Protocol`: `solve_operator`, `solve_operator_adjoint`,
    `assemble_partial_sum`. The driver does NO arithmetic on physics vectors; it only resolves
    symbolic labels to vectors, hands whole sums to the problem, and routes the opaque results.
  - `probe(problem, directions, omega=None)` -> `(forward, reverse)` dicts keyed by power-tuples `mu`,
    one per sub-probe. Forward = output-space vector; reverse = parameter-space covector (gradient-like).
- `implicit_probing/reference_problems.py` (extended) — `Polynomial.derivative_open_slot` (one open
  derivative slot) and the `ImplicitProblem` hook on the toy (`solve_operator` /
  `solve_operator_adjoint` / `assemble_partial_sum`), a reference implementation of the interface.
- Tests `tests/test_driver.py` — extraction unit tests, probe structure, and the driver's
  probes vs the FD ground truth across symmetric / partially-symmetric / fully-asymmetric probes
  (orders 1-3, ~1e-9 agreement), reverse-probe identity `psi_beta . d = omega(D^{|beta|+1} q)`, the
  order-4 vanishing-top-partial edge case, the `PartialTerm` multiplicity representation, and
  **`TestSolveCounts`** — wraps the toy in a counting problem and asserts the *minimal* solve count
  (`prod(p_k+1) - 1` forward, `prod(p_k+1)` adjoint) over total/partial/no repetition. Efficiency, not
  just numerical correctness, is tested: a correct-but-wasteful traversal would pass every value check
  but fail this one.

Full suite green: **53 tests + 29 subtests, ~0.5s**. Run:
`PYTHONPATH=$PWD <env-python> -m pytest tests/ -q`.

**User-facing docs.** `docs/overview.md` — a short tour: the idea (probe huge, implicit derivative
tensors via shared-operator solves), the module map, and how to use `probe(...)` + implement the
`ImplicitProblem` interface (`ImplicitPolynomialProblem` is the template). Example snippets are
run-verified.

**FEniCS (DOLFINx) hook — step 1 done.** `implicit_probing/fenics.py` — `FenicsImplicitProblem`
implements the `ImplicitProblem` interface for a modern-FEniCS PDE, **frozen** at a user-supplied
expansion point `(theta, u)` (the user does the nonlinear solve outside, with the *real* BCs; the
class only takes the *homogenized* BCs). One uniform recipe turns each `PartialTerm` into a UFL form:
pair the output test function with `omega` / an adjoint vector via `ufl.replace`; nest `ufl.derivative`
for filled directions; one more `ufl.derivative` *with no direction* for the open slot. Forms are
summed per request and assembled once. Pluggable `forward_solver`/`adjoint_solver`; default is one
reused LU (adjoint via transpose solve). Mixed spaces used on purpose (example: theta CG2, u CG3,
observation test function CG1) to catch space-conflation bugs.
- Test `tests/test_fenics.py` (dolfinx-gated via `pytest.importorskip`; runs in the `fenicsx` conda
  env, skips in `t3toolbox`): forward probes vs finite differences (symmetric/asymmetric, ~1e-8) and
  reverse probes vs `omega`-paired forward probes (exact discrete adjointness, ~1e-11, swept over the
  lattice with no extra solves).
- Env: a dedicated **`fenicsx`** conda env (DOLFINx 0.11) with implicit_probing `pip install -e` +
  pytest. DOLFINx is conda-only (no pip extra). The numpy-only core is unaffected.
- **Step 2 done:** `examples/fenics_poisson.py` (the full nonlinear-Poisson example: probe sweep + FD
  validation + reverse adjointness + optional pyvista viz of the state and QoI-gradient fields) and
  `docs/fenics_hook.md` (the hook's interface mapping, the one-recipe form builder, BC handling,
  pluggable solvers, validation strategy, and the conda-env install).

**`omega` is a per-probe argument (not a problem attribute), and linear composition.**

- `probe(problem, directions, omega=None)`; `omega` is threaded to
  `assemble_partial_sum(self, terms, omega)`; problems no longer store it. `omega=None` computes
  forward probes only (skips the adjoint solves + reverse pass). The `OMEGA` sentinel stays in
  `PartialTerm.pairing` so a problem can tell an output-functional pairing from an incremental adjoint
  -- which is exactly what composition relies on.
- `implicit_probing/composition.py` -- `ComposedProblem(inner, input_map=C, output_map=W)`
  probes the composed map `f = W o q o C` for linear input/output maps: pre-maps directions by `C`,
  post-maps a forward probe by `W` and a reverse probe by `C^T`, pulls `omega` back by `W^T`. The
  inner problem and the driver are unchanged; it is itself an `ImplicitProblem`, so compositions nest.
  Plus a `LinearOperator` protocol and a `MatrixOperator` (numpy) adapter.
- Tests: `tests/test_composition.py` (toy: numpy `C`/`W` vs FD, reverse adjointness on the
  composed map, identity-maps recover the inner) and `tests/test_fenics_composition.py` (gated: theta
  from low-order polynomial features + observation restricted to boundary dofs). Example:
  `examples/fenics_composition.py` (the same FEniCS setup, with the dimension-reduction story + the
  per-feature QoI gradient). Docs: `docs/composition.md`.

**Verification helpers promoted to the library, and the examples restructured.**

- `implicit_probing/validation.py` -- vector-agnostic, testing-only (a probe is exact and far cheaper,
  so the real workflow never finite-differences). `forward_probe_by_finite_difference(q, theta0,
  direction_orders, *, perturb=..., h, richardson)` (tensor-product central differences,
  Richardson-extrapolated) and `reverse_forward_adjointness(forward, reverse, directions, omega,
  *, pair_input, pair_output)` (max rel error of the exact identity `reverse[mu].d_k ==
  omega.forward[mu+e_k]`). Array arithmetic lives in the `perturb` / `pair` hooks (numpy defaults), so
  the toy needs no hooks and FEniCS/PETSc supplies its own. `validation` itself imports no numpy.
- `reference_problems.forward_probe_by_finite_difference` is now a thin numpy wrapper over it
  (signature + toy tests unchanged). The 4 hand-rolled FEniCS FD copies + stencil tables are gone.
- Examples are linear scripts split into labelled `PDE SETUP` / `PROBING` / `RESULTS` / verification
  banners so the ~5 probing lines stand out. New `examples/toy_polynomial.py` (numpy-only on-ramp).
  The FEniCS examples now foreground the value prop (exact, shared-operator linear solves) and end with
  a one-call `validation` cross-check. Verified by running both examples in the `fenicsx` env.
- `tests/test_validation.py` (new) covers the helpers on the toy, incl. a custom `perturb` hook path.
  FEniCS tests call `validation` instead of re-pasting stencils. Full suite green in both envs
  (`t3toolbox`: 67 passed / 2 skipped; `fenicsx`: 72 passed, nothing skipped).

**JAX hook (Taylor-mode automatic differentiation) -- done.**

- `implicit_probing/jax.py` -- `JaxImplicitProblem(R, Q, theta0, u0)` for a map whose `R(theta, u)` /
  `Q(theta, u)` are plain JAX callables. **Frozen** at `(theta0, u0)` like the FEniCS hook (user does
  the nonlinear/equilibrium solve outside); assembles `A = d_u R` via `jax.jacfwd` and LU-factorizes it
  once (`jax.scipy.linalg.lu_factor`; adjoint = `lu_solve(..., trans=1)`). Pluggable
  `forward_solver`/`adjoint_solver`. Optional **pip extra** `[jax]`; the numpy core never imports JAX.
- Each `PartialTerm` -> a directional mixed partial by **Taylor-mode `jax.experimental.jet`**: lift
  each direction into the stacked `w=(theta,u)`, fold one order-`m` jet per distinct direction (the
  multiplicity payoff -- `O(j^2)`, not the `O(2^j)` of nested `jvp`); the open slot is one reverse-mode
  `grad` of `pairing . partial(w)` sliced to the theta-/u-block. `jet` returns derivatives directly (no
  `1/k!`), verified empirically; `jet`-in-`jet` and `grad`/`jacfwd`-through-`jet` all compose.
- **Performance is structure-keyed `jit`**: the per-term kernel is `jax.jit` with the *structure*
  (function, multiplicities tuple, open slot, `p`) **static** and the direction *vectors* **traced**,
  so one compiled kernel serves every lattice node + direction value of that structure. Without it XLA
  recompiles per distinct vector (closure-captured constants) and high order is unaffordable; with it,
  same-structure probes are instant. Remaining cost is one-time XLA *compile* of each high-order jet
  kernel (tens of seconds at order 3+); runtime after compile is trivial. Eager (`disable_jit`) is
  catastrophically slow for nested high-order jets -- jit is required.
- **Kernel reuse / canonical ordering (maintainer's call).** Structurally-equivalent partials must hit
  ONE kernel. Two tiers: (1) *within-block* descending-multiplicity ordering -- universal, hoisted to
  the driver (`driver._canonical` in `_lower`), now part of the `PartialTerm` contract + tested
  (`test_each_block_is_in_canonical_descending_multiplicity_order`); (2) the *theta/u merge* -- valid
  only for a stacked-variable AD backend (`jet` differentiates along whole-`w` directions; UFL cannot),
  so it stays in the JAX hook. Measured on the `a^2 b` toy probe: 40 -> 34 distinct kernels, the gap
  widening at higher order.
- `examples/jax_deq.py` -- a **Deep Equilibrium Model** (fixed-point RNN): `u = tanh(W u + U x + b)`,
  `theta = W` (flattened), `q = C u`. Probes the Taylor expansion of the equilibrium output in the
  recurrent weights; labelled sections + a one-call `validation` cross-check (FD ~1e-9, adjointness
  ~1e-14). `docs/jax_hook.md` documents the interface, the jet recipe, the structure-keyed jit, BC-free
  freezing, and the x64 requirement.
- `tests/test_jax.py` (gated via `pytest.importorskip("jax")`; runs in `t3toolbox`, x64 enabled): the
  hook's probes vs the **numpy reference** on the same polynomials re-coded in JAX (exact, ~1e-15 at
  every order -- same driver, different partial machinery), vs finite differences at low order, the
  exact reverse/forward adjoint identity, and the minimal solve count. `t3toolbox` full suite now
  **74 passed / 2 skipped** (the JAX compile makes it ~45s; the numpy core alone stays ~1s).

## Design decisions locked

- Package / repo / import name: `implicit_probing` (GitHub renamed; local `origin` updated; local
  directory renamed to `implicit_probing` to match — repo, import package, and folder now all agree).
- **Flat, pure-functional package** (refactored from the original `backend/` subpackage). Small
  top-level modules (`multiset`, `symbolic`, `driver`, `composition`, `reference_problems`, `fenics`);
  the public API is re-exported from `implicit_probing/__init__.py` (`probe`, `ImplicitProblem`,
  `PartialTerm`, `OMEGA`, `Multiset`, `subset_lattice`, `ComposedProblem`, `LinearOperator`,
  `MatrixOperator`). **No backend/frontend split and no separate OO frontend** — the `probe(...)` free
  function plus the `ImplicitProblem` protocol is the API (this supersedes the original "backend now,
  OO frontend later" plan). Dependency isolation comes from the import graph: `__init__` imports only
  the dependency-free + numpy modules, so `import implicit_probing` needs nothing but the stdlib;
  `fenics` (and a future `jax`) are explicit submodule imports, never pulled in by `__init__`.
  `reference_problems` stays an explicit `implicit_probing.reference_problems` import (numpy-only),
  not in the top-level `__all__`.
- **Folder dependency rule.** `implicit_probing/` (the importable library) is stable and may be
  imported by any folder; `tests/`, `examples/`, `docs/` never import one another. Shared code is
  therefore either promoted into the library (if generic — e.g. `validation.py`) or duplicated across
  leaf folders (if problem-specific — e.g. a specific FEniCS PDE, written out in both example and test,
  legible vs. decisive). Finite differences are testing-only; the real workflow uses exact probes.
- Deps: `numpy` required (used by `reference_problems`; the probing core needs none), `jax` an optional
  extra; **no T3Toolbox dependency**.
- `theta_0` will be a first-class input (multi-point gathering = trivial outer loop), for the
  maintainer's future global-polynomial work.
- **`PartialTerm` directions encode multiplicity** (`theta_dirs`/`u_vecs` are `(vector, multiplicity)`
  pairs, not flat repeated tuples). Rationale: mixed partials commute, so each partial is symmetric in
  its theta- and u-slots — the directions are a *multiset*, and the flat tuple encoded a spurious
  order. The multiplicity is exactly what an AD backend (the planned JAX hook) wants. (`probe`'s own
  inputs/outputs use a *separate* scheme — see the next bullet; `Multiset` stays purely internal, and
  `(vector, multiplicity)` pairs appear only past `_lower`, where labels become unhashable vectors.)
- **`probe` speaks `(vector, max_power)` pairs in, power-tuples out** — `Multiset` never crosses the
  boundary. `probe(problem, directions, omega)` with `directions = ((a, 2), (b, 1))` (the distinct axes
  + how far to probe each); returns dicts keyed by power-tuples, `forward[(2,1)] = D^3 q [a^2 b]`.
  Rationale (first-principles, after much back-and-forth): a forward probe *is* a mixed partial /
  Taylor coefficient of `q` restricted to the slice `theta0 + sum_k s_k d_k`, so its natural name is
  the differentiation multi-index `mu` — which is also exactly what the downstream Taylor /
  Tucker-tensor-train fitting indexes by. The label-`Multiset` mirrors the *algorithm* (paper Section 4)
  and remains the internal index; the power-tuple mirrors the *result* and is the boundary index.
  `probe` assigns position-labels to the directions, runs the unchanged engine, and translates the
  `Multiset` keys -> power-tuples on the way out. (Supersedes the earlier "keep `Multiset` at the probe
  boundary" decision.) `Multiset` / `subset_lattice` stay exported for advanced/engine use.

The core method is complete: symbolic engine + numeric driver, validated end-to-end against finite
differences. Candidate next slices (maintainer to choose):

1. **Autodiff-framework hooks.** FEniCS/DOLFINx hook **done** (`fenics.py`) and JAX hook **done**
   (`jax.py`, Taylor-mode `jet`; see above). No framework hook is outstanding; further backends
   (PyTorch, Enzyme, ...) would follow the same `ImplicitProblem` recipe.
2. **Bring reference/example content into the library — done for the verification machinery**
   (`validation.py`) and the examples (labelled-section restructure + `toy_polynomial.py`). The
   problem-specific PDE setup is deliberately *not* promoted (folder dependency rule + the PDE internals
   are the lesson, so they stay visible in the example). Possible follow-ups if wanted: reusable FEniCS
   helpers (homogenized-BC setup, observation operators) — but only if they stay generic.
3. **More docs.** Done: `docs/overview.md` (+ examples section), `examples/*`, `docs/fenics_hook.md`,
   `docs/jax_hook.md`, `docs/composition.md`. Still wanted: a Sphinx build.
4. **Probe-to-tensor bridge (optional).** Package the forward/reverse probes into whatever a
   downstream consumer (e.g. T3Toolbox fitting) expects — kept out of this repo unless wanted.

## Interface notes locked in (for the hook authors)

- Driver requests sums, never singletons; the whole (possibly mixed-R/Q) sum goes to the problem so
  it does all assembly + cross-term addition (FEniCS: one combined form).
- Driver is vector-type-agnostic: zero arithmetic on physics vectors; integer coefficients applied
  inside `assemble_partial_sum`; `A` assembled/factorized once and reused for all solves.
- Forward probe -> output vector; reverse probe -> parameter covector (open slot left free).
- A `PartialTerm` with `open_slot` set and `pairing` (OMEGA or a vhat-vector) means: leave that slot
  as the test function, pair the output with omega (Q-terms) or contract with the adjoint vector
  (R-terms). `reference_problems.ImplicitPolynomialProblem.assemble_partial_sum` is the reference.

## Reminders for later

- **Dirichlet/essential BCs** (lands with the autodiff hooks): state eq -> real BCs; adjoint + all
  incremental eqs -> homogenized (zeroed-BC-dof). See maintainer memory `dirichlet-bc-handling`.
- Autodiff hooks (FEniCS `dl.derivative()`, JAX, and later others) should be **optional extras** — a
  user installs only the frameworks they actually use.
- Git: branch `main`, tracking `origin/main`; all slices committed and pushed (Algorithm 1, the toy
  reference problem, Algorithm 2, `docs/overview.md`, the FEniCS hook + example, the omega-as-argument
  refactor, and linear composition). Commits are authored by Nick with a
  `Co-Authored-By: Claude ...` trailer — Blake is a package author (LICENSE/pyproject/headers) but
  not on commit trailers while he is not at the keyboard.
