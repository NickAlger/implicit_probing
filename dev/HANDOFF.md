# implicit_probing — handoff / current state

## Where we are

**Slice 1 complete: the symbolic differentiation engine (Algorithm 1, t4s.pdf Section 4).** Pure
Python, no numpy.

- `implicit_probing/backend/multiset.py` — immutable, hashable `Multiset` (elements may themselves be
  `Multiset`s) + `subset_lattice` (every sub-multiset of `alpha`, nondecreasing cardinality).
- `implicit_probing/backend/symbolic.py` — `Pairing` (`ID` / `OMEGA` / `adjoint(delta)`), `Term`
  `(rho, tau, mu, Gamma)`, the three seeds (`seed_forward_q`, `seed_residual_r`, `seed_reverse`),
  `differentiate_term` (eqs 19-20), and `differentiate_over_lattice` (Algorithm 1). Plus
  `format_term` / `format_expansion` for human-readable debugging output.
- Tests: `tests/backend/test_multiset.py`, `tests/backend/test_symbolic.py` — validated against the
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

- `implicit_probing/backend/symbolic.py` (added) — `extract_state_rhs` / `extract_adjoint_rhs`:
  isolate the operator term (`A uhat_beta` / `A* vhat_beta`) and return the rest (eqs 10-11, 17-18).
- `implicit_probing/backend/driver.py` — the **vector-type-agnostic** driver:
  - `PartialTerm` — the lowered request DTO: `(coefficient, function R/Q, theta_dirs, u_vecs,
    open_slot in {None,'theta','u'}, pairing in {None, OMEGA, vhat-vector})`.
  - `OMEGA` — sentinel meaning "pair the output with the problem's omega functional".
  - `ImplicitProblem` — the 3-method `Protocol`: `solve_operator`, `solve_operator_adjoint`,
    `assemble_partial_sum`. The driver does NO arithmetic on physics vectors; it only resolves
    symbolic labels to vectors, hands whole sums to the problem, and routes the opaque results.
  - `probe(problem, alpha, direction_vectors)` -> `(forward, reverse)` dicts over every `beta <=
    alpha`. Forward = output-space vector; reverse = parameter-space covector (gradient-like).
- `implicit_probing/reference_problems.py` (extended) — `Polynomial.derivative_open_slot` (one open
  derivative slot) and the `ImplicitProblem` hook on the toy (`solve_operator` /
  `solve_operator_adjoint` / `assemble_partial_sum`), a reference implementation of the interface.
- Tests `tests/backend/test_driver.py` — extraction unit tests, probe structure, and the driver's
  probes vs the FD ground truth across symmetric / partially-symmetric / fully-asymmetric probes
  (orders 1-3, ~1e-9 agreement), reverse-probe identity `psi_beta . d = omega(D^{|beta|+1} q)`, and
  the order-4 vanishing-top-partial edge case.

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

- `probe(problem, alpha, direction_vectors, omega=None)`; `omega` is threaded to
  `assemble_partial_sum(self, terms, omega)`; problems no longer store it. `omega=None` computes
  forward probes only (skips the adjoint solves + reverse pass). The `OMEGA` sentinel stays in
  `PartialTerm.pairing` so a problem can tell an output-functional pairing from an incremental adjoint
  -- which is exactly what composition relies on.
- `implicit_probing/backend/composition.py` -- `ComposedProblem(inner, input_map=C, output_map=W)`
  probes the composed map `f = W o q o C` for linear input/output maps: pre-maps directions by `C`,
  post-maps a forward probe by `W` and a reverse probe by `C^T`, pulls `omega` back by `W^T`. The
  inner problem and the driver are unchanged; it is itself an `ImplicitProblem`, so compositions nest.
  Plus a `LinearOperator` protocol and a `MatrixOperator` (numpy) adapter.
- Tests: `tests/backend/test_composition.py` (toy: numpy `C`/`W` vs FD, reverse adjointness on the
  composed map, identity-maps recover the inner) and `tests/test_fenics_composition.py` (gated: theta
  from low-order polynomial features + observation restricted to boundary dofs). Docs:
  `docs/composition.md`.

## Design decisions locked

- Package / repo / import name: `implicit_probing` (GitHub renamed; local `origin` updated; local
  directory renamed to `implicit_probing` to match — repo, import package, and folder now all agree).
- Pure functional backend now; thin OO frontend later.
- Deps: `numpy` required (for slice 2+), `jax` an optional extra; **no T3Toolbox dependency**.
- `theta_0` will be a first-class input (multi-point gathering = trivial outer loop), for the
  maintainer's future global-polynomial work.

## Next steps (Algorithms 1 & 2 are done and validated — pick the next direction)

The core method is complete: symbolic engine + numeric driver, validated end-to-end against finite
differences. Candidate next slices (maintainer to choose):

1. **Autodiff-framework hooks.** FEniCS/DOLFINx hook **done** (`fenics.py`, see above). Remaining: a
   **JAX** hook (nested `jvp`/`jacfwd`), as an optional pip extra.
2. **Thin OO frontend.** Now that the functional backend works, add the light OO layer (e.g. an
   `ImplicitProblem` base/adapters + a top-level `probe(...)` entry point) per the original plan.
3. **Examples + more docs.** Done: `docs/overview.md`, `examples/fenics_poisson.py`,
   `docs/fenics_hook.md`. Still wanted: a toy-only example script and a Sphinx build.
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
