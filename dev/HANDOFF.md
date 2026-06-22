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

## Design decisions locked

- Package / repo / import name: `implicit_probing` (GitHub renamed; local `origin` updated; local
  directory still `ImplicitHigherDerivatives`).
- Pure functional backend now; thin OO frontend later.
- Deps: `numpy` required (for slice 2+), `jax` an optional extra; **no T3Toolbox dependency**.
- `theta_0` will be a first-class input (multi-point gathering = trivial outer loop), for the
  maintainer's future global-polynomial work.

## Next steps (Algorithms 1 & 2 are done and validated — pick the next direction)

The core method is complete: symbolic engine + numeric driver, validated end-to-end against finite
differences. Candidate next slices (maintainer to choose):

1. **Autodiff-framework hooks.** Implement `ImplicitProblem` for **FEniCS** (assemble the partial-sum
   requests as single combined forms via `dl.derivative()`; respect the Dirichlet-BC rule — see
   memory `dirichlet-bc-handling`) and/or **JAX** (nested `jvp`/`jacfwd`). Optional extras so users
   install only what they use.
2. **Thin OO frontend.** Now that the functional backend works, add the light OO layer (e.g. an
   `ImplicitProblem` base/adapters + a top-level `probe(...)` entry point) per the original plan.
3. **Examples + docs.** A worked example script (toy problem -> probes), and fold the design
   rationale into `docs/` (the math map, the assembly/interface contract).
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
- Git: local branch is still `master`; GitHub default is likely `main` — reconcile before/at first
  push. Nothing committed yet (awaiting maintainer go-ahead).
