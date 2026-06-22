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

Full suite green: **43 tests + 20 subtests, ~0.24s**. Run:
`PYTHONPATH=$PWD <env-python> -m pytest tests/ -q`.

## Design decisions locked

- Package / repo / import name: `implicit_probing` (GitHub renamed; local `origin` updated; local
  directory still `ImplicitHigherDerivatives`).
- Pure functional backend now; thin OO frontend later.
- Deps: `numpy` required (for slice 2+), `jax` an optional extra; **no T3Toolbox dependency**.
- `theta_0` will be a first-class input (multi-point gathering = trivial outer loop), for the
  maintainer's future global-polynomial work.

## Next slice (Algorithm 2 — the numeric driver)

1. **RHS extraction (symbolic).** From the `R` expansion `D_beta`, isolate the operator term
   `(ID, 'R', empty, {beta})` (`= A uhat_beta`) and read off `b_beta = -(everything else)`; likewise
   `c_beta` from the `R^adj` expansion. Pure-symbolic; sits next to `symbolic.py`.
2. **Problem interface.** Formalize the callable set already prototyped by
   `ImplicitPolynomialProblem`: `solve_state`, a linearized operator with `solve_A` / `solve_A_adjoint`
   (`A`, `A*`), directional partials `partial_R(theta_dirs, u_vecs)` / `partial_Q(...)`, and `omega`.
   Decide protocol/ABC vs a duck-typed convention.
3. **Numeric driver (Algorithm 2).** Walk the lattice, solve the incremental state/adjoint systems,
   assemble forward probes `y_beta` and reverse probes `psi_beta`. Map the symbolic direction labels
   in `alpha` to actual probe vectors.
4. **Validation.** Already wired: compare driver probes to `forward_probe_by_finite_difference` on
   `make_toy_problem`, sweeping symmetric / partially-symmetric / fully-asymmetric `alpha` and orders
   1..4 (degree-3 toy => order-4 exercises the vanishing-top-order edge case). Reverse probes checked
   by `psi = omega(forward probe)`.

## Reminders for later

- **Dirichlet/essential BCs** (lands with the autodiff hooks): state eq -> real BCs; adjoint + all
  incremental eqs -> homogenized (zeroed-BC-dof). See maintainer memory `dirichlet-bc-handling`.
- Autodiff hooks (FEniCS `dl.derivative()`, JAX, and later others) should be **optional extras** — a
  user installs only the frameworks they actually use.
- Git: local branch is still `master`; GitHub default is likely `main` — reconcile before/at first
  push. Nothing committed yet (awaiting maintainer go-ahead).
