# implicit_probing — handoff / current state

## Where we are

**Slice 1 complete: the symbolic differentiation engine (Algorithm 1, t4s.pdf Section 4).** Pure
Python, no numpy. Full test suite green (27 tests + 14 subtests, ~0.05s).

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

Run: `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q`.

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
2. **Problem interface.** User-supplied callables: `solve_state() -> u`; a linearized operator with
   `solve(b)` / `solve_adjoint(c)` (`A`, `A*`); directional partials `partial_R(theta_dirs, u_vecs)`,
   `partial_Q(...)`; and the output functional `omega`.
3. **Numeric driver (Algorithm 2).** Walk the lattice, solve the incremental state/adjoint systems,
   assemble forward probes `y_beta` and reverse probes `psi_beta`.
4. **Validation.** A small concrete implicit problem; probes vs finite differences / AD.

## Reminders for later

- **Dirichlet/essential BCs** (lands with the autodiff hooks): state eq -> real BCs; adjoint + all
  incremental eqs -> homogenized (zeroed-BC-dof). See maintainer memory `dirichlet-bc-handling`.
- Autodiff hooks (FEniCS `dl.derivative()`, JAX, and later others) should be **optional extras** — a
  user installs only the frameworks they actually use.
- Git: local branch is still `master`; GitHub default is likely `main` — reconcile before/at first
  push. Nothing committed yet (awaiting maintainer go-ahead).
