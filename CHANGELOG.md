# Changelog

All notable changes to `implicit_probing` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are date-based
(`YYYY.MINOR.PATCH`).

## 2026.0.0 — unreleased

First public release. Implements Algorithms 1 & 2 of Section 4 of the T4S paper
(Alger, Christierson, Chen & Ghattas, 2026; arXiv:2603.21141).

### Added
- **Symbolic differentiation engine** (Algorithm 1): pure-Python multiset/term algebra that expands a
  probe into directional partial derivatives of `R` and `Q` over the multiset-subset lattice.
- **Numeric driver** (Algorithm 2): the vector-type-agnostic `probe(problem, directions, omega=None)`,
  returning forward (output-space) and reverse (parameter-space) probes keyed by power-tuples.
- **`ImplicitProblem` protocol** (`solve_operator`, `solve_operator_adjoint`, `assemble_partial_sum`)
  with a numpy reference implementation and a toy polynomial map with exact derivatives.
- **FEniCS/DOLFINx hook** (`implicit_probing.fenics`): one UFL-form recipe per `PartialTerm`, frozen at
  a user-supplied expansion point with homogenized BCs.
- **JAX hook** (`implicit_probing.jax`): Taylor-mode (`jax.experimental.jet`) partials with
  structure-keyed `jit`. Optional `[jax]` extra.
- **Linear input/output composition** (`ComposedProblem`) for `W ∘ q ∘ C`.
- **Validation helpers** (`implicit_probing.validation`): finite-difference ground truth and the
  exact reverse/forward adjointness identity.
- Runnable examples (`toy_polynomial`, two FEniCS scripts, a JAX deep-equilibrium model) and a Sphinx
  documentation site (overview + per-hook guides + autosummary API reference).
