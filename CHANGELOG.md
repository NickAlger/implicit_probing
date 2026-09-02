# Changelog

All notable changes to `implicit_probing` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are date-based
(`YYYY.MINOR.PATCH`).

## Unreleased

### Added
- **Batched probes** (JAX hook, numpy reference, `MatrixOperator`): a direction of shape `(B, p)` or
  an `omega` of shape `(B, n_q)` is a *batch* of B independent probes at the frozen expansion point,
  handled in ONE `probe` call — one lattice walk, one multi-right-hand-side solve on the shared LU
  per node, every jet kernel `vmap`-ed over the batch. Single inputs in the same call are shared by
  every member (so one call batches over directions, over output functionals, or both). A result is
  batched iff its request used a batched input, per key: `forward[mu]` iff a batched direction has
  `mu_k >= 1`, `reverse[mu]` iff `omega` is batched or such a direction exists — so `forward[(0,..)]`
  and, under direction-only batching, the gradient `reverse[(0,..)]` stay single. 2-D is always a
  batch (even B = 1); mismatched batch sizes and arrays of more than two dimensions raise. The
  driver is untouched; the unbatched path runs the exact kernels it always did; batched values agree
  with a loop of single probes to round-off. Measured on a deep-equilibrium model at J = 4:
  2.4× (D = 8) to 22× (D = 1024) faster per probe than the loop, and 1–1.8× faster than a hand-rolled
  `jax.vmap` over `probe`. New module `implicit_probing.batching` (the batch-size rule shared by the
  hooks). Design, review record and the FEniCSx plan: `dev/batched_probes_design.md`.
- **`JaxImplicitProblem.refreeze(theta0, u0)`**: move the frozen expansion point in place, keeping
  every compiled jet kernel. The kernels are jitted with the R-/Q-view closures static and the
  point traced, so multi-point probing (many expansion points) needs one reused instance; constructing a
  fresh instance per point recompiles every kernel at every point and exhausts XLA executable
  memory after a few tens of points (found while probing a trained deep-equilibrium model at 128
  sample points: ~3 s/point and an LLVM allocation failure at ~55 points, vs 0.08 s/point with
  `refreeze`). Custom solvers are point-specific, so a problem built with them must be handed
  fresh ones. The class docstring now states the reuse rule.

### Changed
- Nomenclature: *batch* now means B probes at ONE expansion point (the feature above); probing many
  expansion points is *multi-point* probing (the `refreeze` loop). The `refreeze` docstring and the
  entry above were reworded accordingly. The `refreeze` docstring also warns against wrapping
  `probe` in an outer `jax.jit` that closes over the problem (the frozen point is baked in and goes
  stale after `refreeze`).

## 2026.0.0 — 2026-07-10

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
