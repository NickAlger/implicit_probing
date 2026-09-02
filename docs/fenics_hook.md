# The FEniCS (DOLFINx) hook

`implicit_probing.fenics.FenicsImplicitProblem` implements the
[`ImplicitProblem`](overview.md#using-it-on-your-own-problem) interface for a map defined by a modern
FEniCS / DOLFINx PDE, so you can `probe` its derivatives. This page explains how it works and how to
use it. (See `examples/fenics_poisson.py` for a complete, validated worked example.)

## Installation

DOLFINx is distributed via conda, not pip, so the FEniCS hook is **not** a pip extra — it simply
requires a conda environment with DOLFINx, into which you install implicit_probing:

```bash
conda create -n fenicsx -c conda-forge fenics-dolfinx mpich pyvista
conda run -n fenicsx pip install -e .      # implicit_probing (numpy-only core) into the env
```

The numpy-only core of implicit_probing never imports DOLFINx; only `implicit_probing.fenics` does.

## What it does (and does not) do

The class is **frozen at an expansion point** `(theta, u)` that *you* supply already solved: you do the
nonlinear state solve `R(theta, u) = 0` yourself, however you like, with the **real** boundary
conditions. The class never runs a nonlinear solve. It only assembles the linearized operator
`A = d_u R` once and turns derivative-probe requests into UFL forms.

```python
from implicit_probing import probe
from implicit_probing.fenics import FenicsImplicitProblem

problem = FenicsImplicitProblem(R_form, Q_form, theta, u, bcs=[bc_homog])
forward, reverse = probe(problem, directions, omega)   # directions: ((d, max_power), ...) of Functions
```

- `R_form`, `Q_form` — the residual and the observation, each a UFL 1-form linear in a test function
  (the "output mode"). They may live in **different** spaces — e.g. theta in CG2, `u` in CG3, the
  observation test function in CG1. Using distinct spaces is encouraged: it makes any accidental
  conflation of the parameter, state, and observation spaces fail loudly.
- `theta`, `u` — the frozen point (`u` already solves `R(theta, u) = 0`).
- `bcs` — the **homogenized** (zero-valued) Dirichlet BCs of the state space.
- `direction_vectors` — a `{label: Function}` map giving the smooth parameter-space directions to
  probe in (use smooth fields, not random dof vectors, so finite-difference checks stay clean).
- `omega` — the output functional (a `Function` in the observation space), a **per-probe** argument
  rather than a property of the problem; the reverse probes differentiate `omega(q)`. Pass
  `omega=None` for forward probes only.

## The one recipe behind `assemble_partial_sum`

Every `PartialTerm` the driver requests becomes a UFL form by three moves:

1. **Pairing** — what to do with the output test function `v`: keep it (forward probes / residual
   RHS); `ufl.replace(form, {v: omega})` (ω-paired); or `ufl.replace(form, {v: v_hat})` (paired with
   an incremental adjoint).
2. **Filled directions** — `theta_dirs` and `u_vecs` arrive as `(vector, multiplicity)` pairs (the
   partial is symmetric in each block, so they are a multiset). `ufl.derivative` takes one direction at
   a time, so we nest `ufl.derivative(form, theta, d)` *multiplicity* times per distinct `d` (and
   likewise `ufl.derivative(form, u, u_hat)`) — this hook unrolls the repetition rather than exploiting
   it.
3. **Open slot** — for reverse objects, one more `ufl.derivative(form, u)` or
   `ufl.derivative(form, theta)` *with no explicit direction*, which introduces a fresh test function
   in that space (the free / "open" slot).

All terms in a single request share a test-function space, so they are summed into one combined form
and assembled **once** — assembling a single FEniCS form is far cheaper than assembling many and
adding them.

## Boundary conditions (the subtle part)

The state solve uses the **real** (possibly inhomogeneous) Dirichlet BCs — that is your job, outside
the class. Everything the class does uses **homogenized** BCs, because the BC data is constant in the
parameter, so its derivatives vanish on the constrained dofs:

- `A = d_u R` is assembled with the homogenized BCs (identity rows on constrained dofs);
- the incremental right-hand sides `b_beta`, `c_beta` have their constrained dofs zeroed;
- the probe *outputs* (forward and reverse probes) are returned untouched.

The class infers which assembled vectors are solve right-hand sides (and so need the homogenized BCs)
from the request itself, so the driver stays boundary-condition agnostic.

## Solvers

By default `A` is LU-factorized once and reused for every probe — forward solves use the
factorization, adjoint solves use its transpose solve. For large problems, pass `forward_solver`
and/or `adjoint_solver` (callables mapping a RHS `PETSc.Vec` to the solution `PETSc.Vec`) to plug in
your own Krylov solver and preconditioner.

## Choosing the solver: `ksp_factory`

By default the hook LU-factorizes `A = d_u R` once with PETSc's default factor package and reuses it
for every probe. To choose the solver yourself, pass `ksp_factory`, a callable `PETSc.Mat -> PETSc.KSP`
that the hook calls once on the assembled operator — MUMPS for a saddle-point operator that needs
pivoting, a KSP configured through the PETSc options database, an iterative solver with your
preconditioner:

```python
def mumps(A):
    ksp = PETSc.KSP().create(A.getComm()); ksp.setOperators(A)
    ksp.setType("preonly"); ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")
    return ksp

problem = FenicsImplicitProblem(R_form, Q_form, theta0, u0, bcs=bcs, ksp_factory=mumps)
```

`forward_solver` / `adjoint_solver` callables (`Vec -> Vec`) still work and take precedence, but the
KSP route is the one batched probes can exploit (next section). Native PETSc LU does not run under
MPI, so a parallel run needs a factory (MUMPS, SuperLU_dist, ...) in any case.

## Batched probes

A Python **list** of B Functions in place of a direction — or of `omega` — is a *batch*: B independent
probes at the frozen point, handled by ONE `probe` call.

```python
D = [d_1, d_2, ..., d_B]                              # B directions in V_theta
forward, reverse = probe(problem, [(D, 3)], omega)   # omega single: shared by every member
forward[(2,)]     # a list of B Vecs (one second directional derivative per member)
reverse[(2,)]     # a list of B Vecs
forward[(0,)]     # ONE Vec: q(theta0), shared -- not a list
```

What happens inside, per lattice node: the B right-hand sides are assembled into one dense PETSc
matrix with `A`'s row layout and solved together with `KSP.matSolve` / `matSolveTranspose` on the
hook's KSP — a direct solver traverses its factorization once for all B (MUMPS: ~11× over B single
solves at 40k dofs, B = 64), an iterative KSP loops internally and stays correct — and the combined
UFL form of each request is built **once**, with *slot* coefficient Functions standing in for the
batched lists, then assembled per member after copying that member's dofs into the slots. So the
form construction (nesting the derivatives, `fem.form`) and the linear solves are shared across the
batch; the integration over the mesh is still done B times. That is the part MPI mesh partitioning
parallelizes, and it composes with batching.

Rules:

- **Broadcasting.** A single Function in a batched call is shared by every member: batch over
  directions at a fixed `omega`, over `omega` at a fixed direction (B gradients from one adjoint
  `matSolve`: `probe(problem, [], OMEGAS)`), or over both; mixed requests `[(D, 2), (b, 1)]` are fine.
- **What comes back batched — per key.** `forward[mu]` is a list iff some batched direction `k` has
  `mu_k ≥ 1`; `reverse[mu]` is a list iff `omega` is batched or such a direction exists. So
  `forward[(0, …)]` is always a single Vec, and under direction-only batching so is the gradient
  `reverse[(0, …)]`. **Mind the type**: a single Vec where you expected a list still supports
  `vec[b]` — it returns entry `b`, silently. Check `isinstance(x, list)` where it matters.
- A batch must be a `list`; a tuple or set is refused. Batched lists of different lengths raise
  (inside the walk, after the first solves). Solve counts are one per node, not per member.
- **Custom solvers** keep their `Vec -> Vec` contract and are applied per member (no multi-RHS
  win); the `ksp_factory` route gets it.
- **Memory.** Every node keeps its B incrementals and adjoints alive for the walk, roughly
  `2 · (#nodes) · B · n_u · 8` bytes, plus the dense blocks; MUMPS additionally centralizes a dense
  right-hand side on rank 0. Chunk the batch yourself; the hook does not.
- A MUMPS multi-right-hand-side solve is a different MUMPS code path from a single solve (per-solve
  options such as iterative refinement may not apply); agreement with looped solves is at `1e-15`.
- **Composition.** `ComposedProblem` passes lists through; a FEniCS-side `LinearOperator` used with
  batched probes must map a `(B, m)` block to a list of B Functions and a list of B Vecs back to a
  `(B, m)` block (`tests/test_fenics_composition.py` shows the pattern).

Measured on T3Polynomial's mixed-form Darcy problem at nx = 60 (18k state dofs, MUMPS via
`ksp_factory`), J = 4, B = 64, laptop, one batched probe versus a loop of 64 single probes, agreeing to
`2e-13`:

| | wall | form construction etc. | mesh integration (assembly calls) | linear solves |
|---|---|---|---|---|
| loop of 64 single probes | 44.3 s (692 ms/probe) | 31.6 s | 8.6 s (1216 calls) | 4.1 s (576 solves) |
| one batched probe | **14.1 s (221 ms/probe)** | 5.3 s | 8.4 s (1153 calls) | 0.5 s (9 multi-RHS solves) |

The loop was dominated by per-probe form construction, which batching amortizes; the solves shrink
8×; the integration is B× in both, as expected. Larger B amortizes further.

## Validating probes

Two complementary checks (both shown in the example):

- **Forward probes vs finite differences** — re-solve the PDE at `theta0 + sum_k s_k d_k` and take a
  tensor product of central differences. This is an independent ground truth.
- **Reverse probes vs ω-paired forward probes** — the exact discrete adjointness identity
  `reverse[mu] . d_k == omega . forward[mu + e_k]`, which needs no extra solves and holds to solver
  precision. Since the forward probes are anchored to finite differences, this verifies the reverse
  probes too.
