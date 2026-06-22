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
from implicit_probing.fenics import FenicsImplicitProblem
from implicit_probing.backend.driver import probe

problem = FenicsImplicitProblem(R_form, Q_form, theta, u, omega, bcs=[bc_homog])
forward, reverse = probe(problem, alpha, direction_vectors)
```

- `R_form`, `Q_form` — the residual and the observation, each a UFL 1-form linear in a test function
  (the "output mode"). They may live in **different** spaces — e.g. theta in CG2, `u` in CG3, the
  observation test function in CG1. Using distinct spaces is encouraged: it makes any accidental
  conflation of the parameter, state, and observation spaces fail loudly.
- `theta`, `u` — the frozen point (`u` already solves `R(theta, u) = 0`).
- `omega` — the output functional (a `Function` in the observation space), paired with the output in
  reverse probes.
- `bcs` — the **homogenized** (zero-valued) Dirichlet BCs of the state space.
- `direction_vectors` — a `{label: Function}` map giving the smooth parameter-space directions to
  probe in (use smooth fields, not random dof vectors, so finite-difference checks stay clean).

## The one recipe behind `assemble_partial_sum`

Every `PartialTerm` the driver requests becomes a UFL form by three moves:

1. **Pairing** — what to do with the output test function `v`: keep it (forward probes / residual
   RHS); `ufl.replace(form, {v: omega})` (ω-paired); or `ufl.replace(form, {v: v_hat})` (paired with
   an incremental adjoint).
2. **Filled directions** — one `ufl.derivative(form, theta, d)` per probing direction, one
   `ufl.derivative(form, u, u_hat)` per incremental state.
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

## Validating probes

Two complementary checks (both shown in the example):

- **Forward probes vs finite differences** — re-solve the PDE at `theta0 + sum_k s_k d_k` and take a
  tensor product of central differences. This is an independent ground truth.
- **Reverse probes vs ω-paired forward probes** — the exact discrete adjointness identity
  `psi_beta . d_k == omega . forward[beta + {k}]`, which needs no extra solves and holds to solver
  precision. Since the forward probes are anchored to finite differences, this verifies the reverse
  probes too.
