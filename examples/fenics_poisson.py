# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Worked example: derivative probing of a nonlinear-Poisson parameter-to-observation map (DOLFINx).

State equation (nonlinear, Dirichlet on left/bottom/right, zero-Neumann on top):

    -div(exp(theta) grad u) + u^3 = f      in Omega = (0,1)^2,
                                u = g      on left/bottom/right  (smooth, inhomogeneous),
            exp(theta) du/dn = 0           on top.

with a Gaussian source f at the centre. The parameter ``theta`` is a log-diffusivity field; the
observation is the weak trace of ``u`` on the top edge, ``Q(u; v) = int_top u v ds`` (a vector over
the observation-space dofs, zero off the top). We probe the derivatives ``D^j q(theta0)`` of the
parameter-to-observation map ``q(theta) = Q(u(theta))``.

The file is in three labelled parts so you can see at a glance what is what: **PDE SETUP** (plain
FEniCS, nothing to do with this library), **PROBING** (the few lines that actually use
implicit_probing), and **RESULTS**. A short verification block at the end cross-checks the probes
against finite differences -- that is for confidence only; finite differences are slower and only
approximate, whereas the probes are exact.

Deliberately uses three different spaces -- theta in CG2, u in CG3, the observation test function in
CG1 -- so any accidental conflation of the parameter, state, and observation spaces would fail loudly.

Run (in a conda DOLFINx environment with implicit_probing installed):

    python examples/fenics_poisson.py
"""
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
import dolfinx.fem.petsc as petsc_fem

from implicit_probing import probe
from implicit_probing import validation                    # finite-difference cross-check (testing only)
from implicit_probing.fenics import FenicsImplicitProblem

comm = MPI.COMM_WORLD


# ====================================================================================================
# PDE SETUP (FEniCS)  --  this defines the map q(theta); it has nothing to do with implicit_probing
# ====================================================================================================

# --- mesh and the three (deliberately different) function spaces ---
msh = mesh.create_unit_square(comm, 24, 24)
x = ufl.SpatialCoordinate(msh)
V_theta = fem.functionspace(msh, ("Lagrange", 2))   # parameter   (log-diffusivity)
V_u     = fem.functionspace(msh, ("Lagrange", 3))   # state
V_q     = fem.functionspace(msh, ("Lagrange", 1))   # observation test function

# --- problem data: Gaussian source, smooth inhomogeneous Dirichlet g on three sides ---
f = 50.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.05 ** 2))

dirichlet_dofs = fem.locate_dofs_geometrical(
    V_u, lambda xx: np.isclose(xx[0], 0.0) | np.isclose(xx[0], 1.0) | np.isclose(xx[1], 0.0))
g = fem.Function(V_u)
g.interpolate(lambda xx: np.sin(np.pi * xx[0]) * (1.0 + xx[1]))   # smooth, genuinely inhomogeneous
bc_real = fem.dirichletbc(g, dirichlet_dofs)                       # for the state solve (true BCs)
bc_homog = fem.dirichletbc(fem.Function(V_u), dirichlet_dofs)      # for A / A* / incremental RHS (zeroed)

# top edge measure for the observation
fdim = msh.topology.dim - 1
top_facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.isclose(xx[1], 1.0))
ft = mesh.meshtags(msh, fdim, top_facets, np.full(len(top_facets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
TOP = 1
v_Q = ufl.TestFunction(V_q)


def residual(theta_func, u_func, v):
    return (ufl.exp(theta_func) * ufl.dot(ufl.grad(u_func), ufl.grad(v)) * ufl.dx
            + u_func ** 3 * v * ufl.dx - f * v * ufl.dx)


_solves = [0]
def solve_state(theta_func):
    """The user's own nonlinear solve (Newton via SNES), with the *real* Dirichlet BCs."""
    u_func = fem.Function(V_u)
    u_func.interpolate(g)                                  # start from a BC-satisfying field
    _solves[0] += 1
    prob = petsc_fem.NonlinearProblem(
        residual(theta_func, u_func, ufl.TestFunction(V_u)), u_func, bcs=[bc_real],
        petsc_options_prefix=f"state_{_solves[0]}_",
        petsc_options={"snes_rtol": 1e-13, "snes_atol": 1e-14, "ksp_type": "preonly",
                       "pc_type": "lu", "snes_error_if_not_converged": True})
    prob.solve()
    return u_func


def observe(theta_func):
    """The end-to-end map q(theta): solve, then assemble the weak top-trace observation vector."""
    u_func = solve_state(theta_func)
    qv = petsc_fem.assemble_vector(fem.form(u_func * v_Q * ds(TOP)))
    qv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return qv.array.copy()


# the frozen expansion point (theta0, u0): the user does this nonlinear solve once, themselves
theta0 = fem.Function(V_theta)
theta0.interpolate(lambda xx: 0.3 * np.sin(np.pi * xx[0]) * np.cos(np.pi * xx[1]))
u0 = solve_state(theta0)
print(f"state solved at theta0  (u0 range [{u0.x.array.min():+.3f}, {u0.x.array.max():+.3f}])")


# ====================================================================================================
# PROBING (implicit_probing)  --  the whole point of the example
# ====================================================================================================
# Freeze the problem at (theta0, u0) and hand it to the driver. Every probe below -- of every order,
# in every direction -- is computed from *linearized* solves that all share the ONE operator A = d_u R
# (factorized once, reused), with no further nonlinear solves and no finite-difference error: the probe
# values are exact to solver precision. This is what makes probing strictly cheaper than, and superior
# to, differencing the nonlinear map.
omega = fem.Function(V_q)                                 # the QoI: a smooth top-trace weight ...
omega.interpolate(lambda xx: np.sin(np.pi * xx[0]))       # ... so omega(q) = int_top sin(pi x) u ds
problem = FenicsImplicitProblem(
    residual(theta0, u0, ufl.TestFunction(V_u)),          # R_form, frozen at (theta0, u0)
    u0 * v_Q * ds(TOP),                                   # Q_form (observation)
    theta0, u0, bcs=[bc_homog])

# smooth probing directions (NOT random dof vectors -- keeps the finite-difference check below clean)
def field(fn):
    d = fem.Function(V_theta); d.interpolate(fn); return d

d1 = field(lambda xx: np.sin(np.pi * xx[0]) * np.sin(np.pi * xx[1]))
d2 = field(lambda xx: np.cos(np.pi * xx[0]) * np.sin(2 * np.pi * xx[1]))
d3 = field(lambda xx: np.sin(2 * np.pi * xx[0]) * np.cos(np.pi * xx[1]))

# directions are (field, max_power) pairs; this asks for every probe up to d1^2 d2^1 (orders 1-3).
directions = [(d1, 2), (d2, 1)]
forward, reverse = probe(problem, directions, omega)
print(f"one probe call returned forward + reverse for all {len(forward)} sub-probes "
      "(every lower order came for free)")


# ====================================================================================================
# RESULTS  --  the actually-useful outputs
# ====================================================================================================
print("\nforward probes, keyed by power-tuple (i, j) = order along (d1, d2)  (norm of the obs vector):")
for mu in sorted(forward):
    if sum(mu) == 0:
        continue
    print(f"  {str(mu):<8} ||D^{sum(mu)} q|| = {np.linalg.norm(forward[mu].array):.4e}")

# reverse[(0,0)] is the gradient of the QoI omega(q) w.r.t. the whole theta field -- one adjoint solve.
print(f"\nQoI gradient field (reverse[(0,0)]) over the parameter space: norm "
      f"{np.linalg.norm(reverse[(0, 0)].array):.4e}")

# a fully-asymmetric order-3 probe runs just the same (distinct directions => a bigger box):
fwd3, _ = probe(problem, [(d1, 1), (d2, 1), (d3, 1)])
print(f"fully-asymmetric D^3 q (d1,d2,d3): ||probe|| = {np.linalg.norm(fwd3[(1, 1, 1)].array):.4e}")


# ====================================================================================================
# verification (testing only -- the probes above are exact; finite differences are slow + approximate,
# and are used here purely to demonstrate that the probes are right -- see tests/ for the full sweep)
# ====================================================================================================
def perturb(point, scale, direction):                    # the one FEniCS-specific hook the helper needs
    moved = fem.Function(V_theta)
    moved.x.array[:] = point.x.array + scale * direction.x.array
    moved.x.scatter_forward()
    return moved

print("\n(cross-check vs finite differences -- confidence only, not how you'd compute these:)")
print(f"  {'(i,j)':<8}{'order':<7}{'rel err vs FD'}")
for mu in sorted(forward):
    if sum(mu) == 0:
        continue
    spec = [(directions[k][0], mu[k]) for k in range(len(mu)) if mu[k] > 0]
    fd = validation.forward_probe_by_finite_difference(observe, theta0, spec, perturb=perturb, h=2e-3)
    rel = np.linalg.norm(forward[mu].array - fd) / max(np.linalg.norm(fd), 1e-30)
    print(f"  {str(mu):<8}{sum(mu):<7}{rel:.2e}")

adj = validation.reverse_forward_adjointness(
    forward, reverse, directions, omega,
    pair_input=lambda rev, d: rev.array @ d.x.array,      # reverse covector (PETSc Vec) . direction (Function)
    pair_output=lambda om, fwd: om.x.array @ fwd.array)   # omega (Function) . forward output (PETSc Vec)
print(f"  reverse/forward adjointness (exact identity) max rel err: {adj:.2e}")


# ====================================================================================================
# optional visualization of the state and the QoI gradient (best-effort; needs pyvista + a renderer)
# ====================================================================================================
try:
    from pathlib import Path
    import pyvista
    from dolfinx import plot
    out_dir = Path("out_fenics_poisson"); out_dir.mkdir(exist_ok=True)
    grad = fem.Function(V_theta)                 # reverse[(0,0)] = gradient of omega(q) w.r.t. theta
    grad.x.array[:] = reverse[(0, 0)].array
    for field, space, name in [(u0, V_u, "state_u0"), (grad, V_theta, "qoi_gradient")]:
        cells, types, pts = plot.vtk_mesh(space)
        grid = pyvista.UnstructuredGrid(cells, types, pts)
        grid.point_data[name] = field.x.array.real
        grid.set_active_scalars(name)
        pl = pyvista.Plotter(off_screen=True)
        pl.add_mesh(grid.warp_by_scalar(), show_edges=False)
        pl.screenshot(str(out_dir / f"{name}.png"))
    print(f"\nwrote {out_dir}/state_u0.png and {out_dir}/qoi_gradient.png")
except Exception as exc:                          # pragma: no cover - visualization is optional
    print(f"\n(visualization skipped: {type(exc).__name__})")
