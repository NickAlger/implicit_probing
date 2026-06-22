# Authors: Nick Alger and Blake Christierson
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

This demonstrates the whole pipeline: build the PDE, do the (user's own) nonlinear solve, hand the
frozen ``(theta, u)`` to ``FenicsImplicitProblem``, and call ``probe``. It then validates the probes:
forward probes against finite differences (re-solving the PDE), and reverse probes against
omega-paired forward probes (exact discrete adjointness).

Deliberately uses three different spaces -- theta in CG2, u in CG3, the observation test function in
CG1 -- so any accidental conflation of the parameter, state, and observation spaces would fail loudly.

Run (in a conda DOLFINx environment with implicit_probing installed):

    python examples/fenics_poisson.py
"""
import itertools

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
import dolfinx.fem.petsc as petsc_fem

from implicit_probing.backend.multiset import Multiset, subset_lattice
from implicit_probing.backend.driver import probe
from implicit_probing.fenics import FenicsImplicitProblem

comm = MPI.COMM_WORLD

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


# --- the frozen expansion point (theta0, u0) ---
theta0 = fem.Function(V_theta)
theta0.interpolate(lambda xx: 0.3 * np.sin(np.pi * xx[0]) * np.cos(np.pi * xx[1]))
u0 = solve_state(theta0)
print(f"state solved at theta0  (u0 range [{u0.x.array.min():+.3f}, {u0.x.array.max():+.3f}])")

# observation functional omega (a smooth top-trace weight) -> QoI omega(q) = int_top sin(pi x) u ds.
# It is a per-probe choice, passed to probe(...) below, not stored on the problem.
omega = fem.Function(V_q)
omega.interpolate(lambda xx: np.sin(np.pi * xx[0]))

problem = FenicsImplicitProblem(
    residual(theta0, u0, ufl.TestFunction(V_u)),       # R_form, frozen at (theta0, u0)
    u0 * v_Q * ds(TOP),                                # Q_form (observation)
    theta0, u0, bcs=[bc_homog])

# --- smooth probing directions (NOT random dof vectors -- keeps the FD ground truth clean) ---
directions = {
    1: ("sin(pi x) sin(pi y)", lambda xx: np.sin(np.pi * xx[0]) * np.sin(np.pi * xx[1])),
    2: ("cos(pi x) sin(2pi y)", lambda xx: np.cos(np.pi * xx[0]) * np.sin(2 * np.pi * xx[1])),
    3: ("sin(2pi x) cos(pi y)", lambda xx: np.sin(2 * np.pi * xx[0]) * np.cos(np.pi * xx[1])),
}
dir_funcs = {}
for k, (_, fn) in directions.items():
    d = fem.Function(V_theta); d.interpolate(fn); dir_funcs[k] = d


# --- finite-difference ground truth for a forward probe (tensor product of central differences) ---
_STENCILS = {1: ((-1, 1), (-0.5, 0.5)),
             2: ((-1, 0, 1), (1.0, -2.0, 1.0)),
             3: ((-2, -1, 1, 2), (-0.5, 1.0, -1.0, 0.5))}

def forward_probe_fd(direction_orders, h=2e-3):
    per = [(dir_funcs[k], _STENCILS[m][0], _STENCILS[m][1], m) for k, m in direction_orders]
    total = None
    for picks in itertools.product(*[range(len(off)) for _, off, _, _ in per]):
        tp = fem.Function(V_theta); tp.x.array[:] = theta0.x.array
        coeff = 1.0
        for (d, off, wts, m), i in zip(per, picks):
            tp.x.array[:] += off[i] * h * d.x.array
            coeff *= wts[i] / h ** m
        contrib = coeff * observe(tp)
        total = contrib if total is None else total + contrib
    return total


# --- probe, and validate every sub-probe ---
alpha = Multiset([1, 1, 2])     # exercises orders 1-3: symmetric {1,1}, asymmetric {1,2}, mixed {1,1,2}
forward, reverse = probe(problem, alpha, dir_funcs, omega)

print(f"\nprobe(alpha={{1,1,2}}) returned forward + reverse for all {len(subset_lattice(alpha))} sub-probes")
print("\nForward probes vs finite differences:")
print(f"  {'beta':<12}{'order':<7}{'||D^|b| q||':<14}{'rel err vs FD'}")
for beta in subset_lattice(alpha):
    if len(beta) == 0:
        continue
    spec = [(k, c) for k, c in beta.items()]
    y = forward[beta].array
    y_fd = forward_probe_fd(spec)
    rel = np.linalg.norm(y - y_fd) / max(np.linalg.norm(y_fd), 1e-30)
    print(f"  {str(beta):<12}{len(beta):<7}{np.linalg.norm(y):<14.4e}{rel:.2e}")

print("\nReverse probes vs omega-paired forward probes (exact discrete adjointness):")
om = omega.x.array
max_adj = 0.0
for beta in subset_lattice(alpha):
    for k, dvec in dir_funcs.items():
        child = beta.add(k)
        if not child.issubmultiset(alpha):
            continue
        lhs = float(np.dot(reverse[beta].array, dvec.x.array))
        rhs = float(np.dot(om, forward[child].array))
        max_adj = max(max_adj, abs(lhs - rhs) / max(abs(rhs), 1e-30))
print(f"  max relative error  psi_beta . d_k  ==  omega . forward[beta+k]:  {max_adj:.2e}")

# fully-asymmetric order-3 probe also runs (no FD here, just shown to work):
fwd3, _ = probe(problem, Multiset([1, 2, 3]), dir_funcs)
print(f"\nfully-asymmetric D^3 q (d1,d2,d3): ||probe|| = {np.linalg.norm(fwd3[Multiset([1, 2, 3])].array):.4e}")
print("\nThe gradient of the QoI (reverse[empty], a field over the parameter space) has norm "
      f"{np.linalg.norm(reverse[Multiset([])].array):.4e}")

# --- optional visualization of the state and the QoI gradient (best-effort; needs pyvista + a renderer) ---
try:
    from pathlib import Path
    import pyvista
    from dolfinx import plot
    out_dir = Path("out_fenics_poisson"); out_dir.mkdir(exist_ok=True)
    grad = fem.Function(V_theta)                 # reverse[empty] = gradient of omega(q) w.r.t. theta
    grad.x.array[:] = reverse[Multiset([])].array
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
