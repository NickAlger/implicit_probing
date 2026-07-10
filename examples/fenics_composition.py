# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Worked example: composing the nonlinear-Poisson map with linear input and output maps.

The raw map ``q : theta -> obs`` from ``examples/fenics_poisson.py`` has a high-dimensional input (the
log-diffusivity field ``theta``, one value per CG2 dof) and a high-dimensional output (the weak
top-trace observation, a vector over *every* CG1 dof, nonzero only on the top edge). Often you want a
*reduced* map:

- parameterize ``theta`` by a few **features** ``x`` via a linear map ``theta = C x`` -- here ``x`` are
  the coefficients of low-order polynomials of position; and
- keep only the **boundary** entries of the observation via a linear selection ``z = W obs``.

The composed map is ``f = W o q o C : x -> z`` (a handful of feature inputs to a handful of boundary
outputs). ``ComposedProblem`` probes ``f`` directly -- because ``C`` and ``W`` are linear, the composed
probes are just linear images of ``q``'s probes, and the PDE state solves are untouched.

The file is in labelled parts: the two **LINEAR MAPS** (the actual subject here), the **PDE SETUP**
(identical to fenics_poisson.py), the **COMPOSE + PROBE** step, **RESULTS**, and a finite-difference
verification block (testing only).

Run in a conda DOLFINx environment with implicit_probing installed:

    python examples/fenics_composition.py
"""
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
import dolfinx.fem.petsc as petsc_fem

from implicit_probing import probe, ComposedProblem
from implicit_probing import validation                    # finite-difference cross-check (testing only)
from implicit_probing.fenics import FenicsImplicitProblem

comm = MPI.COMM_WORLD


# ====================================================================================================
# LINEAR MAPS  --  the subject of this example: each is just an apply / apply_transpose pair
# ====================================================================================================

class FeatureParameterization:
    """Input map C: feature coefficients in R^m -> a theta field, theta = sum_i x_i (feature_i in V_theta)."""
    def __init__(self, P, V_theta):
        self.P = P                                   # (n_theta_dofs, m): features interpolated into V_theta
        self.V = V_theta

    def apply(self, x):                              # R^m -> a V_theta Function (a probing direction)
        d = fem.Function(self.V); d.x.array[:] = self.P @ np.asarray(x); d.x.scatter_forward()
        return d

    def apply_transpose(self, theta_covec):          # a V_theta-dual PETSc Vec -> R^m covector
        return self.P.T @ theta_covec.array


class BoundarySelection:
    """Output map W: the observation vector over all dofs -> just its boundary-dof entries."""
    def __init__(self, dofs, V_q):
        self.dofs = dofs
        self.V = V_q

    def apply(self, obs):                            # a V_q-dual PETSc Vec -> R^(#boundary dofs)
        return obs.array[self.dofs].copy()

    def apply_transpose(self, z):                    # R^(#boundary dofs) -> a V_q Function (the inner omega)
        w = fem.Function(self.V); w.x.array[:] = 0.0; w.x.array[self.dofs] = np.asarray(z); w.x.scatter_forward()
        return w


# ====================================================================================================
# PDE SETUP (FEniCS)  --  the same nonlinear Poisson state problem as examples/fenics_poisson.py
# ====================================================================================================

msh = mesh.create_unit_square(comm, 24, 24)
x = ufl.SpatialCoordinate(msh)
V_theta = fem.functionspace(msh, ("Lagrange", 2))
V_u     = fem.functionspace(msh, ("Lagrange", 3))
V_q     = fem.functionspace(msh, ("Lagrange", 1))

f = 50.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.05 ** 2))
dirichlet_dofs = fem.locate_dofs_geometrical(
    V_u, lambda xx: np.isclose(xx[0], 0.0) | np.isclose(xx[0], 1.0) | np.isclose(xx[1], 0.0))
g = fem.Function(V_u); g.interpolate(lambda xx: np.sin(np.pi * xx[0]) * (1.0 + xx[1]))
bc_real = fem.dirichletbc(g, dirichlet_dofs)
bc_homog = fem.dirichletbc(fem.Function(V_u), dirichlet_dofs)
fdim = msh.topology.dim - 1
top_facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.isclose(xx[1], 1.0))
ft = mesh.meshtags(msh, fdim, top_facets, np.full(len(top_facets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
TOP = 1
v_Q = ufl.TestFunction(V_q)

_solves = [0]
def solve_state(theta_func):
    u_func = fem.Function(V_u); u_func.interpolate(g)
    vR = ufl.TestFunction(V_u)
    R = (ufl.exp(theta_func) * ufl.dot(ufl.grad(u_func), ufl.grad(vR)) * ufl.dx
         + u_func ** 3 * vR * ufl.dx - f * vR * ufl.dx)
    _solves[0] += 1
    prob = petsc_fem.NonlinearProblem(
        R, u_func, bcs=[bc_real], petsc_options_prefix=f"state_{_solves[0]}_",
        petsc_options={"snes_rtol": 1e-13, "snes_atol": 1e-14, "ksp_type": "preonly",
                       "pc_type": "lu", "snes_error_if_not_converged": True})
    prob.solve()
    return u_func

def observe(theta_func):                             # the full V_q observation vector
    u_func = solve_state(theta_func)
    qv = petsc_fem.assemble_vector(fem.form(u_func * v_Q * ds(TOP)))
    qv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return qv.array.copy()

theta0 = fem.Function(V_theta)
theta0.interpolate(lambda xx: 0.3 * np.sin(np.pi * xx[0]) * np.cos(np.pi * xx[1]))
u0 = solve_state(theta0)
vR = ufl.TestFunction(V_u)
R_form = (ufl.exp(theta0) * ufl.dot(ufl.grad(u0), ufl.grad(vR)) * ufl.dx
          + u0 ** 3 * vR * ufl.dx - f * vR * ufl.dx)
inner = FenicsImplicitProblem(R_form, u0 * v_Q * ds(TOP), theta0, u0, bcs=[bc_homog])


# ====================================================================================================
# COMPOSE + PROBE (implicit_probing)
# ====================================================================================================

# Input map C: low-order polynomial features of position, interpolated into V_theta.
features = {
    "1": lambda xx: np.ones_like(xx[0]), "x": lambda xx: xx[0], "y": lambda xx: xx[1],
    "x^2": lambda xx: xx[0] ** 2, "xy": lambda xx: xx[0] * xx[1], "y^2": lambda xx: xx[1] ** 2,
}
P = np.zeros((theta0.x.array.size, len(features)))
for i, feat in enumerate(features.values()):
    fi = fem.Function(V_theta); fi.interpolate(feat); P[:, i] = fi.x.array
C = FeatureParameterization(P, V_theta)

# Output map W: the observation restricted to the top-boundary dofs.
top_dofs = fem.locate_dofs_geometrical(V_q, lambda xx: np.isclose(xx[1], 1.0))
W = BoundarySelection(top_dofs, V_q)

composed = ComposedProblem(inner, input_map=C, output_map=W)    # itself an ImplicitProblem; probes nest

n_features, n_obs = len(features), len(top_dofs)
print("Composed map  f = W o q o C  :  x (features) -> z (boundary observations)")
print(f"  input  : {n_features} polynomial features        (vs {theta0.x.array.size} theta dofs in the full field)")
print(f"  output : {n_obs} boundary observation dofs   (vs {u0.function_space.dofmap.index_map.size_global}+ in the full domain)")

# Probe directions live in feature space (R^m), given as (vector, max_power) pairs; omega is on the
# reduced output. Here omega = ones, so omega(z) = sum of the boundary observations = the total
# top-trace integral of u (a partition of unity).
rng = np.random.default_rng(0)
x1 = rng.standard_normal(n_features)
x2 = rng.standard_normal(n_features)
x_directions = [(x1, 2), (x2, 1)]
omega = np.ones(n_obs)

forward, reverse = probe(composed, x_directions, omega)
# The PDE state solves are untouched by C, W -- they only reparameterize the input and re-observe the
# output -- so probing the reduced map costs exactly what probing the raw map does.


# ====================================================================================================
# RESULTS  --  the per-feature QoI gradient (directly interpretable)
# ====================================================================================================
# reverse[(0,0)] is one number per feature: how each polynomial mode of the log-diffusivity affects the
# total top-trace. A single adjoint solve gives the whole feature-space gradient.
print("\nGradient of the QoI (total top-trace) w.r.t. each feature:")
for name, value in zip(features, reverse[(0, 0)]):
    print(f"  d/d[{name:<3}] = {value:+.4e}")


# ====================================================================================================
# verification (testing only -- the probes above are exact; finite differences are slow + approximate)
# ====================================================================================================
def perturb(point, scale, direction):                # FEniCS hook: theta perturbed along a C-mapped dir
    moved = fem.Function(V_theta)
    moved.x.array[:] = point.x.array + scale * direction.x.array
    moved.x.scatter_forward()
    return moved

print("\n(cross-check vs finite differences of the re-solved composed map -- confidence only:)")
print(f"  {'(i,j)':<8}{'order':<7}{'rel err vs FD'}")
for mu in sorted(forward):
    if sum(mu) == 0:
        continue
    spec = [(C.apply(x_directions[k][0]), mu[k]) for k in range(len(mu)) if mu[k] > 0]  # dirs pre-mapped by C
    fd_full = validation.forward_probe_by_finite_difference(observe, theta0, spec, perturb=perturb, h=2e-3)
    fd = fd_full[top_dofs]                            # W: restrict to the boundary dofs
    rel = np.linalg.norm(forward[mu] - fd) / max(np.linalg.norm(fd), 1e-30)
    print(f"  {str(mu):<8}{sum(mu):<7}{rel:.2e}")

# composed forward/reverse are plain numpy (W-codomain and C-domain), so the default numpy pairing works
adj = validation.reverse_forward_adjointness(forward, reverse, x_directions, omega)
print(f"  reverse/forward adjointness (exact identity) max rel err: {adj:.2e}")
