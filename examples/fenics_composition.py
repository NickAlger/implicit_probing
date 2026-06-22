# Authors: Nick Alger and Blake Christierson
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
outputs). ``ComposedProblem`` probes ``f`` directly -- because ``C`` and ``W`` are linear, the
composed probes are just linear images of ``q``'s probes, and the PDE state solves are untouched.

This script builds ``C`` and ``W``, probes ``f``, and validates: forward probes against finite
differences of the re-solved composed map, and reverse probes against the (exact) adjointness identity.

Run in a conda DOLFINx environment with implicit_probing installed:

    python examples/fenics_composition.py
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
from implicit_probing.backend.composition import ComposedProblem
from implicit_probing.fenics import FenicsImplicitProblem

comm = MPI.COMM_WORLD


# ----------------------------------------------------------------------------------------------------
# Two problem-specific linear operators (each just needs apply / apply_transpose)
# ----------------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------------
# The nonlinear-Poisson state problem (same as examples/fenics_poisson.py)
# ----------------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------------
# Build the linear maps and the composed problem
# ----------------------------------------------------------------------------------------------------

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

composed = ComposedProblem(inner, input_map=C, output_map=W)

n_features, n_obs = len(features), len(top_dofs)
print("Composed map  f = W o q o C  :  x (features) -> z (boundary observations)")
print(f"  input  : {n_features} polynomial features        (vs {theta0.x.array.size} theta dofs in the full field)")
print(f"  output : {n_obs} boundary observation dofs   (vs {u0.function_space.dofmap.index_map.size_global}+ in the full domain)")


# ----------------------------------------------------------------------------------------------------
# Probe the composed map, and validate
# ----------------------------------------------------------------------------------------------------

# Probe directions live in feature space (R^m); omega is on the reduced output. Here omega = ones, so
# omega(z) = sum of the boundary observations = the total top-trace integral of u (a partition of unity).
rng = np.random.default_rng(0)
x_directions = {1: rng.standard_normal(n_features), 2: rng.standard_normal(n_features)}
omega = np.ones(n_obs)

alpha = Multiset([1, 1, 2])
forward, reverse = probe(composed, alpha, x_directions, omega)


# finite-difference ground truth: W applied to a mixed central difference of the re-solved observation,
# perturbing theta along the C-mapped feature directions.
_STENCILS = {1: ((-1, 1), (-0.5, 0.5)), 2: ((-1, 0, 1), (1.0, -2.0, 1.0)),
             3: ((-2, -1, 1, 2), (-0.5, 1.0, -1.0, 0.5))}
def composed_forward_fd(direction_orders, h=2e-3):
    per = [(C.apply(x_directions[k]), _STENCILS[m][0], _STENCILS[m][1], m) for k, m in direction_orders]
    total = None
    for picks in itertools.product(*[range(len(off)) for _, off, _, _ in per]):
        tp = fem.Function(V_theta); tp.x.array[:] = theta0.x.array
        coeff = 1.0
        for (d, off, wts, m), i in zip(per, picks):
            tp.x.array[:] += off[i] * h * d.x.array
            coeff *= wts[i] / h ** m
        contrib = coeff * observe(tp)
        total = contrib if total is None else total + contrib
    return total[top_dofs]                            # W: restrict to the boundary dofs

print("\nForward probes vs finite differences (composed map):")
print(f"  {'beta':<12}{'order':<7}{'rel err vs FD'}")
for beta in subset_lattice(alpha):
    if len(beta) == 0:
        continue
    y = forward[beta]
    y_fd = composed_forward_fd([(k, c) for k, c in beta.items()])
    rel = np.linalg.norm(y - y_fd) / max(np.linalg.norm(y_fd), 1e-30)
    print(f"  {str(beta):<12}{len(beta):<7}{rel:.2e}")

print("\nReverse probes vs omega-paired forward probes (exact discrete adjointness):")
max_adj = 0.0
for beta in subset_lattice(alpha):
    for k, xvec in x_directions.items():
        child = beta.add(k)
        if not child.issubmultiset(alpha):
            continue
        lhs = float(reverse[beta] @ xvec)
        rhs = float(omega @ forward[child])
        max_adj = max(max_adj, abs(lhs - rhs) / max(abs(rhs), 1e-30))
print(f"  max relative error  psi_beta . xhat_k  ==  omega . forward[beta+k]:  {max_adj:.2e}")

# The gradient of the QoI w.r.t. the features (reverse[empty]) is one number per feature -- directly
# interpretable: how each polynomial mode of the log-diffusivity affects the total top-trace.
print("\nGradient of the QoI (total top-trace) w.r.t. each feature:")
for name, value in zip(features, reverse[Multiset([])]):
    print(f"  d/d[{name:<3}] = {value:+.4e}")
