# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
#
# Gated test: composing the FEniCS nonlinear-Poisson map with a linear INPUT map (theta from a few
# low-order polynomial features) and a linear OUTPUT map (the observation restricted to the boundary
# dofs). Runs only where dolfinx is importable.
import unittest

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
import ufl
from dolfinx import mesh, fem
import dolfinx.fem.petsc as petsc_fem
from petsc4py import PETSc

from implicit_probing.multiset import Multiset, subset_lattice
from implicit_probing.driver import probe
from implicit_probing.composition import ComposedProblem
from implicit_probing.fenics import FenicsImplicitProblem
from implicit_probing import validation


class FeatureParameterization:
    """Input map C: a feature vector in R^m -> a theta field, theta = sum_i x_i * (feature_i in V_theta)."""
    def __init__(self, P, V_theta):
        self.P = P                                  # (n_theta, m): columns are features in V_theta
        self.V = V_theta

    def apply(self, x):                             # R^m -> V_theta Function (a probing direction)
        d = fem.Function(self.V)
        d.x.array[:] = self.P @ np.asarray(x)
        d.x.scatter_forward()
        return d

    def apply_transpose(self, theta_covec):         # V_theta-dual PETSc Vec -> R^m covector
        return self.P.T @ theta_covec.array


class TopBoundarySelection:
    """Output map W: the observation vector (V_q over the whole domain) -> just its top-boundary dofs."""
    def __init__(self, dofs, V_q):
        self.dofs = dofs
        self.V = V_q

    def apply(self, obs):                           # V_q-dual PETSc Vec -> R^(#boundary dofs)
        return obs.array[self.dofs].copy()

    def apply_transpose(self, z):                   # R^(#boundary dofs) -> V_q Function (the inner omega)
        w = fem.Function(self.V)
        w.x.array[:] = 0.0
        w.x.array[self.dofs] = np.asarray(z)
        w.x.scatter_forward()
        return w


def _build():
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, 16, 16)
    x = ufl.SpatialCoordinate(msh)
    V_theta = fem.functionspace(msh, ("Lagrange", 2))
    V_u = fem.functionspace(msh, ("Lagrange", 3))
    V_q = fem.functionspace(msh, ("Lagrange", 1))

    f = 50.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.05 ** 2))
    dofs_D = fem.locate_dofs_geometrical(
        V_u, lambda xx: np.isclose(xx[0], 0.0) | np.isclose(xx[0], 1.0) | np.isclose(xx[1], 0.0))
    g = fem.Function(V_u); g.interpolate(lambda xx: np.sin(np.pi * xx[0]) * (1.0 + xx[1]))
    bc_real = fem.dirichletbc(g, dofs_D)
    bc_homog = fem.dirichletbc(fem.Function(V_u), dofs_D)
    fdim = msh.topology.dim - 1
    top = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.isclose(xx[1], 1.0))
    ft = mesh.meshtags(msh, fdim, top, np.full(len(top), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    v_Q = ufl.TestFunction(V_q)

    counter = [0]
    def solve_state(theta):
        u = fem.Function(V_u); u.interpolate(g)
        vR = ufl.TestFunction(V_u)
        R = (ufl.exp(theta) * ufl.dot(ufl.grad(u), ufl.grad(vR)) * ufl.dx
             + u ** 3 * vR * ufl.dx - f * vR * ufl.dx)
        counter[0] += 1
        prob = petsc_fem.NonlinearProblem(
            R, u, bcs=[bc_real], petsc_options_prefix=f"comp_{counter[0]}_",
            petsc_options={"snes_rtol": 1e-13, "snes_atol": 1e-14, "ksp_type": "preonly",
                           "pc_type": "lu", "snes_error_if_not_converged": True})
        prob.solve()
        return u

    def observe(theta):                              # full V_q observation vector
        u = solve_state(theta)
        qv = petsc_fem.assemble_vector(fem.form(u * v_Q * ds(1)))
        qv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return qv.array.copy()

    theta0 = fem.Function(V_theta)
    theta0.interpolate(lambda xx: 0.3 * np.sin(np.pi * xx[0]) * np.cos(np.pi * xx[1]))
    u0 = solve_state(theta0)
    vR = ufl.TestFunction(V_u)
    R_form = (ufl.exp(theta0) * ufl.dot(ufl.grad(u0), ufl.grad(vR)) * ufl.dx
              + u0 ** 3 * vR * ufl.dx - f * vR * ufl.dx)
    inner = FenicsImplicitProblem(R_form, u0 * v_Q * ds(1), theta0, u0, bcs=[bc_homog])

    # input map C: a few low-order polynomial features interpolated into V_theta
    features = [lambda xx: np.ones_like(xx[0]), lambda xx: xx[0], lambda xx: xx[1],
                lambda xx: xx[0] ** 2, lambda xx: xx[0] * xx[1], lambda xx: xx[1] ** 2]
    P = np.zeros((theta0.x.array.size, len(features)))
    for i, feat in enumerate(features):
        fi = fem.Function(V_theta); fi.interpolate(feat); P[:, i] = fi.x.array
    C = FeatureParameterization(P, V_theta)

    # output map W: the top-boundary dofs of the observation space
    top_dofs = fem.locate_dofs_geometrical(V_q, lambda xx: np.isclose(xx[1], 1.0))
    W = TopBoundarySelection(top_dofs, V_q)

    return dict(inner=inner, observe=observe, theta0=theta0, C=C, W=W,
                m_features=len(features), m_obs=len(top_dofs), top_dofs=top_dofs)


def _perturb(point, scale, direction):
    """FEniCS hook for validation.forward_probe_by_finite_difference: a fresh point + scale*direction."""
    moved = fem.Function(point.function_space)
    moved.x.array[:] = point.x.array + scale * direction.x.array
    moved.x.scatter_forward()
    return moved


class TestComposedFenics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = _build()
        cls.composed = ComposedProblem(cls.ctx["inner"], cls.ctx["C"], cls.ctx["W"])
        rng = np.random.default_rng(0)
        m = cls.ctx["m_features"]
        cls.x_dirs = {1: rng.standard_normal(m), 2: rng.standard_normal(m)}
        cls.omega_z = rng.standard_normal(cls.ctx["m_obs"])   # functional on the reduced (boundary) output

    def test_forward_probes_match_finite_difference(self):
        # D^j f(x0)(xhat...) = W( D^j q(theta0)(C xhat...) ); the FD perturbs theta along C-mapped dirs.
        C, ctx = self.ctx["C"], self.ctx
        for alpha in [Multiset([1]), Multiset([1, 1]), Multiset([1, 2])]:
            forward, _ = probe(self.composed, alpha, self.x_dirs)
            for beta in subset_lattice(alpha):
                if len(beta) == 0:
                    continue
                with self.subTest(alpha=alpha, beta=beta):
                    spec = [(C.apply(self.x_dirs[k]), count) for k, count in beta.items()]
                    obs_fd = validation.forward_probe_by_finite_difference(
                        ctx["observe"], ctx["theta0"], spec, perturb=_perturb, h=1e-3)
                    expected = obs_fd[ctx["top_dofs"]]            # W applied to the FD observation
                    np.testing.assert_allclose(forward[beta], expected, atol=1e-5)

    def test_reverse_probes_adjointness(self):
        # On the composed map f: reverse[beta] . xhat_k == omega_z . forward[beta + {k}] (exact).
        alpha = Multiset([1, 1, 2])
        forward, reverse = probe(self.composed, alpha, self.x_dirs, self.omega_z)
        # composed forward/reverse are plain numpy (W-codomain and C-domain), so default numpy pairing works
        err = validation.reverse_forward_adjointness(forward, reverse, alpha, self.x_dirs, self.omega_z)
        self.assertLess(err, 1e-7, f"max adjointness rel err {err:.2e}")

    def test_output_shapes(self):
        forward, reverse = probe(self.composed, Multiset([1, 2]), self.x_dirs, self.omega_z)
        for beta in subset_lattice(Multiset([1, 2])):
            self.assertEqual(forward[beta].shape, (self.ctx["m_obs"],))      # boundary observation
            self.assertEqual(reverse[beta].shape, (self.ctx["m_features"],)) # feature-space covector


if __name__ == "__main__":
    unittest.main()
