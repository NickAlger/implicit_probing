# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
#
# Gated test for the DOLFINx hook: runs only where dolfinx is importable (the `fenicsx` conda env),
# and is skipped (not failed) elsewhere (e.g. the numpy-only `t3toolbox` env).
import unittest

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI
import ufl
from dolfinx import mesh, fem
import dolfinx.fem.petsc as petsc_fem
from petsc4py import PETSc

from implicit_probing.driver import probe
from implicit_probing.fenics import FenicsImplicitProblem
from implicit_probing import validation


def _build_problem():
    """The nonlinear Poisson example: -div(exp(theta) grad u) + u^3 = f, mixed CG2/CG3/CG1 spaces."""
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
    def solve_state(theta_func):
        uu = fem.Function(V_u); uu.interpolate(g)
        vR = ufl.TestFunction(V_u)
        R = (ufl.exp(theta_func) * ufl.dot(ufl.grad(uu), ufl.grad(vR)) * ufl.dx
             + uu ** 3 * vR * ufl.dx - f * vR * ufl.dx)
        counter[0] += 1
        prob = petsc_fem.NonlinearProblem(
            R, uu, bcs=[bc_real], petsc_options_prefix=f"st_{counter[0]}_",
            petsc_options={"snes_rtol": 1e-13, "snes_atol": 1e-14, "ksp_type": "preonly",
                           "pc_type": "lu", "snes_error_if_not_converged": True})
        prob.solve()
        return uu

    def q_of(theta_func):
        uu = solve_state(theta_func)
        qv = petsc_fem.assemble_vector(fem.form(uu * v_Q * ds(1)))
        qv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return qv.array.copy()

    theta0 = fem.Function(V_theta)
    theta0.interpolate(lambda xx: 0.3 * np.sin(np.pi * xx[0]) * np.cos(np.pi * xx[1]))
    u0 = solve_state(theta0)

    Q = u0 * v_Q * ds(1)
    omega = fem.Function(V_q); omega.interpolate(lambda xx: np.sin(np.pi * xx[0]))
    vR = ufl.TestFunction(V_u)
    R_form = (ufl.exp(theta0) * ufl.dot(ufl.grad(u0), ufl.grad(vR)) * ufl.dx
              + u0 ** 3 * vR * ufl.dx - f * vR * ufl.dx)
    problem = FenicsImplicitProblem(R_form, Q, theta0, u0, bcs=[bc_homog])

    return dict(problem=problem, q_of=q_of, theta0=theta0, V_theta=V_theta, omega=omega)


def _direction(V_theta, fn):
    d = fem.Function(V_theta); d.interpolate(fn)
    return d


def _perturb(point, scale, direction):
    """FEniCS hook for validation.forward_probe_by_finite_difference: a fresh point + scale*direction."""
    moved = fem.Function(point.function_space)
    moved.x.array[:] = point.x.array + scale * direction.x.array
    moved.x.scatter_forward()
    return moved


class TestFenicsProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = _build_problem()
        V = cls.ctx["V_theta"]
        cls.d1 = _direction(V, lambda xx: np.sin(np.pi * xx[0]) * np.sin(np.pi * xx[1]))
        cls.d2 = _direction(V, lambda xx: np.cos(np.pi * xx[0]) * np.sin(2 * np.pi * xx[1]))

    def test_forward_probes_match_finite_difference(self):
        prob, q_of = self.ctx["problem"], self.ctx["q_of"]
        theta0, d1, d2 = self.ctx["theta0"], self.d1, self.d2
        cases = [
            ("order1",            [(d1, 1)],            (1,),   1e-6),
            ("order2 symmetric",  [(d1, 2)],            (2,),   1e-5),
            ("order2 asymmetric", [(d1, 1), (d2, 1)],   (1, 1), 1e-5),
        ]
        for name, directions, mu, atol in cases:
            with self.subTest(symmetry=name):
                forward, _ = probe(prob, directions)
                y = forward[mu].array
                spec = [(directions[k][0], mu[k]) for k in range(len(mu)) if mu[k] > 0]
                y_fd = validation.forward_probe_by_finite_difference(
                    q_of, theta0, spec, perturb=_perturb, h=1e-3)
                rel = np.linalg.norm(y - y_fd) / max(np.linalg.norm(y_fd), 1e-30)
                self.assertLess(rel, atol, f"{name}: rel err {rel:.2e}")

    def test_reverse_probes_match_omega_paired_forward(self):
        # Discrete adjointness (exact -- no finite differences): pairing a reverse probe with a
        # direction equals omega paired with the forward probe one order higher in that axis,
        #     reverse[mu] . d_k  ==  omega . forward[mu + e_k].
        # The forward probes are anchored to finite differences above, so this verifies the reverse
        # probes against them to solver precision, and needs no extra PDE solves (just dot products).
        prob, omega = self.ctx["problem"], self.ctx["omega"]
        directions = [(self.d1, 2), (self.d2, 1)]
        forward, reverse = probe(prob, directions, omega)
        err = validation.reverse_forward_adjointness(
            forward, reverse, directions, omega,
            pair_input=lambda rev, d: rev.array @ d.x.array,      # reverse covector (PETSc Vec) . dir (Function)
            pair_output=lambda om, fwd: om.x.array @ fwd.array)   # omega (Function) . forward output (PETSc Vec)
        self.assertLess(err, 1e-8, f"max adjointness rel err {err:.2e}")


if __name__ == "__main__":
    unittest.main()
