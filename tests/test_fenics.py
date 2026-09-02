# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
#
# Gated test for the DOLFINx hook: runs only where dolfinx is importable, and is skipped (not
# failed) elsewhere (e.g. a numpy-only environment without dolfinx installed).
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
from implicit_probing.fenics import FenicsImplicitProblem, direct_lu
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


class _CountingProblem:
    """Counts the driver's solves at the ``solve_operator`` boundary (a batched solve counts once)."""
    def __init__(self, inner):
        self.inner = inner
        self.n_forward = 0
        self.n_adjoint = 0

    def solve_operator(self, b):
        self.n_forward += 1
        return self.inner.solve_operator(b)

    def solve_operator_adjoint(self, c):
        self.n_adjoint += 1
        return self.inner.solve_operator_adjoint(c)

    def assemble_partial_sum(self, terms, omega):
        return self.inner.assemble_partial_sum(terms, omega)


def _mumps_factory(A):
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    return ksp


class TestFenicsBatchedProbes(unittest.TestCase):
    """The batched contract for the DOLFINx hook (``dev/batched_probes_design.md``): a LIST of B
    Functions in place of a direction or of ``omega`` is a batch of B probes at the frozen point --
    one lattice walk, one multi-right-hand-side ``KSP.matSolve`` per node, the combined form built once
    per request with slot coefficients. Results are lists of Vecs exactly where they depend on a
    batched input (per key), single Vecs otherwise. Checked member-by-member against a loop of
    single probes; solve counts stay per node; the KSP-factory path (MUMPS) and the custom-solver
    loop fallback give the same answers."""
    B = 3

    @classmethod
    def setUpClass(cls):
        cls.ctx = _build_problem()
        cls.prob = cls.ctx["problem"]
        V, cls.omega = cls.ctx["V_theta"], cls.ctx["omega"]
        cls.D = [_direction(V, lambda xx, k=k: np.sin((k + 1) * np.pi * xx[0]) * np.cos((k + 2) * np.pi * xx[1]) + 0.3 * k)
                 for k in range(cls.B)]                       # a batch of B directions (a list)
        cls.b = _direction(V, lambda xx: np.cos(np.pi * xx[0]) * np.sin(2 * np.pi * xx[1]))   # single
        Vq = cls.omega.function_space
        cls.OMS = []
        for k in range(cls.B):                                # a batch of B output functionals
            f = fem.Function(Vq)
            f.interpolate(lambda xx, k=k: np.sin((k + 1) * np.pi * xx[0]) + 0.1 * k)
            cls.OMS.append(f)

    @staticmethod
    def _arr(v):
        return np.asarray(v.array)

    def _check(self, directions, omega, directions_of, omega_of, fwd_batched, rev_batched, problem=None):
        problem = self.prob if problem is None else problem
        forward, reverse = probe(problem, directions, omega)
        loops = [probe(problem, directions_of(j), omega_of(j)) for j in range(self.B)]
        self.assertEqual(set(forward), set(loops[0][0]))
        for mu in forward:
            for family, got, ref, batched in (
                    ("forward", forward[mu], [l[0][mu] for l in loops], fwd_batched(mu)),
                    ("reverse", reverse[mu], [l[1][mu] for l in loops], rev_batched(mu))):
                with self.subTest(family=family, mu=mu):
                    if batched:
                        self.assertIsInstance(got, list, f"{family}{mu} should be a batch")
                        self.assertEqual(len(got), self.B)
                        for j in range(self.B):
                            np.testing.assert_allclose(self._arr(got[j]), self._arr(ref[j]), rtol=1e-11, atol=1e-13)
                    else:                                     # a single Vec shared by every member
                        self.assertIsInstance(got, PETSc.Vec, f"{family}{mu} should be single")
                        for j in range(self.B):
                            np.testing.assert_allclose(self._arr(got), self._arr(ref[j]), rtol=1e-11, atol=1e-13)

    def test_batched_directions_shared_omega(self):
        # direction-only batching: q(theta0) and the gradient reverse[(0,)] stay single Vecs
        self._check([(self.D, 2)], self.omega, lambda j: [(self.D[j], 2)], lambda j: self.omega,
                    fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: mu[0] >= 1)

    def test_batched_omega_shared_direction(self):
        self._check([(self.b, 2)], self.OMS, lambda j: [(self.b, 2)], lambda j: self.OMS[j],
                    fwd_batched=lambda mu: False, rev_batched=lambda mu: True)

    def test_mixed_request_is_batched_per_key(self):
        self._check([(self.D, 2), (self.b, 1)], self.OMS,
                    lambda j: [(self.D[j], 2), (self.b, 1)], lambda j: self.OMS[j],
                    fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: True)

    def test_gradient_sketch_shape(self):
        forward, reverse = probe(self.prob, [], self.OMS)     # B gradients, one adjoint matSolve
        self.assertIsInstance(forward[()], PETSc.Vec)
        self.assertIsInstance(reverse[()], list)
        self.assertEqual(len(reverse[()]), self.B)

    def test_solve_counts_are_per_node(self):
        counting = _CountingProblem(self.prob)
        probe(counting, [(self.D, 2), (self.b, 1)], self.OMS)
        L = 3 * 2
        self.assertEqual(counting.n_forward, L - 1)
        self.assertEqual(counting.n_adjoint, L)

    def test_default_solver_is_direct_lu_mumps_where_available(self):
        # The default is a direct factorization: MUMPS if this PETSc build has it, else native LU.
        expected = "mumps" if PETSc.Sys.hasExternalPackage("mumps") else "petsc"
        self.assertEqual(direct_lu().package, expected)
        self.assertEqual(self.prob._ksp.getType(), "preonly")
        self.assertEqual(self.prob._ksp.getPC().getType(), "lu")
        self.assertEqual(self.prob._ksp.getPC().getFactorSolverType(), expected)

    def test_ksp_factory_choices_agree(self):
        # An injected factory carries the batched solves; a hand-written MUMPS factory, the library's
        # direct_lu('mumps'), and native LU via direct_lu('petsc') all give the same probes.
        p = self.prob
        factories = {"hand-written mumps": _mumps_factory, "direct_lu('mumps')": direct_lu("mumps")}
        if MPI.COMM_WORLD.size == 1:                       # PETSc's native LU is sequential only
            factories["direct_lu('petsc')"] = direct_lu("petsc")
        for name, factory in factories.items():
            with self.subTest(factory=name):
                pf = FenicsImplicitProblem(p.R_form, p.Q_form, p.theta, p.u, bcs=p.bcs, ksp_factory=factory)
                self._check([(self.D, 2)], self.OMS, lambda j: [(self.D[j], 2)], lambda j: self.OMS[j],
                            fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: True, problem=pf)

    def test_custom_solvers_take_the_per_member_loop(self):
        p = self.prob
        seen = []
        def fwd(bvec):
            self.assertIsInstance(bvec, PETSc.Vec); seen.append("f")
            x = bvec.duplicate(); p._ksp.solve(bvec, x); return x
        def adj(cvec):
            self.assertIsInstance(cvec, PETSc.Vec); seen.append("a")
            x = cvec.duplicate(); p._ksp.solveTranspose(cvec, x); return x
        pc = FenicsImplicitProblem(p.R_form, p.Q_form, p.theta, p.u, bcs=p.bcs,
                                   forward_solver=fwd, adjoint_solver=adj)
        self._check([(self.D, 2)], self.omega, lambda j: [(self.D[j], 2)], lambda j: self.omega,
                    fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: mu[0] >= 1, problem=pc)
        self.assertTrue(seen)                                  # the callables were used

    def test_batch_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            probe(self.prob, [(self.D, 1)], self.OMS[:2])

    def test_tuple_is_refused(self):
        with self.assertRaises(TypeError):
            probe(self.prob, [(tuple(self.D), 1)], self.omega)


if __name__ == "__main__":
    unittest.main()
