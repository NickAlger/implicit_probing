# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
#
# Gated test for the JAX hook: runs only where jax is importable (e.g. the `t3toolbox` conda env),
# and is skipped (not failed) elsewhere. High-order probes need float64, so x64 is enabled below.
#
# The JAX problem re-codes the trusted numpy toy's polynomials (same coefficients) as JAX callables, so
# we get THREE independent checks of the hook: (1) its probes vs the numpy reference implementation's
# probes on the identical map -- exact at every order, the two hooks sharing only the driver, not the
# partial machinery (analytic tensor contraction vs Taylor-mode jet); (2) low-order forward probes vs an
# independent finite-difference ground truth (which touches only the numpy end-to-end map); and (3) the
# exact reverse/forward adjoint identity. Plus the minimal solve count.
import math
import unittest

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from implicit_probing import probe
from implicit_probing import validation
from implicit_probing.reference_problems import make_toy_problem
from implicit_probing.jax import JaxImplicitProblem


def _poly_eval(coeffs, w):
    """Evaluate a Polynomial (symmetric Taylor-coefficient form) in JAX -- mirrors Polynomial.__call__."""
    out = jnp.asarray(coeffs[0])
    for m in range(1, len(coeffs)):
        term = jnp.asarray(coeffs[m])                                  # (out_dim,) + (in_dim,)*m
        for _ in range(m):
            term = jnp.tensordot(term, w, axes=([term.ndim - 1], [0]))
        out = out + term / math.factorial(m)
    return out


def _jax_problem_from_toy(toy):
    """A JaxImplicitProblem whose R, Q are the toy's polynomials re-coded in JAX (identical map)."""
    Rc = [np.asarray(c) for c in toy.R.coeffs]
    Qc = [np.asarray(c) for c in toy.Q.coeffs]
    R = lambda theta, u: _poly_eval(Rc, jnp.concatenate([theta, u]))
    Q = lambda theta, u: _poly_eval(Qc, jnp.concatenate([theta, u]))
    return JaxImplicitProblem(R, Q, jnp.asarray(toy.theta0), jnp.asarray(toy.u0))


class TestJaxProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toy = make_toy_problem(seed=0)              # numpy reference: theta in R^2, u in R^3, q in R^2
        cls.jp = _jax_problem_from_toy(cls.toy)
        cls.a = np.array([1.0, 0.3])
        cls.b = np.array([0.4, -0.6])
        cls.omega = np.array([1.0, -0.5])

    def test_operator_matches_numpy_state_jacobian(self):
        # A = d_u R from jax.jacfwd must equal the reference's analytic state jacobian.
        self.assertLess(np.linalg.norm(np.asarray(self.jp.A) - self.toy.A()), 1e-12)

    def test_probes_match_numpy_reference(self):
        # Strongest check: same driver, two DIFFERENT problem hooks (jet AD vs analytic contraction) on
        # the identical polynomial map -> identical probes at every order, to round-off.
        directions = [(self.a, 2), (self.b, 1)]
        f_ref, r_ref = probe(self.toy, directions, self.omega)
        f_jax, r_jax = probe(self.jp,
                             [(jnp.asarray(self.a), 2), (jnp.asarray(self.b), 1)],
                             jnp.asarray(self.omega))
        for mu in f_ref:
            with self.subTest(mu=mu):
                fwd_rel = (np.linalg.norm(np.asarray(f_jax[mu]) - f_ref[mu])
                           / max(np.linalg.norm(f_ref[mu]), 1e-30))
                rev_rel = (np.linalg.norm(np.asarray(r_jax[mu]) - r_ref[mu])
                           / max(np.linalg.norm(r_ref[mu]), 1e-30))
                self.assertLess(fwd_rel, 1e-10, f"forward {mu}: rel {fwd_rel:.2e}")
                self.assertLess(rev_rel, 1e-10, f"reverse {mu}: rel {rev_rel:.2e}")

    def test_forward_probes_match_finite_difference(self):
        # Independent anchor: finite differences of the numpy end-to-end map (no shared partial code).
        # Capped at order 2 -- an order-3 central difference is itself only good to ~1e-5, too coarse to
        # certify an exact probe; the reference comparison above covers the high orders to round-off.
        a, b = jnp.asarray(self.a), jnp.asarray(self.b)
        cases = [
            ("order1",            [(a, 1)],          (1,),   1e-6),
            ("order2 symmetric",  [(a, 2)],          (2,),   1e-5),
            ("order2 asymmetric", [(a, 1), (b, 1)],  (1, 1), 1e-5),
        ]
        for name, directions, mu, atol in cases:
            with self.subTest(symmetry=name):
                forward, _ = probe(self.jp, directions)
                y = np.asarray(forward[mu])
                spec = [(np.asarray(directions[k][0]), mu[k]) for k in range(len(mu)) if mu[k] > 0]
                y_fd = validation.forward_probe_by_finite_difference(
                    self.toy.q, self.toy.theta0, spec, h=1e-3)
                rel = np.linalg.norm(y - y_fd) / max(np.linalg.norm(y_fd), 1e-30)
                self.assertLess(rel, atol, f"{name}: rel err {rel:.2e}")

    def test_reverse_probes_match_omega_paired_forward(self):
        # Exact discrete adjointness: reverse[mu] . d_k == omega . forward[mu + e_k], swept over the
        # lattice. No finite differences; cross-checks the reverse (adjoint-solve) probes against the
        # forward ones to solver precision.
        directions = [(jnp.asarray(self.a), 2), (jnp.asarray(self.b), 1)]
        forward, reverse = probe(self.jp, directions, jnp.asarray(self.omega))
        err = validation.reverse_forward_adjointness(
            forward, reverse, directions, jnp.asarray(self.omega),
            pair_input=lambda rev, d: float(np.dot(np.asarray(rev), np.asarray(d))),
            pair_output=lambda om, fwd: float(np.dot(np.asarray(om), np.asarray(fwd))))
        self.assertLess(err, 1e-9, f"max adjointness rel err {err:.2e}")


class _CountingProblem:
    """Wraps a problem, counting the linearized solves the driver performs (at the solve boundary)."""
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


class TestJaxSolveCounts(unittest.TestCase):
    """The hook must drive the MINIMAL number of solves: ``prod(p_k+1) - 1`` forward, ``prod(p_k+1)``
    adjoint (the empty node is the user's base state solve; the base adjoint is a real solve). Orders
    kept low here so the high-order jet kernels stay cheap to compile."""
    @classmethod
    def setUpClass(cls):
        cls.jp = _jax_problem_from_toy(make_toy_problem(seed=0))
        cls.omega = jnp.asarray(np.array([0.7, -0.4]))

    @staticmethod
    def _directions(powers):
        rng = np.random.default_rng(0)
        return [(jnp.asarray(rng.standard_normal(2)), p) for p in powers]

    def test_forward_and_adjoint_solve_counts(self):
        for powers in [(2,), (2, 1), (1, 1, 1)]:
            with self.subTest(powers=powers):
                problem = _CountingProblem(self.jp)
                probe(problem, self._directions(powers), self.omega)
                L = math.prod(p + 1 for p in powers)
                self.assertEqual(problem.n_forward, L - 1)
                self.assertEqual(problem.n_adjoint, L)

    def test_no_adjoint_solves_when_omega_is_none(self):
        powers = (2, 1)
        problem = _CountingProblem(self.jp)
        probe(problem, self._directions(powers))             # omega=None -> forward probes only
        L = math.prod(p + 1 for p in powers)
        self.assertEqual(problem.n_forward, L - 1)
        self.assertEqual(problem.n_adjoint, 0)


if __name__ == "__main__":
    unittest.main()
