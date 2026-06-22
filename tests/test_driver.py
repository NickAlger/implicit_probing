# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
import itertools
import unittest

import numpy as np

from implicit_probing.multiset import Multiset
from implicit_probing.symbolic import (
    Term, ID, OMEGA, adjoint,
    seed_residual_r, seed_reverse, differentiate_over_lattice,
    extract_state_rhs, extract_adjoint_rhs,
)
from implicit_probing.driver import PartialTerm, probe
from implicit_probing.reference_problems import make_toy_problem, forward_probe_by_finite_difference


def ms(*xs):
    return Multiset(xs)


def power_tuples(directions):
    """Every power-tuple (the sub-probe lattice) for a ``((vector, max_power), ...)`` directions list."""
    return itertools.product(*[range(p + 1) for _, p in directions])


def fd_spec(directions, mu):
    """The (vector, order) list a finite-difference probe wants, for the power-tuple ``mu``."""
    return [(directions[k][0], mu[k]) for k in range(len(mu)) if mu[k] > 0]


class TestExtraction(unittest.TestCase):
    def test_state_rhs_removes_operator_term(self):
        D = differentiate_over_lattice(seed_residual_r(), ms(1, 1))
        beta = ms(1, 1)
        operator = Term(ID, 'R', ms(), Multiset([beta]))
        self.assertEqual(D[beta].get(operator), 1)              # present before
        rhs = extract_state_rhs(D[beta], beta)
        self.assertNotIn(operator, rhs)                          # removed
        # the remaining terms are exactly D_beta minus the operator term
        self.assertEqual(rhs, {t: c for t, c in D[beta].items() if t != operator})

    def test_state_rhs_first_order(self):
        D = differentiate_over_lattice(seed_residual_r(), ms(1))
        # A uhat_1 = b_1 with b_1 = -(d_theta R theta_1); the remaining term is the theta-derivative
        self.assertEqual(extract_state_rhs(D[ms(1)], ms(1)), {Term(ID, 'R', ms(1), ms()): 1})

    def test_state_rhs_missing_operator_raises(self):
        with self.assertRaises(ValueError):
            extract_state_rhs({Term(ID, 'Q', ms(), ms()): 1}, ms(1))

    def test_adjoint_rhs_base_case_is_minus_omega_du_Q(self):
        # beta = empty: c_empty = -omega(d_u Q); the operator term A* v is removed, leaving omega(Q).
        rhs = extract_adjoint_rhs(seed_reverse(), ms())
        self.assertEqual(rhs, {Term(OMEGA, 'Q', ms(), ms()): 1})

    def test_adjoint_rhs_removes_operator_term(self):
        D = differentiate_over_lattice(seed_reverse(), ms(1))
        beta = ms(1)
        operator = Term(adjoint(beta), 'R', ms(), ms())
        self.assertEqual(D[beta].get(operator), 1)
        self.assertNotIn(operator, extract_adjoint_rhs(D[beta], beta))


class TestProbeStructure(unittest.TestCase):
    def setUp(self):
        self.prob = make_toy_problem(seed=0)
        self.a = np.array([1.0, 0.3])
        self.b = np.array([0.4, -0.6])
        self.omega = np.array([0.7, -0.4])

    def test_returns_probe_for_every_power_tuple_with_right_shapes(self):
        directions = [(self.a, 2), (self.b, 1)]
        forward, reverse = probe(self.prob, directions, self.omega)
        keys = set(power_tuples(directions))
        self.assertEqual(set(forward), keys)
        self.assertEqual(set(reverse), keys)
        for mu in keys:
            self.assertEqual(forward[mu].shape, (self.prob.n_q,))   # output-space vector
            self.assertEqual(reverse[mu].shape, (self.prob.p,))     # parameter-space covector

    def test_forward_origin_is_q_at_expansion_point(self):
        forward, _ = probe(self.prob, [(self.a, 1)])
        np.testing.assert_allclose(forward[(0,)], self.prob.q(self.prob.theta0), atol=1e-12)


class TestProbeAgainstFiniteDifference(unittest.TestCase):
    """The driver's probes must match the independent FD ground truth for every probing symmetry."""

    def setUp(self):
        self.prob = make_toy_problem(seed=0)
        self.a = np.array([1.0, 0.3])
        self.b = np.array([0.4, -0.6])
        self.c = np.array([-0.2, 0.9])
        self.omega = np.array([0.7, -0.4])

    def test_forward_probes_all_symmetries(self):
        cases = {
            'order1':            [(self.a, 1)],
            'order2 symmetric':  [(self.a, 2)],
            'order2 asymmetric': [(self.a, 1), (self.b, 1)],
            'order3 symmetric':  [(self.a, 3)],
            'order3 partial':    [(self.a, 2), (self.b, 1)],
            'order3 asymmetric': [(self.a, 1), (self.b, 1), (self.c, 1)],
        }
        for name, directions in cases.items():
            with self.subTest(symmetry=name):
                forward, _ = probe(self.prob, directions)
                for mu in power_tuples(directions):
                    if sum(mu) == 0:
                        expected = self.prob.q(self.prob.theta0)  # forward[(0,...,0)] = q(theta0), exact
                    else:
                        expected = forward_probe_by_finite_difference(self.prob, fd_spec(directions, mu), h=1e-2)
                    np.testing.assert_allclose(forward[mu], expected, atol=1e-5)

    def test_reverse_probes_match_omega_of_forward(self):
        # reverse[mu] is the covector with  reverse[mu] . d_open = omega( D^{|mu|+1} q (mu-dirs, d_open) ).
        d_open = np.array([0.5, -0.8])
        for directions in [[(self.a, 1)], [(self.a, 2)], [(self.a, 1), (self.b, 1)]]:
            with self.subTest(directions=tuple(p for _, p in directions)):
                _, reverse = probe(self.prob, directions, self.omega)
                for mu in power_tuples(directions):
                    augmented = fd_spec(directions, mu) + [(d_open, 1)]
                    rhs = self.omega @ forward_probe_by_finite_difference(self.prob, augmented, h=1e-2)
                    lhs = reverse[mu] @ d_open
                    np.testing.assert_allclose(lhs, rhs, atol=1e-5)

    def test_order_four_edge_case(self):
        # Degree-3 toy => 4th-order partials vanish, but the order-4 incremental solves still run.
        # The driver must assemble D^4 q correctly (looser tol: 4th-order finite differences are noisier).
        directions = [(self.a, 4)]
        forward, _ = probe(self.prob, directions)
        expected = forward_probe_by_finite_difference(self.prob, fd_spec(directions, (4,)), h=3e-2)
        np.testing.assert_allclose(forward[(4,)], expected, atol=1e-3)


class _RecordingProblem:
    """Wraps an ImplicitProblem, capturing every PartialTerm handed to assemble_partial_sum."""
    def __init__(self, inner):
        self.inner = inner
        self.terms = []

    def solve_operator(self, b):
        return self.inner.solve_operator(b)

    def solve_operator_adjoint(self, c):
        return self.inner.solve_operator_adjoint(c)

    def assemble_partial_sum(self, terms, omega):
        self.terms.extend(terms)
        return self.inner.assemble_partial_sum(terms, omega)


class TestPartialTermRepresentation(unittest.TestCase):
    """theta_dirs / u_vecs are (vector, multiplicity) pairs -- a multiset, not a flat sequence."""

    def setUp(self):
        a = np.array([1.0, 0.3])
        b = np.array([0.4, -0.6])
        self.rec = _RecordingProblem(make_toy_problem(seed=0))
        probe(self.rec, [(a, 2), (b, 1)], omega=np.array([0.7, -0.4]))  # power 2 on direction a
        self.assertTrue(self.rec.terms)                          # something was actually assembled

    def test_directions_are_vector_multiplicity_pairs(self):
        for t in self.rec.terms:
            for block in (t.theta_dirs, t.u_vecs):
                for pair in block:
                    self.assertEqual(len(pair), 2)               # (vector, multiplicity)
                    _, mult = pair
                    self.assertIsInstance(mult, int)
                    self.assertGreaterEqual(mult, 1)

    def test_each_block_groups_distinct_directions(self):
        # within a block the entries are distinct directions (grouped by multiplicity, not split)
        for t in self.rec.terms:
            for block in (t.theta_dirs, t.u_vecs):
                ids = [id(vec) for vec, _ in block]
                self.assertEqual(len(ids), len(set(ids)))

    def test_repetition_is_encoded_not_flattened(self):
        # probing D^3 q with direction 1 twice must surface a multiplicity >= 2 in each block somewhere
        # (the old flat representation would have lost this). theta: the d^3/dtheta^3 . d1^2 d2 term;
        # u: the incremental RHS for uhat_{1,1} carries uhat_1^2.
        max_theta = max((m for t in self.rec.terms for _, m in t.theta_dirs), default=0)
        max_u = max((m for t in self.rec.terms for _, m in t.u_vecs), default=0)
        self.assertGreaterEqual(max_theta, 2)
        self.assertGreaterEqual(max_u, 2)


if __name__ == '__main__':
    unittest.main()
