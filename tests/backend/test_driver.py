# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
import unittest

import numpy as np

from implicit_probing.backend.multiset import Multiset, subset_lattice
from implicit_probing.backend.symbolic import (
    Term, ID, OMEGA, adjoint,
    seed_residual_r, seed_reverse, differentiate_over_lattice,
    extract_state_rhs, extract_adjoint_rhs,
)
from implicit_probing.backend.driver import PartialTerm, probe
from implicit_probing.reference_problems import make_toy_problem, forward_probe_by_finite_difference


def ms(*xs):
    return Multiset(xs)


def spec_of(beta, direction_vectors):
    """The (vector, order) list a finite-difference probe wants, from a label-multiset beta."""
    return [(direction_vectors[label], count) for label, count in beta.items()]


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
        self.dirs = {1: np.array([1.0, 0.3]), 2: np.array([0.4, -0.6])}
        self.omega = np.array([0.7, -0.4])

    def test_returns_probe_for_every_subset_with_right_shapes(self):
        alpha = ms(1, 1, 2)
        forward, reverse = probe(self.prob, alpha, self.dirs, self.omega)
        self.assertEqual(set(forward), set(subset_lattice(alpha)))
        self.assertEqual(set(reverse), set(subset_lattice(alpha)))
        for beta in subset_lattice(alpha):
            self.assertEqual(forward[beta].shape, (self.prob.n_q,))   # output-space vector
            self.assertEqual(reverse[beta].shape, (self.prob.p,))     # parameter-space covector

    def test_forward_empty_is_q_at_expansion_point(self):
        forward, _ = probe(self.prob, ms(1), self.dirs)
        np.testing.assert_allclose(forward[ms()], self.prob.q(self.prob.theta0), atol=1e-12)


class TestProbeAgainstFiniteDifference(unittest.TestCase):
    """The driver's probes must match the independent FD ground truth for every probing symmetry."""

    def setUp(self):
        self.prob = make_toy_problem(seed=0)
        self.dirs = {
            1: np.array([1.0, 0.3]),
            2: np.array([0.4, -0.6]),
            3: np.array([-0.2, 0.9]),
        }
        self.omega = np.array([0.7, -0.4])

    def test_forward_probes_all_symmetries(self):
        cases = {
            'order1':            ms(1),
            'order2 symmetric':  ms(1, 1),
            'order2 asymmetric': ms(1, 2),
            'order3 symmetric':  ms(1, 1, 1),
            'order3 partial':    ms(1, 1, 2),
            'order3 asymmetric': ms(1, 2, 3),
        }
        for name, alpha in cases.items():
            with self.subTest(symmetry=name):
                forward, _ = probe(self.prob, alpha, self.dirs)
                for beta in subset_lattice(alpha):
                    if len(beta) == 0:
                        expected = self.prob.q(self.prob.theta0)  # forward[empty] = q(theta0), exact
                    else:
                        expected = forward_probe_by_finite_difference(self.prob, spec_of(beta, self.dirs), h=1e-2)
                    np.testing.assert_allclose(forward[beta], expected, atol=1e-5)

    def test_reverse_probes_match_omega_of_forward(self):
        # psi_beta is the covector with  psi_beta . d_open = omega( D^{|beta|+1} q (beta-dirs, d_open) ).
        d_open = np.array([0.5, -0.8])
        for alpha in [ms(1), ms(1, 1), ms(1, 2)]:
            with self.subTest(alpha=alpha):
                _, reverse = probe(self.prob, alpha, self.dirs, self.omega)
                for beta in subset_lattice(alpha):
                    augmented = spec_of(beta, self.dirs) + [(d_open, 1)]
                    rhs = self.omega @ forward_probe_by_finite_difference(self.prob, augmented, h=1e-2)
                    lhs = reverse[beta] @ d_open
                    np.testing.assert_allclose(lhs, rhs, atol=1e-5)

    def test_order_four_edge_case(self):
        # Degree-3 toy => 4th-order partials vanish, but the order-4 incremental solves still run.
        # The driver must assemble D^4 q correctly (looser tol: 4th-order finite differences are noisier).
        alpha = ms(1, 1, 1, 1)
        forward, _ = probe(self.prob, alpha, self.dirs)
        expected = forward_probe_by_finite_difference(self.prob, spec_of(alpha, self.dirs), h=3e-2)
        np.testing.assert_allclose(forward[alpha], expected, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
