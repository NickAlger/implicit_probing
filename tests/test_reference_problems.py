# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
import itertools
import unittest

import numpy as np

from implicit_probing.reference_problems import (
    Polynomial,
    ImplicitPolynomialProblem,
    make_toy_problem,
    forward_probe_by_finite_difference,
)


def _random_polynomial(out_dim, in_dim, degree, seed):
    """A random symmetric-coefficient Polynomial, for property/derivative checks."""
    from implicit_probing.reference_problems import _random_symmetric
    rng = np.random.default_rng(seed)
    coeffs = [0.7 * rng.standard_normal((out_dim,))]
    coeffs += [_random_symmetric(out_dim, in_dim, m, rng, 0.5) for m in range(1, degree + 1)]
    return Polynomial(coeffs)


class TestPolynomial(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)
        self.P = _random_polynomial(out_dim=2, in_dim=4, degree=3, seed=7)

    def test_call_equals_zeroth_derivative(self):
        w = self.rng.standard_normal(4)
        np.testing.assert_allclose(self.P(w), self.P.derivative(w, ()))

    def test_vanishes_above_degree(self):
        w = self.rng.standard_normal(4)
        dirs = [self.rng.standard_normal(4) for _ in range(self.P.degree + 1)]
        np.testing.assert_allclose(self.P.derivative(w, dirs), 0.0, atol=1e-12)

    def test_multilinear_symmetry(self):
        # the derivative is symmetric in its direction arguments
        w = self.rng.standard_normal(4)
        a, b, c = (self.rng.standard_normal(4) for _ in range(3))
        base = self.P.derivative(w, [a, b, c])
        for perm in itertools.permutations([a, b, c]):
            np.testing.assert_allclose(self.P.derivative(w, list(perm)), base, atol=1e-12)

    def test_linearity_in_a_direction(self):
        w, a, b = (self.rng.standard_normal(4) for _ in range(3))
        # homogeneity
        np.testing.assert_allclose(self.P.derivative(w, [3.0 * a]), 3.0 * self.P.derivative(w, [a]))
        # additivity
        np.testing.assert_allclose(
            self.P.derivative(w, [a + b]),
            self.P.derivative(w, [a]) + self.P.derivative(w, [b]),
        )

    def test_jacobian_matches_directional_derivative(self):
        w = self.rng.standard_normal(4)
        J = self.P.jacobian(w)
        for i in range(4):
            e_i = np.eye(4)[i]
            np.testing.assert_allclose(J[:, i], self.P.derivative(w, [e_i]), atol=1e-12)

    def test_first_and_second_derivative_vs_finite_difference(self):
        w = self.rng.standard_normal(4)
        a, b = self.rng.standard_normal(4), self.rng.standard_normal(4)
        eps = 1e-4
        # first directional derivative
        fd1 = (self.P(w + eps * a) - self.P(w - eps * a)) / (2 * eps)
        np.testing.assert_allclose(self.P.derivative(w, [a]), fd1, atol=1e-6)
        # mixed second derivative d^2/da db
        fd2 = (self.P(w + eps * a + eps * b) - self.P(w + eps * a - eps * b)
               - self.P(w - eps * a + eps * b) + self.P(w - eps * a - eps * b)) / (4 * eps ** 2)
        np.testing.assert_allclose(self.P.derivative(w, [a, b]), fd2, atol=1e-5)

    def test_structural_shape_validation(self):
        with self.assertRaises(ValueError):
            Polynomial([np.zeros((2,)), np.zeros((2, 3, 3))])  # missing the order-1 tensor's shape contract


class TestImplicitProblem(unittest.TestCase):
    def setUp(self):
        self.prob = make_toy_problem(seed=0)

    def test_solve_state_residual_is_zero(self):
        u0 = self.prob.u0
        np.testing.assert_allclose(self.prob.residual(self.prob.theta0, u0), 0.0, atol=1e-12)
        # also at a perturbed theta
        theta = np.array([0.1, -0.2])
        np.testing.assert_allclose(self.prob.residual(theta, self.prob.solve_state(theta)), 0.0, atol=1e-12)

    def test_operator_is_well_conditioned(self):
        self.assertLess(np.linalg.cond(self.prob.A()), 10.0)

    def test_state_jacobian_vs_finite_difference(self):
        theta, u = self.prob.theta0, self.prob.u0
        A = self.prob.A()
        eps = 1e-6
        A_fd = np.zeros((self.prob.n_u, self.prob.n_u))
        for j in range(self.prob.n_u):
            e_j = np.eye(self.prob.n_u)[j]
            A_fd[:, j] = (self.prob.residual(theta, u + eps * e_j)
                          - self.prob.residual(theta, u - eps * e_j)) / (2 * eps)
        np.testing.assert_allclose(A, A_fd, atol=1e-6)

    def test_operator_solves(self):
        b = np.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(self.prob.A() @ self.prob.solve_A(b), b, atol=1e-12)
        np.testing.assert_allclose(self.prob.A().T @ self.prob.solve_A_adjoint(b), b, atol=1e-12)

    def test_q_matches_Q_at_solution(self):
        q = self.prob.q(self.prob.theta0)
        np.testing.assert_allclose(q, self.prob.Q(self.prob.w0), atol=1e-12)

    def test_partials_are_lifted_polynomial_derivatives(self):
        th1, th2 = np.array([1.0, 0.0]), np.array([0.3, -0.7])
        uv = np.array([0.2, -0.1, 0.4])
        w0 = self.prob.w0
        # pure theta, pure u, and mixed
        np.testing.assert_allclose(
            self.prob.partial_Q([th1, th2], []),
            self.prob.Q.derivative(w0, [self.prob.lift_theta(th1), self.prob.lift_theta(th2)]))
        np.testing.assert_allclose(
            self.prob.partial_R([], [uv]),
            self.prob.R.derivative(w0, [self.prob.lift_u(uv)]))
        np.testing.assert_allclose(
            self.prob.partial_Q([th1], [uv]),
            self.prob.Q.derivative(w0, [self.prob.lift_theta(th1), self.prob.lift_u(uv)]))

    def test_structural_validation(self):
        rng = np.random.default_rng(0)
        R = Polynomial([rng.standard_normal((3,)), rng.standard_normal((3, 5))])
        Q = Polynomial([rng.standard_normal((2,)), rng.standard_normal((2, 5))])
        with self.assertRaises(ValueError):  # n_u mismatch
            ImplicitPolynomialProblem(R, Q, np.zeros(2), p=2, n_u=4, n_q=2)


def _make_constant_state_problem(p, n_u, n_q, seed):
    """An explicit problem: R(theta,u) = u - c, so u(theta) = c and q(theta) = Q(theta, c)."""
    rng = np.random.default_rng(seed)
    n = p + n_u
    c = 0.2 * rng.standard_normal(n_u)
    R = Polynomial([-c, np.concatenate([np.zeros((n_u, p)), np.eye(n_u)], axis=1)])  # u - c
    from implicit_probing.reference_problems import _random_symmetric
    Q_coeffs = [0.5 * rng.standard_normal((n_q,))]
    Q_coeffs += [_random_symmetric(n_q, n, m, rng, 0.4) for m in range(1, 4)]
    Q = Polynomial(Q_coeffs)
    return ImplicitPolynomialProblem(R, Q, np.zeros(p), p, n_u, n_q)


class TestForwardProbeGroundTruth(unittest.TestCase):
    """The finite-difference ground truth must reproduce the exact probe for ANY probing symmetry."""

    def test_matches_exact_on_explicit_map(self):
        # On a problem with constant state, q(theta) = Q(theta, c) is an explicit cubic in theta, so the
        # forward probe equals Q's exact theta-partial -- checkable for symmetric, partial, asymmetric.
        prob = _make_constant_state_problem(p=2, n_u=3, n_q=2, seed=1)
        w0 = prob.w0
        d1 = np.array([1.0, 0.0])
        d2 = np.array([0.4, -0.6])
        d3 = np.array([-0.2, 0.9])
        specs = {
            'order1': [(d1, 1)],
            'order2 symmetric': [(d1, 2)],
            'order2 asymmetric': [(d1, 1), (d2, 1)],
            'order3 symmetric': [(d1, 3)],
            'order3 partial': [(d1, 2), (d2, 1)],
            'order3 asymmetric': [(d1, 1), (d2, 1), (d3, 1)],
        }
        for name, spec in specs.items():
            with self.subTest(symmetry=name):
                lifted = [prob.lift_theta(d) for d, m in spec for _ in range(m)]
                exact = prob.Q.derivative(w0, lifted)
                fd = forward_probe_by_finite_difference(prob, spec, h=0.1)
                np.testing.assert_allclose(fd, exact, atol=1e-6)

    def test_first_order_probe_on_implicit_map(self):
        # End-to-end on the genuinely implicit toy: the j=1 forward probe via the adjoint/implicit
        # chain rule (hand-written) must match the finite-difference ground truth.
        prob = make_toy_problem(seed=0)
        theta_hat = np.array([0.7, -0.4])
        # A u_hat = -d_theta R theta_hat;  Dq theta_hat = d_theta Q theta_hat + d_u Q u_hat
        u_hat = prob.solve_A(-prob.partial_R([theta_hat], []))
        Dq = prob.partial_Q([theta_hat], []) + prob.partial_Q([], [u_hat])
        fd = forward_probe_by_finite_difference(prob, [(theta_hat, 1)], h=1e-2)
        np.testing.assert_allclose(Dq, fd, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
