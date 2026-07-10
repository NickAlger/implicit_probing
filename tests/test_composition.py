# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
import itertools
import unittest

import numpy as np

from implicit_probing.driver import probe
from implicit_probing.composition import ComposedProblem, MatrixOperator
from implicit_probing.reference_problems import make_toy_problem, forward_probe_by_finite_difference


def power_tuples(directions):
    """Every power-tuple (the sub-probe lattice) for a ``((vector, max_power), ...)`` directions list."""
    return itertools.product(*[range(p + 1) for _, p in directions])


class TestComposedProblem(unittest.TestCase):
    def setUp(self):
        self.inner = make_toy_problem(seed=0)            # theta in R^2, q in R^2
        rng = np.random.default_rng(1)
        self.C = rng.standard_normal((2, 3))             # input map: features R^3 -> theta R^2
        self.W = rng.standard_normal((1, 2))             # output map: obs R^2 -> reduced R^1
        self.composed = ComposedProblem(self.inner, MatrixOperator(self.C), MatrixOperator(self.W))
        self.x1 = rng.standard_normal(3)
        self.x2 = rng.standard_normal(3)
        self.omega_z = rng.standard_normal(1)            # output functional on the reduced output

    def test_forward_probes_match_W_times_fd(self):
        # D^j f(x0)(xhat...) = W( D^j q(theta0)(C xhat...) ); anchor to the inner's FD ground truth.
        cases = [
            [(self.x1, 1)],
            [(self.x1, 2)],
            [(self.x1, 1), (self.x2, 1)],
            [(self.x1, 2), (self.x2, 1)],
        ]
        for directions in cases:
            with self.subTest(directions=tuple(p for _, p in directions)):
                forward, _ = probe(self.composed, directions)
                for mu in power_tuples(directions):
                    if sum(mu) == 0:
                        continue
                    # FD perturbs the INNER theta along the C-mapped feature directions
                    spec = [(self.C @ directions[k][0], mu[k]) for k in range(len(mu)) if mu[k] > 0]
                    expected = self.W @ forward_probe_by_finite_difference(self.inner, spec, h=1e-2)
                    np.testing.assert_allclose(forward[mu], expected, atol=1e-5)

    def test_reverse_probes_adjointness_on_composed_map(self):
        # On the composed map f: reverse[mu] . xhat_k == omega_z . forward[mu + e_k] (exact).
        directions = [(self.x1, 2), (self.x2, 1)]
        forward, reverse = probe(self.composed, directions, self.omega_z)
        for mu in power_tuples(directions):
            for k, (xvec, p_k) in enumerate(directions):
                if mu[k] >= p_k:
                    continue
                child = mu[:k] + (mu[k] + 1,) + mu[k + 1:]
                lhs = float(reverse[mu] @ xvec)
                rhs = float(self.omega_z @ forward[child])
                with self.subTest(mu=mu, k=k):
                    np.testing.assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-12)

    def test_output_shapes(self):
        directions = [(self.x1, 1), (self.x2, 1)]
        forward, reverse = probe(self.composed, directions, self.omega_z)
        for mu in power_tuples(directions):
            self.assertEqual(forward[mu].shape, (1,))    # reduced output space (z)
            self.assertEqual(reverse[mu].shape, (3,))    # input (feature) covector

    def test_identity_maps_recover_inner(self):
        # ComposedProblem with no maps must reproduce probing the inner directly.
        composed = ComposedProblem(self.inner)             # identity input/output maps
        a = np.array([1.0, 0.3])
        b = np.array([0.4, -0.6])
        omega = np.array([0.7, -0.4])
        directions = [(a, 1), (b, 1)]
        f_c, r_c = probe(composed, directions, omega)
        f_i, r_i = probe(self.inner, directions, omega)
        for mu in power_tuples(directions):
            np.testing.assert_allclose(f_c[mu], f_i[mu])
            np.testing.assert_allclose(r_c[mu], r_i[mu])


if __name__ == '__main__':
    unittest.main()
