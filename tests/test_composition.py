# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
import unittest

import numpy as np

from implicit_probing.multiset import Multiset, subset_lattice
from implicit_probing.driver import probe
from implicit_probing.composition import ComposedProblem, MatrixOperator
from implicit_probing.reference_problems import make_toy_problem, forward_probe_by_finite_difference


def ms(*xs):
    return Multiset(xs)


class TestComposedProblem(unittest.TestCase):
    def setUp(self):
        self.inner = make_toy_problem(seed=0)            # theta in R^2, q in R^2
        rng = np.random.default_rng(1)
        self.C = rng.standard_normal((2, 3))             # input map: features R^3 -> theta R^2
        self.W = rng.standard_normal((1, 2))             # output map: obs R^2 -> reduced R^1
        self.composed = ComposedProblem(self.inner, MatrixOperator(self.C), MatrixOperator(self.W))
        self.x_dirs = {1: rng.standard_normal(3), 2: rng.standard_normal(3), 3: rng.standard_normal(3)}
        self.omega_z = rng.standard_normal(1)            # output functional on the reduced output

    def test_forward_probes_match_W_times_fd(self):
        # D^j f(x0)(xhat...) = W( D^j q(theta0)(C xhat...) ); anchor to the inner's FD ground truth.
        for alpha in [ms(1), ms(1, 1), ms(1, 2), ms(1, 1, 2)]:
            with self.subTest(alpha=alpha):
                forward, _ = probe(self.composed, alpha, self.x_dirs)
                for beta in subset_lattice(alpha):
                    if len(beta) == 0:
                        continue
                    spec = [(self.C @ self.x_dirs[k], count) for k, count in beta.items()]
                    expected = self.W @ forward_probe_by_finite_difference(self.inner, spec, h=1e-2)
                    np.testing.assert_allclose(forward[beta], expected, atol=1e-5)

    def test_reverse_probes_adjointness_on_composed_map(self):
        # On the composed map f: reverse[beta] . xhat_k == omega_z . forward[beta + {k}] (exact).
        alpha = ms(1, 1, 2)
        forward, reverse = probe(self.composed, alpha, self.x_dirs, self.omega_z)
        for beta in subset_lattice(alpha):
            for k, xvec in self.x_dirs.items():
                child = beta.add(k)
                if not child.issubmultiset(alpha):
                    continue
                lhs = float(reverse[beta] @ xvec)
                rhs = float(self.omega_z @ forward[child])
                with self.subTest(beta=beta, k=k):
                    np.testing.assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-12)

    def test_output_shapes(self):
        forward, reverse = probe(self.composed, ms(1, 2), self.x_dirs, self.omega_z)
        for beta in subset_lattice(ms(1, 2)):
            self.assertEqual(forward[beta].shape, (1,))    # reduced output space (z)
            self.assertEqual(reverse[beta].shape, (3,))    # input (feature) covector

    def test_identity_maps_recover_inner(self):
        # ComposedProblem with no maps must reproduce probing the inner directly.
        composed = ComposedProblem(self.inner)             # identity input/output maps
        dirs = {1: np.array([1.0, 0.3]), 2: np.array([0.4, -0.6])}
        omega = np.array([0.7, -0.4])
        f_c, r_c = probe(composed, ms(1, 2), dirs, omega)
        f_i, r_i = probe(self.inner, ms(1, 2), dirs, omega)
        for beta in subset_lattice(ms(1, 2)):
            np.testing.assert_allclose(f_c[beta], f_i[beta])
            np.testing.assert_allclose(r_c[beta], r_i[beta])


if __name__ == '__main__':
    unittest.main()
