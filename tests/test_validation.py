# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
#
# Tests for the finite-difference verification helpers (implicit_probing/validation.py), exercised on
# the numpy toy -- including a custom perturb/pair hook path so the vector-agnostic design is covered
# without needing a FEniCS environment (the FEniCS tests exercise the hooks for real).
import unittest

import numpy as np

from implicit_probing.driver import probe
from implicit_probing import validation
from implicit_probing.reference_problems import make_toy_problem

_D1 = np.array([1.0, 0.3])
_D2 = np.array([0.4, -0.6])


class TestForwardProbeFiniteDifference(unittest.TestCase):
    def setUp(self):
        self.problem = make_toy_problem()

    def test_matches_driver_probe_across_symmetries(self):
        prob = self.problem
        cases = [
            ("order1",            [(_D1, 1)],            (1,)),
            ("order2 symmetric",  [(_D1, 2)],            (2,)),
            ("order2 asymmetric", [(_D1, 1), (_D2, 1)],  (1, 1)),
            ("order3 mixed",      [(_D1, 2), (_D2, 1)],  (2, 1)),
        ]
        for name, directions, mu in cases:
            with self.subTest(case=name):
                forward, _ = probe(prob, directions)
                spec = [(directions[k][0], mu[k]) for k in range(len(mu)) if mu[k] > 0]
                fd = validation.forward_probe_by_finite_difference(prob.q, prob.theta0, spec)
                np.testing.assert_allclose(forward[mu], fd, rtol=1e-5, atol=1e-7)

    def test_perturb_hook_is_used_so_engine_never_touches_the_vector(self):
        # Box the parameter point in an opaque type: the engine must reach it ONLY through perturb / q,
        # never with numpy arithmetic of its own, so a non-array point has to work identically.
        prob = self.problem

        def box(v):
            return ("boxed", v)

        def perturb(point, scale, direction):
            return box(point[1] + scale * direction)

        def q(point):
            return prob.q(point[1])

        spec = [(_D1, 2), (_D2, 1)]
        boxed = validation.forward_probe_by_finite_difference(q, box(prob.theta0), spec, perturb=perturb)
        plain = validation.forward_probe_by_finite_difference(prob.q, prob.theta0, spec)
        np.testing.assert_allclose(boxed, plain, rtol=1e-12, atol=1e-14)


class TestReverseForwardAdjointness(unittest.TestCase):
    def setUp(self):
        self.problem = make_toy_problem()
        self.directions = [(_D1, 2), (_D2, 1)]
        self.omega = np.array([1.0, 0.0])
        self.forward, self.reverse = probe(self.problem, self.directions, self.omega)

    def test_zero_on_correct_probes(self):
        err = validation.reverse_forward_adjointness(
            self.forward, self.reverse, self.directions, self.omega)
        self.assertLess(err, 1e-9)

    def test_custom_pairings_match_default(self):
        err = validation.reverse_forward_adjointness(
            self.forward, self.reverse, self.directions, self.omega,
            pair_input=lambda a, b: float(np.dot(a, b)),
            pair_output=lambda a, b: float(np.dot(a, b)))
        self.assertLess(err, 1e-9)

    def test_detects_a_corrupted_reverse_probe(self):
        bad = dict(self.reverse)
        origin = (0, 0)
        bad[origin] = bad[origin] + np.array([1.0, 0.0])   # perturb the gradient reverse[(0,0)]
        err = validation.reverse_forward_adjointness(
            self.forward, bad, self.directions, self.omega)
        self.assertGreater(err, 1e-3)


if __name__ == "__main__":
    unittest.main()
