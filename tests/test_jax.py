# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
#
# Gated test for the JAX hook: runs only where jax is importable (install the `[jax]` extra), and is
# skipped (not failed) elsewhere. High-order probes need float64, so x64 is enabled below.
#
# The JAX problem re-codes the trusted numpy toy's polynomials (same coefficients) as JAX callables, so
# we get THREE independent checks of the hook: (1) its probes vs the numpy reference implementation's
# probes on the identical map -- exact at every order, the two hooks sharing only the driver, not the
# partial machinery (analytic tensor contraction vs Taylor-mode jet); (2) low-order forward probes vs an
# independent finite-difference ground truth (which touches only the numpy end-to-end map); and (3) the
# exact reverse/forward adjoint identity. Plus the minimal solve count.
import math
import unittest
from unittest import mock

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


class TestRefreeze(unittest.TestCase):
    """``refreeze`` must be exactly equivalent to fresh construction at the new point -- same map,
    same machinery, only the (traced) expansion point moved -- while keeping the compiled kernels."""

    def test_refreeze_matches_fresh_construction(self):
        toy = make_toy_problem(seed=0)
        jp = _jax_problem_from_toy(toy)
        a = jnp.asarray([1.0, 0.3])
        b = jnp.asarray([0.4, -0.6])
        omega = jnp.asarray([1.0, -0.5])
        probe(jp, [(a, 2), (b, 1)], omega)              # warm the kernels at the original point

        rng = np.random.default_rng(3)                  # a second point (the equivalence holds
        theta1 = jnp.asarray(rng.standard_normal(2))    # whether or not it solves R = 0; both
        u1 = jnp.asarray(rng.standard_normal(3))        # objects run the identical computation)
        fresh = JaxImplicitProblem(jp.R, jp.Q, theta1, u1)
        out = jp.refreeze(theta1, u1)
        self.assertIs(out, jp)                          # returns self (chainable)

        f_re, r_re = probe(jp, [(a, 2), (b, 1)], omega)
        f_fr, r_fr = probe(fresh, [(a, 2), (b, 1)], omega)
        self.assertEqual(set(f_re), set(f_fr))
        for mu in f_fr:
            np.testing.assert_allclose(np.asarray(f_re[mu]), np.asarray(f_fr[mu]),
                                       rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(np.asarray(r_re[mu]), np.asarray(r_fr[mu]),
                                       rtol=1e-13, atol=1e-14)

    def test_refreeze_shape_mismatch_raises(self):
        jp = _jax_problem_from_toy(make_toy_problem(seed=0))
        with self.assertRaises(ValueError):
            jp.refreeze(jnp.zeros(3), jnp.zeros(3))     # theta dim is 2

    def test_refreeze_with_stale_custom_solvers_raises(self):
        toy = make_toy_problem(seed=0)
        base = _jax_problem_from_toy(toy)
        custom = JaxImplicitProblem(base.R, base.Q, jnp.asarray(toy.theta0), jnp.asarray(toy.u0),
                                    forward_solver=base.solve_operator,
                                    adjoint_solver=base.solve_operator_adjoint)
        with self.assertRaises(ValueError):             # custom solvers are point-specific
            custom.refreeze(jnp.asarray(toy.theta0), jnp.asarray(toy.u0))


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



class TestJaxBatchedProbes(unittest.TestCase):
    """The batched-probe contract (``dev/batched_probes_design.md`` §3): a ``(B, .)`` direction or
    ``omega`` means B independent probes at the frozen point in ONE ``probe`` call; single inputs are
    shared by every member; a result is batched iff its request used a batched input -- per key,
    ``forward[mu]`` iff some batched direction k has ``mu_k >= 1``, ``reverse[mu]`` iff ``omega`` is
    batched or such a direction exists. Every batched value is checked member-by-member against a loop
    of single probes (tolerances, not equality: the vmapped kernel agrees with the single one to
    round-off) and, once, against the numpy reference's own batched path."""
    B = 3

    @classmethod
    def setUpClass(cls):
        cls.toy = make_toy_problem(seed=0)               # theta in R^2, u in R^3, q in R^2
        cls.jp = _jax_problem_from_toy(cls.toy)          # ONE instance for the class: a fresh instance
        rng = np.random.default_rng(11)                  #   recompiles every (batched) kernel
        cls.V = jnp.asarray(rng.standard_normal((cls.B, 2)))    # a batch of B theta-directions
        cls.b = jnp.asarray(rng.standard_normal(2))             # a single theta-direction
        cls.OM = jnp.asarray(rng.standard_normal((cls.B, 2)))   # a batch of B output functionals
        cls.om = jnp.asarray(rng.standard_normal(2))            # a single output functional

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()          # drop this class's compiled kernels (many structures x batch
                                    # patterns); the suite otherwise accumulates executables until
                                    # LLVM cannot map more memory

    def _loop(self, problem, directions_of, omega_of):
        outs = [probe(problem, directions_of(j), omega_of(j)) for j in range(self.B)]
        return [o[0] for o in outs], [o[1] for o in outs]

    def _check(self, directions, omega, directions_of, omega_of, fwd_batched, rev_batched,
               problem=None):
        """Batched probe vs a loop of single probes; asserts the per-key batchedness and the values."""
        problem = self.jp if problem is None else problem
        forward, reverse = probe(problem, directions, omega)
        fwd_loop, rev_loop = self._loop(problem, directions_of, omega_of)
        self.assertEqual(set(forward), set(fwd_loop[0]))
        for mu in forward:
            for family, got, ref, batched in (
                    ('forward', forward[mu], [f[mu] for f in fwd_loop], fwd_batched(mu)),
                    ('reverse', reverse[mu], [r[mu] for r in rev_loop], rev_batched(mu))):
                got = np.asarray(got)
                with self.subTest(family=family, mu=mu):
                    if batched:
                        self.assertEqual((got.ndim, got.shape[0]), (2, self.B), f'{family}{mu}: {got.shape}')
                        for j in range(self.B):
                            np.testing.assert_allclose(got[j], np.asarray(ref[j]), rtol=1e-12, atol=1e-13)
                    else:                                   # single: one vector shared by the batch
                        self.assertEqual(got.ndim, 1, f'{family}{mu}: {got.shape}')
                        for j in range(self.B):
                            np.testing.assert_allclose(got, np.asarray(ref[j]), rtol=1e-12, atol=1e-13)
        return forward, reverse

    def test_batched_directions_shared_omega(self):
        # Direction-only batching: forward[(0,)] = q(theta0) AND the gradient reverse[(0,)] stay single
        # (neither depends on the batched direction); every other key is batched.
        self._check([(self.V, 2)], self.om, lambda j: [(self.V[j], 2)], lambda j: self.om,
                    fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: mu[0] >= 1)

    def test_batched_omega_shared_direction(self):
        # omega-only batching: the forward chain is single, the whole adjoint chain is batched.
        self._check([(self.b, 2)], self.OM, lambda j: [(self.b, 2)], lambda j: self.OM[j],
                    fwd_batched=lambda mu: False, rev_batched=lambda mu: True)

    def test_both_batched(self):
        self._check([(self.V, 2)], self.OM, lambda j: [(self.V[j], 2)], lambda j: self.OM[j],
                    fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: True)

    def test_mixed_request_is_batched_per_key(self):
        # ((V, 2), (b, 1)) with V batched, b single: every key with power 0 on the batched axis is single,
        # forward and reverse -- the returned dicts are ragged across keys, by design.
        self._check([(self.V, 2), (self.b, 1)], self.om,
                    lambda j: [(self.V[j], 2), (self.b, 1)], lambda j: self.om,
                    fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: mu[0] >= 1)

    def test_no_directions_batched_omega_is_the_gradient_sketch(self):
        # B gradients from ONE adjoint solve: the reduction-stage use (probe(problem, [], OMEGAS)).
        forward, reverse = probe(self.jp, [], self.OM)
        self.assertEqual(np.asarray(forward[()]).shape, (2,))          # q(theta0), shared
        self.assertEqual(np.asarray(reverse[()]).shape, (self.B, 2))   # one gradient per omega
        for j in range(self.B):
            _, r_j = probe(self.jp, [], self.OM[j])
            np.testing.assert_allclose(np.asarray(reverse[()][j]), np.asarray(r_j[()]), rtol=1e-12, atol=1e-13)

    def test_matches_numpy_reference_batched_path(self):
        # Two hooks, one driver, the same batched inputs: the numpy reference (loop-and-stack) is the
        # oracle for the JAX hook's vmapped kernels, at every order and key.
        f_ref, r_ref = probe(self.toy, [(np.asarray(self.V), 2), (np.asarray(self.b), 1)], np.asarray(self.OM))
        f_jax, r_jax = probe(self.jp, [(self.V, 2), (self.b, 1)], self.OM)
        for mu in f_ref:
            with self.subTest(mu=mu):
                self.assertEqual(np.shape(f_ref[mu]), np.shape(f_jax[mu]))
                self.assertEqual(np.shape(r_ref[mu]), np.shape(r_jax[mu]))
                np.testing.assert_allclose(np.asarray(f_jax[mu]), f_ref[mu], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(np.asarray(r_jax[mu]), r_ref[mu], rtol=1e-10, atol=1e-12)

    def test_batch_of_one_is_still_a_batch(self):
        forward, reverse = probe(self.jp, [(self.V[:1], 2)], self.om)     # 2-D is always a batch
        self.assertEqual(np.asarray(forward[(1,)]).shape, (1, 2))
        self.assertEqual(np.asarray(reverse[(1,)]).shape, (1, 2))
        self.assertEqual(np.asarray(forward[(0,)]).shape, (2,))

    def test_batch_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            probe(self.jp, [(self.V, 1)], jnp.zeros((self.B + 1, 2)))

    def test_more_than_two_dims_raises(self):
        with self.assertRaises(ValueError):
            probe(self.jp, [(jnp.zeros((2, 2, 2)), 1)], self.om)

    def test_batched_after_refreeze(self):
        # The batched kernels take the point as a traced argument, so a refrozen problem batches
        # correctly (the failure mode of an outer jit closing over the problem: stale point). Uses
        # the shared instance and moves it back afterwards (a fresh instance would recompile).
        rng = np.random.default_rng(5)
        theta0, u0 = self.jp.theta0, self.jp.u0
        self.jp.refreeze(jnp.asarray(rng.standard_normal(2)), jnp.asarray(rng.standard_normal(3)))
        try:
            self._check([(self.V, 2)], self.OM, lambda j: [(self.V[j], 2)], lambda j: self.OM[j],
                        fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: True)
        finally:
            self.jp.refreeze(theta0, u0)

    def test_custom_solvers_keep_the_single_vector_contract(self):
        # A user's solver sees one right-hand side at a time even in a batched probe (the loop
        # fallback). Installed on the shared instance and removed afterwards.
        lu_solve = self.jp._lu_solve
        def single_only(trans):
            def solver(rhs):
                assert jnp.ndim(rhs) == 1, 'custom solver must be handed single vectors'
                return lu_solve(rhs, trans=trans)
            return solver
        self.jp._forward_solver, self.jp._adjoint_solver = single_only(0), single_only(1)
        try:
            self._check([(self.V, 2)], self.OM, lambda j: [(self.V[j], 2)], lambda j: self.OM[j],
                        fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: True)
        finally:
            self.jp._forward_solver = self.jp._adjoint_solver = None

    def test_solve_counts_unchanged_by_batching(self):
        # One batched solve per lattice node, regardless of B -- the efficiency invariant.
        problem = _CountingProblem(self.jp)
        probe(problem, [(self.V, 2), (self.b, 1)], self.OM)
        L = 3 * 2
        self.assertEqual(problem.n_forward, L - 1)
        self.assertEqual(problem.n_adjoint, L)

    def test_unbatched_requests_never_touch_the_batched_kernel(self):
        # The single path is the existing kernel, untouched: an unbatched probe must not dispatch to
        # the vmapped twin (which would change its numerics at round-off and its compile behaviour).
        import implicit_probing.jax as hook
        with mock.patch.object(hook, '_term_value_batched',
                               side_effect=AssertionError('batched kernel called for an unbatched request')):
            probe(self.jp, [(self.b, 2)], self.om)

    def test_adjointness_holds_per_member(self):
        # reverse[mu] . d_k == omega . forward[mu + e_k], member by member (the helper takes single
        # vectors, so a batched result is sliced before pairing).
        forward, reverse = probe(self.jp, [(self.V, 2), (self.b, 1)], self.OM)
        for j in range(self.B):
            f_j = {mu: (v[j] if np.ndim(v) == 2 else v) for mu, v in forward.items()}
            r_j = {mu: (v[j] if np.ndim(v) == 2 else v) for mu, v in reverse.items()}
            err = validation.reverse_forward_adjointness(
                f_j, r_j, [(self.V[j], 2), (self.b, 1)], self.OM[j],
                pair_input=lambda rev, d: float(np.dot(np.asarray(rev), np.asarray(d))),
                pair_output=lambda om, fwd: float(np.dot(np.asarray(om), np.asarray(fwd))))
            self.assertLess(err, 1e-9, f'member {j}: max adjointness rel err {err:.2e}')

    def test_composed_problem_batches_through_matrix_operators(self):
        # f = W o q o C with MatrixOperators: batched input-space directions and batched reduced-output
        # functionals flow through C / W^T / W / C^T with a leading batch axis, member-for-member.
        from implicit_probing.composition import ComposedProblem, MatrixOperator
        rng = np.random.default_rng(2)
        C = jnp.asarray(rng.standard_normal((2, 3)))             # features R^3 -> theta R^2
        W = jnp.asarray(rng.standard_normal((1, 2)))             # obs R^2 -> reduced R^1
        composed = ComposedProblem(self.jp, MatrixOperator(C), MatrixOperator(W))
        X = jnp.asarray(rng.standard_normal((self.B, 3)))
        OMz = jnp.asarray(rng.standard_normal((self.B, 1)))
        self._check([(X, 2)], OMz, lambda j: [(X[j], 2)], lambda j: OMz[j],
                    fwd_batched=lambda mu: mu[0] >= 1, rev_batched=lambda mu: True, problem=composed)
        _, reverse = probe(composed, [], OMz)                       # the gradient sketch, composed
        self.assertEqual(np.asarray(reverse[()]).shape, (self.B, 3))
