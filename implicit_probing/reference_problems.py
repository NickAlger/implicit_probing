# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Toy implicitly-defined maps with exact derivatives, for testing the probing algorithms.

A reference problem supplies everything the derivative-probing driver (Algorithm 2, ``driver.py``)
asks of a problem:

- ``solve_state(theta)``           -> solve the state equation ``R(theta, u) = 0`` for ``u``;
- ``A()`` / ``solve_A`` / ``solve_A_adjoint`` -> the linearized state operator ``A = d_u R`` at the
  expansion point and its adjoint, with linear solves;
- ``partial_R`` / ``partial_Q``    -> directional mixed partials ``d_theta^a d_u^b R`` and
  ``d_theta^a d_u^b Q``, contracted against given theta-direction vectors and incremental-state
  vectors (the workhorse the driver composes into probes);
- ``q(theta)``                     -> the end-to-end map ``q = Q(theta, u(theta))``, used to build an
  *independent* finite-difference ground truth for the probes.

The output functional ``omega`` is NOT held by the problem -- it is a per-probe argument to ``probe``
(and to ``assemble_partial_sum``), since it is a choice of quantity of interest, not a property of the
map.

Everything here is polynomial, so the partials are exact at all orders (and vanish above the total
degree). A scalar polynomial is stored in the symmetric Taylor-coefficient form

    P(w) = sum_{m=0}^{d} (1/m!) T_m(w, ..., w),      T_m symmetric of order m,

for which the directional derivative is a single exact tensor contraction

    D^k P(w)(delta_1, ..., delta_k) = sum_{m>=k} (1/(m-k)!) T_m(w^{m-k}, delta_1, ..., delta_k).

This is the homogenization view: choosing each ``delta`` in the theta-block or the u-block of
``w = (theta, u)`` yields any mixed partial ``d_theta^a d_u^b`` directly, no differentiation step.
"""
import itertools
import math
import typing as typ

import numpy as np
from numpy.typing import NDArray

from implicit_probing.driver import OMEGA, PartialTerm
from implicit_probing import validation

__all__ = [
    'Polynomial',
    'ImplicitPolynomialProblem',
    'make_toy_problem',
    'forward_probe_by_finite_difference',
]


# ----------------------------------------------------------------------------------------------------
# Symmetric-coefficient-tensor polynomials (exact directional derivatives)
# ----------------------------------------------------------------------------------------------------

class Polynomial:
    """A vector-valued polynomial ``P: R^in_dim -> R^out_dim`` in symmetric Taylor-coefficient form.

    ``P(w)[o] = sum_{m=0}^{degree} (1/m!) sum_{i_1..i_m} coeffs[m][o, i_1, ..., i_m] w[i_1]...w[i_m]``,
    with each ``coeffs[m]`` symmetric in its ``m`` input axes. Directional derivatives of any order are
    exact tensor contractions (``derivative``), which is what makes this a trustworthy oracle for code
    that computes derivatives.
    """
    __slots__ = ('coeffs', 'out_dim', 'in_dim', 'degree')

    def __init__(
            self,
            coeffs: typ.Sequence[NDArray],   # len=degree+1, coeffs[m].shape == (out_dim,) + (in_dim,)*m
    ):
        if len(coeffs) == 0:
            raise ValueError('coeffs must contain at least the constant (order-0) tensor')
        out_dim = coeffs[0].shape[0]
        in_dim = coeffs[1].shape[1] if len(coeffs) > 1 else 0
        for m, T in enumerate(coeffs):
            expected = (out_dim,) + (in_dim,) * m
            if T.shape != expected:
                raise ValueError(f'coeffs[{m}] has shape {T.shape}, expected {expected}')
        self.coeffs = [np.asarray(T, dtype=float) for T in coeffs]
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.degree = len(coeffs) - 1

    def __call__(
            self,
            w: NDArray,                      # shape (in_dim,)
    ) -> NDArray:                            # shape (out_dim,)
        return self.derivative(w, ())

    def derivative(
            self,
            w:          NDArray,             # shape (in_dim,); evaluation point
            directions: typ.Sequence[NDArray],  # k vectors, each shape (in_dim,)
    ) -> NDArray:                            # shape (out_dim,); D^k P(w)(directions...)
        k = len(directions)
        result = np.zeros(self.out_dim)
        for m in range(k, self.degree + 1):
            term = self.coeffs[m]                                    # (out_dim,) + (in_dim,)*m
            for d in directions:                                     # contract k axes with the directions
                term = np.tensordot(term, d, axes=([term.ndim - 1], [0]))
            for _ in range(m - k):                                   # contract the rest with w
                term = np.tensordot(term, w, axes=([term.ndim - 1], [0]))
            result = result + term / math.factorial(m - k)
        return result

    def jacobian(
            self,
            w: NDArray,                      # shape (in_dim,)
    ) -> NDArray:                            # shape (out_dim, in_dim); the gradient/Jacobian d P / d w
        J = np.zeros((self.out_dim, self.in_dim))
        for m in range(1, self.degree + 1):
            term = self.coeffs[m]                                    # (out_dim,) + (in_dim,)*m
            for _ in range(m - 1):
                term = np.tensordot(term, w, axes=([term.ndim - 1], [0]))  # leaves (out_dim, in_dim)
            J = J + term / math.factorial(m - 1)
        return J

    def derivative_open_slot(
            self,
            w:          NDArray,             # shape (in_dim,)
            directions: typ.Sequence[NDArray],  # k filled vectors, each shape (in_dim,)
    ) -> NDArray:                            # shape (out_dim, in_dim); one derivative slot left free
        """``D^{k+1} P(w)(directions..., e_i)`` for each basis ``e_i`` -- the partial with one open slot.

        The free column index ranges over all of ``w``; the caller slices out the theta- or u-block it
        wants open. (``derivative_open_slot(w, ()) == jacobian(w)``.)
        """
        k = len(directions)
        result = np.zeros((self.out_dim, self.in_dim))
        for m in range(k + 1, self.degree + 1):
            term = self.coeffs[m]                                    # (out_dim,) + (in_dim,)*m
            for d in directions:                                     # contract the k filled directions
                term = np.tensordot(term, d, axes=([term.ndim - 1], [0]))
            for _ in range(m - k - 1):                               # contract all but one of the rest with w
                term = np.tensordot(term, w, axes=([term.ndim - 1], [0]))
            result = result + term / math.factorial(m - k - 1)       # term shape (out_dim, in_dim)
        return result


def _symmetrize_input_axes(
        T: NDArray,                          # shape (out_dim,) + (in_dim,)*m
        m: int,                              # number of trailing input axes to symmetrize over
) -> NDArray:
    """Average ``T`` over all permutations of its ``m`` input axes (axis 0, the output, is fixed)."""
    if m <= 1:
        return T
    input_axes = range(1, 1 + m)
    acc = np.zeros_like(T)
    for perm in itertools.permutations(input_axes):
        acc += np.transpose(T, (0,) + perm)
    return acc / math.factorial(m)


def _random_symmetric(
        out_dim: int,
        in_dim:  int,
        m:       int,                        # tensor order (number of input axes)
        rng:     np.random.Generator,
        scale:   float,
) -> NDArray:                                # shape (out_dim,) + (in_dim,)*m, symmetric in the input axes
    if m == 0:
        return scale * rng.standard_normal((out_dim,))
    T = scale * rng.standard_normal((out_dim,) + (in_dim,) * m)
    return _symmetrize_input_axes(T, m)


# ----------------------------------------------------------------------------------------------------
# An implicitly-defined polynomial map  q(theta) = Q(theta, u(theta)),  R(theta, u) = 0
# ----------------------------------------------------------------------------------------------------

class ImplicitPolynomialProblem:
    """A toy ``q(theta) = Q(theta, u(theta))`` with ``u`` defined implicitly by ``R(theta, u) = 0``.

    ``R`` and ``Q`` are polynomials in the stacked variable ``w = (theta, u)`` of dimension
    ``p + n_u``. The expansion point is ``theta0``; ``u0 = u(theta0)`` is found by Newton's method and
    cached. See the module docstring for the interface this exposes to the probing driver.
    """

    def __init__(
            self,
            R:      Polynomial,              # residual; out_dim=n_u, in_dim=p+n_u
            Q:      Polynomial,              # output map; out_dim=n_q, in_dim=p+n_u
            theta0: NDArray,                 # shape (p,); expansion point
            p:      int,                     # parameter (theta) dimension
            n_u:    int,                     # state (u) dimension
            n_q:    int,                     # output (q) dimension
    ):
        if R.in_dim != p + n_u or R.out_dim != n_u:
            raise ValueError(f'R must map R^{p + n_u} -> R^{n_u}, got R^{R.in_dim} -> R^{R.out_dim}')
        if Q.in_dim != p + n_u or Q.out_dim != n_q:
            raise ValueError(f'Q must map R^{p + n_u} -> R^{n_q}, got R^{Q.in_dim} -> R^{Q.out_dim}')
        if theta0.shape != (p,):
            raise ValueError('theta0 must have shape (p,)')
        self.R = R
        self.Q = Q
        self.theta0 = np.asarray(theta0, dtype=float)
        self.p = p
        self.n_u = n_u
        self.n_q = n_q
        self._u0: typ.Optional[NDArray] = None

    # --- stacking and lifting between (theta, u) and the polynomial's variable w ---

    def stack(self, theta: NDArray, u: NDArray) -> NDArray:      # (p,), (n_u,) -> (p+n_u,)
        return np.concatenate([theta, u])

    def lift_theta(self, d: NDArray) -> NDArray:                 # (p,)   -> (p+n_u,), zeros in the u-block
        return np.concatenate([d, np.zeros(self.n_u)])

    def lift_u(self, d: NDArray) -> NDArray:                     # (n_u,) -> (p+n_u,), zeros in the theta-block
        return np.concatenate([np.zeros(self.p), d])

    # --- state equation and its linearization ---

    def residual(self, theta: NDArray, u: NDArray) -> NDArray:   # -> (n_u,)
        return self.R(self.stack(theta, u))

    def state_jacobian(self, theta: NDArray, u: NDArray) -> NDArray:  # -> (n_u, n_u); d_u R
        return self.R.jacobian(self.stack(theta, u))[:, self.p:]

    def solve_state(
            self,
            theta:   typ.Optional[NDArray] = None,   # defaults to theta0
            tol:     float = 1e-14,
            maxiter: int = 100,
    ) -> NDArray:                                    # -> (n_u,)
        """Newton solve of ``R(theta, u) = 0`` for ``u`` (from ``u = 0``)."""
        theta = self.theta0 if theta is None else theta
        u = np.zeros(self.n_u)
        for _ in range(maxiter):
            r = self.residual(theta, u)
            if np.linalg.norm(r) < tol:
                return u
            u = u - np.linalg.solve(self.state_jacobian(theta, u), r)
        raise RuntimeError(f'state solve did not converge (residual {np.linalg.norm(r):.2e})')

    @property
    def u0(self) -> NDArray:                          # cached u(theta0)
        if self._u0 is None:
            self._u0 = self.solve_state(self.theta0)
        return self._u0

    @property
    def w0(self) -> NDArray:                          # the expansion point in stacked coordinates
        return self.stack(self.theta0, self.u0)

    def A(self) -> NDArray:                            # the linearized state operator A = d_u R at (theta0, u0)
        return self.state_jacobian(self.theta0, self.u0)

    def solve_A(self, b: NDArray) -> NDArray:          # A u_hat = b
        return np.linalg.solve(self.A(), b)

    def solve_A_adjoint(self, c: NDArray) -> NDArray:  # A^T v_hat = c
        return np.linalg.solve(self.A().T, c)

    # --- the end-to-end map (for independent finite-difference ground truth) ---

    def q(self, theta: NDArray) -> NDArray:            # -> (n_q,)
        return self.Q(self.stack(theta, self.solve_state(theta)))

    # --- directional mixed partials the probing driver consumes ---

    def partial_R(
            self,
            theta_dirs: typ.Sequence[NDArray],   # a theta-direction vectors, each shape (p,)
            u_vecs:     typ.Sequence[NDArray],   # b incremental-state vectors, each shape (n_u,)
    ) -> NDArray:                                # -> (n_u,);  d_theta^a d_u^b R at (theta0, u0)
        dirs = [self.lift_theta(d) for d in theta_dirs] + [self.lift_u(v) for v in u_vecs]
        return self.R.derivative(self.w0, dirs)

    def partial_Q(
            self,
            theta_dirs: typ.Sequence[NDArray],   # a theta-direction vectors, each shape (p,)
            u_vecs:     typ.Sequence[NDArray],   # b incremental-state vectors, each shape (n_u,)
    ) -> NDArray:                                # -> (n_q,);  d_theta^a d_u^b Q at (theta0, u0)
        dirs = [self.lift_theta(d) for d in theta_dirs] + [self.lift_u(v) for v in u_vecs]
        return self.Q.derivative(self.w0, dirs)

    # --- the driver.ImplicitProblem interface (a reference implementation of the hook) ---

    def solve_operator(self, b: NDArray) -> NDArray:          # A x = b
        return self.solve_A(b)

    def solve_operator_adjoint(self, c: NDArray) -> NDArray:  # A^T x = c
        return self.solve_A_adjoint(c)

    def assemble_partial_sum(
            self,
            terms: typ.Sequence[PartialTerm],
            omega: typ.Optional[NDArray],        # (n_q,) output functional; resolves OMEGA pairings
    ) -> NDArray:                                # -> the assembled sum (shape depends on the terms)
        """Assemble ``sum_i terms[i]`` for the polynomial problem (one numpy contraction per term).

        For FEniCS this is where one would build a single combined form and assemble once; the
        polynomial just loops and adds (the interface is what is being demonstrated, not a speedup).
        """
        result = None
        for t in terms:
            F = self.R if t.function == 'R' else self.Q
            # theta_dirs / u_vecs are (vector, multiplicity) pairs; the dense symmetric contraction
            # can't use the multiplicity, so just expand it back to a flat list of direction vectors.
            dirs = ([self.lift_theta(d) for d, m in t.theta_dirs for _ in range(m)]
                    + [self.lift_u(v) for v, m in t.u_vecs for _ in range(m)])
            if t.open_slot is None:
                contribution = t.coefficient * F.derivative(self.w0, dirs)        # (out_dim,)
            else:
                G = F.derivative_open_slot(self.w0, dirs)                          # (out_dim, in_dim)
                block = G[:, :self.p] if t.open_slot == 'theta' else G[:, self.p:]  # (out_dim, slot_dim)
                pairing = omega if t.pairing is OMEGA else t.pairing               # (out_dim,) covector
                contribution = t.coefficient * (pairing @ block)                   # (slot_dim,)
            result = contribution if result is None else result + contribution
        return result


def make_toy_problem(
        seed:   int = 0,
        p:      int = 2,     # theta dimension (>= 2 so distinct directions exist for asymmetric probes)
        n_u:    int = 3,     # state dimension (the "3x3 system")
        n_q:    int = 2,     # output dimension
        degree: int = 3,     # total polynomial degree in (theta, u)
) -> ImplicitPolynomialProblem:
    """A deterministic toy problem: total-degree-``degree`` ``R`` and ``Q`` with random small coefficients.

    The residual ``R`` has a dominant, well-conditioned linear-in-``u`` block (so ``d_u R`` is
    invertible and the root stays near 0), and small higher-order coefficients that make the
    coefficients of ``u``, ``u^2`` depend on ``theta`` (the "theta-dependent coefficients" structure).
    With ``degree = 3`` every mixed partial up to order 3 is generically nonzero and order-4+ vanish.
    """
    rng = np.random.default_rng(seed)
    n = p + n_u

    # R: residual (out_dim = n_u), with a dominant u-linear block.
    R_coeffs = [0.1 * rng.standard_normal((n_u,))]                          # T0 (small => root near 0)
    T1 = 0.3 * rng.standard_normal((n_u, n))                                # T1
    T1[:, p:] = 3.0 * np.eye(n_u) + 0.2 * rng.standard_normal((n_u, n_u))   # dominant u-block => A invertible
    R_coeffs.append(T1)
    R_higher_scales = [0.15, 0.05, 0.02, 0.01]
    for m in range(2, degree + 1):
        R_coeffs.append(_random_symmetric(n_u, n, m, rng, R_higher_scales[m - 2]))

    # Q: output map (out_dim = n_q), no conditioning constraint.
    Q_scales = [0.5, 0.5, 0.3, 0.2, 0.1]
    Q_coeffs = [Q_scales[0] * rng.standard_normal((n_q,))]
    for m in range(1, degree + 1):
        Q_coeffs.append(_random_symmetric(n_q, n, m, rng, Q_scales[m]))

    return ImplicitPolynomialProblem(
        R=Polynomial(R_coeffs), Q=Polynomial(Q_coeffs),
        theta0=np.zeros(p), p=p, n_u=n_u, n_q=n_q,
    )


# ----------------------------------------------------------------------------------------------------
# Independent finite-difference ground truth for a forward probe (any probing symmetry)
# ----------------------------------------------------------------------------------------------------

def forward_probe_by_finite_difference(
        problem:          ImplicitPolynomialProblem,
        direction_orders: typ.Sequence[typ.Tuple[NDArray, int]],  # the multiset {d_k^{m_k}}: (vector (p,), order)
        h:                float = 1e-2,
        richardson:       bool = True,
) -> NDArray:                                                     # -> (n_q,)
    """Independent ground truth for the forward probe ``D^j q(theta0)`` at the toy's expansion point.

    A thin numpy convenience over :func:`implicit_probing.validation.forward_probe_by_finite_difference`
    (the vector-agnostic engine), fixing the parameter perturbation to plain numpy and feeding the toy's
    end-to-end map ``q`` and expansion point ``theta0``. See that function for the central-difference /
    Richardson details and the symmetry handling. Finite differences are a *test* of the probes -- the
    probing driver computes them exactly and far more cheaply.
    """
    return validation.forward_probe_by_finite_difference(
        problem.q, problem.theta0, direction_orders, h=h, richardson=richardson)
