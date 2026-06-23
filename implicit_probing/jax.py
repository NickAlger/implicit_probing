# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""JAX implementation of the ``ImplicitProblem`` interface (Algorithm 2 hook).

``JaxImplicitProblem`` is **frozen** at a user-supplied expansion point ``(theta0, u0)``: the user
solves the nonlinear state equation ``R(theta0, u) = 0`` themselves (by whatever means) and hands the
solved arrays to this class, which then provides derivative probes at that point. The class never does
a nonlinear solve. It assembles the linearized state operator ``A = d_u R`` once and LU-factorizes it
(reused for every forward and adjoint solve, the latter as a transpose solve).

The whole hook rests on one helper, ``_directional_partial``: every ``PartialTerm`` the driver requests
is a directional mixed partial ``d_theta^a d_u^b {R, Q}`` contracted against the supplied direction
vectors, optionally with one open slot (paired with ``omega`` / an incremental adjoint). Those partials
are taken by **Taylor-mode automatic differentiation** (``jax.experimental.jet``): a direction of
multiplicity ``m`` costs a *single order-``m`` jet*, not ``m`` nested ``jvp``s -- the exact payoff the
``(vector, multiplicity)`` encoding in ``PartialTerm`` was designed to enable (cost ~``O(j^2)`` for an
order-``j`` directional derivative rather than the ``O(2^j)`` of nested forward mode). Distinct
directions are handled by nesting one jet per direction; the open slot is one more (reverse-mode)
derivative of the pairing-contracted partial.

The user writes ``R`` and ``Q`` as ordinary JAX callables of ``(theta, u)``; this module wires them to
the driver. It works in single or double precision, but high-order probes want ``float64`` -- enable it
with ``jax.config.update("jax_enable_x64", True)`` before building the problem.

This module imports ``jax`` and is therefore an OPTIONAL part of implicit_probing (the core package
needs only numpy); install it with the ``jax`` extra.
"""
import functools
import typing as typ

import jax
import jax.numpy as jnp
from jax.experimental import jet

from implicit_probing.driver import OMEGA

__all__ = ['JaxImplicitProblem']


def _deriv_along(fn, d, order):
    """Wrap ``fn`` into ``x -> D^order fn(x)[d, ..., d]`` (order-``order`` directional derivative along ``d``).

    Uses one Taylor-mode jet of length ``order``: feeding the input series ``(d, 0, ..., 0)`` makes the
    perturbed input ``x + t d``, and ``jet`` returns the derivatives directly (no ``1/k!``), so the
    last series entry is the ``order``-th directional derivative. The result is again a function of the
    base point, so these wrappers compose: nesting them differentiates along several directions.
    """
    tail = [jnp.zeros_like(d)] * (order - 1)

    def out(x):
        _, series = jet.jet(fn, (x,), ([d, *tail],))   # series[k] = D^{k+1} fn(x)[d^{k+1}]
        return series[order - 1]

    return out


@functools.partial(jax.jit, static_argnames=('F', 'mults', 'open_slot', 'p'))
def _term_value(
        w,                                  # stacked point (theta, u), shape (p + n_u,)
        dir_vecs,                           # tuple of lifted direction vectors, each (p + n_u,) -- TRACED
        pairing,                            # output-space covector, or None when open_slot is None -- TRACED
        *,
        F,                                  # callable w -> output vector (the R- or Q-view); STATIC
        mults,                              # tuple of per-direction multiplicities; STATIC
        open_slot,                          # None | 'theta' | 'u'; STATIC
        p,                                  # theta dimension (for slicing the open slot); STATIC
):
    """One ``PartialTerm``'s value (sans its integer coefficient): a directional mixed partial of ``F``.

    Mixed partials commute, so each distinct direction is folded in independently (order = its
    multiplicity) via one Taylor-mode jet; an empty ``dir_vecs`` is the order-0 partial ``F(w)``. With
    an open slot, one more (reverse-mode) derivative of ``pairing . partial`` w.r.t. the whole point is
    taken and the theta- or u-block sliced out (matching the toy's ``derivative_open_slot`` + slice).

    Jitted with the term's *structure* (``F``, ``mults``, ``open_slot``, ``p``) static and the *vectors*
    (``w``, ``dir_vecs``, ``pairing``) traced, so **one compiled kernel serves every lattice node and
    direction value sharing that structure** -- the reuse that makes high-order probing affordable
    (otherwise XLA recompiles per distinct direction/incremental vector).
    """
    def partial(x):
        fn = F
        for d, mult in zip(dir_vecs, mults):
            fn = _deriv_along(fn, d, mult)
        return fn(x)

    if open_slot is None:
        return partial(w)                                          # (out_dim,)
    grad_w = jax.grad(lambda x: jnp.dot(pairing, partial(x)))(w)   # covector over all of w
    return grad_w[:p] if open_slot == 'theta' else grad_w[p:]      # slice the open slot


class JaxImplicitProblem:
    """``ImplicitProblem`` for a JAX map ``q(theta) = Q(theta, u(theta))``, frozen at ``(theta0, u0)``.

    Parameters
    ----------
    R : callable
        State residual ``R(theta, u) -> array`` of shape ``(n_u,)``; ``u0`` must solve ``R(theta0, u) = 0``.
    Q : callable
        Output/observation map ``Q(theta, u) -> array`` of shape ``(n_q,)``.
    theta0, u0 : array
        The frozen expansion point; ``theta0`` has shape ``(p,)`` and ``u0`` shape ``(n_u,)``.
    forward_solver, adjoint_solver : callable | None
        Optional custom solvers, each mapping a right-hand side to a solution. If omitted, a single
        reused LU factorization of ``A = d_u R`` is used (adjoint via its transpose solve).
    """

    def __init__(self, R, Q, theta0, u0, *, forward_solver=None, adjoint_solver=None):
        self.R = R
        self.Q = Q
        self.theta0 = jnp.asarray(theta0)
        self.u0 = jnp.asarray(u0)
        if self.theta0.ndim != 1 or self.u0.ndim != 1:
            raise ValueError('theta0 and u0 must be 1-D arrays')
        self.p = int(self.theta0.shape[0])
        self.n_u = int(self.u0.shape[0])
        self.w0 = jnp.concatenate([self.theta0, self.u0])     # the expansion point in stacked coords

        # Stacked-variable views F(w) = F(theta, u): a single argument so jet/grad differentiate the
        # theta- and u-slots uniformly (a theta-direction is lifted to (d, 0), a u-direction to (0, v)).
        self._R_w = lambda w: R(w[:self.p], w[self.p:])
        self._Q_w = lambda w: Q(w[:self.p], w[self.p:])

        # A = d_u R at (theta0, u0); assembled and LU-factorized once, reused for every solve.
        self.A = jax.jacfwd(lambda u: R(self.theta0, u))(self.u0)   # (n_u, n_u)
        if self.A.shape != (self.n_u, self.n_u):
            raise ValueError(f'd_u R has shape {self.A.shape}, expected the square ({self.n_u}, {self.n_u}); '
                             'R must map to the state space (R.out_dim == u dimension)')
        self._forward_solver = forward_solver
        self._adjoint_solver = adjoint_solver
        self._lu = jax.scipy.linalg.lu_factor(self.A) if (forward_solver is None or adjoint_solver is None) else None

    # --- ImplicitProblem interface ---

    def solve_operator(self, b):
        """Solve ``A x = b`` for the incremental state ``x`` (``A = d_u R`` at the expansion point)."""
        if self._forward_solver is not None:
            return self._forward_solver(b)
        return jax.scipy.linalg.lu_solve(self._lu, b)

    def solve_operator_adjoint(self, c):
        """Solve ``A* x = c`` for the incremental adjoint ``x`` (transpose of the same factorization)."""
        if self._adjoint_solver is not None:
            return self._adjoint_solver(c)
        return jax.scipy.linalg.lu_solve(self._lu, c, trans=1)

    def assemble_partial_sum(self, terms, omega):
        """Assemble ``sum_i terms[i]``, resolving ``OMEGA`` pairings to ``omega`` (one jet per term)."""
        result = None
        for t in terms:
            F = self._R_w if t.function == 'R' else self._Q_w
            # Lift every direction into the stacked (theta, u) space -- so the kernel never sees the
            # theta/u split -- then sort by multiplicity. The driver already emits each block in
            # canonical order; this re-sort is the part that CANNOT be hoisted there, because unifying
            # the theta and u blocks is specific to a stacked-variable AD backend (jet differentiates
            # along whole-w directions; UFL, say, cannot merge them). The payoff: structurally
            # equivalent partials -- same function and same multiset of multiplicities, however the
            # directions split across theta/u or are ordered -- hit ONE compiled jet kernel.
            pairs = ([(self._lift_theta(d), mult) for d, mult in t.theta_dirs]
                     + [(self._lift_u(v), mult) for v, mult in t.u_vecs])
            pairs.sort(key=lambda vec_mult: vec_mult[1], reverse=True)
            dir_vecs = tuple(vec for vec, mult in pairs)
            mults = tuple(mult for vec, mult in pairs)
            pairing = None if t.open_slot is None else (omega if t.pairing is OMEGA else t.pairing)
            value = _term_value(self.w0, dir_vecs, pairing,
                                F=F, mults=mults, open_slot=t.open_slot, p=self.p)
            contribution = t.coefficient * value
            result = contribution if result is None else result + contribution
        return result

    # --- internals: lift a theta-/u-direction into the stacked (theta, u) space ---

    def _lift_theta(self, d):                                  # (p,)   -> (p + n_u,), zeros in the u-block
        return jnp.concatenate([d, jnp.zeros(self.n_u, dtype=self.w0.dtype)])

    def _lift_u(self, v):                                      # (n_u,) -> (p + n_u,), zeros in the theta-block
        return jnp.concatenate([jnp.zeros(self.p, dtype=self.w0.dtype), v])
