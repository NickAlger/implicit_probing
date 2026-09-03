# Authors: Blake Christierson and Nick Alger
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

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import jet

from implicit_probing.batching import infer_batch_size
from implicit_probing.driver import OMEGA, probe

__all__ = ['JaxImplicitProblem', 'compiled_probe', 'lu_solver_factory', 'HostSolver']


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


def _term_value_impl(w, dir_vecs, pairing, *, F, mults, open_slot, p):
    """The shared body of ``_term_value`` (single) and ``_term_value_batched`` (vmapped); see the
    former for the contract. Never called outside those two jitted wrappers."""
    def partial(x):
        fn = F
        for d, mult in zip(dir_vecs, mults):
            fn = _deriv_along(fn, d, mult)
        return fn(x)

    if open_slot is None:
        return partial(w)                                          # (out_dim,)
    grad_w = jax.grad(lambda x: jnp.dot(pairing, partial(x)))(w)   # covector over all of w
    return grad_w[:p] if open_slot == 'theta' else grad_w[p:]      # slice the open slot


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
    return _term_value_impl(w, dir_vecs, pairing, F=F, mults=mults, open_slot=open_slot, p=p)


@functools.partial(jax.jit, static_argnames=('F', 'mults', 'open_slot', 'p', 'axes'))
def _term_value_batched(
        w,                                  # stacked point (theta, u), shape (p + n_u,) -- TRACED, SHARED
        dir_vecs,                           # tuple of lifted directions, each (p + n_u,) or (B, p + n_u) -- TRACED
        pairing,                            # covector (out,) or (B, out), or None -- TRACED
        *,
        F, mults, open_slot, p,             # the term's structure, as in ``_term_value``; STATIC
        axes,                               # (per-direction 0|None, pairing 0|None): which leaves carry
                                            #   the batch axis; STATIC (part of the cache key)
):
    """The batched twin of ``_term_value``: the same kernel ``vmap``-ed over a leading batch axis of
    the batched leaves, with the expansion point ``w`` shared (``in_axes=None``) -- so a ``refreeze``
    (new ``w``) never recompiles; only a new structure, a new batch size, or a new batched-leaf
    pattern does. ``jit`` is OUTSIDE and the statics are bound INSIDE (``vmap`` maps keyword arguments
    over axis 0, so the structure cannot pass through it). Per-leaf ``axes`` (``None`` for a shared
    vector) instead of materialized ``(B, .)`` broadcasts: no B-fold copies of shared directions, and
    XLA skips the replicated work when only the pairing is batched."""
    impl = functools.partial(_term_value_impl, F=F, mults=mults, open_slot=open_slot, p=p)
    return jax.vmap(impl, in_axes=(None, axes[0], axes[1]))(w, dir_vecs, pairing)


def _leading_batch_size(vec):
    """The hook's notion of a batched vector: a leading batch axis. ``(B, n)`` -> B; ``(n,)`` -> None;
    anything else raises (2-D is ALWAYS a batch, even B = 1)."""
    ndim = jnp.ndim(vec)
    if ndim == 1:
        return None
    if ndim == 2:
        return int(jnp.shape(vec)[0])
    raise ValueError(f'a vector must be 1-D (single) or 2-D (a batch, leading axis); got ndim={ndim}')


class JaxImplicitProblem:
    """``ImplicitProblem`` for a JAX map ``q(theta) = Q(theta, u(theta))``, frozen at ``(theta0, u0)``.

    To probe the same map at MANY expansion points (multi-point probing), reuse one instance via
    :py:meth:`refreeze` rather than constructing per point -- construction creates the closures that
    key the jit cache, so per-point instances recompile every jet kernel (and eventually exhaust XLA
    memory).

    **Batched probes.** A direction of shape ``(B, p)`` or an ``omega`` of shape ``(B, n_q)`` means B
    independent probes at this expansion point, handled in ONE ``probe`` call: the lattice is walked
    once, every linearized solve is a multi-right-hand-side solve on the shared LU, and every jet
    kernel is ``vmap``-ed over the batch. Single (1-D) inputs in the same call are broadcast -- shared
    by every batch member -- so one call can batch over directions at a fixed ``omega``, over ``omega``
    at a fixed direction, or over both. A result is batched, shape ``(B, .)``, iff its request used a
    batched input: ``forward[mu]`` iff some batched direction ``k`` has ``mu_k >= 1``; ``reverse[mu]``
    iff ``omega`` is batched or some batched direction ``k`` has ``mu_k >= 1``. So ``forward[(0, ..)]``
    (= ``q(theta0)``) is never batched, and under direction-only batching neither is the gradient
    ``reverse[(0, ..)]`` -- the returned dicts can be ragged across keys. 2-D is always a batch, even
    B = 1; batched inputs with different B raise; every distinct B (and batched-leaf pattern)
    compiles the kernels once more, so batch in chunks of a FIXED size. Design and measurements:
    ``dev/batched_probes_design.md``.

    Parameters
    ----------
    R : callable
        State residual ``R(theta, u) -> array`` of shape ``(n_u,)``; ``u0`` must solve ``R(theta0, u) = 0``.
    Q : callable
        Output/observation map ``Q(theta, u) -> array`` of shape ``(n_q,)``.
    theta0, u0 : array
        The frozen expansion point; ``theta0`` has shape ``(p,)`` and ``u0`` shape ``(n_u,)``.
    forward_solver, adjoint_solver : callable | None
        Optional custom solvers, each mapping ONE right-hand side ``(n_u,)`` to a solution -- opaque
        Python callables, bound to this expansion point (e.g. closures over a factorization computed
        elsewhere). In a batched probe they are applied per row, unless the callable carries
        ``accepts_batch = True`` and takes a ``(B, n_u)`` block whole. They cannot enter a
        :func:`compiled_probe` (the point is traced there); use ``solver_factory`` or a
        :class:`HostSolver` instead.
    solver_factory : callable | None
        ``A -> (solve, solve_adjoint)``: builds JAX-traceable single-vector solvers from the assembled
        operator ``A = d_u R`` (called at construction and at every ``refreeze``; batches are
        ``vmap``-ed unless the solver carries ``accepts_batch = True``). The JAX twin of the FEniCSx
        hook's ``ksp_factory``. Default :func:`lu_solver_factory` -- one reused LU factorization,
        adjoint via its transpose solve. Any ``jax.scipy.sparse.linalg`` solver with a preconditioner
        formed from ``A`` fits. Ignored when both custom solvers are given.
    """

    def __init__(self, R, Q, theta0, u0, *, forward_solver=None, adjoint_solver=None,
                 solver_factory=None):
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
        self._solver_factory = solver_factory if solver_factory is not None else lu_solver_factory
        self._fwd, self._adj = ((None, None) if (forward_solver is not None and adjoint_solver is not None)
                                else self._solver_factory(self.A))

    def refreeze(self, theta0, u0, *, forward_solver=None, adjoint_solver=None):
        """Move the frozen expansion point to ``(theta0, u0)`` IN PLACE, keeping every compiled kernel.

        The jet kernels (``_term_value``) are jitted with the term's *structure* static -- including
        ``F``, the R-/Q-view closures created in ``__init__`` -- and the expansion point traced. The
        jit cache is therefore keyed by those closure objects: multi-point probing (many expansion
        points) wants ONE problem instance refrozen per point. A fresh instance per point recreates
        the closures, misses the cache, and recompiles every kernel at every point -- and the
        accumulated compilations exhaust XLA executable memory after a few tens of points.

        Do NOT wrap ``probe(problem, ...)`` in an outer ``jax.jit`` that closes over the problem: the
        frozen point is then baked into the compiled program as a constant, and after ``refreeze`` the
        stale program silently returns the OLD point's jets. The hook's own kernels are immune (the
        point is a traced argument), including the batched ones.

        As in ``__init__``, the caller supplies a solved point (``u0`` solving ``R(theta0, u) = 0``);
        nothing is re-solved here, but the linearized operator ``A = d_u R`` is re-assembled and the
        solvers rebuilt from it by the ``solver_factory``. Shapes must match the original problem
        (the compiled kernels are shape-specialized). A problem built with custom solvers must be
        given fresh ones (they are specific to the frozen point); the factory path needs no
        arguments. The alternative to the refreeze loop is :func:`compiled_probe`, which takes the
        point as an argument.

        Returns ``self``, so ``probe(problem.refreeze(x, u), ...)`` reads naturally.
        """
        theta0 = jnp.asarray(theta0)
        u0 = jnp.asarray(u0)
        if theta0.shape != (self.p,) or u0.shape != (self.n_u,):
            raise ValueError(f'refreeze shapes {theta0.shape} / {u0.shape} do not match the frozen '
                             f'problem ({self.p},) / ({self.n_u},); build a new problem instead')
        if (self._forward_solver is not None or self._adjoint_solver is not None) \
                and forward_solver is None and adjoint_solver is None:
            raise ValueError('this problem was built with custom solvers, which are specific to the '
                             'frozen point; pass fresh forward_solver/adjoint_solver to refreeze')
        self.theta0 = theta0
        self.u0 = u0
        self.w0 = jnp.concatenate([theta0, u0])
        self.A = jax.jacfwd(lambda u: self.R(theta0, u))(u0)
        self._forward_solver = forward_solver
        self._adjoint_solver = adjoint_solver
        self._fwd, self._adj = ((None, None) if (forward_solver is not None and adjoint_solver is not None)
                                else self._solver_factory(self.A))
        return self

    # --- ImplicitProblem interface ---

    def solve_operator(self, b):
        """Solve ``A x = b`` for the incremental state ``x`` (``A = d_u R`` at the expansion point).
        A batched ``b`` of shape ``(B, n_u)`` is one multi-right-hand-side solve (rows convention)."""
        if self._forward_solver is not None:
            return _apply_custom(self._forward_solver, b)
        return _apply_traceable(self._fwd, b)

    def solve_operator_adjoint(self, c):
        """Solve ``A* x = c`` for the incremental adjoint ``x`` (the transpose solve of the same
        factorization); batched ``c`` of shape ``(B, n_u)`` likewise."""
        if self._adjoint_solver is not None:
            return _apply_custom(self._adjoint_solver, c)
        return _apply_traceable(self._adj, c)

    def compiled(self, powers, *, host_solver=None):
        """:func:`compiled_probe` for this problem's ``R``, ``Q`` and ``solver_factory``: the whole
        lattice walk for the lattice pattern ``powers`` as ONE jitted function of the point. A
        problem built with point-bound custom solvers cannot be compiled (pass a :class:`HostSolver`)."""
        if (self._forward_solver is not None or self._adjoint_solver is not None) and host_solver is None:
            raise ValueError('this problem uses point-bound forward_solver/adjoint_solver callables, '
                             'which cannot enter a compiled probe (the point is traced there); build '
                             'it with solver_factory=..., or pass host_solver=HostSolver(...)')
        return compiled_probe(self.R, self.Q, powers, solver_factory=self._solver_factory,
                              host_solver=host_solver)

    def assemble_partial_sum(self, terms, omega):
        """Assemble ``sum_i terms[i]``, resolving ``OMEGA`` pairings to ``omega`` (one jet per term).

        Batched requests (a ``(B, .)`` vector among the inputs the request uses -- see the class
        docstring and ``batching.infer_batch_size``) run the vmapped kernel with shared vectors
        broadcast; an unbatched request runs the single kernel unchanged.
        """
        B = infer_batch_size(terms, omega, _leading_batch_size)
        lifts = {}                                   # id(vector) -> lifted copy, cached per request (a
        result = None                                #   direction recurs across many terms; batched
        for t in terms:                              #   lifts are B-fold, so caching matters)
            F = self._R_w if t.function == 'R' else self._Q_w
            # Lift every direction into the stacked (theta, u) space -- so the kernel never sees the
            # theta/u split -- then sort by multiplicity. The driver already emits each block in
            # canonical order; this re-sort is the part that CANNOT be hoisted there, because unifying
            # the theta and u blocks is specific to a stacked-variable AD backend (jet differentiates
            # along whole-w directions; UFL, say, cannot merge them). The payoff: structurally
            # equivalent partials -- same function and same multiset of multiplicities, however the
            # directions split across theta/u or are ordered -- hit ONE compiled jet kernel.
            pairs = ([(self._lift(d, 'theta', lifts), mult) for d, mult in t.theta_dirs]
                     + [(self._lift(v, 'u', lifts), mult) for v, mult in t.u_vecs])
            pairs.sort(key=lambda vec_mult: vec_mult[1], reverse=True)
            dir_vecs = tuple(vec for vec, mult in pairs)
            mults = tuple(mult for vec, mult in pairs)
            pairing = None if t.open_slot is None else (omega if t.pairing is OMEGA else t.pairing)
            if B is None:
                value = _term_value(self.w0, dir_vecs, pairing,
                                    F=F, mults=mults, open_slot=t.open_slot, p=self.p)
            else:
                axes = (tuple(0 if jnp.ndim(v) == 2 else None for v in dir_vecs),
                        0 if (pairing is not None and jnp.ndim(pairing) == 2) else None)
                if axes[1] is None and all(a is None for a in axes[0]):
                    # a term with no batched leaf inside a batched request: single kernel, broadcast
                    single = _term_value(self.w0, dir_vecs, pairing,
                                         F=F, mults=mults, open_slot=t.open_slot, p=self.p)
                    value = jnp.broadcast_to(single, (B,) + single.shape)
                else:
                    value = _term_value_batched(self.w0, dir_vecs, pairing, F=F, mults=mults,
                                                open_slot=t.open_slot, p=self.p, axes=axes)
            contribution = t.coefficient * value
            result = contribution if result is None else result + contribution
        return result

    # --- internals: lift a theta-/u-direction into the stacked (theta, u) space ---

    def _lift(self, vec, block, cache):
        """``(p,)``/``(n_u,)`` -> ``(p + n_u,)`` (or ``(B, .)`` -> ``(B, p + n_u)``) with zeros in the other
        block; memoized by object identity within one request."""
        key = id(vec)
        if key not in cache:
            cache[key] = self._lift_theta(vec) if block == 'theta' else self._lift_u(vec)
        return cache[key]

    def _lift_theta(self, d):                                  # (p,)   -> (p + n_u,), zeros in the u-block
        pad = jnp.zeros(jnp.shape(d)[:-1] + (self.n_u,), dtype=self.w0.dtype)
        return jnp.concatenate([d, pad], axis=-1)

    def _lift_u(self, v):                                      # (n_u,) -> (p + n_u,), zeros in the theta-block
        pad = jnp.zeros(jnp.shape(v)[:-1] + (self.p,), dtype=self.w0.dtype)
        return jnp.concatenate([pad, v], axis=-1)


def _apply_custom(solver, b):
    """Apply an opaque user-supplied ``vector -> vector`` solver: to a single right-hand side, to a
    ``(B, n_u)`` block whole if it declares ``accepts_batch = True``, else to every row (the loop
    fallback: custom solvers keep their single-vector contract)."""
    if jnp.ndim(b) == 2 and not getattr(solver, 'accepts_batch', False):
        return jnp.stack([solver(row) for row in b])
    return solver(b)


def _apply_traceable(solver, b):
    """Apply a factory-built (JAX-traceable) single-vector solver: ``vmap`` over a ``(B, n_u)`` block
    unless it declares ``accepts_batch = True`` (the LU default does, with a multi-RHS solve)."""
    if jnp.ndim(b) == 2 and not getattr(solver, 'accepts_batch', False):
        return jax.vmap(solver)(b)
    return solver(b)


def lu_solver_factory(A):
    """The default ``solver_factory``: one LU factorization of ``A``, reused; returns
    ``(solve, solve_adjoint)``, each taking a single right-hand side or (``accepts_batch``) a
    ``(B, n_u)`` block as ONE multi-right-hand-side triangular solve -- rows convention, always
    explicit: a ``(B, n_u)`` block with ``B == n_u`` is indistinguishable from a column layout by
    shape, and the wrong orientation is silently wrong."""
    lu = jax.scipy.linalg.lu_factor(A)

    def _solve(b, trans):
        if jnp.ndim(b) == 2:
            return jax.scipy.linalg.lu_solve(lu, b.T, trans=trans).T
        return jax.scipy.linalg.lu_solve(lu, b, trans=trans)

    def solve(b):
        return _solve(b, 0)

    def solve_adjoint(c):
        return _solve(c, 1)

    solve.accepts_batch = solve_adjoint.accepts_batch = True
    return solve, solve_adjoint


class HostSolver:
    """Bridge for HOST (non-JAX) solvers -- scipy, PETSc, anything that cannot be traced -- into a
    :func:`compiled_probe`: each solve becomes a ``jax.pure_callback`` that hands the concrete
    right-hand side(s) to your Python callable at execution time.

    The callables are looked up at CALL time, not at trace time, so set them per expansion point
    (the refreeze contract, restated): factorize on the host next to your nonlinear solve, then
    ``host.set(forward, adjoint)`` before calling the compiled function. Each callable maps one
    ``(n_u,)`` right-hand side to its solution; a callable with ``accepts_batch = True`` receives a
    ``(B, n_u)`` block whole (one host round-trip per lattice node instead of B). The cost is one host
    round-trip per solve and no fusion across it -- which an opaque solver never allowed anyway.
    """

    def __init__(self, forward=None, adjoint=None):
        self.forward = forward
        self.adjoint = adjoint

    def set(self, forward, adjoint):
        """Install the solvers for the current expansion point; returns ``self``."""
        self.forward = forward
        self.adjoint = adjoint
        return self

    def _bridge(self, transpose):
        def run(b_np):
            fn = self.adjoint if transpose else self.forward
            if fn is None:
                raise RuntimeError('HostSolver: call set(forward, adjoint) before the compiled probe')
            b_np = np.array(b_np, copy=True)   # a private copy: never hand the device buffer's view to host code
            if b_np.ndim == 2 and not getattr(fn, 'accepts_batch', False):
                return np.stack([np.asarray(fn(row), dtype=b_np.dtype) for row in b_np])
            return np.asarray(fn(b_np), dtype=b_np.dtype)

        def solve(b):
            return jax.pure_callback(run, jax.ShapeDtypeStruct(b.shape, b.dtype), b,
                                     vmap_method='sequential')

        solve.accepts_batch = True
        return solve


def compiled_probe(R, Q, powers, *, solver_factory=None, host_solver=None):
    """The whole lattice walk for one lattice pattern as ONE jitted function of the point:
    ``f(theta0, u0, directions, omega=None) -> (forward, reverse)``.

    ``powers`` is the pattern (the ``max_power`` per direction, e.g. ``(4,)`` or ``(2, 1)``);
    ``directions`` is a tuple of one array per power, ``(p,)`` or ``(B, p)`` for a batch; ``omega``
    is ``(n_q,)``, ``(B, n_q)`` or ``None``; the results are the same dicts ``probe`` returns, with the
    same per-key batching. Inside the program the problem is built at the TRACED point -- ``A``, its
    factorization (or the ``solver_factory``'s solvers) and every jet are one XLA program -- so
    moving the point never recompiles and can never go stale: the point is an argument. Only a new
    shape (of ``theta0``, ``u0``, ``directions`` or ``omega``, incl. the batch size) retraces, and a
    new ``f`` (new ``R``/``Q`` closures or pattern) is a new program. Build ``f`` once per pattern.

    What it buys and costs (measured, a DEQ with state dim 256, D = 64): 3.6× (J = 4), 4.7× (J = 6),
    7× (two directions) over the eager per-term kernels; the compile is one program per pattern --
    10 s at J = 4 (faster than the per-term kernels' 13 s), 56 s at J = 6, ~8 min for a (3, 3)
    pattern (XLA's passes are superlinear in program size). With ``jax_compilation_cache_dir`` set,
    a second process skips the XLA compile (10.3 s -> 4.6 s at J = 4; the rest is tracing).

    Solvers: ``solver_factory`` (traceable, built from the traced ``A`` inside the program; default
    LU) or ``host_solver`` (a :class:`HostSolver` bridging opaque host solvers via
    ``jax.pure_callback``). Point-bound ``forward_solver``/``adjoint_solver`` closures cannot be used
    here.
    """
    powers = tuple(int(p) for p in powers)
    if any(p < 1 for p in powers):
        raise ValueError(f'powers must be positive integers, got {powers}')

    @jax.jit
    def f(theta0, u0, directions, omega=None):
        if len(directions) != len(powers):
            raise ValueError(f'expected {len(powers)} direction arrays for powers {powers}, '
                             f'got {len(directions)}')
        if host_solver is not None:
            prob = JaxImplicitProblem(R, Q, theta0, u0, forward_solver=host_solver._bridge(False),
                                      adjoint_solver=host_solver._bridge(True))
        else:
            prob = JaxImplicitProblem(R, Q, theta0, u0, solver_factory=solver_factory)
        return probe(prob, list(zip(directions, powers)), omega)

    return f
