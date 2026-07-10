# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Composing an implicitly-defined map with linear input and output maps.

If ``C`` is a linear map on the input (e.g. a few features -> the full parameter field) and ``W`` a
linear map on the output (e.g. the raw observation -> a reduced observation), the composed map
``f = W o q o C`` has derivative probes that are simple linear images of ``q``'s probes. Because ``C``
and ``W`` are linear -- their own higher derivatives vanish and they pass straight through the chain
rule:

    forward:  D^j f(x0)(xhat...) = W( D^j q(theta0)(C xhat...) )
    reverse:  psi^f_beta = C^T( psi^q_beta ),   with q's reverse taken with the pulled-back functional
              W^T omega and the C-mapped directions.

The state space, the operator ``A = d_u R``, and the incremental solves are untouched -- ``C`` and
``W`` only reparameterize the input and re-observe the output. So ``ComposedProblem`` wraps an inner
``ImplicitProblem`` and, in ``assemble_partial_sum``, pre-maps the direction vectors through ``C``,
pulls ``omega`` back through ``W^T``, and post-maps the assembled result (``W`` on a forward output,
``C^T`` on a reverse output, nothing on a state right-hand side). The driver and the inner problem are
unchanged, and compositions nest.
"""
import dataclasses
import typing as typ

from implicit_probing.driver import OMEGA  # noqa: F401  (kept: documents the pairing the inner resolves)

__all__ = ['LinearOperator', 'MatrixOperator', 'ComposedProblem']


@typ.runtime_checkable
class LinearOperator(typ.Protocol):
    """A linear map with a forward action and a transpose action, both on opaque vectors.

    ``apply`` maps an input *vector* to an output vector; ``apply_transpose`` maps an output *covector*
    back to an input covector. The vectors must be whatever the inner problem produces and consumes
    (numpy arrays for the toy, PETSc vectors / DOLFINx Functions for FEniCS).
    """
    def apply(self, v: typ.Any) -> typ.Any: ...
    def apply_transpose(self, w: typ.Any) -> typ.Any: ...


class MatrixOperator:
    """A ``LinearOperator`` backed by a matrix ``M``: ``apply = M @ v``, ``apply_transpose = M.T @ v``.

    Works for any ``M`` supporting ``@`` and ``.T`` (dense or sparse numpy/scipy).
    """
    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, v):
        return self.matrix @ v

    def apply_transpose(self, w):
        return self.matrix.T @ w


class ComposedProblem:
    """Wrap an inner ``ImplicitProblem`` as ``f = W o q o C`` with linear input/output maps.

    ``probe(ComposedProblem(inner, input_map=C, output_map=W), alpha, x_directions, omega)`` probes the
    composed map: ``x_directions`` live in the input (``C``-domain) space, forward probes come back in
    the output (``W``-codomain) space, ``omega`` is a covector there, and reverse probes are covectors
    in the input space. Either map may be ``None`` (identity). The inner problem and the driver are
    unchanged; ``ComposedProblem`` is itself an ``ImplicitProblem``, so compositions nest.
    """
    def __init__(self, inner, input_map: typ.Optional[LinearOperator] = None,
                 output_map: typ.Optional[LinearOperator] = None):
        self.inner = inner
        self.input_map = input_map      # C: input space -> parameter space (or None = identity)
        self.output_map = output_map    # W: observation space -> reduced output (or None = identity)

    def solve_operator(self, b):
        return self.inner.solve_operator(b)          # the state operator A is unchanged by C, W

    def solve_operator_adjoint(self, c):
        return self.inner.solve_operator_adjoint(c)

    def assemble_partial_sum(self, terms, omega):
        # omega lives on the final (W-codomain) output; pull it back to the inner's output space.
        inner_omega = omega
        if self.output_map is not None and omega is not None:
            inner_omega = self.output_map.apply_transpose(omega)

        # pre-map the probing directions through C (cached by identity within this request, since the
        # same direction vector recurs across many terms of a high-order probe).
        if self.input_map is not None:
            mapped: typ.Dict[int, typ.Any] = {}

            def C(d):
                key = id(d)
                if key not in mapped:
                    mapped[key] = self.input_map.apply(d)
                return mapped[key]

            terms = [dataclasses.replace(t, theta_dirs=tuple((C(d), mult) for d, mult in t.theta_dirs))
                     for t in terms]

        out = self.inner.assemble_partial_sum(terms, inner_omega)

        # post-map the assembled result by request type (state right-hand sides are left untouched --
        # same request-type dispatch the inner uses to decide boundary conditions).
        t0 = terms[0]
        if t0.open_slot is None and t0.function == 'Q' and self.output_map is not None:
            return self.output_map.apply(out)            # a forward probe -> reduced output space
        if t0.open_slot == 'theta' and self.input_map is not None:
            return self.input_map.apply_transpose(out)   # a reverse probe -> input-space covector
        return out
