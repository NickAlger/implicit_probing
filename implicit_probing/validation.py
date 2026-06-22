# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Finite-difference verification of derivative probes -- testing infrastructure, NOT a workflow.

These helpers exist only to *check* that a probing driver / problem hook is correct. They are not part
of the intended use of the library: a probe is exact and far cheaper than finite differences -- it is a
set of linearized solves that all share one factorized operator, versus a fresh nonlinear solve per
stencil point -- so real tasks call ``probe`` directly and never come here. We use these to validate
it; nothing more.

Both helpers are vector-type agnostic, like the driver: the array arithmetic lives in caller-supplied
hooks (a ``perturb`` for the parameter point, ``pair`` functionals for the dot products), defaulting to
plain numpy ``+`` / ``@`` so the numpy toy needs no hooks while a FEniCS/PETSc problem supplies its own.

- ``forward_probe_by_finite_difference`` -- an independent ground truth for a forward probe
  ``D^j q(theta0)`` of any symmetry, by a tensor product of central differences of the end-to-end map
  ``q`` (Richardson-extrapolated). Touches only ``q`` -- never the analytic partials or the driver.
- ``reverse_forward_adjointness`` -- the exact discrete-adjoint identity tying a reverse probe to the
  forward probe one order higher, ``psi_beta . d_k == omega(D^{|beta|+1} q(..., d_k))``, swept over the
  lattice. No finite differences: it cross-checks the reverse probes (which used the adjoint solves)
  against the forward probes to solver precision.
"""
import itertools
import typing as typ

__all__ = [
    'forward_probe_by_finite_difference',
    'reverse_forward_adjointness',
]


# Central finite-difference stencils for the m-th derivative: (offsets, weights), leading error O(h^2).
_CENTRAL_STENCILS: typ.Dict[int, typ.Tuple[typ.Tuple[int, ...], typ.Tuple[float, ...]]] = {
    1: ((-1, 1),           (-0.5, 0.5)),
    2: ((-1, 0, 1),        (1.0, -2.0, 1.0)),
    3: ((-2, -1, 1, 2),    (-0.5, 1.0, -1.0, 0.5)),
    4: ((-2, -1, 0, 1, 2), (1.0, -4.0, 6.0, -4.0, 1.0)),
}


def _default_perturb(point, scale, direction):
    """Move ``point`` by ``scale * direction`` (numpy default; returns a new object, never mutates)."""
    return point + scale * direction


def _default_pair(covector, vector):
    """Pair a covector with a vector (numpy inner product)."""
    return covector @ vector


def _mixed_central_difference(q, theta0, direction_orders, h, perturb):
    """One tensor-product central-difference estimate (step ``h``) of the mixed derivative of ``q``."""
    per_direction = []  # (direction, offsets, h-scaled weights)
    for direction, m in direction_orders:
        offsets, weights = _CENTRAL_STENCILS[m]
        per_direction.append((direction, offsets, tuple(w / h ** m for w in weights)))
    total = None
    for picks in itertools.product(*[range(len(offsets)) for (_, offsets, _) in per_direction]):
        point = theta0
        coeff = 1.0
        for (direction, offsets, weights), idx in zip(per_direction, picks):
            point = perturb(point, offsets[idx] * h, direction)   # composes: theta0 + sum_k s_k d_k
            coeff *= weights[idx]
        contribution = coeff * q(point)
        total = contribution if total is None else total + contribution
    return total


def forward_probe_by_finite_difference(
        q,                                  # end-to-end map: a parameter point -> output vector
        theta0,                             # the expansion point (opaque)
        direction_orders,                   # the multiset {d_k^{m_k}}: a sequence of (direction, multiplicity)
        *,
        perturb:    typ.Callable = _default_perturb,   # (point, scale, direction) -> moved point
        h:          float = 1e-2,
        richardson: bool = True,
) -> typ.Any:                               # -> output vector (whatever ``q`` returns)
    """Independent ground truth for the forward probe ``D^j q(theta0)`` applied to the given directions.

    ``direction_orders`` lists the *distinct* probing directions with their multiplicities, so this
    handles any symmetry: fully symmetric ``[(d, j)]``, fully asymmetric ``[(d1,1),(d2,1),(d3,1)]``,
    and anything between, e.g. ``[(d1,2),(d2,1)]``. The probe equals the mixed derivative
    ``d_{s_1}^{m_1} ... d_{s_l}^{m_l} q(theta0 + sum_k s_k d_k)`` at ``s = 0``, estimated here by a
    tensor product of central differences (Richardson-extrapolated when ``richardson`` is set, which
    cancels the leading ``O(h^2)`` error). It touches only ``q`` -- never the analytic partials or the
    probing driver -- so it independently validates a probe.

    ``perturb(point, scale, direction)`` returns ``point`` moved by ``scale * direction`` without
    mutating it; the default is numpy ``point + scale * direction``. A FEniCS problem passes a closure
    that builds a fresh ``Function``. ``q`` and ``perturb`` are the only places this touches the
    problem's vector types.
    """
    estimate = _mixed_central_difference(q, theta0, direction_orders, h, perturb)
    if richardson:
        finer = _mixed_central_difference(q, theta0, direction_orders, h / 2, perturb)
        estimate = (4.0 * finer - estimate) / 3.0  # cancels the leading O(h^2) error
    return estimate


def reverse_forward_adjointness(
        forward:    typ.Mapping,            # power-tuple mu -> forward probe (output vector), from probe()
        reverse:    typ.Mapping,            # power-tuple mu -> reverse probe (parameter covector), from probe()
        directions: typ.Sequence,           # ((vector, max_power), ...) -- the same sequence passed to probe()
        omega:      typ.Any,                # the output functional the reverse probes were taken with
        *,
        pair_input:  typ.Callable = _default_pair,   # (reverse covector, direction vector) -> scalar
        pair_output: typ.Callable = _default_pair,   # (omega, forward output)              -> scalar
) -> float:
    """Max relative error of the exact reverse/forward adjoint identity, swept over the lattice.

    For every power-tuple ``mu`` and axis ``k`` with ``mu + e_k`` still in the probed box, a correct
    pair of probes satisfies, with no approximation,

        reverse[mu] . directions[k][0]  ==  omega . forward[mu + e_k],

    i.e. pairing the reverse probe's open slot with axis ``k``'s direction reproduces ``omega`` applied
    to the forward probe one order higher in that axis. No finite differences are involved -- this
    cross-checks the reverse probes (which used the adjoint solves) against the forward probes
    (themselves anchored by ``forward_probe_by_finite_difference``) to solver precision. Returns the
    largest relative discrepancy over the sweep (``0.0`` if there are no eligible ``(mu, k)`` pairs).

    ``pair_input`` / ``pair_output`` are the two inner products (parameter space and output space),
    defaulting to numpy ``@``; a FEniCS problem passes closures that extract the underlying arrays.
    """
    worst = 0.0
    for mu in itertools.product(*[range(max_power + 1) for _, max_power in directions]):
        for k, (vector, max_power) in enumerate(directions):
            if mu[k] >= max_power:           # mu + e_k leaves the box -> that forward probe wasn't computed
                continue
            child = mu[:k] + (mu[k] + 1,) + mu[k + 1:]
            lhs = float(pair_input(reverse[mu], vector))
            rhs = float(pair_output(omega, forward[child]))
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30))
    return worst
