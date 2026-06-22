# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Algorithm 2 of t4s.pdf Section 4: the derivative-probing driver.

Given a problem that can (a) solve linear systems with the linearized state operator ``A = d_u R`` and
its adjoint, and (b) *assemble sums* of directional partial derivatives of ``R`` and ``Q``, this walks
the multiset-subset lattice of the probing directions, solving the incremental state and adjoint
systems at each node, and assembles every forward and reverse probe of ``D^j q(theta0)``.

**The driver is vector-type agnostic.** It performs NO arithmetic on the problem's state/output
vectors. It only (i) builds *symbolic* descriptions of the sums to assemble -- resolving the symbolic
direction labels to the caller's direction vectors and the already-computed incremental solves --
(ii) hands each whole sum to ``problem.assemble_partial_sum``, and (iii) routes the returned opaque
vectors into the next solve or back to the caller. All linear algebra on physics vectors lives in the
problem. Integer coefficients ride along in the request and are applied by the problem during
assembly. This lets one driver serve a numpy toy, a FEniCS/PETSc PDE, or a JAX problem unchanged.

Querying *sums* (rather than individual partials) is deliberate: a single combined assembly (e.g. one
FEniCS form) is far cheaper than assembling many and adding them, and handing the whole -- possibly
mixed-``R``/``Q`` -- sum to the problem keeps even the cross-term additions out of the driver.
"""
import dataclasses
import typing as typ

from implicit_probing.backend.multiset import Multiset, subset_lattice
from implicit_probing.backend.symbolic import (
    ID, OMEGA as _SYMBOLIC_OMEGA,
    seed_forward_q, seed_residual_r, seed_reverse,
    differentiate_over_lattice, extract_state_rhs, extract_adjoint_rhs,
)

__all__ = [
    'OMEGA',
    'PartialTerm',
    'ImplicitProblem',
    'probe',
]


class _OmegaFunctional:
    """Sentinel marking that a term's output is paired with the output functional omega.

    The driver keeps this sentinel in a term's ``pairing`` -- rather than substituting the concrete
    ``omega`` vector -- so that a problem (e.g. a composed problem) can tell an output-functional
    pairing apart from an incremental-adjoint vector and treat them differently. The actual ``omega``
    is supplied to ``assemble_partial_sum`` separately.
    """
    def __repr__(self) -> str:
        return 'OMEGA'


OMEGA = _OmegaFunctional()  # the single sentinel instance; problems test ``term.pairing is OMEGA``


@dataclasses.dataclass
class PartialTerm:
    """One term of a sum the driver asks a problem to assemble.

    Represents ``coefficient * pairing( d_theta^a d_u^b function (theta_dirs..., [open], u_vecs...,
    [open]) )``, evaluated at the expansion point: an ``a = len(theta_dirs)``-th theta-partial and
    ``b = len(u_vecs)``-th u-partial of ``R`` or ``Q``, contracted against the given direction
    vectors, with at most one slot left OPEN (it becomes the test-function / free slot, so the result
    is a covector in that slot's space) and an optional outer pairing. All vectors are opaque to the
    driver -- they are whatever the problem produced and consumes.
    """
    coefficient: int                      # integer multiplier (may be negative)
    function:    str                      # 'R' or 'Q'
    theta_dirs:  typ.Tuple                # filled theta-direction vectors
    u_vecs:      typ.Tuple                # filled incremental-state vectors
    open_slot:   typ.Optional[str]        # None | 'theta' | 'u' -- which slot is left free
    pairing:     typ.Any                  # None | OMEGA | an incremental-adjoint vector


@typ.runtime_checkable
class ImplicitProblem(typ.Protocol):
    """What the probing driver requires of a problem. Every physics vector is opaque to the driver.

    The problem owns the expansion point ``(theta0, u0)`` and the operator ``A = d_u R`` there, and is
    responsible for its own setup (e.g. the state solve) before the driver's first call. The output
    functional ``omega`` is NOT a property of the problem -- it is a per-probe choice of quantity of
    interest, supplied to ``probe`` and passed through to ``assemble_partial_sum``.
    """

    def solve_operator(self, b: typ.Any) -> typ.Any:
        """Solve the linearized state system ``A x = b`` for ``x`` (``A = d_u R`` at the expansion point)."""

    def solve_operator_adjoint(self, c: typ.Any) -> typ.Any:
        """Solve the adjoint system ``A* x = c`` for ``x``."""

    def assemble_partial_sum(self, terms: typ.Sequence[PartialTerm], omega: typ.Any) -> typ.Any:
        """Assemble ``sum_i terms[i]`` as one vector, resolving ``OMEGA`` pairings to ``omega``.

        ``omega`` is the output functional (a covector in the output space); it is used only by terms
        whose ``pairing`` is the ``OMEGA`` sentinel (forward and residual terms ignore it).
        """


def _resolve_pairing(symbolic_pairing, v_hat):
    """Map a symbolic outer pairing to a request pairing: None / OMEGA sentinel / an adjoint vector."""
    if symbolic_pairing == ID:
        return None
    if symbolic_pairing == _SYMBOLIC_OMEGA:
        return OMEGA
    if symbolic_pairing.is_adjoint:
        return v_hat[symbolic_pairing.delta]
    raise ValueError(f'unexpected symbolic pairing {symbolic_pairing!r}')


def _lower(
        expansion,                         # an Expansion (dict Term -> int)
        direction_vectors,                 # label -> theta-direction vector
        u_hat,                             # multiset -> incremental-state vector
        v_hat,                             # multiset -> incremental-adjoint vector (or None if unused)
        open_slot,                         # None | 'theta' | 'u'
        sign,                              # +1 for probes, -1 for right-hand sides
):
    """Lower a symbolic expansion into a list of ``PartialTerm`` by substituting vectors for labels."""
    terms = []
    for term, coeff in expansion.items():
        theta_dirs = tuple(direction_vectors[label] for label in term.theta.expanded())
        u_vecs = tuple(u_hat[gamma] for gamma in term.incrementals.expanded())
        pairing = _resolve_pairing(term.pairing, v_hat)
        terms.append(PartialTerm(sign * coeff, term.function, theta_dirs, u_vecs, open_slot, pairing))
    return terms


def probe(
        problem:           ImplicitProblem,
        alpha:             Multiset,            # multiset of probing-direction LABELS
        direction_vectors: typ.Mapping,         # label -> the actual theta-direction vector (opaque)
        omega:             typ.Any = None,      # output functional; None -> forward probes only
) -> typ.Tuple[typ.Dict[Multiset, typ.Any], typ.Dict[Multiset, typ.Any]]:
    """Algorithm 2: forward (and, if ``omega`` is given, reverse) probes over the sub-multisets of ``alpha``.

    Returns ``(forward, reverse)``, each a dict over every ``beta <= alpha``:

    - ``forward[beta]`` = ``D^|beta| q(theta0)`` applied to ``beta``'s directions -- an output-space
      vector. ``forward[empty]`` is ``q(theta0)`` itself.
    - ``reverse[beta]`` = ``omega(D^{|beta|+1} q(theta0))`` applied to ``beta``'s directions with one
      slot left open -- a parameter-space covector (so it gives the sensitivity to *every* open
      direction from a single adjoint solve). ``reverse[empty]`` is the gradient of ``omega(q)``.

    ``omega`` is the output functional: a single covector in the output space, a per-probe choice of
    quantity of interest, NOT a property of the problem. If ``omega is None``, only forward probes are
    computed (the adjoint solves and the reverse pass are skipped) and ``reverse`` is empty.
    """
    reverse_wanted = omega is not None
    # Symbolic expansions for every node (Algorithm 1), once. The reverse seed serves both the adjoint
    # residual R^adj (-> the c_beta right-hand sides) and the gradient g (-> the reverse probes).
    expansions_R = differentiate_over_lattice(seed_residual_r(), alpha)
    expansions_q = differentiate_over_lattice(seed_forward_q(), alpha)
    expansions_reverse = differentiate_over_lattice(seed_reverse(), alpha) if reverse_wanted else None

    u_hat: typ.Dict[Multiset, typ.Any] = {}  # nonempty beta -> incremental state (uhat_empty = u0 is never referenced)
    v_hat: typ.Dict[Multiset, typ.Any] = {}  # every beta -> incremental adjoint (vhat_empty = v, the base adjoint)

    # Traverse the lattice by nondecreasing cardinality, solving the incremental system(s) per node.
    for beta in subset_lattice(alpha):
        if len(beta) > 0:  # the base state uhat_empty = u0 is the problem's own state solve, not a system here
            b_terms = _lower(extract_state_rhs(expansions_R[beta], beta),
                             direction_vectors, u_hat, None, open_slot=None, sign=-1)
            u_hat[beta] = problem.solve_operator(problem.assemble_partial_sum(b_terms, omega))
        if reverse_wanted:
            c_terms = _lower(extract_adjoint_rhs(expansions_reverse[beta], beta),
                             direction_vectors, u_hat, v_hat, open_slot='u', sign=-1)
            v_hat[beta] = problem.solve_operator_adjoint(problem.assemble_partial_sum(c_terms, omega))

    # Assemble every forward probe (and reverse probe, if omega was given).
    forward: typ.Dict[Multiset, typ.Any] = {}
    reverse: typ.Dict[Multiset, typ.Any] = {}
    for beta in subset_lattice(alpha):
        forward[beta] = problem.assemble_partial_sum(
            _lower(expansions_q[beta], direction_vectors, u_hat, None, open_slot=None, sign=1), omega)
        if reverse_wanted:
            reverse[beta] = problem.assemble_partial_sum(
                _lower(expansions_reverse[beta], direction_vectors, u_hat, v_hat, open_slot='theta', sign=1), omega)
    return forward, reverse
