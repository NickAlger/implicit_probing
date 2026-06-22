# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Algorithm 1 of t4s.pdf Section 4: symbolic differentiation of derivative-probe expansions.

A derivative probe of an implicitly-defined map expands, via the chain rule and the implicit function
theorem, into a sum of directional *partial* derivatives of the output map ``Q`` and the state
residual ``R``, contracted against incremental state variables (and, for reverse probes, incremental
adjoints). This module generates those expansions *symbolically* — it produces the algebraic formulas,
not any numbers. The numeric driver (Algorithm 2, ``driver.py``) consumes them.

Symbolic term ``(rho, tau, mu, Gamma)`` (``Term``) represents

    rho( d_theta^|mu| d_u^|Gamma| tau  Theta^mu  Uhat^Gamma ),

i.e. the ``|mu|``-th theta-partial and ``|Gamma|``-th u-partial of the function ``tau`` in {Q, R},
contracted against the direction vectors indexed by ``mu`` and the incremental state variables indexed
by ``Gamma``, all under an outer pairing ``rho``. Here:

- ``rho`` (``Pairing``) — the outer pairing: ``ID`` (forward probes of ``q`` and residual
  derivatives of ``R``), ``OMEGA`` (the output functional, for reverse objects built from ``Q``), or
  an incremental adjoint ``vhat_delta`` (``adjoint(delta)``, for reverse objects built from ``R``).
- ``tau`` — which function the term differentiates, ``'Q'`` or ``'R'``.
- ``mu`` (``Multiset`` of directions) — the explicit ``Theta^mu`` direction factors.
- ``Gamma`` (``Multiset`` of ``Multiset``s) — the ``Uhat^Gamma`` incremental-state factors; each
  element ``gamma`` indexes one incremental state ``uhat_gamma = D^|gamma| u Theta^gamma``.

A sum of terms is a ``dict`` mapping ``Term`` -> integer coefficient (an *expansion*). The four
objects of derivative probing are produced by the *same* differentiation algorithm applied to three
seeds: the forward map ``q`` (``seed_forward_q``), the residual ``R`` (``seed_residual_r``), and a
shared reverse seed for the gradient ``g`` and the adjoint residual ``R^adj`` (``seed_reverse``).

The single-direction differentiation rule is t4s.pdf eqs (19)-(20); the traversal of the
multiset-subset lattice is Algorithm 1.
"""
import collections
import dataclasses
import typing as typ

from implicit_probing.multiset import Multiset, subset_lattice

__all__ = [
    'Pairing', 'ID', 'OMEGA', 'adjoint',
    'Term',
    'Expansion', 'LatticeExpansions',
    'seed_forward_q', 'seed_residual_r', 'seed_reverse',
    'differentiate_term', 'differentiate_over_lattice',
    'extract_state_rhs', 'extract_adjoint_rhs',
    'format_term', 'format_expansion',
]


@dataclasses.dataclass(frozen=True, repr=False)
class Pairing:
    """The outer pairing ``rho`` of a symbolic term (t4s.pdf Section 4.4).

    Three kinds:

    - ``'id'``      — no outer pairing (forward probes of ``q``; derivatives of the residual ``R``).
    - ``'omega'``   — the output functional ``omega`` (reverse objects built from ``Q``: the gradient).
    - ``'adjoint'`` — an incremental adjoint variable ``vhat_delta``, indexed by the order multiset
                      ``delta`` (reverse objects built from ``R``: the adjoint residual). ``delta``
                      empty is the base adjoint ``v`` itself.

    Use the module constants ``ID`` / ``OMEGA`` and the constructor ``adjoint(delta)`` rather than
    instantiating directly.
    """
    kind: str                                  # 'id' | 'omega' | 'adjoint'
    delta: typ.Optional[Multiset] = None       # order multiset of vhat_delta; only for kind == 'adjoint'

    @property
    def is_adjoint(self) -> bool:
        return self.kind == 'adjoint'

    def differentiate(self, direction) -> 'Pairing':
        """Raise the order of an incremental-adjoint pairing: ``vhat_delta`` -> ``vhat_{delta+{i}}`` (eq 20)."""
        if not self.is_adjoint:
            raise ValueError('only an adjoint pairing can be differentiated')
        return adjoint(self.delta.add(direction))

    def __repr__(self) -> str:
        if self.kind == 'id':
            return 'id'
        if self.kind == 'omega':
            return 'omega'
        return f'vhat_{self.delta!r}'


ID = Pairing('id')
OMEGA = Pairing('omega')


def adjoint(delta: Multiset) -> Pairing:
    """The incremental-adjoint pairing ``vhat_delta`` of order multiset ``delta``."""
    return Pairing('adjoint', delta)


@dataclasses.dataclass(frozen=True, repr=False)
class Term:
    """One symbolic term ``(rho, tau, mu, Gamma)`` of a derivative-probe expansion (t4s.pdf eqs 9, 16).

    See the module docstring for the meaning of the four fields. ``Term`` is frozen and hashable, so
    a sum of terms is a ``dict[Term, int]``.
    """
    pairing: Pairing                # rho: outer pairing (ID / OMEGA / vhat_delta)
    function: str                   # tau: 'Q' or 'R'
    theta: Multiset                 # mu: multiset of directions (the Theta^mu factors)
    incrementals: Multiset          # Gamma: multiset of multisets (each indexes a uhat factor)

    def __repr__(self) -> str:
        return format_term(self)


Expansion = typ.Dict[Term, int]                    # a symbolic sum: Term -> integer coefficient
LatticeExpansions = typ.Dict[Multiset, Expansion]  # beta -> the expansion D_beta of D^|beta| F Theta^beta


# --- seeds (the order-zero representations D_empty of the four probing objects) ---

def seed_forward_q() -> Expansion:
    """Order-zero representation of the parameter-to-output map ``q = Q(theta, u(theta))``."""
    return {Term(ID, 'Q', Multiset(), Multiset()): 1}


def seed_residual_r() -> Expansion:
    """Order-zero representation of the state residual ``R(theta, u)``."""
    return {Term(ID, 'R', Multiset(), Multiset()): 1}


def seed_reverse() -> Expansion:
    """Order-zero representation shared by the gradient ``g`` and the adjoint residual ``R^adj``.

    Both equal the Lagrangian ``L = omega(Q) + R(v)`` symbolically; they differ only in *which*
    argument is left open during numeric assembly (theta for ``g``, u for ``R^adj``), not in the
    symbolic expansion. See t4s.pdf Sections 4.3-4.4.
    """
    return {
        Term(OMEGA, 'Q', Multiset(), Multiset()): 1,
        Term(adjoint(Multiset()), 'R', Multiset(), Multiset()): 1,
    }


# --- the differentiation rule (eqs 19-20) and the lattice traversal (Algorithm 1) ---

def differentiate_term(
        term: Term,
        direction,                         # the direction index i to differentiate in
) -> Expansion:
    """Total derivative of one symbolic term in one direction (t4s.pdf eqs 19-20).

    Returns the coefficient contributions ``{new_term: multiplier}`` of differentiating ``term`` once
    in ``direction``. The caller multiplies these by the term's own coefficient and accumulates them.
    """
    rho, tau, mu, Gamma = term.pairing, term.function, term.theta, term.incrementals
    out: typ.Counter = collections.Counter()

    # (19a) differentiate the explicit theta-dependence:  mu -> mu + {i}
    out[Term(rho, tau, mu.add(direction), Gamma)] += 1

    # (19b) differentiate through u, introducing a new first-order incremental state uhat_{i}:
    #       Gamma gains the singleton multiset {i} as a new element.
    out[Term(rho, tau, mu, Gamma.add(Multiset([direction])))] += 1

    # (19c) differentiate each existing incremental-state factor:  uhat_gamma -> uhat_{gamma+{i}}.
    #       A factor of multiplicity m contributes that multiplicity (product rule), so we add m.
    for gamma, multiplicity in Gamma.items():
        raised_Gamma = Gamma.remove(gamma).add(gamma.add(direction))
        out[Term(rho, tau, mu, raised_Gamma)] += multiplicity

    # (20) differentiate the outer pairing, if it is an incremental adjoint:  vhat_delta -> vhat_{delta+{i}}
    if rho.is_adjoint:
        out[Term(rho.differentiate(direction), tau, mu, Gamma)] += 1

    return dict(out)


def differentiate_over_lattice(
        seed: Expansion,                   # the order-zero representation D_empty of the object
        alpha: Multiset,                   # the probing-directions multiset
) -> LatticeExpansions:
    """Algorithm 1: the expansions ``D_beta`` of ``D^|beta| F Theta^beta`` for every ``beta <= alpha``.

    ``seed`` is the order-zero representation of the object ``F`` (one of ``seed_forward_q`` /
    ``seed_residual_r`` / ``seed_reverse``). The lattice of sub-multisets of ``alpha`` is traversed by
    nondecreasing cardinality, and each node is obtained by differentiating an already-computed parent
    ``beta - {i}`` once in a direction ``i`` of ``beta``.
    """
    expansions: LatticeExpansions = {Multiset(): dict(seed)}
    for beta in subset_lattice(alpha):
        if len(beta) == 0:
            continue  # the base node is the input seed
        i = beta.any_element()
        parent = expansions[beta.remove(i)]
        acc: typ.Counter = collections.Counter()
        for term, coeff in parent.items():
            for new_term, multiplier in differentiate_term(term, i).items():
                acc[new_term] += coeff * multiplier
        expansions[beta] = {t: c for t, c in acc.items() if c != 0}
    return expansions


# --- isolating the operator term to read off the incremental right-hand sides (Algorithm 2 setup) ---

def extract_state_rhs(
        expansion: Expansion,              # D_beta of the residual R, i.e. D^|beta| R Theta^beta
        beta:      Multiset,
) -> Expansion:
    """The incremental-state right-hand side terms ``b_beta`` (still symbolic), t4s.pdf eqs (10)-(11).

    The incremental state equation ``0 = D^|beta| R Theta^beta`` contains exactly one operator term,
    ``(ID, 'R', empty, {beta})`` = ``d_u R uhat_beta`` = ``A uhat_beta``. This isolates it (the future
    left-hand side of ``A uhat_beta = b_beta``) and returns every *other* term; the caller negates and
    assembles them as ``b_beta``.
    """
    operator = Term(ID, 'R', Multiset(), Multiset([beta]))
    if expansion.get(operator, 0) != 1:
        raise ValueError(f'expected the operator term {operator!r} with coefficient 1 in the state '
                         f'expansion for beta={beta!r}; got coefficient {expansion.get(operator, 0)}')
    return {t: c for t, c in expansion.items() if t != operator}


def extract_adjoint_rhs(
        expansion: Expansion,              # D_beta of the reverse seed, i.e. D^|beta| of the Lagrangian
        beta:      Multiset,
) -> Expansion:
    """The incremental-adjoint right-hand side terms ``c_beta`` (still symbolic), t4s.pdf eqs (17)-(18).

    Under an open u-slot the term ``(adjoint(beta), 'R', empty, empty)`` is ``(d_u R)(vhat_beta)`` =
    ``A* vhat_beta``; this isolates it (the future left-hand side of ``A* vhat_beta = c_beta``) and
    returns every other term, to be assembled (with the open u-slot) and negated as ``c_beta``. The
    base case ``beta = empty`` recovers ``A* v = -omega(d_u Q)`` (the adjoint equation, eq 14).
    """
    operator = Term(adjoint(beta), 'R', Multiset(), Multiset())
    if expansion.get(operator, 0) != 1:
        raise ValueError(f'expected the operator term {operator!r} with coefficient 1 in the adjoint '
                         f'expansion for beta={beta!r}; got coefficient {expansion.get(operator, 0)}')
    return {t: c for t, c in expansion.items() if t != operator}


# --- human-readable formatting (for debugging and examples; not used by the algorithm) ---

def format_term(term: Term) -> str:
    """A compact human-readable rendering of a single ``Term``."""
    a, b = len(term.theta), len(term.incrementals)
    core = f'd_theta^{a} d_u^{b} {term.function}  Theta^{term.theta!r} Uhat^{term.incrementals!r}'
    rho = term.pairing
    if rho.kind == 'id':
        return core
    if rho.kind == 'omega':
        return f'omega( {core} )'
    return f'( {core} )(vhat_{rho.delta!r})'


def format_expansion(expansion: Expansion) -> str:
    """A multi-line, deterministically-ordered rendering of a symbolic sum."""
    lines = []
    for term, coeff in sorted(expansion.items(), key=lambda tc: repr(tc[0])):
        sign = '+' if coeff >= 0 else '-'
        lines.append(f'  {sign} {abs(coeff)} * {format_term(term)}')
    return '\n'.join(lines)
