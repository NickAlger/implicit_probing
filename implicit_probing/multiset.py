# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Immutable multisets and the multiset-subset lattice.

A *multiset* (bag) is a collection in which elements may repeat. This module provides a small
hashable, immutable ``Multiset`` type and ``subset_lattice``, which enumerates every sub-multiset of a
given multiset, ordered by nondecreasing cardinality.

These are the combinatorial backbone of the symbolic derivative-probing engine (``symbolic.py``,
Algorithm 1 of the T4S paper, Section 4). A derivative probe is indexed by a multiset ``alpha`` of the
probing directions; the incremental state/adjoint variables are indexed by its sub-multisets
``beta``; and the symbolic terms themselves contain a multiset of directions ``mu`` and a multiset
*of multisets* ``Gamma`` (the incremental-state factors). Because ``Multiset`` is hashable and its
elements may themselves be ``Multiset``s, one type serves all of these roles.
"""
import itertools
import typing as typ

__all__ = [
    'Multiset',
    'subset_lattice',
]


class Multiset:
    """An immutable, hashable multiset (bag): a collection in which elements may repeat.

    Elements must be hashable; they may themselves be ``Multiset``s (the engine uses multisets of
    multisets). Instances are treated as immutable — ``add`` / ``remove`` / ``+`` / ``-`` all return
    *new* ``Multiset``s and never mutate ``self`` — which is what makes them safe to use as dictionary
    keys and as elements of other multisets.

    Equality and hashing are by contents and are order-independent: ``Multiset([1, 2]) ==
    Multiset([2, 1])``.
    """
    __slots__ = ('_counts', '_hash')

    def __init__(
            self,
            elements: typ.Iterable = (),   # an iterable of (hashable) elements, with repeats
    ):
        counts: typ.Dict[typ.Any, int] = {}
        for e in elements:
            counts[e] = counts.get(e, 0) + 1
        self._counts = counts
        self._hash = None

    @classmethod
    def from_counts(
            cls,
            counts: typ.Mapping,           # element -> count; non-positive counts are dropped
    ) -> 'Multiset':
        """Build a multiset directly from an element-to-count mapping."""
        self = cls.__new__(cls)
        self._counts = {e: int(c) for e, c in counts.items() if c > 0}
        self._hash = None
        return self

    # --- queries ---

    def count(self, element) -> int:
        """Multiplicity of ``element`` (0 if absent)."""
        return self._counts.get(element, 0)

    def items(self) -> typ.Tuple[typ.Tuple[typ.Any, int], ...]:
        """``(element, count)`` pairs, ordered deterministically by element ``repr``."""
        return tuple(sorted(self._counts.items(), key=lambda ec: repr(ec[0])))

    def distinct(self) -> typ.Tuple:
        """The distinct elements, ordered deterministically (no multiplicities)."""
        return tuple(e for e, _ in self.items())

    def expanded(self) -> typ.Tuple:
        """The elements repeated by multiplicity, deterministically ordered.

        ``Multiset([1, 1, 2]).expanded() == (1, 1, 2)``. Useful for fanning a multiset of direction
        labels / incremental indices out into the flat list of vectors an assembly request needs.
        """
        return tuple(e for e, c in self.items() for _ in range(c))

    @property
    def cardinality(self) -> int:
        """Total number of elements counted with multiplicity (``== len(self)``)."""
        return sum(self._counts.values())

    def __len__(self) -> int:
        return sum(self._counts.values())

    def __contains__(self, element) -> bool:
        return element in self._counts

    def any_element(self):
        """A deterministic element of the multiset (the ``repr``-minimal one). Raises if empty."""
        if not self._counts:
            raise ValueError('any_element() called on an empty Multiset')
        return min(self._counts, key=repr)

    # --- derived multisets (never mutate self) ---

    def add(self, element, n: int = 1) -> 'Multiset':
        """A copy with ``n`` more copies of ``element``."""
        if n < 0:
            raise ValueError('add count must be nonnegative')
        counts = dict(self._counts)
        counts[element] = counts.get(element, 0) + n
        return Multiset.from_counts(counts)

    def remove(self, element, n: int = 1) -> 'Multiset':
        """A copy with ``n`` fewer copies of ``element``. Raises if fewer than ``n`` are present."""
        if n < 0:
            raise ValueError('remove count must be nonnegative')
        have = self._counts.get(element, 0)
        if n > have:
            raise ValueError(f'cannot remove {n} copies of {element!r}; only {have} present')
        counts = dict(self._counts)
        counts[element] = have - n
        return Multiset.from_counts(counts)

    def __add__(self, other: 'Multiset') -> 'Multiset':
        """Multiset sum: multiplicities add."""
        counts = dict(self._counts)
        for e, c in other._counts.items():
            counts[e] = counts.get(e, 0) + c
        return Multiset.from_counts(counts)

    def __sub__(self, other: 'Multiset') -> 'Multiset':
        """Multiset difference: multiplicities subtract, clamped at zero."""
        counts = dict(self._counts)
        for e, c in other._counts.items():
            if e in counts:
                counts[e] -= c
        return Multiset.from_counts(counts)

    # --- containment / ordering ---

    def issubmultiset(self, other: 'Multiset') -> bool:
        """True if every element's multiplicity here is ``<=`` its multiplicity in ``other`` (non-strict)."""
        return all(c <= other._counts.get(e, 0) for e, c in self._counts.items())

    def __le__(self, other: 'Multiset') -> bool:
        return self.issubmultiset(other)

    # --- equality / hashing / display ---

    def __eq__(self, other) -> bool:
        return isinstance(other, Multiset) and self._counts == other._counts

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._counts.items()))
        return self._hash

    def __repr__(self) -> str:
        inner = ', '.join(repr(e) for e, c in self.items() for _ in range(c))
        return '{|' + inner + '|}'


def subset_lattice(
        alpha: Multiset,                   # the directions multiset
) -> typ.Tuple[Multiset, ...]:
    """Every sub-multiset ``beta`` of ``alpha`` (including the empty multiset and ``alpha`` itself).

    Returned in nondecreasing cardinality order, so that a traversal computing each node from a
    smaller parent (Algorithm 1) always finds the parent already done. The number of nodes is
    ``prod(count_i + 1)`` over the distinct elements of ``alpha`` — from ``|alpha| + 1`` when all
    directions coincide (fully symmetric) up to ``2 ** |alpha|`` when all differ (fully asymmetric).
    """
    distinct = alpha.items()  # ((element, count), ...), deterministically ordered
    ranges = [range(count + 1) for (_, count) in distinct]
    subs = []
    for choice in itertools.product(*ranges):
        counts = {distinct[idx][0]: c for idx, c in enumerate(choice) if c > 0}
        subs.append(Multiset.from_counts(counts))
    subs.sort(key=len)  # stable: nondecreasing cardinality
    return tuple(subs)
