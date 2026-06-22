# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""implicit_probing: forward and reverse derivative probes of implicitly-defined maps.

The public API is re-exported here, so the common path is a single flat import::

    from implicit_probing import probe, ImplicitProblem, Multiset

``probe`` walks the multiset-subset lattice of the probing directions and returns the forward and
reverse probes of ``D^j q(theta0)``; a problem just implements the three-method ``ImplicitProblem``
protocol. The whole probing machinery is dependency-free (it does no arithmetic on physics vectors),
so importing this package pulls in nothing but the standard library.

Concrete problems live in their own modules, imported explicitly so their dependencies stay optional:

- ``implicit_probing.reference_problems`` -- a numpy toy map with exact derivatives + a
  finite-difference ground truth (also the reference implementation of the interface).
- ``implicit_probing.fenics`` -- a DOLFINx (modern FEniCS) hook (requires a conda dolfinx env).
"""
from implicit_probing.driver import probe, ImplicitProblem, PartialTerm, OMEGA
from implicit_probing.multiset import Multiset, subset_lattice
from implicit_probing.composition import ComposedProblem, LinearOperator, MatrixOperator

__version__ = "2026.0.0"

__all__ = [
    'probe',
    'ImplicitProblem',
    'PartialTerm',
    'OMEGA',
    'Multiset',
    'subset_lattice',
    'ComposedProblem',
    'LinearOperator',
    'MatrixOperator',
]
