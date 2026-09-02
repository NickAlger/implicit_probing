# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Batch-size inference shared by the hooks (``dev/batched_probes_design.md`` §3).

A *batch* is B independent ``(direction set, omega)`` probes at ONE expansion point, handled in one
``probe`` call. The driver never learns about it: a batch is just another opaque vector type, and each
hook decides what a batched vector looks like (a leading batch axis for array hooks, a ``list`` for the
FEniCSx hook). What the hooks share is the RULE for reading the batch size off a request:

- B comes from the batched vectors the request USES -- the ``theta_dirs`` and ``u_vecs`` of its terms,
  an adjoint ``pairing`` vector, and ``omega`` only through terms whose ``pairing is OMEGA``. The
  driver hands ``omega`` to every ``assemble_partial_sum`` call, including forward and state-RHS
  requests that never touch it; counting it there would wrongly batch the forward chain when only
  ``omega`` is batched.
- Batched vectors with different B in one request raise (a structural error).
- A request with no batched vector is unbatched (``None``): the hook takes its unchanged single path.

This module is dependency-free; the hook supplies ``size_of``, its notion of "batched".
"""
import typing as typ

from implicit_probing.driver import OMEGA, PartialTerm

__all__ = ['infer_batch_size']


def infer_batch_size(
        terms:   typ.Sequence[PartialTerm],
        omega:   typ.Any,                          # the output functional (may be None)
        size_of: typ.Callable[[typ.Any], typ.Optional[int]],  # vector -> B if batched, else None
) -> typ.Optional[int]:                            # -> B, or None for an unbatched request
    """The batch size of one ``assemble_partial_sum`` request (see the module docstring for the rule).

    ``size_of`` is the hook's test: it returns the batch size of a batched vector and ``None`` for a
    single one, and should raise on a malformed vector (e.g. an array of more than two dimensions).
    """
    sizes = set()

    def note(vec):
        b = size_of(vec)
        if b is not None:
            sizes.add(int(b))

    for t in terms:
        for d, _ in t.theta_dirs:
            note(d)
        for v, _ in t.u_vecs:
            note(v)
        if t.pairing is OMEGA:
            if omega is not None:
                note(omega)
        elif t.pairing is not None:
            note(t.pairing)
    if not sizes:
        return None
    if len(sizes) > 1:
        raise ValueError(f'batched vectors with different batch sizes in one request: '
                         f'{sorted(sizes)} (every batched input of a probe must share one B)')
    return sizes.pop()
