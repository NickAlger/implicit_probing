# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Release JAX's compiled programs between test modules.

Every compiled program holds memory mappings for the life of the process, and Linux caps those
(``vm.max_map_count``, 65,530 by default). One pytest process runs the FEniCSx hook, the per-term jet
kernels, and several whole-probe programs (``compiled_probe``); this session saw LLVM fail with
"Cannot allocate memory" once that accumulation crossed the limit, and one uncaptured hard crash
of the full suite in the dolfinx environment. Dropping the caches per module keeps the count
bounded at no cost to the tests (each module re-warms what it needs).
"""
import pytest


@pytest.fixture(autouse=True, scope='module')
def _clear_jax_caches_per_module():
    yield
    try:
        import jax
    except ImportError:                          # numpy-only environments: nothing to clear
        return
    jax.clear_caches()
