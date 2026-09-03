# Unreproduced hard crash of the full test suite (dolfinx env), 2026-09-02

*Status: seen once, not reproduced in three further full runs. Mitigation in place (below). Kept so
that a recurrence is recognized and diagnosed rather than rediscovered.*

## What happened

Right after `TestSolverFactory` and `TestCompiledProbe` were added to `tests/test_jax.py` (the
compiled-probe unit, commit `b7d6375`), the first full run of the suite in the **dolfinx
environment** (`PYTHONPATH=$PWD <fenicsx-python> -m pytest tests/ -q`) aborted with a Python fatal
error rather than a test failure. Only the tail of the interpreter's crash dump was captured:

    File "<frozen runpy>", line 198 in _run_module_as_main
    Extension modules: numpy._core._multiarray_umath, ..., petsc4py.PETSc, mpi4py.MPI, _cffi_backend,
    jaxlib.cpu_feature_guard, ..., scipy.linalg._decomp_update (total: 33)

The head of the dump (`Fatal Python error: ...` / `Current thread ...` / the test frame) scrolled
off, so **the crashing test is unknown**. The process had PETSc and MPI initialized (the FEniCSx
tests run before the JAX tests in file order) and had compiled several whole-probe programs.

Reruns, all in the same environment, all green:

| run | command | result |
|---|---|---|
| 2 | `tests/test_jax.py -q -x -p no:cacheprovider`, `-X faulthandler` | 34 passed, 98 subtests |
| 3 | `tests/ -q -p no:cacheprovider`, `-X faulthandler` | 125 passed, 260 subtests |
| 4 | same as 3, in the background | 125 passed, 260 subtests |

## Suspects, in order

1. **Compiled-program accumulation hitting `vm.max_map_count` (65,530 here).** Every XLA
   executable holds memory mappings for the life of the process. Earlier the same day, the suite
   in the jax-only env failed deterministically with `LLVM compilation error: Cannot allocate
   memory` once the batched-probe tests compiled enough kernel variants (fixed then by reusing one
   problem instance per test class and `jax.clear_caches()` at class teardown). The dolfinx-env
   process compiles strictly more (the FEniCSx tests' ffcx kernels plus every JAX program), so it
   sits nearer the limit; a crash at the margin would be load-dependent and non-reproducible.
2. **`HostSolver` (`jax.pure_callback` calling scipy) in a process with MPI initialized.** The
   callback runs on an XLA host thread. An earlier variant of the bridge that handed the callback's
   zero-copy device-buffer view straight to scipy produced a heap corruption
   (`corrupted size vs. prev_size`) and wrong results; it was fixed (the bridge copies its input)
   *before* the crash, and the copy has held in every run since — but the pairing of a host
   callback thread with an MPI-initialized process is the other place where a heap error could
   originate.
3. Something unrelated and rare (a PETSc/MUMPS thread interaction at process exit, say).

## Mitigation in place

`tests/conftest.py` (commit `d812f5c`): a module-scoped autouse fixture calls `jax.clear_caches()`
after every test module, bounding the compiled-program count in one pytest process. This addresses
suspect 1 only.

## If it recurs

- Run with `-X faulthandler -p no:cacheprovider` and capture EVERYTHING to a file
  (`> crash.log 2>&1`); the `Current thread` block names the test.
- Bisect the interaction: `tests/test_fenics.py tests/test_jax.py` together vs each alone; then
  `tests/test_jax.py::TestCompiledProbe::test_host_solver_bridge_rebinds_per_point` alone in the
  dolfinx env (suspect 2).
- Watch the mapping count of the pytest process near the end of the run:
  `wc -l /proc/<pid>/maps` against `/proc/sys/vm/max_map_count` (suspect 1).
- Try `OMP_NUM_THREADS=1` (MUMPS/BLAS threading) and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`
  (XLA host threads) to separate the thread-interaction hypotheses.
