# Contributing to implicit_probing

Thanks for your interest! This package is deliberately small and **focused on the derivative-probing
machinery** (Algorithms 1 & 2 of the T4S paper). [`CLAUDE.md`](CLAUDE.md) is the project guide — it
covers the architecture and the rationale behind the design. Bug reports, documentation fixes, and new
framework hooks (following the existing FEniCS / JAX pattern) are all welcome.

## Development setup

```bash
git clone https://github.com/NickAlger/implicit_probing
cd implicit_probing
pip install -e ".[jax]"     # core + JAX hook; drop [jax] for the numpy-only core
```

The **FEniCS hook** also needs DOLFINx, which is not pip-installable — install it separately (see the
[FEniCS install guide](https://fenicsproject.org/download/)). The FEniCS and JAX tests skip
automatically when their dependency is absent.

## Running the tests

```bash
pytest tests/ -q
```

All tests must pass, and new behavior should come with tests. Note that the driver has **solve-count**
tests, not just numerical ones: a change to the lattice traversal must preserve the minimal number of
linear solves (`tests/test_driver.py::TestSolveCounts`).

## Code style and conventions

Mirror the surrounding code. The conventions — array shape-comment style, the file-header block, one
`__all__` per module, citing the `t4s.pdf` equation/algorithm a piece of code implements, and
"structural problems raise unconditionally" — are documented in [`CLAUDE.md`](CLAUDE.md). There is no
auto-formatter; the alignment is deliberate, so please don't reflow unrelated code.

## Pull requests

Branch off `main`, keep changes focused, and open a PR. By contributing you agree that your
contributions are licensed under the project's [MIT License](LICENSE).
