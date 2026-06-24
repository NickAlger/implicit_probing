# implicit_probing — public release checklist

Living checklist for the first **public** release. The code/feature set is considered complete; what
remains is packaging and presentation. Read alongside `dev/HANDOFF.md` (where-we-are) — this file is
the release punch-list.

## Release decisions (locked)

- **Versioning: CalVer, `2026.0.0`** — kept deliberately (this is "the 2026 release"), *not* switched
  to SemVer `1.0.0`. So the literal version strings in `pyproject.toml` / `__init__.py` stay; "v1.0"
  is just shorthand for "the first public release". No prose should still call it WIP / early-dev.
- **Distribution target: PyPI** (project name normalizes to `implicit-probing`).
- **Repo visibility:** currently **private**; flip to **public** at release time.
- **Authorship on commits:** Nick, with a `Co-Authored-By: Claude` trailer. Blake is a package author
  (LICENSE / pyproject / headers) but not on commit trailers while not at the keyboard.

## Verified state (2026-06-23)

- Full test suite **green**: 79 passed, 69 subtests, ~46 s in the `fenicsx` conda env (numpy core +
  the gated FEniCS & JAX suites all run).
- Sphinx docs **build** cleanly (`sphinx -b html docs docs/_build/html`).
- Git clean of build artifacts (`*.egg-info/`, `docs/_build/`, `out_*/` all gitignored, not tracked).

---

## P0 — Release blockers

- [x] **1. Remove the README WIP banner / "early development" status, add install + usage.** **Done**
 : banner + status gone; new **Install** block (core / `[jax]` / FEniCS caveat) and a
  **Quickstart** showing the toy `probe(...)` call. The snippet is **run-verified against the installed
  wheel** (the `array([...])` outputs are the real values).
- [~] **2. Add CI** (`.github/workflows/ci.yml`). **Written**, three jobs: `core`
  (numpy-only matrix, Python 3.9–3.13 — backs the version classifiers), `jax` (`.[jax]` on 3.12), and
  `fenics` (full suite inside the `dolfinx/dolfinx:stable` container). YAML validated locally; **cannot
  actually run until the branch is pushed to GitHub**. Likely first-run tweaks: the dolfinx image may
  be PEP 668 externally-managed (pip may need a venv or `--break-system-packages`), and Python 3.9 in
  the `core` job may surface install/syntax issues (if so, bump `requires-python`). Watch the first run
  and iterate.
- [~] **3. Confirm the PyPI name is free** and validate the build. Name: **confirmed available** —
  `GET https://pypi.org/simple/implicit-probing/` returns **404**, which the simple index does only
  when no project is registered (a reserved/zero-release name 200s). `implicit_probing` and
  `implicit-probing` normalize to the same name; the name is claimed at first `twine upload`. Build:
  **done & clean** — `python -m build` produced sdist+wheel and `twine check` **PASSED** on both
  (verified locally in a throwaway venv, Python 3.11). METADATA confirmed: `License-Expression: MIT`,
  all 10 classifiers, `Requires-Python >=3.9`, `numpy` + `[jax]`/`[docs]` extras. Only **actual
  `twine upload`** remains (at release, once name is claimed).

## P1 — Should-do for a credible release

- [x] **4. Fix stale `pyproject.toml` metadata comment** (lines ~10–12 said the numeric driver was
  "forthcoming" — it's done). **Done**.
- [x] **5. Flesh out `classifiers`.** **Done**: Development Status 5 - Production/Stable,
  Intended Audience :: Science/Research, OS Independent, Python 3 + 3.9–3.13, Topic ::
  Scientific/Engineering :: Mathematics. **No** `License ::` classifier — MIT is declared via the SPDX
  `license = "MIT"` field, and recent setuptools (PEP 639) errors if both are present. The 3.9–3.13
  span is a claim CI (#2) must back.
- [x] **6. Ship a `py.typed` marker.** **Done & verified**: empty `implicit_probing/py.typed` +
  `[tool.setuptools.package-data]`. Confirmed present in the built wheel *and* in the installed
  location after `pip install` of the wheel.
- [ ] **7. Commit + host the docs.** Commit: **done** (the Sphinx scaffolding). Hosting: still to
  decide (ReadTheDocs vs GH Pages) — note the repo is private until release, which affects RTD/Pages
  wiring.
- [ ] **8. `CITATION.cff` + a "How to cite" README section**, pointing at the **arXiv preprint** of the
  T4S paper. **BLOCKED:** the algorithms cited here (Algorithms 1 & 2) only exist as algorithm boxes in
  the *revised* paper; the current arXiv version has only the prose/derivations. Gated on uploading the
  revised paper to arXiv. Revisit once that's live.
- [x] **9. Document the FEniCS install caveat.** **Done**: the README Install block points
  to the [FEniCS install guide](https://fenicsproject.org/download/) and leaves DOLFINx to the user; no
  install prescribed. `[jax]` stays a normal pip extra; FEniCS is not a pip extra (correctly). (Could
  also echo this one-liner in `docs/fenics_hook.md` if wanted — optional.)

## P2 — Nice-to-have

- [x] **CHANGELOG.md** — `2026.0.0 — unreleased` initial-release entry. **Done**.
- [ ] **CONTRIBUTING.md** — decision: add; content TBD (talk through what to put in it).
- [x] **Dependency lower bounds** — **done**: `numpy>=1.21`, `jax>=0.4`. Best-effort floors (CI tests
  current versions, not the floor); permissive enough not to force ancient pins.
- [x] **Clean-env smoke test** — **done** (run once locally): fresh venv → `pip install` the wheel →
  full public API imports → `examples/toy_polynomial.py` runs from the installed wheel (probes ~1e-12
  vs FD, adjointness ~1e-16) → `py.typed` present. Caught nothing broken. (Worth keeping as a manual
  pre-release step; could later be folded into CI as a post-build job.)
- [ ] **PyPI publish workflow** (trusted publishing / token) so future releases are one tag. Decision:
  add; maintainer hasn't done this before — walk through it.

---

## Suggested order

#1 (README) and #4–#5 (metadata) are quick wins that make `twine check` (#3) meaningful; #6
(`py.typed`) wants the clean-env smoke test to confirm it's packaged; #2 (CI) and #7 (docs hosting) are
the larger lifts. #8 (citation) is externally blocked on the arXiv revision. Nothing here is blocked on
code changes.
