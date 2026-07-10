# implicit_probing — public release checklist

Living checklist for the first **public** release. The code/feature set is considered complete; what
remains is packaging and presentation. Read alongside `dev/HANDOFF.md` (where-we-are) — this file is
the release punch-list.

## Release decisions (locked)

- **Versioning: CalVer, `2026.0.0`** — kept deliberately (this is "the 2026 release"), *not* switched
  to SemVer `1.0.0`. So the literal version strings in `pyproject.toml` / `__init__.py` stay; "v1.0"
  is just shorthand for "the first public release". No prose should still call it WIP / early-dev.
- **Distribution target: PyPI** (project name normalizes to `implicit-probing`).
- **Repo visibility:** **public** (flipped 2026-07-10). The GitHub Pages docs site is live at
  <https://nickalger.github.io/implicit_probing/>.
- **Authorship on commits:** Nick, with a `Co-Authored-By: Claude` trailer. Blake is the **first/lead
  package author** (LICENSE / pyproject / headers / `CITATION.cff`) but not on commit trailers while not
  at the keyboard. The *paper* citation keeps its own author order (Alger, Christierson, Chen & Ghattas;
  see #8) — the software-first vs. paper-order difference is deliberate.

## Verified state (2026-07-10)

- Full test suite **green**: 79 passed, 69 subtests in the `fenicsx` conda env (numpy core + the gated
  FEniCS & JAX suites all run; wall time ~45–60 s, dominated by the one-time JAX compile).
- Sphinx docs **build** cleanly with warnings-as-errors (`sphinx -W -b html docs docs/_build/html`);
  the autosummary API reference now generates **one page per function and one page per method**.
- **Docs auto-deploy** is wired up (`.github/workflows/docs.yml` → GitHub Pages; see #7).
- Git clean of build artifacts (`*.egg-info/`, `docs/_build/`, `docs/api/generated/`, `out_*/` all
  gitignored, not tracked).

---

## P0 — Release blockers

- [x] **1. Remove the README WIP banner / "early development" status, add install + usage.** **Done**
 : banner + status gone; new **Install** block (core / `[jax]` / FEniCS caveat) and a
  **Quickstart** showing the toy `probe(...)` call. The snippet is **run-verified against the installed
  wheel** (the `array([...])` outputs are the real values).
- [x] **2. Add CI** (`.github/workflows/ci.yml`). **Done & green on GitHub.** Three jobs: `core`
  (numpy-only matrix, Python 3.9–3.13 — backs the version classifiers), `jax` (`.[jax]` on 3.12), and
  `fenics` (full suite inside the `dolfinx/dolfinx:stable` container). All green on push to `main`. The
  anticipated first-run snags did not bite: plain `pip install` works in the dolfinx image (also proven
  by the green docs build, which installs `.[jax,docs]` the same way), so no `--break-system-packages`
  was needed, and Python 3.9 installs cleanly.
- [x] **3. Publish to PyPI.** **DONE — `implicit_probing 2026.0.0` is live on PyPI** (published
  2026-07-10 via `publish.yml` Trusted Publishing, `pypi` environment with a required-reviewer gate).
  Verified: `pip install implicit_probing` into a clean venv imports the full public API and probes
  match the README quickstart; PyPI METADATA is correct (`License-Expression: MIT`, `Requires-Python
  >=3.9`, `numpy` + `[jax]`/`[docs]` extras, **Blake-first** `Author-email`, `py.typed` shipped).
  <https://pypi.org/project/implicit-probing/>

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
- [x] **7. Commit + host the docs.** **Done — site is live** at
  <https://nickalger.github.io/implicit_probing/>. GitHub Pages via GitHub Actions
  (`.github/workflows/docs.yml`): builds the Sphinx site inside the `dolfinx/dolfinx:stable` image
  (autosummary imports the FEniCS/JAX hooks, so dolfinx + jax must be present) and deploys to Pages on
  every push to `main` (plus manual `workflow_dispatch`); the build runs `sphinx -W`, so a doc
  regression fails CI. Pages Source = **GitHub Actions** (set at go-public). API pages are titled by
  short name (module prefix stripped; methods keep the `Class.method` form for disambiguation).
- [x] **8. `CITATION.cff` + a "How to cite" README section**, pointing at the **arXiv preprint** of the
  T4S paper. **Done** — unblocked now that the revised paper has an arXiv id (**arXiv:2603.21141**;
  Alger, Christierson, Chen & Ghattas 2026). `CITATION.cff` (software authors Blake-first, with a
  `preferred-citation` for the paper in its own author order) + a README **How to cite** section with
  the reference and a BibTeX entry. `t4s.pdf` filename references (which lived in the sibling T3Toolbox
  repo) were replaced repo-wide with "the T4S paper (arXiv:2603.21141)" so the docs are self-contained.
- [x] **9. Document the FEniCS install caveat.** **Done**: the README Install block points
  to the [FEniCS install guide](https://fenicsproject.org/download/) and leaves DOLFINx to the user; no
  install prescribed. `[jax]` stays a normal pip extra; FEniCS is not a pip extra (correctly). (Could
  also echo this one-liner in `docs/fenics_hook.md` if wanted — optional.)

## P2 — Nice-to-have

- [x] **CHANGELOG.md** — `2026.0.0 — unreleased` initial-release entry. **Done**.
- [x] **CONTRIBUTING.md** — **done**: minimal stub (dev setup incl. the FEniCS caveat, running the
  tests with the solve-count note, a pointer to `ARCHITECTURE.md` for the deep conventions, and a PR/MIT
  note). Kept minimal on purpose — `ARCHITECTURE.md` already carries the architecture and house rules.
- [x] **Dependency lower bounds** — **done**: `numpy>=1.21`, `jax>=0.4`. Best-effort floors (CI tests
  current versions, not the floor); permissive enough not to force ancient pins.
- [x] **Clean-env smoke test** — **done** (run once locally): fresh venv → `pip install` the wheel →
  full public API imports → `examples/toy_polynomial.py` runs from the installed wheel (probes ~1e-12
  vs FD, adjointness ~1e-16) → `py.typed` present. Caught nothing broken. (Worth keeping as a manual
  pre-release step; could later be folded into CI as a post-build job.)
- [x] **PyPI publish workflow** — **done**: `.github/workflows/publish.yml`, Trusted Publishing (OIDC,
  no stored token), triggered on a published GitHub Release; build + `twine check` + publish split into
  two jobs. YAML validated locally. The one-time PyPI/GitHub-side setup is in "PyPI Trusted Publishing
  setup" below (done at release time).

---

## What's left

**Nothing — `2026.0.0` is released.** Every checklist item is done: repo public, CI green, docs live at
<https://nickalger.github.io/implicit_probing/>, and the package on PyPI (`pip install implicit_probing`).
The "PyPI Trusted Publishing setup" section below now serves as the runbook for **future** releases (bump
the version in `pyproject.toml` **and** `implicit_probing/__init__.py`, tag `vX.Y.Z`, draft a Release; a
version cannot be re-uploaded, so each release needs a fresh version).

---

## PyPI Trusted Publishing setup

`.github/workflows/publish.yml` uses **Trusted Publishing (OIDC)** — no API token or GitHub secret. The
index mints a short-lived credential from GitHub's OIDC identity. The workflow has **two paths sharing
one build**: a published GitHub Release → real **PyPI** (job `pypi`, environment `pypi`), and a manual
**Run workflow** → **TestPyPI** (job `testpypi`, environment `testpypi`) for the dry-run. Each path needs
its own pending-publisher registration + GitHub environment. Real-PyPI one-time setup (do the TestPyPI
dry-run below first):

1. **PyPI account** — an account at <https://pypi.org> with a verified email and **2FA enabled** (PyPI
   requires 2FA to upload).
2. **Register a pending trusted publisher** (this works *before* the project exists). Go to
   <https://pypi.org/manage/account/publishing/> → "Add a new pending publisher" and enter:
   - **PyPI Project Name:** `implicit_probing`
   - **Owner:** `NickAlger`
   - **Repository name:** `implicit_probing`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`  ← must match `environment.name` in `publish.yml`

   The first successful publish creates the project and promotes the pending publisher to a regular one.
3. **GitHub environment** — create an Environment named `pypi` (repo Settings → Environments → New
   environment → `pypi`). Optional but recommended: add a required-reviewer rule so a human approves
   each publish.
4. **Version must match the build** — `python -m build` reads the version from `pyproject.toml`
   (`2026.0.0`), *not* the git tag; tag the release to match (e.g. `v2026.0.0`).
5. **Release** — Releases → Draft a new release → create the tag → Publish. The `release: published`
   event triggers `publish.yml` (build → `twine check` → upload).

### TestPyPI dry-run (recommended before the first real release)

`publish.yml` has a manual `workflow_dispatch` path that uploads to **TestPyPI** — no temporary edit, no
revert, and it can't misroute a real release (the real-PyPI job runs only on `release: published`).
One-time setup, parallel to the PyPI setup above:

- **TestPyPI pending publisher** — at <https://test.pypi.org/manage/account/publishing/>, add a pending
  publisher with the **same** values as the PyPI one except **Environment name: `testpypi`**.
- **GitHub `testpypi` environment** — repo Settings → Environments → New environment → `testpypi`.

Then run it: **Actions → "Publish" → "Run workflow"** (on `main`). It builds, runs `twine check`, and
uploads `2026.0.0` to TestPyPI (job `testpypi`). Verify at
<https://test.pypi.org/project/implicit-probing/>; optionally install it back with
`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple implicit-probing`
(the extra index lets `numpy` resolve from real PyPI). A given version can be uploaded only **once** per
index, so a second dry-run needs a throwaway `.devN` version bump. Once the dry-run is clean, do the
real-PyPI setup above and cut the release.
