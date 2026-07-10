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
- [~] **7. Commit + host the docs.** Commit: **done**. Hosting: **decided — GitHub Pages via GitHub
  Actions** (`.github/workflows/docs.yml`): builds the Sphinx site inside the `dolfinx/dolfinx:stable`
  image (autosummary imports the FEniCS/JAX hooks, so dolfinx + jax must be present) and deploys to
  Pages on every push to `main` (plus manual `workflow_dispatch`). The build runs `sphinx -W`, so a doc
  regression fails CI. **Remaining (one-time, at go-public):** repo Settings → Pages → "Build and
  deployment" → Source = **GitHub Actions**; note that Pages for a *private* repo needs a paid plan, so
  the site publishes once the repo is flipped public (until then the build job passes but deploy won't
  serve). Site URL will be `https://nickalger.github.io/implicit_probing/`.
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

## Suggested order

#1 (README) and #4–#5 (metadata) are quick wins that make `twine check` (#3) meaningful; #6
(`py.typed`) wants the clean-env smoke test to confirm it's packaged; #2 (CI) and #7 (docs hosting) are
the larger lifts. #8 (citation) is now done (arXiv:2603.21141 is available). Nothing here is blocked on
code changes.

---

## PyPI Trusted Publishing setup

`.github/workflows/publish.yml` uses **Trusted Publishing (OIDC)** — no API token or GitHub secret.
PyPI mints a short-lived credential from GitHub's OIDC identity. One-time setup, done at release time:

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

**First-release safety (optional, recommended):** dry-run against **TestPyPI** first — register a
pending publisher the same way at <https://test.pypi.org>, temporarily add
`with: { repository-url: https://test.pypi.org/legacy/ }` to the publish step, cut a pre-release to
verify the whole pipeline, then revert. Catches any pipeline/metadata problem before it touches real
PyPI.
