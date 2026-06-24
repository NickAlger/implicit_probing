# Notes for Claude Code

The project guide — architecture, the math, the conventions, and current state — lives in
[`ARCHITECTURE.md`](ARCHITECTURE.md). **Read it first.** It is the canonical design reference for this
repo; this file is intentionally thin so that contributors who don't use Claude aren't sent to a
Claude-specific document.

Related docs:

- [`README.md`](README.md) — what the package is, install, quickstart.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup, running the tests, PR flow.
- [`dev/HANDOFF.md`](dev/HANDOFF.md) — internal status / handoff (where-we-are + next steps).
- [`dev/RELEASE_CHECKLIST.md`](dev/RELEASE_CHECKLIST.md) — the path to a public PyPI release.

## Running here (maintainer machine)

Use the maintainer's conda env (see maintainer memory — the shell autoloads the wrong env by default).
Run the suite with the env's Python and `PYTHONPATH=$PWD`:

    PYTHONPATH=$PWD <env-python> -m pytest tests/ -q

The env path is maintainer-local (see `~/.claude/`). The FEniCS and JAX tests skip when their
dependency is absent, so they only run in an env that has them.
