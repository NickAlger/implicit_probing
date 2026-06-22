# implicit_probing — project guide

Read this first. A standalone Python package for **derivative probing of implicitly-defined maps**.

## What this is

A clean, from-scratch implementation of **Algorithms 1 & 2, Section 4** of the T4S paper ("Tucker
Tensor Train Taylor Series", Alger, Christierson, Chen & Ghattas 2026). Given a parameter-to-output
map

    q(theta) = Q(theta, u(theta)),   where the state u(theta) solves   R(theta, u) = 0,

it computes **forward and reverse probes of the higher-derivative tensors** `D^j q(theta_0)` — the
contractions of those tensors against direction vectors that serve as training data for a local
Taylor surrogate. Probing is the only practical access to these tensors: they are far too large to
form, and because `u` depends implicitly on `theta`, individual entries are not directly available.

This package is **laser-focused on the derivative machinery** and stands alone (depends on nothing but
`numpy`). It deliberately does NOT include the Tucker-tensor-train format, fitting/optimization, or
manifold geometry — those live in the sibling package **T3Toolbox** (the *fitting* side, Sections
5-6). Probe output here is plain arrays + functionals, with no tensor-format coupling, so any consumer
can use it.

Non-goal (context only — NOT built here): the maintainer's longer-term research fits a *global*
polynomial (coefficients as a Tucker tensor train) to probe data gathered at *many* expansion points.
Keep `theta_0` a first-class input so multi-point gathering stays a trivial outer loop.

## The paper (`t4s.pdf` lives in the sibling T3Toolbox repo)

Reference for the algorithms, not a design target — when good general-purpose library design and the
paper's specific usage conflict, library design wins. Section 4 is the relevant part; Sections 2-3
give the setting and notation. **The math, in brief:**

- A *forward probe* applies a derivative tensor to direction vectors; a *reverse probe* leaves one
  slot open and pairs the rest with an output functional `omega`.
- Every probe expands (chain rule + implicit function theorem) into directional **partial**
  derivatives of `Q` and `R`, contracted against **incremental state** variables
  `uhat_beta = D^|beta| u Theta^beta` (and, for reverse probes, **incremental adjoints** `vhat_beta`).
- Each `uhat_beta` solves a linear system `A uhat_beta = b_beta` with the SAME operator `A = d_u R`;
  only the right-hand side changes. Adjoints solve `A* vhat_beta = c_beta`. The RHS depends only on
  lower-order incrementals, so a probe **traverses the lattice of multiset-subsets `beta` of the
  direction multiset `alpha`**, reusing every lower-order sub-probe.
- **Algorithm 1** (symbolic, no numerics) generates the expansions; **Algorithm 2** (numeric) walks
  the lattice doing the solves and assembling the probes. Cost: lattice size `prod(alpha_i + 1)`,
  from `O(j)` solves when directions repeat to `O(2^j)` when all differ.

## Architecture

**Pure-functional backend now; thin OO frontend later** (added once the numerics work).

- `implicit_probing/backend/` — stateless functions, each module with its own `__all__`.
- The symbolic engine (Algorithm 1) is **pure Python, no numpy**: it manipulates multisets and
  dictionaries of symbolic terms. This is the novel core, unit-tested against the paper's hand-derived
  expansions.

### Symbolic term representation (Algorithm 1)

A symbolic term is a tuple `(rho, tau, mu, Gamma)` (`symbolic.Term`):

- `rho` — outer pairing: `ID` (forward / residual), `OMEGA` (gradient), or `adjoint(delta)` =
  `vhat_delta` (adjoint residual). `symbolic.Pairing`.
- `tau` — which function, `'Q'` or `'R'`.
- `mu` — a **multiset of directions** (the `Theta^mu` / `d_theta` slots).
- `Gamma` — a **multiset of multisets** (the `Uhat^Gamma` incremental-state factors).

A sum of terms is a `dict[Term, int]` (term -> integer coefficient). The four probing objects (forward
`q`, residual `R`, gradient `g`, adjoint residual `R^adj`) are the SAME differentiation algorithm
applied to three seeds (`g` and `R^adj` share one seed, differing only at numeric assembly). The
single-direction rule is t4s.pdf eqs (19)-(20); the lattice traversal is Algorithm 1.

## Code style

Mirror T3Toolbox's conventions:

- **Signature shape-comment style** for array code: the trailing `# shape` comment is the type the
  language can't express (one arg per line, name/type/`#`-comment aligned). The array-heavy numeric
  driver (slice 2+) follows this; the symbolic engine is non-array code, so shape comments mostly
  don't apply there.
- File header block: `# Authors / # Copyright / # Github`.
- One `__all__` per backend module.
- Cite the t4s.pdf equation/algorithm number in the docstring when code implements one.
- House philosophy: **structural problems raise unconditionally** (wrong shape/inconsistent lengths);
  numerical preconditions are a later concern. No auto-formatter near the deliberately-aligned style.

## Verification & testing

- Correctness against **ground truth**: the symbolic engine vs the paper's hand-derived expansions;
  later, the numeric probes vs finite differences / AD.
- `unittest` in `tests/` (mirrors the package as `tests/backend/`). Pattern: `subTest` over cases.
- **Run** with the maintainer's env Python and `PYTHONPATH=$PWD`:
  `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q`. (The env path is maintainer-local; see
  `~/.claude/`.)

## Knowledge routing

- User-facing design rationale -> `docs/` (Sphinx, added later). Internal status/handoff ->
  `dev/HANDOFF.md` (read it for where-we-are + next steps).
- **Essential/Dirichlet BCs** — a constraint for the *future* autodiff-hook layer, NOT the symbolic
  engine: the state equation enforces the real BCs, but the adjoint and ALL incremental equations
  enforce **homogenized** (zeroed-BC-dof) versions. (Also kept in maintainer memory.)

## Current state

**Slice 1 done: the symbolic engine (Algorithm 1)** — `backend/multiset.py` + `backend/symbolic.py` +
tests, all green. Next: RHS extraction (isolate `A uhat_beta`), the problem interface, and the numeric
driver (Algorithm 2). See `dev/HANDOFF.md`.
