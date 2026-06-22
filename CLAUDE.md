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

**Flat, pure-functional package.** Small stateless modules at the top level of `implicit_probing/`,
each with its own `__all__`; the public names are re-exported from `implicit_probing/__init__.py`, so
the common path is one flat import (`from implicit_probing import probe, ImplicitProblem, Multiset`).
There is no backend/frontend split and no separate OO layer planned — the `probe(...)` free function
plus the `ImplicitProblem` protocol already *is* the API.

- The symbolic engine (Algorithm 1) is **pure Python, no numpy**: it manipulates multisets and
  dictionaries of symbolic terms. This is the novel core, unit-tested against the paper's hand-derived
  expansions.
- **Dependency isolation is a property of the import graph, not the layout.** The whole probing
  machinery (`multiset`, `symbolic`, `driver`, `composition`) is dependency-free, so `__init__` pulls
  in nothing but the stdlib; numpy enters only through `reference_problems`. Concrete hooks with heavy
  deps (`fenics`, future `jax`) live in their own modules, imported explicitly (`implicit_probing.fenics`)
  and never from `__init__`, so a missing dep only bites when that hook is actually touched.
- **Folder dependency rule.** `implicit_probing/` (the importable library) is the stable core: any
  folder may import from it, but `tests/`, `examples/`, and `docs/` never import one another. So shared
  code either earns its place in the library, if generic, or is duplicated across leaf folders, if
  problem-specific — e.g. the generic finite-difference verifier lives in `validation.py`, whereas a
  specific FEniCS PDE is written out in full in *both* its example and its test (legible vs. decisive,
  two different jobs). Finite differences are testing-only: a probe is exact and far cheaper, so the
  real workflow never uses them.

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

### The driver and problem interface (Algorithm 2)

`driver.py` is the numeric driver: `probe(problem, directions, omega=None)` walks the lattice and
returns `(forward, reverse)`. **User-facing API:** `directions` is a sequence of `(vector, max_power)`
pairs (the distinct axes + how far to probe each); the returned dicts are keyed by **power-tuples** `mu`
— `forward[mu]` is the mixed partial `d_s^{mu_0} d_t^{mu_1} ... q` (an output vector), `reverse[mu]` a
parameter covector. `probe` resolves the pairs to internal position-labels, runs the label-`Multiset`
engine unchanged, then translates the `Multiset` keys back to power-tuples on the way out — so `Multiset`
never crosses the boundary (the math reason: a probe is a Taylor coefficient of `q` on the slice through
the directions, intrinsically multi-indexed; the algorithm's multiset notation is the *internal* layer).
It is **vector-type agnostic** — it does no arithmetic on physics vectors, only lowering symbolic terms
to `PartialTerm` requests, handing whole sums to the problem, and routing the opaque results. A problem implements the 3-method `ImplicitProblem` protocol:
`solve_operator` (`A x = b`), `solve_operator_adjoint` (`A* x = c`), and `assemble_partial_sum`
(assemble a *whole sum* of partial-derivative terms — requesting sums, not singletons, lets a hook
like FEniCS assemble one combined form). `reference_problems.py` (numpy) holds the toy
implicit map + the exact-derivative `Polynomial` + a finite-difference ground truth;
`ImplicitPolynomialProblem` is the reference implementation of the interface.

## Code style

Mirror T3Toolbox's conventions:

- **Signature shape-comment style** for array code: the trailing `# shape` comment is the type the
  language can't express (one arg per line, name/type/`#`-comment aligned). The array-heavy code
  (`reference_problems.py`, future hooks) follows this; the symbolic engine and the vector-agnostic
  driver are non-array code, so shape comments mostly don't apply there.
- File header block: `# Authors / # Copyright / # Github`.
- One `__all__` per module.
- Cite the t4s.pdf equation/algorithm number in the docstring when code implements one.
- House philosophy: **structural problems raise unconditionally** (wrong shape/inconsistent lengths);
  numerical preconditions are a later concern. No auto-formatter near the deliberately-aligned style.

## Verification & testing

- Correctness against **ground truth**: the symbolic engine vs the paper's hand-derived expansions,
  and the numeric probes (Algorithm 2) vs an independent finite-difference ground truth
  (`implicit_probing.validation`, vector-agnostic; `reference_problems.forward_probe_by_finite_difference`
  is the numpy convenience wrapper), swept over symmetric / partial / asymmetric probes.
- **Solve-count (efficiency) is a first-class test, not just numerical correctness**: the lattice
  algebra exists to minimize linearized solves, and a correct-but-wasteful traversal would pass every
  value check. `tests/test_driver.py::TestSolveCounts` wraps the toy in a counting problem and asserts
  the driver performs *exactly* `prod(p_k+1) - 1` forward and `prod(p_k+1)` adjoint solves (the empty
  node is the user's nonlinear base solve; the base adjoint is a real solve), swept over total / partial
  / no repetition. Count at the `solve_operator` boundary (what the driver asks for), not `solve_A`.
- `unittest` in `tests/` (flat, mirroring the flat package). Pattern: `subTest` over cases.
- **Run** with the maintainer's env Python and `PYTHONPATH=$PWD`:
  `PYTHONPATH=$PWD <env-python> -m pytest tests/ -q`. (The env path is maintainer-local; see
  `~/.claude/`.)

## Knowledge routing

- User-facing docs -> `docs/` (`docs/overview.md` is the user-facing tour; Sphinx later). Internal
  status/handoff -> `dev/HANDOFF.md` (read it for where-we-are + next steps).
- **Essential/Dirichlet BCs** — a constraint for the *future* autodiff-hook layer, NOT the symbolic
  engine: the state equation enforces the real BCs, but the adjoint and ALL incremental equations
  enforce **homogenized** (zeroed-BC-dof) versions. (Also kept in maintainer memory.)

## Current state

**Algorithms 1 & 2 are implemented and validated — the method runs end-to-end.** The symbolic engine
(`multiset.py`, `symbolic.py`) and the numeric driver (`driver.py`) are done; the toy reference problem
+ finite-difference ground truth validate the driver's probes across symmetric / partial / asymmetric
symmetries (~1e-9). FEniCS/DOLFINx hook (`fenics.py`) + linear composition (`composition.py`) done.
Vector-agnostic verification helpers (FD ground truth + the exact reverse/forward adjoint identity) in
`validation.py`. Runnable examples in `examples/` (`toy_polynomial.py` numpy-only, plus the two FEniCS
scripts), each split into labelled probing / problem-setup / verification sections. User-facing tour in
`docs/overview.md`. Full suite green (`pytest tests/ -q`). No core algorithm work remains; the main open
candidate is a JAX hook. See `dev/HANDOFF.md`.
