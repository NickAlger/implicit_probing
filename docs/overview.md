# Overview: probing higher derivatives of implicitly-defined maps

A short tour of what `implicit_probing` does, how the pieces fit together, and how to use it.

## The problem

Many models compute an output only after solving an expensive implicit system (for example a PDE).
Write the parameter-to-output map as

```
q(theta) = Q(theta, u(theta)),   where the state u(theta) solves   R(theta, u) = 0.
```

Tasks like optimization, uncertainty quantification, and building local surrogate models often need
the higher derivatives `D^j q(theta0)` of such a map. Two things make these hard to get at:

- **They are enormous.** The `j`-th derivative of a map between high-dimensional spaces is a
  `(j+1)`-index tensor — far too large to form or store.
- **Their entries are not directly accessible.** Because `u` depends on `theta` only *implicitly*
  (through the solve), you cannot read off an entry without, in effect, differentiating through the
  solver.

## The idea

You never actually need the whole tensor — you need to **probe** it: contract it with a few vectors.
`implicit_probing` computes two kinds of probe of `D^j q(theta0)`:

- a **forward probe** — apply the derivative tensor to `j` direction vectors, giving an output vector;
- a **reverse probe** — leave one slot open and pair the output with a functional `omega`, giving a
  covector — a gradient-like sensitivity to *every* open direction from a single solve.

The fact that makes this cheap: every probe reduces to a handful of **linearized solves that all
share the same operator** — the linearized state operator `A = d_u R` (or its adjoint) — differing
only in their right-hand sides. The work is organized over the *subsets* of the probing directions,
so a high-order probe reuses all of its lower-order sub-probes for free.

This package is a clean, standalone implementation of Algorithms 1 and 2 from Section 4 of the T4S
paper. It depends only on NumPy, and its probe output is plain arrays — so it can feed any downstream
use (surrogate fitting, optimization, etc.).

## How the code is organized

The probing machinery is a set of small, pure-functional modules at the top level of the package. The
names you use are re-exported from `implicit_probing` directly, so the common path is one flat import:
`from implicit_probing import probe, ImplicitProblem`. Concrete problems live in their own modules,
imported explicitly so their dependencies stay optional.

| module | role |
| --- | --- |
| `multiset` | `Multiset` and `subset_lattice` — the index sets the symbolic engine is organized over (internal; the `probe` API speaks `(vector, power)` pairs and power-tuples, not `Multiset`). |
| `symbolic` | **Algorithm 1**: a pure-symbolic engine that works out *which* partial derivatives of `R` and `Q` each probe is a sum of. No numbers. |
| `driver` | **Algorithm 2**: `probe(...)`, which walks the lattice, asks your problem to do the solves and assemble the partial-derivative sums, and returns the probes. |
| `composition` | `ComposedProblem` — probe `W ∘ q ∘ C` for linear input/output maps `C`, `W` (see `composition.md`). |
| `reference_problems` | a toy implicit map (`make_toy_problem`) with exact derivatives — for testing and as a worked example of the problem interface. Imported explicitly (`implicit_probing.reference_problems`); needs numpy. |
| `validation` | finite-difference ground truth + the exact reverse/forward adjoint identity, for *checking* probes. This is testing infrastructure, **not** part of the real workflow — a probe is exact and far cheaper than differencing. Imported explicitly (`implicit_probing.validation`). |

The probing machinery itself does no arithmetic on physics vectors, so the top-level
`import implicit_probing` pulls in nothing but the standard library; numpy loads only when you reach
for `reference_problems` (or your own numpy-backed problem).

## Examples

Runnable scripts in `examples/`, each in labelled sections so the probing stands out from the
problem setup:

- **`examples/toy_polynomial.py`** — numpy only, no heavy dependencies: the smallest end-to-end use
  (build a problem, `probe`, read the probes). The best first read.
- **`examples/fenics_poisson.py`** — the same three probing lines against a real nonlinear-Poisson
  PDE (DOLFINx), with the QoI-gradient field as the payoff.
- **`examples/fenics_composition.py`** — probing a dimension-reduced map `W ∘ q ∘ C` (see
  `composition.md`).

## Using it on the built-in toy

```python
import numpy as np
from implicit_probing import probe
from implicit_probing.reference_problems import make_toy_problem

problem = make_toy_problem()        # q(theta) = Q(theta, u(theta)), R(theta, u) = 0; theta in R^2

# Directions are (vector, max_power) pairs: this asks for every probe up to a^2 b^1.
a = np.array([1.0, 0.3])
b = np.array([0.4, -0.6])
directions = [(a, 2), (b, 1)]
omega = np.array([1.0, 0.0])        # output functional (the QoI) the reverse probes differentiate

forward, reverse = probe(problem, directions, omega)

forward[(2, 1)]              # D^3 q(theta0) applied to (a, a, b)       -> output vector in R^2
reverse[(2, 1)]             # omega(D^4 q(theta0)(a, a, b, .))         -> covector in R^2 (parameter space)
forward[(1, 0)]             # the lower-order sub-probe D^1 q(theta0)(a) comes for free
```

A single `probe(directions, ...)` call returns the forward and reverse probe for **every** power-tuple
`mu` in the box `∏_k {0, ..., max_power_k}`, since the lower-order probes are computed along the way.
A probe is a mixed partial derivative of the map restricted to the directions, named by its
differentiation multi-index — picture `theta = theta0 + s*a + t*b`:

- `forward[(i, j)]` is `∂_s^i ∂_t^j q(theta0 + s a + t b)|_0 = D^{i+j} q(theta0)[a^i b^j]` (an
  output-space vector). `forward[(0, 0)]` is just `q(theta0)`.
- `reverse[(i, j)]` is the covector with `reverse[(i, j)] @ d = omega(D^{i+j+1} q(theta0)[a^i b^j, d])`
  for any extra direction `d`. `reverse[(0, 0)]` is the gradient of `omega(q)`.

The probes are exact (up to the cost of the linear solves); the toy ships with a finite-difference
ground truth (`forward_probe_by_finite_difference`) used to verify them.

`omega` is the output functional — a single covector in the output space and a per-probe choice of
quantity of interest, not a property of the problem. It is needed only for the reverse probes; pass
`omega=None` (the default) to compute forward probes alone and skip the adjoint solves.

## Using it on your own problem

`probe` is generic. To run it on your model, pass an object implementing the
`implicit_probing.ImplicitProblem` interface — three methods:

```python
class ImplicitProblem(Protocol):
    def solve_operator(self, b):                   ...  # solve A x = b   (A = d_u R at the expansion point)
    def solve_operator_adjoint(self, c):           ...  # solve A* x = c
    def assemble_partial_sum(self, terms, omega):  ...  # assemble sum_i terms[i]; resolve OMEGA pairings to omega
```

Two things to know:

- **The driver does no arithmetic on your vectors** — they are opaque to it. It only triggers solves
  (through your two `solve_*` methods) and asks you to assemble sums of partial derivatives. So the
  same driver works whether your vectors are NumPy arrays, FEniCS functions, or PETSc vectors.
- **`assemble_partial_sum` is asked for a whole sum, not one partial at a time.** It receives a list
  of `PartialTerm`s — each a coefficient times a directional partial derivative of `R` or `Q`,
  possibly with one slot left open (the test function) and an outer pairing (the functional `OMEGA`,
  or an adjoint vector) — and returns their sum as one vector. The directions a partial is contracted
  against come as `(vector, multiplicity)` pairs (the partial is symmetric in each block, so they are
  a multiset): a backend can exploit the multiplicity or just expand it back to a flat list. Requesting
  the whole sum lets you build a single combined form and assemble it once, which matters for
  performance in libraries like FEniCS.

`reference_problems.ImplicitPolynomialProblem` is a complete, readable implementation of these three
methods for a polynomial map — use it as a template.
