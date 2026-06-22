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
`from implicit_probing import probe, ImplicitProblem, Multiset`. Concrete problems live in their own
modules, imported explicitly so their dependencies stay optional.

| module | role |
| --- | --- |
| `multiset` | `Multiset` and `subset_lattice` — the index sets the probes are organized over. |
| `symbolic` | **Algorithm 1**: a pure-symbolic engine that works out *which* partial derivatives of `R` and `Q` each probe is a sum of. No numbers. |
| `driver` | **Algorithm 2**: `probe(...)`, which walks the lattice, asks your problem to do the solves and assemble the partial-derivative sums, and returns the probes. |
| `composition` | `ComposedProblem` — probe `W ∘ q ∘ C` for linear input/output maps `C`, `W` (see `composition.md`). |
| `reference_problems` | a toy implicit map (`make_toy_problem`) with exact derivatives, plus a finite-difference ground truth — for testing and as a worked example of the problem interface. Imported explicitly (`implicit_probing.reference_problems`); needs numpy. |

The probing machinery itself does no arithmetic on physics vectors, so the top-level
`import implicit_probing` pulls in nothing but the standard library; numpy loads only when you reach
for `reference_problems` (or your own numpy-backed problem).

## Using it on the built-in toy

```python
import numpy as np
from implicit_probing import Multiset, probe
from implicit_probing.reference_problems import make_toy_problem

problem = make_toy_problem()        # q(theta) = Q(theta, u(theta)), R(theta, u) = 0; theta in R^2

# Probe directions are given as a multiset of *labels*, plus the vector each label stands for.
alpha = Multiset([1, 1, 2])         # probe the 3rd derivative in directions (d1, d1, d2)
directions = {1: np.array([1.0, 0.3]), 2: np.array([0.4, -0.6])}
omega = np.array([1.0, 0.0])        # output functional (the QoI) the reverse probes differentiate

forward, reverse = probe(problem, alpha, directions, omega)

forward[alpha]               # D^3 q(theta0) applied to (d1, d1, d2)   -> output vector in R^2
reverse[alpha]               # omega(D^4 q(theta0)(d1, d1, d2, .))     -> covector in R^2 (parameter space)
forward[Multiset([1])]       # the lower-order sub-probe D^1 q(theta0)(d1) comes for free
```

A single `probe(alpha, ...)` call returns the forward and reverse probe for **every** sub-multiset of
`alpha`, since the lower-order probes are computed along the way:

- `forward[beta]` is `D^|beta| q(theta0)` applied to `beta`'s directions (an output-space vector).
  `forward[Multiset([])]` is just `q(theta0)`.
- `reverse[beta]` is the covector `omega(D^{|beta|+1} q(theta0))` with one slot open;
  `reverse[beta] @ d` equals `omega` applied to the order-`(|beta|+1)` forward probe in `beta`'s
  directions plus the extra direction `d`. `reverse[Multiset([])]` is the gradient of `omega(q)`.

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
  or an adjoint vector) — and returns their sum as one vector. Requesting the whole sum lets you build
  a single combined form and assemble it once, which matters for performance in libraries like FEniCS.

`reference_problems.ImplicitPolynomialProblem` is a complete, readable implementation of these three
methods for a polynomial map — use it as a template.
