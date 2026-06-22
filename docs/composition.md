# Composing with linear input/output maps

Often you don't want to probe the raw map `q : theta -> obs` itself, but a version with linear maps on
the input and/or output:

- the **input** is a few features `x` that define the full parameter field via a linear map
  `theta = C x` (e.g. a low-order polynomial parameterization); and/or
- the **output** of interest is a linear function `z = W obs` of the raw observation (e.g. a selection
  matrix that restricts a FEniCS observation vector to the boundary dofs it is actually supported on).

`implicit_probing.backend.composition.ComposedProblem` lets you probe the composed map
`f = W o q o C` directly.

## Why it's cheap and exact

Because `C` and `W` are **linear**, their higher derivatives vanish and they pass straight through the
chain rule, so the probes of `f` are simple linear images of the probes of `q`:

```
forward:  D^j f(x0)(xhat...) = W( D^j q(theta0)(C xhat...) )
reverse:  psi^f_beta = C^T( psi^q_beta ),   q's reverse taken with the pulled-back functional W^T omega
```

The state space, the operator `A = d_u R`, and all the incremental solves are **untouched** — `C` and
`W` only reparameterize the input and re-observe the output. So composition is purely a transformation
at the boundaries of probing: pre-map the directions by `C`, post-map a forward probe by `W` and a
reverse probe by `C^T`, and pull the output functional back by `W^T`.

## Using it

```python
from implicit_probing.backend.composition import ComposedProblem
from implicit_probing.backend.driver import probe

composed = ComposedProblem(inner_problem, input_map=C, output_map=W)   # f = W o q o C
forward, reverse = probe(composed, alpha, x_directions, omega)
```

- `inner_problem` — any `ImplicitProblem` (the toy, the FEniCS hook, ...).
- `x_directions` — a `{label: vector}` map in the **input** (`C`-domain) space.
- `omega` — the output functional, a covector in the **reduced output** (`W`-codomain) space.
- `forward[beta]` comes back in the reduced output space; `reverse[beta]` is a covector in the input
  space. Either map may be `None` (identity). `ComposedProblem` is itself an `ImplicitProblem`, so
  compositions nest.

## The linear operators

`C` and `W` are `LinearOperator`s — objects with `apply(v)` (forward action) and `apply_transpose(w)`
(transpose action), producing/consuming whatever vectors the inner problem uses (numpy arrays for the
toy; PETSc vectors / DOLFINx Functions for FEniCS). For a plain matrix there is a ready-made adapter:

```python
from implicit_probing.backend.composition import MatrixOperator
C = MatrixOperator(P)   # apply = P @ v,  apply_transpose = P.T @ v
```

For FEniCS, you typically write small operators of your own — e.g. an input map that builds a
`theta` Function from feature coefficients, and an output selection that reads off the boundary dofs.
Worked examples of both are in `tests/test_fenics_composition.py` (boundary-dof selection `W` and a
polynomial-feature parameterization `C`), validated against finite differences and the reverse
adjointness identity.
