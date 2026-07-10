# implicit_probing

Probing the higher-derivative tensors of maps that depend *implicitly* on the solution of a large
system of equations.

Many scientific models evaluate a quantity of interest only after solving an expensive implicit
system (e.g. a PDE): for a parameter `theta`,

```
q(theta) = Q(theta, u(theta)),     where the state u(theta) solves     R(theta, u) = 0.
```

Local surrogates, optimization, and uncertainty quantification often need the higher derivatives
`D^j q(theta_0)` of such a map. Those derivative tensors are far too large to form, and because the
state `u` depends implicitly on `theta`, their entries are not directly accessible — they can be
reached only by **probing**: contracting the tensor against direction vectors.

This package computes **forward and reverse derivative probes** of `D^j q(theta_0)` from the partial
derivatives of `R` and `Q`, by the adjoint method and the implicit function theorem. It is a clean,
standalone implementation of **Algorithms 1 and 2 of Section 4** of:

> Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026). *Tucker Tensor Train Taylor Series*.
> arXiv preprint arXiv:2603.21141.

Each probe reduces to a set of linearized solves that all share the same operator (the linearized
state operator `A = d_u R`, or its adjoint) and differ only in their right-hand sides. The work is
organized as a traversal of the lattice of multiset-subsets of the probing directions, so that
high-order probes reuse all of their lower-order sub-probes.

## Install

```bash
pip install implicit_probing          # core: symbolic engine + numeric driver (numpy only)
pip install "implicit_probing[jax]"    # add the JAX hook (Taylor-mode autodiff)
```

The FEniCS/DOLFINx hook needs DOLFINx, which is **not** pip-installable — install it separately (see
the [FEniCS install guide](https://fenicsproject.org/download/)), then use `implicit_probing.fenics`.

## Quickstart

```python
import numpy as np
from implicit_probing import probe
from implicit_probing.reference_problems import make_toy_problem

# A built-in toy: q(theta) = Q(theta, u(theta)) with R(theta, u) = 0, theta and q in R^2 and a 3-dof
# implicit state u. Swap this for your own object implementing the ImplicitProblem protocol.
problem = make_toy_problem()

# Probing directions as (vector, max_power) pairs: probe `a` up to power 2 and `b` up to power 1,
# i.e. ask for every mixed derivative up to a^2 b. omega is an output-space functional (the QoI).
a = np.array([1.0, 0.3])
b = np.array([0.4, -0.6])
omega = np.array([1.0, 0.0])

forward, reverse = probe(problem, [(a, 2), (b, 1)], omega)

forward[(2, 1)]   # D^3 q [a, a, b], an output-space vector   ->  array([-0.1197, -0.0638])
reverse[(0, 0)]   # gradient of omega(q) w.r.t. theta          ->  array([ 0.2077,  0.2953])
```

`forward[mu]` is the mixed partial of order `mu` (a Taylor coefficient of `q` on the slice through the
probing directions); `reverse[mu]` is the matching parameter-space covector, from a single adjoint
solve. Every lower-order sub-probe falls out of the same shared-operator solves for free. For a real
problem you implement the three-method `ImplicitProblem` protocol — see the
[overview](docs/overview.md) and the FEniCS/JAX scripts under [`examples/`](examples/).

## Scope

`implicit_probing` is **laser-focused on the derivative machinery** and depends on nothing but
`numpy`. Its probe output is plain arrays and functionals, with no particular tensor format baked in,
so it can feed any downstream consumer (for example the Tucker-tensor-train *fitting* machinery in the
sibling package [T3Toolbox](https://github.com/NickAlger/T3Toolbox), which implements the
complementary side of the T4S method).

## Status

First public release (`2026.0.0`). Implemented and validated end-to-end: the **symbolic
differentiation engine** (Algorithm 1) and the **numeric driver** (Algorithm 2), the
`ImplicitProblem` interface with a numpy reference implementation, a **FEniCS/DOLFINx hook**, a
**JAX hook** (Taylor-mode automatic differentiation), and linear input/output composition.

## How to cite

If you use this package, please cite the paper it implements:

> Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026). *Tucker Tensor Train Taylor Series*.
> arXiv preprint arXiv:2603.21141.

```bibtex
@article{alger2026t4s,
  title   = {Tucker Tensor Train Taylor Series},
  author  = {Alger, Nick and Christierson, Blake and Chen, Peng and Ghattas, Omar},
  journal = {arXiv preprint arXiv:2603.21141},
  year    = {2026},
}
```

To cite the software itself, see [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository"
also reads it).

## Authors

* Blake Christierson (bechristierson@utexas.edu)
* Nick Alger (nalger225@gmail.com)
