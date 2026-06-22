**WORK IN PROGRESS — DO NOT USE**

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

> Alger, Christierson, Chen & Ghattas (2026), *Tucker Tensor Train Taylor Series*.

Each probe reduces to a set of linearized solves that all share the same operator (the linearized
state operator `A = d_u R`, or its adjoint) and differ only in their right-hand sides. The work is
organized as a traversal of the lattice of multiset-subsets of the probing directions, so that
high-order probes reuse all of their lower-order sub-probes.

## Scope

`implicit_probing` is **laser-focused on the derivative machinery** and depends on nothing but
`numpy`. Its probe output is plain arrays and functionals, with no particular tensor format baked in,
so it can feed any downstream consumer (for example the Tucker-tensor-train *fitting* machinery in the
sibling package [T3Toolbox](https://github.com/NickAlger/T3Toolbox), which implements the
complementary side of the T4S method).

## Status

Early development. Implemented so far: the **symbolic differentiation engine** (Algorithm 1) — the
pure-Python core that generates the probe expansions. The numeric driver (Algorithm 2), a problem
interface, and autodiff-framework hooks are forthcoming.

## Authors

* Nick Alger (nalger225@gmail.com)
* Blake Christierson (bechristierson@utexas.edu)
