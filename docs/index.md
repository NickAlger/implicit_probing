<!-- Authors: Nick Alger and Blake Christierson -->
<!-- Copyright: MIT License (2026) -->
<!-- Github: https://github.com/NickAlger/implicit_probing -->

# implicit_probing

Forward and reverse **derivative probes of implicitly-defined maps**. For a parameter-to-output map

```
q(theta) = Q(theta, u(theta)),     where the state u(theta) solves     R(theta, u) = 0,
```

`implicit_probing` computes the forward and reverse probes of the higher-derivative tensors
`D^j q(theta0)` — the contractions against direction vectors that serve as training data for a local
Taylor surrogate. It is a clean, standalone implementation of **Algorithms 1 and 2 of Section 4** of
the T4S paper (Alger, Christierson, Chen & Ghattas, 2026), and the probing machinery depends on
nothing but the standard library (numpy enters only with a concrete problem).

New here? Start with the [Overview](overview.md) for the tour, then the
[API reference](api/index.rst) for per-object detail.

```{toctree}
:maxdepth: 2
:caption: Guide

overview
composition
fenics_hook
jax_hook
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```
