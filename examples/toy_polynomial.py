# Authors: Blake Christierson and Nick Alger
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Worked example (numpy only, no FEniCS): derivative probing of the toy implicit polynomial map.

This is the smallest end-to-end use of the library, with no heavy dependencies -- a good first read.
The map is

    q(theta) = Q(theta, u(theta)),   where the state u(theta) solves   R(theta, u) = 0,

with ``R`` and ``Q`` low-degree polynomials (so every derivative is available in closed form, which is
what lets the finite-difference cross-check at the bottom be an independent ground truth). The probing
code -- build a problem, call ``probe``, read the forward/reverse probes -- is *identical* to what you
would write for a real PDE; only the problem object changes (compare ``examples/fenics_poisson.py``).

Run:

    python examples/toy_polynomial.py
"""
import numpy as np

from implicit_probing import probe
from implicit_probing import validation               # finite-difference cross-check (testing only)
from implicit_probing.reference_problems import make_toy_problem


# ====================================================================================================
# PROBLEM  --  in a real task this is your PDE; here it is the built-in polynomial toy
# ====================================================================================================
problem = make_toy_problem()        # q: R^2 -> R^2 through an implicit 3-dof state u; exact derivatives

# Directions are (vector, max_power) pairs: probe direction a up to power 2 and b up to power 1 -- i.e.
# ask for every derivative up to a^2 b^1. omega is a covector in the output space (the QoI).
a = np.array([1.0, 0.3])
b = np.array([0.4, -0.6])
directions = [(a, 2), (b, 1)]
omega = np.array([1.0, 0.0])


# ====================================================================================================
# PROBING  (implicit_probing)  --  the whole point
# ====================================================================================================
# One call returns the forward AND reverse probe for EVERY sub-probe, keyed by the power-tuple (i, j) =
# d_s^i d_t^j q(theta0 + s*a + t*b). The lower orders fall out of the same shared-operator solves, free.
forward, reverse = probe(problem, directions, omega)


# ====================================================================================================
# RESULTS  --  the useful outputs
# ====================================================================================================
print("forward probes, keyed by power-tuple (i, j) = differentiation order along (a, b):")
for mu, value in sorted(forward.items()):
    note = "   (= q(theta0))" if sum(mu) == 0 else ""
    print(f"  {str(mu):<8} D^{sum(mu)} q = {np.array2string(value, precision=4)}{note}")

# reverse[(0,0)] is the gradient of the QoI omega(q) w.r.t. theta -- a parameter-space covector from a
# single adjoint solve (the sensitivity to *every* direction at once).
print(f"\nQoI gradient  d omega(q)/d theta  =  {np.array2string(reverse[(0, 0)], precision=4)}")


# ====================================================================================================
# verification (testing only -- the probes above are already exact; finite differences are slower and
# only approximate, and are used here purely to demonstrate that the probes are right)
# ====================================================================================================
print("\n(cross-check against finite differences -- for confidence only, never how you'd compute these)")
print(f"  {'(i,j)':<9}{'order':<7}{'rel err vs FD'}")
for mu, value in sorted(forward.items()):
    if sum(mu) == 0:
        continue
    spec = [(directions[k][0], mu[k]) for k in range(len(mu)) if mu[k] > 0]   # the distinct dirs + powers
    fd = validation.forward_probe_by_finite_difference(problem.q, problem.theta0, spec)
    rel = np.linalg.norm(value - fd) / max(np.linalg.norm(fd), 1e-30)
    print(f"  {str(mu):<9}{sum(mu):<7}{rel:.2e}")

adj = validation.reverse_forward_adjointness(forward, reverse, directions, omega)
print(f"  reverse/forward adjointness (exact identity) max rel err: {adj:.2e}")
