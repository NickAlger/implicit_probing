# Authors: Nick Alger and Blake Christierson
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

from implicit_probing import Multiset, subset_lattice, probe
from implicit_probing import validation               # finite-difference cross-check (testing only)
from implicit_probing.reference_problems import make_toy_problem


# ====================================================================================================
# PROBLEM  --  in a real task this is your PDE; here it is the built-in polynomial toy
# ====================================================================================================
problem = make_toy_problem()        # q: R^2 -> R^2 through an implicit 3-dof state u; exact derivatives

# Probe directions are a multiset of *labels* plus the parameter-space vector each label stands for.
# omega is a covector in the output space -- the quantity of interest the reverse probes differentiate.
directions = {1: np.array([1.0, 0.3]), 2: np.array([0.4, -0.6])}
omega = np.array([1.0, 0.0])


# ====================================================================================================
# PROBING  (implicit_probing)  --  the whole point
# ====================================================================================================
# One call returns the forward AND reverse probe for EVERY sub-multiset of alpha: the lower-order
# probes fall out of the very same shared-operator linear solves, at no extra cost.
alpha = Multiset([1, 1, 2])         # probe the 3rd derivative in directions (d1, d1, d2)
forward, reverse = probe(problem, alpha, directions, omega)


# ====================================================================================================
# RESULTS  --  the useful outputs
# ====================================================================================================
print("forward probes  D^|beta| q(theta0) applied to beta's directions:")
for beta in subset_lattice(alpha):
    label = str(beta) if len(beta) else "{} (= q(theta0))"
    print(f"  order {len(beta)}  {label:<16} {np.array2string(forward[beta], precision=4)}")

# reverse[empty] is the gradient of the QoI omega(q) w.r.t. theta -- a parameter-space covector got
# from a single adjoint solve (it gives the sensitivity to *every* direction at once).
print(f"\nQoI gradient  d omega(q)/d theta  =  {np.array2string(reverse[Multiset([])], precision=4)}")


# ====================================================================================================
# verification (testing only -- the probes above are already exact; finite differences are slower and
# only approximate, and are used here purely to demonstrate that the probes are right)
# ====================================================================================================
print("\n(cross-check against finite differences -- for confidence only, never how you'd compute these)")
print(f"  {'beta':<13}{'order':<7}{'rel err vs FD'}")
for beta in subset_lattice(alpha):
    if len(beta) == 0:
        continue
    spec = [(directions[k], count) for k, count in beta.items()]   # {d_k^{count}}
    fd = validation.forward_probe_by_finite_difference(problem.q, problem.theta0, spec)
    rel = np.linalg.norm(forward[beta] - fd) / max(np.linalg.norm(fd), 1e-30)
    print(f"  {str(beta):<13}{len(beta):<7}{rel:.2e}")

adj = validation.reverse_forward_adjointness(forward, reverse, alpha, directions, omega)
print(f"  reverse/forward adjointness (exact identity) max rel err: {adj:.2e}")
