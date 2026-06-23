# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Worked example (JAX): the input -> output Taylor series of a *trained* Deep Equilibrium Model.

`examples/jax_deq.py` probes a DEQ's output against its recurrent **weights** (a training-time view).
This companion probes the more natural deployment-time object: with the network **frozen** (all weights
fixed), how does the equilibrium output respond to the **input**? The hidden state is the equilibrium

    u = tanh(W u + U x + b),      i.e.   R(x, u) = u - tanh(W u + U x + b) = 0,

and the output is `q = C u`. Now the parameter the library differentiates is the input `x` (the weights
`W, U, b` and readout `C` are constants), so a probe is a coefficient of the **local Taylor expansion
of the trained network around an operating point `x0`** -- exactly a local surrogate of the input ->
output map. The first-order reverse probe is the output's sensitivity to each input (a saliency /
adjoint gradient); the higher-order forward probes are its curvature along chosen input directions
(relevant to input-noise propagation and robustness).

How the input enters: through the learned **input injection** `U x`, a linear map from input space into
the hidden state that is re-added at every fixed-point iteration -- so every hidden unit sees a learned
combination of the input components, and the equilibrium `u*(x)` (hence `q`) depends on `x`.

The probing code (build a problem, call `probe`, read the probes) is identical to every other example;
only what plays the role of `theta` changes.

Run:

    python examples/jax_deq_input.py
"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)   # high-order probes need float64 (set before any jax arrays)
import jax.numpy as jnp

from implicit_probing import probe
from implicit_probing import validation               # finite-difference cross-check (testing only)
from implicit_probing.jax import JaxImplicitProblem


# ====================================================================================================
# PROBLEM  --  a *trained* (frozen) Deep Equilibrium Model;  theta = the input x
# ====================================================================================================
N_U, N_IN, N_Q = 16, 4, 3                    # hidden state / input / output dimensions

# The trained network: all weights fixed. W is scaled so the fixed-point iteration is a contraction
# (spectral norm < 1, since |tanh'| <= 1), which is what makes the equilibrium well defined.
rng = np.random.default_rng(0)
W = 0.6 * rng.standard_normal((N_U, N_U)); W = jnp.asarray(W / np.linalg.norm(W, 2) * 0.6)
U = jnp.asarray(0.7 * rng.standard_normal((N_U, N_IN)))   # the input injection (input -> hidden)
bias = jnp.asarray(0.3 * rng.standard_normal(N_U))
C = jnp.asarray(rng.standard_normal((N_Q, N_U)))          # readout (hidden -> output)


def R(x, u):                                 # state residual:  R(x, u) = u - tanh(W u + U x + b)
    return u - jnp.tanh(W @ u + U @ x + bias)


def Q(x, u):                                 # observation:  q = C u  (sees the input only through u)
    return C @ u


def solve_equilibrium(x, u_init=None, iters=200, tol=1e-13):
    """Run the network to its equilibrium at input `x`: Picard iteration (a contraction here)."""
    u = jnp.zeros(N_U) if u_init is None else u_init
    for _ in range(iters):
        u_next = u - R(x, u)                 # = tanh(W u + U x + b)
        if float(jnp.linalg.norm(u_next - u)) < tol:
            return u_next
        u = u_next
    return u


x0 = jnp.asarray(rng.standard_normal(N_IN))  # the input operating point we expand around
u0 = solve_equilibrium(x0)                    # the equilibrium hidden state there
print(f"equilibrium residual ||R(x0, u0)|| = {float(jnp.linalg.norm(R(x0, u0))):.2e}")

# Freeze the problem at (x0, u0): the hook assembles A = d_u R once and serves every probe.
problem = JaxImplicitProblem(R, Q, x0, u0)

# Directions are (vector, max_power) pairs in INPUT space: two input perturbations, probed up to
# a^2 b^1. omega is the QoI covector in the output space.
a = jnp.asarray(rng.standard_normal(N_IN)); a = a / jnp.linalg.norm(a)   # unit input-space
b = jnp.asarray(rng.standard_normal(N_IN)); b = b / jnp.linalg.norm(b)   # perturbations
directions = [(a, 2), (b, 1)]
omega = jnp.asarray(np.eye(N_Q)[0])          # QoI = the first output component


# ====================================================================================================
# PROBING  (implicit_probing)  --  the whole point
# ====================================================================================================
# One call returns the forward AND reverse probe for EVERY sub-probe, keyed by the power-tuple (i, j) =
# d_s^i d_t^j q(x0 + s*a + t*b): the local Taylor expansion of the trained network's output in the
# input. Every probe is an exact linearized solve sharing the one operator A = d_u R.
forward, reverse = probe(problem, directions, omega)


# ====================================================================================================
# RESULTS  --  the local Taylor expansion of the output in the input
# ====================================================================================================
print("\nforward probes, keyed by power-tuple (i, j) = differentiation order along input dirs (a, b):")
for mu, value in sorted(forward.items()):
    note = "   (= q(x0), the output at the operating point)" if sum(mu) == 0 else ""
    print(f"  {str(mu):<8} D^{sum(mu)} q = {np.array2string(np.asarray(value), precision=4)}{note}")

# reverse[(0,0)] is the gradient of the QoI omega(q) w.r.t. the input -- the output's sensitivity to
# each of the N_IN inputs (an adjoint/saliency gradient), from a single adjoint solve.
print(f"\ninput sensitivity  d omega(q) / d x  ({N_IN} inputs), one adjoint solve:")
print(f"  {np.array2string(np.asarray(reverse[(0, 0)]), precision=4)}")


# ====================================================================================================
# verification (testing only -- the probes above are already exact; finite differences are slower and
# only approximate, used here purely to demonstrate that the probes are right)
# ====================================================================================================
print("\n(cross-check against finite differences -- for confidence only, never how you'd compute these)")

def q_end_to_end(x_np):                       # numpy input point -> numpy output, for the FD oracle
    u = solve_equilibrium(jnp.asarray(x_np))
    return np.asarray(Q(jnp.asarray(x_np), u))

a_np, b_np = np.asarray(a), np.asarray(b)
print(f"  {'(i,j)':<9}{'order':<7}{'rel err vs FD'}")
for mu in sorted(forward):
    if sum(mu) == 0 or sum(mu) > 2:           # order-3 central differences are too coarse to certify
        continue
    spec = [((a_np, b_np)[k], mu[k]) for k in range(len(mu)) if mu[k] > 0]
    fd = validation.forward_probe_by_finite_difference(q_end_to_end, np.asarray(x0), spec, h=1e-3)
    rel = np.linalg.norm(np.asarray(forward[mu]) - fd) / max(np.linalg.norm(fd), 1e-30)
    print(f"  {str(mu):<9}{sum(mu):<7}{rel:.2e}")

adj = validation.reverse_forward_adjointness(
    forward, reverse, directions, omega,
    pair_input=lambda rev, d: float(np.dot(np.asarray(rev), np.asarray(d))),
    pair_output=lambda om, fwd: float(np.dot(np.asarray(om), np.asarray(fwd))))
print(f"  reverse/forward adjointness (exact identity) max rel err: {adj:.2e}")
