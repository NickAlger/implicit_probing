# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/implicit_probing
"""Worked example (JAX): derivative probing of a Deep Equilibrium Model (a fixed-point RNN).

A deep equilibrium model (Bai, Kolter & Koltun 2019) is a recurrent network run to its fixed point:
the hidden state ``u`` is the equilibrium of one layer applied to itself,

    u = tanh(W u + U x + b),      i.e.   R(theta, u) = u - tanh(W u + U x + b) = 0,

with output ``q = C u``. Here the recurrent weights ``W`` are the parameter ``theta`` (flattened); the
input ``x``, input weights ``U``, bias ``b`` and readout ``C`` are fixed. So this is exactly the
library's setting ``q(theta) = Q(theta, u(theta))`` with ``u`` defined *implicitly* by an equation that
must be solved -- and a high-order probe is the **Taylor expansion of the equilibrium output in the
recurrent weights**, the sensitivity/curvature of the learned fixed point.

Why JAX: ``R`` and ``Q`` are written as ordinary JAX functions and the hook takes all their
directional partials by automatic differentiation (Taylor-mode ``jet``) -- no hand-derived derivatives
of the nested ``tanh``. The probing code below (build a problem, call ``probe``, read the probes) is
*identical* to the numpy and FEniCS examples; only the problem object changes.

This is the training-time view (output vs the weights). For the deployment-time companion -- the
input -> output map of the same network frozen -- see ``examples/jax_deq_input.py``.

The first probe spends a few seconds compiling the high-order AD kernels (one-time XLA warm-up, nearly
independent of the network size -- it is the derivative *order* that costs, not the dimensions); the
probes themselves are then immediate.

Run:

    python examples/jax_deq.py
"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)   # high-order probes need float64 (set before any jax arrays)
import jax.numpy as jnp

from implicit_probing import probe
from implicit_probing import validation               # finite-difference cross-check (testing only)
from implicit_probing.jax import JaxImplicitProblem


# ====================================================================================================
# PROBLEM  --  a Deep Equilibrium Model;  theta = the recurrent weights W (flattened)
# ====================================================================================================
N_U, N_IN, N_Q = 16, 4, 3                     # hidden state / input / output dimensions

# Fixed (non-probed) pieces of the network: input, input weights, bias, linear readout.
rng = np.random.default_rng(0)
x_in = jnp.asarray(rng.standard_normal(N_IN))
U = jnp.asarray(0.7 * rng.standard_normal((N_U, N_IN)))
bias = jnp.asarray(0.3 * rng.standard_normal(N_U))
C = jnp.asarray(rng.standard_normal((N_Q, N_U)))

# The expansion point: recurrent weights W0, scaled so the fixed-point iteration is a contraction
# (spectral norm < 1, since |tanh'| <= 1), and flattened into the parameter vector theta0.
W0 = 0.6 * rng.standard_normal((N_U, N_U))
W0 = W0 / np.linalg.norm(W0, 2) * 0.6        # ||W0||_2 = 0.6  ->  comfortably contractive
theta0 = jnp.asarray(W0.reshape(-1))         # p = N_U * N_U = 9


def R(theta, u):                             # state residual:  R(theta, u) = u - tanh(W u + U x + b)
    W = theta.reshape(N_U, N_U)
    return u - jnp.tanh(W @ u + U @ x_in + bias)


def Q(theta, u):                             # observation:  q = C u  (sees the weights only through u)
    return C @ u


def solve_equilibrium(theta, u_init=None, iters=200, tol=1e-13):
    """Run the RNN to its equilibrium: Picard iteration u <- tanh(W u + U x + b) (a contraction here)."""
    u = jnp.zeros(N_U) if u_init is None else u_init
    for _ in range(iters):
        u_next = u - R(theta, u)             # = tanh(W u + U x + b)
        if float(jnp.linalg.norm(u_next - u)) < tol:
            return u_next
        u = u_next
    return u


u0 = solve_equilibrium(theta0)               # the equilibrium hidden state at the expansion point
print(f"equilibrium residual ||R(theta0, u0)|| = {float(jnp.linalg.norm(R(theta0, u0))):.2e}")

# Freeze the problem at (theta0, u0): the hook assembles A = d_u R once and serves every probe.
problem = JaxImplicitProblem(R, Q, theta0, u0)

# Directions are (vector, max_power) pairs in weight space: two weight perturbations, probed up to
# a^2 b^1 (every derivative through a^2 b). omega is the QoI covector in the output space.
a = jnp.asarray(rng.standard_normal(theta0.shape[0])); a = a / jnp.linalg.norm(a)   # unit weight-space
b = jnp.asarray(rng.standard_normal(theta0.shape[0])); b = b / jnp.linalg.norm(b)   # perturbations
directions = [(a, 2), (b, 1)]
omega = jnp.asarray(np.eye(N_Q)[0])          # QoI = the first output component


# ====================================================================================================
# PROBING  (implicit_probing)  --  the whole point
# ====================================================================================================
# One call returns the forward AND reverse probe for EVERY sub-probe, keyed by the power-tuple (i, j) =
# d_s^i d_t^j q(theta0 + s*a + t*b). Every probe is an exact linearized solve sharing the one operator
# A = d_u R (the implicit-function-theorem differentiation of the fixed point, to all orders).
forward, reverse = probe(problem, directions, omega)


# ====================================================================================================
# RESULTS  --  the Taylor expansion of the equilibrium output in the recurrent weights
# ====================================================================================================
print("\nforward probes, keyed by power-tuple (i, j) = differentiation order along weight dirs (a, b):")
for mu, value in sorted(forward.items()):
    note = "   (= q(theta0), the equilibrium output)" if sum(mu) == 0 else ""
    print(f"  {str(mu):<8} D^{sum(mu)} q = {np.array2string(np.asarray(value), precision=4)}{note}")

# reverse[(0,0)] is the gradient of the QoI omega(q) w.r.t. the recurrent weights -- the sensitivity of
# the equilibrium output to *every* one of the N_U*N_U weights, from a single adjoint solve. Reshape it
# back to the weight matrix and summarize (printing all weights would be unwieldy at this size).
grad_W = np.asarray(reverse[(0, 0)]).reshape(N_U, N_U)
i_max = tuple(int(k) for k in np.unravel_index(np.argmax(np.abs(grad_W)), grad_W.shape))
print(f"\nQoI gradient  d omega(q) / d W  ({N_U}x{N_U} = {theta0.shape[0]} weights), one adjoint solve:")
print(f"  Frobenius norm {np.linalg.norm(grad_W):.4f};  most sensitive weight W{list(i_max)} "
      f"= {grad_W[i_max]:+.4f}")


# ====================================================================================================
# verification (testing only -- the probes above are already exact; finite differences are slower and
# only approximate, used here purely to demonstrate that the probes are right)
# ====================================================================================================
print("\n(cross-check against finite differences -- for confidence only, never how you'd compute these)")

def q_end_to_end(theta_np):                  # numpy parameter point -> numpy output, for the FD oracle
    u = solve_equilibrium(jnp.asarray(theta_np))
    return np.asarray(Q(jnp.asarray(theta_np), u))

a_np, b_np = np.asarray(a), np.asarray(b)
print(f"  {'(i,j)':<9}{'order':<7}{'rel err vs FD'}")
for mu in sorted(forward):
    if sum(mu) == 0 or sum(mu) > 2:          # order-3 central differences are too coarse to certify
        continue
    spec = [((a_np, b_np)[k], mu[k]) for k in range(len(mu)) if mu[k] > 0]
    fd = validation.forward_probe_by_finite_difference(q_end_to_end, np.asarray(theta0), spec, h=1e-3)
    rel = np.linalg.norm(np.asarray(forward[mu]) - fd) / max(np.linalg.norm(fd), 1e-30)
    print(f"  {str(mu):<9}{sum(mu):<7}{rel:.2e}")

adj = validation.reverse_forward_adjointness(
    forward, reverse, directions, omega,
    pair_input=lambda rev, d: float(np.dot(np.asarray(rev), np.asarray(d))),
    pair_output=lambda om, fwd: float(np.dot(np.asarray(om), np.asarray(fwd))))
print(f"  reverse/forward adjointness (exact identity) max rel err: {adj:.2e}")
