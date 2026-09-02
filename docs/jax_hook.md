# The JAX hook

`implicit_probing.jax.JaxImplicitProblem` implements the
[`ImplicitProblem`](overview.md#using-it-on-your-own-problem) interface for a map whose residual `R`
and observation `Q` are ordinary JAX functions, so you can `probe` its derivatives. The hook takes
every directional partial of `R` and `Q` by **automatic differentiation** — you never hand-derive a
derivative. This page explains how it works and how to use it. (See `examples/jax_deq.py` for a
complete, validated worked example: a deep equilibrium model / fixed-point RNN.)

## Installation

JAX is a pip package, so the hook is an optional **pip extra**:

```bash
pip install -e ".[jax]"        # or: pip install "implicit_probing[jax]"
```

The numpy-only core of implicit_probing never imports JAX; only `implicit_probing.jax` does.

**Enable double precision.** High-order derivatives lose accuracy fast in float32, so set

```python
import jax
jax.config.update("jax_enable_x64", True)   # before creating any JAX arrays
```

at the top of your program. The hook works in either precision, but the probes are only as good as the
arithmetic underneath them.

## What it does (and does not) do

The class is **frozen at an expansion point** `(theta0, u0)` that *you* supply already solved: you do
the nonlinear state solve `R(theta0, u) = 0` yourself, however you like (Newton, a fixed-point
iteration, an external solver). The class never runs a nonlinear solve. It only assembles the
linearized operator `A = d_u R` once (with `jax.jacfwd`), LU-factorizes it, and turns derivative-probe
requests into AD calls.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from implicit_probing import probe
from implicit_probing.jax import JaxImplicitProblem

def R(theta, u): ...      # state residual  -> array of shape (n_u,);  u0 solves R(theta0, u) = 0
def Q(theta, u): ...      # observation     -> array of shape (n_q,)

problem = JaxImplicitProblem(R, Q, theta0, u0)
forward, reverse = probe(problem, directions, omega)   # directions: ((d, max_power), ...), d a (p,) array
```

- `R`, `Q` — callables of `(theta, u)`, where `theta` has shape `(p,)` and `u` shape `(n_u,)`. `R`
  must map to the state space (`R.out_dim == n_u`, so that `A = d_u R` is square).
- `theta0`, `u0` — the frozen point, as 1-D arrays (`u0` already solves `R(theta0, u) = 0`).
- `directions` — `(vector, max_power)` pairs of parameter-space directions (each a `(p,)` JAX array,
  or `(B, p)` for a batch of B — see [Batched probes](#batched-probes)).
- `omega` — the output functional (a `(n_q,)` covector, or `(B, n_q)` for a batch), a **per-probe**
  argument; the reverse probes differentiate `omega(q)`. Pass `omega=None` for forward probes only.

## The one recipe behind `assemble_partial_sum`: Taylor-mode `jet`

Every `PartialTerm` the driver requests is a directional mixed partial of `R` or `Q`, contracted
against the given direction vectors, optionally with one open slot. The hook computes it in three
moves:

1. **Lift each direction into the stacked variable** `w = (theta, u)`: a `theta`-direction `d` becomes
   `(d, 0)`, a `u`-direction `v` becomes `(0, v)`. The partial is then a directional derivative of the
   single-argument function `F(w) = {R, Q}(w[:p], w[p:])`.
2. **Fold one Taylor-mode jet per distinct direction.** `theta_dirs` and `u_vecs` arrive as
   `(vector, multiplicity)` pairs. The hook pushes one order-`m` jet (`jax.experimental.jet`) along a
   direction of multiplicity `m` — *not* `m` nested `jvp`s. `jet` returns derivatives directly (no
   `1/k!`), so the order-`m` series entry is exactly `D^m F[d^m]`; distinct directions are handled by
   nesting one jet per direction. This is the multiplicity payoff the `(vector, multiplicity)` encoding
   was designed for: an order-`j` directional derivative costs ~`O(j^2)` rather than the `O(2^j)` of
   nested forward mode.
3. **Open slot** — for reverse objects, one more derivative in a free direction: the hook takes the
   reverse-mode `grad` of `pairing · partial(w)` with respect to `w` and slices out the `theta`- or
   `u`-block (matching the open-slot covector the driver expects).

Integer coefficients are applied outside the AD, and the terms are summed into one vector.

## Kernel reuse: the structure-keyed `jit`

The per-term computation is a single `jax.jit`-compiled kernel, **keyed on the term's structure**
(which function `R`/`Q`, the tuple of multiplicities, the open slot) with the **direction vectors as
traced arguments**. So one compiled kernel serves every lattice node and every direction value that
share that structure — without this, XLA would recompile for each distinct direction/incremental
vector, and high-order probing would be unaffordable.

Two canonicalizations maximize that reuse, so that *structurally equivalent* partials hit one kernel:

- The driver emits each symmetric block in **canonical descending-multiplicity order** (a property of
  the `PartialTerm` contract, shared by all backends), so e.g. `a^2 b` and `c^2 d` — and the two fold
  orders of a `{2,1}` partial — collapse together.
- The hook additionally **merges the `theta` and `u` blocks** (step 1 above): because `jet`
  differentiates along whole-`w` directions, it does not matter which block a direction came from.
  This is specific to a stacked-variable AD backend (a UFL backend cannot do it), so it lives here, not
  in the driver.

**First-call compile cost.** Compiling the high-order jet kernels is a one-time XLA cost that grows
with the derivative *order* (and the complexity of `R`, `Q`) — a few to tens of seconds for an
order-3 probe — but is nearly **independent of the problem's dimensions** (it is the order that costs,
not the size). The probes themselves are then immediate, and repeated probes of the same structure
reuse the compiled kernel. For faster reruns across processes, enable JAX's persistent compilation
cache.

## Solvers

By default `A = d_u R` is LU-factorized once and reused for every probe — forward solves use the
factorization, adjoint solves use its transpose solve. For large problems, pass `forward_solver`
and/or `adjoint_solver` (callables mapping a right-hand side to a solution) to plug in your own
matrix-free / Krylov solver and preconditioner.

## Batched probes

The D probes you want at one expansion point are independent, and one at a time they are
dispatch-bound: at a small state dimension a single probe is a few milliseconds of Python and XLA
dispatch on microseconds of arithmetic. So the hook accepts a **batch**: a direction of shape `(B, p)`
or an `omega` of shape `(B, n_q)` means B independent probes at the frozen point, handled by ONE
`probe` call — one lattice walk, one multi-right-hand-side solve on the shared LU per node, and every
jet kernel `vmap`-ed over the batch.

```python
V = jnp.asarray(rng.standard_normal((B, p)))          # B directions at the same point
forward, reverse = probe(problem, [(V, 3)], omega)    # omega single: shared by every member
forward[(2,)].shape   # (B, n_q)   -- one second directional derivative per member
reverse[(2,)].shape   # (B, p)
forward[(0,)].shape   # (n_q,)     -- q(theta0): one vector, shared, NOT batched
```

The rules, all of which follow from "a batch is just another vector type; the driver never looks":

- **Broadcasting.** A single (1-D) input in a batched call is shared by every member. So one call
  can batch over directions at a fixed `omega`, over `omega` at a fixed direction (B functionals,
  one adjoint solve per node — the gradient-sketch pattern `probe(problem, [], OMEGAS)` gives B
  gradients from one solve), or over both; a mixed request `[(V, 2), (b, 1)]` with `V` batched and
  `b` single is fine.
- **What comes back batched — per key.** A result is batched, shape `(B, …)`, iff its request used
  a batched input: `forward[mu]` iff some batched direction `k` has `mu_k ≥ 1`; `reverse[mu]` iff
  `omega` is batched or such a direction exists. Hence `forward[(0, …)] = q(theta0)` is never
  batched; under direction-only batching the gradient `reverse[(0, …)]` is not either; in a mixed
  request every key with power 0 on the batched axis is single. The dicts are therefore **ragged
  across keys** — numpy-style assignment `Y[:, t] = forward[(t,)]` broadcasts the shared entries for
  free, but `jnp.stack` over keys will not.
- **2-D is always a batch**, even `B = 1`; a `(1, n_q)` row covector is a batch of one. Batched inputs
  with different B raise (the check runs inside the walk, so it surfaces after the first solves).
  Arrays of more than two dimensions raise.
- **Solve counts are per node, not per member**, and the unbatched path is untouched — an
  unbatched call runs the exact kernels it always did; batched values agree with a loop of single
  probes to round-off (`1e-16`), not bit-for-bit.
- **Compilation.** The batched kernels are the single kernels `vmap`-ed with the expansion point as a
  shared traced argument, so — as with `refreeze` — moving the point never recompiles. A new batch
  size (or a new pattern of which inputs are batched) recompiles every kernel structure the probe
  touches, and a J = 4 single-direction probe has 58 of them: tens of seconds per new B. **Batch in
  chunks of a fixed size and pad the last chunk.** Compiled executables also accumulate for the life
  of the process (each holds memory mappings, and Linux caps those at `vm.max_map_count`, 65,530 by
  default), so a long-lived process that keeps introducing new batch sizes or batching patterns can
  eventually fail to compile with "LLVM compilation error: Cannot allocate memory";
  `jax.clear_caches()` releases them.
- **Memory.** Every lattice node keeps its incremental state and adjoint alive for the walk:
  roughly `2 · (#nodes) · B · n_u · 8` bytes. Chunk accordingly; the hook does not chunk for you.
- **Custom solvers** keep their single-vector contract: `forward_solver` / `adjoint_solver` are
  applied per member (a loop), so they stay correct but do not get the multi-right-hand-side win.
- **Composition.** `MatrixOperator` handles a leading batch axis; a custom `LinearOperator` used
  with batched probes must accept `(B, n)` inputs itself.

**Do not wrap `probe` in your own outer `jax.jit` that closes over the problem.** The frozen point
is then baked into the compiled program as a constant, and after `refreeze` the stale program
silently returns the OLD point's jets. The hook's own kernels, batched or not, take the point as an
argument and are immune.

Measured on a trained deep-equilibrium model (state dim 256, 16 inputs, 8 outputs, J = 4, laptop
CPU, 8 cores), milliseconds per probe, all agreeing with the loop to `1e-15`:

| D at one point | loop | user-side `jax.vmap` over `probe` | in-hook batch (this contract) | outer `jit(vmap)` (stale after refreeze) |
|---|---|---|---|---|
| 8 | 61 | 44 | **25** | 4.9 |
| 64 | 63 | 12 | **11** | 2.8 |
| 1024 | 63 | 5.0 | **2.8** | 0.96 |

The in-hook path beats a hand-rolled vmap by 1–1.8× because its batched kernels dispatch as compiled
units while the eager steps between them stay on the fast path; the remaining gap to the outer jit is
the eager lattice walk itself, which a later "compiled probe" (the walk jitted with the point as an
argument) would close without the stale-point trap. The reproducible benchmark is T3Polynomial's
`scripts/x03_batched_probe_bench.py`; the design record is `dev/batched_probes_design.md`.

## Validating probes

Two complementary checks (both shown in the example):

- **Forward probes vs finite differences** — re-solve the equilibrium at `theta0 + sum_k s_k d_k` and
  take a tensor product of central differences (`implicit_probing.validation`). An independent ground
  truth; reliable at low order (an order-3 central difference is itself only good to ~`1e-5`).
- **Reverse probes vs ω-paired forward probes** — the exact discrete adjointness identity
  `reverse[mu] . d_k == omega . forward[mu + e_k]`, which needs no extra solves and holds to solver
  precision.

For a problem whose `R`, `Q` are polynomial, you can additionally check the hook's probes against an
exact analytic oracle at *every* order; the test suite does this against the numpy reference problem.
