# Minimal jax-only reproduction of the batched reverse-jet failure (2026-09-21); see
# dev/jet_vmap_grad_nan_2026_09_21.md. Run: python dev/jet_vmap_grad_nan_repro_2026_09_21.py
"""jax-only reproduction: jit(vmap(grad(pairing . D F[d])))[:p] where F is linear in w[:p] (so the
theta-gradient is a structural zero) and contains an (N, N) broadcast-and-reduce of w[p:]."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from jax.experimental import jet
print('jax', jax.__version__, jax.devices())
N, p = 64, 16; n_u = N + 1; n_w = p + n_u
rng = np.random.default_rng(0)
y = (np.arange(N) + 0.5) / N
C = jnp.asarray((y[:, None] - y[None, :]) ** 2); eps = (4.0 / N) ** 2
Bm = jnp.asarray(rng.standard_normal((N, p)) / N)
def F(w):                                                     # (n_u,): linear in w[:p]
    g = w[p:p + N]
    return jnp.concatenate([Bm @ w[:p] - jnp.exp((g[None, :] - C) / eps).sum(axis=0), jnp.zeros(1)])
def d_jet(fn, d):  return lambda x: jet.jet(fn, (x,), ([d],))[1][0]
def d_jvp(fn, d):  return lambda x: jax.jvp(fn, (x,), (d,))[1]
def make(deriv):
    def impl(w, d, pr):
        return jax.grad(lambda x: jnp.dot(pr, deriv(F, d)(x)))(w)[:p]
    return jax.jit(jax.vmap(impl, in_axes=(None, 0, 0))), jax.jit(impl)
w0 = jnp.concatenate([jnp.asarray(rng.standard_normal(p)), 1e-3 * jnp.asarray(rng.standard_normal(N)), jnp.zeros(1)])
for name, deriv in (('jet', d_jet), ('jvp', d_jvp)):
    batched, single = make(deriv)
    out = []
    for B in (32, 63, 64, 100, 128):
        dv = jnp.asarray(rng.standard_normal((B, n_w))); pr = jnp.asarray(rng.standard_normal((B, n_u)))
        vb = np.asarray(batched(w0, dv, pr)); vs = np.asarray(single(w0, dv[0], pr[0]))
        ok = np.isfinite(vb).all() and np.abs(vb).max() <= 1e-12
        out.append(f'B={B} {"ok " if ok else "BAD"}(max|batched| {np.abs(vb).max():.0e}, |single| {np.abs(vs).max():.0e})')
    print(f'{name}: ' + ' | '.join(out), flush=True)
