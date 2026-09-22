# Batched reverse jets: `jax.experimental.jet` under `vmap ∘ grad ∘ jit` returns garbage (2026-09-21)

*Found in T3Polynomial's entropic-OT example (`dev/ot_design_2026_09_21.md` §8 there; repro
script `scripts/x08_batched_reverse_jets_bug.py`). Reported here as an upstream item for the JAX
hook; **no library code changed** — Nick's call. jax 0.10.2, CPU, float64.*

## Symptom

The eager batched probe (`probe` on a `JaxImplicitProblem` with a `(B, p)` direction batch and a
`(B, n_q)` omega batch — the `_term_value_batched` = `jit(vmap(grad ∘ jet))` kernel) returns inf /
NaN / garbage of magnitude 1e100–1e285 for the **reverse jets of order ≥ 2** (`reverse[(1,)]` and
up) once the batch is large enough: on the OT problem (p = 16, n_u = 65, n_q = 64) from B = 64;
at n_q = 48 from somewhere in 65..100; at n_q = 40 not up to 100; empirically B·n_q ≳ 4096.
Forward jets, `reverse[(0,)]`, batches ≤ 32, single-direction probes and the **compiled probe**
(`compiled_probe`, the same terms inlined into one program) are all correct to 1e-9. x03 (DEQ,
n_q = 8, B up to 1024) never saw it: the trigger needs a residual shape the DEQ lacks (below).

## What it is not

Not arithmetic (three log-sum-exp formulations and exponent clipping fail alike; blur width
irrelevant); not XLA runtime flags (`--xla_cpu_enable_fast_math=false`, `--xla_cpu_use_thunk_runtime=false`,
`--xla_cpu_multi_thread_eigen=false`, `taskset -c 0`: all unchanged); not async dispatch or buffer
lifetime (`block_until_ready()` after every kernel and solve, or a host round-trip of every value:
unchanged); not the output slice (`grad_w[p:]` replaced by a selection matmul: unchanged); not
the hook's lifting / orientation logic (single kernel with identical inputs is right).
Deterministic per compiled program (identical garbage across repeats in one process; different
garbage across processes/configurations) — a miscompile, not a race.

## Localization

Instrumenting `_term_value_batched` to compare each batched term with its single twin (rows 0,
B−1): at B = 63 all 43 term calls of a J = 2 probe agree to 1e-12; at B = 64 the **first wrong
term** is the R-term with `mults=(1,)`, `open_slot='theta'`, one batched direction `(64, 81)` and a
batched adjoint-incremental pairing `(64, 65)` — ∂_θ(v̂ᵀ D_w R[d]). Its exact value is
**identically zero** (the OT residual is linear in θ with a constant matrix), the single kernel
returns exactly 0, the batched kernel returns 8.8e242. Reducing R with that kernel standalone:
any **(N, N) broadcast-and-reduce of the state** inside the jet (`exp((g[None,:] − C)/ε).sum(0)`,
with or without the log, with or without the bordered row) triggers it; `tanh(A g)` does not.

**Graph sensitivity (2026-09-22).** Rewriting the OT residual's linear θ-term from
``nu0 + B @ x`` (B = nu0-scaled modes) to ``nu0 * (1 + G @ x)`` — the same function to
round-off — makes the eager batched kernel correct at every B on the full-size problem
(`T3Polynomial/scripts/x08` now reports ok throughout). The jax-only repro below, which uses the
``Bm @ w[:p]`` form, still fails. So the trigger is a specific lowering of "matvec of the θ-block
feeding an add" with a structural-zero cotangent, not the residual's mathematics; a report to jax
should include both forms.

## Minimal jax-only reproduction (`dev/jet_vmap_grad_nan_repro_2026_09_21.py`)

    F(w) = concat([Bm @ w[:p] − exp((w[p:p+N][None, :] − C)/ε).sum(0), 0])      # linear in w[:p]
    kernel(w, d, ω) = grad_w( ω · jet_1(F)(w)[d] )[:p]                            # exact answer: 0
    jit(vmap(kernel, in_axes=(None, 0, 0)))(w0, D, Ω):   B = 32 ok | 63 NaN | 64 NaN | 100 NaN | 128 ok
    the same kernel with jax.jvp in place of jet.jet:    ok at every B

So: a defect of `jax.experimental.jet` composed with `vmap`, `grad` and `jit`, surfacing where the
reverse pass carries a structural zero through a broadcast-and-reduce, above a size that looks
like an XLA buffer/fusion threshold. Worth a jax issue with the script above.

## Workaround, measured on the full-size OT problem (N = 64, n = 16, J = 2 and 4; B = 64, 100)

`_deriv_along` rewritten as **nested `jax.jvp`** (order-k directional derivative = k nested
jvps along the same direction; `jet` nowhere):

| `_deriv_along` | J = 2, B = 64 / 100 | J = 4, B = 64 / 100 | warm wall, J = 4, B = 64 / 100 |
|---|---|---|---|
| jet (current) | garbage / garbage | garbage / garbage | 1.5 s / 2.5 s |
| jvp for order 1 only, jet above | garbage / garbage | garbage / garbage | — |
| **nested jvp at every order** | **ok (5e-12) / ok (5e-12)** | **ok (2e-9) / ok (2e-8)** | **0.7 s / 1.2 s** |

(errors = max relative deviation from single-direction jet probes over orders 0..J). The
higher-multiplicity jet terms are broken too (order-1-only jvp does not suffice). At J ≤ 4 the
nested-jvp kernels are not slower here — the O(j²)-vs-O(2^j) argument for `jet` (docs/jax_hook.md)
does not bite at these orders; compile times were comparable (37 s vs 45 s at J = 4).

## Proposed direction (revised after Nick's cost objection, 2026-09-21)

**Ruling (Nick, 2026-09-21): option 2 is the default** — the compiled probe is the production path for
batched JAX probes. T3Polynomial's datagen already defaults to it (`use_compiled=None` → auto).
What it means here (still to do): document it as the recommended batched path in `docs/jax_hook.md`
(the eager batched kernel stays for small B / varying patterns), and consider a guard on the eager
path (warn or refuse at B·n_q ≳ 4096 for JAX problems) until the jet rule is found. Items 1, 4, 5
remain open; 3 is dropped as a default.

Nested jvp is a defensible stopgap at J ≤ 4 and a poor library default: its traced graph grows as
2^j against jet's j² (16 vs 16 at J = 4 — hence the tie above — but 64 vs 36 at J = 6 and 256 vs
64 at J = 8), XLA's compile passes are superlinear in program size, and the number of distinct
eager kernels grows with J too (58 at J = 4, 118 at J = 6). No measurement exists above J = 4.

1. **Find the responsible jet rule and keep jet.** The repro fails with exp of a broadcast +
   reduce inside the jet and passes with tanh of a matvec. Swap exp for a polynomial in the repro
   to separate the exp rule from the broadcast/reduce rules; if one rule is at fault, vendor a
   patched `jet` into the hook (it is a small pure-Python module) and/or report upstream. This
   keeps the j² cost. First thing to do.
2. **Compiled probe as the production path for batched JAX probes** (immune, and the faster path
   per §13's measurements; T3Polynomial's datagen already uses it). Its high-order compile cost
   (56 s at J = 6, minutes for two-direction patterns) is the jet-based cost already accepted.
3. **Nested jvp as an opt-in only** (J ≤ 4, eager path, varying patterns).
4. Add the test regardless: batched vs single probes at B ≥ 64 on a residual linear in θ with a
   broadcast-and-reduce (the repro's F); check `np.isfinite`.
5. Measure J = 6 compile and warm time for jet vs nested jvp on the OT problem (a few minutes, not
   yet run — needs a machine with no other JAX compile in flight).
