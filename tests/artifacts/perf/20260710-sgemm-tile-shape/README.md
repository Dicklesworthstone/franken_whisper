# tile_shape load imbalance (bit-exact) — MEASURED, then GATE-CORRECTED 2026-07-11

> **CORRECTION 2026-07-11 (same day, cc_fw) — the e2e projection below is a GREP-THE-GATE
> error; the 1.089× LAYER number is real but OFF the default hot path. Read this first.**
>
> The admissible sweep measured `ft_kernel_cpu::matmul_tensor_contiguous_f32` (the f32 2-D
> tiler) at the turbo linear shapes, and the harness projected e2e via a baked-in "linear GEMMs
> = 72.8% of encoder_window". **That 72.8% predates the default int8 encoder and is now false.**
> The DEFAULT large-v3-turbo / tiny.en engine takes the `enc_attn_out_i8i32_for` branch
> (`src/native_engine/encoder.rs:507-523`, default-ON for calibrated models), which quantizes
> **every** encoder linear — q/k/v/fc1/fc2 to i7 maddubs and attn.out to i8×i32 — and its own
> comment states it **"yields 0 f32 GEMMs/layer"**. `enc_linear` (encoder.rs:1036) dispatches
> `Some(w_i7) => nn::matmul_bias_i7` (int8); only the `None` fallback reaches
> `nn::matmul_bias → matmul → matmul_tensor_contiguous_f32` → `sgemm_2d_parallel` → `tile_shape`.
>
> **So in the default turbo engine the 32 encoder linears NEVER call the function this lever
> optimizes.** The only default-path f32 GEMM that still hits the 2-D tiler is **conv2**
> (`[~3000, 3840]×[3840, 1280]`, k=3840 ∈ (1024, 8192], one op PER WINDOW — not per-layer),
> whose share of the encoder is ~1-3%. tile_shape's true default-turbo e2e impact is therefore
> **~0.1-0.3% (conv2 only), NOT the ~5.6% projected below.**
>
> **CONSEQUENCE:** the "flip `set_sgemm_tile_balanced` default ON" recommendation is
> **DOWNGRADED** — it would not move default turbo/tiny.en e2e. The 1.089× is only realizable
> on configs that run f32 encoder linears: `FRANKEN_WHISPER_ENC_INT8`-off + a non-calibrated
> (unknown) model, or `FW_ENC_ATTN_OUT_I8I32=0`. Not the shipped deployment. The bit-exactness
> and the function-level 1.089× stand; only the e2e attribution was wrong.
>
> **LESSON (project_grep_the_gate, again):** I trusted the harness's baked-in linear-fraction
> instead of grepping which branch the default engine takes. The int8 encoder made the f32
> linear layer dead. Always confirm the DEFAULT dispatch reaches the function under test.
>
> ---
>
> **UPDATE 2026-07-11 — ADMISSIBLE PERF CAPTURED; lever is no longer blocked.**
> The sole missing piece (a 32-thread perf A/B on a host with `available_parallelism >= 32`)
> is now measured on **thinkstation1 (64 cores)**: built remotely via `rch` (vmi1227854), run
> LOCALLY at T=32. Raw table: `20260711-thinkstation1-admissible-sweep.txt`. Replica-faithful —
> the real f32 path (`sgemm_2d_parallel`, lib.rs:951) reads B **unpacked with stride n**, exactly
> like the harness `tiled()` arm, and real-A (5x7)=3.68ms sits naturally between the swept 4x8 and
> 8x4 grids (no systematic offset).
>
> **RESULT.** The naive uniform **4x8** grid (already gated behind `set_sgemm_tile_balanced`,
> frankentorch `86a54f1a`, default OFF) is a clean bit-exact win on ALL THREE turbo shapes here:
> qkv 1.008x (~neutral), **fc1 1.143x**, **fc2 1.126x** => **layer 1.089x => projected e2e ~1.056x**.
>
> **THE PARKED WORRY IS FALSIFIED ON THE TARGET BOX.** The 5975WX baseline had 4x8 *regress* fc1
> (0.958x); on thinkstation1 fc1 4x8 = 1.143x, a WIN. The fc1 regression was host-specific to the
> 5975WX. There is no fc1 catch here.
>
> **THE B-BYTES-AWARE SELECTOR IS MARGINAL HERE.** The refined selector rule (largest divisor
> `q <= T/2` with `ceil(n/q) >= MIN_BLOCK_COLS`; the `p>=2` clause avoids the p=1 A-locality cliff
> that made a pure minimize-B rule overshoot fc1 to 1x32) DOES pick the measured per-shape optimum
> (qkv/fc2 4x8, fc1 2x16), but that optimum is only **1.098x** vs naive's **1.089x** — a ~0.8%
> layer delta, entirely from fc1 (2x16 1.176x vs 4x8 1.143x), and that 2.9% fc1 gap is within ~1 cv
> (4%) of the noise. NOT worth a shape-dependent policy on a shared, shipped crate.
>
> **RECOMMENDATION (owner-gated, cross-repo shared crate — surface, do not flip unilaterally):**
> flip `set_sgemm_tile_balanced` default to ON. Bit-exact, fc1-regression concern does not
> reproduce on the target box, clean +1.089x layer (~1.056x e2e). Residual risk is only the
> unrelated FMA/golden-f32 shared-crate concern in `project_isa_baseline`.
>
> ---
>
> **UPDATE 2026-07-10 — the patch below is SUPERSEDED by a landed runtime knob.**
> `frankentorch 86a54f1a` adds `set_sgemm_tile_balanced(bool)` (default **off** = historical
> grid, bit-exact either way), so the policy no longer needs a source patch to evaluate — a
> bench can flip it inside one binary and time the **real** `matmul_tensor_contiguous_f32`
> both ways. Keep the patch only as a record of the minimal source change.
>
> The A/B was rebuilt on the real function (`benches/sgemm_tile_shape.rs`, with an
> `exercise_proof` that panics unless flipping the flag changes the grid). **Keep gate NOT
> MET:** on `hetzner2` (16 hw threads) cv(paired ratio) is 16.8–24.2%, and the NULL CONTROL
> — where both policies yield the *same* grid — itself fails at 6.8–17.7% with a systematic
> +2–3% bias. The host cannot satisfy cv<5 at any effect size. Default stays OFF.
>
> Mechanism confirmed anyway at T=14 (15 tiles on 14 threads): fc1 **1.689×**, fc2 **1.686×**,
> 23–24/25 paired wins (sign test p≈1e-4), matching `ceil(15/14)·14/15 = 1.87×`. qkv 0.946× —
> the balanced grid is **not** uniformly good, mirroring fc1's regression at T=32.

**Status: PARKED, patch prepared and validated, NOT applied.** Blocked on the absence of
any box where an admissible 32-thread perf A/B can run. Filed by `cc_fw` 2026-07-10.

Apply against `frankentorch` (validated with `git apply --check` at `1fb80836`):

```
cd /data/projects/frankentorch
git apply /data/projects/franken_whisper/tests/artifacts/perf/20260710-sgemm-tile-shape/tile_shape_balanced.patch
```

---

## The defect (certain, no measurement needed)

`gemm::tile_shape` chooses the 2-D tile grid as

```rust
let p = (threads as f64).sqrt().floor().max(1.0) as usize; // M-blocks
let q = threads.div_ceil(p);                               // N-blocks
```

`p * q != threads` for many thread counts. **At `T = 32`: `p = 5`, `q = 7` ⇒ 35 tiles on
32 threads** — a straggler wave of 3 tiles runs while 29 threads idle. `T = 32` is exactly
franken_whisper's encoder thread cap (`project_encoder_thread_cap_win`), so this is the
case that matters. `T = 64 → 8×8` and `T = 16 → 4×4` are already balanced and do not move.

The patch picks `p` = the largest **divisor** of `T` that is `≤ sqrt(T)`, so `p*q == T`
exactly (32 → 4×8).

> **Polarity differs from what shipped.** This *patch* defaults the balanced grid **ON**
> (`FT_SGEMM_TILE_BALANCED=0` disables). The *landed knob* (`86a54f1a`) defaults **OFF**
> (`FT_SGEMM_TILE_BALANCED=1` enables, or `set_sgemm_tile_balanced(true)`), because the
> keep gate has not been met. Prefer the knob; use this patch only as a record of the
> minimal source change.

**Bit-exact.** Each output element's full k-accumulation happens inside one serial
micro-kernel call; neither the row nor the column count changes that order (same invariant
as `gemm_row_split_matches_single_bit_exact`). Verified empirically on two independent rch
workers by `crates/ft-kernel-cpu/examples/sgemm_tile_shape_ab.rs`: the balanced grid is
bit-identical to the shipped path on all three turbo shapes. **So this lever needs no
transcript/WER gate** — only a perf number.

---

## Baseline already captured (32-core Zen3, AMD 5975WX, 32 rayon threads, x86-64-v3)

Measured **before** the local-build directive, interleaved, arm order rotated every rep,
min-of-9, cv ≤ 5%. `A` = real `matmul_tensor_contiguous_f32_into`; grid arms are
`matrixmultiply` in an explicit grid, bit-exact vs `A`.

| shape | A (ft, 5×7) | balanced 4×8 | ratio |
|---|---|---|---|
| turbo qkv/out `[1500,1280]×[1280,1280]` | 4.10 ms | — | **1.022×** |
| turbo fc1 `[1500,1280]×[1280,5120]` | 16.12 ms | — | **0.958×** ← REGRESSES |
| turbo fc2 `[1500,5120]×[5120,1280]` | 17.00 ms | — | **1.241×** |

Layer total (4×qkv/out + fc1 + fc2): **49.52 → 46.57 ms = 1.063×**
⇒ projected encoder **1.045×** ⇒ **e2e ≈ 1.04×**.

### The catch — read before landing

**A uniform balanced grid is NOT a clean win: fc1 regresses (0.958×).** The straggler
count is not the only force. More column blocks shrink the `B` slice each thread must
re-stream (`k·nb·4` bytes), which fc1 wants (`B` = 26.2 MB, `n` = 5120) and fc2 does not
(`n` = 1280, and at `q=16` `nb` clamps to `MIN_BLOCK_COLS=128`, collapsing to 20 tiles and
under-filling the pool). Grid sweep on the same box, bit-exact throughout:

| shape | 32×1 | 16×2 | 8×4 | 4×8 | 2×16 |
|---|---|---|---|---|---|
| qkv/out | 0.881× | 0.943× | **1.016×** | 1.001× | 0.972× |
| fc1 | 0.778× | 0.970× | 0.805× | 0.958× | **1.085×** |
| fc2 | 0.947× | 1.066× | 1.086× | **1.111×** | 1.063× |

**Therefore the right fix is not "balance the tile count".** It is to choose among the
divisor pairs `(p,q)` of `T` the one minimizing per-thread `B` bytes (`k·nb·4`) subject to
`nb ≥ MIN_BLOCK_COLS` and `p·q` tiles actually filling the pool. The attached patch is the
*minimal, safe* first step (balance only); the B-bytes-aware selector is the real lever and
is unmeasured. Do not land the attached patch and claim 1.24× — the honest number for it is
**1.063× on the layer**, and it costs fc1 4%.

---

## BLOCKER — why this is parked

A perf number is the only thing missing, and there is currently nowhere to produce one.

1. **Local `cargo build/bench/test` suspended by owner** — `/data` at 96% (90 G free);
   11 concurrent local cargo builds were draining ~450 GB/h.
2. **No sampled `rch` worker can field 32 threads.** `rch` is healthy (28/28, 12 workers)
   and runs pure-cargo tests fine, but this example landed on three different workers:

   | worker | `available_parallelism` | verdict |
   |---|---|---|
   | `vmi1264463` | 8 | NOT ADMISSIBLE (4× oversubscribed) |
   | `ovh-a` (`fixmydocuments`) | 16 | NOT ADMISSIBLE (2× oversubscribed) |
   | `hz2` | *not captured* | cannot be certified |

   Forcing a 32-thread pool on a smaller host makes rayon work-steal across the
   oversubscription, which **smears the very 35-vs-32 straggler being measured**. Worker
   core count is not selectable (`rch exec` has no `--worker` / `--min-cores` flag), so
   retrying does not help.
3. **`rch exec -- maturin build` fails open to a LOCAL build** (no strict-remote flag;
   observed 18 MB offload then a 179 MB local `release/` tree). Not used here, but it is why
   no build fallback exists.

### Two measurement rules this cost us

**(a) Ratios are NOT worker-invariant, so a ratio from one `rch exec` may never be compared
against a ratio from another** (franken_networkx `br-r37-c1-839yx`). The three runs above
reported layer ratios of 1.060× / 0.981× / 1.018×. That spread is *expected* under worker
non-invariance and carries **no information** — it is not evidence of anything, and an
earlier draft of this note wrongly cited the disagreement as proof of inadmissibility. The
actual reason each run is inadmissible is the core count in the table. The correct substrate
— which `sgemm_tile_shape_ab` already uses — is **both arms in ONE binary and ONE
invocation**, order rotated per rep, so host identity and drift cancel *within* a run. Never
use the forbidden `stash ORIG / bench / pop / bench NEW` recipe: it assumes one machine and
it uses `git stash`.

**(b) Single-binary is necessary but NOT sufficient.** Generalizing `PERF_LEDGER`'s
"only same-worker single-`rch exec` A/B is admissible": the host must **also** have
`available_parallelism ≥ the thread count under test`. A perfectly interleaved single-process
A/B on an 8-core box cannot measure a 32-thread scheduling effect. The example now prints an
explicit `PERF *** NOT ADMISSIBLE ***` verdict in that case — **do not quote a run that
emitted it.** Bit-exactness, by contrast, *is* host-independent and is certified by every run.

## Unblock (any one)
- Free disk headroom on the 5975WX box, then run
  `cargo run --release -p ft-kernel-cpu --example sgemm_tile_shape_ab` there (32 threads).
- Give `rch` a worker-capability filter (`--min-cores 32`) so the example lands on a big box.
- Run it on any host with ≥ 32 physical cores and paste the table.

Then implement the **B-bytes-aware** divisor selection rather than the naive balance, and
re-sweep the grid table above.

## Related
- `frankentorch 8e3e7c9d` — `F32_2D_TALL_MAX_K` 1536→8192 (landed, bit-exact, 1.057× on fc2).
  That change is what routes fc2 through the 2-D tiler at all, so it is a prerequisite for
  this lever mattering.
- `frankentorch 1fb80836` — `set_sdpa_poly_exp()` setter.
- Beads: `bd-transcript-gate-unrunnable-xu9g` (the umbrella measurement blocker).
- Harness: `frankentorch crates/ft-kernel-cpu/examples/sgemm_tile_shape_ab.rs` (carries a
  replication-fidelity arm `B`; a prior dig drew a false conclusion from a wrong replication).
