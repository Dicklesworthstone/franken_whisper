# cc-lane finalization — maintenance keeps, reject index, gated remainder

Terminal record for the `cc_fw` lane (SDPA / int8-SDPA / encoder-non-GEMM / mel / VAD), 2026-07-10.
Every rejected lever below carries a **reject ID** (commit), its **null-control / measurement**, and
a **retry-condition**. The vein is at its byte-exact frontier; this is the map, not a survey.

Provenance for all e2e/self-time figures: shipped `e2e_probe` sha256
`272102fd7cd643bf449eeed18002874cc98241f74290d2937a8d606a10b0c776`, local 5975WX unless a worker is
named. Null-control discipline: paired same-invocation A/A control, with the
candidate **median** gated against the null's bootstrap 95% CI using a 2×
margin. CV is provenance only.

## 1. MAINTENANCE KEEPS (self-speedups, not campaign wins)

| lever | commit(s) | result | proof |
|---|---|---|---|
| `FT_SDPA_POLY_EXP` default-on for `large-v3-turbo` | franken `94714c1` | **1.0722× e2e** (cv 0.8%, 5/5) | transcript byte-identical 3/3, WER Δ 0.000 — `docs/PR_ft_sdpa_poly_exp_turbo.md` |
| poly-exp setter (per-model opt-in) | ft `1fb80836` | enables the flip without env (edition-2024 `#![forbid(unsafe_code)]`-safe) | — |
| 2-D tile the large-K reused-output sgemm (`F32_2D_TALL_MAX_K` 1536→8192) | ft `8e3e7c9d` | **bit-exact**, 1.057× fc2 | `gemm_2d_parallel_is_bit_exact_vs_serial` |
| decouple SDPA row-block split guard from `BR` (latent scheduler bug) + runtime `BR` knob | ft `0fef5755` | **bit-exact**, default-identical | 12/12 gemm tests |
| runtime tile-grid policy + same-binary A/B that executes the real fn | ft `86a54f1a` `e959c67e` | bit-exact, gated | admissibility guard |
| A/B methodology (ABBA harness, median-vs-null gate, ISA probes) | ft `c870a4d4` `2ba080ba` `37ee5949` `3b8cdebc` | fleet-adopted | null control 1.1163×→1.0018× |
| bd-0522 HF-token honesty fix (native diarize ungated from a token it never uses) | franken `84afe64` | benchmark-honesty prerequisite | 643 backend tests green |

## 2. REJECT INDEX (every rejected lever: ID · null-control/measurement · retry-condition)

| lever | reject ID | null-control / measurement | retry-condition |
|---|---|---|---|
| bd-4hc0 `matrixmultiply→gemm` swap ("P0, 3.75×") | `2bbff39` `fe97df1` | **FALSIFIED** — 1.00–1.07× on the REAL ft path (0.934× @16t); "3.75×" measured code the engine never runs | `ft-kernel-cpu` gains a 2-D path for k>1024 AND the target is not int8-default (franken turbo is int8) |
| SDPA **BR tile** sweep | `a410602` `efacdf8` `bbe6ed0` | median **1.0305×** inside null **[0.9384,1.0455]** (`hz2`, n=41 ABBA) | a ≥32-core rch worker to measure the production T=32 nested path |
| SDPA **flat-vs-nested rayon** | `59b77db` | median **1.0128×** inside null **[0.9465,1.0276]** (`ovh-a`) | a ≥32-core worker AND turbo's flat variant clears its own null there |
| **f16-GEMV** fc2-prefill scheduler reroute | `d6168ce` `ca2edb2` | fc2 medians inside floor; **tq=100 row-morsel is faster** — the proposed gate would regress it | none — the nominal 12–24× weight re-stream is concurrent/cache-served, not DRAM |
| SDPA softmax pass-elimination / scale-fold / reciprocal | `91b44b1` | +3.9 ms/window; `sc` is 384 KiB L2-resident so fusion buys nothing | a DRAM-resident, bandwidth-bound softmax scratch (never, at these shapes) |
| int8-SDPA (scores / out) | ledger 2026-07-04 | **0.14× / 0.77×** — d_head=64 thin dim, quant overhead > tiny f32 time | a fused int8 attention kernel with d_head ≫ 64 (owner/multi-day) |
| SDPA `sc` / `out` zero-init alloc | ledger (2857 / 55df007) | **0.5%** — glibc recycles the size class; `__memset` below floor | none |
| encoder **LayerNorm** | `1519` | compute-bound f64 SoA at 2.5× memory floor **AND a faithfulness feature** (more precise than ggml) | never byte-exact (f64 sum order); don't trade faithfulness |
| encoder **residual add** parallelization | 2026-07-02 | a measured wash/slight-loss; kept serial (DRAM-floored) | none |
| encoder **conv** im2col / weight-transpose | `conv_im2col` audit | reshape-audit-complete; weight-pretranspose hoisted to load | a new constant-recompute site (hunt complete, `55df007`) |
| **mel** (twiddles / hann / FFT / filterbank) | `55df007` | **0.27% of e2e**; twiddles+hann cached, FFT SIMD, cfft arena landed | none — below any median floor regardless of speedup |
| **VAD** | `9162a27` | **bridge-only**, zero native surface | native VAD ever implemented |
| `tile_shape` balanced grid at T=32 | `a410602` (parked, `tests/artifacts/perf/20260710-sgemm-tile-shape/`) | T=14 decidable (1.686×) but naive balance regresses fc1; T=32 unmeasurable | a ≥32-core worker AND a B-bytes-aware selector (unbuilt) |

### Current measurement invariants

| Surface | Evidence commit | Current state |
|---|---|---|
| SDPA exponential | `5935d68` | exponential evaluation is **23.7%** of the fused kernel |
| i7 GEMM | `44833c2` | about **28% self-time**, default-on through `enc_attn_out_i8i32_for` |
| Dequant/dispatch attribution | `3910c9c` | the measured **9.91%** is Rayon dispatch, with zero dequant arithmetic |
| Statistical gate | `bbe6ed0` | candidate median vs same-invocation A/A bootstrap 95% CI with a 2× margin; CV is provenance only |
| SDPA BR selector | `bbe6ed0` | measured medians remain inside the null floor |

## 3. GATED remainder (not one-cycle byte-exact; owner / cod / hardware)

1. **SDPA-sgemm `gemm`-crate swap** — ~1.12× on the SDPA frame (~3% e2e), WER-neutral (rel 3.8e-7)
   but **not byte-identical**; needs the `gemm` dep tree in **shared `ft_kernel_cpu`**. Owner ship/skip.
2. **cod's M4×N4 i7 tile** — the ~28% encoder frame, compute-bound at ~60% of the measured maddubs
   peak (`3b8cdebc`), ~1.4–1.6× headroom. cod's lane.
3. **Draft/speculative decoding** — R(8)≈3.7×, output-identical, needs an owner-supplied draft model.
4. **Hardware** — GPU offload (Metal path exists for macOS; CUDA nouveau-blocked here) or AVX512-VNNI
   silicon (`vpdpbusd` collapses the i7 widening chain; production Zen3 has none).
5. **frankentorch bench-harness AVX2 config** (`3db5a82`) — rch strips `RUSTFLAGS`, so a committed
   `.cargo/config` is the fix; owner-gated on FMA vs 3 hardcoded golden-f32 asserts.

**HOLDING** at the cc byte-exact frontier. This document + `docs/PR_ft_sdpa_poly_exp_turbo.md` are the
terminal record.
