# RESULTS — Native Engine Extreme-Optimization Arc (run 20260605T0218Z)

Final landing summary for the profile-driven optimization loop on
`src/native_engine/`. Host: **Apple M4 Pro**, 14 cores (10P+4E), 64 GB,
macOS 26.2, `release-perf` profile (opt3, thin-LTO, codegen-units=1). All
"final" hyperfine numbers below were taken **interleaved** (REF/CAND runs
alternated back-to-back on the same host under the same concurrent agent
load) so they compare like-for-like rather than across drifting machine load.

Baseline (this run, git `0553cbb`, release-perf, warm page cache):
**tiny.en jfk 1.225 s ± 0.029** (15 runs); **large-v3-turbo jfk 32.92 s ± 0.16**
(5 runs). whisper-cli same host: tiny 0.57 s CPU; large 9.63 s CPU / 2.11 s Metal.

---

## 1. Levers Landed

Six commits, profile-justified, each kept only on measured wins
(bit-identical unless noted under DISC-004):

| Commit | Lever | Effect (measured, this host) |
|--------|-------|------------------------------|
| `2ed6471` | Background model **version-tag hash** at load | sha256 over the model moved off the run path: **3.0 s → 0** on the critical path (overlaps encode/decode; tag itself unchanged) |
| `0361bb2` | **Parallel f16→f32 dequant + tiled parallel weight transpose** at load (`LoadedModel::from_ggml`) | `model_weights` span **2.4 s → 0.47 s** |
| `bdbdd21` | **Language-detect encode reuse** — reuse window-0's encode instead of a hidden duplicate encoder pass | **−8.8 s** (eliminates the duplicate window-0 encoder window on large) |
| `ee36fd4` | **Parallel serial glue ops** (layer_norm / softmax / gelu / im2col / attention-head loops / logits GEMV / cross-K-V) | encoder **8.3 → 5.8 s**; tiny wall **0.92 → 0.56 s** |
| `0989a5a` | **Tail-window encoder-context truncation** (audio_ctx-style, DISC-004, default-on with `FRANKEN_WHISPER_NATIVE_TAIL_TRUNCATE=0` kill switch) | tail-window encode **4.2 s → 0.24 s** (~94%); large wall ~11.0 → ~7.0 s. Main transcript byte-identical; divergence confined to spurious trailing-silence hallucination on tail windows (see DISC-004) |
| `5bb778b` | **Decoder per-token path** — KV buffer reuse, in-place logits band reads, hoisted window-constant cross K^T/V transforms, threaded QKV, head-parallel cross-attn | **large 62.6 → 39.9 ms/tok**; **tiny 11.9 → 6.99 ms/tok**. Bit-identical (4 new bitwise nn tests); also drops ~490 MB cross-buffers on large |

---

## 2. Evidence-Backed Abandons (zero-change passes 2 & 4)

Levers that were implemented and **measured**, then dropped because the
evidence did not justify them — recorded so the next loop does not re-pay
the same investigation:

- **Fused QKV projection** — bit-identical *proven*, but **~16% slower**: one
  wide column-parallel sgemm has more thread-feeding overhead than 3 narrow
  calls at these shapes.
- **bmm-batched per-head attention** — bit-identical, **~4% slower** after a
  proper interleaved A/B (a transient low-load reading had briefly lied).
- **Parallel residual adds** — slower: spawn storm beats a memory-bound add.
- **Encoder scratch-reuse arena** (eliminate the 2× `x.clone()` per encoder
  block) — bit-identical, removes ~490 MB transient/window, but **wall- and
  RSS-neutral**: the encoder is sgemm-**compute**-bound and ft's output `Vec`
  is `vec![0.0; n]` = calloc with lazy first-touch pages the sgemm writes
  anyway. No wall-clock movement.
- **ft-side matmul arena (lever 6, quantified)** — fresh out-`Vec` per ft
  matmul call (1472 calls/large-encoder-window). Microbench: fresh calloc+write
  550 ms/window vs reuse-one-buffer 503 ms/window ⇒ **~9% theoretical**
  (~47 ms/large-window) of alloc-bookkeeping + first-touch fault overhead a
  reuse arena *could* reclaim. This is a **frankentorch** change, out of scope
  for this crate, and below the loop's score bar; flagged for the orchestrator.

---

## 3. Final Interleaved Numbers (same host, concurrent load, runs interleaved)

### tiny.en, jfk.wav
| Engine | Wall | Verdict |
|--------|------|---------|
| **native** | **475 ms ± 13** | **2.33× FASTER** than whisper-cli CPU |
| whisper-cli CPU | 1105 ms ± 21 | |

### large-v3-turbo, jfk.wav
| Engine | Wall | User CPU | Note |
|--------|------|----------|------|
| **native** | **9.731 s ± 0.272** | **53.8 s** | parity with whisper-cli CPU, *less* user CPU |
| whisper-cli CPU | 9.585 s ± 0.224 | 65.4 s | |
| whisper-cli Metal | 2.169 s | — | GPU reference (out of native CPU scope) |

### Cumulative vs session start (release profile)
| Model | Session start | Final | Speedup | RTF (final) |
|-------|---------------|-------|---------|-------------|
| tiny.en | 1.57 s | **0.475 s** | **3.3×** | **0.043** (faster than realtime) |
| large-v3-turbo | 44.6 s | **9.73 s** | **4.6×** | **0.88** (faster than realtime) |

(RTF = wall / 11 s of jfk audio.)

---

## 4. Release-profile discovery (follow-up, do NOT auto-apply)

The `release` profile ships `opt-level = "z"` (size-optimized). Building the
perf binary under plain `release` cost **~26% on large-v3-turbo** vs the
`release-perf` (opt3 + thin-LTO) numbers above. **Recommendation:** evaluate
promoting the `dist`/`release` profile to opt3 (or aligning it with
`release-perf`) as a follow-up. **Not changed here** — the `Cargo.toml`
release profile was deliberately left untouched; this is an orchestrator/owner
decision, not a perf-loop edit.

---

## 5. Cross-references

- Hotspot table + hypothesis ledger: `HOTSPOTS.md` (this dir).
- Spans timelines: `spans_large_timeline.txt`, `spans_*.jsonl` (this dir);
  enable with `FRANKEN_WHISPER_PERF_SPANS=1`.
- Tail-truncation accuracy/precision analysis + kill switch: `DISCREPANCIES.md`
  **DISC-004**.
- Promotion-criteria status: `docs/native_engine_contract.md` §Performance.
- Tracking bead: **bd-2th6** (criterion benches landed in Round 2, pass 1 —
  see §6 below).

---

## 6. Round 2 — f16 compute + cross-repo sgemm + decoder attribution

Round 1 took the in-scope levers to convergence (large 9.73 s, tiny 0.475 s).
Round 2 (commits `a236433`, `8abea12`, `c703035`, `0a5c939`, + the
franken-decision-0.3.2 migration / landing commit) reopened the three
authorized frontiers — f16 weight traffic, the encoder sgemm, and the ft-side
microkernel — and harvested the one that paid off (decoder f16 compute, now the
production default) while definitively rejecting the rest with measured proof.

Measurement discipline unchanged from Round 1: interleaved A/B vs a pre-change
REF binary, min/p25 of ≥6 pairs, host under concurrent agent load,
`release-perf` profile, `FRANKEN_WHISPER_PERF_SPANS=1` for attribution.

### Pass-by-pass

| Pass | Lever | Result | Disposition |
|------|-------|--------|-------------|
| 1 | Criterion bench substrate (`benches/native_engine_bench.rs`): mel 30 s, encoder window (tiny+large), decoder token step (tiny+large), logits GEMV, e2e tiny jfk; saved baselines + `[[bench]]` registration | Baseline **round2-pre** saved: mel 54.1 ms; enc tiny 123 ms / large 5.71 s; tok-step tiny 7.2 ms / large 43 ms; logits_gemv_large 8.96 ms; e2e tiny 382 ms | Measurement infra landed (bd-2th6 deliverable) |
| 2 | f16-resident decoder compute, **fused** dequant-in-GEMV (per-element widen inside the dot loop) | Micro WIN (tok-step large −54 %, logits −10 %) but **e2e REGRESSION** (tiny +27 %, large +12 %): the per-element scalar widen serialized the FMA and blocked autovectorization | Built, conformance-clean, shipped **default OFF** as opt-in env switch; root cause carried into pass 3 |
| 3 | **Vectorized** f16 dequant → flip decoder f16 to **default ON**. `WeightMat::F16` now `Vec<Float16>`; bulk-SIMD `convert_to_f32_slice` row-dequant into reused scratch then 8-lane `dot8` | Dequant **13.9 → 56.2 GB/s** (4×, NEON fp16 slice path); isolated f16 GEMV [1280²] ~1080 → ~205 µs (5.3×). **e2e large −11.5 % min / −8.6 % p25** (was +12 % regression), tiny within noise (−0.3 %/−1.6 %). Conformance: byte-exact goldens both models ON+OFF. **Encoder f16 panels prototyped then SKIPPED** — pure overhead (+0.6 %…+6.5 %) on every large encoder matmul: f16 wins only in the GEMV/bandwidth regime (decoder M=1), not GEMM (encoder M=1500, compute-bound) | **DEFAULT FLIPPED ON**; env var becomes opt-OUT kill switch; encoder frontier closed |
| 4 | ft-side sgemm overhead (CROSS-REPO `frankentorch` ft-kernel-cpu): output-buffer reuse + col-parallel/block-size tuning on M4 Pro | Whisper-large shapes all take the row-split (TALL) path: col-parallel **inapplicable**; block-size sweep **REGRESSES** (oversubscription re-streams the 26 MB B panel, mlp_fc +8.7 %/+16.6 %); `_into` output reuse **NEUTRAL** (alloc is calloc-lazy, sub-ms vs 8–36 ms compute-bound GEMM). Consumer wiring (`EncoderScratch`) BIT-EXACT but **no robust e2e win** (criterion CI pure noise) → reverted | **ft-side rejected** with bit-exact proof. Retained additive `matmul_tensor_contiguous_f32_into` API + whisper benches (ft `4af78e91`); rejection artifact ft `43d0a7b0`, dir `tests/artifacts/perf/20260606T030959Z-m4pro-whisper-sgemm/` (+ `701ca1c4`). No franken_whisper consumer code landed |
| 5 | Criterion attribution of the decoder token step + logits wider-parallelism | Per-sub-part attribution table delivered (see below). Landed lever: **size-gated logits GEMV widening** (8→12 workers for out ≥ 16384, i.e. the [51866×1280] vocab product only) → logits_gemv_large **−1.9 %…−3.9 %** (p<0.05, bit-identical disjoint row bands). REJECTED with measured proof: cross_attn f16 K/V (already at parallel floor — serial 6.87 ms vs parallel 1.59 ms, 4.3×; f16-OFF cross_attn 1.76 ms ≈ f16-ON 1.59 ms, dtype nearly irrelevant), cross_attn wider head-workers (neutral), per-token-Linear widening (+29 % — too few rows/band) | Logits widening landed; bandwidth-bound-cross-attn thesis measured-and-rejected |

### Decoder attribution table (large-v3-turbo, f16 ON = default, real jfk-derived state, 200 steps)

| Sub-part | ms/tok | % | Note |
|----------|--------|---|------|
| mlp_fc_gelu_proj | 2.38 | 23.0 % | already f16; compute/gelu-bound |
| logits_gemv | 2.13 | 20.6 % | ← attacked (size-gated 12-worker widening) |
| cross_attn | 1.59 | 15.4 % | at parallel floor; f32 K/V read NOT the bottleneck |
| self_qkv_proj | 1.55 | 15.0 % | already f16; 3× threaded GEMV |
| self_attn | 1.00 | 9.6 % | KV-cache attention; grows w/ depth |
| cross_q_proj | 0.55 | 5.4 % | |
| cross_out_proj | 0.55 | 5.3 % | |
| self_out_proj | 0.55 | 5.3 % | |
| {self,cross,mlp}_ln+clone / embed / final_ln | ~0.012 / ~0.003 | — | negligible |

f32 path (f16 OFF) for contrast: **38.9 ms/tok** — mlp 19.3 ms (48.6 %), logits
6.9 ms, projections 2.4–3.3 ms each — i.e. pass-3 f16 already crushed every
weight-bound Linear (mlp 8×, logits 3.3×).

### Current decoder floors (f16 default ON)

| Path | ms/tok |
|------|--------|
| large-v3-turbo (f16 ON, production) | **10.3** |
| large-v3-turbo (f16 OFF, contrast) | 38.9 |
| tiny.en (f16 ON) | **5.2** |

### Convergence statement

Every remaining decoder sub-part now sits at its parallel/compute floor:
mlp_fc and self_qkv are already f16 and compute-bound; cross_attn is at its
head-parallel floor (proven by the 4.3× serial-vs-parallel diagnostic, and by
f16-OFF cross_attn being within 11 % of f16-ON — the K/V dtype is not the
lever); logits is the one sub-part that still had daylight to the
bandwidth thesis, now harvested. The encoder is compute-bound GEMM where
halving resident weight bytes buys no compute time (f16 panels measured pure
overhead). The **only frontier left is a GEMM microkernel rewrite** of the
ft-kernel-cpu sgemm — and that has already been rejected by frankentorch's own
packed-panel / Strassen pilots (`20260603T18*`), so it is out of reach for this
loop. Round 2 is therefore converged: the decoder f16 default-ON switch is the
net win (large e2e −11.5 %), with a small bit-identical logits widening on top.

### Conformance / golden gate (Round 2 landing)

f16 default ON is the production path. Golden extract+diff (`examples/native_ab`)
vs `/tmp/fw_golden/{tiny.en,large-v3-turbo}.json`: **byte-exact on both models**
(sha256 identical — `7a45577a…` tiny, `c6702b3a…` large). Full lib suite
**3086/3086** green; integration suites native_engine_e2e 6, no_canned_phrases 6,
conformance_comparator 26, cli_integration 82 — all green. fmt --check clean,
clippy --all-targets -D warnings clean.


## 7. Round 3 — Scale (2026-06-06; artifacts in ../20260606T2341Z-scale-baseline/)

The unprofiled dimension: real-world long audio (the original use case is
2-hour call recordings). Fixture: a real 2h call, 297 windows, 25,348 tokens.

| Pass | Outcome | Evidence |
|------|---------|----------|
| 1 Scale profile | Linear scaling proven (no super-linear bends); 2h tiny seq = 604s, RTF 0.083, 2.05GB RSS. Rejected with numbers: full-mel memory, prompt-quadratic (capped, mean 39 tok), window fixed overheads, report serialization (0.35% at 25k tokens). Found: tiny long-audio = 74% decode_loop; IF backend 1.43x SLOWER at scale + 67% seam word-diff. | HOTSPOTS.md |
| 2 Tiny token floor | gemv_f16 PAR_THRESHOLD 1<<16 -> 1<<19 (measured crossover: serial wins <=393k MACs). self_qkv 1.38 -> 0.32 ms/tok; 600s decode_loop 6.83 -> 3.83 ms/tok (-44%); byte-identical. | PASS2_RESULTS.md |
| 3 IF at scale | Contiguous ranges (rolling prompt within ranges; seams = workers-1) + no-oversubscription budget (MIN 5 threads/worker). tiny 1.43x -> 0.73x vs seq; word-diff 67% -> 22%; new contract: 1 worker == sequential byte-exact. | PASS3_RESULTS.md |
| 4 Re-profile | ZERO-CHANGE floor verdict: **2h call = 262.9s via IF (2.30x round delta; RTF 0.036)**; decode floor flat 4ms/tok across 297 windows; skew 0.85%; every knob swept and rejected. | PASS4_RESULTS.md |
| 5 Reserve | Skipped-as-redundant (pass 4 covered the scope hours earlier). | — |

Round-3 convergence: both long-audio regimes at measured floors —
tiny/long = serial-GEMV compute floor; large = encoder GEMM (closed in
rounds 1-2). The 2-hour-call headline across the whole project: from
"mock engine, no real output" 3 days ago to 262.9s of real transcription
(13.7x faster than realtime) in memory-safe Rust.
