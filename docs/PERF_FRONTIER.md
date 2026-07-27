# Perf Frontier — actionable handoff for the next (owner-gated) optimization session

> Forward-looking playbook, not a log. The historical record is `PERF_LEDGER.md`
> (measured wins) and `NEGATIVE_EVIDENCE.md` (rejections + blockers). This file is
> the short answer to "what's left and exactly how to do it." Owned by swarm agent
> **BlackThrush**. Last updated 2026-07-27.
>
> The byte-exact envelope includes AVX2 encoder activation quantization
> (`26feafd`), AVX2+F16C decoder weight quantization
> (`3e7f295`/`991df99`), and default-on `FW_ENC_FREE_F32` (`78ba068`).
> Ratios outside the live-incumbent table below are maintenance self-speedups
> or internal profiling point estimates, not campaign or competitive claims.

## Current tiny.en segment-timestamp behavior

For the exact `tiny.en` architecture in segment-timestamp mode, the default
policy suppresses cross-window prompt carry while preserving the initial
prompt. Explicit `max_context` or `FW_TINY_EN_TS_CONTEXT=1` restores prompt
carry, and the failed-window retry remains a conservative fallback. On the
124.5 s / 5-window `track01.wav` fixture the default emits 21 segments and
1,301 characters with full segment/timestamp/text parity against the
same-binary historical arm.

## Current live-incumbent matched-greedy CPU results

These are the current competitive figures. Each row runs the actual
`whisper-cli` incumbent side-by-side with franken in the same invocation, at
matched thread counts and greedy decode on both sides (`-bs 1 -bo 1`).

| Clip | Model | Mode | Result vs `whisper.cpp` |
|---|---|---|---|
| `track01.wav` (124.5 s / 5 windows) | large-v3-turbo | no timestamps | **2.07× faster** |
| `track01.wav` (124.5 s / 5 windows) | tiny.en | no timestamps | **1.10× faster** |
| `track01.wav` (124.5 s / 5 windows) | tiny.en | segment timestamps | **1.41× faster** |

The no-timestamp rows come from `scripts/whisper_cpp_ab.sh`; the
segment-timestamp row comes from `examples/incumbent_ab.rs`. The harnesses
self-report executable SHA-256 values, run per-engine A/A null controls in the
same invocation, and gate on the comparison median against the bootstrap 95%
null interval with a 2× margin. CV is provenance only. Full measurement and
quality evidence lives in `PERF_LEDGER.md`; rejected and non-comparable work
lives in `NEGATIVE_EVIDENCE.md`.

## Live full-pipeline span breakdown (measured 2026-07-12, real `fw transcribe`, not isolated benches)

`FRANKEN_WHISPER_PERF_SPANS=1 fw transcribe --input jfk.wav --no-persist` (single 11 s window):

| span | tiny.en (ms) | turbo (ms) | note |
|---|---|---|---|
| encoder_window | 80 | 1441 | per-window compute — dominates (ledger ceiling) |
| **model_weights** | **59** | **745** | **one-time load** (`from_ggml`: format-dequant→f32→i7/i8 requant→layout) |
| decode_loop | 48 | 231 | per-window token decode |
| model_parse | 14 | 182 | one-time ggml file parse |
| cross_kv | 9 | 36 | per-window cross-attn KV precompute |
| mel | 2.5 | 4 | per-window |
| backend_run (total) | 216 | 2666 | |

**This confirms the per-window compute ceiling** (encoder+decode dominate, both audited-at-ceiling)
**but reframes "load is sub-floor":** load (`model_parse`+`model_weights`) is **~35 % of single-shot
turbo wall time** (927 / 2666 ms) — sub-floor only for BATCH/long-file/server-resident workloads
(`load_resident` amortizes it to ~0), NOT for single-clip CLI / serverless / first-request latency.

Current load attribution is split between error-feedback quantization's serial
dependency chain and bandwidth-bound f16→f32 dequantization. The latter sustains
about 27 GB/s including reads and writes. The single-shot load path is
characterized at its current CPU floor; retry predicates and rejected
implementations live in `NEGATIVE_EVIDENCE.md`.

## State: the byte-exact, autonomously-verifiable envelope is CLOSED

Everything that could be landed with a *quick, local, byte-exact* verify has been:

- **Peripheral IO/DB/export lane — fully optimized** (this session, 10 measured wins,
  all default-on, byte-exact): tty zlib `bufread`; `BufWriter` on every export writer
  (~40×) + sync incremental; streaming SHA-256 checksums (full + incremental export);
  64 KiB checksum read buffer (~1.16×); per-statement-savepoint skip on persist (1.48×)
  + sync import; DB-level N+1 → `IN (…)` on incremental export (1.32×); app-level N+1
  batch on routing history (**~14×**).
- **Transcription hot path — at its byte-exact ceiling**: encoder full int8 already
  default-ON for **both calibrated models — turbo AND tiny.en** (`calibrated_encoder_int8_model`
  = `tiny_en || is_large_v3_turbo`, shipped `a997f37`, ~1.47× encoder; `FW_ENC_ATTN_OUT_I8I32=0`
  kills); int8 logits head default-ON; `nn::quantize_act_i8_into` already AVX2-vectorized w/ correct
  round-half-away; SDPA poly-exp shipped for turbo; decode alloc-light rewrite landed. Measured/closed.
- **Flag audit (2026-07-12): every byte-exact `FW_*` win is already default-ON** — nothing dormant
  to flip. Verified default-ON: `FW_I8_BATCH_4COL`, `FW_I8_BATCH_2COL`, `FW_I7_M2N4`,
  `FW_I7_QKV_HEADMAJOR_ROWCO`, `FW_F16_BATCH_M2COL`, `FW_BATCH_GEMV_ROW_MORSEL`,
  `FW_SDPA_GATHER_CHUNKS` (=16), `FW_PERSIST_SKIP_STMT_SP`, `FW_STORAGE_BATCH_HISTORY`,
  `FW_SYNC_BATCH_{QUERY,IMPORT,SKIP_STMT_SP}`. The remaining **default-OFF** flags are all
  **lossy/quality** (NOT byte-exact → owner/WER-gated, cannot be autonomously flipped):
  `FW_CROSS_V_BLOCK`, `FW_DEC_EF`, `FW_ENC_INT8_ATTN_IN`, `FW_SIMD_EXP`, `FT_SDPA_POLY_EXP` (tiny.en).
- **Two rejections** kept the discipline honest: `load_run_details` scan (sub-floor),
  persist multi-row INSERT (regression). Unifying rule: **batching helps only when it
  cuts execution COUNT, not per-row work.**
- **Sweeps that found nothing** (so nobody re-runs them): the youtube batch pipeline
  already shares one engine across all videos (`transcribe_and_render(&engine, …)` — no
  per-video model-reload N+1); `Regex::new` in hot loops (none); raw-`File` write/read
  loops (all buffered or one-time SHA); `Vec::contains`/O(n²) in hot paths (none); stdout
  per-item emit (streaming-unsafe to buffer, sub-floor for batch dumps).
- **LLVM-leaves-perf antipattern sweep — CLOSED** (re-verified against current code
  2026-07-12, post the recent default-on flips): the four exploited scalar-hot-loop classes
  are all covered. **argmax** (index-tracking reduction, loop-carried `best_i` ⇒ won't
  autovec) is the ONLY one that needed hand-AVX2 — landed `argmax_idx` 5.10× byte-exact
  (`decode.rs:614`, `[[project_argmax_avx2_landed]]`); its *siblings* are NOT levers:
  **max/min folds** (`decode.rs:387` timestamp-rule `max_text_logprob`, `decode.rs:1941`
  lang-detect, softmax) already lower to `llvm.vector.reduce.fmax` (byte-identical, ~1.2–1.36×,
  sub-noise — ledger `7469`/`7478`/`7779`); **`.round()`** quant maps are AVX2'd
  (`encoder.rs:1228`, `nn.rs:2332`); **gather** (gelu) exhausted. No uncovered index-tracking
  hot serial loop exists (grep). Don't re-grep this vein.
- **Cross-attention K/V — verified fully optimized** (2026-07-12, the last per-window area not yet
  re-checked this session): the per-window K/V PROJECTION (`encoder_out @ Wk/Wv`, tq=1500) runs the
  dequant-once f32 sgemm (`cross_proj_f32_enabled` **DEFAULT-ON**, mod.rs:724; **2.25×** on turbo,
  golden-checked, `examples/cross_f16path_probe`); the per-TOKEN K/V read is **f16 by default**
  (byte-identical) with int8/block-wise variants gated for quality (`FW_CROSS_V_BLOCK`,
  [[project_cross_v_block_win]]). `cross_attn` is only ~4.4% of decode. No byte-exact cross lever.
  **With this, every per-window area (encoder int8/FLOP/SDPA/conv/LN, decode mlp/logits/qkv/self-attn/
  cross, mel) is personally re-verified closed this session, and the load path is floored (above) —
  the autonomous byte-exact frontier is empirically exhausted; remaining levers are owner/infra only.**

## tiny.en full-int8 configuration

Commit `a997f37` supplies the current default: `calibrated_encoder_int8_model()`
returns `tiny_en || is_large_v3_turbo`, so tiny.en uses the quality-gated full
int8 encoder (q/k/v/fc1/fc2 i7 plus attention-out i8). Setting
`FW_ENC_ATTN_OUT_I8I32=0` selects the f32 path. `FW_ENC_INT8_FC1` is inert while
the full-int8 branch is selected.

## Remaining levers — all need the model-bench + corpus-WER loop + owner sign-off

| lever | est. e2e | evidence in hand | why gated | validate before flip |
|---|---|---|---|---|
| **ToMe / layer-pruning** (encoder FLOP reduction) | large (turbo) | space mapped; tail-truncation already landed | changes output structurally | full WER + segment-timing corpus |
| **poly-exp variants / GPU** | — | poly-exp turbo shipped; GTX1070 = nouveau (no CUDA) | owner / infra | — |

## Import path

`FW_SYNC_BATCH_IMPORT` is default-on. The runs, segments, and events paths
prefetch existing rows in batches and dispatch through the shared
`apply_{run,segment,event}_row` conflict logic. Intra-chunk seen maps preserve
the reference behavior for duplicate keys. The byte-exact import surface has
no pending lever.

## Recipes (so the next session doesn't rediscover them)

- **Fast byte-exactness check, NO build** (~0.3 s/clip): prebuilt
  `/data/tmp/cargo-target/release/fw` + `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole` +
  `FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL=…/ggml-tiny.en.bin` +
  `FRANKEN_WHISPER_MODEL_DIR=…/models`, `fw transcribe --input <clip> --no-persist`,
  diff stdout with the `FW_*` flag off/on. Timing is NOT measurable this way (load-
  dominated); use it only to reject-or-gather byte-exact evidence.
- **Warm perf sizing**: first require at least 120 GiB free with `df -h /data`,
  then build strictly remotely:
  `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench
  --bench native_engine_bench -- encoder_window_tiny` (+ `e2e_tiny_jfk`,
  `decoder_token_step_tiny`), and A/B the flag via
  external env on the same cached binary. **TWO GOTCHAS learned 2026-07-12:** (1) the
  `e2e_tiny_jfk` A/B needs an **idle box** — on a loaded host wall-clock swings ~22% (load
  2→26 mid-run) and buries any sub-15% lever. (2) Do NOT run `f32 then flag` each rep — the
  flag arm always runs second, so a warming/contending machine makes it *look* slower
  (this exact confound made `FW_ENC_INT8_FC1` look like an e2e regression when it likely
  isn't). Use ABBA / randomized order, and note both `encoder::forward` and the production
  `encoder::forward_from_full_mel_window` **ignore the thread-hint** (same ft rayon pool),
  so `encoder_window` IS representative of the encode *work* — the divergence was
  measurement, not a real code-path difference.
- **Corpus WER vs the original**: `legacy_whispercpp/whisper.cpp/build/bin/whisper-cli`
  is the reference (not on `$PATH`); tiny.en + turbo models + jfk/other clips live in
  `legacy_whispercpp/whisper.cpp/models/` and `sample_audio_files/`, `tests/fixtures/audio/`.
  **BASELINE ESTABLISHED + BLOCKER SHARPENED (2026-07-12):** the SHIPPING int8 `fw` is
  **byte-identical (normalized) to whisper.cpp on jfk** (`whisper-cli -m tiny.en -nt -t 8` vs
  `fw transcribe --no-persist`) — so the default int8 engine is faithful on real speech; the
  gated-lever WER baseline on jfk is **≈0**. **mp3-corpus limitation RESOLVED (`decode_to_wav` example, `e221630`):** `whisper-cli` can't read
  `.mp3`, but `cargo run --release --example decode_to_wav -- <mp3> <wav>` (built-in symphonia, no
  ffmpeg) makes any mp3 whisper-cli-readable. So `example_audio_track_01.mp3` now HAS a reference.
  **ENCODER-INT8 PROPER-NOUN WER (2026-07-12, positive):** the concern gating the shipped int8 was
  proper-noun safety — MEASURED on the track01 proper-noun clip (turbo): fw (shipping int8) vs
  whisper.cpp turbo = **271 vs 283 words (~96%), ~22 diff lines (~4% word variance)**, and the proper
  nouns are CORRECT (FrankenSearch, Twitter, XF, CAS×2, Daniel). No content-drop (turbo covers the full
  span). ⇒ the shipped int8 encoder is **proper-noun-faithful** on this clip; the ~4% is normal
  cross-quant/decode variance, not a quality bug. Still only 2 real-speech clips (jfk + track01) on box;
  a full corpus-WER for the remaining gated levers needs the **owner to supply more diverse speech**,
  but the mp3-corpus tooling is now in-tree and the int8 encoder is validated proper-noun-safe on the
  one proper-noun clip available.

## Recommendation

Use only the three live-incumbent matched-greedy rows at the top of this
document for competitive copy: **2.07×** large-v3-turbo no timestamps,
**1.10×** tiny.en no timestamps, and **1.41×** tiny.en segment timestamps.
Self-speedups are maintenance evidence and do not count as campaign output.
The byte-exact performance frontier is closed; redirect work to the
owner-scoped items below.

**Validation COVERAGE + what's resource-blocked (so the owner knows what to supply to extend it):** the
numbers above cover **English** speech on the **two real on-box models** — `tiny.en` (74 MB, English-only)
and `large-v3-turbo` (1.5 GB, multilingual-capable). Two axes remain UN-measured and are **blocked on
missing on-box assets, not effort**: (1) **multilingual** — turbo *can* do it, but there is **no non-English
audio on box** (all clips — jfk / track01 / sjobs / test_10s_speech — are English); (2) the **intermediate
models** (base / small / medium) — only 562 KB *test stubs* are present, no real weights. To extend the
faster-and-faithful validation to those, the owner needs to drop in non-English speech clips and/or the real
base/small/medium ggml models; then the same harness (`decode_to_wav` → interleaved `fw` vs `whisper-cli`,
WER + word-agreement) applies unchanged.

Pause the autonomous *byte-exact* loop — further ticks only re-measure settled ground or
land sub-floor micro-levers the ledger reverts. The encoder FLOP-reduction row
is **measured dead on CPU** — `NEGATIVE_EVIDENCE` closes all three redundancy axes with data:
DEPTH (layer-pruning fatal at skip-1: `=31` mangles proper nouns + repetition-loops track01 (−27% words) though it's jfk-byte-identical; `=30` breaks even jfk — `7092` + 2026-07-12 update), SEQUENCE (ToMe frames not mergeable,
`4518`), SPECTRAL (weights near-full-rank, `4640`); Nyström/CountSketch/PQ/low-rank/Strassen all
rejected (`4552`). So the genuinely-remaining levers are **owner/infra only**: (1) a **Linux GPU
compute stack** (GTX 1070 is on nouveau → no CUDA/OpenCL/Vulkan — the encoder GEMM/SDPA is the
sole out-of-crate lever); (2) a **cheap multilingual DRAFT model** to unlock speculative decode
(verify amortization R(K)≈3.7× de-risked, but the draft-model-FREE **layer-skip self-draft is
MEASURED-DEAD** — `FW_DRAFT_ACCEPT_LAYERS` probe, NEGATIVE_EVIDENCE `6675`/`252`: k-of-4-layer early-exit
argmax matches the full-model argmax only **0% / 0% / 11.8%** (k=1/2/3) vs the 47% / 65% / 82% break-even,
because the distilled 4-layer decoder's layers are all load-bearing — so the drafter MUST be a real
separate model with a smaller logits head, not a self-skip); (3) **AVX-512-VNNI hardware** (int8 encoder GEMM
is 0.89× on this AVX2-no-VNNI box). No autonomously-landable byte-exact perf lever remains
(re-verified against current code 2026-07-12: encoder int8 maximal, import N+1 default-on, IO
swept, fresh shipped-tiny.en encoder profile = exp `__expf_fma` ~9% [poly-exp owns it: turbo-on,
tiny.en regressed-off] + rayon `__sched_yield` [contention-inflated] + int8-GEMM bulk — no new
hot spot).
