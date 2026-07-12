# Perf Frontier — actionable handoff for the next (owner-gated) optimization session

> Forward-looking playbook, not a log. The historical record is `PERF_LEDGER.md`
> (measured wins) and `NEGATIVE_EVIDENCE.md` (rejections + blockers). This file is
> the short answer to "what's left and exactly how to do it." Owned by swarm agent
> **BlackThrush**. Last updated 2026-07-12.

## State: the byte-exact, autonomously-verifiable envelope is CLOSED

Everything that could be landed with a *quick, local, byte-exact* verify has been:

- **Peripheral IO/DB/export lane — fully optimized** (this session, 10 measured wins,
  all default-on, byte-exact): tty zlib `bufread`; `BufWriter` on every export writer
  (~40×) + sync incremental; streaming SHA-256 checksums (full + incremental export);
  64 KiB checksum read buffer (~1.16×); per-statement-savepoint skip on persist (1.48×)
  + sync import; DB-level N+1 → `IN (…)` on incremental export (1.32×); app-level N+1
  batch on routing history (**~14×**).
- **Transcription hot path — at its byte-exact ceiling**: encoder full int8 already
  default-ON for calibrated models (turbo, 1.47×); int8 logits head default-ON;
  `nn::quantize_act_i8_into` already AVX2-vectorized w/ correct round-half-away; SDPA
  poly-exp shipped for turbo; decode alloc-light rewrite landed. Measured/closed.
- **Two rejections** kept the discipline honest: `load_run_details` scan (sub-floor),
  persist multi-row INSERT (regression). Unifying rule: **batching helps only when it
  cuts execution COUNT, not per-row work.**

## Remaining levers — all need the model-bench + corpus-WER loop + owner sign-off

| lever | est. e2e | evidence in hand | why gated | validate before flip |
|---|---|---|---|---|
| **`FW_ENC_INT8_FC1` for tiny.en** (fc1-only encoder int8) | ~1.9% encoder ≈ **sub-1% e2e** | `encoder_window_tiny` 83.1→81.5 ms (CIs disjoint); transcript **byte-identical on all 5 clips** incl. real speech | global flag (unknown >4-layer models untested); sub-floor e2e | corpus WER on tiny.en + scope the default to tiny.en if desired |
| **tiny.en encoder int8 *calibration*** (enable the full `enc_attn_out_i8i32`, not a flag) | up to ~1.47× encoder (turbo-sized) | full-int8 flag is **calibration-inert** for tiny.en — needs a calibration entry, not a flip | quality (proper nouns) unproven for tiny.en; needs `ENCODER_INT8_CALIBRATION_ID` work | proper-noun corpus WER vs whisper-cli |
| **ToMe / layer-pruning** (encoder FLOP reduction) | large (turbo) | space mapped; tail-truncation already landed | changes output structurally | full WER + segment-timing corpus |
| **poly-exp variants / GPU** | — | poly-exp turbo shipped; GTX1070 = nouveau (no CUDA) | owner / infra | — |

## Recipes (so the next session doesn't rediscover them)

- **Fast byte-exactness check, NO build** (~0.3 s/clip): prebuilt
  `/data/tmp/cargo-target/release/fw` + `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole` +
  `FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL=…/ggml-tiny.en.bin` +
  `FRANKEN_WHISPER_MODEL_DIR=…/models`, `fw transcribe --input <clip> --no-persist`,
  diff stdout with the `FW_*` flag off/on. Timing is NOT measurable this way (load-
  dominated); use it only to reject-or-gather byte-exact evidence.
- **Warm perf sizing** (needs ~6 min local build): `RCH_MIN_LOCAL_TIME_MS=999999999`
  `FRANKEN_WHISPER_MODEL_DIR=…` `cargo bench --bench native_engine_bench --
  encoder_window_tiny` (+ `e2e_tiny_jfk`, `decoder_token_step_tiny`), A/B the flag via
  external env on the same cached binary.
- **Corpus WER vs the original**: `legacy_whispercpp/whisper.cpp/build/bin/whisper-cli`
  is the reference (not on `$PATH`); tiny.en + turbo models + jfk/other clips live in
  `legacy_whispercpp/whisper.cpp/models/` and `sample_audio_files/`, `tests/fixtures/audio/`.

## Recommendation

Pause the autonomous *byte-exact* loop — further ticks only re-measure settled ground or
land sub-floor micro-levers the ledger reverts. Schedule a deliberate **owner-authorized
model-bench + WER session** and start from row 1 (`FW_ENC_INT8_FC1`, already 5-clip
byte-exact) or row 2 (tiny.en calibration, the bigger prize).
