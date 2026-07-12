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
- **Sweeps that found nothing** (so nobody re-runs them): the youtube batch pipeline
  already shares one engine across all videos (`transcribe_and_render(&engine, …)` — no
  per-video model-reload N+1); `Regex::new` in hot loops (none); raw-`File` write/read
  loops (all buffered or one-time SHA); `Vec::contains`/O(n²) in hot paths (none); stdout
  per-item emit (streaming-unsafe to buffer, sub-floor for batch dumps).

## Remaining levers — all need the model-bench + corpus-WER loop + owner sign-off

| lever | est. e2e | evidence in hand | why gated | validate before flip |
|---|---|---|---|---|
| **`FW_ENC_INT8_FC1` for tiny.en** (fc1-only encoder int8) | ~1.9% **encoder** (isolated); **e2e sub-floor + UNMEASURED reliably** | `encoder_window_tiny` fc1 ~1.9% faster (warm); byte-identical on 5 clips. e2e A/B looked like a regression but was **confounded** (see note) | quality (global flag, >4-layer WER unproven); e2e sub-floor for tiny.en's ~20%-of-e2e small encoder | a **run-order-safe** e2e A/B on an idle box (ABBA, not f32-then-fc1 each rep) + corpus WER |
| **tiny.en encoder int8 *calibration*** (enable the full `enc_attn_out_i8i32`, not a flag) | up to ~1.47× encoder (turbo-sized) | full-int8 flag is **calibration-inert** for tiny.en — needs a calibration entry, not a flip | quality (proper nouns) unproven for tiny.en; needs `ENCODER_INT8_CALIBRATION_ID` work | proper-noun corpus WER vs whisper-cli |
| **ToMe / layer-pruning** (encoder FLOP reduction) | large (turbo) | space mapped; tail-truncation already landed | changes output structurally | full WER + segment-timing corpus |
| **poly-exp variants / GPU** | — | poly-exp turbo shipped; GTX1070 = nouveau (no CUDA) | owner / infra | — |

## The one remaining BYTE-EXACT lever (no WER gate) — import N+1, intricate

The sync **export** N+1 is landed (`FW_SYNC_BATCH_QUERY`, ~1.32×). Its mirror, the **import**
path, is the last un-optimized IO site — byte-exact (no quality gate), just careful.

**UPDATE 2026-07-12 (`d2b5b14`): the RUNS table is now LANDED** behind `FW_SYNC_BATCH_IMPORT`
(**default-OFF**). `import_runs` dispatches legacy vs batched; both call a shared `apply_run_row`
(existing row passed as `&[SqliteValue]`, so the 11-field identical-compare is bit-for-bit the
same), and `flush_run_chunk` does the `WHERE id IN (…)` prefetch + a seen-map for intra-chunk
duplicate ids. Gate: `sync::tests` 348/0 incl. `flush_run_chunk_matches_per_line_reference`.
**SEGMENTS also LANDED** (`8199711`, composite `(run_id,idx)`: prefetch `WHERE run_id IN (…)` +
map by `(run_id,idx)` + shared `apply_segment_row`/`record_segment_pre` + seen-map; 349/0 tests +
E2E byte-identical). **Only `import_events` `(run_id, seq)` remains** — same recipe. Both hazards below still apply. Not a quick tick —
rushing a conflict-semantics change on the sync path is how the `quantize_act_i7` re-dig burned a
turn; do the composite-key tables in a dedicated pass.

- **Sites** (`src/sync.rs`): `import_table_runs` loop `SELECT … WHERE id=?1` **per line**
  (~:1202); `import_table_segments` `WHERE run_id=?1 AND idx=?2` (~:1384); `import_table_events`
  `WHERE run_id=?1 AND seq=?2` (~:1536). One query per JSONL line = N+1.
- **Recipe**: chunk the lines; per chunk collect keys → one `WHERE … IN (…)` → pre-fetch a
  `HashMap<key, full_row>`; process lines in original order against the map, applying the exact
  same identical-compare + `ConflictPolicy` (Reject/Skip/Overwrite) logic.
- **Hazard 1 — full row, not existence**: the per-line SELECT returns all columns for an 11-field
  identical-vs-conflict compare, so the map must hold full rows (not a `HashSet` of ids).
- **Hazard 2 — intra-chunk duplicate ids**: the per-line version's later duplicate SEES the
  earlier line's INSERT. A pre-fetch queried before any insert does not. Maintain a `seen` map
  updated on every insert/delete within the chunk so duplicate-id files stay byte-exact.
- **Composite keys**: segments/events key on `(run_id, idx)` / `(run_id, seq)`; if fsqlite lacks
  row-value `IN`, batch by `run_id` and index the map by the composite key.
- **Expected magnitude**: import is INSERT-dominated (the persist multi-row-INSERT reject proved
  per-row B-tree work isn't batchable), so batching only the SELECT setup nets **< the export's
  1.32×**. Real but modest.
- **Gate**: `sync::tests` round-trips + a NEW intra-chunk-duplicate-id test must stay byte-exact;
  put it behind a `FW_SYNC_BATCH_IMPORT` kill-switch/A/B arm mirroring `FW_SYNC_BATCH_QUERY`.

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

## Recommendation

Pause the autonomous *byte-exact* loop — further ticks only re-measure settled ground or
land sub-floor micro-levers the ledger reverts. Schedule a deliberate **owner-authorized
model-bench + WER session** and start from row 1 (`FW_ENC_INT8_FC1`, already 5-clip
byte-exact) or row 2 (tiny.en calibration, the bigger prize).
