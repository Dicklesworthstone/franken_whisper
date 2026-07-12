# Perf Frontier — actionable handoff for the next (owner-gated) optimization session

> Forward-looking playbook, not a log. The historical record is `PERF_LEDGER.md`
> (measured wins) and `NEGATIVE_EVIDENCE.md` (rejections + blockers). This file is
> the short answer to "what's left and exactly how to do it." Owned by swarm agent
> **BlackThrush**. Last updated 2026-07-12.

## ⚠ TOP PRIORITY IS NOW A CORRECTNESS BUG, NOT A PERF LEVER (2026-07-12)

The byte-exact **perf** envelope is exhausted (below), but this session surfaced a bigger issue and
worked it end-to-end. **This outranks every perf lever below.** Full write-up: `NEGATIVE_EVIDENCE`
2026-07-12 / `project_final_window_early_eot_bug`.

- **Bug:** shipping default **drops ~48% of long-form content on tiny.en** (bd-r0qd), reproduced on the
  in-repo `example_audio_track_01.mp3` — **two 30 s windows dropped** in TS mode
  (`fw transcribe --json`: gaps `[24.88-54.88]`, `[80.08-110.08]`). A "faster" long-form tiny.en is
  partly from decoding LESS (non-comparable).
- **Root cause:** greedy decode with **no temperature fallback** — a window that closes no timestamp
  (`result_len==0`) while carrying the prior-window prompt (prompt × int8 numerics → early `eot`) is
  dropped (empty, full `CHUNK_CS` advance).
- **Severity bound:** **tiny.en ONLY** — `large-v3-turbo` covers the full clip (no drops); its stronger
  decoder doesn't early-EOT. Quality-seeking (turbo) users unaffected.
- **Landed this session (all byte-exact / default-OFF):** `FW_RETRY_FAILED_WINDOW=1` (`1caba18`) retries
  a failed window once with the prompt cleared → **recovers real audio EXACTLY = whisper.cpp** (track01
  643→1301 chars); a `tracing::warn!` (`1d777f0`) surfaces the otherwise-silent drop (track01 warns 2×
  at seek 24.88/80.08); a `decode_to_wav` example (`e221630`) makes mp3s whisper-cli-readable.
- **CAVEAT keeping the retry OFF:** it drops the prompt entirely for the failed window, so on
  repetitive/tiled audio it re-transcribes covered content (jfk×3 239→379ch). Real speech is fine.
- **Owner action (pick one):** (a) implement the proper whisper.cpp **temperature fallback** (tracks
  `prompt_reset_since`, avoids even the minor tiled +1, covers non-prompt failure modes) in
  `transcribe_samples` — the correct superset; or (b) **flip `FW_RETRY_FAILED_WINDOW` default-on** —
  the case is now fully evidenced (measured, not asserted): fixes the ~48% drop; **more faithful to
  whisper.cpp** on both real audio (recovers exactly) and tiled audio (jfk×3 "country": wc 7, default 4
  DROPS, retry 8 — retry 3× closer to wc); **test-safe** (`FW_RETRY_FAILED_WINDOW=1 cargo test --lib
  native_engine` = 238/0); **cheap** (encode-reused, `f3d8550`); **safety-audited** (retry TS-only,
  pipeline no_ts-only ⇒ no desync). It is left OFF only because it reverses the deliberate greedy/temp-0
  design (an owner call), not for any measured risk. NB: only the LIB native_engine suite was run with
  the flag; the integration/conformance suites route to whisper.cpp so are likely flag-agnostic, but
  confirm before flipping.

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

## ⚠ CORRECTION 2026-07-12: tiny.en FULL int8 is ALREADY SHIPPED (rows 1 & 2 below were STALE)

Verified against current code (`a997f37 "perf(native): default quality-safe encoder int8"`):
`calibrated_encoder_int8_model()` returns `tiny_en || is_large_v3_turbo`, so **tiny.en gets the
full quality-safe int8 encoder (q/k/v/fc1/fc2 i7 + attn.out i8, the ~1.47× lever) DEFAULT-ON** —
it was calibrated `2026-07-10`, not "uncalibrated/pending" as the old rows (and memory) claimed.
Empirically confirmed (prebuilt `fw`, tiny.en): unset **≠** `FW_ENC_ATTN_OUT_I8I32=0` (the f32
kill-switch) and unset **==** `=1` — i.e. the shipped default IS int8. `FW_ENC_INT8_FC1` is
therefore **inert in the default config** (branch precedence: the full-int8 branch runs first;
fc1-only is only reachable with `FW_ENC_ATTN_OUT_I8I32=0`). **This invalidates the a18fed2 "fc1-int8
WER-neutral proxy" evidence** — that transcript-diff compared default-int8 vs default-int8 (a no-op
flag), NOT f32 vs fc1-int8. See NEGATIVE_EVIDENCE 2026-07-12 for the full correction.

## Remaining levers — all need the model-bench + corpus-WER loop + owner sign-off

| lever | est. e2e | evidence in hand | why gated | validate before flip |
|---|---|---|---|---|
| ~~`FW_ENC_INT8_FC1` for tiny.en~~ **MOOT** | — | tiny.en already ships the strictly-more-aggressive FULL int8 (above); fc1-only is superseded/inert in the default config | n/a — not a lever | n/a |
| ~~tiny.en encoder int8 *calibration*~~ **DONE (shipped `a997f37`)** | ~1.47× encoder, LIVE | `calibrated_encoder_int8_model` includes tiny.en; policy `calibrated_model_budget_pass` (asserted by a unit test). Default-on, `FW_ENC_ATTN_OUT_I8I32=0` kills | not gated — shipped | — |
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
**SEGMENTS (`8199711`) and EVENTS (`40fbcdf`) also LANDED** — the import N+1 batch is now
**COMPLETE for all 3 tables** (runs/segments/events) under the single `FW_SYNC_BATCH_IMPORT` flag.
Composite tables prefetch `WHERE run_id IN (…)` + map by `(run_id,idx)`/`(run_id,seq)` + seen-map;
shared `apply_*_row`/`record_*_pre` keep legacy==batched byte-identical. Gate: `sync::tests` 350/0
(+ `flush_{run,segment,event}_chunk_matches_per_line_reference`) + E2E A/B byte-identical off-vs-on
for all 3 tables incl. the conflict/noop re-import path. **The peripheral IO/DB lane is now fully
optimized; nothing left in this vein.** Remaining work is a soak then the default-on flip (a single
flag decision). The recipe + hazards below are retained as the historical record.

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
  **BASELINE ESTABLISHED + BLOCKER SHARPENED (2026-07-12):** the SHIPPING int8 `fw` is
  **byte-identical (normalized) to whisper.cpp on jfk** (`whisper-cli -m tiny.en -nt -t 8` vs
  `fw transcribe --no-persist`) — so the default int8 engine is faithful on real speech; the
  gated-lever WER baseline on jfk is **≈0**. **But the actual gate is the CORPUS, not just owner
  sign-off:** the only clean-speech-with-reference clip on this box is **jfk** — `whisper-cli`
  **cannot read `.mp3`** (so `example_audio_track_01.mp3`, the discriminating real-speech clip,
  has no whisper.cpp reference), and `tests/fixtures/audio/test_10s_speech.wav` is **non-speech**
  (whisper.cpp → "bell dings", `fw` → empty; unusable for WER). ⇒ Evaluating ANY gated lever
  (encoder int8 variants, poly-exp) at corpus-WER first needs the **owner to provide a diverse
  speech corpus with references** (or decoded `.wav`s of the mp3s). The method is proven; the
  data is missing.

## Recommendation

Pause the autonomous *byte-exact* loop — further ticks only re-measure settled ground or
land sub-floor micro-levers the ledger reverts. Schedule a deliberate **owner-authorized
model-bench + WER session** and start from row 1 (`FW_ENC_INT8_FC1`, already 5-clip
byte-exact) or row 2 (tiny.en calibration, the bigger prize).
