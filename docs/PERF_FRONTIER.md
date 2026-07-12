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
  design (an owner call), not for any measured risk. **Test-safety is now COMPLETE** (238/0 lib + the
  integration/conformance suites confirmed flag-agnostic by construction: `native_engine_e2e.rs`
  transcribes only `jfk.wav` = single-window, so the retry — which needs a carried prompt / multi-window
  — never fires; `conformance_harness.rs` validates replay/backend metadata, not live native decode).
  No test uses a multi-window native clip, so nothing exercises the retry outside the 238/0 lib suite.

## ⚡ Current-code headline vs whisper.cpp is ~2.2× on the encoder — the "~1.2×" boilerplate is STALE

Measured 2026-07-12 on an **idle box** (load ~7), **turbo**, `jfk.wav`, **matched 32 threads**
(whisper.cpp's *best* — `-t 16`=4382 ms, **`-t 32`=~3210 ms**, `-t 48`=3212, `-t 64`=5448 ms, the
all-core freq-throttle wall [[project_encoder_wall_is_clock_throttle]]):

| stage | fw (ms) | whisper.cpp `-t 32` (ms) | fw speedup |
|---|---|---|---|
| **encoder** (82% of compute, ts-independent) | **~1404** (`encoder_window` span, 3 reps 1420/1379/1414) | **~3210** (`encode`, 3 reps 3227/3231/3184) | **2.29×** |
| decode | ~223 | ~285 `batchd` + 36 `sample` | ~1.4× |
| compute (enc+dec+xkv+mel) | ~1666 | ~3538 | **2.12×** |
| total wall incl. load (single-shot) | ~2593 (compute + 927 load) | ~4398 (compute + 860 load) | **1.70×** |

**The encoder 2.29× is the solid number** (isolated per-engine spans, matched threads, ts-independent, the
dominant stage; fw ships full-int8 encoder, wc runs f16; both reps low-variance). **Why it's a real
regime change and not noise:** [[project_realistic_workload_dominated]] measured the *pre-int8* encoder as
**~tied** (franken 3.53 s/win vs wc 3.19 s/win); the `a997f37` full-int8 ship (2026-07-09) is what moved
franken's encoder ~3.5 s → ~1.4 s. So the ubiquitous `NEGATIVE_EVIDENCE` boilerplate "~1.2× ts / ~1.68–1.8×
no_ts vs whisper.cpp" is **STALE (pre-int8)** — treat those trailing ratios in older entries as such.

**CAVEAT (do not over-quote the total/compute rows):** [[project_realistic_workload_dominated]] proved the
*whole-pipeline* ratio on THIS shared box **swings 0.90–1.22× with ambient load 8→100** (franken's 32-thread
rayon oversubscribes harder than wc's OpenMP under contention) — a true whole-pipeline number needs a
*dedicated quiet* box. This run was load ~7 with stable reps, so the **isolated encoder span** is trustworthy,
but the compute (~2.1×) / total-wall (~1.7×) rows are load-~7 point estimates, not guaranteed floors. Also the
decode row is fw-TS vs wc-`-nt` (not matched). **Quote the encoder; treat the rest as directional.**

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

- **CANDIDATE (byte-exact) — BUILT + MEASURED = WASH, reverted 2026-07-12.** Hypothesis: `quantize_mat_to_i7`
  (nn.rs:573) reads each output column **strided** (`w_t.data[i*out+o]`, stride `out`) — columns `o`/`o+1`
  share a cache line so the column loop re-reads it ~16× ⇒ memory-read-bound, so a **cache-blocked
  transpose-quant** (own a contiguous `BC=64`-column block, sweep rows, read each line ~once) would win.
  Implemented behind `FW_QUANT_BLOCKED`, **byte-identical** (unit-test-passed across partial blocks + live
  shapes; the tricky part was preserving the default **error-feedback** rounding — EF diffuses the residual
  along the contraction dim WITHIN each column, so a per-column `err[BC]` in the i-outer loop keeps it
  bit-exact). **But the `model_weights` span did NOT move: default ≈ blocked ≈ ~640 ms turbo (ABBA, both
  EF and non-EF).** So the 16× read amplification is REAL but NOT the bottleneck: the default path is EF,
  whose per-column serial `err`-chain is **latency-bound not bandwidth-bound**, and `model_weights` is
  anyway dominated by the ggml-Q-format **dequant + f32 convert + layout**, not the i7 re-quant. Reverted
  (byte-exact impl in git history; do not re-attempt cache-blocking this kernel — it's not memory-bound).
  **Lesson: "strided ⇒ memory-bound" is a HYPOTHESIS ([[project_nominal_vs_dram_bytes]]); a serial
  dependency (EF) or a dominating sibling stage can make the amplified reads free.** Load stays "at parity
  with wc"; cold-start latency is not an autonomously-movable lever here.
- **The OTHER load component — f16→f32 weight dequant — is BANDWIDTH-BOUND (probe, 2026-07-12).** The
  turbo weights are f16; `ggml::dequant_f16_parallel` (ggml.rs:623) converts them f16→f32 with a scalar
  per-element `half::f16::to_f32`. Hypothesis: SIMD `HalfFloatSliceExt::convert_to_f32_slice` (F16C
  `vcvtph2ps`, already used in the GEMV path) would beat it. `examples/f16_dequant_probe` (100M f16, 8
  workers, best-of-7): **SIMD 22.5 ms vs scalar 21.7 ms = 0.96× (WASH), 0/100M differing bits.** Both hit
  ~27 GB/s (read f16 + write 2× f32) ⇒ **bandwidth-bound, not compute-bound** — `half::to_f32` already
  autovectorises. So neither piece of `model_weights` is movable: the quant is EF-latency-bound, the
  dequant is bandwidth-bound. **The single-shot load path is fully characterized and at its floor.** (The
  memory's f16-conversion audit [[project_half_from_f32_software_no_site]] covered `from_f32` sites; this
  measures the previously-uncovered `to_f32` LOAD dequant. Don't re-dig either.)

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

## Import N+1 — DONE + flipped default-ON (no byte-exact lever remains)

The sync **export** N+1 is landed (`FW_SYNC_BATCH_QUERY`, ~1.32×). Its mirror, the **import**
path, was the last un-optimized IO site — byte-exact (no quality gate). **It is now COMPLETE.**

**CORRECTION 2026-07-12 (was stale): `FW_SYNC_BATCH_IMPORT` is DEFAULT-ON, not "default-OFF
pending a flip."** The runs/segments/events batch landed (`d2b5b14`/`8199711`/`40fbcdf`) and the
flip to default-on shipped in **`f38d83c` "flip FW_SYNC_BATCH_IMPORT default-ON — measured ~1.29×
import, byte-exact"** (verified against `sync.rs:56` = `… != Some("0")`, comment "**Default ON**",
`FW_SYNC_BATCH_IMPORT=0` kills). All three tables dispatch legacy vs batched through the SAME
`apply_{run,segment,event}_row` conflict logic (differing only in where `existing` comes from):
runs prefetch `WHERE id IN (…)`, composite tables prefetch `WHERE run_id IN (…)` + map by
`(run_id,idx)`/`(run_id,seq)`, each with an intra-chunk seen-map ⇒ byte-identical. Gate:
`sync::tests` 350/0 (now exercised through the batched path by default) +
`flush_{run,segment,event}_chunk_matches_per_line_reference` + full-CLI export→import A/B
byte-identical off-vs-on incl. the conflict/noop re-import path. **There is no pending byte-exact
lever — the soak+flip is already done.** The recipe + hazards below are retained as historical record.

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

Pause the autonomous *byte-exact* loop — further ticks only re-measure settled ground or
land sub-floor micro-levers the ledger reverts. **Rows 1 & 2 of the table above are no longer
the start point** (both MOOT/DONE per the §CORRECTION: `FW_ENC_INT8_FC1` is inert under the
shipped full int8; tiny.en calibration shipped `a997f37`). And the encoder FLOP-reduction row
is **measured dead on CPU** — `NEGATIVE_EVIDENCE` closes all three redundancy axes with data:
DEPTH (layer-pruning fatal at skip-1: `=31` mangles proper nouns + repetition-loops track01 (−27% words) though it's jfk-byte-identical; `=30` breaks even jfk — `7092` + 2026-07-12 update), SEQUENCE (ToMe frames not mergeable,
`4518`), SPECTRAL (weights near-full-rank, `4640`); Nyström/CountSketch/PQ/low-rank/Strassen all
rejected (`4552`). So the genuinely-remaining levers are **owner/infra only**: (1) a **Linux GPU
compute stack** (GTX 1070 is on nouveau → no CUDA/OpenCL/Vulkan — the encoder GEMM/SDPA is the
sole out-of-crate lever); (2) a **cheap multilingual DRAFT model** to unlock speculative decode
(verify amortization R(K)≈3.7× de-risked, but a layer-skip self-draft can't clear break-even —
the drafter must also shrink the logits head); (3) **AVX-512-VNNI hardware** (int8 encoder GEMM
is 0.89× on this AVX2-no-VNNI box). No autonomously-landable byte-exact perf lever remains
(re-verified against current code 2026-07-12: encoder int8 maximal, import N+1 default-on, IO
swept, fresh shipped-tiny.en encoder profile = exp `__expf_fma` ~9% [poly-exp owns it: turbo-on,
tiny.en regressed-off] + rayon `__sched_yield` [contention-inflated] + int8-GEMM bulk — no new
hot spot).
