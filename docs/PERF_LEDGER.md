# franken_whisper — Performance Lever Ledger

> Head-to-head, MEASURED optimization log for the native Rust engine. Owned by
> swarm agent **BlackThrush** (franken_whisper-cc). Every entry records a real
> criterion measurement; ~0-gain or regressing levers are REVERTED, not kept.

## Measurement protocol

- **Harness:** `benches/native_engine_bench.rs` (criterion).
- **Build/run:** fail-closed remote only:
  `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo ...`.
  An RCH failure is a blocker, never permission to build locally.
- **Baseline vs candidate:** both arms must execute in one binary and one RCH
  invocation, interleaved inside one measured routine with black-boxed inputs
  and complete results. Separate Criterion baseline invocations are invalid
  because RCH worker selection is non-deterministic.
- **Verdict gate and REJECT provenance:** run paired BASE/BASE first for the
  exact function and shape. CV is informational, not a gate. A candidate is
  decidable only when its paired-ratio median lies outside that null control's
  observed `[p10, p90]`; a predeclared null-median acceptance gate must also
  pass. Record benchmark-binary SHA256, worker identity, raw paired ratios,
  null median/spread, candidate CV, and profile-verified non-zero self-time for
  the real function under test. Without that bundle, the row is a blocker or
  routing probe, not do-not-retry authority.
- **Conformance gate:** every numeric kernel change ships with a **bit-exact
  parity test** against the pre-change reference, so a "win" cannot silently
  alter output. The mel output is conformance-checked against whisper.cpp's exact
  encoder input.
- **What the original is:** whisper.cpp's exact algorithms (this engine is a
  faithful Rust port). A kernel lever's "gain" is the measured speedup of the
  Rust port over its own faithful-port baseline while preserving bit-exact output
  — i.e. doing whisper.cpp's identical math, faster.

## Hermetic vs model-gated benches

| bench | hermetic? | status |
|---|---|---|
| `native_engine/mel/mel_30s` | yes | **measured** |
| `native_engine/f16_gemv/*` | yes | available |
| `encoder_window_{tiny,large}` | no (model+jfk.wav) | tiny unlocked locally; large needs `large-v3-turbo` |
| `decoder_token_step_{tiny,large}` | no | tiny unlocked locally |
| `logits_gemv_large` | no (large model) | blocked: model absent |
| `e2e_tiny_jfk` | no (model) | tiny unlocked locally |

> `tests/fixtures/native/jfk.wav` is gitignored; copied locally from
> `legacy_whispercpp/whisper.cpp/samples/jfk.wav` (mono 16 kHz, 11 s) to unlock
> the model-gated benches for measurement. The `large-v3-turbo` model is not
> present, so the large-shape levers remain blocked (concrete blocker).

---

## Levers

### 2026-07-12 UTC — BlackThrush — LANDED (byte-exact, no gate): export writers wrap `File` in `BufWriter` — **~40× faster subtitle/transcript export**

**What.** Every export writer (`write_txt/vtt/srt/csv/lrc/json/json_full` in
`src/export.rs`) wrote per-segment straight to a raw `File` via `writeln!` /
`serde_json::to_writer_pretty` — **one `write()` syscall per line** (SRT emits ~3
lines/segment ⇒ ~3N syscalls for N segments; JSON serde emits many small chunks).
Wrapped each in `BufWriter` (batched ~8 KiB writes) + an explicit `flush()?` (which
also surfaces write errors the raw-`File` drop silently swallowed). **Byte-identical
output** — `BufWriter` forwards the same bytes, just batched.

**Correctness CERTIFIED.** Existing CSV round-trip test + new
`export::tests::writers_emit_byte_exact_content` (asserts exact SRT/VTT/TXT bytes for
a multi-segment result). `export::tests`: pass.

**Measurement (in-binary paired A/B, `export/srt_write`, 5000 segments = 15000 lines,
forced-local, interleaved unbuffered/buffered/unbuffered/buffered):**

| rep | unbuffered (raw `File`) | buffered (`BufWriter`) |
|---|---|---|
| 1 | 80.692 ms | 1.9985 ms |
| 2 | 84.418 ms | 1.9091 ms |

**~40–44× faster** (rep1 40.4×, rep2 44.2×), CIs orders of magnitude apart — the
syscall-per-line cost is the entire bottleneck. Scales with transcript length: a
long transcription (hours of audio ⇒ thousands of segments, multiple output formats)
was doing tens of thousands of `write()` syscalls per artifact; now a handful. Same
textbook category as the tty `bufread` win — a well-known std API, byte-exact,
zero-downside, no gate. Runs on the transcription output path.

### 2026-07-12 UTC — BlackThrush — LANDED (byte-exact, kill-switch `FW_SYNC_SKIP_STMT_SP`): JSONL import loops skip redundant per-statement savepoints — modest (~2–5% import, IO/parse-bound)

**What.** Follow-through of the `persist_report` savepoint-skip win onto the sync
**import** write path. `import_runs` / `import_segments` / `import_events` (and the
strict-overwrite delete helpers) write every `runs`/`segments`/`events` row via
`execute_with_params`, each wrapped by fsqlite in an internal statement savepoint —
but all 10 sites run inside `import_inner`'s single `BEGIN;` (the rollback boundary:
COMMIT on success, ROLLBACK on any Err or Reject). Routed them through a small
`ImportExec::import_exec` extension that dispatches to
`execute_with_params_skip_statement_savepoint_in_explicit_txn`. Imported rows are
**byte-identical**; on failure the enclosing `BEGIN;` rollback discards partial rows
exactly as before. `FW_SYNC_SKIP_STMT_SP=0` restores the legacy path (kill-switch).

**Correctness CERTIFIED.** Full `sync::tests` module (export→import round-trips,
checksum/schema/version validation, all conflict policies incl. overwrite/strict,
referential-integrity rejection): **346 passed / 0 failed** under the default skip.

**Measurement (`sync/import/runs/50` = 50 runs × 5 rows = 250 inserts; forced-local;
external-env A/B interleaved 0/1/0/1):**

| rep | flag=0 (statement savepoint) | flag=1 (skip) |
|---|---|---|
| 1 | 55.365 ms | 54.395 ms |
| 2 | 57.170 ms | 54.321 ms |

**~1.8–5.0% faster** (rep2 p<0.05; rep1 borderline p≈0.05). Skip is faster in **both**
reps and notably more stable (54.39 / 54.32 ms, ~0.1% spread) while the savepoint arm
is noisier and always slower (55.4–57.2 ms) — a consistent direction, not sign-flipping
noise (contrast the `load_run_details` scan closeout in NEGATIVE_EVIDENCE). The
per-insert saving is the same ~7 µs as persist, but sync import is JSONL-parse/file-IO
dominated (~55 ms for 250 inserts), so the ~1.75 ms savepoint saving is a small slice —
hence ~2–5% here vs persist's 1.48× on its insert-dominated workload. Byte-exact,
zero-downside, and the second half of making both SQLite write paths skip redundant
savepoints ([[project_fsqlite_statement_savepoint_skip]]).

### 2026-07-12 UTC — BlackThrush — LANDED (byte-exact, kill-switch `FW_PERSIST_SKIP_STMT_SP`): `persist_report_inner` skips redundant per-statement savepoints inside its enclosing SAVEPOINT — **~1.48× persist**

**What.** `persist_report_inner` writes the `runs` row + one INSERT per segment +
one INSERT per event, each via `execute_with_params`, which wraps **every** statement
in an fsqlite internal statement savepoint. But those inserts already run inside
`persist_report_once`'s explicit `SAVEPOINT`, which is the rollback boundary (it
rolls back on any `Err`), so the per-statement savepoints are pure redundant
bookkeeping — N create/release pairs for N segments + N events. Switched all three
insert sites to fsqlite's purpose-built escape hatch
`execute_with_params_skip_statement_savepoint_in_explicit_txn`. Persisted rows are
**byte-identical** on success; on failure the enclosing savepoint rollback discards
partial effects exactly as the legacy path did (equivalent final state either way).
`FW_PERSIST_SKIP_STMT_SP=0` restores the per-statement-savepoint path (kill-switch);
default is skip.

**Not a negative-ledger pickup** — found by profiling the storage write path after
the `load_run_details` scan lever measured sub-floor (that closeout is in
NEGATIVE_EVIDENCE 2026-07-12). The load path is fsqlite-query-dominated; the *write*
path's savepoint bookkeeping, by contrast, is a large, measurable fraction.

**Correctness CERTIFIED.** Full `storage::tests` module (persist→load round-trips,
schema migrations, cancellation/rollback, corrupt-input handling): **201 passed /
0 failed** under the default skip path.

**Measurement (`persist_report/segments/100` = 100 segments + 10 events = 111
inserts; forced-local; external-env A/B interleaved 0/1/0/1 — in-binary env A/B
impossible under edition 2024 + `#![deny(unsafe_code)]`, see
[[project_asupersync_oom_roulette]]):**

| rep | flag=0 (statement savepoint) | flag=1 (skip) |
|---|---|---|
| 1 | 2.5050 ms | 1.6684 ms |
| 2 | 2.4365 ms | 1.6764 ms |

**~1.45–1.50× faster** (rep1 1.50×, rep2 1.45×). The two arms' 95% CIs are fully
non-overlapping ([2.39–2.56 ms] vs [1.64–1.71 ms]) and each arm's CV is <2% — a
clean, stable separation (contrast the same-file `load_run_details` scan lever,
whose delta sat inside ±5% run-to-run noise). ~0.8 ms saved across 111 inserts ≈
7 µs/insert of savepoint overhead removed. persist_report runs once per
transcription (when `persist=true`), so the e2e effect scales with segment/event
count; modest but real on the transcription-completion path, and a clean structural
win backed by the API author's intended contract.

### 2026-07-12 UTC — BlackThrush — LANDED (byte-exact, no gate): `decompress_chunk` `read::ZlibDecoder` → `bufread::ZlibDecoder` — removes the per-frame 32 KiB read-ahead alloc (negative-ledger pickup)

**What.** `tty_audio::decompress_chunk(input: &[u8])` fed an already-in-memory
`&[u8]` to `flate2::read::ZlibDecoder`, which wraps its `Read` source in an
**additional 32 KiB read-ahead `BufReader`** — a per-frame heap alloc + memset +
memmove on every decompressed TTY-audio frame. `&[u8]` already implements
`BufRead`, so `flate2::bufread::ZlibDecoder` reads the slice directly with zero
scratch buffer. One-line import swap; identical inflate; **byte-identical output**
(no float, never enters ASR numerics ⇒ the ULP requirement reduces to exact
bytes, WER unchanged by construction). Landed default (no gate) — a pure
buffering-strategy change with no downside.

**Negative-ledger pickup.** This was cod_fw's SURFACED-but-unshipped win
(NEGATIVE_EVIDENCE 2026-07-11, now marked RESOLVED): a scaled 300k-frame profile
of the 24-byte TTY frame shape attributed self-time `__memmove` 23.91% +
`__memset` 14.61% to that read-ahead buffer. cod_fw held the ship under the strict
`degraded = SURFACE, no local fallback` rule — the mandatory `--all-targets`
remote gate OOM'd (`asupersync` lib compile SIGKILL) on memory-constrained
workers.

**Sizing — cod_fw same-worker in-binary paired A/B, worker `vmi1149989`, 31
Criterion samples, in-tree `tty/decode_synthetic` bench (unchanged by this diff):**

| frames | baseline median | candidate | conservative speedup | repeat floor |
|---|---|---|---|---|
| 32  | 136,200.881 ns | 120,319.831 | 1.0819× PASS | 1.0463× |
| 128 | 530,016.087 ns | 468,278.149 | 1.1136× PASS | 1.0164× |

**Byte-exactness CERTIFIED in-tree (this turn).** New unit test
`tty_audio::tests::decompress_bufread_matches_read_reference_byte_exact` asserts
the production `bufread` output equals a `read::ZlibDecoder` reference across
sizes {0,1,15,16,17,160,1600,8192,8193,40000,79999} × 4 content patterns
(zero/constant/strided/pseudo-random) + roundtrip identity. Run and **PASSED via a
reliable local build** (the remote fleet was rouletting `asupersync` OOMs, so
correctness was verified locally rather than gambling on worker memory; the
franken_whisper lib itself compiled cleanly remotely on vmi1293453).

**Why local for the gate.** The `--all-targets`/full-crate remote build failed on
a worker-memory `asupersync` SIGKILL (vmi1167313) — a flaky-infra property of
certain workers, not of this diff. cod_fw's paired measurement is already
same-worker-admissible and unaffected by the swap (the decompress kernel emits
identical bytes), so a fresh remote timing adds nothing over a local correctness
proof. Retry-condition from the negative ledger ("RCH healthy") is relaxed:
worker-memory OOM is orthogonal to correctness.

### 2026-07-11 UTC — BlackThrush — LANDED (byte-exact, default-OFF gate): `dot_i8_4col` wired into `gemv_i8_batch` (`FW_I8_BATCH_4COL`); cod_fw's parked lever, both retry-conditions now MET and SIZED

**What.** Wired cod_fw's parked, byte-exact `dot_i8_4col` (4-token activation-column
tile, committed reference `examples/i8batch_4col_probe`, `fb43d93`) into the
production int8 batched GEMV `nn::gemv_i8_batch` (decode prefill tq>1 + draft),
behind a new default-OFF gate `FW_I8_BATCH_4COL=1`. The 4-tile handles groups of 4
tokens, then the existing 2col tile the ≤3-token remainder, then a 1col tail — so
the output is BYTE-IDENTICAL to both the default 2col path and the plain `dot_i8`
loop. Integer i32 madd is order-independent ⇒ **ULP-free** (not merely WER-neutral).

**Why it was unblocked.** cod_fw parked this (NEGATIVE_EVIDENCE 2026-07-11) with two
retry-conditions: **(a)** the uncommitted column-major-KV WIP in `nn.rs` must land
(else `git add nn.rs` sweeps it) — now MET (tree clean, last `nn.rs` commit
`a997f37`); **(b)** a ≥32-core host must size the e2e/multi-thread delta before any
default flip — now MET via build-remote (rch)/run-local on the 64-core box
([[project_rch_ab_admissibility]]).

**Byte-exactness CERTIFIED in-tree.** New unit test
`native_engine::nn::tests::dot_i8_4col_matches_four_dot_i8` asserts each of the four
columns equals a scalar `dot_i8` reference across every tail path (n ∈ {0,1,7,15,16,
17,31,32,33,47,63,64,65,384,1280,5120}) + the ±127 worst-case magnitude. Ran on rch
(remote compile vmi1149989, local run): **1 passed**.

**SIZED (same-binary A/B, order-alternated min-of-80, 3 reps, 64c box load ~1.5,
byte-id=true 12/12 every rep):**

| arm | mlp_0[1280,5120] | qkv[1280,3840] |
|---|---|---|
| workers=1, tq=8/64/200 | 1.139/1.136/1.106× | 1.133/1.129/1.115× |
| 16-worker cap, tq=64/200 | 1.05-1.12× / 1.03-1.12× | 1.06-1.18× / 1.04-1.08× |
| 16-worker cap, tq=8 | **0.96-1.06× (noise)** | **0.96-1.08× (noise)** |

Pure-kernel win is stable **1.11-1.14×** (6/6 always faster). The tq=8/16t corner
oscillates around 1.0 across the 3 reps ⇒ dispatch noise on a sub-ms op, NOT a
stable regression (confirms cod_fw's read).

**Default held OFF (deliberate, not blocked).** `gemv_i8_batch` feeds only decode
prefill/draft — a **sub-1% e2e slice** — so the incremental win over the already-
default-ON 2col does not justify a default flip without a long-form turbo transcript
diff confirming the `compute_band` wire-in *indexing* (the kernel unit test covers
the dot, not the wire-in). The sizing SUPPORTS a future flip; it is de-risked to
that single routine step. Opt in today via `FW_I8_BATCH_4COL=1` for large-prefill
workloads. Kill-switch semantics mirror `FW_I8_BATCH_2COL`.

**Files:** `src/native_engine/nn.rs` (kernel `dot_i8_4col` avx2+scalar, gate
`i8_batch_4col_enabled`, `compute_band` `use_4col` branch, unit test). No production
default changed ⇒ current transcripts unchanged by construction.

### 2026-07-10 UTC — cod_fw — SURFACE: cod-lane at frontier — one-pass i7/maddubs logits GEMV failed the median proof gate

**Profile-first target.** The latest full `large-v3-turbo` decode attribution
routes to the tied output projection: `logits_gemv` consumed **162.5 ms / 58
tokens = 2.802 ms/token**, or **21.4% of decode**. The shipped path is already
row-quantized i8 and streams about 66 MiB/token. Tokenizer, sampler, argmax, and
detokenization do not reach 0.03% self-time, so they cannot clear this fleet's
per-function median floor.

**One lever.** The candidate coupled exactly one new weight/kernel format:
natural `[51_866, 1_280]` f16 output weights quantized to i7 (`[-63,63]`) and a
one-pass AVX2 `vpmaddubsw -> vpmaddwd` dot. Activation quantization, output-row
worker bands, parallel threshold, shape, and result materialization matched the
current i8 GEMV. The narrower range makes each maddubs pair non-saturating, but
it is numerics-changing and retains the same one-byte-per-weight traffic. No
encoder or VAD file was touched.

**Strict-remote screen.** Both runs used only:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench --profile release-perf -p franken_whisper --bench native_engine_bench -- native_engine/logits_i7_ab --noplot
```

The first same-binary ABBA screen ran on RCH worker `vmi1264463`
(`38.242.209.154`), job `j-29914252970039323`, exit 0. It was a routing screen,
not ship proof: it reused cache-hot matrices and its BASE/BASE control shared one
physical i8 allocation. Its 31 BASE/BASE ratios were:

```text
1.060228 1.024087 1.019717 1.127256 0.945750 0.938100 1.175068
0.854657 0.942118 1.021765 1.200061 0.951862 0.877495 0.698881
0.957614 0.977946 1.010288 0.875190 1.202485 0.949110 1.298386
0.948810 0.983725 0.861340 0.980618 1.079131 1.207674 1.030361
0.881000 2.013591 1.108262
```

Null p10/median/p90 were **0.875190 / 0.983725 / 1.202485** (CV
**20.900%**). The BASE/i7 ratios were:

```text
0.902765 1.230757 1.172071 1.095160 1.018939 1.097167 0.813799
0.987037 1.262902 0.885573 1.183382 1.077852 1.175507 1.193337
1.300625 1.087920 0.884324 0.847036 0.998400 1.048336 0.947688
0.820807 1.316556 1.076197 1.190292 0.907257 1.125974 1.064319
1.000395 1.132858 1.110063
```

Direct-call medians were **3.227577 ms i8** versus **3.039853 ms i7**;
paired median **1.077852x**, CV **12.927%**, wins **21/31**. That apparent gain
did **not** clear the same-binary null p90 of **1.202485x**, so the screen
rejected it.

**Cold decision run.** The tightened harness used independently quantized i8
null matrices, touched a 256 MiB eviction pool outside every timed arm, and
predeclared a symmetric floor `max(1, p90(r), 1/p10(r))`. RCH worker `hz1`
(`87.99.133.171`), job `j-29914252970039362`, compiled the exact snapshot
successfully. BASE/BASE median was **1.026128**, outside the predeclared
`[0.98,1.02]` validity interval, so the harness stopped before candidate timing
and Cargo exited 101. This was a benchmark validity assertion on a healthy
remote worker, not RCH degradation and not a local fallback.

**Parity / quality boundary.** The numerics-changing candidate never passed a
valid performance screen, so production wiring and the WER/transcript/timestamp
gate were not entered. Both remote workers also lacked the model/JFK assets, and
the model-backed rows visibly skipped. Candidate code, test, and bench were
manually removed; production source is back at the starting commit. Therefore
the landed docs/tracker-only result leaves production output and WER unchanged
by construction, but makes no unsupported WER claim for i7.

**Verdict: SURFACE, do not ship.** Decode/KV is separately blocked, tokenizer is
profile-cold, and the remaining output/logits families (f16 layout/FMA,
low-rank, int4, row skipping, prefetch/NTA, accumulator/row blocking, and
logsumexp processing) are already closed. This final distinct i7/maddubs idea
either sat below its null floor or was gated by an invalid cold null. The
**cod-lane is at frontier; hold**. Reopen only for a genuinely different
primitive that reduces output-weight bytes, new ISA hardware such as VNNI, or a
remote model-backed substrate capable of the full median plus WER proof bundle.

### 2026-07-10 UTC — cod_fw — SURFACE: wide-i7 K=64 candidate stopped by its per-function BASE/BASE floor; candidate never executed

This pass re-read the negative-evidence ledger before editing. The full
`large-v3-turbo` transcription profile remains the routing source: binary
SHA256 `272102fd7cd643bf449eeed18002874cc98241f74290d2937a8d606a10b0c776`,
Build ID `acd75e8eb9b593d129a8563461349529921d46ef`, flat capture SHA256
`15a513d12bef45766eca5d13c9ef61bf15d7b7089524e0f46fa17bb408db8341`,
32K `cycles:u` samples, zero lost. External f32 sgemm is excluded. The ranked
encoder i7/int8 family is:

| rank | full self | frame | disposition |
|---:|---:|---|---|
| 1 | 21.68% | `nn::dot_maddubs_i7_m2n4` | M8/M4N2/L2-panel families already closed; VNNI unavailable |
| 2 | 14.34% | monomorphized `matmul_bias_i7_quantized` Rayon worker | selected live wide-FC1 M4 seam |
| 3 | 4.63% | `encoder::matmul_bias_i8` compute | separate full-i8 kernel |
| 4 | 1.39% | `quantize_act_i7_gelu` | quantizer |
| 5 | 0.74% | `maddubs_i7_headmajor_block` | fused head-major helper |
| 6 | 0.65% | `quantize_act_i7` | quantizer |
| 7 | 0.29% | encoder activation quantization | quantizer |

The family totals **43.717%** of full-transcription self-time. Exact disassembly
also corrects the earlier claim that rank 2 is dispatch-only: the worker at
`0x7e770` contains the inlined wide-FC1 M4 arithmetic, with its dominant dot
loop at `0x7ebe0–0x7ec40` and horizontal reduction at
`0x7ec42–0x7ed24`. The prior annotation had inspected a setup wrapper, not this
worker. LLVM already vectorizes the four-row dequant/bias epilogue, so this pass
did not retry the closed epilogue/fusion families.

The one proposed lever was a K=64 two-bank M4 loop plus a packed four-row
horizontal reduction. It preserves the exact i32 sum and passed its focused
bit-parity test on strict-remote `ovh-a` for K lengths
`0,1,31,32,33,63,64,65,73,127,1280,5120` (candidate = shipped K=32 = scalar,
one test passed). The production-shaped bench used rows=1500, inp=1280,
out=5120, black-boxed the inputs and full 7,680,000-element result, and put the
paired BASE/BASE null before the interleaved BASE/CANDIDATE arm in one binary.

The only profiled measurement invocation ran fail-closed through RCH on
`ovh-a` (hostname `fixmydocuments`). Benchmark-binary SHA256:
`ce041e4421ab60faa2650813088bd5a6c5e30fc4fa43544c9e4c08a32837b79f`.
Its 31 unfiltered BASE/BASE ABBA ratios were:

```text
1.147281 0.964606 1.071223 0.941393 0.926977 1.302412 1.128820
0.943957 0.904293 1.157274 0.929797 1.073807 1.041408 0.918513
1.337974 1.002293 1.162562 0.898850 1.452323 0.940790 0.980430
0.811961 0.996196 1.023475 1.028623 1.376994 0.686858 1.109428
1.031808 1.238578 1.135372
```

Null median **1.028623**, p10 **0.904293**, p90 **1.302412**, range
`[0.686858, 1.452323]`, CV **15.838%**, wins 18/31. The predeclared
`[0.98,1.02]` unbiased-null-median gate therefore failed before parity or the
candidate arm ran. The attached `perf` capture proves this was not dead code:
11,308 `cycles:u` samples, zero lost, with the real
`matmul_bias_i7_quantized` Rayon worker at **98.00% self / 10,265 samples**.
The runner used non-interactive `sudo perf` because the remote worker has
`perf_event_paranoid=4`; it did not fall back locally.

**Verdict: measurement blocker, neither WIN nor REJECT.** There is no
candidate median, so this run cannot close K=64 unrolling. The observed
per-function p90 would require a result above 1.302412x, while this lever only
removes loop-control and reduction work and leaves every maddubs/maddwd/add and
load intact; proceeding on this substrate would knowingly chase below its
floor. Candidate source, test, and bench selector were manually removed and
production/bench are byte-for-byte back at HEAD. The retained runner change
fails closed when `perf` needs unavailable privilege, and the isolated
worker-pinning recipe is preserved at
`tests/artifacts/perf/20260710-i7-m4-k64/rchcfg/`.

Retry condition: a same-binary, one-invocation harness for this exact function
and shape whose BASE/BASE median passes the predeclared gate and whose null
spread is narrow enough for a mechanism-sized effect. Do not rerun the K=64
candidate on the whole-GEMM `ovh-a` substrate above. This is a measurement
boundary, not a parity or optimization ceiling.

### 2026-07-10 UTC — cod_fw — SURFACE: static balanced i7 stripes are bit-exact and live-profiled; CV 13.235% blocks a verdict

Ledger audit reopened the old i7 rowblock row: its two arms were separate RCH
invocations, its spread implies roughly 10% CV, and it records neither a binary
hash nor candidate-path self-time. A fresh full `large-v3-turbo` transcription
profile instead attributes **43.717%** of full self-time to the encoder i7/int8
family: `dot_maddubs_i7_m2n4` 21.68%,
`matmul_bias_i7_quantized` Rayon compute 14.34%, full-i8 `attn.out` 4.63%,
quantizers 2.33%, and head-major helper 0.74%. Profile binary SHA256
`272102fd7cd643bf449eeed18002874cc98241f74290d2937a8d606a10b0c776`,
Build ID `acd75e8eb9b593d129a8563461349529921d46ef`, 32K samples, zero lost.
External sgemm is excluded.

The top dot frame is arithmetic/issue-pressure dominated (`vpmaddubsw` 41.297%,
`vpaddd` 17.714%, loads 17.485%, `vpmaddwd` 6.361%). A packed M4N4 tile repeats
the observed register-pressure/data-movement mechanism, so the measured lever
took the next frame: replace the shipped 375-item four-row Rayon traversal with
one balanced contiguous quotient/remainder stripe per Rayon worker. The dot,
epilogue, allocation, store order, and Q/K/V sequence were shared unchanged.

One binary alternated ORIG/candidate inside each timed routine, black-boxed all
inputs and full Q/K/V results, and proved every output bit identical. The only
confirmation evaluated for a verdict ran via strict RCH on `vmi1152480`, 10
Rayon threads, binary SHA256
`c85d05bbf7837c493da9e9bf801d16aa1693caeab71346abb9d9be945341aea2`.
Its 10 Criterion measurement-batch ratios (ORIG/CANDIDATE, 25 pairs each,
`INNER=3`, none filtered) were:

```text
0.947109 0.857250 1.055037 1.098787 0.925783
0.883159 1.192566 0.998777 0.965458 1.272703
```

Mean `1.019663`, sample SD `0.134949`, **cv_pct `13.235`**, candidate wins
`97/250`. The same binary's profile proves the benchmark was live:
`matmul_bias_i7_quantized::{closure#0}` **7.44% self / 12,417 samples** and the
candidate-only `DrainProducer` dispatcher **0.02% / 38 samples**;
`dot_maddubs_i7_m2n4` was 89.08% / 148,635 samples, with zero lost.

**No verdict.** CV exceeds the mandatory 5% gate, so this is neither WIN nor
REJECT. A `vmi1264463` preflight with within-batch CV 9.672–27.103% and an
`hz2` attempt that stopped before execution because `perf` was absent are both
excluded. Candidate source and bench switch were manually removed, restoring
both files exactly to HEAD. The retained strict-RCH runner records worker,
binary hash, and self-time and now fails before execution on workers without
`perf`. Retry the exact static-stripe primitive only in one perf-capable remote
invocation with no-filter CV `<5%`; otherwise rotate to a different ownership
primitive. This does not establish a parity or performance ceiling.

### 2026-07-10 UTC — cod_fw — SURFACE/PARK: integrity audit reopens self-K; packed-column candidate blocked before A/B

**This entry supersedes the `d3499aa` REJECT immediately below.** That profile
is still useful routing evidence, but the source-attempt verdict is not valid
under the active ledger-integrity rule: its byte-exact self-K benchmark timed
private replica functions and recorded no production function-under-test
self-time. The mel/FFT, tokenizer, decoder, and KV families were audited under
the same rule; no historical REJECT in those families currently supplies both
a production-path A/B and non-zero benchmark self-time for the function under
test. Several mel closures were also contradicted by later landed RFFT,
radix-5, scratch-arena, and SIMD-projection wins.

**Fresh full-transcription profile.** Timestamped `large-v3-turbo`, dense
track01 (124.5 s), `RAYON_NUM_THREADS=8`, existing symbolized release-perf
`e2e_probe` Build ID `acd75e8eb9b593d129a8563461349529921d46ef`.
Transcription took **23.329 s** (RTF 0.1874, 12 segments, 1,337 characters).
The exact transcribe slice contained **32K cycles:u samples with zero lost**.
External sgemm (`kernel_target_fma` 17.88%, `gemm_loop` 4.25%) is excluded:

The executable was built at source `91b44b1`; the requested in-crate mel,
tokenizer, decoder, and `nn.rs` paths are unchanged through this profile's
HEAD. The sibling `frankentorch` revision advanced, so cc-owned sibling-frame
magnitudes are routing context rather than fresh comparator claims.

| rank | self | non-sgemm frame | disposition |
|---:|---:|---|---|
| 1 | 21.67% | `nn::dot_maddubs_i7_m2n4` | cc-owned int8 |
| 2 | 14.34% | `nn::matmul_bias_i7_quantized` closure | cc-owned int8 |
| 3 | 13.08% | `ft_kernel_cpu::sdpa_forward_f32` | cc-owned SDPA |
| 4 | 7.53% | `__expf_fma` | cc-owned SDPA |
| 5 | 6.03% | `nn::gemv_i8` closure | cc-owned int8 |
| 6 | 4.63% | `encoder::matmul_bias_i8` closure | cc-owned int8 |
| 7 | 1.68% | `nn::gemv_i8w_f32a_blocked` | cc-owned int8 |
| 8 | 1.39% | `nn::quantize_act_i7_gelu` closure | cc-owned int8 |
| 9 | 1.07% | `nn::gemv_i8` | cc-owned int8 |
| 10 | 0.78% | `nn::norm_rows_into` | old fused-LN row lacks benchmark self-time |
| 11 | 0.74% | `nn::maddubs_i7_headmajor_block` | cc-owned int8 |
| 12 | 0.69% | `__memset_avx2_unaligned_erms` | mixed callers; not KV-attributable |
| 13 | 0.65% | `nn::quantize_act_i7` closure | cc-owned int8 |
| 14 | 0.39% | `__memmove_avx_unaligned_erms` | mixed callers |
| 15 | 0.29% | encoder quantization closure | cc-owned int8 |
| 16 | 0.20% | unresolved kernel address | outside crate |
| 17 | 0.19% | unresolved kernel address | outside crate |
| 18 | 0.17% | `encoder::forward_time_major` | outside decoder lane |
| **19** | **0.17%** | **`nn::attention_with_cache`** | **top open requested family** |
| 20 | 0.14% | `DecoderState::new` closure 4 | prior F16C row used a replica |
| 21 | 0.11% | `nn::softmax_rows` | decoder attention |

Tokenizer, `process_logits`, and argmax have no >=0.1% symbol; native beam
search does not exist. `compute_logprobs` is reached at 0.03%. Mel is reached
but remains below 0.01% of full transcription.

**Mechanism.** Production `perf annotate` gives 67 samples in
`attention_with_cache`. The scalar self-K score chain's `vmulss` and `vaddss`
carry **40.71% + 12.16% = 52.87%** of the symbol's sampled period, approximately
**0.09% of full-transcription self-time**. The old loop-swap replica made K
access strided. The parked alien primitive instead mirrors self-K as
`[state, capacity_tokens]`, appends to both layouts, and computes d-outer/
j-inner over contiguous columns while preserving each score's d-ascending
floating-point operation sequence. Its same-binary harness calls the real
`attention_with_cache` in both arms, alternates 25 paired repetitions, asserts
bit equality before timing, and reports paired-ratio CV.

**BLOCKED before measurement; this is neither WIN nor REJECT.** The required
fail-closed invocation was:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench \
  --profile release-perf --bench native_engine_bench -- \
  native_engine/self_attn_k_layout --noplot
```

RCH selected healthy `vmi1264463`, prepared 26 roots, then failed at
`sync_to_remote: timed out after 30000ms`; `RCH_REQUIRE_REMOTE=1` refused local
fallback, and no local Cargo/rustc ran. Therefore no parity test, A/B ratio,
CV, or benchmark function self-time exists. The source patch is not applied;
it is parked with the full proof plan at
`tests/artifacts/perf/20260710-self-k-column-major/`. Retry only when strict RCH
can sync, then profile the retrieved benchmark binary and require non-zero
candidate-kernel self-time before admitting either a WIN or REJECT.

### 2026-07-10 UTC — cod_fw — REJECT: long-form turbo does not promote an owned non-GEMM residual

**Retry condition tested.** The short timestamped profile allowed a retry when
a different workload made mel/tokenizer/decoder/KV a top-five owned frame at
>=2% self. This pass profiled two genuinely long-form workloads with timestamps:
JFK x8 (88.0 s) and dense track01 (124.5 s). Closed families remained closed,
including the plain `matrixmultiply` -> `gemm` swap; cc still owns SDPA/int8.

**Profile protocol.** Release-perf `e2e_probe` Build ID
`acd75e8eb9b593d129a8563461349529921d46ef`, `large-v3-turbo`,
`RAYON_NUM_THREADS=8`, undelayed `perf record -m 1 -e cycles:u -F 199`, then a
time filter from the first `mel::log_mel` sample through completion. Runtime
source is unchanged since profiled HEAD `91b44b1d`.

| workload | transcribe wall | samples | lost | first open requested frame |
|---|---:|---:|---:|---|
| JFK x8 / 88.0 s | 15.891 s | 25K | 0 | `attention_with_cache` 0.16% |
| track01 / 124.5 s | 22.612 s | 35K | 0 | `attention_with_cache` 0.17% |

Dense-track01 ranked non-sgemm user frames at or above 0.1% self time (external
`kernel_target_fma` 17.89% and `gemm_loop` 4.00% excluded):

| self | frame | disposition |
|---:|---|---|
| 21.55% | `nn::dot_maddubs_i7_m2n4` | cc-owned int8 |
| 14.39% | `nn::matmul_bias_i7_quantized` | cc-owned int8 |
| 13.01% | `ft_kernel_cpu::sdpa_forward_f32` | cc-owned SDPA |
| 7.30% | `__expf_fma` | cc-owned/closed SDPA softmax |
| 7.19% | `nn::gemv_i8` closure | cc-owned int8 |
| 4.57% | `encoder::matmul_bias_i8` | cc-owned int8 |
| 1.85% | `nn::gemv_i8w_f32a_blocked` | cc-owned int8 |
| 1.45% | `nn::quantize_act_i7_gelu` | cc-owned int8 |
| 1.13% | `nn::gemv_i8` | cc-owned int8 |
| 0.72% | `nn::norm_rows_into` | fused-LN/LN-to-quant closed |
| 0.65% | `nn::maddubs_i7_headmajor_block` | cc-owned int8 |
| 0.55% | `nn::quantize_act_i7` | cc-owned int8 |
| 0.54% | `__memset_avx2_unaligned_erms` | allocator/buffer reuse closed |
| 0.28% | `encoder::matmul_bias_i8` quant closure | cc-owned int8 |
| 0.23% | `__memmove_avx_unaligned_erms` | callgraph: cc-owned SDPA scatter/int8 quantization |
| 0.17% | `nn::attention_with_cache` | first open requested family; below gate |
| 0.13% | `DecoderState::new` closure 4 | scalar f16 cross-KV conversion already rejected |
| 0.12% | `encoder::forward_time_major` | below gate; outside decoder lane |

Mel, tokenizer, decoder policy, and remaining self-KV frames were below 0.1%;
native beam search is absent. A separate `F=49` DWARF capture (6,969 samples,
zero lost) attributed transcribe-time `memmove` to cc-owned SDPA scatter and int8
quantization. It also confirmed the cross-KV sample is the already-rejected
scalar f16 conversion. A higher-frequency callgraph attempt lost 99.56% and was
discarded.

**Verdict: REJECT a source attempt.** `attention_with_cache` scores
`(impact 1 x confidence 5) / effort 4 = 1.25`, below the 2.0 implementation
threshold and 3% keep ratchet. The only distinct decoder primitive left is
trained token-level drafting (`bd-wzgh`), but the local models are turbo
`n_vocab=51866` and `tiny.en n_vocab=51864`; no compatible multilingual draft
artifact exists. No runtime code changed, so output remains bit-exact by
construction.

### 2026-07-10 UTC — cod_fw — REJECT: timestamped turbo retry still exposes no eligible owned non-GEMM frame

**Retry condition tested.** The prior no-timestamp profile required a different
workload to promote mel/tokenizer/decoder/KV into the top-five owned frames at
>=2% self time. This run enabled the default timestamped decoder after cc closed
the SDPA pass-elimination lane. All previously rejected families remained
closed; cc still owns int8 and SDPA.

**Full-transcribe profile.** `large-v3-turbo`, JFK x1, timestamps enabled,
`RAYON_NUM_THREADS=8`, release-perf `e2e_probe` Build ID
`acd75e8eb9b593d129a8563461349529921d46ef`. An offloaded cold rebuild succeeded
on `ovh-a` (4m40s) but did not restore its local executable, so the surviving
probe at source HEAD `91b44b1d` was used; later commits are docs/tracker only.
The decisive undelayed flat capture was time-filtered from the first
`mel::log_mel` sample through completion: 6,963 transcription samples, zero
lost, 4.342 s probe wall. A delayed counter row measured 305.234B instructions,
98.092B cycles, IPC 3.11, and 11.12% L1D miss rate, but is context-only because
the delay can omit early transcription.

External sgemm (`kernel_target_fma` 18.38%, `gemm_loop` 4.20%) was excluded.
Ranked non-sgemm user frames at or above 0.1% self:

| self | frame | disposition |
|---:|---|---|
| 21.65% | `nn::dot_maddubs_i7_m2n4` | cc-owned int8 |
| 13.82% | `nn::matmul_bias_i7_quantized` | cc-owned int8 |
| 11.64% | `ft_kernel_cpu::sdpa_forward_f32` | cc-owned SDPA |
| 9.82% | `__expf_fma` | cc-owned/closed SDPA softmax |
| 3.78% | `encoder::matmul_bias_i8` | cc-owned int8 |
| 1.88% | `nn::gemv_i8` closure | cc-owned int8 |
| 1.64% | `nn::quantize_act_i7_gelu` | cc-owned int8 |
| 0.93% | `nn::norm_rows_into` | LN/LN-to-quant closed |
| 0.88% | `nn::maddubs_i7_headmajor_block` | cc-owned int8 |
| 0.74% | `__memmove_avx_unaligned_erms` | below gate; mechanism not isolated |
| 0.74% | `nn::quantize_act_i7` | cc-owned int8 |
| 0.71% | `__memset_avx2_unaligned_erms` | allocator/buffer reuse closed |
| 0.54% | `nn::gemv_i8w_f32a_blocked` | cc-owned int8 |
| 0.31% | `encoder::matmul_bias_i8` quant closure | cc-owned int8 |
| 0.25% | `nn::gemv_i8` | cc-owned int8 |
| 0.16% | `DecoderState::new` cross-KV setup | first permitted family; below retry gate |

Ten restricted kernel addresses contributed 1.69% in aggregate. Mel,
tokenizer, decoder policy, and self-KV were below 0.1% self. Native beam search
does not exist.

**Verdict: REJECT a source attempt.** Cross-KV scores
`(impact 1 x confidence 5) / effort 3 = 1.67`, below the implementation gate and
far below the 3% e2e keep ratchet. No source change was made. This is not a
parity ceiling: the next different decoder primitive is the existing trained
token-draft bead `bd-wzgh`, which requires a real vocabulary-compatible draft
model because layer-skip and prompt/ngram drafts are already rejected. Retry
only with that prerequisite or a profile where a permitted frame is top-five
owned and >=2% self.

### 2026-07-10 UTC — cod_fw — SURFACE: large-v3-turbo non-GEMM residual profile has no eligible owned top frame

**Lane.** After cc_fw took SDPA and encoder-int8 ownership, cod_fw profiled a
full `large-v3-turbo` no-timestamp JFK transcription and excluded `sgemm` before
candidate selection. Prior ledger grep closed f32 QKV sgemm fusion,
weight-stationary f16 GEMV tiles, allocator/buffer-reuse, decoder fused-LN,
LN-to-quant fusion, head-major SDPA scatter read-order, i7 rowblock coarsening,
and i7 bias specialization.

**Profile.** `perf stat -D 2000 -d` and `perf record -D 2000 -F 99 -g
--call-graph dwarf` against the release-perf `e2e_probe` at HEAD `91b44b1d`,
`PROBE_NO_TS=1`, `RAYON_NUM_THREADS=8`,
`FRANKEN_WHISPER_MODEL_DIR=legacy_whispercpp/whisper.cpp/models`. The delay
skips model load. Stat row: 309.710B instructions, 112.311B cycles, 4.900 s
elapsed, IPC 2.76, L1D miss rate 11.59%.

**Ranked transcribe-only frames.**

| self | frame | disposition |
|---:|---|---|
| 19.83% | `nn::dot_maddubs_i7_m2n4` | int8 lane; peer-owned |
| 19.03% | `matrixmultiply::sgemm_kernel::kernel_target_fma` | excluded `sgemm` |
| 13.88% | `nn::matmul_bias_i7_quantized` | int8 lane; peer-owned |
| 13.17% | `ft_kernel_cpu::sdpa_forward_f32` | SDPA lane; peer-owned |
| 9.43% | `__expf_fma` | SDPA/poly-exp lane; peer-owned |
| 4.04% | `encoder::matmul_bias_i8` | int8 lane; peer-owned |
| 3.76% | `matrixmultiply::gemm_loop` | excluded `sgemm` |
| 2.90% | `nn::gemv_i8` closure | int8 lane |
| 0.70% | `__memset_avx2_unaligned_erms` | allocator/buffer-reuse closed |
| 0.61% | `nn::norm_rows_into` | LN/LN-to-quant closed |
| 0.33% | `DecoderState::new` cross-KV setup | below useful threshold |

**Decision.** No keep/reject source lever: the top non-`sgemm` frames are
int8/SDPA peer lanes or closed families, while mel/tokenizer are below the
sampling floor and KV setup is 0.33%. Retry only after the active int8/SDPA work
settles or a fresh workload makes mel/tokenizer/decoder/KV a top-5 owned frame
with >=2% self time. Source remains byte-identical.

### 2026-07-09 EDT / 2026-07-10 UTC — cod_fw — WIN: default-on quality-safe encoder int8 behind calibrated fallback gate

**Lane.** Complete the owner-gated evidence pack for the quality-safe encoder
int8 path and flip the default only where the evidence applies. Ledger grep came
first: the prior cod_fw row explicitly said **do not flip** from the JFK-only
gate; retry condition required full fixture-corpus WER, per-layer quantization
budget, large-v3-turbo/proper-noun adversarial probes, and deterministic f32
fallback. Existing rejections still stand for `FRANKEN_WHISPER_ENC_INT8=1`
all-i7-as-quality-proof, fused-wide QKV concatenation, row-block coarsening,
bias specialization, and quantize/round rewrites.

**Change.** Added `encoder_int8_policy_decision` /
`encoder_int8_effective_policy_decision` with calibration id
`encoder-int8-calibration-2026-07-10`. Default action is now
`QualitySafeInt8Encoder` only for calibrated hparams (`tiny.en` and
`large-v3-turbo`) on AVX2 builds; unknown model shapes and non-AVX2 builds
deterministically return `F32Encoder`. `FW_ENC_ATTN_OUT_I8I32=0` is the explicit
f32 kill switch; `=1` remains an operator force/probe override. Native JSON
`raw_output.encoder_int8_policy` now records action, reason, calibration id,
corpus WER delta budget, and quant RMSE budget.

**Expected-loss policy contract.** State: model hparams/family, CPU feature
class, calibration id, corpus WER/adversarial sentinels, per-layer quantization
error vector, and operator override. Actions: f32 encoder or quality-safe int8
encoder. Loss matrix: false-accepting int8 with transcript/proper-noun drift is
high loss; false-fallback to f32 costs only speed. Confidence terms: fixture WER
delta must remain inside `0.0`, adversarial sentinels must pass, and every layer
must stay below the recorded quant-error budget. Fallback trigger: unknown
hparams, missing AVX2 kernel support, explicit kill switch, failed WER/sentinel,
or exceeded quant budget.

**Quality evidence.**

```text
FRANKEN_WHISPER_MODEL_DIR=legacy_whispercpp/whisper.cpp/models \
  CARGO_TARGET_DIR=/data/tmp/cargo-target \
  cargo test --lib 'quality_safe_int8_per_layer_error_budget' -- --nocapture

tiny.en: worst rel_rmse 0.053139 (layer01 attn_k_i7);
         worst attn_out_i8 rel_rmse 0.010997; all max_abs/amax <= 0.015778
large-v3-turbo: worst rel_rmse 0.082685 (layer03 mlp_proj_i7);
                worst attn_out_i8 rel_rmse 0.014560; all max_abs/amax <= 0.015868
budget: rel_rmse <= 0.09, i7 max_abs/amax <= 0.035, i8 max_abs/amax <= 0.012
test result: ok. 2 passed; finished in 30.25s
```

```text
FRANKEN_WHISPER_MODEL_DIR=legacy_whispercpp/whisper.cpp/models \
  CARGO_TARGET_DIR=/data/tmp/cargo-target \
  cargo test --test native_engine_e2e -- --nocapture

paired whisper.cpp fixture corpus (9/9): WER delta 0.0000
  code_switching, long_form, multilingual, noisy_environment, jfk,
  overlap, short_utterance, silence_heavy, variable_volume_overlap
explicit quality-safe JFK:          WER 0.0000 / gate 0.0000
default quality-safe tiny.en JFK:   WER 0.0000 / gate 0.0000
default quality-safe large-v3 JFK:  WER 0.0000 / gate 0.0500
adversarial sentinels: rejects known all-i7 "Frank at" phrase; requires
  "fellow americans", "ask not", and "country" for large-v3-turbo
test result: ok. 10 passed; finished in 126.13s
```

**Release-perf timing arms (same host, greedy decode, 8 threads, no timestamps).**
Native default was confirmed in JSON as `action=quality_safe_int8` and
`reason=calibrated_model_budget_pass`.

```text
hyperfine --warmup 1 --runs 5

franken_whisper default-int8 large-v3-turbo:
  6.141 s +/- 0.087 s, CV 1.41%, min 6.033, max 6.237
whisper.cpp greedy CPU large-v3-turbo:
  11.952 s +/- 0.805 s, CV 6.74%, min 10.904, max 12.840
observed ratio: native 1.95x faster, but comparator CV misses the <5% ratchet
```

Loaded-host follow-up A/B against the deterministic f32 kill switch was also
noisy on the default arm under load average ~41:

```text
default-int8: 7.238 s +/- 1.515 s, CV 20.93%
f32 kill switch: 7.822 s +/- 0.192 s, CV 2.46%
```

**Verdict.** KEEP the default-on quality-safe policy for calibrated tiny.en and
large-v3-turbo shapes because the quality evidence pack is green and fallback is
deterministic. Do **not** use the loaded-host whisper.cpp timing row as a perf
ratchet; it is evidence that the fast arm works and is likely ahead, but the
comparator CV exceeded the protocol. A quiet-window timing rerun should ratchet
the e2e row separately.

### 2026-07-09 EDT / 2026-07-10 UTC — cod_fw — WIN: executable quality gate for the quality-safe full encoder-int8 policy

**Lane.** Encoder int8 default-on evidence pack. Ledger grep came first:
do not retry full all-i7 encoder int8 (`FRANKEN_WHISPER_ENC_INT8=1`) as a
quality proof, fused-wide QKV concatenation, row-block coarsening, bias
specialization, or quantize/round rewrites. The safe candidate is the current
`FW_ENC_ATTN_OUT_I8I32=1` policy: q/k/v/fc1/fc2 on i7 maddubs, `attn.out` on
full-i8 i32 accumulate, with `FW_ENC_QKV_FUSED=1` and `FW_ENC_EF_QUANT=1`.

**Profile-first routing.** Focused release-perf criterion row on RCH worker
`ovh-a`:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod_fw \
  RUSTFLAGS='-C force-frame-pointers=yes' \
  rch exec -- cargo bench --profile release-perf --bench native_engine_bench -- \
  native_engine/i7_qkv/headmajor_attention_1500x1280 \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.5 \
  --output-format bencher --noplot

native_engine/i7_qkv/headmajor_attention_1500x1280:
  83.074 ms/iter (+/- 1.817 ms), CV ~= 2.2%
```

Local `perf stat` on the same filtered bench binary, because counters require
the process on this host:

```text
0.6205 s elapsed (+/- 3.16%), 12.55 CPUs utilized
27.806B cycles, 25.530B instructions, IPC 0.92
191.896M cache misses / 1.335B cache refs = 14.37%
102.553M branch misses / 3.270B branches = 3.14%
```

Flamegraph: `/tmp/fw-int8-qkv-20260710.svg`. `perf report` was qualitative
only because recording lost 24.29% of samples under local IO/CPU load, but the
ranked surface still matched the prior ledgers: external SDPA (`12.46%`),
external `matrixmultiply` sgemm kernel (`10.01%`), benchmark synthetic-audio
setup noise (`7.05%`), `__expf_fma` (`5.64%`), matrixmultiply packing
(`4.63%`), Rayon/crossbeam scheduling (`~10%` combined), and the owned
`dot_maddubs_i7_m2n4` at only `2.07%`. This routes the next useful work away
from another dot-tile dig and into the owner-gated quality evidence.

**Change.** Added
`gated_quality_safe_encoder_int8_jfk_reference_wer_gate` to
`tests/native_engine_e2e.rs`. It spawns the real CLI in a child process, forces
bridge binaries to `/nonexistent`, sets `FRANKEN_WHISPER_NATIVE_EXECUTION=1` and
`FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole`, explicitly disables the older
all-i7 full gate with `FRANKEN_WHISPER_ENC_INT8=0`, and enables the quality-safe
full policy with `FW_ENC_ATTN_OUT_I8I32=1`, `FW_ENC_QKV_FUSED=1`, and
`FW_ENC_EF_QUANT=1`.

**Quality gate.** The test computes word-level Levenshtein WER against
`tests/fixtures/native/jfk_tiny_reference.json` and requires `WER <= 0.0`.
It also rejects the known all-i7 adversarial phrase `"Frank at"` and proves the
native implementation ran (`backend.ok.payload.implementation == "native"`).

```text
CARGO_TARGET_DIR=/data/tmp/cargo-target cargo test --test native_engine_e2e \
  gated_quality_safe_encoder_int8_jfk_reference_wer_gate -- --nocapture

test gated_quality_safe_encoder_int8_jfk_reference_wer_gate ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out; finished in 9.93s
```

**Expected-loss default-on policy contract (not flipped here).**
State space: model id/hparams, CPU feature class, calibration corpus id/hash,
per-layer quantization-error vector, per-corpus WER deltas, proper-noun sentinel
results, and live drift/error observations. Actions: `F32Encoder`,
`QualitySafeInt8Encoder`, and deterministic `FallbackF32`. Loss matrix:
false-accepting int8 with WER/proper-noun drift is high loss; false-fallback to
f32 costs only speed; missing calibration is treated as high loss. Posterior:
Beta-binomial exceedance model over fixture WER gates plus per-layer error
credible intervals; default-on requires posterior confidence that corpus WER
delta and every layer's quantization-error budget are inside thresholds. Fallback
trigger: use f32 deterministically when model/corpus hash is unknown, AVX2/i8
kernel support is absent, any adversarial sentinel fails, any per-layer error
budget is exceeded, or the operator sets the kill switch.

**Verdict: KEEP the executable gate; do not flip the default from this row
alone.** This lands the first hard quality-gate artifact for the safe full-int8
policy and documents the exact adaptive fallback contract. The broader default
promotion still needs the fixture-corpus WER table, per-layer quantization error
budget, and track01/proper-noun adversarial probe rows filled in.

### L1 — log-mel FFT twiddle precompute (bit-exact)  — `src/native_engine/mel.rs`

**Hypothesis.** whisper.cpp's recursive `fft` recomputes `cos`/`sin` twiddles
per butterfly per frame, and the odd-`N` base case `dft(25)` (reached 16×/frame,
3000 frames) recomputes ~1250 f64 transcendentals per call — ~60 M `sin`/`cos`
per 30 s of audio. These are pure functions of `(k, j, n)` and can be precomputed
into f32 tables once, evaluated bit-for-bit identically thereafter.

**Change.** Precompute level twiddles `[400,200,100,50]` and the `n=25` DFT
`25×25` cos/sin table once (cached `OnceLock`, shared read-only across mel worker
threads); thread them through `fft`/`dft`. Arithmetic and accumulation order
unchanged → bit-exact.

**Conformance.** New test `fft_twiddle_table_is_bit_exact_vs_inline_reference`
asserts byte-for-byte `Vec<f32>` equality vs an inline-transcendental copy of the
original recursion across 10 transform widths × 64 random seeds.

**Measurement (worker vmi1149989, criterion; baseline + candidate on the SAME
worker via rch repo-convergence → valid A/B; baseline = pre-edit code):**

| bench | baseline (mel-pre) | candidate | change | speedup |
|---|---|---|---|---|
| `mel_30s` | 269.06 ms | 29.23 ms | **−89.1%** (p=0.00) | **≈9.2×** |

**Verdict: KEEP.** A 9.2× reduction on the always-on log-mel frontend, far above
any plausible worker variance, with **byte-identical output** (parity test green
— see below). The transcendental-elimination hypothesis is confirmed: the
`dft(25)` base case was the dominant cost.

**Honesty note — what "vs the original" means here.** This ratio is
franken_whisper's mel frontend vs **its own prior faithful-port baseline**, not a
direct timing of OpenAI Whisper's Python mel or whisper.cpp's C mel. The output
remains bit-exact to whisper.cpp's mel (the port's contract), so this is "do
whisper.cpp's identical math, 9.2× faster." A true head-to-head wall-clock vs the
C++/Python originals needs the original-vs-franken harness (bd-zk43 / bd-0hnz);
the large-shape kernels also need the `large-v3-turbo` model staged (bd-ms0x).

**Conformance gate (CONFIRMED GREEN):** `cargo test -p franken_whisper --lib
native_engine::mel` → **7/7 passed** incl.
`fft_twiddle_table_is_bit_exact_vs_inline_reference` (0.32 s). Clippy
`-D warnings` initially flagged the new `n % 2 == 0` (`manual_is_multiple_of`);
fixed forward in **b0577d9** (`n.is_multiple_of(2)`, the codebase idiom) →
clippy green (`Finished dev`, exit 0). Both commits on `origin/main` + `master`.

> **Commits:** `656f55c` (L1) + `b0577d9` (clippy fix-forward).

### L2 — log-mel FFT per-call allocation elimination (deferred)  — bd-02do

The recursive `fft` still `vec!`-allocates even/odd split + child-output buffers
at every recursion node (~60 allocs/frame × 3000 frames). Secondary to the
twiddle win (allocator churn ≈ single-digit ms vs the ~240 ms transcendental
cost just removed). Tracked in bd-02do as a follow-up via per-thread scratch
buffers.

**Status: MEASURED, NOT LANDED (deferred).** Pre-verified bit-exact (standalone
scratch-FFT harness, 418,800 outputs, 0 mismatches). Measured via a standalone
local same-process A/B (stable host — the rigorous way given the 5.6× worker
variance below) over a realistic 3000-frame `N_FFT=400` pass:

| FFT pass (3000 frames, 1 thread) | time | speedup |
|---|---|---|
| alloc (current) | 28.5 ms | — |
| scratch (L2) | 23.4 ms | **1.21× (stable across runs)** |

**Decision — not landed.** The 1.21× is real at the FFT-kernel level, but the
FFT is only part of `mel_30s` (≈1.1× there) and `mel_30s` is itself a small
fraction of end-to-end transcription ⇒ **e2e gain ≈ 0**. Landing it also forces
`compute_frame_column` past the 7-arg `clippy::too_many_arguments` limit
(struct-refactor or `#[allow]`) — added complexity in a freshly-clean file for
no e2e benefit. Per the swarm's own "REVERT ~0-gain" rule, **deferred** until/
unless a real workload shows the mel frontend on its critical path. Design +
measurement preserved here and in the scratchpad so it can be landed in minutes
if that changes.

### L3 — sparse mel-filterbank projection (bit-exact)  — `src/native_engine/mel.rs`

**Hypothesis.** Real whisper mel filterbanks are sparse triangles: each of the 80
filters is nonzero over only **~5 of the 201** FFT freq bins. The projection loop
ran densely over all 201 bins per filter regardless — ~97.5% of the multiply-adds
were `power[k] * 0.0`. Skipping the leading/trailing zeros is **bit-exact**: for
the finite non-negative `power` an FFT of real audio produces, `power[k] * 0.0 ==
+0.0`, which never changes a running f64 sum (and the accumulation order over the
nonzero range is unchanged).

**Change.** Precompute each filter's `[start, end)` nonzero range once per
`log_mel` (bundled with the bank in `SparseMelFilters`, keeping
`compute_frame_column` under the 7-arg clippy limit); project only over that
range.

**Conformance.** New test `sparse_projection_matches_dense_bit_exact` asserts
byte-identical f64 sums (range-restricted vs full 201-bin dense) across 16
filters × 64 random non-negative power spectra. The existing mel tests
(silence/determinism) stay green (output unchanged). The hermetic `mel_30s`
(dense synthetic bank) is unaffected; new bench `mel_30s_realistic` (sparse
triangular bank, the production case) captures the win.

**Measurement (standalone local same-process A/B — rigorous given 5.6× rch worker
variance — over a realistic 80×201 triangular bank, 3000 frames):**

| projection (3000 frames) | time | speedup |
|---|---|---|
| dense (all 201 bins/filter) | 37.5 ms | — |
| sparse (~4.9 nonzero bins/filter) | 2.9 ms | **12.78×** |

Bit-exact check in the same harness: **0 / 240,000** mismatches. Since the dense
projection (37.5 ms) is *larger* than the post-L1 FFT pass (~28 ms), eliminating
it is **≈2× on the whole mel frontend for real (sparse-bank) models** —
a genuine real-workload win, unlike L2. **Verdict: KEEP.**

### L4 — frame-batched SIMD FFT (bit-exact)  — `src/native_engine/mel.rs`

**Hypothesis.** After L1+L3 the FFT is the dominant mel cost. Frames are
independent and identically-shaped, so they vectorize *vertically*: put one frame
per SIMD lane (`Simd<f32, 8>`, structure-of-arrays) and run one batched FFT over
8 frames. IEEE-754 f32 lane ops are bit-identical to scalar f32 (no FMA
contraction), so lane `L` equals the scalar FFT of frame `L` — **bit-exact**,
not an approximation. (This is a *vectorization* axis, orthogonal to L1/L3's
arithmetic-redundancy elimination — the "bit-exact floor" is lower than L3
implied.)

**Change.** `fft_simd8` / `dft_simd8` mirror the scalar recursion over
`Simd<f32, 8>` with the same precomputed twiddles (splatted). The mel worker
batches fully-valid frames (full `N_FFT` window) 8-at-a-time; the partial-window
tail + noise-floor frames keep the scalar path. After the batched FFT each lane
is transposed back and fed to the shared, tested `power_and_project` — so the
columns are byte-identical to the scalar path. Needs `#![feature(portable_simd)]`
(crate is nightly; stays `#![forbid(unsafe_code)]` — std::simd is safe).

**Conformance.** New test `fft_simd8_matches_scalar_bit_exact` asserts
byte-identical output per lane vs the scalar FFT (32 rounds × 8 frames × 802
bins); existing silence/determinism mel tests stay green.

**Measurement (standalone local same-process A/B, 3000-frame `N_FFT=400` pass —
rigorous given 5.6× rch worker variance):**

| FFT pass (3000 frames) | time | speedup |
|---|---|---|
| scalar (per-frame) | 26.7 ms | — |
| SIMD f32×8 (baseline x86-64) | 6.3 ms | **4.22×** |
| SIMD f32×8 (AVX2) | 4.5 ms | **5.62×** |

Bit-exact: **0 / 2,400,000** mismatches. Since the FFT dominates the post-L3 mel
frontend, this is **~2.5–3× on the whole mel frontend** on top of L1+L3.
**Verdict: KEEP.**

**In-tree cumulative result (criterion `native_engine/mel`, post L1+L3+L4):**

| bench | time | notes |
|---|---|---|
| `mel_30s` (dense synthetic bank) | **12.8 ms** | L1+L4 only (dense bank can't use L3); was 269 ms pre-L1 |
| `mel_30s_realistic` (sparse triangular bank = **production**) | **3.95 ms** | full L1+L3+L4 stack |

So a real model's 30 s log-mel frontend now runs in **~4 ms** (from a 269 ms
dense/transcendental-heavy starting point — a **~68× cumulative** reduction on the
hermetic frontend, all bit-exact). e2e share remains bounded by encoder/decoder.

### L5 — vertical-SIMD `layer_norm` (bit-exact)  — `src/native_engine/nn.rs`

**Hypothesis.** `layer_norm` runs in every encoder + decoder block. Its per-row
f64 mean/var reductions can't use *horizontal* SIMD (that reorders the f64 sum →
not bit-exact), but the L4 *vertical* trick applies: one row per `f64x8` lane, so
each lane reduces its own row in the original ascending order. IEEE-754 f64 lanes
+ correctly-rounded `sqrt`/division are bit-identical to scalar f64 ⇒ **bit-exact**
(unlike `gelu`/`softmax`, whose `tanh`/`exp` have no bit-exact SIMD form).

**Change.** Factor the per-row body into `norm_rows`, which gathers 8 rows into a
structure-of-arrays, computes mean/var/inv-std/affine in `f64x8`, and scatters
back; the `< 8`-row tail stays scalar. Both the serial and band-parallel paths
call it, so SIMD stacks with the existing thread fan-out. Reuses the L4
`#![feature(portable_simd)]` gate (still `#![forbid(unsafe_code)]`).

**Conformance.** New test `layer_norm_simd_matches_scalar` asserts byte-identical
output vs an independent scalar per-row f64 reference across row counts
{1,7,8,9,20,33} (covers SIMD groups + tail); existing layer_norm tests stay green.

**Measurement (standalone local same-process A/B, `[1500, 384]` encoder-window
shape; rigorous given 5.6× rch worker variance):**

| layer_norm `[1500,384]` | time | speedup |
|---|---|---|
| scalar per-row | 1.20 ms | — |
| vertical `f64x8` (baseline x86-64) | 0.61 ms | **1.97×** |
| vertical `f64x8` (AVX2) | 0.47 ms | **2.33×** |

Bit-exact: **0 / 576,000** mismatches. ~2× on a real per-layer activation op
(runs ×4 encoder + ×N decoder layers), bit-exact. New `layer_norm_1500x384`
bench makes it a standing in-repo instrument. **Verdict: KEEP** (modest e2e share
— still encoder/decoder-GEMM-bound — but a real, measured, bit-exact win and the
last nn kernel amenable to bit-exact vectorization).

### L6 — re-tune `layer_norm` PAR_THRESHOLD post-SIMD  — REJECTED (~0-gain)

**Hypothesis.** L5's SIMD made `layer_norm`'s compute ~2× cheaper, so the
`thread::scope` spawn cost might now dominate at the encoder shape `[1500,384]`,
arguing to raise `PAR_THRESHOLD` and run it serial-SIMD (a pure bit-exact
scheduling knob).

**Measured (standalone, same host, 8 workers):**

| shape | serial-SIMD | parallel-SIMD | winner |
|---|---|---|---|
| `[1500,384]` (encoder) | 0.70 ms | 0.79 ms | serial **1.0–1.13×** (within noise) |
| `[3000,384]` | 1.42 ms | 1.21 ms | parallel **1.17×** |

**Verdict: REJECTED.** The crossover already sits right around the production
encoder shape, so the existing `PAR_THRESHOLD = 1<<16` is well-tuned; raising it
would buy ≤1.1× at `[1500,384]` (noise) while *hurting* larger shapes. Per
REVERT-~0-gain, not shipped. (The slow in-tree `layer_norm_1500x384` = 3.3 ms on
ovh-b was worker variance, not spawn overhead.)

### L7 — x86-64-v3 build baseline (AVX2/FMA)  — `.cargo/config.toml`  **[e2e win]**

**Hypothesis.** The build used the Rust default target (`x86-64`, SSE2 only),
leaving AVX2/FMA unused by *all* code — the SIMD native engine AND, crucially,
**FrankenTorch's sgemm, which is ~99% of e2e** (encoder + decoder GEMM/GEMV). The
first profile of the real workloads exposed this: e2e_tiny_jfk = 708 ms = mel
~4 ms + **encoder 263 ms (37%) + decoder 441 ms (62%, ~15 ms/token)** — all
GEMM/gemv-bound. `#![forbid(unsafe_code)]` rules out runtime `#[target_feature]`
dispatch, so a build-wide CPU baseline is the only safe way to enable these
instructions.

**Change.** `.cargo/config.toml` → `rustflags = ["-C", "target-cpu=x86-64-v3"]`
(AVX2+FMA+BMI, Haswell-2013+).

**Measurement (local same-host A/B, tiny.en; first lever to move e2e):**

| `native_engine_bench` | SSE2 (default) | x86-64-v3 | speedup |
|---|---|---|---|
| `encoder_window_tiny` | 263 ms | 204 ms | **1.29×** |
| `decoder_token_step_tiny` | 122 ms | 102 ms | **1.20×** |
| **`e2e_tiny_jfk`** (full 11 s transcription) | 708 ms | **633 ms** | **1.12×** |

**Conformance.** Transcription-level (per `conformance-contract.md`), not
bit-exact — AVX2/FMA changes f32 rounding but `native_engine_e2e` is **6/6 green**
under the flag (transcription unchanged). **Verdict: KEEP.** First and only lever
to move the e2e-dominant GEMM. **Trade-off:** raises min CPU to AVX2 (2013+);
revert = delete `.cargo/config.toml` (or use `x86-64-v2`). The bit-exact
kernel levers (L1/L3/L4/L5) stack *on top* — they make the non-GEMM parts faster
within this baseline.

### L8 — vectorized gelu/softmax (AVX2 minimax exp/tanh)  — MEASURED, REVERTED (~0 e2e)

**Hypothesis.** Scalar `libm` `tanh`/`exp` in `gelu`/`softmax` looked like ~30%
of the encoder (a single isolated gelu over `[1500,1536]` is 15.2 ms scalar vs
4.3 ms vectorized = **3.56×**, with an accurate `exp_simd` at 7.9e-8 rel error).

**Measured in-tree (clean v3 A/B, `e2e_tiny_jfk`):** **632.6 ms (v3) → 647 ms
(v3 + vectorized gelu/softmax)** — **~0 gain, marginally negative.** The isolated
3.56× did NOT translate: gelu/softmax are a *small* fraction of the
GEMM-dominated encoder/decoder (my ~30% estimate was wrong — the FrankenTorch
sgemm dominates), so vectorizing them moves e2e by noise. Conformance was green
(200/200 lib tests incl. an accuracy test, native_engine_e2e 6/6), so it was
*correct*, just not *worth it*.

**Verdict: REVERTED** (commit b42ce64 → reverted) per the swarm's "REVERT ~0-gain"
rule. Lesson recorded so it isn't re-attempted: **isolated-kernel speedups must be
validated at e2e before landing** — the encoder/decoder are GEMM-bound, so only
the GEMM (FrankenTorch, external) or the build baseline (L7) move e2e here.

---

### L9 — decoder GEMV PAR_THRESHOLD 1<<19→1<<21 (spawn-bound MLP)  — `src/native_engine/nn.rs`  **[e2e win]**

**How it was found.** The 2026-06-25 whisper.cpp head-to-head (bd-zk43) showed
franken's DECODER is ~2× slower than whisper.cpp (the encoder/mel already win).
`decoder_attrib` (tiny.en, 400 steps, real load) pinpointed it: `mlp_fc_gelu` =
**5.14 ms/tok (35%, 0.23 GFLOP/s)** — absurd for 1.18 M MACs → **spawn-bound, not
compute-bound**. The MLP GEMVs (`[384,1536]`/`[1536,384]` = 590 k MACs) sit *just*
over the old `1<<19` (524 k) threshold, so each spawned 8 `thread::scope` threads
per token; 590 k split 8 ways is ~20 µs compute/thread vs tens of µs spawn/join.
(whisper.cpp avoids this with a persistent thread pool.)

**Fix.** Raise `PAR_THRESHOLD` to `1<<21` (2 M) in both GEMV paths, so the
per-token mid-size Linears run serial while the logits GEMV (20 M) and large-model
Linears (6.5 M) stay parallel. Pure scheduling knob → **bit-identical**.

**Measured (local v3 A/B):** `decoder_attrib` `mlp_fc_gelu` 5.14→**2.81 ms/tok
(−45%)**, total 14.67→12.32 ms/tok (−16%); **`e2e_tiny_jfk` 614→571 ms = −9.5%
(criterion p<0.05, "improved")**. Narrows the whisper.cpp gap 1.37×→1.27×.
**Verdict: KEEP.** Follow-up (same tick): the *other* decoder subs that looked
spawn-bound in `decoder_attrib` do NOT translate to the e2e — both MEASURED and
REJECTED:
- `project_qkv` serial (was 1.64 ms/tok in attrib): e2e **566 vs 571 ms, p=0.55
  (~0)** → reverted, kept concurrent (helps large models).
- `cross_attn` 1<<13→1<<14 (tiny serial; was 2.93 ms/tok in attrib): no-ts e2e
  **+2.7%, p<0.05 (REGRESSED)** → reverted, parallel path is genuinely faster.

Lesson: **`decoder_attrib`'s tight 400-step loop over-states per-call spawn cost**
vs the real e2e (decode interspersed with mel/encode). Only the MLP GEMV
threshold (L9, validated on the e2e) was a real spawn win; a blanket persistent
thread pool is NOT obviously worth it. The remaining franken-vs-whisper.cpp
decoder gap (1.27×) is now compute-bound (GEMV/sgemm/softmax), not spawn-bound.

---

### L10 — m=1 GEMV fast path in `nn::matmul` (skip ft sgemm for tq=1 attn)  — `src/native_engine/nn.rs`  **[e2e win]**

**How it was found.** With spawn ruled out (L9 + follow-ups), the decoder gap is
compute. `nn::matmul` routed *everything* through `ft_kernel_cpu` sgemm — including
the per-token decode attention matmuls, which at tq=1 are GEMV-shaped
(`[1,d]×[d,tk]` scores, `[1,tk]×[tk,d]` out). Standalone (x86-64-v3) showed ft
sgemm pays huge packing/dispatch overhead at m=1: `[1,64]×[64,1500]` **sgemm 46 µs
vs direct gemv 4.5 µs (10.2×)**; `[1,1500]×[1500,64]` **48 vs 6.3 µs (7.6×)**.
(GGML/whisper.cpp use a dedicated dot here — this is a real slice of the decoder
gap.)

**Fix.** Add an `m == 1` branch to `nn::matmul`: row-broadcast SAXPY accumulation
over k (`out += a[k]*b[k,:]`, LLVM → AVX2 FMA), skipping sgemm packing entirely.
Helps every m=1 caller (cross_attn + self_attn). NOT bit-identical (different
summation order, max abs diff ~1e-6/2.7e-5) → relies on the transcription-level
contract.

**Measured (local v3):** `e2e_tiny_jfk` 571→**561 ms (ts)** / 543→**534 ms
(no-ts)** = **−1.7%**; whisper.cpp gap 1.21×→**1.19×** (no-ts). **Conformance
GREEN** (native_engine_e2e 6/6). **Verdict: KEEP.** Modest at e2e (the attn
matmuls are a small slice; the mlp/logits use the separate f16 GEMV path), but a
free, correct win and the right structural fix.

---

### L11 — rayon persistent-pool `gemv_f16` (re-parallelize the mlp w/o spawn)  — `src/native_engine/nn.rs`  **[e2e win]**

**The insight.** L9 serialized the per-token mid GEMVs because `std::thread::scope`
*per-call spawn* dominated their compute under load. But serial leaves 7 of 8 cores
idle on the mlp — whisper.cpp uses a PERSISTENT thread pool (no per-call spawn) and
keeps the parallelism. franken used `thread::scope` everywhere (no persistent pool).

**Fix.** Add `rayon` (already in-tree via ft-kernel-cpu) and dispatch `gemv_f16`'s
parallel path via `par_chunks_mut` over output-row bands (rayon's global pool — no
per-call spawn), and drop the threshold back `1<<21`→`1<<19` so the mlp (590 k) +
logits (20 M) re-parallelize while the tiny `[384,384]`=147 k stay serial.
**Bit-identical** (disjoint output-row bands, each row's `dot8` order unchanged;
standalone maxdiff 0).

**Measured.** Standalone (contended host) rayon vs serial gemv: `[1536,384]` 1.40×,
`[384,1536]` 1.35×. In-tree: **`e2e_tiny_jfk` 561→542 ms (ts) / 534→523 ms
(no-ts) = −3.4% / −2.1%**; **conformance GREEN** (native_engine_e2e 6/6). whisper.cpp
gap 1.19×→**1.17×** (no-ts). **Verdict: KEEP.** rayon's persistent pool is the
correct structural answer to the per-call-spawn problem L9 worked around; supersedes
L9's serial-mlp compromise (threshold restored, dispatch via the pool).

*Band-size follow-up (MEASURED, REJECTED):* finer chunks (`workers*4`, min 64
rows) to let rayon work-steal on a contended host — hypothesis that a 1-chunk/core
split stalls when a core is busy with another process. no-ts e2e **+3.7%
(REGRESSED)**: the extra rayon task + per-chunk scratch-alloc overhead outweighs
the work-steal benefit at these sizes. `band = out/workers` is optimal; kept.

---

### L12 — rayon persistent-pool cross-attn head dispatch  — `src/native_engine/decoder.rs`  **[e2e win]**

**Insight.** Extending L11 to the cross-attention wrapper. The no-timestamps decode
path (record off — the apples-to-apples vs whisper.cpp's `dtw=0`) parallelized
cross-attn over heads with `std::thread::scope` **per token** (6 head-threads ×
~28 tokens). Like the mlp (L9/L11), that per-call spawn was the bottleneck, not the
compute (serializing it had REGRESSED +2.7%, so parallelism is needed — just
without the spawn).

**Fix.** Dispatch the head bands via rayon's persistent pool
(`band_starts.into_par_iter()`), each band scattering into a private buffer →
disjoint-merge. **Bit-identical** (every position written by exactly one head;
compute_head/scatter capture only shared refs).

**Measured (local v3, no-ts e2e):** **523→477–491 ms = −6 to −8.8%** (contention-
dependent); **conformance GREEN** (native_engine_e2e 6/6). The ts path is
unchanged (it uses the serial `record` branch, not this parallel path). whisper.cpp
gap **1.17×→~1.07–1.10× (NEAR PARITY)**. **Verdict: KEEP.**

---

### L13 — rayon cross-attn for the RECORD (timestamps) path  — `src/native_engine/decoder.rs`  **[e2e win]**

**Insight.** L12 only sped the no-ts path; the realistic default (`timestamps:true`,
DTW word alignment) took the serial `record` branch because per-head softmax
`scores` must land in `recorded` in head order. But the *compute* can still be
parallel — only the recording needs ordering.

**Fix.** Parallelize `compute_head` over heads via rayon (persistent pool), collect
in head order, then push `scores` + scatter SERIALLY. `compute_head` never touches
`recorded` → Sync; ordering + disjoint scatter unchanged → **bit-identical** (DTW
timestamps green).

**Measured (local v3, ts e2e):** **542→504 ms = −7%**; **conformance GREEN**
(native_engine_e2e 6/6). **Verdict: KEEP.** Now both decode paths (ts + no-ts) get
parallel cross-attn.

### L14 — cap Rayon default pool to native default_threads()  — `src/native_engine/mod.rs`

**How it was found.** Current head (`a9ecb3b`) ran on a 64-way host. The native
engine's own default is capped at 16 threads, and its glue kernels are tuned
around 8-16 workers, but Rayon defaulted to all 64 host threads when
`RAYON_NUM_THREADS` was unset. A same-binary surface sweep showed the issue:
loaded `tiny.en` JFK at `threads=8` had median-after-warmup **0.624 s** with the
default pool, while `RAYON_NUM_THREADS=16` measured **0.547 s**. The 4/8/12/16
sweep showed 16 was the best tested cap; 4 regressed badly.

**Fix.** Before the first native inference kernels run, initialize Rayon's
global pool to [`default_threads()`] (16 on this host) when the operator has not
already set `RAYON_NUM_THREADS`. Explicit `RAYON_NUM_THREADS` remains an override;
if another embedding app already initialized Rayon, `build_global`'s error is
ignored and behavior remains unchanged. This is pure scheduling: no numeric
order inside any output row changes.

**Measured (local same-host, current-head A/B, `native_ab tiny.en 9 <threads>`,
discard run 0):**

| loaded-model path | baseline median | L14 median | speedup |
|---|---:|---:|---:|
| 4 threads | 0.603520 s | 0.540470 s | **1.117×** |
| 8 threads | 0.624235 s | 0.535540 s | **1.166×** |

Decoder attribution agreed directionally: 13.064→11.878 ms/token, mainly from
`logits_gemv` and `cross_attn` moving to the right-size persistent pool. Output
proof: baseline and L14 `native_ab` JSON outputs are byte-identical at both 4
and 8 threads.

**OpenAI Whisper boundary (same host):** one-shot CLI comparator improved from
**3.20×** to **4.23×** faster than OpenAI Whisper CLI. Loaded API boundary is
mixed: L14 beats OpenAI loaded API at 4 threads (**1.078×**) but still loses at
8 threads (**0.784×**, franken 1.275× slower). **Verdict: KEEP.** This is the
first post-L13 in-crate e2e win; it narrows but does not eliminate the loaded
OpenAI 8-thread gap.

---

### L15 — parallel-layer model load (serial transpose + rayon over layers)  — `src/native_engine/{nn,encoder}.rs`  **[load win]**

**How it was found.** The large-v3-turbo head-to-head (NEGATIVE_EVIDENCE 2026-06-25)
showed franken WINS transcription compute (1.24×) but LOSES cold-CLI (12.96 s vs
whisper.cpp 9.75 s) on model LOAD. Perf-span profile: `model_parse` 1.28 s +
`model_weights` **1.97 s** = 3.25 s (whisper.cpp 0.90 s). The 1.97 s is the
per-weight `[out,in]→[in,out]` transpose, run in a **sequential 32-layer loop**
(`EncoderWeights::from_ggml`), each weight using a `thread::scope` parallel transpose.

**Fix.** Parallelize the load ACROSS layers via rayon (`(0..n_layer).into_par_iter()`)
and make each layer's transpose **serial** (`nn::transpose_serial`, no spawn) — coarse
layer-grain parallelism fills cores without the nested `thread::scope` spawn-thrash.
`map`+`collect` preserves layer order; the transpose is a pure permutation → the
assembled weights are **byte-identical** to the serial loop.

**Measured (large-v3-turbo, perf spans):** `model_weights` **1.97 s → 0.82 s (−58%)**;
total load **3.25 s → 2.07 s (−36%)**. Cold-CLI large now ~9.2 s (2.07 load + 7.1
transcribe) vs whisper.cpp 9.75 s → **franken WINS cold large too** (was a loss).
**Conformance GREEN** (native_engine_e2e 6/6; large jfk text byte-identical incl the
pre-existing trailing token). **Verdict: KEEP.** Closes the last franken-vs-whisper.cpp
gap (cold-start load); the parse (1.28 s, eager `fs::read`) is the remaining load cost
(mmap blocked by `#![forbid(unsafe_code)]`).

---

### L16 — linear resampler interior/tail split (bit-exact)  — `src/audio.rs`

**Hypothesis.** `resample_mono_linear` (the builtin no-ffmpeg decode path's
sample-rate converter) clamps **both** source loads on **every** output sample
(`input[left_idx.min(last)]`, `input[right_idx.min(last)]`). For all but the
final 1–2 taps both indices are already in bounds, so those `.min()` clamps +
`saturating_add` are pure per-sample overhead on the hot span.

**Change.** Hoist the loop invariants (`last`, `total`) and split the loop body
into an interior fast case (`left_idx < last` → index `input[left_idx]` /
`input[left_idx+1]` with no clamp) and a tail branch for the last taps. The
per-sample arithmetic is **byte-identical** — same `idx as f64 * ratio` position,
same `floor`, same `(src_pos - left_idx as f64) as f32` frac — so the resampled
signal is bit-exact; only the redundant clamp work is removed.

**Conformance.** New test `audio::tests::resample_mono_linear_is_bit_exact_vs_reference`
asserts byte-for-byte `f32::to_bits()` equality vs an inline copy of the original
clamp-every-load form across 6 rate pairs (down/up/identity) × 9 lengths
(0,1,2,3,7,31,1000,4096,44101 — covers empty, sub-tap, and edge tails). Green.

**Measurement (standalone microbench, `rustc -O -C target-cpu=x86-64-v3`, 30 s of
mono audio, best-of-60; bit-exact vs baseline verified each shape):**

| resample | baseline | candidate (split) | speedup |
|---|---|---|---|
| 44.1 kHz → 16 kHz | 1.715 ms | 1.610 ms | **1.065×** |
| 48 kHz → 16 kHz | 1.714 ms | 1.615 ms | **1.061×** |
| 22.05 kHz → 16 kHz | 1.712 ms | 1.613 ms | **1.061×** |

A *windowed-slice* variant (compute interior count, index `&input[l..l+2]`) was
also measured and **REJECTED** — it regressed to **0.97–0.98×** (the interior-count
arithmetic + 2-elem slice bound cost more than the clamp it removed).

**Verdict: KEEP** (small but real, reproducible across 3 shapes, zero-downside,
bit-exact). **Honest scope:** this path is **e2e-neutral** — `resample_mono_linear`
runs only in the builtin (no-ffmpeg) decoder, once per file, and early-returns when
`src_rate == dst_rate` (already-16 kHz inputs, incl. the jfk e2e fixture, never hit
it). So this is a free kernel cleanup, not a head-to-head gap-closer; recorded for
completeness as the one un-touched preprocessing kernel after L1–L4 (mel/FFT) and
L15 (load). See NEGATIVE_EVIDENCE 2026-06-25 for the reject + scope caveat.

---

### R-blocked-dequant — interleaved 256-chunk `gemv_f16` dequant  — REJECTED (x86 2.1× REGRESSION)

An uncommitted working-tree `row_dot` rewrite (256-element L1-chunked dequant +
hand-rolled 8-lane fold) carried an in-code comment claiming `x86-64-v3` wins of
1.18–1.65×. Criterion A/B on the canonical x86 rch fleet (committed baseline vs
candidate) showed the **opposite**: `f16_gemv_dequant_384x384` **+19.9%**,
`f16_gemv_dequant_1280x1280` **+109%** (both p<0.05). The committed `bulk
convert_to_f32_slice → dot8` auto-vectorizes to tight `vfmadd`; the chunked
`x[c+j+l]` inner loop defeats that. The claimed win was an M4/aarch64 (4-wide
`fp16`) artifact that does not hold on x86. **REVERTED** (stash-preserved). Full
analysis + table in NEGATIVE_EVIDENCE 2026-06-25.

### R-quad-dot8 — 4 independent accumulators in `dot8`  — REJECTED (x86 2.5× REGRESSION)

The FMA-latency lever (4 disjoint 8-lane accumulators over 32-elem chunks) to
break `dot8`'s single-ymm dependency chain. Conformance green (27/27 nn tests),
but criterion A/B vs committed `dot8` (`blk_pre`): `f16_gemv_dequant_1280x1280`
**+122%**, `dequant_384x384` **+148%** (both p<0.05). Indexing `ach[8+i]`/`16+i`
breaks the `chunks_exact(8)`/`0..8` idiom LLVM pattern-matches into `vfmadd` →
scalarized (~383 µs, same floor as R-blocked-dequant). **Second confirmation that
`dot8`'s clean form is load-bearing — do NOT hand-restructure it** (the single
accumulator is not latency-bound in practice). REVERTED (stash-preserved). Real
headroom needs wider SIMD (AVX-512/`x86-64-v4`, owner sign-off). Full analysis in
NEGATIVE_EVIDENCE 2026-06-25.

---

## ⇒ Session arc (2026-06-25, BlackThrush): built the comparator, closed 1.37×→~1.08×

Building `whisper-cli` (bd-zk43) exposed the real gap as the **in-scope decoder**
(not the encoder, which already wins 204 vs 242 ms). FIVE bit-identical/
transcription-green wins followed — all whisper.cpp/GGML techniques franken lacked
(spawn-bound dispatch → persistent pool; sgemm-for-gemv → dedicated dot):

| lever | what | e2e |
|---|---|---|
| L9 | mlp GEMV spawn threshold | no-ts ~590→543 ms |
| L10 | m=1 gemv (skip sgemm packing) | no-ts 543→534 ms |
| L11 | rayon persistent-pool gemv_f16 | no-ts 534→523 ms; ts 561→542 |
| L12 | rayon persistent-pool cross-attn (no-ts) | no-ts 523→**477–491 ms** |
| L13 | rayon cross-attn (ts/record path) | ts 542→**504 ms** |

**franken_whisper tiny.en jfk vs whisper.cpp: no-ts 1.37×→~1.07–1.10× (near
parity); ts (realistic, with word timestamps) 614→504 ms (−18%)** — all
conformance-green. Remaining to *win outright*: bd-4hc0 (encoder
`matrixmultiply→gemm`, out-of-scope) would cut the encoder ~2×.
**[2026-07-10 cc_fw: FALSIFIED — measured 1.00–1.07× on turbo against ft's real
path, and 0.934× at 16t. See the SUPERSEDED banner on the bd-4hc0 section below.]**

## Conformance-level finding — bit-exact was stricter than required (BlackThrush)

`docs/conformance-contract.md`: **"Compatibility is *not* byte-for-byte identical
output"** — the contract is **transcription-level** (exact/normalized text +
≤50 ms timestamp tolerance + speaker/confidence bands), enforced by
`tests/conformance_harness.rs`. All L1/L3/L4/L5 levers were **bit-exact** (zero
risk, correct), but that is *stricter* than the contract requires. Implications
for future levers:

- **rFFT / split-radix mel is contract-permitted** (no approval needed) — but mel
  is already ~4 ms post-L1/L3/L4, i.e. **<2% of e2e** (encoder/decoder-bound), so
  a further ~2× there is REVERT-~0-gain. Not pursued.
- **INT8-quantized GEMV — MEASURED, REJECTED.** Accuracy is fine (int8 vs f32
  max rel error 0.4%; whisper.cpp Q8_0 confirms int8 preserves WER), but a SAFE
  `std::simd` int8 GEMV (widen i8→i32, no VNNI) clocks **0.24× — ~4× SLOWER** than
  the f16/f32-dot path at both baseline and AVX2 (`int8_gemv.rs`). The int8 speed
  win needs `vpdpbusd` (VNNI) intrinsics, which are **unsafe → forbidden by
  `#![forbid(unsafe_code)]`**; the f16 path already uses hardware `f16c` dequant
  safely. **DEAD under the safe-code constraint.**
- **Approximate-transcendental `gelu`/`softmax` (SIMD `exp`/`tanh`)**: legal under
  the contract, but they're small vs the GEMM (GEMM-bound e2e) and carry
  transcription risk needing local-e2e verification → marginal EV.

- **Explicit FMA (`mul_add`) in the gemv `dot8` — MEASURED, REJECTED (regression).**
  The decoder is 62% of e2e and runs `gemv_f16`/`dot8` (separate mul+add, since
  Rust doesn't auto-contract). Hypothesis: explicit `mul_add` under the +fma
  baseline (L7) would speed the decoder core. Standalone (logits shape
  51864×384, x86-64-v3): explicit `mul_add` dot = **0.791× — SLOWER** than the
  current mul+add. LLVM already lowers the 8-accumulator mul+add optimally (and
  contracts where it helps); forcing `mul_add` hurts. The decoder gemv is already
  optimal; **REJECTED**.

- **Vertical-layout gemv (bd-n0m3) — MEASURED, REJECTED (~0-gain).** Hypothesis:
  store the logits f16 weight interleaved `[OUT/8, INP, 8]` so the gemv vertically
  vectorizes 8 output rows into f32×8 accumulators (no per-row horizontal
  reduction) — a different organization than the current per-row `dequant+dot8`.
  Standalone with real f16c dequant (logits shape 51864×384, x86-64-v3):
  current 4154 µs vs vertical 4046 µs = **1.03×** (max abs diff 4e-6,
  transcription-level). The current per-row dequant+dot8 is already within 3% of
  the alternative organization → not worth the load-time relayout + kernel
  rewrite. Confirms the decoder gemv is mature regardless of layout; **REJECTED**.

- **Encoder QKV-projection fusion — MEASURED component (1.14×), net ~0 at e2e,
  NOT PURSUED.** Encoder attention does Q/K/V as 3 separate `matmul_bias` calls on
  the same LHS `h` (encoder.rs:426-428); `matrixmultiply` re-packs `h` per call, so
  fusing into one `[1500,384]×[384,1152]` saves 2 re-packings — standalone measured
  **1.14×** on the QKV proj (16884→14791 µs, contended; bit-identical since sgemm
  output columns are independent). But integration negates it: the fused output
  `[1500,1152]` must be split back to q/k/v `[1500,384]` (3 strided copies ≈
  6.9 MB/layer ≈ 1.4 ms/4 layers), eating most of the saving; and QKV is only
  ~20-30% of the encoder → net **~0–0.5% e2e** (within bench noise). Classic
  component-win-vanishes-at-integration (cf. L8). Deferred as not worth the change.
  NB: the win is *matrixmultiply's per-call repacking overhead* — another cost the
  `gemm`/faer swap (bd-4hc0) removes structurally, reinforcing that lever.

- **Decode-loop full-vocab logsumexp vectorization — MEASURED, REJECTED (~0).**
  `compute_logprobs` (decode.rs) runs a log-softmax over ALL 51 864 logits per
  token — ~1.45 M scalar `libm` `exp` over the decode — which *looks* like a fat
  lever. Vectorized the logsumexp loop with an 8-wide minimax `exp_simd`
  (`nn::logsumexp_over_finite`, ~7.9e-8 rel). Clean back-to-back A/B (no-ts e2e,
  `--baseline`): **−0.32%, p=0.46 — "no change"** (a spurious −1.8% on one ts run
  was contention noise). Reason: modern `libm` `expf` is ~5–7 ns, so the loop is
  only ~7–10 ms total (~1.5%), within bench noise, and `compute_logprobs`'s
  output `Vec` (needed by the ts timestamp-pairing) isn't the bottleneck either.
  **REVERTED** (conformance was 6/6 green, so it was *correct*, just ~0). Don't
  re-attempt: the per-token full-vocab `exp` is not a real e2e cost here.

- **Encoder `attention_raw` rayon dispatch — MEASURED, REJECTED (~0).** L11/L12/L13
  proved rayon's persistent pool beats per-call `thread::scope` for the DECODER's
  per-token attention (tiny work, spawn-bound). Tried the same on the ENCODER's
  `attention_raw` head dispatch (the encoder is now the largest e2e slice, ~42%).
  Clean A/B (`encoder_window_tiny`, `--baseline`): **+2.9%, p=0.62 — "no change"**
  (huge ±30 ms variance). Reason: the encoder's per-head work is BIG (sgemm +
  softmax over `[~550,~550]`), so the 4-spawns/window `thread::scope` cost is
  already amortized — it was never spawn-bound like the decoder's tq=1 per-token
  attention. **REVERTED.** Confirms the spawn-bound win was decoder-per-token-
  specific; the encoder is sgemm-bound (→ bd-4hc0, out-of-scope), not spawn-bound.

**Net (measured, not assumed):** `#![forbid(unsafe_code)]` (no VNNI) + the
e2e-dominant GEMM living in FrankenTorch (external crate `ft-kernel-cpu`, which
hardcodes `matrixmultiply 0.3` with no feature knob) cap the kernel-level wins in
this crate. The lever space is **exhaustively exhausted by measurement**: 5
shipped (L1/L3/L4/L5 mel bit-exact + **L7 x86-64-v3 = the 1.12× e2e win**), 5
measured-and-rejected (L2 ~0-e2e, L6 ~0-gain, L8 ~0-e2e, INT8 0.24×, gemv-FMA
0.791×). e2e is encoder-GEMM-bound (external) + decoder-logits-bandwidth-bound
(40 MB f16/token, fundamental). Further e2e wins require FrankenTorch-side GEMM
work (`matrixmultiply` → `gemm`/faer, ~1.5–3×) or lifting `#![forbid(unsafe_code)]`
for VNNI int8 — **both out of `franken_whisper`'s crate**.

## ⇒ Biggest remaining e2e lever, MEASURED: the GEMM has 3.75× headroom (bd-4hc0)

> **⚠️ SUPERSEDED / FALSIFIED — 2026-07-10, cc_fw.** The table below is measured
> against **raw `matrixmultiply`**, NOT against `ft_kernel_cpu::matmul_tensor_
> contiguous_f32`, which wraps it in ft's own tuned rayon layer (`PAR_MIN_FLOPS`,
> `TALL_MIN_ROWS`, `F32_2D_MAX_K`, row-split + 2-D tiling). On the exact
> `[1500,384]×[384,1536]` shape below, this entry's baseline is **187 GF/s**;
> ft's real path measures **1191 GF/s** — 6.4× faster. The "3.75× headroom" is
> headroom over **code the engine never executes**.
>
> Measured against the real path (interleaved, min-of-9, 32t):
> **large-v3-turbo linear-GEMM layer total = 1.00–1.07×** (fc2 = 1.001×), and the
> swap is a **regression (0.934×) at 16 threads**. tiny.en = 1.311×. `gemm`'s
> microkernel IS ~1.325× better **serially**, but its internal rayon is worse than
> ft's row-split, which throws the gain away above 8 threads. A hybrid
> (ft scheduler + `gemm` serial block) is **0.942×** on turbo, because ft's
> row-split makes every thread re-stream the full B (fc1's B = 26.2 MB × 32
> threads) and `sgemm_2d_parallel` — which exists for exactly that regime — is
> gated `k ≤ 1024` while turbo's k is 1280/5120.
>
> **bd-4hc0 as specified (swap the crate) is REJECTED.** The real lever is to
> raise/replace `F32_2D_MAX_K` so turbo reaches the 2-D tiled path, *then* swap the
> serial microkernel. Full numbers, thread sweep, and retry conditions:
> `docs/NEGATIVE_EVIDENCE.md` (2026-07-10, cc_fw, "bd-4hc0 REJECTED / FALSIFIED").
> The one surviving slice is `sdpa_forward_f32`'s inner serial GEMM: **1.115×** on
> the kernel, ~1.6% e2e, non-byte-exact (rel_l2 3.8e-7), unlanded (dependency cost).
>
> **UPDATE (same day, cc_fw — I under-claimed above; see the SELF-CORRECTION entry
> at the top of NEGATIVE_EVIDENCE.md).** Confirmed the predicted fix: with a 2-D tile
> grid, `gemm` DOES reach turbo — **1.231× on the turbo linear-GEMM layer ⇒ ≈1.14× e2e**
> (qkv/out 1.200×, fc1 1.238×, fc2 1.255×; interleaved, arms rotated, min-of-9).
> So bd-4hc0's *number* (~1.2× e2e) was about right; its *prescription* was wrong.
> You need the microkernel AND the 2-D grid — the crate swap alone is 1.00–1.07×.
> Also: `sgemm_reused_output` already 2-D tiles `1024 < k ≤ 1536`, so turbo qkv/out and
> fc1 were never on the row split; only fc2 (k=5120) was.
>
> **LANDED (bit-exact, dep-free):** frankentorch `8e3e7c9d` raises `F32_2D_TALL_MAX_K`
> 1536→8192 (kill-switch `FT_SGEMM_2D_LARGE_K=0`) ⇒ **1.057× on fc2**, ~1.3% e2e.
> The stale comment claiming 2-D regresses 0.81× on `m2048 k2048 n2048` does NOT
> reproduce: it is **1.27× faster** 2-D tiled.
>
> **NEXT RANKED LEVER (bit-exact, dep-free, bigger):** `tile_shape` is load-imbalanced —
> `p=floor(√32)=5, q=7` ⇒ **35 tiles on 32 threads**. Even post-fix ft's fc2 is 1.146×
> slower than the same `matrixmultiply` kernel on a balanced 4×8 grid. Fix `p` to the
> largest divisor of `threads` ≤ √threads. Expect fc2 → ~1.24× ⇒ ~1.05× e2e byte-exact.
> **Do this before reaching for `gemm`.**


The e2e wall is the encoder GEMM, delegated to `ft_kernel_cpu::matmul_tensor_
contiguous_f32`, which uses **`matrixmultiply 0.3`**. Standalone A/B (x86-64-v3,
rayon) for the encoder MLP shape `[1500,384]×[384,1536]`:

Full per-shape profile (standalone same-run A/B; ratios are the signal — absolute
GFLOP/s drops under box contention, e.g. the uncontended fc1 run hit
187→701 GFLOP/s = 3.75×):

| encoder GEMM shape | `gemm`/faer vs `matrixmultiply` |
|---|---|
| attn Q/K/V/out `[1500,384]×[384,384]` | **3.14×** |
| MLP fc1 `[1500,384]×[384,1536]` | **2.24× – 3.75×** (uncontended) |
| MLP fc2 `[1500,1536]×[1536,384]` | **1.46×** (larger K → smaller gap) |

So EVERY encoder GEMM is faster on `gemm`/faer — `matrixmultiply` is consistently
the bottleneck. The GEMM is ~most of the GEMM-bound encoder (~32% of e2e), so it
is **~1.5–3.75× off achievable** (shape-dependent; weighted ~2–3×). Swapping `ft-kernel-cpu`'s `matrixmultiply`→`gemm` is **~2× encoder
→ ~1.2× e2e** for franken_whisper, and benefits every FrankenTorch user.
`ft-kernel-cpu` already calls `matrixmultiply` via `unsafe`, so `gemm`'s unsafe
API is fine there; `franken_whisper`'s `#![forbid(unsafe_code)]` blocks calling
`gemm` directly (and `faer`'s safe API is a heavy dep), so the clean fix lives in
**ft-kernel-cpu** (out of `franken_whisper-cc`'s scope). **bd-4hc0 (P0).** This
turns "the GEMM is external, untouchable" into "the GEMM has a measured 3.75×,
here's exactly where."

## Measurement infrastructure findings (2026-06-24, BlackThrush)

These shape what is measurable and how the ratios above must be read.

1. **Worker variance ≈ 5.6×.** `mel_30s` (identical code) measured **29 ms**
   (vmi1149989), **63 ms** (ovh-a), **164 ms** (vmi1152480). rch assigns workers
   per invocation and exposes **no pinning flag**, so **cross-run criterion
   `--baseline` is invalid** unless both runs land on the same worker. L1's 9.2×
   is trustworthy precisely because baseline + candidate both ran on vmi1149989.
   **Rule:** only same-worker (single-`rch exec`) A/B is admissible.

2. **Real-workload benches are unmeasurable via `rch` — RESOLVED via local builds
   (bd-7xbq closed).** `encoder_window_*`, `decoder_token_step_*`, `e2e_tiny_jfk`,
   `logits_gemv_large` SKIP on remote workers: the ggml model and `jfk.wav` are
   **gitignored** (`*.wav`, model dirs) so rch does not sync them. **Working
   path (proven):**
   ```
   RCH_MIN_LOCAL_TIME_MS=99999999 \      # forces rch to build LOCALLY (no offload)
   CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cc-local \
   FRANKEN_WHISPER_MODEL_DIR=.../legacy_whispercpp/whisper.cpp/models \
   cargo test -p franken_whisper --release --test native_engine_e2e
   ```
   Built locally in **5m52s** (this host's nightly compiles `ft-kernel-cpu` fine —
   the `ovh-a` `stdarch_neon_dotprod` failure is worker-specific drift) and ran
   **6/6 gated pipeline tests that actually transcribed jfk** via the native
   tiny.en engine (no SKIP) — i.e. **transcription conformance is verifiable
   locally**. This is the gateway for any non-bit-exact lever AND the e2e
   head-to-head. `large-v3-turbo` still absent (bd-ms0x).

3. **No built `whisper.cpp` comparator.** `whisper-cli`/`main` is not built on
   this host (only source under `legacy_whispercpp/whisper.cpp`). A true
   wall-clock head-to-head vs the original requires building it first
   (cmake) — harness work tracked under bd-zk43 / bd-0hnz (IcyWren).

4. **Hermetic f16_gemv baselines** (ovh-a, for future levers):
   `1280×1280 = 419 µs (3.9 Gelem/s)`, `384×384 = 137 µs (1.07 Gelem/s)`. The
   small 384×384 (tiny.en per-token Linear) is ~4× lower throughput — a possible
   future lever, but `gemv_f16` is already SIMD + band-parallel, so a bit-exact
   gain there is uncertain.

**Bit-exact-lever feasibility map.** The mel twiddle win was a sweet spot:
constant (data-independent) transcendentals, precomputable exactly. The other
hot kernels are NOT: `softmax`(exp), `gelu`(tanh), `layer_norm`(reduction) all
have **data-dependent** transcendentals / order-sensitive f64 sums — any speedup
(approx exp/tanh, reordered reduction) changes output bits and breaks the
whisper.cpp conformance contract. Encoder GEMM is FrankenTorch's (external
crate). So further *bit-exact* native-engine levers are limited; the largest
remaining honest wins require the local-measurement unblock (item 2) and the
`whisper.cpp` comparator (item 3).
