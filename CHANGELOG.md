# Changelog

All notable changes to [franken_whisper](https://github.com/Dicklesworthstone/franken_whisper) are documented in this file.

franken_whisper is an agent-first Rust ASR orchestration stack with adaptive Bayesian backend routing, real-time NDJSON streaming, speculative cancel-correct transcription, and SQLite-backed persistence. It wraps `whisper.cpp`, `insanely-fast-whisper`, and `whisper-diarization` behind a unified 10-stage composable pipeline, then adds in-process Rust native engines under a staged conformance-gated rollout.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Commit links point to the canonical GitHub repository.

---

## [Unreleased] (since v0.1.0)

152 commits since v0.1.0

Compare: [`v0.1.0...main`](https://github.com/Dicklesworthstone/franken_whisper/compare/v0.1.0...main)

### YouTube Ingestion (`franken_whisper youtube`) (post-2026-06-06)

- **New `youtube` subcommand: download YouTube audio and transcribe it into deep-linked markdown + JSON** (bd-s63n; epic bd-27v1) — a single command takes video URLs, playlist URLs (auto-expanded), and/or a `--batch-file`, downloads each via `yt-dlp`, transcribes it through the same engine as `transcribe`, and writes a `{upload_date} - {title} [{id}]` markdown/JSON pair (plus kept audio and a `.fw_youtube_manifest.json` state file) into `youtube_transcripts/`. The run is **idempotent and resumable** — re-runs skip videos already `done`, retry failed ones unless `--no-retry`, and Ctrl+C cancels cleanly (kills yt-dlp, aborts transcription, leaves the manifest honest). Pairs naturally with the in-process native engine: no cloud, no API keys. Verified end-to-end against real YouTube (`watch?v=jNQXAC9IVRw --model tiny.en` → `2005-04-24 - Me at the zoo [jNQXAC9IVRw].md` in ~0.5 s).
  - [`b232a52`](https://github.com/Dicklesworthstone/franken_whisper/commit/b232a52) — **wave A**: `yt-dlp` orchestration (resolution/version-probe with a 90-day staleness warning and a `FRANKEN_WHISPER_YTDLP_BIN` override), collision-proof filename sanitizer (UTF-8-preserving, path-hostile-char folding, id suffix never truncated), and the markdown + JSON renderers (H1 title, metadata line, source link, honesty note, pause/speaker-grouped paragraphs with `youtu.be` deep-linked timestamps; structured `video`/`run`/`utterances` JSON).
  - [`d965a6f`](https://github.com/Dicklesworthstone/franken_whisper/commit/d965a6f) — **wave B/1**: cancel-correct ingestion pipeline + manifest state machine (per-video `discovered`/`done`/`failed` states, concurrency-bounded downloads, resumable re-runs, cooperative cancellation).
  - [`85b14f9`](https://github.com/Dicklesworthstone/franken_whisper/commit/85b14f9) — **wave B/2**: wired the `franken_whisper youtube` CLI subcommand (`--url`/positional URLs, `--batch-file`, `--output-dir`, `--model`, `--language`, `--backend`, `--diarize`, `--concurrency`, `--no-keep-audio`, `--no-retry`, `--abort-on-error`, `--json-summary`) plus a gated e2e integration test.
  - [`ed4f554`](https://github.com/Dicklesworthstone/franken_whisper/commit/ed4f554) — **live-test fix**: switched the download format selector to the forgiving `bestaudio/best` (a strict selector + a stale yt-dlp failed during real-YouTube testing); YouTube breakage is a yt-dlp concern, kept current with `yt-dlp -U`.

### Native Engine Performance Wave — Round 2: f16 decoder compute (post-2026-06-06)

- **f16-resident decoder compute is now the default production path** (bd-2th6 Round 2) — a second profile-driven optimization round reopened the three frontiers left at Round-1 convergence (f16 weight traffic, the encoder sgemm, the ft-side microkernel) and harvested the one that paid off. The decoder now keeps its matmul weights f16-resident and dequantizes them with a vectorized NEON-fp16 slice path (`convert_to_f32_slice` → 8-lane `dot8`) instead of f32-everywhere, **shaving −11.5 % off the large-v3-turbo e2e wall (min, interleaved A/B)** with byte-exact golden transcripts on both models. Decoder floors drop to **10.3 ms/token (large, f16 ON)** vs 38.9 ms/token (f16 OFF) and **5.2 ms/token (tiny)**. The encoder stays f32 — it is compute-bound GEMM where halving resident weight bytes buys no compute time (f16 panels measured pure overhead). Full pass-by-pass record, attribution table, and convergence statement: `tests/artifacts/perf/20260605T0218Z-native-engine-baseline/RESULTS.md` §6.
  - [`a236433`](https://github.com/Dicklesworthstone/franken_whisper/commit/a236433) — criterion bench substrate (`benches/native_engine_bench.rs`: mel, encoder window, decoder token step, logits GEMV, e2e tiny) with saved baselines — the outstanding bd-2th6 deliverable.
  - [`8abea12`](https://github.com/Dicklesworthstone/franken_whisper/commit/8abea12) — f16-resident decoder compute path with fused dequant-in-GEMV, shipped **opt-in / default off**: micro-bench win but an e2e regression (per-element widen serialized the FMA), kept as an env switch pending vectorization.
  - [`c703035`](https://github.com/Dicklesworthstone/franken_whisper/commit/c703035) — vectorized f16 dequant (dequant 13.9 → 56.2 GB/s, isolated f16 GEMV 5.3×) → **decoder f16 flipped to default ON**; the env var `FRANKEN_WHISPER_NATIVE_F16_COMPUTE` becomes an opt-**out** kill switch. Encoder f16 panels prototyped then skipped with measured proof.
  - [`0a5c939`](https://github.com/Dicklesworthstone/franken_whisper/commit/0a5c939) — decoder token-step per-sub-part attribution (zero-overhead unless `FRANKEN_WHISPER_PERF_SPANS=1`) + size-gated logits GEMV widening (8→12 workers for the vocab product only; logits_gemv_large −1.9…−3.9 %, bit-identical). cross_attn f16 K/V and wider head-workers rejected with measured floor diagnostics.
  - **franken-decision 0.3.2 migration + Round-2 landing** (this commit) — migrated the Bayesian backend router to the franken-decision 0.3.2 API: `evaluate()` now returns `Result<DecisionOutcome, ValidationError>`, and a validation failure engages the existing deterministic static-priority fallback (evidence-ledgered with the error string + `tracing::warn`); `DecisionContract::update_posterior` now returns `Result<(), UpdatePosteriorError>`, and a failed update is logged + skipped so it can never kill a transcription (the uniform prior is the safe default). Adds tests that a `ValidationError` engages the static fallback and that posterior-update failures fail-closed without mutating. Also fixed `stage_budget_timeout_maps_to_timeout_error_code` to use a multi-worker runtime (asupersync 0.3.2 cannot advance the budget timer concurrently with a blocking `spawn_blocking` body on a `current_thread` runtime; the production `run_stage_with_budget` always runs on a multi-worker runtime). f16 default-ON golden gate byte-exact on both models; full lib suite 3086/3086 green.

### Native Engine Performance Wave (post-2026-06-05)

- **Profile-driven optimization pass on the in-process Rust whisper engine** (bd-2th6) — six measured levers land a **3.3× speedup on tiny.en (1.57 s → 0.475 s, RTF 0.043)** and **4.6× on large-v3-turbo (44.6 s → 9.73 s, RTF 0.88)** on this host (Apple M4 Pro, 14 cores, `release-perf`). Interleaved same-host hyperfine: native tiny.en is **2.33× faster than whisper-cli CPU** (475 ms ± 13 vs 1105 ms ± 21); native large-v3-turbo reaches **parity** with whisper-cli CPU (9.731 s ± 0.272 vs 9.585 s ± 0.224) at *lower* user CPU (53.8 s vs 65.4 s). Both models are now faster than realtime. Each lever kept only on a measured win; bit-identical except DISC-004's scoped tail truncation.
  - [`2ed6471`](https://github.com/Dicklesworthstone/franken_whisper/commit/2ed6471) — version-tag SHA-256 hashed on a background thread at load, moving ~3.0 s off the run's critical path (tag value unchanged).
  - [`0361bb2`](https://github.com/Dicklesworthstone/franken_whisper/commit/0361bb2) — parallel f16→f32 dequant + tiled parallel weight transpose at load (`model_weights` span 2.4 s → 0.47 s).
  - [`bdbdd21`](https://github.com/Dicklesworthstone/franken_whisper/commit/bdbdd21) — language auto-detect reuses window-0's encode instead of a hidden duplicate encoder pass (−8.8 s on large).
  - [`ee36fd4`](https://github.com/Dicklesworthstone/franken_whisper/commit/ee36fd4) — parallelized the serial glue ops between fork-join matmuls (layer_norm / softmax / gelu / im2col / attention-head loops / logits GEMV / cross-K-V): encoder 8.3 → 5.8 s, tiny wall 0.92 → 0.56 s.
  - [`0989a5a`](https://github.com/Dicklesworthstone/franken_whisper/commit/0989a5a) — tail-window encoder-context truncation (whisper.cpp `audio_ctx`-style; default-on, kill switch `FRANKEN_WHISPER_NATIVE_TAIL_TRUNCATE=0`): tail-window encode 4.2 s → 0.24 s (~94%). Main transcript byte-identical; divergence confined to spurious trailing-silence hallucinations on tail windows. See `DISCREPANCIES.md` **DISC-004**.
  - [`5bb778b`](https://github.com/Dicklesworthstone/franken_whisper/commit/5bb778b) — decoder per-token path: KV buffer reuse, in-place logits band reads, hoisted window-constant cross K^T/V transforms, threaded QKV, head-parallel cross-attention (large 62.6 → 39.9 ms/tok, tiny 11.9 → 6.99 ms/tok; bit-identical, 4 new bitwise tests; ~490 MB cross-buffers dropped on large).
  - **Evidence-backed abandons** — fused QKV (bit-identical but ~16% slower), bmm-batched heads (~4% slower), parallel residual adds (slower), encoder scratch arena (wall/RSS-neutral; sgemm-compute-bound, ft calloc pages are lazy), and an ft-side matmul arena (~9% theoretical/large-window, a frankentorch change, below bar). Full arc + interleaved numbers: `tests/artifacts/perf/20260605T0218Z-native-engine-baseline/RESULTS.md`.
  - **Follow-up (not applied):** the `release` profile's `opt-level = "z"` costs ~26% on large vs `release-perf`; evaluate a `dist` profile change separately.

### Real Native Whisper Engine on FrankenTorch Kernels (post-2026-06-04)

- **All three native "pilot" mocks replaced by a genuine in-process Rust ASR engine** (bd-frp7 epic; engines bd-jryr / bd-s8w8 / bd-cidv; harness bd-4slu) — the former canned-phrase `WhisperCppPilot` / `InsanelyFastPilot` / `DiarizationPilot` are deleted. The native engine now parses whisper.cpp ggml `.bin` model files, computes the log-mel frontend, and runs the encoder/decoder transformer forward passes on `ft-kernel-cpu` (FrankenTorch) compute kernels, decoding tokens with whisper's timestamp rules — real speech recognition, no FFI, no network.
  - **Honest availability** (bd-jryr) — `native_engine::native_model_available()` header-sniffs a resolvable, well-formed ggml model (magic + supported dense `ftype`, no tensor load) instead of the previous dishonest `always true` constant. With no model present, the router stays bridge-only rather than advertising a fake native recovery path.
  - **DTW word-level timestamps** (bd-rjsx) — cross-attention weights of the model's alignment heads are recorded during decode and aligned via dynamic time warping to produce real per-word timings (`DecodeOutput::word_timings`), paid for only when `word_timestamps` is set.
  - **Honest diarizer tagging** (bd-cidv) — the native whisper-diarization path runs real ASR then the orchestrator's text/temporal heuristic diarizer, labeling segments `SPEAKER_NN`. Its `raw_output` declares `"diarizer": "text-temporal-heuristic"` (plus a `diarizer_note`) so no consumer mistakes it for neural diarization; the ECAPA upgrade remains tracked separately.
  - **Rollout-machinery e2e proof** (bd-4slu) — `tests/native_engine_e2e.rs` drives the full library dispatch (ingest → normalize → backend) through the CLI with `FRANKEN_WHISPER_NATIVE_EXECUTION=1` and `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole|primary`, pointing every bridge binary at `/nonexistent` so a produced transcript can only have come from the native engine. Covers whisper-cpp, insanely-fast, and diarization native paths, plus the honest bridge-only-unavailable error path. All gated on the real `tiny.en` model (skip-not-fail when absent).
  - **Bridge-vs-native conformance gate** (bd-4slu) — a doubly-gated comparison (`tiny.en` model AND `whisper-cli` present) runs the reference bridge and the native engine on `jfk.wav`, asserting WER ≤ 0.10 and per-segment timestamps within a documented **native-rollout tolerance profile** (0.3 s, not the canonical 50 ms) to absorb greedy-vs-beam divergence. See `DISCREPANCIES.md` DISC-003.
  - **Gated model fetch** — `scripts/fetch_test_models.sh` provisions `ggml-tiny.en.bin` (sha256-pinned, idempotent, `--force`) into `${FRANKEN_WHISPER_TEST_MODEL_DIR:-~/.cache/franken_whisper/test-models}` and verifies the in-repo `jfk.wav` fixture, turning the otherwise-skipped native conformance/e2e suites into real runs.

### Speculative Streaming: CLI Integration (post-2026-03-21)

- **`--speculative` flag wired through the CLI orchestrator** ([`4f54fbd`](https://github.com/Dicklesworthstone/franken_whisper/commit/4f54fbd0750565135f3c49c446b09b6615b57802)) — previously `TranscribeArgs::to_request()` rejected every speculative request with `FW-INVALID-REQUEST` and the documentation-vs-behavior gap was wide. The integration adds:
  - `SpeculativeRequest` to `BackendParams` (a serde-friendly mirror of the CLI knobs that the orchestrator converts to `streaming::SpeculativeConfig` at dispatch time, keeping `model.rs` free of any dependency on `streaming.rs`).
  - `audio::slice_pcm_wav_to_temp_path()` — in-memory `hound` slicer producing per-window WAVs from a normalized 16 kHz mono PCM source. Preserves source spec, clamps OOB ranges, rejects non-PCM input.
  - `execute_backend_speculative()` in the orchestrator, replacing the regular `Backend` stage when `request.backend_params.speculative.is_some()`. Drives `SpeculativeStreamingPipeline::process_file_with_models()` with a per-window runner that calls `backend::execute()` for the fast and quality lanes; forwards every `transcript.partial / confirm / retract / correct / speculation_stats` event into the run-wide NDJSON log.
- **Speculative dispatch hardening** ([`2dbab6f`](https://github.com/Dicklesworthstone/franken_whisper/commit/2dbab6fadbd059973b86124354e39989129476a4)) — three follow-up correctness fixes against the first cut:
  1. *Sticky backend*: the new `backend::resolve_static_backend()` walks the static priority list once and returns a single concrete `BackendKind`; the dispatch path forces that kind onto every per-window invocation so `--backend auto` runs are engine-consistent end to end instead of letting per-window Bayesian routing produce a mixed-engine transcript.
  2. *Partial event preservation on failure*: the inner `Result` is now returned wrapped in an always-`Ok` outer carrier so the outer match can forward `emitted_events` whether the speculation pipeline succeeded or failed mid-window. The `backend.error` event also gains a `windows_processed_before_failure` field.
  3. *Populated `ReplayEnvelope` for speculative runs*: `inter.backend_runtime = Some(backend::runtime_metadata(resolved))` and `inter.backend_output_sha256 = sha256_segments(&merged)`, so speculative runs land replay envelopes that downstream drift detection / conformance tooling can meaningfully diff.

### Storage: Concurrent-Persist Durability Contract (post-2026-03-21)

- **`storage::tests::concurrent_persist_10_threads_with_segments_and_events` tightened to require 10/10 successful persists** ([`eede00a`](https://github.com/Dicklesworthstone/franken_whisper/commit/eede00a7838c4ac4aba47f0c6a00cfaddbafbb8f)) — the test was previously written defensively against an older `fsqlite` MVCC limitation that silently dropped some concurrent writes; its assertions accepted "majority survives" (`>= 5 of 10`) as success. The underlying MVCC concurrent-persistence gap has since been resolved upstream in `fsqlite-mvcc`, so the test now enforces full durability instead of legitimizing data loss. Per-thread `false` returns from the retry-exhaustion paths are now described as diagnostics (so the aggregate assertion can name which thread didn't land) rather than as accepted dropped writes.

### Documentation Refresh (post-2026-03-21)

- **README comprehensive rewrite** ([`9ddb9db`](https://github.com/Dicklesworthstone/franken_whisper/commit/9ddb9dbab2528fdd2c2f2b10cc82ff6bd74de6a3)) — full top-to-bottom rewrite (~4,300 lines) reflecting the current architecture and capabilities, with worked examples (Bayesian routing arithmetic, mu-law encoding, Brier-score decomposition), anatomy walkthroughs (routing decision, TTY audio session, conformance check, speculative window), operational topics (state directory layout, CLI exit codes, threat model, performance tuning, disk usage growth, extension guides, debugging recipes), a use-cases gallery, JSON schema reference, and embedding-in-other-languages patterns.
- **Removed stale "frankensqlite MVCC limitation" Limitations entry** from the README, since the underlying bug has been resolved upstream in the current `/dp/frankensqlite` checkout that the path dependency picks up.
- **README test count corrected** from the marker-derived 3,860 to the runner-derived 3,668 (3,030 library + 638 integration across 26 integration test files).

### Workspace Hygiene (post-2026-03-21)

- **Pre-existing clippy gate failures fixed** ([`47fe23b`](https://github.com/Dicklesworthstone/franken_whisper/commit/47fe23b4699b766864c73244d4e2b814cdcc9e7d)) — `src/backend/whisper_diarization.rs:311` and `tests/conformance_harness.rs:696` `?`-operator rewrites; `tests/metamorphic_audio_tests.rs:35` unused `generate_silence` annotated `#[allow(dead_code)]`; `tests/metamorphic_audio_tests.rs:73` `repeat().take()` modernized to `repeat_n()` per `clippy::manual_repeat_n` on Rust 1.82+.
- **`Cargo.lock` refresh** ([`7855910`](https://github.com/Dicklesworthstone/franken_whisper/commit/7855910e8cfc6d43c91d1ec2f9b7b07e38d32c89)) tracking upstream workspace bumps from sibling-agent commits that touched workspace crates while this branch was off the build path.

---

### Release Pipeline and Distribution

- **`dist.yml` + `release-automation.yml` release pipeline** — multi-platform binary builds (Linux x86_64/ARM64, macOS Intel/ARM64, Windows x86_64), tag-driven release automation that watches `Cargo.toml` and creates `v{VERSION}` tags when the version changes ([`d819076`](https://github.com/Dicklesworthstone/franken_whisper/commit/d819076b5eab5a09c218e680a20e8dae3589ae91))
- Dist asset names aligned with `install.sh` expectations ([`d70d7c3`](https://github.com/Dicklesworthstone/franken_whisper/commit/d70d7c3b9ae546c45f2d488aacbcbaa353f76151))
- Test job's `cargo fmt --check` made advisory; toolchain channel matched against `rust-toolchain.toml` ([`4a406d0`](https://github.com/Dicklesworthstone/franken_whisper/commit/4a406d01a1152720a7aaa36fad9196b678d02bd3), [`7cf1bfc`](https://github.com/Dicklesworthstone/franken_whisper/commit/7cf1bfcb6c6999f8e9364803d43654c84c38ac7a))
- **curl|bash installer** with SHA-256 checksum verification for one-line installs ([`e06d87f`](https://github.com/Dicklesworthstone/franken_whisper/commit/e06d87f1c812d2acfd03459973a4cf36a9ced23c))
- Installer strips `v` prefix from version in asset URLs ([`4cc9b66`](https://github.com/Dicklesworthstone/franken_whisper/commit/4cc9b664c82abbd4f4d30e1145de8b8f56000a2d))
- Installer handles versioned binary names in release archives ([`6a48022`](https://github.com/Dicklesworthstone/franken_whisper/commit/6a480224261a235b942e7e172c5f70ed29fc38e2))

### Dependency Migration to crates.io

- **`asupersync`, `franken-kernel`, `franken-evidence`, `franken-decision` migrated from local path dependencies to crates.io v0.3.0** — removes the requirement to clone sibling FrankenSuite repositories alongside `franken_whisper` for normal builds ([`d75c33f`](https://github.com/Dicklesworthstone/franken_whisper/commit/d75c33f4e9af26cd5a0fea804d8f3cf4d4d48218))
- `asupersync` bumped to 0.3.1 ([`e8c1508`](https://github.com/Dicklesworthstone/franken_whisper/commit/e8c1508c3830cd56d465130a4b345b418ccbae07))
- `Cargo.lock` refreshed against upstream dependency bumps ([`da4125b`](https://github.com/Dicklesworthstone/franken_whisper/commit/da4125bb1c85719f25bc39659e7ec78d2e589977))

### Conformance: Native Engine Rollout Governance

- **Backend version drift detection, native-pilot fixture generator, and expanded invariant coverage** — `NativeEngineRolloutStage` enum (Shadow → Validated → Fallback → Primary → Sole), shadow run comparator validating segment parity + replay envelope between bridge and native, `scripts/gen_native_fixtures.py` corpus generator, `tests/COVERAGE.md` mapping spec clauses to tests, and 30+ new conformance fixtures covering long segments, word-level boundaries, replay drift, timestamp edge cases, speaker label patterns, multilingual / code-switching / noisy-overlap corpora ([`c72eed3`](https://github.com/Dicklesworthstone/franken_whisper/commit/c72eed330428dab607eaced659cffda0a5e80083))
- `tests/COVERAGE.md` tracking spec clause coverage ([`5a1ec0d`](https://github.com/Dicklesworthstone/franken_whisper/commit/5a1ec0db090d0e69d4814ea92dfdc30fe56489cd))
- Conformance fixtures expanded with very long segments + word-level boundary cases ([`1f16dc5`](https://github.com/Dicklesworthstone/franken_whisper/commit/1f16dc59e2046f75b56a28f7bf5c5afd9210f508)) and timestamp / long segment edge cases ([`48078fe`](https://github.com/Dicklesworthstone/franken_whisper/commit/48078fe4d67fe7f6efbf27749a4622aa799c64da))

### Pipeline: Word-Level Timestamps and Cancellation Threading

- **Word-level timestamp pipeline support** (`WordTimestampParams`: `enabled`, `max_len`, `token_threshold`, `token_sum_threshold`) plus end-to-end **cancellation token threading through pipeline stages** so stage budgets are honored by the cancellation tokens, mid-pipeline cancellation no longer leaks partial subprocesses or partial transactions, and conformance assertions tighten around word-level segment invariants ([`f1cbd31`](https://github.com/Dicklesworthstone/franken_whisper/commit/f1cbd31560ea0d1586be4771224077ea5ed4729b))
- Honor stage budgets in cancellation tokens (bd-xunn) ([`9e40cda`](https://github.com/Dicklesworthstone/franken_whisper/commit/9e40cda96a89ad575e4acf9325463159587bea44))
- Block speculative CLI flag where unsupported; stream pipe output reliably; bump deps ([`25ff052`](https://github.com/Dicklesworthstone/franken_whisper/commit/25ff0521c334ef3cdbf728c74422010700e22ceb))

### Adaptive Router: Calibration Observations

- **`record_adaptive_router_prediction` captures predicted probability at routing decision time** so calibration observations (predicted vs. actual outcome) feed the Brier-score sliding window even when the router falls back to static priority ([`ef80eb8`](https://github.com/Dicklesworthstone/franken_whisper/commit/ef80eb8daff5d6385d477a73cbe3db63a5e717c9))
- Calibration fallback path now records the adaptive prediction it would have made (bd-k87e), preventing fallback runs from silently corrupting calibration data ([`ed91ddd`](https://github.com/Dicklesworthstone/franken_whisper/commit/ed91ddddcbef30f9401cecb8568b24529d8d436e))
- Non-finite calibration predictions sanitized (bd-bu04) ([`1c77e43`](https://github.com/Dicklesworthstone/franken_whisper/commit/1c77e4362650aa1c5c7996a7aa44be01d95be2e5))

### Metamorphic Test Suites

- **`tests/metamorphic_accelerate_tests.rs` (+475 lines)** — softmax / normalization / attention scoring invariants under permutation, scaling, and zero-padding transformations ([`47583d5`](https://github.com/Dicklesworthstone/franken_whisper/commit/47583d5073ea6049ff36fe846a9120db5278e676))
- **`tests/metamorphic_speculation_tests.rs` (+481 lines)** — string distance and window-invariant metamorphic properties ([`3c0c972`](https://github.com/Dicklesworthstone/franken_whisper/commit/3c0c97271a285a2e53358b246f4178d4d5025ecb))
- **`tests/metamorphic_audio_tests.rs` (+515 lines)** — audio processing invariants (resampling commutativity, mono mixing properties, silence-padding equivalence) ([`78b9db6`](https://github.com/Dicklesworthstone/franken_whisper/commit/78b9db692d10a8e1cccc185f1b7d925e8e710adc))

### Audio Pipeline Hardening

- **Native audio backend expanded with format detection** and sync improvements ([`1ad594e`](https://github.com/Dicklesworthstone/franken_whisper/commit/1ad594ebf8b0bc8d6412d3075d1d84924125db86))
- Audio frame mono averaging uses actual frame length rather than nominal channel count ([`ad9c51b`](https://github.com/Dicklesworthstone/franken_whisper/commit/ad9c51b39fcc98b1d84ac4c40172b4933e89abf6))
- `normalize_cpu` empty-input edge case guarded; two-lane streaming executor propagates thread panics rather than silently hanging ([`2d8457a`](https://github.com/Dicklesworthstone/franken_whisper/commit/2d8457abe503721a7e76d9dd50ae7e510f512046))
- Audio normalization + streaming I/O hardened with cancellation checks ([`59a33df`](https://github.com/Dicklesworthstone/franken_whisper/commit/59a33df78527039b58afd5c26189e1df9bc3a834), [`13c8c80`](https://github.com/Dicklesworthstone/franken_whisper/commit/13c8c801cd050c9c80865f49840006e234fc71d1))

### Robot Mode: Structured Error Envelopes and Diagnostics

- **Structured `run_error` envelope for invalid robot requests** — malformed CLI args now produce a well-formed NDJSON error event instead of a human-readable stderr message ([`1e576a9`](https://github.com/Dicklesworthstone/franken_whisper/commit/1e576a9d021ff8f4fd23ac246b35e07e35546c9d))
- Robot mode avoids human-decorated stderr ([`2a5982e`](https://github.com/Dicklesworthstone/franken_whisper/commit/2a5982e9f5943c57b40fce3f853451598c89dd48))
- Robot error emitted on worker thread panic ([`8ab2ef8`](https://github.com/Dicklesworthstone/franken_whisper/commit/8ab2ef8ba7f1b46d6631e8969844eb1ab4e85ab3))
- Database health check hardened (`robot health` integrity probes resilient to migration / corruption edge cases) ([`83b23be`](https://github.com/Dicklesworthstone/franken_whisper/commit/83b23be88778121e29d67762f9dc89910be373fe), [`fe058e2`](https://github.com/Dicklesworthstone/franken_whisper/commit/fe058e247cbfb9bf691e5896a3414cfe0762407c))
- ffmpeg health probe improved ([`092bc1e`](https://github.com/Dicklesworthstone/franken_whisper/commit/092bc1e80555a1c552e8e81b485be57951d0afc7))

### Sync Lock Robustness

- Sync lock kept alive while owning PID runs (bd-qxf6); stale locks age-gated when PID unknown (bd-3jfj/bd-pvol); EPERM on `kill -0` treated as alive (bd-aijz); Windows `tasklist` errors treated as unknown (bd-7xh8); Windows PID liveness check improved (bd-ht0i) ([`396c652`](https://github.com/Dicklesworthstone/franken_whisper/commit/396c652e49b82558abf6dd89163610b98a43baa5), [`1aa7cf4`](https://github.com/Dicklesworthstone/franken_whisper/commit/1aa7cf4faff927fb9af3352036eb4145d14e1e79), [`eb0b9b8`](https://github.com/Dicklesworthstone/franken_whisper/commit/eb0b9b8076147c961f312bfac11f397636cfd4a7), [`80d74db`](https://github.com/Dicklesworthstone/franken_whisper/commit/80d74db596c6d5dba831097108474817b40189cc), [`afa5c16`](https://github.com/Dicklesworthstone/franken_whisper/commit/afa5c16a4b0cb9397dcedd2d54101ec4dbaa9bf7), [`4e1ee33`](https://github.com/Dicklesworthstone/franken_whisper/commit/4e1ee33f774078c4a44e9e1ff5b091c37fcc65fb))
- Sync lock parsing hardened; SQLite table identifiers quoted ([`da9a844`](https://github.com/Dicklesworthstone/franken_whisper/commit/da9a84453214243b7e2de7ee4ff0bdd41cb1c322))
- Sync lock release path hardened alongside ffmpeg bundle work (bd-g7b2) ([`b36ef02`](https://github.com/Dicklesworthstone/franken_whisper/commit/b36ef0296c26e8e8bf77fa3ab998c4fd5e40360c))

### ffmpeg Bundle and Stdin Hygiene

- ffmpeg bundle scan hardened (bd-wk8p); stdin extension sanitization + ffmpeg download test stabilization (bd-o8fo) ([`79f3492`](https://github.com/Dicklesworthstone/franken_whisper/commit/79f3492dbeb01a5ed47706766e9935e46b22de49), [`eb220c8`](https://github.com/Dicklesworthstone/franken_whisper/commit/eb220c810b437a6594ccbfac3c668f5777df13bc))
- `tty-audio retransmit-plan` protocol version corrected ([`6076902`](https://github.com/Dicklesworthstone/franken_whisper/commit/60769026c8298f96ecd478540b17ce60ac1df802))
- Legacy `tty-audio` version consistency enforced (bd-ithy) ([`036fbcf`](https://github.com/Dicklesworthstone/franken_whisper/commit/036fbcf4d990a112039422d4ef8ee7bef990727e))

### Secrets Redaction

- **HuggingFace token redacted in command logs** via `render_command_for_log()` — diarization invocations no longer leak `--hf-token` values into tracing output ([`60bf952`](https://github.com/Dicklesworthstone/franken_whisper/commit/60bf9521520865767c69fa07bb4d03e6a68248a8), [`8b3249a`](https://github.com/Dicklesworthstone/franken_whisper/commit/8b3249aa420454b502c633657167fc70006fe828))

### Storage and Diarization Fixes

- Storage concurrency tests stabilized (bd-pdyy) ([`f82031d`](https://github.com/Dicklesworthstone/franken_whisper/commit/f82031d0ae1c27210325f4a85f38c2d170a60ddc))
- Diarize clustering hardened (bd-0ydr) ([`3adb608`](https://github.com/Dicklesworthstone/franken_whisper/commit/3adb6080da2787c66bcb7432198db213ed7c4ca3))
- Orchestrator UBS critical findings resolved (bd-9ftg) ([`ad4d98b`](https://github.com/Dicklesworthstone/franken_whisper/commit/ad4d98b54cc9502077af370b033c5d98cf8da3f3))
- SRT parsing fallback fixed ([`d409056`](https://github.com/Dicklesworthstone/franken_whisper/commit/d409056ba265bd1f7688578e1326edf2a1acb26e))
- Diarization transcript fallback and command pipe output loss fixed ([`a8328bc`](https://github.com/Dicklesworthstone/franken_whisper/commit/a8328bcdf02b93f67ab788476c7ba198667af819))
- Empty parent components ignored when ensuring directories exist ([`0a21253`](https://github.com/Dicklesworthstone/franken_whisper/commit/0a2125335e8e0cd7ab11b696c850b61dd3d503f0))
- GPU cancellation evidence ledger tests: missing `expected_loss` entries supplied ([`a20b828`](https://github.com/Dicklesworthstone/franken_whisper/commit/a20b82808d93ae65cc52d34e13f7edbac9463fc4))
- Speculation partials handled safely when missing ([`44729dc`](https://github.com/Dicklesworthstone/franken_whisper/commit/44729dc222fd6b7cc69d42f3b4481bd3207ff20e))

### CLI and Robot Surface

- **Expanded CLI** with structured subcommands and output modes (+122 lines) ([`ef221f7`](https://github.com/Dicklesworthstone/franken_whisper/commit/ef221f7ba39c1f8d6bd1caf36f2b34c8454608b6))
- **Expanded robot output** with structured analysis views (+90 lines) ([`32ba54f`](https://github.com/Dicklesworthstone/franken_whisper/commit/32ba54f2e53c9e5fff07ed92abe854e797a70f21))
- **`routing_decision` event builder** extracted in `robot` module; backends output aligned with NDJSON schema ([`6beedc3`](https://github.com/Dicklesworthstone/franken_whisper/commit/6beedc3db49efcf2c26d3f712ee0d78abf62c125))
- CLI argument parsing edge case corrected ([`8a8a58e`](https://github.com/Dicklesworthstone/franken_whisper/commit/8a8a58e0baba1df01dc4eecb6672a013f1ae7412))
- CLI integration tests and orchestrator refinement ([`a7cca9e`](https://github.com/Dicklesworthstone/franken_whisper/commit/a7cca9e0c5d627be27ab0bb0aafab48884504c35))

### Speculative Streaming

- **Adaptive window controller** for speculative streaming pipeline — dynamically adjusts decode windows based on correction rates ([`61b7d40`](https://github.com/Dicklesworthstone/franken_whisper/commit/61b7d40073f41268280033ceac0a8251e0179923))
- None-timestamp segment handling corrected in speculation and TUI; run detail load errors surfaced ([`dd04e9c`](https://github.com/Dicklesworthstone/franken_whisper/commit/dd04e9cad44621911b5cf68e69c5f03330665e55))
- Speculation test error assertions simplified with `expect_err` ([`33423d5`](https://github.com/Dicklesworthstone/franken_whisper/commit/33423d582363e619be4c8ea0161fbb98720bac21))

### Audio Processing

- **XDG_STATE_HOME support** for tool state storage; ffmpeg extracted to tmpdir instead of cwd ([`10be321`](https://github.com/Dicklesworthstone/franken_whisper/commit/10be321a493db1ce8617f82172c6d2b12716ef88))
- Active region boundaries clamped to actual audio duration, preventing out-of-bounds processing ([`2bca2f9`](https://github.com/Dicklesworthstone/franken_whisper/commit/2bca2f9fd48b5be23b3434bf5bef25ed5953bc1e))
- Audio processing pipeline refinement ([`162da2a`](https://github.com/Dicklesworthstone/franken_whisper/commit/162da2ad1154970e7a9bc0c1451186e3254f9215))

### TTY Audio Transport

- **TTY audio demo script** for interactive testing ([`6a5234d`](https://github.com/Dicklesworthstone/franken_whisper/commit/6a5234dff4a43a6fa375ca6165519fff062a6222))
- Extended TTY audio pipeline with telemetry tests ([`b8edfaf`](https://github.com/Dicklesworthstone/franken_whisper/commit/b8edfafa1add287c6d666e185681fb2e071feb4a))
- Frame validation extracted; sequence gaps rejected on finalize ([`f6f5f41`](https://github.com/Dicklesworthstone/franken_whisper/commit/f6f5f41ef83ffa347c8c6d75bfce885d3433bc44))
- TTY telemetry test assertions refined ([`ab2cf8e`](https://github.com/Dicklesworthstone/franken_whisper/commit/ab2cf8e2b82c27a52ee78da2e48e13c8baa1db08))

### Storage and Persistence

- **Migration-safe column detection** and defensive query wrappers (+35 lines) ([`23b5062`](https://github.com/Dicklesworthstone/franken_whisper/commit/23b5062619fa5a9c73efffcda659be61f102d7df))
- **Replay pack validation** hardened alongside storage queries ([`5248ad5`](https://github.com/Dicklesworthstone/franken_whisper/commit/5248ad5e37e43e1b093f1f82f262898a0dabd593))
- Silent `unwrap_or_default` replaced with explicit JSON parse error propagation ([`c2f5474`](https://github.com/Dicklesworthstone/franken_whisper/commit/c2f5474936b844e99b24beec2f27fcedad0e23bb))
- `SqliteValue::Text(String)` and `Blob(Vec<u8>)` migrated to `Arc`-based types for reduced cloning overhead ([`4066d5d`](https://github.com/Dicklesworthstone/franken_whisper/commit/4066d5d7f14f085c7eb1fcd6d2fcfbca52e27b1c))
- Storage query patterns simplified; dead code paths removed ([`64d340e`](https://github.com/Dicklesworthstone/franken_whisper/commit/64d340e4d6e6fe933d8da8184a1b0a07f89edcc1), [`42f718a`](https://github.com/Dicklesworthstone/franken_whisper/commit/42f718a2e018e2297acda91a1fcdb9f96146c002))
- Storage queries improved alongside TTY audio demo extension ([`527f443`](https://github.com/Dicklesworthstone/franken_whisper/commit/527f443356c2ede453b4d3474524f91df5832954))

### Sync Pipeline

- **Parallel sync pipeline** with progress tracking (+44 lines) ([`1999137`](https://github.com/Dicklesworthstone/franken_whisper/commit/1999137e6ba3e2c31b201e5c65381ebaaaea5d1b))
- Sync timeout overflow fixed with `saturating_mul`; `backend.ok` preferred as rollout stage; atomic rename made portable ([`9f94946`](https://github.com/Dicklesworthstone/franken_whisper/commit/9f94946753a775d253da17f5b8d3e88a8195d442))
- Off-by-one corrected in sync progress tracking ([`0105d9a`](https://github.com/Dicklesworthstone/franken_whisper/commit/0105d9a29f9b443ec82e692e5ba546cc20d85047))
- Sync pipeline refined ([`6b5e29b`](https://github.com/Dicklesworthstone/franken_whisper/commit/6b5e29b39afa624ff9360d9d8e86681f0b85c885))

### Pipeline Orchestration

- **Backend module expansion** with improved pipeline orchestration ([`d9166c3`](https://github.com/Dicklesworthstone/franken_whisper/commit/d9166c3a3c42175a472d904b54f660cf28a52a74))
- `stage_start` cleared on checkpoint failure to prevent elapsed time leaking across pipeline stages ([`b15578e`](https://github.com/Dicklesworthstone/franken_whisper/commit/b15578ea6b403b1cca148c5fec22b5e3c49dd067))
- Pipeline stage transitions refined ([`5f3f187`](https://github.com/Dicklesworthstone/franken_whisper/commit/5f3f1870d393feed4d3b4116a3cc68664680112f))
- 8 pipeline stage `.expect()` panics replaced with proper `FwError` returns ([`0c4d557`](https://github.com/Dicklesworthstone/franken_whisper/commit/0c4d557ac5926e094a3db8b8415a4899413576e9))
- Backend/storage pipeline streamlined for robustness ([`b19eec9`](https://github.com/Dicklesworthstone/franken_whisper/commit/b19eec97545e06eef5e3b8842dfca10595180cf9))
- Module streamlining ([`ca4f587`](https://github.com/Dicklesworthstone/franken_whisper/commit/ca4f587c5e03a4d3a0e44feb386c9380bb89c0c4))

### Acceleration

- Division by zero in `layer_norm_cpu` prevented by clamping epsilon floor ([`b3c8b56`](https://github.com/Dicklesworthstone/franken_whisper/commit/b3c8b56e9290a8b73a6a527b72091419e5107d19))

### CI and Quality Gates

- **Persistent worker quarantine with TTL** for quality-gate routing ([`9ecf72d`](https://github.com/Dicklesworthstone/franken_whisper/commit/9ecf72da3615ebe4dd9d667b06cdd42574a81086))
- **flock guard and preblock** for unreachable workers in quality-gate routing ([`77900fb`](https://github.com/Dicklesworthstone/franken_whisper/commit/77900fb76984aa0183bc50ca45a04d3bc44ad4e5))
- Quality-gate retry routing hardened with worker quarantine and expanded retryable patterns ([`1ff4910`](https://github.com/Dicklesworthstone/franken_whisper/commit/1ff4910c067afb70605d524c3b27d1114ed54ac0))
- Stale-sibling validation rejected when dependency sync is degraded ([`670a758`](https://github.com/Dicklesworthstone/franken_whisper/commit/670a75803995f1ae372cb0a7fdec9db80e2e2ce8))

### Documentation

- Major README refresh with updated architecture and feature documentation (+1,867 lines across three commits) ([`2a2bb9f`](https://github.com/Dicklesworthstone/franken_whisper/commit/2a2bb9fd333df71e1e0069d3086d2e8a49315c7c), [`7d207b8`](https://github.com/Dicklesworthstone/franken_whisper/commit/7d207b8ba23ee18573b12a9a8d9dbeadf16258be), [`ba227a6`](https://github.com/Dicklesworthstone/franken_whisper/commit/ba227a60fab5558e0d09c06171c4e37e5babd49e))
- Robot event catalog updated with `routing_decision`, `backends.discovery`, and streaming events ([`ecf702b`](https://github.com/Dicklesworthstone/franken_whisper/commit/ecf702bea3219c697871788f4976d48b0f9dbb40))
- `CHANGELOG.md` rebuilt from git history with live commit links ([`d678a9e`](https://github.com/Dicklesworthstone/franken_whisper/commit/d678a9ee57822cbdee44f14e477a21018f4628a3), [`0059e01`](https://github.com/Dicklesworthstone/franken_whisper/commit/0059e019eeae376dc5bb4298bd74f3ca26d01dd6))
- `DISCREPANCIES.md` tracking known native-vs-bridge divergences (introduced alongside conformance work in `c72eed3`)
- Implementation tracker and execution packet documentation updated ([`b34893e`](https://github.com/Dicklesworthstone/franken_whisper/commit/b34893e0e637ece3023c75572036e054b635ebf9))

---

## [v0.1.0] — 2026-03-04

**Initial release.** GitHub Release: [`v0.1.0`](https://github.com/Dicklesworthstone/franken_whisper/releases/tag/v0.1.0) | Published: 2026-03-04

Lightweight tag on [`6a51618`](https://github.com/Dicklesworthstone/franken_whisper/commit/6a51618d0da9b69421f4bc1b7deccb1780b0b558). 59 commits from initial commit (2026-02-22) to tag. 81 files changed, +33,184 / -1,321 lines.

Compare: [`2e9f2e9...v0.1.0`](https://github.com/Dicklesworthstone/franken_whisper/compare/2e9f2e97e0ce8f37d71c2ae00eece2904f3fdd19...v0.1.0)

### Release Artifacts

| Platform | Archive | Size |
|----------|---------|------|
| Linux x86_64 | `franken_whisper-0.1.0-linux_amd64.tar.gz` | 2.3 MB |
| macOS arm64 (Apple Silicon) | `franken_whisper-0.1.0-darwin_arm64.tar.gz` | 2.1 MB |
| Windows x64 | `franken_whisper-0.1.0-windows_amd64.zip` | 2.2 MB |

SHA-256 checksums: [`checksums-sha256.txt`](https://github.com/Dicklesworthstone/franken_whisper/releases/download/v0.1.0/checksums-sha256.txt)

### Core Architecture

- **10-stage composable pipeline**: Ingest, Normalize, VAD, Source Separate, Backend, Accelerate, Align, Punctuate, Diarize, Persist — stages composed dynamically per-request, skipped when unnecessary, budgeted independently, profiled automatically ([`2e9f2e9`](https://github.com/Dicklesworthstone/franken_whisper/commit/2e9f2e97e0ce8f37d71c2ae00eece2904f3fdd19))
- **Adaptive Bayesian backend routing** with explicit loss matrix, posterior calibration, and deterministic fallback across `whisper.cpp`, `insanely-fast-whisper`, and `whisper-diarization`
- **Real-time NDJSON streaming** with stable schema (v1.0.0) — every pipeline stage emits sequenced, timestamped events for agent consumption
- **Cooperative cancellation** via `CancellationToken`-based graceful shutdown with resource cleanup
- **Per-stage timeout budgets** with automatic latency profiling and tuning recommendations
- **12 structured `FW-*` error codes** (`FW-IO`, `FW-CMD-TIMEOUT`, `FW-BACKEND-UNAVAILABLE`, etc.) for machine-readable error handling
- **`#![forbid(unsafe_code)]`** enforced — zero unsafe blocks across entire codebase
- Rust 2024 edition, nightly toolchain, `cargo clippy --all-targets -- -D warnings` enforced

### Audio Processing

- **Built-in Rust audio decoder** via symphonia: MP3, AAC, FLAC, WAV, OGG, ALAC decoded natively with zero ffmpeg dependency for common formats ([`934a8f6`](https://github.com/Dicklesworthstone/franken_whisper/commit/934a8f639793aec6ae80b3ad823e74369fa3119c))
- Built-in Rust decoder made the primary normalization path; ffmpeg demoted to fallback ([`934a8f6`](https://github.com/Dicklesworthstone/franken_whisper/commit/934a8f639793aec6ae80b3ad823e74369fa3119c))
- **Automatic ffmpeg provisioning** with bridge-native recovery fallback and sync validation tests ([`ca25b05`](https://github.com/Dicklesworthstone/franken_whisper/commit/ca25b051e774e21f2ff9d8bbdaa8cd35d65a73a6))
- ffmpeg provisioning subsystem unit tests ([`e794156`](https://github.com/Dicklesworthstone/franken_whisper/commit/e7941569d294b805a966ba95bae941a82de23b7f))
- **VAD redesign** and audio pipeline hardening with streaming engine trait ([`80969cb`](https://github.com/Dicklesworthstone/franken_whisper/commit/80969cba857644458005f5263bb82bffbd97749e))
- Audio duration probing hardened for edge cases ([`109ebc1`](https://github.com/Dicklesworthstone/franken_whisper/commit/109ebc1dd671a93f4d1ce4e06f415823b520ecc3))
- Input validation: reject directory paths with `is_file` guard ([`830a443`](https://github.com/Dicklesworthstone/franken_whisper/commit/830a4431e799305223bcf54ab473a069ca651819))

### Speculative Streaming

- **Speculative cancel-correct streaming pipeline** with evidence ledger and file processing — dual-model fast+quality architecture ([`ad4b54a`](https://github.com/Dicklesworthstone/franken_whisper/commit/ad4b54a39c58a8c5e37e66f4470778d0c48308e6))
- TUI speculation display and TTY transcript control frame support ([`847658d`](https://github.com/Dicklesworthstone/franken_whisper/commit/847658d66d1f3364ccc550344b69f6eb7c7be594))
- Conformance validation extended for speculation events ([`30044ab`](https://github.com/Dicklesworthstone/franken_whisper/commit/30044abd4cdec52cda71be7ea350c5006856560b))
- Comprehensive integration test suites for speculative pipeline ([`bc2a526`](https://github.com/Dicklesworthstone/franken_whisper/commit/bc2a526e0ba15d7714b602c5f365059c5f6e56a5))
- Massive edge-case test expansion for speculative pipeline ([`a2d5234`](https://github.com/Dicklesworthstone/franken_whisper/commit/a2d5234c220ede7be14a3f7b06a75c1e08eae182))

### TTY Audio Transport

- **mulaw+zlib+base64 NDJSON protocol** for low-bandwidth audio relay over PTY links with handshake, integrity checks, and deterministic retransmission
- VAD/separate/punctuate/diarize pipeline stages wired into TTY session protocol with deep audit tests ([`f183d08`](https://github.com/Dicklesworthstone/franken_whisper/commit/f183d08a7146949f1ab7544d5a704f68fa7409a7))
- `ControlFrameType` references updated to `TtyControlFrame` enum ([`d625089`](https://github.com/Dicklesworthstone/franken_whisper/commit/d62508926b3a3a5254a0e26037a4684d4c33bf8a))

### Storage and Persistence

- **SQLite-backed run history** with JSONL export/import and SHA-256 replay envelopes
- **Rollback-safe legacy v1 → v2 runs schema migration** ([`1424a8d`](https://github.com/Dicklesworthstone/franken_whisper/commit/1424a8de48201a640ba86da6da9ff61d71a3bb07))
- **`acceleration_json` column** propagated through sync pipeline with fail-closed overwrite semantics ([`9c7df9d`](https://github.com/Dicklesworthstone/franken_whisper/commit/9c7df9d426b03261bd4250bac0f01b9e967f4391))
- **Cascade-aware overwrite tracking** with `overwrite-strict` conflict policy for verified child-row replacement ([`4c7cc45`](https://github.com/Dicklesworthstone/franken_whisper/commit/4c7cc4515d47287cee3337f8793b874f2ac93c35), [`e4db512`](https://github.com/Dicklesworthstone/franken_whisper/commit/e4db512b96c3779693e04788566663342f2e3e5d))
- Migration error messages enriched with contextual details ([`07f24f6`](https://github.com/Dicklesworthstone/franken_whisper/commit/07f24f6afbee2fcd9318f6a178c7252192718f12))
- MVCC visibility retry in concurrent write test ([`4e713a2`](https://github.com/Dicklesworthstone/franken_whisper/commit/4e713a209906f085466d41b8788b88671b4ae639))
- Incremental export corrected to use `finished_at` instead of `started_at` ([`8974203`](https://github.com/Dicklesworthstone/franken_whisper/commit/8974203d9b0bf0816a655e60c5a82f5f45657842))
- Massive test expansion covering storage, protocol, and streaming improvements ([`2418ea7`](https://github.com/Dicklesworthstone/franken_whisper/commit/2418ea7fa1047c77df0b75d900d132a6d41356a2))

### Diarization

- **TinyDiarize support**: whisper.cpp's built-in speaker-turn detection without HF token ([`7abe271`](https://github.com/Dicklesworthstone/franken_whisper/commit/7abe27163dfcbd7e146d469c7aba562f531f4dd6))
- `dead_code` annotations removed from pipeline stages alongside TinyDiarize addition ([`7abe271`](https://github.com/Dicklesworthstone/franken_whisper/commit/7abe27163dfcbd7e146d469c7aba562f531f4dd6))
- Speaker constraints integration tests ([`3412914`](https://github.com/Dicklesworthstone/franken_whisper/commit/3412914ffa6f26b3f20a3f8f85e8d82a1029f89a))

### Orchestration and Backend Routing

- **Parallel processing and retry logic** for transcription orchestration ([`5c70d3d`](https://github.com/Dicklesworthstone/franken_whisper/commit/5c70d3de4fd55080c6a559e1e83d61ef721349a3))
- **whisper_cpp backend hardening** with expanded orchestrator capabilities and pipeline robustness ([`1f6f694`](https://github.com/Dicklesworthstone/franken_whisper/commit/1f6f694603ef1c9682a82a6efe9f17bff3147d1b))
- Routing mode helper extracted; shutdown and storage resilience hardened; sync cursor upgraded to composite key ([`7cad3b5`](https://github.com/Dicklesworthstone/franken_whisper/commit/7cad3b5d3a215cacb8500b121c1794ac14642bca))
- Segment timestamps sanitized at extraction boundary ([`bd99c6c`](https://github.com/Dicklesworthstone/franken_whisper/commit/bd99c6cf07403a0285f6c63daeec16e5160e1ea9))
- Mutex poisoning handled gracefully in two-lane streaming executor ([`b1682e8`](https://github.com/Dicklesworthstone/franken_whisper/commit/b1682e8da91985d92d7bfbabbe571526f7d9700e))
- Stdout/stderr pipe deadlock prevented by reading on dedicated threads ([`dd6cbb2`](https://github.com/Dicklesworthstone/franken_whisper/commit/dd6cbb2421c1e3d333df78d791d7d235e17ac143))
- Comprehensive test coverage for all backend engines and routing layer ([`2640d14`](https://github.com/Dicklesworthstone/franken_whisper/commit/2640d1491a270ce413c22b6ea71725312618984f))
- Duration estimation and native engine edge-case tests ([`22c3a03`](https://github.com/Dicklesworthstone/franken_whisper/commit/22c3a036cdca5b6d4cf276c67efdbadb768664e1))

### Robot Mode

- **`robot health` subcommand** for database diagnostics ([`aff4551`](https://github.com/Dicklesworthstone/franken_whisper/commit/aff4551254ad96d43c646e5f9257d46de9d9da45))
- Golden test output fixtures added ([`146d225`](https://github.com/Dicklesworthstone/franken_whisper/commit/146d2256ae7e4d30b475bad157e67a4c28e5fff7))
- Integration tests for robot subcommands ([`0433627`](https://github.com/Dicklesworthstone/franken_whisper/commit/0433627cf11c552ed090c0a20a07c99a1fc3d295))

### Event and Observability

- Event fingerprinting enriched with seq+stage tuples; ingest failure determinism test added ([`acbb240`](https://github.com/Dicklesworthstone/franken_whisper/commit/acbb2406139a8007f0e952c2c253cd4bc7cfd18b))
- `BetaPosterior::new` input validation with assertions and tests ([`e4a898f`](https://github.com/Dicklesworthstone/franken_whisper/commit/e4a898f39e6ccf8cab1e17e3980f587cf22bf0d1))
- Conformance validation and robot event contract extended for speculation events ([`30044ab`](https://github.com/Dicklesworthstone/franken_whisper/commit/30044abd4cdec52cda71be7ea350c5006856560b))

### Sync and Replay

- Sync and `replay_pack` edge-case tests with schema validation ([`8d787f2`](https://github.com/Dicklesworthstone/franken_whisper/commit/8d787f2127921285251c71483236ba0e0735a7c7))

### Testing

- 2,939 tests passing at release (now ~3,668 unit + integration tests at HEAD; see `[Unreleased] → Documentation Refresh` for the marker-vs-runner count correction)
- Unit test coverage expanded across orchestrator, error, model, accelerate, robot, and process modules ([`1ed8429`](https://github.com/Dicklesworthstone/franken_whisper/commit/1ed8429b9e751190566b080cc5b294fcfaa29793))
- Conformance harness with 50ms cross-engine timestamp tolerance and drift detection

### Infrastructure and Tooling

- Boilerplate and `dead_code` annotations eliminated across cli, conformance, and orchestrator ([`e70b860`](https://github.com/Dicklesworthstone/franken_whisper/commit/e70b86082ed279ca6032142192a3d305b905bfa5))
- `rch` quality-gate wrapper added ([`6f5f45e`](https://github.com/Dicklesworthstone/franken_whisper/commit/6f5f45e7ebd3f7010f3fd017729f7f87685824d5))
- MCP agent mail configuration for Codex, Cursor, and Gemini ([`27b1a98`](https://github.com/Dicklesworthstone/franken_whisper/commit/27b1a9822edd4ae9c2b0253ea3e97c5581c4c997))
- GitHub social preview image (1280×640, 24px border) ([`098da6c`](https://github.com/Dicklesworthstone/franken_whisper/commit/098da6c89e85dfbc2dd52009e327be67da3d03be))

### Initial Commit

- [`2e9f2e9`](https://github.com/Dicklesworthstone/franken_whisper/commit/2e9f2e97e0ce8f37d71c2ae00eece2904f3fdd19) — 2026-02-22 — franken_whisper ASR orchestration stack foundation

---

<!-- agent-metadata
  repo: Dicklesworthstone/franken_whisper
  generated: 2026-05-18
  commits_analyzed: 211
  commits_since_v0.1.0: 152
  added_since_prior_generation:
    - speculative_cli_integration (4f54fbd + hardening 2dbab6f)
    - storage_durability_test_tightening (eede00a; fsqlite MVCC fixed upstream)
    - readme_rewrite_and_drift_corrections (9ddb9db + README test-count + stale-MVCC-limit removal)
    - workspace_hygiene (47fe23b clippy + 7855910 Cargo.lock)
  tags: v0.1.0 (lightweight, on 6a51618)
  releases: v0.1.0 (GitHub Release, published 2026-03-04)
  schema_version_at_HEAD: 3
  workstreams_added_since_2026-03-21:
    - release_pipeline (dist.yml + release-automation.yml)
    - crates_io_migration (asupersync/franken-kernel/evidence/decision 0.3.1)
    - conformance_native_rollout_governance (c72eed3, NativeEngineRolloutStage)
    - word_level_timestamps + cancellation_threading (f1cbd31)
    - adaptive_router_calibration (ef80eb8, ed91ddd)
    - metamorphic_test_suites (audio + speculation + accelerate)
    - health_probe_diagnostics_hardening
    - sync_lock_pid_liveness (bd-3jfj/bd-pvol/bd-aijz/bd-7xh8/bd-ht0i/bd-qxf6/bd-uczx)
    - ffmpeg_bundle_stdin_hardening (bd-wk8p/bd-o8fo/bd-g7b2)
    - secrets_redaction_in_logs
    - structured_run_error_for_invalid_robot_requests
-->

[Unreleased]: https://github.com/Dicklesworthstone/franken_whisper/compare/v0.1.0...main
[v0.1.0]: https://github.com/Dicklesworthstone/franken_whisper/releases/tag/v0.1.0
