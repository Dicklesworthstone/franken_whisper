# Feature Parity Tracker

> Tracks the migration status of features from the three legacy projects
> (legacy_whispercpp, legacy_insanely_fast_whisper, legacy_whisper_diarization)
> into the unified franken_whisper Rust codebase.
>
> Status key:
> - **Done** -- Feature is implemented and functional in the Rust codebase.
> - **Partial** -- Feature exists but is incomplete or limited compared to the legacy version.
> - **Todo** -- Feature is planned but not yet implemented.
> - **N/A** -- Feature does not apply to this legacy project or is intentionally excluded.

---

## 1. legacy_whispercpp Features

Features originating from or inspired by the whisper.cpp (C/C++) project.

| Feature | Legacy Source | Rust Module | Status | Notes |
|---|---|---|---|---|
| Basic transcription (whisper_cpp backend) | legacy_whispercpp | `backend::whisper_cpp` | Done | Subprocess bridge to `whisper-cli` with JSON output parsing, segment extraction, and language detection. Configurable via `FRANKEN_WHISPER_WHISPER_CPP_BIN` env var. |
| VAD (Voice Activity Detection) | legacy_whispercpp | `backend::whisper_cpp`, `model::VadParams` | Done | Full Silero VAD parameter passthrough: threshold, min speech/silence duration, max speech duration, speech padding, samples overlap, model path. Forwarded as CLI flags to whisper-cli. |
| GPU acceleration | legacy_whispercpp | `accelerate`, `model::BackendParams` | Done | `--no-gpu` flag, GPU layer offload count. Acceleration module provides confidence normalization via optional frankentorch/frankenjax backends with deterministic CPU fallback. |
| Streaming output (real-time stage events) | legacy_whispercpp | `orchestrator`, `robot` | Done | Real-time NDJSON stage streaming via `transcribe_with_stream()` using mpsc channels. Stage events emitted during pipeline execution. Note: this is pipeline-stage streaming, not audio-chunk streaming like whisper.cpp's `whisper-stream`. |
| Streaming transcription (live audio chunked inference) | legacy_whispercpp | `listen`, `capture`, `robot` | Done | `fw robot listen` runs native rolling-buffer inference over microphone, stdin PCM, or paced file replay and emits append-only NDJSON transcript deltas while speech is still in progress. This dedicated live driver bypasses the batch backend-capability flag; batch engines may still report `supports_streaming: false`. |
| Word-level timestamps | legacy_whispercpp | `backend::whisper_cpp` | Done | Parsed from whisper-cli JSON output into `TranscriptionSegment` with `start_sec`/`end_sec`. Split-on-word boundary mode (`-sow`) supported. |
| Decoding parameters | legacy_whispercpp | `model::DecodingParams`, `backend::whisper_cpp` | Done | Full passthrough: best_of, beam_size, max_context, max_segment_length, temperature, temperature_increment, entropy_threshold, logprob_threshold, no_speech_threshold. |
| Multiple output formats | legacy_whispercpp | `model::OutputFormat`, `backend::whisper_cpp` | Done | Supports txt, vtt, srt, csv, json, json-full, lrc. Artifact paths collected and returned in result. |
| ffmpeg normalization | legacy_whispercpp | `audio` | Done | Primary path converts input to 16kHz mono PCM WAV via ffmpeg subprocess (cancellable with timeout). When ffmpeg is unavailable, file-based normalization falls back to built-in Rust decode/resample for common formats. Duration probing remains ffprobe-based when available. |
| Microphone capture | legacy_whispercpp | `audio`, `model::InputSource::Microphone` | Done | Captures via ffmpeg with platform-specific defaults (ALSA on Linux, AVFoundation on macOS). Configurable device, format, source, and duration. |
| Confidence scoring | legacy_whispercpp | `model::TranscriptionSegment`, `accelerate` | Done | Per-segment confidence field parsed from backend output. Acceleration module normalizes confidence scores (softmax) with optional GPU path and deterministic CPU fallback. |
| Quantized model support | legacy_whispercpp | `backend::whisper_cpp` | Done | Model path passthrough via `-m` flag. Quantized ggml models (Q4/Q5/Q8) usable by providing the model file path. |
| Threading parameters | legacy_whispercpp | `backend::whisper_cpp`, `model::BackendParams` | Done | Thread count (`-t`) and processor count (`-p`) forwarded to whisper-cli. |
| Audio windowing | legacy_whispercpp | `model::BackendParams`, `backend::whisper_cpp` | Done | Offset (`-ot`) and duration (`-d`) parameters, audio context size (`-ac`), and word threshold (`-wt`) supported. |
| Initial prompt / prompt biasing | legacy_whispercpp | `model::BackendParams`, `backend::whisper_cpp` | Done | `--prompt` and `--carry-initial-prompt` flags forwarded. |
| TinyDiarize (speaker turn detection) | legacy_whispercpp | `backend::whisper_cpp`, `model::BackendParams` | Done | `--tdrz` flag forwarded to whisper-cli when `tiny_diarize` is enabled. Activates lightweight speaker-turn token injection. |
| HTTP server mode | legacy_whispercpp | -- | N/A | Intentionally excluded. franken_whisper is a CLI/library, not an HTTP server. |

---

## 2. legacy_insanely_fast_whisper Features

Features originating from or inspired by the insanely-fast-whisper (Python) project.

| Feature | Legacy Source | Rust Module | Status | Notes |
|---|---|---|---|---|
| Basic transcription (insanely-fast-whisper backend) | legacy_insanely_fast_whisper | `backend::insanely_fast` | Done | Subprocess bridge to `insanely-fast-whisper` CLI. JSON output parsing with segment extraction. Configurable via `FRANKEN_WHISPER_INSANELY_FAST_BIN` env var. |
| GPU batching | legacy_insanely_fast_whisper | `backend::insanely_fast`, `model::BackendParams` | Done | Batch size parameter forwarded via `--batch-size`. |
| Flash Attention 2 | legacy_insanely_fast_whisper | `backend::insanely_fast`, `model::BackendParams` | Done | `--flash True` flag forwarded when `flash_attention` is enabled in backend params. |
| Diarization (pyannote integration) | legacy_insanely_fast_whisper | `backend::insanely_fast` | Done | HuggingFace token passthrough (`--hf-token`), explicit hard count forwarding to the subprocess (`--speaker-count-hard K` maps to external `--num-speakers K`), and diarization model selection. Soft `--speaker-count-range` / `--speaker-count-prior` inputs are reserved for the native acoustic overlay and are never hardened into external min/max flags. Token resolved from request params, `FRANKEN_WHISPER_HF_TOKEN`, or `HF_TOKEN`. |
| Automatic device selection | legacy_insanely_fast_whisper | `backend::insanely_fast`, `model::BackendParams` | Done | `--device-id` forwarded from `gpu_device` param. |
| Timestamp granularity modes | legacy_insanely_fast_whisper | `backend::insanely_fast`, `model::TimestampLevel` | Done | Chunk and word timestamp levels forwarded via `--timestamp` flag. |
| Transcript path override | legacy_insanely_fast_whisper | `model::BackendParams` | Done | `--transcript-path` override for insanely-fast output artifact location. |
| Ergonomic CLI defaults | legacy_insanely_fast_whisper | `cli` | Done | Sensible defaults for backend (auto), model, and task. Minimal required args for common use. |
| Model selection | legacy_insanely_fast_whisper | `backend::insanely_fast` | Done | `--model-name` forwarded from request model parameter. |

---

## 3. legacy_whisper_diarization Features

Features originating from or inspired by the whisper-diarization (Python) project.

| Feature | Legacy Source | Rust Module | Status | Notes |
|---|---|---|---|---|
| Diarization (whisper-diarization backend) | legacy_whisper_diarization | `backend::whisper_diarization` | Done | Subprocess bridge to `diarize.py` script. Parses SRT output for speaker-labeled segments with timestamps. |
| Source separation (Demucs vocal isolation) | legacy_whisper_diarization | `backend::whisper_diarization` | Partial | Supported via `--no-stem` toggle in `DiarizationConfig`. The Demucs stage runs inside the legacy Python script; no native Rust source separation exists. |
| Forced alignment (CTC) | legacy_whisper_diarization | `backend::whisper_diarization` | Partial | CTC forced alignment runs inside the legacy Python pipeline via `diarize.py`. `--suppress_numerals` toggle forwarded. No native Rust CTC aligner implementation. |
| Punctuation restoration | legacy_whisper_diarization | `backend::whisper_diarization` | Partial | Multilingual punctuation restoration runs inside the legacy Python pipeline. No native Rust punctuation restoration model. |
| Speaker embedding extraction (NeMo TitaNet) | legacy_whisper_diarization | `backend::whisper_diarization`, `ecapa_conformance`, `ecapa_inference`, `diarization`, `orchestrator` | Partial | The legacy bridge still runs TitaNet. Explicit native `neural` routing now runs the pinned ECAPA package through a safe-Rust PCM frontend and forward pass, then the common constraints/count/temporal/projection stack. It remains Partial because public-corpus accuracy/calibration has not passed promotion gates and `auto` does not select it. |
| SRT output parsing | legacy_whisper_diarization | `backend::whisper_diarization` | Done | Hardened SRT parser handles timestamp extraction and speaker label recognition from diarization output. |
| Device and batch size configuration | legacy_whisper_diarization | `model::DiarizationConfig` | Done | `--device`, `--batch-size`, and `--whisper-model` forwarded. Env fallback via `FRANKEN_WHISPER_DIARIZATION_DEVICE`. |
| Suppress numerals option | legacy_whisper_diarization | `model::DiarizationConfig` | Done | `--suppress_numerals` flag forwarded to improve CTC alignment stability. |
| Parallel processing mode | legacy_whisper_diarization | -- | N/A | The legacy `diarize_parallel.py` variant is not separately bridged. Parallelism is managed at the orchestrator level. |

---

## 4. Cross-Cutting / Novel franken_whisper Features

Features that span all legacy projects or are new to franken_whisper.

| Feature | Legacy Source | Rust Module | Status | Notes |
|---|---|---|---|---|
| CLI interface | All three | `cli`, `main` | Done | clap-based CLI with subcommands: `transcribe`, `robot run/schema/backends/health/triage/routing-history/listen`, `youtube`, `runs`, `sync export-jsonl/import-jsonl`, `tty-audio encode/decode/retransmit-plan`, `tui`. |
| Robot mode (NDJSON output) | None (novel) | `robot` | Done | Structured NDJSON line-oriented output with schema version 1.1.0. Coverage includes lifecycle events (`run_start`, `stage`, `run_complete`, `run_error`), discovery/diagnostics (`backends.discovery`, `health.report`), routing-history output (`routing_decision` via `robot routing-history`), speculative streaming events (`transcript.partial`, `transcript.confirm`, `transcript.retract`, `transcript.correct`, `transcript.speculation_stats`), and the additive live-session family (`listen.session_start`, `speech_started`, `transcript.delta`, `utterance_end`, `listen.warning`, `listen.session_stats`) plus default-off adaptive `listen.controller` decisions. Required fields are enforced for the core live family; controller decisions carry the shared envelope plus controller state, action, calibration, and fallback evidence. |
| TUI mode | None (novel) | `tui` | Done | frankentui-based terminal UI with runs/timeline/events panes. Focus cycling, keyboard navigation, auto-refresh. Feature-gated behind `--features tui`. |
| SQLite persistence | None (novel) | `storage` | Done | frankensqlite-backed `RunStore` with runs, segments, and events tables. Schema initialization, run persistence, query by ID, recent runs listing. |
| JSONL export/import sync | None (novel) | `sync` | Done | One-way atomic sync with file locking (stale lock detection, archive). Manifest with schema version, row counts, SHA-256 checksums. Conflict policies: reject, skip, overwrite. Export format version 1.0. |
| Confidence scoring / normalization | None (novel) | `accelerate` | Done | Softmax confidence normalization with optional GPU acceleration (frankentorch/frankenjax feature gates). Pre/post probability mass tracking. Deterministic CPU fallback guaranteed. |
| Schema versioning | None (novel) | `robot`, `sync` | Done | Robot schema version 1.1.0 (additive listen-family bump from 1.0.0). Sync schema version 1.1 with export format version 1.0. Schema versions embedded in all NDJSON envelopes and JSONL manifests. |
| Error codes | None (novel) | `error` | Done | Two-tier error code system: per-variant codes (FW-IO, FW-JSON, FW-CMD-MISSING, FW-CMD-FAILED, FW-CMD-TIMEOUT, FW-BACKEND-UNAVAILABLE, FW-INVALID-REQUEST, FW-STORAGE, FW-UNSUPPORTED, FW-MISSING-ARTIFACT, FW-CANCELLED, FW-STAGE-TIMEOUT) and grouped robot codes (FW-ROBOT-TIMEOUT, FW-ROBOT-BACKEND, FW-ROBOT-REQUEST, FW-ROBOT-STORAGE, FW-ROBOT-CANCELLED, FW-ROBOT-EXEC). |
| Backend health probing | None (novel) | `backend` | Done | `probe_system_health()` with per-backend reports: binary discovery, version probing, availability status, RFC3339 timestamps. Thread-safe cached health state with TTL. Exposed through robot diagnostics surfaces including `robot backends` (`backends.discovery`) and `robot health` (`health.report`). |
| TTY audio codec (mu-law + zlib + base64) | None (novel) | `tty_audio` | Done | NDJSON-framed audio transport. Encode/decode with mu-law compression, zlib deflation, base64 encoding. Per-frame CRC32 and SHA-256 integrity. Protocol version negotiation (handshake/ack). Control frames: retransmit_request, ack, backpressure. Recovery modes for missing/corrupt frames. |
| Real-time streaming driver | None (novel) | `listen`, `capture`, `robot` | Done | `fw robot listen`: cpal/ffmpeg/stdin-pcm capture -> causal VAD -> bounded rolling buffer with prompt carry -> greedy step decode -> emission policy (AlignAtt default, endpoint-commit baseline, LocalAgreement-2 fallback) -> append-only NDJSON deltas + mutable partials, async quality-confirm lane, utterance-granular crash-durable SQLite persistence. Additive schema 1.1.0 family; see `docs/realtime-streaming.md`. |
| Differential kernel oracle | None (novel) | `native_engine`, `tests/conformance_oracle_tests.rs` | Done | Feature-gated (`fj-oracle`) cross-checks of matmul/softmax/layer-norm/conv1d against the frankenjax `fj-lax` interpreter on seeded random tensors, tolerance 1e-4 max-abs; divergences dump full JSON tensor artifacts. |
| Replay packs | None (novel) | `replay_pack` | Done | Self-contained replay artifact directory: `env.json` (runtime snapshot), `manifest.json` (file inventory with content hashes), `repro.lock` (frozen decision state with routing evidence). |
| Conformance checking | None (novel) | `conformance` | Done | Segment invariant validation (overlap detection, timestamp epsilon). Segment compatibility comparison with configurable tolerance (timestamp, text, speaker). Replay envelope comparison (input hash, backend identity/version, output hash). |
| Structured logging | None (novel) | `logging` | Done | tracing-subscriber with `RUST_LOG` env filter. JSON output mode via `RUST_LOG_FORMAT=json`. Default level INFO. Output to stderr. |
| Adaptive backend routing | None (novel) | `backend` | Done | Decision-contract-based auto backend selection. Loss matrix, posterior/confidence terms, calibration scoring. Routing policy versioning. Evidence ledger with per-run routing history. Static deterministic fallback trigger. |
| Cancellation / orchestration | None (novel) | `orchestrator` | Done | Budget-aware pipeline with deadline checkpoints. Cancellation tokens for subprocess abort. Trace ID generation. Evidence recording. Integration with asupersync runtime. |
| Replay envelope / drift detection | None (novel) | `model::ReplayEnvelope`, `conformance` | Done | SHA-256 hashes for input content and output payload. Backend identity and version capture. Comparison API for regression drift detection across engine versions. |
| StreamingEngine trait | None (novel) | `backend` | Done | Extension trait on `Engine` for real-time incremental segment delivery via callback. Default implementation delegates to `Engine::run` and replays segments, providing backward compatibility. `WhisperCppEngine` implements it natively. Enables live TUI and streaming robot mode output. |
| Brier-score calibration | None (novel) | `backend::CalibrationState` | Done | Sliding-window Brier-score tracker for adaptive routing confidence calibration. Records predicted success probabilities vs. actual binary outcomes. Brier threshold (0.35) triggers fallback to static routing when miscalibration detected. Evidence JSON snapshot for diagnostics. |
| Routing evidence ledger | None (novel) | `backend::RoutingEvidenceLedger` | Done | Circular buffer (capacity 200) of `RoutingEvidenceLedgerEntry` records. Each entry captures decision ID, trace ID, observed state, chosen action, posterior snapshot, calibration/Brier scores, e-process value, fallback status, and actual outcome once resolved. Diagnostic aggregation (fallback rate, success rate, avg Brier). Satisfies Alien-Artifact Engineering Contract requirements. |
| Composable pipeline stages | None (novel) | `orchestrator::PipelineConfig` | Done | `PipelineStage` enum with 10 stages: Ingest, Normalize, Vad, Separate, Backend, Accelerate, Align, Punctuate, Diarize, Persist. `PipelineBuilder` for ergonomic construction with `stage()`, `without()`, `build()`. Validation enforces dependency ordering (Ingest before Normalize, Normalize before Backend, Backend before Accelerate/Align). |
| Unified output normalization | None (novel) | `backend::normalize` | Done | Centralized `NormalizedOutput` intermediate representation. Per-backend normalizers: `normalize_whisper_cpp`, `normalize_insanely_fast`, `normalize_whisper_diarization`. Each validates JSON structure, extracts segments via shared helper, falls back to `transcript_from_segments`. `to_transcription_result` converts to final form. |
| Health report infrastructure | None (novel) | `robot` | Done | `HealthReport` struct with dependency checks (backends, ffmpeg, database), resource snapshot (disk/memory), and overall status (ok/degraded/unavailable). `build_health_report()` probes all subsystems. `emit_health_report()` emits NDJSON `health.report` event. Includes `/proc/meminfo` parsing on Linux. |
| Graceful Ctrl+C shutdown | None (novel) | `cli::ShutdownController` | Done | Global `AtomicBool` flag set by `ctrlc` handler. Optional callback for cancellation token integration. `signal_exit_code()` returns 130 (128 + SIGINT). Programmatic trigger and reset for testing. |
| Stage latency profiling | None (novel) | `orchestrator` | Done | `stage_latency_profile()` computes per-stage queue/service/external latency from pipeline events. Budget utilization analysis with tuning recommendations (increase/decrease/keep). Artifact format: `stage_latency_decomposition_v1`. |
| Live transcription view (TUI) | None (novel) | `tui::LiveTranscriptionView` | Done | Real-time segment display component with auto-scroll, speaker labels, timestamps, elapsed time, backend status bar. Segment retention limit (10,000 default) with oldest-first drain. Scroll-up disables auto-scroll; scroll-to-bottom re-enables. Designed for embedding in larger TUI layouts. |
| Word-level timestamp params | None (novel) | `model::WordTimestampParams` | Done | Configurable word-level timestamp extraction for whisper.cpp: enabled flag, max_len, token_threshold, token_sum_threshold. |
| Insanely-fast tuning params | None (novel) | `model::InsanelyFastTuningParams` | Done | Extended tuning for insanely-fast-whisper: device_map strategy (auto/sequential), torch_dtype selection, disable_better_transformer flag. |
| Alignment configuration | None (novel) | `model::AlignmentConfig` | Done | Forced alignment parameters for whisper-diarization: alignment_model, interpolate_method, return_char_alignments. |
| Punctuation configuration | None (novel) | `model::PunctuationConfig` | Done | Punctuation restoration parameters: model selection, enabled toggle. |
| Source separation configuration | None (novel) | `model::SourceSeparationConfig` | Done | Demucs parameters: enabled toggle, model name, shifts for test-time augmentation, overlap factor. |
| franken_evidence integration | None (novel) | `orchestrator` | Done | Integration with `franken_evidence::EvidenceLedger` crate for structured evidence recording in pipeline context. `record_evidence()` and `record_evidence_values()` on `PipelineCx`. |
| franken_decision integration | None (novel) | `backend` | Done | Integration with `franken_decision` crate for decision-contract evaluation. `DecisionContract`, `EvalContext`, `FallbackPolicy`, `LossMatrix`, `Posterior` types used in adaptive routing. |

---

## Summary

### Per-Legacy-Project Completion

| Legacy Project | Done | Partial | Todo | N/A | Total | Completion % |
|---|---|---|---|---|---|---|
| legacy_whispercpp | 15 | 0 | 0 | 1 | 16 | 94% |
| legacy_insanely_fast_whisper | 9 | 0 | 0 | 0 | 9 | 100% |
| legacy_whisper_diarization | 4 | 4 | 0 | 1 | 9 | 44% (89% via bridge) |

### Cross-Cutting / Novel Features

| Status | Count |
|---|---|
| Done | 30 |
| Partial | 0 |
| Todo | 0 |
| N/A | 0 |
| **Total** | **30** |

### Overall Completion

| Category | Done | Partial | Todo | N/A | Total Tracked |
|---|---|---|---|---|---|
| All features | 58 | 4 | 0 | 2 | 64 |
| **Completion (Done + Partial)** | | | | | **97%** |
| **Completion (Done only)** | | | | | **91%** |

### Key Gaps Remaining

1. **Remaining native diarization stages**: The classical Rust acoustic
   diarizer and explicit safe-Rust ECAPA execution through the common diarizer
   are operational. Neural accuracy/calibration and automatic rollout remain
   uncertified. Source separation, forced alignment, and punctuation
   restoration still delegate to the legacy Python environment.

---

*This document is part of the spec-first porting workflow defined in AGENTS.md.*
