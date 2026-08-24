# PROPOSED_ARCHITECTURE.md

## 1. Architectural Intent

`franken_whisper` is a synthesis architecture, not a straight port. It combines:
- whisper.cpp's local performance + stream ergonomics,
- insanely-fast-whisper's GPU batch UX,
- whisper-diarization's alignment/speaker pipeline,

and integrates with:
- `asupersync` for cancel-correct orchestration and bounded cleanup,
- `frankensqlite` (`fsqlite`) for all durable state,
- `frankentorch` for required native compute kernels and optional `frankenjax`
  as an independent differential-test oracle,
- optional `frankentui` for human operator UX.

## 2. Top-Level Components

1. `ingest` (`src/audio.rs`):
- input acquisition from file, stdin, mic/line-in.
- ffmpeg normalization to canonical WAV (`16kHz`, mono, PCM16).
- cancellation-aware via `CancellationToken` checkpoint before and during normalization.

2. `orchestrator` (`src/orchestrator.rs`):
- execution control and cancellation model via `PipelineCx`.
- selects backend strategy and stage ordering.
- emits structured stage events for robot mode and telemetry through `EventLog`.
- manages resource lifecycle through `FinalizerRegistry`.
- drives the asupersync `Runtime` for async stage execution with per-stage budgets.

3. `backend engines` (`src/backend/mod.rs`, `src/backend/*.rs`):
- narrow engine contract via `Engine` trait: normalized audio + hints -> unified segment/event output.
- `StreamingEngine` extension trait for real-time incremental segment delivery.
- current implementation includes bridge adapters for `whisper_cpp`, `insanely_fast`, and `whisper_diarization`.
- target state is full native Rust engines (`WhisperEngineCpu`, `WhisperEngineGpu`, `DiarizationEngine`) sharing one contract.
- native execution through required FrankenTorch CPU kernels, with target-gated
  Metal selection on supported macOS operations; the optional `fj-oracle`
  feature is conformance-only and never a production compute backend.
- all engine execution paths accept `CancellationToken` for deadline-aware abort.

4. `native acoustic diarization` (`src/diarization.rs`):
- memory-safe waveform analysis at 16 kHz with a 25 ms frame / 10 ms hop.
- independent voice and channel features, nullable pitch, and multiscale
  Haar-like feature-trajectory contrasts.
- bounded streaming change detection, robust hard/soft known-interval
  enrollment, deterministic constrained clustering, temporal refinement,
  unknown rejection, and overlap suspicion.
- an independent turn timeline projected onto unchanged ASR text at legal DTW
  word boundaries.
- no gender or identity inference; no raw waveform, feature vector, or
  biometric template persistence.

5. `postprocess + acceleration` (`src/accelerate.rs`, `src/backend/normalize.rs`):
- unified output normalization via `NormalizedOutput` in `backend/normalize.rs` converts each backend's native JSON into a common intermediate representation.
- per-backend normalizers: `normalize_whisper_cpp`, `normalize_insanely_fast`, `normalize_whisper_diarization`.
- `to_transcription_result` converts the normalized intermediate form into `TranscriptionResult`.
- run confidence normalization acceleration pass with deterministic CPU fallback.
- acceleration pass is cancellation-aware via `apply_with_token`.

6. `persistence` (`src/storage.rs`, `frankensqlite` only):
- run metadata, segments, events, and artifacts index in `fsqlite` DB via `RunStore`.
- schema versioning through `_meta` table with forward migration system.
- current schema version: 4 (v1 = base tables, v2 = legacy-safe `runs`
  rebuild for `replay_json` + `acceleration_json`, v3 = query indexes,
  v4 = normalized privacy-safe diarization reports, turns, hint audit rows,
  and speaker-profile summaries).
- `persist_report_with_token` accepts `CancellationToken` for deadline-aware persistence.
- JSONL sync/export as adjunct audit stream via `src/sync.rs`.

7. `interfaces`:
- library API (`franken_whisper` crate, `src/lib.rs`).
- CLI robot mode (real-time NDJSON stage events + final envelope) via `src/robot.rs`.
- CLI health report infrastructure via `robot.rs`: `HealthReport`, `build_health_report()`, `emit_health_report()` with dependency checks, resource snapshots, and overall status.
- optional TUI mode via `frankentui` feature (`src/tui.rs`) with runs/timeline/events panes plus `LiveTranscriptionView` for real-time segment display.
- TTY/PTY low-bandwidth audio relay via `src/tty_audio.rs`.
- graceful Ctrl+C shutdown via `cli::ShutdownController` with `AtomicBool` flag and optional callback integration.

## 3. Pipeline Stages

### 3.0 Composable Pipeline (`PipelineConfig`)

The pipeline is now defined as a composable sequence of discrete `PipelineStage` values,
managed via `PipelineConfig` and constructed with the `PipelineBuilder` fluent API.

```text
Ingest -> Normalize -> Vad -> Separate -> Backend -> Accelerate -> Align -> Punctuate -> Diarize -> Persist
```

The 10 canonical stages are defined by the `PipelineStage` enum:

| Stage       | Label         | Purpose                                                     |
|-------------|--------------|--------------------------------------------------------------|
| `Ingest`    | `ingest`     | Materialize input from file/stdin/URL into a local temp file |
| `Normalize` | `normalize`  | Normalize audio to 16 kHz mono WAV via ffmpeg                |
| `Vad`       | `vad`        | Voice activity detection pre-filtering                       |
| `Separate`  | `separate`   | Source separation / vocal isolation (Demucs-inspired)        |
| `Backend`   | `backend`    | Execute the transcription backend engine                     |
| `Accelerate`| `acceleration`| GPU confidence normalization pass                           |
| `Align`     | `align`      | CTC-based forced alignment for timestamp correction          |
| `Punctuate` | `punctuate`  | Punctuation restoration post-processing                      |
| `Diarize`   | `diarize`    | Typed acoustic/external diarization, provenance gating, and transcript projection |
| `Persist`   | `persist`    | Persist the run report to frankensqlite                      |

**Dependency constraints** enforced by `PipelineConfig::validate()`:
- `Normalize` requires `Ingest` before it.
- `Backend` requires `Normalize` before it.
- `Accelerate` requires `Backend` before it.
- `Align` requires `Backend` before it.
- No duplicate stages allowed.

**Builder pattern**:
```rust
let config = PipelineBuilder::new()
    .stage(PipelineStage::Ingest)
    .stage(PipelineStage::Normalize)
    .stage(PipelineStage::Backend)
    .stage(PipelineStage::Persist)  // skip Vad, Separate, Accelerate, Align, Punctuate, Diarize
    .build()?;

// Or start from defaults and remove stages:
let config = PipelineBuilder::default_stages()
    .without(PipelineStage::Vad)
    .without(PipelineStage::Separate)
    .build()?;
```

The orchestrator drives these stages through `run_pipeline` / `run_pipeline_body`, which
executes as an async task on the asupersync `Runtime`. Each stage runs inside
`run_stage_with_budget`, which spawns a blocking closure on the runtime's blocking thread
pool and wraps it with an `asupersync::time::timeout` envelope.

Stage contracts:
- each stage has explicit start/end events emitted via `EventLog::push`.
- each stage has a configurable budget (env-var overridable via `FRANKEN_WHISPER_STAGE_BUDGET_*_MS`).
- `checkpoint_or_emit` is called between stages to enforce pipeline-level deadlines.
- stage errors include machine-readable code + context (`{stage}.timeout`, `{stage}.cancelled`, `{stage}.error`).
- streamed stage events and persisted events share the same monotonic sequence.
- backend output is validated against the `segment-monotonic-v1` conformance contract before the pipeline accepts it.

### 3.1 Default Stage Budgets

| Budget Key                 | Default (ms) | Env Override                                      | Notes |
|----------------------------|-------------:|--------------------------------------------------|-------|
| Ingest                     | 15,000       | `FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS`         | Pipeline stage |
| Normalize                  | 180,000      | `FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS`      | Pipeline stage |
| Vad                        | 10,000       | `FRANKEN_WHISPER_STAGE_BUDGET_VAD_MS`            | Pipeline stage |
| Separate                   | 30,000       | `FRANKEN_WHISPER_STAGE_BUDGET_SEPARATE_MS`       | Pipeline stage |
| Backend                    | 900,000      | `FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS`        | Pipeline stage |
| Acceleration               | 20,000       | `FRANKEN_WHISPER_STAGE_BUDGET_ACCELERATION_MS`   | Pipeline stage |
| Align                      | 30,000       | `FRANKEN_WHISPER_STAGE_BUDGET_ALIGN_MS`          | Pipeline stage |
| Punctuate                  | 10,000       | `FRANKEN_WHISPER_STAGE_BUDGET_PUNCTUATE_MS`      | Pipeline stage |
| Diarize                    | 30,000       | `FRANKEN_WHISPER_STAGE_BUDGET_DIARIZE_MS`        | Pipeline stage |
| Persist                    | 20,000       | `FRANKEN_WHISPER_STAGE_BUDGET_PERSIST_MS`        | Pipeline stage |
| Probe policy budget        | 8,000        | `FRANKEN_WHISPER_STAGE_BUDGET_PROBE_MS`          | Policy/diagnostic budget knob (not a `PipelineStage`) |
| Finalizer cleanup budget   | 5,000        | `FRANKEN_WHISPER_STAGE_BUDGET_CLEANUP_MS`        | Bounded finalizer execution budget |

## 3.2 Compatibility Envelope + Conformance Harness

Compatibility for native-engine parity must be explicit:
- text/transcript parity envelope (exact-match where deterministic, tolerance band where stochastic),
- timestamp tolerance bounds (per segment and aggregate drift),
- speaker-label stability expectations,
- confidence calibration comparability targets (not bitwise equality).

Harness requirements:
- golden audio corpus with expected run envelopes,
- invariant checks (monotonic timestamps, non-overlap unless policy allows overlap, stable event ordering),
- replay determinism checks at contract level (stage codes/order/schema), not floating-point bitwise identity.

Current enforcement:
- runtime segment contract validation (`segment-monotonic-v1`) fails closed on invariant violations.
- `conformance::validate_segment_invariants` is invoked on every backend result before the pipeline proceeds.

## 4. Data Model (Core)

### `TranscribeRequest`
- input source descriptor (`InputSource::File`, `InputSource::Stdin`, `InputSource::Microphone`)
- backend preference (`auto`, `whisper_cpp`, `insanely_fast`, `whisper_diarization`)
- model/language/task hints
- diarization toggle
- persistence toggle
- timeout_ms (optional pipeline deadline)
- `BackendParams` (GPU device, flash attention, VAD/decoding params, word timestamp params, insanely-fast tuning, alignment/punctuation/source-separation config)

### `TranscriptionResult`
- backend identity + version metadata
- canonical transcript text
- per-segment records (`start_sec`, `end_sec`, `text`, `speaker?`, `confidence?`)
- acceleration metadata (`backend`, normalized flag, notes, mass summary)
- raw backend payload (JSON) for forensic replay

### `RunEvent`
- monotonic sequence (`seq: u64`)
- RFC 3339 timestamp
- stage / code / message
- payload JSON (includes `trace_id` and `elapsed_ms` when stage timing is active)

### `StreamedRunEvent`
- `run_id: String`
- `event: RunEvent`
- used for real-time event delivery from pipeline to robot mode CLI via `mpsc::channel`

### `RunReport`
- full run envelope: run_id, trace_id, timestamps, request, result, events, warnings, evidence
- `ReplayEnvelope` with content hashes (normalized input SHA-256, backend output SHA-256, engine identity/version)

## 5. Asupersync Deep Integration

### PipelineCx and CancellationToken

The orchestrator creates a `PipelineCx` at run start, which encapsulates:
- `TraceId` from `franken_kernel` (timestamp + random, unique per run),
- `Budget` from `franken_kernel` for remaining-time tracking,
- optional deadline (`chrono::DateTime<Utc>`) for absolute timeout enforcement,
- evidence ledger (`Vec<Value>`) for decision-contract artifacts, with `record_evidence()` (from `franken_evidence::EvidenceLedger`) and `record_evidence_values()` methods,
- `FinalizerRegistry` for resource cleanup.

`PipelineCx::cancellation_token()` produces a lightweight `CancellationToken` (`Send + Sync + Clone + Copy`) that is threaded into every pipeline stage:

```
PipelineCx
  |
  +-- cancellation_token() -> CancellationToken
  |     |
  |     +-- audio::materialize_input_with_token (ingest stage)
  |     +-- audio::normalize_to_wav_with_timeout (normalize stage)
  |     +-- backend::execute / execute_with_order (backend stage)
  |     +-- accelerate::apply_with_token (acceleration stage)
  |     +-- storage::RunStore::persist_report_with_token (persist stage)
  |
  +-- checkpoint() -> FwResult<()>  (called between stages)
  +-- cancellation_evidence() -> Value  (emitted on cancellation)
```

The `CancellationToken::checkpoint()` method checks the deadline against the current wall clock and returns `Err(FwError::Cancelled(_))` if the deadline has passed. Each stage calls `checkpoint()` at entry and at suitable internal points.

### Async Runtime

`FrankenWhisperEngine` owns an `asupersync::runtime::Runtime` configured with 2 worker threads and 1-4 blocking threads. The pipeline is spawned as an async task. Individual stages use `asupersync::runtime::spawn_blocking` to execute CPU-bound or subprocess work without blocking the async runtime, and `asupersync::time::timeout` to enforce per-stage budgets.

### Checkpoint Protocol

Between every stage transition, `checkpoint_or_emit` is called. On cancellation, it:
1. Records cancellation evidence (stage, timestamp, deadline, budget remaining, overdue_ms).
2. Emits a stage event with the cancellation evidence attached.
3. Returns `Err(FwError::Cancelled(_))` to unwind the pipeline.

The caller (`run_pipeline`) unconditionally runs `pcx.run_finalizers()` after `run_pipeline_body` returns, ensuring cleanup regardless of success or failure.

## 5.1 FinalizerRegistry

The `FinalizerRegistry` provides LIFO resource cleanup. Finalizers are registered during pipeline execution and executed in reverse order at pipeline shutdown.

Finalizer variants:
- `TempDir(PathBuf)`: records temp directory for cleanup (actual removal via `tempfile::TempDir::Drop`).
- `Process(u32)`: sends `kill -9` to a subprocess PID (best-effort, silently succeeds if process already exited).
- `Custom(Box<dyn FnOnce() + Send>)`: arbitrary cleanup closure.

The registry is always drained via `run_all()` at pipeline exit, even on error paths.

## 6. Backend Engine Contract

### Engine Trait

```rust
pub trait Engine: Send + Sync {
    fn name(&self) -> &'static str;
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> EngineCapabilities;
    fn is_available(&self) -> bool;
    fn run(&self, request, normalized_wav, work_dir, timeout) -> FwResult<TranscriptionResult>;
}
```

Each backend adapter (`WhisperCppEngine`, `InsanelyFastEngine`, `WhisperDiarizationEngine`) implements this trait. `EngineCapabilities` declares support for diarization, translation, word timestamps, GPU, and streaming.

### StreamingEngine Trait

```rust
pub trait StreamingEngine: Engine {
    fn run_streaming(
        &self,
        request, normalized_wav, work_dir, timeout,
        on_segment: Box<dyn Fn(TranscriptionSegment) + Send>,
    ) -> FwResult<TranscriptionResult>;
}
```

The `StreamingEngine` trait extends `Engine` with a `run_streaming` method that delivers `TranscriptionSegment` values incrementally via a callback. The default implementation delegates to `Engine::run` and replays segments through the callback, providing backward compatibility: any `Engine` implementor can opt into `StreamingEngine` without additional work.

This enables real-time segment delivery for TUI display and streaming robot mode output.

### Backend Selection Policy

Default auto policy:
- if diarization requested: prefer `insanely_fast` (if available + token), then `whisper_diarization`, then `whisper_cpp`.
- else: prefer `whisper_cpp`, then `insanely_fast`, then `whisper_diarization`.

Adaptive decision-contract policy (current):
- each auto run computes an evidence ledger card via `franken_decision` with:
  - explicit state space (per-backend availability and empirical metrics),
  - explicit action candidates (available backends),
  - explicit loss matrix terms,
  - posterior success terms,
  - calibration metric (`posterior_margin`),
  - deterministic fallback trigger.
- routing policy ID: `backend-selection-v1.0`, evidence schema version: `1.0`.
- evidence is emitted as `backend.routing.decision_contract` and captured in run artifacts.
- calibration guardrail: if `calibration_score < min_calibration` (default 0.3), the adaptive order is discarded in favor of static priority.
- safe mode: `FRANKEN_WHISPER_ROUTING_SAFE_MODE=1` forces static priority ordering.
- deterministic fallback trigger status is always included in emitted routing evidence.

**Brier-score confidence calibration** (`CalibrationState`):
- Sliding window (size matches `ROUTER_HISTORY_WINDOW = 50`) of `CalibrationObservation` records.
- Each observation pairs the predicted success probability with the actual binary outcome.
- Brier score: `BS = (1/N) * sum(predicted_i - actual_i)^2`. BS=0.0 is perfect; BS=1.0 is worst.
- When `brier_score > 0.35` (`ADAPTIVE_FALLBACK_BRIER_THRESHOLD`), the router falls back to static priority.
- Minimum sample threshold: `ADAPTIVE_MIN_SAMPLES = 5` outcomes required before trusting empirical estimates.

**Routing evidence ledger** (`RoutingEvidenceLedger`):
- Circular buffer with capacity 200 (`EVIDENCE_LEDGER_CAPACITY`).
- Each `RoutingEvidenceLedgerEntry` records: decision_id, trace_id, timestamp, observed_state, chosen_action, recommended_order, fallback_active/reason, posterior_snapshot, calibration_score, brier_score, e_process, ci_width, adaptive_mode, policy_id, loss_matrix_hash, per-backend availability, duration_bucket, diarize flag.
- Outcome resolution: `resolve_outcome()` links actual results (success/failure, latency) to prior decisions.
- Diagnostic aggregation via `diagnostics()`: total entries, fallback rate, resolved success rate, avg calibration/Brier scores.
- Exposed via `robot routing-history` CLI command for post-hoc analysis.

## 7. Unified Output Normalization (`backend/normalize.rs`)

All backend outputs pass through a centralized normalization layer before entering the pipeline's postprocessing stages.

### NormalizedOutput

```rust
pub struct NormalizedOutput {
    pub transcript: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub raw_output: Value,
}
```

### Per-Backend Normalizers

| Backend              | Function                         | Key extraction paths                           |
|---------------------|----------------------------------|-------------------------------------------------|
| whisper.cpp         | `normalize_whisper_cpp`          | `.text`, `.segments[]`, `.result.language`       |
| insanely-fast       | `normalize_insanely_fast`        | `.text`, `.chunks[].timestamp`, `.language`      |
| whisper-diarization | `normalize_whisper_diarization`  | `.transcript_txt`, segments from SRT parse       |

Each normalizer:
1. Validates the root JSON is an object.
2. Extracts segments via the shared `extract_segments_from_json` helper.
3. Falls back to `transcript_from_segments` if the primary text field is empty.
4. Preserves the raw output verbatim for forensic replay.

`to_transcription_result` converts a `NormalizedOutput` into a `TranscriptionResult` with acceleration and artifact fields defaulted (filled in by later pipeline stages).

## 7.1 Expanded Model-Layer Configuration (`src/model.rs`)

The data model provides typed configuration structs for the new pipeline stages, allowing fine-grained control over each processing step:

| Config Struct            | Purpose                                              | Key Fields                                                 |
|-------------------------|------------------------------------------------------|------------------------------------------------------------|
| `WordTimestampParams`   | Word-level timestamp extraction (whisper.cpp)        | `enabled`, `max_len`, `token_threshold`, `token_sum_threshold` |
| `InsanelyFastTuningParams` | Extended insanely-fast-whisper tuning             | `device_map` (auto/sequential), `torch_dtype`, `disable_better_transformer` |
| `AlignmentConfig`       | Forced alignment (CTC) for whisper-diarization       | `alignment_model`, `interpolate_method`, `return_char_alignments` |
| `PunctuationConfig`     | Punctuation restoration post-processing              | `model`, `enabled`                                         |
| `SourceSeparationConfig`| Demucs source separation                             | `enabled`, `model`, `shifts`, `overlap`                    |

These are aggregated into `BackendParams` and selectively consumed by each backend engine.

## 7.2 Stage Latency Profiling (`orchestrator`)

The orchestrator provides stage-level latency decomposition via `stage_latency_profile()`:

- Computes per-stage queue time, service time, and external process time from pipeline events.
- Calculates budget utilization ratio per stage.
- Generates tuning recommendations: `increase_budget` (utilization >= 90%), `decrease_budget_candidate` (utilization <= 30%), or `keep_budget`.
- Artifact format: `stage_latency_decomposition_v1` with RFC 3339 timestamp.
- Summary includes total queue/service/external time and observed stage count.

## 8. Persistence Contract (`fsqlite` Only)

### Schema Versioning and Migration

The `RunStore` maintains a versioned schema tracked in the `_meta` table:

```sql
CREATE TABLE IF NOT EXISTS _meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Current schema version: **4** (`RunStore::SCHEMA_VERSION`).

Migration system:
- `initialize_schema()` creates base tables (runs, segments, events, _meta) then calls `run_migrations()`.
- `run_migrations()` reads `current_schema_version()` from `_meta` and applies forward migrations in order.
- Each migration runs inside a transaction or savepoint and rolls back on
  failure. Legacy table rebuilds copy canonical values deterministically before
  the old structure is replaced.
- If the DB schema version exceeds `SCHEMA_VERSION`, the store refuses to open with an explicit upgrade error.

Migration history:
| Version | Description                                             |
|---------|---------------------------------------------------------|
| 1       | Base schema (runs, segments, events, _meta tables)     |
| 2       | Add `acceleration_json` and `replay_json` columns to runs |
| 3       | Add hot-path indexes for runs and events |
| 4       | Add normalized diarization reports, turns, hint audit rows, and privacy-safe speaker-profile summaries |

### Source of Truth Tables

- `runs`: id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json.
- `segments`: run_id, idx, start_sec, end_sec, speaker, text, confidence (PK: run_id + idx).
- `events`: run_id, seq, ts_rfc3339, stage, code, message, payload_json (PK: run_id + seq).
- `diarization_reports`: one implementation/contract/fallback summary per run.
- `diarization_turns`: ordered independent speaker turns per run.
- `speaker_hints`: canonical known-interval audit rows; the CLI
  `--speaker-hints` source path is never stored.
- `speaker_profile_summaries`: reliability, duration, anchoring, and channel
  counts only; raw acoustic vectors and PCM are schema-forbidden.

JSONL import commits canonical `runs.request_json` / `runs.result_json`, then
deterministically rebuilds the normalized diarization index. JSONL remains an
audit/recovery representation; SQLite is authoritative.

### Constraints

- no `rusqlite` usage (only `fsqlite` from `frankensqlite`).
- persistence accepts `CancellationToken` for deadline-aware writes.
- transactions: persist_report runs inside BEGIN/COMMIT with ROLLBACK on error.
- JSONL export/import is one-way operation with lock discipline.
- CLI sync surface is explicit (`sync export-jsonl`, `sync import-jsonl`), locked, and transactional.
- default import conflict behavior is `reject`; `overwrite` requires explicit operator policy.

## 9. Robot Mode Event Schema

### Schema Version Tracking

Robot mode output is versioned via `ROBOT_SCHEMA_VERSION` (currently `"1.1.0"`, the additive live-listen family bump from `1.0.0`). Every emitted NDJSON line includes a `schema_version` field so consumers can detect contract changes.

### Event Types

| Event          | Required Fields                                                                                              |
|---------------|--------------------------------------------------------------------------------------------------------------|
| `run_start`   | `event`, `schema_version`, `request`                                                                        |
| `stage`       | `event`, `schema_version`, `run_id`, `seq`, `ts`, `stage`, `code`, `message`, `payload`                     |
| `run_complete`| `event`, `schema_version`, `run_id`, `trace_id`, `started_at`, `finished_at`, `backend`, `language`, `transcript`, `segments`, `acceleration`, `warnings`, `evidence` |
| `run_error`   | `event`, `schema_version`, `code`, `message`                                                                |
| `backends.discovery` | `event`, `schema_version`, `backends`                                                                 |
| `routing_decision` | `event`, `schema_version`, `run_id`, `ts`, `code`                                                          |
| `transcript.partial` | `event`, `schema_version`, `run_id`, `seq`, `ts`, `text`, `start_sec`, `end_sec`, `confidence`, `speaker` |
| `transcript.confirm` | `event`, `schema_version`, `run_id`, `seq`, `window_id`, `quality_model_id`, `drift`, `latency_ms`, `ts` |
| `transcript.retract` | `event`, `schema_version`, `run_id`, `retracted_seq`, `window_id`, `reason`, `quality_model_id`, `ts` |
| `transcript.correct` | `event`, `schema_version`, `run_id`, `correction_id`, `replaces_seq`, `window_id`, `segments`, `drift`, `latency_ms`, `ts` |
| `transcript.speculation_stats` | `event`, `schema_version`, `run_id`, `windows_processed`, `corrections_emitted`, `confirmations_emitted`, `correction_rate`, `mean_fast_latency_ms`, `mean_quality_latency_ms`, `current_window_size_ms`, `mean_drift_wer`, `ts` |
| `health.report`| `event`, `schema_version`, `ts`, `backends`, `ffmpeg`, `database`, `resources`, `overall_status`            |
| `listen.session_start` | `event`, `schema_version`, `run_id`, `seq`, `ts`, `source`, `capture_backend`, `device`, `sample_rate_hz`, `fast_model`, `quality_model`, `policy`, `step_ms`, `partials`, `vad` |
| `speech_started` | `event`, `schema_version`, `run_id`, `seq`, `ts`, `utterance_id`, `t_session_sec` |
| `transcript.delta` | `event`, `schema_version`, `run_id`, `seq`, `ts`, `utterance_id`, `text`, `t0_sec`, `t1_sec`, `tokens`, `confidence`, `stability` |
| `utterance_end` | `event`, `schema_version`, `run_id`, `seq`, `ts`, `utterance_id`, `reason`, `t0_sec`, `t1_sec`, `text`, `text_truncated`, `delta_count` |
| `listen.warning` | `event`, `schema_version`, `run_id`, `seq`, `ts`, `reason`, `detail` |
| `listen.session_stats` | `event`, `schema_version`, `run_id`, `seq`, `ts`, `final`, bounded session counters and latency summaries |
| `listen.controller` | shared listen envelope plus `controller`, `from`, `to`, `reason`, `brier`, `observations`, `fallback_active` (only with default-off `--adaptive`) |
| `youtube.run_start` | shared YouTube envelope plus resolved request/output settings |
| `youtube.discovered` | shared YouTube envelope plus `id`, `title`, `url` |
| `youtube.downloading` / `youtube.transcribing` | shared YouTube envelope plus `id` |
| `youtube.downloaded` | shared YouTube envelope plus `id`, `audio_path`, `bytes`, `reused` |
| `youtube.done` | shared YouTube envelope plus `id`, output paths, `wall_ms`, `rtf` |
| `youtube.skipped` | shared YouTube envelope plus `id`, `reason` |
| `youtube.failed` | shared YouTube envelope plus `id`, `title`, `error`, `attempts` |
| `youtube.run_complete` | shared YouTube envelope plus aggregate `done`, `skipped`, `failed`, `cancelled` |

### Streaming Delivery

Robot mode CLI (`robot run`) creates an `mpsc::channel`, passes the sender to `FrankenWhisperEngine::transcribe_with_stream`, and drains the receiver on the main thread. Stage events are emitted as they arrive via `emit_robot_stage`. After the pipeline worker completes, any remaining buffered events are flushed, then `emit_robot_complete` or `emit_robot_error` is emitted.

The `robot schema` subcommand emits a self-describing JSON document with event type specifications, required fields, and examples for each event type.

The diarization stage emits `diarize.rollout` before `diarize.start`. Its
payload records requested/resolved implementation, the fail-closed rollout
stage, configuration validity, and whether runtime-verified external speaker
evidence exists. Hint paths and acoustic vectors are never emitted.

The `robot backends` and `robot health` subcommands emit standalone NDJSON diagnostic events (`backends.discovery` and `health.report`). The `robot routing-history` subcommand emits line-oriented `routing_decision` entries derived from persisted routing events for post-hoc analysis. That event is part of the documented robot schema surface, but it is emitted by `robot routing-history` rather than the live `robot run` stream.

## 10. Optional TTY/PTY Low-Bandwidth Audio Mode

Protocol direction (robust-mode target):
- versioned frame protocol with sequence ids, timing metadata, and explicit integrity fields,
- deterministic gap/duplicate/out-of-order detection with machine-readable error codes,
- explicit backpressure semantics and recovery policy for constrained PTY transport,
- replayable framing that can be persisted and re-run through the same transcribe contract.

Implementation note:
- current prototype uses 8kHz mono u-law + zlib + base64 NDJSON framing; compression strategy should stay measurement-driven.

Goal:
- usable audio relay over terminals and constrained PTY channels.

## 11. Quality Gates

- `cargo fmt --check`
- `cargo check --all-targets`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`

## 12. Extension Envelope

**Recently implemented** (formerly future):
- `StreamingEngine` trait with default backward-compatible implementation and `WhisperCppEngine` native support,
- Brier-score confidence calibration (`CalibrationState`) integrated into adaptive routing,
- routing evidence ledger (`RoutingEvidenceLedger`) with full decision audit trail,
- composable pipeline stages (Vad, Separate, Align, Punctuate, Diarize) with builder API,
- `LiveTranscriptionView` TUI component for real-time segment display,
- health report infrastructure (`HealthReport`, dependency checks, resource snapshots),
- stage latency profiling with tuning recommendations,
- graceful Ctrl+C shutdown controller,
- the native `fw robot listen` rolling-buffer live driver with microphone,
  stdin-PCM, and paced file-replay capture, and
- default-off, Brier-gated live cadence/holdback controllers with deterministic
  fallback and `listen.controller` evidence.

**Remaining future packets**:
- deeper native compute kernels beyond current acceleration pass (Vad, Separate,
  Align, and Punctuate still have placeholder or transitional portions),
- corpus-calibrated acoustic confidence and optional ECAPA-style neural
  embeddings behind the same diarization contract,
- native `StreamingEngine` implementations for `InsanelyFastEngine` and `WhisperDiarizationEngine`,
- advanced frankentui dashboard modes: waveform visualization, speaker color-coding, search/filter (while preserving robot mode contract).

## 13. Operational Methodology Artifacts

Execution discipline is tracked in:
- `docs/operational-playbook.md` (phase gates and execution loop),
- `docs/master-todo-bead-map.md` (tracker-to-work-unit mapping),
- `docs/definition_of_done.md` (merge gates),
- `docs/risk-register.md` (active risk + fallback ledger),
- `docs/FRANKENTUI_METHODOLOGY.md` (TUI integration and planning methodology).
