# FEATURE_PARITY.md

## Legacy-to-Franken Mapping

| Capability | whisper.cpp | insanely-fast-whisper | whisper-diarization | franken_whisper target |
|---|---|---|---|---|
| File transcription | Yes | Yes | Yes | Yes (unified) |
| ffmpeg normalization | Practical guidance/tools | Implicit via upstream libs | Required by pipeline | First-class ingest stage |
| Mic/stream input | Yes (examples) | Not primary | Not primary | Unified capture/stream interface |
| GPU-optimized inference | Partial (backend-specific) | Core strength | Depends on stack | Engine contract + accel hooks (bridge adapters are transitional) |
| Diarization | Basic stereo/tiny diarize modes | Optional (HF token path) | Core strength | Rust-native acoustic engine plus provenance-checked external normalization |
| Word/chunk timestamps | Yes | Yes | Yes (post-alignment) | Normalized segment schema |
| CLI ergonomics | Broad but dense flag surface | Opinionated and simple | Minimal | Human + robot UX split |
| Structured machine output | Limited | JSON transcript file | text/srt outputs | Stable NDJSON progress + final envelope |
| Durable local storage | No unified model | No unified model | No unified model | `frankensqlite` (`fsqlite`) only |
| Library API surface | Low | Low | Low | First-class Rust library |

## Phase-by-Phase Parity Targets

### Phase 1 (current)
- [x] Spec documents and extraction map.
- [x] Architecture contract and stage model.
- [x] `fsqlite`-only persistence contract.

### Phase 2 (scaffold)
- [x] CLI + library end-to-end skeleton.
- [x] File/stdin/mic ingestion contracts.
- [x] ffmpeg normalization wrapper.
- [x] backend adapters (CLI bridge mode, transitional while native engines are ported).
- [x] robot-mode NDJSON contract.
- [x] run/segment/event persistence in `fsqlite`.

### Phase 2.5 (durability + diagnostics)
- [x] JSONL sync: export (db→jsonl with manifest, SHA-256 checksums, atomic rename).
- [x] JSONL sync: import (jsonl→db with manifest validation, conflict policy, transactions).
- [x] Sync locking model (lock file with PID/timestamp, stale detection).
- [x] Conflict semantics: identical payload no-op; divergent payload reject/overwrite via explicit policy.
- [x] `runs --id` CLI command (retrieve full run details by ID).
- [x] `robot backends` diagnostics (per-backend binary, env overrides, availability).
- [x] Real-time robot stage stream includes explicit `*.error` stage events on pipeline failures.
- [x] Integration test suite (12 tests covering storage, sync, diagnostics, CLI validation).

### Phase 3 (parity packets)
- [x] whisper.cpp parity packet (output formats: txt/vtt/srt/csv/json-full/lrc; language/task/detect-only switches; decode params: best-of/beam-size/temperature/thresholds; VAD pipeline with model/threshold/duration/overlap controls; split-on-word; no-timestamps).
- [x] insanely-fast parity packet (batch-size; timestamp level chunk/word with word-level segment extraction; speaker constraints num/min/max; GPU device selection; Flash Attention 2 toggle).
- [x] diarization parity packet (no-stem source separation toggle; whisper-model override; suppress-numerals; device/batch-size controls; hardened SRT parsing for dot-separated timestamps; expanded speaker label recognition: speaker/spk/spkr/s0 patterns).
- [x] line-in and stream behavior parity envelope (cross-input-mode validation: file/stdin/mic InputSource variants; serialization round-trip; error consistency across input modes; 20 integration tests covering Phase 3 contracts).

### Phase 4 (beyond parity)
- [x] native acceleration stage with feature-gated frankentorch/frankenjax + CPU fallback.
- [x] adaptive backend routing evidence ledger (shadow mode, explicit loss + posterior + fallback trigger, deterministic static execution).
- [~] asupersync-managed cancel-safe multi-stage execution (explicit per-stage budgets and timeout codes implemented; deeper region/task attribution remains).
- [x] robust TTY/PTY low-bandwidth audio relay mode (versioned protocol v1; CRC32 + SHA-256 integrity; sequence gap/duplicate detection; fail-closed + skip-missing recovery policies; protocol spec in docs/tty-audio-protocol.md).
- [x] optional frankentui human interface.

### Phase 5 (native-engine convergence)
- [~] Replace bridge adapters with native Rust engine implementations under one engine contract (formal `Engine` trait defined with `name()`, `kind()`, `capabilities()`, `is_available()`, `run()`; `EngineCapabilities` struct; native pilots now exist for whisper.cpp / insanely-fast / diarization with runtime dispatch control via `FRANKEN_WHISPER_NATIVE_EXECUTION` + rollout stage; bridge adapters remain default/safety fallback while pilots converge).
- [x] Land conformance harness with explicit compatibility tolerances (text/timestamp/speaker/calibration envelopes; `SegmentCompatibilityTolerance` with configurable timestamp/text/speaker tolerance; `SegmentComparisonReport` with per-axis drift counts; fixture-driven cross-engine tests in `tests/conformance_harness.rs`).
- [x] Seed fixture-driven conformance harness (`tests/conformance_harness.rs`) with golden-corpus comparison fixtures and expand to corpus-scale replay checks (`compare_segments_with_tolerance`, `compare_replay_envelopes`).
- [x] Add bridge-vs-native corpus pair gating and machine-readable conformance artifact bundle emission (`target/conformance/bridge_native_conformance_bundle.json` from `tests/conformance_harness.rs`).
- [x] Persist replay envelope metadata (input hash, engine identity/version, output hash) for deterministic drift checks (`ReplayEnvelope` struct with `input_content_hash`, `backend_identity`, `backend_version`, `output_payload_hash`; populated in orchestrator; persisted in storage; round-trip tests in `tests/replay_envelope.rs`).
- [x] Engine compatibility spec (docs/engine_compatibility_spec.md: timestamp monotonicity, confidence [0,1] bounds, speaker label validation, cross-engine tolerance bands, replay determinism requirements).
- [x] Runtime segment conformance validation (confidence bounds [0,1], empty speaker label rejection, timestamp ordering) enforced in orchestrator between backend and acceleration stages.
- [x] Add GPU device/stream ownership and cancellation semantics to run-level telemetry (run-level `acceleration_context` now emitted in transcribe JSON and robot `run_complete` when evidence is present; covered by `tests/cli_integration.rs::transcribe_acceleration_context_telemetry_round_trips_in_run_artifacts` and `src/robot.rs::run_complete_includes_acceleration_context_when_present_in_evidence`).

### Phase 6 (native acoustic diarization)

- [x] Define a permutation-invariant diarization/scoring contract with separate
  voice/channel evidence, independent speaker/change confidence, unknown and
  overlap states, deterministic fallback, and bounded resources.
- [x] Implement Rust-native waveform features, multiscale regime-change
  segmentation, robust hard/soft known-interval enrollment, deterministic
  constrained clustering, temporal smoothing, and word-safe transcript
  projection.
- [x] Integrate explicit `auto|acoustic|external|neural` selection, typed
  fallback, stable robot provenance events, cancellation, and DTW/VAD boundary
  hints into the canonical `Diarize` stage.
- [x] Persist normalized reports/turns/hint audits/privacy-safe profile
  summaries in schema v4 and deterministically rebuild them after JSONL import.
- [x] Quarantine the historical six-dimensional text/temporal clusterer from
  acoustic and verified-external evidence gates.
- [~] Promote `auto` through
  Shadow -> Validated -> Fallback -> Primary -> Sole. Default remains `shadow`;
  explicit `--diarization-engine acoustic` is available for focused use.
- [ ] Certify public-corpus DER/JER/calibration and same-host performance.
  Hermetic synthetic tests prove invariants and determinism, not real-corpus
  accuracy.
- [ ] Add the optional ECAPA-style neural implementation tracked by `bd-ohex`
  behind the same output, supervision, privacy, and rollback contract.

## Non-Negotiable Contracts

1. All SQLite concerns use `frankensqlite` (`fsqlite`) only.
2. Robot mode output stays machine-stable and line-oriented.
3. Backend selection is explicit and observable in every run.
4. ffmpeg conversion is transparent to user-facing commands.
5. No compatibility shims for obsolete architecture decisions.
6. Compatibility claims must be backed by executable conformance evidence.
