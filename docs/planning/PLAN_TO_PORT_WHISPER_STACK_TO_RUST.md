# PLAN_TO_PORT_WHISPER_STACK_TO_RUST.md

## Executive Summary

Build `franken_whisper` as a Rust 2024, memory-safe, agent-first speech platform that combines:
- whisper.cpp’s efficient local inference/streaming/VAD model,
- insanely-fast-whisper’s throughput-oriented GPU batching ergonomics,
- whisper-diarization’s post-processing stack (alignment + diarization + punctuation).

This is not a line-by-line translation. Legacy repositories are behavioral oracles.

## Why Port + Synthesize

1. Native Rust safety + maintainability.
2. Unified architecture instead of multiple disjoint toolchains.
3. Deep integration with existing FrankenSuite components.
4. Better agent ergonomics (stable machine outputs, deterministic replay surfaces).

## What We Are Porting

- [x] Multi-source ingestion model (file, mic/line-in, stdin stream).
- [x] ffmpeg-first normalization so users do not manage codecs manually.
- [x] Transcription engine abstraction with backend strategy hooks (bridge adapters are transitional).
- [x] Robot mode contract (NDJSON progress + deterministic run metadata).
- [x] Diarization/alignment-aware data model in the API.
- [x] Durable local run storage via `/dp/frankensqlite` (`fsqlite` API only).

## Explicit Exclusions (Phase 0/1)

| Exclusion | Why |
|---|---|
| Re-implementing ffmpeg codecs in Rust | ffmpeg remains external, wrapped cleanly |
| Shipping full native GPU kernels in first scaffold | requires packetized backend integration work |
| Shipping full parity in one step | staged packets are required for correctness/perf proofing |

## Integration Contracts

### `/dp/asupersync` (required, profound)
- Runtime orchestration and cancellation semantics must align to Asupersync concepts:
  - explicit capabilities/cancellation budgets,
  - deterministic replay-friendly event logging,
  - no orphaned background jobs in long-running modes.

### `/dp/frankentorch` and optional `/dp/frankenjax`
- Backend abstraction must allow:
  - model execution over tensor runtime(s),
  - fallback hierarchy: accelerated backend -> classical backend.
- Current implementation: feature-gated acceleration pass (`gpu-frankentorch`, `gpu-frankenjax`)
  with deterministic CPU fallback.

### `/dp/frankensqlite` (mandatory for all SQLite)
- Do not use `rusqlite`.
- All SQLite-backed storage must use `fsqlite` APIs.
- JSONL synchronization is a one-way, locked, atomic adjunct to fsqlite state.

### `/dp/frankentui` (optional)
- Human-facing interactive mode should be an optional integration path.
- Robot mode remains primary for autonomous agents.
- Current implementation: `--features tui` includes runs list + transcript timeline + stage event panes.

## Phase Plan

### Phase 1 - Spec + Contracts
- `EXISTING_LEGACY_WHISPER_STRUCTURE.md`
- `PROPOSED_ARCHITECTURE.md`
- `FEATURE_PARITY.md`
- `SYNC_STRATEGY.md`
- `RECOVERY_RUNBOOK.md`
- `ALIEN_RECOMMENDATIONS.md`
- `docs/operational-playbook.md`
- `docs/definition_of_done.md`
- `docs/risk-register.md`

### Phase 2 - Scaffold
- Rust crate setup, core models, CLI + robot mode.
- ffmpeg normalization wrapper.
- `fsqlite`-only storage and sync commands.

### Phase 3 - Backend Packets
- `FW-P2C-001`: whisper.cpp shell backend parity slice.
- `FW-P2C-002`: diarization/alignment consolidation.
- `FW-P2C-003`: accelerated backend adapters (frankentorch/frankenjax).
- `FW-P2C-004`: asupersync orchestration mode.

### Phase 4 - UX + TUI
- Optional `/dp/frankentui` surface for human workflows. (implemented baseline in `src/tui.rs`)
- keep robot mode schema stable and backward-safe.

### Phase 5 - Native Engine Convergence + Conformance
- Replace shell/script bridge adapters with native Rust engine implementations under one contract.
- Define explicit compatibility envelope (text/timestamp/speaker/calibration tolerances).
- Build conformance harness (golden corpus + invariants + replay contract checks).
- Persist replay metadata (input hash + engine identity/version + output hash) for drift detection.
- Formalize GPU device/stream ownership + cancellation behavior in orchestration events.

## Success Criteria

- Library API supports end-to-end run pipeline with typed outcomes.
- CLI supports file + mic/line-in + stdin workflows.
- ffmpeg conversion is transparent for arbitrary common formats.
- Robot mode emits parseable NDJSON status and final result envelopes.
- All SQLite concerns are handled through `fsqlite`.
- Sync/recovery commands uphold lock + atomicity invariants.
