# ALIEN_RECOMMENDATIONS.md

> High-EV "alien artifact" recommendations selected for `franken_whisper`.

## 1. Two-Lane Orchestration with Bounded Degradation

Concept:
- lane A: low-latency incremental transcript path,
- lane B: higher-quality deferred refinement path.

Policy:
- serve lane A immediately,
- replace/patch with lane B when confidence improves.

Fallback safety:
- deterministic lane-A-only mode when resources are constrained.

## 2. Confidence-Calibrated Segment Fusion

Problem:
- backends disagree on timestamps/speaker boundaries.

Approach:
- maintain candidate graph of segment hypotheses,
- select path minimizing weighted loss over:
  - timing continuity,
  - lexical consistency,
  - speaker transition plausibility.

Safety:
- if posterior confidence < threshold, emit conservative segmentation and mark low-confidence spans.

## 3. Adaptive Backend Routing with Evidence Ledger

State:
- hardware profile, input duration, language hint, diarization request, recent backend latency/error history.

Action:
- choose backend sequence policy (`whisper_cpp`, `insanely_fast`, `whisper_diarization`, or hybrid).

Reward:
- weighted objective: latency, quality proxy, cost, failure risk.

Requirement:
- every adaptive decision produces a machine-readable evidence artifact in persistence layer.

## 4. Low-Bandwidth TTY Audio Relay

Protocol:
- audio -> 8kHz mono μ-law -> chunk compression -> base64 NDJSON frames.

Why:
- enables remote agent workflows over PTY/TTY channels where binary streaming is awkward.

Fallback:
- if decode integrity fails, reject frame and request retransmit (sequence-based gap detection).

## 5. Explicit Loss Matrix for Automation

Define losses:
- false-positive speaker switches,
- timestamp drift,
- dropped words,
- excessive latency,
- backend crash probability.

Use matrix to:
- drive backend/routing policy,
- justify fallback trigger thresholds,
- calibrate quality-vs-speed modes (`fast`, `balanced`, `max-quality`).

## 6. Deterministic Replay Envelope

Persist:
- normalized input hash,
- backend command+version,
- model identifier,
- orchestration stage events,
- output payload hash.

Outcome:
- any run can be replayed and compared for regressions.

## 7. Incremental Formalization Targets

Near-term proofs/contracts to add:
- no orphan subprocess invariant under cancellation,
- monotonic event sequence invariant,
- sync lock exclusivity invariant,
- import idempotence under duplicate JSONL replay.
