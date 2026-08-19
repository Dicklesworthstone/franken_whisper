# EXISTING_LEGACY_WHISPER_STRUCTURE.md

> Legacy extraction document. This is the behavioral oracle for synthesis.

## 1. Legacy Sources

### A) `legacy_whispercpp/whisper.cpp`

Core contributions:
- C/C++ high-performance local inference.
- broad hardware support (CPU + many GPU/NPU paths).
- rich CLI with extensive decode/VAD/output controls.
- mic/streaming examples (`examples/stream/stream.cpp`).
- ffmpeg conversion guidance and tooling.

Notable behaviors:
- expects/optimizes for 16kHz mono WAV in CLI flows.
- extensive output formats: txt, srt, vtt, csv, json.
- explicit VAD parameter surface.

### B) `legacy_insanely_fast_whisper/insanely-fast-whisper`

Core contributions:
- throughput-oriented, opinionated CLI.
- GPU-first batching and FlashAttention paths.
- strong ergonomics for one-shot file transcription.
- optional diarization hooks through Hugging Face token.

Notable behaviors:
- fixed chunking defaults (`chunk_length_s=30`, batch tuned).
- fast path emphasizes practical CLI usage over architecture purity.

### C) `legacy_whisper_diarization/whisper-diarization`

Core contributions:
- diarization-focused multi-stage pipeline:
  - optional vocal isolation,
  - ASR,
  - forced alignment,
  - speaker diarization,
  - punctuation restoration,
  - speaker-aware transcript generation.

Notable behaviors:
- practical audio-prep dependence on ffmpeg + optional demucs.
- outputs both `.txt` and `.srt`.
- includes parallel pipeline variant for high VRAM systems.

## 2. Common Capability Intersection

Capabilities present across legacy set:
- file-based transcription.
- configurable model/backend choices.
- timestamps and language handling.
- at least partial diarization support.

Unique strengths to preserve:
- whisper.cpp: local portability + real-time streaming + VAD control.
- insanely-fast: throughput-optimized UX and batch ergonomics.
- whisper-diarization: alignment+speaker+punctuation depth.

## 3. Gaps in Legacy State

Cross-project gaps:
- no single architecture with consistent contracts.
- weak agent-first structured output semantics.
- storage/sync durability model not unified.
- no coherent policy for machine-parseable progress traces.

## 4. Normalization + Input Model (Legacy Observations)

Legacy reality:
- users repeatedly convert files manually to 16kHz mono PCM.
- mic support is separate from file workflows.
- line-in and stream input are inconsistently handled.

Required synthesis behavior:
- one ingestion abstraction with ffmpeg-mediated normalization.
- users provide any common media input; pipeline handles conversion.

## 5. SQLite / Persistence Legacy Note

Legacy projects are not built around `/dp/frankensqlite`.

For `franken_whisper`:
- all SQLite-backed persistence must be implemented with `fsqlite`.
- no direct `rusqlite` usage.

## 6. Porting Constraints

1. Do not line-translate Python/C++ source files.
2. Extract behavior into typed Rust contracts.
3. Preserve strengths, drop incidental implementation debt.
4. Keep room for deep integration with asupersync/frankentorch/frankenjax/frankentui/frankensqlite.
