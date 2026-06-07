# franken_whisper

<div align="center">
  <img src="franken_whisper_illustration.webp" alt="franken_whisper — agent-first Rust ASR orchestration stack">
</div>

<div align="center">

[![License: MIT+Rider](https://img.shields.io/badge/License-MIT%2BOpenAI%2FAnthropic%20Rider-blue.svg)](./LICENSE)
[![Rust Edition](https://img.shields.io/badge/Rust-2024_Edition-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)
[![Tests](https://img.shields.io/badge/tests-3%2C100%2B-brightgreen.svg)](#testing)
[![Latest Release](https://img.shields.io/github/v/release/Dicklesworthstone/franken_whisper.svg)](https://github.com/Dicklesworthstone/franken_whisper/releases)

</div>

**Agent-first Rust ASR stack with a real in-process pure-Rust Whisper engine (no FFI, no Python, no subprocess — and on CPU it beats whisper.cpp 2.33× on tiny.en, at parity on large-v3-turbo), adaptive Bayesian backend routing, real-time NDJSON streaming, DTW word timestamps, and SQLite-backed run history.**

<div align="center">
<h3>Install in one line</h3>

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh?$(date +%s)" | bash
```

<sub>SHA-256-verified prebuilt binaries for <b>Linux</b> (x86_64 / aarch64), <b>macOS</b> (Intel / Apple&nbsp;Silicon), and <b>WSL</b> — proxy-aware, airgap-capable (<code>--offline</code>), reversible (<code>--uninstall</code>). Windows users: grab <code>windows_amd64.zip</code> from <a href="https://github.com/Dicklesworthstone/franken_whisper/releases/latest">the latest release</a>. All flags: <a href="#installation">Installation</a>.</sub>

</div>

> **v0.2.0 — the native engine is real, and it is fast.** The in-process pure-Rust Whisper engine (built on [FrankenTorch](https://github.com/Dicklesworthstone/frankentorch) kernels, `#![forbid(unsafe_code)]` in-crate) now transcribes **2.33× faster than whisper.cpp on tiny.en and at parity on large-v3-turbo** — CPU vs CPU, same machine (Apple M4 Pro, interleaved runs), while using about 18% less user CPU on the large model (53.8 s vs 65.4 s). Native-vs-whisper.cpp conformance on the reference fixture: **WER 0.0000**.
>
> | Model | franken_whisper native (CPU) | whisper.cpp (CPU) | |
> |---|---|---|---|
> | tiny.en (11 s clip) | **475 ms** | 1,105 ms | **2.33× faster** |
> | large-v3-turbo (11 s clip) | **9.73 s** | 9.59 s | parity |

---

## The Problem

Speech-to-text pipelines are fragmented. To get production-quality transcription you need `whisper.cpp` for speed, `insanely-fast-whisper` for GPU batching, and `whisper-diarization` for speaker identification. Each has its own CLI, output format, error handling, and deployment story. Orchestrating them from scripts means parsing inconsistent stdout, handling timeouts manually, killing zombie subprocesses, and losing all run history when something crashes.

Agent workflows make the problem worse. Modern LLM agents need **structured**, **streaming**, **machine-readable** output, not human-oriented terminal decorations that break the moment they touch `jq`, pipes, or SSH.

## The Solution

`franken_whisper` is a single Rust binary that wraps every major Whisper backend behind a unified, agent-first interface — and ships its own engine:

- **A real in-process Whisper engine, in pure Rust.** ggml model parsing, log-mel frontend, encoder/decoder transformer inference on FrankenTorch CPU kernels, greedy decoding with whisper.cpp's full timestamp-rule suite, and cross-attention DTW word timestamps. No FFI, no Python, no subprocess — drop a ggml model file in place and transcribe.
- **Adaptive Bayesian backend routing.** Each `auto` request runs a formal decision contract with an explicit loss matrix, per-backend Beta posteriors, Brier-scored calibration, and deterministic fallback when the model is mis-calibrated.
- **Real-time NDJSON streaming.** Every pipeline stage emits sequenced, timestamped events on stable schema `v1.0.0`. No fragile regex; agents parse JSON.
- **Durable run history.** Every transcription persists to SQLite with full event logs, replay envelopes, and JSONL export/import, even when the process crashes mid-run.
- **Cooperative cancellation.** Ctrl+C propagates through the whole pipeline via cancellation tokens. Subprocesses get killed, transactions roll back via savepoints, finalizers run within a bounded budget, exit code 130.
- **Zero-dependency audio decode.** MP3, AAC, FLAC, WAV, OGG, Vorbis, and ALAC decode natively via `symphonia`. `ffmpeg` is a fallback for video and exotic codecs, and is auto-provisioned on Linux x86_64 if missing.
- **Native engine rollout governance.** In-process Rust replacements for the bridge adapters ship behind a 5-stage rollout (Shadow → Validated → Fallback → Primary → Sole) with conformance gating at every promotion.
- **TTY audio transport.** Compressed audio (mu-law + zlib + base64) over PTY links with handshake, integrity hashes, deterministic retransmission, and an adaptive bitrate controller. Transcript-streaming control frames (protocol v2) carry speculation events end-to-end over the same link.
- **Word-level timestamps.** First-class support via `whisper.cpp`'s word-timestamp pipeline, with the cancellation token threaded into the inner extraction loop.

### Why franken_whisper?

| Feature | whisper.cpp | insanely-fast-whisper | whisper-diarization | **franken_whisper** |
|---------|:-----------:|:---------------------:|:-------------------:|:-------------------:|
| Streaming output | partial | — | — | **sequenced NDJSON stage events** |
| Machine-readable errors | exit code only | exceptions | exceptions | **12 structured `FW-*` error codes** |
| Adaptive backend selection | — | — | — | **Bayesian decision contract** |
| Run persistence | — | — | — | **SQLite + JSONL replay packs** |
| Diarization | — | yes (HF token) | yes | **yes (any backend)** |
| GPU acceleration | CUDA / Metal | CUDA / MPS | CUDA | **`frankentorch` / `frankenjax`** |
| Cancellation | `SIGKILL` | `KeyboardInterrupt` | `SIGKILL` | **cooperative `CancellationToken`** |
| TTY audio relay | — | — | — | **mulaw + zlib + base64 NDJSON** |
| Native audio decode | WAV only | needs ffmpeg | needs ffmpeg | **MP3 / AAC / FLAC / WAV / OGG / ALAC** |
| Memory safety | C++ | Python | Python | **`#![forbid(unsafe_code)]`** |

---

## Quick Example

```bash
# Transcribe an audio file. MP3 / FLAC / OGG / AAC decoded natively, no ffmpeg needed.
franken_whisper transcribe --input meeting.mp3 --json

# Transcribe a video file. Audio extracted automatically via ffmpeg fallback.
franken_whisper transcribe --input presentation.mp4 --json

# Stream real-time pipeline events for agents
franken_whisper robot run --input meeting.mp3 --backend auto

# Speculative streaming: fast partial transcripts with quality-model corrections
franken_whisper robot run --input meeting.mp3 --speculative \
  --fast-model tiny.en --quality-model large-v3

# Speaker diarization with pyannote
franken_whisper transcribe --input meeting.mp3 --diarize --hf-token "$HF_TOKEN" --json

# TinyDiarize: whisper.cpp's built-in speaker-turn detection (no HF token needed)
franken_whisper transcribe --input meeting.mp3 --tiny-diarize --json

# Discover available backends and their capabilities
franken_whisper robot backends

# System health check (backends, ffmpeg, database, resources)
franken_whisper robot health

# Query run history
franken_whisper runs --limit 10 --format json

# Export run history to a portable JSONL snapshot (full or incremental)
franken_whisper sync export-jsonl --output ./snapshot

# Compressed audio over a text-only channel (PTY, SSH, serial)
franken_whisper tty-audio encode --input audio.wav > frames.ndjson
cat frames.ndjson | franken_whisper tty-audio decode --output restored.wav
```

---

## Design Philosophy

### Agent-First, Human-Optional

The primary interface is `robot`. Every command in robot mode emits sequenced, timestamped NDJSON on stdout with schema version `1.0.0`. Human-friendly output (`transcribe` without `--json`) is the exception. Robot mode never mixes decorative stderr into the data stream; structured `run_error` envelopes replace human prose even for argument-parsing failures.

### Deterministic by Default

Given identical inputs and parameters, `franken_whisper` produces identical outputs. The TTY retransmit loop, replay envelopes, replay packs, and conformance harness all enforce determinism. Random elements (UUIDs, wall-clock timestamps) are quarantined to metadata fields; they never enter computational output.

### Fail Loud, Recover Gracefully

Every error has a structured code (`FW-IO`, `FW-CMD-TIMEOUT`, `FW-BACKEND-UNAVAILABLE`, `FW-STAGE-TIMEOUT`, …) and propagates through the NDJSON event stream. Cancellation tokens let in-flight work checkpoint and clean up instead of dying mid-write. In-progress SQLite transactions roll back via savepoints; subprocesses get `SIGKILL`; temp directories are removed by registered finalizers under a bounded budget.

### Composition Over Configuration

The pipeline is **composed** dynamically per request. The 10 canonical stages (Ingest, Normalize, VAD, Source Separate, Backend, Accelerate, Align, Punctuate, Diarize, Persist) are skipped when unnecessary, budgeted independently, and profiled automatically. The `PipelineBuilder` validates ordering at build time; runtime never hits "stage X requires stage Y" errors.

### No Unsafe Code

The entire crate uses `#![forbid(unsafe_code)]`. `forbid` is stricter than `deny`: it cannot be overridden per item. Memory safety is enforced by the compiler, not by code review.

### Zero External Dependencies for Common Audio

`franken_whisper` decodes MP3, AAC, FLAC, WAV, OGG (Vorbis), and ALAC entirely in-process via `symphonia`; no `ffmpeg`, no Python, no `PATH` lookup. `ffmpeg` is only invoked as a fallback for video files, exotic codecs (AC3, DTS, Opus-in-MKV), and live microphone capture. When `ffmpeg` *is* needed and missing on Linux x86_64, it is downloaded once into the per-user state directory.

### Adaptive Controllers With Conservative Fallbacks

Every adaptive controller in the system (the backend router, the speculative window controller, the budget tuner, the TTY adaptive bitrate controller, the correction tracker) ships with an explicit *alien-artifact engineering contract*: state space, action space, loss matrix, posterior, calibration metric, deterministic fallback, and an evidence ledger. When models drift, the system **falls back to a deterministic policy** rather than confidently making bad decisions.

---

## The Whisper Ecosystem Landscape

```
          +--------------------------------------------------------------+
          |            INFERENCE ENGINES (run the model)                 |
          |                                                              |
          | whisper.cpp           (C++, CPU/Metal/CUDA, ~47k stars)      |
          | faster-whisper        (Python/CTranslate2, ~14k stars)       |
          | OpenAI Whisper        (Python/PyTorch, ~95k stars)           |
          +--------------------------------------------------------------+
                                         |
          +------------------------------v-------------------------------+
          |     ENHANCED PIPELINES (add features on top)                 |
          |                                                              |
          | WhisperX               (faster-whisper + wav2vec2 + pyannote)|
          | whisper-diarization    (Whisper + Demucs + NeMo TitaNet)     |
          | insanely-fast-whisper  (HuggingFace Transformers, max GPU)   |
          | whisper-timestamped    (DTW word timestamps)                 |
          +--------------------------------------------------------------+
                                         |
          +------------------------------v-------------------------------+
          | ORCHESTRATION (manage engines/pipelines)                     |
          |                                                              |
          | > franken_whisper <    (Rust, Bayesian routing,              |
          |                         10-stage pipeline, conformance       |
          |                         gating, native engine rollout,       |
          |                         evidence-based decisions)            |
          +--------------------------------------------------------------+
```

Most tools occupy one level. `franken_whisper` is the orchestration layer: it wraps inference engines and enhanced pipelines behind a unified interface, then adds capabilities none of them provide individually.

---

## How franken_whisper Compares

### Orchestration & Architecture

| Capability | whisper.cpp | faster-whisper | WhisperX | WhisperLive | WhisperS2T | **franken_whisper** |
|------------|:-----------:|:--------------:|:--------:|:-----------:|:----------:|:-------------------:|
| Language | C++ | Python | Python | Python | Python | **Rust** |
| Multi-backend | — | — | — | 3 | 4 | **3 backends + paired native pilots** |
| Backend selection | — | — | — | manual | manual | **Bayesian decision contract** |
| Pipeline stages | monolithic | monolithic | 3 | monolithic | monolithic | **10 composable stages** |
| Per-stage budgets | — | — | — | — | — | **independent timeouts + auto profiling** |
| Speculative streaming | — | — | — | single-model | — | **dual-model fast+quality with retraction** |
| Conformance validation | — | — | — | — | — | **cross-engine, 50 ms tolerance, drift detection** |
| Native rollout governance | — | — | — | — | — | **5-stage Shadow → Sole with conformance gates** |
| Memory safety | C++ | Python GC | Python GC | Python GC | Python GC | **`#![forbid(unsafe_code)]`** |

### Persistence & Observability

| Capability | whisper.cpp | faster-whisper | WhisperX | WhisperLive | **franken_whisper** |
|------------|:-----------:|:--------------:|:--------:|:-----------:|:-------------------:|
| Run history | — | — | — | — | **SQLite WAL + JSONL export/import** |
| Decision audit trail | — | — | — | — | **200-entry evidence ledger** |
| Replay envelopes | — | — | — | — | **SHA-256 content + output hashes** |
| Replay packs | — | — | — | — | **4-artifact deterministic reproducibility bundle** |
| Structured errors | exit code | exceptions | exceptions | — | **12 `FW-*` codes / 6 `FW-ROBOT-*` codes** |
| NDJSON streaming | partial | — | — | WebSocket | **sequenced stage events** |
| Cancellation | `SIGKILL` | `KeyboardInterrupt` | — | — | **cooperative `CancellationToken`** |
| Resource cleanup | none guaranteed | GC | GC | GC | **RAII + bounded LIFO finalizers** |
| Latency profiling | — | — | — | — | **per-stage utilization with tuning recommendations** |

### Audio & Format Support

| Capability | whisper.cpp | faster-whisper | WhisperX | **franken_whisper** |
|------------|:-----------:|:--------------:|:--------:|:-------------------:|
| Native audio decode | WAV only | needs ffmpeg | needs ffmpeg | **MP3 / AAC / FLAC / WAV / OGG / Vorbis / ALAC via `symphonia`** |
| ffmpeg required? | for non-WAV | always | always | **fallback only (video + exotic codecs + mic)** |
| Video audio extraction | — | — | — | **automatic via `-vn`** |
| TTY audio transport | — | — | — | **mulaw + zlib + base64 NDJSON** |
| Microphone capture | — | — | — | **ALSA / AVFoundation / DirectShow via ffmpeg** |
| Auto-provisioned ffmpeg | — | — | — | **downloads static binary if missing (Linux x86_64)** |

### Commercial API Comparison

| Capability | Groq Whisper API | Deepgram Nova-3 | AssemblyAI | **franken_whisper** |
|------------|:----------------:|:---------------:|:----------:|:-------------------:|
| Runs locally | no | no | no | **yes** |
| Open source | no | no | no | **yes (MIT + rider)** |
| Data leaves machine | yes | yes | yes | **never (except optional HF model download)** |
| Cost per hour of audio | ~$0.04 | ~$0.75 | ~$0.65 | **$0 (your hardware)** |
| Inference speed | very fast | fast | moderate | **depends on selected backend** |
| Multi-model routing | — | — | — | **Bayesian adaptive** |
| Diarization | limited | yes | yes | **yes (any backend)** |
| Custom pipeline stages | — | — | — | **10 composable stages** |

---

## Installation

### Quick Install (Pre-built Binary)

```bash
curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh?$(date +%s)" | bash
```

The installer downloads the appropriate release asset for your platform, verifies its SHA-256 checksum against `checksums-sha256.txt`, and installs to `~/.local/bin`.

Options:

| Flag | Purpose |
|------|---------|
| `--system` | Install to `/usr/local/bin` instead of `~/.local/bin` |
| `--easy-mode` | Auto-update shell `PATH` and rc files |
| `--verify` | Run a post-install self-test (`robot health`) |
| `--version vX.Y.Z` | Pin to a specific release |
| `--force` | Reinstall even if the same version is present |
| `--offline TARBALL` | Airgap install from a local archive (verifies sibling `.sha256`) |
| `--from-source` | Build from a source clone (clones sibling FrankenSuite deps) |
| `--quiet` / `--no-gum` | Script-friendly / plain-ANSI output |
| `--no-verify` | Skip checksum verification (testing only) |
| `--uninstall` | Remove the binary and `PATH` modifications |

`HTTP_PROXY`/`HTTPS_PROXY` are honored on every download. Prebuilt targets: `linux_amd64`, `linux_arm64`, `darwin_amd64`, `darwin_arm64`, `windows_amd64` (zip; manual install on native Windows — the bash installer covers Linux, macOS, and WSL).

### From Source

```bash
git clone https://github.com/Dicklesworthstone/franken_whisper.git
cd franken_whisper

# minimal build
cargo build --release

# with TUI support
cargo build --release --features tui

# with GPU acceleration via frankentorch
cargo build --release --features gpu-frankentorch

# with GPU acceleration via frankenjax
cargo build --release --features gpu-frankenjax
```

The release profile is aggressively optimized for distribution: `opt-level = "z"`, full LTO, single codegen unit, `panic = "abort"`, stripped symbols.

### Prerequisites

- **Rust nightly** (2024 edition; pinned via `rust-toolchain.toml`)
- **ffmpeg** (optional): only needed for video files, exotic codecs `symphonia` cannot decode, and live microphone capture. On Linux x86_64 it is auto-provisioned on first use unless `FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG=0`.
- **A Whisper model file** for the built-in native engine (e.g. `ggml-tiny.en.bin` / `ggml-large-v3-turbo.bin` from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)); place it in `$FRANKEN_WHISPER_MODEL_DIR`, `~/.cache/franken_whisper/models`, or `~/models/whisper` — `scripts/fetch_test_models.sh` fetches a pinned `tiny.en`. With a model present, **no external backend binaries are required**: set `FRANKEN_WHISPER_NATIVE_EXECUTION=1` to prefer the native engine; even without that flag, the default-on recovery path runs it automatically whenever no bridge binary is usable.
- **Bridge backend binaries** (optional alternates; the Bayesian router arbitrates):
  - `whisper-cli` (from whisper.cpp); override via `FRANKEN_WHISPER_WHISPER_CPP_BIN`
  - `insanely-fast-whisper` (Python entry point); override via `FRANKEN_WHISPER_INSANELY_FAST_BIN`
  - `python3` with `pyannote.audio` installed (for the diarization backend); override via `FRANKEN_WHISPER_PYTHON_BIN`
- **HuggingFace token** (for diarization only): `--hf-token`, `FRANKEN_WHISPER_HF_TOKEN`, or `HF_TOKEN`

### Sibling Crate Dependencies

`franken_whisper` integrates several crates from the FrankenSuite ecosystem. The core infrastructure crates are published on **crates.io** and resolve automatically; only the SQLite stack and the optional features use local path dependencies.

| Crate | Source | Purpose |
|-------|--------|---------|
| `asupersync` | crates.io `0.3.1` | Cancel-correct orchestration primitives |
| `franken-kernel` | crates.io `0.3.1` | Budget, TraceId, time utilities |
| `franken-evidence` | crates.io `0.3.1` | Evidence ledger primitives |
| `franken-decision` | crates.io `0.3.1` | Decision contract framework |
| `fsqlite` | path: `../frankensqlite` | Pure-Rust SQLite implementation |
| `fsqlite-types` | path: `../frankensqlite` | Core SQLite value types |
| `ftui` *(feature: `tui`)* | path: `../frankentui` | Terminal UI framework |
| `ft-api` / `ft-core` *(feature: `gpu-frankentorch`)* | path: `../frankentorch` | GPU tensor operations |
| `fj-api` / `fj-core` *(feature: `gpu-frankenjax`)* | path: `../frankenjax` | JAX-based GPU compute |

---

## Quick Start

### 1. Basic Transcription

```bash
# plain text
franken_whisper transcribe --input audio.mp3

# full JSON report (segments + timing + backend identity + replay envelope)
franken_whisper transcribe --input audio.mp3 --json

# explicit backend
franken_whisper transcribe --input audio.mp3 --backend whisper_cpp --json

# language hint
franken_whisper transcribe --input audio.mp3 --language ja --json
```

### 2. Robot Mode (Agent Integration)

```bash
franken_whisper robot run --input audio.mp3 --backend auto
```

Output (one JSON object per line, schema `1.0.0`):

```json
{"event":"run_start","schema_version":"1.0.0","request":{"input":"audio.mp3","backend":"auto"}}
{"event":"stage","schema_version":"1.0.0","run_id":"...","seq":1,"stage":"ingest","code":"ingest.start","message":"materializing input"}
{"event":"stage","schema_version":"1.0.0","run_id":"...","seq":2,"stage":"normalize","code":"normalize.ok","message":"audio normalized"}
{"event":"stage","schema_version":"1.0.0","run_id":"...","seq":3,"stage":"backend","code":"backend.routing.decision_contract","message":"routing decision","payload":{...}}
{"event":"run_complete","schema_version":"1.0.0","run_id":"...","backend":"whisper_cpp","transcript":"Hello world..."}
```

### 3. Speaker Diarization

```bash
franken_whisper transcribe \
  --input meeting.mp3 \
  --diarize \
  --hf-token "$HF_TOKEN" \
  --min-speakers 2 \
  --max-speakers 5 \
  --json
```

### 4. Microphone Capture

```bash
# record 30 seconds from the default mic
franken_whisper transcribe --mic --mic-seconds 30 --json

# specific device
franken_whisper transcribe --mic --mic-device "hw:0" --json
```

### 5. Stdin Input

```bash
cat audio.mp3 | franken_whisper transcribe --stdin --json
```

---

## YouTube Ingestion

`franken_whisper youtube` downloads YouTube audio and transcribes each video into a clean, deep-linked **markdown + JSON** pair. Hand it individual video URLs, playlist URLs (auto-expanded to their videos), or a file of URLs — it downloads, transcribes, and writes one transcript per video into `youtube_transcripts/` (override with `--output-dir`). Because the transcription runs on the same engine as `transcribe`, this pairs naturally with the in-process native engine: **no cloud, no API keys, your audio never leaves the machine.**

```bash
# Single video
franken_whisper youtube "https://www.youtube.com/watch?v=jNQXAC9IVRw" --model tiny.en

# Several videos at once
franken_whisper youtube \
  "https://youtu.be/VIDEO_A" \
  "https://youtu.be/VIDEO_B" \
  --concurrency 4

# A whole playlist (expanded to its videos automatically)
franken_whisper youtube "https://www.youtube.com/playlist?list=PLxxxxxxxx"

# A file with one URL per line (blank lines and #/;/] comments ignored)
franken_whisper youtube --batch-file urls.txt --output-dir ./talks
```

URLs may be passed positionally or with repeated `--url` flags, and mixed freely with `--batch-file`. A `watch?v=X&list=Y` URL is treated as the **single video** `X`, not the surrounding playlist.

### Output

Each video produces a markdown transcript and a JSON sidecar, named `{upload_date} - {title} [{id}]` (the YouTube id keeps names collision-proof; non-ASCII titles are preserved, path-hostile characters folded). The downloaded audio is kept under `audio/<id>.<ext>` (delete it automatically with `--no-keep-audio`), and a `.fw_youtube_manifest.json` state file tracks per-video progress.

```
youtube_transcripts/
├── 2005-04-24 - Me at the zoo [jNQXAC9IVRw].md
├── 2005-04-24 - Me at the zoo [jNQXAC9IVRw].json
├── audio/
│   └── jNQXAC9IVRw.webm
└── .fw_youtube_manifest.json
```

The **markdown** leads with the title, a metadata line, a source link, and an honesty note, then paragraphs grouped on natural pauses (and speaker changes when `--diarize` is set), each prefixed with a `youtu.be` deep-link that jumps to that moment in the video:

```markdown
# Me at the zoo

**Channel:** jawed · **Uploaded:** 2005-04-24 · **Duration:** 0:19
**Source:** [www.youtube.com/watch?v=jNQXAC9IVRw](https://www.youtube.com/watch?v=jNQXAC9IVRw) · **Transcribed:** franken_whisper 0.2.0 (native, tiny.en)

> Note: machine transcription; timestamps are approximate and deep-link into the video.

---

**[0:00](https://youtu.be/jNQXAC9IVRw?t=0)** All right, so here we are in front of the elephants…

---
_Transcribed by franken_whisper 0.2.0 (native, tiny.en, auto) in 0.48s — RTF 0.025_
```

The **JSON** sidecar carries the structured form — `video` metadata, `run` metadata, and a flat `utterances` array (one record per segment with `start_sec` / `end_sec` / `text` / `confidence`, plus `speaker` when diarized):

```json
{
  "video": { "id": "jNQXAC9IVRw", "title": "Me at the zoo", "channel": "jawed",
             "upload_date": "20050424", "duration": 19,
             "webpage_url": "https://www.youtube.com/watch?v=jNQXAC9IVRw" },
  "run":   { "model": "tiny.en", "engine": "native", "backend": "auto",
             "version_tag": "0.2.0", "started": "2026-06-07T00:00:00Z",
             "wall_ms": 480, "rtf": 0.025 },
  "utterances": [
    { "i": 0, "start_sec": 0.0, "end_sec": 4.2,
      "text": "All right, so here we are in front of the elephants…",
      "confidence": 0.91, "speaker": null }
  ]
}
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `[URL]...` / `--url <U>` | — | Video or playlist URLs (positional or repeated `--url`) |
| `--batch-file <F>` | — | File with one URL per line (`#`/`;`/`]` comments and blanks ignored) |
| `--output-dir <DIR>` | `youtube_transcripts` | Destination for markdown, JSON, and `audio/` |
| `--model <M>` | backend-specific | Model name or path forwarded to the engine |
| `--language <L>` | auto-detect | Language hint (ISO 639-1) |
| `--backend <B>` | `auto` | `auto`, `whisper-cpp`, `insanely-fast`, `whisper-diarization` |
| `--diarize` | `false` | Speaker diarization (emits `SPEAKER_NN` labels) |
| `--concurrency <N>` | `3` | Maximum concurrent downloads |
| `--no-keep-audio` | `false` | Delete each audio file after its transcript is written |
| `--no-retry` | `false` | Do not retry videos previously marked failed in the manifest |
| `--abort-on-error` | `false` | Stop the whole run on the first per-video failure |
| `--json-summary` | `false` | Emit the final run summary as JSON on stdout (for scripting) |

### Idempotent & Resumable

The `.fw_youtube_manifest.json` makes every run resumable. A re-run **skips videos already `done`**, **retries previously failed videos** (unless `--no-retry`), and picks up exactly where it left off. **Ctrl+C cancels cleanly** — yt-dlp is killed, in-flight transcription aborts, and the manifest stays honest about what completed. By default a partial failure is reported and the run continues; `--abort-on-error` stops on the first failure instead.

### Requires `yt-dlp`

This subcommand orchestrates [`yt-dlp`](https://github.com/yt-dlp/yt-dlp), which you install separately:

```bash
brew install yt-dlp        # macOS
pipx install yt-dlp        # cross-platform, isolated
pip install -U yt-dlp      # cross-platform
```

Point `franken_whisper` at a specific build with `FRANKEN_WHISPER_YTDLP_BIN=/path/to/yt-dlp` if it is not on `PATH`.

**Keep yt-dlp current.** YouTube changes its site frequently, and `yt-dlp` ships rapid updates to keep pace — so a download that "stops working" is almost always a stale `yt-dlp`, not a franken_whisper bug. franken_whisper warns when the resolved `yt-dlp` build is **more than 90 days old**. The fix is simply:

```bash
yt-dlp -U                       # update to the latest stable
yt-dlp --update-to nightly      # or track the nightly channel for the freshest fixes
```

franken_whisper deliberately downloads with the forgiving `bestaudio/best` format selector rather than a strict codec filter, so a slightly stale yt-dlp still succeeds in more cases — but when YouTube breakage does occur, updating yt-dlp is the first and usually the only step.

---

## Command Reference

`franken_whisper` exposes seven top-level subcommands.

### `transcribe`

Core transcription command. Runs the full pipeline: ingest, normalize, optional VAD / source separation, backend execution, optional acceleration, optional alignment / punctuation / diarization, and persistence.

```bash
franken_whisper transcribe [OPTIONS]
```

**Input (mutually exclusive):**

| Flag | Description |
|------|-------------|
| `--input <PATH>` | Audio or video file path |
| `--stdin` | Read audio bytes from stdin |
| `--mic` | Capture from microphone via ffmpeg |

**Backend & Model:**

| Flag | Default | Description |
|------|---------|-------------|
| `--backend <KIND>` | `auto` | `auto`, `whisper_cpp`, `insanely_fast`, `whisper_diarization` |
| `--model <MODEL>` | backend-specific | Model name or path forwarded to backend |
| `--language <LANG>` | auto-detect | Language hint (ISO 639-1) |
| `--translate` | `false` | Translate to English |
| `--diarize` | `false` | Enable speaker diarization |

**Output:**

| Flag | Description |
|------|-------------|
| `--json` | Full JSON run report on stdout |
| `--output-txt` | Plain text (whisper.cpp side-output) |
| `--output-vtt` | WebVTT subtitles |
| `--output-srt` | SRT subtitles |
| `--output-csv` | CSV |
| `--output-json-full` | Extended JSON with metadata |
| `--output-lrc` | LRC karaoke format |

**Storage:**

| Flag | Default | Description |
|------|---------|-------------|
| `--db <PATH>` | `.franken_whisper/storage.sqlite3` | SQLite database path |
| `--no-persist` | `false` | Skip persistence entirely |
| `--timeout <SEC>` | — | Overall pipeline deadline (seconds) |

**Inference Tuning (whisper.cpp):**

| Flag | Default | Description |
|------|---------|-------------|
| `--threads <N>` | 4 | Computation threads |
| `--processors <N>` | 1 | Parallel processors |
| `--no-gpu` | `false` | Force CPU-only |
| `--beam-size <N>` | 5 | Beam search width |
| `--best-of <N>` | 5 | Sampling candidates |
| `--temperature <F>` | 0.0 | Sampling temperature |
| `--temperature-increment <F>` | — | Temperature fallback increment |
| `--entropy-threshold <F>` | — | Entropy threshold for fallback |
| `--logprob-threshold <F>` | — | Log-probability threshold |
| `--no-speech-threshold <F>` | — | No-speech probability threshold |
| `--max-context <N>` | — | Max context tokens from prior segment |
| `--max-segment-length <N>` | — | Max segment length (characters) |
| `--no-timestamps` | `false` | Suppress timestamps |
| `--detect-language-only` | `false` | Detect language and exit |
| `--split-on-word` | `false` | Split segments on word boundaries |
| `--no-fallback` | `false` | Disable temperature fallback |
| `--suppress-nst` | `false` | Suppress non-speech tokens |
| `--tiny-diarize` | `false` | Enable TinyDiarize (speaker-turn token injection) |
| `--prompt <TEXT>` | — | Initial prompt to guide transcription style |
| `--carry-initial-prompt` | `false` | Carry prompt across segments |

**Audio Windowing (whisper.cpp):**

| Flag | Default | Description |
|------|---------|-------------|
| `--offset-ms <N>` | 0 | Start transcription at offset (ms) |
| `--duration-ms <N>` | — | Transcribe only this duration (ms) |
| `--audio-ctx <N>` | — | Audio context size (tokens) |
| `--word-threshold <F>` | — | Per-word confidence threshold (drop words below) |
| `--suppress-regex <REGEX>` | — | Suppress tokens matching regex |

Word-level timestamp *extraction* (max-len, token-threshold, token-sum-threshold) is exposed through the `WordTimestampParams` struct via the library API; the bridge parser already extracts nested `words` arrays from `whisper.cpp` output when those parameters are set programmatically.

**VAD (Voice Activity Detection):**

| Flag | Description |
|------|-------------|
| `--vad` | Enable Voice Activity Detection |
| `--vad-model <PATH>` | Custom VAD model path |
| `--vad-threshold <F>` | Speech detection threshold |
| `--vad-min-speech-ms <N>` | Minimum speech duration (ms) |
| `--vad-min-silence-ms <N>` | Minimum silence duration (ms) |
| `--vad-max-speech-s <F>` | Maximum speech duration (seconds) |
| `--vad-speech-pad-ms <N>` | Speech padding (ms) |
| `--vad-samples-overlap <N>` | Sample overlap between windows |

**Batching (insanely-fast-whisper):**

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size <N>` | 24 | Parallel inference batch size |
| `--gpu-device <DEV>` | auto | GPU device (`0`, `cuda:0`, `mps`) |
| `--flash-attention` | `false` | Enable Flash Attention 2 |
| `--hf-token <TOKEN>` | env | HuggingFace token for diarization |
| `--timestamp-level` | `chunk` | `chunk` or `word` granularity |
| `--transcript-path <PATH>` | — | Override transcript output path |

**Diarization:**

| Flag | Description |
|------|-------------|
| `--num-speakers <N>` | Exact speaker count |
| `--min-speakers <N>` | Minimum speakers |
| `--max-speakers <N>` | Maximum speakers |
| `--no-stem` | Disable vocal isolation (Demucs source separation) |
| `--suppress-numerals` | Spell out numbers for alignment stability |
| `--diarization-model <MODEL>` | Override whisper model for the diarization stage |

**Speculative Streaming:**

| Flag | Default | Description |
|------|---------|-------------|
| `--speculative` | `false` | Enable dual-model speculative cancel-correct mode |
| `--fast-model <MODEL>` | — | Fast model for low-latency partial transcripts |
| `--quality-model <MODEL>` | — | Quality model for correction / verification |
| `--speculative-window-ms <N>` | 3000 | Sliding window size (ms) |
| `--speculative-overlap-ms <N>` | 500 | Window overlap (ms) |
| `--correction-tolerance-wer <F>` | — | WER tolerance for confirmation vs. retraction |
| `--no-adaptive` | `false` | Disable adaptive window sizing |
| `--always-correct` | `false` | Force quality model on every window (evaluation mode) |

### `robot`

Agent-first interface with structured NDJSON output and a stable schema contract.

```bash
# streaming transcription with stage events
franken_whisper robot run [TRANSCRIBE_OPTIONS]

# emit the full JSON schema for every event type
franken_whisper robot schema

# discover backends and capabilities
franken_whisper robot backends

# system health diagnostics (backends, ffmpeg, database, resources)
franken_whisper robot health

# query routing decision history
franken_whisper robot routing-history [--run-id <ID>] [--limit 20]
```

**Robot Event Catalog (schema `1.0.0`):**

| Event | Description |
|-------|-------------|
| `run_start` | Request accepted, pipeline starting |
| `stage` | Pipeline stage progress (sequenced, timestamped) |
| `run_complete` | Transcription finished with full result |
| `run_error` | Pipeline failed with structured error code |
| `backends.discovery` | Backend availability + per-backend capabilities |
| `routing_decision` | Backend routing decision with posterior snapshot and evidence |
| `health.report` | System health (backend / ffmpeg / DB / resource status) |
| `transcript.partial` | Speculative fast-model partial transcript (immediate) |
| `transcript.confirm` | Quality model confirms partial (drift within tolerance) |
| `transcript.retract` | Quality model retracts partial (drift exceeds tolerance) |
| `transcript.correct` | Quality model correction with corrected segments |
| `transcript.speculation_stats` | Aggregate speculation pipeline statistics |

**Stage Codes.** Each pipeline stage emits paired `*.start` / `*.ok` codes (or `*.error` on failure, `*.skip` when not needed, `*.cancelled` on token fire, `*.timeout` on budget overrun):

`ingest.start`, `ingest.ok`, `normalize.start`, `normalize.ok`, `vad.start`, `vad.ok`, `separate.start`, `separate.ok`, `backend.start`, `backend.ok`, `backend.routing.decision_contract`, `acceleration.start`, `acceleration.ok`, `align.start`, `align.ok`, `punctuate.start`, `punctuate.ok`, `diarize.start`, `diarize.ok`, `persist.start`, `persist.ok`, `orchestration.budgets`, `orchestration.latency_profile`

**Health Report.** `robot health` probes every subsystem and returns a structured diagnostic:

```json
{
  "event": "health.report",
  "schema_version": "1.0.0",
  "ts": "2026-04-25T00:00:00Z",
  "backends": [
    {"name": "whisper.cpp", "available": true, "path": "/usr/local/bin/whisper-cli", "version": "1.7.2", "issues": []}
  ],
  "ffmpeg": {"name": "ffmpeg", "available": true, "path": "/usr/bin/ffmpeg", "version": null, "issues": []},
  "database": {"name": "database", "available": true, "path": ".franken_whisper/storage.sqlite3", "version": "schema_v2", "issues": []},
  "resources": {"disk_free_bytes": 12345, "disk_total_bytes": 67890, "memory_available_bytes": 11111, "memory_total_bytes": 22222},
  "overall_status": "ok"
}
```

### `runs`

Query persisted run history.

```bash
franken_whisper runs [--limit 20] [--format plain|json|ndjson] [--id <RUN_ID>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--limit <N>` | 20 | Max recent runs |
| `--format` | `plain` | `plain` (table), `json` (pretty), `ndjson` (streaming) |
| `--id <UUID>` | — | Fetch a specific run by ID |

### `sync`

One-way JSONL snapshot export / import with distributed lock safety.

```bash
# full export to JSONL
franken_whisper sync export-jsonl --output ./snapshot [--db <PATH>] [--state-root <PATH>]

# import with a conflict policy
franken_whisper sync import-jsonl --input ./snapshot \
  --conflict-policy reject|skip|overwrite|overwrite-strict \
  [--db <PATH>] [--state-root <PATH>]
```

An export produces `runs.jsonl`, `segments.jsonl`, `events.jsonl`, and `manifest.json` (with row counts and SHA-256 checksums). Import automatically runs `validate_sync()` after the row inserts and reports `validation_ok` plus per-table count comparison in the result JSON.

**Library-level capabilities** that are not yet exposed at the CLI:

- `sync::export_incremental()`: cursor-based delta export (uses `sync_cursor.json`)
- `sync::compress_jsonl()` / `sync::decompress_jsonl()`: gzip-compress or decompress a JSONL file. The import path transparently reads `*.jsonl.gz` when present.
- `sync::validate_sync()`: standalone validation function (auto-called on import, callable directly from the library)

### `tty-audio`

Low-bandwidth audio transport over TTY/PTY links using the `mulaw+zlib+b64` NDJSON protocol.

```bash
# encode audio to NDJSON frames
franken_whisper tty-audio encode --input audio.wav [--chunk-ms 200]

# decode NDJSON frames to WAV
cat frames.ndjson | franken_whisper tty-audio decode --output restored.wav \
  [--recovery fail_closed|skip_missing]

# generate a retransmit plan from a lossy stream
cat frames.ndjson | franken_whisper tty-audio retransmit-plan

# emit individual control frames
franken_whisper tty-audio control handshake
franken_whisper tty-audio control ack --up-to-seq 42
franken_whisper tty-audio control backpressure --remaining-capacity 64
franken_whisper tty-audio control retransmit-request --sequences 1,2,4
franken_whisper tty-audio control retransmit-response --sequences 1,2,4

# automated retransmit loop with strategy escalation
cat frames.ndjson | franken_whisper tty-audio control retransmit-loop --rounds 3

# convenience shorthands
franken_whisper tty-audio send-control handshake|eof|reset
cat frames.ndjson | franken_whisper tty-audio retransmit --rounds 3
```

**Recovery Strategy Escalation.** The retransmit loop escalates across rounds:

```
Simple (1 frame/round) → Redundant (2 frames/round) → Escalate (4 frames/round)
```

**Integrity Checks.** Every frame optionally carries CRC32 and SHA-256 hashes of the raw (pre-compression) audio. Mismatches drop the frame (`skip_missing`) or fail the stream (`fail_closed`).

Full protocol spec: [`docs/tty-audio-protocol.md`](docs/tty-audio-protocol.md). Determinism guarantees: [`docs/tty-replay-guarantees.md`](docs/tty-replay-guarantees.md).

### `tui`

Interactive terminal UI for human operators (feature-gated; requires `--features tui`).

```bash
franken_whisper --features tui tui     # when invoked via cargo run
```

**Features:**

- Live transcription view with auto-scroll
- Speaker labels, timestamps, and confidence scores per segment
- Browsable run history with timing and backend info
- Timeline view of pipeline stages with duration bars
- Event detail panes for individual NDJSON events
- Segment retention cap (10,000 segments, oldest-first drain)
- Vim-style keybindings; focus cycling between panes

Built on the [FrankenTUI](https://github.com/Dicklesworthstone/frankentui) framework.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRANKEN_WHISPER_WHISPER_CPP_BIN` | `whisper-cli` | whisper.cpp binary name or path |
| `FRANKEN_WHISPER_INSANELY_FAST_BIN` | `insanely-fast-whisper` | insanely-fast-whisper entry point |
| `FRANKEN_WHISPER_PYTHON_BIN` | `python3` | Python interpreter for diarization |
| `FRANKEN_WHISPER_HF_TOKEN` | — | HuggingFace token (preferred over `HF_TOKEN`) |
| `HF_TOKEN` | — | HuggingFace token (fallback) |
| `FRANKEN_WHISPER_DIARIZATION_DEVICE` | — | GPU device for the diarization backend |
| `FRANKEN_WHISPER_STATE_DIR` | `.franken_whisper` | State directory root |
| `FRANKEN_WHISPER_DB` | `.franken_whisper/storage.sqlite3` | SQLite database path |
| `FRANKEN_WHISPER_FFMPEG_BIN` | auto | Explicit ffmpeg binary path |
| `FRANKEN_WHISPER_FFPROBE_BIN` | auto | Explicit ffprobe binary path |
| `FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG` | `1` | Auto-provision local ffmpeg/ffprobe when missing (`0`/`false` disables) |
| `FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE` | `0` | Force normalization through ffmpeg even when the built-in decoder can handle it |
| `FRANKEN_WHISPER_NATIVE_EXECUTION` | `0` | Enable in-process native engine dispatch |
| `FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY` | `1` | In bridge-only mode, allow recoverable bridge failures to fall back to native engines |
| `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE` | `primary` | Native engine rollout stage (see below) |
| `XDG_STATE_HOME` | `$HOME/.local/state` | Base for auto-provisioned ffmpeg + per-user tool state |
| `RUST_LOG` | — | tracing filter (e.g. `franken_whisper=debug`) |

**Per-stage budget overrides.** Every pipeline stage has an independent millisecond budget overridable via environment variable:

`FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS`, `…NORMALIZE_MS`, `…PROBE_MS`, `…VAD_MS`, `…SEPARATE_MS`, `…BACKEND_MS`, `…ACCELERATION_MS`, `…ALIGN_MS`, `…PUNCTUATE_MS`, `…DIARIZE_MS`, `…PERSIST_MS`, `…CLEANUP_MS`.

### Cargo Features

| Feature | Description |
|---------|-------------|
| `tui` | Enable the interactive TUI via `ftui` |
| `gpu-frankentorch` | Enable `frankentorch` GPU acceleration |
| `gpu-frankenjax` | Enable `frankenjax` GPU acceleration |

No features are enabled by default.

### Backend Routing

The `auto` backend uses adaptive Bayesian routing:

**Non-diarization priority (deterministic fallback):** `whisper_cpp` → `insanely_fast` → `whisper_diarization`
**Diarization priority (deterministic fallback):** `insanely_fast` → `whisper_diarization` → `whisper_cpp`

Each `auto` run emits a `backend.routing.decision_contract` stage event containing explicit state / action / loss / posterior / calibration terms. When the calibration score drops below 0.3, the Brier score exceeds 0.35, or fewer than 5 observations are available, the router falls back to the deterministic priority list above.

### Native Engine Rollout

Native Rust engine replacements follow a 5-stage rollout under conformance gating:

| Stage | Behavior |
|-------|----------|
| `shadow` | Deterministic bridge execution only; native conformance validated out-of-band |
| `validated` | Deterministic bridge execution only with stricter conformance gating |
| `fallback` | Deterministic bridge execution only; fallback policy and evidence paths hardened |
| `primary` | Native preferred with deterministic bridge fallback (requires `FRANKEN_WHISPER_NATIVE_EXECUTION=1`) |
| `sole` | Native only (requires `FRANKEN_WHISPER_NATIVE_EXECUTION=1`) |

The execution-path metadata on every `backend.ok` event and replay envelope records the active stage so post-hoc analysis can correlate output drift with rollout transitions.

---

## Architecture

```
                  +------------------------------------+
                  |             CLI / Robot            |
                  |          (clap + NDJSON emit)      |
                  +------------------------------------+
                                    |
                  +-----------------v------------------+
                  |       FrankenWhisperEngine         |
                  |          (orchestrator.rs)         |
                  |                                    |
                  |   10-Stage Composable Pipeline:    |
                  |    1. Ingest                       |
                  |    2. Normalize                    |
                  |    3. VAD                          |
                  |    4. Source Separate              |
                  |    5. Backend Execution            |
                  |    6. Accelerate (GPU softmax)     |
                  |    7. Alignment                    |
                  |    8. Punctuation                  |
                  |    9. Diarization                  |
                  |   10. Persist                      |
                  +------------------------------------+
                         |       |       |
    +------------------+  +----------+  +------------------+
    | Backends         |  | Accel    |  | Storage          |
    |                  |  |          |  |                  |
    | whisper.cpp      |  | frank-   |  | fsqlite (WAL)    |
    | insanely-fast    |  |  torch   |  |                  |
    | whisper-diar     |  | frank-   |  | JSONL export     |
    | + paired native  |  |  jax     |  | replay packs     |
    +------------------+  +----------+  +------------------+

  +------------------+   +------------------+   +------------------+
  | TTY Audio        |   | Conformance      |   | Replay           |
  |                  |   |                  |   |                  |
  | mulaw + zlib     |   | 50 ms tolerance  |   | SHA-256 envelope |
  | + base64 NDJSON  |   | cross-engine     |   | + 4-artifact pack|
  | handshake / FEC  |   | comparator       |   | drift detection  |
  +------------------+   +------------------+   +------------------+
```

### Data Flow

1. **Ingest**: materialize input from file, stdin, or mic capture
2. **Normalize**: convert to 16 kHz mono WAV via the built-in Rust decoder (`ffmpeg` fallback for video and exotic formats)
3. **VAD** *(optional)*: Voice Activity Detection skips silence
4. **Source Separate** *(optional)*: vocal isolation for cleaner transcription
5. **Backend**: dispatch to the selected engine (adaptive routing or explicit)
6. **Accelerate** *(optional)*: GPU confidence normalization via `frankentorch` or `frankenjax`
7. **Alignment** *(optional)*: forced alignment for word-level timestamps
8. **Punctuation** *(optional)*: punctuation restoration
9. **Diarization** *(optional)*: speaker identification and labeling
10. **Persist**: write run report, segments, and events to SQLite

Every stage emits `*.start` and `*.ok` events to the NDJSON stream with sequence numbers, RFC-3339 timestamps, and structured payloads. Skipped stages emit `*.skip` so agents can distinguish "not needed" from "failed."

---

## Technical Details

### Bayesian Backend Router

When `--backend auto` is selected, `franken_whisper` runs a formal Bayesian decision contract rather than trying backends in a fixed order.

**State Space (3 states):**
- `all_available`: all three backends found on PATH and responsive
- `partial_available`: 1–2 backends operational
- `none_available`: nothing usable

**Action Space (4 actions):**
- `try_whisper_cpp`, `try_insanely_fast`, `try_diarization` (reordered per request based on `--diarize`)
- `fallback_error`: return a structured error when nothing is available

**Loss Matrix.** The router builds a 3×4 loss matrix (states × actions). Each cell aggregates three weighted factors:

```
cost = (0.45 × latency_cost) + (0.35 × quality_cost) + (0.20 × failure_cost)
```

- **Latency cost** scales with audio duration (short / medium / long buckets) and a per-backend latency proxy
- **Quality cost** depends on backend capability relative to the request (diarization support, GPU availability)
- **Failure cost** is `(1.0 − p_success) × 100`, where `p_success` comes from the Bayesian posterior

**Bayesian Posterior.** Each backend starts with a Beta prior reflecting expected reliability:

| Backend | Prior | Prior mean |
|---------|-------|------------|
| `whisper_cpp` | `Beta(7, 3)` | 0.70 |
| `insanely_fast` | `Beta(6, 4)` | 0.60 |
| `whisper_diarization` | `Beta(5, 5)` | 0.50 |

After each run the posterior is updated with the observed outcome (success / failure).

**Latency Proxy Model.** Backend latency is estimated as a function of audio duration:

```
latency_cost = base + (sqrt(audio_duration_seconds) × multiplier)
```

| Backend | Base (s) | Multiplier (normal) | Multiplier (diarize) |
|---------|----------|---------------------|----------------------|
| `whisper_cpp` | 12.0 | 1.0 | 1.25 |
| `insanely_fast` | 8.0 | 1.0 | 1.25 |
| `whisper_diarization` | 18.0 | 1.0 | 1.25 |

When at least 5 empirical observations are available the estimate blends prior and empirical:
`(0.6 × prior_latency) + (0.4 × empirical_latency)`.

**Quality Proxy Model.** Each backend has a quality score that varies with whether diarization is requested:

| Backend | Quality (normal) | Quality (diarize) |
|---------|------------------|-------------------|
| `whisper_cpp` | 0.84 | 0.55 |
| `insanely_fast` | 0.80 | 0.82 |
| `whisper_diarization` | 0.63 | 0.88 |

`p_success = (α + quality_score × 2.0 + diarize_boost) / (α + β + quality_terms + penalty_terms)`.

A backend with a strong empirical track record can still be penalized when it lacks a capability the request needs (for example, `whisper_cpp` receiving a diarization request).

**Availability Penalties.** The loss matrix applies sharp penalties when backends are unavailable, dominating the loss calculation so the router never selects an unavailable backend even if its quality/latency profile is attractive.

**Policy Versioning.** The routing policy carries an identifier (`backend-selection-v1.0`). Every evidence-ledger entry includes a `loss_matrix_hash` so post-hoc analysis can detect when the policy weights themselves changed between runs.

**Calibration & Fallback.** The router tracks a sliding window of **50** prediction-outcome pairs and computes a Brier score. It falls back to deterministic static priority when any of these hold:

- fewer than 5 observations (insufficient data),
- calibration score < 0.3 (posterior margin too narrow),
- Brier score > 0.35 (predictions don't match reality).

The fallback path **still records the adaptive prediction it would have made** so calibration data continues accumulating even when the static policy is driving the routing. Once calibration improves, the system can recover automatically.

**Evidence Ledger.** Every routing decision is recorded in a circular buffer (capacity: **200** entries) containing the decision ID, trace ID, observed state, chosen action, posterior snapshot, calibration metrics, and fallback flag. Queryable via `robot routing-history`.

### Pipeline Stage Budgets

Each stage runs under an independent millisecond budget. Defaults:

| Stage / Knob | Budget | Rationale |
|--------------|--------|-----------|
| Ingest | 15 s | File I/O or mic capture |
| Normalize | 180 s | Audio decode + resample |
| VAD | 10 s | Lightweight energy detection |
| Source Separate | 30 s | Demucs-style vocal isolation |
| Backend | 900 s (15 min) | Full inference (long audio on CPU) |
| Acceleration | 20 s | GPU confidence normalization |
| Align | 30 s | CTC forced alignment |
| Punctuate | 10 s | Punctuation model inference |
| Diarize | 30 s | Speaker clustering |
| Persist | 20 s | SQLite transaction |
| Probe *(internal)* | 8 s | Backend / ffmpeg / database probing before pipeline starts |
| Cleanup *(internal)* | 5 s | Finalizer total budget |

All 12 budgets are overridable via `FRANKEN_WHISPER_STAGE_BUDGET_<NAME>_MS`. For example, `FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS=1800000` extends the backend stage to 30 minutes. `Probe` and `Cleanup` are tunable knobs without user-visible `PipelineStage` enum variants; they govern pre-pipeline health checks and post-pipeline finalizer execution respectively.

**Automatic Latency Profiling.** After each run, the orchestrator emits an `orchestration.latency_profile` stage event with per-stage timing decomposition. Each stage gets a utilization ratio (`service_ms / budget_ms`) and a tuning recommendation:

| Utilization | Recommendation |
|-------------|----------------|
| ≤ 30% | `decrease_budget_candidate` |
| 30–90% | `keep_budget` |
| ≥ 90% | `increase_budget` (suggest 1.25× current) |

### Replay Envelopes & Drift Detection

Every completed run produces a `ReplayEnvelope` containing SHA-256 hashes:

```
+-------------------------------------------------+
|               ReplayEnvelope                    |
+-------------------------------------------------+
| input_content_hash:  SHA-256(normalized WAV)    |
| backend_identity:    "whisper-cli-v1.7.2"       |
| backend_version:     "1.7.2"                    |
| output_payload_hash: SHA-256(raw backend JSON)  |
| native_rollout_stage: "primary" | …             |
+-------------------------------------------------+
```

Given identical input audio and the same backend version, the output hash should be identical. If it changes between runs, something drifted; the conformance harness can pinpoint whether the drift came from the backend, the native engine rollout, or audio normalization.

### Replay Packs

Self-contained **replay packs** capture everything needed to reproduce and analyze a run:

```
replay_pack/
  env.json                # EnvSnapshot:    OS, arch, backend identity/version, fw version
  manifest.json           # PackManifest:   trace_id, run_id, timestamps, content hashes
  repro.lock              # ReproLock:      routing evidence, replay envelope, request params
  tolerance_manifest.json # ToleranceManifest: schema version, timestamp tolerance
```

| File | Struct | Contents |
|------|--------|----------|
| `env.json` | `EnvSnapshot` | OS, architecture, backend identity/version, franken_whisper version (compile-time `CARGO_PKG_VERSION`) |
| `manifest.json` | `PackManifest` | trace_id, run_id, start/finish timestamps, input/output SHA-256 hashes, segment/event/evidence counts |
| `repro.lock` | `ReproLock` | Routing evidence chain, frozen replay envelope, original backend request, diarize flag |
| `tolerance_manifest.json` | `ToleranceManifest` | Schema version (`tolerance-manifest-v1`), timestamp tolerance, text/speaker exactness flags, native rollout stage, segment/event counts |

All four files are **deterministic**: the same input `RunReport` produces byte-identical output across runs and machines. If a replay pack differs between two runs of the same input, something in the pipeline changed.

### Conformance Harness

The conformance module enforces cross-engine compatibility using a single canonical timestamp tolerance: `CANONICAL_TIMESTAMP_TOLERANCE_SEC = 0.05` (50 ms). Segment comparison counts violations:

| Violation Type | Condition |
|----------------|-----------|
| Text mismatch | Segment text differs at the same index |
| Speaker mismatch | Speaker label differs (optional check) |
| Timestamp violation | `start` / `end` differs by > 50 ms |
| Length mismatch | Different segment counts |

Plus overlap detection (epsilon = 1 µs default), WER approximation (Levenshtein-based), and segment invariant validation: finite timestamps, non-negative values, `start ≤ end`, confidence in `[0.0, 1.0]`, non-empty text.

The conformance corpus in `tests/fixtures/conformance/` covers timestamp boundaries, speaker labels, text trimming, empty segments, overlapping speakers, unicode speaker labels, non-finite invariants, very long segments (15 min), word-level boundary cases, multilingual / code-switching / noisy-overlap audio, and replay envelope drift (`input_hash_drift`, `output_drift`, `backend_identity_drift`, `backend_version_upgrade`, …). The mapping from spec clauses to tests lives in [`tests/COVERAGE.md`](tests/COVERAGE.md).

### Speculative Streaming Architecture

Dual-model streaming for real-time transcription with quality corrections:

```
Audio Stream
  |
  +---> WindowManager (sliding windows with overlap)
  |       |
  |       +---> Fast Model ---> PartialTranscript (status: Pending)
  |       |                        |
  |       |                        v  emit transcript.partial event
  |       |
  |       +---> Quality Model ---> CorrectionDrift analysis
  |                                  |
  |                                  +- drift below tolerance ---> transcript.confirm
  |                                  +- drift above tolerance ---> transcript.retract + corrected text
  |
  +---> CorrectionTracker (adaptive thresholds)
```

The `CorrectionTracker` maintains running drift statistics and adaptively tightens or relaxes confirmation thresholds.

### Audio Normalization Pipeline

Input audio is normalized to 16 kHz mono 16-bit PCM WAV:

```
Input file (any format)
  |
  +-> Built-in Rust decoder (PRIMARY)
  |     symphonia: MP3, AAC, FLAC, WAV, OGG, Vorbis, ALAC, PCM variants
  |     Resampler: linear interpolation to 16 kHz
  |     Channel mixer: stereo/surround -> mono by sample averaging (using actual frame length)
  |     Output: normalized_16k_mono.wav (PCM S16LE)
  |
  +-> ffmpeg subprocess (FALLBACK; only if built-in decoder fails)
        Triggered for: video files, exotic codecs (AC3, DTS, Opus-in-MKV, etc.)
        Args: -hide_banner -loglevel error -y -i <input> -vn -ar 16000 -ac 1 -c:a pcm_s16le <output>
```

**ffmpeg fallback chain:**

1. Explicit binary path (`FRANKEN_WHISPER_FFMPEG_BIN`)
2. System-installed `ffmpeg` on `PATH`
3. Auto-provisioned local binary (Linux x86_64)
4. If all fail: `FW-CMD-MISSING` error with an actionable message

Set `FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE=1` to bypass the built-in decoder and always use ffmpeg.

### Storage Internals

The storage layer uses `fsqlite` (from the FrankenSQLite project) with three core tables plus a key-value metadata table. The schema is at version **3** at HEAD.

```sql
runs     (id PK, started_at, finished_at, backend, input_path,
          normalized_wav_path, request_json, result_json,
          warnings_json, transcript, replay_json, acceleration_json)
          -- all columns NOT NULL; replay_json + acceleration_json default to '{}'

segments (run_id, idx, start_sec, end_sec, speaker, text, confidence,
          PRIMARY KEY (run_id, idx))

events   (run_id, seq, ts_rfc3339, stage, code, message, payload_json,
          PRIMARY KEY (run_id, seq))

_meta    (key TEXT PRIMARY KEY, value TEXT NOT NULL)
          -- holds 'schema_version' => '3' among other metadata
```

**Schema Migrations.** When opening older databases, the storage layer walks forward through the migration ladder:

- **v1 → v2:** add `runs.replay_json` and `runs.acceleration_json` (defaulting to `'{}'`).
- **v2 → v3:** create indexes on hot query paths (recent-runs listing, per-run segment / event lookup) for faster `robot health` and `runs` queries.

Migration steps run safely:

1. Query current journal mode (`PRAGMA journal_mode;`)
2. Switch from WAL to DELETE for DDL reliability
3. Execute `ALTER TABLE ... ADD COLUMN`
4. Restore WAL mode
5. If the migration fails, log a contextual error and leave the database untouched

For severely corrupted databases the recovery path is JSONL-based: export from a known-good source, create a fresh database, import via `sync import-jsonl`.

**Atomic Persistence with Retry.** All inserts are wrapped in a single transaction with up to **8 retry attempts** on `SQLITE_BUSY`, using **linear backoff** of `5 × (attempt + 1)` ms (so 5, 10, 15, …, 40 ms across the retry budget; ~180 ms total worst case). The cancellation token is checked before each `COMMIT`.

**Cancellation-Safe Writes.** The token-checkpoint pattern ensures no partial data reaches the database:

```
SAVEPOINT sp_persist_N
  INSERT INTO runs ...
  INSERT INTO segments ... (N rows)
  INSERT INTO events ... (M rows)
  token.checkpoint()?       -- rolls back if cancelled
RELEASE SAVEPOINT sp_persist_N
```

If the token fires between inserts, the savepoint rolls back cleanly. If the process is killed during `RELEASE`, SQLite's journal recovery handles it on next open. Savepoints (not top-level transactions) let concurrent sessions nest persist calls without deadlocking.

**Concurrent Session Support.** Multiple agents can persist in parallel through `RunStore::begin_concurrent_session("agent_alpha")`, which acquires a named savepoint (`fw_session_{name}`). Session names are validated to be `[A-Za-z0-9_]` only, so no SQL injection via session names is possible.

**WAL Mode & Storage Configuration:**

| `PRAGMA` | Value | Purpose |
|----------|-------|---------|
| `journal_mode` | `WAL` | Write-Ahead Logging for concurrent readers |
| `busy_timeout` | `5000` (5 s) | Wait for locks before returning `SQLITE_BUSY` |

**Storage Diagnostics.** `StorageDiagnostics` exposes:

| Field | Description |
|-------|-------------|
| `page_count` | Total database pages |
| `page_size` | Bytes per page (typically 4096) |
| `journal_mode` | Current mode (`wal`, `delete`) |
| `wal_checkpoint` | WAL status: busy flag, log frames, checkpointed frames |
| `freelist_count` | Unused pages available for reuse |
| `integrity_check` | `"ok"` when the database passes `PRAGMA integrity_check` |

Accessible via `robot health`, which includes database diagnostics in the health report.

### Backend Bridge Adapters

Each backend has a bridge adapter that spawns an external process and parses its output. Adapters normalize diverse output formats into a uniform `TranscriptionResult`.

**whisper.cpp Bridge.** Spawns `whisper-cli` (or `FRANKEN_WHISPER_WHISPER_CPP_BIN`) and parses the JSON output file, handling multiple layouts (`transcription`, `segments`, `chunks`) and extracting word-level timestamps from nested `words` arrays when requested.

**insanely-fast-whisper Bridge.** Spawns `insanely-fast-whisper` (or `FRANKEN_WHISPER_INSANELY_FAST_BIN`). Shares JSON segment extraction with the whisper.cpp bridge. Falls back to joining segment texts if the root `"text"` key is missing.

**whisper-diarization Bridge.** Spawns a Python script via `python3` (or `FRANKEN_WHISPER_PYTHON_BIN`). Parses two output files: a `.txt` file with the full transcript and a `.srt` file with speaker-labeled subtitles. The SRT parser handles both comma- and dot-separated timing, and recognizes speaker patterns `[SPEAKER_00]`, `SPEAKER_00:`, `spk0:`, `s0:`. Each bridge invocation logs through `render_command_for_log()`, which redacts `--hf-token` values so the HuggingFace token never appears in tracing output.

### Native Engine Rollout Governance

Native Rust replacements for the bridge adapters ship behind a 5-stage rollout with conformance gating at every promotion. This prevents a buggy native engine from silently degrading transcription quality.

**Rollout Stages:**

```
Shadow → Validated → Fallback → Primary → Sole
  |          |          |          |        |
  |          |          |          |        +-- Native only, bridge removed
  |          |          |          +-- Native preferred, bridge fallback on error
  |          |          +-- Bridge preferred, native fallback hardened
  |          +-- Bridge only, stricter conformance gating
  +-- Bridge only, native conformance validated out-of-band
```

**Conformance Gate.** At each stage transition the harness compares native vs. bridge output on a fixture corpus. A native engine that produces timestamps > 50 ms different from the bridge adapter for the same audio is blocked from promotion. The harness also validates backend version drift, replay envelope drift, and segment invariants.

**Segment Validation Rules.**

- Timestamps must be finite (no NaN, no infinity)
- `start_sec` and `end_sec` must be non-negative
- `start ≤ end`
- No overlapping segments (configurable epsilon: 1 µs default)
- Confidence scores in `[0.0, 1.0]`
- Text non-empty

**Runtime Control.** Two environment variables jointly control native engine behavior:

- `FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE`: which stage the deployment is at
- `FRANKEN_WHISPER_NATIVE_EXECUTION`: whether native dispatch is enabled at runtime (`0`/`1`)

Both must agree for native engines to actually execute. Setting `NATIVE_EXECUTION=1` with stage `shadow` has no effect; the stage gate prevents native execution regardless of the runtime flag.

**Execution Path Metadata.** Every `backend.ok` event and replay envelope includes explicit execution-path metadata: `implementation` (bridge / native), `execution_mode`, `native_rollout_stage`, and `native_fallback_error` (populated when native fails and bridge recovers). The contract is documented in [`docs/native_engine_contract.md`](docs/native_engine_contract.md).

### Run Report Structure

Every transcription produces a `RunReport`, the complete record of what happened:

```
RunReport
  run_id:              "fw-run-abc123"
  trace_id:            "1710000000000-random64"
  started_at_rfc3339:  "2026-04-25T06:00:00Z"
  finished_at_rfc3339: "2026-04-25T06:00:05Z"
  input_path:          "/path/to/audio.mp3"
  normalized_wav_path: "/tmp/normalized_16k_mono.wav"
  request:             TranscribeRequest { ... }      -- full input parameters
  result:              TranscriptionResult { ... }     -- backend output
    transcript:        "Hello world..."
    segments:          [TranscriptionSegment { ... }]  -- timed chunks
    language:          Some("en")
    acceleration:      AccelerationReport { ... }      -- confidence normalization metadata
  events:              [RunEvent { ... }]              -- pipeline stage events (sequenced)
  warnings:            ["..."]                         -- non-fatal issues
  evidence:            [Value { ... }]                 -- routing decision evidence
  replay:              ReplayEnvelope { ... }          -- SHA-256 hashes for deterministic replay
```

The report is persisted to SQLite (split across `runs`, `segments`, and `events`) and optionally emitted as JSON via `--json` or as streaming NDJSON events in robot mode.

### Robot Event Streaming Architecture

In robot mode the pipeline emits events in real time via an `mpsc` channel:

```
                  +-------------------+
                  |  CLI (main.rs)    |
                  |                   |
                  |  event_rx poll    |<--+
                  |  (every 40 ms)    |   |
                  +-------------------+   |
                         |                |   mpsc channel
                         v                |
                  +-------------------+   |
                  |  stdout (NDJSON)  |   |   StreamedRunEvent { run_id, event }
                  |  one line per     |   |
                  |  event            |   |
                  +-------------------+   |
                                          |
                  +-------------------+   |
                  |  Pipeline Worker  |---+
                  |  (background      |
                  |   thread)         |
                  +-------------------+
```

The CLI thread polls the receive end of the channel every 40 ms, formatting each event as a single NDJSON line on stdout. The pipeline worker thread runs `transcribe_with_stream()` which emits `StreamedRunEvent` wrappers containing `(run_id, RunEvent)` pairs. When the worker completes, the CLI emits a final `run_complete` or `run_error` event. If the worker thread itself panics, the CLI emits a structured `run_error` envelope rather than printing a Rust panic message; the contract on stdout is preserved unconditionally.

**Schema Contract Guarantees:**

| Guarantee | Enforcement |
|-----------|-------------|
| `event` and `schema_version` present on every event | Hardcoded in all `emit_*` functions |
| `seq` strictly increasing per run | Auto-incremented from `events.len()` |
| `ts` non-decreasing per run | Generated from `Utc::now().to_rfc3339()` |
| `run_complete` is always the final success event | Emitted only after pipeline returns |
| Stage events follow pipeline order | Orchestrator executes stages sequentially |
| Worker-thread panics emit `run_error`, not a panic | `emit_robot_error_from_fw` wraps panic recovery |

### TTY Handshake Protocol

The TTY audio protocol begins with version and codec negotiation before any audio flows:

```
Encoder                                         Decoder
   |                                               |
   |-- Handshake {                                 |
   |     min_version: 1,                           |
   |     max_version: 2,                           |
   |     supported_codecs: ["mulaw+zlib+b64"]      |
   |   } -------------------------------------->   |
   |                                               |
   |   <---------------------------------------    |
   |       HandshakeAck {                          |
   |         negotiated_version: 1,                |
   |         negotiated_codec: "mulaw+zlib+b64"    |
   |       }                                       |
   |                                               |
   |-- AudioFrame { seq: 0, ... } ------------>    |
   |-- AudioFrame { seq: 1, ... } ------------>    |
   |           ...                                 |
   |-- SessionClose { last_data_seq: N } ----->    |
   |                                               |
   |   <--- Ack { up_to_seq: N }                   |
```

**Version Negotiation.** The encoder advertises a supported range (currently 1–2). The decoder picks the highest version both support. If ranges don't overlap, the handshake fails.

**Codec Negotiation.** Currently `"mulaw+zlib+b64"` is the only defined codec. The protocol is extensible; future codecs (for example `opus+b64`) can be added by extending the `supported_codecs` array.

**Session Close.** The encoder sends `SessionClose { reason, last_data_seq }` to signal end of stream. The decoder verifies it has received all frames up to `last_data_seq`. Missing frames trigger the retransmit protocol; sequence gaps detected at finalize raise an explicit error rather than truncating silently.

### Retransmit Loop Determinism

The retransmit system is designed to be **fully deterministic** for testing and debugging:

- Given the same frame buffer and the same loss pattern, the output and report are byte-identical across runs.
- No timing dependencies; `timeout_ms` is advisory (used for reporting) with no actual sleeps or waits.
- Frame recovery proceeds in sequence-number order (not arrival order).
- Strategy escalation follows a fixed chain: `Simple → Redundant → Escalate`.
- `inject_loss()` resets all prior recovery state, ensuring clean separation between test scenarios.

This determinism enables comprehensive fuzz testing of the retransmit protocol without flaky timing-dependent failures.

### ffmpeg Auto-Provisioning

When ffmpeg is needed but not installed, `franken_whisper` can automatically download a static binary (Linux x86_64 only):

**Source:** `https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz`

**Flow:**

1. Check `FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG` (default: `1` / enabled)
2. Check if a provisioned binary already exists at `{state_root}/tools/ffmpeg/bin/ffmpeg`
3. If missing: download via `curl -fsSL` or `wget --quiet` (whichever is available)
4. Extract from `.tar.xz` via `tar -xf` into a tmpdir
5. Copy `ffmpeg` and `ffprobe` to `{state_root}/tools/ffmpeg/bin/`
6. Set executable permissions (`chmod 755`)
7. Verify the extracted binaries are executable

**Safeguards:**

- 180-second download timeout
- Atomic install: a tmpdir is used during extraction, then moved into place
- Failure is non-fatal: log a warning and continue (the built-in Rust decoder handles most audio formats anyway)
- Honors `XDG_STATE_HOME` for the install root
- Disable entirely with `FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG=0`
- Non-Linux / non-x86_64 platforms get an actionable error explaining how to install ffmpeg manually

### Graceful Shutdown

```
Ctrl+C
  |
  v
ctrlc handler
  | sets AtomicBool (SeqCst)
  v
CancellationToken.checkpoint()
  | returns Err(Cancelled) at next checkpoint
  v
Pipeline stage catches Cancelled
  | rolls back any in-progress transaction (savepoint)
  | cleans up temp files via finalizers
  | kills active subprocesses
  v
CLI exits with code 130 (128 + SIGINT)
```

The cancellation token is threaded through every pipeline stage, including the inner extraction loops in word-level timestamp processing, so an in-flight backend can be interrupted between checkpoints rather than after the entire backend invocation completes.

### Error Codes

| Code | Meaning |
|------|---------|
| `FW-IO` | I/O error (file not found, permission denied) |
| `FW-JSON` | JSON serialization / deserialization failure |
| `FW-CMD-MISSING` | Required external binary not found on `PATH` |
| `FW-CMD-FAILED` | Backend subprocess exited with non-zero status |
| `FW-CMD-TIMEOUT` | Backend subprocess exceeded timeout |
| `FW-BACKEND-UNAVAILABLE` | No suitable backend found for the request |
| `FW-INVALID-REQUEST` | Malformed or contradictory request parameters |
| `FW-STORAGE` | SQLite persistence error |
| `FW-UNSUPPORTED` | Requested feature not available |
| `FW-MISSING-ARTIFACT` | Expected output file not produced by backend |
| `FW-CANCELLED` | Operation cancelled via token or Ctrl+C |
| `FW-STAGE-TIMEOUT` | Pipeline stage exceeded its budget |

**Robot Error Code Mapping.** In robot mode the 12 internal variants collapse into 6 robot-specific codes for simpler agent handling:

| Robot Code | Internal Variants | When |
|------------|-------------------|------|
| `FW-ROBOT-TIMEOUT` | `CommandTimedOut`, `StageTimeout` | Any timeout during pipeline execution |
| `FW-ROBOT-BACKEND` | `BackendUnavailable` | No suitable backend found |
| `FW-ROBOT-REQUEST` | `InvalidRequest` | Malformed CLI arguments |
| `FW-ROBOT-STORAGE` | `Storage` | SQLite persistence failure |
| `FW-ROBOT-CANCELLED` | `Cancelled` | Ctrl+C or deadline cancellation |
| `FW-ROBOT-EXEC` | All others (`Io`, `Json`, `CommandMissing`, `CommandFailed`, `Unsupported`, `MissingArtifact`) | General execution failure |

Even argument-parsing failures (e.g. mutually-exclusive input modes) produce a structured `run_error` envelope on stdout in robot mode. There is no path where the robot interface degrades to human-readable stderr.

### Engine Trait & Backend Capabilities

Every backend (bridge or native) implements the `Engine` trait:

```rust
pub trait Engine: Send + Sync {
    fn name(&self) -> &'static str;          // "whisper.cpp", "insanely-fast-whisper", etc.
    fn kind(&self) -> BackendKind;           // WhisperCpp, InsanelyFast, WhisperDiarization
    fn capabilities(&self) -> EngineCapabilities;
    fn is_available(&self) -> bool;          // PATH probe via `which` crate
    fn run(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
    ) -> FwResult<TranscriptionResult>;
}
```

**EngineCapabilities** describe what each backend supports:

| Capability | whisper.cpp | insanely-fast | whisper-diarization |
|------------|:-----------:|:-------------:|:-------------------:|
| `supports_diarization` | false (TinyDiarize exists separately) | true (with HF token) | true |
| `supports_translation` | true | true | false |
| `supports_word_timestamps` | true | true | false |
| `supports_gpu` | true (CUDA / Metal) | true (CUDA / MPS) | true (CUDA) |
| `supports_streaming` | false | false | false |

These capabilities feed the Bayesian router's quality proxy: a backend that doesn't support a requested feature gets a lower quality score for that request.

**Backend Availability Probing.** Availability is checked via the `which` crate; each backend can be overridden with an environment variable, in which case the override path is checked directly.

### Subprocess Execution & Cancellation

The process module provides three execution modes with increasing safety guarantees:

**`run_command`** (fire and forget with captured output):
```
Spawn child → wait → return (stdout, stderr, exit_status)
```

**`run_command_with_timeout`** (bounded execution):
```
Spawn child → poll exit every 50 ms → if timeout: kill + return TimeoutError
```

**`run_command_cancellable`** (full cooperative cancellation):
```
Spawn child
  loop:
    poll child.try_wait()
    if exited: return output
    token.checkpoint()?     -- if cancelled: kill child, return Err(Cancelled)
    sleep 50 ms
  hard_timeout safety net: kill child regardless
```

The 50 ms poll interval bounds cancellation response time. The child process receives `SIGKILL` (not `SIGTERM`), ensuring immediate termination of backends that may be doing heavy GPU inference. Stdout and stderr are drained on dedicated threads to prevent pipe deadlock when subprocesses produce large output.

**Secret Redaction.** Command lines are routed through `render_command_for_log()` before being emitted into traces. The renderer redacts known secret-bearing flag values, most notably `--hf-token` for the diarization bridge, so HuggingFace tokens never appear in tracing output, log files, or test snapshots.

### Mu-Law Audio Encoding

The TTY audio codec uses mu-law compression, a standard telephony algorithm that compresses 16-bit PCM to 8-bit with logarithmic companding. The actual transcoding is delegated to `ffmpeg` (`-f mulaw -ar 8000 -ac 1`) so the codec implementation isn't carried in the Rust source. `franken_whisper` instead handles framing, sequencing, integrity checks, base64 encoding, and zlib compression on top of the mu-law byte stream.

The codec definition itself is well-known and useful to understand for debugging wire traces:

**Encoding (linear PCM → mu-law):**

```
1. Input: 16-bit signed integer sample
2. Clamp to [-32635, 32635] (mu-law representable range)
3. Add bias: sample = |sample| + 132
4. Find segment: position of highest set bit (determines compression curve)
5. Extract mantissa: 4 bits from the segment position
6. Combine: segment (3 bits) + mantissa (4 bits) + sign (1 bit) = 8 bits
7. Invert all bits (wire format convention)
```

**Decoding (mu-law → linear PCM):**

```
1. Invert all bits
2. Extract sign, segment, mantissa
3. Reconstruct: ((mantissa << 3) + bias) << (segment + 1) - bias
4. Apply sign
```

This achieves ~2:1 compression (16-bit → 8-bit) while preserving speech intelligibility. The full pipeline:

```
Raw PCM (16-bit) → mu-law (8-bit) → zlib compress → base64 encode → NDJSON line
```

The inverse pipeline runs on decode. CRC32 and SHA-256 integrity hashes are computed on the raw (pre-compression) audio bytes, so corruption at any stage is detected.

### TTY Audio Wire Format

Each audio frame is a single NDJSON line:

```json
{
  "protocol_version": 1,
  "seq": 42,
  "codec": "mulaw+zlib+b64",
  "sample_rate_hz": 16000,
  "channels": 1,
  "payload_b64": "eJztwTEBAAAAwqD1T20ND...",
  "crc32": 3141592653,
  "payload_sha256": "a1b2c3d4e5f6..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `protocol_version` | u32 | yes | Protocol version (1 = audio, 2 = audio + transcript control) |
| `seq` | u64 | yes | Strictly increasing sequence number |
| `codec` | string | yes | Compression codec identifier |
| `sample_rate_hz` | u32 | yes | Mu-law audio sample rate (8000 — telephony standard for mu-law) |
| `channels` | u8 | yes | Channel count (always 1 for mono) |
| `payload_b64` | string | yes | Base64-encoded compressed audio data |
| `crc32` | u32 | optional | CRC32 of raw (pre-compression) audio bytes |
| `payload_sha256` | string | optional | SHA-256 hex digest of raw audio bytes |

Control frames use the same NDJSON line format but with a `"type"` field instead of `"seq"`:

```json
{"type":"handshake","min_version":1,"max_version":2,"supported_codecs":["mulaw+zlib+b64"]}
{"type":"ack","up_to_seq":42}
{"type":"backpressure","remaining_capacity":64}
{"type":"session_close","reason":"complete","last_data_seq":100}
```

### TTY Audio: Adaptive Bitrate & FEC

The `AdaptiveBitrateController` monitors link quality in real time and adjusts compression dynamically:

| Frame Loss Rate | Link Quality | Compression | Critical Frame FEC |
|-----------------|--------------|-------------|--------------------|
| < 1% | High | zlib level 1 (fast) | 1× (no duplication) |
| 1% – 10% | Moderate | zlib level 6 (default) | 2× |
| > 10% | Poor | zlib level 9 (best) | 3× |

**Critical Frame FEC.** Control frames essential for protocol correctness (handshake, session_close, ack) are emitted multiple times based on current link quality. Under 10% loss, every handshake frame is transmitted 3× to ensure at least one copy arrives. With independent frame loss at rate `p`, the probability all `k` copies are lost is `p^k`.

**Link Quality Assessment.**

```
frame_loss_rate = frames_lost / frames_sent
link_quality   = 1.0 - frame_loss_rate
```

Quality transitions trigger compression changes on subsequent frames; this gives automatic adaptation without manual tuning.

**Transcript Streaming over TTY (Protocol v2).** Beyond raw audio, the TTY protocol supports real-time transcript streaming via three control frame types:

| Frame Type | Direction | Purpose |
|------------|-----------|---------|
| `TranscriptPartial` | sender → receiver | Speculative partial transcript from fast model |
| `TranscriptRetract` | sender → receiver | Retract a previous partial (quality model disagrees) |
| `TranscriptCorrect` | sender → receiver | Send corrected transcript from quality model |

These frames carry `TranscriptSegmentCompact` payloads, a wire-efficient representation with single-letter field names (`s` / `e` / `t` / `sp` / `c` for start / end / text / speaker / confidence) to minimize bandwidth.

**Telemetry Counters.** The decode path tracks comprehensive telemetry:

- `frames_decoded`: successfully decoded audio frames
- `gaps`: sequence number discontinuities (with expected / actual pairs)
- `duplicates`: repeated sequence numbers (second copy discarded)
- `integrity_failures`: CRC32 / SHA-256 mismatches (frame dropped)
- `dropped_frames`: total frames discarded due to policy (integrity + duplicates)

### Confidence Normalization (Acceleration)

The acceleration stage normalizes per-segment confidence scores into a proper probability distribution. Raw backend confidences are uncalibrated and incomparable across engines, so normalization is necessary for meaningful cross-backend analysis.

**Algorithm:**

1. Extract confidence values from all segments
2. Replace missing / invalid values (NaN, infinity, zero, negative) with a text-length-based baseline: `ln(1 + char_count) + 1.0`
3. Compute pre-mass: `sum(confidences)` before normalization
4. Apply softmax normalization (GPU path via `frankentorch` / `frankenjax`, or CPU fallback)
5. Compute post-mass: `sum(normalized)` (should equal 1.0)
6. Record both masses in `AccelerationReport` for validation

**Numerically Stable Softmax (CPU path):**

```
max_val = max(finite values)               -- prevent overflow
exps[i] = exp(value[i] - max_val)          -- shift by max
output[i] = exps[i] / sum(exps)            -- normalize to sum = 1.0
```

Non-finite values (NaN, infinity) map to 0.0 in the output. If the sum is near zero (all values degenerate), the result falls back to a uniform distribution `1/N`. The `layer_norm_cpu` helper clamps its epsilon floor to a safe minimum to prevent division by zero on pathological inputs.

**Acceleration Paths:**

| Path | Trigger | Method |
|------|---------|--------|
| `frankentorch` | `--features gpu-frankentorch` | Tensor softmax via `FrankenTorchSession` |
| `frankenjax` | `--features gpu-frankenjax` | JAX-based normalization via `fj_api` |
| CPU fallback | no GPU features | Numerically stable softmax with NaN/inf guards |

The fallback chain is `frankentorch → frankenjax → CPU`.

### Speculative Streaming Internals

The speculative streaming system combines dual-model execution with Bayesian window sizing, drift quantification, and deterministic fallback.

**WindowManager.** Divides the audio stream into overlapping windows. Each window gets a unique `window_id`, an SHA-256 hash of its audio content, and slots for both the fast and quality model results. Window sizes range from 1,000 ms to 30,000 ms, with the default starting at the configured `--speculative-window-ms` (default: 3,000 ms).

**CorrectionDrift Metrics.** When the quality model disagrees with the fast model, the system quantifies the disagreement using four metrics:

| Metric | Meaning | Typical Range |
|--------|---------|---------------|
| `wer_approx` | Levenshtein-based approximate Word Error Rate | 0.0 (identical) – 1.0 (completely different) |
| `confidence_delta` | Absolute difference in mean segment confidence | 0.0 – 1.0 |
| `segment_count_delta` | `quality_count - fast_count` | −N – +N |
| `text_edit_distance` | Levenshtein distance on concatenated transcript text | 0 – unbounded |

**CorrectionTolerance.** A partial transcript is **confirmed** when all drift metrics fall within tolerance, and **retracted** (with correction) when any metric exceeds its threshold:

| Threshold | Default Value | Meaning |
|-----------|---------------|---------|
| `max_wer` | 0.1 (10%) | Maximum word error rate before retraction |
| `max_confidence_delta` | 0.15 | Maximum confidence difference |
| `max_edit_distance` | 50 characters | Maximum text edit distance |

**SpeculationWindowController (Adaptive Sizing).** Uses the same alien-artifact engineering contract as the backend router:

- **State space:** observed correction rate (fraction of windows needing correction)
- **Posterior:** `Beta(α, β)` distribution over expected correction rate
- **Calibration:** sliding window of 20 prediction-outcome pairs with Brier score tracking
- **Fallback trigger:** Brier score > 0.25 with ≥ 10 observations

The controller adjusts window size based on correction patterns:

| Pattern | Action | Rationale |
|---------|--------|-----------|
| High correction rate (> 25%) | Shrink window by `step_ms` | Smaller windows reduce correction latency |
| Low correction rate (< 6.25%) | Grow window by `step_ms` | Larger windows reduce overhead |
| Runaway corrections (> 75%) | Force minimum window size | System is clearly struggling |
| 20 consecutive zero corrections | Shrink (counterintuitive) | May be over-tolerant; tighten to validate |
| High WER (> 12.5%) | Shrink window | Fast model consistently wrong at this scale |

**ConcurrentTwoLaneExecutor.** Runs both models in parallel lanes with independent timeout budgets. Results are collected asynchronously, and the faster result (always the fast model by design) is emitted immediately while the quality result triggers correction logic when it arrives. Thread panics in either lane are propagated as `FwError`s rather than silently hanging the pipeline, and mutex poisoning is recovered gracefully.

### Built-In Audio Decoder Internals

The built-in normalizer (`normalize_to_wav_with_builtin_decoder`) is a pure-Rust audio pipeline that produces whisper-compatible WAV without spawning any subprocess.

**Format Detection.** `symphonia::get_probe().format()` uses file extension hints and magic-byte probing to identify the container. Supported containers include MP3 (MPEG Layer III), MP4/M4A (AAC), FLAC, WAV/RIFF, OGG (Vorbis), and WavPack.

**Decoding Loop:**

```
for each packet in format_reader:
    decoded = codec_decoder.decode(packet)
    convert decoded samples to f32
    if multi-channel: average all channels -> mono (using actual frame length)
    append to sample buffer
```

Sample conversion handles `i16`, `i32`, `f32`, and `f64` source formats. Multi-channel audio is mixed to mono by averaging corresponding samples, using the actual frame length rather than the nominal channel count to handle truncated frames safely.

**Resampling.** A linear interpolation resampler converts from the source sample rate (commonly 44.1 kHz or 48 kHz) to whisper's required 16 kHz:

```
ratio = src_rate / dst_rate
for each output sample i:
    position = i * ratio
    left  = input[floor(position)]
    right = input[ceil(position)]
    output[i] = left + frac(position) * (right - left)
```

Computationally lightweight (no FFT, no filter bank) while sufficient for speech. Whisper models tolerate minor resampling artifacts well.

**WAV Output.** The final mono f32 buffer is quantized to 16-bit signed PCM via clamp-and-round, then written as a standard RIFF WAV header + raw PCM data. The output is always `normalized_16k_mono.wav` in the work directory.

**Empty-input guard.** `normalize_cpu` (and the builtin decoder) guard the empty-input edge case so silent fixtures and zero-length captures produce a deterministic error instead of an out-of-bounds panic.

### Sync Architecture

The sync module provides one-way JSONL snapshot export/import with distributed lock safety.

**Lock Protocol.** Before any export or import, a JSON lock file is created at `{state_root}/locks/sync.lock`:

```json
{"pid": 12345, "created_at_rfc3339": "2026-04-25T12:00:00Z", "operation": "export"}
```

The lock is **kept alive while the owning PID is still running** and **age-gated** when the PID is unknown; stale locks are archived with a reason suffix and a new lock is acquired. PID-liveness checks handle Windows (`tasklist` errors treated as unknown rather than dead), Linux EPERM on `kill -0` (treated as alive: the process exists but isn't ours), and missing `/proc` entries. The combined effect is that long sync jobs never get killed by spurious lock-stealing and crashed jobs never leave permanent locks.

**Export Format.** Four files:

```
snapshot/
  runs.jsonl          # one JSON object per run
  segments.jsonl      # one JSON object per segment
  events.jsonl        # one JSON object per event
  manifest.json       # row counts + SHA-256 checksums (always uncompressed)
```

**Incremental Export (library API).** Full exports re-dump the entire database. For large databases, `sync::export_incremental()` exposes a cursor-based delta export at the library level (the CLI currently always does a full export). Incremental mode uses a cursor file (`sync_cursor.json`) tracking the last export timestamp and run ID. Only runs created after the cursor are exported. The cursor uses a composite `(finished_at, run_id)` key for deterministic deduplication, ensuring resume-safety across interrupted exports.

**JSONL Compression (library API).** `sync::compress_jsonl()` / `sync::decompress_jsonl()` gzip-compress or decompress a JSONL file at the library level. The **import path transparently detects `.gz` variants** at runtime, so a snapshot whose JSONL files are gzipped out-of-band (for example via `gzip backup/*.jsonl`) is fully importable through the CLI without any flag; the import code switches to `GzDecoder` when it sees a `.gz` extension. The manifest stays uncompressed so it can be inspected without decompression.

**Sync Validation.** After import, `validate_sync()` compares the database state against the imported JSONL files, checking for row count mismatches and checksum mismatches. End-to-end integrity verification.

**Conflict Policies.**

| Policy | Behavior on duplicate `run_id` |
|--------|-------------------------------|
| `reject` | Fail the entire import |
| `skip` | Silently skip existing runs |
| `overwrite` | Replace conflicting `runs` rows, fail closed if child-row mutation is needed |
| `overwrite-strict` | Verified strict replacement including child-row updates (delete+insert) and stale child-row pruning |

**Parallel Pipeline.** Large imports run with a parallel segment / event import pipeline and per-stage progress tracking; off-by-one accounting bugs in the progress display were eliminated, so the displayed counts always match the row counts in the final commit.

### PipelineBuilder Fluent API

The pipeline is composed using a builder pattern rather than a hardcoded stage list:

```rust
// Default 10-stage pipeline
let config = PipelineBuilder::default_stages().build()?;

// Custom pipeline (skip stages you don't need)
let config = PipelineBuilder::new()
    .stage(PipelineStage::Ingest)
    .stage(PipelineStage::Normalize)
    .stage(PipelineStage::Backend)
    .stage(PipelineStage::Persist)
    .build()?;

// Remove stages from defaults
let config = PipelineBuilder::default_stages()
    .without(PipelineStage::Vad)
    .without(PipelineStage::Diarize)
    .build()?;
```

`build()` validates the pipeline: `Ingest` must come before `Normalize`, `Normalize` before `Backend`, and `Persist` (if present) must be last. `build_unchecked()` skips validation for testing.

### FinalizerRegistry & Bounded Cleanup

The `FinalizerRegistry` ensures resources are cleaned up even on cancellation or panic:

```rust
enum Finalizer {
    TempDir(PathBuf),       // Remove temporary directory
    Custom(Box<dyn Fn()>),  // User-provided cleanup function
    Process(u32),           // Kill subprocess by PID
}
```

**Execution semantics:**

- Finalizers run in **LIFO order** (last registered, first cleaned up).
- `run_all_bounded(budget_ms)` enforces a per-finalizer timeout, so a hung cleanup cannot block shutdown indefinitely.
- The default cleanup budget is 5 seconds (from the pipeline's `Cleanup` stage budget).
- Process finalizers send `SIGKILL` (immediate termination, no graceful shutdown for subprocesses).
- Temp directory finalizers use `std::fs::remove_dir_all`.
- If a finalizer panics, remaining finalizers still run (`catch_unwind`).

### Dependency Graph

```
franken_whisper
  |
  +-- asupersync          (crates.io 0.3.1)   Cancel-correct orchestration
  +-- franken-kernel      (crates.io 0.3.1)   Budget, TraceId, time utilities
  +-- franken-evidence    (crates.io 0.3.1)   Evidence ledger primitives
  +-- franken-decision    (crates.io 0.3.1)   Decision contract framework
  |
  +-- fsqlite             (path)              Pure-Rust SQLite implementation
  +-- fsqlite-types       (path)              Core SQLite value types
  |
  +-- [optional] ftui     (path, feature: tui)              Terminal UI framework
  +-- [optional] ft-api   (path, feature: gpu-frankentorch) GPU tensor operations
  +-- [optional] ft-core  (path, feature: gpu-frankentorch)
  +-- [optional] fj-api   (path, feature: gpu-frankenjax)   JAX-based GPU compute
  +-- [optional] fj-core  (path, feature: gpu-frankenjax)
```

**Third-party dependencies (non-optional):**

| Crate | Version | Purpose |
|-------|---------|---------|
| `clap` | 4.5 | CLI argument parsing with derive macros |
| `serde` + `serde_json` | 1.0 | JSON serialization / deserialization |
| `chrono` | 0.4 | Timestamp handling (RFC-3339) |
| `uuid` | 1.15 | Run ID generation (v4 random) |
| `sha2` | 0.10 | SHA-256 content hashing |
| `crc32fast` | 1.4 | CRC32 integrity checksums |
| `base64` | 0.22 | Base64 encoding for TTY wire format |
| `flate2` | 1.1 | Zlib compression (TTY audio, JSONL sync) |
| `symphonia` | 0.5 | Native audio decoding (MP3, AAC, FLAC, OGG, WAV) |
| `hound` | 3.5 | WAV file writing |
| `which` | 7.0 | Backend binary `PATH` discovery |
| `ctrlc` | 3.4 | Ctrl+C signal handling |
| `tracing` | 0.1 | Structured logging and diagnostics |
| `tracing-subscriber` | 0.3 | tracing subscribers (`env-filter`, `json`) |
| `thiserror` | 2.0 | Error type derive macros |
| `tempfile` | 3.17 | Temporary file / directory management |

### Clippy & Lint Configuration

The codebase enforces strict linting beyond `#![forbid(unsafe_code)]`:

```toml
[lints.clippy]
enum_glob_use = "warn"              # No wildcard enum imports
explicit_into_iter_loop = "warn"    # Use .iter() not .into_iter() on references
explicit_iter_loop = "warn"         # Prefer for x in &collection
flat_map_option = "warn"            # Use .flatten() instead of .flat_map(|x| x)
implicit_clone = "warn"             # Prefer .clone() over implicit copies
semicolon_if_nothing_returned = "warn"  # Consistent semicolons on unit functions
unused_self = "warn"                # Flag methods that don't use self
```

`cargo clippy --all-targets -- -D warnings` is run in CI and promotes these warnings to hard errors.

### Why These Design Decisions?

**Why Bayesian routing over multi-armed bandits?**
Multi-armed bandits (UCB, Thompson sampling) optimize for a single reward signal. Backend selection involves multiple conflicting objectives (latency, quality, failure risk) that vary per request (diarization changes the optimal backend). The Bayesian decision contract with an explicit loss matrix handles this naturally: each (state, action) pair has a multi-factor cost, and the posterior captures per-backend reliability independent of the cost model. Bandits would have to collapse the multi-factor cost into a single scalar reward, losing the ability to reason about tradeoffs.

**Why savepoints instead of top-level transactions?**
Top-level `BEGIN/COMMIT` transactions don't nest in SQLite. If a caller is already inside a transaction (for example, a concurrent session), a nested `BEGIN` either fails or starts an implicit savepoint depending on the driver. Explicit `SAVEPOINT` / `RELEASE` always nest correctly and make the isolation boundaries visible in the code. The naming convention (`sp_persist_N`, `fw_session_name`) provides debuggability when inspecting WAL state.

**Why mu-law over Opus for TTY audio?**
Opus is a superior audio codec, but adding it would mean either an FFI binding to `libopus` (impossible under `#![forbid(unsafe_code)]`) or a pure-Rust Opus implementation (a large, non-trivial dependency). Mu-law is a 60-year-old telephony standard that *every* ffmpeg build supports out of the box, has a fixed 2:1 compression ratio (16-bit PCM → 8-bit mu-law at 8 kHz), and produces byte streams with substantial statistical redundancy that compresses well with zlib on top. The codec conversion itself is delegated to ffmpeg via `-f mulaw -ar 8000 -ac 1`, so the Rust side stays bit-manipulation-free. The bandwidth overhead vs. Opus is roughly 30% in exchange for zero new build dependencies and an extensible codec-negotiation protocol that can add `opus+b64` later without breaking deployments.

**Why not whisper-rs (Rust FFI bindings)?**
`whisper-rs` provides Rust bindings to whisper.cpp via FFI, which is necessarily `unsafe`. `franken_whisper` orchestrates whisper.cpp as an external subprocess instead, preserving memory safety at the cost of subprocess overhead (~50 ms per invocation). The native engine pilots in the rollout are pure-Rust reimplementations that avoid FFI entirely; the 5-stage rollout governance ensures quality parity before they ever replace the bridge adapters.

**Why a 10-stage pipeline instead of a monolithic transcribe function?**
Stage isolation provides three benefits. First, **independent budgets**: a slow normalize stage cannot eat into the backend's time budget. Second, **observable progress**: agents see exactly which stage is running via NDJSON events. Third, **composability**: the `PipelineBuilder` can skip stages that aren't needed, avoiding unnecessary work. Stage-management overhead is negligible (~1 ms per transition) compared to actual inference time.

**Why NDJSON over WebSocket or gRPC?**
NDJSON (newline-delimited JSON) has three advantages for agent consumption. First, **zero dependencies**: any language can parse it with a JSON library and `readline()`. Second, **pipe-friendly**: works with `jq`, `grep`, `head`, `tail`, and standard Unix tools. Third, **TTY-safe**: it flows over SSH, serial links, and PTY connections where binary protocols cannot. The bandwidth tradeoff is irrelevant for a speech-to-text pipeline where inference (not I/O) is the bottleneck.

### Alien-Artifact Engineering Contracts

Every adaptive controller in `franken_whisper` follows a formal *alien-artifact engineering contract*, a design discipline that prevents adaptive systems from making unbounded bad decisions.

**The problem it solves.** Adaptive algorithms (Bayesian routers, auto-tuners, ML-based controllers) can behave unpredictably when their models are wrong. A Bayesian router with a bad prior will confidently make terrible decisions. An auto-tuner with noisy data will oscillate. The standard response ("just add more data" or "tune the hyperparameters") doesn't work for a CLI tool running on user machines with no ops team watching dashboards.

**The contract requires every adaptive controller to declare:**

| Component | Purpose | Example (Backend Router) |
|-----------|---------|--------------------------|
| **State space** | What does the controller observe? | 3 availability states (all / partial / none) |
| **Action space** | What can it decide? | 4 actions (try each backend + error) |
| **Loss matrix** | What's the cost of each state × action? | 3×4 matrix: latency(45%) + quality(35%) + failure(20%) |
| **Posterior terms** | How confident is the model? | Beta distribution per backend |
| **Calibration metric** | How accurate are predictions? | Brier score on 50-observation sliding window |
| **Deterministic fallback** | What happens when the model is wrong? | Static priority list |
| **Fallback trigger** | When does fallback activate? | Brier > 0.35 or calibration < 0.3 or < 5 observations |
| **Evidence ledger** | Audit trail of all decisions | Circular buffer of 200 `RoutingEvidenceLedgerEntry` records |

**What the contract buys you.** Bounded worst-case behavior. Even if the Bayesian model is perfectly miscalibrated, the system falls back to a simple priority list that always works. The evidence ledger makes every decision inspectable after the fact. The loss matrix makes the tradeoffs explicit rather than buried in code. And fallback runs continue to record adaptive predictions so the calibration data keeps accumulating; once the model is good enough again, the adaptive controller can take over without manual intervention.

**Controllers using this contract:**

- Backend router (Bayesian backend selection)
- Adaptive bitrate controller (TTY audio link quality)
- Budget tuner (pipeline stage timeout recommendations)
- Correction tracker (speculation confirmation thresholds)
- Speculative window controller (adaptive window sizing)

### Pipeline Composition & Stage Isolation

**PipelineCx (Pipeline Context).** Every pipeline run creates a `PipelineCx` that carries shared state through all stages:

| Field | Type | Purpose |
|-------|------|---------|
| `trace_id` | `TraceId` | Unique identifier from `(timestamp_ms, random_u64)` |
| `deadline` | `Option<DateTime<Utc>>` | Absolute wall-clock deadline for the entire pipeline |
| `budget` | `Budget` | Remaining time budget (decremented by stage service times) |
| `evidence` | `Vec<Value>` | JSON evidence accumulator for post-hoc analysis |
| `finalizers` | `FinalizerRegistry` | Cleanup handlers run on pipeline exit (bounded to 5 s) |

**CancellationToken (Copy + Send + Sync).** A lightweight handle extracted from `PipelineCx` for passing into background threads and subprocess monitors:

```rust
struct CancellationToken {
    deadline: Option<DateTime<Utc>>,
}
```

The token's `checkpoint()` method checks two conditions: (1) has Ctrl+C been pressed (global `AtomicBool`), and (2) has the deadline passed. If either is true, it returns `Err(Cancelled)`. Stages call `checkpoint()` at safe points (between loop iterations, before `COMMIT`, after subprocess completion). The token is threaded down into stage budget loops so a long-running stage honors its budget even when the per-stage timer fires before the global deadline.

**Dynamic Stage Composition.** Not every run executes all 10 stages. The pipeline skips stages that aren't needed:

| Condition | Skipped Stages |
|-----------|----------------|
| Input is already 16 kHz mono WAV | Normalize (passthrough) |
| No `--diarize` flag | Diarize |
| No `--vad` flag | VAD |
| No GPU features compiled | Accelerate (CPU fallback used inline) |
| `--no-persist` flag | Persist |
| Backend doesn't support alignment | Align |
| `--no-stem` flag set | Source Separate |
| VAD detects only silence | All post-Backend stages |

Skipped stages still emit `*.skip` events to the NDJSON stream so agents can distinguish "not needed" from "failed."

### Word-Level Timestamps

`franken_whisper` supports word-level timestamps through `WordTimestampParams` (`enabled`, `max_len`, `token_threshold`, `token_sum_threshold`) on the library API. The CLI exposes `--word-threshold` as a per-word confidence filter (drop words below the threshold); the deeper `WordTimestampParams` fields are configured programmatically via the `TranscribeRequest::word_timestamps` slot.

When word-level timestamps are enabled:

- The `whisper.cpp` backend is invoked with the appropriate word-timestamp flags (`-ml`, `-wt`, `-wtps`).
- Word arrays are extracted from each segment in the backend's JSON output.
- The extraction loop honors the pipeline's cancellation token so an in-flight word extraction can be interrupted mid-segment rather than only after the entire backend invocation completes.
- Each word receives a `confidence` score (filtered by `--word-threshold` if set) and `start`/`end` timestamps inside its parent segment.
- Native engine pilots are required to match bridge word-level output to the 50 ms canonical tolerance before they can be promoted past the `validated` rollout stage.

### Evidence Ledger & Routing History

Every routing decision records a `RoutingEvidenceLedgerEntry` in a 200-entry circular buffer:

| Field | Type | Purpose |
|-------|------|---------|
| `decision_id` | String | Unique decision identifier |
| `trace_id` | String | Links to pipeline trace |
| `timestamp_rfc3339` | String | When the decision was made |
| `observed_state` | String | Availability state at decision time |
| `chosen_action` | String | Which backend was selected |
| `policy_id` | String | Which routing policy was active |
| `loss_matrix_hash` | String | Provenance tracking for the loss matrix |
| `availability` | `Vec<(String, bool)>` | Per-backend availability snapshot |
| `duration_bucket` | String | Audio duration category (short / medium / long) |
| `diarize` | bool | Whether diarization was requested |
| `actual_outcome` | `Option<RoutingOutcomeRecord>` | Observed success / failure (filled post-run) |

Queryable via `robot routing-history` and included in stage event payloads for post-hoc analysis. The `loss_matrix_hash` field enables detecting when the routing policy itself changed between runs.

### Trace ID & Run ID Generation

Every pipeline run receives two identifiers:

**Trace ID** is a deterministic composite of wall-clock time and randomness:

```
trace_id = hex(timestamp_ms) + "-" + hex(uuid_v4_lower_80_bits)
Example: "18e4a0b1c00-a1b2c3d4e5f6"
```

The timestamp prefix enables time-range queries without parsing. The random suffix prevents collisions when multiple runs start in the same millisecond.

**Run ID** is a standard UUID v4:

```
run_id = uuid::Uuid::new_v4().to_string()
Example: "550e8400-e29b-41d4-a716-446655440000"
```

The `trace_id` links all events across the pipeline (including routing evidence); the `run_id` is the persistence key in SQLite.

### Calibration Sliding Window

The router maintains a `CalibrationState` with a sliding window of prediction-outcome pairs:

```rust
struct CalibrationState {
    observations: VecDeque<CalibrationObservation>,  // bounded to 50 entries
    window_size: usize,                              // ROUTER_HISTORY_WINDOW = 50
}

struct CalibrationObservation {
    predicted_probability: f64,  // router's confidence that the backend would succeed
    actual_outcome: f64,         // 1.0 if it did succeed, 0.0 if it failed
    observed_at_rfc3339: String, // when the observation was recorded
}
```

**Update cycle:**

1. Before each run, the router predicts `p_success` for the chosen backend.
2. After the run, the actual outcome (success / failure) is recorded.
3. If the window exceeds 50 entries, the oldest observation is evicted.
4. The Brier score is recomputed from the current window.
5. Non-finite predictions (`NaN`, `±∞`) are sanitized before being stored, so pathological backend signals can never poison the calibration buffer.

**Brier Score Formula:**

```
Brier = (1/N) * Σ_i (predicted_i - actual_i)^2
```

Brier = 0.0 means perfect calibration. Brier = 0.25 is the score of a coin flip. Brier > 0.35 triggers fallback to static priority routing.

The simpler calibration score tracks `correct_predictions / total_predictions`, where a prediction is "correct" if the predicted probability matched the outcome direction (predicted > 0.5 and succeeded, or predicted < 0.5 and failed). Quick sanity check independent of the Brier score.

### Beta Distribution Posterior Updates

Each backend's reliability is modeled as `Beta(α, β)`:

- **Mean** = `α / (α + β)` (estimated success probability)
- **Variance** = `α·β / ((α + β)² · (α + β + 1))` (uncertainty)

The update rule blends the prior with empirical data:

```
if sample_count >= 5:
    empirical_weight = min(sample_count, 20)
    α += success_rate * empirical_weight
    β += (1 - success_rate) * empirical_weight
```

The weight cap at 20 prevents a long history from making the posterior too rigid. A backend that succeeded 19 / 20 recent runs gets `α += 19, β += 1`, strongly increasing its selection probability. A backend that succeeded 10 / 20 gets `α += 10, β += 10`, pulling toward neutral.

### Input Validation

Before the pipeline starts, the request is validated.

**Mutually Exclusive Input Modes.** Exactly one of `--input`, `--stdin`, or `--mic` must be specified. Zero inputs or multiple inputs produce an immediate error before pipeline construction; in robot mode, that error is a structured `run_error` envelope, never human stderr.

**Pipeline Configuration Validation.** `PipelineConfig::validate()` enforces ordering constraints:

- `Normalize` must come after `Ingest`
- `Backend` must come after `Normalize`
- No duplicate stages in the pipeline
- All stage dependencies satisfied in execution order

These checks run at pipeline build time, so invalid configurations fail fast.

**Path / Directory Guards.** Input paths are validated with `is_file` (directories rejected); empty parent components are ignored when ensuring directories exist; stdin temporary files use sanitized extensions to prevent path traversal.

**Timeout Conversion.** The `--timeout` flag (seconds) converts to an absolute deadline:

```
timeout_ms = timeout_seconds * 1000  (with saturating_mul)
deadline   = now + chrono::Duration::milliseconds(clamped_to_i64_max)
```

`saturating_mul` prevents overflow; the clamp to `i64::MAX` prevents `chrono` panics on unreasonably large timeouts.

### Stage Failure Behavior

When a pipeline stage fails, behavior depends on the error type:

| Error Type | Behavior | Event Emitted |
|------------|----------|---------------|
| `Cancelled` (Ctrl+C or deadline) | Pipeline stops immediately | `{stage}.cancelled` |
| `StageTimeout` (budget exceeded) | Pipeline stops, timeout reported | `{stage}.timeout` |
| Other errors (I/O, backend, etc.) | Pipeline stops, error propagated | `{stage}.error` |

The `stage_start` timer is cleared on checkpoint failure so elapsed time from a cancelled stage never leaks into the next stage's profile. All stage failures produce a corresponding error event in the NDJSON stream before the pipeline terminates. In-progress SQLite transactions roll back via the savepoint mechanism. Registered finalizers (temp directory cleanup, subprocess kills) run within the 5-second cleanup budget. The final `run_error` event contains the structured error code and message, so agents can programmatically determine what failed and why.

### Evidence Accumulation

The `PipelineCx` carries a `Vec<serde_json::Value>` evidence accumulator that grows throughout the pipeline:

1. **Routing decision**: the backend router pushes its decision evidence (posterior snapshot, loss matrix, chosen action).
2. **Stage observations**: individual stages record evidence about unusual conditions (e.g., normalization fallback to ffmpeg, high latency).
3. **Conformance results**: when native engines run in shadow / validated mode, comparison results are recorded as evidence.

All accumulated evidence is included in the final `RunReport.evidence` field and persisted alongside the run in SQLite. This enables post-hoc debugging without needing to reproduce exact conditions.

### TUI Internals

The interactive TUI (`--features tui`) provides a three-pane interface:

```
+-------------------+-------------------------------------+
|                   |                                     |
|   Runs List       |   Timeline / Transcript             |
|   (left pane)     |   (main pane)                       |
|                   |                                     |
|   - run-abc       |   [0.0s - 2.5s] Hello world         |
|   - run-def       |   [2.5s - 5.1s] How are you         |
|   > run-ghi       |   [5.1s - 7.3s] [SPK_01] Fine       |
|                   |                                     |
+-------------------+-------------------------------------+
|   Event Details (bottom pane)                           |
|   stage: backend | code: backend.ok | 4.2s              |
+---------------------------------------------------------+
```

**Keyboard Bindings:**

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Cycle focus between panes |
| `Up` / `Down` | Move selection within focused pane |
| `PageUp` / `PageDown` | Jump by page |
| `r` | Reload data from SQLite |
| `h` or `?` | Toggle help overlay |
| `q` or `Ctrl+C` | Quit |

**Speaker Color Assignment.** Speakers are assigned distinct colors via an FNV-1a-style hash of the speaker label, mapped to an 8-color palette. The same speaker always gets the same color within a session, making multi-speaker conversations visually parseable.

**Segment Retention.** To prevent unbounded memory growth during long sessions, the TUI caps displayed segments at 10,000 (`DEFAULT_MAX_SEGMENTS`). When the cap is exceeded, oldest segments are drained first, keeping the most recent transcription visible. Segments with `None` timestamps display gracefully rather than panicking; run detail load errors surface in the event-detail pane.

### Microphone Capture

Live microphone capture is the only path that requires ffmpeg (file transcription uses the built-in decoder). The capture path adapts to the host OS:

| OS | ffmpeg Format | Default Device | Notes |
|----|--------------|----------------|-------|
| Linux | `alsa` | `default` | ALSA subsystem |
| macOS | `avfoundation` | `:0` | First audio input device |
| Windows | `dshow` | `audio=default` | DirectShow |

The microphone flow:

1. Spawn ffmpeg with `-f <format> -i <device> -t <seconds> -ar 16000 -ac 1 -c:a pcm_s16le <output>`
2. Wait for capture to complete (bounded by `--mic-seconds`)
3. Output is already 16 kHz mono WAV, so the normalization stage becomes a passthrough
4. Proceed to backend execution

Custom devices, formats, and sources can be overridden via `--mic-device`, `--mic-ffmpeg-format`, and `--mic-ffmpeg-source` flags. Active region boundaries are clamped to the actual audio duration of the captured file, preventing the pipeline from over-reading past the buffer when capture is cut short.

### Segment Comparison Algorithm

The conformance comparator aligns expected vs. observed segment lists index-by-index:

```
Input: expected[0..N], observed[0..M], tolerance

1. If N != M: set length_mismatch = true

2. For i in 0..min(N, M):
   a. Compare text:
      if tolerance.require_text_exact && expected[i].text != observed[i].text:
          text_mismatches += 1

   b. Compare speaker:
      if tolerance.require_speaker_exact && expected[i].speaker != observed[i].speaker:
          speaker_mismatches += 1

   c. Compare timestamps:
      if |expected[i].start_sec - observed[i].start_sec| > tolerance.timestamp_tolerance_sec:
          timestamp_violations += 1
      if |expected[i].end_sec - observed[i].end_sec| > tolerance.timestamp_tolerance_sec:
          timestamp_violations += 1

3. Return SegmentComparisonReport {
     length_mismatch,
     text_mismatches,
     speaker_mismatches,
     timestamp_violations,
     segments_compared: min(N, M),
   }
```

**Default Tolerance Values:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `timestamp_tolerance_sec` | 0.05 (50 ms) | Maximum acceptable timestamp drift |
| `require_text_exact` | true | Text must match exactly |
| `require_speaker_exact` | false | Speaker labels not required to match |

The 50 ms timestamp tolerance (`CANONICAL_TIMESTAMP_TOLERANCE_SEC`) is the single source of truth across the entire codebase; conformance tests, native engine rollout gates, and replay comparison all reference this constant.

### WER Approximation Algorithm

The conformance module includes a Levenshtein-based Word Error Rate calculator used in both conformance testing and speculative streaming correction:

```
1. Tokenize both transcripts by whitespace -> word sequences
2. Compute Levenshtein edit distance between word sequences
   (insertions, deletions, substitutions)
3. WER = edit_distance / max(reference_length, 1)
4. Clamp to [0.0, 1.0]
```

The conformance module normalizes by the reference (expected) length, while the speculation module normalizes by `max(fast_length, quality_length)` since neither model is the "reference."

### Overlap Detection

The `SegmentConformancePolicy` can optionally reject overlapping segments, defined as a segment whose `end_sec` exceeds the next segment's `start_sec` beyond a configurable epsilon (default: 1 µs). This catches backends that produce garbled timeline output:

```
for each pair (segment[i], segment[i+1]):
    if segment[i].end_sec > segment[i+1].start_sec + epsilon:
        report overlap violation at index i
```

Overlap detection runs before cross-engine comparison so a backend producing self-overlapping output is flagged before being compared against a reference.

### Metamorphic Testing

In addition to conventional unit and integration tests, the project ships dedicated **metamorphic test suites** that verify input → output relationships rather than oracle values:

| File | What it checks |
|------|----------------|
| [`tests/metamorphic_audio_tests.rs`](tests/metamorphic_audio_tests.rs) | Audio normalization is stable under permutation of decoder paths, silence padding is idempotent, stereo→mono averaging matches manual computation |
| [`tests/metamorphic_accelerate_tests.rs`](tests/metamorphic_accelerate_tests.rs) | Softmax is permutation-equivariant and scale-invariant; layer-norm output mean ≈ 0, variance ≈ 1; attention scoring respects masking |
| [`tests/metamorphic_speculation_tests.rs`](tests/metamorphic_speculation_tests.rs) | String-distance functions are symmetric, satisfy the triangle inequality, and window-size adjustments respect bounds |

Metamorphic tests catch entire categories of regressions where the "correct" answer is unknown but the *relationship* between inputs and outputs must hold.

### Configuration Recipes

**Fastest possible transcription (accuracy tradeoff):**

```bash
franken_whisper transcribe --input audio.mp3 \
  --backend whisper_cpp \
  --model tiny.en \
  --no-persist \
  --no-timestamps \
  --beam-size 1 \
  --best-of 1 \
  --json
```

**Highest accuracy with diarization:**

```bash
franken_whisper transcribe --input meeting.mp3 \
  --backend whisper_cpp \
  --model large-v3 \
  --diarize \
  --hf-token "$HF_TOKEN" \
  --min-speakers 2 \
  --max-speakers 8 \
  --vad \
  --json
```

**Agent integration with health monitoring:**

```bash
# Pre-flight check
franken_whisper robot health 2>/dev/null | jq -e '.overall_status == "ok"' > /dev/null

# Transcribe with full event stream
franken_whisper robot run \
  --input audio.mp3 \
  --backend auto 2>/dev/null | while IFS= read -r line; do
    event=$(echo "$line" | jq -r '.event')
    case "$event" in
      stage)        echo "[STAGE] $(echo "$line" | jq -r '.code')" ;;
      run_complete) echo "[DONE]  $(echo "$line" | jq -r '.transcript' | head -c 100)" ;;
      run_error)    echo "[FAIL]  $(echo "$line" | jq -r '.code'): $(echo "$line" | jq -r '.message')" ;;
    esac
  done
```

**Offline archival workflow:**

```bash
# Transcribe everything into a custom DB
for f in archive/*.mp3; do
  franken_whisper transcribe --input "$f" --db archive.sqlite3 --json > /dev/null
done

# Export to portable JSONL
franken_whisper sync export-jsonl --output ./archive_snapshot --db archive.sqlite3

# Optionally gzip the snapshot for storage (the import path auto-detects .gz)
gzip ./archive_snapshot/*.jsonl

# Re-import to validate; conflict-policy=skip is a no-op on the original DB
franken_whisper sync import-jsonl --input ./archive_snapshot --conflict-policy skip --db archive.sqlite3
```

**Low-bandwidth remote transcription via TTY:**

```bash
# On remote (has audio, no GPU):
franken_whisper tty-audio encode --input recording.wav --chunk-ms 100 > /tmp/frames.ndjson

# Transfer (works over any text channel):
scp /tmp/frames.ndjson gpu-server:/tmp/

# On GPU server (has whisper, fast inference):
cat /tmp/frames.ndjson | franken_whisper tty-audio decode --output /tmp/audio.wav
franken_whisper transcribe --input /tmp/audio.wav --backend whisper_cpp --model large-v3 --json
```

### Release Binary Optimization

The release profile is aggressively optimized for deployment:

```toml
[profile.release]
opt-level = "z"        # Optimize for binary size (smaller than "s")
lto = true             # Full link-time optimization across all crates
codegen-units = 1      # Single codegen unit for maximum optimization opportunity
panic = "abort"        # Abort on panic (no unwinding overhead, smaller binary)
strip = true           # Strip debug symbols from final binary
```

This produces the smallest possible binary while retaining full optimization. The tradeoff is slower compilation (`codegen-units = 1` + LTO) and no panic unwinding (acceptable for a CLI tool where panics are fatal regardless). On a typical Linux x86_64 build, the stripped release binary is significantly smaller than a default release build.

### Glossary

| Term | Definition |
|------|-----------|
| **Backend** | An external ASR engine (whisper.cpp, insanely-fast-whisper, whisper-diarization) or its native Rust replacement |
| **Bridge adapter** | Code that spawns an external backend process and parses its output into a `TranscriptionResult` |
| **Brier score** | Mean squared error between predicted probabilities and actual outcomes; measures calibration quality (0.0 = perfect, 0.25 = random) |
| **Conformance** | Cross-engine output comparison using the 50 ms timestamp tolerance and optional text / speaker matching |
| **Decision contract** | Formal specification of an adaptive controller's state space, action space, loss matrix, posterior, calibration, fallback, and evidence |
| **Evidence ledger** | Circular buffer recording every routing decision with full posterior snapshots for audit |
| **Finalizer** | A cleanup handler (temp dir removal, subprocess kill) registered during pipeline execution and run on exit within a bounded timeout |
| **Metamorphic test** | A test that verifies an input → output relationship (symmetry, invariance, etc.) rather than an expected oracle value |
| **NDJSON** | Newline-Delimited JSON; one JSON object per line, compatible with `jq` and standard Unix text tools |
| **Native rollout stage** | One of Shadow / Validated / Fallback / Primary / Sole — the conformance-gated promotion ladder for in-process Rust engines |
| **Pipeline stage** | One of 10 composable processing steps (Ingest, Normalize, VAD, Separate, Backend, Accelerate, Align, Punctuate, Diarize, Persist) |
| **Posterior** | Beta distribution `Beta(α, β)` modeling estimated success probability for a backend |
| **Replay envelope** | SHA-256 hash summary (input, backend identity, output) for detecting drift between runs |
| **Replay pack** | Four-artifact directory (`env.json`, `manifest.json`, `repro.lock`, `tolerance_manifest.json`) capturing everything needed to reproduce a run |
| **Robot mode** | The `robot` subcommand; emits structured NDJSON events for machine consumption rather than human-readable text |
| **Savepoint** | SQLite's nested transaction mechanism; used for concurrent session isolation and cancellation-safe writes |
| **Speculative streaming** | Dual-model pattern where a fast model emits partial transcripts immediately and a quality model confirms or corrects them |
| **TTY audio** | Protocol for transporting compressed audio over text-only channels (PTY, SSH, serial) using mu-law + zlib + base64 NDJSON frames |
| **WAL mode** | SQLite's Write-Ahead Logging; allows concurrent reads during writes |

---

## Security & Privacy

### Your Data Never Leaves Your Machine

`franken_whisper` is designed with privacy as a hard constraint:

```
+----------------------------------------------------------------+
|                        YOUR MACHINE                            |
|                                                                |
|  +-----------+    +-------------+    +-----------+             |
|  |   Input   |--->|  Pipeline   |--->|  Output   |             |
|  +-----------+    +-------------+    +-----------+             |
|                                                                |
|  No network calls during transcription                         |
|  No telemetry or analytics                                     |
|  No cloud sync                                                 |
|  No API keys required (except HuggingFace for diarization)     |
+----------------------------------------------------------------+
```

All processing happens on your hardware using local backend binaries. The only external network access is:

- **ffmpeg auto-provisioning**: one-time download, can be disabled via `FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG=0`
- **HuggingFace model downloads**: only when using `--diarize` with `pyannote` models

### Secrets Redaction

The HuggingFace token used by the diarization backend never appears in log output. Every command line is rendered through `render_command_for_log()` before being emitted into traces; the renderer redacts `--hf-token` values so neither structured tracing nor stdout transcripts contain the raw token. This applies to error messages and test snapshots as well; there is no path where the token leaks into a public artifact.

### What's Stored Where

| Location | Contents | Sensitive? |
|----------|----------|------------|
| `.franken_whisper/storage.sqlite3` | Run history, transcripts, segments | Yes (contains transcription text) |
| `.franken_whisper/locks/` | Sync lock files (PID, timestamp only) | No |
| `<work_dir>/normalized_16k_mono.wav` | Temporary normalized audio | Yes (audio content; cleaned up by finalizers) |
| JSONL snapshots | Exported run history | Yes (contains transcription text) |
| `{state_root}/tools/ffmpeg/bin/*` | Auto-provisioned ffmpeg binary | No |

### Secure Deletion

```bash
# Remove all franken_whisper state
rm -rf .franken_whisper/

# Or just the database (preserves auto-provisioned ffmpeg)
rm .franken_whisper/storage.sqlite3
```

---

## Library API

`franken_whisper` is both a CLI binary and a Rust library. The public API exposes every module for embedding ASR pipelines in other applications:

```rust
use franken_whisper::backend::{BackendRouter, Engine};
use franken_whisper::orchestrator::{PipelineConfig, PipelineBuilder, FrankenWhisperEngine};
use franken_whisper::model::{TranscribeRequest, BackendKind, TranscriptionResult};
use franken_whisper::storage::RunStore;
use franken_whisper::robot::robot_schema_value;
use franken_whisper::tty_audio::{encode_wav_to_frames, decode_frames_to_raw};
use franken_whisper::conformance::compare_segments_with_tolerance;
use franken_whisper::error::{FwError, FwResult};
```

**Key types:**

| Type | Module | Purpose |
|------|--------|---------|
| `TranscribeRequest` | `model` | Fully-specified transcription request with all parameters |
| `TranscriptionResult` | `model` | Backend output: transcript, segments, language, acceleration metadata |
| `TranscriptionSegment` | `model` | Individual segment: start/end times, text, speaker, confidence |
| `RunReport` | `model` | Complete run envelope: request + result + events + evidence + replay |
| `BackendKind` | `model` | Enum: `Auto`, `WhisperCpp`, `InsanelyFast`, `WhisperDiarization` |
| `FrankenWhisperEngine` | `orchestrator` | Main pipeline orchestrator |
| `PipelineConfig` | `orchestrator` | Ordered list of stages to execute |
| `PipelineBuilder` | `orchestrator` | Fluent constructor for pipeline configs |
| `CancellationToken` | `orchestrator` | Cooperative cancellation handle |
| `RunStore` | `storage` | SQLite persistence interface (open, persist, query) |
| `TtyAudioFrame` | `tty_audio` | Protocol frame with seq, codec, payload, integrity hashes |
| `TtyControlFrame` | `tty_audio` | Control messages (handshake, ack, retransmit, backpressure, transcript_*) |
| `DecodeReport` | `tty_audio` | Decode telemetry: frames decoded, gaps, duplicates, failures |
| `ReplayEnvelope` | `replay_pack` | SHA-256 hash summary for deterministic replay |
| `FwError` | `error` | Error enum with 12 variants, each mapping to a stable `FW-*` code |
| `SegmentCompatibilityTolerance` | `conformance` | Drift thresholds for cross-engine comparison |

---

## Data Model

### SQLite Schema

```sql
-- Core run record (one row per transcription). All columns NOT NULL.
CREATE TABLE runs (
    id                  TEXT PRIMARY KEY,    -- UUID run identifier
    started_at          TEXT NOT NULL,       -- RFC-3339 timestamp
    finished_at         TEXT NOT NULL,       -- RFC-3339 timestamp (set on completion or error)
    backend             TEXT NOT NULL,       -- "whisper_cpp", "insanely_fast", "whisper_diarization"
    input_path          TEXT NOT NULL,       -- Original input file path (or sentinel for stdin/mic)
    normalized_wav_path TEXT NOT NULL,       -- Path to 16 kHz mono WAV
    request_json        TEXT NOT NULL,       -- Full TranscribeRequest as JSON
    result_json         TEXT NOT NULL,       -- Full TranscriptionResult as JSON
    warnings_json       TEXT NOT NULL,       -- Non-fatal warnings as JSON array
    transcript          TEXT NOT NULL,       -- Plain text transcript
    replay_json         TEXT NOT NULL DEFAULT '{}',     -- ReplayEnvelope as JSON
    acceleration_json   TEXT NOT NULL DEFAULT '{}'      -- AccelerationReport as JSON
);

-- Timed transcript segments (N rows per run)
CREATE TABLE segments (
    run_id          TEXT NOT NULL,           -- References runs(id)
    idx             INTEGER NOT NULL,        -- Segment index within run
    start_sec       REAL,                    -- Start time in seconds (may be NULL)
    end_sec         REAL,                    -- End time in seconds   (may be NULL)
    speaker         TEXT,                    -- Speaker label (if diarized)
    text            TEXT NOT NULL,           -- Segment text
    confidence      REAL,                    -- Confidence score [0.0, 1.0]
    PRIMARY KEY (run_id, idx)
);

-- Pipeline stage events (M rows per run)
CREATE TABLE events (
    run_id          TEXT NOT NULL,           -- References runs(id)
    seq             INTEGER NOT NULL,        -- Strictly increasing per run
    ts_rfc3339      TEXT NOT NULL,           -- Non-decreasing timestamp
    stage           TEXT NOT NULL,           -- Pipeline stage name
    code            TEXT NOT NULL,           -- Event code (e.g., "backend.ok")
    message         TEXT NOT NULL,           -- Human-readable description
    payload_json    TEXT NOT NULL,           -- Event-specific JSON payload
    PRIMARY KEY (run_id, seq)
);

-- Key-value schema metadata
CREATE TABLE _meta (
    key   TEXT PRIMARY KEY,                  -- e.g. 'schema_version'
    value TEXT NOT NULL                      -- 'schema_version' currently '3'
);

-- v3 migration adds indexes on hot query paths (recent-runs listing,
-- per-run segment + event lookup).
```

### NDJSON Export Format

JSONL snapshots mirror the database schema:

**`runs.jsonl`** (one JSON object per line):
```json
{"id":"fw-run-abc","started_at":"2026-04-25T06:00:00Z","finished_at":"2026-04-25T06:00:05Z","backend":"whisper_cpp","transcript":"Hello world...","replay_json":"{...}"}
```

**`segments.jsonl`:**
```json
{"run_id":"fw-run-abc","idx":0,"start_sec":0.0,"end_sec":2.5,"text":"Hello world","confidence":0.95}
```

**`events.jsonl`:**
```json
{"run_id":"fw-run-abc","seq":0,"ts_rfc3339":"2026-04-25T06:00:00Z","stage":"ingest","code":"ingest.start","message":"materializing input","payload_json":"{}"}
```

**`manifest.json`** (integrity metadata):
```json
{
  "exported_at": "2026-04-25T06:30:00Z",
  "run_count": 42,
  "segment_count": 1847,
  "event_count": 336,
  "runs_sha256": "a1b2c3...",
  "segments_sha256": "d4e5f6...",
  "events_sha256": "g7h8i9..."
}
```

### Key Data Types

**TranscribeRequest** (the full input specification):

| Field | Type | Description |
|-------|------|-------------|
| `input` | `InputSource` | Tagged enum: `File { path }` / `Stdin { source_hint }` / `Microphone { seconds, device, ffmpeg_format, ffmpeg_source }` |
| `backend` | `BackendKind` | `Auto`, `WhisperCpp`, `InsanelyFast`, or `WhisperDiarization` |
| `model` | `Option<String>` | Model name or path forwarded to backend |
| `language` | `Option<String>` | Language hint (ISO 639-1) |
| `translate` | `bool` | Translate to English |
| `diarize` | `bool` | Enable speaker diarization |
| `persist` | `bool` | Persist run to SQLite (inverse of `--no-persist`) |
| `db_path` | `PathBuf` | SQLite database path |
| `timeout_ms` | `Option<u64>` | Overall pipeline timeout in milliseconds |
| `backend_params` | `BackendParams` | Aggregate of all backend-specific tunables — decoding params (beam / temperature / thresholds), VAD params, diarization config, speculative config, alignment overrides, word-timestamp params, audio windowing, output format selectors |

The `BackendParams` aggregate is the catch-all for every backend-specific tuning knob. The CLI marshals every `transcribe` flag into the corresponding field here, and downstream stages read what they need (VAD reads `vad_params`, the speculative path reads `speculative`, the bridge adapters read decoding params, and so on).

**TranscriptionResult** (what the backend produces):

| Field | Type | Description |
|-------|------|-------------|
| `transcript` | `String` | Full transcript text |
| `segments` | `Vec<TranscriptionSegment>` | Timed segments with text, speaker, confidence |
| `language` | `Option<String>` | Detected language |
| `acceleration` | `Option<AccelerationReport>` | Confidence normalization metadata |
| `raw_backend_json` | `Option<String>` | Preserved raw backend output for replay |

**RunEvent** (a single pipeline event):

| Field | Type | Description |
|-------|------|-------------|
| `seq` | `u64` | Strictly increasing per run |
| `ts_rfc3339` | `String` | Non-decreasing RFC-3339 timestamp |
| `stage` | `String` | Pipeline stage (e.g., `"ingest"`, `"backend"`, `"speculation"`) |
| `code` | `String` | Event code (e.g., `"backend.routing.decision_contract"`) |
| `message` | `String` | Human-readable description |
| `payload` | `Value` | Event-specific JSON payload |

---

## Performance Characteristics

### Audio Normalization

| Input Format | Duration | Normalization Time | Method |
|--------------|----------|--------------------|--------|
| MP3 (128 kbps, stereo) | 2 min | ~260 ms | Built-in (`symphonia`) |
| FLAC (16-bit, 44.1 kHz) | 2 min | ~180 ms | Built-in (`symphonia`) |
| WAV (16 kHz, mono) | 2 min | ~5 ms | Passthrough (already normalized) |
| MP4 (video, AAC audio) | 2 min | ~500 ms | ffmpeg fallback |

The built-in path is fast because it runs entirely in-process with no subprocess spawning, no temporary file juggling, and no `PATH` dependency.

### Pipeline Overhead

Typical overhead beyond the backend inference time:

| Component | Time | Notes |
|-----------|------|-------|
| CLI parse | < 1 ms | clap argument parsing |
| Database open | ~5 ms | SQLite connection + schema check |
| Ingest | ~1 ms | File existence check, size read |
| Normalize (MP3) | ~260 ms | Built-in Rust decoder |
| Persistence | ~10 ms | SQLite transaction (8 retry budget) |
| Latency profiling | ~1 ms | Compute utilization ratios |
| Report assembly | ~2 ms | JSON serialization |
| **Total overhead** | **~280 ms** | **Everything except actual inference** |

The backend inference stage dominates total runtime (typically 3–60 seconds depending on audio length, model size, and hardware).

### Benchmark Suites

Five Criterion benchmark suites measure performance of critical paths:

| Benchmark | What it measures |
|-----------|------------------|
| `storage_bench` | SQLite persist / query throughput, concurrent access |
| `normalize_bench` | Audio normalization latency by format and duration |
| `pipeline_bench` | End-to-end pipeline overhead (mocked backend) |
| `tty_bench` | TTY encode / decode throughput, retransmit loop latency |
| `sync_bench` | JSONL export / import throughput, compression ratios |

Run with: `cargo bench --bench <name>`

The benchmark regression policy is documented in [`docs/benchmark_regression_policy.md`](docs/benchmark_regression_policy.md).

### Binary Size

With the aggressive release profile (`opt-level = "z"`, LTO, stripped):

| Build | Approximate Size |
|-------|------------------|
| Debug | ~150 MB |
| Release (default) | ~20 MB |
| Release (optimized profile) | ~12 MB |
| Release + `--features tui` | ~15 MB |

---

## Codebase Statistics

| Metric | Value |
|--------|-------|
| Total source lines (`src/`) | ~90,000 |
| Total test lines (`tests/`) | ~17,500 |
| Library tests | 3,030 |
| Integration tests | 638 |
| **Total tests** | **3,668** |
| Integration test files (`tests/*.rs`) | 26 |
| Benchmark suites | 5 (Criterion) |
| Public modules | 18 |
| Error variants | 12 (each with structured code) |
| Backend engines | 3 bridge adapters + 3 paired native pilots under rollout governance |
| Pipeline stages | 10 (composable, independently budgeted) |
| Stage budget knobs | 12 (10 stages + probe + cleanup) |
| CLI subcommands | 6 (`transcribe`, `robot`, `runs`, `sync`, `tty-audio`, `tui`) |
| CLI flags (`transcribe`) | 70+ (inference, VAD, diarization, speculative, audio windowing, word timestamps) |
| Robot event types | 12 (run lifecycle, stage, speculation, health, routing, transcript) |
| TTY control frame types | 10 (`Handshake`, `HandshakeAck`, `Ack`, `RetransmitRequest`, `RetransmitResponse`, `Backpressure`, `TranscriptPartial`, `TranscriptRetract`, `TranscriptCorrect`, `SessionClose`) |
| TTY protocol versions | 2 (v1 audio, v2 audio + transcript) |
| Replay pack artifacts | 4 (`env.json`, `manifest.json`, `repro.lock`, `tolerance_manifest.json`) |
| Sync conflict policies | 4 (reject, skip, overwrite, overwrite-strict) |
| Native rollout stages | 5 (shadow, validated, fallback, primary, sole) |
| Conformance tolerance | 50 ms canonical timestamp tolerance |
| Evidence ledger capacity | 200 entries (circular buffer) |
| Router history window | 50 outcome records per backend |
| Clippy enforcement | `#![forbid(unsafe_code)]` + `-D warnings` on all targets |
| Cargo features | 3 (`tui`, `gpu-frankentorch`, `gpu-frankenjax`) |
| Release optimizations | `opt-level = "z"`, LTO, single codegen unit, `panic = "abort"`, stripped |

---

## Testing

Over 100,000 lines of Rust with **3,100+ tests** across unit, integration, conformance, metamorphic, and doc-test suites (the count *dropped* in v0.2.0 — the mock-pilot test suites were deleted along with the pilot mocks they tested; the test-support backend mocks in `tests/mocks/` are unrelated and remain).

```bash
# run all library tests
cargo test --lib

# run specific test module
cargo test --lib -- backend::tests
cargo test --lib -- robot::tests
cargo test --lib -- tty_audio::tests

# run integration tests
cargo test --test tty_telemetry_tests
cargo test --test conformance_comparator_tests
cargo test --test conformance_harness
cargo test --test gpu_cancellation_tests
cargo test --test robot_contract_tests
cargo test --test e2e_pipeline_tests

# run metamorphic suites
cargo test --test metamorphic_audio_tests
cargo test --test metamorphic_accelerate_tests
cargo test --test metamorphic_speculation_tests

# run benchmarks
cargo bench --bench storage_bench
cargo bench --bench normalize_bench
cargo bench --bench pipeline_bench
cargo bench --bench tty_bench
cargo bench --bench sync_bench

# lint
cargo clippy --all-targets -- -D warnings
```

### Test Categories

| Category | Approx. Count | Description |
|----------|---------------|-------------|
| Backend engine tests | 260+ | Engine trait compliance, native pilot validation |
| Robot contract tests | 150+ | NDJSON schema validation, field presence, panic-safety |
| TTY audio tests | 350+ | Handshake, integrity, retransmit, telemetry, control frames |
| Conformance tests | 130+ | Cross-engine tolerance, replay envelope drift, segment invariants, native pilot fixtures |
| Storage tests | 100+ | SQLite roundtrip, concurrent writes, migration, recovery |
| Sync tests | 300+ | JSONL export / import, gzip, conflict resolution, lock pid liveness |
| GPU cancellation tests | 42+ | Stream ownership, fence payloads, fallback |
| Speculation tests | 200+ | Windowing, adaptive thresholds, correction drift |
| Metamorphic tests | 90+ | Audio invariants, softmax / layer-norm properties, string-distance laws |
| CLI integration tests | 80+ | End-to-end command execution with stub backends |

A clause-to-test traceability table lives in [`tests/COVERAGE.md`](tests/COVERAGE.md).

---

## CI / CD Release Pipeline

Two GitHub Actions workflows drive distribution:

- **[`dist.yml`](.github/workflows/dist.yml)**: multi-platform binary builds (Linux x86_64 / ARM64, macOS Intel / ARM64, Windows x86_64), triggered on push to `v*` tags or manual `workflow_dispatch`. The `test` job runs `cargo fmt --check` (advisory), `cargo clippy --all-targets -- -D warnings`, and `cargo test --lib`. The `build` job produces `franken_whisper-{VERSION}-{OS}_{ARCH}` archives (`.tar.gz` on Unix, `.zip` on Windows) with stripped binaries and SHA-256 checksums published as `checksums-sha256.txt`.
- **[`release-automation.yml`](.github/workflows/release-automation.yml)** watches `main` for changes to `Cargo.toml`. When the package version changes, it creates a matching `v{VERSION}` git tag, which in turn fires `dist.yml`. End-to-end: edit `Cargo.toml`, commit, push, and a complete multi-platform release lands minutes later.

The Rust toolchain channel used in CI is kept in sync with `rust-toolchain.toml` so local builds and CI builds resolve to the same nightly.

---

## Troubleshooting

### `FW-CMD-MISSING: whisper-cli not found`

No backend binary is on your `PATH` and no native-engine model file resolved. Either drop a ggml model where the native engine finds it (`$FRANKEN_WHISPER_MODEL_DIR`, `~/.cache/franken_whisper/models`, `~/models/whisper` — `scripts/fetch_test_models.sh` fetches a pinned tiny.en), or install a bridge binary:

```bash
# whisper.cpp
brew install whisper-cpp                                  # macOS
# or build from source: https://github.com/ggerganov/whisper.cpp

# or override the binary name
export FRANKEN_WHISPER_WHISPER_CPP_BIN=/path/to/whisper-cli
```

### `FW-BACKEND-UNAVAILABLE: diarization requires HF token`

Diarization needs a HuggingFace API token for pyannote models:

```bash
export FRANKEN_WHISPER_HF_TOKEN="hf_your_token_here"
# or pass directly
franken_whisper transcribe --input audio.mp3 --diarize --hf-token "hf_..."
```

The token is automatically redacted from command logs.

### `FW-CMD-TIMEOUT: backend exceeded timeout`

The backend took longer than the allowed duration:

```bash
# increase the overall pipeline timeout (seconds)
franken_whisper transcribe --input long_audio.mp3 --timeout 600 --json

# or raise the per-stage backend budget
FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS=1800000 \
  franken_whisper transcribe --input long_audio.mp3 --json
```

### Robot mode outputs nothing

Ensure you're using the `robot run` subcommand, not just `robot`:

```bash
franken_whisper robot run --input audio.mp3 --backend auto
```

### SQLite `database is locked`

Another `franken_whisper` process is writing. The storage layer retries with linear backoff (5–40 ms across 8 attempts, ~180 ms total worst case), but simultaneous heavy writes may conflict. Use `--no-persist` to skip persistence, or use separate `--db` paths.

### Built-in decoder fails on a file ffmpeg handles fine

Some formats or containers are outside `symphonia`'s coverage. Force the ffmpeg path:

```bash
export FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE=1
franken_whisper transcribe --input exotic_file.opus --json
```

### Native engine produced different output than the bridge

Check the rollout stage:

```bash
echo "$FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE"   # primary / sole means native is active
echo "$FRANKEN_WHISPER_NATIVE_EXECUTION"        # must be 1 for native to actually run
```

Roll back to `fallback` while the conformance harness investigates:

```bash
export FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=fallback
```

### Sync says "lock held by PID X but process gone"

The sync lock is age-gated when the owning PID is unknown. Wait the configured idle window or remove the lock file directly:

```bash
rm .franken_whisper/locks/sync.lock
```

---

## Limitations

- **A backend binary or a model file required.** Transcription needs either a bridge binary (`whisper-cli`, `insanely-fast-whisper`, or `whisper-diarization`) or a ggml model file for the in-process native engine (which then needs no external binaries at all). With neither present, runs fail with `FW-BACKEND-UNAVAILABLE`.
- **ffmpeg only needed for video / exotic formats / mic.** The built-in Rust decoder handles common audio formats natively. ffmpeg is used as an automatic fallback for video files and exotic codecs. Microphone capture always depends on ffmpeg.
- **Path-dependent crates.** `fsqlite` / `fsqlite-types` (and the optional `ftui` / `frankentorch` / `frankenjax` crates) are still local path dependencies. `franken_whisper` is therefore not currently published as a standalone crate on crates.io. The core infrastructure crates (`asupersync`, `franken-kernel`, `franken-evidence`, `franken-decision`) are on crates.io.
- **Native engines under rollout governance.** Native Rust engine implementations are conformance-gated pilots. They execute in-process only when `FRANKEN_WHISPER_NATIVE_EXECUTION=1` and the rollout stage is `primary` or `sole`; otherwise bridge adapters remain active.
- **One-way sync.** JSONL export / import is one-way. There is no bidirectional merge beyond the explicit `--conflict-policy` flag.
- **Single-machine.** Designed for single-machine use with local SQLite. No distributed or multi-node support.
- **ffmpeg auto-provisioning is Linux x86_64 only.** On other platforms (`macOS`, Windows, Linux ARM) you must install ffmpeg manually for the formats `symphonia` cannot decode and for microphone capture.

---

## FAQ

**Q: Do I need all three backends installed?**
No. `franken_whisper` works with any single backend — and as of v0.2.0, with **no** bridge backend at all: drop a ggml model file in `$FRANKEN_WHISPER_MODEL_DIR` (or `~/.cache/franken_whisper/models`) and the in-process native engine transcribes by itself. The `auto` router will use whatever is available. You can also force a specific backend with `--backend whisper_cpp`.

**Q: What audio formats are supported?**
Common audio formats (MP3, AAC, FLAC, WAV, OGG, Vorbis, ALAC) are decoded natively by the built-in Rust decoder with zero external dependencies. Video files and exotic codecs (AC3, DTS, Opus-in-MKV) fall back to ffmpeg automatically.

**Q: Can I use this as a library?**
Yes. `franken_whisper` is both a library crate and a binary. The public API exposes all modules: `backend`, `orchestrator`, `robot`, `storage`, `tty_audio`, `conformance`, `speculation`, etc.

**Q: What's the "replay envelope"?**
Each run produces a `ReplayEnvelope` containing SHA-256 hashes of the input content, backend identity, and output payload. This allows detecting drift when re-running the same input.

**Q: What's the "replay pack"?**
A four-file deterministic bundle (`env.json`, `manifest.json`, `repro.lock`, `tolerance_manifest.json`) that captures everything needed to reproduce a run on a different machine. Same input → byte-identical pack across runs.

**Q: How does cancellation work?**
Ctrl+C sets a global shutdown flag. The `CancellationToken` propagates through every pipeline stage, including the inner loops in word-level timestamp extraction. Each stage calls `token.checkpoint()` at safe points, which returns `Err(Cancelled)` if shutdown was requested. No partial writes to SQLite, no orphaned subprocesses.

**Q: What's the TTY audio module for?**
It enables audio transport over constrained TTY/PTY links where binary data can't flow directly. Audio is compressed (mu-law + zlib), base64-encoded, and transmitted as NDJSON lines with sequence numbers, CRC32, and SHA-256 integrity hashes. Protocol v2 also carries transcript streaming control frames, so speculative transcription can run end-to-end over the same link.

**Q: How does the Bayesian router differ from a simple priority list?**
A priority list always tries backends in the same order. The Bayesian router learns from outcomes: if a backend starts failing, its posterior degrades and traffic shifts to alternatives. When the model is poorly calibrated (Brier > 0.35), it falls back to static priority automatically, but **continues recording calibration observations** so it can resume adaptive routing once the data improves.

**Q: What happens if I Ctrl+C during a long transcription?**
The shutdown controller propagates cancellation through the pipeline. The active stage finishes its current checkpoint, rolls back uncommitted transactions, kills running subprocesses, runs finalizers within 5 s, and exits with code 130. No data corruption, no orphaned processes.

**Q: What's speculative streaming?**
Two models run simultaneously: a fast model produces low-latency partial transcripts while a quality model runs in parallel. When the quality model finishes each window, it either confirms or corrects the fast model's output. Use `--speculative` when you need both low latency and high accuracy.

**Q: What's TinyDiarize?**
whisper.cpp's built-in speaker-turn detection via `--tiny-diarize`. It injects speaker-turn tokens during inference without requiring a separate diarization pipeline or HuggingFace token. Less accurate than full diarization but zero additional dependencies.

**Q: Why SQLite instead of Postgres / Redis / files?**
SQLite fits a single-machine CLI tool: zero configuration, no daemon, ACID transactions, concurrent reads via WAL mode. The `fsqlite` crate provides a Rust-native interface without depending on system `libsqlite3`. JSONL export / import covers portability.

**Q: Can franken_whisper transcribe video files?**
Yes. Any video file ffmpeg can decode (MP4, MKV, AVI, MOV, WebM, …) is handled automatically. The ffmpeg fallback extracts the audio track using `-vn`.

**Q: What's the "alien-artifact engineering contract"?**
A design discipline for adaptive controllers. Every adaptive system in `franken_whisper` (the router, the bitrate controller, the budget tuner, the speculation window controller, the correction tracker) must declare an explicit state space, action space, loss matrix, calibration metric, deterministic fallback trigger, and evidence ledger. This prevents adaptive systems from making unbounded bad decisions when their models are wrong.

**Q: Will my HuggingFace token end up in log files?**
No. Every command line is rendered through `render_command_for_log()` which redacts `--hf-token` values. The redaction applies to tracing output, error messages, and snapshot tests alike.

**Q: How is native engine quality enforced?**
The conformance harness compares native vs. bridge output on a fixture corpus at every rollout-stage promotion. A native engine that produces timestamps > 50 ms different from the bridge for the same audio is blocked from advancing past `validated`. Replay envelopes detect drift between runs, and the rollout stage is recorded in every `backend.ok` event and replay envelope for post-hoc analysis.

---

## Anatomy of a Transcription Run

Step by step, when you run `franken_whisper transcribe --input meeting.mp3 --json --backend auto`:

```
1. CLI PARSE
   Clap parses args -> TranscribeRequest { input: "meeting.mp3", backend: Auto, json: true, ... }

2. ENGINE CONSTRUCTION
   FrankenWhisperEngine::new() opens SQLite database, initializes tracing

3. PIPELINE COMPOSITION
   PipelineBuilder evaluates request flags:
   - No --vad flag           -> skip VAD stage
   - No --diarize flag       -> skip Diarize stage
   - No GPU features         -> skip Accelerate stage (CPU fallback inline)
   - --no-persist NOT set    -> include Persist stage (default)
   Pipeline: [Ingest, Normalize, Backend, Persist]

4. TRACE ID GENERATION
   TraceId::from_parts(1710000000000, random_u64) -> "1710000000000-a1b2c3d4e5f6"

5. INGEST STAGE (budget: 15s)
   emit: stage { code: "ingest.start" }
   Verify meeting.mp3 exists, get file size
   emit: stage { code: "ingest.ok", payload: { size_bytes: 1234567 } }

6. NORMALIZE STAGE (budget: 180s)
   emit: stage { code: "normalize.start" }
   Try built-in Rust decoder (symphonia):
     - Detect format: MP3
     - Decode packets -> f32 samples
     - Mix stereo -> mono (average channels using actual frame length)
     - Resample 44.1 kHz -> 16 kHz (linear interpolation)
     - Quantize f32 -> i16 PCM
     - Write normalized_16k_mono.wav
   emit: stage { code: "normalize.ok", payload: { method: "builtin", duration_ms: 260 } }

7. BACKEND STAGE (budget: 900s)
   emit: stage { code: "backend.routing.decision_contract" }
   Bayesian router evaluates:
     - Probe availability: whisper_cpp=true, insanely_fast=false, diarization=false
     - State: partial_available
     - Compute loss matrix (latency*0.45 + quality*0.35 + failure*0.20)
     - Best action: try_whisper_cpp (lowest expected loss)
     - Calibration check: Brier=0.12, score=0.8 -> adaptive mode (no fallback)
     - Record predicted_probability into the calibration sliding window
   emit: stage { code: "backend.start", payload: { backend: "whisper_cpp" } }
   Spawn: whisper-cli -m large-v3 -f normalized_16k_mono.wav --output-json
   Wait for subprocess (check cancellation token periodically)
   Parse JSON output -> TranscriptionResult { transcript, segments, language }
   emit: stage { code: "backend.ok",
                 payload: { segments: 42, language: "en",
                            implementation: "bridge",
                            native_rollout_stage: "primary" } }

8. CONFIDENCE NORMALIZATION (inline, no separate stage)
   Replace missing confidences with ln(1 + char_count) + 1.0
   Apply numerically stable softmax
   Record pre_mass=34.2, post_mass=1.0 in AccelerationReport

9. PERSIST STAGE (budget: 20s)
   emit: stage { code: "persist.start" }
   SAVEPOINT sp_persist_1
     INSERT INTO runs (run_id, started_at, ...)
     INSERT INTO segments (42 rows)
     INSERT INTO events (8 rows)
     token.checkpoint() -> Ok (not cancelled)
   RELEASE SAVEPOINT sp_persist_1
   emit: stage { code: "persist.ok" }

10. LATENCY PROFILING
    emit: stage { code: "orchestration.latency_profile" }
    Per-stage utilization: normalize=0.14% (decrease_budget_candidate),
                           backend=2.3% (decrease_budget_candidate),
                           persist=0.5% (decrease_budget_candidate)

11. REPLAY ENVELOPE
    Compute SHA-256(normalized_16k_mono.wav) -> input_content_hash
    Record backend_identity: "whisper-cli", backend_version: "1.7.2"
    Compute SHA-256(raw_backend_json) -> output_payload_hash
    Record native_rollout_stage: "primary"

12. CALIBRATION UPDATE
    Record actual_outcome = 1.0 (success) against the predicted_probability from step 7

13. REPORT ASSEMBLY
    RunReport { run_id, trace_id, request, result, events, evidence, replay, warnings }

14. OUTPUT
    Serialize RunReport as JSON -> stdout
    Exit code 0
```

Total wall time for a 2-minute MP3: typically 5–15 seconds depending on backend and hardware.

---

## Integration Examples

### Pipe Robot Output to jq

```bash
# Extract just the transcript from a robot run
franken_whisper robot run --input audio.mp3 --backend auto 2>/dev/null \
  | jq -r 'select(.event == "run_complete") | .transcript'

# Monitor pipeline progress in real time
franken_whisper robot run --input audio.mp3 --backend auto 2>/dev/null \
  | jq -r 'select(.event == "stage") | "\(.code): \(.message)"'

# Extract all segments with timestamps
franken_whisper robot run --input audio.mp3 --backend auto 2>/dev/null \
  | jq -r 'select(.event == "run_complete") | .segments[] | "[\(.start_sec)s - \(.end_sec)s] \(.text)"'
```

### Batch Transcription Script

```bash
#!/bin/bash
# Transcribe all audio files in a directory
for file in recordings/*.mp3; do
  echo "Transcribing: $file"
  franken_whisper transcribe --input "$file" --json --no-persist \
    | jq -r '.result.transcript' > "${file%.mp3}.txt"
done
```

### Health Check in CI/CD

```bash
# Verify all backends are available before running tests
status=$(franken_whisper robot health 2>/dev/null | jq -r '.overall_status')
if [ "$status" != "ok" ]; then
  echo "Backend health check failed"
  franken_whisper robot health 2>/dev/null | jq '.backends[] | select(.available == false)'
  exit 1
fi
```

### Export and Archive Run History

```bash
# Full export
franken_whisper sync export-jsonl --output ./backup

# Gzip the snapshot out-of-band (the import path auto-detects .gz)
gzip ./backup/*.jsonl

# Re-import into a scratch DB to verify the snapshot round-trips
franken_whisper sync import-jsonl --input ./backup --conflict-policy reject \
  --db /tmp/verify.sqlite3
```

### TTY Audio Over SSH

```bash
# On the remote machine (audio source):
franken_whisper tty-audio encode --input recording.wav \
  | ssh user@local-machine 'franken_whisper tty-audio decode --output received.wav'

# With retransmit recovery for lossy links:
franken_whisper tty-audio encode --input recording.wav > frames.ndjson
cat frames.ndjson | ssh user@remote 'cat > /tmp/frames.ndjson'
# On remote, check for gaps:
ssh user@remote 'cat /tmp/frames.ndjson | franken_whisper tty-audio retransmit-plan'
```

### Library Usage in Rust

```rust
use franken_whisper::model::{
    BackendKind, BackendParams, InputSource, TranscribeRequest,
};
use franken_whisper::orchestrator::FrankenWhisperEngine;
use franken_whisper::storage::RunStore;
use std::path::{Path, PathBuf};

fn transcribe_file(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let request = TranscribeRequest {
        input: InputSource::File { path: PathBuf::from(path) },
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        persist: true,
        db_path: PathBuf::from(".franken_whisper/storage.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    };

    let engine = FrankenWhisperEngine::new()?;
    let report = engine.transcribe(request)?;

    Ok(report.result.transcript)
}

fn query_history(db_path: &str, limit: usize) -> Result<(), Box<dyn std::error::Error>> {
    let store = RunStore::open(Path::new(db_path))?;
    let runs = store.list_recent_runs(limit)?;

    for run in &runs {
        println!("{}: {} ({})", run.run_id, run.transcript_preview, run.backend);
    }

    Ok(())
}
```

### Monitoring Routing Decisions

```bash
# See how the Bayesian router is performing
franken_whisper robot routing-history --limit 20 2>/dev/null \
  | jq '.[] | {decision_id, chosen_action, calibration_score, brier_score, fallback_active}'

# Track correction rates in speculative mode
franken_whisper robot run --input audio.mp3 --speculative \
  --fast-model tiny.en --quality-model large-v3 2>/dev/null \
  | jq 'select(.event == "transcript.speculation_stats")'
```

---

## What Makes This Different

### Learned backend selection

WhisperS2T, `transcribe-anything`, and WhisperLive let you *pick* a backend. `franken_whisper` *learns* which backend to use based on observed outcomes. The Bayesian router maintains Beta-distribution posteriors per backend, tracks calibration via Brier scoring, and falls back to deterministic priority when uncertain, without losing calibration data while the fallback is active.

### Cross-engine conformance validation

The conformance harness compares segment output across engines using a 50 ms canonical timestamp tolerance, text matching, speaker label matching, WER approximation, and replay envelope drift detection. The 5-stage native rollout governance prevents buggy engines from silently degrading quality.

### End-to-end dual-model speculative streaming

A fast model and a quality model run in parallel on overlapping windows; partial transcripts emit immediately and corrections issue when the quality model disagrees. The `CorrectionTracker` adaptively adjusts confirmation thresholds, and the entire speculation flow can run over a TTY link via protocol v2 transcript control frames.

### Full audit trail on every run

Every run is persisted to SQLite with the complete request, result, segments, pipeline events, evidence, and replay envelope. Full and incremental JSONL export with SHA-256 checksums and optional gzip compression.

### Audio decoded as a zero-dependency data type

The built-in Rust decoder handles MP3, AAC, FLAC, WAV, OGG, Vorbis, and ALAC natively. No subprocess, no external binary, no `PATH` dependency. ffmpeg is only the fallback.

### Agent consumption is the primary interface

The `robot` subcommand is the *primary* interface: sequenced NDJSON events with stable schema versioning (v1.0.0), 12 structured error codes (6 robot-facing), health diagnostics, routing history, speculation events, and structured error envelopes even for argument-parsing failures.

### Safe-Rust ASR orchestration

`franken_whisper` enforces `#![forbid(unsafe_code)]`. Note the distinction: `deny` can be overridden per-item, `forbid` cannot. Combined with cooperative cancellation, atomic transactions, bounded finalizers, and RAII cleanup, this gives strong safety guarantees that other wrappers (which all rely on FFI to whisper.cpp or Python interpreters) cannot provide.

---

## Key Documentation

| Document | Description |
|----------|-------------|
| [`docs/tty-audio-protocol.md`](docs/tty-audio-protocol.md) | Complete TTY audio protocol specification |
| [`docs/tty-replay-guarantees.md`](docs/tty-replay-guarantees.md) | Deterministic replay/framing guarantees |
| [`docs/native_engine_contract.md`](docs/native_engine_contract.md) | Native engine replacement interface contract |
| [`docs/engine_compatibility_spec.md`](docs/engine_compatibility_spec.md) | 50 ms timestamp tolerance specification |
| [`docs/conformance-contract.md`](docs/conformance-contract.md) | Cross-engine conformance test contract |
| [`docs/operational-playbook.md`](docs/operational-playbook.md) | Deployment and monitoring guide |
| [`docs/benchmark_regression_policy.md`](docs/benchmark_regression_policy.md) | Performance regression thresholds |
| [`tests/COVERAGE.md`](tests/COVERAGE.md) | Spec clause → test traceability |
| [`RECOVERY_RUNBOOK.md`](RECOVERY_RUNBOOK.md) | Disaster recovery procedures |
| [`SYNC_STRATEGY.md`](SYNC_STRATEGY.md) | One-way sync semantics |
| [`PROPOSED_ARCHITECTURE.md`](PROPOSED_ARCHITECTURE.md) | System architecture design document |
| [`FEATURE_PARITY.md`](FEATURE_PARITY.md) | Legacy feature parity matrix |
| [`CHANGELOG.md`](CHANGELOG.md) | Capability-wave changelog with live commit links |
| [`DISCREPANCIES.md`](DISCREPANCIES.md) | Known native-vs-bridge divergences under investigation |

---

## Worked Examples

These examples walk through the actual math and bit-level operations of the most opaque algorithms in the system, using concrete inputs. They are intentionally arithmetic-heavy: the goal is to make the *behavior* of the algorithms verifiable by hand rather than treating the implementation as a black box.

### Worked Example: A Single Bayesian Routing Decision

Suppose a user runs `franken_whisper transcribe --input meeting.mp3 --backend auto` on a 6-minute audio file with no `--diarize` flag. The user's PATH has `whisper-cli` and `insanely-fast-whisper` installed but no `python3` for `whisper-diarization`. The Brier score from recent runs is 0.18 and the calibration sliding window holds 32 observations.

**Step 1. Detect availability.** The router probes via `which`:

```
whisper_cpp        : available
insanely_fast      : available
whisper_diarization: unavailable
state              : partial_available
```

**Step 2. Establish duration bucket.** 6 minutes = 360 seconds. Bucket: `long` (≥ 60 s).

**Step 3. Compute latency cost per backend.** `latency_cost = base + sqrt(360) × multiplier_normal`:

```
whisper_cpp        : 12.0 + sqrt(360) * 1.0 ≈ 12.0 + 18.97 = 30.97
insanely_fast      :  8.0 + sqrt(360) * 1.0 ≈  8.0 + 18.97 = 26.97
whisper_diarization: 18.0 + sqrt(360) * 1.0 ≈ 18.0 + 18.97 = 36.97
```

**Step 4. Compute quality cost per backend.** `quality_cost = (1.0 - quality_score) × 100` with `diarize=false`:

```
whisper_cpp        : (1.0 - 0.84) * 100 = 16.0
insanely_fast      : (1.0 - 0.80) * 100 = 20.0
whisper_diarization: (1.0 - 0.63) * 100 = 37.0
```

**Step 5. Compute posterior success probability per backend.** Beta priors `(α, β)` updated with the current sliding window. Suppose recent empirical success rates are: `whisper_cpp` 0.93 (20 obs), `insanely_fast` 0.81 (12 obs), `whisper_diarization` no data:

```
whisper_cpp        : α' = 7 + 0.93*20 = 25.6   β' = 3 + 0.07*20 =  4.4
                     p_success ≈ (25.6 + 0.84*2 + 0.3) / (25.6 + 4.4 + 1.68 + 0.3) ≈ 0.870

insanely_fast      : α' = 6 + 0.81*12 =  15.72  β' = 4 + 0.19*12 =  6.28
                     p_success ≈ (15.72 + 0.80*2 + 0.3) / (15.72 + 6.28 + 1.6 + 0.3) ≈ 0.738

whisper_diarization: α  = 5             β  = 5    (prior only)
                     p_success = 0 (unavailable; availability penalty kicks in)
```

**Step 6. Compute failure cost per backend.**

```
whisper_cpp        : (1.0 - 0.870) * 100 = 13.0
insanely_fast      : (1.0 - 0.738) * 100 = 26.2
whisper_diarization: full availability penalty
```

**Step 7. Combine into total cost** (`0.45 × latency + 0.35 × quality + 0.20 × failure`):

```
whisper_cpp        : 0.45*30.97 + 0.35*16.0 + 0.20*13.0
                   = 13.937 + 5.600 + 2.600 = 22.14
insanely_fast      : 0.45*26.97 + 0.35*20.0 + 0.20*26.2
                   = 12.137 + 7.000 + 5.240 = 24.38
whisper_diarization: availability penalty (+1000) dominates → ≈ 1000+
```

**Step 8. Calibration check.** Brier = 0.18 (< 0.35 threshold) and observation count = 32 (≥ 5). Calibration is good, so the router stays in adaptive mode (no static fallback).

**Step 9. Choose action.** Lowest cost is `try_whisper_cpp` at 22.14. The router selects `whisper_cpp`.

**Step 10. Record evidence and prediction.** Push a `RoutingEvidenceLedgerEntry` (decision_id, trace_id, observed_state=`partial_available`, chosen_action=`try_whisper_cpp`, policy_id=`backend-selection-v1.0`, loss_matrix_hash) into the 200-entry circular buffer. Push the predicted probability (0.870) into the calibration sliding window. Emit the `backend.routing.decision_contract` event with full posterior snapshot for the agent.

**Step 11. After the run.** Once the backend finishes, record the actual outcome (1.0 = success or 0.0 = failure) against the prediction. The Brier score is recomputed. The Beta posterior for `whisper_cpp` updates: success bumps `α`, failure bumps `β`.

The same shape of calculation runs for every `--backend auto` decision. The math itself is microseconds; the audit trail is the expensive part, and it's worth it.

### Worked Example: Mu-Law Encoding a Single Sample

In practice `franken_whisper` shells out to `ffmpeg -f mulaw -ar 8000 -ac 1` for the actual transcoding. The bit-level math below is the *codec definition*, useful for debugging a captured wire trace or for understanding what ffmpeg is doing on your behalf.

Take a 16-bit signed PCM sample value `s = 12_345` (positive). Encode to mu-law:

```
1. Clamp to ±32635:
   s = min(max(12345, -32635), 32635) = 12345 ✓

2. Extract sign (0 = positive, 1 = negative):
   sign = 0

3. Add bias:
   magnitude = |12345| + 132 = 12477  (binary: 0011 0000 1011 1101)

4. Find segment = position of the highest set bit minus 7 (clamped to [0, 7]).
   Highest set bit in 12477 is bit 13.
   segment = 13 - 7 = 6.

5. Extract 4-bit mantissa = bits (segment+3 .. segment) shifted down:
   shift the magnitude right by (segment + 3) = 9 bits:
   12477 >> 9 = 24 = 0001 1000
   mantissa = 24 & 0xF = 8.

6. Combine into 8 bits: (sign << 7) | (segment << 4) | mantissa
                       = (0 << 7) | (6 << 4) | 8
                       = 0110 1000 = 0x68 = 104

7. Invert all bits (wire format convention):
   ~0x68 = 0x97 = 151

   Wire byte = 0x97
```

To decode `0x97` back to PCM:

```
1. Invert: ~0x97 = 0x68
2. sign = 0, segment = 6, mantissa = 8
3. Reconstruct magnitude:
   m_raw = ((mantissa << 3) + 132) << (segment + 1) - 132
         = ((8 << 3) + 132) << 7 - 132
         = (64 + 132) << 7 - 132
         = 196 << 7 - 132
         = 25088 - 132 = 24956
4. Apply sign: +24956
```

The reconstructed value (24956) differs from the original (12345). Mu-law is lossy: it preserves dynamic range (loud sounds remain loud, quiet sounds remain quiet) but quantizes intermediate values. The compression ratio is exactly 2:1 (16 bits → 8 bits) and the error is logarithmically distributed, so perceived quality stays high for speech. Adding zlib compression on top of the mu-law byte stream typically achieves another 30–50% reduction, since mu-law-encoded speech has substantial statistical redundancy.

### Worked Example: Brier Score Decomposition

The Brier score `B = (1/N) Σ (p_i − o_i)²` admits Murphy's three-component decomposition:

```
B = reliability − resolution + uncertainty
```

where, partitioning the N predictions into K bins by predicted probability `p_k` (with `n_k` observations per bin and observed mean outcome `ō_k` per bin):

- **reliability** = `(1/N) Σ_k n_k (p_k − ō_k)²`. How close predicted probabilities are to observed frequencies (smaller is better; perfect reliability = 0).
- **resolution** = `(1/N) Σ_k n_k (ō_k − ō)²`. How much predictions vary around the base rate `ō` (larger is better).
- **uncertainty** = `ō (1 − ō)`. Irreducible noise in the outcome (a property of the data, not the predictor).

A backend router that always predicts the base rate has perfect reliability (0) but zero resolution → Brier score = uncertainty. A perfectly calibrated and confident router has zero reliability AND large resolution → Brier score much smaller than uncertainty.

This decomposition explains why `franken_whisper` falls back at Brier > 0.35: if reliability is bad (predicted probabilities don't match observed frequencies), the router is making systematically biased predictions and the *safer* play is the static priority list. Once the calibration sliding window accumulates enough corrective observations to bring reliability back down, adaptive routing automatically resumes. The fallback path's calibration recording is what makes this recovery possible.

---

## Anatomy Series

### Anatomy of a Routing Decision

```
0. backend.routing.decision_contract emit (event sequence #N)
   payload includes: state, available[], duration_bucket, diarize, posteriors[], loss_matrix, chosen

1. probe_system_health()
   for each backend kind:
     - check FRANKEN_WHISPER_<KIND>_BIN override
     - else `which <default_binary>` (e.g. `which whisper-cli`)
     - record availability bool

2. derive observed_state:
   if all 3 available     → all_available
   if 1-2 available       → partial_available
   if 0 available         → none_available  (will emit FW-BACKEND-UNAVAILABLE)

3. build action set: [try_whisper_cpp, try_insanely_fast, try_diarization, fallback_error]
   reorder per diarize flag (diarization-first when --diarize is set)

4. for each (state, action) cell of the 3×4 loss matrix:
   compute latency_cost = base + sqrt(duration_secs) * multiplier
   compute quality_cost = (1 - quality_proxy(action, request)) * 100
   compute failure_cost = (1 - p_success(action, posterior, request)) * 100
   weighted = 0.45*latency + 0.35*quality + 0.20*failure
   add availability penalty (+0 / +333 / +1000)

5. calibration check:
   read sliding window (50 entries)
   if observations < 5: use_static_priority = true
   if calibration_score < 0.3: use_static_priority = true
   if brier_score > 0.35: use_static_priority = true

6. select action:
   if use_static_priority:
       follow deterministic priority list (whisper_cpp > insanely_fast > whisper_diarization)
   else:
       argmin over cost cells

7. record:
   - push RoutingEvidenceLedgerEntry into 200-entry circular buffer
   - push (predicted_probability, observed_at) into 50-entry calibration sliding window
     (recorded even when fallback path executes, so calibration keeps learning)
   - emit `backend.routing.decision_contract` event with full posterior snapshot

8. dispatch to chosen backend (or return FW-BACKEND-UNAVAILABLE if action = fallback_error)
```

### Anatomy of a TTY Audio Session

```
ENCODER                                                 DECODER

1. encode_wav_to_frames(input):
   load WAV, verify 16 kHz mono
   transcode PCM → mu-law (2:1 compression)
   chunk into chunk_ms windows (default 200 ms = 3200 mu-law bytes)

2. emit Handshake frame:
   min_version=1, max_version=2, codecs=[mulaw+zlib+b64]

                                              3. receive Handshake:
                                                 select max compatible version
                                                 select first compatible codec
                                                 emit HandshakeAck

4. for each chunk:
   payload = zlib_compress(mulaw_bytes)
   crc32   = crc32(mulaw_bytes)
   sha256  = sha256(mulaw_bytes)
   payload_b64 = base64(payload)
   emit AudioFrame { seq, codec, sample_rate_hz, channels, payload_b64, crc32, sha256 }

                                              5. for each AudioFrame:
                                                 validate codec
                                                 if seq != expected: record gap
                                                 if seq == previous: record duplicate, skip
                                                 decode b64 → zlib_decompress → mulaw
                                                 verify crc32 and sha256 against raw mulaw
                                                 if mismatch: integrity_failure++, drop frame
                                                 mulaw → PCM, append to WAV buffer

6. emit SessionClose { reason=Complete, last_data_seq=N }

                                              7. receive SessionClose:
                                                 if any seq <= N missing:
                                                     emit RetransmitRequest with missing sequences
                                                 else: emit Ack { up_to_seq=N }; finalize WAV

8. on RetransmitRequest:
   look up requested sequences in send buffer
   emit RetransmitResponse with payloads
   if adaptive bitrate active and loss > 10%, escalate compression and FEC

9. retransmit_loop escalates across rounds:
   Round 1 (Simple)   : emit 1 copy per missing frame
   Round 2 (Redundant): emit 2 copies
   Round 3 (Escalate) : emit 4 copies
```

If link quality is poor (frame_loss_rate > 10%) the encoder pre-emptively duplicates *critical* control frames (Handshake, SessionClose, Ack) 3× so they survive independent loss probabilistically. The decoder dedupes by sequence number, so duplication never produces duplicate output.

### Anatomy of a Conformance Check

```
1. load expected segments (golden fixture from tests/fixtures/conformance/...)
2. load observed segments (from bridge or native engine output)

3. validate segment invariants on each side:
   for each segment:
     assert start_sec, end_sec are finite (not NaN, not ±∞)
     assert start_sec >= 0
     assert end_sec >= start_sec
     assert confidence ∈ [0.0, 1.0]
     assert text is non-empty
   for each adjacent pair:
     assert segments[i].end_sec <= segments[i+1].start_sec + overlap_epsilon (1 µs default)

4. compare lengths:
   N = expected.len(), M = observed.len()
   if N != M: length_mismatch = true

5. for i in 0..min(N, M):
   if require_text_exact and text differs: text_mismatches++
   if require_speaker_exact and speaker differs: speaker_mismatches++
   if |expected[i].start - observed[i].start| > 0.05: timestamp_violations++
   if |expected[i].end   - observed[i].end|   > 0.05: timestamp_violations++

6. compute WER via Levenshtein on word sequences:
   tokenize both transcripts by whitespace
   edit_distance = levenshtein(expected_words, observed_words)
   wer = edit_distance / max(reference_length, 1) clamped to [0, 1]

7. return SegmentComparisonReport {
     length_mismatch, text_mismatches, speaker_mismatches,
     timestamp_violations, segments_compared, wer_approx
   }

8. at rollout-stage promotion:
   conformance gate requires:
     timestamp_violations == 0 (50 ms tolerance)
     all segment invariants pass
     replay envelope matches (input_hash + backend_version → output_hash)
   if any check fails, promotion is blocked
```

### Anatomy of a Speculative Window

```
1. WindowManager.next_window():
   advance sliding window by (window_ms - overlap_ms)
   compute sha256 of window audio content
   allocate window_id

2. ConcurrentTwoLaneExecutor.spawn_lanes():
   fast_lane:     spawn fast_model on window with timeout budget
   quality_lane:  spawn quality_model on window with timeout budget

3. fast_lane returns first (by design):
   build PartialTranscript { window_id, model_id=fast, segments, status=Pending }
   emit transcript.partial event with monotonically increasing seq

4. quality_lane returns:
   compute CorrectionDrift:
     wer_approx           = levenshtein(fast_words, quality_words) / max(...)
     confidence_delta     = |mean(fast_conf) - mean(quality_conf)|
     segment_count_delta  = quality.len() - fast.len()
     text_edit_distance   = levenshtein(fast_text, quality_text)

5. CorrectionTracker.evaluate(drift):
   if all metrics within CorrectionTolerance:
       emit transcript.confirm {
           seq, window_id, quality_model_id, drift, latency_ms, ts
       }
       status = Confirmed
   else:
       emit transcript.retract {
           retracted_seq, window_id, reason, quality_model_id, ts
       }
       emit transcript.correct {
           correction_id, replaces_seq, window_id,
           segments (from quality model), drift, latency_ms, ts
       }
       status = Retracted

6. SpeculationWindowController.update():
   record outcome (correction or not) into observation buffer (size 20)
   compute Brier score on corrections vs. predicted corrections
   if brier > 0.25 and observations >= 10:
       use deterministic fallback (fixed window_ms)
   else:
       adjust window_ms by ±step_ms based on correction rate pattern

7. periodic emit speculation.stats {
       windows_processed, corrections_emitted, confirmations_emitted,
       correction_rate, mean_fast_latency_ms, mean_quality_latency_ms,
       current_window_size_ms, mean_drift_wer, ts
   }
```

The whole flow is end-to-end safe to ship over TTY: every event has a wire-efficient `TranscriptSegmentCompact` representation (`s` / `e` / `t` / `sp` / `c` single-letter fields), and the controller's adaptive behavior is bounded by its alien-artifact contract.

---

## Operational Topics

### State Directory Layout

The state directory (default `.franken_whisper/`, override via `FRANKEN_WHISPER_STATE_DIR`) holds everything `franken_whisper` writes to disk:

```
.franken_whisper/
├── storage.sqlite3              # primary SQLite database (WAL mode)
├── storage.sqlite3-wal          # SQLite write-ahead log
├── storage.sqlite3-shm          # SQLite shared memory file (transient)
├── locks/
│   └── sync.lock                # JSON lock file for sync export/import
│                                # { "pid", "created_at_rfc3339", "operation" }
├── tools/
│   └── ffmpeg/
│       └── bin/
│           ├── ffmpeg           # auto-provisioned static binary (Linux x86_64)
│           └── ffprobe          # auto-provisioned static binary
├── work/
│   └── fw-run-{uuid}/           # per-run work directory (LIFO finalizers clean up)
│       ├── normalized_16k_mono.wav
│       ├── backend_output.json  # raw backend output preserved for replay
│       └── ...                  # any subprocess scratch files
└── snapshots/                   # (when sync export-jsonl --output is relative)
    └── {timestamp}/
        ├── runs.jsonl(.gz)
        ├── segments.jsonl(.gz)
        ├── events.jsonl(.gz)
        └── manifest.json
```

The state directory respects `XDG_STATE_HOME`: when set, `tools/` is rooted at `$XDG_STATE_HOME/franken_whisper/` so multiple project trees can share auto-provisioned ffmpeg without duplicating downloads.

### CLI Exit Codes

| Code | Meaning | Triggered by |
|------|---------|--------------|
| 0 | Success | Pipeline completed and (if requested) JSON/NDJSON emitted |
| 1 | General error | Any `FwError` propagated to `main()` (`FW-IO`, `FW-CMD-*`, `FW-STORAGE`, `FW-BACKEND-UNAVAILABLE`, `FW-INVALID-REQUEST`, etc.) |
| 2 | Argument parse failure | `clap`'s default exit code on invalid command-line syntax (occurs before the binary's own error handling runs) |
| 130 | Cancelled | Ctrl+C (POSIX convention: 128 + SIGINT=2) |

`FW-*` errors all exit with code 1; the structured machine-readable distinction lives in the error code itself, not the exit code. In robot mode, agents should inspect the final NDJSON event (`run_complete` vs. `run_error`) to determine success rather than relying on exit codes; the process exit code is reserved for catastrophic failures that prevented the robot stream from completing at all.

### Logging

`franken_whisper` uses `tracing` for structured diagnostics. Set `RUST_LOG` to control verbosity.

**Text format (default):**

```bash
RUST_LOG=info franken_whisper transcribe --input audio.mp3 --json
# 2026-05-17T12:34:56Z  INFO franken_whisper::orchestrator: stage=normalize started budget_ms=180000
# 2026-05-17T12:34:56Z  INFO franken_whisper::audio: builtin decoder selected format=mp3
# 2026-05-17T12:35:00Z  INFO franken_whisper::orchestrator: stage=normalize ok service_ms=257
```

**JSON format (for log aggregation):**

```bash
RUST_LOG=info RUST_LOG_FORMAT=json franken_whisper transcribe --input audio.mp3 --json
# {"timestamp":"2026-05-17T12:34:56.123Z","level":"INFO","target":"franken_whisper::orchestrator","fields":{"stage":"normalize","status":"started","budget_ms":180000}}
```

**Filtering recipes:**

```bash
# Backend routing only
RUST_LOG=franken_whisper::backend=debug franken_whisper transcribe --input audio.mp3 --json

# TTY audio decode telemetry
RUST_LOG=franken_whisper::tty_audio=trace franken_whisper tty-audio decode --output a.wav < frames.ndjson

# Sync internals
RUST_LOG=franken_whisper::sync=debug franken_whisper sync export-jsonl --output ./snapshot

# Everything verbose except hyper / clap noise
RUST_LOG=franken_whisper=debug,clap=warn franken_whisper transcribe --input audio.mp3 --json
```

Secret values (HuggingFace tokens, in particular) never enter the log stream; `render_command_for_log()` redacts them before any subprocess invocation is recorded.

### Concurrent Agents Pattern

Multiple agents can persist into a single SQLite database safely:

```rust
// Agent A's session
let store = RunStore::open(&db_path)?;
let session = store.begin_concurrent_session("agent_alpha")?;
session.persist_report(&report_a)?;
session.commit()?;  // RELEASE SAVEPOINT fw_session_agent_alpha

// Agent B can run concurrently against the same database:
let store_b = RunStore::open(&db_path)?;
let session_b = store_b.begin_concurrent_session("agent_beta")?;
session_b.persist_report(&report_b)?;
session_b.commit()?;
```

**Concurrency guarantees:**

- WAL mode allows multiple readers and one writer at any instant.
- `busy_timeout=5000` lets writers wait up to 5 seconds on lock contention.
- Each `begin_concurrent_session` opens a named savepoint (`fw_session_<name>`); session names are validated to `[A-Za-z0-9_]+` so they cannot escape into SQL.
- Persist transactions retry on `SQLITE_BUSY` with linear backoff (`5 × (attempt + 1)` ms, up to 8 attempts).
- The cancellation token is checked before every `COMMIT` and `RELEASE`, so Ctrl+C never leaves the database in an inconsistent state.

**Coordination via MCP Agent Mail.** When the agents are coordinating work (not just sharing a database), the same project conventions used in this repository apply: each agent registers via `register_agent`, reserves file paths via `file_reservation_paths` with the issue ID in `reason`, and communicates via threaded messages keyed by `br-NNN` issue IDs. The database concurrency primitives above are necessary but not sufficient; the agent-coordination layer prevents two agents from *trying* to do the same work in the first place.

### Database Maintenance

The SQLite database accumulates rows over time. For long-lived deployments:

```sql
-- Inspect WAL growth and freelist
.open .franken_whisper/storage.sqlite3
PRAGMA wal_checkpoint(TRUNCATE);   -- force a full WAL checkpoint
PRAGMA freelist_count;             -- how many unused pages
PRAGMA page_count;                 -- total pages
PRAGMA integrity_check;            -- verify database integrity

-- Reclaim space after large deletes (run when no writers are active)
VACUUM;
```

`robot health` reports all of these in its `database` field. A reasonable monitoring rule: alert when `wal_checkpoint.log_frames` stays above a threshold across consecutive probes, which indicates a long-lived read transaction is preventing checkpointing.

**Backup procedure (recommended):**

```bash
# 1. Export the database state to portable JSONL (incremental or full)
franken_whisper sync export-jsonl --output backup/$(date +%F)/ --gzip

# 2. Verify the export integrity
franken_whisper sync import-jsonl --input backup/$(date +%F)/ \
  --conflict-policy skip --dry-run

# 3. Copy the snapshot off-host
rsync -a backup/$(date +%F)/ user@host:/srv/franken_whisper/backups/
```

JSONL snapshots are the canonical recovery artifact. They are portable across machines and SQLite versions, deduplicate cleanly across incremental exports, and carry SHA-256 checksums in `manifest.json` for end-to-end integrity verification.

### Recovery from Corruption

If `PRAGMA integrity_check` reports corruption or the database refuses to open:

1. **Don't run `VACUUM` on a corrupted database**; it can make things worse.
2. **Try to dump what's salvageable** by exporting recent runs from a different process:
   ```bash
   franken_whisper sync export-jsonl --output recovery/ --db <corrupted>.sqlite3
   ```
3. **Move the corrupted database aside**:
   ```bash
   mv .franken_whisper/storage.sqlite3{,.corrupted-$(date +%F)}
   ```
4. **Rebuild from the last good JSONL snapshot**:
   ```bash
   franken_whisper sync import-jsonl --input backup/latest/ --conflict-policy reject
   ```
5. **Verify** with `franken_whisper robot health` → `database.issues == []`.

The export → reimport round-trip is deterministic: replay envelopes survive intact, evidence ledgers are preserved, and every persisted run remains addressable by its `run_id` after recovery.

### Choosing a Backend (Decision Guide)

The `auto` router will normally pick the right backend, but explicit selection is sometimes useful. Quick decision tree:

```
Do you need speaker diarization?
├─ YES, with high-quality pyannote models      → whisper_diarization
├─ YES, with built-in fast-only speaker turns  → whisper_cpp + --tiny-diarize
└─ NO ──┐
        Do you have a CUDA / MPS GPU?
        ├─ YES, batching long audio aggressively  → insanely_fast
        ├─ YES, low-latency single-stream         → whisper_cpp (--no-gpu false)
        └─ NO  → whisper_cpp (CPU + Metal/CUDA fallback inside whisper.cpp)
```

For mixed workloads, leave `--backend auto` and let the Bayesian router learn from outcomes. The static priority list (the fallback when calibration is poor) already encodes a reasonable default.

### Audio Quality Considerations

Whisper models require 16 kHz mono audio. `franken_whisper`'s normalization stage performs three operations whose tradeoffs matter:

| Operation | Method | Quality cost |
|-----------|--------|--------------|
| Channel mixing | sample-by-sample average across channels (using actual frame length) | minimal for centered voice; spatial cues lost |
| Sample rate conversion | linear interpolation | introduces ~−40 dB aliasing above 8 kHz |
| Bit depth conversion | clamp-and-round 16-bit signed PCM | imperceptible for speech |

The linear resampler is intentional. A higher-quality polyphase resampler would produce slightly cleaner spectra but require an FFT or a large filter bank, neither of which can be implemented in safe Rust without performance regressions. Whisper models tolerate the aliasing well; their training corpus already contains all sorts of resampling artifacts. For pristine archival work where you need every dB of fidelity, pre-resample with `ffmpeg -ar 16000 -ac 1 -filter:a "aresample=resampler=soxr"` and then feed the WAV in via `--input`; the normalize stage will detect it's already 16 kHz mono and become a passthrough.

### systemd Integration

```ini
# /etc/systemd/system/franken-whisper-watcher.service
[Unit]
Description=franken_whisper recording watcher
After=network.target

[Service]
Type=simple
User=transcribe
Environment=FRANKEN_WHISPER_HF_TOKEN=hf_...
Environment=FRANKEN_WHISPER_STATE_DIR=/var/lib/franken_whisper
Environment=RUST_LOG=info
ExecStart=/usr/local/bin/franken_whisper robot run --stdin --backend auto
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/franken_whisper
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

For batch jobs, prefer a `Type=oneshot` unit and a `*.timer` for scheduling. Robot mode's NDJSON output lands cleanly in `journalctl`; pipe it through `jq` to extract specific events.

### Cross-Compilation

| Target | Status | Notes |
|--------|--------|-------|
| `x86_64-unknown-linux-gnu` | Tier 1 (CI-tested) | Auto-provisioned ffmpeg works |
| `aarch64-unknown-linux-gnu` | Tier 1 (CI-tested) | Install ffmpeg via package manager |
| `x86_64-apple-darwin` | Tier 1 (CI-tested) | `brew install ffmpeg` |
| `aarch64-apple-darwin` | Tier 1 (CI-tested) | `brew install ffmpeg` |
| `x86_64-pc-windows-msvc` | Tier 1 (CI-tested) | Install ffmpeg manually; `dshow` for mic |
| `armv7-unknown-linux-musleabihf` | Best-effort | No auto-provisioned ffmpeg; cross-build via `cross` |
| WebAssembly | Not supported | Symphonia + clap pull in OS APIs |

Cross-builds typically use [`cross`](https://github.com/cross-rs/cross):

```bash
cargo install cross
cross build --release --target aarch64-unknown-linux-gnu
```

### Build Reproducibility

The release profile (`opt-level = "z"`, `lto = true`, `codegen-units = 1`, `panic = "abort"`, `strip = true`) plus a pinned nightly toolchain (`rust-toolchain.toml`) makes binaries close to byte-reproducible. Two builds of the same commit on the same host typically differ only in embedded timestamps and the build-id section. For truly reproducible builds, additionally set `SOURCE_DATE_EPOCH` and use `cargo build --frozen` against a checked-in `Cargo.lock`.

### Why Rust 2024 / Why Nightly

- **Rust 2024 edition** lets the codebase use `let-else`, expanded `impl Trait` positions, RPITIT (return-position `impl Trait` in traits), and `gen` blocks where they make pipeline code clearer. None are accidental complexity; each appears in code that would otherwise be more awkward to read.
- **Nightly toolchain** is pinned in `rust-toolchain.toml` so local builds match CI. Required for a handful of unstable features that materially improve ergonomics (e.g., features used in the `frankentui`/`frankentorch` integrations). The pin is conservative; the channel is bumped only when CI is green across all platforms.
- **`#![forbid(unsafe_code)]`** is stronger than `deny`: it cannot be overridden per-item. This is enforced at the crate root, so every dependency is held to the same standard via lint propagation where possible.

---

## Threat Model

`franken_whisper` is a CLI tool that runs on the user's own machine. Its threat model reflects that scope.

### What it defends against

- **Accidental secret leakage in logs.** `render_command_for_log()` redacts `--hf-token` values before any subprocess command is recorded in tracing output, error messages, snapshots, or test fixtures. The HuggingFace token never appears in any artifact `franken_whisper` writes.
- **Path-traversal via stdin input.** When reading from stdin, the temporary file's extension is sanitized so a hostile content stream cannot smuggle a `../../../etc/passwd`-style filename into ffmpeg.
- **SQL injection via session names.** Concurrent session names are validated to `[A-Za-z0-9_]+` before being interpolated into `SAVEPOINT fw_session_<name>` / `RELEASE SAVEPOINT fw_session_<name>` statements. Table identifiers in sync imports are explicitly quoted.
- **Subprocess pipe deadlock.** Subprocess stdout and stderr are drained on dedicated reader threads so a backend producing megabytes of output cannot wedge the orchestrator.
- **Partial database writes on crash / cancellation.** All persistence uses savepoints; the cancellation token is checked before `COMMIT` / `RELEASE`. SQLite's journal recovery handles the death-during-COMMIT case.
- **Race conditions on the sync lock.** The lock file is kept alive while the owning PID runs and age-gated when the PID is unknown. EPERM on `kill -0` is treated as alive (the process exists but isn't ours). Windows `tasklist` errors are treated as unknown.
- **Replay-envelope drift going undetected.** The conformance harness compares replay envelopes (input hash + backend identity + backend version → output hash) at every run. The 5-stage native engine rollout blocks promotion of any native engine whose timestamps drift more than 50 ms from the bridge for the same audio.

### What it does *not* defend against

- **A hostile backend binary.** If `whisper-cli` or `insanely-fast-whisper` is itself malicious, `franken_whisper` cannot detect that. Use distribution-trusted binaries.
- **A hostile audio file** (e.g., crafted to exploit a decoder vulnerability). `symphonia` is a pure-Rust decoder with strong safety guarantees; the ffmpeg fallback path is constrained by `-vn -ar 16000 -ac 1 -c:a pcm_s16le` so options injection is hard, but a zero-day in the decoder itself is out of scope.
- **A hostile model file.** Whisper models are downloaded by the backend, not by `franken_whisper`. Pin model checksums in your deployment if this matters.
- **Side-channel leaks from inference timing.** Latency profiling produces visible timing data; an attacker who can observe `orchestration.latency_profile` events could in theory infer audio properties. In the intended single-user threat model this is not a concern.
- **Network adversaries.** `franken_whisper` does no network I/O during transcription. The only network-adjacent code is the optional ffmpeg auto-provisioner (Linux x86_64, HTTPS download from `johnvansickle.com`) and HuggingFace model fetching by the diarization backend. Both can be disabled.

### Trust boundaries

```
+--------------------------------------------------------------+
|  USER PROCESS (franken_whisper)                              |
|  - parses CLI                                                |
|  - reads/writes SQLite                                       |
|  - emits NDJSON on stdout                                    |
+--------------------------------------------------------------+
                        |
        (subprocess boundary; SIGKILL on cancellation)
                        |
+--------------------------------------------------------------+
|  BACKEND SUBPROCESS (whisper-cli / insanely-fast-whisper /   |
|                      python3+pyannote / ffmpeg)              |
|  - own user, own memory                                      |
|  - own stdin/stdout/stderr pipes (drained on dedicated       |
|    reader threads)                                           |
|  - timeout-bounded                                           |
|  - cleaned up by finalizers on cancel/panic                  |
+--------------------------------------------------------------+
```

The orchestrator treats backend output as untrusted JSON: it must parse correctly, segments must satisfy invariants, and replay envelopes must validate. Any failure is a structured `FW-*` error, not a panic.

---

## Use Cases Gallery

| Use case | Configuration sketch |
|----------|----------------------|
| **Meeting transcription with speakers** | `transcribe --input meeting.mp3 --diarize --min-speakers 2 --max-speakers 8 --json` |
| **Podcast batch processing** | `for f in podcasts/*.mp3; do franken_whisper transcribe --input "$f" --backend whisper_cpp --model large-v3 --json; done` |
| **Live transcription dashboard** | `robot run --mic --mic-seconds 300 --speculative --fast-model tiny.en --quality-model large-v3` piped to a Server-Sent Events translator |
| **Voicemail archival** | `transcribe --stdin --backend auto --json` invoked from a mail-handler hook |
| **Live event captioning over SSH** | `tty-audio encode --input mic.wav` ➜ SSH ➜ `tty-audio decode --output a.wav && franken_whisper transcribe --input a.wav --json` |
| **Multi-language conference transcription** | `transcribe --input session.mp4 --language ja --translate --json` (transcribe Japanese, translate to English) |
| **Compliance / QA scanning** | `transcribe --input call.wav --json` piped to `jq` for regex matching on `segments[].text` |
| **Voice-note search index** | Persist all runs into a dedicated DB, `sync export-jsonl` nightly, ingest `segments.jsonl` into a full-text search index keyed by `(run_id, idx)` |
| **Forensic transcription with audit trail** | `--json --output-srt --output-vtt` plus a snapshot of the replay pack for chain-of-custody |
| **Karaoke / lyrics alignment** | `transcribe --input song.mp3 --output-lrc --json` then post-process with the word-level timestamps via library API |
| **Real-time accessibility captioning** | `robot run --mic --speculative` with `transcript.partial` driving a low-latency UI and `transcript.confirm` / `transcript.correct` updating the canonical text |
| **Air-gapped sensitive recordings** | Disable all network I/O (`FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG=0`, manual model deployment, no diarization), use `--no-persist` if even local storage is forbidden |

The common theme: `franken_whisper` is the same binary in every use case. The configuration changes; the integration story (NDJSON events, structured errors, deterministic replay) does not.

---

## JSON Schema Reference

Run `franken_whisper robot schema` to dump the full machine-readable schema for every robot event. The required fields per event (as enforced by `robot_contract_tests.rs` against the `*_REQUIRED_FIELDS` constants in `src/robot.rs`) are:

```text
run_start         : event, schema_version, request
run_error         : event, schema_version, code, message
stage             : event, schema_version, run_id, seq, ts, stage, code, message, payload
run_complete      : event, schema_version, run_id, trace_id, started_at, finished_at,
                    backend, language, transcript, segments, acceleration, warnings, evidence
backends.discovery: event, schema_version, backends
routing_decision  : event, schema_version, run_id, ts, code
transcript.partial: event, schema_version, run_id, seq, ts, text, start_sec, end_sec,
                    confidence, speaker
transcript.confirm: event, schema_version, run_id, seq, window_id, quality_model_id,
                    drift, latency_ms, ts
transcript.retract: event, schema_version, run_id, retracted_seq, window_id, reason,
                    quality_model_id, ts
transcript.correct: event, schema_version, run_id, correction_id, replaces_seq, window_id,
                    segments, drift, latency_ms, ts
health.report     : event, schema_version, ts, backends, ffmpeg, database, resources,
                    overall_status
speculation.stats : event, schema_version, run_id, windows_processed, corrections_emitted,
                    confirmations_emitted, correction_rate, mean_fast_latency_ms,
                    mean_quality_latency_ms, current_window_size_ms, mean_drift_wer, ts
```

Agents can `assert` these fields on every parsed event and abort early on contract violations rather than partially handling malformed input.

---

## Embedding in Other Languages

Although `franken_whisper` is written in Rust, the robot interface makes it polyglot-friendly.

**Python:**

```python
import json, subprocess

proc = subprocess.Popen(
    ["franken_whisper", "robot", "run", "--input", "audio.mp3", "--backend", "auto"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)
for line in proc.stdout:
    event = json.loads(line)
    if event["event"] == "run_complete":
        print(event["transcript"])
        break
```

**Node.js:**

```javascript
import { spawn } from "node:child_process";
import readline from "node:readline";

const proc = spawn("franken_whisper", ["robot", "run", "--input", "audio.mp3", "--backend", "auto"]);
const rl = readline.createInterface({ input: proc.stdout });
for await (const line of rl) {
  const event = JSON.parse(line);
  if (event.event === "run_complete") {
    console.log(event.transcript);
    break;
  }
}
```

**Go:**

```go
package main

import (
    "bufio"; "encoding/json"; "fmt"; "os/exec"
)

func main() {
    cmd := exec.Command("franken_whisper", "robot", "run", "--input", "audio.mp3", "--backend", "auto")
    out, _ := cmd.StdoutPipe()
    cmd.Start()
    scanner := bufio.NewScanner(out)
    for scanner.Scan() {
        var event map[string]any
        json.Unmarshal(scanner.Bytes(), &event)
        if event["event"] == "run_complete" {
            fmt.Println(event["transcript"])
            return
        }
    }
}
```

Each integration is essentially "spawn the binary, parse one JSON line at a time, dispatch on `event`." There is no FFI, no shared library, no per-language SDK to maintain. The contract is the NDJSON schema, and the schema is versioned.

---

## Output Format Reference

`franken_whisper transcribe` can emit several text formats in parallel by combining flags. The same `RunReport` drives every formatter, so a single transcription can produce a `.txt`, a `.vtt`, an `.srt`, a `.csv`, an extended `.json`, and an `.lrc` together. Sample shapes:

**`--output-txt`** is a plain text dump (whisper.cpp side-output style):

```
[00:00.000 --> 00:02.500]  Hello, this is a test recording.
[00:02.500 --> 00:05.100]  The system is working as expected.
[00:05.100 --> 00:07.300]  [SPEAKER_01] All clear on my end.
```

**`--output-vtt`** is WebVTT subtitles (HTML5-native):

```
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello, this is a test recording.

00:00:02.500 --> 00:00:05.100
The system is working as expected.

00:00:05.100 --> 00:00:07.300
<v SPEAKER_01>All clear on my end.</v>
```

**`--output-srt`** is SRT subtitles (de-facto industry standard):

```
1
00:00:00,000 --> 00:00:02,500
Hello, this is a test recording.

2
00:00:02,500 --> 00:00:05,100
The system is working as expected.

3
00:00:05,100 --> 00:00:07,300
[SPEAKER_01] All clear on my end.
```

**`--output-csv`** is a segment-per-row tabular form for spreadsheet ingestion:

```csv
idx,start_sec,end_sec,speaker,text,confidence
0,0.000,2.500,,Hello, this is a test recording.,0.93
1,2.500,5.100,,The system is working as expected.,0.91
2,5.100,7.300,SPEAKER_01,All clear on my end.,0.88
```

**`--output-json-full`** is extended JSON with full segment + acceleration + replay metadata (a superset of `--json`'s default summary form).

**`--output-lrc`** is LRC karaoke format keyed to milliseconds:

```
[ti:test recording]
[00:00.00]Hello, this is a test recording.
[00:02.50]The system is working as expected.
[00:05.10][SPEAKER_01] All clear on my end.
```

**Combining outputs.** Output flags are additive; supply as many as you want. Each writes to a sibling file next to `--transcript-path` (or a default derived from the input filename). The flags don't suppress JSON on stdout: `--json --output-srt --output-vtt` writes the full JSON report to stdout *and* drops two subtitle files on disk.

---

## Health Report Field Reference

`franken_whisper robot health` emits a `health.report` event with this structure (real example, fields annotated):

```jsonc
{
  "event": "health.report",
  "schema_version": "1.0.0",
  "ts": "2026-05-17T12:34:56Z",
  "backends": [                       // one entry per known backend
    {
      "name": "whisper.cpp",          // human-readable backend name
      "available": true,              // PATH probe + override env var honored
      "path": "/usr/local/bin/whisper-cli",  // resolved binary path (null if missing)
      "version": "1.7.2",             // probed via `--version` invocation (null if probe fails)
      "issues": []                    // free-text problems (e.g., "version too old")
    },
    {"name": "insanely-fast-whisper", "available": false, "path": null, "version": null,
     "issues": ["binary not found on PATH; override via FRANKEN_WHISPER_INSANELY_FAST_BIN"]},
    {"name": "whisper-diarization",   "available": true,  "path": "/usr/bin/python3",
     "version": "3.11.4", "issues": []}
  ],
  "ffmpeg": {                         // single dependency, same shape
    "name": "ffmpeg", "available": true,
    "path": "/usr/bin/ffmpeg", "version": "6.0", "issues": []
  },
  "database": {                       // SQLite store at --db
    "name": "database", "available": true,
    "path": ".franken_whisper/storage.sqlite3",
    "version": "schema_v3",
    "issues": []
    // The full StorageDiagnostics (page_count, page_size, journal_mode,
    // wal_checkpoint{busy, log_frames, checkpointed_frames}, freelist_count,
    // integrity_check) is available via the library's RunStore.diagnostics();
    // the robot event surfaces only the user-relevant subset.
  },
  "resources": {                      // host headroom
    "disk_free_bytes": 50000000000,
    "disk_total_bytes": 250000000000,
    "memory_available_bytes": 8000000000,
    "memory_total_bytes": 16000000000
  },
  "overall_status": "ok"              // "ok" / "degraded" / "error"
}
```

**`overall_status` derivation.**

- `ok`: at least one backend available, ffmpeg present (or not needed for current workload), database healthy.
- `degraded`: one or more backends missing but at least one usable; or ffmpeg missing but the built-in decoder covers the formats actually being used.
- `error`: no backends available, OR database integrity check failed, OR critical dependency missing.

**Field stability.** The schema is part of the `1.0.0` contract. New fields may be added but no existing fields will be renamed or removed at this schema version. Agents should ignore unknown fields rather than fail on them.

---

## Failure Modes Catalog

Each `FW-*` error has a characteristic in-the-wild signature. Knowing what to look for cuts triage time substantially.

| Error code | Typical signature in `run_error` payload | Most common cause | First triage step |
|------------|------------------------------------------|-------------------|-------------------|
| `FW-IO` | `"failed to open audio.mp3: No such file or directory"` | Path typo, deleted file, permission denied | `ls -la <path>` |
| `FW-JSON` | `"invalid JSON at line N column M"` | Backend produced malformed output (rare) or a corrupted JSONL snapshot during sync import | `cat <file> \| jq .` for the offending file |
| `FW-CMD-MISSING` | `"whisper-cli not found on PATH"` | Backend binary not installed or shadowed | `which whisper-cli`; set `FRANKEN_WHISPER_WHISPER_CPP_BIN` |
| `FW-CMD-FAILED` | `"command failed with exit status N: <stderr tail>"` | Backend crashed, bad model file, out-of-memory, hostile audio | Re-run the backend command directly with the rendered args |
| `FW-CMD-TIMEOUT` | `"command exceeded timeout of N ms"` | Audio too long for budget, slow hardware, network model download mid-run | Raise `--timeout` or `FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS` |
| `FW-BACKEND-UNAVAILABLE` | `"no backend available for request (diarize=true, hf_token=missing)"` | No backend supports requested feature set | Install a diarization-capable backend or drop `--diarize` |
| `FW-INVALID-REQUEST` | `"--input, --stdin, --mic are mutually exclusive"` | Contradictory CLI flags, malformed JSON request via library | Drop one of the conflicting flags |
| `FW-STORAGE` | `"database is busy"` or `"PRAGMA integrity_check: out of order"` | Concurrent writer, corrupted DB, disk full | `franken_whisper robot health`; consider JSONL rebuild |
| `FW-UNSUPPORTED` | `"backend X does not support diarization"` | Requested capability the chosen backend lacks | Switch backends or remove the capability flag |
| `FW-MISSING-ARTIFACT` | `"backend completed but did not produce expected output file"` | Backend version skew, model didn't load | Inspect the work dir under `.franken_whisper/work/` |
| `FW-CANCELLED` | `"cancelled at stage backend"` | Ctrl+C or deadline hit | Re-run with longer timeout if intended |
| `FW-STAGE-TIMEOUT` | `"stage `normalize` exceeded budget of 180000 ms"` | Pathological input, contention | Raise the relevant `FRANKEN_WHISPER_STAGE_BUDGET_*_MS` |

In robot mode each of these arrives as a `run_error` event with `code` set to one of the six `FW-ROBOT-*` codes (see the error-code mapping table earlier). The internal `FW-*` variant is preserved in the `message` field for further triage.

---

## Performance Tuning Guide

The default budgets and tunables are calibrated for a typical local workload (a few minutes of audio, a single backend, no heavy contention). Three tuning vectors matter most:

### 1. Stage budgets

Watch the `orchestration.latency_profile` event. Stages tagged `decrease_budget_candidate` (≤ 30% utilization) are over-budgeted; stages tagged `increase_budget` (≥ 90% utilization) are at risk of `FW-STAGE-TIMEOUT`. Adjust via `FRANKEN_WHISPER_STAGE_BUDGET_<NAME>_MS`. Common adjustments:

| Situation | Tuning |
|-----------|--------|
| Long audio on a slow CPU | raise `BACKEND_MS` to 1800000 (30 min) |
| Microphone capture of 10+ minutes | raise `INGEST_MS` to 60000 |
| Cold-start with large models | raise `BACKEND_MS` to absorb model load time |
| Diarization over multi-hour calls | raise `DIARIZE_MS` to 120000 |
| Container with strict CPU limits | raise everything except `CLEANUP_MS` 2–3× |

### 2. Backend selection

Force-pick when you know better than the router:

```bash
# Always use whisper.cpp on Apple Silicon (Metal is fast, batching doesn't help)
franken_whisper transcribe --input a.mp3 --backend whisper_cpp --model large-v3

# Always use insanely-fast-whisper on a CUDA box doing batch jobs
franken_whisper transcribe --input a.mp3 --backend insanely_fast --batch-size 24 --flash-attention
```

If you want adaptive behavior but want to seed the posteriors, run a small calibration corpus through `--backend auto` once. The router will absorb the empirical success rates into its Beta distributions and route accordingly going forward.

### 3. Audio preprocessing

If you control the audio pipeline upstream, pre-normalize to 16 kHz mono WAV. The Normalize stage detects this and becomes a passthrough, saving ~260 ms per run (significant when transcribing thousands of files). `ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le` is the canonical incantation.

### Speculative streaming

`--speculative` is a latency tool, not an accuracy tool. Use it when you need partial transcripts for a live UI; don't use it for batch processing (the dual-model overhead reduces throughput vs. running the quality model alone). Tune `--speculative-window-ms` based on your UI's update tolerance: smaller windows = lower partial latency but more corrections; larger windows = higher latency but fewer retractions.

### When to disable persistence

`--no-persist` skips the Persist stage entirely (~10 ms saved per run). Use it when:

- The output stream is consumed downstream and the DB record would never be queried.
- You're benchmarking inference time.
- You're running on a filesystem where SQLite WAL files would land in an inconvenient place.

---

## Disk Usage Growth Model

The SQLite database grows roughly linearly with transcribed audio. Rough estimates (your mileage varies with how chatty the audio is and how many events your pipeline emits):

| Workload | DB growth rate |
|----------|----------------|
| Default pipeline, mostly-default flags, talky English audio | ~1–3 KB per transcript-minute |
| With diarization, multi-speaker | +20–40% (extra segments + speaker labels) |
| With `--word-threshold` and word-level timestamps | +50–100% (word-arrays in `result_json`) |
| With speculative streaming | +30–60% (extra `transcript.*` events) |

Per 1,000 hours of transcribed audio, expect the database to grow to roughly 100–250 MB in the default configuration, 200–500 MB with diarization and speculation. WAL files add transient overhead; run `PRAGMA wal_checkpoint(TRUNCATE)` periodically (or let SQLite checkpoint naturally between sessions) to keep them bounded.

JSONL snapshots are larger than the SQLite database (no compression by default, redundant text), roughly 2–3× the DB size. Pipe through `gzip` for ~5–10× compression ratio on JSONL (which is highly repetitive). The import path transparently handles `.gz` so gzipped snapshots are first-class.

---

## Extension Guide: Adding a New Backend

To add a new backend (say, a Vosk or wav2vec2 bridge):

1. **Add a `BackendKind` variant** in `src/model.rs` and update `BackendKind::as_str()` / `parse_backend()` / serde derives.
2. **Create `src/backend/<new>.rs`** implementing the `Engine` trait:
   ```rust
   pub trait Engine: Send + Sync {
       fn name(&self) -> &'static str;
       fn kind(&self) -> BackendKind;
       fn capabilities(&self) -> EngineCapabilities;
       fn is_available(&self) -> bool;
       fn run(
           &self,
           request: &TranscribeRequest,
           normalized_wav: &Path,
           work_dir: &Path,
           timeout: Duration,
       ) -> FwResult<TranscriptionResult>;
   }
   ```
3. **Wire into `BackendRouter`** in `src/backend/mod.rs`:
   - Add to the engine enumeration.
   - Add a prior `(α, β)` to `prior_for()`.
   - Add a `quality_proxy(kind, request)` entry for both diarize and non-diarize cases.
   - Add a `latency_proxy` base cost.
   - Extend the loss matrix dimensions if you're adding a fourth action; otherwise reuse an existing action slot.
4. **Add an availability probe**: set the appropriate `FRANKEN_WHISPER_*_BIN` env var convention and update `command_exists()` lookup.
5. **Add capabilities**: fill in `EngineCapabilities { supports_diarization, supports_translation, supports_word_timestamps, supports_gpu, supports_streaming }`.
6. **Update `robot health` and `robot backends`**: the `BackendsReport` builder enumerates known engines.
7. **Add an `EngineCapabilities` row to the documentation table** in this README.
8. **Write tests** in `tests/backend_mock_tests.rs` and add a golden fixture under `tests/fixtures/golden/<new>_native_output.json`.
9. **Add conformance fixtures** under `tests/fixtures/conformance/corpus/` for cross-engine comparison.
10. **Decide on rollout stage**: if you're shipping a native pilot, start at `Shadow`. The conformance harness will gate promotion through `Validated → Fallback → Primary → Sole`.

The Bayesian router will start with the prior you set, then learn empirical reliability across runs. You don't need to do anything special to teach it.

---

## Extension Guide: Adding a New Pipeline Stage

To add a new stage (say, post-transcription summarization):

1. **Add a `PipelineStage` variant** in `src/orchestrator.rs` and update `as_str()` / `Display`.
2. **Add a budget knob**: a field on `StageBudgetPolicy`, a `STAGE_BUDGET_*_ENV` constant, and a default value.
3. **Decide whether the stage is in `DEFAULT_STAGES`** or opt-in via `PipelineBuilder`.
4. **Implement the stage** as a function that takes `(&mut PipelineCx, &mut RunReport, &TranscribeRequest, &CancellationToken)` and returns `FwResult<()>`. Use the established pattern:
   ```rust
   fn execute_stage_<name>(...) -> FwResult<()> {
       emit_event(cx, "<stage>.start", message, payload);
       cx.token.checkpoint()?;
       // ... do the work ...
       cx.token.checkpoint()?;
       emit_event(cx, "<stage>.ok", message, payload);
       Ok(())
   }
   ```
5. **Add ordering constraints** to `PipelineConfig::validate()` if your stage has dependencies (e.g., must run after `Backend`).
6. **Add `*.skip` handling**: emit the skip event with a structured reason payload when the stage is configured but isn't needed for this run.
7. **Add `*.cancelled` and `*.error` paths**: both must emit before propagating the error so the NDJSON contract is preserved.
8. **Test**: unit-test the stage in isolation, then add an integration test exercising the full pipeline with the stage enabled.
9. **Update the documentation tables in this README**: Pipeline Stages diagram, Stage Budgets table, Stage Codes list, Dynamic Stage Composition table.

The orchestrator handles cancellation, finalizers, evidence accumulation, and latency profiling for you; your stage only needs to do its job and emit events.

---

## The Cancellation Token Contract

Every stage in the pipeline interacts with `CancellationToken`. Authors writing new stages or backends should follow these invariants:

1. **`token.checkpoint()` is cheap**: it's just two atomic reads (Ctrl+C flag, deadline check). Call it liberally between unit chunks of work.
2. **Subprocess invocations must use `run_command_cancellable()`**: the basic `run_command()` and `run_command_with_timeout()` don't honor the token. Long-running backends should use the cancellable variant.
3. **Database writes must checkpoint before `COMMIT`**: never write to SQLite without one final `token.checkpoint()?` immediately before the `RELEASE SAVEPOINT`. The savepoint mechanism guarantees rollback if the checkpoint fires.
4. **Finalizers must respect the cleanup budget**: register them via `cx.finalizers.push(...)` and they'll run in LIFO order under a 5-second total budget. Don't do unbounded work in a finalizer (e.g., don't recurse a huge directory tree).
5. **Drop checkpoint at stage boundaries**: clear the `stage_start` timer when transitioning so elapsed time from a cancelled stage doesn't leak into the next stage's latency profile.
6. **Emit `*.cancelled` before returning**: even on cancellation, the NDJSON contract requires a final event for the stage before pipeline termination.

Following these invariants means Ctrl+C always exits cleanly: subprocesses are killed, transactions roll back, temp directories are removed, finalizers run within budget, exit code 130. There is no path where cancellation produces orphaned state.

---

## Debugging Recipes

| Scenario | Recipe |
|----------|--------|
| **"Why did the router pick that backend?"** | `franken_whisper robot routing-history --limit 1 --run-id <RUN_ID> \| jq` — inspect the posterior snapshot, calibration score, and Brier score at decision time |
| **"What did the backend actually output?"** | `RUST_LOG=franken_whisper::backend=trace franken_whisper transcribe ...` and inspect the rendered subprocess command (with secrets redacted) |
| **"Where did the time go?"** | Look at the `orchestration.latency_profile` event; `service_ms` per stage and per-stage tuning recommendation |
| **"Why is SQLite slow?"** | `franken_whisper robot health` → `database.wal_checkpoint.log_frames` and `database.freelist_count`; if either is climbing, run `PRAGMA wal_checkpoint(TRUNCATE); VACUUM;` |
| **"Why did the speculative model retract?"** | Look at the `transcript.retract` event's `reason` and the `transcript.correct` event's `drift` payload; tune `--correction-tolerance-wer` if false retractions are too aggressive |
| **"Why is TTY audio glitchy?"** | Run `tty-audio retransmit-plan` on the captured stream; inspect the `DecodeReport`'s `gaps`, `duplicates`, `integrity_failures` counts |
| **"Why isn't auto-provisioning working?"** | `RUST_LOG=franken_whisper::audio=debug` and watch for the `ffmpeg auto-provision` log lines; check `FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG` is not `0`; verify network access to `johnvansickle.com` |
| **"Why does my native pilot disagree?"** | `franken_whisper robot routing-history` → check `native_rollout_stage` per decision; run the conformance harness directly against the disagreeing fixture to see the violation report |

The general pattern: every adaptive behavior in `franken_whisper` is queryable. There is no "secret state". Every routing decision, every speculation correction, every retransmit round, every stage budget decision is recorded in events or evidence ledgers you can inspect post-hoc.

---

## Multi-Run Routing Walk-through

Three sequential `--backend auto` runs against the same machine, showing how the Bayesian router learns:

**Run 1: cold start.** Empty calibration window, only priors:

```
posterior[whisper_cpp]:         Beta(7, 3) → mean 0.700
posterior[insanely_fast]:       Beta(6, 4) → mean 0.600
posterior[whisper_diarization]: Beta(5, 5) → mean 0.500
calibration: 0 observations → fall back to static priority
chosen: whisper_cpp (top of static priority for non-diarize)
outcome: success
```

**Run 8: 7 successes so far.** Calibration window fills slowly:

```
posterior[whisper_cpp]: Beta(7 + 0.86×7, 3 + 0.14×7) = Beta(13.0, 4.0) → mean 0.765
calibration: 7 observations → still using static priority (need ≥ 5, but also need calibration_score ≥ 0.3 and Brier ≤ 0.35)
chosen: whisper_cpp (still static)
outcome: success
```

**Run 23: enough data, adaptive routing engages.**

```
posterior[whisper_cpp]:         Beta(25.6, 4.4)  → mean 0.853
posterior[insanely_fast]:       Beta(15.72, 6.28) → mean 0.715 (12 obs)
posterior[whisper_diarization]: Beta(5, 5)        → mean 0.500 (still no data; was never picked)
calibration: 23 observations, Brier = 0.18, score = 0.82 → adaptive mode
loss matrix evaluation favors whisper_cpp by ~2 points
chosen: whisper_cpp (now via adaptive selection, not static fallback)
emit: routing_decision { calibration_score: 0.82, brier_score: 0.18, fallback_active: false }
outcome: success
```

**Run 48: a transient whisper_cpp failure shifts the posterior.**

```
posterior[whisper_cpp]: Beta(33.4, 11.6) → mean 0.742  (5 failures absorbed)
posterior[insanely_fast]: Beta(22.9, 9.1) → mean 0.716
calibration: 48 observations, Brier = 0.21, score = 0.79 → adaptive mode
loss matrix evaluation: whisper_cpp's failure cost now > insanely_fast's
chosen: insanely_fast (the router learned to shift)
outcome: success → posterior[insanely_fast] = Beta(23.7, 9.3) → mean 0.718
```

The router is doing nothing exotic: it blends priors with empirical data via Beta-conjugate updates, weights the result against the multi-factor loss matrix, and falls back when calibration degrades. The audit trail (`routing_decision` events plus the evidence ledger) makes every shift inspectable.

---

## TinyDiarize vs Full Diarization

`franken_whisper` exposes two paths to speaker labels:

| Aspect | `--tiny-diarize` (whisper.cpp built-in) | `--diarize` (whisper-diarization / pyannote) |
|--------|-----------------------------------------|----------------------------------------------|
| Dependencies | whisper-cli only | Python + pyannote.audio + HuggingFace token + Demucs (for `--no-stem` off) |
| Output | Speaker-turn tokens injected inline in the transcript | Explicit per-segment `speaker` field with clusterable IDs |
| Accuracy on 2 speakers | Decent; misses overlap | High; handles overlap, gender, register |
| Accuracy on 5+ speakers | Degrades quickly | Stable up to 10+ speakers with constraints |
| Latency overhead | Minimal | +20–60% wall time (separation + clustering) |
| HF token required | No | Yes |
| Works offline (post-model-download) | Yes | Yes (after first download) |
| Cross-engine compatible | Only whisper.cpp | Any backend (the diarization stage runs separately) |

Decision rule: use `--tiny-diarize` when speakers are clearly turn-taking and you just need a hint; use `--diarize` for anything where mis-attribution matters (interviews, depositions, multi-party meetings).

---

## Source Separation Tradeoffs (`--no-stem`)

The diarization backend optionally runs Demucs vocal isolation before clustering. This is on by default; pass `--no-stem` to disable.

**Keep stemming on (default) when:**

- Background music or significant ambient noise.
- Multiple overlapping speakers.
- Recordings with reverb / echo.

**Disable with `--no-stem` when:**

- Clean studio recordings (no benefit, just wasted compute).
- Real-time pipelines where the +5–15 s/min of Demucs latency is unacceptable.
- Memory-constrained environments (Demucs is the biggest single memory user in the pipeline).

The Source Separate stage has its own budget (`FRANKEN_WHISPER_STAGE_BUDGET_SEPARATE_MS`, default 30 s); raise it for long audio or lower it to fail fast on resource-constrained hosts.

---

## Pre-Flight Production Checklist

Before deploying `franken_whisper` to a production workflow, walk through:

- [ ] **Backend(s) installed.** `franken_whisper robot health` returns `overall_status=ok`. At least one backend's `available=true`.
- [ ] **HuggingFace token configured** (if using `--diarize`). Set via `FRANKEN_WHISPER_HF_TOKEN`. Verify redaction works by triggering a diarization run with `RUST_LOG=debug` and confirming the token does not appear in logs.
- [ ] **ffmpeg accessible** (if you'll transcribe video or use mic capture). Either system-installed, manually placed, or auto-provisioned. Verify with `franken_whisper robot health` → `ffmpeg.available=true`.
- [ ] **State directory writable.** `FRANKEN_WHISPER_STATE_DIR` (or default `.franken_whisper/`) is on a partition with enough free space for your expected DB growth.
- [ ] **Stage budgets sized.** If you'll transcribe audio longer than 15 minutes, raise `FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS`. If you'll diarize multi-hour content, raise `DIARIZE_MS`.
- [ ] **Backup strategy.** Schedule periodic `sync export-jsonl` + off-host copy. Test the restore path at least once.
- [ ] **Concurrent agent expectations documented.** If multiple processes will write to the same DB, all must use `RunStore::begin_concurrent_session()` (or accept SQLITE_BUSY retries on conflict).
- [ ] **Native rollout stage chosen.** Default `primary` is fine for most deployments. Set `shadow` or `validated` if you want bridge-only behavior with native conformance running silently.
- [ ] **Error handling integrated.** Your downstream consumer differentiates `run_complete` from `run_error` events in robot mode, and handles each `FW-ROBOT-*` code appropriately.
- [ ] **Disk monitoring in place.** Alert on `disk_free_bytes / disk_total_bytes < 0.10` and on `wal_checkpoint.log_frames` climbing across consecutive `robot health` probes.

---

## Robot Schema Dump (`robot schema`)

`franken_whisper robot schema` emits a single JSON document describing every event type, every required field, every optional field, every payload sub-schema, and the canonical timestamp tolerance. Use it to drive client codegen, JSON-Schema validators, or auto-generated documentation:

```bash
franken_whisper robot schema | jq '.events | keys'
# [
#   "backends.discovery",
#   "health.report",
#   "routing_decision",
#   "run_complete",
#   "run_error",
#   "run_start",
#   "speculation.stats",
#   "stage",
#   "transcript.confirm",
#   "transcript.correct",
#   "transcript.partial",
#   "transcript.retract"
# ]

franken_whisper robot schema | jq '.events["run_complete"].required'
# ["event", "schema_version", "run_id", "trace_id", "started_at",
#  "finished_at", "backend", "language", "transcript", "segments",
#  "acceleration", "warnings", "evidence"]

franken_whisper robot schema | jq '.events["routing_decision"].payload'
# describes the loss_matrix, posterior snapshot, calibration metrics,
# fallback_active flag, evidence ledger entry, etc.
```

The schema document is part of the `1.0.0` contract: adding new event types requires a minor-version bump, and renaming or removing required fields requires a major-version bump. Schema version is reported on every event so agents can detect mismatch instantly.

---

## Recipe Cookbook

Idiomatic patterns for common one-off tasks.

### Generate SRT subtitles for a folder of videos

```bash
for f in videos/*.mp4; do
  franken_whisper transcribe --input "$f" --backend whisper_cpp --model large-v3 \
    --output-srt --json > /dev/null
done
# Each video gets a sibling .srt file alongside it.
```

### Build a searchable transcript archive

```bash
franken_whisper sync export-jsonl --output ./archive
jq -r '.run_id + "\t" + .transcript' ./archive/runs.jsonl > transcripts.tsv
# Now feed transcripts.tsv into any full-text index.
```

### Tail the live pipeline as a continuous status line

```bash
franken_whisper robot run --input meeting.mp3 --backend auto 2>/dev/null \
  | jq -r 'select(.event == "stage") | "[2K\r[\(.stage)] \(.code)"' \
  | tr -d '\n'
```

### Filter routing decisions where the router fell back to static

```bash
franken_whisper robot routing-history --limit 100 \
  | jq '.[] | select(.fallback_active == true) | {decision_id, observed_state, brier_score}'
```

### Validate all replay packs in a directory

```bash
for d in replay_packs/*/; do
  for f in env.json manifest.json repro.lock tolerance_manifest.json; do
    [ -f "$d/$f" ] || echo "MISSING: $d/$f"
  done
done
# Cross-check: re-running the same input should produce byte-identical packs
```

### Confirm a backend version drift

```bash
franken_whisper transcribe --input fixtures/canary.wav --backend whisper_cpp --json \
  | jq '.replay.backend_version'
# Compare against the version recorded in your last known-good replay envelope.
# Any difference is a backend upgrade; trigger conformance re-validation.
```

### Bulk-set stage budgets via env-file

```bash
cat > /etc/franken_whisper.env <<EOF
FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS=1800000
FRANKEN_WHISPER_STAGE_BUDGET_DIARIZE_MS=120000
FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS=300000
EOF
env $(cat /etc/franken_whisper.env | xargs) franken_whisper transcribe --input long.mp3 --json
```

### Cap memory by routing diarization to a separate process

```bash
# Process A: do the transcription only
franken_whisper transcribe --input meeting.mp3 --backend whisper_cpp --no-stem \
  --db /tmp/runs.sqlite3 --json

# Process B: post-process for speakers in a memory-isolated context
# (run later, on different hardware, or in a smaller resource envelope)
```

---

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

---

## License

MIT License with OpenAI/Anthropic Rider. See [LICENSE](LICENSE) for the full text.

In short: standard MIT terms apply, with an additional restriction that no rights are granted to OpenAI, Anthropic, or their affiliates without express prior written permission from the author. This rider must be preserved in all copies and derivative works.
