# Real-Time Streaming Architecture (`fw robot listen`)

Reference for the live streaming subsystem (epic `bd-rt-listen-epic-polh`). The
agent-facing quickstart and event contract live in
[README → Real-Time Streaming](../README.md#real-time-streaming-fw-robot-listen);
this document is the architecture, design-rationale, tunables, and
failure-mode reference. Every latency or quality number below traces to a
`docs/PERF_LEDGER.md` entry — projections from planning documents are
deliberately excluded.

---

## 1. What this is

A single-threaded live driver (`run_listen_session`, `src/listen.rs`) composing
independently tested pieces:

```text
capture source ──► resampler ──► SessionBuffer ──► StreamingVad ──► step decode ──► EmissionPolicy ──► NDJSON events
  (cpal mic /        (to 16 kHz     (bounded rolling   (20 ms causal     (native engine,    (AlignAtt /         (stdout,
   ffmpeg /           mono f32)      buffer + prompt    energy VAD)       AudioCtxPolicy     EndpointCommit /    explicitly
   stdin-pcm /                                         │                 Auto; greedy       LocalAgreement)     flushed)
   file-replay)                                        │                 + attn tap)
                                                       ▼
                                              utterance lifecycle (speech_started … utterance_end)
                                                       ▼
                                     confirm lane (background quality model)  +  SQLite persistence
```

It bypasses the 10-stage orchestrator **on purpose**: pipeline stages are
path-typed end-to-end (`PipelineIntermediate` carries `Option<PathBuf>`) with
fixed wall-clock budgets per stage — structurally wrong for an unbounded live
session. This was decided in the epic bead; do not relitigate it here.

Batch robot mode streams sequenced *stage* events, but those events are
buffered per stage and only reach stdout when each stage completes
(time-to-first-text equals full-file latency). The live driver writes NDJSON
directly, one event per line, flushed immediately.

## 2. Design decisions and their rationale

### 2.1 AlignAtt is the emission policy; LocalAgreement ships as fallback

Published SOTA for incremental Whisper-style decoding:

- **Simul-Whisper** (Interspeech 2024, arXiv 2406.10052): 1 s chunks,
  ~1.5% absolute WER cost against offline decoding.
- **SimulStreaming** (IWSLT 2025 winner, arXiv 2506.17077): ~5× faster than
  LocalAgreement-based whisper_streaming.
- **WhisperLiveKit** made AlignAtt its default policy.

We own the decoder and already capture alignment-head cross-attention for DTW
word timestamps (`src/native_engine/dtw.rs`), so the marginal cost of the
attention tap is small (`DecodeParams::record_token_attn`, greedy-only;
requests with effective beam > 1 fail with `FW-UNSUPPORTED`).

v1 applies the attention rule at SEGMENT grain: the token-prefix rule yields a
safe time boundary, and whole segments ending before that boundary commit.
Segment grain keeps append-only reconstruction trivial. Token/word-grain
commits are the recorded refinement path.

LocalAgreement-2 (`--policy local-agreement`, arXiv 2307.14743, ~3.3 s average
lag, double decode) ships as model-agnostic insurance and A/B baseline — never
the default. It needs nothing from decoder internals: two consecutive decodes
over the same un-committed slice origin are directly comparable, and the
longest common segment-text prefix commits. Any commit advances the origin,
invalidating the stored previous output (stale-offset reset).

`--policy endpoint-commit` is the zero-intelligence control arm the latency
harness keeps forever: mutable partials every step, one commit at utterance
close.

### 2.2 Dynamic encoder context is the big per-tick lever

Encoding a full padded 30 s window dominates per-tick cost (~1.8–2.6 s turbo,
~0.65 s tiny.en; PERF_LEDGER). Padding mainly suppresses hallucinations on
tail silence (NPUsper, arXiv 2607.01108) at roughly 3.8× the encode cost. The
live driver instead:

- runs each mid-utterance step decode with `AudioCtxPolicy::Auto` — encoder
  context tracks the live slice, floored at `AUTO_MIN_ENC_CTX` = 512 frames to
  avoid repetition loops from an under-conditioned encoder (probe:
  `audio_ctx_auto_probe`: Auto tracks Full's text on jfk slices at 2–4× less
  decode);
- replaces padding's hallucination protection with VAD gating plus the
  quality-confirm lane;
- keeps the utterance-CLOSE endpoint decode at `AudioCtxPolicy::Full` (batch
  grade, full 30 s padding semantics) so committed utterance text gets the
  conservative treatment.

The general `DecodeParams::audio_ctx` machinery continues to be tuned in
`bd-rt-audio-ctx-n4dj`.

### 2.3 Bounded rolling buffer + prompt carry, not seek-resume

Cross-attention K/V is invalidated whenever encoder input grows, so carrying
decoder state across ticks buys little for a large engine change
(investigation conclusion recorded in `bd-rt-buffer-a6l5`). The shipped shape:

- bounded 16 kHz mono rolling buffer, default cap 12 s (`--max-buffer-sec`),
  trimmed at COMMITTED word boundaries minus a 200 ms keep-back margin
  (mirroring whisper.cpp `--keep`) so a word is never clipped at the seam;
- linguistic context crosses trims as decoder PROMPT text, not audio:
  within-utterance committed text plus the tail (~200 chars, word-boundary
  trimmed) of the previous confirmed utterance (`--no-context` disables).
  String prompt carry matches whisper_streaming's proven approach;
- exact session clock: buffer index → absolute session time via a
  trimmed-samples counter in SAMPLES (never floats), consistent under
  arbitrary push/trim sequences; the resampler's sub-5 ms group delay is
  deliberately ignored (far below the 20 ms mel frame);
- silence retention: trims keep at least `min_tail_sec` of audio so speech
  onset after long silence still receives its full VAD pre-pad;
- allocation-stable: front trims use `Vec::drain` (memmove in place, capacity
  never shrinks) — after warmup the buffer stops allocating (asserted by test).

This is the published pattern (whisper_streaming, SimulStreaming
`--audio_max_len`, whisper.cpp stream `--length`/`--keep`). It turns O(n²)
"re-decode a growing file" into O(1) per step over a bounded window.

### 2.4 Append-only committed text

Industry streams converged on interim-results-plus-finals (Deepgram interim
results with `speech_final` endpointing; AssemblyAI partials with
`UtteranceEnd`; OpenAI Realtime incremental deltas). All allow revising what
was already shown. That is hostile to agents acting on durable state: anything
consumed must be re-writable.

The live contract splits the stream instead:

- `transcript.delta` is append-only: concatenating an utterance's deltas IS
  its committed transcript, and committed text is NEVER rewritten. Retraction
  count is zero by construction (verified in the latency campaign).
- `transcript.partial` is the explicitly mutable preview garnish — safe to
  display, unsafe to persist.
- `utterance_end` is the act-now signal (the Deepgram `speech_final` /
  AssemblyAI `UtteranceEnd` analog), carrying the full committed text.

Consumers key durable actions on deltas + `utterance_end` and treat partials
as ephemeral.

### 2.5 The confirm lane never blocks the live lane

Background worker re-transcribes each CLOSED utterance with the quality model
(default `large-v3-turbo`, loaded lazily on first use), compares via the batch
CorrectionTracker, and emits `transcript.confirm` / `transcript.correct`
keyed by `utterance_id`. These verdicts may arrive after their
`utterance_end`; ordering discipline is identical on the in-loop drain and the
terminal drain. Bound the lane: more than `--confirm-queue-bound` (default 4)
unconfirmed jobs drops the OLDEST with a `confirm_lag` warning; session end
waits at most `--confirm-drain-sec` (default 10 s) before abandoning in-flight
jobs with a `confirm_drain_timeout` warning.

Enablement matrix (resolved up front so `session_start` is truthful):
`auto` ⇒ ON iff the turbo package is installed AND the fast lane did not fall
back to turbo (self-confirmation is meaningless); `none` ⇒ off; explicit spec
⇒ honored unless it names the effective fast model.

### 2.6 Crash-durable persistence at utterance granularity

One session = ONE run row (`backend = "native-listen"`), same SQLite store as
batch runs. Each closed utterance appends its delta segments plus buffered
stream events inside one savepoint transaction and bumps `runs.finished_at`
as "last known alive". A crashed session is recognizable as a listen run
whose events lack the session-end marker. Mutable `transcript.partial`
events are deliberately NOT persisted (unlike batch, where every event is
kept) — the divergence is documented in the storage row-type docs
(`src/storage.rs`). `--no-persist` disables; `--db` selects the store.

## 3. Pipeline stages in detail

| Stage | Implementation | Notes |
|-------|----------------|-------|
| Capture | `CaptureSource` trait, `src/capture.rs` | cpal mic primary (CoreAudio/ALSA/WASAPI) with SPSC ring (`--capture-buffer-sec`, default 30 s, absorbs slow-consumer stalls); ffmpeg fallback when cpal cannot open a device; stdin-pcm (s16le/f32le, configurable rate/channels; refuses terminal stdin); fixture replay for tests |
| Resample | streaming resampler (`bd-rt-resampler-pbk9`) | windowed-sinc (rubato-class), NOT the linear interpolation the batch builtin decoder uses; any rate/channels → 16 kHz mono f32 |
| Buffer | `SessionBuffer`, `src/listen.rs` | §2.3; forced front-trims under pathological continuous speech emit `forced_trim` warnings (audio ahead of the watermark is dropped; text for it rides the last partial — degraded but honest) |
| VAD | `StreamingVad`, `src/listen.rs` | causal: running noise-floor energy pre-gate (`--vad-gate-db`, default 9 dB above floor) + state machine (`--vad-min-speech-ms` 250, `--vad-endpoint-ms` 600). A neural second tier seam exists (`VoiceClassifier`); the evaluated earshot classifier was REJECTED (passes loud harmonic music at every usable threshold — see ignored `earshot_eval_*` tests). 20 ms frames = 320 samples @16 kHz = one encoder frame; single authority for the grid shared by AlignAtt and mel framing. `--no-vad` disables gating entirely (one continuous utterance split only by `--max-utterance-sec`) |
| Step decode | native engine, greedy | every `--step-ms` (default 300): decode the un-committed slice with timestamps ON, transcript cache bypassed (rolling audio can never hit), language pinned to the fast lane's choice, previous confirmed utterance as prompt, `suppress_nst` ON (bracket-noise markers like `[MUSIC]` are chatter on live caption streams), `record_token_attn` ON for AlignAtt |
| Policy | `EmissionPolicy` trait | §2.1; quality gate holds all commits on steps with `no_speech_prob > 0.6` or `mean logprob < −1.0` (counted as `policy_holdbacks`) after alignatt hallucinated confident text on a music-only fixture |
| Endpoint | utterance lifecycle (`bd-rt-endpoint-i3k2`) | every `speech_started` matched by exactly one `utterance_end` (same id, ids from 1, never nested); empty-text closes are normal (breath/cough); force-closes at `--max-utterance-sec` (90) emit reason `max_len`; Ctrl+C closes open utterances with reason `session_end` and maps to exit code 130 |
| Confirm | `ConfirmLane`, `src/listen.rs` | §2.5 |
| Persist | `ListenPersistSink`, `src/listen.rs` + `src/storage.rs` | §2.6 |

Fast-lane model resolution (`bd-rt-model-provision-ffki`): explicit
`--language en` ⇒ pinned `tiny.en` package; unset/non-en ⇒ multilingual `tiny`
(detect-and-pin). Missing packages fall back to the turbo model with a
`fast_model_fallback` warning (confirm lane then disables itself); provision
with `fw pull tiny` / `fw pull tiny-en`.

## 4. Tunables reference (CLI flags and defaults)

| Flag | Default | Meaning |
|------|---------|---------|
| `--source` | `mic` | `mic` \| `stdin-pcm` \| `file-replay` |
| `--input` | — | WAV input for `file-replay` (any rate/channels; resampled) |
| `--realtime-pace` | off | pace replay at real time (default: as fast as possible) |
| `--capture-backend` | `auto` | `auto` \| `cpal` \| `ffmpeg` |
| `--mic-device` | system default | input device name |
| `--list-devices` | off | enumerate input devices as NDJSON and exit (metadata only; never triggers a macOS TCC prompt) |
| `--fast-model` | auto | fast-lane model override (default tiny.en/tiny per language rule) |
| `--policy` | `alignatt` | `alignatt` \| `endpoint-commit` \| `local-agreement` |
| `--alignatt-holdback-ms` | 200 | danger-zone width behind the live edge before tokens commit |
| `--step-ms` | 300 | step decode cadence |
| `--adaptive` | off | bd-rt-adaptive-contract-yw68: adapt step cadence (±50 ms within [200, 1000]) and AlignAtt holdback (±2 frames within [5, 25]) under the alien-artifact contract; Brier-gated deterministic fallback to these configured values; every decision emits a `listen.controller` event and live state rides `listen.session_stats.controllers` |
| `--max-buffer-sec` | 12 | rolling buffer cap |
| `--language` | detect-and-pin | ISO 639-1 hint |
| `--max-seconds` | 0 (unbounded) | end the session after N seconds |
| `--max-utterance-sec` | 90 | force-close pathologically long open speech |
| `--no-partials` | off | suppress mutable `transcript.partial` previews (first remedy for slow consumers) |
| `--stats-interval-sec` | 30 | `listen.session_stats` heartbeat interval (0 = final only) |
| `--no-context` | off | disable cross-trim/cross-utterance prompt carry |
| `--capture-buffer-sec` | 30 | capture ring capacity absorbing consumer stalls |
| `--stdin-rate` / `--stdin-channels` / `--stdin-format` | 16000 / 1 / s16le | stdin-pcm format |
| `--no-vad` | off | disable VAD gating (harness baselines, continuous feeds) |
| `--vad-gate-db` | 9 | energy gate above running noise floor |
| `--vad-min-speech-ms` | 250 | sustained voice before an utterance opens |
| `--vad-endpoint-ms` | 600 | sustained silence that closes an utterance |
| `--quality-model` | `auto` | confirm-lane model: `auto` \| `none` \| explicit spec |
| `--confirm-drain-sec` | 10 | session-end wait for in-flight confirms |
| `--confirm-queue-bound` | 4 | max unconfirmed utterances before oldest-drop |
| `--db` | `.franken_whisper/storage.sqlite3` | run-history store |
| `--no-persist` | off | disable SQLite persistence |

Library note: `ListenConfig`'s library default has persistence OFF; the CLI
turns it on unless `--no-persist`.

## 5. Failure modes and structured diagnostics

Every degradation is a machine-readable `listen.warning {reason, detail}` —
never stderr prose. Taxonomy:

| Warning | Cause | Consumer action |
|---------|-------|-----------------|
| `silent_input` | first seconds pure digital zeros — macOS TCC denial signature (denial delivers zeros, not errors) | remediation text names System Settings → Privacy & Security → Microphone; pre-authorize before headless/SSH use, or pipe `--source stdin-pcm` |
| `capture_overrun` | consumer slower than capture; ring dropped frames | raise `--capture-buffer-sec`; check downstream backpressure; consider `--no-partials` |
| `fallback_capture_backend` | cpal could not open device; ffmpeg took over | expect higher latency; fix host audio for cpal if possible |
| `fast_model_fallback` | tiny package missing; turbo loaded instead | `fw pull tiny` / `fw pull tiny-en`; confirm lane disabled itself |
| `quality_model_unavailable` | turbo package missing for confirm lane | `fw pull whisper` (or accept fast-lane-only output) |
| `confirm_lag` | queue exceeded bound; oldest unconfirmed utterance dropped | raise `--confirm-queue-bound`; faster host; smaller `--max-utterance-sec` |
| `forced_trim` | pathological continuous speech outran policy commits; pre-watermark audio dropped | longer `--max-buffer-sec`; different policy; treat affected span via last partial |
| `decode_behind` | step decodes lagging real time | reduce encode cost (quiet host, fewer threads competing); larger `--step-ms` |
| `persist_degraded` | storage sink fell behind or failed a flush retry | check disk/space; `--no-persist` if history is optional |
| `endpoint_flush_timeout` | endpoint decode overran its budget | see detail payload; usually transient load |

Device discovery is metadata-only (`enumerate_input_devices`): opening a
stream is what triggers the macOS TCC prompt, so health probes never do it.
When a requested device name does not exist, the error lists what DOES exist.

## 6. Latency evidence (what we can honestly claim)

First campaign (`examples/listen_latency_ab.rs`, release build, real binary
over `--source stdin-pcm`, paced 20 ms PCM injection; PERF_LEDGER
2026-08-23, `bd-rt-latency-harness-3dkh`, artifact
`docs/perf_artifacts/listen_latency_campaign1_2026-08-23.json`,
host load ≈28–43):

- TTFT medians 3.1–5.6 s across jfk-derived fixtures (dominated by model load
  + first decode under contention).
- Commit-lag p50 ≈ 0.79–1.21 s behind the committed audio.
- Partial cadence p50 489 ms at `--step-ms 300` (cadence = step + decode time).
- Zero retractions by construction; joined delta text identical across policy
  arms; tone-only negative fixture produced zero output (the quality-gate fix).

That campaign's A/A nulls missed the pre-declared band, so it banks NO
cross-policy comparison — quiet-host reruns can. The campaign's durable yield
was two shipped bug fixes (AlignAtt committed-index soundness across
re-decodes; non-speech holdback gate) plus the attention-tap purity probe
(`examples/attn_tap_purity_probe.rs`: tap on/off transcripts identical).

Deterministic correctness testing of the live path (file-replay e2e, golden
NDJSON streams, lifecycle invariants) is wave 6 bead `bd-rt-e2e-0zo5`.

## 7. Adaptive controllers: shipped, default-off (`--adaptive`)

The deterministic policies above are the default product. The two sanctioned
adaptive controllers (bd-rt-adaptive-contract-yw68) now EXIST behind
`--adaptive`, copying the `SpeculationWindowController` shape: state space
(overrun posterior / correction posterior), action space (step ±50 ms within
[200, 1000]; holdback ±2 frames within [5, 25]), loss asymmetry (missed ticks;
corrections cost ~10x staleness), Beta posteriors with Brier-scored rolling
calibration, deterministic fallback to the configured values when Brier > 0.25
with >= 10 samples, bounded evidence ledger, `listen.controller` events, and a
controller snapshot in every `listen.session_stats`. Default OFF: flipping the
default requires harness evidence per ledger discipline.

## 8. Scope boundaries

- Live sessions transcribe only — no speaker diarization in the live driver;
  attribution stays a batch/post-hoc feature.
- Fast lane decodes greedy-only; AlignAtt requires the attention tap, which
  fails requests with beam > 1 (`FW-UNSUPPORTED`).
- `file-replay` accepts WAV (v1); other containers go through batch
  transcription or stdin-pcm piping.
- Four-speaker Sortformer capacity limits and diarization certification
  caveats (README) apply to batch paths, unaffected by this driver.
