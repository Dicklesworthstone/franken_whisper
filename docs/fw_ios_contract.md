# fw-ios C ABI Contract

`fw-ios` packages the franken_whisper native engine (the same sources the CLI
binary mounts under `src/native_engine/`, included by path — one inference
implementation, zero numerics added) as a **static library with a C ABI** for
iOS hosts, plus a SwiftUI application under `ios/Sources/`.

- **Source of truth for behavior:** `fw-ios/src/lib.rs`.
- **C surface:** `fw-ios/include/fw_ios.h` (hand-maintained).
- **Drift guard:** `tests/fw_ios_header_parity.rs` fails the parent suite if a
  symbol exists on one side and not the other.
- **Engine hooks:** `cfg(target_os = "ios")` modules in
  `src/native_engine/plat.rs` forward span / partial-segment events to the
  Swift host through the callback entry points below.

This document restates the ABI contract for reviewers; when it and the header
disagree, fix the docs — the header plus `lib.rs` are the contract.

## Return codes (`int32_t`)

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | generic failure |
| 2 | usage error (NULL/malformed argument, bad options) |
| 3 | model file I/O |
| 4 | input decode / invalid audio or request |
| 5 | unsupported on target (e.g. `fw_stage_audio_file` off iOS) |
| 6 | cancelled via `fw_request_cancel` |

Pointer-returning entry points return `NULL` on failure; every failure leaves
a human-readable reason available from `fw_last_error_message()` (thread-local,
valid until the next library call on that thread).

## Exported symbols

18 functions (enforced set-equality against `lib.rs` by the parity test):

| Symbol | Signature sketch |
|--------|------------------|
| `fw_version` | `const char * ()` — crate version, static lifetime |
| `fw_last_error_message` | `const char * ()` — thread-local last error |
| `fw_engine_open` | `FwEngine * (const char *model_path)` — parse/load ggml f16/q8_0; multi-second, call off-main; honors `FW_STREAM_LOAD` setenv before first call |
| `fw_engine_close` | `void (FwEngine *)` — releases model, diarizer, denoiser, staged PCM; NULL no-op |
| `fw_engine_info_json` | `const char * (const FwEngine *)` — engine-owned JSON; valid until close; `"{}"` for NULL |
| `fw_engine_load_sortformer` | `int32_t (FwEngine *, receipt_path, package_path)` — same trust-root verification chain as the CLI; ~0.5 GB resident |
| `fw_engine_load_denoiser` | `int32_t (FwEngine *, artifact_path)` — pinned FastEnhancer-S artifact |
| `fw_engine_has_sortformer` / `fw_engine_has_denoiser` | `int32_t (const FwEngine *)` — 1 loaded / 0 not |
| `fw_stage_audio_file` | `int32_t (FwEngine *, bytes, len, ext, denoise, char **out_json)` — mp3/m4a/aac/wav → staged 16 kHz mono; iOS-only (else code 5) |
| `fw_stage_pcm` | `int32_t (FwEngine *, const float *pcm, len, denoise, char **out_json)` — 16 kHz mono f32 in [-1, 1] |
| `fw_run_prepared` | `int32_t (FwEngine *, options_json, char **out_json)` — full transcription/diarization run |
| `fw_live_decode_pcm` | `int32_t (FwEngine *, pcm, len, options_json, char **out_json)` — host-pushed live slice decoded with dynamic context plus the shared AlignAtt commit/preview policy; endpoint calls restore full context |
| `fw_set_progress_callback` | `void (FwProgressFn, void *ctx)` — heartbeat spans |
| `fw_set_segments_callback` | `void (FwSegmentsFn, void *ctx)` — live per-window transcript arrays |
| `fw_request_cancel` / `fw_reset_cancel` | `void ()` — process-wide sticky cooperative cancel (code 6 at next checkpoint); reset before next run |
| `fw_string_free` | `void (char *)` — release every `char **` out-parameter exactly once; NULL no-op |

## Ownership

- `char **out_json` results are **owned by the caller**: release each with
  `fw_string_free()` exactly once.
- `const char *` returns are library-owned (`fw_last_error_message`,
  `fw_version`) or engine-owned (`fw_engine_info_json` — valid until
  `fw_engine_close`).
- Callback `ctx` pointers are never dereferenced by the library; clear a
  callback (`func = NULL`) before releasing its context.

## Threading

- `FwEngine` is **not thread-safe**: serialize every handle-taking call
  (the SwiftUI app wraps the handle in an actor).
- Global entry points (callbacks, cancel, last-error) are safe from any
  thread; last error is per-thread.
- Callbacks fire on whichever thread runs the engine (decode loop / load):
  they must be thread-safe, non-blocking, and must not call back into the
  library.

## Staging → run flow

Stage once, then run:

1. `fw_engine_open(model_path)`
2. optionally `fw_engine_load_sortformer(...)` / `fw_engine_load_denoiser(...)`
3. `fw_stage_audio_file(...)` or `fw_stage_pcm(...)` →
   `{"audio_sec", "skipped_leading_sec", "denoised"}` (sizes progress before
   the long stage; leading silence is trimmed pre-model and every emitted
   timestamp gets the offset restored)
4. `fw_run_prepared(engine, options_json, &result_json)`
5. `fw_string_free(stage_json); fw_string_free(result_json); fw_engine_close(engine);`

`options_json` keys (unknown keys rejected): `language` (null = auto),
`initial_prompt`, `translate`, `diarize` (requires the Sortformer package),
`timestamps`, `word_timestamps` (DTW words → `"words"`), `beam_size`
(clamped [1, 8]; 1 = greedy), `n_threads`.

Result JSON carries `language`, `segments[]`, diarization `turns[]` /
`speaker_segments[]` / mixed & overlap index arrays, optional per-segment
`words`, `dropped_windows`, `audio_sec`, `skipped_leading_sec`, and
`diarization_error`.

**Diarization degrades, transcription does not:** requesting `diarize`
without a loaded diarizer fails fast (code 4) *before* decode; a diarize-stage
failure *after* a successful decode returns the finished speakerless
transcript with the reason in `diarization_error` rather than discarding
minutes of work. Cancellation always aborts the whole call (code 6).

**Live-transcript timebase:** segments streamed through the segments callback
are in the trimmed timebase; only the final result re-adds
`skipped_leading_sec`. Hosts displaying live times should apply the staging
response's offset themselves.

## Host-pushed Live / Keyboard flow

`fw_live_decode_pcm` is the iOS equivalent of the core live driver's step-decode
and emission-policy seam. The host retains continuous microphone ownership and
VAD because iOS keyboard extensions cannot capture audio. For every step it
passes only the currently uncommitted 16 kHz mono slice. Rust uses
`AudioCtxPolicy::Auto`, bypasses the batch transcript cache, records the real
alignment-head attention tap, and applies the same AlignAtt decision as
`fw robot listen`. The result keeps mutable `partial_tail` separate from
append-only `commit_text`; `commit_through_sec` is relative to the supplied
slice so the host can advance its exact sample cursor.

At a detected endpoint the host sets `end_of_utterance`; Rust restores full
30-second encoder context and commits the complete remaining decode. This API
does not fabricate the core quality-confirm lane: `transcript.confirm` and
`transcript.correct` require separate quality-model ownership and insertion
recovery semantics.

The ABI rejects a holdback above 300,000 ms, matching the core live buffer
ceiling; zero still retains the policy's one-frame safety floor.
