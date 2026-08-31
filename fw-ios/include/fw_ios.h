/* fw_ios.h — the C ABI of fw-ios, the franken_whisper engine as a static
 * library for iOS. This header is the contract the Swift side reads; keep it
 * in lockstep with fw-ios/src/lib.rs (the stated source of truth for
 * behavior).
 *
 * ── Contract ──────────────────────────────────────────────────────────────
 *
 * Return codes (int32_t): 0 = success. Non-zero values are stable:
 *   1 generic failure          2 usage (NULL/malformed argument, bad options)
 *   3 model file I/O           4 input decode / invalid audio or request
 *   5 unsupported on target    6 cancelled via fw_request_cancel
 * Pointer-returning entry points return NULL on failure. In every failure
 * case a human-readable reason is available from fw_last_error_message().
 *
 * Ownership:
 *   - `char **out_json` results are OWNED BY THE CALLER; release each with
 *     fw_string_free() exactly once.
 *   - `const char *` returns are owned by the library (fw_last_error_message,
 *     fw_version) or by the engine handle (fw_engine_info_json — valid only
 *     until fw_engine_close).
 *
 * Threading: FwEngine is NOT thread-safe. Serialize every call that takes an
 * engine handle (the app wraps the handle in a Swift actor). The global
 * entry points (callbacks, cancel, error message) are safe from any thread;
 * fw_last_error_message reports the calling thread's last error.
 *
 * Callbacks are invoked from whichever thread runs the engine (decode loop /
 * load). They must be thread-safe and non-blocking. A callback may clear or
 * replace itself, but must not invoke engine work recursively. Clear a callback
 * (func = NULL) before releasing its ctx; clearing from another thread waits
 * for an in-flight callback to return.
 *
 * Cancellation: fw_request_cancel() sets a process-wide flag checked at the
 * engine's cooperative checkpoints; the in-flight call returns code 6. Call
 * fw_reset_cancel() before the next run.
 *
 * Typical session:
 *   engine = fw_engine_open("/…/ggml-large-v3-turbo-q8_0.bin");
 *   fw_engine_load_sortformer(engine, receipt_path, package_path); // optional
 *   fw_engine_load_denoiser(engine, denoiser_path);                // optional
 *   fw_stage_pcm(engine, pcm, n, true, &stage_json);   // or fw_stage_audio_file
 *   fw_run_prepared(engine, "{\"diarize\":true}", &result_json);
 *   fw_string_free(stage_json); fw_string_free(result_json);
 *   fw_engine_close(engine);
 */

#ifndef FW_IOS_H
#define FW_IOS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle: whisper model + optional Sortformer diarizer +
 * optional FastEnhancer denoiser + the staged PCM between fw_stage_* and
 * fw_run_prepared. */
typedef struct FwEngine FwEngine;

/* ── Diagnostics ────────────────────────────────────────────────────────── */

/* The calling thread's last error message. Never NULL; empty string when no
 * error has been recorded. Valid until the next library call on this
 * thread. */
const char *fw_last_error_message(void);

/* The crate version, e.g. "0.1.0". Static lifetime. */
const char *fw_version(void);

/* ── Engine lifecycle ───────────────────────────────────────────────────── */

/* Parse and load a whisper ggml model (.bin; f16 or q8_0) from an absolute
 * path. Multi-second on a phone; call off the main thread. Set FW_STREAM_LOAD
 * (via setenv, before the FIRST engine call of the process) to pread tensors
 * instead of holding the file as one resident blob during load.
 * Returns NULL on failure. */
FwEngine *fw_engine_open(const char *model_path);

/* Release an engine and everything it holds (model, diarizer, denoiser,
 * staged PCM). NULL is a no-op; closing twice is undefined. */
void fw_engine_close(FwEngine *engine);

/* Model info JSON: {crate_version, n_vocab, n_audio_layer, n_text_layer,
 * n_mels, multilingual, default_threads}. Owned by the engine; valid until
 * fw_engine_close. Returns "{}" for NULL. */
const char *fw_engine_info_json(const FwEngine *engine);

/* Authenticate and load the Sortformer diarization package from absolute
 * paths (conversion-receipt JSON + weights.safetensors) — the same trust-root
 * verification chain the CLI runs. ~0.5 GB resident once loaded. */
int32_t fw_engine_load_sortformer(FwEngine *engine,
                                  const char *receipt_path,
                                  const char *package_path);

/* Hydrate the FastEnhancer-S denoiser from its pinned artifact (838 KB). */
int32_t fw_engine_load_denoiser(FwEngine *engine, const char *artifact_path);

/* 1 when the respective optional component is loaded, else 0. */
int32_t fw_engine_has_sortformer(const FwEngine *engine);
int32_t fw_engine_has_denoiser(const FwEngine *engine);

/* ── Staging (stage once, then run) ─────────────────────────────────────────
 *
 * Both stagers return {"audio_sec": …, "skipped_leading_sec": …,
 * "denoised": bool} so the app can size its progress model (whisper decodes
 * 30-second windows sequentially: ceil(audio_sec / 30) windows) before the
 * long stage starts. Denoise runs only when requested AND a denoiser is
 * loaded. Leading silence is trimmed before the model hears anything; every
 * emitted timestamp gets the offset added back. Staged PCM is consumed by
 * the next fw_run_prepared. */

/* Decode an audio file held in memory (mp3/m4a/aac/wav; `ext` is a lowercase
 * extension hint, "" lets the demuxer probe) to 16 kHz mono and stage it.
 * Compiled for iOS targets; on other targets returns 5 (unsupported). */
int32_t fw_stage_audio_file(FwEngine *engine,
                            const uint8_t *bytes, size_t len,
                            const char *ext,
                            bool denoise,
                            char **out_json);

/* Stage 16 kHz mono f32 PCM in [-1, 1] (e.g. a microphone capture). */
int32_t fw_stage_pcm(FwEngine *engine,
                     const float *pcm, size_t len,
                     bool denoise,
                     char **out_json);

/* ── The run ────────────────────────────────────────────────────────────────
 *
 * options_json (NULL or "" = all defaults; unknown keys are rejected):
 *   {
 *     "language": "en" | null,        // null/"auto" = auto-detect
 *     "initial_prompt": "…" | null,
 *     "translate": false,
 *     "diarize": false,               // needs fw_engine_load_sortformer
 *     "timestamps": true,
 *     "word_timestamps": false,       // DTW word timings -> "words"
 *     "beam_size": 1,                 // clamped [1,8]; 1 = greedy
 *     "n_threads": 4                  // default: engine host policy
 *   }
 *
 * Result JSON:
 *   {
 *     "language": "en",
 *     "segments": [{start_sec, end_sec, text, speaker, confidence}, …],
 *     "turns": [{start_ms, end_ms, speaker_ref?, …}, …],          // diarize
 *     "speaker_segments": [{start_sec, end_sec, speaker, text, …}, …],
 *     "mixed_speaker_segment_indices": [...],
 *     "overlap_suspected_segment_indices": [...],
 *     "words": [[{text, start_sec, end_sec}, …], …] | null,       // per segment
 *     "dropped_windows": 0,
 *     "audio_sec": 12.3,
 *     "skipped_leading_sec": 0.0,
 *     "diarization_error": "…" | null
 *   }
 *
 * Diarization degrades, transcription does not: requesting diarize without a
 * loaded diarizer fails fast (code 4) BEFORE the decode starts, but a
 * diarize-stage failure AFTER a successful decode (e.g. audio beyond
 * Sortformer's two-hour capacity) returns the finished speakerless
 * transcript with the reason in "diarization_error" instead of discarding
 * minutes of work. Cancellation always aborts the whole call (code 6).
 *
 * Live-transcript timebase: segments streamed through the segments callback
 * are in the TRIMMED timebase; only the final result adds
 * skipped_leading_sec back. Hosts displaying live times should add the
 * staging response's skipped_leading_sec themselves.
 *
 * Long-running: minutes for long audio. Run off the main thread; watch the
 * progress callback ("encoder_window" spans count decode windows) and the
 * segments callback (one JSON array per finished window, the live
 * transcript). */
int32_t fw_run_prepared(FwEngine *engine,
                        const char *options_json,
                        char **out_json);

/* ── Progress + live transcript ─────────────────────────────────────────── */

/* Engine heartbeat: named spans with a value (milliseconds for engine perf
 * spans such as "encoder_window"; a 0..1 fraction for "audio:denoise";
 * 0.0 for bare stage markers like "whisper:weights", "sortformer:diarize",
 * "done"). Invoked from the engine's thread. */
typedef void (*FwProgressFn)(void *ctx, const char *span, double value);

/* Live transcript: one JSON array of segments per finished decode window,
 * emitted while fw_run_prepared is still running. */
typedef void (*FwSegmentsFn)(void *ctx, const char *segments_json);

/* Install/replace/clear (func = NULL) the process-wide callbacks. `ctx` is
 * never dereferenced by the library, only handed back to `func`. */
void fw_set_progress_callback(FwProgressFn func, void *ctx);
void fw_set_segments_callback(FwSegmentsFn func, void *ctx);

/* ── Cancellation ───────────────────────────────────────────────────────── */

/* Request cooperative cancellation of the in-flight engine call (it returns
 * code 6 at its next checkpoint). Process-wide and sticky: call
 * fw_reset_cancel() before the next run. Safe from any thread. */
void fw_request_cancel(void);
void fw_reset_cancel(void);

/* ── Memory ─────────────────────────────────────────────────────────────── */

/* Release a string this library returned through a `char **` out-parameter.
 * NULL is a no-op. */
void fw_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif /* FW_IOS_H */
