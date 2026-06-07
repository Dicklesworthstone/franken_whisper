//! Real native insanely-fast engine (bd-s8w8).
//!
//! This module is the in-process, pure-Rust insanely-fast-whisper engine. Like
//! [`whisper_cpp_native`](super::whisper_cpp_native) it runs **genuine** ASR
//! inference through [`crate::native_engine`] (real ggml weights, real log-mel
//! frontend, real encoder/decoder forward passes) — there are **no canned
//! phrases, no mock segmentation, and no subprocess execution**. The former
//! "pilot" that fabricated deterministic phrases from audio-energy regions is
//! gone.
//!
//! ## The insanely-fast identity: throughput via parallel *contiguous ranges*
//!
//! [`whisper_cpp_native`](super::whisper_cpp_native) calls
//! [`decode::transcribe_samples`] once over the whole clip; that function
//! windows the audio into 30 s chunks and decodes them **sequentially**,
//! carrying whisper's seek-continuation and a rolling text prompt across window
//! boundaries.
//!
//! This engine trades a *minimum* of that continuity for **throughput**: it
//! partitions the audio into a small number of **contiguous 30 s-aligned
//! ranges** (one per worker) and decodes the ranges **in parallel** across a
//! worker pool. Each worker runs the **real sequential**
//! [`decode::transcribe_samples`] over its whole contiguous span — so whisper's
//! seek-continuation, rolling text prompt, and timestamp-seek behavior are
//! preserved *within* a range. The per-range transcripts are then merged in
//! range order, offsetting each range's timestamps by its base start time.
//!
//! Compared to the old design (hard per-window round-robin, one seam *per
//! window*, no rolling prompt at all), contiguous ranges cut the discontinuities
//! from one-per-30 s-window down to **one per worker boundary** (`n_workers - 1`
//! seams total) and restore the rolling prompt everywhere except at those few
//! seams. This is exactly how production chunked-batched ASR (real
//! `insanely-fast-whisper`) trades a handful of chunk seams for parallelism.
//!
//! ### Honest tradeoff
//!
//! The `n_workers - 1` range boundaries are hard cuts: at each seam the next
//! range starts a fresh decode with no carried prompt and re-seeks from its
//! base offset, so a word straddling a seam may be clipped or duplicated. With
//! `n_workers` ranges that is `n_workers - 1` such seams for the whole clip
//! (versus `n_windows - 1` in the old per-window design — typically ~30×
//! fewer). When *zero* discontinuity matters more than throughput, use the
//! sequential [`whisper_cpp_native`](super::whisper_cpp_native) engine.
//!
//! ### Determinism & the worker-count contract
//!
//! Output is **deterministic for a fixed worker count** (greedy / temperature-0,
//! shared read-only weights, ranges merged in deterministic range order). It now
//! **varies with the worker count by design** — more workers means more (and
//! differently-placed) range seams — exactly like every chunked ASR batcher.
//!
//! Two important properties the gated tests pin down:
//!
//! - **1 worker == sequential, byte-exact.** With one worker there is exactly
//!   one range spanning the whole clip, so [`run`] degenerates to a single
//!   [`decode::transcribe_samples`] call — identical to the
//!   [`whisper_cpp_native`](super::whisper_cpp_native) path.
//! - **N workers: deterministic for fixed N, and word-bounded vs sequential.**
//!   Re-running with the same N yields byte-identical output; the word-diff
//!   versus the 1-worker (sequential) reference stays small because seams are
//!   rare and each range is a real sequential decode.
//!
//! ## Model resolution, silence pre-gate, availability, word timestamps
//!
//! These policies are identical to
//! [`whisper_cpp_native`](super::whisper_cpp_native): the model is resolved from
//! `request.model` then `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`; a cheap energy
//! pre-gate returns an empty result for pure silence without loading any
//! weights; availability is the same honest default-or-scan model-file check;
//! and word-level timestamps reuse that module's interpolated splitting helpers.

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::{FwError, FwResult};
use crate::model::{BackendKind, TranscribeRequest, TranscriptionResult, TranscriptionSegment};
use crate::native_engine::mel::N_SAMPLES_30S;
use crate::native_engine::{self, NativeWhisperModel, decode};

use super::native_audio::analyze_wav;
use super::whisper_cpp_native::{WordTimestampMode, build_segments, word_timestamp_mode};

/// Stable schema tag for the honest native raw-output metadata (shared with the
/// whisper.cpp native engine so the evidence ledger has one schema).
const SCHEMA_VERSION: &str = "native-v2";

/// Default `batch_size` (max concurrent ranges) when the request does not set
/// one — matches the historical insanely-fast default.
const DEFAULT_BATCH_SIZE: usize = 24;

/// One 30 s window in seconds, used to align contiguous ranges to whole 30 s
/// window boundaries (whisper decodes in 30 s chunks internally).
const WINDOW_SECONDS: f64 = 30.0;

/// Minimum intra-op threads a worker must be able to keep for window-range
/// parallelism to be worth enabling.
///
/// The sequential path ([`decode::transcribe_samples`]) already saturates the
/// machine with **intra-op** (rayon) parallelism on every encoder GEMM /
/// decoder GEMV. Layering `n_workers × threads_per_worker` on top of that only
/// helps when each worker still gets enough intra-op threads to run its GEMMs
/// efficiently — otherwise the workers just oversubscribe the cores and thrash
/// (the measured `×1.09–1.43` *slowdown* at `7 workers × 2 threads` on a 14-core
/// host; see `tests/artifacts/perf/.../HOTSPOTS.md`). A worker needs at least
/// this many intra-op threads or we do not spawn it.
///
/// # Why `5` (fewer workers × fuller threads)
///
/// The round-3 pass-3 measurements on a 14-core host (`fw_scale_600.wav`, both
/// models, seq vs IF) show a clean throughput/accuracy tradeoff *per worker*
/// (each extra worker adds one cold-started range seam):
///
/// | workers | tpw | wall ratio (IF/seq) | tiny LCS-diff vs seq |
/// |--------:|----:|--------------------:|---------------------:|
/// | 1       | 14  | 0.96 (byte-exact)   | 0.0 %                |
/// | 2       | 7   | 0.93                | 22 %                 |
/// | 3       | 4   | 0.87                | 41 %                 |
///
/// The marginal wall gain from the 3rd worker (0.93 → 0.87) is small, while its
/// extra seam roughly **doubles** the tiny-model word-diff (a cold range start
/// can drift / loop through its whole span). `5` makes the default `14 / 5 = 2`
/// workers on this host — the sweet spot: still **faster** than sequential
/// (~0.93×) with far less seam drift — and "fewer workers × fuller threads"
/// generally (8 cores ⇒ 1 worker = sequential; 32 cores ⇒ 6 workers). The
/// product `n_workers × threads_per_worker` never exceeds `total_threads`, so
/// there is no oversubscription. Callers that want maximum throughput can raise
/// `batch_size` to lift the worker ceiling; callers that want byte-exact output
/// use the sequential [`whisper_cpp_native`](super::whisper_cpp_native) engine
/// (or `batch_size = 1` here).
const MIN_THREADS_PER_WORKER: usize = 5;

/// The effective [`MIN_THREADS_PER_WORKER`], with a **measurement-only** override
/// via `$FRANKEN_WHISPER_IF_MIN_THREADS` (a positive integer). Production always
/// uses the compiled-in default; the env var exists solely so the perf-loop knob
/// sweep can probe alternate budgets (4 / 5 / 7) without recompiling. An unset,
/// empty, zero, or unparsable value falls back to the const, so the policy is
/// data-derived and the default is unchanged.
fn min_threads_per_worker() -> usize {
    std::env::var("FRANKEN_WHISPER_IF_MIN_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(MIN_THREADS_PER_WORKER)
}

/// Honestly report whether the native insanely-fast engine can run.
///
/// Shares [`whisper_cpp_native::is_available`](super::whisper_cpp_native::is_available)'s
/// exact policy: both engines run the **same** [`crate::native_engine`] over the
/// **same** ggml model files, so they must agree on availability. Reports `true`
/// only when a usable model header exists (the configured default resolves, or
/// any `ggml-*.bin` with a valid header sits in a search dir). Never panics or
/// performs network access.
#[must_use]
pub fn is_available() -> bool {
    super::whisper_cpp_native::is_available()
}

/// Whether a HuggingFace token is present for the given request (diarization
/// gate, unchanged from the bridge contract).
pub(crate) fn hf_token_present_for_request(request: &TranscribeRequest) -> bool {
    super::insanely_fast::hf_token_present_for_request(request)
}

/// Resolve the effective model spec for a request, or a [`FwError`] explaining
/// how to provision one. Identical precedence to the whisper.cpp native engine:
/// `request.model` then `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`, else an
/// actionable [`FwError::BackendUnavailable`] naming the env var and reusing the
/// resolver's search-dir listing.
fn effective_model_spec(request: &TranscribeRequest) -> FwResult<String> {
    if let Some(model) = request.model.clone().filter(|m| !m.is_empty()) {
        return Ok(model);
    }
    if let Some(spec) = native_engine::default_model_spec() {
        return Ok(spec);
    }
    let dirs_hint = native_engine::resolve_model("default")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    Err(FwError::BackendUnavailable(format!(
        "native insanely-fast engine has no model: pass --model, or set \
         $FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL to a model short-name or path. \
         {dirs_hint}"
    )))
}

/// The number of 30 s windows `samples` spans (`ceil(len / N_SAMPLES_30S)`),
/// used only to size the worker pool and partition the clip into contiguous
/// ranges. A clip shorter than one window counts as one window; empty input is
/// zero.
fn n_windows(samples: &[f32]) -> usize {
    samples.len().div_ceil(N_SAMPLES_30S)
}

/// The contiguous sample slice for a half-open window range `[start, end)`
/// (window indices), clamped to the available samples. `start * N_SAMPLES_30S`
/// is the range's base sample offset; `end * N_SAMPLES_30S` its end (saturating
/// at `samples.len()` so the final range keeps the remainder).
fn range_samples(samples: &[f32], range: (usize, usize)) -> &[f32] {
    let (start, end) = range;
    let lo = (start * N_SAMPLES_30S).min(samples.len());
    let hi = (end * N_SAMPLES_30S).min(samples.len());
    &samples[lo..hi.max(lo)]
}

/// Worker-pool sizing: pure, tested. Returns `(n_workers, threads_per_worker)`.
///
/// # Policy (round-3 pass-3, measurement-derived)
///
/// The old policy (`workers = min(windows, batch, total_threads/2)`, then
/// `threads = total_threads / workers`) handed out e.g. `7 workers × 2 threads`
/// on a 14-core host. Because the sequential decode already saturates the cores
/// with **intra-op** (rayon) parallelism, that *oversubscribed* the machine and
/// the parallel path ran `×1.09–1.43` **slower** than sequential (measured;
/// `tests/artifacts/perf/.../HOTSPOTS.md`). The fix is to add workers **only
/// when there is genuine headroom** — i.e. only when each worker can still hold
/// at least [`MIN_THREADS_PER_WORKER`] intra-op threads:
///
/// - `n_workers = min(n_windows, batch_size, total_threads / MIN_THREADS_PER_WORKER)`,
///   floored at 1. With `MIN_THREADS_PER_WORKER = 4` this is ≤ 3 workers on a
///   14-core box (`14/4 = 3`) and **falls back to a single worker** whenever
///   `total_threads < 2 * MIN_THREADS_PER_WORKER` (≤ 7 threads) — at which point
///   intra-op parallelism alone already fills the machine and the run
///   degenerates to the byte-exact sequential path.
/// - `threads_per_worker = max(1, total_threads / n_workers)` — the intra-op
///   budget split evenly. By construction `n_workers × threads_per_worker
///   ≤ total_threads`, so workers **never oversubscribe** the cores.
///
/// `batch_size` still caps concurrency (a request can force `1` to serialize, or
/// a larger value, but the `total_threads / MIN_THREADS_PER_WORKER` ceiling is
/// the real governor). Every branch is guarded so a zero or absurd input never
/// yields a zero worker or zero thread count (which would deadlock or
/// divide-by-zero).
pub(crate) fn plan_workers(
    n_windows: usize,
    batch_size: Option<usize>,
    total_threads: usize,
) -> (usize, usize) {
    let batch = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
    let total_threads = total_threads.max(1);
    // Headroom ceiling: only as many workers as can each keep
    // `MIN_THREADS_PER_WORKER` intra-op threads. This is the lever that prevents
    // oversubscription on top of the already-saturating intra-op parallelism.
    let headroom = (total_threads / min_threads_per_worker()).max(1);
    let n_workers = n_windows.max(1).min(batch).min(headroom).max(1);
    let threads_per_worker = (total_threads / n_workers).max(1);
    (n_workers, threads_per_worker)
}

/// Partition `n_windows` 30 s windows into `n_workers` **contiguous** ranges of
/// whole windows. Returns each range as `[window_start, window_end)` (half-open,
/// in window indices). Earlier ranges get the extra window when the split is
/// uneven (`ceil` distribution), and empty trailing ranges are dropped so the
/// returned vec length is `min(n_workers, n_windows)`.
///
/// Contiguity is the accuracy linchpin: each range is a single contiguous audio
/// span that one worker decodes with the **real sequential**
/// [`decode::transcribe_samples`], so seek-continuation and the rolling prompt
/// are preserved within the range. Seams occur only **between** ranges
/// (`returned_len - 1` of them).
pub(crate) fn plan_ranges(n_windows: usize, n_workers: usize) -> Vec<(usize, usize)> {
    if n_windows == 0 {
        return Vec::new();
    }
    let n_workers = n_workers.max(1);
    // ceil-divide windows across workers so earlier ranges absorb the remainder.
    let per = n_windows.div_ceil(n_workers);
    let mut ranges = Vec::with_capacity(n_workers);
    let mut start = 0usize;
    while start < n_windows {
        let end = (start + per).min(n_windows);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

/// Resolve the total intra-op thread budget for a request (request override,
/// else the engine default), mirroring the whisper.cpp native engine.
fn total_threads_for(request: &TranscribeRequest) -> usize {
    request
        .backend_params
        .threads
        .map_or_else(native_engine::default_threads, |t| {
            usize::try_from(t).unwrap_or_else(|_| native_engine::default_threads())
        })
}

/// Build the per-worker [`decode::DecodeParams`] for a request, with the
/// caller-chosen `threads_per_worker` (the rest of the params are window-shape
/// independent).
fn decode_params(request: &TranscribeRequest, threads_per_worker: usize) -> decode::DecodeParams {
    decode::DecodeParams {
        language: request.language.clone(),
        translate: request.translate,
        timestamps: !request.backend_params.no_timestamps,
        n_threads: threads_per_worker,
        max_text_ctx: None,
        ..decode::DecodeParams::default()
    }
}

/// One contiguous range's decode result, tagged with its starting window index
/// (the range's base time offset is `start_window * 30 s`) for the ordered merge.
struct RangeResult {
    /// Starting window index of the range; its base offset is `start_window * 30 s`.
    start_window: usize,
    output: decode::DecodeOutput,
}

/// Run real native insanely-fast (parallel contiguous-range) inference over
/// `normalized_wav` (guaranteed 16 kHz mono PCM16 by the pipeline) and return a
/// [`TranscriptionResult`].
///
/// See the module docs for the contiguous-range strategy, the worker-count
/// determinism contract, and shared model-resolution / silence-pre-gate /
/// word-timestamp policies.
///
/// # Errors
///
/// - [`FwError::BackendUnavailable`] when no model can be resolved (or when
///   diarization is requested without a HuggingFace token).
/// - [`FwError::Io`] / [`FwError::InvalidRequest`] when the WAV cannot be read.
/// - [`FwError::Cancelled`] when the cancellation token's deadline expires
///   (propagated into every worker; the first error wins).
/// - Whatever model-load or decode errors the native engine surfaces.
pub fn run(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    _work_dir: &Path,
    _timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<TranscriptionResult> {
    if request.diarize && !hf_token_present_for_request(request) {
        return Err(FwError::BackendUnavailable(
            "diarization requires HF token (`--hf-token` or env `FRANKEN_WHISPER_HF_TOKEN` / `HF_TOKEN`)"
                .to_owned(),
        ));
    }
    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Resolve the model spec up front so an unavailability error is reported
    // before any expensive work.
    let spec = effective_model_spec(request)?;

    // Silence pre-gate: cheap energy analysis avoids a multi-GB model load on a
    // pure-silence clip (shared policy with whisper.cpp native).
    let analysis = analyze_wav(normalized_wav, request.backend_params.duration_ms).ok();
    if let Some(analysis) = analysis.as_ref()
        && analysis.active_regions.is_empty()
    {
        return Ok(silence_result(request, &spec, analysis.duration_ms));
    }

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Resolve + load the model (cached, reference-counted, read-only weights).
    let model_path = native_engine::resolve_model(&spec)
        .map_err(|e| FwError::BackendUnavailable(e.to_string()))?;
    let model = NativeWhisperModel::load(&model_path)?;

    let samples = read_normalized_wav(normalized_wav)?;

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Size the worker pool from the 30 s-window count, then partition the clip
    // into that many CONTIGUOUS, 30 s-aligned ranges (one per worker).
    let win_count = n_windows(&samples);
    let batch_size = request
        .backend_params
        .batch_size
        .and_then(|v| usize::try_from(v).ok());
    let total_threads = total_threads_for(request);
    let (n_workers, threads_per_worker) = plan_workers(win_count, batch_size, total_threads);
    let ranges = plan_ranges(win_count, n_workers);

    let params = decode_params(request, threads_per_worker);

    // Decode each contiguous range with the real sequential decode (parallel
    // across workers); merge in range order, offsetting timestamps by each
    // range's base start time.
    let range_results = decode_ranges_parallel(&model, &samples, &ranges, &params, token)?;

    let merged = merge_ranges(&range_results);
    // Seams are the hard cuts BETWEEN ranges: one fewer than the number of
    // (non-empty) ranges actually decoded.
    let seams = ranges.len().saturating_sub(1);

    let word_mode = word_timestamp_mode(request.backend_params.word_timestamps.as_ref());
    let segments = build_segments(
        &merged.segments,
        word_mode,
        request.backend_params.split_on_word,
        request.backend_params.no_timestamps,
        token,
    )?;
    let transcript = super::transcript_from_segments(&segments);
    let language = merged.language.or_else(|| request.language.clone());

    let raw_output = raw_output_json(
        &spec,
        &model_path,
        model.version_tag(),
        win_count,
        n_workers,
        threads_per_worker,
        &ranges,
        seams,
        &merged.windows,
        word_mode,
        request.backend_params.split_on_word,
        false,
    );

    Ok(TranscriptionResult {
        backend: BackendKind::InsanelyFast,
        transcript,
        language,
        segments,
        acceleration: None,
        raw_output,
        artifact_paths: Vec::new(),
    })
}

/// Decode each contiguous `range` (one per worker) on its own scoped thread,
/// running the **real sequential** [`NativeWhisperModel::transcribe`]
/// ([`decode::transcribe_samples`]) over the range's contiguous audio span on
/// the **shared** `Arc<NativeWhisperModel>` (weights are read-only). Returns the
/// per-range outputs in arbitrary completion order (the caller sorts by start
/// window).
///
/// Because each worker runs the full sequential decode over a contiguous span,
/// seek-continuation and the rolling text prompt are preserved **within** every
/// range; the only discontinuities are the `ranges.len() - 1` seams between
/// ranges. With a single range (1 worker) this is byte-identical to the
/// sequential [`whisper_cpp_native`](super::whisper_cpp_native) path.
///
/// Cancellation + first-error: a shared [`AtomicBool`] short-circuits remaining
/// ranges the moment any range errors (or the token expires); the first error
/// observed is returned. For a **fixed** range partition the merged result is
/// deterministic regardless of which worker finished first.
fn decode_ranges_parallel(
    model: &NativeWhisperModel,
    samples: &[f32],
    ranges: &[(usize, usize)],
    params: &decode::DecodeParams,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<RangeResult>> {
    let stop = AtomicBool::new(false);
    let first_error: Mutex<Option<FwError>> = Mutex::new(None);
    let results: Mutex<Vec<RangeResult>> = Mutex::new(Vec::with_capacity(ranges.len()));

    std::thread::scope(|scope| {
        for &(start_window, end_window) in ranges {
            let stop = &stop;
            let first_error = &first_error;
            let results = &results;
            let span = range_samples(samples, (start_window, end_window));
            if span.is_empty() {
                continue;
            }
            scope.spawn(move || {
                // A worker's checkpoint: honor the orchestrator token AND the
                // shared stop flag so a sibling's error/cancellation halts every
                // worker promptly.
                let checkpoint = || -> FwResult<()> {
                    if stop.load(Ordering::Relaxed) {
                        return Err(FwError::Cancelled(
                            "insanely-fast worker stopped".to_owned(),
                        ));
                    }
                    token.map_or(Ok(()), crate::orchestrator::CancellationToken::checkpoint)
                };
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                // Measurement-only (PERF_SPANS): per-worker wall + span samples so
                // worker finish-time SKEW (range imbalance: even duration split,
                // uneven speech density) is observable. No behavior change.
                let worker_t0 = std::time::Instant::now();
                match model.transcribe(span, params, &checkpoint) {
                    Ok(output) => {
                        crate::native_engine::perf_span(
                            "if_worker",
                            worker_t0.elapsed().as_secs_f64() * 1e3,
                            &format!(
                                "\"start_window\":{start_window},\"end_window\":{end_window},\"samples\":{}",
                                span.len()
                            ),
                        );
                        results
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push(RangeResult {
                                start_window,
                                output,
                            });
                    }
                    Err(err) => {
                        // First error wins; signal every worker to stop.
                        let mut slot = first_error.lock().unwrap_or_else(|e| e.into_inner());
                        if slot.is_none() {
                            *slot = Some(err);
                        }
                        stop.store(true, Ordering::Relaxed);
                    }
                }
            });
        }
    });

    if let Some(err) = first_error.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(err);
    }

    let mut results = results.into_inner().unwrap_or_else(|e| e.into_inner());
    results.sort_by_key(|r| r.start_window);
    Ok(results)
}

/// The deterministic merge of per-range outputs into a whole-clip result.
struct MergedOutput {
    segments: Vec<TranscriptionSegment>,
    windows: Vec<decode::WindowStats>,
    language: Option<String>,
}

/// Merge per-range decode outputs in range order. Each range decodes its
/// contiguous span with timestamps *relative to the range start*, so every
/// segment/window-stat is offset by `start_window * 30 s` to map it back into
/// whole-clip time. The detected language (if any) is taken from the first range
/// that reports one.
///
/// Seam semantics: ranges are simply concatenated in start-window order. Each
/// range's internal segments already carry whisper's seek-continuation; the
/// `ranges.len() - 1` boundaries between ranges are plain concatenation points
/// (no de-duplication is attempted — a word straddling a seam may appear in both
/// adjacent ranges, the documented chunked-batcher behavior).
///
/// This is the determinism linchpin: because ranges are merged strictly by their
/// start window, the merged result does not depend on which worker decoded which
/// range or in what order they finished — for a fixed range partition.
fn merge_ranges(results: &[RangeResult]) -> MergedOutput {
    let mut segments = Vec::new();
    let mut windows = Vec::new();
    let mut language = None;

    for result in results {
        let offset = result.start_window as f64 * WINDOW_SECONDS;
        if language.is_none() {
            language = result.output.language.clone();
        }
        for seg in &result.output.segments {
            segments.push(TranscriptionSegment {
                start_sec: seg.start_sec.map(|s| s + offset),
                end_sec: seg.end_sec.map(|s| s + offset),
                text: seg.text.clone(),
                speaker: seg.speaker.clone(),
                confidence: seg.confidence,
            });
        }
        for win in &result.output.windows {
            windows.push(decode::WindowStats {
                avg_logprob: win.avg_logprob,
                no_speech_prob: win.no_speech_prob,
                tokens: win.tokens,
                window_offset_sec: win.window_offset_sec + offset,
            });
        }
    }

    MergedOutput {
        segments,
        windows,
        language,
    }
}

/// Read a normalized 16 kHz mono PCM16 WAV into f32 mono samples (shares the
/// engine's RIFF reader so production and gated e2e tests use one decoder).
fn read_normalized_wav(path: &Path) -> FwResult<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    decode::read_wav_16k_mono(&bytes)
}

/// The honest raw-output metadata JSON for a real-inference parallel
/// contiguous-range run.
///
/// Schema stays `native-v2`-compatible: every legacy field is retained
/// (`parallel_windows`, `workers`, `threads_per_worker`, `windows`, …) and the
/// new range/seam fields are **additive** — `ranges` (the contiguous
/// `[start_window, end_window)` partition) and `seams` (the count of hard cuts
/// between ranges, `ranges.len() - 1`).
#[allow(clippy::too_many_arguments)]
fn raw_output_json(
    spec: &str,
    model_path: &Path,
    version_tag: String,
    parallel_windows: usize,
    workers: usize,
    threads_per_worker: usize,
    ranges: &[(usize, usize)],
    seams: usize,
    windows: &[decode::WindowStats],
    word_mode: WordTimestampMode,
    split_on_word: bool,
    silence: bool,
) -> Value {
    let word_timestamps = if word_mode != WordTimestampMode::None || split_on_word {
        "interpolated"
    } else {
        "none"
    };
    let windows_json: Vec<Value> = windows
        .iter()
        .map(|w| {
            json!({
                "window_offset_sec": w.window_offset_sec,
                "tokens": w.tokens,
                "avg_logprob": w.avg_logprob,
                "no_speech_prob": w.no_speech_prob,
            })
        })
        .collect();
    let ranges_json: Vec<Value> = ranges
        .iter()
        .map(|&(start, end)| {
            json!({
                "start_window": start,
                "end_window": end,
                "start_sec": start as f64 * WINDOW_SECONDS,
            })
        })
        .collect();
    json!({
        "engine": "insanely-fast-native",
        "schema_version": SCHEMA_VERSION,
        "in_process": true,
        "implementation": "real-inference",
        "silence": silence,
        "parallel_windows": parallel_windows,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "ranges": ranges_json,
        "seams": seams,
        "model": spec,
        "model_path": model_path.display().to_string(),
        "model_version_tag": version_tag,
        "windows": windows_json,
        "word_timestamps": word_timestamps,
    })
}

/// Build the empty-but-valid result for a pure-silence clip, taken **without
/// loading the model** (the energy pre-gate already proved there is nothing to
/// transcribe).
fn silence_result(
    request: &TranscribeRequest,
    spec: &str,
    duration_ms: u64,
) -> TranscriptionResult {
    TranscriptionResult {
        backend: BackendKind::InsanelyFast,
        transcript: String::new(),
        language: request.language.clone(),
        segments: Vec::new(),
        acceleration: None,
        raw_output: json!({
            "engine": "insanely-fast-native",
            "schema_version": SCHEMA_VERSION,
            "in_process": true,
            "implementation": "real-inference",
            "silence": true,
            "parallel_windows": 0,
            "workers": 0,
            "threads_per_worker": 0,
            "ranges": [],
            "seams": 0,
            "model": spec,
            "model_loaded": false,
            "duration_ms": duration_ms,
            "windows": [],
            "word_timestamps": "none",
        }),
        artifact_paths: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use crate::backend::Engine;
    use crate::model::{
        BackendKind, BackendParams, InputSource, TranscribeRequest, TranscriptionSegment,
    };
    use crate::native_engine::mel::SAMPLE_RATE;
    use crate::orchestrator::CancellationToken;

    use super::*;

    fn request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::InsanelyFast,
            model: Some("tiny.en".to_owned()),
            language: Some("en".to_owned()),
            translate: false,
            diarize: false,
            persist: false,
            db_path: PathBuf::from("state.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        }
    }

    fn write_pcm16_mono_wav(path: &Path, sample_rate: u32, samples: &[i16]) {
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write wav");
    }

    // ── Engine-trait shape ────────────────────────────────────────────────

    #[test]
    fn native_engine_name_follows_naming_convention() {
        let engine = super::super::InsanelyFastNativeEngine;
        assert_eq!(engine.name(), "insanely-fast-native");
    }

    #[test]
    fn native_engine_kind_matches_bridge_adapter() {
        let native = super::super::InsanelyFastNativeEngine;
        let bridge = super::super::InsanelyFastEngine;
        assert_eq!(native.kind(), bridge.kind());
        assert_eq!(native.kind(), BackendKind::InsanelyFast);
    }

    #[test]
    fn native_engine_name_distinct_from_bridge() {
        let native = super::super::InsanelyFastNativeEngine;
        let bridge = super::super::InsanelyFastEngine;
        assert_ne!(native.name(), bridge.name());
        assert!(native.name().contains("native"));
    }

    #[test]
    fn availability_agrees_with_whisper_cpp_native() {
        // Both engines run the same native engine over the same model files, so
        // their availability must be identical.
        assert_eq!(
            is_available(),
            super::super::whisper_cpp_native::is_available()
        );
    }

    // ── plan_workers pure-function matrix ─────────────────────────────────

    #[test]
    fn plan_workers_typical_split() {
        // 16 threads, plenty of windows, default batch: headroom = 16/5 = 3,
        // workers = min(10, 24, 3) = 3; threads_per_worker = 16 / 3 = 5. The
        // round-3 policy keeps each worker at >= MIN_THREADS_PER_WORKER so the
        // product (3*5=15) never oversubscribes the 16 threads.
        let (w, t) = plan_workers(10, None, 16);
        assert_eq!(w, 3);
        assert_eq!(t, 5);
        assert!(w * t <= 16, "must not oversubscribe");
    }

    #[test]
    fn plan_workers_one_window() {
        // A single window => exactly one worker gets the whole thread budget.
        let (w, t) = plan_workers(1, None, 16);
        assert_eq!(w, 1);
        assert_eq!(t, 16);
    }

    #[test]
    fn plan_workers_small_thread_budget_falls_back_to_one_worker() {
        // < 2 * MIN_THREADS_PER_WORKER (i.e. < 10) threads: headroom = total/5
        // floored at 1 => a single worker (intra-op parallelism alone already
        // fills the machine; degenerates to the byte-exact sequential path).
        for total in 1..=9 {
            let (w, t) = plan_workers(50, None, total);
            assert_eq!(w, 1, "total_threads={total} must yield 1 worker");
            assert_eq!(t, total.max(1));
        }
    }

    #[test]
    fn plan_workers_no_oversubscription_invariant() {
        // For any plausible (windows, threads) the product workers*threads never
        // exceeds the thread budget — the core round-3 oversubscription fix.
        for total in 1..=64 {
            for windows in [1usize, 2, 4, 8, 16, 64, 240] {
                let (w, t) = plan_workers(windows, None, total);
                assert!(
                    w * t <= total.max(1),
                    "windows={windows} total={total} => {w}*{t} > {total}"
                );
                assert!(w >= 1 && t >= 1);
            }
        }
    }

    #[test]
    fn plan_workers_batch_size_one_serializes() {
        // batch_size = 1 caps concurrency to one worker regardless of windows.
        let (w, t) = plan_workers(20, Some(1), 32);
        assert_eq!(w, 1);
        assert_eq!(t, 32);
    }

    #[test]
    fn plan_workers_zero_threads_guarded() {
        // Zero threads must never yield a zero worker or zero thread count.
        let (w, t) = plan_workers(8, None, 0);
        assert!(w >= 1, "workers floored at 1, got {w}");
        assert!(t >= 1, "threads_per_worker floored at 1, got {t}");
    }

    #[test]
    fn plan_workers_zero_windows_guarded() {
        // Defensive: zero windows still floors workers at 1 (run() never calls
        // with zero, but the pure fn must not divide by zero).
        let (w, t) = plan_workers(0, None, 8);
        assert!(w >= 1);
        assert!(t >= 1);
    }

    #[test]
    fn plan_workers_batch_caps_below_headroom() {
        // batch_size limits workers even when threads/windows would allow more:
        // headroom = 64/4 = 16, but batch_size 4 caps to 4 workers; tpw = 64/4 = 16.
        let (w, t) = plan_workers(100, Some(4), 64);
        assert_eq!(w, 4);
        assert_eq!(t, 16);
    }

    // ── n_windows / plan_ranges / range_samples ───────────────────────────

    #[test]
    fn n_windows_counts_30s_chunks() {
        assert_eq!(n_windows(&[]), 0);
        assert_eq!(n_windows(&vec![0.1f32; SAMPLE_RATE]), 1); // 1 s < 30 s.
        assert_eq!(n_windows(&vec![0.1f32; N_SAMPLES_30S]), 1); // exactly 30 s.
        assert_eq!(n_windows(&vec![0.1f32; N_SAMPLES_30S + 1]), 2); // 30 s + 1.
        assert_eq!(n_windows(&vec![0.1f32; N_SAMPLES_30S * 3]), 3);
    }

    #[test]
    fn plan_ranges_single_worker_is_whole_clip() {
        // 1 worker => exactly one range spanning all windows (the byte-exact
        // sequential contract).
        assert_eq!(plan_ranges(5, 1), vec![(0, 5)]);
        assert_eq!(plan_ranges(1, 1), vec![(0, 1)]);
    }

    #[test]
    fn plan_ranges_contiguous_cover_no_gaps() {
        // Ranges are contiguous, half-open, and cover [0, n_windows) exactly.
        for (n, workers) in [(10, 3), (7, 2), (240, 4), (5, 8)] {
            let ranges = plan_ranges(n, workers);
            assert_eq!(ranges.first().map(|r| r.0), Some(0));
            assert_eq!(ranges.last().map(|r| r.1), Some(n));
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].1, pair[1].0, "ranges must be contiguous");
            }
            // never more ranges than workers or windows.
            assert!(ranges.len() <= workers.min(n).max(1));
        }
    }

    #[test]
    fn plan_ranges_empty_clip_is_empty() {
        assert!(plan_ranges(0, 4).is_empty());
    }

    #[test]
    fn range_samples_slices_contiguous_span() {
        // 30 s + 5 s clip, range [1, 2) => the 5 s remainder.
        let samples = vec![0.1f32; N_SAMPLES_30S + 5 * SAMPLE_RATE];
        let span = range_samples(&samples, (1, 2));
        assert_eq!(span.len(), 5 * SAMPLE_RATE);
        // range [0, 1) => the first full 30 s window.
        assert_eq!(range_samples(&samples, (0, 1)).len(), N_SAMPLES_30S);
        // whole-clip range [0, 2) => all samples.
        assert_eq!(range_samples(&samples, (0, 2)).len(), samples.len());
    }

    // ── merge_ranges offsets timestamps by the range's base window ─────────

    fn out_with_segment(start: f64, end: f64, text: &str, offset_sec: f64) -> decode::DecodeOutput {
        decode::DecodeOutput {
            segments: vec![TranscriptionSegment {
                start_sec: Some(start),
                end_sec: Some(end),
                text: text.to_owned(),
                speaker: None,
                confidence: Some(0.9),
            }],
            language: Some("en".to_owned()),
            windows: vec![decode::WindowStats {
                avg_logprob: -0.2,
                no_speech_prob: 0.01,
                tokens: 3,
                window_offset_sec: offset_sec,
            }],
            word_timings: None,
        }
    }

    #[test]
    fn merge_ranges_offsets_by_start_window_times_30s() {
        // Two contiguous ranges: range 0 starts at window 0 (offset 0 s); range 1
        // starts at window 2 (offset 60 s). Each range's decode emits its
        // segments RELATIVE to the range start, so the merge offsets range 1 by
        // 60 s.
        let results = vec![
            RangeResult {
                start_window: 0,
                output: out_with_segment(0.0, 5.0, "first", 0.0),
            },
            RangeResult {
                start_window: 2,
                output: out_with_segment(1.0, 4.0, "second", 0.0),
            },
        ];
        let merged = merge_ranges(&results);
        assert_eq!(merged.segments.len(), 2);
        // Range 0's segment is unshifted.
        assert_eq!(merged.segments[0].start_sec, Some(0.0));
        // Range 1's segment is offset by 2 * 30 s = 60 s.
        assert_eq!(merged.segments[1].start_sec, Some(61.0));
        assert_eq!(merged.segments[1].end_sec, Some(64.0));
        // Window stats offset too.
        assert_eq!(merged.windows[1].window_offset_sec, 60.0);
        assert_eq!(merged.language, Some("en".to_owned()));
    }

    // ── Model resolution / availability ───────────────────────────────────

    #[test]
    fn run_without_model_or_default_is_backend_unavailable() {
        let mut req = request();
        req.model = None;
        if native_engine::default_model_spec().is_some() {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("a.wav");
        let mut samples = vec![0i16; 1_600];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000i16 } else { -9_000 }));
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let err = run(&req, &wav, dir.path(), Duration::from_secs(1), None)
            .expect_err("no model => unavailable");
        assert!(matches!(err, FwError::BackendUnavailable(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL"),
            "message names the env var: {msg}"
        );
    }

    #[test]
    fn run_nonexistent_model_spec_is_backend_unavailable_with_dirs() {
        let mut req = request();
        req.model = Some("definitely-not-a-real-model-zzz".to_owned());
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("a.wav");
        let mut samples = vec![0i16; 1_600];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000i16 } else { -9_000 }));
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let err = run(&req, &wav, dir.path(), Duration::from_secs(1), None)
            .expect_err("missing model => unavailable");
        assert!(matches!(err, FwError::BackendUnavailable(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("ggml-definitely-not-a-real-model-zzz.bin"),
            "message names the searched filename: {msg}"
        );
    }

    // ── Diarization gate ──────────────────────────────────────────────────

    #[test]
    fn run_diarize_without_token_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");
        let mut samples = vec![0i16; 1_600];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000i16 } else { -9_000 }));
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let mut req = request();
        req.diarize = true;
        let result = run(&req, &wav, dir.path(), Duration::from_secs(1), None);
        assert!(result.is_err(), "diarize without HF token should fail");
    }

    // ── Silence pre-gate ──────────────────────────────────────────────────

    #[test]
    fn run_pure_silence_returns_empty_without_loading_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("silence.wav");
        write_pcm16_mono_wav(&wav, 16_000, &vec![0i16; 16_000]);

        let req = request();
        let start = std::time::Instant::now();
        let result = run(&req, &wav, dir.path(), Duration::from_secs(5), None)
            .expect("silence run should succeed");
        let elapsed = start.elapsed();

        assert!(result.segments.is_empty(), "silence => no segments");
        assert!(result.transcript.is_empty(), "silence => empty transcript");
        assert_eq!(result.raw_output["silence"].as_bool(), Some(true));
        assert_eq!(result.raw_output["model_loaded"].as_bool(), Some(false));
        assert_eq!(
            result.raw_output["engine"].as_str(),
            Some("insanely-fast-native")
        );
        assert_eq!(
            result.raw_output["schema_version"].as_str(),
            Some(SCHEMA_VERSION)
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "silence pre-gate should return fast, took {elapsed:?}"
        );
    }

    // ── Cancellation ──────────────────────────────────────────────────────

    #[test]
    fn run_expired_token_is_cancelled_quickly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");
        let mut samples = vec![0i16; 1_600];
        samples.extend((0..16_000).map(|i| if i % 2 == 0 { 9_000i16 } else { -9_000 }));
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let req = request();
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));

        let start = std::time::Instant::now();
        let result = run(&req, &wav, dir.path(), Duration::from_secs(1), Some(&token));
        assert!(result.is_err(), "expired token must cancel");
        assert!(
            matches!(result.unwrap_err(), FwError::Cancelled(_)),
            "expected FW-CANCELLED"
        );
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "cancellation should be fast"
        );
    }

    // ── raw_output schema ─────────────────────────────────────────────────

    #[test]
    fn raw_output_carries_parallel_range_metadata() {
        // 4 windows split into 2 contiguous ranges => 1 seam.
        let ranges = vec![(0usize, 2usize), (2, 4)];
        let json = raw_output_json(
            "tiny.en",
            Path::new("/models/ggml-tiny.en.bin"),
            "fw-native-v1+sha256:abc".to_owned(),
            4,
            2,
            4,
            &ranges,
            1,
            &[],
            WordTimestampMode::Word,
            false,
            false,
        );
        assert_eq!(json["engine"].as_str(), Some("insanely-fast-native"));
        assert_eq!(json["schema_version"].as_str(), Some(SCHEMA_VERSION));
        assert_eq!(json["implementation"].as_str(), Some("real-inference"));
        assert_eq!(json["in_process"].as_bool(), Some(true));
        assert_eq!(json["parallel_windows"].as_u64(), Some(4));
        assert_eq!(json["workers"].as_u64(), Some(2));
        assert_eq!(json["threads_per_worker"].as_u64(), Some(4));
        assert_eq!(json["word_timestamps"].as_str(), Some("interpolated"));
        // Additive native-v2 fields: ranges + seams.
        assert_eq!(json["seams"].as_u64(), Some(1));
        let ranges_json = json["ranges"].as_array().expect("ranges array");
        assert_eq!(ranges_json.len(), 2);
        assert_eq!(ranges_json[0]["start_window"].as_u64(), Some(0));
        assert_eq!(ranges_json[0]["end_window"].as_u64(), Some(2));
        assert_eq!(ranges_json[1]["start_window"].as_u64(), Some(2));
        assert_eq!(ranges_json[1]["start_sec"].as_f64(), Some(60.0));
    }

    // ── Gated end-to-end against the real tiny.en model + jfk.wav ─────────

    /// The exact reference transcript from
    /// `tests/fixtures/native/jfk_tiny_reference.json` (joined, trimmed).
    const JFK_REFERENCE: &str = "And so my fellow Americans ask not what your country can do for \
        you ask what you can do for your country.";

    fn tiny_en_available() -> bool {
        native_engine::find_model_file("tiny.en").is_some()
    }

    fn load_jfk_samples() -> Option<Vec<f32>> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let bytes = std::fs::read(path).ok()?;
        decode::read_wav_16k_mono(&bytes).ok()
    }

    fn load_tiny_en() -> Option<std::sync::Arc<NativeWhisperModel>> {
        let path = native_engine::find_model_file("tiny.en")?;
        NativeWhisperModel::load(&path).ok()
    }

    #[test]
    fn gated_single_window_matches_whisper_cpp_native() {
        if !tiny_en_available() {
            eprintln!("SKIP gated_single_window: tiny.en model missing");
            return;
        }
        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let mut req = request();
        req.model = Some("tiny.en".to_owned());
        req.language = None;

        let engine = super::super::InsanelyFastNativeEngine;
        let result = crate::backend::Engine::run(
            &engine,
            &req,
            Path::new(wav),
            Path::new("."),
            Duration::from_secs(120),
            None,
        )
        .expect("e2e run");

        // jfk.wav is ~11 s => a single 30 s window. The single-window path runs
        // the SAME native engine over the SAME samples as whisper.cpp native, so
        // the transcript must match the shared reference exactly.
        assert_eq!(result.transcript.trim(), JFK_REFERENCE);
        assert_eq!(
            result.raw_output["engine"].as_str(),
            Some("insanely-fast-native")
        );
        assert_eq!(result.raw_output["parallel_windows"].as_u64(), Some(1));
        assert_eq!(result.raw_output["silence"].as_bool(), Some(false));
        assert!(result.segments.len() >= 2, "expected >= 2 segments");
    }

    /// Greedy `DecodeParams` matching what [`run`] builds (timestamps on,
    /// auto-detect language), with the given intra-op thread budget.
    fn det_params(n_threads: usize) -> decode::DecodeParams {
        decode::DecodeParams {
            language: None,
            translate: false,
            timestamps: true,
            n_threads,
            max_text_ctx: None,
            ..decode::DecodeParams::default()
        }
    }

    /// Decode `samples` with an explicit forced worker count (via batch_size),
    /// running the real contiguous-range path, and return the engine-level
    /// (merged, offset) outputs as plain text + segment bounds + (n_workers,
    /// n_windows). This exercises exactly the [`run`] partition/merge machinery.
    fn decode_with_workers(
        model: &NativeWhisperModel,
        samples: &[f32],
        batch_size: Option<usize>,
        total_threads: usize,
    ) -> (String, Vec<(f64, f64)>, usize, usize) {
        let win_count = n_windows(samples);
        let (n_workers, tpw) = plan_workers(win_count, batch_size, total_threads);
        let ranges = plan_ranges(win_count, n_workers);
        let params = det_params(tpw);
        let results =
            decode_ranges_parallel(model, samples, &ranges, &params, None).expect("decode");
        let merged = merge_ranges(&results);
        let text: String = merged
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        let bounds: Vec<(f64, f64)> = merged
            .segments
            .iter()
            .map(|s| (s.start_sec.unwrap_or(0.0), s.end_sec.unwrap_or(0.0)))
            .collect();
        (text, bounds, ranges.len(), win_count)
    }

    /// Direct sequential reference: one `transcribe_samples` over the whole clip
    /// (the byte-golden whisper.cpp-native path), rendered to the same
    /// (text, bounds) shape as [`decode_with_workers`].
    fn sequential_reference(
        model: &NativeWhisperModel,
        samples: &[f32],
        n_threads: usize,
    ) -> (String, Vec<(f64, f64)>) {
        let params = det_params(n_threads);
        let out = model
            .transcribe(samples, &params, &|| Ok(()))
            .expect("sequential decode");
        let text: String = out
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        let bounds: Vec<(f64, f64)> = out
            .segments
            .iter()
            .map(|s| (s.start_sec.unwrap_or(0.0), s.end_sec.unwrap_or(0.0)))
            .collect();
        (text, bounds)
    }

    /// Word-diff rate between two transcripts: the symmetric word-multiset
    /// difference normalized by the larger word count (a coarse, order-agnostic
    /// divergence metric, matching the scale-baseline HOTSPOTS measure).
    fn word_diff_rate(a: &str, b: &str) -> f64 {
        use std::collections::HashMap;
        let count = |s: &str| {
            let mut m: HashMap<String, i64> = HashMap::new();
            for w in s.split_whitespace() {
                *m.entry(w.to_lowercase()).or_default() += 1;
            }
            m
        };
        let ma = count(a);
        let mb = count(b);
        let mut keys: std::collections::HashSet<&String> = ma.keys().collect();
        keys.extend(mb.keys());
        let diff: i64 = keys
            .iter()
            .map(|k| (ma.get(*k).copied().unwrap_or(0) - mb.get(*k).copied().unwrap_or(0)).abs())
            .sum();
        let total = a
            .split_whitespace()
            .count()
            .max(b.split_whitespace().count());
        if total == 0 {
            0.0
        } else {
            diff as f64 / total as f64
        }
    }

    /// jfk concatenated 3× (~33 s => 2 windows) — the fixed multi-window fixture.
    fn jfk3x_samples() -> Option<Vec<f32>> {
        let samples = load_jfk_samples()?;
        let mut long = Vec::with_capacity(samples.len() * 3);
        for _ in 0..3 {
            long.extend_from_slice(&samples);
        }
        Some(long)
    }

    #[test]
    fn gated_one_worker_equals_sequential_byte_exact() {
        // NEW CONTRACT: with a single worker there is exactly one range spanning
        // the whole clip, so the IF path degenerates to one `transcribe_samples`
        // call — byte-identical to the sequential whisper.cpp-native path.
        let Some(model) = load_tiny_en() else {
            eprintln!("SKIP gated_one_worker_equals_sequential: tiny.en model missing");
            return;
        };
        let Some(long) = jfk3x_samples() else {
            eprintln!("SKIP gated_one_worker_equals_sequential: jfk.wav missing");
            return;
        };

        // Force 1 worker (batch_size 1).
        let (text_if1, bounds_if1, n_workers, win_count) =
            decode_with_workers(&model, &long, Some(1), 8);
        assert_eq!(n_workers, 1, "batch_size 1 forces a single worker/range");
        assert!(win_count >= 2, "expected >= 2 windows for ~33s audio");

        let (text_seq, bounds_seq) = sequential_reference(&model, &long, 8);

        assert_eq!(
            text_if1, text_seq,
            "1-worker IF transcript must equal sequential exactly"
        );
        assert_eq!(
            bounds_if1, bounds_seq,
            "1-worker IF timestamps must equal sequential exactly"
        );
        // The whole-clip sequential decode covers both internal 30 s windows:
        // its last segment ends past 30 s.
        assert!(
            bounds_seq.last().is_some_and(|&(_, e)| e >= 30.0 - 1e-6),
            "expected the decode to cover past 30 s: {bounds_seq:?}"
        );
    }

    #[test]
    fn gated_fixed_count_deterministic() {
        // NEW CONTRACT: output varies WITH worker count by design, but is
        // deterministic FOR A FIXED count — re-running the 2-worker partition
        // yields byte-identical text and timestamps.
        let Some(model) = load_tiny_en() else {
            eprintln!("SKIP gated_fixed_count_deterministic: tiny.en model missing");
            return;
        };
        let Some(long) = jfk3x_samples() else {
            eprintln!("SKIP gated_fixed_count_deterministic: jfk.wav missing");
            return;
        };

        // batch_size 8 with a 16-thread budget => headroom 16/5 = 3, capped to
        // the 2 windows => 2 workers (one seam) for this 2-window clip.
        let (text_a, bounds_a, workers_a, _) = decode_with_workers(&model, &long, Some(8), 16);
        let (text_b, bounds_b, workers_b, _) = decode_with_workers(&model, &long, Some(8), 16);
        assert_eq!(workers_a, 2, "expected 2 workers for the 2-window fixture");
        assert_eq!(workers_a, workers_b);
        assert_eq!(text_a, text_b, "fixed worker count must be deterministic");
        assert_eq!(bounds_a, bounds_b, "fixed worker count timestamps stable");

        // Timestamps monotonic non-decreasing across the seam.
        let mut prev_end = -1.0f64;
        for (start, end) in &bounds_a {
            assert!(
                *start + 1e-6 >= prev_end - 1e-6,
                "segment start {start} must not precede previous end {prev_end}"
            );
            assert!(*end + 1e-6 >= *start, "segment end {end} >= start {start}");
            prev_end = *end;
        }
        // The second range (>= 30 s) is present.
        assert!(
            bounds_a.iter().any(|(s, _)| *s >= 30.0 - 1e-6),
            "expected segments in the second range (>= 30 s): {bounds_a:?}"
        );
    }

    #[test]
    fn gated_word_diff_vs_sequential_bounded() {
        // NEW CONTRACT: with contiguous ranges + real per-range sequential decode,
        // the N-worker word-diff vs sequential must be SMALL (seams are rare).
        // Assert < 10 % on the jfk3x fixture (was 67 % with the old hard-window
        // round-robin design).
        let Some(model) = load_tiny_en() else {
            eprintln!("SKIP gated_word_diff_vs_sequential: tiny.en model missing");
            return;
        };
        let Some(long) = jfk3x_samples() else {
            eprintln!("SKIP gated_word_diff_vs_sequential: jfk.wav missing");
            return;
        };

        let (text_par, _, workers, _) = decode_with_workers(&model, &long, Some(8), 16);
        assert_eq!(workers, 2, "expected the 2-worker partition");
        let (text_seq, _) = sequential_reference(&model, &long, 16);

        let rate = word_diff_rate(&text_par, &text_seq);
        assert!(
            rate < 0.10,
            "2-worker word-diff vs sequential must be < 10%, got {:.1}% \
             (par={text_par:?} seq={text_seq:?})",
            rate * 100.0
        );

        // The signature word still recurs in the parallel output.
        assert!(
            text_par.to_lowercase().matches("country").count() >= 2,
            "expected the repeated sentence at least twice: {text_par}"
        );
    }
}
