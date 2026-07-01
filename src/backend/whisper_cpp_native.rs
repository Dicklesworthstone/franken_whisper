//! Real native whisper.cpp engine (bd-jryr).
//!
//! This module is the in-process, pure-Rust whisper engine. It runs genuine
//! ASR inference through [`crate::native_engine`] — it parses a ggml model,
//! computes the log-mel frontend, and runs the encoder/decoder forward passes
//! — and contains **no canned phrases, no mock segmentation, and no subprocess
//! execution**. The former "pilot" that fabricated deterministic phrases from
//! audio-energy regions is gone; the real decoder decides what (and when) words
//! were spoken.
//!
//! ## Model resolution & availability
//!
//! [`run`] resolves the model from `request.model`, falling back to
//! `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`. With neither set it returns
//! [`FwError::BackendUnavailable`] carrying the model resolver's own
//! search-dir-listing message (actionable: drop a `ggml-*.bin` in a search dir,
//! or set the env var). [`is_available`] is now **honest**: it reports `true`
//! only when a usable model header exists (either the configured default
//! resolves, or any `ggml-*.bin` with a valid header sits in a search dir). The
//! old always-`true` lie is dead — with no model, the router stays bridge-only
//! instead of advertising a fake native recovery path.
//!
//! ## Silence pre-gate
//!
//! Before loading the model (a multi-GB cost for large models), [`run`] runs
//! the cheap energy analyzer ([`analyze_wav`]). If the clip is pure silence
//! (zero active regions) it returns an empty-but-valid result tagged
//! `"silence": true` **without loading any weights**. Otherwise the energy
//! analysis is ignored for segmentation: the real engine, not waveform energy,
//! decides segment boundaries.
//!
//! ## Word timestamps
//!
//! When a request asks for word-level timestamps, real segments are split into
//! words with time linearly interpolated within each segment's bounds, tagged
//! `"word_timestamps": "interpolated"`. A follow-up bead (bd-rjsx) replaces the
//! linear interpolation with attention-DTW and flips the flag to `"dtw"`.

use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, TranscribeRequest, TranscriptionResult, TranscriptionSegment, WordTimestampParams,
};
use crate::native_engine::dtw::WordTiming;
use crate::native_engine::{self, NativeWhisperModel, decode};

use super::native_audio::analyze_wav;

/// Stable schema tag for the honest native raw-output metadata.
const SCHEMA_VERSION: &str = "native-v2";

/// Word-timestamp post-processing mode derived from the request.
///
/// `pub(crate)` so sibling native engines (e.g. `insanely_fast_native`) reuse
/// the same word-splitting/grouping policy rather than re-deriving it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WordTimestampMode {
    /// No word-level splitting: keep the engine's native segments.
    None,
    /// Split every segment into individual words.
    Word,
    /// Split into words, then regroup runs of words up to `max_len` characters.
    MaxLen(u32),
}

/// Map the request's [`WordTimestampParams`] to a [`WordTimestampMode`].
///
/// Mirrors the historical control flow: `max_len` (when present) decides the
/// shape (`0` disables, `1` = per-word, `>1` = grouped), otherwise any of
/// `enabled` / `token_threshold` / `token_sum_threshold` enables per-word split.
pub(crate) fn word_timestamp_mode(params: Option<&WordTimestampParams>) -> WordTimestampMode {
    let Some(params) = params else {
        return WordTimestampMode::None;
    };

    if let Some(max_len) = params.max_len {
        return match max_len {
            0 => WordTimestampMode::None,
            1 => WordTimestampMode::Word,
            _ => WordTimestampMode::MaxLen(max_len),
        };
    }

    if params.enabled || params.token_threshold.is_some() || params.token_sum_threshold.is_some() {
        WordTimestampMode::Word
    } else {
        WordTimestampMode::None
    }
}

/// Honestly report whether the native whisper.cpp engine can run.
///
/// Availability is probed **without a request context** (the router calls this
/// before dispatch), so the policy is:
///
/// 1. If `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL` is set and its model header
///    validates, report `true`.
/// 2. Otherwise, report `true` iff **any** `ggml-*.bin` with a valid header
///    exists in the model search dirs — a cheap directory scan. This keeps a
///    request that names an explicit model (with no default configured) from
///    being refused at the dispatch gate while still being honest: with zero
///    usable model files on disk, the engine reports itself unavailable and the
///    router stays bridge-only.
///
/// Never panics or performs any network access (header-only sniffing).
///
/// # Caching
///
/// The underlying probe ([`is_available_uncached`]) re-scans up to five model
/// directories and header-sniffs every `ggml-*.bin` it finds. That is cheap in
/// isolation but the router calls `is_available` on *every* routing decision and
/// the robot health endpoint iterates all three engines, so the same scan was
/// being repeated many times per second under load. We memoize the result
/// behind a process-global [`Mutex`] with a short [`AVAILABILITY_TTL`] so bursts
/// of probes collapse to one scan, while a freshly-provisioned model file still
/// becomes visible within the TTL window — important for tests that create a
/// model file mid-process and then probe availability. Tests that need the cache
/// cleared immediately can call [`reset_availability_cache`] (test-only).
#[must_use]
pub fn is_available() -> bool {
    let now = Instant::now();
    {
        let guard = availability_cache()
            .lock()
            .expect("availability cache lock");
        if let Some((stamped, value)) = *guard
            && now.duration_since(stamped) < AVAILABILITY_TTL
        {
            return value;
        }
    }
    let value = is_available_uncached();
    let mut guard = availability_cache()
        .lock()
        .expect("availability cache lock");
    *guard = Some((Instant::now(), value));
    value
}

/// Time-to-live for the [`is_available`] memoization. Two seconds is short
/// enough that a model file created mid-process (the worst case is a test that
/// drops a `ggml-*.bin` in a search dir and immediately re-probes) is observed
/// promptly, yet long enough that the per-routing-decision and per-health-check
/// probe bursts collapse to a single directory scan.
const AVAILABILITY_TTL: Duration = Duration::from_secs(2);

/// Process-global cache of the most recent availability probe: `(taken_at,
/// available)`. Lazily initialized; guarded by a [`Mutex`] because probes can
/// race across the async runtime's worker threads.
fn availability_cache() -> &'static Mutex<Option<(Instant, bool)>> {
    static CACHE: std::sync::OnceLock<Mutex<Option<(Instant, bool)>>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Clear the [`is_available`] memoization so the next probe re-scans the model
/// search dirs. Test-only: in-process tests that toggle availability (e.g. by
/// creating or removing a model file) need a deterministic flip without waiting
/// for [`AVAILABILITY_TTL`].
#[cfg(test)]
pub(crate) fn reset_availability_cache() {
    *availability_cache()
        .lock()
        .expect("availability cache lock") = None;
}

/// The uncached availability probe. Performs the actual directory scan and
/// header sniffing; see [`is_available`] for the memoizing wrapper.
fn is_available_uncached() -> bool {
    if let Some(spec) = native_engine::default_model_spec()
        && native_engine::native_model_available(&spec)
    {
        return true;
    }
    any_model_in_search_dirs()
}

/// Number of leading bytes covering the ggml magic plus the eleven `i32`
/// hparams (`4 + 11 * 4`). Duplicated from the parser so availability sniffing
/// needs no private engine internals.
const HEADER_SNIFF_LEN: usize = 48;

/// ggml file magic (`"ggml"` as a little-endian `u32`).
const GGML_MAGIC: u32 = 0x6767_6d6c;

/// The directories scanned for `ggml-*.bin` model files, mirroring
/// [`crate::native_engine`]'s precedence so availability never disagrees with
/// resolution.
///
/// 1. `$FRANKEN_WHISPER_MODEL_DIR`
/// 2. `$FRANKEN_WHISPER_TEST_MODEL_DIR`
/// 3. `~/.cache/franken_whisper/models`
/// 4. `~/.cache/franken_whisper/test-models`
/// 5. `~/models/whisper`
fn model_search_dirs() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
    let mut dirs: Vec<PathBuf> = Vec::new();
    for var in [
        "FRANKEN_WHISPER_MODEL_DIR",
        "FRANKEN_WHISPER_TEST_MODEL_DIR",
    ] {
        if let Ok(dir) = std::env::var(var)
            && !dir.is_empty()
        {
            dirs.push(PathBuf::from(dir));
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        dirs.push(home.join(".cache").join("franken_whisper").join("models"));
        dirs.push(
            home.join(".cache")
                .join("franken_whisper")
                .join("test-models"),
        );
        dirs.push(home.join("models").join("whisper"));
    }
    dirs
}

/// Honestly report whether **any** `ggml-*.bin` with a valid header exists in a
/// search dir. A cheap directory scan (header-only sniff per candidate, no
/// network, never panics). Used by [`is_available`] when no default model is
/// configured so a request that names an explicit model is not refused at the
/// dispatch gate — while still keeping availability honest (zero usable model
/// files => `false`).
fn any_model_in_search_dirs() -> bool {
    for dir in model_search_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_ggml = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("ggml-") && n.ends_with(".bin"));
            if is_ggml && header_ftype_ok(&path) {
                return true;
            }
        }
    }
    false
}

/// Read the first 48 bytes of `path` and validate the ggml magic + a supported
/// dense `ftype` (`0` = f32, `1` = f16). Any failure yields `false`.
fn header_ftype_ok(path: &Path) -> bool {
    use std::io::Read as _;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; HEADER_SNIFF_LEN];
    if file.read_exact(&mut buf).is_err() {
        return false;
    }
    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != GGML_MAGIC {
        return false;
    }
    let ftype = i32::from_le_bytes([buf[44], buf[45], buf[46], buf[47]]);
    ftype == 0 || ftype == 1
}

/// Resolve the effective model spec for a request, or a [`FwError`] explaining
/// how to provision one.
///
/// Precedence: `request.model` (when set) then
/// `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`. With neither set, returns
/// [`FwError::BackendUnavailable`] whose message names the env var and reuses
/// [`native_engine::resolve_model`]'s search-dir listing so the fix is obvious.
fn effective_model_spec(request: &TranscribeRequest) -> FwResult<String> {
    if let Some(model) = request.model.clone().filter(|m| !m.is_empty()) {
        return Ok(model);
    }
    if let Some(spec) = native_engine::default_model_spec() {
        return Ok(spec);
    }
    // No request model and no default: produce an actionable message that
    // reuses the resolver's search-dir listing (by attempting a resolution of a
    // placeholder, whose error text enumerates the searched directories).
    let dirs_hint = native_engine::resolve_model("default")
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    Err(FwError::BackendUnavailable(format!(
        "native whisper.cpp engine has no model: pass --model, or set \
         $FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL to a model short-name or path. \
         {dirs_hint}"
    )))
}

/// Build the [`decode::DecodeParams`] for a request.
///
/// `want_dtw_words` enables cross-attention recording + DTW word-timestamp
/// computation in the engine (bd-rjsx); `spec` is passed through as the
/// alignment-head preset hint (e.g. `"tiny.en"`).
fn decode_params(
    request: &TranscribeRequest,
    want_dtw_words: bool,
    spec: &str,
) -> decode::DecodeParams {
    let n_threads = request
        .backend_params
        .threads
        .map_or_else(native_engine::default_threads, |t| {
            usize::try_from(t).unwrap_or_else(|_| native_engine::default_threads())
        });
    decode::DecodeParams {
        language: request.language.clone(),
        translate: request.translate,
        timestamps: !request.backend_params.no_timestamps,
        n_threads,
        // No request field maps to whisper's text-context cap today; use the
        // decoder default (`n_text_ctx/2 - 4`). Plumbed here so a future knob
        // has an obvious home.
        max_text_ctx: None,
        word_timestamps: want_dtw_words,
        model_hint: if want_dtw_words {
            Some(spec.to_owned())
        } else {
            None
        },
    }
}

/// Bridge an optional orchestrator [`CancellationToken`] into the engine's
/// `checkpoint` closure shape (`Fn() -> FwResult<()>`).
fn checkpoint_for(
    token: Option<&crate::orchestrator::CancellationToken>,
) -> impl Fn() -> FwResult<()> + '_ {
    move || token.map_or(Ok(()), crate::orchestrator::CancellationToken::checkpoint)
}

/// Run real native whisper inference over `normalized_wav` (guaranteed 16 kHz
/// mono PCM16 by the pipeline) and return a [`TranscriptionResult`].
///
/// See the module docs for the model-resolution, silence pre-gate, and
/// word-timestamp policies.
///
/// # Errors
///
/// - [`FwError::BackendUnavailable`] when no model can be resolved.
/// - [`FwError::Io`] / [`FwError::InvalidRequest`] when the WAV cannot be read.
/// - [`FwError::Cancelled`] when the cancellation token's deadline expires.
/// - Whatever model-load or decode errors the native engine surfaces.
pub fn run(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    _work_dir: &Path,
    _timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<TranscriptionResult> {
    let t_backend = std::time::Instant::now();
    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Resolve the model spec up front so an unavailability error is reported
    // before any expensive work.
    let spec = effective_model_spec(request)?;

    // Silence pre-gate: the energy analyzer is cheap; loading a (potentially
    // multi-GB) model is not. Pure-silence clips skip the load entirely.
    let analysis = analyze_wav(normalized_wav, request.backend_params.duration_ms).ok();
    if let Some(analysis) = analysis.as_ref()
        && analysis.active_regions.is_empty()
    {
        return Ok(silence_result(request, &spec, analysis.duration_ms));
    }

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Resolve + load the model (cached). A resolution miss here is unavailable.
    let model_path = native_engine::resolve_model(&spec)
        .map_err(|e| FwError::BackendUnavailable(e.to_string()))?;
    let model = NativeWhisperModel::load(&model_path)?;

    // Read the normalized WAV to f32 mono samples.
    let samples = read_normalized_wav(normalized_wav)?;

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Word-timestamp policy: decide whether to ask the engine for real
    // attention-DTW word times. DTW runs when per-word output is requested
    // (a word/maxlen split or `split_on_word`) — the engine then records
    // cross-attention and aligns each word to audio frames (bd-rjsx).
    let word_mode = word_timestamp_mode(request.backend_params.word_timestamps.as_ref());
    let want_words = word_mode != WordTimestampMode::None || request.backend_params.split_on_word;
    // DTW is only meaningful when we keep per-segment timestamps.
    let want_dtw = want_words && !request.backend_params.no_timestamps;

    let params = decode_params(request, want_dtw, &spec);
    let checkpoint = checkpoint_for(token);
    let output = model.transcribe(&samples, &params, &checkpoint)?;

    // Prefer real DTW word timings when the engine produced them; otherwise fall
    // back to the linear-interpolation word split (keeping its existing tag).
    let dtw_words = output
        .word_timings
        .as_ref()
        .filter(|w| w.iter().any(|seg| !seg.is_empty()));
    let used_dtw = dtw_words.is_some();
    let segments = if let Some(word_timings) = dtw_words {
        build_segments_dtw(
            &output.segments,
            word_timings,
            word_mode,
            request.backend_params.no_timestamps,
            token,
        )?
    } else {
        build_segments(
            &output.segments,
            word_mode,
            request.backend_params.split_on_word,
            request.backend_params.no_timestamps,
            token,
        )?
    };
    let transcript = super::transcript_from_segments(&segments);
    let language = output.language.clone().or_else(|| request.language.clone());

    let t_tag = std::time::Instant::now();
    let version_tag = model.version_tag();
    crate::native_engine::perf_span("version_tag", t_tag.elapsed().as_secs_f64() * 1e3, "");
    crate::native_engine::perf_span("backend_run", t_backend.elapsed().as_secs_f64() * 1e3, "");
    let raw_output = raw_output_json(
        &spec,
        &model_path,
        version_tag,
        &output.windows,
        word_mode,
        request.backend_params.split_on_word,
        false,
        used_dtw,
    );

    Ok(TranscriptionResult {
        backend: BackendKind::WhisperCpp,
        transcript,
        language,
        segments,
        acceleration: None,
        raw_output,
        artifact_paths: Vec::new(),
    })
}

/// Streaming entry point.
///
/// The native decoder ([`decode::transcribe_samples`]) is batch-only, so this
/// runs the full [`run`] pathway and then replays the resulting segments
/// through `on_segment` in order. True window-level streaming (emitting each
/// 30 s window as it completes) lands with a follow-up; the previous mock was
/// not truly streaming either. The cancellation token is honored between
/// emitted segments.
///
/// # Errors
///
/// Same as [`run`]; additionally aborts (before emitting all segments) if the
/// cancellation token expires mid-replay.
pub fn run_streaming(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
    on_segment: &dyn Fn(TranscriptionSegment),
) -> FwResult<TranscriptionResult> {
    let mut result = run(request, normalized_wav, work_dir, timeout, token)?;

    for segment in &result.segments {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        on_segment(segment.clone());
    }

    // Record the emitted-segment count for parity with the prior schema, while
    // keeping the honest "real-inference" framing.
    if let Value::Object(map) = &mut result.raw_output {
        map.insert(
            "streaming_emitted_segments".to_owned(),
            json!(result.segments.len()),
        );
    }

    Ok(result)
}

/// Build the final [`TranscriptionSegment`] list from the engine's real
/// segments, applying word-timestamp splitting/grouping and the
/// `no_timestamps` policy.
///
/// `pub(crate)` so sibling native engines (e.g. `insanely_fast_native`) apply
/// the identical word-timestamp post-processing to their merged segments.
pub(crate) fn build_segments(
    engine_segments: &[TranscriptionSegment],
    word_mode: WordTimestampMode,
    split_on_word: bool,
    no_timestamps: bool,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<TranscriptionSegment>> {
    let segments = if word_mode == WordTimestampMode::None && !split_on_word {
        engine_segments.to_vec()
    } else {
        let words = explode_segments_to_words(engine_segments, token)?;
        match word_mode {
            WordTimestampMode::MaxLen(max_len) if max_len > 1 => {
                group_word_segments_by_len(&words, max_len, token)?
            }
            _ => words,
        }
    };

    finalize_segments(&segments, no_timestamps, token)
}

/// Build the final segment list from real **DTW-aligned** word timings
/// (bd-rjsx), exploding each engine segment into per-word segments whose
/// `[start, end]` come straight from cross-attention alignment rather than
/// linear interpolation.
///
/// `word_timings` is aligned 1:1 with `engine_segments` (the engine's contract).
/// A segment with no timed words falls back to interpolating that one segment's
/// text within its bounds, so output never loses words present in the
/// transcript. After the per-word explosion the same `MaxLen` regrouping policy
/// as [`build_segments`] is applied, then the shared finalization.
pub(crate) fn build_segments_dtw(
    engine_segments: &[TranscriptionSegment],
    word_timings: &[Vec<WordTiming>],
    word_mode: WordTimestampMode,
    no_timestamps: bool,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<TranscriptionSegment>> {
    let mut words: Vec<TranscriptionSegment> = Vec::new();
    for (idx, segment) in engine_segments.iter().enumerate() {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        match word_timings.get(idx) {
            Some(timed) if !timed.is_empty() => {
                for w in timed {
                    words.push(TranscriptionSegment {
                        start_sec: Some(w.start_sec),
                        end_sec: Some(w.end_sec),
                        text: w.text.clone(),
                        speaker: None,
                        confidence: segment.confidence,
                    });
                }
            }
            // No DTW words for this segment: interpolate it alone so no words
            // are dropped.
            _ => {
                let interp = explode_segments_to_words(std::slice::from_ref(segment), token)?;
                words.extend(interp);
            }
        }
    }

    let segments = match word_mode {
        WordTimestampMode::MaxLen(max_len) if max_len > 1 => {
            group_word_segments_by_len(&words, max_len, token)?
        }
        _ => words,
    };

    finalize_segments(&segments, no_timestamps, token)
}

/// Split each real segment's text into words, linearly interpolating each
/// word's `[start, end]` within the parent segment's time bounds.
///
/// Segments without timestamps (e.g. under `no_timestamps`) keep `None` bounds
/// for every produced word. Empty/whitespace-only segments are skipped.
fn explode_segments_to_words(
    segments: &[TranscriptionSegment],
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<TranscriptionSegment>> {
    let mut out = Vec::new();
    for segment in segments {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        let words: Vec<&str> = segment.text.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let n = words.len() as f64;
        for (index, word) in words.iter().enumerate() {
            let (start_sec, end_sec) = match (segment.start_sec, segment.end_sec) {
                (Some(start), Some(end)) => {
                    let span = (end - start).max(0.0);
                    let w_start = start + span * (index as f64) / n;
                    let w_end = if index + 1 == words.len() {
                        end
                    } else {
                        start + span * ((index + 1) as f64) / n
                    };
                    (Some(w_start), Some(w_end))
                }
                _ => (None, None),
            };
            out.push(TranscriptionSegment {
                start_sec,
                end_sec,
                text: (*word).to_owned(),
                speaker: None,
                confidence: segment.confidence,
            });
        }
    }
    Ok(out)
}

/// Regroup per-word segments into runs of up to `max_len` characters, joining
/// with single spaces and spanning each group's first/last word bounds.
fn group_word_segments_by_len(
    segments: &[TranscriptionSegment],
    max_len: u32,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<TranscriptionSegment>> {
    let limit = max_len as usize;
    let mut grouped = Vec::new();
    let mut current_text = String::new();
    let mut current_start: Option<f64> = None;
    let mut current_end: Option<f64> = None;
    let mut confidence_sum = 0.0;
    let mut confidence_count = 0u64;

    let flush = |grouped: &mut Vec<TranscriptionSegment>,
                 text: &mut String,
                 start: &mut Option<f64>,
                 end: &mut Option<f64>,
                 conf_sum: &mut f64,
                 conf_count: &mut u64| {
        if text.is_empty() {
            return;
        }
        let confidence = if *conf_count > 0 {
            Some(*conf_sum / *conf_count as f64)
        } else {
            None
        };
        grouped.push(TranscriptionSegment {
            start_sec: *start,
            end_sec: *end,
            text: std::mem::take(text),
            speaker: None,
            confidence,
        });
        *start = None;
        *end = None;
        *conf_sum = 0.0;
        *conf_count = 0;
    };

    for segment in segments {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        let word = segment.text.trim();
        if word.is_empty() {
            continue;
        }

        let word_len = word.chars().count();
        let extra_len = if current_text.is_empty() {
            word_len
        } else {
            1 + word_len
        };

        if !current_text.is_empty() && current_text.chars().count() + extra_len > limit {
            flush(
                &mut grouped,
                &mut current_text,
                &mut current_start,
                &mut current_end,
                &mut confidence_sum,
                &mut confidence_count,
            );
        }

        if current_text.is_empty() {
            current_start = segment.start_sec;
        } else {
            current_text.push(' ');
        }
        current_text.push_str(word);
        current_end = segment.end_sec;
        if let Some(conf) = segment.confidence {
            if conf.is_finite() {
                confidence_sum += conf;
                confidence_count += 1;
            }
        }
    }

    flush(
        &mut grouped,
        &mut current_text,
        &mut current_start,
        &mut current_end,
        &mut confidence_sum,
        &mut confidence_count,
    );

    Ok(grouped)
}

/// Apply the final segment-shaping policy: trim text, clear timestamps under
/// `no_timestamps`, and clamp confidence to `[0, 1]`.
fn finalize_segments(
    segments: &[TranscriptionSegment],
    no_timestamps: bool,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<Vec<TranscriptionSegment>> {
    let mut out = Vec::with_capacity(segments.len());
    for seg in segments {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        out.push(TranscriptionSegment {
            start_sec: if no_timestamps { None } else { seg.start_sec },
            end_sec: if no_timestamps { None } else { seg.end_sec },
            text: seg.text.trim().to_owned(),
            speaker: None,
            confidence: seg
                .confidence
                .map(|c| if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 }),
        });
    }
    Ok(out)
}

/// Read a normalized 16 kHz mono PCM16 WAV into f32 mono samples.
///
/// Delegates to the engine's RIFF reader so production and the gated e2e tests
/// share one decoder. The pipeline guarantees this file is already 16 kHz mono
/// PCM16.
fn read_normalized_wav(path: &Path) -> FwResult<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    decode::read_wav_16k_mono(&bytes)
}

/// The honest raw-output metadata JSON for a real-inference run.
#[allow(clippy::too_many_arguments)]
fn raw_output_json(
    spec: &str,
    model_path: &Path,
    version_tag: String,
    windows: &[decode::WindowStats],
    word_mode: WordTimestampMode,
    split_on_word: bool,
    silence: bool,
    used_dtw: bool,
) -> Value {
    let word_timestamps = if used_dtw {
        // Real cross-attention DTW alignment (bd-rjsx).
        "dtw"
    } else if word_mode != WordTimestampMode::None || split_on_word {
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
    json!({
        "engine": "whisper.cpp-native",
        "schema_version": SCHEMA_VERSION,
        "in_process": true,
        "implementation": "real-inference",
        "silence": silence,
        "model": spec,
        "model_path": model_path.display().to_string(),
        "model_version_tag": version_tag,
        "windows": windows_json,
        "word_timestamps": word_timestamps,
    })
}

/// Build the empty-but-valid result for a pure-silence clip, taken **without
/// loading the model** (the energy pre-gate already proved there is nothing to
/// transcribe — saves a potentially multi-GB model load).
fn silence_result(
    request: &TranscribeRequest,
    spec: &str,
    duration_ms: u64,
) -> TranscriptionResult {
    TranscriptionResult {
        backend: BackendKind::WhisperCpp,
        transcript: String::new(),
        language: request.language.clone(),
        segments: Vec::new(),
        acceleration: None,
        raw_output: json!({
            "engine": "whisper.cpp-native",
            "schema_version": SCHEMA_VERSION,
            "in_process": true,
            "implementation": "real-inference",
            "silence": true,
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
    use std::sync::Mutex;
    use std::time::Duration;

    use crate::backend::Engine;
    use crate::model::{
        BackendKind, BackendParams, InputSource, TranscribeRequest, TranscriptionSegment,
        WordTimestampParams,
    };
    use crate::orchestrator::CancellationToken;

    use super::*;

    fn native_request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::WhisperCpp,
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

    /// A real-shaped engine segment (timed, with text), standing in for what
    /// the native decoder produces — NOT a canned phrase.
    fn seg(start: f64, end: f64, text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: Some(start),
            end_sec: Some(end),
            text: text.to_owned(),
            speaker: None,
            confidence: Some(0.9),
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
        let engine = super::super::WhisperCppNativeEngine;
        assert_eq!(engine.name(), "whisper.cpp-native");
    }

    #[test]
    fn native_engine_kind_matches_bridge_adapter() {
        let native = super::super::WhisperCppNativeEngine;
        let bridge = super::super::WhisperCppEngine;
        assert_eq!(native.kind(), bridge.kind());
        assert_eq!(native.kind(), BackendKind::WhisperCpp);
    }

    #[test]
    fn native_engine_name_distinct_from_bridge() {
        let native = super::super::WhisperCppNativeEngine;
        let bridge = super::super::WhisperCppEngine;
        assert_ne!(native.name(), bridge.name());
        assert!(native.name().contains("native"));
    }

    // ── Model resolution / availability ───────────────────────────────────

    #[test]
    fn run_without_model_or_default_is_backend_unavailable() {
        let mut req = native_request();
        req.model = None;
        // If the operator has a default model configured in this environment we
        // cannot assert the unavailable path; only assert when truly unset.
        if native_engine::default_model_spec().is_some() {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("a.wav");
        // A non-silent clip so we get past the silence pre-gate to the model
        // resolution error.
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
        let mut req = native_request();
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

    #[test]
    fn is_available_is_memoized_within_ttl() {
        // The memoized and uncached probes must agree, and repeated calls within
        // the TTL must be served from the cache (proven by a populated cache
        // entry after the first call). We do not mutate env (forbidden under
        // edition 2024), so we assert consistency rather than a flip.
        reset_availability_cache();
        let direct = is_available_uncached();
        let memoized = is_available();
        assert_eq!(
            direct, memoized,
            "memoized availability must match the uncached probe"
        );
        // A cache entry now exists and a second call returns the same value.
        {
            let guard = availability_cache().lock().expect("lock");
            assert!(guard.is_some(), "first probe must populate the cache");
            assert_eq!(guard.expect("entry").1, memoized);
        }
        assert_eq!(
            is_available(),
            memoized,
            "second probe within TTL must return the cached value"
        );
        // The reset helper clears the cache for the next test.
        reset_availability_cache();
        assert!(
            availability_cache().lock().expect("lock").is_none(),
            "reset must clear the cache"
        );
    }

    // ── Silence pre-gate ──────────────────────────────────────────────────

    #[test]
    fn run_pure_silence_returns_empty_without_loading_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = dir.path().join("silence.wav");
        write_pcm16_mono_wav(&wav, 16_000, &vec![0i16; 16_000]);

        let req = native_request();
        let start = std::time::Instant::now();
        let result = run(&req, &wav, dir.path(), Duration::from_secs(5), None)
            .expect("silence run should succeed");
        let elapsed = start.elapsed();

        assert!(result.segments.is_empty(), "silence => no segments");
        assert!(result.transcript.is_empty(), "silence => empty transcript");
        assert_eq!(result.raw_output["silence"].as_bool(), Some(true));
        assert_eq!(result.raw_output["model_loaded"].as_bool(), Some(false));
        assert_eq!(
            result.raw_output["schema_version"].as_str(),
            Some(SCHEMA_VERSION)
        );
        // The pre-gate must be cheap: no multi-GB model load happened.
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

        let req = native_request();
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

    // ── Word-timestamp helpers operate on REAL-shaped segments ────────────

    #[test]
    fn explode_segments_to_words_interpolates_within_bounds() {
        let segments = vec![seg(0.0, 4.0, "and so my fellow")];
        let words = explode_segments_to_words(&segments, None).expect("explode");
        assert_eq!(words.len(), 4);
        assert!(words.iter().all(|w| !w.text.contains(' ')));
        // First word starts at the segment start, last ends at the segment end.
        assert_eq!(words[0].start_sec, Some(0.0));
        assert_eq!(words[3].end_sec, Some(4.0));
        // Linear interpolation: 4 words over 4s => 1s each.
        assert!((words[1].start_sec.unwrap() - 1.0).abs() < 1e-9);
        assert!((words[2].start_sec.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn explode_segments_without_bounds_keeps_none() {
        let segments = vec![TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: "two words".to_owned(),
            speaker: None,
            confidence: Some(0.5),
        }];
        let words = explode_segments_to_words(&segments, None).expect("explode");
        assert_eq!(words.len(), 2);
        assert!(
            words
                .iter()
                .all(|w| w.start_sec.is_none() && w.end_sec.is_none())
        );
    }

    #[test]
    fn group_word_segments_by_len_groups_runs() {
        let words = explode_segments_to_words(
            &[seg(0.0, 8.0, "the quick brown fox jumps over the lazy dog")],
            None,
        )
        .expect("explode");
        let grouped = group_word_segments_by_len(&words, 10, None).expect("group");
        let texts: Vec<String> = grouped.iter().map(|s| s.text.clone()).collect();
        // max_len = 10 chars per group.
        assert_eq!(
            texts,
            vec!["the quick", "brown fox", "jumps over", "the lazy", "dog"]
        );
        // Group bounds span the constituent words.
        assert_eq!(grouped.first().unwrap().start_sec, Some(0.0));
        assert_eq!(grouped.last().unwrap().end_sec, Some(8.0));
    }

    #[test]
    fn group_word_segments_by_len_ignores_nan_confidence() {
        // One poisoned per-word confidence must not corrupt the grouped average:
        // the NaN is dropped from the mean, leaving a finite result.
        let words = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "alpha".to_owned(),
                speaker: None,
                confidence: Some(0.8),
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: "beta".to_owned(),
                speaker: None,
                confidence: Some(f64::NAN),
            },
            TranscriptionSegment {
                start_sec: Some(2.0),
                end_sec: Some(3.0),
                text: "gamma".to_owned(),
                speaker: None,
                confidence: Some(0.6),
            },
        ];
        let grouped = group_word_segments_by_len(&words, 100, None).expect("group");
        assert_eq!(grouped.len(), 1);
        let conf = grouped[0].confidence.expect("finite confidence");
        assert!(conf.is_finite(), "grouped confidence must be finite: {conf}");
        // Mean of the two finite confidences (0.8, 0.6), NaN excluded.
        assert!((conf - 0.7).abs() < 1e-9, "unexpected mean: {conf}");
    }

    #[test]
    fn build_segments_word_mode_splits_real_segments() {
        let engine_segments = vec![seg(0.0, 3.0, "hello there world")];
        let out = build_segments(
            &engine_segments,
            WordTimestampMode::Word,
            false,
            false,
            None,
        )
        .expect("build");
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|s| !s.text.contains(' ')));
    }

    #[test]
    fn finalize_segments_trims_clamps_and_clears_timestamps() {
        let segs = vec![
            TranscriptionSegment {
                start_sec: Some(0.5),
                end_sec: Some(1.5),
                text: "  hello world  ".to_owned(),
                speaker: None,
                confidence: Some(1.5),
            },
            TranscriptionSegment {
                start_sec: Some(2.0),
                end_sec: Some(3.0),
                text: "ok".to_owned(),
                speaker: None,
                confidence: Some(-0.2),
            },
        ];
        let kept = finalize_segments(&segs, false, None).expect("finalize");
        assert_eq!(kept[0].text, "hello world");
        assert_eq!(kept[0].confidence, Some(1.0));
        assert_eq!(kept[1].confidence, Some(0.0));

        let cleared = finalize_segments(&segs, true, None).expect("finalize");
        assert!(
            cleared
                .iter()
                .all(|s| s.start_sec.is_none() && s.end_sec.is_none())
        );
    }

    #[test]
    fn finalize_segments_sanitizes_nan_confidence() {
        // A NaN confidence must be sanitized to a safe finite value rather than
        // panicking under the nightly clamp-on-NaN behavior.
        let segs = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "hello".to_owned(),
            speaker: None,
            confidence: Some(f64::NAN),
        }];
        let out = finalize_segments(&segs, false, None).expect("finalize");
        assert_eq!(out[0].confidence, Some(0.0));
    }

    #[test]
    fn finalize_segments_cancellation_propagates() {
        let segs = vec![seg(0.0, 1.0, "x")];
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));
        let result = finalize_segments(&segs, false, Some(&token));
        assert!(matches!(result.unwrap_err(), FwError::Cancelled(_)));
    }

    #[test]
    fn word_timestamp_mode_mapping() {
        assert_eq!(word_timestamp_mode(None), WordTimestampMode::None);
        assert_eq!(
            word_timestamp_mode(Some(&WordTimestampParams {
                enabled: true,
                ..Default::default()
            })),
            WordTimestampMode::Word
        );
        assert_eq!(
            word_timestamp_mode(Some(&WordTimestampParams {
                max_len: Some(0),
                ..Default::default()
            })),
            WordTimestampMode::None
        );
        assert_eq!(
            word_timestamp_mode(Some(&WordTimestampParams {
                max_len: Some(10),
                ..Default::default()
            })),
            WordTimestampMode::MaxLen(10)
        );
    }

    #[test]
    fn raw_output_word_flag_is_interpolated_when_requested() {
        let json = raw_output_json(
            "tiny.en",
            Path::new("/models/ggml-tiny.en.bin"),
            "fw-native-v1+sha256:abc".to_owned(),
            &[],
            WordTimestampMode::Word,
            false,
            false,
            false,
        );
        assert_eq!(json["word_timestamps"].as_str(), Some("interpolated"));
        assert_eq!(json["implementation"].as_str(), Some("real-inference"));
        assert_eq!(json["schema_version"].as_str(), Some(SCHEMA_VERSION));
        assert_eq!(json["in_process"].as_bool(), Some(true));
    }

    #[test]
    fn raw_output_word_flag_is_dtw_when_dtw_used() {
        let json = raw_output_json(
            "tiny.en",
            Path::new("/models/ggml-tiny.en.bin"),
            "fw-native-v1+sha256:abc".to_owned(),
            &[],
            WordTimestampMode::Word,
            false,
            false,
            true, // used_dtw
        );
        assert_eq!(json["word_timestamps"].as_str(), Some("dtw"));
    }

    #[test]
    fn build_segments_dtw_uses_real_word_times() {
        let engine = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(2.0),
            text: " hello world".to_owned(),
            speaker: None,
            confidence: Some(0.9),
        }];
        let timings = vec![vec![
            WordTiming {
                text: "hello".to_owned(),
                start_sec: 0.3,
                end_sec: 0.9,
            },
            WordTiming {
                text: "world".to_owned(),
                start_sec: 0.9,
                end_sec: 1.8,
            },
        ]];
        let out =
            build_segments_dtw(&engine, &timings, WordTimestampMode::Word, false, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "hello");
        assert_eq!(out[0].start_sec, Some(0.3));
        assert_eq!(out[0].end_sec, Some(0.9));
        assert_eq!(out[1].text, "world");
        assert_eq!(out[1].start_sec, Some(0.9));
        // Strictly monotonic, non-overlapping.
        assert!(out[0].end_sec <= out[1].start_sec);
    }

    #[test]
    fn build_segments_dtw_falls_back_to_interpolation_for_untimed_segment() {
        // Segment 1 has DTW words; segment 0 has none (empty inner vec) → it is
        // interpolated, so no words are dropped.
        let engine = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: " a b".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: " c".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        let timings = vec![
            Vec::new(),
            vec![WordTiming {
                text: "c".to_owned(),
                start_sec: 1.2,
                end_sec: 1.9,
            }],
        ];
        let out =
            build_segments_dtw(&engine, &timings, WordTimestampMode::Word, false, None).unwrap();
        // "a", "b" (interpolated) + "c" (dtw) = 3 words.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "a");
        assert_eq!(out[1].text, "b");
        assert_eq!(out[2].text, "c");
        assert_eq!(out[2].start_sec, Some(1.2));
    }

    // ── Gated end-to-end against the real tiny.en model + jfk.wav ─────────

    /// The exact reference transcript from
    /// `tests/fixtures/native/jfk_tiny_reference.json` (joined, trimmed).
    const JFK_REFERENCE: &str = "And so my fellow Americans ask not what your country can do for \
        you ask what you can do for your country.";

    fn tiny_en_available() -> bool {
        native_engine::find_model_file("tiny.en").is_some()
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_through_engine_trait() {
        if !tiny_en_available() {
            eprintln!("SKIP gated_e2e_jfk: tiny.en model missing");
            return;
        }
        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let mut req = native_request();
        req.model = Some("tiny.en".to_owned());
        req.language = None;

        let engine = super::super::WhisperCppNativeEngine;
        let result = crate::backend::Engine::run(
            &engine,
            &req,
            Path::new(wav),
            Path::new("."),
            Duration::from_secs(120),
            None,
        )
        .expect("e2e run");

        assert_eq!(result.transcript.trim(), JFK_REFERENCE);
        assert_eq!(
            result.raw_output["schema_version"].as_str(),
            Some(SCHEMA_VERSION)
        );
        assert_eq!(
            result.raw_output["implementation"].as_str(),
            Some("real-inference")
        );
        assert_eq!(result.raw_output["silence"].as_bool(), Some(false));
        assert!(
            result.raw_output["windows"]
                .as_array()
                .is_some_and(|w| !w.is_empty()),
            "windows stats populated"
        );
        assert!(result.segments.len() >= 2, "expected >= 2 segments");
    }

    #[test]
    fn gated_e2e_streaming_replays_all_segments() {
        if !tiny_en_available() {
            eprintln!("SKIP gated_e2e_streaming: tiny.en model missing");
            return;
        }
        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let mut req = native_request();
        req.model = Some("tiny.en".to_owned());
        req.language = None;

        let emitted = Mutex::new(Vec::new());
        let result = run_streaming(
            &req,
            Path::new(wav),
            Path::new("."),
            Duration::from_secs(120),
            None,
            &|s| emitted.lock().expect("lock").push(s),
        )
        .expect("streaming e2e");

        let emitted = emitted.lock().expect("lock");
        assert_eq!(emitted.len(), result.segments.len());
        assert_eq!(
            result.raw_output["streaming_emitted_segments"].as_u64(),
            Some(result.segments.len() as u64)
        );
    }
}
