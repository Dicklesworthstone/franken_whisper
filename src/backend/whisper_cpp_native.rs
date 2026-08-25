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
//! When attention-DTW timings are available, raw decoder-grid observations are
//! normalized once into positive, non-overlapping half-open projection units
//! before they can reach diarization. The raw-output metadata distinguishes
//! real DTW units from the explicit segment interpolation fallback.

use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::conformance::DTW_PROJECTION_SCHEMA_VERSION;
use crate::diarization_projection::{
    CANONICAL_PROJECTION_EPSILON_SEC, CANONICAL_PROJECTION_MIN_DURATION_SEC,
    CanonicalProjectionUnit, ProjectionUnitProvenance, normalize_dtw_projection_units,
};
use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, DiarizationEngine, TranscribeRequest, TranscriptionResult, TranscriptionSegment,
    WordTimestampParams,
};
use crate::native_engine::dtw::WordTiming;
use crate::native_engine::{self, NativeWhisperModel, WhisperHParams, decode};

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct DtwProjectionReport {
    input_engine_segments: usize,
    input_timed_segments: usize,
    canonical_units: usize,
    output_segments: usize,
    decoder_word_units: usize,
    interpolated_fallback_units: usize,
    segment_geometry_fallback_units: usize,
    interpolated_fallback_segments: usize,
    segment_geometry_fallback_segments: usize,
    clamped_units: usize,
    expanded_units: usize,
    timestamps_suppressed: bool,
    word_aligned_safe: bool,
}

impl DtwProjectionReport {
    fn fallback_reasons(&self) -> Vec<&'static str> {
        let mut reasons = Vec::with_capacity(3);
        if self.interpolated_fallback_segments > 0 {
            reasons.push("missing_decoder_word_timestamps");
        }
        if self.segment_geometry_fallback_segments > 0 {
            reasons.push("insufficient_parent_duration_for_millisecond_word_units");
        }
        if self.timestamps_suppressed {
            reasons.push("timestamps_suppressed_by_request");
        }
        reasons
    }

    fn to_json(&self) -> Value {
        json!({
            "schema_version": DTW_PROJECTION_SCHEMA_VERSION,
            "unit": "seconds",
            "interval_semantics": "half_open",
            "timestamp_epsilon_sec": CANONICAL_PROJECTION_EPSILON_SEC,
            "minimum_duration_sec": CANONICAL_PROJECTION_MIN_DURATION_SEC,
            "input_engine_segments": self.input_engine_segments,
            "input_timed_segments": self.input_timed_segments,
            "canonical_units": self.canonical_units,
            "output_segments": self.output_segments,
            "decoder_word_units": self.decoder_word_units,
            "interpolated_fallback_units": self.interpolated_fallback_units,
            "segment_geometry_fallback_units": self.segment_geometry_fallback_units,
            "interpolated_fallback_segments": self.interpolated_fallback_segments,
            "segment_geometry_fallback_segments": self.segment_geometry_fallback_segments,
            "clamped_units": self.clamped_units,
            "expanded_units": self.expanded_units,
            "fallback_reasons": self.fallback_reasons(),
            "word_aligned_safe": self.word_aligned_safe,
            "supported_provenance": ProjectionUnitProvenance::supported_labels(),
        })
    }
}

#[derive(Debug, Clone)]
struct DtwSegmentsOutcome {
    segments: Vec<TranscriptionSegment>,
    #[cfg(test)]
    canonical_units: Vec<CanonicalProjectionUnit>,
    report: DtwProjectionReport,
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
/// 1. If `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL` is set, report the validity of
///    that exact operator choice. An invalid explicit default never falls
///    through to a different model.
/// 2. Otherwise, use the native engine's canonical resolver, which prefers the
///    hash-pinned release package and then considers valid local GGML files.
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
    native_engine::configured_or_release_model_available()
}

/// Number of leading bytes covering the ggml magic plus the eleven `i32`
/// hparams (`4 + 11 * 4`). Duplicated from the parser so availability sniffing
/// needs no private engine internals.
const HEADER_SNIFF_LEN: usize = 48;

/// ggml file magic (`"ggml"` as a little-endian `u32`).
const GGML_MAGIC: u32 = 0x6767_6d6c;

/// Sniff `path`'s 48-byte ggml header into [`WhisperHParams`], or `None` when the
/// file is unreadable, too short, carries the wrong magic, or declares an
/// unsupported (non-dense) `ftype`.
///
/// Header-only: 48 bytes, no weight load, no network, never panics. The eleven
/// `i32` hparams follow the magic in declaration order.
fn header_hparams(path: &Path) -> Option<WhisperHParams> {
    use std::io::Read as _;
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; HEADER_SNIFF_LEN];
    file.read_exact(&mut buf).ok()?;
    if u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) != GGML_MAGIC {
        return None;
    }
    let field = |i: usize| -> i32 {
        let o = 4 + i * 4;
        i32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
    };
    let ftype = field(10);
    if ftype != 0 && ftype != 1 {
        return None;
    }
    Some(WhisperHParams {
        n_vocab: field(0),
        n_audio_ctx: field(1),
        n_audio_state: field(2),
        n_audio_head: field(3),
        n_audio_layer: field(4),
        n_text_ctx: field(5),
        n_text_state: field(6),
        n_text_head: field(7),
        n_text_layer: field(8),
        n_mels: field(9),
        ftype,
    })
}

/// Machine-checked answers to "what can the native engine actually do right
/// now?" — the ground truth behind the three native engines' reported
/// [`EngineCapabilities`](crate::model::EngineCapabilities) (bd-0522).
///
/// Every field is a probe of this build, this machine, and the model the engine
/// would load if asked, rather than a hand-maintained constant that drifts from
/// reality. Cheap enough to call per health check: one 48-byte header read, no
/// weight load, no network, never panics.
///
/// Model *presence* is deliberately absent here: [`is_available`] already
/// answers it, and a capability probe that disagreed with it would be the very
/// drift this type exists to prevent. With no model both model-derived fields
/// are `false`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NativeProbe {
    /// The resolved model carries the multilingual vocabulary, so the decoder's
    /// `translate` task token exists. English-only models (`*.en`) cannot
    /// translate, whatever the CLI accepts.
    pub multilingual: bool,
    /// DTW alignment heads resolve for the resolved model, so word timestamps
    /// come from real cross-attention alignment.
    pub word_timestamps: bool,
    /// A real GPU encoder path is compiled in and enabled on this machine.
    pub gpu: bool,
}

impl NativeProbe {
    /// The honest capability set when no model resolves: nothing model-derived
    /// is supported. `gpu` still reflects the build and machine.
    fn without_model(gpu: bool) -> Self {
        Self {
            multilingual: false,
            word_timestamps: false,
            gpu,
        }
    }
}

/// Probe what the native engine can actually do (see [`NativeProbe`]).
///
/// Selects the same model specification [`run`] would:
/// `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL` first, otherwise the default release
/// package. Discovery validates only the bounded header; execution separately
/// authenticates the complete release artifact before loading tensors.
pub(crate) fn capability_probe() -> NativeProbe {
    let gpu = native_engine::encoder::gpu_encoder_available();

    // Mirror execution selection while retaining discovery's header-only
    // authority. Full release-package authentication remains mandatory in
    // `effective_model_spec` immediately before tensors are loaded.
    let spec = native_engine::default_model_spec().unwrap_or_else(|| "large-v3-turbo".to_owned());
    let path = native_engine::model_probe_path(&spec);

    let Some(path) = path else {
        return NativeProbe::without_model(gpu);
    };
    let Some(hparams) = header_hparams(&path) else {
        return NativeProbe::without_model(gpu);
    };

    // Fall back to the file stem as the preset hint when no spec was configured;
    // `alignment_heads` normalizes it and drops back to the openai "top half of
    // layers" rule for anything it does not recognize.
    let hint = Some(spec);

    NativeProbe {
        multilingual: hparams.is_multilingual(),
        word_timestamps: !native_engine::dtw::alignment_heads(&hparams, hint.as_deref()).is_empty(),
        gpu,
    }
}

/// Resolve the effective model spec for a request, or a [`FwError`] explaining
/// how to provision one.
///
/// Precedence: `request.model` (when set), then
/// `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`, then the authenticated default
/// release package. If none resolves, returns an actionable
/// [`FwError::BackendUnavailable`].
fn effective_model_spec(request: &TranscribeRequest) -> FwResult<String> {
    if let Some(model) = request.model.clone().filter(|m| !m.is_empty()) {
        return Ok(model);
    }
    native_engine::configured_or_release_model_spec().map_err(|error| {
        FwError::BackendUnavailable(format!(
            "native whisper.cpp engine has no usable local model: pass --model, or set \
             $FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL to a model short-name or path. {error}"
        ))
    })
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
        // Initial prompt (whisper `--prompt`): the engine tokenizes it and seeds
        // the first window's carried context. Empty → no-op.
        initial_prompt: request
            .backend_params
            .prompt
            .clone()
            .filter(|p| !p.is_empty()),
        // Beam width (whisper `--beam-size`): the engine beam-searches temp-0
        // windows when > 1; the decoder clamps to [1, 8]. None → greedy.
        beam_size: request
            .backend_params
            .decoding
            .as_ref()
            .and_then(|d| d.beam_size)
            .map(|n| n as usize),
        // Suppress non-speech tokens (whisper `--suppress-nst`) for cleaner text.
        suppress_nst: request.backend_params.suppress_nst,
        // Max carried context (whisper `--max-context`); 0 disables prompt carry.
        max_context: request
            .backend_params
            .decoding
            .as_ref()
            .and_then(|d| d.max_context),
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
        // Streaming encoder-context policy stays at the byte-exact default for
        // bridge backends (the knob is per-request on the engine; the live
        // driver opts in, not the batch bridges).
        audio_ctx: decode::AudioCtxPolicy::Full,
        bypass_transcript_cache: false,
        record_token_attn: false,
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

    // Requested audio window (bd-vgod): `--offset-ms` / `--duration-ms` slice
    // the normalized PCM before decode, so wall-clock scales with the slice.
    let audio_window = requested_audio_window(request);

    // Silence pre-gate: the energy analyzer is cheap; loading a (potentially
    // multi-GB) model is not. Pure-silence clips skip the load entirely.
    // A windowed request skips the whole-file pre-gate: the analyzer scans
    // the full clip, so its verdict would not describe the requested slice.
    if audio_window.is_none() {
        let analysis = analyze_wav(normalized_wav, request.backend_params.duration_ms).ok();
        if let Some(analysis) = analysis.as_ref()
            && analysis.active_regions.is_empty()
        {
            return Ok(silence_result(request, &spec, analysis.duration_ms));
        }
    }

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Resolve + load the model (cached). A resolution miss here is unavailable.
    let model_path = native_engine::resolve_model(&spec)
        .map_err(|e| FwError::BackendUnavailable(e.to_string()))?;
    let model = NativeWhisperModel::load(&model_path)?;

    // Read the normalized WAV to f32 mono samples, then apply the requested
    // window. Timestamps are shifted back into the source timebase after
    // decode so diarization/VAD/alignment stay consistent with the full clip.
    let full_samples = read_normalized_wav(normalized_wav)?;
    let (samples, window_offset_ms) = match audio_window {
        Some((offset_ms, duration_ms)) => {
            let (start, end) = window_sample_bounds(full_samples.len(), offset_ms, duration_ms);
            if start >= end {
                // Offset at/past EOF: an empty-but-valid result, honestly
                // tagged with the requested (empty) window, beats a decode
                // error for a region probe.
                let mut result = silence_result(request, &spec, 0);
                if let Value::Object(map) = &mut result.raw_output {
                    map.insert(
                        "audio_window".to_owned(),
                        json!({
                            "offset_ms": offset_ms,
                            "duration_ms": duration_ms,
                            "timebase": "source",
                            "empty_slice": true,
                        }),
                    );
                }
                return Ok(result);
            }
            (full_samples[start..end].to_vec(), offset_ms)
        }
        None => (full_samples, 0),
    };

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Word-timestamp policy: decide whether to ask the engine for real
    // attention-DTW word times. DTW runs when per-word output is requested
    // (a word/maxlen split or `split_on_word`) — the engine then records
    // cross-attention and aligns each word to audio frames (bd-rjsx).
    let word_mode = word_timestamp_mode(request.backend_params.word_timestamps.as_ref());
    let native_acoustic_diarization = native_acoustic_diarization_requested(request);
    let want_words = word_mode != WordTimestampMode::None
        || request.backend_params.split_on_word
        || native_acoustic_diarization;
    // DTW is only meaningful when we keep per-segment timestamps.
    let want_dtw = want_words && !request.backend_params.no_timestamps;

    let params = decode_params(request, want_dtw, &spec);
    let checkpoint = checkpoint_for(token);
    let mut output = model.transcribe(&samples, &params, &checkpoint)?;

    // Shift every emitted timestamp back into the source-file timebase so a
    // windowed run stays aligned with diarization, VAD, and alignment stages
    // that operate on the full normalized clip.
    if window_offset_ms > 0 {
        shift_decode_output(&mut output, window_offset_ms as f64 / 1000.0);
    }

    // Prefer real DTW word timings when the engine produced them; otherwise fall
    // back to the linear-interpolation word split (keeping its existing tag).
    let dtw_words = output
        .word_timings
        .as_ref()
        .filter(|w| w.iter().any(|seg| !seg.is_empty()));
    let (segments, dtw_projection) = if let Some(word_timings) = dtw_words {
        let outcome = build_segments_dtw(
            &output.segments,
            word_timings,
            word_mode,
            request.backend_params.no_timestamps,
            token,
        )?;
        (outcome.segments, Some(outcome.report))
    } else {
        (
            build_segments(
                &output.segments,
                word_mode,
                request.backend_params.split_on_word,
                request.backend_params.no_timestamps,
                token,
            )?,
            None,
        )
    };
    let transcript = super::transcript_from_segments(&segments);
    let language = output.language.clone().or_else(|| request.language.clone());

    let t_tag = std::time::Instant::now();
    let version_tag = model.version_tag();
    crate::native_engine::perf_span("version_tag", t_tag.elapsed().as_secs_f64() * 1e3, "");
    crate::native_engine::perf_span("backend_run", t_backend.elapsed().as_secs_f64() * 1e3, "");
    let mut raw_output = raw_output_json(
        &spec,
        &model_path,
        version_tag,
        native_engine::encoder_int8_effective_policy_decision(&model.loaded().hparams),
        &output.windows,
        &output.dropped_windows,
        &output.work,
        word_mode,
        request.backend_params.split_on_word,
        false,
        dtw_projection.as_ref(),
    );
    if let (Value::Object(map), Some((offset_ms, duration_ms))) = (&mut raw_output, audio_window) {
        map.insert(
            "audio_window".to_owned(),
            json!({
                "offset_ms": offset_ms,
                "duration_ms": duration_ms,
                "timebase": "source",
            }),
        );
    }
    // Additive native-v2 route provenance: which encoder route actually ran
    // (gpu_fused vs cpu:<decline reason>). A silent GPU->CPU fallback is
    // invisible without this.
    if let Value::Object(map) = &mut raw_output {
        map.insert(
            "encoder_route".to_owned(),
            json!(native_engine::encoder::last_encoder_route()),
        );
    }

    Ok(TranscriptionResult {
        backend: BackendKind::WhisperCpp,
        transcript,
        language,
        segments,
        acceleration: None,
        diarization: None,
        raw_output,
        artifact_paths: Vec::new(),
    })
}

fn native_acoustic_diarization_requested(request: &TranscribeRequest) -> bool {
    request.diarize
        && request
            .backend_params
            .acoustic_diarization
            .as_ref()
            .is_none_or(|config| {
                matches!(
                    config.engine,
                    DiarizationEngine::Auto | DiarizationEngine::Acoustic
                )
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

/// Build the final segment list from real **DTW-aligned** word timings.
///
/// Decoder-grid observations are not projection units: quantization can place
/// a terminal word at `[t, t]`. This adapter validates the raw geometry and
/// normalizes it exactly once into positive half-open intervals. A segment
/// without DTW words uses explicit interpolation. Geometry too short to retain
/// distinct millisecond word boundaries falls back conservatively to one
/// parent-segment unit.
fn build_segments_dtw(
    engine_segments: &[TranscriptionSegment],
    word_timings: &[Vec<WordTiming>],
    word_mode: WordTimestampMode,
    no_timestamps: bool,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<DtwSegmentsOutcome> {
    let normalized = normalize_dtw_projection_units(engine_segments, word_timings, || {
        token.map_or(Ok(()), crate::orchestrator::CancellationToken::checkpoint)
    })?;
    let mut report = DtwProjectionReport {
        input_engine_segments: engine_segments.len(),
        input_timed_segments: normalized.input_timed_segments,
        canonical_units: normalized.segments.len(),
        output_segments: 0,
        decoder_word_units: normalized.decoder_word_units,
        interpolated_fallback_units: normalized.interpolated_fallback_units,
        segment_geometry_fallback_units: normalized.segment_geometry_fallback_units,
        interpolated_fallback_segments: normalized.interpolated_fallback_segments,
        segment_geometry_fallback_segments: normalized.segment_geometry_fallback_segments,
        clamped_units: normalized.clamped_units,
        expanded_units: normalized.expanded_units,
        timestamps_suppressed: no_timestamps,
        word_aligned_safe: !no_timestamps && normalized.word_aligned_safe,
    };
    let words = normalized.segments;
    let segments = match word_mode {
        WordTimestampMode::MaxLen(max_len) if max_len > 1 => {
            group_word_segments_by_len(&words, max_len, token)?
        }
        _ => words,
    };
    let segments = finalize_segments(&segments, no_timestamps, token)?;
    report.output_segments = segments.len();

    Ok(DtwSegmentsOutcome {
        segments,
        #[cfg(test)]
        canonical_units: normalized.canonical_units,
        report,
    })
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
        if let Some(conf) = segment.confidence
            && conf.is_finite()
        {
            confidence_sum += conf;
            confidence_count += 1;
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
            confidence: seg.confidence.map(|c| {
                if c.is_finite() {
                    c.clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }),
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

/// The requested `--offset-ms` / `--duration-ms` audio window, if any.
///
/// Returns `None` when neither flag is set (or both are zero-effect), so the
/// unwindowed path stays byte-identical to the pre-window behavior.
fn requested_audio_window(request: &TranscribeRequest) -> Option<(u64, Option<u64>)> {
    let offset_ms = request.backend_params.offset_ms.unwrap_or(0);
    let duration_ms = request.backend_params.duration_ms.filter(|d| *d > 0);
    if offset_ms == 0 && duration_ms.is_none() {
        None
    } else {
        Some((offset_ms, duration_ms))
    }
}

/// Convert a ms-domain window into clamped sample bounds over 16 kHz PCM.
fn window_sample_bounds(len: usize, offset_ms: u64, duration_ms: Option<u64>) -> (usize, usize) {
    const SAMPLES_PER_MS: u64 = (native_engine::mel::SAMPLE_RATE as u64) / 1000;
    let start = usize::try_from(offset_ms.saturating_mul(SAMPLES_PER_MS))
        .unwrap_or(usize::MAX)
        .min(len);
    let end = match duration_ms {
        Some(duration_ms) => usize::try_from(
            offset_ms
                .saturating_add(duration_ms)
                .saturating_mul(SAMPLES_PER_MS),
        )
        .unwrap_or(usize::MAX)
        .min(len),
        None => len,
    };
    (start, end.max(start))
}

/// Shift every emitted timestamp in a decode output by a uniform offset,
/// keeping windowed runs in the source-file timebase.
fn shift_decode_output(output: &mut decode::DecodeOutput, offset_sec: f64) {
    for segment in &mut output.segments {
        if let Some(start) = segment.start_sec.as_mut() {
            *start += offset_sec;
        }
        if let Some(end) = segment.end_sec.as_mut() {
            *end += offset_sec;
        }
    }
    if let Some(word_timings) = output.word_timings.as_mut() {
        for words in word_timings.iter_mut() {
            for word in words.iter_mut() {
                word.start_sec += offset_sec;
                word.end_sec += offset_sec;
            }
        }
    }
    for window in &mut output.windows {
        window.window_offset_sec += offset_sec;
    }
    for dropped in &mut output.dropped_windows {
        dropped.start_sec += offset_sec;
        dropped.end_sec += offset_sec;
    }
}

/// The honest raw-output metadata JSON for a real-inference run.
#[allow(clippy::too_many_arguments)]
fn raw_output_json(
    spec: &str,
    model_path: &Path,
    version_tag: String,
    encoder_policy: native_engine::EncoderInt8PolicyDecision,
    windows: &[decode::WindowStats],
    dropped_windows: &[decode::DroppedWindow],
    decode_work: &decode::DecodeWorkStats,
    word_mode: WordTimestampMode,
    split_on_word: bool,
    silence: bool,
    dtw_projection: Option<&DtwProjectionReport>,
) -> Value {
    let word_timestamps = if dtw_projection.is_some() {
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
    // Additive native-v2 fields (bd-nqzf): every discarded long-form window is
    // a real content gap and must be machine-addressable, not stderr-only.
    let dropped_windows_json: Vec<Value> = dropped_windows
        .iter()
        .map(|w| {
            json!({
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "reason": w.reason,
                "no_speech_prob": w.no_speech_prob,
                "avg_logprob": w.avg_logprob,
                "retried": w.retried,
            })
        })
        .collect();
    let mut output = json!({
        "engine": "whisper.cpp-native",
        "schema_version": SCHEMA_VERSION,
        "in_process": true,
        "implementation": "real-inference",
        "silence": silence,
        "model": spec,
        "model_path": model_path.display().to_string(),
        "model_version_tag": version_tag,
        "encoder_int8_policy": {
            "action": match encoder_policy.action {
                native_engine::EncoderInt8PolicyAction::F32Encoder => "f32",
                native_engine::EncoderInt8PolicyAction::QualitySafeInt8Encoder => "quality_safe_int8",
            },
            "reason": encoder_policy.reason,
            "calibration_id": encoder_policy.calibration_id,
            "corpus_wer_delta_budget": encoder_policy.corpus_wer_delta_budget,
            "quant_rel_rmse_budget": encoder_policy.quant_rel_rmse_budget,
        },
        "windows": windows_json,
        "dropped_windows": dropped_windows_json,
        "decode_work": {
            "prompt_reset_retries": decode_work.prompt_reset_retries,
            "temperature_fallback_retries": decode_work.temperature_fallback_retries,
        },
        "word_timestamps": word_timestamps,
    });
    if let (Value::Object(map), Some(report)) = (&mut output, dtw_projection) {
        map.insert("projection_timeline".to_owned(), report.to_json());
    }
    output
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
        diarization: None,
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
        BackendKind, BackendParams, DiarizationEngine, DiarizationRequest, InputSource,
        TranscribeRequest, TranscriptionSegment, WordTimestampParams,
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

    #[test]
    fn requested_audio_window_none_without_flags() {
        let request = native_request();
        assert!(requested_audio_window(&request).is_none());

        let mut zeroed = native_request();
        zeroed.backend_params.offset_ms = Some(0);
        zeroed.backend_params.duration_ms = Some(0);
        assert!(
            requested_audio_window(&zeroed).is_none(),
            "zero-effect flags must keep the unwindowed path byte-identical"
        );
    }

    #[test]
    fn requested_audio_window_captures_offset_and_duration() {
        let mut request = native_request();
        request.backend_params.offset_ms = Some(60_000);
        assert_eq!(requested_audio_window(&request), Some((60_000, None)));

        request.backend_params.duration_ms = Some(20_000);
        assert_eq!(
            requested_audio_window(&request),
            Some((60_000, Some(20_000)))
        );

        request.backend_params.offset_ms = None;
        assert_eq!(requested_audio_window(&request), Some((0, Some(20_000))));
    }

    #[test]
    fn window_sample_bounds_slices_and_clamps() {
        // 120 s of 16 kHz audio.
        let len = 120 * 16_000;
        assert_eq!(window_sample_bounds(len, 0, Some(20_000)), (0, 320_000));
        assert_eq!(
            window_sample_bounds(len, 60_000, None),
            (960_000, 1_920_000)
        );
        assert_eq!(
            window_sample_bounds(len, 60_000, Some(20_000)),
            (960_000, 1_280_000)
        );
        // Duration past EOF clamps to EOF.
        assert_eq!(
            window_sample_bounds(len, 110_000, Some(60_000)),
            (1_760_000, 1_920_000)
        );
        // Offset at/past EOF collapses to an empty slice.
        let (start, end) = window_sample_bounds(len, 120_000, Some(1_000));
        assert!(start >= end);
        let (start, end) = window_sample_bounds(len, u64::MAX, None);
        assert!(start >= end);
    }

    #[test]
    fn shift_decode_output_moves_all_timestamps_uniformly() {
        let mut output = decode::DecodeOutput {
            segments: vec![TranscriptionSegment {
                start_sec: Some(0.5),
                end_sec: Some(2.0),
                text: "hello".to_owned(),
                speaker: None,
                confidence: Some(0.9),
            }],
            language: Some("en".to_owned()),
            windows: vec![decode::WindowStats {
                avg_logprob: -0.1,
                no_speech_prob: 0.01,
                tokens: 3,
                window_offset_sec: 0.0,
                token_attn: Vec::new(),
            }],
            dropped_windows: vec![decode::DroppedWindow {
                start_sec: 30.0,
                end_sec: 60.0,
                reason: "window_closed_no_timestamp",
                no_speech_prob: 1e-9,
                avg_logprob: -0.1,
                retried: true,
            }],
            work: decode::DecodeWorkStats::default(),
            word_timings: Some(vec![vec![WordTiming {
                text: "hello".to_owned(),
                start_sec: 0.5,
                end_sec: 2.0,
            }]]),
        };
        shift_decode_output(&mut output, 60.0);
        assert_eq!(output.segments[0].start_sec, Some(60.5));
        assert_eq!(output.segments[0].end_sec, Some(62.0));
        let words = output.word_timings.as_ref().expect("word timings");
        assert!((words[0][0].start_sec - 60.5).abs() < 1e-9);
        assert!((words[0][0].end_sec - 62.0).abs() < 1e-9);
        assert!((output.windows[0].window_offset_sec - 60.0).abs() < 1e-9);
        assert!((output.dropped_windows[0].start_sec - 90.0).abs() < 1e-9);
        assert!((output.dropped_windows[0].end_sec - 120.0).abs() < 1e-9);
    }

    #[test]
    fn native_acoustic_diarization_forces_word_alignment_capture() {
        let mut request = native_request();
        assert!(!native_acoustic_diarization_requested(&request));

        request.diarize = true;
        assert!(
            native_acoustic_diarization_requested(&request),
            "absent typed config defaults to native acoustic diarization"
        );

        request.backend_params.acoustic_diarization = Some(DiarizationRequest {
            engine: DiarizationEngine::Acoustic,
            ..DiarizationRequest::default()
        });
        assert!(native_acoustic_diarization_requested(&request));

        request
            .backend_params
            .acoustic_diarization
            .as_mut()
            .expect("typed config")
            .engine = DiarizationEngine::External;
        assert!(!native_acoustic_diarization_requested(&request));
    }

    #[test]
    fn decode_params_maps_initial_prompt_from_request() {
        let mut request = native_request();
        // No prompt → the engine gets no initial prompt.
        assert_eq!(
            decode_params(&request, false, "tiny.en").initial_prompt,
            None
        );
        // A request prompt (whisper `--prompt`) flows through to the engine's
        // initial_prompt field, which the decoder tokenizes and seeds.
        request.backend_params.prompt = Some("medical terminology".to_owned());
        assert_eq!(
            decode_params(&request, false, "tiny.en")
                .initial_prompt
                .as_deref(),
            Some("medical terminology"),
        );
        // An empty prompt is treated as no prompt (byte-identical default).
        request.backend_params.prompt = Some(String::new());
        assert_eq!(
            decode_params(&request, false, "tiny.en").initial_prompt,
            None
        );
    }

    #[test]
    fn decode_params_maps_beam_size_from_request() {
        use crate::model::DecodingParams;
        let mut request = native_request();
        // No decoding params → None (the engine runs greedy = byte-identical).
        assert_eq!(decode_params(&request, false, "tiny.en").beam_size, None);
        // A request beam size (whisper `--beam-size`) flows to the engine.
        request.backend_params.decoding = Some(DecodingParams {
            beam_size: Some(5),
            ..DecodingParams::default()
        });
        assert_eq!(decode_params(&request, false, "tiny.en").beam_size, Some(5),);
    }

    #[test]
    fn decode_params_maps_suppress_nst_from_request() {
        let mut request = native_request();
        // Default off (whisper.cpp default = byte-identical).
        assert!(!decode_params(&request, false, "tiny.en").suppress_nst);
        // A --suppress-nst request reaches the engine's logit filter.
        request.backend_params.suppress_nst = true;
        assert!(decode_params(&request, false, "tiny.en").suppress_nst);
    }

    #[test]
    fn decode_params_maps_max_context_from_request() {
        use crate::model::DecodingParams;
        let mut request = native_request();
        // No decoding params → None (engine uses its n_text_ctx/2 default).
        assert_eq!(decode_params(&request, false, "tiny.en").max_context, None);
        // --max-context 0 (disable prompt carry) reaches the engine.
        request.backend_params.decoding = Some(DecodingParams {
            max_context: Some(0),
            ..DecodingParams::default()
        });
        assert_eq!(
            decode_params(&request, false, "tiny.en").max_context,
            Some(0)
        );
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
    fn run_without_any_configured_or_release_model_is_backend_unavailable() {
        let mut req = native_request();
        req.model = None;
        // This failure path exists only when neither an explicit default nor a
        // deterministically discoverable local model is present.
        if native_engine::configured_or_release_model_available() {
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
        let cancellation = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));

        let start = std::time::Instant::now();
        let result = run(
            &req,
            &wav,
            dir.path(),
            Duration::from_secs(1),
            Some(&cancellation),
        );
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
        assert!(
            conf.is_finite(),
            "grouped confidence must be finite: {conf}"
        );
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
        let cancellation = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));
        let result = finalize_segments(&segs, false, Some(&cancellation));
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

    fn f32_encoder_policy_fixture() -> native_engine::EncoderInt8PolicyDecision {
        native_engine::EncoderInt8PolicyDecision {
            action: native_engine::EncoderInt8PolicyAction::F32Encoder,
            reason: "unit_test_fixture",
            calibration_id: native_engine::ENCODER_INT8_CALIBRATION_ID,
            corpus_wer_delta_budget: 0.0,
            quant_rel_rmse_budget: 0.09,
        }
    }

    #[test]
    fn raw_output_word_flag_is_interpolated_when_requested() {
        let json = raw_output_json(
            "tiny.en",
            Path::new("/models/ggml-tiny.en.bin"),
            "fw-native-v1+sha256:abc".to_owned(),
            f32_encoder_policy_fixture(),
            &[],
            &[],
            &decode::DecodeWorkStats::default(),
            WordTimestampMode::Word,
            false,
            false,
            None,
        );
        assert_eq!(json["word_timestamps"].as_str(), Some("interpolated"));
        assert_eq!(json["implementation"].as_str(), Some("real-inference"));
        assert_eq!(json["schema_version"].as_str(), Some(SCHEMA_VERSION));
        assert_eq!(json["in_process"].as_bool(), Some(true));
        assert_eq!(json["dropped_windows"], json!([]));
        assert_eq!(json["decode_work"]["prompt_reset_retries"], json!(0));
    }

    #[test]
    fn raw_output_surfaces_dropped_windows_structurally() {
        let dropped = vec![decode::DroppedWindow {
            start_sec: 514.0,
            end_sec: 544.0,
            reason: "window_closed_no_timestamp",
            no_speech_prob: 5.6e-10,
            avg_logprob: -0.111,
            retried: true,
        }];
        let work = decode::DecodeWorkStats {
            prompt_reset_retries: 1,
            ..decode::DecodeWorkStats::default()
        };
        let json = raw_output_json(
            "large-v3-turbo",
            Path::new("/models/ggml-large-v3-turbo.bin"),
            "fw-native-v1+sha256:abc".to_owned(),
            f32_encoder_policy_fixture(),
            &[],
            &dropped,
            &work,
            WordTimestampMode::None,
            false,
            false,
            None,
        );
        let windows = json["dropped_windows"].as_array().expect("array");
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0]["start_sec"], json!(514.0));
        assert_eq!(windows[0]["end_sec"], json!(544.0));
        assert_eq!(windows[0]["reason"], json!("window_closed_no_timestamp"));
        assert_eq!(windows[0]["retried"], json!(true));
        assert_eq!(json["decode_work"]["prompt_reset_retries"], json!(1));
    }

    #[test]
    fn raw_output_word_flag_is_dtw_when_dtw_used() {
        let projection = DtwProjectionReport {
            input_engine_segments: 1,
            input_timed_segments: 1,
            canonical_units: 2,
            output_segments: 2,
            decoder_word_units: 2,
            interpolated_fallback_units: 0,
            segment_geometry_fallback_units: 0,
            interpolated_fallback_segments: 0,
            segment_geometry_fallback_segments: 0,
            clamped_units: 1,
            expanded_units: 1,
            timestamps_suppressed: false,
            word_aligned_safe: true,
        };
        let json = raw_output_json(
            "tiny.en",
            Path::new("/models/ggml-tiny.en.bin"),
            "fw-native-v1+sha256:abc".to_owned(),
            f32_encoder_policy_fixture(),
            &[],
            &[],
            &decode::DecodeWorkStats::default(),
            WordTimestampMode::Word,
            false,
            false,
            Some(&projection),
        );
        assert_eq!(json["word_timestamps"].as_str(), Some("dtw"));
        assert_eq!(
            json["projection_timeline"]["schema_version"].as_str(),
            Some(DTW_PROJECTION_SCHEMA_VERSION)
        );
        assert_eq!(
            json["projection_timeline"]["interval_semantics"].as_str(),
            Some("half_open")
        );
        assert_eq!(
            json["projection_timeline"]["word_aligned_safe"].as_bool(),
            Some(true)
        );
        assert_eq!(json["projection_timeline"]["fallback_reasons"], json!([]));
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
        let outcome =
            build_segments_dtw(&engine, &timings, WordTimestampMode::Word, false, None).unwrap();
        let out = outcome.segments;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "hello");
        assert_eq!(out[0].start_sec, Some(0.3));
        assert_eq!(out[0].end_sec, Some(0.9));
        assert_eq!(out[1].text, "world");
        assert_eq!(out[1].start_sec, Some(0.9));
        // Strictly monotonic, non-overlapping.
        assert!(out[0].end_sec <= out[1].start_sec);
        assert_eq!(outcome.report.decoder_word_units, 2);
        assert!(outcome.report.word_aligned_safe);
    }

    #[test]
    fn dtw_adapter_normalizes_zero_width_terminal_word_for_acoustic_projection() {
        let engine = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: " alpha beta".to_owned(),
            speaker: None,
            confidence: Some(0.9),
        }];
        let timings = vec![vec![
            WordTiming {
                text: "alpha".to_owned(),
                start_sec: 0.0,
                end_sec: 1.0,
            },
            WordTiming {
                text: "beta".to_owned(),
                start_sec: 1.0,
                end_sec: 1.0,
            },
        ]];

        let outcome = build_segments_dtw(&engine, &timings, WordTimestampMode::Word, false, None)
            .expect("valid quantized DTW geometry must normalize");
        let adapted = outcome.segments;
        assert_eq!(adapted.len(), 2);
        assert_eq!(adapted[0].start_sec, Some(0.0));
        assert_eq!(adapted[0].end_sec, Some(0.999));
        assert_eq!(adapted[1].start_sec, Some(0.999));
        assert_eq!(adapted[1].end_sec, Some(1.0));
        assert_eq!(outcome.report.expanded_units, 1);
        assert_eq!(outcome.report.clamped_units, 2);
        assert!(outcome.report.word_aligned_safe);
        assert_eq!(
            outcome.canonical_units[1].provenance,
            ProjectionUnitProvenance::DecoderWordTimestamp
        );

        let turns = vec![crate::model::DiarizationTurn {
            start_ms: 0,
            end_ms: 1_000,
            speaker_ref: Some("speaker_a".to_owned()),
            speaker_confidence: Some(0.9),
            change_confidence: Some(0.8),
            overlap_suspected: false,
            hard_hint_attributed: false,
        }];
        let projection =
            crate::diarization::project_diarization_onto_segments(&adapted, &turns, true)
                .expect("canonical DTW units must satisfy acoustic projection");
        assert_eq!(projection.segments.len(), 2);
        assert!(
            projection
                .segments
                .iter()
                .all(|segment| segment.speaker.as_deref() == Some("speaker_a"))
        );
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
        let outcome =
            build_segments_dtw(&engine, &timings, WordTimestampMode::Word, false, None).unwrap();
        let out = outcome.segments;
        // "a", "b" (interpolated) + "c" (dtw) = 3 words.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "a");
        assert_eq!(out[1].text, "b");
        assert_eq!(out[2].text, "c");
        assert_eq!(out[2].start_sec, Some(1.2));
        assert_eq!(outcome.report.interpolated_fallback_segments, 1);
        assert_eq!(
            outcome.report.fallback_reasons(),
            vec!["missing_decoder_word_timestamps"]
        );
        assert!(!outcome.report.word_aligned_safe);
    }

    #[test]
    fn dtw_projection_clamps_to_parent_and_preserves_punctuation_content() {
        let engine = vec![seg(1.0, 2.0, "hello ...")];
        let timings = vec![vec![
            WordTiming {
                text: "hello".to_owned(),
                start_sec: 0.5,
                end_sec: 1.4,
            },
            WordTiming {
                text: "...".to_owned(),
                start_sec: 2.0,
                end_sec: 2.5,
            },
        ]];
        let outcome = build_segments_dtw(&engine, &timings, WordTimestampMode::Word, false, None)
            .expect("end-of-audio clamping is a supported normalization");
        assert_eq!(
            outcome
                .segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hello", "..."]
        );
        assert_eq!(outcome.segments[0].start_sec, Some(1.0));
        assert_eq!(outcome.segments[1].end_sec, Some(2.0));
        assert_eq!(outcome.report.clamped_units, 2);
        assert!(outcome.report.word_aligned_safe);
    }

    #[test]
    fn dtw_projection_normalizes_sub_epsilon_overlap_but_rejects_material_overlap() {
        let engine = vec![seg(0.0, 1.0, "one two")];
        let harmless = vec![vec![
            WordTiming {
                text: "one".to_owned(),
                start_sec: 0.0,
                end_sec: 0.5,
            },
            WordTiming {
                text: "two".to_owned(),
                start_sec: 0.5 - CANONICAL_PROJECTION_EPSILON_SEC / 2.0,
                end_sec: 1.0,
            },
        ]];
        let outcome = build_segments_dtw(&engine, &harmless, WordTimestampMode::Word, false, None)
            .expect("floating-point adjacency noise must normalize");
        assert_eq!(outcome.segments[0].end_sec, outcome.segments[1].start_sec);

        let material = vec![vec![
            WordTiming {
                text: "one".to_owned(),
                start_sec: 0.0,
                end_sec: 0.5,
            },
            WordTiming {
                text: "two".to_owned(),
                start_sec: 0.5 - CANONICAL_PROJECTION_EPSILON_SEC * 2.0,
                end_sec: 1.0,
            },
        ]];
        let error = build_segments_dtw(&engine, &material, WordTimestampMode::Word, false, None)
            .expect_err("material overlap must fail closed");
        assert!(
            error.to_string().contains("FW-DTW-PROJECTION-WORD-OVERLAP"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn dtw_projection_rejects_non_finite_negative_reversed_and_reordered_words() {
        let engine = vec![seg(0.0, 2.0, "one two")];
        let cases = [
            (
                vec![WordTiming {
                    text: "one".to_owned(),
                    start_sec: f64::NAN,
                    end_sec: 1.0,
                }],
                "FW-DTW-PROJECTION-WORD-NONFINITE",
            ),
            (
                vec![WordTiming {
                    text: "one".to_owned(),
                    start_sec: 0.0,
                    end_sec: f64::INFINITY,
                }],
                "FW-DTW-PROJECTION-WORD-NONFINITE",
            ),
            (
                vec![WordTiming {
                    text: "one".to_owned(),
                    start_sec: -0.1,
                    end_sec: 0.5,
                }],
                "FW-DTW-PROJECTION-WORD-NEGATIVE",
            ),
            (
                vec![WordTiming {
                    text: "one".to_owned(),
                    start_sec: 1.0,
                    end_sec: 0.5,
                }],
                "FW-DTW-PROJECTION-WORD-REVERSED",
            ),
            (
                vec![
                    WordTiming {
                        text: "one".to_owned(),
                        start_sec: 1.0,
                        end_sec: 1.0,
                    },
                    WordTiming {
                        text: "two".to_owned(),
                        start_sec: 0.5,
                        end_sec: 0.5,
                    },
                ],
                "FW-DTW-PROJECTION-WORD-ORDER",
            ),
        ];
        for (timed, expected_code) in cases {
            let error = build_segments_dtw(&engine, &[timed], WordTimestampMode::Word, false, None)
                .expect_err("invalid raw word geometry must fail closed");
            assert!(
                error.to_string().contains(expected_code),
                "expected {expected_code}, got {error}"
            );
        }
    }

    #[test]
    fn dtw_projection_rejects_invalid_parent_and_extra_timing_vectors() {
        let paired = vec![seg(0.0, 1.0, "one")];
        let extra = vec![
            vec![WordTiming {
                text: "one".to_owned(),
                start_sec: 0.0,
                end_sec: 1.0,
            }],
            Vec::new(),
        ];
        let error = build_segments_dtw(&paired, &extra, WordTimestampMode::Word, false, None)
            .expect_err("extra timing vectors must not be silently ignored");
        assert!(
            error
                .to_string()
                .contains("FW-DTW-PROJECTION-EXTRA-SEGMENTS")
        );

        let unpaired = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: None,
            text: "one".to_owned(),
            speaker: None,
            confidence: None,
        }];
        let error = build_segments_dtw(
            &unpaired,
            &[Vec::new()],
            WordTimestampMode::Word,
            false,
            None,
        )
        .expect_err("unpaired parent timestamps must fail closed");
        assert!(error.to_string().contains("FW-DTW-PROJECTION-PARENT-PAIR"));

        let overlapping = vec![seg(0.0, 1.1, "one"), seg(1.0, 2.0, "two")];
        let error = build_segments_dtw(
            &overlapping,
            &[Vec::new(), Vec::new()],
            WordTimestampMode::Word,
            false,
            None,
        )
        .expect_err("material parent overlap must fail closed");
        assert!(
            error
                .to_string()
                .contains("FW-DTW-PROJECTION-PARENT-OVERLAP")
        );

        let invalid_parents = [
            (
                TranscriptionSegment {
                    start_sec: Some(f64::NAN),
                    end_sec: Some(1.0),
                    text: "one".to_owned(),
                    speaker: None,
                    confidence: None,
                },
                "FW-DTW-PROJECTION-PARENT-NONFINITE",
            ),
            (seg(-0.1, 1.0, "one"), "FW-DTW-PROJECTION-PARENT-NEGATIVE"),
            (seg(1.0, 1.0, "one"), "FW-DTW-PROJECTION-PARENT-DURATION"),
            (seg(2.0, 1.0, "one"), "FW-DTW-PROJECTION-PARENT-DURATION"),
        ];
        for (parent, expected_code) in invalid_parents {
            let error = build_segments_dtw(
                &[parent],
                &[Vec::new()],
                WordTimestampMode::Word,
                false,
                None,
            )
            .expect_err("invalid parent geometry must fail closed");
            assert!(
                error.to_string().contains(expected_code),
                "expected {expected_code}, got {error}"
            );
        }
    }

    #[test]
    fn dtw_projection_uses_conservative_segment_fallback_when_milliseconds_are_insufficient() {
        let engine = vec![seg(0.0, 0.001, "one two")];
        let timings = vec![vec![
            WordTiming {
                text: "one".to_owned(),
                start_sec: 0.0,
                end_sec: 0.001,
            },
            WordTiming {
                text: "two".to_owned(),
                start_sec: 0.001,
                end_sec: 0.001,
            },
        ]];
        let outcome = build_segments_dtw(&engine, &timings, WordTimestampMode::Word, false, None)
            .expect("unsupported word geometry must use the documented segment fallback");
        assert_eq!(outcome.segments.len(), 1);
        assert_eq!(outcome.segments[0].text, "one two");
        assert_eq!(
            outcome.canonical_units[0].provenance,
            ProjectionUnitProvenance::SegmentGeometryFallback
        );
        assert_eq!(outcome.report.interpolated_fallback_segments, 0);
        assert_eq!(outcome.report.segment_geometry_fallback_segments, 1);
        assert_eq!(outcome.report.clamped_units, 0);
        assert_eq!(
            outcome.report.fallback_reasons(),
            vec!["insufficient_parent_duration_for_millisecond_word_units"]
        );
        assert!(!outcome.report.word_aligned_safe);
    }

    #[test]
    fn dtw_projection_maxlen_and_no_timestamp_policies_preserve_canonical_proof() {
        let engine = vec![seg(0.0, 2.0, "alpha beta")];
        let timings = vec![vec![
            WordTiming {
                text: "alpha".to_owned(),
                start_sec: 0.0,
                end_sec: 1.0,
            },
            WordTiming {
                text: "beta".to_owned(),
                start_sec: 1.0,
                end_sec: 2.0,
            },
        ]];
        let grouped = build_segments_dtw(
            &engine,
            &timings,
            WordTimestampMode::MaxLen(20),
            false,
            None,
        )
        .expect("grouped canonical units");
        assert_eq!(grouped.segments.len(), 1);
        assert_eq!(grouped.segments[0].text, "alpha beta");
        assert_eq!(grouped.report.canonical_units, 2);
        assert_eq!(grouped.report.output_segments, 1);
        assert!(grouped.report.word_aligned_safe);

        let untimed = build_segments_dtw(&engine, &timings, WordTimestampMode::Word, true, None)
            .expect("no-timestamp output shaping");
        assert!(
            untimed
                .segments
                .iter()
                .all(|segment| segment.start_sec.is_none() && segment.end_sec.is_none())
        );
        assert_eq!(
            untimed.report.fallback_reasons(),
            vec!["timestamps_suppressed_by_request"]
        );
        assert!(!untimed.report.word_aligned_safe);
    }

    #[test]
    fn dtw_projection_is_deterministic_for_many_monotonic_quantized_timelines() {
        for word_count in 1..=64 {
            let parent_end = word_count as f64 * 0.01;
            let engine = vec![seg(0.0, parent_end, "synthetic timeline")];
            let timed = (0..word_count)
                .map(|index| {
                    let start = index as f64 * 0.01;
                    let end = if index % 3 == 0 { start } else { start + 0.007 };
                    WordTiming {
                        text: format!("w{index}"),
                        start_sec: start,
                        end_sec: end,
                    }
                })
                .collect::<Vec<_>>();
            let first = build_segments_dtw(
                &engine,
                std::slice::from_ref(&timed),
                WordTimestampMode::Word,
                false,
                None,
            )
            .expect("arbitrary monotonic finite timeline");
            let second = build_segments_dtw(
                &engine,
                std::slice::from_ref(&timed),
                WordTimestampMode::Word,
                false,
                None,
            )
            .expect("deterministic replay");
            assert_eq!(first.canonical_units, second.canonical_units);
            assert_eq!(first.report, second.report);
            assert_eq!(
                serde_json::to_value(&first.segments).expect("serialize first projection"),
                serde_json::to_value(&second.segments).expect("serialize second projection")
            );
            assert_eq!(first.segments.len(), word_count);
            assert!(
                first
                    .segments
                    .iter()
                    .all(|segment| segment.end_sec > segment.start_sec)
            );
            assert!(
                first
                    .segments
                    .windows(2)
                    .all(|pair| pair[0].end_sec <= pair[1].start_sec)
            );
        }
    }

    #[test]
    fn dtw_projection_honors_cancellation_before_normalization() {
        let engine = vec![seg(0.0, 1.0, "one")];
        let timings = vec![vec![WordTiming {
            text: "one".to_owned(),
            start_sec: 0.0,
            end_sec: 1.0,
        }]];
        let cancellation = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));
        let error = build_segments_dtw(
            &engine,
            &timings,
            WordTimestampMode::Word,
            false,
            Some(&cancellation),
        )
        .expect_err("expired projection token must fail before normalization");
        assert!(matches!(error, FwError::Cancelled(_)));
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
