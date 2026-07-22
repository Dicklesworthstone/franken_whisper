//! Real native whisper-diarization engine (bd-cidv).
//!
//! This module is the in-process, pure-Rust whisper-diarization engine. Like its
//! siblings [`whisper_cpp_native`](super::whisper_cpp_native) and
//! [`insanely_fast_native`](super::insanely_fast_native) it runs **genuine** ASR
//! inference through [`crate::native_engine`] (real ggml weights, real log-mel
//! frontend, real encoder/decoder forward passes) — there are **no canned
//! phrases, no mock segmentation, and no subprocess execution**. The former
//! `DiarizationPilot` that fabricated a fixed rotation of canned meeting phrases
//! from audio-energy regions is gone — it was the last canned-phrase mock left
//! in production.
//!
//! ## Pipeline
//!
//! 1. **Transcribe** (identical path to `whisper_cpp_native`): resolve the model
//!    (`request.model` → `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`), load it, read
//!    the normalized WAV, decode through the engine.
//! 2. **Diarize**: assign a speaker to every real segment via the orchestrator's
//!    heuristic diarizer ([`crate::orchestrator::diarize_segments`]), honoring
//!    the request's num/min/max speaker constraints. Segments are labeled
//!    `SPEAKER_NN` (DISC-002: cross-engine labels need not match exactly).
//!
//! ## Diarizer honesty — NOT a neural speaker encoder
//!
//! The diarization stage is an **acoustic-feature/temporal heuristic**: it
//! clusters segments on temporal position, pacing, turn-taking gaps, and lexical
//! features. It does **not** run a neural speaker encoder (ECAPA/TitaNet) and
//! does not extract per-speaker acoustic embeddings from the waveform. The
//! `raw_output` states this plainly (`"diarizer": "text-temporal-heuristic"` with
//! a `"diarizer_note"` spelling out the limitation); the neural ECAPA upgrade is
//! tracked in bd-ohex. Nothing here implies neural diarization.
//!
//! ## Model resolution, silence pre-gate, availability
//!
//! These policies are identical to
//! [`whisper_cpp_native`](super::whisper_cpp_native): the model is resolved from
//! `request.model` then `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL`; a cheap energy
//! pre-gate returns an empty result for pure silence without loading any weights;
//! and availability **delegates** to
//! [`whisper_cpp_native::is_available`](super::whisper_cpp_native::is_available)
//! (the same native engine runs over the same ggml model files, so they must
//! agree).

use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, SpeakerConstraints, TranscribeRequest, TranscriptionResult, TranscriptionSegment,
};
use crate::native_engine::{self, NativeWhisperModel, decode};
use crate::orchestrator::{self, DiarizeReport};

use super::native_audio::analyze_wav;

/// Stable schema tag for the honest native raw-output metadata (shared with the
/// other native engines so the evidence ledger has one schema).
const SCHEMA_VERSION: &str = "native-v2";

/// Diarizer identity tag for `raw_output`. This is a text/temporal heuristic, not
/// a neural speaker encoder — see the module docs and [`DIARIZER_NOTE`].
const DIARIZER_TAG: &str = "text-temporal-heuristic";

/// Honest one-line statement of the diarizer's quality limitation, surfaced in
/// `raw_output` so no consumer mistakes it for neural diarization.
const DIARIZER_NOTE: &str = "acoustic-feature clustering without neural speaker encoder; \
     ECAPA upgrade tracked in bd-ohex";

/// Honestly report whether the native whisper-diarization engine can run.
///
/// Delegates to
/// [`whisper_cpp_native::is_available`](super::whisper_cpp_native::is_available):
/// this engine runs the **same** [`crate::native_engine`] over the **same** ggml
/// model files, so the two must agree on availability. Reports `true` only when a
/// usable model header exists (the configured default resolves, or any
/// `ggml-*.bin` with a valid header sits in a search dir). Never panics or
/// performs network access.
#[must_use]
pub fn is_available() -> bool {
    super::whisper_cpp_native::is_available()
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
        "native whisper-diarization engine has no model: pass --model, or set \
         $FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL to a model short-name or path. \
         {dirs_hint}"
    )))
}

/// Build the [`decode::DecodeParams`] for a request. Diarization keeps the
/// engine's native segment boundaries (no word-level splitting) since the
/// diarizer assigns one speaker per segment.
fn decode_params(request: &TranscribeRequest) -> decode::DecodeParams {
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
        max_text_ctx: None,
        ..decode::DecodeParams::default()
    }
}

/// Bridge an optional orchestrator [`CancellationToken`] into the engine's
/// `checkpoint` closure shape (`Fn() -> FwResult<()>`).
fn checkpoint_for(
    token: Option<&crate::orchestrator::CancellationToken>,
) -> impl Fn() -> FwResult<()> + '_ {
    move || token.map_or(Ok(()), crate::orchestrator::CancellationToken::checkpoint)
}

/// The effective speaker constraints for diarization: the request's
/// `num`/`min`/`max` speaker constraints, honored end-to-end by the diarizer.
/// `None` means "auto-detect speaker count".
fn speaker_constraints_for(request: &TranscribeRequest) -> Option<SpeakerConstraints> {
    request.backend_params.speaker_constraints.clone()
}

/// Compute the whole-clip audio duration in seconds for the diarizer, preferring
/// the request's duration hint, then the engine's window stats, then the segment
/// timestamps.
fn audio_duration_sec(request: &TranscribeRequest, output: &decode::DecodeOutput) -> Option<f64> {
    if let Some(ms) = request.backend_params.duration_ms {
        return Some(ms as f64 / 1_000.0);
    }
    output
        .segments
        .iter()
        .filter_map(|s| s.end_sec)
        .fold(None, |acc: Option<f64>, end| {
            Some(acc.map_or(end, |m| m.max(end)))
        })
}

/// Run real native whisper-diarization inference over `normalized_wav`
/// (guaranteed 16 kHz mono PCM16 by the pipeline): transcribe through the native
/// engine, then assign a speaker to every real segment with the heuristic
/// diarizer.
///
/// See the module docs for the diarizer-honesty, model-resolution, and
/// silence-pre-gate policies.
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
    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Resolve the model spec up front so an unavailability error is reported
    // before any expensive work.
    let spec = effective_model_spec(request)?;

    // Silence pre-gate: cheap energy analysis avoids a multi-GB model load on a
    // pure-silence clip (shared policy with the sibling native engines).
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

    let samples = read_normalized_wav(normalized_wav)?;

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Real ASR: the engine decides what (and when) words were spoken.
    let params = decode_params(request);
    let checkpoint = checkpoint_for(token);
    let output = model.transcribe(&samples, &params, &checkpoint)?;

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    // Real diarization: assign a speaker to every engine segment via the
    // heuristic diarizer, honoring the request's speaker constraints.
    let mut segments = output.segments.clone();
    let constraints = speaker_constraints_for(request);
    let duration_sec = audio_duration_sec(request, &output);
    let diarize_token = token
        .copied()
        .unwrap_or_else(crate::orchestrator::CancellationToken::unbounded);
    let report = orchestrator::diarize_segments(
        &mut segments,
        duration_sec,
        constraints.as_ref(),
        &diarize_token,
    )?;

    let transcript = super::transcript_from_segments(&segments);
    let language = output.language.clone().or_else(|| request.language.clone());

    // Honest audio provenance from the energy pre-gate (frame RMS, active
    // regions): this is the cheap waveform analysis, NOT the diarizer's clusters.
    let audio_provenance = analysis
        .as_ref()
        .map(super::native_audio::NativeAudioAnalysis::as_json);
    let raw_output = raw_output_json(
        &spec,
        &model_path,
        model.version_tag(),
        &output.windows,
        &report,
        audio_provenance,
        false,
    );

    Ok(TranscriptionResult {
        backend: BackendKind::WhisperDiarization,
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
/// The native decoder is batch-only and diarization needs the whole-clip segment
/// set to cluster speakers, so this runs the full [`run`] pathway and then
/// replays the diarized segments through `on_segment` in order (mirroring the
/// sibling native engines). The cancellation token is honored between emitted
/// segments.
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

    if let Value::Object(map) = &mut result.raw_output {
        map.insert(
            "streaming_emitted_segments".to_owned(),
            json!(result.segments.len()),
        );
    }

    Ok(result)
}

/// Read a normalized 16 kHz mono PCM16 WAV into f32 mono samples (shares the
/// engine's RIFF reader so production and gated e2e tests use one decoder).
fn read_normalized_wav(path: &Path) -> FwResult<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    decode::read_wav_16k_mono(&bytes)
}

/// The honest raw-output metadata JSON for a real-inference diarization run.
///
/// The diarizer fields are deliberately explicit about the heuristic's limits:
/// `"diarizer": "text-temporal-heuristic"` + a `"diarizer_note"` spelling out
/// that there is no neural speaker encoder. Never implies neural diarization.
fn raw_output_json(
    spec: &str,
    model_path: &Path,
    version_tag: String,
    windows: &[decode::WindowStats],
    report: &DiarizeReport,
    audio_provenance: Option<Value>,
    silence: bool,
) -> Value {
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
        "engine": "whisper-diarization-native",
        "schema_version": SCHEMA_VERSION,
        "in_process": true,
        "implementation": "real-inference",
        "silence": silence,
        "model": spec,
        "model_path": model_path.display().to_string(),
        "model_version_tag": version_tag,
        "windows": windows_json,
        "diarizer": DIARIZER_TAG,
        "diarizer_note": DIARIZER_NOTE,
        "silhouette": report.silhouette_score,
        "speakers_detected": report.speakers_detected,
        "segments_labeled": report.segments_labeled,
        "diarizer_notes": report.notes,
        "audio_analysis": audio_provenance,
    })
}

/// Build the empty-but-valid result for a pure-silence clip, taken **without
/// loading the model** (the energy pre-gate already proved there is nothing to
/// transcribe — and nothing to diarize).
fn silence_result(
    request: &TranscribeRequest,
    spec: &str,
    duration_ms: u64,
) -> TranscriptionResult {
    TranscriptionResult {
        backend: BackendKind::WhisperDiarization,
        transcript: String::new(),
        language: request.language.clone(),
        segments: Vec::new(),
        acceleration: None,
        raw_output: json!({
            "engine": "whisper-diarization-native",
            "schema_version": SCHEMA_VERSION,
            "in_process": true,
            "implementation": "real-inference",
            "silence": true,
            "model": spec,
            "model_loaded": false,
            "duration_ms": duration_ms,
            "windows": [],
            "diarizer": DIARIZER_TAG,
            "diarizer_note": DIARIZER_NOTE,
            "silhouette": Value::Null,
            "speakers_detected": 0,
            "segments_labeled": 0,
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
        BackendKind, BackendParams, InputSource, SpeakerConstraints, TranscribeRequest,
    };
    use crate::native_engine::{self, decode};
    use crate::orchestrator::CancellationToken;

    use super::*;

    fn request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::WhisperDiarization,
            model: Some("tiny.en".to_owned()),
            language: Some("en".to_owned()),
            translate: false,
            diarize: true,
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

    /// A real-shaped engine segment, standing in for native decoder output.
    fn seg(start: f64, end: f64, text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: Some(start),
            end_sec: Some(end),
            text: text.to_owned(),
            speaker: None,
            confidence: Some(0.9),
        }
    }

    fn window(offset: f64) -> decode::WindowStats {
        decode::WindowStats {
            avg_logprob: -0.2,
            no_speech_prob: 0.01,
            tokens: 5,
            window_offset_sec: offset,
        }
    }

    // ── Engine-trait shape ────────────────────────────────────────────────

    #[test]
    fn native_engine_name_follows_naming_convention() {
        let engine = super::super::WhisperDiarizationNativeEngine;
        assert_eq!(engine.name(), "whisper-diarization-native");
    }

    #[test]
    fn native_engine_kind_matches_bridge_adapter() {
        let native = super::super::WhisperDiarizationNativeEngine;
        let bridge = super::super::WhisperDiarizationEngine;
        assert_eq!(native.kind(), bridge.kind());
        assert_eq!(native.kind(), BackendKind::WhisperDiarization);
    }

    #[test]
    fn native_engine_name_distinct_from_bridge() {
        let native = super::super::WhisperDiarizationNativeEngine;
        let bridge = super::super::WhisperDiarizationEngine;
        assert_ne!(native.name(), bridge.name());
        assert!(native.name().contains("native"));
    }

    #[test]
    fn availability_agrees_with_whisper_cpp_native() {
        // Same native engine over the same model files: availability must match.
        assert_eq!(
            is_available(),
            super::super::whisper_cpp_native::is_available()
        );
    }

    #[test]
    fn native_available_routes_diarization_here() {
        // The router's native_available for WhisperDiarization must agree with
        // this module (it delegates to whisper_cpp_native::is_available too).
        assert_eq!(
            super::super::native_available(BackendKind::WhisperDiarization),
            is_available()
        );
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
            Some("whisper-diarization-native")
        );
        assert_eq!(
            result.raw_output["schema_version"].as_str(),
            Some(SCHEMA_VERSION)
        );
        // Honest diarizer tag even on the silence path.
        assert_eq!(result.raw_output["diarizer"].as_str(), Some(DIARIZER_TAG));
        assert_eq!(result.raw_output["speakers_detected"].as_u64(), Some(0));
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

    // ── Speaker-label format + raw_output schema (hermetic) ───────────────

    #[test]
    fn diarized_segments_carry_speaker_nn_labels() {
        // Drive the real diarizer through real-shaped segments and assert every
        // segment gets a SPEAKER_NN label (DISC-002 label format).
        let mut segments = vec![
            seg(0.0, 2.0, "and so my fellow americans"),
            seg(5.0, 7.0, "ask not what your country can do"),
            seg(7.0, 9.0, "ask what you can do for your country"),
        ];
        let token = CancellationToken::unbounded();
        let report = orchestrator::diarize_segments(&mut segments, Some(9.0), None, &token)
            .expect("diarize");
        assert_eq!(report.segments_labeled, 3);
        let re = regex_speaker();
        for s in &segments {
            let label = s.speaker.as_deref().expect("speaker assigned");
            assert!(re(label), "label {label:?} must match SPEAKER_NN");
        }
    }

    /// Minimal `^SPEAKER_\d{2}$` matcher (no regex dependency in unit tests).
    fn regex_speaker() -> impl Fn(&str) -> bool {
        |label: &str| {
            let Some(rest) = label.strip_prefix("SPEAKER_") else {
                return false;
            };
            rest.len() == 2 && rest.bytes().all(|b| b.is_ascii_digit())
        }
    }

    #[test]
    fn raw_output_is_honest_about_heuristic_diarizer() {
        let report = DiarizeReport {
            segments_total: 3,
            speakers_detected: 2,
            segments_labeled: 3,
            silhouette_score: Some(0.42),
            notes: vec!["heuristic: acoustic-feature clustering".to_owned()],
        };
        let json = raw_output_json(
            "tiny.en",
            Path::new("/models/ggml-tiny.en.bin"),
            "fw-native-v1+sha256:abc".to_owned(),
            &[window(0.0)],
            &report,
            None,
            false,
        );
        assert_eq!(json["engine"].as_str(), Some("whisper-diarization-native"));
        assert_eq!(json["schema_version"].as_str(), Some(SCHEMA_VERSION));
        assert_eq!(json["implementation"].as_str(), Some("real-inference"));
        assert_eq!(json["in_process"].as_bool(), Some(true));
        // Diarizer honesty: heuristic tag + explanatory note, never "neural"/"ecapa".
        assert_eq!(json["diarizer"].as_str(), Some(DIARIZER_TAG));
        let note = json["diarizer_note"].as_str().expect("note present");
        assert!(
            note.contains("without neural speaker encoder"),
            "note must state the limitation: {note}"
        );
        assert!(
            note.contains("bd-ohex"),
            "note must reference the ECAPA upgrade bead: {note}"
        );
        let serialized = json.to_string().to_lowercase();
        assert!(
            !serialized.contains("ecapa\"") && !serialized.contains("\"neural\""),
            "diarizer must never claim neural/ecapa identity: {serialized}"
        );
        assert_eq!(json["silhouette"].as_f64(), Some(0.42));
        assert_eq!(json["speakers_detected"].as_u64(), Some(2));
        assert_eq!(json["segments_labeled"].as_u64(), Some(3));
    }

    #[test]
    fn audio_duration_sec_prefers_hint_then_segments() {
        let mut req = request();
        req.backend_params.duration_ms = Some(12_000);
        let output = decode::DecodeOutput {
            segments: vec![seg(0.0, 9.0, "x")],
            language: Some("en".to_owned()),
            windows: vec![],
            word_timings: None,
        };
        assert_eq!(audio_duration_sec(&req, &output), Some(12.0));

        req.backend_params.duration_ms = None;
        assert_eq!(audio_duration_sec(&req, &output), Some(9.0));

        let empty = decode::DecodeOutput {
            segments: vec![],
            language: None,
            windows: vec![],
            word_timings: None,
        };
        assert_eq!(audio_duration_sec(&req, &empty), None);
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

    fn write_samples_wav(dir: &Path, name: &str, samples: &[f32]) -> PathBuf {
        let pcm: Vec<i16> = samples
            .iter()
            .map(|s| (if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 } * 32767.0) as i16)
            .collect();
        let path = dir.join(name);
        write_pcm16_mono_wav(&path, 16_000, &pcm);
        path
    }

    #[test]
    fn gated_e2e_jfk_tiny_en_transcript_and_speakers() {
        if !tiny_en_available() {
            eprintln!("SKIP gated_e2e_jfk: tiny.en model missing");
            return;
        }
        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let mut req = request();
        req.model = Some("tiny.en".to_owned());
        req.language = None;

        let engine = super::super::WhisperDiarizationNativeEngine;
        let result = crate::backend::Engine::run(
            &engine,
            &req,
            Path::new(wav),
            Path::new("."),
            Duration::from_secs(120),
            None,
        )
        .expect("e2e run");

        // Transcript matches the shared reference exactly (real inference).
        assert_eq!(result.transcript.trim(), JFK_REFERENCE);
        assert_eq!(
            result.raw_output["implementation"].as_str(),
            Some("real-inference")
        );
        assert_eq!(result.raw_output["silence"].as_bool(), Some(false));
        assert_eq!(result.raw_output["diarizer"].as_str(), Some(DIARIZER_TAG));

        // Every segment carries a SPEAKER_NN label.
        let re = regex_speaker();
        assert!(result.segments.len() >= 2, "expected >= 2 segments");
        for s in &result.segments {
            let label = s.speaker.as_deref().expect("segment has speaker");
            assert!(re(label), "label {label:?} must match SPEAKER_NN");
        }
        assert!(
            result.raw_output["speakers_detected"].as_u64().unwrap_or(0) >= 1,
            "at least one speaker detected"
        );
    }

    #[test]
    fn gated_e2e_min_speakers_two_honored_on_multi_window_clip() {
        let Some(samples) = load_jfk_samples() else {
            eprintln!("SKIP gated_e2e_min_speakers: jfk.wav missing");
            return;
        };
        if !tiny_en_available() {
            eprintln!("SKIP gated_e2e_min_speakers: tiny.en model missing");
            return;
        }
        // Concatenate jfk 3x (~33 s) so the diarizer has many segments to cluster
        // and can comply with a min_speakers=2 constraint.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut long = Vec::with_capacity(samples.len() * 3);
        for _ in 0..3 {
            long.extend_from_slice(&samples);
        }
        let wav = write_samples_wav(dir.path(), "jfk3x.wav", &long);

        let mut req = request();
        req.model = Some("tiny.en".to_owned());
        req.language = None;
        req.backend_params.speaker_constraints = Some(SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(2),
            max_speakers: None,
        });

        let result = run(&req, &wav, dir.path(), Duration::from_secs(180), None)
            .expect("multi-window diarization run");

        let speakers = result.raw_output["speakers_detected"]
            .as_u64()
            .expect("speakers_detected present");
        assert!(
            speakers >= 2,
            "min_speakers=2 should yield >= 2 speakers on a multi-segment clip, got {speakers}"
        );
        let re = regex_speaker();
        for s in &result.segments {
            assert!(re(s.speaker.as_deref().expect("speaker")));
        }
    }

    #[test]
    fn gated_e2e_streaming_replays_all_segments() {
        if !tiny_en_available() {
            eprintln!("SKIP gated_e2e_streaming: tiny.en model missing");
            return;
        }
        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
        let mut req = request();
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
        // Replayed segments carry their speaker labels.
        assert!(emitted.iter().all(|s| s.speaker.is_some()));
    }
}
