//! Unified output format normalization for all backends.
//!
//! Each backend (whisper_cpp, insanely_fast, whisper_diarization) produces
//! output in a different format. This module centralizes the normalization
//! logic into a single place, producing a `NormalizedOutput` that captures
//! the common fields every backend should yield.
//!
//! # Design
//!
//! - Backend-specific raw JSON is accepted as `&serde_json::Value`.
//! - Each `normalize_*` function extracts transcript, segments, and language
//!   from the backend's native format.
//! - `to_transcription_result` converts the normalized intermediate form
//!   into the final `TranscriptionResult` consumed by the rest of the system.

use serde_json::Value;

use super::{extract_segments_from_json, transcript_from_segments};
use crate::error::{FwError, FwResult};
use crate::model::{BackendKind, TranscriptionResult, TranscriptionSegment};

// ---------------------------------------------------------------------------
// NormalizedOutput
// ---------------------------------------------------------------------------

/// Backend-agnostic intermediate representation of a transcription result.
///
/// Every backend normalizer produces this struct, which can then be converted
/// into a `TranscriptionResult` via [`to_transcription_result`].
#[derive(Debug, Clone)]
pub struct NormalizedOutput {
    /// Full transcript text.
    pub transcript: String,
    /// Timed segments extracted from the backend output.
    pub segments: Vec<TranscriptionSegment>,
    /// Detected language, if available.
    pub language: Option<String>,
    /// Original raw output preserved verbatim for audit/replay.
    pub raw_output: Value,
}

// ---------------------------------------------------------------------------
// Per-backend normalizers
// ---------------------------------------------------------------------------

/// Normalize whisper.cpp JSON output.
///
/// Expected structure (from `whisper-cli -oj`):
/// ```json
/// {
///   "text": "...",
///   "segments": [ { "start": f64, "end": f64, "text": "..." } ],
///   "language": "en"
/// }
/// ```
///
/// Language may also be at `result.language`.
pub fn normalize_whisper_cpp(raw_json: &Value) -> FwResult<NormalizedOutput> {
    validate_is_object(raw_json, "whisper_cpp")?;

    let segments = extract_segments_from_json(raw_json);

    let transcript = raw_json
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| transcript_from_segments(&segments));

    let language = raw_json
        .pointer("/result/language")
        .or_else(|| raw_json.get("language"))
        .and_then(Value::as_str)
        .map(str::to_owned);

    Ok(NormalizedOutput {
        transcript,
        segments,
        language,
        raw_output: raw_json.clone(),
    })
}

/// Normalize insanely-fast-whisper JSON output.
///
/// Expected structure:
/// ```json
/// {
///   "text": "...",
///   "chunks": [ { "text": "...", "timestamp": [start, end] } ],
///   "language": "en"
/// }
/// ```
///
/// Chunks may also contain word-level timestamps with nested `"words"` arrays.
pub fn normalize_insanely_fast(raw_json: &Value) -> FwResult<NormalizedOutput> {
    validate_is_object(raw_json, "insanely_fast")?;

    let segments = extract_segments_from_json(raw_json);

    let transcript = raw_json
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| transcript_from_segments(&segments));

    let language = raw_json
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_owned);

    Ok(NormalizedOutput {
        transcript,
        segments,
        language,
        raw_output: raw_json.clone(),
    })
}

/// Normalize whisper-diarization JSON output.
///
/// The diarization backend produces a synthetic JSON envelope:
/// ```json
/// {
///   "transcript_txt": "...",
///   "srt_path": "/path/to/file.srt"
/// }
/// ```
///
/// Segments are not embedded in the JSON itself; instead they come from
/// the SRT file parsed separately. When raw JSON is all we have (e.g. for
/// replay), we extract what we can from the envelope.
pub fn normalize_whisper_diarization(raw_json: &Value) -> FwResult<NormalizedOutput> {
    validate_is_object(raw_json, "whisper_diarization")?;

    let segments = extract_segments_from_json(raw_json);

    // The diarization backend stores the transcript under `transcript_txt`.
    let transcript = raw_json
        .get("transcript_txt")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| transcript_from_segments(&segments));

    // The diarization backend does not emit a language field in its JSON;
    // the caller typically supplies the language from the request.
    let language = raw_json
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_owned);

    Ok(NormalizedOutput {
        transcript,
        segments,
        language,
        raw_output: raw_json.clone(),
    })
}

// ---------------------------------------------------------------------------
// Conversion to TranscriptionResult
// ---------------------------------------------------------------------------

/// Convert a `NormalizedOutput` into a `TranscriptionResult`.
///
/// The `backend` parameter identifies which engine produced the output.
/// Acceleration and artifact paths are not part of normalization and are
/// left as their default (None / empty) values; callers should fill them
/// in after conversion if needed.
pub fn to_transcription_result(
    normalized: NormalizedOutput,
    backend: BackendKind,
) -> TranscriptionResult {
    TranscriptionResult {
        backend,
        transcript: normalized.transcript,
        language: normalized.language,
        segments: normalized.segments,
        acceleration: None,
        diarization: None,
        raw_output: normalized.raw_output,
        artifact_paths: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that the root value is a JSON object, returning a descriptive
/// error if it is not (e.g. if the raw output is null, an array, or a scalar).
fn validate_is_object(value: &Value, backend_name: &str) -> FwResult<()> {
    if !value.is_object() {
        return Err(FwError::InvalidRequest(format!(
            "{backend_name} raw output is not a JSON object"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::model::BackendKind;

    // -----------------------------------------------------------------------
    // Golden file tests
    // -----------------------------------------------------------------------

    fn golden_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/golden")
            .join(name)
    }

    fn load_golden_json(name: &str) -> Value {
        let path = golden_path(name);
        let content = std::fs::read_to_string(&path).expect("failed to read golden file");
        serde_json::from_str(&content).expect("failed to parse golden file")
    }

    #[test]
    fn golden_whisper_cpp_normalization() {
        let raw = load_golden_json("whisper_cpp_output.json");
        let normalized = normalize_whisper_cpp(&raw).expect("should normalize whisper_cpp golden");

        assert_eq!(
            normalized.transcript,
            "Hello world. This is a test of the whisper speech recognition system."
        );
        assert_eq!(normalized.segments.len(), 2);
        assert_eq!(normalized.segments[0].text, "Hello world.");
        assert_eq!(
            normalized.segments[1].text,
            "This is a test of the whisper speech recognition system."
        );
        assert!((normalized.segments[0].start_sec.unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((normalized.segments[0].end_sec.unwrap() - 2.5).abs() < f64::EPSILON);
        assert!((normalized.segments[1].start_sec.unwrap() - 2.5).abs() < f64::EPSILON);
        assert!((normalized.segments[1].end_sec.unwrap() - 6.0).abs() < f64::EPSILON);
        assert_eq!(normalized.language.as_deref(), Some("en"));
        assert_eq!(normalized.raw_output, raw);
    }

    #[test]
    fn golden_insanely_fast_normalization() {
        let raw = load_golden_json("insanely_fast_output.json");
        let normalized =
            normalize_insanely_fast(&raw).expect("should normalize insanely_fast golden");

        assert_eq!(
            normalized.transcript,
            "Hello world. This is a test of the whisper speech recognition system."
        );
        assert_eq!(normalized.segments.len(), 2);
        assert_eq!(normalized.segments[0].text, "Hello world.");
        assert_eq!(
            normalized.segments[1].text,
            "This is a test of the whisper speech recognition system."
        );
        assert!((normalized.segments[0].start_sec.unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((normalized.segments[0].end_sec.unwrap() - 2.5).abs() < f64::EPSILON);
        assert!((normalized.segments[1].start_sec.unwrap() - 2.5).abs() < f64::EPSILON);
        assert!((normalized.segments[1].end_sec.unwrap() - 6.0).abs() < f64::EPSILON);
        // insanely_fast golden file does not have a top-level "language" field
        assert!(normalized.language.is_none());
        assert_eq!(normalized.raw_output, raw);
    }

    #[test]
    fn golden_whisper_diarization_normalization() {
        // The diarization backend produces a synthetic JSON envelope from
        // the .txt transcript. We reconstruct that envelope as the real
        // backend does.
        let txt_content = std::fs::read_to_string(golden_path("diarization_output.txt"))
            .expect("golden txt should exist");
        let raw = json!({
            "transcript_txt": txt_content,
            "srt_path": golden_path("diarization_output.srt").display().to_string(),
        });

        let normalized =
            normalize_whisper_diarization(&raw).expect("should normalize diarization golden");

        // The transcript should be trimmed.
        assert!(
            normalized.transcript.contains("Hello world."),
            "transcript should contain greeting"
        );
        assert!(
            normalized.transcript.contains("SPEAKER_00"),
            "diarization transcript contains speaker labels"
        );
        // Diarization JSON envelope does not embed segments.
        assert!(
            normalized.segments.is_empty(),
            "diarization JSON envelope has no segments array"
        );
        // No language in the synthetic envelope.
        assert!(normalized.language.is_none());
    }

    // -----------------------------------------------------------------------
    // whisper_cpp normalizer unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn whisper_cpp_with_result_language() {
        let raw = json!({
            "text": "bonjour",
            "result": {"language": "fr"},
            "segments": [{"start": 0.0, "end": 1.0, "text": "bonjour"}],
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.language.as_deref(), Some("fr"));
    }

    #[test]
    fn whisper_cpp_empty_text_falls_back_to_segments() {
        let raw = json!({
            "text": "   ",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.transcript, "hello world");
    }

    #[test]
    fn whisper_cpp_missing_text_falls_back_to_segments() {
        let raw = json!({
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "only segments"},
            ],
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.transcript, "only segments");
    }

    #[test]
    fn whisper_cpp_no_segments_returns_empty_vec() {
        let raw = json!({"text": "no segments here"});
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert!(normalized.segments.is_empty());
        assert_eq!(normalized.transcript, "no segments here");
    }

    #[test]
    fn whisper_cpp_no_language_returns_none() {
        let raw = json!({"text": "hello", "segments": []});
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert!(normalized.language.is_none());
    }

    #[test]
    fn whisper_cpp_transcription_array_key() {
        // whisper.cpp can emit "transcription" instead of "segments"
        let raw = json!({
            "text": "via transcription key",
            "transcription": [
                {"start": 0.0, "end": 1.5, "text": "via transcription key"},
            ],
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.segments.len(), 1);
        assert_eq!(normalized.segments[0].text, "via transcription key");
    }

    // -----------------------------------------------------------------------
    // insanely_fast normalizer unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn insanely_fast_with_language() {
        let raw = json!({
            "text": "hola mundo",
            "chunks": [{"text": "hola mundo", "timestamp": [0.0, 2.0]}],
            "language": "es",
        });
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert_eq!(normalized.language.as_deref(), Some("es"));
        assert_eq!(normalized.transcript, "hola mundo");
    }

    #[test]
    fn insanely_fast_empty_text_falls_back_to_segments() {
        let raw = json!({
            "text": "",
            "chunks": [
                {"text": "from chunks", "timestamp": [0.0, 1.0]},
            ],
        });
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert_eq!(normalized.transcript, "from chunks");
    }

    #[test]
    fn insanely_fast_no_chunks_returns_empty_segments() {
        let raw = json!({"text": "plain text only"});
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert!(normalized.segments.is_empty());
        assert_eq!(normalized.transcript, "plain text only");
    }

    #[test]
    fn insanely_fast_word_level_timestamps() {
        let raw = json!({
            "text": "hello world",
            "chunks": [{
                "text": "hello world",
                "timestamp": [0.0, 2.0],
                "words": [
                    {"word": "hello", "start": 0.0, "end": 1.0},
                    {"word": "world", "start": 1.0, "end": 2.0},
                ],
            }],
        });
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert_eq!(normalized.segments.len(), 2);
        assert_eq!(normalized.segments[0].text, "hello");
        assert_eq!(normalized.segments[1].text, "world");
    }

    #[test]
    fn insanely_fast_with_speakers() {
        let raw = json!({
            "text": "hi there",
            "chunks": [
                {"text": "hi", "timestamp": [0.0, 1.0], "speaker": "SPEAKER_00"},
                {"text": "there", "timestamp": [1.0, 2.0], "speaker": "SPEAKER_01"},
            ],
        });
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert_eq!(
            normalized.segments[0].speaker.as_deref(),
            Some("SPEAKER_00")
        );
        assert_eq!(
            normalized.segments[1].speaker.as_deref(),
            Some("SPEAKER_01")
        );
    }

    // -----------------------------------------------------------------------
    // whisper_diarization normalizer unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn diarization_extracts_transcript_txt() {
        let raw = json!({
            "transcript_txt": "  Speaker 0: hello world  ",
            "srt_path": "/tmp/test.srt",
        });
        let normalized = normalize_whisper_diarization(&raw).unwrap();
        assert_eq!(normalized.transcript, "Speaker 0: hello world");
    }

    #[test]
    fn diarization_missing_transcript_txt_returns_empty() {
        let raw = json!({"srt_path": "/tmp/test.srt"});
        let normalized = normalize_whisper_diarization(&raw).unwrap();
        assert!(normalized.transcript.is_empty());
    }

    #[test]
    fn diarization_falls_back_to_segments_when_transcript_missing() {
        let raw = json!({
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
        });
        let normalized = normalize_whisper_diarization(&raw).unwrap();
        assert_eq!(normalized.transcript, "hello world");
    }

    #[test]
    fn diarization_with_embedded_segments() {
        // Future extensibility: if someone adds segments to the envelope.
        let raw = json!({
            "transcript_txt": "hello",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
            ],
        });
        let normalized = normalize_whisper_diarization(&raw).unwrap();
        assert_eq!(normalized.segments.len(), 1);
        assert_eq!(normalized.segments[0].text, "hello");
    }

    #[test]
    fn diarization_with_language() {
        let raw = json!({
            "transcript_txt": "bonjour",
            "language": "fr",
        });
        let normalized = normalize_whisper_diarization(&raw).unwrap();
        assert_eq!(normalized.language.as_deref(), Some("fr"));
    }

    // -----------------------------------------------------------------------
    // to_transcription_result tests
    // -----------------------------------------------------------------------

    #[test]
    fn to_transcription_result_whisper_cpp() {
        let normalized = NormalizedOutput {
            transcript: "hello".to_owned(),
            segments: vec![TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "hello".to_owned(),
                speaker: None,
                confidence: Some(0.95),
            }],
            language: Some("en".to_owned()),
            raw_output: json!({"text": "hello"}),
        };
        let result = to_transcription_result(normalized, BackendKind::WhisperCpp);
        assert_eq!(result.backend, BackendKind::WhisperCpp);
        assert_eq!(result.transcript, "hello");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.segments.len(), 1);
        assert!(result.acceleration.is_none());
        assert!(result.artifact_paths.is_empty());
    }

    #[test]
    fn to_transcription_result_insanely_fast() {
        let normalized = NormalizedOutput {
            transcript: "test".to_owned(),
            segments: vec![],
            language: None,
            raw_output: json!({}),
        };
        let result = to_transcription_result(normalized, BackendKind::InsanelyFast);
        assert_eq!(result.backend, BackendKind::InsanelyFast);
        assert_eq!(result.transcript, "test");
        assert!(result.language.is_none());
    }

    #[test]
    fn to_transcription_result_diarization() {
        let normalized = NormalizedOutput {
            transcript: "speaker 0: hi".to_owned(),
            segments: vec![],
            language: Some("ja".to_owned()),
            raw_output: json!({"transcript_txt": "speaker 0: hi"}),
        };
        let result = to_transcription_result(normalized, BackendKind::WhisperDiarization);
        assert_eq!(result.backend, BackendKind::WhisperDiarization);
        assert_eq!(result.language.as_deref(), Some("ja"));
    }

    #[test]
    fn to_transcription_result_preserves_raw_output() {
        let raw = json!({"complex": {"nested": [1, 2, 3]}});
        let normalized = NormalizedOutput {
            transcript: String::new(),
            segments: vec![],
            language: None,
            raw_output: raw.clone(),
        };
        let result = to_transcription_result(normalized, BackendKind::Auto);
        assert_eq!(result.raw_output, raw);
    }

    // -----------------------------------------------------------------------
    // Edge cases: empty output
    // -----------------------------------------------------------------------

    #[test]
    fn whisper_cpp_empty_object() {
        let raw = json!({});
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert!(normalized.transcript.is_empty());
        assert!(normalized.segments.is_empty());
        assert!(normalized.language.is_none());
    }

    #[test]
    fn insanely_fast_empty_object() {
        let raw = json!({});
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert!(normalized.transcript.is_empty());
        assert!(normalized.segments.is_empty());
        assert!(normalized.language.is_none());
    }

    #[test]
    fn diarization_empty_object() {
        let raw = json!({});
        let normalized = normalize_whisper_diarization(&raw).unwrap();
        assert!(normalized.transcript.is_empty());
        assert!(normalized.segments.is_empty());
        assert!(normalized.language.is_none());
    }

    // -----------------------------------------------------------------------
    // Edge cases: missing fields
    // -----------------------------------------------------------------------

    #[test]
    fn whisper_cpp_null_text_field() {
        let raw = json!({"text": null, "segments": []});
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        // null text means as_str() returns None, so fallback to segments
        assert!(normalized.transcript.is_empty());
    }

    #[test]
    fn insanely_fast_null_text_field() {
        let raw = json!({"text": null, "chunks": []});
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert!(normalized.transcript.is_empty());
    }

    #[test]
    fn whisper_cpp_language_is_number_ignored() {
        let raw = json!({"text": "hello", "language": 42});
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        // language is not a string, so it should be None
        assert!(normalized.language.is_none());
    }

    #[test]
    fn insanely_fast_language_is_number_ignored() {
        let raw = json!({"text": "hello", "language": 42});
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert!(normalized.language.is_none());
    }

    // -----------------------------------------------------------------------
    // Edge cases: malformed JSON (not an object)
    // -----------------------------------------------------------------------

    #[test]
    fn whisper_cpp_null_json_rejected() {
        let raw = json!(null);
        let result = normalize_whisper_cpp(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn whisper_cpp_array_json_rejected() {
        let raw = json!([1, 2, 3]);
        let result = normalize_whisper_cpp(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn whisper_cpp_string_json_rejected() {
        let raw = json!("just a string");
        let result = normalize_whisper_cpp(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn insanely_fast_null_json_rejected() {
        let raw = json!(null);
        let result = normalize_insanely_fast(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn insanely_fast_number_json_rejected() {
        let raw = json!(42);
        let result = normalize_insanely_fast(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn diarization_null_json_rejected() {
        let raw = json!(null);
        let result = normalize_whisper_diarization(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn diarization_array_json_rejected() {
        let raw = json!(["not", "an", "object"]);
        let result = normalize_whisper_diarization(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn diarization_boolean_json_rejected() {
        let raw = json!(true);
        let result = normalize_whisper_diarization(&raw);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Edge cases: segment field extraction
    // -----------------------------------------------------------------------

    #[test]
    fn whisper_cpp_segments_with_offsets() {
        // whisper.cpp uses offsets.from / offsets.to in milliseconds
        let raw = json!({
            "text": "offset-based",
            "transcription": [
                {
                    "offsets": {"from": 1000, "to": 2500},
                    "text": "offset-based",
                },
            ],
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.segments.len(), 1);
        assert!((normalized.segments[0].start_sec.unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((normalized.segments[0].end_sec.unwrap() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn insanely_fast_timestamp_as_object() {
        // insanely_fast can emit timestamps as { "start": ..., "end": ... }
        let raw = json!({
            "text": "obj timestamps",
            "chunks": [
                {
                    "text": "obj timestamps",
                    "timestamp": {"start": 0.5, "end": 1.5},
                },
            ],
        });
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert_eq!(normalized.segments.len(), 1);
        assert!((normalized.segments[0].start_sec.unwrap() - 0.5).abs() < f64::EPSILON);
        assert!((normalized.segments[0].end_sec.unwrap() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn whisper_cpp_segment_confidence_from_probability() {
        let raw = json!({
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi", "probability": 0.88},
            ],
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.segments[0].confidence, Some(0.88));
    }

    #[test]
    fn whisper_cpp_segment_confidence_from_score() {
        let raw = json!({
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi", "score": 0.77},
            ],
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.segments[0].confidence, Some(0.77));
    }

    // -----------------------------------------------------------------------
    // Consistency: same golden input through different normalizers produces
    // equivalent transcript content.
    // -----------------------------------------------------------------------

    #[test]
    fn golden_whisper_cpp_and_insanely_fast_produce_same_transcript() {
        let wcpp = load_golden_json("whisper_cpp_output.json");
        let ifw = load_golden_json("insanely_fast_output.json");

        let wcpp_normalized = normalize_whisper_cpp(&wcpp).unwrap();
        let ifw_normalized = normalize_insanely_fast(&ifw).unwrap();

        assert_eq!(
            wcpp_normalized.transcript, ifw_normalized.transcript,
            "both backends should produce the same transcript from golden files"
        );
        assert_eq!(
            wcpp_normalized.segments.len(),
            ifw_normalized.segments.len(),
            "both backends should produce the same number of segments"
        );
    }

    // -----------------------------------------------------------------------
    // Round-trip: normalize then convert to TranscriptionResult
    // -----------------------------------------------------------------------

    #[test]
    fn golden_whisper_cpp_full_round_trip() {
        let raw = load_golden_json("whisper_cpp_output.json");
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        let result = to_transcription_result(normalized, BackendKind::WhisperCpp);
        assert_eq!(result.backend, BackendKind::WhisperCpp);
        assert!(!result.transcript.is_empty());
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.language.as_deref(), Some("en"));
        assert!(result.acceleration.is_none());
        assert!(result.artifact_paths.is_empty());
    }

    #[test]
    fn golden_insanely_fast_full_round_trip() {
        let raw = load_golden_json("insanely_fast_output.json");
        let normalized = normalize_insanely_fast(&raw).unwrap();
        let result = to_transcription_result(normalized, BackendKind::InsanelyFast);
        assert_eq!(result.backend, BackendKind::InsanelyFast);
        assert!(!result.transcript.is_empty());
        assert_eq!(result.segments.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Multiple segments with varied fields
    // -----------------------------------------------------------------------

    #[test]
    fn whisper_cpp_many_segments() {
        let raw = json!({
            "text": "a b c d",
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "a"},
                {"start": 0.5, "end": 1.0, "text": "b"},
                {"start": 1.0, "end": 1.5, "text": "c"},
                {"start": 1.5, "end": 2.0, "text": "d"},
            ],
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.segments.len(), 4);
        let texts: Vec<&str> = normalized
            .segments
            .iter()
            .map(|s| s.text.as_str())
            .collect();
        assert_eq!(texts, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn insanely_fast_mixed_word_and_chunk_level() {
        // Some chunks have words, some do not
        let raw = json!({
            "text": "mixed",
            "chunks": [
                {
                    "text": "word level",
                    "timestamp": [0.0, 1.0],
                    "words": [
                        {"word": "word", "start": 0.0, "end": 0.5},
                        {"word": "level", "start": 0.5, "end": 1.0},
                    ],
                },
            ],
        });
        let normalized = normalize_insanely_fast(&raw).unwrap();
        // Word-level extraction should produce individual word segments
        assert_eq!(normalized.segments.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Unicode content
    // -----------------------------------------------------------------------

    #[test]
    fn whisper_cpp_unicode_transcript() {
        let raw = json!({
            "text": "日本語のテスト",
            "segments": [{"start": 0.0, "end": 1.0, "text": "日本語のテスト"}],
            "language": "ja",
        });
        let normalized = normalize_whisper_cpp(&raw).unwrap();
        assert_eq!(normalized.transcript, "日本語のテスト");
        assert_eq!(normalized.language.as_deref(), Some("ja"));
    }

    #[test]
    fn insanely_fast_unicode_transcript() {
        let raw = json!({
            "text": "Bonjour le monde",
            "chunks": [{"text": "Bonjour le monde", "timestamp": [0.0, 2.0]}],
            "language": "fr",
        });
        let normalized = normalize_insanely_fast(&raw).unwrap();
        assert_eq!(normalized.transcript, "Bonjour le monde");
        assert_eq!(normalized.language.as_deref(), Some("fr"));
    }
}
