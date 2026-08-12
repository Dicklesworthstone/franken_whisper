//! Minimal shim for the parent crate's `model` module.
//!
//! `native_engine` uses exactly one item from `crate::model`:
//! [`TranscriptionSegment`]. The canonical definition lives in
//! `../../src/model.rs` (search for `pub struct TranscriptionSegment`); that
//! module also drags `clap::ValueEnum` and the diarization model surface,
//! neither of which is wasm-portable, so the one struct is mirrored here
//! field-for-field. If the canonical struct changes shape, the serialized
//! output of the wasm demo drifts from the native CLI — keep them identical.

use serde::{Deserialize, Serialize};

/// Mirror of `franken_whisper::model::TranscriptionSegment` (canonical).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionSegment {
    pub start_sec: Option<f64>,
    pub end_sec: Option<f64>,
    pub text: String,
    pub speaker: Option<String>,
    /// ASR token/text confidence. Speaker assignment confidence lives on the
    /// diarization turn in the native crate.
    pub confidence: Option<f64>,
}

/// Mirror of `franken_whisper::model::MAX_SPEAKER_REF_BYTES` (canonical).
pub const MAX_SPEAKER_REF_BYTES: usize = 256;

/// Mirror of `franken_whisper::model::DiarizationTurn` (canonical).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiarizationTurn {
    pub start_ms: u64,
    pub end_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change_confidence: Option<f64>,
    pub overlap_suspected: bool,
    pub hard_hint_attributed: bool,
}

/// Mirror of `franken_whisper::model::SpeakerAttributedSegment` (canonical).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeakerAttributedSegment {
    /// Run start in seconds (first timed segment of the run), if any.
    pub start_sec: Option<f64>,
    /// Run end in seconds (last timed segment of the run), if any.
    pub end_sec: Option<f64>,
    /// Projected speaker reference, or `None` for an unknown run.
    pub speaker: Option<String>,
    /// Space-joined text of the run's segments, byte-faithful.
    pub text: String,
    /// Number of transcript segments merged into this run.
    pub segment_count: usize,
    /// Duration-weighted mean `speaker_confidence` of this speaker's turns
    /// overlapping the run, when computable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_confidence: Option<f64>,
}
