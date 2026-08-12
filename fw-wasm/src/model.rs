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
