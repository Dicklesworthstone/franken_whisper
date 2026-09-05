//! Shared incremental-transcript emission decisions.
//!
//! The CLI live driver and native Apple boundary both decode with the same
//! engine.  This module keeps the consequential part of that contract shared
//! too: an AlignAtt step may expose a mutable tail, but only an attention-safe
//! prefix becomes append-only committed text.  Platform hosts own capture and
//! session clocks; they do not reimplement this policy.

use crate::native_engine::decode::DecodeOutput;

/// One incremental decode's durable prefix and mutable tail.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LivePolicyDecision {
    pub commit_text: String,
    pub commit_through_sec: Option<f64>,
    pub partial_tail: Option<String>,
    pub commit_tokens: u64,
    pub commit_confidence: Option<f64>,
    pub holdback: bool,
}

fn decision(out: &DecodeOutput, committable: usize) -> LivePolicyDecision {
    let mut result = LivePolicyDecision::default();
    let count = committable.min(out.segments.len());
    let fresh = &out.segments[..count];
    if !fresh.is_empty() {
        result.commit_text = fresh
            .iter()
            .map(|segment| segment.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let confidences: Vec<f64> = fresh
            .iter()
            .filter_map(|segment| segment.confidence)
            .collect();
        result.commit_confidence = if confidences.is_empty() {
            None
        } else {
            Some(confidences.iter().sum::<f64>() / confidences.len() as f64)
        };
        result.commit_through_sec = fresh
            .iter()
            .filter_map(|segment| segment.end_sec)
            .next_back();
        // Preserve the published v1 segment-grain accounting used by
        // `fw robot listen`. Token/word-grain commits are a separate contract.
        result.commit_tokens = fresh.len() as u64;
    }

    let tail = out.segments[count..]
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    result.partial_tail = (!tail.is_empty()).then_some(tail);
    result
}

fn safe_time_sec(out: &DecodeOutput, slice_sec: f64, holdback_frames: u32) -> f64 {
    let slice_frames = (slice_sec.max(0.0) * 50.0) as u32;
    let limit = slice_frames.saturating_sub(holdback_frames.max(1));
    let mut safe_frame = 0_u32;
    for window in &out.windows {
        for tap in &window.token_attn {
            if tap.attn_frame > limit {
                return f64::from(safe_frame) * 0.02;
            }
            safe_frame = safe_frame.max(tap.attn_frame);
        }
    }
    f64::from(safe_frame) * 0.02
}

/// Apply the production AlignAtt live-step contract to one uncommitted audio
/// slice. The final decoded segment is always mutable while audio continues.
///
/// The final-segment guard is intentional even when its attention appears
/// safe. A truncated campaign slice once decoded a confident sentence close
/// while the real audio continued; committing that close made the append-only
/// stream irreparable. A segment becomes durable only after the decoder has
/// started another segment behind it. Low-confidence/no-speech slices likewise
/// expose their text only as a mutable tail.
#[must_use]
pub fn alignatt_step(
    out: &DecodeOutput,
    slice_sec: f64,
    holdback_frames: u32,
) -> LivePolicyDecision {
    let no_speech = out
        .windows
        .iter()
        .map(|window| window.no_speech_prob)
        .fold(0.0_f64, f64::max);
    let avg_logprob = out
        .windows
        .iter()
        .map(|window| window.avg_logprob)
        .fold(0.0_f64, f64::min);
    if no_speech > 0.6 || avg_logprob < -1.0 {
        let mut held = decision(out, 0);
        held.holdback = true;
        return held;
    }

    let safe = safe_time_sec(out, slice_sec, holdback_frames);
    let committable = out
        .segments
        .iter()
        .take(out.segments.len().saturating_sub(1))
        .take_while(|segment| segment.end_sec.is_some_and(|end| end <= safe))
        .count();
    decision(out, committable)
}

/// Audio has closed: commit every decoded segment and clear the mutable tail.
#[must_use]
pub fn finalize(out: &DecodeOutput) -> LivePolicyDecision {
    decision(out, out.segments.len())
}
