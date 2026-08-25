//! Projection-fusion of an independent diarization turn timeline onto
//! transcript segments (projection-fusion-v1, bd-d4py), plus the merged
//! speaker-attributed segment builder.
//!
//! Relocated VERBATIM from `diarization.rs` (which re-exports everything, so
//! call sites and type identities are unchanged) into this dependency-light
//! module: it uses only the shared model/error types and the native decoder's
//! content-free DTW timing observations — no ECAPA/acoustic machinery — so the
//! wasm build (bd-m2jm) can mount the exact fusion and word-normalization code
//! the CLI ships instead of mirroring it.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{FwError, FwResult};
use crate::model::{DiarizationTurn, TranscriptionSegment};
use crate::native_engine::dtw::WordTiming;

/// The canonical projection timestamp epsilon (seconds). Mirrors
/// `crate::conformance::CANONICAL_PROJECTION_EPSILON_SEC`; the parent crate
/// asserts the two stay equal (`projection_epsilon_matches_conformance`), and
/// the constant lives here so this module carries no conformance dependency.
pub const CANONICAL_PROJECTION_EPSILON_SEC: f64 = 1e-6;
/// Smallest positive interval retained for a canonical DTW word unit.
///
/// The parent crate asserts this stays equal to the conformance contract; the
/// value lives here as well so the browser mounts the exact normalizer without
/// pulling in the native conformance module.
pub const CANONICAL_PROJECTION_MIN_DURATION_SEC: f64 = 0.001;

/// Lossless convenience projection of an independent acoustic turn timeline.
///
/// `segments` retain every input segment's text bytes, order, and ASR
/// confidence. The index lists carry evidence that does not fit in the legacy
/// [`TranscriptionSegment`] shape.
#[derive(Debug, Clone)]
pub struct DiarizationProjection {
    pub segments: Vec<TranscriptionSegment>,
    pub mixed_speaker_segment_indices: Vec<usize>,
    pub overlap_suspected_segment_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProjectionUnitProvenance {
    DecoderWordTimestamp,
    SegmentInterpolationFallback,
    SegmentGeometryFallback,
}

impl ProjectionUnitProvenance {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::DecoderWordTimestamp => "decoder_word_timestamp",
            Self::SegmentInterpolationFallback => "segment_interpolation_fallback",
            Self::SegmentGeometryFallback => "segment_geometry_fallback",
        }
    }

    pub(crate) const fn supported_labels() -> [&'static str; 3] {
        [
            Self::DecoderWordTimestamp.as_str(),
            Self::SegmentInterpolationFallback.as_str(),
            Self::SegmentGeometryFallback.as_str(),
        ]
    }
}

/// One unambiguous transcript unit accepted by acoustic projection.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CanonicalProjectionUnit {
    pub(crate) start_sec: f64,
    pub(crate) end_sec: f64,
    pub(crate) text: String,
    pub(crate) confidence: Option<f64>,
    pub(crate) source_segment_index: usize,
    pub(crate) source_word_start_index: usize,
    pub(crate) source_word_end_index: usize,
    pub(crate) provenance: ProjectionUnitProvenance,
    pub(crate) was_clamped: bool,
    pub(crate) was_expanded: bool,
}

impl CanonicalProjectionUnit {
    fn as_segment(&self) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: Some(self.start_sec),
            end_sec: Some(self.end_sec),
            text: self.text.clone(),
            speaker: None,
            confidence: self.confidence,
        }
    }
}

/// Canonical DTW word units plus path-free normalization counters.
#[derive(Debug, Clone)]
pub(crate) struct DtwProjectionNormalization {
    pub(crate) segments: Vec<TranscriptionSegment>,
    #[cfg(test)]
    pub(crate) canonical_units: Vec<CanonicalProjectionUnit>,
    pub(crate) input_timed_segments: usize,
    pub(crate) decoder_word_units: usize,
    pub(crate) interpolated_fallback_units: usize,
    pub(crate) segment_geometry_fallback_units: usize,
    pub(crate) interpolated_fallback_segments: usize,
    pub(crate) segment_geometry_fallback_segments: usize,
    pub(crate) clamped_units: usize,
    pub(crate) expanded_units: usize,
    pub(crate) word_aligned_safe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionNormalizationErrorReason {
    ExtraTimingSegments,
    ParentTimestampPair,
    ParentTimestampNonFinite,
    ParentTimestampNegative,
    ParentDuration,
    ParentOverlap,
    WordTimestampNonFinite,
    WordTimestampNegative,
    WordTimestampReversed,
    WordOrderReversed,
    WordOverlap,
    CanonicalInvariant,
}

impl ProjectionNormalizationErrorReason {
    const fn code(self) -> &'static str {
        match self {
            Self::ExtraTimingSegments => "FW-DTW-PROJECTION-EXTRA-SEGMENTS",
            Self::ParentTimestampPair => "FW-DTW-PROJECTION-PARENT-PAIR",
            Self::ParentTimestampNonFinite => "FW-DTW-PROJECTION-PARENT-NONFINITE",
            Self::ParentTimestampNegative => "FW-DTW-PROJECTION-PARENT-NEGATIVE",
            Self::ParentDuration => "FW-DTW-PROJECTION-PARENT-DURATION",
            Self::ParentOverlap => "FW-DTW-PROJECTION-PARENT-OVERLAP",
            Self::WordTimestampNonFinite => "FW-DTW-PROJECTION-WORD-NONFINITE",
            Self::WordTimestampNegative => "FW-DTW-PROJECTION-WORD-NEGATIVE",
            Self::WordTimestampReversed => "FW-DTW-PROJECTION-WORD-REVERSED",
            Self::WordOrderReversed => "FW-DTW-PROJECTION-WORD-ORDER",
            Self::WordOverlap => "FW-DTW-PROJECTION-WORD-OVERLAP",
            Self::CanonicalInvariant => "FW-DTW-PROJECTION-CANONICAL-INVARIANT",
        }
    }
}

fn projection_normalization_error(
    reason: ProjectionNormalizationErrorReason,
    segment_index: usize,
    word_index: Option<usize>,
    detail: impl std::fmt::Display,
) -> FwError {
    let word = word_index.map_or_else(String::new, |index| format!(" word={index}"));
    FwError::InvalidRequest(format!(
        "dtw projection normalization [{}]: segment={segment_index}{word} {detail}",
        reason.code()
    ))
}

/// Normalize decoder-grid DTW observations into positive, ordered half-open
/// word units shared by native and browser diarization projection.
pub(crate) fn normalize_dtw_projection_units<C>(
    engine_segments: &[TranscriptionSegment],
    word_timings: &[Vec<WordTiming>],
    mut checkpoint: C,
) -> FwResult<DtwProjectionNormalization>
where
    C: FnMut() -> FwResult<()>,
{
    if word_timings.len() > engine_segments.len() {
        return Err(projection_normalization_error(
            ProjectionNormalizationErrorReason::ExtraTimingSegments,
            engine_segments.len(),
            None,
            format_args!(
                "received {} timing vectors for {} engine segments",
                word_timings.len(),
                engine_segments.len()
            ),
        ));
    }

    let mut canonical_units = Vec::new();
    let mut input_timed_segments = 0usize;
    let mut interpolated_fallback_segments = 0usize;
    let mut segment_geometry_fallback_segments = 0usize;
    let mut word_aligned_safe = true;
    let mut previous_parent_end = None;

    for (segment_index, segment) in engine_segments.iter().enumerate() {
        checkpoint()?;
        let (parent_start, parent_end) =
            canonical_parent_bounds(segment, segment_index, previous_parent_end)?;
        previous_parent_end = Some(parent_end);
        match word_timings.get(segment_index) {
            Some(timed) if !timed.is_empty() => {
                input_timed_segments += 1;
                validate_raw_word_geometry(timed, segment_index)?;
                normalize_timed_segment(
                    segment,
                    timed,
                    segment_index,
                    parent_start,
                    parent_end,
                    &mut canonical_units,
                    &mut segment_geometry_fallback_segments,
                    &mut word_aligned_safe,
                );
            }
            _ => append_interpolated_fallback(
                segment,
                segment_index,
                parent_start,
                parent_end,
                &mut canonical_units,
                &mut interpolated_fallback_segments,
                &mut segment_geometry_fallback_segments,
                &mut word_aligned_safe,
            ),
        }
    }

    validate_canonical_projection_units(&canonical_units)?;
    let mut decoder_word_units = 0usize;
    let mut interpolated_fallback_units = 0usize;
    let mut segment_geometry_fallback_units = 0usize;
    let mut clamped_units = 0usize;
    let mut expanded_units = 0usize;
    for unit in &canonical_units {
        match unit.provenance {
            ProjectionUnitProvenance::DecoderWordTimestamp => decoder_word_units += 1,
            ProjectionUnitProvenance::SegmentInterpolationFallback => {
                interpolated_fallback_units += 1;
            }
            ProjectionUnitProvenance::SegmentGeometryFallback => {
                segment_geometry_fallback_units += 1;
            }
        }
        clamped_units += usize::from(unit.was_clamped);
        expanded_units += usize::from(unit.was_expanded);
    }
    let segments = canonical_units
        .iter()
        .map(CanonicalProjectionUnit::as_segment)
        .collect();

    Ok(DtwProjectionNormalization {
        segments,
        #[cfg(test)]
        canonical_units,
        input_timed_segments,
        decoder_word_units,
        interpolated_fallback_units,
        segment_geometry_fallback_units,
        interpolated_fallback_segments,
        segment_geometry_fallback_segments,
        clamped_units,
        expanded_units,
        word_aligned_safe,
    })
}

fn canonical_parent_bounds(
    segment: &TranscriptionSegment,
    segment_index: usize,
    previous_parent_end: Option<f64>,
) -> FwResult<(f64, f64)> {
    let (mut start, end) = match (segment.start_sec, segment.end_sec) {
        (Some(start), Some(end)) => (start, end),
        _ => {
            return Err(projection_normalization_error(
                ProjectionNormalizationErrorReason::ParentTimestampPair,
                segment_index,
                None,
                "parent segment requires paired timestamps",
            ));
        }
    };
    if !start.is_finite() || !end.is_finite() {
        return Err(projection_normalization_error(
            ProjectionNormalizationErrorReason::ParentTimestampNonFinite,
            segment_index,
            None,
            "parent timestamps must be finite",
        ));
    }
    if start < 0.0 || end < 0.0 {
        return Err(projection_normalization_error(
            ProjectionNormalizationErrorReason::ParentTimestampNegative,
            segment_index,
            None,
            "parent timestamps must be non-negative",
        ));
    }
    if let Some(previous_end) = previous_parent_end
        && start < previous_end
    {
        if start + CANONICAL_PROJECTION_EPSILON_SEC < previous_end {
            return Err(projection_normalization_error(
                ProjectionNormalizationErrorReason::ParentOverlap,
                segment_index,
                None,
                "parent segment materially overlaps its predecessor",
            ));
        }
        start = previous_end;
    }
    if end <= start {
        return Err(projection_normalization_error(
            ProjectionNormalizationErrorReason::ParentDuration,
            segment_index,
            None,
            "parent segment must retain positive duration after adjacency normalization",
        ));
    }
    Ok((start, end))
}

fn validate_raw_word_geometry(words: &[WordTiming], segment_index: usize) -> FwResult<()> {
    let mut previous_start = None;
    let mut previous_end = None;
    for (word_index, word) in words.iter().enumerate() {
        if !word.start_sec.is_finite() || !word.end_sec.is_finite() {
            return Err(projection_normalization_error(
                ProjectionNormalizationErrorReason::WordTimestampNonFinite,
                segment_index,
                Some(word_index),
                "word timestamps must be finite",
            ));
        }
        if word.start_sec < 0.0 || word.end_sec < 0.0 {
            return Err(projection_normalization_error(
                ProjectionNormalizationErrorReason::WordTimestampNegative,
                segment_index,
                Some(word_index),
                "word timestamps must be non-negative",
            ));
        }
        if word.end_sec < word.start_sec {
            return Err(projection_normalization_error(
                ProjectionNormalizationErrorReason::WordTimestampReversed,
                segment_index,
                Some(word_index),
                "word end precedes word start",
            ));
        }
        if let Some(prior_start) = previous_start
            && word.start_sec + CANONICAL_PROJECTION_EPSILON_SEC < prior_start
        {
            return Err(projection_normalization_error(
                ProjectionNormalizationErrorReason::WordOrderReversed,
                segment_index,
                Some(word_index),
                "word starts are not monotonic",
            ));
        }
        if let Some(prior_end) = previous_end
            && word.start_sec + CANONICAL_PROJECTION_EPSILON_SEC < prior_end
        {
            return Err(projection_normalization_error(
                ProjectionNormalizationErrorReason::WordOverlap,
                segment_index,
                Some(word_index),
                "word materially overlaps its predecessor",
            ));
        }
        previous_start = Some(word.start_sec);
        previous_end = Some(word.end_sec);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn normalize_timed_segment(
    segment: &TranscriptionSegment,
    timed: &[WordTiming],
    segment_index: usize,
    parent_start: f64,
    parent_end: f64,
    out: &mut Vec<CanonicalProjectionUnit>,
    segment_geometry_fallback_segments: &mut usize,
    word_aligned_safe: &mut bool,
) {
    let required_duration = timed.len() as f64 * CANONICAL_PROJECTION_MIN_DURATION_SEC;
    if parent_end - parent_start < required_duration {
        append_segment_geometry_fallback(
            segment,
            timed,
            segment_index,
            parent_start,
            parent_end,
            out,
            segment_geometry_fallback_segments,
            word_aligned_safe,
        );
        return;
    }

    let mut cursor = parent_start;
    for (word_index, word) in timed.iter().enumerate() {
        let remaining = timed.len() - word_index - 1;
        let latest_end = parent_end - remaining as f64 * CANONICAL_PROJECTION_MIN_DURATION_SEC;
        let latest_start =
            parent_end - (remaining + 1) as f64 * CANONICAL_PROJECTION_MIN_DURATION_SEC;
        let desired_start = word.start_sec.clamp(parent_start, parent_end);
        let start = desired_start.max(cursor).min(latest_start);
        let desired_end = word.end_sec.clamp(parent_start, parent_end);
        let minimum_end = if start == latest_start {
            latest_end
        } else {
            start + CANONICAL_PROJECTION_MIN_DURATION_SEC
        };
        let end = desired_end.max(minimum_end).min(latest_end);
        let was_clamped = (start - word.start_sec).abs() > CANONICAL_PROJECTION_EPSILON_SEC
            || (end - word.end_sec).abs() > CANONICAL_PROJECTION_EPSILON_SEC;
        let was_expanded = word.end_sec - word.start_sec
            < CANONICAL_PROJECTION_MIN_DURATION_SEC - CANONICAL_PROJECTION_EPSILON_SEC;
        out.push(CanonicalProjectionUnit {
            start_sec: start,
            end_sec: end,
            text: word.text.clone(),
            confidence: segment.confidence,
            source_segment_index: segment_index,
            source_word_start_index: word_index,
            source_word_end_index: word_index + 1,
            provenance: ProjectionUnitProvenance::DecoderWordTimestamp,
            was_clamped,
            was_expanded,
        });
        cursor = end;
    }
}

#[allow(clippy::too_many_arguments)]
fn append_interpolated_fallback(
    segment: &TranscriptionSegment,
    segment_index: usize,
    parent_start: f64,
    parent_end: f64,
    out: &mut Vec<CanonicalProjectionUnit>,
    interpolated_fallback_segments: &mut usize,
    segment_geometry_fallback_segments: &mut usize,
    word_aligned_safe: &mut bool,
) {
    *word_aligned_safe = false;
    let words = segment.text.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        *interpolated_fallback_segments += 1;
        return;
    }
    let duration = parent_end - parent_start;
    if duration < words.len() as f64 * CANONICAL_PROJECTION_MIN_DURATION_SEC {
        append_segment_geometry_fallback(
            segment,
            &[],
            segment_index,
            parent_start,
            parent_end,
            out,
            segment_geometry_fallback_segments,
            word_aligned_safe,
        );
        return;
    }
    *interpolated_fallback_segments += 1;

    for (word_index, word) in words.iter().enumerate() {
        let count = words.len() as f64;
        let start = parent_start + duration * word_index as f64 / count;
        let end = if word_index + 1 == words.len() {
            parent_end
        } else {
            parent_start + duration * (word_index + 1) as f64 / count
        };
        out.push(CanonicalProjectionUnit {
            start_sec: start,
            end_sec: end,
            text: (*word).to_owned(),
            confidence: segment.confidence,
            source_segment_index: segment_index,
            source_word_start_index: word_index,
            source_word_end_index: word_index + 1,
            provenance: ProjectionUnitProvenance::SegmentInterpolationFallback,
            was_clamped: false,
            was_expanded: false,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn append_segment_geometry_fallback(
    segment: &TranscriptionSegment,
    timed: &[WordTiming],
    segment_index: usize,
    parent_start: f64,
    parent_end: f64,
    out: &mut Vec<CanonicalProjectionUnit>,
    segment_geometry_fallback_segments: &mut usize,
    word_aligned_safe: &mut bool,
) {
    *segment_geometry_fallback_segments += 1;
    *word_aligned_safe = false;
    let segment_text = segment.text.trim();
    let text = if segment_text.is_empty() {
        timed
            .iter()
            .map(|word| word.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        segment_text.to_owned()
    };
    out.push(CanonicalProjectionUnit {
        start_sec: parent_start,
        end_sec: parent_end,
        text,
        confidence: segment.confidence,
        source_segment_index: segment_index,
        source_word_start_index: 0,
        source_word_end_index: timed.len().max(1),
        provenance: ProjectionUnitProvenance::SegmentGeometryFallback,
        was_clamped: false,
        was_expanded: false,
    });
}

fn validate_canonical_projection_units(units: &[CanonicalProjectionUnit]) -> FwResult<()> {
    let mut previous_end = None;
    for unit in units {
        let valid = unit.start_sec.is_finite()
            && unit.end_sec.is_finite()
            && unit.start_sec >= 0.0
            && unit.end_sec > unit.start_sec
            && unit.source_word_end_index > unit.source_word_start_index
            && previous_end.is_none_or(|end| unit.start_sec >= end);
        if !valid {
            return Err(projection_normalization_error(
                ProjectionNormalizationErrorReason::CanonicalInvariant,
                unit.source_segment_index,
                Some(unit.source_word_start_index),
                "normalizer produced a non-finite, non-positive, reversed, or overlapping unit",
            ));
        }
        previous_end = Some(unit.end_sec);
    }
    Ok(())
}

pub(crate) fn minimum_optional_confidence(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        _ => None,
    }
}

pub(crate) fn maximum_optional_confidence(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

/// Project an immutable acoustic turn timeline onto existing transcript
/// segments without modifying text or ASR confidence.
///
/// When `word_aligned` is true, each input segment is treated as a legal DTW
/// word unit. Otherwise a segment receives a label only when one speaker owns
/// at least 70% of its duration at calibrated-enough confidence.
pub fn project_diarization_onto_segments(
    segments: &[TranscriptionSegment],
    turns: &[DiarizationTurn],
    word_aligned: bool,
) -> FwResult<DiarizationProjection> {
    validate_diarization_turns(turns)?;
    let canonical_segments =
        canonicalize_zero_duration_projection_segments(segments, word_aligned)?;
    validate_projection_segments(&canonical_segments, word_aligned)?;

    let mut projected = Vec::with_capacity(segments.len());
    let mut mixed = Vec::new();
    let mut overlaps = Vec::new();
    for (index, segment) in canonical_segments.iter().enumerate() {
        let mut output = segment.clone();
        output.speaker = None;
        let Some((start_sec, end_sec)) = segment.start_sec.zip(segment.end_sec) else {
            if distinct_known_speakers(turns) > 1 {
                mixed.push(index);
            }
            projected.push(output);
            continue;
        };
        let overlapping = overlapping_turns(start_sec, end_sec, turns);
        if overlapping.iter().any(|(_, turn)| turn.overlap_suspected) {
            overlaps.push(index);
        }
        if word_aligned {
            output.speaker = choose_word_speaker(&overlapping);
        } else {
            let (speaker, is_mixed) =
                choose_dominant_segment_speaker(start_sec, end_sec, &overlapping);
            output.speaker = speaker;
            if is_mixed {
                mixed.push(index);
            }
        }
        projected.push(output);
    }

    // Fusion pass (bd-d4py, projection-fusion-v1): the conservative pass above
    // leaves a segment unlabeled when its evidence misses a gate (low turn
    // confidence, sub-dominant ownership) or when the segment sits in a gap
    // between turns. The turn evidence to label those segments is present in
    // the same payload; use it instead of returning `null`:
    //   (a) any segment overlapping a labeled turn takes the max-overlap
    //       labeled turn, and
    //   (b) a timed segment in a turn gap takes the nearest labeled turn
    //       within `NEIGHBOR_FALLBACK_MAX_GAP_SEC`.
    // Text, timing, ASR confidence, and the report's turn timeline are never
    // modified; only the projected per-segment speaker field is filled.
    fill_unlabeled_segments_from_turns(&mut projected, &canonical_segments, turns);

    // Boundary snapping (bd-d4py): at word granularity, speaker changes that
    // fall mid-clause are re-anchored to the nearest sentence-final
    // punctuation boundary within a small window, using the transcript's own
    // punctuation as the oracle. Measured on real 2-person audio this beats
    // every global time-shift candidate.
    if word_aligned {
        snap_speaker_changes_to_punctuation(&mut projected);
    }

    Ok(DiarizationProjection {
        segments: projected,
        mixed_speaker_segment_indices: mixed,
        overlap_suspected_segment_indices: overlaps,
    })
}

/// Maximum gap (seconds) between an unlabeled segment and the nearest labeled
/// turn for neighbor-fallback attribution (projection-fusion-v1).
const NEIGHBOR_FALLBACK_MAX_GAP_SEC: f64 = 2.0;

/// Number of words on each side of a projected speaker change searched for a
/// sentence-final punctuation boundary to snap to (projection-fusion-v1).
const PUNCTUATION_SNAP_WINDOW_WORDS: usize = 4;

/// Fill projected segments the conservative pass left unlabeled, using the
/// turn timeline that shipped in the same report (bd-d4py).
fn fill_unlabeled_segments_from_turns(
    projected: &mut [TranscriptionSegment],
    canonical_segments: &[TranscriptionSegment],
    turns: &[DiarizationTurn],
) {
    for (index, output) in projected.iter_mut().enumerate() {
        if output.speaker.is_some() {
            continue;
        }
        let Some((start_sec, end_sec)) = canonical_segments
            .get(index)
            .and_then(|segment| segment.start_sec.zip(segment.end_sec))
        else {
            continue;
        };
        // (a) Best labeled overlap, regardless of the primary pass's
        // confidence/dominance gates.
        let overlapping = overlapping_turns(start_sec, end_sec, turns);
        let mut ranked = overlapping
            .iter()
            .filter(|(_, turn)| turn.speaker_ref.is_some())
            .collect::<Vec<_>>();
        ranked.sort_by(|(left_overlap, left), (right_overlap, right)| {
            right_overlap
                .total_cmp(left_overlap)
                .then(left.start_ms.cmp(&right.start_ms))
                .then(left.speaker_ref.cmp(&right.speaker_ref))
        });
        if let Some((_, best)) = ranked.first() {
            output.speaker = best.speaker_ref.clone();
            continue;
        }
        // (b) Nearest labeled neighbor turn within the bounded gap. Hard-hint
        // turns are excluded: a `hard_must_link` interval asserts caller
        // identity only for its own audio, and extrapolating it into a gap
        // would invent attribution the caller never made (§6.1).
        let mut nearest: Option<(f64, &DiarizationTurn)> = None;
        for turn in turns {
            if turn.speaker_ref.is_none() || turn.hard_hint_attributed {
                continue;
            }
            let turn_start = turn.start_ms as f64 / 1_000.0;
            let turn_end = turn.end_ms as f64 / 1_000.0;
            let gap = if turn_end <= start_sec {
                start_sec - turn_end
            } else if turn_start >= end_sec {
                turn_start - end_sec
            } else {
                0.0
            };
            let closer = nearest.is_none_or(|(best_gap, best_turn)| {
                gap < best_gap || (gap == best_gap && turn.start_ms < best_turn.start_ms)
            });
            if gap <= NEIGHBOR_FALLBACK_MAX_GAP_SEC && closer {
                nearest = Some((gap, turn));
            }
        }
        if let Some((_, turn)) = nearest {
            output.speaker = turn.speaker_ref.clone();
        }
    }
}

/// Whether a word ends a sentence for boundary-snapping purposes.
///
/// Trailing closing quotes/brackets are transparent so `you?"` and `done.)`
/// still count as sentence-final.
fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(['"', '\'', ')', ']', '\u{00BB}', '\u{201D}', '\u{2019}'])
        .chars()
        .last()
        .is_some_and(|c| matches!(c, '.' | '?' | '!' | '\u{2026}'))
}

/// Re-anchor projected speaker changes to sentence-final punctuation
/// boundaries within ±[`PUNCTUATION_SNAP_WINDOW_WORDS`] words (bd-d4py).
///
/// Diarization turn boundaries are quantized (80 ms lanes for Sortformer), so
/// a naive time-overlap join puts the last word of each turn on the wrong
/// speaker. The transcript's own punctuation is a better boundary oracle:
/// when a speaker change lands mid-clause but a sentence ends within the
/// window, move the change to just after that sentence end. Labels are the
/// only thing rewritten — never text, timing, or confidence.
fn snap_speaker_changes_to_punctuation(projected: &mut [TranscriptionSegment]) {
    if projected.len() < 2 {
        return;
    }
    // A change at position `i` means projected[i-1].speaker != projected[i].speaker.
    let mut i = 1usize;
    while i < projected.len() {
        let differs = projected[i - 1].speaker != projected[i].speaker
            && projected[i - 1].speaker.is_some()
            && projected[i].speaker.is_some();
        if !differs {
            i += 1;
            continue;
        }
        // Already on a sentence boundary: nothing to snap.
        if ends_sentence(&projected[i - 1].text) {
            i += 1;
            continue;
        }
        // Candidate boundary positions j (change occurs before word j) where
        // word j-1 ends a sentence, within the window. Moving the boundary
        // may only relabel words that currently belong to the run being
        // shrunk, and the shrunk run must survive past the new boundary —
        // snapping moves a change, it never erases a (short) turn.
        let left_label = projected[i - 1].speaker.clone();
        let right_label = projected[i].speaker.clone();
        let lo = i.saturating_sub(PUNCTUATION_SNAP_WINDOW_WORDS).max(1);
        let hi = (i + PUNCTUATION_SNAP_WINDOW_WORDS).min(projected.len() - 1);
        let mut best: Option<usize> = None;
        for j in lo..=hi {
            if j == i || !ends_sentence(&projected[j - 1].text) {
                continue;
            }
            let movable = if j > i {
                // Shrink the right run from the left: words i..j must all be
                // the right label, and the right run must continue at j.
                projected[i..j].iter().all(|seg| seg.speaker == right_label)
                    && projected[j].speaker == right_label
            } else {
                // Shrink the left run from the right: words j..i must all be
                // the left label, and the left run must still exist before j.
                projected[j..i].iter().all(|seg| seg.speaker == left_label)
                    && projected[j - 1].speaker == left_label
            };
            if !movable {
                continue;
            }
            let better = best.is_none_or(|current| {
                j.abs_diff(i) < current.abs_diff(i)
                    || (j.abs_diff(i) == current.abs_diff(i) && j < current)
            });
            if better {
                best = Some(j);
            }
        }
        if let Some(j) = best {
            if j > i {
                for seg in &mut projected[i..j] {
                    seg.speaker = left_label.clone();
                }
            } else {
                for seg in &mut projected[j..i] {
                    seg.speaker = right_label.clone();
                }
            }
            // Continue after the snapped boundary to avoid re-processing.
            i = j.max(i) + 1;
        } else {
            i += 1;
        }
    }
}

/// Canonicalize a zero-width decoder observation for acoustic projection.
///
/// A `whisper.cpp` word split can quantize a middle observation to `[t, t]`.
/// It remains a real transcript observation, so its text, confidence, order,
/// and segment slot must survive.  It is not, however, a time interval from
/// which speaker ownership can be inferred.  Canonicalize only an otherwise
/// well-formed, monotonic zero-width pair to absent timing; projection will
/// then retain it as an explicitly unknown speaker observation.  Other bad
/// timestamp shapes remain fail-closed in [`validate_projection_segments`].
pub(crate) fn canonicalize_zero_duration_projection_segments(
    segments: &[TranscriptionSegment],
    word_aligned: bool,
) -> FwResult<Vec<TranscriptionSegment>> {
    if !word_aligned {
        return Ok(segments.to_vec());
    }

    let mut canonical = Vec::with_capacity(segments.len());
    let mut previous_timed = None;
    for segment in segments {
        let mut output = segment.clone();
        let (Some(start), Some(end)) = (segment.start_sec, segment.end_sec) else {
            canonical.push(output);
            continue;
        };
        if !start.is_finite() || !end.is_finite() || start < 0.0 || end < start {
            canonical.push(output);
            continue;
        }
        let monotonic_with_previous =
            previous_timed.is_none_or(|(previous_start, previous_end)| {
                start >= previous_start
                    && (!word_aligned || start + CANONICAL_PROJECTION_EPSILON_SEC >= previous_end)
            });
        if !monotonic_with_previous {
            return Err(FwError::InvalidRequest(
                "projection segments must have paired finite timestamps in monotonic order"
                    .to_owned(),
            ));
        }
        previous_timed = Some((start, end));

        if start == end {
            output.start_sec = None;
            output.end_sec = None;
        }
        canonical.push(output);
    }
    Ok(canonical)
}

fn validate_diarization_turns(turns: &[DiarizationTurn]) -> FwResult<()> {
    let invalid = |code: &str| {
        FwError::InvalidRequest(format!(
            "diarization.turns.{code}: diarization turns must be finite, labeled consistently, and monotonic"
        ))
    };
    let invalid_pair = |code: &str,
                        current_index: usize,
                        current: &DiarizationTurn,
                        prior_index: usize,
                        prior: &DiarizationTurn| {
        FwError::InvalidRequest(format!(
            "diarization.turns.{code}[current={current_index}:{}-{}:labeled={}:overlap={},prior={prior_index}:{}-{}:labeled={}:overlap={}]: diarization turns must be finite, labeled consistently, and monotonic",
            current.start_ms,
            current.end_ms,
            current.speaker_ref.is_some(),
            current.overlap_suspected,
            prior.start_ms,
            prior.end_ms,
            prior.speaker_ref.is_some(),
            prior.overlap_suspected,
        ))
    };
    let mut previous_key = None;
    let mut maximum_end = 0u64;
    let mut maximum_end_index = 0usize;
    let mut maximum_unmarked_end = 0u64;
    let mut maximum_unmarked_end_index = 0usize;
    let mut maximum_unlabeled_end = 0u64;
    let mut maximum_unlabeled_end_index = 0usize;
    let mut speaker_end = BTreeMap::<Option<&str>, u64>::new();
    for (index, turn) in turns.iter().enumerate() {
        let valid_confidence = |value: Option<f64>| {
            value.is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        };
        let speaker = turn.speaker_ref.as_deref();
        let turn_key = (turn.start_ms, turn.end_ms, speaker);
        let overlaps_any_turn = turn.start_ms < maximum_end;
        let overlaps_unmarked_turn = turn.start_ms < maximum_unmarked_end;
        let overlaps_unlabeled_turn = turn.start_ms < maximum_unlabeled_end;
        let overlaps_same_speaker = speaker_end
            .get(&speaker)
            .is_some_and(|end_ms| turn.start_ms < *end_ms);
        if turn.end_ms <= turn.start_ms {
            return Err(invalid("geometry"));
        }
        if previous_key.is_some_and(|key| key > turn_key) {
            return Err(invalid("ordering"));
        }
        if overlaps_same_speaker {
            return Err(invalid("same_speaker_overlap"));
        }
        if overlaps_any_turn
            && (!turn.overlap_suspected
                || overlaps_unmarked_turn
                || speaker.is_none()
                || overlaps_unlabeled_turn)
        {
            let prior_index = if overlaps_unmarked_turn {
                maximum_unmarked_end_index
            } else if overlaps_unlabeled_turn {
                maximum_unlabeled_end_index
            } else {
                maximum_end_index
            };
            return Err(invalid_pair(
                "overlap_provenance",
                index,
                turn,
                prior_index,
                &turns[prior_index],
            ));
        }
        if turn.speaker_ref.as_ref().is_some_and(|speaker| {
            speaker.trim().is_empty() || speaker.len() > crate::model::MAX_SPEAKER_REF_BYTES
        }) {
            return Err(invalid("speaker_ref"));
        }
        if turn.speaker_ref.is_none() && turn.speaker_confidence.is_some() {
            return Err(invalid("anonymous_confidence"));
        }
        if turn.hard_hint_attributed
            && (turn.speaker_ref.is_none()
                || turn.speaker_confidence.map(f64::to_bits) != Some(1.0_f64.to_bits()))
        {
            return Err(invalid("hard_hint"));
        }
        if !valid_confidence(turn.speaker_confidence) || !valid_confidence(turn.change_confidence) {
            return Err(invalid("confidence"));
        }
        previous_key = Some(turn_key);
        if turn.end_ms > maximum_end {
            maximum_end = turn.end_ms;
            maximum_end_index = index;
        }
        if !turn.overlap_suspected && turn.end_ms > maximum_unmarked_end {
            maximum_unmarked_end = turn.end_ms;
            maximum_unmarked_end_index = index;
        }
        if speaker.is_none() && turn.end_ms > maximum_unlabeled_end {
            maximum_unlabeled_end = turn.end_ms;
            maximum_unlabeled_end_index = index;
        }
        speaker_end
            .entry(speaker)
            .and_modify(|end_ms| *end_ms = (*end_ms).max(turn.end_ms))
            .or_insert(turn.end_ms);
    }
    Ok(())
}

pub(crate) fn validate_projection_segments(
    segments: &[TranscriptionSegment],
    word_aligned: bool,
) -> FwResult<()> {
    let mut previous_start = 0.0_f64;
    let mut previous_end = 0.0_f64;
    for (index, segment) in segments.iter().enumerate() {
        match (segment.start_sec, segment.end_sec) {
            (Some(start), Some(end))
                if start.is_finite()
                    && end.is_finite()
                    && start >= 0.0
                    && end > start
                    && (index == 0 || start >= previous_start)
                    && (!word_aligned
                        || index == 0
                        || start + CANONICAL_PROJECTION_EPSILON_SEC >= previous_end) =>
            {
                previous_start = start;
                previous_end = end;
            }
            (None, None) => {}
            _ => {
                return Err(FwError::InvalidRequest(
                    "projection segments must have paired finite timestamps in monotonic order"
                        .to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn overlapping_turns(
    start_sec: f64,
    end_sec: f64,
    turns: &[DiarizationTurn],
) -> Vec<(f64, &DiarizationTurn)> {
    turns
        .iter()
        .filter_map(|turn| {
            let turn_start = turn.start_ms as f64 / 1_000.0;
            let turn_end = turn.end_ms as f64 / 1_000.0;
            let overlap = end_sec.min(turn_end) - start_sec.max(turn_start);
            (overlap > 0.0).then_some((overlap, turn))
        })
        .collect()
}

fn choose_word_speaker(overlapping: &[(f64, &DiarizationTurn)]) -> Option<String> {
    const MIN_SPEAKER_CONFIDENCE: f64 = 0.30;
    let mut ranked = overlapping
        .iter()
        .filter(|(_, turn)| turn.speaker_ref.is_some())
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_overlap, left), (right_overlap, right)| {
        right_overlap
            .total_cmp(left_overlap)
            .then(left.start_ms.cmp(&right.start_ms))
            .then(left.speaker_ref.cmp(&right.speaker_ref))
    });
    let (_, best) = ranked.first().copied()?;
    let confidence = best.speaker_confidence.unwrap_or(0.0);
    if confidence < MIN_SPEAKER_CONFIDENCE {
        return None;
    }
    best.speaker_ref.clone()
}

fn choose_dominant_segment_speaker(
    start_sec: f64,
    end_sec: f64,
    overlapping: &[(f64, &DiarizationTurn)],
) -> (Option<String>, bool) {
    const MIN_DOMINANCE: f64 = 0.70;
    const MIN_SPEAKER_CONFIDENCE: f64 = 0.30;
    let mut totals = BTreeMap::<String, (f64, f64)>::new();
    for (duration, turn) in overlapping {
        let Some(speaker) = &turn.speaker_ref else {
            continue;
        };
        let entry = totals.entry(speaker.clone()).or_default();
        entry.0 += duration;
        entry.1 += duration * turn.speaker_confidence.unwrap_or(0.0);
    }
    let mut ranked = totals.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|(left_label, left), (right_label, right)| {
        right.0.total_cmp(&left.0).then(left_label.cmp(right_label))
    });
    let Some((speaker, (duration, weighted_confidence))) = ranked.first() else {
        return (None, false);
    };
    let segment_duration = end_sec - start_sec;
    let confidence = weighted_confidence / duration.max(f64::EPSILON);
    let dominant = duration / segment_duration >= MIN_DOMINANCE
        && confidence >= MIN_SPEAKER_CONFIDENCE
        && ranked
            .get(1)
            .is_none_or(|(_, (second_duration, _))| duration > second_duration);
    if dominant {
        (Some(speaker.clone()), false)
    } else {
        (None, ranked.len() > 1)
    }
}

/// Build the merged speaker-attributed view from projected segments
/// (bd-d4py): consecutive segments with the same projected speaker collapse
/// into one run with space-joined, byte-faithful text.
///
/// `turns` supply the duration-weighted mean speaker confidence per run when
/// the run is timed and labeled; unknown runs are retained (speaker `None`)
/// so the merged view covers the complete transcript.
pub fn build_speaker_attributed_segments(
    segments: &[TranscriptionSegment],
    turns: &[DiarizationTurn],
) -> Vec<crate::model::SpeakerAttributedSegment> {
    let mut merged: Vec<crate::model::SpeakerAttributedSegment> = Vec::new();
    for segment in segments {
        match merged.last_mut() {
            Some(run) if run.speaker == segment.speaker => {
                if !segment.text.is_empty() {
                    if !run.text.is_empty() {
                        run.text.push(' ');
                    }
                    run.text.push_str(&segment.text);
                }
                run.segment_count += 1;
                if segment.end_sec.is_some() {
                    run.end_sec = segment.end_sec;
                }
                if run.start_sec.is_none() {
                    run.start_sec = segment.start_sec;
                }
            }
            _ => {
                merged.push(crate::model::SpeakerAttributedSegment {
                    start_sec: segment.start_sec,
                    end_sec: segment.end_sec,
                    speaker: segment.speaker.clone(),
                    text: segment.text.clone(),
                    segment_count: 1,
                    speaker_confidence: None,
                });
            }
        }
    }
    for run in &mut merged {
        let (Some(speaker), Some(start_sec), Some(end_sec)) =
            (run.speaker.as_deref(), run.start_sec, run.end_sec)
        else {
            continue;
        };
        let mut weighted = 0.0f64;
        let mut duration = 0.0f64;
        for turn in turns {
            if turn.speaker_ref.as_deref() != Some(speaker) {
                continue;
            }
            let turn_start = turn.start_ms as f64 / 1_000.0;
            let turn_end = turn.end_ms as f64 / 1_000.0;
            let overlap = end_sec.min(turn_end) - start_sec.max(turn_start);
            if overlap > 0.0
                && let Some(confidence) = turn.speaker_confidence
            {
                weighted += overlap * confidence;
                duration += overlap;
            }
        }
        if duration > 0.0 {
            run.speaker_confidence = Some(weighted / duration);
        }
    }
    merged
}

fn distinct_known_speakers(turns: &[DiarizationTurn]) -> usize {
    turns
        .iter()
        .filter_map(|turn| turn.speaker_ref.as_deref())
        .collect::<BTreeSet<_>>()
        .len()
}

/// Map a native Sortformer diarization output onto the report-shape
/// [`DiarizationTurn`] timeline: `SPEAKER_NN` labels per active lane, mean
/// per-turn lane probability as the speaker confidence, overlap suspicion
/// from the activity overlap channel, and the canonical
/// `(start_ms, end_ms, speaker_ref)` sort. Relocated VERBATIM from
/// `orchestrator::run_native_sortformer_diarization` (which now calls this)
/// so the wasm pipeline (bd-m2jm) runs the same mapping the CLI ships.
///
/// Returns the sorted turns plus the lane→label map the caller's report
/// machinery keys on.
///
/// # Errors
///
/// `ContractViolation` on non-finite/empty turns or probability index
/// overflow; propagates `checkpoint` cancellation.
pub fn sortformer_output_to_diarization_turns(
    output: &crate::sortformer_inference::SortformerDiarizationOutput,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<(
    Vec<DiarizationTurn>,
    std::collections::BTreeMap<usize, String>,
)> {
    const FRAME_MS: u64 = 80;
    const LANES: usize = crate::sortformer_inference::SORTFORMER_SPEAKER_LANES;
    let mut active_lanes = std::collections::BTreeSet::new();
    for turn in &output.turns {
        active_lanes.insert(turn.speaker);
    }
    let labels = active_lanes
        .iter()
        .map(|lane| (*lane, format!("SPEAKER_{lane:02}")))
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut turns = Vec::with_capacity(output.turns.len());
    for (index, turn) in output.turns.iter().enumerate() {
        if index.is_multiple_of(1_024) {
            checkpoint()?;
        }
        let (start_ms, end_ms) = finite_seconds_interval_to_ms(
            f64::from(turn.start_seconds),
            f64::from(turn.end_seconds),
        )
        .ok_or_else(|| {
            FwError::ContractViolation(
                "native Sortformer emitted a non-finite or empty speaker turn".to_owned(),
            )
        })?;
        let start_frame = usize::try_from(start_ms / FRAME_MS).unwrap_or(usize::MAX);
        let end_frame = usize::try_from(end_ms.div_ceil(FRAME_MS)).unwrap_or(usize::MAX);
        let frame_end = end_frame.min(output.frames);
        let frame_start = start_frame.min(frame_end);
        let mut probability_sum = 0.0_f64;
        let mut probability_count = 0_u64;
        let mut overlap_suspected = false;
        for frame in frame_start..frame_end {
            let offset = frame
                .checked_mul(LANES)
                .and_then(|base| base.checked_add(turn.speaker))
                .ok_or_else(|| {
                    FwError::ContractViolation(
                        "native Sortformer probability index overflowed".to_owned(),
                    )
                })?;
            if let Some(probability) = output.probabilities.get(offset) {
                probability_sum += f64::from(*probability);
                probability_count = probability_count.saturating_add(1);
            }
            overlap_suspected |= output
                .activity
                .overlap
                .get(frame)
                .is_some_and(|value| *value != 0);
        }
        let speaker_confidence = if probability_count == 0 {
            0.5
        } else {
            (probability_sum / probability_count as f64).clamp(0.0, 1.0)
        };
        turns.push(DiarizationTurn {
            start_ms,
            end_ms,
            speaker_ref: labels.get(&turn.speaker).cloned(),
            speaker_confidence: Some(speaker_confidence),
            change_confidence: None,
            overlap_suspected,
            hard_hint_attributed: false,
        });
    }
    turns.sort_by(|left, right| {
        (left.start_ms, left.end_ms, left.speaker_ref.as_deref()).cmp(&(
            right.start_ms,
            right.end_ms,
            right.speaker_ref.as_deref(),
        ))
    });
    Ok((turns, labels))
}

/// Finite `(start_sec, end_sec)` → validated `(start_ms, end_ms)` interval.
/// Relocated from `orchestrator` (re-exported there) for the mapping above.
pub fn finite_seconds_interval_to_ms(start_sec: f64, end_sec: f64) -> Option<(u64, u64)> {
    if !start_sec.is_finite() || !end_sec.is_finite() || start_sec < 0.0 || end_sec <= start_sec {
        return None;
    }
    let start_ms = (start_sec * 1_000.0).round();
    let end_ms = (end_sec * 1_000.0).round();
    if start_ms > u64::MAX as f64 || end_ms > u64::MAX as f64 {
        return None;
    }
    let start_ms = start_ms as u64;
    let end_ms = end_ms as u64;
    (end_ms > start_ms).then_some((start_ms, end_ms))
}

#[cfg(test)]
mod tests {
    use crate::model::{DiarizationTurn, TranscriptionSegment};

    /// bd-6xsk regression: whisper.cpp `--split-on-word --max-segment-length`
    /// runs can emit a zero-duration middle segment (`start_ms == end_ms`).
    /// The projection must canonicalize it — text preserved, timestamps
    /// released into the explicitly-unknown-speaker lane (mixed-marked when
    /// several known speakers exist) — instead of rejecting the entire run.
    #[test]
    fn zero_duration_middle_segment_survives_projection() {
        let turn = |start_ms: u64, end_ms: u64, speaker: &str| DiarizationTurn {
            start_ms,
            end_ms,
            speaker_ref: Some(speaker.to_owned()),
            speaker_confidence: Some(0.9),
            change_confidence: None,
            overlap_suspected: false,
            hard_hint_attributed: false,
        };
        let segment =
            |start_sec: Option<f64>, end_sec: Option<f64>, text: &str| TranscriptionSegment {
                start_sec,
                end_sec,
                text: text.to_owned(),
                speaker: None,
                confidence: Some(0.8),
            };
        let segments = vec![
            segment(Some(0.0), Some(1.0), "And so my fellow"),
            // The degenerate observation: a word-aligned slice whose decoder
            // timestamps collapsed onto the same instant.
            segment(Some(1.0), Some(1.0), "Americans"),
            segment(Some(1.5), Some(3.0), "ask not"),
        ];
        let turns = vec![turn(0, 2_000, "A"), turn(2_000, 4_000, "B")];

        let projection = super::project_diarization_onto_segments(&segments, &turns, true)
            .expect("a zero-duration middle segment must not reject the run");

        assert_eq!(projection.segments.len(), 3);
        // Text and order are never modified.
        assert_eq!(projection.segments[1].text, "Americans");
        // The degenerate middle segment is canonicalized into the explicitly
        // unknown-speaker lane, not dropped and not fail-closed.
        assert_eq!(projection.segments[1].start_sec, None);
        assert_eq!(projection.segments[1].end_sec, None);
        assert!(projection.mixed_speaker_segment_indices.contains(&1));
        // Neighboring timed segments keep their timestamps and get labels.
        assert_eq!(projection.segments[0].start_sec, Some(0.0));
        assert_eq!(projection.segments[0].speaker.as_deref(), Some("A"));
        assert_eq!(projection.segments[2].end_sec, Some(3.0));
        assert_eq!(projection.segments[2].speaker.as_deref(), Some("B"));
    }
}
