use crate::error::{FwError, FwResult};
use crate::model::{ReplayEnvelope, TranscriptionSegment};

/// Canonical cross-engine timestamp drift tolerance in seconds.
///
/// This is the single source of truth. The same value (50ms) must appear in:
/// - `SegmentCompatibilityTolerance::default()` (this file)
/// - `docs/engine_compatibility_spec.md` section 1.3
///
/// Regression tests in this module verify code-docs consistency.
pub const CANONICAL_TIMESTAMP_TOLERANCE_SEC: f64 = 0.05;

#[derive(Debug, Clone, Copy)]
pub struct SegmentConformancePolicy {
    pub allow_overlap: bool,
    pub timestamp_epsilon_sec: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SegmentCompatibilityTolerance {
    pub timestamp_tolerance_sec: f64,
    pub require_text_exact: bool,
    pub require_speaker_exact: bool,
}

impl Default for SegmentCompatibilityTolerance {
    fn default() -> Self {
        Self {
            timestamp_tolerance_sec: CANONICAL_TIMESTAMP_TOLERANCE_SEC,
            require_text_exact: true,
            require_speaker_exact: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentComparisonReport {
    pub expected_segments: usize,
    pub observed_segments: usize,
    pub length_mismatch: bool,
    pub text_mismatches: usize,
    pub speaker_mismatches: usize,
    pub timestamp_violations: usize,
}

impl SegmentComparisonReport {
    #[must_use]
    pub fn within_tolerance(self) -> bool {
        !self.length_mismatch
            && self.text_mismatches == 0
            && self.speaker_mismatches == 0
            && self.timestamp_violations == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayComparisonReport {
    pub input_hash_match: bool,
    pub backend_identity_match: bool,
    pub backend_version_match: bool,
    pub output_hash_match: bool,
    pub missing_expected_fields: Vec<String>,
    pub missing_observed_fields: Vec<String>,
}

impl ReplayComparisonReport {
    #[must_use]
    pub fn within_tolerance(&self) -> bool {
        self.input_hash_match
            && self.backend_identity_match
            && self.backend_version_match
            && self.output_hash_match
            && self.missing_expected_fields.is_empty()
            && self.missing_observed_fields.is_empty()
    }
}

impl Default for SegmentConformancePolicy {
    fn default() -> Self {
        Self {
            allow_overlap: false,
            timestamp_epsilon_sec: 1e-6,
        }
    }
}

pub fn validate_segment_invariants(segments: &[TranscriptionSegment]) -> FwResult<()> {
    validate_segment_invariants_with_policy(segments, SegmentConformancePolicy::default())
}

fn conformance_err(index: usize, msg: impl std::fmt::Display) -> FwError {
    FwError::Unsupported(format!("conformance violation: segment {index} {msg}"))
}

pub fn validate_segment_invariants_with_policy(
    segments: &[TranscriptionSegment],
    policy: SegmentConformancePolicy,
) -> FwResult<()> {
    let mut previous_end: Option<f64> = None;

    for (index, segment) in segments.iter().enumerate() {
        if let Some(start) = segment.start_sec
            && !start.is_finite()
        {
            return Err(conformance_err(
                index,
                format_args!("start_sec ({start}) is not finite"),
            ));
        }

        if let Some(end) = segment.end_sec
            && !end.is_finite()
        {
            return Err(conformance_err(
                index,
                format_args!("end_sec ({end}) is not finite"),
            ));
        }

        if let Some(start) = segment.start_sec
            && start < 0.0
        {
            return Err(conformance_err(
                index,
                format_args!("start_sec ({start}) is negative"),
            ));
        }

        if let Some(end) = segment.end_sec
            && end < 0.0
        {
            return Err(conformance_err(
                index,
                format_args!("end_sec ({end}) is negative"),
            ));
        }

        if let (Some(start), Some(end)) = (segment.start_sec, segment.end_sec)
            && end + policy.timestamp_epsilon_sec < start
        {
            return Err(conformance_err(
                index,
                format_args!("end_sec ({end}) is before start_sec ({start})"),
            ));
        }

        if !policy.allow_overlap
            && let (Some(prev_end), Some(start)) = (previous_end, segment.start_sec)
            && start + policy.timestamp_epsilon_sec < prev_end
        {
            return Err(conformance_err(
                index,
                format_args!("start_sec ({start}) overlaps previous end_sec ({prev_end})"),
            ));
        }

        if let Some(confidence) = segment.confidence
            && !(0.0..=1.0).contains(&confidence)
        {
            return Err(conformance_err(
                index,
                format_args!("confidence ({confidence}) is outside [0.0, 1.0]"),
            ));
        }

        if let Some(ref speaker) = segment.speaker
            && speaker.trim().is_empty()
        {
            return Err(conformance_err(index, "has empty speaker label"));
        }

        if let Some(end) = segment.end_sec {
            previous_end = Some(end);
        }
    }

    Ok(())
}

#[must_use]
pub fn compare_segments_with_tolerance(
    expected: &[TranscriptionSegment],
    observed: &[TranscriptionSegment],
    tolerance: SegmentCompatibilityTolerance,
) -> SegmentComparisonReport {
    let mut text_mismatches = 0usize;
    let mut speaker_mismatches = 0usize;
    let mut timestamp_violations = 0usize;

    for (left, right) in expected.iter().zip(observed.iter()) {
        if tolerance.require_text_exact && left.text.trim() != right.text.trim() {
            text_mismatches += 1;
        }

        if tolerance.require_speaker_exact && left.speaker != right.speaker {
            speaker_mismatches += 1;
        }

        if !timestamp_within_tolerance(
            left.start_sec,
            right.start_sec,
            tolerance.timestamp_tolerance_sec,
        ) || !timestamp_within_tolerance(
            left.end_sec,
            right.end_sec,
            tolerance.timestamp_tolerance_sec,
        ) {
            timestamp_violations += 1;
        }
    }

    SegmentComparisonReport {
        expected_segments: expected.len(),
        observed_segments: observed.len(),
        length_mismatch: expected.len() != observed.len(),
        text_mismatches,
        speaker_mismatches,
        timestamp_violations,
    }
}

#[must_use]
pub fn compare_replay_envelopes(
    expected: &ReplayEnvelope,
    observed: &ReplayEnvelope,
) -> ReplayComparisonReport {
    let mut missing_expected = Vec::new();
    let mut missing_observed = Vec::new();

    let input_hash_match = compare_replay_field(
        "input_content_hash",
        expected.input_content_hash.as_deref(),
        observed.input_content_hash.as_deref(),
        &mut missing_expected,
        &mut missing_observed,
    );
    let backend_identity_match = compare_replay_field(
        "backend_identity",
        expected.backend_identity.as_deref(),
        observed.backend_identity.as_deref(),
        &mut missing_expected,
        &mut missing_observed,
    );
    let backend_version_match = compare_replay_field(
        "backend_version",
        expected.backend_version.as_deref(),
        observed.backend_version.as_deref(),
        &mut missing_expected,
        &mut missing_observed,
    );
    let output_hash_match = compare_replay_field(
        "output_payload_hash",
        expected.output_payload_hash.as_deref(),
        observed.output_payload_hash.as_deref(),
        &mut missing_expected,
        &mut missing_observed,
    );

    ReplayComparisonReport {
        input_hash_match,
        backend_identity_match,
        backend_version_match,
        output_hash_match,
        missing_expected_fields: missing_expected,
        missing_observed_fields: missing_observed,
    }
}

fn compare_replay_field(
    field: &str,
    expected: Option<&str>,
    observed: Option<&str>,
    missing_expected: &mut Vec<String>,
    missing_observed: &mut Vec<String>,
) -> bool {
    match (expected, observed) {
        (Some(left), Some(right)) => left == right,
        (Some(_), None) => {
            missing_observed.push(field.to_owned());
            false
        }
        (None, Some(_)) => {
            missing_expected.push(field.to_owned());
            false
        }
        (None, None) => true,
    }
}

fn timestamp_within_tolerance(left: Option<f64>, right: Option<f64>, tolerance_sec: f64) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => (a - b).abs() <= tolerance_sec,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Shadow-run comparison (native-engine replacement contract, bd-1rj.8)
// ---------------------------------------------------------------------------

/// Rollout stage for a native engine relative to its bridge adapter.
///
/// See `docs/native_engine_contract.md` section 5.2 for stage definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeEngineRolloutStage {
    /// Stage 0: Shadow comparison available, bridge is primary.
    Shadow,
    /// Stage 1: CI gate green, native registered but not in auto-priority.
    Validated,
    /// Stage 2: Native is fallback behind bridge in auto-priority.
    Fallback,
    /// Stage 3: Native is primary, bridge is fallback.
    Primary,
    /// Stage 4: Bridge deprecated and removed from auto-priority.
    Sole,
}

impl NativeEngineRolloutStage {
    /// Environment variable for rollout-stage routing controls.
    pub const ENV_VAR: &str = "FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE";

    /// Stable lowercase identifier for docs/evidence/env parsing.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Validated => "validated",
            Self::Fallback => "fallback",
            Self::Primary => "primary",
            Self::Sole => "sole",
        }
    }

    /// Parse rollout stage from named or numeric values.
    ///
    /// Numeric aliases:
    /// - `0`: shadow
    /// - `1`: validated
    /// - `2`: fallback
    /// - `3`: primary
    /// - `4`: sole
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        let normalized = value.trim();
        match normalized.len() {
            1 => match normalized.as_bytes()[0] {
                b'0' => Some(Self::Shadow),
                b'1' => Some(Self::Validated),
                b'2' => Some(Self::Fallback),
                b'3' => Some(Self::Primary),
                b'4' => Some(Self::Sole),
                _ => None,
            },
            4 if normalized.eq_ignore_ascii_case("sole") => Some(Self::Sole),
            6 if normalized.eq_ignore_ascii_case("shadow") => Some(Self::Shadow),
            7 if normalized.eq_ignore_ascii_case("primary") => Some(Self::Primary),
            8 if normalized.eq_ignore_ascii_case("fallback") => Some(Self::Fallback),
            9 if normalized.eq_ignore_ascii_case("validated") => Some(Self::Validated),
            _ => None,
        }
    }
}

/// Result of comparing a native engine run against its bridge adapter run
/// on the same input.
#[derive(Debug, Clone)]
pub struct ShadowRunReport {
    /// Name of the primary (bridge) engine.
    pub primary_engine: String,
    /// Name of the shadow (native) engine.
    pub shadow_engine: String,
    /// Segment-level comparison.
    pub segment_comparison: SegmentComparisonReport,
    /// Replay envelope comparison.
    pub replay_comparison: ReplayComparisonReport,
    /// Whether the shadow run satisfies all rollout gates.
    pub passes_gate: bool,
}

/// Word-error-rate report for a hypothesis transcript against a reference.
///
/// `wer = edits / reference_words` (the standard ASR metric): the minimum
/// word-level insertions + deletions + substitutions to turn the reference into
/// the hypothesis, divided by the reference word count. `0.0` is a perfect
/// match; values can exceed `1.0` when the hypothesis is much longer than the
/// reference. An empty reference yields `wer = 0.0` iff the hypothesis is also
/// empty, else `1.0` (every hypothesis word is an insertion, normalized by the
/// conventional denominator of 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WerReport {
    pub reference_words: usize,
    pub hypothesis_words: usize,
    pub edits: usize,
    pub wer: f64,
}

impl WerReport {
    /// Whether the WER is within `max_wer` (the native-rollout conformance gate
    /// is `0.10`; see the bd-frp7 success criterion and
    /// `docs/native_engine_contract.md` §2.2).
    #[must_use]
    pub fn within(self, max_wer: f64) -> bool {
        self.wer <= max_wer
    }
}

/// Normalize a transcript to comparable words for WER: lowercase, drop every
/// character that is not an ASCII letter, digit, or intra-word apostrophe, then
/// collapse whitespace. Mirrors the ad-hoc harness used for the 2026-07-23
/// long-form WER verdict (`scratchpad/wer3.py`) so in-tree and hand runs agree.
#[must_use]
fn normalize_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || c == '\'' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Word-level Levenshtein edit distance between two token slices (unit costs for
/// insert/delete/substitute), computed with an O(min(n,m)) rolling row.
#[must_use]
fn word_edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let n = reference.len();
    let m = hypothesis.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = usize::from(reference[i - 1] != hypothesis[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Compute the word error rate of `hypothesis` against `reference` using the
/// standard ASR normalization (see [`normalize_words`]). This is the metric the
/// bd-frp7 native-engine success criterion and the shadow-run text-tolerance are
/// defined against; it was previously only computed by an out-of-tree script.
#[must_use]
pub fn word_error_rate(reference: &str, hypothesis: &str) -> WerReport {
    let r = normalize_words(reference);
    let h = normalize_words(hypothesis);
    let edits = word_edit_distance(&r, &h);
    let wer = if r.is_empty() {
        if h.is_empty() { 0.0 } else { 1.0 }
    } else {
        edits as f64 / r.len() as f64
    };
    WerReport {
        reference_words: r.len(),
        hypothesis_words: h.len(),
        edits,
        wer,
    }
}

/// Compare shadow-run results from a bridge adapter and native engine on the
/// same input. Returns a `ShadowRunReport` summarizing parity.
///
/// `passes_gate` is true when segment comparison is `within_tolerance()`
/// (no length mismatch, no text/speaker/timestamp violations).
///
/// Replay envelope comparison is informational only — different engines
/// naturally produce different output hashes and backend identities.
#[must_use]
pub fn compare_shadow_run(
    primary_engine: &str,
    shadow_engine: &str,
    primary_segments: &[TranscriptionSegment],
    shadow_segments: &[TranscriptionSegment],
    primary_replay: &ReplayEnvelope,
    shadow_replay: &ReplayEnvelope,
    tolerance: SegmentCompatibilityTolerance,
) -> ShadowRunReport {
    let segment_comparison =
        compare_segments_with_tolerance(primary_segments, shadow_segments, tolerance);
    let replay_comparison = compare_replay_envelopes(primary_replay, shadow_replay);

    let passes_gate = segment_comparison.within_tolerance();

    ShadowRunReport {
        primary_engine: primary_engine.to_owned(),
        shadow_engine: shadow_engine.to_owned(),
        segment_comparison,
        replay_comparison,
        passes_gate,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NativeEngineRolloutStage, SegmentComparisonReport, SegmentCompatibilityTolerance,
        SegmentConformancePolicy, compare_replay_envelopes, compare_segments_with_tolerance,
        validate_segment_invariants, validate_segment_invariants_with_policy, word_error_rate,
    };
    use crate::model::{ReplayEnvelope, TranscriptionSegment};

    #[test]
    fn wer_identical_transcripts_is_zero() {
        let r = word_error_rate("The quick brown fox.", "the quick brown fox");
        assert_eq!(r.wer, 0.0);
        assert_eq!(r.edits, 0);
        assert_eq!(r.reference_words, 4);
        assert!(r.within(0.10));
    }

    #[test]
    fn wer_normalizes_case_and_punctuation() {
        // Punctuation, case, and whitespace differences must not count as errors;
        // intra-word apostrophes are preserved (don't != dont).
        assert_eq!(
            word_error_rate("Don't stop,  now!", "dont stop now").wer,
            1.0 / 3.0
        );
        assert_eq!(word_error_rate("a b c", "A. B, C?").wer, 0.0);
    }

    #[test]
    fn wer_counts_insertions_deletions_substitutions() {
        // one substitution over 3 ref words
        assert_eq!(word_error_rate("a b c", "a x c").edits, 1);
        // one deletion
        assert_eq!(word_error_rate("a b c", "a c").edits, 1);
        // one insertion
        assert_eq!(word_error_rate("a b c", "a b x c").edits, 1);
        // full replacement: 3 substitutions / 3 ref = 1.0
        assert_eq!(word_error_rate("a b c", "x y z").wer, 1.0);
    }

    #[test]
    fn wer_empty_reference_edge_cases() {
        assert_eq!(word_error_rate("", "").wer, 0.0);
        assert_eq!(word_error_rate("", "spurious words").wer, 1.0);
        // empty hypothesis vs non-empty reference = every word deleted = 1.0
        assert_eq!(word_error_rate("a b c", "").wer, 1.0);
    }

    #[test]
    fn wer_can_exceed_one_when_hypothesis_much_longer() {
        // 3 ref words, hypothesis inserts 4 → 4 edits / 3 = 1.333…
        let r = word_error_rate("a b c", "a b c d e f g");
        assert_eq!(r.edits, 4);
        assert!(r.wer > 1.0);
        assert!(!r.within(0.10));
    }

    #[test]
    fn wer_matches_longform_verdict_fixture() {
        // Regression pin for the 2026-07-23 bd-r0qd verdict: the greedy default
        // that drops a window scores far above the 0.10 gate, while a hypothesis
        // that mirrors the reference scores 0.0. Uses a compact stand-in for the
        // real transcripts so the metric's direction/gating is locked in-tree.
        let reference = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let dropped_half = "alpha beta gamma delta epsilon"; // 5 deletions / 10
        let complete = "Alpha, beta gamma. Delta epsilon zeta eta theta iota kappa!";
        assert_eq!(word_error_rate(reference, dropped_half).wer, 0.5);
        assert!(!word_error_rate(reference, dropped_half).within(0.10));
        assert_eq!(word_error_rate(reference, complete).wer, 0.0);
        assert!(word_error_rate(reference, complete).within(0.10));
    }

    fn segment(start_sec: Option<f64>, end_sec: Option<f64>, text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec,
            end_sec,
            text: text.to_owned(),
            speaker: None,
            confidence: None,
        }
    }

    #[test]
    fn accepts_monotonic_non_overlapping_segments() {
        let segments = vec![
            segment(Some(0.0), Some(0.8), "hello"),
            segment(Some(0.8), Some(1.6), "world"),
            segment(Some(1.7), Some(2.0), "done"),
        ];

        assert!(validate_segment_invariants(&segments).is_ok());
    }

    #[test]
    fn rejects_segment_with_end_before_start() {
        let segments = vec![segment(Some(1.0), Some(0.5), "bad")];
        let error = validate_segment_invariants(&segments)
            .expect_err("segment timestamp ordering must fail");
        assert!(error.to_string().contains("end_sec"));
    }

    #[test]
    fn rollout_stage_parse_accepts_named_values() {
        assert_eq!(
            NativeEngineRolloutStage::parse("shadow"),
            Some(NativeEngineRolloutStage::Shadow)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("validated"),
            Some(NativeEngineRolloutStage::Validated)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("fallback"),
            Some(NativeEngineRolloutStage::Fallback)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("primary"),
            Some(NativeEngineRolloutStage::Primary)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("sole"),
            Some(NativeEngineRolloutStage::Sole)
        );
    }

    #[test]
    fn rollout_stage_parse_accepts_every_ascii_case_permutation() {
        for (name, expected) in [
            ("shadow", NativeEngineRolloutStage::Shadow),
            ("validated", NativeEngineRolloutStage::Validated),
            ("fallback", NativeEngineRolloutStage::Fallback),
            ("primary", NativeEngineRolloutStage::Primary),
            ("sole", NativeEngineRolloutStage::Sole),
        ] {
            let letters = name.as_bytes();
            for mask in 0..(1usize << letters.len()) {
                let mut spelling = letters.to_vec();
                for (index, byte) in spelling.iter_mut().enumerate() {
                    if mask & (1 << index) != 0 {
                        *byte = byte.to_ascii_uppercase();
                    }
                }
                let spelling = std::str::from_utf8(&spelling).expect("ASCII stage name");
                assert_eq!(NativeEngineRolloutStage::parse(spelling), Some(expected));
            }
        }
    }

    #[test]
    fn rollout_stage_parse_accepts_numeric_aliases() {
        assert_eq!(
            NativeEngineRolloutStage::parse("0"),
            Some(NativeEngineRolloutStage::Shadow)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("1"),
            Some(NativeEngineRolloutStage::Validated)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("2"),
            Some(NativeEngineRolloutStage::Fallback)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("3"),
            Some(NativeEngineRolloutStage::Primary)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("4"),
            Some(NativeEngineRolloutStage::Sole)
        );
    }

    #[test]
    fn rollout_stage_as_str_round_trips_with_parse() {
        for stage in [
            NativeEngineRolloutStage::Shadow,
            NativeEngineRolloutStage::Validated,
            NativeEngineRolloutStage::Fallback,
            NativeEngineRolloutStage::Primary,
            NativeEngineRolloutStage::Sole,
        ] {
            assert_eq!(NativeEngineRolloutStage::parse(stage.as_str()), Some(stage));
        }
    }

    #[test]
    fn rollout_stage_parse_rejects_unknown_values() {
        assert_eq!(NativeEngineRolloutStage::parse(""), None);
        assert_eq!(NativeEngineRolloutStage::parse("canary"), None);
        assert_eq!(NativeEngineRolloutStage::parse("99"), None);
    }

    #[test]
    fn rejects_overlap_when_policy_disallows_overlap() {
        let segments = vec![
            segment(Some(0.0), Some(1.0), "a"),
            segment(Some(0.9), Some(1.2), "b"),
        ];
        let error = validate_segment_invariants(&segments)
            .expect_err("overlap should fail under default policy");
        assert!(error.to_string().contains("overlaps"));
    }

    #[test]
    fn allows_overlap_when_policy_allows_it() {
        let segments = vec![
            segment(Some(0.0), Some(1.0), "a"),
            segment(Some(0.9), Some(1.2), "b"),
        ];

        let policy = SegmentConformancePolicy {
            allow_overlap: true,
            timestamp_epsilon_sec: 1e-6,
        };
        assert!(validate_segment_invariants_with_policy(&segments, policy).is_ok());
    }

    #[test]
    fn compare_segments_accepts_small_timestamp_drift_within_tolerance() {
        let expected = vec![
            segment(Some(0.0), Some(1.0), "hello"),
            segment(Some(1.0), Some(2.0), "world"),
        ];
        let observed = vec![
            segment(Some(0.02), Some(1.02), "hello"),
            segment(Some(1.01), Some(2.03), "world"),
        ];

        let report = compare_segments_with_tolerance(
            &expected,
            &observed,
            SegmentCompatibilityTolerance {
                timestamp_tolerance_sec: 0.05,
                require_text_exact: true,
                require_speaker_exact: false,
            },
        );
        assert!(report.within_tolerance());
    }

    #[test]
    fn compare_segments_flags_timestamp_and_text_drift_outside_tolerance() {
        let expected = vec![segment(Some(0.0), Some(1.0), "hello")];
        let observed = vec![segment(Some(0.3), Some(1.4), "HELLO")];

        let report = compare_segments_with_tolerance(
            &expected,
            &observed,
            SegmentCompatibilityTolerance {
                timestamp_tolerance_sec: 0.05,
                require_text_exact: true,
                require_speaker_exact: false,
            },
        );
        assert!(!report.within_tolerance());
        assert_eq!(report.timestamp_violations, 1);
        assert_eq!(report.text_mismatches, 1);
    }

    #[test]
    fn compare_segments_flags_speaker_mismatch_when_required() {
        let mut expected_seg = segment(Some(0.0), Some(1.0), "hello");
        expected_seg.speaker = Some("SPEAKER_00".to_owned());

        let mut observed_seg = segment(Some(0.0), Some(1.0), "hello");
        observed_seg.speaker = Some("SPEAKER_01".to_owned());

        let report = compare_segments_with_tolerance(
            &[expected_seg],
            &[observed_seg],
            SegmentCompatibilityTolerance {
                timestamp_tolerance_sec: 0.05,
                require_text_exact: true,
                require_speaker_exact: true,
            },
        );
        assert!(!report.within_tolerance());
        assert_eq!(report.speaker_mismatches, 1);
    }

    #[test]
    fn compare_replay_envelopes_accepts_identical_metadata() {
        let replay = ReplayEnvelope {
            input_content_hash: Some("input-sha".to_owned()),
            backend_identity: Some("whisper-cli".to_owned()),
            backend_version: Some("whisper 1.0.0".to_owned()),
            output_payload_hash: Some("output-sha".to_owned()),
        };
        let report = compare_replay_envelopes(&replay, &replay);
        assert!(report.within_tolerance());
    }

    #[test]
    fn compare_replay_envelopes_flags_drift_and_missing_fields() {
        let expected = ReplayEnvelope {
            input_content_hash: Some("input-sha".to_owned()),
            backend_identity: Some("whisper-cli".to_owned()),
            backend_version: Some("1.0.0".to_owned()),
            output_payload_hash: Some("output-sha".to_owned()),
        };
        let observed = ReplayEnvelope {
            input_content_hash: Some("different-input".to_owned()),
            backend_identity: Some("whisper-cli".to_owned()),
            backend_version: None,
            output_payload_hash: Some("different-output".to_owned()),
        };
        let report = compare_replay_envelopes(&expected, &observed);
        assert!(!report.within_tolerance());
        assert!(!report.input_hash_match);
        assert!(report.backend_identity_match);
        assert!(!report.backend_version_match);
        assert!(!report.output_hash_match);
        assert_eq!(report.missing_observed_fields, vec!["backend_version"]);
    }

    #[test]
    fn rejects_confidence_above_one() {
        let mut seg = segment(Some(0.0), Some(1.0), "text");
        seg.confidence = Some(1.5);
        let error = validate_segment_invariants(&[seg]).expect_err("confidence > 1.0 should fail");
        assert!(error.to_string().contains("confidence"));
    }

    #[test]
    fn rejects_confidence_below_zero() {
        let mut seg = segment(Some(0.0), Some(1.0), "text");
        seg.confidence = Some(-0.1);
        let error = validate_segment_invariants(&[seg]).expect_err("confidence < 0.0 should fail");
        assert!(error.to_string().contains("confidence"));
    }

    #[test]
    fn accepts_valid_confidence_bounds() {
        let mut seg0 = segment(Some(0.0), Some(0.5), "a");
        seg0.confidence = Some(0.0);
        let mut seg1 = segment(Some(0.5), Some(1.0), "b");
        seg1.confidence = Some(1.0);
        assert!(validate_segment_invariants(&[seg0, seg1]).is_ok());
    }

    #[test]
    fn rejects_empty_speaker_label() {
        let mut seg = segment(Some(0.0), Some(1.0), "text");
        seg.speaker = Some("  ".to_owned());
        let error =
            validate_segment_invariants(&[seg]).expect_err("empty speaker label should fail");
        assert!(error.to_string().contains("empty speaker"));
    }

    #[test]
    fn accepts_valid_speaker_label() {
        let mut seg = segment(Some(0.0), Some(1.0), "text");
        seg.speaker = Some("SPEAKER_00".to_owned());
        assert!(validate_segment_invariants(&[seg]).is_ok());
    }

    #[test]
    fn accepts_none_speaker_and_confidence() {
        let seg = segment(Some(0.0), Some(1.0), "text");
        assert!(validate_segment_invariants(&[seg]).is_ok());
    }

    // -----------------------------------------------------------------------
    // timestamp_within_tolerance edge cases
    // -----------------------------------------------------------------------

    use super::timestamp_within_tolerance;

    #[test]
    fn timestamp_tolerance_both_none_is_match() {
        assert!(timestamp_within_tolerance(None, None, 0.05));
    }

    #[test]
    fn timestamp_tolerance_some_vs_none_is_mismatch() {
        assert!(!timestamp_within_tolerance(Some(1.0), None, 0.05));
        assert!(!timestamp_within_tolerance(None, Some(1.0), 0.05));
    }

    #[test]
    fn timestamp_tolerance_exact_boundary() {
        // Use values that are exact in binary float (0.25, 0.5)
        // diff = 0.25, tolerance = 0.25 → should match (<=)
        assert!(timestamp_within_tolerance(Some(1.0), Some(1.25), 0.25));
        // diff = 0.5, tolerance = 0.25 → should not match
        assert!(!timestamp_within_tolerance(Some(1.0), Some(1.5), 0.25));
    }

    #[test]
    fn timestamp_tolerance_zero_tolerance_requires_exact_match() {
        assert!(timestamp_within_tolerance(Some(1.0), Some(1.0), 0.0));
        assert!(!timestamp_within_tolerance(
            Some(1.0),
            Some(1.0 + f64::EPSILON),
            0.0
        ));
    }

    // -----------------------------------------------------------------------
    // compare_replay_field edge cases
    // -----------------------------------------------------------------------

    use super::compare_replay_field;

    #[test]
    fn replay_field_both_none_is_match() {
        let mut me = Vec::new();
        let mut mo = Vec::new();
        let result = compare_replay_field("f", None, None, &mut me, &mut mo);
        assert!(result);
        assert!(me.is_empty());
        assert!(mo.is_empty());
    }

    #[test]
    fn replay_field_expected_some_observed_none_flags_missing_observed() {
        let mut me = Vec::new();
        let mut mo = Vec::new();
        let result = compare_replay_field("f", Some("val"), None, &mut me, &mut mo);
        assert!(!result);
        assert!(me.is_empty());
        assert_eq!(mo, vec!["f"]);
    }

    #[test]
    fn replay_field_expected_none_observed_some_flags_missing_expected() {
        let mut me = Vec::new();
        let mut mo = Vec::new();
        let result = compare_replay_field("f", None, Some("val"), &mut me, &mut mo);
        assert!(!result);
        assert_eq!(me, vec!["f"]);
        assert!(mo.is_empty());
    }

    #[test]
    fn replay_field_both_some_matching() {
        let mut me = Vec::new();
        let mut mo = Vec::new();
        let result = compare_replay_field("f", Some("abc"), Some("abc"), &mut me, &mut mo);
        assert!(result);
    }

    #[test]
    fn replay_field_both_some_different() {
        let mut me = Vec::new();
        let mut mo = Vec::new();
        let result = compare_replay_field("f", Some("abc"), Some("xyz"), &mut me, &mut mo);
        assert!(!result);
        // No missing fields — both are present, just different
        assert!(me.is_empty());
        assert!(mo.is_empty());
    }

    // -----------------------------------------------------------------------
    // compare_replay_envelopes: both default (all None) should match
    // -----------------------------------------------------------------------

    #[test]
    fn compare_replay_envelopes_both_default_matches() {
        let report =
            compare_replay_envelopes(&ReplayEnvelope::default(), &ReplayEnvelope::default());
        assert!(report.within_tolerance());
        assert!(report.missing_expected_fields.is_empty());
        assert!(report.missing_observed_fields.is_empty());
    }

    // -----------------------------------------------------------------------
    // SegmentComparisonReport: length mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn compare_segments_length_mismatch() {
        let expected = vec![segment(Some(0.0), Some(1.0), "a")];
        let observed = vec![
            segment(Some(0.0), Some(1.0), "a"),
            segment(Some(1.0), Some(2.0), "b"),
        ];
        let report = compare_segments_with_tolerance(
            &expected,
            &observed,
            SegmentCompatibilityTolerance::default(),
        );
        assert!(report.length_mismatch);
        assert!(!report.within_tolerance());
    }

    #[test]
    fn compare_segments_empty_both_sides() {
        let report =
            compare_segments_with_tolerance(&[], &[], SegmentCompatibilityTolerance::default());
        assert!(report.within_tolerance());
        assert_eq!(report.expected_segments, 0);
        assert_eq!(report.observed_segments, 0);
    }

    #[test]
    fn validate_empty_segments_list_succeeds() {
        assert!(validate_segment_invariants(&[]).is_ok());
    }

    // -----------------------------------------------------------------------
    // Edge cases: NaN, infinity, zero-length, large lists
    // -----------------------------------------------------------------------

    #[test]
    fn rejects_nan_confidence() {
        let mut seg = segment(Some(0.0), Some(1.0), "text");
        seg.confidence = Some(f64::NAN);
        let err = validate_segment_invariants(&[seg]).expect_err("NaN confidence should fail");
        assert!(err.to_string().contains("confidence"));
    }

    #[test]
    fn rejects_infinite_confidence() {
        let mut seg = segment(Some(0.0), Some(1.0), "text");
        seg.confidence = Some(f64::INFINITY);
        let err = validate_segment_invariants(&[seg]).expect_err("infinite confidence should fail");
        assert!(err.to_string().contains("confidence"));
    }

    #[test]
    fn rejects_negative_infinite_confidence() {
        let mut seg = segment(Some(0.0), Some(1.0), "text");
        seg.confidence = Some(f64::NEG_INFINITY);
        let err =
            validate_segment_invariants(&[seg]).expect_err("neg-infinite confidence should fail");
        assert!(err.to_string().contains("confidence"));
    }

    #[test]
    fn rejects_nan_start_sec() {
        let seg = segment(Some(f64::NAN), Some(1.0), "text");
        let err = validate_segment_invariants(&[seg]).expect_err("NaN start_sec should fail");
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn rejects_nan_end_sec() {
        let seg = segment(Some(0.0), Some(f64::NAN), "text");
        let err = validate_segment_invariants(&[seg]).expect_err("NaN end_sec should fail");
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn rejects_infinite_start_sec() {
        let seg = segment(Some(f64::INFINITY), Some(1.0), "text");
        let err = validate_segment_invariants(&[seg]).expect_err("infinite start_sec should fail");
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn rejects_infinite_end_sec() {
        let seg = segment(Some(0.0), Some(f64::INFINITY), "text");
        let err = validate_segment_invariants(&[seg]).expect_err("infinite end_sec should fail");
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn rejects_negative_start_sec() {
        let seg = segment(Some(-1.0), Some(1.0), "text");
        let err = validate_segment_invariants(&[seg]).expect_err("negative start_sec should fail");
        assert!(err.to_string().contains("negative"));
    }

    #[test]
    fn rejects_negative_end_sec() {
        let seg = segment(Some(0.0), Some(-0.5), "text");
        let err = validate_segment_invariants(&[seg]).expect_err("negative end_sec should fail");
        assert!(err.to_string().contains("negative"));
    }

    #[test]
    fn accepts_zero_length_segment() {
        // start == end is valid (a point in time).
        let segments = vec![segment(Some(1.0), Some(1.0), "instantaneous")];
        assert!(validate_segment_invariants(&segments).is_ok());
    }

    #[test]
    fn accepts_abutting_segments_at_exact_boundary() {
        // No overlap: seg1 ends at 1.0, seg2 starts at 1.0.
        let segments = vec![
            segment(Some(0.0), Some(1.0), "first"),
            segment(Some(1.0), Some(2.0), "second"),
        ];
        assert!(validate_segment_invariants(&segments).is_ok());
    }

    #[test]
    fn large_segment_list_validates_correctly() {
        let segments: Vec<TranscriptionSegment> = (0..1000)
            .map(|i| segment(Some(i as f64), Some(i as f64 + 0.5), &format!("seg-{i}")))
            .collect();
        assert!(validate_segment_invariants(&segments).is_ok());
    }

    #[test]
    fn large_segment_list_detects_overlap_late_in_sequence() {
        let mut segments: Vec<TranscriptionSegment> = (0..100)
            .map(|i| segment(Some(i as f64), Some(i as f64 + 0.5), &format!("seg-{i}")))
            .collect();
        // Create overlap at segment 99: push start backward into seg 98.
        segments[99] = segment(Some(97.0), Some(99.5), "overlap-seg");
        let err =
            validate_segment_invariants(&segments).expect_err("late overlap should be detected");
        assert!(err.to_string().contains("overlaps"));
    }

    #[test]
    fn comparison_report_within_tolerance_checks_all_fields() {
        use super::SegmentComparisonReport;
        // All zero → within_tolerance
        let good = SegmentComparisonReport {
            expected_segments: 5,
            observed_segments: 5,
            length_mismatch: false,
            text_mismatches: 0,
            speaker_mismatches: 0,
            timestamp_violations: 0,
        };
        assert!(good.within_tolerance());

        // Any nonzero field → not within_tolerance
        let bad_text = SegmentComparisonReport {
            text_mismatches: 1,
            ..good
        };
        assert!(!bad_text.within_tolerance());

        let bad_speaker = SegmentComparisonReport {
            speaker_mismatches: 1,
            ..good
        };
        assert!(!bad_speaker.within_tolerance());

        let bad_timestamp = SegmentComparisonReport {
            timestamp_violations: 1,
            ..good
        };
        assert!(!bad_timestamp.within_tolerance());
    }

    #[test]
    fn replay_comparison_within_tolerance_checks_all_fields() {
        use super::ReplayComparisonReport;
        let good = ReplayComparisonReport {
            input_hash_match: true,
            backend_identity_match: true,
            backend_version_match: true,
            output_hash_match: true,
            missing_expected_fields: vec![],
            missing_observed_fields: vec![],
        };
        assert!(good.within_tolerance());

        let bad = ReplayComparisonReport {
            input_hash_match: false,
            ..good.clone()
        };
        assert!(!bad.within_tolerance());

        let with_missing = ReplayComparisonReport {
            missing_expected_fields: vec!["backend_version".to_owned()],
            ..good
        };
        assert!(!with_missing.within_tolerance());
    }

    #[test]
    fn segments_with_none_timestamps_skip_overlap_check() {
        let segments = vec![segment(None, None, "a"), segment(None, None, "b")];
        assert!(validate_segment_invariants(&segments).is_ok());
    }

    // ── Additional edge case tests ──

    #[test]
    fn compare_segments_none_vs_some_timestamps_counts_as_violation() {
        let expected = vec![segment(Some(0.0), Some(1.0), "hello")];
        let observed = vec![segment(None, None, "hello")];
        let report = compare_segments_with_tolerance(
            &expected,
            &observed,
            SegmentCompatibilityTolerance::default(),
        );
        assert_eq!(
            report.timestamp_violations, 1,
            "Some vs None timestamps should be a violation"
        );
    }

    #[test]
    fn compare_segments_text_not_required_exact_ignores_differences() {
        let expected = vec![segment(Some(0.0), Some(1.0), "hello")];
        let observed = vec![segment(Some(0.0), Some(1.0), "HELLO")];
        let tolerance = SegmentCompatibilityTolerance {
            timestamp_tolerance_sec: 0.05,
            require_text_exact: false,
            require_speaker_exact: false,
        };
        let report = compare_segments_with_tolerance(&expected, &observed, tolerance);
        assert_eq!(report.text_mismatches, 0);
        assert!(report.within_tolerance());
    }

    #[test]
    fn compare_segments_length_mismatch_extra_observed() {
        let expected = vec![segment(Some(0.0), Some(1.0), "a")];
        let observed = vec![
            segment(Some(0.0), Some(1.0), "a"),
            segment(Some(1.0), Some(2.0), "b"),
        ];
        let report = compare_segments_with_tolerance(
            &expected,
            &observed,
            SegmentCompatibilityTolerance::default(),
        );
        assert!(report.length_mismatch);
        assert!(!report.within_tolerance());
        assert_eq!(report.expected_segments, 1);
        assert_eq!(report.observed_segments, 2);
    }

    #[test]
    fn default_tolerance_values_are_reasonable() {
        let t = SegmentCompatibilityTolerance::default();
        assert_eq!(
            t.timestamp_tolerance_sec,
            super::CANONICAL_TIMESTAMP_TOLERANCE_SEC
        );
        assert!(t.require_text_exact);
        assert!(!t.require_speaker_exact);
    }

    #[test]
    fn default_conformance_policy_values() {
        let p = SegmentConformancePolicy::default();
        assert!(!p.allow_overlap);
        assert!((p.timestamp_epsilon_sec - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn overlap_at_epsilon_boundary_is_allowed() {
        // Segments that overlap by exactly epsilon should pass
        let policy = SegmentConformancePolicy {
            allow_overlap: false,
            timestamp_epsilon_sec: 0.001,
        };
        let segments = vec![
            segment(Some(0.0), Some(1.0), "a"),
            segment(Some(0.999), Some(2.0), "b"), // overlap of 0.001 == epsilon
        ];
        assert!(validate_segment_invariants_with_policy(&segments, policy).is_ok());
    }

    #[test]
    fn overlap_beyond_epsilon_is_rejected() {
        let policy = SegmentConformancePolicy {
            allow_overlap: false,
            timestamp_epsilon_sec: 0.001,
        };
        let segments = vec![
            segment(Some(0.0), Some(1.0), "a"),
            segment(Some(0.998), Some(2.0), "b"), // overlap of 0.002 > epsilon
        ];
        assert!(validate_segment_invariants_with_policy(&segments, policy).is_err());
    }

    #[test]
    fn compare_replay_envelopes_partial_fields_present() {
        let expected = ReplayEnvelope {
            input_content_hash: Some("hash_a".to_owned()),
            backend_identity: None,
            backend_version: None,
            output_payload_hash: Some("out_a".to_owned()),
        };
        let observed = ReplayEnvelope {
            input_content_hash: Some("hash_a".to_owned()),
            backend_identity: None,
            backend_version: None,
            output_payload_hash: Some("out_a".to_owned()),
        };
        let report = compare_replay_envelopes(&expected, &observed);
        assert!(report.within_tolerance());
        assert!(report.missing_expected_fields.is_empty());
        assert!(report.missing_observed_fields.is_empty());
    }

    #[test]
    fn segment_with_start_only_no_end_sec_passes() {
        let segments = vec![
            segment(Some(0.0), None, "no end"),
            segment(Some(1.0), Some(2.0), "has end"),
        ];
        // No end_sec on first segment means no overlap check for second
        assert!(validate_segment_invariants(&segments).is_ok());
    }

    #[test]
    fn gap_in_end_sec_does_not_false_positive_overlap() {
        // seg0: end 1.0, seg1: end None, seg2: start 0.5 — should NOT overlap
        // because previous_end stays at 1.0 (seg1 has no end_sec),
        // and seg2 start 0.5 < 1.0 → overlap detected
        let segments = vec![
            segment(Some(0.0), Some(1.0), "a"),
            segment(Some(1.0), None, "b"),
            segment(Some(0.5), Some(2.0), "c"),
        ];
        let err = validate_segment_invariants(&segments).expect_err("overlap should be detected");
        assert!(err.to_string().contains("overlaps"));
    }

    #[test]
    fn compare_speaker_none_vs_some_counts_as_mismatch_when_exact_required() {
        let mut expected_seg = segment(Some(0.0), Some(1.0), "hello");
        expected_seg.speaker = Some("SPEAKER_00".to_owned());

        let observed_seg = segment(Some(0.0), Some(1.0), "hello");
        // observed_seg.speaker is None

        let tolerance = SegmentCompatibilityTolerance {
            timestamp_tolerance_sec: 0.05,
            require_text_exact: true,
            require_speaker_exact: true,
        };
        let report = compare_segments_with_tolerance(&[expected_seg], &[observed_seg], tolerance);
        assert_eq!(report.speaker_mismatches, 1);
    }

    #[test]
    fn end_sec_slightly_before_start_within_epsilon_passes() {
        let policy = SegmentConformancePolicy {
            allow_overlap: false,
            timestamp_epsilon_sec: 0.01,
        };
        // end is 0.005 before start — within epsilon of 0.01 → passes
        let segments = vec![segment(Some(1.0), Some(0.995), "tiny reverse")];
        assert!(validate_segment_invariants_with_policy(&segments, policy).is_ok());
    }

    #[test]
    fn compare_multiple_segments_counts_each_violation_separately() {
        let expected = vec![
            segment(Some(0.0), Some(1.0), "a"),
            segment(Some(1.0), Some(2.0), "b"),
            segment(Some(2.0), Some(3.0), "c"),
        ];
        let observed = vec![
            segment(Some(0.0), Some(1.0), "WRONG"),    // text mismatch
            segment(Some(1.5), Some(2.5), "b"),        // timestamp violation
            segment(Some(2.0), Some(3.0), "ALSO BAD"), // text mismatch
        ];
        let tolerance = SegmentCompatibilityTolerance {
            timestamp_tolerance_sec: 0.05,
            require_text_exact: true,
            require_speaker_exact: false,
        };
        let report = compare_segments_with_tolerance(&expected, &observed, tolerance);
        assert_eq!(report.text_mismatches, 2);
        assert_eq!(report.timestamp_violations, 1);
        assert!(!report.within_tolerance());
    }

    #[test]
    fn replay_all_fields_missing_in_observed_lists_all() {
        let expected = ReplayEnvelope {
            input_content_hash: Some("h1".to_owned()),
            backend_identity: Some("id1".to_owned()),
            backend_version: Some("v1".to_owned()),
            output_payload_hash: Some("out1".to_owned()),
        };
        let observed = ReplayEnvelope::default(); // all None
        let report = compare_replay_envelopes(&expected, &observed);
        assert!(!report.within_tolerance());
        assert_eq!(report.missing_observed_fields.len(), 4);
        assert!(
            report
                .missing_observed_fields
                .contains(&"input_content_hash".to_owned())
        );
        assert!(
            report
                .missing_observed_fields
                .contains(&"backend_identity".to_owned())
        );
        assert!(
            report
                .missing_observed_fields
                .contains(&"backend_version".to_owned())
        );
        assert!(
            report
                .missing_observed_fields
                .contains(&"output_payload_hash".to_owned())
        );
    }

    #[test]
    fn segment_comparison_report_within_tolerance_requires_all_zero() {
        let report = SegmentComparisonReport {
            expected_segments: 5,
            observed_segments: 5,
            length_mismatch: false,
            text_mismatches: 0,
            speaker_mismatches: 0,
            timestamp_violations: 0,
        };
        assert!(report.within_tolerance());

        let report_with_text = SegmentComparisonReport {
            text_mismatches: 1,
            ..report
        };
        assert!(!report_with_text.within_tolerance());

        let report_with_speaker = SegmentComparisonReport {
            speaker_mismatches: 1,
            ..report
        };
        assert!(!report_with_speaker.within_tolerance());

        let report_with_ts = SegmentComparisonReport {
            timestamp_violations: 1,
            ..report
        };
        assert!(!report_with_ts.within_tolerance());
    }

    // -----------------------------------------------------------------------
    // Canonical tolerance regression: code ↔ docs drift detection (bd-1rj.7)
    // -----------------------------------------------------------------------

    #[test]
    fn canonical_constant_matches_default_tolerance() {
        assert_eq!(
            super::CANONICAL_TIMESTAMP_TOLERANCE_SEC,
            SegmentCompatibilityTolerance::default().timestamp_tolerance_sec,
            "CANONICAL_TIMESTAMP_TOLERANCE_SEC must equal SegmentCompatibilityTolerance::default().timestamp_tolerance_sec"
        );
    }

    #[test]
    fn canonical_constant_is_50ms() {
        assert!(
            (super::CANONICAL_TIMESTAMP_TOLERANCE_SEC - 0.05).abs() < f64::EPSILON,
            "canonical tolerance must be 50ms (0.05s)"
        );
    }

    #[test]
    fn engine_compatibility_spec_documents_canonical_tolerance() {
        let spec = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("docs/engine_compatibility_spec.md"),
        )
        .expect("docs/engine_compatibility_spec.md must exist");

        // Section 1.3 must mention the canonical value in both human and code form.
        assert!(
            spec.contains("50ms"),
            "spec must document the tolerance as 50ms"
        );
        assert!(
            spec.contains("0.05"),
            "spec must document the tolerance as 0.05"
        );
        assert!(
            spec.contains("timestamp_tolerance_sec"),
            "spec must reference timestamp_tolerance_sec field name"
        );
    }

    // -----------------------------------------------------------------------
    // Shadow-run comparison (bd-1rj.8)
    // -----------------------------------------------------------------------

    use super::compare_shadow_run;

    #[test]
    fn shadow_run_identical_segments_passes_gate() {
        let segments = vec![
            segment(Some(0.0), Some(1.0), "hello"),
            segment(Some(1.0), Some(2.0), "world"),
        ];
        let replay = ReplayEnvelope {
            input_content_hash: Some("abc123".to_owned()),
            backend_identity: Some("whisper.cpp".to_owned()),
            backend_version: Some("1.0.0".to_owned()),
            output_payload_hash: Some("out456".to_owned()),
        };
        let shadow_replay = ReplayEnvelope {
            input_content_hash: Some("abc123".to_owned()),
            backend_identity: Some("whisper.cpp-native".to_owned()),
            backend_version: Some("0.1.0".to_owned()),
            output_payload_hash: Some("out789".to_owned()),
        };
        let report = compare_shadow_run(
            "whisper.cpp",
            "whisper.cpp-native",
            &segments,
            &segments,
            &replay,
            &shadow_replay,
            SegmentCompatibilityTolerance::default(),
        );
        assert!(report.passes_gate, "identical segments should pass gate");
        assert!(report.segment_comparison.within_tolerance());
        // Replay hashes differ (different engines) — that's expected
        assert!(!report.replay_comparison.output_hash_match);
    }

    #[test]
    fn shadow_run_text_mismatch_fails_gate() {
        let primary = vec![segment(Some(0.0), Some(1.0), "hello")];
        let shadow = vec![segment(Some(0.0), Some(1.0), "Hello")];
        let replay = ReplayEnvelope::default();
        let report = compare_shadow_run(
            "bridge",
            "native",
            &primary,
            &shadow,
            &replay,
            &replay,
            SegmentCompatibilityTolerance::default(),
        );
        assert!(!report.passes_gate, "text mismatch should fail gate");
        assert_eq!(report.segment_comparison.text_mismatches, 1);
    }

    #[test]
    fn shadow_run_timestamp_violation_fails_gate() {
        let primary = vec![segment(Some(0.0), Some(1.0), "hello")];
        let shadow = vec![segment(Some(0.2), Some(1.2), "hello")];
        let replay = ReplayEnvelope::default();
        let report = compare_shadow_run(
            "bridge",
            "native",
            &primary,
            &shadow,
            &replay,
            &replay,
            SegmentCompatibilityTolerance::default(),
        );
        assert!(
            !report.passes_gate,
            "timestamp violation (200ms drift > 50ms tolerance) should fail gate"
        );
        assert_eq!(report.segment_comparison.timestamp_violations, 1);
    }

    #[test]
    fn shadow_run_within_tolerance_drift_passes_gate() {
        let primary = vec![segment(Some(0.0), Some(1.0), "hello")];
        let shadow = vec![segment(Some(0.03), Some(1.02), "hello")];
        let replay = ReplayEnvelope::default();
        let report = compare_shadow_run(
            "bridge",
            "native",
            &primary,
            &shadow,
            &replay,
            &replay,
            SegmentCompatibilityTolerance::default(),
        );
        assert!(
            report.passes_gate,
            "30ms drift within 50ms tolerance should pass gate"
        );
    }

    #[test]
    fn rollout_stage_ordering_is_distinct() {
        let stages = [
            NativeEngineRolloutStage::Shadow,
            NativeEngineRolloutStage::Validated,
            NativeEngineRolloutStage::Fallback,
            NativeEngineRolloutStage::Primary,
            NativeEngineRolloutStage::Sole,
        ];
        for (i, a) in stages.iter().enumerate() {
            for (j, b) in stages.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn native_engine_contract_doc_exists() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/native_engine_contract.md");
        assert!(path.exists(), "docs/native_engine_contract.md must exist");
        let content = std::fs::read_to_string(&path).expect("must be readable");
        assert!(
            content.contains("Shadow-Run"),
            "contract must document shadow-run methodology"
        );
        assert!(
            content.contains("Rollout Gates"),
            "contract must document rollout gates"
        );
        assert!(
            content.contains("Fallback Policy"),
            "contract must document fallback policy"
        );
    }

    // ── Task #205 — conformance edge-case tests ──────────────────────

    #[test]
    fn negative_zero_timestamps_accepted_by_validation() {
        // IEEE 754: -0.0 == 0.0, so -0.0 < 0.0 is false.
        // Document that -0.0 passes the negativity check.
        let segments = vec![
            TranscriptionSegment {
                start_sec: Some(-0.0_f64),
                end_sec: Some(-0.0_f64),
                text: "neg zero".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "normal".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        assert!(
            validate_segment_invariants(&segments).is_ok(),
            "negative zero should be accepted (IEEE 754: -0.0 == 0.0)"
        );
    }

    #[test]
    fn validate_segment_invariants_rejects_significant_overlap() {
        let segments = vec![
            segment(Some(0.0), Some(2.0), "first"),
            segment(Some(1.0), Some(3.0), "overlaps significantly"),
        ];
        // Default epsilon is 1e-6. 1.0 + 1e-6 < 2.0 is true.
        let err = validate_segment_invariants(&segments).expect_err("should reject overlap");
        assert!(err.to_string().contains("overlap"));
    }

    #[test]
    fn validate_segment_invariants_accepts_micro_overlap_within_epsilon() {
        let segments = vec![
            segment(Some(0.0), Some(1.0), "first"),
            segment(Some(0.99999999), Some(2.0), "micro overlap"),
        ];
        // 0.99999999 + 1e-6 > 1.0.
        assert!(validate_segment_invariants(&segments).is_ok());
    }

    #[test]
    fn validate_segment_invariants_policy_allows_explicit_overlap() {
        let segments = vec![
            segment(Some(0.0), Some(2.0), "first"),
            segment(Some(1.0), Some(3.0), "overlapping speaker"),
        ];
        let policy = SegmentConformancePolicy {
            allow_overlap: true,
            timestamp_epsilon_sec: 1e-6,
        };
        assert!(validate_segment_invariants_with_policy(&segments, policy).is_ok());
    }

    #[test]
    fn validate_segment_invariants_rejects_non_monotonic_start_times() {
        let segments = vec![
            segment(Some(2.0), Some(3.0), "second"),
            segment(Some(1.0), Some(2.0), "first (out of order)"),
        ];
        let err = validate_segment_invariants(&segments).expect_err("should reject non-monotonic");
        assert!(err.to_string().contains("out of order") || err.to_string().contains("overlap"));
    }

    #[test]
    fn validate_segment_invariants_rejects_end_before_start() {
        let segments = vec![segment(Some(2.0), Some(1.0), "time traveler")];
        let err = validate_segment_invariants(&segments).expect_err("should reject end < start");
        assert!(err.to_string().contains("before start_sec"));
    }

    #[test]
    fn validate_segment_invariants_accepts_exact_abutting_segments() {
        let segments = vec![
            segment(Some(0.0), Some(1.0), "first"),
            segment(Some(1.0), Some(2.0), "second"),
        ];
        assert!(validate_segment_invariants(&segments).is_ok());
    }

    #[test]
    fn rollout_stage_parse_whitespace_and_mixed_case() {
        // parse() trims Unicode whitespace and compares ASCII case-insensitively.
        assert_eq!(
            NativeEngineRolloutStage::parse("  SHADOW  "),
            Some(NativeEngineRolloutStage::Shadow)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse(" Primary "),
            Some(NativeEngineRolloutStage::Primary)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("\tSole\n"),
            Some(NativeEngineRolloutStage::Sole)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("\u{a0}PRIMARY\u{2003}"),
            Some(NativeEngineRolloutStage::Primary)
        );
        assert_eq!(
            NativeEngineRolloutStage::parse("\u{2028}3\u{2029}"),
            Some(NativeEngineRolloutStage::Primary)
        );
        // Whitespace-only → empty after trim → None.
        assert_eq!(NativeEngineRolloutStage::parse("   "), None);
    }

    #[test]
    fn rollout_stage_parse_rejects_noncanonical_shapes() {
        for value in [
            "so le",
            "03",
            "+3",
            "٣",
            "\u{200b}sole",
            "sole\u{feff}",
            "ſhadow",
            "PRİMARY",
            "ＳＯＬＥ",
        ] {
            assert_eq!(NativeEngineRolloutStage::parse(value), None, "{value:?}");
        }
    }

    #[test]
    fn compare_shadow_run_passes_replay_but_fails_segment_gate() {
        use super::compare_shadow_run;

        // Identical replays → replay_comparison.within_tolerance() is true.
        let replay = ReplayEnvelope {
            input_content_hash: Some("aaa".to_owned()),
            backend_identity: Some("whisper-cpp".to_owned()),
            backend_version: Some("1.0".to_owned()),
            output_payload_hash: Some("bbb".to_owned()),
        };

        // Segments diverge in text → segment_comparison fails.
        let primary = vec![segment(Some(0.0), Some(1.0), "hello world")];
        let shadow = vec![segment(Some(0.0), Some(1.0), "goodbye world")];

        let tolerance = SegmentCompatibilityTolerance {
            require_text_exact: true,
            require_speaker_exact: false,
            timestamp_tolerance_sec: 0.5,
        };

        let report = compare_shadow_run(
            "bridge", "native", &primary, &shadow, &replay, &replay, tolerance,
        );

        // Gate depends only on segment comparison, not replay comparison.
        assert!(!report.passes_gate, "gate should fail on text mismatch");
        assert!(
            report.replay_comparison.within_tolerance(),
            "replay comparison should pass (identical envelopes)"
        );
    }

    #[test]
    fn none_end_sec_does_not_reset_previous_end_for_overlap_check() {
        // seg0 ends at 2.0, seg1 has end_sec=None (previous_end stays 2.0),
        // seg2 starts at 2.0 — should NOT trigger overlap because
        // 2.0 + epsilon < 2.0 is false.
        let segments = vec![
            segment(Some(0.0), Some(2.0), "first"),
            TranscriptionSegment {
                start_sec: Some(2.0),
                end_sec: None,
                text: "no end".to_owned(),
                speaker: None,
                confidence: None,
            },
            segment(Some(2.0), Some(3.0), "third"),
        ];
        assert!(
            validate_segment_invariants(&segments).is_ok(),
            "seg2 starting at previous seg0's end should not be flagged as overlap"
        );
    }

    #[test]
    fn compare_segments_speaker_mismatch_ignored_when_not_required() {
        // require_speaker_exact = false with genuinely different speakers
        // → speaker_mismatches should be 0.
        let expected = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "hello".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: None,
        }];
        let observed = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "hello".to_owned(),
            speaker: Some("SPEAKER_01".to_owned()),
            confidence: None,
        }];

        let tolerance = SegmentCompatibilityTolerance {
            require_text_exact: true,
            require_speaker_exact: false,
            timestamp_tolerance_sec: 0.5,
        };

        let report = compare_segments_with_tolerance(&expected, &observed, tolerance);
        assert_eq!(
            report.speaker_mismatches, 0,
            "speaker mismatch should not be counted when require_speaker_exact is false"
        );
        assert!(report.within_tolerance());
    }
}
