//! Deterministic mapping from anonymous Sortformer lanes to caller references.
//!
//! Hard intervals are caller assertions, not biometric inference. They produce
//! an injective lane mapping only when a unique, sufficiently supported
//! assignment exists; contradictions fail closed. Soft intervals can produce
//! suggestions, but never authoritative names and never override hard evidence.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{FwError, FwResult};
use crate::model::{
    DiarizationRequest, KnownSpeakerInterval, KnownSpeakerPolicy, MAX_KNOWN_SPEAKER_INTERVALS,
};
use crate::sortformer_inference::{SORTFORMER_SPEAKER_LANES, SortformerSpeakerTurn};

pub const SORTFORMER_IDENTITY_MAPPING_SCHEMA: &str = "sortformer-lane-identity-mapping-v1";
pub const SORTFORMER_IDENTITY_MAPPING_ALGORITHM: &str =
    "unique-hard-bipartite-overlap-v1-soft-injective-lane-concentration-v3-max-confidence-union";
const HARD_MIN_SUPPORT_FRACTION: f64 = 0.5;
const SOFT_MIN_SUPPORT_MS: f64 = 500.0;
const SOFT_MIN_SUPPORT_FRACTION: f64 = 0.6;
const SOFT_MIN_MARGIN_FRACTION: f64 = 0.2;
const SCORE_EPSILON_MS: f64 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneIdentityAuthority {
    DerivedFromCallerHardInterval,
    SoftSuggestion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityCapacityStatus {
    WithinCapacity,
    SoftReferencesCapped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneIdentityAssignment {
    pub speaker_lane: usize,
    pub speaker_ref: String,
    pub authority: LaneIdentityAuthority,
    pub support_ms: f64,
    pub support_fraction: f64,
    pub margin_fraction: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerIdentityMapping {
    pub schema_version: String,
    pub algorithm_version: String,
    pub output_semantics: String,
    pub speaker_lane_capacity: usize,
    pub distinct_speaker_references: usize,
    pub capacity_status: IdentityCapacityStatus,
    pub hard_assignments: Vec<LaneIdentityAssignment>,
    pub soft_suggestions: Vec<LaneIdentityAssignment>,
    pub anonymous_lanes: Vec<usize>,
    pub model_weights_mutated: bool,
}

impl SortformerIdentityMapping {
    #[must_use]
    pub fn hard_speaker_ref(&self, lane: usize) -> Option<&str> {
        self.hard_assignments
            .iter()
            .find(|assignment| assignment.speaker_lane == lane)
            .map(|assignment| assignment.speaker_ref.as_str())
    }

    #[must_use]
    pub fn soft_speaker_suggestion(&self, lane: usize) -> Option<&str> {
        self.soft_suggestions
            .iter()
            .find(|assignment| assignment.speaker_lane == lane)
            .map(|assignment| assignment.speaker_ref.as_str())
    }
}

pub fn map_sortformer_lanes(
    turns: &[SortformerSpeakerTurn],
    hints: &[KnownSpeakerInterval],
    audio_duration_ms: u64,
) -> FwResult<SortformerIdentityMapping> {
    map_sortformer_lanes_with_checkpoint(turns, hints, audio_duration_ms, &|| Ok(()))
}

/// Map lanes while honoring cooperative cancellation throughout bounded but
/// potentially large turn/hint scans.
pub fn map_sortformer_lanes_with_checkpoint(
    turns: &[SortformerSpeakerTurn],
    hints: &[KnownSpeakerInterval],
    audio_duration_ms: u64,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<SortformerIdentityMapping> {
    checkpoint()?;
    validate_sortformer_hints(hints, audio_duration_ms)?;
    validate_turns(turns, audio_duration_ms, checkpoint)?;

    let hard_refs = hints
        .iter()
        .filter(|hint| hint.policy == KnownSpeakerPolicy::HardMustLink)
        .map(|hint| hint.speaker_ref.as_str())
        .collect::<BTreeSet<_>>();
    let all_refs = hints
        .iter()
        .map(|hint| hint.speaker_ref.as_str())
        .collect::<BTreeSet<_>>();

    let hard_assignments = solve_hard_assignments(turns, hints, &hard_refs, checkpoint)?;
    let hard_lanes = hard_assignments
        .iter()
        .map(|assignment| assignment.speaker_lane)
        .collect::<BTreeSet<_>>();
    let capacity_exceeded = all_refs.len() > SORTFORMER_SPEAKER_LANES;
    let soft_suggestions = if capacity_exceeded {
        Vec::new()
    } else {
        build_soft_suggestions(turns, hints, &hard_lanes, &hard_refs, checkpoint)?
    };
    let active_lanes = turns
        .iter()
        .map(|turn| turn.speaker)
        .collect::<BTreeSet<_>>();
    let anonymous_lanes = active_lanes
        .into_iter()
        .filter(|lane| !hard_lanes.contains(lane))
        .collect();

    checkpoint()?;
    Ok(SortformerIdentityMapping {
        schema_version: SORTFORMER_IDENTITY_MAPPING_SCHEMA.to_owned(),
        algorithm_version: SORTFORMER_IDENTITY_MAPPING_ALGORITHM.to_owned(),
        output_semantics: "speaker references originate in caller assertions, while lane bindings are overlap-derived and fail closed on ambiguity; every other lane remains anonymous; soft names are non-authoritative suggestions only".to_owned(),
        speaker_lane_capacity: SORTFORMER_SPEAKER_LANES,
        distinct_speaker_references: all_refs.len(),
        capacity_status: if capacity_exceeded {
            IdentityCapacityStatus::SoftReferencesCapped
        } else {
            IdentityCapacityStatus::WithinCapacity
        },
        hard_assignments,
        soft_suggestions,
        anonymous_lanes,
        model_weights_mutated: false,
    })
}

pub fn validate_sortformer_hints(
    hints: &[KnownSpeakerInterval],
    audio_duration_ms: u64,
) -> FwResult<()> {
    if hints.len() > MAX_KNOWN_SPEAKER_INTERVALS {
        return Err(identity_error("too many known speaker intervals"));
    }
    let request = DiarizationRequest {
        known_intervals: hints.to_vec(),
        ..DiarizationRequest::default()
    };
    request
        .validate(audio_duration_ms)
        .map_err(|error| identity_error(error.message))?;
    let hard_refs = hints
        .iter()
        .filter(|hint| hint.policy == KnownSpeakerPolicy::HardMustLink)
        .map(|hint| hint.speaker_ref.as_str())
        .collect::<BTreeSet<_>>();
    if hard_refs.len() > SORTFORMER_SPEAKER_LANES {
        return Err(identity_error(
            "hard speaker references exceed the four-lane model capacity",
        ));
    }

    Ok(())
}

/// Validate every hint invariant that does not depend on the decoded audio
/// duration. Callers can therefore reject malformed or over-capacity identity
/// requests before hashing and materializing the large model package.
pub fn validate_sortformer_hint_structure(hints: &[KnownSpeakerInterval]) -> FwResult<()> {
    validate_sortformer_hints(hints, u64::MAX)
}

fn validate_turns(
    turns: &[SortformerSpeakerTurn],
    audio_duration_ms: u64,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    let duration_seconds = audio_duration_ms as f64 / 1_000.0;
    for (index, turn) in turns.iter().enumerate() {
        if index.is_multiple_of(1_024) {
            checkpoint()?;
        }
        let start = f64::from(turn.start_seconds);
        let end = f64::from(turn.end_seconds);
        if !start.is_finite()
            || !end.is_finite()
            || start < 0.0
            || end <= start
            || end > duration_seconds + 0.050
            || turn.speaker >= SORTFORMER_SPEAKER_LANES
        {
            return Err(identity_error(
                "Sortformer turn violates the finite four-lane audio boundary",
            ));
        }
    }
    Ok(())
}

fn solve_hard_assignments(
    turns: &[SortformerSpeakerTurn],
    hints: &[KnownSpeakerInterval],
    hard_refs: &BTreeSet<&str>,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<LaneIdentityAssignment>> {
    if hard_refs.is_empty() {
        return Ok(Vec::new());
    }
    let refs = hard_refs.iter().copied().collect::<Vec<_>>();
    let support = refs
        .iter()
        .map(|speaker_ref| {
            lane_support(
                turns,
                hints,
                speaker_ref,
                KnownSpeakerPolicy::HardMustLink,
                checkpoint,
            )
        })
        .collect::<FwResult<Vec<_>>>()?;
    if support
        .iter()
        .any(|row| row.iter().sum::<f64>() <= SCORE_EPSILON_MS)
    {
        return Err(identity_error(
            "a hard speaker reference has no overlap with any model lane",
        ));
    }

    let mut candidates = Vec::new();
    enumerate_injective_assignments(
        0,
        refs.len(),
        &mut Vec::new(),
        &mut BTreeSet::new(),
        &support,
        &mut candidates,
    );
    candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let Some((best_score, best_lanes)) = candidates.first() else {
        return Err(identity_error("no hard lane assignment is possible"));
    };
    if *best_score <= SCORE_EPSILON_MS {
        return Err(identity_error("hard lane assignment has no support"));
    }
    if candidates
        .get(1)
        .is_some_and(|(score, _)| (score - best_score).abs() <= SCORE_EPSILON_MS)
    {
        return Err(identity_error(
            "hard speaker evidence has more than one equally supported lane assignment",
        ));
    }

    let mut assignments = Vec::with_capacity(refs.len());
    for (index, speaker_ref) in refs.iter().enumerate() {
        let lane = best_lanes
            .get(index)
            .copied()
            .ok_or_else(|| identity_error("hard assignment result is incomplete"))?;
        let support_row = support
            .get(index)
            .ok_or_else(|| identity_error("hard support matrix is incomplete"))?;
        let selected = support_row
            .get(lane)
            .copied()
            .ok_or_else(|| identity_error("hard assignment lane is outside model capacity"))?;
        let total = support_row.iter().sum::<f64>();
        let fraction = selected / total;
        if selected <= SCORE_EPSILON_MS || fraction <= HARD_MIN_SUPPORT_FRACTION + f64::EPSILON {
            return Err(identity_error(
                "hard speaker evidence is contradictory across model lanes",
            ));
        }
        assignments.push(LaneIdentityAssignment {
            speaker_lane: lane,
            speaker_ref: (*speaker_ref).to_owned(),
            authority: LaneIdentityAuthority::DerivedFromCallerHardInterval,
            support_ms: selected,
            support_fraction: fraction,
            margin_fraction: None,
        });
    }
    assignments.sort_by_key(|assignment| assignment.speaker_lane);
    Ok(assignments)
}

fn enumerate_injective_assignments(
    reference_index: usize,
    reference_count: usize,
    lanes: &mut Vec<usize>,
    used_lanes: &mut BTreeSet<usize>,
    support: &[[f64; SORTFORMER_SPEAKER_LANES]],
    output: &mut Vec<(f64, Vec<usize>)>,
) {
    if reference_index == reference_count {
        let score = lanes
            .iter()
            .enumerate()
            .map(|(index, lane)| support[index][*lane])
            .sum();
        output.push((score, lanes.clone()));
        return;
    }
    for lane in 0..SORTFORMER_SPEAKER_LANES {
        if used_lanes.insert(lane) {
            lanes.push(lane);
            enumerate_injective_assignments(
                reference_index + 1,
                reference_count,
                lanes,
                used_lanes,
                support,
                output,
            );
            lanes.pop();
            used_lanes.remove(&lane);
        }
    }
}

fn build_soft_suggestions(
    turns: &[SortformerSpeakerTurn],
    hints: &[KnownSpeakerInterval],
    hard_lanes: &BTreeSet<usize>,
    hard_refs: &BTreeSet<&str>,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<LaneIdentityAssignment>> {
    let soft_refs = hints
        .iter()
        .filter(|hint| hint.policy == KnownSpeakerPolicy::SoftEnrollment)
        .map(|hint| hint.speaker_ref.as_str())
        .filter(|speaker_ref| !hard_refs.contains(speaker_ref))
        .collect::<BTreeSet<_>>();
    let support_by_ref = soft_refs
        .iter()
        .map(|speaker_ref| {
            Ok((
                *speaker_ref,
                lane_support(
                    turns,
                    hints,
                    speaker_ref,
                    KnownSpeakerPolicy::SoftEnrollment,
                    checkpoint,
                )?,
            ))
        })
        .collect::<FwResult<BTreeMap<_, _>>>()?;
    let mut candidates = Vec::new();
    for (speaker_ref, support) in support_by_ref {
        let mut ranked_lanes = support.iter().copied().enumerate().collect::<Vec<_>>();
        ranked_lanes.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let Some((lane, best)) = ranked_lanes.first().copied() else {
            continue;
        };
        if hard_lanes.contains(&lane) {
            continue;
        }
        let second = ranked_lanes.get(1).map_or(0.0, |(_, score)| *score);
        let total = support.iter().sum::<f64>();
        if total <= SCORE_EPSILON_MS || best <= SCORE_EPSILON_MS {
            continue;
        }
        let fraction = best / total;
        let margin = (best - second) / best;
        if best >= SOFT_MIN_SUPPORT_MS
            && fraction >= SOFT_MIN_SUPPORT_FRACTION
            && margin >= SOFT_MIN_MARGIN_FRACTION
        {
            candidates.push(LaneIdentityAssignment {
                speaker_lane: lane,
                speaker_ref: speaker_ref.to_owned(),
                authority: LaneIdentityAuthority::SoftSuggestion,
                support_ms: best,
                support_fraction: fraction,
                margin_fraction: Some(margin),
            });
        }
    }
    candidates.sort_by(|left, right| {
        right
            .support_ms
            .total_cmp(&left.support_ms)
            .then_with(|| right.support_fraction.total_cmp(&left.support_fraction))
            .then_with(|| left.speaker_ref.cmp(&right.speaker_ref))
            .then_with(|| left.speaker_lane.cmp(&right.speaker_lane))
    });
    let mut used_lanes = BTreeSet::new();
    let mut suggestions = candidates
        .into_iter()
        .filter(|candidate| used_lanes.insert(candidate.speaker_lane))
        .collect::<Vec<_>>();
    suggestions.sort_by_key(|suggestion| suggestion.speaker_lane);
    Ok(suggestions)
}

fn lane_support(
    turns: &[SortformerSpeakerTurn],
    hints: &[KnownSpeakerInterval],
    speaker_ref: &str,
    policy: KnownSpeakerPolicy,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<[f64; SORTFORMER_SPEAKER_LANES]> {
    let mut support = [0.0; SORTFORMER_SPEAKER_LANES];
    let envelope = confidence_envelope(hints, speaker_ref, policy, checkpoint)?;
    for (turn_index, turn) in turns.iter().enumerate() {
        if turn_index.is_multiple_of(1_024) {
            checkpoint()?;
        }
        let start_ms = f64::from(turn.start_seconds) * 1_000.0;
        let end_ms = f64::from(turn.end_seconds) * 1_000.0;
        for &(hint_start_ms, hint_end_ms, confidence) in &envelope {
            let overlap =
                (end_ms.min(hint_end_ms as f64) - start_ms.max(hint_start_ms as f64)).max(0.0);
            if let Some(lane_support) = support.get_mut(turn.speaker) {
                *lane_support += overlap * confidence;
            }
        }
    }
    Ok(support)
}

/// Collapse overlapping assertions for one reference into a non-overlapping
/// envelope whose weight is the maximum active confidence. Repeating or
/// subdividing the same contextual clue must not amplify its authority or
/// change the bipartite lane assignment.
fn confidence_envelope(
    hints: &[KnownSpeakerInterval],
    speaker_ref: &str,
    policy: KnownSpeakerPolicy,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<(u64, u64, f64)>> {
    let mut events = Vec::new();
    for (index, hint) in hints
        .iter()
        .filter(|hint| hint.policy == policy && hint.speaker_ref == speaker_ref)
        .enumerate()
    {
        if index.is_multiple_of(64) {
            checkpoint()?;
        }
        let confidence_bits = if hint.confidence == 0.0 {
            0
        } else {
            hint.confidence.to_bits()
        };
        events.push((hint.start_ms, true, confidence_bits));
        events.push((hint.end_ms, false, confidence_bits));
    }
    events.sort_unstable();

    let mut envelope = Vec::new();
    let mut active = BTreeMap::<u64, usize>::new();
    let mut previous_ms = events.first().map_or(0, |event| event.0);
    let mut index = 0;
    while index < events.len() {
        if index.is_multiple_of(128) {
            checkpoint()?;
        }
        let time_ms = events[index].0;
        if time_ms > previous_ms
            && let Some((&confidence_bits, _)) = active.last_key_value()
        {
            let confidence = f64::from_bits(confidence_bits);
            if confidence > 0.0 {
                envelope.push((previous_ms, time_ms, confidence));
            }
        }
        while index < events.len() && events[index].0 == time_ms {
            let (_, is_start, confidence_bits) = events[index];
            if is_start {
                *active.entry(confidence_bits).or_default() += 1;
            } else {
                let count = active.get_mut(&confidence_bits).ok_or_else(|| {
                    identity_error("speaker-hint confidence envelope is internally unbalanced")
                })?;
                *count -= 1;
                if *count == 0 {
                    active.remove(&confidence_bits);
                }
            }
            index += 1;
        }
        previous_ms = time_ms;
    }
    if !active.is_empty() {
        return Err(identity_error(
            "speaker-hint confidence envelope is internally incomplete",
        ));
    }
    Ok(envelope)
}

fn identity_error(message: impl Into<String>) -> FwError {
    FwError::InvalidRequest(format!("sortformer.identity_mapping: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn turn(start: f32, end: f32, speaker: usize) -> SortformerSpeakerTurn {
        SortformerSpeakerTurn {
            start_seconds: start,
            end_seconds: end,
            speaker,
        }
    }

    fn hint(
        speaker_ref: &str,
        start_ms: u64,
        end_ms: u64,
        policy: KnownSpeakerPolicy,
    ) -> KnownSpeakerInterval {
        KnownSpeakerInterval {
            speaker_ref: speaker_ref.to_owned(),
            start_ms,
            end_ms,
            confidence: 1.0,
            policy,
            provenance: Some("context".to_owned()),
        }
    }

    #[test]
    fn unique_hard_intervals_map_anonymous_lanes_without_mutating_weights() {
        let turns = vec![turn(0.0, 2.0, 2), turn(2.0, 4.0, 0)];
        let hints = vec![
            hint("jeffrey", 100, 1_900, KnownSpeakerPolicy::HardMustLink),
            hint("hang", 2_100, 3_900, KnownSpeakerPolicy::HardMustLink),
        ];
        let mapping = map_sortformer_lanes(&turns, &hints, 4_000).expect("mapping");
        assert_eq!(mapping.hard_speaker_ref(2), Some("jeffrey"));
        assert_eq!(mapping.hard_speaker_ref(0), Some("hang"));
        assert_eq!(
            mapping.hard_assignments[0].authority,
            LaneIdentityAuthority::DerivedFromCallerHardInterval
        );
        assert!(!mapping.model_weights_mutated);
    }

    #[test]
    fn duplicate_hints_do_not_amplify_identity_support() {
        let turns = vec![turn(0.0, 2.0, 2)];
        let repeated = hint("jeffrey", 100, 1_900, KnownSpeakerPolicy::HardMustLink);
        let mapping = map_sortformer_lanes(&turns, &[repeated.clone(), repeated], 2_000)
            .expect("duplicate evidence must collapse to one confidence envelope");
        assert_eq!(mapping.hard_speaker_ref(2), Some("jeffrey"));
        assert_eq!(mapping.hard_assignments[0].support_ms, 1_800.0);
        assert_eq!(mapping.hard_assignments[0].support_fraction, 1.0);
    }

    #[test]
    fn overlapping_hint_confidence_uses_the_max_not_the_sum() {
        let turns = vec![turn(0.0, 2.0, 1)];
        let mut lower = hint("joel", 500, 1_500, KnownSpeakerPolicy::SoftEnrollment);
        lower.confidence = 0.5;
        let mapping = map_sortformer_lanes(
            &turns,
            &[
                hint("joel", 0, 1_000, KnownSpeakerPolicy::SoftEnrollment),
                lower,
            ],
            2_000,
        )
        .expect("overlapping evidence");
        assert_eq!(mapping.soft_suggestions.len(), 1);
        assert_eq!(mapping.soft_suggestions[0].support_ms, 1_250.0);
    }

    #[test]
    fn contradictory_hard_intervals_fail_closed() {
        let turns = vec![turn(0.0, 4.0, 0)];
        let hints = vec![
            hint("jeffrey", 0, 2_000, KnownSpeakerPolicy::HardMustLink),
            hint("hang", 2_000, 4_000, KnownSpeakerPolicy::HardMustLink),
        ];
        let error = map_sortformer_lanes(&turns, &hints, 4_000)
            .expect_err("one lane cannot carry two hard identities");
        assert!(error.to_string().contains("hard"));
    }

    #[test]
    fn equal_hard_assignments_fail_as_ambiguous() {
        let turns = vec![turn(0.0, 1.0, 0), turn(0.0, 1.0, 1)];
        let hints = vec![hint("jeffrey", 0, 1_000, KnownSpeakerPolicy::HardMustLink)];
        assert!(map_sortformer_lanes(&turns, &hints, 1_000).is_err());
    }

    #[test]
    fn soft_intervals_are_suggestions_not_authoritative_names() {
        let turns = vec![turn(0.0, 2.0, 1), turn(2.0, 3.0, 2)];
        let hints = vec![hint("joel", 0, 1_500, KnownSpeakerPolicy::SoftEnrollment)];
        let mapping = map_sortformer_lanes(&turns, &hints, 3_000).expect("mapping");
        assert_eq!(mapping.hard_speaker_ref(1), None);
        assert_eq!(mapping.soft_speaker_suggestion(1), Some("joel"));
        assert_eq!(
            mapping.soft_suggestions[0].authority,
            LaneIdentityAuthority::SoftSuggestion
        );
        assert!(mapping.anonymous_lanes.contains(&1));
    }

    #[test]
    fn weak_soft_evidence_leaves_lane_anonymous() {
        let turns = vec![turn(0.0, 0.2, 3)];
        let hints = vec![hint("joel", 0, 200, KnownSpeakerPolicy::SoftEnrollment)];
        let mapping = map_sortformer_lanes(&turns, &hints, 200).expect("mapping");
        assert!(mapping.soft_suggestions.is_empty());
        assert_eq!(mapping.anonymous_lanes, vec![3]);
    }

    #[test]
    fn one_soft_reference_cannot_label_multiple_lanes() {
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1)];
        let hints = vec![hint("joel", 0, 4_000, KnownSpeakerPolicy::SoftEnrollment)];
        let mapping = map_sortformer_lanes(&turns, &hints, 4_000).expect("mapping");
        assert!(mapping.soft_suggestions.is_empty());
        assert_eq!(mapping.anonymous_lanes, vec![0, 1]);
    }

    #[test]
    fn competing_soft_references_produce_at_most_one_suggestion_per_lane() {
        let turns = vec![turn(0.0, 3.0, 2)];
        let hints = vec![
            hint("hang", 0, 2_000, KnownSpeakerPolicy::SoftEnrollment),
            hint("joel", 0, 1_000, KnownSpeakerPolicy::SoftEnrollment),
        ];
        let mapping = map_sortformer_lanes(&turns, &hints, 3_000).expect("mapping");
        assert_eq!(mapping.soft_suggestions.len(), 1);
        assert_eq!(mapping.soft_speaker_suggestion(2), Some("hang"));
        assert_eq!(mapping.anonymous_lanes, vec![2]);
    }

    #[test]
    fn hard_identity_cannot_be_soft_suggested_on_another_lane() {
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1)];
        let hints = vec![
            hint("jeffrey", 0, 1_500, KnownSpeakerPolicy::HardMustLink),
            hint("jeffrey", 2_000, 3_500, KnownSpeakerPolicy::SoftEnrollment),
        ];
        let mapping = map_sortformer_lanes(&turns, &hints, 4_000).expect("mapping");
        assert_eq!(mapping.hard_speaker_ref(0), Some("jeffrey"));
        assert!(mapping.soft_suggestions.is_empty());
        assert_eq!(mapping.anonymous_lanes, vec![1]);
    }

    #[test]
    fn excess_soft_references_are_reported_as_capacity_capped() {
        let turns = vec![turn(0.0, 1.0, 0)];
        let hints = (0..5)
            .map(|index| {
                hint(
                    &format!("speaker-{index}"),
                    0,
                    1_000,
                    KnownSpeakerPolicy::SoftEnrollment,
                )
            })
            .collect::<Vec<_>>();
        let mapping = map_sortformer_lanes(&turns, &hints, 1_000).expect("mapping");
        assert_eq!(
            mapping.capacity_status,
            IdentityCapacityStatus::SoftReferencesCapped
        );
        assert_eq!(mapping.distinct_speaker_references, 5);
        assert_eq!(mapping.speaker_lane_capacity, 4);
        assert!(mapping.soft_suggestions.is_empty());
    }

    #[test]
    fn large_hint_scans_observe_cancellation() {
        let turns = (0..4_096)
            .map(|index| {
                let start = index as f32 * 0.08;
                turn(start, start + 0.04, index % SORTFORMER_SPEAKER_LANES)
            })
            .collect::<Vec<_>>();
        let hints = vec![hint(
            "speaker",
            0,
            327_680,
            KnownSpeakerPolicy::SoftEnrollment,
        )];
        let polls = AtomicUsize::new(0);
        let error = map_sortformer_lanes_with_checkpoint(&turns, &hints, 327_680, &|| {
            if polls.fetch_add(1, Ordering::Relaxed) >= 7 {
                Err(FwError::Cancelled("test cancellation".to_owned()))
            } else {
                Ok(())
            }
        })
        .expect_err("mapping must stop after cancellation");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(polls.load(Ordering::Relaxed) >= 8);
    }
}
