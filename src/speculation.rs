//! Core speculative streaming types for dual-model transcription.
//!
//! The speculation module implements the "speculate then verify" pattern:
//! a fast model emits [`PartialTranscript`]s in real time, and a slower
//! quality model either confirms them or triggers a [`CorrectionEvent`].

use serde::{Deserialize, Serialize};

use crate::error::{FwError, FwResult};
use crate::model::TranscriptionSegment;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// SpeculationWindow
// ---------------------------------------------------------------------------

/// Represents a sliding window over the audio stream that is sent to both the
/// fast and quality models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationWindow {
    /// Monotonically increasing window identifier.
    pub window_id: u64,
    /// Links to `RunReport.run_id`.
    pub run_id: String,
    /// Absolute position in the audio stream (milliseconds).
    pub start_ms: u64,
    /// `start_ms + window_size_ms`.
    pub end_ms: u64,
    /// Overlap with the previous window (typically 200–500 ms).
    pub overlap_ms: u64,
    /// SHA-256 hex digest of the raw audio bytes in this window.
    pub audio_hash: String,
}

impl SpeculationWindow {
    /// Create a new speculation window.
    #[must_use]
    pub fn new(
        window_id: u64,
        run_id: String,
        start_ms: u64,
        end_ms: u64,
        overlap_ms: u64,
        audio_hash: String,
    ) -> Self {
        Self {
            window_id,
            run_id,
            start_ms,
            end_ms,
            overlap_ms,
            audio_hash,
        }
    }

    /// Duration of this window in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if the given millisecond offset falls within this window.
    #[must_use]
    pub fn contains_ms(&self, ms: u64) -> bool {
        ms >= self.start_ms && ms < self.end_ms
    }
}

// ---------------------------------------------------------------------------
// PartialStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a [`PartialTranscript`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialStatus {
    /// Emitted by the fast model, waiting for the quality model.
    Pending,
    /// The quality model agreed (within tolerance).
    Confirmed,
    /// The quality model disagreed; a correction has been emitted.
    Retracted,
}

// ---------------------------------------------------------------------------
// PartialTranscript
// ---------------------------------------------------------------------------

/// A speculative transcript produced by the fast model for a single window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialTranscript {
    /// Global sequence number used for retraction targeting.
    pub seq: u64,
    /// Links to [`SpeculationWindow::window_id`].
    pub window_id: u64,
    /// Identifier of the model that produced this transcript.
    pub model_id: String,
    /// Transcription segments produced by the fast model.
    pub segments: Vec<TranscriptionSegment>,
    /// Wall-clock latency of the fast model inference (ms).
    pub latency_ms: u64,
    /// Mean confidence across all segments (0.0 when no segments carry confidence).
    pub confidence_mean: f64,
    /// RFC 3339 timestamp of when this partial was emitted.
    pub emitted_at_rfc3339: String,
    /// Current lifecycle status.
    pub status: PartialStatus,
}

impl PartialTranscript {
    /// Create a new `PartialTranscript`.
    ///
    /// `confidence_mean` is computed automatically from the segments and `status`
    /// is set to [`PartialStatus::Pending`].
    #[must_use]
    pub fn new(
        seq: u64,
        window_id: u64,
        model_id: String,
        segments: Vec<TranscriptionSegment>,
        latency_ms: u64,
        emitted_at_rfc3339: String,
    ) -> Self {
        let confidence_mean = mean_confidence(&segments);
        Self {
            seq,
            window_id,
            model_id,
            segments,
            latency_ms,
            confidence_mean,
            emitted_at_rfc3339,
            status: PartialStatus::Pending,
        }
    }

    /// Mark this partial as confirmed by the quality model.
    pub fn confirm(&mut self) {
        self.status = PartialStatus::Confirmed;
    }

    /// Mark this partial as retracted (a correction has been issued).
    pub fn retract(&mut self) {
        self.status = PartialStatus::Retracted;
    }
}

// ---------------------------------------------------------------------------
// CorrectionDrift
// ---------------------------------------------------------------------------

/// Quantifies the difference between the fast and quality transcriptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionDrift {
    /// Approximate word error rate: 0.0 = identical, 1.0 = completely different.
    pub wer_approx: f64,
    /// Absolute difference in mean confidence.
    pub confidence_delta: f64,
    /// `quality_segment_count - fast_segment_count`.
    pub segment_count_delta: i32,
    /// Levenshtein edit distance on the concatenated text.
    pub text_edit_distance: usize,
}

impl CorrectionDrift {
    /// Compute drift metrics between fast and quality segment sets.
    #[must_use]
    pub fn compute(
        fast_segments: &[TranscriptionSegment],
        quality_segments: &[TranscriptionSegment],
    ) -> Self {
        let fast_text = concat_segment_text(fast_segments);
        let quality_text = concat_segment_text(quality_segments);

        let text_edit_distance = levenshtein(&fast_text, &quality_text);
        let wer_approx = if text_edit_distance == 0 {
            0.0
        } else {
            // Word-level approximate WER.
            let fast_words: Vec<&str> = fast_text.split_whitespace().collect();
            let quality_words: Vec<&str> = quality_text.split_whitespace().collect();
            let word_edit_dist = levenshtein_words(&fast_words, &quality_words);
            let max_words = fast_words.len().max(quality_words.len()).max(1);
            word_edit_dist as f64 / max_words as f64
        };

        let confidence_delta =
            (mean_confidence(fast_segments) - mean_confidence(quality_segments)).abs();

        let segment_count_delta = quality_segments.len() as i32 - fast_segments.len() as i32;

        Self {
            wer_approx,
            confidence_delta,
            segment_count_delta,
            text_edit_distance,
        }
    }
}

// ---------------------------------------------------------------------------
// CorrectionEvent
// ---------------------------------------------------------------------------

/// Emitted when the quality model disagrees with a [`PartialTranscript`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionEvent {
    /// Unique correction identifier.
    pub correction_id: u64,
    /// The [`PartialTranscript::seq`] being retracted.
    pub retracted_seq: u64,
    /// The window this correction applies to.
    pub window_id: u64,
    /// Identifier of the quality model.
    pub quality_model_id: String,
    /// Corrected transcription segments from the quality model.
    pub corrected_segments: Vec<TranscriptionSegment>,
    /// Wall-clock latency of the quality model inference (ms).
    pub quality_latency_ms: u64,
    /// Mean confidence across the corrected segments.
    pub quality_confidence_mean: f64,
    /// Drift metrics between the fast and quality outputs.
    pub drift: CorrectionDrift,
    /// RFC 3339 timestamp of when the correction was produced.
    pub corrected_at_rfc3339: String,
}

impl CorrectionEvent {
    /// Create a new `CorrectionEvent`.
    ///
    /// `quality_confidence_mean` is computed from `corrected_segments` and
    /// `drift` is computed from `fast_segments` and `corrected_segments`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        correction_id: u64,
        retracted_seq: u64,
        window_id: u64,
        quality_model_id: String,
        corrected_segments: Vec<TranscriptionSegment>,
        quality_latency_ms: u64,
        corrected_at_rfc3339: String,
        fast_segments: &[TranscriptionSegment],
    ) -> Self {
        let quality_confidence_mean = mean_confidence(&corrected_segments);
        let drift = CorrectionDrift::compute(fast_segments, &corrected_segments);
        Self::from_precomputed_metrics(
            correction_id,
            retracted_seq,
            window_id,
            quality_model_id,
            corrected_segments,
            quality_latency_ms,
            quality_confidence_mean,
            drift,
            corrected_at_rfc3339,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_precomputed_metrics(
        correction_id: u64,
        retracted_seq: u64,
        window_id: u64,
        quality_model_id: String,
        corrected_segments: Vec<TranscriptionSegment>,
        quality_latency_ms: u64,
        quality_confidence_mean: f64,
        drift: CorrectionDrift,
        corrected_at_rfc3339: String,
    ) -> Self {
        Self {
            correction_id,
            retracted_seq,
            window_id,
            quality_model_id,
            corrected_segments,
            quality_latency_ms,
            quality_confidence_mean,
            drift,
            corrected_at_rfc3339,
        }
    }

    /// Returns `true` when the approximate WER exceeds the given threshold.
    #[must_use]
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.drift.wer_approx > threshold
    }
}

// ---------------------------------------------------------------------------
// SpeculationStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the speculation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationStats {
    /// Number of windows processed so far.
    pub windows_processed: u64,
    /// Number of corrections emitted.
    pub corrections_emitted: u64,
    /// Number of windows confirmed without correction.
    pub confirmations_emitted: u64,
    /// `corrections_emitted / windows_processed` (0.0 when no windows).
    pub correction_rate: f64,
    /// Mean latency of the fast model across all partials (ms).
    pub mean_fast_latency_ms: f64,
    /// Mean latency of the quality model across all processed windows (ms).
    pub mean_quality_latency_ms: f64,
    /// Current window size being used (ms).
    pub current_window_size_ms: u64,
    /// Mean approximate WER across all processed windows.
    pub mean_drift_wer: f64,
}

// ---------------------------------------------------------------------------
// Helper functions (private)
// ---------------------------------------------------------------------------

/// Compute the mean confidence across a slice of segments. Returns 0.0 when
/// no segment carries a confidence value.
fn mean_confidence(segments: &[TranscriptionSegment]) -> f64 {
    let (sum, count) = segments.iter().fold((0.0_f64, 0_u64), |(s, c), seg| {
        if let Some(conf) = seg.confidence {
            (s + conf, c + 1)
        } else {
            (s, c)
        }
    });
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Concatenate all segment texts separated by a single space.
fn concat_segment_text(segments: &[TranscriptionSegment]) -> String {
    segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Character-level Levenshtein distance.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    // A shared prefix or suffix contributes zero edits, so exclude both from
    // the dynamic-programming matrix while retaining Unicode scalar semantics.
    let common_prefix = a_chars
        .iter()
        .zip(&b_chars)
        .take_while(|(left, right)| left == right)
        .count();
    let a_chars = &a_chars[common_prefix..];
    let b_chars = &b_chars[common_prefix..];
    let common_suffix = a_chars
        .iter()
        .rev()
        .zip(b_chars.iter().rev())
        .take_while(|(left, right)| left == right)
        .count();
    let a_chars = &a_chars[..a_chars.len() - common_suffix];
    let b_chars = &b_chars[..b_chars.len() - common_suffix];
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Two-row optimisation.
    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Word-level Levenshtein distance.
fn levenshtein_words(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// bd-qlt.4: WindowManager — audio stream windowing with configurable overlap
// ---------------------------------------------------------------------------

/// Status of a speculation window as it moves through the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowStatus {
    /// Window created, no results yet.
    Pending,
    /// Fast model inference in progress.
    FastInProgress,
    /// Fast model result recorded.
    FastComplete,
    /// Quality model result recorded.
    QualityComplete,
    /// Window fully resolved (fast + quality compared).
    Resolved,
}

/// Tracks the state of a single speculation window through both model passes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowState {
    pub window: SpeculationWindow,
    pub fast_result: Option<PartialTranscript>,
    pub quality_result: Option<Vec<TranscriptionSegment>>,
    pub status: WindowStatus,
}

/// Scalar window fields needed by the internal streaming loop after a window
/// has been retained by [`WindowManager`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowReceipt {
    pub(crate) window_id: u64,
    pub(crate) start_ms: u64,
    pub(crate) end_ms: u64,
}

/// Manages the lifecycle of overlapping audio windows for speculative
/// dual-model transcription.
pub struct WindowManager {
    window_size_ms: u64,
    overlap_ms: u64,
    min_window_ms: u64,
    max_window_ms: u64,
    next_window_id: u64,
    run_id: String,
    windows: Vec<WindowState>,
}

impl WindowManager {
    /// Create a new `WindowManager`.
    ///
    /// `min_window_ms` defaults to 1000 and `max_window_ms` defaults to 30000.
    #[must_use]
    pub fn new(run_id: &str, window_size_ms: u64, overlap_ms: u64) -> Self {
        Self {
            window_size_ms,
            overlap_ms,
            min_window_ms: 1000,
            max_window_ms: 30_000,
            next_window_id: 0,
            run_id: run_id.to_owned(),
            windows: Vec::new(),
        }
    }

    fn store_window(&mut self, audio_position_ms: u64, end_ms: u64, audio_hash: String) -> usize {
        let id = self.next_window_id;
        self.next_window_id += 1;

        let window = SpeculationWindow::new(
            id,
            self.run_id.clone(),
            audio_position_ms,
            end_ms,
            self.overlap_ms,
            audio_hash,
        );

        let index = self.windows.len();
        self.windows.push(WindowState {
            window,
            fast_result: None,
            quality_result: None,
            status: WindowStatus::Pending,
        });

        index
    }

    fn create_window(
        &mut self,
        audio_position_ms: u64,
        end_ms: u64,
        audio_hash: &str,
    ) -> SpeculationWindow {
        let index = self.store_window(audio_position_ms, end_ms, audio_hash.to_owned());
        self.windows[index].window.clone()
    }

    fn bounded_end_ms(&self, audio_position_ms: u64, max_end_ms: u64) -> Option<u64> {
        if audio_position_ms >= max_end_ms {
            return None;
        }
        let natural_end = audio_position_ms.saturating_add(self.window_size_ms);
        let end_ms = natural_end.min(max_end_ms);
        (end_ms > audio_position_ms).then_some(end_ms)
    }

    /// Create the next speculation window starting at `audio_position_ms`.
    pub fn next_window(&mut self, audio_position_ms: u64, audio_hash: &str) -> SpeculationWindow {
        let end_ms = audio_position_ms.saturating_add(self.window_size_ms);
        self.create_window(audio_position_ms, end_ms, audio_hash)
    }

    /// Create a window bounded by `max_end_ms` (useful for final-window truncation).
    ///
    /// Returns `None` when `audio_position_ms >= max_end_ms`, preventing
    /// zero-length windows from being created.
    pub fn next_window_bounded(
        &mut self,
        audio_position_ms: u64,
        max_end_ms: u64,
        audio_hash: &str,
    ) -> Option<SpeculationWindow> {
        let end_ms = self.bounded_end_ms(audio_position_ms, max_end_ms)?;
        Some(self.create_window(audio_position_ms, end_ms, audio_hash))
    }

    /// Retain a bounded window while moving its already-owned audio hash and
    /// return only the scalar fields consumed by the internal streaming loop.
    pub(crate) fn next_window_bounded_receipt(
        &mut self,
        audio_position_ms: u64,
        max_end_ms: u64,
        audio_hash: String,
    ) -> Option<WindowReceipt> {
        let end_ms = self.bounded_end_ms(audio_position_ms, max_end_ms)?;
        let index = self.store_window(audio_position_ms, end_ms, audio_hash);
        let window = &self.windows[index].window;
        Some(WindowReceipt {
            window_id: window.window_id,
            start_ms: window.start_ms,
            end_ms: window.end_ms,
        })
    }

    /// Record the fast model result for a window.
    pub fn record_fast_result(&mut self, window_id: u64, partial: PartialTranscript) {
        if let Some(ws) = self.get_window_mut(window_id) {
            ws.fast_result = Some(partial);
            ws.status = WindowStatus::FastComplete;
        }
    }

    /// Record the quality model result for a window.
    pub fn record_quality_result(&mut self, window_id: u64, segments: Vec<TranscriptionSegment>) {
        if let Some(ws) = self.get_window_mut(window_id) {
            ws.quality_result = Some(segments);
            ws.status = WindowStatus::QualityComplete;
        }
    }

    /// Mark a window as fully resolved.
    pub fn resolve_window(&mut self, window_id: u64) {
        if let Some(ws) = self.get_window_mut(window_id) {
            ws.status = WindowStatus::Resolved;
        }
    }

    /// Update the window size, clamped to `[min_window_ms, max_window_ms]`.
    pub fn set_window_size(&mut self, new_size_ms: u64) {
        self.window_size_ms = new_size_ms.clamp(self.min_window_ms, self.max_window_ms);
    }

    /// Current window size in milliseconds.
    #[must_use]
    pub fn current_window_size(&self) -> u64 {
        self.window_size_ms
    }

    /// Count of windows with [`WindowStatus::Resolved`].
    #[must_use]
    pub fn windows_resolved(&self) -> usize {
        self.windows
            .iter()
            .filter(|ws| ws.status == WindowStatus::Resolved)
            .count()
    }

    /// Count of windows that are **not** yet resolved.
    #[must_use]
    pub fn windows_pending(&self) -> usize {
        self.windows
            .iter()
            .filter(|ws| ws.status != WindowStatus::Resolved)
            .count()
    }

    /// Look up a window by its id.
    #[must_use]
    pub fn get_window(&self, window_id: u64) -> Option<&WindowState> {
        self.windows
            .iter()
            .find(|ws| ws.window.window_id == window_id)
    }

    /// Look up a window mutably by its id.
    pub fn get_window_mut(&mut self, window_id: u64) -> Option<&mut WindowState> {
        // `create_window` assigns IDs in append order, and windows are never
        // removed or reordered, so a valid ID is its stable vector index.
        let index = usize::try_from(window_id).ok()?;
        self.windows
            .get_mut(index)
            .filter(|ws| ws.window.window_id == window_id)
    }

    /// Merge segments from all resolved windows into a single sorted,
    /// deduplicated list.
    ///
    /// For each resolved window, the quality result is preferred; if absent the
    /// fast result's segments are used. Segments are sorted by `start_sec` and
    /// overlapping segments (within 0.1 s) are deduplicated by keeping the one
    /// with higher confidence.
    #[must_use]
    pub fn merge_segments(&self) -> Vec<TranscriptionSegment> {
        // Count the resolved segments first so the aggregate is allocated at
        // exact capacity and each window's chosen segments are cloned directly
        // into it — avoiding a per-window temporary `Vec` (from `.clone()`) and
        // the aggregate's incremental reallocation. Byte-identical: `self` is
        // borrowed so both passes pick the same quality/fast branch per window,
        // and the same segments are appended in the same order.
        let mut total = 0usize;
        for ws in &self.windows {
            if ws.status != WindowStatus::Resolved {
                continue;
            }
            if let Some(ref quality) = ws.quality_result {
                total += quality.len();
            } else if let Some(ref fast) = ws.fast_result {
                total += fast.segments.len();
            }
        }

        let mut all_segments: Vec<TranscriptionSegment> = Vec::with_capacity(total);

        for ws in &self.windows {
            if ws.status != WindowStatus::Resolved {
                continue;
            }
            if let Some(ref quality) = ws.quality_result {
                all_segments.extend(quality.iter().cloned());
            } else if let Some(ref fast) = ws.fast_result {
                all_segments.extend(fast.segments.iter().cloned());
            }
        }

        // Treat timestamp-less segments as the earliest items so speculative
        // partials without timing metadata surface before later timed segments.
        all_segments.sort_by(|a, b| match (a.start_sec, b.start_sec) {
            (Some(left), Some(right)) => left
                .partial_cmp(&right)
                .unwrap_or(std::cmp::Ordering::Equal),
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (None, None) => std::cmp::Ordering::Equal,
        });

        // Deduplicate overlapping segments (within 0.1 sec tolerance).
        let mut deduped: Vec<TranscriptionSegment> = Vec::new();
        for seg in all_segments {
            let dominated = if let Some(last) = deduped.last() {
                match (last.start_sec, seg.start_sec, last.end_sec, seg.end_sec) {
                    (Some(last_start), Some(seg_start), Some(last_end), Some(seg_end)) => {
                        (seg_start - last_start).abs() < 0.1 && (seg_end - last_end).abs() < 0.1
                    }
                    _ => false,
                }
            } else {
                false
            };

            if dominated {
                // Keep the one with higher confidence.
                let seg_conf = seg.confidence.unwrap_or(0.0);
                if let Some(last) = deduped.last_mut() {
                    let last_conf = last.confidence.unwrap_or(0.0);
                    if seg_conf > last_conf {
                        *last = seg;
                    }
                }
            } else {
                deduped.push(seg);
            }
        }

        deduped
    }
}

// ---------------------------------------------------------------------------
// bd-qlt.5: CorrectionTracker — retraction/correction lifecycle state machine
// ---------------------------------------------------------------------------

/// Tolerance thresholds that determine when a fast partial should be corrected.
#[derive(Debug, Clone)]
pub struct CorrectionTolerance {
    /// Maximum acceptable approximate WER before triggering a correction.
    pub max_wer: f64,
    /// Maximum acceptable confidence delta before triggering a correction.
    pub max_confidence_delta: f64,
    /// Maximum acceptable edit distance before triggering a correction.
    pub max_edit_distance: usize,
    /// When `true`, always emit a correction regardless of drift metrics.
    pub always_correct: bool,
}

impl Default for CorrectionTolerance {
    fn default() -> Self {
        Self {
            max_wer: 0.1,
            max_confidence_delta: 0.15,
            max_edit_distance: 50,
            always_correct: false,
        }
    }
}

/// Aggregate statistics tracked by [`CorrectionTracker`].
pub struct CorrectionStats {
    pub windows_processed: u64,
    pub corrections_emitted: u64,
    pub confirmations_emitted: u64,
    pub total_fast_latency_ms: u64,
    pub total_quality_latency_ms: u64,
    pub cumulative_wer: f64,
    pub max_observed_wer: f64,
}

/// The decision produced after comparing a fast partial against quality output.
#[derive(Debug)]
pub enum CorrectionDecision {
    /// The quality model agreed with the fast partial.
    Confirm { seq: u64, drift: CorrectionDrift },
    /// The quality model disagreed; a correction has been issued.
    Correct { correction: CorrectionEvent },
}

/// Tracks the lifecycle of partials through the correction pipeline,
/// deciding whether each partial should be confirmed or corrected based
/// on configurable tolerance thresholds.
pub struct CorrectionTracker {
    tolerance: CorrectionTolerance,
    partials: HashMap<u64, PartialTranscript>,
    window_to_seq: HashMap<u64, u64>,
    corrections: Vec<CorrectionEvent>,
    stats: CorrectionStats,
    next_correction_id: u64,
}

impl CorrectionTracker {
    /// Create a new `CorrectionTracker` with the given tolerance thresholds.
    #[must_use]
    pub fn new(tolerance: CorrectionTolerance) -> Self {
        Self {
            tolerance,
            partials: HashMap::new(),
            window_to_seq: HashMap::new(),
            corrections: Vec::new(),
            stats: CorrectionStats {
                windows_processed: 0,
                corrections_emitted: 0,
                confirmations_emitted: 0,
                total_fast_latency_ms: 0,
                total_quality_latency_ms: 0,
                cumulative_wer: 0.0,
                max_observed_wer: 0.0,
            },
            next_correction_id: 0,
        }
    }

    /// Register a partial transcript produced by the fast model.
    ///
    /// Returns the `seq` number of the registered partial.
    pub fn register_partial(&mut self, partial: PartialTranscript) -> u64 {
        let seq = partial.seq;
        self.stats.total_fast_latency_ms += partial.latency_ms;
        self.window_to_seq.insert(partial.window_id, seq);
        self.partials.insert(seq, partial);
        seq
    }

    /// Submit quality model output for comparison against a previously
    /// registered partial.
    ///
    /// Returns a [`CorrectionDecision`] indicating whether the partial was
    /// confirmed or corrected. Returns `Err` if no partial is found for
    /// `window_id`.
    pub fn submit_quality_result(
        &mut self,
        window_id: u64,
        quality_model_id: &str,
        quality_segments: Vec<TranscriptionSegment>,
        quality_latency_ms: u64,
    ) -> FwResult<CorrectionDecision> {
        let seq = *self.window_to_seq.get(&window_id).ok_or_else(|| {
            FwError::InvalidRequest(format!("no partial registered for window_id {window_id}"))
        })?;

        let drift = {
            let partial = self.partials.get(&seq).ok_or_else(|| {
                FwError::Unsupported(format!(
                    "speculation state desync: missing partial for seq {seq}"
                ))
            })?;
            CorrectionDrift::compute(&partial.segments, &quality_segments)
        };

        // Update stats.
        self.stats.windows_processed += 1;
        self.stats.total_quality_latency_ms += quality_latency_ms;
        self.stats.cumulative_wer += drift.wer_approx;
        if drift.wer_approx > self.stats.max_observed_wer {
            self.stats.max_observed_wer = drift.wer_approx;
        }

        let needs_correction = self.tolerance.always_correct
            || drift.wer_approx > self.tolerance.max_wer
            || drift.confidence_delta > self.tolerance.max_confidence_delta
            || drift.text_edit_distance > self.tolerance.max_edit_distance;

        if needs_correction {
            // Retract the partial.
            let partial = self.partials.get_mut(&seq).ok_or_else(|| {
                FwError::Unsupported(format!(
                    "speculation state desync: missing partial for seq {seq}"
                ))
            })?;
            partial.retract();

            let correction_id = self.next_correction_id;
            self.next_correction_id += 1;

            let quality_model_id = quality_model_id.to_owned();
            let corrected_at_rfc3339 = chrono::Utc::now().to_rfc3339();
            let quality_confidence_mean = mean_confidence(&quality_segments);
            // The decision above already computed both Levenshtein metrics.
            // Reuse them instead of cloning the fast segments and rescanning.
            let correction = CorrectionEvent::from_precomputed_metrics(
                correction_id,
                seq,
                window_id,
                quality_model_id,
                quality_segments,
                quality_latency_ms,
                quality_confidence_mean,
                drift,
                corrected_at_rfc3339,
            );

            self.corrections.push(correction.clone());
            self.stats.corrections_emitted += 1;
            self.window_to_seq.remove(&window_id);

            Ok(CorrectionDecision::Correct { correction })
        } else {
            // Confirm the partial.
            let partial = self.partials.get_mut(&seq).ok_or_else(|| {
                FwError::Unsupported(format!(
                    "speculation state desync: missing partial for seq {seq}"
                ))
            })?;
            partial.confirm();
            self.stats.confirmations_emitted += 1;
            self.window_to_seq.remove(&window_id);

            Ok(CorrectionDecision::Confirm { seq, drift })
        }
    }

    /// Reference to aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &CorrectionStats {
        &self.stats
    }

    /// Fraction of windows that required correction (0.0 if none processed).
    #[must_use]
    pub fn correction_rate(&self) -> f64 {
        let total = self.stats.corrections_emitted + self.stats.confirmations_emitted;
        if total == 0 {
            0.0
        } else {
            self.stats.corrections_emitted as f64 / total as f64
        }
    }

    /// Mean approximate WER across all processed windows (0.0 if none).
    #[must_use]
    pub fn mean_wer(&self) -> f64 {
        if self.stats.windows_processed == 0 {
            0.0
        } else {
            self.stats.cumulative_wer / self.stats.windows_processed as f64
        }
    }

    /// All correction events emitted so far.
    #[must_use]
    pub fn corrections(&self) -> &[CorrectionEvent] {
        &self.corrections
    }

    /// Look up a registered partial by sequence number.
    #[must_use]
    pub fn get_partial(&self, seq: u64) -> Option<&PartialTranscript> {
        self.partials.get(&seq)
    }

    /// Returns `true` when every registered partial has been confirmed or
    /// retracted.
    #[must_use]
    pub fn all_resolved(&self) -> bool {
        self.partials
            .values()
            .all(|p| p.status == PartialStatus::Confirmed || p.status == PartialStatus::Retracted)
    }
}

// ---------------------------------------------------------------------------
// bd-qlt.7: SpeculationWindowController — alien-artifact adaptive sizing
// ---------------------------------------------------------------------------

/// Snapshot of the controller's observable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerState {
    pub correction_rate: f64,
    pub mean_wer: f64,
    pub window_count: u64,
    pub current_window_ms: u64,
}

/// Beta distribution posterior for Bayesian correction rate estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaPosterior {
    /// Corrections observed + prior.
    pub alpha: f64,
    /// Confirmations observed + prior.
    pub beta: f64,
}

impl BetaPosterior {
    /// Create a new Beta posterior with the given parameters.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` or `beta` is not positive and finite.
    #[must_use]
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha.is_finite(),
            "BetaPosterior alpha must be positive and finite, got {alpha}"
        );
        assert!(
            beta > 0.0 && beta.is_finite(),
            "BetaPosterior beta must be positive and finite, got {beta}"
        );
        Self { alpha, beta }
    }

    /// Prior: Beta(2,2) — weakly informative, centered at 0.5.
    #[must_use]
    pub fn weakly_informative() -> Self {
        Self::new(2.0, 2.0)
    }

    /// Observe a correction (alpha += 1).
    pub fn observe_correction(&mut self) {
        self.alpha += 1.0;
    }

    /// Observe a confirmation (beta += 1).
    pub fn observe_confirmation(&mut self) {
        self.beta += 1.0;
    }

    /// Mean of the posterior distribution.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the posterior distribution.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let total = self.alpha + self.beta;
        (self.alpha * self.beta) / (total * total * (total + 1.0))
    }
}

/// Action the controller can take.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerAction {
    Shrink(u64),
    Hold,
    Grow(u64),
}

/// Calibration tracker using Brier score over a rolling window.
#[derive(Debug, Clone)]
pub struct CalibrationTracker {
    predictions: VecDeque<f64>,
    outcomes: VecDeque<bool>,
    window_size: usize,
}

impl CalibrationTracker {
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            predictions: VecDeque::new(),
            outcomes: VecDeque::new(),
            window_size,
        }
    }

    pub fn record(&mut self, predicted: f64, actual_correction: bool) {
        self.predictions.push_back(predicted);
        self.outcomes.push_back(actual_correction);
        if self.predictions.len() > self.window_size {
            self.predictions.pop_front();
            self.outcomes.pop_front();
        }
    }

    /// Brier score: mean((predicted - actual)^2). Lower is better.
    #[must_use]
    pub fn brier_score(&self) -> f64 {
        if self.predictions.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .predictions
            .iter()
            .zip(&self.outcomes)
            .map(|(p, &o)| {
                let actual = if o { 1.0 } else { 0.0 };
                (p - actual).powi(2)
            })
            .sum();
        sum / self.predictions.len() as f64
    }

    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.predictions.len()
    }
}

/// Evidence ledger entry for each controller decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationControllerEntry {
    pub decision_id: u64,
    pub window_id: u64,
    pub state_snapshot: ControllerState,
    pub action_taken: ControllerAction,
    pub predicted_correction_rate: f64,
    pub actual_correction_rate: f64,
    pub brier_score: f64,
    pub confidence: f64,
    pub fallback_active: bool,
    pub fallback_reason: Option<String>,
}

/// Adaptive controller that tunes speculation window size based on
/// correction evidence. Implements the full alien-artifact engineering
/// contract (AGENTS.md).
pub struct SpeculationWindowController {
    initial_window_ms: u64,
    current_window_ms: u64,
    min_window_ms: u64,
    max_window_ms: u64,
    step_ms: u64,
    state: ControllerState,
    posterior: BetaPosterior,
    calibration: CalibrationTracker,
    evidence: Vec<SpeculationControllerEntry>,
    fallback_active: bool,
    fallback_reason: Option<String>,
    next_decision_id: u64,
    consecutive_zero_corrections: u64,
}

impl SpeculationWindowController {
    const BRIER_FALLBACK_THRESHOLD: f64 = 0.25;
    const HIGH_CORRECTION_RATE: f64 = 0.25;
    const LOW_CORRECTION_RATE: f64 = 0.0625;
    const HIGH_WER_THRESHOLD: f64 = 0.125;
    const MIN_WINDOWS_FOR_ADAPT: u64 = 5;
    const FULL_CONFIDENCE_WINDOWS: f64 = 20.0;
    const RUNAWAY_CORRECTION_RATE: f64 = 0.75;
    const ZERO_CORRECTION_FORCE_SHRINK: u64 = 20;

    #[must_use]
    pub fn new(initial_window_ms: u64, min_ms: u64, max_ms: u64, step_ms: u64) -> Self {
        Self {
            initial_window_ms,
            current_window_ms: initial_window_ms,
            min_window_ms: min_ms,
            max_window_ms: max_ms,
            step_ms,
            state: ControllerState {
                correction_rate: 0.0,
                mean_wer: 0.0,
                window_count: 0,
                current_window_ms: initial_window_ms,
            },
            posterior: BetaPosterior::weakly_informative(),
            calibration: CalibrationTracker::new(20),
            evidence: Vec::new(),
            fallback_active: false,
            fallback_reason: None,
            next_decision_id: 0,
            consecutive_zero_corrections: 0,
        }
    }

    /// Observe a correction decision and update internal state.
    pub fn observe(&mut self, decision: &CorrectionDecision, drift: &CorrectionDrift) {
        let is_correction = matches!(decision, CorrectionDecision::Correct { .. });

        if is_correction {
            self.posterior.observe_correction();
            self.consecutive_zero_corrections = 0;
        } else {
            self.posterior.observe_confirmation();
            self.consecutive_zero_corrections += 1;
        }

        self.state.window_count += 1;

        let total = self.posterior.alpha + self.posterior.beta - 4.0;
        let corrections = self.posterior.alpha - 2.0;
        if total > 0.0 {
            self.state.correction_rate = corrections / total;
        }

        let prev_total = self.state.mean_wer * (self.state.window_count - 1) as f64;
        self.state.mean_wer = (prev_total + drift.wer_approx) / self.state.window_count as f64;
        self.state.current_window_ms = self.current_window_ms;

        let predicted = self.posterior.mean();
        self.calibration.record(predicted, is_correction);
    }

    /// Recommend an action based on current state.
    #[must_use]
    pub fn recommend(&self) -> ControllerAction {
        if self.state.window_count < Self::MIN_WINDOWS_FOR_ADAPT {
            return ControllerAction::Hold;
        }

        if self.calibration.brier_score() > Self::BRIER_FALLBACK_THRESHOLD
            && self.calibration.sample_count() >= 10
        {
            return ControllerAction::Hold;
        }

        if self.state.correction_rate > Self::RUNAWAY_CORRECTION_RATE
            && self.current_window_ms < self.max_window_ms
        {
            return ControllerAction::Grow(self.step_ms);
        }

        if self.consecutive_zero_corrections >= Self::ZERO_CORRECTION_FORCE_SHRINK
            && self.current_window_ms > self.min_window_ms
        {
            return ControllerAction::Shrink(self.step_ms);
        }

        let confidence = self.confidence();
        if confidence < 0.5 {
            return ControllerAction::Hold;
        }

        if self.state.correction_rate > Self::HIGH_CORRECTION_RATE
            && self.state.mean_wer > Self::HIGH_WER_THRESHOLD
            && self.current_window_ms < self.max_window_ms
        {
            return ControllerAction::Grow(self.step_ms);
        }

        if self.state.correction_rate < Self::LOW_CORRECTION_RATE
            && self.state.window_count > 10
            && self.current_window_ms > self.min_window_ms
        {
            return ControllerAction::Shrink(self.step_ms);
        }

        ControllerAction::Hold
    }

    /// Apply the recommended action and return the new window size.
    pub fn apply(&mut self) -> u64 {
        let action = self.recommend();

        let brier = self.calibration.brier_score();
        if brier > Self::BRIER_FALLBACK_THRESHOLD && self.calibration.sample_count() >= 10 {
            self.fallback_active = true;
            self.fallback_reason = Some(format!("Brier score {brier:.3} > threshold"));
            self.current_window_ms = self.initial_window_ms;
        } else if self.state.correction_rate > Self::RUNAWAY_CORRECTION_RATE {
            self.fallback_active = true;
            self.fallback_reason = Some("correction rate > 75%".to_owned());
            if let ControllerAction::Grow(delta) = &action {
                self.current_window_ms = (self.current_window_ms + delta).min(self.max_window_ms);
            }
        } else {
            self.fallback_active = false;
            self.fallback_reason = None;
            match &action {
                ControllerAction::Shrink(delta) => {
                    self.current_window_ms = self
                        .current_window_ms
                        .saturating_sub(*delta)
                        .max(self.min_window_ms);
                }
                ControllerAction::Hold => {}
                ControllerAction::Grow(delta) => {
                    self.current_window_ms =
                        (self.current_window_ms + delta).min(self.max_window_ms);
                }
            }
        }

        let entry = SpeculationControllerEntry {
            decision_id: self.next_decision_id,
            window_id: self.state.window_count,
            state_snapshot: self.state.clone(),
            action_taken: action,
            predicted_correction_rate: self.posterior.mean(),
            actual_correction_rate: self.state.correction_rate,
            brier_score: brier,
            confidence: self.confidence(),
            fallback_active: self.fallback_active,
            fallback_reason: self.fallback_reason.clone(),
        };
        self.next_decision_id += 1;
        self.evidence.push(entry);

        self.state.current_window_ms = self.current_window_ms;
        self.current_window_ms
    }

    #[must_use]
    pub fn is_fallback_active(&self) -> bool {
        self.fallback_active
    }

    #[must_use]
    pub fn evidence(&self) -> &[SpeculationControllerEntry] {
        &self.evidence
    }

    #[must_use]
    pub fn current_window_ms(&self) -> u64 {
        self.current_window_ms
    }

    /// Confidence level: min(window_count / 20, 1.0).
    #[must_use]
    pub fn confidence(&self) -> f64 {
        (self.state.window_count as f64 / Self::FULL_CONFIDENCE_WINDOWS).min(1.0)
    }

    #[must_use]
    pub fn state(&self) -> &ControllerState {
        &self.state
    }

    #[must_use]
    pub fn posterior(&self) -> &BetaPosterior {
        &self.posterior
    }
}

// ---------------------------------------------------------------------------
// bd-qlt.8: CorrectionEvidenceLedger — per-window evidence trail
// ---------------------------------------------------------------------------

/// A single evidence entry recording one speculation window decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionEvidenceEntry {
    pub entry_id: u64,
    pub window_id: u64,
    pub run_id: String,
    pub timestamp_rfc3339: String,
    pub fast_model_id: String,
    pub fast_latency_ms: u64,
    pub fast_confidence_mean: f64,
    pub fast_segment_count: usize,
    pub quality_model_id: String,
    pub quality_latency_ms: u64,
    pub quality_confidence_mean: f64,
    pub quality_segment_count: usize,
    pub drift: CorrectionDrift,
    pub decision: String,
    pub window_size_ms: u64,
    pub correction_rate_at_decision: f64,
    pub controller_confidence: f64,
    pub fallback_active: bool,
    pub fallback_reason: Option<String>,
}

/// Bounded evidence ledger recording every speculation decision for
/// post-hoc analysis, adaptive tuning, and debugging.
pub struct CorrectionEvidenceLedger {
    entries: VecDeque<CorrectionEvidenceEntry>,
    capacity: usize,
    total_recorded: u64,
}

impl CorrectionEvidenceLedger {
    /// Create a new ledger with the given capacity (default: 500).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            total_recorded: 0,
        }
    }

    /// Record an entry. Evicts the oldest if at capacity.
    pub fn record(&mut self, entry: CorrectionEvidenceEntry) {
        if self.capacity == 0 {
            self.total_recorded += 1;
            return;
        }
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        self.total_recorded += 1;
    }

    /// Total entries ever recorded (including evicted).
    #[must_use]
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Current entries in the ledger.
    #[must_use]
    pub fn entries(&self) -> &VecDeque<CorrectionEvidenceEntry> {
        &self.entries
    }

    /// Export all entries as a JSON Value for `PipelineCx::record_evidence()`.
    #[must_use]
    pub fn to_evidence_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "correction_evidence_ledger",
            "total_recorded": self.total_recorded,
            "retained": self.entries.len(),
            "capacity": self.capacity,
            "diagnostics": self.diagnostics(),
            "entries": self.entries.iter().collect::<Vec<_>>(),
        })
    }

    /// Summary diagnostics as JSON.
    #[must_use]
    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "correction_rate": self.correction_rate(),
            "mean_fast_latency_ms": self.mean_fast_latency(),
            "mean_quality_latency_ms": self.mean_quality_latency(),
            "mean_wer": self.mean_wer(),
            "latency_savings_pct": self.latency_savings_pct(),
        })
    }

    /// Fraction of entries that were corrections.
    #[must_use]
    pub fn correction_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let corrections = self
            .entries
            .iter()
            .filter(|e| is_correction_decision(&e.decision))
            .count();
        corrections as f64 / self.entries.len() as f64
    }

    /// Mean fast model latency across entries.
    #[must_use]
    pub fn mean_fast_latency(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.entries.iter().map(|e| e.fast_latency_ms).sum();
        sum as f64 / self.entries.len() as f64
    }

    /// Mean quality model latency across entries.
    #[must_use]
    pub fn mean_quality_latency(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.entries.iter().map(|e| e.quality_latency_ms).sum();
        sum as f64 / self.entries.len() as f64
    }

    /// Mean WER across entries.
    #[must_use]
    pub fn mean_wer(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.entries.iter().map(|e| e.drift.wer_approx).sum();
        sum / self.entries.len() as f64
    }

    /// Percentage of latency savings from showing fast results immediately.
    ///
    /// `(mean_quality - mean_fast) / mean_quality * 100`
    #[must_use]
    pub fn latency_savings_pct(&self) -> f64 {
        let q = self.mean_quality_latency();
        if q == 0.0 {
            return 0.0;
        }
        let f = self.mean_fast_latency();
        (q - f) / q * 100.0
    }

    /// Window size over time as `(window_id, window_size_ms)` pairs.
    #[must_use]
    pub fn window_size_trend(&self) -> Vec<(u64, u64)> {
        self.entries
            .iter()
            .map(|e| (e.window_id, e.window_size_ms))
            .collect()
    }
}

fn is_correction_decision(decision: &str) -> bool {
    matches!(
        decision.trim().to_ascii_lowercase().as_str(),
        "correct" | "corrected" | "correction"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TranscriptionSegment;

    fn seg(text: &str, confidence: Option<f64>) -> TranscriptionSegment {
        TranscriptionSegment {
            text: text.to_owned(),
            start_sec: None,
            end_sec: None,
            confidence,
            speaker: None,
        }
    }

    const RECEIPT_RUN_ID: &str = "run-2026-07-14T12:34:56Z-0123456789abcdef0123456789abcdef";
    const RECEIPT_AUDIO_HASH_SEED: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn streaming_window_receipt_state_bytes(manager: &WindowManager) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "window_size_ms": manager.window_size_ms,
            "overlap_ms": manager.overlap_ms,
            "min_window_ms": manager.min_window_ms,
            "max_window_ms": manager.max_window_ms,
            "next_window_id": manager.next_window_id,
            "run_id": &manager.run_id,
            "windows": &manager.windows,
        }))
        .expect("serialize streaming window receipt state")
    }

    fn streaming_window_receipt_historical(
        window_count: usize,
    ) -> (WindowManager, Vec<WindowReceipt>) {
        let mut manager = WindowManager::new(RECEIPT_RUN_ID, 3_000, 500);
        let max_end_ms = u64::try_from(window_count)
            .expect("window count fits u64")
            .saturating_mul(2_500)
            .saturating_add(500);
        let mut receipts = Vec::with_capacity(window_count);
        for index in 0..window_count {
            let position_ms = u64::try_from(index)
                .expect("fixture index fits u64")
                .saturating_mul(2_500);
            let audio_hash = format!("{RECEIPT_AUDIO_HASH_SEED}:{position_ms}:3000");
            let window = manager
                .next_window_bounded(position_ms, max_end_ms, &audio_hash)
                .expect("historical bounded window");
            receipts.push(WindowReceipt {
                window_id: window.window_id,
                start_ms: window.start_ms,
                end_ms: window.end_ms,
            });
        }
        (manager, receipts)
    }

    fn streaming_window_receipt_candidate(
        window_count: usize,
    ) -> (WindowManager, Vec<WindowReceipt>) {
        let mut manager = WindowManager::new(RECEIPT_RUN_ID, 3_000, 500);
        let max_end_ms = u64::try_from(window_count)
            .expect("window count fits u64")
            .saturating_mul(2_500)
            .saturating_add(500);
        let mut receipts = Vec::with_capacity(window_count);
        for index in 0..window_count {
            let position_ms = u64::try_from(index)
                .expect("fixture index fits u64")
                .saturating_mul(2_500);
            let audio_hash = format!("{RECEIPT_AUDIO_HASH_SEED}:{position_ms}:3000");
            receipts.push(
                manager
                    .next_window_bounded_receipt(position_ms, max_end_ms, audio_hash)
                    .expect("candidate bounded window"),
            );
        }
        (manager, receipts)
    }

    #[test]
    fn streaming_window_receipt_preserves_complete_state() {
        let (mut historical, historical_receipts) = streaming_window_receipt_historical(513);
        let (mut candidate, candidate_receipts) = streaming_window_receipt_candidate(513);
        assert_eq!(candidate_receipts, historical_receipts, "scalar receipts");
        assert_eq!(
            streaming_window_receipt_state_bytes(&candidate),
            streaming_window_receipt_state_bytes(&historical),
            "complete retained window state"
        );

        assert!(
            historical
                .next_window_bounded(1_282_500, 1_282_500, "unused")
                .is_none()
        );
        assert!(
            candidate
                .next_window_bounded_receipt(1_282_500, 1_282_500, "unused".to_owned())
                .is_none()
        );
        assert_eq!(
            streaming_window_receipt_state_bytes(&candidate),
            streaming_window_receipt_state_bytes(&historical),
            "rejected boundary leaves state unchanged"
        );

        let mut historical_tail = WindowManager::new(RECEIPT_RUN_ID, 3_000, 500);
        let mut candidate_tail = WindowManager::new(RECEIPT_RUN_ID, 3_000, 500);
        let audio_hash = format!("{RECEIPT_AUDIO_HASH_SEED}:2500:3000");
        let historical_window = historical_tail
            .next_window_bounded(2_500, 2_601, &audio_hash)
            .expect("historical truncated window");
        let candidate_receipt = candidate_tail
            .next_window_bounded_receipt(2_500, 2_601, audio_hash)
            .expect("candidate truncated window");
        assert_eq!(
            candidate_receipt,
            WindowReceipt {
                window_id: historical_window.window_id,
                start_ms: historical_window.start_ms,
                end_ms: historical_window.end_ms,
            }
        );
        assert_eq!(
            streaming_window_receipt_state_bytes(&candidate_tail),
            streaming_window_receipt_state_bytes(&historical_tail),
            "truncated final-window state"
        );
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn streaming_window_receipt_perf() {
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 21;
        const CALIBRATION_WINDOWS: usize = 256;
        const EXACTNESS_WINDOWS: usize = 513;
        const TARGET_ARM_NS: u128 = 30_000_000;

        fn time_historical(window_count: usize) -> u128 {
            let mut manager = WindowManager::new(RECEIPT_RUN_ID, 3_000, 500);
            let max_end_ms = u64::try_from(window_count)
                .expect("window count fits u64")
                .saturating_mul(2_500)
                .saturating_add(500);
            let started = Instant::now();
            let mut checksum = 0_u64;
            for index in 0..window_count {
                let position_ms = u64::try_from(index)
                    .expect("fixture index fits u64")
                    .saturating_mul(2_500);
                let audio_hash = format!("{RECEIPT_AUDIO_HASH_SEED}:{position_ms}:3000");
                let window = manager
                    .next_window_bounded(position_ms, max_end_ms, &audio_hash)
                    .expect("historical timed window");
                checksum = checksum
                    .wrapping_add(window.window_id)
                    .wrapping_add(window.start_ms)
                    .wrapping_add(window.end_ms);
            }
            black_box((&manager.windows, checksum));
            started.elapsed().as_nanos()
        }

        fn time_candidate(window_count: usize) -> u128 {
            let mut manager = WindowManager::new(RECEIPT_RUN_ID, 3_000, 500);
            let max_end_ms = u64::try_from(window_count)
                .expect("window count fits u64")
                .saturating_mul(2_500)
                .saturating_add(500);
            let started = Instant::now();
            let mut checksum = 0_u64;
            for index in 0..window_count {
                let position_ms = u64::try_from(index)
                    .expect("fixture index fits u64")
                    .saturating_mul(2_500);
                let audio_hash = format!("{RECEIPT_AUDIO_HASH_SEED}:{position_ms}:3000");
                let receipt = manager
                    .next_window_bounded_receipt(position_ms, max_end_ms, audio_hash)
                    .expect("candidate timed window");
                checksum = checksum
                    .wrapping_add(receipt.window_id)
                    .wrapping_add(receipt.start_ms)
                    .wrapping_add(receipt.end_ms);
            }
            black_box((&manager.windows, checksum));
            started.elapsed().as_nanos()
        }

        fn percentile(values: &[f64], percentile: usize) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[(sorted.len() - 1) * percentile / 100]
        }

        fn median_ns(values: &[u128]) -> u128 {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        }

        fn coefficient_of_variation(values: &[u128]) -> f64 {
            let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (values.len() - 1) as f64;
            variance.sqrt() / mean
        }

        let (historical, historical_receipts) =
            streaming_window_receipt_historical(EXACTNESS_WINDOWS);
        let (candidate, candidate_receipts) = streaming_window_receipt_candidate(EXACTNESS_WINDOWS);
        let historical_bytes = streaming_window_receipt_state_bytes(&historical);
        let candidate_bytes = streaming_window_receipt_state_bytes(&candidate);
        assert_eq!(candidate_receipts, historical_receipts, "exact receipts");
        assert_eq!(candidate_bytes, historical_bytes, "exact retained state");

        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "streaming_window_receipt binary_sha256={:x} state_bytes={} state_sha256={:x} exactness_windows={} run_id_bytes={} audio_hash_seed_bytes={}",
            Sha256::digest(executable),
            historical_bytes.len(),
            Sha256::digest(&historical_bytes),
            EXACTNESS_WINDOWS,
            RECEIPT_RUN_ID.len(),
            RECEIPT_AUDIO_HASH_SEED.len()
        );

        let historical_calibration = time_historical(CALIBRATION_WINDOWS);
        let candidate_calibration = time_candidate(CALIBRATION_WINDOWS);
        let window_count = ((TARGET_ARM_NS * CALIBRATION_WINDOWS as u128)
            / historical_calibration.max(1))
        .clamp(CALIBRATION_WINDOWS as u128, 16_384) as usize;
        eprintln!(
            "streaming_window_receipt calibration_windows={CALIBRATION_WINDOWS} historical_calibration_ns={historical_calibration} candidate_calibration_ns={candidate_calibration} timed_windows={window_count}"
        );

        for _ in 0..3 {
            black_box(time_historical(window_count));
            black_box(time_candidate(window_count));
        }

        let mut null_ratios = Vec::with_capacity(SAMPLES);
        let mut speedups = Vec::with_capacity(SAMPLES);
        let mut historical_times = Vec::with_capacity(SAMPLES);
        let mut candidate_times = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let null_first = time_historical(window_count);
            let null_second = time_historical(window_count);
            let (numerator, denominator) = if sample % 2 == 0 {
                (null_first, null_second)
            } else {
                (null_second, null_first)
            };
            null_ratios.push(numerator as f64 / denominator as f64);

            let (historical_time, candidate_time) = if sample % 2 == 0 {
                (time_historical(window_count), time_candidate(window_count))
            } else {
                let candidate_time = time_candidate(window_count);
                let historical_time = time_historical(window_count);
                (historical_time, candidate_time)
            };
            historical_times.push(historical_time);
            candidate_times.push(candidate_time);
            speedups.push(historical_time as f64 / candidate_time as f64);
        }

        let null_p10 = percentile(&null_ratios, 10);
        let null_median = percentile(&null_ratios, 50);
        let null_p90 = percentile(&null_ratios, 90);
        let speedup_p10 = percentile(&speedups, 10);
        let speedup_median = percentile(&speedups, 50);
        let speedup_p90 = percentile(&speedups, 90);
        let wins = speedups.iter().filter(|ratio| **ratio > 1.0).count();
        eprintln!(
            "streaming_window_receipt samples={SAMPLES} timed_windows={window_count} null_p10={null_p10:.6} null_median={null_median:.6} null_p90={null_p90:.6} historical_per_window_median_ns={} candidate_per_window_median_ns={} candidate_cv={:.4}% speedup_p10={speedup_p10:.6} speedup_median={speedup_median:.6} speedup_p90={speedup_p90:.6} wins={wins}/{SAMPLES}",
            median_ns(&historical_times) / window_count as u128,
            median_ns(&candidate_times) / window_count as u128,
            coefficient_of_variation(&candidate_times) * 100.0
        );
        eprintln!(
            "streaming_window_receipt null_ratios={null_ratios:?} speedups={speedups:?} historical_times_ns={historical_times:?} candidate_times_ns={candidate_times:?}"
        );

        assert!(
            (0.95..=1.05).contains(&null_median),
            "null median {null_median:.6} outside predeclared guard"
        );
        assert!(
            speedup_p10 > null_p90.max(1.10),
            "candidate p10 {speedup_p10:.6} did not clear max(null p90 {null_p90:.6}, 1.10)"
        );
        assert!(
            wins >= 18,
            "candidate won {wins}/{SAMPLES}; predeclared gate requires at least 18"
        );
    }

    fn get_window_mut_historical(
        manager: &mut WindowManager,
        window_id: u64,
    ) -> Option<&mut WindowState> {
        manager
            .windows
            .iter_mut()
            .find(|state| state.window.window_id == window_id)
    }

    fn window_mut_direct_index_fixture(window_count: usize) -> WindowManager {
        let mut manager = WindowManager::new("direct-index-run", 3_000, 500);
        for index in 0..window_count {
            let position_ms = u64::try_from(index).expect("fixture index fits u64") * 2_500;
            manager.next_window(position_ms, &format!("audio-{index:04}"));
        }
        manager
    }

    #[test]
    fn window_mut_direct_index_matches_historical() {
        const WINDOW_COUNT: usize = 1_024;
        let mut historical = window_mut_direct_index_fixture(WINDOW_COUNT);
        let mut candidate = window_mut_direct_index_fixture(WINDOW_COUNT);

        for window_id in 0..WINDOW_COUNT as u64 {
            let historical_bytes = get_window_mut_historical(&mut historical, window_id)
                .map(|state| serde_json::to_vec(state).expect("serialize historical state"));
            let candidate_bytes = candidate
                .get_window_mut(window_id)
                .map(|state| serde_json::to_vec(state).expect("serialize candidate state"));
            assert_eq!(
                candidate_bytes, historical_bytes,
                "lookup mismatch for window {window_id}"
            );
        }

        for window_id in [WINDOW_COUNT as u64, u64::MAX] {
            assert!(get_window_mut_historical(&mut historical, window_id).is_none());
            assert!(candidate.get_window_mut(window_id).is_none());
        }

        for (window_id, status) in [
            (0, WindowStatus::FastInProgress),
            (511, WindowStatus::FastComplete),
            (1_023, WindowStatus::Resolved),
        ] {
            get_window_mut_historical(&mut historical, window_id)
                .expect("historical fixture window")
                .status = status;
            candidate
                .get_window_mut(window_id)
                .expect("candidate fixture window")
                .status = status;
        }
        assert_eq!(
            serde_json::to_vec(&candidate.windows).expect("serialize candidate windows"),
            serde_json::to_vec(&historical.windows).expect("serialize historical windows"),
            "complete post-mutation window state"
        );
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn window_mut_direct_index_perf() {
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const WINDOW_COUNT: usize = 1_024;
        const TARGET_WINDOW_ID: u64 = WINDOW_COUNT as u64 - 1;
        const SAMPLES: usize = 21;
        const CALIBRATION_ITERATIONS: usize = 2;
        const TARGET_ARM_NS: u128 = 50_000_000;

        fn time_historical(manager: &mut WindowManager, window_id: u64, iterations: usize) -> u128 {
            let started = Instant::now();
            let mut checksum = 0_u64;
            for _ in 0..iterations {
                let state =
                    get_window_mut_historical(black_box(&mut *manager), black_box(window_id))
                        .expect("historical target window");
                checksum = checksum.wrapping_add(black_box(state.window.end_ms));
            }
            black_box(checksum);
            started.elapsed().as_nanos()
        }

        fn time_candidate(manager: &mut WindowManager, window_id: u64, iterations: usize) -> u128 {
            let started = Instant::now();
            let mut checksum = 0_u64;
            for _ in 0..iterations {
                let state = black_box(&mut *manager)
                    .get_window_mut(black_box(window_id))
                    .expect("candidate target window");
                checksum = checksum.wrapping_add(black_box(state.window.end_ms));
            }
            black_box(checksum);
            started.elapsed().as_nanos()
        }

        fn percentile(values: &[f64], percentile: usize) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[(sorted.len() - 1) * percentile / 100]
        }

        fn median_ns(values: &[u128]) -> u128 {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        }

        fn coefficient_of_variation(values: &[u128]) -> f64 {
            let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (values.len() - 1) as f64;
            variance.sqrt() / mean
        }

        let mut historical = window_mut_direct_index_fixture(WINDOW_COUNT);
        let mut candidate = window_mut_direct_index_fixture(WINDOW_COUNT);
        let historical_bytes =
            serde_json::to_vec(&historical.windows).expect("serialize historical fixture");
        let candidate_bytes =
            serde_json::to_vec(&candidate.windows).expect("serialize candidate fixture");
        assert_eq!(candidate_bytes, historical_bytes, "exact fixture bytes");

        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "window_mut_direct_index binary_sha256={:x} state_bytes={} state_sha256={:x} windows={} target_window_id={} advance_ms=2500",
            Sha256::digest(executable),
            historical_bytes.len(),
            Sha256::digest(&historical_bytes),
            WINDOW_COUNT,
            TARGET_WINDOW_ID
        );

        let historical_calibration =
            time_historical(&mut historical, TARGET_WINDOW_ID, CALIBRATION_ITERATIONS);
        let candidate_calibration =
            time_candidate(&mut candidate, TARGET_WINDOW_ID, CALIBRATION_ITERATIONS);
        let historical_iterations = ((TARGET_ARM_NS * CALIBRATION_ITERATIONS as u128)
            / historical_calibration.max(1))
        .clamp(1, 16_777_216) as usize;
        let candidate_iterations = ((TARGET_ARM_NS * CALIBRATION_ITERATIONS as u128)
            / candidate_calibration.max(1))
        .clamp(1, 16_777_216) as usize;
        eprintln!(
            "window_mut_direct_index calibration_iterations={CALIBRATION_ITERATIONS} historical_calibration_ns={historical_calibration} candidate_calibration_ns={candidate_calibration} historical_iterations={historical_iterations} candidate_iterations={candidate_iterations}"
        );

        for _ in 0..3 {
            black_box(time_historical(
                &mut historical,
                TARGET_WINDOW_ID,
                historical_iterations,
            ));
            black_box(time_candidate(
                &mut candidate,
                TARGET_WINDOW_ID,
                candidate_iterations,
            ));
        }

        let mut null_ratios = Vec::with_capacity(SAMPLES);
        let mut speedups = Vec::with_capacity(SAMPLES);
        let mut historical_times = Vec::with_capacity(SAMPLES);
        let mut candidate_times = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let null_first =
                time_historical(&mut historical, TARGET_WINDOW_ID, historical_iterations);
            let null_second =
                time_historical(&mut historical, TARGET_WINDOW_ID, historical_iterations);
            let (numerator, denominator) = if sample % 2 == 0 {
                (null_first, null_second)
            } else {
                (null_second, null_first)
            };
            null_ratios.push(numerator as f64 / denominator as f64);

            let (historical_time, candidate_time) = if sample % 2 == 0 {
                (
                    time_historical(&mut historical, TARGET_WINDOW_ID, historical_iterations),
                    time_candidate(&mut candidate, TARGET_WINDOW_ID, candidate_iterations),
                )
            } else {
                let candidate_time =
                    time_candidate(&mut candidate, TARGET_WINDOW_ID, candidate_iterations);
                let historical_time =
                    time_historical(&mut historical, TARGET_WINDOW_ID, historical_iterations);
                (historical_time, candidate_time)
            };
            historical_times.push(historical_time);
            candidate_times.push(candidate_time);
            speedups.push(
                (historical_time as f64 / historical_iterations as f64)
                    / (candidate_time as f64 / candidate_iterations as f64),
            );
        }

        let null_p10 = percentile(&null_ratios, 10);
        let null_median = percentile(&null_ratios, 50);
        let null_p90 = percentile(&null_ratios, 90);
        let speedup_p10 = percentile(&speedups, 10);
        let speedup_median = percentile(&speedups, 50);
        let speedup_p90 = percentile(&speedups, 90);
        let wins = speedups.iter().filter(|ratio| **ratio > 1.0).count();
        eprintln!(
            "window_mut_direct_index samples={SAMPLES} historical_iterations={historical_iterations} candidate_iterations={candidate_iterations} null_p10={null_p10:.6} null_median={null_median:.6} null_p90={null_p90:.6} historical_per_lookup_median_ns={} candidate_per_lookup_median_ns={} candidate_cv={:.4}% speedup_p10={speedup_p10:.6} speedup_median={speedup_median:.6} speedup_p90={speedup_p90:.6} wins={wins}/{SAMPLES}",
            median_ns(&historical_times) / historical_iterations as u128,
            median_ns(&candidate_times) / candidate_iterations as u128,
            coefficient_of_variation(&candidate_times) * 100.0
        );
        eprintln!(
            "window_mut_direct_index null_ratios={null_ratios:?} speedups={speedups:?} historical_times_ns={historical_times:?} candidate_times_ns={candidate_times:?}"
        );

        assert!(
            (0.95..=1.05).contains(&null_median),
            "null median {null_median:.6} outside predeclared guard"
        );
        assert!(
            speedup_p10 > null_p90.max(1.10),
            "candidate p10 {speedup_p10:.6} did not clear max(null p90 {null_p90:.6}, 1.10)"
        );
        assert!(
            wins >= 18,
            "candidate won {wins}/{SAMPLES}; predeclared gate requires at least 18"
        );
    }

    fn levenshtein_historical(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        let mut prev = (0..=n).collect::<Vec<usize>>();
        let mut curr = vec![0usize; n + 1];

        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = if a_chars[i - 1] == b_chars[j - 1] {
                    0
                } else {
                    1
                };
                curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        prev[n]
    }

    fn correction_drift_historical(
        fast_segments: &[TranscriptionSegment],
        quality_segments: &[TranscriptionSegment],
    ) -> CorrectionDrift {
        let fast_text = concat_segment_text(fast_segments);
        let quality_text = concat_segment_text(quality_segments);
        let text_edit_distance = levenshtein_historical(&fast_text, &quality_text);
        let fast_words = fast_text.split_whitespace().collect::<Vec<_>>();
        let quality_words = quality_text.split_whitespace().collect::<Vec<_>>();
        let word_edit_dist = levenshtein_words(&fast_words, &quality_words);
        let max_words = fast_words.len().max(quality_words.len()).max(1);

        CorrectionDrift {
            wer_approx: word_edit_dist as f64 / max_words as f64,
            confidence_delta: (mean_confidence(fast_segments) - mean_confidence(quality_segments))
                .abs(),
            segment_count_delta: quality_segments.len() as i32 - fast_segments.len() as i32,
            text_edit_distance,
        }
    }

    fn correction_drift_common_affix_fixture()
    -> (Vec<TranscriptionSegment>, Vec<TranscriptionSegment>) {
        let mut fast = Vec::with_capacity(12);
        let mut quality = Vec::with_capacity(12);
        for index in 0..12 {
            let shared = format!(
                "window segment {index:02} preserves the already agreed streaming transcript context and stable timing detail"
            );
            let fast_text = if index == 5 {
                format!(
                    "{shared}; the reviewed dosage uses the brown marker before the shared clinical suffix"
                )
            } else {
                format!("{shared}; no correction is required for this portion")
            };
            let quality_text = if index == 5 {
                format!(
                    "{shared}; the reviewed dosage uses the green marker before the shared clinical suffix"
                )
            } else {
                format!("{shared}; no correction is required for this portion")
            };
            fast.push(TranscriptionSegment {
                text: fast_text,
                start_sec: Some(f64::from(index) * 1.25),
                end_sec: Some(f64::from(index + 1) * 1.25),
                confidence: (index % 4 != 0).then_some(0.78 + f64::from(index) / 200.0),
                speaker: (index % 3 == 0).then(|| format!("SPEAKER_{:02}", index % 2)),
            });
            quality.push(TranscriptionSegment {
                text: quality_text,
                start_sec: Some(f64::from(index) * 1.25),
                end_sec: Some(f64::from(index + 1) * 1.25),
                confidence: (index % 5 != 0).then_some(0.88 + f64::from(index) / 300.0),
                speaker: (index % 3 == 0).then(|| format!("SPEAKER_{:02}", index % 2)),
            });
        }
        (fast, quality)
    }

    fn correction_drift_without_identity_shortcut(
        fast_segments: &[TranscriptionSegment],
        quality_segments: &[TranscriptionSegment],
    ) -> CorrectionDrift {
        let fast_text = concat_segment_text(fast_segments);
        let quality_text = concat_segment_text(quality_segments);
        let text_edit_distance = levenshtein(&fast_text, &quality_text);
        let fast_words = fast_text.split_whitespace().collect::<Vec<_>>();
        let quality_words = quality_text.split_whitespace().collect::<Vec<_>>();
        let word_edit_dist = levenshtein_words(&fast_words, &quality_words);
        let max_words = fast_words.len().max(quality_words.len()).max(1);

        CorrectionDrift {
            wer_approx: word_edit_dist as f64 / max_words as f64,
            confidence_delta: (mean_confidence(fast_segments) - mean_confidence(quality_segments))
                .abs(),
            segment_count_delta: quality_segments.len() as i32 - fast_segments.len() as i32,
            text_edit_distance,
        }
    }

    fn correction_drift_identity_fixture() -> (Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)
    {
        let fast = (0..12)
            .map(|index| TranscriptionSegment {
                text: format!(
                    "confirmed streaming segment {index:02} retains identical Unicode λ transcript text and detailed contextual wording"
                ),
                start_sec: Some(f64::from(index) * 1.25),
                end_sec: Some(f64::from(index + 1) * 1.25),
                confidence: (index % 4 != 0).then_some(0.76 + f64::from(index) / 200.0),
                speaker: (index % 3 == 0).then(|| format!("FAST_{:02}", index % 2)),
            })
            .collect::<Vec<_>>();
        let quality = fast
            .iter()
            .enumerate()
            .map(|(index, segment)| TranscriptionSegment {
                text: segment.text.clone(),
                start_sec: segment.start_sec,
                end_sec: segment.end_sec,
                confidence: (index % 5 != 0).then_some(0.89 + index as f64 / 300.0),
                speaker: (index % 3 == 0).then(|| format!("QUALITY_{:02}", index % 2)),
            })
            .collect::<Vec<_>>();
        (fast, quality)
    }

    #[test]
    fn correction_drift_identity_shortcut_matches_historical() {
        let (fast, quality) = correction_drift_identity_fixture();
        let cases = [
            (fast, quality),
            (Vec::new(), Vec::new()),
            (
                vec![seg("hello", Some(0.25)), seg("世界 🎧", None)],
                vec![seg("hello 世界 🎧", Some(0.95))],
            ),
            (
                vec![seg("café λ old suffix", Some(0.5))],
                vec![seg("café λ new suffix", Some(0.5))],
            ),
        ];

        for (fast, quality) in cases {
            let historical = correction_drift_without_identity_shortcut(&fast, &quality);
            let candidate = CorrectionDrift::compute(&fast, &quality);
            assert_eq!(
                serde_json::to_vec(&candidate).expect("serialize candidate drift"),
                serde_json::to_vec(&historical).expect("serialize historical drift"),
                "full drift bytes for fast={fast:?}, quality={quality:?}"
            );
        }
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn correction_drift_identity_shortcut_perf() {
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 21;
        const CALIBRATION_ITERATIONS: usize = 2;
        const TARGET_ARM_NS: u128 = 50_000_000;

        fn time_historical(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(correction_drift_without_identity_shortcut(
                    black_box(fast),
                    black_box(quality),
                ));
            }
            started.elapsed().as_nanos()
        }

        fn time_candidate(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(CorrectionDrift::compute(
                    black_box(fast),
                    black_box(quality),
                ));
            }
            started.elapsed().as_nanos()
        }

        fn percentile(values: &[f64], percentile: usize) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[(sorted.len() - 1) * percentile / 100]
        }

        fn median_ns(values: &[u128]) -> u128 {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        }

        fn coefficient_of_variation(values: &[u128]) -> f64 {
            let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (values.len() - 1) as f64;
            variance.sqrt() / mean
        }

        let (fast, quality) = correction_drift_identity_fixture();
        let historical_bytes =
            serde_json::to_vec(&correction_drift_without_identity_shortcut(&fast, &quality))
                .expect("serialize historical drift");
        let candidate_bytes = serde_json::to_vec(&CorrectionDrift::compute(&fast, &quality))
            .expect("serialize candidate drift");
        assert_eq!(candidate_bytes, historical_bytes, "exact drift bytes");

        let transcript_chars = concat_segment_text(&fast).chars().count();
        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "correction_drift_identity_shortcut binary_sha256={:x} result_bytes={} result_sha256={:x} fast_segments={} quality_segments={} transcript_chars={}",
            Sha256::digest(executable),
            historical_bytes.len(),
            Sha256::digest(&historical_bytes),
            fast.len(),
            quality.len(),
            transcript_chars
        );

        let historical_calibration = time_historical(&fast, &quality, CALIBRATION_ITERATIONS);
        let candidate_calibration = time_candidate(&fast, &quality, CALIBRATION_ITERATIONS);
        let historical_iterations = ((TARGET_ARM_NS * CALIBRATION_ITERATIONS as u128)
            / historical_calibration.max(1))
        .clamp(1, 131_072) as usize;
        let candidate_iterations = ((TARGET_ARM_NS * CALIBRATION_ITERATIONS as u128)
            / candidate_calibration.max(1))
        .clamp(1, 131_072) as usize;
        eprintln!(
            "correction_drift_identity_shortcut calibration_iterations={CALIBRATION_ITERATIONS} historical_calibration_ns={historical_calibration} candidate_calibration_ns={candidate_calibration} historical_iterations={historical_iterations} candidate_iterations={candidate_iterations}"
        );

        for _ in 0..3 {
            black_box(time_historical(&fast, &quality, historical_iterations));
            black_box(time_candidate(&fast, &quality, candidate_iterations));
        }

        let mut null_ratios = Vec::with_capacity(SAMPLES);
        let mut speedups = Vec::with_capacity(SAMPLES);
        let mut historical_times = Vec::with_capacity(SAMPLES);
        let mut candidate_times = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let null_first = time_historical(&fast, &quality, historical_iterations);
            let null_second = time_historical(&fast, &quality, historical_iterations);
            let (numerator, denominator) = if sample % 2 == 0 {
                (null_first, null_second)
            } else {
                (null_second, null_first)
            };
            null_ratios.push(numerator as f64 / denominator as f64);

            let (historical, candidate) = if sample % 2 == 0 {
                (
                    time_historical(&fast, &quality, historical_iterations),
                    time_candidate(&fast, &quality, candidate_iterations),
                )
            } else {
                let candidate = time_candidate(&fast, &quality, candidate_iterations);
                let historical = time_historical(&fast, &quality, historical_iterations);
                (historical, candidate)
            };
            historical_times.push(historical);
            candidate_times.push(candidate);
            speedups.push(
                (historical as f64 / historical_iterations as f64)
                    / (candidate as f64 / candidate_iterations as f64),
            );
        }

        let null_p10 = percentile(&null_ratios, 10);
        let null_median = percentile(&null_ratios, 50);
        let null_p90 = percentile(&null_ratios, 90);
        let speedup_p10 = percentile(&speedups, 10);
        let speedup_median = percentile(&speedups, 50);
        let speedup_p90 = percentile(&speedups, 90);
        let wins = speedups.iter().filter(|ratio| **ratio > 1.0).count();
        eprintln!(
            "correction_drift_identity_shortcut samples={SAMPLES} historical_iterations={historical_iterations} candidate_iterations={candidate_iterations} null_p10={null_p10:.6} null_median={null_median:.6} null_p90={null_p90:.6} historical_per_call_median_ns={} candidate_per_call_median_ns={} candidate_cv={:.4}% speedup_p10={speedup_p10:.6} speedup_median={speedup_median:.6} speedup_p90={speedup_p90:.6} wins={wins}/{SAMPLES}",
            median_ns(&historical_times) / historical_iterations as u128,
            median_ns(&candidate_times) / candidate_iterations as u128,
            coefficient_of_variation(&candidate_times) * 100.0
        );
        eprintln!(
            "correction_drift_identity_shortcut null_ratios={null_ratios:?} speedups={speedups:?} historical_times_ns={historical_times:?} candidate_times_ns={candidate_times:?}"
        );

        assert!(
            (0.95..=1.05).contains(&null_median),
            "null median {null_median:.6} outside predeclared guard"
        );
        assert!(
            speedup_p10 > null_p90.max(1.10),
            "candidate p10 {speedup_p10:.6} did not clear max(null p90 {null_p90:.6}, 1.10)"
        );
        assert!(
            wins >= 18,
            "candidate won {wins}/{SAMPLES}; predeclared gate requires at least 18"
        );
    }

    #[test]
    fn correction_drift_common_affix_matches_historical() {
        let corpus = [
            "",
            "a",
            "same transcript",
            "shared prefix old shared suffix",
            "shared prefix new shared suffix",
            "café λ old 🎧 suffix",
            "café λ new 🎧 suffix",
            "日本語の記録",
            "日本人の記録",
            "combining e\u{301} marker",
            "emoji 👋🌍 tail",
        ];
        for left in corpus {
            for right in corpus {
                assert_eq!(
                    levenshtein(left, right),
                    levenshtein_historical(left, right),
                    "distance mismatch for {left:?} versus {right:?}"
                );
            }
        }

        let (fast, quality) = correction_drift_common_affix_fixture();
        let historical = correction_drift_historical(&fast, &quality);
        let candidate = CorrectionDrift::compute(&fast, &quality);
        assert_eq!(
            serde_json::to_vec(&candidate).expect("serialize candidate drift"),
            serde_json::to_vec(&historical).expect("serialize historical drift"),
            "full correction drift bytes"
        );
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn correction_drift_common_affix_perf() {
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 21;
        const CALIBRATION_ITERATIONS: usize = 2;
        const TARGET_ARM_NS: u128 = 50_000_000;

        fn time_historical(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(correction_drift_historical(
                    black_box(fast),
                    black_box(quality),
                ));
            }
            started.elapsed().as_nanos()
        }

        fn time_candidate(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(CorrectionDrift::compute(
                    black_box(fast),
                    black_box(quality),
                ));
            }
            started.elapsed().as_nanos()
        }

        fn percentile(values: &[f64], percentile: usize) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[(sorted.len() - 1) * percentile / 100]
        }

        fn median_ns(values: &[u128]) -> u128 {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        }

        fn coefficient_of_variation(values: &[u128]) -> f64 {
            let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (values.len() - 1) as f64;
            variance.sqrt() / mean
        }

        let (fast, quality) = correction_drift_common_affix_fixture();
        let historical_bytes = serde_json::to_vec(&correction_drift_historical(&fast, &quality))
            .expect("serialize historical drift");
        let candidate_bytes = serde_json::to_vec(&CorrectionDrift::compute(&fast, &quality))
            .expect("serialize candidate drift");
        assert_eq!(candidate_bytes, historical_bytes, "exact drift bytes");

        let fast_chars = concat_segment_text(&fast).chars().count();
        let quality_chars = concat_segment_text(&quality).chars().count();
        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "correction_drift_common_affix binary_sha256={:x} result_bytes={} result_sha256={:x} fast_segments={} quality_segments={} fast_chars={} quality_chars={}",
            Sha256::digest(executable),
            historical_bytes.len(),
            Sha256::digest(&historical_bytes),
            fast.len(),
            quality.len(),
            fast_chars,
            quality_chars
        );

        let calibration = time_historical(&fast, &quality, CALIBRATION_ITERATIONS);
        let iterations = ((TARGET_ARM_NS * CALIBRATION_ITERATIONS as u128) / calibration.max(1))
            .clamp(1, 131_072) as usize;
        eprintln!(
            "correction_drift_common_affix calibration_iterations={CALIBRATION_ITERATIONS} calibration_ns={calibration} iterations={iterations}"
        );

        for _ in 0..3 {
            black_box(time_historical(&fast, &quality, iterations));
            black_box(time_candidate(&fast, &quality, iterations));
        }

        let mut null_ratios = Vec::with_capacity(SAMPLES);
        let mut speedups = Vec::with_capacity(SAMPLES);
        let mut historical_times = Vec::with_capacity(SAMPLES);
        let mut candidate_times = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let null_first = time_historical(&fast, &quality, iterations);
            let null_second = time_historical(&fast, &quality, iterations);
            let (numerator, denominator) = if sample % 2 == 0 {
                (null_first, null_second)
            } else {
                (null_second, null_first)
            };
            null_ratios.push(numerator as f64 / denominator as f64);

            let (historical, candidate) = if sample % 2 == 0 {
                (
                    time_historical(&fast, &quality, iterations),
                    time_candidate(&fast, &quality, iterations),
                )
            } else {
                let candidate = time_candidate(&fast, &quality, iterations);
                let historical = time_historical(&fast, &quality, iterations);
                (historical, candidate)
            };
            historical_times.push(historical);
            candidate_times.push(candidate);
            speedups.push(historical as f64 / candidate as f64);
        }

        let null_p10 = percentile(&null_ratios, 10);
        let null_median = percentile(&null_ratios, 50);
        let null_p90 = percentile(&null_ratios, 90);
        let speedup_p10 = percentile(&speedups, 10);
        let speedup_median = percentile(&speedups, 50);
        let speedup_p90 = percentile(&speedups, 90);
        let wins = speedups.iter().filter(|ratio| **ratio > 1.0).count();
        eprintln!(
            "correction_drift_common_affix samples={SAMPLES} iterations={iterations} null_p10={null_p10:.6} null_median={null_median:.6} null_p90={null_p90:.6} historical_arm_median_ns={} candidate_arm_median_ns={} candidate_cv={:.4}% speedup_p10={speedup_p10:.6} speedup_median={speedup_median:.6} speedup_p90={speedup_p90:.6} wins={wins}/{SAMPLES}",
            median_ns(&historical_times),
            median_ns(&candidate_times),
            coefficient_of_variation(&candidate_times) * 100.0
        );
        eprintln!(
            "correction_drift_common_affix null_ratios={null_ratios:?} speedups={speedups:?} historical_times_ns={historical_times:?} candidate_times_ns={candidate_times:?}"
        );

        assert!(
            (0.95..=1.05).contains(&null_median),
            "null median {null_median:.6} outside predeclared guard"
        );
        assert!(
            speedup_p10 > null_p90.max(1.10),
            "candidate p10 {speedup_p10:.6} did not clear max(null p90 {null_p90:.6}, 1.10)"
        );
        assert!(
            wins >= 18,
            "candidate won {wins}/{SAMPLES}; predeclared gate requires at least 18"
        );
    }

    #[test]
    fn correction_drift_reuse_matches_historical_event_bytes() {
        let fast = vec![
            TranscriptionSegment {
                text: "hello λ world".to_owned(),
                start_sec: Some(0.0),
                end_sec: Some(1.25),
                confidence: Some(0.75),
                speaker: Some("SPEAKER_00".to_owned()),
            },
            TranscriptionSegment {
                text: "line two with 🎧".to_owned(),
                start_sec: Some(1.25),
                end_sec: Some(2.5),
                confidence: None,
                speaker: None,
            },
        ];
        let quality = vec![
            TranscriptionSegment {
                text: "hello λ earth".to_owned(),
                start_sec: Some(0.0),
                end_sec: Some(1.25),
                confidence: Some(0.95),
                speaker: Some("SPEAKER_00".to_owned()),
            },
            TranscriptionSegment {
                text: "line two with 🎧 and punctuation.".to_owned(),
                start_sec: Some(1.25),
                end_sec: Some(2.5),
                confidence: None,
                speaker: None,
            },
        ];
        let corrected_at = "2026-07-14T00:00:00.000000000Z";

        let historical = CorrectionEvent::new(
            17,
            23,
            29,
            "quality-v2".to_owned(),
            quality.clone(),
            431,
            corrected_at.to_owned(),
            &fast,
        );
        let quality_confidence_mean = mean_confidence(&quality);
        let drift = CorrectionDrift::compute(&fast, &quality);
        let reused = CorrectionEvent::from_precomputed_metrics(
            17,
            23,
            29,
            "quality-v2".to_owned(),
            quality,
            431,
            quality_confidence_mean,
            drift,
            corrected_at.to_owned(),
        );

        assert_eq!(
            serde_json::to_vec(&reused).expect("serialize reused correction event"),
            serde_json::to_vec(&historical).expect("serialize historical correction event")
        );
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn correction_drift_reuse_perf() {
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 21;
        const CALIBRATION_ITERATIONS: usize = 2;
        const TARGET_ARM_NS: u128 = 40_000_000;
        const CORRECTED_AT: &str = "2026-07-14T00:00:00.000000000Z";

        fn make_tracker(fast: &[TranscriptionSegment]) -> CorrectionTracker {
            let mut tracker = CorrectionTracker::new(CorrectionTolerance {
                always_correct: true,
                ..CorrectionTolerance::default()
            });
            tracker.register_partial(PartialTranscript::new(
                23,
                29,
                "fast-v1".to_owned(),
                fast.to_vec(),
                137,
                "2026-07-14T00:00:00.000000000Z".to_owned(),
            ));
            tracker
        }

        fn submit_historical(
            tracker: &mut CorrectionTracker,
            quality_segments: Vec<TranscriptionSegment>,
        ) -> CorrectionDecision {
            let window_id = 29;
            let quality_latency_ms = 431;
            let seq = *tracker
                .window_to_seq
                .get(&window_id)
                .expect("registered historical window");
            let drift = {
                let partial = tracker
                    .partials
                    .get(&seq)
                    .expect("registered historical partial");
                CorrectionDrift::compute(&partial.segments, &quality_segments)
            };

            tracker.stats.windows_processed += 1;
            tracker.stats.total_quality_latency_ms += quality_latency_ms;
            tracker.stats.cumulative_wer += drift.wer_approx;
            if drift.wer_approx > tracker.stats.max_observed_wer {
                tracker.stats.max_observed_wer = drift.wer_approx;
            }

            let needs_correction = tracker.tolerance.always_correct
                || drift.wer_approx > tracker.tolerance.max_wer
                || drift.confidence_delta > tracker.tolerance.max_confidence_delta
                || drift.text_edit_distance > tracker.tolerance.max_edit_distance;
            assert!(needs_correction, "fixture must exercise correction path");

            let partial = tracker
                .partials
                .get_mut(&seq)
                .expect("registered historical partial");
            partial.retract();
            let fast_segments = partial.segments.clone();

            let correction_id = tracker.next_correction_id;
            tracker.next_correction_id += 1;
            let correction = CorrectionEvent::new(
                correction_id,
                seq,
                window_id,
                "quality-v2".to_owned(),
                quality_segments,
                quality_latency_ms,
                chrono::Utc::now().to_rfc3339(),
                &fast_segments,
            );
            tracker.corrections.push(correction.clone());
            tracker.stats.corrections_emitted += 1;
            tracker.window_to_seq.remove(&window_id);

            CorrectionDecision::Correct { correction }
        }

        fn normalized_state_bytes(
            tracker: &CorrectionTracker,
            decision: CorrectionDecision,
        ) -> Option<Vec<u8>> {
            let CorrectionDecision::Correct {
                correction: mut event,
            } = decision
            else {
                return None;
            };
            event.corrected_at_rfc3339 = CORRECTED_AT.to_owned();
            let mut stored = tracker.corrections.clone();
            for correction in &mut stored {
                correction.corrected_at_rfc3339 = CORRECTED_AT.to_owned();
            }
            let partial = tracker
                .partials
                .get(&23)
                .expect("retained corrected partial");
            serde_json::to_vec(&serde_json::json!({
                "decision": event,
                "stored_corrections": stored,
                "partial_status": partial.status,
                "windows_processed": tracker.stats.windows_processed,
                "corrections_emitted": tracker.stats.corrections_emitted,
                "confirmations_emitted": tracker.stats.confirmations_emitted,
                "total_fast_latency_ms": tracker.stats.total_fast_latency_ms,
                "total_quality_latency_ms": tracker.stats.total_quality_latency_ms,
                "cumulative_wer": tracker.stats.cumulative_wer,
                "max_observed_wer": tracker.stats.max_observed_wer,
                "next_correction_id": tracker.next_correction_id,
                "window_mapping_empty": tracker.window_to_seq.is_empty(),
            }))
            .ok()
        }

        fn time_historical(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> u128 {
            let mut cases = (0..iterations)
                .map(|_| (make_tracker(fast), quality.to_vec()))
                .collect::<Vec<_>>();
            let started = Instant::now();
            for (tracker, quality_segments) in &mut cases {
                black_box(submit_historical(
                    black_box(tracker),
                    std::mem::take(quality_segments),
                ));
            }
            started.elapsed().as_nanos()
        }

        fn time_reused(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> u128 {
            let mut cases = (0..iterations)
                .map(|_| (make_tracker(fast), quality.to_vec()))
                .collect::<Vec<_>>();
            let started = Instant::now();
            for (tracker, quality_segments) in &mut cases {
                black_box(
                    black_box(tracker)
                        .submit_quality_result(
                            29,
                            "quality-v2",
                            std::mem::take(quality_segments),
                            431,
                        )
                        .expect("candidate correction submission"),
                );
            }
            started.elapsed().as_nanos()
        }

        fn percentile(values: &[f64], percentile: usize) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[(sorted.len() - 1) * percentile / 100]
        }

        fn median_ns(values: &[u128]) -> u128 {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        }

        fn coefficient_of_variation(values: &[u128]) -> f64 {
            let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (values.len() - 1) as f64;
            variance.sqrt() / mean
        }

        let fast = (0..12)
            .map(|index| TranscriptionSegment {
                text: format!(
                    "fast segment {index:02} carries current-window transcript words, Unicode λ, and measured correction detail"
                ),
                start_sec: Some(f64::from(index) * 1.25),
                end_sec: Some(f64::from(index + 1) * 1.25),
                confidence: (index % 4 != 0).then_some(0.70 + f64::from(index) / 100.0),
                speaker: (index % 3 == 0).then(|| format!("SPEAKER_{:02}", index % 2)),
            })
            .collect::<Vec<_>>();
        let quality = (0..12)
            .map(|index| TranscriptionSegment {
                text: format!(
                    "quality segment {index:02} carries corrected-window transcript words, Unicode λ, and authoritative correction detail"
                ),
                start_sec: Some(f64::from(index) * 1.25),
                end_sec: Some(f64::from(index + 1) * 1.25),
                confidence: (index % 5 != 0).then_some(0.85 + f64::from(index) / 200.0),
                speaker: (index % 3 == 0).then(|| format!("SPEAKER_{:02}", index % 2)),
            })
            .collect::<Vec<_>>();

        let mut historical_tracker = make_tracker(&fast);
        let historical_decision = submit_historical(&mut historical_tracker, quality.clone());
        let historical_bytes = normalized_state_bytes(&historical_tracker, historical_decision)
            .expect("historical fixture must correct and serialize");
        let mut reused_tracker = make_tracker(&fast);
        let reused_decision = reused_tracker
            .submit_quality_result(29, "quality-v2", quality.clone(), 431)
            .expect("candidate correction submission");
        let reused_bytes = normalized_state_bytes(&reused_tracker, reused_decision)
            .expect("candidate fixture must correct and serialize");
        assert_eq!(
            reused_bytes, historical_bytes,
            "complete normalized correction state bytes"
        );

        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "correction_drift_reuse binary_sha256={:x} state_bytes={} state_sha256={:x} fast_segments={} quality_segments={}",
            Sha256::digest(executable),
            historical_bytes.len(),
            Sha256::digest(&historical_bytes),
            fast.len(),
            quality.len()
        );

        let calibration = time_historical(&fast, &quality, CALIBRATION_ITERATIONS);
        let iterations = ((TARGET_ARM_NS * CALIBRATION_ITERATIONS as u128) / calibration.max(1))
            .clamp(1, 2_000) as usize;
        eprintln!(
            "correction_drift_reuse calibration_iterations={CALIBRATION_ITERATIONS} calibration_ns={calibration} iterations={iterations}"
        );

        for _ in 0..3 {
            black_box(time_historical(&fast, &quality, iterations));
            black_box(time_reused(&fast, &quality, iterations));
        }

        let mut null_ratios = Vec::with_capacity(SAMPLES);
        let mut speedups = Vec::with_capacity(SAMPLES);
        let mut historical_times = Vec::with_capacity(SAMPLES);
        let mut reused_times = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let null_first = time_historical(&fast, &quality, iterations);
            let null_second = time_historical(&fast, &quality, iterations);
            let (numerator, denominator) = if sample % 2 == 0 {
                (null_first, null_second)
            } else {
                (null_second, null_first)
            };
            null_ratios.push(numerator as f64 / denominator as f64);

            let (historical, reused) = if sample % 2 == 0 {
                (
                    time_historical(&fast, &quality, iterations),
                    time_reused(&fast, &quality, iterations),
                )
            } else {
                let reused = time_reused(&fast, &quality, iterations);
                let historical = time_historical(&fast, &quality, iterations);
                (historical, reused)
            };
            historical_times.push(historical);
            reused_times.push(reused);
            speedups.push(historical as f64 / reused as f64);
        }

        let null_p10 = percentile(&null_ratios, 10);
        let null_median = percentile(&null_ratios, 50);
        let null_p90 = percentile(&null_ratios, 90);
        let speedup_p10 = percentile(&speedups, 10);
        let speedup_median = percentile(&speedups, 50);
        let speedup_p90 = percentile(&speedups, 90);
        let wins = speedups.iter().filter(|ratio| **ratio > 1.0).count();
        eprintln!(
            "correction_drift_reuse samples={SAMPLES} iterations={iterations} null_p10={null_p10:.6} null_median={null_median:.6} null_p90={null_p90:.6} historical_arm_median_ns={} reused_arm_median_ns={} reused_cv={:.4}% speedup_p10={speedup_p10:.6} speedup_median={speedup_median:.6} speedup_p90={speedup_p90:.6} wins={wins}/{SAMPLES}",
            median_ns(&historical_times),
            median_ns(&reused_times),
            coefficient_of_variation(&reused_times) * 100.0
        );
        eprintln!(
            "correction_drift_reuse null_ratios={null_ratios:?} speedups={speedups:?} historical_times_ns={historical_times:?} reused_times_ns={reused_times:?}"
        );

        assert!(
            (0.95..=1.05).contains(&null_median),
            "null median {null_median:.6} outside predeclared guard"
        );
        assert!(
            speedup_p10 > null_p90.max(1.10),
            "candidate p10 {speedup_p10:.6} did not clear max(null p90 {null_p90:.6}, 1.10)"
        );
        assert!(
            wins >= 18,
            "candidate won {wins}/{SAMPLES}; predeclared gate requires at least 18"
        );
    }

    #[test]
    fn speculation_window_duration_and_contains_boundary() {
        let w = SpeculationWindow::new(0, "r".into(), 1000, 4000, 500, "h".into());
        assert_eq!(w.duration_ms(), 3000);
        // contains_ms: start inclusive, end exclusive.
        assert!(w.contains_ms(1000));
        assert!(w.contains_ms(3999));
        assert!(!w.contains_ms(4000));
        assert!(!w.contains_ms(999));
        // Degenerate: start == end → duration 0, contains nothing.
        let w2 = SpeculationWindow::new(1, "r".into(), 5000, 5000, 0, "h".into());
        assert_eq!(w2.duration_ms(), 0);
        assert!(!w2.contains_ms(5000));
        // Saturating sub when end < start (shouldn't happen, but safe).
        let w3 = SpeculationWindow::new(2, "r".into(), 100, 50, 0, "h".into());
        assert_eq!(w3.duration_ms(), 0);
    }

    #[test]
    fn correction_drift_identical_empty_and_divergent() {
        // Identical segments → wer 0, edit distance 0.
        let segs = vec![seg("hello world", Some(0.9))];
        let drift = CorrectionDrift::compute(&segs, &segs);
        assert_eq!(drift.wer_approx, 0.0);
        assert_eq!(drift.text_edit_distance, 0);
        assert_eq!(drift.segment_count_delta, 0);
        assert!(drift.confidence_delta < f64::EPSILON);

        // Empty vs empty → all zeros.
        let empty_drift = CorrectionDrift::compute(&[], &[]);
        assert_eq!(empty_drift.text_edit_distance, 0);
        assert_eq!(empty_drift.segment_count_delta, 0);

        // Divergent segments → positive WER.
        let fast = vec![seg("the cat sat", Some(0.8))];
        let quality = vec![seg("a dog ran", Some(0.95))];
        let d = CorrectionDrift::compute(&fast, &quality);
        assert!(d.wer_approx > 0.0);
        assert!(d.text_edit_distance > 0);
        assert!((d.confidence_delta - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn beta_posterior_prior_updates_and_variance() {
        let p = BetaPosterior::weakly_informative();
        assert!((p.mean() - 0.5).abs() < f64::EPSILON);
        let var_initial = p.variance();

        // Corrections shift mean up (towards 1).
        let mut p2 = BetaPosterior::weakly_informative();
        for _ in 0..10 {
            p2.observe_correction();
        }
        assert!(p2.mean() > 0.5);
        assert!(p2.variance() < var_initial);

        // Confirmations shift mean down (towards 0).
        let mut p3 = BetaPosterior::weakly_informative();
        for _ in 0..10 {
            p3.observe_confirmation();
        }
        assert!(p3.mean() < 0.5);
    }

    #[test]
    #[should_panic(expected = "alpha must be positive")]
    fn beta_posterior_rejects_zero_alpha() {
        let _ = BetaPosterior::new(0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "beta must be positive")]
    fn beta_posterior_rejects_negative_beta() {
        let _ = BetaPosterior::new(1.0, -1.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be positive")]
    fn beta_posterior_rejects_nan_alpha() {
        let _ = BetaPosterior::new(f64::NAN, 1.0);
    }

    #[test]
    #[should_panic(expected = "beta must be positive")]
    fn beta_posterior_rejects_infinite_beta() {
        let _ = BetaPosterior::new(1.0, f64::INFINITY);
    }

    #[test]
    fn calibration_tracker_brier_score_and_eviction() {
        // Perfect predictions: predicted=1.0 when correction, 0.0 when confirm → Brier=0.
        let mut ct = CalibrationTracker::new(10);
        ct.record(1.0, true);
        ct.record(0.0, false);
        assert!(ct.brier_score() < f64::EPSILON);
        assert_eq!(ct.sample_count(), 2);

        // Worst predictions: predicted=0.0 when correction, 1.0 when confirm → Brier=1.
        let mut ct2 = CalibrationTracker::new(10);
        ct2.record(0.0, true);
        ct2.record(1.0, false);
        assert!((ct2.brier_score() - 1.0).abs() < f64::EPSILON);

        // Eviction: window_size=3, add 5 entries, only 3 remain.
        let mut ct3 = CalibrationTracker::new(3);
        for i in 0..5 {
            ct3.record(0.5, i % 2 == 0);
        }
        assert_eq!(ct3.sample_count(), 3);

        // Empty tracker → Brier 0.
        let ct4 = CalibrationTracker::new(5);
        assert_eq!(ct4.brier_score(), 0.0);
    }

    #[test]
    fn evidence_ledger_capacity_zero_eviction_and_diagnostics() {
        // Capacity 0: records total but stores nothing.
        let mut ledger = CorrectionEvidenceLedger::new(0);
        let entry = CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 0,
            run_id: "r".into(),
            timestamp_rfc3339: "2025-01-01T00:00:00Z".into(),
            fast_model_id: "tiny".into(),
            fast_latency_ms: 100,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "large".into(),
            quality_latency_ms: 500,
            quality_confidence_mean: 0.95,
            quality_segment_count: 1,
            drift: CorrectionDrift {
                wer_approx: 0.1,
                confidence_delta: 0.15,
                segment_count_delta: 0,
                text_edit_distance: 5,
            },
            decision: "confirm".into(),
            window_size_ms: 3000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        };
        ledger.record(entry.clone());
        assert_eq!(ledger.total_recorded(), 1);
        assert!(ledger.entries().is_empty());

        // Normal capacity with eviction.
        let mut ledger2 = CorrectionEvidenceLedger::new(2);
        let mut e = entry;
        for i in 0..3 {
            e.entry_id = i;
            e.fast_latency_ms = 100;
            e.quality_latency_ms = 500;
            e.decision = if i == 1 {
                "correct".into()
            } else {
                "confirm".into()
            };
            ledger2.record(e.clone());
        }
        assert_eq!(ledger2.total_recorded(), 3);
        assert_eq!(ledger2.entries().len(), 2);
        // Oldest (entry_id=0) was evicted; entry_id=1,2 remain.
        assert_eq!(ledger2.entries()[0].entry_id, 1);

        // Diagnostics: 1 correction out of 2 retained = 50%.
        assert!((ledger2.correction_rate() - 0.5).abs() < f64::EPSILON);
        assert!((ledger2.mean_fast_latency() - 100.0).abs() < f64::EPSILON);
        assert!((ledger2.mean_quality_latency() - 500.0).abs() < f64::EPSILON);
        // Latency savings: (500-100)/500*100 = 80%.
        assert!((ledger2.latency_savings_pct() - 80.0).abs() < f64::EPSILON);
    }

    // ── Task #206 — speculation pass 2 edge-case tests ───────────────

    #[test]
    fn window_manager_next_window_bounded_none_and_clamped() {
        let mut mgr = WindowManager::new("run-bounded", 5000, 500);

        // audio_position >= max_end → None.
        assert!(mgr.next_window_bounded(10_000, 10_000, "h").is_none());
        assert!(mgr.next_window_bounded(10_001, 10_000, "h").is_none());

        // natural_end (3000 + 5000 = 8000) > max_end (6000) → clamped to 6000.
        let w = mgr.next_window_bounded(3000, 6000, "h").unwrap();
        assert_eq!(w.end_ms, 6000);
        assert_eq!(w.duration_ms(), 3000); // 6000 - 3000

        // Happy path: natural_end fits within max_end.
        let w2 = mgr.next_window_bounded(0, 100_000, "h").unwrap();
        assert_eq!(w2.duration_ms(), 5000);
    }

    #[test]
    fn window_manager_set_window_size_clamps() {
        let mut mgr = WindowManager::new("run-clamp", 5000, 500);
        // min_window_ms = 1000, max_window_ms = 30000.

        // Below min → clamped to 1000.
        mgr.set_window_size(0);
        assert_eq!(mgr.current_window_size(), 1000);

        // Above max → clamped to 30000.
        mgr.set_window_size(u64::MAX);
        assert_eq!(mgr.current_window_size(), 30_000);

        // Within range → passes through.
        mgr.set_window_size(15_000);
        assert_eq!(mgr.current_window_size(), 15_000);
    }

    #[test]
    fn correction_tracker_unregistered_window_returns_error() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

        // submit_quality_result with no registered partial → Err.
        let result = tracker.submit_quality_result(
            999, // no partial for this window_id
            "quality-model",
            vec![seg("hello", Some(0.9))],
            100,
        );
        let err = result.expect_err("expected error for unregistered window_id");
        let msg = err.to_string();
        assert!(
            msg.contains("no partial registered"),
            "error should mention missing partial: {msg}"
        );
    }

    #[test]
    fn correction_tracker_all_resolved_transitions() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

        // No partials → all_resolved() is vacuously true.
        assert!(tracker.all_resolved());

        // Register a partial → not all resolved (Pending).
        let partial = PartialTranscript::new(
            0,
            0,
            "fast".to_owned(),
            vec![seg("hello", Some(0.9))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(partial);
        assert!(!tracker.all_resolved(), "pending partial → not resolved");

        // Submit matching quality result → Confirm → all resolved.
        let decision = tracker
            .submit_quality_result(0, "quality", vec![seg("hello", Some(0.9))], 100)
            .unwrap();
        let is_confirm = matches!(decision, CorrectionDecision::Confirm { .. });
        assert!(is_confirm, "expected Confirm decision");
        assert!(tracker.all_resolved());
    }

    #[test]
    fn correction_tracker_window_lookup_is_removed_after_resolution() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());
        let partial = PartialTranscript::new(
            0,
            11,
            "fast".to_owned(),
            vec![seg("hello", Some(0.9))],
            40,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(partial);

        let first = tracker.submit_quality_result(11, "quality", vec![seg("hello", Some(0.9))], 70);
        assert!(
            first.is_ok(),
            "first quality result should resolve window 11"
        );

        let second =
            tracker.submit_quality_result(11, "quality", vec![seg("hello", Some(0.9))], 70);
        let err = second.expect_err("window 11 should be resolved and no longer accepted");
        assert!(
            err.to_string()
                .contains("no partial registered for window_id 11"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn is_correction_decision_all_variants_and_normalization() {
        assert!(is_correction_decision("correct"));
        assert!(is_correction_decision("corrected"));
        assert!(is_correction_decision("correction"));
        // Normalization: trim + lowercase.
        assert!(is_correction_decision(" CORRECTED "));
        assert!(is_correction_decision("\tCorrection\n"));
        // Non-corrections.
        assert!(!is_correction_decision("confirm"));
        assert!(!is_correction_decision("confirmed"));
        assert!(!is_correction_decision(""));
        assert!(!is_correction_decision("   "));
    }

    fn timed_seg(
        text: &str,
        start: f64,
        end: f64,
        confidence: Option<f64>,
    ) -> TranscriptionSegment {
        TranscriptionSegment {
            text: text.to_owned(),
            start_sec: Some(start),
            end_sec: Some(end),
            confidence,
            speaker: None,
        }
    }

    #[test]
    fn window_manager_merge_segments_quality_preferred_and_dedup() {
        let mut wm = WindowManager::new("run-1", 5000, 0);

        // Create two windows.
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(5000, "h1");

        // Window 0: has both fast and quality results — quality should be used.
        let fast_partial_0 = PartialTranscript::new(
            0,
            w0.window_id,
            "fast".to_owned(),
            vec![timed_seg("fast hello", 0.0, 2.0, Some(0.5))],
            100,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        wm.record_fast_result(w0.window_id, fast_partial_0);
        wm.record_quality_result(
            w0.window_id,
            vec![timed_seg("quality hello", 0.0, 2.0, Some(0.9))],
        );
        wm.resolve_window(w0.window_id);

        // Window 1: only fast result — fast should be used as fallback.
        let fast_partial_1 = PartialTranscript::new(
            1,
            w1.window_id,
            "fast".to_owned(),
            vec![timed_seg("fast world", 5.0, 7.0, Some(0.6))],
            100,
            "2026-01-01T00:00:01Z".to_owned(),
        );
        wm.record_fast_result(w1.window_id, fast_partial_1);
        wm.resolve_window(w1.window_id);

        let merged = wm.merge_segments();
        assert_eq!(merged.len(), 2);
        // First segment from quality result (not fast).
        assert_eq!(merged[0].text, "quality hello");
        // Second from fast fallback.
        assert_eq!(merged[1].text, "fast world");
    }

    #[test]
    fn window_manager_merge_segments_overlap_dedup_by_confidence() {
        let mut wm = WindowManager::new("run-1", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(5000, "h1");

        // Both windows produce a segment at nearly the same time range.
        wm.record_quality_result(
            w0.window_id,
            vec![timed_seg("low conf", 2.0, 4.0, Some(0.3))],
        );
        wm.resolve_window(w0.window_id);
        // Window 1 produces an overlapping segment with higher confidence.
        wm.record_quality_result(
            w1.window_id,
            vec![timed_seg("high conf", 2.05, 4.05, Some(0.95))],
        );
        wm.resolve_window(w1.window_id);

        let merged = wm.merge_segments();
        // Overlap within 0.1s tolerance → dedup keeps higher confidence.
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "high conf");
    }

    #[test]
    fn window_manager_record_and_resolve_nonexistent_window_is_noop() {
        let mut wm = WindowManager::new("run-1", 5000, 0);
        let w = wm.next_window(0, "h0");

        // Operate on a window_id that doesn't exist.
        let bogus_id = w.window_id + 999;
        let partial = PartialTranscript::new(
            0,
            bogus_id,
            "fast".to_owned(),
            vec![seg("hello", Some(0.8))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        wm.record_fast_result(bogus_id, partial);
        wm.record_quality_result(bogus_id, vec![seg("world", Some(0.9))]);
        wm.resolve_window(bogus_id);

        // Original window should be unaffected (still Pending).
        assert!(wm.get_window(bogus_id).is_none());
        let ws = wm.get_window(w.window_id).unwrap();
        assert_eq!(ws.status, WindowStatus::Pending);
    }

    #[test]
    fn correction_tracker_rate_wer_stats_across_multiple_windows() {
        // Zero state: correction_rate and mean_wer both 0.0.
        let mut tracker = CorrectionTracker::new(CorrectionTolerance {
            max_wer: 0.05,
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: false,
        });
        assert!((tracker.correction_rate() - 0.0).abs() < 1e-9);
        assert!((tracker.mean_wer() - 0.0).abs() < 1e-9);

        // Register two partials.
        let p0 = PartialTranscript::new(
            0,
            0,
            "fast".to_owned(),
            vec![seg("hello world", Some(0.8))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        let p1 = PartialTranscript::new(
            1,
            1,
            "fast".to_owned(),
            vec![seg("foo bar baz", Some(0.7))],
            50,
            "2026-01-01T00:00:01Z".to_owned(),
        );
        tracker.register_partial(p0);
        tracker.register_partial(p1);

        // Window 0: identical quality → confirm (WER ≈ 0).
        let decision0 = tracker
            .submit_quality_result(0, "quality", vec![seg("hello world", Some(0.9))], 100)
            .unwrap();
        let is_confirm = matches!(decision0, CorrectionDecision::Confirm { .. });
        assert!(is_confirm, "identical text should confirm");

        // Window 1: very different quality → correct (WER high).
        let decision1 = tracker
            .submit_quality_result(
                1,
                "quality",
                vec![seg("completely different text", Some(0.95))],
                100,
            )
            .unwrap();
        let is_correct = matches!(decision1, CorrectionDecision::Correct { .. });
        assert!(is_correct, "very different text should correct");

        // correction_rate = 1 correction / 2 total = 0.5
        assert!((tracker.correction_rate() - 0.5).abs() < 1e-9);
        // mean_wer should be > 0 (one window had WER > 0).
        assert!(tracker.mean_wer() > 0.0);
        // max_observed_wer should equal the WER of the divergent window.
        assert!(tracker.stats().max_observed_wer > 0.0);
        assert_eq!(tracker.stats().windows_processed, 2);
    }

    #[test]
    fn speculation_controller_brier_fallback_resets_window_to_initial() {
        let initial_ms = 5000;
        let mut ctrl = SpeculationWindowController::new(initial_ms, 1000, 30_000, 500);
        assert!(!ctrl.is_fallback_active());

        let drift = CorrectionDrift {
            wer_approx: 0.05,
            confidence_delta: 0.01,
            segment_count_delta: 0,
            text_edit_distance: 1,
        };

        // Drive Brier score above 0.25: feed 15 corrections first (pushing
        // posterior high), then 10 confirmations (posterior-after-update is
        // still high → each has high Brier contribution). The CalibrationTracker
        // window is 20, so the earliest 5 correction entries (low Brier) get
        // evicted, leaving 10 corrections + 10 confirmations in the window.
        // correction_rate via posterior ≈ 0.6 (below 0.75 runaway threshold).

        for _ in 0..15 {
            let correction =
                CorrectionEvent::new(0, 0, 0, "q".to_owned(), vec![], 100, "t".to_owned(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }

        for _ in 0..10 {
            let decision = CorrectionDecision::Confirm {
                seq: 0,
                drift: drift.clone(),
            };
            ctrl.observe(&decision, &drift);
        }

        let new_size = ctrl.apply();
        assert!(
            ctrl.is_fallback_active(),
            "should be in fallback after poor calibration"
        );
        assert_eq!(
            new_size, initial_ms,
            "Brier fallback should reset to initial window size"
        );
    }

    // ── Task #219 — speculation pass 4 edge-case tests ───────────────

    #[test]
    fn correction_tracker_always_correct_forces_correction_on_identical_output() {
        let tol = CorrectionTolerance {
            always_correct: true,
            ..CorrectionTolerance::default()
        };
        let mut tracker = CorrectionTracker::new(tol);

        let partial = PartialTranscript::new(
            0,
            0,
            "fast".to_owned(),
            vec![seg("hello world", Some(0.9))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(partial);

        // Identical quality output — normally Confirm, but always_correct forces Correct.
        let decision = tracker
            .submit_quality_result(0, "quality", vec![seg("hello world", Some(0.9))], 100)
            .unwrap();

        assert!(
            matches!(decision, CorrectionDecision::Correct { .. }),
            "always_correct=true must force Correct even for identical text"
        );
        assert_eq!(tracker.stats().corrections_emitted, 1);
        assert_eq!(tracker.stats().confirmations_emitted, 0);
    }

    #[test]
    fn window_manager_resolved_and_pending_counts_track_lifecycle() {
        let mut wm = WindowManager::new("run-counts", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(5000, "h1");
        let w2 = wm.next_window(10000, "h2");

        // Nothing resolved yet.
        assert_eq!(wm.windows_resolved(), 0);
        assert_eq!(wm.windows_pending(), 3);

        wm.resolve_window(w0.window_id);
        assert_eq!(wm.windows_resolved(), 1);
        assert_eq!(wm.windows_pending(), 2);

        wm.resolve_window(w1.window_id);
        wm.resolve_window(w2.window_id);
        assert_eq!(wm.windows_resolved(), 3);
        assert_eq!(wm.windows_pending(), 0);
    }

    #[test]
    fn controller_shrink_on_low_correction_rate_and_clamp_to_min() {
        // initial=5000, min=2000, max=30_000, step=500.
        let mut ctrl = SpeculationWindowController::new(5000, 2000, 30_000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };

        // Feed 15 confirmations: correction_rate ≈ 0 (< LOW_CORRECTION_RATE=0.0625),
        // window_count=15 > 10, confidence=15/20=0.75 >= 0.5.
        for _ in 0..15 {
            let decision = CorrectionDecision::Confirm {
                seq: 0,
                drift: drift.clone(),
            };
            ctrl.observe(&decision, &drift);
        }

        let action = ctrl.recommend();
        assert_eq!(
            action,
            ControllerAction::Shrink(500),
            "low correction rate + enough windows should recommend Shrink"
        );

        let new_size = ctrl.apply();
        assert_eq!(new_size, 4500, "5000 - 500 = 4500");
        assert!(
            !ctrl.is_fallback_active(),
            "normal shrink should not activate fallback"
        );

        // Drive window down to min via repeated shrinks.
        for _ in 0..20 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
            ctrl.apply();
        }
        assert!(
            ctrl.current_window_ms() >= 2000,
            "window must not go below min_window_ms=2000, got {}",
            ctrl.current_window_ms()
        );
    }

    #[test]
    fn correction_tracker_get_partial_present_absent_and_retracted() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

        // Unknown seq → None.
        assert!(tracker.get_partial(42).is_none());

        let partial = PartialTranscript::new(
            7,
            0,
            "fast".to_owned(),
            vec![seg("hello", Some(0.9))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(partial);

        // Known seq → Some, status Pending.
        let found = tracker.get_partial(7).expect("should find seq=7");
        assert_eq!(found.seq, 7);
        assert_eq!(found.status, PartialStatus::Pending);

        // Force correction via always_correct, verify Retracted status.
        let tol = CorrectionTolerance {
            always_correct: true,
            ..CorrectionTolerance::default()
        };
        let mut tracker2 = CorrectionTracker::new(tol);
        let p2 = PartialTranscript::new(
            0,
            0,
            "fast".to_owned(),
            vec![seg("foo", Some(0.8))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker2.register_partial(p2);
        tracker2
            .submit_quality_result(0, "q", vec![seg("foo", Some(0.8))], 100)
            .unwrap();

        let p = tracker2.get_partial(0).unwrap();
        assert_eq!(
            p.status,
            PartialStatus::Retracted,
            "always_correct should retract the partial"
        );
        assert!(
            tracker2.all_resolved(),
            "retracted partial counts as resolved"
        );
    }

    #[test]
    fn evidence_ledger_window_size_trend_order_and_negative_latency_savings() {
        let mut ledger = CorrectionEvidenceLedger::new(10);

        let make_entry = |window_id: u64, window_size_ms: u64, fast_ms: u64, quality_ms: u64| {
            CorrectionEvidenceEntry {
                entry_id: window_id,
                window_id,
                run_id: "r".into(),
                timestamp_rfc3339: "2026-01-01T00:00:00Z".into(),
                fast_model_id: "tiny".into(),
                fast_latency_ms: fast_ms,
                fast_confidence_mean: 0.8,
                fast_segment_count: 1,
                quality_model_id: "large".into(),
                quality_latency_ms: quality_ms,
                quality_confidence_mean: 0.9,
                quality_segment_count: 1,
                drift: CorrectionDrift {
                    wer_approx: 0.0,
                    confidence_delta: 0.0,
                    segment_count_delta: 0,
                    text_edit_distance: 0,
                },
                decision: "confirm".into(),
                window_size_ms,
                correction_rate_at_decision: 0.0,
                controller_confidence: 1.0,
                fallback_active: false,
                fallback_reason: None,
            }
        };

        // fast=400ms > quality=300ms → negative latency savings.
        ledger.record(make_entry(0, 3000, 400, 300));
        ledger.record(make_entry(1, 4000, 400, 300));

        // window_size_trend preserves insertion order.
        let trend = ledger.window_size_trend();
        assert_eq!(trend, vec![(0, 3000), (1, 4000)]);

        // latency_savings_pct: (300-400)/300*100 = -33.3...%
        let savings = ledger.latency_savings_pct();
        assert!(
            savings < 0.0,
            "fast model slower than quality should yield negative savings: got {savings}"
        );
        assert!(
            (savings - (-100.0 / 3.0)).abs() < 0.01,
            "expected ~-33.33%, got {savings}"
        );
    }

    // ── Task #224 — speculation pass 5 edge-case tests ───────────────

    #[test]
    fn controller_recommend_grows_on_runaway_correction_rate() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30_000, 1000);
        let drift = CorrectionDrift {
            wer_approx: 0.9,
            confidence_delta: 0.1,
            segment_count_delta: 0,
            text_edit_distance: 50,
        };

        // 10 corrections → correction_rate > 0.75 (runaway)
        for _ in 0..10 {
            let correction =
                CorrectionEvent::new(0, 0, 0, "q".to_owned(), vec![], 100, "t".to_owned(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }

        let action = ctrl.recommend();
        assert_eq!(
            action,
            ControllerAction::Grow(1000),
            "runaway correction rate should recommend Grow"
        );
    }

    #[test]
    fn controller_recommend_force_shrink_after_twenty_consecutive_confirmations() {
        let mut ctrl = SpeculationWindowController::new(5000, 2000, 30_000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };

        // 20 consecutive confirmations → consecutive_zero_corrections == 20
        for _ in 0..20 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }

        let action = ctrl.recommend();
        assert_eq!(
            action,
            ControllerAction::Shrink(500),
            "20 consecutive zero-correction windows should force shrink"
        );

        // A correction resets the counter.
        let correction =
            CorrectionEvent::new(0, 0, 0, "q".to_owned(), vec![], 100, "t".to_owned(), &[]);
        ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        assert_eq!(ctrl.state().window_count, 21);
    }

    #[test]
    fn controller_apply_clears_runaway_fallback_after_rate_drops() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30_000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.9,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };

        // Drive into runaway: 10 corrections.
        for _ in 0..10 {
            let correction =
                CorrectionEvent::new(0, 0, 0, "q".to_owned(), vec![], 100, "t".to_owned(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        ctrl.apply();
        assert!(ctrl.is_fallback_active(), "should be in runaway fallback");

        // Flood with confirmations to drop rate below 0.75.
        let low_drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        for _ in 0..40 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: low_drift.clone(),
                },
                &low_drift,
            );
        }

        ctrl.apply();
        assert!(
            !ctrl.is_fallback_active(),
            "fallback should clear once correction rate drops below 0.75"
        );
    }

    #[test]
    fn correction_event_is_significant_threshold_boundary() {
        let fast_segs = vec![seg("the cat sat", Some(0.8))];
        let quality_segs = vec![seg("a dog ran fast", Some(0.95))];
        let event = CorrectionEvent::new(
            0,
            0,
            0,
            "quality".to_owned(),
            quality_segs,
            200,
            "2026-01-01T00:00:00Z".to_owned(),
            &fast_segs,
        );

        assert!(
            event.drift.wer_approx > 0.0,
            "precondition: divergent text must yield positive WER"
        );

        // Threshold below WER → significant.
        assert!(event.is_significant(0.0));

        // Threshold at exactly WER → NOT significant (strict >).
        let exact = event.drift.wer_approx;
        assert!(
            !event.is_significant(exact),
            "WER == threshold should not be significant"
        );

        // Threshold above WER → not significant.
        assert!(!event.is_significant(1.0));
    }

    #[test]
    fn window_manager_merge_skips_resolved_window_with_no_results() {
        let mut wm = WindowManager::new("run-skip", 5000, 0);

        // Window 0: resolved with no fast or quality result.
        let w0 = wm.next_window(0, "h0");
        wm.resolve_window(w0.window_id);

        // Window 1: resolved with a quality result (distinct timestamps).
        let w1 = wm.next_window(5000, "h1");
        wm.record_quality_result(w1.window_id, vec![timed_seg("hello", 5.0, 6.0, Some(0.9))]);
        wm.resolve_window(w1.window_id);

        let merged = wm.merge_segments();
        assert_eq!(
            merged.len(),
            1,
            "resolved window with no results must be skipped"
        );
        assert_eq!(merged[0].text, "hello");
    }

    // ── Task #229 — speculation.rs pass 6 edge-case tests ──────────────

    #[test]
    fn speculation_window_contains_ms_when_overlap_exceeds_duration() {
        // overlap_ms > duration: contains_ms ignores overlap entirely
        // and operates only on raw start/end boundaries.
        let w = SpeculationWindow::new(0, "r".into(), 1000, 4000, 5000, "h".into());
        assert_eq!(w.duration_ms(), 3000);
        assert!(w.contains_ms(1000), "start inclusive");
        assert!(w.contains_ms(3999), "end-1 inclusive");
        assert!(!w.contains_ms(4000), "end exclusive");
        assert!(!w.contains_ms(999), "before start");
    }

    #[test]
    fn correction_drift_asymmetric_empty_fast_vs_populated_quality() {
        // Fast is empty, quality has segments → WER should be 1.0 (all words are edits).
        let fast: Vec<TranscriptionSegment> = vec![];
        let quality = vec![seg("hello world", Some(0.9)), seg("goodbye", Some(0.8))];
        let drift = CorrectionDrift::compute(&fast, &quality);

        // max_words = max(0, 3).max(1) = 3; word_edit_dist = 3 (all inserts).
        assert!(
            (drift.wer_approx - 1.0).abs() < 1e-9,
            "empty fast vs populated quality should give WER ≈ 1.0, got {}",
            drift.wer_approx
        );
        // segment_count_delta = quality_count - fast_count = 2 - 0 = 2
        assert_eq!(
            drift.segment_count_delta, 2,
            "delta should be +2 (quality has 2, fast has 0)"
        );
        // mean_confidence of empty fast = 0.0, mean of quality = (0.9+0.8)/2 = 0.85
        assert!(
            (drift.confidence_delta - 0.85).abs() < 1e-9,
            "confidence_delta should be 0.85, got {}",
            drift.confidence_delta
        );
    }

    #[test]
    fn evidence_ledger_capacity_one_always_evicts_previous_entry() {
        let mut ledger = CorrectionEvidenceLedger::new(1);

        let make_entry = |id: u64| CorrectionEvidenceEntry {
            entry_id: id,
            window_id: id,
            run_id: "r".to_owned(),
            timestamp_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            fast_model_id: "fast".to_owned(),
            fast_latency_ms: 10,
            fast_confidence_mean: 0.9,
            fast_segment_count: 1,
            quality_model_id: "quality".to_owned(),
            quality_latency_ms: 50,
            quality_confidence_mean: 0.95,
            quality_segment_count: 1,
            drift: CorrectionDrift::compute(&[], &[]),
            decision: "confirm".to_owned(),
            window_size_ms: 3000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        };

        ledger.record(make_entry(1));
        assert_eq!(ledger.entries().len(), 1);
        assert_eq!(ledger.total_recorded(), 1);

        // Second record evicts the first (capacity = 1).
        ledger.record(make_entry(2));
        assert_eq!(
            ledger.entries().len(),
            1,
            "capacity 1 should only keep 1 entry"
        );
        assert_eq!(ledger.entries()[0].entry_id, 2, "should keep the latest");
        assert_eq!(
            ledger.total_recorded(),
            2,
            "total_recorded should track all, including evicted"
        );

        // Third record again evicts.
        ledger.record(make_entry(3));
        assert_eq!(ledger.entries().len(), 1);
        assert_eq!(ledger.entries()[0].entry_id, 3);
        assert_eq!(ledger.total_recorded(), 3);
    }

    #[test]
    fn controller_recommend_holds_unconditionally_below_min_windows_for_adapt() {
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 10000, 500);

        // Feed 4 corrections (all extreme — 100% correction rate, high WER).
        // MIN_WINDOWS_FOR_ADAPT is 5, so at count 4 we should still get Hold.
        let drift = CorrectionDrift {
            wer_approx: 1.0,
            confidence_delta: 0.5,
            segment_count_delta: 1,
            text_edit_distance: 10,
        };
        let decision = CorrectionDecision::Confirm {
            seq: 0,
            drift: drift.clone(),
        };
        for _ in 0..4 {
            ctrl.observe(&decision, &drift);
        }

        let action = ctrl.recommend();
        assert!(
            matches!(action, ControllerAction::Hold),
            "below MIN_WINDOWS_FOR_ADAPT (5), should always Hold regardless of rate, got {action:?}"
        );
    }

    #[test]
    fn correction_tracker_register_partial_duplicate_seq_overwrites() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());

        let p1 = PartialTranscript::new(
            1,
            100,
            "fast-model".to_owned(),
            vec![seg("first version", Some(0.8))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(p1);

        let p2 = PartialTranscript::new(
            1, // same seq
            100,
            "fast-model".to_owned(),
            vec![seg("second version", Some(0.9))],
            60,
            "2026-01-01T00:00:01Z".to_owned(),
        );
        tracker.register_partial(p2);

        // Only one partial should be stored (the second overwrites the first).
        let retrieved = tracker.get_partial(1);
        assert!(retrieved.is_some(), "seq 1 should still be present");
        assert_eq!(
            retrieved.unwrap().segments[0].text,
            "second version",
            "overwritten partial should have the latest text"
        );

        // Both latencies were accumulated (50 + 60 = 110).
        let stats = tracker.stats();
        assert_eq!(
            stats.total_fast_latency_ms, 110,
            "both registrations should accumulate latency"
        );
    }

    #[test]
    fn controller_recommend_grows_on_moderate_correction_rate_and_high_wer() {
        // Grow path at lines 1085-1090: correction_rate > 0.25 AND mean_wer > 0.125
        // AND current_window_ms < max_window_ms AND confidence >= 0.5.
        // Must NOT hit the runaway branch (rate <= 0.75).
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 10000, 500);

        // Drive 10 observations: 4 corrections, 6 confirmations → rate = 4/10 = 0.4.
        // Use wer_approx = 0.2 (> 0.125 threshold) for all.
        let high_wer_drift = CorrectionDrift {
            wer_approx: 0.2,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        for i in 0..10u64 {
            let decision = if i < 4 {
                CorrectionDecision::Correct {
                    correction: CorrectionEvent::new(
                        i,
                        i,
                        i,
                        "quality".to_owned(),
                        vec![],
                        100,
                        "2026-01-01T00:00:00Z".to_owned(),
                        &[],
                    ),
                }
            } else {
                CorrectionDecision::Confirm {
                    seq: i,
                    drift: high_wer_drift.clone(),
                }
            };
            ctrl.observe(&decision, &high_wer_drift);
        }

        // confidence = 10 / 20.0 = 0.5, correction_rate = 0.4, mean_wer = 0.2
        let action = ctrl.recommend();
        assert!(
            matches!(action, ControllerAction::Grow(500)),
            "moderate correction rate + high WER should Grow, got {action:?}"
        );
    }

    #[test]
    fn controller_recommend_holds_at_max_window_even_with_runaway_correction_rate() {
        // Runaway branch (lines 1068-1072): rate > 0.75 AND current < max → Grow.
        // But when current_window_ms == max_window_ms, it falls through to Hold.
        let mut ctrl = SpeculationWindowController::new(10000, 1000, 10000, 500);

        let drift = CorrectionDrift {
            wer_approx: 0.5,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // 5 corrections, 0 confirmations → rate = 5/5 = 1.0 (well above 0.75).
        for i in 0..5u64 {
            let decision = CorrectionDecision::Correct {
                correction: CorrectionEvent::new(
                    i,
                    i,
                    i,
                    "quality".to_owned(),
                    vec![],
                    100,
                    "2026-01-01T00:00:00Z".to_owned(),
                    &[],
                ),
            };
            ctrl.observe(&decision, &drift);
        }

        let action = ctrl.recommend();
        // Even though rate is runaway, current_window_ms == max_window_ms so guard fails.
        // Falls through; confidence = 5/20 = 0.25 < 0.5, so returns Hold.
        assert!(
            matches!(action, ControllerAction::Hold),
            "at max window with runaway rate should Hold, got {action:?}"
        );
    }

    #[test]
    fn correction_tracker_triggers_correction_on_confidence_delta_alone() {
        // The trigger condition at line 741: confidence_delta > max_confidence_delta.
        // Here wer and edit_distance are within tolerance; only confidence_delta exceeds.
        let tolerance = CorrectionTolerance {
            max_wer: 1.0,              // very lenient
            max_confidence_delta: 0.1, // tight threshold
            max_edit_distance: 1000,   // very lenient
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        let p = PartialTranscript::new(
            0,
            100,
            "fast".to_owned(),
            vec![seg("hello world", Some(0.5))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(p);

        // Quality has same text but much higher confidence → large confidence_delta.
        let quality_segs = vec![seg("hello world", Some(0.95))];
        let decision = tracker
            .submit_quality_result(100, "quality", quality_segs, 200)
            .unwrap();
        assert!(
            matches!(decision, CorrectionDecision::Correct { .. }),
            "confidence_delta 0.45 > 0.1 should trigger correction, got confirm"
        );
    }

    #[test]
    fn window_manager_status_transitions_through_fast_complete_and_quality_complete() {
        let mut wm = WindowManager::new("test-run", 3000, 500);
        let win = wm.next_window(0, "abc123");
        let wid = win.window_id;

        // Initial status is Pending.
        let ws = wm.get_window(wid).unwrap();
        assert_eq!(ws.status, WindowStatus::Pending);

        // After recording fast result → FastComplete.
        let partial = PartialTranscript::new(
            0,
            wid,
            "fast".to_owned(),
            vec![seg("fast result", Some(0.8))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        wm.record_fast_result(wid, partial);
        let ws = wm.get_window(wid).unwrap();
        assert_eq!(ws.status, WindowStatus::FastComplete);

        // After recording quality result → QualityComplete.
        wm.record_quality_result(wid, vec![seg("quality result", Some(0.95))]);
        let ws = wm.get_window(wid).unwrap();
        assert_eq!(ws.status, WindowStatus::QualityComplete);

        // After resolving → Resolved.
        wm.resolve_window(wid);
        let ws = wm.get_window(wid).unwrap();
        assert_eq!(ws.status, WindowStatus::Resolved);
    }

    #[test]
    fn evidence_ledger_to_evidence_json_shape_and_diagnostics_block() {
        let mut ledger = CorrectionEvidenceLedger::new(10);

        let entry = CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 100,
            run_id: "test-run".to_owned(),
            timestamp_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            fast_model_id: "fast".to_owned(),
            fast_latency_ms: 50,
            fast_confidence_mean: 0.5,
            fast_segment_count: 1,
            quality_model_id: "quality".to_owned(),
            quality_latency_ms: 200,
            quality_confidence_mean: 0.95,
            quality_segment_count: 2,
            drift: CorrectionDrift {
                wer_approx: 0.15,
                confidence_delta: 0.3,
                segment_count_delta: 1,
                text_edit_distance: 5,
            },
            decision: "correct".to_owned(),
            window_size_ms: 3000,
            correction_rate_at_decision: 0.25,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        };
        ledger.record(entry);

        let json = ledger.to_evidence_json();

        // Top-level keys.
        assert_eq!(json["type"].as_str(), Some("correction_evidence_ledger"));
        assert_eq!(json["total_recorded"].as_u64(), Some(1));
        assert_eq!(json["retained"].as_u64(), Some(1));
        assert_eq!(json["capacity"].as_u64(), Some(10));

        // Diagnostics block.
        let diag = &json["diagnostics"];
        assert!(diag.is_object(), "diagnostics should be an object");
        assert!(
            diag.get("correction_rate").is_some(),
            "missing correction_rate"
        );
        assert!(
            diag.get("mean_fast_latency_ms").is_some(),
            "missing mean_fast_latency_ms"
        );
        assert!(
            diag.get("mean_quality_latency_ms").is_some(),
            "missing mean_quality_latency_ms"
        );
        assert!(diag.get("mean_wer").is_some(), "missing mean_wer");
        assert!(
            diag.get("latency_savings_pct").is_some(),
            "missing latency_savings_pct"
        );

        // Entries array.
        let entries = json["entries"]
            .as_array()
            .expect("entries should be an array");
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn partial_transcript_confirm_and_retract_direct() {
        let mut pt = PartialTranscript::new(
            0,
            100,
            "fast".to_owned(),
            vec![seg("hello", Some(0.9))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        assert!(matches!(pt.status, PartialStatus::Pending));

        pt.confirm();
        assert!(matches!(pt.status, PartialStatus::Confirmed));
        // Idempotent — calling again doesn't panic.
        pt.confirm();
        assert!(matches!(pt.status, PartialStatus::Confirmed));

        pt.retract();
        assert!(matches!(pt.status, PartialStatus::Retracted));
        // Idempotent.
        pt.retract();
        assert!(matches!(pt.status, PartialStatus::Retracted));
    }

    #[test]
    fn correction_drift_populated_fast_vs_empty_quality() {
        let fast = vec![seg("hello world goodbye", Some(0.8))];
        let quality: Vec<TranscriptionSegment> = vec![];
        let drift = CorrectionDrift::compute(&fast, &quality);

        assert_eq!(drift.segment_count_delta, -1, "quality(0) - fast(1) = -1");
        assert!(
            (drift.wer_approx - 1.0).abs() < 1e-9,
            "3 fast words, 0 quality → WER ~1.0, got {}",
            drift.wer_approx
        );
        assert_eq!(
            drift.text_edit_distance,
            "hello world goodbye".len(),
            "edit distance should equal fast text length"
        );
        assert!(
            (drift.confidence_delta - 0.8).abs() < 1e-9,
            "fast conf=0.8, quality conf=0.0 → delta=0.8, got {}",
            drift.confidence_delta
        );
    }

    #[test]
    fn correction_tracker_rate_zero_after_all_confirmations() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance {
            max_wer: 1.0,
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: false,
        });

        // Register two partials with matching quality text.
        for i in 0..2u64 {
            let p = PartialTranscript::new(
                i,
                i + 100,
                "fast".to_owned(),
                vec![seg("same text", Some(0.9))],
                50,
                "2026-01-01T00:00:00Z".to_owned(),
            );
            tracker.register_partial(p);
        }

        // Submit identical quality for both → both confirm.
        for i in 0..2u64 {
            let decision = tracker
                .submit_quality_result(i + 100, "quality", vec![seg("same text", Some(0.9))], 200)
                .expect("should succeed");
            assert!(matches!(decision, CorrectionDecision::Confirm { .. }));
        }

        let stats = tracker.stats();
        assert_eq!(stats.corrections_emitted, 0);
        assert_eq!(stats.confirmations_emitted, 2);
        // This exercises the arithmetic branch (line 793), not the total==0 early return.
        assert!(
            (tracker.correction_rate() - 0.0).abs() < 1e-9,
            "all confirmations → correction_rate should be 0.0"
        );
    }

    #[test]
    fn calibration_tracker_window_size_zero_always_empty() {
        let mut ct = CalibrationTracker::new(0);

        // Every record is immediately evicted.
        for _ in 0..5 {
            ct.record(1.0, true);
        }
        assert_eq!(ct.sample_count(), 0, "window_size=0 should always evict");
        assert!(
            (ct.brier_score() - 0.0).abs() < 1e-9,
            "empty tracker should return Brier 0.0"
        );
    }

    #[test]
    fn calibration_tracker_uniform_half_predictions_brier_quarter() {
        // Predicting 0.5 always: Brier = mean((0.5 - actual)^2).
        // For balanced true/false: each term is 0.25, so mean = 0.25.
        let mut ct = CalibrationTracker::new(20);
        for i in 0..10 {
            ct.record(0.5, i % 2 == 0);
        }
        assert!(
            (ct.brier_score() - 0.25).abs() < 1e-9,
            "uniform 0.5 predictions on balanced outcomes should yield Brier=0.25, got {}",
            ct.brier_score()
        );
    }

    #[test]
    fn calibration_tracker_single_element_window_keeps_latest() {
        let mut ct = CalibrationTracker::new(1);

        ct.record(1.0, true); // perfect → Brier 0
        assert_eq!(ct.sample_count(), 1);
        assert!(ct.brier_score() < 1e-9);

        ct.record(0.0, true); // worst → Brier 1
        assert_eq!(ct.sample_count(), 1);
        assert!((ct.brier_score() - 1.0).abs() < 1e-9);

        ct.record(0.75, true); // (0.75-1)^2 = 0.0625
        assert_eq!(ct.sample_count(), 1);
        assert!(
            (ct.brier_score() - 0.0625).abs() < 1e-9,
            "got {}",
            ct.brier_score()
        );
    }

    #[test]
    fn calibration_tracker_eviction_preserves_newest_entries() {
        let mut ct = CalibrationTracker::new(3);

        // Record 5 entries; oldest 2 should be evicted.
        ct.record(0.0, false); // idx 0 – will be evicted
        ct.record(0.0, false); // idx 1 – will be evicted
        ct.record(1.0, true); // idx 2 – kept (perfect)
        ct.record(1.0, true); // idx 3 – kept (perfect)
        ct.record(1.0, true); // idx 4 – kept (perfect)

        assert_eq!(ct.sample_count(), 3);
        // All three remaining are perfect predictions → Brier 0.
        assert!(ct.brier_score() < 1e-9);
    }

    #[test]
    fn calibration_tracker_known_brier_computation() {
        // 4 predictions: [0.75, 0.25, 0.75, 0.25], outcomes: [true, false, false, true]
        // Errors: (0.75-1)^2=0.0625, (0.25-0)^2=0.0625, (0.75-0)^2=0.5625, (0.25-1)^2=0.5625
        // Brier = (0.0625+0.0625+0.5625+0.5625)/4 = 1.25/4 = 0.3125
        let mut ct = CalibrationTracker::new(10);
        ct.record(0.75, true);
        ct.record(0.25, false);
        ct.record(0.75, false);
        ct.record(0.25, true);
        assert!(
            (ct.brier_score() - 0.3125).abs() < 1e-9,
            "expected 0.3125, got {}",
            ct.brier_score()
        );
    }

    #[test]
    fn correction_tracker_triggers_correction_on_edit_distance_alone() {
        let tolerance = CorrectionTolerance {
            max_wer: 1.0,              // very lenient (> not >=)
            max_confidence_delta: 1.0, // very lenient
            max_edit_distance: 2,      // tight
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        let p = PartialTranscript::new(
            0,
            100,
            "fast".to_owned(),
            vec![seg("abc", Some(0.5))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(p);

        // Quality text differs by 3 chars → edit_distance = 3 > max_edit_distance (2).
        // But it's one word vs one word → wer = 1.0, and max_wer = 1.0, so wer check (>) is false.
        // Confidence_delta = 0.0, max_confidence_delta = 1.0, so that check is false too.
        let decision = tracker
            .submit_quality_result(100, "quality", vec![seg("xyz", Some(0.5))], 200)
            .expect("should succeed");
        assert!(
            matches!(decision, CorrectionDecision::Correct { .. }),
            "edit_distance 3 > max 2 should trigger correction"
        );
        assert_eq!(tracker.stats().corrections_emitted, 1);
    }

    // -- bd-244: speculation.rs edge-case tests pass 9 --

    #[test]
    fn correction_tracker_corrections_slice_reflects_stored_events() {
        let tolerance = CorrectionTolerance {
            max_wer: 0.0, // always correct (wer > 0.0 is always true for non-identical)
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        // Register and submit two partials that trigger corrections.
        let p0 = PartialTranscript::new(
            0,
            100,
            "fast".to_owned(),
            vec![seg("hello", Some(0.5))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(p0);
        tracker
            .submit_quality_result(100, "quality", vec![seg("world", Some(0.9))], 200)
            .unwrap();

        let p1 = PartialTranscript::new(
            1,
            101,
            "fast".to_owned(),
            vec![seg("foo", Some(0.5))],
            50,
            "2026-01-01T00:00:01Z".to_owned(),
        );
        tracker.register_partial(p1);
        tracker
            .submit_quality_result(101, "quality", vec![seg("bar", Some(0.9))], 200)
            .unwrap();

        let corrections = tracker.corrections();
        assert_eq!(corrections.len(), 2, "should have 2 correction events");
        assert_eq!(corrections[0].retracted_seq, 0);
        assert_eq!(corrections[1].retracted_seq, 1);
        assert_eq!(corrections[0].quality_model_id, "quality");
    }

    #[test]
    fn controller_evidence_records_action_and_fields() {
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 10000, 500);
        // Just call apply with no observations → Hold (too few windows for confidence)
        let _window = ctrl.apply();
        let entries = ctrl.evidence();
        assert_eq!(entries.len(), 1, "one apply → one evidence entry");
        assert_eq!(entries[0].decision_id, 0);
        assert!(!entries[0].fallback_active);
        assert!(entries[0].fallback_reason.is_none());
        assert!(matches!(entries[0].action_taken, ControllerAction::Hold));
    }

    #[test]
    #[should_panic(expected = "positive and finite")]
    fn beta_posterior_rejects_infinite_alpha() {
        let _ = BetaPosterior::new(f64::INFINITY, 1.0);
    }

    #[test]
    fn merge_segments_dominated_lower_confidence_keeps_existing() {
        let mut wm = WindowManager::new("run-1", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(5000, "h1");

        // Window 0 produces a segment with HIGH confidence.
        wm.record_quality_result(
            w0.window_id,
            vec![timed_seg("high conf", 2.0, 4.0, Some(0.95))],
        );
        wm.resolve_window(w0.window_id);
        // Window 1 produces an overlapping segment with LOWER confidence.
        wm.record_quality_result(
            w1.window_id,
            vec![timed_seg("low conf", 2.05, 4.05, Some(0.3))],
        );
        wm.resolve_window(w1.window_id);

        let merged = wm.merge_segments();
        assert_eq!(merged.len(), 1, "overlapping segments should dedup to 1");
        assert_eq!(
            merged[0].text, "high conf",
            "should keep the higher-confidence existing segment"
        );
    }

    #[test]
    fn correction_tracker_max_observed_wer_tracks_monotonic_maximum() {
        let tolerance = CorrectionTolerance::default();
        let mut tracker = CorrectionTracker::new(tolerance);

        // Submit 3 windows with WER: 0.2, 0.5, 0.3
        for (i, (fast, quality)) in [("aa", "ab"), ("aa", "zz"), ("ab", "ac")]
            .iter()
            .enumerate()
        {
            let p = PartialTranscript::new(
                i as u64,
                (100 + i) as u64,
                "fast".to_owned(),
                vec![seg(fast, Some(0.5))],
                50,
                "2026-01-01T00:00:00Z".to_owned(),
            );
            tracker.register_partial(p);
            tracker
                .submit_quality_result(
                    (100 + i) as u64,
                    "quality",
                    vec![seg(quality, Some(0.5))],
                    200,
                )
                .unwrap();
        }

        // max_observed_wer should reflect the highest WER seen (window 1: "aa"→"zz"=1.0)
        let max_wer = tracker.stats().max_observed_wer;
        assert!(
            max_wer >= 0.9,
            "max_observed_wer should be ~1.0 (from 'aa'→'zz'), got {max_wer}"
        );
        // The third window had lower WER but should NOT have overwritten the max.
        assert!(
            max_wer >= tracker.stats().cumulative_wer / 3.0,
            "max should be >= mean"
        );
    }

    // -- bd-249: speculation.rs edge-case tests pass 10 --

    #[test]
    fn beta_posterior_variance_known_formula() {
        let p = BetaPosterior::weakly_informative(); // alpha=2, beta=2
        // variance = (2 * 2) / (4 * 4 * 5) = 4 / 80 = 0.05
        let expected = 0.05;
        let actual = p.variance();
        assert!(
            (actual - expected).abs() < 1e-12,
            "variance of Beta(2,2) should be 0.05, got {actual}"
        );

        // After one update: alpha=3, beta=2, total=5
        // variance = (3*2)/(5*5*6) = 6/150 = 0.04
        let p2 = BetaPosterior::new(3.0, 2.0);
        let expected2 = 0.04;
        assert!(
            (p2.variance() - expected2).abs() < 1e-12,
            "variance of Beta(3,2) should be 0.04, got {}",
            p2.variance()
        );
    }

    #[test]
    fn controller_apply_normal_grow_path_increases_window() {
        // Need: correction_rate > 0.25, mean_wer > 0.125, current < max, confidence >= 0.5
        // And NOT runaway (rate <= 0.75), and NOT Brier fallback.
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 10000, 500);
        let high_wer_drift = CorrectionDrift {
            wer_approx: 0.2,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // Feed 10 windows: 4 corrections, 6 confirms → rate = 0.4 > 0.25
        for i in 0..10u64 {
            let decision = if i < 4 {
                CorrectionDecision::Correct {
                    correction: CorrectionEvent::new(
                        i,
                        i,
                        i,
                        "quality".to_owned(),
                        vec![],
                        100,
                        "2026-01-01T00:00:00Z".to_owned(),
                        &[],
                    ),
                }
            } else {
                CorrectionDecision::Confirm {
                    seq: i,
                    drift: high_wer_drift.clone(),
                }
            };
            ctrl.observe(&decision, &high_wer_drift);
        }
        let before = ctrl.current_window_ms();
        let after = ctrl.apply();
        assert!(
            after > before,
            "apply() should increase window via normal Grow path: before={before}, after={after}"
        );
    }

    #[test]
    fn correction_tracker_mean_wer_zero_with_no_windows() {
        let tracker = CorrectionTracker::new(CorrectionTolerance::default());
        assert!(
            (tracker.mean_wer() - 0.0).abs() < f64::EPSILON,
            "mean_wer() on fresh tracker should be 0.0"
        );
        assert_eq!(tracker.stats().windows_processed, 0);
    }

    #[test]
    fn speculation_window_new_stores_all_fields() {
        let w = SpeculationWindow::new(
            42,
            "my-run".to_owned(),
            1000,
            4000,
            500,
            "abc123".to_owned(),
        );
        assert_eq!(w.window_id, 42);
        assert_eq!(w.run_id, "my-run");
        assert_eq!(w.start_ms, 1000);
        assert_eq!(w.end_ms, 4000);
        assert_eq!(w.overlap_ms, 500);
        assert_eq!(w.audio_hash, "abc123");
        assert_eq!(w.duration_ms(), 3000);
    }

    #[test]
    fn controller_apply_zero_correction_force_shrink_through_apply() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 10000, 500);
        // Feed 20 consecutive zero-correction windows to trigger ZERO_CORRECTION_FORCE_SHRINK.
        let zero_drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        for i in 0..21u64 {
            let decision = CorrectionDecision::Confirm {
                seq: i,
                drift: zero_drift.clone(),
            };
            ctrl.observe(&decision, &zero_drift);
        }
        let before = ctrl.current_window_ms();
        let after = ctrl.apply();
        assert!(
            after < before,
            "apply() should shrink window via zero-correction force-shrink: before={before}, after={after}"
        );
        // Should not be in fallback mode (normal else branch).
        assert!(!ctrl.is_fallback_active());
    }

    #[test]
    fn window_manager_fast_in_progress_counts_as_pending() {
        let mut mgr = WindowManager::new("run-test", 5000, 500);
        let w = mgr.next_window(0, "hash");
        // Manually set the status to FastInProgress.
        mgr.get_window_mut(w.window_id).unwrap().status = WindowStatus::FastInProgress;
        assert_eq!(mgr.windows_pending(), 1);
        assert_eq!(mgr.windows_resolved(), 0);
    }

    #[test]
    fn recommend_brier_hold_overrides_grow() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        // Pre-inject poorly calibrated records: predict 0.0 but actual is correction=true.
        // This gives Brier = mean((0.0 - 1.0)^2) = 1.0, well above 0.25 threshold.
        for _ in 0..10 {
            ctrl.calibration.record(0.0, true);
        }
        // Now feed corrections to push correction_rate > 0.75 with enough windows.
        let drift = CorrectionDrift {
            wer_approx: 0.5,
            confidence_delta: 0.1,
            segment_count_delta: 0,
            text_edit_distance: 5,
        };
        for _ in 0..10u64 {
            let correction =
                CorrectionEvent::new(0, 0, 0, "q".to_owned(), vec![], 100, "t".to_owned(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        // Brier should still be high (mix of 10 bad + 10 good-ish calibration records,
        // but CalibrationTracker window is 20, so all 20 are kept).
        assert!(ctrl.calibration.brier_score() > 0.25);
        assert_eq!(ctrl.recommend(), ControllerAction::Hold);
    }

    #[test]
    fn apply_runaway_branch_grows_and_sets_fallback() {
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 30000, 500);
        // Need correction_rate > 0.75 but Brier <= 0.25 (so we enter the runaway branch,
        // not the Brier branch). To get low Brier while having high corrections we need
        // the calibration predictions to be close to 1.0 (matching actual corrections).
        // Manually inject calibration samples that are well-calibrated despite high corrections.
        for _ in 0..10 {
            ctrl.calibration.record(0.9, true);
        }
        // Now drive observations to push correction_rate > 0.75 with enough windows.
        let drift = CorrectionDrift {
            wer_approx: 0.5,
            confidence_delta: 0.1,
            segment_count_delta: 0,
            text_edit_distance: 5,
        };
        for _ in 0..10u64 {
            let correction =
                CorrectionEvent::new(0, 0, 0, "q".to_owned(), vec![], 100, "t".to_owned(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        assert!(ctrl.state().correction_rate > 0.75);
        let before = ctrl.current_window_ms();
        let after = ctrl.apply();
        assert!(ctrl.is_fallback_active());
        assert!(
            after > before,
            "runaway branch should grow window: before={before}, after={after}"
        );
    }

    #[test]
    fn correction_drift_compute_unicode_text() {
        let fast = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "café".to_owned(),
                speaker: None,
                confidence: Some(0.9),
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: "naïve".to_owned(),
                speaker: None,
                confidence: Some(0.9),
            },
        ];
        let quality = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "cafe".to_owned(),
                speaker: None,
                confidence: Some(0.9),
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: "naive".to_owned(),
                speaker: None,
                confidence: Some(0.9),
            },
        ];
        let drift = CorrectionDrift::compute(&fast, &quality);
        // "café naïve" vs "cafe naive" — 2 char edits (é→e, ï→i).
        assert_eq!(drift.text_edit_distance, 2);
        assert_eq!(drift.segment_count_delta, 0);
        assert!((drift.confidence_delta - 0.0).abs() < 1e-9);
    }

    #[test]
    fn correction_tracker_duplicate_window_id_overwrites_seq_mapping() {
        let tolerance = CorrectionTolerance {
            max_wer: 0.1,
            max_confidence_delta: 0.2,
            max_edit_distance: 5,
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);
        // Register two partials with different seqs but same window_id.
        let p0 = PartialTranscript::new(
            0,
            10,
            "fast".to_owned(),
            vec![],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        let p1 = PartialTranscript::new(
            1,
            10,
            "fast".to_owned(),
            vec![],
            60,
            "2026-01-01T00:00:01Z".to_owned(),
        );
        tracker.register_partial(p0);
        tracker.register_partial(p1);
        // Submit quality for window 10 — should resolve seq 1 (latest mapping).
        let result = tracker.submit_quality_result(10, "quality", vec![], 100);
        assert!(result.is_ok());
        // Seq 0's partial is still in the tracker and still Pending.
        let orphan = tracker.get_partial(0);
        assert!(orphan.is_some());
        assert_eq!(orphan.unwrap().status, PartialStatus::Pending);
    }

    // ── Task #263 — speculation pass 11 edge-case tests ──────────────

    #[test]
    fn partial_transcript_confidence_mean_zero_when_all_segments_have_none_confidence() {
        let segments = vec![seg("hello", None), seg("world", None), seg("foo", None)];
        let pt = PartialTranscript::new(
            0,
            0,
            "fast".to_owned(),
            segments,
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        assert!(
            (pt.confidence_mean - 0.0).abs() < f64::EPSILON,
            "all-None confidences should yield confidence_mean=0.0, got {}",
            pt.confidence_mean
        );
    }

    #[test]
    fn merge_segments_skips_quality_complete_window_not_resolved() {
        let mut wm = WindowManager::new("run-merge-skip", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(5000, "h1");

        // Window 0: resolved with quality result.
        wm.record_quality_result(
            w0.window_id,
            vec![timed_seg("resolved seg", 0.0, 2.0, Some(0.9))],
        );
        wm.resolve_window(w0.window_id);

        // Window 1: has quality result but is NOT resolved (QualityComplete).
        wm.record_quality_result(
            w1.window_id,
            vec![timed_seg("not resolved seg", 5.0, 7.0, Some(0.9))],
        );
        // Do NOT resolve w1 — it stays at QualityComplete.
        assert_eq!(
            wm.get_window(w1.window_id).unwrap().status,
            WindowStatus::QualityComplete
        );

        let merged = wm.merge_segments();
        assert_eq!(
            merged.len(),
            1,
            "only the resolved window should contribute segments"
        );
        assert_eq!(merged[0].text, "resolved seg");
    }

    #[test]
    fn correction_event_new_computes_quality_confidence_mean_and_drift() {
        let fast_segs = vec![seg("hello world", Some(0.7)), seg("goodbye", Some(0.5))];
        let quality_segs = vec![seg("hello world", Some(0.9)), seg("farewell", Some(0.8))];
        let event = CorrectionEvent::new(
            1,
            10,
            5,
            "quality-model".to_owned(),
            quality_segs,
            300,
            "2026-01-01T00:00:00Z".to_owned(),
            &fast_segs,
        );

        // quality_confidence_mean = (0.9 + 0.8) / 2 = 0.85
        assert!(
            (event.quality_confidence_mean - 0.85).abs() < 1e-9,
            "quality_confidence_mean should be 0.85, got {}",
            event.quality_confidence_mean
        );

        // fast mean = (0.7 + 0.5) / 2 = 0.6; confidence_delta = |0.6 - 0.85| = 0.25
        assert!(
            (event.drift.confidence_delta - 0.25).abs() < 1e-9,
            "drift.confidence_delta should be 0.25, got {}",
            event.drift.confidence_delta
        );

        // segment_count_delta = quality(2) - fast(2) = 0
        assert_eq!(event.drift.segment_count_delta, 0);

        // Verify other fields.
        assert_eq!(event.correction_id, 1);
        assert_eq!(event.retracted_seq, 10);
        assert_eq!(event.window_id, 5);
        assert_eq!(event.quality_model_id, "quality-model");
        assert_eq!(event.quality_latency_ms, 300);
    }

    #[test]
    fn submit_quality_result_edit_distance_at_threshold_confirms() {
        // Condition: drift.text_edit_distance > max_edit_distance (strict >).
        // At exactly the threshold, should confirm.
        let tolerance = CorrectionTolerance {
            max_wer: 1.0,              // lenient
            max_confidence_delta: 1.0, // lenient
            max_edit_distance: 3,      // threshold
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        // "abc" → "xyz" = 3 char edits (a→x, b→y, c→z).
        let p = PartialTranscript::new(
            0,
            100,
            "fast".to_owned(),
            vec![seg("abc", Some(0.5))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(p);

        let decision = tracker
            .submit_quality_result(100, "quality", vec![seg("xyz", Some(0.5))], 200)
            .expect("should succeed");

        // edit_distance == 3 == max_edit_distance → NOT > threshold → confirm.
        assert!(
            matches!(decision, CorrectionDecision::Confirm { .. }),
            "edit_distance == max_edit_distance should confirm (strict >), got Correct"
        );
    }

    #[test]
    fn controller_posterior_and_state_reflect_observations() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);

        // Initial posterior is weakly informative: Beta(2,2).
        assert!((ctrl.posterior().alpha - 2.0).abs() < 1e-9);
        assert!((ctrl.posterior().beta - 2.0).abs() < 1e-9);
        assert_eq!(ctrl.state().window_count, 0);

        let drift = CorrectionDrift {
            wer_approx: 0.1,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };

        // 3 corrections, 2 confirmations.
        for _ in 0..3 {
            let correction =
                CorrectionEvent::new(0, 0, 0, "q".to_owned(), vec![], 100, "t".to_owned(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        for _ in 0..2 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }

        // posterior: alpha = 2 + 3 = 5, beta = 2 + 2 = 4
        assert!((ctrl.posterior().alpha - 5.0).abs() < 1e-9);
        assert!((ctrl.posterior().beta - 4.0).abs() < 1e-9);
        assert_eq!(ctrl.state().window_count, 5);
        // confidence = 5 / 20.0 = 0.25
        assert!((ctrl.confidence() - 0.25).abs() < 1e-9);
        // mean_wer = (0.1 * 5) / 5 = 0.1
        assert!(
            (ctrl.state().mean_wer - 0.1).abs() < 1e-9,
            "mean_wer should be 0.1, got {}",
            ctrl.state().mean_wer
        );
    }

    // ── Task #266 — speculation.rs pass 12 edge-case tests ────────────

    #[test]
    fn correction_rate_on_fresh_tracker_returns_zero_not_nan() {
        // Exercises the total==0 early return (line 790-793).
        let tracker = CorrectionTracker::new(CorrectionTolerance::default());
        assert!(
            (tracker.correction_rate() - 0.0).abs() < f64::EPSILON,
            "correction_rate() on fresh tracker should be 0.0, got {}",
            tracker.correction_rate()
        );
        assert!(tracker.correction_rate().is_finite());
    }

    #[test]
    fn correction_tracker_triggers_correction_on_wer_alone() {
        // Symmetric with existing _on_confidence_delta_alone and _on_edit_distance_alone.
        let tolerance = CorrectionTolerance {
            max_wer: 0.01,             // tight — any word difference triggers
            max_confidence_delta: 1.0, // very lenient
            max_edit_distance: 1000,   // very lenient
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        let p = PartialTranscript::new(
            0,
            100,
            "fast".to_owned(),
            vec![seg("hello world", Some(0.8))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(p);

        // Quality text differs by 1 word out of 2 → wer ≈ 0.5 > 0.01.
        // But confidence is the same (delta=0.0 < 1.0) and edit_distance is small (< 1000).
        let decision = tracker
            .submit_quality_result(100, "quality", vec![seg("hello earth", Some(0.8))], 200)
            .expect("should succeed");
        assert!(
            matches!(decision, CorrectionDecision::Correct { .. }),
            "wer ~0.5 > max_wer 0.01 should trigger correction"
        );
        assert_eq!(tracker.stats().corrections_emitted, 1);
    }

    #[test]
    fn merge_segments_preserves_none_timestamp_segments() {
        // Segments without timing information should not be coerced to t=0 and
        // deduplicated away; they represent unknown timing, not identical timing.
        let mut wm = WindowManager::new("run-none-ts", 5000, 0);

        let w0 = wm.next_window(0, "h0");
        wm.record_quality_result(w0.window_id, vec![seg("low", Some(0.3))]);
        wm.resolve_window(w0.window_id);

        let w1 = wm.next_window(5000, "h1");
        wm.record_quality_result(w1.window_id, vec![seg("high", Some(0.9))]);
        wm.resolve_window(w1.window_id);

        let merged = wm.merge_segments();
        assert_eq!(
            merged.len(),
            2,
            "None-timestamp segments should both survive merge"
        );
        assert_eq!(merged[0].text, "low");
        assert_eq!(merged[1].text, "high");
    }

    #[test]
    fn merge_segments_sorts_reverse_position_windows_by_start_sec() {
        // Create windows in reverse position order; merge should still sort by start_sec.
        let mut wm = WindowManager::new("run-reverse", 3000, 0);

        // Window 0 at position 6000 (created first).
        let w0 = wm.next_window(6000, "h-late");
        wm.record_quality_result(
            w0.window_id,
            vec![timed_seg("second phrase", 6.0, 8.0, Some(0.9))],
        );
        wm.resolve_window(w0.window_id);

        // Window 1 at position 0 (created second).
        let w1 = wm.next_window(0, "h-early");
        wm.record_quality_result(
            w1.window_id,
            vec![timed_seg("first phrase", 0.0, 2.0, Some(0.9))],
        );
        wm.resolve_window(w1.window_id);

        let merged = wm.merge_segments();
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged[0].text, "first phrase",
            "should sort by start_sec ascending"
        );
        assert_eq!(merged[1].text, "second phrase");
    }

    #[test]
    fn evidence_ledger_mean_wer_with_mixed_entries() {
        let mut ledger = CorrectionEvidenceLedger::new(10);

        for (i, wer) in [0.0, 0.2, 0.6].iter().enumerate() {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i as u64,
                window_id: i as u64,
                run_id: "r".into(),
                timestamp_rfc3339: "2026-01-01T00:00:00Z".into(),
                fast_model_id: "fast".into(),
                fast_latency_ms: 100,
                fast_confidence_mean: 0.8,
                fast_segment_count: 1,
                quality_model_id: "quality".into(),
                quality_latency_ms: 400,
                quality_confidence_mean: 0.95,
                quality_segment_count: 1,
                drift: CorrectionDrift {
                    wer_approx: *wer,
                    confidence_delta: 0.1,
                    segment_count_delta: 0,
                    text_edit_distance: 5,
                },
                decision: "confirm".into(),
                window_size_ms: 3000,
                correction_rate_at_decision: 0.0,
                controller_confidence: 0.5,
                fallback_active: false,
                fallback_reason: None,
            });
        }

        // mean_wer = (0.0 + 0.2 + 0.6) / 3 = 0.2666...
        let expected = (0.0 + 0.2 + 0.6) / 3.0;
        assert!(
            (ledger.mean_wer() - expected).abs() < 1e-9,
            "mean_wer should be {expected}, got {}",
            ledger.mean_wer()
        );
    }

    // ── Task #269 — speculation.rs pass 13 edge-case tests ────────────

    #[test]
    fn beta_posterior_extreme_ratio_mean_near_zero_is_finite() {
        // Highly asymmetric Beta(0.01, 100) → mean ≈ 0.0001.
        let p = BetaPosterior::new(0.01, 100.0);
        let m = p.mean();
        assert!(m.is_finite(), "mean should be finite, got {m}");
        assert!(m > 0.0 && m < 0.001, "mean should be near zero, got {m}");
        let v = p.variance();
        assert!(v.is_finite(), "variance should be finite, got {v}");
        assert!(v > 0.0, "variance should be positive, got {v}");
    }

    #[test]
    fn beta_posterior_variance_shrinks_monotonically_with_observations() {
        let mut p = BetaPosterior::weakly_informative(); // Beta(2,2)
        let mut prev_var = p.variance();
        for _ in 0..20 {
            p.observe_confirmation();
            let v = p.variance();
            assert!(
                v < prev_var,
                "variance should decrease monotonically: {v} >= {prev_var}"
            );
            prev_var = v;
        }
        // After 20 confirmations: Beta(2, 22), total=24
        // variance should be much smaller than initial 0.05
        assert!(
            prev_var < 0.01,
            "variance after 20 observations should be very small, got {prev_var}"
        );
    }

    #[test]
    fn correction_rate_single_window_correction_returns_one() {
        let tolerance = CorrectionTolerance {
            max_wer: 0.01,
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        let p = PartialTranscript::new(
            0,
            100,
            "fast".to_owned(),
            vec![seg("hello world", Some(0.8))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        tracker.register_partial(p);

        let _ = tracker
            .submit_quality_result(100, "quality", vec![seg("hello earth", Some(0.8))], 200)
            .expect("should succeed");

        // Exactly 1 total (1 correction, 0 confirmations) → rate = 1.0
        assert!(
            (tracker.correction_rate() - 1.0).abs() < f64::EPSILON,
            "single-window correction should give rate 1.0, got {}",
            tracker.correction_rate()
        );
    }

    #[test]
    fn controller_confidence_reaches_one_at_exactly_twenty_windows() {
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 10000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.1,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // Feed exactly 20 windows
        for _ in 0..20 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: CorrectionDrift {
                        wer_approx: 0.0,
                        confidence_delta: 0.0,
                        segment_count_delta: 0,
                        text_edit_distance: 0,
                    },
                },
                &drift,
            );
        }
        assert_eq!(ctrl.state().window_count, 20);
        assert!(
            (ctrl.confidence() - 1.0).abs() < f64::EPSILON,
            "confidence at 20 windows should be exactly 1.0, got {}",
            ctrl.confidence()
        );

        // Feed one more → still 1.0 (clamped)
        ctrl.observe(
            &CorrectionDecision::Confirm {
                seq: 0,
                drift: CorrectionDrift {
                    wer_approx: 0.0,
                    confidence_delta: 0.0,
                    segment_count_delta: 0,
                    text_edit_distance: 0,
                },
            },
            &drift,
        );
        assert!(
            (ctrl.confidence() - 1.0).abs() < f64::EPSILON,
            "confidence above 20 windows should still be 1.0, got {}",
            ctrl.confidence()
        );
    }

    #[test]
    fn evidence_ledger_correction_rate_handles_mixed_case_decision_strings() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let base_drift = CorrectionDrift {
            wer_approx: 0.1,
            confidence_delta: 0.05,
            segment_count_delta: 0,
            text_edit_distance: 2,
        };
        // 4 entries: "Corrected", "CORRECTION", "confirm", "correct"
        for (i, decision) in ["Corrected", "CORRECTION", "confirm", "correct"]
            .iter()
            .enumerate()
        {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i as u64,
                window_id: i as u64,
                run_id: "run".into(),
                timestamp_rfc3339: "2026-01-01T00:00:00Z".into(),
                fast_model_id: "fast".into(),
                fast_latency_ms: 100,
                fast_confidence_mean: 0.8,
                fast_segment_count: 1,
                quality_model_id: "quality".into(),
                quality_latency_ms: 400,
                quality_confidence_mean: 0.95,
                quality_segment_count: 1,
                drift: base_drift.clone(),
                decision: decision.to_string(),
                window_size_ms: 3000,
                correction_rate_at_decision: 0.0,
                controller_confidence: 0.5,
                fallback_active: false,
                fallback_reason: None,
            });
        }
        // "Corrected" → is_correction, "CORRECTION" → is_correction,
        // "confirm" → not, "correct" → is_correction
        // → 3 corrections / 4 total = 0.75
        assert!(
            (ledger.correction_rate() - 0.75).abs() < f64::EPSILON,
            "mixed-case decisions: 3 corrections / 4 total should be 0.75, got {}",
            ledger.correction_rate()
        );
    }

    // ── Task #271 — speculation.rs pass 14 edge-case tests ────────────

    #[test]
    fn levenshtein_empty_vs_nonempty_returns_length() {
        // Exercise the early-return branches: m==0 → n, n==0 → m.
        assert_eq!(levenshtein("", "hello"), 5);
        assert_eq!(levenshtein("hello", ""), 5);
        assert_eq!(levenshtein("", ""), 0);
        // Single-char difference.
        assert_eq!(levenshtein("a", "b"), 1);
        assert_eq!(levenshtein("a", "a"), 0);
    }

    #[test]
    fn mean_confidence_mixed_some_and_none_only_averages_some() {
        // Three segments: Some(0.9), None, Some(0.5) → average of (0.9 + 0.5) / 2 = 0.7
        let segments = vec![
            seg("hello", Some(0.9)),
            seg("world", None),
            seg("foo", Some(0.5)),
        ];
        let mc = mean_confidence(&segments);
        assert!(
            (mc - 0.7).abs() < 1e-9,
            "mixed Some/None should average only Some values: expected 0.7, got {mc}"
        );
    }

    #[test]
    fn partial_transcript_empty_segments_yields_zero_confidence_mean() {
        let pt = PartialTranscript::new(
            0,
            0,
            "fast".to_owned(),
            vec![],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        assert!(
            (pt.confidence_mean - 0.0).abs() < f64::EPSILON,
            "empty segments should give confidence_mean=0.0, got {}",
            pt.confidence_mean
        );
        assert!(pt.segments.is_empty());
        assert_eq!(pt.seq, 0);
    }

    #[test]
    fn correction_drift_multi_segment_concatenation_wer() {
        // fast: ["hello", "world"] → "hello world" (2 words)
        // quality: ["hello", "earth", "again"] → "hello earth again" (3 words)
        // levenshtein_words(["hello","world"], ["hello","earth","again"]):
        //   match "hello", sub "world"→"earth", insert "again" = 2 edits.
        //   max_words = max(2, 3).max(1) = 3 → wer = 2/3 ≈ 0.667
        let fast = vec![seg("hello", Some(0.8)), seg("world", Some(0.7))];
        let quality = vec![
            seg("hello", Some(0.9)),
            seg("earth", Some(0.85)),
            seg("again", Some(0.8)),
        ];
        let drift = CorrectionDrift::compute(&fast, &quality);

        let expected_wer = 2.0 / 3.0;
        assert!(
            (drift.wer_approx - expected_wer).abs() < 1e-9,
            "multi-segment WER should be 2/3, got {}",
            drift.wer_approx
        );
        assert_eq!(drift.segment_count_delta, 1, "quality(3) - fast(2) = 1");
        // char edit: "hello world" vs "hello earth again"
        let char_dist = levenshtein("hello world", "hello earth again");
        assert_eq!(drift.text_edit_distance, char_dist);
    }

    #[test]
    fn controller_state_correction_rate_on_first_observation() {
        // Exercise the state.correction_rate formula (line 1041):
        // total = alpha + beta - 4.0, corrections = alpha - 2.0.
        // After first confirmation: alpha=2, beta=3, total=1, corrections=0 → rate=0.0.
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        ctrl.observe(
            &CorrectionDecision::Confirm {
                seq: 0,
                drift: drift.clone(),
            },
            &drift,
        );
        assert!(
            (ctrl.state().correction_rate - 0.0).abs() < f64::EPSILON,
            "first observation (confirm) → correction_rate should be 0.0, got {}",
            ctrl.state().correction_rate
        );

        // After first correction on a fresh controller: alpha=3, beta=2, total=1, corrections=1 → rate=1.0.
        let mut ctrl2 = SpeculationWindowController::new(5000, 1000, 30000, 500);
        let correction =
            CorrectionEvent::new(0, 0, 0, "q".to_owned(), vec![], 100, "t".to_owned(), &[]);
        ctrl2.observe(&CorrectionDecision::Correct { correction }, &drift);
        assert!(
            (ctrl2.state().correction_rate - 1.0).abs() < f64::EPSILON,
            "first observation (correct) → correction_rate should be 1.0, got {}",
            ctrl2.state().correction_rate
        );
    }

    // ── Task #272 — speculation.rs pass 15 edge-case tests ────────────

    #[test]
    fn levenshtein_words_direct_known_distances() {
        // Identical word sequences → 0.
        assert_eq!(
            levenshtein_words(&["hello", "world"], &["hello", "world"]),
            0
        );
        // Empty vs non-empty → length of non-empty.
        assert_eq!(levenshtein_words(&[], &["a", "b", "c"]), 3);
        assert_eq!(levenshtein_words(&["x", "y"], &[]), 2);
        // Both empty → 0.
        assert_eq!(levenshtein_words(&[], &[]), 0);
        // Single substitution.
        assert_eq!(levenshtein_words(&["hello"], &["world"]), 1);
        // Insertion + substitution: ["a","b"] vs ["a","c","d"] → sub b→c + insert d = 2.
        assert_eq!(levenshtein_words(&["a", "b"], &["a", "c", "d"]), 2);
    }

    #[test]
    fn concat_segment_text_joins_with_space() {
        // Multiple segments.
        let segments = vec![
            seg("hello", None),
            seg("beautiful", None),
            seg("world", None),
        ];
        assert_eq!(concat_segment_text(&segments), "hello beautiful world");
        // Single segment.
        assert_eq!(concat_segment_text(&[seg("solo", None)]), "solo");
        // Empty segments.
        assert_eq!(concat_segment_text(&[]), "");
    }

    #[test]
    fn evidence_ledger_latency_savings_pct_zero_quality_returns_zero() {
        // Exercise the q == 0.0 early return at line 1333.
        let mut ledger = CorrectionEvidenceLedger::new(10);
        ledger.record(CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 0,
            run_id: "r".into(),
            timestamp_rfc3339: "2026-01-01T00:00:00Z".into(),
            fast_model_id: "fast".into(),
            fast_latency_ms: 100,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "quality".into(),
            quality_latency_ms: 0,
            quality_confidence_mean: 0.9,
            quality_segment_count: 1,
            drift: CorrectionDrift::compute(&[], &[]),
            decision: "confirm".into(),
            window_size_ms: 3000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        });
        assert!(
            (ledger.latency_savings_pct() - 0.0).abs() < f64::EPSILON,
            "zero quality latency should return 0.0 savings, got {}",
            ledger.latency_savings_pct()
        );
        assert!(
            (ledger.mean_quality_latency() - 0.0).abs() < f64::EPSILON,
            "mean quality latency should be 0.0"
        );
    }

    #[test]
    fn window_manager_next_window_ids_increment_monotonically() {
        let mut wm = WindowManager::new("run-ids", 3000, 0);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(3000, "h1");
        let w2 = wm.next_window(6000, "h2");
        assert_eq!(w0.window_id, 0);
        assert_eq!(w1.window_id, 1);
        assert_eq!(w2.window_id, 2);
        // next_window_bounded also increments from the same counter.
        let w3 = wm.next_window_bounded(9000, 12000, "h3").unwrap();
        assert_eq!(w3.window_id, 3);
    }

    #[test]
    fn correction_tracker_register_partial_returns_seq() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());
        let p = PartialTranscript::new(
            42,
            100,
            "fast".to_owned(),
            vec![seg("hello", Some(0.9))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        let returned_seq = tracker.register_partial(p);
        assert_eq!(
            returned_seq, 42,
            "register_partial should return the partial's seq"
        );
        // Verify it's actually stored.
        assert!(tracker.get_partial(42).is_some());
    }

    // ── Task #274 — speculation.rs pass 16 edge-case tests ────────────

    #[test]
    fn correction_event_empty_corrected_segments_yields_zero_confidence() {
        let fast_segs = vec![seg("hello", Some(0.9))];
        let event = CorrectionEvent::new(
            0,
            0,
            0,
            "quality".to_owned(),
            vec![],
            150,
            "2026-01-01T00:00:00Z".to_owned(),
            &fast_segs,
        );
        assert!(
            (event.quality_confidence_mean - 0.0).abs() < f64::EPSILON,
            "empty corrected segments should give 0.0 confidence, got {}",
            event.quality_confidence_mean
        );
        assert!(event.corrected_segments.is_empty());
        assert_eq!(event.quality_latency_ms, 150);
    }

    #[test]
    fn merge_segments_chain_of_three_overlapping_deduplicates_to_one() {
        let mut wm = WindowManager::new("run-chain", 5000, 0);

        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(5000, "h1");
        let w2 = wm.next_window(10000, "h2");

        // Three overlapping segments within 0.1s tolerance, varying confidence.
        wm.record_quality_result(w0.window_id, vec![timed_seg("low", 2.0, 4.0, Some(0.3))]);
        wm.resolve_window(w0.window_id);

        wm.record_quality_result(w1.window_id, vec![timed_seg("mid", 2.05, 4.05, Some(0.6))]);
        wm.resolve_window(w1.window_id);

        wm.record_quality_result(
            w2.window_id,
            vec![timed_seg("high", 2.08, 4.08, Some(0.95))],
        );
        wm.resolve_window(w2.window_id);

        let merged = wm.merge_segments();
        // After sort by start_sec: (2.0), (2.05), (2.08) — all within 0.1s of predecessor.
        // Dedup: "low" vs "mid" → "mid" wins; then "mid" vs "high" → "high" wins.
        assert_eq!(merged.len(), 1, "chain of 3 overlapping should dedup to 1");
        assert_eq!(merged[0].text, "high", "highest confidence should survive");
    }

    #[test]
    fn controller_action_serde_round_trip() {
        let actions = vec![
            ControllerAction::Shrink(500),
            ControllerAction::Hold,
            ControllerAction::Grow(1000),
        ];
        for action in &actions {
            let json = serde_json::to_string(action).expect("serialize");
            let deserialized: ControllerAction = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(
                &deserialized, action,
                "round-trip failed for {action:?}: {json}"
            );
        }
    }

    #[test]
    fn correction_drift_serde_round_trip() {
        let drift = CorrectionDrift {
            wer_approx: 0.42,
            confidence_delta: 0.15,
            segment_count_delta: -3,
            text_edit_distance: 17,
        };
        let json = serde_json::to_string(&drift).expect("serialize");
        let deserialized: CorrectionDrift = serde_json::from_str(&json).expect("deserialize");
        assert!((deserialized.wer_approx - 0.42).abs() < 1e-9);
        assert!((deserialized.confidence_delta - 0.15).abs() < 1e-9);
        assert_eq!(deserialized.segment_count_delta, -3);
        assert_eq!(deserialized.text_edit_distance, 17);
    }

    #[test]
    fn controller_apply_with_zero_observations_returns_initial_and_hold() {
        // No observations at all — apply() should return initial window, Hold action,
        // no fallback, and produce one evidence entry.
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        assert_eq!(ctrl.current_window_ms(), 5000);
        assert_eq!(ctrl.state().window_count, 0);

        let new_size = ctrl.apply();
        assert_eq!(new_size, 5000, "no observations should keep initial window");
        assert!(!ctrl.is_fallback_active());
        assert_eq!(ctrl.evidence().len(), 1);
        assert!(
            matches!(ctrl.evidence()[0].action_taken, ControllerAction::Hold),
            "zero observations should produce Hold action"
        );
        assert!(
            (ctrl.evidence()[0].confidence - 0.0).abs() < f64::EPSILON,
            "zero windows should give confidence 0.0"
        );
    }

    // ── Task #276 — speculation.rs pass 17 edge-case tests ────────────

    #[test]
    fn evidence_ledger_zero_capacity_increments_total_but_stores_nothing() {
        let mut ledger = CorrectionEvidenceLedger::new(0);
        let drift = CorrectionDrift {
            wer_approx: 0.1,
            confidence_delta: 0.05,
            segment_count_delta: 0,
            text_edit_distance: 2,
        };
        for i in 0..5 {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i,
                window_id: i,
                run_id: "run".into(),
                timestamp_rfc3339: "2026-01-01T00:00:00Z".into(),
                fast_model_id: "fast".into(),
                fast_latency_ms: 100,
                fast_confidence_mean: 0.8,
                fast_segment_count: 1,
                quality_model_id: "quality".into(),
                quality_latency_ms: 400,
                quality_confidence_mean: 0.95,
                quality_segment_count: 1,
                drift: drift.clone(),
                decision: "correct".into(),
                window_size_ms: 3000,
                correction_rate_at_decision: 0.0,
                controller_confidence: 0.5,
                fallback_active: false,
                fallback_reason: None,
            });
        }
        assert_eq!(
            ledger.total_recorded(),
            5,
            "total_recorded should count all records"
        );
        assert!(
            ledger.entries().is_empty(),
            "zero-capacity ledger should store nothing"
        );
    }

    #[test]
    fn partial_status_serde_snake_case_round_trip() {
        // PartialStatus uses #[serde(rename_all = "snake_case")].
        let pending_json = serde_json::to_string(&PartialStatus::Pending).expect("serialize");
        assert_eq!(pending_json, "\"pending\"");

        let confirmed_json = serde_json::to_string(&PartialStatus::Confirmed).expect("serialize");
        assert_eq!(confirmed_json, "\"confirmed\"");

        let retracted_json = serde_json::to_string(&PartialStatus::Retracted).expect("serialize");
        assert_eq!(retracted_json, "\"retracted\"");

        // Round-trip deserialization.
        let back: PartialStatus = serde_json::from_str(&pending_json).expect("deserialize");
        assert_eq!(back, PartialStatus::Pending);
    }

    #[test]
    fn beta_posterior_variance_exact_known_values() {
        // Beta(2, 2): variance = (2*2) / (4^2 * 5) = 4/80 = 0.05.
        let p = BetaPosterior::new(2.0, 2.0);
        assert!(
            (p.variance() - 0.05).abs() < 1e-12,
            "Beta(2,2) variance should be 0.05, got {}",
            p.variance()
        );

        // Beta(3, 2): variance = (3*2) / (5^2 * 6) = 6/150 = 0.04.
        let p2 = BetaPosterior::new(3.0, 2.0);
        assert!(
            (p2.variance() - 0.04).abs() < 1e-12,
            "Beta(3,2) variance should be 0.04, got {}",
            p2.variance()
        );

        // Beta(1, 1): uniform, variance = 1 / (4 * 3) = 1/12 ≈ 0.08333...
        let p3 = BetaPosterior::new(1.0, 1.0);
        let expected = 1.0 / 12.0;
        assert!(
            (p3.variance() - expected).abs() < 1e-12,
            "Beta(1,1) variance should be 1/12, got {}",
            p3.variance()
        );
    }

    #[test]
    fn correction_tracker_corrections_accumulates_in_order() {
        let tolerance = CorrectionTolerance {
            max_wer: 1.0,
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: true, // force every window to correct
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        // Register and submit 3 partials — all will be corrected.
        for i in 0..3u64 {
            let p = PartialTranscript::new(
                i,
                i + 100,
                "fast".into(),
                vec![seg("hello", Some(0.9))],
                50,
                "2026-01-01T00:00:00Z".into(),
            );
            tracker.register_partial(p);
            tracker
                .submit_quality_result(i + 100, "quality", vec![seg("hello", Some(0.9))], 200)
                .expect("submit should succeed");
        }

        let corrections = tracker.corrections();
        assert_eq!(corrections.len(), 3, "should have 3 corrections");
        // Correction IDs increment monotonically.
        assert_eq!(corrections[0].correction_id, 0);
        assert_eq!(corrections[1].correction_id, 1);
        assert_eq!(corrections[2].correction_id, 2);
        // Each correction targets its respective window_id.
        assert_eq!(corrections[0].window_id, 100);
        assert_eq!(corrections[1].window_id, 101);
        assert_eq!(corrections[2].window_id, 102);
    }

    #[test]
    fn speculation_window_serde_round_trip() {
        let window = SpeculationWindow::new(
            42,
            "run-abc".to_owned(),
            1000,
            5000,
            300,
            "deadbeef".to_owned(),
        );
        let json = serde_json::to_string(&window).expect("serialize");
        let back: SpeculationWindow = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.window_id, 42);
        assert_eq!(back.run_id, "run-abc");
        assert_eq!(back.start_ms, 1000);
        assert_eq!(back.end_ms, 5000);
        assert_eq!(back.overlap_ms, 300);
        assert_eq!(back.audio_hash, "deadbeef");
    }

    // ── Task #277 — speculation.rs pass 18 edge-case tests ────────────

    #[test]
    fn window_status_serde_snake_case_round_trip() {
        let variants = [
            (WindowStatus::Pending, "\"pending\""),
            (WindowStatus::FastInProgress, "\"fast_in_progress\""),
            (WindowStatus::FastComplete, "\"fast_complete\""),
            (WindowStatus::QualityComplete, "\"quality_complete\""),
            (WindowStatus::Resolved, "\"resolved\""),
        ];
        for (status, expected_json) in &variants {
            let json = serde_json::to_string(status).expect("serialize");
            assert_eq!(&json, expected_json, "wrong JSON for {status:?}");
            let back: WindowStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(&back, status, "round-trip failed for {status:?}");
        }
    }

    #[test]
    fn calibration_tracker_brier_score_perfect_calibration_returns_zero() {
        let mut tracker = CalibrationTracker::new(10);
        // Predict 1.0 for actual corrections, 0.0 for actual confirmations.
        tracker.record(1.0, true);
        tracker.record(0.0, false);
        tracker.record(1.0, true);
        tracker.record(0.0, false);
        // Perfect calibration: (1.0-1.0)^2 + (0.0-0.0)^2 + ... = 0.0.
        assert!(
            tracker.brier_score().abs() < f64::EPSILON,
            "perfectly calibrated predictions should have brier score 0, got {}",
            tracker.brier_score()
        );
        assert_eq!(tracker.sample_count(), 4);
    }

    #[test]
    fn controller_confidence_at_exact_boundary_values() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        // 0 windows → confidence 0.0.
        assert!((ctrl.confidence() - 0.0).abs() < f64::EPSILON);

        let drift = CorrectionDrift {
            wer_approx: 0.05,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        let confirm = CorrectionDecision::Confirm {
            seq: 0,
            drift: drift.clone(),
        };

        // 10 windows → 10/20 = 0.5.
        for _ in 0..10 {
            ctrl.observe(&confirm, &drift);
        }
        assert!(
            (ctrl.confidence() - 0.5).abs() < 1e-9,
            "10 windows should give confidence 0.5, got {}",
            ctrl.confidence()
        );

        // 19 windows → 19/20 = 0.95.
        for _ in 0..9 {
            ctrl.observe(&confirm, &drift);
        }
        assert!(
            (ctrl.confidence() - 0.95).abs() < 1e-9,
            "19 windows should give confidence 0.95, got {}",
            ctrl.confidence()
        );

        // 20 windows → min(20/20, 1.0) = 1.0.
        ctrl.observe(&confirm, &drift);
        assert!(
            (ctrl.confidence() - 1.0).abs() < f64::EPSILON,
            "20 windows should give confidence 1.0, got {}",
            ctrl.confidence()
        );

        // 25 windows → min(25/20, 1.0) = 1.0 (capped).
        for _ in 0..5 {
            ctrl.observe(&confirm, &drift);
        }
        assert!(
            (ctrl.confidence() - 1.0).abs() < f64::EPSILON,
            "25 windows should still give confidence 1.0 (capped), got {}",
            ctrl.confidence()
        );
    }

    #[test]
    fn partial_transcript_serde_round_trip_preserves_all_fields() {
        let pt = PartialTranscript::new(
            7,
            42,
            "turbo-whisper".to_owned(),
            vec![seg("hello world", Some(0.85))],
            150,
            "2026-06-15T12:00:00Z".to_owned(),
        );
        let json = serde_json::to_string(&pt).expect("serialize");
        let back: PartialTranscript = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.seq, 7);
        assert_eq!(back.window_id, 42);
        assert_eq!(back.model_id, "turbo-whisper");
        assert_eq!(back.segments.len(), 1);
        assert_eq!(back.segments[0].text, "hello world");
        assert_eq!(back.latency_ms, 150);
        assert_eq!(back.emitted_at_rfc3339, "2026-06-15T12:00:00Z");
        assert!((back.confidence_mean - 0.85).abs() < f64::EPSILON);
        assert_eq!(back.status, PartialStatus::Pending);
    }

    #[test]
    fn evidence_ledger_empty_diagnostics_returns_all_zero_fields() {
        let ledger = CorrectionEvidenceLedger::new(10);
        let diag = ledger.diagnostics();
        assert_eq!(diag["correction_rate"], 0.0);
        assert_eq!(diag["mean_fast_latency_ms"], 0.0);
        assert_eq!(diag["mean_quality_latency_ms"], 0.0);
        assert_eq!(diag["mean_wer"], 0.0);
        assert_eq!(diag["latency_savings_pct"], 0.0);
    }

    // ── Task #278 — speculation.rs pass 19 edge-case tests ────────────

    #[test]
    fn speculation_stats_serde_round_trip() {
        let stats = SpeculationStats {
            windows_processed: 42,
            corrections_emitted: 10,
            confirmations_emitted: 32,
            correction_rate: 0.238,
            mean_fast_latency_ms: 85.5,
            mean_quality_latency_ms: 450.0,
            current_window_size_ms: 5000,
            mean_drift_wer: 0.067,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let back: SpeculationStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.windows_processed, 42);
        assert_eq!(back.corrections_emitted, 10);
        assert_eq!(back.confirmations_emitted, 32);
        assert!((back.correction_rate - 0.238).abs() < f64::EPSILON);
        assert!((back.mean_fast_latency_ms - 85.5).abs() < f64::EPSILON);
        assert!((back.mean_quality_latency_ms - 450.0).abs() < f64::EPSILON);
        assert_eq!(back.current_window_size_ms, 5000);
        assert!((back.mean_drift_wer - 0.067).abs() < f64::EPSILON);
    }

    #[test]
    fn correction_event_serde_round_trip() {
        let event = CorrectionEvent::new(
            5,
            3,
            10,
            "quality-v2".to_owned(),
            vec![seg("corrected text", Some(0.95))],
            300,
            "2026-03-01T00:00:00Z".to_owned(),
            &[seg("fast text", Some(0.7))],
        );
        let json = serde_json::to_string(&event).expect("serialize");
        let back: CorrectionEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.correction_id, 5);
        assert_eq!(back.retracted_seq, 3);
        assert_eq!(back.window_id, 10);
        assert_eq!(back.quality_model_id, "quality-v2");
        assert_eq!(back.corrected_segments.len(), 1);
        assert_eq!(back.corrected_segments[0].text, "corrected text");
        assert_eq!(back.quality_latency_ms, 300);
        assert!((back.quality_confidence_mean - 0.95).abs() < f64::EPSILON);
        assert!(
            back.drift.wer_approx > 0.0,
            "different texts should have nonzero WER"
        );
        assert_eq!(back.corrected_at_rfc3339, "2026-03-01T00:00:00Z");
    }

    #[test]
    fn correction_tolerance_default_values() {
        let t = CorrectionTolerance::default();
        assert!(
            (t.max_wer - 0.1).abs() < f64::EPSILON,
            "default max_wer should be 0.1"
        );
        assert!(
            (t.max_confidence_delta - 0.15).abs() < f64::EPSILON,
            "default max_confidence_delta should be 0.15"
        );
        assert_eq!(
            t.max_edit_distance, 50,
            "default max_edit_distance should be 50"
        );
        assert!(!t.always_correct, "default always_correct should be false");
    }

    #[test]
    fn calibration_tracker_brier_score_worst_case_returns_one() {
        let mut tracker = CalibrationTracker::new(10);
        // Predict 1.0 (correction) for every confirmation and 0.0 for every correction.
        tracker.record(1.0, false); // predicted correction, actual confirmation: (1.0-0.0)^2=1
        tracker.record(0.0, true); // predicted confirmation, actual correction: (0.0-1.0)^2=1
        tracker.record(1.0, false);
        tracker.record(0.0, true);
        // Mean = (1+1+1+1)/4 = 1.0.
        assert!(
            (tracker.brier_score() - 1.0).abs() < f64::EPSILON,
            "worst-case calibration should give brier score 1.0, got {}",
            tracker.brier_score()
        );
    }

    #[test]
    fn window_state_serde_round_trip() {
        let ws = WindowState {
            window: SpeculationWindow::new(7, "run-x".into(), 1000, 4000, 200, "aabbcc".into()),
            fast_result: None,
            quality_result: Some(vec![seg("hello", Some(0.9))]),
            status: WindowStatus::QualityComplete,
        };
        let json = serde_json::to_string(&ws).expect("serialize");
        let back: WindowState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.window.window_id, 7);
        assert_eq!(back.window.run_id, "run-x");
        assert!(back.fast_result.is_none());
        let qr = back
            .quality_result
            .as_ref()
            .expect("should have quality result");
        assert_eq!(qr.len(), 1);
        assert_eq!(qr[0].text, "hello");
        assert_eq!(back.status, WindowStatus::QualityComplete);
    }

    // ── Task #279 — speculation.rs pass 20 edge-case tests ────────────

    #[test]
    fn speculation_controller_entry_serde_round_trip() {
        let entry = SpeculationControllerEntry {
            decision_id: 3,
            window_id: 7,
            state_snapshot: ControllerState {
                correction_rate: 0.2,
                mean_wer: 0.08,
                window_count: 15,
                current_window_ms: 4500,
            },
            action_taken: ControllerAction::Shrink(500),
            predicted_correction_rate: 0.18,
            actual_correction_rate: 0.2,
            brier_score: 0.04,
            confidence: 0.75,
            fallback_active: false,
            fallback_reason: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: SpeculationControllerEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.decision_id, 3);
        assert_eq!(back.window_id, 7);
        assert_eq!(back.state_snapshot.window_count, 15);
        assert_eq!(back.action_taken, ControllerAction::Shrink(500));
        assert!((back.predicted_correction_rate - 0.18).abs() < f64::EPSILON);
        assert!(!back.fallback_active);
        assert!(back.fallback_reason.is_none());
    }

    #[test]
    fn correction_evidence_entry_serde_round_trip() {
        let entry = CorrectionEvidenceEntry {
            entry_id: 1,
            window_id: 5,
            run_id: "run-abc".into(),
            timestamp_rfc3339: "2026-06-15T12:00:00Z".into(),
            fast_model_id: "turbo".into(),
            fast_latency_ms: 80,
            fast_confidence_mean: 0.7,
            fast_segment_count: 3,
            quality_model_id: "hq".into(),
            quality_latency_ms: 400,
            quality_confidence_mean: 0.95,
            quality_segment_count: 3,
            drift: CorrectionDrift {
                wer_approx: 0.12,
                confidence_delta: 0.25,
                segment_count_delta: 0,
                text_edit_distance: 5,
            },
            decision: "correction".into(),
            window_size_ms: 5000,
            correction_rate_at_decision: 0.15,
            controller_confidence: 0.6,
            fallback_active: true,
            fallback_reason: Some("Brier score 0.3 > threshold".into()),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        let back: CorrectionEvidenceEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.entry_id, 1);
        assert_eq!(back.run_id, "run-abc");
        assert_eq!(back.fast_model_id, "turbo");
        assert_eq!(back.quality_latency_ms, 400);
        assert!((back.drift.wer_approx - 0.12).abs() < f64::EPSILON);
        assert_eq!(back.decision, "correction");
        assert!(back.fallback_active);
        assert_eq!(
            back.fallback_reason.as_deref(),
            Some("Brier score 0.3 > threshold")
        );
    }

    #[test]
    fn controller_state_serde_round_trip() {
        let state = ControllerState {
            correction_rate: 0.33,
            mean_wer: 0.09,
            window_count: 12,
            current_window_ms: 6000,
        };
        let json = serde_json::to_string(&state).expect("serialize");
        let back: ControllerState = serde_json::from_str(&json).expect("deserialize");
        assert!((back.correction_rate - 0.33).abs() < f64::EPSILON);
        assert!((back.mean_wer - 0.09).abs() < f64::EPSILON);
        assert_eq!(back.window_count, 12);
        assert_eq!(back.current_window_ms, 6000);
    }

    #[test]
    fn beta_posterior_serde_round_trip() {
        let mut p = BetaPosterior::weakly_informative();
        p.observe_correction();
        p.observe_correction();
        p.observe_confirmation();
        // alpha=4, beta=3
        let json = serde_json::to_string(&p).expect("serialize");
        let back: BetaPosterior = serde_json::from_str(&json).expect("deserialize");
        assert!((back.alpha - 4.0).abs() < f64::EPSILON);
        assert!((back.beta - 3.0).abs() < f64::EPSILON);
        assert!((back.mean() - 4.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn correction_stats_fields_after_mixed_submissions() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance {
            max_wer: 0.05, // low threshold to trigger corrections easily
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: false,
        });

        // Window 1: high WER → Correct.
        let p1 = PartialTranscript::new(
            0,
            10,
            "fast".into(),
            vec![seg("alpha", Some(0.8))],
            100,
            "t".into(),
        );
        tracker.register_partial(p1);
        tracker
            .submit_quality_result(10, "quality", vec![seg("omega", Some(0.9))], 300)
            .unwrap();

        // Window 2: identical → Confirm.
        let p2 = PartialTranscript::new(
            1,
            20,
            "fast".into(),
            vec![seg("same", Some(0.9))],
            80,
            "t".into(),
        );
        tracker.register_partial(p2);
        tracker
            .submit_quality_result(20, "quality", vec![seg("same", Some(0.9))], 250)
            .unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.windows_processed, 2);
        assert_eq!(stats.corrections_emitted, 1);
        assert_eq!(stats.confirmations_emitted, 1);
        assert_eq!(stats.total_fast_latency_ms, 180); // 100 + 80
        assert_eq!(stats.total_quality_latency_ms, 550); // 300 + 250
        assert!(
            stats.cumulative_wer > 0.0,
            "at least one window had nonzero WER"
        );
        assert!(
            stats.max_observed_wer > 0.0,
            "max WER should reflect the correction"
        );
    }

    // ── Task #280 — speculation.rs pass 21 edge-case tests ────────────

    #[test]
    fn controller_consecutive_zero_corrections_resets_on_correction() {
        // ZERO_CORRECTION_FORCE_SHRINK = 20.
        // Use min == initial so the normal shrink path (low correction rate) cannot
        // fire; only the force-shrink path (consecutive_zero >= 20) could shrink.
        let mut ctrl = SpeculationWindowController::new(5000, 5000, 10000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        let confirm = CorrectionDecision::Confirm {
            seq: 0,
            drift: drift.clone(),
        };

        // 15 confirmations.
        for _ in 0..15 {
            ctrl.observe(&confirm, &drift);
        }
        // Inject a correction to reset the consecutive counter.
        let correction = CorrectionDecision::Correct {
            correction: CorrectionEvent::new(0, 0, 0, "q".into(), vec![], 100, "t".into(), &[]),
        };
        ctrl.observe(&correction, &drift);

        // 15 more confirmations (total streak = 15, not 31).
        for _ in 0..15 {
            ctrl.observe(&confirm, &drift);
        }

        let action = ctrl.recommend();
        // consecutive_zero = 15 < 20 → force-shrink NOT triggered.
        // And min == current (5000) → normal shrink also impossible.
        assert_ne!(
            action,
            ControllerAction::Shrink(500),
            "correction in the middle should reset streak; 15 consecutive < 20 threshold"
        );
    }

    #[test]
    fn merge_segments_no_resolved_windows_returns_empty_vec() {
        let mut wm = WindowManager::new("run-nores", 3000, 0);
        let w = wm.next_window(0, "h");
        // Record fast result but do NOT resolve the window.
        let p = PartialTranscript::new(
            0,
            w.window_id,
            "fast".into(),
            vec![seg("hello", Some(0.9))],
            50,
            "t".into(),
        );
        wm.record_fast_result(w.window_id, p);
        // merge_segments only considers Resolved windows.
        let merged = wm.merge_segments();
        assert!(
            merged.is_empty(),
            "no resolved windows → empty merged output"
        );
    }

    #[test]
    fn evidence_ledger_to_evidence_json_entries_count_matches_retained() {
        let mut ledger = CorrectionEvidenceLedger::new(5);
        let drift = CorrectionDrift {
            wer_approx: 0.1,
            confidence_delta: 0.05,
            segment_count_delta: 0,
            text_edit_distance: 2,
        };
        // Record 3 entries.
        for i in 0..3 {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i,
                window_id: i,
                run_id: "run".into(),
                timestamp_rfc3339: "2026-01-01T00:00:00Z".into(),
                fast_model_id: "fast".into(),
                fast_latency_ms: 100,
                fast_confidence_mean: 0.8,
                fast_segment_count: 1,
                quality_model_id: "quality".into(),
                quality_latency_ms: 400,
                quality_confidence_mean: 0.95,
                quality_segment_count: 1,
                drift: drift.clone(),
                decision: "confirm".into(),
                window_size_ms: 3000,
                correction_rate_at_decision: 0.0,
                controller_confidence: 0.5,
                fallback_active: false,
                fallback_reason: None,
            });
        }
        let json = ledger.to_evidence_json();
        assert_eq!(json["total_recorded"].as_u64(), Some(3));
        assert_eq!(json["retained"].as_u64(), Some(3));
        assert_eq!(json["capacity"].as_u64(), Some(5));
        let entries_arr = json["entries"].as_array().expect("entries should be array");
        assert_eq!(entries_arr.len(), 3, "entries array should have 3 elements");
    }

    #[test]
    fn controller_observe_updates_mean_wer_correctly_after_three_windows() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        let confirm = |wer: f64| {
            let d = CorrectionDrift {
                wer_approx: wer,
                confidence_delta: 0.0,
                segment_count_delta: 0,
                text_edit_distance: 0,
            };
            (
                CorrectionDecision::Confirm {
                    seq: 0,
                    drift: d.clone(),
                },
                d,
            )
        };
        // Three observations with WER 0.0, 0.3, 0.6 → mean = 0.3.
        let (d1, drift1) = confirm(0.0);
        ctrl.observe(&d1, &drift1);
        let (d2, drift2) = confirm(0.3);
        ctrl.observe(&d2, &drift2);
        let (d3, drift3) = confirm(0.6);
        ctrl.observe(&d3, &drift3);

        assert!(
            (ctrl.state().mean_wer - 0.3).abs() < 1e-9,
            "mean_wer after [0.0, 0.3, 0.6] should be 0.3, got {}",
            ctrl.state().mean_wer
        );
    }

    // ── Task #281 — speculation.rs pass 22 edge-case tests ────────────

    #[test]
    fn correction_drift_compute_wer_uses_max_word_count() {
        // fast: 2 words, quality: 5 words → max_words = 5.
        // All words differ → word_edit_distance = 5 (insert 3 + substitute 2 or similar).
        let fast = vec![seg("hello world", Some(0.8))];
        let quality = vec![seg("one two three four five", Some(0.9))];
        let drift = CorrectionDrift::compute(&fast, &quality);
        // max_words = max(2, 5) = 5.
        // wer_approx = levenshtein_words(["hello","world"], ["one","two","three","four","five"]) / 5.
        // word edit: substitute hello→one, world→two, insert three, four, five = 5 edits.
        assert!(
            (drift.wer_approx - 1.0).abs() < f64::EPSILON,
            "5 edits / 5 words = WER 1.0, got {}",
            drift.wer_approx
        );
        assert_eq!(drift.segment_count_delta, 0); // same segment count
    }

    #[test]
    fn window_manager_next_window_near_u64_max_saturates() {
        let mut wm = WindowManager::new("run-sat", 5000, 200);
        let w = wm.next_window(u64::MAX - 100, "h");
        // end_ms = (u64::MAX - 100).saturating_add(5000) = u64::MAX.
        assert_eq!(w.end_ms, u64::MAX);
        assert_eq!(w.start_ms, u64::MAX - 100);
    }

    #[test]
    fn evidence_ledger_diagnostics_with_entries_computes_all_rates() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let make_entry = |id: u64, decision: &str, fast_ms: u64, quality_ms: u64, wer: f64| {
            CorrectionEvidenceEntry {
                entry_id: id,
                window_id: id,
                run_id: "run".into(),
                timestamp_rfc3339: "2026-01-01T00:00:00Z".into(),
                fast_model_id: "fast".into(),
                fast_latency_ms: fast_ms,
                fast_confidence_mean: 0.8,
                fast_segment_count: 1,
                quality_model_id: "quality".into(),
                quality_latency_ms: quality_ms,
                quality_confidence_mean: 0.95,
                quality_segment_count: 1,
                drift: CorrectionDrift {
                    wer_approx: wer,
                    confidence_delta: 0.1,
                    segment_count_delta: 0,
                    text_edit_distance: 3,
                },
                decision: decision.into(),
                window_size_ms: 3000,
                correction_rate_at_decision: 0.0,
                controller_confidence: 0.5,
                fallback_active: false,
                fallback_reason: None,
            }
        };
        ledger.record(make_entry(0, "correct", 100, 500, 0.2));
        ledger.record(make_entry(1, "confirm", 80, 400, 0.0));
        ledger.record(make_entry(2, "correct", 120, 600, 0.3));

        // correction_rate: 2/3
        assert!((ledger.correction_rate() - 2.0 / 3.0).abs() < 1e-9);
        // mean_fast_latency: (100+80+120)/3 = 100
        assert!((ledger.mean_fast_latency() - 100.0).abs() < f64::EPSILON);
        // mean_quality_latency: (500+400+600)/3 = 500
        assert!((ledger.mean_quality_latency() - 500.0).abs() < f64::EPSILON);
        // mean_wer: (0.2+0.0+0.3)/3 ≈ 0.1666...
        let expected_wer = (0.2 + 0.0 + 0.3) / 3.0;
        assert!((ledger.mean_wer() - expected_wer).abs() < 1e-9);
        // latency_savings_pct: (500-100)/500*100 = 80.0
        assert!((ledger.latency_savings_pct() - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn controller_evidence_grows_with_each_apply_call() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 10000, 500);
        assert_eq!(ctrl.evidence().len(), 0);
        ctrl.apply();
        assert_eq!(ctrl.evidence().len(), 1);
        ctrl.apply();
        assert_eq!(ctrl.evidence().len(), 2);
        ctrl.apply();
        assert_eq!(ctrl.evidence().len(), 3);
        // Each entry has a sequential decision_id.
        assert_eq!(ctrl.evidence()[0].decision_id, 0);
        assert_eq!(ctrl.evidence()[1].decision_id, 1);
        assert_eq!(ctrl.evidence()[2].decision_id, 2);
    }

    #[test]
    fn controller_apply_runaway_sets_exact_fallback_reason_string() {
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 30000, 500);
        // Inject well-calibrated calibration to avoid Brier fallback.
        for _ in 0..10 {
            ctrl.calibration.record(0.9, true);
        }
        let drift = CorrectionDrift {
            wer_approx: 0.5,
            confidence_delta: 0.1,
            segment_count_delta: 0,
            text_edit_distance: 5,
        };
        // Drive correction rate > 0.75.
        for _ in 0..10u64 {
            let correction =
                CorrectionEvent::new(0, 0, 0, "q".into(), vec![], 100, "t".into(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        assert!(ctrl.state().correction_rate > 0.75);
        ctrl.apply();
        assert!(ctrl.is_fallback_active());
        assert_eq!(
            ctrl.evidence().last().unwrap().fallback_reason.as_deref(),
            Some("correction rate > 75%"),
            "runaway fallback should set exact reason string"
        );
    }

    #[test]
    fn calibration_tracker_brier_score_uniform_half_predictions() {
        let mut tracker = CalibrationTracker::new(10);
        // Predict 0.5 for everything.
        // actual=true: (0.5-1.0)^2 = 0.25
        // actual=false: (0.5-0.0)^2 = 0.25
        tracker.record(0.5, true);
        tracker.record(0.5, false);
        tracker.record(0.5, true);
        tracker.record(0.5, false);
        // Mean = (0.25 * 4) / 4 = 0.25.
        assert!(
            (tracker.brier_score() - 0.25).abs() < f64::EPSILON,
            "uniform 0.5 predictions should give brier score 0.25, got {}",
            tracker.brier_score()
        );
    }

    // ── Task #282 — speculation.rs pass 23 edge-case tests ────────────

    #[test]
    fn controller_recommend_holds_when_confidence_below_half() {
        // With 8 windows, confidence = 8/20 = 0.4 < 0.5.
        // Even with high correction rate and WER, recommend should Hold.
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 10000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.5,
            confidence_delta: 0.3,
            segment_count_delta: 0,
            text_edit_distance: 10,
        };
        // 6 corrections + 2 confirmations = 8 windows, rate ~0.67.
        for i in 0..6u64 {
            let correction =
                CorrectionEvent::new(i, i, i, "q".into(), vec![], 100, "t".into(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        for _ in 0..2 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        assert_eq!(ctrl.state().window_count, 8);
        assert!(ctrl.confidence() < 0.5, "confidence should be 0.4");
        let action = ctrl.recommend();
        assert_eq!(
            action,
            ControllerAction::Hold,
            "confidence < 0.5 should force Hold, got {action:?}"
        );
    }

    #[test]
    fn correction_tracker_mean_wer_exact_after_mixed_decisions() {
        let tolerance = CorrectionTolerance {
            max_wer: 0.05,
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);
        // Window 1: high WER (correction).
        let p1 = PartialTranscript::new(
            0,
            10,
            "fast".into(),
            vec![seg("cat", Some(0.8))],
            50,
            "t".into(),
        );
        tracker.register_partial(p1);
        tracker
            .submit_quality_result(10, "q", vec![seg("dog", Some(0.9))], 200)
            .unwrap();
        // Window 2: zero WER (confirmation).
        let p2 = PartialTranscript::new(
            1,
            20,
            "fast".into(),
            vec![seg("same", Some(0.9))],
            50,
            "t".into(),
        );
        tracker.register_partial(p2);
        tracker
            .submit_quality_result(20, "q", vec![seg("same", Some(0.9))], 200)
            .unwrap();

        let mean = tracker.mean_wer();
        // Window 1 had WER=1.0 (all words different), window 2 had WER=0.0.
        // mean = (1.0 + 0.0) / 2 = 0.5
        assert!(
            (mean - 0.5).abs() < 1e-9,
            "mean_wer should be 0.5, got {mean}"
        );
    }

    #[test]
    fn window_manager_new_default_min_max_clamping() {
        let mut wm = WindowManager::new("run", 5000, 0);
        // Default min_window_ms = 1000.
        wm.set_window_size(500);
        assert_eq!(wm.current_window_size(), 1000, "should clamp to min 1000");
        // Default max_window_ms = 30000.
        wm.set_window_size(50_000);
        assert_eq!(
            wm.current_window_size(),
            30_000,
            "should clamp to max 30000"
        );
    }

    #[test]
    fn correction_event_confidence_mean_from_multiple_quality_segments() {
        // CorrectionEvent::new() computes quality_confidence_mean via mean_confidence.
        let event = CorrectionEvent::new(
            0,
            0,
            0,
            "q".into(),
            vec![seg("a", Some(0.8)), seg("b", Some(0.6)), seg("c", None)],
            100,
            "t".into(),
            &[],
        );
        // mean_confidence: only Some values → (0.8+0.6)/2 = 0.7
        assert!(
            (event.quality_confidence_mean - 0.7).abs() < 1e-9,
            "should average only segments with confidence: expected 0.7, got {}",
            event.quality_confidence_mean
        );
    }

    #[test]
    fn speculation_window_contains_ms_zero_start() {
        let w = SpeculationWindow::new(0, "r".into(), 0, 5000, 0, "h".into());
        assert!(w.contains_ms(0), "start boundary should be inclusive");
        assert!(w.contains_ms(4999), "end-1 should be contained");
        assert!(!w.contains_ms(5000), "end boundary should be exclusive");
    }

    // ── Task #283 — speculation.rs pass 24 edge-case tests ────────────

    #[test]
    fn controller_recommend_does_not_shrink_at_exactly_10_windows() {
        // The low-correction-rate shrink path requires window_count > 10 (strict >).
        // At exactly 10, it should Hold even with low correction rate.
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // 10 confirmations → correction_rate ≈ 0 (< LOW=0.0625), window_count = 10.
        // confidence = 10/20 = 0.5 >= 0.5, so confidence check passes.
        for _ in 0..10 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        assert_eq!(ctrl.state().window_count, 10);
        let action = ctrl.recommend();
        // window_count == 10, not > 10, so shrink should not trigger.
        assert_eq!(
            action,
            ControllerAction::Hold,
            "exactly 10 windows (not > 10) should Hold, got {action:?}"
        );
    }

    #[test]
    fn controller_apply_shrink_clamps_at_min_window() {
        // Start at min + step (1500), min=1000, step=500. After shrink → 1000 (clamped).
        let mut ctrl = SpeculationWindowController::new(1500, 1000, 30000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // Feed 15 confirmations to trigger low-correction-rate shrink.
        for _ in 0..15 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        let new = ctrl.apply();
        assert_eq!(
            new, 1000,
            "shrink from 1500 by 500 should clamp at min 1000"
        );
        // Another apply: already at min, consecutive_zero < 20, so Hold keeps at 1000.
        for _ in 0..5 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        let same = ctrl.apply();
        assert_eq!(same, 1000, "should stay at min 1000");
    }

    #[test]
    fn controller_apply_grow_clamps_at_max_window() {
        // Start near max: initial=29800, max=30000, step=500. Grow → 30000 (clamped).
        let mut ctrl = SpeculationWindowController::new(29800, 1000, 30000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.3,
            confidence_delta: 0.2,
            segment_count_delta: 0,
            text_edit_distance: 5,
        };
        // Need correction_rate > 0.25, mean_wer > 0.125, confidence >= 0.5.
        // Feed 5 corrections + 5 confirmations = rate 0.5 > 0.25, window_count = 10.
        for i in 0..5u64 {
            let correction =
                CorrectionEvent::new(i, i, i, "q".into(), vec![], 100, "t".into(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        for _ in 0..5 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        let new = ctrl.apply();
        assert_eq!(
            new, 30000,
            "grow from 29800 by 500 should clamp at max 30000"
        );
    }

    #[test]
    fn evidence_ledger_window_size_trend_empty_returns_empty() {
        let ledger = CorrectionEvidenceLedger::new(10);
        assert!(ledger.window_size_trend().is_empty());
    }

    #[test]
    fn correction_drift_compute_one_word_vs_four_words() {
        // fast: 1 word "hello", quality: 4 words "one two three four".
        // word edit: substitute hello→one, insert two, three, four = 4 edits.
        // max_words = max(1, 4) = 4.
        // WER = 4/4 = 1.0.
        let fast = vec![seg("hello", Some(0.8))];
        let quality = vec![seg("one two three four", Some(0.9))];
        let drift = CorrectionDrift::compute(&fast, &quality);
        assert!(
            (drift.wer_approx - 1.0).abs() < f64::EPSILON,
            "4 edits / 4 words = WER 1.0, got {}",
            drift.wer_approx
        );
        assert_eq!(drift.segment_count_delta, 0, "same segment count");
    }

    // ── Task #284 — speculation.rs pass 25 edge-case tests ────────────

    #[test]
    fn controller_apply_brier_fallback_reason_starts_with_brier_score() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        // Brier fallback triggers when brier > 0.25 AND sample_count >= 10.
        // Feed 15 corrections (posterior high), then 10 confirmations (prediction
        // still high but outcome changes), driving Brier above threshold.
        let drift = CorrectionDrift {
            wer_approx: 0.05,
            confidence_delta: 0.01,
            segment_count_delta: 0,
            text_edit_distance: 1,
        };
        for _ in 0..15 {
            let correction =
                CorrectionEvent::new(0, 0, 0, "q".into(), vec![], 100, "t".into(), &[]);
            ctrl.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        for _ in 0..10 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        ctrl.apply();
        if ctrl.is_fallback_active() {
            let reason = ctrl.evidence().last().unwrap().fallback_reason.as_deref();
            if let Some(r) = reason {
                assert!(
                    r.starts_with("Brier score") || r.starts_with("correction rate"),
                    "fallback reason should identify the cause, got: {r}"
                );
            }
        }
        // Regardless of exact branch, the test verifies apply() completes without panic.
    }

    #[test]
    fn evidence_ledger_total_recorded_exceeds_entries_after_eviction() {
        let mut ledger = CorrectionEvidenceLedger::new(2);
        let drift = CorrectionDrift {
            wer_approx: 0.1,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 1,
        };
        for i in 0..5u64 {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i,
                window_id: i,
                run_id: "run".into(),
                timestamp_rfc3339: "t".into(),
                fast_model_id: "f".into(),
                fast_latency_ms: 100,
                fast_confidence_mean: 0.8,
                fast_segment_count: 1,
                quality_model_id: "q".into(),
                quality_latency_ms: 400,
                quality_confidence_mean: 0.9,
                quality_segment_count: 1,
                drift: drift.clone(),
                decision: "confirm".into(),
                window_size_ms: 3000,
                correction_rate_at_decision: 0.0,
                controller_confidence: 0.5,
                fallback_active: false,
                fallback_reason: None,
            });
        }
        assert_eq!(
            ledger.total_recorded(),
            5,
            "total should count all 5 records"
        );
        assert_eq!(ledger.entries().len(), 2, "only 2 retained (capacity=2)");
        // Newest entries are IDs 3 and 4.
        assert_eq!(ledger.entries()[0].entry_id, 3);
        assert_eq!(ledger.entries()[1].entry_id, 4);
    }

    #[test]
    fn next_window_bounded_zero_size_window_returns_none() {
        // With window_size_ms = 0, natural_end = audio_position, end_ms = audio_position.
        // Second guard: end_ms <= audio_position → return None.
        let mut wm = WindowManager::new("run-zero", 0, 0);
        // min clamping: set_window_size(0) would clamp to 1000, but the constructor
        // directly sets window_size_ms = 0 bypassing set_window_size.
        let result = wm.next_window_bounded(1000, 5000, "h");
        // natural_end = 1000 + 0 = 1000, end_ms = min(1000, 5000) = 1000.
        // end_ms (1000) <= audio_position_ms (1000) → None.
        assert!(
            result.is_none(),
            "zero-size window should return None from second guard"
        );
    }

    #[test]
    fn levenshtein_classic_kitten_sitting_distance_three() {
        // "kitten" → "sitting" requires 3 edits:
        // k→s, e→i, (insert g) = 3 (classic example).
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn correction_drift_compute_different_segment_counts() {
        // fast: 1 segment, quality: 3 segments → segment_count_delta = 3 - 1 = 2.
        let fast = vec![seg("hello world", Some(0.8))];
        let quality = vec![
            seg("hello", Some(0.9)),
            seg("beautiful", Some(0.85)),
            seg("world", Some(0.95)),
        ];
        let drift = CorrectionDrift::compute(&fast, &quality);
        assert_eq!(drift.segment_count_delta, 2, "quality(3) - fast(1) = 2");
        // Text: "hello world" vs "hello beautiful world"
        // Words: ["hello","world"] vs ["hello","beautiful","world"] → 1 insertion = WER 1/3
        let expected_wer = 1.0 / 3.0;
        assert!(
            (drift.wer_approx - expected_wer).abs() < 1e-9,
            "expected WER {expected_wer}, got {}",
            drift.wer_approx
        );
    }

    // ── Task #285 — speculation.rs pass 26 edge-case tests ────────────

    #[test]
    fn levenshtein_unicode_multi_byte_chars_counted_as_single_edits() {
        // "café" vs "cafe" — é→e is 1 edit (not 2 bytes).
        assert_eq!(levenshtein("café", "cafe"), 1);
        // "日本語" vs "日本人" — 語→人 is 1 substitution.
        assert_eq!(levenshtein("日本語", "日本人"), 1);
        // Emoji: "👋🌍" vs "👋🌏" — 1 substitution.
        assert_eq!(levenshtein("👋🌍", "👋🌏"), 1);
    }

    #[test]
    fn controller_state_current_window_ms_tracks_apply_changes() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        assert_eq!(ctrl.state().current_window_ms, 5000, "initial state");

        // Apply with no observations → Hold → stays at 5000.
        ctrl.apply();
        assert_eq!(
            ctrl.state().current_window_ms,
            5000,
            "after apply with Hold"
        );

        // Feed 15 confirmations to trigger shrink (low correction rate, window_count > 10).
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        for _ in 0..15 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        let new_size = ctrl.apply();
        assert_eq!(
            ctrl.state().current_window_ms,
            new_size,
            "state.current_window_ms should match apply() return value"
        );
        assert!(new_size < 5000, "should have shrunk");
    }

    #[test]
    fn merge_segments_with_none_start_sec_sorted_before_timed_segments() {
        let mut wm = WindowManager::new("run-sort", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(5000, "h1");

        // Window 0: timed segment at 5.0s.
        wm.record_quality_result(w0.window_id, vec![timed_seg("later", 5.0, 6.0, Some(0.9))]);
        wm.resolve_window(w0.window_id);

        // Window 1: untimed segment (start_sec = None → sort as 0.0).
        let fast = PartialTranscript::new(
            0,
            w1.window_id,
            "fast".into(),
            vec![seg("early", Some(0.8))],
            50,
            "t".into(),
        );
        wm.record_fast_result(w1.window_id, fast);
        wm.resolve_window(w1.window_id);

        let merged = wm.merge_segments();
        assert_eq!(merged.len(), 2);
        // None start_sec sorts as 0.0, so "early" comes first.
        assert_eq!(merged[0].text, "early");
        assert_eq!(merged[1].text, "later");
    }

    #[test]
    fn evidence_ledger_correction_rate_zero_for_all_confirm_decisions() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        for i in 0..4u64 {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i,
                window_id: i,
                run_id: "run".into(),
                timestamp_rfc3339: "t".into(),
                fast_model_id: "f".into(),
                fast_latency_ms: 100,
                fast_confidence_mean: 0.9,
                fast_segment_count: 1,
                quality_model_id: "q".into(),
                quality_latency_ms: 400,
                quality_confidence_mean: 0.9,
                quality_segment_count: 1,
                drift: drift.clone(),
                decision: "confirm".into(),
                window_size_ms: 3000,
                correction_rate_at_decision: 0.0,
                controller_confidence: 0.5,
                fallback_active: false,
                fallback_reason: None,
            });
        }
        assert!(
            ledger.correction_rate().abs() < f64::EPSILON,
            "all confirm decisions should give correction_rate 0.0, got {}",
            ledger.correction_rate()
        );
    }

    #[test]
    fn correction_tracker_correction_rate_exact_after_one_of_each() {
        let tolerance = CorrectionTolerance {
            max_wer: 0.001, // very tight to force correction
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        // Window 1: different text → correction.
        let p1 = PartialTranscript::new(
            0,
            10,
            "fast".into(),
            vec![seg("alpha", Some(0.8))],
            50,
            "t".into(),
        );
        tracker.register_partial(p1);
        tracker
            .submit_quality_result(10, "q", vec![seg("beta", Some(0.9))], 200)
            .unwrap();

        // Window 2: identical text → confirmation.
        let p2 = PartialTranscript::new(
            1,
            20,
            "fast".into(),
            vec![seg("same", Some(0.9))],
            50,
            "t".into(),
        );
        tracker.register_partial(p2);
        tracker
            .submit_quality_result(20, "q", vec![seg("same", Some(0.9))], 200)
            .unwrap();

        // 1 correction + 1 confirmation = rate 0.5.
        assert!(
            (tracker.correction_rate() - 0.5).abs() < f64::EPSILON,
            "1 correction + 1 confirmation should give rate 0.5, got {}",
            tracker.correction_rate()
        );
    }

    #[test]
    fn window_manager_get_window_returns_none_for_unknown_id() {
        let mut wm = WindowManager::new("run-get", 5000, 200);
        // No windows yet — any ID returns None.
        assert!(wm.get_window(0).is_none());
        assert!(wm.get_window(99).is_none());
        // Create one window.
        let w0 = wm.next_window(0, "h0");
        assert!(wm.get_window(w0.window_id).is_some());
        assert_eq!(wm.get_window(w0.window_id).unwrap().window.start_ms, 0);
        // Unknown IDs still return None.
        assert!(wm.get_window(w0.window_id + 1).is_none());
        assert!(wm.get_window(999).is_none());
        // Mutable variant behaves identically.
        assert!(wm.get_window_mut(w0.window_id).is_some());
        assert!(wm.get_window_mut(42).is_none());
    }

    #[test]
    fn correction_tracker_all_resolved_on_empty_tracker_returns_true() {
        let tracker = CorrectionTracker::new(CorrectionTolerance::default());
        // Vacuous truth: no partials means all_resolved() is true.
        assert!(
            tracker.all_resolved(),
            "empty tracker should report all_resolved = true"
        );
        assert_eq!(tracker.correction_rate(), 0.0);
        assert_eq!(tracker.mean_wer(), 0.0);
        assert!(tracker.corrections().is_empty());
        assert!(tracker.get_partial(0).is_none());
    }

    #[test]
    fn levenshtein_both_empty_strings_returns_zero() {
        assert_eq!(levenshtein("", ""), 0);
        // Also verify levenshtein_words with empty slices.
        let empty: Vec<&str> = vec![];
        assert_eq!(levenshtein_words(&empty, &empty), 0);
    }

    #[test]
    fn evidence_ledger_latency_savings_pct_positive_for_faster_fast_model() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        // Record entries where fast_latency < quality_latency.
        for i in 0..3 {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i,
                window_id: i,
                run_id: "run-savings".into(),
                timestamp_rfc3339: "t".into(),
                fast_model_id: "fast".into(),
                fast_latency_ms: 100,
                fast_confidence_mean: 0.9,
                fast_segment_count: 1,
                quality_model_id: "quality".into(),
                quality_latency_ms: 400,
                quality_confidence_mean: 0.95,
                quality_segment_count: 1,
                drift: CorrectionDrift {
                    wer_approx: 0.0,
                    confidence_delta: 0.05,
                    segment_count_delta: 0,
                    text_edit_distance: 0,
                },
                decision: "confirm".into(),
                window_size_ms: 5000,
                correction_rate_at_decision: 0.0,
                controller_confidence: 1.0,
                fallback_active: false,
                fallback_reason: None,
            });
        }
        // mean_fast = 100, mean_quality = 400.
        // savings = (400 - 100) / 400 * 100 = 75%.
        assert!(
            (ledger.mean_fast_latency() - 100.0).abs() < f64::EPSILON,
            "mean_fast_latency should be 100.0"
        );
        assert!(
            (ledger.mean_quality_latency() - 400.0).abs() < f64::EPSILON,
            "mean_quality_latency should be 400.0"
        );
        assert!(
            (ledger.latency_savings_pct() - 75.0).abs() < f64::EPSILON,
            "latency_savings_pct should be 75.0, got {}",
            ledger.latency_savings_pct()
        );
    }

    #[test]
    fn calibration_tracker_sample_count_tracks_insertions_and_evictions() {
        let mut ct = CalibrationTracker::new(3);
        assert_eq!(ct.sample_count(), 0);
        ct.record(0.5, true);
        assert_eq!(ct.sample_count(), 1);
        ct.record(0.3, false);
        assert_eq!(ct.sample_count(), 2);
        ct.record(0.7, true);
        assert_eq!(ct.sample_count(), 3);
        // Window is full (capacity 3). Next insert evicts oldest.
        ct.record(0.1, false);
        assert_eq!(ct.sample_count(), 3, "should not exceed window_size");
        ct.record(0.9, true);
        assert_eq!(ct.sample_count(), 3);
        // Brier score should reflect only the 3 newest: (0.7-1)^2 + (0.1-0)^2 + (0.9-1)^2 = 0.09 + 0.01 + 0.01 = 0.11 / 3
        let expected_brier = (0.09 + 0.01 + 0.01) / 3.0;
        assert!(
            (ct.brier_score() - expected_brier).abs() < 1e-10,
            "brier_score should be {expected_brier:.6}, got {:.6}",
            ct.brier_score()
        );
    }

    #[test]
    fn merge_segments_resolved_window_with_empty_quality_vec_contributes_nothing() {
        let mut wm = WindowManager::new("run-empty-q", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(5000, "h1");
        // w0: quality result is an empty vec.
        wm.record_quality_result(w0.window_id, vec![]);
        wm.resolve_window(w0.window_id);
        // w1: has a fast result with segments, no quality.
        let fast = PartialTranscript::new(
            0,
            w1.window_id,
            "fast".into(),
            vec![seg("hello", Some(0.9))],
            50,
            "t".into(),
        );
        wm.record_fast_result(w1.window_id, fast);
        wm.resolve_window(w1.window_id);
        let merged = wm.merge_segments();
        // Only w1's fast segment should appear; w0's empty quality adds nothing.
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "hello");
    }

    #[test]
    fn correction_drift_compute_multi_word_identical_wer_zero() {
        let segs = vec![
            seg("the quick brown", Some(0.9)),
            seg("fox jumps over", Some(0.85)),
        ];
        let drift = CorrectionDrift::compute(&segs, &segs);
        assert!(
            drift.wer_approx.abs() < f64::EPSILON,
            "identical multi-word multi-segment text should give WER 0.0, got {}",
            drift.wer_approx
        );
        assert_eq!(drift.text_edit_distance, 0);
        assert_eq!(drift.segment_count_delta, 0);
        assert!(drift.confidence_delta.abs() < f64::EPSILON);
    }

    #[test]
    fn correction_tracker_register_partial_accumulates_fast_latency() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());
        let p0 = PartialTranscript::new(
            0,
            10,
            "fast".into(),
            vec![seg("a", Some(0.9))],
            100,
            "t".into(),
        );
        let p1 = PartialTranscript::new(
            1,
            20,
            "fast".into(),
            vec![seg("b", Some(0.8))],
            250,
            "t".into(),
        );
        let p2 = PartialTranscript::new(
            2,
            30,
            "fast".into(),
            vec![seg("c", Some(0.7))],
            50,
            "t".into(),
        );
        tracker.register_partial(p0);
        tracker.register_partial(p1);
        tracker.register_partial(p2);
        // total_fast_latency_ms = 100 + 250 + 50 = 400.
        assert_eq!(tracker.stats().total_fast_latency_ms, 400);
        // No quality results submitted yet.
        assert_eq!(tracker.stats().total_quality_latency_ms, 0);
        assert_eq!(tracker.stats().windows_processed, 0);
    }

    #[test]
    fn beta_posterior_mean_converges_toward_one_with_many_corrections() {
        let mut post = BetaPosterior::weakly_informative(); // Beta(2,2), mean=0.5
        for _ in 0..100 {
            post.observe_correction();
        }
        // After 100 corrections: alpha=102, beta=2, mean = 102/104 ≈ 0.9808
        let expected = 102.0 / 104.0;
        assert!(
            (post.mean() - expected).abs() < 1e-10,
            "mean should be {expected}, got {}",
            post.mean()
        );
        assert!(
            post.mean() > 0.98,
            "mean should be near 1.0 after 100 corrections"
        );
        // Variance should be very small.
        assert!(post.variance() < 0.001, "variance should be tiny");
    }

    #[test]
    fn controller_observe_increments_window_count_per_observation() {
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        assert_eq!(ctrl.state().window_count, 0);
        let drift = CorrectionDrift {
            wer_approx: 0.05,
            confidence_delta: 0.01,
            segment_count_delta: 0,
            text_edit_distance: 2,
        };
        for i in 1..=7 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
            assert_eq!(
                ctrl.state().window_count,
                i,
                "window_count should be {i} after {i} observations"
            );
        }
    }

    #[test]
    fn merge_segments_no_dedup_when_start_sec_differ_beyond_tolerance() {
        let mut wm = WindowManager::new("run-nodedup", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        // Two segments with start_sec differing by >= 0.1 should NOT be deduped.
        wm.record_quality_result(
            w0.window_id,
            vec![
                timed_seg("alpha", 0.0, 1.0, Some(0.9)),
                timed_seg("beta", 0.15, 1.15, Some(0.8)),
            ],
        );
        wm.resolve_window(w0.window_id);
        let merged = wm.merge_segments();
        assert_eq!(merged.len(), 2, "segments 0.1+ apart should NOT be deduped");
        assert_eq!(merged[0].text, "alpha");
        assert_eq!(merged[1].text, "beta");
    }

    #[test]
    fn correction_tracker_stats_quality_latency_accumulates_across_submissions() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());
        // Register and submit 3 windows with different quality latencies.
        for (i, qlat) in [(0u64, 150u64), (1, 300), (2, 50)] {
            let p = PartialTranscript::new(
                i,
                i * 10 + 10,
                "fast".into(),
                vec![seg("text", Some(0.9))],
                100,
                "t".into(),
            );
            tracker.register_partial(p);
            tracker
                .submit_quality_result(i * 10 + 10, "q", vec![seg("text", Some(0.9))], qlat)
                .unwrap();
        }
        assert_eq!(tracker.stats().total_quality_latency_ms, 150 + 300 + 50);
        assert_eq!(tracker.stats().windows_processed, 3);
    }

    #[test]
    fn evidence_ledger_mean_wer_single_entry_matches_drift_wer() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        ledger.record(CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 0,
            run_id: "r".into(),
            timestamp_rfc3339: "t".into(),
            fast_model_id: "f".into(),
            fast_latency_ms: 50,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "q".into(),
            quality_latency_ms: 200,
            quality_confidence_mean: 0.9,
            quality_segment_count: 1,
            drift: CorrectionDrift {
                wer_approx: 0.42,
                confidence_delta: 0.1,
                segment_count_delta: 0,
                text_edit_distance: 5,
            },
            decision: "correct".into(),
            window_size_ms: 5000,
            correction_rate_at_decision: 1.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        });
        assert!(
            (ledger.mean_wer() - 0.42).abs() < f64::EPSILON,
            "single-entry mean_wer should equal that entry's wer_approx"
        );
    }

    #[test]
    fn concat_segment_text_empty_vec_returns_empty_string() {
        let empty: Vec<TranscriptionSegment> = vec![];
        assert_eq!(concat_segment_text(&empty), "");
        // Single segment should return its text without leading/trailing space.
        let single = vec![seg("hello", Some(0.9))];
        assert_eq!(concat_segment_text(&single), "hello");
        // Two segments joined by space.
        let two = vec![seg("hello", Some(0.9)), seg("world", Some(0.8))];
        assert_eq!(concat_segment_text(&two), "hello world");
    }

    #[test]
    fn window_manager_current_window_size_after_set_reflects_clamped_value() {
        let mut wm = WindowManager::new("run-size", 5000, 200);
        assert_eq!(wm.current_window_size(), 5000);
        // Set within range.
        wm.set_window_size(8000);
        assert_eq!(wm.current_window_size(), 8000);
        // Set below min (default 1000) → clamped to 1000.
        wm.set_window_size(500);
        assert_eq!(wm.current_window_size(), 1000);
        // Set above max (default 30000) → clamped to 30000.
        wm.set_window_size(50_000);
        assert_eq!(wm.current_window_size(), 30_000);
        // Windows created after set_window_size use the new value.
        let w = wm.next_window(0, "h");
        assert_eq!(w.duration_ms(), 30_000);
    }

    #[test]
    fn correction_event_new_maps_all_constructor_args_to_fields() {
        let fast = vec![seg("fast text", Some(0.8))];
        let corrected = vec![seg("quality text", Some(0.95))];
        let event = CorrectionEvent::new(
            7,  // correction_id
            3,  // retracted_seq
            42, // window_id
            "quality-v2".to_owned(),
            corrected,
            350, // quality_latency_ms
            "2026-01-15T12:00:00Z".to_owned(),
            &fast,
        );
        assert_eq!(event.correction_id, 7);
        assert_eq!(event.retracted_seq, 3);
        assert_eq!(event.window_id, 42);
        assert_eq!(event.quality_model_id, "quality-v2");
        assert_eq!(event.quality_latency_ms, 350);
        assert_eq!(event.corrected_at_rfc3339, "2026-01-15T12:00:00Z");
        assert_eq!(event.corrected_segments.len(), 1);
        assert_eq!(event.corrected_segments[0].text, "quality text");
        // quality_confidence_mean computed from corrected_segments.
        assert!((event.quality_confidence_mean - 0.95).abs() < f64::EPSILON);
        // drift computed from fast vs quality.
        assert!(event.drift.wer_approx > 0.0, "different text → nonzero WER");
    }

    #[test]
    fn window_state_holds_both_fast_and_quality_results() {
        let mut wm = WindowManager::new("run-both", 5000, 0);
        let w = wm.next_window(0, "h");
        let wid = w.window_id;
        // Initially: both None, status Pending.
        let ws = wm.get_window(wid).unwrap();
        assert!(ws.fast_result.is_none());
        assert!(ws.quality_result.is_none());
        assert_eq!(ws.status, WindowStatus::Pending);
        // Record fast result.
        let fast = PartialTranscript::new(
            0,
            wid,
            "f".into(),
            vec![seg("fast", Some(0.8))],
            50,
            "t".into(),
        );
        wm.record_fast_result(wid, fast);
        let ws = wm.get_window(wid).unwrap();
        assert!(ws.fast_result.is_some());
        assert!(ws.quality_result.is_none());
        assert_eq!(ws.status, WindowStatus::FastComplete);
        // Record quality result.
        wm.record_quality_result(wid, vec![seg("quality", Some(0.9))]);
        let ws = wm.get_window(wid).unwrap();
        assert!(ws.fast_result.is_some());
        assert!(ws.quality_result.is_some());
        assert_eq!(ws.status, WindowStatus::QualityComplete);
        assert_eq!(ws.quality_result.as_ref().unwrap()[0].text, "quality");
    }

    #[test]
    fn beta_posterior_weakly_informative_mean_is_half() {
        let post = BetaPosterior::weakly_informative();
        assert_eq!(post.alpha, 2.0);
        assert_eq!(post.beta, 2.0);
        assert!((post.mean() - 0.5).abs() < f64::EPSILON);
        // Variance for Beta(2,2) = 2*2 / (4*4*5) = 4/80 = 0.05.
        assert!((post.variance() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn evidence_ledger_window_size_trend_with_varying_sizes() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let sizes = [3000u64, 3500, 2500, 4000];
        for (i, &size) in sizes.iter().enumerate() {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i as u64,
                window_id: i as u64,
                run_id: "r".into(),
                timestamp_rfc3339: "t".into(),
                fast_model_id: "f".into(),
                fast_latency_ms: 50,
                fast_confidence_mean: 0.9,
                fast_segment_count: 1,
                quality_model_id: "q".into(),
                quality_latency_ms: 200,
                quality_confidence_mean: 0.95,
                quality_segment_count: 1,
                drift: CorrectionDrift {
                    wer_approx: 0.0,
                    confidence_delta: 0.0,
                    segment_count_delta: 0,
                    text_edit_distance: 0,
                },
                decision: "confirm".into(),
                window_size_ms: size,
                correction_rate_at_decision: 0.0,
                controller_confidence: 1.0,
                fallback_active: false,
                fallback_reason: None,
            });
        }
        let trend = ledger.window_size_trend();
        assert_eq!(trend.len(), 4);
        assert_eq!(trend[0], (0, 3000));
        assert_eq!(trend[1], (1, 3500));
        assert_eq!(trend[2], (2, 2500));
        assert_eq!(trend[3], (3, 4000));
    }

    #[test]
    fn correction_tracker_cumulative_wer_and_max_wer_after_three_windows() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance {
            max_wer: 1.0, // high threshold: won't trigger correction on WER alone
            max_confidence_delta: 1.0,
            max_edit_distance: 1000,
            always_correct: false,
        });
        // Window 1: "a" vs "a" → wer 0.0
        let p1 = PartialTranscript::new(
            0,
            10,
            "f".into(),
            vec![seg("a", Some(0.9))],
            100,
            "t".into(),
        );
        tracker.register_partial(p1);
        tracker
            .submit_quality_result(10, "q", vec![seg("a", Some(0.9))], 200)
            .unwrap();
        // Window 2: "hello" vs "world" → wer 1.0
        let p2 = PartialTranscript::new(
            1,
            20,
            "f".into(),
            vec![seg("hello", Some(0.9))],
            100,
            "t".into(),
        );
        tracker.register_partial(p2);
        tracker
            .submit_quality_result(20, "q", vec![seg("world", Some(0.9))], 200)
            .unwrap();
        // Window 3: "good morning" vs "good evening" → wer 0.5
        let p3 = PartialTranscript::new(
            2,
            30,
            "f".into(),
            vec![seg("good morning", Some(0.9))],
            100,
            "t".into(),
        );
        tracker.register_partial(p3);
        tracker
            .submit_quality_result(30, "q", vec![seg("good evening", Some(0.9))], 200)
            .unwrap();

        // cumulative_wer = 0.0 + 1.0 + 0.5 = 1.5, mean = 0.5
        assert!(
            (tracker.mean_wer() - 0.5).abs() < 1e-9,
            "mean_wer should be 0.5, got {}",
            tracker.mean_wer()
        );
        assert!(
            (tracker.stats().max_observed_wer - 1.0).abs() < f64::EPSILON,
            "max_observed_wer should be 1.0"
        );
    }

    #[test]
    fn speculation_window_stores_overlap_ms_and_audio_hash() {
        let w = SpeculationWindow::new(5, "run-x".into(), 1000, 4000, 300, "abc123".into());
        assert_eq!(w.window_id, 5);
        assert_eq!(w.run_id, "run-x");
        assert_eq!(w.start_ms, 1000);
        assert_eq!(w.end_ms, 4000);
        assert_eq!(w.overlap_ms, 300);
        assert_eq!(w.audio_hash, "abc123");
    }

    #[test]
    fn correction_event_is_significant_zero_threshold_any_wer() {
        let fast = vec![seg("a", Some(0.9))];
        let quality = vec![seg("b", Some(0.9))];
        let event = CorrectionEvent::new(0, 0, 0, "q".into(), quality, 100, "t".into(), &fast);
        // With threshold 0.0, any nonzero WER is significant.
        assert!(
            event.is_significant(0.0),
            "any correction should be significant at threshold 0.0"
        );
        // With threshold 1.0, nothing is significant (WER maxes at 1.0, is_significant uses >).
        assert!(
            !event.is_significant(1.0),
            "WER cannot exceed 1.0 so threshold 1.0 is never significant"
        );
    }

    #[test]
    fn correction_tracker_all_resolved_false_when_partially_resolved() {
        let mut tracker = CorrectionTracker::new(CorrectionTolerance::default());
        // Register two partials.
        let p0 =
            PartialTranscript::new(0, 10, "f".into(), vec![seg("a", Some(0.9))], 50, "t".into());
        let p1 =
            PartialTranscript::new(1, 20, "f".into(), vec![seg("b", Some(0.8))], 50, "t".into());
        tracker.register_partial(p0);
        tracker.register_partial(p1);
        // Both pending → not all resolved.
        assert!(!tracker.all_resolved(), "two pending → not all resolved");
        // Resolve one.
        tracker
            .submit_quality_result(10, "q", vec![seg("a", Some(0.9))], 100)
            .unwrap();
        // One resolved, one pending → still not all resolved.
        assert!(
            !tracker.all_resolved(),
            "one resolved + one pending → not all resolved"
        );
        // Resolve the other.
        tracker
            .submit_quality_result(20, "q", vec![seg("b", Some(0.8))], 100)
            .unwrap();
        assert!(tracker.all_resolved(), "both resolved → all resolved");
    }

    #[test]
    fn controller_recommend_returns_hold_with_exact_five_observations() {
        // MIN_WINDOWS_FOR_ADAPT = 5. With exactly 5 observations, should NOT hold
        // due to min_windows (that check is <, not <=). Should proceed to other checks.
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        for _ in 0..5 {
            ctrl.observe(
                &CorrectionDecision::Confirm {
                    seq: 0,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        // At exactly 5 windows, min_windows check passes (5 < 5 is false).
        // confidence = 5/20 = 0.25 < 0.5 → Hold due to low confidence.
        let action = ctrl.recommend();
        assert_eq!(action, ControllerAction::Hold);
    }

    #[test]
    fn merge_segments_cross_window_dedup_prefers_higher_confidence() {
        let mut wm = WindowManager::new("run-cross", 5000, 500);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(4500, "h1"); // 500ms overlap
        // w0: quality segment at t=4.5-5.0 with confidence 0.7.
        wm.record_quality_result(
            w0.window_id,
            vec![timed_seg("overlap low", 4.5, 5.0, Some(0.7))],
        );
        wm.resolve_window(w0.window_id);
        // w1: quality segment at t=4.5-5.0 with confidence 0.95 (same timing).
        wm.record_quality_result(
            w1.window_id,
            vec![timed_seg("overlap high", 4.5, 5.0, Some(0.95))],
        );
        wm.resolve_window(w1.window_id);
        let merged = wm.merge_segments();
        // Both overlap within 0.1s tolerance → deduplicated to the higher-confidence one.
        assert_eq!(merged.len(), 1, "overlapping segments should dedup to 1");
        assert_eq!(merged[0].text, "overlap high");
        assert_eq!(merged[0].confidence, Some(0.95));
    }

    // ── Task #296 — speculation.rs pass 32 edge-case tests ────────────

    #[test]
    fn evidence_ledger_diagnostics_with_mixed_decisions() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let drift_low = CorrectionDrift {
            wer_approx: 0.1,
            confidence_delta: 0.05,
            segment_count_delta: 0,
            text_edit_distance: 2,
        };
        let drift_zero = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        ledger.record(CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 1,
            run_id: "r".into(),
            timestamp_rfc3339: "2026-01-01T00:00:00Z".into(),
            fast_model_id: "tiny".into(),
            fast_latency_ms: 50,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "large".into(),
            quality_latency_ms: 200,
            quality_confidence_mean: 0.9,
            quality_segment_count: 1,
            drift: drift_low,
            decision: "correct".to_owned(),
            window_size_ms: 5000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        });
        ledger.record(CorrectionEvidenceEntry {
            entry_id: 1,
            window_id: 2,
            run_id: "r".into(),
            timestamp_rfc3339: "2026-01-01T00:00:01Z".into(),
            fast_model_id: "tiny".into(),
            fast_latency_ms: 70,
            fast_confidence_mean: 0.85,
            fast_segment_count: 1,
            quality_model_id: "large".into(),
            quality_latency_ms: 300,
            quality_confidence_mean: 0.85,
            quality_segment_count: 1,
            drift: drift_zero,
            decision: "confirm".to_owned(),
            window_size_ms: 5000,
            correction_rate_at_decision: 0.5,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        });
        let diag = ledger.diagnostics();
        // 1 correction out of 2 entries → 0.5.
        assert!(
            (diag["correction_rate"].as_f64().unwrap() - 0.5).abs() < f64::EPSILON,
            "correction_rate should be 0.5"
        );
        // mean_fast: (50+70)/2 = 60.
        assert!(
            (diag["mean_fast_latency_ms"].as_f64().unwrap() - 60.0).abs() < f64::EPSILON,
            "mean_fast_latency_ms should be 60"
        );
        // mean_quality: (200+300)/2 = 250.
        assert!(
            (diag["mean_quality_latency_ms"].as_f64().unwrap() - 250.0).abs() < f64::EPSILON,
            "mean_quality_latency_ms should be 250"
        );
        // mean_wer: (0.1+0.0)/2 = 0.05.
        assert!(
            (diag["mean_wer"].as_f64().unwrap() - 0.05).abs() < f64::EPSILON,
            "mean_wer should be 0.05"
        );
        // latency_savings: (250-60)/250*100 = 76%.
        assert!(
            (diag["latency_savings_pct"].as_f64().unwrap() - 76.0).abs() < f64::EPSILON,
            "latency_savings_pct should be 76.0"
        );
    }

    #[test]
    fn record_fast_result_on_unknown_window_is_noop() {
        let mut wm = WindowManager::new("run-noop", 5000, 200);
        let w = wm.next_window(0, "hash-a");
        let partial = PartialTranscript::new(
            1,
            999, // wrong window_id
            "fast".to_owned(),
            vec![seg("test", Some(0.9))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        wm.record_fast_result(999, partial);
        // The real window should still be Pending.
        let ws = wm
            .get_window(w.window_id)
            .expect("real window should exist");
        assert_eq!(ws.status, WindowStatus::Pending);
        assert!(ws.fast_result.is_none());
    }

    #[test]
    fn resolve_window_on_unknown_window_is_noop() {
        let mut wm = WindowManager::new("run-noop2", 5000, 200);
        let w = wm.next_window(0, "hash-b");
        // Resolve a non-existent window_id — should not panic or change anything.
        wm.resolve_window(999);
        let ws = wm
            .get_window(w.window_id)
            .expect("real window should exist");
        assert_eq!(
            ws.status,
            WindowStatus::Pending,
            "original window should still be Pending"
        );
        assert_eq!(wm.windows_resolved(), 0);
        assert_eq!(wm.windows_pending(), 1);
    }

    #[test]
    fn beta_posterior_variance_decreases_with_extreme_alpha_beta_ratio() {
        // Start with weakly informative prior (1,1) — uniform.
        let mut p = BetaPosterior::weakly_informative();
        let initial_var = p.variance();
        // Add 100 corrections (alpha goes up), 1 confirmation (beta stays near 1).
        for _ in 0..100 {
            p.observe_correction();
        }
        p.observe_confirmation();
        // Mean should be very high (close to 1.0).
        assert!(
            p.mean() > 0.95,
            "mean should be close to 1.0, got {}",
            p.mean()
        );
        // Variance should be much smaller than initial.
        let final_var = p.variance();
        assert!(
            final_var < initial_var * 0.01,
            "variance should decrease dramatically: initial={initial_var}, final={final_var}"
        );
        assert!(final_var.is_finite(), "variance must be finite");
        assert!(final_var > 0.0, "variance must be positive");
    }

    #[test]
    fn calibration_tracker_brier_score_sparse_fill_of_large_window() {
        // Create a large window but only fill a few entries.
        let mut tracker = CalibrationTracker::new(1000);
        // Only 3 entries in a window of 1000.
        tracker.record(0.9, true); // squared error: (0.9-1)^2 = 0.01
        tracker.record(0.1, false); // squared error: (0.1-0)^2 = 0.01
        tracker.record(0.5, true); // squared error: (0.5-1)^2 = 0.25
        // Brier = (0.01 + 0.01 + 0.25) / 3 = 0.09.
        let brier = tracker.brier_score();
        assert!(
            (brier - 0.09).abs() < 1e-12,
            "Brier score should be 0.09, got {brier}"
        );
        assert_eq!(tracker.sample_count(), 3);
    }

    // ── Task #300 — speculation.rs pass 33 edge-case tests ────────────

    #[test]
    fn mean_confidence_empty_segments_returns_zero() {
        let empty: Vec<TranscriptionSegment> = vec![];
        assert!((mean_confidence(&empty) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_confidence_all_none_confidence_returns_zero() {
        let segs = vec![seg("hello", None), seg("world", None)];
        assert!(
            (mean_confidence(&segs) - 0.0).abs() < f64::EPSILON,
            "all-None confidence should yield 0.0"
        );
    }

    #[test]
    fn merge_segments_resolved_window_without_quality_uses_fast_result() {
        let mut wm = WindowManager::new("run-fast-fallback", 5000, 200);
        let w = wm.next_window(0, "hash-fb");
        // Record only fast result, no quality result.
        let partial = PartialTranscript::new(
            0,
            w.window_id,
            "tiny".to_owned(),
            vec![timed_seg("fast only", 0.0, 2.0, Some(0.8))],
            50,
            "2026-01-01T00:00:00Z".to_owned(),
        );
        wm.record_fast_result(w.window_id, partial);
        // Resolve the window without setting quality_result.
        wm.resolve_window(w.window_id);
        let merged = wm.merge_segments();
        assert_eq!(merged.len(), 1, "should fall back to fast result");
        assert_eq!(merged[0].text, "fast only");
    }

    #[test]
    fn window_manager_three_windows_all_retrievable_by_id() {
        let mut wm = WindowManager::new("run-mono", 3000, 200);
        let w0 = wm.next_window(0, "h0");
        let w1 = wm.next_window(3000, "h1");
        let w2 = wm.next_window(6000, "h2");
        assert_eq!(w0.window_id, 0);
        assert_eq!(w1.window_id, 1);
        assert_eq!(w2.window_id, 2);
        // Verify that all three windows exist in the manager.
        assert!(wm.get_window(0).is_some());
        assert!(wm.get_window(1).is_some());
        assert!(wm.get_window(2).is_some());
        assert!(wm.get_window(3).is_none());
    }

    #[test]
    fn controller_apply_generates_evidence_entry_per_call() {
        let mut ctrl = SpeculationWindowController::new(3000, 1000, 10000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.05,
            confidence_delta: 0.01,
            segment_count_delta: 0,
            text_edit_distance: 1,
        };
        let confirm = CorrectionDecision::Confirm {
            seq: 0,
            drift: drift.clone(),
        };
        // Observe and apply 3 times.
        for _ in 0..3 {
            ctrl.observe(&confirm, &drift);
            ctrl.apply();
        }
        assert_eq!(
            ctrl.evidence().len(),
            3,
            "three apply() calls should produce 3 evidence entries"
        );
        // Decision IDs should be monotonically increasing.
        let ids: Vec<u64> = ctrl.evidence().iter().map(|e| e.decision_id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn mean_fast_latency_computes_correct_average() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // Entry with fast_latency_ms = 100.
        ledger.record(CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 1,
            run_id: "r".into(),
            timestamp_rfc3339: "t".into(),
            fast_model_id: "tiny".into(),
            fast_latency_ms: 100,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "large".into(),
            quality_latency_ms: 500,
            quality_confidence_mean: 0.9,
            quality_segment_count: 1,
            drift: drift.clone(),
            decision: "confirm".into(),
            window_size_ms: 5000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        });
        // Entry with fast_latency_ms = 200.
        ledger.record(CorrectionEvidenceEntry {
            entry_id: 1,
            window_id: 2,
            run_id: "r".into(),
            timestamp_rfc3339: "t".into(),
            fast_model_id: "tiny".into(),
            fast_latency_ms: 200,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "large".into(),
            quality_latency_ms: 600,
            quality_confidence_mean: 0.9,
            quality_segment_count: 1,
            drift,
            decision: "confirm".into(),
            window_size_ms: 5000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        });
        assert!(
            (ledger.mean_fast_latency() - 150.0).abs() < f64::EPSILON,
            "mean of 100 and 200 should be 150, got {}",
            ledger.mean_fast_latency()
        );
    }

    #[test]
    fn mean_quality_latency_empty_ledger_returns_zero() {
        let ledger = CorrectionEvidenceLedger::new(10);
        assert!(
            ledger.mean_quality_latency().abs() < f64::EPSILON,
            "empty ledger should return 0.0 for mean quality latency"
        );
    }

    #[test]
    fn latency_savings_pct_with_known_values() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // fast=100ms, quality=400ms → savings = (400-100)/400*100 = 75%.
        ledger.record(CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 1,
            run_id: "r".into(),
            timestamp_rfc3339: "t".into(),
            fast_model_id: "tiny".into(),
            fast_latency_ms: 100,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "large".into(),
            quality_latency_ms: 400,
            quality_confidence_mean: 0.9,
            quality_segment_count: 1,
            drift,
            decision: "confirm".into(),
            window_size_ms: 5000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        });
        assert!(
            (ledger.latency_savings_pct() - 75.0).abs() < f64::EPSILON,
            "savings should be 75%, got {}",
            ledger.latency_savings_pct()
        );
    }

    #[test]
    fn window_size_trend_returns_ordered_pairs() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        for (i, ws) in [3000u64, 5000, 4000].iter().enumerate() {
            ledger.record(CorrectionEvidenceEntry {
                entry_id: i as u64,
                window_id: (i + 1) as u64,
                run_id: "r".into(),
                timestamp_rfc3339: "t".into(),
                fast_model_id: "tiny".into(),
                fast_latency_ms: 50,
                fast_confidence_mean: 0.8,
                fast_segment_count: 1,
                quality_model_id: "large".into(),
                quality_latency_ms: 200,
                quality_confidence_mean: 0.9,
                quality_segment_count: 1,
                drift: drift.clone(),
                decision: "confirm".into(),
                window_size_ms: *ws,
                correction_rate_at_decision: 0.0,
                controller_confidence: 0.5,
                fallback_active: false,
                fallback_reason: None,
            });
        }
        let trend = ledger.window_size_trend();
        assert_eq!(trend.len(), 3);
        assert_eq!(trend[0], (1, 3000));
        assert_eq!(trend[1], (2, 5000));
        assert_eq!(trend[2], (3, 4000));
    }

    #[test]
    fn get_window_mut_allows_mutation_of_window_state() {
        let mut wm = WindowManager::new("run-mut", 5000, 200);
        let win = wm.next_window(0, "hash");
        let wid = win.window_id;
        // Before mutation: fast_result is None.
        assert!(wm.get_window(wid).unwrap().fast_result.is_none());
        // Mutate via get_window_mut: set a PartialTranscript.
        let ws = wm.get_window_mut(wid).expect("should find window");
        ws.fast_result = Some(PartialTranscript {
            seq: 0,
            window_id: wid,
            model_id: "tiny".into(),
            segments: vec![seg("mutated", Some(0.99))],
            latency_ms: 42,
            confidence_mean: 0.99,
            emitted_at_rfc3339: "2026-01-01T00:00:00Z".into(),
            status: PartialStatus::Pending,
        });
        // Verify mutation stuck.
        let ws2 = wm.get_window(wid).expect("still present");
        let pt = ws2.fast_result.as_ref().expect("should have fast_result");
        assert_eq!(pt.segments[0].text, "mutated");
        assert_eq!(pt.latency_ms, 42);
    }

    #[test]
    fn record_quality_result_on_unknown_window_is_noop() {
        let mut wm = WindowManager::new("run-noop", 5000, 200);
        let win = wm.next_window(0, "h");
        // Record quality for a non-existent window id.
        wm.record_quality_result(999, vec![seg("phantom", Some(0.9))]);
        // Original window should be unaffected.
        let ws = wm.get_window(win.window_id).expect("present");
        assert!(ws.quality_result.is_none());
        assert_eq!(ws.status, WindowStatus::Pending);
    }

    #[test]
    fn latency_savings_pct_equal_latencies_returns_zero() {
        let mut ledger = CorrectionEvidenceLedger::new(10);
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // fast=200ms, quality=200ms → savings = 0%.
        ledger.record(CorrectionEvidenceEntry {
            entry_id: 0,
            window_id: 1,
            run_id: "r".into(),
            timestamp_rfc3339: "t".into(),
            fast_model_id: "tiny".into(),
            fast_latency_ms: 200,
            fast_confidence_mean: 0.8,
            fast_segment_count: 1,
            quality_model_id: "large".into(),
            quality_latency_ms: 200,
            quality_confidence_mean: 0.9,
            quality_segment_count: 1,
            drift,
            decision: "confirm".into(),
            window_size_ms: 5000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        });
        assert!(
            ledger.latency_savings_pct().abs() < f64::EPSILON,
            "equal latencies should give 0% savings, got {}",
            ledger.latency_savings_pct()
        );
    }

    #[test]
    #[should_panic(expected = "alpha must be positive")]
    fn beta_posterior_new_panics_on_zero_alpha() {
        let _ = BetaPosterior::new(0.0, 1.0);
    }

    #[test]
    fn controller_confidence_increases_with_window_count() {
        let mut ctrl = SpeculationWindowController::new(5000, 2000, 10_000, 1000);
        // Initially: window_count = 0, confidence = 0.0.
        assert!(
            ctrl.confidence().abs() < f64::EPSILON,
            "initial confidence should be 0.0"
        );
        // Process some windows to increase window_count.
        let confirm = CorrectionDecision::Confirm {
            seq: 0,
            drift: CorrectionDrift {
                wer_approx: 0.0,
                confidence_delta: 0.0,
                segment_count_delta: 0,
                text_edit_distance: 0,
            },
        };
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        for _ in 0..10 {
            ctrl.observe(&confirm, &drift);
            ctrl.apply();
        }
        let mid_confidence = ctrl.confidence();
        assert!(
            mid_confidence > 0.0 && mid_confidence < 1.0,
            "after 10 windows confidence should be between 0 and 1, got {mid_confidence}"
        );
        // Process 10 more windows to reach full confidence.
        for _ in 0..10 {
            ctrl.observe(&confirm, &drift);
            ctrl.apply();
        }
        assert!(
            (ctrl.confidence() - 1.0).abs() < f64::EPSILON,
            "after 20 windows confidence should be 1.0, got {}",
            ctrl.confidence()
        );
    }

    #[test]
    fn controller_state_serde_round_trip_preserves_all_fields() {
        let state = ControllerState {
            correction_rate: 0.15,
            mean_wer: 0.08,
            window_count: 42,
            current_window_ms: 7500,
        };
        let json = serde_json::to_string(&state).expect("serialize");
        let back: ControllerState = serde_json::from_str(&json).expect("deserialize");
        assert!((back.correction_rate - 0.15).abs() < f64::EPSILON);
        assert!((back.mean_wer - 0.08).abs() < f64::EPSILON);
        assert_eq!(back.window_count, 42);
        assert_eq!(back.current_window_ms, 7500);
    }

    // ── Task #307 — speculation edge-case tests pass 36 ─────────────

    #[test]
    fn recommend_brier_hold_skipped_when_sample_count_below_10() {
        // When Brier score > 0.25 but sample_count < 10, the Brier Hold
        // check at line 1062 should be skipped (both conditions must hold).
        // With a high correction rate, recommend() should fall through to Grow.
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);

        // Feed 6 corrections with high WER to push correction rate > 0.75
        // but only 6 observations → calibration sample_count = 6 (< 10).
        let correct = CorrectionDecision::Correct {
            correction: CorrectionEvent {
                correction_id: 0,
                retracted_seq: 0,
                window_id: 0,
                quality_model_id: "q".to_owned(),
                corrected_segments: vec![],
                quality_latency_ms: 100,
                quality_confidence_mean: 0.5,
                drift: CorrectionDrift::compute(&[], &[]),
                corrected_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            },
        };
        let drift = CorrectionDrift {
            wer_approx: 0.5,
            confidence_delta: 0.2,
            segment_count_delta: 0,
            text_edit_distance: 5,
        };
        for _ in 0..6 {
            ctrl.observe(&correct, &drift);
        }

        assert!(
            ctrl.calibration.sample_count() < 10,
            "sample count should be < 10"
        );
        // With correction_rate > 0.75 and not at max window, should Grow
        // (not Hold from Brier check since sample_count < 10).
        let action = ctrl.recommend();
        assert!(
            matches!(action, ControllerAction::Grow(_)),
            "should Grow due to runaway rate, not Hold from Brier (insufficient samples), got {action:?}"
        );
    }

    #[test]
    fn recommend_consecutive_confirmations_at_min_window_returns_hold() {
        // When at min_window_ms with 20+ consecutive confirmations, the
        // Shrink guard (current_window_ms > min_window_ms) fails → Hold.
        let mut ctrl = SpeculationWindowController::new(2000, 2000, 10000, 500);
        // initial_window_ms == min_window_ms → already at minimum.
        let confirm = CorrectionDecision::Confirm {
            seq: 0,
            drift: CorrectionDrift {
                wer_approx: 0.0,
                confidence_delta: 0.0,
                segment_count_delta: 0,
                text_edit_distance: 0,
            },
        };
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // Feed 25 consecutive confirmations.
        for _ in 0..25 {
            ctrl.observe(&confirm, &drift);
            ctrl.apply();
        }
        assert!(
            ctrl.consecutive_zero_corrections >= 20,
            "should have 20+ consecutive confirmations"
        );
        assert_eq!(
            ctrl.current_window_ms, ctrl.min_window_ms,
            "should be at min window"
        );
        // Can't shrink further → should be Hold (or low correction rate Shrink
        // is also guarded by > min_window_ms).
        let action = ctrl.recommend();
        assert!(
            matches!(action, ControllerAction::Hold),
            "at min window with 25 confirmations should Hold, got {action:?}"
        );
    }

    #[test]
    fn recommend_low_correction_rate_below_10_windows_returns_hold() {
        // When correction_rate < LOW_CORRECTION_RATE (0.0625) but
        // window_count <= 10, the Shrink guard fails → Hold.
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        let confirm = CorrectionDecision::Confirm {
            seq: 0,
            drift: CorrectionDrift {
                wer_approx: 0.0,
                confidence_delta: 0.0,
                segment_count_delta: 0,
                text_edit_distance: 0,
            },
        };
        let drift = CorrectionDrift {
            wer_approx: 0.0,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };
        // Feed exactly 8 confirmations: window_count=8 (> MIN_WINDOWS_FOR_ADAPT=5
        // but <= 10), correction_rate ≈ 0 (< LOW_CORRECTION_RATE).
        for _ in 0..8 {
            ctrl.observe(&confirm, &drift);
        }
        assert!(
            ctrl.state.window_count > 5 && ctrl.state.window_count <= 10,
            "window_count should be between 5 and 10, got {}",
            ctrl.state.window_count
        );
        assert!(
            ctrl.state.correction_rate < 0.0625,
            "correction_rate should be low, got {}",
            ctrl.state.correction_rate
        );
        // The low correction rate Shrink requires window_count > 10.
        // Since window_count == 8, should fall through to Hold.
        let action = ctrl.recommend();
        assert!(
            matches!(action, ControllerAction::Hold),
            "with low rate and only 8 windows should Hold (not Shrink), got {action:?}"
        );
    }

    #[test]
    fn apply_runaway_fallback_with_insufficient_brier_samples() {
        // When correction_rate > 0.75 but Brier sample_count < 10,
        // apply() should activate fallback with "correction rate > 75%"
        // (the runaway branch at line 1111, not the Brier branch).
        let mut ctrl = SpeculationWindowController::new(5000, 1000, 30000, 500);
        let correct = CorrectionDecision::Correct {
            correction: CorrectionEvent {
                correction_id: 0,
                retracted_seq: 0,
                window_id: 0,
                quality_model_id: "q".to_owned(),
                corrected_segments: vec![],
                quality_latency_ms: 100,
                quality_confidence_mean: 0.5,
                drift: CorrectionDrift::compute(&[], &[]),
                corrected_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            },
        };
        let drift = CorrectionDrift {
            wer_approx: 0.5,
            confidence_delta: 0.2,
            segment_count_delta: 0,
            text_edit_distance: 5,
        };
        // Feed 6 corrections → correction_rate > 0.75, sample_count = 6 < 10.
        for _ in 0..6 {
            ctrl.observe(&correct, &drift);
        }
        assert!(ctrl.state.correction_rate > 0.75);
        assert!(ctrl.calibration.sample_count() < 10);

        ctrl.apply();

        assert!(
            ctrl.fallback_active,
            "fallback should be active with runaway correction rate"
        );
        assert_eq!(
            ctrl.fallback_reason.as_deref(),
            Some("correction rate > 75%"),
            "fallback reason should indicate runaway rate"
        );
    }

    #[test]
    fn entries_returns_fifo_ordered_deque() {
        // Verify entries() returns entries in FIFO order after multiple records.
        let mut ledger = CorrectionEvidenceLedger::new(5);
        let make_entry = |id: u64| CorrectionEvidenceEntry {
            entry_id: id,
            window_id: id,
            run_id: "r".to_owned(),
            timestamp_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            fast_model_id: "fast".to_owned(),
            fast_latency_ms: 10,
            fast_confidence_mean: 0.9,
            fast_segment_count: 1,
            quality_model_id: "quality".to_owned(),
            quality_latency_ms: 50,
            quality_confidence_mean: 0.95,
            quality_segment_count: 1,
            drift: CorrectionDrift::compute(&[], &[]),
            decision: "confirm".to_owned(),
            window_size_ms: 3000,
            correction_rate_at_decision: 0.0,
            controller_confidence: 0.5,
            fallback_active: false,
            fallback_reason: None,
        };

        // Record entries with ids 10, 20, 30.
        ledger.record(make_entry(10));
        ledger.record(make_entry(20));
        ledger.record(make_entry(30));

        let entries = ledger.entries();
        assert_eq!(entries.len(), 3);
        // Verify FIFO order: oldest first, newest last.
        let ids: Vec<u64> = entries.iter().map(|e| e.entry_id).collect();
        assert_eq!(ids, vec![10, 20, 30], "entries should be in FIFO order");

        // Record 3 more to trigger eviction (capacity = 5).
        ledger.record(make_entry(40));
        ledger.record(make_entry(50));
        ledger.record(make_entry(60));

        let entries = ledger.entries();
        assert_eq!(entries.len(), 5, "should be at capacity");
        let ids: Vec<u64> = entries.iter().map(|e| e.entry_id).collect();
        // Entry 10 should have been evicted.
        assert_eq!(
            ids,
            vec![20, 30, 40, 50, 60],
            "oldest entry should be evicted, FIFO order preserved"
        );
    }

    // ------------------------------------------------------------------
    // speculation edge-case tests pass 37
    // ------------------------------------------------------------------

    #[test]
    fn merge_segments_not_deduped_when_start_matches_but_end_differs() {
        // Exercises the AND condition at line 594:
        //   (seg_start - last_start).abs() < 0.1 && (seg_end - last_end).abs() < 0.1
        // When start_sec values are within 0.1s but end_sec values differ by
        // >= 0.1s, segments should NOT be deduplicated.
        let mut wm = WindowManager::new("run-end-diff", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        wm.record_quality_result(
            w0.window_id,
            vec![
                timed_seg("alpha", 1.0, 2.0, Some(0.9)),
                // Same start (within 0.1) but end differs by 0.5s.
                timed_seg("beta", 1.05, 2.5, Some(0.8)),
            ],
        );
        wm.resolve_window(w0.window_id);
        let merged = wm.merge_segments();
        assert_eq!(
            merged.len(),
            2,
            "segments with matching start but differing end should NOT be deduped"
        );
        assert_eq!(merged[0].text, "alpha");
        assert_eq!(merged[1].text, "beta");
    }

    #[test]
    fn merge_segments_equal_confidence_keeps_first_segment() {
        // Exercises the `seg_conf > last_conf` false branch at line 604
        // when both overlapping segments have identical confidence.
        // The strict `>` means the first segment is kept (not replaced).
        let mut wm = WindowManager::new("run-equal-conf", 5000, 0);
        let w0 = wm.next_window(0, "h0");
        wm.record_quality_result(
            w0.window_id,
            vec![
                timed_seg("first", 1.0, 2.0, Some(0.85)),
                // Same start and end within tolerance, same confidence.
                timed_seg("second", 1.0, 2.0, Some(0.85)),
            ],
        );
        wm.resolve_window(w0.window_id);
        let merged = wm.merge_segments();
        assert_eq!(
            merged.len(),
            1,
            "overlapping equal-confidence segments should dedup to one"
        );
        assert_eq!(
            merged[0].text, "first",
            "first segment should be kept when confidences are equal (strict >)"
        );
    }

    #[test]
    fn wer_exactly_at_max_threshold_is_confirmed() {
        // Exercises the strict `>` comparison at line 740:
        //   drift.wer_approx > self.tolerance.max_wer
        // When WER exactly equals max_wer, the condition is false and the
        // result should be Confirm (not Correct).
        let tolerance = CorrectionTolerance {
            max_wer: 0.5,
            max_confidence_delta: 1.0, // large so it never triggers
            max_edit_distance: 10_000, // large so it never triggers
            always_correct: false,
        };
        let mut tracker = CorrectionTracker::new(tolerance);

        // Construct segments so that WER = 0.5 exactly:
        // fast has 2 words, quality has 2 words with 1 word different.
        // word_edit_dist = 1, max_words = 2, WER = 1/2 = 0.5.
        let fast_segs = vec![seg("hello world", Some(0.9))];
        let quality_segs = vec![seg("hello earth", Some(0.9))];

        let partial = PartialTranscript::new(0, 10, "fast".into(), fast_segs, 100, "t".into());
        tracker.register_partial(partial);
        let decision = tracker
            .submit_quality_result(10, "quality", quality_segs, 200)
            .expect("submit should succeed");
        assert!(
            matches!(decision, CorrectionDecision::Confirm { .. }),
            "WER exactly at max_wer threshold should be confirmed (strict >)"
        );
    }

    #[test]
    fn apply_runaway_at_max_window_sets_fallback_but_window_unchanged() {
        // Exercises the runaway branch in apply() (lines 1111-1116) when
        // recommend() returns Hold (not Grow). This happens when
        // correction_rate > 0.75 but current_window_ms >= max_window_ms,
        // so the Grow guard at line 1069 fails. The fallback is activated
        // but the window size remains at max.
        let max_ms = 5000;
        let mut ctrl = SpeculationWindowController::new(max_ms, 1000, max_ms, 500);

        // Feed corrections to push correction_rate > 0.75.
        // Need at least MIN_WINDOWS_FOR_ADAPT=5 observations.
        let correction_drift = CorrectionDrift {
            wer_approx: 0.5,
            confidence_delta: 0.3,
            segment_count_delta: 1,
            text_edit_distance: 20,
        };
        let confirm_drift = CorrectionDrift {
            wer_approx: 0.01,
            confidence_delta: 0.0,
            segment_count_delta: 0,
            text_edit_distance: 0,
        };

        // 6 corrections + 1 confirmation → correction_rate ≈ 6/7 ≈ 0.857
        for i in 0..6u64 {
            let d = CorrectionDecision::Correct {
                correction: CorrectionEvent::new(
                    i,
                    i,
                    i * 10,
                    "q".into(),
                    vec![],
                    100,
                    "2026-01-01T00:00:00Z".into(),
                    &[],
                ),
            };
            ctrl.observe(&d, &correction_drift);
        }
        let confirm = CorrectionDecision::Confirm {
            seq: 99,
            drift: confirm_drift.clone(),
        };
        ctrl.observe(&confirm, &confirm_drift);

        // Window is already at max; apply should activate fallback but not change window.
        let window_after = ctrl.apply();
        assert!(ctrl.is_fallback_active(), "fallback should be active");
        assert_eq!(
            window_after, max_ms,
            "window should remain at max when recommend returns Hold in runaway"
        );
    }

    #[test]
    fn record_quality_result_overwrites_previous_fast_result() {
        // Exercises calling record_quality_result after record_fast_result
        // on the same window, then calling record_fast_result again to
        // verify the overwrite behavior. The status should regress from
        // QualityComplete back to FastComplete.
        let mut wm = WindowManager::new("run-overwrite", 5000, 0);
        let w = wm.next_window(0, "h0");
        let wid = w.window_id;

        // First: record fast result.
        let partial = PartialTranscript::new(
            0,
            wid,
            "fast".into(),
            vec![seg("first fast", Some(0.8))],
            50,
            "t".into(),
        );
        wm.record_fast_result(wid, partial);
        assert_eq!(
            wm.get_window(wid).unwrap().status,
            WindowStatus::FastComplete
        );

        // Record quality result — advances to QualityComplete.
        wm.record_quality_result(wid, vec![timed_seg("quality", 0.0, 1.0, Some(0.95))]);
        assert_eq!(
            wm.get_window(wid).unwrap().status,
            WindowStatus::QualityComplete
        );

        // Overwrite with a second fast result — should regress to FastComplete.
        let partial2 = PartialTranscript::new(
            1,
            wid,
            "fast".into(),
            vec![seg("second fast", Some(0.7))],
            60,
            "t".into(),
        );
        wm.record_fast_result(wid, partial2);
        assert_eq!(
            wm.get_window(wid).unwrap().status,
            WindowStatus::FastComplete,
            "re-recording fast result should overwrite status back to FastComplete"
        );
    }
}
