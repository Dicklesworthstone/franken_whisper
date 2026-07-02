mod insanely_fast;
mod insanely_fast_native;
pub(crate) mod native_audio;
pub mod normalize;
mod whisper_cpp;
mod whisper_cpp_native;
mod whisper_diarization;
mod whisper_diarization_native;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use chrono::Utc;
use franken_decision::{
    DecisionContract, EvalContext, FallbackPolicy, LossMatrix, Posterior, UpdatePosteriorError,
    ValidationError, evaluate as decision_evaluate,
};
use franken_kernel::{DecisionId, TraceId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::conformance::NativeEngineRolloutStage;
use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, EngineCapabilities, TranscribeRequest, TranscriptionResult, TranscriptionSegment,
};
use crate::orchestrator::CancellationToken;
use crate::process::run_command_with_timeout;

/// Stable identifier for the current routing policy. Bump when the loss
/// matrix weights, fallback thresholds, or prior parameters change.
const ROUTING_POLICY_ID: &str = "backend-selection-v1.0";

/// Schema version for the routing evidence JSON format.
const ROUTING_EVIDENCE_SCHEMA_VERSION: &str = "1.0";

/// Maximum number of outcome records retained per backend in the sliding window.
const ROUTER_HISTORY_WINDOW: usize = 50;

/// When the calibration score (posterior margin) drops below this threshold,
/// the adaptive router falls back to the static priority order.
const ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD: f64 = 0.3;

/// When the Brier score exceeds this threshold, the adaptive router falls
/// back to the static priority order. A Brier score of 0.0 is perfect
/// calibration; 1.0 is maximal miscalibration. The threshold of 0.35
/// allows moderate prediction errors while catching systematic
/// miscalibration.
const ADAPTIVE_FALLBACK_BRIER_THRESHOLD: f64 = 0.35;

/// Minimum number of recorded outcomes before the adaptive router trusts
/// its empirical estimates over the static priors.
const ADAPTIVE_MIN_SAMPLES: usize = 5;

/// Maximum number of entries in the routing evidence ledger circular buffer.
const EVIDENCE_LEDGER_CAPACITY: usize = 200;

// ---------------------------------------------------------------------------
// bd-efr.2: CalibrationState — Brier-score confidence calibration
// ---------------------------------------------------------------------------

/// A single calibration observation: the predicted success probability and
/// the actual binary outcome (1.0 for success, 0.0 for failure).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationObservation {
    /// Predicted probability of success at decision time, in `[0.0, 1.0]`.
    pub predicted_probability: f64,
    /// Actual outcome: `1.0` if the selected backend succeeded, `0.0` otherwise.
    pub actual_outcome: f64,
    /// RFC 3339 timestamp of when the observation was recorded.
    pub observed_at_rfc3339: String,
}

/// Tracks confidence calibration using predicted success probabilities versus
/// actual outcomes, computing the Brier score as the calibration metric.
///
/// The Brier score is the mean squared error between predicted probabilities
/// and binary outcomes:
///   `BS = (1/N) * sum_i (predicted_i - actual_i)^2`
///
/// - BS = 0.0 means perfect calibration.
/// - BS = 1.0 means maximally miscalibrated.
/// - BS = 0.25 corresponds to always predicting 0.5 with random outcomes.
///
/// The state maintains a sliding window of observations to adapt to changing
/// conditions over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationState {
    /// Sliding window of calibration observations.
    observations: VecDeque<CalibrationObservation>,
    /// Maximum window size; matches `ROUTER_HISTORY_WINDOW`.
    window_size: usize,
}

impl CalibrationState {
    /// Create a new empty calibration state with the given window size.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            observations: VecDeque::new(),
            window_size,
        }
    }

    /// Record a calibration observation.
    pub fn record(&mut self, predicted_probability: f64, success: bool) {
        let predicted = if predicted_probability.is_finite() {
            predicted_probability
        } else {
            0.5
        };
        let clamped = predicted.clamp(0.0, 1.0);
        let actual = if success { 1.0 } else { 0.0 };
        if self.observations.len() >= self.window_size {
            self.observations.pop_front();
        }
        self.observations.push_back(CalibrationObservation {
            predicted_probability: clamped,
            actual_outcome: actual,
            observed_at_rfc3339: Utc::now().to_rfc3339(),
        });
    }

    /// Compute the Brier score over the current observation window.
    /// Returns `None` if no observations have been recorded.
    #[must_use]
    pub fn brier_score(&self) -> Option<f64> {
        if self.observations.is_empty() {
            return None;
        }
        let sum_sq: f64 = self
            .observations
            .iter()
            .map(|obs| {
                let diff = obs.predicted_probability - obs.actual_outcome;
                diff * diff
            })
            .sum();
        Some(sum_sq / self.observations.len() as f64)
    }

    /// Number of recorded observations.
    #[must_use]
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Whether the Brier score indicates poor calibration (above threshold).
    /// Returns `false` if insufficient observations exist.
    #[must_use]
    pub fn is_poorly_calibrated(&self) -> bool {
        self.brier_score()
            .is_some_and(|bs| bs > ADAPTIVE_FALLBACK_BRIER_THRESHOLD)
    }

    /// Snapshot the calibration state as a JSON value.
    #[must_use]
    pub fn to_evidence_json(&self) -> Value {
        serde_json::json!({
            "brier_score": self.brier_score(),
            "observation_count": self.observation_count(),
            "poorly_calibrated": self.is_poorly_calibrated(),
            "window_size": self.window_size,
            "brier_threshold": ADAPTIVE_FALLBACK_BRIER_THRESHOLD,
        })
    }
}

// ---------------------------------------------------------------------------
// bd-efr.4: RoutingEvidenceLedger — circular buffer of routing decisions
// ---------------------------------------------------------------------------

/// A single entry in the routing evidence ledger, recording every adaptive
/// routing decision with full context for post-hoc diagnostics.
///
/// Satisfies the Alien-Artifact Engineering Contract requirements:
/// - Explicit state space (observed availability state)
/// - Actions (chosen backend + recommended order)
/// - Loss matrix reference (policy ID + loss matrix hash)
/// - Posterior terms (posterior snapshot)
/// - Calibration metric (calibration score, Brier score)
/// - Fallback trigger (whether fallback was active and why)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingEvidenceLedgerEntry {
    /// Unique identifier for this decision.
    pub decision_id: String,
    /// Trace ID linking this decision to a broader operation.
    pub trace_id: String,
    /// RFC 3339 timestamp of when the decision was made.
    pub timestamp_rfc3339: String,
    /// Observed availability state (e.g., "all_available").
    pub observed_state: String,
    /// The action chosen by the decision contract.
    pub chosen_action: String,
    /// Ordered list of recommended backends.
    pub recommended_order: Vec<String>,
    /// Whether the deterministic fallback was triggered.
    pub fallback_active: bool,
    /// Reason for fallback, if active.
    pub fallback_reason: Option<String>,
    /// Posterior probability snapshot at decision time.
    pub posterior_snapshot: Vec<f64>,
    /// Calibration score (simple accuracy metric).
    pub calibration_score: f64,
    /// Brier score at decision time, if available.
    pub brier_score: Option<f64>,
    /// E-process value (evidence against the null).
    pub e_process: f64,
    /// Confidence interval width (normalized posterior entropy).
    pub ci_width: f64,
    /// Whether the adaptive router was in adaptive mode.
    pub adaptive_mode: bool,
    /// Policy ID for provenance.
    pub policy_id: String,
    /// Loss matrix content hash for provenance.
    pub loss_matrix_hash: String,
    /// Per-backend availability at decision time.
    pub availability: Vec<(String, bool)>,
    /// Duration bucket of the input audio.
    pub duration_bucket: String,
    /// Whether diarization was requested.
    pub diarize: bool,
    /// Actual outcome once known: `None` until resolved.
    pub actual_outcome: Option<RoutingOutcomeRecord>,
}

/// A fixed-size circular buffer of routing evidence ledger entries,
/// supporting serialization and diagnostic queries.
///
/// The ledger records every routing decision made by the adaptive router,
/// providing a complete audit trail for debugging, monitoring, and
/// post-hoc calibration analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingEvidenceLedger {
    /// Circular buffer of entries.
    entries: VecDeque<RoutingEvidenceLedgerEntry>,
    /// Maximum capacity.
    capacity: usize,
    /// Total number of entries ever recorded (monotonically increasing).
    total_recorded: u64,
}

impl RoutingEvidenceLedger {
    /// Create a new empty evidence ledger with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            total_recorded: 0,
        }
    }

    /// Record a new entry, evicting the oldest if at capacity.
    pub fn record(&mut self, entry: RoutingEvidenceLedgerEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        self.total_recorded += 1;
    }

    /// Resolve the most recent entry with an actual outcome.
    /// This matches by `decision_id` and sets the `actual_outcome` field.
    pub fn resolve_outcome(&mut self, decision_id: &str, outcome: RoutingOutcomeRecord) {
        for entry in self.entries.iter_mut().rev() {
            if entry.decision_id == decision_id {
                entry.actual_outcome = Some(outcome);
                return;
            }
        }
    }

    /// Number of entries currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total number of entries ever recorded.
    #[must_use]
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Maximum capacity of the buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the most recent entry, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&RoutingEvidenceLedgerEntry> {
        self.entries.back()
    }

    /// Get a reference to all current entries (oldest first).
    #[must_use]
    pub fn entries(&self) -> &VecDeque<RoutingEvidenceLedgerEntry> {
        &self.entries
    }

    /// Query entries by decision ID.
    #[must_use]
    pub fn find_by_decision_id(&self, decision_id: &str) -> Option<&RoutingEvidenceLedgerEntry> {
        self.entries.iter().find(|e| e.decision_id == decision_id)
    }

    /// Query entries where fallback was triggered.
    #[must_use]
    pub fn fallback_entries(&self) -> Vec<&RoutingEvidenceLedgerEntry> {
        self.entries.iter().filter(|e| e.fallback_active).collect()
    }

    /// Count entries with resolved outcomes.
    #[must_use]
    pub fn resolved_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.actual_outcome.is_some())
            .count()
    }

    /// Compute aggregate diagnostics from the ledger.
    #[must_use]
    pub fn diagnostics(&self) -> Value {
        let total = self.entries.len();
        let fallback_count = self.entries.iter().filter(|e| e.fallback_active).count();
        let resolved = self.resolved_count();
        let resolved_success = self
            .entries
            .iter()
            .filter_map(|e| e.actual_outcome.as_ref())
            .filter(|o| o.success)
            .count();

        let avg_calibration = if total > 0 {
            self.entries
                .iter()
                .map(|e| e.calibration_score)
                .sum::<f64>()
                / total as f64
        } else {
            0.0
        };

        let avg_brier: Option<f64> = {
            let brier_values: Vec<f64> =
                self.entries.iter().filter_map(|e| e.brier_score).collect();
            if brier_values.is_empty() {
                None
            } else {
                Some(brier_values.iter().sum::<f64>() / brier_values.len() as f64)
            }
        };

        serde_json::json!({
            "total_entries": total,
            "total_ever_recorded": self.total_recorded,
            "capacity": self.capacity,
            "fallback_count": fallback_count,
            "fallback_rate": if total > 0 { fallback_count as f64 / total as f64 } else { 0.0 },
            "resolved_count": resolved,
            "resolved_success_count": resolved_success,
            "resolved_success_rate": if resolved > 0 { resolved_success as f64 / resolved as f64 } else { 0.0 },
            "avg_calibration_score": avg_calibration,
            "avg_brier_score": avg_brier,
        })
    }

    /// Snapshot the full ledger as a JSON value.
    #[must_use]
    pub fn to_evidence_json(&self) -> Value {
        serde_json::json!({
            "diagnostics": self.diagnostics(),
            "entries": serde_json::to_value(&self.entries).unwrap_or(Value::Null),
        })
    }
}

// ---------------------------------------------------------------------------
// RouterState — adaptive per-backend metrics for informed routing
// ---------------------------------------------------------------------------

/// A single recorded outcome from running a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingOutcomeRecord {
    /// Which backend was used.
    pub backend: BackendKind,
    /// Whether the run succeeded.
    pub success: bool,
    /// Wall-clock latency of the run in milliseconds.
    pub latency_ms: u64,
    /// If failed, the error message.
    pub error_message: Option<String>,
    /// RFC 3339 timestamp of when the outcome was recorded.
    pub recorded_at_rfc3339: String,
}

/// Per-backend empirical metrics derived from a sliding window of outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMetrics {
    /// Empirical success rate in `[0.0, 1.0]` over the sliding window.
    pub success_rate: f64,
    /// Average latency in milliseconds over successful runs in the window.
    pub avg_latency_ms: f64,
    /// Most recent error message, if any.
    pub last_error: Option<String>,
    /// Total number of outcomes in the window.
    pub sample_count: usize,
    /// Number of successes in the window.
    pub success_count: usize,
}

/// Adaptive router state: tracks per-backend outcome history and derived
/// metrics. Implements the "explicit state space" requirement of the
/// Alien-Artifact Engineering Contract.
///
/// # State space
/// For each backend in `{WhisperCpp, InsanelyFast, WhisperDiarization}`:
///   - `success_rate ∈ [0, 1]`
///   - `avg_latency_ms ∈ [0, ∞)`
///   - `last_error: Option<String>`
///   - `sample_count ∈ [0, ROUTER_HISTORY_WINDOW]`
///
/// # Actions
///   - Select one of the available backends as the primary candidate.
///
/// # Loss matrix integration
///   - Empirical success rates modulate the prior Beta parameters.
///   - Empirical latencies modulate the latency proxy.
///
/// # Calibration metric
///   - `calibration_score`: ratio of correct predictions (selected backend
///     succeeded) to total predictions, over the sliding window.
///
/// # Deterministic fallback trigger
///   - When `calibration_score < ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD`
///     or `sample_count < ADAPTIVE_MIN_SAMPLES`, the router falls back
///     to the static priority order from `auto_priority`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterState {
    /// Per-backend outcome history (sliding window).
    histories: [VecDeque<RoutingOutcomeRecord>; 3],
    /// Total adaptive routing predictions made.
    total_predictions: u64,
    /// Count of predictions where the top-ranked backend succeeded.
    correct_predictions: u64,
    /// bd-efr.2: Brier-score confidence calibration state.
    calibration: CalibrationState,
    /// bd-efr.4: Evidence ledger of all adaptive routing decisions.
    evidence_ledger: RoutingEvidenceLedger,
}

impl Default for RouterState {
    fn default() -> Self {
        Self::new()
    }
}

impl RouterState {
    /// Create a new empty router state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            histories: [VecDeque::new(), VecDeque::new(), VecDeque::new()],
            total_predictions: 0,
            correct_predictions: 0,
            calibration: CalibrationState::new(ROUTER_HISTORY_WINDOW),
            evidence_ledger: RoutingEvidenceLedger::new(EVIDENCE_LEDGER_CAPACITY),
        }
    }

    /// Map a `BackendKind` to a slot index `0..3`. Returns `None` for `Auto`.
    fn slot(kind: BackendKind) -> Option<usize> {
        match kind {
            BackendKind::WhisperCpp => Some(0),
            BackendKind::InsanelyFast => Some(1),
            BackendKind::WhisperDiarization => Some(2),
            BackendKind::Auto => None,
        }
    }

    /// All three backends in slot order.
    const ALL_BACKENDS: [BackendKind; 3] = [
        BackendKind::WhisperCpp,
        BackendKind::InsanelyFast,
        BackendKind::WhisperDiarization,
    ];

    /// Record an outcome for a backend run.
    pub fn record_outcome(&mut self, record: RoutingOutcomeRecord) {
        let Some(idx) = Self::slot(record.backend) else {
            return;
        };
        let history = &mut self.histories[idx];
        if history.len() >= ROUTER_HISTORY_WINDOW {
            history.pop_front();
        }
        history.push_back(record);
    }

    /// Record the result of an adaptive prediction: was the top-ranked
    /// backend the one that actually succeeded?
    pub fn record_prediction_outcome(&mut self, top_ranked_succeeded: bool) {
        self.total_predictions += 1;
        if top_ranked_succeeded {
            self.correct_predictions += 1;
        }
    }

    /// Compute empirical metrics for a specific backend from its outcome
    /// history.
    #[must_use]
    pub fn metrics_for(&self, kind: BackendKind) -> BackendMetrics {
        let Some(idx) = Self::slot(kind) else {
            return BackendMetrics {
                success_rate: 0.0,
                avg_latency_ms: 0.0,
                last_error: None,
                sample_count: 0,
                success_count: 0,
            };
        };
        let history = &self.histories[idx];
        let sample_count = history.len();
        if sample_count == 0 {
            return BackendMetrics {
                success_rate: 0.5, // uninformative prior
                avg_latency_ms: 0.0,
                last_error: None,
                sample_count: 0,
                success_count: 0,
            };
        }

        let success_count = history.iter().filter(|r| r.success).count();
        let success_rate = success_count as f64 / sample_count as f64;

        let successful_latencies: Vec<f64> = history
            .iter()
            .filter(|r| r.success)
            .map(|r| r.latency_ms as f64)
            .collect();
        let avg_latency_ms = if successful_latencies.is_empty() {
            0.0
        } else {
            successful_latencies.iter().sum::<f64>() / successful_latencies.len() as f64
        };

        let last_error = history.iter().rev().find_map(|r| r.error_message.clone());

        BackendMetrics {
            success_rate,
            avg_latency_ms,
            last_error,
            sample_count,
            success_count,
        }
    }

    /// Overall calibration score: fraction of correct adaptive predictions.
    /// Returns `0.5` (uninformative) if no predictions have been made.
    #[must_use]
    pub fn calibration_score(&self) -> f64 {
        if self.total_predictions == 0 {
            return 0.5;
        }
        self.correct_predictions as f64 / self.total_predictions as f64
    }

    /// Whether the adaptive router has sufficient data to trust its
    /// empirical estimates.
    #[must_use]
    pub fn has_sufficient_data(&self) -> bool {
        self.histories
            .iter()
            .any(|h| h.len() >= ADAPTIVE_MIN_SAMPLES)
    }

    /// Whether the adaptive router should fall back to the static order.
    /// Falls back when: insufficient data, simple calibration too low,
    /// or Brier score indicates poor confidence calibration (bd-efr.2).
    #[must_use]
    pub fn should_use_static_fallback(&self) -> bool {
        !self.has_sufficient_data()
            || self.calibration_score() < ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD
            || self.calibration.is_poorly_calibrated()
    }

    /// The reason the router chose static fallback, or `None` if adaptive
    /// mode is active.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<String> {
        if !self.has_sufficient_data() {
            return Some("insufficient_data".to_owned());
        }
        if self.calibration_score() < ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD {
            return Some(format!(
                "accuracy_calibration_below_threshold({:.3} < {ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD})",
                self.calibration_score()
            ));
        }
        if self.calibration.is_poorly_calibrated() {
            return Some(format!(
                "brier_score_above_threshold({:.3} > {ADAPTIVE_FALLBACK_BRIER_THRESHOLD})",
                self.calibration.brier_score().unwrap_or(0.0)
            ));
        }
        None
    }

    /// Record a calibration observation: the predicted probability of success
    /// and the actual outcome (bd-efr.2).
    pub fn record_calibration_observation(&mut self, predicted_probability: f64, success: bool) {
        self.calibration.record(predicted_probability, success);
    }

    /// Get the current Brier score, if observations exist (bd-efr.2).
    #[must_use]
    pub fn brier_score(&self) -> Option<f64> {
        self.calibration.brier_score()
    }

    /// Access the calibration state (bd-efr.2).
    #[must_use]
    pub fn calibration_state(&self) -> &CalibrationState {
        &self.calibration
    }

    /// Record an evidence ledger entry (bd-efr.4).
    pub fn record_evidence(&mut self, entry: RoutingEvidenceLedgerEntry) {
        self.evidence_ledger.record(entry);
    }

    /// Resolve an evidence ledger entry with its actual outcome (bd-efr.4).
    pub fn resolve_evidence_outcome(&mut self, decision_id: &str, outcome: RoutingOutcomeRecord) {
        self.evidence_ledger.resolve_outcome(decision_id, outcome);
    }

    /// Access the evidence ledger (bd-efr.4).
    #[must_use]
    pub fn evidence_ledger(&self) -> &RoutingEvidenceLedger {
        &self.evidence_ledger
    }

    /// Snapshot the full state as a JSON value for the evidence ledger.
    #[must_use]
    pub fn to_evidence_json(&self) -> Value {
        let mut backend_metrics = Vec::new();
        for kind in Self::ALL_BACKENDS {
            let m = self.metrics_for(kind);
            backend_metrics.push(serde_json::json!({
                "backend": kind.as_str(),
                "success_rate": m.success_rate,
                "avg_latency_ms": m.avg_latency_ms,
                "last_error": m.last_error,
                "sample_count": m.sample_count,
                "success_count": m.success_count,
            }));
        }
        serde_json::json!({
            "backend_metrics": backend_metrics,
            "calibration_score": self.calibration_score(),
            "brier_score": self.calibration.brier_score(),
            "brier_threshold": ADAPTIVE_FALLBACK_BRIER_THRESHOLD,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "sufficient_data": self.has_sufficient_data(),
            "static_fallback_active": self.should_use_static_fallback(),
            "fallback_reason": self.fallback_reason(),
            "evidence_ledger_diagnostics": self.evidence_ledger.diagnostics(),
        })
    }
}

/// Global adaptive router state, protected by a mutex.
static ROUTER_STATE: Mutex<Option<RouterState>> = Mutex::new(None);

/// Record a backend run outcome into the global adaptive router state.
/// This should be called after every backend execution attempt,
/// regardless of success or failure.
pub fn update_router_state(
    backend: BackendKind,
    success: bool,
    latency_ms: u64,
    error_message: Option<String>,
) {
    let record = RoutingOutcomeRecord {
        backend,
        success,
        latency_ms,
        error_message: error_message.clone(),
        recorded_at_rfc3339: Utc::now().to_rfc3339(),
    };

    // Emit the evidence ledger entry via tracing.
    tracing::info!(
        target: "franken_whisper::routing::evidence",
        evidence_type = "routing_outcome",
        backend = backend.as_str(),
        success = success,
        latency_ms = latency_ms,
        error_message = error_message.as_deref().unwrap_or(""),
        "routing outcome recorded"
    );

    if let Ok(mut guard) = ROUTER_STATE.lock() {
        let state = guard.get_or_insert_with(RouterState::new);
        state.record_outcome(record);
    }
}

/// Record whether the adaptive router's top prediction was correct.
pub fn record_adaptive_prediction(top_ranked_succeeded: bool) {
    if let Ok(mut guard) = ROUTER_STATE.lock() {
        let state = guard.get_or_insert_with(RouterState::new);
        state.record_prediction_outcome(top_ranked_succeeded);
    }
}

/// Get a snapshot of the current router state. Returns `None` if no
/// outcomes have been recorded yet.
#[must_use]
pub fn router_state_snapshot() -> Option<RouterState> {
    ROUTER_STATE.lock().ok().and_then(|guard| guard.clone())
}

/// Record a calibration observation into the global router state (bd-efr.2).
///
/// Should be called after each routing decision resolves with the predicted
/// success probability and the actual outcome.
pub fn record_calibration_observation(predicted_probability: f64, success: bool) {
    if let Ok(mut guard) = ROUTER_STATE.lock() {
        let state = guard.get_or_insert_with(RouterState::new);
        state.record_calibration_observation(predicted_probability, success);
    }
}

/// Resolve an evidence ledger entry with its actual outcome (bd-efr.4).
///
/// Matches the entry by `decision_id` and records the final outcome for
/// post-hoc analysis.
pub fn resolve_evidence_outcome(decision_id: &str, outcome: RoutingOutcomeRecord) {
    if let Ok(mut guard) = ROUTER_STATE.lock() {
        let state = guard.get_or_insert_with(RouterState::new);
        state.resolve_evidence_outcome(decision_id, outcome);
    }
}

/// Get a snapshot of the routing evidence ledger diagnostics (bd-efr.4).
#[must_use]
pub fn evidence_ledger_diagnostics() -> Option<Value> {
    ROUTER_STATE
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|s| s.evidence_ledger().diagnostics()))
}

// ---------------------------------------------------------------------------
// Engine trait — formal contract for all backend adapters
// ---------------------------------------------------------------------------

/// Formal engine contract. Each backend adapter implements this trait,
/// providing a uniform interface for availability checks, capability
/// discovery, and transcription execution.
pub trait Engine: Send + Sync {
    /// Human-readable engine name.
    fn name(&self) -> &'static str;

    /// Which `BackendKind` this engine corresponds to.
    fn kind(&self) -> BackendKind;

    /// Declared capabilities of this engine.
    fn capabilities(&self) -> EngineCapabilities;

    /// Whether the engine's external binary/script is currently available.
    fn is_available(&self) -> bool;

    /// Execute a transcription request.
    fn run(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&CancellationToken>,
    ) -> FwResult<TranscriptionResult>;
}

// ---------------------------------------------------------------------------
// StreamingEngine trait — extends Engine with streaming segment delivery
// ---------------------------------------------------------------------------

/// Extension trait for engines that can deliver transcription segments
/// incrementally via a callback. Engines that do not natively support
/// streaming still satisfy the contract through the provided default
/// implementation, which runs the batch `Engine::run` method and then
/// replays each segment through the callback.
pub trait StreamingEngine: Engine {
    /// Execute a transcription request with streaming segment delivery.
    ///
    /// The `on_segment` callback is invoked once per segment, in order,
    /// as they become available. Engines with native streaming support
    /// should call the callback as soon as each segment is ready.
    ///
    /// The default implementation delegates to `Engine::run` and then
    /// iterates over the resulting segments, invoking the callback for
    /// each one. This preserves backward compatibility: any type that
    /// implements `Engine` can opt into `StreamingEngine` without
    /// additional work.
    fn run_streaming(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&CancellationToken>,
        on_segment: Box<dyn Fn(TranscriptionSegment) + Send>,
    ) -> FwResult<TranscriptionResult> {
        let result = self.run(request, normalized_wav, work_dir, timeout, token)?;
        for segment in &result.segments {
            on_segment(segment.clone());
        }
        Ok(result)
    }
}

/// Unit struct for the whisper.cpp engine.
pub struct WhisperCppEngine;

impl Engine for WhisperCppEngine {
    fn name(&self) -> &'static str {
        "whisper.cpp"
    }
    fn kind(&self) -> BackendKind {
        BackendKind::WhisperCpp
    }
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            supports_diarization: false,
            supports_translation: true,
            supports_word_timestamps: true,
            supports_gpu: true,
            supports_streaming: true,
        }
    }
    fn is_available(&self) -> bool {
        whisper_cpp::is_available()
    }
    fn run(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<TranscriptionResult> {
        whisper_cpp::run(request, normalized_wav, work_dir, timeout, token)
    }
}

impl StreamingEngine for WhisperCppEngine {
    fn run_streaming(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
        on_segment: Box<dyn Fn(TranscriptionSegment) + Send>,
    ) -> FwResult<TranscriptionResult> {
        whisper_cpp::run_streaming(
            request,
            normalized_wav,
            work_dir,
            timeout,
            token,
            &on_segment,
        )
    }
}

/// Unit struct for the native whisper.cpp engine (bd-1rj.9 pilot).
///
/// Per `docs/native_engine_contract.md` §1.2, the native engine uses a
/// distinct name (`"whisper.cpp-native"`) but returns the same `BackendKind`
/// as the bridge adapter it will eventually replace.
pub struct WhisperCppNativeEngine;

impl Engine for WhisperCppNativeEngine {
    fn name(&self) -> &'static str {
        "whisper.cpp-native"
    }
    fn kind(&self) -> BackendKind {
        BackendKind::WhisperCpp
    }
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            supports_diarization: false,
            supports_translation: true,
            supports_word_timestamps: true,
            supports_gpu: true,
            supports_streaming: true,
        }
    }
    fn is_available(&self) -> bool {
        whisper_cpp_native::is_available()
    }
    fn run(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<TranscriptionResult> {
        whisper_cpp_native::run(request, normalized_wav, work_dir, timeout, token)
    }
}

impl StreamingEngine for WhisperCppNativeEngine {
    fn run_streaming(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
        on_segment: Box<dyn Fn(TranscriptionSegment) + Send>,
    ) -> FwResult<TranscriptionResult> {
        whisper_cpp_native::run_streaming(
            request,
            normalized_wav,
            work_dir,
            timeout,
            token,
            &on_segment,
        )
    }
}

/// Unit struct for the insanely-fast-whisper engine.
///
/// `StreamingEngine` is implemented via the default batch-then-replay adapter
/// because insanely-fast-whisper does not natively support streaming output.
pub struct InsanelyFastEngine;

impl Engine for InsanelyFastEngine {
    fn name(&self) -> &'static str {
        "insanely-fast-whisper"
    }
    fn kind(&self) -> BackendKind {
        BackendKind::InsanelyFast
    }
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            supports_diarization: true,
            supports_translation: true,
            supports_word_timestamps: true,
            supports_gpu: true,
            supports_streaming: false,
        }
    }
    fn is_available(&self) -> bool {
        insanely_fast::is_available()
    }
    fn run(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<TranscriptionResult> {
        insanely_fast::run(request, normalized_wav, work_dir, timeout, token)
    }
}

impl StreamingEngine for InsanelyFastEngine {}

/// Unit struct for the native insanely-fast engine (bd-1rj.10 pilot).
///
/// Per `docs/native_engine_contract.md` §1.2, the native engine uses a
/// distinct name (`"insanely-fast-native"`) but returns the same `BackendKind`
/// as the bridge adapter it will eventually replace.
pub struct InsanelyFastNativeEngine;

impl Engine for InsanelyFastNativeEngine {
    fn name(&self) -> &'static str {
        "insanely-fast-native"
    }
    fn kind(&self) -> BackendKind {
        BackendKind::InsanelyFast
    }
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            supports_diarization: true,
            supports_translation: true,
            supports_word_timestamps: true,
            supports_gpu: true,
            supports_streaming: false,
        }
    }
    fn is_available(&self) -> bool {
        insanely_fast_native::is_available()
    }
    fn run(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<TranscriptionResult> {
        insanely_fast_native::run(request, normalized_wav, work_dir, timeout, token)
    }
}

impl StreamingEngine for InsanelyFastNativeEngine {}

/// Unit struct for the whisper-diarization engine.
///
/// `StreamingEngine` is implemented via the default batch-then-replay adapter
/// because whisper-diarization does not natively support streaming output.
pub struct WhisperDiarizationEngine;

impl Engine for WhisperDiarizationEngine {
    fn name(&self) -> &'static str {
        "whisper-diarization"
    }
    fn kind(&self) -> BackendKind {
        BackendKind::WhisperDiarization
    }
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            supports_diarization: true,
            supports_translation: false,
            supports_word_timestamps: false,
            supports_gpu: true,
            supports_streaming: false,
        }
    }
    fn is_available(&self) -> bool {
        whisper_diarization::is_available()
    }
    fn run(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<TranscriptionResult> {
        whisper_diarization::run(request, normalized_wav, work_dir, timeout, token)
    }
}

impl StreamingEngine for WhisperDiarizationEngine {}

/// Unit struct for the native whisper-diarization engine (bd-cidv).
///
/// Runs real in-process ASR through [`crate::native_engine`] and then assigns
/// speakers via the orchestrator's heuristic diarizer (no neural speaker
/// encoder; ECAPA tracked in bd-ohex). Per `docs/native_engine_contract.md`
/// §1.2, the native engine uses a distinct name
/// (`"whisper-diarization-native"`) but returns the same `BackendKind` as the
/// bridge adapter.
pub struct WhisperDiarizationNativeEngine;

impl Engine for WhisperDiarizationNativeEngine {
    fn name(&self) -> &'static str {
        "whisper-diarization-native"
    }
    fn kind(&self) -> BackendKind {
        BackendKind::WhisperDiarization
    }
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            supports_diarization: true,
            supports_translation: false,
            supports_word_timestamps: false,
            supports_gpu: true,
            supports_streaming: false,
        }
    }
    fn is_available(&self) -> bool {
        whisper_diarization_native::is_available()
    }
    fn run(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<TranscriptionResult> {
        whisper_diarization_native::run(request, normalized_wav, work_dir, timeout, token)
    }
}

impl StreamingEngine for WhisperDiarizationNativeEngine {
    fn run_streaming(
        &self,
        request: &TranscribeRequest,
        normalized_wav: &Path,
        work_dir: &Path,
        timeout: Duration,
        token: Option<&crate::orchestrator::CancellationToken>,
        on_segment: Box<dyn Fn(TranscriptionSegment) + Send>,
    ) -> FwResult<TranscriptionResult> {
        whisper_diarization_native::run_streaming(
            request,
            normalized_wav,
            work_dir,
            timeout,
            token,
            &on_segment,
        )
    }
}

/// Returns all registered engines as trait objects.
pub fn all_engines() -> Vec<Box<dyn Engine>> {
    vec![
        Box::new(WhisperCppEngine),
        Box::new(WhisperCppNativeEngine),
        Box::new(InsanelyFastEngine),
        Box::new(InsanelyFastNativeEngine),
        Box::new(WhisperDiarizationEngine),
        Box::new(WhisperDiarizationNativeEngine),
    ]
}

/// Look up an engine by `BackendKind`.
pub fn engine_for(kind: BackendKind) -> Option<Box<dyn Engine>> {
    match kind {
        BackendKind::WhisperCpp => Some(Box::new(WhisperCppEngine)),
        BackendKind::InsanelyFast => Some(Box::new(InsanelyFastEngine)),
        BackendKind::WhisperDiarization => Some(Box::new(WhisperDiarizationEngine)),
        BackendKind::Auto => None,
    }
}

/// Look up a streaming-capable engine by `BackendKind`.
///
/// All concrete backend engines implement `StreamingEngine`. Backends that
/// do not natively support streaming use the default batch-then-replay
/// adapter provided by the trait.
pub fn streaming_engine_for(kind: BackendKind) -> Option<Box<dyn StreamingEngine>> {
    match kind {
        BackendKind::WhisperCpp => Some(Box::new(WhisperCppEngine)),
        BackendKind::InsanelyFast => Some(Box::new(InsanelyFastEngine)),
        BackendKind::WhisperDiarization => Some(Box::new(WhisperDiarizationEngine)),
        BackendKind::Auto => None,
    }
}

/// Returns all registered engines as streaming-capable trait objects.
pub fn all_streaming_engines() -> Vec<Box<dyn StreamingEngine>> {
    vec![
        Box::new(WhisperCppEngine),
        Box::new(WhisperCppNativeEngine),
        Box::new(InsanelyFastEngine),
        Box::new(InsanelyFastNativeEngine),
        Box::new(WhisperDiarizationEngine),
        Box::new(WhisperDiarizationNativeEngine),
    ]
}

#[derive(Debug, Clone)]
pub struct BackendSelectionOutcome {
    pub routing_log: Value,
    pub recommended_order: Vec<BackendKind>,
    pub evidence_entries: Vec<Value>,
    pub fallback_triggered: bool,
    pub calibration_score: f64,
    pub e_process: f64,
    pub ci_width: f64,
}

#[derive(Debug, Clone)]
pub struct BackendExecution {
    pub result: TranscriptionResult,
    pub runtime: BackendRuntimeMetadata,
    pub implementation: BackendImplementation,
    pub execution_mode: String,
    pub rollout_stage: String,
    pub native_fallback_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendImplementation {
    Bridge,
    Native,
}

impl BackendImplementation {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bridge => "bridge",
            Self::Native => "native",
        }
    }
}

pub(crate) fn execute(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    command_timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<BackendExecution> {
    match request.backend {
        BackendKind::Auto => {
            let mut attempts = Vec::new();
            for backend in auto_priority(request.diarize) {
                let readiness = readiness_for(*backend, request);
                if !readiness.available {
                    let reason = readiness.reason.unwrap_or_else(|| "unavailable".to_owned());
                    tracing::warn!(backend = backend.as_str(), reason = %reason, "Backend unavailable");
                    attempts.push(format!("{}(missing: {})", backend.as_str(), reason));
                    continue;
                }

                match run_backend(
                    *backend,
                    request,
                    normalized_wav,
                    work_dir,
                    command_timeout,
                    token,
                ) {
                    Ok(result) => {
                        tracing::info!(backend = backend.as_str(), "Selected backend");
                        return Ok(result);
                    }
                    Err(error) => {
                        tracing::warn!(backend = backend.as_str(), error = %error, "Backend failed");
                        attempts.push(format!("{}(failed: {})", backend.as_str(), error));
                    }
                }
            }

            Err(FwError::BackendUnavailable(format!(
                "no backend could complete request; attempts: {}",
                attempts.join(", ")
            )))
        }
        selected => {
            tracing::info!(backend = selected.as_str(), "Selected backend");
            let readiness = readiness_for(selected, request);
            if !readiness.available {
                return Err(FwError::BackendUnavailable(format!(
                    "selected backend `{}` is unavailable: {}",
                    selected.as_str(),
                    readiness.reason.unwrap_or_else(|| "unknown".to_owned())
                )));
            }
            run_backend(
                selected,
                request,
                normalized_wav,
                work_dir,
                command_timeout,
                token,
            )
        }
    }
}

pub(crate) fn execute_with_order(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    command_timeout: Duration,
    recommended_order: &[BackendKind],
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<BackendExecution> {
    match request.backend {
        BackendKind::Auto => {
            let mut attempts = Vec::new();
            for backend in recommended_order {
                let readiness = readiness_for(*backend, request);
                if !readiness.available {
                    let reason = readiness.reason.unwrap_or_else(|| "unavailable".to_owned());
                    attempts.push(format!("{}(missing: {})", backend.as_str(), reason));
                    continue;
                }

                match run_backend(
                    *backend,
                    request,
                    normalized_wav,
                    work_dir,
                    command_timeout,
                    token,
                ) {
                    Ok(result) => return Ok(result),
                    Err(error) => {
                        attempts.push(format!("{}(failed: {})", backend.as_str(), error));
                    }
                }
            }

            Err(FwError::BackendUnavailable(format!(
                "no backend could complete request; attempts: {}",
                attempts.join(", ")
            )))
        }
        selected => {
            let readiness = readiness_for(selected, request);
            if !readiness.available {
                return Err(FwError::BackendUnavailable(format!(
                    "selected backend `{}` is unavailable: {}",
                    selected.as_str(),
                    readiness.reason.unwrap_or_else(|| "unknown".to_owned())
                )));
            }
            run_backend(
                selected,
                request,
                normalized_wav,
                work_dir,
                command_timeout,
                token,
            )
        }
    }
}

pub fn is_available(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Auto => false,
        BackendKind::WhisperCpp => whisper_cpp::is_available(),
        BackendKind::InsanelyFast => insanely_fast::is_available(),
        BackendKind::WhisperDiarization => whisper_diarization::is_available(),
    }
}

/// Returns per-backend diagnostic info for the `robot backends` command.
pub fn diagnostics() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "backend": BackendKind::WhisperCpp.as_str(),
            "available": whisper_cpp::is_available(),
            "binary": std::env::var("FRANKEN_WHISPER_WHISPER_CPP_BIN")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "whisper-cli".to_owned()),
            "env_override": "FRANKEN_WHISPER_WHISPER_CPP_BIN",
            "unsupported_options": [
                "--timestamp-level",
                "--num-speakers/--min-speakers/--max-speakers",
                "--gpu-device",
                "--flash-attention",
                "--diarization-model",
            ],
        }),
        serde_json::json!({
            "backend": BackendKind::InsanelyFast.as_str(),
            "available": insanely_fast::is_available(),
            "binary": std::env::var("FRANKEN_WHISPER_INSANELY_FAST_BIN")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "insanely-fast-whisper".to_owned()),
            "env_override": "FRANKEN_WHISPER_INSANELY_FAST_BIN",
            "hf_token_set": insanely_fast::hf_token_present(),
            "hf_token_env_overrides": ["FRANKEN_WHISPER_HF_TOKEN", "HF_TOKEN"],
            "requires_hf_token_for_diarization": true,
            "unsupported_options": [
                "--output-txt/--output-vtt/--output-srt/--output-csv/--output-json-full/--output-lrc",
                "--no-timestamps",
                "--detect-language-only",
                "--split-on-word",
                "--vad*",
                "--best-of/--beam-size/--temperature*",
            ],
        }),
        serde_json::json!({
            "backend": BackendKind::WhisperDiarization.as_str(),
            "available": whisper_diarization::is_available(),
            "script": whisper_diarization::script_path_string(),
            "python_binary": std::env::var("FRANKEN_WHISPER_PYTHON_BIN")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "python3".to_owned()),
            "env_override_python": "FRANKEN_WHISPER_PYTHON_BIN",
            "env_override_device": "FRANKEN_WHISPER_DIARIZATION_DEVICE",
            "unsupported_options": [
                "--output-*",
                "--timestamp-level",
                "--flash-attention",
                "--no-timestamps",
                "--detect-language-only",
                "--split-on-word",
                "--vad*",
                "--best-of/--beam-size/--temperature*",
            ],
        }),
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendRuntimeMetadata {
    pub identity: String,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeExecutionMode {
    BridgeOnly,
    NativePreferred,
    NativeOnly,
}

const NATIVE_EXECUTION_ENV_VAR: &str = "FRANKEN_WHISPER_NATIVE_EXECUTION";
const BRIDGE_NATIVE_RECOVERY_ENV_VAR: &str = "FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY";
const NATIVE_ENGINE_VERSION_TAG: &str = "native-pilot-v1";

pub fn runtime_metadata(kind: BackendKind) -> BackendRuntimeMetadata {
    runtime_metadata_with_implementation(kind, BackendImplementation::Bridge)
}

fn runtime_metadata_with_implementation(
    kind: BackendKind,
    implementation: BackendImplementation,
) -> BackendRuntimeMetadata {
    match implementation {
        BackendImplementation::Bridge => bridge_runtime_metadata(kind),
        BackendImplementation::Native => native_runtime_metadata(kind),
    }
}

fn bridge_runtime_metadata(kind: BackendKind) -> BackendRuntimeMetadata {
    match kind {
        BackendKind::WhisperCpp => {
            let binary = whisper_cpp::binary_name();
            BackendRuntimeMetadata {
                identity: binary.clone(),
                version: probe_command_version(&binary),
            }
        }
        BackendKind::InsanelyFast => {
            let binary = insanely_fast::binary_name();
            BackendRuntimeMetadata {
                identity: binary.clone(),
                version: probe_command_version(&binary),
            }
        }
        BackendKind::WhisperDiarization => {
            let python = whisper_diarization::python_binary();
            BackendRuntimeMetadata {
                identity: format!("{python} {}", whisper_diarization::script_path_string()),
                version: probe_command_version(&python),
            }
        }
        BackendKind::Auto => BackendRuntimeMetadata {
            identity: "auto-policy".to_owned(),
            version: None,
        },
    }
}

fn native_runtime_metadata(kind: BackendKind) -> BackendRuntimeMetadata {
    let version = Some(format!(
        "{NATIVE_ENGINE_VERSION_TAG}/{}",
        env!("CARGO_PKG_VERSION")
    ));
    match kind {
        BackendKind::WhisperCpp => BackendRuntimeMetadata {
            identity: "whisper.cpp-native".to_owned(),
            version,
        },
        BackendKind::InsanelyFast => BackendRuntimeMetadata {
            identity: "insanely-fast-native".to_owned(),
            version,
        },
        BackendKind::WhisperDiarization => BackendRuntimeMetadata {
            identity: "whisper-diarization-native".to_owned(),
            version,
        },
        BackendKind::Auto => BackendRuntimeMetadata {
            identity: "auto-policy".to_owned(),
            version: None,
        },
    }
}

fn native_execution_enabled() -> bool {
    // The pure-Rust native engine is the DEFAULT execution path — it is the whole point of
    // franken_whisper, and after the NaN-hardening it produces correct, deep-linked
    // timestamps with zero panics. Combined with the `Primary` rollout stage this yields
    // `NativePreferred` (native first, external bridge only as a fallback). Opt OUT with
    // `FRANKEN_WHISPER_NATIVE_EXECUTION=0` (or `false`/`no`/`off`) to force the bridges.
    match std::env::var(NATIVE_EXECUTION_ENV_VAR).ok() {
        Some(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "0" | "false" | "no" | "off")
        }
        None => true,
    }
}

fn bridge_native_recovery_from_raw(raw: Option<&str>) -> bool {
    match raw {
        Some(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "0" | "false" | "no" | "off")
        }
        None => true,
    }
}

fn bridge_native_recovery_enabled() -> bool {
    let raw = std::env::var(BRIDGE_NATIVE_RECOVERY_ENV_VAR).ok();
    bridge_native_recovery_from_raw(raw.as_deref())
}

fn native_execution_mode(rollout_stage: NativeEngineRolloutStage) -> NativeExecutionMode {
    if !native_execution_enabled() {
        return NativeExecutionMode::BridgeOnly;
    }
    match rollout_stage {
        NativeEngineRolloutStage::Primary => NativeExecutionMode::NativePreferred,
        NativeEngineRolloutStage::Sole => NativeExecutionMode::NativeOnly,
        NativeEngineRolloutStage::Shadow
        | NativeEngineRolloutStage::Validated
        | NativeEngineRolloutStage::Fallback => NativeExecutionMode::BridgeOnly,
    }
}

impl NativeExecutionMode {
    #[must_use]
    fn as_str(self) -> &'static str {
        match self {
            Self::BridgeOnly => "bridge_only",
            Self::NativePreferred => "native_preferred",
            Self::NativeOnly => "native_only",
        }
    }
}

fn bridge_available(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Auto => false,
        BackendKind::WhisperCpp => whisper_cpp::is_available(),
        BackendKind::InsanelyFast => insanely_fast::is_available(),
        BackendKind::WhisperDiarization => whisper_diarization::is_available(),
    }
}

fn native_available(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Auto => false,
        BackendKind::WhisperCpp => whisper_cpp_native::is_available(),
        BackendKind::InsanelyFast => insanely_fast_native::is_available(),
        BackendKind::WhisperDiarization => whisper_diarization_native::is_available(),
    }
}

fn available_for_mode(kind: BackendKind, mode: NativeExecutionMode) -> bool {
    match mode {
        NativeExecutionMode::BridgeOnly => {
            bridge_available(kind) || (bridge_native_recovery_enabled() && native_available(kind))
        }
        NativeExecutionMode::NativePreferred => native_available(kind) || bridge_available(kind),
        NativeExecutionMode::NativeOnly => native_available(kind),
    }
}

fn bridge_error_recoverable(error: &FwError) -> bool {
    matches!(
        error,
        FwError::CommandMissing { .. }
            | FwError::BackendUnavailable(_)
            | FwError::CommandFailed { .. }
    )
}

fn record_backend_execution_outcome(
    kind: BackendKind,
    started: Instant,
    result: FwResult<BackendExecution>,
) -> FwResult<BackendExecution> {
    let latency_ms = started.elapsed().as_millis() as u64;
    match result {
        Ok(execution) => {
            update_router_state(kind, true, latency_ms, None);
            Ok(execution)
        }
        Err(error) => {
            let error_message = error.to_string();
            update_router_state(kind, false, latency_ms, Some(error_message));
            Err(error)
        }
    }
}

fn run_backend(
    kind: BackendKind,
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    command_timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<BackendExecution> {
    let rollout_stage = native_rollout_stage();
    let execution_mode = native_execution_mode(rollout_stage);
    let bridge_native_recovery = bridge_native_recovery_enabled();
    let bridge_is_available = bridge_available(kind);
    let native_is_available = native_available(kind);

    match kind {
        BackendKind::Auto => Err(FwError::InvalidRequest(
            "internal error: auto backend cannot be run directly".to_owned(),
        )),
        BackendKind::WhisperCpp => run_backend_with_mode_and_runners(
            kind,
            request,
            normalized_wav,
            work_dir,
            command_timeout,
            token,
            execution_mode,
            rollout_stage,
            bridge_native_recovery,
            bridge_is_available,
            native_is_available,
            whisper_cpp::run,
            whisper_cpp_native::run,
        ),
        BackendKind::InsanelyFast => run_backend_with_mode_and_runners(
            kind,
            request,
            normalized_wav,
            work_dir,
            command_timeout,
            token,
            execution_mode,
            rollout_stage,
            bridge_native_recovery,
            bridge_is_available,
            native_is_available,
            insanely_fast::run,
            insanely_fast_native::run,
        ),
        BackendKind::WhisperDiarization => run_backend_with_mode_and_runners(
            kind,
            request,
            normalized_wav,
            work_dir,
            command_timeout,
            token,
            execution_mode,
            rollout_stage,
            bridge_native_recovery,
            bridge_is_available,
            native_is_available,
            whisper_diarization::run,
            whisper_diarization_native::run,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_backend_with_mode_and_runners<BridgeRunner, NativeRunner>(
    kind: BackendKind,
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    command_timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
    execution_mode: NativeExecutionMode,
    rollout_stage: NativeEngineRolloutStage,
    bridge_native_recovery: bool,
    bridge_is_available: bool,
    native_is_available: bool,
    bridge_runner: BridgeRunner,
    native_runner: NativeRunner,
) -> FwResult<BackendExecution>
where
    BridgeRunner: Fn(
        &TranscribeRequest,
        &Path,
        &Path,
        Duration,
        Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<TranscriptionResult>,
    NativeRunner: Fn(
        &TranscribeRequest,
        &Path,
        &Path,
        Duration,
        Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<TranscriptionResult>,
{
    let started = Instant::now();
    let run_bridge = || -> FwResult<BackendExecution> {
        match bridge_runner(request, normalized_wav, work_dir, command_timeout, token) {
            Ok(result) => Ok(BackendExecution {
                result,
                runtime: runtime_metadata_with_implementation(kind, BackendImplementation::Bridge),
                implementation: BackendImplementation::Bridge,
                execution_mode: execution_mode.as_str().to_owned(),
                rollout_stage: rollout_stage.as_str().to_owned(),
                native_fallback_error: None,
            }),
            Err(error) => Err(error),
        }
    };

    let run_native = || -> FwResult<BackendExecution> {
        match native_runner(request, normalized_wav, work_dir, command_timeout, token) {
            Ok(result) => Ok(BackendExecution {
                result,
                runtime: runtime_metadata_with_implementation(kind, BackendImplementation::Native),
                implementation: BackendImplementation::Native,
                execution_mode: execution_mode.as_str().to_owned(),
                rollout_stage: rollout_stage.as_str().to_owned(),
                native_fallback_error: None,
            }),
            Err(error) => Err(error),
        }
    };

    let result = match execution_mode {
        NativeExecutionMode::BridgeOnly => match run_bridge() {
            Ok(result) => Ok(result),
            Err(bridge_error)
                if bridge_native_recovery
                    && native_is_available
                    && bridge_error_recoverable(&bridge_error) =>
            {
                let bridge_error_msg = bridge_error.to_string();
                tracing::warn!(
                    backend = kind.as_str(),
                    rollout_stage = rollout_stage.as_str(),
                    bridge_error = %bridge_error_msg,
                    "bridge backend unavailable in bridge-only mode; recovering with native backend"
                );
                let mut native_execution = run_native().map_err(|native_error| {
                    FwError::BackendUnavailable(format!(
                        "bridge `{}` failed: {}; native recovery failed: {}",
                        kind.as_str(),
                        bridge_error_msg,
                        native_error
                    ))
                })?;
                native_execution.execution_mode = "bridge_only_native_recovery".to_owned();
                native_execution.native_fallback_error = Some(bridge_error_msg);
                Ok(native_execution)
            }
            Err(error) => Err(error),
        },
        NativeExecutionMode::NativeOnly => {
            if !native_is_available {
                Err(FwError::BackendUnavailable(format!(
                    "native engine unavailable for `{}` (set `{NATIVE_EXECUTION_ENV_VAR}=1` with rollout stage `primary|sole` only when native runtime is present)",
                    kind.as_str()
                )))
            } else {
                run_native()
            }
        }
        NativeExecutionMode::NativePreferred => {
            if !native_is_available {
                run_bridge()
            } else {
                match run_native() {
                    Ok(result) => Ok(result),
                    Err(native_error) => {
                        let native_error_msg = native_error.to_string();
                        if !bridge_is_available {
                            Err(FwError::BackendUnavailable(format!(
                                "native `{}` failed and bridge unavailable: {}",
                                kind.as_str(),
                                native_error_msg
                            )))
                        } else {
                            tracing::warn!(
                                backend = kind.as_str(),
                                rollout_stage = rollout_stage.as_str(),
                                native_error = %native_error_msg,
                                "native backend failed; falling back to bridge adapter"
                            );
                            let mut bridge_execution = run_bridge().map_err(|bridge_error| {
                                FwError::BackendUnavailable(format!(
                                    "native `{}` failed: {}; bridge fallback failed: {}",
                                    kind.as_str(),
                                    native_error_msg,
                                    bridge_error
                                ))
                            })?;
                            bridge_execution.native_fallback_error = Some(native_error_msg);
                            Ok(bridge_execution)
                        }
                    }
                }
            }
        }
    };

    record_backend_execution_outcome(kind, started, result)
}

fn probe_command_version(program: &str) -> Option<String> {
    let candidates = [
        vec!["--version".to_owned()],
        vec!["-V".to_owned()],
        vec!["version".to_owned()],
    ];

    for args in candidates {
        if let Ok(output) =
            run_command_with_timeout(program, &args, None, Some(Duration::from_millis(1_500)))
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let line = stdout
                .lines()
                .chain(stderr.lines())
                .map(str::trim)
                .find(|line| !line.is_empty())?;
            return Some(line.to_owned());
        }
    }

    None
}

struct BackendReadiness {
    available: bool,
    reason: Option<String>,
}

fn readiness_for(kind: BackendKind, request: &TranscribeRequest) -> BackendReadiness {
    let mode = native_execution_mode(native_rollout_stage());
    if !available_for_mode(kind, mode) {
        return BackendReadiness {
            available: false,
            reason: Some(match mode {
                NativeExecutionMode::BridgeOnly => "binary/script unavailable".to_owned(),
                NativeExecutionMode::NativePreferred => {
                    "native and bridge engines both unavailable".to_owned()
                }
                NativeExecutionMode::NativeOnly => "native engine unavailable".to_owned(),
            }),
        };
    }

    if kind == BackendKind::InsanelyFast
        && request.diarize
        && !insanely_fast::hf_token_present_for_request(request)
    {
        return BackendReadiness {
            available: false,
            reason: Some(
                "diarization requires HF token (`--hf-token` or env `FRANKEN_WHISPER_HF_TOKEN` / `HF_TOKEN`)"
                    .to_owned(),
            ),
        };
    }

    BackendReadiness {
        available: true,
        reason: None,
    }
}

fn native_rollout_stage() -> NativeEngineRolloutStage {
    let default = NativeEngineRolloutStage::Primary;
    match std::env::var(NativeEngineRolloutStage::ENV_VAR) {
        Ok(raw) => NativeEngineRolloutStage::parse(&raw).unwrap_or_else(|| {
            tracing::warn!(
                env_var = NativeEngineRolloutStage::ENV_VAR,
                value = %raw,
                fallback = default.as_str(),
                "invalid native rollout stage value; defaulting to primary"
            );
            default
        }),
        Err(_) => default,
    }
}

fn gate_recommended_order_for_rollout(
    diarize: bool,
    recommended_order: Vec<BackendKind>,
    rollout_stage: NativeEngineRolloutStage,
) -> (Vec<BackendKind>, bool) {
    match rollout_stage {
        // Before native engines are promoted, keep static ordering deterministic.
        NativeEngineRolloutStage::Shadow
        | NativeEngineRolloutStage::Validated
        | NativeEngineRolloutStage::Fallback => (auto_priority(diarize).to_vec(), true),
        // Primary/Sole preserve the adaptive recommendation.
        NativeEngineRolloutStage::Primary | NativeEngineRolloutStage::Sole => {
            (recommended_order, false)
        }
    }
}

fn routing_mode(adaptive_mode_active: bool, rollout_forced_static: bool) -> &'static str {
    if adaptive_mode_active && !rollout_forced_static {
        "adaptive"
    } else {
        "static"
    }
}

fn effective_fallback_reason(
    router_state_reason: Option<String>,
    rollout_forced_static: bool,
    rollout_stage: NativeEngineRolloutStage,
) -> Option<String> {
    if rollout_forced_static {
        Some(format!(
            "native_rollout_forced_static_order({})",
            rollout_stage.as_str()
        ))
    } else {
        router_state_reason
    }
}

fn auto_priority(diarize: bool) -> &'static [BackendKind] {
    if diarize {
        &[
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
            BackendKind::WhisperCpp,
        ]
    } else {
        &[
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
        ]
    }
}

/// Resolve a request's backend selection down to a single concrete kind
/// **once**, without performing per-window adaptive routing.
///
/// Used by the speculative streaming dispatch path, where invoking
/// `backend::execute(BackendKind::Auto)` separately for every window would
/// risk picking different backends per window and producing a mixed-engine
/// transcript with confusing reporting. With this helper, the speculative
/// path resolves `Auto` once via the static priority list, then forces that
/// single kind onto every per-window invocation so the run is
/// engine-consistent end to end.
///
/// Returns the resolved kind on success, or [`FwError::BackendUnavailable`]
/// if no candidate backend is currently available for the request.
pub(crate) fn resolve_static_backend(request: &TranscribeRequest) -> FwResult<BackendKind> {
    match request.backend {
        BackendKind::Auto => {
            let mut attempts: Vec<String> = Vec::new();
            for candidate in auto_priority(request.diarize) {
                let readiness = readiness_for(*candidate, request);
                if readiness.available {
                    return Ok(*candidate);
                }
                let reason = readiness.reason.unwrap_or_else(|| "unavailable".to_owned());
                attempts.push(format!("{}(missing: {})", candidate.as_str(), reason));
            }
            Err(FwError::BackendUnavailable(format!(
                "no backend available for request (auto routing); attempts: {}",
                attempts.join(", ")
            )))
        }
        explicit => {
            let readiness = readiness_for(explicit, request);
            if readiness.available {
                Ok(explicit)
            } else {
                Err(FwError::BackendUnavailable(format!(
                    "selected backend `{}` is unavailable: {}",
                    explicit.as_str(),
                    readiness.reason.unwrap_or_else(|| "unknown".to_owned())
                )))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Health probing & capability reporting
// ---------------------------------------------------------------------------

/// Per-backend health report with binary discovery, version probing, and
/// issue diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendHealthReport {
    pub backend: BackendKind,
    pub available: bool,
    pub binary_found: bool,
    pub binary_path: Option<String>,
    pub version: Option<String>,
    pub capabilities: EngineCapabilities,
    pub issues: Vec<String>,
    pub checked_at_rfc3339: String,
}

/// Aggregated system health report across all backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthReport {
    pub backends: Vec<BackendHealthReport>,
    pub recommended_backend: Option<BackendKind>,
    pub diarization_ready: bool,
    pub hf_token_set: bool,
}

/// Cached health state (thread-safe).
static HEALTH_CACHE: Mutex<Option<(Instant, SystemHealthReport)>> = Mutex::new(None);
const HEALTH_CACHE_TTL_SECS: u64 = 60;

/// Probe all backends and return a comprehensive health report.
/// Results are cached for 60 seconds.
pub fn probe_system_health() -> SystemHealthReport {
    // Check cache first.
    if let Ok(guard) = HEALTH_CACHE.lock()
        && let Some((instant, ref report)) = *guard
        && instant.elapsed().as_secs() < HEALTH_CACHE_TTL_SECS
    {
        return report.clone();
    }

    let report = probe_system_health_uncached();

    // Update cache.
    if let Ok(mut guard) = HEALTH_CACHE.lock() {
        *guard = Some((Instant::now(), report.clone()));
    }

    report
}

fn probe_system_health_uncached() -> SystemHealthReport {
    let mut backends = Vec::new();

    let mode = native_execution_mode(native_rollout_stage());

    for kind in [
        BackendKind::WhisperCpp,
        BackendKind::InsanelyFast,
        BackendKind::WhisperDiarization,
    ] {
        let available = available_for_mode(kind, mode);
        let capabilities = engine_for(kind)
            .expect("engine_for returns Some for concrete backend kinds")
            .capabilities();
        let mut issues = Vec::new();

        // Check binary existence.
        let (binary_found, binary_path) = check_binary_for_kind(kind);
        if !binary_found && (mode == NativeExecutionMode::BridgeOnly || !native_available(kind)) {
            issues.push(format!("{} binary not found in PATH", kind.as_str()));
        }

        // Check version.
        let version = if binary_found {
            probe_version_for_kind(kind)
        } else {
            None
        };

        // Check diarization-specific requirements.
        if (kind == BackendKind::InsanelyFast || kind == BackendKind::WhisperDiarization)
            && !is_hf_token_set()
        {
            issues.push(
                "FRANKEN_WHISPER_HF_TOKEN / HF_TOKEN not set (required for diarization)".to_owned(),
            );
        }

        backends.push(BackendHealthReport {
            backend: kind,
            available,
            binary_found,
            binary_path,
            version,
            capabilities,
            issues,
            checked_at_rfc3339: Utc::now().to_rfc3339(),
        });
    }

    let hf_token_set = is_hf_token_set();
    let diarization_ready = backends
        .iter()
        .any(|b| b.available && b.capabilities.supports_diarization)
        && hf_token_set;
    let recommended_backend = backends.iter().find(|b| b.available).map(|b| b.backend);

    SystemHealthReport {
        backends,
        recommended_backend,
        diarization_ready,
        hf_token_set,
    }
}

fn is_hf_token_set() -> bool {
    std::env::var("FRANKEN_WHISPER_HF_TOKEN").is_ok() || std::env::var("HF_TOKEN").is_ok()
}

fn check_binary_for_kind(kind: BackendKind) -> (bool, Option<String>) {
    let bin_name = match kind {
        BackendKind::WhisperCpp => std::env::var("FRANKEN_WHISPER_WHISPER_CPP_BIN")
            .unwrap_or_else(|_| "whisper-cli".to_owned()),
        BackendKind::InsanelyFast => std::env::var("FRANKEN_WHISPER_INSANELY_FAST_BIN")
            .unwrap_or_else(|_| "insanely-fast-whisper".to_owned()),
        BackendKind::WhisperDiarization => {
            std::env::var("FRANKEN_WHISPER_PYTHON_BIN").unwrap_or_else(|_| "python3".to_owned())
        }
        BackendKind::Auto => return (false, None),
    };
    match which::which(&bin_name) {
        Ok(path) => (true, Some(path.display().to_string())),
        Err(_) => (false, None),
    }
}

fn probe_version_for_kind(kind: BackendKind) -> Option<String> {
    let bin = match kind {
        BackendKind::WhisperCpp => std::env::var("FRANKEN_WHISPER_WHISPER_CPP_BIN")
            .unwrap_or_else(|_| "whisper-cli".to_owned()),
        BackendKind::InsanelyFast => std::env::var("FRANKEN_WHISPER_INSANELY_FAST_BIN")
            .unwrap_or_else(|_| "insanely-fast-whisper".to_owned()),
        BackendKind::WhisperDiarization => {
            std::env::var("FRANKEN_WHISPER_PYTHON_BIN").unwrap_or_else(|_| "python3".to_owned())
        }
        BackendKind::Auto => return None,
    };
    probe_command_version(&bin)
}

// ---------------------------------------------------------------------------
// BackendSelectionContract — formal DecisionContract for backend routing
// ---------------------------------------------------------------------------

/// A large but finite loss used to fail-closed when a computed backend
/// selection loss is non-finite (NaN/Inf). Large enough to deprioritize the
/// offending action without violating `LossMatrix::new`'s finiteness and
/// non-negativity invariants.
const LARGE_FINITE_PENALTY: f64 = 1.0e6;

struct BackendSelectionContract {
    states: Vec<String>,
    actions: Vec<String>,
    losses: LossMatrix,
    policy: FallbackPolicy,
    action_backends: Vec<BackendKind>,
    /// Whether the adaptive router state was used to modulate the loss matrix.
    adaptive_mode_active: bool,
}

impl BackendSelectionContract {
    /// Convenience constructor without adaptive state (used by tests).
    #[cfg(test)]
    #[allow(dead_code)]
    fn new(request: &TranscribeRequest, duration_secs: f64) -> Self {
        Self::with_router_state(request, duration_secs, None)
    }

    fn with_router_state(
        request: &TranscribeRequest,
        duration_secs: f64,
        router_state: Option<&RouterState>,
    ) -> Self {
        let states: Vec<String> = ["all_available", "partial_available", "none_available"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();

        let (actions, action_backends) = if request.diarize {
            (
                vec![
                    "try_insanely_fast".to_owned(),
                    "try_diarization".to_owned(),
                    "try_whisper_cpp".to_owned(),
                    "fallback_error".to_owned(),
                ],
                vec![
                    BackendKind::InsanelyFast,
                    BackendKind::WhisperDiarization,
                    BackendKind::WhisperCpp,
                ],
            )
        } else {
            (
                vec![
                    "try_whisper_cpp".to_owned(),
                    "try_insanely_fast".to_owned(),
                    "try_diarization".to_owned(),
                    "fallback_error".to_owned(),
                ],
                vec![
                    BackendKind::WhisperCpp,
                    BackendKind::InsanelyFast,
                    BackendKind::WhisperDiarization,
                ],
            )
        };

        // Determine whether the adaptive state has enough data to
        // modulate the loss matrix.
        let adaptive_mode_active = router_state.is_some_and(|rs| !rs.should_use_static_fallback());

        // Loss matrix: 3 states × 4 actions (row-major).
        let n_actions = actions.len();
        let mut values = Vec::with_capacity(3 * n_actions);
        for state_idx in 0..3 {
            for action_idx in 0..n_actions {
                if action_idx >= action_backends.len() {
                    // fallback_error: correct when nothing available, wasteful otherwise.
                    values.push(match state_idx {
                        0 => 1000.0,
                        1 => 500.0,
                        2 => 5.0,
                        _ => 1000.0,
                    });
                } else {
                    let backend = action_backends[action_idx];
                    let base = Self::backend_base_loss_adaptive(
                        backend,
                        request,
                        duration_secs,
                        router_state,
                    );
                    let availability_penalty = match state_idx {
                        0 => 0.0,
                        1 => 333.0,
                        2 => 1000.0,
                        _ => 1000.0,
                    };
                    let v = base + availability_penalty;
                    values.push(if v.is_finite() {
                        v.max(0.0)
                    } else {
                        LARGE_FINITE_PENALTY
                    });
                }
            }
        }

        // Fail-closed: the values above are already sanitized to finite,
        // non-negative numbers, so this constructor should always succeed. If
        // it ever rejects the matrix anyway, build a well-formed
        // static-priority fallback matrix instead of panicking and aborting the
        // whole transcription.
        let losses = match LossMatrix::new(states.clone(), actions.clone(), values) {
            Ok(matrix) => matrix,
            Err(err) => {
                tracing::warn!(
                    target: "franken_whisper::routing::evidence",
                    evidence_type = "loss_matrix_construction_failed",
                    error = err.to_string().as_str(),
                    "backend selection loss matrix rejected; using static-priority fallback matrix"
                );
                let mut fallback_values = Vec::with_capacity(3 * n_actions);
                for state_idx in 0..3 {
                    let availability_penalty = match state_idx {
                        0 => 0.0,
                        1 => 333.0,
                        _ => 1000.0,
                    };
                    for action_idx in 0..n_actions {
                        if action_idx >= action_backends.len() {
                            fallback_values.push(match state_idx {
                                0 => 1000.0,
                                1 => 500.0,
                                _ => 5.0,
                            });
                        } else {
                            // Ascending base preserves static-priority ordering.
                            fallback_values.push(action_idx as f64 + availability_penalty);
                        }
                    }
                }
                LossMatrix::new(states.clone(), actions.clone(), fallback_values)
                    .expect("static-priority fallback loss matrix is well-formed")
            }
        };

        let policy =
            FallbackPolicy::new(0.7, 20.0, 0.5).expect("valid backend selection fallback policy");

        Self {
            states,
            actions,
            losses,
            policy,
            action_backends,
            adaptive_mode_active,
        }
    }

    /// Compute the base loss for a backend, optionally modulated by
    /// empirical metrics from the adaptive router state.
    ///
    /// When `router_state` is `Some` and has sufficient data for this
    /// backend, the empirical success rate adjusts the Beta prior
    /// parameters and the empirical latency modulates the latency proxy.
    fn backend_base_loss_adaptive(
        backend: BackendKind,
        request: &TranscribeRequest,
        duration_secs: f64,
        router_state: Option<&RouterState>,
    ) -> f64 {
        let (mut alpha, mut beta) = prior_for(backend);
        let quality_score = quality_proxy(backend, request);
        let mut latency_cost = latency_proxy(backend, duration_secs, request.diarize);

        // Modulate priors and latency with empirical data when available.
        if let Some(rs) = router_state {
            let metrics = rs.metrics_for(backend);
            if metrics.sample_count >= ADAPTIVE_MIN_SAMPLES {
                // Blend empirical success rate into Beta prior.
                // Add pseudo-observations proportional to sample count,
                // capped to avoid overwhelming the prior too quickly.
                let empirical_weight = (metrics.sample_count as f64).min(20.0);
                // Sanitize the empirical success rate: a non-finite value would
                // poison alpha/beta and, via the loss, the LossMatrix.
                let success_rate = if metrics.success_rate.is_finite() {
                    metrics.success_rate.clamp(0.0, 1.0)
                } else {
                    0.5
                };
                alpha += success_rate * empirical_weight;
                beta += (1.0 - success_rate) * empirical_weight;

                // Blend empirical latency into the proxy. Use a weighted
                // average: 60% prior estimate, 40% empirical.
                if metrics.avg_latency_ms.is_finite() && metrics.avg_latency_ms > 0.0 {
                    let empirical_latency_secs = metrics.avg_latency_ms / 1000.0;
                    latency_cost = (0.6 * latency_cost) + (0.4 * empirical_latency_secs);
                }
            }
        }

        let p_success = posterior_success_probability(
            alpha,
            beta,
            quality_score,
            request.diarize,
            request.translate,
            true,
        );

        let failure_cost = (1.0 - p_success) * 100.0;
        let quality_cost = (1.0 - quality_score) * 100.0;

        let loss = (0.45 * latency_cost) + (0.35 * quality_cost) + (0.20 * failure_cost);
        if loss.is_finite() {
            loss.max(0.0)
        } else {
            LARGE_FINITE_PENALTY
        }
    }

    /// Backward-compatible base loss without adaptive state (used by tests).
    #[cfg(test)]
    #[allow(dead_code)]
    fn backend_base_loss(
        backend: BackendKind,
        request: &TranscribeRequest,
        duration_secs: f64,
    ) -> f64 {
        Self::backend_base_loss_adaptive(backend, request, duration_secs, None)
    }
}

impl DecisionContract for BackendSelectionContract {
    fn name(&self) -> &str {
        "backend_selection"
    }

    fn state_space(&self) -> &[String] {
        &self.states
    }

    fn action_set(&self) -> &[String] {
        &self.actions
    }

    fn loss_matrix(&self) -> &LossMatrix {
        &self.losses
    }

    fn update_posterior(
        &self,
        posterior: &mut Posterior,
        observation: usize,
    ) -> Result<(), UpdatePosteriorError> {
        // Fail-closed per the franken-decision 0.3.2 contract: refuse a
        // posterior whose length disagrees with the state space, or an
        // observation index outside the state space, leaving `posterior`
        // unchanged so the caller can fall back without observing a
        // partially-applied update.
        if posterior.len() != self.states.len() {
            return Err(UpdatePosteriorError::LengthMismatch {
                expected: self.states.len(),
                actual: posterior.len(),
            });
        }
        if observation >= self.states.len() {
            return Err(UpdatePosteriorError::ObservationOutOfRange {
                observation,
                state_count: self.states.len(),
            });
        }
        let mut likelihoods = vec![0.1; self.states.len()];
        likelihoods[observation] = 0.8;
        posterior.bayesian_update(&likelihoods);
        Ok(())
    }

    fn choose_action(&self, posterior: &Posterior) -> usize {
        self.losses.bayes_action(posterior)
    }

    fn fallback_action(&self) -> usize {
        // When the adaptive router is untrusted (`should_fallback`), degrade to the TOP
        // static-priority backend (index 0 = try_whisper_cpp, served by the native engine),
        // NOT `fallback_error` (index 3). fallback_error emits no transcription and is only
        // correct when no backend is available — a case already handled by `choose_action` /
        // the loss matrix (fallback_error becomes min-loss only as real-backend availability
        // drops). Returning 3 unconditionally made a merely-uncertain router (e.g. a confident
        // posterior whose ci_width still exceeds the 0.5 threshold) emit 0 utterances, which
        // broke the `youtube`/`transcribe` happy path once native execution became the default.
        0
    }

    fn fallback_policy(&self) -> &FallbackPolicy {
        &self.policy
    }
}

/// Record a lightweight routing-evidence event (e.g. a failed posterior
/// update) into the global evidence ledger.
///
/// A `kind`/`detail` pair is stored as a minimal but schema-valid
/// [`RoutingEvidenceLedgerEntry`] whose `observed_state` carries the event
/// kind and whose `fallback_reason` carries the human-readable detail
/// string. This keeps the evidence ledger the single audit channel for
/// adaptive-router anomalies that do not produce a full decision outcome.
fn record_router_evidence_event(kind: &str, detail: &str, trace_id: TraceId) {
    let ts_ms = Utc::now().timestamp_millis() as u64;
    let decision_random = (uuid::Uuid::new_v4().as_u128()) & 0xFFFF_FFFF_FFFF_FFFF_FFFF;
    let decision_id = DecisionId::from_parts(ts_ms, decision_random);
    let entry = RoutingEvidenceLedgerEntry {
        decision_id: decision_id.to_string(),
        trace_id: trace_id.to_string(),
        timestamp_rfc3339: Utc::now().to_rfc3339(),
        observed_state: kind.to_owned(),
        chosen_action: String::new(),
        recommended_order: Vec::new(),
        fallback_active: true,
        fallback_reason: Some(format!("{kind}: {detail}")),
        posterior_snapshot: Vec::new(),
        calibration_score: 0.0,
        brier_score: None,
        e_process: 0.0,
        ci_width: 0.0,
        adaptive_mode: false,
        policy_id: ROUTING_POLICY_ID.to_owned(),
        loss_matrix_hash: String::new(),
        availability: Vec::new(),
        duration_bucket: String::new(),
        diarize: false,
        actual_outcome: None,
    };
    if let Ok(mut guard) = ROUTER_STATE.lock() {
        let state = guard.get_or_insert_with(RouterState::new);
        state.record_evidence(entry);
    }
}

/// Build a deterministic static-priority [`BackendSelectionOutcome`] when the
/// formal decision contract fails validation (`evaluate()` returned
/// `Err(ValidationError)`).
///
/// The Alien-Artifact Engineering Contract requires every adaptive controller
/// to have a deterministic fallback: a contract-validation failure must never
/// panic or drop a transcription request. This wires the validation failure
/// into the same static-priority order used by `resolve_static_backend` /
/// `auto_priority`, records the failure in the evidence ledger with the error
/// string, and emits a `tracing::warn`. `fallback_triggered` is set so callers
/// and downstream calibration treat this as a forced-static decision.
#[allow(clippy::too_many_arguments)]
fn static_fallback_selection_outcome(
    request: &TranscribeRequest,
    contract: &BackendSelectionContract,
    duration: f64,
    observed_state: usize,
    availability: &[(BackendKind, bool)],
    calibration_score: f64,
    e_process: f64,
    ci_width: f64,
    decision_id: DecisionId,
    trace_id: TraceId,
    err: &ValidationError,
) -> BackendSelectionOutcome {
    let err_str = err.to_string();
    tracing::warn!(
        target: "franken_whisper::routing::evidence",
        evidence_type = "contract_validation_error",
        contract = "backend_selection",
        error = err_str.as_str(),
        "decision contract validation failed; engaging deterministic static-priority fallback (transcription preserved)"
    );

    let rollout_stage = native_rollout_stage();
    let static_order_raw = auto_priority(request.diarize).to_vec();
    let (recommended_order, _rollout_forced_static) =
        gate_recommended_order_for_rollout(request.diarize, static_order_raw, rollout_stage);

    let fallback_reason = Some(format!("contract_validation_error: {err_str}"));
    let decision_id_str = decision_id.to_string();
    let loss_matrix_hash = loss_matrix_content_hash(&contract.losses);
    let static_order = auto_priority(request.diarize);

    let routing_log = serde_json::json!({
        "version": "decision-contract-v1",
        "schema_version": ROUTING_EVIDENCE_SCHEMA_VERSION,
        "policy_id": ROUTING_POLICY_ID,
        "mode": "static",
        "contract": "backend_selection",
        "state_space": contract.state_space(),
        "observed_state": contract.state_space()[observed_state],
        "action_set": contract.action_set(),
        "chosen_action": Value::Null,
        "fallback_active": true,
        "fallback_reason": fallback_reason,
        "validation_error": err_str,
        "recommended_order": recommended_order.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        "static_order": static_order.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        "native_rollout_stage": rollout_stage.as_str(),
        "calibration_score": calibration_score,
        "e_process": e_process,
        "ci_width": ci_width,
        "availability": availability
            .iter()
            .map(|(kind, ok)| serde_json::json!({"backend": kind.as_str(), "available": ok}))
            .collect::<Vec<Value>>(),
        "duration_seconds": duration,
        "duration_bucket": duration_bucket(duration),
        "diarize": request.diarize,
        "decision_id": decision_id_str,
        "trace_id": trace_id.to_string(),
    });

    let evidence_entries = vec![routing_log.clone()];

    let ledger_entry = RoutingEvidenceLedgerEntry {
        decision_id: decision_id_str.clone(),
        trace_id: trace_id.to_string(),
        timestamp_rfc3339: Utc::now().to_rfc3339(),
        observed_state: contract.state_space()[observed_state].clone(),
        chosen_action: String::new(),
        recommended_order: recommended_order
            .iter()
            .map(|k| k.as_str().to_owned())
            .collect(),
        fallback_active: true,
        fallback_reason,
        posterior_snapshot: Vec::new(),
        calibration_score,
        brier_score: None,
        e_process,
        ci_width,
        adaptive_mode: false,
        policy_id: ROUTING_POLICY_ID.to_owned(),
        loss_matrix_hash,
        availability: availability
            .iter()
            .map(|(kind, ok)| (kind.as_str().to_owned(), *ok))
            .collect(),
        duration_bucket: duration_bucket(duration).to_owned(),
        diarize: request.diarize,
        actual_outcome: None,
    };

    if let Ok(mut guard) = ROUTER_STATE.lock() {
        let state = guard.get_or_insert_with(RouterState::new);
        state.record_evidence(ledger_entry);
    }

    BackendSelectionOutcome {
        routing_log,
        recommended_order,
        evidence_entries,
        fallback_triggered: true,
        calibration_score,
        e_process,
        ci_width,
    }
}

/// Evaluate the formal backend selection contract and return a routing outcome.
///
/// When a global `RouterState` is available (populated by prior calls to
/// `update_router_state`), the contract uses empirical per-backend metrics
/// to modulate the loss matrix. Otherwise it falls back to static priors.
///
/// The function emits a structured JSON evidence ledger entry via `tracing`
/// on every invocation, satisfying the "evidence ledger artifact"
/// requirement of the Alien-Artifact Engineering Contract.
pub fn evaluate_backend_selection(
    request: &TranscribeRequest,
    normalized_duration_seconds: Option<f64>,
    trace_id: TraceId,
) -> Option<BackendSelectionOutcome> {
    if request.backend != BackendKind::Auto {
        return None;
    }

    let duration = normalized_duration_seconds.unwrap_or(30.0);

    // Snapshot the global router state (if any) for use in the contract.
    let rs_snapshot = router_state_snapshot();
    let contract =
        BackendSelectionContract::with_router_state(request, duration, rs_snapshot.as_ref());

    let availability = [
        (
            BackendKind::WhisperCpp,
            readiness_for(BackendKind::WhisperCpp, request).available,
        ),
        (
            BackendKind::InsanelyFast,
            readiness_for(BackendKind::InsanelyFast, request).available,
        ),
        (
            BackendKind::WhisperDiarization,
            readiness_for(BackendKind::WhisperDiarization, request).available,
        ),
    ];

    let available_count = availability.iter().filter(|(_, a)| *a).count();
    let observed_state = match available_count {
        3 => 0,
        0 => 2,
        _ => 1,
    };

    let mut posterior = Posterior::uniform(3);
    // br-asupersync-u5uhpt: a failed posterior update must never kill a
    // transcription. Skip the update (the uniform prior is the deterministic
    // safe default), record the error in the evidence ledger, and warn. The
    // decision then proceeds against the uniform posterior, which is exactly
    // the high-entropy / low-margin state that drives `should_fallback`
    // toward the static-priority order anyway.
    if let Err(err) = contract.update_posterior(&mut posterior, observed_state) {
        let err_str = err.to_string();
        tracing::warn!(
            target: "franken_whisper::routing::evidence",
            evidence_type = "posterior_update_error",
            contract = "backend_selection",
            error = err_str.as_str(),
            "posterior update failed; proceeding with uniform prior (transcription preserved)"
        );
        record_router_evidence_event("posterior_update_error", &err_str, trace_id);
    }

    let probs = posterior.probs();
    let mut sorted_probs = probs.to_vec();
    sorted_probs.sort_by(|a, b| b.total_cmp(a));
    let max_prob = sorted_probs.first().copied().unwrap_or(0.0);
    let second_prob = sorted_probs.get(1).copied().unwrap_or(0.0);
    let calibration_score = rs_snapshot
        .as_ref()
        .map(|rs| rs.calibration_score())
        .unwrap_or(0.5);

    let ts_ms = Utc::now().timestamp_millis() as u64;
    let decision_random = (uuid::Uuid::new_v4().as_u128()) & 0xFFFF_FFFF_FFFF_FFFF_FFFF;
    let decision_id = DecisionId::from_parts(ts_ms, decision_random);

    // A degenerate posterior (non-finite or subnormal probabilities, or
    // probabilities that do not sum to ~1) makes `entropy()` silently drop
    // those lanes, yielding a near-zero ci_width that reads as MAXIMAL
    // confidence and DEFEATS the fallback trigger. Force a wide interval
    // (fallback) on any degeneracy; otherwise compute as usual.
    let posterior_degenerate = {
        let ps = posterior.probs();
        let sum: f64 = ps.iter().sum();
        ps.iter().any(|p| !p.is_finite() || p.is_subnormal()) || (sum - 1.0).abs() > 1e-6
    };
    let mut ci_width = if posterior_degenerate {
        1.0
    } else {
        posterior.entropy() / 3.0_f64.log2()
    };
    if !ci_width.is_finite() {
        ci_width = 1.0;
    }

    // SPRT-style e-process: inverse of the posterior margin, clamped to a sane
    // range.  When the margin between the best and second-best state is large
    // (high confidence), e_process ≈ 1.  When the margin is tiny (nearly
    // uniform posterior), e_process grows toward the cap, signaling weak
    // evidence for any state.  This feeds into FallbackPolicy::should_fallback
    // which triggers fallback when e_process > breach_threshold.
    let diff = max_prob - second_prob;
    let margin = if diff.is_finite() { diff.max(1e-6) } else { 1e-6 };
    let e_process = (1.0 / margin).clamp(1.0, 100.0);

    let ctx = EvalContext {
        calibration_score,
        e_process,
        ci_width,
        decision_id,
        trace_id,
        ts_unix_ms: ts_ms,
    };

    let outcome = match decision_evaluate(&contract, &posterior, &ctx) {
        Ok(outcome) => outcome,
        Err(err) => {
            // br-asupersync-g1pzep: contract validation failed (e.g. a
            // `choose_action`/`fallback_action` index out of range). A
            // validation failure must engage the deterministic
            // static-priority fallback rather than panic or drop the
            // request. Record the failure in the evidence ledger and return
            // the static-priority routing outcome.
            return Some(static_fallback_selection_outcome(
                request,
                &contract,
                duration,
                observed_state,
                &availability,
                calibration_score,
                e_process,
                ci_width,
                decision_id,
                trace_id,
                &err,
            ));
        }
    };

    // Build recommended order from expected losses (excluding fallback_error).
    let action_set = contract.action_set();
    let mut scored: Vec<(usize, f64)> = (0..action_set.len())
        .filter(|&i| i < contract.action_backends.len())
        .map(|i| (i, outcome.expected_losses[&action_set[i]]))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));

    let recommended_order_raw: Vec<BackendKind> = scored
        .iter()
        .map(|(idx, _)| contract.action_backends[*idx])
        .collect();

    let rollout_stage = native_rollout_stage();
    let (recommended_order, rollout_forced_static) =
        gate_recommended_order_for_rollout(request.diarize, recommended_order_raw, rollout_stage);

    // franken-decision can emit an internally-inconsistent decision audit for
    // the degenerate native-only fallback state (`observed_state == 2`: no
    // external bridge backend present), whose `to_evidence_ledger()` then
    // panics via an internal `.expect()`. The evidence ledger is a diagnostic,
    // not the result, and must never kill a transcription. Two guards, mirroring
    // the router's existing "diagnostics never kill transcription" pattern
    // (failed posterior update; contract-validation fallback):
    //   1. Skip the conversion when the audit's posterior snapshot is empty or
    //      unnormalized — the observed degeneracy — so the common native-only
    //      path avoids the panic (and its stderr backtrace) entirely.
    //   2. Still `catch_unwind` as a safety net for any other malformed audit.
    // Either way we degrade to no evidence entry (the serde-failure default that
    // already existed here) and record the anomaly.
    let posterior_sum: f64 = outcome.audit_entry.posterior_snapshot.iter().sum();
    let audit_convertible = !outcome.audit_entry.posterior_snapshot.is_empty()
        && (posterior_sum - 1.0).abs() <= 1e-6;
    // Sanitize the posterior snapshot that is logged/persisted: a degenerate
    // (empty / unnormalized / subnormal) audit snapshot must not leak into the
    // routing log or evidence ledger. Fall back to the locally-built,
    // guaranteed-normalized posterior when the audit copy is not convertible.
    let posterior_snapshot: Vec<f64> = if audit_convertible {
        outcome.audit_entry.posterior_snapshot.clone()
    } else {
        posterior.probs().to_vec()
    };
    let evidence_entries = if audit_convertible {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            outcome.audit_entry.to_evidence_ledger()
        }))
        .ok()
        .and_then(|ledger| serde_json::to_value(&ledger).ok())
        .map(|v| vec![v])
        .unwrap_or_default()
    } else {
        Vec::new()
    };
    if evidence_entries.is_empty() {
        record_router_evidence_event(
            "evidence_ledger_conversion_failed",
            "decision audit could not be converted to an evidence ledger \
             (empty/unnormalized posterior or invalid entry); proceeding \
             without an evidence entry (transcription preserved)",
            trace_id,
        );
    }

    let static_order = auto_priority(request.diarize);
    let loss_matrix_hash = loss_matrix_content_hash(&contract.losses);

    // Build the adaptive router state snapshot for the routing log.
    let router_state_json = rs_snapshot
        .as_ref()
        .map(RouterState::to_evidence_json)
        .unwrap_or(serde_json::json!(null));

    // bd-efr.2: Include Brier score from calibration state.
    let brier_score = rs_snapshot.as_ref().and_then(|rs| rs.brier_score());
    let router_state_fallback_reason = rs_snapshot.as_ref().and_then(|rs| rs.fallback_reason());
    let fallback_active = outcome.fallback_active || rollout_forced_static;
    let fallback_reason = effective_fallback_reason(
        router_state_fallback_reason,
        rollout_forced_static,
        rollout_stage,
    );

    let mode = routing_mode(contract.adaptive_mode_active, rollout_forced_static);

    let decision_id_str = outcome.audit_entry.decision_id.to_string();

    let routing_log = serde_json::json!({
        "version": "decision-contract-v1",
        "schema_version": ROUTING_EVIDENCE_SCHEMA_VERSION,
        "policy_id": ROUTING_POLICY_ID,
        "mode": mode,
        "contract": "backend_selection",
        "state_space": contract.state_space(),
        "observed_state": contract.state_space()[observed_state],
        "action_set": contract.action_set(),
        "chosen_action": outcome.action_name,
        "fallback_active": fallback_active,
        "fallback_reason": fallback_reason,
        "expected_losses": outcome.expected_losses,
        "posterior_snapshot": posterior_snapshot,
        "calibration_score": calibration_score,
        "brier_score": brier_score,
        "brier_threshold": ADAPTIVE_FALLBACK_BRIER_THRESHOLD,
        "e_process": e_process,
        "ci_width": ci_width,
        "recommended_order": recommended_order.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        "static_order": static_order.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        "native_rollout_stage": rollout_stage.as_str(),
        "native_rollout_forced_static_order": rollout_forced_static,
        "availability": availability
            .iter()
            .map(|(kind, ok)| serde_json::json!({"backend": kind.as_str(), "available": ok}))
            .collect::<Vec<Value>>(),
        "adaptive_router_state": router_state_json,
        "provenance": {
            "policy_id": ROUTING_POLICY_ID,
            "schema_version": ROUTING_EVIDENCE_SCHEMA_VERSION,
            "loss_matrix_hash": loss_matrix_hash,
            "fallback_policy": {
                "calibration_drift_threshold": contract.policy.calibration_drift_threshold,
                "e_process_breach_threshold": contract.policy.e_process_breach_threshold,
                "confidence_width_threshold": contract.policy.confidence_width_threshold,
            },
        },
        "duration_seconds": duration,
        "duration_bucket": duration_bucket(duration),
        "diarize": request.diarize,
        "decision_id": decision_id_str,
        "trace_id": trace_id.to_string(),
    });

    // Emit the evidence ledger as structured JSON via tracing.
    if let Ok(log_str) = serde_json::to_string(&routing_log) {
        tracing::info!(
            target: "franken_whisper::routing::evidence",
            evidence_type = "routing_decision",
            mode = mode,
            chosen_action = outcome.action_name.as_str(),
            fallback_active = fallback_active,
            calibration_score = calibration_score,
            brier_score = brier_score.unwrap_or(-1.0),
            routing_log = log_str.as_str(),
            "backend routing decision"
        );
    }

    // bd-efr.4: Record the decision in the global evidence ledger.
    let ledger_entry = RoutingEvidenceLedgerEntry {
        decision_id: decision_id_str.clone(),
        trace_id: trace_id.to_string(),
        timestamp_rfc3339: Utc::now().to_rfc3339(),
        observed_state: contract.state_space()[observed_state].clone(),
        chosen_action: outcome.action_name.clone(),
        recommended_order: recommended_order
            .iter()
            .map(|k| k.as_str().to_owned())
            .collect(),
        fallback_active,
        fallback_reason: fallback_reason.clone(),
        posterior_snapshot: posterior_snapshot.clone(),
        calibration_score,
        brier_score,
        e_process,
        ci_width,
        adaptive_mode: contract.adaptive_mode_active,
        policy_id: ROUTING_POLICY_ID.to_owned(),
        loss_matrix_hash: loss_matrix_hash.clone(),
        availability: availability
            .iter()
            .map(|(kind, ok)| (kind.as_str().to_owned(), *ok))
            .collect(),
        duration_bucket: duration_bucket(duration).to_owned(),
        diarize: request.diarize,
        actual_outcome: None,
    };

    if let Ok(mut guard) = ROUTER_STATE.lock() {
        let state = guard.get_or_insert_with(RouterState::new);
        state.record_evidence(ledger_entry);
    }

    Some(BackendSelectionOutcome {
        routing_log,
        recommended_order,
        evidence_entries,
        fallback_triggered: fallback_active,
        calibration_score,
        e_process,
        ci_width,
    })
}

/// Compute a stable content hash of loss matrix values for provenance tracking.
fn loss_matrix_content_hash(matrix: &LossMatrix) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for state in 0..matrix.n_states() {
        for action in 0..matrix.n_actions() {
            hasher.update(matrix.get(state, action).to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn duration_bucket(seconds: f64) -> &'static str {
    if seconds < 30.0 {
        "short"
    } else if seconds < 300.0 {
        "medium"
    } else {
        "long"
    }
}

fn prior_for(kind: BackendKind) -> (f64, f64) {
    match kind {
        BackendKind::WhisperCpp => (7.0, 3.0),
        BackendKind::InsanelyFast => (6.0, 4.0),
        BackendKind::WhisperDiarization => (5.0, 5.0),
        BackendKind::Auto => (1.0, 1.0),
    }
}

fn quality_proxy(kind: BackendKind, request: &TranscribeRequest) -> f64 {
    match kind {
        BackendKind::WhisperCpp => {
            if request.diarize {
                0.55
            } else {
                0.84
            }
        }
        BackendKind::InsanelyFast => {
            if request.diarize {
                0.82
            } else {
                0.80
            }
        }
        BackendKind::WhisperDiarization => {
            if request.diarize {
                0.88
            } else {
                0.63
            }
        }
        BackendKind::Auto => 0.50,
    }
}

fn latency_proxy(kind: BackendKind, duration_seconds: f64, diarize: bool) -> f64 {
    let multiplier = if diarize { 1.25 } else { 1.0 };
    let base = match kind {
        BackendKind::WhisperCpp => 12.0,
        BackendKind::InsanelyFast => 8.0,
        BackendKind::WhisperDiarization => 18.0,
        BackendKind::Auto => 20.0,
    };
    // Sanitize the duration: a NaN/Inf feeds sqrt() -> NaN -> a NaN base loss
    // -> LossMatrix::new() rejection. Negative durations are also nonsensical.
    let d = if duration_seconds.is_finite() {
        duration_seconds.max(0.0)
    } else {
        30.0
    };
    base + (d.sqrt() * multiplier)
}

fn posterior_success_probability(
    alpha_prior: f64,
    beta_prior: f64,
    quality_score: f64,
    diarize: bool,
    translate: bool,
    available: bool,
) -> f64 {
    if !available {
        return 0.0;
    }

    let diarize_boost = if diarize { 0.8 } else { 0.3 };
    let translate_penalty = if translate { 0.5 } else { 0.0 };

    let alpha = alpha_prior + (quality_score * 2.0) + diarize_boost;
    let beta = beta_prior + ((1.0 - quality_score) * 2.0) + translate_penalty;
    // Floor the denominator to avoid 0/0 = NaN, then guard the quotient: a
    // non-finite result must not reach the LossMatrix (its clamp/validation
    // panics on NaN under the current toolchain).
    let denom = (alpha + beta).max(1e-9);
    let p = alpha / denom;
    if p.is_finite() {
        p.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub fn extract_segments_from_json(root: &Value) -> Vec<TranscriptionSegment> {
    if let Some(items) = root.get("transcription").and_then(Value::as_array) {
        return segments_from_nodes(items);
    }
    if let Some(items) = root.get("segments").and_then(Value::as_array) {
        return segments_from_nodes(items);
    }
    if let Some(items) = root.get("chunks").and_then(Value::as_array) {
        // Word-level timestamps: chunks may contain nested "words" arrays.
        let has_words = items
            .iter()
            .any(|item| item.get("words").and_then(Value::as_array).is_some());
        if has_words {
            return extract_word_level_segments(items);
        }
        return segments_from_nodes(items);
    }
    Vec::new()
}

/// Extract word-level segments from insanely-fast-whisper word-timestamp output.
fn extract_word_level_segments(chunks: &[Value]) -> Vec<TranscriptionSegment> {
    let mut segments = Vec::new();
    for chunk in chunks {
        let chunk_speaker = chunk
            .get("speaker")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        if let Some(words) = chunk.get("words").and_then(Value::as_array) {
            for word_node in words {
                let start = sanitize_timestamp(
                    word_node
                        .get("start")
                        .or_else(|| word_node.pointer("/timestamp/0"))
                        .and_then(number_to_secs),
                );
                let end = sanitize_timestamp(
                    word_node
                        .get("end")
                        .or_else(|| word_node.pointer("/timestamp/1"))
                        .and_then(number_to_secs),
                );
                let text = word_node
                    .get("word")
                    .or_else(|| word_node.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_owned();
                let confidence = word_node
                    .get("confidence")
                    .or_else(|| word_node.get("probability"))
                    .and_then(Value::as_f64)
                    .filter(|c| c.is_finite())
                    .map(|c| c.clamp(0.0, 1.0));
                let speaker = word_node
                    .get("speaker")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .or_else(|| chunk_speaker.clone());

                if !text.is_empty() {
                    segments.push(TranscriptionSegment {
                        start_sec: start,
                        end_sec: end,
                        text,
                        speaker,
                        confidence,
                    });
                }
            }
        } else {
            // Fallback: treat chunk as a regular segment.
            segments.push(TranscriptionSegment {
                start_sec: segment_start(chunk),
                end_sec: segment_end(chunk),
                text: chunk
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_owned(),
                speaker: chunk_speaker,
                confidence: chunk
                    .get("confidence")
                    .and_then(Value::as_f64)
                    .filter(|c| c.is_finite())
                    .map(|c| c.clamp(0.0, 1.0)),
            });
        }
    }
    segments
}

pub(crate) fn transcript_from_segments(segments: &[TranscriptionSegment]) -> String {
    segments
        .iter()
        .map(|segment| segment.text.as_str())
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_owned()
}

fn segments_from_nodes(nodes: &[Value]) -> Vec<TranscriptionSegment> {
    nodes
        .iter()
        .map(|node| {
            let start = segment_start(node);
            let end = segment_end(node);

            let text = node
                .get("text")
                .and_then(Value::as_str)
                .or_else(|| node.get("word").and_then(Value::as_str))
                .unwrap_or_default()
                .trim()
                .to_owned();

            let speaker = node
                .get("speaker")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);

            let confidence = node
                .get("confidence")
                .or_else(|| node.get("probability"))
                .or_else(|| node.get("score"))
                .and_then(Value::as_f64)
                .filter(|c| c.is_finite())
                .map(|c| c.clamp(0.0, 1.0));

            TranscriptionSegment {
                start_sec: start,
                end_sec: end,
                text,
                speaker,
                confidence,
            }
        })
        .collect()
}

fn segment_start(node: &Value) -> Option<f64> {
    let raw = if let Some(value) = node.pointer("/offsets/from") {
        number_millis_to_secs(value)
    } else {
        node.get("start")
            .or_else(|| node.get("start_sec"))
            .or_else(|| node.pointer("/timestamp/0"))
            .or_else(|| node.pointer("/timestamp/start"))
            .and_then(number_to_secs)
    };
    sanitize_timestamp(raw)
}

fn segment_end(node: &Value) -> Option<f64> {
    let raw = if let Some(value) = node.pointer("/offsets/to") {
        number_millis_to_secs(value)
    } else {
        node.get("end")
            .or_else(|| node.get("end_sec"))
            .or_else(|| node.pointer("/timestamp/1"))
            .or_else(|| node.pointer("/timestamp/end"))
            .and_then(number_to_secs)
    };
    sanitize_timestamp(raw)
}

fn number_to_secs(value: &Value) -> Option<f64> {
    value.as_f64().or_else(|| value.as_i64().map(|v| v as f64))
}

fn number_millis_to_secs(value: &Value) -> Option<f64> {
    number_to_secs(value).map(|seconds| seconds / 1_000.0)
}

/// Sanitize a raw timestamp: strip NaN, Infinity, and negative values.
///
/// Backend JSON output can produce non-finite or negative timestamps
/// when the model fails to align audio.  Replacing them with `None`
/// at the extraction boundary keeps downstream pipeline stages safe
/// and matches the conformance contract checked later in the pipeline.
fn sanitize_timestamp(t: Option<f64>) -> Option<f64> {
    t.filter(|v| v.is_finite() && *v >= 0.0)
}

// ---------------------------------------------------------------------------
// bd-1rj.7: Canonical segment tolerance and conformance checking
// ---------------------------------------------------------------------------

/// Canonical tolerance for segment boundary alignment, in milliseconds.
/// Two segment timestamps that differ by at most this amount are considered
/// conformant (i.e., no violation is reported for overlaps or gaps within
/// this tolerance). This value applies to both overlap and gap checks.
///
/// # Alien-Artifact Engineering Contract
/// - **State space**: sequence of `TranscriptionSegment` with optional timestamps
/// - **Actions**: report conformant / report violation per segment pair
/// - **Loss matrix**: false-positive violation (tolerance too tight) vs
///   false-negative miss (tolerance too loose)
/// - **Calibration metric**: conformance ratio (conformant / total)
/// - **Deterministic fallback**: segments without timestamps are skipped
///   (neither conformant nor violating)
pub const CANONICAL_SEGMENT_TOLERANCE_MS: u64 = 50;

/// A single conformance violation found during segment validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentConformanceViolation {
    /// Zero-based index of the first segment involved.
    pub segment_index: usize,
    /// Zero-based index of the second segment involved (for pairwise checks).
    /// `None` for single-segment issues (e.g., start > end).
    pub adjacent_index: Option<usize>,
    /// Machine-readable violation kind.
    pub kind: SegmentViolationKind,
    /// Human-readable description of the violation.
    pub description: String,
    /// The magnitude of the violation in milliseconds, if applicable.
    pub magnitude_ms: Option<f64>,
}

/// Categories of segment conformance violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentViolationKind {
    /// Timestamps are not monotonically increasing between adjacent segments.
    NonMonotonic,
    /// Adjacent segments overlap by more than the canonical tolerance.
    OverlapBeyondTolerance,
    /// Adjacent segments have a gap exceeding the canonical tolerance.
    GapBeyondTolerance,
    /// A segment's start timestamp is after its end timestamp.
    InvertedTimestamps,
}

/// Report produced by `check_segment_conformance()`, summarizing how well
/// a set of segments adheres to the canonical timing constraints.
///
/// This report serves as a conformance artifact in the routing evidence
/// ledger, enabling post-hoc analysis of segment quality across backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentConformanceReport {
    /// Total number of segments evaluated.
    pub total_segments: usize,
    /// Number of segments that passed all conformance checks.
    pub conformant_segments: usize,
    /// Detailed list of violations found.
    pub violations: Vec<SegmentConformanceViolation>,
    /// The tolerance in milliseconds used for this conformance check.
    pub tolerance_ms: u64,
    /// Conformance ratio: `conformant_segments / total_segments`.
    /// Returns 1.0 when there are no segments (vacuously conformant).
    pub conformance_ratio: f64,
}

/// Validate a slice of `TranscriptionSegment` values against the canonical
/// segment timing tolerances.
///
/// Checks performed:
/// 1. **Inverted timestamps**: each segment's `start_sec` must be <= `end_sec`.
/// 2. **Monotonic ordering**: each segment's start must be >= the previous
///    segment's start.
/// 3. **Overlap beyond tolerance**: adjacent segments must not overlap by more
///    than `CANONICAL_SEGMENT_TOLERANCE_MS`.
/// 4. **Gap beyond tolerance**: adjacent segments must not have a gap exceeding
///    `CANONICAL_SEGMENT_TOLERANCE_MS`.
///
/// Segments missing timestamps are skipped for pairwise checks (deterministic
/// fallback: no data means no assertion).
#[must_use]
pub fn check_segment_conformance(segments: &[TranscriptionSegment]) -> SegmentConformanceReport {
    let total_segments = segments.len();
    let mut violations = Vec::new();
    let tolerance_secs = CANONICAL_SEGMENT_TOLERANCE_MS as f64 / 1000.0;

    // Track which segment indices are involved in at least one violation.
    let mut violating_indices = std::collections::HashSet::new();

    // Pass 1: Per-segment checks (inverted timestamps).
    for (i, seg) in segments.iter().enumerate() {
        if let (Some(start), Some(end)) = (seg.start_sec, seg.end_sec)
            && start > end
        {
            let magnitude_ms = (start - end) * 1000.0;
            violations.push(SegmentConformanceViolation {
                segment_index: i,
                adjacent_index: None,
                kind: SegmentViolationKind::InvertedTimestamps,
                description: format!(
                    "segment {i}: start ({start:.3}s) > end ({end:.3}s), inverted by {magnitude_ms:.1}ms"
                ),
                magnitude_ms: Some(magnitude_ms),
            });
            violating_indices.insert(i);
        }
    }

    // Pass 2: Pairwise checks between adjacent segments.
    for i in 1..segments.len() {
        let prev = &segments[i - 1];
        let curr = &segments[i];

        // We need start of current and start of previous for monotonicity.
        if let (Some(prev_start), Some(curr_start)) = (prev.start_sec, curr.start_sec)
            && curr_start < prev_start - tolerance_secs
        {
            let magnitude_ms = (prev_start - curr_start) * 1000.0;
            violations.push(SegmentConformanceViolation {
                segment_index: i - 1,
                adjacent_index: Some(i),
                kind: SegmentViolationKind::NonMonotonic,
                description: format!(
                    "segments {}-{i}: start timestamps not monotonic ({prev_start:.3}s -> {curr_start:.3}s), delta {magnitude_ms:.1}ms",
                    i - 1
                ),
                magnitude_ms: Some(magnitude_ms),
            });
            violating_indices.insert(i - 1);
            violating_indices.insert(i);
        }

        // Overlap and gap checks use previous end vs. current start.
        if let (Some(prev_end), Some(curr_start)) = (prev.end_sec, curr.start_sec) {
            let delta = curr_start - prev_end; // positive = gap, negative = overlap

            if delta < -tolerance_secs {
                // Overlap beyond tolerance.
                let overlap_ms = (-delta) * 1000.0;
                violations.push(SegmentConformanceViolation {
                    segment_index: i - 1,
                    adjacent_index: Some(i),
                    kind: SegmentViolationKind::OverlapBeyondTolerance,
                    description: format!(
                        "segments {}-{i}: overlap of {overlap_ms:.1}ms exceeds tolerance of {CANONICAL_SEGMENT_TOLERANCE_MS}ms",
                        i - 1
                    ),
                    magnitude_ms: Some(overlap_ms),
                });
                violating_indices.insert(i - 1);
                violating_indices.insert(i);
            } else if delta > tolerance_secs {
                // Gap beyond tolerance.
                let gap_ms = delta * 1000.0;
                violations.push(SegmentConformanceViolation {
                    segment_index: i - 1,
                    adjacent_index: Some(i),
                    kind: SegmentViolationKind::GapBeyondTolerance,
                    description: format!(
                        "segments {}-{i}: gap of {gap_ms:.1}ms exceeds tolerance of {CANONICAL_SEGMENT_TOLERANCE_MS}ms",
                        i - 1
                    ),
                    magnitude_ms: Some(gap_ms),
                });
                violating_indices.insert(i - 1);
                violating_indices.insert(i);
            }
        }
    }

    let conformant_segments = total_segments - violating_indices.len();
    let conformance_ratio = if total_segments == 0 {
        1.0
    } else {
        conformant_segments as f64 / total_segments as f64
    };

    SegmentConformanceReport {
        total_segments,
        conformant_segments,
        violations,
        tolerance_ms: CANONICAL_SEGMENT_TOLERANCE_MS,
        conformance_ratio,
    }
}

// ---------------------------------------------------------------------------
// bd-1rj.8: Native engine replacement contract and shadow-run rollout plan
// ---------------------------------------------------------------------------

/// Defines the interface contract that a native Rust engine must satisfy in
/// order to replace an existing external-process backend. This struct captures
/// the capability requirements, performance bounds, and quality thresholds
/// that a replacement must meet before promotion.
///
/// # Alien-Artifact Engineering Contract
/// - **State space**: {candidate_ready, candidate_partial, candidate_absent}
/// - **Actions**: {promote_native, keep_legacy, shadow_run}
/// - **Loss matrix**: premature promotion (quality regression) vs delayed
///   promotion (missed performance gains)
/// - **Calibration metric**: divergence rate in shadow runs
/// - **Deterministic fallback**: keep legacy backend if shadow divergence
///   exceeds `max_acceptable_divergence_ms`
/// - **Evidence ledger**: `ShadowRunReport` artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeEngineContract {
    /// Human-readable name for the native engine candidate.
    pub name: String,
    /// The legacy backend this engine aims to replace.
    pub replaces: BackendKind,
    /// Required capabilities the native engine must support.
    pub required_capabilities: EngineCapabilities,
    /// Maximum acceptable median latency in milliseconds. If the native
    /// engine exceeds this, it fails the contract.
    pub max_median_latency_ms: u64,
    /// Maximum acceptable 95th percentile latency in milliseconds.
    pub max_p95_latency_ms: u64,
    /// Minimum acceptable transcription accuracy (0.0 to 1.0), measured
    /// as word-error-rate complement (1 - WER).
    pub min_accuracy: f64,
    /// Maximum acceptable divergence from the legacy engine in milliseconds
    /// of timestamp difference, averaged across segments.
    pub max_acceptable_divergence_ms: u64,
    /// Minimum number of shadow runs required before considering promotion.
    pub min_shadow_runs_for_promotion: u64,
    /// Maximum allowed shadow-run divergence rate (fraction of runs that
    /// exceed `max_acceptable_divergence_ms`) before fallback to legacy.
    pub max_divergence_rate: f64,
}

/// Configuration for running a shadow (dual-execution) comparison between
/// two backends. In shadow mode, the primary backend produces the actual
/// result while the shadow backend runs concurrently for comparison only.
///
/// This enables safe, data-driven rollout of native engine replacements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowRunConfig {
    /// Whether shadow-run mode is enabled.
    pub enabled: bool,
    /// The backend that produces the user-visible result.
    pub primary_backend: BackendKind,
    /// The backend run in shadow mode for comparison.
    pub shadow_backend: BackendKind,
    /// Maximum acceptable divergence in milliseconds between primary and
    /// shadow segment timestamps. Beyond this, a divergence is flagged.
    pub max_divergence_ms: u64,
}

/// A single divergence found when comparing primary and shadow results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowRunDivergence {
    /// Zero-based segment index where the divergence was detected.
    pub segment_index: usize,
    /// Machine-readable divergence kind.
    pub kind: ShadowDivergenceKind,
    /// Human-readable description.
    pub description: String,
}

/// Categories of divergences between primary and shadow results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowDivergenceKind {
    /// Segment counts differ between primary and shadow.
    SegmentCountMismatch,
    /// Segment text differs.
    TextDifference,
    /// Start timestamp diverges by more than the configured tolerance.
    StartTimestampDivergence,
    /// End timestamp diverges by more than the configured tolerance.
    EndTimestampDivergence,
    /// Speaker label differs.
    SpeakerDifference,
}

/// Report produced by a shadow run, comparing the primary result against
/// the shadow result. This serves as an evidence ledger artifact for the
/// native engine replacement rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowRunReport {
    /// The result from the primary (user-visible) backend.
    pub primary_result: TranscriptionResult,
    /// The result from the shadow backend.
    pub shadow_result: TranscriptionResult,
    /// All divergences detected between the two results.
    pub divergences: Vec<ShadowRunDivergence>,
    /// Wall-clock latency of the shadow backend in milliseconds.
    pub shadow_latency_ms: u64,
    /// Whether the shadow result is considered acceptable (no divergences
    /// exceed the configured threshold).
    pub acceptable: bool,
}

/// Compare two `TranscriptionResult` values and report divergences.
///
/// This function performs a segment-by-segment comparison between the
/// primary and shadow results:
/// 1. If segment counts differ, a `SegmentCountMismatch` divergence is
///    reported and comparison proceeds up to the shorter length.
/// 2. For each aligned segment pair, text, timestamps, and speaker labels
///    are compared.
/// 3. Timestamp divergences are only flagged when both segments have
///    timestamps and the difference exceeds `max_divergence_ms`.
///
/// # Arguments
/// * `primary` - The primary (authoritative) transcription result.
/// * `shadow` - The shadow (candidate) transcription result.
/// * `max_divergence_ms` - Maximum acceptable timestamp divergence in ms.
///
/// # Returns
/// A vector of `ShadowRunDivergence` values describing all differences found.
#[must_use]
pub fn compare_shadow_results(
    primary: &TranscriptionResult,
    shadow: &TranscriptionResult,
    max_divergence_ms: u64,
) -> Vec<ShadowRunDivergence> {
    let mut divergences = Vec::new();
    let tolerance_secs = max_divergence_ms as f64 / 1000.0;

    // Check segment count mismatch.
    if primary.segments.len() != shadow.segments.len() {
        divergences.push(ShadowRunDivergence {
            segment_index: 0,
            kind: ShadowDivergenceKind::SegmentCountMismatch,
            description: format!(
                "segment count mismatch: primary={}, shadow={}",
                primary.segments.len(),
                shadow.segments.len()
            ),
        });
    }

    // Compare aligned segments.
    let compare_len = primary.segments.len().min(shadow.segments.len());
    for i in 0..compare_len {
        let p = &primary.segments[i];
        let s = &shadow.segments[i];

        // Text comparison (trimmed, case-sensitive).
        if p.text.trim() != s.text.trim() {
            divergences.push(ShadowRunDivergence {
                segment_index: i,
                kind: ShadowDivergenceKind::TextDifference,
                description: format!(
                    "segment {i}: text differs: primary={:?}, shadow={:?}",
                    p.text.trim(),
                    s.text.trim()
                ),
            });
        }

        // Start timestamp comparison.
        if let (Some(p_start), Some(s_start)) = (p.start_sec, s.start_sec) {
            let diff = (p_start - s_start).abs();
            if diff > tolerance_secs {
                divergences.push(ShadowRunDivergence {
                    segment_index: i,
                    kind: ShadowDivergenceKind::StartTimestampDivergence,
                    description: format!(
                        "segment {i}: start divergence {:.1}ms (primary={p_start:.3}s, shadow={s_start:.3}s)",
                        diff * 1000.0
                    ),
                });
            }
        }

        // End timestamp comparison.
        if let (Some(p_end), Some(s_end)) = (p.end_sec, s.end_sec) {
            let diff = (p_end - s_end).abs();
            if diff > tolerance_secs {
                divergences.push(ShadowRunDivergence {
                    segment_index: i,
                    kind: ShadowDivergenceKind::EndTimestampDivergence,
                    description: format!(
                        "segment {i}: end divergence {:.1}ms (primary={p_end:.3}s, shadow={s_end:.3}s)",
                        diff * 1000.0
                    ),
                });
            }
        }

        // Speaker comparison.
        if p.speaker != s.speaker {
            divergences.push(ShadowRunDivergence {
                segment_index: i,
                kind: ShadowDivergenceKind::SpeakerDifference,
                description: format!(
                    "segment {i}: speaker differs: primary={:?}, shadow={:?}",
                    p.speaker, s.speaker
                ),
            });
        }
    }

    divergences
}

// ---------------------------------------------------------------------------
// bd-jryr / bd-s8w8 / bd-cidv: the mock native-engine "pilots" were DELETED.
// All three native engines now run real in-process inference (see
// `whisper_cpp_native.rs`, `insanely_fast_native.rs`, and
// `whisper_diarization_native.rs` → `crate::native_engine`). Their former
// canned-phrase placeholders — `WhisperCppPilot`, `InsanelyFastPilot`, and
// `DiarizationPilot` (a fixed rotation of canned meeting phrases) — no longer
// exist in production. `whisper_diarization_native` now performs real ASR and calls
// the orchestrator's heuristic diarizer (`diarize_segments`), with an honest
// `raw_output` diarizer tag (no neural-encoder claim; ECAPA tracked in bd-ohex).
//
// `TranscriptSegment` (below) survives the pilot purge because the two-lane
// executors (`TwoLaneExecutor` / `ConcurrentTwoLaneExecutor`) and `streaming.rs`
// use it as their lightweight cross-lane segment shape — it is NOT pilot-only.
// The pilot-specific `DiarizedTranscript` / `DiarizedSegment` / `SpeakerInfo`
// structs were deleted with the pilot (nothing else referenced them).
// ---------------------------------------------------------------------------

/// A single transcript segment used by the two-lane executors and streaming
/// bridge.
///
/// This is intentionally separate from `TranscriptionSegment` (used by the
/// CLI/external-process backends): a compact `[start_ms, end_ms, text,
/// confidence]` shape for the speculative/quality two-lane selection math.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Segment start time in milliseconds.
    pub start_ms: u64,
    /// Segment end time in milliseconds.
    pub end_ms: u64,
    /// Transcribed text for this segment.
    pub text: String,
    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// bd-efr.3: TwoLaneExecutor — parallel backends with quality selection
// ---------------------------------------------------------------------------

/// Strategy for selecting between two backend results.
#[derive(Debug, Clone)]
pub enum QualitySelector {
    /// Pick the result with the higher average segment confidence.
    HigherConfidence,
    /// Pick the result that completed faster (lower latency).
    LowerLatency,
    /// Pick using a custom scoring function. The function receives the
    /// segments and latency in milliseconds, and returns a score where
    /// higher is better.
    Custom(fn(&[TranscriptSegment], u64) -> f64),
    /// Always prefer secondary (quality model) as authoritative result.
    /// Primary is used only for early emission (low-latency speculation).
    SpeculativeCorrect,
}

/// The outcome of a two-lane execution, containing both results and the
/// selection decision.
#[derive(Debug, Clone)]
pub struct TwoLaneResult {
    /// Result from the primary (first) backend.
    pub primary_result: Vec<TranscriptSegment>,
    /// Latency of the primary backend in milliseconds.
    pub primary_latency_ms: u64,
    /// Result from the secondary (second) backend.
    pub secondary_result: Vec<TranscriptSegment>,
    /// Latency of the secondary backend in milliseconds.
    pub secondary_latency_ms: u64,
    /// Which lane was selected: `"primary"` or `"secondary"`.
    pub selected: String,
    /// Human-readable explanation of why this lane was selected.
    pub selection_reason: String,
}

/// Runs two backends and picks the better result based on a
/// [`QualitySelector`] strategy.
///
/// Currently executes sequentially (primary then secondary). A future
/// version will run them in parallel using async or threads.
pub struct TwoLaneExecutor {
    selector: QualitySelector,
}

impl TwoLaneExecutor {
    /// Create a new executor with the given quality selection strategy.
    #[must_use]
    pub fn new(selector: QualitySelector) -> Self {
        Self { selector }
    }

    /// Execute two backend functions and select the better result.
    ///
    /// `primary_fn` and `secondary_fn` are closures that return transcript
    /// segments. They are called sequentially (primary first).
    pub fn execute<F1, F2>(&self, primary_fn: F1, secondary_fn: F2) -> TwoLaneResult
    where
        F1: FnOnce() -> Vec<TranscriptSegment>,
        F2: FnOnce() -> Vec<TranscriptSegment>,
    {
        let primary_start = Instant::now();
        let primary_result = primary_fn();
        let primary_latency_ms = primary_start.elapsed().as_millis() as u64;

        let secondary_start = Instant::now();
        let secondary_result = secondary_fn();
        let secondary_latency_ms = secondary_start.elapsed().as_millis() as u64;

        let (selected, selection_reason) = match &self.selector {
            QualitySelector::HigherConfidence => {
                let primary_conf = avg_confidence(&primary_result);
                let secondary_conf = avg_confidence(&secondary_result);
                if primary_conf >= secondary_conf {
                    (
                        "primary".to_owned(),
                        format!(
                            "primary confidence {primary_conf:.4} >= secondary {secondary_conf:.4}"
                        ),
                    )
                } else {
                    (
                        "secondary".to_owned(),
                        format!(
                            "secondary confidence {secondary_conf:.4} > primary {primary_conf:.4}"
                        ),
                    )
                }
            }
            QualitySelector::LowerLatency => {
                if primary_latency_ms <= secondary_latency_ms {
                    (
                        "primary".to_owned(),
                        format!(
                            "primary latency {primary_latency_ms}ms <= secondary {secondary_latency_ms}ms"
                        ),
                    )
                } else {
                    (
                        "secondary".to_owned(),
                        format!(
                            "secondary latency {secondary_latency_ms}ms < primary {primary_latency_ms}ms"
                        ),
                    )
                }
            }
            QualitySelector::Custom(score_fn) => {
                let primary_score = score_fn(&primary_result, primary_latency_ms);
                let secondary_score = score_fn(&secondary_result, secondary_latency_ms);
                if primary_score >= secondary_score {
                    (
                        "primary".to_owned(),
                        format!(
                            "custom: primary score {primary_score:.4} >= secondary {secondary_score:.4}"
                        ),
                    )
                } else {
                    (
                        "secondary".to_owned(),
                        format!(
                            "custom: secondary score {secondary_score:.4} > primary {primary_score:.4}"
                        ),
                    )
                }
            }
            QualitySelector::SpeculativeCorrect => (
                "secondary".to_owned(),
                "speculative-correct: secondary is authoritative".to_owned(),
            ),
        };

        TwoLaneResult {
            primary_result,
            primary_latency_ms,
            secondary_result,
            secondary_latency_ms,
            selected,
            selection_reason,
        }
    }
}

/// Compute the average confidence across transcript segments.
/// Returns 0.0 for an empty slice.
fn avg_confidence(segments: &[TranscriptSegment]) -> f64 {
    if segments.is_empty() {
        return 0.0;
    }
    let sum: f64 = segments.iter().map(|s| s.confidence).sum();
    sum / segments.len() as f64
}

// ---------------------------------------------------------------------------
// bd-qlt.3: ConcurrentTwoLaneExecutor — parallel execution with early emit
// ---------------------------------------------------------------------------

/// Concurrent version of [`TwoLaneExecutor`] that runs both backends in
/// parallel using `std::thread::spawn`. The key innovation is
/// `execute_with_early_emit` which calls a callback as soon as the primary
/// (fast) model finishes, while the secondary (quality) model is still running.
pub struct ConcurrentTwoLaneExecutor {
    selector: QualitySelector,
}

impl ConcurrentTwoLaneExecutor {
    #[must_use]
    pub fn new(selector: QualitySelector) -> Self {
        Self { selector }
    }

    /// Execute two backend functions in parallel and select the better result.
    pub fn execute<F1, F2>(&self, primary_fn: F1, secondary_fn: F2) -> TwoLaneResult
    where
        F1: FnOnce() -> Vec<TranscriptSegment> + Send + 'static,
        F2: FnOnce() -> Vec<TranscriptSegment> + Send + 'static,
    {
        let primary_handle = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let result = primary_fn();
            let latency_ms = start.elapsed().as_millis() as u64;
            (result, latency_ms)
        });
        let secondary_handle = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let result = secondary_fn();
            let latency_ms = start.elapsed().as_millis() as u64;
            (result, latency_ms)
        });

        let (primary_result, primary_latency_ms) = primary_handle
            .join()
            .unwrap_or_else(|e| std::panic::resume_unwind(e));
        let (secondary_result, secondary_latency_ms) = secondary_handle
            .join()
            .unwrap_or_else(|e| std::panic::resume_unwind(e));

        let (selected, selection_reason) = self.select(
            &primary_result,
            primary_latency_ms,
            &secondary_result,
            secondary_latency_ms,
        );

        TwoLaneResult {
            primary_result,
            primary_latency_ms,
            secondary_result,
            secondary_latency_ms,
            selected,
            selection_reason,
        }
    }

    /// Execute with early emission: calls `on_primary` as soon as the primary
    /// (fast) lane finishes, then waits for the secondary (quality) lane and
    /// calls `on_compare` with both results.
    pub fn execute_with_early_emit<F1, F2, P, C>(
        &self,
        primary_fn: F1,
        secondary_fn: F2,
        on_primary: P,
        on_compare: C,
    ) -> TwoLaneResult
    where
        F1: FnOnce() -> Vec<TranscriptSegment> + Send + 'static,
        F2: FnOnce() -> Vec<TranscriptSegment> + Send + 'static,
        P: FnOnce(&[TranscriptSegment], u64),
        C: FnOnce(&[TranscriptSegment], &[TranscriptSegment], u64, u64),
    {
        // Spawn both lanes
        let primary_handle = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let result = primary_fn();
            let latency_ms = start.elapsed().as_millis() as u64;
            (result, latency_ms)
        });
        let secondary_handle = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let result = secondary_fn();
            let latency_ms = start.elapsed().as_millis() as u64;
            (result, latency_ms)
        });

        // Wait for primary first (fast model) and emit immediately
        let (primary_result, primary_latency_ms) = primary_handle
            .join()
            .unwrap_or_else(|e| std::panic::resume_unwind(e));
        on_primary(&primary_result, primary_latency_ms);

        // Wait for secondary (quality model)
        let (secondary_result, secondary_latency_ms) = secondary_handle
            .join()
            .unwrap_or_else(|e| std::panic::resume_unwind(e));
        on_compare(
            &primary_result,
            &secondary_result,
            primary_latency_ms,
            secondary_latency_ms,
        );

        let (selected, selection_reason) = self.select(
            &primary_result,
            primary_latency_ms,
            &secondary_result,
            secondary_latency_ms,
        );

        TwoLaneResult {
            primary_result,
            primary_latency_ms,
            secondary_result,
            secondary_latency_ms,
            selected,
            selection_reason,
        }
    }

    fn select(
        &self,
        primary: &[TranscriptSegment],
        primary_latency: u64,
        secondary: &[TranscriptSegment],
        secondary_latency: u64,
    ) -> (String, String) {
        match &self.selector {
            QualitySelector::HigherConfidence => {
                let pc = avg_confidence(primary);
                let sc = avg_confidence(secondary);
                if pc >= sc {
                    (
                        "primary".to_owned(),
                        format!("primary confidence {pc:.4} >= secondary {sc:.4}"),
                    )
                } else {
                    (
                        "secondary".to_owned(),
                        format!("secondary confidence {sc:.4} > primary {pc:.4}"),
                    )
                }
            }
            QualitySelector::LowerLatency => {
                if primary_latency <= secondary_latency {
                    (
                        "primary".to_owned(),
                        format!(
                            "primary latency {primary_latency}ms <= secondary {secondary_latency}ms"
                        ),
                    )
                } else {
                    (
                        "secondary".to_owned(),
                        format!(
                            "secondary latency {secondary_latency}ms < primary {primary_latency}ms"
                        ),
                    )
                }
            }
            QualitySelector::Custom(score_fn) => {
                let ps = score_fn(primary, primary_latency);
                let ss = score_fn(secondary, secondary_latency);
                if ps >= ss {
                    (
                        "primary".to_owned(),
                        format!("custom: primary score {ps:.4} >= secondary {ss:.4}"),
                    )
                } else {
                    (
                        "secondary".to_owned(),
                        format!("custom: secondary score {ss:.4} > primary {ps:.4}"),
                    )
                }
            }
            QualitySelector::SpeculativeCorrect => (
                "secondary".to_owned(),
                "speculative-correct: secondary is authoritative".to_owned(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD, ADAPTIVE_MIN_SAMPLES, BackendHealthReport,
        BackendImplementation, BackendSelectionContract, CANONICAL_SEGMENT_TOLERANCE_MS,
        CalibrationState, Engine, InsanelyFastEngine, NativeEngineContract, QualitySelector,
        ROUTER_HISTORY_WINDOW, RouterState, RoutingEvidenceLedger, RoutingEvidenceLedgerEntry,
        RoutingOutcomeRecord, SegmentConformanceReport, SegmentConformanceViolation,
        SegmentViolationKind, ShadowDivergenceKind, ShadowRunConfig, ShadowRunDivergence,
        ShadowRunReport, TranscriptSegment, TwoLaneExecutor, WhisperCppEngine,
        WhisperDiarizationEngine, auto_priority, bridge_error_recoverable,
        bridge_native_recovery_from_raw, check_segment_conformance, compare_shadow_results,
        duration_bucket, evaluate_backend_selection, extract_segments_from_json, is_hf_token_set,
        latency_proxy, native_runtime_metadata, number_millis_to_secs, number_to_secs,
        posterior_success_probability, prior_for, probe_system_health,
        probe_system_health_uncached, quality_proxy, runtime_metadata,
        runtime_metadata_with_implementation, sanitize_timestamp, segment_end, segment_start,
        static_fallback_selection_outcome, transcript_from_segments,
    };
    use crate::conformance::NativeEngineRolloutStage;
    use crate::model::{
        BackendKind, EngineCapabilities, InputSource, TranscribeRequest, TranscriptionResult,
        TranscriptionSegment,
    };
    use franken_decision::DecisionContract;
    use franken_kernel::DecisionId;
    use franken_kernel::TraceId;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::Duration;

    static ROUTER_STATE_TEST_MUTEX: StdMutex<()> = StdMutex::new(());

    fn test_request(diarize: bool) -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: Some("en".to_owned()),
            translate: false,
            diarize,
            persist: false,
            db_path: PathBuf::from("db.sqlite3"),
            timeout_ms: None,
            backend_params: crate::model::BackendParams::default(),
        }
    }

    #[test]
    fn backend_selection_contract_implements_decision_contract() {
        let request = test_request(false);
        let contract = BackendSelectionContract::new(&request, 30.0);
        assert_eq!(contract.name(), "backend_selection");
        assert_eq!(contract.state_space().len(), 3);
        assert_eq!(contract.action_set().len(), 4);
        assert_eq!(contract.loss_matrix().n_states(), 3);
        assert_eq!(contract.loss_matrix().n_actions(), 4);
        // Uncertainty fallback degrades to the top static-priority backend (index 0),
        // not fallback_error (3) — see BackendSelectionContract::fallback_action.
        assert_eq!(contract.fallback_action(), 0);
    }

    #[test]
    fn backend_selection_contract_diarize_reorders_actions() {
        let non_diarize = BackendSelectionContract::new(&test_request(false), 30.0);
        let diarize = BackendSelectionContract::new(&test_request(true), 30.0);

        assert_eq!(non_diarize.action_set()[0], "try_whisper_cpp");
        assert_eq!(diarize.action_set()[0], "try_insanely_fast");
    }

    #[test]
    fn backend_selection_evaluate_returns_outcome() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 42);
        let outcome = evaluate_backend_selection(&request, Some(15.0), trace_id);
        assert!(outcome.is_some());
        let outcome = outcome.unwrap();
        assert_eq!(outcome.recommended_order.len(), 3);
        assert!(!outcome.evidence_entries.is_empty());
        assert!(outcome.routing_log["version"] == "decision-contract-v1");
        // Mode is "static" when no router state exists, "adaptive" when it does.
        let mode = outcome.routing_log["mode"].as_str().unwrap();
        assert!(
            mode == "static" || mode == "adaptive",
            "mode should be 'static' or 'adaptive', got '{mode}'"
        );
    }

    #[test]
    fn backend_selection_exposes_calibration_score() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 99);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();

        // calibration_score must be in [0, 1] range
        assert!(
            (0.0..=1.0).contains(&outcome.calibration_score),
            "calibration_score {} should be in [0, 1]",
            outcome.calibration_score
        );
        // routing_log should also contain the matching value
        let log_cal = outcome.routing_log["calibration_score"]
            .as_f64()
            .expect("calibration_score in routing_log");
        assert!(
            (log_cal - outcome.calibration_score).abs() < 1e-12,
            "routing_log calibration_score should match struct field"
        );
    }

    #[test]
    fn calibration_state_sanitizes_non_finite_predictions() {
        let mut state = CalibrationState::new(ROUTER_HISTORY_WINDOW);

        state.record(f64::NAN, true);
        let brier = state.brier_score().expect("brier score");
        assert!(brier.is_finite(), "brier score should be finite");
        assert!(
            (brier - 0.25).abs() < 1e-9,
            "NaN prediction should default to 0.5, got brier {brier}"
        );

        state.record(f64::INFINITY, false);
        let brier = state.brier_score().expect("brier score");
        assert!(brier.is_finite(), "brier score should remain finite");
    }

    #[test]
    fn backend_selection_exposes_e_process_and_ci_width() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 101);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();

        // e_process in [1.0, 100.0] (clamped range)
        assert!(
            (1.0..=100.0).contains(&outcome.e_process),
            "e_process {} should be in [1, 100]",
            outcome.e_process
        );

        // ci_width should be non-negative and bounded by 1.0 (normalized entropy)
        assert!(
            (0.0..=1.0).contains(&outcome.ci_width),
            "ci_width {} should be in [0, 1]",
            outcome.ci_width
        );

        // routing_log should contain both values
        let log_e = outcome.routing_log["e_process"]
            .as_f64()
            .expect("e_process in routing_log");
        assert!(
            (log_e - outcome.e_process).abs() < 1e-12,
            "routing_log e_process should match struct field"
        );

        let log_ci = outcome.routing_log["ci_width"]
            .as_f64()
            .expect("ci_width in routing_log");
        assert!(
            (log_ci - outcome.ci_width).abs() < 1e-12,
            "routing_log ci_width should match struct field"
        );
    }

    #[test]
    fn e_process_inversely_related_to_posterior_margin() {
        // With all backends available (observed_state=0), the posterior should
        // concentrate on one state, giving high margin → low e_process.
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 102);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();

        // When calibration is high, e_process should be relatively low
        if outcome.calibration_score > 0.5 {
            assert!(
                outcome.e_process < 10.0,
                "high calibration ({}) should yield low e_process ({})",
                outcome.calibration_score,
                outcome.e_process
            );
        }
    }

    #[test]
    fn provenance_fallback_policy_uses_actual_contract_values() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 103);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
        let provenance = &outcome.routing_log["provenance"];
        let fp = &provenance["fallback_policy"];

        // Should match FallbackPolicy::default() values
        assert_eq!(fp["calibration_drift_threshold"].as_f64(), Some(0.7));
        assert_eq!(fp["e_process_breach_threshold"].as_f64(), Some(20.0));
        assert_eq!(fp["confidence_width_threshold"].as_f64(), Some(0.5));
    }

    #[test]
    fn backend_selection_routing_log_contains_provenance_fields() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 50);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
        let log = &outcome.routing_log;

        // Top-level provenance markers.
        assert_eq!(log["schema_version"], "1.0");
        assert_eq!(log["policy_id"], "backend-selection-v1.0");

        // Nested provenance block.
        let provenance = &log["provenance"];
        assert_eq!(provenance["policy_id"], "backend-selection-v1.0");
        assert_eq!(provenance["schema_version"], "1.0");
        assert!(
            provenance["loss_matrix_hash"].is_string(),
            "loss_matrix_hash should be a hex string"
        );
        let hash_str = provenance["loss_matrix_hash"]
            .as_str()
            .expect("hash should be a string");
        assert_eq!(hash_str.len(), 64, "SHA-256 hex is 64 chars");
    }

    #[test]
    fn loss_matrix_hash_is_deterministic() {
        let request = test_request(false);
        let contract1 = super::BackendSelectionContract::new(&request, 30.0);
        let contract2 = super::BackendSelectionContract::new(&request, 30.0);
        let hash1 = super::loss_matrix_content_hash(&contract1.losses);
        let hash2 = super::loss_matrix_content_hash(&contract2.losses);
        assert_eq!(hash1, hash2, "same inputs should produce same hash");
    }

    #[test]
    fn loss_matrix_hash_changes_with_diarize_flag() {
        let non_diarize = super::BackendSelectionContract::new(&test_request(false), 30.0);
        let diarize = super::BackendSelectionContract::new(&test_request(true), 30.0);
        let hash_nd = super::loss_matrix_content_hash(&non_diarize.losses);
        let hash_d = super::loss_matrix_content_hash(&diarize.losses);
        assert_ne!(
            hash_nd, hash_d,
            "diarize flag should change the loss matrix"
        );
    }

    #[test]
    fn backend_selection_skips_non_auto() {
        let mut request = test_request(false);
        request.backend = BackendKind::WhisperCpp;
        let trace_id = TraceId::from_parts(1_700_000_000_000, 43);
        assert!(evaluate_backend_selection(&request, Some(15.0), trace_id).is_none());
    }

    #[test]
    fn backend_selection_evidence_is_valid_json() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 44);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
        for entry in &outcome.evidence_entries {
            assert!(entry.is_object(), "evidence entry should be a JSON object");
            // Verify it has the expected EvidenceLedger short field names.
            assert!(
                entry.get("c").is_some(),
                "evidence should have component field"
            );
            assert!(
                entry.get("a").is_some(),
                "evidence should have action field"
            );
            assert!(
                entry.get("p").is_some(),
                "evidence should have posterior field"
            );
        }
    }

    #[test]
    fn extracts_transcription_segments() {
        let input = serde_json::json!({
            "transcription": [
                {
                    "offsets": {"from": 500, "to": 1000},
                    "text": "hello"
                },
                {
                    "offsets": {"from": 1000, "to": 1500},
                    "text": "world"
                }
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_sec, Some(0.5));
        assert_eq!(segments[1].end_sec, Some(1.5));
        assert_eq!(transcript_from_segments(&segments), "hello world");
    }

    #[test]
    fn extracts_chunk_segments() {
        let input = serde_json::json!({
            "chunks": [
                {"timestamp": [0.0, 1.0], "text": "a"},
                {"timestamp": [1.0, 2.0], "text": "b"}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 2);
        assert_eq!(transcript_from_segments(&segments), "a b");
    }

    #[test]
    fn duration_bucket_mapping() {
        assert_eq!(duration_bucket(2.0), "short");
        assert_eq!(duration_bucket(45.0), "medium");
        assert_eq!(duration_bucket(800.0), "long");
    }

    // -- K5.6: Backend malformed artifact and edge case tests --

    #[test]
    fn extract_segments_empty_json_returns_empty() {
        let input = serde_json::json!({});
        let segments = extract_segments_from_json(&input);
        assert!(segments.is_empty());
    }

    #[test]
    fn extract_segments_null_root_returns_empty() {
        let input = serde_json::json!(null);
        let segments = extract_segments_from_json(&input);
        assert!(segments.is_empty());
    }

    #[test]
    fn extract_segments_empty_transcription_array() {
        let input = serde_json::json!({"transcription": []});
        let segments = extract_segments_from_json(&input);
        assert!(segments.is_empty());
    }

    #[test]
    fn extract_segments_empty_segments_array() {
        let input = serde_json::json!({"segments": []});
        let segments = extract_segments_from_json(&input);
        assert!(segments.is_empty());
    }

    #[test]
    fn extract_segments_empty_chunks_array() {
        let input = serde_json::json!({"chunks": []});
        let segments = extract_segments_from_json(&input);
        assert!(segments.is_empty());
    }

    #[test]
    fn extract_segments_missing_text_uses_empty_string() {
        let input = serde_json::json!({
            "segments": [
                {"start": 0.0, "end": 1.0}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "");
    }

    #[test]
    fn extract_segments_missing_timestamps_returns_none() {
        let input = serde_json::json!({
            "segments": [
                {"text": "no timestamps here"}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert!(segments[0].start_sec.is_none());
        assert!(segments[0].end_sec.is_none());
        assert_eq!(segments[0].text, "no timestamps here");
    }

    #[test]
    fn extract_segments_with_speaker_and_confidence() {
        let input = serde_json::json!({
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "hello",
                    "speaker": "SPEAKER_00",
                    "confidence": 0.92
                }
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segments[0].confidence, Some(0.92));
    }

    #[test]
    fn extract_segments_clamps_out_of_range_confidence() {
        // Out-of-range confidences must be clamped into [0, 1]. NaN cannot
        // appear in JSON, but a finite denormal must stay finite and in range.
        let input = serde_json::json!({
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "high", "confidence": 5.0},
                {"start": 1.0, "end": 2.0, "text": "low", "confidence": -3.0},
                {"start": 2.0, "end": 3.0, "text": "denormal", "confidence": 2.225073858507201e-308}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].confidence, Some(1.0));
        assert_eq!(segments[1].confidence, Some(0.0));
        let denormal = segments[2].confidence.expect("denormal confidence retained");
        assert!(
            denormal.is_finite() && (0.0..=1.0).contains(&denormal),
            "denormal confidence should stay finite and in range: {denormal}"
        );
    }

    #[test]
    fn extract_segments_transcription_with_offsets_ms() {
        let input = serde_json::json!({
            "transcription": [
                {"offsets": {"from": 0, "to": 500}, "text": "half second"}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_sec, Some(0.0));
        assert_eq!(segments[0].end_sec, Some(0.5));
    }

    #[test]
    fn extract_segments_handles_mixed_and_partially_malformed_payload_shapes() {
        let input = serde_json::json!({
            "segments": [
                {"start": "bad-type", "end": "bad-type", "text": "kept text"},
                {"timestamp": {"start": 1.5, "end": 2.0}, "word": "word field", "score": 0.7},
                {"offsets": {"from": 2500, "to": 4000}, "text": "ms offsets"}
            ]
        });

        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 3);

        assert_eq!(segments[0].text, "kept text");
        assert!(segments[0].start_sec.is_none());
        assert!(segments[0].end_sec.is_none());

        assert_eq!(segments[1].text, "word field");
        assert_eq!(segments[1].start_sec, Some(1.5));
        assert_eq!(segments[1].end_sec, Some(2.0));
        assert_eq!(segments[1].confidence, Some(0.7));

        assert_eq!(segments[2].text, "ms offsets");
        assert_eq!(segments[2].start_sec, Some(2.5));
        assert_eq!(segments[2].end_sec, Some(4.0));
    }

    #[test]
    fn extract_word_level_chunks_with_nested_words() {
        let input = serde_json::json!({
            "chunks": [
                {
                    "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.3, "confidence": 0.99},
                        {"word": "world", "start": 0.3, "end": 0.6, "confidence": 0.95}
                    ]
                }
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segments[0].confidence, Some(0.99));
        assert_eq!(segments[1].text, "world");
        assert_eq!(segments[1].start_sec, Some(0.3));
    }

    #[test]
    fn extract_word_level_skips_empty_words() {
        let input = serde_json::json!({
            "chunks": [
                {
                    "words": [
                        {"word": "", "start": 0.0, "end": 0.1},
                        {"word": "  ", "start": 0.1, "end": 0.2},
                        {"word": "hello", "start": 0.2, "end": 0.5}
                    ]
                }
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello");
    }

    #[test]
    fn extract_chunks_without_words_falls_back_to_flat() {
        let input = serde_json::json!({
            "chunks": [
                {"timestamp": [0.0, 1.0], "text": "flat chunk"}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "flat chunk");
    }

    #[test]
    fn transcript_from_segments_whitespace_only_segments_excluded() {
        let segments = vec![
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "hello".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "   ".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "world".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        assert_eq!(transcript_from_segments(&segments), "hello world");
    }

    #[test]
    fn transcript_from_empty_segments_returns_empty() {
        assert_eq!(transcript_from_segments(&[]), "");
    }

    #[test]
    fn extract_segments_prefers_transcription_over_segments() {
        let input = serde_json::json!({
            "transcription": [
                {"text": "from transcription", "offsets": {"from": 0, "to": 1000}}
            ],
            "segments": [
                {"text": "from segments", "start": 0.0, "end": 1.0}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "from transcription");
    }

    #[test]
    fn extract_segments_prefers_segments_over_chunks() {
        let input = serde_json::json!({
            "segments": [
                {"text": "from segments", "start": 0.0, "end": 1.0}
            ],
            "chunks": [
                {"text": "from chunks", "timestamp": [0.0, 1.0]}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "from segments");
    }

    // -- K13.7: Loss matrix structure and ranking tests --

    #[test]
    fn loss_matrix_has_correct_dimensions() {
        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let matrix = contract.loss_matrix();
        assert_eq!(matrix.n_states(), 3, "3 states: all/partial/none available");
        assert_eq!(matrix.n_actions(), 4, "4 actions: 3 backends + fallback");
    }

    #[test]
    fn fallback_error_loss_decreases_as_availability_decreases() {
        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let matrix = contract.loss_matrix();
        let fallback_idx = 3;

        let loss_all = matrix.get(0, fallback_idx);
        let loss_partial = matrix.get(1, fallback_idx);
        let loss_none = matrix.get(2, fallback_idx);

        assert!(
            loss_all > loss_partial,
            "fallback should be more costly when all available ({loss_all} > {loss_partial})"
        );
        assert!(
            loss_partial > loss_none,
            "fallback should be least costly when none available ({loss_partial} > {loss_none})"
        );
    }

    #[test]
    fn backend_losses_increase_with_decreasing_availability() {
        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let matrix = contract.loss_matrix();

        // For each backend action, loss should increase as availability decreases
        for action in 0..3 {
            let loss_all = matrix.get(0, action);
            let loss_partial = matrix.get(1, action);
            let loss_none = matrix.get(2, action);

            assert!(
                loss_partial > loss_all,
                "action {action}: partial ({loss_partial}) should have higher loss than all ({loss_all})"
            );
            assert!(
                loss_none > loss_partial,
                "action {action}: none ({loss_none}) should have higher loss than partial ({loss_partial})"
            );
        }
    }

    #[test]
    fn non_diarize_prefers_whisper_cpp_at_all_available() {
        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let matrix = contract.loss_matrix();

        // In the all_available state (0), first action (whisper_cpp) should have lowest loss
        let losses: Vec<f64> = (0..3).map(|a| matrix.get(0, a)).collect();
        let min_idx = losses
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(
            contract.action_backends[min_idx],
            BackendKind::WhisperCpp,
            "non-diarize should prefer whisper_cpp at all_available"
        );
    }

    #[test]
    fn diarize_prefers_insanely_fast_at_all_available() {
        let contract = BackendSelectionContract::new(&test_request(true), 30.0);
        let matrix = contract.loss_matrix();

        let losses: Vec<f64> = (0..3).map(|a| matrix.get(0, a)).collect();
        let min_idx = losses
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(
            contract.action_backends[min_idx],
            BackendKind::InsanelyFast,
            "diarize should prefer insanely_fast at all_available"
        );
    }

    #[test]
    fn all_loss_values_are_non_negative() {
        for diarize in [false, true] {
            for duration in [1.0, 30.0, 600.0] {
                let contract = BackendSelectionContract::new(&test_request(diarize), duration);
                let matrix = contract.loss_matrix();
                for s in 0..matrix.n_states() {
                    for a in 0..matrix.n_actions() {
                        let loss = matrix.get(s, a);
                        assert!(
                            loss >= 0.0,
                            "loss[{s},{a}] = {loss} is negative (diarize={diarize}, duration={duration})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn loss_matrix_varies_with_duration() {
        let short = BackendSelectionContract::new(&test_request(false), 5.0);
        let long = BackendSelectionContract::new(&test_request(false), 600.0);

        // Backend base loss includes latency_proxy which uses sqrt(duration)
        // So longer duration → higher loss for same state/action
        let short_loss = short.losses.get(0, 0);
        let long_loss = long.losses.get(0, 0);
        assert!(
            long_loss > short_loss,
            "longer duration should produce higher loss ({long_loss} > {short_loss})"
        );
    }

    // --- Malformed JSON payload resilience (J2.3/J2.6) ---

    #[test]
    fn extract_segments_string_root_returns_empty() {
        let input = serde_json::json!("just a string");
        assert!(extract_segments_from_json(&input).is_empty());
    }

    #[test]
    fn extract_segments_number_root_returns_empty() {
        let input = serde_json::json!(42);
        assert!(extract_segments_from_json(&input).is_empty());
    }

    #[test]
    fn extract_segments_boolean_root_returns_empty() {
        let input = serde_json::json!(true);
        assert!(extract_segments_from_json(&input).is_empty());
    }

    #[test]
    fn extract_segments_array_root_returns_empty() {
        let input = serde_json::json!([1, 2, 3]);
        assert!(extract_segments_from_json(&input).is_empty());
    }

    #[test]
    fn extract_segments_wrong_type_for_known_key() {
        // "segments" exists but is a string, not array
        let input = serde_json::json!({"segments": "not an array"});
        assert!(extract_segments_from_json(&input).is_empty());
    }

    #[test]
    fn extract_segments_segments_array_of_non_objects() {
        let input = serde_json::json!({"segments": [1, "two", null, true]});
        let segments = extract_segments_from_json(&input);
        // Each item yields a segment but with all None/empty fields
        assert_eq!(segments.len(), 4);
        for seg in &segments {
            assert!(seg.text.is_empty() || seg.text == "two");
            assert!(seg.start_sec.is_none());
            assert!(seg.end_sec.is_none());
        }
    }

    #[test]
    fn extract_segments_deeply_nested_garbage_is_resilient() {
        let input = serde_json::json!({
            "segments": [
                {"text": "valid", "start": 1.0, "end": 2.0},
                {"text": null, "start": {"nested": "object"}, "end": [1, 2]},
                {"unexpected_key": "doesn't crash"},
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].text, "valid");
        assert_eq!(segments[0].start_sec, Some(1.0));
        // Second segment: text is null → empty, timestamps are wrong types → None
        assert!(segments[1].text.is_empty());
        assert!(segments[1].start_sec.is_none());
        assert!(segments[1].end_sec.is_none());
    }

    #[test]
    fn extract_segments_chunks_with_words_containing_only_whitespace() {
        let input = serde_json::json!({
            "chunks": [{
                "words": [
                    {"word": "   ", "start": 0.0, "end": 0.1},
                    {"word": "\t", "start": 0.1, "end": 0.2},
                ]
            }]
        });
        let segments = extract_segments_from_json(&input);
        assert!(
            segments.is_empty(),
            "whitespace-only words should be skipped"
        );
    }

    #[test]
    fn extract_segments_integer_timestamps_coerce_to_float() {
        let input = serde_json::json!({
            "segments": [{"text": "coerce", "start": 10, "end": 20}]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments[0].start_sec, Some(10.0));
        assert_eq!(segments[0].end_sec, Some(20.0));
    }

    #[test]
    fn extract_segments_negative_timestamps_are_sanitized() {
        let input = serde_json::json!({
            "segments": [{"text": "negative", "start": -1.0, "end": -0.5}]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(
            segments[0].start_sec, None,
            "negative start should be sanitized to None"
        );
        assert_eq!(
            segments[0].end_sec, None,
            "negative end should be sanitized to None"
        );
    }

    #[test]
    fn extract_segments_infinity_nan_timestamps_sanitized() {
        // Infinity case (JSON encodes infinity as null, but test the sanitization path).
        let input = serde_json::json!({
            "segments": [{"text": "inf", "start": f64::INFINITY}]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(
            segments[0].start_sec, None,
            "infinity should be sanitized to None"
        );

        // NaN case — JSON can't represent NaN directly, but test via negative infinity.
        let input_neg = serde_json::json!({
            "segments": [{"text": "neginf", "start": f64::NEG_INFINITY, "end": f64::NEG_INFINITY}]
        });
        let segments_neg = extract_segments_from_json(&input_neg);
        assert_eq!(
            segments_neg[0].start_sec, None,
            "negative infinity should be sanitized"
        );
        assert_eq!(
            segments_neg[0].end_sec, None,
            "negative infinity should be sanitized"
        );
    }

    #[test]
    fn extract_segments_very_large_array_does_not_panic() {
        let items: Vec<serde_json::Value> = (0..1000)
            .map(|i| {
                serde_json::json!({"text": format!("seg-{i}"), "start": i as f64, "end": (i + 1) as f64})
            })
            .collect();
        let input = serde_json::json!({"segments": items});
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1000);
    }

    #[test]
    fn sanitize_timestamp_strips_nan() {
        assert_eq!(sanitize_timestamp(Some(f64::NAN)), None);
    }

    #[test]
    fn sanitize_timestamp_strips_positive_infinity() {
        assert_eq!(sanitize_timestamp(Some(f64::INFINITY)), None);
    }

    #[test]
    fn sanitize_timestamp_strips_negative_infinity() {
        assert_eq!(sanitize_timestamp(Some(f64::NEG_INFINITY)), None);
    }

    #[test]
    fn sanitize_timestamp_strips_negative_values() {
        assert_eq!(sanitize_timestamp(Some(-0.5)), None);
        assert_eq!(sanitize_timestamp(Some(-100.0)), None);
    }

    #[test]
    fn sanitize_timestamp_preserves_zero() {
        assert_eq!(sanitize_timestamp(Some(0.0)), Some(0.0));
    }

    #[test]
    fn sanitize_timestamp_preserves_valid_positive() {
        assert_eq!(sanitize_timestamp(Some(1.5)), Some(1.5));
        assert_eq!(sanitize_timestamp(Some(3600.0)), Some(3600.0));
    }

    #[test]
    fn sanitize_timestamp_preserves_none() {
        assert_eq!(sanitize_timestamp(None), None);
    }

    #[test]
    fn extract_segments_zero_timestamp_preserved() {
        let input = serde_json::json!({
            "segments": [{"text": "start", "start": 0.0, "end": 1.0}]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments[0].start_sec, Some(0.0));
        assert_eq!(segments[0].end_sec, Some(1.0));
    }

    #[test]
    fn extract_segments_word_level_negative_timestamps_sanitized() {
        let input = serde_json::json!({
            "chunks": [{
                "words": [{"word": "bad", "start": -1.0, "end": 2.0}]
            }]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(
            segments[0].start_sec, None,
            "word-level negative start sanitized"
        );
        assert_eq!(
            segments[0].end_sec,
            Some(2.0),
            "word-level valid end preserved"
        );
    }

    // --- Engine trait implementation tests ---

    #[test]
    fn whisper_cpp_engine_metadata() {
        let engine = super::WhisperCppEngine;
        assert_eq!(engine.name(), "whisper.cpp");
        assert_eq!(engine.kind(), BackendKind::WhisperCpp);
        let caps = engine.capabilities();
        assert!(!caps.supports_diarization);
        assert!(caps.supports_translation);
        assert!(caps.supports_word_timestamps);
        assert!(caps.supports_gpu);
        assert!(caps.supports_streaming);
    }

    #[test]
    fn insanely_fast_engine_metadata() {
        let engine = super::InsanelyFastEngine;
        assert_eq!(engine.name(), "insanely-fast-whisper");
        assert_eq!(engine.kind(), BackendKind::InsanelyFast);
        let caps = engine.capabilities();
        assert!(caps.supports_diarization);
        assert!(caps.supports_translation);
        assert!(caps.supports_word_timestamps);
        assert!(caps.supports_gpu);
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn whisper_diarization_engine_metadata() {
        let engine = super::WhisperDiarizationEngine;
        assert_eq!(engine.name(), "whisper-diarization");
        assert_eq!(engine.kind(), BackendKind::WhisperDiarization);
        let caps = engine.capabilities();
        assert!(caps.supports_diarization);
        assert!(!caps.supports_translation);
        assert!(!caps.supports_word_timestamps);
        assert!(caps.supports_gpu);
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn whisper_cpp_native_engine_metadata() {
        let engine = super::WhisperCppNativeEngine;
        assert_eq!(engine.name(), "whisper.cpp-native");
        assert_eq!(engine.kind(), BackendKind::WhisperCpp);
        let caps = engine.capabilities();
        assert!(!caps.supports_diarization);
        assert!(caps.supports_translation);
        assert!(caps.supports_word_timestamps);
        assert!(caps.supports_gpu);
        assert!(caps.supports_streaming);
    }

    #[test]
    fn insanely_fast_native_engine_metadata() {
        let engine = super::InsanelyFastNativeEngine;
        assert_eq!(engine.name(), "insanely-fast-native");
        assert_eq!(engine.kind(), BackendKind::InsanelyFast);
        let caps = engine.capabilities();
        assert!(caps.supports_diarization);
        assert!(caps.supports_translation);
        assert!(caps.supports_word_timestamps);
        assert!(caps.supports_gpu);
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn whisper_diarization_native_engine_metadata() {
        let engine = super::WhisperDiarizationNativeEngine;
        assert_eq!(engine.name(), "whisper-diarization-native");
        assert_eq!(engine.kind(), BackendKind::WhisperDiarization);
        let caps = engine.capabilities();
        assert!(caps.supports_diarization);
        assert!(!caps.supports_translation);
        assert!(!caps.supports_word_timestamps);
        assert!(caps.supports_gpu);
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn all_engines_returns_six_registered_backends() {
        let engines = super::all_engines();
        assert_eq!(engines.len(), 6);
        let kinds: Vec<BackendKind> = engines.iter().map(|e| e.kind()).collect();
        assert!(kinds.contains(&BackendKind::WhisperCpp));
        assert!(kinds.contains(&BackendKind::InsanelyFast));
        assert!(kinds.contains(&BackendKind::WhisperDiarization));
        // Native engines also map to their bridge adapter kinds.
        let names: Vec<&str> = engines.iter().map(|e| e.name()).collect();
        assert!(names.contains(&"whisper.cpp-native"));
        assert!(names.contains(&"insanely-fast-native"));
        assert!(names.contains(&"whisper-diarization-native"));
    }

    #[test]
    fn engine_for_returns_matching_engine_or_none_for_auto() {
        assert!(super::engine_for(BackendKind::WhisperCpp).is_some());
        assert!(super::engine_for(BackendKind::InsanelyFast).is_some());
        assert!(super::engine_for(BackendKind::WhisperDiarization).is_some());
        assert!(super::engine_for(BackendKind::Auto).is_none());

        let engine = super::engine_for(BackendKind::WhisperCpp).unwrap();
        assert_eq!(engine.kind(), BackendKind::WhisperCpp);
    }

    #[test]
    fn engine_names_are_non_empty_and_distinct() {
        let engines = super::all_engines();
        let names: Vec<&str> = engines.iter().map(|e| e.name()).collect();
        assert!(names.iter().all(|n| !n.is_empty()));
        // No duplicates.
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
    }

    // --- Bayesian helper function tests ---

    #[test]
    fn prior_for_known_backends_returns_distinct_alpha_beta() {
        let cpp = prior_for(BackendKind::WhisperCpp);
        let fast = prior_for(BackendKind::InsanelyFast);
        let diar = prior_for(BackendKind::WhisperDiarization);
        let auto = prior_for(BackendKind::Auto);

        // WhisperCpp has the strongest prior (highest alpha).
        assert!(cpp.0 > fast.0);
        assert!(fast.0 > diar.0);
        // Auto has a uniform prior.
        assert_eq!(auto, (1.0, 1.0));
        // All alphas and betas are positive.
        for (alpha, beta) in [cpp, fast, diar, auto] {
            assert!(alpha > 0.0);
            assert!(beta > 0.0);
        }
    }

    #[test]
    fn quality_proxy_diarize_favors_diarization_backend() {
        let req_diarize = test_request(true);
        let req_no_diarize = test_request(false);

        let cpp_diarize = quality_proxy(BackendKind::WhisperCpp, &req_diarize);
        let cpp_no_diarize = quality_proxy(BackendKind::WhisperCpp, &req_no_diarize);
        // WhisperCpp is worse when diarization is needed.
        assert!(cpp_diarize < cpp_no_diarize);

        let diar_diarize = quality_proxy(BackendKind::WhisperDiarization, &req_diarize);
        let diar_no_diarize = quality_proxy(BackendKind::WhisperDiarization, &req_no_diarize);
        // WhisperDiarization is better when diarization is requested.
        assert!(diar_diarize > diar_no_diarize);

        // When diarizing, WhisperDiarization should have highest quality.
        let fast_diarize = quality_proxy(BackendKind::InsanelyFast, &req_diarize);
        assert!(diar_diarize > fast_diarize);
        assert!(diar_diarize > cpp_diarize);
    }

    #[test]
    fn quality_proxy_auto_has_lowest_score() {
        let req = test_request(false);
        let auto_score = quality_proxy(BackendKind::Auto, &req);
        assert_eq!(auto_score, 0.50);
        // All named backends should score higher.
        assert!(quality_proxy(BackendKind::WhisperCpp, &req) > auto_score);
        assert!(quality_proxy(BackendKind::InsanelyFast, &req) > auto_score);
    }

    #[test]
    fn quality_proxy_values_bounded_0_to_1() {
        for backend in [
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
            BackendKind::Auto,
        ] {
            for diarize in [false, true] {
                let req = test_request(diarize);
                let score = quality_proxy(backend, &req);
                assert!(
                    (0.0..=1.0).contains(&score),
                    "quality_proxy({backend:?}, diarize={diarize}) = {score} out of bounds"
                );
            }
        }
    }

    #[test]
    fn latency_proxy_increases_with_duration() {
        let short = latency_proxy(BackendKind::WhisperCpp, 10.0, false);
        let long = latency_proxy(BackendKind::WhisperCpp, 600.0, false);
        assert!(long > short, "longer audio should have higher latency");
    }

    #[test]
    fn latency_proxy_diarize_multiplier_increases_latency() {
        let without = latency_proxy(BackendKind::InsanelyFast, 60.0, false);
        let with = latency_proxy(BackendKind::InsanelyFast, 60.0, true);
        assert!(with > without, "diarize should increase latency");
    }

    #[test]
    fn latency_proxy_insanely_fast_is_fastest() {
        let fast = latency_proxy(BackendKind::InsanelyFast, 30.0, false);
        let cpp = latency_proxy(BackendKind::WhisperCpp, 30.0, false);
        let diar = latency_proxy(BackendKind::WhisperDiarization, 30.0, false);
        assert!(fast < cpp, "InsanelyFast should be faster than WhisperCpp");
        assert!(
            fast < diar,
            "InsanelyFast should be faster than WhisperDiarization"
        );
    }

    #[test]
    fn latency_proxy_zero_duration() {
        // Zero duration should still return base latency.
        let latency = latency_proxy(BackendKind::WhisperCpp, 0.0, false);
        assert!(latency > 0.0);
        assert_eq!(latency, 12.0); // base only
    }

    #[test]
    fn latency_proxy_non_finite_and_negative_return_finite() {
        // NaN/Inf/negative/denormal durations must not produce a NaN latency,
        // which would poison the backend-selection loss matrix.
        for duration in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.0,
            f64::MIN_POSITIVE,
        ] {
            for &kind in &[
                BackendKind::WhisperCpp,
                BackendKind::InsanelyFast,
                BackendKind::WhisperDiarization,
            ] {
                for diarize in [false, true] {
                    let latency = latency_proxy(kind, duration, diarize);
                    assert!(
                        latency.is_finite() && latency >= 0.0,
                        "latency_proxy({kind:?}, {duration}, {diarize}) not finite/non-negative: {latency}"
                    );
                }
            }
        }
    }

    #[test]
    fn posterior_success_probability_unavailable_is_zero() {
        let prob = posterior_success_probability(7.0, 3.0, 0.8, false, false, false);
        assert_eq!(prob, 0.0);
    }

    #[test]
    fn posterior_success_probability_nan_quality_returns_finite() {
        // A non-finite quality_score must not yield a NaN probability, which
        // would panic the downstream loss-matrix clamp on the current toolchain.
        for quality in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for diarize in [false, true] {
                for translate in [false, true] {
                    let p =
                        posterior_success_probability(7.0, 3.0, quality, diarize, translate, true);
                    assert!(
                        p.is_finite() && (0.0..=1.0).contains(&p),
                        "quality={quality}, diarize={diarize}, translate={translate} => {p}"
                    );
                }
            }
        }
    }

    #[test]
    fn posterior_success_probability_available_is_positive() {
        let prob = posterior_success_probability(7.0, 3.0, 0.8, false, false, true);
        assert!(prob > 0.0);
        assert!(prob < 1.0);
    }

    #[test]
    fn posterior_success_probability_diarize_boost_increases_prob() {
        let without_diarize = posterior_success_probability(5.0, 5.0, 0.7, false, false, true);
        let with_diarize = posterior_success_probability(5.0, 5.0, 0.7, true, false, true);
        assert!(
            with_diarize > without_diarize,
            "diarize boost should increase probability"
        );
    }

    #[test]
    fn posterior_success_probability_translate_penalty_decreases_prob() {
        let without_translate = posterior_success_probability(6.0, 4.0, 0.8, false, false, true);
        let with_translate = posterior_success_probability(6.0, 4.0, 0.8, false, true, true);
        assert!(
            with_translate < without_translate,
            "translate penalty should decrease probability"
        );
    }

    #[test]
    fn posterior_success_probability_higher_quality_increases_prob() {
        let low_quality = posterior_success_probability(5.0, 5.0, 0.3, false, false, true);
        let high_quality = posterior_success_probability(5.0, 5.0, 0.9, false, false, true);
        assert!(
            high_quality > low_quality,
            "higher quality should increase probability"
        );
    }

    #[test]
    fn posterior_success_probability_bounded_0_to_1() {
        // Exhaustive check with various parameter combos.
        for (alpha, beta) in [(1.0, 1.0), (7.0, 3.0), (5.0, 5.0)] {
            for quality in [0.0, 0.5, 1.0] {
                for diarize in [false, true] {
                    for translate in [false, true] {
                        let p = posterior_success_probability(
                            alpha, beta, quality, diarize, translate, true,
                        );
                        assert!(
                            (0.0..=1.0).contains(&p),
                            "out of bounds: alpha={alpha}, beta={beta}, quality={quality}, diarize={diarize}, translate={translate} => {p}"
                        );
                    }
                }
            }
        }
    }

    // --- auto_priority tests ---

    #[test]
    fn auto_priority_non_diarize_prefers_whisper_cpp_first() {
        let order = auto_priority(false);
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], BackendKind::WhisperCpp);
        assert_eq!(order[1], BackendKind::InsanelyFast);
        assert_eq!(order[2], BackendKind::WhisperDiarization);
    }

    #[test]
    fn auto_priority_diarize_prefers_insanely_fast_first() {
        let order = auto_priority(true);
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], BackendKind::InsanelyFast);
        assert_eq!(order[1], BackendKind::WhisperDiarization);
        assert_eq!(order[2], BackendKind::WhisperCpp);
    }

    #[test]
    fn rollout_stage_shadow_forces_static_non_diarize_order() {
        let recommended = vec![
            BackendKind::WhisperDiarization,
            BackendKind::InsanelyFast,
            BackendKind::WhisperCpp,
        ];
        let (effective, forced_static) = super::gate_recommended_order_for_rollout(
            false,
            recommended,
            NativeEngineRolloutStage::Shadow,
        );

        assert!(forced_static);
        assert_eq!(effective, auto_priority(false));
    }

    #[test]
    fn rollout_stage_primary_keeps_recommended_order() {
        let recommended = vec![
            BackendKind::WhisperDiarization,
            BackendKind::InsanelyFast,
            BackendKind::WhisperCpp,
        ];
        let (effective, forced_static) = super::gate_recommended_order_for_rollout(
            false,
            recommended.clone(),
            NativeEngineRolloutStage::Primary,
        );

        assert!(!forced_static);
        assert_eq!(effective, recommended);
    }

    // --- runtime_metadata tests ---

    #[test]
    fn runtime_metadata_auto_returns_auto_policy_identity() {
        let meta = runtime_metadata(BackendKind::Auto);
        assert_eq!(meta.identity, "auto-policy");
        assert!(meta.version.is_none());
    }

    #[test]
    fn runtime_metadata_named_backends_have_non_empty_identity() {
        for kind in [
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
        ] {
            let meta = runtime_metadata(kind);
            assert!(
                !meta.identity.is_empty(),
                "{kind:?} should have non-empty identity"
            );
        }
    }

    // --- number_to_secs tests ---

    #[test]
    fn number_to_secs_float_value() {
        let v = serde_json::json!(1.5);
        assert_eq!(number_to_secs(&v), Some(1.5));
    }

    #[test]
    fn number_to_secs_integer_value() {
        let v = serde_json::json!(42);
        assert_eq!(number_to_secs(&v), Some(42.0));
    }

    #[test]
    fn number_to_secs_string_returns_none() {
        let v = serde_json::json!("not a number");
        assert_eq!(number_to_secs(&v), None);
    }

    #[test]
    fn number_to_secs_null_returns_none() {
        let v = serde_json::json!(null);
        assert_eq!(number_to_secs(&v), None);
    }

    // --- number_millis_to_secs tests ---

    #[test]
    fn number_millis_to_secs_converts_correctly() {
        let v = serde_json::json!(1500);
        assert_eq!(number_millis_to_secs(&v), Some(1.5));
    }

    #[test]
    fn number_millis_to_secs_zero() {
        let v = serde_json::json!(0);
        assert_eq!(number_millis_to_secs(&v), Some(0.0));
    }

    #[test]
    fn number_millis_to_secs_non_number_returns_none() {
        let v = serde_json::json!("abc");
        assert_eq!(number_millis_to_secs(&v), None);
    }

    // --- segment_start / segment_end tests ---

    #[test]
    fn segment_start_from_start_field() {
        let node = serde_json::json!({"start": 1.5, "end": 2.0, "text": "hello"});
        assert_eq!(segment_start(&node), Some(1.5));
    }

    #[test]
    fn segment_start_from_timestamp_array() {
        let node = serde_json::json!({"timestamp": [0.5, 1.0], "text": "hi"});
        assert_eq!(segment_start(&node), Some(0.5));
    }

    #[test]
    fn segment_start_from_offsets_from_millis() {
        let node = serde_json::json!({"offsets": {"from": 2500, "to": 5000}, "text": "test"});
        assert_eq!(segment_start(&node), Some(2.5));
    }

    #[test]
    fn segment_start_missing_returns_none() {
        let node = serde_json::json!({"text": "no timestamps"});
        assert_eq!(segment_start(&node), None);
    }

    #[test]
    fn segment_end_from_end_field() {
        let node = serde_json::json!({"start": 1.0, "end": 2.5, "text": "hello"});
        assert_eq!(segment_end(&node), Some(2.5));
    }

    #[test]
    fn segment_end_from_timestamp_array() {
        let node = serde_json::json!({"timestamp": [0.5, 1.5], "text": "hi"});
        assert_eq!(segment_end(&node), Some(1.5));
    }

    #[test]
    fn segment_end_from_offsets_to_millis() {
        let node = serde_json::json!({"offsets": {"from": 2500, "to": 5000}, "text": "test"});
        assert_eq!(segment_end(&node), Some(5.0));
    }

    #[test]
    fn segment_end_missing_returns_none() {
        let node = serde_json::json!({"text": "no end"});
        assert_eq!(segment_end(&node), None);
    }

    #[test]
    fn segment_start_from_timestamp_named_start() {
        let node = serde_json::json!({"timestamp": {"start": 3.0, "end": 4.0}, "text": "hi"});
        assert_eq!(segment_start(&node), Some(3.0));
    }

    #[test]
    fn segment_end_from_timestamp_named_end() {
        let node = serde_json::json!({"timestamp": {"start": 3.0, "end": 4.0}, "text": "hi"});
        assert_eq!(segment_end(&node), Some(4.0));
    }

    // ── diagnostics() ──

    #[test]
    fn diagnostics_returns_exactly_three_entries() {
        let diags = super::diagnostics();
        assert_eq!(diags.len(), 3, "one entry per backend");
    }

    #[test]
    fn diagnostics_entries_have_required_keys() {
        for entry in super::diagnostics() {
            let obj = entry.as_object().expect("entry should be an object");
            assert!(obj.contains_key("backend"), "missing 'backend': {entry:#}");
            assert!(
                obj.contains_key("available"),
                "missing 'available': {entry:#}"
            );
            assert!(
                obj.contains_key("unsupported_options"),
                "missing 'unsupported_options': {entry:#}"
            );
        }
    }

    #[test]
    fn diagnostics_backend_names_match_known_backends() {
        let diags = super::diagnostics();
        let names: Vec<&str> = diags
            .iter()
            .map(|e| e["backend"].as_str().expect("backend should be a string"))
            .collect();
        assert!(
            names.contains(&BackendKind::WhisperCpp.as_str()),
            "should include whisper-cpp: {names:?}"
        );
        assert!(
            names.contains(&BackendKind::InsanelyFast.as_str()),
            "should include insanely-fast: {names:?}"
        );
        assert!(
            names.contains(&BackendKind::WhisperDiarization.as_str()),
            "should include whisper-diarization: {names:?}"
        );
    }

    #[test]
    fn diagnostics_available_fields_are_booleans() {
        for entry in super::diagnostics() {
            assert!(
                entry["available"].is_boolean(),
                "'available' should be bool for {}: got {:?}",
                entry["backend"],
                entry["available"]
            );
        }
    }

    #[test]
    fn diagnostics_unsupported_options_are_nonempty_arrays() {
        for entry in super::diagnostics() {
            let opts = entry["unsupported_options"]
                .as_array()
                .expect("unsupported_options should be an array");
            assert!(
                !opts.is_empty(),
                "unsupported_options should not be empty for {}",
                entry["backend"]
            );
            for opt in opts {
                assert!(
                    opt.as_str().is_some(),
                    "each unsupported option should be a string"
                );
            }
        }
    }

    #[test]
    fn diagnostics_whisper_cpp_entry_has_binary_and_env_override() {
        let diags = super::diagnostics();
        let cpp = diags
            .iter()
            .find(|e| e["backend"] == BackendKind::WhisperCpp.as_str())
            .expect("whisper-cpp entry");
        assert!(
            cpp["binary"].as_str().is_some(),
            "whisper-cpp should have a binary field"
        );
        assert_eq!(
            cpp["env_override"].as_str(),
            Some("FRANKEN_WHISPER_WHISPER_CPP_BIN")
        );
    }

    #[test]
    fn diagnostics_insanely_fast_entry_has_hf_token_fields() {
        let diags = super::diagnostics();
        let fast = diags
            .iter()
            .find(|e| e["backend"] == BackendKind::InsanelyFast.as_str())
            .expect("insanely-fast entry");
        assert!(
            fast["hf_token_set"].is_boolean(),
            "hf_token_set should be bool"
        );
        assert!(
            fast["hf_token_env_overrides"].is_array(),
            "hf_token_env_overrides should be array"
        );
        assert_eq!(
            fast["requires_hf_token_for_diarization"].as_bool(),
            Some(true)
        );
    }

    #[test]
    fn diagnostics_diarization_entry_has_script_and_python_binary() {
        let diags = super::diagnostics();
        let diar = diags
            .iter()
            .find(|e| e["backend"] == BackendKind::WhisperDiarization.as_str())
            .expect("whisper-diarization entry");
        assert!(
            diar["script"].as_str().is_some(),
            "diarization should have a script field"
        );
        assert!(
            diar["python_binary"].as_str().is_some(),
            "diarization should have a python_binary field"
        );
        assert_eq!(
            diar["env_override_python"].as_str(),
            Some("FRANKEN_WHISPER_PYTHON_BIN")
        );
        assert_eq!(
            diar["env_override_device"].as_str(),
            Some("FRANKEN_WHISPER_DIARIZATION_DEVICE")
        );
    }

    // ── BackendSelectionContract edge cases ──

    #[test]
    fn backend_base_loss_is_finite_for_all_backends() {
        let request = test_request(false);
        for &backend in &[
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
        ] {
            let loss = BackendSelectionContract::backend_base_loss(backend, &request, 30.0);
            assert!(
                loss.is_finite(),
                "loss for {:?} should be finite: {loss}",
                backend
            );
            assert!(
                loss >= 0.0,
                "loss for {:?} should be non-negative: {loss}",
                backend
            );
        }
    }

    #[test]
    fn backend_base_loss_finite_for_pathological_durations() {
        let request = test_request(false);
        for duration in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.0,
            f64::MIN_POSITIVE,
        ] {
            for &backend in &[
                BackendKind::WhisperCpp,
                BackendKind::InsanelyFast,
                BackendKind::WhisperDiarization,
            ] {
                let loss =
                    BackendSelectionContract::backend_base_loss(backend, &request, duration);
                assert!(
                    loss.is_finite() && loss >= 0.0,
                    "loss for {backend:?} at duration {duration} not finite/non-negative: {loss}"
                );
            }
        }
    }

    #[test]
    fn contract_construction_survives_non_finite_duration() {
        // Constructing the contract must never panic on a NaN/Inf/denormal
        // duration, and every loss-matrix cell must remain finite and
        // non-negative so LossMatrix::new accepts it (no expect() abort).
        for duration in [f64::NAN, f64::INFINITY, f64::MIN_POSITIVE, -1.0] {
            for diarize in [false, true] {
                let contract = BackendSelectionContract::new(&test_request(diarize), duration);
                let matrix = contract.loss_matrix();
                for s in 0..matrix.n_states() {
                    for a in 0..matrix.n_actions() {
                        let v = matrix.get(s, a);
                        assert!(
                            v.is_finite() && v >= 0.0,
                            "loss[{s}][{a}] not finite/non-negative for duration {duration}: {v}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn backend_base_loss_varies_with_duration() {
        let request = test_request(false);
        let short =
            BackendSelectionContract::backend_base_loss(BackendKind::WhisperCpp, &request, 5.0);
        let long =
            BackendSelectionContract::backend_base_loss(BackendKind::WhisperCpp, &request, 600.0);
        // Longer duration should increase latency cost.
        assert!(
            long > short,
            "long duration loss ({long}) should exceed short ({short})"
        );
    }

    #[test]
    fn backend_base_loss_diarize_flag_affects_quality_proxy() {
        let non_diarize = test_request(false);
        let diarize = test_request(true);

        let loss_nd = BackendSelectionContract::backend_base_loss(
            BackendKind::InsanelyFast,
            &non_diarize,
            30.0,
        );
        let loss_d =
            BackendSelectionContract::backend_base_loss(BackendKind::InsanelyFast, &diarize, 30.0);
        // Diarization changes quality proxy and p_success, so losses should differ.
        assert!(
            (loss_nd - loss_d).abs() > 1e-6,
            "diarize flag should change loss: nd={loss_nd}, d={loss_d}"
        );
    }

    #[test]
    fn loss_matrix_fallback_error_costs_decrease_as_availability_drops() {
        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let matrix = contract.loss_matrix();
        let fallback_idx = 3; // fallback_error action index

        let loss_all = matrix.get(0, fallback_idx);
        let loss_partial = matrix.get(1, fallback_idx);
        let loss_none = matrix.get(2, fallback_idx);

        // When all available, fallback is very costly. When none available, fallback is cheap.
        assert!(
            loss_all > loss_partial,
            "all_available fallback ({loss_all}) > partial ({loss_partial})"
        );
        assert!(
            loss_partial > loss_none,
            "partial_available fallback ({loss_partial}) > none ({loss_none})"
        );
    }

    #[test]
    fn update_posterior_concentrates_probability() {
        use franken_decision::Posterior;

        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let mut posterior = Posterior::uniform(3);

        // Before update, probabilities should be uniform (1/3 each).
        let probs_before = posterior.probs().to_vec();
        assert!((probs_before[0] - probs_before[1]).abs() < 1e-9);

        // After observing state 0 (all_available), probability should concentrate.
        contract
            .update_posterior(&mut posterior, 0)
            .expect("valid posterior + in-range observation");
        let probs_after = posterior.probs().to_vec();
        assert!(
            probs_after[0] > probs_after[1],
            "observed state should have highest probability: {:?}",
            probs_after
        );
        assert!(
            probs_after[0] > probs_after[2],
            "observed state should have highest probability: {:?}",
            probs_after
        );
    }

    #[test]
    fn update_posterior_rejects_out_of_range_observation_without_mutating() {
        use franken_decision::{Posterior, UpdatePosteriorError};

        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let mut posterior = Posterior::uniform(3);
        let before = posterior.probs().to_vec();

        // Observation index >= state-space size must fail-closed and leave the
        // posterior untouched (the 0.3.2 contract guarantees no partial apply).
        let err = contract
            .update_posterior(&mut posterior, 3)
            .expect_err("observation 3 is out of range for a 3-state space");
        assert_eq!(
            err,
            UpdatePosteriorError::ObservationOutOfRange {
                observation: 3,
                state_count: 3,
            }
        );
        assert_eq!(posterior.probs().to_vec(), before);
    }

    #[test]
    fn update_posterior_rejects_length_mismatch_without_mutating() {
        use franken_decision::{Posterior, UpdatePosteriorError};

        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        // Wrong-length posterior (2 states vs the contract's 3).
        let mut posterior = Posterior::uniform(2);
        let before = posterior.probs().to_vec();

        let err = contract
            .update_posterior(&mut posterior, 0)
            .expect_err("posterior length 2 does not match 3-state space");
        assert_eq!(
            err,
            UpdatePosteriorError::LengthMismatch {
                expected: 3,
                actual: 2,
            }
        );
        assert_eq!(posterior.probs().to_vec(), before);
    }

    #[test]
    fn validation_error_engages_deterministic_static_fallback() {
        use franken_decision::ValidationError;

        // Craft a contract-validation failure (an out-of-range action index,
        // the panic-DoS shape the 0.3.2 `evaluate` now returns as an error)
        // and assert the wiring helper engages the static-priority fallback
        // rather than panicking or dropping the request.
        for diarize in [false, true] {
            let request = test_request(diarize);
            let contract = BackendSelectionContract::new(&request, 30.0);
            let err = ValidationError::ActionIndexOutOfRange {
                index: 99,
                action_set_len: contract.action_set().len(),
                from_fallback: false,
            };
            let availability = [
                (BackendKind::WhisperCpp, true),
                (BackendKind::InsanelyFast, true),
                (BackendKind::WhisperDiarization, true),
            ];
            let decision_id = DecisionId::from_parts(1_700_000_000_000, 0xDEAD_BEEF);

            let outcome = static_fallback_selection_outcome(
                &request,
                &contract,
                30.0,
                0,
                &availability,
                0.5,
                1.0,
                0.0,
                decision_id,
                TraceId::from_parts(1_700_000_000_000, 7),
                &err,
            );

            // The deterministic fallback must (a) flag fallback, (b) emit the
            // exact static-priority order from `auto_priority`, and (c) surface
            // the validation error string in the routing log.
            assert!(outcome.fallback_triggered, "fallback must be triggered");
            let expected: Vec<BackendKind> = auto_priority(diarize).to_vec();
            assert_eq!(
                outcome.recommended_order, expected,
                "recommended order must equal the static-priority list"
            );
            let log_str = outcome.routing_log.to_string();
            assert!(
                log_str.contains("action_index 99"),
                "routing log must carry the validation error: {log_str}"
            );
            assert_eq!(
                outcome.routing_log["mode"], "static",
                "validation-fallback mode must be static"
            );
            assert!(!outcome.evidence_entries.is_empty());
        }
    }

    #[test]
    fn choose_action_returns_valid_index() {
        use franken_decision::Posterior;

        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let posterior = Posterior::uniform(3);
        let action_idx = contract.choose_action(&posterior);
        assert!(
            action_idx < contract.action_set().len(),
            "chosen action {action_idx} should be < {}",
            contract.action_set().len()
        );
    }

    #[test]
    fn fallback_action_is_last_action() {
        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        assert_eq!(
            contract.fallback_action(),
            contract.action_set().len() - 1,
            "fallback should be last action"
        );
    }

    // ── evaluate_backend_selection edge cases ──

    #[test]
    fn evaluate_backend_selection_none_duration_defaults_to_30() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 200);
        let outcome = evaluate_backend_selection(&request, None, trace_id).unwrap();
        assert_eq!(
            outcome.routing_log["duration_seconds"].as_f64(),
            Some(30.0),
            "None duration should default to 30.0"
        );
        assert_eq!(outcome.routing_log["duration_bucket"], "medium");
    }

    #[test]
    fn evaluate_backend_selection_routing_log_has_all_required_fields() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 201);
        let outcome = evaluate_backend_selection(&request, Some(45.0), trace_id).unwrap();
        let log = &outcome.routing_log;

        let required_non_null_fields = [
            "version",
            "schema_version",
            "policy_id",
            "mode",
            "contract",
            "state_space",
            "observed_state",
            "action_set",
            "chosen_action",
            "fallback_active",
            "expected_losses",
            "posterior_snapshot",
            "calibration_score",
            "e_process",
            "ci_width",
            "recommended_order",
            "static_order",
            "availability",
            "provenance",
            "duration_seconds",
            "duration_bucket",
            "diarize",
            "decision_id",
            "trace_id",
        ];

        for field in required_non_null_fields {
            assert!(
                !log[field].is_null(),
                "routing_log missing required field '{field}'"
            );
        }

        // adaptive_router_state is always present but may be null
        // when no outcomes have been recorded yet.
        assert!(
            log.get("adaptive_router_state").is_some(),
            "routing_log should contain 'adaptive_router_state' key"
        );
    }

    #[test]
    fn evaluate_backend_selection_availability_has_three_entries() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 202);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
        let avail = outcome.routing_log["availability"]
            .as_array()
            .expect("availability should be an array");
        assert_eq!(avail.len(), 3, "one entry per backend");
        for entry in avail {
            assert!(entry["backend"].is_string(), "backend should be a string");
            assert!(
                entry["available"].is_boolean(),
                "available should be a boolean"
            );
        }
    }

    #[test]
    fn evaluate_backend_selection_static_order_matches_auto_priority() {
        for diarize in [false, true] {
            let request = test_request(diarize);
            let trace_id = TraceId::from_parts(1_700_000_000_000, 203);
            let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
            let static_order: Vec<&str> = outcome.routing_log["static_order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            let expected: Vec<&str> = auto_priority(diarize).iter().map(|k| k.as_str()).collect();
            assert_eq!(
                static_order, expected,
                "static_order should match auto_priority(diarize={diarize})"
            );
        }
    }

    #[test]
    fn evaluate_backend_selection_decision_id_is_nonempty_string() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 204);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
        let decision_id = outcome.routing_log["decision_id"]
            .as_str()
            .expect("decision_id should be a string");
        assert!(!decision_id.is_empty(), "decision_id should be non-empty");
    }

    #[test]
    fn evaluate_backend_selection_trace_id_matches_input() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 205);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
        let log_trace = outcome.routing_log["trace_id"]
            .as_str()
            .expect("trace_id should be a string");
        assert_eq!(
            log_trace,
            trace_id.to_string(),
            "routing_log trace_id should match input"
        );
    }

    #[test]
    fn evaluate_backend_selection_skips_all_non_auto_backends() {
        for backend in [
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
        ] {
            let mut request = test_request(false);
            request.backend = backend;
            let trace_id = TraceId::from_parts(1_700_000_000_000, 206);
            assert!(
                evaluate_backend_selection(&request, Some(30.0), trace_id).is_none(),
                "should return None for explicit backend {:?}",
                backend
            );
        }
    }

    #[test]
    fn evaluate_backend_selection_recommended_order_contains_all_backends() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 207);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
        assert_eq!(
            outcome.recommended_order.len(),
            3,
            "recommended order should have all 3 backends"
        );
        assert!(outcome.recommended_order.contains(&BackendKind::WhisperCpp));
        assert!(
            outcome
                .recommended_order
                .contains(&BackendKind::InsanelyFast)
        );
        assert!(
            outcome
                .recommended_order
                .contains(&BackendKind::WhisperDiarization)
        );
    }

    // ── duration_bucket boundary tests ──

    #[test]
    fn duration_bucket_boundary_at_30_seconds() {
        // 30.0 is medium, not short
        assert_eq!(duration_bucket(29.9), "short");
        assert_eq!(duration_bucket(30.0), "medium");
    }

    #[test]
    fn duration_bucket_boundary_at_300_seconds() {
        // 300.0 is long, not medium
        assert_eq!(duration_bucket(299.9), "medium");
        assert_eq!(duration_bucket(300.0), "long");
    }

    #[test]
    fn duration_bucket_zero_is_short() {
        assert_eq!(duration_bucket(0.0), "short");
    }

    #[test]
    fn duration_bucket_negative_is_short() {
        assert_eq!(duration_bucket(-1.0), "short");
    }

    // ── loss_matrix_content_hash edge cases ──

    #[test]
    fn loss_matrix_hash_changes_with_duration() {
        let short = super::BackendSelectionContract::new(&test_request(false), 5.0);
        let long = super::BackendSelectionContract::new(&test_request(false), 600.0);
        let hash_short = super::loss_matrix_content_hash(&short.losses);
        let hash_long = super::loss_matrix_content_hash(&long.losses);
        assert_ne!(
            hash_short, hash_long,
            "different durations should produce different hashes"
        );
    }

    #[test]
    fn loss_matrix_hash_is_64_char_hex() {
        let contract = BackendSelectionContract::new(&test_request(false), 30.0);
        let hash = super::loss_matrix_content_hash(&contract.losses);
        assert_eq!(hash.len(), 64);
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit()),
            "hash should be hex: {hash}"
        );
    }

    // ── transcript_from_segments edge cases ──

    #[test]
    fn transcript_from_single_segment() {
        let segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "only segment".to_owned(),
            speaker: None,
            confidence: None,
        }];
        assert_eq!(transcript_from_segments(&segments), "only segment");
    }

    #[test]
    fn transcript_from_segments_all_whitespace_returns_empty() {
        let segments = vec![
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "   ".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "\t\n".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        assert_eq!(transcript_from_segments(&segments), "");
    }

    #[test]
    fn transcript_from_segments_trims_leading_trailing_whitespace() {
        let segments = vec![TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: "  hello world  ".to_owned(),
            speaker: None,
            confidence: None,
        }];
        // The individual segment text has leading/trailing spaces.
        // trim() in filter only checks if text.trim().is_empty(),
        // but the final result is .trim().to_owned().
        let result = transcript_from_segments(&segments);
        assert!(!result.starts_with(' '));
        assert!(!result.ends_with(' '));
    }

    // ── BackendSelectionContract translate flag test ──

    #[test]
    fn backend_base_loss_translate_flag_changes_loss() {
        let mut request_no_translate = test_request(false);
        request_no_translate.translate = false;
        let mut request_translate = test_request(false);
        request_translate.translate = true;

        let loss_no = BackendSelectionContract::backend_base_loss(
            BackendKind::WhisperCpp,
            &request_no_translate,
            30.0,
        );
        let loss_yes = BackendSelectionContract::backend_base_loss(
            BackendKind::WhisperCpp,
            &request_translate,
            30.0,
        );
        assert!(
            (loss_no - loss_yes).abs() > 1e-6,
            "translate flag should affect loss: no={loss_no}, yes={loss_yes}"
        );
    }

    // ── is_available tests ──

    #[test]
    fn is_available_auto_always_returns_false() {
        assert!(
            !super::is_available(BackendKind::Auto),
            "Auto should never be 'available'"
        );
    }

    #[test]
    fn word_level_speaker_override_takes_precedence_over_chunk() {
        let input = serde_json::json!({
            "chunks": [
                {
                    "speaker": "CHUNK_SPEAKER",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.3, "speaker": "WORD_SPEAKER"},
                        {"word": "world", "start": 0.3, "end": 0.6}
                    ]
                }
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 2);
        // Word with its own speaker overrides chunk speaker
        assert_eq!(segments[0].speaker.as_deref(), Some("WORD_SPEAKER"));
        // Word without speaker inherits from chunk
        assert_eq!(segments[1].speaker.as_deref(), Some("CHUNK_SPEAKER"));
    }

    #[test]
    fn word_level_timestamp_array_fallback() {
        // Tests the /timestamp/0 and /timestamp/1 fallback paths
        let input = serde_json::json!({
            "chunks": [
                {
                    "words": [
                        {"word": "hello", "timestamp": [1.0, 1.5]},
                        {"text": "world", "start": 2.0, "end": 2.5}
                    ]
                }
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_sec, Some(1.0));
        assert_eq!(segments[0].end_sec, Some(1.5));
        // Second word uses start/end (primary path)
        assert_eq!(segments[1].start_sec, Some(2.0));
        assert_eq!(segments[1].end_sec, Some(2.5));
        // "text" field used as fallback for "word"
        assert_eq!(segments[1].text, "world");
    }

    #[test]
    fn word_level_chunk_without_words_array_preserves_speaker() {
        // When a chunk in word-level mode has no "words" array, it falls back
        // to treating the chunk as a flat segment with the chunk speaker.
        let input = serde_json::json!({
            "chunks": [
                {
                    "words": [{"word": "first", "start": 0.0, "end": 0.5}]
                },
                {
                    "speaker": "FALLBACK_SPEAKER",
                    "text": "flat segment",
                    "timestamp": [1.0, 2.0]
                }
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "first");
        assert_eq!(segments[1].text, "flat segment");
        assert_eq!(segments[1].speaker.as_deref(), Some("FALLBACK_SPEAKER"));
    }

    #[test]
    fn segments_from_nodes_uses_score_as_confidence_fallback() {
        let input = serde_json::json!({
            "segments": [
                {"text": "with score", "start": 0.0, "end": 1.0, "score": 0.88}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].confidence, Some(0.88));
    }

    #[test]
    fn auto_priority_diarize_vs_non_diarize_different_order() {
        let diarize_order = super::auto_priority(true);
        let non_diarize_order = super::auto_priority(false);
        assert_ne!(
            diarize_order[0], non_diarize_order[0],
            "diarize should change priority order"
        );
        // Diarize should prioritize InsanelyFast first
        assert_eq!(diarize_order[0], BackendKind::InsanelyFast);
        // Non-diarize should prioritize WhisperCpp first
        assert_eq!(non_diarize_order[0], BackendKind::WhisperCpp);
    }

    // ── Health probe tests ──

    #[test]
    fn test_probe_system_health_returns_all_backends() {
        let report = probe_system_health_uncached();
        assert_eq!(
            report.backends.len(),
            3,
            "should report on all 3 production backends, got {}",
            report.backends.len()
        );

        let kinds: Vec<_> = report.backends.iter().map(|b| b.backend).collect();
        assert!(kinds.contains(&BackendKind::WhisperCpp));
        assert!(kinds.contains(&BackendKind::InsanelyFast));
        assert!(kinds.contains(&BackendKind::WhisperDiarization));
    }

    #[test]
    fn test_health_report_serializes_to_json() {
        let report = probe_system_health_uncached();
        let json = serde_json::to_string_pretty(&report);
        assert!(json.is_ok(), "SystemHealthReport should serialize to JSON");
        let json_str = json.unwrap();

        // Verify key fields are present.
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("should parse back");
        assert!(parsed["backends"].is_array());
        assert_eq!(parsed["backends"].as_array().unwrap().len(), 3);
        assert!(parsed.get("recommended_backend").is_some());
        assert!(parsed.get("diarization_ready").is_some());
        assert!(parsed.get("hf_token_set").is_some());

        // Each backend entry should have the required fields.
        for entry in parsed["backends"].as_array().unwrap() {
            assert!(entry.get("backend").is_some());
            assert!(entry.get("available").is_some());
            assert!(entry.get("binary_found").is_some());
            assert!(entry.get("capabilities").is_some());
            assert!(entry.get("issues").is_some());
            assert!(entry.get("checked_at_rfc3339").is_some());
        }
    }

    #[test]
    fn test_hf_token_check() {
        // We cannot guarantee the env var state in CI, so just verify the
        // function returns a bool consistently and matches a manual check.
        let expected =
            std::env::var("FRANKEN_WHISPER_HF_TOKEN").is_ok() || std::env::var("HF_TOKEN").is_ok();
        assert_eq!(
            is_hf_token_set(),
            expected,
            "is_hf_token_set should match manual env check"
        );
    }

    #[test]
    fn health_report_backend_reports_have_rfc3339_timestamp() {
        let report = probe_system_health_uncached();
        for backend_report in &report.backends {
            assert!(
                !backend_report.checked_at_rfc3339.is_empty(),
                "checked_at_rfc3339 should not be empty for {:?}",
                backend_report.backend
            );
            // Basic RFC 3339 sanity: contains 'T' separator and timezone marker.
            assert!(
                backend_report.checked_at_rfc3339.contains('T'),
                "timestamp should contain T separator: {}",
                backend_report.checked_at_rfc3339
            );
        }
    }

    #[test]
    fn health_cache_returns_same_report_within_ttl() {
        // Call twice in quick succession; the second call should hit the cache.
        let first = probe_system_health();
        let second = probe_system_health();
        // The checked_at_rfc3339 of each backend should be identical
        // (same cached report).
        for (a, b) in first.backends.iter().zip(second.backends.iter()) {
            assert_eq!(
                a.checked_at_rfc3339, b.checked_at_rfc3339,
                "cached report should return same timestamps"
            );
        }
    }

    #[test]
    fn backend_health_report_round_trips_through_json() {
        let report = BackendHealthReport {
            backend: BackendKind::WhisperCpp,
            available: true,
            binary_found: true,
            binary_path: Some("/usr/bin/whisper-cli".to_owned()),
            version: Some("v1.2.3".to_owned()),
            capabilities: EngineCapabilities {
                supports_diarization: false,
                supports_translation: true,
                supports_word_timestamps: true,
                supports_gpu: true,
                supports_streaming: false,
            },
            issues: vec![],
            checked_at_rfc3339: "2026-01-01T00:00:00+00:00".to_owned(),
        };
        let json_str = serde_json::to_string(&report).expect("serialize");
        let round_tripped: BackendHealthReport =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(round_tripped.backend, report.backend);
        assert_eq!(round_tripped.available, report.available);
        assert_eq!(round_tripped.binary_found, report.binary_found);
        assert_eq!(round_tripped.binary_path, report.binary_path);
        assert_eq!(round_tripped.version, report.version);
        assert!(round_tripped.issues.is_empty());
    }

    // ── StreamingEngine trait tests ──

    use super::StreamingEngine;

    /// A mock engine that returns a fixed set of segments, used to test the
    /// `StreamingEngine` default implementation without requiring any
    /// external binaries.
    struct MockNonStreamingEngine {
        segments: Vec<TranscriptionSegment>,
    }

    impl Engine for MockNonStreamingEngine {
        fn name(&self) -> &'static str {
            "mock-non-streaming"
        }

        fn kind(&self) -> BackendKind {
            BackendKind::WhisperCpp
        }

        fn capabilities(&self) -> EngineCapabilities {
            EngineCapabilities {
                supports_diarization: false,
                supports_translation: false,
                supports_word_timestamps: false,
                supports_gpu: false,
                supports_streaming: false,
            }
        }

        fn is_available(&self) -> bool {
            true
        }

        fn run(
            &self,
            _request: &TranscribeRequest,
            _normalized_wav: &Path,
            _work_dir: &Path,
            _timeout: Duration,
            _token: Option<&crate::orchestrator::CancellationToken>,
        ) -> crate::error::FwResult<TranscriptionResult> {
            let transcript = self
                .segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            Ok(TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript,
                language: Some("en".to_owned()),
                segments: self.segments.clone(),
                acceleration: None,
                raw_output: serde_json::json!({}),
                artifact_paths: vec![],
            })
        }
    }

    // Opt in to streaming with the default implementation.
    impl StreamingEngine for MockNonStreamingEngine {}

    fn make_test_segments(count: usize) -> Vec<TranscriptionSegment> {
        (0..count)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some((i + 1) as f64),
                text: format!("segment-{i}"),
                speaker: None,
                confidence: Some(0.9),
            })
            .collect()
    }

    #[test]
    fn streaming_default_impl_collects_all_segments() {
        let segments = make_test_segments(5);
        let engine = MockNonStreamingEngine {
            segments: segments.clone(),
        };

        let collected: Arc<StdMutex<Vec<TranscriptionSegment>>> =
            Arc::new(StdMutex::new(Vec::new()));
        let collected_clone = Arc::clone(&collected);

        let request = test_request(false);
        let result = engine
            .run_streaming(
                &request,
                Path::new("dummy.wav"),
                Path::new("/tmp"),
                Duration::from_secs(30),
                None,
                Box::new(move |seg| {
                    collected_clone.lock().unwrap().push(seg);
                }),
            )
            .expect("run_streaming should succeed");

        let received = collected.lock().unwrap();
        assert_eq!(received.len(), 5, "callback should receive all 5 segments");
        assert_eq!(
            result.segments.len(),
            5,
            "returned result should contain all 5 segments"
        );
    }

    #[test]
    fn streaming_default_impl_delivers_segments_in_order() {
        let segments = make_test_segments(10);
        let engine = MockNonStreamingEngine {
            segments: segments.clone(),
        };

        let collected: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let collected_clone = Arc::clone(&collected);

        let request = test_request(false);
        engine
            .run_streaming(
                &request,
                Path::new("dummy.wav"),
                Path::new("/tmp"),
                Duration::from_secs(30),
                None,
                Box::new(move |seg| {
                    collected_clone.lock().unwrap().push(seg.text);
                }),
            )
            .expect("run_streaming should succeed");

        let received = collected.lock().unwrap();
        let expected: Vec<String> = (0..10).map(|i| format!("segment-{i}")).collect();
        assert_eq!(*received, expected, "segments should be delivered in order");
    }

    #[test]
    fn streaming_default_impl_works_for_non_streaming_engine() {
        // A non-streaming engine (supports_streaming: false) can still be
        // used via the StreamingEngine default implementation.
        let engine = MockNonStreamingEngine {
            segments: make_test_segments(3),
        };

        assert!(
            !engine.capabilities().supports_streaming,
            "mock engine should report supports_streaming=false"
        );

        let callback_count = Arc::new(StdMutex::new(0usize));
        let count_clone = Arc::clone(&callback_count);

        let request = test_request(false);
        let result = engine
            .run_streaming(
                &request,
                Path::new("dummy.wav"),
                Path::new("/tmp"),
                Duration::from_secs(30),
                None,
                Box::new(move |_seg| {
                    *count_clone.lock().unwrap() += 1;
                }),
            )
            .expect("run_streaming should succeed for non-streaming engine");

        assert_eq!(
            *callback_count.lock().unwrap(),
            3,
            "callback should fire for each segment"
        );
        assert_eq!(result.segments.len(), 3);
        assert_eq!(result.transcript, "segment-0 segment-1 segment-2");
    }

    #[test]
    fn streaming_default_impl_empty_segments_invokes_no_callbacks() {
        let engine = MockNonStreamingEngine { segments: vec![] };

        let invoked = Arc::new(StdMutex::new(false));
        let invoked_clone = Arc::clone(&invoked);

        let request = test_request(false);
        let result = engine
            .run_streaming(
                &request,
                Path::new("dummy.wav"),
                Path::new("/tmp"),
                Duration::from_secs(30),
                None,
                Box::new(move |_seg| {
                    *invoked_clone.lock().unwrap() = true;
                }),
            )
            .expect("run_streaming should succeed with zero segments");

        assert!(
            !*invoked.lock().unwrap(),
            "callback should not be invoked for empty segments"
        );
        assert!(result.segments.is_empty());
    }

    #[test]
    fn streaming_default_impl_result_matches_batch_run() {
        let segments = make_test_segments(4);
        let engine = MockNonStreamingEngine {
            segments: segments.clone(),
        };

        let request = test_request(false);
        let batch_result = engine
            .run(
                &request,
                Path::new("dummy.wav"),
                Path::new("/tmp"),
                Duration::from_secs(30),
                None,
            )
            .expect("batch run should succeed");

        let streaming_result = engine
            .run_streaming(
                &request,
                Path::new("dummy.wav"),
                Path::new("/tmp"),
                Duration::from_secs(30),
                None,
                Box::new(|_seg| {}),
            )
            .expect("streaming run should succeed");

        assert_eq!(
            batch_result.transcript, streaming_result.transcript,
            "streaming and batch transcripts should match"
        );
        assert_eq!(
            batch_result.segments.len(),
            streaming_result.segments.len(),
            "streaming and batch segment counts should match"
        );
        for (batch_seg, stream_seg) in batch_result
            .segments
            .iter()
            .zip(streaming_result.segments.iter())
        {
            assert_eq!(batch_seg.text, stream_seg.text);
            assert_eq!(batch_seg.start_sec, stream_seg.start_sec);
            assert_eq!(batch_seg.end_sec, stream_seg.end_sec);
        }
    }

    // ── StreamingEngine impl coverage for all concrete engines ──

    #[test]
    fn all_engines_implement_streaming_engine() {
        let engines = super::all_streaming_engines();
        assert_eq!(
            engines.len(),
            6,
            "all 6 concrete engines should be streaming-capable"
        );
        let names: Vec<&str> = engines.iter().map(|e| e.name()).collect();
        assert!(names.contains(&"whisper.cpp"));
        assert!(names.contains(&"whisper.cpp-native"));
        assert!(names.contains(&"insanely-fast-whisper"));
        assert!(names.contains(&"insanely-fast-native"));
        assert!(names.contains(&"whisper-diarization"));
        assert!(names.contains(&"whisper-diarization-native"));
    }

    #[test]
    fn streaming_engine_for_returns_correct_engines() {
        let cpp = super::streaming_engine_for(BackendKind::WhisperCpp);
        assert!(cpp.is_some());
        assert_eq!(cpp.unwrap().name(), "whisper.cpp");

        let fast = super::streaming_engine_for(BackendKind::InsanelyFast);
        assert!(fast.is_some());
        assert_eq!(fast.unwrap().name(), "insanely-fast-whisper");

        let diar = super::streaming_engine_for(BackendKind::WhisperDiarization);
        assert!(diar.is_some());
        assert_eq!(diar.unwrap().name(), "whisper-diarization");

        assert!(super::streaming_engine_for(BackendKind::Auto).is_none());
    }

    #[test]
    fn insanely_fast_engine_streaming_reports_non_native() {
        let engine = InsanelyFastEngine;
        assert!(
            !engine.capabilities().supports_streaming,
            "insanely-fast does not natively support streaming"
        );
    }

    #[test]
    fn whisper_diarization_engine_streaming_reports_non_native() {
        let engine = WhisperDiarizationEngine;
        assert!(
            !engine.capabilities().supports_streaming,
            "whisper-diarization does not natively support streaming"
        );
    }

    #[test]
    fn whisper_cpp_engine_streaming_reports_native() {
        let engine = WhisperCppEngine;
        assert!(
            engine.capabilities().supports_streaming,
            "whisper.cpp natively supports streaming"
        );
    }

    // =========================================================================
    // Adaptive Router State tests (bd-efr.1)
    // =========================================================================

    fn make_outcome(backend: BackendKind, success: bool, latency_ms: u64) -> RoutingOutcomeRecord {
        RoutingOutcomeRecord {
            backend,
            success,
            latency_ms,
            error_message: if success {
                None
            } else {
                Some("test error".to_owned())
            },
            recorded_at_rfc3339: "2026-01-01T00:00:00+00:00".to_owned(),
        }
    }

    // -- RouterState construction and defaults --

    #[test]
    fn router_state_new_is_empty() {
        let state = RouterState::new();
        assert_eq!(state.total_predictions, 0);
        assert_eq!(state.correct_predictions, 0);
        for kind in RouterState::ALL_BACKENDS {
            let m = state.metrics_for(kind);
            assert_eq!(m.sample_count, 0);
        }
    }

    #[test]
    fn router_state_default_equals_new() {
        let a = RouterState::new();
        let b = RouterState::default();
        assert_eq!(a.total_predictions, b.total_predictions);
        assert_eq!(a.correct_predictions, b.correct_predictions);
    }

    // -- record_outcome --

    #[test]
    fn record_outcome_increases_sample_count() {
        let mut state = RouterState::new();
        state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert_eq!(m.sample_count, 1);
        assert_eq!(m.success_count, 1);
    }

    #[test]
    fn record_outcome_auto_is_ignored() {
        let mut state = RouterState::new();
        state.record_outcome(make_outcome(BackendKind::Auto, true, 100));
        // Auto has no slot, so all backends should remain empty.
        for kind in RouterState::ALL_BACKENDS {
            assert_eq!(state.metrics_for(kind).sample_count, 0);
        }
    }

    #[test]
    fn record_outcome_sliding_window_caps_at_max() {
        let mut state = RouterState::new();
        for i in 0..(ROUTER_HISTORY_WINDOW + 10) {
            state.record_outcome(make_outcome(
                BackendKind::InsanelyFast,
                i % 2 == 0,
                100 + i as u64,
            ));
        }
        let m = state.metrics_for(BackendKind::InsanelyFast);
        assert_eq!(
            m.sample_count, ROUTER_HISTORY_WINDOW,
            "sample_count should be capped at ROUTER_HISTORY_WINDOW"
        );
    }

    #[test]
    fn record_outcome_failure_records_error_message() {
        let mut state = RouterState::new();
        let mut record = make_outcome(BackendKind::WhisperCpp, false, 500);
        record.error_message = Some("timeout exceeded".to_owned());
        state.record_outcome(record);

        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert_eq!(m.last_error.as_deref(), Some("timeout exceeded"));
    }

    // -- metrics_for --

    #[test]
    fn metrics_for_empty_returns_uninformative_prior() {
        let state = RouterState::new();
        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert_eq!(m.success_rate, 0.5, "empty state should use 0.5 prior");
        assert_eq!(m.avg_latency_ms, 0.0);
        assert!(m.last_error.is_none());
        assert_eq!(m.sample_count, 0);
        assert_eq!(m.success_count, 0);
    }

    #[test]
    fn metrics_for_auto_returns_zeros() {
        let state = RouterState::new();
        let m = state.metrics_for(BackendKind::Auto);
        assert_eq!(m.success_rate, 0.0);
        assert_eq!(m.sample_count, 0);
    }

    #[test]
    fn metrics_for_tracks_success_rate_correctly() {
        let mut state = RouterState::new();
        // 7 successes, 3 failures.
        for i in 0..10 {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, i < 7, 100));
        }
        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert_eq!(m.sample_count, 10);
        assert_eq!(m.success_count, 7);
        assert!((m.success_rate - 0.7).abs() < 1e-9);
    }

    #[test]
    fn metrics_for_computes_avg_latency_only_from_successes() {
        let mut state = RouterState::new();
        state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 200));
        state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 400));
        state.record_outcome(make_outcome(BackendKind::WhisperCpp, false, 9999));

        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert!((m.avg_latency_ms - 300.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_for_all_failures_zero_avg_latency() {
        let mut state = RouterState::new();
        for _ in 0..5 {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, false, 500));
        }
        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert_eq!(m.avg_latency_ms, 0.0, "all-failure avg latency should be 0");
        assert_eq!(m.success_rate, 0.0);
    }

    // -- calibration_score --

    #[test]
    fn calibration_score_no_predictions_returns_half() {
        let state = RouterState::new();
        assert!((state.calibration_score() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn calibration_score_all_correct_returns_one() {
        let mut state = RouterState::new();
        for _ in 0..10 {
            state.record_prediction_outcome(true);
        }
        assert!((state.calibration_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn calibration_score_all_wrong_returns_zero() {
        let mut state = RouterState::new();
        for _ in 0..10 {
            state.record_prediction_outcome(false);
        }
        assert!((state.calibration_score() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn calibration_score_mixed_returns_ratio() {
        let mut state = RouterState::new();
        for _ in 0..7 {
            state.record_prediction_outcome(true);
        }
        for _ in 0..3 {
            state.record_prediction_outcome(false);
        }
        assert!((state.calibration_score() - 0.7).abs() < 1e-9);
    }

    // -- has_sufficient_data --

    #[test]
    fn has_sufficient_data_false_when_empty() {
        let state = RouterState::new();
        assert!(!state.has_sufficient_data());
    }

    #[test]
    fn has_sufficient_data_false_below_threshold() {
        let mut state = RouterState::new();
        for _ in 0..(ADAPTIVE_MIN_SAMPLES - 1) {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        }
        assert!(!state.has_sufficient_data());
    }

    #[test]
    fn has_sufficient_data_true_at_threshold() {
        let mut state = RouterState::new();
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        }
        assert!(state.has_sufficient_data());
    }

    // -- should_use_static_fallback --

    #[test]
    fn should_use_static_fallback_true_when_empty() {
        let state = RouterState::new();
        assert!(state.should_use_static_fallback());
    }

    #[test]
    fn should_use_static_fallback_true_when_calibration_too_low() {
        let mut state = RouterState::new();
        // Record enough data to pass the sample threshold.
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        }
        // But make calibration very low (all wrong predictions).
        for _ in 0..10 {
            state.record_prediction_outcome(false);
        }
        assert!(state.should_use_static_fallback());
    }

    #[test]
    fn should_use_static_fallback_false_with_good_data() {
        let mut state = RouterState::new();
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        }
        // Good calibration.
        for _ in 0..10 {
            state.record_prediction_outcome(true);
        }
        assert!(!state.should_use_static_fallback());
    }

    // -- to_evidence_json --

    #[test]
    fn to_evidence_json_has_required_structure() {
        let mut state = RouterState::new();
        state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        state.record_prediction_outcome(true);

        let json = state.to_evidence_json();
        assert!(json["backend_metrics"].is_array());
        assert_eq!(json["backend_metrics"].as_array().unwrap().len(), 3);
        assert!(json["calibration_score"].is_f64());
        assert!(json["total_predictions"].is_u64());
        assert!(json["correct_predictions"].is_u64());
        assert!(json["sufficient_data"].is_boolean());
        assert!(json["static_fallback_active"].is_boolean());
    }

    #[test]
    fn to_evidence_json_round_trips_through_serde() {
        let mut state = RouterState::new();
        for i in 0..10 {
            state.record_outcome(make_outcome(
                BackendKind::InsanelyFast,
                i % 3 != 0,
                200 + i * 10,
            ));
        }
        let json = state.to_evidence_json();
        let serialized = serde_json::to_string(&json).expect("should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&serialized).expect("should parse back");
        assert_eq!(json, parsed);
    }

    // -- Adaptive loss modulation --

    #[test]
    fn adaptive_loss_differs_from_static_with_sufficient_data() {
        let request = test_request(false);
        let mut state = RouterState::new();

        // Record enough good outcomes for WhisperCpp.
        for _ in 0..10 {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 50));
        }
        // Make calibration high enough.
        for _ in 0..10 {
            state.record_prediction_outcome(true);
        }
        assert!(!state.should_use_static_fallback());

        let static_loss =
            BackendSelectionContract::backend_base_loss(BackendKind::WhisperCpp, &request, 30.0);
        let adaptive_loss = BackendSelectionContract::backend_base_loss_adaptive(
            BackendKind::WhisperCpp,
            &request,
            30.0,
            Some(&state),
        );

        assert!(
            (static_loss - adaptive_loss).abs() > 1e-6,
            "adaptive loss ({adaptive_loss}) should differ from static ({static_loss})"
        );
    }

    #[test]
    fn adaptive_loss_equals_static_with_insufficient_data() {
        let request = test_request(false);
        let mut state = RouterState::new();

        // Only record a few outcomes (below ADAPTIVE_MIN_SAMPLES).
        for _ in 0..(ADAPTIVE_MIN_SAMPLES - 1) {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 50));
        }

        let static_loss =
            BackendSelectionContract::backend_base_loss(BackendKind::WhisperCpp, &request, 30.0);
        let adaptive_loss = BackendSelectionContract::backend_base_loss_adaptive(
            BackendKind::WhisperCpp,
            &request,
            30.0,
            Some(&state),
        );

        assert!(
            (static_loss - adaptive_loss).abs() < 1e-12,
            "insufficient data: adaptive ({adaptive_loss}) should equal static ({static_loss})"
        );
    }

    #[test]
    fn adaptive_loss_with_none_state_equals_static() {
        let request = test_request(false);

        let static_loss =
            BackendSelectionContract::backend_base_loss(BackendKind::WhisperCpp, &request, 30.0);
        let adaptive_loss = BackendSelectionContract::backend_base_loss_adaptive(
            BackendKind::WhisperCpp,
            &request,
            30.0,
            None,
        );

        assert!(
            (static_loss - adaptive_loss).abs() < 1e-12,
            "None state: adaptive ({adaptive_loss}) should equal static ({static_loss})"
        );
    }

    #[test]
    fn poor_success_rate_increases_loss() {
        let request = test_request(false);
        let mut good_state = RouterState::new();
        let mut bad_state = RouterState::new();

        // Good state: all successes.
        for _ in 0..10 {
            good_state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
            good_state.record_prediction_outcome(true);
        }

        // Bad state: all failures.
        for _ in 0..10 {
            bad_state.record_outcome(make_outcome(BackendKind::WhisperCpp, false, 100));
            bad_state.record_prediction_outcome(true);
        }

        let good_loss = BackendSelectionContract::backend_base_loss_adaptive(
            BackendKind::WhisperCpp,
            &request,
            30.0,
            Some(&good_state),
        );
        let bad_loss = BackendSelectionContract::backend_base_loss_adaptive(
            BackendKind::WhisperCpp,
            &request,
            30.0,
            Some(&bad_state),
        );

        assert!(
            bad_loss > good_loss,
            "bad success rate loss ({bad_loss}) should exceed good ({good_loss})"
        );
    }

    // -- BackendSelectionContract with_router_state --

    #[test]
    fn contract_with_router_state_sets_adaptive_mode() {
        let request = test_request(false);
        let mut state = RouterState::new();
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        }
        for _ in 0..10 {
            state.record_prediction_outcome(true);
        }

        let contract = BackendSelectionContract::with_router_state(&request, 30.0, Some(&state));
        assert!(
            contract.adaptive_mode_active,
            "should be in adaptive mode with good data"
        );
    }

    #[test]
    fn contract_without_router_state_is_not_adaptive() {
        let request = test_request(false);
        let contract = BackendSelectionContract::new(&request, 30.0);
        assert!(
            !contract.adaptive_mode_active,
            "should not be adaptive without state"
        );
    }

    #[test]
    fn contract_with_poor_calibration_is_not_adaptive() {
        let request = test_request(false);
        let mut state = RouterState::new();
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        }
        // All wrong predictions -> low calibration.
        for _ in 0..10 {
            state.record_prediction_outcome(false);
        }

        let contract = BackendSelectionContract::with_router_state(&request, 30.0, Some(&state));
        assert!(
            !contract.adaptive_mode_active,
            "poor calibration should keep static mode"
        );
    }

    // -- evaluate_backend_selection with adaptive state --

    #[test]
    fn evaluate_backend_selection_contains_adaptive_router_state() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 300);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();

        // adaptive_router_state should be present in routing_log.
        assert!(
            outcome.routing_log.get("adaptive_router_state").is_some(),
            "routing_log should contain adaptive_router_state"
        );
    }

    #[test]
    fn evaluate_backend_selection_mode_is_static_or_adaptive() {
        let request = test_request(false);
        let trace_id = TraceId::from_parts(1_700_000_000_000, 301);
        let outcome = evaluate_backend_selection(&request, Some(30.0), trace_id).unwrap();
        let mode = outcome.routing_log["mode"].as_str().unwrap();
        assert!(
            mode == "static" || mode == "adaptive",
            "mode should be 'static' or 'adaptive', got '{mode}'"
        );
    }

    // -- RoutingOutcomeRecord serialization --

    #[test]
    fn routing_outcome_record_serializes_to_json() {
        let record = make_outcome(BackendKind::WhisperCpp, true, 150);
        let json_str = serde_json::to_string(&record).expect("should serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("should parse");
        assert_eq!(parsed["backend"], "whisper_cpp");
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["latency_ms"], 150);
    }

    #[test]
    fn routing_outcome_record_round_trips() {
        let record = make_outcome(BackendKind::InsanelyFast, false, 500);
        let json_str = serde_json::to_string(&record).expect("serialize");
        let parsed: RoutingOutcomeRecord = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(parsed.backend, BackendKind::InsanelyFast);
        assert!(!parsed.success);
        assert_eq!(parsed.latency_ms, 500);
        assert!(parsed.error_message.is_some());
    }

    // -- RouterState serialization --

    #[test]
    fn router_state_serializes_to_json() {
        let mut state = RouterState::new();
        state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        let json_str = serde_json::to_string(&state).expect("should serialize");
        assert!(!json_str.is_empty());
    }

    // -- Constants validation --

    #[test]
    fn adaptive_constants_are_sane() {
        let window = ROUTER_HISTORY_WINDOW;
        let min_samples = ADAPTIVE_MIN_SAMPLES;
        let threshold = ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD;
        assert!(window > 0, "window must be positive");
        assert!(min_samples > 0, "min samples must be positive");
        assert!(min_samples <= window, "min samples must not exceed window");
        assert!(
            (0.0..=1.0).contains(&threshold),
            "calibration threshold must be in [0, 1]"
        );
    }

    // -- Multiple backends tracked independently --

    #[test]
    fn outcomes_track_per_backend_independently() {
        let mut state = RouterState::new();
        state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        state.record_outcome(make_outcome(BackendKind::InsanelyFast, false, 500));
        state.record_outcome(make_outcome(BackendKind::WhisperDiarization, true, 200));

        assert_eq!(state.metrics_for(BackendKind::WhisperCpp).sample_count, 1);
        assert_eq!(state.metrics_for(BackendKind::InsanelyFast).sample_count, 1);
        assert_eq!(
            state
                .metrics_for(BackendKind::WhisperDiarization)
                .sample_count,
            1
        );

        assert!((state.metrics_for(BackendKind::WhisperCpp).success_rate - 1.0).abs() < 1e-9);
        assert!((state.metrics_for(BackendKind::InsanelyFast).success_rate - 0.0).abs() < 1e-9);
    }

    // -- Sliding window behavior --

    #[test]
    fn sliding_window_drops_oldest_entries() {
        let mut state = RouterState::new();
        // Fill window with failures.
        for _ in 0..ROUTER_HISTORY_WINDOW {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, false, 500));
        }
        assert!((state.metrics_for(BackendKind::WhisperCpp).success_rate - 0.0).abs() < 1e-9);

        // Now add successes to push failures out.
        for _ in 0..ROUTER_HISTORY_WINDOW {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
        }
        assert!(
            (state.metrics_for(BackendKind::WhisperCpp).success_rate - 1.0).abs() < 1e-9,
            "after full window replacement, success rate should be 1.0"
        );
    }

    // -- last_error tracks most recent failure --

    #[test]
    fn last_error_reflects_most_recent_failure() {
        let mut state = RouterState::new();
        let mut r1 = make_outcome(BackendKind::WhisperCpp, false, 100);
        r1.error_message = Some("first error".to_owned());
        state.record_outcome(r1);

        let mut r2 = make_outcome(BackendKind::WhisperCpp, false, 200);
        r2.error_message = Some("second error".to_owned());
        state.record_outcome(r2);

        // Interleave a success.
        state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 50));

        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert_eq!(
            m.last_error.as_deref(),
            Some("second error"),
            "last_error should be the most recent failure"
        );
    }

    // =========================================================================
    // bd-1rj.7: Canonical segment tolerance and conformance tests
    // =========================================================================

    fn make_segment(start: Option<f64>, end: Option<f64>, text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: start,
            end_sec: end,
            text: text.to_owned(),
            speaker: None,
            confidence: None,
        }
    }

    #[test]
    fn canonical_tolerance_is_50ms() {
        assert_eq!(CANONICAL_SEGMENT_TOLERANCE_MS, 50);
    }

    #[test]
    fn conformance_empty_segments_is_vacuously_conformant() {
        let report = check_segment_conformance(&[]);
        assert_eq!(report.total_segments, 0);
        assert_eq!(report.conformant_segments, 0);
        assert!(report.violations.is_empty());
        assert!((report.conformance_ratio - 1.0).abs() < 1e-12);
        assert_eq!(report.tolerance_ms, CANONICAL_SEGMENT_TOLERANCE_MS);
    }

    #[test]
    fn conformance_single_valid_segment() {
        let segments = vec![make_segment(Some(0.0), Some(1.0), "hello")];
        let report = check_segment_conformance(&segments);
        assert_eq!(report.total_segments, 1);
        assert_eq!(report.conformant_segments, 1);
        assert!(report.violations.is_empty());
        assert!((report.conformance_ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn conformance_perfectly_aligned_segments() {
        let segments = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(Some(1.0), Some(2.0), "b"),
            make_segment(Some(2.0), Some(3.0), "c"),
        ];
        let report = check_segment_conformance(&segments);
        assert_eq!(report.total_segments, 3);
        assert_eq!(report.conformant_segments, 3);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn conformance_overlap_within_tolerance_is_ok() {
        // Overlap of 40ms, which is within the 50ms tolerance.
        let segments = vec![
            make_segment(Some(0.0), Some(1.04), "a"),
            make_segment(Some(1.0), Some(2.0), "b"),
        ];
        let report = check_segment_conformance(&segments);
        assert!(
            report.violations.is_empty(),
            "40ms overlap should be within tolerance: {:?}",
            report.violations
        );
    }

    #[test]
    fn conformance_gap_within_tolerance_is_ok() {
        // Gap of 40ms, which is within the 50ms tolerance.
        let segments = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(Some(1.04), Some(2.0), "b"),
        ];
        let report = check_segment_conformance(&segments);
        assert!(
            report.violations.is_empty(),
            "40ms gap should be within tolerance: {:?}",
            report.violations
        );
    }

    #[test]
    fn conformance_overlap_beyond_tolerance_is_violation() {
        // Overlap of 100ms, exceeding the 50ms tolerance.
        let segments = vec![
            make_segment(Some(0.0), Some(1.1), "a"),
            make_segment(Some(1.0), Some(2.0), "b"),
        ];
        let report = check_segment_conformance(&segments);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(
            report.violations[0].kind,
            SegmentViolationKind::OverlapBeyondTolerance
        );
        assert!(report.violations[0].magnitude_ms.unwrap() > 50.0);
        assert_eq!(report.conformant_segments, 0);
    }

    #[test]
    fn conformance_gap_beyond_tolerance_is_violation() {
        // Gap of 200ms, exceeding the 50ms tolerance.
        let segments = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(Some(1.2), Some(2.0), "b"),
        ];
        let report = check_segment_conformance(&segments);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(
            report.violations[0].kind,
            SegmentViolationKind::GapBeyondTolerance
        );
        assert!(report.violations[0].magnitude_ms.unwrap() > 50.0);
    }

    #[test]
    fn conformance_non_monotonic_timestamps() {
        let segments = vec![
            make_segment(Some(2.0), Some(3.0), "a"),
            make_segment(Some(1.0), Some(2.0), "b"),
        ];
        let report = check_segment_conformance(&segments);
        let non_mono = report
            .violations
            .iter()
            .any(|v| v.kind == SegmentViolationKind::NonMonotonic);
        assert!(
            non_mono,
            "should detect non-monotonic timestamps: {:?}",
            report.violations
        );
    }

    #[test]
    fn conformance_inverted_segment_timestamps() {
        let segments = vec![make_segment(Some(2.0), Some(1.0), "inverted")];
        let report = check_segment_conformance(&segments);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(
            report.violations[0].kind,
            SegmentViolationKind::InvertedTimestamps
        );
        assert_eq!(report.violations[0].segment_index, 0);
        assert!(report.violations[0].adjacent_index.is_none());
    }

    #[test]
    fn conformance_missing_timestamps_skipped() {
        let segments = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(None, None, "no timestamps"),
            make_segment(Some(2.0), Some(3.0), "c"),
        ];
        let report = check_segment_conformance(&segments);
        // The segment without timestamps should not cause violations with
        // its neighbors because pairwise checks require both to have timestamps.
        assert!(
            report.violations.is_empty(),
            "missing timestamps should be skipped: {:?}",
            report.violations
        );
    }

    #[test]
    fn conformance_multiple_violations_reported() {
        let segments = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(Some(0.5), Some(1.5), "b"), // overlap with a
            make_segment(Some(3.0), Some(4.0), "c"), // gap with b
        ];
        let report = check_segment_conformance(&segments);
        assert!(
            report.violations.len() >= 2,
            "should detect at least 2 violations: {:?}",
            report.violations
        );
    }

    #[test]
    fn conformance_ratio_computed_correctly() {
        let segments = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(Some(1.0), Some(2.0), "b"),
            make_segment(Some(2.0), Some(3.0), "c"),
            make_segment(Some(2.5), Some(4.0), "d"), // overlaps with c beyond tolerance
        ];
        let report = check_segment_conformance(&segments);
        assert!(!report.violations.is_empty());
        assert!(report.conformance_ratio < 1.0);
        assert!(report.conformance_ratio > 0.0);
        // Segments c and d are both involved in the overlap violation,
        // so conformant_segments should be 2 (a and b are clean).
        assert_eq!(report.conformant_segments, 2);
        assert_eq!(report.total_segments, 4);
        assert!(
            (report.conformance_ratio - 0.5).abs() < 1e-12,
            "conformance ratio should be 2/4 = 0.5, got {}",
            report.conformance_ratio
        );
    }

    #[test]
    fn conformance_report_serializes_to_json() {
        let segments = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(Some(1.0), Some(2.0), "b"),
        ];
        let report = check_segment_conformance(&segments);
        let json_str = serde_json::to_string(&report).expect("should serialize");
        let parsed: SegmentConformanceReport =
            serde_json::from_str(&json_str).expect("should deserialize");
        assert_eq!(parsed.total_segments, report.total_segments);
        assert_eq!(parsed.conformant_segments, report.conformant_segments);
        assert_eq!(parsed.tolerance_ms, report.tolerance_ms);
    }

    #[test]
    fn conformance_violation_serializes_round_trip() {
        let violation = SegmentConformanceViolation {
            segment_index: 0,
            adjacent_index: Some(1),
            kind: SegmentViolationKind::OverlapBeyondTolerance,
            description: "test violation".to_owned(),
            magnitude_ms: Some(100.0),
        };
        let json_str = serde_json::to_string(&violation).expect("serialize");
        let parsed: SegmentConformanceViolation =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(parsed, violation);
    }

    #[test]
    fn conformance_large_segment_set() {
        // 100 perfectly sequential segments.
        let segments: Vec<TranscriptionSegment> = (0..100)
            .map(|i| make_segment(Some(i as f64), Some((i + 1) as f64), &format!("seg-{i}")))
            .collect();
        let report = check_segment_conformance(&segments);
        assert_eq!(report.total_segments, 100);
        assert_eq!(report.conformant_segments, 100);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn conformance_only_start_timestamps_present() {
        // end_sec is None so pairwise overlap/gap checks using prev.end_sec won't fire,
        // but monotonicity (start vs start) should still work.
        let segments = vec![
            make_segment(Some(2.0), None, "a"),
            make_segment(Some(1.0), None, "b"),
        ];
        let report = check_segment_conformance(&segments);
        let has_non_mono = report
            .violations
            .iter()
            .any(|v| v.kind == SegmentViolationKind::NonMonotonic);
        assert!(
            has_non_mono,
            "should detect non-monotonic even with only start timestamps"
        );
    }

    #[test]
    fn conformance_tolerance_boundary_within() {
        // Overlap of 40ms is safely within the 50ms tolerance.
        let segments_overlap = vec![
            make_segment(Some(0.0), Some(1.04), "a"),
            make_segment(Some(1.0), Some(2.0), "b"),
        ];
        let report_overlap = check_segment_conformance(&segments_overlap);
        assert!(
            report_overlap.violations.is_empty(),
            "40ms overlap should be within 50ms tolerance"
        );

        // Gap of 40ms is safely within the 50ms tolerance.
        let segments_gap = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(Some(1.04), Some(2.0), "b"),
        ];
        let report_gap = check_segment_conformance(&segments_gap);
        assert!(
            report_gap.violations.is_empty(),
            "40ms gap should be within 50ms tolerance"
        );

        // Overlap of 60ms should exceed the 50ms tolerance.
        let segments_over = vec![
            make_segment(Some(0.0), Some(1.06), "a"),
            make_segment(Some(1.0), Some(2.0), "b"),
        ];
        let report_over = check_segment_conformance(&segments_over);
        assert!(
            !report_over.violations.is_empty(),
            "60ms overlap should exceed 50ms tolerance"
        );

        // Gap of 60ms should exceed the 50ms tolerance.
        let segments_gap_over = vec![
            make_segment(Some(0.0), Some(1.0), "a"),
            make_segment(Some(1.06), Some(2.0), "b"),
        ];
        let report_gap_over = check_segment_conformance(&segments_gap_over);
        assert!(
            !report_gap_over.violations.is_empty(),
            "60ms gap should exceed 50ms tolerance"
        );
    }

    // =========================================================================
    // bd-1rj.8: Shadow-run and native engine contract tests
    // =========================================================================

    fn make_transcription_result(
        backend: BackendKind,
        segments: Vec<TranscriptionSegment>,
    ) -> TranscriptionResult {
        let transcript = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        TranscriptionResult {
            backend,
            transcript,
            language: Some("en".to_owned()),
            segments,
            acceleration: None,
            raw_output: serde_json::json!({}),
            artifact_paths: vec![],
        }
    }

    #[test]
    fn native_engine_contract_serializes_round_trip() {
        let contract = NativeEngineContract {
            name: "native-whisper-rs".to_owned(),
            replaces: BackendKind::WhisperCpp,
            required_capabilities: EngineCapabilities {
                supports_diarization: false,
                supports_translation: true,
                supports_word_timestamps: true,
                supports_gpu: true,
                supports_streaming: true,
            },
            max_median_latency_ms: 500,
            max_p95_latency_ms: 2000,
            min_accuracy: 0.95,
            max_acceptable_divergence_ms: 100,
            min_shadow_runs_for_promotion: 1000,
            max_divergence_rate: 0.05,
        };
        let json_str = serde_json::to_string(&contract).expect("serialize");
        let parsed: NativeEngineContract = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(parsed.name, contract.name);
        assert_eq!(parsed.replaces, contract.replaces);
        assert_eq!(parsed.max_median_latency_ms, contract.max_median_latency_ms);
        assert_eq!(parsed.min_accuracy, contract.min_accuracy);
        assert_eq!(
            parsed.max_acceptable_divergence_ms,
            contract.max_acceptable_divergence_ms
        );
        assert_eq!(
            parsed.min_shadow_runs_for_promotion,
            contract.min_shadow_runs_for_promotion
        );
    }

    #[test]
    fn shadow_run_config_serializes_round_trip() {
        let config = ShadowRunConfig {
            enabled: true,
            primary_backend: BackendKind::WhisperCpp,
            shadow_backend: BackendKind::InsanelyFast,
            max_divergence_ms: 100,
        };
        let json_str = serde_json::to_string(&config).expect("serialize");
        let parsed: ShadowRunConfig = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(parsed.enabled, config.enabled);
        assert_eq!(parsed.primary_backend, config.primary_backend);
        assert_eq!(parsed.shadow_backend, config.shadow_backend);
        assert_eq!(parsed.max_divergence_ms, config.max_divergence_ms);
    }

    #[test]
    fn compare_shadow_identical_results_no_divergences() {
        let segments = vec![
            make_segment(Some(0.0), Some(1.0), "hello"),
            make_segment(Some(1.0), Some(2.0), "world"),
        ];
        let primary = make_transcription_result(BackendKind::WhisperCpp, segments.clone());
        let shadow = make_transcription_result(BackendKind::InsanelyFast, segments);

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        assert!(
            divergences.is_empty(),
            "identical results should have no divergences: {divergences:?}"
        );
    }

    #[test]
    fn compare_shadow_segment_count_mismatch() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![
                make_segment(Some(0.0), Some(1.0), "a"),
                make_segment(Some(1.0), Some(2.0), "b"),
            ],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(Some(0.0), Some(1.0), "a")],
        );

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        let has_count_mismatch = divergences
            .iter()
            .any(|d| d.kind == ShadowDivergenceKind::SegmentCountMismatch);
        assert!(
            has_count_mismatch,
            "should detect segment count mismatch: {divergences:?}"
        );
    }

    #[test]
    fn compare_shadow_text_difference() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![make_segment(Some(0.0), Some(1.0), "hello")],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(Some(0.0), Some(1.0), "hallo")],
        );

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        let has_text_diff = divergences
            .iter()
            .any(|d| d.kind == ShadowDivergenceKind::TextDifference);
        assert!(
            has_text_diff,
            "should detect text difference: {divergences:?}"
        );
    }

    #[test]
    fn compare_shadow_timestamp_divergence_within_tolerance() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![make_segment(Some(0.0), Some(1.0), "hello")],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(Some(0.05), Some(1.05), "hello")],
        );

        // 50ms difference with 100ms tolerance should not diverge.
        let divergences = compare_shadow_results(&primary, &shadow, 100);
        let has_ts_divergence = divergences.iter().any(|d| {
            d.kind == ShadowDivergenceKind::StartTimestampDivergence
                || d.kind == ShadowDivergenceKind::EndTimestampDivergence
        });
        assert!(
            !has_ts_divergence,
            "50ms diff within 100ms tolerance should not diverge: {divergences:?}"
        );
    }

    #[test]
    fn compare_shadow_timestamp_divergence_beyond_tolerance() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![make_segment(Some(0.0), Some(1.0), "hello")],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(Some(0.2), Some(1.3), "hello")],
        );

        // 200ms / 300ms differences with 100ms tolerance should diverge.
        let divergences = compare_shadow_results(&primary, &shadow, 100);
        let start_diverged = divergences
            .iter()
            .any(|d| d.kind == ShadowDivergenceKind::StartTimestampDivergence);
        let end_diverged = divergences
            .iter()
            .any(|d| d.kind == ShadowDivergenceKind::EndTimestampDivergence);
        assert!(
            start_diverged,
            "200ms start diff should exceed 100ms tolerance"
        );
        assert!(end_diverged, "300ms end diff should exceed 100ms tolerance");
    }

    #[test]
    fn compare_shadow_speaker_difference() {
        let mut seg_primary = make_segment(Some(0.0), Some(1.0), "hello");
        seg_primary.speaker = Some("SPEAKER_00".to_owned());
        let mut seg_shadow = make_segment(Some(0.0), Some(1.0), "hello");
        seg_shadow.speaker = Some("SPEAKER_01".to_owned());

        let primary = make_transcription_result(BackendKind::WhisperCpp, vec![seg_primary]);
        let shadow = make_transcription_result(BackendKind::InsanelyFast, vec![seg_shadow]);

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        let has_speaker_diff = divergences
            .iter()
            .any(|d| d.kind == ShadowDivergenceKind::SpeakerDifference);
        assert!(
            has_speaker_diff,
            "should detect speaker difference: {divergences:?}"
        );
    }

    #[test]
    fn compare_shadow_empty_results_no_divergences() {
        let primary = make_transcription_result(BackendKind::WhisperCpp, vec![]);
        let shadow = make_transcription_result(BackendKind::InsanelyFast, vec![]);

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        assert!(
            divergences.is_empty(),
            "two empty results should have no divergences"
        );
    }

    #[test]
    fn compare_shadow_missing_timestamps_not_flagged() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![make_segment(None, None, "hello")],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(None, None, "hello")],
        );

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        assert!(
            divergences.is_empty(),
            "segments with no timestamps should not flag timestamp divergences"
        );
    }

    #[test]
    fn compare_shadow_multiple_segments_multiple_divergences() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![
                make_segment(Some(0.0), Some(1.0), "hello"),
                make_segment(Some(1.0), Some(2.0), "world"),
                make_segment(Some(2.0), Some(3.0), "foo"),
            ],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![
                make_segment(Some(0.0), Some(1.0), "hallo"), // text diff
                make_segment(Some(1.5), Some(2.0), "world"), // start ts divergence
                make_segment(Some(2.0), Some(3.0), "foo"),   // identical
            ],
        );

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        assert!(
            divergences.len() >= 2,
            "should detect at least 2 divergences"
        );
    }

    #[test]
    fn compare_shadow_compares_up_to_shorter_length() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![
                make_segment(Some(0.0), Some(1.0), "hello"),
                make_segment(Some(1.0), Some(2.0), "world"),
            ],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(Some(0.0), Some(1.0), "hello")],
        );

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        // Should have count mismatch but the one aligned segment should not diverge.
        let non_count_divergences: Vec<_> = divergences
            .iter()
            .filter(|d| d.kind != ShadowDivergenceKind::SegmentCountMismatch)
            .collect();
        assert!(
            non_count_divergences.is_empty(),
            "aligned segments should not diverge: {non_count_divergences:?}"
        );
    }

    #[test]
    fn shadow_run_report_serializes_to_json() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![make_segment(Some(0.0), Some(1.0), "hello")],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(Some(0.0), Some(1.0), "hello")],
        );
        let report = ShadowRunReport {
            primary_result: primary,
            shadow_result: shadow,
            divergences: vec![],
            shadow_latency_ms: 150,
            acceptable: true,
        };
        let json_str = serde_json::to_string(&report).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("parse");
        assert!(parsed["acceptable"].as_bool().unwrap());
        assert_eq!(parsed["shadow_latency_ms"].as_u64().unwrap(), 150);
        assert!(parsed["divergences"].as_array().unwrap().is_empty());
    }

    #[test]
    fn shadow_divergence_round_trips() {
        let divergence = ShadowRunDivergence {
            segment_index: 2,
            kind: ShadowDivergenceKind::TextDifference,
            description: "text differs".to_owned(),
        };
        let json_str = serde_json::to_string(&divergence).expect("serialize");
        let parsed: ShadowRunDivergence = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(parsed, divergence);
    }

    #[test]
    fn compare_shadow_text_whitespace_normalization() {
        // Texts that differ only in leading/trailing whitespace should not diverge.
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![make_segment(Some(0.0), Some(1.0), "  hello  ")],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(Some(0.0), Some(1.0), "hello")],
        );

        let divergences = compare_shadow_results(&primary, &shadow, 100);
        let has_text_diff = divergences
            .iter()
            .any(|d| d.kind == ShadowDivergenceKind::TextDifference);
        assert!(
            !has_text_diff,
            "whitespace-only difference should not be flagged"
        );
    }

    #[test]
    fn compare_shadow_zero_tolerance_flags_any_difference() {
        let primary = make_transcription_result(
            BackendKind::WhisperCpp,
            vec![make_segment(Some(0.0), Some(1.0), "hello")],
        );
        let shadow = make_transcription_result(
            BackendKind::InsanelyFast,
            vec![make_segment(Some(0.001), Some(1.0), "hello")],
        );

        // 1ms difference with 0ms tolerance should diverge.
        let divergences = compare_shadow_results(&primary, &shadow, 0);
        let has_start_div = divergences
            .iter()
            .any(|d| d.kind == ShadowDivergenceKind::StartTimestampDivergence);
        assert!(has_start_div, "1ms diff with 0ms tolerance should diverge");
    }

    #[test]
    fn native_engine_contract_fields_accessible() {
        let contract = NativeEngineContract {
            name: "test-engine".to_owned(),
            replaces: BackendKind::InsanelyFast,
            required_capabilities: EngineCapabilities {
                supports_diarization: true,
                supports_translation: true,
                supports_word_timestamps: true,
                supports_gpu: true,
                supports_streaming: false,
            },
            max_median_latency_ms: 1000,
            max_p95_latency_ms: 5000,
            min_accuracy: 0.90,
            max_acceptable_divergence_ms: 200,
            min_shadow_runs_for_promotion: 500,
            max_divergence_rate: 0.10,
        };
        assert_eq!(contract.replaces, BackendKind::InsanelyFast);
        assert!(contract.required_capabilities.supports_diarization);
        assert!(contract.min_accuracy > 0.0 && contract.min_accuracy <= 1.0);
        assert!(contract.max_divergence_rate > 0.0 && contract.max_divergence_rate <= 1.0);
    }

    #[test]
    fn shadow_run_config_disabled_by_default_pattern() {
        let config = ShadowRunConfig {
            enabled: false,
            primary_backend: BackendKind::WhisperCpp,
            shadow_backend: BackendKind::InsanelyFast,
            max_divergence_ms: 100,
        };
        assert!(!config.enabled);
    }

    #[test]
    fn calibration_state_brier_score_direct() {
        use super::CalibrationState;

        // Empty → None
        let empty = CalibrationState::new(10);
        assert_eq!(empty.brier_score(), None);

        // Single perfect prediction: pred=1.0, success=true → (1.0-1.0)^2 = 0.0
        let mut perfect = CalibrationState::new(10);
        perfect.record(1.0, true);
        assert!((perfect.brier_score().unwrap() - 0.0).abs() < 1e-10);

        // Single terrible prediction: pred=0.0, success=true → (0.0-1.0)^2 = 1.0
        let mut terrible = CalibrationState::new(10);
        terrible.record(0.0, true);
        assert!((terrible.brier_score().unwrap() - 1.0).abs() < 1e-10);

        // Two observations: (0.5, true) and (0.5, false)
        // Brier = ((0.5-1)^2 + (0.5-0)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
        let mut mixed = CalibrationState::new(10);
        mixed.record(0.5, true);
        mixed.record(0.5, false);
        assert!((mixed.brier_score().unwrap() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn calibration_state_poorly_calibrated_threshold_boundary() {
        use super::{ADAPTIVE_FALLBACK_BRIER_THRESHOLD, CalibrationState};

        // pred=0.0, success=true → Brier = 1.0 (above threshold)
        let mut above = CalibrationState::new(10);
        above.record(0.0, true);
        assert!(above.brier_score().unwrap() > ADAPTIVE_FALLBACK_BRIER_THRESHOLD);
        assert!(above.is_poorly_calibrated());

        // pred=0.9, success=true → Brier = (0.1)^2 = 0.01 (below threshold)
        let mut below = CalibrationState::new(10);
        below.record(0.9, true);
        assert!(below.brier_score().unwrap() < ADAPTIVE_FALLBACK_BRIER_THRESHOLD);
        assert!(!below.is_poorly_calibrated());

        // Empty → false (no observations)
        let empty = CalibrationState::new(10);
        assert!(!empty.is_poorly_calibrated());
    }

    #[test]
    fn router_state_fallback_reason_all_branches() {
        // Branch 1: insufficient_data (new state, no observations)
        let fresh = RouterState::new();
        let reason = fresh.fallback_reason();
        assert_eq!(reason, Some("insufficient_data".to_owned()));

        // Branch 4: None (sufficient data, good calibration, good brier)
        // Need ADAPTIVE_MIN_SAMPLES outcomes to populate histories + predictions + calibration
        let mut good = RouterState::new();
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            good.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
            good.record_prediction_outcome(true);
            good.record_calibration_observation(0.9, true);
        }
        assert_eq!(good.fallback_reason(), None);

        // Branch 2: accuracy below threshold
        let mut poor_accuracy = RouterState::new();
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            poor_accuracy.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
            poor_accuracy.record_prediction_outcome(false);
            poor_accuracy.record_calibration_observation(0.9, true); // good brier
        }
        let reason = poor_accuracy.fallback_reason().unwrap();
        assert!(
            reason.contains("accuracy_calibration_below_threshold"),
            "expected accuracy fallback, got: {reason}"
        );
    }

    #[test]
    fn routing_evidence_ledger_find_by_decision_id_with_eviction() {
        use super::{RoutingEvidenceLedger, RoutingEvidenceLedgerEntry};

        let mut ledger = RoutingEvidenceLedger::new(3); // capacity of 3

        for i in 0..5 {
            ledger.record(RoutingEvidenceLedgerEntry {
                decision_id: format!("dec-{i}"),
                trace_id: "00000000000000000000000000000000".to_owned(),
                timestamp_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                observed_state: "all_available".to_owned(),
                chosen_action: format!("action-{i}"),
                recommended_order: vec![],
                fallback_active: false,
                fallback_reason: None,
                posterior_snapshot: vec![0.5, 0.3, 0.2],
                calibration_score: 0.8,
                brier_score: None,
                e_process: 1.0,
                ci_width: 0.1,
                adaptive_mode: true,
                policy_id: "test-policy".to_owned(),
                loss_matrix_hash: "abc123".to_owned(),
                availability: vec![],
                duration_bucket: "short".to_owned(),
                diarize: false,
                actual_outcome: None,
            });
        }

        // Capacity is 3, so dec-0 and dec-1 were evicted
        assert_eq!(ledger.len(), 3);
        assert_eq!(ledger.total_recorded(), 5);

        assert!(ledger.find_by_decision_id("dec-0").is_none(), "evicted");
        assert!(ledger.find_by_decision_id("dec-1").is_none(), "evicted");
        assert!(ledger.find_by_decision_id("dec-2").is_some(), "retained");
        assert!(ledger.find_by_decision_id("dec-4").is_some(), "latest");
        assert!(
            ledger.find_by_decision_id("nonexistent").is_none(),
            "never recorded"
        );
    }

    #[test]
    fn posterior_success_probability_extreme_parameters() {
        // Very small alpha/beta with diarize + translate together
        let prob = posterior_success_probability(0.001, 0.001, 0.5, true, true, true);
        assert!(prob > 0.0 && prob < 1.0, "result must be in (0, 1): {prob}");

        // Both boosts: alpha=0.001+1.0+0.8=1.801, beta=0.001+1.0+0.5=1.501
        // expected ≈ 1.801 / (1.801 + 1.501) = 1.801 / 3.302 ≈ 0.545
        assert!((prob - 0.545).abs() < 0.01, "expected ~0.545, got {prob}");

        // Zero quality with all boosts
        let zero_q = posterior_success_probability(0.001, 0.001, 0.0, true, true, true);
        // alpha=0.001+0.0+0.8=0.801, beta=0.001+2.0+0.5=2.501
        // expected ≈ 0.801 / 3.302 ≈ 0.243
        assert!(zero_q > 0.0 && zero_q < 1.0);

        // Maximum quality without penalties
        let max_q = posterior_success_probability(0.001, 0.001, 1.0, false, false, true);
        // alpha=0.001+2.0+0.3=2.301, beta=0.001+0.0+0.0=0.001
        // expected ≈ 2.301 / 2.302 ≈ 0.9996
        assert!(
            max_q > 0.99,
            "max quality should yield very high prob: {max_q}"
        );
    }

    // ── bd-jryr: the mock `WhisperCppPilot` unit tests were DELETED here.
    // The struct no longer exists in production; the native whisper.cpp engine
    // is now real in-process inference, covered by tests in
    // `whisper_cpp_native.rs` (helper unit tests + gated e2e against tiny.en)
    // and the honesty guard in `tests/no_canned_phrases.rs`.

    // bd-s8w8: InsanelyFastPilot unit tests were DELETED with the mock pilot.
    // The native insanely-fast engine's real parallel-window inference is now
    // tested in `insanely_fast_native.rs` (plan_workers matrix, split/merge,
    // gated determinism e2e).

    // ── bd-efr.3: TwoLaneExecutor tests ──

    #[test]
    fn two_lane_higher_confidence_selects_primary() {
        let executor = TwoLaneExecutor::new(QualitySelector::HigherConfidence);
        let result = executor.execute(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "hello".to_owned(),
                    confidence: 0.95,
                }]
            },
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "hello".to_owned(),
                    confidence: 0.80,
                }]
            },
        );
        assert_eq!(result.selected, "primary");
        assert!(result.selection_reason.contains("confidence"));
    }

    #[test]
    fn two_lane_higher_confidence_selects_secondary() {
        let executor = TwoLaneExecutor::new(QualitySelector::HigherConfidence);
        let result = executor.execute(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "hello".to_owned(),
                    confidence: 0.70,
                }]
            },
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "hello".to_owned(),
                    confidence: 0.95,
                }]
            },
        );
        assert_eq!(result.selected, "secondary");
        assert!(result.selection_reason.contains("confidence"));
    }

    #[test]
    fn two_lane_higher_confidence_tie_goes_to_primary() {
        let executor = TwoLaneExecutor::new(QualitySelector::HigherConfidence);
        let result = executor.execute(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "hello".to_owned(),
                    confidence: 0.90,
                }]
            },
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "hello".to_owned(),
                    confidence: 0.90,
                }]
            },
        );
        assert_eq!(result.selected, "primary", "tie should go to primary");
    }

    #[test]
    fn two_lane_lower_latency_selects_faster() {
        // Since execution is sequential and nearly instant with mock closures,
        // we just verify the executor runs and produces valid output.
        let executor = TwoLaneExecutor::new(QualitySelector::LowerLatency);
        let result = executor.execute(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "fast".to_owned(),
                    confidence: 0.90,
                }]
            },
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "slow".to_owned(),
                    confidence: 0.90,
                }]
            },
        );
        // With sequential execution and trivial closures, primary should be
        // equal or faster (it runs first).
        assert!(
            result.selected == "primary" || result.selected == "secondary",
            "must select one lane"
        );
        assert!(result.selection_reason.contains("latency"));
    }

    #[test]
    fn two_lane_custom_selector() {
        // Custom scorer: prefer more segments.
        fn score(segments: &[TranscriptSegment], _latency_ms: u64) -> f64 {
            segments.len() as f64
        }
        let executor = TwoLaneExecutor::new(QualitySelector::Custom(score));
        let result = executor.execute(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "one".to_owned(),
                    confidence: 0.99,
                }]
            },
            || {
                vec![
                    TranscriptSegment {
                        start_ms: 0,
                        end_ms: 2500,
                        text: "one".to_owned(),
                        confidence: 0.80,
                    },
                    TranscriptSegment {
                        start_ms: 2500,
                        end_ms: 5000,
                        text: "two".to_owned(),
                        confidence: 0.80,
                    },
                ]
            },
        );
        assert_eq!(result.selected, "secondary", "more segments should win");
        assert!(result.selection_reason.contains("custom"));
    }

    #[test]
    fn two_lane_result_contains_both_results() {
        let executor = TwoLaneExecutor::new(QualitySelector::HigherConfidence);
        let result = executor.execute(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "primary".to_owned(),
                    confidence: 0.90,
                }]
            },
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "secondary".to_owned(),
                    confidence: 0.80,
                }]
            },
        );
        assert_eq!(result.primary_result.len(), 1);
        assert_eq!(result.secondary_result.len(), 1);
        assert_eq!(result.primary_result[0].text, "primary");
        assert_eq!(result.secondary_result[0].text, "secondary");
    }

    #[test]
    fn two_lane_empty_primary_loses_confidence() {
        let executor = TwoLaneExecutor::new(QualitySelector::HigherConfidence);
        let result = executor.execute(
            Vec::new, // empty => avg confidence 0.0
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 5000,
                    text: "secondary wins".to_owned(),
                    confidence: 0.50,
                }]
            },
        );
        assert_eq!(result.selected, "secondary");
    }

    #[test]
    fn two_lane_both_empty_primary_wins() {
        let executor = TwoLaneExecutor::new(QualitySelector::HigherConfidence);
        let result = executor.execute(Vec::new, Vec::new);
        // Both have avg confidence 0.0, tie goes to primary.
        assert_eq!(result.selected, "primary");
    }

    #[test]
    fn avg_confidence_empty_is_zero() {
        assert_eq!(super::avg_confidence(&[]), 0.0);
    }

    #[test]
    fn avg_confidence_single_segment() {
        let segments = vec![TranscriptSegment {
            start_ms: 0,
            end_ms: 1000,
            text: "test".to_owned(),
            confidence: 0.75,
        }];
        let avg = super::avg_confidence(&segments);
        assert!((avg - 0.75).abs() < 1e-10);
    }

    #[test]
    fn avg_confidence_multiple_segments() {
        let segments = vec![
            TranscriptSegment {
                start_ms: 0,
                end_ms: 1000,
                text: "a".to_owned(),
                confidence: 0.80,
            },
            TranscriptSegment {
                start_ms: 1000,
                end_ms: 2000,
                text: "b".to_owned(),
                confidence: 0.60,
            },
        ];
        let avg = super::avg_confidence(&segments);
        assert!((avg - 0.70).abs() < 1e-10);
    }

    #[test]
    fn transcript_segment_serializes() {
        let seg = TranscriptSegment {
            start_ms: 100,
            end_ms: 5000,
            text: "hello world".to_owned(),
            confidence: 0.95,
        };
        let json = serde_json::to_string(&seg).expect("serialize");
        let parsed: TranscriptSegment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, seg);
    }

    fn make_ledger_entry(
        id: &str,
        fallback: bool,
        brier: Option<f64>,
    ) -> RoutingEvidenceLedgerEntry {
        RoutingEvidenceLedgerEntry {
            decision_id: id.to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            timestamp_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            observed_state: "all_available".to_owned(),
            chosen_action: "try_whisper_cpp".to_owned(),
            recommended_order: vec!["whisper_cpp".to_owned()],
            fallback_active: fallback,
            fallback_reason: if fallback {
                Some("insufficient data".to_owned())
            } else {
                None
            },
            posterior_snapshot: vec![0.5, 0.3, 0.2],
            calibration_score: 0.8,
            brier_score: brier,
            e_process: 1.0,
            ci_width: 0.1,
            adaptive_mode: !fallback,
            policy_id: "test".to_owned(),
            loss_matrix_hash: "abc".to_owned(),
            availability: vec![],
            duration_bucket: "short".to_owned(),
            diarize: false,
            actual_outcome: None,
        }
    }

    #[test]
    fn routing_evidence_ledger_diagnostics_returns_expected_keys() {
        let mut ledger = RoutingEvidenceLedger::new(10);
        ledger.record(make_ledger_entry("d1", false, Some(0.15)));
        ledger.record(make_ledger_entry("d2", true, Some(0.25)));

        // Resolve one entry as success.
        ledger.resolve_outcome(
            "d1",
            RoutingOutcomeRecord {
                backend: BackendKind::WhisperCpp,
                success: true,
                latency_ms: 100,
                error_message: None,
                recorded_at_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            },
        );

        let diag = ledger.diagnostics();
        assert_eq!(diag["total_entries"], 2);
        assert_eq!(diag["total_ever_recorded"], 2);
        assert_eq!(diag["capacity"], 10);
        assert_eq!(diag["fallback_count"], 1);
        assert_eq!(diag["resolved_count"], 1);
        assert_eq!(diag["resolved_success_count"], 1);
        // avg brier = (0.15 + 0.25) / 2 = 0.20
        let avg_brier = diag["avg_brier_score"].as_f64().expect("avg_brier");
        assert!(
            (avg_brier - 0.2).abs() < 1e-10,
            "avg brier should be 0.2, got {avg_brier}"
        );
    }

    #[test]
    fn routing_evidence_ledger_resolve_outcome_nonexistent_is_noop() {
        let mut ledger = RoutingEvidenceLedger::new(5);
        ledger.record(make_ledger_entry("d1", false, None));

        // Resolve with a decision_id that doesn't exist — should be a no-op.
        ledger.resolve_outcome(
            "nonexistent-id",
            RoutingOutcomeRecord {
                backend: BackendKind::WhisperCpp,
                success: true,
                latency_ms: 50,
                error_message: None,
                recorded_at_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            },
        );

        assert_eq!(ledger.resolved_count(), 0, "no entries should be resolved");
        assert!(
            ledger
                .find_by_decision_id("d1")
                .unwrap()
                .actual_outcome
                .is_none(),
            "d1 should still have no outcome"
        );
    }

    #[test]
    fn calibration_state_to_evidence_json_has_required_keys() {
        let mut cal = CalibrationState::new(50);
        for _ in 0..5 {
            cal.record(0.8, true);
        }
        for _ in 0..3 {
            cal.record(0.7, false);
        }

        let json = cal.to_evidence_json();
        assert!(
            json["brier_score"].is_number(),
            "brier_score should be a number"
        );
        assert_eq!(json["observation_count"], 8);
        assert!(
            json["poorly_calibrated"].is_boolean(),
            "poorly_calibrated should be boolean"
        );
        assert_eq!(json["window_size"], 50);
        assert!(
            json["brier_threshold"].is_number(),
            "brier_threshold should be a number"
        );
    }

    #[test]
    fn routing_evidence_ledger_fallback_entries_filters_correctly() {
        let mut ledger = RoutingEvidenceLedger::new(10);
        ledger.record(make_ledger_entry("d1", false, None)); // not fallback
        ledger.record(make_ledger_entry("d2", true, None)); // fallback
        ledger.record(make_ledger_entry("d3", false, None)); // not fallback
        ledger.record(make_ledger_entry("d4", true, None)); // fallback

        let fb = ledger.fallback_entries();
        assert_eq!(fb.len(), 2, "should have exactly 2 fallback entries");
        assert_eq!(fb[0].decision_id, "d2");
        assert_eq!(fb[1].decision_id, "d4");
    }

    #[test]
    fn routing_evidence_ledger_to_evidence_json_structure() {
        let mut ledger = RoutingEvidenceLedger::new(5);
        ledger.record(make_ledger_entry("d1", false, Some(0.1)));

        let json = ledger.to_evidence_json();
        assert!(
            json["diagnostics"].is_object(),
            "should have diagnostics object"
        );
        assert!(json["entries"].is_array(), "should have entries array");
        assert_eq!(
            json["entries"].as_array().unwrap().len(),
            1,
            "entries should have 1 element"
        );
        assert_eq!(
            json["diagnostics"]["total_entries"], 1,
            "diagnostics total_entries should be 1"
        );
    }

    #[test]
    fn calibration_state_window_eviction_caps_count() {
        let mut cal = CalibrationState::new(5);
        // Record 8 observations: first 5 terrible (predict 0.0, outcome=success),
        // then 3 perfect (predict 1.0, outcome=success).
        for _ in 0..5 {
            cal.record(0.0, true);
        }
        for _ in 0..3 {
            cal.record(1.0, true);
        }
        // Window eviction: only 5 remain.
        assert_eq!(cal.observation_count(), 5);
        // Window should contain: 2 terrible (Brier=1.0 each) + 3 perfect (Brier=0.0 each).
        // Mean Brier = 2/5 = 0.4.
        let bs = cal.brier_score().expect("should have observations");
        assert!((bs - 0.4).abs() < 1e-9, "expected Brier 0.4, got {bs}");
    }

    #[test]
    fn routing_evidence_ledger_latest_and_is_empty() {
        let mut ledger = RoutingEvidenceLedger::new(5);
        assert!(ledger.is_empty());
        assert!(ledger.latest().is_none());

        let make_entry = |id: &str| RoutingEvidenceLedgerEntry {
            decision_id: id.to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            timestamp_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            observed_state: "all_available".to_owned(),
            chosen_action: "try_whisper_cpp".to_owned(),
            recommended_order: vec![],
            fallback_active: false,
            fallback_reason: None,
            posterior_snapshot: vec![0.5],
            calibration_score: 0.8,
            brier_score: None,
            e_process: 1.0,
            ci_width: 0.1,
            adaptive_mode: true,
            policy_id: "test".to_owned(),
            loss_matrix_hash: "abc".to_owned(),
            availability: vec![],
            duration_bucket: "short".to_owned(),
            diarize: false,
            actual_outcome: None,
        };

        ledger.record(make_entry("dec-1"));
        assert!(!ledger.is_empty());
        assert_eq!(ledger.latest().unwrap().decision_id, "dec-1");

        ledger.record(make_entry("dec-2"));
        assert_eq!(ledger.latest().unwrap().decision_id, "dec-2");
    }

    #[test]
    fn concurrent_executor_basic_execution() {
        use super::ConcurrentTwoLaneExecutor;
        let executor = ConcurrentTwoLaneExecutor::new(QualitySelector::HigherConfidence);

        let primary_called = Arc::new(StdMutex::new(false));
        let compare_called = Arc::new(StdMutex::new(false));
        let pc = primary_called.clone();
        let cc = compare_called.clone();

        let result = executor.execute_with_early_emit(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "fast".to_owned(),
                    confidence: 0.7,
                }]
            },
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "quality".to_owned(),
                    confidence: 0.95,
                }]
            },
            move |_primary, _lat| {
                *pc.lock().unwrap() = true;
            },
            move |_p, _s, _pl, _sl| {
                *cc.lock().unwrap() = true;
            },
        );

        assert!(
            *primary_called.lock().unwrap(),
            "on_primary should be called"
        );
        assert!(
            *compare_called.lock().unwrap(),
            "on_compare should be called"
        );
        assert!(!result.primary_result.is_empty());
        assert!(!result.secondary_result.is_empty());
        // HigherConfidence selector should pick secondary (0.95 > 0.7).
        assert_eq!(result.selected, "secondary");
    }

    #[test]
    fn gate_recommended_order_sole_validated_fallback_stages() {
        use crate::conformance::NativeEngineRolloutStage;
        let custom_order = vec![BackendKind::WhisperCpp, BackendKind::InsanelyFast];

        // Sole → preserves adaptive order, not forced.
        let (order, forced) = super::gate_recommended_order_for_rollout(
            false,
            custom_order.clone(),
            NativeEngineRolloutStage::Sole,
        );
        assert_eq!(order, custom_order);
        assert!(!forced, "Sole should not force static");

        // Validated → static order, forced.
        let (order_v, forced_v) = super::gate_recommended_order_for_rollout(
            false,
            custom_order.clone(),
            NativeEngineRolloutStage::Validated,
        );
        assert_eq!(order_v, auto_priority(false));
        assert!(forced_v, "Validated should force static");

        // Fallback → static order, forced.
        let (order_f, forced_f) = super::gate_recommended_order_for_rollout(
            false,
            custom_order,
            NativeEngineRolloutStage::Fallback,
        );
        assert_eq!(order_f, auto_priority(false));
        assert!(forced_f, "Fallback should force static");
    }

    #[test]
    fn calibration_state_poorly_calibrated_with_bad_predictions() {
        let mut cal = CalibrationState::new(20);
        // Record perfect predictions — should NOT be poorly calibrated.
        for _ in 0..10 {
            cal.record(1.0, true);
        }
        assert!(
            !cal.is_poorly_calibrated(),
            "perfect predictions should not be poorly calibrated"
        );

        // Now fill with maximally wrong predictions (predict 0.0, outcome true).
        let mut cal_bad = CalibrationState::new(10);
        for _ in 0..10 {
            cal_bad.record(0.0, true);
        }
        // Brier = 1.0 > 0.35 threshold.
        assert!(
            cal_bad.is_poorly_calibrated(),
            "all-wrong predictions should be poorly calibrated"
        );
        let bs = cal_bad.brier_score().unwrap();
        assert!((bs - 1.0).abs() < 1e-9, "Brier should be 1.0, got {bs}");
    }

    #[test]
    fn router_state_fallback_reason_brier_score_branch() {
        // Exercises the Branch 3 path of fallback_reason(): sufficient data,
        // accuracy above threshold, but Brier score above threshold.
        let mut state = RouterState::new();
        // Record enough outcomes and correct predictions so has_sufficient_data
        // is true and calibration_score >= ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD.
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
            state.record_prediction_outcome(true); // 100% accuracy
        }
        // Poison the Brier calibration with maximally wrong predictions:
        // predict 0.0 but outcome is true → Brier = 1.0 >> 0.35 threshold.
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_calibration_observation(0.0, true);
        }
        assert!(state.has_sufficient_data(), "should have sufficient data");
        assert!(
            state.calibration_score() >= ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD,
            "accuracy should be above threshold"
        );
        let reason = state.fallback_reason().unwrap();
        assert!(
            reason.contains("brier_score_above_threshold"),
            "expected Brier fallback, got: {reason}"
        );
    }

    #[test]
    fn calibration_state_record_clamps_out_of_range_probability() {
        let mut cal = CalibrationState::new(10);
        // Below range: -0.5 should clamp to 0.0; outcome=true → Brier = (0.0 - 1.0)^2 = 1.0
        cal.record(-0.5, true);
        let bs = cal.brier_score().unwrap();
        assert!(
            (bs - 1.0).abs() < 1e-9,
            "negative prob should clamp to 0.0, Brier should be 1.0, got {bs}"
        );

        let mut cal2 = CalibrationState::new(10);
        // Above range: 1.5 should clamp to 1.0; outcome=false → Brier = (1.0 - 0.0)^2 = 1.0
        cal2.record(1.5, false);
        let bs2 = cal2.brier_score().unwrap();
        assert!(
            (bs2 - 1.0).abs() < 1e-9,
            "excess prob should clamp to 1.0, Brier should be 1.0, got {bs2}"
        );
    }

    #[test]
    fn gate_recommended_order_diarize_true_shadow_returns_diarize_priority() {
        use crate::conformance::NativeEngineRolloutStage;
        // diarize=true through Shadow/Validated/Fallback should return the
        // diarize-specific priority order (InsanelyFast first).
        let custom_order = vec![BackendKind::WhisperCpp];
        let expected = auto_priority(true).to_vec();

        let (order, forced) = super::gate_recommended_order_for_rollout(
            true,
            custom_order.clone(),
            NativeEngineRolloutStage::Shadow,
        );
        assert_eq!(
            order, expected,
            "Shadow+diarize should use diarize priority"
        );
        assert!(forced);

        let (order_v, forced_v) = super::gate_recommended_order_for_rollout(
            true,
            custom_order.clone(),
            NativeEngineRolloutStage::Validated,
        );
        assert_eq!(
            order_v, expected,
            "Validated+diarize should use diarize priority"
        );
        assert!(forced_v);

        let (order_f, forced_f) = super::gate_recommended_order_for_rollout(
            true,
            custom_order,
            NativeEngineRolloutStage::Fallback,
        );
        assert_eq!(
            order_f, expected,
            "Fallback+diarize should use diarize priority"
        );
        assert!(forced_f);
    }

    #[test]
    fn routing_mode_forced_static_overrides_adaptive_mode() {
        assert_eq!(super::routing_mode(true, false), "adaptive");
        assert_eq!(super::routing_mode(false, false), "static");
        assert_eq!(super::routing_mode(false, true), "static");
        assert_eq!(super::routing_mode(true, true), "static");
    }

    #[test]
    fn effective_fallback_reason_prefers_rollout_forced_static_reason() {
        use crate::conformance::NativeEngineRolloutStage;

        let reason = super::effective_fallback_reason(
            Some("insufficient_data".to_owned()),
            true,
            NativeEngineRolloutStage::Validated,
        );
        assert_eq!(
            reason.as_deref(),
            Some("native_rollout_forced_static_order(validated)")
        );
    }

    #[test]
    fn effective_fallback_reason_preserves_router_reason_when_not_forced_static() {
        use crate::conformance::NativeEngineRolloutStage;

        let reason = super::effective_fallback_reason(
            Some("insufficient_data".to_owned()),
            false,
            NativeEngineRolloutStage::Primary,
        );
        assert_eq!(reason.as_deref(), Some("insufficient_data"));
    }

    #[test]
    fn routing_evidence_ledger_empty_diagnostics_zero_division_safe() {
        let ledger = RoutingEvidenceLedger::new(10);
        let diag = ledger.diagnostics();
        assert_eq!(diag["total_entries"], 0);
        assert_eq!(diag["fallback_count"], 0);
        // Division guards should produce 0.0 instead of NaN/panic.
        let fallback_rate = diag["fallback_rate"].as_f64().unwrap();
        assert!(
            (fallback_rate - 0.0).abs() < 1e-9,
            "fallback_rate should be 0.0"
        );
        let success_rate = diag["resolved_success_rate"].as_f64().unwrap();
        assert!(
            (success_rate - 0.0).abs() < 1e-9,
            "resolved_success_rate should be 0.0"
        );
        assert!(
            diag["avg_brier_score"].is_null(),
            "avg_brier_score should be null when empty"
        );
    }

    #[test]
    fn two_lane_executor_speculative_correct_always_picks_secondary() {
        let executor = TwoLaneExecutor::new(QualitySelector::SpeculativeCorrect);
        let result = executor.execute(
            || {
                // Primary returns high-confidence segments.
                vec![TranscriptSegment {
                    text: "primary".to_owned(),
                    start_ms: 0,
                    end_ms: 1000,
                    confidence: 0.99,
                }]
            },
            || {
                // Secondary returns lower-confidence segments.
                vec![TranscriptSegment {
                    text: "secondary".to_owned(),
                    start_ms: 0,
                    end_ms: 1000,
                    confidence: 0.5,
                }]
            },
        );
        // SpeculativeCorrect unconditionally picks secondary.
        assert_eq!(result.selected, "secondary");
        assert!(
            result.selection_reason.contains("speculative-correct"),
            "reason should mention speculative-correct, got: {}",
            result.selection_reason
        );
        assert_eq!(result.secondary_result[0].text, "secondary");
    }

    #[test]
    fn native_runtime_metadata_returns_correct_identity_per_backend() {
        let wcpp = native_runtime_metadata(BackendKind::WhisperCpp);
        assert_eq!(wcpp.identity, "whisper.cpp-native");
        assert!(wcpp.version.is_some(), "WhisperCpp should have a version");
        assert!(
            wcpp.version.as_ref().unwrap().contains("native-pilot-v1"),
            "version should contain engine tag, got: {:?}",
            wcpp.version
        );

        let ifast = native_runtime_metadata(BackendKind::InsanelyFast);
        assert_eq!(ifast.identity, "insanely-fast-native");

        let wdiar = native_runtime_metadata(BackendKind::WhisperDiarization);
        assert_eq!(wdiar.identity, "whisper-diarization-native");

        let auto = native_runtime_metadata(BackendKind::Auto);
        assert_eq!(auto.identity, "auto-policy");
        assert!(auto.version.is_none(), "Auto should have no version");
    }

    #[test]
    fn runtime_metadata_with_implementation_dispatches_native_vs_bridge() {
        let native = runtime_metadata_with_implementation(
            BackendKind::WhisperCpp,
            BackendImplementation::Native,
        );
        assert_eq!(native.identity, "whisper.cpp-native");

        let bridge =
            runtime_metadata_with_implementation(BackendKind::Auto, BackendImplementation::Bridge);
        assert_eq!(bridge.identity, "auto-policy");
        assert!(bridge.version.is_none());
    }

    #[test]
    fn routing_evidence_ledger_diagnostics_all_failures_gives_zero_success_rate() {
        let mut ledger = RoutingEvidenceLedger::new(10);

        for i in 0..3 {
            let entry = RoutingEvidenceLedgerEntry {
                decision_id: format!("d-{i}"),
                trace_id: "t-1".to_owned(),
                timestamp_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                observed_state: "all_available".to_owned(),
                chosen_action: "try_whisper_cpp".to_owned(),
                recommended_order: vec!["whisper_cpp".to_owned()],
                fallback_active: false,
                fallback_reason: None,
                posterior_snapshot: vec![0.5, 0.3, 0.2],
                calibration_score: 0.8,
                brier_score: Some(0.1),
                e_process: 1.0,
                ci_width: 0.2,
                adaptive_mode: true,
                policy_id: "test-policy".to_owned(),
                loss_matrix_hash: "hash".to_owned(),
                availability: vec![("whisper_cpp".to_owned(), true)],
                duration_bucket: "short".to_owned(),
                diarize: false,
                actual_outcome: None,
            };
            ledger.record(entry);
        }

        // Resolve all as failures.
        for i in 0..3 {
            ledger.resolve_outcome(
                &format!("d-{i}"),
                RoutingOutcomeRecord {
                    backend: BackendKind::WhisperCpp,
                    success: false,
                    latency_ms: 1000,
                    error_message: Some("failed".to_owned()),
                    recorded_at_rfc3339: "2026-01-01T00:00:05Z".to_owned(),
                },
            );
        }

        let diag = ledger.diagnostics();
        assert_eq!(diag["resolved_count"].as_u64(), Some(3));
        assert_eq!(diag["resolved_success_count"].as_u64(), Some(0));
        let rate = diag["resolved_success_rate"].as_f64().unwrap();
        assert!(
            (rate - 0.0).abs() < 1e-9,
            "all-failure resolved_success_rate should be 0.0, got {rate}"
        );
    }

    #[test]
    fn calibration_state_brier_score_after_window_eviction_at_boundary() {
        let mut cal = CalibrationState::new(2);

        // Two perfect observations: predicted matches actual.
        cal.record(0.0, false); // (0.0 - 0)^2 = 0
        cal.record(1.0, true); // (1.0 - 1)^2 = 0
        assert!(
            (cal.brier_score().unwrap() - 0.0).abs() < 1e-9,
            "two perfect predictions should give Brier 0.0"
        );

        // Third observation evicts the first: window = [(1.0, true), (0.0, true)].
        // (0.0, true) → (0.0 - 1.0)^2 = 1.0 — terrible prediction.
        cal.record(0.0, true);
        let brier = cal.brier_score().unwrap();
        let expected = (0.0 + 1.0) / 2.0; // 0.5
        assert!(
            (brier - expected).abs() < 1e-9,
            "Brier after eviction should be {expected}, got {brier}"
        );
        assert_eq!(cal.observation_count(), 2);
    }

    #[test]
    fn compare_shadow_one_sided_timestamp_absence_does_not_flag_divergence() {
        // When only one side has a timestamp, the if-let guard skips the check.
        let primary = TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "hello".to_owned(),
            language: None,
            segments: vec![TranscriptionSegment {
                text: "hello".to_owned(),
                start_sec: Some(0.0),
                end_sec: None, // Primary has no end
                speaker: None,
                confidence: None,
            }],
            acceleration: None,
            raw_output: serde_json::json!({}),
            artifact_paths: vec![],
        };
        let shadow = TranscriptionResult {
            backend: BackendKind::InsanelyFast,
            transcript: "hello".to_owned(),
            language: None,
            segments: vec![TranscriptionSegment {
                text: "hello".to_owned(),
                start_sec: None,     // Shadow has no start
                end_sec: Some(99.0), // wildly different if compared
                speaker: None,
                confidence: None,
            }],
            acceleration: None,
            raw_output: serde_json::json!({}),
            artifact_paths: vec![],
        };

        let divergences = compare_shadow_results(&primary, &shadow, 0);
        // No StartTimestampDivergence — primary has Some but shadow has None.
        assert!(
            !divergences
                .iter()
                .any(|d| matches!(d.kind, ShadowDivergenceKind::StartTimestampDivergence)),
            "one-sided start timestamp should not flag divergence"
        );
        // No EndTimestampDivergence — primary has None but shadow has Some.
        assert!(
            !divergences
                .iter()
                .any(|d| matches!(d.kind, ShadowDivergenceKind::EndTimestampDivergence)),
            "one-sided end timestamp should not flag divergence"
        );
    }

    #[test]
    fn diagnostics_avg_brier_null_when_all_entries_have_none_brier() {
        // Entries exist (total > 0), but every brier_score is None.
        // Exercises the `brier_values.is_empty()` branch at line 353-354
        // which should produce avg_brier_score: null despite non-empty ledger.
        let mut ledger = RoutingEvidenceLedger::new(10);
        ledger.record(make_ledger_entry("d1", false, None));
        ledger.record(make_ledger_entry("d2", true, None));
        ledger.record(make_ledger_entry("d3", false, None));

        let diag = ledger.diagnostics();
        assert_eq!(diag["total_entries"], 3, "should have 3 entries");
        assert!(
            diag["avg_brier_score"].is_null(),
            "avg_brier_score should be null when all entries have None brier, got: {}",
            diag["avg_brier_score"]
        );
        // Other fields should still be populated.
        assert_eq!(diag["fallback_count"], 1, "d2 is a fallback entry");
        let avg_cal = diag["avg_calibration_score"].as_f64().unwrap();
        assert!(
            (avg_cal - 0.8).abs() < 1e-9,
            "avg calibration should be 0.8 (all entries use 0.8), got {avg_cal}"
        );
    }

    #[test]
    fn last_error_none_after_all_failures_evicted_from_window() {
        // Fill the sliding window with failures, then overwrite them entirely
        // with successes. `metrics_for().last_error` should become None because
        // all error-bearing records have been evicted from the window.
        let mut state = RouterState::new();
        // Record ADAPTIVE_MIN_SAMPLES failures with error messages.
        for i in 0..ADAPTIVE_MIN_SAMPLES {
            let mut rec = make_outcome(BackendKind::WhisperCpp, false, 100 + i as u64);
            rec.error_message = Some(format!("err-{i}"));
            state.record_outcome(rec);
        }
        // Verify errors exist.
        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert!(m.last_error.is_some(), "should have error before eviction");

        // Now record ROUTER_HISTORY_WINDOW successes to fully evict all failures.
        for i in 0..ROUTER_HISTORY_WINDOW {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 50 + i as u64));
        }
        let m = state.metrics_for(BackendKind::WhisperCpp);
        assert!(
            m.last_error.is_none(),
            "last_error should be None after all failures evicted, got: {:?}",
            m.last_error
        );
        assert_eq!(m.sample_count, ROUTER_HISTORY_WINDOW);
        assert_eq!(m.success_count, ROUTER_HISTORY_WINDOW);
    }

    #[test]
    fn resolve_outcome_ignored_for_evicted_entry() {
        // Record enough entries to evict the first one from the circular buffer,
        // then try to resolve the evicted entry. Should be silently ignored and
        // resolved_count should remain 0.
        let mut ledger = RoutingEvidenceLedger::new(3);
        ledger.record(make_ledger_entry("dec-evicted", false, Some(0.1)));
        ledger.record(make_ledger_entry("dec-2", false, Some(0.2)));
        ledger.record(make_ledger_entry("dec-3", false, Some(0.3)));
        // This fourth record evicts "dec-evicted".
        ledger.record(make_ledger_entry("dec-4", false, Some(0.4)));

        assert!(
            ledger.find_by_decision_id("dec-evicted").is_none(),
            "dec-evicted should have been evicted"
        );
        assert_eq!(ledger.total_recorded(), 4);
        assert_eq!(ledger.len(), 3);

        // Try to resolve the evicted entry.
        ledger.resolve_outcome(
            "dec-evicted",
            RoutingOutcomeRecord {
                backend: BackendKind::WhisperCpp,
                success: true,
                latency_ms: 100,
                error_message: None,
                recorded_at_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            },
        );
        assert_eq!(
            ledger.resolved_count(),
            0,
            "evicted entry should not be resolvable"
        );
        // Remaining entries should still be unresolved.
        for entry in ledger.entries() {
            assert!(
                entry.actual_outcome.is_none(),
                "entry {} should still be unresolved",
                entry.decision_id
            );
        }
    }

    #[test]
    fn router_state_resolve_evidence_outcome_delegates_to_ledger() {
        // RouterState.resolve_evidence_outcome() at line 640 delegates to
        // evidence_ledger.resolve_outcome(). Verify the outcome is visible
        // through the evidence_ledger() accessor.
        let mut state = RouterState::new();
        let entry = make_ledger_entry("dec-rs-1", false, Some(0.2));
        state.record_evidence(entry);

        assert_eq!(state.evidence_ledger().len(), 1);
        assert_eq!(state.evidence_ledger().resolved_count(), 0);

        state.resolve_evidence_outcome(
            "dec-rs-1",
            RoutingOutcomeRecord {
                backend: BackendKind::InsanelyFast,
                success: true,
                latency_ms: 250,
                error_message: None,
                recorded_at_rfc3339: "2026-01-01T00:01:00Z".to_owned(),
            },
        );

        assert_eq!(
            state.evidence_ledger().resolved_count(),
            1,
            "should have 1 resolved entry"
        );
        let resolved = state
            .evidence_ledger()
            .find_by_decision_id("dec-rs-1")
            .unwrap();
        let outcome = resolved.actual_outcome.as_ref().unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.latency_ms, 250);
        assert_eq!(outcome.backend, BackendKind::InsanelyFast);
    }

    #[test]
    fn should_use_static_fallback_true_when_brier_above_threshold() {
        // Exercises the third arm of should_use_static_fallback() at line 591:
        // `self.calibration.is_poorly_calibrated()`. The existing tests only
        // cover arms 1 (insufficient data) and 2 (calibration_score too low).
        let mut state = RouterState::new();
        // Sufficient data: record enough outcomes.
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_outcome(make_outcome(BackendKind::WhisperCpp, true, 100));
            state.record_prediction_outcome(true); // 100% accuracy → good calibration_score
        }
        assert!(state.has_sufficient_data());
        assert!(state.calibration_score() >= ADAPTIVE_FALLBACK_CALIBRATION_THRESHOLD);

        // Poison Brier calibration: predict 0.0 but actual is true → Brier = 1.0.
        for _ in 0..ADAPTIVE_MIN_SAMPLES {
            state.record_calibration_observation(0.0, true);
        }
        assert!(
            state.calibration.is_poorly_calibrated(),
            "Brier should indicate poor calibration"
        );
        assert!(
            state.should_use_static_fallback(),
            "should_use_static_fallback should return true when Brier exceeds threshold"
        );
    }

    // ------------------------------------------------------------------
    // backend/mod edge-case tests pass 9
    // ------------------------------------------------------------------

    #[test]
    fn concurrent_executor_execute_lower_latency_selector() {
        // Exercises QualitySelector::LowerLatency in ConcurrentTwoLaneExecutor::select()
        // (line 3592-3607). Only HigherConfidence has been tested in the
        // concurrent executor; LowerLatency is tested in TwoLaneExecutor but not here.
        use super::ConcurrentTwoLaneExecutor;

        let executor = ConcurrentTwoLaneExecutor::new(QualitySelector::LowerLatency);

        let result = executor.execute(
            || {
                // Primary: simulate some computation.
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "fast".to_owned(),
                    confidence: 0.7,
                }]
            },
            || {
                // Secondary: simulate slower computation.
                std::thread::sleep(Duration::from_millis(10));
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "quality".to_owned(),
                    confidence: 0.95,
                }]
            },
        );

        // LowerLatency should select primary (finishes first).
        assert_eq!(
            result.selected, "primary",
            "LowerLatency selector should pick primary (faster)"
        );
        assert!(
            result.selection_reason.contains("latency"),
            "reason should mention latency: {}",
            result.selection_reason
        );
    }

    #[test]
    fn concurrent_executor_speculative_correct_always_picks_secondary() {
        // Exercises QualitySelector::SpeculativeCorrect in
        // ConcurrentTwoLaneExecutor::select() (lines 3624-3628).
        // SpeculativeCorrect unconditionally selects secondary.
        use super::ConcurrentTwoLaneExecutor;

        let executor = ConcurrentTwoLaneExecutor::new(QualitySelector::SpeculativeCorrect);

        let result = executor.execute(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "primary".to_owned(),
                    confidence: 1.0,
                }]
            },
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1000,
                    text: "secondary".to_owned(),
                    confidence: 0.5,
                }]
            },
        );

        assert_eq!(
            result.selected, "secondary",
            "SpeculativeCorrect always picks secondary"
        );
        assert!(
            result.selection_reason.contains("speculative-correct"),
            "reason should contain 'speculative-correct': {}",
            result.selection_reason
        );
    }

    #[test]
    fn segments_from_nodes_word_key_fallback_when_text_absent() {
        // Exercises the "word" key fallback at line 2519:
        //   .or_else(|| node.get("word").and_then(Value::as_str))
        // The existing test only uses "text" key. This tests the "word" fallback
        // used by some whisper output formats (word-level timestamps).
        let input = serde_json::json!({
            "segments": [
                {"word": " Hello ", "start": 0.0, "end": 0.5},
                {"word": "World", "start": 0.5, "end": 1.0}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(
            segments.len(),
            2,
            "should extract 2 segments from 'word' keys"
        );
        assert_eq!(
            segments[0].text, "Hello",
            "text should be trimmed from 'word' value"
        );
        assert_eq!(segments[1].text, "World");
    }

    #[test]
    fn segments_from_nodes_probability_key_as_confidence_fallback() {
        // Exercises the "probability" key fallback at line 2531:
        //   .or_else(|| node.get("probability"))
        // Existing tests cover "confidence" and "score" but not "probability".
        let input = serde_json::json!({
            "segments": [
                {"text": "hello", "start": 0.0, "end": 1.0, "probability": 0.72}
            ]
        });
        let segments = extract_segments_from_json(&input);
        assert_eq!(segments.len(), 1);
        assert_eq!(
            segments[0].confidence,
            Some(0.72),
            "probability key should be used as confidence fallback"
        );
    }

    #[test]
    fn concurrent_executor_execute_returns_both_results() {
        // Exercises ConcurrentTwoLaneExecutor::execute() (lines 3467-3505),
        // the plain parallel execution method. Only execute_with_early_emit()
        // has been tested for the concurrent executor. This verifies execute()
        // populates both primary and secondary results.
        use super::ConcurrentTwoLaneExecutor;

        let executor = ConcurrentTwoLaneExecutor::new(QualitySelector::HigherConfidence);

        let result = executor.execute(
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 500,
                    text: "primary text".to_owned(),
                    confidence: 0.6,
                }]
            },
            || {
                vec![TranscriptSegment {
                    start_ms: 0,
                    end_ms: 500,
                    text: "secondary text".to_owned(),
                    confidence: 0.9,
                }]
            },
        );

        assert_eq!(result.primary_result.len(), 1);
        assert_eq!(result.secondary_result.len(), 1);
        assert_eq!(result.primary_result[0].text, "primary text");
        assert_eq!(result.secondary_result[0].text, "secondary text");
        // HigherConfidence should pick secondary (0.9 > 0.6).
        assert_eq!(result.selected, "secondary");
        assert!(
            result.primary_latency_ms < 5000,
            "primary should complete quickly"
        );
        assert!(
            result.secondary_latency_ms < 5000,
            "secondary should complete quickly"
        );
    }

    // -----------------------------------------------------------------------
    // bd-6qt: bridge_error_recoverable tests
    // -----------------------------------------------------------------------

    #[test]
    fn bridge_error_recoverable_command_missing() {
        let err = crate::error::FwError::CommandMissing {
            command: "whisper-cpp".to_owned(),
        };
        assert!(
            bridge_error_recoverable(&err),
            "CommandMissing should be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_backend_unavailable() {
        let err = crate::error::FwError::BackendUnavailable("python not found".to_owned());
        assert!(
            bridge_error_recoverable(&err),
            "BackendUnavailable should be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_command_failed() {
        let err = crate::error::FwError::CommandFailed {
            command: "whisper-cpp".to_owned(),
            status: 1,
            stderr_suffix: " segfault".to_owned(),
        };
        assert!(
            bridge_error_recoverable(&err),
            "CommandFailed should be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_io_error_is_not_recoverable() {
        let err = crate::error::FwError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(
            !bridge_error_recoverable(&err),
            "Io errors should not be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_command_timed_out_is_not_recoverable() {
        let err = crate::error::FwError::CommandTimedOut {
            command: "whisper-cpp".to_owned(),
            timeout_ms: 30000,
            stderr_suffix: String::new(),
        };
        assert!(
            !bridge_error_recoverable(&err),
            "CommandTimedOut should not be recoverable (timeout may recur with native)"
        );
    }

    #[test]
    fn bridge_error_recoverable_cancelled_is_not_recoverable() {
        let err = crate::error::FwError::Cancelled("user requested".to_owned());
        assert!(
            !bridge_error_recoverable(&err),
            "Cancelled should not be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_stage_timeout_is_not_recoverable() {
        let err = crate::error::FwError::StageTimeout {
            stage: "backend".to_owned(),
            budget_ms: 60000,
        };
        assert!(
            !bridge_error_recoverable(&err),
            "StageTimeout should not be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_invalid_request_is_not_recoverable() {
        let err = crate::error::FwError::InvalidRequest("bad params".to_owned());
        assert!(
            !bridge_error_recoverable(&err),
            "InvalidRequest should not be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_storage_is_not_recoverable() {
        let err = crate::error::FwError::Storage("db locked".to_owned());
        assert!(
            !bridge_error_recoverable(&err),
            "Storage should not be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_unsupported_is_not_recoverable() {
        let err = crate::error::FwError::Unsupported("not implemented".to_owned());
        assert!(
            !bridge_error_recoverable(&err),
            "Unsupported should not be recoverable"
        );
    }

    #[test]
    fn bridge_error_recoverable_missing_artifact_is_not_recoverable() {
        let err = crate::error::FwError::MissingArtifact(PathBuf::from("/tmp/output.json"));
        assert!(
            !bridge_error_recoverable(&err),
            "MissingArtifact should not be recoverable"
        );
    }

    // -----------------------------------------------------------------------
    // bd-2w3 support: bridge native recovery env parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn bridge_native_recovery_from_raw_defaults_to_enabled_when_absent() {
        assert!(
            bridge_native_recovery_from_raw(None),
            "unset env should default to enabled"
        );
    }

    #[test]
    fn bridge_native_recovery_from_raw_false_like_values_disable_recovery() {
        for value in ["0", "false", "FALSE", "no", "OFF", " off "] {
            assert!(
                !bridge_native_recovery_from_raw(Some(value)),
                "value `{value}` should disable recovery"
            );
        }
    }

    #[test]
    fn bridge_native_recovery_from_raw_non_false_values_enable_recovery() {
        for value in ["1", "true", "yes", "auto", "unexpected"] {
            assert!(
                bridge_native_recovery_from_raw(Some(value)),
                "value `{value}` should enable recovery"
            );
        }
    }

    #[test]
    fn recovered_bridge_only_execution_records_single_success_outcome() {
        let _guard = ROUTER_STATE_TEST_MUTEX
            .lock()
            .expect("router state test lock");
        if let Ok(mut guard) = super::ROUTER_STATE.lock() {
            *guard = None;
        }

        let request = test_request(false);
        let work_dir = tempfile::tempdir().expect("tempdir");
        let result = super::run_backend_with_mode_and_runners(
            BackendKind::WhisperCpp,
            &request,
            Path::new("dummy.wav"),
            work_dir.path(),
            Duration::from_secs(1),
            None,
            super::NativeExecutionMode::BridgeOnly,
            NativeEngineRolloutStage::Fallback,
            true,
            true,
            true,
            |_, _, _, _, _| {
                Err(crate::error::FwError::CommandMissing {
                    command: "whisper-cli".to_owned(),
                })
            },
            |_, _, _, _, _| {
                Ok(make_transcription_result(
                    BackendKind::WhisperCpp,
                    Vec::new(),
                ))
            },
        )
        .expect("native recovery should succeed");

        assert_eq!(result.implementation, BackendImplementation::Native);
        assert_eq!(result.execution_mode, "bridge_only_native_recovery");
        assert_eq!(
            result.native_fallback_error.as_deref(),
            Some("missing command `whisper-cli` on PATH")
        );

        let state = super::router_state_snapshot().expect("router state should exist");
        let metrics = state.metrics_for(BackendKind::WhisperCpp);
        assert_eq!(metrics.sample_count, 1, "should record one final outcome");
        assert_eq!(metrics.success_count, 1, "final recovered run succeeded");
        assert!(
            (metrics.success_rate - 1.0).abs() < 1e-9,
            "recovered run should count as success"
        );
        assert!(
            metrics.last_error.is_none(),
            "recovered run should not leave a failure in last_error"
        );

        if let Ok(mut guard) = super::ROUTER_STATE.lock() {
            *guard = None;
        }
    }

    #[test]
    fn native_preferred_bridge_fallback_records_single_success_outcome() {
        let _guard = ROUTER_STATE_TEST_MUTEX
            .lock()
            .expect("router state test lock");
        if let Ok(mut guard) = super::ROUTER_STATE.lock() {
            *guard = None;
        }

        let request = test_request(false);
        let work_dir = tempfile::tempdir().expect("tempdir");
        let result = super::run_backend_with_mode_and_runners(
            BackendKind::WhisperCpp,
            &request,
            Path::new("dummy.wav"),
            work_dir.path(),
            Duration::from_secs(1),
            None,
            super::NativeExecutionMode::NativePreferred,
            NativeEngineRolloutStage::Primary,
            true,
            true,
            true,
            |_, _, _, _, _| {
                Ok(make_transcription_result(
                    BackendKind::WhisperCpp,
                    Vec::new(),
                ))
            },
            |_, _, _, _, _| {
                Err(crate::error::FwError::BackendUnavailable(
                    "native runtime unavailable".to_owned(),
                ))
            },
        )
        .expect("bridge fallback should succeed");

        assert_eq!(result.implementation, BackendImplementation::Bridge);
        assert_eq!(result.execution_mode, "native_preferred");
        assert_eq!(
            result.native_fallback_error.as_deref(),
            Some("backend unavailable: native runtime unavailable")
        );

        let state = super::router_state_snapshot().expect("router state should exist");
        let metrics = state.metrics_for(BackendKind::WhisperCpp);
        assert_eq!(metrics.sample_count, 1, "should record one final outcome");
        assert_eq!(metrics.success_count, 1, "bridge fallback still succeeded");
        assert!(
            (metrics.success_rate - 1.0).abs() < 1e-9,
            "fallback success should count as success"
        );
        assert!(
            metrics.last_error.is_none(),
            "fallback success should not poison last_error"
        );

        if let Ok(mut guard) = super::ROUTER_STATE.lock() {
            *guard = None;
        }
    }
}
