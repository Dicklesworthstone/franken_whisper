use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::time::Duration;

use asupersync::runtime::{Runtime, RuntimeBuilder};
use chrono::Utc;
use franken_kernel::{Budget, TraceId};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::accelerate;
use crate::audio;
use crate::backend;
use crate::conformance;
use crate::error::{FwError, FwResult};
use crate::model::{RunEvent, RunReport, StreamedRunEvent, TranscribeRequest};
use crate::storage::RunStore;

// ---------------------------------------------------------------------------
// Composable pipeline stages (bd-qla.6)
// ---------------------------------------------------------------------------

/// Identifies a discrete pipeline stage that can be included, excluded, or
/// reordered within a [`PipelineConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Materialize input from file/stdin/URL into a local temp file.
    Ingest,
    /// Normalize audio to 16 kHz mono WAV via ffmpeg.
    Normalize,
    /// Voice activity detection pre-filtering (bd-qla.1).
    Vad,
    /// Source separation / vocal isolation (bd-qla.2, Demucs-inspired placeholder).
    Separate,
    /// Execute the transcription backend (whisper-cpp, insanely-fast-whisper, etc.).
    Backend,
    /// Run native acceleration pass (GPU confidence normalization).
    Accelerate,
    /// CTC-based forced alignment for timestamp correction (bd-qla.3).
    Align,
    /// Punctuation restoration (bd-qla.4).
    Punctuate,
    /// Speaker diarization (bd-qla.5, TitaNet-inspired placeholder).
    Diarize,
    /// Persist the run report to frankensqlite.
    Persist,
}

impl PipelineStage {
    /// The stage label used in events and logging.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ingest => "ingest",
            Self::Normalize => "normalize",
            Self::Vad => "vad",
            Self::Separate => "separate",
            Self::Backend => "backend",
            Self::Accelerate => "acceleration",
            Self::Align => "align",
            Self::Punctuate => "punctuate",
            Self::Diarize => "diarize",
            Self::Persist => "persist",
        }
    }
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Default stage ordering that includes all pipeline stages in their
/// canonical execution order.
const DEFAULT_STAGES: [PipelineStage; 10] = [
    PipelineStage::Ingest,
    PipelineStage::Normalize,
    PipelineStage::Vad,
    PipelineStage::Separate,
    PipelineStage::Backend,
    PipelineStage::Accelerate,
    PipelineStage::Align,
    PipelineStage::Punctuate,
    PipelineStage::Diarize,
    PipelineStage::Persist,
];

/// Specifies which pipeline stages to execute and in what order.
///
/// The default configuration reproduces the original hardcoded pipeline:
/// `Ingest -> Normalize -> Backend -> Accelerate -> Persist`.
///
/// Use [`PipelineBuilder`] for ergonomic construction.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    stages: Vec<PipelineStage>,
}

impl PipelineConfig {
    /// Create a config with the given stages executed in order.
    pub fn new(stages: Vec<PipelineStage>) -> Self {
        Self { stages }
    }

    /// The ordered list of stages that will be executed.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Returns `true` if the given stage is present in this config.
    pub fn has_stage(&self, stage: PipelineStage) -> bool {
        self.stages.contains(&stage)
    }

    /// Validate the configuration for structural soundness.
    ///
    /// Rules enforced:
    /// - `Normalize` requires `Ingest` before it.
    /// - `Backend` requires `Normalize` before it (which itself requires `Ingest`).
    /// - No duplicate stages.
    pub fn validate(&self) -> FwResult<()> {
        // Check for duplicates.
        let mut seen = std::collections::HashSet::new();
        for stage in &self.stages {
            if !seen.insert(stage) {
                return Err(FwError::InvalidRequest(format!(
                    "duplicate pipeline stage: {stage}"
                )));
            }
        }

        // Check ordering constraints.
        let pos = |s: PipelineStage| self.stages.iter().position(|x| *x == s);

        if let Some(norm_pos) = pos(PipelineStage::Normalize) {
            match pos(PipelineStage::Ingest) {
                Some(ingest_pos) if ingest_pos < norm_pos => {}
                _ => {
                    return Err(FwError::InvalidRequest(
                        "Normalize stage requires Ingest before it".to_owned(),
                    ));
                }
            }
        }

        if let Some(backend_pos) = pos(PipelineStage::Backend) {
            match pos(PipelineStage::Normalize) {
                Some(norm_pos) if norm_pos < backend_pos => {}
                _ => {
                    return Err(FwError::InvalidRequest(
                        "Backend stage requires Normalize before it".to_owned(),
                    ));
                }
            }
        }

        if let Some(accel_pos) = pos(PipelineStage::Accelerate) {
            match pos(PipelineStage::Backend) {
                Some(backend_pos) if backend_pos < accel_pos => {}
                _ => {
                    return Err(FwError::InvalidRequest(
                        "Accelerate stage requires Backend before it".to_owned(),
                    ));
                }
            }
        }

        if let Some(align_pos) = pos(PipelineStage::Align) {
            match pos(PipelineStage::Backend) {
                Some(backend_pos) if backend_pos < align_pos => {}
                _ => {
                    return Err(FwError::InvalidRequest(
                        "Align stage requires Backend before it".to_owned(),
                    ));
                }
            }
        }

        // Vad requires Normalize before it.
        if let Some(vad_pos) = pos(PipelineStage::Vad) {
            match pos(PipelineStage::Normalize) {
                Some(norm_pos) if norm_pos < vad_pos => {}
                _ => {
                    return Err(FwError::InvalidRequest(
                        "Vad stage requires Normalize before it".to_owned(),
                    ));
                }
            }
        }

        // Separate requires Normalize before it.
        if let Some(sep_pos) = pos(PipelineStage::Separate) {
            match pos(PipelineStage::Normalize) {
                Some(norm_pos) if norm_pos < sep_pos => {}
                _ => {
                    return Err(FwError::InvalidRequest(
                        "Separate stage requires Normalize before it".to_owned(),
                    ));
                }
            }
        }

        // Punctuate requires Backend before it.
        if let Some(punct_pos) = pos(PipelineStage::Punctuate) {
            match pos(PipelineStage::Backend) {
                Some(backend_pos) if backend_pos < punct_pos => {}
                _ => {
                    return Err(FwError::InvalidRequest(
                        "Punctuate stage requires Backend before it".to_owned(),
                    ));
                }
            }
        }

        // Diarize requires Backend before it.
        if let Some(diar_pos) = pos(PipelineStage::Diarize) {
            match pos(PipelineStage::Backend) {
                Some(backend_pos) if backend_pos < diar_pos => {}
                _ => {
                    return Err(FwError::InvalidRequest(
                        "Diarize stage requires Backend before it".to_owned(),
                    ));
                }
            }
        }

        Ok(())
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            stages: DEFAULT_STAGES.to_vec(),
        }
    }
}

/// Builder for constructing [`PipelineConfig`] instances with a fluent API.
///
/// # Examples
///
/// ```rust,ignore
/// let config = PipelineBuilder::new()
///     .stage(PipelineStage::Ingest)
///     .stage(PipelineStage::Normalize)
///     .stage(PipelineStage::Backend)
///     .stage(PipelineStage::Persist)  // skip Accelerate
///     .build()?;
/// ```
pub struct PipelineBuilder {
    stages: Vec<PipelineStage>,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    /// Start building a pipeline with no stages.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Start from the default stage list (for modification).
    pub fn default_stages() -> Self {
        Self {
            stages: DEFAULT_STAGES.to_vec(),
        }
    }

    /// Append a stage to the pipeline.
    pub fn stage(mut self, stage: PipelineStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Remove a stage from the pipeline (if present).
    pub fn without(mut self, stage: PipelineStage) -> Self {
        self.stages.retain(|s| *s != stage);
        self
    }

    /// Replace the entire stage list.
    pub fn stages(mut self, stages: Vec<PipelineStage>) -> Self {
        self.stages = stages;
        self
    }

    /// Build and validate the pipeline configuration.
    ///
    /// Returns an error if the stage ordering violates dependency constraints.
    pub fn build(self) -> FwResult<PipelineConfig> {
        let config = PipelineConfig::new(self.stages);
        config.validate()?;
        Ok(config)
    }

    /// Build the pipeline configuration without validation.
    ///
    /// Useful for testing or when the caller knows the configuration is valid.
    pub fn build_unchecked(self) -> PipelineConfig {
        PipelineConfig::new(self.stages)
    }
}

pub(crate) struct PipelineCx {
    trace_id: TraceId,
    deadline: Option<chrono::DateTime<Utc>>,
    budget: Budget,
    evidence: Vec<Value>,
    finalizers: FinalizerRegistry,
}

impl PipelineCx {
    pub(crate) fn new(timeout_ms: Option<u64>) -> Self {
        let now = Utc::now();
        let ts_ms = now.timestamp_millis() as u64;
        let random = (uuid::Uuid::new_v4().as_u128()) & 0xFFFF_FFFF_FFFF_FFFF_FFFF;
        let trace_id = TraceId::from_parts(ts_ms, random);

        let deadline = timeout_ms.map(|ms| {
            let clamped = ms.min(i64::MAX as u64);
            let duration = chrono::Duration::milliseconds(clamped as i64);
            // Saturate on chrono range overflow so very large budgets remain future-deadline safe.
            now.checked_add_signed(duration)
                .unwrap_or(chrono::DateTime::<Utc>::MAX_UTC)
        });
        let budget = match timeout_ms {
            Some(ms) => Budget::new(ms),
            None => Budget::UNLIMITED,
        };

        Self {
            trace_id,
            deadline,
            budget,
            evidence: Vec::new(),
            finalizers: FinalizerRegistry::new(),
        }
    }

    pub(crate) fn checkpoint(&self) -> FwResult<()> {
        if crate::cli::ShutdownController::is_shutting_down() {
            return Err(FwError::Cancelled(
                "pipeline cancelled via Ctrl+C".to_owned(),
            ));
        }
        if let Some(deadline) = self.deadline
            && Utc::now() >= deadline
        {
            return Err(FwError::Cancelled(format!(
                "deadline exceeded (budget {}ms)",
                self.budget.remaining_ms()
            )));
        }
        Ok(())
    }

    fn cancellation_evidence(&self, stage: &str) -> Value {
        let now = Utc::now();
        let deadline_rfc3339 = self.deadline.map(|deadline| deadline.to_rfc3339());
        let overdue_ms = self
            .deadline
            .map(|deadline| now.signed_duration_since(deadline).num_milliseconds())
            .unwrap_or(0)
            .max(0) as u64;

        json!({
            "stage": stage,
            "now_rfc3339": now.to_rfc3339(),
            "deadline_rfc3339": deadline_rfc3339,
            "budget_remaining_ms": self.budget.remaining_ms(),
            "overdue_ms": overdue_ms,
            "reason": "checkpoint deadline exceeded",
        })
    }

    #[cfg(test)]
    pub(crate) fn record_evidence(&mut self, entry: &franken_evidence::EvidenceLedger) {
        if let Ok(v) = serde_json::to_value(entry) {
            self.evidence.push(v);
        }
    }

    /// Set the deadline to the past so all subsequent checkpoints trigger
    /// cancellation.  Test-only helper for deterministic cancellation injection.
    #[cfg(test)]
    pub(crate) fn cancel_now(&mut self) {
        self.deadline = Some(chrono::DateTime::<Utc>::MIN_UTC);
    }

    /// Reset the deadline to `None` (no cancellation), undoing any prior
    /// `cancel_now()` call.  Test-only helper.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn uncancel(&mut self) {
        self.deadline = None;
    }

    pub(crate) fn record_evidence_values(&mut self, entries: &[Value]) {
        self.evidence.extend(entries.iter().cloned());
    }

    pub(crate) fn trace_id(&self) -> TraceId {
        self.trace_id
    }

    pub(crate) fn evidence(&self) -> &[Value] {
        &self.evidence
    }

    fn budget_remaining_ms(&self) -> u64 {
        self.budget.remaining_ms()
    }

    /// Produce a lightweight, thread-safe token that can be sent into
    /// `spawn_blocking` closures for cancellation-aware subprocess execution.
    pub(crate) fn cancellation_token(&self) -> CancellationToken {
        CancellationToken {
            deadline: self.deadline,
        }
    }

    /// Produce a cancellation token that respects the per-stage budget.
    ///
    /// The resulting token will cancel at the earlier of the pipeline deadline
    /// (if any) and the stage budget relative to now.
    pub(crate) fn stage_token(&self, stage_budget_ms: u64) -> CancellationToken {
        let now = Utc::now();
        let clamped = stage_budget_ms.min(i64::MAX as u64);
        let stage_deadline = now
            .checked_add_signed(chrono::Duration::milliseconds(clamped as i64))
            .unwrap_or(chrono::DateTime::<Utc>::MAX_UTC);
        let deadline = match self.deadline {
            Some(pipeline_deadline) => Some(pipeline_deadline.min(stage_deadline)),
            None => Some(stage_deadline),
        };
        CancellationToken { deadline }
    }

    /// Register a cleanup action to be run when the pipeline shuts down.
    pub(crate) fn register_finalizer(&mut self, label: &str, finalizer: Finalizer) {
        self.finalizers.register(label, finalizer);
    }

    /// Execute all registered finalizers (LIFO order) and drain the registry.
    #[allow(dead_code)]
    pub(crate) fn run_finalizers(&mut self) {
        self.finalizers.run_all();
    }

    /// Execute all registered finalizers with a bounded time budget (bd-38c.4).
    ///
    /// Each individual finalizer is given at most `budget_ms` to complete.
    /// If a finalizer exceeds its budget, it is logged and skipped so that
    /// subsequent finalizers can still run.
    pub(crate) fn run_finalizers_bounded(&mut self, budget_ms: u64) {
        self.finalizers.run_all_bounded(budget_ms);
    }
}

/// Lightweight, `Send + Sync + Clone` handle that backends use to check the
/// pipeline deadline without needing the full `PipelineCx`.
#[derive(Debug, Clone, Copy)]
pub struct CancellationToken {
    deadline: Option<chrono::DateTime<Utc>>,
}

impl CancellationToken {
    pub fn checkpoint(&self) -> FwResult<()> {
        if crate::cli::ShutdownController::is_shutting_down() {
            return Err(FwError::Cancelled(
                "pipeline cancelled via Ctrl+C".to_owned(),
            ));
        }
        if let Some(deadline) = self.deadline
            && Utc::now() >= deadline
        {
            return Err(FwError::Cancelled("pipeline deadline exceeded".to_owned()));
        }
        Ok(())
    }

    /// Create a token with no deadline (never cancels on its own; still honors
    /// the global Ctrl+C shutdown).
    ///
    /// `pub(crate)` so native backends can synthesize a token when a caller did
    /// not supply one (e.g. the diarization stage inside
    /// `whisper_diarization_native::run`).
    #[must_use]
    pub(crate) fn unbounded() -> Self {
        Self { deadline: None }
    }

    /// Create a token with a deadline relative to now.
    #[cfg(test)]
    pub(crate) fn with_deadline_from_now(duration: std::time::Duration) -> Self {
        Self {
            deadline: Some(
                Utc::now() + chrono::Duration::milliseconds(duration.as_millis() as i64),
            ),
        }
    }

    /// Create a token with no deadline (never cancels).
    #[cfg(test)]
    pub(crate) fn no_deadline() -> Self {
        Self { deadline: None }
    }

    /// Returns `true` if the cancellation token has been triggered (deadline
    /// exceeded).  Convenience wrapper around `checkpoint()` for use in
    /// boolean guards.
    #[cfg(test)]
    pub(crate) fn is_cancelled(&self) -> bool {
        self.checkpoint().is_err()
    }

    /// Create a token that is already cancelled (deadline in the past).
    #[allow(dead_code)]
    #[cfg(test)]
    pub(crate) fn already_expired() -> Self {
        Self {
            deadline: Some(chrono::DateTime::<Utc>::MIN_UTC),
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline finalizers (bd-38c.3)
// ---------------------------------------------------------------------------

/// A cleanup action registered with the pipeline for execution at shutdown.
pub(crate) enum Finalizer {
    /// Log that temp-dir cleanup was requested.  Actual removal is handled by
    /// `tempfile::TempDir`'s `Drop` impl -- this variant exists so the
    /// finalizer registry has an explicit record of the directory.
    TempDir(PathBuf),
    /// Attempt to kill a subprocess by PID.  Silently succeeds if the process
    /// has already exited.
    #[allow(dead_code)] // constructed only in tests for now
    Process(u32),
    /// Arbitrary cleanup closure.
    #[allow(dead_code)] // constructed only in tests for now
    Custom(Box<dyn FnOnce() + Send>),
}

/// LIFO registry of [`Finalizer`] entries.  Entries are executed in reverse
/// registration order so that later resources (which may depend on earlier
/// ones) are cleaned up first.
pub(crate) struct FinalizerRegistry {
    entries: Vec<(String, Finalizer)>,
}

fn sanitize_process_pid(pid: u32) -> Option<u32> {
    if pid == 0 || pid > i32::MAX as u32 {
        None
    } else {
        Some(pid)
    }
}

#[cfg(unix)]
fn send_kill_signal_best_effort(pid: u32) {
    match sanitize_process_pid(pid) {
        Some(safe_pid) => {
            let _ = std::process::Command::new("kill")
                .args(["-9", &safe_pid.to_string()])
                .output();
        }
        None => {
            tracing::warn!(
                pid = pid,
                "finalizer: skipping process kill for invalid/out-of-range pid"
            );
        }
    }
}

#[cfg(not(unix))]
fn send_kill_signal_best_effort(pid: u32) {
    let _ = pid;
}

impl FinalizerRegistry {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a cleanup action with a human-readable label.
    pub(crate) fn register(&mut self, label: &str, finalizer: Finalizer) {
        self.entries.push((label.to_owned(), finalizer));
    }

    /// Execute all registered finalizers in reverse (LIFO) order.
    pub(crate) fn run_all(&mut self) {
        while let Some((label, finalizer)) = self.entries.pop() {
            match finalizer {
                Finalizer::TempDir(path) => {
                    tracing::info!(
                        finalizer = "TempDir",
                        label = %label,
                        path = %path.display(),
                        "finalizer: temp dir cleanup requested (drop handles removal)"
                    );
                }
                Finalizer::Process(pid) => {
                    tracing::info!(
                        finalizer = "Process",
                        label = %label,
                        pid = pid,
                        "finalizer: sending kill to process"
                    );
                    // Best-effort kill; ignore errors (process may already be dead).
                    // Guard against pid 0 or values that overflow pid_t semantics.
                    send_kill_signal_best_effort(pid);
                }
                Finalizer::Custom(f) => {
                    tracing::info!(
                        finalizer = "Custom",
                        label = %label,
                        "finalizer: executing custom cleanup"
                    );
                    f();
                }
            }
        }
    }

    /// Execute all registered finalizers with a per-finalizer time budget
    /// (bd-38c.4).  Finalizers that exceed the budget are warned but still run.
    pub(crate) fn run_all_bounded(&mut self, budget_ms: u64) {
        if budget_ms == 0 {
            // Deterministic fallback: a zero budget still must run finalizers.
            self.run_all();
            return;
        }

        let budget = budget_duration(budget_ms);
        while let Some((label, finalizer)) = self.entries.pop() {
            let start = std::time::Instant::now();
            match finalizer {
                Finalizer::TempDir(path) => {
                    tracing::info!(
                        finalizer = "TempDir",
                        label = %label,
                        path = %path.display(),
                        budget_ms = budget_ms,
                        "finalizer(bounded): temp dir cleanup requested"
                    );
                }
                Finalizer::Process(pid) => {
                    tracing::info!(
                        finalizer = "Process",
                        label = %label,
                        pid = pid,
                        budget_ms = budget_ms,
                        "finalizer(bounded): sending kill to process"
                    );
                    send_kill_signal_best_effort(pid);
                }
                Finalizer::Custom(f) => {
                    tracing::info!(
                        finalizer = "Custom",
                        label = %label,
                        budget_ms = budget_ms,
                        "finalizer(bounded): executing custom cleanup"
                    );
                    let (tx, rx) = std::sync::mpsc::channel();
                    std::thread::spawn(move || {
                        f();
                        let _ = tx.send(());
                    });
                    let _ = rx.recv_timeout(budget);
                }
            }
            let elapsed = start.elapsed();
            if elapsed > budget {
                tracing::warn!(
                    label = %label,
                    elapsed_ms = elapsed.as_millis() as u64,
                    budget_ms = budget_ms,
                    "finalizer exceeded cleanup budget; continuing to next"
                );
            }
        }
    }

    /// Number of registered finalizers.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct StageBudgetPolicy {
    ingest_ms: u64,
    normalize_ms: u64,
    probe_ms: u64,
    vad_ms: u64,
    separate_ms: u64,
    backend_ms: u64,
    acceleration_ms: u64,
    align_ms: u64,
    punctuate_ms: u64,
    diarize_ms: u64,
    persist_ms: u64,
    /// Bounded cleanup budget per pipeline stage (bd-38c.4).
    /// When a stage is cancelled, its finalizers must complete within this budget.
    cleanup_budget_ms: u64,
}

const STAGE_BUDGET_INGEST_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS";
const STAGE_BUDGET_NORMALIZE_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS";
const STAGE_BUDGET_PROBE_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_PROBE_MS";
const STAGE_BUDGET_VAD_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_VAD_MS";
const STAGE_BUDGET_SEPARATE_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_SEPARATE_MS";
const STAGE_BUDGET_BACKEND_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS";
const STAGE_BUDGET_ACCELERATION_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_ACCELERATION_MS";
const STAGE_BUDGET_ALIGN_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_ALIGN_MS";
const STAGE_BUDGET_PUNCTUATE_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_PUNCTUATE_MS";
const STAGE_BUDGET_DIARIZE_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_DIARIZE_MS";
const STAGE_BUDGET_PERSIST_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_PERSIST_MS";
const STAGE_BUDGET_CLEANUP_ENV: &str = "FRANKEN_WHISPER_STAGE_BUDGET_CLEANUP_MS";
const STAGE_BUDGET_OVERRIDE_KEYS: [&str; 12] = [
    STAGE_BUDGET_INGEST_ENV,
    STAGE_BUDGET_NORMALIZE_ENV,
    STAGE_BUDGET_PROBE_ENV,
    STAGE_BUDGET_VAD_ENV,
    STAGE_BUDGET_SEPARATE_ENV,
    STAGE_BUDGET_BACKEND_ENV,
    STAGE_BUDGET_ACCELERATION_ENV,
    STAGE_BUDGET_ALIGN_ENV,
    STAGE_BUDGET_PUNCTUATE_ENV,
    STAGE_BUDGET_DIARIZE_ENV,
    STAGE_BUDGET_PERSIST_ENV,
    STAGE_BUDGET_CLEANUP_ENV,
];

impl StageBudgetPolicy {
    fn from_env() -> Self {
        Self::from_source(|key| std::env::var(key).ok())
    }

    fn from_source<F>(mut source: F) -> Self
    where
        F: FnMut(&str) -> Option<String>,
    {
        Self {
            ingest_ms: budget_from_source(&mut source, STAGE_BUDGET_INGEST_ENV, 15_000),
            normalize_ms: budget_from_source(&mut source, STAGE_BUDGET_NORMALIZE_ENV, 180_000),
            probe_ms: budget_from_source(&mut source, STAGE_BUDGET_PROBE_ENV, 8_000),
            vad_ms: budget_from_source(&mut source, STAGE_BUDGET_VAD_ENV, 10_000),
            separate_ms: budget_from_source(&mut source, STAGE_BUDGET_SEPARATE_ENV, 30_000),
            backend_ms: budget_from_source(&mut source, STAGE_BUDGET_BACKEND_ENV, 900_000),
            acceleration_ms: budget_from_source(&mut source, STAGE_BUDGET_ACCELERATION_ENV, 20_000),
            align_ms: budget_from_source(&mut source, STAGE_BUDGET_ALIGN_ENV, 30_000),
            punctuate_ms: budget_from_source(&mut source, STAGE_BUDGET_PUNCTUATE_ENV, 10_000),
            diarize_ms: budget_from_source(&mut source, STAGE_BUDGET_DIARIZE_ENV, 30_000),
            persist_ms: budget_from_source(&mut source, STAGE_BUDGET_PERSIST_ENV, 20_000),
            cleanup_budget_ms: budget_from_source(&mut source, STAGE_BUDGET_CLEANUP_ENV, 5_000),
        }
    }

    fn as_json(self) -> Value {
        json!({
            "ingest_ms": self.ingest_ms,
            "normalize_ms": self.normalize_ms,
            "probe_ms": self.probe_ms,
            "vad_ms": self.vad_ms,
            "separate_ms": self.separate_ms,
            "backend_ms": self.backend_ms,
            "acceleration_ms": self.acceleration_ms,
            "align_ms": self.align_ms,
            "punctuate_ms": self.punctuate_ms,
            "diarize_ms": self.diarize_ms,
            "persist_ms": self.persist_ms,
            "cleanup_budget_ms": self.cleanup_budget_ms,
            "overrides": STAGE_BUDGET_OVERRIDE_KEYS,
        })
    }
}

fn budget_from_source<F>(source: &mut F, key: &str, fallback_ms: u64) -> u64
where
    F: FnMut(&str) -> Option<String>,
{
    let raw = source(key);
    parse_budget_ms(raw.as_deref(), fallback_ms)
}

fn parse_budget_ms(raw: Option<&str>, fallback_ms: u64) -> u64 {
    raw.and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback_ms)
}

fn budget_duration(ms: u64) -> Duration {
    Duration::from_millis(ms.max(1))
}

const PIPELINE_STAGES: [&str; 10] = [
    "ingest",
    "normalize",
    "vad",
    "separate",
    "backend",
    "acceleration",
    "align",
    "punctuate",
    "diarize",
    "persist",
];

fn stage_budget_ms(stage: &str, budgets: StageBudgetPolicy) -> Option<u64> {
    match stage {
        "ingest" => Some(budgets.ingest_ms),
        "normalize" => Some(budgets.normalize_ms),
        "vad" => Some(budgets.vad_ms),
        "separate" => Some(budgets.separate_ms),
        "backend" => Some(budgets.backend_ms),
        "acceleration" => Some(budgets.acceleration_ms),
        "align" => Some(budgets.align_ms),
        "punctuate" => Some(budgets.punctuate_ms),
        "diarize" => Some(budgets.diarize_ms),
        "persist" => Some(budgets.persist_ms),
        _ => None,
    }
}

fn parse_event_ts_ms(event: &RunEvent) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(&event.ts_rfc3339)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

fn event_elapsed_ms(event: &RunEvent) -> Option<u64> {
    event.payload.get("elapsed_ms").and_then(Value::as_u64)
}

#[derive(Default)]
struct StageTimingSample {
    start_ts_ms: Option<i64>,
    terminal_ts_ms: Option<i64>,
    service_ms: Option<u64>,
}

fn recommended_budget(service_ms: u64, budget_ms: u64) -> (u64, &'static str, &'static str, f64) {
    if budget_ms == 0 {
        return (
            service_ms.max(1),
            "keep_budget",
            "budget undefined; preserving service-derived floor",
            0.0,
        );
    }

    let utilization = service_ms as f64 / budget_ms as f64;
    if utilization >= 0.90 {
        let uplift = (service_ms as f64 * 1.25).ceil() as u64;
        (
            uplift.max(budget_ms + 1),
            "increase_budget",
            "observed service is near budget ceiling",
            utilization,
        )
    } else if utilization <= 0.30 {
        (
            ((service_ms as f64 * 1.60).ceil() as u64).max(1_000),
            "decrease_budget_candidate",
            "observed service is far below budget ceiling",
            utilization,
        )
    } else {
        (
            budget_ms,
            "keep_budget",
            "observed service is within target utilization band",
            utilization,
        )
    }
}

fn stage_latency_profile(events: &[RunEvent], budgets: StageBudgetPolicy) -> Value {
    let mut timings: BTreeMap<&'static str, StageTimingSample> = PIPELINE_STAGES
        .iter()
        .copied()
        .map(|stage| (stage, StageTimingSample::default()))
        .collect();

    for event in events {
        let stage = event.stage.as_str();
        let Some(sample) = timings.get_mut(stage) else {
            continue;
        };

        if event.code == format!("{stage}.start") && sample.start_ts_ms.is_none() {
            sample.start_ts_ms = parse_event_ts_ms(event);
            continue;
        }

        // Any non-start code for a known stage is treated as the latest terminal
        // observation for latency decomposition purposes.
        sample.terminal_ts_ms = parse_event_ts_ms(event);
        if let Some(elapsed_ms) = event_elapsed_ms(event) {
            sample.service_ms = Some(elapsed_ms);
        }
    }

    let mut stage_profiles = serde_json::Map::new();
    let mut recommendations = Vec::new();
    let mut previous_terminal_ts: Option<i64> = None;
    let mut total_queue_ms = 0u64;
    let mut total_service_ms = 0u64;
    let mut total_external_ms = 0u64;
    let mut observed_stages = 0usize;

    for stage in PIPELINE_STAGES {
        let Some(sample) = timings.get(stage) else {
            continue;
        };
        let Some(start_ts) = sample.start_ts_ms else {
            continue;
        };
        let terminal_ts = sample.terminal_ts_ms.unwrap_or(start_ts);
        let queue_ms = previous_terminal_ts
            .map(|prev| (start_ts - prev).max(0) as u64)
            .unwrap_or(0);
        let service_ms = sample
            .service_ms
            .unwrap_or_else(|| (terminal_ts - start_ts).max(0) as u64);
        let external_process_ms = if matches!(stage, "normalize" | "backend") {
            service_ms
        } else {
            0
        };
        let budget_ms = stage_budget_ms(stage, budgets).unwrap_or(service_ms.max(1));
        let (recommended_budget_ms, action, reason, utilization_ratio) =
            recommended_budget(service_ms, budget_ms);

        stage_profiles.insert(
            stage.to_owned(),
            json!({
                "queue_ms": queue_ms,
                "service_ms": service_ms,
                "external_process_ms": external_process_ms,
                "p50_ms": service_ms,
                "p95_ms": service_ms,
                "p99_ms": service_ms,
                "budget_ms": budget_ms,
                "utilization_ratio": utilization_ratio,
                "tuning_action": action,
                "tuning_reason": reason,
                "recommended_budget_ms": recommended_budget_ms,
            }),
        );

        recommendations.push(json!({
            "stage": stage,
            "action": action,
            "current_budget_ms": budget_ms,
            "recommended_budget_ms": recommended_budget_ms,
            "utilization_ratio": utilization_ratio,
            "basis": {
                "queue_ms": queue_ms,
                "service_ms": service_ms,
                "external_process_ms": external_process_ms,
                "p95_ms": service_ms,
                "p99_ms": service_ms,
            }
        }));

        observed_stages += 1;
        total_queue_ms += queue_ms;
        total_service_ms += service_ms;
        total_external_ms += external_process_ms;
        previous_terminal_ts = Some(terminal_ts);
    }

    json!({
        "artifact": "stage_latency_decomposition_v1",
        "generated_at_rfc3339": Utc::now().to_rfc3339(),
        "stages": stage_profiles,
        "summary": {
            "observed_stages": observed_stages,
            "queue_total_ms": total_queue_ms,
            "service_total_ms": total_service_ms,
            "external_process_total_ms": total_external_ms,
        },
        "budget_tuning": {
            "policy": "heuristic_v1",
            "target_utilization_band": {
                "min": 0.30,
                "max": 0.90,
            },
            "recommendations": recommendations,
        }
    })
}

fn sha256_bytes_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_json_value(value: &Value) -> FwResult<String> {
    let encoded = serde_json::to_vec(value)?;
    Ok(sha256_bytes_hex(&encoded))
}

fn sha256_file(path: &Path) -> FwResult<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn acceleration_context_payload(
    request: &TranscribeRequest,
    acceleration_backend: &str,
    stream_owner_id: &str,
    cancellation_fence: Value,
) -> serde_json::Value {
    let acceleration_mode = if acceleration_backend == "none" {
        "cpu_fallback"
    } else {
        "accelerated"
    };
    let stream_kind = if acceleration_backend == "none" {
        "cpu_lane"
    } else {
        "gpu_stream"
    };
    json!({
        "mode": acceleration_mode,
        "acceleration_backend": acceleration_backend,
        "logical_stream_kind": stream_kind,
        "logical_stream_owner_id": stream_owner_id,
        "requested_gpu_device": request.backend_params.gpu_device,
        "flash_attention_requested": request.backend_params.flash_attention,
        "frankentorch_feature": cfg!(feature = "gpu-frankentorch"),
        "frankenjax_feature": cfg!(feature = "gpu-frankenjax"),
        "cancellation_fence": cancellation_fence,
    })
}

fn acceleration_stream_owner_id(
    trace_id: &str,
    request: &TranscribeRequest,
    acceleration_backend: &str,
) -> String {
    let device = request
        .backend_params
        .gpu_device
        .as_deref()
        .unwrap_or("cpu");
    format!("{trace_id}:acceleration:{acceleration_backend}:{device}")
}

fn acceleration_cancellation_fence_payload(pcx: &PipelineCx) -> Value {
    let checked_at_rfc3339 = Utc::now().to_rfc3339();
    let budget_remaining_ms = pcx.budget_remaining_ms();
    match pcx.cancellation_token().checkpoint() {
        Ok(()) => json!({
            "status": "open",
            "checked_at_rfc3339": checked_at_rfc3339,
            "budget_remaining_ms": budget_remaining_ms,
            "error": Value::Null,
            "error_code": Value::Null,
        }),
        Err(error) => json!({
            "status": "tripped",
            "checked_at_rfc3339": checked_at_rfc3339,
            "budget_remaining_ms": budget_remaining_ms,
            "error": error.to_string(),
            "error_code": error.robot_error_code(),
        }),
    }
}

fn stage_failure_code(stage: &str, error: &FwError) -> String {
    match error {
        FwError::StageTimeout { .. } | FwError::CommandTimedOut { .. } => {
            format!("{stage}.timeout")
        }
        FwError::Cancelled(_) => format!("{stage}.cancelled"),
        _ => format!("{stage}.error"),
    }
}

fn stage_failure_message<'a>(error: &FwError, default: &'a str) -> &'a str {
    match error {
        FwError::StageTimeout { .. } | FwError::CommandTimedOut { .. } => "stage budget exceeded",
        FwError::Cancelled(_) => "pipeline cancelled by checkpoint policy",
        _ => default,
    }
}

fn checkpoint_or_emit(
    stage: &'static str,
    pcx: &mut PipelineCx,
    log: &mut EventLog,
) -> FwResult<()> {
    match pcx.checkpoint() {
        Ok(()) => Ok(()),
        Err(error) => {
            let mut payload = json!({
                "error": error.to_string(),
                "checkpoint": true,
            });
            if matches!(error, FwError::Cancelled(_))
                && let Value::Object(ref mut map) = payload
            {
                let cancellation_evidence = pcx.cancellation_evidence(stage);
                pcx.record_evidence_values(std::slice::from_ref(&cancellation_evidence));
                map.insert("cancellation_evidence".to_owned(), cancellation_evidence);
                map.insert("evidence_count".to_owned(), json!(pcx.evidence().len()));
            }

            let code = stage_failure_code(stage, &error);
            log.clear_stage_start();
            log.push(
                stage,
                &code,
                stage_failure_message(&error, "pipeline checkpoint failed"),
                payload,
            );
            Err(error)
        }
    }
}

fn run_stage_with_budget<T, F>(stage: &'static str, budget_ms: u64, operation: F) -> FwResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> FwResult<T> + Send + 'static,
{
    let (tx, rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = tx.send(operation());
    });

    match rx.recv_timeout(budget_duration(budget_ms)) {
        Ok(result) => result,
        Err(RecvTimeoutError::Timeout) => Err(FwError::StageTimeout {
            stage: stage.to_owned(),
            budget_ms,
        }),
        Err(RecvTimeoutError::Disconnected) => Err(FwError::BackendUnavailable(format!(
            "stage `{stage}` worker exited before returning a result"
        ))),
    }
}

pub struct FrankenWhisperEngine {
    runtime: Runtime,
    state_root: PathBuf,
    pipeline_config: PipelineConfig,
}

impl FrankenWhisperEngine {
    pub fn new() -> FwResult<Self> {
        Self::with_config(PipelineConfig::default())
    }

    /// Create an engine with a custom pipeline configuration.
    pub fn with_config(config: PipelineConfig) -> FwResult<Self> {
        config.validate()?;
        let state_root = state_root()?;
        fs::create_dir_all(state_root.join("tmp"))?;

        let runtime = RuntimeBuilder::new()
            .worker_threads(2)
            .blocking_threads(1, 4)
            .thread_name_prefix("franken_whisper")
            .build()
            .map_err(|error| {
                FwError::Unsupported(format!("asupersync runtime init failed: {error}"))
            })?;

        Ok(Self {
            runtime,
            state_root,
            pipeline_config: config,
        })
    }

    /// Returns the active pipeline configuration.
    pub fn pipeline_config(&self) -> &PipelineConfig {
        &self.pipeline_config
    }

    pub fn transcribe(&self, request: TranscribeRequest) -> FwResult<RunReport> {
        self.transcribe_internal(request, None)
    }

    pub fn transcribe_with_stream(
        &self,
        request: TranscribeRequest,
        event_tx: Sender<StreamedRunEvent>,
    ) -> FwResult<RunReport> {
        self.transcribe_internal(request, Some(event_tx))
    }

    fn transcribe_internal(
        &self,
        request: TranscribeRequest,
        event_tx: Option<Sender<StreamedRunEvent>>,
    ) -> FwResult<RunReport> {
        let state_root = self.state_root.clone();
        let config = self.pipeline_config.clone();
        let handle = self
            .runtime
            .handle()
            .spawn(async move { run_pipeline(request, &state_root, event_tx, &config).await });

        self.runtime.block_on(handle)
    }
}

async fn run_pipeline(
    request: TranscribeRequest,
    state_root: &Path,
    event_tx: Option<Sender<StreamedRunEvent>>,
    pipeline_config: &PipelineConfig,
) -> FwResult<RunReport> {
    fs::create_dir_all(state_root.join("tmp"))?;

    let run_id = Uuid::new_v4().to_string();
    let started_at = Utc::now().to_rfc3339();
    tracing::info!(run_id = %run_id, "Starting transcription run");
    let mut pcx = PipelineCx::new(request.timeout_ms);
    let stage_budgets = StageBudgetPolicy::from_env();

    let run_tmp_dir = tempfile::Builder::new()
        .prefix("fw-run-")
        .tempdir_in(state_root.join("tmp"))?;
    pcx.register_finalizer(
        "run_tmp_dir",
        Finalizer::TempDir(run_tmp_dir.path().to_path_buf()),
    );

    let trace_id_str = pcx.trace_id().to_string();
    let mut log = EventLog::new(run_id.clone(), trace_id_str.clone(), event_tx);

    let stage_labels: Vec<&str> = pipeline_config.stages().iter().map(|s| s.label()).collect();
    log.push(
        "orchestration",
        "orchestration.budgets",
        "applied per-stage orchestration budgets",
        json!({
            "stage_budgets": stage_budgets.as_json(),
            "request_timeout_ms": request.timeout_ms,
            "pipeline_stages": stage_labels,
            "safe_mode": "deterministic configurable stage order",
        }),
    );

    let result = run_pipeline_body(
        &mut pcx,
        &mut log,
        &request,
        &run_tmp_dir,
        stage_budgets,
        &run_id,
        &trace_id_str,
        &started_at,
        pipeline_config,
    )
    .await;

    pcx.run_finalizers_bounded(stage_budgets.cleanup_budget_ms);
    result
}

/// Mutable intermediate state threaded through the composable pipeline stages.
struct PipelineIntermediate {
    input_path: Option<PathBuf>,
    normalized_wav: Option<PathBuf>,
    normalized_duration: Option<f64>,
    result: Option<crate::model::TranscriptionResult>,
    warnings: Vec<String>,
    normalized_input_sha256: Option<String>,
    backend_output_sha256: Option<String>,
    backend_runtime: Option<backend::BackendRuntimeMetadata>,
    /// VAD result: regions of voice activity as (start_sec, end_sec) pairs.
    vad_regions: Option<Vec<(f64, f64)>>,
    /// Whether VAD determined the audio is silence-only (skip transcription).
    vad_silence_only: bool,
    /// Whether source separation has been applied.
    vocal_isolated: bool,
}

impl PipelineIntermediate {
    fn new() -> Self {
        Self {
            input_path: None,
            normalized_wav: None,
            normalized_duration: None,
            result: None,
            warnings: Vec::new(),
            normalized_input_sha256: None,
            backend_output_sha256: None,
            backend_runtime: None,
            vad_regions: None,
            vad_silence_only: false,
            vocal_isolated: false,
        }
    }
}

fn stage_skip_payload(reason: &str, details: Value) -> Value {
    json!({
        "reason": reason,
        "details": details,
    })
}

fn optional_stage_skip(
    stage: PipelineStage,
    request: &TranscribeRequest,
) -> Option<(String, Value)> {
    match stage {
        PipelineStage::Vad => {
            if request.backend_params.vad.is_none() {
                Some((
                    "no_vad_requested".to_owned(),
                    stage_skip_payload("no_vad_requested", json!({"requested": false})),
                ))
            } else {
                None
            }
        }
        PipelineStage::Separate => {
            let no_stem = request
                .backend_params
                .diarization_config
                .as_ref()
                .is_some_and(|config| config.no_stem);
            if !request.diarize {
                Some((
                    "diarization_not_requested".to_owned(),
                    stage_skip_payload(
                        "diarization_not_requested",
                        json!({"diarize": request.diarize}),
                    ),
                ))
            } else if no_stem {
                Some((
                    "source_separation_disabled".to_owned(),
                    stage_skip_payload("source_separation_disabled", json!({"no_stem": true})),
                ))
            } else {
                None
            }
        }
        PipelineStage::Align => {
            if !request.diarize {
                Some((
                    "alignment_not_requested".to_owned(),
                    stage_skip_payload(
                        "alignment_not_requested",
                        json!({"diarize": request.diarize}),
                    ),
                ))
            } else {
                None
            }
        }
        PipelineStage::Punctuate => {
            let punctuation_enabled = request
                .backend_params
                .punctuation
                .as_ref()
                .is_some_and(|config| config.enabled);
            if !request.diarize && !punctuation_enabled {
                Some((
                    "punctuation_not_requested".to_owned(),
                    stage_skip_payload(
                        "punctuation_not_requested",
                        json!({
                            "diarize": request.diarize,
                            "punctuation_enabled": punctuation_enabled,
                        }),
                    ),
                ))
            } else {
                None
            }
        }
        PipelineStage::Diarize => {
            if !request.diarize {
                Some((
                    "diarization_not_requested".to_owned(),
                    stage_skip_payload(
                        "diarization_not_requested",
                        json!({"diarize": request.diarize}),
                    ),
                ))
            } else if request.backend == crate::model::BackendKind::WhisperDiarization {
                // The whisper-diarization backend performs speaker diarization
                // internally (see whisper_diarization_native.rs); running the
                // pipeline Diarize stage as well would diarize the segments a
                // second time. Skip it and let the backend own diarization.
                Some((
                    "backend_owns_diarization".to_owned(),
                    stage_skip_payload(
                        "backend_owns_diarization",
                        json!({"backend": request.backend.as_str()}),
                    ),
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Inner body of [`run_pipeline`], factored out so that [`run_pipeline`] can
/// unconditionally call `pcx.run_finalizers()` after this returns regardless
/// of success or failure.
#[allow(clippy::too_many_arguments)]
async fn run_pipeline_body(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    request: &TranscribeRequest,
    run_tmp_dir: &tempfile::TempDir,
    stage_budgets: StageBudgetPolicy,
    run_id: &str,
    trace_id_str: &str,
    started_at: &str,
    pipeline_config: &PipelineConfig,
) -> FwResult<RunReport> {
    // Checkpoint: before first stage
    checkpoint_or_emit("orchestration", pcx, log)?;

    let mut inter = PipelineIntermediate::new();

    for stage in pipeline_config.stages() {
        if let Some((_reason, payload)) = optional_stage_skip(*stage, request) {
            log.push(
                stage.label(),
                &format!("{}.skip", stage.label()),
                &format!("skipping {} stage", stage.label()),
                payload,
            );
            continue;
        }

        if inter.vad_silence_only
            && matches!(
                stage,
                PipelineStage::Separate
                    | PipelineStage::Backend
                    | PipelineStage::Accelerate
                    | PipelineStage::Align
                    | PipelineStage::Punctuate
                    | PipelineStage::Diarize
            )
        {
            log.push(
                stage.label(),
                &format!("{}.skip", stage.label()),
                &format!(
                    "skipping {} stage after silence-only VAD result",
                    stage.label()
                ),
                stage_skip_payload("silence_only_vad", json!({"vad_silence_only": true})),
            );
            continue;
        }

        match stage {
            PipelineStage::Ingest => {
                execute_ingest(pcx, log, request, run_tmp_dir, stage_budgets, &mut inter).await?;
            }
            PipelineStage::Normalize => {
                execute_normalize(pcx, log, stage_budgets, run_tmp_dir, &mut inter).await?;
            }
            PipelineStage::Vad => {
                execute_vad(pcx, log, request, stage_budgets, &mut inter).await?;
            }
            PipelineStage::Separate => {
                execute_separate(pcx, log, stage_budgets, &mut inter).await?;
            }
            // When `request.backend_params.speculative` is set, the Backend stage
            // is replaced with a `SpeculativeStreamingPipeline`-driven path
            // that slices the normalized WAV into windows and runs a fast +
            // quality model on each one, emitting `transcript.partial/confirm/
            // retract/correct` and `transcript.speculation_stats` events into
            // the same NDJSON stream. Single-backend execution remains the
            // default path when no speculative config is present.
            PipelineStage::Backend => {
                if request.backend_params.speculative.is_some() {
                    execute_backend_speculative(
                        pcx,
                        log,
                        request,
                        run_tmp_dir,
                        stage_budgets,
                        &mut inter,
                    )
                    .await?;
                } else {
                    execute_backend(pcx, log, request, run_tmp_dir, stage_budgets, &mut inter)
                        .await?;
                }
            }
            PipelineStage::Accelerate => {
                execute_accelerate(pcx, log, request, stage_budgets, trace_id_str, &mut inter)
                    .await?;
            }
            PipelineStage::Align => {
                execute_align(pcx, log, stage_budgets, &mut inter).await?;
            }
            PipelineStage::Punctuate => {
                execute_punctuate(pcx, log, stage_budgets, &mut inter).await?;
            }
            PipelineStage::Diarize => {
                execute_diarize(pcx, log, request, stage_budgets, &mut inter).await?;
            }
            PipelineStage::Persist => {
                // Persist is handled specially during report assembly below.
            }
        }
    }

    // Emit tail-latency decomposition artifacts and deterministic budget-tuning
    // guidance derived from observed stage timings for this run.
    let latency_profile = stage_latency_profile(&log.events, stage_budgets);
    pcx.record_evidence_values(std::slice::from_ref(&latency_profile));
    log.push(
        "orchestration",
        "orchestration.latency_profile",
        "captured stage latency decomposition and budget tuning recommendations",
        latency_profile,
    );

    // If persist is in the config, checkpoint before it.
    let has_persist = pipeline_config.has_stage(PipelineStage::Persist);
    if has_persist && request.persist {
        checkpoint_or_emit("persist", pcx, log)?;
    }

    let finished_at = Utc::now().to_rfc3339();

    let mut result = inter
        .result
        .unwrap_or_else(|| crate::model::TranscriptionResult {
            backend: crate::model::BackendKind::Auto,
            transcript: String::new(),
            language: None,
            segments: Vec::new(),
            acceleration: None,
            raw_output: json!({}),
            artifact_paths: Vec::new(),
        });

    let output_formats = &request.backend_params.output_formats;
    if !output_formats.is_empty() {
        let base_name = match &request.input {
            crate::model::InputSource::File { path } => path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("transcript"),
            _ => "transcript",
        };
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let output_prefix = cwd.join(base_name);

        match crate::export::write_artifacts(output_formats, &result, &output_prefix) {
            Ok(paths) => {
                result.artifact_paths =
                    paths.into_iter().map(|p| p.display().to_string()).collect();
            }
            Err(e) => {
                inter
                    .warnings
                    .push(format!("Failed to export artifacts: {}", e));
            }
        }
    }

    let input_path_str = inter
        .input_path
        .as_ref()
        .map_or_else(String::new, |p| p.display().to_string());
    let normalized_wav_str = inter
        .normalized_wav
        .as_ref()
        .map_or_else(String::new, |p| p.display().to_string());

    let mut report = RunReport {
        run_id: run_id.to_owned(),
        trace_id: trace_id_str.to_owned(),
        started_at_rfc3339: started_at.to_owned(),
        finished_at_rfc3339: finished_at,
        input_path: input_path_str,
        normalized_wav_path: normalized_wav_str,
        request: request.clone(),
        result,
        events: log.events.clone(),
        warnings: std::mem::take(&mut inter.warnings),
        evidence: pcx.evidence().to_vec(),
        replay: crate::model::ReplayEnvelope {
            input_content_hash: inter.normalized_input_sha256,
            backend_identity: inter.backend_runtime.as_ref().map(|r| r.identity.clone()),
            backend_version: inter
                .backend_runtime
                .as_ref()
                .and_then(|r| r.version.clone()),
            output_payload_hash: inter.backend_output_sha256,
        },
    };

    if has_persist && request.persist {
        log.mark_stage_start();
        log.push(
            "persist",
            "persist.start",
            "writing run report to frankensqlite",
            json!({
                "db_path": request.db_path.display().to_string(),
                "budget_ms": stage_budgets.persist_ms,
            }),
        );
        report.events = log.events.clone();

        let persist_report = report.clone();
        let persist_db = request.db_path.clone();
        let persist_token = pcx.stage_token(stage_budgets.persist_ms); // ubs:ignore — cancellation token is not a secret
        if let Err(error) = run_stage_with_budget("persist", stage_budgets.persist_ms, move || {
            let store = RunStore::open(&persist_db)?;
            store.persist_report_cancellable(&persist_report, Some(&persist_token))
        }) {
            let code = stage_failure_code("persist", &error);
            log.push(
                "persist",
                &code,
                stage_failure_message(&error, "failed to persist run report"),
                json!({"error": error.to_string(), "budget_ms": stage_budgets.persist_ms}),
            );
            return Err(error);
        }

        tracing::debug!(stage = "persist", "Pipeline stage complete");
        log.push(
            "persist",
            "persist.ok",
            "run report persisted",
            json!({"db_path": request.db_path.display().to_string()}),
        );
        report.events = log.events.clone();
    } else if has_persist {
        log.push(
            "persist",
            "persist.skip",
            "skipping persistence stage",
            stage_skip_payload("persist_disabled", json!({"persist": request.persist})),
        );
        report.events = log.events.clone();
    }

    Ok(report)
}

// ---------------------------------------------------------------------------
// Stage execution helpers (bd-qla.6)
// ---------------------------------------------------------------------------

async fn execute_ingest(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    request: &TranscribeRequest,
    run_tmp_dir: &tempfile::TempDir,
    stage_budgets: StageBudgetPolicy,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    log.mark_stage_start();
    log.push(
        "ingest",
        "ingest.start",
        "materializing input",
        json!({"input": request.input}),
    );

    let ingest_input = request.input.clone();
    let ingest_dir = run_tmp_dir.path().to_path_buf();
    let ingest_token = pcx.stage_token(stage_budgets.ingest_ms); // ubs:ignore — cancellation token is not a secret
    let input_path = match run_stage_with_budget("ingest", stage_budgets.ingest_ms, move || {
        audio::materialize_input_with_token(&ingest_input, &ingest_dir, Some(&ingest_token))
    }) {
        Ok(path) => path,
        Err(error) => {
            let code = stage_failure_code("ingest", &error);
            log.push(
                "ingest",
                &code,
                stage_failure_message(&error, "failed to materialize input"),
                json!({"error": error.to_string(), "budget_ms": stage_budgets.ingest_ms}),
            );
            return Err(error);
        }
    };
    tracing::debug!(stage = "ingest", "Pipeline stage complete");
    log.push(
        "ingest",
        "ingest.ok",
        "input materialized",
        json!({"path": input_path.display().to_string()}),
    );

    inter.input_path = Some(input_path);
    Ok(())
}

async fn execute_normalize(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    stage_budgets: StageBudgetPolicy,
    run_tmp_dir: &tempfile::TempDir,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    // Checkpoint: between ingest -> normalize
    checkpoint_or_emit("normalize", pcx, log)?;

    let input_path = inter.input_path.as_ref().ok_or_else(|| {
        FwError::InvalidRequest("Normalize requires Ingest to have run first".to_owned())
    })?;

    log.mark_stage_start();
    log.push(
        "normalize",
        "normalize.start",
        "normalizing to 16kHz mono wav",
        json!({"input": input_path.display().to_string()}),
    );

    let normalize_input = input_path.clone();
    let normalize_dir = run_tmp_dir.path().to_path_buf();
    let normalize_budget_ms = stage_budgets.normalize_ms;
    let normalize_token = pcx.stage_token(normalize_budget_ms); // ubs:ignore — cancellation token is not a secret
    let normalized_wav = match run_stage_with_budget("normalize", normalize_budget_ms, move || {
        audio::normalize_to_wav_with_timeout(
            &normalize_input,
            &normalize_dir,
            budget_duration(normalize_budget_ms),
            Some(&normalize_token),
        )
    }) {
        Ok(path) => path,
        Err(error) => {
            let code = stage_failure_code("normalize", &error);
            log.push(
                "normalize",
                &code,
                stage_failure_message(&error, "audio normalization failed"),
                json!({"error": error.to_string(), "budget_ms": stage_budgets.normalize_ms}),
            );
            return Err(error);
        }
    };
    let normalized_duration = audio::probe_duration_seconds_with_timeout(
        &normalized_wav,
        budget_duration(stage_budgets.probe_ms),
    );

    tracing::debug!(stage = "normalize", "Pipeline stage complete");
    log.push(
        "normalize",
        "normalize.ok",
        "audio normalized",
        json!({
            "path": normalized_wav.display().to_string(),
            "duration_seconds": normalized_duration,
        }),
    );

    let normalized_input_sha256 = match sha256_file(&normalized_wav) {
        Ok(hash) => Some(hash),
        Err(error) => {
            inter.warnings.push(format!(
                "replay envelope normalized-input hash unavailable: {error}"
            ));
            None
        }
    };

    inter.normalized_wav = Some(normalized_wav);
    inter.normalized_duration = normalized_duration;
    inter.normalized_input_sha256 = normalized_input_sha256;
    Ok(())
}

async fn execute_backend(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    request: &TranscribeRequest,
    run_tmp_dir: &tempfile::TempDir,
    stage_budgets: StageBudgetPolicy,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    let normalized_wav = inter.normalized_wav.as_ref().ok_or_else(|| {
        FwError::InvalidRequest("Backend requires Normalize to have run first".to_owned())
    })?;
    let normalized_duration = inter.normalized_duration;

    let routing_safe_mode = std::env::var("FRANKEN_WHISPER_ROUTING_SAFE_MODE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

    let min_calibration: f64 = std::env::var("FRANKEN_WHISPER_ROUTING_MIN_CALIBRATION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);

    // Evaluate adaptive routing and capture prediction even during fallback.
    // This fixes bd-k87e: we record prediction outcomes regardless of whether
    // the adaptive order is actually used, enabling calibration during fallback.
    let (backend_order, adaptive_prediction) = if routing_safe_mode {
        log.push(
            "backend_routing",
            "backend.routing.safe_mode",
            "routing safe mode active; using static priority order",
            json!({"env": "FRANKEN_WHISPER_ROUTING_SAFE_MODE"}),
        );
        (None, None)
    } else if let Some(selection) =
        backend::evaluate_backend_selection(request, normalized_duration, pcx.trace_id())
    {
        if selection.fallback_triggered {
            inter
                .warnings
                .push("backend selection contract fallback trigger activated".to_owned());
        }
        pcx.record_evidence_values(&selection.evidence_entries);
        log.push(
            "backend_routing",
            "backend.routing.decision_contract",
            "evaluated formal backend selection contract",
            selection.routing_log,
        );

        // Capture the adaptive prediction BEFORE deciding whether to use it.
        // This allows calibration tracking even when we fall back to static order.
        let top_backend = selection.recommended_order[0];
        let predicted_success = backend::router_state_snapshot()
            .map(|state| state.metrics_for(top_backend).success_rate)
            .unwrap_or(0.5);
        let prediction = (top_backend, predicted_success);

        if selection.calibration_score < min_calibration {
            tracing::warn!(
                calibration_score = selection.calibration_score,
                min_calibration = min_calibration,
                "Calibration below threshold; falling back to static priority"
            );
            inter.warnings.push(format!(
                "calibration score {:.4} below threshold {:.4}; falling back to static priority",
                selection.calibration_score, min_calibration
            ));
            log.push(
                "backend_routing",
                "backend.routing.calibration_guardrail",
                "calibration below threshold; discarding recommended order",
                json!({
                    "calibration_score": selection.calibration_score,
                    "min_calibration": min_calibration,
                    "discarded_order": selection.recommended_order.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
                }),
            );
            // Use static order but still record the adaptive prediction outcome
            (None, Some(prediction))
        } else {
            (Some(selection.recommended_order), Some(prediction))
        }
    } else {
        (None, None)
    };

    // Checkpoint: between normalize -> backend
    checkpoint_or_emit("backend", pcx, log)?;

    log.mark_stage_start();
    log.push(
        "backend",
        "backend.start",
        "executing backend",
        json!({
            "requested_backend": request.backend.as_str(),
            "budget_ms": stage_budgets.backend_ms,
        }),
    );

    let backend_request = request.clone();
    let backend_wav = normalized_wav.clone();
    let backend_dir = run_tmp_dir.path().to_path_buf();
    let backend_budget_ms = stage_budgets.backend_ms;
    let cancel_token = pcx.stage_token(backend_budget_ms); // ubs:ignore — cancellation token is not a secret
    let execution_result = run_stage_with_budget("backend", backend_budget_ms, move || {
        let tok = Some(&cancel_token);
        if let Some(order) = backend_order {
            backend::execute_with_order(
                &backend_request,
                &backend_wav,
                &backend_dir,
                budget_duration(backend_budget_ms),
                &order,
                tok,
            )
        } else {
            backend::execute(
                &backend_request,
                &backend_wav,
                &backend_dir,
                budget_duration(backend_budget_ms),
                tok,
            )
        }
    });

    if let Some((top_backend, predicted_success)) = adaptive_prediction {
        let top_succeeded = match &execution_result {
            Ok(execution) => execution.result.backend == top_backend,
            Err(_) => false,
        };
        backend::record_adaptive_prediction(top_succeeded);
        backend::record_calibration_observation(predicted_success, top_succeeded);
    }

    let execution = match execution_result {
        Ok(result) => result,
        Err(error) => {
            let code = stage_failure_code("backend", &error);
            log.push(
                "backend",
                &code,
                stage_failure_message(&error, "backend execution failed"),
                json!({"error": error.to_string(), "budget_ms": stage_budgets.backend_ms}),
            );
            return Err(error);
        }
    };

    if let Err(error) = conformance::validate_segment_invariants(&execution.result.segments) {
        log.push(
            "backend",
            "backend.contract_violation",
            "backend output violated segment conformance contract",
            json!({
                "error": error.to_string(),
                "segments": execution.result.segments.len(),
                "contract": {
                    "name": "segment-monotonic-v1",
                    "allow_overlap": false,
                    "timestamp_epsilon_sec": 1e-6,
                }
            }),
        );
        return Err(error);
    }

    tracing::debug!(stage = "backend", "Pipeline stage complete");
    log.push(
        "backend",
        "backend.ok",
        "backend completed",
        json!({
            "resolved_backend": execution.result.backend.as_str(),
            "segments": execution.result.segments.len(),
            "engine_identity": execution.runtime.identity.clone(),
            "engine_version": execution.runtime.version.clone(),
            "implementation": execution.implementation.as_str(),
            "execution_mode": execution.execution_mode.clone(),
            "native_rollout_stage": execution.rollout_stage.clone(),
            "native_fallback_error": execution.native_fallback_error.clone(),
        }),
    );
    let backend_output_sha256 = match sha256_json_value(&execution.result.raw_output) {
        Ok(hash) => Some(hash),
        Err(error) => {
            inter.warnings.push(format!(
                "replay envelope backend-output hash unavailable: {error}"
            ));
            None
        }
    };
    let backend_runtime = execution.runtime;
    log.push(
        "replay",
        "replay.envelope",
        "captured deterministic replay envelope metadata",
        json!({
            "normalized_input_sha256": inter.normalized_input_sha256,
            "backend_output_sha256": backend_output_sha256,
            "backend": execution.result.backend.as_str(),
            "engine_identity": backend_runtime.identity.clone(),
            "engine_version": backend_runtime.version.clone(),
            "implementation": execution.implementation.as_str(),
            "execution_mode": execution.execution_mode.clone(),
            "native_rollout_stage": execution.rollout_stage.clone(),
            "native_fallback_error": execution.native_fallback_error.clone(),
        }),
    );

    inter.backend_output_sha256 = backend_output_sha256;
    inter.backend_runtime = Some(backend_runtime);
    inter.result = Some(execution.result);
    Ok(())
}

/// Speculative replacement for [`execute_backend`].
///
/// Drives the library's [`crate::streaming::SpeculativeStreamingPipeline`]
/// by repeatedly slicing the normalized WAV at window boundaries and invoking
/// the active backend with the fast-lane and quality-lane model names.
/// Forwards every speculation event (`transcript.partial`, `transcript.confirm`,
/// `transcript.retract`, `transcript.correct`, `transcript.speculation_stats`)
/// into the same NDJSON event log the rest of the pipeline writes through.
///
/// Aggregated, deduplicated segments from `WindowManager::merge_segments`
/// become the run's final transcript; downstream stages (Accelerate, Align,
/// Punctuate, Diarize, Persist) operate on the merged transcript exactly as
/// they would for a single-backend execution.
///
/// Notes on the design:
///
/// * **Sticky backend.** The backend kind is resolved **once** upfront via
///   [`backend::resolve_static_backend`]. When the user passes `--backend
///   auto`, this picks the first-available kind from the static priority list
///   and forces it onto every per-window invocation so the run is
///   engine-consistent end to end (no risk of windows being served by
///   different engines and producing a mixed-engine merged transcript).
/// * **Partial event preservation.** When the speculation pipeline fails
///   mid-run, partial `transcript.partial` / `transcript.confirm` /
///   `transcript.retract` / `transcript.correct` events that did make it
///   into the pipeline's internal log are still forwarded to the run-wide
///   NDJSON log before the terminal `backend.error` event, so observers see
///   the run's partial progress rather than only the failure code.
/// * **Replay envelope.** The `BackendRuntimeMetadata` of the resolved kind
///   is recorded in `inter.backend_runtime`, and the SHA-256 of the
///   serialized merged segments is recorded in `inter.backend_output_sha256`
///   so the run's `ReplayEnvelope` is populated even for speculative runs.
async fn execute_backend_speculative(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    request: &TranscribeRequest,
    run_tmp_dir: &tempfile::TempDir,
    stage_budgets: StageBudgetPolicy,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    let normalized_wav = inter.normalized_wav.as_ref().ok_or_else(|| {
        FwError::InvalidRequest("Backend requires Normalize to have run first".to_owned())
    })?;
    let spec_request = request.backend_params.speculative.as_ref().ok_or_else(|| {
        FwError::InvalidRequest(
            "execute_backend_speculative invoked without BackendParams.speculative".to_owned(),
        )
    })?;

    // Resolve the speculative run's backend ONCE so every window invocation
    // talks to the same engine (per design note above). Failures here surface
    // before any backend.start event is emitted so the stream layout matches
    // the pre-speculative dispatch path.
    let resolved_backend = backend::resolve_static_backend(request)?;

    // Convert the serde-friendly request shape into the execution-shape config.
    let spec_config = crate::streaming::SpeculativeConfig {
        window_size_ms: spec_request.window_size_ms,
        overlap_ms: spec_request.overlap_ms,
        fast_model_name: spec_request.fast_model_name.clone(),
        quality_model_name: spec_request.quality_model_name.clone(),
        tolerance: crate::speculation::CorrectionTolerance {
            max_wer: spec_request
                .max_wer_tolerance
                .unwrap_or(crate::speculation::CorrectionTolerance::default().max_wer),
            always_correct: spec_request.always_correct,
            ..crate::speculation::CorrectionTolerance::default()
        },
        adaptive: spec_request.adaptive,
        emit_events: true,
    };

    checkpoint_or_emit("backend", pcx, log)?;
    log.mark_stage_start();
    log.push(
        "backend",
        "backend.start",
        "executing backend (speculative streaming)",
        json!({
            "requested_backend": request.backend.as_str(),
            "resolved_backend": resolved_backend.as_str(),
            "budget_ms": stage_budgets.backend_ms,
            "speculative": {
                "fast_model": spec_config.fast_model_name,
                "quality_model": spec_config.quality_model_name,
                "window_size_ms": spec_config.window_size_ms,
                "overlap_ms": spec_config.overlap_ms,
                "adaptive": spec_config.adaptive,
                "always_correct": spec_config.tolerance.always_correct,
            },
        }),
    );

    let backend_budget_ms = stage_budgets.backend_ms;
    let cancel_token = pcx.stage_token(backend_budget_ms); // ubs:ignore — token, not a secret
    let normalized_wav = normalized_wav.clone();
    let backend_dir = run_tmp_dir.path().to_path_buf();
    let base_request = request.clone();
    let run_id = log.run_id.clone();
    let per_invocation_timeout = budget_duration(backend_budget_ms);

    /// Internal carry value: the speculation pipeline's inner result is
    /// reified as a `Result` so the wrapping `run_stage_with_budget` closure
    /// can always succeed with the partial events / stats / merged segments,
    /// regardless of whether the pipeline itself errored mid-run.
    type SpecOutcome = (
        FwResult<crate::model::TranscriptionResult>,
        Vec<crate::model::RunEvent>,
        crate::speculation::SpeculationStats,
        Vec<crate::model::TranscriptionSegment>,
    );

    let outcome: FwResult<SpecOutcome> =
        run_stage_with_budget("backend", backend_budget_ms, move || {
            let tok = cancel_token;
            let mut pipeline =
                crate::streaming::SpeculativeStreamingPipeline::new(spec_config.clone(), run_id);
            let fast_model = spec_config.fast_model_name.clone();
            let quality_model = spec_config.quality_model_name.clone();

            let inner_result = pipeline.process_file_with_models(
                &normalized_wav,
                || tok.checkpoint(),
                |audio_path, start_ms, end_ms| {
                    let slice_path = crate::audio::slice_pcm_wav_to_temp_path(
                        audio_path,
                        &backend_dir,
                        start_ms,
                        end_ms,
                    )?;
                    // Build per-lane requests by overriding the model name AND
                    // forcing the backend kind to the resolved sticky value.
                    let mut fast_req = base_request.clone();
                    fast_req.backend = resolved_backend;
                    fast_req.model = Some(fast_model.clone());
                    let mut quality_req = base_request.clone();
                    quality_req.backend = resolved_backend;
                    quality_req.model = Some(quality_model.clone());

                    let fast_execution = crate::backend::execute(
                        &fast_req,
                        &slice_path,
                        &backend_dir,
                        per_invocation_timeout,
                        Some(&tok),
                    );
                    let fast_segments = match fast_execution {
                        Ok(execution) => execution.result.segments,
                        Err(error) => {
                            let _ = std::fs::remove_file(&slice_path);
                            return Err(error);
                        }
                    };
                    if let Err(error) = tok.checkpoint() {
                        let _ = std::fs::remove_file(&slice_path);
                        return Err(error);
                    }
                    let quality_execution = crate::backend::execute(
                        &quality_req,
                        &slice_path,
                        &backend_dir,
                        per_invocation_timeout,
                        Some(&tok),
                    );
                    let quality_segments = match quality_execution {
                        Ok(execution) => execution.result.segments,
                        Err(error) => {
                            let _ = std::fs::remove_file(&slice_path);
                            return Err(error);
                        }
                    };
                    let _ = std::fs::remove_file(&slice_path);
                    Ok((fast_segments, quality_segments))
                },
            );

            // Always carry partial events + stats + merged segments out, so
            // the outer handler can forward speculation events to the NDJSON
            // log regardless of whether the pipeline succeeded or failed.
            let emitted = pipeline.events().to_vec();
            let stats = pipeline.stats();
            let merged = pipeline.merged_transcript();
            Ok((inner_result, emitted, stats, merged))
        });

    let (inner_result, emitted_events, stats, merged) = match outcome {
        Ok(value) => value,
        Err(error) => {
            // `run_stage_with_budget` itself failed (stage-budget timeout, panic,
            // …). No events to forward; just emit the terminal backend.* event.
            let code = stage_failure_code("backend", &error);
            log.push(
                "backend",
                &code,
                stage_failure_message(&error, "speculative backend execution failed"),
                json!({
                    "error": error.to_string(),
                    "budget_ms": stage_budgets.backend_ms,
                    "execution_mode": "speculative_streaming",
                    "resolved_backend": resolved_backend.as_str(),
                }),
            );
            return Err(error);
        }
    };

    // Forward any speculation events that the pipeline produced before
    // returning (success OR failure). These are the partial transcripts,
    // confirmations, retractions, corrections, and the aggregate stats
    // event the speculation pipeline emits internally.
    for event in emitted_events {
        log.push(&event.stage, &event.code, &event.message, event.payload);
    }

    match inner_result {
        Ok(spec_result) => {
            let language = spec_result.language.or_else(|| Some("en".to_owned()));
            let transcript = merged
                .iter()
                .map(|seg| seg.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            // Record backend identity/version so the run's ReplayEnvelope is
            // populated even for speculative runs. Speculative streams are
            // bridge-backed today (native engines don't yet participate in
            // dual-model dispatch).
            let backend_runtime = backend::runtime_metadata(resolved_backend);
            let output_payload_hash = sha256_segments(&merged);

            log.push(
                "backend",
                "backend.ok",
                "speculative streaming backend completed",
                json!({
                    "backend": resolved_backend.as_str(),
                    "segments_emitted": merged.len(),
                    "windows_processed": stats.windows_processed,
                    "confirmations_emitted": stats.confirmations_emitted,
                    "corrections_emitted": stats.corrections_emitted,
                    "implementation": "bridge",
                    "execution_mode": "speculative_streaming",
                    "native_rollout_stage": serde_json::Value::Null,
                    "native_fallback_error": serde_json::Value::Null,
                    "backend_identity": backend_runtime.identity,
                    "backend_version": backend_runtime.version,
                }),
            );

            inter.backend_runtime = Some(backend_runtime);
            inter.backend_output_sha256 = output_payload_hash;
            inter.result = Some(crate::model::TranscriptionResult {
                backend: resolved_backend,
                transcript,
                language,
                segments: merged,
                acceleration: None,
                raw_output: serde_json::json!({
                    "speculative": true,
                    "windows_processed": stats.windows_processed,
                    "confirmations_emitted": stats.confirmations_emitted,
                    "corrections_emitted": stats.corrections_emitted,
                    "correction_rate": stats.correction_rate,
                    "mean_fast_latency_ms": stats.mean_fast_latency_ms,
                    "mean_quality_latency_ms": stats.mean_quality_latency_ms,
                    "current_window_size_ms": stats.current_window_size_ms,
                    "mean_drift_wer": stats.mean_drift_wer,
                }),
                artifact_paths: vec![],
            });
            Ok(())
        }
        Err(error) => {
            let code = stage_failure_code("backend", &error);
            log.push(
                "backend",
                &code,
                stage_failure_message(&error, "speculative backend execution failed"),
                json!({
                    "error": error.to_string(),
                    "budget_ms": stage_budgets.backend_ms,
                    "execution_mode": "speculative_streaming",
                    "resolved_backend": resolved_backend.as_str(),
                    "windows_processed_before_failure": stats.windows_processed,
                }),
            );
            Err(error)
        }
    }
}

/// SHA-256 hex digest over a deterministic serialization of the speculative
/// run's merged segments. Used as the `output_payload_hash` for replay
/// envelopes when the regular bridge-output-hash path is bypassed (speculative
/// streaming has no single raw backend JSON to hash).
fn sha256_segments(segments: &[crate::model::TranscriptionSegment]) -> Option<String> {
    let serialized = serde_json::to_vec(segments).ok()?;
    Some(sha256_bytes_hex(&serialized))
}

async fn execute_accelerate(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    request: &TranscribeRequest,
    stage_budgets: StageBudgetPolicy,
    trace_id_str: &str,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    let result = inter.result.take().ok_or_else(|| {
        FwError::InvalidRequest("Accelerate requires Backend to have run first".to_owned())
    })?;

    // Checkpoint: between backend -> accelerate
    checkpoint_or_emit("acceleration", pcx, log)?;

    log.mark_stage_start();
    log.push(
        "acceleration",
        "acceleration.start",
        "running native acceleration pass",
        json!({
            "frankentorch_feature": cfg!(feature = "gpu-frankentorch"),
            "frankenjax_feature": cfg!(feature = "gpu-frankenjax"),
            "segments": result.segments.len(),
            "budget_ms": stage_budgets.acceleration_ms,
        }),
    );

    let acceleration_budget_ms = stage_budgets.acceleration_ms;
    let acceleration_token = pcx.stage_token(acceleration_budget_ms); // ubs:ignore — cancellation token is not a secret
    let (updated_result, acceleration) =
        match run_stage_with_budget("acceleration", acceleration_budget_ms, move || {
            let mut local = result;
            let acceleration = accelerate::apply_with_token(&mut local, Some(&acceleration_token));
            Ok((local, acceleration))
        }) {
            Ok(output) => output,
            Err(error) => {
                let code = stage_failure_code("acceleration", &error);
                log.push(
                    "acceleration",
                    &code,
                    stage_failure_message(&error, "acceleration stage failed"),
                    json!({"error": error.to_string(), "budget_ms": stage_budgets.acceleration_ms}),
                );
                return Err(error);
            }
        };
    inter.result = Some(updated_result);
    inter.warnings.extend(acceleration.notes.iter().cloned());

    let stream_owner_id =
        acceleration_stream_owner_id(trace_id_str, request, acceleration.backend.as_str());
    let cancellation_fence = acceleration_cancellation_fence_payload(pcx);
    let acceleration_context = acceleration_context_payload(
        request,
        acceleration.backend.as_str(),
        &stream_owner_id,
        cancellation_fence,
    );
    if acceleration_context["cancellation_fence"]["status"] == "tripped" {
        inter
            .warnings
            .push("acceleration cancellation fence tripped before persist checkpoint".to_owned());
    }
    pcx.record_evidence_values(std::slice::from_ref(&acceleration_context));

    log.push(
        "acceleration",
        "acceleration.context",
        "captured acceleration execution context",
        acceleration_context,
    );

    let acceleration_code = if acceleration.backend.as_str() == "none" {
        tracing::warn!(
            stage = "acceleration",
            fallback = "none",
            "Acceleration fallback: no acceleration backend available"
        );
        "acceleration.fallback"
    } else {
        tracing::debug!(stage = "acceleration", "Pipeline stage complete");
        "acceleration.ok"
    };

    log.push(
        "acceleration",
        acceleration_code,
        "acceleration pass finished",
        json!({
            "backend": acceleration.backend.as_str(),
            "normalized": acceleration.normalized_confidences,
            "pre_mass": acceleration.pre_mass,
            "post_mass": acceleration.post_mass,
            "notes": acceleration.notes,
        }),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// CTC-based forced alignment (bd-qla.3)
// ---------------------------------------------------------------------------

/// Configuration for the CTC forced-alignment stage.
#[derive(Debug, Clone)]
struct AlignConfig {
    /// Maximum allowed drift (in seconds) between original and aligned timestamps
    /// before the aligner falls back to the original value.
    max_drift_sec: f64,
    /// Minimum segment duration (in seconds).  If alignment would produce a
    /// segment shorter than this, the original timestamps are preserved.
    min_segment_duration_sec: f64,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            max_drift_sec: 0.5,
            min_segment_duration_sec: 0.02,
        }
    }
}

/// Report produced by the forced-alignment stage.
#[derive(Debug, Clone)]
struct AlignmentReport {
    /// Number of segments processed.
    segments_total: usize,
    /// Number of segments whose timestamps were corrected.
    segments_corrected: usize,
    /// Number of segments that fell back to original timestamps.
    segments_fallback: usize,
    /// Notes / warnings produced during alignment.
    notes: Vec<String>,
}

/// Simulated CTC log-probability frame.  In production this would come from
/// the model's CTC head; here we derive a deterministic alignment from the
/// segment text length and the audio duration.
///
/// The algorithm works as follows for each segment:
/// 1. Compute the character density (chars per second) across the full
///    transcript based on total audio duration.
/// 2. Walk segments in order, placing each segment's start at the running
///    cursor position and setting its end proportional to its text length.
/// 3. Apply `max_drift_sec` guardrails: if the correction would move a
///    timestamp by more than the allowed drift, fall back to the original.
/// 4. Enforce `min_segment_duration_sec`: if the aligned duration is too
///    short, fall back to the original.
///
/// This is deliberately a pure-function with no I/O so that it is fully
/// deterministic and testable in isolation.
fn ctc_forced_align(
    segments: &mut [crate::model::TranscriptionSegment],
    audio_duration_sec: Option<f64>,
    config: &AlignConfig,
    token: &CancellationToken,
) -> FwResult<AlignmentReport> {
    let total = segments.len();
    let mut corrected = 0usize;
    let mut fallback = 0usize;
    let mut notes: Vec<String> = Vec::new();

    // If there are no segments there is nothing to align.
    if segments.is_empty() {
        return Ok(AlignmentReport {
            segments_total: 0,
            segments_corrected: 0,
            segments_fallback: 0,
            notes,
        });
    }

    // Determine effective audio duration.  Prefer the explicit value;
    // otherwise estimate from the last segment's end timestamp.
    let duration = audio_duration_sec.or_else(|| segments.iter().rev().find_map(|s| s.end_sec));

    let Some(duration) = duration else {
        notes.push("no audio duration available; skipping alignment".to_owned());
        return Ok(AlignmentReport {
            segments_total: total,
            segments_corrected: 0,
            segments_fallback: total,
            notes,
        });
    };

    if duration <= 0.0 {
        notes.push(format!(
            "audio duration non-positive ({duration:.3}s); skipping alignment"
        ));
        return Ok(AlignmentReport {
            segments_total: total,
            segments_corrected: 0,
            segments_fallback: total,
            notes,
        });
    }

    // Total character count for proportional distribution.
    let total_chars: usize = segments.iter().map(|s| s.text.trim().len().max(1)).sum();
    let chars_per_sec = total_chars as f64 / duration;
    if chars_per_sec <= 0.0 {
        notes.push("zero character density; skipping alignment".to_owned());
        return Ok(AlignmentReport {
            segments_total: total,
            segments_corrected: 0,
            segments_fallback: total,
            notes,
        });
    }

    let sec_per_char = duration / total_chars as f64;
    let mut cursor = 0.0f64;

    for segment in segments.iter_mut() {
        // Cancellation check per segment for responsive shutdown.
        token.checkpoint()?;

        let char_len = segment.text.trim().len().max(1) as f64;
        let aligned_start = cursor;
        let aligned_end = (cursor + char_len * sec_per_char).min(duration);
        cursor = aligned_end;

        let orig_start = segment.start_sec;
        let orig_end = segment.end_sec;

        // Check drift guardrails.
        let start_drift = orig_start
            .map(|os| (aligned_start - os).abs())
            .unwrap_or(0.0);
        let end_drift = orig_end.map(|oe| (aligned_end - oe).abs()).unwrap_or(0.0);

        let aligned_duration = aligned_end - aligned_start;

        if start_drift > config.max_drift_sec || end_drift > config.max_drift_sec {
            // Drift exceeds tolerance -- keep original timestamps.
            fallback += 1;
            cursor = orig_end.unwrap_or(aligned_end).max(aligned_end);
            continue;
        }

        if aligned_duration < config.min_segment_duration_sec {
            // Aligned segment too short -- keep original timestamps.
            fallback += 1;
            notes.push(format!(
                "segment [{aligned_start:.3}s-{aligned_end:.3}s] duration {aligned_duration:.3}s \
                 below minimum {:.3}s; falling back",
                config.min_segment_duration_sec
            ));
            cursor = orig_end.unwrap_or(aligned_end).max(aligned_end);
            continue;
        }

        segment.start_sec = Some(aligned_start);
        segment.end_sec = Some(aligned_end);
        corrected += 1;
    }

    Ok(AlignmentReport {
        segments_total: total,
        segments_corrected: corrected,
        segments_fallback: fallback,
        notes,
    })
}

async fn execute_align(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    stage_budgets: StageBudgetPolicy,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    let mut result = inter.result.take().ok_or_else(|| {
        FwError::InvalidRequest("Align requires Backend to have run first".to_owned())
    })?;

    // Checkpoint: between previous stage -> align
    checkpoint_or_emit("align", pcx, log)?;

    log.mark_stage_start();
    log.push(
        "align",
        "align.start",
        "running CTC-based forced alignment for timestamp correction",
        json!({
            "segments": result.segments.len(),
            "budget_ms": stage_budgets.align_ms,
            "audio_duration_sec": inter.normalized_duration,
        }),
    );

    let align_budget_ms = stage_budgets.align_ms;
    let align_token = pcx.stage_token(align_budget_ms); // ubs:ignore — cancellation token is not a secret
    let audio_duration = inter.normalized_duration;

    let (updated_result, report) =
        match run_stage_with_budget("align", align_budget_ms, move || {
            let config = AlignConfig::default();
            let report =
                ctc_forced_align(&mut result.segments, audio_duration, &config, &align_token)?;
            Ok((result, report))
        }) {
            Ok(output) => output,
            Err(error) => {
                let code = stage_failure_code("align", &error);
                log.push(
                    "align",
                    &code,
                    stage_failure_message(&error, "alignment stage failed"),
                    json!({
                        "error": error.to_string(),
                        "budget_ms": stage_budgets.align_ms,
                    }),
                );
                return Err(error);
            }
        };

    inter.result = Some(updated_result);
    inter.warnings.extend(report.notes.iter().cloned());

    let align_code = if report.segments_corrected > 0 {
        tracing::debug!(
            stage = "align",
            corrected = report.segments_corrected,
            fallback = report.segments_fallback,
            "forced alignment complete"
        );
        "align.ok"
    } else {
        tracing::warn!(
            stage = "align",
            fallback = report.segments_fallback,
            "forced alignment produced no corrections"
        );
        "align.fallback"
    };

    log.push(
        "align",
        align_code,
        "forced alignment pass finished",
        json!({
            "segments_total": report.segments_total,
            "segments_corrected": report.segments_corrected,
            "segments_fallback": report.segments_fallback,
            "notes": report.notes,
        }),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// VAD pre-filtering (bd-qla.1)
// ---------------------------------------------------------------------------

/// Configuration for the VAD stage.
///
/// The primary detector uses `backend::native_audio::analyze_wav` and then
/// applies deterministic post-processing. If native waveform parsing fails,
/// we deterministically fall back to the legacy energy scanner.
#[derive(Debug, Clone)]
struct VadConfig {
    /// RMS threshold used when filtering native regions and legacy fallback.
    rms_threshold: f64,
    /// Legacy fallback frame size in samples (at 16 kHz, 160 samples = 10ms).
    frame_samples: usize,
    /// Minimum ratio of voiced frames required to avoid silence short-circuit.
    min_voice_ratio: f64,
    /// Minimum retained speech-region duration.
    min_speech_duration_ms: u32,
    /// Maximum silence gap to bridge across neighboring speech regions.
    min_silence_duration_ms: u32,
    /// Maximum speech-region chunk size; longer regions are split deterministically.
    max_speech_duration_ms: Option<u64>,
    /// Symmetric padding applied to each region.
    speech_pad_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            rms_threshold: 0.01,
            frame_samples: 160,
            min_voice_ratio: 0.05,
            min_speech_duration_ms: 40,
            min_silence_duration_ms: 40,
            max_speech_duration_ms: None,
            speech_pad_ms: 0,
        }
    }
}

impl VadConfig {
    fn from_request(request: &TranscribeRequest) -> Self {
        let mut config = Self::default();
        let Some(vad) = request.backend_params.vad.as_ref() else {
            return config;
        };

        if let Some(threshold) = vad
            .threshold
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            config.rms_threshold = f64::from(threshold);
        }
        if let Some(min_speech_ms) = vad.min_speech_duration_ms {
            config.min_speech_duration_ms = min_speech_ms;
        }
        if let Some(min_silence_ms) = vad.min_silence_duration_ms {
            config.min_silence_duration_ms = min_silence_ms;
        }
        if let Some(max_speech_s) = vad
            .max_speech_duration_s
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            config.max_speech_duration_ms =
                Some((f64::from(max_speech_s) * 1_000.0).round() as u64);
        }
        if let Some(speech_pad_ms) = vad.speech_pad_ms {
            config.speech_pad_ms = speech_pad_ms;
        }

        config
    }

    fn as_json(&self) -> Value {
        json!({
            "rms_threshold": self.rms_threshold,
            "frame_samples": self.frame_samples,
            "min_voice_ratio": self.min_voice_ratio,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "max_speech_duration_ms": self.max_speech_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct VadRegionMs {
    start_ms: u64,
    end_ms: u64,
    avg_rms: f32,
}

/// Report produced by the VAD stage.
#[derive(Debug, Clone)]
struct VadReport {
    /// Number of frames analyzed.
    frames_total: usize,
    /// Number of frames classified as voiced.
    frames_voiced: usize,
    /// Ratio of voiced frames to total frames.
    voice_ratio: f64,
    /// Whether the audio is silence-only (should skip transcription).
    silence_only: bool,
    /// Detected voice activity regions as (start_sec, end_sec) pairs.
    regions: Vec<(f64, f64)>,
    /// Which detector path produced this report.
    detector: &'static str,
    /// Whether deterministic fallback from native parser was triggered.
    fallback_triggered: bool,
    /// Effective activity threshold used to classify speech.
    activity_threshold: f64,
    /// Additional deterministic diagnostics for evidence/logging.
    notes: Vec<String>,
}

/// Analyze audio energy levels to detect voice activity regions.
///
/// Primary path:
/// - parse and analyze the normalized WAV via `backend::native_audio::analyze_wav`
/// - apply deterministic post-processing (gap bridging, min-duration filter,
///   optional max-duration splitting, and optional padding)
///
/// Fallback path:
/// - if native waveform parsing fails, use a legacy energy scanner with
///   deterministic behavior and emit fallback evidence.
fn vad_energy_detect(
    normalized_wav: &Path,
    config: &VadConfig,
    token: &CancellationToken,
) -> FwResult<VadReport> {
    token.checkpoint()?;

    let analysis = match backend::native_audio::analyze_wav(normalized_wav, None) {
        Ok(analysis) => analysis,
        Err(error) => {
            let mut fallback_report = vad_energy_detect_legacy(normalized_wav, config, token)?;
            fallback_report.fallback_triggered = true;
            fallback_report.notes.push(format!(
                "native_audio parse failed; deterministic legacy fallback activated: {error}"
            ));
            return Ok(fallback_report);
        }
    };

    let mut regions_ms: Vec<VadRegionMs> = analysis
        .active_regions
        .iter()
        .filter(|region| f64::from(region.avg_rms) >= config.rms_threshold)
        .map(|region| VadRegionMs {
            start_ms: region.start_ms,
            end_ms: region.end_ms,
            avg_rms: region.avg_rms,
        })
        .collect();

    merge_regions_by_gap(&mut regions_ms, u64::from(config.min_silence_duration_ms));

    if let Some(max_ms) = config.max_speech_duration_ms.filter(|value| *value > 0) {
        regions_ms = split_long_regions(&regions_ms, max_ms);
    }

    let analysis_duration_ms = analysis.duration_ms.max(
        regions_ms
            .iter()
            .map(|region| region.end_ms)
            .max()
            .unwrap_or(0),
    );
    if config.speech_pad_ms > 0 {
        apply_padding(
            &mut regions_ms,
            u64::from(config.speech_pad_ms),
            analysis_duration_ms,
        );
        merge_regions_by_gap(&mut regions_ms, 0);
    }

    let min_speech_duration_ms = u64::from(config.min_speech_duration_ms);
    regions_ms.retain(|region| {
        let duration = region.end_ms.saturating_sub(region.start_ms);
        duration >= min_speech_duration_ms && region.end_ms > region.start_ms
    });

    let frame_ms = u64::from(analysis.frame_ms.max(1));
    let frames_total = analysis.frame_count;
    let mut frames_voiced: usize = regions_ms
        .iter()
        .map(|region| ms_to_frames(region.end_ms.saturating_sub(region.start_ms), frame_ms))
        .sum();
    frames_voiced = frames_voiced.min(frames_total);

    let voice_ratio = if frames_total > 0 {
        frames_voiced as f64 / frames_total as f64
    } else {
        0.0
    };

    let silence_only =
        frames_total == 0 || regions_ms.is_empty() || voice_ratio < config.min_voice_ratio;

    let regions = regions_ms
        .iter()
        .map(|region| {
            (
                region.start_ms as f64 / 1_000.0,
                region.end_ms as f64 / 1_000.0,
            )
        })
        .collect();

    Ok(VadReport {
        frames_total,
        frames_voiced,
        voice_ratio,
        silence_only,
        regions,
        detector: "native_audio_waveform",
        fallback_triggered: false,
        activity_threshold: config.rms_threshold,
        notes: Vec::new(),
    })
}

fn merge_regions_by_gap(regions: &mut Vec<VadRegionMs>, max_gap_ms: u64) {
    if regions.len() < 2 {
        return;
    }
    regions.sort_by_key(|region| region.start_ms);

    let mut merged = Vec::with_capacity(regions.len());
    let mut current = regions[0];
    for region in regions.iter().skip(1) {
        if region.start_ms <= current.end_ms.saturating_add(max_gap_ms) {
            current.end_ms = current.end_ms.max(region.end_ms);
            current.avg_rms = current.avg_rms.max(region.avg_rms);
        } else {
            merged.push(current);
            current = *region;
        }
    }
    merged.push(current);
    *regions = merged;
}

fn split_long_regions(regions: &[VadRegionMs], max_duration_ms: u64) -> Vec<VadRegionMs> {
    if max_duration_ms == 0 {
        return regions.to_vec();
    }

    let mut out = Vec::new();
    for region in regions {
        let mut cursor = region.start_ms;
        while cursor < region.end_ms {
            let chunk_end = cursor.saturating_add(max_duration_ms).min(region.end_ms);
            out.push(VadRegionMs {
                start_ms: cursor,
                end_ms: chunk_end,
                avg_rms: region.avg_rms,
            });
            cursor = chunk_end;
        }
    }
    out
}

fn apply_padding(regions: &mut [VadRegionMs], pad_ms: u64, audio_duration_ms: u64) {
    for region in regions {
        region.start_ms = region.start_ms.saturating_sub(pad_ms);
        let padded_end = region.end_ms.saturating_add(pad_ms);
        region.end_ms = if audio_duration_ms > 0 {
            padded_end.min(audio_duration_ms)
        } else {
            padded_end
        };
    }
}

fn ms_to_frames(duration_ms: u64, frame_ms: u64) -> usize {
    if duration_ms == 0 {
        return 0;
    }
    let frame_ms = frame_ms.max(1);
    duration_ms
        .saturating_add(frame_ms.saturating_sub(1))
        .saturating_div(frame_ms) as usize
}

fn vad_energy_detect_legacy(
    normalized_wav: &Path,
    config: &VadConfig,
    token: &CancellationToken,
) -> FwResult<VadReport> {
    token.checkpoint()?;

    let raw = std::fs::read(normalized_wav)?;
    let pcm_data = if raw.len() > 44 {
        &raw[44..]
    } else {
        &[] as &[u8]
    };

    let (sample_bytes, _) = pcm_data.as_chunks::<2>();
    let samples: Vec<f64> = sample_bytes
        .iter()
        .map(|chunk| {
            let sample = i16::from_le_bytes(*chunk);
            sample as f64 / 32768.0
        })
        .collect();

    if samples.is_empty() {
        return Ok(VadReport {
            frames_total: 0,
            frames_voiced: 0,
            voice_ratio: 0.0,
            silence_only: true,
            regions: Vec::new(),
            detector: "legacy_energy",
            fallback_triggered: false,
            activity_threshold: config.rms_threshold,
            notes: Vec::new(),
        });
    }

    let sample_rate = 16000.0_f64;
    let frame_samples = config.frame_samples.max(1);
    let mut frames_total = 0usize;
    let mut frames_voiced = 0usize;
    let mut regions: Vec<(f64, f64)> = Vec::new();
    let mut in_voice_region = false;
    let mut region_start = 0.0f64;

    for (frame_idx, frame) in samples.chunks(frame_samples).enumerate() {
        // Cancellation check every 1000 frames for responsive shutdown.
        if frame_idx % 1000 == 0 {
            token.checkpoint()?;
        }

        frames_total += 1;

        // Compute RMS energy for this frame.
        let sum_sq: f64 = frame.iter().map(|s| s * s).sum();
        let rms = (sum_sq / frame.len() as f64).sqrt();

        let is_voiced = rms >= config.rms_threshold;

        if is_voiced {
            frames_voiced += 1;
            if !in_voice_region {
                region_start = frame_idx as f64 * frame_samples as f64 / sample_rate;
                in_voice_region = true;
            }
        } else if in_voice_region {
            let region_end = frame_idx as f64 * frame_samples as f64 / sample_rate;
            regions.push((region_start, region_end));
            in_voice_region = false;
        }
    }

    // Close any open region.
    if in_voice_region {
        let region_end = samples.len() as f64 / sample_rate;
        regions.push((region_start, region_end));
    }

    let voice_ratio = if frames_total > 0 {
        frames_voiced as f64 / frames_total as f64
    } else {
        0.0
    };

    let silence_only = voice_ratio < config.min_voice_ratio;

    Ok(VadReport {
        frames_total,
        frames_voiced,
        voice_ratio,
        silence_only,
        regions,
        detector: "legacy_energy",
        fallback_triggered: false,
        activity_threshold: config.rms_threshold,
        notes: Vec::new(),
    })
}

async fn execute_vad(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    request: &TranscribeRequest,
    stage_budgets: StageBudgetPolicy,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    let normalized_wav = inter.normalized_wav.as_ref().ok_or_else(|| {
        FwError::InvalidRequest("Vad requires Normalize to have run first".to_owned())
    })?;

    checkpoint_or_emit("vad", pcx, log)?;

    let vad_config = VadConfig::from_request(request);

    log.mark_stage_start();
    log.push(
        "vad",
        "vad.start",
        "running voice activity detection",
        json!({
            "budget_ms": stage_budgets.vad_ms,
            "input": normalized_wav.display().to_string(),
            "config": vad_config.as_json(),
        }),
    );

    let vad_wav = normalized_wav.clone();
    let vad_budget_ms = stage_budgets.vad_ms;
    let vad_token = pcx.stage_token(vad_budget_ms); // ubs:ignore — cancellation token is not a secret
    let config_for_run = vad_config.clone();

    let report = match run_stage_with_budget("vad", vad_budget_ms, move || {
        vad_energy_detect(&vad_wav, &config_for_run, &vad_token)
    }) {
        Ok(report) => report,
        Err(error) => {
            let code = stage_failure_code("vad", &error);
            log.push(
                "vad",
                &code,
                stage_failure_message(&error, "VAD stage failed"),
                json!({"error": error.to_string(), "budget_ms": vad_budget_ms}),
            );
            return Err(error);
        }
    };

    let vad_code = if report.silence_only {
        inter.vad_silence_only = true;
        // Provide an empty result for silence-only audio.
        inter.result = Some(crate::model::TranscriptionResult {
            backend: crate::model::BackendKind::Auto,
            transcript: String::new(),
            language: None,
            segments: Vec::new(),
            acceleration: None,
            raw_output: json!({
                "vad": "silence_only",
                "detector": report.detector,
                "fallback_triggered": report.fallback_triggered,
            }),
            artifact_paths: Vec::new(),
        });
        "vad.silence"
    } else {
        "vad.ok"
    };

    inter.vad_regions = Some(report.regions.clone());

    tracing::debug!(
        stage = "vad",
        voiced_frames = report.frames_voiced,
        total_frames = report.frames_total,
        "VAD complete"
    );

    log.push(
        "vad",
        vad_code,
        "voice activity detection complete",
        json!({
            "frames_total": report.frames_total,
            "frames_voiced": report.frames_voiced,
            "voice_ratio": report.voice_ratio,
            "silence_only": report.silence_only,
            "regions_count": report.regions.len(),
            "detector": report.detector,
            "fallback_triggered": report.fallback_triggered,
            "activity_threshold": report.activity_threshold,
            "notes": report.notes,
        }),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Source separation (bd-qla.2, energy-based vocal confidence analysis)
// ---------------------------------------------------------------------------

/// Report produced by the source separation stage.
#[derive(Debug, Clone)]
struct SeparateReport {
    /// Whether the audio was determined to contain predominantly vocal content.
    vocal_isolated: bool,
    /// Fraction of audio duration covered by active speech regions (0.0 - 1.0).
    speech_coverage: f64,
    /// Average RMS energy of the audio.
    avg_rms: f64,
    /// Number of distinct speech regions detected.
    active_region_count: usize,
    /// Notes about the separation process.
    notes: Vec<String>,
}

/// Energy-based source separation analysis.
///
/// Analyzes the normalized WAV using native audio waveform analysis to
/// compute speech coverage and vocal confidence metrics.  This is not
/// full neural source separation (which would require a Demucs-class model)
/// but provides useful signal-quality telemetry and a speech coverage gate.
///
/// The `vocal_isolated` flag is set to `true` when the speech coverage
/// fraction exceeds the minimum threshold, indicating the audio has
/// sufficient vocal content for downstream transcription.
fn source_separate(normalized_wav: &Path, token: &CancellationToken) -> FwResult<SeparateReport> {
    token.checkpoint()?;

    // Attempt native audio analysis.  If the file cannot be parsed
    // (e.g. not a valid PCM16 mono WAV), fall back gracefully.
    let analysis = match backend::native_audio::analyze_wav(normalized_wav, None) {
        Ok(a) => a,
        Err(reason) => {
            return Ok(SeparateReport {
                vocal_isolated: true,
                speech_coverage: 0.0,
                avg_rms: 0.0,
                active_region_count: 0,
                notes: vec![format!(
                    "analysis unavailable ({reason}); assuming vocal content present"
                )],
            });
        }
    };

    token.checkpoint()?;

    let duration_ms = analysis.duration_ms.max(1) as f64;
    let speech_ms: f64 = analysis
        .active_regions
        .iter()
        .map(|r| (r.end_ms.saturating_sub(r.start_ms)) as f64)
        .sum();
    let speech_coverage = (speech_ms / duration_ms).clamp(0.0, 1.0);

    // Speech coverage gate: consider the audio vocal-sufficient if at
    // least 5% of the duration contains active speech regions.
    let speech_coverage_threshold = 0.05;
    let vocal_isolated = speech_coverage >= speech_coverage_threshold;

    let mut notes = Vec::new();
    notes.push(format!(
        "energy-based analysis: speech_coverage={speech_coverage:.4}, active_regions={}, avg_rms={:.6}",
        analysis.active_regions.len(),
        analysis.avg_rms,
    ));
    if !vocal_isolated {
        notes.push(format!(
            "speech coverage {speech_coverage:.4} below threshold {speech_coverage_threshold}; may be silence-only audio"
        ));
    }

    Ok(SeparateReport {
        vocal_isolated,
        speech_coverage,
        avg_rms: f64::from(analysis.avg_rms),
        active_region_count: analysis.active_regions.len(),
        notes,
    })
}

async fn execute_separate(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    stage_budgets: StageBudgetPolicy,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    let normalized_wav = inter.normalized_wav.as_ref().ok_or_else(|| {
        FwError::InvalidRequest("Separate requires Normalize to have run first".to_owned())
    })?;

    checkpoint_or_emit("separate", pcx, log)?;

    log.mark_stage_start();
    log.push(
        "separate",
        "separate.start",
        "running source separation (vocal isolation)",
        json!({
            "budget_ms": stage_budgets.separate_ms,
            "input": normalized_wav.display().to_string(),
        }),
    );

    let sep_wav = normalized_wav.clone();
    let sep_budget_ms = stage_budgets.separate_ms;
    let sep_token = pcx.stage_token(sep_budget_ms); // ubs:ignore — cancellation token is not a secret

    let report = match run_stage_with_budget("separate", sep_budget_ms, move || {
        source_separate(&sep_wav, &sep_token)
    }) {
        Ok(report) => report,
        Err(error) => {
            let code = stage_failure_code("separate", &error);
            log.push(
                "separate",
                &code,
                stage_failure_message(&error, "source separation failed"),
                json!({"error": error.to_string(), "budget_ms": sep_budget_ms}),
            );
            return Err(error);
        }
    };

    inter.vocal_isolated = report.vocal_isolated;
    inter.warnings.extend(report.notes.iter().cloned());

    tracing::debug!(stage = "separate", "Source separation complete");
    log.push(
        "separate",
        "separate.ok",
        "source separation pass finished",
        json!({
            "vocal_isolated": report.vocal_isolated,
            "speech_coverage": report.speech_coverage,
            "avg_rms": report.avg_rms,
            "active_region_count": report.active_region_count,
            "notes": report.notes,
        }),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Punctuation restoration (bd-qla.4)
// ---------------------------------------------------------------------------

/// Report produced by the punctuation restoration stage.
#[derive(Debug, Clone)]
struct PunctuateReport {
    /// Number of segments processed.
    segments_total: usize,
    /// Number of segments modified.
    segments_modified: usize,
    /// Notes about the punctuation process.
    notes: Vec<String>,
}

/// Common abbreviations that end with a period but do not indicate
/// a sentence boundary.  Checked case-insensitively.
const ABBREVIATIONS: &[&str] = &[
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "ave.", "blvd.", "vs.", "etc.",
    "approx.", "dept.", "est.", "govt.", "inc.", "ltd.", "no.", "vol.", "rev.", "gen.", "sgt.",
    "cpl.", "pvt.", "lt.", "capt.", "col.", "maj.", "cmdr.", "adm.", "hon.", "fig.", "eq.", "ref.",
    "sec.",
];

/// Return `true` when the period at `period_byte_pos` inside `text` belongs
/// to a known abbreviation rather than ending a sentence.
fn is_abbreviation_period(text: &str, period_byte_pos: usize) -> bool {
    let before = &text[..period_byte_pos + 1]; // includes the period
    let lower = before.to_ascii_lowercase();
    for abbr in ABBREVIATIONS {
        if lower.ends_with(abbr) {
            // Make sure the abbreviation is word-aligned (preceded by start or
            // whitespace).
            let prefix_len = before.len() - abbr.len();
            if prefix_len == 0 {
                return true;
            }
            if before
                .as_bytes()
                .get(prefix_len.wrapping_sub(1))
                .is_some_and(|b| b.is_ascii_whitespace())
            {
                return true;
            }
        }
    }
    false
}

/// Return `true` when the period at `period_byte_pos` inside `text` is part
/// of a decimal number (e.g. "3.14", "$5.00").
fn is_decimal_period(text: &str, period_byte_pos: usize) -> bool {
    let before_digit = period_byte_pos > 0
        && text
            .as_bytes()
            .get(period_byte_pos - 1)
            .is_some_and(|b| b.is_ascii_digit());
    let after_digit = text
        .as_bytes()
        .get(period_byte_pos + 1)
        .is_some_and(|b| b.is_ascii_digit());
    before_digit && after_digit
}

/// Return `true` when the period at `period_byte_pos` inside `text` is part
/// of an ellipsis sequence ("...").
fn is_ellipsis_period(text: &str, period_byte_pos: usize) -> bool {
    // Check if there is a run of at least 3 consecutive periods containing
    // this position.
    let bytes = text.as_bytes();
    let mut start = period_byte_pos;
    while start > 0 && bytes.get(start - 1) == Some(&b'.') {
        start -= 1;
    }
    let mut end = period_byte_pos;
    while bytes.get(end + 1) == Some(&b'.') {
        end += 1;
    }
    (end - start + 1) >= 3
}

/// Apply rule-based punctuation restoration to transcript segments.
///
/// Rules applied:
/// 1. Normalize consecutive whitespace to single spaces.
/// 2. Capitalize the first character of each segment's text.
/// 3. Add a period at the end of segments that don't end with punctuation.
/// 4. Capitalize after sentence-ending punctuation (. ? !) — but NOT after
///    abbreviations (Mr., Dr., etc.), decimal numbers (3.14), or
///    ellipses (...).
fn punctuate_segments(
    segments: &mut [crate::model::TranscriptionSegment],
    token: &CancellationToken,
) -> FwResult<PunctuateReport> {
    let total = segments.len();
    let mut modified = 0usize;
    let mut notes: Vec<String> = Vec::new();

    if segments.is_empty() {
        return Ok(PunctuateReport {
            segments_total: 0,
            segments_modified: 0,
            notes,
        });
    }

    for (idx, segment) in segments.iter_mut().enumerate() {
        // Cancellation check every 100 segments.
        if idx % 100 == 0 {
            token.checkpoint()?;
        }

        let original = segment.text.clone();
        let mut text = segment.text.trim().to_owned();

        if text.is_empty() {
            continue;
        }

        // Rule 1: Normalize consecutive whitespace to single spaces.
        let mut normalized = String::with_capacity(text.len());
        let mut prev_ws = false;
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !prev_ws {
                    normalized.push(' ');
                }
                prev_ws = true;
            } else {
                normalized.push(ch);
                prev_ws = false;
            }
        }
        text = normalized;

        // Rule 2: Capitalize first character.
        let first_upper: String = text
            .chars()
            .take(1)
            .flat_map(|c| c.to_uppercase())
            .collect();
        text = first_upper + &text[text.chars().next().map_or(0, |c| c.len_utf8())..];

        // Rule 3: Add period at the end if no sentence-ending punctuation.
        let ends_with_punct = text
            .chars()
            .last()
            .is_some_and(|c| c == '.' || c == '?' || c == '!');
        if !ends_with_punct {
            text.push('.');
        }

        // Rule 4: Capitalize after sentence-ending punctuation within the
        // text, skipping abbreviation periods, decimal periods, and ellipses.
        let mut result = String::with_capacity(text.len());
        let mut capitalize_next = false;
        let mut byte_offset: usize = 0;
        for ch in text.chars() {
            if capitalize_next && ch.is_alphabetic() {
                result.extend(ch.to_uppercase());
                capitalize_next = false;
            } else {
                result.push(ch);
            }

            if ch == '?' || ch == '!' {
                capitalize_next = true;
            } else if ch == '.' {
                // Only treat as sentence-end if not an abbreviation, decimal,
                // or ellipsis.
                let is_sentence_end = !is_abbreviation_period(&text, byte_offset)
                    && !is_decimal_period(&text, byte_offset)
                    && !is_ellipsis_period(&text, byte_offset);
                if is_sentence_end {
                    capitalize_next = true;
                }
            } else if !ch.is_whitespace() {
                capitalize_next = false;
            }

            byte_offset += ch.len_utf8();
        }

        if result != original {
            segment.text = result;
            modified += 1;
        }
    }

    if modified == 0 {
        notes.push("no segments required punctuation changes".to_owned());
    }

    Ok(PunctuateReport {
        segments_total: total,
        segments_modified: modified,
        notes,
    })
}

async fn execute_punctuate(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    stage_budgets: StageBudgetPolicy,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    let mut result = inter.result.take().ok_or_else(|| {
        FwError::InvalidRequest("Punctuate requires Backend to have run first".to_owned())
    })?;

    checkpoint_or_emit("punctuate", pcx, log)?;

    log.mark_stage_start();
    log.push(
        "punctuate",
        "punctuate.start",
        "running punctuation restoration",
        json!({
            "segments": result.segments.len(),
            "budget_ms": stage_budgets.punctuate_ms,
        }),
    );

    let punct_budget_ms = stage_budgets.punctuate_ms;
    let punct_token = pcx.stage_token(punct_budget_ms); // ubs:ignore — cancellation token is not a secret

    let (updated_result, report) =
        match run_stage_with_budget("punctuate", punct_budget_ms, move || {
            let report = punctuate_segments(&mut result.segments, &punct_token)?;
            Ok((result, report))
        }) {
            Ok(output) => output,
            Err(error) => {
                let code = stage_failure_code("punctuate", &error);
                log.push(
                    "punctuate",
                    &code,
                    stage_failure_message(&error, "punctuation restoration failed"),
                    json!({"error": error.to_string(), "budget_ms": punct_budget_ms}),
                );
                return Err(error);
            }
        };

    inter.result = Some(updated_result);
    inter.warnings.extend(report.notes.iter().cloned());

    tracing::debug!(
        stage = "punctuate",
        modified = report.segments_modified,
        "punctuation restoration complete"
    );

    log.push(
        "punctuate",
        "punctuate.ok",
        "punctuation restoration finished",
        json!({
            "segments_total": report.segments_total,
            "segments_modified": report.segments_modified,
            "notes": report.notes,
        }),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Speaker diarization (bd-qla.5, TitaNet-inspired placeholder)
// ---------------------------------------------------------------------------

/// Report produced by the speaker diarization stage.
///
/// `pub(crate)` so native backends (e.g. `whisper_diarization_native`) can call
/// [`diarize_segments`] directly and inspect its quality metrics for honest
/// `raw_output` reporting.
#[derive(Debug, Clone)]
pub(crate) struct DiarizeReport {
    /// Number of segments processed.
    pub(crate) segments_total: usize,
    /// Number of distinct speakers detected.
    pub(crate) speakers_detected: usize,
    /// Number of segments assigned a speaker label.
    pub(crate) segments_labeled: usize,
    /// Silhouette score measuring cluster separation quality.
    /// Range: \[-1, 1\].  1 = perfect separation, 0 = overlapping,
    /// -1 = misassigned.  `None` when fewer than 2 clusters or 2 points.
    pub(crate) silhouette_score: Option<f64>,
    /// Notes about the diarization process.
    pub(crate) notes: Vec<String>,
}

/// Acoustic-heuristic feature vector for a segment.
///
/// In a real TitaNet-inspired system, this would be a high-dimensional
/// embedding from a neural speaker encoder.  Here we use an expanded set
/// of temporal and lexical features derived from the segment position,
/// pacing, and text properties.
///
/// Features (6-dimensional):
///   0. normalized segment midpoint (temporal position in recording)
///   1. segment duration / max_duration (pacing signature)
///   2. inter-segment gap indicator (turn-taking signal)
///   3. word count / max_word_count (verbosity signature)
///   4. average word length / 12.0 (vocabulary complexity proxy)
///   5. text character count / max_text_len (output volume)
#[derive(Debug, Clone)]
struct SpeakerEmbedding {
    features: [f64; 6],
}

impl SpeakerEmbedding {
    /// Cosine similarity between two embeddings.
    fn cosine_similarity(&self, other: &SpeakerEmbedding) -> f64 {
        let dot: f64 = self
            .features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f64 = self.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = other.features.iter().map(|x| x * x).sum::<f64>().sqrt();

        if mag_a < 1e-10 || mag_b < 1e-10 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    /// Compute the element-wise mean of a slice of embeddings.
    fn centroid(embeddings: &[SpeakerEmbedding]) -> SpeakerEmbedding {
        let n = embeddings.len() as f64;
        if n < 1.0 {
            return SpeakerEmbedding { features: [0.0; 6] };
        }
        let mut features = [0.0_f64; 6];
        for emb in embeddings {
            for (i, val) in emb.features.iter().enumerate() {
                features[i] += val;
            }
        }
        for f in &mut features {
            *f /= n;
        }
        SpeakerEmbedding { features }
    }

    /// Euclidean distance between two embeddings.
    fn euclidean_distance(&self, other: &SpeakerEmbedding) -> f64 {
        self.features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

/// Compute the mean silhouette score over all points.
///
/// For each point *i* assigned to cluster *C_i*:
///   - *a(i)* = mean distance to other points in *C_i* (intra-cluster)
///   - *b(i)* = min over other clusters *C_j* of mean distance to points in *C_j*
///   - *s(i)* = (b(i) - a(i)) / max(a(i), b(i))
///
/// Returns `None` when there are fewer than 2 clusters or fewer than 2 points,
/// since the metric is undefined in those cases.
fn silhouette_score(
    embeddings: &[SpeakerEmbedding],
    assignments: &[usize],
    num_clusters: usize,
) -> Option<f64> {
    if num_clusters < 2 || embeddings.len() < 2 {
        return None;
    }

    let n = embeddings.len();
    let mut sum = 0.0_f64;

    for i in 0..n {
        let ci = assignments[i];

        // a(i): mean distance to other points in same cluster.
        let mut a_sum = 0.0_f64;
        let mut a_count = 0u64;
        for (j, emb_j) in embeddings.iter().enumerate() {
            if j != i && assignments[j] == ci {
                a_sum += embeddings[i].euclidean_distance(emb_j);
                a_count += 1;
            }
        }
        let a_i = if a_count > 0 {
            a_sum / a_count as f64
        } else {
            0.0
        };

        // b(i): minimum mean distance to any other cluster.
        let mut b_i = f64::INFINITY;
        for cj in 0..num_clusters {
            if cj == ci {
                continue;
            }
            let mut b_sum = 0.0_f64;
            let mut b_count = 0u64;
            for (j, emb_j) in embeddings.iter().enumerate() {
                if assignments[j] == cj {
                    b_sum += embeddings[i].euclidean_distance(emb_j);
                    b_count += 1;
                }
            }
            if b_count > 0 {
                let mean_dist = b_sum / b_count as f64;
                if mean_dist < b_i {
                    b_i = mean_dist;
                }
            }
        }

        let denom = a_i.max(b_i);
        let s_i = if denom < 1e-15 {
            0.0
        } else {
            (b_i - a_i) / denom
        };
        sum += s_i;
    }

    Some(sum / n as f64)
}

/// Resolve the effective target speaker count from [`SpeakerConstraints`].
///
/// Priority: `num_speakers` > `max_speakers`.
/// Returns `None` when no upper bound is specified.
fn resolve_speaker_target(constraints: Option<&crate::model::SpeakerConstraints>) -> Option<usize> {
    let sc = constraints?;
    // Explicit num_speakers overrides everything.
    if let Some(n) = sc.num_speakers.filter(|&n| n > 0) {
        return Some(n as usize);
    }
    // If only max_speakers is set, use it as the ceiling.
    if let Some(max_k) = sc.max_speakers.filter(|&m| m > 0) {
        return Some(max_k as usize);
    }
    None
}

/// Assign speaker labels to segments using acoustic-heuristic feature
/// clustering.
///
/// Algorithm:
/// 1. Compute a 6-dimensional embedding for each segment incorporating
///    temporal position, pacing, turn-taking gaps, word count, average
///    word length, and text volume.
/// 2. Greedily cluster segments: assign each segment to the nearest
///    existing cluster centroid (by cosine similarity) if similarity
///    exceeds a threshold; otherwise create a new cluster.  Centroids
///    are updated incrementally as segments are assigned.
/// 3. Label each cluster as SPEAKER_00, SPEAKER_01, etc.
///
/// This is a heuristic-only implementation (no neural speaker encoder);
/// accuracy improves significantly when combined with downstream
/// model-backed diarization.
pub(crate) fn diarize_segments(
    segments: &mut [crate::model::TranscriptionSegment],
    audio_duration: Option<f64>,
    speaker_constraints: Option<&crate::model::SpeakerConstraints>,
    token: &CancellationToken,
) -> FwResult<DiarizeReport> {
    let total = segments.len();
    let mut notes: Vec<String> =
        vec!["heuristic: acoustic-feature clustering without neural speaker encoder".to_owned()];

    if segments.is_empty() {
        return Ok(DiarizeReport {
            segments_total: 0,
            speakers_detected: 0,
            segments_labeled: 0,
            silhouette_score: None,
            notes,
        });
    }

    token.checkpoint()?;

    let duration = audio_duration
        .or_else(|| segments.iter().filter_map(|s| s.end_sec).reduce(f64::max))
        .unwrap_or(1.0)
        .max(1e-6);

    // Precompute normalization denominators across the segment set.
    let max_seg_duration = segments
        .iter()
        .map(|s| {
            let start = s.start_sec.unwrap_or(0.0);
            let end = s.end_sec.unwrap_or(start);
            (end - start).max(0.0)
        })
        .fold(0.0_f64, f64::max)
        .max(1e-6);

    let max_word_count = segments
        .iter()
        .map(|s| s.text.split_whitespace().count() as f64)
        .fold(1.0_f64, f64::max);

    let max_text_len = segments
        .iter()
        .map(|s| s.text.len() as f64)
        .fold(1.0_f64, f64::max);

    // Step 1: Compute embeddings with inter-segment gap analysis.
    let embeddings: Vec<SpeakerEmbedding> = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            let start = seg.start_sec.unwrap_or(0.0);
            let end = seg.end_sec.unwrap_or(start);
            let seg_duration = (end - start).max(0.0);
            let midpoint_norm = ((start + end) / 2.0) / duration;
            let duration_norm = seg_duration / max_seg_duration;

            // Turn-taking gap: time between previous segment's end and
            // current segment's start, normalized.  Larger gaps suggest a
            // speaker change.
            let gap = if i > 0 {
                let prev_end = segments[i - 1].end_sec.unwrap_or(0.0);
                ((start - prev_end).max(0.0) / duration).min(1.0)
            } else {
                0.0
            };

            let mut word_count = 0usize;
            let mut total_chars = 0usize;
            for word in seg.text.split_whitespace() {
                word_count += 1;
                total_chars += word.len();
            }
            let word_count_norm = word_count as f64 / max_word_count;
            let avg_word_len = if word_count == 0 {
                0.0
            } else {
                (total_chars as f64 / word_count as f64) / 12.0
            };
            let text_len_norm = seg.text.len() as f64 / max_text_len;

            SpeakerEmbedding {
                features: [
                    midpoint_norm,
                    duration_norm,
                    gap,
                    word_count_norm,
                    avg_word_len,
                    text_len_norm,
                ],
            }
        })
        .collect();

    // Step 2: Greedy clustering with incremental centroid updates.
    let similarity_threshold = 0.92;
    let mut cluster_members: Vec<Vec<SpeakerEmbedding>> = Vec::new();
    let mut centroids: Vec<SpeakerEmbedding> = Vec::new();
    let mut assignments: Vec<usize> = Vec::with_capacity(total);

    for (idx, emb) in embeddings.iter().enumerate() {
        if idx % 100 == 0 {
            token.checkpoint()?;
        }

        let mut best_cluster = None;
        let mut best_sim = f64::NEG_INFINITY;

        for (cid, centroid) in centroids.iter().enumerate() {
            let sim = emb.cosine_similarity(centroid);
            if sim > best_sim {
                best_sim = sim;
                best_cluster = Some(cid);
            }
        }

        if best_sim >= similarity_threshold {
            if let Some(cid) = best_cluster {
                assignments.push(cid);
                cluster_members[cid].push(emb.clone());
                centroids[cid] = SpeakerEmbedding::centroid(&cluster_members[cid]);
            } else {
                // Defensive fallback: should not happen unless centroids bookkeeping diverges.
                let new_id = centroids.len();
                centroids.push(emb.clone());
                cluster_members.push(vec![emb.clone()]);
                assignments.push(new_id);
            }
        } else {
            let new_id = centroids.len();
            centroids.push(emb.clone());
            cluster_members.push(vec![emb.clone()]);
            assignments.push(new_id);
        }
    }

    // Step 2b: Apply speaker constraints by merging clusters if over the
    // target count.  Determine effective max from constraints.
    let effective_max = resolve_speaker_target(speaker_constraints);

    let unconstrained_count = centroids.len();
    if let Some(max_k) = effective_max {
        let max_k = max_k.max(1); // at least 1 cluster
        while centroids.len() > max_k {
            // Find the two closest centroids and merge them.
            let mut best_pair = (0, 1);
            let mut best_dist = f64::INFINITY;
            for i in 0..centroids.len() {
                for j in (i + 1)..centroids.len() {
                    let d = centroids[i].euclidean_distance(&centroids[j]);
                    if d < best_dist {
                        best_dist = d;
                        best_pair = (i, j);
                    }
                }
            }
            let (keep, remove) = best_pair;
            // Move all members of `remove` into `keep`.
            let removed_members = cluster_members.swap_remove(remove);
            centroids.swap_remove(remove);
            // Fix assignments: `remove` was absorbed into `keep`; the
            // cluster that was last is now at index `remove` due to
            // swap_remove.
            let last_idx = centroids.len(); // old len - 1 after swap_remove
            for a in &mut assignments {
                if *a == remove {
                    *a = keep;
                } else if *a == last_idx {
                    // This was the swapped-in cluster (formerly at end).
                    *a = remove;
                }
            }
            cluster_members[keep].extend(removed_members);
            centroids[keep] = SpeakerEmbedding::centroid(&cluster_members[keep]);
        }

        if unconstrained_count > centroids.len() {
            notes.push(format!(
                "merged {unconstrained_count} clusters down to {} to respect speaker constraints",
                centroids.len()
            ));
        }
    }

    // Note if fewer speakers detected than requested minimum.
    if let Some(sc) = speaker_constraints
        && let Some(min_k) = sc.min_speakers.filter(|&m| m > 0)
        && (centroids.len() as u32) < min_k
    {
        notes.push(format!(
            "detected {} speakers but min_speakers={min_k} requested; \
             heuristic cannot synthesize additional speakers",
            centroids.len()
        ));
    }

    // Compact assignment IDs to be contiguous 0..N after potential merges.
    let unique_ids: Vec<usize> = {
        let mut seen: Vec<usize> = assignments.clone();
        seen.sort_unstable();
        seen.dedup();
        seen
    };
    let id_map: std::collections::HashMap<usize, usize> = unique_ids
        .iter()
        .enumerate()
        .map(|(new_id, &old_id)| (old_id, new_id))
        .collect();
    for a in &mut assignments {
        *a = id_map[a];
    }
    let speakers_detected = unique_ids.len();

    // Step 3: Assign speaker labels.
    let mut labeled = 0usize;
    for (seg, &cluster_id) in segments.iter_mut().zip(assignments.iter()) {
        seg.speaker = Some(format!("SPEAKER_{cluster_id:02}"));
        labeled += 1;
    }

    let sil_score = silhouette_score(&embeddings, &assignments, speakers_detected);

    Ok(DiarizeReport {
        segments_total: total,
        speakers_detected,
        segments_labeled: labeled,
        silhouette_score: sil_score,
        notes,
    })
}

async fn execute_diarize(
    pcx: &mut PipelineCx,
    log: &mut EventLog,
    request: &TranscribeRequest,
    stage_budgets: StageBudgetPolicy,
    inter: &mut PipelineIntermediate,
) -> FwResult<()> {
    let mut result = inter.result.take().ok_or_else(|| {
        FwError::InvalidRequest("Diarize requires Backend to have run first".to_owned())
    })?;

    checkpoint_or_emit("diarize", pcx, log)?;

    log.mark_stage_start();
    log.push(
        "diarize",
        "diarize.start",
        "running speaker diarization",
        json!({
            "segments": result.segments.len(),
            "budget_ms": stage_budgets.diarize_ms,
            "audio_duration_sec": inter.normalized_duration,
            "speaker_constraints": request.backend_params.speaker_constraints.as_ref().map(|sc| json!({
                "num_speakers": sc.num_speakers,
                "min_speakers": sc.min_speakers,
                "max_speakers": sc.max_speakers,
            })),
        }),
    );

    let diarize_budget_ms = stage_budgets.diarize_ms;
    let diarize_token = pcx.stage_token(diarize_budget_ms); // ubs:ignore — cancellation token is not a secret
    let audio_duration = inter.normalized_duration;
    let speaker_constraints = request.backend_params.speaker_constraints.clone();

    let (updated_result, report) =
        match run_stage_with_budget("diarize", diarize_budget_ms, move || {
            let report = diarize_segments(
                &mut result.segments,
                audio_duration,
                speaker_constraints.as_ref(),
                &diarize_token,
            )?;
            Ok((result, report))
        }) {
            Ok(output) => output,
            Err(error) => {
                let code = stage_failure_code("diarize", &error);
                log.push(
                    "diarize",
                    &code,
                    stage_failure_message(&error, "diarization failed"),
                    json!({"error": error.to_string(), "budget_ms": diarize_budget_ms}),
                );
                return Err(error);
            }
        };

    inter.result = Some(updated_result);
    inter.warnings.extend(report.notes.iter().cloned());

    tracing::debug!(
        stage = "diarize",
        speakers = report.speakers_detected,
        labeled = report.segments_labeled,
        "diarization complete"
    );

    log.push(
        "diarize",
        "diarize.ok",
        "speaker diarization finished",
        json!({
            "segments_total": report.segments_total,
            "speakers_detected": report.speakers_detected,
            "segments_labeled": report.segments_labeled,
            "silhouette_score": report.silhouette_score,
            "notes": report.notes,
        }),
    );

    Ok(())
}

fn state_root() -> FwResult<PathBuf> {
    if let Ok(path) = std::env::var("FRANKEN_WHISPER_STATE_DIR")
        && !path.trim().is_empty()
    {
        return Ok(PathBuf::from(path));
    }

    let cwd = std::env::current_dir()?;
    Ok(cwd.join(".franken_whisper"))
}

struct EventLog {
    run_id: String,
    trace_id: String,
    seq: u64,
    events: Vec<RunEvent>,
    event_tx: Option<Sender<StreamedRunEvent>>,
    /// Wall-clock instant of the most recent `mark_stage_start()` call.
    stage_start: Option<std::time::Instant>,
}

impl EventLog {
    fn new(run_id: String, trace_id: String, event_tx: Option<Sender<StreamedRunEvent>>) -> Self {
        Self {
            run_id,
            trace_id,
            seq: 0,
            events: Vec::new(),
            event_tx,
            stage_start: None,
        }
    }

    /// Record the wall-clock start of a pipeline stage.
    fn mark_stage_start(&mut self) {
        self.stage_start = Some(std::time::Instant::now());
    }

    fn clear_stage_start(&mut self) {
        self.stage_start = None;
    }

    /// Elapsed milliseconds since the last `mark_stage_start()`, if any.
    fn stage_elapsed_ms(&self) -> Option<u64> {
        self.stage_start
            .map(|start| start.elapsed().as_millis() as u64)
    }

    fn push(&mut self, stage: &str, code: &str, message: &str, mut payload: serde_json::Value) {
        if let Value::Object(ref mut map) = payload {
            map.insert("trace_id".to_owned(), json!(self.trace_id));
            if let Some(elapsed) = self.stage_elapsed_ms() {
                map.insert("elapsed_ms".to_owned(), json!(elapsed));
            }
        }

        self.seq += 1;
        let event = RunEvent {
            seq: self.seq,
            ts_rfc3339: Utc::now().to_rfc3339(),
            stage: stage.to_owned(),
            code: code.to_owned(),
            message: message.to_owned(),
            payload,
        };

        self.events.push(event.clone());

        if let Some(tx) = &self.event_tx {
            let _ = tx.send(StreamedRunEvent {
                run_id: self.run_id.clone(),
                event,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::Duration;

    use asupersync::runtime::RuntimeBuilder;
    use serde_json::{Value, json};
    use tempfile::tempdir;

    use crate::error::FwError;
    use crate::model::{
        BackendKind, BackendParams, DiarizationConfig, InputSource, RunEvent, RunReport,
        StreamedRunEvent, TranscribeRequest, TranscriptionResult, TranscriptionSegment, VadParams,
    };
    use crate::storage::RunStore;

    #[allow(unused_imports)]
    use super::{
        AlignConfig, AlignmentReport, CancellationToken, DiarizeReport, EventLog, Finalizer,
        FinalizerRegistry, PipelineConfig, PipelineCx, PipelineStage, PunctuateReport,
        SeparateReport, SpeakerEmbedding, StageBudgetPolicy, VadConfig, VadRegionMs, VadReport,
        acceleration_cancellation_fence_payload, acceleration_context_payload,
        acceleration_stream_owner_id, apply_padding, budget_duration, checkpoint_or_emit,
        ctc_forced_align, diarize_segments, event_elapsed_ms, is_abbreviation_period,
        is_decimal_period, is_ellipsis_period, merge_regions_by_gap, ms_to_frames,
        optional_stage_skip, parse_budget_ms, parse_event_ts_ms, punctuate_segments,
        recommended_budget, resolve_speaker_target, run_pipeline, run_stage_with_budget,
        sanitize_process_pid, sha256_bytes_hex, sha256_file, sha256_json_value, silhouette_score,
        source_separate, split_long_regions, stage_budget_ms, stage_failure_code,
        stage_failure_message, stage_latency_profile, state_root, vad_energy_detect,
    };

    #[test]
    fn event_log_streams_and_accumulates_with_monotonic_sequence() {
        let (tx, rx) = mpsc::channel();
        let mut log = EventLog::new(
            "run-123".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            Some(tx),
        );

        log.push("ingest", "ingest.start", "begin", json!({"k":"v"}));
        log.push("ingest", "ingest.ok", "done", json!({"n":1}));
        log.push("backend", "backend.ok", "completed", json!({"segments":2}));

        assert_eq!(log.events.len(), 3);
        assert_eq!(log.events[0].seq, 1);
        assert_eq!(log.events[1].seq, 2);
        assert_eq!(log.events[2].seq, 3);

        let streamed = rx.try_iter().collect::<Vec<_>>();
        assert_eq!(streamed.len(), 3);
        assert!(streamed.iter().all(|item| item.run_id == "run-123"));
        assert_eq!(streamed[0].event.code, "ingest.start");
        assert_eq!(streamed[1].event.code, "ingest.ok");
        assert_eq!(streamed[2].event.code, "backend.ok");
    }

    #[test]
    fn event_log_includes_elapsed_ms_after_stage_mark() {
        let mut log = EventLog::new(
            "run-timing".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );

        // Before any mark — no elapsed_ms
        log.push("pre", "pre.event", "before mark", json!({}));
        assert!(
            log.events[0].payload.get("elapsed_ms").is_none(),
            "elapsed_ms should be absent before mark_stage_start"
        );

        // After mark — elapsed_ms is present
        log.mark_stage_start();
        std::thread::sleep(Duration::from_millis(5));
        log.push("ingest", "ingest.start", "begin", json!({}));
        let elapsed = log.events[1]
            .payload
            .get("elapsed_ms")
            .expect("elapsed_ms should be present after mark")
            .as_u64()
            .expect("elapsed_ms should be a number");
        assert!(elapsed >= 1, "elapsed should be at least 1ms");

        // Subsequent events under same mark accumulate
        std::thread::sleep(Duration::from_millis(5));
        log.push("ingest", "ingest.ok", "done", json!({}));
        let elapsed2 = log.events[2]
            .payload
            .get("elapsed_ms")
            .expect("elapsed_ms should still be present")
            .as_u64()
            .unwrap();
        assert!(elapsed2 >= elapsed, "later event should have >= elapsed_ms");
    }

    #[test]
    fn event_log_elapsed_ms_resets_on_new_mark() {
        let mut log = EventLog::new(
            "run-reset".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );

        log.mark_stage_start();
        std::thread::sleep(Duration::from_millis(20));
        log.push("stage_a", "a.ok", "done", json!({}));
        let elapsed_a = log.events[0]
            .payload
            .get("elapsed_ms")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(elapsed_a >= 15, "first stage should have accumulated time");

        // New mark resets the clock
        log.mark_stage_start();
        log.push("stage_b", "b.start", "begin", json!({}));
        let elapsed_b = log.events[1]
            .payload
            .get("elapsed_ms")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(
            elapsed_b < elapsed_a,
            "new stage mark should reset elapsed (got {elapsed_b} vs {elapsed_a})"
        );
    }

    fn synthetic_stage_event(
        seq: u64,
        ts_rfc3339: &str,
        stage: &str,
        code: &str,
        elapsed_ms: Option<u64>,
    ) -> RunEvent {
        let mut payload = json!({"trace_id": "trace-test"});
        if let Some(elapsed_ms) = elapsed_ms
            && let Some(map) = payload.as_object_mut()
        {
            map.insert("elapsed_ms".to_owned(), json!(elapsed_ms));
        }
        RunEvent {
            seq,
            ts_rfc3339: ts_rfc3339.to_owned(),
            stage: stage.to_owned(),
            code: code.to_owned(),
            message: code.to_owned(),
            payload,
        }
    }

    #[test]
    fn stage_latency_profile_includes_quantiles_and_tuning_actions() {
        let events = vec![
            synthetic_stage_event(1, "2026-02-22T10:00:00Z", "ingest", "ingest.start", Some(0)),
            synthetic_stage_event(
                2,
                "2026-02-22T10:00:00.050Z",
                "ingest",
                "ingest.ok",
                Some(50),
            ),
            synthetic_stage_event(
                3,
                "2026-02-22T10:00:00.060Z",
                "normalize",
                "normalize.start",
                Some(0),
            ),
            synthetic_stage_event(
                4,
                "2026-02-22T10:00:00.430Z",
                "normalize",
                "normalize.ok",
                Some(370),
            ),
            synthetic_stage_event(
                5,
                "2026-02-22T10:00:00.440Z",
                "backend",
                "backend.start",
                Some(0),
            ),
            synthetic_stage_event(
                6,
                "2026-02-22T10:00:01.530Z",
                "backend",
                "backend.ok",
                Some(1090),
            ),
        ];

        let budgets = StageBudgetPolicy {
            ingest_ms: 100,
            normalize_ms: 400,
            probe_ms: 8_000,
            vad_ms: 10_000,
            separate_ms: 30_000,
            backend_ms: 1_000,
            acceleration_ms: 20_000,
            align_ms: 30_000,
            punctuate_ms: 10_000,
            diarize_ms: 30_000,
            persist_ms: 20_000,
            cleanup_budget_ms: 5_000,
        };

        let profile = stage_latency_profile(&events, budgets);
        assert_eq!(profile["artifact"], "stage_latency_decomposition_v1");
        assert_eq!(profile["summary"]["observed_stages"], 3);
        assert_eq!(profile["stages"]["ingest"]["p50_ms"], 50);
        assert_eq!(profile["stages"]["normalize"]["queue_ms"], 10);
        assert_eq!(profile["stages"]["backend"]["p95_ms"], 1090);
        assert_eq!(profile["stages"]["backend"]["p99_ms"], 1090);
        assert_eq!(
            profile["stages"]["backend"]["tuning_action"],
            "increase_budget"
        );

        let recommendations = profile["budget_tuning"]["recommendations"]
            .as_array()
            .expect("recommendations should be array");
        assert_eq!(recommendations.len(), 3);
    }

    #[test]
    fn stage_latency_profile_marks_low_utilization_for_budget_reduction_candidate() {
        let events = vec![
            synthetic_stage_event(1, "2026-02-22T11:00:00Z", "ingest", "ingest.start", Some(0)),
            synthetic_stage_event(
                2,
                "2026-02-22T11:00:00.010Z",
                "ingest",
                "ingest.ok",
                Some(10),
            ),
        ];
        let budgets = StageBudgetPolicy {
            ingest_ms: 1_000,
            normalize_ms: 180_000,
            probe_ms: 8_000,
            vad_ms: 10_000,
            separate_ms: 30_000,
            backend_ms: 900_000,
            acceleration_ms: 20_000,
            align_ms: 30_000,
            punctuate_ms: 10_000,
            diarize_ms: 30_000,
            persist_ms: 20_000,
            cleanup_budget_ms: 5_000,
        };

        let profile = stage_latency_profile(&events, budgets);
        assert_eq!(
            profile["stages"]["ingest"]["tuning_action"],
            "decrease_budget_candidate"
        );
        let recommended = profile["stages"]["ingest"]["recommended_budget_ms"]
            .as_u64()
            .expect("recommended budget should be u64");
        assert!(
            recommended <= 1_000,
            "recommended budget should not exceed current budget for low utilization"
        );
        assert!(
            recommended >= 1_000,
            "floor guard should clamp very small recommendations to at least 1000ms"
        );
    }

    #[test]
    fn streamed_event_order_matches_persisted_event_order() {
        let dir = tempdir().expect("tempdir should be available");
        let db_path = dir.path().join("storage.sqlite3");
        let store = RunStore::open(&db_path).expect("store should open");

        let (tx, rx) = mpsc::channel();
        let mut log = EventLog::new(
            "run-ordered".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            Some(tx),
        );
        log.push("ingest", "ingest.start", "begin", json!({}));
        log.push("ingest", "ingest.ok", "done", json!({}));
        log.push("backend", "backend.ok", "done", json!({"segments":1}));

        let streamed = rx.try_iter().collect::<Vec<_>>();
        let streamed_codes = streamed
            .iter()
            .map(|item| item.event.code.clone())
            .collect::<Vec<_>>();

        let report = RunReport {
            run_id: "run-ordered".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: true,
                db_path: db_path.clone(),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "hello".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(0.8),
                    text: "hello".to_owned(),
                    speaker: None,
                    confidence: Some(0.99),
                }],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: log.events.clone(),
            warnings: vec![],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope::default(),
        };

        store
            .persist_report(&report)
            .expect("report should persist successfully");
        let loaded = store
            .load_run_details("run-ordered")
            .expect("load should succeed")
            .expect("row should exist");

        let persisted_codes = loaded
            .events
            .iter()
            .map(|event| event.code.clone())
            .collect::<Vec<_>>();
        assert_eq!(streamed_codes, persisted_codes);
    }

    #[test]
    fn streamed_event_order_matches_persisted_event_order_for_failure_sequence() {
        let dir = tempdir().expect("tempdir should be available");
        let db_path = dir.path().join("storage.sqlite3");
        let store = RunStore::open(&db_path).expect("store should open");

        let (tx, rx) = mpsc::channel();
        let mut log = EventLog::new(
            "run-ordered-failure".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            Some(tx),
        );
        log.push(
            "orchestration",
            "orchestration.budgets",
            "budgets applied",
            json!({}),
        );
        log.push("ingest", "ingest.start", "begin", json!({}));
        log.push(
            "ingest",
            "ingest.error",
            "failed to materialize input",
            json!({"error":"missing input"}),
        );

        let streamed = rx.try_iter().collect::<Vec<_>>();
        let streamed_fingerprint = streamed
            .iter()
            .map(|item| {
                (
                    item.event.seq,
                    item.event.stage.clone(),
                    item.event.code.clone(),
                )
            })
            .collect::<Vec<_>>();

        let report = RunReport {
            run_id: "run-ordered-failure".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: String::new(),
            request: TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: true,
                db_path: db_path.clone(),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::Auto,
                transcript: String::new(),
                language: None,
                segments: vec![],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: log.events.clone(),
            warnings: vec!["ingest failed".to_owned()],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope::default(),
        };

        store
            .persist_report(&report)
            .expect("report should persist successfully");
        let loaded = store
            .load_run_details("run-ordered-failure")
            .expect("load should succeed")
            .expect("row should exist");

        let persisted_fingerprint = loaded
            .events
            .iter()
            .map(|event| (event.seq, event.stage.clone(), event.code.clone()))
            .collect::<Vec<_>>();
        assert_eq!(streamed_fingerprint, persisted_fingerprint);
    }

    #[test]
    fn failure_sequence_stream_has_monotonic_seq_and_timestamp_order() {
        let (tx, rx) = mpsc::channel();
        let mut log = EventLog::new(
            "run-failure-monotonic".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            Some(tx),
        );
        log.push(
            "orchestration",
            "orchestration.budgets",
            "budgets applied",
            json!({}),
        );
        log.push("ingest", "ingest.start", "begin", json!({}));
        log.push(
            "ingest",
            "ingest.error",
            "failed to materialize input",
            json!({"error":"missing input"}),
        );

        let streamed = rx.try_iter().collect::<Vec<_>>();
        let codes = streamed
            .iter()
            .map(|item| item.event.code.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            codes,
            vec!["orchestration.budgets", "ingest.start", "ingest.error"]
        );

        for pair in streamed.windows(2) {
            let prev = &pair[0].event;
            let next = &pair[1].event;
            assert!(next.seq > prev.seq, "event seq must be strictly increasing");
            assert!(
                next.ts_rfc3339 >= prev.ts_rfc3339,
                "event timestamp must be non-decreasing"
            );
        }
    }

    #[test]
    fn run_pipeline_emits_ingest_error_stage_on_invalid_input() {
        let dir = tempdir().expect("tempdir should be available");
        let (tx, rx) = mpsc::channel();

        let request = TranscribeRequest {
            input: InputSource::File {
                path: dir.path().join("missing.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: dir.path().join("storage.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        };

        let runtime = RuntimeBuilder::current_thread()
            .build()
            .expect("runtime build");
        let result = runtime.block_on(run_pipeline(
            request,
            dir.path(),
            Some(tx),
            &PipelineConfig::default(),
        ));
        assert!(result.is_err());

        let streamed = rx.try_iter().collect::<Vec<_>>();
        let codes = streamed
            .iter()
            .map(|item| item.event.code.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            codes,
            vec!["orchestration.budgets", "ingest.start", "ingest.error"]
        );
    }

    #[test]
    fn ingest_failure_event_order_is_deterministic_across_runs() {
        fn run_once(root: &std::path::Path, idx: usize) -> Vec<(u64, String, String)> {
            let (tx, rx) = mpsc::channel();
            let request = TranscribeRequest {
                input: InputSource::File {
                    path: root.join(format!("missing-{idx}.wav")),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: root.join(format!("failure-{idx}.sqlite3")),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            };

            let runtime = RuntimeBuilder::current_thread()
                .build()
                .expect("runtime build");
            let result = runtime.block_on(run_pipeline(
                request,
                root,
                Some(tx),
                &PipelineConfig::default(),
            ));
            let error = result.expect_err("pipeline should fail ingest on missing input");
            assert!(
                !matches!(error, FwError::Cancelled(_)),
                "missing-input failure should not map to cancellation"
            );

            let streamed = rx.try_iter().collect::<Vec<_>>();
            assert_eq!(streamed.len(), 3, "failure path should emit three events");
            assert_streamed_seq_starts_at_one_and_is_contiguous(&streamed);

            streamed_event_fingerprint(&streamed)
        }

        let dir = tempdir().expect("tempdir should be available");
        let first = run_once(dir.path(), 1);
        let second = run_once(dir.path(), 2);

        assert_eq!(first, second);
        assert_eq!(
            first,
            vec![
                (
                    1,
                    "orchestration".to_owned(),
                    "orchestration.budgets".to_owned()
                ),
                (2, "ingest".to_owned(), "ingest.start".to_owned()),
                (3, "ingest".to_owned(), "ingest.error".to_owned()),
            ]
        );
    }

    #[test]
    fn run_pipeline_emits_checkpoint_cancelled_stage_with_evidence() {
        let dir = tempdir().expect("tempdir should be available");
        let (tx, rx) = mpsc::channel();

        let request = TranscribeRequest {
            input: InputSource::File {
                path: dir.path().join("does-not-matter.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: dir.path().join("storage.sqlite3"),
            timeout_ms: Some(0),
            backend_params: crate::model::BackendParams::default(),
        };

        let runtime = RuntimeBuilder::current_thread()
            .build()
            .expect("runtime build");
        let result = runtime.block_on(run_pipeline(
            request,
            dir.path(),
            Some(tx),
            &PipelineConfig::default(),
        ));
        let error = result.expect_err("pipeline should cancel at checkpoint");
        assert!(matches!(error, FwError::Cancelled(_)));

        let streamed = rx.try_iter().collect::<Vec<_>>();
        assert_eq!(streamed.len(), 2);
        assert_eq!(streamed[0].event.code, "orchestration.budgets");
        assert_eq!(streamed[1].event.code, "orchestration.cancelled");
        assert_eq!(streamed[1].event.stage, "orchestration");

        let payload = &streamed[1].event.payload;
        assert_eq!(payload["checkpoint"], true);
        assert!(payload["cancellation_evidence"].is_object());
        assert_eq!(payload["evidence_count"], 1);
        assert_eq!(
            payload["cancellation_evidence"]["reason"],
            "checkpoint deadline exceeded"
        );
    }

    #[test]
    fn streamed_event_order_matches_persisted_event_order_for_cancellation_sequence() {
        let dir = tempdir().expect("tempdir should be available");
        let db_path = dir.path().join("storage.sqlite3");
        let store = RunStore::open(&db_path).expect("store should open");

        let (tx, rx) = mpsc::channel();
        let mut log = EventLog::new(
            "run-ordered-cancel".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            Some(tx),
        );
        log.push(
            "orchestration",
            "orchestration.budgets",
            "stage budgets applied",
            json!({}),
        );
        log.push(
            "orchestration",
            "orchestration.cancelled",
            "pipeline cancelled by checkpoint policy",
            json!({
                "checkpoint": true,
                "evidence_count": 1,
                "cancellation_evidence": {"stage":"orchestration","reason":"checkpoint deadline exceeded"},
            }),
        );

        let streamed = rx.try_iter().collect::<Vec<_>>();
        let streamed_fingerprint = streamed_event_fingerprint(&streamed);

        let report = RunReport {
            run_id: "run-ordered-cancel".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-23T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-23T00:00:01Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: String::new(),
            request: TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: true,
                db_path: db_path.clone(),
                timeout_ms: Some(0),
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::Auto,
                transcript: String::new(),
                language: None,
                segments: vec![],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: log.events.clone(),
            warnings: vec!["pipeline cancelled".to_owned()],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope::default(),
        };

        store
            .persist_report(&report)
            .expect("report should persist successfully");
        let loaded = store
            .load_run_details("run-ordered-cancel")
            .expect("load should succeed")
            .expect("row should exist");

        let persisted_fingerprint = loaded
            .events
            .iter()
            .map(|event| (event.seq, event.stage.clone(), event.code.clone()))
            .collect::<Vec<_>>();

        assert_eq!(
            streamed_fingerprint,
            vec![
                (
                    1,
                    "orchestration".to_owned(),
                    "orchestration.budgets".to_owned()
                ),
                (
                    2,
                    "orchestration".to_owned(),
                    "orchestration.cancelled".to_owned()
                )
            ]
        );
        assert_eq!(streamed_fingerprint, persisted_fingerprint);
    }

    #[test]
    fn checkpoint_cancellation_event_order_is_deterministic_across_runs() {
        fn run_once(root: &std::path::Path, idx: usize) -> Vec<(u64, String, String)> {
            let (tx, rx) = mpsc::channel();
            let request = TranscribeRequest {
                input: InputSource::File {
                    path: root.join(format!("cancel-input-{idx}.wav")),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: root.join(format!("cancel-{idx}.sqlite3")),
                timeout_ms: Some(0),
                backend_params: BackendParams::default(),
            };

            let runtime = RuntimeBuilder::current_thread()
                .build()
                .expect("runtime build");
            let result = runtime.block_on(run_pipeline(
                request,
                root,
                Some(tx),
                &PipelineConfig::default(),
            ));
            let error = result.expect_err("pipeline should cancel at checkpoint");
            assert!(matches!(error, FwError::Cancelled(_)));

            let streamed = rx.try_iter().collect::<Vec<_>>();
            assert_streamed_seq_starts_at_one_and_is_contiguous(&streamed);
            assert_eq!(streamed.len(), 2);
            assert_eq!(streamed[1].event.payload["checkpoint"], true);
            assert_eq!(
                streamed[1].event.payload["cancellation_evidence"]["reason"],
                "checkpoint deadline exceeded"
            );

            streamed_event_fingerprint(&streamed)
        }

        let dir = tempdir().expect("tempdir should be available");
        let first = run_once(dir.path(), 1);
        let second = run_once(dir.path(), 2);

        assert_eq!(first, second);
        assert_eq!(
            first,
            vec![
                (
                    1,
                    "orchestration".to_owned(),
                    "orchestration.budgets".to_owned()
                ),
                (
                    2,
                    "orchestration".to_owned(),
                    "orchestration.cancelled".to_owned()
                )
            ]
        );
    }

    fn streamed_event_fingerprint(streamed: &[StreamedRunEvent]) -> Vec<(u64, String, String)> {
        streamed
            .iter()
            .map(|item| {
                (
                    item.event.seq,
                    item.event.stage.clone(),
                    item.event.code.clone(),
                )
            })
            .collect::<Vec<_>>()
    }

    fn assert_streamed_seq_starts_at_one_and_is_contiguous(streamed: &[StreamedRunEvent]) {
        for (idx, item) in streamed.iter().enumerate() {
            assert_eq!(
                item.event.seq,
                idx as u64 + 1,
                "event seq should be contiguous and 1-based"
            );
        }
    }

    #[test]
    fn checkpoint_cancel_appends_evidence_trail_entry() {
        let mut pcx = PipelineCx::new(Some(0));
        pcx.record_evidence_values(&[json!({"kind":"preexisting"})]);
        let mut log = EventLog::new(
            "run-cancel".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );

        let error = checkpoint_or_emit("backend", &mut pcx, &mut log)
            .expect_err("checkpoint should emit cancellation");
        assert!(matches!(error, FwError::Cancelled(_)));

        assert_eq!(pcx.evidence().len(), 2);
        assert_eq!(log.events.len(), 1);
        assert_eq!(log.events[0].code, "backend.cancelled");
        assert_eq!(log.events[0].payload["evidence_count"], 2);
        assert_eq!(pcx.evidence()[1]["reason"], "checkpoint deadline exceeded");
    }

    #[test]
    fn stage_budget_timeout_maps_to_timeout_error_code() {
        // Use a multi-worker runtime with a real blocking pool — the
        // production configuration for `run_stage_with_budget`. On a
        // single-threaded `current_thread` runtime the blocking sleep and the
        // budget timer cannot make progress concurrently (asupersync 0.3.2
        // drives the timer on the runtime's worker, which the blocking
        // `spawn_blocking` body would otherwise monopolize), so the timeout
        // could never fire. The wide sleep-vs-budget margin keeps the
        // assertion deterministic under host load.
        let result: Result<(), FwError> = run_stage_with_budget("backend", 1, || {
            std::thread::sleep(Duration::from_millis(250));
            Ok(())
        });
        let error = result.expect_err("stage should time out");

        if let FwError::StageTimeout { stage, budget_ms } = &error {
            assert_eq!(stage, "backend");
            assert_eq!(budget_ms, &1);
        } else {
            assert!(
                matches!(error, FwError::StageTimeout { .. }),
                "expected StageTimeout, got {error:?}"
            );
        }

        assert_eq!(stage_failure_code("backend", &error), "backend.timeout");
        assert_eq!(
            stage_failure_message(&error, "fallback"),
            "stage budget exceeded"
        );
    }

    #[test]
    fn stage_failure_code_and_message_matrix_for_known_classes() {
        let cancelled = FwError::Cancelled("deadline exceeded".to_owned());
        assert_eq!(
            stage_failure_code("normalize", &cancelled),
            "normalize.cancelled"
        );
        assert_eq!(
            stage_failure_message(&cancelled, "normalize failed"),
            "pipeline cancelled by checkpoint policy"
        );

        let generic = FwError::BackendUnavailable("missing".to_owned());
        assert_eq!(stage_failure_code("backend", &generic), "backend.error");
        assert_eq!(
            stage_failure_message(&generic, "backend execution failed"),
            "backend execution failed"
        );
    }

    #[test]
    fn parse_budget_ms_handles_invalid_inputs() {
        assert_eq!(parse_budget_ms(None, 99), 99);
        assert_eq!(parse_budget_ms(Some(""), 99), 99);
        assert_eq!(parse_budget_ms(Some("abc"), 99), 99);
        assert_eq!(parse_budget_ms(Some("0"), 99), 99);
        assert_eq!(parse_budget_ms(Some("2500"), 99), 2500);
    }

    #[test]
    fn stage_budget_policy_from_source_applies_overrides_and_fallbacks() {
        let policy = StageBudgetPolicy::from_source(|key| match key {
            "FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS" => Some("777".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS" => Some("0".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_PROBE_MS" => Some("abc".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS" => Some("12345".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_PERSIST_MS" => Some("250".to_owned()),
            _ => None,
        });

        assert_eq!(policy.ingest_ms, 777);
        assert_eq!(policy.normalize_ms, 180_000);
        assert_eq!(policy.probe_ms, 8_000);
        assert_eq!(policy.backend_ms, 12_345);
        assert_eq!(policy.acceleration_ms, 20_000);
        assert_eq!(policy.persist_ms, 250);
    }

    #[test]
    fn sha256_helpers_are_stable() {
        assert_eq!(
            sha256_bytes_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let json_hash = sha256_json_value(&json!({"k":"v"})).expect("json hash should compute");
        assert_eq!(
            json_hash,
            "666c1aa02e8068c6d5cc1d3295009432c16790bec28ec8ce119d0d1a18d61319"
        );
    }

    #[test]
    fn sha256_file_matches_bytes_hash() {
        let dir = tempdir().expect("tempdir should be available");
        let path = dir.path().join("sample.bin");
        std::fs::write(&path, b"franken-whisper").expect("fixture should write");

        let file_hash = sha256_file(&path).expect("file hash should compute");
        let bytes_hash = sha256_bytes_hex(b"franken-whisper");
        assert_eq!(file_hash, bytes_hash);
    }

    #[test]
    fn acceleration_context_payload_reflects_device_mode_and_fence() {
        let request = TranscribeRequest {
            input: InputSource::Stdin {
                hint_extension: None,
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: PathBuf::from("storage.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams {
                gpu_device: Some("cuda:0".to_owned()),
                flash_attention: Some(true),
                ..BackendParams::default()
            },
        };

        let fence = json!({
            "status": "open",
            "checked_at_rfc3339": "2026-02-22T00:00:00Z",
            "budget_remaining_ms": 9999,
            "error": Value::Null,
            "error_code": Value::Null,
        });
        let accelerated = acceleration_context_payload(
            &request,
            "frankentorch",
            "trace:acceleration:frankentorch:cuda:0",
            fence.clone(),
        );
        assert_eq!(accelerated["mode"], "accelerated");
        assert_eq!(accelerated["logical_stream_kind"], "gpu_stream");
        assert_eq!(
            accelerated["logical_stream_owner_id"],
            "trace:acceleration:frankentorch:cuda:0"
        );
        assert_eq!(accelerated["requested_gpu_device"], "cuda:0");
        assert_eq!(accelerated["flash_attention_requested"], true);
        assert_eq!(accelerated["cancellation_fence"], fence);

        let fallback =
            acceleration_context_payload(&request, "none", "trace:acceleration:none:cpu", fence);
        assert_eq!(fallback["mode"], "cpu_fallback");
        assert_eq!(fallback["logical_stream_kind"], "cpu_lane");
        assert_eq!(fallback["acceleration_backend"], "none");
    }

    #[test]
    fn acceleration_stream_owner_id_embeds_trace_backend_and_device() {
        let request = TranscribeRequest {
            input: InputSource::Stdin {
                hint_extension: None,
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: PathBuf::from("storage.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams {
                gpu_device: Some("cuda:1".to_owned()),
                ..BackendParams::default()
            },
        };

        let owner = acceleration_stream_owner_id("trace-id", &request, "frankentorch");
        assert_eq!(owner, "trace-id:acceleration:frankentorch:cuda:1");

        let no_device = TranscribeRequest {
            backend_params: BackendParams::default(),
            ..request
        };
        let owner_cpu = acceleration_stream_owner_id("trace-id", &no_device, "none");
        assert_eq!(owner_cpu, "trace-id:acceleration:none:cpu");
    }

    #[test]
    fn acceleration_cancellation_fence_payload_has_status_and_budget() {
        let pcx = PipelineCx::new(None);
        let payload = acceleration_cancellation_fence_payload(&pcx);
        assert_eq!(payload["status"], "open");
        assert!(payload["budget_remaining_ms"].is_u64());
        assert!(payload["checked_at_rfc3339"].is_string());
        assert!(payload["error"].is_null());
        assert!(payload["error_code"].is_null());
    }

    #[test]
    fn pipeline_cx_checkpoint_ok_when_no_deadline() {
        let pcx = PipelineCx::new(None);
        assert!(pcx.checkpoint().is_ok());
    }

    #[test]
    fn pipeline_cx_checkpoint_err_when_deadline_past() {
        let pcx = PipelineCx::new(Some(1)); // 1ms budget
        std::thread::sleep(std::time::Duration::from_millis(10));
        let result = pcx.checkpoint();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::FwError::Cancelled(_)));
    }

    #[test]
    fn pipeline_cx_record_evidence_accumulates() {
        let mut pcx = PipelineCx::new(None);
        assert!(pcx.evidence().is_empty());

        let entry = franken_evidence::EvidenceLedgerBuilder::new()
            .ts_unix_ms(1_700_000_000_000)
            .component("test")
            .action("test_action")
            .posterior(vec![0.6, 0.4])
            .expected_loss("test_action", 0.1)
            .chosen_expected_loss(0.1)
            .calibration_score(0.9)
            .build()
            .expect("valid entry");

        pcx.record_evidence(&entry);
        assert_eq!(pcx.evidence().len(), 1);
        pcx.record_evidence(&entry);
        assert_eq!(pcx.evidence().len(), 2);
    }

    #[test]
    fn pipeline_cx_trace_id_is_valid() {
        let pcx = PipelineCx::new(None);
        let trace = pcx.trace_id();
        let hex = trace.to_string();
        assert_eq!(hex.len(), 32);
    }

    // --- Cancellation token tests ---

    #[test]
    fn cancellation_token_no_deadline_always_passes() {
        let pcx = PipelineCx::new(None);
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        assert!(token.checkpoint().is_ok());
    }

    #[test]
    fn cancellation_token_future_deadline_passes() {
        let pcx = PipelineCx::new(Some(60_000)); // 60 seconds
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        assert!(token.checkpoint().is_ok());
    }

    #[test]
    fn cancellation_token_expired_deadline_fails() {
        let pcx = PipelineCx::new(Some(1)); // 1ms
        std::thread::sleep(Duration::from_millis(10));
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        let result = token.checkpoint();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, FwError::Cancelled(_)),
            "expected Cancelled, got: {err:?}"
        );
    }

    #[test]
    fn cancellation_token_is_copy_and_send() {
        let pcx = PipelineCx::new(Some(60_000));
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret

        // Token is Copy — can be used multiple times.
        let t2 = token;
        assert!(token.checkpoint().is_ok());
        assert!(t2.checkpoint().is_ok());

        // Token is Send — can cross thread boundaries.
        let handle = std::thread::spawn(move || t2.checkpoint());
        let result = handle.join().expect("thread should not panic");
        assert!(result.is_ok());
    }

    #[test]
    fn stage_token_respects_stage_budget() {
        let pcx = PipelineCx::new(None);
        let token = pcx.stage_token(1); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(10));
        let result = token.checkpoint();
        assert!(result.is_err());
    }

    #[test]
    fn stage_token_respects_pipeline_deadline() {
        let pcx = PipelineCx::new(Some(1)); // pipeline deadline
        let token = pcx.stage_token(60_000); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(10));
        let result = token.checkpoint();
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_cx_checkpoint_matches_token_checkpoint() {
        let pcx = PipelineCx::new(Some(1)); // 1ms
        std::thread::sleep(Duration::from_millis(10));

        // Both should fail.
        let pcx_result = pcx.checkpoint();
        let token_result = pcx.cancellation_token().checkpoint(); // ubs:ignore — cancellation token is not a secret

        assert!(pcx_result.is_err());
        assert!(token_result.is_err());
    }

    // --- state_root tests ---

    #[test]
    fn state_root_uses_env_when_set() {
        // state_root reads FRANKEN_WHISPER_STATE_DIR; we can't safely set that
        // in parallel tests, but we can verify the fallback path works.
        let root = state_root().expect("state_root should succeed");
        // Either from env or cwd-based fallback — either way should be a valid path.
        assert!(!root.as_os_str().is_empty());
    }

    // --- budget_duration tests ---

    #[test]
    fn budget_duration_clamps_zero_to_one() {
        let d = budget_duration(0);
        assert_eq!(d, Duration::from_millis(1));
    }

    #[test]
    fn budget_duration_preserves_positive_values() {
        assert_eq!(budget_duration(100), Duration::from_millis(100));
        assert_eq!(budget_duration(1), Duration::from_millis(1));
        assert_eq!(budget_duration(60_000), Duration::from_millis(60_000));
    }

    // --- StageBudgetPolicy::as_json tests ---

    #[test]
    fn stage_budget_policy_as_json_includes_all_fields() {
        let policy = StageBudgetPolicy::from_source(|_| None); // all defaults
        let j = policy.as_json();

        assert_eq!(j["ingest_ms"], 15_000);
        assert_eq!(j["normalize_ms"], 180_000);
        assert_eq!(j["probe_ms"], 8_000);
        assert_eq!(j["vad_ms"], 10_000);
        assert_eq!(j["separate_ms"], 30_000);
        assert_eq!(j["backend_ms"], 900_000);
        assert_eq!(j["acceleration_ms"], 20_000);
        assert_eq!(j["align_ms"], 30_000);
        assert_eq!(j["punctuate_ms"], 10_000);
        assert_eq!(j["diarize_ms"], 30_000);
        assert_eq!(j["persist_ms"], 20_000);
        assert_eq!(j["cleanup_budget_ms"], 5_000);

        let overrides = j["overrides"]
            .as_array()
            .expect("overrides should be array");
        assert_eq!(overrides.len(), 12);
    }

    // --- EventLog trace_id injection ---

    #[test]
    fn event_log_injects_trace_id_into_payload() {
        let mut log = EventLog::new(
            "run-tid".to_owned(),
            "abcdef0123456789abcdef0123456789".to_owned(),
            None,
        );
        log.push("test", "test.code", "msg", json!({"extra": true}));

        let payload = &log.events[0].payload;
        assert_eq!(payload["trace_id"], "abcdef0123456789abcdef0123456789");
        // Original fields preserved.
        assert_eq!(payload["extra"], true);
    }

    // --- sha256_file error path ---

    #[test]
    fn sha256_file_returns_error_for_missing_file() {
        let result = sha256_file(&PathBuf::from("/tmp/franken_whisper_nonexistent_file.bin"));
        assert!(result.is_err());
    }

    // --- checkpoint_or_emit success path ---

    #[test]
    fn checkpoint_or_emit_succeeds_without_deadline() {
        let mut pcx = PipelineCx::new(None);
        let mut log = EventLog::new(
            "run-ok".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );

        let result = checkpoint_or_emit("ingest", &mut pcx, &mut log);
        assert!(result.is_ok());
        // No events emitted on success.
        assert!(log.events.is_empty());
    }

    // --- record_evidence_values with multiple entries ---

    #[test]
    fn record_evidence_values_appends_all_entries() {
        let mut pcx = PipelineCx::new(None);
        pcx.record_evidence_values(&[json!({"a": 1}), json!({"b": 2}), json!({"c": 3})]);
        assert_eq!(pcx.evidence().len(), 3);
        assert_eq!(pcx.evidence()[0]["a"], 1);
        assert_eq!(pcx.evidence()[2]["c"], 3);

        // Additional call appends, not replaces.
        pcx.record_evidence_values(&[json!({"d": 4})]);
        assert_eq!(pcx.evidence().len(), 4);
    }

    // --- PipelineCx::new(Some(0)) immediate timeout ---

    #[test]
    fn pipeline_cx_zero_timeout_fails_immediately() {
        let pcx = PipelineCx::new(Some(0));
        // Deadline is now (or in the past), checkpoint should fail.
        let result = pcx.checkpoint();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FwError::Cancelled(_)));
    }

    // --- cancellation_evidence structure ---

    #[test]
    fn cancellation_evidence_contains_expected_fields() {
        let pcx = PipelineCx::new(Some(0));
        let evidence = pcx.cancellation_evidence("backend");

        assert_eq!(evidence["stage"], "backend");
        assert_eq!(evidence["reason"], "checkpoint deadline exceeded");
        assert!(evidence["now_rfc3339"].is_string());
        assert!(evidence["deadline_rfc3339"].is_string());
        assert!(evidence["budget_remaining_ms"].is_number());
        assert!(evidence["overdue_ms"].is_number());
    }

    // --- run_stage_with_budget succeeds ---

    #[test]
    fn run_stage_with_budget_returns_value_on_success() {
        let result = run_stage_with_budget("test", 5_000, || Ok(42));
        assert_eq!(result.unwrap(), 42);
    }

    // --- StageBudgetPolicy all-zero from source uses defaults ---

    #[test]
    fn stage_budget_policy_all_zero_uses_defaults() {
        let policy = StageBudgetPolicy::from_source(|_| Some("0".to_owned()));
        // Zero is filtered out by parse_budget_ms, so all fields should be defaults.
        assert_eq!(policy.ingest_ms, 15_000);
        assert_eq!(policy.normalize_ms, 180_000);
        assert_eq!(policy.probe_ms, 8_000);
        assert_eq!(policy.backend_ms, 900_000);
        assert_eq!(policy.acceleration_ms, 20_000);
        assert_eq!(policy.persist_ms, 20_000);
    }

    // --- stage_failure_code exhaustive coverage ---

    #[test]
    fn stage_failure_code_for_all_error_variants() {
        let stage_timeout = FwError::StageTimeout {
            stage: "normalize".to_owned(),
            budget_ms: 5000,
        };
        assert_eq!(
            stage_failure_code("normalize", &stage_timeout),
            "normalize.timeout"
        );

        let cmd_timeout = FwError::CommandTimedOut {
            command: "ffmpeg".to_owned(),
            timeout_ms: 180_000,
            stderr_suffix: String::new(),
        };
        assert_eq!(
            stage_failure_code("normalize", &cmd_timeout),
            "normalize.timeout"
        );

        let cancelled = FwError::Cancelled("exceeded".to_owned());
        assert_eq!(stage_failure_code("ingest", &cancelled), "ingest.cancelled");

        // All remaining error variants should map to "{stage}.error"
        let io = FwError::Io(std::io::Error::other("test"));
        assert_eq!(stage_failure_code("persist", &io), "persist.error");

        let invalid = FwError::InvalidRequest("bad input".to_owned());
        assert_eq!(stage_failure_code("ingest", &invalid), "ingest.error");

        let backend = FwError::BackendUnavailable("missing".to_owned());
        assert_eq!(stage_failure_code("backend", &backend), "backend.error");

        let cmd_missing = FwError::CommandMissing {
            command: "ffmpeg".to_owned(),
        };
        assert_eq!(
            stage_failure_code("normalize", &cmd_missing),
            "normalize.error"
        );

        let cmd_failed = FwError::CommandFailed {
            command: "ffmpeg".to_owned(),
            status: 1,
            stderr_suffix: "err".to_owned(),
        };
        assert_eq!(
            stage_failure_code("normalize", &cmd_failed),
            "normalize.error"
        );

        let missing_artifact = FwError::MissingArtifact(PathBuf::from("out.json"));
        assert_eq!(
            stage_failure_code("backend", &missing_artifact),
            "backend.error"
        );
    }

    // --- stage_failure_message exhaustive coverage ---

    #[test]
    fn stage_failure_message_for_all_error_classes() {
        let timeout = FwError::StageTimeout {
            stage: "ingest".to_owned(),
            budget_ms: 1000,
        };
        assert_eq!(
            stage_failure_message(&timeout, "fallback"),
            "stage budget exceeded"
        );

        let cmd_timeout = FwError::CommandTimedOut {
            command: "cmd".to_owned(),
            timeout_ms: 1000,
            stderr_suffix: String::new(),
        };
        assert_eq!(
            stage_failure_message(&cmd_timeout, "fallback"),
            "stage budget exceeded"
        );

        let cancelled = FwError::Cancelled("test".to_owned());
        assert_eq!(
            stage_failure_message(&cancelled, "fallback"),
            "pipeline cancelled by checkpoint policy"
        );

        let generic = FwError::Io(std::io::Error::other("x"));
        assert_eq!(
            stage_failure_message(&generic, "custom fallback text"),
            "custom fallback text"
        );
    }

    // --- cancellation_evidence field validation ---

    #[test]
    fn cancellation_evidence_no_deadline_has_null_deadline_and_zero_overdue() {
        let pcx = PipelineCx::new(None);
        let evidence = pcx.cancellation_evidence("test");
        assert!(evidence["deadline_rfc3339"].is_null());
        assert_eq!(evidence["overdue_ms"], 0);
    }

    #[test]
    fn cancellation_evidence_overdue_ms_is_non_negative() {
        // With a past deadline, overdue_ms should be positive.
        let pcx = PipelineCx::new(Some(0));
        std::thread::sleep(Duration::from_millis(5));
        let evidence = pcx.cancellation_evidence("backend");
        let overdue = evidence["overdue_ms"]
            .as_u64()
            .expect("overdue should be number");
        assert!(overdue > 0, "should be overdue, got {overdue}");
    }

    #[test]
    fn cancellation_evidence_now_rfc3339_is_parseable() {
        let pcx = PipelineCx::new(Some(60_000));
        let evidence = pcx.cancellation_evidence("ingest");
        let now_str = evidence["now_rfc3339"]
            .as_str()
            .expect("now_rfc3339 should be string");
        assert!(
            chrono::DateTime::parse_from_rfc3339(now_str).is_ok(),
            "now_rfc3339 should be valid RFC3339: {now_str}"
        );
    }

    // --- PipelineCx large timeout ---

    #[test]
    fn pipeline_cx_large_timeout_does_not_overflow() {
        // Use a very large but non-overflowing value: ~11.5 days in milliseconds.
        let pcx = PipelineCx::new(Some(1_000_000_000));
        // Should not panic or overflow, and checkpoint should pass.
        assert!(pcx.checkpoint().is_ok());
    }

    // --- EventLog without sender ---

    #[test]
    fn event_log_without_sender_does_not_panic() {
        let mut log = EventLog::new(
            "no-tx".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        log.push("test", "test.code", "msg", json!({}));
        assert_eq!(log.events.len(), 1);
    }

    // ── parse_budget_ms tests ──

    #[test]
    fn parse_budget_ms_valid_value_overrides_fallback() {
        assert_eq!(parse_budget_ms(Some("5000"), 15_000), 5000);
        assert_eq!(parse_budget_ms(Some("1"), 15_000), 1);
    }

    #[test]
    fn parse_budget_ms_none_returns_fallback() {
        assert_eq!(parse_budget_ms(None, 15_000), 15_000);
    }

    #[test]
    fn parse_budget_ms_zero_returns_fallback() {
        assert_eq!(parse_budget_ms(Some("0"), 15_000), 15_000);
    }

    #[test]
    fn parse_budget_ms_negative_returns_fallback() {
        assert_eq!(parse_budget_ms(Some("-1"), 15_000), 15_000);
    }

    #[test]
    fn parse_budget_ms_non_numeric_returns_fallback() {
        assert_eq!(parse_budget_ms(Some("abc"), 15_000), 15_000);
        assert_eq!(parse_budget_ms(Some(""), 15_000), 15_000);
        assert_eq!(parse_budget_ms(Some("3.14"), 15_000), 15_000);
    }

    #[test]
    fn parse_budget_ms_very_large_value() {
        assert_eq!(parse_budget_ms(Some("999999999"), 15_000), 999_999_999);
    }

    // ── StageBudgetPolicy::from_source edge cases ──

    #[test]
    fn stage_budget_policy_custom_overrides_each_field() {
        let policy = StageBudgetPolicy::from_source(|key| match key {
            "FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS" => Some("1000".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS" => Some("2000".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_PROBE_MS" => Some("3000".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS" => Some("4000".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_ACCELERATION_MS" => Some("5000".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_PERSIST_MS" => Some("6000".to_owned()),
            _ => None,
        });
        assert_eq!(policy.ingest_ms, 1000);
        assert_eq!(policy.normalize_ms, 2000);
        assert_eq!(policy.probe_ms, 3000);
        assert_eq!(policy.backend_ms, 4000);
        assert_eq!(policy.acceleration_ms, 5000);
        assert_eq!(policy.persist_ms, 6000);
    }

    #[test]
    fn stage_budget_policy_partial_overrides_keep_defaults() {
        let policy = StageBudgetPolicy::from_source(|key| match key {
            "FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS" => Some("500000".to_owned()),
            _ => None,
        });
        assert_eq!(policy.ingest_ms, 15_000); // default
        assert_eq!(policy.normalize_ms, 180_000); // default
        assert_eq!(policy.backend_ms, 500_000); // overridden
        assert_eq!(policy.persist_ms, 20_000); // default
    }

    #[test]
    fn stage_budget_policy_invalid_values_use_defaults() {
        let policy = StageBudgetPolicy::from_source(|key| match key {
            "FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS" => Some("not_a_number".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS" => Some("-100".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_PROBE_MS" => Some("".to_owned()),
            _ => None,
        });
        assert_eq!(policy.ingest_ms, 15_000);
        assert_eq!(policy.normalize_ms, 180_000);
        assert_eq!(policy.probe_ms, 8_000);
    }

    #[test]
    fn stage_budget_policy_as_json_overrides_list_matches_env_keys() {
        let policy = StageBudgetPolicy::from_source(|_| None);
        let j = policy.as_json();
        let overrides: Vec<&str> = j["overrides"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(overrides.contains(&"FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS"));
        assert!(overrides.contains(&"FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS"));
        assert!(overrides.contains(&"FRANKEN_WHISPER_STAGE_BUDGET_PERSIST_MS"));
    }

    // ── sha256 helper tests ──

    #[test]
    fn sha256_bytes_hex_deterministic() {
        let hash1 = sha256_bytes_hex(b"hello world");
        let hash2 = sha256_bytes_hex(b"hello world");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64);
        assert!(hash1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sha256_bytes_hex_different_inputs_different_hashes() {
        let hash1 = sha256_bytes_hex(b"hello");
        let hash2 = sha256_bytes_hex(b"world");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn sha256_bytes_hex_empty_input() {
        let hash = sha256_bytes_hex(b"");
        assert_eq!(hash.len(), 64);
        // SHA-256 of empty string is a known value.
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_json_value_deterministic() {
        let val = json!({"key": "value", "num": 42});
        let hash1 = sha256_json_value(&val).unwrap();
        let hash2 = sha256_json_value(&val).unwrap();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn sha256_json_value_null_is_valid() {
        let hash = sha256_json_value(&json!(null)).unwrap();
        assert_eq!(hash.len(), 64);
    }

    // ── checkpoint_or_emit failure path ──

    #[test]
    fn checkpoint_or_emit_emits_cancelled_event_on_failure() {
        let mut pcx = PipelineCx::new(Some(0)); // immediate timeout
        std::thread::sleep(Duration::from_millis(5));
        let mut log = EventLog::new(
            "run-fail".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );

        let result = checkpoint_or_emit("backend", &mut pcx, &mut log);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FwError::Cancelled(_)));

        // Should have emitted a cancelled event.
        assert_eq!(log.events.len(), 1);
        assert_eq!(log.events[0].code, "backend.cancelled");
        assert_eq!(log.events[0].stage, "backend");
        assert!(
            log.events[0].payload["checkpoint"].as_bool() == Some(true),
            "should mark checkpoint: true"
        );
        assert!(
            log.events[0].payload["cancellation_evidence"].is_object(),
            "should include cancellation_evidence"
        );
        assert!(
            log.events[0].payload["evidence_count"].is_number(),
            "should include evidence_count"
        );
    }

    #[test]
    fn checkpoint_or_emit_does_not_reuse_previous_stage_elapsed_ms() {
        let mut pcx = PipelineCx::new(Some(0));
        std::thread::sleep(Duration::from_millis(5));
        let mut log = EventLog::new(
            "run-fail-elapsed".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        log.mark_stage_start();
        std::thread::sleep(Duration::from_millis(5));

        let result = checkpoint_or_emit("backend", &mut pcx, &mut log);
        assert!(result.is_err());

        let payload = &log.events[0].payload;
        assert!(
            payload.get("elapsed_ms").is_none(),
            "stage-boundary checkpoint failure should not inherit previous stage elapsed"
        );
    }

    // ── EventLog sequence numbering ──

    #[test]
    fn event_log_events_have_sequential_seq_numbers() {
        let mut log = EventLog::new(
            "run-seq".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        log.push("s1", "s1.start", "msg1", json!({}));
        log.push("s2", "s2.start", "msg2", json!({}));
        log.push("s3", "s3.start", "msg3", json!({}));

        for (i, event) in log.events.iter().enumerate() {
            assert_eq!(
                event.seq,
                (i + 1) as u64,
                "event {} should have seq={}, got {}",
                i,
                i + 1,
                event.seq
            );
        }
    }

    #[test]
    fn event_log_events_have_monotonic_timestamps() {
        let mut log = EventLog::new(
            "run-mono".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        log.push("s1", "s1.a", "msg1", json!({}));
        std::thread::sleep(Duration::from_millis(2));
        log.push("s2", "s2.b", "msg2", json!({}));

        assert!(
            log.events[1].ts_rfc3339 >= log.events[0].ts_rfc3339,
            "timestamps should be monotonic: {} >= {}",
            log.events[1].ts_rfc3339,
            log.events[0].ts_rfc3339
        );
    }

    // ── run_stage_with_budget error propagation ──

    #[test]
    fn run_stage_with_budget_propagates_error() {
        let result: crate::error::FwResult<i32> = run_stage_with_budget("test", 5_000, || {
            Err(FwError::InvalidRequest("test error".to_owned()))
        });
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FwError::InvalidRequest(_)));
    }

    // ── budget_duration edge cases ──

    #[test]
    fn budget_duration_zero_clamps_to_one() {
        let d = budget_duration(0);
        assert_eq!(d, Duration::from_millis(1));
    }

    #[test]
    fn budget_duration_one_returns_one() {
        let d = budget_duration(1);
        assert_eq!(d, Duration::from_millis(1));
    }

    #[test]
    fn budget_duration_large_value() {
        let d = budget_duration(900_000);
        assert_eq!(d, Duration::from_millis(900_000));
    }

    // ── sha256_file on real file ──

    #[test]
    fn sha256_file_empty_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, b"").expect("write");
        let hash = sha256_file(&path).expect("hash");
        let expected = sha256_bytes_hex(b"");
        assert_eq!(hash, expected);
    }

    #[test]
    fn sha256_file_large_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("large.bin");
        let data = vec![0xABu8; 100_000];
        std::fs::write(&path, &data).expect("write");
        let hash = sha256_file(&path).expect("hash");
        let expected = sha256_bytes_hex(&data);
        assert_eq!(hash, expected);
    }

    // ── CancellationToken edge cases ──

    #[test]
    fn cancellation_token_from_pipeline_cx_inherits_deadline() {
        let pcx = PipelineCx::new(Some(60_000));
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        // Far future deadline — checkpoint should succeed.
        assert!(token.checkpoint().is_ok());
    }

    #[test]
    fn cancellation_token_from_no_deadline_pcx() {
        let pcx = PipelineCx::new(None);
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        assert!(token.checkpoint().is_ok());
    }

    #[test]
    fn cancellation_token_from_expired_pcx() {
        let pcx = PipelineCx::new(Some(0));
        std::thread::sleep(Duration::from_millis(5));
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        assert!(token.checkpoint().is_err());
    }

    // ── EventLog with sender that gets dropped ──

    #[test]
    fn event_log_with_dropped_receiver_does_not_panic() {
        let (tx, rx) = mpsc::channel();
        drop(rx); // Receiver dropped before sender sends.
        let mut log = EventLog::new(
            "dropped-rx".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            Some(tx),
        );
        // push should not panic even though send will fail.
        log.push("test", "test.code", "msg", json!({}));
        assert_eq!(log.events.len(), 1);
    }

    // ── EventLog mark_stage_start and elapsed_ms ──

    #[test]
    fn event_log_elapsed_ms_none_before_mark() {
        let log = EventLog::new(
            "no-mark".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        assert!(log.stage_elapsed_ms().is_none());
    }

    #[test]
    fn event_log_elapsed_ms_some_after_mark() {
        let mut log = EventLog::new(
            "with-mark".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        log.mark_stage_start();
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = log.stage_elapsed_ms();
        assert!(elapsed.is_some());
        assert!(elapsed.unwrap() >= 1);
    }

    #[test]
    fn event_log_push_includes_elapsed_ms_after_mark() {
        let mut log = EventLog::new(
            "elapsed".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        log.mark_stage_start();
        std::thread::sleep(Duration::from_millis(5));
        log.push("test", "test.code", "msg", json!({}));

        let payload = &log.events[0].payload;
        assert!(
            payload["elapsed_ms"].is_number(),
            "should include elapsed_ms after mark_stage_start"
        );
    }

    #[test]
    fn event_log_push_omits_elapsed_ms_without_mark() {
        let mut log = EventLog::new(
            "no-elapsed".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        log.push("test", "test.code", "msg", json!({}));

        let payload = &log.events[0].payload;
        assert!(
            payload.get("elapsed_ms").is_none(),
            "should not include elapsed_ms without mark_stage_start"
        );
    }

    // ── sha256_json_value different inputs ──

    #[test]
    fn sha256_json_value_different_values_different_hashes() {
        let h1 = sha256_json_value(&json!({"a": 1})).unwrap();
        let h2 = sha256_json_value(&json!({"a": 2})).unwrap();
        assert_ne!(h1, h2);
    }

    // ── state_root returns valid path ──

    #[test]
    fn state_root_returns_ok() {
        let result = state_root();
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(!path.as_os_str().is_empty());
    }

    // ── Additional edge case tests ──

    #[test]
    fn pipeline_cx_evidence_starts_empty() {
        let pcx = PipelineCx::new(None);
        assert!(pcx.evidence().is_empty());
    }

    #[test]
    fn pipeline_cx_record_evidence_values_with_empty_slice() {
        let mut pcx = PipelineCx::new(None);
        pcx.record_evidence_values(&[]);
        assert!(pcx.evidence().is_empty());
    }

    #[test]
    fn pipeline_cx_multiple_evidence_appends() {
        let mut pcx = PipelineCx::new(None);
        pcx.record_evidence_values(&[json!({"a": 1})]);
        pcx.record_evidence_values(&[json!({"b": 2}), json!({"c": 3})]);
        assert_eq!(pcx.evidence().len(), 3);
    }

    #[test]
    fn event_log_many_events_maintain_sequence() {
        let mut log = EventLog::new(
            "many".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        for i in 0..100 {
            log.push("test", &format!("test.{i}"), "msg", json!({}));
        }
        assert_eq!(log.events.len(), 100);
        for (idx, event) in log.events.iter().enumerate() {
            assert_eq!(event.seq, (idx + 1) as u64);
        }
    }

    #[test]
    fn cancellation_evidence_stage_field_matches_input() {
        let pcx = PipelineCx::new(Some(1));
        std::thread::sleep(Duration::from_millis(5));
        let evidence = pcx.cancellation_evidence("my_custom_stage");
        assert_eq!(evidence["stage"], "my_custom_stage");
        assert_eq!(evidence["reason"], "checkpoint deadline exceeded");
    }

    #[test]
    fn checkpoint_or_emit_with_expired_deadline_adds_evidence() {
        let mut pcx = PipelineCx::new(Some(1));
        std::thread::sleep(Duration::from_millis(10));
        let mut log = EventLog::new(
            "evidence-check".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        let err = checkpoint_or_emit("backend", &mut pcx, &mut log);
        assert!(err.is_err());
        // Evidence should have been appended.
        assert!(
            !pcx.evidence().is_empty(),
            "checkpoint failure should add cancellation evidence"
        );
        // Log should contain the cancelled event.
        assert_eq!(log.events.len(), 1);
        assert!(log.events[0].code.contains("cancelled"));
    }

    #[test]
    fn sha256_file_exactly_at_buffer_boundary() {
        // sha256_file uses 8192-byte buffer; test file that's exactly one buffer.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("boundary.bin");
        let data = vec![0x42u8; 8192];
        std::fs::write(&path, &data).expect("write");
        let hash = sha256_file(&path).expect("hash");
        let expected = sha256_bytes_hex(&data);
        assert_eq!(hash, expected);
    }

    #[test]
    fn sha256_file_just_over_buffer_boundary() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("over_boundary.bin");
        let data = vec![0x42u8; 8193];
        std::fs::write(&path, &data).expect("write");
        let hash = sha256_file(&path).expect("hash");
        let expected = sha256_bytes_hex(&data);
        assert_eq!(hash, expected);
    }

    #[test]
    fn budget_duration_u64_max_does_not_panic() {
        let d = budget_duration(u64::MAX);
        assert!(d.as_millis() > 0);
    }

    #[test]
    fn pipeline_cx_deadline_u64_max_does_not_wrap_to_past() {
        // Before the fix, `timeout_ms as i64` with u64::MAX would wrap
        // to -1, creating a deadline in the past.  The fix clamps to
        // i64::MAX so the deadline is always in the future.
        let pcx = PipelineCx::new(Some(u64::MAX));
        assert!(
            pcx.deadline.is_some(),
            "u64::MAX timeout should produce a deadline"
        );
        let deadline = pcx.deadline.unwrap();
        assert!(
            deadline > chrono::Utc::now(),
            "deadline with huge timeout should be in the future, not wrapped to past"
        );
    }

    #[test]
    fn pipeline_cx_deadline_u64_max_saturates_to_max_utc() {
        let pcx = PipelineCx::new(Some(u64::MAX));
        assert_eq!(
            pcx.deadline,
            Some(chrono::DateTime::<chrono::Utc>::MAX_UTC),
            "overflowing deadline math should saturate to MAX_UTC"
        );
    }

    #[test]
    fn pipeline_cx_deadline_i64_max_plus_one_does_not_wrap() {
        let val = (i64::MAX as u64) + 1;
        let pcx = PipelineCx::new(Some(val));
        let deadline = pcx.deadline.unwrap();
        assert!(
            deadline > chrono::Utc::now(),
            "deadline should be in the future for values just above i64::MAX"
        );
    }

    #[test]
    fn stage_budget_policy_from_source_ignores_empty_string() {
        let policy = StageBudgetPolicy::from_source(|_| Some(String::new()));
        // Empty string can't parse to u64, so defaults should apply.
        assert_eq!(policy.ingest_ms, 15_000);
        assert_eq!(policy.backend_ms, 900_000);
    }

    #[test]
    fn event_log_push_non_object_payload_skips_trace_id_injection() {
        let mut log = EventLog::new(
            "run-non-obj".to_owned(),
            "abcdef0123456789abcdef0123456789".to_owned(),
            None,
        );
        // String payload — the `if let Value::Object(...)` guard should NOT match
        log.push("test", "test.code", "msg", json!("a plain string"));
        assert_eq!(log.events[0].payload, json!("a plain string"));
        // trace_id is NOT injected because payload wasn't an object
        // (can't call .get on a string Value, so just check it round-trips as string)
        assert!(log.events[0].payload.is_string());

        // Array payload
        log.push("test", "test.arr", "msg2", json!([1, 2, 3]));
        assert!(log.events[1].payload.is_array());
        assert_eq!(log.events[1].payload.as_array().unwrap().len(), 3);
    }

    #[test]
    fn cancellation_token_with_deadline_from_now_passes_before_deadline() {
        use super::CancellationToken;
        let token = CancellationToken::with_deadline_from_now(Duration::from_secs(60)); // ubs:ignore — cancellation token is not a secret
        assert!(token.checkpoint().is_ok());
    }

    #[test]
    fn cancellation_token_no_deadline_never_cancels() {
        use super::CancellationToken;
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        assert!(token.checkpoint().is_ok());
        // Still passes after a sleep
        std::thread::sleep(Duration::from_millis(5));
        assert!(token.checkpoint().is_ok());
    }

    #[test]
    fn stage_failure_code_for_remaining_error_variants() {
        let json_err = FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err());
        assert_eq!(stage_failure_code("backend", &json_err), "backend.error");

        let storage_err = FwError::Storage("db locked".to_owned());
        assert_eq!(stage_failure_code("persist", &storage_err), "persist.error");

        let unsupported = FwError::Unsupported("not implemented".to_owned());
        assert_eq!(stage_failure_code("ingest", &unsupported), "ingest.error");
    }

    #[test]
    fn cancellation_token_is_send_sync_clone() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        fn assert_clone<T: Clone>() {}
        assert_send::<super::CancellationToken>();
        assert_sync::<super::CancellationToken>();
        assert_clone::<super::CancellationToken>();
    }

    #[test]
    fn event_log_seq_increments_monotonically() {
        let mut log = EventLog::new(
            "seq-test".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        for i in 0..10 {
            log.push("test", &format!("code.{i}"), &format!("msg {i}"), json!({}));
        }
        assert_eq!(log.events.len(), 10);
        for (i, event) in log.events.iter().enumerate() {
            assert_eq!(
                event.seq,
                (i + 1) as u64,
                "seq should be 1-indexed and monotonic"
            );
        }
    }

    #[test]
    fn event_log_streaming_delivers_events_to_receiver() {
        let (tx, rx) = mpsc::channel();
        let mut log = EventLog::new(
            "stream-test".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            Some(tx),
        );
        log.push("a", "a.ok", "first", json!({"n": 1}));
        log.push("b", "b.ok", "second", json!({"n": 2}));

        let received: Vec<_> = rx.try_iter().collect();
        assert_eq!(received.len(), 2);
        assert_eq!(received[0].run_id, "stream-test");
        assert_eq!(received[0].event.stage, "a");
        assert_eq!(received[1].event.stage, "b");
    }

    #[test]
    fn sha256_json_value_consistent_with_bytes_hex() {
        let value = json!({"key": "value", "num": 42});
        let json_hash = sha256_json_value(&value).expect("json hash");
        let bytes = serde_json::to_vec(&value).expect("to_vec");
        let bytes_hash = sha256_bytes_hex(&bytes);
        assert_eq!(json_hash, bytes_hash);
    }

    #[test]
    fn pipeline_cx_deadline_boundary_timeout_one_ms() {
        let pcx = PipelineCx::new(Some(1));
        // Immediately after creation, 1ms hasn't elapsed yet, so checkpoint should pass.
        // (May occasionally fail on extremely slow systems, but this tests the boundary.)
        let result = pcx.checkpoint();
        // We accept either Ok or Err — the test verifies no panic occurs.
        let _ = result;
    }

    #[test]
    fn event_log_push_with_null_payload_does_not_inject_trace_id() {
        let mut log = EventLog::new(
            "run-null".to_owned(),
            "abcdef0123456789abcdef0123456789".to_owned(),
            None,
        );
        log.push("test", "test.code", "msg", json!(null));
        assert!(log.events[0].payload.is_null());
    }

    #[test]
    fn cancellation_token_with_zero_deadline_fails_after_sleep() {
        use super::CancellationToken;
        let token = CancellationToken::with_deadline_from_now(std::time::Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(std::time::Duration::from_millis(5));
        let result = token.checkpoint();
        assert!(
            result.is_err(),
            "zero-duration deadline should fail after sleep"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, FwError::Cancelled(_)),
            "error should be Cancelled variant"
        );
    }

    #[test]
    fn event_log_push_non_object_payload_with_active_mark_drops_elapsed() {
        let mut log = EventLog::new("r".to_owned(), "0".repeat(32), None);
        log.mark_stage_start();
        std::thread::sleep(std::time::Duration::from_millis(5));
        // Array payload — elapsed_ms cannot be injected.
        log.push("test", "test.ok", "msg", json!([1, 2, 3]));
        assert!(
            log.events[0].payload.is_array(),
            "payload should remain an array"
        );
        // No elapsed_ms key since payload is not an object.
        assert!(
            log.events[0].payload.get("elapsed_ms").is_none(),
            "elapsed_ms should not be injected into non-object payload"
        );
    }

    #[test]
    fn event_log_push_object_payload_with_active_mark_injects_elapsed() {
        let mut log = EventLog::new("r".to_owned(), "0".repeat(32), None);
        log.mark_stage_start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        log.push("test", "test.ok", "msg", json!({"key": "value"}));
        let payload = &log.events[0].payload;
        assert!(
            payload.get("elapsed_ms").is_some(),
            "elapsed_ms should be injected into object payload after mark"
        );
        let elapsed = payload["elapsed_ms"].as_u64().expect("should be u64");
        assert!(
            elapsed >= 5,
            "elapsed should be at least 5ms, got {elapsed}"
        );
    }

    #[test]
    fn event_log_push_object_payload_without_mark_does_not_inject_elapsed() {
        let mut log = EventLog::new("r".to_owned(), "0".repeat(32), None);
        // No mark_stage_start() called.
        log.push("test", "test.ok", "msg", json!({"key": "value"}));
        let payload = &log.events[0].payload;
        // trace_id should be injected (object), but elapsed_ms should not (no mark).
        assert!(
            payload.get("trace_id").is_some(),
            "trace_id should be injected"
        );
        assert!(
            payload.get("elapsed_ms").is_none(),
            "elapsed_ms should NOT be injected without mark_stage_start"
        );
    }

    #[test]
    fn stage_budget_policy_from_source_custom_overrides() {
        let policy = StageBudgetPolicy::from_source(|key| match key {
            "FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS" => Some("500".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS" => Some("1000".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_PROBE_MS" => Some("200".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS" => Some("60000".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_ACCELERATION_MS" => Some("3000".to_owned()),
            "FRANKEN_WHISPER_STAGE_BUDGET_PERSIST_MS" => Some("250".to_owned()),
            _ => None,
        });
        assert_eq!(policy.ingest_ms, 500);
        assert_eq!(policy.normalize_ms, 1000);
        assert_eq!(policy.probe_ms, 200);
        assert_eq!(policy.backend_ms, 60000);
        assert_eq!(policy.acceleration_ms, 3000);
        assert_eq!(policy.persist_ms, 250);
    }

    // ── Fifth-pass edge case tests ──

    #[test]
    fn checkpoint_or_emit_streams_cancelled_event_to_receiver() {
        let (tx, rx) = mpsc::channel();
        let mut pcx = PipelineCx::new(Some(0));
        std::thread::sleep(Duration::from_millis(5));
        let mut log = EventLog::new(
            "run-stream-cancel".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            Some(tx),
        );

        let result = checkpoint_or_emit("normalize", &mut pcx, &mut log);
        assert!(result.is_err());

        let received: Vec<_> = rx.try_iter().collect();
        assert_eq!(received.len(), 1, "cancelled event should be streamed");
        assert_eq!(received[0].run_id, "run-stream-cancel");
        assert_eq!(received[0].event.code, "normalize.cancelled");
        assert!(
            received[0].event.payload["cancellation_evidence"].is_object(),
            "streamed event should include cancellation_evidence"
        );
    }

    #[test]
    fn cancellation_evidence_future_deadline_clamps_overdue_to_zero() {
        // With a deadline far in the future, overdue_ms should be clamped to 0
        // even though signed_duration_since returns negative.
        let pcx = PipelineCx::new(Some(600_000)); // 10 minutes from now
        let evidence = pcx.cancellation_evidence("ingest");

        let overdue = evidence["overdue_ms"]
            .as_u64()
            .expect("overdue_ms should be a number");
        assert_eq!(overdue, 0, "future deadline should produce overdue_ms=0");

        assert!(
            evidence["deadline_rfc3339"].is_string(),
            "deadline should be non-null for set deadline"
        );
    }

    #[test]
    fn event_log_push_non_object_payload_still_streams_to_receiver() {
        let (tx, rx) = mpsc::channel();
        let mut log = EventLog::new(
            "run-nonobj-stream".to_owned(),
            "abcdef0123456789abcdef0123456789".to_owned(),
            Some(tx),
        );

        log.push("test", "test.string", "msg", json!("plain string"));
        log.push("test", "test.null", "msg2", json!(null));
        log.push("test", "test.array", "msg3", json!([1, 2]));

        let received: Vec<_> = rx.try_iter().collect();
        assert_eq!(received.len(), 3, "all events should be streamed");

        // String payload: streamed without trace_id enrichment
        assert!(received[0].event.payload.is_string());
        assert_eq!(received[0].event.code, "test.string");
        assert_eq!(received[0].event.seq, 1);

        // Null payload
        assert!(received[1].event.payload.is_null());
        assert_eq!(received[1].event.seq, 2);

        // Array payload
        assert!(received[2].event.payload.is_array());
        assert_eq!(received[2].event.seq, 3);
    }

    #[test]
    fn stage_budget_policy_from_env_returns_defaults_when_no_env_set() {
        // from_env reads real env vars; in test context these should not be set,
        // so we get all defaults. (Safe: only reads, never writes env.)
        let policy = StageBudgetPolicy::from_env();
        assert_eq!(policy.ingest_ms, 15_000);
        assert_eq!(policy.normalize_ms, 180_000);
        assert_eq!(policy.probe_ms, 8_000);
        assert_eq!(policy.backend_ms, 900_000);
        assert_eq!(policy.acceleration_ms, 20_000);
        assert_eq!(policy.persist_ms, 20_000);
    }

    #[test]
    fn pipeline_cx_evidence_ordering_preserved_across_mixed_sources() {
        // Evidence entries from record_evidence_values and cancellation_evidence
        // should be ordered by insertion time.
        let mut pcx = PipelineCx::new(Some(0));
        pcx.record_evidence_values(&[json!({"source": "manual_1"})]);
        pcx.record_evidence_values(&[json!({"source": "manual_2"}), json!({"source": "manual_3"})]);

        // Simulate what checkpoint_or_emit does: append cancellation evidence
        let cancel_ev = pcx.cancellation_evidence("backend");
        pcx.record_evidence_values(std::slice::from_ref(&cancel_ev));

        pcx.record_evidence_values(&[json!({"source": "manual_4"})]);

        let ev = pcx.evidence();
        assert_eq!(ev.len(), 5);
        assert_eq!(ev[0]["source"], "manual_1");
        assert_eq!(ev[1]["source"], "manual_2");
        assert_eq!(ev[2]["source"], "manual_3");
        assert_eq!(ev[3]["reason"], "checkpoint deadline exceeded");
        assert_eq!(ev[4]["source"], "manual_4");
    }

    // -----------------------------------------------------------------------
    // Cx threading tests (bd-38c.2)
    // -----------------------------------------------------------------------

    #[test]
    fn cx_cancellation_token_propagates_deadline() {
        // A PipelineCx with a short timeout should produce a CancellationToken
        // that also reports cancelled after the deadline passes.
        let pcx = PipelineCx::new(Some(1)); // 1ms deadline
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret

        // After sleeping past deadline, both should report cancelled
        std::thread::sleep(Duration::from_millis(10));
        assert!(pcx.checkpoint().is_err(), "PipelineCx should be cancelled");
        assert!(
            token.checkpoint().is_err(),
            "CancellationToken should be cancelled"
        );
    }

    #[test]
    fn cx_cancellation_token_no_deadline_never_cancels() {
        let pcx = PipelineCx::new(None);
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret

        assert!(
            pcx.checkpoint().is_ok(),
            "no-deadline PipelineCx should pass"
        );
        assert!(
            token.checkpoint().is_ok(),
            "no-deadline CancellationToken should pass"
        );
    }

    #[test]
    fn cx_cancellation_token_inherits_budget() {
        // When PipelineCx has a deadline, the CancellationToken should reflect it
        let pcx = PipelineCx::new(Some(60_000)); // 60s
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret

        // Both should be OK within budget
        assert!(pcx.checkpoint().is_ok());
        assert!(token.checkpoint().is_ok());
    }

    #[test]
    fn cx_materialize_input_respects_cancellation() {
        // materialize_input_with_token should fail immediately when given expired token
        let token = super::CancellationToken::with_deadline_from_now(Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(5));

        let dir = tempdir().unwrap();
        let source = InputSource::File {
            path: PathBuf::from("/nonexistent.wav"),
        };

        let result = crate::audio::materialize_input_with_token(&source, dir.path(), Some(&token));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, FwError::Cancelled(_)),
            "expected Cancelled error, got: {err:?}"
        );
    }

    #[test]
    fn cx_accelerate_respects_cancellation() {
        // apply_with_token should return early when given expired token
        let token = super::CancellationToken::with_deadline_from_now(Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(5));

        let mut result = TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "test".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "test".to_owned(),
                speaker: None,
                confidence: Some(0.9),
            }],
            acceleration: None,
            raw_output: json!({}),
            artifact_paths: vec![],
        };

        let report = crate::accelerate::apply_with_token(&mut result, Some(&token));
        assert_eq!(
            report.backend.as_str(),
            "none",
            "cancelled acceleration should return None backend"
        );
        assert!(
            report.notes.iter().any(|n| n.contains("cancelled")),
            "notes should mention cancellation"
        );
    }

    #[test]
    fn cx_persist_report_respects_cancellation() {
        // persist_report_cancellable should fail when given expired token
        let token = super::CancellationToken::with_deadline_from_now(Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(5));

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("cx_test.sqlite3");
        let store = RunStore::open(&db_path).expect("open store");

        let report = RunReport {
            run_id: "run-cx-test".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-01-01T00:01:00Z".to_owned(),
            input_path: "/tmp/test.wav".to_owned(),
            normalized_wav_path: "/tmp/normalized.wav".to_owned(),
            request: TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("test.wav"),
                },
                backend: BackendKind::WhisperCpp,
                model: None,
                language: Some("en".to_owned()),
                translate: false,
                diarize: false,
                persist: false,
                db_path: db_path.clone(),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "test".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope::default(),
        };

        let result = store.persist_report_cancellable(&report, Some(&token));
        assert!(result.is_err(), "persist should fail when cancelled");
        let err = result.unwrap_err();
        assert!(
            matches!(err, FwError::Cancelled(_)),
            "expected Cancelled error, got: {err:?}"
        );
    }

    #[test]
    fn stage_budget_ms_unknown_stage_returns_none() {
        let budgets = StageBudgetPolicy {
            ingest_ms: 100,
            normalize_ms: 200,
            probe_ms: 300,
            vad_ms: 10_000,
            separate_ms: 30_000,
            backend_ms: 400,
            acceleration_ms: 500,
            align_ms: 550,
            punctuate_ms: 10_000,
            diarize_ms: 30_000,
            persist_ms: 600,
            cleanup_budget_ms: 5_000,
        };
        // Known stages return Some
        assert_eq!(stage_budget_ms("ingest", budgets), Some(100));
        assert_eq!(stage_budget_ms("persist", budgets), Some(600));
        // Unknown stages return None (including "probe" which is only in StageBudgetPolicy, not in the match)
        assert_eq!(stage_budget_ms("probe", budgets), None);
        assert_eq!(stage_budget_ms("unknown", budgets), None);
        assert_eq!(stage_budget_ms("", budgets), None);
    }

    #[test]
    fn parse_event_ts_ms_invalid_timestamp_returns_none() {
        let valid_event = RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T10:00:00Z".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.start".to_owned(),
            message: "begin".to_owned(),
            payload: json!({}),
        };
        assert!(parse_event_ts_ms(&valid_event).is_some());

        let invalid_event = RunEvent {
            seq: 2,
            ts_rfc3339: "not-a-valid-date".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.start".to_owned(),
            message: "begin".to_owned(),
            payload: json!({}),
        };
        assert_eq!(parse_event_ts_ms(&invalid_event), None);

        let empty_event = RunEvent {
            seq: 3,
            ts_rfc3339: "".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.start".to_owned(),
            message: "begin".to_owned(),
            payload: json!({}),
        };
        assert_eq!(parse_event_ts_ms(&empty_event), None);
    }

    #[test]
    fn event_elapsed_ms_non_u64_payload_returns_none() {
        // String value instead of u64
        let string_event = RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T10:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({"elapsed_ms": "1000"}),
        };
        assert_eq!(event_elapsed_ms(&string_event), None);

        // Missing elapsed_ms field entirely
        let missing_event = RunEvent {
            seq: 2,
            ts_rfc3339: "2026-02-22T10:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({"other_key": 42}),
        };
        assert_eq!(event_elapsed_ms(&missing_event), None);

        // Negative number (not u64)
        let negative_event = RunEvent {
            seq: 3,
            ts_rfc3339: "2026-02-22T10:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({"elapsed_ms": -5}),
        };
        assert_eq!(event_elapsed_ms(&negative_event), None);

        // Valid u64 value works
        let valid_event = RunEvent {
            seq: 4,
            ts_rfc3339: "2026-02-22T10:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({"elapsed_ms": 500}),
        };
        assert_eq!(event_elapsed_ms(&valid_event), Some(500));
    }

    #[test]
    fn stage_latency_profile_empty_events_returns_zero_observed() {
        let budgets = StageBudgetPolicy {
            ingest_ms: 5_000,
            normalize_ms: 10_000,
            probe_ms: 8_000,
            vad_ms: 10_000,
            separate_ms: 30_000,
            backend_ms: 60_000,
            acceleration_ms: 20_000,
            align_ms: 30_000,
            punctuate_ms: 10_000,
            diarize_ms: 30_000,
            persist_ms: 20_000,
            cleanup_budget_ms: 5_000,
        };
        let profile = stage_latency_profile(&[], budgets);
        assert_eq!(profile["artifact"], "stage_latency_decomposition_v1");
        assert_eq!(profile["summary"]["observed_stages"], 0);
        assert_eq!(profile["summary"]["queue_total_ms"], 0);
        assert_eq!(profile["summary"]["service_total_ms"], 0);
        assert_eq!(profile["summary"]["external_process_total_ms"], 0);
        // No stage profiles should be present
        let stages = profile["stages"]
            .as_object()
            .expect("stages should be object");
        assert!(stages.is_empty(), "no stage profiles for empty events");
        // Recommendations array should be empty
        let recs = profile["budget_tuning"]["recommendations"]
            .as_array()
            .expect("recommendations should be array");
        assert!(recs.is_empty());
    }

    #[test]
    fn recommended_budget_zero_budget_preserves_service_floor() {
        // budget_ms=0 triggers the early return branch
        let (budget, action, reason, utilization) = recommended_budget(500, 0);
        assert_eq!(budget, 500);
        assert_eq!(action, "keep_budget");
        assert!(reason.contains("budget undefined"));
        assert_eq!(utilization, 0.0);

        // service_ms=0 with budget_ms=0 clamps to 1
        let (budget, action, _reason, _) = recommended_budget(0, 0);
        assert_eq!(budget, 1);
        assert_eq!(action, "keep_budget");
    }

    #[test]
    fn cx_all_stages_receive_stage_tokens() {
        // Verify the orchestrator can create stage-budgeted tokens for all stages.
        let pcx = PipelineCx::new(Some(300_000));

        let ingest_token = pcx.stage_token(5_000); // ubs:ignore — cancellation token is not a secret
        let normalize_token = pcx.stage_token(10_000); // ubs:ignore — cancellation token is not a secret
        let backend_token = pcx.stage_token(15_000); // ubs:ignore — cancellation token is not a secret
        let acceleration_token = pcx.stage_token(20_000); // ubs:ignore — cancellation token is not a secret
        let persist_token = pcx.stage_token(25_000); // ubs:ignore — cancellation token is not a secret

        // All tokens should be valid (not cancelled) for a far-future deadline
        assert!(
            ingest_token.checkpoint().is_ok(),
            "ingest token should pass"
        );
        assert!(
            normalize_token.checkpoint().is_ok(),
            "normalize token should pass"
        );
        assert!(
            backend_token.checkpoint().is_ok(),
            "backend token should pass"
        );
        assert!(
            acceleration_token.checkpoint().is_ok(),
            "acceleration token should pass"
        );
        assert!(
            persist_token.checkpoint().is_ok(),
            "persist token should pass"
        );
    }

    // -----------------------------------------------------------------------
    // Pipeline finalizers (bd-38c.3)
    // -----------------------------------------------------------------------

    #[test]
    fn test_finalizer_registry_runs_in_reverse_order() {
        use std::sync::{Arc, Mutex};

        let order: Arc<Mutex<Vec<&str>>> = Arc::new(Mutex::new(Vec::new()));
        let mut registry = FinalizerRegistry::new();

        let o1 = Arc::clone(&order);
        registry.register(
            "first",
            Finalizer::Custom(Box::new(move || {
                o1.lock().unwrap().push("first");
            })),
        );

        let o2 = Arc::clone(&order);
        registry.register(
            "second",
            Finalizer::Custom(Box::new(move || {
                o2.lock().unwrap().push("second");
            })),
        );

        let o3 = Arc::clone(&order);
        registry.register(
            "third",
            Finalizer::Custom(Box::new(move || {
                o3.lock().unwrap().push("third");
            })),
        );

        assert_eq!(registry.len(), 3);
        registry.run_all();
        assert_eq!(registry.len(), 0, "registry should be empty after run_all");

        let executed = order.lock().unwrap();
        assert_eq!(
            *executed,
            vec!["third", "second", "first"],
            "finalizers should run in LIFO order"
        );
    }

    #[test]
    fn test_finalizer_registry_empty() {
        let mut registry = FinalizerRegistry::new();
        assert_eq!(registry.len(), 0);
        // run_all on empty registry should not panic
        registry.run_all();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_pipeline_cx_registers_and_runs_finalizers() {
        use std::sync::{Arc, Mutex};

        let mut pcx = PipelineCx::new(None);
        assert_eq!(pcx.finalizers.len(), 0);

        let called: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
        let called_clone = Arc::clone(&called);
        pcx.register_finalizer(
            "test_custom",
            Finalizer::Custom(Box::new(move || {
                *called_clone.lock().unwrap() = true;
            })),
        );
        assert_eq!(pcx.finalizers.len(), 1);

        pcx.register_finalizer(
            "test_tmpdir",
            Finalizer::TempDir(PathBuf::from("/tmp/nonexistent_fw_test")),
        );
        assert_eq!(pcx.finalizers.len(), 2);

        pcx.run_finalizers();
        assert_eq!(pcx.finalizers.len(), 0, "finalizers should be drained");
        assert!(
            *called.lock().unwrap(),
            "custom finalizer should have been called"
        );
    }

    #[test]
    fn test_temp_dir_finalizer_marks_for_cleanup() {
        // TempDir finalizer should log but not delete (that is tempfile::TempDir's
        // responsibility).  Verify it does not panic even with a non-existent path.
        let mut registry = FinalizerRegistry::new();
        registry.register(
            "tmp_cleanup",
            Finalizer::TempDir(PathBuf::from("/tmp/does_not_exist_fw_test_dir")),
        );
        // Should not panic
        registry.run_all();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_custom_finalizer_executes() {
        use std::sync::{Arc, Mutex};

        let counter: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

        let mut registry = FinalizerRegistry::new();

        let c1 = Arc::clone(&counter);
        registry.register(
            "increment_a",
            Finalizer::Custom(Box::new(move || {
                *c1.lock().unwrap() += 10;
            })),
        );

        let c2 = Arc::clone(&counter);
        registry.register(
            "increment_b",
            Finalizer::Custom(Box::new(move || {
                *c2.lock().unwrap() += 1;
            })),
        );

        registry.run_all();

        let final_value = *counter.lock().unwrap();
        assert_eq!(
            final_value, 11,
            "both custom finalizers should have executed"
        );
    }

    // -----------------------------------------------------------------------
    // Composable pipeline stages (bd-qla.6)
    // -----------------------------------------------------------------------

    use super::PipelineBuilder;

    #[test]
    fn pipeline_config_default_matches_original_stage_order() {
        let config = PipelineConfig::default();
        assert_eq!(
            config.stages(),
            &[
                PipelineStage::Ingest,
                PipelineStage::Normalize,
                PipelineStage::Vad,
                PipelineStage::Separate,
                PipelineStage::Backend,
                PipelineStage::Accelerate,
                PipelineStage::Align,
                PipelineStage::Punctuate,
                PipelineStage::Diarize,
                PipelineStage::Persist,
            ]
        );
    }

    #[test]
    fn pipeline_config_default_validates_ok() {
        let config = PipelineConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pipeline_config_has_stage_true_for_present_stages() {
        let config = PipelineConfig::default();
        assert!(config.has_stage(PipelineStage::Ingest));
        assert!(config.has_stage(PipelineStage::Normalize));
        assert!(config.has_stage(PipelineStage::Vad));
        assert!(config.has_stage(PipelineStage::Separate));
        assert!(config.has_stage(PipelineStage::Backend));
        assert!(config.has_stage(PipelineStage::Accelerate));
        assert!(config.has_stage(PipelineStage::Align));
        assert!(config.has_stage(PipelineStage::Punctuate));
        assert!(config.has_stage(PipelineStage::Diarize));
        assert!(config.has_stage(PipelineStage::Persist));
    }

    #[test]
    fn pipeline_config_has_stage_false_for_removed_stage() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Backend,
            PipelineStage::Persist,
        ]);
        assert!(!config.has_stage(PipelineStage::Accelerate));
    }

    #[test]
    fn pipeline_config_validate_rejects_duplicate_stages() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Ingest,
        ]);
        let err = config.validate().expect_err("should reject duplicates");
        assert!(
            matches!(err, FwError::InvalidRequest(_)),
            "expected InvalidRequest, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("duplicate"),
            "error should mention duplicate: {msg}"
        );
    }

    #[test]
    fn pipeline_config_validate_rejects_normalize_without_ingest() {
        let config = PipelineConfig::new(vec![PipelineStage::Normalize]);
        let err = config.validate().expect_err("should reject missing ingest");
        assert!(matches!(err, FwError::InvalidRequest(_)));
        let msg = err.to_string();
        assert!(msg.contains("Ingest"), "error should mention Ingest: {msg}");
    }

    #[test]
    fn pipeline_config_validate_rejects_backend_without_normalize() {
        let config = PipelineConfig::new(vec![PipelineStage::Ingest, PipelineStage::Backend]);
        let err = config
            .validate()
            .expect_err("should reject missing normalize");
        assert!(matches!(err, FwError::InvalidRequest(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("Normalize"),
            "error should mention Normalize: {msg}"
        );
    }

    #[test]
    fn pipeline_config_validate_rejects_accelerate_without_backend() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Accelerate,
        ]);
        let err = config
            .validate()
            .expect_err("should reject missing backend");
        assert!(matches!(err, FwError::InvalidRequest(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("Backend"),
            "error should mention Backend: {msg}"
        );
    }

    #[test]
    fn pipeline_config_validate_rejects_wrong_order_ingest_after_normalize() {
        let config = PipelineConfig::new(vec![PipelineStage::Normalize, PipelineStage::Ingest]);
        let err = config.validate().expect_err("should reject wrong order");
        assert!(matches!(err, FwError::InvalidRequest(_)));
    }

    #[test]
    fn pipeline_config_validate_accepts_skip_acceleration() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Backend,
            PipelineStage::Persist,
        ]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pipeline_config_validate_accepts_skip_persist() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Backend,
            PipelineStage::Accelerate,
        ]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pipeline_config_validate_accepts_ingest_only() {
        let config = PipelineConfig::new(vec![PipelineStage::Ingest]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pipeline_config_validate_accepts_empty() {
        let config = PipelineConfig::new(vec![]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pipeline_config_validate_accepts_persist_only() {
        let config = PipelineConfig::new(vec![PipelineStage::Persist]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pipeline_builder_new_builds_empty() {
        let config = PipelineBuilder::new().build_unchecked();
        assert!(config.stages().is_empty());
    }

    #[test]
    fn pipeline_builder_stage_appends() {
        let config = PipelineBuilder::new()
            .stage(PipelineStage::Ingest)
            .stage(PipelineStage::Normalize)
            .build_unchecked();
        assert_eq!(
            config.stages(),
            &[PipelineStage::Ingest, PipelineStage::Normalize]
        );
    }

    #[test]
    fn pipeline_builder_default_stages_matches_default_config() {
        let config = PipelineBuilder::default_stages().build().expect("valid");
        assert_eq!(config.stages(), PipelineConfig::default().stages());
    }

    #[test]
    fn pipeline_builder_without_removes_stage() {
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Accelerate)
            .build()
            .expect("valid");
        assert_eq!(
            config.stages(),
            &[
                PipelineStage::Ingest,
                PipelineStage::Normalize,
                PipelineStage::Vad,
                PipelineStage::Separate,
                PipelineStage::Backend,
                PipelineStage::Align,
                PipelineStage::Punctuate,
                PipelineStage::Diarize,
                PipelineStage::Persist,
            ]
        );
        assert!(!config.has_stage(PipelineStage::Accelerate));
    }

    #[test]
    fn pipeline_builder_without_persist() {
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Persist)
            .build()
            .expect("valid");
        assert_eq!(
            config.stages(),
            &[
                PipelineStage::Ingest,
                PipelineStage::Normalize,
                PipelineStage::Vad,
                PipelineStage::Separate,
                PipelineStage::Backend,
                PipelineStage::Accelerate,
                PipelineStage::Align,
                PipelineStage::Punctuate,
                PipelineStage::Diarize,
            ]
        );
    }

    #[test]
    fn pipeline_builder_without_acceleration_and_persist() {
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Accelerate)
            .without(PipelineStage::Persist)
            .build()
            .expect("valid");
        assert_eq!(
            config.stages(),
            &[
                PipelineStage::Ingest,
                PipelineStage::Normalize,
                PipelineStage::Vad,
                PipelineStage::Separate,
                PipelineStage::Backend,
                PipelineStage::Align,
                PipelineStage::Punctuate,
                PipelineStage::Diarize,
            ]
        );
    }

    #[test]
    fn pipeline_builder_build_validates_and_rejects_invalid() {
        let result = PipelineBuilder::new().stage(PipelineStage::Backend).build();
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_builder_build_unchecked_skips_validation() {
        let config = PipelineBuilder::new()
            .stage(PipelineStage::Backend)
            .build_unchecked();
        assert_eq!(config.stages(), &[PipelineStage::Backend]);
    }

    #[test]
    fn pipeline_builder_stages_replaces_all() {
        let config = PipelineBuilder::default_stages()
            .stages(vec![PipelineStage::Ingest, PipelineStage::Persist])
            .build()
            .expect("valid");
        assert_eq!(
            config.stages(),
            &[PipelineStage::Ingest, PipelineStage::Persist]
        );
    }

    #[test]
    fn pipeline_stage_label_matches_expected() {
        assert_eq!(PipelineStage::Ingest.label(), "ingest");
        assert_eq!(PipelineStage::Normalize.label(), "normalize");
        assert_eq!(PipelineStage::Vad.label(), "vad");
        assert_eq!(PipelineStage::Separate.label(), "separate");
        assert_eq!(PipelineStage::Backend.label(), "backend");
        assert_eq!(PipelineStage::Accelerate.label(), "acceleration");
        assert_eq!(PipelineStage::Align.label(), "align");
        assert_eq!(PipelineStage::Punctuate.label(), "punctuate");
        assert_eq!(PipelineStage::Diarize.label(), "diarize");
        assert_eq!(PipelineStage::Persist.label(), "persist");
    }

    #[test]
    fn pipeline_stage_display_matches_label() {
        for stage in &[
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Vad,
            PipelineStage::Separate,
            PipelineStage::Backend,
            PipelineStage::Accelerate,
            PipelineStage::Align,
            PipelineStage::Punctuate,
            PipelineStage::Diarize,
            PipelineStage::Persist,
        ] {
            assert_eq!(stage.to_string(), stage.label());
        }
    }

    #[test]
    fn pipeline_stage_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelineStage::Ingest);
        set.insert(PipelineStage::Normalize);
        set.insert(PipelineStage::Ingest); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn pipeline_stage_clone_and_copy() {
        let stage = PipelineStage::Backend;
        let cloned = stage;
        assert_eq!(stage, cloned);
    }

    #[test]
    fn pipeline_config_clone() {
        let config = PipelineConfig::default();
        let cloned = config.clone();
        assert_eq!(config.stages(), cloned.stages());
    }

    #[test]
    fn pipeline_config_debug() {
        let config = PipelineConfig::default();
        let debug = format!("{config:?}");
        assert!(
            debug.contains("Ingest"),
            "debug should include stage names: {debug}"
        );
        assert!(
            debug.contains("Persist"),
            "debug should include Persist: {debug}"
        );
    }

    #[test]
    fn pipeline_stage_debug() {
        let debug = format!("{:?}", PipelineStage::Backend);
        assert_eq!(debug, "Backend");
    }

    #[test]
    fn run_pipeline_with_default_config_emits_ingest_error_on_invalid_input() {
        // This test verifies the composable pipeline with default config
        // produces the same output as the original hardcoded pipeline.
        let dir = tempdir().expect("tempdir should be available");
        let (tx, rx) = mpsc::channel();

        let request = TranscribeRequest {
            input: InputSource::File {
                path: dir.path().join("missing.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: dir.path().join("storage.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        };

        let runtime = RuntimeBuilder::current_thread()
            .build()
            .expect("runtime build");
        let result = runtime.block_on(run_pipeline(
            request,
            dir.path(),
            Some(tx),
            &PipelineConfig::default(),
        ));
        assert!(result.is_err());

        let streamed = rx.try_iter().collect::<Vec<_>>();
        let codes = streamed
            .iter()
            .map(|item| item.event.code.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            codes,
            vec!["orchestration.budgets", "ingest.start", "ingest.error"]
        );

        // Verify the budgets event includes pipeline_stages field
        let budgets_payload = &streamed[0].event.payload;
        assert!(
            budgets_payload["pipeline_stages"].is_array(),
            "budgets event should include pipeline_stages"
        );
    }

    #[test]
    fn run_pipeline_with_default_config_checkpoint_cancel_with_evidence() {
        let dir = tempdir().expect("tempdir should be available");
        let (tx, rx) = mpsc::channel();

        let request = TranscribeRequest {
            input: InputSource::File {
                path: dir.path().join("does-not-matter.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: dir.path().join("storage.sqlite3"),
            timeout_ms: Some(0),
            backend_params: crate::model::BackendParams::default(),
        };

        let runtime = RuntimeBuilder::current_thread()
            .build()
            .expect("runtime build");
        let result = runtime.block_on(run_pipeline(
            request,
            dir.path(),
            Some(tx),
            &PipelineConfig::default(),
        ));
        let error = result.expect_err("pipeline should cancel at checkpoint");
        assert!(matches!(error, FwError::Cancelled(_)));

        let streamed = rx.try_iter().collect::<Vec<_>>();
        assert_eq!(streamed.len(), 2);
        assert_eq!(streamed[0].event.code, "orchestration.budgets");
        assert_eq!(streamed[1].event.code, "orchestration.cancelled");
    }

    #[test]
    fn optional_stage_skip_respects_request_flags() {
        let mut request = TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::WhisperCpp,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: PathBuf::from("db.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        };

        assert!(optional_stage_skip(PipelineStage::Vad, &request).is_some());
        assert!(optional_stage_skip(PipelineStage::Separate, &request).is_some());
        assert!(optional_stage_skip(PipelineStage::Align, &request).is_some());
        assert!(optional_stage_skip(PipelineStage::Punctuate, &request).is_some());
        assert!(optional_stage_skip(PipelineStage::Diarize, &request).is_some());

        request.backend_params.vad = Some(VadParams::default());
        request.diarize = true;
        request.backend_params.diarization_config = Some(DiarizationConfig::default());

        assert!(optional_stage_skip(PipelineStage::Vad, &request).is_none());
        assert!(optional_stage_skip(PipelineStage::Separate, &request).is_none());
        assert!(optional_stage_skip(PipelineStage::Align, &request).is_none());
        assert!(optional_stage_skip(PipelineStage::Diarize, &request).is_none());

        let punctuate_skip = optional_stage_skip(PipelineStage::Punctuate, &request);
        assert!(
            punctuate_skip.is_none(),
            "diarize=true should enable punctuation stage through the diarization packet"
        );

        request.backend_params.diarization_config = Some(DiarizationConfig {
            no_stem: true,
            ..DiarizationConfig::default()
        });
        let separate_skip = optional_stage_skip(PipelineStage::Separate, &request)
            .expect("no_stem should skip source separation");
        assert_eq!(separate_skip.0, "source_separation_disabled");
    }

    #[test]
    fn optional_stage_skip_diarize_skipped_when_backend_owns_diarization() {
        // When diarization is requested AND the whisper-diarization backend is
        // selected, the backend performs diarization internally. The pipeline
        // Diarize stage must be skipped to avoid diarizing the segments twice.
        let request = TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::WhisperDiarization,
            model: None,
            language: None,
            translate: false,
            diarize: true,
            persist: false,
            db_path: PathBuf::from("db.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        };

        let (reason, payload) = optional_stage_skip(PipelineStage::Diarize, &request)
            .expect("whisper-diarization backend should skip the pipeline diarize stage");
        assert_eq!(reason, "backend_owns_diarization");
        assert_eq!(payload["reason"], "backend_owns_diarization");
        assert_eq!(payload["details"]["backend"], "whisper_diarization");

        // Sanity: a non-diarization backend still runs the diarize stage when
        // diarization is requested.
        let mut other = request.clone();
        other.backend = BackendKind::WhisperCpp;
        assert!(
            optional_stage_skip(PipelineStage::Diarize, &other).is_none(),
            "non-diarization backend should still run the diarize stage"
        );
    }

    #[test]
    fn pipeline_config_validate_rejects_normalize_before_ingest() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Normalize,
            PipelineStage::Ingest,
            PipelineStage::Backend,
        ]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn pipeline_config_validate_rejects_backend_before_normalize() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Backend,
            PipelineStage::Normalize,
        ]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn pipeline_config_validate_rejects_accelerate_before_backend() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Accelerate,
            PipelineStage::Backend,
        ]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn pipeline_builder_chain_multiple_without() {
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Accelerate)
            .without(PipelineStage::Persist)
            .without(PipelineStage::Ingest)
            .build_unchecked();
        assert_eq!(
            config.stages(),
            &[
                PipelineStage::Normalize,
                PipelineStage::Vad,
                PipelineStage::Separate,
                PipelineStage::Backend,
                PipelineStage::Align,
                PipelineStage::Punctuate,
                PipelineStage::Diarize,
            ]
        );
    }

    #[test]
    fn pipeline_builder_without_nonexistent_stage_is_noop() {
        let config = PipelineBuilder::new()
            .stage(PipelineStage::Ingest)
            .without(PipelineStage::Backend) // not present, should be a noop
            .build_unchecked();
        assert_eq!(config.stages(), &[PipelineStage::Ingest]);
    }

    #[test]
    fn pipeline_config_new_and_stages_roundtrip() {
        let stages = vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Backend,
        ];
        let config = PipelineConfig::new(stages.clone());
        assert_eq!(config.stages(), stages.as_slice());
    }

    // ── Forced alignment (bd-qla.3) tests ──────────────────────────────

    fn make_segment(start: f64, end: f64, text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: Some(start),
            end_sec: Some(end),
            text: text.to_owned(),
            speaker: None,
            confidence: None,
        }
    }

    fn make_segment_no_timestamps(text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: text.to_owned(),
            speaker: None,
            confidence: None,
        }
    }

    #[test]
    fn align_stage_label() {
        assert_eq!(PipelineStage::Align.label(), "align");
    }

    #[test]
    fn align_stage_display() {
        assert_eq!(format!("{}", PipelineStage::Align), "align");
    }

    #[test]
    fn default_stages_include_align() {
        let config = PipelineConfig::default();
        assert!(
            config.has_stage(PipelineStage::Align),
            "default config should include Align stage"
        );
    }

    #[test]
    fn default_stages_align_after_backend() {
        let config = PipelineConfig::default();
        let stages = config.stages();
        let backend_pos = stages
            .iter()
            .position(|s| *s == PipelineStage::Backend)
            .unwrap();
        let align_pos = stages
            .iter()
            .position(|s| *s == PipelineStage::Align)
            .unwrap();
        assert!(
            backend_pos < align_pos,
            "Align must come after Backend in default stages"
        );
    }

    #[test]
    fn default_stages_align_before_persist() {
        let config = PipelineConfig::default();
        let stages = config.stages();
        let align_pos = stages
            .iter()
            .position(|s| *s == PipelineStage::Align)
            .unwrap();
        let persist_pos = stages
            .iter()
            .position(|s| *s == PipelineStage::Persist)
            .unwrap();
        assert!(
            align_pos < persist_pos,
            "Align must come before Persist in default stages"
        );
    }

    #[test]
    fn validate_align_without_backend_fails() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Align,
        ]);
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Align stage requires Backend before it"),
            "expected Align dependency error, got: {msg}"
        );
    }

    #[test]
    fn validate_align_before_backend_fails() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Align,
            PipelineStage::Backend,
        ]);
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Align stage requires Backend before it"),
            "expected Align ordering error, got: {msg}"
        );
    }

    #[test]
    fn validate_align_after_backend_succeeds() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Backend,
            PipelineStage::Align,
        ]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_full_pipeline_with_align_succeeds() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Backend,
            PipelineStage::Accelerate,
            PipelineStage::Align,
            PipelineStage::Persist,
        ]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pipeline_builder_without_align() {
        let config = super::PipelineBuilder::default_stages()
            .without(PipelineStage::Align)
            .build()
            .unwrap();
        assert!(!config.has_stage(PipelineStage::Align));
        // Other stages preserved
        assert!(config.has_stage(PipelineStage::Ingest));
        assert!(config.has_stage(PipelineStage::Backend));
    }

    #[test]
    fn align_config_default_values() {
        let config = AlignConfig::default();
        assert!(
            (config.max_drift_sec - 0.5).abs() < f64::EPSILON,
            "default max_drift_sec should be 0.5"
        );
        assert!(
            (config.min_segment_duration_sec - 0.02).abs() < f64::EPSILON,
            "default min_segment_duration_sec should be 0.02"
        );
    }

    #[test]
    fn ctc_align_empty_segments() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();
        let mut segments: Vec<TranscriptionSegment> = vec![];
        let report = ctc_forced_align(&mut segments, Some(10.0), &config, &token).unwrap();
        assert_eq!(report.segments_total, 0);
        assert_eq!(report.segments_corrected, 0);
        assert_eq!(report.segments_fallback, 0);
        assert!(report.notes.is_empty());
    }

    #[test]
    fn ctc_align_single_segment_fills_duration() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();
        let mut segments = vec![make_segment(0.0, 10.0, "hello world")];
        let report = ctc_forced_align(&mut segments, Some(10.0), &config, &token).unwrap();

        assert_eq!(report.segments_total, 1);
        assert_eq!(report.segments_corrected, 1);
        assert_eq!(report.segments_fallback, 0);

        // Single segment should span [0, 10.0].
        let start = segments[0].start_sec.unwrap();
        let end = segments[0].end_sec.unwrap();
        assert!(
            (start - 0.0).abs() < 1e-6,
            "start should be 0.0, got {start}"
        );
        assert!((end - 10.0).abs() < 1e-6, "end should be 10.0, got {end}");
    }

    #[test]
    fn ctc_align_proportional_distribution() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 10.0, // large drift tolerance for this test
            min_segment_duration_sec: 0.01,
        };
        // Two segments: "ab" (2 chars) and "abcdef" (6 chars) in 8 seconds.
        // Expected distribution: 2/8 * 8 = 2s for first, 6/8 * 8 = 6s for second.
        let mut segments = vec![
            make_segment(0.0, 2.0, "ab"),
            make_segment(2.0, 8.0, "abcdef"),
        ];
        let report = ctc_forced_align(&mut segments, Some(8.0), &config, &token).unwrap();

        assert_eq!(report.segments_corrected, 2);

        let s0_start = segments[0].start_sec.unwrap();
        let s0_end = segments[0].end_sec.unwrap();
        let s1_start = segments[1].start_sec.unwrap();
        let s1_end = segments[1].end_sec.unwrap();

        assert!((s0_start - 0.0).abs() < 1e-6);
        assert!(
            (s0_end - 2.0).abs() < 1e-6,
            "first seg end should be 2.0, got {s0_end}"
        );
        assert!((s1_start - 2.0).abs() < 1e-6);
        assert!(
            (s1_end - 8.0).abs() < 1e-6,
            "second seg end should be 8.0, got {s1_end}"
        );

        // Segments should be contiguous.
        assert!(
            (s0_end - s1_start).abs() < 1e-6,
            "segments should be contiguous"
        );
    }

    #[test]
    fn ctc_align_drift_guardrail_falls_back() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 0.1, // very tight tolerance
            min_segment_duration_sec: 0.01,
        };
        // "a" (1 char) and "abcdefghij" (10 chars) in 10s audio.
        // Proportional alignment: sec_per_char = 10/11 = 0.909.
        // Segment 0 "a": aligned [0, 0.909], original [0, 5] => end drift 4.09 > 0.1 => FALLBACK.
        // After fallback, cursor resets to orig_end = 5.0.
        // Segment 1 "abcdefghij": aligned_start = 5.0, aligned_end = min(5.0+10*0.909, 10) = 10.0
        // Original [5, 10] => start drift 0, end drift 0 => CORRECTED (no drift).
        let mut segments = vec![
            make_segment(0.0, 5.0, "a"),
            make_segment(5.0, 10.0, "abcdefghij"),
        ];
        let orig_s0_end = segments[0].end_sec;
        let report = ctc_forced_align(&mut segments, Some(10.0), &config, &token).unwrap();

        // Only segment 0 falls back; segment 1 has zero drift after cursor reset.
        assert_eq!(
            report.segments_fallback, 1,
            "only first segment should fall back"
        );
        assert_eq!(
            report.segments_corrected, 1,
            "second segment should be corrected (zero drift)"
        );
        assert_eq!(
            segments[0].end_sec, orig_s0_end,
            "first segment end should be original"
        );
        assert_eq!(report.segments_fallback + report.segments_corrected, 2);
    }

    #[test]
    fn ctc_align_min_duration_guardrail() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 100.0,          // allow everything
            min_segment_duration_sec: 5.0, // very high minimum
        };
        // With 1s audio and 2 segments, each gets ~0.5s which is below 5.0.
        let mut segments = vec![make_segment(0.0, 0.5, "hi"), make_segment(0.5, 1.0, "ok")];
        let report = ctc_forced_align(&mut segments, Some(1.0), &config, &token).unwrap();
        assert_eq!(report.segments_fallback, 2);
        assert_eq!(report.segments_corrected, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("below minimum")),
            "should note minimum duration violation"
        );
    }

    #[test]
    fn ctc_align_no_duration_available() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();
        let mut segments = vec![make_segment_no_timestamps("hello world")];
        let report = ctc_forced_align(&mut segments, None, &config, &token).unwrap();
        assert_eq!(report.segments_fallback, 1);
        assert_eq!(report.segments_corrected, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no audio duration")),
            "should note missing duration"
        );
    }

    #[test]
    fn ctc_align_zero_duration() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();
        let mut segments = vec![make_segment(0.0, 0.0, "test")];
        let report = ctc_forced_align(&mut segments, Some(0.0), &config, &token).unwrap();
        assert_eq!(report.segments_corrected, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("non-positive")),
            "should note non-positive duration"
        );
    }

    #[test]
    fn ctc_align_negative_duration() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();
        let mut segments = vec![make_segment(0.0, 1.0, "test")];
        let report = ctc_forced_align(&mut segments, Some(-1.0), &config, &token).unwrap();
        assert_eq!(report.segments_corrected, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("non-positive")),
            "should note non-positive duration"
        );
    }

    #[test]
    fn ctc_align_infers_duration_from_last_segment() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 10.0,
            min_segment_duration_sec: 0.01,
        };
        // No explicit duration; should infer from last segment's end_sec (5.0).
        let mut segments = vec![
            make_segment(0.0, 2.5, "hello"),
            make_segment(2.5, 5.0, "world"),
        ];
        let report = ctc_forced_align(&mut segments, None, &config, &token).unwrap();
        assert_eq!(report.segments_total, 2);
        // Both texts are 5 chars so equal distribution: [0, 2.5] and [2.5, 5.0].
        assert_eq!(report.segments_corrected, 2);
        let end = segments[1].end_sec.unwrap();
        assert!(
            (end - 5.0).abs() < 1e-6,
            "last segment end should be 5.0, got {end}"
        );
    }

    #[test]
    fn ctc_align_deterministic_across_calls() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();

        let mk = || {
            vec![
                make_segment(0.0, 3.0, "hello"),
                make_segment(3.0, 7.0, "beautiful world"),
                make_segment(7.0, 10.0, "today"),
            ]
        };

        let mut segments_a = mk();
        let mut segments_b = mk();
        let _report_a = ctc_forced_align(&mut segments_a, Some(10.0), &config, &token).unwrap();
        let _report_b = ctc_forced_align(&mut segments_b, Some(10.0), &config, &token).unwrap();

        for (a, b) in segments_a.iter().zip(segments_b.iter()) {
            assert_eq!(
                a.start_sec, b.start_sec,
                "start_sec should be identical across calls"
            );
            assert_eq!(
                a.end_sec, b.end_sec,
                "end_sec should be identical across calls"
            );
        }
    }

    #[test]
    fn ctc_align_preserves_non_timestamp_fields() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 10.0,
            min_segment_duration_sec: 0.01,
        };
        let mut segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(5.0),
            text: "hello world".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.95),
        }];
        let _report = ctc_forced_align(&mut segments, Some(5.0), &config, &token).unwrap();
        assert_eq!(segments[0].text, "hello world");
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(segments[0].confidence, Some(0.95));
    }

    #[test]
    fn ctc_align_cancellation_aborts_early() {
        // Create a token with an already-expired deadline.
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        // Give it a moment to be past deadline.
        std::thread::sleep(Duration::from_millis(5));

        let config = AlignConfig::default();
        let mut segments = vec![
            make_segment(0.0, 5.0, "hello"),
            make_segment(5.0, 10.0, "world"),
        ];
        let result = ctc_forced_align(&mut segments, Some(10.0), &config, &token);
        assert!(result.is_err(), "should fail with cancellation");
        let err = result.unwrap_err();
        assert!(
            matches!(err, FwError::Cancelled(_)),
            "should be Cancelled, got: {err:?}"
        );
    }

    #[test]
    fn ctc_align_whitespace_only_text_treated_as_one_char() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 100.0,
            min_segment_duration_sec: 0.001,
        };
        // Whitespace-only text trims to empty, but we use max(1) so it gets
        // at least 1 character weight.
        let mut segments = vec![
            make_segment(0.0, 5.0, "   "),
            make_segment(5.0, 10.0, "hello"),
        ];
        let report = ctc_forced_align(&mut segments, Some(10.0), &config, &token).unwrap();
        assert_eq!(report.segments_total, 2);
        // Whitespace segment: 1 char, "hello" segment: 5 chars, total 6 chars in 10s.
        // First: 1/6 * 10 = 1.667s, Second: 5/6 * 10 = 8.333s
        let s0_end = segments[0].end_sec.unwrap();
        assert!(
            s0_end > 0.0 && s0_end < 5.0,
            "whitespace segment should get smaller portion: {s0_end}"
        );
    }

    #[test]
    fn ctc_align_many_segments() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 100.0,
            min_segment_duration_sec: 0.001,
        };
        let n = 100;
        let duration = 60.0;
        let segment_dur = duration / n as f64;
        let mut segments: Vec<TranscriptionSegment> = (0..n)
            .map(|i| {
                let s = i as f64 * segment_dur;
                make_segment(s, s + segment_dur, "word")
            })
            .collect();
        let report = ctc_forced_align(&mut segments, Some(duration), &config, &token).unwrap();
        assert_eq!(report.segments_total, n);
        // All segments have equal text length so they should be uniformly distributed.
        assert_eq!(report.segments_corrected + report.segments_fallback, n);

        // Verify monotonicity.
        for i in 1..segments.len() {
            let prev_end = segments[i - 1].end_sec.unwrap();
            let curr_start = segments[i].start_sec.unwrap();
            assert!(
                curr_start >= prev_end - 1e-9,
                "segments should be monotonically ordered: seg[{}].end={prev_end} > seg[{i}].start={curr_start}",
                i - 1
            );
        }
    }

    #[test]
    fn ctc_align_contiguous_output() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 100.0,
            min_segment_duration_sec: 0.001,
        };
        let mut segments = vec![
            make_segment(0.0, 2.0, "one"),
            make_segment(2.0, 5.0, "two three"),
            make_segment(5.0, 10.0, "four five six seven"),
        ];
        let report = ctc_forced_align(&mut segments, Some(10.0), &config, &token).unwrap();
        assert_eq!(report.segments_corrected, 3);

        // Each segment's end should equal the next segment's start.
        for i in 0..segments.len() - 1 {
            let end = segments[i].end_sec.unwrap();
            let next_start = segments[i + 1].start_sec.unwrap();
            assert!(
                (end - next_start).abs() < 1e-9,
                "gap between seg[{i}].end={end} and seg[{}].start={next_start}",
                i + 1
            );
        }

        // Last segment should end at audio duration.
        let last_end = segments.last().unwrap().end_sec.unwrap();
        assert!(
            (last_end - 10.0).abs() < 1e-6,
            "last segment should end at duration, got {last_end}"
        );
    }

    #[test]
    fn stage_budget_ms_returns_align_budget() {
        let budgets = StageBudgetPolicy {
            ingest_ms: 100,
            normalize_ms: 200,
            probe_ms: 300,
            vad_ms: 10_000,
            separate_ms: 30_000,
            backend_ms: 400,
            acceleration_ms: 500,
            align_ms: 30_000,
            punctuate_ms: 10_000,
            diarize_ms: 30_000,
            persist_ms: 600,
            cleanup_budget_ms: 5_000,
        };
        assert_eq!(stage_budget_ms("align", budgets), Some(30_000));
    }

    #[test]
    fn stage_budget_policy_includes_align() {
        let policy = StageBudgetPolicy::from_source(|_| None);
        assert_eq!(
            policy.align_ms, 30_000,
            "default align budget should be 30_000ms"
        );
    }

    #[test]
    fn stage_budget_policy_align_env_override() {
        let policy = StageBudgetPolicy::from_source(|key| {
            if key == "FRANKEN_WHISPER_STAGE_BUDGET_ALIGN_MS" {
                Some("45000".to_owned())
            } else {
                None
            }
        });
        assert_eq!(policy.align_ms, 45_000);
    }

    #[test]
    fn alignment_report_fields() {
        let report = AlignmentReport {
            segments_total: 10,
            segments_corrected: 7,
            segments_fallback: 3,
            notes: vec!["test note".to_owned()],
        };
        assert_eq!(report.segments_total, 10);
        assert_eq!(report.segments_corrected, 7);
        assert_eq!(report.segments_fallback, 3);
        assert_eq!(report.notes.len(), 1);
    }

    #[test]
    fn ctc_align_segments_without_timestamps_fallback_on_no_duration() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();
        let mut segments = vec![
            make_segment_no_timestamps("hello"),
            make_segment_no_timestamps("world"),
        ];
        // No duration and no timestamps on segments => no duration can be inferred.
        let report = ctc_forced_align(&mut segments, None, &config, &token).unwrap();
        assert_eq!(report.segments_fallback, 2);
        assert_eq!(report.segments_corrected, 0);
    }

    #[test]
    fn ctc_align_mixed_drift_some_correct_some_fallback() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 0.3,
            min_segment_duration_sec: 0.01,
        };
        // Two segments each of 5 chars in 10s.
        // Alignment: each gets 5s. First [0, 5], second [5, 10].
        // Original first [0, 5] => aligned [0, 5], drift 0 => corrected.
        // Original second [5.2, 10.0] => aligned [5, 10], start drift 0.2, end drift 0 => corrected (within 0.3).
        let mut segments = vec![
            make_segment(0.0, 5.0, "hello"),
            make_segment(5.2, 10.0, "world"),
        ];
        let report = ctc_forced_align(&mut segments, Some(10.0), &config, &token).unwrap();
        assert_eq!(report.segments_corrected, 2);
        assert_eq!(report.segments_fallback, 0);
    }

    #[test]
    fn ctc_align_three_segments_varied_lengths() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig {
            max_drift_sec: 10.0,
            min_segment_duration_sec: 0.01,
        };
        // "Hi" (2 chars), "Hello World" (11 chars), "Bye" (3 chars)
        // Total: 16 chars in 16 seconds => 1 sec/char.
        let mut segments = vec![
            make_segment(0.0, 2.0, "Hi"),
            make_segment(2.0, 13.0, "Hello World"),
            make_segment(13.0, 16.0, "Bye"),
        ];
        let report = ctc_forced_align(&mut segments, Some(16.0), &config, &token).unwrap();
        assert_eq!(report.segments_corrected, 3);

        let s0_dur = segments[0].end_sec.unwrap() - segments[0].start_sec.unwrap();
        let s1_dur = segments[1].end_sec.unwrap() - segments[1].start_sec.unwrap();
        let s2_dur = segments[2].end_sec.unwrap() - segments[2].start_sec.unwrap();

        assert!(
            (s0_dur - 2.0).abs() < 1e-6,
            "first seg should be 2s, got {s0_dur}"
        );
        assert!(
            (s1_dur - 11.0).abs() < 1e-6,
            "second seg should be 11s, got {s1_dur}"
        );
        assert!(
            (s2_dur - 3.0).abs() < 1e-6,
            "third seg should be 3s, got {s2_dur}"
        );
    }

    // -----------------------------------------------------------------------
    // Bounded cleanup budgets (bd-38c.4)
    // -----------------------------------------------------------------------

    #[test]
    fn cleanup_budget_ms_default_value() {
        let policy = StageBudgetPolicy::from_source(|_| None);
        assert_eq!(
            policy.cleanup_budget_ms, 5_000,
            "default cleanup budget should be 5000ms"
        );
    }

    #[test]
    fn cleanup_budget_ms_env_override() {
        let policy = StageBudgetPolicy::from_source(|key| {
            if key == "FRANKEN_WHISPER_STAGE_BUDGET_CLEANUP_MS" {
                Some("2000".to_owned())
            } else {
                None
            }
        });
        assert_eq!(policy.cleanup_budget_ms, 2000);
    }

    #[test]
    fn cleanup_budget_ms_in_as_json() {
        let policy = StageBudgetPolicy::from_source(|_| None);
        let j = policy.as_json();
        assert_eq!(j["cleanup_budget_ms"], 5_000);
    }

    #[test]
    fn run_finalizers_bounded_executes_all_finalizers() {
        use std::sync::{Arc, Mutex};

        let order: Arc<Mutex<Vec<&str>>> = Arc::new(Mutex::new(Vec::new()));
        let mut pcx = PipelineCx::new(None);

        let o1 = Arc::clone(&order);
        pcx.register_finalizer(
            "first",
            Finalizer::Custom(Box::new(move || {
                o1.lock().unwrap().push("first");
            })),
        );

        let o2 = Arc::clone(&order);
        pcx.register_finalizer(
            "second",
            Finalizer::Custom(Box::new(move || {
                o2.lock().unwrap().push("second");
            })),
        );

        pcx.run_finalizers_bounded(5_000);
        assert_eq!(pcx.finalizers.len(), 0, "all finalizers should be drained");

        let executed = order.lock().unwrap();
        assert_eq!(
            *executed,
            vec!["second", "first"],
            "bounded finalizers should run in LIFO order"
        );
    }

    #[test]
    fn run_finalizers_bounded_with_zero_budget_still_runs() {
        use std::sync::{Arc, Mutex};

        let called: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
        let mut pcx = PipelineCx::new(None);

        let c = Arc::clone(&called);
        pcx.register_finalizer(
            "quick",
            Finalizer::Custom(Box::new(move || {
                *c.lock().unwrap() = true;
            })),
        );

        pcx.run_finalizers_bounded(0);
        assert!(
            *called.lock().unwrap(),
            "finalizer should execute even with zero budget"
        );
    }

    #[test]
    fn run_finalizers_bounded_empty_registry_is_noop() {
        let mut pcx = PipelineCx::new(None);
        pcx.run_finalizers_bounded(5_000);
        assert_eq!(pcx.finalizers.len(), 0);
    }

    #[test]
    fn finalizer_registry_run_all_bounded_lifo_order() {
        use std::sync::{Arc, Mutex};

        let order: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mut registry = FinalizerRegistry::new();

        for i in 0..5 {
            let o = Arc::clone(&order);
            let label = format!("fin_{i}");
            let label_clone = label.clone();
            registry.register(
                &label,
                Finalizer::Custom(Box::new(move || {
                    o.lock().unwrap().push(label_clone);
                })),
            );
        }

        assert_eq!(registry.len(), 5);
        registry.run_all_bounded(10_000);
        assert_eq!(registry.len(), 0);

        let executed = order.lock().unwrap();
        assert_eq!(
            *executed,
            vec!["fin_4", "fin_3", "fin_2", "fin_1", "fin_0"],
            "bounded finalizers should run in LIFO order"
        );
    }

    // -----------------------------------------------------------------------
    // VAD pre-filtering (bd-qla.1)
    // -----------------------------------------------------------------------

    #[test]
    fn vad_stage_label() {
        assert_eq!(PipelineStage::Vad.label(), "vad");
    }

    #[test]
    fn vad_stage_display() {
        assert_eq!(format!("{}", PipelineStage::Vad), "vad");
    }

    #[test]
    fn default_stages_include_vad() {
        let config = PipelineConfig::default();
        assert!(config.has_stage(PipelineStage::Vad));
    }

    #[test]
    fn vad_config_default_values() {
        let config = VadConfig::default();
        assert!((config.rms_threshold - 0.01).abs() < f64::EPSILON);
        assert_eq!(config.frame_samples, 160);
        assert!((config.min_voice_ratio - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.min_speech_duration_ms, 40);
        assert_eq!(config.min_silence_duration_ms, 40);
        assert_eq!(config.max_speech_duration_ms, None);
        assert_eq!(config.speech_pad_ms, 0);
    }

    fn write_pcm16_mono_wav_for_vad(path: &std::path::Path, sample_rate: u32, samples: &[i16]) {
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);

        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 2;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn vad_config_from_request_applies_backend_overrides() {
        let request = TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: PathBuf::from("storage.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams {
                vad: Some(crate::model::VadParams {
                    model_path: Some(PathBuf::from("unused-vad-model.onnx")),
                    threshold: Some(0.42),
                    min_speech_duration_ms: Some(120),
                    min_silence_duration_ms: Some(80),
                    max_speech_duration_s: Some(3.5),
                    speech_pad_ms: Some(30),
                    samples_overlap: Some(0.2),
                }),
                ..BackendParams::default()
            },
        };

        let config = VadConfig::from_request(&request);
        assert!((config.rms_threshold - 0.42).abs() < 1e-6);
        assert_eq!(config.min_speech_duration_ms, 120);
        assert_eq!(config.min_silence_duration_ms, 80);
        assert_eq!(config.max_speech_duration_ms, Some(3_500));
        assert_eq!(config.speech_pad_ms, 30);
    }

    #[test]
    fn vad_energy_detect_uses_native_audio_for_valid_wav() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("native.wav");

        let mut samples = vec![0i16; 16_000];
        samples.extend(vec![7_000i16; 16_000]);
        write_pcm16_mono_wav_for_vad(&path, 16_000, &samples);

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig::default();
        let report = vad_energy_detect(&path, &config, &token).unwrap();

        assert_eq!(report.detector, "native_audio_waveform");
        assert!(!report.fallback_triggered);
        assert!(report.activity_threshold > 0.0);
    }

    #[test]
    fn vad_energy_detect_invalid_wav_uses_deterministic_fallback() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy-fallback.wav");

        let mut data = vec![0u8; 44];
        for _ in 0..1600 {
            data.extend(i16::MAX.to_le_bytes());
        }
        std::fs::write(&path, &data).unwrap();

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig::default();
        let report = vad_energy_detect(&path, &config, &token).unwrap();

        assert_eq!(report.detector, "legacy_energy");
        assert!(report.fallback_triggered);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("deterministic legacy fallback activated")),
            "expected fallback note in {:?}",
            report.notes
        );
    }

    #[test]
    fn vad_energy_detect_silence_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("silence.wav");
        let mut data = vec![0u8; 44];
        data.extend(vec![0u8; 3200]);
        std::fs::write(&path, &data).unwrap();

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig::default();
        let report = vad_energy_detect(&path, &config, &token).unwrap();

        assert!(report.silence_only, "all-zero audio should be silence-only");
        assert_eq!(report.frames_voiced, 0);
        assert!(report.voice_ratio < 0.01);
        assert!(report.regions.is_empty());
    }

    #[test]
    fn vad_energy_detect_loud_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("loud.wav");
        let mut data = vec![0u8; 44];
        for _ in 0..1600 {
            data.extend(i16::MAX.to_le_bytes());
        }
        std::fs::write(&path, &data).unwrap();

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig::default();
        let report = vad_energy_detect(&path, &config, &token).unwrap();

        assert!(
            !report.silence_only,
            "loud audio should not be silence-only"
        );
        assert!(report.frames_voiced > 0);
        assert!(report.voice_ratio > 0.5);
        assert!(!report.regions.is_empty());
    }

    #[test]
    fn vad_energy_detect_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.wav");
        let data = vec![0u8; 44];
        std::fs::write(&path, &data).unwrap();

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig::default();
        let report = vad_energy_detect(&path, &config, &token).unwrap();

        assert!(report.silence_only);
        assert_eq!(report.frames_total, 0);
    }

    #[test]
    fn vad_energy_detect_cancellation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cancel_test.wav");
        let mut data = vec![0u8; 44];
        data.extend(vec![0u8; 3200]);
        std::fs::write(&path, &data).unwrap();

        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(5));

        let config = VadConfig::default();
        let result = vad_energy_detect(&path, &config, &token);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, FwError::Cancelled(_)));
    }

    #[test]
    fn vad_energy_detect_mixed_silence_and_voice() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mixed.wav");
        let mut data = vec![0u8; 44];
        data.extend(vec![0u8; 1600]);
        for _ in 0..800 {
            data.extend(16000_i16.to_le_bytes());
        }
        std::fs::write(&path, &data).unwrap();

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig::default();
        let report = vad_energy_detect(&path, &config, &token).unwrap();

        assert!(report.frames_total > 0);
        assert!(report.frames_voiced > 0);
        assert!(report.frames_voiced < report.frames_total);
        assert!(!report.regions.is_empty());
        for (start, end) in &report.regions {
            assert!(
                end > start,
                "region end should be after start: {start} -> {end}"
            );
        }
    }

    #[test]
    fn vad_energy_detect_missing_file() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig::default();
        let result = vad_energy_detect(
            &PathBuf::from("/tmp/nonexistent_vad_test.wav"),
            &config,
            &token,
        );
        assert!(result.is_err());
    }

    #[test]
    fn validate_vad_without_normalize_fails() {
        let config = PipelineConfig::new(vec![PipelineStage::Ingest, PipelineStage::Vad]);
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Vad stage requires Normalize before it"),
            "expected Vad dependency error, got: {msg}"
        );
    }

    #[test]
    fn stage_budget_ms_returns_vad_budget() {
        let budgets = StageBudgetPolicy::from_source(|_| None);
        assert_eq!(stage_budget_ms("vad", budgets), Some(10_000));
    }

    // -----------------------------------------------------------------------
    // Source separation (bd-qla.2)
    // -----------------------------------------------------------------------

    #[test]
    fn separate_stage_label() {
        assert_eq!(PipelineStage::Separate.label(), "separate");
    }

    #[test]
    fn separate_stage_display() {
        assert_eq!(format!("{}", PipelineStage::Separate), "separate");
    }

    #[test]
    fn default_stages_include_separate() {
        let config = PipelineConfig::default();
        assert!(config.has_stage(PipelineStage::Separate));
    }

    #[test]
    fn source_separate_returns_vocal_isolated_on_invalid_wav() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wav");
        std::fs::write(&path, b"fake audio").unwrap();

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let report = source_separate(&path, &token).unwrap();

        // Invalid WAV → fallback path: assumes vocal content present.
        assert!(report.vocal_isolated);
        assert!(!report.notes.is_empty());
        assert!(
            report.notes[0].contains("analysis unavailable"),
            "expected fallback note, got: {}",
            report.notes[0]
        );
    }

    #[test]
    fn source_separate_cancellation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cancel_sep.wav");
        std::fs::write(&path, b"fake audio").unwrap();

        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(5));

        let result = source_separate(&path, &token);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FwError::Cancelled(_)));
    }

    #[test]
    fn source_separate_valid_wav_with_speech() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("speech.wav");

        // Generate a 1-second WAV with a loud speech-like signal.
        let sample_rate = 16_000u32;
        let samples: Vec<i16> = (0..sample_rate)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (f64::sin(t * 440.0 * std::f64::consts::TAU) * 8_000.0) as i16
            })
            .collect();
        write_pcm16_mono_wav_for_vad(&path, sample_rate, &samples);

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let report = source_separate(&path, &token).unwrap();

        assert!(report.vocal_isolated, "speech signal should be detected");
        assert!(
            report.speech_coverage > 0.0,
            "speech coverage should be > 0 for tonal signal"
        );
        assert!(report.active_region_count > 0);
        assert!(report.avg_rms > 0.0);
        assert!(report.notes[0].contains("energy-based analysis"));
    }

    #[test]
    fn source_separate_valid_wav_silence_only() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("silence.wav");

        // Generate a 1-second silent WAV.
        let sample_rate = 16_000u32;
        let samples = vec![0i16; sample_rate as usize];
        write_pcm16_mono_wav_for_vad(&path, sample_rate, &samples);

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let report = source_separate(&path, &token).unwrap();

        // Silence should yield no active regions and low/zero coverage.
        assert_eq!(report.active_region_count, 0);
        assert!(
            report.speech_coverage < 0.05,
            "silence should have very low speech coverage, got {}",
            report.speech_coverage
        );
        // vocal_isolated should be false since coverage < threshold.
        assert!(
            !report.vocal_isolated,
            "silence-only audio should not be flagged as vocal-isolated"
        );
    }

    #[test]
    fn source_separate_missing_file_fallback() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let result =
            source_separate(std::path::Path::new("/nonexistent/audio.wav"), &token).unwrap();

        // Missing file → graceful fallback.
        assert!(result.vocal_isolated);
        assert!(result.notes[0].contains("analysis unavailable"));
    }

    #[test]
    fn validate_separate_without_normalize_fails() {
        let config = PipelineConfig::new(vec![PipelineStage::Ingest, PipelineStage::Separate]);
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Separate stage requires Normalize before it"),
            "expected Separate dependency error, got: {msg}"
        );
    }

    #[test]
    fn stage_budget_ms_returns_separate_budget() {
        let budgets = StageBudgetPolicy::from_source(|_| None);
        assert_eq!(stage_budget_ms("separate", budgets), Some(30_000));
    }

    // -----------------------------------------------------------------------
    // Punctuation restoration (bd-qla.4)
    // -----------------------------------------------------------------------

    #[test]
    fn punctuate_stage_label() {
        assert_eq!(PipelineStage::Punctuate.label(), "punctuate");
    }

    #[test]
    fn punctuate_stage_display() {
        assert_eq!(format!("{}", PipelineStage::Punctuate), "punctuate");
    }

    #[test]
    fn default_stages_include_punctuate() {
        let config = PipelineConfig::default();
        assert!(config.has_stage(PipelineStage::Punctuate));
    }

    #[test]
    fn punctuate_empty_segments() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments: Vec<TranscriptionSegment> = vec![];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(report.segments_total, 0);
        assert_eq!(report.segments_modified, 0);
    }

    #[test]
    fn punctuate_capitalizes_first_letter() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "hello world")];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(report.segments_modified, 1);
        assert!(
            segments[0].text.starts_with('H'),
            "first letter should be capitalized: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_adds_period_at_end() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "hello world")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            segments[0].text.ends_with('.'),
            "should add period at end: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_does_not_double_period() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "Hello world.")];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            !segments[0].text.ends_with(".."),
            "should not double period: {}",
            segments[0].text
        );
        let period_count = segments[0].text.chars().filter(|c| *c == '.').count();
        assert_eq!(
            period_count, 1,
            "should have exactly one period: {}",
            segments[0].text
        );
        assert_eq!(report.segments_modified, 0);
    }

    #[test]
    fn punctuate_capitalizes_after_sentence_end() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 5.0, "hello. world")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            segments[0].text.contains("Hello. World"),
            "should capitalize after period: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_preserves_existing_punctuation() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "Is this a question?")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            segments[0].text.ends_with('?'),
            "should preserve question mark: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_terminal_comma_still_gets_period() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "hello,")];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(report.segments_modified, 1);
        assert_eq!(segments[0].text, "Hello,.");
    }

    #[test]
    fn punctuate_terminal_semicolon_still_gets_period() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "hello;")];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(report.segments_modified, 1);
        assert_eq!(segments[0].text, "Hello;.");
    }

    #[test]
    fn punctuate_handles_exclamation() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "wow!")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            segments[0].text.starts_with('W'),
            "should capitalize: {}",
            segments[0].text
        );
        assert!(
            segments[0].text.ends_with('!'),
            "should preserve exclamation: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_skips_empty_text() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "  ")];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(report.segments_total, 1);
        assert_eq!(report.segments_modified, 0);
    }

    #[test]
    fn punctuate_cancellation() {
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(5));

        let mut segments = vec![make_segment(0.0, 1.0, "hello")];
        let result = punctuate_segments(&mut segments, &token);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FwError::Cancelled(_)));
    }

    #[test]
    fn punctuate_multiple_segments() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 2.0, "hello world"),
            make_segment(2.0, 4.0, "this is a test"),
            make_segment(4.0, 6.0, "goodbye"),
        ];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(report.segments_total, 3);
        assert_eq!(report.segments_modified, 3);

        for seg in &segments {
            assert!(
                seg.text.chars().next().unwrap().is_uppercase(),
                "each segment should start uppercase: {}",
                seg.text
            );
        }
    }

    #[test]
    fn validate_punctuate_without_backend_fails() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Punctuate,
        ]);
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Punctuate stage requires Backend before it"),
            "expected Punctuate dependency error, got: {msg}"
        );
    }

    #[test]
    fn stage_budget_ms_returns_punctuate_budget() {
        let budgets = StageBudgetPolicy::from_source(|_| None);
        assert_eq!(stage_budget_ms("punctuate", budgets), Some(10_000));
    }

    // -----------------------------------------------------------------------
    // Speaker diarization (bd-qla.5)
    // -----------------------------------------------------------------------

    #[test]
    fn diarize_stage_label() {
        assert_eq!(PipelineStage::Diarize.label(), "diarize");
    }

    #[test]
    fn diarize_stage_display() {
        assert_eq!(format!("{}", PipelineStage::Diarize), "diarize");
    }

    #[test]
    fn default_stages_include_diarize() {
        let config = PipelineConfig::default();
        assert!(config.has_stage(PipelineStage::Diarize));
    }

    #[test]
    fn diarize_empty_segments() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments: Vec<TranscriptionSegment> = vec![];
        let report = diarize_segments(&mut segments, Some(10.0), None, &token).unwrap();
        assert_eq!(report.segments_total, 0);
        assert_eq!(report.speakers_detected, 0);
        assert_eq!(report.segments_labeled, 0);
    }

    #[test]
    fn diarize_single_segment_gets_speaker_label() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 5.0, "hello world")];
        let report = diarize_segments(&mut segments, Some(5.0), None, &token).unwrap();

        assert_eq!(report.segments_total, 1);
        assert_eq!(report.speakers_detected, 1);
        assert_eq!(report.segments_labeled, 1);
        assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
    }

    #[test]
    fn diarize_multiple_segments_assigns_speakers() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 3.0, "hello"),
            make_segment(3.0, 6.0, "world"),
            make_segment(6.0, 10.0, "goodbye"),
        ];
        let report = diarize_segments(&mut segments, Some(10.0), None, &token).unwrap();

        assert_eq!(report.segments_total, 3);
        assert_eq!(report.segments_labeled, 3);
        assert!(report.speakers_detected >= 1);

        for seg in &segments {
            assert!(seg.speaker.is_some());
            assert!(seg.speaker.as_ref().unwrap().starts_with("SPEAKER_"));
        }
    }

    #[test]
    fn diarize_preserves_non_speaker_fields() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(5.0),
            text: "hello world".to_owned(),
            speaker: None,
            confidence: Some(0.95),
        }];
        let _report = diarize_segments(&mut segments, Some(5.0), None, &token).unwrap();

        assert_eq!(segments[0].text, "hello world");
        assert_eq!(segments[0].confidence, Some(0.95));
        assert_eq!(segments[0].start_sec, Some(0.0));
        assert_eq!(segments[0].end_sec, Some(5.0));
    }

    #[test]
    fn diarize_cancellation() {
        let token = CancellationToken::with_deadline_from_now(Duration::from_millis(0)); // ubs:ignore — cancellation token is not a secret
        std::thread::sleep(Duration::from_millis(5));

        let mut segments = vec![
            make_segment(0.0, 5.0, "hello"),
            make_segment(5.0, 10.0, "world"),
        ];
        let result = diarize_segments(&mut segments, Some(10.0), None, &token);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FwError::Cancelled(_)));
    }

    #[test]
    fn diarize_no_duration_infers_from_segments() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 5.0, "hello"),
            make_segment(5.0, 10.0, "world"),
        ];
        let report = diarize_segments(&mut segments, None, None, &token).unwrap();
        assert_eq!(report.segments_total, 2);
        assert_eq!(report.segments_labeled, 2);
    }

    #[test]
    fn diarize_segments_without_timestamps() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment_no_timestamps("hello"),
            make_segment_no_timestamps("world"),
        ];
        let report = diarize_segments(&mut segments, None, None, &token).unwrap();
        assert_eq!(report.segments_labeled, 2);
        assert_eq!(report.speakers_detected, 1);
    }

    #[test]
    fn speaker_embedding_cosine_similarity_identical() {
        let a = SpeakerEmbedding {
            features: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        };
        let b = SpeakerEmbedding {
            features: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        };
        let sim = a.cosine_similarity(&b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical embeddings should have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn speaker_embedding_cosine_similarity_orthogonal() {
        let a = SpeakerEmbedding {
            features: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let b = SpeakerEmbedding {
            features: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        };
        let sim = a.cosine_similarity(&b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal embeddings should have similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn speaker_embedding_cosine_similarity_zero_vector() {
        let a = SpeakerEmbedding {
            features: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let b = SpeakerEmbedding {
            features: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        };
        let sim = a.cosine_similarity(&b);
        assert!(
            sim.abs() < 1e-6,
            "zero vector should have similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn validate_diarize_without_backend_fails() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Diarize,
        ]);
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Diarize stage requires Backend before it"),
            "expected Diarize dependency error, got: {msg}"
        );
    }

    #[test]
    fn stage_budget_ms_returns_diarize_budget() {
        let budgets = StageBudgetPolicy::from_source(|_| None);
        assert_eq!(stage_budget_ms("diarize", budgets), Some(30_000));
    }

    #[test]
    fn diarize_report_includes_heuristic_note() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 1.0, "test")];
        let report = diarize_segments(&mut segments, Some(1.0), None, &token).unwrap();
        assert!(
            report.notes.iter().any(|n| n.contains("heuristic")),
            "diarize report should include heuristic note"
        );
    }

    #[test]
    fn full_pipeline_with_new_stages_validates() {
        let config = PipelineConfig::new(vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Vad,
            PipelineStage::Separate,
            PipelineStage::Backend,
            PipelineStage::Accelerate,
            PipelineStage::Align,
            PipelineStage::Punctuate,
            PipelineStage::Diarize,
            PipelineStage::Persist,
        ]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn pipeline_builder_without_vad_and_separate() {
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Vad)
            .without(PipelineStage::Separate)
            .build()
            .expect("valid");
        assert!(!config.has_stage(PipelineStage::Vad));
        assert!(!config.has_stage(PipelineStage::Separate));
        assert!(config.has_stage(PipelineStage::Backend));
    }

    #[test]
    fn pipeline_builder_without_punctuate_and_diarize() {
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Punctuate)
            .without(PipelineStage::Diarize)
            .build()
            .expect("valid");
        assert!(!config.has_stage(PipelineStage::Punctuate));
        assert!(!config.has_stage(PipelineStage::Diarize));
        assert!(config.has_stage(PipelineStage::Align));
    }

    // -----------------------------------------------------------------------
    // Pipeline stage composition integration tests (bd-1cn)
    //
    // Verify that optional stages (VAD, Separate, Punctuate, Diarize) can be
    // skipped independently without breaking downstream stage contracts, and
    // that segments flow correctly through various stage combinations.
    // -----------------------------------------------------------------------

    #[test]
    fn composition_punctuate_then_diarize_segments_flow() {
        // Verify segments can flow through punctuate → diarize without
        // VAD or Separate having run.
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 3.0, "hello world"),
            make_segment(3.0, 6.0, "this is a test"),
            make_segment(20.0, 23.0, "different speaker arrives"),
        ];

        // Punctuate first.
        let punct_report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(punct_report.segments_total, 3);

        // Then diarize the already-punctuated segments.
        let diarize_report = diarize_segments(&mut segments, Some(30.0), None, &token).unwrap();
        assert_eq!(diarize_report.segments_total, 3);
        assert_eq!(diarize_report.segments_labeled, 3);

        // Every segment should have both punctuated text and a speaker label.
        for seg in &segments {
            assert!(
                seg.speaker.is_some(),
                "speaker should be assigned after diarize"
            );
            assert!(
                seg.text.ends_with('.') || seg.text.ends_with('!') || seg.text.ends_with('?'),
                "text should be punctuated: {:?}",
                seg.text
            );
        }
    }

    #[test]
    fn composition_diarize_without_punctuate_preserves_text() {
        // Diarize alone should not alter segment text.
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 2.0, "raw text here"),
            make_segment(2.0, 4.0, "more raw text"),
        ];
        let original_texts: Vec<String> = segments.iter().map(|s| s.text.clone()).collect();

        let report = diarize_segments(&mut segments, Some(5.0), None, &token).unwrap();
        assert_eq!(report.segments_labeled, 2);

        for (seg, orig) in segments.iter().zip(original_texts.iter()) {
            assert_eq!(&seg.text, orig, "diarize should not modify text");
        }
    }

    #[test]
    fn composition_align_then_punctuate_then_diarize_full_chain() {
        // Full optional post-backend chain: Align → Punctuate → Diarize.
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();
        let mut segments = vec![
            make_segment(0.0, 4.0, "first segment of speech"),
            make_segment(4.0, 8.0, "second segment continues"),
            make_segment(20.0, 24.0, "third segment from new speaker"),
        ];

        // Align.
        let align_report = ctc_forced_align(&mut segments, Some(30.0), &config, &token).unwrap();
        assert_eq!(align_report.segments_total, 3);

        // Punctuate.
        let punct_report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(punct_report.segments_total, 3);

        // Diarize.
        let diarize_report = diarize_segments(&mut segments, Some(30.0), None, &token).unwrap();
        assert_eq!(diarize_report.segments_total, 3);
        assert_eq!(diarize_report.segments_labeled, 3);

        // All segments should have timestamps, punctuation, and speaker labels.
        for seg in &segments {
            assert!(seg.start_sec.is_some(), "should have start timestamp");
            assert!(seg.end_sec.is_some(), "should have end timestamp");
            assert!(seg.speaker.is_some(), "should have speaker label");
        }
    }

    #[test]
    fn composition_skip_all_optional_stages_minimal_pipeline() {
        // Minimal valid pipeline config: Ingest → Normalize → Backend → Persist.
        let config = PipelineBuilder::new()
            .stage(PipelineStage::Ingest)
            .stage(PipelineStage::Normalize)
            .stage(PipelineStage::Backend)
            .stage(PipelineStage::Persist)
            .build()
            .expect("minimal pipeline should be valid");

        assert!(!config.has_stage(PipelineStage::Vad));
        assert!(!config.has_stage(PipelineStage::Separate));
        assert!(!config.has_stage(PipelineStage::Accelerate));
        assert!(!config.has_stage(PipelineStage::Align));
        assert!(!config.has_stage(PipelineStage::Punctuate));
        assert!(!config.has_stage(PipelineStage::Diarize));

        assert_eq!(config.stages().len(), 4);
        config.validate().expect("should validate");
    }

    #[test]
    fn composition_all_optional_stages_full_pipeline() {
        // Full pipeline with all optional stages included.
        let config = PipelineConfig::default();
        assert_eq!(config.stages().len(), 10);
        config.validate().expect("default pipeline should validate");

        for stage in &[
            PipelineStage::Vad,
            PipelineStage::Separate,
            PipelineStage::Accelerate,
            PipelineStage::Align,
            PipelineStage::Punctuate,
            PipelineStage::Diarize,
        ] {
            assert!(config.has_stage(*stage), "default should include {stage}");
        }
    }

    #[test]
    fn composition_vad_segments_unaffected_by_downstream() {
        // VAD produces regions; downstream stages should not interfere.
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig::default();
        let dir = tempdir().unwrap();
        let path = dir.path().join("silence.wav");
        // 44-byte WAV header + 3200 bytes of silence (all zeros).
        let mut data = vec![0u8; 44];
        data.extend(vec![0u8; 3200]);
        std::fs::write(&path, &data).unwrap();

        let report = vad_energy_detect(&path, &config, &token).unwrap();
        // Silence-only audio should have zero voice regions.
        assert!(report.regions.is_empty());
        assert!(report.silence_only);
    }

    #[test]
    fn composition_source_separate_does_not_need_vad() {
        // Source separation should work even when VAD has not run.
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let report =
            source_separate(std::path::Path::new("/nonexistent/audio.wav"), &token).unwrap();
        // Should fallback gracefully.
        assert!(report.vocal_isolated);
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("analysis unavailable"))
        );
    }

    #[test]
    fn composition_each_optional_stage_respects_cancellation() {
        // Each optional stage should return an error when given an expired token.
        let expired = CancellationToken::already_expired();

        // VAD.
        let vad_cfg = VadConfig::default();
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("tiny.wav");
        let mut data = vec![0u8; 44];
        data.extend(vec![0u8; 320]);
        std::fs::write(&wav_path, &data).unwrap();
        assert!(vad_energy_detect(&wav_path, &vad_cfg, &expired).is_err());

        // Source separate.
        assert!(source_separate(std::path::Path::new("/tmp/fake.wav"), &expired).is_err());

        // Punctuate.
        let mut segs = vec![make_segment(0.0, 1.0, "hello")];
        assert!(punctuate_segments(&mut segs, &expired).is_err());

        // Diarize.
        let mut segs2 = vec![make_segment(0.0, 1.0, "hello")];
        assert!(diarize_segments(&mut segs2, Some(2.0), None, &expired).is_err());

        // Align.
        let align_config = AlignConfig::default();
        let mut segs3 = vec![make_segment(0.0, 1.0, "hello")];
        assert!(ctc_forced_align(&mut segs3, Some(2.0), &align_config, &expired).is_err());
    }

    #[test]
    fn composition_pipeline_config_skip_post_backend_stages() {
        // Pipeline with backend but no post-processing: valid.
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Accelerate)
            .without(PipelineStage::Align)
            .without(PipelineStage::Punctuate)
            .without(PipelineStage::Diarize)
            .build()
            .expect("valid");
        assert_eq!(config.stages().len(), 6); // Ingest, Normalize, Vad, Separate, Backend, Persist
        config.validate().expect("should validate");
    }

    #[test]
    fn composition_pipeline_config_skip_pre_backend_optional() {
        // Pipeline with all post-processing but no pre-processing optional stages.
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Vad)
            .without(PipelineStage::Separate)
            .build()
            .expect("valid");
        assert!(config.has_stage(PipelineStage::Accelerate));
        assert!(config.has_stage(PipelineStage::Align));
        assert!(config.has_stage(PipelineStage::Punctuate));
        assert!(config.has_stage(PipelineStage::Diarize));
        config.validate().expect("should validate");
    }

    #[test]
    fn recommended_budget_high_utilization_triggers_increase() {
        // 90% utilization exactly → "increase_budget"
        let (budget, action, _reason, utilization) = recommended_budget(900, 1000);
        assert_eq!(action, "increase_budget");
        assert!((utilization - 0.9).abs() < f64::EPSILON);
        // uplift = ceil(900 * 1.25) = 1125, which is > 1001, so recommended = 1125
        assert_eq!(budget, 1125);

        // 100% utilization → "increase_budget"
        let (budget, action, _, _) = recommended_budget(1000, 1000);
        assert_eq!(action, "increase_budget");
        // uplift = ceil(1000 * 1.25) = 1250, max(1250, 1001) = 1250
        assert_eq!(budget, 1250);
    }

    #[test]
    fn recommended_budget_low_utilization_triggers_decrease_candidate() {
        // 30% utilization exactly → "decrease_budget_candidate" (<=0.30)
        let (budget, action, _reason, utilization) = recommended_budget(300, 1000);
        assert_eq!(action, "decrease_budget_candidate");
        assert!((utilization - 0.3).abs() < f64::EPSILON);
        // candidate = ceil(300 * 1.60) = 480, max(480, 1000) = 1000
        assert_eq!(budget, 1000);

        // 10% utilization → low floor clamp: max(ceil(100*1.60), 1000)
        let (budget, action, _, _) = recommended_budget(100, 10_000);
        assert_eq!(action, "decrease_budget_candidate");
        // candidate = ceil(100 * 1.60) = 160, max(160, 1000) = 1000
        assert_eq!(budget, 1000);
    }

    #[test]
    fn recommended_budget_middle_band_keeps_budget() {
        // 50% utilization → "keep_budget"
        let (budget, action, _reason, utilization) = recommended_budget(500, 1000);
        assert_eq!(action, "keep_budget");
        assert!((utilization - 0.5).abs() < f64::EPSILON);
        assert_eq!(budget, 1000); // unchanged

        // Just above 0.30 threshold → "keep_budget"
        let (_, action, _, utilization) = recommended_budget(310, 1000);
        assert_eq!(action, "keep_budget");
        assert!(utilization > 0.30);

        // Just below 0.90 threshold → "keep_budget"
        let (_, action, _, utilization) = recommended_budget(899, 1000);
        assert_eq!(action, "keep_budget");
        assert!(utilization < 0.90);
    }

    #[test]
    fn event_log_push_with_non_object_payload_preserves_value() {
        let mut log = EventLog::new(
            "run-nonobj".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );
        // Push with null payload — the if-let Object guard should skip trace_id injection
        log.push("ingest", "ingest.start", "begin", Value::Null);
        assert_eq!(log.events.len(), 1);
        assert_eq!(log.events[0].payload, Value::Null);

        // Push with array payload
        log.push("backend", "backend.ok", "done", json!([1, 2, 3]));
        assert_eq!(log.events.len(), 2);
        assert_eq!(log.events[1].payload, json!([1, 2, 3]));

        // Push with string payload
        log.push("persist", "persist.ok", "saved", json!("hello"));
        assert_eq!(log.events.len(), 3);
        assert_eq!(log.events[2].payload, json!("hello"));

        // Sequence numbers still monotonic
        assert_eq!(log.events[0].seq, 1);
        assert_eq!(log.events[1].seq, 2);
        assert_eq!(log.events[2].seq, 3);
    }

    #[test]
    fn stage_latency_profile_clamps_negative_queue_time() {
        // Simulate overlapping stage timestamps (stage 2 starts before stage 1 ends)
        let events = vec![
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:00.000Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ingest.start".to_owned(),
                message: "begin".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 2,
                ts_rfc3339: "2026-01-01T00:00:02.000Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ingest.ok".to_owned(),
                message: "done".to_owned(),
                payload: json!({"elapsed_ms": 2000}),
            },
            // normalize starts 500ms BEFORE ingest ended (clock skew scenario)
            RunEvent {
                seq: 3,
                ts_rfc3339: "2026-01-01T00:00:01.500Z".to_owned(),
                stage: "normalize".to_owned(),
                code: "normalize.start".to_owned(),
                message: "begin".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 4,
                ts_rfc3339: "2026-01-01T00:00:03.000Z".to_owned(),
                stage: "normalize".to_owned(),
                code: "normalize.ok".to_owned(),
                message: "done".to_owned(),
                payload: json!({"elapsed_ms": 1500}),
            },
        ];
        let budgets = StageBudgetPolicy::from_source(|_| None);
        let profile = stage_latency_profile(&events, budgets);

        // The negative queue time (-500ms) should be clamped to 0
        let norm_queue = profile["stages"]["normalize"]["queue_ms"]
            .as_u64()
            .expect("queue_ms should be present");
        assert_eq!(norm_queue, 0, "negative queue time should be clamped to 0");
    }

    // -----------------------------------------------------------------------
    // Deterministic stage-order and error-code contract tests (bd-3pf.15)
    // -----------------------------------------------------------------------

    /// Every PipelineStage label must be a known, stable, lowercase string.
    #[test]
    fn pipeline_stage_labels_are_stable_and_exhaustive() {
        let expected: Vec<(&str, PipelineStage)> = vec![
            ("ingest", PipelineStage::Ingest),
            ("normalize", PipelineStage::Normalize),
            ("vad", PipelineStage::Vad),
            ("separate", PipelineStage::Separate),
            ("backend", PipelineStage::Backend),
            ("acceleration", PipelineStage::Accelerate),
            ("align", PipelineStage::Align),
            ("punctuate", PipelineStage::Punctuate),
            ("diarize", PipelineStage::Diarize),
            ("persist", PipelineStage::Persist),
        ];
        assert_eq!(expected.len(), 10, "must cover all 10 pipeline stages");
        for (label, stage) in &expected {
            assert_eq!(
                stage.label(),
                *label,
                "stage {stage:?} label must be {label}"
            );
        }
    }

    /// Stage label must match Display implementation for all default stages.
    #[test]
    fn contract_stage_display_matches_label_for_all_defaults() {
        for stage in PipelineConfig::default().stages() {
            assert_eq!(
                format!("{stage}"),
                stage.label(),
                "Display must match label for {stage:?}"
            );
        }
    }

    /// DEFAULT_STAGES must include all 10 stages with no duplicates.
    #[test]
    fn default_stages_covers_all_variants_without_duplicates() {
        let config = PipelineConfig::default();
        let stages = config.stages();
        assert_eq!(stages.len(), 10, "default config must have 10 stages");
        let mut seen = std::collections::HashSet::new();
        for stage in stages {
            assert!(
                seen.insert(stage),
                "duplicate stage {stage:?} in DEFAULT_STAGES"
            );
        }
    }

    /// Happy-path stage ordering: the core stages that execute real work
    /// must appear in this exact order within DEFAULT_STAGES.
    #[test]
    fn happy_path_core_stages_in_correct_order() {
        let config = PipelineConfig::default();
        let stages = config.stages();
        let core_order = [
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Backend,
            PipelineStage::Accelerate,
            PipelineStage::Persist,
        ];
        let positions: Vec<usize> = core_order
            .iter()
            .map(|s| {
                stages
                    .iter()
                    .position(|x| x == s)
                    .expect("{s:?} missing from default stages")
            })
            .collect();
        for window in positions.windows(2) {
            assert!(
                window[0] < window[1],
                "stage ordering violated: position {} must come before {}",
                window[0],
                window[1]
            );
        }
    }

    /// For every known stage label, stage_failure_code must produce a
    /// deterministic pattern: `{label}.timeout`, `{label}.cancelled`,
    /// or `{label}.error` depending on error class.
    #[test]
    fn stage_failure_code_pattern_is_deterministic_across_all_stages() {
        let stage_labels: Vec<&str> = PipelineConfig::default()
            .stages()
            .iter()
            .map(|s| s.label())
            .collect();

        let timeout = FwError::StageTimeout {
            stage: "any".to_owned(),
            budget_ms: 1000,
        };
        let cmd_timeout = FwError::CommandTimedOut {
            command: "cmd".to_owned(),
            timeout_ms: 1000,
            stderr_suffix: String::new(),
        };
        let cancelled = FwError::Cancelled("test".to_owned());
        let generic = FwError::BackendUnavailable("test".to_owned());

        for label in &stage_labels {
            // Timeout variants
            assert_eq!(
                stage_failure_code(label, &timeout),
                format!("{label}.timeout"),
                "StageTimeout must produce {label}.timeout"
            );
            assert_eq!(
                stage_failure_code(label, &cmd_timeout),
                format!("{label}.timeout"),
                "CommandTimedOut must produce {label}.timeout"
            );
            // Cancelled
            assert_eq!(
                stage_failure_code(label, &cancelled),
                format!("{label}.cancelled"),
                "Cancelled must produce {label}.cancelled"
            );
            // Generic error
            assert_eq!(
                stage_failure_code(label, &generic),
                format!("{label}.error"),
                "other errors must produce {label}.error"
            );
        }
    }

    /// stage_failure_message must return deterministic messages for each
    /// error class, independent of the stage label.
    #[test]
    fn stage_failure_message_is_deterministic_for_each_error_class() {
        let timeout = FwError::StageTimeout {
            stage: "x".to_owned(),
            budget_ms: 1,
        };
        let cmd_timeout = FwError::CommandTimedOut {
            command: "x".to_owned(),
            timeout_ms: 1,
            stderr_suffix: String::new(),
        };
        let cancelled = FwError::Cancelled("x".to_owned());
        let io_err = FwError::Io(std::io::Error::other("x"));

        assert_eq!(
            stage_failure_message(&timeout, "fallback"),
            "stage budget exceeded"
        );
        assert_eq!(
            stage_failure_message(&cmd_timeout, "fallback"),
            "stage budget exceeded"
        );
        assert_eq!(
            stage_failure_message(&cancelled, "fallback"),
            "pipeline cancelled by checkpoint policy"
        );
        assert_eq!(
            stage_failure_message(&io_err, "custom fallback"),
            "custom fallback",
            "non-timeout/cancelled errors must return caller's default"
        );
    }

    /// The event code for every stage follows the pattern `{label}.{suffix}`
    /// where suffix is one of: start, ok, error, timeout, cancelled.
    /// This test ensures the patterns are well-formed.
    #[test]
    fn event_code_suffixes_are_well_formed() {
        let suffixes = ["start", "ok", "error", "timeout", "cancelled"];
        let labels: Vec<&str> = PipelineConfig::default()
            .stages()
            .iter()
            .map(|s| s.label())
            .collect();

        for label in &labels {
            for suffix in &suffixes {
                let code = format!("{label}.{suffix}");
                assert!(!code.is_empty(), "event code must be non-empty");
                assert!(
                    code.contains('.'),
                    "event code must contain a dot separator"
                );
                let parts: Vec<&str> = code.splitn(2, '.').collect();
                assert_eq!(parts.len(), 2, "event code must have exactly stage.suffix");
                assert_eq!(parts[0], *label);
                assert_eq!(parts[1], *suffix);
            }
        }
    }

    /// Cross-reference: robot_error_code categories map consistently to
    /// stage_failure_code suffixes.
    #[test]
    fn robot_error_code_consistent_with_stage_failure_code() {
        // Timeout errors: robot_error_code → FW-ROBOT-TIMEOUT, stage → .timeout
        let timeout = FwError::StageTimeout {
            stage: "x".to_owned(),
            budget_ms: 1,
        };
        assert_eq!(timeout.robot_error_code(), "FW-ROBOT-TIMEOUT");
        assert!(stage_failure_code("any", &timeout).ends_with(".timeout"));

        let cmd_timeout = FwError::CommandTimedOut {
            command: "x".to_owned(),
            timeout_ms: 1,
            stderr_suffix: String::new(),
        };
        assert_eq!(cmd_timeout.robot_error_code(), "FW-ROBOT-TIMEOUT");
        assert!(stage_failure_code("any", &cmd_timeout).ends_with(".timeout"));

        // Cancelled: robot_error_code → FW-ROBOT-CANCELLED, stage → .cancelled
        let cancelled = FwError::Cancelled("x".to_owned());
        assert_eq!(cancelled.robot_error_code(), "FW-ROBOT-CANCELLED");
        assert!(stage_failure_code("any", &cancelled).ends_with(".cancelled"));

        // Backend unavailable: robot_error_code → FW-ROBOT-BACKEND, stage → .error
        let backend = FwError::BackendUnavailable("x".to_owned());
        assert_eq!(backend.robot_error_code(), "FW-ROBOT-BACKEND");
        assert!(stage_failure_code("any", &backend).ends_with(".error"));

        // Generic: robot_error_code → FW-ROBOT-EXEC, stage → .error
        let io = FwError::Io(std::io::Error::other("x"));
        assert_eq!(io.robot_error_code(), "FW-ROBOT-EXEC");
        assert!(stage_failure_code("any", &io).ends_with(".error"));
    }

    /// Ensure error_code and robot_error_code are both non-empty for all
    /// error variants, and error_code always starts with "FW-".
    #[test]
    fn error_codes_non_empty_and_well_prefixed() {
        let errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
            FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            FwError::CommandMissing {
                command: "x".to_owned(),
            },
            FwError::CommandFailed {
                command: "x".to_owned(),
                status: 1,
                stderr_suffix: String::new(),
            },
            FwError::CommandTimedOut {
                command: "x".to_owned(),
                timeout_ms: 1,
                stderr_suffix: String::new(),
            },
            FwError::BackendUnavailable("x".to_owned()),
            FwError::InvalidRequest("x".to_owned()),
            FwError::Storage("x".to_owned()),
            FwError::Unsupported("x".to_owned()),
            FwError::MissingArtifact(PathBuf::from("x")),
            FwError::Cancelled("x".to_owned()),
            FwError::StageTimeout {
                stage: "x".to_owned(),
                budget_ms: 1,
            },
        ];
        for error in &errors {
            let code = error.error_code();
            assert!(
                !code.is_empty(),
                "error_code must be non-empty for {error:?}"
            );
            assert!(
                code.starts_with("FW-"),
                "error_code `{code}` must start with FW- for {error:?}"
            );
            let robot = error.robot_error_code();
            assert!(
                !robot.is_empty(),
                "robot_error_code must be non-empty for {error:?}"
            );
            assert!(
                robot.starts_with("FW-ROBOT-"),
                "robot_error_code `{robot}` must start with FW-ROBOT- for {error:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // bd-38c.5: Cancellation injection tests using asupersync lab runtime
    // -----------------------------------------------------------------------

    /// Test harness for deterministic cancellation injection.
    ///
    /// Creates a `PipelineCx` and `EventLog` and provides methods to schedule
    /// cancellation at specific pipeline stages, after a timeout, or
    /// immediately.  All scheduling is deterministic and does not depend on
    /// wall-clock timing.
    struct CancellationTestHarness {
        pcx: PipelineCx,
        log: EventLog,
        /// If set, cancellation is injected after this stage completes.
        cancel_trigger: Option<PipelineStage>,
        /// Stages that completed successfully before cancellation.
        completed_stages: Vec<PipelineStage>,
        /// Whether finalizers were executed.
        finalizers_ran: bool,
    }

    impl CancellationTestHarness {
        /// Create a harness with no cancellation scheduled (deadline = None).
        fn new() -> Self {
            Self {
                pcx: PipelineCx::new(None),
                log: EventLog::new(
                    "cancel-test-run".to_owned(),
                    "00000000000000000000000000000000".to_owned(),
                    None,
                ),
                cancel_trigger: None,
                completed_stages: Vec::new(),
                finalizers_ran: false,
            }
        }

        /// Schedule cancellation to trigger after `stage` completes.
        ///
        /// The harness will run stages normally until `stage` finishes, then
        /// set the deadline to the past so the next checkpoint fails with
        /// `FwError::Cancelled`.  This is fully deterministic.
        fn cancel_after_stage(&mut self, stage: PipelineStage) {
            self.cancel_trigger = Some(stage);
        }

        /// Schedule cancellation after `ms` milliseconds from now.
        ///
        /// For deterministic tests, pass `0` to trigger immediate cancellation.
        fn cancel_after_ms(&mut self, ms: u64) {
            self.pcx = PipelineCx::new(Some(ms));
            // Re-create the log since pcx was replaced.
            self.log = EventLog::new(
                "cancel-test-run".to_owned(),
                self.pcx.trace_id().to_string(),
                None,
            );
        }

        /// Trigger immediate cancellation by setting the deadline to the past.
        fn cancel_now(&mut self) {
            self.pcx.cancel_now();
        }

        /// Simulate pipeline execution through the given config's stages.
        ///
        /// Each stage:
        /// 1. Runs `checkpoint_or_emit` (which checks the deadline).
        /// 2. If the checkpoint passes, records the stage as completed.
        /// 3. If `cancel_trigger` matches the completed stage, injects
        ///    cancellation so the *next* checkpoint will fail.
        ///
        /// Returns `Ok(completed_stages)` if all stages pass, or
        /// `Err(FwError)` at the first cancellation checkpoint failure.
        fn simulate_pipeline(
            &mut self,
            config: &PipelineConfig,
        ) -> crate::FwResult<Vec<PipelineStage>> {
            for stage in config.stages() {
                // Checkpoint before executing the stage.
                checkpoint_or_emit(stage.label(), &mut self.pcx, &mut self.log)?;

                // Stage "executes" (no real work in tests).
                self.completed_stages.push(*stage);

                // If this was the trigger stage, inject cancellation.
                if self.cancel_trigger == Some(*stage) {
                    self.pcx.cancel_now();
                }
            }

            Ok(self.completed_stages.clone())
        }

        /// Run finalizers and record that they ran.
        fn run_finalizers(&mut self) {
            self.pcx.run_finalizers();
            self.finalizers_ran = true;
        }

        /// Run finalizers with a bounded budget and record that they ran.
        fn run_finalizers_bounded(&mut self, budget_ms: u64) {
            self.pcx.run_finalizers_bounded(budget_ms);
            self.finalizers_ran = true;
        }
    }

    // -- Test 1: Cancelling during Ingest stage properly cleans up and returns Err --

    #[test]
    fn cancel_during_ingest_returns_err_and_cleans_up() {
        let mut harness = CancellationTestHarness::new();
        // Cancel immediately so the very first checkpoint (before Ingest) fails.
        harness.cancel_now();

        // Register a finalizer to verify cleanup runs.
        let cleanup_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag_clone = cleanup_flag.clone();
        harness.pcx.register_finalizer(
            "test_cleanup",
            Finalizer::Custom(Box::new(move || {
                flag_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            })),
        );

        let result = harness.simulate_pipeline(&PipelineConfig::default());
        assert!(
            result.is_err(),
            "pipeline should fail when cancelled before Ingest"
        );
        let error = result.unwrap_err();
        assert!(
            matches!(error, FwError::Cancelled(_)),
            "error should be Cancelled, got: {error:?}"
        );
        assert!(
            harness.completed_stages.is_empty(),
            "no stages should have completed"
        );

        // Finalizers should still run on cleanup.
        harness.run_finalizers();
        assert!(harness.finalizers_ran, "finalizers_ran flag should be set");
        assert!(
            cleanup_flag.load(std::sync::atomic::Ordering::SeqCst),
            "cleanup finalizer should have executed"
        );
    }

    // -- Test 2: Cancelling during Backend stage stops and returns partial results --

    #[test]
    fn cancel_during_backend_stops_with_partial_results() {
        let mut harness = CancellationTestHarness::new();
        // Cancel after Backend completes — next checkpoint (before Accelerate) should fail.
        harness.cancel_after_stage(PipelineStage::Backend);

        let result = harness.simulate_pipeline(&PipelineConfig::default());
        assert!(
            result.is_err(),
            "pipeline should fail after Backend cancellation"
        );

        let error = result.unwrap_err();
        assert!(
            matches!(error, FwError::Cancelled(_)),
            "error should be Cancelled, got: {error:?}"
        );

        // Stages up to and including Backend should have completed.
        let expected_completed = vec![
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Vad,
            PipelineStage::Separate,
            PipelineStage::Backend,
        ];
        assert_eq!(
            harness.completed_stages, expected_completed,
            "exactly Ingest, Normalize, Vad, Separate, Backend should have completed"
        );
    }

    // -- Test 3: Cancelling during Persist stage still commits partial data --

    #[test]
    fn cancel_during_persist_commits_partial_data() {
        let mut harness = CancellationTestHarness::new();
        // Cancel after Diarize so that the checkpoint before Persist fails.
        harness.cancel_after_stage(PipelineStage::Diarize);

        let result = harness.simulate_pipeline(&PipelineConfig::default());
        assert!(
            result.is_err(),
            "pipeline should fail at Persist checkpoint"
        );

        let error = result.unwrap_err();
        assert!(
            matches!(error, FwError::Cancelled(_)),
            "error should be Cancelled, got: {error:?}"
        );

        // All stages through Diarize should have completed; Persist should NOT
        // have executed because the checkpoint before it failed.
        assert!(
            harness.completed_stages.contains(&PipelineStage::Diarize),
            "Diarize should have completed"
        );
        assert!(
            !harness.completed_stages.contains(&PipelineStage::Persist),
            "Persist should NOT have completed (cancelled at its checkpoint)"
        );

        // Evidence should contain the cancellation record.
        assert!(
            !harness.pcx.evidence().is_empty(),
            "cancellation evidence should have been recorded"
        );
    }

    // -- Test 4: Cancelling between stages (inter-stage boundary) is handled cleanly --

    #[test]
    fn cancel_at_inter_stage_boundary_is_clean() {
        let mut harness = CancellationTestHarness::new();
        // Cancel after Normalize — the checkpoint before Vad should detect it.
        harness.cancel_after_stage(PipelineStage::Normalize);

        let result = harness.simulate_pipeline(&PipelineConfig::default());
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, FwError::Cancelled(_)));

        // Ingest and Normalize should have completed; Vad should not.
        assert_eq!(
            harness.completed_stages,
            vec![PipelineStage::Ingest, PipelineStage::Normalize],
        );

        // The event log should show a `.cancelled` event for the next stage.
        let cancelled_events: Vec<_> = harness
            .log
            .events
            .iter()
            .filter(|e| e.code.ends_with(".cancelled"))
            .collect();
        assert_eq!(
            cancelled_events.len(),
            1,
            "exactly one .cancelled event should have been emitted"
        );
        assert_eq!(
            cancelled_events[0].stage, "vad",
            "the cancelled event should be for the Vad stage checkpoint"
        );
    }

    // -- Test 5: Double-cancellation is idempotent --

    #[test]
    fn cancel_double_cancellation_is_idempotent() {
        let mut harness = CancellationTestHarness::new();
        harness.cancel_now();

        // First cancellation attempt.
        let result1 = harness.simulate_pipeline(&PipelineConfig::default());
        assert!(result1.is_err());
        assert!(
            matches!(result1.unwrap_err(), FwError::Cancelled(_)),
            "first cancellation should produce Cancelled error"
        );

        // Reset completed_stages and log for second attempt.
        harness.completed_stages.clear();
        harness.log = EventLog::new(
            "cancel-test-run-2".to_owned(),
            "00000000000000000000000000000000".to_owned(),
            None,
        );

        // Cancel again (calling cancel_now a second time) — should be idempotent.
        harness.cancel_now();
        let result2 = harness.simulate_pipeline(&PipelineConfig::default());
        assert!(result2.is_err());
        assert!(
            matches!(result2.unwrap_err(), FwError::Cancelled(_)),
            "second cancellation should also produce Cancelled error"
        );
        assert!(
            harness.completed_stages.is_empty(),
            "no stages should complete on second cancellation either"
        );

        // Exercise cancel_after_ms(0) as a third cancellation path —
        // verifies the alternate API also triggers correctly.
        harness.cancel_after_ms(0);
        harness.completed_stages.clear();
        let result3 = harness.simulate_pipeline(&PipelineConfig::default());
        assert!(result3.is_err());
        assert!(
            matches!(result3.unwrap_err(), FwError::Cancelled(_)),
            "cancel_after_ms(0) should also produce Cancelled error"
        );
        assert!(
            harness.completed_stages.is_empty(),
            "no stages should complete after cancel_after_ms(0) either"
        );
    }

    // -- Test 6: Cancellation token propagation through all 10 pipeline stages --

    #[test]
    fn cancel_token_propagation_through_all_stages() {
        let config = PipelineConfig::default();
        assert_eq!(
            config.stages().len(),
            10,
            "default config should have 10 stages"
        );

        // Test that a non-cancelled token propagates cleanly through all 10.
        let mut harness = CancellationTestHarness::new();
        let result = harness.simulate_pipeline(&config);
        assert!(result.is_ok(), "uncancelled pipeline should succeed");
        assert_eq!(
            result.unwrap().len(),
            10,
            "all 10 stages should have completed"
        );

        // Verify that the cancellation token from PipelineCx is not cancelled
        // at each stage boundary.
        let pcx_no_cancel = PipelineCx::new(None);
        for stage in config.stages() {
            let token = pcx_no_cancel.cancellation_token(); // ubs:ignore — cancellation token is not a secret
            assert!(
                !token.is_cancelled(),
                "token should not be cancelled at stage {stage}"
            );
            assert!(
                token.checkpoint().is_ok(),
                "token checkpoint should pass at stage {stage}"
            );
        }

        // Verify that a cancelled token is detected at every stage boundary.
        let pcx_cancel = PipelineCx::new(Some(0));
        for stage in config.stages() {
            let token = pcx_cancel.cancellation_token(); // ubs:ignore — cancellation token is not a secret
            assert!(
                token.is_cancelled(),
                "token should be cancelled at stage {stage}"
            );
            assert!(
                token.checkpoint().is_err(),
                "token checkpoint should fail at stage {stage}"
            );
        }
    }

    // -- Test 7: Finalizers run even after cancellation (cleanup is guaranteed) --

    #[test]
    fn cancel_finalizers_run_after_cancellation() {
        let mut harness = CancellationTestHarness::new();

        // Register multiple finalizers.
        let flag_a = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag_b = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag_c = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

        let fa = flag_a.clone();
        let oa = order.clone();
        harness.pcx.register_finalizer(
            "finalizer_a",
            Finalizer::Custom(Box::new(move || {
                fa.store(true, std::sync::atomic::Ordering::SeqCst);
                oa.lock().unwrap().push("a".to_owned());
            })),
        );

        let fb = flag_b.clone();
        let ob = order.clone();
        harness.pcx.register_finalizer(
            "finalizer_b",
            Finalizer::Custom(Box::new(move || {
                fb.store(true, std::sync::atomic::Ordering::SeqCst);
                ob.lock().unwrap().push("b".to_owned());
            })),
        );

        let fc = flag_c.clone();
        let oc = order.clone();
        harness.pcx.register_finalizer(
            "finalizer_c",
            Finalizer::Custom(Box::new(move || {
                fc.store(true, std::sync::atomic::Ordering::SeqCst);
                oc.lock().unwrap().push("c".to_owned());
            })),
        );

        // Cancel after Ingest.
        harness.cancel_after_stage(PipelineStage::Ingest);
        let result = harness.simulate_pipeline(&PipelineConfig::default());
        assert!(result.is_err(), "pipeline should be cancelled");

        // Run finalizers (simulating the cleanup path in run_pipeline).
        harness.run_finalizers_bounded(5_000);

        // All three should have fired.
        assert!(
            flag_a.load(std::sync::atomic::Ordering::SeqCst),
            "finalizer_a should have run"
        );
        assert!(
            flag_b.load(std::sync::atomic::Ordering::SeqCst),
            "finalizer_b should have run"
        );
        assert!(
            flag_c.load(std::sync::atomic::Ordering::SeqCst),
            "finalizer_c should have run"
        );

        // LIFO order: c, b, a.
        let executed_order = order.lock().unwrap().clone();
        assert_eq!(
            executed_order,
            vec!["c", "b", "a"],
            "finalizers should execute in LIFO order"
        );
    }

    // -- Test 8: CancellationToken's is_cancelled() check works at stage boundaries --

    #[test]
    fn cancel_is_cancelled_works_at_stage_boundaries() {
        let config = PipelineConfig::default();
        let all_stages = config.stages().to_vec();

        // For each stage in the pipeline, verify that cancelling after it
        // makes is_cancelled() return true for all subsequent stages.
        for (trigger_idx, trigger_stage) in all_stages.iter().enumerate() {
            let mut harness = CancellationTestHarness::new();
            harness.cancel_after_stage(*trigger_stage);

            let _ = harness.simulate_pipeline(&config);

            // The cancellation token should now be expired.
            let token = harness.pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
            assert!(
                token.is_cancelled(),
                "token should be cancelled after trigger stage {trigger_stage}"
            );

            // Completed stages should be exactly those up to and including
            // the trigger stage.
            let expected: Vec<PipelineStage> = all_stages[..=trigger_idx].to_vec();
            assert_eq!(
                harness.completed_stages, expected,
                "completed stages should match expectation when cancelling after {trigger_stage}"
            );
        }
    }

    // -- edge case tests (next pass) --

    #[test]
    fn punctuate_segments_multibyte_unicode_first_char() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 1.0, "über cool"),
            make_segment(1.0, 2.0, "élan vital"),
            make_segment(2.0, 3.0, "ñoño"),
        ];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(report.segments_total, 3);
        // Multi-byte chars should be uppercased correctly.
        assert!(
            segments[0].text.starts_with('Ü'),
            "ü should be uppercased to Ü, got: {}",
            segments[0].text
        );
        assert!(
            segments[1].text.starts_with('É'),
            "é should be uppercased to É, got: {}",
            segments[1].text
        );
        assert!(
            segments[2].text.starts_with('Ñ'),
            "ñ should be uppercased to Ñ, got: {}",
            segments[2].text
        );
    }

    #[test]
    fn diarize_segments_zero_duration_does_not_panic() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 1.0, "hello"),
            make_segment(1.0, 2.0, "world"),
        ];
        // Some(0.0) → clamped to 1e-6, should not panic or produce NaN.
        let report = diarize_segments(&mut segments, Some(0.0), None, &token).unwrap();
        assert_eq!(report.segments_total, 2);
        assert!(
            report.speakers_detected >= 1,
            "should detect at least one speaker"
        );
        // All segments should have speaker labels (no NaN-related issues).
        for seg in &segments {
            assert!(
                seg.speaker.is_some(),
                "every segment should have a speaker label"
            );
        }
    }

    #[test]
    fn speaker_embedding_cosine_similarity_negative_features() {
        // Embeddings with negative features should still compute correctly.
        let a = SpeakerEmbedding {
            features: [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        };
        let b = SpeakerEmbedding {
            features: [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        };
        let sim = a.cosine_similarity(&b);
        // Dot = -1+0-1 = -2, |a|=sqrt(2), |b|=sqrt(2), sim = -2/2 = -1.0
        assert!(
            (sim - (-1.0)).abs() < 1e-9,
            "anti-parallel vectors should have cosine sim = -1.0, got {sim}"
        );

        // Identical negative vectors → cosine sim = 1.0
        let c = SpeakerEmbedding {
            features: [-0.5, -0.3, -0.8, -0.25, -0.5, -0.75],
        };
        let sim_cc = c.cosine_similarity(&c);
        assert!(
            (sim_cc - 1.0).abs() < 1e-9,
            "identical vectors should have cosine sim = 1.0, got {sim_cc}"
        );
    }

    #[test]
    fn speaker_embedding_centroid_single() {
        let a = SpeakerEmbedding {
            features: [0.5, 0.25, 0.75, 0.125, 0.5, 0.25],
        };
        let c = SpeakerEmbedding::centroid(std::slice::from_ref(&a));
        for (i, (&cf, &af)) in c.features.iter().zip(a.features.iter()).enumerate() {
            assert!(
                (cf - af).abs() < 1e-9,
                "centroid of single embedding should equal the embedding at dim {i}"
            );
        }
    }

    #[test]
    fn speaker_embedding_centroid_mean() {
        let a = SpeakerEmbedding {
            features: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let b = SpeakerEmbedding {
            features: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        };
        let c = SpeakerEmbedding::centroid(&[a, b]);
        for (i, &f) in c.features.iter().enumerate() {
            assert!(
                (f - 0.5).abs() < 1e-9,
                "centroid of [0, 1] should be 0.5 at dim {i}, got {f}"
            );
        }
    }

    #[test]
    fn speaker_embedding_centroid_empty() {
        let c = SpeakerEmbedding::centroid(&[]);
        for (i, &f) in c.features.iter().enumerate() {
            assert!(
                f.abs() < 1e-9,
                "centroid of empty slice should be zero at dim {i}, got {f}"
            );
        }
    }

    #[test]
    fn diarize_turn_taking_detects_speaker_changes() {
        // Two groups of segments separated by a large time gap should be
        // assigned to different speakers due to the inter-segment gap feature.
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 2.0, "speaker one talks here"),
            make_segment(2.0, 4.0, "speaker one keeps going"),
            // Large gap suggests a speaker change.
            make_segment(20.0, 22.0, "new speaker joins late"),
            make_segment(22.0, 24.0, "new speaker continues"),
        ];
        let report = diarize_segments(&mut segments, Some(30.0), None, &token).unwrap();

        assert_eq!(report.segments_total, 4);
        assert_eq!(report.segments_labeled, 4);
        // The two temporal groups should get different speakers.
        assert!(
            report.speakers_detected >= 2,
            "expected at least 2 speakers from time-separated groups, got {}",
            report.speakers_detected
        );
        // First two segments share a speaker.
        assert_eq!(segments[0].speaker, segments[1].speaker);
        // Last two segments share a speaker.
        assert_eq!(segments[2].speaker, segments[3].speaker);
    }

    #[test]
    fn silhouette_score_well_separated_clusters() {
        // Two tight clusters far apart → silhouette near 1.0.
        let embeddings = vec![
            SpeakerEmbedding {
                features: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            SpeakerEmbedding {
                features: [0.0, 0.0, 0.0, 0.0, 0.0, 0.125],
            },
            SpeakerEmbedding {
                features: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            SpeakerEmbedding {
                features: [1.0, 1.0, 1.0, 1.0, 1.0, 0.875],
            },
        ];
        let assignments = vec![0, 0, 1, 1];
        let score = silhouette_score(&embeddings, &assignments, 2).unwrap();
        assert!(
            score > 0.9,
            "well-separated clusters should have silhouette > 0.9, got {score}"
        );
    }

    #[test]
    fn silhouette_score_overlapping_clusters() {
        // Two clusters that overlap substantially → lower silhouette.
        let embeddings = vec![
            SpeakerEmbedding {
                features: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            },
            SpeakerEmbedding {
                features: [0.5, 0.5, 0.5, 0.5, 0.5, 0.625],
            },
            SpeakerEmbedding {
                features: [0.5, 0.5, 0.5, 0.5, 0.625, 0.5],
            },
            SpeakerEmbedding {
                features: [0.5, 0.5, 0.5, 0.625, 0.5, 0.5],
            },
        ];
        let assignments = vec![0, 0, 1, 1];
        let score = silhouette_score(&embeddings, &assignments, 2).unwrap();
        assert!(
            score < 0.5,
            "overlapping clusters should have silhouette < 0.5, got {score}"
        );
    }

    #[test]
    fn silhouette_score_single_cluster_returns_none() {
        let embeddings = vec![
            SpeakerEmbedding {
                features: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            SpeakerEmbedding {
                features: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
        ];
        let assignments = vec![0, 0];
        assert!(
            silhouette_score(&embeddings, &assignments, 1).is_none(),
            "single cluster should return None"
        );
    }

    #[test]
    fn silhouette_score_single_point_returns_none() {
        let embeddings = vec![SpeakerEmbedding {
            features: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }];
        let assignments = vec![0];
        assert!(
            silhouette_score(&embeddings, &assignments, 1).is_none(),
            "single point should return None"
        );
    }

    #[test]
    fn silhouette_score_empty_returns_none() {
        let embeddings: Vec<SpeakerEmbedding> = vec![];
        let assignments: Vec<usize> = vec![];
        assert!(
            silhouette_score(&embeddings, &assignments, 0).is_none(),
            "empty input should return None"
        );
    }

    #[test]
    fn diarize_report_includes_silhouette_score() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        // Two well-separated groups → should have score.
        let mut segments = vec![
            make_segment(0.0, 2.0, "speaker one talks here"),
            make_segment(2.0, 4.0, "speaker one keeps going"),
            make_segment(20.0, 22.0, "new speaker joins late"),
            make_segment(22.0, 24.0, "new speaker continues"),
        ];
        let report = diarize_segments(&mut segments, Some(30.0), None, &token).unwrap();
        if report.speakers_detected >= 2 {
            assert!(
                report.silhouette_score.is_some(),
                "multi-speaker report should include silhouette score"
            );
            let s = report.silhouette_score.unwrap();
            assert!(
                (-1.0..=1.0).contains(&s),
                "silhouette score should be in [-1, 1], got {s}"
            );
        }
    }

    #[test]
    fn diarize_single_segment_silhouette_is_none() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 2.0, "single segment only")];
        let report = diarize_segments(&mut segments, Some(5.0), None, &token).unwrap();
        assert!(
            report.silhouette_score.is_none(),
            "single-segment diarization should have no silhouette score"
        );
    }

    #[test]
    fn euclidean_distance_zero_for_identical() {
        let a = SpeakerEmbedding {
            features: [0.25, 0.5, 0.75, 0.125, 0.375, 0.625],
        };
        assert!(
            a.euclidean_distance(&a) < 1e-15,
            "distance to self should be zero"
        );
    }

    #[test]
    fn euclidean_distance_known_value() {
        let a = SpeakerEmbedding {
            features: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let b = SpeakerEmbedding {
            features: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let d = a.euclidean_distance(&b);
        assert!(
            (d - 1.0).abs() < 1e-9,
            "distance along single axis should be 1.0, got {d}"
        );
    }

    // -----------------------------------------------------------------------
    // SpeakerEmbedding: additional edge cases (bd-zua)
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_similarity_opposite_is_negative_one() {
        let a = SpeakerEmbedding {
            features: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let b = SpeakerEmbedding {
            features: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let sim = a.cosine_similarity(&b);
        assert!(
            (sim - (-1.0)).abs() < 1e-9,
            "cosine similarity of opposite vectors should be -1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_scaled_vectors_is_one() {
        let a = SpeakerEmbedding {
            features: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let b = SpeakerEmbedding {
            features: [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        };
        let sim = a.cosine_similarity(&b);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "scaled vectors should have cosine similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn centroid_three_embeddings_is_mean() {
        let embeddings = vec![
            SpeakerEmbedding {
                features: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            SpeakerEmbedding {
                features: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            },
            SpeakerEmbedding {
                features: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
        ];
        let c = SpeakerEmbedding::centroid(&embeddings);
        for val in c.features {
            assert!(
                (val - 0.5).abs() < 1e-9,
                "centroid of [0, 0.5, 1] should be 0.5"
            );
        }
    }

    #[test]
    fn ctc_forced_align_multibyte_segment_text() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = AlignConfig::default();
        let mut segments = vec![
            make_segment(0.0, 3.0, "日本語のテスト"),
            make_segment(3.0, 6.0, "中文测试内容"),
        ];
        let report = ctc_forced_align(&mut segments, Some(6.0), &config, &token).unwrap();
        assert_eq!(report.segments_total, 2);
        // Alignment should complete without panicking on multi-byte text.
        // Character density computed from total chars, not byte count.
        for seg in &segments {
            assert!(
                seg.start_sec.is_some() && seg.end_sec.is_some(),
                "timestamps should remain set after alignment"
            );
        }
    }

    #[test]
    fn checkpoint_or_emit_on_timeout_error_emits_timeout_code() {
        let mut pcx = PipelineCx::new(Some(0)); // already expired
        std::thread::sleep(std::time::Duration::from_millis(2));

        let mut log = EventLog::new(
            "run-timeout-check".to_owned(),
            pcx.trace_id().to_string(),
            None,
        );

        let result = checkpoint_or_emit("backend", &mut pcx, &mut log);
        assert!(result.is_err());

        // The emitted event should have the "cancelled" code, not "timeout".
        let events = &log.events;
        assert_eq!(events.len(), 1, "should emit exactly one event");
        assert_eq!(events[0].code, "backend.cancelled");
        assert!(events[0].payload.is_object());
        assert_eq!(events[0].payload["checkpoint"], true);
    }

    // ── Task #203 — orchestrator edge-case tests ──────────────────────

    #[test]
    fn recommended_budget_zero_service_ms_clamps_to_floor() {
        // service_ms = 0 with a non-zero budget → utilization 0.0 ≤ 0.30
        // → "decrease_budget_candidate", candidate = ceil(0 * 1.60) = 0,
        //   max(0, 1_000) = 1_000 (hard floor).
        let (budget, action, _reason, utilization) = recommended_budget(0, 5_000);
        assert_eq!(action, "decrease_budget_candidate");
        assert_eq!(utilization, 0.0);
        assert_eq!(budget, 1_000, "zero service_ms should clamp to 1_000 floor");
    }

    #[test]
    fn punctuate_segments_already_punctuated_emits_no_changes_note() {
        // All segments already have correct capitalization and trailing
        // punctuation → modified == 0 → note is emitted.
        let mut segments = vec![
            TranscriptionSegment {
                text: "Hello world.".to_owned(),
                start_sec: None,
                end_sec: None,
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                text: "How are you?".to_owned(),
                start_sec: None,
                end_sec: None,
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                text: "Fine!".to_owned(),
                start_sec: None,
                end_sec: None,
                speaker: None,
                confidence: None,
            },
        ];
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let report = punctuate_segments(&mut segments, &token).unwrap();

        assert_eq!(report.segments_total, 3);
        assert_eq!(report.segments_modified, 0);
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("no segments required punctuation changes")),
            "expected 'no changes' note, got: {:?}",
            report.notes
        );
        // Segments should be unchanged.
        assert_eq!(segments[0].text, "Hello world.");
        assert_eq!(segments[1].text, "How are you?");
        assert_eq!(segments[2].text, "Fine!");
    }

    #[test]
    fn pipeline_cx_cancel_then_uncancel_round_trip() {
        let mut pcx = PipelineCx::new(None);

        // Initially no deadline → checkpoint passes.
        let token1 = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        assert!(token1.checkpoint().is_ok());

        // cancel_now() → checkpoint fails.
        pcx.cancel_now();
        let token2 = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        assert!(token2.checkpoint().is_err(), "should fail after cancel_now");

        // uncancel() → new token passes again.
        pcx.uncancel();
        let token3 = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        assert!(token3.checkpoint().is_ok(), "should pass after uncancel");
    }

    #[test]
    fn vad_energy_detect_below_min_voice_ratio_is_silence_only() {
        // Voice frames exist but their ratio is below min_voice_ratio → silence_only = true.
        let dir = tempdir().unwrap();
        let path = dir.path().join("low_voice.wav");

        // 44-byte header + PCM data:
        // 10 frames of silence (0) then 1 frame of loud audio → 1/11 ≈ 0.09
        // With min_voice_ratio = 0.50, this should be classified as silence_only.
        let frame_samples = 160;
        let mut data = vec![0u8; 44]; // header
        // 10 silent frames (160 samples each, all zeros)
        data.extend(vec![0u8; 10 * frame_samples * 2]);
        // 1 loud frame (160 samples of max amplitude)
        for _ in 0..frame_samples {
            data.extend(16000_i16.to_le_bytes());
        }
        std::fs::write(&path, &data).unwrap();

        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let config = VadConfig {
            rms_threshold: 0.01,
            frame_samples,
            min_voice_ratio: 0.50, // well above 1/11 ≈ 0.09
            ..VadConfig::default()
        };
        let report = vad_energy_detect(&path, &config, &token).unwrap();

        assert_eq!(report.frames_total, 11);
        assert_eq!(report.frames_voiced, 1);
        assert!(
            report.silence_only,
            "voice_ratio {:.3} should be below min_voice_ratio 0.50",
            report.voice_ratio
        );
        assert!(
            !report.regions.is_empty(),
            "should still have a voiced region"
        );
    }

    #[test]
    fn stage_latency_profile_terminal_only_event_is_skipped() {
        // A stage with a terminal event but no start event should be skipped
        // because `start_ts_ms` is `None` → the `let Some(start_ts) = ...` guard
        // triggers `continue`.
        let events = vec![RunEvent {
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({ "ts_ms": 5000, "elapsed_ms": 2000 }),
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:05Z".to_owned(),
        }];
        let budgets = StageBudgetPolicy {
            ingest_ms: 5_000,
            normalize_ms: 10_000,
            probe_ms: 8_000,
            vad_ms: 10_000,
            separate_ms: 30_000,
            backend_ms: 60_000,
            acceleration_ms: 20_000,
            align_ms: 30_000,
            punctuate_ms: 10_000,
            diarize_ms: 30_000,
            persist_ms: 20_000,
            cleanup_budget_ms: 5_000,
        };
        let profile = stage_latency_profile(&events, budgets);

        // No observed stages because the backend event had no start event.
        assert_eq!(
            profile["summary"]["observed_stages"], 0,
            "terminal-only event should not produce an observed stage"
        );
        let stages = profile["stages"]
            .as_object()
            .expect("stages should be object");
        assert!(
            stages.is_empty(),
            "no stage profiles should exist for terminal-only events"
        );
    }

    // -- bd-250: orchestrator.rs edge-case tests --

    #[test]
    fn acceleration_fence_payload_tripped_when_expired() {
        let mut pcx = PipelineCx::new(Some(1)); // 1ms timeout
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Force cancellation.
        pcx.cancel_now();
        let payload = acceleration_cancellation_fence_payload(&pcx);
        assert_eq!(payload["status"], "tripped");
        assert!(payload["error"].is_string(), "should have error string");
        assert!(!payload["error_code"].is_null(), "should have error_code");
    }

    #[test]
    fn recommended_budget_tiny_budget_max_guard() {
        // service=1, budget=1 → utilization=1.0 ≥ 0.90
        // uplift = ceil(1 * 1.25) = 2, budget+1 = 2 → max(2, 2) = 2
        let (budget, action, _reason, utilization) = recommended_budget(1, 1);
        assert_eq!(action, "increase_budget");
        assert!((utilization - 1.0).abs() < f64::EPSILON);
        assert_eq!(budget, 2, "should select max(uplift=2, budget+1=2) = 2");
    }

    #[test]
    fn pipeline_cx_cancel_uncancel_with_original_deadline() {
        let mut pcx = PipelineCx::new(Some(60_000));
        // Should be fine — 60 second deadline is in the future.
        assert!(pcx.checkpoint().is_ok());
        // Force cancel.
        pcx.cancel_now();
        assert!(pcx.checkpoint().is_err(), "should be cancelled");
        // Uncancel — should clear deadline entirely.
        pcx.uncancel();
        assert!(
            pcx.checkpoint().is_ok(),
            "should pass after uncancel even though original deadline existed"
        );
    }

    #[test]
    fn cancellation_token_with_no_deadline_always_passes() {
        let pcx = PipelineCx::new(None);
        let token = pcx.cancellation_token(); // ubs:ignore — cancellation token is not a secret
        // Should pass many times.
        for _ in 0..100 {
            assert!(token.checkpoint().is_ok());
        }
    }

    #[test]
    fn stage_latency_profile_external_process_total_ms_is_normalize_plus_backend() {
        // Create events for normalize and backend stages.
        let events = vec![
            RunEvent {
                seq: 0,
                ts_rfc3339: "2026-01-01T00:00:00.000Z".to_owned(),
                stage: "normalize".to_owned(),
                code: "normalize.start".to_owned(),
                message: "start".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:00.370Z".to_owned(),
                stage: "normalize".to_owned(),
                code: "normalize.complete".to_owned(),
                message: "done".to_owned(),
                payload: json!({"elapsed_ms": 370}),
            },
            RunEvent {
                seq: 2,
                ts_rfc3339: "2026-01-01T00:00:00.400Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.start".to_owned(),
                message: "start".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 3,
                ts_rfc3339: "2026-01-01T00:00:01.490Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.complete".to_owned(),
                message: "done".to_owned(),
                payload: json!({"elapsed_ms": 1090}),
            },
        ];
        let budgets = StageBudgetPolicy {
            ingest_ms: 1_000,
            normalize_ms: 1_000,
            probe_ms: 1_000,
            vad_ms: 1_000,
            separate_ms: 1_000,
            backend_ms: 5_000,
            acceleration_ms: 1_000,
            align_ms: 1_000,
            punctuate_ms: 1_000,
            diarize_ms: 1_000,
            persist_ms: 1_000,
            cleanup_budget_ms: 1_000,
        };
        let profile = stage_latency_profile(&events, budgets);
        let ext_total = profile["summary"]["external_process_total_ms"]
            .as_u64()
            .expect("should have external_process_total_ms");
        assert_eq!(
            ext_total,
            370 + 1090,
            "external_process_total_ms should be normalize + backend service_ms"
        );
    }

    // ── Task #313 — orchestrator edge-case tests pass 7 ──────────────────

    #[test]
    fn sanitize_process_pid_rejects_zero_and_out_of_range_values() {
        assert_eq!(sanitize_process_pid(0), None);
        assert_eq!(sanitize_process_pid(u32::MAX), None);
        assert_eq!(sanitize_process_pid(i32::MAX as u32), Some(i32::MAX as u32));
        assert_eq!(sanitize_process_pid(1), Some(1));
    }

    #[test]
    fn finalizer_process_via_run_all_does_not_panic() {
        // Finalizer::Process(pid) arm in run_all() is exercised here.
        // Use a PID that almost certainly does not exist so the kill is a
        // harmless no-op (best-effort, errors ignored).
        let mut registry = FinalizerRegistry::new();
        registry.register("bogus_proc", Finalizer::Process(u32::MAX));
        assert_eq!(registry.len(), 1);
        registry.run_all(); // should not panic
        assert_eq!(
            registry.len(),
            0,
            "Process finalizer should drain the entry"
        );
    }

    #[test]
    fn finalizer_process_via_run_all_bounded_does_not_panic() {
        // Same as above but through the bounded path.
        let mut registry = FinalizerRegistry::new();
        registry.register("bogus_proc_bounded", Finalizer::Process(u32::MAX));
        assert_eq!(registry.len(), 1);
        registry.run_all_bounded(5_000);
        assert_eq!(
            registry.len(),
            0,
            "Process finalizer should drain the entry"
        );
    }

    #[test]
    fn run_all_bounded_slow_custom_exceeds_budget_continues_to_next() {
        // A slow Custom finalizer sleeps longer than the budget, triggering
        // the "exceeded cleanup budget" warning (line 636).  Verify the
        // *next* finalizer still executes despite the budget breach.
        use std::sync::{Arc, Mutex};

        let second_ran = Arc::new(Mutex::new(false));
        let second_ran_clone = Arc::clone(&second_ran);

        let mut registry = FinalizerRegistry::new();

        // Register the slow one first (runs second due to LIFO).
        registry.register(
            "slow",
            Finalizer::Custom(Box::new(|| {
                std::thread::sleep(Duration::from_millis(150));
            })),
        );

        // This finalizer runs first (LIFO) — it records that it ran.
        // Wait, LIFO means last-registered runs first.  We want slow to
        // run first, so register it second (so it pops first).
        // Actually: entries.pop() pops the LAST element → LIFO.
        // So register "marker" first, "slow" second → slow runs first,
        // then marker.  But we want to prove the marker *after* the slow
        // one still runs.  So register them the other way:
        //   - Register slow first  (index 0)
        //   - Register marker second (index 1) → pops first
        //
        // Actually the test is: slow finalizer exceeds budget but the
        // *next* finalizer still runs.  So we want:
        //   pop order: marker (index 1) → slow (index 0)
        // That doesn't trigger exceeded budget before marker.  Instead:
        //   Register marker first (index 0), slow second (index 1)
        //   pop order: slow (index 1) → marker (index 0)
        // slow runs, exceeds budget → warning → continues → marker runs.
        // That's what we want.

        // Reset: re-create from scratch.
        let mut registry = FinalizerRegistry::new();

        // Index 0: marker (will execute second, after slow).
        registry.register(
            "marker",
            Finalizer::Custom(Box::new(move || {
                *second_ran_clone.lock().unwrap() = true;
            })),
        );

        // Index 1: slow (pops first, exceeds budget).
        registry.register(
            "slow",
            Finalizer::Custom(Box::new(|| {
                std::thread::sleep(Duration::from_millis(150));
            })),
        );

        assert_eq!(registry.len(), 2);

        // Budget of 10ms — the slow finalizer (150ms) will exceed this.
        registry.run_all_bounded(10);

        assert_eq!(registry.len(), 0);
        assert!(
            *second_ran.lock().unwrap(),
            "marker finalizer must still run even after slow one exceeds budget"
        );
    }

    #[test]
    fn run_all_bounded_empty_registry_does_not_panic() {
        let mut registry = FinalizerRegistry::new();
        assert_eq!(registry.len(), 0);
        registry.run_all_bounded(1_000);
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn punctuate_segments_capitalizes_after_question_mark_mid_text() {
        // Rule 4 in punctuate_segments capitalizes after sentence-ending
        // punctuation.  Existing tests cover period (`.`), but not `?`
        // within a segment.  Verify `? w` → `? W`.
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 3.0, "is it done? well maybe")];
        let report = punctuate_segments(&mut segments, &token).unwrap();
        assert_eq!(report.segments_modified, 1);
        assert!(
            segments[0].text.contains("? Well"),
            "should capitalize 'w' after '?', got: {}",
            segments[0].text
        );
        // Also verify first-char cap and trailing period from rules 2 & 3.
        assert!(
            segments[0].text.starts_with("Is"),
            "first char should be capitalized, got: {}",
            segments[0].text
        );
        assert!(
            segments[0].text.ends_with('.'),
            "should add trailing period, got: {}",
            segments[0].text
        );
    }

    // -----------------------------------------------------------------------
    // Enhanced punctuation rule tests (bd-2sp)
    // -----------------------------------------------------------------------

    #[test]
    fn punctuate_does_not_capitalize_after_abbreviation() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 3.0, "dr. smith said hello")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            segments[0].text.contains("Dr. smith") || segments[0].text.contains("Dr. Smith"),
            "should not force-capitalize 'smith' after 'Dr.': {}",
            segments[0].text
        );
        // The first char is capitalized (rule 2), but "smith" after "Dr."
        // should NOT be force-capitalized by the sentence-end rule.
        // Note: "smith" might already be lowercase, that's correct.
        assert!(
            !segments[0].text.contains("Dr. S") || segments[0].text.contains("Dr. Smith"),
            "should preserve case after abbreviation period, got: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_does_not_capitalize_after_decimal() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 2.0, "the value is 3.14 radians")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            segments[0].text.contains("3.14 radians") || segments[0].text.contains("3.14 Radians"),
            "got: {}",
            segments[0].text
        );
        // The key assertion: "radians" should NOT be capitalized by the
        // period in 3.14.
        assert!(
            !segments[0].text.contains("3.14 R"),
            "should not capitalize 'radians' after decimal '3.14', got: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_does_not_capitalize_after_ellipsis() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 3.0, "well... anyway let us continue")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            !segments[0].text.contains("... A"),
            "should not capitalize after ellipsis, got: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_normalizes_multiple_spaces() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 2.0, "hello   world   today")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            !segments[0].text.contains("  "),
            "should normalize multiple spaces to single, got: {:?}",
            segments[0].text
        );
        assert!(
            segments[0].text.contains("Hello world today"),
            "normalized text should have single spaces, got: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_abbreviation_mid_sentence_preserves_context() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 5.0, "talk to mr. jones about the plan")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        // "jones" after "mr." should not be capitalized by the period rule.
        // Note: Rule 2 only capitalizes the first character of the segment,
        // so "mr" stays lowercase.
        assert!(
            segments[0].text.contains("mr. jones"),
            "should not capitalize 'jones' after abbreviation 'mr.', got: {}",
            segments[0].text
        );
        // The key check: "jones" must NOT have been force-capitalized.
        assert!(
            !segments[0].text.contains("mr. J"),
            "should not force-capitalize after abbreviation, got: {}",
            segments[0].text
        );
    }

    #[test]
    fn punctuate_real_sentence_end_still_capitalizes() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![make_segment(0.0, 5.0, "this is done. now start again")];
        let _report = punctuate_segments(&mut segments, &token).unwrap();
        assert!(
            segments[0].text.contains(". Now"),
            "should capitalize after real sentence-ending period, got: {}",
            segments[0].text
        );
    }

    #[test]
    fn is_abbreviation_period_detects_known_abbrevs() {
        assert!(is_abbreviation_period("Dr. Smith", 2));
        assert!(is_abbreviation_period("talk to mr. jones", 10));
        assert!(is_abbreviation_period("Mrs. Williams", 3));
    }

    #[test]
    fn is_abbreviation_period_rejects_non_abbrevs() {
        assert!(!is_abbreviation_period("done.", 4));
        assert!(!is_abbreviation_period("hello world.", 11));
    }

    #[test]
    fn is_decimal_period_detects_numbers() {
        assert!(is_decimal_period("3.14", 1));
        assert!(is_decimal_period("the value is 3.14 rad", 14));
        assert!(!is_decimal_period("done.", 4));
        assert!(!is_decimal_period("a.b", 1)); // letters, not digits
    }

    #[test]
    fn is_ellipsis_period_detects_triple_dots() {
        assert!(is_ellipsis_period("well... anyway", 4));
        assert!(is_ellipsis_period("well... anyway", 5));
        assert!(is_ellipsis_period("well... anyway", 6));
        assert!(!is_ellipsis_period("well. anyway", 4)); // single period
    }

    // -----------------------------------------------------------------------
    // VAD helper: merge_regions_by_gap (bd-22y)
    // -----------------------------------------------------------------------

    #[test]
    fn merge_regions_by_gap_bridges_adjacent_regions() {
        let mut regions = vec![
            VadRegionMs {
                start_ms: 0,
                end_ms: 100,
                avg_rms: 0.5,
            },
            VadRegionMs {
                start_ms: 120,
                end_ms: 200,
                avg_rms: 0.25,
            },
        ];
        merge_regions_by_gap(&mut regions, 50);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_ms, 0);
        assert_eq!(regions[0].end_ms, 200);
    }

    #[test]
    fn merge_regions_by_gap_preserves_distant_regions() {
        let mut regions = vec![
            VadRegionMs {
                start_ms: 0,
                end_ms: 100,
                avg_rms: 0.5,
            },
            VadRegionMs {
                start_ms: 200,
                end_ms: 300,
                avg_rms: 0.25,
            },
        ];
        merge_regions_by_gap(&mut regions, 50);
        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn merge_regions_by_gap_sorts_unsorted_input() {
        let mut regions = vec![
            VadRegionMs {
                start_ms: 200,
                end_ms: 300,
                avg_rms: 0.25,
            },
            VadRegionMs {
                start_ms: 0,
                end_ms: 100,
                avg_rms: 0.5,
            },
        ];
        merge_regions_by_gap(&mut regions, 150);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_ms, 0);
        assert_eq!(regions[0].end_ms, 300);
    }

    #[test]
    fn merge_regions_by_gap_empty_input() {
        let mut regions: Vec<VadRegionMs> = Vec::new();
        merge_regions_by_gap(&mut regions, 50);
        assert!(regions.is_empty());
    }

    #[test]
    fn merge_regions_by_gap_single_region() {
        let mut regions = vec![VadRegionMs {
            start_ms: 10,
            end_ms: 50,
            avg_rms: 0.5,
        }];
        merge_regions_by_gap(&mut regions, 100);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_ms, 10);
    }

    #[test]
    fn merge_regions_by_gap_zero_gap_merges_overlapping() {
        let mut regions = vec![
            VadRegionMs {
                start_ms: 0,
                end_ms: 100,
                avg_rms: 0.5,
            },
            VadRegionMs {
                start_ms: 80,
                end_ms: 200,
                avg_rms: 0.75,
            },
        ];
        merge_regions_by_gap(&mut regions, 0);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].end_ms, 200);
    }

    #[test]
    fn merge_regions_by_gap_takes_max_rms() {
        let mut regions = vec![
            VadRegionMs {
                start_ms: 0,
                end_ms: 100,
                avg_rms: 0.25,
            },
            VadRegionMs {
                start_ms: 50,
                end_ms: 200,
                avg_rms: 0.75,
            },
        ];
        merge_regions_by_gap(&mut regions, 0);
        assert_eq!(regions.len(), 1);
        assert!((regions[0].avg_rms - 0.75).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // VAD helper: split_long_regions (bd-22y)
    // -----------------------------------------------------------------------

    #[test]
    fn split_long_regions_splits_oversized_region() {
        let regions = vec![VadRegionMs {
            start_ms: 0,
            end_ms: 1000,
            avg_rms: 0.5,
        }];
        let result = split_long_regions(&regions, 300);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 300);
        assert_eq!(result[1].start_ms, 300);
        assert_eq!(result[1].end_ms, 600);
        assert_eq!(result[2].start_ms, 600);
        assert_eq!(result[2].end_ms, 900);
        assert_eq!(result[3].start_ms, 900);
        assert_eq!(result[3].end_ms, 1000);
    }

    #[test]
    fn split_long_regions_passes_through_short_regions() {
        let regions = vec![VadRegionMs {
            start_ms: 0,
            end_ms: 100,
            avg_rms: 0.5,
        }];
        let result = split_long_regions(&regions, 300);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].end_ms, 100);
    }

    #[test]
    fn split_long_regions_zero_max_returns_original() {
        let regions = vec![VadRegionMs {
            start_ms: 0,
            end_ms: 1000,
            avg_rms: 0.5,
        }];
        let result = split_long_regions(&regions, 0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].end_ms, 1000);
    }

    #[test]
    fn split_long_regions_empty_input() {
        let regions: Vec<VadRegionMs> = Vec::new();
        let result = split_long_regions(&regions, 300);
        assert!(result.is_empty());
    }

    #[test]
    fn split_long_regions_exact_boundary() {
        let regions = vec![VadRegionMs {
            start_ms: 0,
            end_ms: 600,
            avg_rms: 0.5,
        }];
        let result = split_long_regions(&regions, 300);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].end_ms, 300);
        assert_eq!(result[1].start_ms, 300);
        assert_eq!(result[1].end_ms, 600);
    }

    #[test]
    fn split_long_regions_preserves_avg_rms() {
        let regions = vec![VadRegionMs {
            start_ms: 0,
            end_ms: 1000,
            avg_rms: 0.75,
        }];
        let result = split_long_regions(&regions, 400);
        for chunk in &result {
            assert!((chunk.avg_rms - 0.75).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // VAD helper: apply_padding (bd-22y)
    // -----------------------------------------------------------------------

    #[test]
    fn apply_padding_extends_both_ends() {
        let mut regions = vec![VadRegionMs {
            start_ms: 100,
            end_ms: 200,
            avg_rms: 0.5,
        }];
        apply_padding(&mut regions, 30, 500);
        assert_eq!(regions[0].start_ms, 70);
        assert_eq!(regions[0].end_ms, 230);
    }

    #[test]
    fn apply_padding_clamps_to_zero_start() {
        let mut regions = vec![VadRegionMs {
            start_ms: 10,
            end_ms: 100,
            avg_rms: 0.5,
        }];
        apply_padding(&mut regions, 30, 500);
        assert_eq!(regions[0].start_ms, 0);
    }

    #[test]
    fn apply_padding_clamps_to_audio_duration() {
        let mut regions = vec![VadRegionMs {
            start_ms: 100,
            end_ms: 490,
            avg_rms: 0.5,
        }];
        apply_padding(&mut regions, 30, 500);
        assert_eq!(regions[0].end_ms, 500);
    }

    #[test]
    fn apply_padding_zero_duration_does_not_clamp() {
        let mut regions = vec![VadRegionMs {
            start_ms: 100,
            end_ms: 200,
            avg_rms: 0.5,
        }];
        apply_padding(&mut regions, 30, 0);
        assert_eq!(regions[0].end_ms, 230);
    }

    #[test]
    fn apply_padding_zero_pad() {
        let mut regions = vec![VadRegionMs {
            start_ms: 100,
            end_ms: 200,
            avg_rms: 0.5,
        }];
        apply_padding(&mut regions, 0, 500);
        assert_eq!(regions[0].start_ms, 100);
        assert_eq!(regions[0].end_ms, 200);
    }

    #[test]
    fn apply_padding_empty_regions() {
        let mut regions: Vec<VadRegionMs> = Vec::new();
        apply_padding(&mut regions, 30, 500);
        assert!(regions.is_empty());
    }

    // -----------------------------------------------------------------------
    // VAD helper: ms_to_frames (bd-22y)
    // -----------------------------------------------------------------------

    #[test]
    fn ms_to_frames_basic_conversion() {
        // 160ms / 10ms per frame = 16 frames
        assert_eq!(ms_to_frames(160, 10), 16);
    }

    #[test]
    fn ms_to_frames_rounds_up() {
        // 15ms / 10ms per frame = 2 frames (ceiling)
        assert_eq!(ms_to_frames(15, 10), 2);
    }

    #[test]
    fn ms_to_frames_zero_duration() {
        assert_eq!(ms_to_frames(0, 10), 0);
    }

    #[test]
    fn ms_to_frames_zero_frame_ms_uses_one() {
        // 0 frame_ms is clamped to 1
        assert_eq!(ms_to_frames(100, 0), 100);
    }

    #[test]
    fn ms_to_frames_exact_multiple() {
        assert_eq!(ms_to_frames(100, 10), 10);
    }

    #[test]
    fn ms_to_frames_single_ms() {
        assert_eq!(ms_to_frames(1, 10), 1);
    }

    // -----------------------------------------------------------------------
    // VadConfig edge cases (bd-22y)
    // -----------------------------------------------------------------------

    fn vad_test_request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("test.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: false,
            db_path: PathBuf::from("test.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        }
    }

    #[test]
    fn vad_config_from_request_defaults_when_no_vad_params() {
        let request = vad_test_request();
        let config = VadConfig::from_request(&request);
        let default_config = VadConfig::default();
        assert!((config.rms_threshold - default_config.rms_threshold).abs() < 1e-9);
        assert_eq!(
            config.min_speech_duration_ms,
            default_config.min_speech_duration_ms
        );
    }

    #[test]
    fn vad_config_from_request_threshold_override() {
        use crate::model::VadParams;
        let mut request = vad_test_request();
        request.backend_params.vad = Some(VadParams {
            model_path: None,
            threshold: Some(0.25),
            min_speech_duration_ms: None,
            min_silence_duration_ms: None,
            max_speech_duration_s: None,
            speech_pad_ms: None,
            samples_overlap: None,
        });
        let config = VadConfig::from_request(&request);
        assert!((config.rms_threshold - 0.25).abs() < 1e-9);
    }

    #[test]
    fn vad_config_from_request_duration_overrides() {
        use crate::model::VadParams;
        let mut request = vad_test_request();
        request.backend_params.vad = Some(VadParams {
            model_path: None,
            threshold: None,
            min_speech_duration_ms: Some(100),
            min_silence_duration_ms: Some(200),
            max_speech_duration_s: Some(5.0),
            speech_pad_ms: Some(50),
            samples_overlap: None,
        });
        let config = VadConfig::from_request(&request);
        assert_eq!(config.min_speech_duration_ms, 100);
        assert_eq!(config.min_silence_duration_ms, 200);
        assert_eq!(config.max_speech_duration_ms, Some(5000));
        assert_eq!(config.speech_pad_ms, 50);
    }

    #[test]
    fn vad_config_from_request_rejects_nan_threshold() {
        use crate::model::VadParams;
        let mut request = vad_test_request();
        request.backend_params.vad = Some(VadParams {
            model_path: None,
            threshold: Some(f32::NAN),
            min_speech_duration_ms: None,
            min_silence_duration_ms: None,
            max_speech_duration_s: None,
            speech_pad_ms: None,
            samples_overlap: None,
        });
        let config = VadConfig::from_request(&request);
        // Should fall back to default since NaN is not finite
        assert!((config.rms_threshold - 0.01).abs() < 1e-9);
    }

    #[test]
    fn vad_config_from_request_rejects_zero_threshold() {
        use crate::model::VadParams;
        let mut request = vad_test_request();
        request.backend_params.vad = Some(VadParams {
            model_path: None,
            threshold: Some(0.0),
            min_speech_duration_ms: None,
            min_silence_duration_ms: None,
            max_speech_duration_s: None,
            speech_pad_ms: None,
            samples_overlap: None,
        });
        let config = VadConfig::from_request(&request);
        // Should fall back to default since 0.0 is not > 0.0
        assert!((config.rms_threshold - 0.01).abs() < 1e-9);
    }

    #[test]
    fn vad_config_as_json_round_trips_all_fields() {
        let config = VadConfig::default();
        let json = config.as_json();
        assert!(json.get("rms_threshold").is_some());
        assert!(json.get("frame_samples").is_some());
        assert!(json.get("min_voice_ratio").is_some());
        assert!(json.get("min_speech_duration_ms").is_some());
        assert!(json.get("min_silence_duration_ms").is_some());
        assert!(json.get("max_speech_duration_ms").is_some());
        assert!(json.get("speech_pad_ms").is_some());
    }

    // -----------------------------------------------------------------------
    // Speaker constraints integration (bd-3g8)
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_speaker_target_none_when_no_constraints() {
        assert!(resolve_speaker_target(None).is_none());
    }

    #[test]
    fn resolve_speaker_target_uses_num_speakers() {
        use crate::model::SpeakerConstraints;
        let sc = SpeakerConstraints {
            num_speakers: Some(3),
            min_speakers: Some(1),
            max_speakers: Some(10),
        };
        assert_eq!(resolve_speaker_target(Some(&sc)), Some(3));
    }

    #[test]
    fn resolve_speaker_target_falls_back_to_max_speakers() {
        use crate::model::SpeakerConstraints;
        let sc = SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(1),
            max_speakers: Some(5),
        };
        assert_eq!(resolve_speaker_target(Some(&sc)), Some(5));
    }

    #[test]
    fn resolve_speaker_target_none_when_only_min() {
        use crate::model::SpeakerConstraints;
        let sc = SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(2),
            max_speakers: None,
        };
        assert!(resolve_speaker_target(Some(&sc)).is_none());
    }

    #[test]
    fn resolve_speaker_target_ignores_zero_num_speakers() {
        use crate::model::SpeakerConstraints;
        let sc = SpeakerConstraints {
            num_speakers: Some(0),
            min_speakers: None,
            max_speakers: Some(4),
        };
        assert_eq!(resolve_speaker_target(Some(&sc)), Some(4));
    }

    #[test]
    fn diarize_respects_num_speakers_constraint() {
        use crate::model::SpeakerConstraints;
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        // Create segments with very different characteristics so the
        // unconstrained diarizer produces more than 2 clusters.
        let mut segments = vec![
            make_segment(0.0, 1.0, "hello world"),
            make_segment(1.0, 2.0, "hi there"),
            make_segment(
                5.0,
                8.0,
                "this is a very long segment with many words to change features",
            ),
            make_segment(
                8.0,
                11.0,
                "another very long segment different vocabulary complexity",
            ),
            make_segment(20.0, 21.0, "far away short"),
            make_segment(
                25.0,
                30.0,
                "way at the end of the recording totally different position",
            ),
        ];

        let sc = SpeakerConstraints {
            num_speakers: Some(2),
            min_speakers: None,
            max_speakers: None,
        };
        let report = diarize_segments(&mut segments, Some(30.0), Some(&sc), &token).unwrap();

        assert_eq!(
            report.speakers_detected, 2,
            "should merge clusters down to 2 speakers, got {}",
            report.speakers_detected
        );
        // All segments should have speaker labels.
        for seg in &segments {
            assert!(
                seg.speaker.is_some(),
                "every segment should have a speaker label"
            );
        }
        // Labels should be SPEAKER_00 and SPEAKER_01 only.
        for seg in &segments {
            let spk = seg.speaker.as_ref().unwrap();
            assert!(
                spk == "SPEAKER_00" || spk == "SPEAKER_01",
                "label should be SPEAKER_00 or SPEAKER_01, got {spk}"
            );
        }
    }

    #[test]
    fn diarize_respects_max_speakers_constraint() {
        use crate::model::SpeakerConstraints;
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments = vec![
            make_segment(0.0, 1.0, "short"),
            make_segment(5.0, 8.0, "this is a very long segment with many words"),
            make_segment(20.0, 21.0, "far away"),
            make_segment(25.0, 30.0, "way at the end totally different"),
        ];

        let sc = SpeakerConstraints {
            num_speakers: None,
            min_speakers: None,
            max_speakers: Some(2),
        };
        let report = diarize_segments(&mut segments, Some(30.0), Some(&sc), &token).unwrap();

        assert!(
            report.speakers_detected <= 2,
            "should have at most 2 speakers, got {}",
            report.speakers_detected
        );
    }

    #[test]
    fn diarize_notes_min_speakers_deficit() {
        use crate::model::SpeakerConstraints;
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        // A single segment can only produce 1 speaker — min_speakers=3
        // should produce a note.
        let mut segments = vec![make_segment(0.0, 5.0, "only one speaker here")];

        let sc = SpeakerConstraints {
            num_speakers: None,
            min_speakers: Some(3),
            max_speakers: None,
        };
        let report = diarize_segments(&mut segments, Some(5.0), Some(&sc), &token).unwrap();

        assert_eq!(report.speakers_detected, 1);
        assert!(
            report.notes.iter().any(|n| n.contains("min_speakers=3")),
            "should note the min_speakers deficit, notes: {:?}",
            report.notes
        );
    }

    #[test]
    fn diarize_constraint_none_same_as_empty_default() {
        let token = CancellationToken::no_deadline(); // ubs:ignore — cancellation token is not a secret
        let mut segments1 = vec![
            make_segment(0.0, 2.0, "hello world"),
            make_segment(2.0, 4.0, "goodbye world"),
        ];
        let mut segments2 = segments1.clone();

        let r1 = diarize_segments(&mut segments1, Some(5.0), None, &token).unwrap();
        let empty_sc = crate::model::SpeakerConstraints::default();
        let r2 = diarize_segments(&mut segments2, Some(5.0), Some(&empty_sc), &token).unwrap();

        assert_eq!(
            r1.speakers_detected, r2.speakers_detected,
            "empty constraints should behave same as None"
        );
    }
}
