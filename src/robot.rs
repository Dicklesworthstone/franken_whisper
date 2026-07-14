use std::path::Path;

use serde::Serialize;
use serde_json::{Value, json};

use crate::error::{FwError, FwResult};
use crate::model::{
    AccelerationReport, BackendDiscoveryEntry, BackendKind, BackendsReport, RunEvent, RunReport,
    TranscriptionSegment,
};

pub const ROBOT_SCHEMA_VERSION: &str = "1.0.0";

pub const STAGE_REQUIRED_FIELDS: &[&str] = &[
    "event",
    "schema_version",
    "run_id",
    "seq",
    "ts",
    "stage",
    "code",
    "message",
    "payload",
];
pub const RUN_ERROR_REQUIRED_FIELDS: &[&str] = &["event", "schema_version", "code", "message"];
pub const RUN_START_REQUIRED_FIELDS: &[&str] = &["event", "schema_version", "request"];
pub const RUN_COMPLETE_REQUIRED_FIELDS: &[&str] = &[
    "event",
    "schema_version",
    "run_id",
    "trace_id",
    "started_at",
    "finished_at",
    "backend",
    "language",
    "transcript",
    "segments",
    "acceleration",
    "warnings",
    "evidence",
];
pub const BACKENDS_DISCOVERY_REQUIRED_FIELDS: &[&str] = &["event", "schema_version", "backends"];
pub const ROUTING_DECISION_REQUIRED_FIELDS: &[&str] =
    &["event", "schema_version", "run_id", "ts", "code"];

pub const TRANSCRIPT_PARTIAL_REQUIRED_FIELDS: &[&str] = &[
    "event",
    "schema_version",
    "run_id",
    "seq",
    "ts",
    "text",
    "start_sec",
    "end_sec",
    "confidence",
    "speaker",
];

pub const TRANSCRIPT_CONFIRM_REQUIRED_FIELDS: &[&str] = &[
    "event",
    "schema_version",
    "run_id",
    "seq",
    "window_id",
    "quality_model_id",
    "drift",
    "latency_ms",
    "ts",
];

pub const HEALTH_REPORT_REQUIRED_FIELDS: &[&str] = &[
    "event",
    "schema_version",
    "ts",
    "backends",
    "ffmpeg",
    "database",
    "resources",
    "overall_status",
];

pub const TRANSCRIPT_RETRACT_REQUIRED_FIELDS: &[&str] = &[
    "event",
    "schema_version",
    "run_id",
    "retracted_seq",
    "window_id",
    "reason",
    "quality_model_id",
    "ts",
];

pub const TRANSCRIPT_CORRECT_REQUIRED_FIELDS: &[&str] = &[
    "event",
    "schema_version",
    "run_id",
    "correction_id",
    "replaces_seq",
    "window_id",
    "segments",
    "drift",
    "latency_ms",
    "ts",
];

pub const SPECULATION_STATS_REQUIRED_FIELDS: &[&str] = &[
    "event",
    "schema_version",
    "run_id",
    "windows_processed",
    "corrections_emitted",
    "confirmations_emitted",
    "correction_rate",
    "mean_fast_latency_ms",
    "mean_quality_latency_ms",
    "current_window_size_ms",
    "mean_drift_wer",
    "ts",
];

pub fn emit_robot_start(request_summary: serde_json::Value) -> FwResult<()> {
    emit_line(&run_start_value(request_summary))
}

pub fn emit_robot_error(message: &str, code: &str) -> FwResult<()> {
    emit_line(&run_error_value(message, code))
}

/// Emit a `run_error` event directly from an [`FwError`], using its
/// [`error_code()`](FwError::error_code) as the machine-readable code.
pub fn emit_robot_error_from_fw(error: &FwError) -> FwResult<()> {
    emit_line(&json!({
        "event": "run_error",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "code": error.error_code(),
        "message": error.to_string(),
    }))
}

pub fn emit_robot_stage(run_id: &str, event: &RunEvent) -> FwResult<()> {
    emit_line(&BorrowedStage::new(run_id, event))
}

pub fn emit_robot_complete(report: &RunReport) -> FwResult<()> {
    emit_line(&BorrowedComplete::new(report))
}

pub fn emit_robot_report(report: &RunReport) -> FwResult<()> {
    for event in &report.events {
        emit_robot_stage(&report.run_id, event)?;
    }
    emit_robot_complete(report)
}

/// Build a [`BackendsReport`] by probing all registered engines.
#[must_use]
pub fn build_backends_report() -> BackendsReport {
    let engines = crate::backend::all_engines();
    let backends = engines
        .iter()
        .map(|engine| BackendDiscoveryEntry {
            name: engine.name().to_owned(),
            kind: engine.kind(),
            available: engine.is_available(),
            capabilities: engine.capabilities(),
        })
        .collect();
    BackendsReport { backends }
}

/// Construct the `backends.discovery` NDJSON event value.
#[must_use]
pub fn backends_discovery_value(report: &BackendsReport) -> serde_json::Value {
    json!({
        "event": "backends.discovery",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "backends": report.backends,
    })
}

/// Emit a single `backends.discovery` NDJSON line to stdout.
pub fn emit_robot_backends_discovery(report: &BackendsReport) -> FwResult<()> {
    emit_line(&backends_discovery_value(report))
}

/// Construct the `routing_decision` NDJSON event value for `robot routing-history`.
#[must_use]
pub fn routing_decision_value(
    run_id: &str,
    ts: &str,
    code: &str,
    payload: &Value,
) -> serde_json::Value {
    json!({
        "event": "routing_decision",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "run_id": run_id,
        "ts": ts,
        "code": code,
        "decision_id": payload.get("decision_id"),
        "chosen_action": payload.get("chosen_action"),
        "calibration_score": payload.get("calibration_score"),
        "e_process": payload.get("e_process"),
        "fallback_active": payload.get("fallback_active"),
        "recommended_order": payload.get("recommended_order"),
        "mode": payload.get("mode"),
    })
}

// ---------------------------------------------------------------------------
// bd-20g.3: Streaming partial transcript events
// ---------------------------------------------------------------------------

/// Construct a `transcript.partial` NDJSON event value for a single segment.
#[must_use]
pub fn transcript_partial_value(
    run_id: &str,
    seq: u64,
    ts: &str,
    segment: &TranscriptionSegment,
) -> serde_json::Value {
    json!({
        "event": "transcript.partial",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "run_id": run_id,
        "seq": seq,
        "ts": ts,
        "text": segment.text,
        "start_sec": segment.start_sec,
        "end_sec": segment.end_sec,
        "confidence": segment.confidence,
        "speaker": segment.speaker,
    })
}

/// Emit a single `transcript.partial` NDJSON line to stdout.
pub fn emit_transcript_partial(
    run_id: &str,
    seq: u64,
    ts: &str,
    segment: &TranscriptionSegment,
) -> FwResult<()> {
    emit_line(&transcript_partial_value(run_id, seq, ts, segment))
}

/// Emit `transcript.partial` events for a batch of segments, assigning
/// sequential `seq` numbers starting from `start_seq`.
pub fn emit_transcript_partials(
    run_id: &str,
    start_seq: u64,
    ts: &str,
    segments: &[TranscriptionSegment],
) -> FwResult<()> {
    for (i, segment) in segments.iter().enumerate() {
        let seq = start_seq + i as u64;
        emit_transcript_partial(run_id, seq, ts, segment)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// bd-qlt.2: Speculative cancel-correct streaming robot events
// ---------------------------------------------------------------------------

/// Construct a `transcript.confirm` NDJSON event value.
#[must_use]
pub fn transcript_confirm_value(
    run_id: &str,
    seq: u64,
    window_id: u64,
    drift: &crate::speculation::CorrectionDrift,
    quality_latency_ms: u64,
    quality_model_id: &str,
) -> serde_json::Value {
    json!({
        "event": "transcript.confirm",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "run_id": run_id,
        "seq": seq,
        "window_id": window_id,
        "quality_model_id": quality_model_id,
        "drift": {
            "wer_approx": drift.wer_approx,
            "confidence_delta": drift.confidence_delta,
            "segment_count_delta": drift.segment_count_delta,
            "text_edit_distance": drift.text_edit_distance,
        },
        "latency_ms": quality_latency_ms,
        "ts": chrono::Utc::now().to_rfc3339(),
    })
}

/// Emit a `transcript.confirm` NDJSON line to stdout.
pub fn emit_transcript_confirm(
    run_id: &str,
    seq: u64,
    window_id: u64,
    drift: &crate::speculation::CorrectionDrift,
    quality_latency_ms: u64,
    quality_model_id: &str,
) -> FwResult<()> {
    emit_line(&transcript_confirm_value(
        run_id,
        seq,
        window_id,
        drift,
        quality_latency_ms,
        quality_model_id,
    ))
}

/// Construct a `transcript.retract` NDJSON event value.
#[must_use]
pub fn transcript_retract_value(
    run_id: &str,
    retracted_seq: u64,
    window_id: u64,
    reason: &str,
    quality_model_id: &str,
) -> serde_json::Value {
    json!({
        "event": "transcript.retract",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "run_id": run_id,
        "retracted_seq": retracted_seq,
        "window_id": window_id,
        "reason": reason,
        "quality_model_id": quality_model_id,
        "ts": chrono::Utc::now().to_rfc3339(),
    })
}

/// Emit a `transcript.retract` NDJSON line to stdout.
pub fn emit_transcript_retract(
    run_id: &str,
    retracted_seq: u64,
    window_id: u64,
    reason: &str,
    quality_model_id: &str,
) -> FwResult<()> {
    emit_line(&transcript_retract_value(
        run_id,
        retracted_seq,
        window_id,
        reason,
        quality_model_id,
    ))
}

/// Construct a `transcript.correct` NDJSON event value.
#[must_use]
pub fn transcript_correct_value(
    run_id: &str,
    correction: &crate::speculation::CorrectionEvent,
) -> serde_json::Value {
    json!({
        "event": "transcript.correct",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "run_id": run_id,
        "correction_id": correction.correction_id,
        "replaces_seq": correction.retracted_seq,
        "window_id": correction.window_id,
        "segments": correction.corrected_segments,
        "quality_model_id": correction.quality_model_id,
        "drift": {
            "wer_approx": correction.drift.wer_approx,
            "confidence_delta": correction.drift.confidence_delta,
            "segment_count_delta": correction.drift.segment_count_delta,
            "text_edit_distance": correction.drift.text_edit_distance,
        },
        "latency_ms": correction.quality_latency_ms,
        "ts": correction.corrected_at_rfc3339,
    })
}

/// Emit a `transcript.correct` NDJSON line to stdout.
pub fn emit_transcript_correct(
    run_id: &str,
    correction: &crate::speculation::CorrectionEvent,
) -> FwResult<()> {
    emit_line(&transcript_correct_value(run_id, correction))
}

/// Construct a `transcript.speculation_stats` NDJSON event value.
#[must_use]
pub fn speculation_stats_value(
    run_id: &str,
    stats: &crate::speculation::SpeculationStats,
) -> serde_json::Value {
    json!({
        "event": "transcript.speculation_stats",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "run_id": run_id,
        "windows_processed": stats.windows_processed,
        "corrections_emitted": stats.corrections_emitted,
        "confirmations_emitted": stats.confirmations_emitted,
        "correction_rate": stats.correction_rate,
        "mean_fast_latency_ms": stats.mean_fast_latency_ms,
        "mean_quality_latency_ms": stats.mean_quality_latency_ms,
        "current_window_size_ms": stats.current_window_size_ms,
        "mean_drift_wer": stats.mean_drift_wer,
        "ts": chrono::Utc::now().to_rfc3339(),
    })
}

/// Emit a `transcript.speculation_stats` NDJSON line to stdout.
pub fn emit_speculation_stats(
    run_id: &str,
    stats: &crate::speculation::SpeculationStats,
) -> FwResult<()> {
    emit_line(&speculation_stats_value(run_id, stats))
}

// ---------------------------------------------------------------------------
// bd-20g.5: Health report for system readiness check
// ---------------------------------------------------------------------------

/// Status of a dependency check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Ok,
    Degraded,
    Unavailable,
}

impl CheckStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Degraded => "degraded",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Result of checking a single dependency or subsystem.
#[derive(Debug, Clone)]
pub struct DependencyCheck {
    pub name: String,
    pub available: bool,
    pub path: Option<String>,
    pub version: Option<String>,
    pub issues: Vec<String>,
}

/// System resource snapshot.
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    pub disk_free_bytes: Option<u64>,
    pub disk_total_bytes: Option<u64>,
    pub memory_available_bytes: Option<u64>,
    pub memory_total_bytes: Option<u64>,
}

/// Full health report combining all subsystem checks.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub ts: String,
    pub backends: Vec<DependencyCheck>,
    pub ffmpeg: DependencyCheck,
    pub database: DependencyCheck,
    pub resources: ResourceSnapshot,
    pub overall_status: CheckStatus,
}

/// Check if ffmpeg is available via env override, PATH, or provisioned bundle.
#[must_use]
pub fn check_ffmpeg() -> DependencyCheck {
    let path = crate::audio::locate_ffmpeg_program(None);
    let available = path.is_some();
    let issues = if available {
        vec![]
    } else {
        vec!["ffmpeg not found on PATH or provisioned bundle".to_owned()]
    };
    DependencyCheck {
        name: "ffmpeg".to_owned(),
        available,
        path: path.map(|p| p.display().to_string()),
        version: None,
        issues,
    }
}

/// Check database accessibility by verifying the path's parent directory
/// exists and is writable.
#[must_use]
pub fn check_database(db_path: &Path) -> DependencyCheck {
    let mut issues = Vec::new();
    let raw_parent = db_path.parent().unwrap_or_else(|| Path::new(""));
    let parent_path = if raw_parent == Path::new("") {
        Path::new(".")
    } else {
        raw_parent
    };
    let parent_exists = parent_path.exists();
    if !parent_exists {
        issues.push(format!(
            "database parent directory does not exist: {}",
            parent_path.display()
        ));
    }
    if parent_exists {
        match std::fs::metadata(parent_path) {
            Ok(metadata) if !metadata.is_dir() => {
                issues.push(format!(
                    "database parent path is not a directory: {}",
                    parent_path.display()
                ));
            }
            Ok(metadata) if metadata.permissions().readonly() => {
                issues.push(format!(
                    "database parent directory is not writable: {}",
                    parent_path.display()
                ));
            }
            Ok(_) => {}
            Err(error) => {
                issues.push(format!(
                    "failed to inspect database parent directory {}: {error}",
                    parent_path.display()
                ));
            }
        }
    }

    if db_path.exists() {
        match std::fs::metadata(db_path) {
            Ok(metadata) if metadata.is_dir() => {
                issues.push(format!(
                    "database path is a directory, not a file: {}",
                    db_path.display()
                ));
            }
            Ok(metadata) if metadata.permissions().readonly() => {
                issues.push(format!(
                    "database file is not writable: {}",
                    db_path.display()
                ));
            }
            Ok(_) => {}
            Err(error) => {
                issues.push(format!(
                    "failed to inspect database path {}: {error}",
                    db_path.display()
                ));
            }
        }
    }

    let available = issues.is_empty();
    DependencyCheck {
        name: "database".to_owned(),
        available,
        path: Some(db_path.display().to_string()),
        version: None,
        issues,
    }
}

/// Gather system resource information (disk and memory).
#[must_use]
pub fn snapshot_resources() -> ResourceSnapshot {
    // We intentionally provide a best-effort snapshot without unsafe code.
    // On Linux we parse /proc/meminfo; on other platforms we return None.
    let (memory_available_bytes, memory_total_bytes) = read_meminfo();

    ResourceSnapshot {
        disk_free_bytes: None,
        disk_total_bytes: None,
        memory_available_bytes,
        memory_total_bytes,
    }
}

/// Parse `/proc/meminfo` for MemTotal and MemAvailable (Linux only).
fn read_meminfo() -> (Option<u64>, Option<u64>) {
    let Ok(content) = std::fs::read_to_string("/proc/meminfo") else {
        return (None, None);
    };
    let mut total: Option<u64> = None;
    let mut available: Option<u64> = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_meminfo_kb(rest);
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }
    (available, total)
}

/// Parse a meminfo value like "  16384000 kB" into bytes.
fn parse_meminfo_kb(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    let numeric_part = trimmed.strip_suffix("kB").unwrap_or(trimmed).trim();
    numeric_part.parse::<u64>().ok().map(|kb| kb * 1024)
}

/// Build a comprehensive health report by probing all subsystems.
#[must_use]
pub fn build_health_report(db_path: &Path) -> HealthReport {
    let ts = chrono::Utc::now().to_rfc3339();

    // Check backends using the existing engine infrastructure.
    let engines = crate::backend::all_engines();
    let backends: Vec<DependencyCheck> = engines
        .iter()
        .map(|engine| {
            let available = engine.is_available();
            DependencyCheck {
                name: engine.name().to_owned(),
                available,
                path: None,
                version: None,
                issues: if available {
                    vec![]
                } else {
                    vec![format!("{} backend not available", engine.name())]
                },
            }
        })
        .collect();

    let ffmpeg = check_ffmpeg();
    let database = check_database(db_path);
    let resources = snapshot_resources();

    // Determine overall status.
    let any_backend_available = backends.iter().any(|b| b.available);
    let overall_status = if any_backend_available && ffmpeg.available && database.available {
        CheckStatus::Ok
    } else if any_backend_available || ffmpeg.available {
        CheckStatus::Degraded
    } else {
        CheckStatus::Unavailable
    };

    HealthReport {
        ts,
        backends,
        ffmpeg,
        database,
        resources,
        overall_status,
    }
}

/// Construct the `health.report` NDJSON event value.
#[must_use]
pub fn health_report_value(report: &HealthReport) -> serde_json::Value {
    let backends_json: Vec<serde_json::Value> =
        report.backends.iter().map(dependency_check_json).collect();

    json!({
        "event": "health.report",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "ts": report.ts,
        "backends": backends_json,
        "ffmpeg": dependency_check_json(&report.ffmpeg),
        "database": dependency_check_json(&report.database),
        "resources": {
            "disk_free_bytes": report.resources.disk_free_bytes,
            "disk_total_bytes": report.resources.disk_total_bytes,
            "memory_available_bytes": report.resources.memory_available_bytes,
            "memory_total_bytes": report.resources.memory_total_bytes,
        },
        "overall_status": report.overall_status.as_str(),
    })
}

/// Emit a single `health.report` NDJSON line to stdout.
pub fn emit_health_report(report: &HealthReport) -> FwResult<()> {
    emit_line(&health_report_value(report))
}

/// Extract acceleration stream ownership/cancellation telemetry from run evidence.
#[must_use]
pub fn acceleration_context_from_evidence(evidence: &[Value]) -> Option<Value> {
    acceleration_context_ref_from_evidence(evidence).cloned()
}

fn acceleration_context_ref_from_evidence(evidence: &[Value]) -> Option<&Value> {
    evidence.iter().rev().find(|entry| {
        let has_stream_owner = entry
            .get("logical_stream_owner_id")
            .and_then(Value::as_str)
            .is_some();
        let has_fence = entry
            .get("cancellation_fence")
            .is_some_and(Value::is_object);
        has_stream_owner && has_fence
    })
}

fn dependency_check_json(check: &DependencyCheck) -> serde_json::Value {
    json!({
        "name": check.name,
        "available": check.available,
        "path": check.path,
        "version": check.version,
        "issues": check.issues,
    })
}

#[must_use]
pub fn robot_schema_value() -> serde_json::Value {
    json!({
        "version": ROBOT_SCHEMA_VERSION,
        "schema_version": ROBOT_SCHEMA_VERSION,
        "line_oriented": "ndjson",
        "events": {
            "run_start": {
                "required": RUN_START_REQUIRED_FIELDS,
                "example": run_start_value(json!({"backend": "auto"})),
            },
            "stage": {
                "required": STAGE_REQUIRED_FIELDS,
                "ordering_contract": {
                    "seq": "strictly increasing per run",
                    "ts": "non-decreasing RFC3339 timestamp per run",
                    "failure_path_example": [
                        "orchestration.budgets",
                        "ingest.start",
                        "ingest.error"
                    ],
                    "cancellation_path_example": [
                        "orchestration.budgets",
                        "orchestration.cancelled"
                    ],
                    "replay_rule": "persisted stage event order must match streamed stage event order",
                },
                "example": json!({
                    "event": "stage",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "run_id": "run-123",
                    "seq": 1,
                    "ts": "2026-02-22T00:00:00Z",
                    "stage": "ingest",
                    "code": "ingest.start",
                    "message": "materializing input",
                    "payload": {"input": "..."},
                }),
            },
            "run_complete": {
                "required": RUN_COMPLETE_REQUIRED_FIELDS,
                "example": json!({
                    "event": "run_complete",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "run_id": "run-123",
                    "trace_id": "00000000000000000000000000000000",
                    "started_at": "2026-02-22T00:00:00Z",
                    "finished_at": "2026-02-22T00:00:02Z",
                    "backend": "whisper_cpp",
                    "language": "en",
                    "transcript": "hello world",
                    "segments": [],
                    "acceleration": {"backend": "none", "normalized_confidences": true},
                    "acceleration_context": {
                        "logical_stream_owner_id": "trace:acceleration:none:cpu",
                        "logical_stream_kind": "cpu_lane",
                        "acceleration_backend": "none",
                        "mode": "cpu_fallback",
                        "cancellation_fence": {"status": "open"}
                    },
                    "warnings": [],
                    "evidence": [],
                }),
            },
            "run_error": {
                "required": RUN_ERROR_REQUIRED_FIELDS,
                "example": run_error_value("backend failed", "FW-ROBOT-EXEC"),
            },
            "backends.discovery": {
                "required": BACKENDS_DISCOVERY_REQUIRED_FIELDS,
                "example": json!({
                    "event": "backends.discovery",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "backends": [
                        {
                            "name": "whisper.cpp",
                            "kind": "whisper_cpp",
                            "available": true,
                            "capabilities": {
                                "supports_diarization": false,
                                "supports_translation": true,
                                "supports_word_timestamps": true,
                                "supports_gpu": true,
                                "supports_streaming": false,
                            }
                        }
                    ]
                }),
            },
            "routing_decision": {
                "required": ROUTING_DECISION_REQUIRED_FIELDS,
                "example": json!({
                    "event": "routing_decision",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "run_id": "run-123",
                    "ts": "2026-02-22T00:00:00Z",
                    "code": "backend.routing.decision_contract",
                    "decision_id": "dec-123",
                    "chosen_action": "try_whisper_cpp",
                    "calibration_score": 0.82,
                    "e_process": 1.4,
                    "fallback_active": false,
                    "recommended_order": ["whisper_cpp", "insanely_fast", "whisper_diarization"],
                    "mode": "adaptive",
                }),
            },
            "transcript.partial": {
                "required": TRANSCRIPT_PARTIAL_REQUIRED_FIELDS,
                "example": json!({
                    "event": "transcript.partial",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "run_id": "run-123",
                    "seq": 0,
                    "ts": "2026-02-22T00:00:01Z",
                    "text": "hello world",
                    "start_sec": 0.0,
                    "end_sec": 1.5,
                    "confidence": 0.95,
                    "speaker": "SPEAKER_00",
                }),
            },
            "transcript.confirm": {
                "required": TRANSCRIPT_CONFIRM_REQUIRED_FIELDS,
                "example": json!({
                    "event": "transcript.confirm",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "run_id": "run-123",
                    "seq": 0,
                    "window_id": 42,
                    "quality_model_id": "whisper-large",
                    "drift": {
                        "wer_approx": 0.0,
                        "confidence_delta": 0.05,
                        "segment_count_delta": 0,
                        "text_edit_distance": 0,
                    },
                    "latency_ms": 210,
                    "ts": "2026-02-22T00:00:01Z",
                }),
            },
            "transcript.retract": {
                "required": TRANSCRIPT_RETRACT_REQUIRED_FIELDS,
                "example": json!({
                    "event": "transcript.retract",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "run_id": "run-123",
                    "retracted_seq": 0,
                    "window_id": 42,
                    "reason": "quality_correction",
                    "quality_model_id": "whisper-large",
                    "ts": "2026-02-22T00:00:01Z",
                }),
            },
            "transcript.correct": {
                "required": TRANSCRIPT_CORRECT_REQUIRED_FIELDS,
                "example": json!({
                    "event": "transcript.correct",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "run_id": "run-123",
                    "correction_id": 7,
                    "replaces_seq": 0,
                    "window_id": 42,
                    "segments": [{
                        "start_sec": 0.0,
                        "end_sec": 1.5,
                        "text": "hello world",
                        "speaker": "SPEAKER_00",
                        "confidence": 0.97,
                    }],
                    "quality_model_id": "whisper-large",
                    "drift": {
                        "wer_approx": 0.25,
                        "confidence_delta": 0.12,
                        "segment_count_delta": 0,
                        "text_edit_distance": 2,
                    },
                    "latency_ms": 210,
                    "ts": "2026-02-22T00:00:01Z",
                }),
            },
            "transcript.speculation_stats": {
                "required": SPECULATION_STATS_REQUIRED_FIELDS,
                "example": json!({
                    "event": "transcript.speculation_stats",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "run_id": "run-123",
                    "windows_processed": 42,
                    "corrections_emitted": 10,
                    "confirmations_emitted": 32,
                    "correction_rate": 0.2381,
                    "mean_fast_latency_ms": 48.5,
                    "mean_quality_latency_ms": 207.4,
                    "current_window_size_ms": 3000,
                    "mean_drift_wer": 0.11,
                    "ts": "2026-02-22T00:00:02Z",
                }),
            },
            "health.report": {
                "required": HEALTH_REPORT_REQUIRED_FIELDS,
                "example": json!({
                    "event": "health.report",
                    "schema_version": ROBOT_SCHEMA_VERSION,
                    "ts": "2026-02-22T00:00:00Z",
                    "backends": [
                        {
                            "name": "whisper.cpp",
                            "available": true,
                            "path": "/usr/local/bin/whisper-cli",
                            "version": null,
                            "issues": [],
                        }
                    ],
                    "ffmpeg": {
                        "name": "ffmpeg",
                        "available": true,
                        "path": "/usr/bin/ffmpeg",
                        "version": null,
                        "issues": [],
                    },
                    "database": {
                        "name": "database",
                        "available": true,
                        "path": "db.sqlite3",
                        "version": null,
                        "issues": [],
                    },
                    "resources": {
                        "disk_free_bytes": null,
                        "disk_total_bytes": null,
                        "memory_available_bytes": 8_000_000_000_u64,
                        "memory_total_bytes": 16_000_000_000_u64,
                    },
                    "overall_status": "ok",
                }),
            },
        }
    })
}

fn emit_line<T: Serialize>(value: &T) -> FwResult<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

fn run_start_value(request_summary: serde_json::Value) -> serde_json::Value {
    json!({
        "event": "run_start",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "request": request_summary,
    })
}

fn run_error_value(message: &str, code: &str) -> serde_json::Value {
    json!({
        "event": "run_error",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "code": code,
        "message": message,
    })
}

#[derive(Serialize)]
struct BorrowedStage<'a> {
    event: &'static str,
    schema_version: &'static str,
    run_id: &'a str,
    seq: u64,
    ts: &'a str,
    stage: &'a str,
    code: &'a str,
    message: &'a str,
    payload: &'a Value,
}

impl<'a> BorrowedStage<'a> {
    fn new(run_id: &'a str, event: &'a RunEvent) -> Self {
        Self {
            event: "stage",
            schema_version: ROBOT_SCHEMA_VERSION,
            run_id,
            seq: event.seq,
            ts: &event.ts_rfc3339,
            stage: &event.stage,
            code: &event.code,
            message: &event.message,
            payload: &event.payload,
        }
    }
}

#[derive(Serialize)]
struct BorrowedComplete<'a> {
    event: &'static str,
    schema_version: &'static str,
    run_id: &'a str,
    trace_id: &'a str,
    started_at: &'a str,
    finished_at: &'a str,
    backend: BackendKind,
    language: Option<&'a str>,
    transcript: &'a str,
    segments: &'a [TranscriptionSegment],
    acceleration: Option<&'a AccelerationReport>,
    warnings: &'a [String],
    evidence: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    acceleration_context: Option<&'a Value>,
}

impl<'a> BorrowedComplete<'a> {
    fn new(report: &'a RunReport) -> Self {
        Self {
            event: "run_complete",
            schema_version: ROBOT_SCHEMA_VERSION,
            run_id: &report.run_id,
            trace_id: &report.trace_id,
            started_at: &report.started_at_rfc3339,
            finished_at: &report.finished_at_rfc3339,
            backend: report.result.backend,
            language: report.result.language.as_deref(),
            transcript: &report.result.transcript,
            segments: &report.result.segments,
            acceleration: report.result.acceleration.as_ref(),
            warnings: &report.warnings,
            evidence: &report.evidence,
            acceleration_context: acceleration_context_ref_from_evidence(&report.evidence),
        }
    }
}

#[cfg(test)]
fn run_stage_value(run_id: &str, event: &RunEvent) -> serde_json::Value {
    json!({
        "event": "stage",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "run_id": run_id,
        "seq": event.seq,
        "ts": event.ts_rfc3339,
        "stage": event.stage,
        "code": event.code,
        "message": event.message,
        "payload": event.payload,
    })
}

#[cfg(test)]
fn run_complete_value(report: &RunReport) -> serde_json::Value {
    let mut value = json!({
        "event": "run_complete",
        "schema_version": ROBOT_SCHEMA_VERSION,
        "run_id": report.run_id,
        "trace_id": report.trace_id,
        "started_at": report.started_at_rfc3339,
        "finished_at": report.finished_at_rfc3339,
        "backend": report.result.backend,
        "language": report.result.language,
        "transcript": report.result.transcript,
        "segments": report.result.segments,
        "acceleration": report.result.acceleration,
        "warnings": report.warnings,
        "evidence": report.evidence,
    });
    if let Some(acceleration_context) = acceleration_context_from_evidence(&report.evidence)
        && let Some(object) = value.as_object_mut()
    {
        object.insert("acceleration_context".to_owned(), acceleration_context);
    }
    value
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::model::RunEvent;

    use super::{
        BorrowedComplete, BorrowedStage, HEALTH_REPORT_REQUIRED_FIELDS,
        RUN_COMPLETE_REQUIRED_FIELDS, RUN_ERROR_REQUIRED_FIELDS, RUN_START_REQUIRED_FIELDS,
        SPECULATION_STATS_REQUIRED_FIELDS, STAGE_REQUIRED_FIELDS,
        TRANSCRIPT_CONFIRM_REQUIRED_FIELDS, TRANSCRIPT_CORRECT_REQUIRED_FIELDS,
        TRANSCRIPT_PARTIAL_REQUIRED_FIELDS, TRANSCRIPT_RETRACT_REQUIRED_FIELDS, robot_schema_value,
        run_complete_value, run_error_value, run_stage_value, run_start_value,
        transcript_partial_value,
    };

    #[test]
    fn error_envelope_has_terminal_shape() {
        let value = run_error_value("boom", "FW-ROBOT-EXEC");
        assert_eq!(value["event"], "run_error");
        assert_eq!(value["code"], "FW-ROBOT-EXEC");
        assert_eq!(value["message"], "boom");
    }

    #[test]
    fn stage_envelope_preserves_event_payload() {
        let event = RunEvent {
            seq: 7,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "normalize".to_owned(),
            code: "normalize.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({"duration_seconds": 12.34}),
        };

        let value = run_stage_value("run-xyz", &event);
        assert_eq!(value["event"], "stage");
        assert_eq!(value["run_id"], "run-xyz");
        assert_eq!(value["seq"], 7);
        assert_eq!(value["payload"]["duration_seconds"], 12.34);
    }

    #[test]
    fn stage_envelope_contains_required_fields() {
        let event = RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.start".to_owned(),
            message: "begin".to_owned(),
            payload: json!({"input": "file.wav"}),
        };
        let value = run_stage_value("run-123", &event);

        for field in STAGE_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "missing required stage field `{field}`"
            );
        }
    }

    #[test]
    fn run_start_envelope_contains_required_fields() {
        let value = run_start_value(json!({"backend": "auto", "language": "en"}));
        for field in RUN_START_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "run_start missing required field `{field}`"
            );
        }
        assert_eq!(value["event"], "run_start");
    }

    #[test]
    fn run_error_envelope_contains_required_fields() {
        let value = run_error_value("pipeline crashed", "FW-ROBOT-EXEC");
        for field in RUN_ERROR_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "run_error missing required field `{field}`"
            );
        }
        assert_eq!(value["event"], "run_error");
    }

    #[test]
    fn run_complete_envelope_contains_required_fields() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "golden-run".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:05Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: Some("en".to_owned()),
                translate: false,
                diarize: false,
                persist: true,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "golden test".to_owned(),
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

        let value = run_complete_value(&report);
        for field in RUN_COMPLETE_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "run_complete missing required field `{field}`"
            );
        }
        assert_eq!(value["event"], "run_complete");
        assert_eq!(value["run_id"], "golden-run");
        assert_eq!(value["trace_id"], "00000000000000000000000000000000");
        assert_eq!(value["transcript"], "golden test");
    }

    #[test]
    fn run_complete_segments_serializes_speaker_and_confidence() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
            TranscriptionSegment,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "seg-run".to_owned(),
            trace_id: "11111111111111111111111111111111".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:02Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: true,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperDiarization,
                transcript: "hello world".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(1.5),
                    text: "hello world".to_owned(),
                    speaker: Some("SPEAKER_00".to_owned()),
                    confidence: Some(0.95),
                }],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec!["test warning".to_owned()],
            evidence: vec![json!({"test": true})],
            replay: crate::model::ReplayEnvelope::default(),
        };

        let value = run_complete_value(&report);
        let segments = value["segments"]
            .as_array()
            .expect("segments should be array");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0]["speaker"], "SPEAKER_00");
        assert_eq!(segments[0]["confidence"], 0.95);
        assert_eq!(segments[0]["text"], "hello world");

        let warnings = value["warnings"].as_array().expect("warnings array");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0], "test warning");

        let evidence = value["evidence"].as_array().expect("evidence array");
        assert_eq!(evidence.len(), 1);
    }

    #[test]
    fn schema_declares_required_fields_for_each_event_type() {
        let schema = robot_schema_value();
        let events = schema["events"]
            .as_object()
            .expect("events should be object");

        assert_eq!(
            events["run_start"]["required"]
                .as_array()
                .expect("array")
                .len(),
            RUN_START_REQUIRED_FIELDS.len()
        );
        assert_eq!(
            events["stage"]["required"].as_array().expect("array").len(),
            STAGE_REQUIRED_FIELDS.len()
        );
        assert_eq!(
            events["run_complete"]["required"]
                .as_array()
                .expect("array")
                .len(),
            RUN_COMPLETE_REQUIRED_FIELDS.len()
        );
        assert_eq!(
            events["run_error"]["required"]
                .as_array()
                .expect("array")
                .len(),
            RUN_ERROR_REQUIRED_FIELDS.len()
        );
    }

    #[test]
    fn schema_examples_satisfy_their_own_required_fields() {
        let schema = robot_schema_value();
        let events = schema["events"]
            .as_object()
            .expect("events should be object");

        for (event_type, spec) in events {
            assert!(
                spec["required"].is_array(),
                "{event_type} should have required array"
            );
            let required = spec["required"].as_array().expect("required array");
            let example = &spec["example"];

            for field in required {
                let field_name = field.as_str().expect("required field should be string");
                assert!(
                    example.get(field_name).is_some(),
                    "schema example for `{event_type}` is missing required field `{field_name}`"
                );
            }
        }
    }

    #[test]
    fn stage_envelope_field_types_are_correct() {
        let event = RunEvent {
            seq: 42,
            ts_rfc3339: "2026-02-22T12:34:56Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.start".to_owned(),
            message: "running whisper_cpp".to_owned(),
            payload: json!({"key": "value"}),
        };
        let value = run_stage_value("run-types", &event);

        assert!(value["event"].is_string(), "event should be string");
        assert!(value["run_id"].is_string(), "run_id should be string");
        assert!(value["seq"].is_u64(), "seq should be u64");
        assert!(value["ts"].is_string(), "ts should be string");
        assert!(value["stage"].is_string(), "stage should be string");
        assert!(value["code"].is_string(), "code should be string");
        assert!(value["message"].is_string(), "message should be string");
        assert!(value["payload"].is_object(), "payload should be object");
    }

    #[test]
    fn run_complete_field_types_are_correct() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "types-run".to_owned(),
            trace_id: "aaaabbbbccccddddeeeeffffaaaabbbb".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:05Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "typed test".to_owned(),
                language: None,
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

        let value = run_complete_value(&report);

        assert!(value["event"].is_string(), "event should be string");
        assert!(value["run_id"].is_string(), "run_id should be string");
        assert!(value["trace_id"].is_string(), "trace_id should be string");
        assert!(
            value["started_at"].is_string(),
            "started_at should be string"
        );
        assert!(
            value["finished_at"].is_string(),
            "finished_at should be string"
        );
        assert!(value["backend"].is_string(), "backend should be string");
        assert!(
            value["transcript"].is_string(),
            "transcript should be string"
        );
        assert!(value["segments"].is_array(), "segments should be array");
        assert!(value["warnings"].is_array(), "warnings should be array");
        assert!(value["evidence"].is_array(), "evidence should be array");
        // language and acceleration may be null (Option types)
        assert!(
            value["language"].is_string() || value["language"].is_null(),
            "language should be string or null"
        );
        assert!(
            value["acceleration"].is_object() || value["acceleration"].is_null(),
            "acceleration should be object or null"
        );
    }

    #[test]
    fn run_complete_evidence_with_multiple_entries() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "evidence-run".to_owned(),
            trace_id: "trace-abc-123".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "evidence test".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![
                json!({"contract": "backend_selection", "action": "try_whisper_cpp"}),
                json!({"contract": "calibration", "score": 0.85}),
                json!({"contract": "checkpoint", "stage": "normalize", "result": "ok"}),
            ],
            replay: crate::model::ReplayEnvelope::default(),
        };

        let value = run_complete_value(&report);
        let evidence = value["evidence"].as_array().expect("evidence array");
        assert_eq!(evidence.len(), 3);
        assert_eq!(evidence[0]["contract"], "backend_selection");
        assert_eq!(evidence[1]["score"], 0.85);
        assert_eq!(evidence[2]["stage"], "normalize");

        // Verify trace_id propagation
        assert_eq!(value["trace_id"], "trace-abc-123");
    }

    // --- Error envelope edge cases ---

    #[test]
    fn error_value_covers_all_robot_error_codes() {
        for code in [
            "FW-ROBOT-EXEC",
            "FW-ROBOT-TIMEOUT",
            "FW-ROBOT-BACKEND",
            "FW-ROBOT-REQUEST",
            "FW-ROBOT-STORAGE",
            "FW-ROBOT-CANCELLED",
        ] {
            let value = run_error_value("test message", code);
            assert_eq!(value["event"], "run_error");
            assert_eq!(value["code"], code);
            assert_eq!(value["message"], "test message");
        }
    }

    #[test]
    fn error_value_with_very_long_message() {
        let long_msg = "x".repeat(10_000);
        let value = run_error_value(&long_msg, "FW-ROBOT-EXEC");
        let msg = value["message"].as_str().expect("message should be string");
        assert_eq!(msg.len(), 10_000);
    }

    #[test]
    fn error_value_with_special_characters() {
        let msg = "failed: \"path/to/file\" with\nnewline & <xml> chars";
        let value = run_error_value(msg, "FW-ROBOT-EXEC");
        assert_eq!(value["message"], msg);
    }

    // --- Stage envelope edge cases ---

    #[test]
    fn stage_value_with_empty_object_payload() {
        let event = RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.start".to_owned(),
            message: "starting".to_owned(),
            payload: json!({}),
        };
        let value = run_stage_value("run-empty-payload", &event);
        assert!(value["payload"].as_object().expect("object").is_empty());
    }

    #[test]
    fn stage_value_with_null_payload() {
        let event = RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "test".to_owned(),
            code: "test.code".to_owned(),
            message: "msg".to_owned(),
            payload: json!(null),
        };
        let value = run_stage_value("run-null-payload", &event);
        assert!(value["payload"].is_null());
    }

    #[test]
    fn stage_value_seq_zero_is_valid() {
        let event = RunEvent {
            seq: 0,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.start".to_owned(),
            message: "first".to_owned(),
            payload: json!({}),
        };
        let value = run_stage_value("run-seq0", &event);
        assert_eq!(value["seq"], 0);
    }

    // --- Schema edge cases ---

    #[test]
    fn schema_version_is_string() {
        let schema = robot_schema_value();
        assert!(schema["version"].is_string(), "version should be a string");
        assert!(
            schema["schema_version"].is_string(),
            "schema_version should be a string"
        );
    }

    #[test]
    fn schema_document_version_matches_robot_schema_version() {
        let schema = robot_schema_value();
        assert_eq!(schema["version"], super::ROBOT_SCHEMA_VERSION);
        assert_eq!(schema["schema_version"], super::ROBOT_SCHEMA_VERSION);
    }

    #[test]
    fn schema_line_oriented_is_ndjson() {
        let schema = robot_schema_value();
        assert_eq!(schema["line_oriented"], "ndjson");
    }

    // --- Start value edge cases ---

    #[test]
    fn run_start_value_with_empty_request_object() {
        let value = run_start_value(json!({}));
        assert_eq!(value["event"], "run_start");
        assert!(value["request"].is_object());
    }

    #[test]
    fn run_start_value_preserves_nested_request_fields() {
        let request = json!({
            "backend": "whisper_cpp",
            "language": "en",
            "nested": {"key": "val"}
        });
        let value = run_start_value(request);
        assert_eq!(value["request"]["nested"]["key"], "val");
    }

    #[test]
    fn run_complete_with_null_optionals_still_has_required_fields() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "null-opt".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::InsanelyFast,
                transcript: String::new(),
                language: None,
                segments: vec![],
                acceleration: None,
                raw_output: json!(null),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope::default(),
        };

        let value = run_complete_value(&report);
        for field in RUN_COMPLETE_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "run_complete with null optionals missing field `{field}`"
            );
        }
    }

    // --- NDJSON formatting: each envelope must serialize to a single JSON line ---

    /// Helper: build a minimal RunReport for testing.
    fn test_report(
        events: Vec<RunEvent>,
        evidence: Vec<serde_json::Value>,
    ) -> crate::model::RunReport {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        RunReport {
            run_id: "ndjson-test".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
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
            events,
            warnings: vec![],
            evidence,
            replay: crate::model::ReplayEnvelope::default(),
        }
    }

    fn test_event(seq: u64, stage: &str, code: &str) -> RunEvent {
        RunEvent {
            seq,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: stage.to_owned(),
            code: code.to_owned(),
            message: format!("{code} message"),
            payload: json!({"seq": seq}),
        }
    }

    #[test]
    fn ndjson_run_start_serializes_to_single_line() {
        let value = run_start_value(json!({"backend": "auto", "note": "line\nbreak"}));
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(
            !line.contains('\n'),
            "NDJSON line must not contain newlines"
        );
        // Must parse back to valid JSON.
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("parse back");
        assert_eq!(parsed["event"], "run_start");
    }

    #[test]
    fn ndjson_run_error_serializes_to_single_line() {
        let value = run_error_value("multi\nline\nerror", "FW-ROBOT-EXEC");
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(
            !line.contains('\n'),
            "NDJSON line must not contain newlines"
        );
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("parse back");
        assert_eq!(parsed["event"], "run_error");
        // The message should preserve the newlines inside the JSON string.
        assert!(parsed["message"].as_str().unwrap().contains('\n'));
    }

    #[test]
    fn ndjson_stage_serializes_to_single_line() {
        let event = test_event(1, "backend", "backend.ok");
        let value = run_stage_value("run-ndjson", &event);
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(
            !line.contains('\n'),
            "NDJSON line must not contain newlines"
        );
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("parse back");
        assert_eq!(parsed["event"], "stage");
    }

    #[test]
    fn borrowed_stage_serialization_matches_owned_value_bytes() {
        let cases = [
            (
                "run",
                RunEvent {
                    seq: 7,
                    ts_rfc3339: "ts".to_owned(),
                    stage: "stage".to_owned(),
                    code: "code".to_owned(),
                    message: "message".to_owned(),
                    payload: json!(null),
                },
            ),
            (
                "run-\"quoted\"\\line\nnext",
                RunEvent {
                    seq: u64::MAX,
                    ts_rfc3339: "2026-07-13T21:30:00-04:00".to_owned(),
                    stage: "backend/λ".to_owned(),
                    code: "backend.\0ok".to_owned(),
                    message: "line one\nline two 🚀".to_owned(),
                    payload: json!({
                        "ordered_first": [true, false, null],
                        "ordered_second": {"quote": "\\\"", "emoji": "🎧"},
                    }),
                },
            ),
            (
                "",
                RunEvent {
                    seq: 0,
                    ts_rfc3339: String::new(),
                    stage: String::new(),
                    code: String::new(),
                    message: String::new(),
                    payload: json!([0, "scalar", {"deep": [{"value": 1.25}]}]),
                },
            ),
        ];

        for (run_id, event) in &cases {
            let owned = serde_json::to_vec(&run_stage_value(run_id, event)).expect("owned stage");
            let borrowed =
                serde_json::to_vec(&BorrowedStage::new(run_id, event)).expect("borrowed stage");
            assert_eq!(borrowed, owned, "stage bytes differ for run_id {run_id:?}");
        }

        assert_eq!(
            serde_json::to_string(&BorrowedStage::new(cases[0].0, &cases[0].1))
                .expect("golden stage"),
            r#"{"event":"stage","schema_version":"1.0.0","run_id":"run","seq":7,"ts":"ts","stage":"stage","code":"code","message":"message","payload":null}"#,
        );
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn borrowed_stage_serialization_perf() {
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 21;

        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "robot_stage_serialization binary_sha256={:x}",
            Sha256::digest(executable)
        );

        fn time_owned(run_id: &str, event: &RunEvent, iterations: usize) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                let value = run_stage_value(black_box(run_id), black_box(event));
                let encoded = serde_json::to_string(black_box(&value)).expect("owned stage");
                drop(black_box(encoded));
            }
            started.elapsed().as_nanos()
        }

        fn time_borrowed(run_id: &str, event: &RunEvent, iterations: usize) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                let value = BorrowedStage::new(black_box(run_id), black_box(event));
                let encoded = serde_json::to_string(black_box(&value)).expect("borrowed stage");
                drop(black_box(encoded));
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

        fn run_shape(name: &str, event: &RunEvent, iterations: usize) {
            let run_id = "run-perf-0123456789abcdef";
            let owned_bytes =
                serde_json::to_vec(&run_stage_value(run_id, event)).expect("owned stage");
            let borrowed_bytes =
                serde_json::to_vec(&BorrowedStage::new(run_id, event)).expect("borrowed stage");
            assert_eq!(borrowed_bytes, owned_bytes, "{name} byte parity");

            for _ in 0..3 {
                black_box(time_owned(run_id, event, iterations));
                black_box(time_borrowed(run_id, event, iterations));
            }

            let mut null_ratios = Vec::with_capacity(SAMPLES);
            let mut speedups = Vec::with_capacity(SAMPLES);
            let mut owned_times = Vec::with_capacity(SAMPLES);
            let mut borrowed_times = Vec::with_capacity(SAMPLES);

            for sample in 0..SAMPLES {
                let null_first = time_owned(run_id, event, iterations);
                let null_second = time_owned(run_id, event, iterations);
                let (null_numerator, null_denominator) = if sample % 2 == 0 {
                    (null_first, null_second)
                } else {
                    (null_second, null_first)
                };
                null_ratios.push(null_numerator as f64 / null_denominator as f64);

                let (owned, borrowed) = if sample % 2 == 0 {
                    (
                        time_owned(run_id, event, iterations),
                        time_borrowed(run_id, event, iterations),
                    )
                } else {
                    let borrowed = time_borrowed(run_id, event, iterations);
                    let owned = time_owned(run_id, event, iterations);
                    (owned, borrowed)
                };
                owned_times.push(owned);
                borrowed_times.push(borrowed);
                speedups.push(owned as f64 / borrowed as f64);
            }

            eprintln!(
                "robot_stage_serialization shape={name} samples={SAMPLES} iterations={iterations} null_median={:.6} null_p90={:.6} owned_arm_median_ns={} borrowed_arm_median_ns={} speedup_median={:.6} speedup_p10={:.6}",
                percentile(&null_ratios, 50),
                percentile(&null_ratios, 90),
                median_ns(&owned_times),
                median_ns(&borrowed_times),
                percentile(&speedups, 50),
                percentile(&speedups, 10),
            );
            eprintln!(
                "robot_stage_serialization shape={name} null_ratios={null_ratios:?} speedups={speedups:?}"
            );
        }

        let current_like = RunEvent {
            seq: 3,
            ts_rfc3339: "2026-07-13T21:30:00-04:00".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "backend completed".to_owned(),
            payload: json!({"resolved_backend": "native", "elapsed_ms": 711}),
        };
        let nested = RunEvent {
            seq: 9,
            ts_rfc3339: "2026-07-13T21:30:00-04:00".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.routing.decision_contract".to_owned(),
            message: "routing evidence ready".to_owned(),
            payload: json!({
                "posterior": (0..32)
                    .map(|index| json!({
                        "backend": format!("candidate-{index}"),
                        "probability": f64::from(index) / 32.0,
                        "available": index % 3 != 0,
                    }))
                    .collect::<Vec<_>>(),
                "fallback_active": false,
            }),
        };

        run_shape("current_like", &current_like, 20_000);
        run_shape("nested", &nested, 2_000);
    }

    #[test]
    fn ndjson_run_complete_serializes_to_single_line() {
        let report = test_report(vec![], vec![json!({"k": "v"})]);
        let value = run_complete_value(&report);
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(
            !line.contains('\n'),
            "NDJSON line must not contain newlines"
        );
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("parse back");
        assert_eq!(parsed["event"], "run_complete");
    }

    #[test]
    fn borrowed_complete_serialization_matches_owned_value_bytes() {
        use crate::model::{AccelerationBackend, AccelerationReport, TranscriptionSegment};

        fn assert_byte_parity(report: &crate::model::RunReport) -> Vec<u8> {
            let owned = serde_json::to_vec(&run_complete_value(report)).expect("owned complete");
            let borrowed =
                serde_json::to_vec(&BorrowedComplete::new(report)).expect("borrowed complete");
            assert_eq!(borrowed, owned, "run_complete bytes differ");
            borrowed
        }

        let mut empty = test_report(vec![], vec![]);
        empty.result.language = None;
        empty.result.transcript.clear();
        let empty_bytes = assert_byte_parity(&empty);
        assert_eq!(
            String::from_utf8(empty_bytes).expect("empty complete is UTF-8"),
            r#"{"event":"run_complete","schema_version":"1.0.0","run_id":"ndjson-test","trace_id":"00000000000000000000000000000000","started_at":"2026-02-22T00:00:00Z","finished_at":"2026-02-22T00:00:01Z","backend":"whisper_cpp","language":null,"transcript":"","segments":[],"acceleration":null,"warnings":[],"evidence":[]}"#,
        );

        let mut rich = test_report(vec![], vec![]);
        rich.run_id = "run-\"quoted\"\\line\nnext".to_owned();
        rich.result.transcript = "hello\nλ 🎧 \\ \"world\"".repeat(8);
        rich.result.segments = vec![
            TranscriptionSegment {
                start_sec: Some(-0.0),
                end_sec: Some(1.25),
                text: "first\nsegment".to_owned(),
                speaker: Some("SPEAKER_00".to_owned()),
                confidence: Some(0.9375),
            },
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "第二段 🚀".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        rich.result.acceleration = Some(AccelerationReport {
            backend: AccelerationBackend::Frankentorch,
            input_values: 257,
            normalized_confidences: true,
            pre_mass: Some(3.5),
            post_mass: None,
            notes: vec!["accelerated\npath".to_owned()],
        });
        rich.warnings = vec!["warning \"quoted\"".to_owned()];
        rich.evidence = vec![
            json!({
                "logical_stream_owner_id": "ignored",
                "cancellation_fence": [],
            }),
            json!({
                "logical_stream_owner_id": "owner-first",
                "cancellation_fence": {"generation": 1},
            }),
            json!({"nested": [{"ordered_first": true}, null]}),
            json!({
                "logical_stream_owner_id": "owner-last",
                "cancellation_fence": {},
                "extra": "kept",
            }),
        ];
        let rich_line =
            String::from_utf8(assert_byte_parity(&rich)).expect("rich complete is UTF-8");
        let evidence_offset = rich_line.find("\"evidence\"").expect("evidence field");
        let context_offset = rich_line
            .rfind("\"acceleration_context\"")
            .expect("acceleration context field");
        assert!(
            context_offset > evidence_offset,
            "context must remain final"
        );
        let parsed: serde_json::Value = serde_json::from_str(&rich_line).expect("parse rich");
        assert_eq!(
            parsed["acceleration_context"]["logical_stream_owner_id"],
            "owner-last"
        );
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn borrowed_complete_serialization_perf() {
        use crate::model::{AccelerationBackend, AccelerationReport, TranscriptionSegment};
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 21;

        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "robot_complete_serialization binary_sha256={:x}",
            Sha256::digest(executable)
        );

        fn time_owned(report: &crate::model::RunReport, iterations: usize) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                let value = run_complete_value(black_box(report));
                let encoded = serde_json::to_string(black_box(&value)).expect("owned complete");
                drop(black_box(encoded));
            }
            started.elapsed().as_nanos()
        }

        fn time_borrowed(report: &crate::model::RunReport, iterations: usize) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                let value = BorrowedComplete::new(black_box(report));
                let encoded = serde_json::to_string(black_box(&value)).expect("borrowed complete");
                drop(black_box(encoded));
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

        fn run_shape(name: &str, report: &crate::model::RunReport, iterations: usize) {
            let owned_bytes =
                serde_json::to_vec(&run_complete_value(report)).expect("owned complete");
            let borrowed_bytes =
                serde_json::to_vec(&BorrowedComplete::new(report)).expect("borrowed complete");
            assert_eq!(borrowed_bytes, owned_bytes, "{name} byte parity");
            eprintln!(
                "robot_complete_serialization shape={name} output_bytes={} output_sha256={:x}",
                owned_bytes.len(),
                Sha256::digest(&owned_bytes)
            );

            for _ in 0..3 {
                black_box(time_owned(report, iterations));
                black_box(time_borrowed(report, iterations));
            }

            let mut null_ratios = Vec::with_capacity(SAMPLES);
            let mut speedups = Vec::with_capacity(SAMPLES);
            let mut owned_times = Vec::with_capacity(SAMPLES);
            let mut borrowed_times = Vec::with_capacity(SAMPLES);

            for sample in 0..SAMPLES {
                let null_first = time_owned(report, iterations);
                let null_second = time_owned(report, iterations);
                let (null_numerator, null_denominator) = if sample % 2 == 0 {
                    (null_first, null_second)
                } else {
                    (null_second, null_first)
                };
                null_ratios.push(null_numerator as f64 / null_denominator as f64);

                let (owned, borrowed) = if sample % 2 == 0 {
                    (
                        time_owned(report, iterations),
                        time_borrowed(report, iterations),
                    )
                } else {
                    let borrowed = time_borrowed(report, iterations);
                    let owned = time_owned(report, iterations);
                    (owned, borrowed)
                };
                owned_times.push(owned);
                borrowed_times.push(borrowed);
                speedups.push(owned as f64 / borrowed as f64);
            }

            eprintln!(
                "robot_complete_serialization shape={name} samples={SAMPLES} iterations={iterations} null_p10={:.6} null_median={:.6} null_p90={:.6} owned_arm_median_ns={} borrowed_arm_median_ns={} speedup_p10={:.6} speedup_median={:.6} speedup_p90={:.6}",
                percentile(&null_ratios, 10),
                percentile(&null_ratios, 50),
                percentile(&null_ratios, 90),
                median_ns(&owned_times),
                median_ns(&borrowed_times),
                percentile(&speedups, 10),
                percentile(&speedups, 50),
                percentile(&speedups, 90),
            );
            eprintln!(
                "robot_complete_serialization shape={name} null_ratios={null_ratios:?} speedups={speedups:?}"
            );
        }

        let mut current_like = test_report(vec![], vec![]);
        current_like.result.transcript =
            "The quick brown fox crosses the low-latency speech pipeline. ".repeat(18);
        current_like.result.segments = (0..10)
            .map(|index| TranscriptionSegment {
                start_sec: Some(f64::from(index) * 0.75),
                end_sec: Some(f64::from(index + 1) * 0.75),
                text: format!("segment {index}: measured robot output"),
                speaker: (index % 3 == 0).then(|| format!("SPEAKER_{index:02}")),
                confidence: Some(0.9 + f64::from(index) / 1_000.0),
            })
            .collect();
        current_like.warnings = vec!["normalization used ffmpeg".to_owned()];
        current_like.evidence = vec![
            json!({"contract": "backend_selection", "action": "native"}),
            json!({"contract": "calibration", "score": 0.93}),
            json!({"contract": "checkpoint", "stage": "persist", "result": "ok"}),
        ];

        let mut heavy = test_report(vec![], vec![]);
        heavy.result.transcript = "wide transcript payload with Unicode λ and 🎧; ".repeat(700);
        heavy.result.segments = (0..128)
            .map(|index| TranscriptionSegment {
                start_sec: Some(f64::from(index) * 0.5),
                end_sec: Some(f64::from(index + 1) * 0.5),
                text: format!(
                    "segment {index:03}: long nested transcription text with escaped \"quotes\""
                ),
                speaker: Some(format!("SPEAKER_{:02}", index % 8)),
                confidence: Some(0.8 + f64::from(index % 20) / 100.0),
            })
            .collect();
        heavy.result.acceleration = Some(AccelerationReport {
            backend: AccelerationBackend::Frankentorch,
            input_values: 65_536,
            normalized_confidences: true,
            pre_mass: Some(127.75),
            post_mass: Some(127.5),
            notes: (0..8).map(|index| format!("note-{index}")).collect(),
        });
        heavy.warnings = (0..16)
            .map(|index| format!("heavy warning {index}"))
            .collect();
        heavy.evidence = (0..32)
            .map(|index| {
                json!({
                    "contract": format!("contract-{index}"),
                    "posterior": {"accepted": index % 2 == 0, "score": f64::from(index) / 32.0},
                    "actions": ["keep", "fallback"],
                })
            })
            .collect();
        heavy.evidence.push(json!({
            "logical_stream_owner_id": "stream-owner-heavy",
            "cancellation_fence": {"epoch": 17, "closed": false},
            "calibration": {"ece": 0.0125},
        }));

        run_shape("current_like", &current_like, 5_000);
        run_shape("heavy", &heavy, 100);
    }

    // --- emit_robot_report shape tests (value-level, not stdout) ---

    #[test]
    fn emit_report_with_empty_events_emits_only_complete() {
        // With no events, emit_robot_report should produce exactly one value:
        // run_complete. We verify the value layer produces the right shape.
        let report = test_report(vec![], vec![]);
        let complete = run_complete_value(&report);
        assert_eq!(complete["event"], "run_complete");
        assert!(complete["segments"].is_array());
    }

    #[test]
    fn emit_report_with_multiple_events_preserves_order() {
        let events = vec![
            test_event(1, "ingest", "ingest.start"),
            test_event(2, "normalize", "normalize.ok"),
            test_event(3, "backend", "backend.start"),
            test_event(4, "backend", "backend.ok"),
            test_event(5, "persist", "persist.ok"),
        ];
        let report = test_report(events, vec![]);

        // Collect all stage values in order.
        let stage_values: Vec<serde_json::Value> = report
            .events
            .iter()
            .map(|e| run_stage_value(&report.run_id, e))
            .collect();

        assert_eq!(stage_values.len(), 5);
        for (i, sv) in stage_values.iter().enumerate() {
            assert_eq!(sv["seq"], (i + 1) as u64);
            assert_eq!(sv["event"], "stage");
        }
        // Stage order matches insertion order.
        assert_eq!(stage_values[0]["code"], "ingest.start");
        assert_eq!(stage_values[4]["code"], "persist.ok");
    }

    #[test]
    fn emit_report_run_id_consistent_across_stages_and_complete() {
        let events = vec![
            test_event(1, "ingest", "ingest.ok"),
            test_event(2, "backend", "backend.ok"),
        ];
        let report = test_report(events, vec![]);

        for event in &report.events {
            let sv = run_stage_value(&report.run_id, event);
            assert_eq!(
                sv["run_id"], "ndjson-test",
                "stage run_id should match report"
            );
        }
        let cv = run_complete_value(&report);
        assert_eq!(
            cv["run_id"], "ndjson-test",
            "complete run_id should match report"
        );
    }

    // --- Round-trip serialization tests ---

    #[test]
    fn run_start_value_round_trips_through_json() {
        let original = run_start_value(json!({"backend": "whisper_cpp", "lang": "en"}));
        let json_str = serde_json::to_string(&original).expect("serialize");
        let round_tripped: serde_json::Value =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn run_error_value_round_trips_through_json() {
        let original = run_error_value("error msg", "FW-ROBOT-TIMEOUT");
        let json_str = serde_json::to_string(&original).expect("serialize");
        let round_tripped: serde_json::Value =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn run_stage_value_round_trips_through_json() {
        let event = test_event(42, "normalize", "normalize.ok");
        let original = run_stage_value("run-rt", &event);
        let json_str = serde_json::to_string(&original).expect("serialize");
        let round_tripped: serde_json::Value =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn run_complete_value_round_trips_through_json() {
        let report = test_report(vec![], vec![json!({"evidence": true})]);
        let original = run_complete_value(&report);
        let json_str = serde_json::to_string(&original).expect("serialize");
        let round_tripped: serde_json::Value =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(original, round_tripped);
    }

    // --- Additional edge cases ---

    #[test]
    fn stage_value_with_large_seq_number() {
        let event = RunEvent {
            seq: u64::MAX,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "max seq".to_owned(),
            payload: json!({}),
        };
        let value = run_stage_value("run-maxseq", &event);
        assert_eq!(value["seq"], u64::MAX);
    }

    #[test]
    fn run_complete_with_multiple_warnings() {
        let mut report = test_report(vec![], vec![]);
        report.warnings = vec![
            "warning 1".to_owned(),
            "warning 2".to_owned(),
            "warning 3".to_owned(),
        ];
        let value = run_complete_value(&report);
        let warnings = value["warnings"].as_array().expect("array");
        assert_eq!(warnings.len(), 3);
        assert_eq!(warnings[0], "warning 1");
        assert_eq!(warnings[2], "warning 3");
    }

    #[test]
    fn run_complete_empty_transcript_is_empty_string() {
        let mut report = test_report(vec![], vec![]);
        report.result.transcript = String::new();
        let value = run_complete_value(&report);
        assert_eq!(value["transcript"], "");
    }

    #[test]
    fn run_start_value_with_null_request_is_valid() {
        let value = run_start_value(json!(null));
        assert_eq!(value["event"], "run_start");
        assert!(value["request"].is_null());
    }

    #[test]
    fn schema_has_expected_event_types() {
        let schema = robot_schema_value();
        let events = schema["events"]
            .as_object()
            .expect("events should be object");
        assert_eq!(
            events.len(),
            12,
            "expected 12 event types including routing history and speculation events, got {}",
            events.len()
        );
        for expected in [
            "run_start",
            "stage",
            "run_complete",
            "run_error",
            "backends.discovery",
            "routing_decision",
            "transcript.partial",
            "transcript.confirm",
            "transcript.retract",
            "transcript.correct",
            "transcript.speculation_stats",
            "health.report",
        ] {
            assert!(
                events.contains_key(expected),
                "missing schema event type `{expected}`"
            );
        }
    }

    #[test]
    fn stage_value_with_unicode_content() {
        let event = RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "\u{1F600} café résumé".to_owned(),
            payload: json!({"text": "\u{4E16}\u{754C}"}),
        };
        let value = run_stage_value("run-unicode", &event);
        assert!(value["message"].as_str().unwrap().contains('\u{1F600}'));
        assert_eq!(value["payload"]["text"], "\u{4E16}\u{754C}");
        // Verify it serializes to valid single-line JSON.
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(!line.contains('\n'));
    }

    #[test]
    fn stage_value_with_deeply_nested_payload() {
        let event = RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "deep".to_owned(),
            payload: json!({"l1": {"l2": {"l3": {"l4": [1, 2, 3]}}}}),
        };
        let value = run_stage_value("run-deep", &event);
        assert_eq!(value["payload"]["l1"]["l2"]["l3"]["l4"][2], 3);
    }

    #[test]
    fn run_complete_with_acceleration_report() {
        use crate::model::{
            AccelerationBackend, AccelerationReport, BackendKind, BackendParams, InputSource,
            RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "accel-run".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:05Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "accel test".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![],
                acceleration: Some(AccelerationReport {
                    backend: AccelerationBackend::Frankentorch,
                    input_values: 42,
                    normalized_confidences: true,
                    pre_mass: Some(0.95),
                    post_mass: Some(0.99),
                    notes: vec!["accelerated".to_owned()],
                }),
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope::default(),
        };

        let value = run_complete_value(&report);
        assert!(value["acceleration"].is_object());
        assert_eq!(value["acceleration"]["backend"], "frankentorch");
    }

    #[test]
    fn run_complete_includes_acceleration_context_when_present_in_evidence() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "accel-ctx-run".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:05Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "accel ctx".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![
                json!({"artifact": "other"}),
                json!({
                    "logical_stream_owner_id": "trace:acceleration:none:cpu",
                    "logical_stream_kind": "cpu_lane",
                    "acceleration_backend": "none",
                    "mode": "cpu_fallback",
                    "cancellation_fence": {"status": "open"},
                }),
            ],
            replay: crate::model::ReplayEnvelope::default(),
        };

        let value = run_complete_value(&report);
        assert_eq!(
            value["acceleration_context"]["logical_stream_owner_id"],
            "trace:acceleration:none:cpu"
        );
        assert_eq!(
            value["acceleration_context"]["cancellation_fence"]["status"],
            "open"
        );
    }

    #[test]
    fn run_complete_with_many_segments() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
            TranscriptionSegment,
        };
        use std::path::PathBuf;

        let segments: Vec<TranscriptionSegment> = (0..50)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some(i as f64 + 1.0),
                text: format!("segment {i}"),
                speaker: if i % 2 == 0 {
                    Some(format!("SPEAKER_{:02}", i % 3))
                } else {
                    None
                },
                confidence: if i % 3 == 0 { Some(0.95) } else { None },
            })
            .collect();

        let report = RunReport {
            run_id: "many-seg".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:50Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "many segments".to_owned(),
                language: Some("en".to_owned()),
                segments,
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope::default(),
        };

        let value = run_complete_value(&report);
        let segs = value["segments"].as_array().expect("array");
        assert_eq!(segs.len(), 50);
        assert_eq!(segs[0]["text"], "segment 0");
        assert_eq!(segs[49]["text"], "segment 49");
    }

    #[test]
    fn error_value_with_empty_strings() {
        let value = run_error_value("", "");
        assert_eq!(value["event"], "run_error");
        assert_eq!(value["code"], "");
        assert_eq!(value["message"], "");
    }

    #[test]
    fn required_fields_constants_are_non_empty_and_unique() {
        use std::collections::HashSet;

        let all_constants: &[&[&str]] = &[
            STAGE_REQUIRED_FIELDS,
            RUN_ERROR_REQUIRED_FIELDS,
            RUN_START_REQUIRED_FIELDS,
            RUN_COMPLETE_REQUIRED_FIELDS,
        ];
        for fields in all_constants {
            assert!(
                !fields.is_empty(),
                "required fields constant must not be empty"
            );
            let unique: HashSet<_> = fields.iter().collect();
            assert_eq!(
                unique.len(),
                fields.len(),
                "required fields must be unique: {fields:?}"
            );
            for f in *fields {
                assert!(!f.is_empty(), "field name must not be empty string");
            }
        }
    }

    #[test]
    fn run_complete_backend_serializes_all_variants() {
        use crate::model::BackendKind;

        let backends = [
            (BackendKind::WhisperCpp, "whisper_cpp"),
            (BackendKind::InsanelyFast, "insanely_fast"),
            (BackendKind::WhisperDiarization, "whisper_diarization"),
            (BackendKind::Auto, "auto"),
        ];
        for (backend, expected_str) in backends {
            let mut report = test_report(vec![], vec![]);
            report.result.backend = backend;
            let value = run_complete_value(&report);
            assert_eq!(
                value["backend"], expected_str,
                "backend {backend:?} should serialize to `{expected_str}`"
            );
        }
    }

    #[test]
    fn stage_value_preserves_exact_timestamp_string() {
        let precise_ts = "2026-02-22T12:34:56.789012345Z";
        let event = RunEvent {
            seq: 1,
            ts_rfc3339: precise_ts.to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({}),
        };
        let value = run_stage_value("run-ts", &event);
        assert_eq!(
            value["ts"].as_str().unwrap(),
            precise_ts,
            "timestamp must be preserved exactly"
        );
    }

    #[test]
    fn run_complete_with_empty_run_id_and_trace_id() {
        let mut report = test_report(vec![], vec![]);
        report.run_id = String::new();
        report.trace_id = String::new();
        let value = run_complete_value(&report);
        assert_eq!(value["run_id"], "");
        assert_eq!(value["trace_id"], "");
        // Still must have all required fields present
        for field in RUN_COMPLETE_REQUIRED_FIELDS {
            assert!(value.get(*field).is_some(), "missing field `{field}`");
        }
    }

    #[test]
    fn run_start_with_array_request() {
        let value = run_start_value(json!(["item1", "item2"]));
        assert_eq!(value["event"], "run_start");
        assert!(value["request"].is_array());
        assert_eq!(value["request"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn run_complete_does_not_include_replay_field() {
        // Replay data is stored-only (persisted to SQLite), not streamed
        // via robot mode. It must NOT appear in run_complete output.
        let report = test_report(vec![], vec![]);
        let value = run_complete_value(&report);
        assert!(
            value.get("replay").is_none(),
            "run_complete should NOT include replay (it's stored-only, not streamed)"
        );
    }

    #[test]
    fn error_value_with_newlines_and_control_chars() {
        let message = "line1\nline2\ttab\rcarriage\0null";
        let value = run_error_value(message, "FW-MULTILINE");
        // JSON serialization should handle control characters.
        let json_str = serde_json::to_string(&value).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("parse");
        assert_eq!(parsed["message"].as_str().unwrap(), message);
    }

    #[test]
    fn run_complete_value_has_exactly_required_fields_plus_event() {
        let report = test_report(vec![], vec![]);
        let value = run_complete_value(&report);
        let obj = value.as_object().expect("object");
        // Every key in the output should be in the required fields list
        for key in obj.keys() {
            assert!(
                RUN_COMPLETE_REQUIRED_FIELDS.contains(&key.as_str()),
                "unexpected field `{key}` in run_complete output"
            );
        }
        // Every required field should be in the output
        for field in RUN_COMPLETE_REQUIRED_FIELDS {
            assert!(
                obj.contains_key(*field),
                "missing required field `{field}` in run_complete output"
            );
        }
    }

    #[test]
    fn stage_value_has_exactly_required_fields() {
        let event = test_event(1, "ingest", "ingest.ok");
        let value = run_stage_value("run-exact", &event);
        let obj = value.as_object().expect("object");
        for key in obj.keys() {
            assert!(
                STAGE_REQUIRED_FIELDS.contains(&key.as_str()),
                "unexpected field `{key}` in stage output"
            );
        }
        for field in STAGE_REQUIRED_FIELDS {
            assert!(
                obj.contains_key(*field),
                "missing required field `{field}` in stage output"
            );
        }
    }

    #[test]
    fn run_complete_language_null_when_none() {
        let mut report = test_report(vec![], vec![]);
        report.result.language = None;
        let value = run_complete_value(&report);
        assert!(
            value["language"].is_null(),
            "None language should serialize to null"
        );
    }

    #[test]
    fn run_complete_does_not_leak_internal_paths_or_request() {
        let report = test_report(vec![], vec![]);
        let value = run_complete_value(&report);
        // These fields are in RunReport but intentionally excluded from streaming.
        assert!(
            value.get("input_path").is_none(),
            "input_path should not be streamed"
        );
        assert!(
            value.get("normalized_wav_path").is_none(),
            "normalized_wav_path should not be streamed"
        );
        assert!(
            value.get("request").is_none(),
            "request should not be streamed"
        );
        assert!(
            value.get("raw_output").is_none(),
            "raw_output should not be streamed"
        );
    }

    #[test]
    fn run_complete_with_acceleration_serializes_to_single_line() {
        let mut report = test_report(vec![], vec![]);
        report.result.acceleration = Some(crate::model::AccelerationReport {
            backend: crate::model::AccelerationBackend::Frankentorch,
            input_values: 100,
            normalized_confidences: true,
            pre_mass: Some(0.95),
            post_mass: Some(0.88),
            notes: vec!["batch mode".to_owned(), "GPU A100".to_owned()],
        });
        let value = run_complete_value(&report);
        let serialized = serde_json::to_string(&value).expect("serialize");
        assert!(
            !serialized.contains('\n'),
            "NDJSON envelope must be a single line"
        );
        assert_eq!(value["acceleration"]["backend"], "frankentorch");
        let notes = value["acceleration"]["notes"]
            .as_array()
            .expect("notes array");
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0], "batch mode");
    }

    #[test]
    fn schema_examples_have_correct_event_type_values() {
        let schema = robot_schema_value();
        let events = schema["events"].as_object().expect("events map");
        for (event_name, event_def) in events {
            let example = &event_def["example"];
            assert_eq!(
                example["event"].as_str().expect("event field"),
                event_name.as_str(),
                "schema example for '{event_name}' should have event == '{event_name}'"
            );
        }
    }

    #[test]
    fn run_start_value_with_scalar_request() {
        // run_start_value accepts any Value, including scalars.
        let value = run_start_value(json!("simple_string_request"));
        assert_eq!(value["event"], "run_start");
        assert_eq!(value["request"], "simple_string_request");

        let value2 = run_start_value(json!(42));
        assert_eq!(value2["request"], 42);
    }

    // ── Fifth-pass edge case tests ──

    #[test]
    fn run_complete_does_not_include_events_field() {
        // Events are emitted individually as stage envelopes; they should NOT
        // appear in the run_complete envelope to avoid duplication.
        let events = vec![
            test_event(1, "ingest", "ingest.ok"),
            test_event(2, "backend", "backend.ok"),
        ];
        let report = test_report(events, vec![]);
        assert_eq!(report.events.len(), 2, "report does have events");
        let value = run_complete_value(&report);
        assert!(
            value.get("events").is_none(),
            "run_complete should NOT include events field"
        );
    }

    #[test]
    fn run_complete_with_frankenjax_acceleration() {
        let mut report = test_report(vec![], vec![]);
        report.result.acceleration = Some(crate::model::AccelerationReport {
            backend: crate::model::AccelerationBackend::Frankenjax,
            input_values: 256,
            normalized_confidences: false,
            pre_mass: None,
            post_mass: None,
            notes: vec![],
        });
        let value = run_complete_value(&report);
        assert_eq!(value["acceleration"]["backend"], "frankenjax");
        assert_eq!(value["acceleration"]["input_values"], 256);
        assert_eq!(value["acceleration"]["normalized_confidences"], false);
        assert!(value["acceleration"]["pre_mass"].is_null());
        assert!(
            value["acceleration"]["notes"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn stage_value_with_all_empty_string_fields() {
        let event = RunEvent {
            seq: 0,
            ts_rfc3339: String::new(),
            stage: String::new(),
            code: String::new(),
            message: String::new(),
            payload: json!({}),
        };
        let value = run_stage_value("", &event);
        assert_eq!(value["run_id"], "");
        assert_eq!(value["ts"], "");
        assert_eq!(value["stage"], "");
        assert_eq!(value["code"], "");
        assert_eq!(value["message"], "");
        // Should still have all required fields present
        for field in STAGE_REQUIRED_FIELDS {
            assert!(value.get(*field).is_some(), "missing field `{field}`");
        }
    }

    #[test]
    fn run_complete_value_is_deterministic() {
        let report = test_report(
            vec![test_event(1, "ingest", "ingest.ok")],
            vec![json!({"k": "v"})],
        );
        let value1 = run_complete_value(&report);
        let value2 = run_complete_value(&report);
        assert_eq!(
            value1, value2,
            "run_complete_value should produce identical output for same input"
        );
        let json1 = serde_json::to_string(&value1).expect("serialize");
        let json2 = serde_json::to_string(&value2).expect("serialize");
        assert_eq!(json1, json2, "serialized JSON should be byte-identical");
    }

    #[test]
    fn run_start_value_has_exactly_three_fields() {
        let value = run_start_value(json!({"backend": "auto"}));
        let obj = value.as_object().expect("should be object");
        assert_eq!(
            obj.len(),
            3,
            "run_start should have exactly 3 fields (event, schema_version, request), got {}",
            obj.len()
        );
        assert!(obj.contains_key("event"));
        assert!(obj.contains_key("schema_version"));
        assert!(obj.contains_key("request"));
    }

    #[test]
    fn test_run_start_has_schema_version() {
        let value = run_start_value(json!({"backend": "auto"}));
        assert_eq!(
            value["schema_version"],
            super::ROBOT_SCHEMA_VERSION,
            "run_start event should contain schema_version"
        );
    }

    #[test]
    fn test_run_error_has_schema_version() {
        let value = run_error_value("something failed", "FW-ROBOT-EXEC");
        assert_eq!(
            value["schema_version"],
            super::ROBOT_SCHEMA_VERSION,
            "run_error event should contain schema_version"
        );
    }

    #[test]
    fn test_run_stage_has_schema_version() {
        let event = RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.start".to_owned(),
            message: "begin".to_owned(),
            payload: json!({"input": "file.wav"}),
        };
        let value = run_stage_value("run-sv", &event);
        assert_eq!(
            value["schema_version"],
            super::ROBOT_SCHEMA_VERSION,
            "stage event should contain schema_version"
        );
    }

    #[test]
    fn test_run_complete_has_schema_version() {
        let report = test_report(vec![], vec![]);
        let value = run_complete_value(&report);
        assert_eq!(
            value["schema_version"],
            super::ROBOT_SCHEMA_VERSION,
            "run_complete event should contain schema_version"
        );
    }

    #[test]
    fn test_schema_version_is_valid_semver() {
        let version = super::ROBOT_SCHEMA_VERSION;
        let parts: Vec<&str> = version.split('.').collect();
        assert_eq!(
            parts.len(),
            3,
            "schema version should have 3 parts (major.minor.patch), got: {version}"
        );
        for (i, part) in parts.iter().enumerate() {
            assert!(
                part.parse::<u32>().is_ok(),
                "schema version part {i} ({part}) should be a valid integer"
            );
        }
    }

    #[test]
    fn test_emit_robot_error_from_fw_has_code() {
        use crate::error::FwError;

        // Test with a few representative FwError variants to verify the
        // JSON envelope produced by emit_robot_error_from_fw contains the
        // expected error_code() and message.
        let cases: Vec<(FwError, &str)> = vec![
            (
                FwError::Io(std::io::Error::other("disk read failed")),
                "FW-IO",
            ),
            (
                FwError::BackendUnavailable("whisper not found".to_owned()),
                "FW-BACKEND-UNAVAILABLE",
            ),
            (FwError::Cancelled("user abort".to_owned()), "FW-CANCELLED"),
            (
                FwError::StageTimeout {
                    stage: "backend".to_owned(),
                    budget_ms: 5000,
                },
                "FW-STAGE-TIMEOUT",
            ),
            (FwError::Storage("database locked".to_owned()), "FW-STORAGE"),
        ];

        for (error, expected_code) in &cases {
            // We cannot capture stdout in a unit test easily, so we verify
            // the JSON value that would be emitted by reconstructing it the
            // same way emit_robot_error_from_fw does.
            let value = json!({
                "event": "run_error",
                "schema_version": super::ROBOT_SCHEMA_VERSION,
                "code": error.error_code(),
                "message": error.to_string(),
            });

            assert_eq!(value["event"], "run_error");
            assert_eq!(
                value["code"].as_str().unwrap(),
                *expected_code,
                "error_code mismatch for {:?}",
                error
            );
            assert!(
                !value["message"].as_str().unwrap().is_empty(),
                "message should not be empty for {:?}",
                error
            );
            assert_eq!(value["schema_version"], super::ROBOT_SCHEMA_VERSION);

            // Verify all required run_error fields are present.
            for field in super::RUN_ERROR_REQUIRED_FIELDS {
                assert!(
                    value.get(*field).is_some(),
                    "missing required field `{field}` in emit_robot_error_from_fw output"
                );
            }
        }
    }

    // ── backends.discovery tests ──

    fn make_test_backends_report() -> crate::model::BackendsReport {
        use crate::model::{BackendDiscoveryEntry, BackendKind, EngineCapabilities};

        crate::model::BackendsReport {
            backends: vec![
                BackendDiscoveryEntry {
                    name: "whisper.cpp".to_owned(),
                    kind: BackendKind::WhisperCpp,
                    available: true,
                    capabilities: EngineCapabilities {
                        supports_diarization: false,
                        supports_translation: true,
                        supports_word_timestamps: true,
                        supports_gpu: true,
                        supports_streaming: false,
                    },
                },
                BackendDiscoveryEntry {
                    name: "insanely-fast-whisper".to_owned(),
                    kind: BackendKind::InsanelyFast,
                    available: false,
                    capabilities: EngineCapabilities {
                        supports_diarization: true,
                        supports_translation: true,
                        supports_word_timestamps: true,
                        supports_gpu: true,
                        supports_streaming: false,
                    },
                },
                BackendDiscoveryEntry {
                    name: "whisper-diarization".to_owned(),
                    kind: BackendKind::WhisperDiarization,
                    available: false,
                    capabilities: EngineCapabilities {
                        supports_diarization: true,
                        supports_translation: false,
                        supports_word_timestamps: false,
                        supports_gpu: true,
                        supports_streaming: false,
                    },
                },
            ],
        }
    }

    #[test]
    fn backends_discovery_event_type_is_correct() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        assert_eq!(value["event"], "backends.discovery");
    }

    #[test]
    fn backends_discovery_has_schema_version() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        assert_eq!(value["schema_version"], super::ROBOT_SCHEMA_VERSION);
    }

    #[test]
    fn backends_discovery_contains_required_fields() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        for field in super::BACKENDS_DISCOVERY_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "backends.discovery missing required field `{field}`"
            );
        }
    }

    #[test]
    fn backends_discovery_has_exactly_required_fields() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        let obj = value.as_object().expect("should be object");
        for key in obj.keys() {
            assert!(
                super::BACKENDS_DISCOVERY_REQUIRED_FIELDS.contains(&key.as_str()),
                "unexpected field `{key}` in backends.discovery output"
            );
        }
        for field in super::BACKENDS_DISCOVERY_REQUIRED_FIELDS {
            assert!(
                obj.contains_key(*field),
                "missing required field `{field}` in backends.discovery output"
            );
        }
    }

    #[test]
    fn backends_discovery_backends_is_array() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        assert!(
            value["backends"].is_array(),
            "backends field should be an array"
        );
    }

    #[test]
    fn backends_discovery_includes_all_three_backends() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        let backends = value["backends"].as_array().expect("backends array");
        assert_eq!(backends.len(), 3);
    }

    #[test]
    fn backends_discovery_entry_has_expected_fields() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        let backends = value["backends"].as_array().expect("backends array");

        let expected_entry_fields = ["name", "kind", "available", "capabilities"];
        for entry in backends {
            for field in &expected_entry_fields {
                assert!(
                    entry.get(*field).is_some(),
                    "backend entry missing field `{field}`: {entry}"
                );
            }
        }
    }

    #[test]
    fn backends_discovery_entry_capabilities_has_expected_flags() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        let backends = value["backends"].as_array().expect("backends array");

        let expected_cap_fields = [
            "supports_diarization",
            "supports_translation",
            "supports_word_timestamps",
            "supports_gpu",
            "supports_streaming",
        ];
        for entry in backends {
            let caps = &entry["capabilities"];
            for field in &expected_cap_fields {
                assert!(
                    caps.get(*field).is_some(),
                    "capabilities missing field `{field}` in entry: {entry}"
                );
                assert!(
                    caps[*field].is_boolean(),
                    "capabilities.{field} should be boolean in entry: {entry}"
                );
            }
        }
    }

    #[test]
    fn backends_discovery_whisper_cpp_capabilities_are_correct() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        let backends = value["backends"].as_array().expect("backends array");
        let wcpp = backends
            .iter()
            .find(|b| b["kind"] == "whisper_cpp")
            .expect("whisper_cpp entry");
        assert_eq!(wcpp["name"], "whisper.cpp");
        assert_eq!(wcpp["available"], true);
        assert_eq!(wcpp["capabilities"]["supports_diarization"], false);
        assert_eq!(wcpp["capabilities"]["supports_translation"], true);
        assert_eq!(wcpp["capabilities"]["supports_word_timestamps"], true);
        assert_eq!(wcpp["capabilities"]["supports_gpu"], true);
        assert_eq!(wcpp["capabilities"]["supports_streaming"], false);
    }

    #[test]
    fn backends_discovery_insanely_fast_capabilities_are_correct() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        let backends = value["backends"].as_array().expect("backends array");
        let ifw = backends
            .iter()
            .find(|b| b["kind"] == "insanely_fast")
            .expect("insanely_fast entry");
        assert_eq!(ifw["name"], "insanely-fast-whisper");
        assert_eq!(ifw["available"], false);
        assert_eq!(ifw["capabilities"]["supports_diarization"], true);
        assert_eq!(ifw["capabilities"]["supports_translation"], true);
    }

    #[test]
    fn backends_discovery_whisper_diarization_capabilities_are_correct() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        let backends = value["backends"].as_array().expect("backends array");
        let wd = backends
            .iter()
            .find(|b| b["kind"] == "whisper_diarization")
            .expect("whisper_diarization entry");
        assert_eq!(wd["name"], "whisper-diarization");
        assert_eq!(wd["available"], false);
        assert_eq!(wd["capabilities"]["supports_diarization"], true);
        assert_eq!(wd["capabilities"]["supports_translation"], false);
        assert_eq!(wd["capabilities"]["supports_word_timestamps"], false);
    }

    #[test]
    fn backends_discovery_serializes_to_single_ndjson_line() {
        let report = make_test_backends_report();
        let value = super::backends_discovery_value(&report);
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(
            !line.contains('\n'),
            "NDJSON line must not contain newlines"
        );
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("parse back");
        assert_eq!(parsed["event"], "backends.discovery");
    }

    #[test]
    fn backends_discovery_round_trips_through_json() {
        let report = make_test_backends_report();
        let original = super::backends_discovery_value(&report);
        let json_str = serde_json::to_string(&original).expect("serialize");
        let round_tripped: serde_json::Value =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn backends_discovery_with_empty_backends() {
        let report = crate::model::BackendsReport { backends: vec![] };
        let value = super::backends_discovery_value(&report);
        assert_eq!(value["event"], "backends.discovery");
        let backends = value["backends"].as_array().expect("backends array");
        assert!(backends.is_empty());
        // Required fields still present.
        for field in super::BACKENDS_DISCOVERY_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "missing field `{field}` with empty backends"
            );
        }
    }

    #[test]
    fn backends_discovery_value_is_deterministic() {
        let report = make_test_backends_report();
        let v1 = super::backends_discovery_value(&report);
        let v2 = super::backends_discovery_value(&report);
        assert_eq!(v1, v2);
        let json1 = serde_json::to_string(&v1).expect("serialize");
        let json2 = serde_json::to_string(&v2).expect("serialize");
        assert_eq!(json1, json2, "serialized JSON should be byte-identical");
    }

    #[test]
    fn build_backends_report_returns_all_registered_backends() {
        let report = super::build_backends_report();
        assert_eq!(
            report.backends.len(),
            6,
            "should have exactly 6 backends (incl. 3 native pilots), got {}",
            report.backends.len()
        );
    }

    #[test]
    fn build_backends_report_covers_all_backend_kinds() {
        use crate::model::BackendKind;

        let report = super::build_backends_report();
        let kinds: Vec<BackendKind> = report.backends.iter().map(|b| b.kind).collect();
        assert!(kinds.contains(&BackendKind::WhisperCpp));
        assert!(kinds.contains(&BackendKind::InsanelyFast));
        assert!(kinds.contains(&BackendKind::WhisperDiarization));
    }

    #[test]
    fn build_backends_report_names_are_non_empty() {
        let report = super::build_backends_report();
        for entry in &report.backends {
            assert!(
                !entry.name.is_empty(),
                "backend name should not be empty for {:?}",
                entry.kind
            );
        }
    }

    #[test]
    fn build_backends_report_produces_valid_discovery_event() {
        let report = super::build_backends_report();
        let value = super::backends_discovery_value(&report);
        assert_eq!(value["event"], "backends.discovery");
        assert_eq!(value["schema_version"], super::ROBOT_SCHEMA_VERSION);
        let backends = value["backends"].as_array().expect("backends array");
        assert_eq!(backends.len(), 6);
        // Each entry has required structure.
        for entry in backends {
            assert!(entry["name"].is_string());
            assert!(entry["kind"].is_string());
            assert!(entry["available"].is_boolean());
            assert!(entry["capabilities"].is_object());
        }
    }

    #[test]
    fn backends_discovery_schema_example_satisfies_required_fields() {
        let schema = super::robot_schema_value();
        let disc = &schema["events"]["backends.discovery"];
        let required = disc["required"].as_array().expect("required array");
        let example = &disc["example"];
        for field in required {
            let field_name = field.as_str().expect("field string");
            assert!(
                example.get(field_name).is_some(),
                "schema example for backends.discovery missing required field `{field_name}`"
            );
        }
    }

    // ── routing_decision tests ──

    #[test]
    fn routing_decision_event_type_is_correct() {
        let payload = json!({
            "decision_id": "dec-1",
            "chosen_action": "try_whisper_cpp",
            "calibration_score": 0.91,
            "e_process": 1.23,
            "fallback_active": false,
            "recommended_order": ["whisper_cpp", "insanely_fast"],
            "mode": "adaptive",
        });
        let value = super::routing_decision_value(
            "run-123",
            "2026-02-22T00:00:00Z",
            "backend.routing.decision_contract",
            &payload,
        );
        assert_eq!(value["event"], "routing_decision");
        assert_eq!(value["schema_version"], super::ROBOT_SCHEMA_VERSION);
    }

    #[test]
    fn routing_decision_contains_required_fields() {
        let value = super::routing_decision_value(
            "run-123",
            "2026-02-22T00:00:00Z",
            "backend.routing.decision_contract",
            &json!({}),
        );
        for field in super::ROUTING_DECISION_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "routing_decision missing required field `{field}`"
            );
        }
    }

    #[test]
    fn routing_decision_schema_example_satisfies_required_fields() {
        let schema = super::robot_schema_value();
        let entry = &schema["events"]["routing_decision"];
        let required = entry["required"].as_array().expect("required array");
        let example = &entry["example"];
        for field in required {
            let field_name = field.as_str().expect("field string");
            assert!(
                example.get(field_name).is_some(),
                "schema example for routing_decision missing required field `{field_name}`"
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // bd-20g.3: transcript.partial event tests
    // ══════════════════════════════════════════════════════════════════════

    fn make_test_segment() -> crate::model::TranscriptionSegment {
        crate::model::TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.5),
            text: "hello world".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.95),
        }
    }

    fn make_test_segment_no_optionals() -> crate::model::TranscriptionSegment {
        crate::model::TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: "bare segment".to_owned(),
            speaker: None,
            confidence: None,
        }
    }

    #[test]
    fn transcript_partial_event_type_is_correct() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-tp", 0, "2026-02-22T00:00:01Z", &seg);
        assert_eq!(value["event"], "transcript.partial");
    }

    #[test]
    fn transcript_partial_has_schema_version() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-tp", 0, "2026-02-22T00:00:01Z", &seg);
        assert_eq!(value["schema_version"], super::ROBOT_SCHEMA_VERSION);
    }

    #[test]
    fn transcript_partial_contains_required_fields() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-tp", 0, "2026-02-22T00:00:01Z", &seg);
        for field in TRANSCRIPT_PARTIAL_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "transcript.partial missing required field `{field}`"
            );
        }
    }

    #[test]
    fn transcript_partial_has_exactly_required_fields() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-tp", 0, "2026-02-22T00:00:01Z", &seg);
        let obj = value.as_object().expect("should be object");
        for key in obj.keys() {
            assert!(
                TRANSCRIPT_PARTIAL_REQUIRED_FIELDS.contains(&key.as_str()),
                "unexpected field `{key}` in transcript.partial output"
            );
        }
        for field in TRANSCRIPT_PARTIAL_REQUIRED_FIELDS {
            assert!(
                obj.contains_key(*field),
                "missing required field `{field}` in transcript.partial output"
            );
        }
    }

    #[test]
    fn transcript_partial_preserves_segment_data() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-tp", 42, "2026-02-22T00:00:01Z", &seg);
        assert_eq!(value["run_id"], "run-tp");
        assert_eq!(value["seq"], 42);
        assert_eq!(value["ts"], "2026-02-22T00:00:01Z");
        assert_eq!(value["text"], "hello world");
        assert_eq!(value["start_sec"], 0.0);
        assert_eq!(value["end_sec"], 1.5);
        assert_eq!(value["confidence"], 0.95);
        assert_eq!(value["speaker"], "SPEAKER_00");
    }

    #[test]
    fn transcript_partial_with_none_optionals_serializes_nulls() {
        let seg = make_test_segment_no_optionals();
        let value = transcript_partial_value("run-tp", 0, "2026-02-22T00:00:01Z", &seg);
        assert!(
            value["start_sec"].is_null(),
            "None start_sec should be null"
        );
        assert!(value["end_sec"].is_null(), "None end_sec should be null");
        assert!(
            value["confidence"].is_null(),
            "None confidence should be null"
        );
        assert!(value["speaker"].is_null(), "None speaker should be null");
        assert_eq!(value["text"], "bare segment");
    }

    #[test]
    fn transcript_partial_serializes_to_single_ndjson_line() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-tp", 0, "2026-02-22T00:00:01Z", &seg);
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(
            !line.contains('\n'),
            "NDJSON line must not contain newlines"
        );
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("parse back");
        assert_eq!(parsed["event"], "transcript.partial");
    }

    #[test]
    fn transcript_partial_round_trips_through_json() {
        let seg = make_test_segment();
        let original = transcript_partial_value("run-tp", 5, "2026-02-22T12:00:00Z", &seg);
        let json_str = serde_json::to_string(&original).expect("serialize");
        let round_tripped: serde_json::Value =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn transcript_partial_value_is_deterministic() {
        let seg = make_test_segment();
        let v1 = transcript_partial_value("run-det", 0, "2026-02-22T00:00:00Z", &seg);
        let v2 = transcript_partial_value("run-det", 0, "2026-02-22T00:00:00Z", &seg);
        assert_eq!(v1, v2);
        let json1 = serde_json::to_string(&v1).expect("serialize");
        let json2 = serde_json::to_string(&v2).expect("serialize");
        assert_eq!(json1, json2, "serialized JSON should be byte-identical");
    }

    #[test]
    fn transcript_partial_with_unicode_text() {
        let seg = crate::model::TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(2.0),
            text: "日本語のテスト".to_owned(),
            speaker: Some("話者_01".to_owned()),
            confidence: Some(0.88),
        };
        let value = transcript_partial_value("run-unicode", 0, "2026-02-22T00:00:00Z", &seg);
        assert_eq!(value["text"], "日本語のテスト");
        assert_eq!(value["speaker"], "話者_01");
        // Single-line NDJSON.
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(!line.contains('\n'));
    }

    #[test]
    fn transcript_partial_with_empty_text() {
        let seg = crate::model::TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(0.5),
            text: String::new(),
            speaker: None,
            confidence: Some(0.1),
        };
        let value = transcript_partial_value("run-empty", 0, "2026-02-22T00:00:00Z", &seg);
        assert_eq!(value["text"], "");
    }

    #[test]
    fn transcript_partial_with_newlines_in_text() {
        let seg = crate::model::TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "line one\nline two".to_owned(),
            speaker: None,
            confidence: None,
        };
        let value = transcript_partial_value("run-nl", 0, "2026-02-22T00:00:00Z", &seg);
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(!line.contains('\n'), "NDJSON must not contain raw newlines");
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("parse back");
        assert!(parsed["text"].as_str().unwrap().contains('\n'));
    }

    #[test]
    fn transcript_partial_seq_zero_is_valid() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-tp", 0, "2026-02-22T00:00:00Z", &seg);
        assert_eq!(value["seq"], 0);
    }

    #[test]
    fn transcript_partial_large_seq() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-tp", u64::MAX, "2026-02-22T00:00:00Z", &seg);
        assert_eq!(value["seq"], u64::MAX);
    }

    #[test]
    fn transcript_partial_with_zero_timestamps() {
        let seg = crate::model::TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(0.0),
            text: "instant".to_owned(),
            speaker: None,
            confidence: Some(1.0),
        };
        let value = transcript_partial_value("run-zero-ts", 0, "2026-02-22T00:00:00Z", &seg);
        assert_eq!(value["start_sec"], 0.0);
        assert_eq!(value["end_sec"], 0.0);
    }

    #[test]
    fn transcript_partial_with_high_precision_timestamps() {
        let seg = crate::model::TranscriptionSegment {
            start_sec: Some(123.456_789),
            end_sec: Some(456.789_012),
            text: "precise".to_owned(),
            speaker: None,
            confidence: Some(0.999_999),
        };
        let value = transcript_partial_value("run-prec", 0, "2026-02-22T00:00:00Z", &seg);
        assert_eq!(value["start_sec"], 123.456_789);
        assert_eq!(value["end_sec"], 456.789_012);
    }

    #[test]
    fn transcript_partial_with_empty_run_id() {
        let seg = make_test_segment();
        let value = transcript_partial_value("", 0, "2026-02-22T00:00:00Z", &seg);
        assert_eq!(value["run_id"], "");
        // Still has all required fields.
        for field in TRANSCRIPT_PARTIAL_REQUIRED_FIELDS {
            assert!(value.get(*field).is_some(), "missing field `{field}`");
        }
    }

    #[test]
    fn transcript_partial_field_types_are_correct() {
        let seg = make_test_segment();
        let value = transcript_partial_value("run-types", 1, "2026-02-22T00:00:00Z", &seg);
        assert!(value["event"].is_string());
        assert!(value["schema_version"].is_string());
        assert!(value["run_id"].is_string());
        assert!(value["seq"].is_u64());
        assert!(value["ts"].is_string());
        assert!(value["text"].is_string());
        assert!(value["start_sec"].is_f64());
        assert!(value["end_sec"].is_f64());
        assert!(value["confidence"].is_f64());
        assert!(value["speaker"].is_string());
    }

    #[test]
    fn transcript_partial_schema_example_satisfies_required_fields() {
        let schema = super::robot_schema_value();
        let tp = &schema["events"]["transcript.partial"];
        let required = tp["required"].as_array().expect("required array");
        let example = &tp["example"];
        for field in required {
            let field_name = field.as_str().expect("field string");
            assert!(
                example.get(field_name).is_some(),
                "schema example for transcript.partial missing required field `{field_name}`"
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // bd-20g.5: health.report event tests
    // ══════════════════════════════════════════════════════════════════════

    fn make_test_health_report() -> super::HealthReport {
        super::HealthReport {
            ts: "2026-02-22T00:00:00Z".to_owned(),
            backends: vec![
                super::DependencyCheck {
                    name: "whisper.cpp".to_owned(),
                    available: true,
                    path: Some("/usr/local/bin/whisper-cli".to_owned()),
                    version: None,
                    issues: vec![],
                },
                super::DependencyCheck {
                    name: "insanely-fast-whisper".to_owned(),
                    available: false,
                    path: None,
                    version: None,
                    issues: vec!["insanely-fast-whisper not found".to_owned()],
                },
            ],
            ffmpeg: super::DependencyCheck {
                name: "ffmpeg".to_owned(),
                available: true,
                path: Some("/usr/bin/ffmpeg".to_owned()),
                version: None,
                issues: vec![],
            },
            database: super::DependencyCheck {
                name: "database".to_owned(),
                available: true,
                path: Some("/tmp/test.sqlite3".to_owned()),
                version: None,
                issues: vec![],
            },
            resources: super::ResourceSnapshot {
                disk_free_bytes: None,
                disk_total_bytes: None,
                memory_available_bytes: Some(8_000_000_000),
                memory_total_bytes: Some(16_000_000_000),
            },
            overall_status: super::CheckStatus::Ok,
        }
    }

    #[test]
    fn health_report_event_type_is_correct() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert_eq!(value["event"], "health.report");
    }

    #[test]
    fn health_report_has_schema_version() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert_eq!(value["schema_version"], super::ROBOT_SCHEMA_VERSION);
    }

    #[test]
    fn health_report_contains_required_fields() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        for field in HEALTH_REPORT_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "health.report missing required field `{field}`"
            );
        }
    }

    #[test]
    fn health_report_has_exactly_required_fields() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        let obj = value.as_object().expect("should be object");
        for key in obj.keys() {
            assert!(
                HEALTH_REPORT_REQUIRED_FIELDS.contains(&key.as_str()),
                "unexpected field `{key}` in health.report output"
            );
        }
        for field in HEALTH_REPORT_REQUIRED_FIELDS {
            assert!(
                obj.contains_key(*field),
                "missing required field `{field}` in health.report output"
            );
        }
    }

    #[test]
    fn health_report_backends_is_array() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert!(value["backends"].is_array(), "backends should be array");
        assert_eq!(value["backends"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn health_report_ffmpeg_is_object() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert!(value["ffmpeg"].is_object(), "ffmpeg should be object");
        assert_eq!(value["ffmpeg"]["name"], "ffmpeg");
        assert_eq!(value["ffmpeg"]["available"], true);
    }

    #[test]
    fn health_report_database_is_object() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert!(value["database"].is_object(), "database should be object");
        assert_eq!(value["database"]["name"], "database");
        assert_eq!(value["database"]["available"], true);
    }

    #[test]
    fn health_report_resources_is_object() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert!(value["resources"].is_object(), "resources should be object");
        assert_eq!(
            value["resources"]["memory_available_bytes"],
            8_000_000_000_u64
        );
        assert_eq!(value["resources"]["memory_total_bytes"], 16_000_000_000_u64);
    }

    #[test]
    fn health_report_overall_status_ok() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert_eq!(value["overall_status"], "ok");
    }

    #[test]
    fn health_report_overall_status_degraded() {
        let mut report = make_test_health_report();
        report.overall_status = super::CheckStatus::Degraded;
        let value = super::health_report_value(&report);
        assert_eq!(value["overall_status"], "degraded");
    }

    #[test]
    fn health_report_overall_status_unavailable() {
        let mut report = make_test_health_report();
        report.overall_status = super::CheckStatus::Unavailable;
        let value = super::health_report_value(&report);
        assert_eq!(value["overall_status"], "unavailable");
    }

    #[test]
    fn health_report_serializes_to_single_ndjson_line() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(
            !line.contains('\n'),
            "NDJSON line must not contain newlines"
        );
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("parse back");
        assert_eq!(parsed["event"], "health.report");
    }

    #[test]
    fn health_report_round_trips_through_json() {
        let report = make_test_health_report();
        let original = super::health_report_value(&report);
        let json_str = serde_json::to_string(&original).expect("serialize");
        let round_tripped: serde_json::Value =
            serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(original, round_tripped);
    }

    #[test]
    fn health_report_with_no_backends() {
        let mut report = make_test_health_report();
        report.backends = vec![];
        let value = super::health_report_value(&report);
        let backends = value["backends"].as_array().expect("backends array");
        assert!(backends.is_empty());
        // Required fields still present.
        for field in HEALTH_REPORT_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "missing field `{field}` with empty backends"
            );
        }
    }

    #[test]
    fn health_report_backend_entry_has_expected_fields() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        let backends = value["backends"].as_array().expect("backends array");
        let expected_fields = ["name", "available", "path", "version", "issues"];
        for entry in backends {
            for field in &expected_fields {
                assert!(
                    entry.get(*field).is_some(),
                    "backend entry missing field `{field}`: {entry}"
                );
            }
        }
    }

    #[test]
    fn health_report_unavailable_backend_has_issues() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        let backends = value["backends"].as_array().expect("backends array");
        let unavailable = backends
            .iter()
            .find(|b| b["available"] == false)
            .expect("should have unavailable backend");
        let issues = unavailable["issues"].as_array().expect("issues array");
        assert!(!issues.is_empty(), "unavailable backend should have issues");
    }

    #[test]
    fn health_report_available_backend_has_no_issues() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        let backends = value["backends"].as_array().expect("backends array");
        let available = backends
            .iter()
            .find(|b| b["available"] == true)
            .expect("should have available backend");
        let issues = available["issues"].as_array().expect("issues array");
        assert!(issues.is_empty(), "available backend should have no issues");
    }

    #[test]
    fn health_report_resources_with_all_none() {
        let mut report = make_test_health_report();
        report.resources = super::ResourceSnapshot {
            disk_free_bytes: None,
            disk_total_bytes: None,
            memory_available_bytes: None,
            memory_total_bytes: None,
        };
        let value = super::health_report_value(&report);
        assert!(value["resources"]["disk_free_bytes"].is_null());
        assert!(value["resources"]["disk_total_bytes"].is_null());
        assert!(value["resources"]["memory_available_bytes"].is_null());
        assert!(value["resources"]["memory_total_bytes"].is_null());
    }

    #[test]
    fn health_report_resources_with_all_populated() {
        let mut report = make_test_health_report();
        report.resources = super::ResourceSnapshot {
            disk_free_bytes: Some(100_000_000_000),
            disk_total_bytes: Some(500_000_000_000),
            memory_available_bytes: Some(8_000_000_000),
            memory_total_bytes: Some(32_000_000_000),
        };
        let value = super::health_report_value(&report);
        assert_eq!(value["resources"]["disk_free_bytes"], 100_000_000_000_u64);
        assert_eq!(value["resources"]["disk_total_bytes"], 500_000_000_000_u64);
        assert_eq!(
            value["resources"]["memory_available_bytes"],
            8_000_000_000_u64
        );
        assert_eq!(value["resources"]["memory_total_bytes"], 32_000_000_000_u64);
    }

    #[test]
    fn health_report_schema_example_satisfies_required_fields() {
        let schema = super::robot_schema_value();
        let hr = &schema["events"]["health.report"];
        let required = hr["required"].as_array().expect("required array");
        let example = &hr["example"];
        for field in required {
            let field_name = field.as_str().expect("field string");
            assert!(
                example.get(field_name).is_some(),
                "schema example for health.report missing required field `{field_name}`"
            );
        }
    }

    #[test]
    fn health_report_preserves_timestamp() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert_eq!(value["ts"], "2026-02-22T00:00:00Z");
    }

    #[test]
    fn health_report_database_path_preserved() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert_eq!(value["database"]["path"], "/tmp/test.sqlite3");
    }

    #[test]
    fn health_report_ffmpeg_path_preserved() {
        let report = make_test_health_report();
        let value = super::health_report_value(&report);
        assert_eq!(value["ffmpeg"]["path"], "/usr/bin/ffmpeg");
    }

    // --- CheckStatus tests ---

    #[test]
    fn check_status_as_str_all_variants() {
        assert_eq!(super::CheckStatus::Ok.as_str(), "ok");
        assert_eq!(super::CheckStatus::Degraded.as_str(), "degraded");
        assert_eq!(super::CheckStatus::Unavailable.as_str(), "unavailable");
    }

    #[test]
    fn check_status_equality() {
        assert_eq!(super::CheckStatus::Ok, super::CheckStatus::Ok);
        assert_eq!(super::CheckStatus::Degraded, super::CheckStatus::Degraded);
        assert_eq!(
            super::CheckStatus::Unavailable,
            super::CheckStatus::Unavailable
        );
        assert_ne!(super::CheckStatus::Ok, super::CheckStatus::Degraded);
        assert_ne!(super::CheckStatus::Ok, super::CheckStatus::Unavailable);
        assert_ne!(
            super::CheckStatus::Degraded,
            super::CheckStatus::Unavailable
        );
    }

    // --- check_database tests ---

    #[test]
    fn check_database_with_existing_parent() {
        let check = super::check_database(std::path::Path::new("/tmp/test.sqlite3"));
        assert_eq!(check.name, "database");
        assert!(check.available, "/tmp exists so parent is accessible");
        assert!(check.issues.is_empty());
        assert_eq!(check.path.as_deref(), Some("/tmp/test.sqlite3"));
    }

    #[test]
    fn check_database_with_nonexistent_parent() {
        let check =
            super::check_database(std::path::Path::new("/nonexistent_dir_abc123/db.sqlite3"));
        assert!(!check.available, "nonexistent parent should be unavailable");
        assert!(!check.issues.is_empty(), "should report issues");
    }

    #[test]
    fn check_database_with_parent_file_is_unavailable() {
        use std::fs;

        let dir = tempfile::tempdir().expect("tempdir");
        let parent_file = dir.path().join("parent_file");
        fs::write(&parent_file, "not a directory").expect("write");

        let db_path = parent_file.join("db.sqlite3");
        let check = super::check_database(&db_path);
        assert!(
            !check.available,
            "parent file should not be treated as a writable directory"
        );
        assert!(
            check
                .issues
                .iter()
                .any(|issue| issue.contains("parent path is not a directory")),
            "issue should mention parent path not being a directory: {:?}",
            check.issues
        );
    }

    #[test]
    fn check_database_with_relative_path() {
        // A relative path like "db.sqlite3" has parent "" which is treated as
        // the current directory.
        let check = super::check_database(std::path::Path::new("db.sqlite3"));
        assert_eq!(check.name, "database");
        assert!(
            check.available,
            "relative path with empty parent should be ok"
        );
    }

    #[test]
    fn check_database_existing_directory_is_unavailable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let check = super::check_database(dir.path());
        assert!(
            !check.available,
            "directory path should not be treated as a writable database file"
        );
        assert!(
            check
                .issues
                .iter()
                .any(|issue| issue.contains("directory, not a file")),
            "issue should mention directory path: {:?}",
            check.issues
        );
    }

    #[test]
    fn check_database_read_only_file_is_unavailable() {
        use std::fs;

        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("readonly.sqlite3");
        fs::write(&db_path, "").expect("create db placeholder");

        let mut perms = fs::metadata(&db_path).expect("metadata").permissions();
        perms.set_readonly(true);
        fs::set_permissions(&db_path, perms).expect("set readonly");

        let check = super::check_database(&db_path);
        assert!(
            !check.available,
            "read-only database file should not be reported as writable"
        );
        assert!(
            check
                .issues
                .iter()
                .any(|issue| issue.contains("database file is not writable")),
            "issue should mention unwritable file: {:?}",
            check.issues
        );
    }

    // --- parse_meminfo_kb tests ---

    #[test]
    fn parse_meminfo_kb_typical_value() {
        let result = super::parse_meminfo_kb("  16384000 kB");
        assert_eq!(result, Some(16_384_000 * 1024));
    }

    #[test]
    fn parse_meminfo_kb_without_suffix() {
        let result = super::parse_meminfo_kb("  1024");
        assert_eq!(result, Some(1024 * 1024));
    }

    #[test]
    fn parse_meminfo_kb_non_numeric() {
        let result = super::parse_meminfo_kb("  not_a_number kB");
        assert_eq!(result, None);
    }

    #[test]
    fn parse_meminfo_kb_empty_string() {
        let result = super::parse_meminfo_kb("");
        assert_eq!(result, None);
    }

    #[test]
    fn parse_meminfo_kb_zero() {
        let result = super::parse_meminfo_kb("0 kB");
        assert_eq!(result, Some(0));
    }

    // --- snapshot_resources tests ---

    #[test]
    fn snapshot_resources_returns_valid_struct() {
        let snap = super::snapshot_resources();
        // On Linux with /proc/meminfo, memory fields should be Some.
        // On other platforms, they may be None. Either way the struct is valid.
        if let Some(avail) = snap.memory_available_bytes {
            assert!(avail > 0, "available memory should be positive if present");
        }
        if let Some(total) = snap.memory_total_bytes {
            assert!(total > 0, "total memory should be positive if present");
        }
        if let (Some(avail), Some(total)) = (snap.memory_available_bytes, snap.memory_total_bytes) {
            assert!(
                avail <= total,
                "available memory ({avail}) should not exceed total ({total})"
            );
        }
    }

    // --- build_health_report integration test ---

    #[test]
    fn build_health_report_returns_valid_report() {
        let report = super::build_health_report(std::path::Path::new("/tmp/test_health.sqlite3"));
        // Timestamp should be non-empty.
        assert!(!report.ts.is_empty(), "timestamp should be non-empty");
        // Should have at least one backend.
        assert!(
            !report.backends.is_empty(),
            "should report at least one backend"
        );
        // ffmpeg check should have a name.
        assert_eq!(report.ffmpeg.name, "ffmpeg");
        // database check should have a name.
        assert_eq!(report.database.name, "database");
        // overall_status should be one of the three variants.
        assert!(
            matches!(
                report.overall_status,
                super::CheckStatus::Ok
                    | super::CheckStatus::Degraded
                    | super::CheckStatus::Unavailable
            ),
            "overall_status should be a valid variant"
        );
    }

    #[test]
    fn build_health_report_produces_valid_ndjson_event() {
        let report = super::build_health_report(std::path::Path::new("/tmp/test_health.sqlite3"));
        let value = super::health_report_value(&report);
        assert_eq!(value["event"], "health.report");
        assert_eq!(value["schema_version"], super::ROBOT_SCHEMA_VERSION);
        // Required fields present.
        for field in HEALTH_REPORT_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "build_health_report output missing field `{field}`"
            );
        }
        // Serializes to single line.
        let line = serde_json::to_string(&value).expect("serialize");
        assert!(!line.contains('\n'), "NDJSON must be single line");
    }

    #[test]
    fn build_health_report_backends_have_names() {
        let report = super::build_health_report(std::path::Path::new("/tmp/test.sqlite3"));
        for backend in &report.backends {
            assert!(!backend.name.is_empty(), "backend name should not be empty");
        }
    }

    #[test]
    fn build_health_report_with_nonexistent_db_path() {
        let report =
            super::build_health_report(std::path::Path::new("/nonexistent_abc123/db.sqlite3"));
        assert!(
            !report.database.available,
            "db with nonexistent parent should be unavailable"
        );
        assert!(!report.database.issues.is_empty());
        // Overall status should be degraded or unavailable since db is not accessible.
        assert_ne!(
            report.overall_status,
            super::CheckStatus::Ok,
            "should not be ok when database is unavailable (unless no backends available either)"
        );
    }

    // --- Required fields constants tests ---

    #[test]
    fn transcript_partial_required_fields_are_non_empty_and_unique() {
        use std::collections::HashSet;
        assert!(
            !TRANSCRIPT_PARTIAL_REQUIRED_FIELDS.is_empty(),
            "required fields must not be empty"
        );
        let unique: HashSet<_> = TRANSCRIPT_PARTIAL_REQUIRED_FIELDS.iter().collect();
        assert_eq!(
            unique.len(),
            TRANSCRIPT_PARTIAL_REQUIRED_FIELDS.len(),
            "required fields must be unique"
        );
        for f in TRANSCRIPT_PARTIAL_REQUIRED_FIELDS {
            assert!(!f.is_empty(), "field name must not be empty string");
        }
    }

    #[test]
    fn health_report_required_fields_are_non_empty_and_unique() {
        use std::collections::HashSet;
        assert!(
            !HEALTH_REPORT_REQUIRED_FIELDS.is_empty(),
            "required fields must not be empty"
        );
        let unique: HashSet<_> = HEALTH_REPORT_REQUIRED_FIELDS.iter().collect();
        assert_eq!(
            unique.len(),
            HEALTH_REPORT_REQUIRED_FIELDS.len(),
            "required fields must be unique"
        );
        for f in HEALTH_REPORT_REQUIRED_FIELDS {
            assert!(!f.is_empty(), "field name must not be empty string");
        }
    }

    // --- DependencyCheck edge cases ---

    #[test]
    fn dependency_check_json_with_all_none_optionals() {
        let check = super::DependencyCheck {
            name: "test-dep".to_owned(),
            available: false,
            path: None,
            version: None,
            issues: vec![],
        };
        let json = super::dependency_check_json(&check);
        assert_eq!(json["name"], "test-dep");
        assert_eq!(json["available"], false);
        assert!(json["path"].is_null());
        assert!(json["version"].is_null());
        assert!(json["issues"].as_array().unwrap().is_empty());
    }

    #[test]
    fn dependency_check_json_with_all_populated() {
        let check = super::DependencyCheck {
            name: "ffmpeg".to_owned(),
            available: true,
            path: Some("/usr/bin/ffmpeg".to_owned()),
            version: Some("6.1.2".to_owned()),
            issues: vec!["warning: old version".to_owned()],
        };
        let json = super::dependency_check_json(&check);
        assert_eq!(json["name"], "ffmpeg");
        assert_eq!(json["available"], true);
        assert_eq!(json["path"], "/usr/bin/ffmpeg");
        assert_eq!(json["version"], "6.1.2");
        let issues = json["issues"].as_array().unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0], "warning: old version");
    }

    #[test]
    fn dependency_check_json_serializes_to_single_line() {
        let check = super::DependencyCheck {
            name: "test".to_owned(),
            available: true,
            path: Some("/path/with\nnewline".to_owned()),
            version: None,
            issues: vec!["issue with\nnewline".to_owned()],
        };
        let json = super::dependency_check_json(&check);
        let line = serde_json::to_string(&json).expect("serialize");
        assert!(!line.contains('\n'), "should be single line");
    }

    #[test]
    fn transcript_confirm_value_contains_required_fields_and_drift() {
        use crate::speculation::CorrectionDrift;

        let drift = CorrectionDrift {
            wer_approx: 0.12,
            confidence_delta: -0.05,
            segment_count_delta: 1,
            text_edit_distance: 3,
        };
        let value = super::transcript_confirm_value("run-tc", 7, 42, &drift, 210, "whisper-large");
        assert_eq!(value["event"], "transcript.confirm");
        assert_eq!(value["run_id"], "run-tc");
        assert_eq!(value["seq"], 7);
        assert_eq!(value["window_id"], 42);
        assert_eq!(value["quality_model_id"], "whisper-large");
        assert_eq!(value["latency_ms"], 210);
        assert_eq!(value["drift"]["wer_approx"], 0.12);
        assert_eq!(value["drift"]["text_edit_distance"], 3);
        for field in TRANSCRIPT_CONFIRM_REQUIRED_FIELDS {
            assert!(value.get(*field).is_some(), "missing field `{field}`");
        }
    }

    #[test]
    fn transcript_retract_value_contains_required_fields_and_reason() {
        let value = super::transcript_retract_value(
            "run-ret",
            5,
            11,
            "quality_correction",
            "whisper-large",
        );
        assert_eq!(value["event"], "transcript.retract");
        assert_eq!(value["retracted_seq"], 5);
        assert_eq!(value["window_id"], 11);
        assert_eq!(value["reason"], "quality_correction");
        assert_eq!(value["quality_model_id"], "whisper-large");
        assert!(value["ts"].is_string());
        for field in TRANSCRIPT_RETRACT_REQUIRED_FIELDS {
            assert!(value.get(*field).is_some(), "missing field `{field}`");
        }
    }

    #[test]
    fn transcript_correct_value_contains_required_fields_and_segments() {
        use crate::model::TranscriptionSegment;
        use crate::speculation::{CorrectionDrift, CorrectionEvent};

        let event = CorrectionEvent {
            correction_id: 7,
            retracted_seq: 3,
            window_id: 42,
            corrected_segments: vec![TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.5),
                text: "corrected text".to_owned(),
                speaker: Some("SPEAKER_00".to_owned()),
                confidence: Some(0.97),
            }],
            quality_model_id: "whisper-large".to_owned(),
            quality_latency_ms: 310,
            quality_confidence_mean: 0.95,
            drift: CorrectionDrift {
                wer_approx: 0.25,
                confidence_delta: 0.12,
                segment_count_delta: 0,
                text_edit_distance: 2,
            },
            corrected_at_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
        };
        let value = super::transcript_correct_value("run-corr", &event);
        assert_eq!(value["event"], "transcript.correct");
        assert_eq!(value["correction_id"], 7);
        assert_eq!(value["replaces_seq"], 3);
        assert_eq!(value["window_id"], 42);
        assert_eq!(value["drift"]["wer_approx"], 0.25);
        let segs = value["segments"].as_array().expect("segments array");
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0]["text"], "corrected text");
        for field in TRANSCRIPT_CORRECT_REQUIRED_FIELDS {
            assert!(value.get(*field).is_some(), "missing field `{field}`");
        }
    }

    #[test]
    fn speculation_stats_value_contains_required_fields() {
        use crate::speculation::SpeculationStats;

        let stats = SpeculationStats {
            windows_processed: 42,
            corrections_emitted: 10,
            confirmations_emitted: 32,
            correction_rate: 0.2381,
            mean_fast_latency_ms: 48.5,
            mean_quality_latency_ms: 207.4,
            current_window_size_ms: 3000,
            mean_drift_wer: 0.11,
        };
        let value = super::speculation_stats_value("run-stats", &stats);
        assert_eq!(value["event"], "transcript.speculation_stats");
        assert_eq!(value["run_id"], "run-stats");
        assert_eq!(value["windows_processed"], 42);
        assert_eq!(value["corrections_emitted"], 10);
        assert_eq!(value["confirmations_emitted"], 32);
        assert_eq!(value["current_window_size_ms"], 3000);
        assert!(value["ts"].is_string());
        for field in SPECULATION_STATS_REQUIRED_FIELDS {
            assert!(value.get(*field).is_some(), "missing field `{field}`");
        }
    }

    #[test]
    fn snapshot_resources_memory_fields_not_swapped() {
        let snap = super::snapshot_resources();
        if let (Some(avail), Some(total)) = (snap.memory_available_bytes, snap.memory_total_bytes) {
            assert!(
                avail <= total,
                "memory_available ({avail}) should be <= memory_total ({total})"
            );
            assert!(total >= 1_048_576, "total memory should be >= 1 MiB");
        }
    }

    // -- bd-251: robot.rs edge-case tests --

    #[test]
    fn parse_meminfo_kb_suffix_only_returns_none() {
        use super::parse_meminfo_kb;
        // " kB" → strip suffix → "" → parse fails → None
        assert_eq!(parse_meminfo_kb(" kB"), None);
        // "kB" → strip suffix → "" → parse fails → None
        assert_eq!(parse_meminfo_kb("kB"), None);
    }

    #[test]
    fn parse_meminfo_kb_valid_with_and_without_suffix() {
        use super::parse_meminfo_kb;
        // Normal case with kB suffix.
        assert_eq!(parse_meminfo_kb("16384 kB"), Some(16384 * 1024));
        // No suffix.
        assert_eq!(parse_meminfo_kb("8192"), Some(8192 * 1024));
        // Whitespace only → None.
        assert_eq!(parse_meminfo_kb("   "), None);
    }

    #[test]
    fn check_database_nonexistent_parent_reports_issue() {
        use std::path::Path;
        let dep = super::check_database(Path::new("/nonexistent_xyzzy_dir/test.db"));
        assert!(
            !dep.available,
            "should be unavailable for nonexistent parent"
        );
        assert!(
            !dep.issues.is_empty(),
            "should have issues for nonexistent parent"
        );
        assert!(
            dep.issues[0].contains("parent directory does not exist"),
            "issue should mention parent directory"
        );
    }

    #[test]
    fn check_database_current_dir_is_available() {
        use std::path::Path;
        let dep = super::check_database(Path::new("test.db"));
        // Parent is "" which maps to current dir — should be available.
        assert!(
            dep.available,
            "database in current directory should be available"
        );
        assert!(dep.issues.is_empty(), "should have no issues");
    }

    #[test]
    fn check_ffmpeg_result_has_correct_name() {
        let dep = super::check_ffmpeg();
        assert_eq!(dep.name, "ffmpeg");
        if dep.available {
            assert!(dep.path.is_some(), "path should be Some when available");
            assert!(dep.issues.is_empty(), "no issues when available");
        } else {
            assert!(dep.path.is_none(), "path should be None when unavailable");
            assert!(!dep.issues.is_empty(), "issues when unavailable");
            assert!(dep.issues[0].contains("not found"));
        }
    }

    // -----------------------------------------------------------------------
    // bd-1tl: acceleration_context_from_evidence tests
    // -----------------------------------------------------------------------

    #[test]
    fn acceleration_context_from_evidence_returns_none_on_empty() {
        let evidence: Vec<serde_json::Value> = vec![];
        assert!(super::acceleration_context_from_evidence(&evidence).is_none());
    }

    #[test]
    fn acceleration_context_from_evidence_returns_none_without_required_fields() {
        let evidence = vec![
            json!({"some_key": "some_value"}),
            json!({"logical_stream_owner_id": "trace:accel:none:cpu"}),
            json!({"cancellation_fence": {"epoch": 1}}),
        ];
        assert!(
            super::acceleration_context_from_evidence(&evidence).is_none(),
            "should return None when no single entry has both required fields"
        );
    }

    #[test]
    fn acceleration_context_from_evidence_returns_matching_entry() {
        let matching = json!({
            "logical_stream_owner_id": "trace:acceleration:none:cpu",
            "cancellation_fence": {"epoch": 1, "reason": "budget_exceeded"},
            "extra_field": 42,
        });
        let evidence = vec![json!({"unrelated": true}), matching.clone()];
        let result = super::acceleration_context_from_evidence(&evidence);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), matching);
    }

    #[test]
    fn acceleration_context_from_evidence_returns_last_matching_entry() {
        let first_match = json!({
            "logical_stream_owner_id": "trace:accel:v1",
            "cancellation_fence": {"epoch": 1},
        });
        let second_match = json!({
            "logical_stream_owner_id": "trace:accel:v2",
            "cancellation_fence": {"epoch": 2},
        });
        let evidence = vec![first_match, second_match.clone()];
        let result = super::acceleration_context_from_evidence(&evidence).unwrap();
        assert_eq!(
            result["logical_stream_owner_id"], "trace:accel:v2",
            "should return the last (most recent) matching entry"
        );
    }

    #[test]
    fn acceleration_context_from_evidence_ignores_null_stream_owner() {
        let evidence = vec![json!({
            "logical_stream_owner_id": null,
            "cancellation_fence": {"epoch": 1},
        })];
        assert!(
            super::acceleration_context_from_evidence(&evidence).is_none(),
            "null stream owner should not match (as_str returns None)"
        );
    }

    #[test]
    fn acceleration_context_from_evidence_ignores_non_object_fence() {
        let evidence = vec![json!({
            "logical_stream_owner_id": "trace:accel:cpu",
            "cancellation_fence": "not-an-object",
        })];
        assert!(
            super::acceleration_context_from_evidence(&evidence).is_none(),
            "string cancellation_fence should not match (is_object returns false)"
        );
    }

    #[test]
    fn acceleration_context_from_evidence_ignores_array_fence() {
        let evidence = vec![json!({
            "logical_stream_owner_id": "trace:accel:cpu",
            "cancellation_fence": [1, 2, 3],
        })];
        assert!(
            super::acceleration_context_from_evidence(&evidence).is_none(),
            "array cancellation_fence should not match"
        );
    }

    #[test]
    fn acceleration_context_from_evidence_accepts_empty_object_fence() {
        let entry = json!({
            "logical_stream_owner_id": "trace:accel:cpu",
            "cancellation_fence": {},
        });
        let evidence = vec![entry.clone()];
        let result = super::acceleration_context_from_evidence(&evidence).unwrap();
        assert_eq!(result, entry, "empty object is still a valid object fence");
    }

    #[test]
    fn acceleration_context_from_evidence_skips_numeric_stream_owner() {
        let evidence = vec![json!({
            "logical_stream_owner_id": 12345,
            "cancellation_fence": {"epoch": 1},
        })];
        assert!(
            super::acceleration_context_from_evidence(&evidence).is_none(),
            "numeric stream owner should not match (as_str returns None)"
        );
    }

    // -----------------------------------------------------------------------
    // bd-1tl: run_complete_value includes acceleration_context when present
    // -----------------------------------------------------------------------

    #[test]
    fn run_complete_value_includes_acceleration_context_when_evidence_has_it() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "accel-ctx-test".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-26T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-26T00:00:01Z".to_owned(),
            input_path: "test.wav".to_owned(),
            normalized_wav_path: "norm.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("test.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("test.db"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "hello".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![json!({
                "logical_stream_owner_id": "trace:acceleration:none:cpu",
                "cancellation_fence": {"epoch": 1},
                "logical_stream_kind": "cpu_lane",
            })],
            replay: Default::default(),
        };

        let value = super::run_complete_value(&report);
        assert!(
            value.get("acceleration_context").is_some(),
            "run_complete should include acceleration_context when evidence matches"
        );
        assert_eq!(
            value["acceleration_context"]["logical_stream_owner_id"],
            "trace:acceleration:none:cpu"
        );
    }

    #[test]
    fn run_complete_value_omits_acceleration_context_when_evidence_lacks_it() {
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunReport, TranscriptionResult,
        };
        use std::path::PathBuf;

        let report = RunReport {
            run_id: "no-ctx-test".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-02-26T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-26T00:00:01Z".to_owned(),
            input_path: "test.wav".to_owned(),
            normalized_wav_path: "norm.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("test.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: false,
                persist: false,
                db_path: PathBuf::from("test.db"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "hello".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![json!({"unrelated": "data"})],
            replay: Default::default(),
        };

        let value = super::run_complete_value(&report);
        assert!(
            value.get("acceleration_context").is_none(),
            "run_complete should not include acceleration_context when evidence has no matching entry"
        );
    }

    // -----------------------------------------------------------------------
    // bd-1tl: transcript_retract_value schema compliance
    // -----------------------------------------------------------------------

    #[test]
    fn transcript_retract_value_contains_required_fields() {
        let value = super::transcript_retract_value(
            "run-retract",
            42,
            7,
            "quality model disagrees",
            "whisper-large-v3",
        );
        for field in super::TRANSCRIPT_RETRACT_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "transcript.retract missing required field `{field}`"
            );
        }
        assert_eq!(value["event"], "transcript.retract");
        assert_eq!(value["retracted_seq"], 42);
        assert_eq!(value["window_id"], 7);
        assert_eq!(value["reason"], "quality model disagrees");
        assert_eq!(value["quality_model_id"], "whisper-large-v3");
    }

    // -----------------------------------------------------------------------
    // bd-1tl: speculation_stats_value edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn speculation_stats_value_zero_windows_produces_zero_rate() {
        let stats = crate::speculation::SpeculationStats {
            windows_processed: 0,
            corrections_emitted: 0,
            confirmations_emitted: 0,
            correction_rate: 0.0,
            mean_fast_latency_ms: 0.0,
            mean_quality_latency_ms: 0.0,
            current_window_size_ms: 5000,
            mean_drift_wer: 0.0,
        };
        let value = super::speculation_stats_value("run-zero", &stats);
        assert_eq!(value["windows_processed"], 0);
        assert_eq!(value["correction_rate"], 0.0);
        assert_eq!(value["current_window_size_ms"], 5000);
    }

    // -----------------------------------------------------------------------
    // bd-1tl: transcript_partial_value edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn transcript_partial_value_with_all_none_optional_fields() {
        use crate::model::TranscriptionSegment;
        let seg = TranscriptionSegment {
            text: "hello".to_owned(),
            start_sec: None,
            end_sec: None,
            confidence: None,
            speaker: None,
        };
        let value = super::transcript_partial_value("run-opt", 0, "2026-02-26T00:00:00Z", &seg);
        assert!(value["start_sec"].is_null());
        assert!(value["end_sec"].is_null());
        assert!(value["confidence"].is_null());
        assert!(value["speaker"].is_null());
        assert_eq!(value["text"], "hello");
    }

    #[test]
    fn transcript_partial_value_with_all_fields_populated() {
        use crate::model::TranscriptionSegment;
        let seg = TranscriptionSegment {
            text: "world".to_owned(),
            start_sec: Some(1.5),
            end_sec: Some(3.25),
            confidence: Some(0.875),
            speaker: Some("SPEAKER_01".to_owned()),
        };
        let value = super::transcript_partial_value("run-full", 5, "2026-02-26T00:01:00Z", &seg);
        assert_eq!(value["text"], "world");
        assert_eq!(value["start_sec"], 1.5);
        assert_eq!(value["end_sec"], 3.25);
        assert_eq!(value["confidence"], 0.875);
        assert_eq!(value["speaker"], "SPEAKER_01");
        assert_eq!(value["seq"], 5);
    }

    // -----------------------------------------------------------------------
    // bd-1tl: backends_discovery_value edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn backends_discovery_value_empty_backends_list() {
        use crate::model::BackendsReport;
        let report = BackendsReport { backends: vec![] };
        let value = super::backends_discovery_value(&report);
        assert_eq!(value["event"], "backends.discovery");
        assert!(value["backends"].as_array().unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // bd-1tl: CheckStatus edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn check_status_debug_representation() {
        assert_eq!(format!("{:?}", super::CheckStatus::Ok), "Ok");
        assert_eq!(format!("{:?}", super::CheckStatus::Degraded), "Degraded");
        assert_eq!(
            format!("{:?}", super::CheckStatus::Unavailable),
            "Unavailable"
        );
    }

    #[test]
    fn check_status_copy_semantics() {
        let status = super::CheckStatus::Degraded;
        let copied = status;
        let copied2 = status;
        assert_eq!(copied, copied2);
        assert_eq!(status, super::CheckStatus::Degraded);
    }
}
