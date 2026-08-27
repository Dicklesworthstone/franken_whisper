use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::Utc;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
// The JSONL sync drives the database synchronously from the CLI. `fsqlite`'s
// own `Connection` is async as of the frankensqlite async migration, so this
// module speaks to it through the crate's blocking facade; every `Connection`
// below is that facade, with the same statement methods and error type.
use crate::storage::{BlockingConnection as Connection, RunStore};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{FwError, FwResult};

/// Whether the JSONL import loops write their rows with the fsqlite "skip
/// statement savepoint in explicit txn" escape hatch (default). Every import
/// INSERT/DELETE runs inside `import_inner`'s single `BEGIN;` transaction, which
/// is the rollback boundary (COMMIT on success, ROLLBACK on any Err or Reject),
/// so the per-statement savepoint fsqlite wraps around each `execute_with_params`
/// is redundant bookkeeping. `FW_SYNC_SKIP_STMT_SP=0` restores the per-statement
/// savepoint path. Imported rows are byte-identical either way. Mirrors the
/// `persist_report` win (`FW_PERSIST_SKIP_STMT_SP`).
fn sync_skip_stmt_sp_enabled() -> bool {
    std::env::var("FW_SYNC_SKIP_STMT_SP").ok().as_deref() != Some("0")
}

/// How many `run_id`s the incremental `_for_runs` export writers pack into each
/// `WHERE run_id IN (…)` query, collapsing the legacy per-run N+1 `SELECT` into one
/// query per chunk (query *setup* dominates for the few-row-per-run results).
/// Default 512 (well under fsqlite's 32766-variable limit); `FW_SYNC_BATCH_QUERY=0`
/// forces 1 (= one query per run = the legacy N+1). Rows are grouped by run_id and
/// emitted in `run_ids` order regardless, so the JSONL is byte-identical either way.
fn sync_query_batch_size() -> usize {
    if std::env::var("FW_SYNC_BATCH_QUERY").ok().as_deref() == Some("0") {
        1
    } else {
        512
    }
}

/// Import-side mirror of the export N+1 batching (all three tables): pre-fetch the
/// existing rows for a chunk with one `WHERE … IN (…)` query instead of one
/// `SELECT … WHERE key = ?` per JSONL line (query *setup* dominates the few-row
/// lookups). **Default ON** (`FW_SYNC_BATCH_IMPORT=0` restores the legacy per-line
/// N+1, matching the `FW_SYNC_BATCH_QUERY` kill-switch convention). Proven safe to
/// flip: the per-line and batched paths call the SAME `apply_{run,segment,event}_row`
/// conflict logic (differing only in where `existing` comes from) ⇒ byte-identical
/// imported rows; verified by `sync::tests` (350/0, now exercised through the batched
/// path by default) + a full-CLI export→import A/B (all 3 tables byte-identical incl.
/// the conflict/noop re-import path). Measured **~1.29× on `sync/import/runs/50`**
/// (50.9→39.6 ms, external-env ABBA), matching the export N+1's 1.32×.
fn sync_batch_import_enabled() -> bool {
    std::env::var("FW_SYNC_BATCH_IMPORT").ok().as_deref() != Some("0")
}

/// The 12 `runs` columns in `SELECT`/`INSERT` order, as an owned slice — the
/// shared representation of "the existing row" for [`apply_run_row`]. Both the
/// per-line `SELECT` result (a `Row`) and a prefetched/just-inserted row reduce to
/// this, and `<[SqliteValue]>::get(i)` returns `Option<&SqliteValue>` exactly like
/// `Row::get(i)`, so the 11-field identical-compare is bit-for-bit the same.
fn run_row_to_cols(row: &fsqlite::Row) -> Vec<SqliteValue> {
    (0..12)
        .map(|i| row.get(i).cloned().unwrap_or(SqliteValue::Null))
        .collect()
}

/// Extension over `fsqlite::Connection` for the import write loops: dispatches to
/// the statement-savepoint-skipping executor when enabled (see
/// [`sync_skip_stmt_sp_enabled`]). Behaviourally identical to `execute_with_params`
/// on success; on failure the enclosing `BEGIN;` rollback discards partial rows.
trait ImportExec {
    fn import_exec(
        &self,
        sql: &str,
        params: &[SqliteValue],
    ) -> Result<usize, fsqlite::FrankenError>;
}

impl ImportExec for Connection {
    fn import_exec(
        &self,
        sql: &str,
        params: &[SqliteValue],
    ) -> Result<usize, fsqlite::FrankenError> {
        if sync_skip_stmt_sp_enabled() {
            self.execute_with_params_skip_statement_savepoint_in_explicit_txn(sql, params)
        } else {
            self.execute_with_params(sql, params)
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SCHEMA_VERSION: &str = "1.1";
const EXPORT_FORMAT_VERSION: &str = "1.0";
const LOCK_STALE_SECONDS: i64 = 300; // 5 minutes
const MAX_SYNC_VALIDATION_CHILD_MISMATCHES: usize = 128;

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncManifest {
    pub schema_version: String,
    pub export_format_version: String,
    pub created_at_rfc3339: String,
    pub source_db_path: String,
    pub row_counts: RowCounts,
    pub checksums: FileChecksums,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowCounts {
    pub runs: u64,
    pub segments: u64,
    pub events: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileChecksums {
    pub runs_jsonl_sha256: String,
    pub segments_jsonl_sha256: String,
    pub events_jsonl_sha256: String,
}

// ---------------------------------------------------------------------------
// Lock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LockInfo {
    pid: u32,
    created_at_rfc3339: String,
    operation: String,
}

pub struct SyncLock {
    path: PathBuf,
    released: bool,
    info: LockInfo,
    payload: String,
}

impl SyncLock {
    pub fn acquire(state_root: &Path, operation: &str) -> FwResult<Self> {
        Self::acquire_with_after_create(state_root, operation, || {})
    }

    fn acquire_with_after_create(
        state_root: &Path,
        operation: &str,
        after_create: impl FnOnce(),
    ) -> FwResult<Self> {
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir)?;
        let path = locks_dir.join("sync.lock");

        if path.exists()
            && let Some(info) = read_lock_info(&path)?
        {
            if is_lock_stale(&info) {
                archive_stale_lock(&path, "stale")?;
            } else {
                return Err(FwError::Storage(format!(
                    "sync lock held by pid {} since {}; \
                     remove {} if the process is dead",
                    info.pid,
                    info.created_at_rfc3339,
                    path.display()
                )));
            }
        }

        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            operation: operation.to_owned(),
        };
        let payload = serde_json::to_string_pretty(&info)?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                FwError::Storage(format!(
                    "failed to acquire sync lock {}: {error}",
                    path.display()
                ))
            })?;
        after_create();
        file.write_all(payload.as_bytes())?;
        file.sync_all()?;

        Ok(Self {
            path,
            released: false,
            info,
            payload,
        })
    }

    pub fn release(mut self) -> FwResult<()> {
        self.release_inner()
    }

    fn release_inner(&mut self) -> FwResult<()> {
        if self.released {
            return Ok(());
        }
        if self.path.exists() {
            match fs::read_to_string(&self.path) {
                Ok(contents) => {
                    if contents == self.payload {
                        fs::remove_file(&self.path)?;
                    } else if let Ok(current) = serde_json::from_str::<LockInfo>(&contents) {
                        if lock_info_matches(&self.info, &current) {
                            fs::remove_file(&self.path)?;
                        } else {
                            tracing::warn!(
                                lock_path = %self.path.display(),
                                "sync lock contents changed; skipping removal"
                            );
                        }
                    } else {
                        tracing::warn!(
                            lock_path = %self.path.display(),
                            "sync lock contents unreadable; skipping removal"
                        );
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        lock_path = %self.path.display(),
                        error = %error,
                        "failed to read sync lock; skipping removal"
                    );
                }
            }
        }
        self.released = true;
        Ok(())
    }
}

impl Drop for SyncLock {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

fn lock_info_matches(a: &LockInfo, b: &LockInfo) -> bool {
    a.pid == b.pid && a.created_at_rfc3339 == b.created_at_rfc3339 && a.operation == b.operation
}

fn read_lock_info(path: &Path) -> FwResult<Option<LockInfo>> {
    read_lock_info_at(path, SystemTime::now())
}

fn read_lock_info_at(path: &Path, now: SystemTime) -> FwResult<Option<LockInfo>> {
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .map_err(|error| {
            FwError::Storage(format!(
                "failed to inspect sync lock {}: {error}",
                path.display()
            ))
        })?;
    let bytes = fs::read(path).map_err(|error| {
        FwError::Storage(format!(
            "failed to read sync lock {}: {error}",
            path.display()
        ))
    })?;
    let contents = match String::from_utf8(bytes) {
        Ok(contents) => contents,
        Err(_) => {
            return handle_unreadable_lock(path, modified, now, "non-UTF-8");
        }
    };

    match serde_json::from_str::<LockInfo>(&contents) {
        Ok(info) => Ok(Some(info)),
        Err(_) => handle_unreadable_lock(path, modified, now, "invalid JSON"),
    }
}

fn handle_unreadable_lock(
    path: &Path,
    modified: SystemTime,
    now: SystemTime,
    detail: &str,
) -> FwResult<Option<LockInfo>> {
    let age_seconds = now
        .duration_since(modified)
        .map(|age| age.as_secs())
        .unwrap_or(0);
    if age_seconds > LOCK_STALE_SECONDS as u64 {
        archive_stale_lock(path, "corrupt")?;
        return Ok(None);
    }

    Err(FwError::Storage(format!(
        "sync lock {} is {detail} but only {age_seconds}s old; refusing to archive a potentially active partial lock",
        path.display()
    )))
}

fn is_lock_stale(info: &LockInfo) -> bool {
    // Check if PID is still alive
    let pid_alive = pid_is_alive(info.pid);
    if !pid_alive {
        #[cfg(target_os = "linux")]
        {
            return true;
        }

        #[cfg(not(target_os = "linux"))]
        {
            match chrono::DateTime::parse_from_rfc3339(&info.created_at_rfc3339) {
                Ok(created) => {
                    let age = Utc::now().signed_duration_since(created);
                    if age.num_seconds() > LOCK_STALE_SECONDS {
                        tracing::warn!(
                            pid = info.pid,
                            age_seconds = age.num_seconds(),
                            "sync lock age exceeds stale threshold and pid liveness is unknown; treating as stale"
                        );
                        return true;
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        pid = info.pid,
                        error = %error,
                        "sync lock timestamp invalid; treating as stale"
                    );
                    return true;
                }
            }

            return false;
        }
    }

    // Check age
    match chrono::DateTime::parse_from_rfc3339(&info.created_at_rfc3339) {
        Ok(created) => {
            let age = Utc::now().signed_duration_since(created);
            if age.num_seconds() > LOCK_STALE_SECONDS {
                tracing::warn!(
                    pid = info.pid,
                    age_seconds = age.num_seconds(),
                    "sync lock age exceeds stale threshold but process is alive; refusing to treat as stale"
                );
            }
        }
        Err(error) => {
            tracing::warn!(
                pid = info.pid,
                error = %error,
                "sync lock timestamp invalid; leaving lock in place"
            );
        }
    }

    false
}

fn archive_stale_lock(path: &Path, reason: &str) -> FwResult<()> {
    let timestamp = Utc::now().timestamp_millis();
    let archived = path.with_file_name(format!("sync.lock.{reason}.{timestamp}.json"));
    fs::rename(path, archived)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn pid_is_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(all(unix, not(target_os = "linux")))]
fn pid_is_alive(pid: u32) -> bool {
    match std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                return true;
            }

            let stderr = String::from_utf8_lossy(&output.stderr).to_ascii_lowercase();
            stderr.contains("operation not permitted")
                || stderr.contains("permission denied")
                || stderr.contains("not permitted")
        }
        Err(_) => pid == std::process::id(),
    }
}

#[cfg(windows)]
fn pid_is_alive(pid: u32) -> bool {
    if pid == std::process::id() {
        return true;
    }

    let filter = format!("PID eq {pid}");
    let output = std::process::Command::new("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let needle = format!("\"{pid}\"");
            stdout.contains(&needle)
        }
        Ok(_) => false,
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Export (db -> jsonl)
// ---------------------------------------------------------------------------

pub fn export(db_path: &Path, output_dir: &Path, state_root: &Path) -> FwResult<SyncManifest> {
    let _lock = SyncLock::acquire(state_root, "export")?;
    export_inner(db_path, output_dir)
}

fn with_read_snapshot<T>(
    connection: &Connection,
    operation: impl FnOnce() -> FwResult<T>,
) -> FwResult<T> {
    connection
        .execute("BEGIN;")
        .map_err(|error| FwError::Storage(error.to_string()))?;

    match operation() {
        Ok(value) => {
            if let Err(error) = connection.execute("COMMIT;") {
                let _ = connection.execute("ROLLBACK;");
                return Err(FwError::Storage(error.to_string()));
            }
            Ok(value)
        }
        Err(error) => {
            let _ = connection.execute("ROLLBACK;");
            Err(error)
        }
    }
}

fn export_inner(db_path: &Path, output_dir: &Path) -> FwResult<SyncManifest> {
    export_inner_with_after_runs(db_path, output_dir, || {})
}

fn export_inner_with_after_runs(
    db_path: &Path,
    output_dir: &Path,
    after_runs: impl FnOnce(),
) -> FwResult<SyncManifest> {
    fs::create_dir_all(output_dir)?;

    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    verify_schema_exists(connection)?;

    let (
        (runs_count, runs_sha256),
        (segments_count, segments_sha256),
        (events_count, events_sha256),
    ) = with_read_snapshot(&connection, || {
        // Export runs
        let runs_tmp = output_dir.join("runs.jsonl.tmp");
        let runs_final = output_dir.join("runs.jsonl");
        let runs = export_table_runs(&connection, &runs_tmp)?;
        atomic_rename(&runs_tmp, &runs_final)?;

        after_runs();

        // Export segments
        let segments_tmp = output_dir.join("segments.jsonl.tmp");
        let segments_final = output_dir.join("segments.jsonl");
        let segments = export_table_segments(&connection, &segments_tmp)?;
        atomic_rename(&segments_tmp, &segments_final)?;

        // Export events
        let events_tmp = output_dir.join("events.jsonl.tmp");
        let events_final = output_dir.join("events.jsonl");
        let events = export_table_events(&connection, &events_tmp)?;
        atomic_rename(&events_tmp, &events_final)?;

        Ok((runs, segments, events))
    })?;

    // Checksums streamed while writing (HashingWriter) — no second pass to re-read
    // the JSONL files. Identical digest to `sha256_file` of the written bytes.
    let checksums = FileChecksums {
        runs_jsonl_sha256: runs_sha256,
        segments_jsonl_sha256: segments_sha256,
        events_jsonl_sha256: events_sha256,
    };

    let manifest = SyncManifest {
        schema_version: SCHEMA_VERSION.to_owned(),
        export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
        created_at_rfc3339: Utc::now().to_rfc3339(),
        source_db_path: db_path.display().to_string(),
        row_counts: RowCounts {
            runs: runs_count,
            segments: segments_count,
            events: events_count,
        },
        checksums,
    };

    let manifest_path = output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    atomic_write_bytes(&manifest_path, manifest_json.as_bytes())?;

    Ok(manifest)
}

fn export_table_runs(connection: &Connection, path: &Path) -> FwResult<(u64, String)> {
    let rows = connection
        .query(
            "SELECT id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json \
             FROM runs ORDER BY started_at ASC",
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;

    let file = fs::File::create(path)?;
    let mut writer = HashingWriter::new(BufWriter::new(file));
    let mut count = 0u64;

    for row in rows {
        let obj = serde_json::json!({
            "id": value_to_json(row.get(0)),
            "started_at": value_to_json(row.get(1)),
            "finished_at": value_to_json(row.get(2)),
            "backend": value_to_json(row.get(3)),
            "input_path": value_to_json(row.get(4)),
            "normalized_wav_path": value_to_json(row.get(5)),
            "request_json": value_to_json(row.get(6)),
            "result_json": value_to_json(row.get(7)),
            "warnings_json": value_to_json(row.get(8)),
            "transcript": value_to_json(row.get(9)),
            "replay_json": value_to_json(row.get(10)),
            "acceleration_json": value_to_json(row.get(11)),
        });
        writeln!(writer, "{}", serde_json::to_string(&obj)?)?;
        count += 1;
    }
    writer.flush()?;

    Ok((count, writer.finalize_hex()))
}

fn export_table_segments(connection: &Connection, path: &Path) -> FwResult<(u64, String)> {
    let rows = connection
        .query(
            "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence \
             FROM segments ORDER BY run_id ASC, idx ASC",
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;

    let file = fs::File::create(path)?;
    let mut writer = HashingWriter::new(BufWriter::new(file));
    let mut count = 0u64;

    for row in rows {
        let obj = serde_json::json!({
            "run_id": value_to_json(row.get(0)),
            "idx": value_to_json(row.get(1)),
            "start_sec": value_to_json(row.get(2)),
            "end_sec": value_to_json(row.get(3)),
            "speaker": value_to_json(row.get(4)),
            "text": value_to_json(row.get(5)),
            "confidence": value_to_json(row.get(6)),
        });
        writeln!(writer, "{}", serde_json::to_string(&obj)?)?;
        count += 1;
    }
    writer.flush()?;

    Ok((count, writer.finalize_hex()))
}

fn export_table_events(connection: &Connection, path: &Path) -> FwResult<(u64, String)> {
    let rows = connection
        .query(
            "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json \
             FROM events ORDER BY run_id ASC, seq ASC",
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;

    let file = fs::File::create(path)?;
    let mut writer = HashingWriter::new(BufWriter::new(file));
    let mut count = 0u64;

    for row in rows {
        let obj = serde_json::json!({
            "run_id": value_to_json(row.get(0)),
            "seq": value_to_json(row.get(1)),
            "ts_rfc3339": value_to_json(row.get(2)),
            "stage": value_to_json(row.get(3)),
            "code": value_to_json(row.get(4)),
            "message": value_to_json(row.get(5)),
            "payload_json": value_to_json(row.get(6)),
        });
        writeln!(writer, "{}", serde_json::to_string(&obj)?)?;
        count += 1;
    }
    writer.flush()?;

    Ok((count, writer.finalize_hex()))
}

// ---------------------------------------------------------------------------
// Incremental Export (changed records only)
// ---------------------------------------------------------------------------

/// Tracks the database mutation high-water mark of the last incremental
/// export so unchanged subsequent calls do not re-emit an aggregate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncCursor {
    /// Monotonic sequence assigned by SQLite triggers to the newest parent or
    /// child mutation observed by the completed export snapshot.
    #[serde(default)]
    pub last_mutation_seq: i64,
    /// Stable identity of the database lineage that assigned the mutation
    /// sequence. Cursors without an identity are legacy and force one safe
    /// full re-export; a different identity fails closed.
    #[serde(default)]
    pub database_id: Option<String>,
    /// Informational domain timestamp of the final exported run. Selection is
    /// never based on this caller-controlled value.
    pub last_export_rfc3339: String,
    /// Informational run ID paired with `last_export_rfc3339` for diagnostics
    /// and compatibility with cursor files created before schema v6.
    #[serde(default)]
    pub last_export_run_id: Option<String>,
    /// Number of runs exported in the last batch.
    pub last_run_count: u64,
}

/// Mode selector for the export function family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportMode {
    /// Export every record in the database.
    Full,
    /// Export only run aggregates whose database mutation sequence is greater
    /// than the cursor position, or all records when no cursor exists.
    Incremental,
}

/// Manifest for an incremental export.  It is a superset of [`SyncManifest`]
/// that additionally carries the cursor used to produce the snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalExportManifest {
    pub schema_version: String,
    pub export_format_version: String,
    pub export_mode: String,
    pub created_at_rfc3339: String,
    pub source_db_path: String,
    pub row_counts: RowCounts,
    pub checksums: FileChecksums,
    /// `None` when this is the first incremental export (no prior cursor).
    pub cursor_used: Option<SyncCursor>,
    /// Updated cursor that should be persisted and passed to the next call.
    pub cursor_after: SyncCursor,
}

const CURSOR_FILENAME: &str = "sync_cursor.json";
const EMPTY_INCREMENTAL_CURSOR_TS: &str = "";

/// Perform an incremental export using the database-owned mutation sequence.
/// When no cursor file exists inside `state_root`, all records are exported
/// (equivalent to a full export) and a new cursor file is written.
pub fn export_incremental(
    db_path: &Path,
    output_dir: &Path,
    state_root: &Path,
) -> FwResult<IncrementalExportManifest> {
    let _lock = SyncLock::acquire(state_root, "export_incremental")?;
    export_incremental_inner(db_path, output_dir, state_root)
}

fn export_incremental_inner(
    db_path: &Path,
    output_dir: &Path,
    state_root: &Path,
) -> FwResult<IncrementalExportManifest> {
    export_incremental_inner_with_after_runs(db_path, output_dir, state_root, || {})
}

fn export_incremental_inner_with_after_runs(
    db_path: &Path,
    output_dir: &Path,
    state_root: &Path,
    after_runs: impl FnOnce(),
) -> FwResult<IncrementalExportManifest> {
    fs::create_dir_all(output_dir)?;

    // Preserve export's fail-closed behavior for a missing/non-schema DB, then
    // let RunStore apply the current schema migration before opening the
    // dedicated read connection. Incremental export is a public library API
    // and cannot assume some earlier RunStore user already installed v6.
    let preflight = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    verify_schema_exists(&preflight)?;
    drop(preflight);
    drop(RunStore::open(db_path)?);

    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    verify_schema_exists(&connection)?;

    let cursor_path = state_root.join(CURSOR_FILENAME);
    let cursor_used = load_cursor(&cursor_path)?;

    let (runs_export, (segments_count, segments_sha256), (events_count, events_sha256)) =
        with_read_snapshot(&connection, || {
            // --- runs ---
            let runs_tmp = output_dir.join("runs.jsonl.tmp");
            let runs_final = output_dir.join("runs.jsonl");
            let runs_export = export_table_runs_incremental_snapshot(
                &connection,
                &runs_tmp,
                cursor_used.as_ref(),
            )?;
            atomic_rename(&runs_tmp, &runs_final)?;

            after_runs();

            // --- segments ---
            let segments_tmp = output_dir.join("segments.jsonl.tmp");
            let segments_final = output_dir.join("segments.jsonl");
            let segments =
                export_table_segments_for_runs(&connection, &segments_tmp, &runs_export.run_ids)?;
            atomic_rename(&segments_tmp, &segments_final)?;

            // --- events ---
            let events_tmp = output_dir.join("events.jsonl.tmp");
            let events_final = output_dir.join("events.jsonl");
            let events =
                export_table_events_for_runs(&connection, &events_tmp, &runs_export.run_ids)?;
            atomic_rename(&events_tmp, &events_final)?;

            Ok((runs_export, segments, events))
        })?;
    let runs_count = runs_export.count;
    let runs_sha256 = runs_export.sha256;

    // Checksums streamed while writing (HashingWriter) — no re-read pass. Same digest
    // as `sha256_file` of the written bytes.
    let checksums = FileChecksums {
        runs_jsonl_sha256: runs_sha256,
        segments_jsonl_sha256: segments_sha256,
        events_jsonl_sha256: events_sha256,
    };

    // The high-water mark is read inside the same snapshot as all three output
    // tables. It therefore advances past every mutation represented by the
    // published aggregate files, including mutations of child rows.
    let (new_cursor_ts, new_cursor_run_id) =
        match runs_export.last_position {
            Some((ts, run_id)) => (ts, Some(run_id)),
            None => cursor_used
                .as_ref()
                .map(|c| (c.last_export_rfc3339.clone(), c.last_export_run_id.clone()))
                .unwrap_or_else(|| (EMPTY_INCREMENTAL_CURSOR_TS.to_owned(), None)),
        };

    let cursor_after = SyncCursor {
        last_mutation_seq: runs_export.high_water_mark,
        database_id: Some(runs_export.database_id),
        last_export_rfc3339: new_cursor_ts,
        last_export_run_id: new_cursor_run_id,
        last_run_count: runs_count,
    };

    let manifest = IncrementalExportManifest {
        schema_version: SCHEMA_VERSION.to_owned(),
        export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
        export_mode: "incremental".to_owned(),
        created_at_rfc3339: Utc::now().to_rfc3339(),
        source_db_path: db_path.display().to_string(),
        row_counts: RowCounts {
            runs: runs_count,
            segments: segments_count,
            events: events_count,
        },
        checksums,
        cursor_used: cursor_used.clone(),
        cursor_after: cursor_after.clone(),
    };

    let manifest_path = output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    atomic_write_bytes(&manifest_path, manifest_json.as_bytes())?;

    // Advance the incremental cursor only after the export snapshot is fully
    // published. If manifest publication fails, retaining the old cursor is
    // safer than skipping runs that were never durably exported.
    save_cursor(&cursor_path, &cursor_after)?;

    Ok(manifest)
}

/// Load the cursor from `path`, returning `None` if the file does not exist.
fn load_cursor(path: &Path) -> FwResult<Option<SyncCursor>> {
    if !path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(path)?;
    let cursor: SyncCursor = serde_json::from_str(&contents)
        .map_err(|error| FwError::Storage(format!("invalid sync cursor: {error}")))?;
    if cursor.last_mutation_seq < 0 {
        return Err(FwError::Storage(
            "invalid sync cursor: last_mutation_seq must be non-negative".to_owned(),
        ));
    }
    Ok(Some(cursor))
}

/// Persist the cursor as a pretty-printed JSON file.
fn save_cursor(path: &Path, cursor: &SyncCursor) -> FwResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(cursor)?;
    atomic_write_bytes(path, json.as_bytes())?;
    Ok(())
}

struct IncrementalRunsSnapshot {
    count: u64,
    sha256: String,
    run_ids: Vec<String>,
    last_position: Option<(String, String)>,
    high_water_mark: i64,
    database_id: String,
}

struct MutationAuthority {
    database_id: String,
    high_water_mark: i64,
}

fn load_mutation_authority(connection: &Connection) -> FwResult<MutationAuthority> {
    let identity_rows = connection
        .query("SELECT value FROM _meta WHERE key = 'sync_database_id'")
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let database_id = identity_rows
        .first()
        .map(|row| value_to_string_sqlite(row.get(0)))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            FwError::Storage(
                "incremental export requires a non-empty sync_database_id".to_owned(),
            )
        })?;
    uuid::Uuid::parse_str(&database_id).map_err(|error| {
        FwError::Storage(format!(
            "invalid sync_database_id `{database_id}`: {error}"
        ))
    })?;

    let high_water_mark = connection
        .query("SELECT COALESCE(MAX(mutation_seq), 0) FROM sync_run_mutations")
        .map_err(|error| FwError::Storage(error.to_string()))?
        .first()
        .and_then(|row| row.get(0).and_then(|value| value.as_integer()))
        .ok_or_else(|| {
            FwError::Storage(
                "incremental export mutation high-water mark is not an integer".to_owned(),
            )
        })?;

    Ok(MutationAuthority {
        database_id,
        high_water_mark,
    })
}

fn validated_cursor_mutation_seq(
    cursor: Option<&SyncCursor>,
    authority: &MutationAuthority,
) -> FwResult<Option<i64>> {
    let Some(cursor) = cursor else {
        return Ok(None);
    };

    let Some(cursor_database_id) = cursor.database_id.as_deref() else {
        // Pre-identity cursors cannot safely interpret their database-local
        // numeric sequence. Sequence zero re-exports every current aggregate.
        return Ok(Some(0));
    };
    if cursor_database_id != authority.database_id {
        return Err(FwError::Storage(format!(
            "sync cursor database identity mismatch: cursor={cursor_database_id}, database={}",
            authority.database_id
        )));
    }
    if !(0..=authority.high_water_mark).contains(&cursor.last_mutation_seq) {
        return Err(FwError::Storage(format!(
            "invalid sync cursor: mutation sequence {} is outside database range 0..={}",
            cursor.last_mutation_seq, authority.high_water_mark
        )));
    }
    Ok(Some(cursor.last_mutation_seq))
}

fn export_table_runs_incremental_snapshot(
    connection: &Connection,
    path: &Path,
    cursor: Option<&SyncCursor>,
) -> FwResult<IncrementalRunsSnapshot> {
    let authority = load_mutation_authority(connection)?;
    let cursor_mutation_seq = validated_cursor_mutation_seq(cursor, &authority)?;

    let (sql, params) = match cursor_mutation_seq {
        Some(last_mutation_seq) => (
            "SELECT r.id, r.started_at, r.finished_at, r.backend, r.input_path, \
             r.normalized_wav_path, r.request_json, r.result_json, r.warnings_json, r.transcript, r.replay_json, r.acceleration_json \
             FROM runs AS r \
             JOIN sync_run_mutations AS m ON m.run_id = r.id \
             WHERE m.mutation_seq > ?1 \
             ORDER BY r.finished_at ASC, r.id ASC"
                .to_owned(),
            vec![SqliteValue::Integer(last_mutation_seq)],
        ),
        None => (
            "SELECT r.id, r.started_at, r.finished_at, r.backend, r.input_path, \
             r.normalized_wav_path, r.request_json, r.result_json, r.warnings_json, r.transcript, r.replay_json, r.acceleration_json \
             FROM runs AS r \
             JOIN sync_run_mutations AS m ON m.run_id = r.id \
             ORDER BY r.finished_at ASC, r.id ASC"
                .to_owned(),
            vec![],
        ),
    };

    let rows = if params.is_empty() {
        connection
            .query(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?
    } else {
        connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?
    };

    // BufWriter batches the per-row `writeln!` into ~8 KiB write() syscalls instead
    // of one syscall per JSONL line (the full-export writers above are already
    // buffered; these incremental ones were not). Byte-identical output.
    let mut file = HashingWriter::new(BufWriter::new(fs::File::create(path)?));
    let mut count = 0u64;
    let mut run_ids = Vec::with_capacity(rows.len());
    let mut last_position = None;

    for row in rows {
        let run_id = value_to_string_sqlite(row.get(0));
        let finished_at = value_to_string_sqlite(row.get(2));
        let obj = serde_json::json!({
            "id": value_to_json(row.get(0)),
            "started_at": value_to_json(row.get(1)),
            "finished_at": value_to_json(row.get(2)),
            "backend": value_to_json(row.get(3)),
            "input_path": value_to_json(row.get(4)),
            "normalized_wav_path": value_to_json(row.get(5)),
            "request_json": value_to_json(row.get(6)),
            "result_json": value_to_json(row.get(7)),
            "warnings_json": value_to_json(row.get(8)),
            "transcript": value_to_json(row.get(9)),
            "replay_json": value_to_json(row.get(10)),
            "acceleration_json": value_to_json(row.get(11)),
        });
        writeln!(file, "{}", serde_json::to_string(&obj)?)?;
        let replaces_last = last_position.as_ref().is_none_or(|(last_ts, last_id)| {
            finished_at > *last_ts || (finished_at == *last_ts && run_id > *last_id)
        });
        if replaces_last {
            last_position = Some((finished_at, run_id.clone()));
        }
        run_ids.push(run_id);
        count += 1;
    }
    file.flush()?;
    file.get_ref().get_ref().sync_all()?;

    Ok(IncrementalRunsSnapshot {
        count,
        sha256: file.finalize_hex(),
        run_ids,
        last_position,
        high_water_mark: authority.high_water_mark,
        database_id: authority.database_id,
    })
}

#[cfg(test)]
fn export_table_runs_incremental(
    connection: &Connection,
    path: &Path,
    cursor: Option<&SyncCursor>,
) -> FwResult<(u64, String)> {
    let snapshot = export_table_runs_incremental_snapshot(connection, path, cursor)?;
    Ok((snapshot.count, snapshot.sha256))
}

/// Collect all run IDs that match the incremental filter, so we can scope
/// segments and events to those runs.
#[cfg(test)]
fn collect_exported_run_ids(
    connection: &Connection,
    cursor: Option<&SyncCursor>,
) -> FwResult<Vec<String>> {
    let authority = load_mutation_authority(connection)?;
    let cursor_mutation_seq = validated_cursor_mutation_seq(cursor, &authority)?;
    let (sql, params) = match cursor_mutation_seq {
        Some(last_mutation_seq) => (
            "SELECT r.id FROM runs AS r \
             JOIN sync_run_mutations AS m ON m.run_id = r.id \
             WHERE m.mutation_seq > ?1 \
             ORDER BY r.finished_at ASC, r.id ASC"
                .to_owned(),
            vec![SqliteValue::Integer(last_mutation_seq)],
        ),
        None => (
            "SELECT r.id FROM runs AS r \
             JOIN sync_run_mutations AS m ON m.run_id = r.id \
             ORDER BY r.finished_at ASC, r.id ASC"
                .to_owned(),
            vec![],
        ),
    };

    let rows = if params.is_empty() {
        connection
            .query(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?
    } else {
        connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?
    };

    Ok(rows
        .iter()
        .map(|row| value_to_string_sqlite(row.get(0)))
        .collect())
}

/// Export segments belonging to a specific set of run_ids.
fn export_table_segments_for_runs(
    connection: &Connection,
    path: &Path,
    run_ids: &[String],
) -> FwResult<(u64, String)> {
    // BufWriter batches the per-row `writeln!` into ~8 KiB write() syscalls instead
    // of one syscall per JSONL line (the full-export writers above are already
    // buffered; these incremental ones were not). Byte-identical output.
    let mut file = HashingWriter::new(BufWriter::new(fs::File::create(path)?));
    // Batch the per-run N+1 `SELECT` into one `WHERE run_id IN (…)` query per chunk,
    // grouping rows by run_id and emitting in `run_ids` order (idx-ascending within
    // each run) ⇒ byte-identical JSONL to the legacy per-run path. Chunk size 1
    // (`FW_SYNC_BATCH_QUERY=0`) reproduces the legacy one-query-per-run behavior.
    let mut by_run: HashMap<String, Vec<String>> = HashMap::new();
    for chunk in run_ids.chunks(sync_query_batch_size()) {
        let placeholders = (1..=chunk.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence \
             FROM segments WHERE run_id IN ({placeholders}) ORDER BY run_id ASC, idx ASC"
        );
        let params: Vec<SqliteValue> = chunk
            .iter()
            .map(|run_id| SqliteValue::Text(run_id.clone().into()))
            .collect();
        let rows = connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?;

        for row in rows {
            let run_id = value_to_string_sqlite(row.get(0));
            let obj = serde_json::json!({
                "run_id": value_to_json(row.get(0)),
                "idx": value_to_json(row.get(1)),
                "start_sec": value_to_json(row.get(2)),
                "end_sec": value_to_json(row.get(3)),
                "speaker": value_to_json(row.get(4)),
                "text": value_to_json(row.get(5)),
                "confidence": value_to_json(row.get(6)),
            });
            by_run
                .entry(run_id)
                .or_default()
                .push(serde_json::to_string(&obj)?);
        }
    }

    let mut count = 0u64;
    for run_id in run_ids {
        if let Some(lines) = by_run.get(run_id) {
            for line in lines {
                writeln!(file, "{line}")?;
                count += 1;
            }
        }
    }
    file.flush()?;
    file.get_ref().get_ref().sync_all()?;

    Ok((count, file.finalize_hex()))
}

/// Export events belonging to a specific set of run_ids.
fn export_table_events_for_runs(
    connection: &Connection,
    path: &Path,
    run_ids: &[String],
) -> FwResult<(u64, String)> {
    // BufWriter batches the per-row `writeln!` into ~8 KiB write() syscalls instead
    // of one syscall per JSONL line (the full-export writers above are already
    // buffered; these incremental ones were not). Byte-identical output.
    let mut file = HashingWriter::new(BufWriter::new(fs::File::create(path)?));
    // Batch the per-run N+1 `SELECT` into one `WHERE run_id IN (…)` query per chunk;
    // group by run_id and emit in `run_ids` order (seq-ascending within each run) ⇒
    // byte-identical JSONL. `FW_SYNC_BATCH_QUERY=0` (chunk 1) = legacy per-run path.
    let mut by_run: HashMap<String, Vec<String>> = HashMap::new();
    for chunk in run_ids.chunks(sync_query_batch_size()) {
        let placeholders = (1..=chunk.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json \
             FROM events WHERE run_id IN ({placeholders}) ORDER BY run_id ASC, seq ASC"
        );
        let params: Vec<SqliteValue> = chunk
            .iter()
            .map(|run_id| SqliteValue::Text(run_id.clone().into()))
            .collect();
        let rows = connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?;

        for row in rows {
            let run_id = value_to_string_sqlite(row.get(0));
            let obj = serde_json::json!({
                "run_id": value_to_json(row.get(0)),
                "seq": value_to_json(row.get(1)),
                "ts_rfc3339": value_to_json(row.get(2)),
                "stage": value_to_json(row.get(3)),
                "code": value_to_json(row.get(4)),
                "message": value_to_json(row.get(5)),
                "payload_json": value_to_json(row.get(6)),
            });
            by_run
                .entry(run_id)
                .or_default()
                .push(serde_json::to_string(&obj)?);
        }
    }

    let mut count = 0u64;
    for run_id in run_ids {
        if let Some(lines) = by_run.get(run_id) {
            for line in lines {
                writeln!(file, "{line}")?;
                count += 1;
            }
        }
    }
    file.flush()?;
    file.get_ref().get_ref().sync_all()?;

    Ok((count, file.finalize_hex()))
}

/// Find the greatest domain tuple among aggregates mutated after `cursor`.
/// Selection uses only the database-owned sequence; the returned domain tuple
/// mirrors the informational position stored by the production exporter.
#[cfg(test)]
fn max_export_position(
    connection: &Connection,
    cursor: Option<&SyncCursor>,
) -> FwResult<Option<(String, String)>> {
    let authority = load_mutation_authority(connection)?;
    let cursor_mutation_seq = validated_cursor_mutation_seq(cursor, &authority)?;
    let (sql, params) = match cursor_mutation_seq {
        Some(last_mutation_seq) => (
            "SELECT r.finished_at, r.id FROM runs AS r \
             JOIN sync_run_mutations AS m ON m.run_id = r.id \
             WHERE m.mutation_seq > ?1 \
             ORDER BY r.finished_at DESC, r.id DESC LIMIT 1"
                .to_owned(),
            vec![SqliteValue::Integer(last_mutation_seq)],
        ),
        None => (
            "SELECT r.finished_at, r.id FROM runs AS r \
             JOIN sync_run_mutations AS m ON m.run_id = r.id \
             ORDER BY r.finished_at DESC, r.id DESC LIMIT 1"
                .to_owned(),
            vec![],
        ),
    };

    let rows = if params.is_empty() {
        connection
            .query(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?
    } else {
        connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?
    };

    let Some(row) = rows.first() else {
        return Ok(None);
    };

    let ts = value_to_string_sqlite(row.get(0));
    let run_id = value_to_string_sqlite(row.get(1));
    if ts.is_empty() || run_id.is_empty() {
        Ok(None)
    } else {
        Ok(Some((ts, run_id)))
    }
}

// ---------------------------------------------------------------------------
// Import (jsonl -> db)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ConflictPolicy {
    Reject,
    Overwrite,
    OverwriteStrict,
    Skip,
}

impl ConflictPolicy {
    const fn allows_child_row_mutation(self) -> bool {
        matches!(self, Self::OverwriteStrict)
    }
}

#[derive(Debug)]
pub struct ImportResult {
    pub runs_imported: u64,
    pub segments_imported: u64,
    pub events_imported: u64,
    pub conflicts: Vec<SyncConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    pub table: String,
    pub key: String,
    pub reason: String,
}

#[derive(Debug)]
struct JsonlInputPaths {
    runs: PathBuf,
    segments: PathBuf,
    events: PathBuf,
}

impl JsonlInputPaths {
    fn resolve(input_dir: &Path) -> Self {
        Self {
            runs: resolve_jsonl_path(input_dir, "runs"),
            segments: resolve_jsonl_path(input_dir, "segments"),
            events: resolve_jsonl_path(input_dir, "events"),
        }
    }
}

pub fn import(
    db_path: &Path,
    input_dir: &Path,
    state_root: &Path,
    conflict_policy: ConflictPolicy,
) -> FwResult<ImportResult> {
    let _lock = SyncLock::acquire(state_root, "import")?;
    import_inner(db_path, input_dir, conflict_policy)
}

fn import_inner(
    db_path: &Path,
    input_dir: &Path,
    conflict_policy: ConflictPolicy,
) -> FwResult<ImportResult> {
    import_inner_with_batch_mode(
        db_path,
        input_dir,
        conflict_policy,
        sync_batch_import_enabled(),
    )
}

fn import_inner_with_batch_mode(
    db_path: &Path,
    input_dir: &Path,
    conflict_policy: ConflictPolicy,
    batch_import_enabled: bool,
) -> FwResult<ImportResult> {
    // Validate manifest
    let manifest_path = input_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(FwError::Storage(format!(
            "manifest.json not found in {}",
            input_dir.display()
        )));
    }
    let manifest_text = fs::read_to_string(&manifest_path)?;
    let manifest: SyncManifest = serde_json::from_str(&manifest_text)
        .map_err(|error| FwError::Storage(format!("invalid manifest: {error}")))?;

    // Validate schema version (exact major match)
    validate_schema_version(&manifest)?;
    // Validate export format version (exact major match)
    validate_export_format_version(&manifest)?;

    // Resolve the physical inputs once. Checksum validation and row import must
    // consume the same plain-or-gzip files even if the directory changes later.
    let input_paths = validate_checksums(&manifest, input_dir)?;

    // Open DB and ensure schema exists
    if let Some(parent) = db_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let store = RunStore::open(db_path)?;
    let connection = store.connection();
    ensure_schema(connection)?;

    // Begin transaction
    connection
        .execute("BEGIN;")
        .map_err(|error| FwError::Storage(error.to_string()))?;

    let result = import_tables(
        connection,
        &input_paths,
        &manifest,
        conflict_policy,
        batch_import_enabled,
    );

    match result {
        Ok(import_result) => {
            let finalize = (|| -> FwResult<()> {
                if !import_result.conflicts.is_empty() {
                    write_conflicts_file(input_dir, &import_result.conflicts)?;
                    if conflict_policy == ConflictPolicy::Reject {
                        return Err(FwError::Storage(format!(
                            "import rejected due to {} conflict(s); see sync_conflicts.jsonl",
                            import_result.conflicts.len()
                        )));
                    }
                }

                // The normalized diarization tables are a derived cache, but
                // they must advance atomically with their canonical run JSON.
                // Rebuild inside this import transaction so malformed typed
                // payloads cannot commit only the canonical half.
                store.rebuild_diarization_index()?;
                connection
                    .execute("COMMIT;")
                    .map_err(|error| FwError::Storage(error.to_string()))?;
                Ok(())
            })();
            match finalize {
                Ok(()) => Ok(import_result),
                Err(error) => {
                    let _ = connection.execute("ROLLBACK;");
                    Err(error)
                }
            }
        }
        Err(error) => {
            let _ = connection.execute("ROLLBACK;");
            Err(error)
        }
    }
}

fn import_tables(
    connection: &Connection,
    input_paths: &JsonlInputPaths,
    manifest: &SyncManifest,
    conflict_policy: ConflictPolicy,
    batch_import_enabled: bool,
) -> FwResult<ImportResult> {
    let mut conflicts = Vec::new();
    let mut run_tracking = RunImportTracking::default();

    // Import runs first (referential parent)
    let runs_imported = import_runs(
        connection,
        &input_paths.runs,
        conflict_policy,
        &mut conflicts,
        &mut run_tracking,
        batch_import_enabled,
    )
    .map_err(|error| FwError::Storage(format!("runs import failed: {error}")))?;

    // Validate row count
    if runs_imported != manifest.row_counts.runs && conflicts.is_empty() {
        return Err(FwError::Storage(format!(
            "row count mismatch for runs: expected {}, processed {}",
            manifest.row_counts.runs, runs_imported
        )));
    }

    // Import segments
    let segments_imported = import_segments(
        connection,
        &input_paths.segments,
        conflict_policy,
        &mut conflicts,
        &run_tracking.imported_run_ids,
        &run_tracking.overwritten_run_ids,
        &run_tracking.inserted_run_ids,
        &run_tracking.overwritten_segment_idxs_before,
        batch_import_enabled,
    )
    .map_err(|error| FwError::Storage(format!("segments import failed: {error}")))?;

    if segments_imported != manifest.row_counts.segments && conflicts.is_empty() {
        return Err(FwError::Storage(format!(
            "row count mismatch for segments: expected {}, processed {}",
            manifest.row_counts.segments, segments_imported
        )));
    }

    // Import events
    let events_imported = import_events(
        connection,
        &input_paths.events,
        conflict_policy,
        &mut conflicts,
        &run_tracking.imported_run_ids,
        &run_tracking.overwritten_run_ids,
        &run_tracking.inserted_run_ids,
        &run_tracking.overwritten_event_seqs_before,
        batch_import_enabled,
    )
    .map_err(|error| FwError::Storage(format!("events import failed: {error}")))?;

    if events_imported != manifest.row_counts.events && conflicts.is_empty() {
        return Err(FwError::Storage(format!(
            "row count mismatch for events: expected {}, processed {}",
            manifest.row_counts.events, events_imported
        )));
    }

    Ok(ImportResult {
        runs_imported,
        segments_imported,
        events_imported,
        conflicts,
    })
}

#[derive(Default)]
struct RunImportTracking {
    imported_run_ids: HashSet<String>,
    inserted_run_ids: HashSet<String>,
    overwritten_run_ids: HashSet<String>,
    overwritten_segment_idxs_before: HashMap<String, HashSet<i64>>,
    overwritten_event_seqs_before: HashMap<String, HashSet<i64>>,
}

fn import_runs(
    connection: &Connection,
    path: &Path,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
    tracking: &mut RunImportTracking,
    batch_import_enabled: bool,
) -> FwResult<u64> {
    let reader = open_jsonl_reader(path)?;
    let mut count = 0u64;

    if batch_import_enabled {
        // Batched: pre-fetch the existing rows for a chunk of ids with one
        // `WHERE id IN (…)` query, then run the SAME `apply_run_row` conflict logic
        // per line ⇒ byte-identical to the per-line path. Chunked to bound memory.
        let chunk_size = sync_query_batch_size().max(1);
        let mut chunk: Vec<(String, serde_json::Value)> = Vec::with_capacity(chunk_size);
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: serde_json::Value = serde_json::from_str(line)?;
            let id = json_str(&row, "id")?;
            let backend = json_str(&row, "backend")?;
            let result_json = json_str(&row, "result_json")?;
            crate::storage::validate_run_backend_json(&id, &backend, &result_json)?;
            chunk.push((id, row));
            if chunk.len() >= chunk_size {
                count +=
                    flush_run_chunk(connection, &mut chunk, conflict_policy, conflicts, tracking)?;
            }
        }
        count += flush_run_chunk(connection, &mut chunk, conflict_policy, conflicts, tracking)?;
    } else {
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: serde_json::Value = serde_json::from_str(line)?;
            let id = json_str(&row, "id")?;
            let backend = json_str(&row, "backend")?;
            let result_json = json_str(&row, "result_json")?;
            crate::storage::validate_run_backend_json(&id, &backend, &result_json)?;

            // Per-line existing lookup (the N+1 the batched path collapses).
            let existing_rows = connection
                .query_with_params(
                    "SELECT id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json FROM runs WHERE id = ?1",
                    &[SqliteValue::Text(id.clone().into())],
                )
                .map_err(|error| {
                    FwError::Storage(format!("query runs existing `{id}` failed: {error}"))
                })?;
            let existing = existing_rows.first().map(run_row_to_cols);
            apply_run_row(
                connection,
                &id,
                &row,
                existing.as_deref(),
                conflict_policy,
                conflicts,
                tracking,
            )?;
            count += 1;
        }
    }

    Ok(count)
}

/// Convert one `runs` JSONL row into the database column order shared by INSERT,
/// UPDATE, and the batched import seen-map.
fn run_cols_from_json(id: &str, row: &serde_json::Value) -> FwResult<Vec<SqliteValue>> {
    Ok(vec![
        SqliteValue::Text(id.to_owned().into()),
        SqliteValue::Text(json_str(row, "started_at")?.into()),
        SqliteValue::Text(json_str(row, "finished_at")?.into()),
        SqliteValue::Text(json_str(row, "backend")?.into()),
        SqliteValue::Text(json_str(row, "input_path")?.into()),
        SqliteValue::Text(json_str(row, "normalized_wav_path")?.into()),
        SqliteValue::Text(json_str(row, "request_json")?.into()),
        SqliteValue::Text(json_str(row, "result_json")?.into()),
        SqliteValue::Text(json_str(row, "warnings_json")?.into()),
        SqliteValue::Text(json_str(row, "transcript")?.into()),
        SqliteValue::Text(json_string_or_default(row, "replay_json", "{}").into()),
        SqliteValue::Text(json_string_or_default(row, "acceleration_json", "{}").into()),
    ])
}

/// Process one `runs` JSONL row against `existing` (the current DB/seen row, or
/// `None`), applying the identical-compare + [`ConflictPolicy`]. **Shared by the
/// per-line and batched import paths** so their imported rows are byte-identical —
/// the only difference is where `existing` comes from (a per-line `SELECT` vs a
/// prefetched `WHERE id IN (…)` map). Returns `Some(written_cols)` when a row was
/// INSERTed or updated (so the batched caller can update its intra-chunk seen-map),
/// else `None`.
fn apply_run_row(
    connection: &Connection,
    id: &str,
    row: &serde_json::Value,
    existing: Option<&[SqliteValue]>,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
    tracking: &mut RunImportTracking,
) -> FwResult<Option<Vec<SqliteValue>>> {
    tracking.imported_run_ids.insert(id.to_owned());

    if let Some(existing) = existing {
        // Compare payload: if identical, noop; if different, apply conflict policy.
        let identical = value_to_string_sqlite(existing.get(1)) == json_str(row, "started_at")?
            && value_to_string_sqlite(existing.get(2)) == json_str(row, "finished_at")?
            && value_to_string_sqlite(existing.get(3)) == json_str(row, "backend")?
            && value_to_string_sqlite(existing.get(4)) == json_str(row, "input_path")?
            && value_to_string_sqlite(existing.get(5)) == json_str(row, "normalized_wav_path")?
            && value_to_string_sqlite(existing.get(6)) == json_str(row, "request_json")?
            && value_to_string_sqlite(existing.get(7)) == json_str(row, "result_json")?
            && value_to_string_sqlite(existing.get(8)) == json_str(row, "warnings_json")?
            && value_to_string_sqlite(existing.get(9)) == json_str(row, "transcript")?
            && value_to_string_sqlite(existing.get(10))
                == json_string_or_default(row, "replay_json", "{}")
            && value_to_string_sqlite(existing.get(11))
                == json_string_or_default(row, "acceleration_json", "{}");

        if identical {
            return Ok(None); // identical — noop
        }

        match conflict_policy {
            ConflictPolicy::Reject => {
                conflicts.push(SyncConflict {
                    table: "runs".to_owned(),
                    key: id.to_owned(),
                    reason: "different payload for same id".to_owned(),
                });
                return Ok(None);
            }
            ConflictPolicy::Skip => {
                return Ok(None);
            }
            ConflictPolicy::Overwrite | ConflictPolicy::OverwriteStrict => {
                let existing_segment_idxs =
                    query_segment_idxs_for_run(connection, id).map_err(|error| {
                        FwError::Storage(format!(
                            "overwrite capture segments `{id}` failed: {error}"
                        ))
                    })?;
                tracking
                    .overwritten_segment_idxs_before
                    .entry(id.to_owned())
                    .or_default()
                    .extend(existing_segment_idxs);

                let existing_event_seqs =
                    query_event_seqs_for_run(connection, id).map_err(|error| {
                        FwError::Storage(format!("overwrite capture events `{id}` failed: {error}"))
                    })?;
                tracking
                    .overwritten_event_seqs_before
                    .entry(id.to_owned())
                    .or_default()
                    .extend(existing_event_seqs);
                tracking.overwritten_run_ids.insert(id.to_owned());

                if conflict_policy == ConflictPolicy::Overwrite {
                    // Preserve child rows as the comparison authority for the
                    // upcoming segment/event imports. The enclosing transaction
                    // rolls this parent update back if either child payload differs.
                    let cols = run_cols_from_json(id, row)?;
                    connection
                        .import_exec(
                            "UPDATE runs SET started_at = ?2, finished_at = ?3, backend = ?4, \
                             input_path = ?5, normalized_wav_path = ?6, request_json = ?7, \
                             result_json = ?8, warnings_json = ?9, transcript = ?10, \
                             replay_json = ?11, acceleration_json = ?12 WHERE id = ?1",
                            &cols,
                        )
                        .map_err(|error| {
                            FwError::Storage(format!(
                                "overwrite update runs `{id}` failed: {error}"
                            ))
                        })?;
                    return Ok(Some(cols));
                }

                connection
                    .import_exec(
                        "DELETE FROM segments WHERE run_id = ?1",
                        &[SqliteValue::Text(id.to_owned().into())],
                    )
                    .map_err(|error| {
                        FwError::Storage(format!(
                            "overwrite delete segments `{id}` failed: {error}"
                        ))
                    })?;
                connection
                    .import_exec(
                        "DELETE FROM events WHERE run_id = ?1",
                        &[SqliteValue::Text(id.to_owned().into())],
                    )
                    .map_err(|error| {
                        FwError::Storage(format!("overwrite delete events `{id}` failed: {error}"))
                    })?;
                connection
                    .import_exec(
                        "DELETE FROM runs WHERE id = ?1",
                        &[SqliteValue::Text(id.to_owned().into())],
                    )
                    .map_err(|error| {
                        FwError::Storage(format!("overwrite delete runs `{id}` failed: {error}"))
                    })?;
                // Fall through to INSERT.
            }
        }
    }

    let cols = run_cols_from_json(id, row)?;
    connection
        .import_exec(
            "INSERT INTO runs (id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            &cols,
        )
        .map_err(|error| FwError::Storage(format!("insert runs `{id}` failed: {error}")))?;
    tracking.inserted_run_ids.insert(id.to_owned());

    Ok(Some(cols))
}

/// Batched-import worker: one `WHERE id IN (…)` prefetch for the whole `chunk`,
/// then [`apply_run_row`] per line in original order. An intra-chunk **seen-map**
/// (the prefetched map, updated on every INSERT) reproduces the per-line invariant
/// that a duplicate id later in the file sees the earlier line's insert. Drains
/// `chunk`. Returns the number of lines processed (== `chunk` len on entry).
fn flush_run_chunk(
    connection: &Connection,
    chunk: &mut Vec<(String, serde_json::Value)>,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
    tracking: &mut RunImportTracking,
) -> FwResult<u64> {
    if chunk.is_empty() {
        return Ok(0);
    }

    // Distinct ids for the IN clause (a duplicate id in the file → one bind).
    let mut seen = HashSet::new();
    let unique_ids: Vec<String> = chunk
        .iter()
        .map(|(id, _)| id.clone())
        .filter(|id| seen.insert(id.clone()))
        .collect();
    let placeholders = (1..=unique_ids.len())
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json FROM runs WHERE id IN ({placeholders})"
    );
    let params: Vec<SqliteValue> = unique_ids
        .iter()
        .map(|id| SqliteValue::Text(id.clone().into()))
        .collect();
    let rows = connection
        .query_with_params(&sql, &params)
        .map_err(|error| FwError::Storage(format!("batch query runs existing failed: {error}")))?;

    let mut existing_map: HashMap<String, Vec<SqliteValue>> = HashMap::with_capacity(rows.len());
    for r in &rows {
        let cols = run_row_to_cols(r);
        existing_map.insert(value_to_string_sqlite(cols.first()), cols);
    }

    let mut count = 0u64;
    for (id, row) in chunk.drain(..) {
        let existing = existing_map.get(&id).cloned();
        if let Some(inserted) = apply_run_row(
            connection,
            &id,
            &row,
            existing.as_deref(),
            conflict_policy,
            conflicts,
            tracking,
        )? {
            existing_map.insert(id, inserted); // seen-map: later dup sees this write
        }
        count += 1;
    }

    Ok(count)
}

/// The 7 `segments` columns in `SELECT`/`INSERT` order — the shared "existing row"
/// representation for [`apply_segment_row`] (see [`run_row_to_cols`]).
fn segment_row_to_cols(row: &fsqlite::Row) -> Vec<SqliteValue> {
    (0..7)
        .map(|i| row.get(i).cloned().unwrap_or(SqliteValue::Null))
        .collect()
}

/// Per-line bookkeeping for `import_segments` that is independent of where the
/// "existing row" comes from — shared verbatim by the legacy and batched paths.
#[derive(Default)]
struct SegmentImportState {
    imported_idx_by_run: HashMap<String, HashSet<i64>>,
    imported_idx_by_overwritten_run: HashMap<String, HashSet<i64>>,
    known_run_ids: HashSet<String>,
}

/// The per-line FK check + overwrite/imported idx tracking (runs before the
/// existing-row lookup in both paths, so batching cannot change it).
#[allow(
    clippy::too_many_arguments,
    reason = "row import bookkeeping keeps conflict authorities explicit and borrowed"
)]
fn record_segment_pre(
    connection: &Connection,
    run_id: &str,
    idx: i64,
    key: &str,
    conflict_policy: ConflictPolicy,
    imported_run_ids: &HashSet<String>,
    overwritten_run_ids: &HashSet<String>,
    state: &mut SegmentImportState,
) -> FwResult<()> {
    ensure_run_reference_exists(
        connection,
        run_id,
        "segments",
        key,
        &mut state.known_run_ids,
    )?;
    if conflict_policy == ConflictPolicy::Overwrite && overwritten_run_ids.contains(run_id) {
        state
            .imported_idx_by_overwritten_run
            .entry(run_id.to_owned())
            .or_default()
            .insert(idx);
    }
    if conflict_policy.allows_child_row_mutation() && imported_run_ids.contains(run_id) {
        state
            .imported_idx_by_run
            .entry(run_id.to_owned())
            .or_default()
            .insert(idx);
    }
    Ok(())
}

/// Validate a child row that belongs to a target run which existed before this
/// import. `Skip` is a whole-aggregate policy: accepted rows still count toward
/// manifest reconciliation, but none may query, compare, or mutate child state.
fn skip_preexisting_child_row(
    run_id: &str,
    row: &serde_json::Value,
    conflict_policy: ConflictPolicy,
    inserted_run_ids: &HashSet<String>,
    required_fields: &[&str],
) -> FwResult<bool> {
    if conflict_policy != ConflictPolicy::Skip || inserted_run_ids.contains(run_id) {
        return Ok(false);
    }
    for field in required_fields {
        json_str(row, field)?;
    }
    Ok(true)
}

/// Apply one `segments` JSONL row against `existing` (current DB/seen row, or
/// `None`). **Shared by the per-line and batched paths** ⇒ byte-identical imported
/// rows. Returns `Some(inserted_cols)` when a row was INSERTed (so the batched
/// caller can update its seen-map), else `None`.
#[allow(
    clippy::too_many_arguments,
    reason = "row application keeps database, policy, and conflict sink explicit"
)]
fn apply_segment_row(
    connection: &Connection,
    run_id: &str,
    idx: i64,
    key: &str,
    row: &serde_json::Value,
    existing: Option<&[SqliteValue]>,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
) -> FwResult<Option<Vec<SqliteValue>>> {
    if let Some(existing) = existing {
        let identical = optional_floats_equal(
            sqlite_to_optional_f64(existing.get(2)),
            json_to_optional_f64(row, "start_sec"),
        ) && optional_floats_equal(
            sqlite_to_optional_f64(existing.get(3)),
            json_to_optional_f64(row, "end_sec"),
        ) && sqlite_to_optional_text(existing.get(4))
            == json_to_optional_text(row, "speaker")
            && value_to_string_sqlite(existing.get(5)) == json_str(row, "text")?
            && optional_floats_equal(
                sqlite_to_optional_f64(existing.get(6)),
                json_to_optional_f64(row, "confidence"),
            );

        if identical {
            return Ok(None);
        }

        match conflict_policy {
            ConflictPolicy::Reject => {
                conflicts.push(SyncConflict {
                    table: "segments".to_owned(),
                    key: key.to_owned(),
                    reason: "duplicate composite key".to_owned(),
                });
                return Ok(None);
            }
            ConflictPolicy::Skip => {
                return Ok(None);
            }
            ConflictPolicy::Overwrite => {
                return Err(FwError::Storage(format!(
                    "overwrite would require updating conflicting segment row `{run_id}/{idx}`, \
                     but child-row UPDATE is unsupported in this runtime; \
                     re-import into an empty target DB for strict replacement"
                )));
            }
            ConflictPolicy::OverwriteStrict => {
                connection
                    .import_exec(
                        "DELETE FROM segments WHERE run_id = ?1 AND idx = ?2",
                        &[SqliteValue::Text(run_id.to_owned().into()), SqliteValue::Integer(idx)],
                    )
                    .map_err(|error| {
                        FwError::Storage(format!(
                            "strict overwrite delete conflicting segment `{run_id}/{idx}` failed: {error}"
                        ))
                    })?;
            }
        }
    }

    let cols = vec![
        SqliteValue::Text(run_id.to_owned().into()),
        SqliteValue::Integer(idx),
        json_optional_float(row, "start_sec"),
        json_optional_float(row, "end_sec"),
        json_optional_text(row, "speaker"),
        SqliteValue::Text(json_str(row, "text")?.into()),
        json_optional_float(row, "confidence"),
    ];
    connection
        .import_exec(
            "INSERT INTO segments (run_id, idx, start_sec, end_sec, speaker, text, confidence) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            &cols,
        )
        .map_err(|error| {
            FwError::Storage(format!("insert segments `{run_id}/{idx}` failed: {error}"))
        })?;

    Ok(Some(cols))
}

/// Batched-import worker for `segments`: one `WHERE run_id IN (…)` prefetch for the
/// whole chunk (composite `(run_id, idx)` keyed map — fsqlite has no row-value `IN`),
/// then [`apply_segment_row`] per line in order. The prefetched map doubles as the
/// intra-chunk seen-map (updated on every INSERT), reproducing the per-line invariant
/// that a duplicate `(run_id, idx)` later in the file sees the earlier line's insert.
/// Drains `chunk`. Returns the number of lines processed.
#[allow(
    clippy::too_many_arguments,
    reason = "segment batching keeps parent-policy sets explicit and borrowed"
)]
fn flush_segment_chunk(
    connection: &Connection,
    chunk: &mut Vec<(String, i64, serde_json::Value)>,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
    state: &mut SegmentImportState,
    imported_run_ids: &HashSet<String>,
    overwritten_run_ids: &HashSet<String>,
    inserted_run_ids: &HashSet<String>,
) -> FwResult<u64> {
    if chunk.is_empty() {
        return Ok(0);
    }

    let mut seen = HashSet::new();
    let unique_run_ids: Vec<String> = chunk
        .iter()
        .map(|(run_id, _, _)| run_id.clone())
        .filter(|run_id| {
            conflict_policy != ConflictPolicy::Skip || inserted_run_ids.contains(run_id)
        })
        .filter(|run_id| seen.insert(run_id.clone()))
        .collect();
    let mut existing_map: HashMap<(String, i64), Vec<SqliteValue>> = HashMap::new();
    if !unique_run_ids.is_empty() {
        let placeholders = (1..=unique_run_ids.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence FROM segments WHERE run_id IN ({placeholders})"
        );
        let params: Vec<SqliteValue> = unique_run_ids
            .iter()
            .map(|run_id| SqliteValue::Text(run_id.clone().into()))
            .collect();
        let rows = connection
            .query_with_params(&sql, &params)
            .map_err(|error| {
                FwError::Storage(format!("batch query segments existing failed: {error}"))
            })?;

        for row in &rows {
            let cols = segment_row_to_cols(row);
            let run_id = value_to_string_sqlite(cols.first());
            if let Some(SqliteValue::Integer(idx)) = cols.get(1) {
                existing_map.insert((run_id, *idx), cols);
            }
        }
    }

    let mut count = 0u64;
    for (run_id, idx, row) in chunk.drain(..) {
        let key = format!("{run_id}/{idx}");
        record_segment_pre(
            connection,
            &run_id,
            idx,
            &key,
            conflict_policy,
            imported_run_ids,
            overwritten_run_ids,
            state,
        )?;
        if skip_preexisting_child_row(
            &run_id,
            &row,
            conflict_policy,
            inserted_run_ids,
            &["text"],
        )? {
            count += 1;
            continue;
        }
        let existing = existing_map.get(&(run_id.clone(), idx)).cloned();
        if let Some(cols) = apply_segment_row(
            connection,
            &run_id,
            idx,
            &key,
            &row,
            existing.as_deref(),
            conflict_policy,
            conflicts,
        )? {
            existing_map.insert((run_id, idx), cols); // seen-map for intra-chunk dup
        }
        count += 1;
    }

    Ok(count)
}

#[allow(
    clippy::too_many_arguments,
    reason = "segment import keeps parent-policy sets explicit and borrowed"
)]
fn import_segments(
    connection: &Connection,
    path: &Path,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
    imported_run_ids: &HashSet<String>,
    overwritten_run_ids: &HashSet<String>,
    inserted_run_ids: &HashSet<String>,
    overwritten_segment_idxs_before: &HashMap<String, HashSet<i64>>,
    batch_import_enabled: bool,
) -> FwResult<u64> {
    let reader = open_jsonl_reader(path)?;
    let mut count = 0u64;
    let mut state = SegmentImportState::default();

    if batch_import_enabled {
        let chunk_size = sync_query_batch_size().max(1);
        let mut chunk: Vec<(String, i64, serde_json::Value)> = Vec::with_capacity(chunk_size);
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: serde_json::Value = serde_json::from_str(line)?;
            let run_id = json_str(&row, "run_id")?;
            let idx = row
                .get("idx")
                .and_then(|value| value.as_i64())
                .ok_or_else(|| FwError::Storage("missing idx in segments row".to_owned()))?;
            chunk.push((run_id, idx, row));
            if chunk.len() >= chunk_size {
                count += flush_segment_chunk(
                    connection,
                    &mut chunk,
                    conflict_policy,
                    conflicts,
                    &mut state,
                    imported_run_ids,
                    overwritten_run_ids,
                    inserted_run_ids,
                )?;
            }
        }
        count += flush_segment_chunk(
            connection,
            &mut chunk,
            conflict_policy,
            conflicts,
            &mut state,
            imported_run_ids,
            overwritten_run_ids,
            inserted_run_ids,
        )?;
    } else {
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let row: serde_json::Value = serde_json::from_str(line)?;
            let run_id = json_str(&row, "run_id")?;
            let idx = row
                .get("idx")
                .and_then(|value| value.as_i64())
                .ok_or_else(|| FwError::Storage("missing idx in segments row".to_owned()))?;

            let key = format!("{run_id}/{idx}");
            record_segment_pre(
                connection,
                &run_id,
                idx,
                &key,
                conflict_policy,
                imported_run_ids,
                overwritten_run_ids,
                &mut state,
            )?;
            if skip_preexisting_child_row(
                &run_id,
                &row,
                conflict_policy,
                inserted_run_ids,
                &["text"],
            )? {
                count += 1;
                continue;
            }

            let existing_rows = connection
                .query_with_params(
                    "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence FROM segments WHERE run_id = ?1 AND idx = ?2",
                    &[SqliteValue::Text(run_id.clone().into()), SqliteValue::Integer(idx)],
                )
                .map_err(|error| {
                    FwError::Storage(format!(
                        "query segments existing `{run_id}/{idx}` failed: {error}"
                    ))
                })?;
            let existing = existing_rows.first().map(segment_row_to_cols);
            apply_segment_row(
                connection,
                &run_id,
                idx,
                &key,
                &row,
                existing.as_deref(),
                conflict_policy,
                conflicts,
            )?;
            count += 1;
        }
    }

    if conflict_policy == ConflictPolicy::Overwrite && !overwritten_run_ids.is_empty() {
        assert_no_stale_segments_for_overwritten_runs(
            overwritten_run_ids,
            overwritten_segment_idxs_before,
            &state.imported_idx_by_overwritten_run,
        )?;
    }
    if conflict_policy == ConflictPolicy::OverwriteStrict && !imported_run_ids.is_empty() {
        delete_stale_segments_for_strict_overwrite(
            connection,
            imported_run_ids,
            &state.imported_idx_by_run,
        )?;
    }

    Ok(count)
}

/// The 7 `events` columns in `SELECT`/`INSERT` order (see [`segment_row_to_cols`]).
fn event_row_to_cols(row: &fsqlite::Row) -> Vec<SqliteValue> {
    (0..7)
        .map(|i| row.get(i).cloned().unwrap_or(SqliteValue::Null))
        .collect()
}

/// Per-line bookkeeping for `import_events` (mirror of [`SegmentImportState`]).
#[derive(Default)]
struct EventImportState {
    imported_seq_by_run: HashMap<String, HashSet<i64>>,
    imported_seq_by_overwritten_run: HashMap<String, HashSet<i64>>,
    known_run_ids: HashSet<String>,
}

/// The per-line FK check + overwrite/imported seq tracking (mirror of
/// [`record_segment_pre`]).
#[allow(
    clippy::too_many_arguments,
    reason = "row import bookkeeping keeps conflict authorities explicit and borrowed"
)]
fn record_event_pre(
    connection: &Connection,
    run_id: &str,
    seq: i64,
    key: &str,
    conflict_policy: ConflictPolicy,
    imported_run_ids: &HashSet<String>,
    overwritten_run_ids: &HashSet<String>,
    state: &mut EventImportState,
) -> FwResult<()> {
    ensure_run_reference_exists(connection, run_id, "events", key, &mut state.known_run_ids)?;
    if conflict_policy == ConflictPolicy::Overwrite && overwritten_run_ids.contains(run_id) {
        state
            .imported_seq_by_overwritten_run
            .entry(run_id.to_owned())
            .or_default()
            .insert(seq);
    }
    if conflict_policy.allows_child_row_mutation() && imported_run_ids.contains(run_id) {
        state
            .imported_seq_by_run
            .entry(run_id.to_owned())
            .or_default()
            .insert(seq);
    }
    Ok(())
}

/// Apply one `events` JSONL row against `existing` (mirror of [`apply_segment_row`];
/// the compare is all-TEXT). Returns `Some(inserted_cols)` when INSERTed, else `None`.
#[allow(
    clippy::too_many_arguments,
    reason = "row application keeps database, policy, and conflict sink explicit"
)]
fn apply_event_row(
    connection: &Connection,
    run_id: &str,
    seq: i64,
    key: &str,
    row: &serde_json::Value,
    existing: Option<&[SqliteValue]>,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
) -> FwResult<Option<Vec<SqliteValue>>> {
    if let Some(existing) = existing {
        let identical = value_to_string_sqlite(existing.get(2)) == json_str(row, "ts_rfc3339")?
            && value_to_string_sqlite(existing.get(3)) == json_str(row, "stage")?
            && value_to_string_sqlite(existing.get(4)) == json_str(row, "code")?
            && value_to_string_sqlite(existing.get(5)) == json_str(row, "message")?
            && value_to_string_sqlite(existing.get(6)) == json_str(row, "payload_json")?;
        if identical {
            return Ok(None);
        }

        match conflict_policy {
            ConflictPolicy::Reject => {
                conflicts.push(SyncConflict {
                    table: "events".to_owned(),
                    key: key.to_owned(),
                    reason: "duplicate composite key".to_owned(),
                });
                return Ok(None);
            }
            ConflictPolicy::Skip => {
                return Ok(None);
            }
            ConflictPolicy::Overwrite => {
                return Err(FwError::Storage(format!(
                    "overwrite would require updating conflicting event row `{run_id}/{seq}`, \
                     but child-row UPDATE is unsupported in this runtime; \
                     re-import into an empty target DB for strict replacement"
                )));
            }
            ConflictPolicy::OverwriteStrict => {
                connection
                    .import_exec(
                        "DELETE FROM events WHERE run_id = ?1 AND seq = ?2",
                        &[SqliteValue::Text(run_id.to_owned().into()), SqliteValue::Integer(seq)],
                    )
                    .map_err(|error| {
                        FwError::Storage(format!(
                            "strict overwrite delete conflicting event `{run_id}/{seq}` failed: {error}"
                        ))
                    })?;
            }
        }
    }

    let cols = vec![
        SqliteValue::Text(run_id.to_owned().into()),
        SqliteValue::Integer(seq),
        SqliteValue::Text(json_str(row, "ts_rfc3339")?.into()),
        SqliteValue::Text(json_str(row, "stage")?.into()),
        SqliteValue::Text(json_str(row, "code")?.into()),
        SqliteValue::Text(json_str(row, "message")?.into()),
        SqliteValue::Text(json_str(row, "payload_json")?.into()),
    ];
    connection
        .import_exec(
            "INSERT INTO events (run_id, seq, ts_rfc3339, stage, code, message, payload_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            &cols,
        )
        .map_err(|error| {
            FwError::Storage(format!("insert events `{run_id}/{seq}` failed: {error}"))
        })?;

    Ok(Some(cols))
}

/// Batched-import worker for `events` (mirror of [`flush_segment_chunk`]; composite
/// `(run_id, seq)` key, prefetched by `run_id`, seen-map for intra-chunk dups).
#[allow(
    clippy::too_many_arguments,
    reason = "event batching keeps parent-policy sets explicit and borrowed"
)]
fn flush_event_chunk(
    connection: &Connection,
    chunk: &mut Vec<(String, i64, serde_json::Value)>,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
    state: &mut EventImportState,
    imported_run_ids: &HashSet<String>,
    overwritten_run_ids: &HashSet<String>,
    inserted_run_ids: &HashSet<String>,
) -> FwResult<u64> {
    if chunk.is_empty() {
        return Ok(0);
    }

    let mut seen = HashSet::new();
    let unique_run_ids: Vec<String> = chunk
        .iter()
        .map(|(run_id, _, _)| run_id.clone())
        .filter(|run_id| {
            conflict_policy != ConflictPolicy::Skip || inserted_run_ids.contains(run_id)
        })
        .filter(|run_id| seen.insert(run_id.clone()))
        .collect();
    let mut existing_map: HashMap<(String, i64), Vec<SqliteValue>> = HashMap::new();
    if !unique_run_ids.is_empty() {
        let placeholders = (1..=unique_run_ids.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json FROM events WHERE run_id IN ({placeholders})"
        );
        let params: Vec<SqliteValue> = unique_run_ids
            .iter()
            .map(|run_id| SqliteValue::Text(run_id.clone().into()))
            .collect();
        let rows = connection
            .query_with_params(&sql, &params)
            .map_err(|error| {
                FwError::Storage(format!("batch query events existing failed: {error}"))
            })?;

        for row in &rows {
            let cols = event_row_to_cols(row);
            let run_id = value_to_string_sqlite(cols.first());
            if let Some(SqliteValue::Integer(seq)) = cols.get(1) {
                existing_map.insert((run_id, *seq), cols);
            }
        }
    }

    let mut count = 0u64;
    for (run_id, seq, row) in chunk.drain(..) {
        let key = format!("{run_id}/{seq}");
        record_event_pre(
            connection,
            &run_id,
            seq,
            &key,
            conflict_policy,
            imported_run_ids,
            overwritten_run_ids,
            state,
        )?;
        if skip_preexisting_child_row(
            &run_id,
            &row,
            conflict_policy,
            inserted_run_ids,
            &["ts_rfc3339", "stage", "code", "message", "payload_json"],
        )? {
            count += 1;
            continue;
        }
        let existing = existing_map.get(&(run_id.clone(), seq)).cloned();
        if let Some(cols) = apply_event_row(
            connection,
            &run_id,
            seq,
            &key,
            &row,
            existing.as_deref(),
            conflict_policy,
            conflicts,
        )? {
            existing_map.insert((run_id, seq), cols); // seen-map for intra-chunk dup
        }
        count += 1;
    }

    Ok(count)
}

#[allow(
    clippy::too_many_arguments,
    reason = "event import keeps parent-policy sets explicit and borrowed"
)]
fn import_events(
    connection: &Connection,
    path: &Path,
    conflict_policy: ConflictPolicy,
    conflicts: &mut Vec<SyncConflict>,
    imported_run_ids: &HashSet<String>,
    overwritten_run_ids: &HashSet<String>,
    inserted_run_ids: &HashSet<String>,
    overwritten_event_seqs_before: &HashMap<String, HashSet<i64>>,
    batch_import_enabled: bool,
) -> FwResult<u64> {
    let reader = open_jsonl_reader(path)?;
    let mut count = 0u64;
    let mut state = EventImportState::default();

    if batch_import_enabled {
        let chunk_size = sync_query_batch_size().max(1);
        let mut chunk: Vec<(String, i64, serde_json::Value)> = Vec::with_capacity(chunk_size);
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: serde_json::Value = serde_json::from_str(line)?;
            let run_id = json_str(&row, "run_id")?;
            let seq = row
                .get("seq")
                .and_then(|value| value.as_i64())
                .ok_or_else(|| FwError::Storage("missing seq in events row".to_owned()))?;
            chunk.push((run_id, seq, row));
            if chunk.len() >= chunk_size {
                count += flush_event_chunk(
                    connection,
                    &mut chunk,
                    conflict_policy,
                    conflicts,
                    &mut state,
                    imported_run_ids,
                    overwritten_run_ids,
                    inserted_run_ids,
                )?;
            }
        }
        count += flush_event_chunk(
            connection,
            &mut chunk,
            conflict_policy,
            conflicts,
            &mut state,
            imported_run_ids,
            overwritten_run_ids,
            inserted_run_ids,
        )?;
    } else {
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let row: serde_json::Value = serde_json::from_str(line)?;
            let run_id = json_str(&row, "run_id")?;
            let seq = row
                .get("seq")
                .and_then(|value| value.as_i64())
                .ok_or_else(|| FwError::Storage("missing seq in events row".to_owned()))?;

            let key = format!("{run_id}/{seq}");
            record_event_pre(
                connection,
                &run_id,
                seq,
                &key,
                conflict_policy,
                imported_run_ids,
                overwritten_run_ids,
                &mut state,
            )?;
            if skip_preexisting_child_row(
                &run_id,
                &row,
                conflict_policy,
                inserted_run_ids,
                &["ts_rfc3339", "stage", "code", "message", "payload_json"],
            )? {
                count += 1;
                continue;
            }

            let existing_rows = connection
                .query_with_params(
                    "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json FROM events WHERE run_id = ?1 AND seq = ?2",
                    &[SqliteValue::Text(run_id.clone().into()), SqliteValue::Integer(seq)],
                )
                .map_err(|error| {
                    FwError::Storage(format!("query events existing `{run_id}/{seq}` failed: {error}"))
                })?;
            let existing = existing_rows.first().map(event_row_to_cols);
            apply_event_row(
                connection,
                &run_id,
                seq,
                &key,
                &row,
                existing.as_deref(),
                conflict_policy,
                conflicts,
            )?;
            count += 1;
        }
    }

    if conflict_policy == ConflictPolicy::Overwrite && !overwritten_run_ids.is_empty() {
        assert_no_stale_events_for_overwritten_runs(
            overwritten_run_ids,
            overwritten_event_seqs_before,
            &state.imported_seq_by_overwritten_run,
        )?;
    }
    if conflict_policy == ConflictPolicy::OverwriteStrict && !imported_run_ids.is_empty() {
        delete_stale_events_for_strict_overwrite(
            connection,
            imported_run_ids,
            &state.imported_seq_by_run,
        )?;
    }

    Ok(count)
}

fn assert_no_stale_segments_for_overwritten_runs(
    overwritten_run_ids: &HashSet<String>,
    existing_idx_before_by_run: &HashMap<String, HashSet<i64>>,
    imported_idx_by_run: &HashMap<String, HashSet<i64>>,
) -> FwResult<()> {
    for run_id in overwritten_run_ids {
        let Some(existing_idxs) = existing_idx_before_by_run.get(run_id) else {
            continue;
        };
        for idx in existing_idxs {
            let keep = imported_idx_by_run
                .get(run_id)
                .is_some_and(|idx_set| idx_set.contains(idx));
            if keep {
                continue;
            }
            return Err(FwError::Storage(format!(
                "overwrite would require deleting stale segment `{run_id}/{idx}`, \
                 but child-row DELETE is unsupported in this runtime; \
                 re-import into an empty target DB for strict replacement"
            )));
        }
    }
    Ok(())
}

fn assert_no_stale_events_for_overwritten_runs(
    overwritten_run_ids: &HashSet<String>,
    existing_seq_before_by_run: &HashMap<String, HashSet<i64>>,
    imported_seq_by_run: &HashMap<String, HashSet<i64>>,
) -> FwResult<()> {
    for run_id in overwritten_run_ids {
        let Some(existing_seqs) = existing_seq_before_by_run.get(run_id) else {
            continue;
        };
        for seq in existing_seqs {
            let keep = imported_seq_by_run
                .get(run_id)
                .is_some_and(|seq_set| seq_set.contains(seq));
            if keep {
                continue;
            }
            return Err(FwError::Storage(format!(
                "overwrite would require deleting stale event `{run_id}/{seq}`, \
                 but child-row DELETE is unsupported in this runtime; \
                 re-import into an empty target DB for strict replacement"
            )));
        }
    }
    Ok(())
}

fn delete_stale_segments_for_strict_overwrite(
    connection: &Connection,
    imported_run_ids: &HashSet<String>,
    imported_idx_by_run: &HashMap<String, HashSet<i64>>,
) -> FwResult<()> {
    for run_id in imported_run_ids {
        let existing_idxs = query_segment_idxs_for_run(connection, run_id)?;
        for idx in existing_idxs {
            let keep = imported_idx_by_run
                .get(run_id)
                .is_some_and(|idx_set| idx_set.contains(&idx));
            if keep {
                continue;
            }
            connection
                .import_exec(
                    "DELETE FROM segments WHERE run_id = ?1 AND idx = ?2",
                    &[
                        SqliteValue::Text(run_id.clone().into()),
                        SqliteValue::Integer(idx),
                    ],
                )
                .map_err(|error| {
                    FwError::Storage(format!(
                        "strict overwrite delete stale segment `{run_id}/{idx}` failed: {error}"
                    ))
                })?;
        }
    }
    Ok(())
}

fn delete_stale_events_for_strict_overwrite(
    connection: &Connection,
    imported_run_ids: &HashSet<String>,
    imported_seq_by_run: &HashMap<String, HashSet<i64>>,
) -> FwResult<()> {
    for run_id in imported_run_ids {
        let existing_seqs = query_event_seqs_for_run(connection, run_id)?;
        for seq in existing_seqs {
            let keep = imported_seq_by_run
                .get(run_id)
                .is_some_and(|seq_set| seq_set.contains(&seq));
            if keep {
                continue;
            }
            connection
                .import_exec(
                    "DELETE FROM events WHERE run_id = ?1 AND seq = ?2",
                    &[
                        SqliteValue::Text(run_id.clone().into()),
                        SqliteValue::Integer(seq),
                    ],
                )
                .map_err(|error| {
                    FwError::Storage(format!(
                        "strict overwrite delete stale event `{run_id}/{seq}` failed: {error}"
                    ))
                })?;
        }
    }
    Ok(())
}

fn query_segment_idxs_for_run(connection: &Connection, run_id: &str) -> FwResult<HashSet<i64>> {
    let rows = connection
        .query_with_params(
            "SELECT idx FROM segments WHERE run_id = ?1",
            &[SqliteValue::Text(run_id.to_owned().into())],
        )
        .map_err(|error| {
            FwError::Storage(format!(
                "query existing segments for overwrite `{run_id}` failed: {error}"
            ))
        })?;
    let mut idxs = HashSet::new();
    for row in rows {
        idxs.insert(value_to_i64_sqlite(row.get(0)));
    }
    Ok(idxs)
}

fn query_event_seqs_for_run(connection: &Connection, run_id: &str) -> FwResult<HashSet<i64>> {
    let rows = connection
        .query_with_params(
            "SELECT seq FROM events WHERE run_id = ?1",
            &[SqliteValue::Text(run_id.to_owned().into())],
        )
        .map_err(|error| {
            FwError::Storage(format!(
                "query existing events for overwrite `{run_id}` failed: {error}"
            ))
        })?;
    let mut seqs = HashSet::new();
    for row in rows {
        seqs.insert(value_to_i64_sqlite(row.get(0)));
    }
    Ok(seqs)
}

fn ensure_run_reference_exists(
    connection: &Connection,
    run_id: &str,
    table: &str,
    key: &str,
    known_run_ids: &mut HashSet<String>,
) -> FwResult<()> {
    if known_run_ids.contains(run_id) {
        return Ok(());
    }

    let parent = connection
        .query_with_params(
            "SELECT id FROM runs WHERE id = ?1 LIMIT 1",
            &[SqliteValue::Text(run_id.to_owned().into())],
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;

    if parent.is_empty() {
        return Err(FwError::Storage(format!(
            "referential integrity violation: {table} row `{key}` references missing runs.id `{run_id}`"
        )));
    }

    known_run_ids.insert(run_id.to_owned());
    Ok(())
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_schema_version(manifest: &SyncManifest) -> FwResult<()> {
    let expected_major = SCHEMA_VERSION.split('.').next().unwrap_or("1");
    let actual_major = manifest.schema_version.split('.').next().unwrap_or("0");

    if expected_major != actual_major {
        return Err(FwError::Storage(format!(
            "schema version mismatch: expected major {expected_major}, got {}",
            manifest.schema_version
        )));
    }
    Ok(())
}

fn validate_export_format_version(manifest: &SyncManifest) -> FwResult<()> {
    let expected_major = EXPORT_FORMAT_VERSION.split('.').next().unwrap_or("1");
    let actual_major = manifest
        .export_format_version
        .split('.')
        .next()
        .unwrap_or("0");

    if expected_major != actual_major {
        return Err(FwError::Storage(format!(
            "export format version mismatch: expected major {expected_major}, got {}",
            manifest.export_format_version
        )));
    }

    Ok(())
}

fn validate_checksums(
    manifest: &SyncManifest,
    input_dir: &Path,
) -> FwResult<JsonlInputPaths> {
    let input_paths = JsonlInputPaths::resolve(input_dir);
    validate_checksums_for_paths(manifest, &input_paths)?;
    Ok(input_paths)
}

fn validate_checksums_for_paths(
    manifest: &SyncManifest,
    input_paths: &JsonlInputPaths,
) -> FwResult<()> {
    let checks = [
        (
            "runs.jsonl",
            &input_paths.runs,
            &manifest.checksums.runs_jsonl_sha256,
        ),
        (
            "segments.jsonl",
            &input_paths.segments,
            &manifest.checksums.segments_jsonl_sha256,
        ),
        (
            "events.jsonl",
            &input_paths.events,
            &manifest.checksums.events_jsonl_sha256,
        ),
    ];

    for (logical_filename, path, expected) in checks {
        if !path.exists() {
            return Err(FwError::Storage(format!(
                "missing export file: {logical_filename} or {logical_filename}.gz in {}",
                path.parent().unwrap_or_else(|| Path::new(".")).display()
            )));
        }
        let actual = sha256_jsonl_file(path)?;
        if &actual != expected {
            return Err(FwError::Storage(format!(
                "checksum mismatch for {} (logical {logical_filename} bytes): expected {expected}, got {actual}",
                path.display()
            )));
        }
    }

    Ok(())
}

fn ensure_schema(connection: &Connection) -> FwResult<()> {
    let sql = r#"
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    backend TEXT NOT NULL,
    input_path TEXT NOT NULL,
    normalized_wav_path TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    transcript TEXT NOT NULL,
    replay_json TEXT NOT NULL DEFAULT '{}',
    acceleration_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS segments (
    run_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    start_sec REAL,
    end_sec REAL,
    speaker TEXT,
    text TEXT NOT NULL,
    confidence REAL,
    PRIMARY KEY (run_id, idx)
);

CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts_rfc3339 TEXT NOT NULL,
    stage TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"#;
    connection
        .execute(sql)
        .map_err(|error| FwError::Storage(error.to_string()))?;
    ensure_runs_replay_column(connection)?;
    Ok(())
}

/// Read-only schema check for export: verifies that required tables exist
/// without creating them.  Returns an error when the DB has no schema (e.g.
/// freshly created by `Connection::open`).
fn verify_schema_exists(connection: &Connection) -> FwResult<()> {
    for table in &["runs", "segments", "events"] {
        let table_ident = sql_ident_sync(table);
        let sql = format!("SELECT 1 FROM {table_ident} LIMIT 1");
        if connection.query(&sql).is_err() {
            return Err(FwError::Storage(format!(
                "export requires an existing database with schema; \
                 missing table `{table}`"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

fn atomic_rename(from: &Path, to: &Path) -> FwResult<()> {
    replace_file_atomically(from, to)?;
    sync_parent_dir(to)?;
    Ok(())
}

fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> FwResult<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
    }
    replace_file_atomically(&tmp, path)?;
    sync_parent_dir(path)?;
    Ok(())
}

fn replace_file_atomically(from: &Path, to: &Path) -> FwResult<()> {
    #[cfg(unix)]
    {
        fs::rename(from, to)?;
        Ok(())
    }

    #[cfg(not(unix))]
    {
        if to.exists() {
            fs::remove_file(to)?;
        }
        fs::rename(from, to)?;
        Ok(())
    }
}

fn sync_parent_dir(path: &Path) -> FwResult<()> {
    #[cfg(not(unix))]
    let _ = path;
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            let dir = fs::File::open(parent)?;
            dir.sync_all()?;
        }
    }
    Ok(())
}

/// A `Write` adapter that streams every byte through SHA-256 as it forwards them to
/// the inner writer. Lets the full-export writers compute each JSONL's checksum while
/// writing it — avoiding a second full pass to re-read the file (`sha256_file`). The
/// digest equals `sha256_file` of the written bytes exactly (same bytes, same hash).
struct HashingWriter<W: Write> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn finalize_hex(self) -> String {
        format!("{:x}", self.hasher.finalize())
    }

    fn get_ref(&self) -> &W {
        &self.inner
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
fn sha256_file(path: &Path) -> FwResult<String> {
    sha256_reader(fs::File::open(path)?)
}

/// Hash the logical JSONL bytes, transparently decompressing a `.gz` input.
fn sha256_jsonl_file(path: &Path) -> FwResult<String> {
    sha256_reader(open_jsonl_reader(path)?)
}

fn sha256_reader(mut reader: impl Read) -> FwResult<String> {
    let mut hasher = Sha256::new();
    // 64 KiB read buffer (matches the native-engine hasher): 8× fewer `read()`
    // syscalls than the previous 8 KiB when checksumming large export/import JSONL.
    // Same SHA regardless of chunk size.
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn value_to_json(value: Option<&SqliteValue>) -> serde_json::Value {
    match value {
        Some(SqliteValue::Text(text)) => serde_json::Value::String(text.to_string()),
        Some(SqliteValue::Integer(number)) => serde_json::json!(number),
        Some(SqliteValue::Float(number)) => serde_json::json!(number),
        Some(SqliteValue::Blob(blob)) => {
            serde_json::Value::String(format!("<blob:{}>", blob.len()))
        }
        Some(SqliteValue::Null) | None => serde_json::Value::Null,
    }
}

fn value_to_string_sqlite(value: Option<&SqliteValue>) -> String {
    match value {
        Some(SqliteValue::Text(text)) => text.to_string(),
        Some(SqliteValue::Integer(number)) => number.to_string(),
        Some(SqliteValue::Float(number)) => number.to_string(),
        _ => String::new(),
    }
}

fn sqlite_to_optional_f64(value: Option<&SqliteValue>) -> Option<f64> {
    match value {
        Some(SqliteValue::Float(number)) => Some(*number),
        Some(SqliteValue::Integer(number)) => Some(*number as f64),
        _ => None,
    }
}

fn sqlite_to_optional_text(value: Option<&SqliteValue>) -> Option<String> {
    match value {
        Some(SqliteValue::Text(text)) => Some(text.to_string()),
        _ => None,
    }
}

fn json_str(value: &serde_json::Value, key: &str) -> FwResult<String> {
    value
        .get(key)
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| FwError::Storage(format!("missing string field `{key}` in JSONL row")))
}

fn json_string_or_default(value: &serde_json::Value, key: &str, fallback: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| fallback.to_owned())
}

fn ensure_runs_replay_column(connection: &Connection) -> FwResult<()> {
    let mut rows = table_columns_sync(connection, "runs")?;
    let mut has_replay = rows
        .iter()
        .any(|row| value_to_string_sqlite(row.get(1)) == "replay_json");
    if !has_replay {
        recreate_table_with_added_column_sync(
            connection,
            "runs",
            &rows,
            "replay_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )?;
        rows = table_columns_sync(connection, "runs")?;
        has_replay = true;
    }

    let has_acceleration = rows
        .iter()
        .any(|row| value_to_string_sqlite(row.get(1)) == "acceleration_json");
    if has_replay && !has_acceleration {
        recreate_table_with_added_column_sync(
            connection,
            "runs",
            &rows,
            "acceleration_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )?;
    }
    Ok(())
}

fn table_columns_sync(connection: &Connection, table: &str) -> FwResult<Vec<fsqlite::Row>> {
    connection
        .query(&format!("PRAGMA table_info({});", sql_ident_sync(table)))
        .map_err(|error| FwError::Storage(error.to_string()))
}

fn recreate_table_with_added_column_sync(
    connection: &Connection,
    table: &str,
    existing_columns: &[fsqlite::Row],
    new_column: &str,
    new_column_def: &str,
) -> FwResult<()> {
    let table_ident = sql_ident_sync(table);
    let existing_names = existing_columns
        .iter()
        .map(|row| {
            let name = value_to_string_sqlite(row.get(1));
            if name.is_empty() {
                return Err(FwError::Storage(
                    "invalid PRAGMA table_info row: empty column name".to_owned(),
                ));
            }
            Ok(sql_ident_sync(&name))
        })
        .collect::<FwResult<Vec<_>>>()?;
    let existing_cols_csv = existing_names.join(", ");
    let rows = connection
        .query(&format!("SELECT {existing_cols_csv} FROM {table_ident};"))
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let row_values = rows
        .iter()
        .map(|row| {
            (0..existing_names.len())
                .map(|index| row.get(index).cloned().unwrap_or(SqliteValue::Null))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut column_defs = existing_columns
        .iter()
        .map(reconstruct_column_definition_sync)
        .collect::<FwResult<Vec<_>>>()?;
    column_defs.push(format!("{} {}", sql_ident_sync(new_column), new_column_def));

    connection
        .execute(&format!("DROP TABLE {table_ident};"))
        .map_err(|error| FwError::Storage(error.to_string()))?;
    connection
        .execute(&format!(
            "CREATE TABLE {table_ident} ({});",
            column_defs.join(", ")
        ))
        .map_err(|error| FwError::Storage(error.to_string()))?;

    if !row_values.is_empty() {
        let placeholders = (1..=existing_names.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let insert_sql =
            format!("INSERT INTO {table_ident} ({existing_cols_csv}) VALUES ({placeholders});");
        for params in row_values {
            connection
                .execute_with_params(&insert_sql, &params)
                .map_err(|error| FwError::Storage(error.to_string()))?;
        }
    }

    Ok(())
}

fn sql_ident_sync(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn value_to_i64_sqlite(value: Option<&SqliteValue>) -> i64 {
    match value {
        Some(SqliteValue::Integer(number)) => *number,
        Some(SqliteValue::Text(text)) => text.parse::<i64>().unwrap_or(0),
        _ => 0,
    }
}

fn reconstruct_column_definition_sync(row: &fsqlite::Row) -> FwResult<String> {
    let name = value_to_string_sqlite(row.get(1));
    if name.is_empty() {
        return Err(FwError::Storage(
            "invalid PRAGMA table_info row: empty column name".to_owned(),
        ));
    }
    let typ = value_to_string_sqlite(row.get(2));
    let not_null = value_to_i64_sqlite(row.get(3)) != 0;
    let default_value = value_to_string_sqlite(row.get(4));
    let is_primary_key = value_to_i64_sqlite(row.get(5)) != 0;

    let mut def = if typ.is_empty() {
        sql_ident_sync(&name)
    } else {
        format!("{} {typ}", sql_ident_sync(&name))
    };
    if not_null {
        def.push_str(" NOT NULL");
    }
    if !default_value.is_empty() {
        def.push_str(" DEFAULT ");
        def.push_str(&default_value);
    }
    if is_primary_key {
        def.push_str(" PRIMARY KEY");
    }
    Ok(def)
}

fn json_optional_float(value: &serde_json::Value, key: &str) -> SqliteValue {
    match value.get(key) {
        Some(serde_json::Value::Number(number)) => match number.as_f64() {
            Some(float) => SqliteValue::Float(float),
            None => SqliteValue::Null,
        },
        _ => SqliteValue::Null,
    }
}

fn json_optional_text(value: &serde_json::Value, key: &str) -> SqliteValue {
    match value.get(key) {
        Some(serde_json::Value::String(text)) => SqliteValue::Text(text.clone().into()),
        _ => SqliteValue::Null,
    }
}

fn json_to_optional_f64(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key).and_then(serde_json::Value::as_f64)
}

fn json_to_optional_text(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
}

fn optional_floats_equal(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
        (None, None) => true,
        _ => false,
    }
}

fn write_conflicts_file(input_dir: &Path, conflicts: &[SyncConflict]) -> FwResult<()> {
    let conflicts_path = input_dir.join("sync_conflicts.jsonl");
    let mut file = fs::File::create(&conflicts_path)?;
    for conflict in conflicts {
        writeln!(file, "{}", serde_json::to_string(conflict)?)?;
    }
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Compression support
// ---------------------------------------------------------------------------

/// Compression mode for JSONL files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    /// No compression — plain JSONL.
    None,
    /// Gzip compression (.jsonl.gz).
    Gzip,
}

/// Gzip-compresses a JSONL file, writing the result to `output_path`.
pub fn compress_jsonl(input_path: &Path, output_path: &Path) -> FwResult<()> {
    let tmp = output_path.with_extension("tmp");
    let input = fs::File::open(input_path)?;
    let writer = BufWriter::new(fs::File::create(&tmp)?);

    let mut encoder = GzEncoder::new(writer, Compression::default());
    std::io::copy(&mut BufReader::new(input), &mut encoder)?;
    let mut writer = encoder.finish()?;
    writer.flush()?;
    writer.get_ref().sync_all()?;

    replace_file_atomically(&tmp, output_path)?;
    sync_parent_dir(output_path)?;
    Ok(())
}

/// Decompresses a gzip-compressed JSONL file, writing the result to `output_path`.
pub fn decompress_jsonl(input_path: &Path, output_path: &Path) -> FwResult<()> {
    let tmp = output_path.with_extension("tmp");
    let input = fs::File::open(input_path)?;
    let mut decoder = GzDecoder::new(BufReader::new(input));
    let mut writer = BufWriter::new(fs::File::create(&tmp)?);

    std::io::copy(&mut decoder, &mut writer)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;

    replace_file_atomically(&tmp, output_path)?;
    sync_parent_dir(output_path)?;
    Ok(())
}

/// Open a JSONL file for reading, transparently decompressing if the path
/// ends with `.gz`.
fn open_jsonl_reader(path: &Path) -> FwResult<Box<dyn BufRead>> {
    let file = fs::File::open(path)?;
    if path.extension().is_some_and(|ext| ext == "gz") {
        let decoder = GzDecoder::new(BufReader::new(file));
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

// ---------------------------------------------------------------------------
// Sync validation
// ---------------------------------------------------------------------------

/// Report produced by [`validate_sync`] comparing a SQLite database against
/// a JSONL export directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncValidationReport {
    /// Number of runs in the SQLite database.
    pub db_run_count: u64,
    /// Number of runs in the JSONL export.
    pub jsonl_run_count: u64,
    /// Number of segments in the database.
    pub db_segment_count: u64,
    /// Number of segments in the JSONL export.
    pub jsonl_segment_count: u64,
    /// Number of events in the database.
    pub db_event_count: u64,
    /// Number of events in the JSONL export.
    pub jsonl_event_count: u64,
    /// Run IDs present in the database but absent from the JSONL export.
    pub missing_from_jsonl: Vec<String>,
    /// Run IDs present in the JSONL export but absent from the database.
    pub missing_from_db: Vec<String>,
    /// Run IDs where the record exists in both but data differs.
    pub mismatched_records: Vec<String>,
    /// Bounded child-row diagnostics keyed by `run_id/idx` or `run_id/seq`.
    pub child_mismatches: Vec<SyncConflict>,
    /// `true` when additional child-row diagnostics exceeded the report cap.
    pub child_mismatches_truncated: bool,
    /// `true` when the database and JSONL export are in perfect agreement.
    pub is_valid: bool,
}

type ChildValidationKey = (String, i64);

#[derive(Debug, PartialEq)]
struct SegmentValidationRow {
    start_sec: Option<f64>,
    end_sec: Option<f64>,
    speaker: Option<String>,
    text: String,
    confidence: Option<f64>,
}

#[derive(Debug, PartialEq, Eq)]
struct EventValidationRow {
    ts_rfc3339: String,
    stage: String,
    code: String,
    message: String,
    payload_json: String,
}

/// Validate that a SQLite database and a JSONL export directory are in sync.
///
/// Opens the database, reads the runs/segments/events JSONL files from
/// `jsonl_dir`, and compares every persisted run, segment, and event field.
/// Supports reading both plain `.jsonl` and compressed `.jsonl.gz` files (the
/// `.gz` variant is preferred when both exist). All database reads share one
/// read transaction so a concurrent writer cannot produce a torn comparison.
pub fn validate_sync(db_path: &Path, jsonl_dir: &Path) -> FwResult<SyncValidationReport> {
    validate_sync_with_after_run_ids(db_path, jsonl_dir, || Ok(()))
}

fn validate_sync_with_after_run_ids<F>(
    db_path: &Path,
    jsonl_dir: &Path,
    after_run_ids: F,
) -> FwResult<SyncValidationReport>
where
    F: FnOnce() -> FwResult<()>,
{
    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    with_read_snapshot(&connection, || {
        validate_sync_snapshot(&connection, jsonl_dir, after_run_ids)
    })
}

fn validate_sync_snapshot<F>(
    connection: &Connection,
    jsonl_dir: &Path,
    after_run_ids: F,
) -> FwResult<SyncValidationReport>
where
    F: FnOnce() -> FwResult<()>,
{
    verify_schema_exists(connection)?;
    let input_paths = require_validation_jsonl_inputs(jsonl_dir)?;

    // --- Collect DB run IDs ---
    let db_run_rows = connection
        .query("SELECT id FROM runs ORDER BY id ASC")
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let db_run_ids: HashSet<String> = db_run_rows
        .iter()
        .map(|row| value_to_string_sqlite(row.get(0)))
        .collect();
    after_run_ids()?;

    // --- Collect child rows from this same DB snapshot ---
    let db_segment_map = load_db_segment_map(connection)?;
    let db_event_map = load_db_event_map(connection)?;
    let db_segment_count = db_segment_map.len() as u64;
    let db_event_count = db_event_map.len() as u64;

    // --- Read JSONL runs ONCE (the map + the id set) ---
    // `runs.jsonl` was previously read twice — `collect_jsonl_ids` here and
    // `load_jsonl_run_map` later. Load the map once and derive the id set from its
    // keys: both include exactly the lines with a string `id` (skipping empty lines,
    // parsing every line), so `map.keys()` == `collect_jsonl_ids(runs, "id")` and the
    // parse-error behavior is identical.
    let jsonl_run_map = load_jsonl_run_map(&input_paths.runs)?;
    let jsonl_run_ids: HashSet<String> = jsonl_run_map.keys().cloned().collect();

    // --- Parse and validate JSONL children ---
    let jsonl_segment_map = load_jsonl_segment_map(&input_paths.segments)?;
    let jsonl_segment_count = jsonl_segment_map.len() as u64;

    let jsonl_event_map = load_jsonl_event_map(&input_paths.events)?;
    let jsonl_event_count = jsonl_event_map.len() as u64;

    // --- Compare ---
    let mut missing_from_jsonl: Vec<String> =
        db_run_ids.difference(&jsonl_run_ids).cloned().collect();
    missing_from_jsonl.sort();

    let mut missing_from_db: Vec<String> = jsonl_run_ids.difference(&db_run_ids).cloned().collect();
    missing_from_db.sort();

    // --- Compare record content for shared run IDs ---
    // (`jsonl_run_map` already loaded above — no second read of runs.jsonl.)
    let shared_ids: Vec<&String> = db_run_ids.intersection(&jsonl_run_ids).collect();
    let mut mismatched_run_ids: HashSet<String> = HashSet::new();

    // Prefetch the DB rows for all shared ids with one `WHERE id IN (…)` per chunk
    // instead of one `SELECT … WHERE id = ?1` per shared run (query *setup* dominates
    // the single-row lookups) — the read-only mirror of the import N+1 batch. Byte-
    // identical: the compare is unchanged and `mismatched_records` is sorted below, so
    // fetch order is irrelevant. `FW_SYNC_BATCH_QUERY=0` (chunk 1) restores the legacy
    // per-id N+1. `shared_ids` ⊆ `db_run_ids`, so every id resolves in the map.
    let mut db_run_map: HashMap<String, Vec<SqliteValue>> =
        HashMap::with_capacity(shared_ids.len());
    for chunk in shared_ids.chunks(sync_query_batch_size().max(1)) {
        let placeholders = (1..=chunk.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, \
             transcript, replay_json, acceleration_json FROM runs WHERE id IN ({placeholders})"
        );
        let params: Vec<SqliteValue> = chunk
            .iter()
            .map(|id| SqliteValue::Text((**id).clone().into()))
            .collect();
        let rows = connection
            .query_with_params(&sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?;
        for row in &rows {
            let cols = run_row_to_cols(row);
            db_run_map.insert(value_to_string_sqlite(cols.first()), cols);
        }
    }

    for id in &shared_ids {
        if let Some(db_row) = db_run_map.get(*id)
            && let Some(jsonl_value) = jsonl_run_map.get(*id)
        {
            let matches = value_to_string_sqlite(db_row.get(1))
                == json_str_or_empty(jsonl_value, "started_at")
                && value_to_string_sqlite(db_row.get(2))
                    == json_str_or_empty(jsonl_value, "finished_at")
                && value_to_string_sqlite(db_row.get(3))
                    == json_str_or_empty(jsonl_value, "backend")
                && value_to_string_sqlite(db_row.get(4))
                    == json_str_or_empty(jsonl_value, "input_path")
                && value_to_string_sqlite(db_row.get(5))
                    == json_str_or_empty(jsonl_value, "normalized_wav_path")
                && value_to_string_sqlite(db_row.get(6))
                    == json_str_or_empty(jsonl_value, "request_json")
                && value_to_string_sqlite(db_row.get(7))
                    == json_str_or_empty(jsonl_value, "result_json")
                && value_to_string_sqlite(db_row.get(8))
                    == json_str_or_empty(jsonl_value, "warnings_json")
                && value_to_string_sqlite(db_row.get(9))
                    == json_str_or_empty(jsonl_value, "transcript")
                && value_to_string_sqlite(db_row.get(10))
                    == json_string_or_default(jsonl_value, "replay_json", "{}")
                && value_to_string_sqlite(db_row.get(11))
                    == json_string_or_default(jsonl_value, "acceleration_json", "{}");
            if !matches {
                mismatched_run_ids.insert((*id).clone());
            }
        }
    }

    let mut child_mismatches = Vec::new();
    let mut child_mismatches_truncated = false;
    collect_child_mismatches(
        &db_segment_map,
        &jsonl_segment_map,
        "segments",
        &mut child_mismatches,
        &mut child_mismatches_truncated,
    );
    collect_child_mismatches(
        &db_event_map,
        &jsonl_event_map,
        "events",
        &mut child_mismatches,
        &mut child_mismatches_truncated,
    );

    // Parent-row mismatches retain the existing run-ID diagnostic contract;
    // child-row diagnostics are structured and bounded separately.
    let mut mismatched_records: Vec<String> = mismatched_run_ids.into_iter().collect();
    mismatched_records.sort();

    let is_valid = missing_from_jsonl.is_empty()
        && missing_from_db.is_empty()
        && mismatched_records.is_empty()
        && child_mismatches.is_empty()
        && !child_mismatches_truncated
        && db_run_ids.len() as u64 == jsonl_run_ids.len() as u64
        && db_segment_count == jsonl_segment_count
        && db_event_count == jsonl_event_count;

    Ok(SyncValidationReport {
        db_run_count: db_run_ids.len() as u64,
        jsonl_run_count: jsonl_run_ids.len() as u64,
        db_segment_count,
        jsonl_segment_count,
        db_event_count,
        jsonl_event_count,
        missing_from_jsonl,
        missing_from_db,
        mismatched_records,
        child_mismatches,
        child_mismatches_truncated,
        is_valid,
    })
}

fn load_db_segment_map(
    connection: &Connection,
) -> FwResult<HashMap<ChildValidationKey, SegmentValidationRow>> {
    let rows = connection
        .query(
            "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence \
             FROM segments ORDER BY run_id ASC, idx ASC",
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let mut map = HashMap::with_capacity(rows.len());
    for row in &rows {
        let run_id = sqlite_required_text(row.get(0), "segments.run_id")?;
        let idx = sqlite_required_i64(row.get(1), "segments.idx")?;
        let value = SegmentValidationRow {
            start_sec: sqlite_nullable_f64(row.get(2), "segments.start_sec")?,
            end_sec: sqlite_nullable_f64(row.get(3), "segments.end_sec")?,
            speaker: sqlite_nullable_text(row.get(4), "segments.speaker")?,
            text: sqlite_required_text(row.get(5), "segments.text")?,
            confidence: sqlite_nullable_f64(row.get(6), "segments.confidence")?,
        };
        if map.insert((run_id, idx), value).is_some() {
            return Err(FwError::Storage(
                "duplicate composite key in SQLite segments table".to_owned(),
            ));
        }
    }
    Ok(map)
}

fn load_db_event_map(
    connection: &Connection,
) -> FwResult<HashMap<ChildValidationKey, EventValidationRow>> {
    let rows = connection
        .query(
            "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json \
             FROM events ORDER BY run_id ASC, seq ASC",
        )
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let mut map = HashMap::with_capacity(rows.len());
    for row in &rows {
        let run_id = sqlite_required_text(row.get(0), "events.run_id")?;
        let seq = sqlite_required_i64(row.get(1), "events.seq")?;
        let value = EventValidationRow {
            ts_rfc3339: sqlite_required_text(row.get(2), "events.ts_rfc3339")?,
            stage: sqlite_required_text(row.get(3), "events.stage")?,
            code: sqlite_required_text(row.get(4), "events.code")?,
            message: sqlite_required_text(row.get(5), "events.message")?,
            payload_json: sqlite_required_text(row.get(6), "events.payload_json")?,
        };
        if map.insert((run_id, seq), value).is_some() {
            return Err(FwError::Storage(
                "duplicate composite key in SQLite events table".to_owned(),
            ));
        }
    }
    Ok(map)
}

fn load_jsonl_segment_map(
    path: &Path,
) -> FwResult<HashMap<ChildValidationKey, SegmentValidationRow>> {
    let mut map = HashMap::new();
    if !path.exists() {
        return Ok(map);
    }
    for (line_index, line) in open_jsonl_reader(path)?.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(trimmed)?;
        let run_id = json_str(&row, "run_id")?;
        let idx = json_required_i64(&row, "idx", "segments")?;
        let value = SegmentValidationRow {
            start_sec: json_nullable_f64(&row, "start_sec", "segments")?,
            end_sec: json_nullable_f64(&row, "end_sec", "segments")?,
            speaker: json_nullable_text(&row, "speaker", "segments")?,
            text: json_str(&row, "text")?,
            confidence: json_nullable_f64(&row, "confidence", "segments")?,
        };
        let key = (run_id, idx);
        if map.insert(key.clone(), value).is_some() {
            return Err(FwError::Storage(format!(
                "duplicate composite key {}/{} in segments JSONL at line {}",
                key.0,
                key.1,
                line_index + 1
            )));
        }
    }
    Ok(map)
}

fn load_jsonl_event_map(
    path: &Path,
) -> FwResult<HashMap<ChildValidationKey, EventValidationRow>> {
    let mut map = HashMap::new();
    if !path.exists() {
        return Ok(map);
    }
    for (line_index, line) in open_jsonl_reader(path)?.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(trimmed)?;
        let run_id = json_str(&row, "run_id")?;
        let seq = json_required_i64(&row, "seq", "events")?;
        let value = EventValidationRow {
            ts_rfc3339: json_str(&row, "ts_rfc3339")?,
            stage: json_str(&row, "stage")?,
            code: json_str(&row, "code")?,
            message: json_str(&row, "message")?,
            payload_json: json_str(&row, "payload_json")?,
        };
        let key = (run_id, seq);
        if map.insert(key.clone(), value).is_some() {
            return Err(FwError::Storage(format!(
                "duplicate composite key {}/{} in events JSONL at line {}",
                key.0,
                key.1,
                line_index + 1
            )));
        }
    }
    Ok(map)
}

fn collect_child_mismatches<T: PartialEq>(
    db_rows: &HashMap<ChildValidationKey, T>,
    jsonl_rows: &HashMap<ChildValidationKey, T>,
    table: &str,
    diagnostics: &mut Vec<SyncConflict>,
    diagnostics_truncated: &mut bool,
) {
    let mut keys: HashSet<ChildValidationKey> = db_rows.keys().cloned().collect();
    keys.extend(jsonl_rows.keys().cloned());
    let mut keys: Vec<ChildValidationKey> = keys.into_iter().collect();
    keys.sort();

    for key in keys {
        let run_id = &key.0;
        let index = key.1;
        let reason = match (db_rows.get(&key), jsonl_rows.get(&key)) {
            (Some(db), Some(jsonl)) if db == jsonl => continue,
            (Some(_), Some(_)) => "payload differs",
            (Some(_), None) => "missing from JSONL",
            (None, Some(_)) => "missing from database",
            (None, None) => continue,
        };
        if diagnostics.len() < MAX_SYNC_VALIDATION_CHILD_MISMATCHES {
            diagnostics.push(SyncConflict {
                table: table.to_owned(),
                key: format!("{run_id}/{index}"),
                reason: reason.to_owned(),
            });
        } else {
            *diagnostics_truncated = true;
        }
    }
}

fn sqlite_required_text(value: Option<&SqliteValue>, field: &str) -> FwResult<String> {
    match value {
        Some(SqliteValue::Text(text)) => Ok(text.to_string()),
        _ => Err(FwError::Storage(format!(
            "invalid non-text value in SQLite field `{field}`"
        ))),
    }
}

fn sqlite_required_i64(value: Option<&SqliteValue>, field: &str) -> FwResult<i64> {
    match value {
        Some(SqliteValue::Integer(number)) => Ok(*number),
        _ => Err(FwError::Storage(format!(
            "invalid non-integer value in SQLite field `{field}`"
        ))),
    }
}

fn sqlite_nullable_f64(value: Option<&SqliteValue>, field: &str) -> FwResult<Option<f64>> {
    match value {
        Some(SqliteValue::Float(number)) if number.is_finite() => Ok(Some(*number)),
        Some(SqliteValue::Integer(number)) => Ok(Some(*number as f64)),
        Some(SqliteValue::Null) => Ok(None),
        _ => Err(FwError::Storage(format!(
            "invalid numeric value in SQLite field `{field}`"
        ))),
    }
}

fn sqlite_nullable_text(value: Option<&SqliteValue>, field: &str) -> FwResult<Option<String>> {
    match value {
        Some(SqliteValue::Text(text)) => Ok(Some(text.to_string())),
        Some(SqliteValue::Null) => Ok(None),
        _ => Err(FwError::Storage(format!(
            "invalid text value in SQLite field `{field}`"
        ))),
    }
}

fn json_required_i64(value: &serde_json::Value, key: &str, table: &str) -> FwResult<i64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| FwError::Storage(format!("invalid `{key}` in {table} JSONL row")))
}

fn json_nullable_f64(
    value: &serde_json::Value,
    key: &str,
    table: &str,
) -> FwResult<Option<f64>> {
    match value.get(key) {
        Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(number)) => number
            .as_f64()
            .filter(|number| number.is_finite())
            .map(Some)
            .ok_or_else(|| {
                FwError::Storage(format!("invalid `{key}` in {table} JSONL row"))
            }),
        _ => Err(FwError::Storage(format!(
            "invalid `{key}` in {table} JSONL row"
        ))),
    }
}

fn json_nullable_text(
    value: &serde_json::Value,
    key: &str,
    table: &str,
) -> FwResult<Option<String>> {
    match value.get(key) {
        Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(text)) => Ok(Some(text.clone())),
        _ => Err(FwError::Storage(format!(
            "invalid `{key}` in {table} JSONL row"
        ))),
    }
}

/// Resolve a JSONL file path, preferring `.jsonl.gz` over `.jsonl`.
fn resolve_jsonl_path(dir: &Path, stem: &str) -> PathBuf {
    let gz_path = dir.join(format!("{stem}.jsonl.gz"));
    if gz_path.exists() {
        gz_path
    } else {
        dir.join(format!("{stem}.jsonl"))
    }
}

fn require_validation_jsonl_inputs(dir: &Path) -> FwResult<JsonlInputPaths> {
    let input_paths = JsonlInputPaths::resolve(dir);
    for (logical_name, path) in [
        ("runs.jsonl", &input_paths.runs),
        ("segments.jsonl", &input_paths.segments),
        ("events.jsonl", &input_paths.events),
    ] {
        if !path.exists() {
            return Err(FwError::Storage(format!(
                "missing sync validation input: {logical_name} or {logical_name}.gz in {}",
                dir.display()
            )));
        }
    }
    Ok(input_paths)
}

/// Count the rows in a table via `SELECT COUNT(*)`.
#[cfg(test)]
fn count_table(connection: &Connection, table: &str) -> FwResult<u64> {
    let table_ident = sql_ident_sync(table);
    let sql = format!("SELECT COUNT(*) FROM {table_ident}");
    let rows = connection
        .query(&sql)
        .map_err(|error| FwError::Storage(error.to_string()))?;
    match rows.first() {
        Some(row) => match row.get(0) {
            Some(SqliteValue::Integer(n)) => Ok(*n as u64),
            _ => Ok(0),
        },
        None => Ok(0),
    }
}

/// Collect all values of a given string `key` from a JSONL file into a set.
///
/// Test-only since `7632a2b` removed the production caller (`validate_sync` now
/// uses `load_jsonl_run_map().keys()` for the same set — a single read instead of
/// two); kept as the equivalence reference exercised by the `collect_jsonl_ids_*`
/// unit tests. `#[cfg(test)]` so a non-test build doesn't warn dead_code.
#[cfg(test)]
fn collect_jsonl_ids(path: &Path, key: &str) -> FwResult<HashSet<String>> {
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let reader = open_jsonl_reader(path)?;
    let mut ids = HashSet::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)?;
        if let Some(id_str) = value.get(key).and_then(serde_json::Value::as_str) {
            ids.insert(id_str.to_owned());
        }
    }
    Ok(ids)
}

/// Count non-empty lines in a JSONL file.
#[cfg(test)]
fn count_jsonl_lines(path: &Path) -> FwResult<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let reader = open_jsonl_reader(path)?;
    let mut count = 0u64;
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

/// Load all runs from a JSONL file into a map keyed by run ID.
fn load_jsonl_run_map(
    path: &Path,
) -> FwResult<std::collections::HashMap<String, serde_json::Value>> {
    let mut map = std::collections::HashMap::new();
    if !path.exists() {
        return Ok(map);
    }
    let reader = open_jsonl_reader(path)?;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)?;
        if let Some(id_str) = value.get("id").and_then(serde_json::Value::as_str) {
            if map.contains_key(id_str) {
                return Err(FwError::Storage(format!(
                    "duplicate run id {id_str:?} in runs JSONL at line {}",
                    line_index + 1
                )));
            }
            map.insert(id_str.to_owned(), value);
        }
    }
    Ok(map)
}

/// Extract a string field from a JSON value, returning an empty string on
/// missing or non-string values.
fn json_str_or_empty(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_owned()
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Return the maximum `started_at` timestamp from the `runs` table,
/// optionally filtered to rows where `started_at > after_ts`.
///
/// Returns `None` if no matching rows exist.
pub fn max_started_at(conn: &Connection, after_ts: Option<&str>) -> FwResult<Option<String>> {
    let (sql, params): (&str, Vec<SqliteValue>) = if let Some(ts) = after_ts {
        (
            "SELECT MAX(started_at) FROM runs WHERE started_at > ?1",
            vec![SqliteValue::Text(ts.to_owned().into())],
        )
    } else {
        ("SELECT MAX(started_at) FROM runs", vec![])
    };

    let rows = conn
        .query_with_params(sql, &params)
        .map_err(|e| FwError::Storage(e.to_string()))?;

    if rows.is_empty() {
        return Ok(None);
    }

    match rows[0].get(0) {
        Some(SqliteValue::Text(s)) if !s.is_empty() => Ok(Some(s.to_string())),
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;
    use tempfile::tempdir;

    use crate::model::{
        BackendKind, BackendParams, DiarizationEngine, DiarizationFallbackStatus,
        DiarizationOperationalPartitionMethod, DiarizationOperationalPartitionSummary,
        DiarizationReport, DiarizationRequest, DiarizationTurn, InputSource, KnownSpeakerInterval,
        KnownSpeakerPolicy, NeuralSpeakerRepresentationReason, NeuralSpeakerRepresentationStatus,
        NeuralSpeakerRepresentationSummary, RunEvent, RunReport, SpeakerCountCalibrationStatus,
        SpeakerCountEstimate, SpeakerCountEvidenceLane, SpeakerCountLaneEvidence,
        SpeakerCountOutcome, SpeakerCountOutcomeReason, SpeakerCountOutcomeStatus,
        SpeakerCountPosteriorBin, SpeakerCountRange, SpeakerCountRequest,
        SpeakerCountResourceSummary, SpeakerEvidenceReason, SpeakerEvidenceSummary,
        SpeakerHintDisposition, SpeakerHintEvidenceSummary, SpeakerProfileSummary,
        TranscribeRequest, TranscriptionResult, TranscriptionSegment,
    };
    use crate::storage::{BlockingConnection, RunStore};

    use super::*;

    fn write_jsonl_snapshot(
        output_dir: &Path,
        runs_rows: &[serde_json::Value],
        segments_rows: &[serde_json::Value],
        events_rows: &[serde_json::Value],
    ) {
        fs::create_dir_all(output_dir).expect("output dir should exist");

        let runs_path = output_dir.join("runs.jsonl");
        let segments_path = output_dir.join("segments.jsonl");
        let events_path = output_dir.join("events.jsonl");

        let runs_jsonl = runs_rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("runs row should serialize"))
            .collect::<Vec<_>>()
            .join("\n");
        let segments_jsonl = segments_rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("segments row should serialize"))
            .collect::<Vec<_>>()
            .join("\n");
        let events_jsonl = events_rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("events row should serialize"))
            .collect::<Vec<_>>()
            .join("\n");

        fs::write(
            &runs_path,
            if runs_jsonl.is_empty() {
                String::new()
            } else {
                format!("{runs_jsonl}\n")
            },
        )
        .expect("runs jsonl should write");
        fs::write(
            &segments_path,
            if segments_jsonl.is_empty() {
                String::new()
            } else {
                format!("{segments_jsonl}\n")
            },
        )
        .expect("segments jsonl should write");
        fs::write(
            &events_path,
            if events_jsonl.is_empty() {
                String::new()
            } else {
                format!("{events_jsonl}\n")
            },
        )
        .expect("events jsonl should write");

        let manifest = SyncManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "fixture.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: runs_rows.len() as u64,
                segments: segments_rows.len() as u64,
                events: events_rows.len() as u64,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: sha256_file(&runs_path).expect("runs checksum"),
                segments_jsonl_sha256: sha256_file(&segments_path).expect("segments checksum"),
                events_jsonl_sha256: sha256_file(&events_path).expect("events checksum"),
            },
        };

        fs::write(
            output_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");
    }

    fn fixture_report(id: &str, db_path: &Path) -> RunReport {
        RunReport {
            run_id: id.to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-01-01T00:00:05Z".to_owned(),
            input_path: "test.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("test.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: Some("en".to_owned()),
                translate: false,
                diarize: false,
                persist: true,
                db_path: db_path.to_path_buf(),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "hello world from sync test".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![
                    TranscriptionSegment {
                        start_sec: Some(0.0),
                        end_sec: Some(2.5),
                        text: "hello world".to_owned(),
                        speaker: Some("SPEAKER_00".to_owned()),
                        confidence: Some(0.95),
                    },
                    TranscriptionSegment {
                        start_sec: Some(2.5),
                        end_sec: Some(5.0),
                        text: "from sync test".to_owned(),
                        speaker: None,
                        confidence: None,
                    },
                ],
                acceleration: None,
                diarization: None,
                raw_output: json!({"test": true}),
                artifact_paths: vec!["out.json".to_owned()],
            },
            events: vec![
                RunEvent {
                    seq: 1,
                    ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
                    stage: "ingest".to_owned(),
                    code: "ingest.ok".to_owned(),
                    message: "input materialized".to_owned(),
                    payload: json!({"path": "test.wav"}),
                },
                RunEvent {
                    seq: 2,
                    ts_rfc3339: "2026-01-01T00:00:03Z".to_owned(),
                    stage: "backend".to_owned(),
                    code: "backend.ok".to_owned(),
                    message: "backend completed".to_owned(),
                    payload: json!({"resolved_backend": "whisper_cpp"}),
                },
            ],
            warnings: vec!["test warning".to_owned()],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope {
                input_content_hash: Some("sync-input-hash".to_owned()),
                backend_identity: Some("whisper-cli".to_owned()),
                backend_version: Some("whisper 1.2.3".to_owned()),
                output_payload_hash: Some("sync-output-hash".to_owned()),
                pcm_sha256: None,
                live_note: None,
            },
        }
    }

    fn attach_fixture_diarization(report: &mut RunReport) {
        report.request.diarize = true;
        let diarization_request = DiarizationRequest {
            engine: DiarizationEngine::Acoustic,
            speaker_count: SpeakerCountRequest::Range {
                minimum: 1,
                maximum: 2,
            },
            known_intervals: vec![KnownSpeakerInterval {
                speaker_ref: "speaker_a".to_owned(),
                start_ms: 0,
                end_ms: 2_000,
                confidence: 0.92,
                policy: KnownSpeakerPolicy::SoftEnrollment,
                provenance: Some("synthetic_sync_fixture".to_owned()),
            }],
            ..DiarizationRequest::default()
        };
        let hint_document_sha256 = crate::model::speaker_hint_document_sha256(&diarization_request);
        report.request.backend_params.acoustic_diarization = Some(diarization_request);
        report.result.diarization = Some(DiarizationReport {
            implementation: "native-acoustic-v2".to_owned(),
            contract_version: "acoustic-diarization-v3".to_owned(),
            feature_schema: "acoustic-feature-v2".to_owned(),
            speaker_evidence_mode: crate::model::DiarizationSpeakerEvidenceMode::AcousticV2,
            normalized_input_sha256:
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
            hint_document_sha256: Some(hint_document_sha256),
            turns: vec![DiarizationTurn {
                start_ms: 0,
                end_ms: 2_000,
                speaker_ref: Some("speaker_a".to_owned()),
                speaker_confidence: Some(0.9),
                change_confidence: Some(0.82),
                overlap_suspected: false,
                hard_hint_attributed: false,
            }],
            profiles: vec![SpeakerProfileSummary {
                speaker_ref: "speaker_a".to_owned(),
                frame_count: 180,
                voiced_duration_ms: 1_800,
                reliability: 0.88,
                voice_profile_count: 1,
                channel_profile_count: 1,
                training_accepted_count: 1,
                training_downweighted_count: 0,
                training_quarantined_count: 0,
                anchored: false,
                soft_hint_contradiction: Some(0.04),
            }],
            hint_evidence: vec![SpeakerHintEvidenceSummary {
                hint_index: 0,
                speaker_ref: "speaker_a".to_owned(),
                policy: KnownSpeakerPolicy::SoftEnrollment,
                disposition: SpeakerHintDisposition::PartiallyAccepted,
                usable_tracklet_count: 3,
                accepted_tracklet_count: 2,
                rejected_tracklet_count: 1,
                profile_accepted_tracklet_count: 1,
                profile_downweighted_tracklet_count: 1,
                profile_quarantined_tracklet_count: 0,
                applied_weight: 20.0,
                contradiction_score: Some(0.04),
            }],
            speaker_queries: Vec::new(),
            speaker_count: SpeakerCountOutcome {
                request: SpeakerCountRequest::Range {
                    minimum: 1,
                    maximum: 2,
                },
                estimate: Some(synthetic_speaker_count_estimate()),
                status: SpeakerCountOutcomeStatus::Resolved,
                supported_speaker_count: 1,
                active_speaker_refs: vec!["speaker_a".to_owned()],
                dominant_speaker_share: 0.9,
                unknown_voiced_share: 0.1,
                reasons: vec![SpeakerCountOutcomeReason::EvidenceSupportedCount],
                speaker_evidence: vec![SpeakerEvidenceSummary {
                    speaker_ref: "speaker_a".to_owned(),
                    assigned_tracklet_count: 3,
                    independent_tracklet_count: 2,
                    recurrence_episode_count: 2,
                    voiced_frame_count: 180,
                    independent_voiced_frame_count: 120,
                    voiced_duration_ms: 1_800,
                    mean_assignment_confidence: 0.9,
                    profile_reliability: 0.88,
                    hard_anchored: false,
                    separated_from_supported_speakers: true,
                    reasons: vec![SpeakerEvidenceReason::SupportedByIndependentRecurrence],
                    supported: true,
                }],
            },
            fallback_status: DiarizationFallbackStatus::NotNeeded,
            operational_partition: Some(DiarizationOperationalPartitionSummary {
                schema_version: "diarization-operational-partition-v2".to_owned(),
                method: DiarizationOperationalPartitionMethod::ProbabilisticConsensus,
                selected_count: 1,
                confidence: 0.7,
                calibration_sha256: crate::diarization::acoustic_speaker_pair_calibration_sha256(),
                authority: SpeakerCountCalibrationStatus::DevelopmentUncertified,
            }),
            neural_representation: None,
            speaker_segments: Vec::new(),
            diagnostics: Vec::new(),
        });
        report.result.raw_output = json!({
            "engine": "whisper.cpp-native",
            "projection_timeline": {
                "schema_version": crate::conformance::DTW_PROJECTION_SCHEMA_VERSION,
                "unit": "seconds",
                "interval_semantics": "half_open",
                "timestamp_epsilon_sec": crate::conformance::CANONICAL_PROJECTION_EPSILON_SEC,
                "minimum_duration_sec": crate::conformance::CANONICAL_PROJECTION_MIN_DURATION_SEC,
                "input_engine_segments": 2,
                "input_timed_segments": 2,
                "canonical_units": 2,
                "output_segments": 2,
                "decoder_word_units": 2,
                "interpolated_fallback_units": 0,
                "segment_geometry_fallback_units": 0,
                "interpolated_fallback_segments": 0,
                "segment_geometry_fallback_segments": 0,
                "clamped_units": 0,
                "expanded_units": 0,
                "fallback_reasons": [],
                "word_aligned_safe": true,
                "supported_provenance": [
                    "decoder_word_timestamp",
                    "segment_interpolation_fallback",
                    "segment_geometry_fallback"
                ]
            }
        });
    }

    fn attach_fixture_neural_diarization(report: &mut RunReport) {
        attach_fixture_diarization(report);
        report
            .request
            .backend_params
            .acoustic_diarization
            .as_mut()
            .expect("fixture diarization request")
            .engine = DiarizationEngine::EcapaFused;
        let neural_request = report
            .request
            .backend_params
            .acoustic_diarization
            .as_ref()
            .expect("fixture neural diarization request");
        let diarization = report
            .result
            .diarization
            .as_mut()
            .expect("fixture diarization report");
        diarization.implementation = "native-ecapa-fused-v1".to_owned();
        diarization.contract_version =
            crate::diarization::NEURAL_DIARIZATION_CONTRACT_VERSION.to_owned();
        diarization.feature_schema = "acoustic-feature-v2".to_owned();
        diarization.speaker_evidence_mode =
            crate::model::DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel;
        let fused_calibration_sha256 = crate::diarization::ecapa_speaker_pair_calibration_sha256(
            crate::model::DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel,
        );
        diarization
            .speaker_count
            .estimate
            .as_mut()
            .expect("fixture speaker count estimate")
            .calibration_sha256 = fused_calibration_sha256.clone();
        diarization.hint_document_sha256 =
            Some(crate::model::speaker_hint_document_sha256(neural_request));
        let operational_partition = diarization
            .operational_partition
            .as_mut()
            .expect("fixture neural operational partition");
        operational_partition.method = DiarizationOperationalPartitionMethod::EcapaFusedConsensus;
        operational_partition.calibration_sha256 = fused_calibration_sha256;
        diarization.neural_representation = Some(NeuralSpeakerRepresentationSummary {
            schema_version: "neural-speaker-representation-summary-v1".to_owned(),
            provider_version: crate::diarization::ECAPA_SPEAKER_REPRESENTATION_VERSION.to_owned(),
            expected_model_package_sha256: crate::ecapa_conformance::ECAPA_PACKAGE_SHA256
                .to_owned(),
            loaded_model_package_sha256: Some(
                crate::ecapa_conformance::ECAPA_PACKAGE_SHA256.to_owned(),
            ),
            model_load_source: Some(crate::model::NeuralModelLoadSource::PackageVerified),
            status: NeuralSpeakerRepresentationStatus::Degraded,
            embedded_tracklet_count: 3,
            zero_padded_tracklet_count: 1,
            skipped_tracklet_count: 1,
            reasons: vec![NeuralSpeakerRepresentationReason::ShortTracklet],
        });
    }

    fn synthetic_speaker_count_estimate() -> SpeakerCountEstimate {
        let first_probability = 0.7_f64;
        let second_probability = 0.15_f64;
        let unresolved_probability = 0.15_f64;
        let estimate = SpeakerCountEstimate {
            schema_version: "speaker-count-estimate-v2".to_owned(),
            selected_count: Some(1),
            supported_range: Some(SpeakerCountRange {
                minimum: 1,
                maximum: 2,
            }),
            posterior: vec![
                SpeakerCountPosteriorBin {
                    count: 1,
                    probability: first_probability,
                },
                SpeakerCountPosteriorBin {
                    count: 2,
                    probability: second_probability,
                },
            ],
            unresolved_probability,
            entropy_bits: -first_probability * first_probability.log2()
                - second_probability * second_probability.log2()
                - unresolved_probability * unresolved_probability.log2(),
            stability: 0.8,
            constraint_lower_bound: 1,
            candidate_upper_bound: 2,
            calibration_status: SpeakerCountCalibrationStatus::DevelopmentUncertified,
            calibration_sha256: crate::diarization::acoustic_speaker_pair_calibration_sha256(),
            evidence_sha256: "d".repeat(64),
            lanes: vec![
                SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::MergeRisk,
                    available: true,
                    proposed_count: Some(1),
                    confidence: 0.8,
                    unavailable_reason: None,
                },
                SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::SparseNormalizedEigengap,
                    available: true,
                    proposed_count: Some(1),
                    confidence: 0.65,
                    unavailable_reason: None,
                },
                SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::FeatureJackknife,
                    available: true,
                    proposed_count: Some(1),
                    confidence: 0.8,
                    unavailable_reason: None,
                },
                SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::EffectiveOccupancy,
                    available: true,
                    proposed_count: Some(1),
                    confidence: 0.88,
                    unavailable_reason: None,
                },
                SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::ConstraintGraph,
                    available: true,
                    proposed_count: Some(1),
                    confidence: 1.0,
                    unavailable_reason: None,
                },
                SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::CallerPrior,
                    available: true,
                    proposed_count: Some(1),
                    confidence: 1.0,
                    unavailable_reason: None,
                },
            ],
            resources: SpeakerCountResourceSummary {
                prototype_count: 2,
                affinity_pair_evaluations: 2,
                retained_sparse_edges: 1,
                estimated_peak_buffer_bytes: 512,
                stability_replicates: 5,
                solver_iterations: 8,
                solver_sparse_matvec_terms: 128,
                solver_residual: Some(1.0e-8),
            },
        };
        estimate.validate().expect("synthetic count estimate");
        estimate
    }

    fn test_cursor(
        connection: &Connection,
        last_mutation_seq: i64,
        ts: &str,
        run_id: Option<&str>,
    ) -> SyncCursor {
        SyncCursor {
            last_mutation_seq,
            database_id: Some(
                load_mutation_authority(connection)
                    .expect("load mutation authority")
                    .database_id,
            ),
            last_export_rfc3339: ts.to_owned(),
            last_export_run_id: run_id.map(str::to_owned),
            last_run_count: 0,
        }
    }

    fn run_mutation_seq(connection: &Connection, run_id: &str) -> i64 {
        connection
            .query_with_params(
                "SELECT mutation_seq FROM sync_run_mutations WHERE run_id = ?1",
                &[SqliteValue::Text(run_id.to_owned().into())],
            )
            .expect("query run mutation sequence")
            .first()
            .and_then(|row| row.get(0).and_then(|value| value.as_integer()))
            .expect("run mutation sequence")
    }

    fn mutation_high_water(connection: &Connection) -> i64 {
        connection
            .query("SELECT COALESCE(MAX(mutation_seq), 0) FROM sync_run_mutations")
            .expect("query mutation high-water mark")
            .first()
            .and_then(|row| row.get(0).and_then(|value| value.as_integer()))
            .expect("mutation high-water mark")
    }

    #[test]
    fn export_empty_db_produces_valid_manifest() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        // Open DB to create schema
        let _store = RunStore::open(&db_path).expect("store open");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export should succeed");

        assert_eq!(manifest.schema_version, "1.1");
        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);

        // Verify files exist
        assert!(export_dir.join("manifest.json").exists());
        assert!(export_dir.join("runs.jsonl").exists());
        assert!(export_dir.join("segments.jsonl").exists());
        assert!(export_dir.join("events.jsonl").exists());
    }

    #[test]
    fn import_rejects_parent_result_backend_mismatch_before_persisting_run() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("incoherent-export");
        let fixture_db = dir.path().join("fixture.sqlite3");
        let report = fixture_report("sync-backend-mismatch", &fixture_db);
        let run_row = json!({
            "id": report.run_id,
            "started_at": report.started_at_rfc3339,
            "finished_at": report.finished_at_rfc3339,
            "backend": "native-listen",
            "input_path": report.input_path,
            "normalized_wav_path": report.normalized_wav_path,
            "request_json": serde_json::to_string(&report.request).expect("request json"),
            "result_json": serde_json::to_string(&report.result).expect("result json"),
            "warnings_json": serde_json::to_string(&report.warnings).expect("warnings json"),
            "transcript": report.result.transcript,
            "replay_json": serde_json::to_string(&report.replay).expect("replay json"),
            "acceleration_json": "{}",
        });
        write_jsonl_snapshot(&export_dir, &[run_row], &[], &[]);

        for batch_import_enabled in [false, true] {
            let db_path = dir.path().join(format!("import-{batch_import_enabled}.sqlite3"));
            let error = import_inner_with_batch_mode(
                &db_path,
                &export_dir,
                ConflictPolicy::Reject,
                batch_import_enabled,
            )
            .expect_err("incoherent parent/result backends must fail import");
            let message = error.to_string();
            assert!(
                message.contains("sync-backend-mismatch"),
                "import error omitted run id: {message}"
            );
            assert!(
                message.contains("backend mismatch"),
                "unexpected import error: {message}"
            );

            let store = RunStore::open(&db_path).expect("open rolled-back import database");
            assert!(
                store.list_recent_runs(1).expect("list runs").is_empty(),
                "incoherent run must not survive the failed import"
            );
        }
    }

    #[test]
    fn full_export_uses_one_snapshot_across_parent_and_child_queries() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("full_snapshot.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("full-snapshot-first", &db_path))
            .expect("persist first report");

        let (runs_exported_tx, runs_exported_rx) = std::sync::mpsc::sync_channel(0);
        let (writer_committed_tx, writer_committed_rx) = std::sync::mpsc::sync_channel(0);
        let export_db_path = db_path.clone();
        let export_output_dir = export_dir.clone();
        let export_thread = std::thread::spawn(move || {
            export_inner_with_after_runs(&export_db_path, &export_output_dir, || {
                runs_exported_tx.send(()).expect("signal runs exported");
                writer_committed_rx
                    .recv()
                    .expect("wait for concurrent writer");
            })
        });

        runs_exported_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("wait for runs export");
        store
            .persist_report(&fixture_report("full-snapshot-second", &db_path))
            .expect("persist concurrent report");
        writer_committed_tx
            .send(())
            .expect("release snapshot export");

        let manifest = export_thread
            .join()
            .expect("snapshot export thread")
            .expect("snapshot export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 2);

        for (file_name, run_id_key) in [
            ("runs.jsonl", "id"),
            ("segments.jsonl", "run_id"),
            ("events.jsonl", "run_id"),
        ] {
            let rows: Vec<serde_json::Value> = fs::read_to_string(export_dir.join(file_name))
                .expect("read snapshot file")
                .lines()
                .map(|line| serde_json::from_str(line).expect("parse snapshot row"))
                .collect();
            assert!(!rows.is_empty(), "{file_name} should contain first run rows");
            assert!(
                rows.iter()
                    .all(|row| row[run_id_key] == "full-snapshot-first"),
                "{file_name} must wholly exclude the concurrent run"
            );
        }
    }

    #[test]
    fn export_import_round_trip() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        // Persist a report to source DB
        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("run-sync-1", &db_path);
        store.persist_report(&report).expect("persist");

        // Export
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 2);

        // Import into a fresh DB
        let target_db = dir.path().join("target.sqlite3");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");

        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
        assert!(result.conflicts.is_empty());

        // Verify data in target DB
        let target_store = RunStore::open(&target_db).expect("target store");
        let runs = target_store.list_recent_runs(10).expect("list runs");
        assert_eq!(runs.len(), 1);
        assert!(runs[0].transcript_preview.contains("hello world"));
        let details = target_store
            .load_run_details("run-sync-1")
            .expect("load")
            .expect("details");
        assert_eq!(
            details.replay.input_content_hash.as_deref(),
            Some("sync-input-hash")
        );
        assert_eq!(
            details.replay.backend_identity.as_deref(),
            Some("whisper-cli")
        );
        assert_eq!(
            details.replay.backend_version.as_deref(),
            Some("whisper 1.2.3")
        );
        assert_eq!(
            details.replay.output_payload_hash.as_deref(),
            Some("sync-output-hash")
        );
    }

    #[test]
    fn diarization_round_trip_rebuilds_normalized_index() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("source.sqlite3");
        let target_db = dir.path().join("target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let mut report = fixture_report("run-diarization-sync", &source_db);
        attach_fixture_diarization(&mut report);
        source_store
            .persist_report(&report)
            .expect("diarization report should persist");

        export(&source_db, &export_dir, &state_root).expect("export");
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");

        let target =
            BlockingConnection::open(target_db.display().to_string()).expect("target connection");
        for table in [
            "diarization_reports",
            "diarization_turns",
            "speaker_hints",
            "speaker_profile_summaries",
        ] {
            let rows = target
                .query(&format!("SELECT COUNT(*) FROM {table};"))
                .expect("normalized index count");
            assert_eq!(
                value_to_string_sqlite(rows[0].get(0)),
                "1",
                "{table} should be rebuilt from canonical runs JSONL"
            );
        }

        let canonical = target
            .query(
                "SELECT request_json, result_json FROM runs \
                 WHERE id = 'run-diarization-sync';",
            )
            .expect("canonical run row");
        let request: TranscribeRequest =
            serde_json::from_str(&value_to_string_sqlite(canonical[0].get(0)))
                .expect("typed request should survive JSONL");
        let result: TranscriptionResult =
            serde_json::from_str(&value_to_string_sqlite(canonical[0].get(1)))
                .expect("typed result should survive JSONL");
        assert_eq!(
            request
                .backend_params
                .acoustic_diarization
                .as_ref()
                .map(|request| request.engine),
            Some(DiarizationEngine::Acoustic)
        );
        assert_eq!(
            result
                .diarization
                .as_ref()
                .map(|report| report.implementation.as_str()),
            Some("native-acoustic-v2")
        );
        let recovered_request = request
            .backend_params
            .acoustic_diarization
            .as_ref()
            .expect("typed diarization request");
        assert_eq!(
            recovered_request.speaker_count,
            SpeakerCountRequest::Range {
                minimum: 1,
                maximum: 2
            }
        );
        let recovered_report = result
            .diarization
            .as_ref()
            .expect("typed diarization result");
        assert_eq!(
            recovered_report.speaker_count.status,
            SpeakerCountOutcomeStatus::Resolved
        );
        assert_eq!(
            recovered_report.speaker_count.reasons,
            vec![SpeakerCountOutcomeReason::EvidenceSupportedCount]
        );
        let recovered_estimate = recovered_report
            .speaker_count
            .estimate
            .as_ref()
            .expect("speaker-count estimate should survive JSONL recovery");
        recovered_estimate
            .validate()
            .expect("recovered speaker-count estimate should remain valid");
        assert_eq!(
            recovered_estimate.schema_version,
            "speaker-count-estimate-v2"
        );
        assert_eq!(recovered_estimate.selected_count, Some(1));
        assert_eq!(recovered_estimate.posterior.len(), 2);
        assert_eq!(recovered_estimate.resources.prototype_count, 2);
        assert_eq!(recovered_estimate.resources.retained_sparse_edges, 1);
        assert_eq!(recovered_estimate.resources.solver_residual, Some(1.0e-8));
        assert_eq!(
            recovered_report.hint_evidence[0].disposition,
            SpeakerHintDisposition::PartiallyAccepted
        );
        assert!(
            !value_to_string_sqlite(canonical[0].get(1)).contains("feature_vector"),
            "canonical JSONL payload must not acquire reusable biometric vectors"
        );

        let recovered = RunStore::open(&target_db)
            .expect("recovered store")
            .load_run_details("run-diarization-sync")
            .expect("recovered details query")
            .expect("recovered details");
        assert_eq!(
            recovered
                .diarization
                .as_ref()
                .map(|report| report.turns.as_slice()),
            report
                .result
                .diarization
                .as_ref()
                .map(|report| report.turns.as_slice()),
            "typed speaker turns must survive SQLite -> JSONL -> fresh SQLite"
        );
        let recovered_entropy = recovered
            .diarization
            .as_ref()
            .and_then(|report| report.speaker_count.estimate.as_ref())
            .map(|estimate| estimate.entropy_bits)
            .expect("recovered speaker-count entropy");
        let mut expected_diarization = report.result.diarization.clone();
        let expected_entropy = expected_diarization
            .as_ref()
            .and_then(|report| report.speaker_count.estimate.as_ref())
            .map(|estimate| estimate.entropy_bits)
            .expect("expected speaker-count entropy");
        let entropy_tolerance = 4.0 * f64::EPSILON * expected_entropy.abs().max(1.0);
        assert!(
            (recovered_entropy - expected_entropy).abs() <= entropy_tolerance,
            "derived speaker-count entropy must survive JSON numeric normalization within \
             floating-point roundoff: recovered={recovered_entropy}, expected={expected_entropy}"
        );
        expected_diarization
            .as_mut()
            .and_then(|report| report.speaker_count.estimate.as_mut())
            .expect("expected speaker-count estimate")
            .entropy_bits = recovered_entropy;
        assert_eq!(
            recovered.diarization, expected_diarization,
            "count mode, reasons, occupancy evidence, hint dispositions, and fallback status must \
             survive SQLite -> JSONL -> fresh SQLite exactly apart from derived-float roundoff"
        );
        assert_eq!(
            recovered
                .projection_timeline
                .as_ref()
                .expect("recovered projection timeline"),
            &report.result.raw_output["projection_timeline"],
            "projection provenance must survive recovery without drift"
        );
        assert!(
            !serde_json::to_string(&recovered.projection_timeline)
                .expect("serialize recovered projection timeline")
                .contains("feature_vector"),
            "recovered projection metadata must remain privacy-safe"
        );
    }

    #[test]
    fn neural_diarization_round_trip_rebuilds_normalized_index() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("neural-source.sqlite3");
        let target_db = dir.path().join("neural-target.sqlite3");
        let export_dir = dir.path().join("neural-export");
        let state_root = dir.path().join("neural-state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let mut report = fixture_report("run-neural-diarization-sync", &source_db);
        attach_fixture_neural_diarization(&mut report);
        source_store
            .persist_report(&report)
            .expect("neural diarization report should persist");

        export(&source_db, &export_dir, &state_root).expect("export");
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");

        let target =
            BlockingConnection::open(target_db.display().to_string()).expect("target connection");
        let derived = target
            .query(
                "SELECT implementation, contract_version, feature_schema, \
                 supported_speaker_count FROM diarization_reports \
                 WHERE run_id = 'run-neural-diarization-sync';",
            )
            .expect("normalized neural report");
        assert_eq!(derived.len(), 1);
        assert_eq!(
            value_to_string_sqlite(derived[0].get(0)),
            "native-ecapa-fused-v1"
        );
        assert_eq!(
            value_to_string_sqlite(derived[0].get(1)),
            crate::diarization::NEURAL_DIARIZATION_CONTRACT_VERSION
        );
        assert_eq!(
            value_to_string_sqlite(derived[0].get(2)),
            "acoustic-feature-v2"
        );
        assert_eq!(value_to_string_sqlite(derived[0].get(3)), "1");
        for table in [
            "diarization_turns",
            "speaker_hints",
            "speaker_profile_summaries",
        ] {
            let rows = target
                .query(&format!(
                    "SELECT COUNT(*) FROM {table} \
                     WHERE run_id = 'run-neural-diarization-sync';"
                ))
                .expect("normalized neural index count");
            assert_eq!(
                value_to_string_sqlite(rows[0].get(0)),
                "1",
                "{table} should be rebuilt for the neural contract"
            );
        }

        let canonical = target
            .query(
                "SELECT request_json, result_json FROM runs \
                 WHERE id = 'run-neural-diarization-sync';",
            )
            .expect("canonical neural run row");
        let request: TranscribeRequest =
            serde_json::from_str(&value_to_string_sqlite(canonical[0].get(0)))
                .expect("typed neural request should survive JSONL");
        let result: TranscriptionResult =
            serde_json::from_str(&value_to_string_sqlite(canonical[0].get(1)))
                .expect("typed neural result should survive JSONL");
        assert_eq!(
            request
                .backend_params
                .acoustic_diarization
                .as_ref()
                .map(|request| request.engine),
            Some(DiarizationEngine::EcapaFused)
        );
        let neural_summary = result
            .diarization
            .as_ref()
            .and_then(|diarization| diarization.neural_representation.as_ref())
            .expect("neural provenance should survive JSONL");
        assert_eq!(
            neural_summary.status,
            NeuralSpeakerRepresentationStatus::Degraded
        );
        assert_eq!(neural_summary.embedded_tracklet_count, 3);
        assert_eq!(
            neural_summary.reasons,
            vec![NeuralSpeakerRepresentationReason::ShortTracklet]
        );
        assert_eq!(
            result
                .diarization
                .as_ref()
                .and_then(|diarization| diarization.operational_partition.as_ref())
                .map(|partition| partition.method),
            Some(DiarizationOperationalPartitionMethod::EcapaFusedConsensus),
            "typed fused partition provenance must survive SQLite -> JSONL -> fresh SQLite",
        );
        assert!(
            !value_to_string_sqlite(canonical[0].get(1)).contains("feature_vector"),
            "canonical neural JSONL must not contain reusable biometric vectors"
        );

        let recovered = RunStore::open(&target_db)
            .expect("recovered neural store")
            .load_run_details("run-neural-diarization-sync")
            .expect("recovered neural details query")
            .expect("recovered neural details");
        assert_eq!(
            recovered
                .diarization
                .as_ref()
                .and_then(|diarization| diarization.neural_representation.as_ref()),
            report
                .result
                .diarization
                .as_ref()
                .and_then(|diarization| diarization.neural_representation.as_ref()),
            "typed neural provenance must survive SQLite -> JSONL -> fresh SQLite"
        );
    }

    #[test]
    fn diarization_rebuild_failure_rolls_back_canonical_import() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("source.sqlite3");
        let target_db = dir.path().join("target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let mut report = fixture_report("run-invalid-diarization-import", &source_db);
        attach_fixture_diarization(&mut report);
        source_store.persist_report(&report).expect("persist");
        export(&source_db, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let line = fs::read_to_string(&runs_path).expect("runs jsonl");
        let mut row: serde_json::Value =
            serde_json::from_str(line.trim()).expect("canonical run row");
        let mut result: serde_json::Value = serde_json::from_str(
            row["result_json"]
                .as_str()
                .expect("embedded result JSON string"),
        )
        .expect("embedded result JSON");
        result["diarization"]["contract_version"] = json!("acoustic-diarization-v999");
        row["result_json"] = json!(serde_json::to_string(&result).expect("serialize result"));
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&row).expect("serialize row")),
        )
        .expect("rewrite runs JSONL");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
                .expect("manifest JSON");
        manifest["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("runs checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("rewrite manifest");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("invalid derived payload must abort the whole import");
        assert!(
            error
                .to_string()
                .contains("cannot rebuild diarization report")
        );
        assert!(
            error.to_string().contains(
                "expected exactly acoustic-diarization-v3 or neural-diarization-common-v2"
            ),
            "unknown acoustic contract versions must remain fail-closed: {error}"
        );

        let target =
            BlockingConnection::open(target_db.display().to_string()).expect("target connection");
        let rows = target
            .query("SELECT COUNT(*) FROM runs;")
            .expect("canonical run count");
        assert_eq!(
            value_to_string_sqlite(rows[0].get(0)),
            "0",
            "canonical rows must roll back with the derived-index rebuild"
        );
    }

    #[test]
    fn import_detects_checksum_mismatch() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt a file
        fs::write(export_dir.join("runs.jsonl"), "corrupted data\n").expect("write");

        let target_db = dir.path().join("target.sqlite3");
        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("checksum mismatch"));
    }

    #[test]
    fn import_duplicate_noop_on_identical_payload() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("run-dup-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Import back into the same DB (all rows already exist, identical)
        let result = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import same data");

        assert_eq!(result.runs_imported, 1);
        assert!(
            result.conflicts.is_empty(),
            "expected no conflicts, got: {:?}",
            result.conflicts
        );
    }

    #[test]
    fn import_reject_policy_fails_on_conflicting_payload_and_writes_conflicts() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("run-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let first_line = fs::read_to_string(&runs_path)
            .expect("runs.jsonl should be readable")
            .lines()
            .next()
            .expect("runs.jsonl should contain first line")
            .to_owned();
        let mut mutated: serde_json::Value =
            serde_json::from_str(&first_line).expect("valid runs line");
        mutated["transcript"] = serde_json::json!("mutated transcript payload");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("runs.jsonl overwrite should succeed");

        let manifest = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            serde_json::json!(sha256_file(&runs_path).expect("checksum"));
        fs::write(
            &manifest,
            serde_json::to_string_pretty(&manifest_value).expect("serialize manifest"),
        )
        .expect("manifest rewrite should succeed");

        let result = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject);
        assert!(result.is_err());
        let conflicts_path = export_dir.join("sync_conflicts.jsonl");
        assert!(conflicts_path.exists());
        let conflicts = fs::read_to_string(conflicts_path).expect("conflicts should be readable");
        assert!(conflicts.contains("\"table\":\"runs\""));
    }

    #[test]
    fn import_overwrite_policy_replaces_conflicting_payload() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("run-overwrite-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let first_line = fs::read_to_string(&runs_path)
            .expect("runs.jsonl should be readable")
            .lines()
            .next()
            .expect("runs.jsonl should contain first line")
            .to_owned();
        let mut mutated: serde_json::Value =
            serde_json::from_str(&first_line).expect("valid runs line");
        mutated["transcript"] = serde_json::json!("overwrite transcript");
        let mut result_json: serde_json::Value = serde_json::from_str(
            mutated["result_json"]
                .as_str()
                .expect("result_json string should be present"),
        )
        .expect("result_json should parse");
        result_json["transcript"] = serde_json::json!("overwrite transcript");
        mutated["result_json"] =
            serde_json::json!(serde_json::to_string(&result_json).expect("serialize result json"));
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("runs.jsonl overwrite should succeed");

        let manifest = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            serde_json::json!(sha256_file(&runs_path).expect("checksum"));
        fs::write(
            &manifest,
            serde_json::to_string_pretty(&manifest_value).expect("serialize manifest"),
        )
        .expect("manifest rewrite should succeed");

        let result = import(
            &db_path,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("overwrite import should succeed");
        assert_eq!(result.runs_imported, 1);

        let store = RunStore::open(&db_path).expect("store open");
        let run = store
            .load_run_details("run-overwrite-1")
            .expect("load should succeed")
            .expect("run should exist");
        assert_eq!(run.transcript, "overwrite transcript");
    }

    #[test]
    fn lock_prevents_concurrent_sync() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");

        let lock = SyncLock::acquire(&state_root, "test").expect("first lock");

        // Second acquire should fail
        let second = SyncLock::acquire(&state_root, "test");
        assert!(second.is_err());

        lock.release().expect("release");

        // Now should succeed
        let third = SyncLock::acquire(&state_root, "test").expect("after release");
        third.release().expect("release");
    }

    #[test]
    fn export_error_releases_lock_via_drop_guard() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("missing-schema.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let result = export(&db_path, &export_dir, &state_root);
        assert!(result.is_err(), "export should fail when schema is absent");

        let lock_path = state_root.join("locks/sync.lock");
        assert!(
            !lock_path.exists(),
            "lock file should be removed even when export fails"
        );
    }

    #[test]
    fn fresh_corrupt_lock_fails_closed_without_archiving() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");
        let lock_path = locks_dir.join("sync.lock");
        fs::write(&lock_path, "{not-json").expect("write corrupt lock");

        let error = SyncLock::acquire(&state_root, "test")
            .err()
            .expect("fresh corrupt lock must block acquisition");
        assert!(
            error.to_string().contains("potentially active partial lock"),
            "unexpected error: {error}"
        );
        assert_eq!(
            fs::read_to_string(&lock_path).expect("fresh corrupt lock remains"),
            "{not-json"
        );

        let archived = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .any(|name| name.starts_with("sync.lock.corrupt."));
        assert!(!archived, "fresh corrupt lock must not be archived");
    }

    #[test]
    fn fresh_non_utf8_lock_fails_closed_without_archiving() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");
        let lock_path = locks_dir.join("sync.lock");
        let lock_bytes = [0xff, 0xfe, 0xfd];
        fs::write(&lock_path, lock_bytes).expect("write non-utf8 lock");

        let error = SyncLock::acquire(&state_root, "test")
            .err()
            .expect("fresh non-UTF-8 lock must block acquisition");
        assert!(
            error.to_string().contains("potentially active partial lock"),
            "unexpected error: {error}"
        );
        assert_eq!(
            fs::read(&lock_path).expect("fresh non-UTF-8 lock remains"),
            lock_bytes
        );

        let archived = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .any(|name| name.starts_with("sync.lock.corrupt."));
        assert!(!archived, "fresh non-UTF-8 lock must not be archived");
    }

    #[test]
    fn stale_corrupt_lock_is_archived_and_replaced() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");
        let lock_path = locks_dir.join("sync.lock");
        fs::write(&lock_path, "{not-json").expect("write corrupt lock");

        let modified = fs::metadata(&lock_path)
            .and_then(|metadata| metadata.modified())
            .expect("lock modified time");
        let after_stale_threshold = modified
            .checked_add(std::time::Duration::from_secs(
                LOCK_STALE_SECONDS as u64 + 1,
            ))
            .expect("representable stale time");
        assert!(
            read_lock_info_at(&lock_path, after_stale_threshold)
                .expect("stale corrupt lock should be recoverable")
                .is_none()
        );

        let lock = SyncLock::acquire(&state_root, "test").expect("acquire after recovery");
        lock.release().expect("release");

        let archived = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .any(|name| name.starts_with("sync.lock.corrupt."));
        assert!(archived, "stale corrupt lock should be archived");
    }

    #[test]
    fn partial_lock_creation_window_fails_closed() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let first_state_root = state_root.clone();
        let (created_tx, created_rx) = std::sync::mpsc::sync_channel(0);
        let (continue_tx, continue_rx) = std::sync::mpsc::sync_channel(0);

        let first = std::thread::spawn(move || {
            SyncLock::acquire_with_after_create(&first_state_root, "first", || {
                created_tx.send(()).expect("signal lock creation");
                continue_rx
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .expect("release partial-lock barrier");
            })
        });

        created_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("first acquisition should create lock");
        let lock_path = state_root.join("locks/sync.lock");
        let second = SyncLock::acquire(&state_root, "second");
        let partial_lock_remained = lock_path.exists();
        let archived_during_race = fs::read_dir(lock_path.parent().expect("locks dir"))
            .expect("read locks dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .any(|name| name.starts_with("sync.lock.corrupt."));

        continue_tx.send(()).expect("continue first acquisition");
        let first_lock = first
            .join()
            .expect("first acquisition thread should not panic")
            .expect("first acquisition should complete");
        first_lock.release().expect("release first lock");

        let error = second
            .err()
            .expect("fresh partial lock must block second acquisition");
        assert!(
            error.to_string().contains("potentially active partial lock"),
            "unexpected error: {error}"
        );
        assert!(
            partial_lock_remained,
            "contender must not rename the first acquisition's partial lock"
        );
        assert!(
            !archived_during_race,
            "contender must not archive the first acquisition's partial lock"
        );
    }

    #[test]
    fn stale_lock_is_archived_before_new_lock_is_granted() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");

        let stale = LockInfo {
            pid: u32::MAX - 1,
            created_at_rfc3339: "2000-01-01T00:00:00Z".to_owned(),
            operation: "export".to_owned(),
        };
        fs::write(
            locks_dir.join("sync.lock"),
            serde_json::to_string_pretty(&stale).expect("serialize"),
        )
        .expect("write stale lock");

        let lock = SyncLock::acquire(&state_root, "import").expect("acquire should recover");
        lock.release().expect("release");

        let archived = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .any(|name| name.starts_with("sync.lock.stale."));
        assert!(archived, "stale lock should be archived");
    }

    #[test]
    fn import_rejects_segment_row_with_missing_parent_run() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(
            &export_dir,
            &[],
            &[json!({
                "run_id": "missing-run",
                "idx": 0,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "speaker": serde_json::Value::Null,
                "text": "orphan segment",
                "confidence": 0.9,
            })],
            &[],
        );

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("import should fail on orphan segment reference");
        let error_text = error.to_string();
        assert!(error_text.contains("referential integrity violation"));
        assert!(error_text.contains("segments"));
        assert!(error_text.contains("missing-run"));
    }

    #[test]
    fn import_rejects_event_row_with_missing_parent_run() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(
            &export_dir,
            &[],
            &[],
            &[json!({
                "run_id": "missing-run",
                "seq": 1,
                "ts_rfc3339": "2026-01-01T00:00:01Z",
                "stage": "backend",
                "code": "backend.ok",
                "message": "completed",
                "payload_json": "{}",
            })],
        );

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("import should fail on orphan event reference");
        let error_text = error.to_string();
        assert!(error_text.contains("referential integrity violation"));
        assert!(error_text.contains("events"));
        assert!(error_text.contains("missing-run"));
    }

    #[test]
    fn import_rejects_major_schema_version_mismatch() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(&export_dir, &[], &[], &[]);

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["schema_version"] = json!("2.0");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("major schema mismatch must fail");
        assert!(error.to_string().contains("schema version mismatch"));
    }

    #[test]
    fn import_rejects_major_export_format_version_mismatch() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(&export_dir, &[], &[], &[]);

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["export_format_version"] = json!("2.0");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("major export format mismatch must fail");
        assert!(error.to_string().contains("export format version mismatch"));
    }

    #[test]
    fn import_accepts_backward_compatible_minor_schema_version() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(&export_dir, &[], &[], &[]);

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["schema_version"] = json!("1.7");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("minor version should be accepted");
        assert_eq!(result.runs_imported, 0);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 0);
    }

    #[test]
    fn import_accepts_backward_compatible_minor_export_format_version() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(&export_dir, &[], &[], &[]);

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["export_format_version"] = json!("1.5");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("minor export format version should be accepted");
        assert_eq!(result.runs_imported, 0);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 0);
    }

    #[test]
    fn import_row_count_mismatch_errors_when_no_conflicts_present() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("row-mismatch-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["row_counts"]["events"] = json!(9999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        // keep checksum valid by not mutating data files.
        let target_db = dir.path().join("target.sqlite3");
        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("row mismatch should fail");
        assert!(error.to_string().contains("row count mismatch for events"));
    }

    #[test]
    fn import_row_count_mismatch_is_deferred_when_conflicts_exist() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("row-mismatch-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let first_line = fs::read_to_string(&runs_path)
            .expect("runs.jsonl should be readable")
            .lines()
            .next()
            .expect("runs.jsonl should contain first line")
            .to_owned();
        let mut mutated: serde_json::Value =
            serde_json::from_str(&first_line).expect("valid runs line");
        mutated["transcript"] = json!("mutated transcript payload");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("runs write");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("runs checksum"));
        manifest_value["row_counts"]["events"] = json!(9999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        let error = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("conflict should fail before row-count mismatch");
        let text = error.to_string();
        assert!(text.contains("import rejected due to"));
        assert!(!text.contains("row count mismatch"));
    }

    #[test]
    fn import_surfaces_error_when_conflicts_file_cannot_be_written() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("conflict-write-fail-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let first_line = fs::read_to_string(&runs_path)
            .expect("runs.jsonl should be readable")
            .lines()
            .next()
            .expect("runs.jsonl should contain first line")
            .to_owned();
        let mut mutated: serde_json::Value =
            serde_json::from_str(&first_line).expect("valid runs line");
        mutated["transcript"] = json!("mutated transcript payload");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("runs write");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest read"))
                .expect("manifest json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("runs checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("manifest serialize"),
        )
        .expect("manifest write");

        // Deterministically block conflict artifact creation by occupying
        // the expected file path with a directory.
        let conflict_path = export_dir.join("sync_conflicts.jsonl");
        fs::create_dir(&conflict_path).expect("create conflict path directory");

        let result = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject);

        let error = result.expect_err("conflict artifact write should fail");
        let text = error.to_string();
        assert!(
            text.contains("directory")
                || text.contains("Directory")
                || text.contains("sync_conflicts.jsonl"),
            "unexpected error: {text}"
        );
    }

    #[test]
    fn stale_lock_archive_name_follows_deterministic_pattern() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");

        let stale = LockInfo {
            pid: u32::MAX - 1,
            created_at_rfc3339: "2000-01-01T00:00:00Z".to_owned(),
            operation: "export".to_owned(),
        };
        fs::write(
            locks_dir.join("sync.lock"),
            serde_json::to_string_pretty(&stale).expect("serialize"),
        )
        .expect("write stale lock");

        let _lock = SyncLock::acquire(&state_root, "test").expect("acquire");

        let archived: Vec<String> = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("sync.lock.stale."))
            .collect();

        assert_eq!(archived.len(), 1, "exactly one stale archive should exist");
        // Pattern: sync.lock.stale.{unix_timestamp}.json
        let name = &archived[0];
        assert!(name.ends_with(".json"), "archive should end with .json");
        let parts: Vec<&str> = name.split('.').collect();
        assert_eq!(parts.len(), 5, "expected sync.lock.stale.<ts>.json");
        assert!(
            parts[3].parse::<u64>().is_ok(),
            "timestamp part should be a valid u64"
        );
    }

    #[test]
    fn double_export_produces_consistent_manifest() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("double-export-1", &db_path);
        store.persist_report(&report).expect("persist");

        let export_dir_1 = dir.path().join("export1");
        let manifest_1 = export(&db_path, &export_dir_1, &state_root).expect("export 1");

        let export_dir_2 = dir.path().join("export2");
        let manifest_2 = export(&db_path, &export_dir_2, &state_root).expect("export 2");

        assert_eq!(manifest_1.row_counts, manifest_2.row_counts);
        assert_eq!(manifest_1.schema_version, manifest_2.schema_version);
        assert_eq!(manifest_1.checksums, manifest_2.checksums);
    }

    // -----------------------------------------------------------------------
    // Helper function unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn is_lock_stale_returns_true_for_old_dead_pid() {
        let info = LockInfo {
            pid: u32::MAX - 1, // Almost certainly doesn't exist
            created_at_rfc3339: "2000-01-01T00:00:00Z".to_owned(),
            operation: "test".to_owned(),
        };
        assert!(is_lock_stale(&info));
    }

    #[test]
    fn is_lock_stale_returns_false_for_alive_pid_even_if_ancient() {
        // Alive PID should keep lock even if timestamp is ancient.
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: "2000-01-01T00:00:00Z".to_owned(),
            operation: "test".to_owned(),
        };
        assert!(!is_lock_stale(&info));
    }

    #[test]
    fn is_lock_stale_returns_false_for_current_pid_and_recent_time() {
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            operation: "test".to_owned(),
        };
        assert!(!is_lock_stale(&info));
    }

    #[test]
    fn pid_is_alive_current_process_returns_true() {
        assert!(pid_is_alive(std::process::id()));
    }

    #[test]
    fn optional_floats_equal_both_none() {
        assert!(optional_floats_equal(None, None));
    }

    #[test]
    fn optional_floats_equal_both_same() {
        assert!(optional_floats_equal(Some(1.5), Some(1.5)));
    }

    #[test]
    fn optional_floats_equal_mismatched_some_none() {
        assert!(!optional_floats_equal(Some(1.0), None));
        assert!(!optional_floats_equal(None, Some(1.0)));
    }

    #[test]
    fn optional_floats_equal_within_epsilon() {
        assert!(optional_floats_equal(Some(1.0), Some(1.0 + 1e-10)));
    }

    #[test]
    fn optional_floats_equal_outside_epsilon() {
        assert!(!optional_floats_equal(Some(1.0), Some(1.0 + 1e-8)));
    }

    #[test]
    fn json_str_returns_error_on_missing_key() {
        let value = json!({"a": "hello"});
        let err = json_str(&value, "missing").expect_err("missing key");
        assert!(err.to_string().contains("missing string field"));
    }

    #[test]
    fn json_string_or_default_uses_fallback() {
        let value = json!({"a": "hello"});
        assert_eq!(json_string_or_default(&value, "a", "def"), "hello");
        assert_eq!(json_string_or_default(&value, "missing", "def"), "def");
    }

    #[test]
    fn value_to_json_all_variants() {
        use serde_json::Value as JV;
        assert_eq!(
            value_to_json(Some(&SqliteValue::Text("hi".to_owned().into()))),
            JV::String("hi".to_owned())
        );
        assert_eq!(value_to_json(Some(&SqliteValue::Integer(42))), json!(42));
        assert_eq!(value_to_json(Some(&SqliteValue::Float(1.5))), json!(1.5));
        assert!(matches!(
            value_to_json(Some(&SqliteValue::Blob(vec![1, 2].into()))),
            JV::String(_)
        ));
        assert_eq!(value_to_json(Some(&SqliteValue::Null)), JV::Null);
        assert_eq!(value_to_json(None), JV::Null);
    }

    #[test]
    fn validate_schema_version_rejects_major_mismatch() {
        let manifest = SyncManifest {
            schema_version: "2.0".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "db.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        let err = validate_schema_version(&manifest).expect_err("should reject");
        assert!(err.to_string().contains("schema version mismatch"));
    }

    #[test]
    fn validate_schema_version_accepts_minor_difference() {
        let manifest = SyncManifest {
            schema_version: "1.99".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "db.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        assert!(validate_schema_version(&manifest).is_ok());
    }

    // ── Additional helper-function unit tests ──

    #[test]
    fn value_to_string_sqlite_all_variants() {
        assert_eq!(
            value_to_string_sqlite(Some(&SqliteValue::Text("hello".to_owned().into()))),
            "hello"
        );
        assert_eq!(
            value_to_string_sqlite(Some(&SqliteValue::Integer(42))),
            "42"
        );
        assert_eq!(
            value_to_string_sqlite(Some(&SqliteValue::Float(3.5))),
            "3.5"
        );
        assert_eq!(
            value_to_string_sqlite(Some(&SqliteValue::Blob(vec![1, 2, 3].into()))),
            ""
        );
        assert_eq!(value_to_string_sqlite(Some(&SqliteValue::Null)), "");
        assert_eq!(value_to_string_sqlite(None), "");
    }

    #[test]
    fn sqlite_to_optional_f64_all_variants() {
        assert_eq!(
            sqlite_to_optional_f64(Some(&SqliteValue::Float(1.5))),
            Some(1.5)
        );
        assert_eq!(
            sqlite_to_optional_f64(Some(&SqliteValue::Integer(7))),
            Some(7.0)
        );
        assert_eq!(
            sqlite_to_optional_f64(Some(&SqliteValue::Text("nope".to_owned().into()))),
            None
        );
        assert_eq!(sqlite_to_optional_f64(Some(&SqliteValue::Null)), None);
        assert_eq!(sqlite_to_optional_f64(None), None);
    }

    #[test]
    fn sqlite_to_optional_text_all_variants() {
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Text("hi".to_owned().into()))),
            Some("hi".to_owned())
        );
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Integer(1))),
            None
        );
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Float(1.0))),
            None
        );
        assert_eq!(sqlite_to_optional_text(None), None);
    }

    #[test]
    fn json_optional_float_extracts_number_or_null() {
        let value = json!({"score": 42.5, "name": "test", "missing": null});
        match json_optional_float(&value, "score") {
            SqliteValue::Float(f) => assert!((f - 42.5).abs() < 1e-9),
            other => assert!(matches!(other, SqliteValue::Float(_)), "expected Float"),
        }
        assert!(matches!(
            json_optional_float(&value, "name"),
            SqliteValue::Null
        ));
        assert!(matches!(
            json_optional_float(&value, "missing"),
            SqliteValue::Null
        ));
        assert!(matches!(
            json_optional_float(&value, "absent"),
            SqliteValue::Null
        ));
    }

    #[test]
    fn json_optional_text_extracts_string_or_null() {
        let value = json!({"name": "alice", "count": 5});
        match json_optional_text(&value, "name") {
            SqliteValue::Text(s) => assert_eq!(&*s, "alice"),
            other => assert!(matches!(other, SqliteValue::Text(_)), "expected Text"),
        }
        assert!(matches!(
            json_optional_text(&value, "count"),
            SqliteValue::Null
        ));
        assert!(matches!(
            json_optional_text(&value, "absent"),
            SqliteValue::Null
        ));
    }

    #[test]
    fn json_to_optional_f64_extracts_or_returns_none() {
        let value = json!({"score": 42.5, "label": "test"});
        assert_eq!(json_to_optional_f64(&value, "score"), Some(42.5));
        assert_eq!(json_to_optional_f64(&value, "label"), None);
        assert_eq!(json_to_optional_f64(&value, "missing"), None);
    }

    #[test]
    fn json_to_optional_text_extracts_or_returns_none() {
        let value = json!({"label": "hello", "count": 5});
        assert_eq!(
            json_to_optional_text(&value, "label"),
            Some("hello".to_owned())
        );
        assert_eq!(json_to_optional_text(&value, "count"), None);
        assert_eq!(json_to_optional_text(&value, "missing"), None);
    }

    #[test]
    fn json_str_returns_ok_for_present_string() {
        let value = json!({"name": "alice"});
        assert_eq!(json_str(&value, "name").unwrap(), "alice");
    }

    #[test]
    fn json_str_returns_error_for_non_string_value() {
        let value = json!({"count": 42});
        let err = json_str(&value, "count").expect_err("non-string should fail");
        assert!(err.to_string().contains("missing string field"));
    }

    #[test]
    fn json_string_or_default_returns_value_for_non_string_type() {
        let value = json!({"count": 42});
        // Non-string value should fall back to default.
        assert_eq!(
            json_string_or_default(&value, "count", "fallback"),
            "fallback"
        );
    }

    #[test]
    fn optional_floats_equal_nan_is_not_equal() {
        // NaN != NaN in float arithmetic; (NaN - NaN).abs() is NaN, which fails <= 1e-9.
        assert!(!optional_floats_equal(Some(f64::NAN), Some(f64::NAN)));
    }

    #[test]
    fn sync_manifest_serde_round_trip() {
        let manifest = SyncManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: "2026-02-22T12:00:00Z".to_owned(),
            source_db_path: "/tmp/test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 10,
                segments: 50,
                events: 30,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: "abc123".to_owned(),
                segments_jsonl_sha256: "def456".to_owned(),
                events_jsonl_sha256: "ghi789".to_owned(),
            },
        };

        let json = serde_json::to_string_pretty(&manifest).expect("serialize");
        let deserialized: SyncManifest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.schema_version, "1.1");
        assert_eq!(deserialized.export_format_version, "1.0");
        assert_eq!(deserialized.created_at_rfc3339, "2026-02-22T12:00:00Z");
        assert_eq!(deserialized.source_db_path, "/tmp/test.sqlite3");
        assert_eq!(deserialized.row_counts, manifest.row_counts);
        assert_eq!(deserialized.checksums, manifest.checksums);
    }

    #[test]
    fn row_counts_equality_and_inequality() {
        let a = RowCounts {
            runs: 1,
            segments: 2,
            events: 3,
        };
        let b = RowCounts {
            runs: 1,
            segments: 2,
            events: 3,
        };
        let c = RowCounts {
            runs: 1,
            segments: 2,
            events: 4,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn file_checksums_equality_and_inequality() {
        let a = FileChecksums {
            runs_jsonl_sha256: "aaa".to_owned(),
            segments_jsonl_sha256: "bbb".to_owned(),
            events_jsonl_sha256: "ccc".to_owned(),
        };
        let b = a.clone();
        let mut c = a.clone();
        c.events_jsonl_sha256 = "zzz".to_owned();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn sha256_file_empty_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.txt");
        fs::write(&path, b"").expect("write empty");
        let hash = sha256_file(&path).expect("hash");
        // SHA-256 of empty input is a well-known constant.
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_file_known_content() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("hello.txt");
        fs::write(&path, b"hello\n").expect("write");
        let hash = sha256_file(&path).expect("hash");
        // SHA-256 of "hello\n" — deterministic.
        assert_eq!(
            hash,
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        );
    }

    #[test]
    fn sha256_file_nonexistent_returns_error() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("nonexistent.txt");
        assert!(sha256_file(&path).is_err());
    }

    #[test]
    fn atomic_rename_moves_file() {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("src.txt");
        let dst = dir.path().join("dst.txt");
        fs::write(&src, b"payload").expect("write");
        atomic_rename(&src, &dst).expect("rename");
        assert!(!src.exists());
        assert!(dst.exists());
        assert_eq!(fs::read_to_string(&dst).expect("read"), "payload");
    }

    #[test]
    fn atomic_rename_overwrites_existing_destination_file() {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("src.txt");
        let dst = dir.path().join("dst.txt");
        fs::write(&src, b"new").expect("write src");
        fs::write(&dst, b"old").expect("write dst");

        atomic_rename(&src, &dst).expect("rename");

        assert!(!src.exists(), "source should be moved");
        assert_eq!(fs::read_to_string(&dst).expect("read"), "new");
    }

    #[test]
    fn atomic_write_bytes_creates_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("output.txt");
        atomic_write_bytes(&path, b"content here").expect("write");
        assert_eq!(fs::read_to_string(&path).expect("read"), "content here");
        // Temp file should not remain.
        let tmp = path.with_extension("tmp");
        assert!(!tmp.exists());
    }

    #[test]
    fn sync_lock_double_release_is_idempotent() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock = SyncLock::acquire(&state_root, "test").expect("acquire");
        lock.release().expect("first release");
        // After explicit release, drop should be a no-op (no panic, no error).
    }

    #[test]
    fn sync_lock_drop_cleans_up_lock_file() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock_path = state_root.join("locks/sync.lock");

        {
            let _lock = SyncLock::acquire(&state_root, "test").expect("acquire");
            assert!(lock_path.exists(), "lock file should exist while held");
        }
        // After drop, lock file should be removed.
        assert!(
            !lock_path.exists(),
            "lock file should be cleaned up on drop"
        );
    }

    #[test]
    fn sync_conflict_serde_round_trip() {
        let conflict = SyncConflict {
            table: "runs".to_owned(),
            key: "run-123".to_owned(),
            reason: "different payload for same id".to_owned(),
        };
        let json = serde_json::to_string(&conflict).expect("serialize");
        let deserialized: SyncConflict = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.table, "runs");
        assert_eq!(deserialized.key, "run-123");
        assert_eq!(deserialized.reason, "different payload for same id");
    }

    #[test]
    fn is_lock_stale_with_invalid_rfc3339_returns_false_for_alive_pid() {
        // When timestamp is unparseable, only PID liveness is checked.
        // Current PID is alive, so should return false.
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: "not-a-timestamp".to_owned(),
            operation: "test".to_owned(),
        };
        assert!(
            !is_lock_stale(&info),
            "unparseable timestamp with alive PID should not be stale"
        );
    }

    #[test]
    fn is_lock_stale_with_invalid_rfc3339_dead_pid_returns_true() {
        let info = LockInfo {
            pid: u32::MAX - 1,
            created_at_rfc3339: "not-a-timestamp".to_owned(),
            operation: "test".to_owned(),
        };
        assert!(
            is_lock_stale(&info),
            "dead PID should be stale regardless of timestamp"
        );
    }

    #[test]
    fn validate_checksums_missing_file_returns_error() {
        let dir = tempdir().expect("tempdir");
        let input_dir = dir.path().join("export");
        fs::create_dir_all(&input_dir).expect("mkdir");
        // Only create runs.jsonl — segments and events missing.
        fs::write(input_dir.join("runs.jsonl"), b"").expect("write");

        let manifest = SyncManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: sha256_file(&input_dir.join("runs.jsonl")).expect("hash"),
                segments_jsonl_sha256: "fake".to_owned(),
                events_jsonl_sha256: "fake".to_owned(),
            },
        };

        let err = validate_checksums(&manifest, &input_dir).expect_err("missing file");
        assert!(
            err.to_string().contains("missing export file"),
            "got: {}",
            err
        );
    }

    #[test]
    fn validate_checksums_matching_checksums_succeeds() {
        let dir = tempdir().expect("tempdir");
        let input_dir = dir.path().join("export");
        fs::create_dir_all(&input_dir).expect("mkdir");

        for name in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            fs::write(input_dir.join(name), b"").expect("write");
        }

        let manifest = SyncManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: sha256_file(&input_dir.join("runs.jsonl")).expect("hash"),
                segments_jsonl_sha256: sha256_file(&input_dir.join("segments.jsonl"))
                    .expect("hash"),
                events_jsonl_sha256: sha256_file(&input_dir.join("events.jsonl")).expect("hash"),
            },
        };

        assert!(validate_checksums(&manifest, &input_dir).is_ok());
    }

    #[test]
    fn value_to_json_blob_format_includes_length() {
        let blob = SqliteValue::Blob(vec![0xDE, 0xAD, 0xBE, 0xEF].into());
        let result = value_to_json(Some(&blob));
        assert_eq!(result, serde_json::Value::String("<blob:4>".to_owned()));
    }

    #[test]
    fn import_missing_manifest_returns_error() {
        let dir = tempdir().expect("tempdir");
        let empty_dir = dir.path().join("no-manifest");
        fs::create_dir_all(&empty_dir).expect("mkdir");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let err = import(&target_db, &empty_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("missing manifest");
        assert!(
            err.to_string().contains("manifest.json not found"),
            "got: {}",
            err
        );
    }

    #[test]
    fn verify_schema_exists_rejects_empty_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        // Open a connection without creating tables.
        let conn = Connection::open(db_path.display().to_string()).expect("connection should open");
        let err = verify_schema_exists(&conn).expect_err("empty db should fail");
        assert!(err.to_string().contains("missing table"), "got: {}", err);
    }

    #[test]
    fn ensure_schema_creates_all_tables() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("schema_test.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("connection should open");
        ensure_schema(&conn).expect("ensure_schema");
        // verify_schema_exists should now succeed.
        verify_schema_exists(&conn).expect("schema should now exist");
    }

    #[test]
    fn ensure_runs_replay_column_adds_column_to_old_schema() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("old_schema.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("connection should open");

        // Create table WITHOUT replay_json column.
        conn.execute(
            "CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                backend TEXT NOT NULL,
                input_path TEXT NOT NULL,
                normalized_wav_path TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                transcript TEXT NOT NULL
            );",
        )
        .expect("create old schema");

        // First call should add the column.
        ensure_runs_replay_column(&conn).expect("add column");

        // Second call should be a no-op.
        ensure_runs_replay_column(&conn).expect("idempotent");

        // Verify column exists.
        let rows = conn.query("PRAGMA table_info(runs);").expect("pragma");
        let has_replay = rows
            .iter()
            .any(|row| value_to_string_sqlite(row.get(1)) == "replay_json");
        assert!(
            has_replay,
            "replay_json column should exist after migration"
        );
    }

    #[test]
    fn export_import_multiple_runs() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multi.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        for i in 0..3 {
            store
                .persist_report(&fixture_report(&format!("multi-{i}"), &db_path))
                .expect("persist");
        }
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);
        let sanity_store = RunStore::open(&db_path).expect("reopen store");
        assert_eq!(sanity_store.list_recent_runs(10).expect("list").len(), 3);
        drop(sanity_store);

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 3);
        assert_eq!(manifest.row_counts.segments, 6); // 2 per run
        assert_eq!(manifest.row_counts.events, 6); // 2 per run

        let target_db = dir.path().join("target.sqlite3");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 3);
        assert_eq!(result.segments_imported, 6);
        assert_eq!(result.events_imported, 6);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn validate_checksums_detects_hash_mismatch() {
        let dir = tempdir().expect("tempdir");
        let input_dir = dir.path().join("export");
        fs::create_dir_all(&input_dir).expect("mkdir");

        for name in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            fs::write(input_dir.join(name), b"").expect("write");
        }

        let empty_hash = sha256_file(&input_dir.join("runs.jsonl")).expect("hash");
        let manifest = SyncManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: empty_hash.clone(),
                segments_jsonl_sha256: empty_hash,
                events_jsonl_sha256: "wrong_hash_value".to_owned(),
            },
        };

        let err = validate_checksums(&manifest, &input_dir).expect_err("hash mismatch");
        assert!(
            err.to_string().contains("checksum mismatch"),
            "got: {}",
            err
        );
    }

    #[test]
    fn json_str_returns_error_for_null_value() {
        let value = json!({"key": null});
        let err = json_str(&value, "key").expect_err("null should fail");
        assert!(err.to_string().contains("missing string field"));
    }

    #[test]
    fn export_import_unicode_content() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("unicode.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("unicode-1", &db_path);
        report.result.transcript = "\u{1F600} \u{4E16}\u{754C} caf\u{00E9}".to_owned();
        report.result.segments[0].text = "\u{1F600} \u{4E16}\u{754C}".to_owned();
        report.result.segments[1].text = "caf\u{00E9}".to_owned();
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);

        let target_db = dir.path().join("target.sqlite3");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("unicode-1")
            .expect("load")
            .expect("details");
        assert!(details.transcript.contains('\u{1F600}'));
        assert!(details.transcript.contains("caf\u{00E9}"));
    }

    #[test]
    fn write_conflicts_file_creates_valid_jsonl() {
        let dir = tempdir().expect("tempdir");
        let conflicts = vec![
            SyncConflict {
                table: "runs".to_owned(),
                key: "run-1".to_owned(),
                reason: "payload mismatch".to_owned(),
            },
            SyncConflict {
                table: "segments".to_owned(),
                key: "run-1/0".to_owned(),
                reason: "duplicate composite key".to_owned(),
            },
        ];

        write_conflicts_file(dir.path(), &conflicts).expect("write");

        let content = fs::read_to_string(dir.path().join("sync_conflicts.jsonl")).expect("read");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: SyncConflict = serde_json::from_str(lines[0]).expect("parse line 0");
        assert_eq!(first.table, "runs");
        let second: SyncConflict = serde_json::from_str(lines[1]).expect("parse line 1");
        assert_eq!(second.key, "run-1/0");
    }

    #[test]
    fn import_overwrite_segment_conflict_fails_closed() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("seg-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate a segment's text in the JSONL.
        let seg_path = export_dir.join("segments.jsonl");
        let original = fs::read_to_string(&seg_path).expect("read segments");
        let mutated = original.replace("hello world", "replaced text");
        fs::write(&seg_path, &mutated).expect("write segments");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("manifest write");

        let error = import(
            &db_path,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect_err("overwrite import should fail when segment conflict needs UPDATE");
        let text = error.to_string();
        assert!(
            text.contains("overwrite would require updating conflicting segment row"),
            "unexpected error: {text}"
        );
    }

    #[test]
    fn import_overwrite_changed_parent_with_identical_children_updates_parent() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("source.sqlite3");
        let target_db = dir.path().join("target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let mut source_report = fixture_report("overwrite-parent-1", &source_db);
        source_report.result.transcript = "updated parent payload".to_owned();
        source_store
            .persist_report(&source_report)
            .expect("persist source");

        let target_store = RunStore::open(&target_db).expect("target store");
        target_store
            .persist_report(&fixture_report("overwrite-parent-1", &target_db))
            .expect("persist target");

        export(&source_db, &export_dir, &state_root).expect("export source");
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("non-strict overwrite should update only the parent");

        let details = target_store
            .load_run_details("overwrite-parent-1")
            .expect("load target")
            .expect("target run should remain");
        assert_eq!(details.transcript, "updated parent payload");
        assert_eq!(details.segments, source_report.result.segments);
        assert_eq!(details.events.len(), source_report.events.len());
        for (actual, expected) in details.events.iter().zip(&source_report.events) {
            assert_eq!(actual.seq, expected.seq);
            assert_eq!(actual.ts_rfc3339, expected.ts_rfc3339);
            assert_eq!(actual.stage, expected.stage);
            assert_eq!(actual.code, expected.code);
            assert_eq!(actual.message, expected.message);
            assert_eq!(actual.payload, expected.payload);
        }
    }

    #[test]
    fn import_overwrite_changed_parent_and_segment_rolls_back_atomically() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("source.sqlite3");
        let target_db = dir.path().join("target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let mut source_report = fixture_report("overwrite-atomic-1", &source_db);
        source_report.result.transcript = "source parent payload".to_owned();
        source_report.result.segments[0].text = "source child payload".to_owned();
        source_store
            .persist_report(&source_report)
            .expect("persist source");

        let target_store = RunStore::open(&target_db).expect("target store");
        let mut target_report = fixture_report("overwrite-atomic-1", &target_db);
        target_report.result.transcript = "target parent payload".to_owned();
        target_report.result.segments[0].text = "target child payload".to_owned();
        target_store
            .persist_report(&target_report)
            .expect("persist target");

        export(&source_db, &export_dir, &state_root).expect("export source");
        let error = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect_err("non-strict overwrite must compare the retained target child");
        assert!(
            error
                .to_string()
                .contains("overwrite would require updating conflicting segment row"),
            "unexpected error: {error}"
        );

        let details = target_store
            .load_run_details("overwrite-atomic-1")
            .expect("load target after rollback")
            .expect("target run should remain");
        assert_eq!(details.transcript, "target parent payload");
        assert_eq!(details.segments[0].text, "target child payload");
        assert_eq!(details.segments.len(), target_report.result.segments.len());
        assert_eq!(details.events.len(), target_report.events.len());
    }

    #[test]
    fn import_overwrite_changed_parent_and_event_rolls_back_atomically() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("source.sqlite3");
        let target_db = dir.path().join("target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let mut source_report = fixture_report("overwrite-event-atomic-1", &source_db);
        source_report.result.transcript = "source parent payload".to_owned();
        source_report.events[0].message = "source event payload".to_owned();
        source_store
            .persist_report(&source_report)
            .expect("persist source");

        let target_store = RunStore::open(&target_db).expect("target store");
        let mut target_report = fixture_report("overwrite-event-atomic-1", &target_db);
        target_report.result.transcript = "target parent payload".to_owned();
        target_report.events[0].message = "target event payload".to_owned();
        target_store
            .persist_report(&target_report)
            .expect("persist target");

        export(&source_db, &export_dir, &state_root).expect("export source");
        let error = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect_err("non-strict overwrite must compare the retained target event");
        assert!(
            error
                .to_string()
                .contains("overwrite would require updating conflicting event row"),
            "unexpected error: {error}"
        );

        let details = target_store
            .load_run_details("overwrite-event-atomic-1")
            .expect("load target after rollback")
            .expect("target run should remain");
        assert_eq!(details.transcript, "target parent payload");
        assert_eq!(details.events[0].message, "target event payload");
        assert_eq!(details.segments.len(), target_report.result.segments.len());
        assert_eq!(details.events.len(), target_report.events.len());
    }

    #[test]
    fn import_overwrite_event_conflict_fails_closed() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("evt-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate an event's message in the JSONL.
        let evt_path = export_dir.join("events.jsonl");
        let original = fs::read_to_string(&evt_path).expect("read events");
        let mutated = original.replace("input materialized", "replaced event message");
        fs::write(&evt_path, &mutated).expect("write events");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("manifest write");

        let error = import(
            &db_path,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect_err("overwrite import should fail when event conflict needs UPDATE");
        let text = error.to_string();
        assert!(
            text.contains("overwrite would require updating conflicting event row"),
            "unexpected error: {text}"
        );
    }

    #[test]
    fn import_overwrite_run_with_stale_children_fails_closed() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("source.sqlite3");
        let target_db = dir.path().join("target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let source_report = fixture_report("overwrite-stale-1", &source_db);
        source_store
            .persist_report(&source_report)
            .expect("persist source");

        let target_store = RunStore::open(&target_db).expect("target store");
        let mut target_report = fixture_report("overwrite-stale-1", &target_db);
        target_report.result.segments.push(TranscriptionSegment {
            start_sec: Some(5.0),
            end_sec: Some(6.0),
            text: "stale tail segment".to_owned(),
            speaker: None,
            confidence: Some(0.1),
        });
        target_report.events.push(RunEvent {
            seq: 3,
            ts_rfc3339: "2026-01-01T00:00:04Z".to_owned(),
            stage: "persist".to_owned(),
            code: "persist.ok".to_owned(),
            message: "stale extra event".to_owned(),
            payload: json!({"stale": true}),
        });
        target_store
            .persist_report(&target_report)
            .expect("persist target");

        export(&source_db, &export_dir, &state_root).expect("export source");
        let error = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect_err("overwrite import should fail when stale child rows require DELETE");
        let text = error.to_string();
        assert!(
            text.contains("overwrite would require deleting stale segment")
                || text.contains("overwrite would require deleting stale event"),
            "unexpected error: {text}"
        );
    }

    #[test]
    fn import_overwrite_strict_segment_conflict_replaces_row() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("seg-strict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let seg_path = export_dir.join("segments.jsonl");
        let original = fs::read_to_string(&seg_path).expect("read segments");
        let mut rows: Vec<serde_json::Value> = original
            .lines()
            .map(|line| serde_json::from_str(line).expect("parse segment row"))
            .collect();
        assert!(
            !rows.is_empty(),
            "fixture should emit at least one segment row"
        );
        rows[0]["text"] = json!("strict replaced text");
        let mutated = rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("serialize segment row"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&seg_path, format!("{mutated}\n")).expect("write segments");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("manifest write");

        import(
            &db_path,
            &export_dir,
            &state_root,
            ConflictPolicy::OverwriteStrict,
        )
        .expect("strict overwrite import should update conflicting segment");

        let conn = Connection::open(db_path.display().to_string()).expect("open db");
        let rows = conn
            .query_with_params(
                "SELECT text FROM segments WHERE run_id = ?1 AND idx = ?2",
                &[
                    SqliteValue::Text("seg-strict-1".to_owned().into()),
                    SqliteValue::Integer(0),
                ],
            )
            .expect("query segment");
        assert!(!rows.is_empty(), "segment row should exist");
        assert_eq!(
            value_to_string_sqlite(rows[0].get(0)),
            "strict replaced text",
            "strict overwrite should replace segment payload in segments table"
        );
    }

    #[test]
    fn import_overwrite_strict_event_conflict_replaces_row() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("evt-strict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let evt_path = export_dir.join("events.jsonl");
        let original = fs::read_to_string(&evt_path).expect("read events");
        let mutated = original.replace("input materialized", "strict replaced event");
        fs::write(&evt_path, &mutated).expect("write events");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
                .expect("manifest json");
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("manifest write");

        import(
            &db_path,
            &export_dir,
            &state_root,
            ConflictPolicy::OverwriteStrict,
        )
        .expect("strict overwrite import should update conflicting event");

        let details = store
            .load_run_details("evt-strict-1")
            .expect("load")
            .expect("exists");
        assert!(
            details
                .events
                .iter()
                .any(|event| event.message == "strict replaced event"),
            "strict overwrite should replace event payload"
        );
    }

    #[test]
    fn import_overwrite_strict_prunes_stale_children() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("source.sqlite3");
        let target_db = dir.path().join("target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let source_report = fixture_report("strict-prune-1", &source_db);
        source_store
            .persist_report(&source_report)
            .expect("persist source");

        let target_store = RunStore::open(&target_db).expect("target store");
        target_store
            .persist_report(&fixture_report("strict-prune-1", &target_db))
            .expect("persist target");

        let target_conn = Connection::open(target_db.display().to_string()).expect("target conn");
        target_conn
            .execute_with_params(
                "INSERT INTO segments (run_id, idx, start_sec, end_sec, speaker, text, confidence) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    SqliteValue::Text("strict-prune-1".to_owned().into()),
                    SqliteValue::Integer(99),
                    SqliteValue::Float(9.0),
                    SqliteValue::Float(10.0),
                    SqliteValue::Null,
                    SqliteValue::Text("stale strict segment".to_owned().into()),
                    SqliteValue::Float(0.1),
                ],
            )
            .expect("insert stale segment");
        target_conn
            .execute_with_params(
                "INSERT INTO events (run_id, seq, ts_rfc3339, stage, code, message, payload_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    SqliteValue::Text("strict-prune-1".to_owned().into()),
                    SqliteValue::Integer(99),
                    SqliteValue::Text("2026-01-01T00:00:09Z".to_owned().into()),
                    SqliteValue::Text("persist".to_owned().into()),
                    SqliteValue::Text("persist.ok".to_owned().into()),
                    SqliteValue::Text("stale strict event".to_owned().into()),
                    SqliteValue::Text("{\"stale\":true}".to_owned().into()),
                ],
            )
            .expect("insert stale event");

        export(&source_db, &export_dir, &state_root).expect("export source");
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::OverwriteStrict,
        )
        .expect("strict overwrite import should prune stale child rows");

        let details = target_store
            .load_run_details("strict-prune-1")
            .expect("load")
            .expect("exists");
        assert_eq!(
            details.segments.len(),
            source_report.result.segments.len(),
            "strict overwrite should prune stale segments"
        );
        assert_eq!(
            details.events.len(),
            source_report.events.len(),
            "strict overwrite should prune stale events"
        );
        assert!(
            details
                .segments
                .iter()
                .all(|segment| segment.text != "stale strict segment"),
            "stale strict segment should be removed"
        );
        assert!(
            details
                .events
                .iter()
                .all(|event| event.message != "stale strict event"),
            "stale strict event should be removed"
        );
    }

    #[test]
    fn import_idempotent_on_identical_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("idempotent-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import
        let result_1 =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import 1");
        assert_eq!(result_1.runs_imported, 1);
        assert!(result_1.conflicts.is_empty());

        // Second import of same data → no conflicts (idempotent upsert)
        let result_2 = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import 2 should succeed (idempotent)");
        assert!(
            result_2.conflicts.is_empty(),
            "re-import of identical data should produce no conflicts"
        );
    }

    #[test]
    fn lock_info_serde_round_trip() {
        let info = LockInfo {
            pid: 12345,
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            operation: "export".to_owned(),
        };
        let json = serde_json::to_string(&info).expect("serialize");
        let parsed: LockInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.pid, 12345);
        assert_eq!(parsed.operation, "export");
        assert_eq!(parsed.created_at_rfc3339, "2026-01-01T00:00:00Z");
    }

    #[test]
    fn conflict_policy_all_variants_are_distinct() {
        assert_ne!(ConflictPolicy::Reject, ConflictPolicy::Overwrite);
        assert_ne!(ConflictPolicy::Reject, ConflictPolicy::OverwriteStrict);
        assert_ne!(ConflictPolicy::Overwrite, ConflictPolicy::OverwriteStrict);
        // Verify both can be used without panic.
        let _ = format!("{:?}", ConflictPolicy::Reject);
        let _ = format!("{:?}", ConflictPolicy::Overwrite);
        let _ = format!("{:?}", ConflictPolicy::OverwriteStrict);
    }

    #[test]
    fn import_malformed_manifest_json_returns_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("target.sqlite3");
        let input_dir = dir.path().join("bad_manifest");
        let state_root = dir.path().join("state");

        fs::create_dir_all(&input_dir).expect("mkdir");
        fs::write(input_dir.join("manifest.json"), "not valid json {{")
            .expect("write bad manifest");

        let result = import(&db_path, &input_dir, &state_root, ConflictPolicy::Reject);
        assert!(result.is_err());
        let err_text = result.unwrap_err().to_string();
        assert!(
            err_text.contains("manifest") || err_text.contains("invalid"),
            "expected manifest error, got: {err_text}"
        );
    }

    #[test]
    fn sha256_file_deterministic_for_same_content() {
        let dir = tempdir().expect("tempdir");
        let file_a = dir.path().join("a.txt");
        let file_b = dir.path().join("b.txt");
        let content = "deterministic content\n".repeat(100);
        fs::write(&file_a, &content).expect("write a");
        fs::write(&file_b, &content).expect("write b");

        let hash_a = sha256_file(&file_a).expect("hash a");
        let hash_b = sha256_file(&file_b).expect("hash b");
        assert_eq!(hash_a, hash_b, "same content should produce same hash");
        assert_eq!(hash_a.len(), 64, "SHA-256 hex should be 64 chars");
    }

    #[test]
    fn atomic_write_bytes_large_content() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("large.bin");
        let content: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        atomic_write_bytes(&path, &content).expect("write");
        let read_back = fs::read(&path).expect("read");
        assert_eq!(read_back.len(), 100_000);
        assert_eq!(read_back, content);
    }

    #[test]
    fn atomic_rename_moves_content() {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("source.txt");
        let dst = dir.path().join("destination.txt");
        fs::write(&src, b"rename test content").expect("write");
        atomic_rename(&src, &dst).expect("rename");
        assert!(!src.exists());
        assert_eq!(
            fs::read_to_string(&dst).expect("read"),
            "rename test content"
        );
    }

    #[test]
    fn export_manifest_contains_expected_version_fields() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("version_check.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        assert_eq!(manifest.schema_version, SCHEMA_VERSION);
        assert_eq!(manifest.export_format_version, EXPORT_FORMAT_VERSION);
        assert!(!manifest.created_at_rfc3339.is_empty());
        assert!(manifest.source_db_path.contains("version_check.sqlite3"));
    }

    #[test]
    fn export_with_data_includes_segments_and_events() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("full_data.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("full-data-1", &db_path);
        store.persist_report(&report).expect("persist");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        // fixture_report has 1 run, 2 segments, 2 events
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 2);

        // JSONL files should exist and be non-empty
        let runs_content = fs::read_to_string(export_dir.join("runs.jsonl")).expect("read");
        assert!(!runs_content.trim().is_empty());
        let segs_content = fs::read_to_string(export_dir.join("segments.jsonl")).expect("read");
        assert!(!segs_content.trim().is_empty());
        let events_content = fs::read_to_string(export_dir.join("events.jsonl")).expect("read");
        assert!(!events_content.trim().is_empty());
    }

    #[test]
    fn write_conflicts_file_empty_array_creates_empty_file() {
        let dir = tempdir().expect("tempdir");
        write_conflicts_file(dir.path(), &[]).expect("write empty");

        let content = fs::read_to_string(dir.path().join("sync_conflicts.jsonl")).expect("read");
        assert!(
            content.trim().is_empty(),
            "empty conflict list should produce empty file"
        );
    }

    #[test]
    fn export_import_preserves_segment_optional_fields() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("optional_fields.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("optional-1", &db_path);
        // One segment with all fields, one with all optional fields None
        report.result.segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "with fields".to_owned(),
                speaker: Some("SPEAKER_00".to_owned()),
                confidence: Some(0.95),
            },
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "no optional fields".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.segments_imported, 2);

        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("optional-1")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments.len(), 2);
        assert_eq!(details.segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(details.segments[0].confidence, Some(0.95));
        assert!(details.segments[1].start_sec.is_none());
        assert!(details.segments[1].speaker.is_none());
        assert!(details.segments[1].confidence.is_none());
    }

    #[test]
    fn export_import_multiple_runs_preserves_all_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multi.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        for i in 0..3 {
            let mut report = fixture_report(&format!("multi-{i}"), &db_path);
            report.started_at_rfc3339 = format!("2026-01-0{i}T00:00:00Z");
            store.persist_report(&report).expect("persist");
        }
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        export(&db_path, &export_dir, &state_root).expect("export");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 3);
        // fixture_report has 2 segments and 2 events each
        assert_eq!(result.segments_imported, 6);
        assert_eq!(result.events_imported, 6);
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let runs = target_store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 3);
    }

    #[test]
    fn sync_conflict_with_unicode_fields_round_trips() {
        let conflict = SyncConflict {
            table: "résultats".to_owned(),
            key: "clé-\u{1F600}".to_owned(),
            reason: "données incompatibles".to_owned(),
        };
        let json = serde_json::to_string(&conflict).expect("serialize");
        let parsed: SyncConflict = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.table, "résultats");
        assert!(parsed.key.contains('\u{1F600}'));
        assert_eq!(parsed.reason, "données incompatibles");
    }

    #[test]
    fn import_overwrite_counts_include_replaced_rows() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("overwrite-count-1", &db_path);
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // First import
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("import 1");
        // Second import — same data, should overwrite
        let result = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("import 2");
        // Overwrite should process all rows
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
    }

    #[test]
    fn export_empty_db_produces_zero_row_counts() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_counts.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);
    }

    #[test]
    fn import_jsonl_with_blank_lines_skips_them() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        // Create a valid export, then inject blank lines into the JSONL files.
        let db_path = dir.path().join("source.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("blank-lines-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Inject blank and whitespace-only lines into each JSONL file.
        for filename in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            let path = export_dir.join(filename);
            let original = fs::read_to_string(&path).expect("read");
            let with_blanks = format!("\n  \n\t\n{original}\n  \n");
            fs::write(&path, &with_blanks).expect("write");
        }

        // Update manifest checksums to match modified files.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read")).expect("json");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&export_dir.join("runs.jsonl")).expect("checksum"));
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&export_dir.join("segments.jsonl")).expect("checksum"));
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&export_dir.join("events.jsonl")).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write manifest");

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
    }

    #[test]
    fn import_malformed_runs_jsonl_returns_error() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        // Create a valid export first.
        let db_path = dir.path().join("source.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("malformed-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Replace runs.jsonl with malformed JSON.
        let runs_path = export_dir.join("runs.jsonl");
        fs::write(&runs_path, "{not valid json\n").expect("write malformed");

        // Update checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read")).expect("json");
        mval["checksums"]["runs_jsonl_sha256"] = json!(sha256_file(&runs_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("serialize"),
        )
        .expect("write manifest");

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("malformed JSONL should fail");
        let text = err.to_string();
        assert!(
            text.contains("key must be") || text.contains("json") || text.contains("JSON"),
            "error should mention JSON parse failure: {text}"
        );
    }

    #[test]
    fn optional_floats_equal_infinities_are_not_equal() {
        // INF - INF = NaN, so optional_floats_equal returns false for infinities.
        // This documents the current behavior.
        assert!(!optional_floats_equal(
            Some(f64::INFINITY),
            Some(f64::INFINITY)
        ));
        assert!(!optional_floats_equal(
            Some(f64::NEG_INFINITY),
            Some(f64::NEG_INFINITY)
        ));
        assert!(!optional_floats_equal(
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY)
        ));
    }

    #[test]
    fn json_string_or_default_with_null_value_returns_fallback() {
        let value = json!({"key": null});
        assert_eq!(
            json_string_or_default(&value, "key", "fallback"),
            "fallback",
            "null value should trigger fallback"
        );
    }

    #[test]
    fn sync_lock_drop_after_release_does_not_panic() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock = SyncLock::acquire(&state_root, "test").expect("acquire");
        lock.release().expect("explicit release");
        // Drop happens here — should not panic even though already released.
    }

    #[test]
    fn release_skips_lock_if_contents_changed() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock = SyncLock::acquire(&state_root, "test").expect("acquire");
        let lock_path = state_root.join("locks").join("sync.lock");

        let other = LockInfo {
            pid: 9999,
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            operation: "import".to_owned(),
        };
        fs::write(
            &lock_path,
            serde_json::to_string_pretty(&other).expect("serialize"),
        )
        .expect("overwrite lock");

        lock.release().expect("release should succeed");

        let contents = fs::read_to_string(&lock_path).expect("lock should remain");
        let parsed: LockInfo = serde_json::from_str(&contents).expect("parse lock JSON");
        assert_eq!(parsed.pid, 9999);
        assert_eq!(parsed.operation, "import");
    }

    #[test]
    fn import_result_counts_reflect_actual_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("counts-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn release_after_external_lock_file_deletion_succeeds() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let lock = SyncLock::acquire(&state_root, "test").expect("acquire");
        let lock_path = state_root.join("locks").join("sync.lock");
        assert!(lock_path.exists(), "lock file should exist after acquire");
        // Simulate external deletion (e.g. operator cleanup).
        fs::remove_file(&lock_path).expect("manual remove");
        assert!(!lock_path.exists());
        // release() should succeed even though the file is gone.
        lock.release()
            .expect("release after external deletion should succeed");
    }

    #[test]
    fn pid_is_alive_returns_false_for_nonexistent_pid() {
        // PID u32::MAX - 1 should not correspond to a running process.
        assert!(
            !pid_is_alive(u32::MAX - 1),
            "pid_is_alive should return false for a nonexistent PID"
        );
    }

    #[test]
    fn validate_schema_version_no_dot_same_major_accepts() {
        // A version string with no dot uses the unwrap_or fallback.
        // SCHEMA_VERSION is "1.1", so expected_major is "1".
        // A manifest with schema_version "1" (no dot) → actual_major is "1" → match.
        let manifest = SyncManifest {
            schema_version: "1".to_owned(),
            export_format_version: "1".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "test.db".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        assert!(
            validate_schema_version(&manifest).is_ok(),
            "schema_version '1' (no dot) should match major '1'"
        );
    }

    #[test]
    fn validate_schema_version_no_dot_wrong_major_rejects() {
        let manifest = SyncManifest {
            schema_version: "2".to_owned(),
            export_format_version: "1".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "test.db".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        let err = validate_schema_version(&manifest).expect_err("major 2 should be rejected");
        let text = err.to_string();
        assert!(text.contains("schema version mismatch"), "error: {text}");
        assert!(
            text.contains("2"),
            "error should mention the bad version: {text}"
        );
    }

    #[test]
    fn import_row_count_mismatch_for_runs_errors() {
        // Existing test only covers events mismatch (line 468).
        // This exercises the runs mismatch path at line 444.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("runs-mismatch-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["row_counts"]["runs"] = json!(9999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        let target_db = dir.path().join("target.sqlite3");
        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("runs row count mismatch should fail");
        assert!(
            error.to_string().contains("row count mismatch for runs"),
            "error should mention runs mismatch: {}",
            error
        );
    }

    #[test]
    fn lock_acquire_error_message_contains_pid_and_path() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let _lock = SyncLock::acquire(&state_root, "test").expect("first acquire");
        let second = SyncLock::acquire(&state_root, "test");
        assert!(second.is_err(), "second acquire should fail");
        let err = second.err().unwrap();
        let text = err.to_string();
        assert!(
            text.contains("sync lock held by pid"),
            "error should mention pid: {text}"
        );
        assert!(
            text.contains("sync.lock"),
            "error should mention lock file path: {text}"
        );
    }

    #[test]
    fn import_row_count_mismatch_for_segments_errors() {
        // Existing test covers events mismatch (line 468) and runs mismatch.
        // This exercises the segments mismatch path at line 456.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("seg-mismatch-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["row_counts"]["segments"] = json!(9999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        let target_db = dir.path().join("target.sqlite3");
        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("segments mismatch should fail");
        assert!(
            error
                .to_string()
                .contains("row count mismatch for segments"),
            "error should mention segments mismatch: {}",
            error
        );
    }

    #[test]
    fn import_reject_conflict_on_mutated_segment() {
        // The Reject path inside import_segments (line 637) is never exercised
        // by existing tests. This creates a segment conflict specifically.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("seg-conflict-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import — establishes the data.
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("first import");

        // Mutate a segment in segments.jsonl.
        let seg_path = export_dir.join("segments.jsonl");
        let seg_content = fs::read_to_string(&seg_path).expect("read");
        let first_line = seg_content.lines().next().expect("has at least one line");
        let mut mutated: serde_json::Value =
            serde_json::from_str(first_line).expect("valid segment");
        mutated["text"] = json!("MUTATED SEGMENT TEXT");
        let new_content = format!(
            "{}\n{}",
            serde_json::to_string(&mutated).expect("serialize"),
            seg_content.lines().skip(1).collect::<Vec<_>>().join("\n")
        );
        fs::write(&seg_path, format!("{new_content}\n")).expect("write");

        // Update manifest checksums.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        // Second import with Reject should fail with conflicts.
        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("mutated segment should cause conflict");
        let text = error.to_string();
        assert!(text.contains("import rejected due to"), "error: {text}");
    }

    #[test]
    fn export_fails_on_partial_schema_missing_events_table() {
        // verify_schema_exists loops over "runs", "segments", "events".
        // Existing test only covers fully missing schema (empty DB).
        // This hits the third iteration: DB has runs + segments but not events.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("partial.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, \
             finished_at TEXT NOT NULL, backend TEXT NOT NULL, input_path TEXT NOT NULL, \
             normalized_wav_path TEXT NOT NULL, request_json TEXT NOT NULL, \
             result_json TEXT NOT NULL, warnings_json TEXT NOT NULL, transcript TEXT NOT NULL);",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
             start_sec REAL, end_sec REAL, speaker TEXT, text TEXT NOT NULL, \
             confidence REAL, PRIMARY KEY (run_id, idx));",
        )
        .expect("create segments");
        drop(conn);

        let error = export(&db_path, &export_dir, &state_root)
            .expect_err("missing events table should fail");
        let text = error.to_string();
        assert!(
            text.contains("missing table `events`"),
            "error should mention events table: {text}"
        );
    }

    #[test]
    fn import_rollback_on_mid_import_error_preserves_db() {
        // When import_tables returns Err, the ROLLBACK branch (line 424)
        // should leave the target DB clean.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("rollback-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt segments.jsonl: valid JSON but missing required idx field.
        let seg_path = export_dir.join("segments.jsonl");
        fs::write(&seg_path, r#"{"run_id":"rollback-1","text":"bad"}"#)
            .expect("write corrupt segments");

        // Update checksums for the modified file.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        manifest_value["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        // Import should fail (missing idx).
        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
        assert!(err.is_err(), "import should fail due to missing idx");

        // Target DB should exist but have no runs (rollback cleaned up).
        let target_store = RunStore::open(&target_db).expect("target store");
        let runs = target_store.list_recent_runs(10).expect("list");
        assert!(
            runs.is_empty(),
            "rollback should leave 0 runs in target DB, found {}",
            runs.len()
        );
    }

    #[test]
    fn import_malformed_segments_jsonl_returns_error() {
        // Exercises the serde_json::from_str error path at line 597
        // (malformed JSON in segments.jsonl).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("bad-seg-json-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Replace segments.jsonl with invalid JSON.
        let seg_path = export_dir.join("segments.jsonl");
        fs::write(&seg_path, "{not valid json at all!!!}\n").expect("write");

        // Update checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        manifest_value["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("malformed segments should fail");
        let text = err.to_string();
        assert!(
            text.contains("key must be") || text.contains("json") || text.contains("JSON"),
            "error should mention JSON parse failure: {text}"
        );
    }

    #[test]
    fn import_reject_conflict_on_mutated_event() {
        // Exercises the Reject path inside import_events (line 727).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("evt-conflict-1", &db_path);
        report.events.push(RunEvent {
            seq: 99,
            ts_rfc3339: "2026-01-01T00:00:09Z".to_owned(),
            stage: "test".to_owned(),
            code: "test.ok".to_owned(),
            message: "original".to_owned(),
            payload: json!({}),
        });
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import — establishes the data.
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("first import");

        // Mutate an event in events.jsonl.
        let evt_path = export_dir.join("events.jsonl");
        let evt_content = fs::read_to_string(&evt_path).expect("read");
        let first_line = evt_content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid event");
        mutated["message"] = json!("MUTATED MESSAGE");
        let new_content = format!(
            "{}\n{}",
            serde_json::to_string(&mutated).expect("serialize"),
            evt_content.lines().skip(1).collect::<Vec<_>>().join("\n")
        );
        fs::write(&evt_path, format!("{new_content}\n")).expect("write");

        // Update checksums.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("mutated event should cause conflict");
        let text = error.to_string();
        assert!(text.contains("import rejected due to"), "error: {text}");
    }

    #[test]
    fn import_empty_export_dir_with_valid_manifest_but_missing_jsonl_errors() {
        // Manifest references JSONL files that don't exist on disk.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        // Create a real export, then delete the data files but keep manifest.
        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("ghost-files", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Remove the actual data files.
        let _ = fs::remove_file(export_dir.join("runs.jsonl"));
        let _ = fs::remove_file(export_dir.join("segments.jsonl"));
        let _ = fs::remove_file(export_dir.join("events.jsonl"));

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("missing JSONL files should fail");
        let text = error.to_string();
        // The error should mention the missing file or a read failure.
        assert!(
            text.contains("runs.jsonl")
                || text.contains("No such file")
                || text.contains("not found"),
            "error should reference missing file: {text}"
        );
    }

    #[test]
    fn import_corrupt_manifest_json_returns_error() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("bad_manifest");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");
        fs::create_dir_all(&export_dir).expect("mkdir");
        fs::write(export_dir.join("manifest.json"), "{not valid json!!!").expect("write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("corrupt manifest should fail");
        let text = error.to_string();
        assert!(
            text.contains("invalid manifest"),
            "error should mention invalid manifest: {text}"
        );
    }

    #[test]
    fn import_checksum_mismatch_returns_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("checksum-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt runs.jsonl content to invalidate checksum.
        let runs_path = export_dir.join("runs.jsonl");
        fs::write(&runs_path, "extra garbage line\n").expect("write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("checksum mismatch should fail");
        let text = error.to_string();
        assert!(
            text.contains("checksum mismatch"),
            "error should mention checksum: {text}"
        );
    }

    #[test]
    fn overwrite_replaces_run_data_on_second_import() {
        // When Overwrite policy encounters a different payload for the same run_id,
        // it deletes and re-inserts. Verify the new data is present afterwards.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("ow-replace-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import.
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("first import");

        // Mutate the transcript in runs.jsonl.
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["transcript"] = json!("OVERWRITTEN transcript text");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("s")),
        )
        .expect("write");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("checksum"));
        // Update row_counts.runs to 1 since we only have 1 line now.
        manifest_value["row_counts"]["runs"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write");

        // Second import with Overwrite — should succeed, replacing the run data.
        let result = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("overwrite import");
        // Verify the target DB now has the overwritten transcript column.
        let conn = Connection::open(target_db.display().to_string()).expect("open target db");
        let rows = conn
            .query_with_params(
                "SELECT transcript FROM runs WHERE id = ?1",
                &[SqliteValue::Text("ow-replace-1".to_owned().into())],
            )
            .expect("query");
        assert!(!rows.is_empty(), "run should exist after overwrite");
        let transcript = value_to_string_sqlite(rows[0].get(0));
        assert_eq!(
            transcript, "OVERWRITTEN transcript text",
            "overwrite should replace old transcript column"
        );
        assert_eq!(result.runs_imported, 1, "should report 1 run imported");
    }

    #[test]
    fn overwrite_run_with_missing_children_fails_closed() {
        // Overwrite imports that remove existing child rows must fail closed in
        // this runtime because child-row DELETE/UPDATE are intentionally blocked.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("cascade-1", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import — populates runs, segments, events.
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("first import");

        // Mutate the run transcript and clear segments/events from the import.
        // This requires deleting stale child rows in target DB.
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["transcript"] = json!("cascade-overwritten");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("s")),
        )
        .expect("write");

        // Empty out segments.jsonl so no segments are re-imported.
        let seg_path = export_dir.join("segments.jsonl");
        fs::write(&seg_path, "").expect("write empty segments");

        // Empty out events.jsonl so no events are re-imported.
        let evt_path = export_dir.join("events.jsonl");
        fs::write(&evt_path, "").expect("write empty events");

        // Update manifest checksums and row counts.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("checksum"));
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        manifest_value["row_counts"]["runs"] = json!(1);
        manifest_value["row_counts"]["segments"] = json!(0);
        manifest_value["row_counts"]["events"] = json!(0);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("serialize"),
        )
        .expect("write manifest");

        // Second overwrite import should fail closed with explicit guidance.
        let error = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect_err("overwrite import should fail closed");
        let message = error.to_string();
        assert!(
            message.contains("child-row DELETE is unsupported"),
            "should explain fail-closed overwrite behavior: {message}"
        );
    }

    #[test]
    fn value_to_json_all_sqlite_variants() {
        use super::value_to_json;
        assert_eq!(
            value_to_json(Some(&SqliteValue::Text("hello".to_owned().into()))),
            json!("hello")
        );
        assert_eq!(value_to_json(Some(&SqliteValue::Integer(42))), json!(42));
        assert_eq!(value_to_json(Some(&SqliteValue::Float(2.75))), json!(2.75));
        assert_eq!(
            value_to_json(Some(&SqliteValue::Blob(vec![1, 2, 3].into()))),
            json!("<blob:3>")
        );
        assert_eq!(
            value_to_json(Some(&SqliteValue::Blob(vec![].into()))),
            json!("<blob:0>")
        );
        assert_eq!(value_to_json(Some(&SqliteValue::Null)), json!(null));
        assert_eq!(value_to_json(None), json!(null));
    }

    #[test]
    fn export_import_with_special_json_characters_in_transcript() {
        // Verify that transcripts containing JSON-special characters
        // (quotes, backslashes, newlines) survive the export/import round-trip.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("special_chars.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("special-chars", &db_path);
        report.result.transcript = "He said \"hello\"\nand\\backslash\ttab\0null".to_owned();
        report.result.segments[0].text = "line1\nline2".to_owned();
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");

        let target_store = RunStore::open(&target_db).expect("open target");
        let details = target_store
            .load_run_details("special-chars")
            .expect("load")
            .expect("exists");
        assert!(
            details.transcript.contains("\\"),
            "backslash should survive round-trip"
        );
        assert_eq!(
            details.segments[0].text, "line1\nline2",
            "newline in segment text should survive"
        );
    }

    #[test]
    fn import_identical_event_twice_is_noop() {
        // When import encounters the same event with identical fields,
        // it is a noop (line 721-723) — no conflict, no error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("idempotent_event.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("idem-evt", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Import twice — second import should be idempotent.
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("first import");
        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("second import should succeed (identical data)");
        assert!(
            result.conflicts.is_empty(),
            "identical re-import should have no conflicts"
        );
    }

    #[test]
    fn ensure_schema_is_idempotent() {
        // Calling ensure_schema twice on the same DB should not error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("schema_idem.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("first schema create");
        ensure_schema(&conn).expect("second schema create should be idempotent");

        // Tables should all exist.
        for table in ["runs", "segments", "events"] {
            let sql = format!("SELECT 1 FROM {table} LIMIT 1");
            let result = conn.query(&sql);
            assert!(result.is_ok(), "table {table} should exist");
        }
    }

    #[test]
    fn export_with_replay_json_field_round_trips() {
        // Verify that the replay_json column is included in exports
        // and correctly imported via ensure_runs_replay_column.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("replay_export.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("replay-rt", &db_path);
        report.replay.input_content_hash = Some("abc123".to_owned());
        report.replay.backend_identity = Some("whisper-cli".to_owned());
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Verify runs.jsonl contains replay_json field.
        let runs_content = fs::read_to_string(export_dir.join("runs.jsonl")).expect("read");
        assert!(
            runs_content.contains("replay_json"),
            "runs.jsonl should contain replay_json field"
        );
        assert!(
            runs_content.contains("abc123"),
            "runs.jsonl should contain replay hash"
        );

        // Import and verify replay data round-trips.
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        let target_store = RunStore::open(&target_db).expect("open target");
        let details = target_store
            .load_run_details("replay-rt")
            .expect("load")
            .expect("exists");
        assert_eq!(details.replay.input_content_hash.as_deref(), Some("abc123"));
        assert_eq!(
            details.replay.backend_identity.as_deref(),
            Some("whisper-cli")
        );
    }

    // ── Ninth-pass edge case tests ──

    #[test]
    fn is_lock_stale_returns_false_for_alive_pid_with_aged_timestamp() {
        // PID is alive (current process) but timestamp is >300 seconds old.
        // The lock should remain valid while the process is alive.
        let old_time = Utc::now() - chrono::Duration::seconds(LOCK_STALE_SECONDS + 60);
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: old_time.to_rfc3339(),
            operation: "export".to_owned(),
        };
        assert!(
            !is_lock_stale(&info),
            "alive PID should keep lock even if timestamp is old"
        );
    }

    #[test]
    fn verify_schema_exists_rejects_missing_segments_table() {
        // DB has runs + events tables but NO segments → should fail on iteration 2.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_segments.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, \
             finished_at TEXT NOT NULL, backend TEXT NOT NULL, input_path TEXT NOT NULL, \
             normalized_wav_path TEXT NOT NULL, request_json TEXT NOT NULL, \
             result_json TEXT NOT NULL, warnings_json TEXT NOT NULL, transcript TEXT NOT NULL);",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE events (run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
             ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, code TEXT NOT NULL, \
             message TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, seq));",
        )
        .expect("create events");

        let err = verify_schema_exists(&conn).expect_err("missing segments should fail");
        let text = err.to_string();
        assert!(
            text.contains("missing table `segments`"),
            "should mention segments: {text}"
        );
    }

    #[test]
    fn sync_lock_file_records_operation_name() {
        // The lock file should contain the operation name as JSON.
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");

        let lock = SyncLock::acquire(&state_root, "test_op").expect("acquire");
        let lock_path = state_root.join("locks").join("sync.lock");
        assert!(lock_path.exists(), "lock file should exist");

        let contents = fs::read_to_string(&lock_path).expect("read lock");
        let info: LockInfo = serde_json::from_str(&contents).expect("parse lock JSON");
        assert_eq!(info.operation, "test_op");
        assert_eq!(info.pid, std::process::id());
        assert!(!info.created_at_rfc3339.is_empty());

        drop(lock); // Release cleans up
        assert!(!lock_path.exists(), "lock should be removed after drop");
    }

    #[test]
    fn export_import_row_counts_preserved_in_round_trip() {
        // Export a DB with specific counts, then verify the manifest row counts
        // match, and after import into a fresh DB the counts still hold.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("counts.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        for i in 0..3 {
            store
                .persist_report(&fixture_report(&format!("run-{i}"), &db_path))
                .expect("persist");
        }
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 3);
        // fixture_report creates 2 segments and 2 events per run
        assert_eq!(manifest.row_counts.segments, 6);
        assert_eq!(manifest.row_counts.events, 6);

        let target_db = dir.path().join("target.sqlite3");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 3);
        assert_eq!(result.segments_imported, 6);
        assert_eq!(result.events_imported, 6);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn validate_checksums_fails_on_first_missing_file() {
        // validate_checksums iterates ["runs.jsonl", "segments.jsonl", "events.jsonl"]
        // and should fail on the first missing file (runs.jsonl).
        let dir = tempdir().expect("tempdir");
        let input_dir = dir.path().join("partial");
        fs::create_dir_all(&input_dir).expect("mkdir");

        // Create only segments.jsonl and events.jsonl but NOT runs.jsonl.
        fs::write(input_dir.join("segments.jsonl"), "").expect("write");
        fs::write(input_dir.join("events.jsonl"), "").expect("write");

        let manifest = SyncManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: "fake".to_owned(),
                segments_jsonl_sha256: "fake".to_owned(),
                events_jsonl_sha256: "fake".to_owned(),
            },
        };

        let err = validate_checksums(&manifest, &input_dir).expect_err("missing runs.jsonl");
        let text = err.to_string();
        assert!(
            text.contains("missing export file") && text.contains("runs.jsonl"),
            "should fail on runs.jsonl specifically: {text}"
        );
    }

    #[test]
    fn import_events_with_seq_as_string_returns_missing_seq_error() {
        // When seq is present but a string (not integer), as_i64() returns None
        // and import_events should return "missing seq in events row" error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("string-seq", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Overwrite events.jsonl with seq as a string instead of integer.
        let events_path = export_dir.join("events.jsonl");
        fs::write(
            &events_path,
            r#"{"run_id":"string-seq","seq":"1","ts_rfc3339":"2026-01-01T00:00:01Z","stage":"ingest","code":"ingest.ok","message":"test","payload_json":"{}"}"#,
        )
        .expect("write");

        // Update manifest checksums and row count for the modified file.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_val: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_val["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&events_path).expect("hash"));
        manifest_val["row_counts"]["events"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_val).expect("ser"),
        )
        .expect("write manifest");

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
        let err = result.expect_err("string seq should fail import");
        assert!(
            err.to_string().contains("missing seq"),
            "error should mention missing seq: {}",
            err
        );
    }

    #[test]
    fn import_segments_with_idx_as_float_returns_missing_idx_error() {
        // idx as 0.5 (float, not integer): as_i64() returns None → "missing idx" error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("float-idx", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let seg_path = export_dir.join("segments.jsonl");
        fs::write(
            &seg_path,
            r#"{"run_id":"float-idx","idx":0.5,"text":"bad","start_sec":null,"end_sec":null,"speaker":null,"confidence":null}"#,
        )
        .expect("write");

        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_val: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_val["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("hash"));
        manifest_val["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_val).expect("ser"),
        )
        .expect("write manifest");

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
        let err = result.expect_err("float idx should fail import");
        assert!(
            err.to_string().contains("missing idx"),
            "error should mention missing idx: {}",
            err
        );
    }

    #[test]
    fn lock_acquire_fails_when_state_root_path_is_invalid() {
        // Attempting to create locks dir under /dev/null (not a directory) should fail.
        let state_root = PathBuf::from("/dev/null");
        let result = SyncLock::acquire(&state_root, "test_op");
        assert!(result.is_err(), "lock acquire under /dev/null should fail");
    }

    #[test]
    fn import_tolerates_extra_unknown_fields_in_jsonl_rows() {
        // JSONL rows may contain extra fields beyond what import expects.
        // Since parsing uses serde_json::Value (not strict struct deserialization),
        // extra fields should be silently ignored.
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(
            &export_dir,
            &[json!({
                "id": "extra-fields-run",
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:05Z",
                "backend": "whisper_cpp",
                "input_path": "test.wav",
                "normalized_wav_path": "norm.wav",
                "request_json": "{}",
                "result_json": "{\"backend\":\"whisper_cpp\",\"transcript\":\"hello\",\"language\":null,\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}",
                "warnings_json": "[]",
                "transcript": "hello",
                "replay_json": "{}",
                "extra_field_1": "should be ignored",
                "extra_field_2": 42,
                "metadata": {"nested": true},
            })],
            &[json!({
                "run_id": "extra-fields-run",
                "idx": 0,
                "text": "hello",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "speaker": null,
                "confidence": null,
                "bonus_field": "ignored",
            })],
            &[json!({
                "run_id": "extra-fields-run",
                "seq": 1,
                "ts_rfc3339": "2026-01-01T00:00:01Z",
                "stage": "backend",
                "code": "backend.ok",
                "message": "done",
                "payload_json": "{}",
                "unknown_col": [1, 2, 3],
            })],
        );

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import should succeed despite extra fields");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 1);
        assert_eq!(result.events_imported, 1);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn archive_stale_lock_creates_file_with_json_extension() {
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");
        let locks_dir = state_root.join("locks");
        fs::create_dir_all(&locks_dir).expect("locks dir");

        // Create a stale lock (old timestamp, dead pid).
        let old_time = Utc::now() - chrono::Duration::seconds(LOCK_STALE_SECONDS + 120);
        let stale_info = LockInfo {
            pid: u32::MAX - 1,
            created_at_rfc3339: old_time.to_rfc3339(),
            operation: "export".to_owned(),
        };
        fs::write(
            locks_dir.join("sync.lock"),
            serde_json::to_string(&stale_info).expect("ser"),
        )
        .expect("write stale lock");

        // Acquire should archive the stale lock and succeed.
        let lock = SyncLock::acquire(&state_root, "new_op").expect("acquire");
        lock.release().expect("release");

        // Verify the archived file has the expected naming pattern: sync.lock.stale.{timestamp}.json
        let archived_names: Vec<String> = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("sync.lock.stale."))
            .collect();
        assert_eq!(archived_names.len(), 1, "one stale archive expected");
        assert!(
            archived_names[0].ends_with(".json"),
            "archived filename should end with .json: {}",
            archived_names[0]
        );
    }

    #[test]
    fn validate_schema_version_empty_string_rejects() {
        let manifest = SyncManifest {
            schema_version: String::new(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: String::new(),
                segments_jsonl_sha256: String::new(),
                events_jsonl_sha256: String::new(),
            },
        };
        let err = validate_schema_version(&manifest).expect_err("empty version should fail");
        assert!(
            err.to_string().contains("schema version mismatch"),
            "error: {}",
            err
        );
    }

    #[test]
    fn exported_runs_jsonl_has_all_expected_keys() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("keys-run", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let content = fs::read_to_string(export_dir.join("runs.jsonl")).expect("read");
        let line = content
            .lines()
            .next()
            .expect("should have at least one line");
        let row: serde_json::Value = serde_json::from_str(line).expect("parse");
        let obj = row.as_object().expect("should be object");

        let expected_keys = [
            "id",
            "started_at",
            "finished_at",
            "backend",
            "input_path",
            "normalized_wav_path",
            "request_json",
            "result_json",
            "warnings_json",
            "transcript",
            "replay_json",
            "acceleration_json",
        ];
        for key in &expected_keys {
            assert!(
                obj.contains_key(*key),
                "runs JSONL should contain key `{key}`"
            );
        }
        assert_eq!(
            obj.len(),
            expected_keys.len(),
            "should have exactly {} keys, got {}",
            expected_keys.len(),
            obj.len()
        );
    }

    #[test]
    fn exported_segments_preserve_null_optional_fields() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let mut report = fixture_report("null-opts", &db_path);
        report.result.segments = vec![crate::model::TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: "bare text".to_owned(),
            speaker: None,
            confidence: None,
        }];
        let store = RunStore::open(&db_path).expect("store");
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let content = fs::read_to_string(export_dir.join("segments.jsonl")).expect("read");
        let line = content.lines().next().expect("should have segment line");
        let row: serde_json::Value = serde_json::from_str(line).expect("parse");

        assert!(row["start_sec"].is_null(), "None start_sec → null");
        assert!(row["end_sec"].is_null(), "None end_sec → null");
        assert!(row["speaker"].is_null(), "None speaker → null");
        assert!(row["confidence"].is_null(), "None confidence → null");
        assert_eq!(row["text"], "bare text");
    }

    #[test]
    fn validate_checksums_succeeds_when_all_match() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("chk-ok", &db_path))
            .expect("persist");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        validate_checksums(&manifest, &export_dir).expect("checksums should match");
    }

    #[test]
    fn import_runs_without_replay_json_field_uses_default() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        write_jsonl_snapshot(
            &export_dir,
            &[json!({
                "id": "no-replay",
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:05Z",
                "backend": "whisper_cpp",
                "input_path": "test.wav",
                "normalized_wav_path": "norm.wav",
                "request_json": "{}",
                "result_json": "{\"backend\":\"whisper_cpp\",\"transcript\":\"hello\",\"language\":null,\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}",
                "warnings_json": "[]",
                "transcript": "hello",
            })],
            &[],
            &[],
        );

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import should succeed without replay_json");
        assert_eq!(result.runs_imported, 1);

        let store = RunStore::open(&target_db).expect("store");
        let details = store
            .load_run_details("no-replay")
            .expect("query")
            .expect("exists");
        assert!(
            details.replay.input_content_hash.is_none(),
            "missing replay_json should default to empty envelope"
        );
    }

    #[test]
    fn import_row_count_mismatch_for_events_errors() {
        // Runs and segments mismatch are tested; verify events mismatch too.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("evt-mis", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Inflate the manifest's events row count so it doesn't match actual data.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read")).expect("json");
        manifest_value["row_counts"]["events"] = json!(999);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write");

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("events row count mismatch should error");
        let msg = err.to_string();
        assert!(
            msg.contains("row count mismatch") && msg.contains("events"),
            "error should mention events mismatch, got: {msg}"
        );
    }

    #[test]
    fn exported_events_jsonl_has_all_expected_keys() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("evt_keys.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("evt-keys", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let events_text =
            fs::read_to_string(export_dir.join("events.jsonl")).expect("events.jsonl");
        let first_line = events_text.lines().next().expect("at least one event line");
        let obj: serde_json::Value = serde_json::from_str(first_line).expect("parse");
        let map = obj.as_object().expect("should be object");
        let expected_keys = [
            "run_id",
            "seq",
            "ts_rfc3339",
            "stage",
            "code",
            "message",
            "payload_json",
        ];
        for key in &expected_keys {
            assert!(map.contains_key(*key), "events JSONL missing key: {key}");
        }
        assert_eq!(
            map.len(),
            expected_keys.len(),
            "events JSONL should have exactly {} keys",
            expected_keys.len()
        );
    }

    #[test]
    fn import_identical_segment_twice_is_noop() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("seg_noop.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("seg-noop", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Import into same DB — segments already exist with identical data.
        let result = import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("identical re-import should succeed");
        assert!(
            result.conflicts.is_empty(),
            "identical segment data should produce no conflicts"
        );
        assert!(
            result.segments_imported > 0,
            "segments should still be counted"
        );
    }

    #[test]
    fn is_lock_stale_returns_false_for_alive_fresh_lock() {
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: Utc::now().to_rfc3339(),
            operation: "test".to_owned(),
        };
        assert!(
            !is_lock_stale(&info),
            "lock with current pid and fresh timestamp should not be stale"
        );
    }

    #[test]
    fn json_str_returns_error_for_integer_value() {
        let value = json!({"id": 42});
        let err = json_str(&value, "id").expect_err("integer should fail as_str");
        assert!(
            err.to_string().contains("missing string field"),
            "error should mention missing string field, got: {}",
            err
        );
    }

    // ── Comprehensive edge-case tests (added) ──

    #[test]
    fn export_empty_db_produces_valid_jsonl_files() {
        // Verify that exporting an empty DB yields syntactically valid (empty)
        // JSONL files that can be successfully re-imported.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_jsonl.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let _store = RunStore::open(&db_path).expect("store open");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        // All JSONL files should exist and be empty.
        for filename in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            let path = export_dir.join(filename);
            assert!(path.exists(), "{filename} should exist");
            let content = fs::read_to_string(&path).expect("read");
            assert!(
                content.trim().is_empty(),
                "{filename} should be empty for an empty DB, got: {content:?}"
            );
        }

        // Checksums should be for empty files.
        assert_eq!(
            manifest.checksums.runs_jsonl_sha256, manifest.checksums.segments_jsonl_sha256,
            "all checksums should be identical for empty files"
        );

        // Re-importing should succeed with zero rows.
        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import of empty export");
        assert_eq!(result.runs_imported, 0);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 0);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn import_with_reject_policy_fails_on_different_duplicates() {
        // Import data, then import the same run IDs with different payloads
        // under Reject policy. The import should fail and produce conflicts.
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let run_json = json!({
            "id": "reject-dup-1",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:05Z",
            "backend": "whisper_cpp",
            "input_path": "test.wav",
            "normalized_wav_path": "norm.wav",
            "request_json": "{}",
            "result_json": "{\"backend\":\"whisper_cpp\",\"transcript\":\"original\",\"language\":null,\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}",
            "warnings_json": "[]",
            "transcript": "original transcript",
            "replay_json": "{}",
        });

        // First import.
        write_jsonl_snapshot(&export_dir, std::slice::from_ref(&run_json), &[], &[]);
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("first import should succeed");

        // Second import with different transcript.
        let export_dir_2 = dir.path().join("export2");
        let mut mutated = run_json.clone();
        mutated["transcript"] = json!("DIFFERENT transcript");
        write_jsonl_snapshot(&export_dir_2, &[mutated], &[], &[]);

        let err = import(
            &target_db,
            &export_dir_2,
            &state_root,
            ConflictPolicy::Reject,
        )
        .expect_err("reject policy should fail on different duplicate");
        let text = err.to_string();
        assert!(
            text.contains("import rejected due to"),
            "error should mention rejection: {text}"
        );
    }

    #[test]
    fn import_identical_data_twice_with_reject_is_noop() {
        // Importing identical data twice under Reject policy should succeed
        // (no conflicts because the data is the same -- effectively a "skip").
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let run_json = json!({
            "id": "skip-dup-1",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:05Z",
            "backend": "whisper_cpp",
            "input_path": "test.wav",
            "normalized_wav_path": "norm.wav",
            "request_json": "{}",
            "result_json": "{\"backend\":\"whisper_cpp\",\"transcript\":\"hello\",\"language\":null,\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}",
            "warnings_json": "[]",
            "transcript": "hello",
            "replay_json": "{}",
        });
        let seg_json = json!({
            "run_id": "skip-dup-1",
            "idx": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "speaker": null,
            "text": "hello",
            "confidence": 0.9,
        });
        let evt_json = json!({
            "run_id": "skip-dup-1",
            "seq": 1,
            "ts_rfc3339": "2026-01-01T00:00:01Z",
            "stage": "backend",
            "code": "backend.ok",
            "message": "done",
            "payload_json": "{}",
        });

        write_jsonl_snapshot(
            &export_dir,
            std::slice::from_ref(&run_json),
            std::slice::from_ref(&seg_json),
            std::slice::from_ref(&evt_json),
        );

        // First import.
        let r1 = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("first import");
        assert_eq!(r1.runs_imported, 1);
        assert_eq!(r1.segments_imported, 1);
        assert_eq!(r1.events_imported, 1);
        assert!(r1.conflicts.is_empty());

        // Second import of identical data.
        let r2 = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("second import of identical data should succeed");
        assert_eq!(r2.runs_imported, 1);
        assert_eq!(r2.segments_imported, 1);
        assert_eq!(r2.events_imported, 1);
        assert!(
            r2.conflicts.is_empty(),
            "identical re-import should produce zero conflicts (skip semantics)"
        );

        // Verify DB still has exactly 1 run, 1 segment, 1 event.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn.query("SELECT count(*) FROM runs").expect("query");
        let count = match rows[0].get(0) {
            Some(SqliteValue::Integer(n)) => *n,
            other => {
                assert!(
                    matches!(other, Some(SqliteValue::Integer(_))),
                    "expected integer, got {other:?}"
                );
                0
            }
        };
        assert_eq!(count, 1, "should still have exactly 1 run after 2 imports");
    }

    #[test]
    fn checksum_validation_catches_post_export_corruption() {
        // Export from a real DB, then modify a JSONL file, then attempt
        // import. The checksum validation should catch the corruption.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("corrupt-check", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt segments.jsonl by appending extra data (simulating file damage).
        let seg_path = export_dir.join("segments.jsonl");
        let mut seg_file = OpenOptions::new()
            .append(true)
            .open(&seg_path)
            .expect("open segments for append");
        seg_file
            .write_all(b"CORRUPTED DATA APPENDED\n")
            .expect("append corruption");
        drop(seg_file);

        // Do NOT update manifest -- the whole point is the checksum won't match.
        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("corrupted file should fail checksum validation");
        let text = err.to_string();
        assert!(
            text.contains("checksum mismatch") && text.contains("segments.jsonl"),
            "error should mention segments.jsonl checksum mismatch: {text}"
        );
    }

    #[test]
    fn export_creates_manifest_with_correct_checksums() {
        // Verify the manifest file exists, is valid JSON, and its checksums
        // match the actual SHA-256 of the exported JSONL files.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("manifest_check.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("manifest-chk", &db_path))
            .expect("persist");
        let manifest = export(&db_path, &export_dir, &state_root).expect("export");

        // Verify manifest.json exists on disk.
        let manifest_path = export_dir.join("manifest.json");
        assert!(manifest_path.exists(), "manifest.json should exist on disk");

        // Parse it back from disk and compare.
        let disk_manifest: SyncManifest =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
                .expect("parse manifest from disk");

        assert_eq!(disk_manifest.schema_version, manifest.schema_version);
        assert_eq!(disk_manifest.row_counts, manifest.row_counts);
        assert_eq!(disk_manifest.checksums, manifest.checksums);

        // Independently compute checksums and verify they match.
        let runs_hash = sha256_file(&export_dir.join("runs.jsonl")).expect("hash runs");
        let segs_hash = sha256_file(&export_dir.join("segments.jsonl")).expect("hash segments");
        let evts_hash = sha256_file(&export_dir.join("events.jsonl")).expect("hash events");

        assert_eq!(manifest.checksums.runs_jsonl_sha256, runs_hash);
        assert_eq!(manifest.checksums.segments_jsonl_sha256, segs_hash);
        assert_eq!(manifest.checksums.events_jsonl_sha256, evts_hash);
    }

    #[test]
    fn import_validates_manifest_checksums_before_modifying_db() {
        // When checksums don't match, the DB should remain completely untouched.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("validate-first", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Corrupt events.jsonl.
        let evt_path = export_dir.join("events.jsonl");
        fs::write(&evt_path, "this is not valid jsonl content\n").expect("corrupt events");

        // Attempt import -- should fail at checksum validation, before any DB writes.
        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("checksum mismatch should prevent import");
        assert!(
            err.to_string().contains("checksum mismatch"),
            "should fail on checksum: {}",
            err
        );

        // Target DB should either not exist or have zero rows.
        if target_db.exists() {
            let conn = Connection::open(target_db.display().to_string()).expect("open");
            // The DB might have been created by the import attempt but schema check
            // happens after validation, so tables might not exist. That's fine.
            let tables = conn
                .query("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='runs'")
                .unwrap_or_default();
            let table_count = tables
                .first()
                .and_then(|r| match r.get(0) {
                    Some(SqliteValue::Integer(n)) => Some(*n),
                    _ => None,
                })
                .unwrap_or(0);
            if table_count > 0 {
                let rows = conn.query("SELECT count(*) FROM runs").expect("query runs");
                let run_count = match rows[0].get(0) {
                    Some(SqliteValue::Integer(n)) => *n,
                    _ => 0,
                };
                assert_eq!(
                    run_count, 0,
                    "DB should have zero runs after failed checksum validation"
                );
            }
        }
    }

    #[test]
    fn export_import_very_long_transcript() {
        // Test with a transcript exceeding 10KB to verify no truncation
        // or buffer issues in the JSONL export/import pipeline.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("long_transcript.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("long-text-1", &db_path);

        // Create a >10KB transcript.
        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(300);
        assert!(
            long_text.len() > 10_000,
            "transcript should be >10KB, got {} bytes",
            long_text.len()
        );
        report.result.transcript = long_text.clone();
        report.result.segments[0].text = long_text[..5000].to_owned();
        report.result.segments[1].text = long_text[5000..].to_owned();
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import long transcript");
        assert_eq!(result.runs_imported, 1);
        assert!(result.conflicts.is_empty());

        // Verify full text survived the round trip.
        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("long-text-1")
            .expect("load")
            .expect("exists");
        assert_eq!(
            details.transcript.len(),
            long_text.len(),
            "transcript length should be preserved"
        );
        assert_eq!(details.transcript, long_text);
    }

    #[test]
    fn export_import_unicode_in_all_fields() {
        // Verify Unicode survives in segments (speaker, text), events (message,
        // stage, code), and run-level fields (transcript, input_path).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("full_unicode.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("uni-all-1", &db_path);

        // Unicode in transcript
        report.result.transcript =
            "\u{1F3B5} \u{65E5}\u{672C}\u{8A9E} \u{D55C}\u{AD6D}\u{C5B4} \u{0410}\u{0411}\u{0412} caf\u{00E9}"
                .to_owned();

        // Unicode in segment text and speaker
        report.result.segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.5),
                text: "\u{4F60}\u{597D}\u{4E16}\u{754C}".to_owned(), // Chinese: "Hello World"
                speaker: Some("\u{8A71}\u{8005}_00".to_owned()),     // Japanese-style speaker label
                confidence: Some(0.92),
            },
            TranscriptionSegment {
                start_sec: Some(1.5),
                end_sec: Some(3.0),
                text: "\u{041F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043C}\u{0438}\u{0440}"
                    .to_owned(), // Russian: "Hello world"
                speaker: Some(
                    "\u{0413}\u{043E}\u{0432}\u{043E}\u{0440}\u{044F}\u{0449}\u{0438}\u{0439}_01"
                        .to_owned(),
                ),
                confidence: None,
            },
        ];

        // Unicode in events
        report.events = vec![RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            stage: "einlesen".to_owned(),                  // German
            code: "\u{00E9}v\u{00E9}nement.ok".to_owned(), // French: "evenement.ok"
            message: "\u{6210}\u{529F}".to_owned(),        // Chinese: "success"
            payload: json!({"\u{30C7}\u{30FC}\u{30BF}": "\u{5024}"}), // Japanese keys/values
        }];

        store
            .persist_report(&report)
            .expect("persist unicode report");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export unicode");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 1);

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import unicode");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 1);
        assert!(result.conflicts.is_empty());

        // Verify round-tripped data.
        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("uni-all-1")
            .expect("load")
            .expect("exists");
        assert!(details.transcript.contains('\u{1F3B5}'));
        assert!(details.transcript.contains("caf\u{00E9}"));
        assert_eq!(details.segments.len(), 2);
        assert!(details.segments[0].text.contains('\u{4F60}'));
        assert_eq!(
            details.segments[0].speaker.as_deref(),
            Some("\u{8A71}\u{8005}_00")
        );

        // Verify event data via direct SQL query.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn
            .query_with_params(
                "SELECT code, message FROM events WHERE run_id = ?1 AND seq = 1",
                &[SqliteValue::Text("uni-all-1".to_owned().into())],
            )
            .expect("query events");
        let code = value_to_string_sqlite(rows[0].get(0));
        let message = value_to_string_sqlite(rows[0].get(1));
        assert!(
            code.contains('\u{00E9}'),
            "event code should contain accented char"
        );
        assert_eq!(message, "\u{6210}\u{529F}");
    }

    #[test]
    fn export_import_empty_segments_list() {
        // A run with zero segments should export and import cleanly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_segments.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("no-segs-1", &db_path);
        report.result.segments = vec![]; // Empty segments
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 2); // fixture_report has 2 events

        // Verify segments.jsonl is empty.
        let seg_content = fs::read_to_string(export_dir.join("segments.jsonl")).expect("read");
        assert!(
            seg_content.trim().is_empty(),
            "segments.jsonl should be empty for a run with no segments"
        );

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 2);
        assert!(result.conflicts.is_empty());

        // Verify target DB has the run but zero segments.
        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("no-segs-1")
            .expect("load")
            .expect("exists");
        assert!(
            details.segments.is_empty(),
            "imported run should have zero segments"
        );
    }

    #[test]
    fn export_import_many_events() {
        // Verify that 100+ events for a single run export and import correctly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("many_events.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("many-evt-1", &db_path);

        // Generate 150 events.
        report.events = (1..=150)
            .map(|seq| RunEvent {
                seq,
                ts_rfc3339: format!("2026-01-01T00:00:{:02}Z", seq % 60),
                stage: format!("stage_{}", seq % 5),
                code: format!("code_{seq}.ok"),
                message: format!("event number {seq}"),
                payload: json!({"seq": seq, "data": "x".repeat(seq as usize % 50)}),
            })
            .collect();

        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.events, 150);

        // Verify events.jsonl has 150 lines.
        let events_content =
            fs::read_to_string(export_dir.join("events.jsonl")).expect("read events");
        let event_lines: Vec<&str> = events_content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(
            event_lines.len(),
            150,
            "should have 150 event lines in JSONL"
        );

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import many events");
        assert_eq!(result.events_imported, 150);
        assert!(result.conflicts.is_empty());

        // Spot-check a few events via direct SQL.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn
            .query_with_params(
                "SELECT count(*) FROM events WHERE run_id = ?1",
                &[SqliteValue::Text("many-evt-1".to_owned().into())],
            )
            .expect("query");
        let count = match rows[0].get(0) {
            Some(SqliteValue::Integer(n)) => *n,
            other => {
                assert!(
                    matches!(other, Some(SqliteValue::Integer(_))),
                    "expected integer count, got {other:?}"
                );
                0
            }
        };
        assert_eq!(count, 150, "DB should contain 150 events for this run");

        // Verify first and last events.
        let first = conn
            .query_with_params(
                "SELECT message FROM events WHERE run_id = ?1 AND seq = 1",
                &[SqliteValue::Text("many-evt-1".to_owned().into())],
            )
            .expect("query first");
        assert_eq!(value_to_string_sqlite(first[0].get(0)), "event number 1");

        let last = conn
            .query_with_params(
                "SELECT message FROM events WHERE run_id = ?1 AND seq = 150",
                &[SqliteValue::Text("many-evt-1".to_owned().into())],
            )
            .expect("query last");
        assert_eq!(value_to_string_sqlite(last[0].get(0)), "event number 150");
    }

    #[test]
    fn checksum_corruption_in_events_file_detected() {
        // Export, then modify only events.jsonl (leaving runs and segments intact).
        // The checksum validation should catch this specific file's corruption.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("evt-corrupt", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Modify events.jsonl slightly (change a single character).
        let evt_path = export_dir.join("events.jsonl");
        let original = fs::read_to_string(&evt_path).expect("read");
        let modified = original.replacen("ingest", "Ingest", 1); // Capitalize one word
        assert_ne!(original, modified, "modification should change content");
        fs::write(&evt_path, &modified).expect("write modified events");

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("modified events should fail checksum");
        let text = err.to_string();
        assert!(
            text.contains("checksum mismatch") && text.contains("events.jsonl"),
            "error should specifically mention events.jsonl: {text}"
        );
    }

    #[test]
    fn export_import_run_with_no_events() {
        // A run with zero events should export and import cleanly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_events.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("no-evt-1", &db_path);
        report.events = vec![]; // Empty events
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 0);

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 0);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn export_import_run_with_no_segments_and_no_events() {
        // A completely bare run (no segments, no events) round-trips.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("bare_run.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("bare-1", &db_path);
        report.result.segments = vec![];
        report.events = vec![];
        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import bare run");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 0);
        assert_eq!(result.events_imported, 0);
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("bare-1")
            .expect("load")
            .expect("exists");
        assert!(details.segments.is_empty());
    }

    #[test]
    fn sync_lock_concurrent_acquire_error_preserves_first_lock() {
        // Acquiring a second lock should fail, and the first lock should
        // remain intact and releasable.
        let dir = tempdir().expect("tempdir");
        let state_root = dir.path().join("state");

        let lock1 = SyncLock::acquire(&state_root, "export").expect("first lock");
        let lock_path = state_root.join("locks/sync.lock");
        assert!(lock_path.exists(), "lock file should exist");

        // Read the lock info to verify it belongs to the first operation.
        let info: LockInfo =
            serde_json::from_str(&fs::read_to_string(&lock_path).expect("read lock"))
                .expect("parse lock");
        assert_eq!(info.operation, "export");

        // Second acquire should fail.
        let err = SyncLock::acquire(&state_root, "import");
        assert!(err.is_err(), "second acquire should fail");

        // First lock's file should still be there and unchanged.
        assert!(
            lock_path.exists(),
            "first lock file should still exist after failed second acquire"
        );
        let info_after: LockInfo =
            serde_json::from_str(&fs::read_to_string(&lock_path).expect("read lock again"))
                .expect("parse lock again");
        assert_eq!(
            info_after.operation, "export",
            "lock file should still reflect first operation"
        );
        assert_eq!(info_after.pid, info.pid);

        // Release the first lock.
        lock1.release().expect("release first lock");
        assert!(
            !lock_path.exists(),
            "lock file should be removed after release"
        );
    }

    #[test]
    fn export_import_many_segments() {
        // Verify that 50+ segments for a single run export and import correctly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("many_segments.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("many-seg-1", &db_path);

        // Generate 60 segments.
        report.result.segments = (0..60)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64 * 2.0),
                end_sec: Some((i as f64 * 2.0) + 1.9),
                text: format!("segment number {i} with some text content"),
                speaker: if i % 3 == 0 {
                    Some(format!("SPEAKER_{:02}", i % 5))
                } else {
                    None
                },
                confidence: if i % 2 == 0 {
                    Some(0.8 + (i as f64 * 0.001))
                } else {
                    None
                },
            })
            .collect();

        store.persist_report(&report).expect("persist");

        let manifest = export(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.segments, 60);

        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import many segments");
        assert_eq!(result.segments_imported, 60);
        assert!(result.conflicts.is_empty());

        // Verify count in target DB.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn
            .query_with_params(
                "SELECT count(*) FROM segments WHERE run_id = ?1",
                &[SqliteValue::Text("many-seg-1".to_owned().into())],
            )
            .expect("query");
        let count = match rows[0].get(0) {
            Some(SqliteValue::Integer(n)) => *n,
            other => {
                assert!(
                    matches!(other, Some(SqliteValue::Integer(_))),
                    "expected integer count, got {other:?}"
                );
                0
            }
        };
        assert_eq!(count, 60, "DB should contain 60 segments");

        // Spot-check last segment.
        let last = conn
            .query_with_params(
                "SELECT text FROM segments WHERE run_id = ?1 AND idx = 59",
                &[SqliteValue::Text("many-seg-1".to_owned().into())],
            )
            .expect("query last segment");
        assert_eq!(
            value_to_string_sqlite(last[0].get(0)),
            "segment number 59 with some text content"
        );
    }

    #[test]
    fn overwrite_policy_on_identical_data_is_still_noop() {
        // Overwrite policy with identical data should succeed without errors
        // (identical data = noop, just like Reject).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("ow-noop-1", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import.
        import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("first import");

        // Second import with Overwrite -- identical data should be a noop.
        let result = import(
            &target_db,
            &export_dir,
            &state_root,
            ConflictPolicy::Overwrite,
        )
        .expect("second import should succeed");
        assert_eq!(result.runs_imported, 1);
        assert!(
            result.conflicts.is_empty(),
            "identical data under Overwrite should produce no conflicts"
        );

        // DB should still have exactly 1 run.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn.query("SELECT count(*) FROM runs").expect("query");
        let count = match rows[0].get(0) {
            Some(SqliteValue::Integer(n)) => *n,
            _ => 0,
        };
        assert_eq!(count, 1, "should have exactly 1 run");
    }

    #[test]
    fn checksum_corruption_in_runs_file_detected() {
        // Corrupt only the runs.jsonl file; segments and events remain valid.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("runs-corrupt", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Append a byte to runs.jsonl to invalidate its checksum.
        let runs_path = export_dir.join("runs.jsonl");
        let mut f = OpenOptions::new()
            .append(true)
            .open(&runs_path)
            .expect("open");
        f.write_all(b" ").expect("append space");
        drop(f);

        let err = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("corrupted runs.jsonl should fail checksum");
        let text = err.to_string();
        assert!(
            text.contains("checksum mismatch") && text.contains("runs.jsonl"),
            "error should mention runs.jsonl: {text}"
        );
    }

    #[test]
    fn export_import_transcript_with_embedded_json() {
        // Transcript containing valid JSON strings, braces, brackets -- should
        // survive JSONL encoding without corruption.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("json_transcript.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        let mut report = fixture_report("json-in-text", &db_path);
        report.result.transcript =
            r#"The output was {"key": "value", "array": [1,2,3]} and that's it."#.to_owned();
        report.result.segments[0].text = r#"{"nested": {"deep": true}}"#.to_owned();
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");
        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let details = target_store
            .load_run_details("json-in-text")
            .expect("load")
            .expect("exists");
        assert!(details.transcript.contains(r#""key": "value""#));
        assert!(details.segments[0].text.contains(r#""deep": true"#));
    }

    #[test]
    fn import_creates_parent_directories_for_target_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&fixture_report("parent-dir", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Target DB in nested non-existent directory.
        let target_db = dir
            .path()
            .join("a")
            .join("b")
            .join("c")
            .join("target.sqlite3");
        assert!(!target_db.parent().unwrap().exists());

        let result =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 1);
        assert!(target_db.exists(), "target DB should have been created");
    }

    #[test]
    fn exported_runs_jsonl_ordered_ascending_by_started_at() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("order.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        // Insert runs in non-chronological order.
        for (id, ts) in [
            ("run-c", "2026-03-01T00:00:00Z"),
            ("run-a", "2026-01-01T00:00:00Z"),
            ("run-b", "2026-02-01T00:00:00Z"),
        ] {
            let mut report = fixture_report(id, &db_path);
            report.started_at_rfc3339 = ts.to_owned();
            store.persist_report(&report).expect("persist");
        }

        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_text = fs::read_to_string(export_dir.join("runs.jsonl")).expect("runs.jsonl");
        let ids: Vec<String> = runs_text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).expect("parse");
                v["id"].as_str().unwrap().to_owned()
            })
            .collect();
        // ORDER BY started_at ASC: a (Jan) < b (Feb) < c (Mar).
        assert_eq!(ids, vec!["run-a", "run-b", "run-c"]);
    }

    #[test]
    fn json_to_optional_f64_converts_integer_to_f64() {
        // JSON integer values should be converted to f64 via as_f64().
        let value = json!({"count": 42, "score": 2.5, "label": "text"});
        assert_eq!(
            json_to_optional_f64(&value, "count"),
            Some(42.0),
            "integer JSON value should convert to f64"
        );
        assert_eq!(
            json_to_optional_f64(&value, "score"),
            Some(2.5),
            "float JSON value should pass through"
        );
        assert_eq!(
            json_to_optional_f64(&value, "label"),
            None,
            "string JSON value should be None"
        );
    }

    #[test]
    fn atomic_write_bytes_overwrites_existing_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("overwrite_me.txt");

        atomic_write_bytes(&path, b"original content").expect("first write");
        assert_eq!(fs::read_to_string(&path).expect("read"), "original content");

        atomic_write_bytes(&path, b"new content").expect("overwrite");
        assert_eq!(fs::read_to_string(&path).expect("read"), "new content");
    }

    #[test]
    fn value_to_json_float_nan_serializes_as_null() {
        // serde_json::json!(f64::NAN) produces null.
        let result = value_to_json(Some(&SqliteValue::Float(f64::NAN)));
        assert!(
            result.is_null(),
            "NaN float should serialize as null via json!, got: {result}"
        );
    }

    #[test]
    fn json_optional_float_with_integer_value_returns_float() {
        // json!(42) is a Number whose as_f64() returns Some(42.0).
        // Existing tests only use float values (42.5); this tests the integer path.
        let value = json!({"count": 42, "neg": -7});
        match json_optional_float(&value, "count") {
            SqliteValue::Float(f) => assert!((f - 42.0).abs() < f64::EPSILON),
            other => assert!(
                matches!(other, SqliteValue::Float(_)),
                "expected Float(42.0), got {other:?}"
            ),
        }
        match json_optional_float(&value, "neg") {
            SqliteValue::Float(f) => assert!((f - (-7.0)).abs() < f64::EPSILON),
            other => assert!(
                matches!(other, SqliteValue::Float(_)),
                "expected Float(-7.0), got {other:?}"
            ),
        }
    }

    #[test]
    fn exported_segments_jsonl_ordered_by_run_id_and_idx() {
        // export_table_segments uses ORDER BY run_id ASC, idx ASC.
        // Persist two runs with segments, then verify export order.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("order_seg.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Persist run "bbb" first, then "aaa" — export should reorder by run_id ASC.
        let mut report_b = fixture_report("bbb-run", &db_path);
        report_b.result.segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "seg-b-0".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: "seg-b-1".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        store.persist_report(&report_b).expect("persist b");

        let mut report_a = fixture_report("aaa-run", &db_path);
        report_a.result.segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "seg-a-0".to_owned(),
            speaker: None,
            confidence: None,
        }];
        store.persist_report(&report_a).expect("persist a");

        let export_dir = dir.path().join("export");
        let seg_path = export_dir.join("segments.jsonl");
        let manifest = export_inner(&db_path, &export_dir).expect("export");
        assert_eq!(manifest.row_counts.segments, 3);

        let lines: Vec<serde_json::Value> = fs::read_to_string(&seg_path)
            .expect("read")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("parse"))
            .collect();
        assert_eq!(lines.len(), 3);
        // First should be aaa-run (ASC by run_id)
        assert_eq!(lines[0]["run_id"], "aaa-run");
        assert_eq!(lines[0]["text"], "seg-a-0");
        // Then bbb-run idx=0, idx=1
        assert_eq!(lines[1]["run_id"], "bbb-run");
        assert_eq!(lines[1]["text"], "seg-b-0");
        assert_eq!(lines[2]["run_id"], "bbb-run");
        assert_eq!(lines[2]["text"], "seg-b-1");
    }

    #[test]
    fn exported_events_jsonl_ordered_by_run_id_and_seq() {
        // export_table_events uses ORDER BY run_id ASC, seq ASC.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("order_evt.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Persist "zzz" first, "aaa" second — export should reorder by run_id ASC.
        let report_z = fixture_report("zzz-run", &db_path);
        store.persist_report(&report_z).expect("persist z");

        let report_a = fixture_report("aaa-run", &db_path);
        store.persist_report(&report_a).expect("persist a");

        let export_dir = dir.path().join("export");
        let evt_path = export_dir.join("events.jsonl");
        let _manifest = export_inner(&db_path, &export_dir).expect("export");

        let lines: Vec<serde_json::Value> = fs::read_to_string(&evt_path)
            .expect("read")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("parse"))
            .collect();
        // Events from aaa-run should come before zzz-run events.
        let first_run_id = lines[0]["run_id"].as_str().expect("run_id");
        let last_run_id = lines.last().expect("last")["run_id"]
            .as_str()
            .expect("run_id");
        assert_eq!(first_run_id, "aaa-run");
        assert_eq!(last_run_id, "zzz-run");
        // Within each run, seq should be ascending.
        let aaa_seqs: Vec<u64> = lines
            .iter()
            .filter(|l| l["run_id"] == "aaa-run")
            .map(|l| l["seq"].as_u64().expect("seq"))
            .collect();
        for window in aaa_seqs.windows(2) {
            assert!(window[0] <= window[1], "seqs should be ascending");
        }
    }

    #[test]
    fn import_segments_missing_text_field_returns_error() {
        // import_segments calls json_str(&row, "text")? which errors on missing field.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("missing_text.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let report = fixture_report("run-1", &db_path);
        store.persist_report(&report).expect("persist");

        let export_dir = dir.path().join("export");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite segments.jsonl with a row missing the "text" field.
        let seg_path = export_dir.join("segments.jsonl");
        let bad_segment = json!({
            "run_id": "run-1",
            "idx": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "speaker": null,
            "confidence": null
            // "text" field intentionally omitted
        });
        fs::write(
            &seg_path,
            format!("{}\n", serde_json::to_string(&bad_segment).expect("ser")),
        )
        .expect("write");

        // Update manifest checksums and row counts.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        manifest_value["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write manifest");

        // Import into a fresh DB — should fail with "missing string field `text`".
        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("should fail on missing text");
        assert!(
            err.to_string().contains("missing string field") && err.to_string().contains("text"),
            "error should mention missing text field: {}",
            err
        );
    }

    #[test]
    fn import_events_missing_message_field_returns_error() {
        // import_events calls json_str(&row, "message")? which errors on missing field.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("missing_msg.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let report = fixture_report("run-1", &db_path);
        store.persist_report(&report).expect("persist");

        let export_dir = dir.path().join("export");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite events.jsonl with a row missing "message".
        let evt_path = export_dir.join("events.jsonl");
        let bad_event = json!({
            "run_id": "run-1",
            "seq": 1,
            "ts_rfc3339": "2026-01-01T00:00:01Z",
            "stage": "ingest",
            "code": "ingest.ok",
            "payload_json": "{}"
            // "message" intentionally omitted
        });
        fs::write(
            &evt_path,
            format!("{}\n", serde_json::to_string(&bad_event).expect("ser")),
        )
        .expect("write");

        // Update manifest checksums and row counts.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        manifest_value["row_counts"]["events"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write manifest");

        // Import into a fresh DB — should fail.
        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("should fail on missing message");
        assert!(
            err.to_string().contains("missing string field") && err.to_string().contains("message"),
            "error should mention missing message field: {}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // bd-246.1: Incremental export tests
    // -----------------------------------------------------------------------

    #[test]
    fn incremental_export_first_run_exports_all_records() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_first.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("incr-1", &db_path);
        store.persist_report(&report).expect("persist");

        let manifest =
            export_incremental(&db_path, &export_dir, &state_root).expect("incremental export");

        assert_eq!(manifest.export_mode, "incremental");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.segments, 2);
        assert_eq!(manifest.row_counts.events, 2);
        assert!(
            manifest.cursor_used.is_none(),
            "first export has no prior cursor"
        );
        assert!(manifest.cursor_after.last_mutation_seq > 0);
        assert!(!manifest.cursor_after.last_export_rfc3339.is_empty());
        assert_eq!(manifest.cursor_after.last_run_count, 1);
    }

    #[test]
    fn incremental_export_uses_one_snapshot_and_next_delta_cannot_skip_writer() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_snapshot.sqlite3");
        let state_root = dir.path().join("state");
        let first_export_dir = dir.path().join("export1");
        let second_export_dir = dir.path().join("export2");

        let store = RunStore::open(&db_path).expect("store open");
        let mut first_report = fixture_report("snapshot-first", &db_path);
        first_report.finished_at_rfc3339 = "2026-01-01T00:00:05Z".to_owned();
        store
            .persist_report(&first_report)
            .expect("persist first report");

        let (runs_exported_tx, runs_exported_rx) = std::sync::mpsc::sync_channel(0);
        let (writer_committed_tx, writer_committed_rx) = std::sync::mpsc::sync_channel(0);
        let export_db_path = db_path.clone();
        let export_state_root = state_root.clone();
        let export_output_dir = first_export_dir.clone();
        let export_thread = std::thread::spawn(move || {
            export_incremental_inner_with_after_runs(
                &export_db_path,
                &export_output_dir,
                &export_state_root,
                || {
                    runs_exported_tx.send(()).expect("signal runs exported");
                    writer_committed_rx
                        .recv()
                        .expect("wait for concurrent writer");
                },
            )
        });

        runs_exported_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("wait for runs export");
        let mut second_report = fixture_report("snapshot-second", &db_path);
        second_report.finished_at_rfc3339 = "2026-06-15T12:00:05Z".to_owned();
        store
            .persist_report(&second_report)
            .expect("persist concurrent report");
        writer_committed_tx
            .send(())
            .expect("release snapshot export");

        let first_manifest = export_thread
            .join()
            .expect("snapshot export thread")
            .expect("snapshot export");
        assert_eq!(first_manifest.row_counts.runs, 1);
        assert_eq!(first_manifest.row_counts.segments, 2);
        assert_eq!(first_manifest.row_counts.events, 2);
        assert_eq!(
            first_manifest.cursor_after.last_export_run_id.as_deref(),
            Some("snapshot-first")
        );

        for (file_name, run_id_key) in [
            ("runs.jsonl", "id"),
            ("segments.jsonl", "run_id"),
            ("events.jsonl", "run_id"),
        ] {
            let rows: Vec<serde_json::Value> =
                fs::read_to_string(first_export_dir.join(file_name))
                    .expect("read first snapshot file")
                    .lines()
                    .map(|line| serde_json::from_str(line).expect("parse first snapshot row"))
                    .collect();
            assert!(!rows.is_empty(), "{file_name} should contain first run rows");
            assert!(
                rows.iter()
                    .all(|row| row[run_id_key] == "snapshot-first"),
                "{file_name} must wholly exclude the concurrent run"
            );
        }

        let second_manifest = export_incremental_inner(
            &db_path,
            &second_export_dir,
            &state_root,
        )
        .expect("next delta export");
        assert_eq!(second_manifest.row_counts.runs, 1);
        assert_eq!(second_manifest.row_counts.segments, 2);
        assert_eq!(second_manifest.row_counts.events, 2);
        assert_eq!(
            second_manifest
                .cursor_used
                .as_ref()
                .and_then(|cursor| cursor.last_export_run_id.as_deref()),
            Some("snapshot-first")
        );
        assert_eq!(
            second_manifest.cursor_after.last_export_run_id.as_deref(),
            Some("snapshot-second")
        );
        assert!(
            second_manifest.cursor_after.last_mutation_seq
                > second_manifest
                    .cursor_used
                    .as_ref()
                    .expect("prior snapshot cursor")
                    .last_mutation_seq,
            "the concurrent commit must remain beyond the first snapshot high-water mark"
        );

        for (file_name, run_id_key) in [
            ("runs.jsonl", "id"),
            ("segments.jsonl", "run_id"),
            ("events.jsonl", "run_id"),
        ] {
            let rows: Vec<serde_json::Value> =
                fs::read_to_string(second_export_dir.join(file_name))
                    .expect("read next delta file")
                    .lines()
                    .map(|line| serde_json::from_str(line).expect("parse next delta row"))
                    .collect();
            assert!(!rows.is_empty(), "{file_name} should contain next run rows");
            assert!(
                rows.iter()
                    .all(|row| row[run_id_key] == "snapshot-second"),
                "{file_name} must wholly include only the concurrent run in the next delta"
            );
        }
    }

    #[test]
    fn incremental_export_multi_run_batched_round_trips() {
        // Exercises the batched `WHERE run_id IN (…)` path in
        // export_table_{segments,events}_for_runs with MULTIPLE run_ids in one export
        // (the single-run incremental tests never hit the multi-param IN clause), and
        // verifies a full export→import round-trip so the cross-run grouping preserves
        // every segment/event.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multi.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        for i in 0..3 {
            let report = fixture_report(&format!("multi-{i}"), &db_path);
            store.persist_report(&report).expect("persist");
        }

        let manifest =
            export_incremental(&db_path, &export_dir, &state_root).expect("incremental export");
        assert_eq!(manifest.row_counts.runs, 3);
        assert_eq!(manifest.row_counts.segments, 6); // 3 runs × 2 segments
        assert_eq!(manifest.row_counts.events, 6); // 3 runs × 2 events

        let import_db = dir.path().join("import.sqlite3");
        let result = import_inner(&import_db, &export_dir, ConflictPolicy::Reject).expect("import");
        assert_eq!(result.runs_imported, 3);

        let imported = RunStore::open(&import_db).expect("open imported");
        for i in 0..3 {
            let details = imported
                .load_run_details(&format!("multi-{i}"))
                .expect("load")
                .expect("run should exist");
            assert_eq!(details.segments.len(), 2);
            assert_eq!(details.events.len(), 2);
        }
    }

    #[test]
    fn incremental_export_second_run_exports_only_new_records() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_second.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");

        // First run establishes a non-empty mutation cursor.
        let mut report1 = fixture_report("incr-old", &db_path);
        report1.finished_at_rfc3339 = "2026-01-01T00:00:05Z".to_owned();
        store.persist_report(&report1).expect("persist 1");

        // First incremental export (captures everything)
        let export_dir_1 = dir.path().join("export1");
        let manifest1 =
            export_incremental(&db_path, &export_dir_1, &state_root).expect("first export");
        assert_eq!(manifest1.row_counts.runs, 1);

        // A later database mutation must form the next delta.
        let mut report2 = fixture_report("incr-new", &db_path);
        report2.finished_at_rfc3339 = "2026-06-15T12:00:05Z".to_owned();
        store.persist_report(&report2).expect("persist 2");

        // Second incremental export (should only capture the new run)
        let export_dir_2 = dir.path().join("export2");
        let manifest2 =
            export_incremental(&db_path, &export_dir_2, &state_root).expect("second export");

        assert_eq!(manifest2.row_counts.runs, 1, "only the new run");
        assert_eq!(
            manifest2.row_counts.segments, 2,
            "segments for new run only"
        );
        assert_eq!(manifest2.row_counts.events, 2, "events for new run only");
        assert!(manifest2.cursor_used.is_some(), "should carry prior cursor");
        assert_eq!(
            manifest2.cursor_used.as_ref().unwrap().last_export_rfc3339,
            manifest1.cursor_after.last_export_rfc3339,
        );

        // Verify that the exported runs.jsonl contains only the new run
        let runs_content = fs::read_to_string(export_dir_2.join("runs.jsonl")).expect("read runs");
        let lines: Vec<&str> = runs_content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 1);
        let run_obj: serde_json::Value = serde_json::from_str(lines[0]).expect("parse run line");
        assert_eq!(run_obj["id"].as_str().unwrap(), "incr-new");
    }

    #[test]
    fn incremental_export_backdated_insert_after_cursor_exports_exactly_once() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_backdated_insert.sqlite3");
        let state_root = dir.path().join("state");
        let store = RunStore::open(&db_path).expect("store open");

        let mut anchor = fixture_report("cursor-anchor", &db_path);
        anchor.finished_at_rfc3339 = "2026-08-01T00:00:05Z".to_owned();
        store.persist_report(&anchor).expect("persist anchor");
        let first = export_incremental(&db_path, &dir.path().join("export1"), &state_root)
            .expect("first export");
        assert_eq!(first.row_counts.runs, 1);

        let mut backdated = fixture_report("backdated-after-cursor", &db_path);
        backdated.finished_at_rfc3339 = "2020-01-01T00:00:05Z".to_owned();
        store
            .persist_report(&backdated)
            .expect("persist backdated run");

        let second_export = dir.path().join("export2");
        let second = export_incremental(&db_path, &second_export, &state_root)
            .expect("export backdated delta");
        assert_eq!(second.row_counts.runs, 1);
        assert!(
            second.cursor_after.last_mutation_seq > first.cursor_after.last_mutation_seq,
            "database mutation sequence must advance independently of finished_at"
        );
        let row: serde_json::Value = serde_json::from_str(
            fs::read_to_string(second_export.join("runs.jsonl"))
                .expect("read backdated run")
                .trim(),
        )
        .expect("parse backdated run");
        assert_eq!(row["id"], "backdated-after-cursor");
        assert_eq!(row["finished_at"], "2020-01-01T00:00:05Z");

        let third = export_incremental(&db_path, &dir.path().join("export3"), &state_root)
            .expect("no-change export");
        assert_eq!(third.row_counts.runs, 0, "successful delta exports once");
        assert_eq!(
            third.cursor_after.last_mutation_seq,
            second.cursor_after.last_mutation_seq,
            "no-change export must retain the exact high-water mark"
        );
    }

    #[test]
    fn incremental_export_parent_update_after_cursor_exports_exactly_once() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_parent_update.sqlite3");
        let state_root = dir.path().join("state");
        let store = RunStore::open(&db_path).expect("store open");

        store
            .persist_report(&fixture_report("updated-parent", &db_path))
            .expect("persist run");
        export_incremental(&db_path, &dir.path().join("export1"), &state_root)
            .expect("first export");

        let writer = Connection::open(db_path.display().to_string()).expect("writer open");
        writer
            .execute(
                "UPDATE runs SET finished_at = '2020-01-01T00:00:05Z', \
                 transcript = 'revised transcript' WHERE id = 'updated-parent'",
            )
            .expect("update exported parent");

        let second_export = dir.path().join("export2");
        let second = export_incremental(&db_path, &second_export, &state_root)
            .expect("export parent update");
        assert_eq!(second.row_counts.runs, 1);
        let row: serde_json::Value = serde_json::from_str(
            fs::read_to_string(second_export.join("runs.jsonl"))
                .expect("read updated parent")
                .trim(),
        )
        .expect("parse updated parent");
        assert_eq!(row["id"], "updated-parent");
        assert_eq!(row["transcript"], "revised transcript");

        let third = export_incremental(&db_path, &dir.path().join("export3"), &state_root)
            .expect("no-change export");
        assert_eq!(third.row_counts.runs, 0, "parent update must not repeat");
    }

    #[test]
    fn incremental_export_child_only_update_exports_complete_aggregate_once() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_child_update.sqlite3");
        let state_root = dir.path().join("state");
        let store = RunStore::open(&db_path).expect("store open");

        store
            .persist_report(&fixture_report("updated-child", &db_path))
            .expect("persist run");
        export_incremental(&db_path, &dir.path().join("export1"), &state_root)
            .expect("first export");

        let writer = Connection::open(db_path.display().to_string()).expect("writer open");
        writer
            .execute(
                "UPDATE segments SET text = 'revised segment' \
                 WHERE run_id = 'updated-child' AND idx = 0",
            )
            .expect("update exported segment");
        writer
            .execute(
                "UPDATE events SET message = 'revised event' \
                 WHERE run_id = 'updated-child' AND seq = 1",
            )
            .expect("update exported event");

        let second_export = dir.path().join("export2");
        let second = export_incremental(&db_path, &second_export, &state_root)
            .expect("export child update");
        assert_eq!(second.row_counts.runs, 1);
        assert_eq!(second.row_counts.segments, 2);
        assert_eq!(second.row_counts.events, 2);

        let segments: Vec<serde_json::Value> =
            fs::read_to_string(second_export.join("segments.jsonl"))
                .expect("read segments")
                .lines()
                .map(|line| serde_json::from_str(line).expect("parse segment"))
                .collect();
        assert!(segments.iter().any(|row| row["text"] == "revised segment"));
        let events: Vec<serde_json::Value> = fs::read_to_string(second_export.join("events.jsonl"))
            .expect("read events")
            .lines()
            .map(|line| serde_json::from_str(line).expect("parse event"))
            .collect();
        assert!(events.iter().any(|row| row["message"] == "revised event"));

        let third = export_incremental(&db_path, &dir.path().join("export3"), &state_root)
            .expect("no-change export");
        assert_eq!(third.row_counts.runs, 0, "child update must not repeat");
        assert_eq!(third.row_counts.segments, 0);
        assert_eq!(third.row_counts.events, 0);

        writer
            .execute(
                "DELETE FROM segments WHERE run_id = 'updated-child' AND idx = 1",
            )
            .expect("delete exported segment");
        let fourth = export_incremental(
            &db_path,
            &dir.path().join("export4"),
            &state_root,
        )
        .expect("export child deletion");
        assert_eq!(fourth.row_counts.runs, 1);
        assert_eq!(
            fourth.row_counts.segments, 1,
            "child deletion must export the complete surviving child set"
        );
        assert_eq!(fourth.row_counts.events, 2);
    }

    #[test]
    fn incremental_export_no_new_records_produces_zero_counts() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_noop.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("incr-only", &db_path);
        report.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        store.persist_report(&report).expect("persist");

        // First export: captures everything
        let export_dir_1 = dir.path().join("export1");
        export_incremental(&db_path, &export_dir_1, &state_root).expect("first export");

        // Second export: nothing new
        let export_dir_2 = dir.path().join("export2");
        let manifest =
            export_incremental(&db_path, &export_dir_2, &state_root).expect("second export");

        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);
    }

    #[test]
    fn incremental_export_rejects_cursor_ahead_of_database_high_water() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_cursor_ahead.sqlite3");
        let state_root = dir.path().join("state");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("cursor-ahead", &db_path))
            .expect("persist run");

        let connection = Connection::open(db_path.display().to_string()).expect("connection");
        let forged_sequence = mutation_high_water(&connection) + 1;
        let cursor_path = state_root.join(CURSOR_FILENAME);
        save_cursor(
            &cursor_path,
            &test_cursor(
                &connection,
                forged_sequence,
                "2099-01-01T00:00:00Z",
                Some("unrelated-database"),
            ),
        )
        .expect("write mismatched cursor");

        let error = export_incremental(
            &db_path,
            &dir.path().join("export"),
            &state_root,
        )
        .expect_err("cursor from a newer or different database must fail closed");
        assert!(
            error.to_string().contains("outside database range"),
            "unexpected cursor mismatch error: {error}"
        );
        let retained = load_cursor(&cursor_path)
            .expect("reload cursor")
            .expect("cursor retained");
        assert_eq!(retained.last_mutation_seq, forged_sequence);
    }

    #[test]
    fn incremental_export_rejects_in_range_cursor_from_different_database() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("cursor_source.sqlite3");
        let target_db = dir.path().join("cursor_target.sqlite3");
        let source_state = dir.path().join("source_state");
        let target_state = dir.path().join("target_state");

        let source = RunStore::open(&source_db).expect("open cursor source");
        source
            .persist_report(&fixture_report("source-run", &source_db))
            .expect("persist source run");
        let source_manifest = export_incremental(
            &source_db,
            &dir.path().join("source_export"),
            &source_state,
        )
        .expect("export source cursor");

        let target = RunStore::open(&target_db).expect("open cursor target");
        for run_id in ["target-run-a", "target-run-b"] {
            target
                .persist_report(&fixture_report(run_id, &target_db))
                .expect("persist target run");
        }
        let target_connection =
            Connection::open(target_db.display().to_string()).expect("open target connection");
        let target_authority =
            load_mutation_authority(&target_connection).expect("load target authority");
        assert!(
            source_manifest.cursor_after.last_mutation_seq <= target_authority.high_water_mark,
            "the foreign cursor must be numerically in range to exercise the identity guard"
        );
        assert_ne!(
            source_manifest.cursor_after.database_id.as_deref(),
            Some(target_authority.database_id.as_str()),
            "independent databases must have distinct identities"
        );

        let cursor_path = target_state.join(CURSOR_FILENAME);
        save_cursor(&cursor_path, &source_manifest.cursor_after).expect("install foreign cursor");
        let target_export = dir.path().join("target_export");
        let error = export_incremental(&target_db, &target_export, &target_state)
            .expect_err("an in-range foreign cursor must fail closed");
        assert!(
            error.to_string().contains("database identity mismatch"),
            "unexpected foreign cursor error: {error}"
        );
        assert!(
            !target_export.join("manifest.json").exists(),
            "identity mismatch must be rejected before snapshot publication"
        );
        assert!(
            !target_export.join("runs.jsonl").exists(),
            "identity mismatch must not publish a partial runs delta"
        );
        let retained = load_cursor(&cursor_path)
            .expect("reload foreign cursor")
            .expect("foreign cursor retained");
        assert_eq!(retained.database_id, source_manifest.cursor_after.database_id);
    }

    #[test]
    fn incremental_export_empty_db_produces_zero_counts_and_creates_cursor() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_empty.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        let manifest = export_incremental(&db_path, &export_dir, &state_root).expect("export");

        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.row_counts.segments, 0);
        assert_eq!(manifest.row_counts.events, 0);
        assert!(manifest.cursor_used.is_none());
        assert_eq!(manifest.cursor_after.last_mutation_seq, 0);
        assert_eq!(
            manifest.cursor_after.last_export_rfc3339,
            EMPTY_INCREMENTAL_CURSOR_TS
        );
        assert!(manifest.cursor_after.last_export_run_id.is_none());
        // Cursor file should exist after export.
        assert!(state_root.join(CURSOR_FILENAME).exists());
    }

    #[test]
    fn incremental_export_empty_db_does_not_skip_later_backfilled_run() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_empty_then_backfill.sqlite3");
        let state_root = dir.path().join("state");
        let export_dir_1 = dir.path().join("export1");
        let export_dir_2 = dir.path().join("export2");

        let store = RunStore::open(&db_path).expect("store open");

        let first_manifest =
            export_incremental(&db_path, &export_dir_1, &state_root).expect("first export");
        assert_eq!(first_manifest.row_counts.runs, 0);
        assert_eq!(
            first_manifest.cursor_after.last_export_rfc3339,
            EMPTY_INCREMENTAL_CURSOR_TS
        );

        let mut report = fixture_report("backfilled-run", &db_path);
        report.finished_at_rfc3339 = "2024-01-01T00:00:05Z".to_owned();
        store
            .persist_report(&report)
            .expect("persist backfilled run");

        let second_manifest =
            export_incremental(&db_path, &export_dir_2, &state_root).expect("second export");
        assert_eq!(
            second_manifest.row_counts.runs, 1,
            "a later run with an older finished_at must still be exported after an empty initial export"
        );

        let runs = fs::read_to_string(export_dir_2.join("runs.jsonl")).expect("read runs");
        let lines: Vec<&str> = runs
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 1);
        let run_obj: serde_json::Value = serde_json::from_str(lines[0]).expect("parse run");
        assert_eq!(run_obj["id"].as_str().unwrap(), "backfilled-run");
    }

    #[test]
    fn incremental_export_does_not_advance_cursor_when_manifest_publish_fails() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_manifest_fail.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("incr-manifest-fail", &db_path);
        report.finished_at_rfc3339 = "2026-07-01T12:00:05Z".to_owned();
        store.persist_report(&report).expect("persist");

        fs::create_dir_all(&export_dir).expect("create export dir");
        fs::create_dir(export_dir.join("manifest.json")).expect("block manifest path with dir");

        let error =
            export_incremental(&db_path, &export_dir, &state_root).expect_err("export should fail");
        assert!(
            error.to_string().contains("Is a directory")
                || error.to_string().contains("is a directory"),
            "expected manifest publication failure, got: {error}"
        );
        assert!(
            !state_root.join(CURSOR_FILENAME).exists(),
            "cursor must not advance when manifest publication fails"
        );
    }

    #[test]
    fn incremental_export_cursor_persists_between_calls() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_cursor.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");

        // Insert three runs before the first cursor is established.
        for (i, ts) in [
            "2026-01-01T00:00:00Z",
            "2026-02-01T00:00:00Z",
            "2026-03-01T00:00:00Z",
        ]
        .iter()
        .enumerate()
        {
            let mut report = fixture_report(&format!("cursor-{i}"), &db_path);
            report.started_at_rfc3339 = ts.to_string();
            store.persist_report(&report).expect("persist");
        }

        // First export: all 3 runs.
        let export_dir_1 = dir.path().join("export1");
        let m1 = export_incremental(&db_path, &export_dir_1, &state_root).expect("export 1");
        assert_eq!(m1.row_counts.runs, 3);
        drop(store);

        // Reopen both the database and the on-disk cursor to exercise restart.
        let store = RunStore::open(&db_path).expect("reopen store");

        // Second export: no new runs.
        let export_dir_2 = dir.path().join("export2");
        let m2 = export_incremental(&db_path, &export_dir_2, &state_root).expect("export 2");
        assert_eq!(m2.row_counts.runs, 0);
        assert_eq!(
            m2.cursor_used.as_ref().unwrap().last_mutation_seq,
            m1.cursor_after.last_mutation_seq,
        );
        assert_eq!(
            m2.cursor_used.as_ref().unwrap().last_export_rfc3339,
            m1.cursor_after.last_export_rfc3339,
        );

        // Insert a fourth run.
        let mut report4 = fixture_report("cursor-3", &db_path);
        report4.started_at_rfc3339 = "2026-04-01T00:00:00Z".to_owned();
        store.persist_report(&report4).expect("persist 4");

        // Third export: only the new run.
        let export_dir_3 = dir.path().join("export3");
        let m3 = export_incremental(&db_path, &export_dir_3, &state_root).expect("export 3");
        assert_eq!(m3.row_counts.runs, 1);
        assert_eq!(m3.row_counts.segments, 2);
        assert_eq!(m3.row_counts.events, 2);
    }

    #[test]
    fn incremental_export_result_can_be_imported() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_import.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("incr-importable", &db_path);
        report.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        store.persist_report(&report).expect("persist");

        let export_dir = dir.path().join("export");
        let incr_manifest =
            export_incremental(&db_path, &export_dir, &state_root).expect("incremental export");

        // Write a standard SyncManifest so import() can consume it.
        let sync_manifest = SyncManifest {
            schema_version: incr_manifest.schema_version.clone(),
            export_format_version: incr_manifest.export_format_version.clone(),
            created_at_rfc3339: incr_manifest.created_at_rfc3339.clone(),
            source_db_path: incr_manifest.source_db_path.clone(),
            row_counts: incr_manifest.row_counts.clone(),
            checksums: incr_manifest.checksums.clone(),
        };
        let manifest_path = export_dir.join("manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&sync_manifest).expect("ser"),
        )
        .expect("write manifest");

        let target_db = dir.path().join("target.sqlite3");
        let result = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import incremental output");
        assert_eq!(result.runs_imported, 1);
        assert_eq!(result.segments_imported, 2);
        assert_eq!(result.events_imported, 2);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn incremental_export_manifest_has_valid_checksums() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_checksum.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("cksum-1", &db_path))
            .expect("persist");

        let manifest = export_incremental(&db_path, &export_dir, &state_root).expect("export");

        // Re-compute checksums and compare.
        let actual_runs = sha256_file(&export_dir.join("runs.jsonl")).expect("hash");
        let actual_segments = sha256_file(&export_dir.join("segments.jsonl")).expect("hash");
        let actual_events = sha256_file(&export_dir.join("events.jsonl")).expect("hash");

        assert_eq!(manifest.checksums.runs_jsonl_sha256, actual_runs);
        assert_eq!(manifest.checksums.segments_jsonl_sha256, actual_segments);
        assert_eq!(manifest.checksums.events_jsonl_sha256, actual_events);
    }

    #[test]
    fn sync_cursor_serde_round_trip() {
        let cursor = SyncCursor {
            last_mutation_seq: 42,
            database_id: Some("00000000-0000-0000-0000-000000000042".to_owned()),
            last_export_rfc3339: "2026-06-15T12:00:00Z".to_owned(),
            last_export_run_id: Some("run-42".to_owned()),
            last_run_count: 42,
        };
        let json = serde_json::to_string_pretty(&cursor).expect("serialize");
        let parsed: SyncCursor = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.last_export_rfc3339, "2026-06-15T12:00:00Z");
        assert_eq!(parsed.last_mutation_seq, 42);
        assert_eq!(parsed.database_id, cursor.database_id);
        assert_eq!(parsed.last_export_run_id.as_deref(), Some("run-42"));
        assert_eq!(parsed.last_run_count, 42);
    }

    #[test]
    fn sync_cursor_deserializes_legacy_shape_without_run_id() {
        let legacy = r#"{"last_export_rfc3339":"2026-06-15T12:00:00Z","last_run_count":42}"#;
        let parsed: SyncCursor = serde_json::from_str(legacy).expect("deserialize legacy cursor");
        assert_eq!(parsed.last_mutation_seq, 0);
        assert!(parsed.database_id.is_none());
        assert_eq!(parsed.last_export_rfc3339, "2026-06-15T12:00:00Z");
        assert!(parsed.last_export_run_id.is_none());
        assert_eq!(parsed.last_run_count, 42);
    }

    #[test]
    fn incremental_export_manifest_serde_round_trip() {
        let manifest = IncrementalExportManifest {
            schema_version: "1.1".to_owned(),
            export_format_version: "1.0".to_owned(),
            export_mode: "incremental".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 1,
                segments: 2,
                events: 3,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: "aaa".to_owned(),
                segments_jsonl_sha256: "bbb".to_owned(),
                events_jsonl_sha256: "ccc".to_owned(),
            },
            cursor_used: Some(SyncCursor {
                last_mutation_seq: 5,
                database_id: Some("00000000-0000-0000-0000-000000000005".to_owned()),
                last_export_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                last_export_run_id: Some("run-5".to_owned()),
                last_run_count: 5,
            }),
            cursor_after: SyncCursor {
                last_mutation_seq: 6,
                database_id: Some("00000000-0000-0000-0000-000000000005".to_owned()),
                last_export_rfc3339: "2026-06-15T12:00:00Z".to_owned(),
                last_export_run_id: Some("run-6".to_owned()),
                last_run_count: 1,
            },
        };

        let json = serde_json::to_string_pretty(&manifest).expect("serialize");
        let parsed: IncrementalExportManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.export_mode, "incremental");
        assert_eq!(parsed.row_counts.runs, 1);
        assert!(parsed.cursor_used.is_some());
        assert_eq!(parsed.cursor_after.last_run_count, 1);
    }

    #[test]
    fn load_cursor_returns_none_when_file_missing() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("nonexistent_cursor.json");
        let result = load_cursor(&path).expect("should succeed with None");
        assert!(result.is_none());
    }

    #[test]
    fn save_and_load_cursor_round_trip() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("cursor_rt.json");
        let cursor = SyncCursor {
            last_mutation_seq: 7,
            database_id: Some("00000000-0000-0000-0000-000000000007".to_owned()),
            last_export_rfc3339: "2026-03-15T08:30:00Z".to_owned(),
            last_export_run_id: Some("run-7".to_owned()),
            last_run_count: 7,
        };
        save_cursor(&path, &cursor).expect("save");
        let loaded = load_cursor(&path).expect("load").expect("should exist");
        assert_eq!(loaded.last_export_rfc3339, "2026-03-15T08:30:00Z");
        assert_eq!(loaded.last_mutation_seq, 7);
        assert_eq!(loaded.database_id, cursor.database_id);
        assert_eq!(loaded.last_export_run_id.as_deref(), Some("run-7"));
        assert_eq!(loaded.last_run_count, 7);
    }

    #[test]
    fn load_cursor_rejects_corrupt_json() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("corrupt_cursor.json");
        fs::write(&path, "{not valid json}").expect("write");
        let result = load_cursor(&path);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid sync cursor")
        );
    }

    #[test]
    fn load_cursor_rejects_negative_mutation_sequence() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("negative_cursor.json");
        fs::write(
            &path,
            r#"{"last_mutation_seq":-1,"last_export_rfc3339":"","last_run_count":0}"#,
        )
        .expect("write negative cursor");
        let error = load_cursor(&path).expect_err("negative cursor must fail closed");
        assert!(
            error.to_string().contains("must be non-negative"),
            "unexpected negative cursor error: {error}"
        );
    }

    #[test]
    fn incremental_export_multiple_runs_same_timestamp_exports_all() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_same_ts.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");

        // Insert two runs with the same timestamp.
        for i in 0..2 {
            let mut report = fixture_report(&format!("same-ts-{i}"), &db_path);
            report.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
            store.persist_report(&report).expect("persist");
        }

        // First export: both runs.
        let export_dir = dir.path().join("export1");
        let m1 = export_incremental(&db_path, &export_dir, &state_root).expect("export");
        assert_eq!(m1.row_counts.runs, 2);

        // Second export: no new mutations despite the shared domain timestamp.
        let export_dir_2 = dir.path().join("export2");
        let m2 = export_incremental(&db_path, &export_dir_2, &state_root).expect("export 2");
        assert_eq!(m2.row_counts.runs, 0);
    }

    #[test]
    fn incremental_export_same_timestamp_new_run_after_cursor_is_exported() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_same_ts_new.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");

        let mut first = fixture_report("same-ts-z", &db_path);
        first.finished_at_rfc3339 = "2026-01-01T00:00:05Z".to_owned();
        store.persist_report(&first).expect("persist first");

        let export_dir_1 = dir.path().join("export1");
        let first_manifest =
            export_incremental(&db_path, &export_dir_1, &state_root).expect("export 1");
        assert_eq!(first_manifest.row_counts.runs, 1);

        let mut second = fixture_report("same-ts-a", &db_path);
        second.finished_at_rfc3339 = "2026-01-01T00:00:05Z".to_owned();
        store.persist_report(&second).expect("persist second");

        let export_dir_2 = dir.path().join("export2");
        let second_manifest =
            export_incremental(&db_path, &export_dir_2, &state_root).expect("export 2");

        assert_eq!(
            second_manifest.row_counts.runs, 1,
            "new run with identical finished_at should still be exported"
        );
        let runs = fs::read_to_string(export_dir_2.join("runs.jsonl")).expect("read runs");
        let lines: Vec<&str> = runs
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 1);
        let row: serde_json::Value = serde_json::from_str(lines[0]).expect("parse");
        assert_eq!(
            row["id"].as_str(),
            Some("same-ts-a"),
            "a same-clock mutation must not depend on lexicographic ID order"
        );
    }

    #[test]
    fn export_mode_enum_variants() {
        // Verify both variants are distinct.
        assert_ne!(ExportMode::Full, ExportMode::Incremental);
        let _ = format!("{:?}", ExportMode::Full);
        let _ = format!("{:?}", ExportMode::Incremental);
    }

    #[test]
    fn max_started_at_empty_db_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        // Access the raw connection via the store's public API indirectly:
        // Open a separate connection for this test.
        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        // No runs -> no export position.
        let result = max_export_position(&conn, None).expect("should succeed");
        assert_eq!(result, None, "empty db should return None");
        drop(store);
    }

    #[test]
    fn max_export_position_at_mutation_high_water_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ts_filter.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        let report = fixture_report("ts-run", &db_path);
        store.persist_report(&report).expect("persist");

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        // A cursor at the database high-water mark has no matching mutations.
        let far_future = test_cursor(
            &conn,
            mutation_high_water(&conn),
            "2099-12-31T23:59:59Z",
            Some("zzzzzz"),
        );
        let result = max_export_position(&conn, Some(&far_future)).expect("should succeed");
        assert_eq!(result, None, "no runs after the mutation high-water mark");

        // With no filter, should return the run's finished_at + id tuple.
        let result = max_export_position(&conn, None).expect("should succeed");
        assert!(result.is_some(), "should find a run with no filter");
    }

    #[test]
    fn ensure_run_reference_exists_missing_run_returns_formatted_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ref_check.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        drop(store);

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");
        ensure_schema(&conn).expect("schema");

        let mut known_run_ids = HashSet::new();
        let err = ensure_run_reference_exists(
            &conn,
            "ghost-run-id",
            "segments",
            "ghost-run-id/0",
            &mut known_run_ids,
        )
        .expect_err("should fail for missing run");
        let msg = err.to_string();
        assert!(
            msg.contains("referential integrity violation"),
            "should mention referential integrity: {msg}"
        );
        assert!(msg.contains("ghost-run-id"), "should mention run_id: {msg}");
        assert!(msg.contains("segments"), "should mention table name: {msg}");
    }

    #[test]
    fn export_table_segments_for_runs_with_empty_run_ids_produces_empty_file() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_ids.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        // Persist a run with segments — but we won't export it (empty run_ids).
        let report = fixture_report("has-segments", &db_path);
        store.persist_report(&report).expect("persist");

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");
        let output_path = dir.path().join("segments.jsonl");
        let (count, _sha256) =
            export_table_segments_for_runs(&conn, &output_path, &[]).expect("should succeed");
        assert_eq!(count, 0, "empty run_ids should produce 0 rows");

        let content = fs::read_to_string(&output_path).expect("read");
        assert!(
            content.is_empty(),
            "output file should be empty but got: {content}"
        );
    }

    #[test]
    fn save_cursor_creates_nested_parent_directories() {
        let dir = tempdir().expect("tempdir");
        let nested = dir.path().join("a").join("b").join("c").join("cursor.json");
        assert!(
            !nested.parent().unwrap().exists(),
            "dirs should not exist yet"
        );

        let cursor = SyncCursor {
            last_mutation_seq: 7,
            database_id: Some("00000000-0000-0000-0000-000000000007".to_owned()),
            last_export_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            last_export_run_id: Some("run-a".to_owned()),
            last_run_count: 7,
        };
        save_cursor(&nested, &cursor).expect("save should create dirs");

        assert!(nested.exists(), "cursor file should exist");
        let loaded = load_cursor(&nested)
            .expect("load")
            .expect("should have content");
        assert_eq!(loaded.last_export_rfc3339, "2026-01-01T00:00:00Z");
        assert_eq!(loaded.last_export_run_id.as_deref(), Some("run-a"));
        assert_eq!(loaded.last_run_count, 7);
    }

    // ── bd-246.2: Validation tests ──

    #[test]
    fn validate_sync_matching_db_and_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validate.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        let report = fixture_report("val-1", &db_path);
        store.persist_report(&report).expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(
            validation.is_valid,
            "matching DB and JSONL should be valid: {validation:?}"
        );
        assert_eq!(validation.db_run_count, 1);
        assert_eq!(validation.jsonl_run_count, 1);
        assert_eq!(validation.db_segment_count, 2);
        assert_eq!(validation.jsonl_segment_count, 2);
        assert_eq!(validation.db_event_count, 2);
        assert_eq!(validation.jsonl_event_count, 2);
        assert!(validation.missing_from_db.is_empty());
        assert!(validation.missing_from_jsonl.is_empty());
        assert!(validation.mismatched_records.is_empty());
    }

    #[test]
    fn validate_sync_multiple_runs_matching() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_multi.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        for i in 0..3 {
            store
                .persist_report(&fixture_report(&format!("val-multi-{i}"), &db_path))
                .expect("persist");
        }
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        export(&db_path, &export_dir, &state_root).expect("export");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(validation.is_valid, "should be valid: {validation:?}");
        assert_eq!(validation.db_run_count, 3);
        assert_eq!(validation.jsonl_run_count, 3);
        assert_eq!(validation.db_segment_count, 6);
        assert_eq!(validation.jsonl_segment_count, 6);
        assert_eq!(validation.db_event_count, 6);
        assert_eq!(validation.jsonl_event_count, 6);
    }

    #[test]
    fn validate_sync_detects_missing_runs_from_jsonl() {
        // DB has a run that is not in the JSONL export.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_missing_jsonl.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-a", &db_path))
            .expect("persist");

        // Export with only run-a
        export(&db_path, &export_dir, &state_root).expect("export");

        // Add another run to DB after export
        store
            .persist_report(&fixture_report("run-b", &db_path))
            .expect("persist");
        drop(store);
        let conn = Connection::open(db_path.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should not be valid");
        assert_eq!(validation.db_run_count, 2);
        assert_eq!(validation.jsonl_run_count, 1);
        assert!(
            validation.missing_from_jsonl.contains(&"run-b".to_owned()),
            "run-b should be missing from JSONL: {:?}",
            validation.missing_from_jsonl
        );
    }

    #[test]
    fn validate_sync_detects_missing_runs_from_db() {
        // JSONL has a run that is not in the DB (e.g. DB was wiped).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_missing_db.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-x", &db_path))
            .expect("persist run-x");
        store
            .persist_report(&fixture_report("run-y", &db_path))
            .expect("persist run-y");

        // Export with both runs
        export(&db_path, &export_dir, &state_root).expect("export");
        drop(store);

        // Create a fresh DB with only run-x
        let fresh_db = dir.path().join("fresh.sqlite3");
        let fresh_store = RunStore::open(&fresh_db).expect("fresh store");
        fresh_store
            .persist_report(&fixture_report("run-x", &fresh_db))
            .expect("persist run-x");
        drop(fresh_store);
        let conn = Connection::open(fresh_db.display().to_string()).expect("open");
        ensure_schema(&conn).expect("schema");
        drop(conn);

        let validation = validate_sync(&fresh_db, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should not be valid");
        assert!(
            validation.missing_from_db.contains(&"run-y".to_owned()),
            "run-y should be missing from DB: {:?}",
            validation.missing_from_db
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_mismatch.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("mismatch-1", &db_path))
            .expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate the transcript in the runs.jsonl
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["transcript"] = json!("DIFFERENT TRANSCRIPT");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("serialize")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should not be valid");
        assert!(
            validation
                .mismatched_records
                .contains(&"mismatch-1".to_owned()),
            "mismatch-1 should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_detects_same_count_child_payload_mutations() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validate-child-content.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("child-content", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");
        assert!(
            validate_sync(&db_path, &export_dir)
                .expect("validate untouched export")
                .is_valid
        );

        let segments_path = export_dir.join("segments.jsonl");
        let mut segments = fs::read_to_string(&segments_path)
            .expect("read segments")
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("segment row"))
            .collect::<Vec<_>>();
        let segment_count = segments.len();
        segments[0]["text"] = json!("same-count segment mutation");
        fs::write(
            &segments_path,
            segments
                .iter()
                .map(|row| serde_json::to_string(row).expect("serialize segment"))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .expect("write segments");

        let events_path = export_dir.join("events.jsonl");
        let mut events = fs::read_to_string(&events_path)
            .expect("read events")
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("event row"))
            .collect::<Vec<_>>();
        let event_count = events.len();
        events[0]["message"] = json!("same-count event mutation");
        fs::write(
            &events_path,
            events
                .iter()
                .map(|row| serde_json::to_string(row).expect("serialize event"))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .expect("write events");

        let validation = validate_sync(&db_path, &export_dir).expect("validate mutations");
        assert!(!validation.is_valid);
        assert_eq!(validation.db_segment_count, segment_count as u64);
        assert_eq!(validation.jsonl_segment_count, segment_count as u64);
        assert_eq!(validation.db_event_count, event_count as u64);
        assert_eq!(validation.jsonl_event_count, event_count as u64);
        assert!(validation.mismatched_records.is_empty());
        assert!(validation.child_mismatches.iter().any(|mismatch| {
            mismatch.table == "segments"
                && mismatch.key == "child-content/0"
                && mismatch.reason == "payload differs"
        }));
        assert!(validation.child_mismatches.iter().any(|mismatch| {
            mismatch.table == "events"
                && mismatch.key == "child-content/1"
                && mismatch.reason == "payload differs"
        }));
        assert!(!validation.child_mismatches_truncated);
    }

    #[test]
    fn validate_sync_distinguishes_null_speaker_from_empty_text() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validate-child-null.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("child-null", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let segments_path = export_dir.join("segments.jsonl");
        let mut rows = fs::read_to_string(&segments_path)
            .expect("read segments")
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("segment row"))
            .collect::<Vec<_>>();
        assert_eq!(rows[1]["speaker"], serde_json::Value::Null);
        rows[1]["speaker"] = json!("");
        fs::write(
            &segments_path,
            rows.iter()
                .map(|row| serde_json::to_string(row).expect("serialize segment"))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .expect("write segments");

        let validation = validate_sync(&db_path, &export_dir).expect("validate null mutation");
        assert!(!validation.is_valid);
        assert!(validation.child_mismatches.iter().any(|mismatch| {
            mismatch.table == "segments"
                && mismatch.key == "child-null/1"
                && mismatch.reason == "payload differs"
        }));
    }

    #[test]
    fn validate_sync_distinguishes_null_confidence_from_zero() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validate-child-null-confidence.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("child-null-confidence", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let segments_path = export_dir.join("segments.jsonl");
        let mut rows = fs::read_to_string(&segments_path)
            .expect("read segments")
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("segment row"))
            .collect::<Vec<_>>();
        assert_eq!(rows[1]["confidence"], serde_json::Value::Null);
        rows[1]["confidence"] = json!(0.0);
        fs::write(
            &segments_path,
            rows.iter()
                .map(|row| serde_json::to_string(row).expect("serialize segment"))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .expect("write segments");

        let validation = validate_sync(&db_path, &export_dir).expect("validate zero mutation");
        assert!(!validation.is_valid);
        assert!(validation.child_mismatches.iter().any(|mismatch| {
            mismatch.table == "segments"
                && mismatch.key == "child-null-confidence/1"
                && mismatch.reason == "payload differs"
        }));
    }

    #[test]
    fn validate_sync_rejects_malformed_and_duplicate_child_rows() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validate-child-shape.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("child-shape", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let segments_path = export_dir.join("segments.jsonl");
        let original = fs::read_to_string(&segments_path).expect("read segments");
        let mut rows = original
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("segment row"))
            .collect::<Vec<_>>();
        rows[0]
            .as_object_mut()
            .expect("segment object")
            .remove("text");
        fs::write(
            &segments_path,
            rows.iter()
                .map(|row| serde_json::to_string(row).expect("serialize segment"))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n",
        )
        .expect("write malformed segments");
        let malformed = validate_sync(&db_path, &export_dir)
            .expect_err("missing child payload field must fail closed");
        assert!(malformed.to_string().contains("text"));

        let original_rows = original.lines().collect::<Vec<_>>();
        assert!(original_rows.len() >= 2, "fixture must have two segments");
        fs::write(
            &segments_path,
            format!("{}\n{}\n", original_rows[0], original_rows[0]),
        )
        .expect("write duplicate segments");
        let duplicate = validate_sync(&db_path, &export_dir)
            .expect_err("duplicate child key must fail closed");
        let duplicate_message = duplicate.to_string();
        assert!(duplicate_message.contains("duplicate composite key"));
        assert!(duplicate_message.contains("child-shape/0"));

        fs::write(&segments_path, &original).expect("restore segments");
        let events_path = export_dir.join("events.jsonl");
        let events = fs::read_to_string(&events_path).expect("read events");
        let first_event = events.lines().next().expect("fixture has event");
        fs::write(&events_path, format!("{first_event}\n{first_event}\n"))
            .expect("write duplicate events");
        let duplicate_event = validate_sync(&db_path, &export_dir)
            .expect_err("duplicate event key must fail closed");
        let duplicate_event_message = duplicate_event.to_string();
        assert!(duplicate_event_message.contains("duplicate composite key"));
        assert!(duplicate_event_message.contains("child-shape/1"));
    }

    #[test]
    fn validate_sync_ignores_child_row_order() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validate-child-order.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("child-order", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        for filename in ["segments.jsonl", "events.jsonl"] {
            let path = export_dir.join(filename);
            let content = fs::read_to_string(&path).expect("read child rows");
            let mut lines = content.lines().collect::<Vec<_>>();
            lines.reverse();
            fs::write(&path, lines.join("\n") + "\n").expect("reverse child rows");
        }

        let validation = validate_sync(&db_path, &export_dir).expect("validate reordered rows");
        assert!(validation.is_valid, "row order is not part of sync identity");
    }

    #[test]
    fn validate_sync_reads_one_database_snapshot_during_concurrent_commit() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validate-snapshot.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("snapshot-run", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");
        let writer = Connection::open(db_path.display().to_string()).expect("writer open");

        let during_commit = validate_sync_with_after_run_ids(&db_path, &export_dir, || {
            writer
                .execute(
                    "UPDATE segments SET text = 'committed after snapshot' \
                     WHERE run_id = 'snapshot-run' AND idx = 0",
                )
                .expect("concurrent WAL commit");
            Ok(())
        })
        .expect("validate pinned snapshot");
        assert!(
            during_commit.is_valid,
            "validation must compare the pre-commit snapshot consistently"
        );

        let after_commit = validate_sync(&db_path, &export_dir).expect("validate committed state");
        assert!(
            !after_commit.is_valid,
            "a later validation must observe the committed child mutation"
        );
        assert!(
            after_commit
                .child_mismatches
                .iter()
                .any(|mismatch| mismatch.table == "segments")
        );
    }

    #[test]
    fn child_mismatch_diagnostics_are_bounded_without_hiding_invalidity() {
        let db_rows = (0..=MAX_SYNC_VALIDATION_CHILD_MISMATCHES)
            .map(|idx| {
                (
                    (format!("run-{idx:03}"), idx as i64),
                    EventValidationRow {
                        ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                        stage: "test".to_owned(),
                        code: "test.event".to_owned(),
                        message: "message".to_owned(),
                        payload_json: "{}".to_owned(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let jsonl_rows = HashMap::new();
        let mut diagnostics = Vec::new();
        let mut truncated = false;

        collect_child_mismatches(
            &db_rows,
            &jsonl_rows,
            "events",
            &mut diagnostics,
            &mut truncated,
        );

        assert_eq!(diagnostics.len(), MAX_SYNC_VALIDATION_CHILD_MISMATCHES);
        assert!(truncated);
    }

    #[test]
    fn validate_sync_empty_db_and_empty_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_empty.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        export(&db_path, &export_dir, &state_root).expect("export");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(validation.is_valid, "empty DB and JSONL should be valid");
        assert_eq!(validation.db_run_count, 0);
        assert_eq!(validation.jsonl_run_count, 0);
        assert_eq!(validation.db_segment_count, 0);
        assert_eq!(validation.jsonl_segment_count, 0);
        assert_eq!(validation.db_event_count, 0);
        assert_eq!(validation.jsonl_event_count, 0);
    }

    #[test]
    fn validate_sync_requires_all_canonical_jsonl_inputs() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("validation-inputs.sqlite3");
        let _store = RunStore::open(&db_path).expect("store open");

        for missing_stem in ["runs", "segments", "events"] {
            let export_dir = dir.path().join(format!("missing-{missing_stem}"));
            fs::create_dir_all(&export_dir).expect("create incomplete export");
            for present_stem in ["runs", "segments", "events"] {
                if present_stem != missing_stem {
                    fs::write(export_dir.join(format!("{present_stem}.jsonl")), "")
                        .expect("write empty canonical input");
                }
            }

            let error = validate_sync(&db_path, &export_dir)
                .expect_err("incomplete validation input must fail closed");
            let message = error.to_string();
            assert!(
                message.contains("missing sync validation input"),
                "unexpected missing-input error: {message}"
            );
            assert!(
                message.contains(&format!("{missing_stem}.jsonl")),
                "missing-input error omitted logical file: {message}"
            );
        }
    }

    #[test]
    fn validate_sync_rejects_same_and_different_duplicate_run_rows() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("duplicate-runs.sqlite3");
        let _store = RunStore::open(&db_path).expect("store open");
        let first = json!({"id": "duplicate-run", "marker": "first"});

        for (case, second) in [
            ("same", first.clone()),
            (
                "different",
                json!({"id": "duplicate-run", "marker": "second"}),
            ),
        ] {
            let export_dir = dir.path().join(format!("duplicate-{case}"));
            write_jsonl_snapshot(&export_dir, &[first.clone(), second], &[], &[]);

            let error = validate_sync(&db_path, &export_dir)
                .expect_err("duplicate run IDs must fail before comparison");
            let message = error.to_string();
            assert!(
                message.contains("duplicate run id")
                    && message.contains("duplicate-run")
                    && message.contains("line 2"),
                "unexpected duplicate-run error: {message}"
            );
        }
    }

    #[test]
    fn validate_sync_report_serde_round_trip() {
        let report = SyncValidationReport {
            db_run_count: 5,
            jsonl_run_count: 4,
            db_segment_count: 10,
            jsonl_segment_count: 8,
            db_event_count: 15,
            jsonl_event_count: 12,
            missing_from_jsonl: vec!["run-5".to_owned()],
            missing_from_db: vec![],
            mismatched_records: vec!["run-2".to_owned()],
            child_mismatches: vec![SyncConflict {
                table: "segments".to_owned(),
                key: "run-2/0".to_owned(),
                reason: "payload differs".to_owned(),
            }],
            child_mismatches_truncated: false,
            is_valid: false,
        };
        let json_text = serde_json::to_string_pretty(&report).expect("serialize");
        let deserialized: SyncValidationReport =
            serde_json::from_str(&json_text).expect("deserialize");
        assert_eq!(deserialized.db_run_count, 5);
        assert_eq!(deserialized.jsonl_run_count, 4);
        assert!(!deserialized.is_valid);
        assert_eq!(deserialized.missing_from_jsonl, vec!["run-5"]);
        assert_eq!(deserialized.mismatched_records, vec!["run-2"]);
        assert_eq!(deserialized.child_mismatches.len(), 1);
    }

    // ── bd-246.2: Compression tests ──

    #[test]
    fn compression_round_trip() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("data.jsonl");
        let compressed = dir.path().join("data.jsonl.gz");
        let decompressed = dir.path().join("data_back.jsonl");

        let content = "{\"id\":\"run-1\",\"text\":\"hello world\"}\n\
                        {\"id\":\"run-2\",\"text\":\"second line\"}\n";
        fs::write(&original, content).expect("write original");

        compress_jsonl(&original, &compressed).expect("compress");
        assert!(compressed.exists(), "compressed file should exist");

        // Compressed file should be different from original (not plain text)
        let compressed_bytes = fs::read(&compressed).expect("read compressed");
        assert_ne!(
            compressed_bytes,
            content.as_bytes(),
            "compressed should differ from original"
        );

        decompress_jsonl(&compressed, &decompressed).expect("decompress");
        let recovered = fs::read_to_string(&decompressed).expect("read decompressed");
        assert_eq!(recovered, content, "round-trip should preserve content");
    }

    #[test]
    fn compression_empty_file() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("empty.jsonl");
        let compressed = dir.path().join("empty.jsonl.gz");
        let decompressed = dir.path().join("empty_back.jsonl");

        fs::write(&original, "").expect("write empty");
        compress_jsonl(&original, &compressed).expect("compress empty");
        decompress_jsonl(&compressed, &decompressed).expect("decompress empty");

        let recovered = fs::read_to_string(&decompressed).expect("read");
        assert_eq!(recovered, "", "empty file round-trip");
    }

    #[test]
    fn compression_large_content() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("large.jsonl");
        let compressed = dir.path().join("large.jsonl.gz");
        let decompressed = dir.path().join("large_back.jsonl");

        let mut content = String::new();
        for i in 0..1000 {
            content.push_str(&format!(
                "{{\"id\":\"run-{i}\",\"text\":\"line number {i} with some repeated data aaabbbccc\"}}\n"
            ));
        }
        fs::write(&original, &content).expect("write large");

        compress_jsonl(&original, &compressed).expect("compress");

        // Compressed should be smaller than original for repetitive data
        let original_size = fs::metadata(&original).expect("metadata").len();
        let compressed_size = fs::metadata(&compressed).expect("metadata").len();
        assert!(
            compressed_size < original_size,
            "compressed ({compressed_size}) should be smaller than original ({original_size})"
        );

        decompress_jsonl(&compressed, &decompressed).expect("decompress");
        let recovered = fs::read_to_string(&decompressed).expect("read");
        assert_eq!(recovered, content, "large content round-trip");
    }

    #[test]
    fn compression_mode_enum_equality() {
        assert_eq!(CompressionMode::None, CompressionMode::None);
        assert_eq!(CompressionMode::Gzip, CompressionMode::Gzip);
        assert_ne!(CompressionMode::None, CompressionMode::Gzip);
        let _ = format!("{:?}", CompressionMode::None);
        let _ = format!("{:?}", CompressionMode::Gzip);
    }

    #[test]
    fn open_jsonl_reader_plain_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("data.jsonl");
        fs::write(&path, "{\"id\":\"a\"}\n{\"id\":\"b\"}\n").expect("write");

        let reader = open_jsonl_reader(&path).expect("open");
        let lines: Vec<String> = reader
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .expect("read lines");
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"a\""));
        assert!(lines[1].contains("\"b\""));
    }

    #[test]
    fn open_jsonl_reader_compressed_file() {
        let dir = tempdir().expect("tempdir");
        let plain = dir.path().join("data.jsonl");
        let gz = dir.path().join("data.jsonl.gz");

        let content = "{\"id\":\"c\"}\n{\"id\":\"d\"}\n";
        fs::write(&plain, content).expect("write");
        compress_jsonl(&plain, &gz).expect("compress");

        let reader = open_jsonl_reader(&gz).expect("open gz");
        let lines: Vec<String> = reader
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .expect("read lines");
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"c\""));
        assert!(lines[1].contains("\"d\""));
    }

    #[test]
    fn validate_sync_with_compressed_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_gz.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("gz-val-1", &db_path))
            .expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Compress all JSONL files and remove the originals
        for stem in ["runs", "segments", "events"] {
            let plain = export_dir.join(format!("{stem}.jsonl"));
            let gz = export_dir.join(format!("{stem}.jsonl.gz"));
            compress_jsonl(&plain, &gz).expect("compress");
            fs::remove_file(&plain).expect("remove plain");
        }

        let validation = validate_sync(&db_path, &export_dir).expect("validate gz");
        assert!(
            validation.is_valid,
            "should be valid with compressed JSONL: {validation:?}"
        );
        assert_eq!(validation.db_run_count, 1);
        assert_eq!(validation.jsonl_run_count, 1);
        assert_eq!(validation.db_segment_count, 2);
        assert_eq!(validation.jsonl_segment_count, 2);
        assert_eq!(validation.db_event_count, 2);
        assert_eq!(validation.jsonl_event_count, 2);
    }

    #[test]
    fn gzip_only_export_imports_logical_jsonl_bytes_exactly() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("gzip_source.sqlite3");
        let target_db = dir.path().join("gzip_target.sqlite3");
        let plain_export_dir = dir.path().join("plain_export");
        let gzip_export_dir = dir.path().join("gzip_export");
        let export_state_root = dir.path().join("export_state");
        let import_state_root = dir.path().join("import_state");

        let source_store = RunStore::open(&source_db).expect("source store");
        let source_report = fixture_report("gzip-import-1", &source_db);
        source_store
            .persist_report(&source_report)
            .expect("persist source report");
        let manifest = export(&source_db, &plain_export_dir, &export_state_root)
            .expect("plain export");

        fs::create_dir_all(&gzip_export_dir).expect("gzip export dir");
        fs::copy(
            plain_export_dir.join("manifest.json"),
            gzip_export_dir.join("manifest.json"),
        )
        .expect("copy manifest");

        for (stem, expected_logical_hash) in [
            ("runs", &manifest.checksums.runs_jsonl_sha256),
            ("segments", &manifest.checksums.segments_jsonl_sha256),
            ("events", &manifest.checksums.events_jsonl_sha256),
        ] {
            let plain = plain_export_dir.join(format!("{stem}.jsonl"));
            let gzip = gzip_export_dir.join(format!("{stem}.jsonl.gz"));
            compress_jsonl(&plain, &gzip).expect("compress JSONL");
            let physical_hash = sha256_file(&gzip).expect("hash physical gzip bytes");
            assert_ne!(
                physical_hash.as_str(),
                expected_logical_hash.as_str(),
                "physical gzip bytes must not accidentally satisfy the manifest"
            );
            let logical_hash = sha256_jsonl_file(&gzip).expect("hash logical JSONL bytes");
            assert_eq!(
                logical_hash.as_str(),
                expected_logical_hash.as_str(),
                "decompressed JSONL bytes must preserve the export checksum"
            );
        }

        let result = import(
            &target_db,
            &gzip_export_dir,
            &import_state_root,
            ConflictPolicy::Reject,
        )
        .expect("gzip-only import");
        assert_eq!(result.runs_imported, manifest.row_counts.runs);
        assert_eq!(result.segments_imported, manifest.row_counts.segments);
        assert_eq!(result.events_imported, manifest.row_counts.events);
        assert!(result.conflicts.is_empty());

        let target_store = RunStore::open(&target_db).expect("target store");
        let imported = target_store
            .load_run_details("gzip-import-1")
            .expect("load imported run")
            .expect("imported run exists");
        assert_eq!(imported.transcript, source_report.result.transcript);
        assert_eq!(imported.segments, source_report.result.segments);
        assert_eq!(imported.events.len(), source_report.events.len());

        let validation = validate_sync(&target_db, &gzip_export_dir)
            .expect("validate gzip-only import");
        assert!(validation.is_valid, "gzip-only import must round-trip");
    }

    #[test]
    fn validate_sync_detects_segment_count_mismatch() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_seg_mismatch.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("seg-val", &db_path))
            .expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Append an extra line to segments.jsonl
        let seg_path = export_dir.join("segments.jsonl");
        let mut seg_file = OpenOptions::new()
            .append(true)
            .open(&seg_path)
            .expect("open segments for append");
        writeln!(
            seg_file,
            "{{\"run_id\":\"seg-val\",\"idx\":99,\"start_sec\":0.0,\"end_sec\":1.0,\"speaker\":null,\"text\":\"extra\",\"confidence\":0.5}}"
        )
        .expect("append");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect segment count mismatch");
        assert_eq!(validation.db_segment_count, 2);
        assert_eq!(validation.jsonl_segment_count, 3);
    }

    #[test]
    fn resolve_jsonl_path_prefers_gz() {
        let dir = tempdir().expect("tempdir");
        let base = dir.path();

        // Only plain file exists
        fs::write(base.join("runs.jsonl"), "plain").expect("write");
        let resolved = resolve_jsonl_path(base, "runs");
        assert_eq!(
            resolved,
            base.join("runs.jsonl"),
            "should resolve to plain when no gz"
        );

        // Create gz file — should now prefer it
        fs::write(base.join("runs.jsonl.gz"), "gzipped").expect("write gz");
        let resolved_gz = resolve_jsonl_path(base, "runs");
        assert_eq!(
            resolved_gz,
            base.join("runs.jsonl.gz"),
            "should prefer .gz when both exist"
        );
    }

    #[test]
    fn count_jsonl_lines_nonexistent_returns_zero() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("does_not_exist.jsonl");
        let count = count_jsonl_lines(&missing).expect("count");
        assert_eq!(count, 0, "nonexistent file should return 0 lines");
    }

    #[test]
    fn collect_jsonl_ids_nonexistent_returns_empty() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("does_not_exist.jsonl");
        let ids = collect_jsonl_ids(&missing, "id").expect("collect");
        assert!(ids.is_empty(), "nonexistent file should return empty set");
    }

    #[test]
    fn compress_jsonl_nonexistent_source_returns_error() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("no_such_file.jsonl");
        let output = dir.path().join("output.jsonl.gz");
        let result = compress_jsonl(&missing, &output);
        assert!(result.is_err(), "compressing nonexistent file should fail");
    }

    #[test]
    fn decompress_jsonl_nonexistent_source_returns_error() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("no_such_file.jsonl.gz");
        let output = dir.path().join("output.jsonl");
        let result = decompress_jsonl(&missing, &output);
        assert!(
            result.is_err(),
            "decompressing nonexistent file should fail"
        );
    }

    #[test]
    fn decompress_jsonl_invalid_gz_returns_error() {
        let dir = tempdir().expect("tempdir");
        let bad_gz = dir.path().join("bad.jsonl.gz");
        let output = dir.path().join("output.jsonl");
        fs::write(&bad_gz, "this is not gzip data").expect("write");
        let result = decompress_jsonl(&bad_gz, &output);
        assert!(result.is_err(), "decompressing invalid gz should fail");
    }

    #[test]
    fn compression_unicode_content() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("unicode.jsonl");
        let compressed = dir.path().join("unicode.jsonl.gz");
        let decompressed = dir.path().join("unicode_back.jsonl");

        let content = "{\"id\":\"u-1\",\"text\":\"\u{1F600} \u{4E16}\u{754C} caf\u{00E9}\"}\n";
        fs::write(&original, content).expect("write");
        compress_jsonl(&original, &compressed).expect("compress");
        decompress_jsonl(&compressed, &decompressed).expect("decompress");

        let recovered = fs::read_to_string(&decompressed).expect("read");
        assert_eq!(recovered, content, "unicode content round-trip");
    }

    #[test]
    fn json_str_or_empty_returns_empty_for_non_string_types() {
        let obj = json!({
            "name": "alice",
            "age": 42,
            "active": true,
            "scores": [1, 2, 3],
            "meta": {"nested": "object"},
            "gone": null
        });
        assert_eq!(json_str_or_empty(&obj, "name"), "alice");
        assert_eq!(json_str_or_empty(&obj, "age"), "");
        assert_eq!(json_str_or_empty(&obj, "active"), "");
        assert_eq!(json_str_or_empty(&obj, "scores"), "");
        assert_eq!(json_str_or_empty(&obj, "meta"), "");
        assert_eq!(json_str_or_empty(&obj, "gone"), "");
        assert_eq!(json_str_or_empty(&obj, "missing_key"), "");
    }

    #[test]
    fn collect_jsonl_ids_skips_blank_lines_and_missing_keys() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("mixed.jsonl");
        // Mix of: valid id, blank line, line missing "id" key, another valid id
        let content = r#"{"id":"run-1","status":"ok"}

{"status":"orphan"}
{"id":"run-2","status":"done"}
{"id":42}
"#;
        fs::write(&path, content).expect("write");
        let ids = collect_jsonl_ids(&path, "id").expect("collect");
        assert_eq!(ids.len(), 2, "should find exactly 2 string ids");
        assert!(ids.contains("run-1"));
        assert!(ids.contains("run-2"));
        // "42" should be excluded since it's an integer, not a string
        assert!(!ids.contains("42"));
    }

    #[test]
    fn load_jsonl_run_map_indexes_by_id() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("runs.jsonl");
        let content = r#"{"id":"r-a","backend":"whisper-cpp","started_at":"2025-01-01T00:00:00Z"}
{"id":"r-b","backend":"insanely-fast","started_at":"2025-01-02T00:00:00Z"}
"#;
        fs::write(&path, content).expect("write");
        let map = load_jsonl_run_map(&path).expect("load");
        assert_eq!(map.len(), 2);
        assert_eq!(
            map["r-a"].get("backend").and_then(|v| v.as_str()),
            Some("whisper-cpp")
        );
        assert_eq!(
            map["r-b"].get("backend").and_then(|v| v.as_str()),
            Some("insanely-fast")
        );
    }

    #[test]
    fn count_table_on_empty_table_returns_zero() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        let _store = RunStore::open(&db_path).expect("open");
        // After RunStore::open the tables exist but are empty.
        let conn = Connection::open(db_path.display().to_string()).expect("open sqlite conn");
        let run_count = count_table(&conn, "runs").expect("count runs");
        let seg_count = count_table(&conn, "segments").expect("count segments");
        let evt_count = count_table(&conn, "events").expect("count events");
        assert_eq!(run_count, 0);
        assert_eq!(seg_count, 0);
        assert_eq!(evt_count, 0);
    }

    #[test]
    fn count_jsonl_lines_skips_whitespace_only_lines() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("sparse.jsonl");
        // 3 data lines + 2 blank lines + 1 whitespace-only line
        let content = "{\"id\":\"a\"}\n\n{\"id\":\"b\"}\n   \n\n{\"id\":\"c\"}\n";
        fs::write(&path, content).expect("write");
        let count = count_jsonl_lines(&path).expect("count");
        assert_eq!(count, 3, "should count only non-empty/non-whitespace lines");
    }

    // -- seventeenth-pass edge case tests --

    #[test]
    fn max_export_position_with_partial_mutation_filter_returns_matching_maximum() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("partial_filter.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Persist two reports in a known mutation order.
        let mut early = fixture_report("run-early", &db_path);
        early.finished_at_rfc3339 = "2026-01-01T00:00:05Z".to_owned();
        store.persist_report(&early).expect("persist early");

        let mut late = fixture_report("run-late", &db_path);
        late.finished_at_rfc3339 = "2026-06-15T12:00:05Z".to_owned();
        store.persist_report(&late).expect("persist late");

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        // A cursor positioned at the early mutation should return late's tuple.
        let after_early = test_cursor(
            &conn,
            run_mutation_seq(&conn, "run-early"),
            "2026-01-01T00:00:05Z",
            Some("run-early"),
        );
        let result = max_export_position(&conn, Some(&after_early)).expect("query");
        assert_eq!(
            result,
            Some(("2026-06-15T12:00:05Z".to_owned(), "run-late".to_owned())),
            "should return the max among filtered runs"
        );

        // A cursor positioned at the late mutation should produce no newer rows.
        let after_late = test_cursor(
            &conn,
            run_mutation_seq(&conn, "run-late"),
            "2026-06-15T12:00:05Z",
            Some("run-late"),
        );
        let result = max_export_position(&conn, Some(&after_late)).expect("query");
        assert_eq!(result, None, "no runs strictly after the late mutation");
    }

    #[test]
    fn export_table_events_for_runs_empty_ids_produces_empty_file() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("evt_empty.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("evt-run", &db_path))
            .expect("persist");

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        let out_path = dir.path().join("events_empty.jsonl");
        let (count, _sha256) = export_table_events_for_runs(&conn, &out_path, &[]).expect("export");
        assert_eq!(count, 0, "empty run_ids should produce zero rows");
        let content = fs::read_to_string(&out_path).expect("read");
        assert!(content.is_empty(), "file should be empty");
    }

    #[test]
    fn collect_exported_run_ids_with_mutation_cursor_filters_correctly() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("collect_ids.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let mut r1 = fixture_report("run-a", &db_path);
        r1.finished_at_rfc3339 = "2026-01-01T00:00:05Z".to_owned();
        store.persist_report(&r1).expect("persist r1");

        let mut r2 = fixture_report("run-b", &db_path);
        r2.finished_at_rfc3339 = "2026-03-01T00:00:05Z".to_owned();
        store.persist_report(&r2).expect("persist r2");

        let mut r3 = fixture_report("run-c", &db_path);
        r3.finished_at_rfc3339 = "2026-06-01T00:00:05Z".to_owned();
        store.persist_report(&r3).expect("persist r3");

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        // None filter → all 3 IDs
        let all = collect_exported_run_ids(&conn, None).expect("all");
        assert_eq!(all.len(), 3, "no filter should return all runs");

        // Cursor at run-a should return later mutations run-b and run-c.
        let cursor = test_cursor(
            &conn,
            run_mutation_seq(&conn, "run-a"),
            "2026-01-01T00:00:05Z",
            Some("run-a"),
        );
        let filtered = collect_exported_run_ids(&conn, Some(&cursor)).expect("filtered");
        assert_eq!(filtered.len(), 2, "should exclude run-a");
        assert!(
            !filtered.contains(&"run-a".to_owned()),
            "run-a should be excluded"
        );
    }

    #[test]
    fn validate_sync_detects_event_count_mismatch() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_evt_mismatch.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("evt-val", &db_path))
            .expect("persist");

        export(&db_path, &export_dir, &state_root).expect("export");

        // Append an extra line to events.jsonl
        let evt_path = export_dir.join("events.jsonl");
        let mut evt_file = OpenOptions::new()
            .append(true)
            .open(&evt_path)
            .expect("open events for append");
        writeln!(
            evt_file,
            "{{\"run_id\":\"evt-val\",\"seq\":99,\"ts_rfc3339\":\"2026-01-01T00:00:09Z\",\"stage\":\"extra\",\"code\":\"extra.ok\",\"message\":\"extra event\",\"payload_json\":\"{{}}\"}}"
        )
        .expect("append");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect event count mismatch");
        assert_eq!(validation.db_event_count, 2);
        assert_eq!(validation.jsonl_event_count, 3);
    }

    #[test]
    fn export_table_runs_incremental_filters_by_mutation_sequence() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("inc_runs.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let mut r1 = fixture_report("inc-a", &db_path);
        r1.finished_at_rfc3339 = "2026-01-15T00:00:05Z".to_owned();
        store.persist_report(&r1).expect("persist r1");

        let mut r2 = fixture_report("inc-b", &db_path);
        r2.finished_at_rfc3339 = "2026-05-20T00:00:05Z".to_owned();
        store.persist_report(&r2).expect("persist r2");

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        // No filter → both runs exported.
        let all_path = dir.path().join("all_runs.jsonl");
        let (all_count, all_sha256) =
            export_table_runs_incremental(&conn, &all_path, None).expect("all");
        assert_eq!(all_count, 2, "no filter should export both runs");
        assert_eq!(
            all_sha256,
            super::sha256_file(&all_path).expect("re-read hash"),
            "incremental streamed hash must match sha256_file of the written bytes"
        );

        // Cursor at inc-a -> only inc-b.
        let filtered_path = dir.path().join("filtered_runs.jsonl");
        let cursor = test_cursor(
            &conn,
            run_mutation_seq(&conn, "inc-a"),
            "2026-01-15T00:00:05Z",
            Some("inc-a"),
        );
        let (filtered_count, _sha256) =
            export_table_runs_incremental(&conn, &filtered_path, Some(&cursor)).expect("filtered");
        assert_eq!(filtered_count, 1, "filter should export only inc-b");

        // Read and verify it's inc-b
        let content = fs::read_to_string(&filtered_path).expect("read");
        let line: serde_json::Value = serde_json::from_str(content.trim()).expect("parse");
        assert_eq!(line["id"], "inc-b", "exported run should be inc-b");
    }

    #[test]
    fn import_runs_missing_id_field_returns_error() {
        // Exercises json_str(&row, "id")? in import_runs when "id" key is absent.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("run-noid", &db_path))
            .expect("persist");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite runs.jsonl with a valid JSON row that has no "id" key.
        let runs_path = export_dir.join("runs.jsonl");
        let bad_row = json!({
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:05Z",
            "backend": "whisper_cpp",
            "input_path": "test.wav",
            "normalized_wav_path": "normalized.wav",
            "request_json": "{}",
            "result_json": "{}",
            "warnings_json": "[]",
            "transcript": "hello"
        });
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&bad_row).expect("ser")),
        )
        .expect("write");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        mval["checksums"]["runs_jsonl_sha256"] = json!(sha256_file(&runs_path).expect("checksum"));
        mval["row_counts"]["runs"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("ser"),
        )
        .expect("write manifest");

        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("missing id should fail");
        let text = err.to_string();
        assert!(
            text.contains("missing string field") && text.contains("id"),
            "error should mention missing 'id' field: {text}"
        );
    }

    #[test]
    fn import_segments_missing_run_id_field_returns_error() {
        // Exercises json_str(&row, "run_id")? in import_segments when "run_id" is absent.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("run-seg", &db_path))
            .expect("persist");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite segments.jsonl with a row missing "run_id".
        let seg_path = export_dir.join("segments.jsonl");
        let bad_seg = json!({
            "idx": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "text": "hello",
            "speaker": null,
            "confidence": null
        });
        fs::write(
            &seg_path,
            format!("{}\n", serde_json::to_string(&bad_seg).expect("ser")),
        )
        .expect("write");

        // Update manifest checksums.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        mval["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        mval["row_counts"]["segments"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("ser"),
        )
        .expect("write manifest");

        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("missing run_id should fail");
        let text = err.to_string();
        assert!(
            text.contains("missing string field") && text.contains("run_id"),
            "error should mention missing 'run_id' field: {text}"
        );
    }

    #[test]
    fn import_malformed_events_jsonl_returns_error() {
        // Exercises serde_json::from_str error path in import_events (line 1062).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("bad-evt", &db_path))
            .expect("persist");
        export_inner(&db_path, &export_dir).expect("export");

        // Replace events.jsonl with malformed JSON.
        let evt_path = export_dir.join("events.jsonl");
        fs::write(&evt_path, "{not valid json!!!}\n").expect("write");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        mval["checksums"]["events_jsonl_sha256"] = json!(sha256_file(&evt_path).expect("checksum"));
        mval["row_counts"]["events"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("ser"),
        )
        .expect("write manifest");

        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("malformed events should fail");
        let text = err.to_string();
        assert!(
            text.contains("key must be") || text.contains("json") || text.contains("JSON"),
            "error should mention JSON parse failure: {text}"
        );
    }

    #[test]
    fn load_jsonl_run_map_nonexistent_returns_empty() {
        let dir = tempdir().expect("tempdir");
        let nonexistent = dir.path().join("does_not_exist.jsonl");
        let map = load_jsonl_run_map(&nonexistent).expect("should succeed");
        assert!(
            map.is_empty(),
            "nonexistent file should produce empty map, got {} entries",
            map.len()
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_backend_field() {
        // Exercises the backend comparison branch at line 1560-1561 in validate_sync,
        // independent of the transcript comparison (which is tested separately).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_backend.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("bk-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate only the backend field in runs.jsonl (leave transcript unchanged).
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["backend"] = json!("totally_different_backend");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect backend field mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"bk-mismatch".to_owned()),
            "bk-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_input_path_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_input_path.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("input-path-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["input_path"] = json!("different.wav");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect input_path mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"input-path-mismatch".to_owned()),
            "input-path-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_normalized_wav_path_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_norm_path.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("norm-path-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["normalized_wav_path"] = json!("different_norm.wav");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(
            !validation.is_valid,
            "should detect normalized_wav_path mismatch"
        );
        assert!(
            validation
                .mismatched_records
                .contains(&"norm-path-mismatch".to_owned()),
            "norm-path-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_request_json_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_request_json.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("request-json-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["request_json"] = json!("{\"mutated\":true}");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect request_json mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"request-json-mismatch".to_owned()),
            "request-json-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_result_json_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_result_json.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("result-json-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["result_json"] = json!("{\"mutated_result\":true}");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect result_json mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"result-json-mismatch".to_owned()),
            "result-json-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_warnings_json_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_warnings_json.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("warnings-json-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["warnings_json"] = json!("[\"extra warning\"]");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect warnings_json mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"warnings-json-mismatch".to_owned()),
            "warnings-json-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn optional_floats_equal_zero_and_negative_values() {
        // Zero vs zero.
        assert!(optional_floats_equal(Some(0.0), Some(0.0)));
        // Negative matching values.
        assert!(optional_floats_equal(Some(-1.5), Some(-1.5)));
        // Negative divergent values.
        assert!(!optional_floats_equal(Some(-1.0), Some(-2.0)));
        // Sign mismatch.
        assert!(!optional_floats_equal(Some(-1.0), Some(1.0)));
    }

    #[test]
    fn collect_jsonl_ids_deduplicates_repeated_ids() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("dedup.jsonl");
        let content = [
            r#"{"id": "run-1", "data": 1}"#,
            r#"{"id": "run-1", "data": 2}"#,
            r#"{"id": "run-1", "data": 3}"#,
            r#"{"id": "run-2", "data": 4}"#,
        ]
        .join("\n");
        fs::write(&path, content).expect("write");

        let ids = collect_jsonl_ids(&path, "id").expect("collect");
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("run-1"));
        assert!(ids.contains("run-2"));
    }

    #[test]
    fn load_jsonl_run_map_skips_non_string_and_missing_ids() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("mixed.jsonl");
        let content = [
            r#"{"id": "valid-1", "text": "ok"}"#,
            r#"{"id": 42, "text": "integer id"}"#,
            r#"{"id": null, "text": "null id"}"#,
            r#"{"text": "no id field"}"#,
            "",
            r#"{"id": "valid-2", "text": "also ok"}"#,
        ]
        .join("\n");
        fs::write(&path, content).expect("write");

        let map = load_jsonl_run_map(&path).expect("load");
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("valid-1"));
        assert!(map.contains_key("valid-2"));
    }

    #[test]
    fn count_jsonl_lines_skips_blanks() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("sparse.jsonl");
        let content = [r#"{"id": "a"}"#, "", "   ", r#"{"id": "b"}"#, ""].join("\n");
        fs::write(&path, content).expect("write");

        let count = count_jsonl_lines(&path).expect("count");
        assert_eq!(count, 2, "should skip blank/whitespace-only lines");
    }

    #[test]
    fn count_jsonl_lines_missing_file_is_zero() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("does_not_exist.jsonl");
        let count = count_jsonl_lines(&path).expect("count");
        assert_eq!(count, 0);
    }

    // ── Task #211 — sync pass 2 edge-case tests ─────────────────────

    #[test]
    fn resolve_jsonl_path_neither_file_exists_returns_plain_jsonl() {
        let dir = tempdir().expect("tempdir");
        // Neither runs.jsonl nor runs.jsonl.gz exists.
        let path = resolve_jsonl_path(dir.path(), "runs");
        assert!(
            path.ends_with("runs.jsonl"),
            "should return .jsonl variant as fallback: {}",
            path.display()
        );
        assert!(
            !path.exists(),
            "returned path should not exist on disk: {}",
            path.display()
        );
    }

    #[test]
    fn load_cursor_valid_json_wrong_schema_returns_error() {
        let dir = tempdir().expect("tempdir");
        let cursor_path = dir.path().join("cursor.json");
        // Valid JSON but missing required fields (last_export_rfc3339, last_run_count).
        fs::write(&cursor_path, r#"{"wrong_field": "value", "another": 0}"#).expect("write");

        let result = load_cursor(&cursor_path);
        assert!(result.is_err(), "should fail on wrong schema");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("invalid sync cursor"),
            "error should mention invalid sync cursor: {msg}"
        );
    }

    #[test]
    fn sync_lock_release_inner_double_call_is_idempotent() {
        let dir = tempdir().expect("tempdir");
        let mut lock = SyncLock::acquire(dir.path(), "test-double-release").unwrap();

        // First release.
        lock.release_inner().unwrap();
        assert!(
            !dir.path().join("sync.lock").exists(),
            "lock file should be removed"
        );

        // Second release — hits the `if self.released { return Ok(()) }` guard.
        lock.release_inner().unwrap();

        // Prevent Drop from running file removal again (already released).
        lock.released = true;
    }

    #[test]
    fn collect_jsonl_ids_empty_file_returns_empty_set() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.jsonl");
        fs::write(&path, "").expect("write");

        let ids = collect_jsonl_ids(&path, "id").expect("collect");
        assert!(ids.is_empty(), "empty file should produce empty id set");
    }

    #[test]
    fn load_jsonl_run_map_valid_entries_preserves_all_fields() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("runs.jsonl");
        let line = serde_json::json!({
            "id": "run-123",
            "started_at": "2026-01-01T00:00:00Z",
            "extra_field": "preserved"
        });
        fs::write(&path, serde_json::to_string(&line).unwrap()).expect("write");

        let map = load_jsonl_run_map(&path).expect("load");
        assert_eq!(map.len(), 1);
        let entry = map.get("run-123").expect("should contain run-123");
        assert_eq!(entry["started_at"].as_str(), Some("2026-01-01T00:00:00Z"));
        assert_eq!(
            entry["extra_field"].as_str(),
            Some("preserved"),
            "extra fields should be preserved in the map"
        );
    }

    // ── Task #228 — sync.rs edge-case tests ────────────────────────────

    #[test]
    fn is_lock_stale_exactly_at_boundary_is_not_stale() {
        // Lock exactly LOCK_STALE_SECONDS (300s) old uses strict `>`, so it
        // should NOT be considered stale.
        let boundary_time = Utc::now() - chrono::Duration::seconds(LOCK_STALE_SECONDS);
        let info = LockInfo {
            pid: std::process::id(),
            created_at_rfc3339: boundary_time.to_rfc3339(),
            operation: "test".to_owned(),
        };
        assert!(
            !is_lock_stale(&info),
            "lock exactly at LOCK_STALE_SECONDS boundary should NOT be stale (strict >)"
        );
    }

    #[test]
    fn sql_ident_sync_escapes_embedded_double_quote() {
        // A column name with an embedded `"` must produce `"col""name"` (SQL standard).
        let result = sql_ident_sync("col\"name");
        assert_eq!(
            result, "\"col\"\"name\"",
            "embedded double-quote should be doubled"
        );

        // No quote → plain quoting.
        let plain = sql_ident_sync("simple");
        assert_eq!(plain, "\"simple\"");

        // Multiple quotes.
        let multi = sql_ident_sync("a\"b\"c");
        assert_eq!(multi, "\"a\"\"b\"\"c\"");
    }

    #[test]
    fn value_to_i64_sqlite_all_variants() {
        // Integer variant.
        assert_eq!(value_to_i64_sqlite(Some(&SqliteValue::Integer(42))), 42);

        // Text with valid integer string.
        assert_eq!(
            value_to_i64_sqlite(Some(&SqliteValue::Text("99".to_owned().into()))),
            99
        );

        // Text with non-parseable string → unwrap_or(0).
        assert_eq!(
            value_to_i64_sqlite(Some(&SqliteValue::Text("not_a_number".to_owned().into()))),
            0,
            "non-parseable text should return 0"
        );

        // None → 0.
        assert_eq!(value_to_i64_sqlite(None), 0);

        // Float → wildcard → 0.
        assert_eq!(
            value_to_i64_sqlite(Some(&SqliteValue::Float(std::f64::consts::PI))),
            0,
            "Float variant should fall through to 0"
        );
    }

    #[test]
    fn optional_floats_equal_exactly_at_epsilon_boundary_returns_true() {
        // Use values near zero where subtraction is exact.
        // (0.0 - 1e-9).abs() == 1e-9, and 1e-9 <= 1e-9 is true.
        assert!(
            optional_floats_equal(Some(0.0), Some(1e-9)),
            "difference of exactly 1e-9 should be equal (<=)"
        );
        // Confirm symmetry.
        assert!(
            optional_floats_equal(Some(1e-9), Some(0.0)),
            "symmetry: reversed args should also be equal"
        );
        // Just above boundary: 2e-9 > 1e-9 → not equal.
        assert!(
            !optional_floats_equal(Some(0.0), Some(2e-9)),
            "difference of 2e-9 should NOT be equal"
        );
    }

    #[test]
    fn max_started_at_returns_actual_max_from_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("max_started.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let mut early = fixture_report("run-early", &db_path);
        early.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        store.persist_report(&early).expect("persist early");

        let mut late = fixture_report("run-late", &db_path);
        late.started_at_rfc3339 = "2026-06-15T12:00:00Z".to_owned();
        store.persist_report(&late).expect("persist late");

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        // No filter → returns the overall maximum.
        let result = max_started_at(&conn, None).expect("query");
        assert_eq!(
            result,
            Some("2026-06-15T12:00:00Z".to_owned()),
            "should return the latest started_at"
        );

        // Filter after early → still returns late.
        let result = max_started_at(&conn, Some("2026-01-01T00:00:00Z")).expect("query");
        assert_eq!(
            result,
            Some("2026-06-15T12:00:00Z".to_owned()),
            "filter after early should return late"
        );

        // Filter beyond all → None.
        let result = max_started_at(&conn, Some("2099-12-31T23:59:59Z")).expect("query");
        assert_eq!(result, None, "no runs after far-future timestamp");

        drop(store);
    }

    #[test]
    fn import_skip_policy_preserves_existing_data_on_conflict() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("skip_src.sqlite3");
        let target_db = dir.path().join("skip_target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        // Persist a report in source DB and export it.
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-skip-1", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Import into target DB (first time — no conflict).
        let result1 =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip).expect("import 1");
        assert_eq!(result1.runs_imported, 1);
        assert!(
            result1.conflicts.is_empty(),
            "first import should have no conflicts"
        );

        // Import again — same run_id exists, Skip policy should silently skip the duplicate.
        let result2 =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip).expect("import 2");
        // The second import processes rows but skips duplicates.
        assert!(
            result2.conflicts.is_empty(),
            "Skip policy should not produce conflicts"
        );

        // Verify only one run exists in the target DB (not duplicated).
        let target_store = RunStore::open(&target_db).expect("open target");
        let runs = target_store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 1, "Skip should not duplicate the run");
    }

    #[test]
    fn import_valid_json_but_invalid_manifest_structure_returns_descriptive_error() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("bad_shape");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");
        fs::create_dir_all(&export_dir).expect("mkdir");

        // Write valid JSON that is NOT a valid SyncManifest (an array instead of object).
        fs::write(export_dir.join("manifest.json"), "[]").expect("write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("wrong structure should fail");
        let text = error.to_string();
        assert!(
            text.contains("invalid manifest"),
            "error should mention 'invalid manifest', got: {text}"
        );
    }

    #[test]
    fn compress_decompress_jsonl_round_trip_preserves_content() {
        let dir = tempdir().expect("tempdir");
        let original = dir.path().join("data.jsonl");
        let compressed = dir.path().join("data.jsonl.gz");
        let decompressed = dir.path().join("restored.jsonl");

        let content = "{\"id\":1,\"text\":\"hello world\"}\n{\"id\":2,\"text\":\"foo bar\"}\n";
        fs::write(&original, content).expect("write original");

        compress_jsonl(&original, &compressed).expect("compress");
        assert!(compressed.exists(), "compressed file should exist");
        // Compressed content should differ from original (gzip header).
        let raw_compressed = fs::read(&compressed).expect("read compressed");
        assert_ne!(
            raw_compressed,
            content.as_bytes(),
            "compressed should differ from original"
        );

        decompress_jsonl(&compressed, &decompressed).expect("decompress");
        let restored = fs::read_to_string(&decompressed).expect("read decompressed");
        assert_eq!(
            restored, content,
            "round-trip should preserve content exactly"
        );
    }

    #[test]
    fn import_valid_json_object_but_missing_required_fields_returns_invalid_manifest() {
        let dir = tempdir().expect("tempdir");
        let export_dir = dir.path().join("missing_fields");
        let state_root = dir.path().join("state");
        let target_db = dir.path().join("target.sqlite3");
        fs::create_dir_all(&export_dir).expect("mkdir");

        // Valid JSON object but missing all SyncManifest fields.
        fs::write(export_dir.join("manifest.json"), r#"{"foo": "bar"}"#).expect("write");

        let error = import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect_err("missing fields should fail");
        let text = error.to_string();
        assert!(
            text.contains("invalid manifest"),
            "error should mention 'invalid manifest', got: {text}"
        );
    }

    #[test]
    fn compress_jsonl_empty_file_produces_valid_gzip() {
        let dir = tempdir().expect("tempdir");
        let empty = dir.path().join("empty.jsonl");
        let compressed = dir.path().join("empty.jsonl.gz");
        let decompressed = dir.path().join("empty_restored.jsonl");

        fs::write(&empty, "").expect("write empty");
        compress_jsonl(&empty, &compressed).expect("compress empty");
        decompress_jsonl(&compressed, &decompressed).expect("decompress empty");

        let restored = fs::read_to_string(&decompressed).expect("read");
        assert_eq!(
            restored, "",
            "round-trip of empty file should produce empty file"
        );
    }

    // ── Task #256 — sync edge-case tests ────────────────────────────

    #[test]
    fn ensure_runs_replay_column_also_adds_acceleration_json() {
        // The existing test only checks for replay_json.  This verifies that
        // acceleration_json is also added when starting from a legacy schema
        // that has neither column.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("accel_check.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("open");

        conn.execute(
            "CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                backend TEXT NOT NULL,
                input_path TEXT NOT NULL,
                normalized_wav_path TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                transcript TEXT NOT NULL
            );",
        )
        .expect("create old schema");

        ensure_runs_replay_column(&conn).expect("migration");

        let rows = conn.query("PRAGMA table_info(runs);").expect("pragma");
        let has_acceleration = rows
            .iter()
            .any(|row| value_to_string_sqlite(row.get(1)) == "acceleration_json");
        assert!(
            has_acceleration,
            "acceleration_json column should exist after migration from legacy schema"
        );
    }

    #[test]
    fn ensure_runs_replay_column_adds_only_acceleration_to_partially_migrated_schema() {
        // Schema already has replay_json but NOT acceleration_json.
        // The function should skip adding replay_json and only add acceleration_json.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("partial_migrate.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("open");

        conn.execute(
            "CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                backend TEXT NOT NULL,
                input_path TEXT NOT NULL,
                normalized_wav_path TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                transcript TEXT NOT NULL,
                replay_json TEXT NOT NULL DEFAULT '{}'
            );",
        )
        .expect("create partially-migrated schema");

        // Insert a row to prove data survives the migration.
        conn.execute(
            "INSERT INTO runs (id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json) \
             VALUES ('r1', '2026-01-01', '2026-01-02', 'auto', 'in.wav', 'norm.wav', '{}', '{}', '[]', 'hello', '{}')",
        )
        .expect("insert test row");

        ensure_runs_replay_column(&conn).expect("migration");

        let rows = conn.query("PRAGMA table_info(runs);").expect("pragma");
        let has_acceleration = rows
            .iter()
            .any(|row| value_to_string_sqlite(row.get(1)) == "acceleration_json");
        assert!(
            has_acceleration,
            "acceleration_json should be added to partially-migrated schema"
        );

        // Verify existing data survived.
        let data = conn
            .query("SELECT transcript FROM runs WHERE id = 'r1'")
            .expect("query");
        assert_eq!(
            value_to_string_sqlite(data[0].get(0)),
            "hello",
            "existing row data should survive migration"
        );
    }

    #[test]
    fn import_skip_policy_with_conflicting_run_preserves_original_transcript() {
        // The existing Skip test re-imports identical data, which never actually
        // reaches the Skip branch (it takes the identical→noop shortcut).
        // This test mutates the transcript to force a true payload conflict.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("skip_run.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("run-skip-conflict", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import into a fresh target.
        let target_db = dir.path().join("target.sqlite3");
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip).expect("import 1");

        // Mutate the transcript in runs.jsonl.
        let runs_path = export_dir.join("runs.jsonl");
        let original_line = fs::read_to_string(&runs_path)
            .expect("read")
            .lines()
            .next()
            .expect("at least one line")
            .to_owned();
        let mut mutated: serde_json::Value = serde_json::from_str(&original_line).expect("parse");
        mutated["transcript"] = json!("MUTATED TRANSCRIPT FOR SKIP TEST");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write mutated");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["runs_jsonl_sha256"] =
            json!(sha256_file(&runs_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write manifest");

        // Re-import with Skip — should silently skip the conflicting run.
        let result2 =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip).expect("import 2");
        assert!(
            result2.conflicts.is_empty(),
            "Skip policy should not produce conflicts"
        );

        // Verify original transcript is preserved.
        let conn = Connection::open(target_db.display().to_string()).expect("open target");
        let rows = conn
            .query_with_params(
                "SELECT transcript FROM runs WHERE id = ?1",
                &[SqliteValue::Text("run-skip-conflict".to_owned().into())],
            )
            .expect("query");
        let transcript = value_to_string_sqlite(rows[0].get(0));
        assert_eq!(
            transcript, "hello world from sync test",
            "Skip should preserve original transcript, not overwrite with mutated"
        );
    }

    #[test]
    fn import_skip_policy_preserves_preexisting_run_as_a_whole_aggregate() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("skip_aggregate_source.sqlite3");
        let target_db = dir.path().join("skip_aggregate_target.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");
        let run_id = "run-skip-aggregate";

        let store = RunStore::open(&source_db).expect("store");
        store
            .persist_report(&fixture_report(run_id, &source_db))
            .expect("persist");
        export(&source_db, &export_dir, &state_root).expect("export");
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip)
            .expect("seed target");

        let dump_target = || {
            let connection = Connection::open(target_db.display().to_string())
                .expect("open target connection");
            let strings = |columns: Vec<SqliteValue>| -> Vec<String> {
                columns
                    .iter()
                    .map(|value| value_to_string_sqlite(Some(value)))
                    .collect()
            };
            (
                connection
                    .query("SELECT id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json FROM runs ORDER BY id")
                    .expect("dump runs")
                    .iter()
                    .map(run_row_to_cols)
                    .map(|columns| strings(columns))
                    .collect::<Vec<_>>(),
                connection
                    .query("SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence FROM segments ORDER BY run_id, idx")
                    .expect("dump segments")
                    .iter()
                    .map(segment_row_to_cols)
                    .map(|columns| strings(columns))
                    .collect::<Vec<_>>(),
                connection
                    .query("SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json FROM events ORDER BY run_id, seq")
                    .expect("dump events")
                    .iter()
                    .map(event_row_to_cols)
                    .map(|columns| strings(columns))
                    .collect::<Vec<_>>(),
            )
        };
        let original_aggregate = dump_target();

        let read_rows = |name: &str| -> Vec<serde_json::Value> {
            fs::read_to_string(export_dir.join(name))
                .expect("read JSONL")
                .lines()
                .map(|line| serde_json::from_str(line).expect("parse JSONL row"))
                .collect()
        };
        let runs = read_rows("runs.jsonl");
        let mut segments = read_rows("segments.jsonl");
        let mut events = read_rows("events.jsonl");

        // The parent remains byte-identical. Its early no-op must not stop the
        // child phase from treating it as a target-preexisting aggregate.
        segments[0]["text"] = json!("archive-only replacement");
        segments.push(json!({
            "run_id": run_id,
            "idx": 99,
            "start_sec": 5.0,
            "end_sec": 6.0,
            "speaker": null,
            "text": "archive-only child",
            "confidence": null
        }));
        events[0]["message"] = json!("archive-only replacement");
        events.push(json!({
            "run_id": run_id,
            "seq": 99,
            "ts_rfc3339": "2026-01-01T00:00:04Z",
            "stage": "archive",
            "code": "archive.only",
            "message": "archive-only child",
            "payload_json": "{}"
        }));
        write_jsonl_snapshot(&export_dir, &runs, &segments, &events);

        for batch_import_enabled in [false, true] {
            let result = import_inner_with_batch_mode(
                &target_db,
                &export_dir,
                ConflictPolicy::Skip,
                batch_import_enabled,
            )
            .expect("skip aggregate import");
            assert!(result.conflicts.is_empty(), "Skip remains silent");
            assert_eq!(result.runs_imported, 1, "processed parent row count");
            assert_eq!(result.segments_imported, 3, "processed segment row count");
            assert_eq!(result.events_imported, 3, "processed event row count");
            assert_eq!(
                dump_target(),
                original_aggregate,
                "Skip must preserve every stored parent and child column (batch={batch_import_enabled})"
            );
        }

        let mut conflicting_runs = runs.clone();
        conflicting_runs[0]["transcript"] = json!("archive-only transcript");
        write_jsonl_snapshot(&export_dir, &conflicting_runs, &segments, &events);
        for batch_import_enabled in [false, true] {
            import_inner_with_batch_mode(
                &target_db,
                &export_dir,
                ConflictPolicy::Skip,
                batch_import_enabled,
            )
            .expect("skip conflicting aggregate");
            assert_eq!(
                dump_target(),
                original_aggregate,
                "a conflicting parent must also preserve its whole aggregate (batch={batch_import_enabled})"
            );
        }

        let mut duplicate_run = runs[0].clone();
        duplicate_run["transcript"] = json!("later duplicate parent");
        write_jsonl_snapshot(
            &export_dir,
            &[runs[0].clone(), duplicate_run],
            &segments,
            &events,
        );
        for batch_import_enabled in [false, true] {
            let duplicate_target = dir
                .path()
                .join(format!("duplicate_target_{batch_import_enabled}.sqlite3"));
            let result = import_inner_with_batch_mode(
                &duplicate_target,
                &export_dir,
                ConflictPolicy::Skip,
                batch_import_enabled,
            )
            .expect("import archive-created duplicate parent");
            assert_eq!(result.runs_imported, 2);
            assert_eq!(result.segments_imported, 3);
            assert_eq!(result.events_imported, 3);

            let connection = Connection::open(duplicate_target.display().to_string())
                .expect("open duplicate target");
            let run_rows = connection
                .query("SELECT transcript FROM runs")
                .expect("query duplicate target run");
            assert_eq!(run_rows.len(), 1);
            assert_eq!(
                value_to_string_sqlite(run_rows[0].get(0)),
                "hello world from sync test",
                "first parent row remains authoritative"
            );
            let segment_count = connection
                .query("SELECT COUNT(*) FROM segments")
                .expect("count segments");
            assert_eq!(value_to_i64_sqlite(segment_count[0].get(0)), 3);
            let event_count = connection
                .query("SELECT COUNT(*) FROM events")
                .expect("count events");
            assert_eq!(value_to_i64_sqlite(event_count[0].get(0)), 3);
        }

        let child_only_segments = vec![json!({
            "run_id": run_id,
            "idx": 100,
            "start_sec": 6.0,
            "end_sec": 7.0,
            "speaker": null,
            "text": "child-only archive segment",
            "confidence": null
        })];
        let child_only_events = vec![json!({
            "run_id": run_id,
            "seq": 100,
            "ts_rfc3339": "2026-01-01T00:00:05Z",
            "stage": "archive",
            "code": "archive.child_only",
            "message": "child-only archive event",
            "payload_json": "{}"
        })];
        write_jsonl_snapshot(
            &export_dir,
            &[],
            &child_only_segments,
            &child_only_events,
        );

        for batch_import_enabled in [false, true] {
            let child_only_result = import_inner_with_batch_mode(
                &target_db,
                &export_dir,
                ConflictPolicy::Skip,
                batch_import_enabled,
            )
            .expect("skip child-only archive");
            assert_eq!(child_only_result.runs_imported, 0);
            assert_eq!(child_only_result.segments_imported, 1);
            assert_eq!(child_only_result.events_imported, 1);
            assert!(child_only_result.conflicts.is_empty());
            assert_eq!(
                dump_target(),
                original_aggregate,
                "child-only archives must not graft onto target-resident runs (batch={batch_import_enabled})"
            );
        }

        let malformed_segment = json!({
            "run_id": run_id,
            "idx": 101,
            "start_sec": 7.0,
            "end_sec": 8.0,
            "speaker": null,
            "confidence": null
        });
        write_jsonl_snapshot(&export_dir, &[], &[malformed_segment], &[]);
        for batch_import_enabled in [false, true] {
            let error = import_inner_with_batch_mode(
                &target_db,
                &export_dir,
                ConflictPolicy::Skip,
                batch_import_enabled,
            )
            .expect_err("skipped child rows must still validate required fields");
            assert!(
                error.to_string().contains("missing string field `text`"),
                "unexpected malformed skipped-row error: {error}"
            );
            assert_eq!(
                dump_target(),
                original_aggregate,
                "malformed skipped rows must roll the transaction back (batch={batch_import_enabled})"
            );
        }

        let malformed_event = json!({
            "run_id": run_id,
            "seq": 101,
            "ts_rfc3339": "2026-01-01T00:00:06Z",
            "stage": "archive",
            "code": "archive.malformed",
            "message": "missing payload"
        });
        write_jsonl_snapshot(&export_dir, &[], &[], &[malformed_event]);
        for batch_import_enabled in [false, true] {
            let error = import_inner_with_batch_mode(
                &target_db,
                &export_dir,
                ConflictPolicy::Skip,
                batch_import_enabled,
            )
            .expect_err("skipped events must still validate required fields");
            assert!(
                error
                    .to_string()
                    .contains("missing string field `payload_json`"),
                "unexpected malformed skipped-event error: {error}"
            );
            assert_eq!(
                dump_target(),
                original_aggregate,
                "malformed skipped events must roll the transaction back (batch={batch_import_enabled})"
            );
        }
    }

    #[test]
    fn import_skip_policy_with_conflicting_segment_preserves_original_text() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("skip_seg.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("run-seg-skip", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import into fresh target.
        let target_db = dir.path().join("target.sqlite3");
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip).expect("import 1");

        // Mutate a segment's text in segments.jsonl.
        let seg_path = export_dir.join("segments.jsonl");
        let original = fs::read_to_string(&seg_path).expect("read");
        let mutated = original.replace("hello world", "SKIP SEGMENT MUTATED");
        assert_ne!(original, mutated, "mutation should change the content");
        fs::write(&seg_path, &mutated).expect("write mutated segments");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["segments_jsonl_sha256"] =
            json!(sha256_file(&seg_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write manifest");

        // Re-import with Skip.
        let result2 =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip).expect("import 2");
        assert!(
            result2.conflicts.is_empty(),
            "Skip should produce no conflicts"
        );

        // Verify original segment text preserved.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn
            .query_with_params(
                "SELECT text FROM segments WHERE run_id = ?1 AND idx = 0",
                &[SqliteValue::Text("run-seg-skip".to_owned().into())],
            )
            .expect("query");
        let text = value_to_string_sqlite(rows[0].get(0));
        assert_eq!(
            text, "hello world",
            "Skip should preserve original segment text"
        );
    }

    #[test]
    fn import_skip_policy_with_conflicting_event_preserves_original_message() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("skip_evt.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("run-evt-skip", &db_path);
        store.persist_report(&report).expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // First import into fresh target.
        let target_db = dir.path().join("target.sqlite3");
        import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip).expect("import 1");

        // Mutate an event's message in events.jsonl.
        let evt_path = export_dir.join("events.jsonl");
        let original = fs::read_to_string(&evt_path).expect("read");
        let mutated = original.replace("input materialized", "SKIP EVENT MUTATED");
        assert_ne!(original, mutated, "mutation should change the content");
        fs::write(&evt_path, &mutated).expect("write mutated events");

        // Update manifest checksum.
        let manifest_path = export_dir.join("manifest.json");
        let mut manifest_value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        manifest_value["checksums"]["events_jsonl_sha256"] =
            json!(sha256_file(&evt_path).expect("checksum"));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest_value).expect("ser"),
        )
        .expect("write manifest");

        // Re-import with Skip.
        let result2 =
            import(&target_db, &export_dir, &state_root, ConflictPolicy::Skip).expect("import 2");
        assert!(
            result2.conflicts.is_empty(),
            "Skip should produce no conflicts"
        );

        // Verify original event message preserved.
        let conn = Connection::open(target_db.display().to_string()).expect("open");
        let rows = conn
            .query_with_params(
                "SELECT message FROM events WHERE run_id = ?1 AND seq = 1",
                &[SqliteValue::Text("run-evt-skip".to_owned().into())],
            )
            .expect("query");
        let message = value_to_string_sqlite(rows[0].get(0));
        assert_eq!(
            message, "input materialized",
            "Skip should preserve original event message"
        );
    }

    // ── Task #264 — sync.rs pass 12 edge-case tests ──────────────

    #[test]
    fn validate_sync_detects_mismatched_started_at_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_started.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("started-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate only the started_at field in runs.jsonl.
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["started_at"] = json!("1999-01-01T00:00:00Z");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect started_at mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"started-mismatch".to_owned()),
            "started-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_finished_at_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_finished.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("finished-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate only the finished_at field in runs.jsonl.
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["finished_at"] = json!("1999-12-31T23:59:59Z");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect finished_at mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"finished-mismatch".to_owned()),
            "finished-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn incremental_export_no_new_records_retains_cursor_values_exactly() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_retain.sqlite3");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("retain-cursor", &db_path))
            .expect("persist");

        // First export captures the run.
        let export_dir_1 = dir.path().join("export1");
        let m1 = export_incremental(&db_path, &export_dir_1, &state_root).expect("first export");
        assert_eq!(m1.row_counts.runs, 1);
        let first_cursor_ts = m1.cursor_after.last_export_rfc3339.clone();
        let first_cursor_run_id = m1.cursor_after.last_export_run_id.clone();
        let first_cursor_seq = m1.cursor_after.last_mutation_seq;

        // Second export: no new records.
        let export_dir_2 = dir.path().join("export2");
        let m2 = export_incremental(&db_path, &export_dir_2, &state_root).expect("second export");
        assert_eq!(m2.row_counts.runs, 0);

        // Cursor values should be EXACTLY retained from the first export.
        assert_eq!(
            m2.cursor_after.last_mutation_seq, first_cursor_seq,
            "mutation high-water mark should be retained when no changes exist"
        );
        assert_eq!(
            m2.cursor_after.last_export_rfc3339, first_cursor_ts,
            "cursor timestamp should be retained when no new records exist"
        );
        assert_eq!(
            m2.cursor_after.last_export_run_id, first_cursor_run_id,
            "cursor run_id should be retained when no new records exist"
        );
    }

    #[test]
    fn incremental_export_with_legacy_none_run_id_cursor_works() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("incr_legacy.sqlite3");
        let state_root = dir.path().join("state");
        fs::create_dir_all(&state_root).expect("create state dir");

        let store = RunStore::open(&db_path).expect("store open");
        let mut report = fixture_report("legacy-run", &db_path);
        report.finished_at_rfc3339 = "2026-06-15T12:00:00Z".to_owned();
        store.persist_report(&report).expect("persist");

        // Manually write a pre-identity cursor. Its numeric sequence is
        // deliberately far ahead: without a database identity it must be
        // ignored rather than interpreted in this database's sequence space.
        let legacy_cursor = r#"{"last_mutation_seq":999999,"last_export_rfc3339":"2026-01-01T00:00:00Z","last_run_count":0}"#;
        fs::write(state_root.join(CURSOR_FILENAME), legacy_cursor).expect("write legacy cursor");

        // Missing identity and run-id fields deserialize via defaults.
        let export_dir = dir.path().join("export");
        let manifest = export_incremental(&db_path, &export_dir, &state_root)
            .expect("export with legacy cursor");

        // A missing database identity deliberately treats the effective
        // sequence as zero and causes a one-time safe full re-export.
        assert_eq!(
            manifest.row_counts.runs, 1,
            "current run should be re-exported after a legacy cursor upgrade"
        );
        // cursor_used should reflect the legacy cursor with None run_id.
        assert!(
            manifest.cursor_used.is_some(),
            "cursor_used should be populated"
        );
        assert_eq!(
            manifest.cursor_used.as_ref().unwrap().last_mutation_seq,
            999999
        );
        assert!(manifest.cursor_used.as_ref().unwrap().database_id.is_none());
        assert!(
            manifest
                .cursor_used
                .as_ref()
                .unwrap()
                .last_export_run_id
                .is_none(),
            "legacy cursor should have None run_id"
        );
        // cursor_after should now have a run_id (upgraded from legacy).
        assert!(
            manifest.cursor_after.last_export_run_id.is_some(),
            "cursor_after should have a run_id after export"
        );
        assert!(
            manifest.cursor_after.database_id.is_some(),
            "cursor_after should bind the upgraded cursor to this database"
        );
    }

    #[test]
    fn value_to_i64_sqlite_negative_integer_and_negative_text() {
        // Negative integer via Integer variant.
        assert_eq!(
            value_to_i64_sqlite(Some(&SqliteValue::Integer(-42))),
            -42,
            "negative Integer should be returned directly"
        );

        // Negative integer as text.
        assert_eq!(
            value_to_i64_sqlite(Some(&SqliteValue::Text("-99".to_owned().into()))),
            -99,
            "negative text should parse correctly"
        );

        // Blob variant → wildcard → 0.
        assert_eq!(
            value_to_i64_sqlite(Some(&SqliteValue::Blob(vec![1, 2, 3].into()))),
            0,
            "Blob variant should fall through to 0"
        );
    }

    // ── Task #267 — sync.rs pass 13 edge-case tests ──────────────────

    #[test]
    fn validate_sync_detects_mismatched_replay_json_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_replay.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("replay-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate only the replay_json field in runs.jsonl.
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["replay_json"] = json!("{\"injected\":true}");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "should detect replay_json mismatch");
        assert!(
            validation
                .mismatched_records
                .contains(&"replay-mismatch".to_owned()),
            "replay-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_detects_mismatched_acceleration_json_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("val_accel.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("accel-mismatch", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Mutate only the acceleration_json field in runs.jsonl.
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read");
        let first_line = content.lines().next().expect("has line");
        let mut mutated: serde_json::Value = serde_json::from_str(first_line).expect("valid run");
        mutated["acceleration_json"] = json!("{\"gpu\":\"a100\"}");
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&mutated).expect("ser")),
        )
        .expect("write");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(
            !validation.is_valid,
            "should detect acceleration_json mismatch"
        );
        assert!(
            validation
                .mismatched_records
                .contains(&"accel-mismatch".to_owned()),
            "accel-mismatch should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn validate_sync_returns_error_for_db_without_schema() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("bare.sqlite3");
        let export_dir = dir.path().join("export");

        // Create a bare SQLite file with no tables.
        let _conn = Connection::open(db_path.display().to_string()).expect("create bare db");

        // Create empty JSONL files so validate_sync doesn't fail on missing files.
        fs::create_dir_all(&export_dir).expect("mkdir");
        for name in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
            fs::write(export_dir.join(name), "").expect("write empty file");
        }

        let err = validate_sync(&db_path, &export_dir).expect_err("should fail on no-schema DB");
        let msg = err.to_string();
        assert!(
            msg.contains("missing table"),
            "error should mention missing table, got: {msg}"
        );
    }

    #[test]
    fn json_optional_text_returns_null_for_explicit_json_null_value() {
        let value = json!({"name": "alice", "gone": null, "flag": true});
        // Explicit null value → Null (not panicking on the Some(Value::Null) arm).
        assert!(matches!(
            json_optional_text(&value, "gone"),
            SqliteValue::Null
        ));
        // Boolean value → Null.
        assert!(matches!(
            json_optional_text(&value, "flag"),
            SqliteValue::Null
        ));
    }

    #[test]
    fn json_to_optional_f64_returns_none_for_explicit_null_and_boolean() {
        let value = json!({"score": 42.5, "empty": null, "flag": true, "nested": {"a": 1}});
        // Explicit null → None.
        assert_eq!(json_to_optional_f64(&value, "empty"), None);
        // Boolean → None (as_f64 returns None for booleans).
        assert_eq!(json_to_optional_f64(&value, "flag"), None);
        // Object → None.
        assert_eq!(json_to_optional_f64(&value, "nested"), None);
    }

    // ── Task #270 — sync.rs pass 14 edge-case tests ──────────────────

    #[test]
    fn optional_floats_equal_nan_vs_finite_returns_false() {
        // NaN minus a finite value is NaN; NaN <= 1e-9 is false.
        assert!(
            !optional_floats_equal(Some(f64::NAN), Some(0.0)),
            "NaN vs 0.0 should not be equal"
        );
        assert!(
            !optional_floats_equal(Some(42.0), Some(f64::NAN)),
            "42.0 vs NaN should not be equal"
        );
        assert!(
            !optional_floats_equal(Some(f64::NAN), None),
            "Some(NaN) vs None should not be equal"
        );
    }

    #[test]
    fn json_to_optional_text_returns_none_for_explicit_null_and_non_string() {
        let value = json!({"name": "alice", "gone": null, "count": 42, "flag": true});
        // Explicit null → None.
        assert_eq!(json_to_optional_text(&value, "gone"), None);
        // Number → None.
        assert_eq!(json_to_optional_text(&value, "count"), None);
        // Boolean → None.
        assert_eq!(json_to_optional_text(&value, "flag"), None);
    }

    #[test]
    fn validate_sync_cross_missing_runs_both_directions() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cross_missing.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        // DB has run-a but NOT run-b.
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-a", &db_path))
            .expect("persist run-a");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Now add run-b to DB but remove run-a.
        store
            .persist_report(&fixture_report("run-b", &db_path))
            .expect("persist run-b");
        let conn = Connection::open(db_path.display().to_string()).expect("open connection");
        conn.execute_with_params(
            "DELETE FROM runs WHERE id = ?1",
            &[SqliteValue::Text("run-a".to_owned().into())],
        )
        .expect("delete run-a from DB");
        drop(conn);

        // JSONL has run-a (from export), DB has run-b. Both directions missing.
        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!validation.is_valid, "cross-missing runs should be invalid");
        assert!(
            !validation.missing_from_jsonl.is_empty(),
            "run-b should be in missing_from_jsonl"
        );
        assert!(
            !validation.missing_from_db.is_empty(),
            "run-a should be in missing_from_db"
        );
    }

    #[test]
    fn save_and_load_cursor_round_trip_preserves_run_id() {
        let dir = tempdir().expect("tempdir");
        let cursor_path = dir.path().join("nested/deep/cursor.json");
        let cursor = SyncCursor {
            last_mutation_seq: 42,
            database_id: Some("00000000-0000-0000-0000-000000000042".to_owned()),
            last_export_rfc3339: "2026-06-15T12:00:00Z".to_owned(),
            last_export_run_id: Some("run-xyz-123".to_owned()),
            last_run_count: 42,
        };
        save_cursor(&cursor_path, &cursor).expect("save");
        let loaded = load_cursor(&cursor_path)
            .expect("load")
            .expect("should be Some");
        assert_eq!(loaded.last_export_rfc3339, "2026-06-15T12:00:00Z");
        assert_eq!(
            loaded.last_export_run_id,
            Some("run-xyz-123".to_owned()),
            "last_export_run_id should round-trip through save/load"
        );
        assert_eq!(loaded.last_run_count, 42);
    }

    #[test]
    fn legacy_cursor_without_mutation_sequence_safely_reexports_all_runs() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("none_run_id.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Two runs with the same timestamp but different IDs.
        let mut r1 = fixture_report("run-a", &db_path);
        r1.finished_at_rfc3339 = "2026-06-01T00:00:05Z".to_owned();
        store.persist_report(&r1).expect("persist r1");

        let mut r2 = fixture_report("run-b", &db_path);
        r2.finished_at_rfc3339 = "2026-06-01T00:00:05Z".to_owned();
        store.persist_report(&r2).expect("persist r2");

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        // Pre-v6 cursor files deserialize with sequence zero. Their legacy
        // timestamp/run-id fields are intentionally ignored, causing one safe
        // full re-export after migration.
        let mut cursor = test_cursor(&conn, 0, "2026-06-01T00:00:05Z", None);
        cursor.database_id = None;
        let ids = collect_exported_run_ids(&conn, Some(&cursor)).expect("collect");
        assert_eq!(
            ids.len(),
            2,
            "legacy cursor should safely re-export every current run once"
        );
        assert!(ids.contains(&"run-a".to_owned()));
        assert!(ids.contains(&"run-b".to_owned()));
    }

    // ── Task #273 — sync.rs pass 15 edge-case tests ──────────────────

    #[test]
    fn collect_jsonl_ids_reads_from_compressed_gz_file() {
        let dir = tempdir().expect("tempdir");
        let plain = dir.path().join("runs.jsonl");
        let gz = dir.path().join("runs.jsonl.gz");
        let content = r#"{"id":"gz-1","text":"first"}
{"id":"gz-2","text":"second"}
{"id":"gz-3","text":"third"}
"#;
        fs::write(&plain, content).expect("write plain");
        compress_jsonl(&plain, &gz).expect("compress");

        // collect_jsonl_ids should transparently decompress via open_jsonl_reader.
        let ids = collect_jsonl_ids(&gz, "id").expect("collect from gz");
        assert_eq!(ids.len(), 3);
        assert!(ids.contains("gz-1"));
        assert!(ids.contains("gz-2"));
        assert!(ids.contains("gz-3"));
    }

    #[test]
    fn load_jsonl_run_map_duplicate_id_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("dup.jsonl");
        let content = [
            r#"{"id": "dup-1", "version": "first"}"#,
            r#"{"id": "dup-1", "version": "second"}"#,
            r#"{"id": "dup-2", "version": "only"}"#,
        ]
        .join("\n");
        fs::write(&path, content).expect("write");

        let error = load_jsonl_run_map(&path).expect_err("duplicate run ID must fail closed");
        let message = error.to_string();
        assert!(message.contains("duplicate run id"));
        assert!(message.contains("dup-1"));
        assert!(message.contains("line 2"));
    }

    #[test]
    fn validate_sync_detects_mismatched_transcript_field() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("transcript_mm.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("txn-1", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Tamper with transcript field in runs.jsonl
        let runs_path = export_dir.join("runs.jsonl");
        let content = fs::read_to_string(&runs_path).expect("read runs");
        let tampered = content.replace("hello world from sync test", "TAMPERED TRANSCRIPT");
        assert_ne!(content, tampered, "sanity: content should have changed");
        fs::write(&runs_path, tampered).expect("write tampered");

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(
            !validation.is_valid,
            "tampered transcript should make validation fail"
        );
        assert!(
            validation.mismatched_records.contains(&"txn-1".to_owned()),
            "txn-1 should be in mismatched_records: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn sync_cursor_deserialize_missing_run_id_defaults_to_none() {
        // SyncCursor has #[serde(default)] on last_export_run_id.
        // Older cursors without this field should deserialize to None.
        let json = r#"{"last_export_rfc3339":"2026-01-01T00:00:00Z","last_run_count":5}"#;
        let cursor: SyncCursor = serde_json::from_str(json).expect("deserialize");
        assert_eq!(cursor.last_mutation_seq, 0);
        assert_eq!(cursor.last_export_rfc3339, "2026-01-01T00:00:00Z");
        assert_eq!(
            cursor.last_export_run_id, None,
            "missing field should default to None"
        );
        assert_eq!(cursor.last_run_count, 5);

        // With the field present: should deserialize normally.
        let json_with = r#"{"last_export_rfc3339":"2026-06-01T00:00:00Z","last_export_run_id":"run-42","last_run_count":10}"#;
        let cursor2: SyncCursor = serde_json::from_str(json_with).expect("deserialize");
        assert_eq!(cursor2.last_mutation_seq, 0);
        assert_eq!(cursor2.last_export_run_id, Some("run-42".to_owned()));
    }

    #[test]
    fn count_jsonl_lines_from_compressed_gz_file() {
        let dir = tempdir().expect("tempdir");
        let plain = dir.path().join("events.jsonl");
        let gz = dir.path().join("events.jsonl.gz");
        let content = r#"{"seq":1}
{"seq":2}

{"seq":3}
"#;
        fs::write(&plain, content).expect("write plain");
        compress_jsonl(&plain, &gz).expect("compress");

        // count_jsonl_lines should transparently decompress gz.
        let count = count_jsonl_lines(&gz).expect("count from gz");
        assert_eq!(count, 3, "should count 3 non-empty lines from gz file");
    }

    // ── Task #275 — sync.rs pass 16 edge-case tests ──────────────────

    #[test]
    fn validate_sync_legacy_jsonl_without_replay_json_key_matches_db_default() {
        // When JSONL row lacks "replay_json" key, json_string_or_default returns "{}",
        // matching the DB's default value. Should NOT be flagged as a mismatch.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("legacy_replay.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("legacy-1", &db_path))
            .expect("persist");
        export(&db_path, &export_dir, &state_root).expect("export");

        // Strip "replay_json" and "acceleration_json" from the JSONL to simulate legacy.
        let runs_path = export_dir.join("runs.jsonl");
        let original = fs::read_to_string(&runs_path).expect("read");
        let mut parsed: serde_json::Value = serde_json::from_str(original.trim()).expect("parse");
        parsed.as_object_mut().unwrap().remove("replay_json");
        parsed.as_object_mut().unwrap().remove("acceleration_json");
        fs::write(&runs_path, serde_json::to_string(&parsed).unwrap()).expect("write");

        // Update the DB to have default "{}" for both columns.
        let conn = Connection::open(db_path.display().to_string()).expect("open connection");
        conn.execute_with_params(
            "UPDATE runs SET replay_json = '{}', acceleration_json = '{}' WHERE id = ?1",
            &[SqliteValue::Text("legacy-1".to_owned().into())],
        )
        .expect("reset to defaults");
        drop(conn);

        let validation = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(
            validation.mismatched_records.is_empty(),
            "legacy JSONL without replay_json should match DB default: {:?}",
            validation.mismatched_records
        );
    }

    #[test]
    fn load_jsonl_run_map_reads_from_compressed_gz_file() {
        let dir = tempdir().expect("tempdir");
        let plain = dir.path().join("runs.jsonl");
        let gz = dir.path().join("runs.jsonl.gz");
        let content = r#"{"id":"gz-a","backend":"whisper"}
{"id":"gz-b","backend":"insanely-fast"}
"#;
        fs::write(&plain, content).expect("write plain");
        compress_jsonl(&plain, &gz).expect("compress");

        let map = load_jsonl_run_map(&gz).expect("load from gz");
        assert_eq!(map.len(), 2);
        assert_eq!(map["gz-a"]["backend"].as_str(), Some("whisper"));
        assert_eq!(map["gz-b"]["backend"].as_str(), Some("insanely-fast"));
    }

    #[test]
    fn ensure_run_reference_exists_caches_known_run_id_for_second_call() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ref_cache.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("cached-run", &db_path))
            .expect("persist");
        drop(store);

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");

        let mut known = HashSet::new();
        // First call: queries DB, inserts into known_run_ids.
        ensure_run_reference_exists(&conn, "cached-run", "segments", "key-1", &mut known)
            .expect("first call should succeed");
        assert!(
            known.contains("cached-run"),
            "run_id should be cached after first successful lookup"
        );

        // Second call: should return Ok from cache without DB query.
        // (We can't prove no query happened, but we can verify it succeeds
        // and the cache still contains the id.)
        ensure_run_reference_exists(&conn, "cached-run", "events", "key-2", &mut known)
            .expect("second call should succeed from cache");
        assert_eq!(known.len(), 1, "should still have exactly 1 cached id");
    }

    #[test]
    fn count_table_with_populated_data_returns_correct_count() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("populated.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Persist two runs so `runs` has 2 rows and `segments`/`events` get populated.
        let r1 = fixture_report("run-a", &db_path);
        let r2 = fixture_report("run-b", &db_path);
        store.persist_report(&r1).expect("persist r1");
        store.persist_report(&r2).expect("persist r2");
        drop(store);

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");
        let run_count = count_table(&conn, "runs").expect("count runs");
        assert_eq!(run_count, 2, "should count 2 runs");

        // segments table should be populated from fixture_report's segments.
        let seg_count = count_table(&conn, "segments").expect("count segments");
        assert!(seg_count > 0, "fixture_report adds at least one segment");
    }

    #[test]
    fn export_table_runs_incremental_no_matching_runs_produces_empty_file() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("inc_empty.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Persist a run with an old timestamp.
        let mut r = fixture_report("old-run", &db_path);
        r.finished_at_rfc3339 = "2020-01-01T00:00:00Z".to_owned();
        store.persist_report(&r).expect("persist");
        drop(store);

        let conn = Connection::open(db_path.display().to_string()).expect("open connection");
        let output_path = dir.path().join("runs.jsonl");

        // Cursor at the mutation high-water mark — no runs should match.
        let cursor = test_cursor(
            &conn,
            mutation_high_water(&conn),
            "2099-01-01T00:00:00Z",
            Some("zzz"),
        );
        let (count, _sha256) =
            export_table_runs_incremental(&conn, &output_path, Some(&cursor)).expect("export");
        assert_eq!(count, 0, "no runs should match a current high-water cursor");

        let content = fs::read_to_string(&output_path).expect("read");
        assert!(
            content.trim().is_empty(),
            "file should be empty when no runs match"
        );
    }

    #[test]
    fn json_str_returns_error_for_array_and_boolean_values() {
        let obj = json!({"list": [1,2,3], "flag": true});
        let err_list = json_str(&obj, "list").expect_err("array should not be a string");
        assert!(err_list.to_string().contains("missing string field"));
        let err_flag = json_str(&obj, "flag").expect_err("boolean should not be a string");
        assert!(err_flag.to_string().contains("missing string field"));
    }

    #[test]
    fn json_str_returns_error_message_includes_key_name() {
        let obj = json!({"x": 1});
        let err = json_str(&obj, "my_special_key").expect_err("missing key");
        let msg = err.to_string();
        assert!(
            msg.contains("my_special_key"),
            "error message should include the key name, got: {msg}"
        );
    }

    #[test]
    fn sqlite_to_optional_text_returns_none_for_non_text_variants() {
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Integer(42))),
            None
        );
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Float(std::f64::consts::PI))),
            None
        );
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Blob(vec![1, 2].into()))),
            None
        );
        assert_eq!(sqlite_to_optional_text(Some(&SqliteValue::Null)), None);
        assert_eq!(sqlite_to_optional_text(None), None);
        // Text returns Some.
        assert_eq!(
            sqlite_to_optional_text(Some(&SqliteValue::Text("hello".to_owned().into()))),
            Some("hello".to_owned())
        );
    }

    #[test]
    fn sync_validation_report_serde_round_trip_with_mismatches() {
        let report = SyncValidationReport {
            db_run_count: 5,
            jsonl_run_count: 4,
            db_segment_count: 10,
            jsonl_segment_count: 10,
            db_event_count: 20,
            jsonl_event_count: 18,
            missing_from_jsonl: vec!["run-extra".to_owned()],
            missing_from_db: vec![],
            mismatched_records: vec!["run-001".to_owned(), "run-002".to_owned()],
            child_mismatches: vec![],
            child_mismatches_truncated: true,
            is_valid: false,
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let parsed: SyncValidationReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.db_run_count, 5);
        assert_eq!(parsed.jsonl_run_count, 4);
        assert_eq!(parsed.db_event_count, 20);
        assert_eq!(parsed.jsonl_event_count, 18);
        assert_eq!(parsed.missing_from_jsonl, vec!["run-extra"]);
        assert!(parsed.missing_from_db.is_empty());
        assert_eq!(parsed.mismatched_records.len(), 2);
        assert!(parsed.child_mismatches_truncated);
        assert!(!parsed.is_valid);
    }

    #[test]
    fn validate_checksums_detects_tampered_file() {
        let dir = tempdir().expect("tempdir");
        let runs = dir.path().join("runs.jsonl");
        let segments = dir.path().join("segments.jsonl");
        let events = dir.path().join("events.jsonl");
        fs::write(&runs, "original\n").expect("write runs");
        fs::write(&segments, "").expect("write segments");
        fs::write(&events, "").expect("write events");
        let manifest = SyncManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "test.db".to_owned(),
            row_counts: RowCounts {
                runs: 1,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: sha256_file(&runs).expect("sha"),
                segments_jsonl_sha256: sha256_file(&segments).expect("sha"),
                events_jsonl_sha256: sha256_file(&events).expect("sha"),
            },
        };
        // Valid checksums should pass.
        assert!(validate_checksums(&manifest, dir.path()).is_ok());
        // Tamper with runs.jsonl.
        fs::write(&runs, "tampered\n").expect("tamper");
        let err = validate_checksums(&manifest, dir.path()).expect_err("should fail");
        match err {
            FwError::Storage(msg) => assert!(
                msg.contains("checksum mismatch"),
                "error should mention checksum mismatch, got: {msg}"
            ),
            other => assert!(
                matches!(other, FwError::Storage(_)),
                "expected FwError::Storage, got: {other:?}"
            ),
        }
    }

    #[test]
    fn load_cursor_returns_none_for_nonexistent_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("missing_cursor.json");
        let result = load_cursor(&path).expect("should not error");
        assert!(
            result.is_none(),
            "nonexistent cursor file should return None"
        );
    }

    #[test]
    fn save_cursor_then_load_cursor_preserves_all_fields() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("cursor_rt2.json");
        let cursor = SyncCursor {
            last_mutation_seq: 7,
            database_id: Some("00000000-0000-0000-0000-000000000007".to_owned()),
            last_export_rfc3339: "2026-02-15T10:00:00Z".to_owned(),
            last_export_run_id: Some("run-42".to_owned()),
            last_run_count: 7,
        };
        save_cursor(&path, &cursor).expect("save");
        let loaded = load_cursor(&path)
            .expect("load should not error")
            .expect("cursor should exist after save");
        assert_eq!(loaded.last_export_rfc3339, "2026-02-15T10:00:00Z");
        assert_eq!(loaded.last_export_run_id.as_deref(), Some("run-42"));
        assert_eq!(loaded.last_run_count, 7);
    }

    #[test]
    fn load_cursor_returns_error_for_invalid_json() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bad_cursor.json");
        fs::write(&path, "this is not json").expect("write");
        let err = load_cursor(&path).expect_err("should error on invalid JSON");
        match err {
            FwError::Storage(msg) => assert!(
                msg.contains("invalid sync cursor"),
                "error should mention invalid sync cursor, got: {msg}"
            ),
            other => assert!(
                matches!(other, FwError::Storage(_)),
                "expected FwError::Storage, got: {other:?}"
            ),
        }
    }

    #[test]
    fn atomic_write_bytes_creates_file_and_removes_tmp() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("output.txt");
        let tmp = path.with_extension("tmp");
        atomic_write_bytes(&path, b"hello world").expect("write");
        assert!(path.exists(), "target file should exist");
        assert!(!tmp.exists(), "tmp file should be cleaned up");
        let content = fs::read_to_string(&path).expect("read");
        assert_eq!(content, "hello world");
    }

    #[test]
    fn validate_checksums_detects_missing_file() {
        let dir = tempdir().expect("tempdir");
        // Create segments and events but NOT runs.
        fs::write(dir.path().join("segments.jsonl"), "").expect("write");
        fs::write(dir.path().join("events.jsonl"), "").expect("write");
        let manifest = SyncManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "test.db".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: "deadbeef".to_owned(),
                segments_jsonl_sha256: sha256_file(&dir.path().join("segments.jsonl"))
                    .expect("sha"),
                events_jsonl_sha256: sha256_file(&dir.path().join("events.jsonl")).expect("sha"),
            },
        };
        let err = validate_checksums(&manifest, dir.path()).expect_err("should fail");
        match err {
            FwError::Storage(msg) => assert!(
                msg.contains("missing export file"),
                "error should mention missing file, got: {msg}"
            ),
            other => assert!(
                matches!(other, FwError::Storage(_)),
                "expected FwError::Storage, got: {other:?}"
            ),
        }
    }

    #[test]
    fn max_export_position_no_cursor_returns_latest_run() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        let mut report1 = fixture_report("run-a", &db_path);
        report1.finished_at_rfc3339 = "2026-01-01T00:00:01Z".to_owned();
        store.persist_report(&report1).expect("persist");
        let mut report2 = fixture_report("run-b", &db_path);
        report2.finished_at_rfc3339 = "2026-01-01T00:00:05Z".to_owned();
        store.persist_report(&report2).expect("persist");

        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let pos = max_export_position(&conn, None).expect("max_export_position");
        let (ts, run_id) = pos.expect("should be Some");
        assert_eq!(ts, "2026-01-01T00:00:05Z");
        assert_eq!(run_id, "run-b");
    }

    #[test]
    fn max_export_position_with_cursor_filters_earlier_mutations() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        let mut r1 = fixture_report("run-x", &db_path);
        r1.finished_at_rfc3339 = "2026-01-01T00:00:01Z".to_owned();
        store.persist_report(&r1).expect("persist");
        let mut r2 = fixture_report("run-y", &db_path);
        r2.finished_at_rfc3339 = "2026-01-01T00:00:10Z".to_owned();
        store.persist_report(&r2).expect("persist");

        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let cursor = test_cursor(
            &conn,
            run_mutation_seq(&conn, "run-x"),
            "2026-01-01T00:00:05Z",
            None,
        );
        let pos = max_export_position(&conn, Some(&cursor)).expect("max_export_position");
        let (ts, run_id) = pos.expect("should find run-y after cursor");
        assert_eq!(ts, "2026-01-01T00:00:10Z");
        assert_eq!(run_id, "run-y");
    }

    #[test]
    fn max_export_position_empty_db_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        let _store = RunStore::open(&db_path).expect("store open");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let pos = max_export_position(&conn, None).expect("max_export_position");
        assert!(pos.is_none(), "empty db should return None");
    }

    #[test]
    fn ensure_schema_idempotent_on_existing_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("schema_idem.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        ensure_schema(&conn).expect("first ensure_schema");
        // Calling again should not error (IF NOT EXISTS).
        ensure_schema(&conn).expect("second ensure_schema should be idempotent");
        // Verify tables still queryable.
        let runs_rows = conn.query("SELECT COUNT(*) FROM runs").expect("runs");
        assert!(!runs_rows.is_empty(), "runs table should exist");
    }

    #[test]
    fn atomic_write_bytes_creates_parent_dirs_not_needed() {
        // atomic_write_bytes writes to a file within an existing directory.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("data.bin");
        atomic_write_bytes(&path, &[0xDE, 0xAD, 0xBE, 0xEF]).expect("write");
        let bytes = fs::read(&path).expect("read");
        assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn query_segment_idxs_for_run_returns_correct_indices() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("seg_idx.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-seg-idx", &db_path))
            .expect("persist");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let idxs = query_segment_idxs_for_run(&conn, "run-seg-idx").expect("query");
        // fixture_report produces 2 segments (idx 0 and 1).
        assert_eq!(idxs.len(), 2);
        assert!(idxs.contains(&0));
        assert!(idxs.contains(&1));
    }

    #[test]
    fn query_event_seqs_for_run_returns_correct_seqs() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("evt_seq.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-evt-seq", &db_path))
            .expect("persist");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let seqs = query_event_seqs_for_run(&conn, "run-evt-seq").expect("query");
        // fixture_report produces 2 events (seq 1 and 2).
        assert_eq!(seqs.len(), 2);
        assert!(seqs.contains(&1));
        assert!(seqs.contains(&2));
    }

    #[test]
    fn query_segment_idxs_for_run_returns_empty_for_unknown_run() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("seg_empty.sqlite3");
        let _store = RunStore::open(&db_path).expect("store open");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let idxs = query_segment_idxs_for_run(&conn, "nonexistent-run").expect("query");
        assert!(idxs.is_empty(), "unknown run_id should yield empty set");
    }

    #[test]
    fn verify_schema_exists_rejects_missing_events_table() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_events.sqlite3");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        // Create runs and segments but NOT events.
        conn.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY); \
             CREATE TABLE segments (run_id TEXT, idx INTEGER, PRIMARY KEY(run_id, idx));",
        )
        .expect("create partial schema");
        let err = verify_schema_exists(&conn).expect_err("missing events should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("events"),
            "error should mention missing events table, got: {msg}"
        );
    }

    #[test]
    fn ensure_run_reference_exists_succeeds_for_existing_run() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ref_ok.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("ref-run", &db_path))
            .expect("persist");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let mut known = std::collections::HashSet::new();
        ensure_run_reference_exists(&conn, "ref-run", "segments", "key-1", &mut known)
            .expect("existing run should succeed");
        // After the first call, the run_id should be cached in `known`.
        assert!(known.contains("ref-run"), "run_id should be cached");
    }

    #[test]
    fn query_event_seqs_for_run_returns_empty_for_unknown_run() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("evt_empty.sqlite3");
        let _store = RunStore::open(&db_path).expect("store open");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let seqs = query_event_seqs_for_run(&conn, "nonexistent-run").expect("query");
        assert!(seqs.is_empty(), "unknown run_id should yield empty set");
    }

    #[test]
    fn verify_schema_exists_succeeds_after_ensure_schema_with_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("full_schema.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("pop-run", &db_path))
            .expect("persist");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        // Should succeed: all three tables exist and have data.
        verify_schema_exists(&conn).expect("schema with data should pass");
    }

    #[test]
    fn count_table_returns_nonzero_after_insert() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("count_test.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("count-r1", &db_path))
            .expect("persist r1");
        store
            .persist_report(&fixture_report("count-r2", &db_path))
            .expect("persist r2");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let run_count = count_table(&conn, "runs").expect("count runs");
        assert_eq!(run_count, 2, "should have 2 runs");
        // Each fixture_report has 2 segments.
        let seg_count = count_table(&conn, "segments").expect("count segments");
        assert_eq!(seg_count, 4, "should have 4 segments (2 per run)");
        // Each fixture_report has 2 events.
        let evt_count = count_table(&conn, "events").expect("count events");
        assert_eq!(evt_count, 4, "should have 4 events (2 per run)");
    }

    #[test]
    fn sync_lock_release_allows_reacquisition() {
        let dir = tempdir().expect("tempdir");
        let lock = SyncLock::acquire(dir.path(), "first_op").expect("first acquire");
        // Release explicitly.
        lock.release().expect("release");
        // Now we should be able to acquire again.
        let lock2 = SyncLock::acquire(dir.path(), "second_op").expect("second acquire");
        drop(lock2);
    }

    #[test]
    fn ensure_run_reference_exists_error_message_includes_table_and_key() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ref_err.sqlite3");
        let _store = RunStore::open(&db_path).expect("store open");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let mut known = std::collections::HashSet::new();
        let err =
            ensure_run_reference_exists(&conn, "missing-run", "segments", "seg-key-42", &mut known)
                .expect_err("should fail for missing run");
        let msg = format!("{err}");
        assert!(
            msg.contains("segments"),
            "error should mention table name, got: {msg}"
        );
        assert!(
            msg.contains("seg-key-42"),
            "error should mention key, got: {msg}"
        );
        assert!(
            msg.contains("missing-run"),
            "error should mention run_id, got: {msg}"
        );
    }

    #[test]
    fn export_table_runs_produces_valid_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ex_runs.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-exp-1", &db_path))
            .expect("persist");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let out = dir.path().join("runs.jsonl");
        let (count, sha256) = export_table_runs(&conn, &out).expect("export");
        assert_eq!(count, 1, "should export 1 run");
        assert_eq!(
            sha256,
            super::sha256_file(&out).expect("re-read hash"),
            "streamed hash must match sha256_file of the written bytes"
        );
        let content = fs::read_to_string(&out).expect("read");
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).expect("parse");
        assert_eq!(parsed["id"], "run-exp-1");
        assert!(
            parsed.get("transcript").is_some(),
            "should have transcript field"
        );
    }

    #[test]
    fn export_table_segments_produces_valid_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ex_segs.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-exp-s", &db_path))
            .expect("persist");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let out = dir.path().join("segments.jsonl");
        let (count, sha256) = export_table_segments(&conn, &out).expect("export");
        assert_eq!(count, 2, "fixture has 2 segments");
        assert_eq!(
            sha256,
            super::sha256_file(&out).expect("re-read hash"),
            "streamed hash must match sha256_file of the written bytes"
        );
        let content = fs::read_to_string(&out).expect("read");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("parse");
        assert_eq!(first["run_id"], "run-exp-s");
        assert_eq!(first["idx"], 0);
    }

    #[test]
    fn export_table_events_produces_valid_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ex_evts.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-exp-e", &db_path))
            .expect("persist");
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        let out = dir.path().join("events.jsonl");
        let (count, sha256) = export_table_events(&conn, &out).expect("export");
        assert_eq!(count, 2, "fixture has 2 events");
        assert_eq!(
            sha256,
            super::sha256_file(&out).expect("re-read hash"),
            "streamed hash must match sha256_file of the written bytes"
        );
        let content = fs::read_to_string(&out).expect("read");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("parse");
        assert_eq!(first["run_id"], "run-exp-e");
        assert!(first.get("stage").is_some(), "should have stage field");
    }

    #[test]
    fn conflict_policy_skip_equals_itself() {
        assert_eq!(ConflictPolicy::Skip, ConflictPolicy::Skip);
        assert_ne!(ConflictPolicy::Skip, ConflictPolicy::Reject);
        assert_ne!(ConflictPolicy::Skip, ConflictPolicy::Overwrite);
        assert_ne!(ConflictPolicy::Skip, ConflictPolicy::OverwriteStrict);
        let _ = format!("{:?}", ConflictPolicy::Skip);
    }

    #[test]
    fn archive_stale_lock_renames_with_reason_in_filename() {
        let dir = tempdir().expect("tempdir");
        let locks_dir = dir.path().join("locks");
        fs::create_dir_all(&locks_dir).expect("create locks dir");
        let lock_path = locks_dir.join("sync.lock");
        fs::write(&lock_path, b"test lock content").expect("write lock");
        assert!(lock_path.exists(), "lock file should exist before archive");
        archive_stale_lock(&lock_path, "stale").expect("archive");
        assert!(!lock_path.exists(), "lock file should be moved");
        // Find the archived file.
        let entries: Vec<_> = fs::read_dir(&locks_dir)
            .expect("read dir")
            .filter_map(Result::ok)
            .collect();
        assert_eq!(entries.len(), 1, "should have exactly one archived file");
        let name = entries[0].file_name().to_string_lossy().to_string();
        assert!(
            name.contains("stale"),
            "archived filename should contain reason, got: {name}"
        );
        assert!(
            name.starts_with("sync.lock.stale."),
            "archived filename should follow expected format, got: {name}"
        );
    }

    #[test]
    fn assert_no_stale_segments_passes_when_all_indices_replaced() {
        let mut overwritten = HashSet::new();
        overwritten.insert("run-1".to_owned());

        let mut existing: HashMap<String, HashSet<i64>> = HashMap::new();
        existing.insert("run-1".to_owned(), HashSet::from([0, 1, 2]));

        let mut imported: HashMap<String, HashSet<i64>> = HashMap::new();
        imported.insert("run-1".to_owned(), HashSet::from([0, 1, 2]));

        assert!(
            assert_no_stale_segments_for_overwritten_runs(&overwritten, &existing, &imported)
                .is_ok(),
            "should pass when all existing indices are in imported set"
        );
    }

    #[test]
    fn assert_no_stale_segments_errors_on_missing_imported_index() {
        let mut overwritten = HashSet::new();
        overwritten.insert("run-1".to_owned());

        let mut existing: HashMap<String, HashSet<i64>> = HashMap::new();
        existing.insert("run-1".to_owned(), HashSet::from([0, 1, 2]));

        let mut imported: HashMap<String, HashSet<i64>> = HashMap::new();
        // Only import indices 0 and 1, leaving index 2 stale.
        imported.insert("run-1".to_owned(), HashSet::from([0, 1]));

        let err = assert_no_stale_segments_for_overwritten_runs(&overwritten, &existing, &imported)
            .expect_err("should fail when a stale segment exists");
        let msg = err.to_string();
        assert!(
            msg.contains("stale segment"),
            "error should mention stale segment: {msg}"
        );
        assert!(msg.contains("run-1"), "error should mention run id: {msg}");
    }

    #[test]
    fn assert_no_stale_events_errors_on_missing_imported_seq() {
        let mut overwritten = HashSet::new();
        overwritten.insert("run-a".to_owned());

        let mut existing: HashMap<String, HashSet<i64>> = HashMap::new();
        existing.insert("run-a".to_owned(), HashSet::from([1, 2, 3]));

        let mut imported: HashMap<String, HashSet<i64>> = HashMap::new();
        // Only import seqs 1 and 2, leaving seq 3 stale.
        imported.insert("run-a".to_owned(), HashSet::from([1, 2]));

        let err = assert_no_stale_events_for_overwritten_runs(&overwritten, &existing, &imported)
            .expect_err("should fail when a stale event exists");
        let msg = err.to_string();
        assert!(
            msg.contains("stale event"),
            "error should mention stale event: {msg}"
        );
        assert!(msg.contains("run-a"), "error should mention run id: {msg}");
    }

    #[test]
    fn table_columns_sync_returns_columns_for_known_table() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cols_sync.sqlite3");
        // Open via RunStore to initialize schema, then use raw connection.
        let _store = RunStore::open(&db_path).expect("store");

        let conn = Connection::open(db_path.display().to_string()).expect("open");
        let columns = table_columns_sync(&conn, "segments").expect("table_columns_sync");
        // segments table has 7 columns: run_id, idx, start_sec, end_sec,
        // speaker, text, confidence.
        assert_eq!(columns.len(), 7, "segments table should have 7 columns");
    }

    #[test]
    fn reconstruct_column_definition_sync_produces_defs_for_events_table() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("reconstr_sync.sqlite3");
        let _store = RunStore::open(&db_path).expect("store");

        let conn = Connection::open(db_path.display().to_string()).expect("open");
        let columns = table_columns_sync(&conn, "events").expect("columns");
        // events table: run_id, seq, ts_rfc3339, stage, code, message, payload_json.
        assert_eq!(columns.len(), 7, "events table should have 7 columns");

        let mut names = Vec::new();
        for row in &columns {
            let def = reconstruct_column_definition_sync(row).expect("reconstruct");
            let name = value_to_string_sqlite(row.get(1));
            assert!(
                def.contains(&name),
                "definition should contain column name `{name}`: {def}"
            );
            names.push(name);
        }
        assert!(
            names.contains(&"run_id".to_owned()),
            "events should have run_id column"
        );
        assert!(
            names.contains(&"payload_json".to_owned()),
            "events should have payload_json column"
        );
    }

    #[test]
    fn assert_no_stale_events_passes_when_all_seqs_replaced() {
        let mut overwritten = HashSet::new();
        overwritten.insert("run-x".to_owned());

        let mut existing: HashMap<String, HashSet<i64>> = HashMap::new();
        existing.insert("run-x".to_owned(), HashSet::from([1, 2, 3]));

        let mut imported: HashMap<String, HashSet<i64>> = HashMap::new();
        imported.insert("run-x".to_owned(), HashSet::from([1, 2, 3]));

        assert!(
            assert_no_stale_events_for_overwritten_runs(&overwritten, &existing, &imported).is_ok(),
            "should pass when all existing event seqs are in imported set"
        );
    }

    #[test]
    fn sync_parent_dir_succeeds_for_tempdir_file() {
        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("test.jsonl");
        std::fs::write(&file_path, "{}").expect("write");
        assert!(
            sync_parent_dir(&file_path).is_ok(),
            "sync_parent_dir should succeed for a file in tempdir"
        );
    }

    #[test]
    fn incremental_export_manifest_cursor_none_round_trip() {
        let manifest = IncrementalExportManifest {
            schema_version: "0.1.0".to_owned(),
            export_format_version: "1".to_owned(),
            export_mode: "incremental".to_owned(),
            created_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            source_db_path: "/tmp/test.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: 0,
                segments: 0,
                events: 0,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: "aaa".to_owned(),
                segments_jsonl_sha256: "bbb".to_owned(),
                events_jsonl_sha256: "ccc".to_owned(),
            },
            cursor_used: None,
            cursor_after: SyncCursor {
                last_mutation_seq: 1,
                database_id: Some("00000000-0000-0000-0000-000000000001".to_owned()),
                last_export_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                last_export_run_id: Some("run-new".to_owned()),
                last_run_count: 0,
            },
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let parsed: IncrementalExportManifest = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.cursor_used.is_none(), "cursor_used should be None");
        assert_eq!(parsed.row_counts.runs, 0);
        assert_eq!(parsed.export_mode, "incremental");
    }

    #[test]
    fn import_result_debug_displays_counts() {
        let result = ImportResult {
            runs_imported: 5,
            segments_imported: 20,
            events_imported: 15,
            conflicts: vec![SyncConflict {
                table: "runs".to_owned(),
                key: "run-1".to_owned(),
                reason: "conflict".to_owned(),
            }],
        };
        let debug = format!("{:?}", result);
        assert!(
            debug.contains("runs_imported"),
            "debug output should contain field name: {debug}"
        );
        assert!(
            debug.contains("5"),
            "debug output should contain count: {debug}"
        );
    }

    #[test]
    fn collect_exported_run_ids_no_cursor_returns_all_runs() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("collect_ids.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut r1 = fixture_report("run-a", &db_path);
        r1.finished_at_rfc3339 = "2026-01-01T00:00:01Z".to_owned();
        store.persist_report(&r1).expect("persist r1");

        let mut r2 = fixture_report("run-b", &db_path);
        r2.finished_at_rfc3339 = "2026-01-01T00:00:02Z".to_owned();
        store.persist_report(&r2).expect("persist r2");

        let conn = Connection::open(db_path.display().to_string()).expect("open");
        let ids = collect_exported_run_ids(&conn, None).expect("collect");
        assert_eq!(ids.len(), 2, "should find both runs");
        assert_eq!(
            ids[0], "run-a",
            "first should be run-a (earlier finished_at)"
        );
        assert_eq!(
            ids[1], "run-b",
            "second should be run-b (later finished_at)"
        );
    }

    // ------------------------------------------------------------------
    // sync edge-case tests pass 25
    // ------------------------------------------------------------------

    #[test]
    fn max_export_position_returns_none_when_finished_at_is_empty() {
        // Exercises the defensive guard at line 724:
        //   if ts.is_empty() || run_id.is_empty() { Ok(None) }
        // which protects against a row where finished_at is empty-string.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_ts.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = fixture_report("run-empty-ts", &db_path);
        store.persist_report(&report).expect("persist");

        // Forcibly overwrite `finished_at` with empty string via raw SQL.
        let conn = Connection::open(db_path.display().to_string()).expect("conn");
        conn.execute_with_params(
            "UPDATE runs SET finished_at = ?1 WHERE id = ?2",
            &[
                SqliteValue::Text(String::new().into()),
                SqliteValue::Text("run-empty-ts".to_owned().into()),
            ],
        )
        .expect("update finished_at to empty");

        let pos = max_export_position(&conn, None).expect("should not error");
        assert!(
            pos.is_none(),
            "empty finished_at should cause max_export_position to return None"
        );
    }

    #[test]
    fn assert_no_stale_segments_skips_run_without_existing_entries() {
        // Exercises the `continue` branch at line 1312-1313:
        //   let Some(existing_idxs) = existing_idx_before_by_run.get(run_id) else { continue; };
        // When overwritten_run_ids contains a run_id that has NO entry in
        // existing_idx_before_by_run (e.g., a brand-new run that didn't exist
        // in the target DB before import), the function should skip it and
        // return Ok(()).
        let mut overwritten = HashSet::new();
        overwritten.insert("new-run".to_owned());

        // existing_idx_before_by_run is empty — "new-run" has no pre-existing segments.
        let existing: HashMap<String, HashSet<i64>> = HashMap::new();

        let mut imported: HashMap<String, HashSet<i64>> = HashMap::new();
        imported.insert("new-run".to_owned(), HashSet::from([0, 1, 2]));

        assert!(
            assert_no_stale_segments_for_overwritten_runs(&overwritten, &existing, &imported)
                .is_ok(),
            "run_id absent from existing map should be skipped (no stale segments)"
        );
    }

    #[test]
    fn import_events_missing_run_id_field_returns_error() {
        // Exercises json_str(&row, "run_id")? in import_events (line 1213)
        // when the "run_id" key is absent. The analogous test for
        // import_segments exists (import_segments_missing_run_id_field_returns_error)
        // but import_events was missing this coverage.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("run-evt", &db_path))
            .expect("persist");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite events.jsonl with a row that has valid seq but no "run_id".
        let evt_path = export_dir.join("events.jsonl");
        let bad_event = json!({
            "seq": 1,
            "ts_rfc3339": "2026-01-01T00:00:01Z",
            "stage": "ingest",
            "code": "ingest.ok",
            "message": "input materialized",
            "payload_json": "{}"
            // "run_id" intentionally omitted
        });
        fs::write(
            &evt_path,
            format!("{}\n", serde_json::to_string(&bad_event).expect("ser")),
        )
        .expect("write");

        // Update manifest checksums and row counts.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        mval["checksums"]["events_jsonl_sha256"] = json!(sha256_file(&evt_path).expect("checksum"));
        mval["row_counts"]["events"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("ser"),
        )
        .expect("write manifest");

        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("missing run_id in events should fail");
        let text = err.to_string();
        assert!(
            text.contains("missing string field") && text.contains("run_id"),
            "error should mention missing 'run_id' field: {text}"
        );
    }

    #[test]
    fn import_runs_missing_started_at_in_new_run_returns_error() {
        // Exercises json_str(&row, "started_at")? at line 1044 inside the
        // INSERT block of import_runs. The existing test
        // import_runs_missing_id_field_returns_error exercises the missing "id"
        // path, but never reaches the INSERT block. This test provides a valid
        // "id" (so it passes line 930) but omits "started_at", which errors at
        // line 1044 during the INSERT parameter construction.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");

        let store = RunStore::open(&db_path).expect("open");
        store
            .persist_report(&fixture_report("run-insert", &db_path))
            .expect("persist");
        export_inner(&db_path, &export_dir).expect("export");

        // Overwrite runs.jsonl with a row that has "id" but not "started_at".
        let runs_path = export_dir.join("runs.jsonl");
        let valid_result_json = serde_json::to_string(
            &fixture_report("run-no-started-at", &db_path).result,
        )
        .expect("serialize valid result");
        let bad_row = json!({
            "id": "run-no-started-at",
            "finished_at": "2026-01-01T00:00:05Z",
            "backend": "whisper_cpp",
            "input_path": "test.wav",
            "normalized_wav_path": "normalized.wav",
            "request_json": "{}",
            "result_json": valid_result_json,
            "warnings_json": "[]",
            "transcript": "hello"
            // "started_at" intentionally omitted
        });
        fs::write(
            &runs_path,
            format!("{}\n", serde_json::to_string(&bad_row).expect("ser")),
        )
        .expect("write");

        // Update manifest checksums and row counts.
        let manifest_path = export_dir.join("manifest.json");
        let mut mval: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read"))
                .expect("parse");
        mval["checksums"]["runs_jsonl_sha256"] = json!(sha256_file(&runs_path).expect("checksum"));
        mval["row_counts"]["runs"] = json!(1);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&mval).expect("ser"),
        )
        .expect("write manifest");

        let import_db = dir.path().join("import.sqlite3");
        let err = import_inner(&import_db, &export_dir, ConflictPolicy::Reject)
            .expect_err("missing started_at should fail");
        let text = err.to_string();
        assert!(
            text.contains("missing string field") && text.contains("started_at"),
            "error should mention missing 'started_at' field: {text}"
        );
    }

    #[test]
    fn collect_jsonl_ids_malformed_json_returns_error() {
        // Exercises the serde_json::from_str error propagation at line 2035
        // inside collect_jsonl_ids. Existing tests all use valid JSON lines;
        // this test provides malformed JSON to exercise the `?` propagation.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bad.jsonl");
        fs::write(&path, "this is not valid json\n").expect("write");

        let result = collect_jsonl_ids(&path, "id");
        assert!(
            result.is_err(),
            "malformed JSON line should cause collect_jsonl_ids to return an error"
        );
    }

    // -----------------------------------------------------------------------
    // bd-2l2: compress_jsonl / decompress_jsonl round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn compress_decompress_round_trip_preserves_content() {
        let dir = tempdir().expect("tempdir");
        let input_path = dir.path().join("data.jsonl");
        let compressed_path = dir.path().join("data.jsonl.gz");
        let output_path = dir.path().join("data_recovered.jsonl");

        let original =
            "{\"id\":\"run-1\",\"text\":\"hello\"}\n{\"id\":\"run-2\",\"text\":\"world\"}\n";
        fs::write(&input_path, original).expect("write input");

        compress_jsonl(&input_path, &compressed_path).expect("compress should succeed");
        assert!(compressed_path.exists(), "compressed file should exist");

        decompress_jsonl(&compressed_path, &output_path).expect("decompress should succeed");
        let recovered = fs::read_to_string(&output_path).expect("read output");
        assert_eq!(
            recovered, original,
            "round-trip should preserve content exactly"
        );
    }

    #[test]
    fn compress_jsonl_empty_file() {
        let dir = tempdir().expect("tempdir");
        let input_path = dir.path().join("empty.jsonl");
        let compressed_path = dir.path().join("empty.jsonl.gz");

        fs::write(&input_path, "").expect("write empty");
        compress_jsonl(&input_path, &compressed_path).expect("compress empty should succeed");
        assert!(compressed_path.exists());

        let output_path = dir.path().join("empty_recovered.jsonl");
        decompress_jsonl(&compressed_path, &output_path).expect("decompress empty should succeed");
        let recovered = fs::read_to_string(&output_path).expect("read");
        assert!(
            recovered.is_empty(),
            "empty input should produce empty output"
        );
    }

    #[test]
    fn compress_jsonl_nonexistent_input_returns_error() {
        let dir = tempdir().expect("tempdir");
        let input_path = dir.path().join("no_such_file.jsonl");
        let compressed_path = dir.path().join("output.jsonl.gz");

        let result = compress_jsonl(&input_path, &compressed_path);
        assert!(result.is_err(), "nonexistent input should return error");
    }

    #[test]
    fn decompress_jsonl_nonexistent_input_returns_error() {
        let dir = tempdir().expect("tempdir");
        let input_path = dir.path().join("no_such_file.jsonl.gz");
        let output_path = dir.path().join("output.jsonl");

        let result = decompress_jsonl(&input_path, &output_path);
        assert!(
            result.is_err(),
            "nonexistent compressed input should return error"
        );
    }

    #[test]
    fn decompress_jsonl_corrupted_data_returns_error() {
        let dir = tempdir().expect("tempdir");
        let input_path = dir.path().join("corrupted.jsonl.gz");
        let output_path = dir.path().join("output.jsonl");

        fs::write(&input_path, b"not valid gzip data").expect("write corrupted");
        let result = decompress_jsonl(&input_path, &output_path);
        assert!(result.is_err(), "corrupted gzip data should return error");
    }

    #[test]
    fn compress_jsonl_large_content() {
        let dir = tempdir().expect("tempdir");
        let input_path = dir.path().join("large.jsonl");
        let compressed_path = dir.path().join("large.jsonl.gz");
        let output_path = dir.path().join("large_recovered.jsonl");

        let mut content = String::new();
        for i in 0..100 {
            content.push_str(&format!(
                "{{\"id\":\"run-{i}\",\"data\":\"{}\"}}\n",
                "x".repeat(200)
            ));
        }
        fs::write(&input_path, &content).expect("write large");

        compress_jsonl(&input_path, &compressed_path).expect("compress large");
        decompress_jsonl(&compressed_path, &output_path).expect("decompress large");
        let recovered = fs::read_to_string(&output_path).expect("read large");
        assert_eq!(
            recovered, content,
            "large round-trip should preserve content"
        );

        let compressed_size = fs::metadata(&compressed_path).expect("meta").len();
        let original_size = content.len() as u64;
        assert!(
            compressed_size < original_size,
            "compressed size ({compressed_size}) should be smaller than original ({original_size})"
        );
    }

    // -----------------------------------------------------------------------
    // bd-2l2: max_started_at function tests (distinct from max_export_position tests)
    // -----------------------------------------------------------------------

    #[test]
    fn max_started_at_fn_empty_db_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let _store = RunStore::open(&db_path).expect("store open");

        let conn = Connection::open(db_path.display().to_string()).expect("conn open");
        let result = max_started_at(&conn, None).expect("query should succeed");
        assert!(result.is_none(), "empty DB should return None");
    }

    #[test]
    fn max_started_at_fn_with_data_returns_latest() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");

        let report1 = {
            let mut r = fixture_report("run-1", &db_path);
            r.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
            r
        };
        let report2 = {
            let mut r = fixture_report("run-2", &db_path);
            r.started_at_rfc3339 = "2026-02-15T12:00:00Z".to_owned();
            r
        };
        store.persist_report(&report1).expect("save report1");
        store.persist_report(&report2).expect("save report2");

        let conn = Connection::open(db_path.display().to_string()).expect("conn open");
        let result = max_started_at(&conn, None).expect("query should succeed");
        assert_eq!(
            result,
            Some("2026-02-15T12:00:00Z".to_owned()),
            "should return the latest started_at"
        );
    }

    #[test]
    fn max_started_at_fn_with_after_ts_filter_returns_only_newer() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");

        let report1 = {
            let mut r = fixture_report("run-a", &db_path);
            r.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
            r
        };
        let report2 = {
            let mut r = fixture_report("run-b", &db_path);
            r.started_at_rfc3339 = "2026-02-01T00:00:00Z".to_owned();
            r
        };
        let report3 = {
            let mut r = fixture_report("run-c", &db_path);
            r.started_at_rfc3339 = "2026-03-01T00:00:00Z".to_owned();
            r
        };
        store.persist_report(&report1).expect("save report1");
        store.persist_report(&report2).expect("save report2");
        store.persist_report(&report3).expect("save report3");

        let conn = Connection::open(db_path.display().to_string()).expect("conn open");

        let result =
            max_started_at(&conn, Some("2026-01-15T00:00:00Z")).expect("query should succeed");
        assert_eq!(
            result,
            Some("2026-03-01T00:00:00Z".to_owned()),
            "should return the max among rows after the filter"
        );
    }

    #[test]
    fn max_started_at_fn_with_filter_excluding_all_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let store = RunStore::open(&db_path).expect("store open");

        let report = {
            let mut r = fixture_report("run-old", &db_path);
            r.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
            r
        };
        store.persist_report(&report).expect("save report");

        let conn = Connection::open(db_path.display().to_string()).expect("conn open");
        let result =
            max_started_at(&conn, Some("2026-12-31T23:59:59Z")).expect("query should succeed");
        assert!(result.is_none(), "filter after all rows should return None");
    }

    // -----------------------------------------------------------------------
    // bd-1ao: validate_sync tests
    // -----------------------------------------------------------------------

    #[test]
    fn validate_sync_after_export_reports_valid() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-valid", &db_path))
            .expect("persist");
        drop(store);

        export(&db_path, &export_dir, &state_root).expect("export");

        let report = validate_sync(&db_path, &export_dir).expect("validate should succeed");
        assert!(
            report.is_valid,
            "freshly exported db should validate as synced"
        );
        assert_eq!(report.db_run_count, 1);
        assert_eq!(report.jsonl_run_count, 1);
        assert!(report.missing_from_jsonl.is_empty());
        assert!(report.missing_from_db.is_empty());
        assert!(report.mismatched_records.is_empty());
        assert_eq!(report.db_segment_count, report.jsonl_segment_count);
        assert_eq!(report.db_event_count, report.jsonl_event_count);
    }

    #[test]
    fn flush_run_chunk_matches_per_line_reference() {
        // The batched import (`flush_run_chunk`) must produce a byte-identical `runs`
        // table to the per-line path, including: an existing-row conflict (→ Overwrite),
        // a fresh insert, and an intra-chunk DUPLICATE id (the 2nd occurrence must see
        // the 1st's insert via the seen-map, exactly as the per-line SELECT would).
        let dir = tempdir().expect("tempdir");
        let mk = |name: &str| {
            let c = Connection::open(dir.path().join(name).display().to_string()).expect("open");
            ensure_schema(&c).expect("schema");
            c
        };
        let row = |id: &str, backend: &str| {
            serde_json::json!({
                "id": id, "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:01:00Z", "backend": backend,
                "input_path": "/a.wav", "normalized_wav_path": "/a16.wav",
                "request_json": "{}", "result_json": "{}", "warnings_json": "[]",
                "transcript": "hello", "replay_json": "{}", "acceleration_json": "{}"
            })
        };
        let sel_all = "SELECT id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json FROM runs ORDER BY id";
        let dump = |c: &Connection| -> Vec<Vec<String>> {
            c.query_with_params(sel_all, &[])
                .expect("dump")
                .iter()
                .map(|r| {
                    run_row_to_cols(r)
                        .iter()
                        .map(|v| value_to_string_sqlite(Some(v)))
                        .collect()
                })
                .collect()
        };

        let cref = mk("ref.sqlite3");
        let cbat = mk("bat.sqlite3");
        // Seed both DBs identically with x=cpu (via the shared insert path).
        for c in [&cref, &cbat] {
            apply_run_row(
                c,
                "x",
                &row("x", "cpu"),
                None,
                ConflictPolicy::Overwrite,
                &mut Vec::new(),
                &mut RunImportTracking::default(),
            )
            .expect("seed");
        }

        let import_rows = vec![
            ("x".to_string(), row("x", "gpu")), // conflicts with seed → overwrite
            ("y".to_string(), row("y", "cpu")), // new
            ("y".to_string(), row("y", "cpu")), // intra-chunk dup, identical → noop
            ("z".to_string(), row("z", "cpu")), // new
        ];

        // Reference: per-line apply_run_row with a per-line SELECT.
        {
            let mut tracking = RunImportTracking::default();
            let mut conflicts = Vec::new();
            for (id, r) in &import_rows {
                let existing_rows = cref
                    .query_with_params(
                        "SELECT id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json FROM runs WHERE id = ?1",
                        &[SqliteValue::Text(id.clone().into())],
                    )
                    .expect("per-line select");
                let existing = existing_rows.first().map(run_row_to_cols);
                apply_run_row(
                    &cref,
                    id,
                    r,
                    existing.as_deref(),
                    ConflictPolicy::Overwrite,
                    &mut conflicts,
                    &mut tracking,
                )
                .expect("per-line apply");
            }
        }

        // Batched: one prefetch + seen-map.
        {
            let mut chunk = import_rows.clone();
            let n = flush_run_chunk(
                &cbat,
                &mut chunk,
                ConflictPolicy::Overwrite,
                &mut Vec::new(),
                &mut RunImportTracking::default(),
            )
            .expect("flush");
            assert_eq!(n, 4, "every line counts once");
            assert!(chunk.is_empty(), "chunk drained");
        }

        let d = dump(&cbat);
        assert_eq!(d, dump(&cref), "batched import diverged from per-line");
        assert_eq!(
            d.len(),
            3,
            "x,y,z — the intra-chunk dup must not double-insert y"
        );
        assert_eq!(d[0][3], "gpu", "x overwritten to gpu");
    }

    #[test]
    fn flush_segment_chunk_matches_per_line_reference() {
        // Batched segment import (composite (run_id, idx) key) must produce a
        // byte-identical `segments` table to the per-line path: an existing-row
        // conflict (→ OverwriteStrict delete+insert), a fresh insert, and an
        // intra-chunk DUPLICATE (run_id, idx) that must see the earlier insert.
        let dir = tempdir().expect("tempdir");
        let mk = |name: &str| {
            let c = Connection::open(dir.path().join(name).display().to_string()).expect("open");
            ensure_schema(&c).expect("schema");
            c
        };
        let run_row = |id: &str| {
            serde_json::json!({
                "id": id, "started_at": "t0", "finished_at": "t1", "backend": "cpu",
                "input_path": "/a.wav", "normalized_wav_path": "/a16.wav",
                "request_json": "{}", "result_json": "{}", "warnings_json": "[]",
                "transcript": "hi", "replay_json": "{}", "acceleration_json": "{}"
            })
        };
        let seg = |run_id: &str, idx: i64, text: &str| {
            serde_json::json!({
                "run_id": run_id, "idx": idx, "start_sec": (idx as f64),
                "end_sec": (idx as f64) + 1.0, "speaker": serde_json::Value::Null,
                "text": text, "confidence": 0.9
            })
        };
        let sel = "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence FROM segments ORDER BY run_id, idx";
        let dump = |c: &Connection| -> Vec<Vec<String>> {
            c.query_with_params(sel, &[])
                .expect("dump")
                .iter()
                .map(|r| {
                    segment_row_to_cols(r)
                        .iter()
                        .map(|v| value_to_string_sqlite(Some(v)))
                        .collect()
                })
                .collect()
        };

        let cref = mk("segref.sqlite3");
        let cbat = mk("segbat.sqlite3");
        // Both DBs need run "r1" (FK check) and a seed segment (r1,0)="old".
        for c in [&cref, &cbat] {
            apply_run_row(
                c,
                "r1",
                &run_row("r1"),
                None,
                ConflictPolicy::OverwriteStrict,
                &mut Vec::new(),
                &mut RunImportTracking::default(),
            )
            .expect("seed run");
            apply_segment_row(
                c,
                "r1",
                0,
                "r1/0",
                &seg("r1", 0, "old"),
                None,
                ConflictPolicy::OverwriteStrict,
                &mut Vec::new(),
            )
            .expect("seed seg");
        }

        let import_rows = vec![
            ("r1".to_string(), 0i64, seg("r1", 0, "new")), // conflicts seed → overwrite-strict
            ("r1".to_string(), 1i64, seg("r1", 1, "b")),   // new
            ("r1".to_string(), 1i64, seg("r1", 1, "b")),   // intra-chunk dup, identical → noop
        ];

        // Reference: per-line.
        {
            let mut st = SegmentImportState::default();
            let mut cf = Vec::new();
            let (imp, ovr) = (HashSet::new(), HashSet::new());
            for (run_id, idx, row) in &import_rows {
                let key = format!("{run_id}/{idx}");
                record_segment_pre(
                    &cref,
                    run_id,
                    *idx,
                    &key,
                    ConflictPolicy::OverwriteStrict,
                    &imp,
                    &ovr,
                    &mut st,
                )
                .expect("pre");
                let existing_rows = cref
                    .query_with_params(
                        "SELECT run_id, idx, start_sec, end_sec, speaker, text, confidence FROM segments WHERE run_id = ?1 AND idx = ?2",
                        &[SqliteValue::Text(run_id.clone().into()), SqliteValue::Integer(*idx)],
                    )
                    .expect("select");
                let existing = existing_rows.first().map(segment_row_to_cols);
                apply_segment_row(
                    &cref,
                    run_id,
                    *idx,
                    &key,
                    row,
                    existing.as_deref(),
                    ConflictPolicy::OverwriteStrict,
                    &mut cf,
                )
                .expect("apply");
            }
        }

        // Batched.
        {
            let mut st = SegmentImportState::default();
            let (imp, ovr, skipped) = (HashSet::new(), HashSet::new(), HashSet::new());
            let mut chunk = import_rows.clone();
            let n = flush_segment_chunk(
                &cbat,
                &mut chunk,
                ConflictPolicy::OverwriteStrict,
                &mut Vec::new(),
                &mut st,
                &imp,
                &ovr,
                &skipped,
            )
            .expect("flush");
            assert_eq!(n, 3);
            assert!(chunk.is_empty());
        }

        let d = dump(&cbat);
        assert_eq!(
            d,
            dump(&cref),
            "batched segment import diverged from per-line"
        );
        assert_eq!(
            d.len(),
            2,
            "(r1,0)+(r1,1); intra-chunk dup must not double-insert"
        );
        assert_eq!(d[0][5], "new", "(r1,0) overwritten to 'new'");
    }

    #[test]
    fn flush_event_chunk_matches_per_line_reference() {
        // Batched event import (composite (run_id, seq) key) must produce a
        // byte-identical `events` table to the per-line path (conflict → OverwriteStrict,
        // fresh insert, intra-chunk duplicate (run_id, seq) → seen-map).
        let dir = tempdir().expect("tempdir");
        let mk = |name: &str| {
            let c = Connection::open(dir.path().join(name).display().to_string()).expect("open");
            ensure_schema(&c).expect("schema");
            c
        };
        let run_row = |id: &str| {
            serde_json::json!({
                "id": id, "started_at": "t0", "finished_at": "t1", "backend": "cpu",
                "input_path": "/a.wav", "normalized_wav_path": "/a16.wav",
                "request_json": "{}", "result_json": "{}", "warnings_json": "[]",
                "transcript": "hi", "replay_json": "{}", "acceleration_json": "{}"
            })
        };
        let ev = |run_id: &str, seq: i64, message: &str| {
            serde_json::json!({
                "run_id": run_id, "seq": seq, "ts_rfc3339": "2026-01-01T00:00:00Z",
                "stage": "decode", "code": "info", "message": message, "payload_json": "{}"
            })
        };
        let sel = "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json FROM events ORDER BY run_id, seq";
        let dump = |c: &Connection| -> Vec<Vec<String>> {
            c.query_with_params(sel, &[])
                .expect("dump")
                .iter()
                .map(|r| {
                    event_row_to_cols(r)
                        .iter()
                        .map(|v| value_to_string_sqlite(Some(v)))
                        .collect()
                })
                .collect()
        };

        let cref = mk("evref.sqlite3");
        let cbat = mk("evbat.sqlite3");
        for c in [&cref, &cbat] {
            apply_run_row(
                c,
                "r1",
                &run_row("r1"),
                None,
                ConflictPolicy::OverwriteStrict,
                &mut Vec::new(),
                &mut RunImportTracking::default(),
            )
            .expect("seed run");
            apply_event_row(
                c,
                "r1",
                0,
                "r1/0",
                &ev("r1", 0, "old"),
                None,
                ConflictPolicy::OverwriteStrict,
                &mut Vec::new(),
            )
            .expect("seed ev");
        }

        let import_rows = vec![
            ("r1".to_string(), 0i64, ev("r1", 0, "new")), // conflicts seed → overwrite-strict
            ("r1".to_string(), 1i64, ev("r1", 1, "b")),   // new
            ("r1".to_string(), 1i64, ev("r1", 1, "b")),   // intra-chunk dup, identical → noop
        ];

        {
            let mut st = EventImportState::default();
            let mut cf = Vec::new();
            let (imp, ovr) = (HashSet::new(), HashSet::new());
            for (run_id, seq, row) in &import_rows {
                let key = format!("{run_id}/{seq}");
                record_event_pre(
                    &cref,
                    run_id,
                    *seq,
                    &key,
                    ConflictPolicy::OverwriteStrict,
                    &imp,
                    &ovr,
                    &mut st,
                )
                .expect("pre");
                let existing_rows = cref
                    .query_with_params(
                        "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json FROM events WHERE run_id = ?1 AND seq = ?2",
                        &[SqliteValue::Text(run_id.clone().into()), SqliteValue::Integer(*seq)],
                    )
                    .expect("select");
                let existing = existing_rows.first().map(event_row_to_cols);
                apply_event_row(
                    &cref,
                    run_id,
                    *seq,
                    &key,
                    row,
                    existing.as_deref(),
                    ConflictPolicy::OverwriteStrict,
                    &mut cf,
                )
                .expect("apply");
            }
        }

        {
            let mut st = EventImportState::default();
            let (imp, ovr, skipped) = (HashSet::new(), HashSet::new(), HashSet::new());
            let mut chunk = import_rows.clone();
            let n = flush_event_chunk(
                &cbat,
                &mut chunk,
                ConflictPolicy::OverwriteStrict,
                &mut Vec::new(),
                &mut st,
                &imp,
                &ovr,
                &skipped,
            )
            .expect("flush");
            assert_eq!(n, 3);
            assert!(chunk.is_empty());
        }

        let d = dump(&cbat);
        assert_eq!(
            d,
            dump(&cref),
            "batched event import diverged from per-line"
        );
        assert_eq!(
            d.len(),
            2,
            "(r1,0)+(r1,1); intra-chunk dup must not double-insert"
        );
        assert_eq!(d[0][5], "new", "(r1,0) message overwritten to 'new'");
    }

    #[test]
    fn validate_sync_empty_db_and_empty_export_is_valid() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let _store = RunStore::open(&db_path).expect("store open");
        export(&db_path, &export_dir, &state_root).expect("export empty");

        let report = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(report.is_valid, "empty db + empty export should be valid");
        assert_eq!(report.db_run_count, 0);
        assert_eq!(report.jsonl_run_count, 0);
    }

    #[test]
    fn validate_sync_detects_run_missing_from_jsonl() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        // Export empty db first
        let store = RunStore::open(&db_path).expect("store open");
        export(&db_path, &export_dir, &state_root).expect("export empty");

        // Then add a run to the db (without re-exporting)
        store
            .persist_report(&fixture_report("run-extra", &db_path))
            .expect("persist");
        drop(store);

        let report = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(!report.is_valid, "should not be valid with extra db run");
        assert_eq!(report.db_run_count, 1);
        assert_eq!(report.jsonl_run_count, 0);
        assert_eq!(report.missing_from_jsonl, vec!["run-extra"]);
        assert!(report.missing_from_db.is_empty());
    }

    #[test]
    fn validate_sync_detects_run_missing_from_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("full.sqlite3");
        let empty_db_path = dir.path().join("empty.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        // Export db with a run
        let store = RunStore::open(&db_path).expect("store open");
        store
            .persist_report(&fixture_report("run-orphan", &db_path))
            .expect("persist");
        drop(store);

        export(&db_path, &export_dir, &state_root).expect("export with run");

        // Create a separate empty database, then validate it against
        // the export that has the run.
        let _empty_store = RunStore::open(&empty_db_path).expect("open empty");

        let report = validate_sync(&empty_db_path, &export_dir).expect("validate");
        assert!(
            !report.is_valid,
            "should not be valid with orphaned jsonl run"
        );
        assert_eq!(report.db_run_count, 0);
        assert_eq!(report.jsonl_run_count, 1);
        assert!(report.missing_from_jsonl.is_empty());
        assert_eq!(report.missing_from_db, vec!["run-orphan"]);
    }

    #[test]
    fn validate_sync_multiple_runs_all_matching() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multi.sqlite3");
        let export_dir = dir.path().join("export");
        let state_root = dir.path().join("state");

        let store = RunStore::open(&db_path).expect("store open");
        for i in 0..5 {
            let mut report = fixture_report(&format!("run-{i}"), &db_path);
            report.started_at_rfc3339 = format!("2026-01-0{i}T00:00:00Z");
            store.persist_report(&report).expect("persist");
        }
        drop(store);

        export(&db_path, &export_dir, &state_root).expect("export");

        let report = validate_sync(&db_path, &export_dir).expect("validate");
        assert!(report.is_valid, "5 matching runs should validate");
        assert_eq!(report.db_run_count, 5);
        assert_eq!(report.jsonl_run_count, 5);
    }
}
