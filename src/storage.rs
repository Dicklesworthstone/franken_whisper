use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use fsqlite::{Connection, FrankenError, Row};
use fsqlite_types::value::SqliteValue;
use serde::de::DeserializeOwned;

use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, ReplayEnvelope, RunEvent, RunReport, RunSummary, StoredRunDetails,
    TranscriptionResult,
};

/// Drive `future` to completion on this thread's runtime.
///
/// The runtime is built once per thread rather than once per statement: a
/// single `persist_report` issues many statements, and the concurrency tests
/// drive several threads at once.
///
/// The future is boxed before it is polled. `fsqlite`'s statement futures nest
/// deeply (statement dispatch → DML → triggers → nested statement execution —
/// deep enough that `fsqlite` raises its own `recursion_limit` to type-check
/// them), so both their state machines and the synchronous poll chain that
/// drives them are large. Boxing moves the outermost state machine off the
/// stack; it measurably reduces the depth but does **not** on its own make a
/// debug build fit in libtest's default 8 MiB thread stack — that needs the
/// `RUST_MIN_STACK` set in `.cargo/config.toml`, which is where the full
/// rationale lives. One allocation per statement is negligible against the
/// statement itself.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    thread_local! {
        static RUNTIME: asupersync::runtime::Runtime =
            asupersync::runtime::RuntimeBuilder::current_thread()
                .build()
                .expect("build franken_whisper storage runtime");
    }
    RUNTIME.with(|runtime| runtime.block_on(Box::pin(future)))
}

/// Synchronous facade over the now-`async` [`fsqlite::Connection`].
///
/// The frankensqlite async migration turned every statement entry point —
/// `open`, `query`, `execute`, and the `*_with_params` variants — into
/// `async fn`. Every consumer of the run store (the CLI in `main.rs`, robot
/// mode, and the JSONL sync) is synchronous and is never itself driven by an
/// executor, so propagating `async` across the whole storage surface would buy
/// no concurrency and would force a runtime into `main`. Instead the bridge
/// lives here, mirroring the `block_on` bridge frankensqlite added for its own
/// synchronous harnesses.
///
/// Method names, argument types, and error types are deliberately identical to
/// the `Connection` inherent methods they wrap, so the call sites in this
/// module and in [`crate::sync`] keep their existing shape and error mapping.
pub struct BlockingConnection {
    inner: Connection,
}

impl std::fmt::Debug for BlockingConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockingConnection").finish_non_exhaustive()
    }
}

impl BlockingConnection {
    pub fn open(path: impl Into<String>) -> Result<Self, FrankenError> {
        block_on(Connection::open(path)).map(|inner| Self { inner })
    }

    pub fn query(&self, sql: &str) -> Result<Vec<Row>, FrankenError> {
        block_on(self.inner.query(sql))
    }

    pub fn query_with_params(
        &self,
        sql: &str,
        params: &[SqliteValue],
    ) -> Result<Vec<Row>, FrankenError> {
        block_on(self.inner.query_with_params(sql, params))
    }

    pub fn execute(&self, sql: &str) -> Result<usize, FrankenError> {
        block_on(self.inner.execute(sql))
    }

    pub fn execute_with_params(
        &self,
        sql: &str,
        params: &[SqliteValue],
    ) -> Result<usize, FrankenError> {
        block_on(self.inner.execute_with_params(sql, params))
    }

    pub fn execute_with_params_skip_statement_savepoint_in_explicit_txn(
        &self,
        sql: &str,
        params: &[SqliteValue],
    ) -> Result<usize, FrankenError> {
        block_on(
            self.inner
                .execute_with_params_skip_statement_savepoint_in_explicit_txn(sql, params),
        )
    }
}

pub struct RunStore {
    connection: BlockingConnection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RunsMigrationSnapshotRow {
    id: String,
    started_at: String,
    finished_at: String,
    backend: String,
    input_path: String,
    normalized_wav_path: String,
    request_json: String,
    result_json: String,
    warnings_json: String,
    transcript: String,
    replay_json: String,
    acceleration_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TableColumn {
    name: String,
    declared_type: String,
    not_null: bool,
    default_value: Option<String>,
    is_primary_key: bool,
}

const PERSIST_BUSY_RETRY_ATTEMPTS: usize = 8;
const PERSIST_BUSY_BASE_BACKOFF_MS: u64 = 5;
static PERSIST_SAVEPOINT_COUNTER: AtomicU64 = AtomicU64::new(1);

impl std::fmt::Debug for RunStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunStore").finish_non_exhaustive()
    }
}

impl RunStore {
    pub fn open(db_path: &Path) -> FwResult<Self> {
        if let Some(parent) = db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        for attempt in 0..=PERSIST_BUSY_RETRY_ATTEMPTS {
            let connection_result = BlockingConnection::open(db_path.display().to_string());
            match connection_result {
                Ok(connection) => {
                    let store = Self { connection };
                    match store.initialize_schema() {
                        Ok(()) => return Ok(store),
                        Err(error)
                            if is_busy_storage_error(&error)
                                && attempt < PERSIST_BUSY_RETRY_ATTEMPTS =>
                        {
                            let delay_ms = PERSIST_BUSY_BASE_BACKOFF_MS * (attempt as u64 + 1);
                            std::thread::sleep(Duration::from_millis(delay_ms));
                        }
                        Err(error) => return Err(error),
                    }
                }
                Err(error) => {
                    let fw_err = FwError::Storage(error.to_string());
                    if is_busy_storage_error(&fw_err) && attempt < PERSIST_BUSY_RETRY_ATTEMPTS {
                        let delay_ms = PERSIST_BUSY_BASE_BACKOFF_MS * (attempt as u64 + 1);
                        std::thread::sleep(Duration::from_millis(delay_ms));
                    } else {
                        return Err(fw_err);
                    }
                }
            }
        }
        Err(FwError::Storage(
            "open retry loop exhausted unexpectedly".to_owned(),
        ))
    }

    pub fn persist_report(&self, report: &RunReport) -> FwResult<()> {
        self.persist_report_cancellable(report, None)
    }

    pub(crate) fn persist_report_cancellable(
        &self,
        report: &RunReport,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<()> {
        for attempt in 0..=PERSIST_BUSY_RETRY_ATTEMPTS {
            if let Some(tok) = token {
                tok.checkpoint()?;
            }

            match self.persist_report_once(report, token) {
                Ok(()) => return Ok(()),
                Err(error)
                    if is_busy_storage_error(&error) && attempt < PERSIST_BUSY_RETRY_ATTEMPTS =>
                {
                    let delay_ms = PERSIST_BUSY_BASE_BACKOFF_MS * (attempt as u64 + 1);
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                Err(error) => return Err(error),
            }
        }

        Err(FwError::Storage(
            "persist retry loop exhausted unexpectedly".to_owned(),
        ))
    }

    fn persist_report_once(
        &self,
        report: &RunReport,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<()> {
        tracing::debug!(run_id = %report.run_id, stage = "persist", "Entering persist_report");
        let savepoint_name = next_persist_savepoint_name();
        let begin_sql = format!("SAVEPOINT {savepoint_name};");
        self.connection
            .execute(&begin_sql)
            .map_err(|error| FwError::Storage(error.to_string()))?;

        let result = self.persist_report_inner(report, token);
        match result {
            Ok(()) => {
                // Checkpoint before commit — cancellation triggers rollback.
                if let Some(tok) = token
                    && let Err(err) = tok.checkpoint()
                {
                    let _ = rollback_savepoint(&self.connection, &savepoint_name);
                    return Err(err);
                }
                let release_sql = format!("RELEASE SAVEPOINT {savepoint_name};");
                self.connection
                    .execute(&release_sql)
                    .map_err(|error| FwError::Storage(error.to_string()))?;
                Ok(())
            }
            Err(error) => {
                let _ = rollback_savepoint(&self.connection, &savepoint_name);
                Err(error)
            }
        }
    }

    pub fn list_recent_runs(&self, limit: usize) -> FwResult<Vec<RunSummary>> {
        self.list_recent_runs_cancellable(limit, None)
    }

    pub(crate) fn list_recent_runs_cancellable(
        &self,
        limit: usize,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<Vec<RunSummary>> {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }

        // Project only the 140-char preview in SQL (fsqlite's `ColumnSubstrPrefix`
        // fast path) so a large transcript is never materialized or transferred
        // just to be truncated. `substr(transcript, 1, 140)` returns the first 140
        // characters, byte-identical to the historical `.chars().take(140)` for
        // TEXT (and coerced numeric) values. A BLOB value renders differently under
        // projection (`<blob:<=140>` vs the full `<blob:N>`), so if the projected
        // result contains any BLOB transcript we re-run with the full column and
        // fall back to the historical rendering — byte-exact for every DB state.
        let build_sql = |expr: &str| {
            if limit > 0 {
                format!(
                    "SELECT id, started_at, finished_at, backend, {expr} FROM runs \
                     ORDER BY started_at DESC, id DESC LIMIT {limit}"
                )
            } else {
                format!(
                    "SELECT id, started_at, finished_at, backend, {expr} FROM runs \
                     ORDER BY started_at DESC, id DESC"
                )
            }
        };

        let rows = self
            .connection
            .query(&build_sql("substr(transcript, 1, 140)"))
            .map_err(|error| FwError::Storage(error.to_string()))?;

        // BLOB transcripts render by byte length, which the projection would
        // shorten; re-run with the full column so the preview matches the
        // historical `<blob:N>` marker exactly.
        let rows = if rows
            .iter()
            .any(|row| matches!(row.get(4), Some(SqliteValue::Blob(_))))
        {
            self.connection
                .query(&build_sql("transcript"))
                .map_err(|error| FwError::Storage(error.to_string()))?
        } else {
            rows
        };

        rows.into_iter()
            .map(|row| {
                let run_id = value_to_string(row.get(0));
                let started = value_to_string(row.get(1));
                let finished = value_to_string(row.get(2));
                let backend = parse_backend(&value_to_string(row.get(3)));
                let transcript = value_to_string(row.get(4));
                let preview = transcript.chars().take(140).collect::<String>();

                Ok(RunSummary {
                    run_id,
                    started_at_rfc3339: started,
                    finished_at_rfc3339: finished,
                    backend,
                    transcript_preview: preview,
                })
            })
            .collect()
    }

    pub fn load_latest_run_details(&self) -> FwResult<Option<StoredRunDetails>> {
        self.load_latest_run_details_cancellable(None)
    }

    pub(crate) fn load_latest_run_details_cancellable(
        &self,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<Option<StoredRunDetails>> {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }

        // Use SQL ORDER BY and LIMIT 1 for efficient latest-run lookup.
        let rows = self
            .connection
            .query("SELECT id FROM runs ORDER BY started_at DESC, id DESC LIMIT 1")
            .map_err(|error| FwError::Storage(error.to_string()))?;

        if let Some(row) = rows.into_iter().next() {
            let run_id = value_to_string(row.get(0));
            self.load_run_details_cancellable(&run_id, token)
        } else {
            Ok(None)
        }
    }

    pub fn load_run_details(&self, run_id: &str) -> FwResult<Option<StoredRunDetails>> {
        self.load_run_details_cancellable(run_id, None)
    }

    pub(crate) fn load_run_details_cancellable(
        &self,
        run_id: &str,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<Option<StoredRunDetails>> {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }

        let replay_expr = if self.table_has_column("runs", "replay_json")? {
            "replay_json"
        } else {
            "'{}' AS replay_json"
        };
        let acceleration_expr = if self.table_has_column("runs", "acceleration_json")? {
            "acceleration_json"
        } else {
            "'{}' AS acceleration_json"
        };
        let run_sql = format!(
            "SELECT id, started_at, finished_at, backend, result_json, warnings_json, transcript, {replay_expr}, {acceleration_expr} \
             FROM runs WHERE id = ?1 LIMIT 1"
        );
        let rows = self
            .connection
            .query_with_params(&run_sql, &[SqliteValue::Text(run_id.to_owned().into())])
            .map_err(|error| FwError::Storage(error.to_string()))?;

        let Some(row) = rows.first() else {
            return Ok(None);
        };

        let run_id = value_to_string(row.get(0));
        let started_at_rfc3339 = value_to_string(row.get(1));
        let finished_at_rfc3339 = value_to_string(row.get(2));
        let backend = parse_backend(&value_to_string(row.get(3)));
        let result_json = value_to_string(row.get(4));
        let warnings_json = value_to_string(row.get(5));
        let transcript_fallback = value_to_string(row.get(6));
        let replay_json = value_to_string(row.get(7));
        let acceleration_json = value_to_string(row.get(8));

        let result: TranscriptionResult = serde_json::from_str(&result_json).map_err(|error| {
            FwError::Storage(format!("invalid result_json for run {run_id}: {error}"))
        })?;

        let warnings =
            parse_required_json_field::<Vec<String>>("warnings_json", &run_id, &warnings_json)?;
        let replay =
            parse_required_json_field::<ReplayEnvelope>("replay_json", &run_id, &replay_json)?;
        let acceleration_override = parse_optional_acceleration_json(&run_id, &acceleration_json)?;
        let transcript = if result.transcript.trim().is_empty() {
            transcript_fallback
        } else {
            result.transcript.clone()
        };

        // Checkpoint before loading events (second major query).
        if let Some(tok) = token {
            tok.checkpoint()?;
        }

        let event_rows = self
            .connection
            .query_with_params(
                "SELECT seq, ts_rfc3339, stage, code, message, payload_json FROM events WHERE run_id = ?1 ORDER BY seq ASC",
                &[SqliteValue::Text(run_id.clone().into())],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;

        let events = event_rows
            .into_iter()
            .map(|event_row| {
                let payload_json = value_to_string(event_row.get(5));
                let payload = serde_json::from_str(&payload_json).map_err(|error| {
                    FwError::Storage(format!(
                        "invalid event payload for run {} seq {}: {}",
                        run_id,
                        value_to_string(event_row.get(0)),
                        error
                    ))
                })?;

                let seq_text = value_to_string(event_row.get(0));
                let seq = seq_text.parse::<u64>().map_err(|error| {
                    FwError::Storage(format!(
                        "invalid event sequence `{}` for run {}: {}",
                        seq_text, run_id, error
                    ))
                })?;

                Ok(RunEvent {
                    seq,
                    ts_rfc3339: value_to_string(event_row.get(1)),
                    stage: value_to_string(event_row.get(2)),
                    code: value_to_string(event_row.get(3)),
                    message: value_to_string(event_row.get(4)),
                    payload,
                })
            })
            .collect::<FwResult<Vec<_>>>()?;

        Ok(Some(StoredRunDetails {
            run_id,
            started_at_rfc3339,
            finished_at_rfc3339,
            backend,
            transcript,
            segments: result.segments,
            events,
            warnings,
            acceleration: acceleration_override.or(result.acceleration),
            replay,
        }))
    }

    /// Load full details for many runs with two batched `WHERE id/run_id IN (…)`
    /// queries instead of the per-run N+1 (`load_run_details` × N, each 2 queries +
    /// 2 schema scans). Returns details for the run_ids that exist, in input order.
    /// Output is byte-identical to per-run `load_run_details` (asserted by
    /// `load_run_details_batch_matches_per_run`). `FW_STORAGE_BATCH_HISTORY=0` falls
    /// back to the per-run path (kill-switch + A/B arm).
    pub fn load_run_details_batch(&self, run_ids: &[String]) -> FwResult<Vec<StoredRunDetails>> {
        if run_ids.is_empty() {
            return Ok(Vec::new());
        }
        if !batch_history_enabled() {
            let mut out = Vec::with_capacity(run_ids.len());
            for id in run_ids {
                if let Some(details) = self.load_run_details(id)? {
                    out.push(details);
                }
            }
            return Ok(out);
        }

        // Optional-column exprs computed ONCE (not once per run) — see load_run_details.
        let replay_expr = if self.table_has_column("runs", "replay_json")? {
            "replay_json"
        } else {
            "'{}' AS replay_json"
        };
        let acceleration_expr = if self.table_has_column("runs", "acceleration_json")? {
            "acceleration_json"
        } else {
            "'{}' AS acceleration_json"
        };
        let placeholders = (1..=run_ids.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let params: Vec<SqliteValue> = run_ids
            .iter()
            .map(|id| SqliteValue::Text(id.clone().into()))
            .collect();

        let run_sql = format!(
            "SELECT id, started_at, finished_at, backend, result_json, warnings_json, transcript, {replay_expr}, {acceleration_expr} \
             FROM runs WHERE id IN ({placeholders})"
        );
        let run_rows = self
            .connection
            .query_with_params(&run_sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?;
        let mut run_by_id: std::collections::HashMap<String, fsqlite::Row> =
            std::collections::HashMap::with_capacity(run_rows.len());
        for row in run_rows {
            run_by_id.insert(value_to_string(row.get(0)), row);
        }

        let evt_sql = format!(
            "SELECT run_id, seq, ts_rfc3339, stage, code, message, payload_json \
             FROM events WHERE run_id IN ({placeholders}) ORDER BY run_id ASC, seq ASC"
        );
        let evt_rows = self
            .connection
            .query_with_params(&evt_sql, &params)
            .map_err(|error| FwError::Storage(error.to_string()))?;
        let mut events_by_run: std::collections::HashMap<String, Vec<fsqlite::Row>> =
            std::collections::HashMap::new();
        for row in evt_rows {
            events_by_run
                .entry(value_to_string(row.get(0)))
                .or_default()
                .push(row);
        }

        let mut out = Vec::with_capacity(run_ids.len());
        for run_id in run_ids {
            let Some(run_row) = run_by_id.get(run_id) else {
                continue;
            };
            let events = events_by_run.remove(run_id).unwrap_or_default();
            out.push(assemble_run_details_batched(run_row, &events)?);
        }
        Ok(out)
    }

    /// Current schema version. Bump when adding migrations.
    pub const SCHEMA_VERSION: u32 = 3;

    fn initialize_schema(&self) -> FwResult<()> {
        // Enable WAL mode for concurrent read/write support.
        // Use query() since PRAGMA returns result rows that execute() may not handle.
        let _ = self.connection.query("PRAGMA journal_mode=WAL;");
        let _ = self.connection.query("PRAGMA busy_timeout=5000;");

        // Create base tables (v1 schema).
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

CREATE TABLE IF NOT EXISTS _meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

        self.connection
            .execute(sql)
            .map_err(|error| FwError::Storage(error.to_string()))?;

        self.run_migrations()?;
        Ok(())
    }

    /// Read the current schema version from _meta, or 0 if not set.
    fn current_schema_version(&self) -> FwResult<u32> {
        let rows = self
            .connection
            .query("SELECT value FROM _meta WHERE key = 'schema_version';")
            .map_err(|error| {
                let text = error.to_string();
                if text.contains("no such table: _meta") {
                    FwError::Storage("__missing_meta__".to_owned())
                } else {
                    FwError::Storage(text)
                }
            });
        let rows = match rows {
            Ok(rows) => rows,
            Err(FwError::Storage(marker)) if marker == "__missing_meta__" => return Ok(0),
            Err(error) => return Err(error),
        };
        match rows.first() {
            Some(row) => {
                let v = value_to_string(row.get(0));
                v.parse::<u32>()
                    .map_err(|_| FwError::Storage(format!("invalid schema_version in _meta: {v}")))
            }
            None => Ok(0),
        }
    }

    fn set_schema_version(&self, version: u32) -> FwResult<()> {
        self.connection
            .execute_with_params(
                "INSERT OR REPLACE INTO _meta (key, value) VALUES ('schema_version', ?1);",
                &[text_value(version.to_string())],
            )
            .map_err(|error| FwError::Storage(error.to_string()))?;
        Ok(())
    }

    /// Run forward migrations from the current version to SCHEMA_VERSION.
    fn run_migrations(&self) -> FwResult<()> {
        let mut current = self.current_schema_version()?;

        if current > Self::SCHEMA_VERSION {
            return Err(FwError::Storage(format!(
                "DB schema version {current} is newer than supported version {}; \
                 upgrade franken_whisper to open this database",
                Self::SCHEMA_VERSION
            )));
        }

        if current == Self::SCHEMA_VERSION {
            return Ok(());
        }

        // Fresh databases are created with the current table shape in
        // initialize_schema(). Detect that shape explicitly before shortcutting:
        // legacy v1 DBs can also report version=0 when `_meta` is absent and must
        // still flow through v2 migrations to add new columns.
        if current == 0 {
            let has_replay = self.table_has_column("runs", "replay_json")?;
            let has_acceleration = self.table_has_column("runs", "acceleration_json")?;
            if has_replay && has_acceleration {
                self.set_schema_version(2)?;
                current = 2;
            }
        }

        tracing::info!(
            current_version = current,
            target_version = Self::SCHEMA_VERSION,
            "Running schema migrations"
        );

        while current < Self::SCHEMA_VERSION {
            let next = current + 1;
            tracing::info!(from = current, to = next, "Migrating schema");
            self.apply_migration(next).map_err(|error| {
                FwError::Storage(format!("migration to v{next} failed: {error}"))
            })?;
            self.set_schema_version(next)?;
            tracing::info!(version = next, "Migration complete");
            current = next;
        }

        Ok(())
    }

    /// Apply a single migration step. Each migration only adds (never drops).
    fn apply_migration(&self, version: u32) -> FwResult<()> {
        match version {
            1 => {
                // v1: Base schema — already created by initialize_schema.
                // Just set version marker.
                Ok(())
            }
            2 => {
                // v2: Legacy-safe migration for runs table replay/acceleration
                // columns using snapshot -> rebuild -> swap with rollback safety.
                self.migrate_legacy_runs_schema_v2()?;
                Ok(())
            }
            3 => {
                // v3: Add indexes for common query patterns (bd-3i1.5).
                self.connection
                    .execute("CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);")
                    .map_err(|error| FwError::Storage(error.to_string()))?;
                self.connection
                    .execute("CREATE INDEX IF NOT EXISTS idx_runs_backend ON runs(backend);")
                    .map_err(|error| FwError::Storage(error.to_string()))?;
                self.connection
                    .execute("CREATE INDEX IF NOT EXISTS idx_segments_run_id ON segments(run_id);")
                    .map_err(|error| FwError::Storage(error.to_string()))?;
                self.connection
                    .execute(
                        "CREATE INDEX IF NOT EXISTS idx_events_run_id_stage ON events(run_id, stage);",
                    )
                    .map_err(|error| FwError::Storage(error.to_string()))?;
                Ok(())
            }
            _ => Err(FwError::Storage(format!(
                "unknown migration version: {version}"
            ))),
        }
    }

    /// Add a column to a table if it doesn't already exist.
    fn ensure_column_exists(&self, table: &str, column: &str, column_def: &str) -> FwResult<()> {
        let columns = self.table_columns(table)?;
        let exists = columns.iter().any(|entry| entry.name == column);
        if exists {
            return Ok(());
        }

        // fsqlite DDL mutators are more reliable in DELETE mode than WAL mode.
        // Temporarily switch journal mode for schema change, then restore.
        let current_mode = self
            .connection
            .query("PRAGMA journal_mode;")
            .map_err(|error| FwError::Storage(error.to_string()))?
            .first()
            .map(|row| value_to_string(row.get(0)))
            .unwrap_or_default();
        let switched_from_wal = current_mode.eq_ignore_ascii_case("wal");
        let mut journal_mode_restored = !switched_from_wal;
        if switched_from_wal {
            self.connection
                .query("PRAGMA journal_mode='delete';")
                .map_err(|error| FwError::Storage(error.to_string()))?;
        }

        let add_sql = format!(
            "ALTER TABLE {} ADD COLUMN {} {};",
            sql_ident(table),
            sql_ident(column),
            column_def
        );
        let mut fallback_rebuild_columns: Option<Vec<TableColumn>> = None;
        let add_result = {
            let mut result = Ok(());
            for attempt in 0..=PERSIST_BUSY_RETRY_ATTEMPTS {
                match self.connection.execute(&add_sql) {
                    Ok(_) => {
                        result = Ok(());
                        break;
                    }
                    Err(error) => {
                        let wrapped = FwError::Storage(error.to_string());
                        if is_busy_storage_error(&wrapped) && attempt < PERSIST_BUSY_RETRY_ATTEMPTS
                        {
                            let delay_ms = PERSIST_BUSY_BASE_BACKOFF_MS * (attempt as u64 + 1);
                            std::thread::sleep(Duration::from_millis(delay_ms));
                            continue;
                        }

                        let error_text = wrapped.to_string();
                        if error_text.contains("sqlite_master entry not found")
                            && fallback_rebuild_columns.is_none()
                        {
                            fallback_rebuild_columns = Some(columns.clone());
                            result = Ok(());
                            break;
                        }

                        result = Err(FwError::Storage(format!(
                            "failed to add column `{column}` on table `{table}` via ALTER TABLE: {wrapped}"
                        )));
                        break;
                    }
                }
            }
            result
        };
        let add_result = match (add_result, fallback_rebuild_columns) {
            (Ok(()), Some(existing_columns)) => {
                if switched_from_wal {
                    self.connection
                        .query("PRAGMA journal_mode='wal';")
                        .map_err(|error| FwError::Storage(error.to_string()))?;
                    journal_mode_restored = true;
                }
                self.recreate_table_with_added_column(table, &existing_columns, column, column_def)
            }
            (result, _) => result,
        };

        if switched_from_wal && !journal_mode_restored {
            let restore = self
                .connection
                .query("PRAGMA journal_mode='wal';")
                .map_err(|error| FwError::Storage(error.to_string()));
            if let Err(restore_error) = restore {
                return match add_result {
                    Ok(_) => Err(FwError::Storage(format!(
                        "column `{column}` added on table `{table}`, but failed to restore WAL mode: {restore_error}"
                    ))),
                    Err(add_error) => Err(FwError::Storage(format!(
                        "{add_error}; failed to restore WAL mode: {restore_error}"
                    ))),
                };
            }
        }

        add_result?;
        if !self.table_has_column(table, column)? {
            return Err(FwError::Storage(format!(
                "column `{column}` still missing on table `{table}` after migration"
            )));
        }
        Ok(())
    }

    fn table_columns(&self, table: &str) -> FwResult<Vec<TableColumn>> {
        let rows = self
            .connection
            .query(&format!("PRAGMA table_info({});", sql_ident(table)))
            .map_err(|error| FwError::Storage(error.to_string()))?;

        Ok(rows
            .iter()
            .map(|row| TableColumn {
                name: value_to_string(row.get(1)),
                declared_type: value_to_string(row.get(2)),
                not_null: value_to_i64(row.get(3)) != 0,
                default_value: match row.get(4) {
                    Some(SqliteValue::Text(value)) if !value.is_empty() => Some(value.to_string()),
                    Some(value) => {
                        let text = value_to_string(Some(value));
                        (!text.is_empty()).then_some(text)
                    }
                    None => None,
                },
                is_primary_key: value_to_i64(row.get(5)) != 0,
            })
            .collect())
    }

    fn table_has_column(&self, table: &str, column: &str) -> FwResult<bool> {
        let columns = self.table_columns(table)?;
        Ok(columns.iter().any(|entry| entry.name == column))
    }

    fn table_index_definitions(&self, table: &str) -> FwResult<Vec<(String, String)>> {
        let rows = self
            .connection
            .query_with_params(
                "SELECT name, sql FROM sqlite_master \
                 WHERE type = 'index' AND tbl_name = ?1 AND sql IS NOT NULL \
                 ORDER BY name ASC;",
                &[text_value(table.to_owned())],
            )
            .map_err(|error| {
                FwError::Storage(format!("index introspection for `{table}` failed: {error}"))
            })?;

        Ok(rows
            .iter()
            .filter_map(|row| {
                let name = value_to_string(row.get(0));
                match row.get(1) {
                    Some(SqliteValue::Text(sql)) if !name.is_empty() && !sql.is_empty() => {
                        Some((name, sql.to_string()))
                    }
                    _ => None,
                }
            })
            .collect())
    }

    fn load_runs_snapshot_rows(
        &self,
        table: &str,
        has_replay_column: bool,
        has_acceleration_column: bool,
    ) -> FwResult<Vec<RunsMigrationSnapshotRow>> {
        let table_ident = sql_ident(table);
        let replay_expr = if has_replay_column {
            "replay_json"
        } else {
            "'{}'"
        };
        let acceleration_expr = if has_acceleration_column {
            "acceleration_json"
        } else {
            "'{}'"
        };
        let sql = format!(
            "SELECT id, started_at, finished_at, backend, input_path, normalized_wav_path, \
             request_json, result_json, warnings_json, transcript, {replay_expr} AS replay_json, \
             {acceleration_expr} AS acceleration_json \
             FROM {table_ident} ORDER BY rowid ASC;"
        );
        let rows = self
            .connection
            .query(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?;

        Ok(rows
            .iter()
            .map(|row| RunsMigrationSnapshotRow {
                id: value_to_string(row.get(0)),
                started_at: value_to_string(row.get(1)),
                finished_at: value_to_string(row.get(2)),
                backend: value_to_string(row.get(3)),
                input_path: value_to_string(row.get(4)),
                normalized_wav_path: value_to_string(row.get(5)),
                request_json: value_to_string(row.get(6)),
                result_json: value_to_string(row.get(7)),
                warnings_json: value_to_string(row.get(8)),
                transcript: value_to_string(row.get(9)),
                replay_json: value_to_string(row.get(10)),
                acceleration_json: value_to_string(row.get(11)),
            })
            .collect())
    }

    fn migrate_legacy_runs_schema_v2(&self) -> FwResult<()> {
        self.migrate_legacy_runs_schema_v2_inner(false)
    }

    #[cfg(test)]
    fn migrate_legacy_runs_schema_v2_with_failpoint(&self) -> FwResult<()> {
        self.migrate_legacy_runs_schema_v2_inner(true)
    }

    fn migrate_legacy_runs_schema_v2_inner(&self, failpoint_after_swap: bool) -> FwResult<()> {
        let has_replay = self.table_has_column("runs", "replay_json")?;
        let has_acceleration = self.table_has_column("runs", "acceleration_json")?;
        if has_replay && has_acceleration {
            return Ok(());
        }

        let source_snapshot = self.load_runs_snapshot_rows("runs", has_replay, has_acceleration)?;
        let expected_rows = source_snapshot.len();
        let index_defs = self.table_index_definitions("runs")?;
        let rebuilt_table_name = "__fw_runs_v2_rebuild";
        let rebuilt_ident = sql_ident(rebuilt_table_name);

        let mut transaction_open = false;
        let migration_result = (|| -> FwResult<()> {
            self.connection
                .execute("BEGIN IMMEDIATE;")
                .map_err(|error| FwError::Storage(error.to_string()))?;
            transaction_open = true;

            self.connection
                .execute(&format!("DROP TABLE IF EXISTS {rebuilt_ident};"))
                .map_err(|error| FwError::Storage(error.to_string()))?;
            self.connection
                .execute(&format!(
                    "CREATE TABLE {rebuilt_ident} (\
                        id TEXT PRIMARY KEY,\
                        started_at TEXT NOT NULL,\
                        finished_at TEXT NOT NULL,\
                        backend TEXT NOT NULL,\
                        input_path TEXT NOT NULL,\
                        normalized_wav_path TEXT NOT NULL,\
                        request_json TEXT NOT NULL,\
                        result_json TEXT NOT NULL,\
                        warnings_json TEXT NOT NULL,\
                        transcript TEXT NOT NULL,\
                        replay_json TEXT NOT NULL DEFAULT '{{}}',\
                        acceleration_json TEXT NOT NULL DEFAULT '{{}}'\
                    );"
                ))
                .map_err(|error| FwError::Storage(error.to_string()))?;

            for row in &source_snapshot {
                self.connection
                    .execute_with_params(
                        &format!(
                            "INSERT INTO {rebuilt_ident} (\
                                id, started_at, finished_at, backend, input_path, normalized_wav_path, \
                                request_json, result_json, warnings_json, transcript, replay_json, acceleration_json\
                            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"
                        ),
                        &[
                            text_value(row.id.clone()),
                            text_value(row.started_at.clone()),
                            text_value(row.finished_at.clone()),
                            text_value(row.backend.clone()),
                            text_value(row.input_path.clone()),
                            text_value(row.normalized_wav_path.clone()),
                            text_value(row.request_json.clone()),
                            text_value(row.result_json.clone()),
                            text_value(row.warnings_json.clone()),
                            text_value(row.transcript.clone()),
                            text_value(row.replay_json.clone()),
                            text_value(row.acceleration_json.clone()),
                        ],
                    )
                    .map_err(|error| FwError::Storage(error.to_string()))?;
            }

            let rebuilt_snapshot = self.load_runs_snapshot_rows(rebuilt_table_name, true, true)?;
            if rebuilt_snapshot.len() != expected_rows {
                return Err(FwError::Storage(format!(
                    "legacy migration integrity check failed: rebuilt row count {} != source row count {}",
                    rebuilt_snapshot.len(),
                    expected_rows
                )));
            }
            if rebuilt_snapshot != source_snapshot {
                return Err(FwError::Storage(
                    "legacy migration integrity check failed: rebuilt runs snapshot differs from source snapshot"
                        .to_owned(),
                ));
            }

            for (index_name, _) in &index_defs {
                self.connection
                    .execute(&format!("DROP INDEX IF EXISTS {};", sql_ident(index_name)))
                    .map_err(|error| {
                        FwError::Storage(format!(
                            "drop index `{index_name}` before runs swap failed: {error}"
                        ))
                    })?;
            }

            self.connection
                .execute(&format!("DROP TABLE IF EXISTS {};", sql_ident("runs")))
                .map_err(|error| FwError::Storage(error.to_string()))?;
            self.connection
                .execute(&format!(
                    "ALTER TABLE {rebuilt_ident} RENAME TO {};",
                    sql_ident("runs")
                ))
                .map_err(|error| FwError::Storage(error.to_string()))?;

            if failpoint_after_swap {
                return Err(FwError::Storage(
                    "simulated migration failure after runs table swap".to_owned(),
                ));
            }

            for (_, sql) in &index_defs {
                self.connection.execute(sql).map_err(|error| {
                    FwError::Storage(format!(
                        "recreate index for migrated runs table failed: {error}"
                    ))
                })?;
            }

            self.connection
                .execute("COMMIT;")
                .map_err(|error| FwError::Storage(error.to_string()))?;
            transaction_open = false;
            Ok(())
        })();

        if migration_result.is_err() && transaction_open {
            let _ = self.connection.execute("ROLLBACK;");
            let _ = self
                .connection
                .execute(&format!("DROP TABLE IF EXISTS {rebuilt_ident};"));
        }
        migration_result.map_err(|error| {
            FwError::Storage(format!(
                "safe legacy runs migration failed; restore from JSONL snapshot if needed: {error}"
            ))
        })?;

        if !self.table_has_column("runs", "replay_json")?
            || !self.table_has_column("runs", "acceleration_json")?
        {
            return Err(FwError::Storage(
                "legacy migration completed but required columns are still missing on runs table"
                    .to_owned(),
            ));
        }

        let post_snapshot = self.load_runs_snapshot_rows("runs", true, true)?;
        if post_snapshot.len() != expected_rows {
            return Err(FwError::Storage(format!(
                "legacy migration post-check failed: runs row count {} != source row count {}",
                post_snapshot.len(),
                expected_rows
            )));
        }
        Ok(())
    }

    fn recreate_table_with_added_column(
        &self,
        table: &str,
        existing_columns: &[TableColumn],
        new_column: &str,
        new_column_def: &str,
    ) -> FwResult<()> {
        if existing_columns.is_empty() {
            return Err(FwError::Storage(format!(
                "cannot rebuild table `{table}` with no discovered columns"
            )));
        }

        let table_ident = sql_ident(table);
        let temp_table_name = format!(
            "__fw_rebuild_{}_{}",
            table
                .chars()
                .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
                .collect::<String>(),
            new_column
                .chars()
                .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
                .collect::<String>()
        );
        let temp_table_ident = sql_ident(&temp_table_name);
        let existing_names = existing_columns
            .iter()
            .map(|row| {
                let name = row.name.clone();
                if name.is_empty() {
                    return Err(FwError::Storage(
                        "invalid PRAGMA table_info row: empty column name".to_owned(),
                    ));
                }
                Ok(sql_ident(&name))
            })
            .collect::<FwResult<Vec<_>>>()?;
        let existing_cols_csv = existing_names.join(", ");

        let index_defs = self.table_index_definitions(table)?;

        let mut column_defs = existing_columns
            .iter()
            .map(reconstruct_column_definition)
            .collect::<FwResult<Vec<_>>>()?;
        column_defs.push(format!("{} {}", sql_ident(new_column), new_column_def));

        let mut transaction_open = false;
        let rebuild_result = (|| -> FwResult<()> {
            self.connection
                .execute("BEGIN IMMEDIATE;")
                .map_err(|error| FwError::Storage(error.to_string()))?;
            transaction_open = true;

            self.connection
                .execute(&format!("DROP TABLE IF EXISTS {temp_table_ident};"))
                .map_err(|error| {
                    FwError::Storage(format!(
                        "drop temporary table `{temp_table_name}` failed: {error}"
                    ))
                })?;
            self.connection
                .execute(&format!(
                    "CREATE TABLE {temp_table_ident} ({});",
                    column_defs.join(", ")
                ))
                .map_err(|error| {
                    FwError::Storage(format!(
                        "create temporary table `{temp_table_name}` failed: {error}"
                    ))
                })?;
            self.connection
                .execute(&format!(
                    "INSERT INTO {temp_table_ident} ({existing_cols_csv}) \
                     SELECT {existing_cols_csv} FROM {table_ident};"
                ))
                .map_err(|error| {
                    FwError::Storage(format!(
                        "copy data from `{table}` to `{temp_table_name}` failed: {error}"
                    ))
                })?;
            for (index_name, _) in &index_defs {
                self.connection
                    .execute(&format!("DROP INDEX IF EXISTS {};", sql_ident(index_name)))
                    .map_err(|error| {
                        FwError::Storage(format!(
                            "drop index `{index_name}` before table rebuild failed: {error}"
                        ))
                    })?;
            }
            self.connection
                .execute(&format!("DROP TABLE IF EXISTS {table_ident};"))
                .map_err(|error| {
                    FwError::Storage(format!("drop original table `{table}` failed: {error}"))
                })?;
            self.connection
                .execute(&format!(
                    "ALTER TABLE {temp_table_ident} RENAME TO {table_ident};"
                ))
                .map_err(|error| {
                    FwError::Storage(format!(
                        "rename temporary table `{temp_table_name}` to `{table}` failed: {error}"
                    ))
                })?;

            for (_, sql) in &index_defs {
                self.connection.execute(sql).map_err(|error| {
                    FwError::Storage(format!(
                        "recreate index for rebuilt table `{table}` failed: {error}"
                    ))
                })?;
            }

            self.connection
                .execute("COMMIT;")
                .map_err(|error| FwError::Storage(error.to_string()))?;
            transaction_open = false;
            Ok(())
        })();

        if rebuild_result.is_err() && transaction_open {
            let _ = self.connection.execute("ROLLBACK;");
            let _ = self
                .connection
                .execute(&format!("DROP TABLE IF EXISTS {temp_table_ident};"));
        }

        rebuild_result
    }

    /// Execute one INSERT for [`persist_report_inner`]. When `skip_sp` is set the
    /// fsqlite per-statement savepoint is skipped (the caller's enclosing
    /// `SAVEPOINT` is the rollback boundary); otherwise the legacy path runs. The
    /// persisted row is identical in both cases.
    fn persist_insert(&self, skip_sp: bool, sql: &str, params: &[SqliteValue]) -> FwResult<()> {
        let result = if skip_sp {
            self.connection
                .execute_with_params_skip_statement_savepoint_in_explicit_txn(sql, params)
        } else {
            self.connection.execute_with_params(sql, params)
        };
        result
            .map(|_| ())
            .map_err(|error| FwError::Storage(error.to_string()))
    }

    fn persist_report_inner(
        &self,
        report: &RunReport,
        token: Option<&crate::orchestrator::CancellationToken>,
    ) -> FwResult<()> {
        // All inserts below run inside `persist_report_once`'s SAVEPOINT, which
        // rolls back on any Err — so the redundant per-statement savepoints can be
        // skipped (byte-identical persisted rows). See [`persist_skip_stmt_sp_enabled`].
        let skip_sp = persist_skip_stmt_sp_enabled();
        let request_json = serde_json::to_string(&report.request)?;
        let result_json = serde_json::to_string(&report.result)?;
        let warnings_json = serde_json::to_string(&report.warnings)?;
        let has_replay = self.table_has_column("runs", "replay_json")?;

        let acceleration_json = match &report.result.acceleration {
            Some(accel) => serde_json::to_string(accel)?,
            None => "{}".to_owned(),
        };

        if has_replay {
            let replay_json = serde_json::to_string(&report.replay)?;
            self.persist_insert(
                skip_sp,
                "INSERT INTO runs (id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                &[
                    text_value(report.run_id.clone()),
                    text_value(report.started_at_rfc3339.clone()),
                    text_value(report.finished_at_rfc3339.clone()),
                    text_value(report.result.backend.as_str().to_owned()),
                    text_value(report.input_path.clone()),
                    text_value(report.normalized_wav_path.clone()),
                    text_value(request_json),
                    text_value(result_json),
                    text_value(warnings_json),
                    text_value(report.result.transcript.clone()),
                    text_value(replay_json),
                    text_value(acceleration_json),
                ],
            )?;
        } else {
            self.persist_insert(
                skip_sp,
                "INSERT INTO runs (id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                &[
                    text_value(report.run_id.clone()),
                    text_value(report.started_at_rfc3339.clone()),
                    text_value(report.finished_at_rfc3339.clone()),
                    text_value(report.result.backend.as_str().to_owned()),
                    text_value(report.input_path.clone()),
                    text_value(report.normalized_wav_path.clone()),
                    text_value(request_json),
                    text_value(result_json),
                    text_value(warnings_json),
                    text_value(report.result.transcript.clone()),
                ],
            )?;
        }

        // Checkpoint before segments batch.
        if let Some(tok) = token {
            tok.checkpoint()?;
        }

        for (index, segment) in report.result.segments.iter().enumerate() {
            self.persist_insert(
                skip_sp,
                "INSERT INTO segments (run_id, idx, start_sec, end_sec, speaker, text, confidence) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    text_value(report.run_id.clone()),
                    SqliteValue::Integer(index as i64),
                    optional_float(segment.start_sec),
                    optional_float(segment.end_sec),
                    optional_text(segment.speaker.as_deref()),
                    text_value(segment.text.clone()),
                    optional_float(segment.confidence),
                ],
            )?;
        }

        // Checkpoint before events batch.
        if let Some(tok) = token {
            tok.checkpoint()?;
        }

        for event in &report.events {
            self.persist_insert(
                skip_sp,
                "INSERT INTO events (run_id, seq, ts_rfc3339, stage, code, message, payload_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    text_value(report.run_id.clone()),
                    SqliteValue::Integer(event.seq as i64),
                    text_value(event.ts_rfc3339.clone()),
                    text_value(event.stage.clone()),
                    text_value(event.code.clone()),
                    text_value(event.message.clone()),
                    text_value(serde_json::to_string(&event.payload)?),
                ],
            )?;
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn ensure_runs_replay_column(&self) -> FwResult<()> {
        self.ensure_column_exists("runs", "replay_json", "TEXT NOT NULL DEFAULT '{}'")
    }

    // -----------------------------------------------------------------------
    // bd-3i1.3: Diagnostic PRAGMAs for storage observability
    // -----------------------------------------------------------------------

    /// Query SQLite diagnostic PRAGMAs and return a structured snapshot of
    /// storage health/observability data.
    pub fn diagnostics(&self) -> FwResult<StorageDiagnostics> {
        let page_count = self.pragma_integer("page_count")?;
        let page_size = self.pragma_integer("page_size")?;

        let journal_mode = {
            let rows = self
                .connection
                .query("PRAGMA journal_mode;")
                .map_err(|error| FwError::Storage(error.to_string()))?;
            rows.first()
                .map(|r| value_to_string(r.get(0)))
                .unwrap_or_default()
        };

        let wal_checkpoint = {
            let rows = self
                .connection
                .query("PRAGMA wal_checkpoint(PASSIVE);")
                .map_err(|error| FwError::Storage(error.to_string()))?;
            if let Some(row) = rows.first() {
                WalCheckpointInfo {
                    busy: value_to_i64(row.get(0)),
                    log_frames: value_to_i64(row.get(1)),
                    checkpointed_frames: value_to_i64(row.get(2)),
                }
            } else {
                WalCheckpointInfo {
                    busy: 0,
                    log_frames: 0,
                    checkpointed_frames: 0,
                }
            }
        };

        let freelist_count = self.pragma_integer("freelist_count")?;

        let integrity_check = {
            let rows = self
                .connection
                .query("PRAGMA integrity_check;")
                .map_err(|error| FwError::Storage(error.to_string()))?;
            rows.first()
                .map(|r| value_to_string(r.get(0)))
                .unwrap_or_else(|| "unknown".to_owned())
        };

        Ok(StorageDiagnostics {
            page_count,
            page_size,
            journal_mode,
            wal_checkpoint,
            freelist_count,
            integrity_check,
        })
    }

    /// Helper: read a single integer value from a PRAGMA.
    fn pragma_integer(&self, pragma_name: &str) -> FwResult<i64> {
        let sql = format!("PRAGMA {pragma_name};");
        let rows = self
            .connection
            .query(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?;
        Ok(rows.first().map(|r| value_to_i64(r.get(0))).unwrap_or(0))
    }

    // -----------------------------------------------------------------------
    // bd-3i1.2: MVCC concurrent session persistence via SAVEPOINTs
    // -----------------------------------------------------------------------

    /// Begin a concurrent persist session using a SAVEPOINT for nested
    /// transaction support. Multiple sessions can operate without blocking
    /// each other within the same connection's transaction.
    pub fn begin_concurrent_session(
        &self,
        session_name: &str,
    ) -> FwResult<ConcurrentPersistSession<'_>> {
        // Validate session name contains only safe characters for SQL identifiers.
        if session_name.is_empty()
            || !session_name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(FwError::Storage(format!(
                "invalid session name: must be non-empty and contain only alphanumeric/underscore characters, got '{session_name}'"
            )));
        }
        let savepoint_name = format!("fw_session_{session_name}");
        let sql = format!("SAVEPOINT {savepoint_name};");
        self.connection
            .execute(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?;
        Ok(ConcurrentPersistSession {
            connection: &self.connection,
            savepoint_name,
            finished: false,
        })
    }
}

/// Diagnostic snapshot returned by [`RunStore::diagnostics()`].
#[derive(Debug, Clone)]
pub struct StorageDiagnostics {
    /// Total number of pages in the database file.
    pub page_count: i64,
    /// Size of each page in bytes.
    pub page_size: i64,
    /// Current journal mode (e.g. "wal", "delete").
    pub journal_mode: String,
    /// WAL checkpoint status from `PRAGMA wal_checkpoint(PASSIVE)`.
    pub wal_checkpoint: WalCheckpointInfo,
    /// Number of free-list pages in the database file.
    pub freelist_count: i64,
    /// Result of `PRAGMA integrity_check` ("ok" when healthy).
    pub integrity_check: String,
}

/// WAL checkpoint information returned as part of [`StorageDiagnostics`].
#[derive(Debug, Clone)]
pub struct WalCheckpointInfo {
    /// Whether the checkpoint was blocked (0 = not blocked).
    pub busy: i64,
    /// Total number of frames in the WAL log.
    pub log_frames: i64,
    /// Number of frames that were checkpointed.
    pub checkpointed_frames: i64,
}

/// A concurrent persist session backed by a SQLite SAVEPOINT.
///
/// Created via [`RunStore::begin_concurrent_session()`]. The session must be
/// explicitly committed or rolled back. If dropped without finishing, the
/// savepoint is automatically released (rolled back) to avoid leaked
/// transaction state.
#[derive(Debug)]
pub struct ConcurrentPersistSession<'conn> {
    connection: &'conn BlockingConnection,
    savepoint_name: String,
    finished: bool,
}

impl<'conn> ConcurrentPersistSession<'conn> {
    /// Commit (release) this savepoint, making all changes within it
    /// permanent in the enclosing transaction.
    pub fn commit(mut self) -> FwResult<()> {
        self.finished = true;
        let sql = format!("RELEASE SAVEPOINT {};", self.savepoint_name);
        self.connection
            .execute(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?;
        Ok(())
    }

    /// Roll back this savepoint, discarding all changes within it.
    pub fn rollback(mut self) -> FwResult<()> {
        self.finished = true;
        let sql = format!("ROLLBACK TO SAVEPOINT {};", self.savepoint_name);
        self.connection
            .execute(&sql)
            .map_err(|error| FwError::Storage(error.to_string()))?;
        // Release the savepoint after rollback to clean up.
        let release_sql = format!("RELEASE SAVEPOINT {};", self.savepoint_name);
        let _ = self.connection.execute(&release_sql);
        Ok(())
    }

    /// Execute arbitrary SQL within this savepoint's scope. Returns the
    /// number of rows affected.
    pub fn execute(&self, sql: &str) -> FwResult<usize> {
        self.connection
            .execute(sql)
            .map_err(|error| FwError::Storage(error.to_string()))
    }

    /// Execute parameterized SQL within this savepoint's scope.
    pub fn execute_with_params(&self, sql: &str, params: &[SqliteValue]) -> FwResult<usize> {
        self.connection
            .execute_with_params(sql, params)
            .map_err(|error| FwError::Storage(error.to_string()))
    }

    /// Query within this savepoint's scope.
    pub fn query(&self, sql: &str) -> FwResult<Vec<fsqlite::Row>> {
        self.connection
            .query(sql)
            .map_err(|error| FwError::Storage(error.to_string()))
    }
}

impl Drop for ConcurrentPersistSession<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // Safety net: release the savepoint so we don't leak transaction state.
            let sql = format!("ROLLBACK TO SAVEPOINT {};", self.savepoint_name);
            let _ = self.connection.execute(&sql);
            let release = format!("RELEASE SAVEPOINT {};", self.savepoint_name);
            let _ = self.connection.execute(&release);
        }
    }
}

/// Whether `persist_report_inner` writes its `runs`/`segments`/`events` rows with
/// the fsqlite "skip statement savepoint in explicit txn" escape hatch (default).
/// Those inserts always run inside `persist_report_once`'s enclosing `SAVEPOINT`,
/// which is the rollback boundary (it rolls back on any `Err`), so the per-INSERT
/// statement savepoint is redundant bookkeeping. Read live so an external-env A/B
/// can toggle it (cold write path). `FW_PERSIST_SKIP_STMT_SP=0` restores the
/// per-statement-savepoint path. Persisted rows are byte-identical either way.
fn persist_skip_stmt_sp_enabled() -> bool {
    std::env::var("FW_PERSIST_SKIP_STMT_SP").ok().as_deref() != Some("0")
}

/// Whether [`RunStore::load_run_details_batch`] loads many runs with two batched
/// `WHERE id/run_id IN (…)` queries (default) instead of one `load_run_details`
/// (two queries + two schema scans) per run — the app-level N+1 in the routing-
/// history command. Output is identical either way (asserted by a batch-vs-per-run
/// test). `FW_STORAGE_BATCH_HISTORY=0` restores the per-run path (kill-switch + A/B).
fn batch_history_enabled() -> bool {
    std::env::var("FW_STORAGE_BATCH_HISTORY").ok().as_deref() != Some("0")
}

/// Assemble a [`StoredRunDetails`] from a batched `runs` row (9 cols: id,
/// started_at, finished_at, backend, result_json, warnings_json, transcript,
/// replay_json, acceleration_json) and its batched `events` rows (7 cols: run_id,
/// seq, ts_rfc3339, stage, code, message, payload_json). Mirrors the per-run
/// assembly in `load_run_details_cancellable` exactly — the only difference is the
/// events carry `run_id` at column 0 (so seq…payload shift +1) to allow grouping.
fn assemble_run_details_batched(
    run_row: &fsqlite::Row,
    event_rows: &[fsqlite::Row],
) -> FwResult<StoredRunDetails> {
    let run_id = value_to_string(run_row.get(0));
    let started_at_rfc3339 = value_to_string(run_row.get(1));
    let finished_at_rfc3339 = value_to_string(run_row.get(2));
    let backend = parse_backend(&value_to_string(run_row.get(3)));
    let result_json = value_to_string(run_row.get(4));
    let warnings_json = value_to_string(run_row.get(5));
    let transcript_fallback = value_to_string(run_row.get(6));
    let replay_json = value_to_string(run_row.get(7));
    let acceleration_json = value_to_string(run_row.get(8));

    let result: TranscriptionResult = serde_json::from_str(&result_json).map_err(|error| {
        FwError::Storage(format!("invalid result_json for run {run_id}: {error}"))
    })?;
    let warnings =
        parse_required_json_field::<Vec<String>>("warnings_json", &run_id, &warnings_json)?;
    let replay = parse_required_json_field::<ReplayEnvelope>("replay_json", &run_id, &replay_json)?;
    let acceleration_override = parse_optional_acceleration_json(&run_id, &acceleration_json)?;
    let transcript = if result.transcript.trim().is_empty() {
        transcript_fallback
    } else {
        result.transcript.clone()
    };

    let events = event_rows
        .iter()
        .map(|event_row| {
            let payload_json = value_to_string(event_row.get(6));
            let payload = serde_json::from_str(&payload_json).map_err(|error| {
                FwError::Storage(format!(
                    "invalid event payload for run {} seq {}: {}",
                    run_id,
                    value_to_string(event_row.get(1)),
                    error
                ))
            })?;
            let seq_text = value_to_string(event_row.get(1));
            let seq = seq_text.parse::<u64>().map_err(|error| {
                FwError::Storage(format!(
                    "invalid event sequence `{}` for run {}: {}",
                    seq_text, run_id, error
                ))
            })?;
            Ok(RunEvent {
                seq,
                ts_rfc3339: value_to_string(event_row.get(2)),
                stage: value_to_string(event_row.get(3)),
                code: value_to_string(event_row.get(4)),
                message: value_to_string(event_row.get(5)),
                payload,
            })
        })
        .collect::<FwResult<Vec<_>>>()?;

    Ok(StoredRunDetails {
        run_id,
        started_at_rfc3339,
        finished_at_rfc3339,
        backend,
        transcript,
        segments: result.segments,
        events,
        warnings,
        acceleration: acceleration_override.or(result.acceleration),
        replay,
    })
}

/// Convert an optional `SqliteValue` to an `i64`, returning 0 for non-integer
/// or missing values.
fn value_to_i64(value: Option<&SqliteValue>) -> i64 {
    match value {
        Some(SqliteValue::Integer(n)) => *n,
        Some(SqliteValue::Text(s)) => s.parse::<i64>().unwrap_or(0),
        _ => 0,
    }
}

fn parse_backend(value: &str) -> BackendKind {
    match value {
        "whisper_cpp" => BackendKind::WhisperCpp,
        "insanely_fast" => BackendKind::InsanelyFast,
        "whisper_diarization" => BackendKind::WhisperDiarization,
        _ => BackendKind::Auto,
    }
}

fn is_busy_storage_error(error: &FwError) -> bool {
    let FwError::Storage(message) = error else {
        return false;
    };
    let lowered = message.to_ascii_lowercase();
    lowered.contains("database is busy")
        || lowered.contains("snapshot conflict")
        || lowered.contains("database is locked")
}

fn text_value(value: String) -> SqliteValue {
    SqliteValue::Text(value.into())
}

fn optional_text(value: Option<&str>) -> SqliteValue {
    match value {
        Some(text) => SqliteValue::Text(text.to_owned().into()),
        None => SqliteValue::Null,
    }
}

fn optional_float(value: Option<f64>) -> SqliteValue {
    match value {
        Some(number) => SqliteValue::Float(number),
        None => SqliteValue::Null,
    }
}

fn value_to_string(value: Option<&SqliteValue>) -> String {
    match value {
        Some(SqliteValue::Text(text)) => text.to_string(),
        Some(SqliteValue::Integer(number)) => number.to_string(),
        Some(SqliteValue::Float(number)) => number.to_string(),
        Some(SqliteValue::Blob(blob)) => format!("<blob:{}>", blob.len()),
        Some(SqliteValue::Null) | None => String::new(),
    }
}

fn parse_required_json_field<T: DeserializeOwned>(
    field_name: &str,
    run_id: &str,
    raw_json: &str,
) -> FwResult<T> {
    serde_json::from_str(raw_json).map_err(|error| {
        FwError::Storage(format!("invalid {field_name} for run {run_id}: {error}"))
    })
}

fn parse_optional_acceleration_json(
    run_id: &str,
    raw_json: &str,
) -> FwResult<Option<crate::model::AccelerationReport>> {
    let trimmed = raw_json.trim();
    if trimmed.is_empty() || trimmed == "{}" || trimmed == "null" {
        return Ok(None);
    }

    parse_required_json_field("acceleration_json", run_id, raw_json).map(Some)
}

fn next_persist_savepoint_name() -> String {
    let id = PERSIST_SAVEPOINT_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("fw_persist_{id}")
}

fn rollback_savepoint(connection: &BlockingConnection, savepoint_name: &str) -> FwResult<()> {
    let rollback_sql = format!("ROLLBACK TO SAVEPOINT {savepoint_name};");
    connection
        .execute(&rollback_sql)
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let release_sql = format!("RELEASE SAVEPOINT {savepoint_name};");
    connection
        .execute(&release_sql)
        .map_err(|error| FwError::Storage(error.to_string()))?;
    Ok(())
}

fn reconstruct_column_definition(column: &TableColumn) -> FwResult<String> {
    let name = column.name.clone();
    if name.is_empty() {
        return Err(FwError::Storage(
            "invalid PRAGMA table_info row: empty column name".to_owned(),
        ));
    }

    let typ = column.declared_type.clone();
    let not_null = column.not_null;
    let default_value = column.default_value.clone().unwrap_or_default();
    let is_primary_key = column.is_primary_key;

    let mut def = if typ.is_empty() {
        sql_ident(&name)
    } else {
        format!("{} {typ}", sql_ident(&name))
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

fn sql_ident(name: &str) -> String {
    let needs_quoting = name.is_empty()
        || name.as_bytes()[0].is_ascii_digit()
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
    if needs_quoting {
        format!("\"{}\"", name.replace('"', "\"\""))
    } else {
        name.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use serde_json::json;
    use tempfile::tempdir;

    use fsqlite_types::value::SqliteValue;

    use crate::model::{
        AccelerationBackend, AccelerationReport, BackendKind, BackendParams, InputSource, RunEvent,
        RunReport, TranscribeRequest, TranscriptionResult, TranscriptionSegment,
    };

    use super::{BlockingConnection, RunStore, value_to_string};

    #[test]
    fn persists_and_lists_runs() {
        let dir = tempdir().expect("tempdir should be created");
        let db_path = dir.path().join("storage.sqlite3");
        let store = RunStore::open(&db_path).expect("store should open");

        let report = RunReport {
            run_id: "run-1".to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
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
                transcript: "hello world".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(1.0),
                    text: "hello world".to_owned(),
                    speaker: None,
                    confidence: Some(0.9),
                }],
                acceleration: Some(AccelerationReport {
                    backend: AccelerationBackend::Frankentorch,
                    input_values: 1,
                    normalized_confidences: true,
                    pre_mass: Some(0.9),
                    post_mass: Some(1.0),
                    notes: vec!["normalized with frankentorch".to_owned()],
                }),
                raw_output: json!({"ok": true}),
                artifact_paths: vec!["out.json".to_owned()],
            },
            events: vec![
                RunEvent {
                    seq: 1,
                    ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                    stage: "test".to_owned(),
                    code: "test.event".to_owned(),
                    message: "ok".to_owned(),
                    payload: json!({"k": "v"}),
                },
                RunEvent {
                    seq: 2,
                    ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
                    stage: "persist".to_owned(),
                    code: "persist.ok".to_owned(),
                    message: "saved".to_owned(),
                    payload: json!({"db":"storage.sqlite3"}),
                },
            ],
            warnings: vec![],
            evidence: vec![],
            replay: crate::model::ReplayEnvelope {
                input_content_hash: Some("input-hash".to_owned()),
                backend_identity: Some("whisper-cli".to_owned()),
                backend_version: Some("whisper 1.2.3".to_owned()),
                output_payload_hash: Some("output-hash".to_owned()),
            },
        };

        store
            .persist_report(&report)
            .expect("report should persist successfully");

        let runs = store
            .list_recent_runs(10)
            .expect("runs should be queryable");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].backend, BackendKind::WhisperCpp);
        assert!(runs[0].transcript_preview.contains("hello"));

        let latest = store
            .load_latest_run_details()
            .expect("latest run query should succeed")
            .expect("latest run should exist");
        assert_eq!(latest.run_id, "run-1");
        assert_eq!(latest.segments.len(), 1);
        assert_eq!(latest.events.len(), 2);
        assert_eq!(latest.events[0].seq, 1);
        assert_eq!(latest.events[1].seq, 2);
        assert_eq!(latest.events[1].code, "persist.ok");
        assert_eq!(
            latest.acceleration.as_ref().map(|meta| meta.backend),
            Some(AccelerationBackend::Frankentorch)
        );
        assert_eq!(
            latest.replay.input_content_hash.as_deref(),
            Some("input-hash")
        );
        assert_eq!(
            latest.replay.backend_identity.as_deref(),
            Some("whisper-cli")
        );
        assert_eq!(
            latest.replay.backend_version.as_deref(),
            Some("whisper 1.2.3")
        );
        assert_eq!(
            latest.replay.output_payload_hash.as_deref(),
            Some("output-hash")
        );
    }

    fn minimal_report(run_id: &str, db_path: &std::path::Path) -> RunReport {
        RunReport {
            run_id: run_id.to_owned(),
            trace_id: "00000000000000000000000000000000".to_owned(),
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
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
                db_path: db_path.to_path_buf(),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "test".to_owned(),
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
        }
    }

    #[test]
    fn empty_segments_and_events_round_trip() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_segs.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-empty", &db_path);
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-empty")
            .expect("query")
            .expect("should exist");
        assert_eq!(details.segments.len(), 0);
        assert_eq!(details.events.len(), 0);
        assert!(details.warnings.is_empty());
        assert!(details.acceleration.is_none());
    }

    #[test]
    fn load_run_details_nonexistent_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_run.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let result = store.load_run_details("nonexistent-run-id").expect("query");
        assert!(result.is_none());
    }

    #[test]
    fn list_recent_runs_on_empty_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_db.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let runs = store.list_recent_runs(10).expect("query");
        assert!(runs.is_empty());
    }

    #[test]
    fn list_recent_runs_respects_limit() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("limit.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        for i in 0..10 {
            store
                .persist_report(&minimal_report(&format!("run-{i}"), &db_path))
                .expect("persist");
        }

        let all = store.list_recent_runs(100).expect("query");
        assert_eq!(all.len(), 10);

        let limited = store.list_recent_runs(3).expect("query");
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn load_latest_run_details_on_empty_db_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_latest.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let result = store.load_latest_run_details().expect("query");
        assert!(result.is_none());
    }

    #[test]
    fn multiple_runs_with_different_backends() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multi_backend.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report1 = minimal_report("run-cpp", &db_path);
        report1.result.backend = BackendKind::WhisperCpp;
        store.persist_report(&report1).expect("persist");

        let mut report2 = minimal_report("run-fast", &db_path);
        report2.result.backend = BackendKind::InsanelyFast;
        store.persist_report(&report2).expect("persist");

        let runs = store.list_recent_runs(10).expect("query");
        assert_eq!(runs.len(), 2);

        let backends: Vec<BackendKind> = runs.iter().map(|r| r.backend).collect();
        assert!(backends.contains(&BackendKind::WhisperCpp));
        assert!(backends.contains(&BackendKind::InsanelyFast));
    }

    #[test]
    fn very_long_transcript_persists_correctly() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("long_text.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let long_text = "word ".repeat(10_000);
        let mut report = minimal_report("run-long", &db_path);
        report.result.transcript = long_text.clone();
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-long")
            .expect("query")
            .expect("should exist");
        assert_eq!(details.transcript, long_text);
    }

    #[test]
    fn replay_envelope_defaults_round_trip() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("replay_default.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-no-replay", &db_path);
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-no-replay")
            .expect("query")
            .expect("should exist");
        assert!(details.replay.input_content_hash.is_none());
        assert!(details.replay.backend_identity.is_none());
        assert!(details.replay.backend_version.is_none());
        assert!(details.replay.output_payload_hash.is_none());
    }

    #[test]
    fn parse_backend_known_variants() {
        use super::parse_backend;
        assert_eq!(parse_backend("whisper_cpp"), BackendKind::WhisperCpp);
        assert_eq!(parse_backend("insanely_fast"), BackendKind::InsanelyFast);
        assert_eq!(
            parse_backend("whisper_diarization"),
            BackendKind::WhisperDiarization
        );
    }

    #[test]
    fn parse_backend_unknown_falls_back_to_auto() {
        use super::parse_backend;
        assert_eq!(parse_backend("unknown_backend"), BackendKind::Auto);
        assert_eq!(parse_backend(""), BackendKind::Auto);
        assert_eq!(parse_backend("WhisperCpp"), BackendKind::Auto); // case sensitive
    }

    #[test]
    fn value_to_string_all_variants() {
        use super::value_to_string;
        assert_eq!(
            value_to_string(Some(&SqliteValue::Text("hello".to_owned().into()))),
            "hello"
        );
        assert_eq!(value_to_string(Some(&SqliteValue::Integer(42))), "42");
        assert_eq!(value_to_string(Some(&SqliteValue::Float(2.75))), "2.75");
        assert_eq!(
            value_to_string(Some(&SqliteValue::Blob(vec![1, 2, 3].into()))),
            "<blob:3>"
        );
        assert_eq!(value_to_string(Some(&SqliteValue::Null)), "");
        assert_eq!(value_to_string(None), "");
    }

    #[test]
    fn text_value_wraps_string() {
        use super::text_value;
        let val = text_value("hello".to_owned());
        assert!(matches!(val, SqliteValue::Text(ref s) if &**s == "hello"));
    }

    #[test]
    fn text_value_empty_string() {
        use super::text_value;
        let val = text_value(String::new());
        assert!(matches!(val, SqliteValue::Text(ref s) if s.is_empty()));
    }

    #[test]
    fn optional_text_some_wraps_to_text() {
        use super::optional_text;
        let val = optional_text(Some("value"));
        assert!(matches!(val, SqliteValue::Text(ref s) if &**s == "value"));
    }

    #[test]
    fn optional_text_none_wraps_to_null() {
        use super::optional_text;
        let val = optional_text(None);
        assert!(matches!(val, SqliteValue::Null));
    }

    #[test]
    fn optional_float_some_wraps_to_float() {
        use super::optional_float;
        let val = optional_float(Some(2.75));
        assert!(matches!(val, SqliteValue::Float(v) if (v - 2.75).abs() < 1e-10));
    }

    #[test]
    fn optional_float_none_wraps_to_null() {
        use super::optional_float;
        let val = optional_float(None);
        assert!(matches!(val, SqliteValue::Null));
    }

    #[test]
    fn transcript_preview_truncates_at_140_chars() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("preview_trunc.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let long_text = "A".repeat(300);
        let mut report = minimal_report("run-preview", &db_path);
        report.result.transcript = long_text;
        store.persist_report(&report).expect("persist");

        let runs = store.list_recent_runs(10).expect("query");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].transcript_preview.len(), 140);
        assert!(runs[0].transcript_preview.chars().all(|c| c == 'A'));
    }

    #[test]
    fn schema_migration_idempotent_on_reopen() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("reopen.sqlite3");

        // Open once to create schema.
        let store1 = RunStore::open(&db_path).expect("first open");
        store1
            .persist_report(&minimal_report("run-first", &db_path))
            .expect("persist");
        drop(store1);

        // Open again — schema already exists, should not fail.
        let store2 = RunStore::open(&db_path).expect("second open");
        let runs = store2.list_recent_runs(10).expect("query");
        assert_eq!(runs.len(), 1, "data should survive reopen");
    }

    #[test]
    fn load_run_details_returns_correct_run_by_id() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("by_id.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut r1 = minimal_report("run-alpha", &db_path);
        r1.result.transcript = "alpha text".to_owned();
        let mut r2 = minimal_report("run-beta", &db_path);
        r2.result.transcript = "beta text".to_owned();
        store.persist_report(&r1).expect("persist r1");
        store.persist_report(&r2).expect("persist r2");

        let details = store
            .load_run_details("run-alpha")
            .expect("query")
            .expect("should exist");
        assert_eq!(details.run_id, "run-alpha");
        assert_eq!(details.transcript, "alpha text");
    }

    #[test]
    fn load_run_details_batch_matches_per_run() {
        // Byte-exactness guard for FW_STORAGE_BATCH_HISTORY: the batched two-query
        // loader must produce StoredRunDetails identical to per-run `load_run_details`
        // for every run (incl. multi-event grouping + the run_id-shifted event layout).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("batch_hist.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let ids = ["batch-a", "batch-b", "batch-c"];
        for (i, id) in ids.iter().enumerate() {
            let mut report = minimal_report(id, &db_path);
            report.result.transcript = format!("transcript {i}");
            report.result.segments = vec![TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some(i as f64 + 1.0),
                text: format!("segment for {id}"),
                speaker: None,
                confidence: Some(0.9),
            }];
            report.events = vec![
                RunEvent {
                    seq: 1,
                    ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                    stage: "ingest".to_owned(),
                    code: "ok".to_owned(),
                    message: format!("first for {id}"),
                    payload: serde_json::json!({"i": i}),
                },
                RunEvent {
                    seq: 2,
                    ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
                    stage: "backend".to_owned(),
                    code: "ok".to_owned(),
                    message: format!("second for {id}"),
                    payload: serde_json::json!({}),
                },
            ];
            store.persist_report(&report).expect("persist");
        }

        let run_ids: Vec<String> = ids.iter().map(|s| (*s).to_owned()).collect();
        let batched = store.load_run_details_batch(&run_ids).expect("batch load");
        assert_eq!(batched.len(), 3);
        for (i, id) in ids.iter().enumerate() {
            let per_run = store
                .load_run_details(id)
                .expect("per-run load")
                .expect("run exists");
            // Compare serialized form (StoredRunDetails is Serialize, not PartialEq) —
            // a direct byte-exact check of the loader output.
            assert_eq!(
                serde_json::to_string(&batched[i]).expect("ser batched"),
                serde_json::to_string(&per_run).expect("ser per-run"),
                "batched run `{id}` must byte-match the per-run loader"
            );
        }
    }

    #[test]
    fn event_with_nested_json_payload_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("nested.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let complex_payload = json!({
            "array": [1, 2, 3],
            "nested": { "key": "value", "bool": true },
            "null_val": null,
            "float": 42.5
        });

        let mut report = minimal_report("run-nested", &db_path);
        report.events = vec![RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            stage: "test".to_owned(),
            code: "test.nested".to_owned(),
            message: "complex payload".to_owned(),
            payload: complex_payload.clone(),
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-nested")
            .expect("query")
            .expect("should exist");
        assert_eq!(details.events.len(), 1);
        assert_eq!(details.events[0].payload, complex_payload);
    }

    #[test]
    fn empty_transcript_falls_back_to_stored_text() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("fallback_text.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-fallback", &db_path);
        // Result transcript is empty/whitespace, but the stored transcript column has text.
        report.result.transcript = "  ".to_owned();
        // We need to persist with a non-empty transcript column — but persist_report
        // writes result.transcript into the transcript column. Let's use a field that
        // would cause the fallback path in load_run_details: result.transcript is
        // empty but we manually wrote to the transcript column.
        // Actually, looking at code: persist writes result.transcript to transcript col,
        // and load checks result.transcript.trim().is_empty() → uses transcript_fallback.
        // So if result.transcript is "  " (whitespace), it will be stored and on load
        // the fallback path triggers and returns the stored column value ("  ").
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-fallback")
            .expect("query")
            .expect("should exist");
        // The fallback path uses the transcript column value, which is "  ".
        assert_eq!(details.transcript, "  ");
    }

    #[test]
    fn warnings_round_trip_when_populated() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("warnings.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-warns", &db_path);
        report.warnings = vec![
            "warning 1".to_owned(),
            "warning 2".to_owned(),
            "unicode: 日本語".to_owned(),
        ];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-warns")
            .expect("query")
            .expect("should exist");
        assert_eq!(details.warnings.len(), 3);
        assert_eq!(details.warnings[2], "unicode: 日本語");
    }

    #[test]
    fn segments_with_optional_fields_round_trip() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("segs_opt.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-segs", &db_path);
        report.result.segments = vec![
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "no timing".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: Some(1.5),
                end_sec: Some(3.0),
                text: "with timing".to_owned(),
                speaker: Some("SPEAKER_01".to_owned()),
                confidence: Some(0.95),
            },
        ];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-segs")
            .expect("query")
            .expect("should exist");
        assert_eq!(details.segments.len(), 2);
        assert!(details.segments[0].start_sec.is_none());
        assert!(details.segments[0].speaker.is_none());
        assert_eq!(details.segments[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(details.segments[1].confidence, Some(0.95));
    }

    #[test]
    fn load_latest_run_details_returns_most_recent() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("latest.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // Insert two runs with different timestamps.
        let mut early = minimal_report("run-early", &db_path);
        early.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        early.result.transcript = "early transcript".to_owned();

        let mut late = minimal_report("run-late", &db_path);
        late.started_at_rfc3339 = "2026-06-15T12:00:00Z".to_owned();
        late.result.transcript = "late transcript".to_owned();

        store.persist_report(&early).expect("persist early");
        store.persist_report(&late).expect("persist late");

        let details = store
            .load_latest_run_details()
            .expect("query")
            .expect("should exist");
        assert_eq!(details.run_id, "run-late");
        assert_eq!(details.transcript, "late transcript");
    }

    #[test]
    fn list_recent_runs_returns_reverse_chronological_order() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("order.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let timestamps = [
            ("run-a", "2026-01-01T00:00:00Z"),
            ("run-b", "2026-03-15T00:00:00Z"),
            ("run-c", "2026-02-10T00:00:00Z"),
        ];

        for (id, ts) in &timestamps {
            let mut report = minimal_report(id, &db_path);
            report.started_at_rfc3339 = ts.to_string();
            store.persist_report(&report).expect("persist");
        }

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 3);
        // Should be sorted by started_at DESC: b (March) > c (Feb) > a (Jan).
        assert_eq!(runs[0].run_id, "run-b");
        assert_eq!(runs[1].run_id, "run-c");
        assert_eq!(runs[2].run_id, "run-a");
    }

    #[test]
    fn run_with_many_events_preserves_order() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("many_events.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-events", &db_path);
        report.events = (0..100)
            .map(|i| RunEvent {
                seq: (i + 1) as u64,
                ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                stage: "test".to_owned(),
                code: format!("event.{i}"),
                message: format!("message-{i}"),
                payload: json!({"idx": i}),
            })
            .collect();
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-events")
            .expect("query")
            .expect("should exist");
        assert_eq!(details.events.len(), 100);

        // Verify order is preserved.
        for (i, event) in details.events.iter().enumerate() {
            assert_eq!(event.seq, (i + 1) as u64);
            assert_eq!(event.code, format!("event.{i}"));
        }
    }

    #[test]
    fn persist_report_twice_same_id_succeeds_idempotently() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("dup.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-dup", &db_path);
        store.persist_report(&report).expect("first persist");

        // fsqlite treats duplicate INSERT as no-op; verify data is still correct.
        let _ = store.persist_report(&report);
        let runs = store.list_recent_runs(10).expect("query");
        assert!(!runs.is_empty(), "at least the original run should remain");
    }

    #[test]
    fn many_segments_round_trip() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("many_segs.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-many-segs", &db_path);
        report.result.segments = (0..200)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some(i as f64 + 1.0),
                text: format!("segment {i}"),
                speaker: if i % 2 == 0 {
                    Some("SPEAKER_00".to_owned())
                } else {
                    None
                },
                confidence: Some(0.5 + (i as f64) / 400.0),
            })
            .collect();
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-many-segs")
            .expect("query")
            .expect("should exist");
        assert_eq!(details.segments.len(), 200);
        assert_eq!(details.segments[0].text, "segment 0");
        assert_eq!(details.segments[199].text, "segment 199");
        assert!(details.segments[0].speaker.is_some());
        assert!(details.segments[1].speaker.is_none());
    }

    #[test]
    fn unicode_transcript_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("unicode.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-unicode", &db_path);
        report.result.transcript = "日本語 中文 한국어 العربية emoji: 🎵🎤".to_owned();
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-unicode")
            .expect("query")
            .expect("should exist");
        assert!(details.transcript.contains("日本語"));
        assert!(details.transcript.contains("🎵🎤"));
    }

    #[test]
    fn open_creates_parent_directories() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("nested").join("deep").join("db.sqlite3");
        assert!(!db_path.parent().unwrap().exists());

        let store = RunStore::open(&db_path).expect("should create parent dirs");
        store
            .persist_report(&minimal_report("run-nested", &db_path))
            .expect("persist");

        let runs = store.list_recent_runs(10).expect("query");
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn value_to_string_empty_blob() {
        use super::value_to_string;
        assert_eq!(
            value_to_string(Some(&SqliteValue::Blob(vec![].into()))),
            "<blob:0>"
        );
    }

    #[test]
    fn optional_float_zero_wraps_to_float() {
        use super::optional_float;
        let val = optional_float(Some(0.0));
        assert!(matches!(val, SqliteValue::Float(v) if v == 0.0));
    }

    #[test]
    fn optional_text_empty_string_wraps_to_text() {
        use super::optional_text;
        let val = optional_text(Some(""));
        assert!(matches!(val, SqliteValue::Text(ref s) if s.is_empty()));
    }

    #[test]
    fn list_recent_runs_limit_one_returns_single() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("limit1.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        for i in 0..5 {
            store
                .persist_report(&minimal_report(&format!("run-{i}"), &db_path))
                .expect("persist");
        }

        let runs = store.list_recent_runs(1).expect("query");
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn replay_envelope_with_all_fields_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("replay_full.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-replay-full", &db_path);
        report.replay = crate::model::ReplayEnvelope {
            input_content_hash: Some("sha256_abc123".to_owned()),
            backend_identity: Some("whisper-cli".to_owned()),
            backend_version: Some("whisper 1.7.2".to_owned()),
            output_payload_hash: Some("sha256_def456".to_owned()),
        };
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-replay-full")
            .expect("query")
            .expect("should exist");
        assert_eq!(
            details.replay.input_content_hash.as_deref(),
            Some("sha256_abc123")
        );
        assert_eq!(
            details.replay.backend_identity.as_deref(),
            Some("whisper-cli")
        );
        assert_eq!(
            details.replay.backend_version.as_deref(),
            Some("whisper 1.7.2")
        );
        assert_eq!(
            details.replay.output_payload_hash.as_deref(),
            Some("sha256_def456")
        );
    }

    #[test]
    fn load_run_details_empty_string_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_id.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let result = store.load_run_details("").expect("query");
        assert!(result.is_none());
    }

    #[test]
    fn list_recent_runs_limit_zero_returns_all_rows() {
        // fsqlite treats LIMIT 0 as "no limit" — returns all rows.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("limit0.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("run-zero", &db_path))
            .expect("persist");
        let runs = store.list_recent_runs(0).expect("query");
        assert_eq!(runs.len(), 1, "LIMIT 0 in fsqlite returns all rows");
    }

    #[test]
    fn list_recent_runs_exact_count_boundary() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("exact.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        for i in 0..3 {
            store
                .persist_report(&minimal_report(&format!("run-{i}"), &db_path))
                .expect("persist");
        }
        // Asking for exactly 3 when 3 exist.
        let runs = store.list_recent_runs(3).expect("query");
        assert_eq!(runs.len(), 3);
        // Asking for 4 when only 3 exist.
        let runs = store.list_recent_runs(4).expect("query");
        assert_eq!(runs.len(), 3);
    }

    #[test]
    fn run_id_with_sql_special_chars_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("sql_chars.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let special_id = "run'; DROP TABLE runs; --";
        store
            .persist_report(&minimal_report(special_id, &db_path))
            .expect("persist");
        let details = store
            .load_run_details(special_id)
            .expect("query")
            .expect("should exist");
        assert_eq!(details.run_id, special_id);
    }

    #[test]
    fn segment_with_negative_timestamps_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("neg_ts.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-neg-ts", &db_path);
        report.result.segments = vec![TranscriptionSegment {
            start_sec: Some(-1.0),
            end_sec: Some(-0.5),
            text: "negative".to_owned(),
            speaker: None,
            confidence: Some(-0.1),
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-neg-ts")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments.len(), 1);
        assert_eq!(details.segments[0].start_sec, Some(-1.0));
        assert_eq!(details.segments[0].end_sec, Some(-0.5));
        assert_eq!(details.segments[0].confidence, Some(-0.1));
    }

    #[test]
    fn transcript_preview_truncates_multibyte_utf8_at_boundary() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multibyte.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // 141 emoji characters (each is 1 char but multi-byte in UTF-8).
        let emoji_transcript: String = std::iter::repeat_n('\u{1F600}', 141).collect();
        let mut report = minimal_report("run-emoji", &db_path);
        report.result.transcript = emoji_transcript;
        store.persist_report(&report).expect("persist");

        let runs = store.list_recent_runs(1).expect("query");
        assert_eq!(runs.len(), 1);
        assert_eq!(
            runs[0].transcript_preview.chars().count(),
            140,
            "multi-byte chars should be truncated by char count, not byte count"
        );
    }

    #[test]
    fn parse_backend_with_whitespace_falls_back_to_auto() {
        use super::parse_backend;
        assert_eq!(parse_backend(" whisper_cpp "), BackendKind::Auto);
        assert_eq!(parse_backend(""), BackendKind::Auto);
    }

    #[test]
    fn parse_backend_case_sensitive() {
        use super::parse_backend;
        assert_eq!(parse_backend("Whisper_Cpp"), BackendKind::Auto);
        assert_eq!(parse_backend("INSANELY_FAST"), BackendKind::Auto);
    }

    #[test]
    fn event_with_null_payload_values_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("null_payload.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-null-payload", &db_path);
        report.events = vec![RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            stage: "test".to_owned(),
            code: "test.null".to_owned(),
            message: "null values".to_owned(),
            payload: json!({"key": null, "nested": {"inner": null}}),
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-null-payload")
            .expect("query")
            .expect("exists");
        assert_eq!(details.events.len(), 1);
        assert!(details.events[0].payload["key"].is_null());
        assert!(details.events[0].payload["nested"]["inner"].is_null());
    }

    #[test]
    fn very_long_run_id_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("long_id.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let long_id = "x".repeat(1000);
        store
            .persist_report(&minimal_report(&long_id, &db_path))
            .expect("persist");
        let details = store
            .load_run_details(&long_id)
            .expect("query")
            .expect("exists");
        assert_eq!(details.run_id, long_id);
    }

    #[test]
    fn segment_with_zero_confidence_and_zero_timestamps() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("zero_vals.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-zeros", &db_path);
        report.result.segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(0.0),
            text: "zero".to_owned(),
            speaker: Some(String::new()),
            confidence: Some(0.0),
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-zeros")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments[0].start_sec, Some(0.0));
        assert_eq!(details.segments[0].end_sec, Some(0.0));
        assert_eq!(details.segments[0].confidence, Some(0.0));
    }

    #[test]
    fn load_run_details_with_acceleration_report() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("accel.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-accel", &db_path);
        report.result.acceleration = Some(AccelerationReport {
            backend: AccelerationBackend::Frankentorch,
            input_values: 100,
            normalized_confidences: true,
            pre_mass: Some(0.95),
            post_mass: Some(0.98),
            notes: vec!["test acceleration".to_owned()],
        });
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-accel")
            .expect("query")
            .expect("exists");
        let accel = details.acceleration.expect("acceleration should exist");
        assert_eq!(accel.backend, AccelerationBackend::Frankentorch);
        assert_eq!(accel.input_values, 100);
        assert!(accel.normalized_confidences);
        assert_eq!(accel.pre_mass, Some(0.95));
        assert_eq!(accel.post_mass, Some(0.98));
        assert_eq!(accel.notes, vec!["test acceleration".to_owned()]);
    }

    #[test]
    fn value_to_string_negative_integer() {
        use super::value_to_string;
        assert_eq!(value_to_string(Some(&SqliteValue::Integer(-42))), "-42");
        assert_eq!(
            value_to_string(Some(&SqliteValue::Integer(i64::MIN))),
            i64::MIN.to_string()
        );
    }

    #[test]
    fn value_to_string_large_float() {
        use super::value_to_string;
        let result = value_to_string(Some(&SqliteValue::Float(1e308)));
        assert!(!result.is_empty());
        // Verify it's parseable back.
        let parsed: f64 = result.parse().expect("should parse back");
        assert_eq!(parsed, 1e308);
    }

    #[test]
    fn optional_float_negative_zero_wraps_correctly() {
        use super::optional_float;
        let val = optional_float(Some(-0.0));
        assert!(matches!(val, SqliteValue::Float(v) if v == 0.0));
    }

    #[test]
    fn persist_report_with_each_backend_kind_lists_correctly() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("all_backends.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let backends = [
            ("run-cpp", BackendKind::WhisperCpp),
            ("run-fast", BackendKind::InsanelyFast),
            ("run-diar", BackendKind::WhisperDiarization),
            ("run-auto", BackendKind::Auto),
        ];
        for (id, backend) in &backends {
            let mut report = minimal_report(id, &db_path);
            report.result.backend = *backend;
            report.started_at_rfc3339 = format!("2026-01-0{}T00:00:00Z", id.len());
            store.persist_report(&report).expect("persist");
        }

        let runs = store.list_recent_runs(10).expect("query");
        assert_eq!(runs.len(), 4);
        let found_backends: Vec<BackendKind> = runs.iter().map(|r| r.backend).collect();
        assert!(found_backends.contains(&BackendKind::WhisperCpp));
        assert!(found_backends.contains(&BackendKind::InsanelyFast));
        assert!(found_backends.contains(&BackendKind::WhisperDiarization));
        assert!(found_backends.contains(&BackendKind::Auto));
    }

    #[test]
    fn event_with_empty_json_object_payload_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_obj.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-empty-payload", &db_path);
        report.events = vec![RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            stage: "test".to_owned(),
            code: "test.empty".to_owned(),
            message: "empty object".to_owned(),
            payload: json!({}),
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-empty-payload")
            .expect("query")
            .expect("exists");
        assert_eq!(details.events.len(), 1);
        assert!(details.events[0].payload.as_object().unwrap().is_empty());
    }

    #[test]
    fn multiple_runs_same_timestamp_all_listed() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("same_ts.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        for i in 0..3 {
            let mut report = minimal_report(&format!("run-same-{i}"), &db_path);
            report.started_at_rfc3339 = "2026-06-15T12:00:00Z".to_owned();
            store.persist_report(&report).expect("persist");
        }

        let runs = store.list_recent_runs(10).expect("query");
        assert_eq!(runs.len(), 3, "all runs with same timestamp should appear");
    }

    #[test]
    fn whitespace_only_transcript_falls_back_to_stored_text() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ws_transcript.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-ws", &db_path);
        report.result.transcript = "   \n\t  ".to_owned();
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-ws")
            .expect("query")
            .expect("exists");
        // The stored transcript column holds the raw text "   \n\t  " but
        // since result.transcript.trim().is_empty() is true, it falls back
        // to the transcript column value. Both are whitespace-only, but
        // the fallback is the raw column text (not trimmed).
        assert_eq!(details.transcript, "   \n\t  ");
    }

    #[test]
    fn load_latest_run_details_with_single_run() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("single.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("only-run", &db_path);
        store.persist_report(&report).expect("persist");

        let latest = store
            .load_latest_run_details()
            .expect("query")
            .expect("should exist");
        assert_eq!(latest.run_id, "only-run");
    }

    #[test]
    fn report_with_evidence_persists_without_crash() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("evidence.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-evidence", &db_path);
        report.evidence = vec![
            json!({"contract": "backend_selection", "action": "try_whisper_cpp"}),
            json!({"calibration": 0.85}),
        ];
        // Evidence is on RunReport but not in the DB schema — this should not crash
        store
            .persist_report(&report)
            .expect("persist should succeed despite evidence");

        let details = store
            .load_run_details("run-evidence")
            .expect("query")
            .expect("exists");
        assert_eq!(details.run_id, "run-evidence");
    }

    #[test]
    fn segment_speaker_with_special_chars_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("special_speaker.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-speaker", &db_path);
        report.result.segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "hello".to_owned(),
            speaker: Some("SPEAKER \"quoted\" & <xml>".to_owned()),
            confidence: Some(0.9),
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-speaker")
            .expect("query")
            .expect("exists");
        assert_eq!(
            details.segments[0].speaker.as_deref(),
            Some("SPEAKER \"quoted\" & <xml>")
        );
    }

    #[test]
    fn warnings_with_unicode_and_newlines_round_trip() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("unicode_warnings.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-uni-warn", &db_path);
        report.warnings = vec![
            "café résumé".to_owned(),
            "line1\nline2\nline3".to_owned(),
            "\u{1F600} emoji warning".to_owned(),
        ];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-uni-warn")
            .expect("query")
            .expect("exists");
        assert_eq!(details.warnings.len(), 3);
        assert_eq!(details.warnings[0], "café résumé");
        assert!(details.warnings[1].contains("line2"));
        assert!(details.warnings[2].contains('\u{1F600}'));
    }

    #[test]
    fn load_run_details_corrupt_result_json_returns_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("corrupt_result.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-corrupt", &db_path);
        store.persist_report(&report).expect("persist");

        // Corrupt result_json directly in the database.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute_with_params(
            "UPDATE runs SET result_json = ?1 WHERE id = ?2",
            &[
                fsqlite_types::value::SqliteValue::Text("{invalid json!!!".to_owned().into()),
                fsqlite_types::value::SqliteValue::Text("run-corrupt".to_owned().into()),
            ],
        )
        .expect("update");

        let err = store
            .load_run_details("run-corrupt")
            .expect_err("corrupt result_json should fail");
        let text = err.to_string();
        assert!(
            text.contains("run-corrupt"),
            "error should contain run_id: {text}"
        );
    }

    #[test]
    fn load_run_details_corrupt_event_payload_returns_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("corrupt_event.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-evt-corrupt", &db_path);
        report.events.push(RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            stage: "test".to_owned(),
            code: "test.ok".to_owned(),
            message: "test event".to_owned(),
            payload: json!({"key": "value"}),
        });
        store.persist_report(&report).expect("persist");

        // Corrupt an event's payload_json.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute_with_params(
            "UPDATE events SET payload_json = ?1 WHERE run_id = ?2 AND seq = ?3",
            &[
                fsqlite_types::value::SqliteValue::Text("{broken".to_owned().into()),
                fsqlite_types::value::SqliteValue::Text("run-evt-corrupt".to_owned().into()),
                fsqlite_types::value::SqliteValue::Integer(1),
            ],
        )
        .expect("update");

        let err = store
            .load_run_details("run-evt-corrupt")
            .expect_err("corrupt event payload should fail");
        let text = err.to_string();
        assert!(
            text.contains("run-evt-corrupt"),
            "error should contain run_id: {text}"
        );
    }

    #[test]
    fn load_run_details_nonexistent_run_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let result = store.load_run_details("nonexistent-run").expect("query");
        assert!(result.is_none(), "nonexistent run should return None");
    }

    #[test]
    fn list_recent_runs_with_zero_limit() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("zero_limit.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-zero-limit", &db_path);
        store.persist_report(&report).expect("persist");

        // fsqlite LIMIT 0 returns all rows (documented quirk)
        let runs = store.list_recent_runs(0).expect("list");
        assert!(
            !runs.is_empty(),
            "LIMIT 0 should return all rows per fsqlite behavior"
        );
    }

    #[test]
    fn load_latest_run_details_with_no_runs_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_runs.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let result = store.load_latest_run_details().expect("query");
        assert!(result.is_none(), "empty DB should return None for latest");
    }

    #[test]
    fn segment_with_very_large_timestamps_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("huge_ts.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-huge", &db_path);
        report.result.segments = vec![TranscriptionSegment {
            start_sec: Some(86400.0), // 24 hours in seconds
            end_sec: Some(86401.5),
            text: "very late segment".to_owned(),
            speaker: None,
            confidence: Some(0.99),
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-huge")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments[0].start_sec, Some(86400.0));
        assert_eq!(details.segments[0].end_sec, Some(86401.5));
    }

    #[test]
    fn load_run_details_corrupt_warnings_json_returns_storage_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("corrupt_warnings.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-bad-warns", &db_path);
        report.warnings = vec!["a real warning".to_owned()];
        store.persist_report(&report).expect("persist");

        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute_with_params(
            "UPDATE runs SET warnings_json = ?1 WHERE id = ?2",
            &[
                SqliteValue::Text("{not valid json".to_owned().into()),
                SqliteValue::Text("run-bad-warns".to_owned().into()),
            ],
        )
        .expect("corrupt");

        let error = store
            .load_run_details("run-bad-warns")
            .expect_err("corrupt warnings_json should fail loudly");
        assert!(error.to_string().contains("invalid warnings_json"));
    }

    #[test]
    fn load_run_details_corrupt_replay_json_returns_storage_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("corrupt_replay.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-bad-replay", &db_path);
        store.persist_report(&report).expect("persist");

        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute_with_params(
            "UPDATE runs SET replay_json = ?1 WHERE id = ?2",
            &[
                SqliteValue::Text("{broken json!!!".to_owned().into()),
                SqliteValue::Text("run-bad-replay".to_owned().into()),
            ],
        )
        .expect("corrupt");

        let error = store
            .load_run_details("run-bad-replay")
            .expect_err("corrupt replay_json should fail loudly");
        assert!(error.to_string().contains("invalid replay_json"));
    }

    #[test]
    fn ensure_runs_replay_column_migrates_legacy_schema() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("legacy.sqlite3");

        // Create a DB manually WITHOUT replay_json (simulating older schema).
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, \
                started_at TEXT NOT NULL, \
                finished_at TEXT NOT NULL, \
                backend TEXT NOT NULL, \
                input_path TEXT NOT NULL, \
                normalized_wav_path TEXT NOT NULL, \
                request_json TEXT NOT NULL, \
                result_json TEXT NOT NULL, \
                warnings_json TEXT NOT NULL, \
                transcript TEXT NOT NULL\
            );",
        )
        .expect("create legacy runs table");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT NOT NULL, \
                idx INTEGER NOT NULL, \
                start_sec REAL, \
                end_sec REAL, \
                speaker TEXT, \
                text TEXT NOT NULL, \
                confidence REAL, \
                PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments table");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT NOT NULL, \
                seq INTEGER NOT NULL, \
                ts_rfc3339 TEXT NOT NULL, \
                stage TEXT NOT NULL, \
                code TEXT NOT NULL, \
                message TEXT NOT NULL, \
                payload_json TEXT NOT NULL, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events table");
        drop(conn);

        // Opening via RunStore should trigger schema migration that adds
        // replay/acceleration columns.
        let store = RunStore::open(&db_path).expect("migration should succeed");
        let columns = store
            .connection
            .query("PRAGMA table_info(runs);")
            .expect("pragma");
        let has_replay = columns
            .iter()
            .any(|row| super::value_to_string(row.get(1)) == "replay_json");
        let has_acceleration = columns
            .iter()
            .any(|row| super::value_to_string(row.get(1)) == "acceleration_json");
        assert!(has_replay, "migration should add replay_json");
        assert!(has_acceleration, "migration should add acceleration_json");

        let report = minimal_report("run-migrated", &db_path);
        store
            .persist_report(&report)
            .expect("persist after migration");
        let details = store
            .load_run_details("run-migrated")
            .expect("query")
            .expect("exists");
        assert!(
            details.replay.input_content_hash.is_none(),
            "migrated column should default to empty replay"
        );
    }

    #[test]
    fn open_on_directory_path_returns_storage_error() {
        let dir = tempdir().expect("tempdir");
        // Point at an existing directory, not a file — SQLite cannot open this.
        let err = RunStore::open(dir.path());
        assert!(err.is_err(), "opening a directory as a DB should fail");
    }

    #[test]
    fn persist_duplicate_id_succeeds_and_both_rows_loadable() {
        // Duplicate run IDs should be rejected by the storage backend.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("dup.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-dup", &db_path);
        store.persist_report(&report).expect("first persist");
        let error = store
            .persist_report(&report)
            .expect_err("second persist should fail on duplicate run id");
        assert!(
            error
                .to_string()
                .contains("UNIQUE constraint failed: runs.id"),
            "duplicate persist should surface unique constraint error: {error}",
        );

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(
            runs.len(),
            1,
            "duplicate persist should not add a second row"
        );
        let details = store
            .load_run_details("run-dup")
            .expect("query")
            .expect("exists");
        assert_eq!(details.transcript, "test");
    }

    #[test]
    fn events_loaded_in_seq_order() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("evt_order.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-order", &db_path);
        // Insert events out of natural order to verify ORDER BY seq ASC.
        report.events = vec![
            RunEvent {
                seq: 3,
                ts_rfc3339: "2026-01-01T00:00:03Z".to_owned(),
                stage: "third".to_owned(),
                code: "c".to_owned(),
                message: "third".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
                stage: "first".to_owned(),
                code: "a".to_owned(),
                message: "first".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 2,
                ts_rfc3339: "2026-01-01T00:00:02Z".to_owned(),
                stage: "second".to_owned(),
                code: "b".to_owned(),
                message: "second".to_owned(),
                payload: json!({}),
            },
        ];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-order")
            .expect("query")
            .expect("exists");
        assert_eq!(details.events.len(), 3);
        assert_eq!(details.events[0].seq, 1, "first by seq");
        assert_eq!(details.events[1].seq, 2, "second by seq");
        assert_eq!(details.events[2].seq, 3, "third by seq");
        assert_eq!(details.events[0].stage, "first");
        assert_eq!(details.events[2].stage, "third");
    }

    #[test]
    fn open_creates_nested_parent_directories() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir
            .path()
            .join("a")
            .join("b")
            .join("c")
            .join("deep.sqlite3");
        assert!(!db_path.parent().unwrap().exists());
        let _store = RunStore::open(&db_path).expect("should create nested dirs");
        assert!(db_path.exists(), "DB file should exist after open");
    }

    #[test]
    fn multiple_segments_with_all_optional_fields() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("full_segs.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-full-segs", &db_path);
        report.result.segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.5),
                text: "first segment".to_owned(),
                speaker: Some("SPEAKER_00".to_owned()),
                confidence: Some(0.95),
            },
            TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "no timestamps".to_owned(),
                speaker: None,
                confidence: None,
            },
            TranscriptionSegment {
                start_sec: Some(3.0),
                end_sec: Some(4.5),
                text: "third with speaker".to_owned(),
                speaker: Some("SPEAKER_01".to_owned()),
                confidence: Some(0.88),
            },
        ];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-full-segs")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments.len(), 3);
        // First: all fields present.
        assert_eq!(details.segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(details.segments[0].confidence, Some(0.95));
        // Second: all optional fields None.
        assert!(details.segments[1].start_sec.is_none());
        assert!(details.segments[1].speaker.is_none());
        assert!(details.segments[1].confidence.is_none());
        // Third: verify speaker and confidence.
        assert_eq!(details.segments[2].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(details.segments[2].confidence, Some(0.88));
    }

    #[test]
    fn load_run_details_returns_correct_backend_for_each_kind() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("backends.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let backends = [
            ("run-auto", BackendKind::Auto),
            ("run-cpp", BackendKind::WhisperCpp),
            ("run-fast", BackendKind::InsanelyFast),
            ("run-diar", BackendKind::WhisperDiarization),
        ];
        for (id, backend) in &backends {
            let mut report = minimal_report(id, &db_path);
            report.result.backend = *backend;
            store.persist_report(&report).expect("persist");
        }

        for (id, expected_backend) in &backends {
            let details = store.load_run_details(id).expect("query").expect("exists");
            assert_eq!(
                details.backend, *expected_backend,
                "backend mismatch for {id}"
            );
        }
    }

    #[test]
    fn list_recent_runs_shows_correct_backend_for_all_kinds() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("backend_list.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut r1 = minimal_report("run-a", &db_path);
        r1.result.backend = BackendKind::WhisperDiarization;
        r1.started_at_rfc3339 = "2026-01-01T00:00:01Z".to_owned();
        store.persist_report(&r1).expect("persist");

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs[0].backend, BackendKind::WhisperDiarization);
    }

    #[test]
    fn load_run_details_corrupt_event_seq_returns_storage_error() {
        // When the event seq column contains a non-numeric value,
        // load_run_details returns a Storage error at line 149-154.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("corrupt_seq.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-bad-seq", &db_path);
        store.persist_report(&report).expect("persist");

        // Insert a corrupt event directly via raw SQL.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("open");
        conn.execute_with_params(
            "INSERT INTO events (run_id, seq, ts_rfc3339, stage, code, message, payload_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            &[
                SqliteValue::Text("run-bad-seq".to_owned().into()),
                SqliteValue::Text("not_a_number".to_owned().into()),
                SqliteValue::Text("2026-01-01T00:00:00Z".to_owned().into()),
                SqliteValue::Text("test".to_owned().into()),
                SqliteValue::Text("test.ok".to_owned().into()),
                SqliteValue::Text("msg".to_owned().into()),
                SqliteValue::Text("{}".to_owned().into()),
            ],
        )
        .expect("insert corrupt event");

        let err = store
            .load_run_details("run-bad-seq")
            .expect_err("should fail on corrupt seq");
        let text = err.to_string();
        assert!(
            text.contains("invalid event sequence"),
            "error should mention invalid event sequence: {text}"
        );
    }

    #[test]
    fn transcript_preview_truncated_at_140_chars() {
        // list_recent_runs truncates transcript to 140 chars for preview (line 67).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("preview_trunc.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let long_transcript = "A".repeat(200);
        let mut report = minimal_report("run-long", &db_path);
        report.result.transcript = long_transcript.clone();
        store.persist_report(&report).expect("persist");

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs[0].transcript_preview.len(), 140);
        assert_eq!(runs[0].transcript_preview, "A".repeat(140));
    }

    #[test]
    fn load_run_details_empty_result_json_returns_error() {
        // When result_json is "" (empty string), deserialization fails at line 115.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_result.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-empty-rj", &db_path);
        store.persist_report(&report).expect("persist");

        // Overwrite result_json with an empty string.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("open");
        conn.execute_with_params(
            "UPDATE runs SET result_json = ?1 WHERE id = ?2",
            &[
                SqliteValue::Text(String::new().into()),
                SqliteValue::Text("run-empty-rj".to_owned().into()),
            ],
        )
        .expect("update");

        let err = store
            .load_run_details("run-empty-rj")
            .expect_err("should fail with empty result_json");
        let text = err.to_string();
        assert!(
            text.contains("invalid result_json"),
            "error should mention invalid result_json: {text}"
        );
    }

    #[test]
    fn transcript_fallback_used_when_result_transcript_is_whitespace() {
        // When result.transcript is whitespace-only, the raw transcript column
        // is used as fallback (lines 121-125).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("fallback_transcript.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-ws-transcript", &db_path);
        report.result.transcript = "   ".to_owned();
        store.persist_report(&report).expect("persist");

        // Update the denormalized transcript column with a meaningful value.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("open");
        conn.execute_with_params(
            "UPDATE runs SET transcript = ?1 WHERE id = ?2",
            &[
                SqliteValue::Text("fallback transcript value".to_owned().into()),
                SqliteValue::Text("run-ws-transcript".to_owned().into()),
            ],
        )
        .expect("update");

        let details = store
            .load_run_details("run-ws-transcript")
            .expect("load")
            .expect("should exist");
        assert_eq!(
            details.transcript, "fallback transcript value",
            "should use transcript column when result.transcript is whitespace"
        );
    }

    #[test]
    fn persist_report_with_unicode_and_special_chars() {
        // Verify full Unicode round-trip through persist → load_run_details.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("unicode.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-unicode", &db_path);
        report.result.transcript =
            "日本語テスト 🎉 <script>alert('xss')</script> café résumé".to_owned();
        report.result.segments = vec![crate::model::TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "Ñoño → «quoted» ∞".to_owned(),
            speaker: Some("Spëaker_00".to_owned()),
            confidence: Some(0.99),
        }];
        report.warnings = vec!["⚠️ warning with emoji".to_owned()];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-unicode")
            .expect("load")
            .expect("should exist");
        assert_eq!(
            details.transcript,
            "日本語テスト 🎉 <script>alert('xss')</script> café résumé"
        );
        assert_eq!(details.segments[0].text, "Ñoño → «quoted» ∞");
        assert_eq!(details.segments[0].speaker.as_deref(), Some("Spëaker_00"));
        assert_eq!(details.warnings[0], "⚠️ warning with emoji");
    }

    #[test]
    fn list_recent_runs_with_identical_timestamps_returns_all() {
        // When multiple runs have the same started_at, all should be returned.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("same_ts.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        for i in 0..5 {
            let mut report = minimal_report(&format!("same-ts-{i}"), &db_path);
            report.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
            store.persist_report(&report).expect("persist");
        }

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(
            runs.len(),
            5,
            "all runs with same timestamp should be returned"
        );
    }

    #[test]
    fn persist_and_load_run_with_many_events() {
        // Verify that a run with 100+ events round-trips correctly.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("many_events.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-many-events", &db_path);
        report.events = (0..150)
            .map(|i| crate::model::RunEvent {
                seq: i as u64,
                ts_rfc3339: format!("2026-01-01T00:00:{:02}Z", i % 60),
                stage: "test".to_owned(),
                code: format!("test.{i}"),
                message: format!("event {i}"),
                payload: json!({"index": i}),
            })
            .collect();
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-many-events")
            .expect("load")
            .expect("exists");
        assert_eq!(details.events.len(), 150, "all 150 events should be loaded");
        assert_eq!(details.events[0].seq, 0);
        assert_eq!(details.events[149].seq, 149);
    }

    #[test]
    fn load_run_details_acceleration_none_round_trips() {
        // When acceleration is None, it should remain None after round-trip.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_accel.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-no-accel", &db_path);
        assert!(report.result.acceleration.is_none());
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-no-accel")
            .expect("load")
            .expect("exists");
        assert!(
            details.acceleration.is_none(),
            "acceleration should remain None"
        );
    }

    #[test]
    fn list_recent_runs_transcript_preview_with_multibyte_chars() {
        // Verify that transcript_preview truncation at 140 chars works
        // correctly with multibyte Unicode characters (char boundary, not byte boundary).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("multibyte_preview.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // Each CJK character is 3 bytes in UTF-8.
        let long_cjk = "漢".repeat(200);
        let mut report = minimal_report("run-cjk", &db_path);
        report.result.transcript = long_cjk;
        store.persist_report(&report).expect("persist");

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(
            runs[0].transcript_preview.chars().count(),
            140,
            "preview should be 140 chars (not bytes)"
        );
        // Every char should be '漢'.
        assert!(runs[0].transcript_preview.chars().all(|c| c == '漢'));
    }

    #[test]
    fn persist_report_with_empty_run_id() {
        // Empty string is technically valid as a run_id.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_id.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("", &db_path);
        store.persist_report(&report).expect("persist empty id");

        let details = store
            .load_run_details("")
            .expect("load")
            .expect("should exist");
        assert_eq!(details.run_id, "");
        assert_eq!(details.transcript, "test");
    }

    #[test]
    fn event_with_seq_zero_round_trips_and_loads_first() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("seq_zero.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-seq0", &db_path);
        report.events = vec![
            RunEvent {
                seq: 2,
                ts_rfc3339: "2026-01-01T00:00:02Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.ok".to_owned(),
                message: "second".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 0,
                ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ingest.ok".to_owned(),
                message: "zeroth".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
                stage: "normalize".to_owned(),
                code: "normalize.ok".to_owned(),
                message: "first".to_owned(),
                payload: json!({}),
            },
        ];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-seq0")
            .expect("query")
            .expect("exists");
        assert_eq!(details.events.len(), 3);
        assert_eq!(details.events[0].seq, 0, "seq=0 should appear first");
        assert_eq!(details.events[0].message, "zeroth");
        assert_eq!(details.events[1].seq, 1);
        assert_eq!(details.events[2].seq, 2);
    }

    #[test]
    fn list_recent_runs_preview_exact_140_chars_not_truncated() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("exact_140.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let exactly_140 = "B".repeat(140);
        let mut report = minimal_report("run-140", &db_path);
        report.result.transcript = exactly_140.clone();
        store.persist_report(&report).expect("persist");

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 1);
        assert_eq!(
            runs[0].transcript_preview, exactly_140,
            "140-char transcript should not be truncated"
        );
        assert_eq!(runs[0].transcript_preview.len(), 140);
    }

    #[test]
    fn load_run_details_both_transcript_sources_empty_returns_empty_string() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("both_empty.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-empty-tx", &db_path);
        report.result.transcript = String::new();
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-empty-tx")
            .expect("query")
            .expect("exists");
        // result.transcript is "" → trim() → "" → is_empty() → true → fallback.
        // Fallback is also "" (same value was stored in the transcript column).
        assert_eq!(details.transcript, "");
    }

    #[test]
    fn segments_loaded_from_result_json_not_segments_table() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("seg_source.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-segsrc", &db_path);
        report.result.segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "from json".to_owned(),
            speaker: None,
            confidence: None,
        }];
        store.persist_report(&report).expect("persist");

        // Delete all rows from the segments table for this run.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute_with_params(
            "DELETE FROM segments WHERE run_id = ?1",
            &[SqliteValue::Text("run-segsrc".to_owned().into())],
        )
        .expect("delete segments rows");

        // load_run_details reads segments from result_json, not the segments table.
        let details = store
            .load_run_details("run-segsrc")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments.len(), 1, "segments come from result_json");
        assert_eq!(details.segments[0].text, "from json");
    }

    #[test]
    fn load_latest_run_details_by_timestamp_not_insertion_order() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("insert_order.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // Insert the chronologically-later run FIRST.
        let mut late = minimal_report("run-late-first", &db_path);
        late.started_at_rfc3339 = "2026-12-31T23:59:59Z".to_owned();
        late.result.transcript = "late run".to_owned();
        store.persist_report(&late).expect("persist late");

        // Then insert the chronologically-earlier run SECOND.
        let mut early = minimal_report("run-early-second", &db_path);
        early.started_at_rfc3339 = "2026-01-01T00:00:00Z".to_owned();
        early.result.transcript = "early run".to_owned();
        store.persist_report(&early).expect("persist early");

        // load_latest uses ASC ordering + rows.last() (bd-2oe workaround),
        // so it should return the chronologically-later run regardless of insertion order.
        let latest = store
            .load_latest_run_details()
            .expect("query")
            .expect("exists");
        assert_eq!(latest.run_id, "run-late-first");
        assert_eq!(latest.transcript, "late run");
    }

    #[test]
    fn segment_nan_confidence_serializes_as_null_and_round_trips_to_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("nan_conf.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-nan", &db_path);
        report.result.segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "nan test".to_owned(),
            speaker: None,
            confidence: Some(f64::NAN),
        }];
        // serde_json serializes NaN as null, so after round-trip confidence becomes None.
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-nan")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments.len(), 1);
        assert!(
            details.segments[0].confidence.is_none(),
            "NaN confidence should round-trip as None (serialized as null)"
        );
    }

    #[test]
    fn list_recent_runs_limit_equals_run_count_returns_all() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("limit_eq.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        for i in 0..5 {
            let mut report = minimal_report(&format!("eq-{i}"), &db_path);
            report.started_at_rfc3339 = format!("2026-01-0{}T00:00:00Z", i + 1);
            store.persist_report(&report).expect("persist");
        }

        let runs = store.list_recent_runs(5).expect("list");
        assert_eq!(runs.len(), 5, "limit==count should return all runs");
        // Verify reverse-chronological order: eq-4 (Jan 5) first.
        assert_eq!(runs[0].run_id, "eq-4");
        assert_eq!(runs[4].run_id, "eq-0");
    }

    #[test]
    fn transcript_from_result_json_used_when_nonempty_over_column() {
        // When result.transcript is non-empty, it takes priority over
        // the transcript column value (lines 122-126).
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("transcript_prio.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-prio", &db_path);
        report.result.transcript = "from result_json".to_owned();
        store.persist_report(&report).expect("persist");

        // Overwrite the denormalized transcript column with a different value.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("open");
        conn.execute_with_params(
            "UPDATE runs SET transcript = ?1 WHERE id = ?2",
            &[
                SqliteValue::Text("from transcript column".to_owned().into()),
                SqliteValue::Text("run-prio".to_owned().into()),
            ],
        )
        .expect("update");

        let details = store
            .load_run_details("run-prio")
            .expect("query")
            .expect("exists");
        assert_eq!(
            details.transcript, "from result_json",
            "non-empty result.transcript should take priority over transcript column"
        );
    }

    #[test]
    fn persist_report_with_100_segments_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("many_segs.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-100-segs", &db_path);
        report.result.segments = (0..100)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some((i + 1) as f64),
                text: format!("segment-{i}"),
                speaker: if i % 3 == 0 {
                    Some(format!("SPEAKER_{:02}", i % 5))
                } else {
                    None
                },
                confidence: Some(0.5 + (i as f64) * 0.005),
            })
            .collect();
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-100-segs")
            .expect("query")
            .expect("exists");
        assert_eq!(details.segments.len(), 100);
        assert_eq!(details.segments[0].text, "segment-0");
        assert_eq!(details.segments[99].text, "segment-99");
        assert!(
            details.segments[0].speaker.is_some(),
            "index 0 mod 3 == 0 → speaker present"
        );
        assert!(
            details.segments[1].speaker.is_none(),
            "index 1 mod 3 != 0 → speaker absent"
        );
    }

    #[test]
    fn value_to_string_float_nan_returns_nan_string() {
        use super::value_to_string;
        let result = value_to_string(Some(&SqliteValue::Float(f64::NAN)));
        assert_eq!(result, "NaN", "NaN float should stringify as \"NaN\"");

        let inf = value_to_string(Some(&SqliteValue::Float(f64::INFINITY)));
        assert_eq!(inf, "inf", "INFINITY should stringify as \"inf\"");
    }

    // -----------------------------------------------------------------------
    // Schema versioning tests (bd-3i1.4)
    // -----------------------------------------------------------------------

    #[test]
    fn fresh_db_at_latest_version() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("fresh.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        let version = store.current_schema_version().expect("version");
        assert_eq!(
            version,
            RunStore::SCHEMA_VERSION,
            "fresh DB should be at latest schema version"
        );
    }

    #[test]
    fn meta_table_created_on_open() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("meta.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let tables = store
            .connection
            .query("SELECT name FROM sqlite_master WHERE type='table' AND name='_meta';")
            .expect("query");
        assert_eq!(tables.len(), 1, "_meta table should exist");
    }

    #[test]
    fn migration_idempotent() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("idempotent.sqlite3");

        let store1 = RunStore::open(&db_path).expect("first open");
        let v1 = store1.current_schema_version().expect("v1");
        drop(store1);

        let store2 = RunStore::open(&db_path).expect("second open");
        let v2 = store2.current_schema_version().expect("v2");

        assert_eq!(v1, v2, "version should be same after re-open");
        assert_eq!(v2, RunStore::SCHEMA_VERSION);
    }

    #[test]
    fn migration_preserves_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("preserve.sqlite3");

        // Create a DB manually (without _meta) that already has modern columns.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, \
                started_at TEXT NOT NULL, \
                finished_at TEXT NOT NULL, \
                backend TEXT NOT NULL, \
                input_path TEXT NOT NULL, \
                normalized_wav_path TEXT NOT NULL, \
                request_json TEXT NOT NULL, \
                result_json TEXT NOT NULL, \
                warnings_json TEXT NOT NULL, \
                transcript TEXT NOT NULL, \
                replay_json TEXT NOT NULL DEFAULT '{}', \
                acceleration_json TEXT NOT NULL DEFAULT '{}'\
            );",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT NOT NULL, \
                idx INTEGER NOT NULL, \
                start_sec REAL, \
                end_sec REAL, \
                speaker TEXT, \
                text TEXT NOT NULL, \
                confidence REAL, \
                PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT NOT NULL, \
                seq INTEGER NOT NULL, \
                ts_rfc3339 TEXT NOT NULL, \
                stage TEXT NOT NULL, \
                code TEXT NOT NULL, \
                message TEXT NOT NULL, \
                payload_json TEXT NOT NULL, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events");
        conn.execute(
            "INSERT INTO runs VALUES ('run-old', '2025-01-01T00:00:00Z', '2025-01-01T00:01:00Z', \
             'whisper_cpp', '/tmp/test.wav', '/tmp/norm.wav', '{}', \
             '{\"backend\":\"whisper_cpp\",\"transcript\":\"hello\",\"language\":\"en\",\"segments\":[],\"acceleration\":null,\"raw_output\":{},\"artifact_paths\":[]}', \
             '[]', 'hello', '{}', '{}');",
        )
        .expect("insert run");
        drop(conn);

        let store = RunStore::open(&db_path).expect("open after migration");
        let version = store.current_schema_version().expect("version");
        assert_eq!(version, RunStore::SCHEMA_VERSION, "should be at latest");

        let details = store
            .load_run_details("run-old")
            .expect("load")
            .expect("should exist");
        assert_eq!(details.run_id, "run-old");
        assert_eq!(details.transcript, "hello");
    }

    #[test]
    fn version_mismatch_detection() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("future.sqlite3");

        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");
        conn.execute_with_params(
            "INSERT INTO _meta (key, value) VALUES ('schema_version', ?1);",
            &[SqliteValue::Text("999".to_owned().into())],
        )
        .expect("insert future version");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, \
             finished_at TEXT NOT NULL, backend TEXT NOT NULL, input_path TEXT NOT NULL, \
             normalized_wav_path TEXT NOT NULL, request_json TEXT NOT NULL, result_json TEXT NOT NULL, \
             warnings_json TEXT NOT NULL, transcript TEXT NOT NULL, replay_json TEXT NOT NULL DEFAULT '{}', \
             acceleration_json TEXT NOT NULL DEFAULT '{}');",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS segments (run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
             start_sec REAL, end_sec REAL, speaker TEXT, text TEXT NOT NULL, confidence REAL, \
             PRIMARY KEY (run_id, idx));",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
             ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, code TEXT NOT NULL, \
             message TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, seq));",
        )
        .expect("create events");
        drop(conn);

        let err = RunStore::open(&db_path)
            .expect_err("should fail for future version")
            .to_string();
        assert!(
            err.contains("newer than supported"),
            "error should mention version mismatch: {err}"
        );
    }

    #[test]
    fn schema_version_exposed_as_constant() {
        let version = RunStore::SCHEMA_VERSION;
        assert!(version >= 1, "schema version should be at least 1");
    }

    #[test]
    fn apply_migration_unknown_version_returns_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("unknown_mig.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let result = store.apply_migration(999);
        assert!(result.is_err(), "unknown migration version should error");
        let err = result.expect_err("already checked is_err").to_string();
        assert!(
            err.contains("unknown migration version: 999"),
            "error message should identify version: {err}"
        );
    }

    #[test]
    fn ensure_column_exists_noop_when_column_already_present() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("col_exists.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // acceleration_json already exists after migration v2.
        // Calling ensure_column_exists again should be a no-op (not an error).
        let result =
            store.ensure_column_exists("runs", "acceleration_json", "TEXT NOT NULL DEFAULT '{}'");
        assert!(
            result.is_ok(),
            "ensure_column_exists should succeed when column already present: {:?}",
            result.err()
        );

        // Also verify the column is still there with correct data.
        let rows = store
            .connection
            .query("PRAGMA table_info(runs);")
            .expect("pragma");
        use super::value_to_string;
        let count = rows
            .iter()
            .filter(|row| value_to_string(row.get(1)) == "acceleration_json")
            .count();
        assert_eq!(count, 1, "column should appear exactly once");
    }

    #[test]
    fn current_schema_version_with_meta_but_no_version_key() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_version_key.sqlite3");

        // Create a bare DB with _meta table but no schema_version row.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");

        // Create the rest of the schema so initialize_schema CREATE IF NOT EXISTS succeeds.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, \
             finished_at TEXT NOT NULL, backend TEXT NOT NULL, input_path TEXT NOT NULL, \
             normalized_wav_path TEXT NOT NULL, request_json TEXT NOT NULL, result_json TEXT NOT NULL, \
             warnings_json TEXT NOT NULL, transcript TEXT NOT NULL, replay_json TEXT NOT NULL DEFAULT '{}', \
             acceleration_json TEXT NOT NULL DEFAULT '{}');",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS segments (run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
             start_sec REAL, end_sec REAL, speaker TEXT, text TEXT NOT NULL, confidence REAL, \
             PRIMARY KEY (run_id, idx));",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
             ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, code TEXT NOT NULL, \
             message TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, seq));",
        )
        .expect("create events");
        drop(conn);

        // Open with RunStore — it will see _meta exists but no schema_version key → Ok(0),
        // then run migrations from 0 to SCHEMA_VERSION.
        let store = RunStore::open(&db_path).expect("open should succeed");
        let version = store.current_schema_version().expect("version");
        assert_eq!(
            version,
            RunStore::SCHEMA_VERSION,
            "after open, should be migrated to latest"
        );
    }

    #[test]
    fn current_schema_version_with_non_numeric_value_returns_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("bad_version.sqlite3");

        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");
        conn.execute_with_params(
            "INSERT INTO _meta (key, value) VALUES ('schema_version', 'not_a_number');",
            &[],
        )
        .expect("insert bad version");
        // Create required tables so initialize_schema doesn't fail before hitting version check.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, \
             finished_at TEXT NOT NULL, backend TEXT NOT NULL, input_path TEXT NOT NULL, \
             normalized_wav_path TEXT NOT NULL, request_json TEXT NOT NULL, result_json TEXT NOT NULL, \
             warnings_json TEXT NOT NULL, transcript TEXT NOT NULL, replay_json TEXT NOT NULL DEFAULT '{}');",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS segments (run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
             start_sec REAL, end_sec REAL, speaker TEXT, text TEXT NOT NULL, confidence REAL, \
             PRIMARY KEY (run_id, idx));",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
             ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, code TEXT NOT NULL, \
             message TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, seq));",
        )
        .expect("create events");
        drop(conn);

        let result = RunStore::open(&db_path);
        assert!(
            result.is_err(),
            "non-numeric schema_version should cause open to fail"
        );
        let err = result.expect_err("already checked is_err").to_string();
        assert!(
            err.contains("invalid schema_version"),
            "error should mention invalid schema_version: {err}"
        );
    }

    #[test]
    fn set_schema_version_overwrites_existing_value() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("overwrite_ver.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // After open, version is SCHEMA_VERSION (currently 2).
        let v1 = store.current_schema_version().expect("v1");
        assert_eq!(v1, RunStore::SCHEMA_VERSION);

        // Overwrite to a different value.
        store.set_schema_version(42).expect("set to 42");
        let v2 = store.current_schema_version().expect("v2");
        assert_eq!(v2, 42, "version should be updated to 42");

        // Overwrite again to confirm the DELETE+INSERT pattern works repeatedly.
        store.set_schema_version(1).expect("set to 1");
        let v3 = store.current_schema_version().expect("v3");
        assert_eq!(v3, 1, "version should be updated to 1");
    }

    // -- bd-3i1.1: CancellationToken checkpoint tests --

    fn expired_token() -> crate::orchestrator::CancellationToken {
        let deadline = std::time::Duration::from_millis(0);
        let token = crate::orchestrator::CancellationToken::with_deadline_from_now(deadline); // ubs:ignore — test cancellation token
        std::thread::sleep(std::time::Duration::from_millis(5));
        token
    }

    fn live_token() -> crate::orchestrator::CancellationToken {
        crate::orchestrator::CancellationToken::with_deadline_from_now(
            std::time::Duration::from_secs(60),
        )
    }

    fn report_with_segments_and_events(
        run_id: &str,
        db_path: &std::path::Path,
        num_segments: usize,
        num_events: usize,
    ) -> RunReport {
        let mut report = minimal_report(run_id, db_path);
        report.result.segments = (0..num_segments)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some(i as f64 + 1.0),
                text: format!("segment {i}"),
                speaker: None,
                confidence: Some(0.9),
            })
            .collect();
        report.events = (0..num_events)
            .map(|i| RunEvent {
                seq: i as u64,
                ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                stage: "test".to_owned(),
                code: format!("test.{i}"),
                message: format!("event {i}"),
                payload: json!({"idx": i}),
            })
            .collect();
        report
    }

    #[test]
    fn cancel_token_persist_entry_checkpoint() {
        let token = expired_token(); // ubs:ignore — test cancellation token
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_entry.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let report = minimal_report("run-cx-entry", &db_path);
        let err = store
            .persist_report_cancellable(&report, Some(&token))
            .expect_err("should cancel");
        assert!(matches!(err, crate::error::FwError::Cancelled(_)));
        let runs = store.list_recent_runs(10).expect("query");
        assert!(runs.is_empty());
    }

    #[test]
    fn cancel_token_persist_live_succeeds() {
        let token = live_token(); // ubs:ignore — test cancellation token
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_live.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let report = report_with_segments_and_events("run-cx-live", &db_path, 5, 3);
        store
            .persist_report_cancellable(&report, Some(&token))
            .expect("should succeed");
        let details = store
            .load_run_details("run-cx-live")
            .expect("q")
            .expect("exists");
        assert_eq!(details.segments.len(), 5);
        assert_eq!(details.events.len(), 3);
    }

    #[test]
    fn cancel_token_persist_none_succeeds() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_none.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let report = report_with_segments_and_events("run-cx-none", &db_path, 3, 2);
        store.persist_report(&report).expect("should succeed");
        let details = store
            .load_run_details("run-cx-none")
            .expect("q")
            .expect("exists");
        assert_eq!(details.segments.len(), 3);
        assert_eq!(details.events.len(), 2);
    }

    #[test]
    fn cancel_token_persist_rollback() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_rollback.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let good = minimal_report("run-good", &db_path);
        store.persist_report(&good).expect("persist good");
        let token = expired_token(); // ubs:ignore — test cancellation token
        let bad = report_with_segments_and_events("run-bad", &db_path, 10, 5);
        assert!(
            store
                .persist_report_cancellable(&bad, Some(&token))
                .is_err()
        );
        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 1);
        assert!(store.load_run_details("run-bad").expect("q").is_none());
    }

    #[test]
    fn cancel_token_list_recent_runs_expired() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_list.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("r", &db_path))
            .expect("p");
        let token = expired_token(); // ubs:ignore — test cancellation token
        let err = store
            .list_recent_runs_cancellable(10, Some(&token))
            .expect_err("cancel");
        assert!(matches!(err, crate::error::FwError::Cancelled(_)));
    }

    #[test]
    fn cancel_token_list_recent_runs_live() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_list_live.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("r", &db_path))
            .expect("p");
        let token = live_token(); // ubs:ignore — test cancellation token
        let runs = store
            .list_recent_runs_cancellable(10, Some(&token))
            .expect("ok");
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn cancel_token_list_recent_runs_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_list_none.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("r", &db_path))
            .expect("p");
        let runs = store.list_recent_runs(10).expect("ok");
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn cancel_token_load_run_details_expired() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_load.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("r", &db_path))
            .expect("p");
        let token = expired_token(); // ubs:ignore — test cancellation token
        let err = store
            .load_run_details_cancellable("r", Some(&token))
            .expect_err("cancel");
        assert!(matches!(err, crate::error::FwError::Cancelled(_)));
    }

    #[test]
    fn cancel_token_load_run_details_live() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_load_live.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let report = report_with_segments_and_events("r", &db_path, 3, 2);
        store.persist_report(&report).expect("p");
        let token = live_token(); // ubs:ignore — test cancellation token
        let details = store
            .load_run_details_cancellable("r", Some(&token))
            .expect("ok")
            .expect("exists");
        assert_eq!(details.segments.len(), 3);
        assert_eq!(details.events.len(), 2);
    }

    #[test]
    fn cancel_token_load_run_details_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_load_none.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("r", &db_path))
            .expect("p");
        let details = store.load_run_details("r").expect("ok").expect("exists");
        assert_eq!(details.run_id, "r");
    }

    #[test]
    fn cancel_token_load_latest_expired() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_latest.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("r", &db_path))
            .expect("p");
        let token = expired_token(); // ubs:ignore — test cancellation token
        let err = store
            .load_latest_run_details_cancellable(Some(&token))
            .expect_err("cancel");
        assert!(matches!(err, crate::error::FwError::Cancelled(_)));
    }

    #[test]
    fn cancel_token_load_latest_live() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_latest_live.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("r", &db_path))
            .expect("p");
        let token = live_token(); // ubs:ignore — test cancellation token
        let details = store
            .load_latest_run_details_cancellable(Some(&token))
            .expect("ok")
            .expect("exists");
        assert_eq!(details.run_id, "r");
    }

    #[test]
    fn cancel_token_load_latest_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_latest_none.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        store
            .persist_report(&minimal_report("r", &db_path))
            .expect("p");
        let details = store
            .load_latest_run_details()
            .expect("ok")
            .expect("exists");
        assert_eq!(details.run_id, "r");
    }

    #[test]
    fn cancel_token_load_nonexistent_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_nonexist.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let token = live_token(); // ubs:ignore — test cancellation token
        let result = store
            .load_run_details_cancellable("nonexistent", Some(&token))
            .expect("ok");
        assert!(result.is_none());
    }

    #[test]
    fn cancel_token_load_latest_empty_db_returns_none() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_empty_latest.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let token = live_token(); // ubs:ignore — test cancellation token
        let result = store
            .load_latest_run_details_cancellable(Some(&token))
            .expect("ok");
        assert!(result.is_none());
    }

    #[test]
    fn cancel_token_public_api_unchanged() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_api.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let report = report_with_segments_and_events("r", &db_path, 4, 3);
        store.persist_report(&report).expect("persist");
        let details = store.load_run_details("r").expect("q").expect("exists");
        assert_eq!(details.segments.len(), 4);
        assert_eq!(details.events.len(), 3);
    }

    #[test]
    fn cancel_token_no_deadline_never_cancels() {
        let token = crate::orchestrator::CancellationToken::no_deadline(); // ubs:ignore — test cancellation token
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cx_nodeadline.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let report = report_with_segments_and_events("r", &db_path, 10, 10);
        store
            .persist_report_cancellable(&report, Some(&token))
            .expect("persist");
        let details = store
            .load_run_details_cancellable("r", Some(&token))
            .expect("q")
            .expect("exists");
        assert_eq!(details.segments.len(), 10);
        assert_eq!(details.events.len(), 10);
        let runs = store
            .list_recent_runs_cancellable(10, Some(&token))
            .expect("list");
        assert_eq!(runs.len(), 1);
        let latest = store
            .load_latest_run_details_cancellable(Some(&token))
            .expect("latest")
            .expect("exists");
        assert_eq!(latest.run_id, "r");
    }

    // -----------------------------------------------------------------------
    // bd-3pf.3: Storage roundtrip and concurrent persistence tests
    // -----------------------------------------------------------------------

    /// Build a fully populated RunReport (every optional field set) for
    /// exhaustive roundtrip verification.
    fn richly_populated_report(run_id: &str, db_path: &std::path::Path) -> RunReport {
        RunReport {
            run_id: run_id.to_owned(),
            trace_id: "aabbccdd00112233aabbccdd00112233".to_owned(),
            started_at_rfc3339: "2026-02-10T08:30:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-10T08:30:42Z".to_owned(),
            input_path: "/data/audio/interview.mp3".to_owned(),
            normalized_wav_path: "/tmp/fw/interview_norm.wav".to_owned(),
            request: TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("/data/audio/interview.mp3"),
                },
                backend: BackendKind::InsanelyFast,
                model: Some("large-v3".to_owned()),
                language: Some("en".to_owned()),
                translate: true,
                diarize: true,
                persist: true,
                db_path: db_path.to_path_buf(),
                timeout_ms: Some(60000),
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::InsanelyFast,
                transcript: "Hello, this is a test. 日本語テスト。".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![
                    TranscriptionSegment {
                        start_sec: Some(0.0),
                        end_sec: Some(2.5),
                        text: "Hello, this is a test.".to_owned(),
                        speaker: Some("SPEAKER_00".to_owned()),
                        confidence: Some(0.95),
                    },
                    TranscriptionSegment {
                        start_sec: Some(2.5),
                        end_sec: Some(5.0),
                        text: "日本語テスト。".to_owned(),
                        speaker: Some("SPEAKER_01".to_owned()),
                        confidence: Some(0.88),
                    },
                    TranscriptionSegment {
                        start_sec: None,
                        end_sec: None,
                        text: "segment without timestamps".to_owned(),
                        speaker: None,
                        confidence: None,
                    },
                ],
                acceleration: Some(AccelerationReport {
                    backend: AccelerationBackend::Frankentorch,
                    input_values: 42,
                    normalized_confidences: true,
                    pre_mass: Some(0.875),
                    post_mass: Some(1.0),
                    notes: vec![
                        "normalized via softmax".to_owned(),
                        "calibration applied".to_owned(),
                    ],
                }),
                raw_output: json!({"model": "large-v3", "ok": true}),
                artifact_paths: vec!["/tmp/fw/out.json".to_owned(), "/tmp/fw/out.srt".to_owned()],
            },
            events: vec![
                RunEvent {
                    seq: 1,
                    ts_rfc3339: "2026-02-10T08:30:00Z".to_owned(),
                    stage: "ingest".to_owned(),
                    code: "ingest.start".to_owned(),
                    message: "ingesting file".to_owned(),
                    payload: json!({"path": "/data/audio/interview.mp3"}),
                },
                RunEvent {
                    seq: 2,
                    ts_rfc3339: "2026-02-10T08:30:01Z".to_owned(),
                    stage: "normalize".to_owned(),
                    code: "normalize.complete".to_owned(),
                    message: "normalized to WAV".to_owned(),
                    payload: json!({"duration_ms": 5000}),
                },
                RunEvent {
                    seq: 3,
                    ts_rfc3339: "2026-02-10T08:30:40Z".to_owned(),
                    stage: "backend".to_owned(),
                    code: "backend.complete".to_owned(),
                    message: "transcription done".to_owned(),
                    payload: json!({"segments": 3, "nested": {"key": [1, 2, 3]}}),
                },
            ],
            warnings: vec![
                "low confidence on segment 3".to_owned(),
                "résumé café — unicode warning".to_owned(),
            ],
            evidence: vec![json!({"contract": "backend_selection", "action": "try_insanely_fast"})],
            replay: crate::model::ReplayEnvelope {
                input_content_hash: Some("sha256_abc123def456".to_owned()),
                backend_identity: Some("insanely-fast-whisper".to_owned()),
                backend_version: Some("0.0.15".to_owned()),
                output_payload_hash: Some("sha256_789xyz".to_owned()),
            },
        }
    }

    #[test]
    fn roundtrip_comprehensive_all_fields() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("roundtrip_all.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = richly_populated_report("run-roundtrip", &db_path);
        store.persist_report(&report).expect("persist");

        // Verify via load_run_details.
        let d = store
            .load_run_details("run-roundtrip")
            .expect("query")
            .expect("should exist");

        assert_eq!(d.run_id, "run-roundtrip");
        assert_eq!(d.started_at_rfc3339, "2026-02-10T08:30:00Z");
        assert_eq!(d.finished_at_rfc3339, "2026-02-10T08:30:42Z");
        assert_eq!(d.backend, BackendKind::InsanelyFast);
        assert!(d.transcript.contains("Hello"));
        assert!(d.transcript.contains("日本語"));

        // Segments round-trip (loaded from result_json).
        assert_eq!(d.segments.len(), 3);
        assert_eq!(d.segments[0].start_sec, Some(0.0));
        assert_eq!(d.segments[0].end_sec, Some(2.5));
        assert_eq!(d.segments[0].text, "Hello, this is a test.");
        assert_eq!(d.segments[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(d.segments[0].confidence, Some(0.95));
        assert_eq!(d.segments[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(d.segments[1].confidence, Some(0.88));
        assert!(d.segments[2].start_sec.is_none());
        assert!(d.segments[2].speaker.is_none());
        assert!(d.segments[2].confidence.is_none());

        // Events round-trip.
        assert_eq!(d.events.len(), 3);
        assert_eq!(d.events[0].seq, 1);
        assert_eq!(d.events[0].stage, "ingest");
        assert_eq!(d.events[0].code, "ingest.start");
        assert_eq!(d.events[0].payload["path"], "/data/audio/interview.mp3");
        assert_eq!(d.events[1].seq, 2);
        assert_eq!(d.events[2].seq, 3);
        assert_eq!(d.events[2].payload["segments"], 3);
        // Nested payload survives.
        assert_eq!(d.events[2].payload["nested"]["key"][1], 2);

        // Warnings.
        assert_eq!(d.warnings.len(), 2);
        assert_eq!(d.warnings[0], "low confidence on segment 3");
        assert!(d.warnings[1].contains("résumé"));

        // Acceleration.
        let accel = d.acceleration.expect("acceleration should exist");
        assert_eq!(accel.backend, AccelerationBackend::Frankentorch);
        assert_eq!(accel.input_values, 42);
        assert!(accel.normalized_confidences);
        assert_eq!(accel.pre_mass, Some(0.875));
        assert_eq!(accel.post_mass, Some(1.0));
        assert_eq!(accel.notes.len(), 2);

        // Replay envelope.
        assert_eq!(
            d.replay.input_content_hash.as_deref(),
            Some("sha256_abc123def456")
        );
        assert_eq!(
            d.replay.backend_identity.as_deref(),
            Some("insanely-fast-whisper")
        );
        assert_eq!(d.replay.backend_version.as_deref(), Some("0.0.15"));
        assert_eq!(
            d.replay.output_payload_hash.as_deref(),
            Some("sha256_789xyz")
        );

        // Verify via list_recent_runs.
        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_id, "run-roundtrip");
        assert_eq!(runs[0].backend, BackendKind::InsanelyFast);
        assert!(runs[0].transcript_preview.contains("Hello"));

        // Verify via load_latest_run_details.
        let latest = store
            .load_latest_run_details()
            .expect("latest query")
            .expect("should exist");
        assert_eq!(latest.run_id, "run-roundtrip");
    }

    /// Persist a report with retry logic for SQLite busy/transaction conflicts
    /// that occur during concurrent multi-connection access.
    fn persist_with_retry(store: &RunStore, report: &RunReport, max_retries: u32) {
        for attempt in 0..max_retries {
            match store.persist_report(report) {
                Ok(()) => return,
                Err(_) if attempt < max_retries - 1 => {
                    std::thread::sleep(std::time::Duration::from_millis(50 * (attempt as u64 + 1)));
                }
                Err(e) => {
                    panic!("persist failed after {max_retries} retries: {e}"); // ubs:ignore — test panic on unexpected persistence failure
                }
            }
        }
    }

    #[test]
    fn concurrent_persist_five_threads() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("concurrent_5.sqlite3");

        // Pre-create the store so schema is initialized.
        let store = RunStore::open(&db_path).expect("store");
        drop(store);

        // Threads race to persist concurrently. `persist_with_retry` absorbs
        // the benign `SQLITE_BUSY` returns that any SQLite implementation
        // emits transiently under contention. The underlying MVCC
        // concurrent-persistence support in `fsqlite` is correct, so we
        // expect ALL five runs to be durable (no fake-serialization mutex,
        // and the assertion below is exact).
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(5));
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let path = db_path.clone();
                let b = barrier.clone();
                std::thread::spawn(move || {
                    let report = minimal_report(&format!("concurrent-{i}"), &path);
                    b.wait(); // Maximize true concurrency at persist time.
                    let store = RunStore::open(&path).expect("thread store");
                    persist_with_retry(&store, &report, 10);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("thread should not panic");
        }

        // Re-open and verify all 5 runs are present.
        let store = RunStore::open(&db_path).expect("store");
        let runs = store.list_recent_runs(100).expect("list");
        assert_eq!(
            runs.len(),
            5,
            "all 5 concurrent runs must be durable, got {}",
            runs.len(),
        );

        // Verify each run can be loaded individually.
        for i in 0..5 {
            let run_id = format!("concurrent-{i}");
            let details = store
                .load_run_details(&run_id)
                .expect("query")
                .expect("should exist");
            assert_eq!(details.run_id, run_id);
        }
    }

    #[test]
    fn concurrent_persist_10_threads_with_segments_and_events() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("concurrent_10.sqlite3");

        let store = RunStore::open(&db_path).expect("store");
        drop(store);

        // Each thread retries the full open+persist cycle to absorb the
        // benign "database is busy" return that any SQLite implementation can
        // emit briefly under concurrent writer startup. The underlying MVCC
        // concurrent-persistence support in `fsqlite` is now correct, so this
        // test requires **every** writer thread to land its run durably.
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let path = db_path.clone();
                std::thread::spawn(move || {
                    let mut report = minimal_report(&format!("c10-{i}"), &path);
                    report.result.transcript = format!("transcript for thread {i}");
                    report.result.segments = (0..5)
                        .map(|s| TranscriptionSegment {
                            start_sec: Some(s as f64),
                            end_sec: Some(s as f64 + 1.0),
                            text: format!("t{i}-seg{s}"),
                            speaker: None,
                            confidence: Some(0.5),
                        })
                        .collect();
                    report.events = (0..3)
                        .map(|e| RunEvent {
                            seq: e as u64,
                            ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                            stage: "test".to_owned(),
                            code: format!("t{i}.e{e}"),
                            message: format!("event {e}"),
                            payload: json!({"thread": i, "event": e}),
                        })
                        .collect();
                    for attempt in 0..30u32 {
                        let store = match RunStore::open(&path) {
                            Ok(s) => s,
                            Err(_) if attempt < 29 => {
                                std::thread::sleep(std::time::Duration::from_millis(
                                    30 * (attempt as u64 + 1),
                                ));
                                continue;
                            }
                            Err(e) => {
                                // Open exhausted the retry budget. We bubble this up via the
                                // `false` thread-return so the outer aggregate assert names
                                // exactly which thread couldn't make progress. The aggregate
                                // assert requires all 10 threads to succeed; this branch only
                                // exists to surface a precise diagnostic instead of a panic
                                // from an unwrap deep inside the retry loop.
                                if super::is_busy_storage_error(&e) {
                                    return false;
                                }
                                panic!("c10-{i} open failed after retries: {e}"); // ubs:ignore — test panic on unexpected open failure
                            }
                        };
                        match store.persist_report(&report) {
                            Ok(()) => return true,
                            Err(_) if attempt < 29 => {
                                std::thread::sleep(std::time::Duration::from_millis(
                                    30 * (attempt as u64 + 1),
                                ));
                            }
                            Err(e) => {
                                // Persist exhausted the retry budget. As with the open path
                                // above, this is surfaced as a `false` thread-return so the
                                // aggregate assertion below names which thread didn't land its
                                // run durably; durability itself is required for ALL threads.
                                if super::is_busy_storage_error(&e) {
                                    return false;
                                }
                                panic!("c10-{i} persist failed after retries: {e}"); // ubs:ignore — test panic on unexpected persist failure
                            }
                        }
                    }
                    false
                })
            })
            .collect();

        let mut successful_threads = 0usize;
        for handle in handles {
            let thread_succeeded = handle.join().expect("thread should not panic");
            if thread_succeeded {
                successful_threads += 1;
            }
        }

        // Force WAL checkpoint so all committed data is visible to new readers.
        {
            let checkpoint_store = RunStore::open(&db_path).expect("checkpoint store");
            let _ = checkpoint_store
                .connection
                .query("PRAGMA wal_checkpoint(TRUNCATE);");
        }

        let mut final_runs_count = 0;
        for _attempt in 0..20 {
            let store = RunStore::open(&db_path).expect("store");
            let runs = store.list_recent_runs(100).expect("list");
            final_runs_count = runs.len();
            if final_runs_count == 10 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Durability requirement: every one of the 10 concurrent persists
        // must be observable from a fresh reader. Anything less is a
        // regression in the storage layer (formerly tolerated as an
        // `fsqlite` MVCC limitation that has since been fixed upstream;
        // see `docs/operational-playbook.md` for the history).
        assert_eq!(
            final_runs_count, 10,
            "all 10 concurrent runs must persist; got {}. Listing the run IDs that DID land helps name the dropped writes.",
            final_runs_count
        );
        assert_eq!(
            successful_threads, 10,
            "all 10 writer threads must report success; got {}. A `false` thread-return means that thread exhausted the open/persist retry budget — investigate host load + the fsqlite busy-window behavior.",
            successful_threads
        );

        // Verify every run that DID persist has intact segments and events.
        let store = RunStore::open(&db_path).expect("store");
        for i in 0..10 {
            let run_id = format!("c10-{i}");
            if let Some(details) = store.load_run_details(&run_id).expect("query") {
                assert_eq!(
                    details.segments.len(),
                    5,
                    "run {run_id} should have 5 segments"
                );
                assert_eq!(details.events.len(), 3, "run {run_id} should have 3 events");
                // Verify event ordering is monotonic.
                for (idx, event) in details.events.iter().enumerate() {
                    assert_eq!(event.seq, idx as u64, "event seq should be monotonic");
                }
            }
        }
    }

    #[test]
    fn transaction_rollback_leaves_no_partial_data() {
        use crate::orchestrator::CancellationToken;

        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("rollback.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // Create an already-expired token (0ms deadline, then sleep past it).
        let token = CancellationToken::with_deadline_from_now(std::time::Duration::from_millis(0)); // ubs:ignore — test cancellation token
        std::thread::sleep(std::time::Duration::from_millis(5));

        let mut report = richly_populated_report("run-rollback", &db_path);
        report.result.segments = (0..50)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some(i as f64 + 1.0),
                text: format!("segment {i}"),
                speaker: None,
                confidence: Some(0.5),
            })
            .collect();

        // Persist with expired token — should fail and rollback.
        let result = store.persist_report_cancellable(&report, Some(&token));
        assert!(result.is_err(), "expired token should cause failure");

        // Verify no partial data remains.
        let runs = store.list_recent_runs(100).expect("list");
        assert!(
            runs.is_empty(),
            "no runs should exist after rollback, found {}",
            runs.len()
        );

        let details = store.load_run_details("run-rollback").expect("query");
        assert!(
            details.is_none(),
            "rollback should leave no partial run data"
        );

        // Verify the DB is still usable after the failed transaction.
        let good_report = minimal_report("run-after-rollback", &db_path);
        store
            .persist_report(&good_report)
            .expect("should persist after rollback");
        let runs = store.list_recent_runs(100).expect("list after");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_id, "run-after-rollback");
    }

    #[test]
    fn sync_export_import_roundtrip_matches() {
        let dir = tempdir().expect("tempdir");
        let source_db = dir.path().join("source.sqlite3");
        let export_dir = dir.path().join("export");
        let target_db = dir.path().join("target.sqlite3");
        let state_root = dir.path().join("state");

        // Populate source DB.
        let store = RunStore::open(&source_db).expect("source store");
        for i in 0..3 {
            let mut report = richly_populated_report(&format!("sync-{i}"), &source_db);
            report.started_at_rfc3339 = format!("2026-01-0{}T00:00:00Z", i + 1);
            report.result.transcript = format!("transcript {i}");
            store.persist_report(&report).expect("persist");
        }
        drop(store);

        // Export.
        let manifest = crate::sync::export(&source_db, &export_dir, &state_root).expect("export");
        assert_eq!(manifest.row_counts.runs, 3);
        assert!(manifest.row_counts.segments > 0);
        assert!(manifest.row_counts.events > 0);

        // Import into fresh DB.
        let result = crate::sync::import(
            &target_db,
            &export_dir,
            &state_root,
            crate::sync::ConflictPolicy::Reject,
        )
        .expect("import");
        assert_eq!(result.runs_imported, 3);
        assert!(result.conflicts.is_empty());

        // Compare source and target.
        let source = RunStore::open(&source_db).expect("reopen source");
        let target = RunStore::open(&target_db).expect("open target");

        let source_runs = source.list_recent_runs(100).expect("source list");
        let target_runs = target.list_recent_runs(100).expect("target list");
        assert_eq!(source_runs.len(), target_runs.len(), "same number of runs");

        // Verify each run matches.
        for i in 0..3 {
            let run_id = format!("sync-{i}");
            let s = source
                .load_run_details(&run_id)
                .expect("source query")
                .expect("exists");
            let t = target
                .load_run_details(&run_id)
                .expect("target query")
                .expect("exists");

            assert_eq!(s.run_id, t.run_id);
            assert_eq!(s.started_at_rfc3339, t.started_at_rfc3339);
            assert_eq!(s.finished_at_rfc3339, t.finished_at_rfc3339);
            assert_eq!(s.backend, t.backend);
            assert_eq!(s.transcript, t.transcript);
            assert_eq!(s.segments.len(), t.segments.len());
            assert_eq!(s.events.len(), t.events.len());
            assert_eq!(s.warnings.len(), t.warnings.len());

            // Compare segments.
            for (ss, ts) in s.segments.iter().zip(t.segments.iter()) {
                assert_eq!(ss.text, ts.text);
                assert_eq!(ss.start_sec, ts.start_sec);
                assert_eq!(ss.end_sec, ts.end_sec);
                assert_eq!(ss.speaker, ts.speaker);
                assert_eq!(ss.confidence, ts.confidence);
            }

            // Compare events.
            for (se, te) in s.events.iter().zip(t.events.iter()) {
                assert_eq!(se.seq, te.seq);
                assert_eq!(se.stage, te.stage);
                assert_eq!(se.code, te.code);
                assert_eq!(se.message, te.message);
                assert_eq!(se.payload, te.payload);
            }
        }
    }

    #[test]
    fn stress_many_runs_no_corruption() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("stress_many.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        for i in 0..30 {
            let mut report = minimal_report(&format!("stress-{i:03}"), &db_path);
            report.started_at_rfc3339 = format!("2026-01-01T{:02}:{:02}:00Z", i / 60, i % 60);
            report.result.transcript = format!("run number {i}");
            report.result.segments = (0..5)
                .map(|s| TranscriptionSegment {
                    start_sec: Some(s as f64),
                    end_sec: Some(s as f64 + 1.0),
                    text: format!("r{i}-s{s}"),
                    speaker: if s % 2 == 0 {
                        Some("SPEAKER_00".to_owned())
                    } else {
                        None
                    },
                    confidence: Some(0.5),
                })
                .collect();
            report.events = vec![RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                stage: "test".to_owned(),
                code: format!("stress.{i}"),
                message: format!("run {i}"),
                payload: json!({"run": i}),
            }];
            store.persist_report(&report).expect("persist");
        }

        // List all.
        let runs = store.list_recent_runs(200).expect("list");
        assert_eq!(runs.len(), 30, "all stress runs should be listed");

        // Verify list respects limit.
        let limited = store.list_recent_runs(10).expect("limited list");
        assert_eq!(limited.len(), 10);

        // Spot-check runs at boundaries.
        for idx in [0, 14, 29] {
            let run_id = format!("stress-{idx:03}");
            let details = store
                .load_run_details(&run_id)
                .expect("query")
                .expect("should exist");
            assert_eq!(details.run_id, run_id);
            assert_eq!(details.segments.len(), 5);
            assert_eq!(details.events.len(), 1);
            assert_eq!(details.events[0].code, format!("stress.{idx}"));
        }

        // Verify latest run (highest timestamp).
        let latest = store
            .load_latest_run_details()
            .expect("latest")
            .expect("exists");
        assert_eq!(
            latest.run_id, "stress-029",
            "latest run should have the highest timestamp"
        );
    }

    #[test]
    fn persist_then_list_includes_new_run() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("persist_list.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // Verify empty initially.
        let before = store.list_recent_runs(100).expect("before");
        assert!(before.is_empty());

        // Persist one.
        let report = minimal_report("new-run", &db_path);
        store.persist_report(&report).expect("persist");

        // Verify it shows up.
        let after = store.list_recent_runs(100).expect("after");
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].run_id, "new-run");
    }

    #[test]
    fn persist_events_then_get_events_returns_in_order() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("events_order.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-evt-order", &db_path);
        // Insert events in reverse seq order to test ORDER BY.
        report.events = vec![
            RunEvent {
                seq: 5,
                ts_rfc3339: "2026-01-01T00:00:05Z".to_owned(),
                stage: "persist".to_owned(),
                code: "e5".to_owned(),
                message: "fifth".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "e1".to_owned(),
                message: "first".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 3,
                ts_rfc3339: "2026-01-01T00:00:03Z".to_owned(),
                stage: "backend".to_owned(),
                code: "e3".to_owned(),
                message: "third".to_owned(),
                payload: json!({}),
            },
        ];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-evt-order")
            .expect("query")
            .expect("exists");

        // Events must be in ascending seq order regardless of insertion order.
        assert_eq!(details.events.len(), 3);
        assert_eq!(details.events[0].seq, 1);
        assert_eq!(details.events[0].message, "first");
        assert_eq!(details.events[1].seq, 3);
        assert_eq!(details.events[1].message, "third");
        assert_eq!(details.events[2].seq, 5);
        assert_eq!(details.events[2].message, "fifth");

        // Verify monotonic ordering.
        for pair in details.events.windows(2) {
            assert!(
                pair[0].seq < pair[1].seq,
                "events must be monotonically ordered: {} < {}",
                pair[0].seq,
                pair[1].seq
            );
        }
    }

    // -----------------------------------------------------------------------
    // bd-3i1.2: Concurrent session (SAVEPOINT) tests
    // -----------------------------------------------------------------------

    #[test]
    fn concurrent_session_commit_persists_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_commit.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // Start a top-level transaction first.
        store
            .connection
            .execute("BEGIN;")
            .expect("begin top-level txn");

        let session = store
            .begin_concurrent_session("alpha")
            .expect("begin session");
        session
            .execute_with_params(
                "INSERT INTO _meta (key, value) VALUES (?1, ?2);",
                &[
                    SqliteValue::Text("session_test".to_owned().into()),
                    SqliteValue::Text("alpha_value".to_owned().into()),
                ],
            )
            .expect("insert via session");
        session.commit().expect("commit session");

        store
            .connection
            .execute("COMMIT;")
            .expect("commit top-level txn");

        // Verify data persisted.
        let rows = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'session_test';")
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(super::value_to_string(rows[0].get(0)), "alpha_value");
    }

    #[test]
    fn concurrent_session_rollback_discards_data() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_rollback.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        store
            .connection
            .execute("BEGIN;")
            .expect("begin top-level txn");

        let session = store
            .begin_concurrent_session("beta")
            .expect("begin session");
        session
            .execute_with_params(
                "INSERT INTO _meta (key, value) VALUES (?1, ?2);",
                &[
                    SqliteValue::Text("rollback_test".to_owned().into()),
                    SqliteValue::Text("should_vanish".to_owned().into()),
                ],
            )
            .expect("insert via session");
        session.rollback().expect("rollback session");

        store
            .connection
            .execute("COMMIT;")
            .expect("commit top-level txn");

        // Verify data was rolled back.
        let rows = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'rollback_test';")
            .expect("query");
        assert!(
            rows.is_empty(),
            "rolled-back session data should not be visible"
        );
    }

    #[test]
    fn concurrent_session_drop_without_commit_rolls_back() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_drop.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        store
            .connection
            .execute("BEGIN;")
            .expect("begin top-level txn");

        {
            let session = store
                .begin_concurrent_session("gamma")
                .expect("begin session");
            session
                .execute_with_params(
                    "INSERT INTO _meta (key, value) VALUES (?1, ?2);",
                    &[
                        SqliteValue::Text("drop_test".to_owned().into()),
                        SqliteValue::Text("should_vanish".to_owned().into()),
                    ],
                )
                .expect("insert via session");
            // Session dropped without commit or explicit rollback.
        }

        store
            .connection
            .execute("COMMIT;")
            .expect("commit top-level txn");

        let rows = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'drop_test';")
            .expect("query");
        assert!(
            rows.is_empty(),
            "dropped session data should not be visible"
        );
    }

    #[test]
    fn concurrent_session_nested_savepoints_work() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_nested.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        store
            .connection
            .execute("BEGIN;")
            .expect("begin top-level txn");

        // Create two sessions (nested savepoints).
        let session_a = store
            .begin_concurrent_session("outer")
            .expect("begin outer");
        session_a
            .execute_with_params(
                "INSERT INTO _meta (key, value) VALUES (?1, ?2);",
                &[
                    SqliteValue::Text("nested_outer".to_owned().into()),
                    SqliteValue::Text("outer_value".to_owned().into()),
                ],
            )
            .expect("insert outer");

        let session_b = store
            .begin_concurrent_session("inner")
            .expect("begin inner");
        session_b
            .execute_with_params(
                "INSERT INTO _meta (key, value) VALUES (?1, ?2);",
                &[
                    SqliteValue::Text("nested_inner".to_owned().into()),
                    SqliteValue::Text("inner_value".to_owned().into()),
                ],
            )
            .expect("insert inner");

        // Roll back inner, commit outer.
        session_b.rollback().expect("rollback inner");
        session_a.commit().expect("commit outer");

        store
            .connection
            .execute("COMMIT;")
            .expect("commit top-level txn");

        // Outer data should be present, inner data should be gone.
        let outer = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'nested_outer';")
            .expect("query outer");
        assert_eq!(outer.len(), 1, "outer session data should persist");

        let inner = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'nested_inner';")
            .expect("query inner");
        assert!(inner.is_empty(), "inner session data should be rolled back");
    }

    #[test]
    fn persist_report_inside_concurrent_session_preserves_outer_savepoint() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_persist_nested.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let session = store
            .begin_concurrent_session("outerpersist")
            .expect("begin session");
        session
            .execute_with_params(
                "INSERT INTO _meta (key, value) VALUES (?1, ?2);",
                &[
                    SqliteValue::Text("outerpersist_key".to_owned().into()),
                    SqliteValue::Text("outerpersist_value".to_owned().into()),
                ],
            )
            .expect("insert via outer session");

        store
            .persist_report(&minimal_report("nested-persist", &db_path))
            .expect("persist within active outer savepoint");

        session
            .commit()
            .expect("outer savepoint should remain valid");

        let meta_rows = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'outerpersist_key';")
            .expect("query meta");
        assert_eq!(meta_rows.len(), 1, "outer session data should survive");
        assert_eq!(value_to_string(meta_rows[0].get(0)), "outerpersist_value");

        let details = store
            .load_run_details("nested-persist")
            .expect("load")
            .expect("persisted run should exist");
        assert_eq!(details.run_id, "nested-persist");
    }

    #[test]
    fn concurrent_session_invalid_name_rejected() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_bad_name.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let err = store
            .begin_concurrent_session("")
            .expect_err("empty name should fail");
        assert!(
            err.to_string().contains("invalid session name"),
            "error should mention invalid name: {}",
            err
        );

        let err2 = store
            .begin_concurrent_session("a; DROP TABLE runs;")
            .expect_err("sql injection name should fail");
        assert!(
            err2.to_string().contains("invalid session name"),
            "error should mention invalid name: {}",
            err2
        );
    }

    #[test]
    fn concurrent_session_query_within_session() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_query.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        store.connection.execute("BEGIN;").expect("begin txn");

        let session = store
            .begin_concurrent_session("query_test")
            .expect("begin session");

        session
            .execute_with_params(
                "INSERT INTO _meta (key, value) VALUES (?1, ?2);",
                &[
                    SqliteValue::Text("query_key".to_owned().into()),
                    SqliteValue::Text("query_value".to_owned().into()),
                ],
            )
            .expect("insert");

        // Query within session should see the uncommitted data.
        let rows = session
            .query("SELECT value FROM _meta WHERE key = 'query_key';")
            .expect("query within session");
        assert_eq!(rows.len(), 1);
        assert_eq!(super::value_to_string(rows[0].get(0)), "query_value");

        session.commit().expect("commit");
        store.connection.execute("COMMIT;").expect("commit txn");
    }

    // -----------------------------------------------------------------------
    // bd-3i1.3: Storage diagnostics tests
    // -----------------------------------------------------------------------

    #[test]
    fn diagnostics_returns_valid_snapshot() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("diag.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let diag = store.diagnostics().expect("diagnostics");

        assert!(diag.page_count > 0, "page_count should be positive");
        assert!(diag.page_size > 0, "page_size should be positive");
        assert_eq!(
            diag.journal_mode.to_lowercase(),
            "wal",
            "journal_mode should be WAL"
        );
        assert!(
            diag.freelist_count >= 0,
            "freelist_count should be non-negative"
        );
        assert_eq!(
            diag.integrity_check, "ok",
            "integrity_check should pass on fresh DB"
        );
    }

    #[test]
    fn diagnostics_after_data_insertion() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("diag_data.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let diag_before = store.diagnostics().expect("diagnostics before");

        // Insert some data.
        for i in 0..10 {
            store
                .persist_report(&minimal_report(&format!("diag-{i}"), &db_path))
                .expect("persist");
        }

        let diag_after = store.diagnostics().expect("diagnostics after");

        assert!(
            diag_after.page_count >= diag_before.page_count,
            "page_count should not decrease after insertions"
        );
        assert_eq!(
            diag_after.integrity_check, "ok",
            "integrity should pass after insertions"
        );
    }

    #[test]
    fn diagnostics_page_size_is_power_of_two() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("diag_ps.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let diag = store.diagnostics().expect("diagnostics");
        assert!(
            diag.page_size > 0 && (diag.page_size & (diag.page_size - 1)) == 0,
            "page_size should be a power of 2, got {}",
            diag.page_size
        );
    }

    #[test]
    fn diagnostics_wal_checkpoint_fields_non_negative() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("diag_wal.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let diag = store.diagnostics().expect("diagnostics");
        assert!(diag.wal_checkpoint.busy >= 0, "busy should be non-negative");
        assert!(
            diag.wal_checkpoint.log_frames >= 0,
            "log_frames should be non-negative"
        );
        assert!(
            diag.wal_checkpoint.checkpointed_frames >= 0,
            "checkpointed_frames should be non-negative"
        );
    }

    #[test]
    fn diagnostics_on_empty_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("diag_empty.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let diag = store.diagnostics().expect("diagnostics");
        assert_eq!(diag.integrity_check, "ok");
        assert!(diag.page_count > 0, "even empty DB has pages for schema");
    }

    // -----------------------------------------------------------------------
    // bd-3i1.5: Index optimization tests
    // -----------------------------------------------------------------------

    #[test]
    fn migration_v3_creates_indexes() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("idx.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let indexes = store
            .connection
            .query("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;")
            .expect("query indexes");
        let index_names: Vec<String> = indexes
            .iter()
            .map(|row| super::value_to_string(row.get(0)))
            .collect();

        assert!(
            index_names.contains(&"idx_runs_started_at".to_owned()),
            "idx_runs_started_at should exist, found: {index_names:?}"
        );
        assert!(
            index_names.contains(&"idx_runs_backend".to_owned()),
            "idx_runs_backend should exist, found: {index_names:?}"
        );
        assert!(
            index_names.contains(&"idx_segments_run_id".to_owned()),
            "idx_segments_run_id should exist, found: {index_names:?}"
        );
        assert!(
            index_names.contains(&"idx_events_run_id_stage".to_owned()),
            "idx_events_run_id_stage should exist, found: {index_names:?}"
        );
    }

    #[test]
    fn migration_v3_indexes_idempotent_on_reopen() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("idx_reopen.sqlite3");

        let store1 = RunStore::open(&db_path).expect("first open");
        drop(store1);

        // Second open should not fail even though indexes already exist.
        let store2 = RunStore::open(&db_path).expect("second open");
        let indexes = store2
            .connection
            .query(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY name;",
            )
            .expect("query indexes");
        assert!(
            indexes.len() >= 4,
            "all 4 indexes should exist after reopen, found {}",
            indexes.len()
        );
    }

    #[test]
    fn migration_v3_indexes_from_v2_db() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("v2_to_v3.sqlite3");

        // Manually create a v2 database.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, \
                started_at TEXT NOT NULL, \
                finished_at TEXT NOT NULL, \
                backend TEXT NOT NULL, \
                input_path TEXT NOT NULL, \
                normalized_wav_path TEXT NOT NULL, \
                request_json TEXT NOT NULL, \
                result_json TEXT NOT NULL, \
                warnings_json TEXT NOT NULL, \
                transcript TEXT NOT NULL, \
                replay_json TEXT NOT NULL DEFAULT '{}', \
                acceleration_json TEXT NOT NULL DEFAULT '{}'\
            );",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT NOT NULL, \
                idx INTEGER NOT NULL, \
                start_sec REAL, \
                end_sec REAL, \
                speaker TEXT, \
                text TEXT NOT NULL, \
                confidence REAL, \
                PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT NOT NULL, \
                seq INTEGER NOT NULL, \
                ts_rfc3339 TEXT NOT NULL, \
                stage TEXT NOT NULL, \
                code TEXT NOT NULL, \
                message TEXT NOT NULL, \
                payload_json TEXT NOT NULL, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");
        conn.execute("INSERT INTO _meta (key, value) VALUES ('schema_version', '2');")
            .expect("set version 2");
        drop(conn);

        // Open with RunStore, which should migrate from v2 to v3.
        let store = RunStore::open(&db_path).expect("open should migrate");
        let version = store.current_schema_version().expect("version");
        assert_eq!(version, 3, "should have migrated to v3");

        // Verify indexes exist.
        let indexes = store
            .connection
            .query("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';")
            .expect("query indexes");
        let names: Vec<String> = indexes
            .iter()
            .map(|row| super::value_to_string(row.get(0)))
            .collect();
        assert!(
            names.contains(&"idx_runs_started_at".to_owned()),
            "idx_runs_started_at should exist after v2->v3 migration"
        );
        assert!(
            names.contains(&"idx_events_run_id_stage".to_owned()),
            "idx_events_run_id_stage should exist after v2->v3 migration"
        );
    }

    #[test]
    fn schema_version_is_3() {
        assert_eq!(
            RunStore::SCHEMA_VERSION,
            3,
            "SCHEMA_VERSION should be 3 after bd-3i1.5"
        );
    }

    #[test]
    fn indexes_do_not_break_existing_queries() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("idx_queries.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // Persist some data and verify all existing query methods still work.
        for i in 0..5 {
            let mut report = minimal_report(&format!("idx-q-{i}"), &db_path);
            report.started_at_rfc3339 = format!("2026-01-0{}T00:00:00Z", i + 1);
            report.result.backend = if i % 2 == 0 {
                BackendKind::WhisperCpp
            } else {
                BackendKind::InsanelyFast
            };
            store.persist_report(&report).expect("persist");
        }

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 5, "all runs should be listed");

        let latest = store
            .load_latest_run_details()
            .expect("latest")
            .expect("exists");
        assert_eq!(latest.run_id, "idx-q-4", "latest run by timestamp");

        let details = store
            .load_run_details("idx-q-2")
            .expect("load")
            .expect("exists");
        assert_eq!(details.run_id, "idx-q-2");
    }

    #[test]
    fn is_busy_storage_error_recognizes_all_busy_patterns() {
        use super::is_busy_storage_error;
        use crate::error::FwError;

        // Positive cases — should return true
        assert!(is_busy_storage_error(&FwError::Storage(
            "database is busy".to_owned()
        )));
        assert!(is_busy_storage_error(&FwError::Storage(
            "Database Is Busy (timeout)".to_owned()
        )));
        assert!(is_busy_storage_error(&FwError::Storage(
            "snapshot conflict detected".to_owned()
        )));
        assert!(is_busy_storage_error(&FwError::Storage(
            "database is locked".to_owned()
        )));

        // Negative cases — should return false
        assert!(!is_busy_storage_error(&FwError::Storage(
            "table not found".to_owned()
        )));
        assert!(!is_busy_storage_error(&FwError::InvalidRequest(
            "database is busy".to_owned()
        )));
        assert!(!is_busy_storage_error(&FwError::Cancelled(
            "cancelled".to_owned()
        )));
    }

    #[test]
    fn value_to_i64_all_variants() {
        use super::value_to_i64;

        assert_eq!(value_to_i64(Some(&SqliteValue::Integer(42))), 42);
        assert_eq!(value_to_i64(Some(&SqliteValue::Integer(-1))), -1);
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Integer(i64::MAX))),
            i64::MAX
        );
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Text("123".to_owned().into()))),
            123
        );
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Text("not_a_number".to_owned().into()))),
            0
        );
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Float(std::f64::consts::PI))),
            0
        );
        assert_eq!(value_to_i64(Some(&SqliteValue::Blob(vec![1, 2].into()))), 0);
        assert_eq!(value_to_i64(Some(&SqliteValue::Null)), 0);
        assert_eq!(value_to_i64(None), 0);
    }

    #[test]
    fn persist_segment_with_infinity_timestamps_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("inf_seg.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let mut report = minimal_report("run-inf", &db_path);
        report.result.segments = vec![TranscriptionSegment {
            start_sec: Some(f64::INFINITY),
            end_sec: Some(f64::NEG_INFINITY),
            speaker: None,
            text: "infinity segment".to_owned(),
            confidence: Some(f64::INFINITY),
        }];

        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-inf")
            .expect("load")
            .expect("exists");
        assert_eq!(details.segments.len(), 1);
        assert_eq!(details.segments[0].text, "infinity segment");
        // INFINITY round-trips through SQLite Float → serde_json:
        // f64::INFINITY persists as a float and may round-trip or become null.
        // The key assertion is that it doesn't panic or corrupt the database.
    }

    #[test]
    fn list_recent_runs_transcript_preview_handles_multibyte_unicode() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("unicode_preview.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Create a transcript with multi-byte chars (each emoji is 1 char but 4 bytes).
        // 141 emoji chars — preview should truncate to exactly 140 chars.
        let emoji_transcript: String = "\u{1F600}".repeat(141);

        let mut report = minimal_report("run-emoji", &db_path);
        report.result.transcript = emoji_transcript;
        store.persist_report(&report).expect("persist");

        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 1);
        assert_eq!(
            runs[0].transcript_preview.chars().count(),
            140,
            "preview should be exactly 140 chars"
        );
    }

    #[test]
    fn persist_and_load_event_with_large_seq_value() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("large_seq.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let mut report = minimal_report("run-bigseq", &db_path);
        // Use a seq that fits in u64 but is large for i64.
        // Since SQLite stores as i64, values > i64::MAX would wrap.
        // Use i64::MAX as u64 (which is 9223372036854775807).
        let large_seq = i64::MAX as u64;
        report.events = vec![RunEvent {
            seq: large_seq,
            ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({"elapsed_ms": 100}),
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-bigseq")
            .expect("load")
            .expect("exists");
        assert_eq!(details.events.len(), 1);
        assert_eq!(details.events[0].seq, large_seq);
    }

    // -- twelfth-pass edge case tests --

    #[test]
    fn value_to_i64_overflow_text_returns_zero() {
        use super::value_to_i64;

        // i64::MAX + 1 overflows → parse fails → 0
        let overflow = format!("{}", i64::MAX as u128 + 1);
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Text(overflow.into()))),
            0,
            "text exceeding i64::MAX should return 0"
        );

        // i64::MIN - 1 overflows → parse fails → 0
        let underflow = format!("{}", i64::MIN as i128 - 1);
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Text(underflow.into()))),
            0,
            "text below i64::MIN should return 0"
        );

        // Very large number → 0
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Text(
                "999999999999999999999999".to_owned().into()
            ))),
            0
        );

        // Negative integer text that IS valid parses correctly
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Text("-42".to_owned().into()))),
            -42
        );
    }

    #[test]
    fn pragma_integer_returns_sensible_values() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("pragma_int.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // page_size is a well-known pragma; should return a positive power-of-2.
        let page_size = store.pragma_integer("page_size").expect("page_size");
        assert!(page_size > 0, "page_size should be positive");
        assert_eq!(
            page_size & (page_size - 1),
            0,
            "page_size should be a power of 2"
        );

        // page_count on a fresh db should be > 0 (at least schema pages).
        let page_count = store.pragma_integer("page_count").expect("page_count");
        assert!(page_count > 0, "page_count should be positive on fresh db");

        // freelist_count on a fresh db should be 0 (no freed pages).
        let freelist = store.pragma_integer("freelist_count").expect("freelist");
        assert_eq!(freelist, 0, "fresh db should have no freelist pages");
    }

    #[test]
    fn ensure_column_exists_adds_new_column() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("add_col.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Verify column does not already exist.
        let before = store
            .connection
            .query("PRAGMA table_info(runs);")
            .expect("pragma");
        let has_new_col = before
            .iter()
            .any(|row| super::value_to_string(row.get(1)) == "test_column_xyz");
        assert!(!has_new_col, "column should not exist yet");

        // Request a new column; migration helper should materialize it.
        store
            .ensure_column_exists("runs", "test_column_xyz", "TEXT NOT NULL DEFAULT 'hello'")
            .expect("add column");

        // Verify DB remains usable and schema was updated.
        let after = store
            .connection
            .query("PRAGMA table_info(runs);")
            .expect("pragma");
        let has_new_col = after
            .iter()
            .any(|row| super::value_to_string(row.get(1)) == "test_column_xyz");
        assert!(has_new_col, "requested column should be present");

        // Calling again is idempotent.
        store
            .ensure_column_exists("runs", "test_column_xyz", "TEXT NOT NULL DEFAULT 'hello'")
            .expect("idempotent call should succeed");

        store
            .persist_report(&minimal_report("compat-add-col", &db_path))
            .expect("store should remain writable");
    }

    #[test]
    fn run_store_open_creates_parent_directories() {
        let dir = tempdir().expect("tempdir");
        let nested = dir.path().join("deep").join("nested").join("dir");
        let db_path = nested.join("store.sqlite3");

        assert!(!nested.exists(), "nested dir should not exist yet");

        let store = RunStore::open(&db_path).expect("open should create parents");
        assert!(nested.exists(), "parent directories should be created");

        // Verify the store is functional.
        store
            .persist_report(&minimal_report("nested-run", &db_path))
            .expect("persist");
        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn load_run_details_event_with_null_payload_round_trips() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("null_payload.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let mut report = minimal_report("run-null-evt", &db_path);
        report.events = vec![RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            stage: "test".to_owned(),
            code: "test.null".to_owned(),
            message: "null payload".to_owned(),
            payload: serde_json::Value::Null,
        }];
        store.persist_report(&report).expect("persist");

        let details = store
            .load_run_details("run-null-evt")
            .expect("load")
            .expect("exists");
        assert_eq!(details.events.len(), 1);
        assert_eq!(details.events[0].payload, serde_json::Value::Null);
        assert_eq!(details.events[0].code, "test.null");
    }

    #[test]
    fn run_store_debug_impl_contains_struct_name() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("debug_fmt.sqlite3");
        let store = RunStore::open(&db_path).expect("open");
        let debug_output = format!("{:?}", store);
        assert!(
            debug_output.contains("RunStore"),
            "Debug output should contain 'RunStore': {debug_output}"
        );
    }

    #[test]
    fn concurrent_session_execute_plain_sql() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_exec.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let session = store
            .begin_concurrent_session("exec_test")
            .expect("session");
        // Happy path: insert via plain execute.
        let rows_affected = session
            .execute("INSERT INTO _meta (key, value) VALUES ('exec_key', 'exec_val')")
            .expect("execute");
        assert!(rows_affected >= 1, "should affect at least 1 row");
        session.commit().expect("commit");

        // Verify the row was persisted.
        let rows = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'exec_key'")
            .expect("query");
        assert_eq!(rows.len(), 1);

        // Error path: invalid SQL returns Err(Storage(...)).
        let session2 = store
            .begin_concurrent_session("exec_err")
            .expect("session2");
        let err = session2.execute("ABSOLUTELY NOT VALID SQL");
        assert!(err.is_err(), "invalid SQL should return error");
        let err_text = err.unwrap_err().to_string();
        assert!(
            err_text.contains("near") || err_text.contains("syntax") || err_text.contains("SQL"),
            "error should mention SQL issue: {err_text}"
        );
        session2.rollback().expect("rollback");
    }

    #[test]
    fn load_run_details_malformed_warnings_json_returns_storage_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("bad_warnings.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let report = minimal_report("run-bad-warn", &db_path);
        store.persist_report(&report).expect("persist");

        // Corrupt warnings_json directly in DB.
        store
            .connection
            .execute_with_params(
                "UPDATE runs SET warnings_json = ?1 WHERE id = ?2",
                &[
                    SqliteValue::Text("{not valid json array!!!}".to_owned().into()),
                    SqliteValue::Text("run-bad-warn".to_owned().into()),
                ],
            )
            .expect("corrupt warnings");

        let error = store
            .load_run_details("run-bad-warn")
            .expect_err("malformed warnings_json should fail loudly");
        assert!(error.to_string().contains("invalid warnings_json"));
    }

    #[test]
    fn load_run_details_malformed_replay_json_returns_storage_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("bad_replay.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let report = minimal_report("run-bad-replay", &db_path);
        store.persist_report(&report).expect("persist");

        // Corrupt replay_json directly in DB.
        store
            .connection
            .execute_with_params(
                "UPDATE runs SET replay_json = ?1 WHERE id = ?2",
                &[
                    SqliteValue::Text("!!! not json !!!".to_owned().into()),
                    SqliteValue::Text("run-bad-replay".to_owned().into()),
                ],
            )
            .expect("corrupt replay");

        let error = store
            .load_run_details("run-bad-replay")
            .expect_err("malformed replay_json should fail loudly");
        assert!(error.to_string().contains("invalid replay_json"));
    }

    #[test]
    fn persist_cancellable_non_busy_error_returns_immediately() {
        // Line 63: non-busy Storage errors are returned without retry.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("non_busy.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        // Force a deterministic non-busy storage error via a trigger that
        // aborts INSERT into runs.
        store
            .connection
            .execute(
                "CREATE TRIGGER runs_abort_insert \
                 BEFORE INSERT ON runs \
                 BEGIN \
                    SELECT RAISE(ABORT, 'forced non-busy failure'); \
                 END;",
            )
            .expect("create abort trigger");
        let report = minimal_report("run-fail", &db_path);
        let start = std::time::Instant::now();
        let err = store
            .persist_report_cancellable(&report, None)
            .expect_err("should fail with storage error");
        let elapsed = start.elapsed();

        // Verify it's a non-busy error (not a busy retry exhaustion).
        let text = err.to_string();
        assert!(
            !text.contains("database is busy") && !text.contains("retry loop exhausted"),
            "should be a non-busy error: {text}"
        );
        // Non-busy errors should return without the full 8-retry backoff cycle
        // (which adds at least 5+10+15+20+25+30+35+40 = 180ms of sleep).
        // Use a generous threshold to tolerate system load during parallel test runs.
        assert!(
            elapsed < Duration::from_secs(2),
            "non-busy error should return promptly (not 8× retry backoff), took {:?}",
            elapsed
        );
    }

    #[test]
    fn concurrent_session_hyphen_and_space_names_rejected() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("sessions.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        for invalid_name in &["my-session", "hello world", "session.v1", "résumé"] {
            let result = store.begin_concurrent_session(invalid_name);
            assert!(result.is_err(), "name '{invalid_name}' should be rejected");
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("invalid session name"),
                "error for '{invalid_name}' should mention 'invalid session name': {err}"
            );
        }

        // Valid name with underscore should succeed.
        let session = store
            .begin_concurrent_session("my_session_1")
            .expect("valid name");
        session.rollback().expect("rollback");
    }

    #[test]
    fn storage_diagnostics_clone_and_debug() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("diag.sqlite3");
        let store = RunStore::open(&db_path).expect("open");

        let diag = store.diagnostics().expect("diagnostics");
        let cloned = diag.clone();
        assert_eq!(diag.page_size, cloned.page_size);
        assert_eq!(diag.journal_mode, cloned.journal_mode);
        assert_eq!(diag.integrity_check, cloned.integrity_check);
        assert_eq!(diag.freelist_count, cloned.freelist_count);

        let debug = format!("{:?}", diag);
        assert!(
            debug.contains("StorageDiagnostics"),
            "Debug should contain struct name"
        );
        assert!(
            debug.contains("WalCheckpointInfo"),
            "Debug should contain nested struct name"
        );
    }

    #[test]
    fn is_busy_storage_error_embedded_substring_and_real_sqlite_format() {
        use super::is_busy_storage_error;
        use crate::error::FwError;

        // ALL-CAPS with longer surrounding message.
        assert!(is_busy_storage_error(&FwError::Storage(
            "SQLITE error: DATABASE IS LOCKED while writing".to_owned()
        )));
        // Real SQLite BUSY error format with "snapshot conflict" embedded.
        assert!(is_busy_storage_error(&FwError::Storage(
            "SQLITE_BUSY snapshot conflict: unable to open read-write transaction".to_owned()
        )));
        // "busy" alone (without "database is busy") should NOT match.
        assert!(!is_busy_storage_error(&FwError::Storage(
            "database busy timeout".to_owned()
        )));
        // Completely unrelated error.
        assert!(!is_busy_storage_error(&FwError::Storage(
            "unique constraint violated".to_owned()
        )));
    }

    #[test]
    fn optional_text_and_optional_float_helpers() {
        use super::{optional_float, optional_text, text_value};

        assert_eq!(
            text_value("hello".to_owned()),
            SqliteValue::Text("hello".to_owned().into())
        );
        assert_eq!(
            optional_text(Some("world")),
            SqliteValue::Text("world".to_owned().into())
        );
        assert_eq!(optional_text(None), SqliteValue::Null);
        assert_eq!(optional_float(Some(2.75)), SqliteValue::Float(2.75));
        assert_eq!(optional_float(None), SqliteValue::Null);
    }

    #[test]
    fn parse_backend_all_known_and_unknown() {
        use super::parse_backend;
        assert_eq!(parse_backend("whisper_cpp"), BackendKind::WhisperCpp);
        assert_eq!(parse_backend("insanely_fast"), BackendKind::InsanelyFast);
        assert_eq!(
            parse_backend("whisper_diarization"),
            BackendKind::WhisperDiarization
        );
        // Unknown strings (including "auto") fall back to Auto.
        assert_eq!(parse_backend("auto"), BackendKind::Auto);
        assert_eq!(parse_backend("something_else"), BackendKind::Auto);
        assert_eq!(parse_backend(""), BackendKind::Auto);
    }

    // ── Task #212 — storage pass 2 edge-case tests ──────────────────

    #[test]
    fn value_to_string_negative_float() {
        use super::value_to_string;
        assert_eq!(value_to_string(Some(&SqliteValue::Float(-1.5))), "-1.5");
        assert_eq!(value_to_string(Some(&SqliteValue::Float(-0.001))), "-0.001");
        // Verify it parses back correctly.
        let s = value_to_string(Some(&SqliteValue::Float(-99.99)));
        let parsed: f64 = s.parse().expect("should parse back to f64");
        assert!((parsed - (-99.99)).abs() < f64::EPSILON);
    }

    #[test]
    fn pragma_integer_nonexistent_pragma_returns_zero() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("pragma_test.sqlite3");
        let store = RunStore::open(&db_path).expect("store should open");

        // A nonexistent PRAGMA returns empty result → unwrap_or(0).
        let val = store
            .pragma_integer("totally_nonexistent_pragma_xyz_42")
            .expect("should not error");
        assert_eq!(val, 0, "nonexistent pragma should fall back to 0");
    }

    #[test]
    fn concurrent_session_execute_with_params_invalid_sql_errors() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("params_err.sqlite3");
        let store = RunStore::open(&db_path).expect("store should open");
        let session = store
            .begin_concurrent_session("params_test")
            .expect("session should start");

        let result = session.execute_with_params(
            "INSERT INTO nonexistent_table_xyz (col) VALUES (?1)",
            &[SqliteValue::Text("val".to_owned().into())],
        );
        assert!(result.is_err(), "invalid table should produce error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("nonexistent_table_xyz") || msg.contains("no such table"),
            "error should mention the table: {msg}"
        );
    }

    #[test]
    fn value_to_i64_min_boundary() {
        use super::value_to_i64;
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Integer(i64::MIN))),
            i64::MIN
        );
        // Text representation should also parse.
        assert_eq!(
            value_to_i64(Some(&SqliteValue::Text(i64::MIN.to_string().into()))),
            i64::MIN
        );
    }

    #[test]
    fn run_store_diagnostics_contains_expected_keys() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("diag.sqlite3");
        let store = RunStore::open(&db_path).expect("store should open");

        let diag = store.diagnostics().expect("diagnostics should succeed");
        assert!(diag.page_count >= 0, "page_count should be non-negative");
        assert!(diag.page_size > 0, "page_size should be positive");
        assert_eq!(
            diag.freelist_count, 0,
            "fresh db should have no freelist pages"
        );
        assert_eq!(
            diag.integrity_check, "ok",
            "fresh db should pass integrity check"
        );
        // WAL mode check.
        assert!(
            diag.journal_mode == "wal" || diag.journal_mode == "delete",
            "journal mode should be wal or delete, got: {}",
            diag.journal_mode
        );
    }

    #[test]
    fn sql_ident_escapes_embedded_double_quote() {
        use super::sql_ident;
        assert_eq!(
            sql_ident("col\"name"),
            "\"col\"\"name\"",
            "embedded double-quote should be doubled"
        );
    }

    #[test]
    fn sql_ident_leaves_plain_name_unquoted() {
        use super::sql_ident;
        assert_eq!(sql_ident("runs"), "runs");
        assert_eq!(sql_ident(""), "\"\"", "empty name still wraps in quotes");
    }

    #[test]
    fn is_busy_storage_error_detects_all_busy_variants() {
        use super::is_busy_storage_error;
        use crate::error::FwError;

        assert!(is_busy_storage_error(&FwError::Storage(
            "database is busy".to_owned()
        )));
        assert!(is_busy_storage_error(&FwError::Storage(
            "snapshot conflict detected".to_owned()
        )));
        assert!(is_busy_storage_error(&FwError::Storage(
            "DATABASE IS LOCKED".to_owned()
        )));
        // Non-matching storage errors.
        assert!(!is_busy_storage_error(&FwError::Storage(
            "no such table".to_owned()
        )));
        // Non-Storage error variant.
        assert!(!is_busy_storage_error(&FwError::InvalidRequest(
            "database is busy".to_owned()
        )));
    }

    #[test]
    fn optional_text_and_optional_float_produce_correct_sqlite_values() {
        use super::{optional_float, optional_text, text_value};
        let pi = std::f64::consts::PI;

        // optional_text
        assert!(matches!(
            optional_text(Some("hello")),
            SqliteValue::Text(ref s) if &**s == "hello"
        ));
        assert!(matches!(optional_text(None), SqliteValue::Null));

        // optional_float
        assert!(matches!(
            optional_float(Some(pi)),
            SqliteValue::Float(f) if (f - pi).abs() < 1e-15
        ));
        assert!(matches!(optional_float(None), SqliteValue::Null));

        // text_value
        assert!(matches!(
            text_value("world".to_owned()),
            SqliteValue::Text(ref s) if &**s == "world"
        ));
    }

    #[test]
    fn value_to_string_blob_includes_byte_count() {
        use super::value_to_string;

        assert_eq!(
            value_to_string(Some(&SqliteValue::Blob(vec![0xDE, 0xAD].into()))),
            "<blob:2>"
        );
        assert_eq!(
            value_to_string(Some(&SqliteValue::Blob(vec![].into()))),
            "<blob:0>",
            "empty blob should show zero length"
        );
    }

    // ── Task #257 — storage edge-case tests ─────────────────────────

    #[test]
    fn corrupt_acceleration_json_returns_storage_error() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("corrupt_accel.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-accel-corrupt", &db_path);
        report.result.acceleration = Some(AccelerationReport {
            backend: AccelerationBackend::Frankentorch,
            input_values: 42,
            normalized_confidences: true,
            pre_mass: Some(0.8),
            post_mass: Some(0.9),
            notes: vec!["test".to_owned()],
        });
        store.persist_report(&report).expect("persist");

        // Corrupt the acceleration_json column directly.
        store
            .connection
            .execute_with_params(
                "UPDATE runs SET acceleration_json = ?1 WHERE id = ?2",
                &[
                    SqliteValue::Text("NOT VALID JSON {{{".to_owned().into()),
                    SqliteValue::Text("run-accel-corrupt".to_owned().into()),
                ],
            )
            .expect("corrupt");

        let error = store
            .load_run_details("run-accel-corrupt")
            .expect_err("corrupt acceleration_json should fail loudly");
        assert!(error.to_string().contains("invalid acceleration_json"));
    }

    #[test]
    fn acceleration_override_from_column_takes_priority_over_result() {
        // When acceleration_json column has valid but different data than
        // result_json, the column value (acceleration_override) wins.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("accel_override.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let mut report = minimal_report("run-accel-override", &db_path);
        report.result.acceleration = Some(AccelerationReport {
            backend: AccelerationBackend::Frankentorch,
            input_values: 100,
            normalized_confidences: false,
            pre_mass: Some(0.5),
            post_mass: Some(0.6),
            notes: vec!["original".to_owned()],
        });
        store.persist_report(&report).expect("persist");

        // Overwrite acceleration_json column with different valid data.
        let override_accel = AccelerationReport {
            backend: AccelerationBackend::Frankentorch,
            input_values: 999,
            normalized_confidences: true,
            pre_mass: Some(0.99),
            post_mass: Some(1.0),
            notes: vec!["overridden".to_owned()],
        };
        let override_json = serde_json::to_string(&override_accel).expect("ser");
        store
            .connection
            .execute_with_params(
                "UPDATE runs SET acceleration_json = ?1 WHERE id = ?2",
                &[
                    SqliteValue::Text(override_json.into()),
                    SqliteValue::Text("run-accel-override".to_owned().into()),
                ],
            )
            .expect("override");

        let details = store
            .load_run_details("run-accel-override")
            .expect("load")
            .expect("exists");
        let accel = details.acceleration.expect("should be present");
        assert_eq!(
            accel.input_values, 999,
            "column override should take priority over result_json"
        );
        assert_eq!(accel.notes, vec!["overridden".to_owned()]);
    }

    #[test]
    fn concurrent_session_execute_with_params_inserts_data() {
        // The only existing test for execute_with_params tests the error path.
        // This tests the happy path: valid INSERT with parameters.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("params_happy.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let session = store
            .begin_concurrent_session("params_ok")
            .expect("session");
        let affected = session
            .execute_with_params(
                "INSERT INTO _meta (key, value) VALUES (?1, ?2)",
                &[
                    SqliteValue::Text("test_key".to_owned().into()),
                    SqliteValue::Text("test_value".to_owned().into()),
                ],
            )
            .expect("insert with params");
        assert!(affected >= 1, "should affect at least 1 row");
        session.commit().expect("commit");

        let rows = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'test_key'")
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            super::value_to_string(rows[0].get(0)),
            "test_value",
            "parameterized insert should persist the value"
        );
    }

    #[test]
    fn persist_report_duplicate_different_data_preserves_original() {
        // Existing test (persist_report_twice_same_id_succeeds_idempotently)
        // re-persists identical data. This tests with DIFFERENT data on the
        // same run_id, verifying original transcript is preserved.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("dup_diff.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let report = minimal_report("run-dup-diff", &db_path);
        store.persist_report(&report).expect("first persist");

        // Create a modified report with different transcript.
        let mut modified = minimal_report("run-dup-diff", &db_path);
        modified.result.transcript = "TOTALLY DIFFERENT TRANSCRIPT".to_owned();
        // Second persist may succeed (no-op) or fail (unique constraint);
        // either way, original data must remain intact.
        let _ = store.persist_report(&modified);

        let details = store
            .load_run_details("run-dup-diff")
            .expect("load")
            .expect("exists");
        assert_eq!(
            details.transcript, "test",
            "original transcript should be preserved, not overwritten"
        );
    }

    #[test]
    fn migration_v2_preserves_custom_index_during_column_addition() {
        // v2 adds the missing columns to a legacy runs table. This verifies
        // that a pre-existing custom index survives that migration path.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("idx_preserve.sqlite3");

        let conn = BlockingConnection::open(db_path.display().to_string()).expect("open");
        // Create v1 schema (no replay_json, no acceleration_json).
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, \
                started_at TEXT NOT NULL, \
                finished_at TEXT NOT NULL, \
                backend TEXT NOT NULL, \
                input_path TEXT NOT NULL, \
                normalized_wav_path TEXT NOT NULL, \
                request_json TEXT NOT NULL, \
                result_json TEXT NOT NULL, \
                warnings_json TEXT NOT NULL, \
                transcript TEXT NOT NULL\
            );",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
                start_sec REAL, end_sec REAL, speaker TEXT, \
                text TEXT NOT NULL, confidence REAL, \
                PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
                ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, \
                code TEXT NOT NULL, message TEXT NOT NULL, \
                payload_json TEXT NOT NULL, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");

        // Add a custom index BEFORE opening with RunStore.
        conn.execute("CREATE INDEX idx_custom_backend ON runs(backend);")
            .expect("create custom index");
        drop(conn);

        // Open with RunStore → triggers migration v1→v2→v3.
        let store = RunStore::open(&db_path).expect("open should migrate");
        assert_eq!(
            store.current_schema_version().expect("version"),
            RunStore::SCHEMA_VERSION
        );

        // Verify the custom index survived the column-add migration.
        let rows = store
            .connection
            .query(
                "SELECT name FROM sqlite_master \
                 WHERE type = 'index' AND name = 'idx_custom_backend';",
            )
            .expect("query indexes");
        assert_eq!(
            rows.len(),
            1,
            "custom index should survive column-add migration"
        );

        // Also verify the new columns exist.
        assert!(
            store
                .table_has_column("runs", "replay_json")
                .expect("check")
        );
        assert!(
            store
                .table_has_column("runs", "acceleration_json")
                .expect("check")
        );
    }

    #[test]
    fn table_has_column_returns_false_for_nonexistent_column() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_col.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        assert!(
            !store
                .table_has_column("runs", "totally_fake_column")
                .expect("should succeed"),
            "nonexistent column should return false"
        );
    }

    #[test]
    fn table_columns_returns_expected_column_count() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("cols.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        // The v3 schema `runs` table has 12 columns:
        // id, started_at, finished_at, backend, input_path,
        // normalized_wav_path, request_json, result_json,
        // warnings_json, transcript, replay_json, acceleration_json.
        let columns = store.table_columns("runs").expect("table_columns");
        assert_eq!(
            columns.len(),
            12,
            "runs table should have 12 columns in v3 schema"
        );
    }

    #[test]
    fn concurrent_session_query_reads_within_savepoint() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("sess_query.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let report = minimal_report("sess-q-1", &db_path);
        store.persist_report(&report).expect("persist");

        let session = store
            .begin_concurrent_session("qtest")
            .expect("begin session");
        let rows = session
            .query("SELECT id FROM runs WHERE id = 'sess-q-1';")
            .expect("session query");
        assert_eq!(rows.len(), 1, "session query should see the persisted run");
        session.commit().expect("commit");
    }

    #[test]
    fn reconstruct_column_definition_produces_valid_defs_for_runs_table() {
        // reconstruct_column_definition is used during table rebuilds.
        // Verify it produces a non-empty, well-formed definition for
        // every column in the runs table.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("reconstruct.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let columns = store.table_columns("runs").expect("columns");
        assert!(!columns.is_empty(), "runs table should have columns");

        let mut names = Vec::new();
        for column in &columns {
            let def = super::reconstruct_column_definition(column).expect("reconstruct");
            let name = column.name.clone();
            // Each definition must contain the quoted column name.
            assert!(
                def.contains(&name),
                "definition should contain column name `{name}`: {def}"
            );
            // Each definition must include the type (TEXT or REAL).
            let typ = column.declared_type.clone();
            if !typ.is_empty() {
                assert!(
                    def.contains(&typ),
                    "definition should contain type `{typ}`: {def}"
                );
            }
            names.push(name);
        }
        // Verify expected columns are present.
        assert!(names.contains(&"id".to_owned()), "runs should have id");
        assert!(
            names.contains(&"transcript".to_owned()),
            "runs should have transcript"
        );
        assert!(
            names.contains(&"replay_json".to_owned()),
            "runs should have replay_json"
        );
    }

    #[test]
    fn table_has_column_returns_false_for_nonexistent_table() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_table.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        // PRAGMA table_info on a nonexistent table returns zero rows,
        // so table_has_column should return false (not error).
        assert!(
            !store
                .table_has_column("totally_fake_table", "any_column")
                .expect("should succeed without error"),
            "nonexistent table should return false, not an error"
        );
    }

    // ── Task #306 — storage edge-case tests pass 2 ──────────────────

    #[test]
    fn apply_migration_v1_is_noop() {
        // apply_migration(1) is the v1 base migration which simply returns Ok(()).
        // Only apply_migration(999) was previously tested; the v1 path was untested.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("mig_v1.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        // Directly invoke the v1 migration — it should succeed as a no-op.
        store
            .apply_migration(1)
            .expect("v1 migration should be a no-op");
        // Schema should still be intact and usable.
        store
            .persist_report(&minimal_report("after-v1", &db_path))
            .expect("persist after v1 migration");
        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 1);
    }

    #[test]
    fn run_migrations_shortcut_v0_to_v2_when_fresh_shape() {
        // When schema_version == 0 but both replay_json AND acceleration_json
        // columns already exist (fresh schema), run_migrations shortcuts to v2
        // before continuing to v3. This tests the code path at lines 441-448.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("shortcut_v0.sqlite3");

        // Create a "fresh shape" DB at version 0: tables with both columns
        // present but no schema_version key in _meta.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, \
                started_at TEXT NOT NULL, \
                finished_at TEXT NOT NULL, \
                backend TEXT NOT NULL, \
                input_path TEXT NOT NULL, \
                normalized_wav_path TEXT NOT NULL, \
                request_json TEXT NOT NULL, \
                result_json TEXT NOT NULL, \
                warnings_json TEXT NOT NULL, \
                transcript TEXT NOT NULL, \
                replay_json TEXT NOT NULL DEFAULT '{}', \
                acceleration_json TEXT NOT NULL DEFAULT '{}'\
            );",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT NOT NULL, \
                idx INTEGER NOT NULL, \
                start_sec REAL, end_sec REAL, speaker TEXT, \
                text TEXT NOT NULL, confidence REAL, \
                PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT NOT NULL, \
                seq INTEGER NOT NULL, \
                ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, \
                code TEXT NOT NULL, message TEXT NOT NULL, \
                payload_json TEXT NOT NULL, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");
        // Crucially: do NOT insert a schema_version row → version == 0.
        drop(conn);

        // Open with RunStore. The shortcut should detect both columns,
        // jump to v2, then migrate to v3.
        let store = RunStore::open(&db_path).expect("shortcut migration");
        let version = store.current_schema_version().expect("version");
        assert_eq!(
            version,
            RunStore::SCHEMA_VERSION,
            "should reach latest version via shortcut"
        );

        // DB should be fully usable.
        store
            .persist_report(&minimal_report("shortcut-run", &db_path))
            .expect("persist");
        let details = store
            .load_run_details("shortcut-run")
            .expect("q")
            .expect("exists");
        assert_eq!(details.run_id, "shortcut-run");
    }

    #[test]
    fn ensure_column_exists_journal_mode_restored_after_add() {
        // ensure_column_exists switches from WAL to DELETE mode for DDL,
        // then restores WAL. Verify the journal mode restoration path.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("journal_restore.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // Verify we start in WAL mode.
        let mode_before = store
            .connection
            .query("PRAGMA journal_mode;")
            .expect("pragma");
        let mode_str = super::value_to_string(mode_before.first().and_then(|r| r.get(0)));
        assert!(
            mode_str.eq_ignore_ascii_case("wal"),
            "should start in WAL mode, got: {mode_str}"
        );

        // Add a column (triggers WAL→DELETE→WAL round-trip).
        store
            .ensure_column_exists("runs", "extra_col", "TEXT NOT NULL DEFAULT 'hi'")
            .expect("add column");

        // Verify journal mode was restored to WAL.
        let mode_after = store
            .connection
            .query("PRAGMA journal_mode;")
            .expect("pragma after");
        let restored = super::value_to_string(mode_after.first().and_then(|r| r.get(0)));
        assert!(
            restored.eq_ignore_ascii_case("wal"),
            "journal mode should be restored to WAL after column addition, got: {restored}"
        );

        // Store should remain fully functional.
        store
            .persist_report(&minimal_report("after-journal", &db_path))
            .expect("persist should work after journal mode restore");
    }

    #[test]
    fn migration_from_v1_preserves_existing_run_data() {
        // Insert a row into a legacy v1 schema (no replay_json, no acceleration_json),
        // then open with RunStore to trigger migration. The existing row should be
        // loadable with default replay/acceleration values.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("v1_data.sqlite3");

        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        // Create v1 schema.
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, \
                started_at TEXT NOT NULL, \
                finished_at TEXT NOT NULL, \
                backend TEXT NOT NULL, \
                input_path TEXT NOT NULL, \
                normalized_wav_path TEXT NOT NULL, \
                request_json TEXT NOT NULL, \
                result_json TEXT NOT NULL, \
                warnings_json TEXT NOT NULL, \
                transcript TEXT NOT NULL\
            );",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
                start_sec REAL, end_sec REAL, speaker TEXT, \
                text TEXT NOT NULL, confidence REAL, \
                PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
                ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, \
                code TEXT NOT NULL, message TEXT NOT NULL, \
                payload_json TEXT NOT NULL, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");

        // Insert a v1 row (no replay_json or acceleration_json columns).
        let result_json = serde_json::json!({
            "backend": "whisper_cpp",
            "transcript": "legacy audio",
            "language": "en",
            "segments": [{"start_sec": 0.0, "end_sec": 1.0, "text": "hello"}],
            "acceleration": null,
            "raw_output": {},
            "artifact_paths": []
        });
        conn.execute_with_params(
            "INSERT INTO runs (id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            &[
                SqliteValue::Text("legacy-run".to_owned().into()),
                SqliteValue::Text("2025-06-01T00:00:00Z".to_owned().into()),
                SqliteValue::Text("2025-06-01T00:00:05Z".to_owned().into()),
                SqliteValue::Text("whisper_cpp".to_owned().into()),
                SqliteValue::Text("old.wav".to_owned().into()),
                SqliteValue::Text("old_norm.wav".to_owned().into()),
                SqliteValue::Text("{}".to_owned().into()),
                SqliteValue::Text(result_json.to_string().into()),
                SqliteValue::Text("[]".to_owned().into()),
                SqliteValue::Text("legacy audio".to_owned().into()),
            ],
        )
        .expect("insert legacy run");
        drop(conn);

        // Open with RunStore → triggers migration (v0→v2→v3).
        let store = RunStore::open(&db_path).expect("open after migration");
        let details = store
            .load_run_details("legacy-run")
            .expect("q")
            .expect("legacy row should be loadable after migration");
        assert_eq!(details.run_id, "legacy-run");
        assert_eq!(details.transcript, "legacy audio");
        // Replay should default to empty (column added with DEFAULT '{}').
        assert!(
            details.replay.input_content_hash.is_none(),
            "migrated legacy run should have empty replay defaults"
        );
    }

    #[test]
    fn migration_from_partial_v1_preserves_existing_replay_payload() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("partial_v1.sqlite3");

        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, \
                started_at TEXT NOT NULL, \
                finished_at TEXT NOT NULL, \
                backend TEXT NOT NULL, \
                input_path TEXT NOT NULL, \
                normalized_wav_path TEXT NOT NULL, \
                request_json TEXT NOT NULL, \
                result_json TEXT NOT NULL, \
                warnings_json TEXT NOT NULL, \
                transcript TEXT NOT NULL, \
                replay_json TEXT NOT NULL DEFAULT '{}'\
            );",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
                start_sec REAL, end_sec REAL, speaker TEXT, \
                text TEXT NOT NULL, confidence REAL, \
                PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
                ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, \
                code TEXT NOT NULL, message TEXT NOT NULL, \
                payload_json TEXT NOT NULL, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");

        let replay_json = serde_json::json!({
            "input_content_hash": "legacy-input-hash",
            "backend_identity": "legacy-whisper",
            "backend_version": "0.9.0",
            "output_payload_hash": "legacy-output-hash"
        })
        .to_string();
        let result_json = serde_json::json!({
            "backend": "whisper_cpp",
            "transcript": "legacy replay run",
            "language": "en",
            "segments": [],
            "acceleration": null,
            "raw_output": {},
            "artifact_paths": []
        })
        .to_string();

        conn.execute_with_params(
            "INSERT INTO runs (id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            &[
                SqliteValue::Text("partial-legacy-run".to_owned().into()),
                SqliteValue::Text("2025-06-02T00:00:00Z".to_owned().into()),
                SqliteValue::Text("2025-06-02T00:00:05Z".to_owned().into()),
                SqliteValue::Text("whisper_cpp".to_owned().into()),
                SqliteValue::Text("old.wav".to_owned().into()),
                SqliteValue::Text("old_norm.wav".to_owned().into()),
                SqliteValue::Text("{}".to_owned().into()),
                SqliteValue::Text(result_json.into()),
                SqliteValue::Text("[]".to_owned().into()),
                SqliteValue::Text("legacy replay run".to_owned().into()),
                SqliteValue::Text(replay_json.into()),
            ],
        )
        .expect("insert partial legacy run");
        drop(conn);

        let store = RunStore::open(&db_path).expect("open after migration");
        assert!(
            store
                .table_has_column("runs", "acceleration_json")
                .expect("check acceleration column"),
            "migration should add acceleration_json to partial v1 schema"
        );
        let details = store
            .load_run_details("partial-legacy-run")
            .expect("query")
            .expect("row exists");
        assert_eq!(
            details.replay.backend_identity.as_deref(),
            Some("legacy-whisper"),
            "existing replay payload should survive migration"
        );
    }

    #[test]
    fn migration_v2_failpoint_rolls_back_runs_swap() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("rollback_failpoint.sqlite3");

        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, \
                started_at TEXT NOT NULL, \
                finished_at TEXT NOT NULL, \
                backend TEXT NOT NULL, \
                input_path TEXT NOT NULL, \
                normalized_wav_path TEXT NOT NULL, \
                request_json TEXT NOT NULL, \
                result_json TEXT NOT NULL, \
                warnings_json TEXT NOT NULL, \
                transcript TEXT NOT NULL\
            );",
        )
        .expect("create legacy runs");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT NOT NULL, idx INTEGER NOT NULL, \
                start_sec REAL, end_sec REAL, speaker TEXT, \
                text TEXT NOT NULL, confidence REAL, \
                PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT NOT NULL, seq INTEGER NOT NULL, \
                ts_rfc3339 TEXT NOT NULL, stage TEXT NOT NULL, \
                code TEXT NOT NULL, message TEXT NOT NULL, \
                payload_json TEXT NOT NULL, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");
        conn.execute_with_params(
            "INSERT INTO _meta (key, value) VALUES ('schema_version', ?1)",
            &[SqliteValue::Text(
                RunStore::SCHEMA_VERSION.to_string().into(),
            )],
        )
        .expect("freeze schema version");
        conn.execute_with_params(
            "INSERT INTO runs (id, started_at, finished_at, backend, input_path, \
             normalized_wav_path, request_json, result_json, warnings_json, transcript) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            &[
                SqliteValue::Text("rollback-run".to_owned().into()),
                SqliteValue::Text("2025-06-03T00:00:00Z".to_owned().into()),
                SqliteValue::Text("2025-06-03T00:00:05Z".to_owned().into()),
                SqliteValue::Text("whisper_cpp".to_owned().into()),
                SqliteValue::Text("legacy.wav".to_owned().into()),
                SqliteValue::Text("legacy_norm.wav".to_owned().into()),
                SqliteValue::Text("{}".to_owned().into()),
                SqliteValue::Text(
                    serde_json::json!({
                        "backend": "whisper_cpp",
                        "transcript": "rollback transcript",
                        "language": "en",
                        "segments": [],
                        "acceleration": null,
                        "raw_output": {},
                        "artifact_paths": []
                    })
                    .to_string()
                    .into(),
                ),
                SqliteValue::Text("[]".to_owned().into()),
                SqliteValue::Text("rollback transcript".to_owned().into()),
            ],
        )
        .expect("insert rollback row");
        drop(conn);

        let store = RunStore::open(&db_path).expect("open");
        let error = store
            .migrate_legacy_runs_schema_v2_with_failpoint()
            .expect_err("failpoint should abort migration");
        assert!(
            error
                .to_string()
                .contains("simulated migration failure after runs table swap"),
            "error should surface failpoint context"
        );
        assert!(
            !store
                .table_has_column("runs", "replay_json")
                .expect("check replay column"),
            "rollback should restore legacy runs schema"
        );
        let rows = store
            .connection
            .query("SELECT id, transcript FROM runs WHERE id = 'rollback-run';")
            .expect("query rows after rollback");
        assert_eq!(rows.len(), 1, "rollback should preserve legacy row");
        assert_eq!(
            super::value_to_string(rows[0].get(1)),
            "rollback transcript",
            "rollback should preserve legacy row contents"
        );
    }

    #[test]
    fn set_schema_version_inserts_when_key_missing() {
        // set_schema_version uses INSERT OR REPLACE, so it must create the
        // schema_version row when it does not yet exist.
        // The existing test (set_schema_version_overwrites_existing_value)
        // covers replacement; this one covers initial insertion.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("ver_insert.sqlite3");

        // Create a bare DB with _meta but no schema_version row.
        let conn = BlockingConnection::open(db_path.display().to_string()).expect("conn");
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            .expect("create _meta");
        // Also create required tables so RunStore operations work.
        conn.execute(
            "CREATE TABLE runs (\
                id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, \
                backend TEXT, input_path TEXT, normalized_wav_path TEXT, \
                request_json TEXT, result_json TEXT, warnings_json TEXT, \
                transcript TEXT, replay_json TEXT DEFAULT '{}', \
                acceleration_json TEXT DEFAULT '{}'\
            );",
        )
        .expect("create runs");
        conn.execute(
            "CREATE TABLE segments (\
                run_id TEXT, idx INTEGER, start_sec REAL, end_sec REAL, \
                speaker TEXT, text TEXT, confidence REAL, PRIMARY KEY (run_id, idx)\
            );",
        )
        .expect("create segments");
        conn.execute(
            "CREATE TABLE events (\
                run_id TEXT, seq INTEGER, ts_rfc3339 TEXT, stage TEXT, \
                code TEXT, message TEXT, payload_json TEXT, \
                PRIMARY KEY (run_id, seq)\
            );",
        )
        .expect("create events");

        // Confirm no schema_version row exists.
        let rows = conn
            .query("SELECT value FROM _meta WHERE key = 'schema_version';")
            .expect("query");
        assert!(rows.is_empty(), "should have no schema_version row yet");
        drop(conn);

        // Open with RunStore — this triggers set_schema_version via run_migrations
        // and should materialize the missing schema_version row.
        let store = RunStore::open(&db_path).expect("open");
        let version = store.current_schema_version().expect("version");
        assert_eq!(
            version,
            RunStore::SCHEMA_VERSION,
            "version should be set to latest after INSERT fallback"
        );

        // Verify the row was actually inserted (not just cached).
        let rows = store
            .connection
            .query("SELECT value FROM _meta WHERE key = 'schema_version';")
            .expect("query");
        assert_eq!(rows.len(), 1, "schema_version row should now exist");
    }

    // ------------------------------------------------------------------
    // storage edge-case tests pass 3
    // ------------------------------------------------------------------

    #[test]
    fn apply_migration_v2_idempotent_when_columns_exist() {
        // Directly calls `apply_migration(2)` which ensures replay_json and
        // acceleration_json columns exist. On a fresh RunStore these columns
        // are already present from initialize_schema, so the call should be
        // a no-op that succeeds without error.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("mig_v2.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        // apply_migration(2) should be idempotent — columns already exist.
        store
            .apply_migration(2)
            .expect("v2 migration should succeed when columns exist");

        // Verify data operations still work after the idempotent migration.
        store
            .persist_report(&minimal_report("after-v2", &db_path))
            .expect("persist");
        let runs = store.list_recent_runs(10).expect("list");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_id, "after-v2");
    }

    #[test]
    fn recreate_table_with_added_column_empty_columns_returns_error() {
        // Exercises the guard in recreate_table_with_added_column:
        //   if existing_columns.is_empty() {
        //       return Err(FwError::Storage("cannot rebuild table ... no discovered columns"));
        //   }
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("empty_cols.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let empty_cols: Vec<super::TableColumn> = vec![];
        let err = store
            .recreate_table_with_added_column("runs", &empty_cols, "new_col", "TEXT DEFAULT ''")
            .expect_err("should fail with empty columns");
        let msg = err.to_string();
        assert!(
            msg.contains("no discovered columns"),
            "error should mention no discovered columns: {msg}"
        );
    }

    #[test]
    fn table_columns_returns_empty_for_nonexistent_table() {
        // Exercises `table_columns` on a table that does not exist.
        // PRAGMA table_info returns zero rows for nonexistent tables.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_such_table.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let columns = store
            .table_columns("completely_fictional_table")
            .expect("columns");
        assert!(
            columns.is_empty(),
            "nonexistent table should return empty column list"
        );
    }

    #[test]
    fn concurrent_session_query_method_reads_within_savepoint() {
        // Exercises `ConcurrentPersistSession::query()` (line 1014-1018).
        // The execute() and execute_with_params() methods on the session are
        // tested elsewhere, but `query()` has never been called in tests.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("session_query.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        store
            .persist_report(&minimal_report("query-run", &db_path))
            .expect("persist");

        let session = store
            .begin_concurrent_session("querytest")
            .expect("begin session");

        // Use session.query() to read data within the savepoint.
        let rows = session
            .query("SELECT id, transcript FROM runs WHERE id = 'query-run'")
            .expect("session query");
        assert_eq!(rows.len(), 1, "should find the persisted run");
        let id = super::value_to_string(rows[0].get(0));
        assert_eq!(id, "query-run");

        session.commit().expect("commit");
    }

    #[test]
    fn ensure_column_exists_on_nonexistent_table_returns_error() {
        // When called with a table that doesn't exist, ensure_column_exists now
        // reaches ALTER TABLE and should surface a clear add-column failure.
        // The exact storage message can vary by backend, so keep the assertion
        // broad but ensure it still explains the add-column failure.
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("no_table.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let err = store
            .ensure_column_exists("phantom_table", "phantom_col", "TEXT DEFAULT ''")
            .expect_err("should fail for nonexistent table");
        let msg = err.to_string();
        assert!(
            msg.contains("no discovered columns") || msg.contains("failed to add column"),
            "error should mention add-column failure: {msg}"
        );
    }
}
