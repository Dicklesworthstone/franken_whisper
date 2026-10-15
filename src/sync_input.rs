// Included in sync::engine so the verified-input boundary uses the existing
// row replay, conflict handling, and version validators rather than a second
// implementation of their contracts.

struct PinnedImport {
    _directory: tempfile::TempDir,
    manifest: SyncManifest,
    paths: JsonlInputPaths,
    deleted_runs: Vec<DeletedRunRecord>,
}

fn pin_import(input_dir: &Path) -> FwResult<PinnedImport> {
    let manifest_path = input_dir.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path).map_err(|error| {
        FwError::Storage(format!(
            "cannot read manifest {}: {error}",
            manifest_path.display()
        ))
    })?;
    let manifest: SyncManifest = serde_json::from_str(&manifest_text)
        .map_err(|error| FwError::Storage(format!("invalid manifest: {error}")))?;
    validate_schema_version(&manifest)?;
    validate_export_format_version(&manifest)?;

    #[derive(Deserialize)]
    struct Extensions {
        #[serde(default)]
        deleted_runs: Option<DeletedRunsChannel>,
    }
    let extensions: Extensions = serde_json::from_str(&manifest_text)
        .map_err(|error| FwError::Storage(format!("invalid manifest: {error}")))?;
    if let Some(channel) = extensions.deleted_runs.as_ref()
        && (channel.format_version != DELETED_RUNS_FORMAT_VERSION
            || channel.file != format!("{DELETED_RUNS_STEM}.jsonl"))
    {
        return Err(FwError::Storage(
            "unsupported deleted-runs format or non-canonical channel file".to_owned(),
        ));
    }

    let directory = tempfile::Builder::new().prefix(".fw-sync-input-").tempdir()?;
    for (stem, digest) in [
        ("runs", manifest.checksums.runs_jsonl_sha256.as_str()),
        ("segments", manifest.checksums.segments_jsonl_sha256.as_str()),
        ("events", manifest.checksums.events_jsonl_sha256.as_str()),
    ] {
        pin_import_channel(input_dir, directory.path(), stem, digest)?;
    }
    if let Some(channel) = extensions.deleted_runs.as_ref() {
        pin_import_channel(
            input_dir,
            directory.path(),
            DELETED_RUNS_STEM,
            &channel.sha256,
        )?;
    }
    let paths = JsonlInputPaths::resolve(directory.path());
    validate_pinned_rows(&paths.runs, "runs", manifest.row_counts.runs)?;
    validate_pinned_rows(&paths.segments, "segments", manifest.row_counts.segments)?;
    validate_pinned_rows(&paths.events, "events", manifest.row_counts.events)?;
    // Reuse the complete tombstone validator, but against the pinned bytes.
    let deleted_runs = load_deleted_runs_channel(&manifest_text, directory.path())?;
    Ok(PinnedImport {
        _directory: directory,
        manifest,
        paths,
        deleted_runs,
    })
}

fn pin_import_channel(
    input_dir: &Path,
    private_dir: &Path,
    stem: &str,
    expected_digest: &str,
) -> FwResult<()> {
    let source = resolve_jsonl_path(input_dir, stem);
    let mut reader = open_jsonl_reader(&source)?;
    let destination = private_dir.join(format!("{stem}.jsonl"));
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let length = reader.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        writer.write_all(&buffer[..length])?;
        hasher.update(&buffer[..length]);
    }
    writer.flush()?;
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected_digest {
        return Err(FwError::Storage(format!(
            "checksum mismatch for {} (logical {stem}.jsonl bytes): expected {expected_digest}, got {actual}",
            source.display()
        )));
    }
    Ok(())
}

fn validate_pinned_rows(path: &Path, table: &str, expected: u64) -> FwResult<()> {
    let mut count = 0u64;
    let mut keys = HashSet::new();
    for (index, line) in open_jsonl_reader(path)?.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let validation = (|| -> FwResult<()> {
            let row: serde_json::Value = serde_json::from_str(&line)?;
            if !row.is_object() {
                return Err(FwError::Storage("record must be a JSON object".to_owned()));
            }
            let (run_id, ordinal) = if table == "runs" {
                let id = json_str(&row, "id")?;
                validate_archived_run(&row)?;
                (id, 0)
            } else {
                let id = json_str(&row, "run_id")?;
                let field = if table == "segments" { "idx" } else { "seq" };
                let ordinal = json_required_i64(&row, field, table)?;
                if ordinal < 0 {
                    return Err(FwError::Storage(format!("{field} must be non-negative")));
                }
                if table == "segments" {
                    // Do not silently coerce malformed speaker/timing fields
                    // into NULL as the permissive SQL conversion helpers do.
                    json_nullable_f64(&row, "start_sec", table)?;
                    json_nullable_f64(&row, "end_sec", table)?;
                    json_nullable_f64(&row, "confidence", table)?;
                    json_nullable_text(&row, "speaker", table)?;
                    json_str(&row, "text")?;
                } else {
                    for field in ["ts_rfc3339", "stage", "code", "message"] {
                        json_str(&row, field)?;
                    }
                    let payload = json_str(&row, "payload_json")?;
                    serde_json::from_str::<serde_json::Value>(&payload)?;
                }
                (id, ordinal)
            };
            if run_id.is_empty() {
                return Err(FwError::Storage("run identity must be non-empty".to_owned()));
            }
            if !keys.insert((run_id, ordinal)) {
                return Err(FwError::Storage(
                    "duplicate primary key in snapshot".to_owned(),
                ));
            }
            Ok(())
        })();
        validation.map_err(|error| {
            FwError::Storage(format!("{table}.jsonl line {}: {error}", index + 1))
        })?;
        count = count
            .checked_add(1)
            .ok_or_else(|| FwError::Storage(format!("row count overflow for {table}")))?;
    }
    // Count the archive, not conflict outcomes. A skip/overwrite conflict must
    // not suppress manifest reconciliation or authenticate a truncated channel.
    if count != expected {
        return Err(FwError::Storage(format!(
            "row count mismatch for {table}: expected {expected}, found {count}"
        )));
    }
    Ok(())
}

/// An absent legacy column has a defined default. A present malformed value
/// does not: treating a number/object/null as absence loses replay provenance.
fn archived_optional_json<'a>(row: &'a serde_json::Value, field: &str) -> FwResult<&'a str> {
    match row.get(field) {
        None => Ok("{}"),
        Some(serde_json::Value::String(value)) => Ok(value),
        Some(_) => Err(FwError::Storage(format!(
            "present `{field}` must be a JSON-encoded string"
        ))),
    }
}

fn validate_archived_run_metadata(row: &serde_json::Value) -> FwResult<()> {
    // All required columns must be checked before conflict resolution can
    // short-circuit. These values are retained verbatim, including timestamp
    // offsets and precision; parsing is validation, not normalization.
    for field in ["started_at", "finished_at"] {
        let timestamp = json_str(row, field)?;
        chrono::DateTime::parse_from_rfc3339(&timestamp).map_err(|error| {
            FwError::Storage(format!("invalid `{field}` in runs JSONL: {error}"))
        })?;
    }
    for field in ["input_path", "normalized_wav_path", "transcript"] {
        json_str(row, field)?;
    }
    let warnings = json_str(row, "warnings_json")?;
    serde_json::from_str::<Vec<String>>(&warnings).map_err(|error| {
        FwError::Storage(format!("invalid `warnings_json` in runs JSONL: {error}"))
    })?;

    let replay = archived_optional_json(row, "replay_json")?;
    let replay: serde_json::Value = serde_json::from_str(replay).map_err(|error| {
        FwError::Storage(format!("invalid `replay_json` in runs JSONL: {error}"))
    })?;
    if !replay.is_object() {
        return Err(FwError::Storage(
            "invalid `replay_json`: expected an object".to_owned(),
        ));
    }
    serde_json::from_value::<crate::model::ReplayEnvelope>(replay).map_err(|error| {
        FwError::Storage(format!("invalid `replay_json` in runs JSONL: {error}"))
    })?;

    // Match the historical no-override encodings accepted by the run loader.
    // A populated override must deserialize as the same public DTO the loader
    // consumes, or import could succeed while history remains unreadable.
    let acceleration = archived_optional_json(row, "acceleration_json")?;
    if !matches!(acceleration.trim(), "" | "{}" | "null") {
        serde_json::from_str::<crate::model::AccelerationReport>(acceleration).map_err(
            |error| {
                FwError::Storage(format!(
                    "invalid `acceleration_json` in runs JSONL: {error}"
                ))
            },
        )?;
    }
    Ok(())
}

fn validate_archived_run(row: &serde_json::Value) -> FwResult<()> {
    validate_archived_run_metadata(row)?;
    let run_id = json_str(row, "id")?;
    let backend = json_str(row, "backend")?;
    let result_json = json_str(row, "result_json")?;
    crate::storage::validate_run_backend_json(&run_id, &backend, &result_json)?;
    let result: crate::model::TranscriptionResult = serde_json::from_str(&result_json)?;
    let request_json = json_str(row, "request_json")?;
    let request_value: serde_json::Value = serde_json::from_str(&request_json)?;
    if !request_value.is_object() {
        return Err(FwError::Storage(
            "canonical request JSON root must be an object".to_owned(),
        ));
    }
    let enabled = match request_value.get("diarize") {
        None => false,
        Some(serde_json::Value::Bool(value)) => *value,
        Some(_) => {
            return Err(FwError::Storage(
                "field `diarize` must be a boolean".to_owned(),
            ));
        }
    };
    let configuration = request_value
        .pointer("/backend_params/acoustic_diarization")
        .filter(|value| !value.is_null());
    if !enabled && configuration.is_none() && result.diarization.is_none() {
        // Preserve the existing contract for minimal legacy/non-batch rows.
        return Ok(());
    }
    if configuration.is_some_and(|value| value.get("speaker_count").is_none()) {
        return Err(FwError::Storage(
            "legacy diarization speaker-count encoding is unsupported".to_owned(),
        ));
    }
    let request: crate::model::TranscribeRequest = serde_json::from_value(request_value)?;
    let effective = request.diarize.then(|| {
        request
            .backend_params
            .acoustic_diarization
            .clone()
            .unwrap_or_default()
    });
    match (effective.as_ref(), result.diarization.as_ref()) {
        (Some(request), Some(report)) => {
            // The runtime's classifier and request binding are authoritative:
            // no separate schema allowlist and no relabeling of Sortformer.
            report
                .validate_against_request(request, None)
                .map_err(|error| {
                    FwError::Storage(format!(
                        "invalid diarization contract for run {run_id}: {error}"
                    ))
                })?;
            if report.implementation.starts_with("native-acoustic")
                && report.speaker_count.estimate.is_none()
            {
                return Err(FwError::Storage(
                    "native acoustic report requires a speaker-count estimate".to_owned(),
                ));
            }
        }
        (None, Some(_)) => {
            return Err(FwError::Storage(
                "diarization report requires an enabled canonical request".to_owned(),
            ));
        }
        (Some(request), None) => {
            let lower_bound = request
                .known_intervals
                .iter()
                .map(|hint| hint.end_ms)
                .max()
                .unwrap_or(0);
            request
                .validate(lower_bound)
                .map_err(|error| FwError::Storage(error.to_string()))?;
        }
        (None, None) => {}
    }
    Ok(())
}

fn import_pinned_inner(
    db_path: &Path,
    input_dir: &Path,
    policy: ConflictPolicy,
) -> FwResult<ImportResult> {
    import_pinned_inner_with_after_pin(db_path, input_dir, policy, || Ok(()))
}

pub(super) fn import_pinned_inner_with_after_pin(
    db_path: &Path,
    input_dir: &Path,
    policy: ConflictPolicy,
    after_pin: impl FnOnce() -> FwResult<()>,
) -> FwResult<ImportResult> {
    // All file reads and preflight failures precede opening/migrating the DB.
    // Only these private, checksummed bytes are replayed, never the source paths.
    let pinned = pin_import(input_dir)?;
    after_pin()?;
    let store = RunStore::open(db_path)?;
    let connection = store.connection();
    ensure_schema(connection)?;
    connection
        .execute("BEGIN;")
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let result = (|| -> FwResult<ImportResult> {
        let imported = import_tables(
            connection,
            &pinned.paths,
            &pinned.manifest,
            &pinned.deleted_runs,
            policy,
            sync_batch_import_enabled(),
        )?;
        if !imported.conflicts.is_empty() {
            // The artifact belongs to the caller's archive, not the private
            // input copy, and a write failure must still roll back the DB.
            write_conflicts_file(input_dir, &imported.conflicts)?;
            if policy == ConflictPolicy::Reject {
                return Err(FwError::Storage(format!(
                    "import rejected due to {} conflict(s); see sync_conflicts.jsonl",
                    imported.conflicts.len()
                )));
            }
        }
        store.rebuild_diarization_index()?;
        connection
            .execute("COMMIT;")
            .map_err(|error| FwError::Storage(error.to_string()))?;
        Ok(imported)
    })();
    if result.is_err() {
        let _ = connection.execute("ROLLBACK;");
    }
    result
}

#[cfg(test)]
mod pinned_channel_tests {
    use super::*;

    fn run_row() -> serde_json::Value {
        serde_json::json!({
            "id": "run", "started_at": "2026-10-15T08:00:00.123456-04:00",
            "finished_at": "2026-10-15T12:00:01.123456Z", "backend": "whisper_cpp",
            "input_path": "fixture.wav", "normalized_wav_path": "normalized.wav",
            "request_json": "{}",
            "result_json": serde_json::json!({
                "backend": "whisper_cpp", "transcript": "", "language": null,
                "segments": [], "acceleration": null, "raw_output": {},
                "artifact_paths": [],
            }).to_string(),
            "warnings_json": "[]", "transcript": "", "replay_json": "{}",
            "acceleration_json": "{}",
        })
    }

    fn write_archive(
        root: &Path,
        runs: &[serde_json::Value],
        segments: &[serde_json::Value],
        events: &[serde_json::Value],
    ) {
        fs::create_dir_all(root).expect("archive directory");
        let write_channel = |stem: &str, rows: &[serde_json::Value]| {
            let mut bytes = Vec::new();
            for row in rows {
                serde_json::to_writer(&mut bytes, row).expect("serialize row");
                bytes.push(b'\n');
            }
            fs::write(root.join(format!("{stem}.jsonl")), &bytes).expect("channel");
            format!("{:x}", Sha256::digest(&bytes))
        };
        let manifest = SyncManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            export_format_version: EXPORT_FORMAT_VERSION.to_owned(),
            created_at_rfc3339: "2026-10-15T12:00:02Z".to_owned(),
            source_db_path: "fixture.sqlite3".to_owned(),
            row_counts: RowCounts {
                runs: runs.len() as u64,
                segments: segments.len() as u64,
                events: events.len() as u64,
            },
            checksums: FileChecksums {
                runs_jsonl_sha256: write_channel("runs", runs),
                segments_jsonl_sha256: write_channel("segments", segments),
                events_jsonl_sha256: write_channel("events", events),
            },
        };
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("manifest");
    }

    #[test]
    fn archived_metadata_retains_legacy_defaults_and_timestamp_spelling() {
        let mut row = run_row();
        row.as_object_mut().expect("object").remove("replay_json");
        row.as_object_mut().expect("object").remove("acceleration_json");
        let before = row.clone();
        validate_archived_run(&row).expect("absent legacy columns remain supported");
        assert_eq!(row, before, "validation must not normalize canonical bytes");
        for sentinel in ["", "{}", "null", " null "] {
            row["acceleration_json"] = serde_json::json!(sentinel);
            validate_archived_run(&row).expect("loader's no-override encoding");
        }
        row["replay_json"] = serde_json::json!(serde_json::json!({
            "input_content_hash": "historical-fingerprint",
            "backend_identity": "fixture-backend", "backend_version": "1.0",
            "output_payload_hash": "historical-output-fingerprint",
        }).to_string());
        row["warnings_json"] = serde_json::json!(r#"["retained warning"]"#);
        row["acceleration_json"] = serde_json::json!(serde_json::json!({
            "backend": "none", "input_values": 0, "normalized_confidences": false,
            "pre_mass": null, "post_mass": null, "notes": [],
        }).to_string());
        validate_archived_run(&row).expect("populated typed metadata");
    }

    #[test]
    fn archived_metadata_rejects_present_values_instead_of_defaulting() {
        for field in ["replay_json", "acceleration_json"] {
            for value in [serde_json::json!(null), serde_json::json!(7), serde_json::json!({})] {
                let mut row = run_row();
                row[field] = value;
                let error = validate_archived_run(&row).expect_err("not an absent column");
                assert!(error.to_string().contains(field), "{error}");
            }
        }
    }

    #[test]
    fn pinned_metadata_failures_precede_open_for_every_conflict_policy() {
        let cases = [
            ("started_at", serde_json::json!("not a timestamp")),
            ("finished_at", serde_json::json!("2026-13-99T00:00:00Z")),
            ("input_path", serde_json::json!(null)),
            ("normalized_wav_path", serde_json::json!(17)),
            ("transcript", serde_json::json!([])),
            ("warnings_json", serde_json::json!("{}")),
            ("warnings_json", serde_json::json!("[1]")),
            ("replay_json", serde_json::json!(null)),
            ("replay_json", serde_json::json!("null")),
            ("replay_json", serde_json::json!("[]")),
            ("replay_json", serde_json::json!(r#"{"backend_identity":7}"#)),
            ("acceleration_json", serde_json::json!({})),
            ("acceleration_json", serde_json::json!(r#"{"backend":"unknown"}"#)),
            ("acceleration_json", serde_json::json!("invalid JSON")),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("archive");
        for (case, (field, value)) in cases.into_iter().enumerate() {
            let mut row = run_row();
            row[field] = value;
            // Correct checksums ensure rejection is semantic, not corruption.
            write_archive(&archive, &[row], &[], &[]);
            for policy in [
                ConflictPolicy::Reject, ConflictPolicy::Skip,
                ConflictPolicy::Overwrite, ConflictPolicy::OverwriteStrict,
            ] {
                for exists in [false, true] {
                    let target = dir.path().join(format!("{case}-{policy:?}-{exists}.sqlite3"));
                    if exists {
                        fs::write(&target, b"unopened destination sentinel").expect("sentinel");
                    }
                    let mut after_pin_called = false;
                    let error = import_pinned_inner_with_after_pin(
                        &target, &archive, policy, || {
                            after_pin_called = true;
                            Ok(())
                        },
                    )
                    .expect_err("bad metadata must fail before the destination is opened");
                    assert!(error.to_string().contains(field), "{field}: {error}");
                    assert!(!after_pin_called, "preflight must reject the archive");
                    if exists {
                        assert_eq!(fs::read(&target).expect("sentinel"), b"unopened destination sentinel");
                    } else {
                        assert!(!target.exists());
                    }
                }
            }
        }
    }

    fn gzip_member(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(bytes).expect("gzip write");
        encoder.finish().expect("gzip finish")
    }

    #[test]
    fn pin_channel_copies_and_hashes_all_gzip_members() {
        let source = tempfile::tempdir().expect("source");
        let private = tempfile::tempdir().expect("private");
        let first = b"{\"record\":1}\n";
        let second = b"{\"record\":2}\n";
        let mut compressed = gzip_member(first);
        compressed.extend(gzip_member(second));
        fs::write(source.path().join("runs.jsonl.gz"), compressed).expect("gzip input");
        fs::write(source.path().join("runs.jsonl"), b"stale plain file").expect("plain shadow");
        let mut logical = first.to_vec();
        logical.extend_from_slice(second);
        let digest = format!("{:x}", Sha256::digest(&logical));
        pin_import_channel(source.path(), private.path(), "runs", &digest).expect("pin");
        assert_eq!(
            fs::read(private.path().join("runs.jsonl")).expect("private bytes"),
            logical
        );
    }

    #[test]
    fn pin_channel_rejects_a_truncated_gzip_tail() {
        let source = tempfile::tempdir().expect("source");
        let private = tempfile::tempdir().expect("private");
        let logical = b"{\"record\":1}\n";
        let mut compressed = gzip_member(logical);
        compressed.truncate(compressed.len() - 4);
        fs::write(source.path().join("runs.jsonl.gz"), compressed).expect("truncated gzip");
        let digest = format!("{:x}", Sha256::digest(logical));
        assert!(pin_import_channel(source.path(), private.path(), "runs", &digest).is_err());
    }

    #[test]
    fn pinned_rows_reject_malformed_speakers_and_event_payloads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let segments = dir.path().join("segments.jsonl");
        let mut row = serde_json::json!({
            "run_id": "run", "idx": 0, "start_sec": null, "end_sec": null,
            "speaker": "SPEAKER_00", "text": "fixture", "confidence": null,
        });
        fs::write(&segments, format!("{row}\n")).expect("valid segment");
        validate_pinned_rows(&segments, "segments", 1).expect("nullable fields are valid");
        row["speaker"] = serde_json::json!(7);
        fs::write(&segments, format!("{row}\n")).expect("malformed speaker");
        let error = validate_pinned_rows(&segments, "segments", 1).expect_err("not NULL");
        assert!(error.to_string().contains("speaker"), "{error}");

        let events = dir.path().join("events.jsonl");
        let mut event = serde_json::json!({
            "run_id": "run", "seq": 1, "ts_rfc3339": "2026-10-15T00:00:00Z",
            "stage": "backend", "code": "backend.ok", "message": "fixture",
            "payload_json": "{}",
        });
        fs::write(&events, format!("{event}\n")).expect("valid event");
        validate_pinned_rows(&events, "events", 1).expect("event control");
        event["payload_json"] = serde_json::json!("broken JSON");
        fs::write(&events, format!("{event}\n")).expect("malformed payload");
        assert!(validate_pinned_rows(&events, "events", 1).is_err());
    }
}
