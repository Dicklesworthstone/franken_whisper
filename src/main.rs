use std::sync::mpsc;
use std::time::Duration;

use clap::Parser;
use franken_whisper::cli::{
    Cli, Command, ControlFrameKind, RobotCommand, RunsOutputFormat, ShutdownController,
    SyncCommand, TtyAudioCommand, TtyAudioControlCommand,
};
use franken_whisper::model::StoredRunDetails;
use franken_whisper::robot::{
    acceleration_context_from_evidence, backends_discovery_value, build_backends_report,
    build_health_report, emit_health_report, emit_robot_complete, emit_robot_error,
    emit_robot_stage, emit_robot_start, robot_schema_value, routing_decision_value,
};
use franken_whisper::storage::RunStore;
use franken_whisper::tty_audio;
use franken_whisper::{FrankenWhisperEngine, FwError, FwResult};

fn main() {
    franken_whisper::logging::init();

    // bd-38c.6: Install graceful Ctrl+C shutdown handler.
    if let Err(e) = ShutdownController::install(None) {
        tracing::warn!("failed to install Ctrl+C handler: {e}");
    }

    let cli = Cli::parse();
    let is_robot = matches!(cli.command, Command::Robot { .. });

    if let Err(error) = run(cli) {
        // If shutdown was triggered via Ctrl+C, exit with signal code.
        if ShutdownController::is_shutting_down() {
            if !is_robot {
                eprintln!("interrupted");
            }
            std::process::exit(ShutdownController::signal_exit_code());
        }
        if !is_robot {
            eprintln!("error: {error}");
        }
        std::process::exit(1);
    }

    // If we completed but Ctrl+C was pressed (e.g. during finalization),
    // use the signal exit code.
    if ShutdownController::is_shutting_down() {
        std::process::exit(ShutdownController::signal_exit_code());
    }
}

fn run(cli: Cli) -> FwResult<()> {
    match cli.command {
        Command::Transcribe(args) => {
            let request = args.to_request()?;
            let engine = FrankenWhisperEngine::new()?;
            let report = engine.transcribe(request)?;

            if args.json {
                let mut value = serde_json::to_value(&report)?;
                if let Some(acceleration_context) =
                    acceleration_context_from_evidence(&report.evidence)
                    && let Some(object) = value.as_object_mut()
                {
                    object.insert("acceleration_context".to_owned(), acceleration_context);
                }
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                println!("{}", report.result.transcript);
            }
            Ok(())
        }
        Command::Robot { command } => match command {
            RobotCommand::Run(args) => {
                emit_robot_start(args.robot_summary())?;
                let request = match args.to_request() {
                    Ok(request) => request,
                    Err(error) => {
                        emit_robot_error(&error.to_string(), error.robot_error_code())?;
                        return Err(error);
                    }
                };

                let (event_tx, event_rx) = mpsc::channel();
                let worker = std::thread::spawn(move || -> FwResult<_> {
                    let engine = FrankenWhisperEngine::new()?;
                    engine.transcribe_with_stream(request, event_tx)
                });

                loop {
                    match event_rx.recv_timeout(Duration::from_millis(40)) {
                        Ok(streamed) => emit_robot_stage(&streamed.run_id, &streamed.event)?,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if worker.is_finished() {
                                break;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }

                while let Ok(streamed) = event_rx.try_recv() {
                    emit_robot_stage(&streamed.run_id, &streamed.event)?;
                }

                match worker.join() {
                    Ok(Ok(report)) => emit_robot_complete(&report),
                    Ok(Err(error)) => {
                        emit_robot_error(&error.to_string(), error.robot_error_code())?;
                        Err(error)
                    }
                    Err(_) => {
                        let error = FwError::Unsupported("robot worker thread panicked".to_owned());
                        emit_robot_error(&error.to_string(), error.robot_error_code())?;
                        Err(error)
                    }
                }
            }
            RobotCommand::Schema => {
                println!("{}", serde_json::to_string_pretty(&robot_schema_value())?);
                Ok(())
            }
            RobotCommand::RoutingHistory(args) => {
                let store = RunStore::open(&args.db)?;
                let details_list =
                    load_routing_history_details(&store, args.run_id.as_deref(), args.limit)?;

                for details in details_list {
                    for event in &details.events {
                        if event.code == "backend.routing.decision_contract"
                            || event.code == "backend.routing.safe_mode"
                            || event.code == "backend.routing.calibration_guardrail"
                        {
                            let entry = routing_decision_value(
                                &details.run_id,
                                &event.ts_rfc3339,
                                &event.code,
                                &event.payload,
                            );
                            println!("{}", serde_json::to_string(&entry)?);
                        }
                    }
                }
                Ok(())
            }
            RobotCommand::Health(args) => {
                let report = build_health_report(&args.db);
                emit_health_report(&report)
            }
            RobotCommand::Backends => {
                println!("{}", backends_command_output()?);
                Ok(())
            }
        },
        Command::Runs(args) => {
            let store = RunStore::open(&args.db)?;

            if let Some(run_id) = &args.id {
                match store.load_run_details(run_id)? {
                    Some(details) => match args.format {
                        RunsOutputFormat::Plain => {
                            println!("{}", details.transcript);
                        }
                        RunsOutputFormat::Json => {
                            println!("{}", serde_json::to_string_pretty(&details)?);
                        }
                        RunsOutputFormat::Ndjson => {
                            println!("{}", serde_json::to_string(&details)?);
                        }
                    },
                    None => {
                        return Err(FwError::InvalidRequest(format!(
                            "no run found with id `{run_id}`"
                        )));
                    }
                }
            } else {
                let runs = store.list_recent_runs(args.limit)?;
                match args.format {
                    RunsOutputFormat::Plain => {
                        for run in runs {
                            println!(
                                "{} | {} | {} | {}",
                                run.started_at_rfc3339,
                                run.backend.as_str(),
                                run.run_id,
                                run.transcript_preview
                            );
                        }
                    }
                    RunsOutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&runs)?);
                    }
                    RunsOutputFormat::Ndjson => {
                        for run in runs {
                            println!("{}", serde_json::to_string(&run)?);
                        }
                    }
                }
            }
            Ok(())
        }
        Command::Sync { command } => match command {
            SyncCommand::Export(args) => {
                let manifest =
                    franken_whisper::sync::export(&args.db, &args.output, &args.state_root)?;
                println!("{}", serde_json::to_string_pretty(&manifest)?);
                Ok(())
            }
            SyncCommand::Import(args) => {
                let result = franken_whisper::sync::import(
                    &args.db,
                    &args.input,
                    &args.state_root,
                    args.conflict_policy,
                )?;
                let validation = franken_whisper::sync::validate_sync(&args.db, &args.input);
                let (validation_report, validation_error) = match validation {
                    Ok(report) => (Some(report), None),
                    Err(error) => (None, Some(error.to_string())),
                };
                let validation_ok = validation_report
                    .as_ref()
                    .map(|report| report.is_valid)
                    .unwrap_or(false);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "runs_imported": result.runs_imported,
                        "segments_imported": result.segments_imported,
                        "events_imported": result.events_imported,
                        "conflicts": result.conflicts,
                        "validation_ok": validation_ok,
                        "validation": validation_report,
                        "validation_error": validation_error,
                    }))?
                );
                Ok(())
            }
        },
        Command::TtyAudio { command } => match command {
            TtyAudioCommand::Encode { input, chunk_ms } => {
                tty_audio::encode_to_stdout(&input, chunk_ms)
            }
            TtyAudioCommand::Decode { output, recovery } => {
                tty_audio::decode_from_stdin_to_wav_with_policy(&output, recovery.into())
            }
            TtyAudioCommand::RetransmitPlan { recovery } => {
                let plan = tty_audio::retransmit_plan_from_stdin(recovery.into())?;
                println!("{}", serde_json::to_string(&plan)?);
                Ok(())
            }
            TtyAudioCommand::Control { command } => match command {
                TtyAudioControlCommand::Handshake {
                    min_version,
                    max_version,
                    supported_codecs,
                } => tty_audio::emit_control_frame_to_stdout(
                    &tty_audio::TtyControlFrame::Handshake {
                        min_version,
                        max_version,
                        supported_codecs,
                    },
                ),
                TtyAudioControlCommand::HandshakeAck {
                    negotiated_version,
                    negotiated_codec,
                } => tty_audio::emit_control_frame_to_stdout(
                    &tty_audio::TtyControlFrame::HandshakeAck {
                        negotiated_version,
                        negotiated_codec,
                    },
                ),
                TtyAudioControlCommand::Ack { up_to_seq } => {
                    tty_audio::emit_control_frame_to_stdout(&tty_audio::TtyControlFrame::Ack {
                        up_to_seq,
                    })
                }
                TtyAudioControlCommand::Backpressure { remaining_capacity } => {
                    tty_audio::emit_control_frame_to_stdout(
                        &tty_audio::TtyControlFrame::Backpressure { remaining_capacity },
                    )
                }
                TtyAudioControlCommand::RetransmitRequest { sequences } => {
                    tty_audio::emit_control_frame_to_stdout(
                        &tty_audio::TtyControlFrame::RetransmitRequest { sequences },
                    )
                }
                TtyAudioControlCommand::RetransmitResponse { sequences } => {
                    tty_audio::emit_control_frame_to_stdout(
                        &tty_audio::TtyControlFrame::RetransmitResponse { sequences },
                    )
                }
                TtyAudioControlCommand::RetransmitLoop { recovery, rounds } => {
                    tty_audio::emit_retransmit_loop_from_stdin(recovery.into(), rounds)
                }
            },

            // bd-2xe.4: convenience send-control command
            TtyAudioCommand::SendControl { frame_type } => send_control_frame(frame_type),

            // bd-2xe.4: convenience retransmit command
            TtyAudioCommand::Retransmit { recovery, rounds } => {
                tty_audio::emit_retransmit_loop_from_stdin(recovery.into(), rounds)
            }
        },
        Command::Tui => franken_whisper::tui::run_tui(),
        Command::Youtube(args) => {
            let opts = args.to_options()?;
            let summary = franken_whisper::youtube::pipeline::run(&opts)?;
            if args.json_summary {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            } else {
                println!(
                    "YouTube ingestion: {} done, {} skipped, {} failed{}",
                    summary.done.len(),
                    summary.skipped.len(),
                    summary.failed.len(),
                    if summary.cancelled {
                        " (cancelled)"
                    } else {
                        ""
                    },
                );
                for f in &summary.failed {
                    eprintln!("  failed {}: {} — {}", f.id, f.title, f.error);
                }
            }
            // Any failed video is a non-zero outcome; cancellation is surfaced
            // through the global shutdown exit code in main().
            if !summary.failed.is_empty() && !summary.cancelled {
                return Err(FwError::Unsupported(format!(
                    "{} youtube video(s) failed",
                    summary.failed.len()
                )));
            }
            Ok(())
        }
    }
}

fn load_routing_history_details(
    store: &RunStore,
    run_id: Option<&str>,
    limit: usize,
) -> FwResult<Vec<StoredRunDetails>> {
    if let Some(run_id) = run_id {
        return Ok(store.load_run_details(run_id)?.into_iter().collect());
    }

    let summaries = store.list_recent_runs(limit)?;
    summaries
        .iter()
        .map(|summary| {
            store.load_run_details(&summary.run_id)?.ok_or_else(|| {
                FwError::Storage(format!(
                    "run `{}` disappeared while loading routing history",
                    summary.run_id
                ))
            })
        })
        .collect()
}

fn backends_command_output() -> FwResult<String> {
    let report = build_backends_report();
    Ok(serde_json::to_string(&backends_discovery_value(&report))?)
}

// ---------------------------------------------------------------------------
// bd-2xe.4: send-control helper
// ---------------------------------------------------------------------------

/// Emit a control frame to stdout based on the simplified `ControlFrameKind`.
///
/// - `Handshake` emits a default handshake with protocol v1 and the standard
///   codec.
/// - `Eof` emits an end-of-stream control frame.
/// - `Reset` emits a stream-reset control frame.
fn send_control_frame(kind: ControlFrameKind) -> FwResult<()> {
    match kind {
        ControlFrameKind::Handshake => {
            tty_audio::emit_control_frame_to_stdout(&tty_audio::TtyControlFrame::Handshake {
                min_version: 1,
                max_version: 1,
                supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
            })
        }
        ControlFrameKind::Eof => tty_audio::emit_session_close(
            &mut std::io::stdout().lock(),
            tty_audio::SessionCloseReason::Normal,
            None,
        ),
        ControlFrameKind::Reset => tty_audio::emit_session_close(
            &mut std::io::stdout().lock(),
            tty_audio::SessionCloseReason::Error,
            None,
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::json;

    use super::{backends_command_output, load_routing_history_details};
    use franken_whisper::model::{
        BackendKind, BackendParams, InputSource, RunReport, TranscribeRequest, TranscriptionResult,
    };
    use franken_whisper::storage::RunStore;
    use tempfile::tempdir;

    fn fixture_report(run_id: &str, db_path: &Path) -> RunReport {
        RunReport {
            run_id: run_id.to_owned(),
            trace_id: format!("trace-{run_id}"),
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
                transcript: "test transcript".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![],
                acceleration: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![],
            replay: franken_whisper::model::ReplayEnvelope::default(),
        }
    }

    #[test]
    fn backends_command_output_matches_robot_contract() {
        let line = backends_command_output().expect("backends command should serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&line).expect("backends command output should be valid json");

        assert_eq!(parsed["event"], "backends.discovery");
        assert_eq!(
            parsed["schema_version"],
            franken_whisper::robot::ROBOT_SCHEMA_VERSION
        );
        assert!(parsed["backends"].is_array());
    }

    #[test]
    fn load_routing_history_details_returns_specific_run_when_present() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("routing_history_specific.sqlite3");
        let store = RunStore::open(&db_path).expect("store");
        let report = fixture_report("routing-run", &db_path);
        store.persist_report(&report).expect("persist");

        let details =
            load_routing_history_details(&store, Some("routing-run"), 10).expect("load details");
        assert_eq!(details.len(), 1, "specific run should yield one record");
        assert_eq!(details[0].run_id, "routing-run");
    }

    #[test]
    fn load_routing_history_details_propagates_corrupt_run_errors() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("routing_history_corrupt.sqlite3");
        let store = RunStore::open(&db_path).expect("store");

        let older = fixture_report("routing-good", &db_path);
        let mut newer = fixture_report("routing-bad", &db_path);
        newer.started_at_rfc3339 = "2026-02-22T00:00:02Z".to_owned();
        newer.finished_at_rfc3339 = "2026-02-22T00:00:03Z".to_owned();

        store.persist_report(&older).expect("persist good");
        store.persist_report(&newer).expect("persist bad");

        let connection = fsqlite::Connection::open(db_path.display().to_string()).expect("conn");
        connection
            .execute_with_params(
                "UPDATE runs SET result_json = ?1 WHERE id = ?2",
                &[
                    fsqlite_types::value::SqliteValue::Text("not valid json".to_owned().into()),
                    fsqlite_types::value::SqliteValue::Text("routing-bad".to_owned().into()),
                ],
            )
            .expect("corrupt result_json");

        let error = load_routing_history_details(&store, None, 10)
            .expect_err("corrupt run should surface an error");
        assert!(
            error.to_string().contains("invalid result_json"),
            "error should expose the corrupt run details: {error}"
        );
    }
}
