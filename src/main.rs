use std::sync::mpsc;
use std::time::Duration;

use clap::Parser;
use franken_whisper::cli::{
    Cli, Command, ControlFrameKind, DifferentialOracleCommand, PublicCorpusCommand, PullModelArg,
    RobotCommand, RobotDocsCommand, RunsOutputFormat, ShutdownController, SyncCommand,
    TtyAudioCommand, TtyAudioControlCommand,
};
use franken_whisper::model::StoredRunDetails;
use franken_whisper::robot::{
    backends_discovery_value, build_backends_report, build_health_report, emit_health_report,
    emit_pretty_run_report, emit_robot_complete, emit_robot_error_from_fw, emit_robot_stage,
    emit_robot_start, robot_schema_value, routing_decision_line,
};
use franken_whisper::storage::RunStore;
use franken_whisper::tty_audio;
use franken_whisper::{FrankenWhisperEngine, FwError, FwResult};

pub(crate) fn main() {
    franken_whisper::logging::init();

    // bd-38c.6: Install graceful Ctrl+C shutdown handler.
    if let Err(e) = ShutdownController::install(None) {
        tracing::warn!("failed to install Ctrl+C handler: {e}");
    }

    let cli = parse_cli();
    let machine_output = command_uses_machine_output(&cli.command);
    let is_sortformer = matches!(cli.command, Command::SortformerDiarize(_));

    if let Err(error) = run(cli) {
        if is_sortformer {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "schema_version": "sortformer-diarization-v1",
                    "certification": "evaluation_only",
                    "status": "error",
                    "code": error.error_code(),
                    "message": "native Sortformer diarization failed",
                    "local_paths_emitted": false,
                }))
                .expect("the fixed Sortformer error envelope must serialize")
            );
        }
        // If shutdown was triggered via Ctrl+C, exit with signal code.
        if ShutdownController::is_shutting_down() {
            if !machine_output {
                eprintln!("interrupted");
            }
            std::process::exit(ShutdownController::signal_exit_code());
        }
        if !machine_output {
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

/// Parse Clap under program control so a syntactically invalid `robot`
/// invocation still emits one path-free JSON error object instead of human
/// prose on stderr. Other commands retain Clap's normal, high-quality errors.
fn parse_cli() -> Cli {
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let command = args.get(1).and_then(|arg| arg.to_str());
    let robot_intent = command.is_some_and(|arg| matches!(arg, "robot" | "agent"));
    let sortformer_intent =
        command.is_some_and(|arg| matches!(arg, "sortformer-diarize" | "sortformer"));
    let json_pull_intent = command == Some("pull")
        && args
            .iter()
            .skip(2)
            .any(|arg| arg.to_str() == Some("--json"));
    match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error)
            if robot_intent
                && !matches!(
                    error.kind(),
                    clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
                ) =>
        {
            let value = serde_json::json!({
                "event": "run_error",
                "schema_version": franken_whisper::robot::ROBOT_SCHEMA_VERSION,
                "code": "FW-INVALID-REQUEST",
                "message": "invalid robot command-line arguments; run `fw robot --help` or the selected subcommand with `--help`",
                "clap_error_kind": format!("{:?}", error.kind()),
            });
            println!(
                "{}",
                serde_json::to_string(&value)
                    .expect("the fixed robot parse-error envelope must serialize")
            );
            std::process::exit(error.exit_code());
        }
        Err(error)
            if sortformer_intent
                && !matches!(
                    error.kind(),
                    clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
                ) =>
        {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "schema_version": "sortformer-diarization-v1",
                    "certification": "evaluation_only",
                    "status": "error",
                    "code": "FW-INVALID-REQUEST",
                    "message": "invalid native Sortformer command-line arguments; run `fw sortformer-diarize --help`",
                    "clap_error_kind": format!("{:?}", error.kind()),
                    "local_paths_emitted": false,
                }))
                .expect("the fixed Sortformer parse-error envelope must serialize")
            );
            std::process::exit(error.exit_code());
        }
        Err(error)
            if json_pull_intent
                && !matches!(
                    error.kind(),
                    clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
                ) =>
        {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "schema_version": "franken-whisper-model-pull-v1",
                    "command": "pull",
                    "model": null,
                    "status": "error",
                    "code": "FW-INVALID-REQUEST",
                    "message": "invalid model-pull command-line arguments; run `fw pull --help`",
                    "clap_error_kind": format!("{:?}", error.kind()),
                    "local_paths_emitted": false,
                }))
                .expect("the fixed model-pull parse-error envelope must serialize")
            );
            std::process::exit(error.exit_code());
        }
        Err(error) => error.exit(),
    }
}

fn command_uses_machine_output(command: &Command) -> bool {
    match command {
        Command::Robot { .. } => true,
        Command::Capabilities(args) => args.json,
        Command::Models(args) => args.json,
        Command::Pull(args) => args.json,
        Command::SortformerDiarize(_) => true,
        Command::ComparisonWorker => true,
        Command::ComparisonCancelProbe(_) => true,
        Command::Doctor(args) => args.json,
        _ => false,
    }
}

fn sortformer_audio_duration_ms(sample_count: usize) -> FwResult<u64> {
    let samples = u64::try_from(sample_count).map_err(|_| {
        FwError::InvalidRequest("native Sortformer sample count exceeds u64".to_owned())
    })?;
    let sample_rate = u64::try_from(
        franken_whisper::sortformer_inference::SORTFORMER_SAMPLE_RATE_HZ,
    )
    .map_err(|_| FwError::InvalidRequest("native Sortformer sample rate exceeds u64".to_owned()))?;
    samples
        .checked_mul(1_000)
        .ok_or_else(|| {
            FwError::InvalidRequest("native Sortformer audio duration overflows u64".to_owned())
        })
        .map(|milliseconds| milliseconds.div_ceil(sample_rate))
}

fn pull_models(model: PullModelArg, json_output: bool) -> FwResult<()> {
    let pull_whisper = matches!(model, PullModelArg::All | PullModelArg::Whisper);
    let pull_sortformer = matches!(model, PullModelArg::All | PullModelArg::Sortformer);
    let requested_model = match model {
        PullModelArg::All => "all",
        PullModelArg::Whisper => "whisper",
        PullModelArg::Sortformer => "sortformer",
    };
    let result = (|| {
        let mut pulled = Vec::new();
        if pull_whisper {
            let outcome = franken_whisper::model_distribution::pull_whisper(
                ShutdownController::is_shutting_down,
                |line| {
                    if !json_output {
                        eprintln!("fw pull whisper: {line}");
                    }
                },
            )?;
            if !json_output {
                eprintln!(
                    "fw pull whisper: ready ({})",
                    if outcome.from_cache {
                        "already cached"
                    } else {
                        "downloaded and verified"
                    }
                );
            }
            pulled.push(serde_json::json!({
                "model": "whisper",
                "status": "ready",
                "from_cache": outcome.from_cache,
                "artifact_version": outcome.package.artifact_version,
                "package_sha256": outcome.package.weights_sha256,
                "distribution_policy": franken_whisper::model_distribution::WHISPER_DISTRIBUTION_POLICY,
                "license": "MIT",
                "weight_bytes_identity_preserved": true,
                "preparation_recipe": franken_whisper::model_distribution::WHISPER_PREPARATION_RECIPE,
            }));
        }
        if pull_sortformer {
            let outcome = franken_whisper::model_distribution::pull_sortformer(
                ShutdownController::is_shutting_down,
                |line| {
                    if !json_output {
                        eprintln!("fw pull sortformer: {line}");
                    }
                },
            )?;
            if !json_output {
                eprintln!(
                    "fw pull sortformer: ready ({})",
                    if outcome.from_cache {
                        "already cached"
                    } else {
                        "downloaded and verified"
                    }
                );
            }
            pulled.push(serde_json::json!({
                "model": "sortformer",
                "status": "ready",
                "from_cache": outcome.from_cache,
                "artifact_version": outcome.package.artifact_version,
                "package_sha256": outcome.package.package_sha256,
                "distribution_policy": franken_whisper::model_distribution::SORTFORMER_DISTRIBUTION_POLICY,
                "model_license_notice": franken_whisper::model_distribution::SORTFORMER_REQUIRED_NOTICE,
            }));
        }
        Ok::<_, FwError>(pulled)
    })();

    match result {
        Ok(pulled) => {
            if json_output {
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "schema_version": "franken-whisper-model-pull-v2",
                        "command": "pull",
                        "model": requested_model,
                        "status": "ready",
                        "models": pulled,
                        "local_paths_emitted": false,
                    }))?
                );
            }
            Ok(())
        }
        Err(error) if json_output => {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "schema_version": "franken-whisper-model-pull-v2",
                    "command": "pull",
                    "model": requested_model,
                    "status": "error",
                    "code": error.error_code(),
                    "message": "model provisioning failed",
                    "local_paths_emitted": false,
                }))?
            );
            Err(error)
        }
        Err(error) => Err(error),
    }
}

fn run(cli: Cli) -> FwResult<()> {
    match cli.command {
        Command::Transcribe(args) => {
            let json = args.json;
            let request = (*args).into_request()?;
            let engine = FrankenWhisperEngine::new()?;
            let report = engine.transcribe(request)?;

            if json {
                emit_pretty_run_report(report)?;
            } else {
                println!("{}", report.result.transcript);
            }
            Ok(())
        }
        Command::Robot { command } => match command {
            RobotCommand::Run(args) => {
                emit_robot_start(args.robot_summary())?;
                let request = match (*args).into_request() {
                    Ok(request) => request,
                    Err(error) => {
                        emit_robot_error_from_fw(&error)?;
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
                        emit_robot_error_from_fw(&error)?;
                        Err(error)
                    }
                    Err(_) => {
                        let error =
                            FwError::ContractViolation("robot worker thread panicked".to_owned());
                        emit_robot_error_from_fw(&error)?;
                        Err(error)
                    }
                }
            }
            RobotCommand::Schema => {
                println!("{}", serde_json::to_string(&robot_schema_value())?);
                Ok(())
            }
            RobotCommand::RoutingHistory(args) => {
                let store = RunStore::open(&args.db)?;
                let details_list =
                    load_routing_history_details(&store, args.run_id.as_deref(), args.limit)?;

                let mut records = 0_usize;
                for details in details_list {
                    for event in &details.events {
                        if event.code == "backend.routing.decision_contract"
                            || event.code == "backend.routing.safe_mode"
                            || event.code == "backend.routing.calibration_guardrail"
                        {
                            records += 1;
                            println!(
                                "{}",
                                routing_decision_line(
                                    &details.run_id,
                                    &event.ts_rfc3339,
                                    &event.code,
                                    &event.payload,
                                )?
                            );
                        }
                    }
                }
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "event": "routing_history.complete",
                        "schema_version": franken_whisper::robot::ROBOT_SCHEMA_VERSION,
                        "records": records,
                    }))?
                );
                Ok(())
            }
            RobotCommand::Health(args) => {
                let report = build_health_report(&args.db);
                let healthy = report.overall_status == franken_whisper::robot::CheckStatus::Ok;
                emit_health_report(&report)?;
                if args.strict && !healthy {
                    return Err(FwError::BackendUnavailable(
                        "health report is not ok; run `fw robot triage` for exact next commands"
                            .to_owned(),
                    ));
                }
                Ok(())
            }
            RobotCommand::Triage(args) => {
                let value = franken_whisper::robot::triage_report_value_with_cancel(
                    &args.db,
                    ShutdownController::is_shutting_down,
                )?;
                let ready = value["quick_ref"]["ready"].as_bool().unwrap_or(false);
                println!("{}", serde_json::to_string(&value)?);
                if args.strict && !ready {
                    return Err(FwError::BackendUnavailable(
                        "no transcription path is ready; follow the triage recommendation"
                            .to_owned(),
                    ));
                }
                Ok(())
            }
            RobotCommand::Backends => {
                println!("{}", backends_command_output()?);
                Ok(())
            }
        },
        Command::Capabilities(args) => {
            let value = franken_whisper::robot::capabilities_value();
            if args.json {
                println!("{}", serde_json::to_string(&value)?);
            } else {
                println!("franken_whisper {}", env!("CARGO_PKG_VERSION"));
                println!("Agent orientation: fw robot triage");
                println!("Machine contract: fw capabilities --json");
                println!("Model readiness: fw models --json");
                println!("Robot schema: fw robot schema");
            }
            Ok(())
        }
        Command::Models(args) => {
            let value = franken_whisper::robot::models_report_value_with_cancel(
                ShutdownController::is_shutting_down,
            )?;
            if args.json {
                println!("{}", serde_json::to_string(&value)?);
            } else {
                println!("FrankenWhisper model readiness (no network access performed)");
                for entry in value["models"].as_array().into_iter().flatten() {
                    println!(
                        "- {}: {}",
                        entry["id"].as_str().unwrap_or("unknown"),
                        entry["runtime_status"].as_str().unwrap_or("unknown")
                    );
                }
                println!("Details: fw models --json");
            }
            Ok(())
        }
        Command::Pull(args) => pull_models(args.model, args.json),
        Command::Doctor(args) => {
            let value = franken_whisper::robot::doctor_report_value_with_cancel(
                &args.db,
                ShutdownController::is_shutting_down,
            )?;
            let ready = value["ready"].as_bool().unwrap_or(false);
            if args.json {
                println!("{}", serde_json::to_string(&value)?);
            } else {
                println!(
                    "FrankenWhisper doctor: {}",
                    value["status"].as_str().unwrap_or("unknown")
                );
                for recommendation in value["recommendations"].as_array().into_iter().flatten() {
                    println!(
                        "- {}",
                        recommendation["command"]
                            .as_str()
                            .unwrap_or("fw robot triage")
                    );
                }
            }
            if args.strict && !ready {
                return Err(FwError::BackendUnavailable(
                    "installation is not ready for transcription; follow the doctor recommendation"
                        .to_owned(),
                ));
            }
            Ok(())
        }
        Command::RobotDocs { command } => match command {
            RobotDocsCommand::Guide => {
                print!("{}", franken_whisper::robot::robot_docs_guide());
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
        Command::DiarizationEval(args) => {
            let current_dir = std::env::current_dir().map_err(|_| {
                FwError::InvalidRequest(
                    "confidential_evaluation.project_root: current directory could not be resolved"
                        .to_owned(),
                )
            })?;
            let project_root =
                franken_whisper::confidential_evaluation::discover_project_root(&current_dir)?;
            let aggregate =
                franken_whisper::confidential_evaluation::run_confidential_evaluation_with_cancel(
                    &project_root,
                    &args.input_root,
                    &args.manifest,
                    &args.output,
                    ShutdownController::is_shutting_down,
                )?;
            println!("{}", serde_json::to_string_pretty(&aggregate)?);
            Ok(())
        }
        Command::DiarizationCorpus { command } => match command {
            PublicCorpusCommand::Registry => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &franken_whisper::public_corpus::public_corpus_registry()
                    )?
                );
                Ok(())
            }
            PublicCorpusCommand::PrepareVoxconverse(args) => {
                let current_dir = std::env::current_dir().map_err(|_| {
                    FwError::InvalidRequest(
                        "public_corpus.project_root: current directory could not be resolved"
                            .to_owned(),
                    )
                })?;
                let project_root =
                    franken_whisper::confidential_evaluation::discover_project_root(&current_dir)?;
                let summary =
                    franken_whisper::public_corpus::prepare_voxconverse_descriptor_with_cancel(
                        franken_whisper::public_corpus::VoxconverseDescriptorPreparationRequest {
                            project_root: &project_root,
                            input_root: &args.input_root,
                            development_audio_root: &args.development_audio_root,
                            test_audio_root: &args.test_audio_root,
                            annotation_root: &args.annotation_root,
                            output_path: &args.output,
                            source_version: &args.source_version,
                            license_acknowledgement_id: &args.license_ack,
                        },
                        ShutdownController::is_shutting_down,
                    )?;
                println!("{}", serde_json::to_string_pretty(&summary)?);
                Ok(())
            }
            PublicCorpusCommand::Build(args) => {
                let current_dir = std::env::current_dir().map_err(|_| {
                    FwError::InvalidRequest(
                        "public_corpus.project_root: current directory could not be resolved"
                            .to_owned(),
                    )
                })?;
                let project_root =
                    franken_whisper::confidential_evaluation::discover_project_root(&current_dir)?;
                let bundle =
                    franken_whisper::public_corpus::build_public_corpus_bundle_with_cancel(
                        &project_root,
                        &args.input_root,
                        &args.descriptor,
                        &args.output,
                        &args.license_ack,
                        ShutdownController::is_shutting_down,
                    )?;
                println!("{}", serde_json::to_string_pretty(&bundle)?);
                Ok(())
            }
            PublicCorpusCommand::Ablate(args) => {
                let current_dir = std::env::current_dir().map_err(|_| {
                    FwError::InvalidRequest(
                        "public_corpus.project_root: current directory could not be resolved"
                            .to_owned(),
                    )
                })?;
                let project_root =
                    franken_whisper::confidential_evaluation::discover_project_root(&current_dir)?;
                let evidence =
                    franken_whisper::public_corpus::run_public_corpus_ablation_with_cancel(
                        franken_whisper::public_corpus::PublicCorpusAblationRequest {
                            project_root: &project_root,
                            input_root: &args.input_root,
                            descriptor_path: &args.descriptor,
                            bundle_output_path: &args.bundle_output,
                            evidence_output_path: &args.output,
                            license_acknowledgement_id: &args.license_ack,
                            maximum_recording_duration_ms: args.maximum_recording_duration_ms,
                            evaluation_stage: args.stage.into(),
                            locked_development_evidence_path: args
                                .locked_development_evidence
                                .as_deref(),
                        },
                        ShutdownController::is_shutting_down,
                    )?;
                println!("{}", serde_json::to_string_pretty(&evidence)?);
                Ok(())
            }
            PublicCorpusCommand::SidecarStudy(args) => {
                let current_dir = std::env::current_dir().map_err(|_| {
                    FwError::InvalidRequest(
                        "public_corpus.project_root: current directory could not be resolved"
                            .to_owned(),
                    )
                })?;
                let project_root =
                    franken_whisper::confidential_evaluation::discover_project_root(&current_dir)?;
                let evidence =
                    franken_whisper::public_corpus::run_public_corpus_sidecar_study_with_cancel(
                        franken_whisper::public_corpus::PublicCorpusSidecarStudyRequest {
                            project_root: &project_root,
                            input_root: &args.input_root,
                            descriptor_path: &args.descriptor,
                            bundle_output_path: &args.bundle_output,
                            evidence_output_path: &args.output,
                            license_acknowledgement_id: &args.license_ack,
                            maximum_recording_duration_ms: args.maximum_recording_duration_ms,
                            evaluation_stage: args.stage.into(),
                            locked_development_evidence_path: args
                                .locked_development_evidence
                                .as_deref(),
                        },
                        ShutdownController::is_shutting_down,
                    )?;
                println!("{}", serde_json::to_string_pretty(&evidence)?);
                Ok(())
            }
            PublicCorpusCommand::CompareModels(args) => {
                let current_dir = std::env::current_dir().map_err(|_| {
                    FwError::InvalidRequest(
                        "public_corpus.project_root: current directory could not be resolved"
                            .to_owned(),
                    )
                })?;
                let project_root =
                    franken_whisper::confidential_evaluation::discover_project_root(&current_dir)?;
                let evidence =
                    franken_whisper::public_corpus::run_public_model_comparison_with_cancel(
                        franken_whisper::public_corpus::PublicModelComparisonRequest {
                            project_root: &project_root,
                            input_root: &args.input_root,
                            descriptor_path: &args.descriptor,
                            bundle_output_path: &args.bundle_output,
                            evidence_output_path: &args.output,
                            license_acknowledgement_id: &args.license_ack,
                            evaluation_split:
                                franken_whisper::diarization::EvaluationSplit::Development,
                            attempt_hard_timeout: Duration::from_secs(
                                franken_whisper::public_corpus::PUBLIC_MODEL_COMPARISON_ATTEMPT_TIMEOUT_SECONDS,
                            ),
                        },
                        ShutdownController::is_shutting_down,
                    )?;
                println!("{}", serde_json::to_string_pretty(&evidence)?);
                Ok(())
            }
        },
        Command::DiarizationOracle { command } => match command {
            DifferentialOracleCommand::Registry => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &franken_whisper::differential_oracle::differential_oracle_registry()
                    )?
                );
                Ok(())
            }
            DifferentialOracleCommand::Run(args) => {
                let current_dir = std::env::current_dir().map_err(|_| {
                    FwError::InvalidRequest(
                        "differential_oracle.project_root: current directory could not be resolved"
                            .to_owned(),
                    )
                })?;
                let project_root =
                    franken_whisper::confidential_evaluation::discover_project_root(&current_dir)?;
                let report =
                    franken_whisper::differential_oracle::run_differential_oracle(
                        franken_whisper::differential_oracle::DifferentialOracleRequest {
                            project_root: &project_root,
                            audio_path: &args.audio,
                            native_document_path: &args.native,
                            reference_document_path: args.reference.as_deref(),
                            output_path: &args.output,
                            tool: args.tool.into(),
                            hard_timeout: Duration::from_secs(args.timeout_seconds),
                            comparison_config: franken_whisper::differential_oracle::DifferentialComparisonConfig::default(),
                        },
                    )?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(())
            }
        },
        Command::SortformerDiarize(args) => {
            let cancellation = franken_whisper::orchestrator::CancellationToken::unbounded();
            let checkpoint = || {
                cancellation.checkpoint().map_err(|_| {
                    FwError::Cancelled("native Sortformer diarization interrupted".to_owned())
                })
            };
            checkpoint()?;
            let speaker_hints = args.load_speaker_hints()?;
            franken_whisper::sortformer_identity::validate_sortformer_hint_structure(
                &speaker_hints,
            )?;
            let (receipt_path, package_path, artifact_source, distribution_policy) = match (
                args.receipt.as_ref(),
                args.package.as_ref(),
            ) {
                (Some(receipt), Some(package)) => {
                    (receipt.clone(), package.clone(), "explicit_paths", None)
                }
                (None, None) => {
                    let cached =
                        franken_whisper::model_distribution::resolve_cached_sortformer_with_cancel(
                            ShutdownController::is_shutting_down,
                        )?;
                    (
                        cached.receipt_path,
                        cached.package_path,
                        "verified_release_cache",
                        Some(franken_whisper::model_distribution::SORTFORMER_DISTRIBUTION_POLICY),
                    )
                }
                _ => {
                    return Err(FwError::InvalidRequest(
                            "--receipt and --package must be supplied together, or both omitted to use `fw pull sortformer` cache"
                                .to_owned(),
                        ));
                }
            };
            let package_admission_started = std::time::Instant::now();
            let package = franken_whisper::sortformer_conformance::load_verified_sortformer_package_with_checkpoint(
                    &receipt_path,
                    &package_path,
                    &checkpoint,
                )?;
            let package_admission_seconds = package_admission_started.elapsed().as_secs_f64();
            let work_dir = tempfile::tempdir()?;
            let normalized = franken_whisper::audio::normalize_to_wav_with_cancel(
                &args.input,
                work_dir.path(),
                &cancellation,
            )?;
            let samples = franken_whisper::audio::read_normalized_wav_16k_mono_with_checkpoint(
                &normalized,
                &checkpoint,
            )?;
            let audio_duration_ms = sortformer_audio_duration_ms(samples.len())?;
            franken_whisper::sortformer_identity::validate_sortformer_hints(
                &speaker_hints,
                audio_duration_ms,
            )?;
            let session_materialization_started = std::time::Instant::now();
            let session = franken_whisper::sortformer_inference::SortformerSession::from_verified_package_with_checkpoint(
                &package,
                &checkpoint,
            )?;
            let session_materialization_seconds =
                session_materialization_started.elapsed().as_secs_f64();
            let inference_started = std::time::Instant::now();
            let output = session.diarize_with_checkpoint(
                franken_whisper::sortformer_inference::SortformerPcm::mono_16khz(&samples),
                &checkpoint,
            )?;
            let inference_seconds = inference_started.elapsed().as_secs_f64();
            let audio_seconds = samples.len() as f64
                / franken_whisper::sortformer_inference::SORTFORMER_SAMPLE_RATE_HZ as f64;
            let real_time_factor = (audio_seconds > 0.0).then(|| inference_seconds / audio_seconds);
            let mut active_speakers = std::collections::BTreeSet::new();
            for (index, turn) in output.turns.iter().enumerate() {
                if index.is_multiple_of(1_024) {
                    checkpoint()?;
                }
                active_speakers.insert(turn.speaker);
            }
            let identity_mapping =
                franken_whisper::sortformer_identity::map_sortformer_lanes_with_checkpoint(
                    &output.turns,
                    &speaker_hints,
                    audio_duration_ms,
                    &checkpoint,
                )?;
            let turns = output
                .turns
                .iter()
                .enumerate()
                .map(|(index, turn)| {
                    if index.is_multiple_of(1_024) {
                        checkpoint()?;
                    }
                    let hard_speaker_ref = identity_mapping.hard_speaker_ref(turn.speaker);
                    Ok(serde_json::json!({
                        "start_seconds": turn.start_seconds,
                        "end_seconds": turn.end_seconds,
                        "speaker_lane": turn.speaker,
                        "speaker_ref": hard_speaker_ref,
                        "speaker_ref_authority": if hard_speaker_ref.is_some() { "derived_from_caller_hard_interval" } else { "anonymous" },
                        "soft_speaker_suggestion": identity_mapping.soft_speaker_suggestion(turn.speaker),
                    }))
                })
                .collect::<FwResult<Vec<_>>>()?;
            checkpoint()?;
            let receipt = package.receipt();
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "schema_version": "sortformer-diarization-v1",
                    "certification": "evaluation_only",
                    "status": "ok",
                    "network_access_performed": false,
                    "local_paths_emitted": false,
                    "model": {
                        "id": receipt.model.model_id,
                        "revision": receipt.model.model_revision,
                        "package_sha256": receipt.package.sha256,
                        "package_bytes": receipt.package.bytes,
                        "distribution_policy": distribution_policy,
                        "artifact_source": artifact_source,
                    },
                    "audio": {
                        "duration_seconds": audio_seconds,
                        "sample_rate_hz": franken_whisper::sortformer_inference::SORTFORMER_SAMPLE_RATE_HZ,
                        "channels": 1,
                    },
                    "result": {
                        "model_frames": output.frames,
                        "active_lane_count": active_speakers.len(),
                        "capacity": {
                            "speaker_lane_capacity": franken_whisper::sortformer_inference::SORTFORMER_SPEAKER_LANES,
                            "status": "four_lane_capped_output_true_speaker_count_unknown",
                            "true_speaker_count_certified": false,
                        },
                        "speech_frames": output.activity.speech.iter().sum::<i64>(),
                        "overlap_frames": output.activity.overlap.iter().sum::<i64>(),
                        "turns": turns,
                        "identity_mapping": identity_mapping,
                    },
                    "performance": {
                        "package_admission_seconds": package_admission_seconds,
                        "session_materialization_seconds": session_materialization_seconds,
                        "model_load_seconds": package_admission_seconds + session_materialization_seconds,
                        "inference_seconds": inference_seconds,
                        "real_time_factor": real_time_factor,
                    },
                }))?
            );
            Ok(())
        }
        Command::ComparisonWorker => {
            let response =
                franken_whisper::public_corpus::run_model_comparison_worker_from_stdio()?;
            println!("{response}");
            Ok(())
        }
        Command::ComparisonCancelProbe(args) => {
            franken_whisper::public_corpus::run_model_comparison_cancel_probe(
                args.descendant,
                args.lease_parent,
                args.root_pid_file.as_deref(),
                args.descendant_pid_file.as_deref(),
            )
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
    let run_ids: Vec<String> = summaries.iter().map(|s| s.run_id.clone()).collect();
    // Two batched queries instead of the per-run N+1 (`load_run_details` × N).
    let details = store.load_run_details_batch(&run_ids)?;
    if details.len() != run_ids.len() {
        // Preserve the per-run error for any run that vanished between the list and
        // the batched load.
        let found: std::collections::HashSet<&str> =
            details.iter().map(|d| d.run_id.as_str()).collect();
        for id in &run_ids {
            if !found.contains(id.as_str()) {
                return Err(FwError::Storage(format!(
                    "run `{id}` disappeared while loading routing history"
                )));
            }
        }
    }
    Ok(details)
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
                diarization: None,
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
    fn sortformer_audio_duration_uses_an_inclusive_millisecond_ceiling() {
        assert_eq!(super::sortformer_audio_duration_ms(0).expect("zero"), 0);
        assert_eq!(
            super::sortformer_audio_duration_ms(1).expect("one sample"),
            1
        );
        assert_eq!(
            super::sortformer_audio_duration_ms(16_000).expect("one second"),
            1_000
        );
        assert_eq!(
            super::sortformer_audio_duration_ms(16_001).expect("one second plus one sample"),
            1_001
        );
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

        let connection =
            franken_whisper::storage::BlockingConnection::open(db_path.display().to_string())
                .expect("conn");
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
