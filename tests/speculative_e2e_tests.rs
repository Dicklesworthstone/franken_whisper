//! E2E speculative pipeline tests with CLI flag verification (bd-qlt.17).
//!
//! Tests the end-to-end flow from CLI argument parsing through pipeline
//! execution to event output, covering:
//! - CLI flag parsing and validation (clap `requires` constraints)
//! - `to_speculative_config()` building correct configs from args
//! - Full pipeline construction and execution with mock closures
//! - Robot event formatting
//! - Stats and report generation

#![forbid(unsafe_code)]

use clap::Parser;

use franken_whisper::cli::{Cli, Command};
use franken_whisper::model::TranscriptionSegment;
use franken_whisper::robot;
use franken_whisper::speculation::{
    CorrectionDecision, CorrectionEvent, CorrectionTolerance, SpeculationStats,
};
use franken_whisper::streaming::{SpeculativeConfig, SpeculativeStreamingPipeline};

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

fn make_seg(text: &str, conf: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(conf),
    }
}

fn make_seg_at(text: &str, start: f64, end: f64, conf: f64) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(start),
        end_sec: Some(end),
        text: text.to_owned(),
        speaker: None,
        confidence: Some(conf),
    }
}

fn default_config() -> SpeculativeConfig {
    SpeculativeConfig {
        window_size_ms: 3000,
        overlap_ms: 500,
        fast_model_name: "test-fast".to_owned(),
        quality_model_name: "test-quality".to_owned(),
        tolerance: CorrectionTolerance::default(),
        adaptive: false,
        emit_events: true,
    }
}

// ============================================================================
// CLI argument validation
// ============================================================================

#[test]
fn e2e_speculative_flag_alone_accepted() {
    let cli = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--speculative",
    ])
    .expect("--speculative alone should parse");
    match cli.command {
        Command::Transcribe(args) => {
            assert!(args.speculative, "speculative flag should be true");
        }
        other => panic!("expected Transcribe, got: {other:?}"),
    }
}

#[test]
fn e2e_speculative_with_all_options() {
    let cli = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--speculative",
        "--fast-model",
        "tiny",
        "--quality-model",
        "large-v3",
        "--speculative-window-ms",
        "5000",
        "--speculative-overlap-ms",
        "1000",
        "--correction-tolerance-wer",
        "0.25",
        "--no-adaptive",
        "--always-correct",
    ])
    .expect("all speculative sub-options should parse");
    match cli.command {
        Command::Transcribe(args) => {
            assert!(args.speculative);
            assert_eq!(args.fast_model.as_deref(), Some("tiny"));
            assert_eq!(args.quality_model.as_deref(), Some("large-v3"));
            assert_eq!(args.speculative_window_ms, Some(5000));
            assert_eq!(args.speculative_overlap_ms, Some(1000));
            assert_eq!(args.correction_tolerance_wer, Some(0.25));
            assert!(args.no_adaptive);
            assert!(args.always_correct);
        }
        other => panic!("expected Transcribe, got: {other:?}"),
    }
}

#[test]
fn e2e_fast_model_without_speculative_rejected() {
    let result = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--fast-model",
        "tiny",
    ]);
    assert!(
        result.is_err(),
        "--fast-model without --speculative should fail"
    );
}

#[test]
fn e2e_quality_model_without_speculative_rejected() {
    let result = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--quality-model",
        "large",
    ]);
    assert!(
        result.is_err(),
        "--quality-model without --speculative should fail"
    );
}

#[test]
fn e2e_window_ms_without_speculative_rejected() {
    let result = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--speculative-window-ms",
        "5000",
    ]);
    assert!(
        result.is_err(),
        "--speculative-window-ms without --speculative should fail"
    );
}

#[test]
fn e2e_no_adaptive_without_speculative_rejected() {
    let result = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--no-adaptive",
    ]);
    assert!(
        result.is_err(),
        "--no-adaptive without --speculative should fail"
    );
}

#[test]
fn e2e_always_correct_without_speculative_rejected() {
    let result = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--always-correct",
    ]);
    assert!(
        result.is_err(),
        "--always-correct without --speculative should fail"
    );
}

#[test]
fn e2e_speculative_preserves_other_flags() {
    let cli = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--speculative",
        "--language",
        "en",
        "--model",
        "large",
    ])
    .expect("speculative combined with other flags should parse");
    match cli.command {
        Command::Transcribe(args) => {
            assert!(args.speculative);
            assert_eq!(args.language.as_deref(), Some("en"));
            assert_eq!(args.model.as_deref(), Some("large"));
        }
        other => panic!("expected Transcribe, got: {other:?}"),
    }
}

// ============================================================================
// Config construction from CLI args
// ============================================================================

#[test]
fn e2e_config_defaults_from_bare_speculative() {
    let cli = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--speculative",
    ])
    .expect("should parse");
    match cli.command {
        Command::Transcribe(args) => {
            let config = args
                .to_speculative_config()
                .expect("geometry should be valid")
                .expect("should produce config when speculative=true");
            assert_eq!(config.window_size_ms, 3000);
            assert_eq!(config.overlap_ms, 500);
            assert_eq!(config.fast_model_name, "auto-fast");
            assert_eq!(config.quality_model_name, "auto-quality");
            assert!(config.adaptive, "adaptive should default to true");
            assert!(!config.tolerance.always_correct);
            assert!(config.emit_events);
        }
        other => panic!("expected Transcribe, got: {other:?}"),
    }
}

#[test]
fn e2e_config_custom_values_propagate() {
    let cli = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--speculative",
        "--fast-model",
        "whisper-tiny",
        "--quality-model",
        "whisper-large",
        "--speculative-window-ms",
        "5000",
        "--speculative-overlap-ms",
        "1000",
        "--correction-tolerance-wer",
        "0.25",
        "--no-adaptive",
    ])
    .expect("should parse");
    match cli.command {
        Command::Transcribe(args) => {
            let config = args
                .to_speculative_config()
                .expect("geometry should be valid")
                .expect("should produce config");
            assert_eq!(config.window_size_ms, 5000);
            assert_eq!(config.overlap_ms, 1000);
            assert_eq!(config.fast_model_name, "whisper-tiny");
            assert_eq!(config.quality_model_name, "whisper-large");
            assert!(!config.adaptive, "no_adaptive should disable adaptive");
            assert!(
                (config.tolerance.max_wer - 0.25).abs() < f64::EPSILON,
                "max_wer should be 0.25, got {}",
                config.tolerance.max_wer
            );
        }
        other => panic!("expected Transcribe, got: {other:?}"),
    }
}

#[test]
fn e2e_config_none_when_not_speculative() {
    let cli = Cli::try_parse_from(["franken_whisper", "transcribe", "--input", "test.wav"])
        .expect("should parse");
    match cli.command {
        Command::Transcribe(args) => {
            assert!(
                args.to_speculative_config()
                    .expect("disabled speculative config is valid")
                    .is_none(),
                "should return None when speculative=false"
            );
        }
        other => panic!("expected Transcribe, got: {other:?}"),
    }
}

#[test]
fn e2e_config_always_correct_sets_tolerance() {
    let cli = Cli::try_parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "test.wav",
        "--speculative",
        "--always-correct",
    ])
    .expect("should parse");
    match cli.command {
        Command::Transcribe(args) => {
            let config = args
                .to_speculative_config()
                .expect("geometry should be valid")
                .expect("should produce config");
            assert!(
                config.tolerance.always_correct,
                "always_correct should be true"
            );
        }
        other => panic!("expected Transcribe, got: {other:?}"),
    }
}

// ============================================================================
// Full pipeline execution
// ============================================================================

#[test]
fn e2e_pipeline_confirm_run_produces_report() {
    let config = SpeculativeConfig {
        tolerance: CorrectionTolerance {
            max_wer: 0.5,
            max_confidence_delta: 0.5,
            max_edit_distance: 100,
            always_correct: false,
        },
        ..default_config()
    };
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-confirm".to_owned());

    // Process 3 windows with identical outputs -> all confirmed.
    for i in 0..3 {
        let text = format!("segment {i}");
        let t = text.clone();
        let t2 = text.clone();
        let result = pipeline
            .process_window(
                "hash-a",
                i * 3000,
                move || vec![make_seg(&t, 0.75)],
                move || vec![make_seg(&t2, 0.75)],
            )
            .expect("should succeed");
        assert!(
            matches!(result, CorrectionDecision::Confirm { .. }),
            "window {i} should be confirmed"
        );
    }

    let stats = pipeline.stats();
    assert_eq!(stats.windows_processed, 3);
    assert_eq!(stats.corrections_emitted, 0);
    assert_eq!(stats.confirmations_emitted, 3);
    assert!(
        stats.correction_rate < f64::EPSILON,
        "correction_rate should be 0.0"
    );
}

#[test]
fn e2e_pipeline_correct_run_produces_corrections() {
    let config = SpeculativeConfig {
        tolerance: CorrectionTolerance {
            max_wer: 0.0,
            max_confidence_delta: 0.0,
            max_edit_distance: 0,
            always_correct: true,
        },
        ..default_config()
    };
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-correct".to_owned());

    // Process 2 windows with always_correct -> all should be corrections.
    for i in 0..2 {
        let fast_text = format!("fast {i}");
        let quality_text = format!("quality {i}");
        let ft = fast_text.clone();
        let qt = quality_text.clone();
        let result = pipeline
            .process_window(
                "hash-b",
                i * 3000,
                move || vec![make_seg(&ft, 0.5)],
                move || vec![make_seg(&qt, 0.75)],
            )
            .expect("should succeed");
        assert!(
            matches!(result, CorrectionDecision::Correct { .. }),
            "window {i} should be corrected (always_correct=true)"
        );
    }

    let stats = pipeline.stats();
    assert_eq!(stats.windows_processed, 2);
    assert_eq!(stats.corrections_emitted, 2);
    assert_eq!(stats.confirmations_emitted, 0);
    assert!(
        (stats.correction_rate - 1.0).abs() < f64::EPSILON,
        "correction_rate should be 1.0"
    );
}

#[test]
fn e2e_pipeline_mixed_run_stats_accurate() {
    // Use tolerant thresholds so identical text confirms, different text corrects.
    let config = SpeculativeConfig {
        tolerance: CorrectionTolerance {
            max_wer: 0.25,
            max_confidence_delta: 0.5,
            max_edit_distance: 100,
            always_correct: false,
        },
        ..default_config()
    };
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-mixed".to_owned());

    // Window 0: identical -> confirm.
    let result0 = pipeline
        .process_window(
            "hash-c",
            0,
            || vec![make_seg("hello world", 0.75)],
            || vec![make_seg("hello world", 0.75)],
        )
        .expect("window 0");
    assert!(matches!(result0, CorrectionDecision::Confirm { .. }));

    // Window 1: completely different -> correct.
    let result1 = pipeline
        .process_window(
            "hash-c",
            3000,
            || vec![make_seg("alpha beta gamma delta", 0.5)],
            || vec![make_seg("one two three four five", 0.75)],
        )
        .expect("window 1");
    assert!(matches!(result1, CorrectionDecision::Correct { .. }));

    // Window 2: identical -> confirm.
    let result2 = pipeline
        .process_window(
            "hash-c",
            6000,
            || vec![make_seg("final segment", 0.75)],
            || vec![make_seg("final segment", 0.75)],
        )
        .expect("window 2");
    assert!(matches!(result2, CorrectionDecision::Confirm { .. }));

    let stats = pipeline.stats();
    assert_eq!(stats.windows_processed, 3);
    assert_eq!(stats.corrections_emitted, 1);
    assert_eq!(stats.confirmations_emitted, 2);

    // correction_rate = 1/3
    let expected_rate = 1.0 / 3.0;
    assert!(
        (stats.correction_rate - expected_rate).abs() < 0.001,
        "correction_rate should be ~0.333, got {}",
        stats.correction_rate
    );
}

#[test]
fn e2e_pipeline_events_monotonic_seq() {
    let config = default_config();
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-monotonic".to_owned());

    for i in 0..4 {
        let text = format!("win {i}");
        let t1 = text.clone();
        let t2 = text.clone();
        pipeline
            .process_window(
                "hash-d",
                i * 3000,
                move || vec![make_seg(&t1, 0.75)],
                move || vec![make_seg(&t2, 0.75)],
            )
            .expect("should succeed");
    }

    let events = pipeline.events();
    assert!(!events.is_empty(), "pipeline should have generated events");

    // Verify event seq values are monotonically increasing.
    for pair in events.windows(2) {
        assert!(
            pair[1].seq > pair[0].seq,
            "event seq should be monotonically increasing: {} should be > {}",
            pair[1].seq,
            pair[0].seq
        );
    }
}

#[test]
fn e2e_pipeline_merged_transcript_complete() {
    let config = SpeculativeConfig {
        tolerance: CorrectionTolerance {
            max_wer: 0.5,
            max_confidence_delta: 0.5,
            max_edit_distance: 100,
            always_correct: false,
        },
        ..default_config()
    };
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-merged".to_owned());

    // Process 3 windows with unique, non-overlapping quality segments.
    let segments_data = [("hello", 0.0, 1.0), ("world", 2.0, 3.0), ("test", 4.0, 5.0)];
    for (i, (text, start, end)) in segments_data.iter().enumerate() {
        let t1 = text.to_string();
        let t2 = text.to_string();
        let s = *start;
        let e = *end;
        pipeline
            .process_window(
                "hash-e",
                (i as u64) * 3000,
                move || vec![make_seg_at(&t1, s, e, 0.5)],
                move || vec![make_seg_at(&t2, s, e, 0.75)],
            )
            .expect("should succeed");
    }

    let merged = pipeline.merged_transcript();
    assert_eq!(
        merged.len(),
        3,
        "merged transcript should have 3 segments, got {}",
        merged.len()
    );

    // Verify all quality segments are present.
    let texts: Vec<&str> = merged.iter().map(|s| s.text.as_str()).collect();
    assert!(texts.contains(&"hello"), "missing 'hello'");
    assert!(texts.contains(&"world"), "missing 'world'");
    assert!(texts.contains(&"test"), "missing 'test'");
}

// ============================================================================
// Robot event formatting
// ============================================================================

#[test]
fn e2e_robot_retract_event_valid_json() {
    let value = robot::transcript_retract_value("run-42", 7, 3, "quality_correction", "large-v3");
    assert!(value.is_object(), "should be a JSON object");
    assert_eq!(value["event"], "transcript.retract");
    assert_eq!(value["run_id"], "run-42");
    assert_eq!(value["retracted_seq"], 7);
    assert_eq!(value["window_id"], 3);
    assert_eq!(value["reason"], "quality_correction");
    assert_eq!(value["quality_model_id"], "large-v3");
    assert!(value.get("ts").is_some(), "should have a ts field");
    assert!(
        value.get("schema_version").is_some(),
        "should have schema_version"
    );
}

#[test]
fn e2e_robot_correct_event_valid_json() {
    let correction = CorrectionEvent::new(
        1,
        5,
        2,
        "large-v3".to_owned(),
        vec![make_seg("corrected text", 0.75)],
        150,
        "2026-02-22T12:00:00Z".to_owned(),
        &[make_seg("fast text", 0.5)],
    );

    let value = robot::transcript_correct_value("run-99", &correction);
    assert!(value.is_object(), "should be a JSON object");
    assert_eq!(value["event"], "transcript.correct");
    assert_eq!(value["run_id"], "run-99");
    assert_eq!(value["correction_id"], 1);
    assert_eq!(value["replaces_seq"], 5);
    assert_eq!(value["window_id"], 2);
    assert!(value["segments"].is_array(), "segments should be an array");
    assert!(value.get("drift").is_some(), "should have drift");
    assert_eq!(value["latency_ms"], 150);
    assert!(value.get("ts").is_some(), "should have ts");
}

#[test]
fn e2e_robot_speculation_stats_valid_json() {
    let stats = SpeculationStats {
        windows_processed: 10,
        corrections_emitted: 3,
        confirmations_emitted: 7,
        correction_rate: 0.25,
        mean_fast_latency_ms: 50.0,
        mean_quality_latency_ms: 200.0,
        current_window_size_ms: 3000,
        mean_drift_wer: 0.125,
    };

    let value = robot::speculation_stats_value("run-stats", &stats);
    assert!(value.is_object(), "should be a JSON object");
    assert_eq!(value["event"], "transcript.speculation_stats");
    assert_eq!(value["run_id"], "run-stats");
    assert_eq!(value["windows_processed"], 10);
    assert_eq!(value["corrections_emitted"], 3);
    assert_eq!(value["confirmations_emitted"], 7);
    assert_eq!(value["correction_rate"], 0.25);
    assert_eq!(value["mean_fast_latency_ms"], 50.0);
    assert_eq!(value["mean_quality_latency_ms"], 200.0);
    assert_eq!(value["current_window_size_ms"], 3000);
    assert_eq!(value["mean_drift_wer"], 0.125);
    assert!(value.get("ts").is_some(), "should have ts");
}

#[test]
fn e2e_robot_events_have_required_fields() {
    assert!(
        !robot::TRANSCRIPT_RETRACT_REQUIRED_FIELDS.is_empty(),
        "retract required fields should be non-empty"
    );
    assert!(
        !robot::TRANSCRIPT_CORRECT_REQUIRED_FIELDS.is_empty(),
        "correct required fields should be non-empty"
    );
    assert!(
        !robot::SPECULATION_STATS_REQUIRED_FIELDS.is_empty(),
        "stats required fields should be non-empty"
    );

    // Verify retract value contains all required fields.
    let retract = robot::transcript_retract_value("run-1", 0, 0, "test", "model");
    for field in robot::TRANSCRIPT_RETRACT_REQUIRED_FIELDS {
        assert!(
            retract.get(field).is_some(),
            "retract event missing required field: {field}"
        );
    }

    // Verify correct value contains all required fields.
    let correction = CorrectionEvent::new(
        0,
        0,
        0,
        "model".to_owned(),
        vec![make_seg("text", 0.75)],
        100,
        "2026-02-22T00:00:00Z".to_owned(),
        &[make_seg("fast", 0.5)],
    );
    let correct = robot::transcript_correct_value("run-1", &correction);
    for field in robot::TRANSCRIPT_CORRECT_REQUIRED_FIELDS {
        assert!(
            correct.get(field).is_some(),
            "correct event missing required field: {field}"
        );
    }

    // Verify stats value contains all required fields.
    let stats = SpeculationStats {
        windows_processed: 0,
        corrections_emitted: 0,
        confirmations_emitted: 0,
        correction_rate: 0.0,
        mean_fast_latency_ms: 0.0,
        mean_quality_latency_ms: 0.0,
        current_window_size_ms: 3000,
        mean_drift_wer: 0.0,
    };
    let stats_value = robot::speculation_stats_value("run-1", &stats);
    for field in robot::SPECULATION_STATS_REQUIRED_FIELDS {
        assert!(
            stats_value.get(field).is_some(),
            "stats event missing required field: {field}"
        );
    }
}

// ============================================================================
// Report generation
// ============================================================================

#[test]
fn e2e_report_includes_correction_events() {
    let config = SpeculativeConfig {
        tolerance: CorrectionTolerance {
            always_correct: true,
            ..CorrectionTolerance::default()
        },
        ..default_config()
    };
    let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-events".to_owned());

    // Process a window that will be corrected (always_correct=true).
    pipeline
        .process_window(
            "hash-f",
            0,
            || vec![make_seg("fast text", 0.5)],
            || vec![make_seg("quality text", 0.75)],
        )
        .expect("should succeed");

    let events = pipeline.events();
    let has_retract = events.iter().any(|e| e.code == "transcript.retract");
    let has_correct = events.iter().any(|e| e.code == "transcript.correct");

    assert!(
        has_retract,
        "events should include transcript.retract after correction"
    );
    assert!(
        has_correct,
        "events should include transcript.correct after correction"
    );
}

#[test]
fn e2e_pipeline_deterministic_output() {
    // Run the same input twice and verify both produce the same correction decisions.
    let make_pipeline = || {
        let config = SpeculativeConfig {
            tolerance: CorrectionTolerance {
                max_wer: 0.25,
                max_confidence_delta: 0.5,
                max_edit_distance: 100,
                always_correct: false,
            },
            ..default_config()
        };
        SpeculativeStreamingPipeline::new(config, "run-deterministic".to_owned())
    };

    let run = |pipeline: &mut SpeculativeStreamingPipeline| -> Vec<bool> {
        let mut decisions = Vec::new();
        // Window 0: identical -> confirm.
        let r0 = pipeline
            .process_window(
                "hash-g",
                0,
                || vec![make_seg("same text", 0.75)],
                || vec![make_seg("same text", 0.75)],
            )
            .expect("w0");
        decisions.push(matches!(r0, CorrectionDecision::Confirm { .. }));

        // Window 1: different -> correct.
        let r1 = pipeline
            .process_window(
                "hash-g",
                3000,
                || vec![make_seg("completely different alpha beta gamma", 0.5)],
                || vec![make_seg("entirely other one two three four", 0.75)],
            )
            .expect("w1");
        decisions.push(matches!(r1, CorrectionDecision::Confirm { .. }));

        decisions
    };

    let mut p1 = make_pipeline();
    let mut p2 = make_pipeline();
    let d1 = run(&mut p1);
    let d2 = run(&mut p2);

    assert_eq!(d1, d2, "same input should produce same decisions");
}
