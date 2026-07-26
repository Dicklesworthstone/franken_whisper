use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::json;

use crate::error::{FwError, FwResult};
use crate::model::{BackendKind, TranscribeRequest, TranscriptionResult, TranscriptionSegment};
use crate::process::{command_exists, run_command_cancellable, run_command_with_timeout};

const DEFAULT_PYTHON_BIN: &str = "python3";

pub fn is_available() -> bool {
    script_path().exists() && command_exists(&python_bin())
}

pub fn run(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<TranscriptionResult> {
    let python = python_bin();
    let script = script_path();
    if !script.exists() {
        return Err(FwError::BackendUnavailable(format!(
            "whisper-diarization script missing at {}",
            script.display()
        )));
    }

    let local_audio = work_dir.join("diarization_input.wav");
    fs::copy(normalized_wav, &local_audio)?;

    let bp = &request.backend_params;

    let mut args = vec![
        script.display().to_string(),
        "-a".to_owned(),
        local_audio.display().to_string(),
    ];

    if let Some(language) = &request.language {
        args.push("--language".to_owned());
        args.push(language.clone());
    }

    // Phase 3: diarization config options.
    if let Some(dc) = &bp.diarization_config {
        if dc.no_stem {
            args.push("--no-stem".to_owned());
        }
        if let Some(model) = &dc.whisper_model {
            args.push("--whisper-model".to_owned());
            args.push(model.clone());
        }
        if dc.suppress_numerals {
            args.push("--suppress_numerals".to_owned());
        }
        if let Some(device) = &dc.device {
            args.push("--device".to_owned());
            args.push(device.clone());
        }
        if let Some(batch) = dc.batch_size {
            args.push("--batch-size".to_owned());
            args.push(batch.to_string());
        }
    }

    // Fall back to env-based device override when no diarization config.
    if bp
        .diarization_config
        .as_ref()
        .and_then(|dc| dc.device.as_ref())
        .is_none()
        && let Ok(device) = std::env::var("FRANKEN_WHISPER_DIARIZATION_DEVICE")
        && !device.trim().is_empty()
    {
        args.push("--device".to_owned());
        args.push(device);
    }

    // GPU device override from top-level param (if not already set via diarization config).
    if bp
        .diarization_config
        .as_ref()
        .and_then(|dc| dc.device.as_ref())
        .is_none()
        && !args.contains(&"--device".to_owned())
        && let Some(device) = &bp.gpu_device
    {
        args.push("--device".to_owned());
        args.push(device.clone());
    }

    // Batch size from top-level if diarization config doesn't have it.
    if bp
        .diarization_config
        .as_ref()
        .and_then(|dc| dc.batch_size)
        .is_none()
        && let Some(batch) = bp.batch_size
    {
        args.push("--batch-size".to_owned());
        args.push(batch.to_string());
    }

    // bd-1rj.4: Forced alignment configuration.
    if let Some(align) = &bp.alignment {
        if let Some(model) = &align.alignment_model {
            args.push("--align-model".to_owned());
            args.push(model.clone());
        }
        if let Some(method) = &align.interpolate_method {
            args.push("--interpolate-method".to_owned());
            args.push(method.clone());
        }
        if align.return_char_alignments {
            args.push("--return-char-alignments".to_owned());
        }
    }

    // bd-1rj.4: Punctuation restoration.
    if let Some(punct) = &bp.punctuation
        && punct.enabled
    {
        args.push("--punctuation-restore".to_owned());
        if let Some(model) = &punct.model {
            args.push(model.clone());
        } else {
            args.push("True".to_owned());
        }
    }

    // bd-1rj.4: Source separation (Demucs).
    if let Some(sep) = &bp.source_separation {
        if sep.enabled {
            // When source separation is enabled and no_stem is not already set,
            // we ensure stemming is active (do not pass --no-stem).
            // The diarize.py script runs Demucs by default unless --no-stem is passed.
        } else if !args.contains(&"--no-stem".to_owned()) {
            args.push("--no-stem".to_owned());
        }
        if let Some(model) = &sep.model {
            args.push("--demucs-model".to_owned());
            args.push(model.clone());
        }
        if let Some(shifts) = sep.shifts {
            args.push("--demucs-shifts".to_owned());
            args.push(shifts.to_string());
        }
        if let Some(overlap) = sep.overlap {
            args.push("--demucs-overlap".to_owned());
            args.push(overlap.to_string());
        }
    }

    if let Some(tok) = token {
        run_command_cancellable(&python, &args, Some(work_dir), tok, Some(timeout))?;
    } else {
        run_command_with_timeout(&python, &args, Some(work_dir), Some(timeout))?;
    }

    let txt_path = local_audio.with_extension("txt");
    let srt_path = local_audio.with_extension("srt");

    if !txt_path.exists() {
        return Err(FwError::MissingArtifact(txt_path));
    }

    let transcript = fs::read_to_string(&txt_path)?;
    let segments = if srt_path.exists() {
        parse_srt_segments(&fs::read_to_string(&srt_path)?)
    } else {
        vec![TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: transcript.trim().to_owned(),
            speaker: None,
            confidence: None,
        }]
    };

    let raw = json!({
        "transcript_txt": transcript,
        "srt_path": srt_path.display().to_string(),
    });

    Ok(TranscriptionResult {
        backend: BackendKind::WhisperDiarization,
        transcript: raw
            .get("transcript_txt")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        language: request.language.clone(),
        segments,
        acceleration: None,
        raw_output: raw,
        artifact_paths: {
            let mut paths = vec![txt_path.display().to_string()];
            if srt_path.exists() {
                paths.push(srt_path.display().to_string());
            }
            paths
        },
    })
}

fn python_bin() -> String {
    std::env::var("FRANKEN_WHISPER_PYTHON_BIN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_PYTHON_BIN.to_owned())
}

pub(crate) fn python_binary() -> String {
    python_bin()
}

pub(crate) fn script_path_string() -> String {
    script_path().display().to_string()
}

fn script_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("legacy_whisper_diarization/whisper-diarization/diarize.py")
}

fn parse_srt_segments(content: &str) -> Vec<TranscriptionSegment> {
    let mut segments = Vec::new();
    let mut block_lines = Vec::new();

    for raw_line in content.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            if !block_lines.is_empty() {
                if let Some(segment) = parse_srt_lines(&block_lines) {
                    segments.push(segment);
                }
                block_lines.clear();
            }
            continue;
        }
        block_lines.push(line);
    }

    if !block_lines.is_empty()
        && let Some(segment) = parse_srt_lines(&block_lines)
    {
        segments.push(segment);
    }

    segments
}

#[cfg(test)]
fn parse_srt_block(block: &str) -> Option<TranscriptionSegment> {
    let lines: Vec<&str> = block.lines().collect();
    parse_srt_lines(&lines)
}

fn parse_srt_lines(lines: &[&str]) -> Option<TranscriptionSegment> {
    if lines.is_empty() {
        return None;
    }

    let (start, end, text) =
        if let Some(timing_idx) = lines.iter().position(|line| line.contains("-->")) {
            if timing_idx + 1 >= lines.len() {
                return None;
            }

            let timing_line = lines[timing_idx];
            let mut timing = timing_line.split("-->").map(str::trim);
            let start = timing.next().and_then(parse_srt_time);
            let end = timing.next().and_then(parse_srt_time);
            let text = lines[(timing_idx + 1)..].join(" ");
            (start, end, text)
        } else {
            if lines.len() < 3 {
                return None;
            }
            let index_line = lines[0].trim();
            if index_line.is_empty() || !index_line.chars().all(|ch| ch.is_ascii_digit()) {
                return None;
            }
            let text = lines[2..].join(" ");
            (None, None, text)
        };

    let text = text.trim().to_owned();
    if text.is_empty() {
        return None;
    }

    let (speaker, clean_text) = extract_speaker_prefix(&text);

    Some(TranscriptionSegment {
        start_sec: start,
        end_sec: end,
        text: clean_text,
        speaker,
        confidence: None,
    })
}

fn parse_srt_time(value: &str) -> Option<f64> {
    // Support both standard ',' and fallback '.' millisecond separator.
    let (hms, ms_str) = if let Some(pos) = value.rfind(',') {
        (&value[..pos], &value[pos + 1..])
    } else {
        let pos = value.rfind('.')?;
        (&value[..pos], &value[pos + 1..])
    };

    let ms = ms_str.parse::<f64>().ok()?;

    let mut hms_parts = hms.split(':');
    let hours = hms_parts.next()?.parse::<f64>().ok()?;
    let minutes = hms_parts.next()?.parse::<f64>().ok()?;
    let seconds = hms_parts.next()?.parse::<f64>().ok()?;

    Some((hours * 3600.0) + (minutes * 60.0) + seconds + (ms / 1_000.0))
}

fn extract_speaker_prefix(text: &str) -> (Option<String>, String) {
    let trimmed = text.trim();

    // Common diarization format: "[SPEAKER_00] hello world"
    if let Some(rest) = trimmed.strip_prefix('[')
        && let Some((head, tail)) = rest.split_once(']')
    {
        let speaker = head.trim();
        let clean_tail = tail.trim_start_matches([':', '-', '|', ' ']).trim();
        if is_speaker_label(speaker) && !clean_tail.is_empty() {
            return (Some(speaker.to_owned()), clean_tail.to_owned());
        }
    }

    for separator in [":", "-", "|"] {
        let mut parts = trimmed.splitn(2, separator);
        let head = parts.next().unwrap_or_default().trim();
        let tail = parts.next().map(str::trim).unwrap_or_default();

        if is_speaker_label(head) && !tail.is_empty() {
            return (Some(head.to_owned()), tail.to_owned());
        }
    }

    (None, trimmed.to_owned())
}

fn is_speaker_label(label: &str) -> bool {
    let lowered = label.trim().to_ascii_lowercase();
    lowered.starts_with("speaker")
        || lowered.starts_with("spk")
        || lowered.starts_with("spkr")
        || matches_short_speaker_label(&lowered)
}

/// Matches compact speaker labels like "s0", "s1", "s02".
fn matches_short_speaker_label(lowered: &str) -> bool {
    let lowered = lowered.trim();
    if lowered.len() >= 2
        && lowered.starts_with('s')
        && lowered[1..].chars().all(|c| c.is_ascii_digit())
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{parse_srt_block, parse_srt_time};

    #[test]
    fn parses_srt_timestamp() {
        let value = parse_srt_time("00:01:02,500").expect("timestamp should parse");
        assert!((value - 62.5).abs() < f64::EPSILON);
    }

    #[test]
    fn parses_srt_block_with_speaker() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\nSpeaker 0: hello there";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.speaker.as_deref(), Some("Speaker 0"));
        assert_eq!(segment.text, "hello there");
    }

    #[test]
    fn parses_srt_block_with_hyphen_speaker_prefix() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\nSPEAKER_01 - check in";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segment.text, "check in");
    }

    #[test]
    fn parses_srt_block_with_bracket_speaker_prefix() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\n[SPEAKER_01] check in";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(segment.text, "check in");
    }

    #[test]
    fn parses_srt_block_with_spk_pipe_prefix() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\nspk2 | all good";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.speaker.as_deref(), Some("spk2"));
        assert_eq!(segment.text, "all good");
    }

    // Phase 3: dot-separated SRT timestamps
    #[test]
    fn parses_srt_timestamp_with_dot_separator() {
        let value = parse_srt_time("00:01:02.500").expect("dot timestamp should parse");
        assert!((value - 62.5).abs() < f64::EPSILON);
    }

    #[test]
    fn parses_srt_block_with_dot_timestamps() {
        let block = "1\n00:00:01.000 --> 00:00:02.500\nhello world";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert!((segment.start_sec.unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((segment.end_sec.unwrap() - 2.5).abs() < f64::EPSILON);
        assert_eq!(segment.text, "hello world");
    }

    #[test]
    fn malformed_timestamp_returns_none() {
        assert!(parse_srt_time("not_a_time").is_none());
        assert!(parse_srt_time("00:01:02").is_none()); // no ms separator
    }

    // Phase 3: expanded speaker label patterns
    #[test]
    fn parses_spkr_prefix() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\nSPKR_1: testing";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.speaker.as_deref(), Some("SPKR_1"));
        assert_eq!(segment.text, "testing");
    }

    #[test]
    fn parses_short_s0_prefix() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\ns0: short label";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.speaker.as_deref(), Some("s0"));
        assert_eq!(segment.text, "short label");
    }

    #[test]
    fn parses_short_s02_prefix() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\ns02 - double digit";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.speaker.as_deref(), Some("s02"));
        assert_eq!(segment.text, "double digit");
    }

    #[test]
    fn parses_speaker_label_mixed_case_and_whitespace() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\n  sPeAkEr_00  :  mixed case and spaces";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.speaker.as_deref(), Some("sPeAkEr_00"));
        assert_eq!(segment.text, "mixed case and spaces");
    }

    #[test]
    fn non_speaker_prefix_has_no_speaker() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\njust text without speaker label";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert!(segment.speaker.is_none());
        assert_eq!(segment.text, "just text without speaker label");
    }

    // --- Malformed payload resilience (J2.3/J2.6) ---

    #[test]
    fn empty_srt_block_returns_none() {
        assert!(parse_srt_block("").is_none());
    }

    #[test]
    fn single_line_srt_block_returns_none() {
        assert!(parse_srt_block("just one line").is_none());
    }

    #[test]
    fn two_line_srt_block_returns_none() {
        assert!(parse_srt_block("1\n00:00:01,000 --> 00:00:02,000").is_none());
    }

    #[test]
    fn srt_block_with_empty_text_returns_none() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\n   ";
        assert!(parse_srt_block(block).is_none());
    }

    #[test]
    fn srt_block_with_garbage_timing_preserves_text() {
        let block = "1\nnot_a_timing_line\nhello there";
        let segment = parse_srt_block(block).expect("segment should parse despite bad timing");
        assert!(segment.start_sec.is_none());
        assert!(segment.end_sec.is_none());
        assert_eq!(segment.text, "hello there");
    }

    #[test]
    fn srt_block_with_partial_timing_preserves_what_parses() {
        let block = "1\n00:00:01,000 --> garbage\nhello";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert!(segment.start_sec.is_some());
        assert!(segment.end_sec.is_none());
    }

    #[test]
    fn srt_multiline_text_concatenated() {
        let block = "1\n00:00:01,000 --> 00:00:02,000\nfirst line\nsecond line";
        let segment = parse_srt_block(block).expect("segment should parse");
        assert_eq!(segment.text, "first line second line");
    }

    #[test]
    fn srt_with_only_whitespace_and_newlines_returns_empty() {
        use super::parse_srt_segments;
        let content = "\n\n\n  \n\n  \n\n";
        let segments = parse_srt_segments(content);
        assert!(segments.is_empty());
    }

    #[test]
    fn mixed_valid_and_corrupt_blocks_extracts_valid_ones() {
        use super::parse_srt_segments;
        let content = "\
1\n00:00:01,000 --> 00:00:02,000\nvalid block\n\n\
corrupted\n\n\
3\n00:00:03,000 --> 00:00:04,000\nalso valid";
        let segments = parse_srt_segments(content);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "valid block");
        assert_eq!(segments[1].text, "also valid");
    }

    #[test]
    fn timestamp_overflow_values_parse_without_panic() {
        let ts = parse_srt_time("99:99:99,999");
        // Should still parse — the arithmetic produces a large number, not a panic.
        assert!(ts.is_some());
    }

    #[test]
    fn timestamp_negative_hours_still_parses() {
        // The parser does not reject negative values — it treats them as arithmetic.
        // This is acceptable; the pipeline tolerates out-of-range times gracefully.
        let ts = parse_srt_time("-1:00:00,000");
        assert!(ts.is_some(), "negative hours parse without panic");
    }

    // -----------------------------------------------------------------------
    // build_args / device priority tests (construct args via `run()` inputs)
    // -----------------------------------------------------------------------

    // We cannot call `run()` directly (it needs python + script), but we can
    // verify the arg-building logic by extracting the relevant lines into a
    // helper and testing that.  Since `build_args` is inlined inside `run()`,
    // we test the device priority and diarization config logic by constructing
    // a TranscribeRequest and asserting on the args vector that `run()` would
    // produce.  We extract the arg-building portion into a test helper.

    fn build_diarization_args(request: &crate::model::TranscribeRequest) -> Vec<String> {
        let bp = &request.backend_params;
        let mut args = vec![
            "dummy_script.py".to_owned(),
            "-a".to_owned(),
            "audio.wav".to_owned(),
        ];
        if let Some(language) = &request.language {
            args.push("--language".to_owned());
            args.push(language.clone());
        }
        if let Some(dc) = &bp.diarization_config {
            if dc.no_stem {
                args.push("--no-stem".to_owned());
            }
            if let Some(model) = &dc.whisper_model {
                args.push("--whisper-model".to_owned());
                args.push(model.clone());
            }
            if dc.suppress_numerals {
                args.push("--suppress_numerals".to_owned());
            }
            if let Some(device) = &dc.device {
                args.push("--device".to_owned());
                args.push(device.clone());
            }
            if let Some(batch) = dc.batch_size {
                args.push("--batch-size".to_owned());
                args.push(batch.to_string());
            }
        }
        if bp
            .diarization_config
            .as_ref()
            .and_then(|dc| dc.device.as_ref())
            .is_none()
        {
            // In real code this reads FRANKEN_WHISPER_DIARIZATION_DEVICE;
            // we skip env-based fallback in tests and test gpu_device instead.
            if !args.contains(&"--device".to_owned())
                && let Some(device) = &bp.gpu_device
            {
                args.push("--device".to_owned());
                args.push(device.clone());
            }
        }
        if bp
            .diarization_config
            .as_ref()
            .and_then(|dc| dc.batch_size)
            .is_none()
            && let Some(batch) = bp.batch_size
        {
            args.push("--batch-size".to_owned());
            args.push(batch.to_string());
        }

        // bd-248: alignment configuration.
        if let Some(align) = &bp.alignment {
            if let Some(model) = &align.alignment_model {
                args.push("--align-model".to_owned());
                args.push(model.clone());
            }
            if let Some(method) = &align.interpolate_method {
                args.push("--interpolate-method".to_owned());
                args.push(method.clone());
            }
            if align.return_char_alignments {
                args.push("--return-char-alignments".to_owned());
            }
        }

        // bd-248: punctuation restoration.
        if let Some(punct) = &bp.punctuation
            && punct.enabled
        {
            args.push("--punctuation-restore".to_owned());
            if let Some(model) = &punct.model {
                args.push(model.clone());
            } else {
                args.push("True".to_owned());
            }
        }

        if let Some(sep) = &bp.source_separation {
            if sep.enabled {
                // ...
            } else if !args.contains(&"--no-stem".to_owned()) {
                args.push("--no-stem".to_owned());
            }
            if let Some(model) = &sep.model {
                args.push("--demucs-model".to_owned());
                args.push(model.clone());
            }
            if let Some(shifts) = sep.shifts {
                args.push("--demucs-shifts".to_owned());
                args.push(shifts.to_string());
            }
            if let Some(overlap) = sep.overlap {
                args.push("--demucs-overlap".to_owned());
                args.push(overlap.to_string());
            }
        }

        args
    }

    fn minimal_diarization_request() -> crate::model::TranscribeRequest {
        use crate::model::{BackendKind, BackendParams, InputSource};
        crate::model::TranscribeRequest {
            input: InputSource::File {
                path: std::path::PathBuf::from("input.wav"),
            },
            backend: BackendKind::WhisperDiarization,
            model: None,
            language: None,
            translate: false,
            diarize: true,
            persist: false,
            db_path: std::path::PathBuf::from("db.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        }
    }

    fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    }

    fn has_flag(args: &[String], flag: &str) -> bool {
        args.iter().any(|a| a == flag)
    }

    #[test]
    fn diarization_minimal_args_include_script_and_audio() {
        let request = minimal_diarization_request();
        let args = build_diarization_args(&request);
        assert!(args.contains(&"-a".to_owned()));
        assert!(args.contains(&"audio.wav".to_owned()));
        assert!(!has_flag(&args, "--language"));
        assert!(!has_flag(&args, "--no-stem"));
    }

    #[test]
    fn diarization_language_flag() {
        let mut request = minimal_diarization_request();
        request.language = Some("fr".to_owned());
        let args = build_diarization_args(&request);
        assert_eq!(arg_value(&args, "--language"), Some("fr"));
    }

    #[test]
    fn diarization_config_no_stem_and_suppress_numerals() {
        use crate::model::DiarizationConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.diarization_config = Some(DiarizationConfig {
            no_stem: true,
            suppress_numerals: true,
            ..DiarizationConfig::default()
        });
        let args = build_diarization_args(&request);
        assert!(has_flag(&args, "--no-stem"));
        assert!(has_flag(&args, "--suppress_numerals"));
    }

    #[test]
    fn diarization_config_whisper_model_and_batch_size() {
        use crate::model::DiarizationConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.diarization_config = Some(DiarizationConfig {
            whisper_model: Some("large-v2".to_owned()),
            batch_size: Some(16),
            ..DiarizationConfig::default()
        });
        let args = build_diarization_args(&request);
        assert_eq!(arg_value(&args, "--whisper-model"), Some("large-v2"));
        assert_eq!(arg_value(&args, "--batch-size"), Some("16"));
    }

    #[test]
    fn diarization_config_device_takes_priority_over_gpu_device() {
        use crate::model::DiarizationConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.diarization_config = Some(DiarizationConfig {
            device: Some("cuda:1".to_owned()),
            ..DiarizationConfig::default()
        });
        request.backend_params.gpu_device = Some("cuda:0".to_owned());
        let args = build_diarization_args(&request);
        // diarization_config.device wins
        assert_eq!(arg_value(&args, "--device"), Some("cuda:1"));
        // gpu_device should NOT produce a second --device flag
        let device_count = args.iter().filter(|a| a.as_str() == "--device").count();
        assert_eq!(device_count, 1, "only one --device flag should appear");
    }

    #[test]
    fn gpu_device_fallback_when_no_diarization_config_device() {
        let mut request = minimal_diarization_request();
        request.backend_params.gpu_device = Some("mps".to_owned());
        let args = build_diarization_args(&request);
        assert_eq!(arg_value(&args, "--device"), Some("mps"));
    }

    #[test]
    fn top_level_batch_size_fallback_when_diarization_config_has_none() {
        let mut request = minimal_diarization_request();
        request.backend_params.batch_size = Some(8);
        let args = build_diarization_args(&request);
        assert_eq!(arg_value(&args, "--batch-size"), Some("8"));
    }

    #[test]
    fn python_bin_default_is_python3() {
        let name = super::python_bin();
        if std::env::var("FRANKEN_WHISPER_PYTHON_BIN").is_err() {
            assert_eq!(name, "python3");
        }
    }

    #[test]
    fn python_binary_matches_python_bin() {
        assert_eq!(super::python_binary(), super::python_bin());
    }

    #[test]
    fn script_path_ends_with_diarize_py() {
        let path = super::script_path();
        assert!(
            path.ends_with("diarize.py"),
            "script_path should end with diarize.py, got: {}",
            path.display()
        );
    }

    #[test]
    fn script_path_string_is_nonempty() {
        let s = super::script_path_string();
        assert!(!s.is_empty());
        assert!(s.contains("diarize.py"));
    }

    #[test]
    fn extract_speaker_prefix_no_speaker_label() {
        let (speaker, text) = super::extract_speaker_prefix("just regular text");
        assert!(speaker.is_none());
        assert_eq!(text, "just regular text");
    }

    #[test]
    fn extract_speaker_prefix_with_colon() {
        let (speaker, text) = super::extract_speaker_prefix("Speaker 1: hello world");
        assert_eq!(speaker.as_deref(), Some("Speaker 1"));
        assert_eq!(text, "hello world");
    }

    #[test]
    fn extract_speaker_prefix_with_bracket() {
        let (speaker, text) = super::extract_speaker_prefix("[SPEAKER_02] hello world");
        assert_eq!(speaker.as_deref(), Some("SPEAKER_02"));
        assert_eq!(text, "hello world");
    }

    #[test]
    fn extract_speaker_prefix_non_speaker_colon_label() {
        // "Note:" is not a speaker label, so it should not be extracted.
        let (speaker, text) = super::extract_speaker_prefix("Note: this is a note");
        assert!(speaker.is_none());
        assert_eq!(text, "Note: this is a note");
    }

    #[test]
    fn matches_short_speaker_label_valid() {
        assert!(super::matches_short_speaker_label("s0"));
        assert!(super::matches_short_speaker_label("s1"));
        assert!(super::matches_short_speaker_label("s02"));
        assert!(super::matches_short_speaker_label("s123"));
    }

    #[test]
    fn matches_short_speaker_label_invalid() {
        assert!(!super::matches_short_speaker_label(""));
        assert!(!super::matches_short_speaker_label("s"));
        assert!(!super::matches_short_speaker_label("speaker"));
        assert!(!super::matches_short_speaker_label("sa"));
        assert!(!super::matches_short_speaker_label("x1"));
    }

    #[test]
    fn is_speaker_label_all_valid_prefixes() {
        assert!(super::is_speaker_label("speaker"));
        assert!(super::is_speaker_label("Speaker 1"));
        assert!(super::is_speaker_label("SPEAKER_01"));
        assert!(super::is_speaker_label("spk0"));
        assert!(super::is_speaker_label("spkr_1"));
        assert!(super::is_speaker_label("s0"));
        assert!(super::is_speaker_label("s42"));
    }

    #[test]
    fn is_speaker_label_invalid() {
        assert!(!super::is_speaker_label("hello"));
        assert!(!super::is_speaker_label("Note"));
        assert!(!super::is_speaker_label(""));
        assert!(!super::is_speaker_label("s")); // too short for short label
    }

    #[test]
    fn parse_srt_segments_multiple_valid_blocks() {
        use super::parse_srt_segments;
        let content = "\
1\n00:00:01,000 --> 00:00:02,000\nhello\n\n\
2\n00:00:02,000 --> 00:00:03,000\nworld\n\n\
3\n00:00:03,000 --> 00:00:04,000\nfoo";
        let segments = parse_srt_segments(content);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[1].text, "world");
        assert_eq!(segments[2].text, "foo");
    }

    #[test]
    fn parse_srt_segments_handles_crlf_and_blank_lines() {
        use super::parse_srt_segments;
        let content = "1\r\n00:00:01,000 --> 00:00:02,000\r\nhello\r\n\r\n\
2\r\n00:00:02,000 --> 00:00:03,000\r\nworld\r\n\r\n   \r\n\
3\r\n00:00:03,000 --> 00:00:04,000\r\nfoo\r\n";
        let segments = parse_srt_segments(content);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[1].text, "world");
        assert_eq!(segments[2].text, "foo");
    }

    #[test]
    fn parse_srt_segments_without_index_line() {
        use super::parse_srt_segments;
        let content = "\
00:00:01,000 --> 00:00:02,000\nhello\n\n\
00:00:02,000 --> 00:00:03,000\nworld";
        let segments = parse_srt_segments(content);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[1].text, "world");
    }

    #[test]
    fn parse_srt_segments_empty_input() {
        use super::parse_srt_segments;
        assert!(parse_srt_segments("").is_empty());
    }

    #[test]
    fn diarization_config_batch_size_wins_over_top_level() {
        use crate::model::DiarizationConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.diarization_config = Some(DiarizationConfig {
            batch_size: Some(32),
            ..DiarizationConfig::default()
        });
        request.backend_params.batch_size = Some(8);
        let args = build_diarization_args(&request);
        // Only the diarization config batch_size=32 should appear
        assert_eq!(arg_value(&args, "--batch-size"), Some("32"));
        let batch_count = args.iter().filter(|a| a.as_str() == "--batch-size").count();
        assert_eq!(batch_count, 1, "only one --batch-size flag should appear");
    }

    #[test]
    fn source_separation_disabled_adds_no_stem() {
        use crate::model::SourceSeparationConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.source_separation = Some(SourceSeparationConfig {
            enabled: false,
            model: None,
            shifts: None,
            overlap: None,
        });
        let args = build_diarization_args(&request);
        assert!(
            has_flag(&args, "--no-stem"),
            "must add --no-stem if source separation is explicitly disabled"
        );
    }

    // -- bd-248: whisper_diarization.rs edge-case tests pass 2 --

    #[test]
    fn alignment_flags_all_three_emitted() {
        use crate::model::AlignmentConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.alignment = Some(AlignmentConfig {
            alignment_model: Some("WAV2VEC2_ASR_LARGE_LV60K_960H".to_owned()),
            interpolate_method: Some("nearest".to_owned()),
            return_char_alignments: true,
        });
        let args = build_diarization_args(&request);
        assert_eq!(
            arg_value(&args, "--align-model"),
            Some("WAV2VEC2_ASR_LARGE_LV60K_960H")
        );
        assert_eq!(arg_value(&args, "--interpolate-method"), Some("nearest"));
        assert!(
            has_flag(&args, "--return-char-alignments"),
            "should include --return-char-alignments"
        );
    }

    #[test]
    fn punctuation_no_model_pushes_true_literal() {
        use crate::model::PunctuationConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.punctuation = Some(PunctuationConfig {
            enabled: true,
            model: None,
        });
        let args = build_diarization_args(&request);
        let punct_idx = args
            .iter()
            .position(|a| a == "--punctuation-restore")
            .expect("should contain --punctuation-restore");
        assert_eq!(
            args[punct_idx + 1],
            "True",
            "should push literal 'True' when no model is specified"
        );
    }

    #[test]
    fn punctuation_with_model_pushes_model_name() {
        use crate::model::PunctuationConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.punctuation = Some(PunctuationConfig {
            enabled: true,
            model: Some("kredor/punctuate-all".to_owned()),
        });
        let args = build_diarization_args(&request);
        let punct_idx = args
            .iter()
            .position(|a| a == "--punctuation-restore")
            .expect("should contain --punctuation-restore");
        assert_eq!(
            args[punct_idx + 1],
            "kredor/punctuate-all",
            "should push model name when specified"
        );
    }

    #[test]
    fn source_separation_enabled_emits_demucs_flags() {
        use crate::model::SourceSeparationConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.source_separation = Some(SourceSeparationConfig {
            enabled: true,
            model: Some("htdemucs".to_owned()),
            shifts: Some(4),
            overlap: Some(0.25),
        });
        let args = build_diarization_args(&request);
        assert_eq!(arg_value(&args, "--demucs-model"), Some("htdemucs"));
        assert_eq!(arg_value(&args, "--demucs-shifts"), Some("4"));
        assert_eq!(arg_value(&args, "--demucs-overlap"), Some("0.25"));
        // When enabled=true, --no-stem should NOT be present.
        assert!(
            !has_flag(&args, "--no-stem"),
            "--no-stem should not be present when source separation is enabled"
        );
    }

    #[test]
    fn alignment_return_char_alignments_only() {
        use crate::model::AlignmentConfig;
        let mut request = minimal_diarization_request();
        request.backend_params.alignment = Some(AlignmentConfig {
            alignment_model: None,
            interpolate_method: None,
            return_char_alignments: true,
        });
        let args = build_diarization_args(&request);
        assert!(
            has_flag(&args, "--return-char-alignments"),
            "should include --return-char-alignments even without model/method"
        );
        assert!(
            !has_flag(&args, "--align-model"),
            "--align-model should not appear when alignment_model is None"
        );
        assert!(
            !has_flag(&args, "--interpolate-method"),
            "--interpolate-method should not appear when interpolate_method is None"
        );
    }
}
