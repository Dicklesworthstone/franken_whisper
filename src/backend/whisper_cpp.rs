use std::fs;
use std::path::Path;
use std::time::Duration;

use crate::backend::{extract_segments_from_json, transcript_from_segments};
use crate::diarization_projection::{
    canonicalize_zero_duration_projection_segments, validate_projection_segments,
};
use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, OutputFormat, TranscribeRequest, TranscriptionResult, TranscriptionSegment,
};
use crate::process::{command_exists, run_command_cancellable, run_command_with_timeout};

const DEFAULT_WHISPER_CPP_BIN: &str = "whisper-cli";

pub fn is_available() -> bool {
    command_exists(&binary())
}

pub(crate) fn binary_name() -> String {
    binary()
}

pub fn run(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<TranscriptionResult> {
    let binary = binary();
    let output_prefix = work_dir.join("whispercpp_output");
    let args = build_args(request, normalized_wav, &output_prefix);

    if let Some(tok) = token {
        run_command_cancellable(&binary, &args, None, tok, Some(timeout))?;
    } else {
        run_command_with_timeout(&binary, &args, None, Some(timeout))?;
    }

    let json_path = Path::new(&format!("{}.json", output_prefix.display())).to_path_buf();
    if !json_path.exists() {
        return Err(FwError::MissingArtifact(json_path));
    }

    let raw: serde_json::Value = serde_json::from_str(&fs::read_to_string(&json_path)?)?;
    let segments = adapt_projection_segments(request, &extract_segments_from_json(&raw))?;

    let transcript = raw
        .get("text")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| transcript_from_segments(&segments));

    let language = raw
        .pointer("/result/language")
        .or_else(|| raw.get("language"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| request.language.clone());

    // Collect all generated artifact paths.
    let mut artifact_paths = vec![json_path.display().to_string()];
    for fmt in &request.backend_params.output_formats {
        let ext = output_format_extension(*fmt);
        let candidate = Path::new(&format!("{}.{ext}", output_prefix.display())).to_path_buf();
        if candidate.exists() {
            artifact_paths.push(candidate.display().to_string());
        }
    }

    Ok(TranscriptionResult {
        backend: BackendKind::WhisperCpp,
        transcript,
        language,
        segments,
        acceleration: None,
        diarization: None,
        raw_output: raw,
        artifact_paths,
    })
}

/// Canonicalize the one degenerate timestamp shape that `whisper-cli` can
/// emit for an explicitly word-split request.
///
/// `-sow` is the adapter's evidence that these are decoder word-boundary
/// observations. A quantized `[t, t]` observation keeps its text, confidence,
/// order, and slot but relinquishes timing because it carries no interval from
/// which speaker ownership can be inferred. The full adapted timeline is then
/// validated with strict word-aligned geometry so this exception cannot hide a
/// reversed, overlapping, non-finite, or unpaired neighbor.
fn adapt_projection_segments(
    request: &TranscribeRequest,
    segments: &[TranscriptionSegment],
) -> FwResult<Vec<TranscriptionSegment>> {
    if !request.backend_params.split_on_word || request.backend_params.no_timestamps {
        return Ok(segments.to_vec());
    }

    let adapted = canonicalize_zero_duration_projection_segments(segments, true)?;
    validate_projection_segments(&adapted, true)?;
    Ok(adapted)
}

pub(crate) fn build_args(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    output_prefix: &Path,
) -> Vec<String> {
    let bp = &request.backend_params;

    let mut args = vec![
        "-f".to_owned(),
        normalized_wav.display().to_string(),
        "-of".to_owned(),
        output_prefix.display().to_string(),
        // Always request JSON output since run() parses it.
        "-oj".to_owned(),
    ];

    if let Some(model) = &request.model {
        args.push("-m".to_owned());
        args.push(model.clone());
    }

    if let Some(language) = &request.language {
        args.push("-l".to_owned());
        args.push(language.clone());
    }

    if request.translate {
        args.push("--translate".to_owned());
    }

    // Additional output formats beyond JSON.
    for fmt in &bp.output_formats {
        args.push(output_format_flag(*fmt).to_owned());
    }

    // Decoding parameters.
    if let Some(dec) = &bp.decoding {
        if let Some(best_of) = dec.best_of {
            args.push("-bo".to_owned());
            args.push(best_of.to_string());
        }
        if let Some(beam_size) = dec.beam_size {
            args.push("-bs".to_owned());
            args.push(beam_size.to_string());
        }
        if let Some(max_context) = dec.max_context {
            args.push("-mc".to_owned());
            args.push(max_context.to_string());
        }
        if let Some(max_len) = dec.max_segment_length {
            args.push("-ml".to_owned());
            args.push(max_len.to_string());
        }
        if let Some(temp) = dec.temperature {
            args.push("--temperature".to_owned());
            args.push(temp.to_string());
        }
        if let Some(temp_inc) = dec.temperature_increment {
            args.push("--temperature-inc".to_owned());
            args.push(temp_inc.to_string());
        }
        if let Some(et) = dec.entropy_threshold {
            args.push("-et".to_owned());
            args.push(et.to_string());
        }
        if let Some(lpt) = dec.logprob_threshold {
            args.push("-lpt".to_owned());
            args.push(lpt.to_string());
        }
        if let Some(ns) = dec.no_speech_threshold {
            args.push("-nth".to_owned());
            args.push(ns.to_string());
        }
    }

    // VAD parameters.
    if let Some(vad) = &bp.vad {
        if let Some(model_path) = &vad.model_path {
            args.push("--vad-model".to_owned());
            args.push(model_path.display().to_string());
        }
        if let Some(threshold) = vad.threshold {
            args.push("--vad-threshold".to_owned());
            args.push(threshold.to_string());
        }
        if let Some(min_speech) = vad.min_speech_duration_ms {
            args.push("--vad-min-speech-duration-ms".to_owned());
            args.push(min_speech.to_string());
        }
        if let Some(min_silence) = vad.min_silence_duration_ms {
            args.push("--vad-min-silence-duration-ms".to_owned());
            args.push(min_silence.to_string());
        }
        if let Some(max_speech) = vad.max_speech_duration_s {
            args.push("--vad-max-speech-duration-s".to_owned());
            args.push(max_speech.to_string());
        }
        if let Some(pad) = vad.speech_pad_ms {
            args.push("--vad-speech-pad-ms".to_owned());
            args.push(pad.to_string());
        }
        if let Some(overlap) = vad.samples_overlap {
            args.push("--vad-samples-overlap".to_owned());
            args.push(overlap.to_string());
        }
    }

    // Threading parameters.
    if let Some(threads) = bp.threads {
        args.push("-t".to_owned());
        args.push(threads.to_string());
    }
    if let Some(processors) = bp.processors {
        args.push("-p".to_owned());
        args.push(processors.to_string());
    }

    // Audio windowing parameters.
    if let Some(offset_ms) = bp.offset_ms {
        args.push("-ot".to_owned());
        args.push(offset_ms.to_string());
    }
    if let Some(duration_ms) = bp.duration_ms {
        args.push("-d".to_owned());
        args.push(duration_ms.to_string());
    }
    if let Some(audio_ctx) = bp.audio_ctx {
        args.push("-ac".to_owned());
        args.push(audio_ctx.to_string());
    }
    if let Some(wt) = bp.word_threshold {
        args.push("-wt".to_owned());
        args.push(wt.to_string());
    }

    // Prompt parameters.
    if let Some(prompt) = &bp.prompt {
        args.push("--prompt".to_owned());
        args.push(prompt.clone());
    }
    if bp.carry_initial_prompt {
        args.push("--carry-initial-prompt".to_owned());
    }

    // Boolean flags.
    if bp.no_timestamps {
        args.push("-nt".to_owned());
    }
    if bp.detect_language_only {
        args.push("--detect-language".to_owned());
    }
    if bp.split_on_word {
        args.push("-sow".to_owned());
    }
    if bp.no_gpu {
        args.push("-ng".to_owned());
    }
    if bp.no_fallback {
        args.push("-nf".to_owned());
    }
    if bp.suppress_nst {
        args.push("-sns".to_owned());
    }
    if bp.tiny_diarize {
        // whisper.cpp accepts -tdrz (or --tinydiarize); --tdrz is not a
        // supported spelling and prevents the CLI from producing its JSON
        // artifact on current releases.
        args.push("-tdrz".to_owned());
    }
    if let Some(regex) = &bp.suppress_regex {
        args.push("--suppress-regex".to_owned());
        args.push(regex.clone());
    }

    // Word-level timestamp parameters (bd-1rj.2).
    if let Some(wt) = &bp.word_timestamps {
        let ml_val = if let Some(max_len) = wt.max_len {
            Some(max_len.to_string())
        } else if wt.enabled {
            Some("1".to_owned())
        } else {
            None
        };
        if let Some(val) = ml_val {
            if let Some(ml_idx) = args.iter().position(|a| a == "-ml") {
                args[ml_idx + 1] = val;
            } else {
                args.push("-ml".to_owned());
                args.push(val);
            }
        }
        if let Some(token_threshold) = wt.token_threshold {
            if let Some(wt_idx) = args.iter().position(|a| a == "-wt") {
                args[wt_idx + 1] = token_threshold.to_string();
            } else {
                args.push("-wt".to_owned());
                args.push(token_threshold.to_string());
            }
        }
        if let Some(token_sum_threshold) = wt.token_sum_threshold {
            if let Some(wtps_idx) = args.iter().position(|a| a == "-wtps") {
                args[wtps_idx + 1] = token_sum_threshold.to_string();
            } else {
                args.push("-wtps".to_owned());
                args.push(token_sum_threshold.to_string());
            }
        }
    }

    args
}

/// Execute a transcription request with streaming segment delivery.
///
/// Segments are delivered to `on_segment` as they are parsed from the
/// whisper.cpp JSON output. Currently this runs the batch process first
/// and then replays segments; the callback enables real-time consumers
/// to process segments as they become available.
pub fn run_streaming(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    work_dir: &Path,
    timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
    on_segment: &dyn Fn(TranscriptionSegment),
) -> FwResult<TranscriptionResult> {
    let result = run(request, normalized_wav, work_dir, timeout, token)?;
    for segment in &result.segments {
        on_segment(segment.clone());
    }
    Ok(result)
}

fn binary() -> String {
    std::env::var("FRANKEN_WHISPER_WHISPER_CPP_BIN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_WHISPER_CPP_BIN.to_owned())
}

fn output_format_flag(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Txt => "-otxt",
        OutputFormat::Vtt => "-ovtt",
        OutputFormat::Srt => "-osrt",
        OutputFormat::Csv => "-ocsv",
        OutputFormat::Json => "-oj",
        OutputFormat::JsonFull => "-ojf",
        OutputFormat::Lrc => "-olrc",
    }
}

fn output_format_extension(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Txt => "txt",
        OutputFormat::Vtt => "vtt",
        OutputFormat::Srt => "srt",
        OutputFormat::Csv => "csv",
        OutputFormat::Json => "json",
        OutputFormat::JsonFull => "json",
        OutputFormat::Lrc => "lrc",
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::model::{
        BackendKind, BackendParams, DecodingParams, InputSource, OutputFormat, TranscribeRequest,
        VadParams, WordTimestampParams,
    };

    use super::{adapt_projection_segments, build_args};

    fn minimal_request() -> TranscribeRequest {
        TranscribeRequest {
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
        }
    }

    fn segment(
        start_sec: Option<f64>,
        end_sec: Option<f64>,
        text: &str,
        confidence: Option<f64>,
    ) -> crate::model::TranscriptionSegment {
        crate::model::TranscriptionSegment {
            start_sec,
            end_sec,
            text: text.to_owned(),
            speaker: None,
            confidence,
        }
    }

    #[test]
    fn split_on_word_adapter_preserves_zero_duration_observation_as_untimed() {
        let mut request = minimal_request();
        request.backend_params.split_on_word = true;
        request.backend_params.decoding = Some(DecodingParams {
            max_segment_length: Some(100),
            ..DecodingParams::default()
        });
        let input = vec![
            segment(Some(0.0), Some(1.0), "first", Some(0.91)),
            segment(Some(1.0), Some(1.0), "zero-width", Some(0.72)),
            segment(Some(1.5), Some(2.0), "third", Some(0.83)),
        ];

        let adapted = adapt_projection_segments(&request, &input)
            .expect("word-split zero-width observation must canonicalize");

        assert_eq!(adapted.len(), input.len());
        assert_eq!(adapted[0], input[0]);
        assert_eq!(adapted[1].text, "zero-width");
        assert_eq!(adapted[1].confidence, Some(0.72));
        assert_eq!(adapted[1].start_sec, None);
        assert_eq!(adapted[1].end_sec, None);
        assert_eq!(adapted[2], input[2]);
    }

    #[test]
    fn split_on_word_adapter_does_not_hide_malformed_neighbors() {
        let mut request = minimal_request();
        request.backend_params.split_on_word = true;

        let overlapping_previous = vec![
            segment(Some(0.0), Some(1.1), "first", None),
            segment(Some(1.0), Some(1.0), "zero-width", None),
            segment(Some(1.0), Some(2.0), "third", None),
        ];
        assert!(adapt_projection_segments(&request, &overlapping_previous).is_err());

        let reversed_next = vec![
            segment(Some(0.0), Some(1.0), "first", None),
            segment(Some(1.0), Some(1.0), "zero-width", None),
            segment(Some(0.9), Some(2.0), "third", None),
        ];
        assert!(adapt_projection_segments(&request, &reversed_next).is_err());

        let unpaired = vec![segment(Some(1.0), None, "missing end", None)];
        assert!(adapt_projection_segments(&request, &unpaired).is_err());
    }

    #[test]
    fn non_word_or_no_timestamp_requests_do_not_gain_canonicalization_authority() {
        let zero_width = vec![segment(Some(1.0), Some(1.0), "external", Some(0.4))];

        let request = minimal_request();
        let unchanged = adapt_projection_segments(&request, &zero_width)
            .expect("non-word adapter must leave the observation untouched");
        assert_eq!(unchanged, zero_width);
        assert!(
            crate::diarization_projection::validate_projection_segments(&unchanged, false)
                .is_err()
        );

        let mut no_timestamps = minimal_request();
        no_timestamps.backend_params.split_on_word = true;
        no_timestamps.backend_params.no_timestamps = true;
        let unchanged = adapt_projection_segments(&no_timestamps, &zero_width)
            .expect("timestamps-suppressed request has no timing authority");
        assert_eq!(unchanged, zero_width);
    }

    #[test]
    fn minimal_args_include_file_output_prefix_and_json() {
        let request = minimal_request();
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        assert!(args.contains(&"-f".to_owned()));
        assert!(args.contains(&"norm.wav".to_owned()));
        assert!(args.contains(&"-of".to_owned()));
        assert!(args.contains(&"/tmp/out".to_owned()));
        assert!(args.contains(&"-oj".to_owned()));
        // Should not contain translate, model, or language flags.
        assert!(!args.contains(&"--translate".to_owned()));
        assert!(!args.contains(&"-m".to_owned()));
        assert!(!args.contains(&"-l".to_owned()));
    }

    #[test]
    fn model_and_language_flags_present() {
        let mut request = minimal_request();
        request.model = Some("models/ggml-large-v3.bin".to_owned());
        request.language = Some("fr".to_owned());

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let model_idx = args.iter().position(|a| a == "-m").expect("-m flag");
        assert_eq!(args[model_idx + 1], "models/ggml-large-v3.bin");

        let lang_idx = args.iter().position(|a| a == "-l").expect("-l flag");
        assert_eq!(args[lang_idx + 1], "fr");
    }

    #[test]
    fn translate_flag_present_when_requested() {
        let mut request = minimal_request();
        request.translate = true;

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        assert!(args.contains(&"--translate".to_owned()));
    }

    #[test]
    fn output_format_flags_appended() {
        let mut request = minimal_request();
        request.backend_params.output_formats = vec![OutputFormat::Srt, OutputFormat::Vtt];

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        assert!(args.contains(&"-osrt".to_owned()));
        assert!(args.contains(&"-ovtt".to_owned()));
    }

    #[test]
    fn decoding_params_produce_correct_flags() {
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams {
            best_of: Some(5),
            beam_size: Some(3),
            max_context: Some(128),
            max_segment_length: Some(40),
            temperature: Some(0.2),
            temperature_increment: Some(0.1),
            entropy_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
        });

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let check = |flag: &str, expected: &str| {
            let idx = args
                .iter()
                .position(|a| a == flag)
                .unwrap_or_else(|| panic!("expected flag {flag}"));
            assert_eq!(
                args[idx + 1],
                expected,
                "flag {flag} should have value {expected}"
            );
        };

        check("-bo", "5");
        check("-bs", "3");
        check("-mc", "128");
        check("-ml", "40");
        check("--temperature", "0.2");
        check("--temperature-inc", "0.1");
        check("-et", "2.4");
        check("-lpt", "-1");
        check("-nth", "0.6");
    }

    #[test]
    fn vad_params_produce_correct_flags() {
        let mut request = minimal_request();
        request.backend_params.vad = Some(VadParams {
            model_path: Some(PathBuf::from("/models/silero.onnx")),
            threshold: Some(0.5),
            min_speech_duration_ms: Some(250),
            min_silence_duration_ms: Some(100),
            max_speech_duration_s: Some(30.0),
            speech_pad_ms: Some(50),
            samples_overlap: Some(0.1),
        });

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let check = |flag: &str, expected: &str| {
            let idx = args
                .iter()
                .position(|a| a == flag)
                .unwrap_or_else(|| panic!("expected flag {flag}"));
            assert_eq!(
                args[idx + 1],
                expected,
                "flag {flag} should have value {expected}"
            );
        };

        check("--vad-model", "/models/silero.onnx");
        check("--vad-threshold", "0.5");
        check("--vad-min-speech-duration-ms", "250");
        check("--vad-min-silence-duration-ms", "100");
        check("--vad-max-speech-duration-s", "30");
        check("--vad-speech-pad-ms", "50");
        check("--vad-samples-overlap", "0.1");
    }

    #[test]
    fn boolean_flags_no_timestamps_detect_language_split_on_word() {
        let mut request = minimal_request();
        request.backend_params.no_timestamps = true;
        request.backend_params.detect_language_only = true;
        request.backend_params.split_on_word = true;

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        assert!(args.contains(&"-nt".to_owned()));
        assert!(args.contains(&"--detect-language".to_owned()));
        assert!(args.contains(&"-sow".to_owned()));
    }

    #[test]
    fn default_params_omit_optional_flags() {
        let request = minimal_request();
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        // No decoding, VAD, or boolean flags should appear.
        for flag in [
            "-bo",
            "-bs",
            "-mc",
            "-ml",
            "--temperature",
            "--temperature-inc",
            "-et",
            "-lpt",
            "-nth",
            "--vad-model",
            "--vad-threshold",
            "-nt",
            "--detect-language",
            "-sow",
        ] {
            assert!(
                !args.contains(&flag.to_owned()),
                "flag {flag} should not appear with default params"
            );
        }
    }

    #[test]
    fn partial_decoding_params_only_emit_set_fields() {
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams {
            best_of: Some(3),
            beam_size: None,
            max_context: None,
            max_segment_length: None,
            temperature: None,
            temperature_increment: None,
            entropy_threshold: None,
            logprob_threshold: None,
            no_speech_threshold: None,
        });

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        assert!(args.contains(&"-bo".to_owned()));
        assert!(!args.contains(&"-bs".to_owned()));
        assert!(!args.contains(&"-mc".to_owned()));
        assert!(!args.contains(&"--temperature".to_owned()));
    }

    #[test]
    fn all_output_formats_generate_distinct_flags() {
        let mut request = minimal_request();
        request.backend_params.output_formats = vec![
            OutputFormat::Txt,
            OutputFormat::Vtt,
            OutputFormat::Srt,
            OutputFormat::Csv,
            OutputFormat::Json,
            OutputFormat::JsonFull,
            OutputFormat::Lrc,
        ];

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        for expected in ["-otxt", "-ovtt", "-osrt", "-ocsv", "-oj", "-ojf", "-olrc"] {
            assert!(
                args.contains(&expected.to_owned()),
                "missing format flag {expected}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // output_format_extension and output_format_flag tests
    // -----------------------------------------------------------------------

    use super::{output_format_extension, output_format_flag};

    #[test]
    fn output_format_flag_all_variants() {
        assert_eq!(output_format_flag(OutputFormat::Txt), "-otxt");
        assert_eq!(output_format_flag(OutputFormat::Vtt), "-ovtt");
        assert_eq!(output_format_flag(OutputFormat::Srt), "-osrt");
        assert_eq!(output_format_flag(OutputFormat::Csv), "-ocsv");
        assert_eq!(output_format_flag(OutputFormat::Json), "-oj");
        assert_eq!(output_format_flag(OutputFormat::JsonFull), "-ojf");
        assert_eq!(output_format_flag(OutputFormat::Lrc), "-olrc");
    }

    #[test]
    fn output_format_extension_all_variants() {
        assert_eq!(output_format_extension(OutputFormat::Txt), "txt");
        assert_eq!(output_format_extension(OutputFormat::Vtt), "vtt");
        assert_eq!(output_format_extension(OutputFormat::Srt), "srt");
        assert_eq!(output_format_extension(OutputFormat::Csv), "csv");
        assert_eq!(output_format_extension(OutputFormat::Json), "json");
        assert_eq!(output_format_extension(OutputFormat::JsonFull), "json");
        assert_eq!(output_format_extension(OutputFormat::Lrc), "lrc");
    }

    #[test]
    fn json_and_json_full_share_extension() {
        // Documenting current behavior: both Json and JsonFull map to "json".
        // This means artifact_paths collection could collide if both are requested.
        assert_eq!(
            output_format_extension(OutputFormat::Json),
            output_format_extension(OutputFormat::JsonFull),
        );
    }

    #[test]
    fn threading_params_produce_correct_flags() {
        let mut request = minimal_request();
        request.backend_params.threads = Some(8);
        request.backend_params.processors = Some(2);
        let args = build_args(
            &request,
            &PathBuf::from("n.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let check = |flag: &str, expected: &str| {
            let idx = args
                .iter()
                .position(|a| a == flag)
                .unwrap_or_else(|| panic!("expected flag {flag}"));
            assert_eq!(args[idx + 1], expected);
        };

        check("-t", "8");
        check("-p", "2");
    }

    #[test]
    fn audio_windowing_params_produce_correct_flags() {
        let mut request = minimal_request();
        request.backend_params.offset_ms = Some(5000);
        request.backend_params.duration_ms = Some(30000);
        request.backend_params.audio_ctx = Some(128);
        request.backend_params.word_threshold = Some(0.01);
        let args = build_args(
            &request,
            &PathBuf::from("n.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let check = |flag: &str, expected: &str| {
            let idx = args
                .iter()
                .position(|a| a == flag)
                .unwrap_or_else(|| panic!("expected flag {flag}"));
            assert_eq!(args[idx + 1], expected);
        };

        check("-ot", "5000");
        check("-d", "30000");
        check("-ac", "128");
        check("-wt", "0.01");
    }

    #[test]
    fn prompt_params_produce_correct_flags() {
        let mut request = minimal_request();
        request.backend_params.prompt = Some("medical terms".to_owned());
        request.backend_params.carry_initial_prompt = true;
        let args = build_args(
            &request,
            &PathBuf::from("n.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let idx = args.iter().position(|a| a == "--prompt").expect("--prompt");
        assert_eq!(args[idx + 1], "medical terms");
        assert!(args.contains(&"--carry-initial-prompt".to_owned()));
    }

    #[test]
    fn gpu_and_decoding_boolean_flags() {
        let mut request = minimal_request();
        request.backend_params.no_gpu = true;
        request.backend_params.no_fallback = true;
        request.backend_params.suppress_nst = true;
        let args = build_args(
            &request,
            &PathBuf::from("n.wav"),
            &PathBuf::from("/tmp/out"),
        );

        assert!(args.contains(&"-ng".to_owned()));
        assert!(args.contains(&"-nf".to_owned()));
        assert!(args.contains(&"-sns".to_owned()));
    }

    #[test]
    fn suppress_regex_flag() {
        let mut request = minimal_request();
        request.backend_params.suppress_regex = Some(r"\[.*\]".to_owned());
        let args = build_args(
            &request,
            &PathBuf::from("n.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let idx = args
            .iter()
            .position(|a| a == "--suppress-regex")
            .expect("--suppress-regex");
        assert_eq!(args[idx + 1], r"\[.*\]");
    }

    #[test]
    fn tiny_diarize_flag() {
        let mut request = minimal_request();
        request.backend_params.tiny_diarize = true;
        let args = build_args(
            &request,
            &PathBuf::from("n.wav"),
            &PathBuf::from("/tmp/out"),
        );
        assert!(args.contains(&"-tdrz".to_owned()));
        assert!(!args.contains(&"--tdrz".to_owned()));
    }

    #[test]
    fn tiny_diarize_omitted_by_default() {
        let request = minimal_request();
        let args = build_args(
            &request,
            &PathBuf::from("n.wav"),
            &PathBuf::from("/tmp/out"),
        );
        assert!(!args.contains(&"-tdrz".to_owned()));
    }

    #[test]
    fn default_params_omit_new_phase4_flags() {
        let request = minimal_request();
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        for flag in [
            "-t",
            "-p",
            "-ot",
            "-d",
            "-ac",
            "-wt",
            "--prompt",
            "--carry-initial-prompt",
            "-ng",
            "-nf",
            "-sns",
            "--suppress-regex",
        ] {
            assert!(
                !args.contains(&flag.to_owned()),
                "flag {flag} should not appear with default params"
            );
        }
    }

    #[test]
    fn binary_default_is_whisper_cli() {
        // When the env var is not set, binary() should return the default name.
        // (This test works because the env var is not set in CI.)
        let name = super::binary();
        if std::env::var("FRANKEN_WHISPER_WHISPER_CPP_BIN").is_err() {
            assert_eq!(name, "whisper-cli");
        }
    }

    #[test]
    fn binary_name_matches_binary() {
        assert_eq!(super::binary_name(), super::binary());
    }

    #[test]
    fn all_flags_are_distinct() {
        let flags: Vec<&str> = [
            OutputFormat::Txt,
            OutputFormat::Vtt,
            OutputFormat::Srt,
            OutputFormat::Csv,
            OutputFormat::Json,
            OutputFormat::JsonFull,
            OutputFormat::Lrc,
        ]
        .iter()
        .map(|f| output_format_flag(*f))
        .collect();

        let unique: std::collections::HashSet<&&str> = flags.iter().collect();
        assert_eq!(
            unique.len(),
            flags.len(),
            "all format flags should be distinct"
        );
    }

    #[test]
    fn all_params_combined() {
        let mut request = minimal_request();
        request.model = Some("models/large.bin".to_owned());
        request.language = Some("de".to_owned());
        request.translate = true;
        request.backend_params = BackendParams {
            output_formats: vec![OutputFormat::Srt, OutputFormat::Txt],
            decoding: Some(DecodingParams {
                best_of: Some(5),
                beam_size: Some(3),
                ..DecodingParams::default()
            }),
            vad: Some(VadParams {
                threshold: Some(0.4),
                ..VadParams::default()
            }),
            threads: Some(4),
            processors: Some(1),
            no_gpu: true,
            prompt: Some("test".to_owned()),
            carry_initial_prompt: true,
            no_fallback: true,
            suppress_nst: true,
            no_timestamps: true,
            detect_language_only: true,
            split_on_word: true,
            offset_ms: Some(1000),
            duration_ms: Some(60000),
            audio_ctx: Some(0),
            word_threshold: Some(0.5),
            suppress_regex: Some(r"\[.*\]".to_owned()),
            ..BackendParams::default()
        };

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        // Core flags.
        assert!(args.contains(&"-f".to_owned()));
        assert!(args.contains(&"-oj".to_owned()));
        assert!(args.contains(&"--translate".to_owned()));
        // Model and language.
        let m_idx = args.iter().position(|a| a == "-m").unwrap();
        assert_eq!(args[m_idx + 1], "models/large.bin");
        let l_idx = args.iter().position(|a| a == "-l").unwrap();
        assert_eq!(args[l_idx + 1], "de");
        // Formats.
        assert!(args.contains(&"-osrt".to_owned()));
        assert!(args.contains(&"-otxt".to_owned()));
        // Decoding.
        assert!(args.contains(&"-bo".to_owned()));
        assert!(args.contains(&"-bs".to_owned()));
        // VAD.
        assert!(args.contains(&"--vad-threshold".to_owned()));
        // Threading.
        assert!(args.contains(&"-t".to_owned()));
        assert!(args.contains(&"-p".to_owned()));
        // Boolean flags.
        assert!(args.contains(&"-ng".to_owned()));
        assert!(args.contains(&"-nf".to_owned()));
        assert!(args.contains(&"-sns".to_owned()));
        assert!(args.contains(&"-nt".to_owned()));
        assert!(args.contains(&"--detect-language".to_owned()));
        assert!(args.contains(&"-sow".to_owned()));
        assert!(args.contains(&"--carry-initial-prompt".to_owned()));
        // Prompt.
        let p_idx = args.iter().position(|a| a == "--prompt").unwrap();
        assert_eq!(args[p_idx + 1], "test");
        // Audio windowing.
        assert!(args.contains(&"-ot".to_owned()));
        assert!(args.contains(&"-d".to_owned()));
        assert!(args.contains(&"-ac".to_owned()));
        assert!(args.contains(&"-wt".to_owned()));
        // Suppress regex.
        assert!(args.contains(&"--suppress-regex".to_owned()));
    }

    #[test]
    fn partial_vad_only_emits_set_fields() {
        let mut request = minimal_request();
        request.backend_params.vad = Some(VadParams {
            threshold: Some(0.6),
            min_speech_duration_ms: Some(300),
            ..VadParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));

        assert!(args.contains(&"--vad-threshold".to_owned()));
        assert!(args.contains(&"--vad-min-speech-duration-ms".to_owned()));
        assert!(!args.contains(&"--vad-model".to_owned()));
        assert!(!args.contains(&"--vad-min-silence-duration-ms".to_owned()));
        assert!(!args.contains(&"--vad-max-speech-duration-s".to_owned()));
        assert!(!args.contains(&"--vad-speech-pad-ms".to_owned()));
        assert!(!args.contains(&"--vad-samples-overlap".to_owned()));
    }

    #[test]
    fn carry_initial_prompt_without_prompt_text() {
        let mut request = minimal_request();
        request.backend_params.carry_initial_prompt = true;
        // No prompt text set.
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        assert!(args.contains(&"--carry-initial-prompt".to_owned()));
        assert!(!args.contains(&"--prompt".to_owned()));
    }

    #[test]
    fn empty_decoding_params_emits_no_decoding_flags() {
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams::default());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        for flag in [
            "-bo",
            "-bs",
            "-mc",
            "-ml",
            "--temperature",
            "--temperature-inc",
            "-et",
            "-lpt",
            "-nth",
        ] {
            assert!(
                !args.contains(&flag.to_owned()),
                "empty DecodingParams should not emit {flag}"
            );
        }
    }

    #[test]
    fn empty_vad_params_emits_no_vad_flags() {
        let mut request = minimal_request();
        request.backend_params.vad = Some(VadParams::default());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        for flag in [
            "--vad-model",
            "--vad-threshold",
            "--vad-min-speech-duration-ms",
            "--vad-min-silence-duration-ms",
            "--vad-max-speech-duration-s",
            "--vad-speech-pad-ms",
            "--vad-samples-overlap",
        ] {
            assert!(
                !args.contains(&flag.to_owned()),
                "empty VadParams should not emit {flag}"
            );
        }
    }

    #[test]
    fn prompt_with_special_characters() {
        let mut request = minimal_request();
        request.backend_params.prompt = Some("Dr. Smith's \"lab\" results & <notes>".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let idx = args.iter().position(|a| a == "--prompt").expect("--prompt");
        assert_eq!(
            args[idx + 1],
            "Dr. Smith's \"lab\" results & <notes>",
            "special characters should be passed through verbatim"
        );
    }

    #[test]
    fn suppress_regex_with_complex_pattern() {
        let mut request = minimal_request();
        request.backend_params.suppress_regex = Some(r"^\[.*?\]$|^♪.*♪$|\(.*\)".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let idx = args
            .iter()
            .position(|a| a == "--suppress-regex")
            .expect("--suppress-regex");
        assert_eq!(args[idx + 1], r"^\[.*?\]$|^♪.*♪$|\(.*\)");
    }

    #[test]
    fn language_auto_passes_through() {
        let mut request = minimal_request();
        request.language = Some("auto".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let idx = args.iter().position(|a| a == "-l").expect("-l flag");
        assert_eq!(args[idx + 1], "auto");
    }

    #[test]
    fn decoding_params_negative_values() {
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams {
            logprob_threshold: Some(-2.5),
            entropy_threshold: Some(-0.1),
            no_speech_threshold: Some(0.0),
            ..DecodingParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let check = |flag: &str, expected: &str| {
            let idx = args.iter().position(|a| a == flag).unwrap();
            assert_eq!(args[idx + 1], expected);
        };
        check("-lpt", "-2.5");
        check("-et", "-0.1");
        check("-nth", "0");
    }

    #[test]
    fn vad_model_path_with_spaces() {
        let mut request = minimal_request();
        request.backend_params.vad = Some(VadParams {
            model_path: Some(PathBuf::from("/path with spaces/silero_vad.onnx")),
            ..VadParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let idx = args
            .iter()
            .position(|a| a == "--vad-model")
            .expect("--vad-model");
        assert_eq!(args[idx + 1], "/path with spaces/silero_vad.onnx");
    }

    #[test]
    fn arg_order_core_flags_come_first() {
        let request = minimal_request();
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        // First three args should be -f, <file>, -of.
        assert_eq!(args[0], "-f");
        assert_eq!(args[1], "n.wav");
        assert_eq!(args[2], "-of");
    }

    #[test]
    fn single_output_format_txt() {
        let mut request = minimal_request();
        request.backend_params.output_formats = vec![OutputFormat::Txt];
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        assert!(args.contains(&"-otxt".to_owned()));
        // Should not have other format flags.
        assert!(!args.contains(&"-osrt".to_owned()));
        assert!(!args.contains(&"-ovtt".to_owned()));
    }

    #[test]
    fn no_output_formats_means_only_json() {
        let request = minimal_request();
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        // Should have -oj (always included) but no other format flags.
        assert!(args.contains(&"-oj".to_owned()));
        for flag in ["-otxt", "-ovtt", "-osrt", "-ocsv", "-ojf", "-olrc"] {
            assert!(
                !args.contains(&flag.to_owned()),
                "should not contain {flag} with no output_formats"
            );
        }
    }

    #[test]
    fn translate_and_detect_language_both_present() {
        let mut request = minimal_request();
        request.translate = true;
        request.backend_params.detect_language_only = true;
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        assert!(args.contains(&"--translate".to_owned()));
        assert!(args.contains(&"--detect-language".to_owned()));
    }

    #[test]
    fn zero_value_threading_params() {
        let mut request = minimal_request();
        request.backend_params.threads = Some(0);
        request.backend_params.processors = Some(0);
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let t_idx = args.iter().position(|a| a == "-t").expect("-t");
        assert_eq!(args[t_idx + 1], "0");
        let p_idx = args.iter().position(|a| a == "-p").expect("-p");
        assert_eq!(args[p_idx + 1], "0");
    }

    #[test]
    fn model_path_with_unicode() {
        let mut request = minimal_request();
        request.model = Some("/modèles/ggml-café.bin".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let idx = args.iter().position(|a| a == "-m").expect("-m");
        assert_eq!(args[idx + 1], "/modèles/ggml-café.bin");
    }

    #[test]
    fn empty_prompt_string_still_emits_flag() {
        let mut request = minimal_request();
        request.backend_params.prompt = Some(String::new());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let idx = args.iter().position(|a| a == "--prompt").expect("--prompt");
        assert_eq!(args[idx + 1], "");
    }

    #[test]
    fn duplicate_output_formats_emit_duplicate_flags() {
        let mut request = minimal_request();
        request.backend_params.output_formats =
            vec![OutputFormat::Srt, OutputFormat::Srt, OutputFormat::Txt];
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let srt_count = args.iter().filter(|a| *a == "-osrt").count();
        assert_eq!(
            srt_count, 2,
            "duplicate formats should emit duplicate flags"
        );
    }

    #[test]
    fn extreme_decoding_values() {
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams {
            best_of: Some(u32::MAX),
            beam_size: Some(0),
            max_context: Some(0),
            max_segment_length: Some(u32::MAX),
            temperature: Some(100.0),
            temperature_increment: Some(0.0),
            entropy_threshold: Some(f32::MIN_POSITIVE),
            logprob_threshold: Some(f32::NEG_INFINITY),
            no_speech_threshold: Some(1.0),
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        // Should produce valid args without panicking.
        let bo_idx = args.iter().position(|a| a == "-bo").expect("-bo");
        assert_eq!(args[bo_idx + 1], u32::MAX.to_string());
        let bs_idx = args.iter().position(|a| a == "-bs").expect("-bs");
        assert_eq!(args[bs_idx + 1], "0");
    }

    #[test]
    fn all_boolean_flags_at_once() {
        let mut request = minimal_request();
        request.translate = true;
        request.backend_params.no_timestamps = true;
        request.backend_params.detect_language_only = true;
        request.backend_params.split_on_word = true;
        request.backend_params.no_gpu = true;
        request.backend_params.no_fallback = true;
        request.backend_params.suppress_nst = true;
        request.backend_params.carry_initial_prompt = true;
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));

        let expected = [
            "--translate",
            "-nt",
            "--detect-language",
            "-sow",
            "-ng",
            "-nf",
            "-sns",
            "--carry-initial-prompt",
        ];
        for flag in expected {
            assert!(
                args.contains(&flag.to_owned()),
                "missing boolean flag {flag}"
            );
        }
    }

    #[test]
    fn suppress_regex_forwarded_to_args() {
        let mut request = minimal_request();
        request.backend_params.suppress_regex = Some(r"\[.*?\]".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let idx = args
            .iter()
            .position(|a| a == "--suppress-regex")
            .expect("--suppress-regex");
        assert_eq!(args[idx + 1], r"\[.*?\]");
    }

    #[test]
    fn audio_windowing_all_params_emitted() {
        let mut request = minimal_request();
        request.backend_params.offset_ms = Some(5000);
        request.backend_params.duration_ms = Some(30000);
        request.backend_params.audio_ctx = Some(128);
        request.backend_params.word_threshold = Some(0.01);
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/tmp/o"));
        let ot_idx = args.iter().position(|a| a == "-ot").expect("-ot");
        assert_eq!(args[ot_idx + 1], "5000");
        let d_idx = args.iter().position(|a| a == "-d").expect("-d");
        assert_eq!(args[d_idx + 1], "30000");
        let ac_idx = args.iter().position(|a| a == "-ac").expect("-ac");
        assert_eq!(args[ac_idx + 1], "128");
        let wt_idx = args.iter().position(|a| a == "-wt").expect("-wt");
        assert_eq!(args[wt_idx + 1], "0.01");
    }

    #[test]
    fn args_always_start_with_file_output_json() {
        let request = minimal_request();
        let args = build_args(
            &request,
            &PathBuf::from("in.wav"),
            &PathBuf::from("/tmp/out"),
        );
        assert_eq!(args[0], "-f");
        assert_eq!(args[1], "in.wav");
        assert_eq!(args[2], "-of");
        assert_eq!(args[3], "/tmp/out");
        assert_eq!(args[4], "-oj");
    }

    #[test]
    fn minimal_request_produces_exactly_five_args() {
        let request = minimal_request();
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        assert_eq!(args.len(), 5, "minimal request: -f <file> -of <prefix> -oj");
    }

    #[test]
    fn max_context_i32_min_emits_valid_string() {
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams {
            max_context: Some(i32::MIN),
            ..DecodingParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));
        let idx = args.iter().position(|a| a == "-mc").expect("-mc");
        assert_eq!(args[idx + 1], i32::MIN.to_string());
        // Verify the string actually contains a negative sign
        assert!(args[idx + 1].starts_with('-'));
    }

    #[test]
    fn flag_value_pairs_are_always_consecutive() {
        let mut request = minimal_request();
        request.model = Some("model.bin".to_owned());
        request.language = Some("ja".to_owned());
        request.backend_params.threads = Some(4);
        request.backend_params.decoding = Some(DecodingParams {
            best_of: Some(3),
            ..DecodingParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));

        // Each value-taking flag should be immediately followed by its value (not another flag)
        let value_flags = ["-f", "-of", "-m", "-l", "-bo", "-t"];
        for flag in value_flags {
            if let Some(idx) = args.iter().position(|a| a == flag) {
                assert!(
                    idx + 1 < args.len(),
                    "flag {flag} at end of args — missing value"
                );
                assert!(
                    !args[idx + 1].starts_with('-') || args[idx + 1].parse::<f64>().is_ok(),
                    "flag {flag} followed by another flag {}, not a value",
                    args[idx + 1]
                );
            }
        }
    }

    #[test]
    fn no_args_contain_empty_strings_for_populated_request() {
        let mut request = minimal_request();
        request.model = Some("m.bin".to_owned());
        request.language = Some("en".to_owned());
        request.backend_params.threads = Some(4);
        request.backend_params.vad = Some(VadParams {
            threshold: Some(0.5),
            ..VadParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));
        for (i, arg) in args.iter().enumerate() {
            assert!(!arg.is_empty(), "arg at index {i} is empty string");
        }
    }

    #[test]
    fn json_flag_always_present_even_with_explicit_json_format() {
        // When user explicitly requests Json in output_formats, -oj appears (at least) twice:
        // once from the always-added base, once from the format list.
        let mut request = minimal_request();
        request.backend_params.output_formats = vec![OutputFormat::Json];
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));
        let oj_count = args.iter().filter(|a| *a == "-oj").count();
        assert_eq!(
            oj_count, 2,
            "explicit Json format + always-on -oj = two occurrences"
        );
    }

    #[test]
    fn build_args_all_boolean_flags_when_all_true() {
        let mut request = minimal_request();
        request.translate = true;
        request.backend_params.no_timestamps = true;
        request.backend_params.detect_language_only = true;
        request.backend_params.split_on_word = true;
        request.backend_params.no_gpu = true;
        request.backend_params.no_fallback = true;
        request.backend_params.suppress_nst = true;
        request.backend_params.carry_initial_prompt = true;
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        let expected = [
            "--translate",
            "-nt",
            "--detect-language",
            "-sow",
            "-ng",
            "-nf",
            "-sns",
            "--carry-initial-prompt",
        ];
        for flag in expected {
            assert!(args.contains(&flag.to_owned()), "missing flag: {flag}");
        }
    }

    #[test]
    fn build_args_no_boolean_flags_when_all_false() {
        // Default request has all bools false.
        let request = minimal_request();
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        let should_be_absent = [
            "--translate",
            "-nt",
            "--detect-language",
            "-sow",
            "-ng",
            "-nf",
            "-sns",
            "--carry-initial-prompt",
        ];
        for flag in should_be_absent {
            assert!(
                !args.contains(&flag.to_owned()),
                "flag {flag} should not appear with defaults"
            );
        }
    }

    #[test]
    fn build_args_all_vad_params_present() {
        let mut request = minimal_request();
        request.backend_params.vad = Some(VadParams {
            model_path: Some(std::path::PathBuf::from("/models/vad.onnx")),
            threshold: Some(0.5),
            min_speech_duration_ms: Some(250),
            min_silence_duration_ms: Some(100),
            max_speech_duration_s: Some(30.0),
            speech_pad_ms: Some(50),
            samples_overlap: Some(0.1),
        });
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        let expected_flags = [
            "--vad-model",
            "--vad-threshold",
            "--vad-min-speech-duration-ms",
            "--vad-min-silence-duration-ms",
            "--vad-max-speech-duration-s",
            "--vad-speech-pad-ms",
            "--vad-samples-overlap",
        ];
        for flag in expected_flags {
            assert!(args.contains(&flag.to_owned()), "missing VAD flag: {flag}");
        }
    }

    #[test]
    fn build_args_translate_flag_present_when_true() {
        let mut request = minimal_request();
        request.translate = true;
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        assert!(
            args.contains(&"--translate".to_owned()),
            "--translate should appear when translate = true"
        );
    }

    #[test]
    fn build_args_no_gpu_flag_present_when_true() {
        let mut request = minimal_request();
        request.backend_params.no_gpu = true;
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        assert!(
            args.contains(&"-ng".to_owned()),
            "-ng should appear when no_gpu = true"
        );

        // Also verify prompt with --prompt flag.
        let mut request2 = minimal_request();
        request2.backend_params.prompt = Some("transcribe medical terms".to_owned());
        let args2 = build_args(&request2, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        let idx = args2
            .iter()
            .position(|a| a == "--prompt")
            .expect("--prompt flag");
        assert_eq!(args2[idx + 1], "transcribe medical terms");
    }

    #[test]
    fn build_args_audio_windowing_zero_values() {
        // Zero values for offset/duration/audio_ctx should still produce flags.
        let mut request = minimal_request();
        request.backend_params.offset_ms = Some(0);
        request.backend_params.duration_ms = Some(0);
        request.backend_params.audio_ctx = Some(0);
        request.backend_params.word_threshold = Some(0.0);
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        let ot_idx = args.iter().position(|a| a == "-ot").expect("-ot");
        assert_eq!(args[ot_idx + 1], "0");
        let d_idx = args.iter().position(|a| a == "-d").expect("-d");
        assert_eq!(args[d_idx + 1], "0");
        let ac_idx = args.iter().position(|a| a == "-ac").expect("-ac");
        assert_eq!(args[ac_idx + 1], "0");
        let wt_idx = args.iter().position(|a| a == "-wt").expect("-wt");
        assert_eq!(args[wt_idx + 1], "0");
    }

    #[test]
    fn build_args_empty_prompt_still_produces_flag() {
        // An empty string prompt should still produce --prompt "".
        let mut request = minimal_request();
        request.backend_params.prompt = Some(String::new());
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        let idx = args.iter().position(|a| a == "--prompt").expect("--prompt");
        assert_eq!(args[idx + 1], "", "empty prompt should produce empty arg");
    }

    #[test]
    fn build_args_carry_initial_prompt_without_prompt() {
        // carry_initial_prompt flag without a prompt value.
        let mut request = minimal_request();
        request.backend_params.carry_initial_prompt = true;
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        assert!(args.contains(&"--carry-initial-prompt".to_owned()));
        assert!(
            !args.contains(&"--prompt".to_owned()),
            "no --prompt flag without prompt value"
        );
    }

    #[test]
    fn build_args_decoding_only_max_segment_length_set() {
        // When only max_segment_length is set in DecodingParams, only -ml appears.
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams {
            best_of: None,
            beam_size: None,
            max_context: None,
            max_segment_length: Some(50),
            temperature: None,
            temperature_increment: None,
            entropy_threshold: None,
            logprob_threshold: None,
            no_speech_threshold: None,
        });
        let args = build_args(&request, &PathBuf::from("in.wav"), &PathBuf::from("/out"));
        let idx = args.iter().position(|a| a == "-ml").expect("-ml");
        assert_eq!(args[idx + 1], "50");
        assert!(!args.contains(&"-bo".to_owned()), "no -bo without best_of");
        assert!(
            !args.contains(&"-bs".to_owned()),
            "no -bs without beam_size"
        );
    }

    #[test]
    fn output_format_json_and_json_full_share_extension() {
        // Json and JsonFull both use "json" extension — documents potential collision.
        use super::output_format_extension;
        assert_eq!(
            output_format_extension(OutputFormat::Json),
            output_format_extension(OutputFormat::JsonFull),
            "Json and JsonFull share the same file extension"
        );
        // But they use different flags.
        use super::output_format_flag;
        assert_ne!(
            output_format_flag(OutputFormat::Json),
            output_format_flag(OutputFormat::JsonFull),
            "Json and JsonFull should have different flags"
        );
    }

    // -----------------------------------------------------------------------
    // bd-1rj.2: Word-level timestamp tests
    // -----------------------------------------------------------------------

    #[test]
    fn word_timestamps_enabled_emits_ml_1() {
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: true,
            ..WordTimestampParams::default()
        });
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let idx = args
            .iter()
            .position(|a| a == "-ml")
            .expect("-ml should be present for word timestamps");
        assert_eq!(args[idx + 1], "1", "word timestamps enable -ml 1");
    }

    #[test]
    fn word_timestamps_disabled_does_not_emit_ml() {
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: false,
            ..WordTimestampParams::default()
        });
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        assert!(
            !args.contains(&"-ml".to_owned()),
            "-ml should not appear when word timestamps disabled and no max_len"
        );
    }

    #[test]
    fn word_timestamps_max_len_overrides_enabled() {
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: true,
            max_len: Some(10),
            ..WordTimestampParams::default()
        });
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let idx = args
            .iter()
            .position(|a| a == "-ml")
            .expect("-ml should be present");
        assert_eq!(
            args[idx + 1],
            "10",
            "max_len should override the enabled=true default of 1"
        );
        // Only one -ml flag should appear
        let ml_count = args.iter().filter(|a| *a == "-ml").count();
        assert_eq!(ml_count, 1, "only one -ml flag should appear");
    }

    #[test]
    fn word_timestamps_max_len_without_enabled() {
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: false,
            max_len: Some(25),
            ..WordTimestampParams::default()
        });
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let idx = args
            .iter()
            .position(|a| a == "-ml")
            .expect("-ml should be present from max_len");
        assert_eq!(args[idx + 1], "25");
    }

    #[test]
    fn word_timestamps_token_thresholds() {
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: true,
            token_threshold: Some(0.01),
            token_sum_threshold: Some(0.15),
            ..WordTimestampParams::default()
        });
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let wt_idx = args
            .iter()
            .position(|a| a == "-wt")
            .expect("-wt should be present");
        assert_eq!(args[wt_idx + 1], "0.01");

        let wtps_idx = args
            .iter()
            .position(|a| a == "-wtps")
            .expect("-wtps should be present");
        assert_eq!(args[wtps_idx + 1], "0.15");
    }

    #[test]
    fn word_timestamps_empty_params_emit_nothing() {
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams::default());
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        assert!(!args.contains(&"-ml".to_owned()));
        // -wt may come from word_threshold in BackendParams, but not from empty WordTimestampParams
        assert!(!args.contains(&"-wtps".to_owned()));
    }

    #[test]
    fn word_timestamps_all_fields_set() {
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: true,
            max_len: Some(5),
            token_threshold: Some(0.02),
            token_sum_threshold: Some(0.20),
        });
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        let ml_idx = args.iter().position(|a| a == "-ml").expect("-ml");
        assert_eq!(args[ml_idx + 1], "5");
        let wt_idx = args.iter().position(|a| a == "-wt").expect("-wt");
        assert_eq!(args[wt_idx + 1], "0.02");
        let wtps_idx = args.iter().position(|a| a == "-wtps").expect("-wtps");
        assert_eq!(args[wtps_idx + 1], "0.2");
    }

    #[test]
    fn word_timestamps_does_not_conflict_with_decoding_ml() {
        // When both decoding.max_segment_length and word_timestamps.max_len
        // are set, the word_timestamps.max_len should take final precedence
        // (it runs after decoding params in build_args).
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams {
            max_segment_length: Some(40),
            ..DecodingParams::default()
        });
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: false,
            max_len: Some(15),
            ..WordTimestampParams::default()
        });
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );

        // The word_timestamps.max_len=15 should overwrite the decoding -ml 40
        let ml_positions: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "-ml")
            .map(|(i, _)| i)
            .collect();
        // There should be exactly one -ml flag after override
        assert_eq!(ml_positions.len(), 1);
        assert_eq!(args[ml_positions[0] + 1], "15");
    }

    #[test]
    fn word_timestamps_none_does_not_emit_any_wt_flags() {
        let request = minimal_request();
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("/tmp/out"),
        );
        assert!(!args.contains(&"-wtps".to_owned()));
    }

    // -----------------------------------------------------------------------
    // bd-1rj.2: Streaming engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn decoding_ml_and_word_timestamps_enabled_without_max_len_uses_enabled_default() {
        // When decoding.max_segment_length is set AND word_timestamps.enabled=true
        // but max_len is None, the word_timestamps default of "1" overrides the decoding param.
        let mut request = minimal_request();
        request.backend_params.decoding = Some(DecodingParams {
            max_segment_length: Some(40),
            ..DecodingParams::default()
        });
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: true,
            max_len: None,
            ..WordTimestampParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));
        let ml_positions: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "-ml")
            .map(|(i, _)| i)
            .collect();
        // One -ml flag: word_timestamps' 1 overrides decoding's 40.
        assert_eq!(ml_positions.len(), 1, "expected one -ml flag");
        assert_eq!(args[ml_positions[0] + 1], "1");
    }

    #[test]
    fn word_threshold_and_token_threshold_overrides_wt_flag() {
        // When both bp.word_threshold and word_timestamps.token_threshold are set,
        // one -wt flag is emitted: the one from WordTimestampParams overrides.
        let mut request = minimal_request();
        request.backend_params.word_threshold = Some(0.01);
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: true,
            token_threshold: Some(0.5),
            ..WordTimestampParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));
        let wt_positions: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "-wt")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(wt_positions.len(), 1, "expected one -wt flag");
        assert_eq!(args[wt_positions[0] + 1], "0.5");
    }

    #[test]
    fn word_timestamps_enabled_only_no_wt_no_wtps() {
        // When only enabled=true with no other WordTimestampParams fields,
        // exactly -ml 1 appears but no -wt or -wtps flags.
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: true,
            max_len: None,
            token_threshold: None,
            token_sum_threshold: None,
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));
        let ml_idx = args.iter().position(|a| a == "-ml").expect("-ml present");
        assert_eq!(args[ml_idx + 1], "1");
        let ml_count = args.iter().filter(|a| *a == "-ml").count();
        assert_eq!(ml_count, 1, "exactly one -ml flag");
        assert!(
            !args.contains(&"-wt".to_owned()),
            "no -wt without token_threshold"
        );
        assert!(
            !args.contains(&"-wtps".to_owned()),
            "no -wtps without token_sum_threshold"
        );
    }

    #[test]
    fn word_timestamps_max_len_zero_emits_ml_0() {
        // Edge case: max_len=0 should produce -ml 0.
        let mut request = minimal_request();
        request.backend_params.word_timestamps = Some(WordTimestampParams {
            enabled: false,
            max_len: Some(0),
            ..WordTimestampParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));
        let ml_idx = args.iter().position(|a| a == "-ml").expect("-ml present");
        assert_eq!(args[ml_idx + 1], "0");
    }

    #[test]
    fn vad_zero_boundary_values_still_produce_flags() {
        // All VAD params set to zero/minimum values — flags should still appear.
        let mut request = minimal_request();
        request.backend_params.vad = Some(VadParams {
            model_path: None,
            threshold: Some(0.0),
            min_speech_duration_ms: Some(0),
            min_silence_duration_ms: Some(0),
            max_speech_duration_s: Some(0.0),
            speech_pad_ms: Some(0),
            samples_overlap: Some(0.0),
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("/o"));
        let check = |flag: &str, expected: &str| {
            let idx = args
                .iter()
                .position(|a| a == flag)
                .unwrap_or_else(|| panic!("expected flag {flag}"));
            assert_eq!(args[idx + 1], expected, "flag {flag}");
        };
        check("--vad-threshold", "0");
        check("--vad-min-speech-duration-ms", "0");
        check("--vad-min-silence-duration-ms", "0");
        check("--vad-max-speech-duration-s", "0");
        check("--vad-speech-pad-ms", "0");
        check("--vad-samples-overlap", "0");
        // model_path is None, so --vad-model should be absent.
        assert!(!args.contains(&"--vad-model".to_owned()));
    }

    #[test]
    fn run_streaming_function_signature_compiles() {
        // Verify that `run_streaming` accepts the expected types.
        // We cannot call it without the binary, but we can verify the
        // function exists and accepts the correct argument types.
        type RunStreamingFn = fn(
            &TranscribeRequest,
            &std::path::Path,
            &std::path::Path,
            std::time::Duration,
            Option<&crate::orchestrator::CancellationToken>,
            &dyn Fn(crate::model::TranscriptionSegment),
        )
            -> crate::error::FwResult<crate::model::TranscriptionResult>;
        let _fn_ptr: RunStreamingFn = super::run_streaming;
        // If this test compiles, the function signature is correct.
    }
}
