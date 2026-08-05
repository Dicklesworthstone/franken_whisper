use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::backend::{extract_segments_from_json, transcript_from_segments};
use crate::error::{FwError, FwResult};
use crate::model::{BackendKind, SpeakerCountRequest, TranscribeRequest, TranscriptionResult};
use crate::process::{command_exists, run_command_cancellable, run_command_with_timeout};

const DEFAULT_BIN: &str = "insanely-fast-whisper";

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
    validate_external_speaker_count(request)?;
    let output_path = output_path_for(request, work_dir);
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let args = build_args(request, normalized_wav, &output_path);

    if let Some(tok) = token {
        run_command_cancellable(&binary, &args, None, tok, Some(timeout))?;
    } else {
        run_command_with_timeout(&binary, &args, None, Some(timeout))?;
    }

    if !output_path.exists() {
        return Err(FwError::MissingArtifact(output_path));
    }

    let raw: serde_json::Value = serde_json::from_str(&fs::read_to_string(&output_path)?)?;
    let segments = extract_segments_from_json(&raw);
    let transcript = raw
        .get("text")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| transcript_from_segments(&segments));

    let language = raw
        .get("language")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| request.language.clone());

    Ok(TranscriptionResult {
        backend: BackendKind::InsanelyFast,
        transcript,
        language,
        segments,
        acceleration: None,
        diarization: None,
        raw_output: raw,
        artifact_paths: vec![output_path.display().to_string()],
    })
}

fn build_args(
    request: &TranscribeRequest,
    normalized_wav: &Path,
    output_path: &Path,
) -> Vec<String> {
    let bp = &request.backend_params;

    let mut args = vec![
        "--file-name".to_owned(),
        normalized_wav.display().to_string(),
        "--transcript-path".to_owned(),
        output_path.display().to_string(),
        "--task".to_owned(),
        if request.translate {
            "translate".to_owned()
        } else {
            "transcribe".to_owned()
        },
    ];

    if let Some(model) = &request.model {
        args.push("--model-name".to_owned());
        args.push(model.clone());
    }

    if let Some(language) = &request.language {
        args.push("--language".to_owned());
        args.push(language.clone());
    }

    if request.diarize
        && let Some(token) = hf_token_for_request(request)
    {
        args.push("--hf-token".to_owned());
        args.push(token);
    }

    // Phase 3: batch size.
    if let Some(batch) = bp.batch_size {
        args.push("--batch-size".to_owned());
        args.push(batch.to_string());
    }

    // Phase 3: timestamp granularity.
    if let Some(level) = bp.timestamp_level {
        args.push("--timestamp".to_owned());
        args.push(
            match level {
                crate::model::TimestampLevel::Chunk => "chunk",
                crate::model::TimestampLevel::Word => "word",
            }
            .to_owned(),
        );
    }

    if request.diarize
        && let Some(count_request) = bp
            .acoustic_diarization
            .as_ref()
            .map(|request| &request.speaker_count)
    {
        match count_request {
            SpeakerCountRequest::HardConstraint { count } => {
                args.push("--num-speakers".to_owned());
                args.push(count.to_string());
            }
            SpeakerCountRequest::Infer
            | SpeakerCountRequest::Prior { .. }
            | SpeakerCountRequest::Range { .. } => {}
        }
    }

    // Phase 3 parity: explicit pyannote diarization model selection.
    if request.diarize
        && let Some(di_config) = &bp.diarization_config
        && let Some(model) = &di_config.whisper_model
    {
        args.push("--diarization_model".to_owned());
        args.push(model.clone());
    }

    // Phase 3: GPU device.
    if let Some(device) = &bp.gpu_device {
        args.push("--device-id".to_owned());
        args.push(device.clone());
    }

    // Phase 3: Flash Attention 2.
    if bp.flash_attention == Some(true) {
        args.push("--flash".to_owned());
        args.push("True".to_owned());
    }

    // bd-1rj.3: Extended tuning parameters.
    if let Some(tuning) = &bp.insanely_fast_tuning {
        if let Some(device_map) = tuning.device_map {
            args.push("--device-map".to_owned());
            args.push(
                match device_map {
                    crate::model::DeviceMapStrategy::Auto => "auto",
                    crate::model::DeviceMapStrategy::Sequential => "sequential",
                }
                .to_owned(),
            );
        }
        if let Some(dtype) = &tuning.torch_dtype {
            args.push("--torch-dtype".to_owned());
            args.push(dtype.clone());
        }
        if tuning.disable_better_transformer {
            args.push("--disable-better-transformer".to_owned());
        }
    }

    args
}

fn validate_external_speaker_count(request: &TranscribeRequest) -> FwResult<()> {
    if request.diarize
        && request
            .backend_params
            .acoustic_diarization
            .as_ref()
            .is_some_and(|diarization| {
                matches!(
                    &diarization.speaker_count,
                    SpeakerCountRequest::Prior { .. } | SpeakerCountRequest::Range { .. }
                ) && matches!(
                    diarization.engine,
                    crate::model::DiarizationEngine::External
                )
            })
    {
        return Err(FwError::InvalidRequest(
            "insanely-fast-whisper cannot faithfully execute a soft speaker-count prior or range"
                .to_owned(),
        ));
    }
    Ok(())
}

fn binary() -> String {
    std::env::var("FRANKEN_WHISPER_INSANELY_FAST_BIN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BIN.to_owned())
}

pub(crate) fn output_path_for(request: &TranscribeRequest, work_dir: &Path) -> PathBuf {
    request
        .backend_params
        .insanely_fast_transcript_path
        .clone()
        .unwrap_or_else(|| work_dir.join("insanely_fast_output.json"))
}

pub(crate) fn hf_token_present() -> bool {
    hf_token().is_some()
}

pub(crate) fn hf_token_present_for_request(request: &TranscribeRequest) -> bool {
    hf_token_for_request(request).is_some()
}

fn hf_token_for_request(request: &TranscribeRequest) -> Option<String> {
    request
        .backend_params
        .insanely_fast_hf_token
        .clone()
        .filter(|value| !value.trim().is_empty())
        .or_else(hf_token)
}

fn hf_token() -> Option<String> {
    std::env::var("FRANKEN_WHISPER_HF_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var("HF_TOKEN")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::model::{
        BackendKind, BackendParams, DeviceMapStrategy, DiarizationConfig, DiarizationRequest,
        InputSource, InsanelyFastTuningParams, SpeakerCountRequest, TranscribeRequest,
    };

    use super::{build_args, output_path_for};

    fn request_with_diarization_model() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::InsanelyFast,
            model: Some("openai/whisper-large-v3".to_owned()),
            language: Some("en".to_owned()),
            translate: false,
            diarize: true,
            persist: false,
            db_path: PathBuf::from("db.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams {
                diarization_config: Some(DiarizationConfig {
                    whisper_model: Some("pyannote/speaker-diarization-3.1".to_owned()),
                    ..DiarizationConfig::default()
                }),
                ..BackendParams::default()
            },
        }
    }

    fn minimal_request() -> TranscribeRequest {
        TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("input.wav"),
            },
            backend: BackendKind::InsanelyFast,
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

    fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    }

    fn has_flag(args: &[String], flag: &str) -> bool {
        args.iter().any(|a| a == flag)
    }

    fn set_speaker_count(request: &mut TranscribeRequest, speaker_count: SpeakerCountRequest) {
        request.backend_params.acoustic_diarization = Some(DiarizationRequest {
            speaker_count,
            ..DiarizationRequest::default()
        });
    }

    #[test]
    fn minimal_args_include_file_transcript_path_and_transcribe_task() {
        let request = minimal_request();
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("out.json"),
        );
        assert_eq!(arg_value(&args, "--file-name"), Some("norm.wav"));
        assert_eq!(arg_value(&args, "--transcript-path"), Some("out.json"));
        assert_eq!(arg_value(&args, "--task"), Some("transcribe"));
        assert!(!has_flag(&args, "--model-name"));
        assert!(!has_flag(&args, "--language"));
    }

    #[test]
    fn translate_flag_sets_task_translate() {
        let mut request = minimal_request();
        request.translate = true;
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("out.json"),
        );
        assert_eq!(arg_value(&args, "--task"), Some("translate"));
    }

    #[test]
    fn model_and_language_flags_present() {
        let mut request = minimal_request();
        request.model = Some("openai/whisper-large-v3".to_owned());
        request.language = Some("de".to_owned());
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("out.json"),
        );
        assert_eq!(
            arg_value(&args, "--model-name"),
            Some("openai/whisper-large-v3")
        );
        assert_eq!(arg_value(&args, "--language"), Some("de"));
    }

    #[test]
    fn batch_size_flag() {
        let mut request = minimal_request();
        request.backend_params.batch_size = Some(24);
        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("out.json"),
        );
        assert_eq!(arg_value(&args, "--batch-size"), Some("24"));
    }

    #[test]
    fn timestamp_level_chunk_and_word() {
        use crate::model::TimestampLevel;
        let mut request = minimal_request();
        request.backend_params.timestamp_level = Some(TimestampLevel::Chunk);
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--timestamp"), Some("chunk"));

        request.backend_params.timestamp_level = Some(TimestampLevel::Word);
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--timestamp"), Some("word"));
    }

    #[test]
    fn hard_speaker_count_only_when_diarize() {
        let mut request = minimal_request();
        set_speaker_count(
            &mut request,
            SpeakerCountRequest::HardConstraint { count: 3 },
        );

        // diarize = false → no speaker constraint flags
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--num-speakers"));

        // diarize = true → flags present
        request.diarize = true;
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--num-speakers"), Some("3"));
        assert!(!has_flag(&args, "--min-speakers"));
        assert!(!has_flag(&args, "--max-speakers"));
    }

    #[test]
    fn gpu_device_and_flash_attention() {
        let mut request = minimal_request();
        request.backend_params.gpu_device = Some("0".to_owned());
        request.backend_params.flash_attention = Some(true);
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--device-id"), Some("0"));
        assert_eq!(arg_value(&args, "--flash"), Some("True"));
    }

    #[test]
    fn flash_attention_false_does_not_emit_flag() {
        let mut request = minimal_request();
        request.backend_params.flash_attention = Some(false);
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--flash"));
    }

    #[test]
    fn build_args_includes_diarization_model_when_requested() {
        let request = request_with_diarization_model();
        let args = build_args(
            &request,
            &PathBuf::from("normalized.wav"),
            &PathBuf::from("out.json"),
        );

        let idx = args
            .iter()
            .position(|arg| arg == "--diarization_model")
            .expect("diarization model flag should exist");
        assert_eq!(
            args.get(idx + 1).map(String::as_str),
            Some("pyannote/speaker-diarization-3.1")
        );
    }

    #[test]
    fn binary_default_is_insanely_fast_whisper() {
        let name = super::binary();
        if std::env::var("FRANKEN_WHISPER_INSANELY_FAST_BIN").is_err() {
            assert_eq!(name, "insanely-fast-whisper");
        }
    }

    #[test]
    fn binary_name_matches_binary() {
        assert_eq!(super::binary_name(), super::binary());
    }

    #[test]
    fn hf_token_returns_none_when_unset() {
        // If neither env var is set, hf_token() returns None.
        if std::env::var("FRANKEN_WHISPER_HF_TOKEN").is_err() && std::env::var("HF_TOKEN").is_err()
        {
            assert!(super::hf_token().is_none());
            assert!(!super::hf_token_present());
        }
    }

    #[test]
    fn diarization_model_requires_diarize_flag() {
        let mut request = request_with_diarization_model();
        request.diarize = false;
        let args = build_args(
            &request,
            &PathBuf::from("normalized.wav"),
            &PathBuf::from("out.json"),
        );
        assert!(
            !has_flag(&args, "--diarization_model"),
            "diarization model flag should not appear when diarize=false"
        );
    }

    #[test]
    fn soft_speaker_count_range_is_not_reinterpreted_as_hard_backend_bounds() {
        let mut request = minimal_request();
        request.diarize = true;
        set_speaker_count(
            &mut request,
            SpeakerCountRequest::Range {
                minimum: 2,
                maximum: 64,
            },
        );
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--num-speakers"));
        assert!(!has_flag(&args, "--min-speakers"));
        assert!(!has_flag(&args, "--max-speakers"));
        super::validate_external_speaker_count(&request)
            .expect("native acoustic overlay owns the soft range");
    }

    #[test]
    fn calibrated_speaker_count_prior_is_rejected_before_subprocess_execution() {
        let mut request = minimal_request();
        request.diarize = true;
        set_speaker_count(
            &mut request,
            SpeakerCountRequest::Prior {
                bins: vec![crate::model::SpeakerCountPriorMass {
                    count: 2,
                    probability: 1.0,
                }],
            },
        );
        request
            .backend_params
            .acoustic_diarization
            .as_mut()
            .expect("diarization request")
            .engine = crate::model::DiarizationEngine::External;
        let error = super::validate_external_speaker_count(&request)
            .expect_err("external backend cannot silently approximate a count prior");
        assert!(error.to_string().contains("cannot faithfully execute"));
    }

    #[test]
    fn flash_attention_none_does_not_emit_flag() {
        let mut request = minimal_request();
        request.backend_params.flash_attention = None;
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--flash"));
    }

    #[test]
    fn all_params_combined() {
        use crate::model::TimestampLevel;
        let mut request = minimal_request();
        request.model = Some("openai/whisper-large-v3-turbo".to_owned());
        request.language = Some("ja".to_owned());
        request.translate = true;
        request.diarize = true;
        request.backend_params.batch_size = Some(16);
        request.backend_params.timestamp_level = Some(TimestampLevel::Word);
        request.backend_params.gpu_device = Some("cuda:1".to_owned());
        request.backend_params.flash_attention = Some(true);
        set_speaker_count(
            &mut request,
            SpeakerCountRequest::HardConstraint { count: 4 },
        );

        let args = build_args(
            &request,
            &PathBuf::from("norm.wav"),
            &PathBuf::from("out.json"),
        );

        assert_eq!(arg_value(&args, "--task"), Some("translate"));
        assert_eq!(
            arg_value(&args, "--model-name"),
            Some("openai/whisper-large-v3-turbo")
        );
        assert_eq!(arg_value(&args, "--language"), Some("ja"));
        assert_eq!(arg_value(&args, "--batch-size"), Some("16"));
        assert_eq!(arg_value(&args, "--timestamp"), Some("word"));
        assert_eq!(arg_value(&args, "--device-id"), Some("cuda:1"));
        assert_eq!(arg_value(&args, "--flash"), Some("True"));
        assert_eq!(arg_value(&args, "--num-speakers"), Some("4"));
        assert!(!has_flag(&args, "--min-speakers"));
        assert!(!has_flag(&args, "--max-speakers"));
    }

    #[test]
    fn diarize_without_diarization_config_omits_diarization_model() {
        let mut request = minimal_request();
        request.diarize = true;
        // No diarization_config set at all.
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--diarization_model"));
    }

    #[test]
    fn diarize_with_empty_diarization_config_omits_model() {
        let mut request = minimal_request();
        request.diarize = true;
        request.backend_params.diarization_config = Some(DiarizationConfig::default());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--diarization_model"));
    }

    #[test]
    fn speaker_count_without_diarize_is_omitted() {
        let mut request = minimal_request();
        request.diarize = false;
        set_speaker_count(
            &mut request,
            SpeakerCountRequest::HardConstraint { count: 5 },
        );
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--num-speakers"));
        assert!(!has_flag(&args, "--min-speakers"));
        assert!(!has_flag(&args, "--max-speakers"));
    }

    #[test]
    fn no_timestamp_level_omits_flag() {
        let request = minimal_request();
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--timestamp"));
    }

    #[test]
    fn no_batch_size_omits_flag() {
        let request = minimal_request();
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--batch-size"));
    }

    #[test]
    fn no_gpu_device_omits_device_id() {
        let request = minimal_request();
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--device-id"));
    }

    #[test]
    fn batch_size_zero_still_emits_flag() {
        let mut request = minimal_request();
        request.backend_params.batch_size = Some(0);
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--batch-size"), Some("0"));
    }

    #[test]
    fn inferred_speaker_count_omits_all_speaker_flags() {
        let mut request = minimal_request();
        request.diarize = true;
        set_speaker_count(&mut request, SpeakerCountRequest::Infer);
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--num-speakers"));
        assert!(!has_flag(&args, "--min-speakers"));
        assert!(!has_flag(&args, "--max-speakers"));
    }

    #[test]
    fn translate_and_diarize_combined() {
        let mut request = minimal_request();
        request.translate = true;
        request.diarize = true;
        request.language = Some("fr".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--task"), Some("translate"));
        assert_eq!(arg_value(&args, "--language"), Some("fr"));
        // diarize=true but no hf_token env var → no --hf-token flag
        if std::env::var("FRANKEN_WHISPER_HF_TOKEN").is_err() && std::env::var("HF_TOKEN").is_err()
        {
            assert!(!has_flag(&args, "--hf-token"));
        }
    }

    #[test]
    fn request_hf_token_override_is_forwarded() {
        let mut request = minimal_request();
        request.diarize = true;
        request.backend_params.insanely_fast_hf_token = Some("hf_from_cli".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--hf-token"), Some("hf_from_cli"));
    }

    #[test]
    fn output_path_uses_request_override_when_present() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_transcript_path =
            Some(PathBuf::from("custom/out.json"));
        let resolved = output_path_for(&request, Path::new("/tmp/work"));
        assert_eq!(resolved, PathBuf::from("custom/out.json"));
    }

    #[test]
    fn output_path_defaults_to_work_dir_when_not_overridden() {
        let request = minimal_request();
        let resolved = output_path_for(&request, Path::new("/tmp/work"));
        assert_eq!(
            resolved,
            PathBuf::from("/tmp/work/insanely_fast_output.json")
        );
    }

    #[test]
    fn model_name_with_complex_path_preserved_exactly() {
        let mut request = minimal_request();
        request.model = Some("org/whisper-large-v3-turbo-int8-fused".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(
            arg_value(&args, "--model-name"),
            Some("org/whisper-large-v3-turbo-int8-fused")
        );
    }

    #[test]
    fn gpu_device_multi_colon_preserved() {
        let mut request = minimal_request();
        request.backend_params.gpu_device = Some("mps:0".to_owned());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--device-id"), Some("mps:0"));
    }

    #[test]
    fn args_order_file_transcript_task_always_first() {
        let mut request = minimal_request();
        request.model = Some("model".to_owned());
        request.language = Some("en".to_owned());
        request.backend_params.batch_size = Some(8);
        let args = build_args(
            &request,
            &PathBuf::from("in.wav"),
            &PathBuf::from("out.json"),
        );
        // First 6 elements should always be --file-name, path, --transcript-path, path, --task, task
        assert_eq!(args[0], "--file-name");
        assert_eq!(args[1], "in.wav");
        assert_eq!(args[2], "--transcript-path");
        assert_eq!(args[3], "out.json");
        assert_eq!(args[4], "--task");
        assert_eq!(args[5], "transcribe");
    }

    // -----------------------------------------------------------------------
    // bd-1rj.3: Extended tuning parameter tests
    // -----------------------------------------------------------------------

    #[test]
    fn device_map_auto() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            device_map: Some(DeviceMapStrategy::Auto),
            ..InsanelyFastTuningParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--device-map"), Some("auto"));
    }

    #[test]
    fn device_map_sequential() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            device_map: Some(DeviceMapStrategy::Sequential),
            ..InsanelyFastTuningParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--device-map"), Some("sequential"));
    }

    #[test]
    fn device_map_none_omits_flag() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            device_map: None,
            ..InsanelyFastTuningParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--device-map"));
    }

    #[test]
    fn torch_dtype_float16() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            torch_dtype: Some("float16".to_owned()),
            ..InsanelyFastTuningParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--torch-dtype"), Some("float16"));
    }

    #[test]
    fn torch_dtype_bfloat16() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            torch_dtype: Some("bfloat16".to_owned()),
            ..InsanelyFastTuningParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--torch-dtype"), Some("bfloat16"));
    }

    #[test]
    fn torch_dtype_none_omits_flag() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams::default());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--torch-dtype"));
    }

    #[test]
    fn disable_better_transformer_flag() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            disable_better_transformer: true,
            ..InsanelyFastTuningParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(has_flag(&args, "--disable-better-transformer"));
    }

    #[test]
    fn disable_better_transformer_false_omits_flag() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            disable_better_transformer: false,
            ..InsanelyFastTuningParams::default()
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--disable-better-transformer"));
    }

    #[test]
    fn no_tuning_params_omits_all_tuning_flags() {
        let request = minimal_request();
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--device-map"));
        assert!(!has_flag(&args, "--torch-dtype"));
        assert!(!has_flag(&args, "--disable-better-transformer"));
    }

    #[test]
    fn all_tuning_params_combined() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            device_map: Some(DeviceMapStrategy::Auto),
            torch_dtype: Some("float16".to_owned()),
            disable_better_transformer: true,
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--device-map"), Some("auto"));
        assert_eq!(arg_value(&args, "--torch-dtype"), Some("float16"));
        assert!(has_flag(&args, "--disable-better-transformer"));
    }

    #[test]
    fn tuning_params_coexist_with_flash_attention() {
        let mut request = minimal_request();
        request.backend_params.flash_attention = Some(true);
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            device_map: Some(DeviceMapStrategy::Auto),
            torch_dtype: Some("float16".to_owned()),
            disable_better_transformer: false,
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--flash"), Some("True"));
        assert_eq!(arg_value(&args, "--device-map"), Some("auto"));
        assert_eq!(arg_value(&args, "--torch-dtype"), Some("float16"));
        assert!(!has_flag(&args, "--disable-better-transformer"));
    }

    #[test]
    fn tuning_params_coexist_with_batch_size_and_device() {
        let mut request = minimal_request();
        request.backend_params.batch_size = Some(32);
        request.backend_params.gpu_device = Some("cuda:0".to_owned());
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams {
            device_map: Some(DeviceMapStrategy::Auto),
            torch_dtype: Some("bfloat16".to_owned()),
            disable_better_transformer: true,
        });
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert_eq!(arg_value(&args, "--batch-size"), Some("32"));
        assert_eq!(arg_value(&args, "--device-id"), Some("cuda:0"));
        assert_eq!(arg_value(&args, "--device-map"), Some("auto"));
        assert_eq!(arg_value(&args, "--torch-dtype"), Some("bfloat16"));
        assert!(has_flag(&args, "--disable-better-transformer"));
    }

    #[test]
    fn empty_tuning_params_emit_nothing() {
        let mut request = minimal_request();
        request.backend_params.insanely_fast_tuning = Some(InsanelyFastTuningParams::default());
        let args = build_args(&request, &PathBuf::from("n.wav"), &PathBuf::from("o.json"));
        assert!(!has_flag(&args, "--device-map"));
        assert!(!has_flag(&args, "--torch-dtype"));
        assert!(!has_flag(&args, "--disable-better-transformer"));
    }

    #[test]
    fn device_map_strategy_serialization_round_trip() {
        for strategy in [DeviceMapStrategy::Auto, DeviceMapStrategy::Sequential] {
            let serialized = serde_json::to_string(&strategy).unwrap();
            let deserialized: DeviceMapStrategy = serde_json::from_str(&serialized).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn insanely_fast_tuning_params_serde_round_trip() {
        let params = InsanelyFastTuningParams {
            device_map: Some(DeviceMapStrategy::Auto),
            torch_dtype: Some("float16".to_owned()),
            disable_better_transformer: true,
        };
        let serialized = serde_json::to_string(&params).unwrap();
        let deserialized: InsanelyFastTuningParams = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.device_map, Some(DeviceMapStrategy::Auto));
        assert_eq!(deserialized.torch_dtype.as_deref(), Some("float16"));
        assert!(deserialized.disable_better_transformer);
    }
}
