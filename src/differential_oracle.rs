//! Development-only differential diagnostics against external diarization tools.
//!
//! External systems are deliberately treated as fallible diagnostic oracles,
//! never as authorities over the native Rust pipeline. The adapter protocol is
//! transcript-free, path-free after execution, and absent from normal
//! transcription. No external tool is a Cargo or runtime dependency.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diarization::{
    ChangePointScore, DiarizationScore, EvaluationTurn, ScoringTurn, score_change_points,
    score_diarization,
};
use crate::error::{FwError, FwResult};
use crate::orchestrator::CancellationToken;
use crate::process::run_command_cancellable_with_probe;

pub const DIFFERENTIAL_ORACLE_PROTOCOL_VERSION: &str =
    "franken-whisper-diarization-oracle-protocol-v2";
pub const DIFFERENTIAL_ORACLE_VERSION_SCHEMA: &str =
    "franken-whisper-diarization-oracle-version-v2";
pub const DIFFERENTIAL_STAGE_DOCUMENT_SCHEMA: &str =
    "franken-whisper-diarization-stage-document-v1";
pub const DIFFERENTIAL_REPORT_SCHEMA: &str = "franken-whisper-differential-report-v2";
pub const DIFFERENTIAL_COMPARATOR_VERSION: &str = "differential-comparator-v2";
/// Cross-language hash encoding: recursively sort object keys, emit compact JSON,
/// and preserve array order and serde_json's scalar rendering.
pub const DIFFERENTIAL_CANONICAL_JSON_VERSION: &str = "lexicographic-canonical-json-v1";

/// Frozen upstream model selected for the first end-to-end learned comparator.
pub const SORTFORMER_ORACLE_MODEL_ID: &str = "nvidia/diar_streaming_sortformer_4spk-v2.1";
/// Immutable Hugging Face repository revision used by the comparator contract.
pub const SORTFORMER_ORACLE_MODEL_REVISION: &str = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d";
/// SHA-256 advertised by Hugging Face LFS metadata for the pinned `.nemo` artifact.
/// The adapter must independently hash its local bytes and attest this exact value.
pub const SORTFORMER_ORACLE_ARTIFACT_SHA256: &str =
    "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8";
/// Byte length advertised by the pinned upstream repository.
pub const SORTFORMER_ORACLE_ARTIFACT_BYTES: u64 = 471_367_680;
/// Fixed number of output speaker slots in the pinned model.
pub const SORTFORMER_ORACLE_MAX_SPEAKERS: usize = 4;
/// Temporal stride of the pinned model's output probabilities.
pub const SORTFORMER_ORACLE_OUTPUT_FRAME_MS: u32 = 80;
/// Maximum end-of-file rounding difference accepted between PCM and stage duration.
pub const SORTFORMER_AUDIO_DURATION_TOLERANCE_MS: u32 = 79;
/// Exact operator adapter version accepted by the frozen contract.
pub const SORTFORMER_ORACLE_ADAPTER_VERSION: &str = "franken-whisper-sortformer-oracle-v2";
/// SHA-256 required by final-output Sortformer diagnostic evidence. Accepted
/// activation-seam authority requires a separately reviewed exporter identity
/// and closure of the executable race; self-reported versions are not identities.
pub const SORTFORMER_ORACLE_ADAPTER_SHA256: &str =
    "8f376c979b7eaca41dc0a438d9aaa41c1c723052b97c45eb2acc59b6d6f00bde";
/// Pinned NeMo Speech source revision expected behind the operator adapter.
pub const SORTFORMER_ORACLE_TOOL_VERSION: &str =
    "nemo-speech-40ace43c7cf151af78dc22027c02feeca7e06b6a";
/// Exact Python version qualified for the external oracle runtime.
pub const SORTFORMER_ORACLE_PYTHON_VERSION: &str = "3.12.12";
/// Exact NeMo package version qualified for the external oracle runtime.
pub const SORTFORMER_ORACLE_NEMO_VERSION: &str = "3.1.0+40ace43c7c";
/// Git source revision that the installed NeMo distribution must attest.
pub const SORTFORMER_ORACLE_NEMO_SOURCE_REVISION: &str = "40ace43c7cf151af78dc22027c02feeca7e06b6a";
/// Exact PyTorch version qualified for the external oracle runtime.
pub const SORTFORMER_ORACLE_TORCH_VERSION: &str = "2.7.1";
/// Exact torchaudio version qualified for the external oracle runtime.
pub const SORTFORMER_ORACLE_TORCHAUDIO_VERSION: &str = "2.7.1";
/// Exact NumPy version qualified for the external oracle runtime.
pub const SORTFORMER_ORACLE_NUMPY_VERSION: &str = "2.4.6";
/// SHA-256 of the canonical JSON serialization of [`sortformer_oracle_contract`].
pub const SORTFORMER_ORACLE_CONTRACT_SHA256: &str =
    "7ac048e3372fe4c622840beddfbeef42944d961408360324cb7276a69c8542c5";

const MAX_DOCUMENT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_DURATION_MS: u64 = 24 * 60 * 60 * 1_000;
const MAX_INTERVALS: usize = 200_000;
const MAX_WORDS: usize = 500_000;
const MAX_CLUSTERS: usize = 200_000;
const MAX_TURNS: usize = 200_000;
const MAX_COMPARISON_CHANGE_POINTS: usize = 2_048;
const MAX_COMPARISON_TURNS: usize = 2_048;
const MAX_COMPARISON_SPEAKERS: usize = 32;
const MAX_SAFE_TOKEN_LEN: usize = 128;
const HASH_HEX_LEN: usize = 64;

/// Supported external diagnostic families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialOracleTool {
    Pyannote,
    NemoSpectral,
    Vbx,
    Eend,
    Diaper,
    Sortformer,
}

impl DifferentialOracleTool {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pyannote => "pyannote",
            Self::NemoSpectral => "nemo_spectral",
            Self::Vbx => "vbx",
            Self::Eend => "eend",
            Self::Diaper => "diaper",
            Self::Sortformer => "sortformer",
        }
    }

    #[must_use]
    pub const fn family(self) -> DifferentialOracleFamily {
        match self {
            Self::Pyannote | Self::NemoSpectral => DifferentialOracleFamily::Cascaded,
            Self::Vbx => DifferentialOracleFamily::BayesianHmm,
            Self::Eend | Self::Diaper | Self::Sortformer => {
                DifferentialOracleFamily::EndToEndAttractor
            }
        }
    }

    #[must_use]
    pub const fn executable_env(self) -> &'static str {
        match self {
            Self::Pyannote => "FRANKEN_WHISPER_PYANNOTE_ORACLE_BIN",
            Self::NemoSpectral => "FRANKEN_WHISPER_NEMO_SPECTRAL_ORACLE_BIN",
            Self::Vbx => "FRANKEN_WHISPER_VBX_ORACLE_BIN",
            Self::Eend => "FRANKEN_WHISPER_EEND_ORACLE_BIN",
            Self::Diaper => "FRANKEN_WHISPER_DIAPER_ORACLE_BIN",
            Self::Sortformer => "FRANKEN_WHISPER_SORTFORMER_ORACLE_BIN",
        }
    }

    #[must_use]
    pub const fn default_program(self) -> &'static str {
        match self {
            Self::Pyannote => "franken-whisper-pyannote-oracle",
            Self::NemoSpectral => "franken-whisper-nemo-spectral-oracle",
            Self::Vbx => "franken-whisper-vbx-oracle",
            Self::Eend => "franken-whisper-eend-oracle",
            Self::Diaper => "franken-whisper-diaper-oracle",
            Self::Sortformer => "franken-whisper-sortformer-oracle",
        }
    }
}

/// Broad architecture class used to avoid treating one implementation as consensus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialOracleFamily {
    Cascaded,
    BayesianHmm,
    EndToEndAttractor,
}

/// Path-free registry entry for one operator-installed adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialOracleRegistryEntry {
    pub tool: DifferentialOracleTool,
    pub family: DifferentialOracleFamily,
    pub executable_env: String,
    pub default_program: String,
    pub protocol_version: String,
    pub authority: DifferentialAuthority,
    pub model_contract: Option<DifferentialOracleModelContract>,
    pub model_contract_sha256: Option<String>,
}

/// Frozen model, input, streaming, and post-processing semantics for an oracle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialOracleModelContract {
    pub schema_version: String,
    pub canonical_json_version: String,
    pub model_id: String,
    pub model_revision: String,
    /// Expected content hash from upstream metadata; local bytes are independently hashed.
    pub upstream_artifact_sha256: String,
    pub upstream_artifact_bytes: u64,
    pub upstream_license: String,
    pub input_format: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub output_frame_ms: u32,
    pub audio_duration_tolerance_ms: u32,
    pub output_end_alignment: String,
    pub validate_all_pcm_samples: bool,
    pub runtime_fingerprint_schema: String,
    pub runtime_fingerprint_required_fields: Vec<String>,
    pub python_version: String,
    pub nemo_version: String,
    pub nemo_source_revision: String,
    pub torch_version: String,
    pub torchaudio_version: String,
    pub numpy_version: String,
    pub maximum_speakers: u16,
    pub speaker_count_mode: String,
    pub label_order: String,
    pub batch_size: u16,
    pub device: String,
    pub compute_dtype: String,
    pub autocast: bool,
    pub quantization: String,
    pub data_loader_workers: u16,
    pub torch_intraop_threads: u16,
    pub torch_interop_threads: u16,
    pub deterministic_algorithms: bool,
    pub chunk_frames: u32,
    pub left_context_frames: u32,
    pub right_context_frames: u32,
    pub fifo_frames: u32,
    pub speaker_cache_update_period_frames: u32,
    pub speaker_cache_frames: u32,
    pub speaker_cache_silence_frames_per_speaker: u32,
    pub speaker_cache_pop_rule: String,
    pub first_full_chunk_cache_pop_frames: u32,
    pub steady_full_chunk_cache_pop_frames: u32,
    pub subsampling_factor: u32,
    pub prediction_score_threshold_millionths: u32,
    pub score_noise_millionths: u32,
    pub latest_score_boost_millionths: u32,
    pub silence_threshold_millionths: u32,
    pub strong_boost_rate_millionths: u32,
    pub weak_boost_rate_millionths: u32,
    pub minimum_positive_scores_rate_millionths: u32,
    pub causal_attention_rate_millionths: u32,
    pub causal_attention_right_context_frames: u32,
    pub maximum_cache_index: u32,
    pub frontend_window_stride_micros: u32,
    pub nominal_input_buffer_latency_ms: u32,
    pub postprocessing_onset_millionths: u32,
    pub postprocessing_offset_millionths: u32,
    pub postprocessing_pad_onset_ms: u32,
    pub postprocessing_pad_offset_ms: u32,
    pub postprocessing_min_duration_on_ms: u32,
    pub postprocessing_min_duration_off_ms: u32,
}

/// Path-free runtime facts attested by the operator adapter and validated by the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialOracleRuntimeFingerprint {
    pub schema_version: String,
    pub python_version: String,
    pub nemo_version: String,
    pub torch_version: String,
    pub torchaudio_version: String,
    pub numpy_version: String,
    pub blas_backend: String,
    pub operating_system: String,
    pub machine_architecture: String,
    pub cpu_feature_tier: String,
    pub device: String,
    pub compute_dtype: String,
    pub autocast: bool,
    pub quantization: String,
    pub torch_intraop_threads: u16,
    pub torch_interop_threads: u16,
    pub data_loader_workers: u16,
    pub deterministic_algorithms: bool,
}

/// Return the exact Sortformer profile accepted by the external adapter seam.
#[must_use]
pub fn sortformer_oracle_contract() -> DifferentialOracleModelContract {
    DifferentialOracleModelContract {
        schema_version: "franken-whisper-sortformer-oracle-contract-v2".to_owned(),
        canonical_json_version: DIFFERENTIAL_CANONICAL_JSON_VERSION.to_owned(),
        model_id: SORTFORMER_ORACLE_MODEL_ID.to_owned(),
        model_revision: SORTFORMER_ORACLE_MODEL_REVISION.to_owned(),
        upstream_artifact_sha256: SORTFORMER_ORACLE_ARTIFACT_SHA256.to_owned(),
        upstream_artifact_bytes: SORTFORMER_ORACLE_ARTIFACT_BYTES,
        upstream_license: "nvidia-open-model-license".to_owned(),
        input_format: "pcm_s16le_mono_wav".to_owned(),
        sample_rate_hz: 16_000,
        channels: 1,
        output_frame_ms: SORTFORMER_ORACLE_OUTPUT_FRAME_MS,
        audio_duration_tolerance_ms: SORTFORMER_AUDIO_DURATION_TOLERANCE_MS,
        output_end_alignment: "output_frame_or_document_duration".to_owned(),
        validate_all_pcm_samples: true,
        runtime_fingerprint_schema: "sortformer-runtime-fingerprint-v1".to_owned(),
        runtime_fingerprint_required_fields: [
            "schema_version",
            "python_version",
            "nemo_version",
            "torch_version",
            "torchaudio_version",
            "numpy_version",
            "blas_backend",
            "operating_system",
            "machine_architecture",
            "cpu_feature_tier",
            "device",
            "compute_dtype",
            "autocast",
            "quantization",
            "torch_intraop_threads",
            "torch_interop_threads",
            "data_loader_workers",
            "deterministic_algorithms",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        python_version: SORTFORMER_ORACLE_PYTHON_VERSION.to_owned(),
        nemo_version: SORTFORMER_ORACLE_NEMO_VERSION.to_owned(),
        nemo_source_revision: SORTFORMER_ORACLE_NEMO_SOURCE_REVISION.to_owned(),
        torch_version: SORTFORMER_ORACLE_TORCH_VERSION.to_owned(),
        torchaudio_version: SORTFORMER_ORACLE_TORCHAUDIO_VERSION.to_owned(),
        numpy_version: SORTFORMER_ORACLE_NUMPY_VERSION.to_owned(),
        maximum_speakers: 4,
        speaker_count_mode: "infer_up_to_four".to_owned(),
        label_order: "arrival_time_order".to_owned(),
        batch_size: 1,
        device: "cpu".to_owned(),
        compute_dtype: "float32".to_owned(),
        autocast: false,
        quantization: "none".to_owned(),
        data_loader_workers: 0,
        torch_intraop_threads: 8,
        torch_interop_threads: 1,
        deterministic_algorithms: true,
        chunk_frames: 340,
        left_context_frames: 1,
        right_context_frames: 40,
        fifo_frames: 40,
        speaker_cache_update_period_frames: 300,
        speaker_cache_frames: 188,
        speaker_cache_silence_frames_per_speaker: 3,
        speaker_cache_pop_rule:
            "min(max(configured_update,chunk-fifo_capacity+current_fifo),current_fifo+chunk)-v1"
                .to_owned(),
        first_full_chunk_cache_pop_frames: 300,
        steady_full_chunk_cache_pop_frames: 340,
        subsampling_factor: 8,
        prediction_score_threshold_millionths: 250_000,
        score_noise_millionths: 0,
        latest_score_boost_millionths: 50_000,
        silence_threshold_millionths: 200_000,
        strong_boost_rate_millionths: 750_000,
        weak_boost_rate_millionths: 1_500_000,
        minimum_positive_scores_rate_millionths: 500_000,
        causal_attention_rate_millionths: 500_000,
        causal_attention_right_context_frames: 7,
        maximum_cache_index: 99_999,
        frontend_window_stride_micros: 10_000,
        nominal_input_buffer_latency_ms: 30_400,
        postprocessing_onset_millionths: 500_000,
        postprocessing_offset_millionths: 500_000,
        postprocessing_pad_onset_ms: 0,
        postprocessing_pad_offset_ms: 0,
        postprocessing_min_duration_on_ms: 0,
        postprocessing_min_duration_off_ms: 0,
    }
}

/// Emit the stable adapter registry without probing the host.
#[must_use]
pub fn differential_oracle_registry() -> Vec<DifferentialOracleRegistryEntry> {
    [
        DifferentialOracleTool::Pyannote,
        DifferentialOracleTool::NemoSpectral,
        DifferentialOracleTool::Vbx,
        DifferentialOracleTool::Eend,
        DifferentialOracleTool::Diaper,
        DifferentialOracleTool::Sortformer,
    ]
    .into_iter()
    .map(|tool| DifferentialOracleRegistryEntry {
        tool,
        family: tool.family(),
        executable_env: tool.executable_env().to_owned(),
        default_program: tool.default_program().to_owned(),
        protocol_version: DIFFERENTIAL_ORACLE_PROTOCOL_VERSION.to_owned(),
        authority: DifferentialAuthority::DiagnosticOnly,
        model_contract: (tool == DifferentialOracleTool::Sortformer)
            .then(sortformer_oracle_contract),
        model_contract_sha256: (tool == DifferentialOracleTool::Sortformer)
            .then(|| SORTFORMER_ORACLE_CONTRACT_SHA256.to_owned()),
    })
    .collect()
}

/// Fixed authority statement attached to every report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialAuthority {
    DiagnosticOnly,
}

/// Integer-millisecond activity or overlap interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialInterval {
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Transcript-free aligned word timing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialWordTiming {
    /// Opaque identity. Lexical text is forbidden by the adapter contract.
    pub word_id: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Transcript-free cluster assignment for one stable acoustic segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialClusterAssignment {
    pub segment_id: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub cluster_label: String,
    pub confidence: Option<f64>,
}

/// Canonical transcript-free output shared by native and external diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialStageDocument {
    pub schema_version: String,
    /// Lowercase SHA-256 of the operator-local recording identity.
    pub recording_key: String,
    pub duration_ms: u64,
    pub speech_activity: Option<Vec<DifferentialInterval>>,
    pub word_timing: Option<Vec<DifferentialWordTiming>>,
    pub change_boundaries_ms: Option<Vec<u64>>,
    pub cluster_assignments: Option<Vec<DifferentialClusterAssignment>>,
    pub overlap: Option<Vec<DifferentialInterval>>,
    pub final_projection: Option<Vec<EvaluationTurn>>,
}

/// Frozen comparison thresholds. They define a diagnostic divergence, not a gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialComparisonConfig {
    pub schema_version: String,
    pub minimum_interval_iou: f64,
    pub word_boundary_collar_ms: u64,
    pub minimum_word_timing_recall: f64,
    pub change_boundary_collar_ms: u64,
    pub minimum_change_f1: f64,
    pub minimum_cluster_segment_coverage: f64,
    pub maximum_cluster_pair_disagreement: f64,
    pub maximum_projection_der: f64,
    pub adjudication_epsilon: f64,
}

impl Default for DifferentialComparisonConfig {
    fn default() -> Self {
        Self {
            schema_version: "differential-comparison-config-v1".to_owned(),
            minimum_interval_iou: 0.95,
            word_boundary_collar_ms: 100,
            minimum_word_timing_recall: 0.95,
            change_boundary_collar_ms: 250,
            minimum_change_f1: 0.90,
            minimum_cluster_segment_coverage: 0.95,
            maximum_cluster_pair_disagreement: 0.05,
            maximum_projection_der: 0.05,
            adjudication_epsilon: 0.001,
        }
    }
}

/// Ordered stage at which two systems may first diverge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialStage {
    SpeechActivity,
    WordTiming,
    ChangeBoundaries,
    ClusterAssignments,
    Overlap,
    FinalProjection,
}

/// Whether a stage was comparable and exceeded its frozen diagnostic threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialStageState {
    Equivalent,
    Divergent,
    MissingNative,
    MissingOracle,
    MissingBoth,
}

/// Reference-assisted interpretation. Even a reference-favored result is diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialAdjudication {
    NoDisagreement,
    InconclusiveNoReference,
    ReferenceFavorsNative,
    ReferenceFavorsOracle,
    ReferenceTied,
    ReferenceStageUnavailable,
}

/// Set-overlap arithmetic for VAD and overlap regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialIntervalScore {
    pub native_duration_ms: u64,
    pub oracle_duration_ms: u64,
    pub intersection_ms: u64,
    pub union_ms: u64,
    pub native_only_ms: u64,
    pub oracle_only_ms: u64,
    pub iou: Option<f64>,
}

/// Timing agreement over opaque shared word identities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialWordTimingScore {
    pub native_word_count: usize,
    pub oracle_word_count: usize,
    pub shared_word_count: usize,
    pub within_collar_count: usize,
    pub timing_recall: Option<f64>,
    pub mean_absolute_boundary_error_ms: Option<f64>,
}

/// Label-permutation-invariant pairwise clustering agreement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialClusterScore {
    pub native_segment_count: usize,
    pub oracle_segment_count: usize,
    pub shared_segment_count: usize,
    pub segment_coverage: Option<f64>,
    pub geometry_disagreement_count: usize,
    pub geometry_disagreement_rate: Option<f64>,
    pub compared_pair_count: u64,
    pub pair_disagreement_count: u64,
    pub pair_disagreement_rate: Option<f64>,
    pub coassignment_precision: Option<f64>,
    pub coassignment_recall: Option<f64>,
    pub coassignment_f1: Option<f64>,
    pub shared_confidence_count: usize,
    pub confidence_availability_disagreement_count: usize,
    pub mean_absolute_confidence_delta: Option<f64>,
}

/// Label-free projection score derived from the authoritative permutation matcher.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialProjectionScore {
    pub native_speaker_time_sec: f64,
    pub missed_speech_sec: f64,
    pub false_alarm_sec: f64,
    pub speaker_confusion_sec: f64,
    pub native_unknown_ms: u64,
    pub oracle_unknown_ms: u64,
    pub unknown_status_disagreement_ms: u64,
    pub der: Option<f64>,
    pub jer: Option<f64>,
    pub mapping_cardinality: usize,
}

/// Stage-specific metric payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "metric", rename_all = "snake_case")]
pub enum DifferentialStageMetric {
    Intervals(DifferentialIntervalScore),
    WordTiming(DifferentialWordTimingScore),
    ChangeBoundaries(ChangePointScore),
    ClusterAssignments(DifferentialClusterScore),
    FinalProjection(DifferentialProjectionScore),
}

/// One native-versus-oracle comparison and optional reference interpretation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialStageComparison {
    pub stage: DifferentialStage,
    pub state: DifferentialStageState,
    pub metric: Option<DifferentialStageMetric>,
    pub diagnostic_loss: Option<f64>,
    pub adjudication: DifferentialAdjudication,
    pub native_reference_loss: Option<f64>,
    pub oracle_reference_loss: Option<f64>,
}

/// External subprocess phase that failed or was skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialExecutionStage {
    ResolveExecutable,
    HashExecutable,
    InputValidation,
    InputPostRunValidation,
    EligibilityValidation,
    VersionProbe,
    VersionValidation,
    OracleRun,
    OracleOutputValidation,
}

/// Stable reason that an optional external diagnostic did not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialSkipReason {
    MissingExecutable,
    UnreadableExecutable,
    ExecutableIdentityMismatch,
    InputContractMismatch,
    InputIdentityMismatch,
    VersionProbeFailed,
    VersionProbeTimedOut,
    InvalidVersionOutput,
    ProtocolVersionMismatch,
    ToolIdentityMismatch,
    ModelContractMismatch,
    OracleRunFailed,
    OracleRunTimedOut,
    InvalidOracleOutput,
    ModelCapacityExceeded,
    ReferenceModelCapacityExceeded,
    OracleIdentityMismatch,
}

/// Completed or cleanly skipped diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialRunStatus {
    Completed,
    Skipped,
}

/// Path-free external-tool provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialProvenance {
    pub protocol_version: String,
    pub tool: DifferentialOracleTool,
    pub family: DifferentialOracleFamily,
    pub tool_version: Option<String>,
    pub adapter_version: Option<String>,
    /// Host-selected contract, present even when the external tool is unavailable.
    pub expected_model_contract_sha256: Option<String>,
    /// Contract actually attested by a successfully validated version probe.
    pub model_contract_sha256: Option<String>,
    pub model_artifact_sha256: Option<String>,
    pub model_artifact_bytes: Option<u64>,
    pub runtime_fingerprint_sha256: Option<String>,
    pub executable_sha256: Option<String>,
    pub version_stdout_sha256: Option<String>,
    pub oracle_stdout_sha256: Option<String>,
    pub audio_sha256: String,
    pub native_document_sha256: String,
    pub reference_document_sha256: Option<String>,
}

/// Retained differential diagnostic. It intentionally contains no paths or content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferentialOracleReport {
    pub schema_version: String,
    pub comparator_version: String,
    pub authority: DifferentialAuthority,
    pub native_incorrectness_claim_permitted: bool,
    pub status: DifferentialRunStatus,
    pub skip_reason: Option<DifferentialSkipReason>,
    pub failure_stage: Option<DifferentialExecutionStage>,
    pub provenance: DifferentialProvenance,
    pub comparison_config: DifferentialComparisonConfig,
    pub comparison_config_sha256: String,
    pub comparisons: Vec<DifferentialStageComparison>,
    pub earliest_divergence: Option<DifferentialStage>,
    /// Hash with this field set to the empty string.
    pub result_sha256: String,
}

/// External-only developer request. Paths cannot be logged through `Debug` or Serde.
pub struct DifferentialOracleRequest<'a> {
    pub project_root: &'a Path,
    pub audio_path: &'a Path,
    pub native_document_path: &'a Path,
    pub reference_document_path: Option<&'a Path>,
    pub output_path: &'a Path,
    pub tool: DifferentialOracleTool,
    pub hard_timeout: Duration,
    pub comparison_config: DifferentialComparisonConfig,
}

/// One canonical external audio input for the in-memory Sortformer seam.
///
/// This request deliberately has neither `Debug` nor Serde implementations so
/// its filesystem path cannot be retained accidentally. The caller must pass
/// the canonical absolute path of a file that it keeps outside retained
/// reports; the returned outcome contains no path.
pub(crate) struct SortformerObservationRequest<'a> {
    pub(crate) audio_path: &'a Path,
    pub(crate) expected_audio_sha256: &'a str,
    pub(crate) expected_duration_ms: u64,
    pub(crate) recording_key: &'a str,
    pub(crate) hard_timeout: Duration,
}

/// Path-free provenance retained for one in-memory Sortformer observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SortformerObservationProvenance {
    pub(crate) protocol_version: String,
    pub(crate) tool: DifferentialOracleTool,
    pub(crate) family: DifferentialOracleFamily,
    pub(crate) authority: DifferentialAuthority,
    pub(crate) tool_version: Option<String>,
    pub(crate) adapter_version: Option<String>,
    pub(crate) expected_model_contract_sha256: String,
    pub(crate) model_contract_sha256: Option<String>,
    pub(crate) model_artifact_sha256: Option<String>,
    pub(crate) model_artifact_bytes: Option<u64>,
    pub(crate) runtime_fingerprint_sha256: Option<String>,
    pub(crate) executable_sha256: Option<String>,
    pub(crate) version_stdout_sha256: Option<String>,
    pub(crate) oracle_stdout_sha256: Option<String>,
    pub(crate) audio_sha256: String,
}

/// Completed or cleanly skipped result from the in-memory Sortformer seam.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SortformerObservationOutcome {
    Completed {
        document: DifferentialStageDocument,
        provenance: SortformerObservationProvenance,
    },
    Skipped {
        reason: DifferentialSkipReason,
        stage: DifferentialExecutionStage,
        provenance: SortformerObservationProvenance,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleVersionDocument {
    schema_version: String,
    protocol_version: String,
    tool: DifferentialOracleTool,
    tool_version: String,
    adapter_version: String,
    model_contract_sha256: Option<String>,
    model_artifact_sha256: Option<String>,
    model_artifact_bytes: Option<u64>,
    runtime_fingerprint: Option<DifferentialOracleRuntimeFingerprint>,
    runtime_fingerprint_sha256: Option<String>,
}

#[derive(Debug, Clone)]
struct ProgramSpec {
    program: String,
    prefix_args: Vec<String>,
}

struct PreparedInputs {
    audio_path: PathBuf,
    native: DifferentialStageDocument,
    reference: Option<DifferentialStageDocument>,
    audio_sha256: String,
    native_sha256: String,
    reference_sha256: Option<String>,
}

#[derive(Debug)]
struct ExternalSuccess {
    version: OracleVersionDocument,
    executable_sha256: String,
    version_stdout_sha256: String,
    oracle_stdout_sha256: String,
    oracle: DifferentialStageDocument,
}

#[derive(Debug, Clone)]
struct ExternalSkip {
    reason: DifferentialSkipReason,
    stage: DifferentialExecutionStage,
    executable_sha256: Option<String>,
    version: Option<Box<OracleVersionDocument>>,
    version_stdout_sha256: Option<String>,
    oracle_stdout_sha256: Option<String>,
}

impl ExternalSkip {
    fn new(reason: DifferentialSkipReason, stage: DifferentialExecutionStage) -> Self {
        Self {
            reason,
            stage,
            executable_sha256: None,
            version: None,
            version_stdout_sha256: None,
            oracle_stdout_sha256: None,
        }
    }

    fn with_executable(mut self, executable_sha256: &str) -> Self {
        self.executable_sha256 = Some(executable_sha256.to_owned());
        self
    }

    fn with_version_probe(mut self, stdout_sha256: &str) -> Self {
        self.version_stdout_sha256 = Some(stdout_sha256.to_owned());
        self
    }

    fn with_valid_version(mut self, version: &OracleVersionDocument, stdout_sha256: &str) -> Self {
        self.version = Some(Box::new(version.clone()));
        self.version_stdout_sha256 = Some(stdout_sha256.to_owned());
        self
    }

    fn with_oracle_stdout(mut self, stdout_sha256: &str) -> Self {
        self.oracle_stdout_sha256 = Some(stdout_sha256.to_owned());
        self
    }
}

/// Run one explicit developer diagnostic and create a new path-free report.
pub fn run_differential_oracle(
    request: DifferentialOracleRequest<'_>,
) -> FwResult<DifferentialOracleReport> {
    let cancellation = CancellationToken::unbounded();
    run_differential_oracle_with_token(request, &cancellation)
}

/// Variant used by library callers whose cancellation source is not the
/// process-global shutdown controller.
pub(crate) fn run_sortformer_observation_with_cancel(
    request: SortformerObservationRequest<'_>,
    token: &CancellationToken,
    is_cancelled: &(dyn Fn() -> bool + Sync),
) -> FwResult<SortformerObservationOutcome> {
    let program = resolve_program(DifferentialOracleTool::Sortformer);
    run_sortformer_observation_with_program_and_probe(
        request,
        &program,
        token,
        Some(is_cancelled),
        Some(SORTFORMER_ORACLE_ADAPTER_SHA256),
    )
}

#[cfg(test)]
fn run_sortformer_observation_with_program(
    request: SortformerObservationRequest<'_>,
    program: &ProgramSpec,
    token: &CancellationToken,
) -> FwResult<SortformerObservationOutcome> {
    run_sortformer_observation_with_program_and_probe(request, program, token, None, None)
}

fn run_sortformer_observation_with_program_and_probe(
    request: SortformerObservationRequest<'_>,
    program: &ProgramSpec,
    token: &CancellationToken,
    additional_cancel: Option<&(dyn Fn() -> bool + Sync)>,
    expected_executable_sha256: Option<&str>,
) -> FwResult<SortformerObservationOutcome> {
    validate_sortformer_observation_request(&request, token)?;
    let external = execute_external(
        DifferentialOracleTool::Sortformer,
        program,
        request.audio_path,
        request.expected_audio_sha256,
        request.expected_duration_ms,
        request.recording_key,
        ExternalRunLimits::new(request.hard_timeout, token)
            .with_additional_cancel(additional_cancel)
            .with_expected_executable_sha256(expected_executable_sha256),
    );
    match external {
        Ok(success) => {
            let provenance = sortformer_observation_provenance_from_success(
                request.expected_audio_sha256,
                &success,
            );
            if success.oracle.recording_key != request.recording_key
                || success.oracle.duration_ms != request.expected_duration_ms
            {
                return Ok(SortformerObservationOutcome::Skipped {
                    reason: DifferentialSkipReason::OracleIdentityMismatch,
                    stage: DifferentialExecutionStage::OracleOutputValidation,
                    provenance,
                });
            }
            validate_sortformer_stage_document(&success.oracle)?;
            Ok(SortformerObservationOutcome::Completed {
                document: success.oracle,
                provenance,
            })
        }
        Err(ExternalRunError::Cancelled(error)) => Err(error),
        Err(ExternalRunError::Skipped(skipped)) => {
            let provenance = sortformer_observation_provenance_from_skip(
                request.expected_audio_sha256,
                &skipped,
            );
            Ok(SortformerObservationOutcome::Skipped {
                reason: skipped.reason,
                stage: skipped.stage,
                provenance,
            })
        }
    }
}

fn validate_sortformer_observation_request(
    request: &SortformerObservationRequest<'_>,
    token: &CancellationToken,
) -> FwResult<()> {
    token.checkpoint()?;
    if request.hard_timeout.is_zero() || request.hard_timeout > Duration::from_secs(24 * 60 * 60) {
        return Err(oracle_request_error(
            "hard_timeout",
            "hard timeout must be within (0, 24 hours]",
        ));
    }
    if !is_sha256_hex(request.expected_audio_sha256) {
        return Err(oracle_request_error(
            "audio_sha256",
            "expected audio identity must be lowercase SHA-256",
        ));
    }
    if !is_sha256_hex(request.recording_key) {
        return Err(oracle_request_error(
            "recording_key",
            "recording key must be lowercase SHA-256",
        ));
    }
    if request.expected_duration_ms == 0 || request.expected_duration_ms > MAX_DURATION_MS {
        return Err(oracle_request_error(
            "duration",
            "duration must be within (0, 24 hours]",
        ));
    }
    if !request.audio_path.is_absolute() {
        return Err(oracle_request_error(
            "audio",
            "observation audio path must be canonical and absolute",
        ));
    }
    let canonical = request
        .audio_path
        .canonicalize()
        .map_err(|_| oracle_request_error("audio", "observation audio could not be resolved"))?;
    if canonical != request.audio_path || !canonical.is_file() {
        return Err(oracle_request_error(
            "audio",
            "observation audio path must name one canonical file",
        ));
    }
    token.checkpoint()
}

fn sortformer_observation_provenance_from_success(
    audio_sha256: &str,
    success: &ExternalSuccess,
) -> SortformerObservationProvenance {
    SortformerObservationProvenance {
        protocol_version: DIFFERENTIAL_ORACLE_PROTOCOL_VERSION.to_owned(),
        tool: DifferentialOracleTool::Sortformer,
        family: DifferentialOracleTool::Sortformer.family(),
        authority: DifferentialAuthority::DiagnosticOnly,
        tool_version: Some(success.version.tool_version.clone()),
        adapter_version: Some(success.version.adapter_version.clone()),
        expected_model_contract_sha256: SORTFORMER_ORACLE_CONTRACT_SHA256.to_owned(),
        model_contract_sha256: success.version.model_contract_sha256.clone(),
        model_artifact_sha256: success.version.model_artifact_sha256.clone(),
        model_artifact_bytes: success.version.model_artifact_bytes,
        runtime_fingerprint_sha256: success.version.runtime_fingerprint_sha256.clone(),
        executable_sha256: Some(success.executable_sha256.clone()),
        version_stdout_sha256: Some(success.version_stdout_sha256.clone()),
        oracle_stdout_sha256: Some(success.oracle_stdout_sha256.clone()),
        audio_sha256: audio_sha256.to_owned(),
    }
}

fn sortformer_observation_provenance_from_skip(
    audio_sha256: &str,
    skipped: &ExternalSkip,
) -> SortformerObservationProvenance {
    SortformerObservationProvenance {
        protocol_version: DIFFERENTIAL_ORACLE_PROTOCOL_VERSION.to_owned(),
        tool: DifferentialOracleTool::Sortformer,
        family: DifferentialOracleTool::Sortformer.family(),
        authority: DifferentialAuthority::DiagnosticOnly,
        tool_version: skipped
            .version
            .as_ref()
            .map(|version| version.tool_version.clone()),
        adapter_version: skipped
            .version
            .as_ref()
            .map(|version| version.adapter_version.clone()),
        expected_model_contract_sha256: SORTFORMER_ORACLE_CONTRACT_SHA256.to_owned(),
        model_contract_sha256: skipped
            .version
            .as_ref()
            .and_then(|version| version.model_contract_sha256.clone()),
        model_artifact_sha256: skipped
            .version
            .as_ref()
            .and_then(|version| version.model_artifact_sha256.clone()),
        model_artifact_bytes: skipped
            .version
            .as_ref()
            .and_then(|version| version.model_artifact_bytes),
        runtime_fingerprint_sha256: skipped
            .version
            .as_ref()
            .and_then(|version| version.runtime_fingerprint_sha256.clone()),
        executable_sha256: skipped.executable_sha256.clone(),
        version_stdout_sha256: skipped.version_stdout_sha256.clone(),
        oracle_stdout_sha256: skipped.oracle_stdout_sha256.clone(),
        audio_sha256: audio_sha256.to_owned(),
    }
}

fn run_differential_oracle_with_token(
    request: DifferentialOracleRequest<'_>,
    token: &CancellationToken,
) -> FwResult<DifferentialOracleReport> {
    validate_comparison_config(&request.comparison_config)?;
    if request.hard_timeout.is_zero() || request.hard_timeout > Duration::from_secs(24 * 60 * 60) {
        return Err(oracle_request_error(
            "hard_timeout",
            "hard timeout must be within (0, 24 hours]",
        ));
    }
    let project_root = request
        .project_root
        .canonicalize()
        .map_err(|_| oracle_request_error("project_root", "project root could not be resolved"))?;
    if !project_root.is_dir() {
        return Err(oracle_request_error(
            "project_root",
            "project root must be a directory",
        ));
    }
    let output_path = validate_external_output(&project_root, request.output_path)?;
    let prepared = prepare_inputs(
        &project_root,
        request.audio_path,
        request.native_document_path,
        request.reference_document_path,
        token,
    )?;
    let program = resolve_program(request.tool);
    let report = build_report(
        request.tool,
        &program,
        &prepared,
        request.hard_timeout,
        request.comparison_config,
        token,
    )?;
    write_new_report(&output_path, &report)?;
    Ok(report)
}

fn build_report(
    tool: DifferentialOracleTool,
    program: &ProgramSpec,
    prepared: &PreparedInputs,
    hard_timeout: Duration,
    comparison_config: DifferentialComparisonConfig,
    token: &CancellationToken,
) -> FwResult<DifferentialOracleReport> {
    token.checkpoint()?;
    if tool == DifferentialOracleTool::Sortformer
        && prepared
            .reference
            .as_ref()
            .and_then(|reference| reference.final_projection.as_deref())
            .is_some_and(|turns| {
                turns
                    .iter()
                    .filter_map(|turn| turn.speaker.as_deref())
                    .collect::<BTreeSet<_>>()
                    .len()
                    > SORTFORMER_ORACLE_MAX_SPEAKERS
            })
    {
        let skipped = ExternalSkip::new(
            DifferentialSkipReason::ReferenceModelCapacityExceeded,
            DifferentialExecutionStage::EligibilityValidation,
        );
        let comparison_config_sha256 = canonical_sha256(&comparison_config)?;
        return finalize_report(DifferentialOracleReport {
            schema_version: DIFFERENTIAL_REPORT_SCHEMA.to_owned(),
            comparator_version: DIFFERENTIAL_COMPARATOR_VERSION.to_owned(),
            authority: DifferentialAuthority::DiagnosticOnly,
            native_incorrectness_claim_permitted: false,
            status: DifferentialRunStatus::Skipped,
            skip_reason: Some(skipped.reason),
            failure_stage: Some(skipped.stage),
            provenance: provenance_from_skip(tool, prepared, &skipped),
            comparison_config,
            comparison_config_sha256,
            comparisons: Vec::new(),
            earliest_divergence: None,
            result_sha256: String::new(),
        });
    }
    let limits = ExternalRunLimits::new(hard_timeout, token).with_expected_executable_sha256(
        (tool == DifferentialOracleTool::Sortformer).then_some(SORTFORMER_ORACLE_ADAPTER_SHA256),
    );
    let external = execute_external(
        tool,
        program,
        &prepared.audio_path,
        &prepared.audio_sha256,
        prepared.native.duration_ms,
        &prepared.native.recording_key,
        limits,
    );
    let comparison_config_sha256 = canonical_sha256(&comparison_config)?;
    match external {
        Ok(success) => {
            if success.oracle.recording_key != prepared.native.recording_key
                || success.oracle.duration_ms != prepared.native.duration_ms
            {
                return finalize_report(DifferentialOracleReport {
                    schema_version: DIFFERENTIAL_REPORT_SCHEMA.to_owned(),
                    comparator_version: DIFFERENTIAL_COMPARATOR_VERSION.to_owned(),
                    authority: DifferentialAuthority::DiagnosticOnly,
                    native_incorrectness_claim_permitted: false,
                    status: DifferentialRunStatus::Skipped,
                    skip_reason: Some(DifferentialSkipReason::OracleIdentityMismatch),
                    failure_stage: Some(DifferentialExecutionStage::OracleOutputValidation),
                    provenance: provenance_from(tool, prepared, Some(&success)),
                    comparison_config,
                    comparison_config_sha256,
                    comparisons: Vec::new(),
                    earliest_divergence: None,
                    result_sha256: String::new(),
                });
            }
            let comparisons = compare_documents_with_token(
                &prepared.native,
                &success.oracle,
                prepared.reference.as_ref(),
                &comparison_config,
                token,
            )?;
            let earliest_divergence = comparisons
                .iter()
                .find(|comparison| comparison.state == DifferentialStageState::Divergent)
                .map(|comparison| comparison.stage);
            finalize_report(DifferentialOracleReport {
                schema_version: DIFFERENTIAL_REPORT_SCHEMA.to_owned(),
                comparator_version: DIFFERENTIAL_COMPARATOR_VERSION.to_owned(),
                authority: DifferentialAuthority::DiagnosticOnly,
                native_incorrectness_claim_permitted: false,
                status: DifferentialRunStatus::Completed,
                skip_reason: None,
                failure_stage: None,
                provenance: provenance_from(tool, prepared, Some(&success)),
                comparison_config,
                comparison_config_sha256,
                comparisons,
                earliest_divergence,
                result_sha256: String::new(),
            })
        }
        Err(ExternalRunError::Cancelled(error)) => Err(error),
        Err(ExternalRunError::Skipped(skipped)) => finalize_report(DifferentialOracleReport {
            schema_version: DIFFERENTIAL_REPORT_SCHEMA.to_owned(),
            comparator_version: DIFFERENTIAL_COMPARATOR_VERSION.to_owned(),
            authority: DifferentialAuthority::DiagnosticOnly,
            native_incorrectness_claim_permitted: false,
            status: DifferentialRunStatus::Skipped,
            skip_reason: Some(skipped.reason),
            failure_stage: Some(skipped.stage),
            provenance: provenance_from_skip(tool, prepared, &skipped),
            comparison_config,
            comparison_config_sha256,
            comparisons: Vec::new(),
            earliest_divergence: None,
            result_sha256: String::new(),
        }),
    }
}

fn provenance_from(
    tool: DifferentialOracleTool,
    prepared: &PreparedInputs,
    success: Option<&ExternalSuccess>,
) -> DifferentialProvenance {
    DifferentialProvenance {
        protocol_version: DIFFERENTIAL_ORACLE_PROTOCOL_VERSION.to_owned(),
        tool,
        family: tool.family(),
        tool_version: success.map(|value| value.version.tool_version.clone()),
        adapter_version: success.map(|value| value.version.adapter_version.clone()),
        expected_model_contract_sha256: (tool == DifferentialOracleTool::Sortformer)
            .then(|| SORTFORMER_ORACLE_CONTRACT_SHA256.to_owned()),
        model_contract_sha256: success
            .and_then(|value| value.version.model_contract_sha256.clone()),
        model_artifact_sha256: success
            .and_then(|value| value.version.model_artifact_sha256.clone()),
        model_artifact_bytes: success.and_then(|value| value.version.model_artifact_bytes),
        runtime_fingerprint_sha256: success
            .and_then(|value| value.version.runtime_fingerprint_sha256.clone()),
        executable_sha256: success.map(|value| value.executable_sha256.clone()),
        version_stdout_sha256: success.map(|value| value.version_stdout_sha256.clone()),
        oracle_stdout_sha256: success.map(|value| value.oracle_stdout_sha256.clone()),
        audio_sha256: prepared.audio_sha256.clone(),
        native_document_sha256: prepared.native_sha256.clone(),
        reference_document_sha256: prepared.reference_sha256.clone(),
    }
}

fn provenance_from_skip(
    tool: DifferentialOracleTool,
    prepared: &PreparedInputs,
    skipped: &ExternalSkip,
) -> DifferentialProvenance {
    DifferentialProvenance {
        protocol_version: DIFFERENTIAL_ORACLE_PROTOCOL_VERSION.to_owned(),
        tool,
        family: tool.family(),
        tool_version: skipped
            .version
            .as_ref()
            .map(|value| value.tool_version.clone()),
        adapter_version: skipped
            .version
            .as_ref()
            .map(|value| value.adapter_version.clone()),
        expected_model_contract_sha256: (tool == DifferentialOracleTool::Sortformer)
            .then(|| SORTFORMER_ORACLE_CONTRACT_SHA256.to_owned()),
        model_contract_sha256: skipped
            .version
            .as_ref()
            .and_then(|value| value.model_contract_sha256.clone()),
        model_artifact_sha256: skipped
            .version
            .as_ref()
            .and_then(|value| value.model_artifact_sha256.clone()),
        model_artifact_bytes: skipped
            .version
            .as_ref()
            .and_then(|value| value.model_artifact_bytes),
        runtime_fingerprint_sha256: skipped
            .version
            .as_ref()
            .and_then(|value| value.runtime_fingerprint_sha256.clone()),
        executable_sha256: skipped.executable_sha256.clone(),
        version_stdout_sha256: skipped.version_stdout_sha256.clone(),
        oracle_stdout_sha256: skipped.oracle_stdout_sha256.clone(),
        audio_sha256: prepared.audio_sha256.clone(),
        native_document_sha256: prepared.native_sha256.clone(),
        reference_document_sha256: prepared.reference_sha256.clone(),
    }
}

fn finalize_report(mut report: DifferentialOracleReport) -> FwResult<DifferentialOracleReport> {
    report.result_sha256 = canonical_sha256(&report)?;
    verify_differential_report(&report)?;
    Ok(report)
}

/// Parse and verify one retained path-free differential report.
pub fn parse_differential_report(bytes: &[u8]) -> FwResult<DifferentialOracleReport> {
    let report = serde_json::from_slice(bytes)
        .map_err(|_| oracle_request_error("report_json", "differential report is invalid"))?;
    verify_differential_report(&report)?;
    Ok(report)
}

/// Verify report authority, state invariants, provenance hashes, and self-hash.
pub fn verify_differential_report(report: &DifferentialOracleReport) -> FwResult<()> {
    if report.schema_version != DIFFERENTIAL_REPORT_SCHEMA
        || report.comparator_version != DIFFERENTIAL_COMPARATOR_VERSION
    {
        return Err(oracle_request_error(
            "report_schema",
            "unsupported differential report schema or comparator",
        ));
    }
    if report.authority != DifferentialAuthority::DiagnosticOnly
        || report.native_incorrectness_claim_permitted
    {
        return Err(oracle_request_error(
            "report_authority",
            "differential reports must remain diagnostic-only",
        ));
    }
    validate_comparison_config(&report.comparison_config)?;
    if canonical_sha256(&report.comparison_config)? != report.comparison_config_sha256 {
        return Err(oracle_request_error(
            "comparison_config_hash",
            "comparison configuration hash does not match",
        ));
    }
    if report.provenance.protocol_version != DIFFERENTIAL_ORACLE_PROTOCOL_VERSION
        || report.provenance.family != report.provenance.tool.family()
        || report
            .provenance
            .tool_version
            .as_deref()
            .is_some_and(|value| !is_safe_version_token(value))
        || report
            .provenance
            .adapter_version
            .as_deref()
            .is_some_and(|value| !is_safe_version_token(value))
    {
        return Err(oracle_request_error(
            "report_provenance",
            "protocol or tool-family provenance is inconsistent",
        ));
    }
    let has_version_attestation = report.provenance.tool_version.is_some();
    if has_version_attestation != report.provenance.adapter_version.is_some() {
        return Err(oracle_request_error(
            "report_provenance",
            "tool and adapter version attestations must appear together",
        ));
    }
    if report.provenance.tool == DifferentialOracleTool::Sortformer
        && has_version_attestation
        && report.provenance.executable_sha256.as_deref() != Some(SORTFORMER_ORACLE_ADAPTER_SHA256)
    {
        return Err(oracle_request_error(
            "report_executable_identity",
            "version-attested Sortformer provenance is not bound to the frozen adapter executable",
        ));
    }
    let has_model_attestation = report.provenance.model_contract_sha256.is_some()
        || report.provenance.model_artifact_sha256.is_some()
        || report.provenance.model_artifact_bytes.is_some()
        || report.provenance.runtime_fingerprint_sha256.is_some();
    let model_contract_valid = if report.provenance.tool == DifferentialOracleTool::Sortformer {
        report.provenance.expected_model_contract_sha256.as_deref()
            == Some(SORTFORMER_ORACLE_CONTRACT_SHA256)
            && if has_version_attestation {
                report.provenance.tool_version.as_deref() == Some(SORTFORMER_ORACLE_TOOL_VERSION)
                    && report.provenance.adapter_version.as_deref()
                        == Some(SORTFORMER_ORACLE_ADAPTER_VERSION)
                    && report.provenance.model_contract_sha256.as_deref()
                        == Some(SORTFORMER_ORACLE_CONTRACT_SHA256)
                    && report.provenance.model_artifact_sha256.as_deref()
                        == Some(SORTFORMER_ORACLE_ARTIFACT_SHA256)
                    && report.provenance.model_artifact_bytes
                        == Some(SORTFORMER_ORACLE_ARTIFACT_BYTES)
                    && report
                        .provenance
                        .runtime_fingerprint_sha256
                        .as_deref()
                        .is_some_and(is_sha256_hex)
            } else {
                !has_model_attestation
            }
    } else {
        report.provenance.expected_model_contract_sha256.is_none() && !has_model_attestation
    };
    if !model_contract_valid {
        return Err(oracle_request_error(
            "report_model_contract",
            "model provenance does not match the tool's frozen contract",
        ));
    }
    for hash in [
        Some(report.provenance.audio_sha256.as_str()),
        Some(report.provenance.native_document_sha256.as_str()),
        report.provenance.reference_document_sha256.as_deref(),
        report.provenance.expected_model_contract_sha256.as_deref(),
        report.provenance.model_contract_sha256.as_deref(),
        report.provenance.model_artifact_sha256.as_deref(),
        report.provenance.runtime_fingerprint_sha256.as_deref(),
        report.provenance.executable_sha256.as_deref(),
        report.provenance.version_stdout_sha256.as_deref(),
        report.provenance.oracle_stdout_sha256.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if !is_sha256_hex(hash) {
            return Err(oracle_request_error(
                "report_hash",
                "report provenance contains an invalid SHA-256",
            ));
        }
    }
    match report.status {
        DifferentialRunStatus::Completed => {
            if report.skip_reason.is_some()
                || report.failure_stage.is_some()
                || report.provenance.tool_version.is_none()
                || report.provenance.adapter_version.is_none()
                || report.provenance.executable_sha256.is_none()
                || report.provenance.version_stdout_sha256.is_none()
                || report.provenance.oracle_stdout_sha256.is_none()
            {
                return Err(oracle_request_error(
                    "completed_report",
                    "completed report is missing required provenance",
                ));
            }
            let expected_stages = [
                DifferentialStage::SpeechActivity,
                DifferentialStage::WordTiming,
                DifferentialStage::ChangeBoundaries,
                DifferentialStage::ClusterAssignments,
                DifferentialStage::Overlap,
                DifferentialStage::FinalProjection,
            ];
            if report.comparisons.len() != expected_stages.len()
                || !report
                    .comparisons
                    .iter()
                    .zip(expected_stages)
                    .all(|(comparison, stage)| comparison.stage == stage)
            {
                return Err(oracle_request_error(
                    "comparison_order",
                    "completed report must contain all six ordered comparisons",
                ));
            }
            if report.comparisons.iter().any(|comparison| {
                comparison
                    .diagnostic_loss
                    .is_some_and(|value| !value.is_finite() || value < 0.0)
                    || comparison
                        .native_reference_loss
                        .is_some_and(|value| !value.is_finite() || value < 0.0)
                    || comparison
                        .oracle_reference_loss
                        .is_some_and(|value| !value.is_finite() || value < 0.0)
                    || matches!(
                        comparison.state,
                        DifferentialStageState::Equivalent | DifferentialStageState::Divergent
                    ) != comparison.metric.is_some()
            }) {
                return Err(oracle_request_error(
                    "comparison_metric",
                    "completed report contains an invalid comparison metric",
                ));
            }
            let earliest = report
                .comparisons
                .iter()
                .find(|comparison| comparison.state == DifferentialStageState::Divergent)
                .map(|comparison| comparison.stage);
            if report.earliest_divergence != earliest {
                return Err(oracle_request_error(
                    "earliest_divergence",
                    "earliest divergence does not match ordered comparisons",
                ));
            }
        }
        DifferentialRunStatus::Skipped => {
            if report.skip_reason.is_none()
                || report.failure_stage.is_none()
                || !report.comparisons.is_empty()
                || report.earliest_divergence.is_some()
            {
                return Err(oracle_request_error(
                    "skipped_report",
                    "skipped report state is inconsistent",
                ));
            }
            validate_skipped_report_provenance(report)?;
        }
    }
    if !is_sha256_hex(&report.result_sha256) {
        return Err(oracle_request_error(
            "result_hash",
            "report result hash is invalid",
        ));
    }
    let mut unhashed = report.clone();
    unhashed.result_sha256.clear();
    if canonical_sha256(&unhashed)? != report.result_sha256 {
        return Err(oracle_request_error(
            "result_hash",
            "report result hash does not match",
        ));
    }
    Ok(())
}

fn validate_skipped_report_provenance(report: &DifferentialOracleReport) -> FwResult<()> {
    let reason = report
        .skip_reason
        .ok_or_else(|| oracle_request_error("skipped_report", "skip reason is missing"))?;
    let stage = report
        .failure_stage
        .ok_or_else(|| oracle_request_error("skipped_report", "failure stage is missing"))?;
    let reason_matches_stage = match reason {
        DifferentialSkipReason::MissingExecutable => {
            stage == DifferentialExecutionStage::ResolveExecutable
        }
        DifferentialSkipReason::UnreadableExecutable => matches!(
            stage,
            DifferentialExecutionStage::ResolveExecutable
                | DifferentialExecutionStage::HashExecutable
        ),
        DifferentialSkipReason::ExecutableIdentityMismatch => {
            stage == DifferentialExecutionStage::HashExecutable
        }
        DifferentialSkipReason::InputContractMismatch => {
            stage == DifferentialExecutionStage::InputValidation
        }
        DifferentialSkipReason::InputIdentityMismatch => matches!(
            stage,
            DifferentialExecutionStage::InputValidation
                | DifferentialExecutionStage::InputPostRunValidation
        ),
        DifferentialSkipReason::ReferenceModelCapacityExceeded => {
            stage == DifferentialExecutionStage::EligibilityValidation
        }
        DifferentialSkipReason::VersionProbeFailed
        | DifferentialSkipReason::VersionProbeTimedOut => {
            stage == DifferentialExecutionStage::VersionProbe
        }
        DifferentialSkipReason::InvalidVersionOutput
        | DifferentialSkipReason::ProtocolVersionMismatch
        | DifferentialSkipReason::ToolIdentityMismatch
        | DifferentialSkipReason::ModelContractMismatch => {
            stage == DifferentialExecutionStage::VersionValidation
        }
        DifferentialSkipReason::OracleRunFailed | DifferentialSkipReason::OracleRunTimedOut => {
            stage == DifferentialExecutionStage::OracleRun
        }
        DifferentialSkipReason::InvalidOracleOutput
        | DifferentialSkipReason::ModelCapacityExceeded
        | DifferentialSkipReason::OracleIdentityMismatch => {
            stage == DifferentialExecutionStage::OracleOutputValidation
        }
    };
    if !reason_matches_stage {
        return Err(oracle_request_error(
            "skipped_report",
            "skip reason and failure stage are inconsistent",
        ));
    }

    let provenance = &report.provenance;
    let has_executable = provenance.executable_sha256.is_some();
    let has_version = provenance.tool_version.is_some();
    let has_version_stdout = provenance.version_stdout_sha256.is_some();
    let has_oracle_stdout = provenance.oracle_stdout_sha256.is_some();
    let valid_presence = match stage {
        DifferentialExecutionStage::EligibilityValidation => {
            !has_executable && !has_version && !has_version_stdout && !has_oracle_stdout
        }
        DifferentialExecutionStage::ResolveExecutable => {
            has_executable == (reason == DifferentialSkipReason::UnreadableExecutable)
                && !has_version
                && !has_version_stdout
                && !has_oracle_stdout
        }
        DifferentialExecutionStage::HashExecutable => {
            has_executable == (reason == DifferentialSkipReason::ExecutableIdentityMismatch)
                && !has_version
                && !has_version_stdout
                && !has_oracle_stdout
        }
        DifferentialExecutionStage::InputValidation => {
            has_executable && !has_version && !has_version_stdout && !has_oracle_stdout
        }
        DifferentialExecutionStage::VersionProbe => {
            has_executable && !has_version && !has_version_stdout && !has_oracle_stdout
        }
        DifferentialExecutionStage::VersionValidation => {
            has_executable && !has_version && has_version_stdout && !has_oracle_stdout
        }
        DifferentialExecutionStage::OracleRun => {
            has_executable && has_version && has_version_stdout && !has_oracle_stdout
        }
        DifferentialExecutionStage::InputPostRunValidation
        | DifferentialExecutionStage::OracleOutputValidation => {
            has_executable && has_version && has_version_stdout && has_oracle_stdout
        }
    };
    if !valid_presence {
        return Err(oracle_request_error(
            "skipped_report",
            "failure stage and retained provenance are inconsistent",
        ));
    }
    Ok(())
}

#[derive(Debug)]
enum ExternalRunError {
    Cancelled(FwError),
    Skipped(ExternalSkip),
}

#[derive(Clone, Copy)]
struct ExternalRunLimits<'a> {
    hard_timeout: Duration,
    token: &'a CancellationToken,
    additional_cancel: Option<&'a (dyn Fn() -> bool + Sync)>,
    expected_executable_sha256: Option<&'a str>,
}

impl<'a> ExternalRunLimits<'a> {
    const fn new(hard_timeout: Duration, token: &'a CancellationToken) -> Self {
        Self {
            hard_timeout,
            token,
            additional_cancel: None,
            expected_executable_sha256: None,
        }
    }

    const fn with_additional_cancel(
        mut self,
        additional_cancel: Option<&'a (dyn Fn() -> bool + Sync)>,
    ) -> Self {
        self.additional_cancel = additional_cancel;
        self
    }

    const fn with_expected_executable_sha256(
        mut self,
        expected_executable_sha256: Option<&'a str>,
    ) -> Self {
        self.expected_executable_sha256 = expected_executable_sha256;
        self
    }
}

fn execute_external(
    tool: DifferentialOracleTool,
    program: &ProgramSpec,
    audio_path: &Path,
    expected_audio_sha256: &str,
    expected_duration_ms: u64,
    recording_key: &str,
    limits: ExternalRunLimits<'_>,
) -> Result<ExternalSuccess, ExternalRunError> {
    let ExternalRunLimits {
        hard_timeout,
        token,
        additional_cancel,
        expected_executable_sha256,
    } = limits;
    token.checkpoint().map_err(ExternalRunError::Cancelled)?;
    let executable_path = which::which(&program.program).map_err(|_| {
        ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::MissingExecutable,
            DifferentialExecutionStage::ResolveExecutable,
        ))
    })?;
    let executable_sha256 = hash_file(&executable_path, token)
        .map_err(|error| classify_hash_error(error, DifferentialExecutionStage::HashExecutable))?;
    if expected_executable_sha256.is_some_and(|expected| expected != executable_sha256) {
        return Err(ExternalRunError::Skipped(
            ExternalSkip::new(
                DifferentialSkipReason::ExecutableIdentityMismatch,
                DifferentialExecutionStage::HashExecutable,
            )
            .with_executable(&executable_sha256),
        ));
    }
    let executable_program = executable_path.to_str().ok_or_else(|| {
        ExternalRunError::Skipped(
            ExternalSkip::new(
                DifferentialSkipReason::UnreadableExecutable,
                DifferentialExecutionStage::ResolveExecutable,
            )
            .with_executable(&executable_sha256),
        )
    })?;
    validate_tool_input(tool, audio_path, expected_duration_ms, token)
        .map_err(|error| enrich_skip_with_executable(error, &executable_sha256))?;
    validate_audio_identity(
        audio_path,
        expected_audio_sha256,
        DifferentialExecutionStage::InputValidation,
        token,
    )
    .map_err(|error| enrich_skip_with_executable(error, &executable_sha256))?;

    let mut version_args = program.prefix_args.clone();
    version_args.extend([
        "--franken-whisper-diarization-oracle-version".to_owned(),
        "--protocol".to_owned(),
        DIFFERENTIAL_ORACLE_PROTOCOL_VERSION.to_owned(),
    ]);
    let version_output = run_command_cancellable_with_probe(
        executable_program,
        &version_args,
        None,
        token,
        Some(Duration::from_secs(15)),
        additional_cancel,
    )
    .map_err(|error| {
        enrich_skip_with_executable(
            classify_command_error(error, DifferentialExecutionStage::VersionProbe),
            &executable_sha256,
        )
    })?;
    let version_stdout_sha256 = bytes_sha256(&version_output.stdout);
    let version: OracleVersionDocument =
        serde_json::from_slice(&version_output.stdout).map_err(|_| {
            ExternalRunError::Skipped(
                ExternalSkip::new(
                    DifferentialSkipReason::InvalidVersionOutput,
                    DifferentialExecutionStage::VersionValidation,
                )
                .with_executable(&executable_sha256)
                .with_version_probe(&version_stdout_sha256),
            )
        })?;
    validate_version_document(&version, tool).map_err(|error| {
        enrich_skip_with_version_probe(error, &executable_sha256, &version_stdout_sha256)
    })?;
    validate_executable_identity(&executable_path, &executable_sha256, token)?;

    let audio_text = audio_path.to_str().ok_or_else(|| {
        ExternalRunError::Skipped(
            ExternalSkip::new(
                DifferentialSkipReason::OracleRunFailed,
                DifferentialExecutionStage::OracleRun,
            )
            .with_executable(&executable_sha256)
            .with_valid_version(&version, &version_stdout_sha256),
        )
    })?;
    let mut run_args = program.prefix_args.clone();
    run_args.extend([
        "--franken-whisper-diarization-oracle-run".to_owned(),
        "--protocol".to_owned(),
        DIFFERENTIAL_ORACLE_PROTOCOL_VERSION.to_owned(),
        "--audio".to_owned(),
        audio_text.to_owned(),
        "--recording-key".to_owned(),
        recording_key.to_owned(),
    ]);
    let output = run_command_cancellable_with_probe(
        executable_program,
        &run_args,
        None,
        token,
        Some(hard_timeout),
        additional_cancel,
    )
    .map_err(|error| {
        enrich_skip_with_valid_version(
            classify_command_error(error, DifferentialExecutionStage::OracleRun),
            &executable_sha256,
            &version,
            &version_stdout_sha256,
        )
    })?;
    validate_executable_identity(&executable_path, &executable_sha256, token)?;
    let oracle_stdout_sha256 = bytes_sha256(&output.stdout);
    validate_audio_identity(
        audio_path,
        expected_audio_sha256,
        DifferentialExecutionStage::InputPostRunValidation,
        token,
    )
    .map_err(|error| {
        enrich_skip_with_oracle_stdout(
            enrich_skip_with_valid_version(
                error,
                &executable_sha256,
                &version,
                &version_stdout_sha256,
            ),
            &oracle_stdout_sha256,
        )
    })?;
    let oracle: DifferentialStageDocument =
        serde_json::from_slice(&output.stdout).map_err(|_| {
            ExternalRunError::Skipped(
                ExternalSkip::new(
                    DifferentialSkipReason::InvalidOracleOutput,
                    DifferentialExecutionStage::OracleOutputValidation,
                )
                .with_executable(&executable_sha256)
                .with_valid_version(&version, &version_stdout_sha256)
                .with_oracle_stdout(&oracle_stdout_sha256),
            )
        })?;
    validate_stage_document_with_token(&oracle, token).map_err(|error| {
        if matches!(error, FwError::Cancelled(_)) {
            return ExternalRunError::Cancelled(error);
        }
        ExternalRunError::Skipped(
            ExternalSkip::new(
                DifferentialSkipReason::InvalidOracleOutput,
                DifferentialExecutionStage::OracleOutputValidation,
            )
            .with_executable(&executable_sha256)
            .with_valid_version(&version, &version_stdout_sha256)
            .with_oracle_stdout(&oracle_stdout_sha256),
        )
    })?;
    token.checkpoint().map_err(ExternalRunError::Cancelled)?;
    validate_tool_stage_document(tool, &oracle).map_err(|reason| {
        ExternalRunError::Skipped(
            ExternalSkip::new(reason, DifferentialExecutionStage::OracleOutputValidation)
                .with_executable(&executable_sha256)
                .with_valid_version(&version, &version_stdout_sha256)
                .with_oracle_stdout(&oracle_stdout_sha256),
        )
    })?;
    token.checkpoint().map_err(ExternalRunError::Cancelled)?;

    Ok(ExternalSuccess {
        version,
        executable_sha256,
        version_stdout_sha256,
        oracle_stdout_sha256,
        oracle,
    })
}

fn validate_version_document(
    version: &OracleVersionDocument,
    expected_tool: DifferentialOracleTool,
) -> Result<(), ExternalRunError> {
    if version.schema_version != DIFFERENTIAL_ORACLE_VERSION_SCHEMA {
        return Err(ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::InvalidVersionOutput,
            DifferentialExecutionStage::VersionValidation,
        )));
    }
    if version.protocol_version != DIFFERENTIAL_ORACLE_PROTOCOL_VERSION {
        return Err(ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::ProtocolVersionMismatch,
            DifferentialExecutionStage::VersionValidation,
        )));
    }
    if version.tool != expected_tool {
        return Err(ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::ToolIdentityMismatch,
            DifferentialExecutionStage::VersionValidation,
        )));
    }
    if !is_safe_version_token(&version.tool_version)
        || !is_safe_version_token(&version.adapter_version)
        || version
            .model_contract_sha256
            .as_deref()
            .is_some_and(|value| !is_sha256_hex(value))
        || version
            .model_artifact_sha256
            .as_deref()
            .is_some_and(|value| !is_sha256_hex(value))
        || version
            .runtime_fingerprint_sha256
            .as_deref()
            .is_some_and(|value| !is_sha256_hex(value))
        || version.model_artifact_bytes == Some(0)
    {
        return Err(ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::InvalidVersionOutput,
            DifferentialExecutionStage::VersionValidation,
        )));
    }
    if expected_tool != DifferentialOracleTool::Sortformer
        && (version.model_contract_sha256.is_some()
            || version.model_artifact_sha256.is_some()
            || version.model_artifact_bytes.is_some()
            || version.runtime_fingerprint.is_some()
            || version.runtime_fingerprint_sha256.is_some())
    {
        return Err(ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::InvalidVersionOutput,
            DifferentialExecutionStage::VersionValidation,
        )));
    }
    let runtime_fingerprint_valid =
        version
            .runtime_fingerprint
            .as_ref()
            .is_some_and(|fingerprint| {
                validate_sortformer_runtime_fingerprint(fingerprint)
                    && canonical_sha256(fingerprint).ok().as_deref()
                        == version.runtime_fingerprint_sha256.as_deref()
            });
    if expected_tool == DifferentialOracleTool::Sortformer
        && (version.tool_version != SORTFORMER_ORACLE_TOOL_VERSION
            || version.adapter_version != SORTFORMER_ORACLE_ADAPTER_VERSION
            || version.model_contract_sha256.as_deref() != Some(SORTFORMER_ORACLE_CONTRACT_SHA256)
            || version.model_artifact_sha256.as_deref() != Some(SORTFORMER_ORACLE_ARTIFACT_SHA256)
            || version.model_artifact_bytes != Some(SORTFORMER_ORACLE_ARTIFACT_BYTES)
            || !runtime_fingerprint_valid)
    {
        return Err(ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::ModelContractMismatch,
            DifferentialExecutionStage::VersionValidation,
        )));
    }
    Ok(())
}

fn validate_sortformer_runtime_fingerprint(
    fingerprint: &DifferentialOracleRuntimeFingerprint,
) -> bool {
    let contract = sortformer_oracle_contract();
    let safe_tokens = [
        fingerprint.python_version.as_str(),
        fingerprint.nemo_version.as_str(),
        fingerprint.torch_version.as_str(),
        fingerprint.torchaudio_version.as_str(),
        fingerprint.numpy_version.as_str(),
        fingerprint.blas_backend.as_str(),
        fingerprint.operating_system.as_str(),
        fingerprint.machine_architecture.as_str(),
        fingerprint.cpu_feature_tier.as_str(),
    ];
    fingerprint.schema_version == contract.runtime_fingerprint_schema
        && safe_tokens.into_iter().all(is_safe_version_token)
        && fingerprint.python_version == contract.python_version
        && fingerprint.nemo_version == contract.nemo_version
        && fingerprint.torch_version == contract.torch_version
        && fingerprint.torchaudio_version == contract.torchaudio_version
        && fingerprint.numpy_version == contract.numpy_version
        && fingerprint.device == contract.device
        && fingerprint.compute_dtype == contract.compute_dtype
        && fingerprint.autocast == contract.autocast
        && fingerprint.quantization == contract.quantization
        && fingerprint.torch_intraop_threads == contract.torch_intraop_threads
        && fingerprint.torch_interop_threads == contract.torch_interop_threads
        && fingerprint.data_loader_workers == contract.data_loader_workers
        && fingerprint.deterministic_algorithms == contract.deterministic_algorithms
}

fn validate_tool_input(
    tool: DifferentialOracleTool,
    audio_path: &Path,
    expected_duration_ms: u64,
    token: &CancellationToken,
) -> Result<(), ExternalRunError> {
    if tool != DifferentialOracleTool::Sortformer {
        return Ok(());
    }
    token.checkpoint().map_err(ExternalRunError::Cancelled)?;
    let mut reader = hound::WavReader::open(audio_path).map_err(|_| input_contract_error())?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != 16_000
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(input_contract_error());
    }
    let mut sample_count = 0u64;
    for sample in reader.samples::<i16>() {
        if sample_count.is_multiple_of(64 * 1024) {
            token.checkpoint().map_err(ExternalRunError::Cancelled)?;
        }
        sample.map_err(|_| input_contract_error())?;
        sample_count = sample_count.saturating_add(1);
    }
    token.checkpoint().map_err(ExternalRunError::Cancelled)?;
    let duration_ms = sample_count
        .saturating_mul(1_000)
        .checked_div(u64::from(spec.sample_rate))
        .ok_or_else(input_contract_error)?;
    if duration_ms.abs_diff(expected_duration_ms)
        > u64::from(SORTFORMER_AUDIO_DURATION_TOLERANCE_MS)
    {
        return Err(input_contract_error());
    }
    Ok(())
}

fn input_contract_error() -> ExternalRunError {
    ExternalRunError::Skipped(ExternalSkip::new(
        DifferentialSkipReason::InputContractMismatch,
        DifferentialExecutionStage::InputValidation,
    ))
}

fn validate_audio_identity(
    audio_path: &Path,
    expected_audio_sha256: &str,
    stage: DifferentialExecutionStage,
    token: &CancellationToken,
) -> Result<(), ExternalRunError> {
    let observed = hash_file(audio_path, token).map_err(|error| {
        if matches!(error, FwError::Cancelled(_)) {
            ExternalRunError::Cancelled(error)
        } else {
            ExternalRunError::Skipped(ExternalSkip::new(
                DifferentialSkipReason::InputIdentityMismatch,
                stage,
            ))
        }
    })?;
    if observed != expected_audio_sha256 {
        return Err(ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::InputIdentityMismatch,
            stage,
        )));
    }
    Ok(())
}

fn validate_executable_identity(
    executable_path: &Path,
    expected_sha256: &str,
    token: &CancellationToken,
) -> Result<(), ExternalRunError> {
    let observed = match hash_file(executable_path, token) {
        Ok(observed) => observed,
        Err(error @ FwError::Cancelled(_)) => return Err(ExternalRunError::Cancelled(error)),
        Err(_) => {
            return Err(ExternalRunError::Skipped(ExternalSkip::new(
                DifferentialSkipReason::UnreadableExecutable,
                DifferentialExecutionStage::HashExecutable,
            )));
        }
    };
    if observed != expected_sha256 {
        return Err(ExternalRunError::Skipped(
            ExternalSkip::new(
                DifferentialSkipReason::ExecutableIdentityMismatch,
                DifferentialExecutionStage::HashExecutable,
            )
            .with_executable(&observed),
        ));
    }
    Ok(())
}

fn classify_hash_error(error: FwError, stage: DifferentialExecutionStage) -> ExternalRunError {
    if matches!(error, FwError::Cancelled(_)) {
        ExternalRunError::Cancelled(error)
    } else {
        ExternalRunError::Skipped(ExternalSkip::new(
            DifferentialSkipReason::UnreadableExecutable,
            stage,
        ))
    }
}

fn classify_command_error(error: FwError, stage: DifferentialExecutionStage) -> ExternalRunError {
    match error {
        FwError::Cancelled(_) => ExternalRunError::Cancelled(error),
        FwError::CommandMissing { .. } => ExternalRunError::Skipped(ExternalSkip::new(
            if stage == DifferentialExecutionStage::VersionProbe {
                DifferentialSkipReason::VersionProbeFailed
            } else {
                DifferentialSkipReason::OracleRunFailed
            },
            stage,
        )),
        FwError::CommandTimedOut { .. } => ExternalRunError::Skipped(ExternalSkip::new(
            if stage == DifferentialExecutionStage::VersionProbe {
                DifferentialSkipReason::VersionProbeTimedOut
            } else {
                DifferentialSkipReason::OracleRunTimedOut
            },
            stage,
        )),
        _ => ExternalRunError::Skipped(ExternalSkip::new(
            if stage == DifferentialExecutionStage::VersionProbe {
                DifferentialSkipReason::VersionProbeFailed
            } else {
                DifferentialSkipReason::OracleRunFailed
            },
            stage,
        )),
    }
}

fn enrich_skip_with_executable(
    error: ExternalRunError,
    executable_sha256: &str,
) -> ExternalRunError {
    match error {
        ExternalRunError::Cancelled(_) => error,
        ExternalRunError::Skipped(skipped) => {
            ExternalRunError::Skipped(skipped.with_executable(executable_sha256))
        }
    }
}

fn enrich_skip_with_version_probe(
    error: ExternalRunError,
    executable_sha256: &str,
    version_stdout_sha256: &str,
) -> ExternalRunError {
    match error {
        ExternalRunError::Cancelled(_) => error,
        ExternalRunError::Skipped(skipped) => ExternalRunError::Skipped(
            skipped
                .with_executable(executable_sha256)
                .with_version_probe(version_stdout_sha256),
        ),
    }
}

fn enrich_skip_with_valid_version(
    error: ExternalRunError,
    executable_sha256: &str,
    version: &OracleVersionDocument,
    version_stdout_sha256: &str,
) -> ExternalRunError {
    match error {
        ExternalRunError::Cancelled(_) => error,
        ExternalRunError::Skipped(skipped) => ExternalRunError::Skipped(
            skipped
                .with_executable(executable_sha256)
                .with_valid_version(version, version_stdout_sha256),
        ),
    }
}

fn enrich_skip_with_oracle_stdout(
    error: ExternalRunError,
    oracle_stdout_sha256: &str,
) -> ExternalRunError {
    match error {
        ExternalRunError::Cancelled(_) => error,
        ExternalRunError::Skipped(skipped) => {
            ExternalRunError::Skipped(skipped.with_oracle_stdout(oracle_stdout_sha256))
        }
    }
}

fn resolve_program(tool: DifferentialOracleTool) -> ProgramSpec {
    let program = std::env::var_os(tool.executable_env())
        .and_then(|value| value.into_string().ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| tool.default_program().to_owned());
    ProgramSpec {
        program,
        prefix_args: Vec::new(),
    }
}

fn prepare_inputs(
    project_root: &Path,
    audio_path: &Path,
    native_document_path: &Path,
    reference_document_path: Option<&Path>,
    token: &CancellationToken,
) -> FwResult<PreparedInputs> {
    let audio_path = canonical_external_file(project_root, audio_path, "audio")?;
    let native_path =
        canonical_external_file(project_root, native_document_path, "native_document")?;
    let reference_path = reference_document_path
        .map(|path| canonical_external_file(project_root, path, "reference_document"))
        .transpose()?;
    let native_bytes = read_capped(&native_path, MAX_DOCUMENT_BYTES, "native_document")?;
    let native = parse_stage_document_with_token(&native_bytes, token)?;
    let (reference, reference_sha256) = if let Some(path) = reference_path {
        let bytes = read_capped(&path, MAX_DOCUMENT_BYTES, "reference_document")?;
        let document = parse_stage_document_with_token(&bytes, token)?;
        if document.recording_key != native.recording_key
            || document.duration_ms != native.duration_ms
        {
            return Err(oracle_request_error(
                "reference_identity",
                "reference and native document identities differ",
            ));
        }
        (Some(document), Some(bytes_sha256(&bytes)))
    } else {
        (None, None)
    };
    let audio_sha256 = hash_file(&audio_path, token)?;
    Ok(PreparedInputs {
        audio_path,
        native,
        reference,
        audio_sha256,
        native_sha256: bytes_sha256(&native_bytes),
        reference_sha256,
    })
}

/// Parse and validate one canonical transcript-free stage document.
pub fn parse_stage_document(bytes: &[u8]) -> FwResult<DifferentialStageDocument> {
    parse_stage_document_with_token(bytes, &CancellationToken::unbounded())
}

fn parse_stage_document_with_token(
    bytes: &[u8],
    token: &CancellationToken,
) -> FwResult<DifferentialStageDocument> {
    let document = serde_json::from_slice(bytes)
        .map_err(|_| oracle_request_error("stage_json", "stage document is invalid"))?;
    token.checkpoint()?;
    validate_stage_document_with_token(&document, token)?;
    Ok(document)
}

/// Validate bounded geometry, opaque identities, confidences, and ordering.
pub fn validate_stage_document(document: &DifferentialStageDocument) -> FwResult<()> {
    validate_stage_document_with_token(document, &CancellationToken::unbounded())
}

fn validate_stage_document_with_token(
    document: &DifferentialStageDocument,
    token: &CancellationToken,
) -> FwResult<()> {
    token.checkpoint()?;
    if document.schema_version != DIFFERENTIAL_STAGE_DOCUMENT_SCHEMA {
        return Err(oracle_request_error(
            "stage_schema",
            "unsupported stage document schema version",
        ));
    }
    if !is_sha256_hex(&document.recording_key) {
        return Err(oracle_request_error(
            "recording_key",
            "recording key must be lowercase SHA-256",
        ));
    }
    if document.duration_ms == 0 || document.duration_ms > MAX_DURATION_MS {
        return Err(oracle_request_error(
            "duration",
            "duration must be within (0, 24 hours]",
        ));
    }
    if let Some(intervals) = &document.speech_activity {
        validate_disjoint_intervals(
            intervals,
            document.duration_ms,
            "speech_activity",
            MAX_INTERVALS,
            token,
        )?;
    }
    if let Some(words) = &document.word_timing {
        if words.len() > MAX_WORDS {
            return Err(oracle_request_error(
                "word_count",
                "word timing exceeds the supported count",
            ));
        }
        let mut ids = BTreeSet::new();
        for (index, word) in words.iter().enumerate() {
            if index.is_multiple_of(4_096) {
                token.checkpoint()?;
            }
            validate_geometry(
                word.start_ms,
                word.end_ms,
                document.duration_ms,
                "word_timing",
            )?;
            if !is_opaque_item_id(&word.word_id, "w-") || !ids.insert(word.word_id.as_str()) {
                return Err(oracle_request_error(
                    "word_id",
                    "word IDs must be unique opaque w- identifiers",
                ));
            }
        }
    }
    if let Some(changes) = &document.change_boundaries_ms
        && (changes.len() > MAX_INTERVALS
            || !changes.windows(2).all(|window| {
                window
                    .first()
                    .zip(window.get(1))
                    .is_some_and(|(left, right)| left < right)
            })
            || changes
                .iter()
                .any(|point| *point == 0 || *point >= document.duration_ms))
    {
        return Err(oracle_request_error(
            "change_boundaries",
            "change boundaries must be strictly ordered internal milliseconds",
        ));
    }
    if let Some(assignments) = &document.cluster_assignments {
        if assignments.len() > MAX_CLUSTERS {
            return Err(oracle_request_error(
                "cluster_count",
                "cluster assignments exceed the supported count",
            ));
        }
        let mut ids = BTreeSet::new();
        for (index, assignment) in assignments.iter().enumerate() {
            if index.is_multiple_of(4_096) {
                token.checkpoint()?;
            }
            validate_geometry(
                assignment.start_ms,
                assignment.end_ms,
                document.duration_ms,
                "cluster_assignment",
            )?;
            if !is_opaque_item_id(&assignment.segment_id, "seg-")
                || !ids.insert(assignment.segment_id.as_str())
                || !is_safe_label(&assignment.cluster_label)
                || assignment
                    .confidence
                    .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            {
                return Err(oracle_request_error(
                    "cluster_assignment",
                    "cluster assignments contain an invalid ID, label, or confidence",
                ));
            }
        }
    }
    if let Some(intervals) = &document.overlap {
        validate_disjoint_intervals(
            intervals,
            document.duration_ms,
            "overlap",
            MAX_INTERVALS,
            token,
        )?;
    }
    if let Some(turns) = &document.final_projection {
        if turns.len() > MAX_TURNS {
            return Err(oracle_request_error(
                "turn_count",
                "final projection exceeds the supported turn count",
            ));
        }
        let _ = turns_to_scoring(turns, document.duration_ms, false, token)?;
    }
    token.checkpoint()?;
    Ok(())
}

fn validate_tool_stage_document(
    tool: DifferentialOracleTool,
    document: &DifferentialStageDocument,
) -> Result<(), DifferentialSkipReason> {
    if tool != DifferentialOracleTool::Sortformer {
        return Ok(());
    }

    validate_sortformer_stage_contract(document)
}

/// Validate the complete output boundary for the future native Rust
/// Sortformer engine, including generic geometry and model-specific semantics.
pub(crate) fn validate_sortformer_stage_document(
    document: &DifferentialStageDocument,
) -> FwResult<()> {
    validate_stage_document(document).map_err(|_| {
        FwError::ContractViolation("native_sortformer.invalid_generic_stage_document".to_owned())
    })?;
    validate_sortformer_stage_contract(document).map_err(|reason| {
        let code = match reason {
            DifferentialSkipReason::ModelCapacityExceeded => "model_capacity_exceeded",
            _ => "invalid_model_stage_document",
        };
        FwError::ContractViolation(format!("native_sortformer.{code}"))
    })
}

fn validate_sortformer_stage_contract(
    document: &DifferentialStageDocument,
) -> Result<(), DifferentialSkipReason> {
    let Some(turns) = document.final_projection.as_deref() else {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    };
    if document.speech_activity.is_none()
        || document.change_boundaries_ms.is_none()
        || document.overlap.is_none()
        || document.word_timing.is_some()
        || document.cluster_assignments.is_some()
    {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    }

    let labels = turns
        .iter()
        .filter_map(|turn| turn.speaker.as_deref())
        .collect::<BTreeSet<_>>();
    if labels.len() > SORTFORMER_ORACLE_MAX_SPEAKERS {
        return Err(DifferentialSkipReason::ModelCapacityExceeded);
    }
    if turns
        .iter()
        .any(|turn| turn.speaker.is_none() || turn.overlap_suspected)
    {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    }
    let mut speaker_indices = BTreeSet::new();
    for label in &labels {
        let Some(suffix) = label.strip_prefix("speaker_") else {
            return Err(DifferentialSkipReason::InvalidOracleOutput);
        };
        let Ok(index) = suffix.parse::<usize>() else {
            return Err(DifferentialSkipReason::InvalidOracleOutput);
        };
        if index >= SORTFORMER_ORACLE_MAX_SPEAKERS || *label != format!("speaker_{index}") {
            return Err(DifferentialSkipReason::InvalidOracleOutput);
        }
        speaker_indices.insert(index);
    }
    if !speaker_indices.iter().copied().eq(0..speaker_indices.len()) {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    }
    if !turns.windows(2).all(|window| {
        window
            .first()
            .zip(window.get(1))
            .is_some_and(|(left, right)| {
                (left.start_ms, left.end_ms, left.speaker.as_deref())
                    <= (right.start_ms, right.end_ms, right.speaker.as_deref())
            })
    }) {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    }
    let mut prior_end_by_speaker = BTreeMap::<&str, u64>::new();
    let mut first_onset_by_speaker_index = BTreeMap::<usize, u64>::new();
    for turn in turns {
        let label = turn
            .speaker
            .as_deref()
            .ok_or(DifferentialSkipReason::InvalidOracleOutput)?;
        let output_frame_ms = u64::from(SORTFORMER_ORACLE_OUTPUT_FRAME_MS);
        if turn.start_ms % output_frame_ms != 0
            || (turn.end_ms != document.duration_ms && turn.end_ms % output_frame_ms != 0)
        {
            return Err(DifferentialSkipReason::InvalidOracleOutput);
        }
        let index = label
            .strip_prefix("speaker_")
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or(DifferentialSkipReason::InvalidOracleOutput)?;
        first_onset_by_speaker_index
            .entry(index)
            .and_modify(|onset| *onset = (*onset).min(turn.start_ms))
            .or_insert(turn.start_ms);
        if prior_end_by_speaker
            .insert(label, turn.end_ms)
            .is_some_and(|prior_end| turn.start_ms < prior_end)
        {
            return Err(DifferentialSkipReason::InvalidOracleOutput);
        }
    }
    let mut onsets = first_onset_by_speaker_index.values().copied();
    let mut prior_onset = onsets.next();
    if onsets.any(|onset| {
        let out_of_order = prior_onset.is_some_and(|prior| prior > onset);
        prior_onset = Some(onset);
        out_of_order
    }) {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    }
    if document
        .change_boundaries_ms
        .as_ref()
        .is_some_and(|changes| {
            changes
                .iter()
                .any(|point| point % u64::from(SORTFORMER_ORACLE_OUTPUT_FRAME_MS) != 0)
        })
    {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    }
    let (derived_speech_activity, derived_overlap, derived_change_boundaries) =
        derive_sortformer_stages(turns).ok_or(DifferentialSkipReason::InvalidOracleOutput)?;
    if document.change_boundaries_ms.as_deref() != Some(derived_change_boundaries.as_slice()) {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    }
    if document.speech_activity.as_deref() != Some(derived_speech_activity.as_slice())
        || document.overlap.as_deref() != Some(derived_overlap.as_slice())
    {
        return Err(DifferentialSkipReason::InvalidOracleOutput);
    }
    Ok(())
}

fn derive_sortformer_stages(
    turns: &[EvaluationTurn],
) -> Option<(
    Vec<DifferentialInterval>,
    Vec<DifferentialInterval>,
    Vec<u64>,
)> {
    let mut events = BTreeMap::<u64, BTreeMap<&str, i32>>::new();
    for turn in turns {
        let speaker = turn.speaker.as_deref()?;
        *events
            .entry(turn.start_ms)
            .or_default()
            .entry(speaker)
            .or_default() += 1;
        *events
            .entry(turn.end_ms)
            .or_default()
            .entry(speaker)
            .or_default() -= 1;
    }

    let mut active = BTreeMap::<&str, i32>::new();
    let mut speech = Vec::new();
    let mut overlap = Vec::new();
    let mut change_boundaries = Vec::new();
    let mut previous_ms = None;
    for (point_ms, deltas) in events {
        if let Some(start_ms) = previous_ms
            && start_ms < point_ms
        {
            if !active.is_empty() {
                push_merged_interval(&mut speech, start_ms, point_ms);
            }
            if active.len() >= 2 {
                push_merged_interval(&mut overlap, start_ms, point_ms);
            }
        }
        let active_before = !active.is_empty();
        let mut membership_changed = false;
        for (speaker, delta) in deltas {
            let was_active = active.contains_key(speaker);
            let next = active.get(speaker).copied().unwrap_or(0) + delta;
            match next {
                0 => {
                    active.remove(speaker);
                }
                1 => {
                    active.insert(speaker, next);
                }
                _ => return None,
            }
            membership_changed |= was_active != active.contains_key(speaker);
        }
        if active_before && !active.is_empty() && membership_changed {
            change_boundaries.push(point_ms);
        }
        previous_ms = Some(point_ms);
    }
    active
        .is_empty()
        .then_some((speech, overlap, change_boundaries))
}

fn push_merged_interval(intervals: &mut Vec<DifferentialInterval>, start_ms: u64, end_ms: u64) {
    if let Some(last) = intervals.last_mut()
        && last.end_ms == start_ms
    {
        last.end_ms = end_ms;
        return;
    }
    intervals.push(DifferentialInterval { start_ms, end_ms });
}

fn validate_comparison_config(config: &DifferentialComparisonConfig) -> FwResult<()> {
    if config.schema_version != "differential-comparison-config-v1" {
        return Err(oracle_request_error(
            "comparison_schema",
            "unsupported comparison configuration schema",
        ));
    }
    for (name, value) in [
        ("minimum_interval_iou", config.minimum_interval_iou),
        (
            "minimum_word_timing_recall",
            config.minimum_word_timing_recall,
        ),
        ("minimum_change_f1", config.minimum_change_f1),
        (
            "minimum_cluster_segment_coverage",
            config.minimum_cluster_segment_coverage,
        ),
        (
            "maximum_cluster_pair_disagreement",
            config.maximum_cluster_pair_disagreement,
        ),
        ("maximum_projection_der", config.maximum_projection_der),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(oracle_request_error(
                "comparison_threshold",
                &format!("{name} must be finite and within [0, 1]"),
            ));
        }
    }
    if !config.adjudication_epsilon.is_finite()
        || !(0.0..=1.0).contains(&config.adjudication_epsilon)
    {
        return Err(oracle_request_error(
            "adjudication_epsilon",
            "adjudication epsilon must be finite and within [0, 1]",
        ));
    }
    Ok(())
}

/// Compare two canonical documents, optionally using a third document for diagnostics.
pub fn compare_documents(
    native: &DifferentialStageDocument,
    oracle: &DifferentialStageDocument,
    reference: Option<&DifferentialStageDocument>,
    config: &DifferentialComparisonConfig,
) -> FwResult<Vec<DifferentialStageComparison>> {
    compare_documents_with_token(
        native,
        oracle,
        reference,
        config,
        &CancellationToken::unbounded(),
    )
}

fn compare_documents_with_token(
    native: &DifferentialStageDocument,
    oracle: &DifferentialStageDocument,
    reference: Option<&DifferentialStageDocument>,
    config: &DifferentialComparisonConfig,
    token: &CancellationToken,
) -> FwResult<Vec<DifferentialStageComparison>> {
    validate_stage_document_with_token(native, token)?;
    validate_stage_document_with_token(oracle, token)?;
    validate_comparison_complexity(native, "native")?;
    validate_comparison_complexity(oracle, "oracle")?;
    validate_comparison_config(config)?;
    if native.recording_key != oracle.recording_key || native.duration_ms != oracle.duration_ms {
        return Err(oracle_request_error(
            "comparison_identity",
            "native and oracle document identities differ",
        ));
    }
    if let Some(reference) = reference {
        validate_stage_document_with_token(reference, token)?;
        validate_comparison_complexity(reference, "reference")?;
        if reference.recording_key != native.recording_key
            || reference.duration_ms != native.duration_ms
        {
            return Err(oracle_request_error(
                "reference_identity",
                "reference and compared document identities differ",
            ));
        }
    }

    let mut comparisons = Vec::with_capacity(6);
    token.checkpoint()?;
    comparisons.push(compare_interval_stage(
        DifferentialStage::SpeechActivity,
        native.speech_activity.as_deref(),
        oracle.speech_activity.as_deref(),
        reference.and_then(|value| value.speech_activity.as_deref()),
        reference.is_some(),
        config,
    ));
    token.checkpoint()?;
    comparisons.push(compare_word_stage(
        native.word_timing.as_deref(),
        oracle.word_timing.as_deref(),
        reference.and_then(|value| value.word_timing.as_deref()),
        reference.is_some(),
        config,
    ));
    token.checkpoint()?;
    comparisons.push(compare_change_stage(
        native.change_boundaries_ms.as_deref(),
        oracle.change_boundaries_ms.as_deref(),
        reference.and_then(|value| value.change_boundaries_ms.as_deref()),
        reference.is_some(),
        config,
    )?);
    token.checkpoint()?;
    comparisons.push(compare_cluster_stage(
        native.cluster_assignments.as_deref(),
        oracle.cluster_assignments.as_deref(),
        reference.and_then(|value| value.cluster_assignments.as_deref()),
        reference.is_some(),
        config,
    ));
    token.checkpoint()?;
    comparisons.push(compare_interval_stage(
        DifferentialStage::Overlap,
        native.overlap.as_deref(),
        oracle.overlap.as_deref(),
        reference.and_then(|value| value.overlap.as_deref()),
        reference.is_some(),
        config,
    ));
    token.checkpoint()?;
    comparisons.push(compare_projection_stage(
        native.final_projection.as_deref(),
        oracle.final_projection.as_deref(),
        reference.and_then(|value| value.final_projection.as_deref()),
        native.duration_ms,
        reference.is_some(),
        config,
        token,
    )?);
    token.checkpoint()?;
    Ok(comparisons)
}

fn validate_comparison_complexity(
    document: &DifferentialStageDocument,
    role: &str,
) -> FwResult<()> {
    if document
        .change_boundaries_ms
        .as_ref()
        .is_some_and(|changes| changes.len() > MAX_COMPARISON_CHANGE_POINTS)
    {
        return Err(oracle_request_error(
            "comparison_change_count",
            &format!("{role} change-point count exceeds the safe comparison cap"),
        ));
    }
    if let Some(turns) = &document.final_projection {
        if turns.len() > MAX_COMPARISON_TURNS {
            return Err(oracle_request_error(
                "comparison_turn_count",
                &format!("{role} turn count exceeds the safe comparison cap"),
            ));
        }
        let mut speakers = turns
            .iter()
            .filter_map(|turn| turn.speaker.as_deref())
            .collect::<BTreeSet<_>>()
            .len();
        speakers += usize::from(turns.iter().any(|turn| turn.speaker.is_none()));
        if speakers > MAX_COMPARISON_SPEAKERS {
            return Err(oracle_request_error(
                "comparison_speaker_count",
                &format!("{role} speaker count exceeds the safe comparison cap"),
            ));
        }
    }
    Ok(())
}

fn compare_interval_stage(
    stage: DifferentialStage,
    native: Option<&[DifferentialInterval]>,
    oracle: Option<&[DifferentialInterval]>,
    reference: Option<&[DifferentialInterval]>,
    reference_document_present: bool,
    config: &DifferentialComparisonConfig,
) -> DifferentialStageComparison {
    compare_optional_stage(
        stage,
        native,
        oracle,
        reference,
        reference_document_present,
        config,
        |left, right| {
            let score = score_intervals(left, right);
            let loss = 1.0 - score.iou.unwrap_or(1.0);
            (
                DifferentialStageMetric::Intervals(score),
                loss,
                loss > 1.0 - config.minimum_interval_iou,
            )
        },
    )
}

fn compare_word_stage(
    native: Option<&[DifferentialWordTiming]>,
    oracle: Option<&[DifferentialWordTiming]>,
    reference: Option<&[DifferentialWordTiming]>,
    reference_document_present: bool,
    config: &DifferentialComparisonConfig,
) -> DifferentialStageComparison {
    compare_optional_stage(
        DifferentialStage::WordTiming,
        native,
        oracle,
        reference,
        reference_document_present,
        config,
        |left, right| {
            let score = score_word_timing(left, right, config.word_boundary_collar_ms);
            let loss = 1.0 - score.timing_recall.unwrap_or(1.0);
            (
                DifferentialStageMetric::WordTiming(score),
                loss,
                loss > 1.0 - config.minimum_word_timing_recall,
            )
        },
    )
}

fn compare_change_stage(
    native: Option<&[u64]>,
    oracle: Option<&[u64]>,
    reference: Option<&[u64]>,
    reference_document_present: bool,
    config: &DifferentialComparisonConfig,
) -> FwResult<DifferentialStageComparison> {
    compare_optional_stage_fallible(
        DifferentialStage::ChangeBoundaries,
        native,
        oracle,
        reference,
        reference_document_present,
        config,
        |left, right| {
            let score = score_change_points(
                &left
                    .iter()
                    .map(|value| *value as f64 / 1_000.0)
                    .collect::<Vec<_>>(),
                &right
                    .iter()
                    .map(|value| *value as f64 / 1_000.0)
                    .collect::<Vec<_>>(),
                config.change_boundary_collar_ms as f64 / 1_000.0,
            )?;
            let loss = 1.0
                - score.f1.unwrap_or({
                    if left.is_empty() && right.is_empty() {
                        1.0
                    } else {
                        0.0
                    }
                });
            Ok((
                DifferentialStageMetric::ChangeBoundaries(score),
                loss,
                loss > 1.0 - config.minimum_change_f1,
            ))
        },
    )
}

fn compare_cluster_stage(
    native: Option<&[DifferentialClusterAssignment]>,
    oracle: Option<&[DifferentialClusterAssignment]>,
    reference: Option<&[DifferentialClusterAssignment]>,
    reference_document_present: bool,
    config: &DifferentialComparisonConfig,
) -> DifferentialStageComparison {
    compare_optional_stage(
        DifferentialStage::ClusterAssignments,
        native,
        oracle,
        reference,
        reference_document_present,
        config,
        |left, right| {
            let score = score_clusters(left, right);
            let coverage_loss = 1.0 - score.segment_coverage.unwrap_or(1.0);
            let geometry_loss = score.geometry_disagreement_rate.unwrap_or(0.0);
            let loss = score
                .pair_disagreement_rate
                .unwrap_or(0.0)
                .max(coverage_loss)
                .max(geometry_loss);
            let divergent = score
                .segment_coverage
                .is_some_and(|coverage| coverage < config.minimum_cluster_segment_coverage)
                || score
                    .pair_disagreement_rate
                    .is_some_and(|rate| rate > config.maximum_cluster_pair_disagreement)
                || score.geometry_disagreement_count > 0;
            (
                DifferentialStageMetric::ClusterAssignments(score),
                loss,
                divergent,
            )
        },
    )
}

fn compare_projection_stage(
    native: Option<&[EvaluationTurn]>,
    oracle: Option<&[EvaluationTurn]>,
    reference: Option<&[EvaluationTurn]>,
    duration_ms: u64,
    reference_document_present: bool,
    config: &DifferentialComparisonConfig,
    token: &CancellationToken,
) -> FwResult<DifferentialStageComparison> {
    compare_optional_stage_fallible(
        DifferentialStage::FinalProjection,
        native,
        oracle,
        reference,
        reference_document_present,
        config,
        |left, right| {
            token.checkpoint()?;
            let left_scoring = turns_to_scoring(left, duration_ms, true, token)?;
            let right_scoring = turns_to_scoring(right, duration_ms, true, token)?;
            let score = score_diarization(&left_scoring, &right_scoring)?;
            token.checkpoint()?;
            let native_unknown = merged_turn_intervals(left, true);
            let oracle_unknown = merged_turn_intervals(right, true);
            let unknown_score = score_intervals(&native_unknown, &oracle_unknown);
            let unknown_status_disagreement_ms = unknown_score
                .native_only_ms
                .saturating_add(unknown_score.oracle_only_ms);
            let all_speech = merged_turn_pair_intervals(left, right);
            let speech_duration_ms = all_speech
                .iter()
                .map(|interval| interval.end_ms - interval.start_ms)
                .sum::<u64>();
            let unknown_loss = (speech_duration_ms > 0)
                .then_some(unknown_status_disagreement_ms as f64 / speech_duration_ms as f64);
            let loss = score.der.unwrap_or(0.0).max(unknown_loss.unwrap_or(0.0));
            Ok((
                DifferentialStageMetric::FinalProjection(projection_score(
                    &score,
                    &unknown_score,
                    unknown_status_disagreement_ms,
                )),
                loss,
                loss > config.maximum_projection_der,
            ))
        },
    )
}

fn compare_optional_stage<T, F>(
    stage: DifferentialStage,
    native: Option<&[T]>,
    oracle: Option<&[T]>,
    reference: Option<&[T]>,
    reference_document_present: bool,
    config: &DifferentialComparisonConfig,
    mut scorer: F,
) -> DifferentialStageComparison
where
    F: FnMut(&[T], &[T]) -> (DifferentialStageMetric, f64, bool),
{
    match (native, oracle) {
        (None, None) => unavailable_comparison(stage, DifferentialStageState::MissingBoth),
        (None, Some(_)) => unavailable_comparison(stage, DifferentialStageState::MissingNative),
        (Some(_), None) => unavailable_comparison(stage, DifferentialStageState::MissingOracle),
        (Some(native), Some(oracle)) => {
            let (metric, loss, divergent) = scorer(native, oracle);
            let (adjudication, native_reference_loss, oracle_reference_loss) = adjudicate(
                reference_stage(reference, reference_document_present),
                native,
                oracle,
                divergent,
                config,
                &mut scorer,
            );
            DifferentialStageComparison {
                stage,
                state: if divergent {
                    DifferentialStageState::Divergent
                } else {
                    DifferentialStageState::Equivalent
                },
                metric: Some(metric),
                diagnostic_loss: Some(loss),
                adjudication,
                native_reference_loss,
                oracle_reference_loss,
            }
        }
    }
}

fn compare_optional_stage_fallible<T, F>(
    stage: DifferentialStage,
    native: Option<&[T]>,
    oracle: Option<&[T]>,
    reference: Option<&[T]>,
    reference_document_present: bool,
    config: &DifferentialComparisonConfig,
    mut scorer: F,
) -> FwResult<DifferentialStageComparison>
where
    F: FnMut(&[T], &[T]) -> FwResult<(DifferentialStageMetric, f64, bool)>,
{
    match (native, oracle) {
        (None, None) => Ok(unavailable_comparison(
            stage,
            DifferentialStageState::MissingBoth,
        )),
        (None, Some(_)) => Ok(unavailable_comparison(
            stage,
            DifferentialStageState::MissingNative,
        )),
        (Some(_), None) => Ok(unavailable_comparison(
            stage,
            DifferentialStageState::MissingOracle,
        )),
        (Some(native), Some(oracle)) => {
            let (metric, loss, divergent) = scorer(native, oracle)?;
            let (adjudication, native_reference_loss, oracle_reference_loss) = if !divergent {
                (DifferentialAdjudication::NoDisagreement, None, None)
            } else {
                match reference_stage(reference, reference_document_present) {
                    ReferenceStage::Present(reference) => {
                        let (_, native_loss, _) = scorer(reference, native)?;
                        let (_, oracle_loss, _) = scorer(reference, oracle)?;
                        (
                            adjudication_from_losses(
                                native_loss,
                                oracle_loss,
                                config.adjudication_epsilon,
                            ),
                            Some(native_loss),
                            Some(oracle_loss),
                        )
                    }
                    ReferenceStage::Missing => (
                        DifferentialAdjudication::ReferenceStageUnavailable,
                        None,
                        None,
                    ),
                    ReferenceStage::NoDocument => (
                        DifferentialAdjudication::InconclusiveNoReference,
                        None,
                        None,
                    ),
                }
            };
            Ok(DifferentialStageComparison {
                stage,
                state: if divergent {
                    DifferentialStageState::Divergent
                } else {
                    DifferentialStageState::Equivalent
                },
                metric: Some(metric),
                diagnostic_loss: Some(loss),
                adjudication,
                native_reference_loss,
                oracle_reference_loss,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ReferenceStage<'a, T> {
    NoDocument,
    Missing,
    Present(&'a [T]),
}

fn reference_stage<T>(
    reference: Option<&[T]>,
    reference_document_present: bool,
) -> ReferenceStage<'_, T> {
    match (reference_document_present, reference) {
        (_, Some(reference)) => ReferenceStage::Present(reference),
        (true, None) => ReferenceStage::Missing,
        (false, None) => ReferenceStage::NoDocument,
    }
}

fn adjudicate<T, F>(
    reference: ReferenceStage<'_, T>,
    native: &[T],
    oracle: &[T],
    divergent: bool,
    config: &DifferentialComparisonConfig,
    scorer: &mut F,
) -> (DifferentialAdjudication, Option<f64>, Option<f64>)
where
    F: FnMut(&[T], &[T]) -> (DifferentialStageMetric, f64, bool),
{
    if !divergent {
        return (DifferentialAdjudication::NoDisagreement, None, None);
    }
    let reference = match reference {
        ReferenceStage::Present(reference) => reference,
        ReferenceStage::Missing => {
            return (
                DifferentialAdjudication::ReferenceStageUnavailable,
                None,
                None,
            );
        }
        ReferenceStage::NoDocument => {
            return (
                DifferentialAdjudication::InconclusiveNoReference,
                None,
                None,
            );
        }
    };
    let (_, native_loss, _) = scorer(reference, native);
    let (_, oracle_loss, _) = scorer(reference, oracle);
    (
        adjudication_from_losses(native_loss, oracle_loss, config.adjudication_epsilon),
        Some(native_loss),
        Some(oracle_loss),
    )
}

fn adjudication_from_losses(
    native_loss: f64,
    oracle_loss: f64,
    epsilon: f64,
) -> DifferentialAdjudication {
    if (native_loss - oracle_loss).abs() <= epsilon {
        DifferentialAdjudication::ReferenceTied
    } else if native_loss < oracle_loss {
        DifferentialAdjudication::ReferenceFavorsNative
    } else {
        DifferentialAdjudication::ReferenceFavorsOracle
    }
}

fn unavailable_comparison(
    stage: DifferentialStage,
    state: DifferentialStageState,
) -> DifferentialStageComparison {
    DifferentialStageComparison {
        stage,
        state,
        metric: None,
        diagnostic_loss: None,
        adjudication: DifferentialAdjudication::ReferenceStageUnavailable,
        native_reference_loss: None,
        oracle_reference_loss: None,
    }
}

fn score_intervals(
    native: &[DifferentialInterval],
    oracle: &[DifferentialInterval],
) -> DifferentialIntervalScore {
    let native_duration_ms = native
        .iter()
        .map(|interval| interval.end_ms - interval.start_ms)
        .sum::<u64>();
    let oracle_duration_ms = oracle
        .iter()
        .map(|interval| interval.end_ms - interval.start_ms)
        .sum::<u64>();
    let mut intersection_ms = 0u64;
    let mut native_index = 0usize;
    let mut oracle_index = 0usize;
    while let (Some(native_interval), Some(oracle_interval)) = (
        native.get(native_index).copied(),
        oracle.get(oracle_index).copied(),
    ) {
        let overlap_start = native_interval.start_ms.max(oracle_interval.start_ms);
        let overlap_end = native_interval.end_ms.min(oracle_interval.end_ms);
        intersection_ms += overlap_end.saturating_sub(overlap_start);
        if native_interval.end_ms <= oracle_interval.end_ms {
            native_index += 1;
        }
        if oracle_interval.end_ms <= native_interval.end_ms {
            oracle_index += 1;
        }
    }
    let union_ms = native_duration_ms
        .saturating_add(oracle_duration_ms)
        .saturating_sub(intersection_ms);
    DifferentialIntervalScore {
        native_duration_ms,
        oracle_duration_ms,
        intersection_ms,
        union_ms,
        native_only_ms: native_duration_ms.saturating_sub(intersection_ms),
        oracle_only_ms: oracle_duration_ms.saturating_sub(intersection_ms),
        iou: (union_ms > 0).then_some(intersection_ms as f64 / union_ms as f64),
    }
}

fn score_word_timing(
    native: &[DifferentialWordTiming],
    oracle: &[DifferentialWordTiming],
    collar_ms: u64,
) -> DifferentialWordTimingScore {
    let oracle_by_id = oracle
        .iter()
        .map(|word| (word.word_id.as_str(), word))
        .collect::<BTreeMap<_, _>>();
    let mut shared_word_count = 0usize;
    let mut within_collar_count = 0usize;
    let mut boundary_error = 0u128;
    for word in native {
        let Some(other) = oracle_by_id.get(word.word_id.as_str()) else {
            continue;
        };
        shared_word_count += 1;
        let start_error = word.start_ms.abs_diff(other.start_ms);
        let end_error = word.end_ms.abs_diff(other.end_ms);
        boundary_error += u128::from(start_error) + u128::from(end_error);
        within_collar_count += usize::from(start_error <= collar_ms && end_error <= collar_ms);
    }
    let denominator = native.len().max(oracle.len());
    DifferentialWordTimingScore {
        native_word_count: native.len(),
        oracle_word_count: oracle.len(),
        shared_word_count,
        within_collar_count,
        timing_recall: (denominator > 0).then_some(within_collar_count as f64 / denominator as f64),
        mean_absolute_boundary_error_ms: (shared_word_count > 0)
            .then_some(boundary_error as f64 / (2 * shared_word_count) as f64),
    }
}

fn score_clusters(
    native: &[DifferentialClusterAssignment],
    oracle: &[DifferentialClusterAssignment],
) -> DifferentialClusterScore {
    let oracle_by_id = oracle
        .iter()
        .map(|assignment| (assignment.segment_id.as_str(), assignment))
        .collect::<BTreeMap<_, _>>();
    let mut contingency = BTreeMap::<(&str, &str), u64>::new();
    let mut native_sizes = BTreeMap::<&str, u64>::new();
    let mut oracle_sizes = BTreeMap::<&str, u64>::new();
    let mut shared = 0u64;
    let mut geometry_disagreement_count = 0usize;
    let mut shared_confidence_count = 0usize;
    let mut confidence_availability_disagreement_count = 0usize;
    let mut confidence_delta_sum = 0.0f64;
    for assignment in native {
        let Some(other) = oracle_by_id.get(assignment.segment_id.as_str()) else {
            continue;
        };
        shared += 1;
        *contingency
            .entry((&assignment.cluster_label, &other.cluster_label))
            .or_default() += 1;
        *native_sizes
            .entry(assignment.cluster_label.as_str())
            .or_default() += 1;
        *oracle_sizes
            .entry(other.cluster_label.as_str())
            .or_default() += 1;
        geometry_disagreement_count +=
            usize::from(assignment.start_ms != other.start_ms || assignment.end_ms != other.end_ms);
        match (assignment.confidence, other.confidence) {
            (Some(left), Some(right)) => {
                shared_confidence_count += 1;
                confidence_delta_sum += (left - right).abs();
            }
            (Some(_), None) | (None, Some(_)) => {
                confidence_availability_disagreement_count += 1;
            }
            (None, None) => {}
        }
    }
    let true_positive = contingency.values().copied().map(choose_two).sum::<u64>();
    let native_positive = native_sizes.values().copied().map(choose_two).sum::<u64>();
    let oracle_positive = oracle_sizes.values().copied().map(choose_two).sum::<u64>();
    let false_negative = native_positive.saturating_sub(true_positive);
    let false_positive = oracle_positive.saturating_sub(true_positive);
    let compared_pair_count = choose_two(shared);
    let disagreement = false_negative.saturating_add(false_positive);
    let maximum_segment_count = native.len().max(oracle.len());
    let precision = (oracle_positive > 0).then_some(true_positive as f64 / oracle_positive as f64);
    let recall = (native_positive > 0).then_some(true_positive as f64 / native_positive as f64);
    let f1 = match (precision, recall) {
        (Some(precision), Some(recall)) if precision + recall > 0.0 => {
            Some(2.0 * precision * recall / (precision + recall))
        }
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    DifferentialClusterScore {
        native_segment_count: native.len(),
        oracle_segment_count: oracle.len(),
        shared_segment_count: usize::try_from(shared).unwrap_or(usize::MAX),
        segment_coverage: (maximum_segment_count > 0)
            .then_some(shared as f64 / maximum_segment_count as f64),
        geometry_disagreement_count,
        geometry_disagreement_rate: (shared > 0)
            .then_some(geometry_disagreement_count as f64 / shared as f64),
        compared_pair_count,
        pair_disagreement_count: disagreement,
        pair_disagreement_rate: (compared_pair_count > 0)
            .then_some(disagreement as f64 / compared_pair_count as f64),
        coassignment_precision: precision,
        coassignment_recall: recall,
        coassignment_f1: f1,
        shared_confidence_count,
        confidence_availability_disagreement_count,
        mean_absolute_confidence_delta: (shared_confidence_count > 0)
            .then_some(confidence_delta_sum / shared_confidence_count as f64),
    }
}

const fn choose_two(value: u64) -> u64 {
    value.saturating_mul(value.saturating_sub(1)) / 2
}

fn projection_score(
    score: &DiarizationScore,
    unknown_score: &DifferentialIntervalScore,
    unknown_status_disagreement_ms: u64,
) -> DifferentialProjectionScore {
    DifferentialProjectionScore {
        native_speaker_time_sec: score.reference_speaker_time_sec,
        missed_speech_sec: score.missed_speech_sec,
        false_alarm_sec: score.false_alarm_sec,
        speaker_confusion_sec: score.speaker_confusion_sec,
        native_unknown_ms: unknown_score.native_duration_ms,
        oracle_unknown_ms: unknown_score.oracle_duration_ms,
        unknown_status_disagreement_ms,
        der: score.der,
        jer: score.jer,
        mapping_cardinality: score.speaker_mapping.len(),
    }
}

fn turns_to_scoring(
    turns: &[EvaluationTurn],
    duration_ms: u64,
    materialize_unknown: bool,
    token: &CancellationToken,
) -> FwResult<Vec<ScoringTurn>> {
    turns
        .iter()
        .enumerate()
        .map(|(index, turn)| {
            if index.is_multiple_of(4_096) {
                token.checkpoint()?;
            }
            validate_geometry(turn.start_ms, turn.end_ms, duration_ms, "final_projection")?;
            if let Some(label) = &turn.speaker
                && !is_safe_label(label)
            {
                return Err(oracle_request_error(
                    "speaker_label",
                    "projection speaker labels must be opaque safe tokens",
                ));
            }
            if turn
                .speaker_confidence
                .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            {
                return Err(oracle_request_error(
                    "speaker_confidence",
                    "projection confidence must be finite and within [0, 1]",
                ));
            }
            Ok(ScoringTurn {
                start_sec: turn.start_ms as f64 / 1_000.0,
                end_sec: turn.end_ms as f64 / 1_000.0,
                speaker: turn
                    .speaker
                    .clone()
                    .or_else(|| materialize_unknown.then(|| "__fw_unknown".to_owned())),
                overlap_suspected: turn.overlap_suspected,
            })
        })
        .collect()
}

fn merged_turn_intervals(
    turns: &[EvaluationTurn],
    unknown_only: bool,
) -> Vec<DifferentialInterval> {
    merge_intervals(
        turns
            .iter()
            .filter(|turn| !unknown_only || turn.speaker.is_none())
            .map(|turn| DifferentialInterval {
                start_ms: turn.start_ms,
                end_ms: turn.end_ms,
            })
            .collect(),
    )
}

fn merged_turn_pair_intervals(
    left: &[EvaluationTurn],
    right: &[EvaluationTurn],
) -> Vec<DifferentialInterval> {
    merge_intervals(
        left.iter()
            .chain(right)
            .map(|turn| DifferentialInterval {
                start_ms: turn.start_ms,
                end_ms: turn.end_ms,
            })
            .collect(),
    )
}

fn merge_intervals(mut intervals: Vec<DifferentialInterval>) -> Vec<DifferentialInterval> {
    intervals.sort_by_key(|interval| (interval.start_ms, interval.end_ms));
    let mut merged = Vec::<DifferentialInterval>::with_capacity(intervals.len());
    for interval in intervals {
        if let Some(previous) = merged.last_mut()
            && interval.start_ms <= previous.end_ms
        {
            previous.end_ms = previous.end_ms.max(interval.end_ms);
        } else {
            merged.push(interval);
        }
    }
    merged
}

fn validate_disjoint_intervals(
    intervals: &[DifferentialInterval],
    duration_ms: u64,
    field: &str,
    maximum: usize,
    token: &CancellationToken,
) -> FwResult<()> {
    if intervals.len() > maximum {
        return Err(oracle_request_error(
            "interval_count",
            &format!("{field} exceeds the supported interval count"),
        ));
    }
    let mut prior_end = 0u64;
    for (index, interval) in intervals.iter().enumerate() {
        if index.is_multiple_of(4_096) {
            token.checkpoint()?;
        }
        validate_geometry(interval.start_ms, interval.end_ms, duration_ms, field)?;
        if index > 0 && interval.start_ms < prior_end {
            return Err(oracle_request_error(
                "interval_order",
                &format!("{field} must be ordered and non-overlapping"),
            ));
        }
        prior_end = interval.end_ms;
    }
    Ok(())
}

fn validate_geometry(start_ms: u64, end_ms: u64, duration_ms: u64, field: &str) -> FwResult<()> {
    if start_ms >= end_ms || end_ms > duration_ms {
        return Err(oracle_request_error(
            "interval_geometry",
            &format!("{field} contains invalid interval geometry"),
        ));
    }
    Ok(())
}

fn is_opaque_item_id(value: &str, prefix: &str) -> bool {
    let Some(suffix) = value.strip_prefix(prefix) else {
        return false;
    };
    !suffix.is_empty()
        && suffix.len() <= 32
        && suffix
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_safe_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_SAFE_TOKEN_LEN
        && !value.starts_with("__fw_")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'+'))
}

fn is_safe_version_token(value: &str) -> bool {
    is_safe_label(value)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == HASH_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_external_file(project_root: &Path, path: &Path, field: &str) -> FwResult<PathBuf> {
    if !path.is_absolute() {
        return Err(oracle_request_error(
            field,
            "external input paths must be absolute",
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|_| oracle_request_error(field, "external input could not be resolved"))?;
    if canonical.starts_with(project_root) || !canonical.is_file() {
        return Err(oracle_request_error(
            field,
            "external input must be a file outside the project tree",
        ));
    }
    Ok(canonical)
}

fn validate_external_output(project_root: &Path, path: &Path) -> FwResult<PathBuf> {
    if !path.is_absolute() || path.exists() {
        return Err(oracle_request_error(
            "output",
            "output must be a new absolute path outside the project tree",
        ));
    }
    let parent = path.parent().ok_or_else(|| {
        oracle_request_error("output", "output path must have a parent directory")
    })?;
    let parent = parent
        .canonicalize()
        .map_err(|_| oracle_request_error("output", "output parent could not be resolved"))?;
    if parent.starts_with(project_root) || !parent.is_dir() {
        return Err(oracle_request_error(
            "output",
            "output parent must be a directory outside the project tree",
        ));
    }
    let filename = path
        .file_name()
        .ok_or_else(|| oracle_request_error("output", "output path must have a filename"))?;
    Ok(parent.join(filename))
}

fn read_capped(path: &Path, limit: u64, field: &str) -> FwResult<Vec<u8>> {
    let file = File::open(path)
        .map_err(|_| oracle_request_error(field, "external document could not be opened"))?;
    if file
        .metadata()
        .map_err(|_| oracle_request_error(field, "external document metadata could not be read"))?
        .len()
        > limit
    {
        return Err(oracle_request_error(
            field,
            "external document exceeds the safety limit",
        ));
    }
    let mut bytes = Vec::new();
    BufReader::new(file)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| oracle_request_error(field, "external document could not be read"))?;
    if u64::try_from(bytes.len()).map_or(true, |length| length > limit) {
        return Err(oracle_request_error(
            field,
            "external document exceeds the safety limit",
        ));
    }
    Ok(bytes)
}

fn hash_file(path: &Path, token: &CancellationToken) -> FwResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        token.checkpoint()?;
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(hasher.finalize()))
}

fn bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize())
}

fn canonical_sha256<T: Serialize>(value: &T) -> FwResult<String> {
    Ok(bytes_sha256(&canonical_json_bytes(value)?))
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> FwResult<Vec<u8>> {
    let value = serde_json::to_value(value)?;
    let mut output = Vec::new();
    write_lexicographic_json(&value, &mut output)?;
    Ok(output)
}

fn write_lexicographic_json(value: &serde_json::Value, output: &mut Vec<u8>) -> FwResult<()> {
    match value {
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => serde_json::to_writer(output, value)?,
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_lexicographic_json(item, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(values) => {
            output.push(b'{');
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| key.as_str());
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)?;
                output.push(b':');
                write_lexicographic_json(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn write_new_report(path: &Path, report: &DifferentialOracleReport) -> FwResult<()> {
    let bytes = serde_json::to_vec_pretty(report)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| oracle_request_error("output", "new output file could not be created"))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(&bytes)
        .map_err(|_| oracle_request_error("output", "output report could not be written"))?;
    writer
        .flush()
        .map_err(|_| oracle_request_error("output", "output report could not be flushed"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|_| oracle_request_error("output", "output report could not be synchronized"))?;
    Ok(())
}

fn oracle_request_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("differential_oracle.{code}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn interval(start_ms: u64, end_ms: u64) -> DifferentialInterval {
        DifferentialInterval { start_ms, end_ms }
    }

    fn word(id: &str, start_ms: u64, end_ms: u64) -> DifferentialWordTiming {
        DifferentialWordTiming {
            word_id: id.to_owned(),
            start_ms,
            end_ms,
        }
    }

    fn cluster(id: &str, label: &str) -> DifferentialClusterAssignment {
        DifferentialClusterAssignment {
            segment_id: id.to_owned(),
            start_ms: 0,
            end_ms: 100,
            cluster_label: label.to_owned(),
            confidence: Some(0.9),
        }
    }

    fn document() -> DifferentialStageDocument {
        DifferentialStageDocument {
            schema_version: DIFFERENTIAL_STAGE_DOCUMENT_SCHEMA.to_owned(),
            recording_key: KEY.to_owned(),
            duration_ms: 2_000,
            speech_activity: Some(vec![interval(100, 1_900)]),
            word_timing: Some(vec![word("w-01", 100, 300), word("w-02", 400, 600)]),
            change_boundaries_ms: Some(vec![1_000]),
            cluster_assignments: Some(vec![
                cluster("seg-01", "a"),
                cluster("seg-02", "a"),
                cluster("seg-03", "b"),
            ]),
            overlap: Some(vec![interval(900, 1_100)]),
            final_projection: Some(vec![
                EvaluationTurn::labeled(100, 1_000, "a"),
                EvaluationTurn::labeled(1_000, 1_900, "b"),
            ]),
        }
    }

    #[test]
    fn registry_covers_independent_architecture_families() {
        let registry = differential_oracle_registry();
        assert_eq!(registry.len(), 6);
        assert!(registry.iter().all(|entry| {
            entry.authority == DifferentialAuthority::DiagnosticOnly
                && entry.protocol_version == DIFFERENTIAL_ORACLE_PROTOCOL_VERSION
        }));
        assert_eq!(
            registry
                .iter()
                .map(|entry| entry.family)
                .collect::<BTreeSet<_>>()
                .len(),
            3
        );
        assert!(registry.iter().all(|entry| {
            (entry.tool == DifferentialOracleTool::Sortformer) == entry.model_contract.is_some()
                && (entry.tool == DifferentialOracleTool::Sortformer)
                    == entry.model_contract_sha256.is_some()
        }));
    }

    #[test]
    fn sortformer_registry_pins_canonical_model_contract() {
        let contract = sortformer_oracle_contract();
        assert_eq!(
            canonical_sha256(&contract).expect("contract hash"),
            SORTFORMER_ORACLE_CONTRACT_SHA256
        );
        assert_eq!(contract.model_id, SORTFORMER_ORACLE_MODEL_ID);
        assert_eq!(contract.model_revision, SORTFORMER_ORACLE_MODEL_REVISION);
        assert_eq!(contract.maximum_speakers, 4);
        assert_eq!(contract.output_frame_ms, 80);
        assert_eq!(contract.device, "cpu");
        assert_eq!(contract.compute_dtype, "float32");
        assert!(!contract.autocast);
        assert_eq!(contract.quantization, "none");
        assert_eq!(
            contract.canonical_json_version,
            DIFFERENTIAL_CANONICAL_JSON_VERSION
        );
        assert_eq!(
            contract.upstream_artifact_sha256,
            SORTFORMER_ORACLE_ARTIFACT_SHA256
        );
        assert_eq!(contract.python_version, SORTFORMER_ORACLE_PYTHON_VERSION);
        assert_eq!(contract.nemo_version, SORTFORMER_ORACLE_NEMO_VERSION);
        assert_eq!(
            contract.nemo_source_revision,
            SORTFORMER_ORACLE_NEMO_SOURCE_REVISION
        );
        assert_eq!(contract.torch_version, SORTFORMER_ORACLE_TORCH_VERSION);
        assert_eq!(
            contract.torchaudio_version,
            SORTFORMER_ORACLE_TORCHAUDIO_VERSION
        );
        assert_eq!(contract.numpy_version, SORTFORMER_ORACLE_NUMPY_VERSION);
        assert_eq!(contract.speaker_cache_update_period_frames, 300);
        assert_eq!(contract.first_full_chunk_cache_pop_frames, 300);
        assert_eq!(contract.steady_full_chunk_cache_pop_frames, 340);
        assert_eq!(contract.left_context_frames, 1);
        assert_eq!(contract.speaker_cache_silence_frames_per_speaker, 3);
        assert_eq!(contract.subsampling_factor, 8);
        assert_eq!(contract.nominal_input_buffer_latency_ms, 30_400);
    }

    #[test]
    fn canonical_json_is_compact_and_lexicographically_keyed() {
        let left: serde_json::Value =
            serde_json::from_str(r#"{"z":[{"b":2,"a":1}],"a":"value"}"#).expect("left JSON");
        let right: serde_json::Value =
            serde_json::from_str(r#"{"a":"value","z":[{"a":1,"b":2}]}"#).expect("right JSON");
        let expected = br#"{"a":"value","z":[{"a":1,"b":2}]}"#;
        assert_eq!(
            canonical_json_bytes(&left).expect("canonical left"),
            expected
        );
        assert_eq!(
            canonical_json_bytes(&right).expect("canonical right"),
            expected
        );
        assert_eq!(
            canonical_sha256(&left).expect("left hash"),
            canonical_sha256(&right).expect("right hash")
        );
    }

    #[test]
    fn parser_rejects_unknown_fields_and_lexical_word_ids() {
        let mut value = serde_json::to_value(document()).expect("serialize");
        value["unexpected"] = serde_json::json!(true);
        assert!(parse_stage_document(&serde_json::to_vec(&value).expect("json")).is_err());

        let mut lexical = document();
        lexical.word_timing.as_mut().expect("words")[0].word_id = "hello".to_owned();
        assert!(validate_stage_document(&lexical).is_err());
    }

    #[test]
    fn sortformer_stage_contract_rejects_nonfinite_unknown_and_over_capacity_output() {
        let mut nonfinite = sortformer_document();
        nonfinite.final_projection.as_mut().expect("projection")[0].speaker_confidence =
            Some(f64::NAN);
        assert!(validate_stage_document(&nonfinite).is_err());
        assert!(matches!(
            validate_sortformer_stage_document(&nonfinite),
            Err(FwError::ContractViolation(message))
                if message == "native_sortformer.invalid_generic_stage_document"
        ));

        let mut unknown = sortformer_document();
        unknown.final_projection.as_mut().expect("projection")[0] =
            EvaluationTurn::unknown(100, 1_000);
        assert_eq!(
            validate_sortformer_stage_contract(&unknown),
            Err(DifferentialSkipReason::InvalidOracleOutput)
        );
        assert!(matches!(
            validate_sortformer_stage_document(&unknown),
            Err(FwError::ContractViolation(message))
                if message == "native_sortformer.invalid_model_stage_document"
        ));

        let mut over_capacity = sortformer_document();
        over_capacity.final_projection = Some(
            (0..=SORTFORMER_ORACLE_MAX_SPEAKERS)
                .map(|index| {
                    EvaluationTurn::labeled(
                        u64::try_from(index).expect("index") * 100,
                        u64::try_from(index + 1).expect("index") * 100,
                        format!("speaker_{index}"),
                    )
                })
                .collect(),
        );
        assert_eq!(
            validate_sortformer_stage_contract(&over_capacity),
            Err(DifferentialSkipReason::ModelCapacityExceeded)
        );
        assert!(matches!(
            validate_sortformer_stage_document(&over_capacity),
            Err(FwError::ContractViolation(message))
                if message == "native_sortformer.model_capacity_exceeded"
        ));
    }

    #[test]
    fn sortformer_stage_contract_enforces_frame_order_and_cross_stage_consistency() {
        let mut wrong_arrival_order = sortformer_document();
        let turns = wrong_arrival_order
            .final_projection
            .as_mut()
            .expect("projection");
        turns[0].speaker = Some("speaker_1".to_owned());
        turns[1].speaker = Some("speaker_0".to_owned());
        assert_eq!(
            validate_tool_stage_document(DifferentialOracleTool::Sortformer, &wrong_arrival_order),
            Err(DifferentialSkipReason::InvalidOracleOutput)
        );

        let mut unaligned = sortformer_document();
        unaligned.final_projection.as_mut().expect("projection")[0].start_ms = 161;
        unaligned.speech_activity = Some(vec![interval(161, 1_920)]);
        assert_eq!(
            validate_tool_stage_document(DifferentialOracleTool::Sortformer, &unaligned),
            Err(DifferentialSkipReason::InvalidOracleOutput)
        );

        let mut contradictory_activity = sortformer_document();
        contradictory_activity.speech_activity = Some(vec![interval(160, 960)]);
        assert_eq!(
            validate_tool_stage_document(
                DifferentialOracleTool::Sortformer,
                &contradictory_activity
            ),
            Err(DifferentialSkipReason::InvalidOracleOutput)
        );

        let mut contradictory_overlap = sortformer_document();
        contradictory_overlap.overlap = Some(vec![interval(160, 240)]);
        assert_eq!(
            validate_tool_stage_document(
                DifferentialOracleTool::Sortformer,
                &contradictory_overlap
            ),
            Err(DifferentialSkipReason::InvalidOracleOutput)
        );

        let mut contradictory_change = sortformer_document();
        contradictory_change.change_boundaries_ms = Some(vec![800]);
        assert_eq!(
            validate_tool_stage_document(DifferentialOracleTool::Sortformer, &contradictory_change),
            Err(DifferentialSkipReason::InvalidOracleOutput)
        );

        let mut flagged_overlap = sortformer_document();
        flagged_overlap
            .final_projection
            .as_mut()
            .expect("projection")[0]
            .overlap_suspected = true;
        assert_eq!(
            validate_tool_stage_document(DifferentialOracleTool::Sortformer, &flagged_overlap),
            Err(DifferentialSkipReason::InvalidOracleOutput)
        );
    }

    #[test]
    fn sortformer_arrival_order_allows_simultaneous_first_onsets() {
        let mut document = sortformer_document();
        document.speech_activity = Some(vec![interval(160, 960)]);
        document.change_boundaries_ms = Some(vec![480]);
        document.overlap = Some(vec![interval(160, 480)]);
        document.final_projection = Some(vec![
            EvaluationTurn::labeled(160, 480, "speaker_1"),
            EvaluationTurn::labeled(160, 960, "speaker_0"),
        ]);
        validate_stage_document(&document).expect("valid generic geometry");
        validate_sortformer_stage_document(&document)
            .expect("native Sortformer output boundary accepts canonical document");
    }

    #[test]
    fn external_boundary_rejects_project_inputs_and_outputs() {
        let project_root = std::env::current_dir()
            .expect("current directory")
            .canonicalize()
            .expect("project root");
        assert!(
            canonical_external_file(&project_root, &project_root.join("Cargo.toml"), "audio")
                .is_err()
        );
        assert!(
            validate_external_output(
                &project_root,
                &project_root.join("must-not-create-differential-report.json")
            )
            .is_err()
        );
    }

    #[test]
    fn interval_score_handles_empty_and_partial_overlap() {
        let empty = score_intervals(&[], &[]);
        assert_eq!(empty.iou, None);
        let score = score_intervals(&[interval(0, 100)], &[interval(50, 150)]);
        assert_eq!(score.intersection_ms, 50);
        assert_eq!(score.union_ms, 150);
        assert!((score.iou.expect("iou") - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn comparison_rejects_quadratic_workloads_before_scoring() {
        let mut excessive_changes = document();
        excessive_changes.duration_ms = 10_000;
        excessive_changes.change_boundaries_ms = Some(
            (1..=u64::try_from(MAX_COMPARISON_CHANGE_POINTS + 1).expect("change cap")).collect(),
        );
        assert!(
            compare_documents(
                &excessive_changes,
                &excessive_changes,
                None,
                &DifferentialComparisonConfig::default()
            )
            .is_err()
        );

        let mut excessive_speakers = document();
        excessive_speakers.final_projection = Some(
            (0..=MAX_COMPARISON_SPEAKERS)
                .map(|index| {
                    EvaluationTurn::labeled(
                        u64::try_from(index).expect("index") * 10,
                        u64::try_from(index + 1).expect("index") * 10,
                        format!("speaker_{index}"),
                    )
                })
                .collect(),
        );
        assert!(
            compare_documents(
                &excessive_speakers,
                &excessive_speakers,
                None,
                &DifferentialComparisonConfig::default()
            )
            .is_err()
        );

        let mut excessive_turns = document();
        excessive_turns.duration_ms = 10_000;
        excessive_turns.final_projection = Some(
            (0..=MAX_COMPARISON_TURNS)
                .map(|index| {
                    EvaluationTurn::labeled(
                        u64::try_from(index).expect("index"),
                        u64::try_from(index + 1).expect("index"),
                        "speaker_0",
                    )
                })
                .collect(),
        );
        assert!(
            compare_documents(
                &excessive_turns,
                &excessive_turns,
                None,
                &DifferentialComparisonConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn command_disappearance_is_attributed_to_the_attempted_stage() {
        let version_error = classify_command_error(
            FwError::CommandMissing {
                command: "disappeared".to_owned(),
            },
            DifferentialExecutionStage::VersionProbe,
        );
        assert!(matches!(
            version_error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::VersionProbeFailed,
                stage: DifferentialExecutionStage::VersionProbe,
                ..
            })
        ));
        let run_error = classify_command_error(
            FwError::CommandMissing {
                command: "disappeared".to_owned(),
            },
            DifferentialExecutionStage::OracleRun,
        );
        assert!(matches!(
            run_error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::OracleRunFailed,
                stage: DifferentialExecutionStage::OracleRun,
                ..
            })
        ));
    }

    #[test]
    fn cluster_score_is_label_permutation_invariant() {
        let native = vec![
            cluster("seg-01", "a"),
            cluster("seg-02", "a"),
            cluster("seg-03", "b"),
        ];
        let oracle = vec![
            cluster("seg-01", "x"),
            cluster("seg-02", "x"),
            cluster("seg-03", "y"),
        ];
        let score = score_clusters(&native, &oracle);
        assert_eq!(score.pair_disagreement_count, 0);
        assert_eq!(score.pair_disagreement_rate, Some(0.0));
        assert_eq!(score.coassignment_f1, Some(1.0));
    }

    #[test]
    fn cluster_comparison_detects_missing_segments_and_anchor_geometry_changes() {
        let native = document();
        let mut missing = native.clone();
        missing
            .cluster_assignments
            .as_mut()
            .expect("cluster assignments")
            .pop();
        let comparisons = compare_documents(
            &native,
            &missing,
            None,
            &DifferentialComparisonConfig::default(),
        )
        .expect("compare missing segment");
        assert_eq!(comparisons[3].state, DifferentialStageState::Divergent);
        assert!(matches!(
            comparisons.get(3).and_then(|value| value.metric.as_ref()),
            Some(DifferentialStageMetric::ClusterAssignments(score))
                if score.shared_segment_count == 2
                    && score.segment_coverage == Some(2.0 / 3.0)
        ));

        let mut moved = native.clone();
        moved.cluster_assignments.as_mut().expect("clusters")[0].end_ms += 1;
        let comparisons = compare_documents(
            &native,
            &moved,
            None,
            &DifferentialComparisonConfig::default(),
        )
        .expect("compare moved anchor");
        assert_eq!(comparisons[3].state, DifferentialStageState::Divergent);
        assert!(matches!(
            comparisons.get(3).and_then(|value| value.metric.as_ref()),
            Some(DifferentialStageMetric::ClusterAssignments(score))
                if score.geometry_disagreement_count == 1
        ));
    }

    #[test]
    fn cluster_confidence_delta_is_reported_without_assuming_cross_tool_calibration() {
        let native = document();
        let mut oracle = native.clone();
        for assignment in oracle
            .cluster_assignments
            .as_mut()
            .expect("cluster assignments")
        {
            assignment.confidence = Some(0.1);
        }
        let comparisons = compare_documents(
            &native,
            &oracle,
            None,
            &DifferentialComparisonConfig::default(),
        )
        .expect("compare confidence scales");
        assert_eq!(comparisons[3].state, DifferentialStageState::Equivalent);
        assert!(matches!(
            comparisons.get(3).and_then(|value| value.metric.as_ref()),
            Some(DifferentialStageMetric::ClusterAssignments(score))
                if score.shared_confidence_count == 3
                    && score.mean_absolute_confidence_delta
                        .is_some_and(|delta| (delta - 0.8).abs() < 1e-12)
        ));
    }

    #[test]
    fn final_projection_is_label_permutation_invariant() {
        let native = document();
        let mut oracle = native.clone();
        for turn in oracle.final_projection.as_mut().expect("turns") {
            turn.speaker = Some(if turn.speaker.as_deref() == Some("a") {
                "x".to_owned()
            } else {
                "y".to_owned()
            });
        }
        let comparisons = compare_documents(
            &native,
            &oracle,
            None,
            &DifferentialComparisonConfig::default(),
        )
        .expect("compare");
        let projection = comparisons
            .iter()
            .find(|comparison| comparison.stage == DifferentialStage::FinalProjection)
            .expect("projection");
        assert_eq!(projection.state, DifferentialStageState::Equivalent);
        assert_eq!(projection.diagnostic_loss, Some(0.0));
    }

    #[test]
    fn matching_unknown_projection_is_equivalent_but_known_unknown_disagreement_is_visible() {
        let mut native = document();
        native.final_projection = Some(vec![EvaluationTurn::unknown(100, 1_900)]);
        let oracle = native.clone();
        let comparisons = compare_documents(
            &native,
            &oracle,
            None,
            &DifferentialComparisonConfig::default(),
        )
        .expect("matching unknown");
        assert_eq!(comparisons[5].state, DifferentialStageState::Equivalent);
        assert_eq!(comparisons[5].diagnostic_loss, Some(0.0));

        let mut known = oracle;
        known.final_projection = Some(vec![EvaluationTurn::labeled(100, 1_900, "speaker_x")]);
        let comparisons = compare_documents(
            &native,
            &known,
            None,
            &DifferentialComparisonConfig::default(),
        )
        .expect("known versus unknown");
        assert_eq!(comparisons[5].state, DifferentialStageState::Divergent);
        assert!(matches!(
            comparisons.get(5).and_then(|value| value.metric.as_ref()),
            Some(DifferentialStageMetric::FinalProjection(score))
                if score.unknown_status_disagreement_ms == 1_800
        ));
    }

    #[test]
    fn earliest_divergence_respects_stage_order() {
        let native = document();
        let mut oracle = native.clone();
        oracle.change_boundaries_ms = Some(vec![200]);
        oracle.final_projection = Some(vec![EvaluationTurn::labeled(100, 1_900, "x")]);
        let comparisons = compare_documents(
            &native,
            &oracle,
            None,
            &DifferentialComparisonConfig::default(),
        )
        .expect("compare");
        let earliest = comparisons
            .iter()
            .find(|comparison| comparison.state == DifferentialStageState::Divergent)
            .map(|comparison| comparison.stage);
        assert_eq!(earliest, Some(DifferentialStage::ChangeBoundaries));
    }

    #[test]
    fn partial_stage_output_is_not_invented_as_a_difference() {
        let native = document();
        let mut oracle = native.clone();
        oracle.word_timing = None;
        let comparisons = compare_documents(
            &native,
            &oracle,
            None,
            &DifferentialComparisonConfig::default(),
        )
        .expect("compare");
        let words = &comparisons[1];
        assert_eq!(words.state, DifferentialStageState::MissingOracle);
        assert_eq!(words.metric, None);
        assert_eq!(
            words.adjudication,
            DifferentialAdjudication::ReferenceStageUnavailable
        );
    }

    #[test]
    fn reference_adjudication_can_favor_either_side_without_minting_authority() {
        let native = document();
        let reference = native.clone();
        let mut oracle = native.clone();
        oracle.speech_activity = Some(vec![interval(100, 1_000)]);
        let comparisons = compare_documents(
            &native,
            &oracle,
            Some(&reference),
            &DifferentialComparisonConfig::default(),
        )
        .expect("compare");
        assert_eq!(
            comparisons[0].adjudication,
            DifferentialAdjudication::ReferenceFavorsNative
        );
    }

    #[test]
    fn missing_reference_stage_is_distinct_from_missing_reference_document() {
        let native = document();
        let mut oracle = native.clone();
        oracle.speech_activity = Some(vec![interval(100, 1_000)]);
        let mut reference = native.clone();
        reference.speech_activity = None;
        let comparisons = compare_documents(
            &native,
            &oracle,
            Some(&reference),
            &DifferentialComparisonConfig::default(),
        )
        .expect("compare");
        assert_eq!(
            comparisons[0].adjudication,
            DifferentialAdjudication::ReferenceStageUnavailable
        );
    }

    #[test]
    fn missing_binary_is_cleanly_skipped() {
        let prepared = PreparedInputs {
            audio_path: PathBuf::from("/dev/null"),
            native: document(),
            reference: None,
            audio_sha256: bytes_sha256(b""),
            native_sha256: bytes_sha256(b"native"),
            reference_sha256: None,
        };
        let report = build_report(
            DifferentialOracleTool::Pyannote,
            &ProgramSpec {
                program: "franken-whisper-definitely-missing-oracle".to_owned(),
                prefix_args: Vec::new(),
            },
            &prepared,
            Duration::from_secs(1),
            DifferentialComparisonConfig::default(),
            &CancellationToken::no_deadline(),
        )
        .expect("skip report");
        assert_eq!(report.status, DifferentialRunStatus::Skipped);
        assert_eq!(
            report.skip_reason,
            Some(DifferentialSkipReason::MissingExecutable)
        );
        assert!(!report.native_incorrectness_claim_permitted);
        let mut forged = report;
        forged.provenance.version_stdout_sha256 = Some(bytes_sha256(b"impossible"));
        forged.result_sha256.clear();
        forged.result_sha256 = canonical_sha256(&forged).expect("forged self hash");
        assert!(verify_differential_report(&forged).is_err());
    }

    #[test]
    fn sortformer_over_capacity_reference_is_retained_as_ineligible() {
        let native = sortformer_document();
        let mut reference = native.clone();
        reference.final_projection = Some(
            (0..=SORTFORMER_ORACLE_MAX_SPEAKERS)
                .map(|index| {
                    EvaluationTurn::labeled(
                        u64::try_from(index).expect("index") * 100,
                        u64::try_from(index + 1).expect("index") * 100,
                        format!("reference_{index}"),
                    )
                })
                .collect(),
        );
        let prepared = PreparedInputs {
            audio_path: PathBuf::from("/does/not/need/to/exist.wav"),
            native,
            reference: Some(reference),
            audio_sha256: bytes_sha256(b"audio"),
            native_sha256: bytes_sha256(b"native"),
            reference_sha256: Some(bytes_sha256(b"reference")),
        };
        let report = build_report(
            DifferentialOracleTool::Sortformer,
            &ProgramSpec {
                program: "franken-whisper-definitely-missing-oracle".to_owned(),
                prefix_args: Vec::new(),
            },
            &prepared,
            Duration::from_secs(1),
            DifferentialComparisonConfig::default(),
            &CancellationToken::no_deadline(),
        )
        .expect("capacity skip report");
        assert_eq!(report.status, DifferentialRunStatus::Skipped);
        assert_eq!(
            report.skip_reason,
            Some(DifferentialSkipReason::ReferenceModelCapacityExceeded)
        );
        assert_eq!(
            report.failure_stage,
            Some(DifferentialExecutionStage::EligibilityValidation)
        );
        assert_eq!(
            report.provenance.expected_model_contract_sha256.as_deref(),
            Some(SORTFORMER_ORACLE_CONTRACT_SHA256)
        );
        assert!(report.provenance.model_contract_sha256.is_none());
        assert!(report.provenance.model_artifact_sha256.is_none());
        assert!(report.provenance.runtime_fingerprint_sha256.is_none());
        verify_differential_report(&report).expect("verified capacity skip");
    }

    #[test]
    fn cancellation_precedes_sortformer_capacity_eligibility_skip() {
        let native = sortformer_document();
        let mut reference = native.clone();
        reference.final_projection = Some(
            (0..=SORTFORMER_ORACLE_MAX_SPEAKERS)
                .map(|index| {
                    EvaluationTurn::labeled(
                        u64::try_from(index).expect("index") * 100,
                        u64::try_from(index + 1).expect("index") * 100,
                        format!("reference_{index}"),
                    )
                })
                .collect(),
        );
        let prepared = PreparedInputs {
            audio_path: PathBuf::from("/does/not/need/to/exist.wav"),
            native,
            reference: Some(reference),
            audio_sha256: bytes_sha256(b"audio"),
            native_sha256: bytes_sha256(b"native"),
            reference_sha256: Some(bytes_sha256(b"reference")),
        };
        let error = build_report(
            DifferentialOracleTool::Sortformer,
            &ProgramSpec {
                program: "franken-whisper-definitely-missing-oracle".to_owned(),
                prefix_args: Vec::new(),
            },
            &prepared,
            Duration::from_secs(1),
            DifferentialComparisonConfig::default(),
            &CancellationToken::already_expired(),
        )
        .expect_err("cancel before capacity classification");
        assert!(matches!(error, FwError::Cancelled(_)));
    }

    fn shell_program(script: String) -> ProgramSpec {
        ProgramSpec {
            program: "sh".to_owned(),
            prefix_args: vec!["-c".to_owned(), script, "oracle-test".to_owned()],
        }
    }

    fn sortformer_runtime_fingerprint() -> DifferentialOracleRuntimeFingerprint {
        DifferentialOracleRuntimeFingerprint {
            schema_version: "sortformer-runtime-fingerprint-v1".to_owned(),
            python_version: SORTFORMER_ORACLE_PYTHON_VERSION.to_owned(),
            nemo_version: SORTFORMER_ORACLE_NEMO_VERSION.to_owned(),
            torch_version: SORTFORMER_ORACLE_TORCH_VERSION.to_owned(),
            torchaudio_version: SORTFORMER_ORACLE_TORCHAUDIO_VERSION.to_owned(),
            numpy_version: SORTFORMER_ORACLE_NUMPY_VERSION.to_owned(),
            blas_backend: "accelerate".to_owned(),
            operating_system: "macos-15.5".to_owned(),
            machine_architecture: "aarch64".to_owned(),
            cpu_feature_tier: "neon-dotprod".to_owned(),
            device: "cpu".to_owned(),
            compute_dtype: "float32".to_owned(),
            autocast: false,
            quantization: "none".to_owned(),
            torch_intraop_threads: 8,
            torch_interop_threads: 1,
            data_loader_workers: 0,
            deterministic_algorithms: true,
        }
    }

    fn version_json(tool: DifferentialOracleTool) -> String {
        let mut version = serde_json::json!({
            "schema_version": DIFFERENTIAL_ORACLE_VERSION_SCHEMA,
            "protocol_version": DIFFERENTIAL_ORACLE_PROTOCOL_VERSION,
            "tool": tool,
            "tool_version": "1.2.3",
            "adapter_version": "adapter-1"
        });
        if tool == DifferentialOracleTool::Sortformer {
            let runtime_fingerprint = sortformer_runtime_fingerprint();
            version["tool_version"] = serde_json::json!(SORTFORMER_ORACLE_TOOL_VERSION);
            version["adapter_version"] = serde_json::json!(SORTFORMER_ORACLE_ADAPTER_VERSION);
            version["model_contract_sha256"] = serde_json::json!(SORTFORMER_ORACLE_CONTRACT_SHA256);
            version["model_artifact_sha256"] = serde_json::json!(SORTFORMER_ORACLE_ARTIFACT_SHA256);
            version["model_artifact_bytes"] = serde_json::json!(SORTFORMER_ORACLE_ARTIFACT_BYTES);
            version["runtime_fingerprint_sha256"] =
                serde_json::json!(canonical_sha256(&runtime_fingerprint).expect("runtime hash"));
            version["runtime_fingerprint"] =
                serde_json::to_value(runtime_fingerprint).expect("runtime JSON");
        }
        version.to_string()
    }

    fn sortformer_document() -> DifferentialStageDocument {
        DifferentialStageDocument {
            schema_version: DIFFERENTIAL_STAGE_DOCUMENT_SCHEMA.to_owned(),
            recording_key: KEY.to_owned(),
            duration_ms: 2_000,
            speech_activity: Some(vec![interval(160, 1_920)]),
            word_timing: None,
            change_boundaries_ms: Some(vec![960]),
            cluster_assignments: None,
            overlap: Some(Vec::new()),
            final_projection: Some(vec![
                EvaluationTurn::labeled(160, 960, "speaker_0"),
                EvaluationTurn::labeled(960, 1_920, "speaker_1"),
            ]),
        }
    }

    fn write_sortformer_audio(path: &Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("create canonical WAV");
        for _ in 0..32_000 {
            writer.write_sample(0i16).expect("write sample");
        }
        writer.finalize().expect("finalize WAV");
    }

    fn successful_shell_program(
        tool: DifferentialOracleTool,
        oracle: &DifferentialStageDocument,
    ) -> ProgramSpec {
        let version = version_json(tool);
        let output = serde_json::to_string(oracle).expect("oracle json");
        shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else printf '%s' '{output}'; fi"
        ))
    }

    fn test_audio_sha256(path: &Path) -> String {
        hash_file(path, &CancellationToken::no_deadline()).expect("hash test audio")
    }

    fn sortformer_observation_request<'a>(
        audio_path: &'a Path,
        audio_sha256: &'a str,
        hard_timeout: Duration,
    ) -> SortformerObservationRequest<'a> {
        SortformerObservationRequest {
            audio_path,
            expected_audio_sha256: audio_sha256,
            expected_duration_ms: 2_000,
            recording_key: KEY,
            hard_timeout,
        }
    }

    #[test]
    fn sortformer_observation_returns_validated_document_and_path_free_provenance() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let audio = directory.path().join("canonical.wav");
        write_sortformer_audio(&audio);
        let audio = audio.canonicalize().expect("canonical audio path");
        let audio_sha256 = test_audio_sha256(&audio);
        let outcome = run_sortformer_observation_with_program(
            sortformer_observation_request(&audio, &audio_sha256, Duration::from_secs(5)),
            &successful_shell_program(DifferentialOracleTool::Sortformer, &sortformer_document()),
            &CancellationToken::no_deadline(),
        )
        .expect("completed observation");

        assert!(
            matches!(&outcome, SortformerObservationOutcome::Completed { .. }),
            "unexpected observation outcome: {outcome:?}"
        );
        let path_text = audio.to_string_lossy();
        if let SortformerObservationOutcome::Completed {
            document,
            provenance,
        } = &outcome
        {
            assert_eq!(document, &sortformer_document());
            validate_stage_document(document).expect("validated returned document");
            validate_tool_stage_document(DifferentialOracleTool::Sortformer, document)
                .expect("validated Sortformer stage profile");
            assert_eq!(
                provenance.expected_model_contract_sha256,
                SORTFORMER_ORACLE_CONTRACT_SHA256
            );
            assert_eq!(
                provenance.model_artifact_sha256.as_deref(),
                Some(SORTFORMER_ORACLE_ARTIFACT_SHA256)
            );
            assert_eq!(provenance.audio_sha256, audio_sha256);
            assert_eq!(provenance.authority, DifferentialAuthority::DiagnosticOnly);
            let serialized = serde_json::to_string(provenance).expect("serialize provenance");
            assert!(!serialized.contains(path_text.as_ref()));
            assert!(!serialized.contains("canonical.wav"));
        }

        let debug = format!("{outcome:?}");
        assert!(!debug.contains(path_text.as_ref()));
    }

    #[test]
    fn sortformer_observation_missing_adapter_is_stable_skip() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let audio = directory.path().join("canonical.wav");
        write_sortformer_audio(&audio);
        let audio = audio.canonicalize().expect("canonical audio path");
        let audio_sha256 = test_audio_sha256(&audio);
        let outcome = run_sortformer_observation_with_program(
            sortformer_observation_request(&audio, &audio_sha256, Duration::from_secs(1)),
            &ProgramSpec {
                program: "franken-whisper-definitely-missing-sortformer-observer".to_owned(),
                prefix_args: Vec::new(),
            },
            &CancellationToken::no_deadline(),
        )
        .expect("missing adapter is a skip");

        assert!(matches!(
            &outcome,
            SortformerObservationOutcome::Skipped {
                reason: DifferentialSkipReason::MissingExecutable,
                stage: DifferentialExecutionStage::ResolveExecutable,
                ..
            }
        ));
        if let SortformerObservationOutcome::Skipped { provenance, .. } = outcome {
            assert!(provenance.executable_sha256.is_none());
            assert!(provenance.tool_version.is_none());
            assert_eq!(provenance.audio_sha256, audio_sha256);
        }
    }

    #[test]
    fn sortformer_observation_malformed_output_and_timeout_are_stable_skips() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let audio = directory.path().join("canonical.wav");
        write_sortformer_audio(&audio);
        let audio = audio.canonicalize().expect("canonical audio path");
        let audio_sha256 = test_audio_sha256(&audio);
        let version = version_json(DifferentialOracleTool::Sortformer);
        let malformed_program = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else printf '%s' 'not-json'; fi"
        ));
        let malformed = run_sortformer_observation_with_program(
            sortformer_observation_request(&audio, &audio_sha256, Duration::from_secs(1)),
            &malformed_program,
            &CancellationToken::no_deadline(),
        )
        .expect("malformed output is a skip");
        assert!(matches!(
            malformed,
            SortformerObservationOutcome::Skipped {
                reason: DifferentialSkipReason::InvalidOracleOutput,
                stage: DifferentialExecutionStage::OracleOutputValidation,
                ..
            }
        ));

        let slow_program = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else while :; do :; done; fi"
        ));
        let timed_out = run_sortformer_observation_with_program(
            sortformer_observation_request(&audio, &audio_sha256, Duration::from_millis(20)),
            &slow_program,
            &CancellationToken::no_deadline(),
        )
        .expect("timeout is a skip");
        assert!(
            matches!(
                &timed_out,
                SortformerObservationOutcome::Skipped {
                    reason: DifferentialSkipReason::OracleRunTimedOut,
                    stage: DifferentialExecutionStage::OracleRun,
                    ..
                }
            ),
            "unexpected timeout outcome: {timed_out:?}"
        );
    }

    #[test]
    fn sortformer_observation_rejects_input_identity_and_post_run_mutation() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let audio = directory.path().join("canonical.wav");
        write_sortformer_audio(&audio);
        let audio = audio.canonicalize().expect("canonical audio path");
        let wrong_identity = bytes_sha256(b"different canonical audio");
        let program =
            successful_shell_program(DifferentialOracleTool::Sortformer, &sortformer_document());
        let mismatch = run_sortformer_observation_with_program(
            sortformer_observation_request(&audio, &wrong_identity, Duration::from_secs(1)),
            &program,
            &CancellationToken::no_deadline(),
        )
        .expect("identity mismatch is a skip");
        assert!(matches!(
            mismatch,
            SortformerObservationOutcome::Skipped {
                reason: DifferentialSkipReason::InputIdentityMismatch,
                stage: DifferentialExecutionStage::InputValidation,
                ..
            }
        ));

        let mutating_audio = directory.path().join("mutating.wav");
        write_sortformer_audio(&mutating_audio);
        let mutating_audio = mutating_audio
            .canonicalize()
            .expect("canonical mutation path");
        let expected_hash = test_audio_sha256(&mutating_audio);
        let version = version_json(DifferentialOracleTool::Sortformer);
        let output = serde_json::to_string(&sortformer_document()).expect("oracle JSON");
        let mutating_program = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else printf x >> \"$5\"; printf '%s' '{output}'; fi"
        ));
        let mutation = run_sortformer_observation_with_program(
            sortformer_observation_request(&mutating_audio, &expected_hash, Duration::from_secs(1)),
            &mutating_program,
            &CancellationToken::no_deadline(),
        )
        .expect("post-run mutation is a skip");
        assert!(matches!(
            mutation,
            SortformerObservationOutcome::Skipped {
                reason: DifferentialSkipReason::InputIdentityMismatch,
                stage: DifferentialExecutionStage::InputPostRunValidation,
                ..
            }
        ));
    }

    #[test]
    fn sortformer_observation_preserves_cancellation() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let audio = directory.path().join("canonical.wav");
        write_sortformer_audio(&audio);
        let audio = audio.canonicalize().expect("canonical audio path");
        let audio_sha256 = test_audio_sha256(&audio);
        let error = run_sortformer_observation_with_program(
            sortformer_observation_request(&audio, &audio_sha256, Duration::from_secs(1)),
            &ProgramSpec {
                program: "franken-whisper-definitely-missing-sortformer-observer".to_owned(),
                prefix_args: Vec::new(),
            },
            &CancellationToken::already_expired(),
        )
        .expect_err("cancellation must not become a skip");
        assert!(matches!(error, FwError::Cancelled(_)));
    }

    #[test]
    fn invalid_version_json_is_cleanly_skipped() {
        let error = execute_external(
            DifferentialOracleTool::Pyannote,
            &shell_program("printf '%s' 'not-json'".to_owned()),
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("invalid version");
        assert!(matches!(
            error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::InvalidVersionOutput,
                ..
            })
        ));
    }

    #[test]
    fn expected_executable_digest_fails_closed_before_adapter_self_report() {
        let expected = "0000000000000000000000000000000000000000000000000000000000000000";
        let error = execute_external(
            DifferentialOracleTool::Sortformer,
            &shell_program("exit 0".to_owned()),
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline())
                .with_expected_executable_sha256(Some(expected)),
        )
        .expect_err("an unpinned adapter executable must be rejected before version probing");

        assert!(matches!(
            error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::ExecutableIdentityMismatch,
                stage: DifferentialExecutionStage::HashExecutable,
                executable_sha256: Some(observed),
                ..
            }) if observed != expected
        ));
    }

    #[test]
    fn executable_recheck_retains_observed_mismatch_and_classifies_read_failure() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let executable = directory.path().join("adapter");
        std::fs::write(&executable, b"changed adapter").expect("write adapter fixture");
        let observed = bytes_sha256(b"changed adapter");
        let mismatch = validate_executable_identity(
            &executable,
            &bytes_sha256(b"original adapter"),
            &CancellationToken::no_deadline(),
        )
        .expect_err("changed executable must fail closed");
        assert!(matches!(
            mismatch,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::ExecutableIdentityMismatch,
                stage: DifferentialExecutionStage::HashExecutable,
                executable_sha256: Some(retained),
                ..
            }) if retained == observed
        ));

        let unreadable = validate_executable_identity(
            &directory.path().join("missing-adapter"),
            &observed,
            &CancellationToken::no_deadline(),
        )
        .expect_err("missing executable cannot be rehashed");
        assert!(matches!(
            unreadable,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::UnreadableExecutable,
                stage: DifferentialExecutionStage::HashExecutable,
                executable_sha256: None,
                ..
            })
        ));
    }

    #[test]
    fn version_mismatch_is_cleanly_skipped() {
        let wrong = serde_json::json!({
            "schema_version": DIFFERENTIAL_ORACLE_VERSION_SCHEMA,
            "protocol_version": "wrong",
            "tool": "pyannote",
            "tool_version": "1",
            "adapter_version": "1"
        });
        let error = execute_external(
            DifferentialOracleTool::Pyannote,
            &shell_program(format!("printf '%s' '{wrong}'")),
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("version mismatch");
        assert!(matches!(
            error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::ProtocolVersionMismatch,
                ..
            })
        ));
    }

    #[test]
    fn non_sortformer_version_rejects_model_provenance_fields() {
        let mut polluted: serde_json::Value =
            serde_json::from_str(&version_json(DifferentialOracleTool::Pyannote))
                .expect("version JSON");
        polluted["model_contract_sha256"] = serde_json::json!(bytes_sha256(b"pollution"));
        let error = execute_external(
            DifferentialOracleTool::Pyannote,
            &shell_program(format!("printf '%s' '{polluted}'")),
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("polluted non-Sortformer version");
        assert!(matches!(
            error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::InvalidVersionOutput,
                ..
            })
        ));
    }

    #[test]
    fn sortformer_rejects_input_and_model_contract_mismatches() {
        let invalid_input = execute_external(
            DifferentialOracleTool::Sortformer,
            &successful_shell_program(DifferentialOracleTool::Sortformer, &sortformer_document()),
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("input contract");
        assert!(matches!(
            invalid_input,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::InputContractMismatch,
                ..
            })
        ));

        let directory = tempfile::tempdir().expect("temporary directory");
        let audio = directory.path().join("canonical.wav");
        write_sortformer_audio(&audio);
        for (field, wrong_value) in [
            (
                "model_contract_sha256",
                serde_json::json!(bytes_sha256(b"wrong contract")),
            ),
            (
                "model_artifact_sha256",
                serde_json::json!(bytes_sha256(b"wrong artifact")),
            ),
            (
                "runtime_fingerprint_sha256",
                serde_json::json!(bytes_sha256(b"wrong runtime fingerprint")),
            ),
            ("model_artifact_bytes", serde_json::json!(1)),
            ("tool_version", serde_json::json!("wrong-tool-version")),
            (
                "adapter_version",
                serde_json::json!("wrong-adapter-version"),
            ),
        ] {
            let mut wrong_version: serde_json::Value =
                serde_json::from_str(&version_json(DifferentialOracleTool::Sortformer))
                    .expect("version JSON");
            wrong_version[field] = wrong_value;
            let error = execute_external(
                DifferentialOracleTool::Sortformer,
                &shell_program(format!("printf '%s' '{wrong_version}'")),
                &audio,
                &test_audio_sha256(&audio),
                2_000,
                KEY,
                ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
            )
            .expect_err("model contract mismatch");
            assert!(matches!(
                error,
                ExternalRunError::Skipped(ExternalSkip {
                    reason: DifferentialSkipReason::ModelContractMismatch,
                    ..
                })
            ));
        }

        let mut missing_runtime: serde_json::Value =
            serde_json::from_str(&version_json(DifferentialOracleTool::Sortformer))
                .expect("version JSON");
        missing_runtime
            .as_object_mut()
            .expect("version object")
            .remove("runtime_fingerprint_sha256");
        let error = execute_external(
            DifferentialOracleTool::Sortformer,
            &shell_program(format!("printf '%s' '{missing_runtime}'")),
            &audio,
            &test_audio_sha256(&audio),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("missing runtime fingerprint");
        assert!(matches!(
            error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::ModelContractMismatch,
                ..
            })
        ));

        for (field, wrong_value) in [
            ("python_version", serde_json::json!("3.12.11")),
            ("nemo_version", serde_json::json!("3.1.0")),
            ("torch_version", serde_json::json!("2.7.0")),
            ("torchaudio_version", serde_json::json!("2.7.0")),
            ("numpy_version", serde_json::json!("2.4.5")),
            ("device", serde_json::json!("cuda")),
        ] {
            let mut wrong_runtime: serde_json::Value =
                serde_json::from_str(&version_json(DifferentialOracleTool::Sortformer))
                    .expect("version JSON");
            wrong_runtime["runtime_fingerprint"][field] = wrong_value;
            let fingerprint: DifferentialOracleRuntimeFingerprint =
                serde_json::from_value(wrong_runtime["runtime_fingerprint"].clone())
                    .expect("runtime fingerprint");
            wrong_runtime["runtime_fingerprint_sha256"] =
                serde_json::json!(canonical_sha256(&fingerprint).expect("runtime hash"));
            let error = execute_external(
                DifferentialOracleTool::Sortformer,
                &shell_program(format!("printf '%s' '{wrong_runtime}'")),
                &audio,
                &test_audio_sha256(&audio),
                2_000,
                KEY,
                ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
            )
            .expect_err("runtime profile mismatch");
            assert!(matches!(
                error,
                ExternalRunError::Skipped(ExternalSkip {
                    reason: DifferentialSkipReason::ModelContractMismatch,
                    ..
                })
            ));
        }
    }

    #[test]
    fn sortformer_validates_pcm_duration_completeness_identity_and_cancellation() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let audio = directory.path().join("canonical.wav");
        write_sortformer_audio(&audio);

        let duration_error = validate_tool_input(
            DifferentialOracleTool::Sortformer,
            &audio,
            90_000,
            &CancellationToken::no_deadline(),
        )
        .expect_err("duration mismatch");
        assert!(matches!(
            duration_error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::InputContractMismatch,
                ..
            })
        ));

        let cancelled = validate_tool_input(
            DifferentialOracleTool::Sortformer,
            &audio,
            2_000,
            &CancellationToken::already_expired(),
        )
        .expect_err("cancel input validation");
        assert!(matches!(
            cancelled,
            ExternalRunError::Cancelled(FwError::Cancelled(_))
        ));

        let identity_error = execute_external(
            DifferentialOracleTool::Sortformer,
            &successful_shell_program(DifferentialOracleTool::Sortformer, &sortformer_document()),
            &audio,
            &bytes_sha256(b"different audio"),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("audio identity mismatch");
        assert!(matches!(
            identity_error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::InputIdentityMismatch,
                ..
            })
        ));

        let mutating_audio = directory.path().join("mutating.wav");
        write_sortformer_audio(&mutating_audio);
        let expected_hash = test_audio_sha256(&mutating_audio);
        let version = version_json(DifferentialOracleTool::Sortformer);
        let output = serde_json::to_string(&sortformer_document()).expect("oracle JSON");
        let mutating_program = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else printf x >> \"$5\"; printf '%s' '{output}'; fi"
        ));
        let mutation_error = execute_external(
            DifferentialOracleTool::Sortformer,
            &mutating_program,
            &mutating_audio,
            &expected_hash,
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("post-run audio mutation");
        assert!(matches!(
            mutation_error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::InputIdentityMismatch,
                stage: DifferentialExecutionStage::InputPostRunValidation,
                ..
            })
        ));

        let truncated = directory.path().join("truncated.wav");
        write_sortformer_audio(&truncated);
        let truncated_len = std::fs::metadata(&truncated)
            .expect("truncated metadata")
            .len();
        OpenOptions::new()
            .write(true)
            .open(&truncated)
            .expect("open truncated fixture")
            .set_len(truncated_len - 2)
            .expect("truncate fixture tail");
        let truncated_error = validate_tool_input(
            DifferentialOracleTool::Sortformer,
            &truncated,
            2_000,
            &CancellationToken::no_deadline(),
        )
        .expect_err("truncated PCM");
        assert!(matches!(
            truncated_error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::InputContractMismatch,
                ..
            })
        ));
    }

    #[test]
    fn sortformer_fixture_replay_is_deterministic_and_cancellable() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let audio = directory.path().join("canonical.wav");
        write_sortformer_audio(&audio);
        let program =
            successful_shell_program(DifferentialOracleTool::Sortformer, &sortformer_document());
        let first = execute_external(
            DifferentialOracleTool::Sortformer,
            &program,
            &audio,
            &test_audio_sha256(&audio),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect("first replay");
        let second = execute_external(
            DifferentialOracleTool::Sortformer,
            &program,
            &audio,
            &test_audio_sha256(&audio),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect("second replay");
        assert_eq!(first.oracle, second.oracle);
        assert_eq!(first.oracle_stdout_sha256, second.oracle_stdout_sha256);
        assert_eq!(first.version_stdout_sha256, second.version_stdout_sha256);

        let version = version_json(DifferentialOracleTool::Sortformer);
        let run_marker = directory.path().join("oracle-run-entered");
        let slow = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else printf '%s' entered > '{}'; sleep 60; fi",
            run_marker.display()
        ));
        let cancelled = execute_external(
            DifferentialOracleTool::Sortformer,
            &slow,
            &audio,
            &test_audio_sha256(&audio),
            2_000,
            KEY,
            ExternalRunLimits::new(
                Duration::from_secs(10),
                &CancellationToken::with_deadline_from_now(Duration::from_secs(2)),
            ),
        )
        .expect_err("cancelled during oracle run");
        assert!(matches!(
            cancelled,
            ExternalRunError::Cancelled(FwError::Cancelled(_))
        ));
        assert!(run_marker.is_file(), "oracle run branch was not entered");
    }

    #[test]
    fn sortformer_report_distinguishes_expected_and_validated_attestations() {
        let version_json = version_json(DifferentialOracleTool::Sortformer);
        let version: OracleVersionDocument =
            serde_json::from_str(&version_json).expect("pinned version document");
        let prepared = PreparedInputs {
            audio_path: PathBuf::from("/not-retained-in-the-report.wav"),
            native: sortformer_document(),
            reference: None,
            audio_sha256: bytes_sha256(b"audio"),
            native_sha256: bytes_sha256(b"native"),
            reference_sha256: None,
        };
        let skipped = ExternalSkip::new(
            DifferentialSkipReason::OracleRunFailed,
            DifferentialExecutionStage::OracleRun,
        )
        .with_executable(SORTFORMER_ORACLE_ADAPTER_SHA256)
        .with_valid_version(&version, &bytes_sha256(version_json.as_bytes()));
        let comparison_config = DifferentialComparisonConfig::default();
        let comparison_config_sha256 =
            canonical_sha256(&comparison_config).expect("comparison configuration hash");
        let report = finalize_report(DifferentialOracleReport {
            schema_version: DIFFERENTIAL_REPORT_SCHEMA.to_owned(),
            comparator_version: DIFFERENTIAL_COMPARATOR_VERSION.to_owned(),
            authority: DifferentialAuthority::DiagnosticOnly,
            native_incorrectness_claim_permitted: false,
            status: DifferentialRunStatus::Skipped,
            skip_reason: Some(skipped.reason),
            failure_stage: Some(skipped.stage),
            provenance: provenance_from_skip(
                DifferentialOracleTool::Sortformer,
                &prepared,
                &skipped,
            ),
            comparison_config,
            comparison_config_sha256,
            comparisons: Vec::new(),
            earliest_divergence: None,
            result_sha256: String::new(),
        })
        .expect("validated-version skip report");
        assert_eq!(
            report.skip_reason,
            Some(DifferentialSkipReason::OracleRunFailed)
        );
        assert_eq!(
            report.provenance.expected_model_contract_sha256.as_deref(),
            Some(SORTFORMER_ORACLE_CONTRACT_SHA256)
        );
        assert_eq!(
            report.provenance.model_contract_sha256.as_deref(),
            Some(SORTFORMER_ORACLE_CONTRACT_SHA256)
        );
        assert_eq!(
            report.provenance.model_artifact_sha256.as_deref(),
            Some(SORTFORMER_ORACLE_ARTIFACT_SHA256)
        );
        assert!(
            report
                .provenance
                .runtime_fingerprint_sha256
                .as_deref()
                .is_some_and(is_sha256_hex)
        );
        verify_differential_report(&report).expect("verified report");

        let mut forged = report.clone();
        forged.provenance.model_artifact_sha256 = Some(bytes_sha256(b"forged artifact"));
        forged.result_sha256.clear();
        assert!(finalize_report(forged).is_err());

        let mut forged = report;
        forged.provenance.executable_sha256 = Some(bytes_sha256(b"unfrozen adapter"));
        forged.result_sha256.clear();
        assert!(finalize_report(forged).is_err());
    }

    #[test]
    fn production_sortformer_report_retains_adapter_digest_mismatch() {
        let prepared = PreparedInputs {
            audio_path: PathBuf::from("/not-opened-before-adapter-admission.wav"),
            native: sortformer_document(),
            reference: None,
            audio_sha256: bytes_sha256(b"audio"),
            native_sha256: bytes_sha256(b"native"),
            reference_sha256: None,
        };
        let report = build_report(
            DifferentialOracleTool::Sortformer,
            &shell_program("exit 0".to_owned()),
            &prepared,
            Duration::from_secs(1),
            DifferentialComparisonConfig::default(),
            &CancellationToken::no_deadline(),
        )
        .expect("adapter mismatch must be retained as a stable skipped report");

        assert_eq!(report.status, DifferentialRunStatus::Skipped);
        assert_eq!(
            report.skip_reason,
            Some(DifferentialSkipReason::ExecutableIdentityMismatch)
        );
        assert_eq!(
            report.failure_stage,
            Some(DifferentialExecutionStage::HashExecutable)
        );
        assert!(
            report
                .provenance
                .executable_sha256
                .as_deref()
                .is_some_and(|observed| observed != SORTFORMER_ORACLE_ADAPTER_SHA256)
        );
        verify_differential_report(&report).expect("verified mismatch report");
    }

    #[test]
    fn invalid_run_json_is_cleanly_skipped() {
        let version = version_json(DifferentialOracleTool::Pyannote);
        let program = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else printf '%s' 'invalid'; fi"
        ));
        let error = execute_external(
            DifferentialOracleTool::Pyannote,
            &program,
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("invalid output");
        assert!(matches!(
            error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::InvalidOracleOutput,
                ..
            })
        ));
    }

    #[test]
    fn nonzero_run_and_timeout_are_clean_skips() {
        let version = version_json(DifferentialOracleTool::Pyannote);
        let failed = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else exit 7; fi"
        ));
        let error = execute_external(
            DifferentialOracleTool::Pyannote,
            &failed,
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_secs(1), &CancellationToken::no_deadline()),
        )
        .expect_err("nonzero");
        assert!(matches!(
            error,
            ExternalRunError::Skipped(ExternalSkip {
                reason: DifferentialSkipReason::OracleRunFailed,
                ..
            })
        ));

        let slow = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else while :; do :; done; fi"
        ));
        let error = execute_external(
            DifferentialOracleTool::Pyannote,
            &slow,
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(Duration::from_millis(40), &CancellationToken::no_deadline()),
        )
        .expect_err("timeout");
        assert!(
            matches!(
                &error,
                ExternalRunError::Skipped(ExternalSkip {
                    reason: DifferentialSkipReason::OracleRunTimedOut,
                    ..
                })
            ),
            "unexpected timeout classification: {error:?}"
        );
    }

    #[test]
    fn cancellation_propagates_instead_of_becoming_a_skip() {
        let error = execute_external(
            DifferentialOracleTool::Pyannote,
            &successful_shell_program(DifferentialOracleTool::Pyannote, &document()),
            Path::new("/dev/null"),
            &bytes_sha256(b""),
            2_000,
            KEY,
            ExternalRunLimits::new(
                Duration::from_secs(1),
                &CancellationToken::already_expired(),
            ),
        )
        .expect_err("cancel");
        assert!(matches!(
            error,
            ExternalRunError::Cancelled(FwError::Cancelled(_))
        ));
    }

    #[test]
    fn completed_report_retains_no_paths_labels_or_word_ids() {
        let native = document();
        let directory = tempfile::tempdir().expect("temporary directory");
        let audio_path = directory.path().join("private-secret-call.m4a");
        std::fs::write(&audio_path, b"audio").expect("write test audio");
        let prepared = PreparedInputs {
            audio_path: audio_path.clone(),
            native: native.clone(),
            reference: None,
            audio_sha256: test_audio_sha256(&audio_path),
            native_sha256: bytes_sha256(b"native"),
            reference_sha256: None,
        };
        let report = build_report(
            DifferentialOracleTool::Pyannote,
            &successful_shell_program(DifferentialOracleTool::Pyannote, &native),
            &prepared,
            Duration::from_secs(1),
            DifferentialComparisonConfig::default(),
            &CancellationToken::no_deadline(),
        )
        .expect("report");
        let json = serde_json::to_string(&report).expect("json");
        for forbidden in ["secret-call", "/private", "\"a\"", "w-01", "seg-01"] {
            assert!(!json.contains(forbidden), "leaked {forbidden}: {json}");
        }
        assert_eq!(report.status, DifferentialRunStatus::Completed);
        assert!(!report.result_sha256.is_empty());
        verify_differential_report(&report).expect("verified report");

        let bytes = serde_json::to_vec(&report).expect("report bytes");
        assert_eq!(
            parse_differential_report(&bytes).expect("parsed report"),
            report
        );
        let mut polluted = report.clone();
        polluted.provenance.expected_model_contract_sha256 = Some(bytes_sha256(b"polluted"));
        assert!(verify_differential_report(&polluted).is_err());
        let mut tampered = report;
        tampered.native_incorrectness_claim_permitted = true;
        assert!(verify_differential_report(&tampered).is_err());
    }

    #[test]
    fn failed_run_retains_safe_partial_provenance() {
        let version = version_json(DifferentialOracleTool::Pyannote);
        let failed = shell_program(format!(
            "if [ \"$1\" = \"--franken-whisper-diarization-oracle-version\" ]; then printf '%s' '{version}'; else exit 9; fi"
        ));
        let prepared = PreparedInputs {
            audio_path: PathBuf::from("/dev/null"),
            native: document(),
            reference: None,
            audio_sha256: bytes_sha256(b""),
            native_sha256: bytes_sha256(b"native"),
            reference_sha256: None,
        };
        let report = build_report(
            DifferentialOracleTool::Pyannote,
            &failed,
            &prepared,
            Duration::from_secs(1),
            DifferentialComparisonConfig::default(),
            &CancellationToken::no_deadline(),
        )
        .expect("skip report");
        assert_eq!(report.status, DifferentialRunStatus::Skipped);
        assert_eq!(report.provenance.tool_version.as_deref(), Some("1.2.3"));
        assert_eq!(
            report.provenance.adapter_version.as_deref(),
            Some("adapter-1")
        );
        assert!(
            report
                .provenance
                .executable_sha256
                .as_deref()
                .is_some_and(is_sha256_hex)
        );
        assert!(report.provenance.oracle_stdout_sha256.is_none());
        verify_differential_report(&report).expect("verified skip");
    }
}
