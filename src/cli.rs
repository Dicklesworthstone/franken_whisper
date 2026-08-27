use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use serde_json::json;

use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, BackendParams, DecodingParams, DiarizationConfig, DiarizationEngine,
    DiarizationFallbackPolicy, DiarizationRequest, InputSource, KnownSpeakerInterval, OutputFormat,
    SpeakerCountPriorMass, SpeakerCountRequest, TimestampLevel, TranscribeRequest, VadParams,
};
use crate::sync::ConflictPolicy;

const MAX_SPEAKER_HINTS_BYTES: u64 = 1024 * 1024;

/// Extended version report used by `--version`. Model weights are deliberately
/// called out because they are not bundled with the project binary and may be
/// governed by licenses other than the project license.
pub const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\nproject license: MIT with OpenAI/Anthropic rider\n",
    "model weights: not bundled; third-party model terms apply\n",
    "Sortformer distribution: hash-pinned GitHub release artifact with license and notice"
);

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SpeakerHintsFile {
    Intervals(Vec<KnownSpeakerInterval>),
    Document(SpeakerHintsDocument),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeakerHintsDocument {
    schema_version: String,
    known_intervals: Vec<KnownSpeakerInterval>,
}

fn parse_speaker_hints(bytes: &[u8]) -> FwResult<Vec<KnownSpeakerInterval>> {
    let parsed: SpeakerHintsFile = serde_json::from_slice(bytes).map_err(|_| {
        FwError::InvalidRequest(
            "speaker hints must be valid speaker-hints-v1 JSON without trailing data".to_owned(),
        )
    })?;
    match parsed {
        SpeakerHintsFile::Intervals(intervals) => Ok(intervals),
        SpeakerHintsFile::Document(SpeakerHintsDocument {
            schema_version,
            known_intervals,
        }) if schema_version == "speaker-hints-v1" => Ok(known_intervals),
        SpeakerHintsFile::Document(_) => Err(FwError::InvalidRequest(
            "speaker hints document must use schema_version speaker-hints-v1".to_owned(),
        )),
    }
}

fn read_speaker_hints(path: &Path) -> FwResult<Vec<KnownSpeakerInterval>> {
    let path_metadata = std::fs::symlink_metadata(path).map_err(|_| {
        FwError::InvalidRequest("speaker hints file could not be inspected".to_owned())
    })?;
    if metadata_is_indirection(&path_metadata) || !path_metadata.is_file() {
        return Err(FwError::InvalidRequest(
            "speaker hints must be a regular non-symlink file".to_owned(),
        ));
    }
    #[cfg(target_family = "unix")]
    let file = {
        use rustix::fs::{Mode, OFlags, open};

        let descriptor = open(
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(|_| {
            FwError::InvalidRequest("speaker hints file could not be opened".to_owned())
        })?;
        std::fs::File::from(descriptor)
    };
    #[cfg(windows)]
    let file = {
        use std::os::windows::fs::OpenOptionsExt as _;

        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)
            .map_err(|_| {
                FwError::InvalidRequest("speaker hints file could not be opened".to_owned())
            })?
    };
    #[cfg(not(any(target_family = "unix", windows)))]
    let file = std::fs::File::open(path).map_err(|_| {
        FwError::InvalidRequest("speaker hints file could not be opened".to_owned())
    })?;
    let descriptor_metadata = file.metadata().map_err(|_| {
        FwError::InvalidRequest("speaker hints file metadata could not be read".to_owned())
    })?;
    if !descriptor_metadata.is_file() {
        return Err(FwError::InvalidRequest(
            "speaker hints must be a regular file".to_owned(),
        ));
    }
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::MetadataExt as _;

        if path_metadata.dev() != descriptor_metadata.dev()
            || path_metadata.ino() != descriptor_metadata.ino()
        {
            return Err(FwError::InvalidRequest(
                "speaker hints file changed while being opened".to_owned(),
            ));
        }
    }
    #[cfg(windows)]
    if metadata_is_indirection(&descriptor_metadata) {
        return Err(FwError::InvalidRequest(
            "speaker hints must be a regular non-symlink file".to_owned(),
        ));
    }
    if descriptor_metadata.len() > MAX_SPEAKER_HINTS_BYTES {
        return Err(FwError::InvalidRequest(
            "speaker hints file exceeds the 1 MiB safety limit".to_owned(),
        ));
    }

    let mut bytes = Vec::new();
    file.take(MAX_SPEAKER_HINTS_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| FwError::InvalidRequest("speaker hints file could not be read".to_owned()))?;
    if bytes.len() as u64 > MAX_SPEAKER_HINTS_BYTES {
        return Err(FwError::InvalidRequest(
            "speaker hints file exceeds the 1 MiB safety limit".to_owned(),
        ));
    }
    parse_speaker_hints(&bytes)
}

fn metadata_is_indirection(metadata: &std::fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        return metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0;
    }
    #[cfg(not(windows))]
    false
}

fn parse_speaker_count_range(value: &str) -> FwResult<(u32, u32)> {
    let Some((minimum, maximum)) = value.split_once("..") else {
        return Err(FwError::InvalidRequest(
            "--speaker-count-range must use MIN..MAX".to_owned(),
        ));
    };
    if maximum.contains("..") {
        return Err(FwError::InvalidRequest(
            "--speaker-count-range must contain exactly one `..` separator".to_owned(),
        ));
    }
    let minimum = minimum.trim().parse::<u32>().map_err(|_| {
        FwError::InvalidRequest("--speaker-count-range MIN must be an unsigned integer".to_owned())
    })?;
    let maximum = maximum.trim().parse::<u32>().map_err(|_| {
        FwError::InvalidRequest("--speaker-count-range MAX must be an unsigned integer".to_owned())
    })?;
    Ok((minimum, maximum))
}

fn parse_speaker_count_prior(value: &str) -> FwResult<Vec<SpeakerCountPriorMass>> {
    let value = value.trim();
    if value.is_empty() {
        return Err(FwError::InvalidRequest(
            "--speaker-count-prior must not be empty".to_owned(),
        ));
    }
    if !value.contains('=') && !value.contains(',') {
        let count = value.parse::<u32>().map_err(|_| {
            FwError::InvalidRequest(
                "--speaker-count-prior point mass must be an unsigned integer".to_owned(),
            )
        })?;
        return Ok(vec![SpeakerCountPriorMass {
            count,
            probability: 1.0,
        }]);
    }

    let mut bins = Vec::new();
    for entry in value.split(',') {
        let Some((count, probability)) = entry.split_once('=') else {
            return Err(FwError::InvalidRequest(
                "--speaker-count-prior distribution must use K=P entries".to_owned(),
            ));
        };
        if probability.contains('=') {
            return Err(FwError::InvalidRequest(
                "--speaker-count-prior entries must contain exactly one `=` separator".to_owned(),
            ));
        }
        let count = count.trim().parse::<u32>().map_err(|_| {
            FwError::InvalidRequest(
                "--speaker-count-prior counts must be unsigned integers".to_owned(),
            )
        })?;
        let probability = probability.trim().parse::<f64>().map_err(|_| {
            FwError::InvalidRequest(
                "--speaker-count-prior probabilities must be finite decimal numbers".to_owned(),
            )
        })?;
        bins.push(SpeakerCountPriorMass { count, probability });
    }
    bins.sort_by_key(|bin| bin.count);
    Ok(bins)
}

// ---------------------------------------------------------------------------
// bd-38c.6: Graceful Ctrl+C shutdown via asupersync cancellation protocol
// ---------------------------------------------------------------------------

/// Global flag indicating that a shutdown signal has been received.
static SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);

/// Coordinates graceful Ctrl+C shutdown.
///
/// When a signal is received the controller sets a global `AtomicBool`, which
/// pipeline stages can poll via [`ShutdownController::is_shutting_down`].
/// Callers may also register a callback that fires on signal receipt (e.g. to
/// cancel a [`CancellationToken`]).
///
/// # Example
/// ```rust,no_run
/// use franken_whisper::cli::ShutdownController;
/// let _guard = ShutdownController::install(None);
/// // … run pipeline …
/// if ShutdownController::is_shutting_down() {
///     eprintln!("interrupted");
/// }
/// ```
pub struct ShutdownController;

impl ShutdownController {
    /// Install the Ctrl+C signal handler.
    ///
    /// `on_signal` is an optional callback invoked from the signal-handler
    /// context.  The typical use is to cancel a pipeline token:
    ///
    /// ```rust,ignore
    /// ShutdownController::install(Some(Box::new(move || {
    ///     cancellation_token.cancel();
    /// })));
    /// ```
    ///
    /// Returns `Ok(())` on success.  Errors are non-fatal (signal handling is
    /// best-effort), so callers may choose to log and continue.
    pub fn install(on_signal: Option<Box<dyn Fn() + Send + Sync + 'static>>) -> FwResult<()> {
        ctrlc::set_handler(move || {
            // Mark the global flag.
            SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
            tracing::info!("shutdown signal received (Ctrl+C)");

            // Fire the optional callback (e.g. cancel a CancellationToken).
            if let Some(ref cb) = on_signal {
                cb();
            }
        })
        .map_err(|e| FwError::Io(std::io::Error::other(format!("ctrlc handler: {e}"))))?;
        Ok(())
    }

    /// Returns `true` once a Ctrl+C (or programmatic trigger) has been received.
    #[must_use]
    pub fn is_shutting_down() -> bool {
        SHUTDOWN_FLAG.load(Ordering::SeqCst)
    }

    /// Programmatically trigger the shutdown flag (useful for testing and
    /// internal cancel paths).
    pub fn trigger_shutdown() {
        SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
    }

    /// Reset the shutdown flag (for testing only).
    #[cfg(test)]
    pub fn reset() {
        SHUTDOWN_FLAG.store(false, Ordering::SeqCst);
    }

    /// The exit code the binary should use when exiting due to a signal.
    #[must_use]
    pub const fn signal_exit_code() -> i32 {
        130 // Convention: 128 + SIGINT(2)
    }
}

#[derive(Debug, Parser)]
#[command(name = "franken_whisper")]
#[command(about = "Agent-first Rust ASR orchestrator with ffmpeg normalization")]
#[command(version, long_version = LONG_VERSION)]
#[command(
    after_help = "Agent orientation: `fw robot triage`\nMachine contract: `fw capabilities --json`\nModel readiness: `fw models --json`\nInstall both native models: `fw pull all --json`"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Transcribe one local audio/video source with native diarization by default.
    Transcribe(Box<TranscribeArgs>),
    /// Emit stable JSON/NDJSON surfaces intended for software agents.
    #[command(visible_alias = "agent")]
    Robot {
        #[command(subcommand)]
        command: RobotCommand,
    },
    /// Query persisted transcription runs.
    Runs(RunsArgs),
    /// Export or import the SQLite state as an auditable JSONL snapshot.
    Sync {
        #[command(subcommand)]
        command: SyncCommand,
    },
    /// Encode, decode, and control low-bandwidth TTY audio streams.
    #[command(visible_alias = "tty")]
    TtyAudio {
        #[command(subcommand)]
        command: TtyAudioCommand,
    },
    /// Score external confidential references/hypotheses and emit aggregates only.
    #[command(name = "diarization-eval")]
    DiarizationEval(ConfidentialEvaluationArgs),
    /// Inspect public-corpus contracts, or write evidence on Linux/Android/Apple.
    #[command(name = "diarization-corpus")]
    DiarizationCorpus {
        #[command(subcommand)]
        command: PublicCorpusCommand,
    },
    /// Run explicit development-only differential diagnostics against external tools.
    #[command(name = "diarization-oracle")]
    DiarizationOracle {
        #[command(subcommand)]
        command: DifferentialOracleCommand,
    },
    /// Run the explicitly selected, evaluation-only native Streaming Sortformer.
    #[command(name = "sortformer-diarize", visible_alias = "sortformer")]
    SortformerDiarize(SortformerDiarizeArgs),
    /// Internal fresh-process lane worker. This is intentionally absent from help.
    #[command(name = "__comparison-worker", hide = true)]
    ComparisonWorker,
    /// Internal process-tree cancellation probe. This is intentionally absent from help.
    #[command(name = "__comparison-cancel-probe", hide = true)]
    ComparisonCancelProbe(ComparisonCancelProbeArgs),
    /// Launch the optional human-oriented terminal interface.
    Tui,
    /// Download YouTube audio (videos / playlists / a URL file) and
    /// transcribe each into a markdown + JSON pair; or search / enrich the
    /// YouTube catalog as deduped agent-curated JSON.
    #[command(subcommand)]
    Youtube(YoutubeCommand),
    /// Describe stable commands, schemas, features, and error recovery as JSON.
    Capabilities(CapabilitiesArgs),
    /// Report built-in and cached model readiness without downloading.
    Models(ModelsArgs),
    /// Explicitly download and verify a release-bound model package.
    Pull(PullArgs),
    /// Diagnose whether this installation can perform useful work.
    Doctor(DoctorArgs),
    /// Print compact agent integration documentation from the running binary.
    #[command(name = "robot-docs")]
    RobotDocs {
        #[command(subcommand)]
        command: RobotDocsCommand,
    },
}

#[derive(Debug, Args)]
pub struct ComparisonCancelProbeArgs {
    #[arg(long, hide = true)]
    pub descendant: bool,
    #[arg(long, hide = true)]
    pub lease_parent: bool,
    #[arg(long, hide = true)]
    pub root_pid_file: Option<PathBuf>,
    #[arg(long, hide = true)]
    pub descendant_pid_file: Option<PathBuf>,
}

/// Output controls for the machine-discoverable capability catalog.
#[derive(Debug, Args)]
pub struct CapabilitiesArgs {
    /// Emit the complete stable JSON contract instead of a compact summary.
    #[arg(long)]
    pub json: bool,
}

/// Output controls for the model and package registry.
#[derive(Debug, Args)]
pub struct ModelsArgs {
    /// Emit the complete stable JSON registry instead of a compact summary.
    #[arg(long)]
    pub json: bool,
}

/// Model family accepted by the explicit downloader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PullModelArg {
    All,
    Whisper,
    Sortformer,
    /// Fast-lane streaming package for `fw robot listen` (English-only).
    TinyEn,
    /// Fast-lane streaming package for `fw robot listen` (multilingual;
    /// the default fast model when `--language` is unset or non-English).
    Tiny,
}

/// Explicit model-download controls. No other command performs network access.
#[derive(Debug, Args)]
pub struct PullArgs {
    /// Model package to fetch into the per-user verified cache.
    #[arg(value_enum, default_value_t = PullModelArg::All)]
    pub model: PullModelArg,

    /// Emit one stable JSON object and suppress human progress output.
    #[arg(long)]
    pub json: bool,
}

/// Detect-only installation diagnostics. This command never downloads, moves,
/// converts, or modifies model artifacts.
#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Path to the frankensqlite database file to inspect.
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,

    /// Emit one JSON object with no terminal decoration.
    #[arg(long)]
    pub json: bool,

    /// Exit non-zero unless both native default model packages are verified.
    #[arg(long)]
    pub strict: bool,
}

/// Built-in documentation topics for software agents.
#[derive(Debug, Subcommand)]
pub enum RobotDocsCommand {
    /// Print the canonical orientation and recovery guide.
    Guide,
}

/// Explicit native Streaming Sortformer invocation.
#[derive(Args)]
pub struct SortformerDiarizeArgs {
    /// Audio input in any format accepted by the native normalizer or ffmpeg fallback.
    #[arg(long)]
    pub input: PathBuf,

    /// Explicit conversion receipt; omit with --package to use the verified cache.
    #[arg(long, requires = "package")]
    pub receipt: Option<PathBuf>,

    /// Explicit safetensors package; omit with --receipt to use the verified cache.
    #[arg(long, requires = "receipt")]
    pub package: Option<PathBuf>,

    /// Optional speaker-hints-v1 JSON used for lane-to-reference mapping.
    #[arg(long)]
    pub speaker_hints: Option<PathBuf>,
}

impl SortformerDiarizeArgs {
    pub fn load_speaker_hints(&self) -> FwResult<Vec<KnownSpeakerInterval>> {
        self.speaker_hints
            .as_deref()
            .map(read_speaker_hints)
            .transpose()
            .map(Option::unwrap_or_default)
    }
}

impl fmt::Debug for SortformerDiarizeArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SortformerDiarizeArgs")
            .field("input", &"<redacted>")
            .field("receipt", &"<redacted>")
            .field("package", &"<redacted>")
            .field("speaker_hints", &"<redacted>")
            .finish()
    }
}

/// Public-corpus registry, preparation, and aggregate evaluation commands.
#[derive(Debug, Subcommand)]
pub enum PublicCorpusCommand {
    /// Emit the built-in corpus/license/conversion registry as JSON.
    Registry,
    /// Prepare a deterministic external VoxConverse descriptor without copying media.
    PrepareVoxconverse(PublicCorpusVoxconversePrepareArgs),
    /// Build a path-free bundle (artifact writing requires Linux/Android/Apple).
    Build(PublicCorpusBuildArgs),
    /// Run frozen ablations (artifact writing requires Linux/Android/Apple).
    Ablate(PublicCorpusAblationArgs),
    /// Run the sidecar study (artifact writing requires Linux/Android/Apple).
    SidecarStudy(PublicCorpusSidecarStudyArgs),
    /// Compare diarizers (artifact writing requires Linux/Android/Apple).
    CompareModels(PublicCorpusModelComparisonArgs),
}

/// Arguments for native VoxConverse descriptor preparation.
#[derive(Args)]
pub struct PublicCorpusVoxconversePrepareArgs {
    /// Absolute external root containing all selected inputs and the new descriptor.
    #[arg(long)]
    pub input_root: PathBuf,

    /// Absolute directory containing the official development WAV files.
    #[arg(long)]
    pub development_audio_root: PathBuf,

    /// Absolute directory containing the official test WAV files.
    #[arg(long)]
    pub test_audio_root: PathBuf,

    /// Absolute root containing the official `dev/` and `test/` RTTM directories.
    #[arg(long)]
    pub annotation_root: PathBuf,

    /// New absolute descriptor path beneath `--input-root`.
    #[arg(long)]
    pub output: PathBuf,

    /// Immutable, path-free upstream version identity recorded in the descriptor.
    #[arg(long)]
    pub source_version: String,

    /// Exact acknowledgement ID emitted by `diarization-corpus registry`.
    #[arg(long)]
    pub license_ack: String,
}

impl fmt::Debug for PublicCorpusVoxconversePrepareArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicCorpusVoxconversePrepareArgs")
            .field("input_root", &"<redacted>")
            .field("development_audio_root", &"<redacted>")
            .field("test_audio_root", &"<redacted>")
            .field("annotation_root", &"<redacted>")
            .field("output", &"<redacted>")
            .field("source_version", &"<redacted>")
            .field("license_ack", &self.license_ack)
            .finish()
    }
}

/// Developer-only external differential-diagnostic commands.
#[derive(Debug, Subcommand)]
pub enum DifferentialOracleCommand {
    /// Emit the path-free external adapter registry as JSON.
    Registry,
    /// Run one external adapter and compare its transcript-free stages.
    Run(DifferentialOracleArgs),
}

/// External tool selected for one development-only diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DifferentialOracleToolArg {
    Pyannote,
    NemoSpectral,
    Vbx,
    Eend,
    Diaper,
    Sortformer,
}

impl From<DifferentialOracleToolArg> for crate::differential_oracle::DifferentialOracleTool {
    fn from(value: DifferentialOracleToolArg) -> Self {
        match value {
            DifferentialOracleToolArg::Pyannote => Self::Pyannote,
            DifferentialOracleToolArg::NemoSpectral => Self::NemoSpectral,
            DifferentialOracleToolArg::Vbx => Self::Vbx,
            DifferentialOracleToolArg::Eend => Self::Eend,
            DifferentialOracleToolArg::Diaper => Self::Diaper,
            DifferentialOracleToolArg::Sortformer => Self::Sortformer,
        }
    }
}

/// Arguments for an external, path-free differential diagnostic.
#[derive(Args)]
pub struct DifferentialOracleArgs {
    /// Operator-installed adapter family to probe.
    #[arg(long, value_enum)]
    pub tool: DifferentialOracleToolArg,

    /// Absolute external audio path; bytes and path are never retained.
    #[arg(long)]
    pub audio: PathBuf,

    /// Absolute external native stage-document path.
    #[arg(long)]
    pub native: PathBuf,

    /// Optional absolute external reference stage-document path.
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// New absolute report path outside the project tree.
    #[arg(long)]
    pub output: PathBuf,

    /// Hard limit for the external adapter run.
    #[arg(long, default_value_t = 1_800)]
    pub timeout_seconds: u64,
}

impl fmt::Debug for DifferentialOracleArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DifferentialOracleArgs")
            .field("tool", &self.tool)
            .field("audio", &"<redacted>")
            .field("native", &"<redacted>")
            .field("reference", &self.reference.as_ref().map(|_| "<redacted>"))
            .field("output", &"<redacted>")
            .field("timeout_seconds", &self.timeout_seconds)
            .finish()
    }
}

/// Arguments for external public-corpus preparation.
#[derive(Args)]
pub struct PublicCorpusBuildArgs {
    /// Absolute external root containing the selected public or licensed inputs.
    #[arg(long)]
    pub input_root: PathBuf,

    /// Absolute path to the external path-bearing corpus descriptor.
    #[arg(long)]
    pub descriptor: PathBuf,

    /// New absolute JSON bundle path outside both the checkout and input root.
    #[arg(long)]
    pub output: PathBuf,

    /// Exact acknowledgement ID emitted by `diarization-corpus registry`.
    #[arg(long)]
    pub license_ack: String,
}

impl fmt::Debug for PublicCorpusBuildArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicCorpusBuildArgs")
            .field("input_root", &"<redacted>")
            .field("descriptor", &"<redacted>")
            .field("output", &"<redacted>")
            .field("license_ack", &self.license_ack)
            .finish()
    }
}

/// Arguments for an external, aggregate-only public feature ablation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PublicCorpusEvaluationStageArg {
    Development,
    Certification,
}

impl From<PublicCorpusEvaluationStageArg> for crate::public_corpus::PublicCorpusEvaluationStage {
    fn from(value: PublicCorpusEvaluationStageArg) -> Self {
        match value {
            PublicCorpusEvaluationStageArg::Development => Self::Development,
            PublicCorpusEvaluationStageArg::Certification => Self::Certification,
        }
    }
}

/// Arguments for an external, aggregate-only public feature ablation.
#[derive(Args)]
pub struct PublicCorpusAblationArgs {
    /// Absolute external root containing selected public inputs.
    #[arg(long)]
    pub input_root: PathBuf,

    /// Absolute path to the external path-bearing corpus descriptor.
    #[arg(long)]
    pub descriptor: PathBuf,

    /// New absolute path for the path-free validated corpus bundle.
    #[arg(long)]
    pub bundle_output: PathBuf,

    /// New absolute path for aggregate-only ablation evidence.
    #[arg(long)]
    pub output: PathBuf,

    /// Exact acknowledgement ID emitted by `diarization-corpus registry`.
    #[arg(long)]
    pub license_ack: String,

    /// Deterministically score only each recording prefix of this duration.
    #[arg(long)]
    pub maximum_recording_duration_ms: Option<u64>,

    /// Development tuning or hash-locked unseen certification.
    #[arg(long, value_enum)]
    pub stage: PublicCorpusEvaluationStageArg,

    /// Existing development evidence required only for certification.
    #[arg(long)]
    pub locked_development_evidence: Option<PathBuf>,
}

impl fmt::Debug for PublicCorpusAblationArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicCorpusAblationArgs")
            .field("input_root", &"<redacted>")
            .field("descriptor", &"<redacted>")
            .field("bundle_output", &"<redacted>")
            .field("output", &"<redacted>")
            .field("license_ack", &self.license_ack)
            .field(
                "maximum_recording_duration_ms",
                &self.maximum_recording_duration_ms,
            )
            .field("stage", &self.stage)
            .field("locked_development_evidence", &"<redacted>")
            .finish()
    }
}

/// Arguments for the external, aggregate-only acoustic sidecar study.
#[derive(Args)]
pub struct PublicCorpusSidecarStudyArgs {
    /// Absolute external root containing selected public inputs.
    #[arg(long)]
    pub input_root: PathBuf,

    /// Absolute path to the external path-bearing corpus descriptor.
    #[arg(long)]
    pub descriptor: PathBuf,

    /// New absolute path for the path-free validated corpus bundle.
    #[arg(long)]
    pub bundle_output: PathBuf,

    /// New absolute path for aggregate-only sidecar-study evidence.
    #[arg(long)]
    pub output: PathBuf,

    /// Exact acknowledgement ID emitted by `diarization-corpus registry`.
    #[arg(long)]
    pub license_ack: String,

    /// Deterministically score only each recording prefix of this duration.
    #[arg(long)]
    pub maximum_recording_duration_ms: Option<u64>,

    /// Development tuning or hash-locked unseen certification.
    #[arg(long, value_enum)]
    pub stage: PublicCorpusEvaluationStageArg,

    /// Existing sidecar development evidence required only for certification.
    #[arg(long)]
    pub locked_development_evidence: Option<PathBuf>,
}

impl fmt::Debug for PublicCorpusSidecarStudyArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicCorpusSidecarStudyArgs")
            .field("input_root", &"<redacted>")
            .field("descriptor", &"<redacted>")
            .field("bundle_output", &"<redacted>")
            .field("output", &"<redacted>")
            .field("license_ack", &self.license_ack)
            .field(
                "maximum_recording_duration_ms",
                &self.maximum_recording_duration_ms,
            )
            .field("stage", &self.stage)
            .field(
                "locked_development_evidence",
                &self
                    .locked_development_evidence
                    .as_ref()
                    .map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Arguments for the aggregate-only learned-versus-native comparison.
#[derive(Args)]
pub struct PublicCorpusModelComparisonArgs {
    /// Absolute external root containing selected public inputs.
    #[arg(long)]
    pub input_root: PathBuf,

    /// Absolute path to the external path-bearing corpus descriptor.
    #[arg(long)]
    pub descriptor: PathBuf,

    /// New absolute path for the path-free validated corpus bundle.
    #[arg(long)]
    pub bundle_output: PathBuf,

    /// New absolute path for aggregate-only model-comparison evidence.
    #[arg(long)]
    pub output: PathBuf,

    /// Exact acknowledgement ID emitted by `diarization-corpus registry`.
    #[arg(long)]
    pub license_ack: String,
}

impl fmt::Debug for PublicCorpusModelComparisonArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicCorpusModelComparisonArgs")
            .field("input_root", &"<redacted>")
            .field("descriptor", &"<redacted>")
            .field("bundle_output", &"<redacted>")
            .field("output", &"<redacted>")
            .field("license_ack", &self.license_ack)
            .finish()
    }
}

/// Arguments for the local-only confidential diarization evaluator.
#[derive(Args)]
pub struct ConfidentialEvaluationArgs {
    /// Absolute external root containing every private source file.
    #[arg(long)]
    pub input_root: PathBuf,

    /// Absolute path to the path-bearing local evaluation manifest.
    #[arg(long)]
    pub manifest: PathBuf,

    /// New absolute JSON path outside the project tree for aggregate output.
    #[arg(long)]
    pub output: PathBuf,
}

impl fmt::Debug for ConfidentialEvaluationArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConfidentialEvaluationArgs")
            .field("input_root", &"<redacted>")
            .field("manifest", &"<redacted>")
            .field("output", &"<redacted>")
            .finish()
    }
}

/// `fw youtube` modes: the ingestion pipeline, or catalog search / enrich.
#[derive(Debug, Subcommand)]
pub enum YoutubeCommand {
    /// Download YouTube audio (videos / playlists / a URL file) and
    /// transcribe each into a markdown + JSON pair.
    Run(Box<YoutubeArgs>),
    /// Search YouTube via yt-dlp and emit deduplicated JSON hits on stdout
    /// (bd-m7fv). Enriched per-video metadata by default; `--flat` keeps the
    /// flat-playlist subset for cheap large sweeps.
    Search(YoutubeSearchArgs),
    /// Enrich specific video URLs or ids into deduplicated JSON hits
    /// (bd-m7fv). Playlist inputs are rejected — ingest those through
    /// `youtube run`.
    Enrich(YoutubeEnrichArgs),
}

/// Arguments for `fw youtube search`.
#[derive(Debug, Args)]
pub struct YoutubeSearchArgs {
    /// Free-text search query.
    #[arg(value_name = "QUERY")]
    pub query: String,

    /// Maximum number of results to return.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,

    /// Keep only the flat-playlist field subset (cheaper large sweeps).
    #[arg(long)]
    pub flat: bool,

    /// Explicit yt-dlp binary override (same resolution as ingestion).
    #[arg(long)]
    pub ytdlp: Option<PathBuf>,
}

/// Arguments for `fw youtube enrich`.
#[derive(Debug, Args)]
pub struct YoutubeEnrichArgs {
    /// Video URLs or ids to enrich (positional; duplicates are dropped).
    #[arg(value_name = "URL_OR_ID")]
    pub targets: Vec<String>,

    /// Explicit yt-dlp binary override (same resolution as ingestion).
    #[arg(long)]
    pub ytdlp: Option<PathBuf>,
}

/// Run `fw youtube search`: probe yt-dlp, run the query, print deduped hits
/// as a JSON array on stdout. Deterministic: first-seen order, no decoration.
///
/// # Errors
///
/// Propagates the yt-dlp probe and search errors.
pub fn run_youtube_search(args: &YoutubeSearchArgs) -> FwResult<()> {
    let info = match args.ytdlp.as_deref() {
        Some(path) => {
            crate::youtube::ytdlp::probe_with_path(path, chrono::Utc::now().date_naive())?
        }
        None => crate::youtube::ytdlp::probe()?,
    };
    let token = crate::orchestrator::CancellationToken::unbounded();
    let hits = crate::youtube::ytdlp::search(&info, &args.query, args.limit, args.flat, &token)?;
    println!("{}", serde_json::to_string_pretty(&hits)?);
    Ok(())
}

/// Run `fw youtube enrich`: probe yt-dlp, fetch each target's metadata, and
/// print the deduplicated hits as a JSON array on stdout.
///
/// # Errors
///
/// Propagates the yt-dlp probe, classification, and fetch errors.
pub fn run_youtube_enrich(args: &YoutubeEnrichArgs) -> FwResult<()> {
    if args.targets.is_empty() {
        return Err(FwError::InvalidRequest(
            "no targets: pass one or more video URLs or ids".to_owned(),
        ));
    }
    let info = match args.ytdlp.as_deref() {
        Some(path) => {
            crate::youtube::ytdlp::probe_with_path(path, chrono::Utc::now().date_naive())?
        }
        None => crate::youtube::ytdlp::probe()?,
    };
    let token = crate::orchestrator::CancellationToken::unbounded();
    let hits = crate::youtube::ytdlp::enrich(&info, &args.targets, &token)?;
    println!("{}", serde_json::to_string_pretty(&hits)?);
    Ok(())
}

/// Arguments for the `youtube` subcommand.
#[derive(Debug, Args)]
pub struct YoutubeArgs {
    /// Video or playlist URLs (also accepted as positional args).
    #[arg(long = "url")]
    pub urls: Vec<String>,

    /// Positional video / playlist URLs.
    #[arg(value_name = "URL")]
    pub positional_urls: Vec<String>,

    /// A file with one URL per line (`#`/`;`/`]` comments and blanks ignored).
    #[arg(long)]
    pub batch_file: Option<PathBuf>,

    /// Output directory for the audio/, markdown, and JSON files.
    #[arg(long, default_value = "youtube_transcripts")]
    pub output_dir: PathBuf,

    /// Model name or path forwarded to the engine.
    #[arg(long)]
    pub model: Option<String>,

    /// Language hint (ISO 639-1); omitted = auto-detect.
    #[arg(long)]
    pub language: Option<String>,

    /// Backend strategy.
    #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
    pub backend: BackendKind,

    /// Explicitly enable speaker diarization (already enabled by default).
    #[arg(long)]
    pub diarize: bool,

    /// Disable the default native speaker diarization stage.
    #[arg(long, conflicts_with = "diarize")]
    pub no_diarize: bool,

    /// Maximum concurrent downloads.
    #[arg(long, default_value_t = 3)]
    pub concurrency: usize,

    /// Delete each audio file after its transcript is written.
    #[arg(long)]
    pub no_keep_audio: bool,

    /// Do not retry videos previously marked failed in the manifest.
    #[arg(long)]
    pub no_retry: bool,

    /// Stop scheduling later waves after a per-video failure.
    #[arg(long)]
    pub abort_on_error: bool,

    /// Emit the final run summary as JSON on stdout (for scripting).
    #[arg(long)]
    pub json_summary: bool,

    /// bd-lun9 batch-wave size: download in waves of this many videos so
    /// untranscribed audio cannot pile up on disk. 0 = all at once.
    #[arg(long, default_value_t = 0)]
    pub batch_size: usize,

    /// Filename style for emitted artifacts (bd-tchp default: slug).
    #[arg(long, value_enum, default_value_t = YoutubeNamingStyleArg::Slug)]
    pub naming_style: YoutubeNamingStyleArg,

    /// Stream per-video NDJSON robot events on stdout (schema 1.1.0,
    /// terminal `youtube.run_complete`). Mutually exclusive with
    /// `--json-summary`: the stream replaces the single-blob output.
    #[arg(long, conflicts_with = "json_summary")]
    pub robot: bool,
}

/// CLI-facing naming style selector (maps onto
/// [`franken_whisper::youtube::naming::NamingStyle`]; kept separate so the
/// library module stays clap-free).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum YoutubeNamingStyleArg {
    /// Lowercase underscore slug (default): `20240115_my_title_dQw4w9WgXcQ`.
    Slug,
    /// Human-readable `{date} - {title} [{id}]`, non-ASCII kept.
    Pretty,
    /// Human-readable with non-ASCII folded to `_`.
    Ascii,
}
impl From<YoutubeNamingStyleArg> for crate::youtube::naming::NamingStyle {
    fn from(value: YoutubeNamingStyleArg) -> Self {
        match value {
            YoutubeNamingStyleArg::Slug => crate::youtube::naming::NamingStyle::Slug,
            YoutubeNamingStyleArg::Pretty => crate::youtube::naming::NamingStyle::Pretty,
            YoutubeNamingStyleArg::Ascii => crate::youtube::naming::NamingStyle::Ascii,
        }
    }
}

impl YoutubeArgs {
    /// Build the pipeline options, validating that at least one input source
    /// was supplied. Positional and `--url` URLs are merged.
    pub fn to_options(&self) -> FwResult<crate::youtube::pipeline::YoutubeRunOptions> {
        let mut urls = self.urls.clone();
        urls.extend(self.positional_urls.iter().cloned());
        if urls.is_empty() && self.batch_file.is_none() {
            return Err(FwError::InvalidRequest(
                "no inputs: pass one or more URLs, --url, or --batch-file".to_owned(),
            ));
        }
        if self.concurrency == 0 {
            return Err(FwError::InvalidRequest(
                "--concurrency must be at least 1".to_owned(),
            ));
        }
        Ok(crate::youtube::pipeline::YoutubeRunOptions {
            urls,
            batch_file: self.batch_file.clone(),
            output_dir: self.output_dir.clone(),
            model: self.model.clone(),
            language: self.language.clone(),
            backend: self.backend,
            diarize: self.diarize || !self.no_diarize,
            concurrency: self.concurrency,
            keep_audio: !self.no_keep_audio,
            retry_failed: !self.no_retry,
            abort_on_error: self.abort_on_error,
            naming_style: self.naming_style.into(),
            batch_size: self.batch_size,
            robot_events: if self.robot {
                crate::youtube::pipeline::YoutubeRobotEvents::Stdout
            } else {
                crate::youtube::pipeline::YoutubeRobotEvents::Off
            },
        })
    }
}

#[derive(Debug, Subcommand)]
pub enum RobotCommand {
    /// Run a transcription pipeline and stream NDJSON events.
    Run(Box<TranscribeArgs>),
    /// Emit the stable robot event schema as one JSON line.
    Schema,
    /// Discover backend capabilities and live availability.
    Backends,
    /// Probe dependencies and runtime resources.
    Health(HealthArgs),
    /// Orient an agent in one round trip with state-aware next commands.
    Triage(HealthArgs),
    /// Query persisted adaptive-routing decisions as NDJSON.
    RoutingHistory(RoutingHistoryArgs),
    /// Live microphone/pipe transcription: continuous NDJSON session events
    /// (schema 1.1.0 listen family). See docs/realtime-streaming.md.
    Listen(Box<ListenArgs>),
}

#[derive(Debug, Args)]
pub struct HealthArgs {
    /// Path to frankensqlite database file.
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,

    /// Exit non-zero when the report is not fully healthy.
    #[arg(long)]
    pub strict: bool,
}

#[derive(Debug, Args)]
pub struct RoutingHistoryArgs {
    /// Path to frankensqlite database file.
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,

    /// Filter to a specific run by ID.
    #[arg(long)]
    pub run_id: Option<String>,

    /// Maximum number of recent runs to scan.
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
}

/// Audio source for `fw robot listen`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ListenSourceArg {
    /// Live microphone (cpal primary, ffmpeg fallback).
    Mic,
    /// Raw PCM piped on stdin (`ffmpeg -i X -f s16le - | fw robot listen --source stdin-pcm`).
    StdinPcm,
    /// Replay an audio file (WAV in v1), optionally paced to real time.
    FileReplay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CaptureBackendArg {
    Auto,
    Cpal,
    Ffmpeg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ListenPolicyArg {
    /// AlignAtt (default): attention-gated incremental commits — deltas
    /// stream mid-utterance as soon as the decoder's alignment heads stop
    /// attending near the live audio edge (bd-rt-alignatt-fry9).
    Alignatt,
    /// Bootstrap baseline: partials every step, one committed delta at
    /// utterance close.
    EndpointCommit,
    /// Fallback (bd-rt-local-agreement-l5x8): commit only what two
    /// consecutive decodes agree on. Model-agnostic insurance baseline;
    /// never the default.
    LocalAgreement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StdinFormatArg {
    S16le,
    F32le,
}

/// Live listen session controls (bd-rt-listen-cmd-i48i). The consolidated
/// flag list lives on the driver bead; `--quality-model`/confirm controls
/// landed with bd-rt-confirm-lane-3okr; `--db` lands with the persistence
/// bead (bd-rt-persist-a66y).
#[derive(Debug, Args)]
pub struct ListenArgs {
    /// Audio source.
    #[arg(long, value_enum, default_value_t = ListenSourceArg::Mic)]
    pub source: ListenSourceArg,

    /// Input file for `--source file-replay`.
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Pace file-replay at real time (default: as fast as possible).
    #[arg(long)]
    pub realtime_pace: bool,

    /// Capture backend for `--source mic`.
    #[arg(long, value_enum, default_value_t = CaptureBackendArg::Auto)]
    pub capture_backend: CaptureBackendArg,

    /// Input device name (default: system default input).
    #[arg(long)]
    pub mic_device: Option<String>,

    /// Fast-lane model override (default: tiny.en for --language en,
    /// multilingual tiny otherwise; missing packages fall back to turbo
    /// with a warning).
    #[arg(long)]
    pub fast_model: Option<String>,

    /// Emission policy.
    #[arg(long, value_enum, default_value_t = ListenPolicyArg::Alignatt)]
    pub policy: ListenPolicyArg,

    /// AlignAtt holdback: how far (ms) the decoder's attention must sit
    /// behind the live audio edge before text commits. Lower = faster,
    /// riskier; higher = safer, laggier.
    #[arg(long, default_value_t = 200)]
    pub alignatt_holdback_ms: u64,

    /// Step decode cadence in milliseconds.
    #[arg(long, default_value_t = 300)]
    pub step_ms: u64,

    /// bd-rt-adaptive-contract-yw68: enable the two adaptive controllers
    /// (step cadence + AlignAtt holdback) under the alien-artifact
    /// contract. Default OFF; each adapts only its one knob, Brier-gated,
    /// with deterministic fallback to these configured values.
    #[arg(long)]
    pub adaptive: bool,

    /// Rolling session buffer cap in seconds.
    #[arg(long, default_value_t = 12.0)]
    pub max_buffer_sec: f64,

    /// Language hint (ISO 639-1); unset = detect on first speech and pin.
    #[arg(long)]
    pub language: Option<String>,

    /// End the session after this many seconds (0 = unbounded).
    #[arg(long, default_value_t = 0.0)]
    pub max_seconds: f64,

    /// Force-close an utterance after this many seconds of open speech.
    #[arg(long, default_value_t = 90.0)]
    pub max_utterance_sec: f64,

    /// Suppress mutable transcript.partial previews (committed deltas and
    /// lifecycle events are unaffected; the first remedy for slow consumers).
    #[arg(long)]
    pub no_partials: bool,

    /// Periodic session_stats heartbeat interval (0 = final only).
    #[arg(long, default_value_t = 30.0)]
    pub stats_interval_sec: f64,

    /// Disable cross-trim/cross-utterance prompt carry.
    #[arg(long)]
    pub no_context: bool,

    /// Capture ring capacity in seconds (absorbs slow-consumer stalls).
    #[arg(long, default_value_t = 30.0)]
    pub capture_buffer_sec: f64,

    /// stdin-pcm sample rate in Hz.
    #[arg(long, default_value_t = 16_000)]
    pub stdin_rate: u32,

    /// stdin-pcm channel count.
    #[arg(long, default_value_t = 1)]
    pub stdin_channels: u16,

    /// stdin-pcm sample format.
    #[arg(long, value_enum, default_value_t = StdinFormatArg::S16le)]
    pub stdin_format: StdinFormatArg,

    /// Disable VAD gating: one continuous utterance split only by
    /// --max-utterance-sec (harness baselines, known-continuous feeds).
    #[arg(long)]
    pub no_vad: bool,

    /// Energy gate above the running noise floor, in dB.
    #[arg(long, default_value_t = 9.0)]
    pub vad_gate_db: f64,

    /// Sustained voice required before an utterance opens (ms).
    #[arg(long, default_value_t = 250)]
    pub vad_min_speech_ms: u64,

    /// Sustained silence that closes an utterance (ms).
    #[arg(long, default_value_t = 600)]
    pub vad_endpoint_ms: u64,

    /// Confirm-lane quality model (bd-rt-confirm-lane-3okr): `auto`
    /// (default; large-v3-turbo when its package is installed, lane off
    /// otherwise), `none` (disable), or an explicit model spec.
    #[arg(long, default_value = "auto")]
    pub quality_model: String,

    /// Seconds the session end waits for in-flight quality confirms before
    /// abandoning them with a `confirm_drain_timeout` warning.
    #[arg(long, default_value_t = 10.0)]
    pub confirm_drain_sec: f64,

    /// Max unconfirmed utterances in the confirm queue before the oldest is
    /// dropped (`confirm_lag` warning). The live lane never blocks.
    #[arg(long, default_value_t = 4)]
    pub confirm_queue_bound: usize,

    /// Persist the session to SQLite at utterance granularity
    /// (bd-rt-persist-a66y, crash-durable). Disabled by `--no-persist`.
    #[arg(long)]
    pub no_persist: bool,

    /// Database file for run history (same store as batch runs).
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,

    /// List input devices as NDJSON and exit (no session).
    #[arg(long)]
    pub list_devices: bool,
}

#[derive(Debug, Subcommand)]
pub enum SyncCommand {
    #[command(name = "export-jsonl")]
    Export(SyncExportArgs),
    #[command(name = "import-jsonl")]
    Import(SyncImportArgs),
}

#[derive(Debug, Args)]
pub struct SyncExportArgs {
    /// Path to frankensqlite database file.
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,

    /// Output directory for JSONL snapshot.
    #[arg(long)]
    pub output: PathBuf,

    /// State root for lock files.
    #[arg(long, default_value = ".franken_whisper")]
    pub state_root: PathBuf,
}

#[derive(Debug, Args)]
pub struct SyncImportArgs {
    /// Path to frankensqlite database file.
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,

    /// Directory containing JSONL snapshot to import.
    #[arg(long)]
    pub input: PathBuf,

    /// State root for lock files.
    #[arg(long, default_value = ".franken_whisper")]
    pub state_root: PathBuf,

    /// Conflict resolution policy.
    #[arg(long, value_enum, default_value_t = ConflictPolicy::Reject)]
    pub conflict_policy: ConflictPolicy,
}

#[derive(Debug, Clone, Args)]
pub struct TranscribeArgs {
    /// Path to input audio/video file.
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Read audio bytes from stdin.
    #[arg(long)]
    pub stdin: bool,

    /// Capture from microphone/line-in via ffmpeg.
    #[arg(long)]
    pub mic: bool,

    /// Recording length when --mic is used.
    #[arg(long, default_value_t = 15)]
    pub mic_seconds: u32,

    /// Device string for microphone capture (OS-specific).
    #[arg(long)]
    pub mic_device: Option<String>,

    /// Explicit ffmpeg input format for mic capture (advanced).
    #[arg(long)]
    pub mic_ffmpeg_format: Option<String>,

    /// Explicit ffmpeg input source for mic capture (advanced).
    #[arg(long)]
    pub mic_ffmpeg_source: Option<String>,

    /// Backend strategy.
    #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
    pub backend: BackendKind,

    /// Backend model hint (forwarded where supported).
    #[arg(long)]
    pub model: Option<String>,

    /// Language hint (e.g., en, es).
    #[arg(long)]
    pub language: Option<String>,

    /// Request translation to English when backend supports it.
    #[arg(long)]
    pub translate: bool,

    /// Explicitly request speaker diarization (already enabled by default).
    #[arg(long)]
    pub diarize: bool,

    /// Disable the default native speaker diarization stage.
    #[arg(long, conflicts_with = "diarize")]
    pub no_diarize: bool,

    /// Speaker diarization implementation.
    #[arg(long, value_enum, default_value_t = DiarizationEngine::Auto)]
    pub diarization_engine: DiarizationEngine,

    /// Conservative action when the requested diarizer lacks evidence.
    #[arg(long, value_enum, default_value_t = DiarizationFallbackPolicy::Acoustic)]
    pub diarization_fallback: DiarizationFallbackPolicy,

    /// Speaker-hints-v1 JSON; its source path is not retained.
    ///
    /// Parsed hint fields become part of the request and may be persisted with
    /// the run unless `--no-persist` is used.
    #[arg(long)]
    pub speaker_hints: Option<PathBuf>,

    /// Remove this many milliseconds from each enrollment interval edge.
    #[arg(long, default_value_t = 100)]
    pub enrollment_edge_guard_ms: u32,

    /// Maximum global native-diarization prototypes (hard ceiling: 512).
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u16).range(1..=512))]
    pub diarization_max_prototypes: u16,

    /// Record consent for future reusable-profile persistence.
    ///
    /// Schema v5 stores privacy-safe summaries only, never raw acoustic vectors.
    #[arg(long)]
    pub persist_speaker_profiles: bool,

    /// Path to frankensqlite database file.
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,

    /// Disable persistence in frankensqlite.
    #[arg(long)]
    pub no_persist: bool,

    /// Pipeline timeout in seconds.
    #[arg(long)]
    pub timeout: Option<u64>,

    /// Print full JSON run report instead of plain transcript.
    #[arg(long)]
    pub json: bool,

    // -- Phase 3: whisper.cpp output format controls --
    /// Also produce plain-text output (whisper.cpp).
    #[arg(long)]
    pub output_txt: bool,

    /// Also produce VTT subtitle output (whisper.cpp).
    #[arg(long)]
    pub output_vtt: bool,

    /// Also produce SRT subtitle output (whisper.cpp).
    #[arg(long)]
    pub output_srt: bool,

    /// Also produce CSV output (whisper.cpp).
    #[arg(long)]
    pub output_csv: bool,

    /// Produce extended JSON output with full metadata (whisper.cpp).
    #[arg(long)]
    pub output_json_full: bool,

    /// Also produce LRC karaoke output (whisper.cpp).
    #[arg(long)]
    pub output_lrc: bool,

    // -- Phase 3: whisper.cpp inference controls --
    /// Suppress timestamps in output (whisper.cpp).
    #[arg(long)]
    pub no_timestamps: bool,

    /// Detect language only then exit (whisper.cpp).
    #[arg(long)]
    pub detect_language_only: bool,

    /// Split on word boundaries instead of tokens (whisper.cpp).
    #[arg(long)]
    pub split_on_word: bool,

    /// Best-of sampling count (whisper.cpp).
    #[arg(long)]
    pub best_of: Option<u32>,

    /// Beam search width (whisper.cpp).
    #[arg(long)]
    pub beam_size: Option<u32>,

    /// Max text context tokens, -1 for unlimited (whisper.cpp).
    #[arg(long)]
    pub max_context: Option<i32>,

    /// Max segment length in characters (whisper.cpp).
    #[arg(long)]
    pub max_segment_length: Option<u32>,

    /// Sampling temperature (whisper.cpp).
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Temperature increment on fallback (whisper.cpp).
    #[arg(long)]
    pub temperature_increment: Option<f32>,

    /// Entropy threshold for decoder (whisper.cpp).
    #[arg(long)]
    pub entropy_threshold: Option<f32>,

    /// Log-prob threshold for decoder (whisper.cpp).
    #[arg(long)]
    pub logprob_threshold: Option<f32>,

    /// No-speech probability threshold (whisper.cpp).
    #[arg(long)]
    pub no_speech_threshold: Option<f32>,

    // -- Phase 3: whisper.cpp VAD controls --
    /// Enable Voice Activity Detection (whisper.cpp).
    #[arg(long)]
    pub vad: bool,

    /// VAD model path (whisper.cpp).
    #[arg(long)]
    pub vad_model: Option<PathBuf>,

    /// VAD speech probability threshold (whisper.cpp).
    #[arg(long)]
    pub vad_threshold: Option<f32>,

    /// VAD minimum speech duration in ms (whisper.cpp).
    #[arg(long)]
    pub vad_min_speech_ms: Option<u32>,

    /// VAD minimum silence duration in ms (whisper.cpp).
    #[arg(long)]
    pub vad_min_silence_ms: Option<u32>,

    /// VAD maximum speech duration in seconds (whisper.cpp).
    #[arg(long)]
    pub vad_max_speech_s: Option<f32>,

    /// VAD speech padding in ms (whisper.cpp).
    #[arg(long)]
    pub vad_speech_pad_ms: Option<u32>,

    /// VAD samples overlap factor (whisper.cpp).
    #[arg(long)]
    pub vad_samples_overlap: Option<f32>,

    // -- Phase 4: whisper.cpp threading, GPU, prompt, audio windowing --
    /// Number of threads for computation (whisper.cpp).
    #[arg(long)]
    pub threads: Option<u32>,

    /// Number of processors for parallel processing (whisper.cpp).
    #[arg(long)]
    pub processors: Option<u32>,

    /// Disable GPU acceleration (whisper.cpp).
    #[arg(long)]
    pub no_gpu: bool,

    /// Initial text prompt for biasing transcription (whisper.cpp).
    #[arg(long)]
    pub prompt: Option<String>,

    /// Always prepend initial prompt to every segment (whisper.cpp).
    #[arg(long)]
    pub carry_initial_prompt: bool,

    /// Disable temperature fallback during decoding (whisper.cpp).
    #[arg(long)]
    pub no_fallback: bool,

    /// Suppress non-speech tokens (whisper.cpp).
    #[arg(long)]
    pub suppress_nst: bool,

    /// Enable TinyDiarize speaker-turn token injection (whisper.cpp).
    #[arg(long)]
    pub tiny_diarize: bool,

    /// Opt in to rule-based segment-text normalization (sentence-casing and
    /// terminal periods). Off by default so `segments[].text` stays faithful
    /// to the decoded transcript.
    #[arg(long)]
    pub normalize_segment_text: bool,

    /// Time offset in milliseconds to start processing (whisper.cpp).
    #[arg(long)]
    pub offset_ms: Option<u64>,

    /// Duration of audio to process in milliseconds (whisper.cpp).
    #[arg(long)]
    pub duration_ms: Option<u64>,

    /// Audio context size, 0 = all (whisper.cpp).
    #[arg(long)]
    pub audio_ctx: Option<i32>,

    /// Word timestamp probability threshold (whisper.cpp).
    #[arg(long)]
    pub word_threshold: Option<f32>,

    /// Regex pattern to suppress matching tokens (whisper.cpp).
    #[arg(long)]
    pub suppress_regex: Option<String>,

    // -- Phase 3: insanely-fast-whisper controls --
    /// Batch size for parallel inference (insanely-fast, diarization).
    #[arg(long)]
    pub batch_size: Option<u32>,

    /// Timestamp granularity: chunk or word (insanely-fast).
    #[arg(long, value_enum)]
    pub timestamp_level: Option<TimestampLevel>,

    /// Explicit hard speaker count. Unsupported speakers remain UNKNOWN.
    #[arg(
        long,
        value_name = "K",
        conflicts_with_all = ["speaker_count_range", "speaker_count_prior"]
    )]
    pub speaker_count_hard: Option<u32>,

    /// Soft bounded speaker-count preference in MIN..MAX form.
    #[arg(
        long,
        value_name = "MIN..MAX",
        conflicts_with_all = ["speaker_count_hard", "speaker_count_prior"]
    )]
    pub speaker_count_range: Option<String>,

    /// Soft count prior: K for point mass or K=P,K=P for a distribution.
    #[arg(
        long,
        value_name = "PRIOR",
        conflicts_with_all = ["speaker_count_hard", "speaker_count_range"]
    )]
    pub speaker_count_prior: Option<String>,

    /// GPU device identifier, e.g. "0" or "mps" (insanely-fast, diarization).
    #[arg(long)]
    pub gpu_device: Option<String>,

    /// Enable Flash Attention 2 (insanely-fast).
    #[arg(long)]
    pub flash_attention: bool,

    /// HuggingFace token override for insanely-fast diarization.
    #[arg(long)]
    pub hf_token: Option<String>,

    /// Output transcript artifact path override for insanely-fast backend.
    #[arg(long)]
    pub transcript_path: Option<PathBuf>,

    // -- Phase 3: diarization pipeline controls --
    /// Disable source separation / vocal isolation (diarization).
    #[arg(long)]
    pub no_stem: bool,

    /// Override diarization whisper model name.
    #[arg(long)]
    pub diarization_model: Option<String>,

    /// Spell out numbers instead of digits for alignment (diarization).
    #[arg(long)]
    pub suppress_numerals: bool,

    // -- Phase 5: Speculative cancel-correct streaming --
    /// Enable speculative cancel-correct streaming mode.
    /// Runs a fast model for instant results while a quality model
    /// confirms or corrects in parallel.
    #[arg(long)]
    pub speculative: bool,

    /// Fast model for speculative mode (default: auto-select smallest available).
    #[arg(long, requires = "speculative")]
    pub fast_model: Option<String>,

    /// Quality model for speculative mode (default: auto-select largest available).
    #[arg(long, requires = "speculative")]
    pub quality_model: Option<String>,

    /// Initial speculation window size in milliseconds (default: 3000).
    #[arg(long, requires = "speculative", default_value = "3000")]
    pub speculative_window_ms: Option<u64>,

    /// Window overlap in milliseconds for speculative mode (default: 500).
    #[arg(long, requires = "speculative", default_value = "500")]
    pub speculative_overlap_ms: Option<u64>,

    /// Maximum WER tolerance before correction in speculative mode (default: 0.1).
    #[arg(long, requires = "speculative")]
    pub correction_tolerance_wer: Option<f64>,

    /// Disable adaptive window sizing in speculative mode.
    #[arg(long, requires = "speculative")]
    pub no_adaptive: bool,

    /// Force all windows to use quality model result (evaluation mode).
    #[arg(long, requires = "speculative")]
    pub always_correct: bool,
}

#[derive(Debug, Args)]
pub struct RunsArgs {
    /// Path to frankensqlite database file.
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,

    /// Fetch a specific run by ID (prints full JSON details).
    #[arg(long)]
    pub id: Option<String>,

    /// Maximum number of recent runs to list.
    #[arg(long, default_value_t = 20)]
    pub limit: usize,

    /// Output format for list mode.
    #[arg(long, value_enum, default_value_t = RunsOutputFormat::Plain)]
    pub format: RunsOutputFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum RunsOutputFormat {
    Plain,
    Json,
    Ndjson,
}

#[derive(Debug, Subcommand)]
pub enum TtyAudioCommand {
    Encode {
        #[arg(long)]
        input: PathBuf,

        #[arg(long, default_value_t = 200)]
        chunk_ms: u32,
    },
    Decode {
        #[arg(long)]
        output: PathBuf,

        /// Frame recovery policy on gaps/corruption.
        #[arg(long, value_enum, default_value_t = TtyAudioRecoveryPolicy::FailClosed)]
        recovery: TtyAudioRecoveryPolicy,
    },
    #[command(name = "retransmit-plan")]
    RetransmitPlan {
        /// Recovery policy used while scanning frame stream.
        #[arg(long, value_enum, default_value_t = TtyAudioRecoveryPolicy::SkipMissing)]
        recovery: TtyAudioRecoveryPolicy,
    },
    Control {
        #[command(subcommand)]
        command: TtyAudioControlCommand,
    },

    // -- bd-2xe.4: convenience subcommands --
    /// Emit a single control frame by kind (handshake, eof, reset).
    #[command(name = "send-control")]
    SendControl {
        /// The kind of control frame to emit.
        frame_type: ControlFrameKind,
    },

    /// Run the retransmit loop reading frame data from stdin.
    Retransmit {
        /// Recovery policy used while scanning frame stream.
        #[arg(long, value_enum, default_value_t = TtyAudioRecoveryPolicy::SkipMissing)]
        recovery: TtyAudioRecoveryPolicy,

        /// Maximum number of deterministic retransmit request rounds.
        #[arg(long, default_value_t = 1)]
        rounds: u32,
    },
}

// ---------------------------------------------------------------------------
// bd-2xe.4: control frame kind for the send-control convenience command
// ---------------------------------------------------------------------------

/// Simplified control frame kind for the `send-control` convenience command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum ControlFrameKind {
    /// Emit a default handshake control frame.
    Handshake,
    /// Emit an EOF control frame signalling end-of-stream.
    Eof,
    /// Emit a reset control frame requesting stream reset.
    Reset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub enum TtyAudioRecoveryPolicy {
    FailClosed,
    SkipMissing,
}

impl From<TtyAudioRecoveryPolicy> for crate::tty_audio::DecodeRecoveryPolicy {
    fn from(value: TtyAudioRecoveryPolicy) -> Self {
        match value {
            TtyAudioRecoveryPolicy::FailClosed => Self::FailClosed,
            TtyAudioRecoveryPolicy::SkipMissing => Self::SkipMissing,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum TtyAudioControlCommand {
    Handshake {
        #[arg(long, default_value_t = 1)]
        min_version: u32,
        #[arg(long, default_value_t = 1)]
        max_version: u32,
        #[arg(
            long = "codec",
            value_delimiter = ',',
            default_value = "mulaw+zlib+b64"
        )]
        supported_codecs: Vec<String>,
    },
    #[command(name = "handshake-ack")]
    HandshakeAck {
        #[arg(long, default_value_t = 1)]
        negotiated_version: u32,
        #[arg(long, default_value = "mulaw+zlib+b64")]
        negotiated_codec: String,
    },
    Ack {
        #[arg(long)]
        up_to_seq: u64,
    },
    Backpressure {
        #[arg(long)]
        remaining_capacity: u64,
    },
    #[command(name = "retransmit-request")]
    RetransmitRequest {
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        sequences: Vec<u64>,
    },
    #[command(name = "retransmit-response")]
    RetransmitResponse {
        #[arg(long, value_delimiter = ',', num_args = 1..)]
        sequences: Vec<u64>,
    },
    #[command(name = "retransmit-loop")]
    RetransmitLoop {
        /// Recovery policy used while scanning frame stream.
        #[arg(long, value_enum, default_value_t = TtyAudioRecoveryPolicy::SkipMissing)]
        recovery: TtyAudioRecoveryPolicy,
        /// Maximum number of deterministic retransmit request rounds to emit.
        #[arg(long, default_value_t = 1)]
        rounds: u32,
    },
}

impl TranscribeArgs {
    /// Build a request while retaining these CLI arguments.
    ///
    /// Terminal production call sites should prefer [`Self::into_request`] so
    /// owned values can be moved into the request.
    pub fn to_request(&self) -> FwResult<TranscribeRequest> {
        self.clone().into_request()
    }

    /// Consume terminal CLI arguments into a request, moving their owned
    /// strings and paths instead of cloning them immediately before the CLI
    /// object is discarded.
    pub fn into_request(mut self) -> FwResult<TranscribeRequest> {
        let effective_diarize = self.diarize || !self.no_diarize;
        let mut mode_count = 0usize;
        if self.input.is_some() {
            mode_count += 1;
        }
        if self.stdin {
            mode_count += 1;
        }
        if self.mic {
            mode_count += 1;
        }

        if mode_count == 0 {
            return Err(FwError::InvalidRequest(
                "specify one of --input, --stdin, or --mic".to_owned(),
            ));
        }
        if mode_count > 1 {
            return Err(FwError::InvalidRequest(
                "--input, --stdin, and --mic are mutually exclusive".to_owned(),
            ));
        }
        if !effective_diarize
            && (self.speaker_hints.is_some()
                || self.persist_speaker_profiles
                || self.diarization_engine != DiarizationEngine::Auto
                || self.diarization_fallback != DiarizationFallbackPolicy::Acoustic
                || self.speaker_count_hard.is_some()
                || self.speaker_count_range.is_some()
                || self.speaker_count_prior.is_some())
        {
            return Err(FwError::InvalidRequest(
                "diarization controls cannot be combined with --no-diarize".to_owned(),
            ));
        }

        let input = if let Some(path) = self.input.take() {
            InputSource::File { path }
        } else if self.stdin {
            InputSource::Stdin {
                hint_extension: None,
            }
        } else {
            InputSource::Microphone {
                seconds: self.mic_seconds,
                device: self.mic_device.take(),
                ffmpeg_format: self.mic_ffmpeg_format.take(),
                ffmpeg_source: self.mic_ffmpeg_source.take(),
            }
        };

        // Build output format list from individual flags.
        let mut output_formats = Vec::new();
        if self.output_txt {
            output_formats.push(OutputFormat::Txt);
        }
        if self.output_vtt {
            output_formats.push(OutputFormat::Vtt);
        }
        if self.output_srt {
            output_formats.push(OutputFormat::Srt);
        }
        if self.output_csv {
            output_formats.push(OutputFormat::Csv);
        }
        if self.output_json_full {
            output_formats.push(OutputFormat::JsonFull);
        }
        if self.output_lrc {
            output_formats.push(OutputFormat::Lrc);
        }

        // Decoding params — only build if any field is set.
        let decoding = if self.best_of.is_some()
            || self.beam_size.is_some()
            || self.max_context.is_some()
            || self.max_segment_length.is_some()
            || self.temperature.is_some()
            || self.temperature_increment.is_some()
            || self.entropy_threshold.is_some()
            || self.logprob_threshold.is_some()
            || self.no_speech_threshold.is_some()
        {
            Some(DecodingParams {
                best_of: self.best_of,
                beam_size: self.beam_size,
                max_context: self.max_context,
                max_segment_length: self.max_segment_length,
                temperature: self.temperature,
                temperature_increment: self.temperature_increment,
                entropy_threshold: self.entropy_threshold,
                logprob_threshold: self.logprob_threshold,
                no_speech_threshold: self.no_speech_threshold,
            })
        } else {
            None
        };

        // VAD params — only build if --vad flag is set.
        let vad = if self.vad {
            Some(VadParams {
                model_path: self.vad_model.take(),
                threshold: self.vad_threshold,
                min_speech_duration_ms: self.vad_min_speech_ms,
                min_silence_duration_ms: self.vad_min_silence_ms,
                max_speech_duration_s: self.vad_max_speech_s,
                speech_pad_ms: self.vad_speech_pad_ms,
                samples_overlap: self.vad_samples_overlap,
            })
        } else {
            None
        };

        let speaker_count = self.speaker_count_request()?;

        // Diarization config — only build if any diarization-specific flag is set.
        let diarization_config =
            if self.no_stem || self.diarization_model.is_some() || self.suppress_numerals {
                Some(DiarizationConfig {
                    no_stem: self.no_stem,
                    whisper_model: self.diarization_model.take(),
                    suppress_numerals: self.suppress_numerals,
                    device: self.gpu_device.clone(),
                    batch_size: self.batch_size,
                })
            } else {
                None
            };

        let known_intervals = self
            .speaker_hints
            .take()
            .as_deref()
            .map(read_speaker_hints)
            .transpose()?
            .unwrap_or_default();
        let has_diarization_request = effective_diarize
            || !known_intervals.is_empty()
            || speaker_count != SpeakerCountRequest::Infer;
        let acoustic_diarization = has_diarization_request.then_some(DiarizationRequest {
            engine: self.diarization_engine,
            fallback: self.diarization_fallback,
            speaker_count,
            known_intervals,
            enrollment_edge_guard_ms: self.enrollment_edge_guard_ms,
            max_prototypes: self.diarization_max_prototypes,
            persist_profiles: self.persist_speaker_profiles,
        });

        let fast_model = self.fast_model.take();
        let quality_model = self.quality_model.take();
        let speculative = self.speculative_request_with_models(fast_model, quality_model)?;

        let backend_params = BackendParams {
            output_formats,
            timestamp_level: self.timestamp_level,
            decoding,
            vad,
            diarization_config,
            acoustic_diarization,
            gpu_device: self.gpu_device.take(),
            flash_attention: if self.flash_attention {
                Some(true)
            } else {
                None
            },
            insanely_fast_hf_token: self.hf_token.take(),
            insanely_fast_transcript_path: self.transcript_path.take(),
            no_timestamps: self.no_timestamps,
            detect_language_only: self.detect_language_only,
            batch_size: self.batch_size,
            split_on_word: self.split_on_word,
            threads: self.threads,
            processors: self.processors,
            no_gpu: self.no_gpu,
            prompt: self.prompt.take(),
            carry_initial_prompt: self.carry_initial_prompt,
            no_fallback: self.no_fallback,
            suppress_nst: self.suppress_nst,
            offset_ms: self.offset_ms,
            duration_ms: self.duration_ms,
            audio_ctx: self.audio_ctx,
            word_threshold: self.word_threshold,
            suppress_regex: self.suppress_regex.take(),
            tiny_diarize: self.tiny_diarize,
            word_timestamps: None,
            insanely_fast_tuning: None,
            alignment: None,
            punctuation: if self.normalize_segment_text {
                Some(crate::model::PunctuationConfig {
                    model: None,
                    enabled: true,
                })
            } else {
                None
            },
            source_separation: None,
            speculative,
        };

        Ok(TranscribeRequest {
            input,
            backend: self.backend,
            model: self.model.take(),
            language: self.language.take(),
            translate: self.translate,
            diarize: effective_diarize,
            persist: !self.no_persist,
            db_path: std::mem::take(&mut self.db),
            timeout_ms: self.timeout.map(|secs| secs.saturating_mul(1000)),
            backend_params,
        })
    }

    fn speaker_count_request(&self) -> FwResult<SpeakerCountRequest> {
        let specified_modes = usize::from(self.speaker_count_hard.is_some())
            + usize::from(self.speaker_count_range.is_some())
            + usize::from(self.speaker_count_prior.is_some());
        if specified_modes > 1 {
            return Err(FwError::InvalidRequest(
                "--speaker-count-hard, --speaker-count-range, and --speaker-count-prior are mutually exclusive"
                    .to_owned(),
            ));
        }
        let request = if let Some(count) = self.speaker_count_hard {
            SpeakerCountRequest::HardConstraint { count }
        } else if let Some(range) = self.speaker_count_range.as_deref() {
            let (minimum, maximum) = parse_speaker_count_range(range)?;
            SpeakerCountRequest::Range { minimum, maximum }
        } else if let Some(prior) = self.speaker_count_prior.as_deref() {
            SpeakerCountRequest::Prior {
                bins: parse_speaker_count_prior(prior)?,
            }
        } else {
            SpeakerCountRequest::Infer
        };
        crate::model::validate_speaker_count_request(&request)
            .map_err(|error| FwError::InvalidRequest(error.to_string()))?;
        Ok(request)
    }

    #[must_use]
    pub fn robot_summary(&self) -> serde_json::Value {
        json!({
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "translate": self.translate,
            "diarize": self.diarize || !self.no_diarize,
            "diarization_engine": self.diarization_engine,
            "diarization_fallback": self.diarization_fallback,
            "speaker_count": self.speaker_count_request().ok(),
            "speaker_hints_present": self.speaker_hints.is_some(),
            "persist_speaker_profiles": self.persist_speaker_profiles,
            "persist": !self.no_persist,
            "db": self.db,
            "speculative": self.speculative,
        })
    }

    /// Build a `SpeculativeConfig` from CLI arguments.
    /// Returns `None` if `--speculative` is not set.
    ///
    /// # Errors
    ///
    /// Returns [`FwError::InvalidRequest`] when the configured window geometry
    /// cannot make forward progress.
    pub fn to_speculative_config(
        &self,
    ) -> FwResult<Option<crate::streaming::SpeculativeConfig>> {
        let Some((window_size_ms, overlap_ms)) = self.speculative_window_geometry()? else {
            return Ok(None);
        };

        Ok(Some(crate::streaming::SpeculativeConfig {
            window_size_ms,
            overlap_ms,
            fast_model_name: self
                .fast_model
                .clone()
                .unwrap_or_else(|| "auto-fast".to_owned()),
            quality_model_name: self
                .quality_model
                .clone()
                .unwrap_or_else(|| "auto-quality".to_owned()),
            tolerance: crate::speculation::CorrectionTolerance {
                max_wer: self.correction_tolerance_wer.unwrap_or(0.1),
                always_correct: self.always_correct,
                ..crate::speculation::CorrectionTolerance::default()
            },
            adaptive: !self.no_adaptive,
            emit_events: true,
        }))
    }

    /// Build a serde-friendly [`SpeculativeRequest`] for storage in
    /// `BackendParams.speculative`. Returns `None` if `--speculative` is not set.
    ///
    /// The orchestrator converts this into a
    /// [`crate::streaming::SpeculativeConfig`] at dispatch time, keeping
    /// `model.rs` free of any dependency on `streaming.rs`.
    ///
    /// # Errors
    ///
    /// Returns [`FwError::InvalidRequest`] when the configured window geometry
    /// cannot make forward progress.
    pub fn to_speculative_request(&self) -> FwResult<Option<crate::model::SpeculativeRequest>> {
        self.speculative_request_with_models(self.fast_model.clone(), self.quality_model.clone())
    }

    fn speculative_window_geometry(&self) -> FwResult<Option<(u64, u64)>> {
        if !self.speculative {
            return Ok(None);
        }

        let window_size_ms = self.speculative_window_ms.unwrap_or(3000);
        let overlap_ms = self.speculative_overlap_ms.unwrap_or(500);
        if window_size_ms == 0 {
            return Err(FwError::InvalidRequest(
                "--speculative-window-ms must be greater than zero".to_owned(),
            ));
        }
        if overlap_ms >= window_size_ms {
            return Err(FwError::InvalidRequest(format!(
                "--speculative-overlap-ms ({overlap_ms}) must be less than --speculative-window-ms ({window_size_ms})"
            )));
        }
        Ok(Some((window_size_ms, overlap_ms)))
    }

    fn speculative_request_with_models(
        &self,
        fast_model: Option<String>,
        quality_model: Option<String>,
    ) -> FwResult<Option<crate::model::SpeculativeRequest>> {
        let Some((window_size_ms, overlap_ms)) = self.speculative_window_geometry()? else {
            return Ok(None);
        };
        Ok(Some(crate::model::SpeculativeRequest {
            window_size_ms,
            overlap_ms,
            fast_model_name: fast_model.unwrap_or_else(|| "auto-fast".to_owned()),
            quality_model_name: quality_model.unwrap_or_else(|| "auto-quality".to_owned()),
            max_wer_tolerance: self.correction_tolerance_wer,
            adaptive: !self.no_adaptive,
            always_correct: self.always_correct,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_args() -> TranscribeArgs {
        TranscribeArgs {
            input: Some(PathBuf::from("test.wav")),
            stdin: false,
            mic: false,
            mic_seconds: 15,
            mic_device: None,
            mic_ffmpeg_format: None,
            mic_ffmpeg_source: None,
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            no_diarize: false,
            diarization_engine: DiarizationEngine::Auto,
            diarization_fallback: DiarizationFallbackPolicy::Acoustic,
            speaker_hints: None,
            enrollment_edge_guard_ms: 100,
            diarization_max_prototypes: 512,
            persist_speaker_profiles: false,
            db: PathBuf::from("db.sqlite3"),
            no_persist: false,
            timeout: None,
            json: false,
            output_txt: false,
            output_vtt: false,
            output_srt: false,
            output_csv: false,
            output_json_full: false,
            output_lrc: false,
            no_timestamps: false,
            detect_language_only: false,
            split_on_word: false,
            best_of: None,
            beam_size: None,
            max_context: None,
            max_segment_length: None,
            temperature: None,
            temperature_increment: None,
            entropy_threshold: None,
            logprob_threshold: None,
            no_speech_threshold: None,
            vad: false,
            vad_model: None,
            vad_threshold: None,
            vad_min_speech_ms: None,
            vad_min_silence_ms: None,
            vad_max_speech_s: None,
            vad_speech_pad_ms: None,
            vad_samples_overlap: None,
            batch_size: None,
            timestamp_level: None,
            speaker_count_hard: None,
            speaker_count_range: None,
            speaker_count_prior: None,
            gpu_device: None,
            flash_attention: false,
            hf_token: None,
            transcript_path: None,
            no_stem: false,
            diarization_model: None,
            suppress_numerals: false,
            threads: None,
            processors: None,
            no_gpu: false,
            prompt: None,
            carry_initial_prompt: false,
            no_fallback: false,
            suppress_nst: false,
            tiny_diarize: false,
            normalize_segment_text: false,
            offset_ms: None,
            duration_ms: None,
            audio_ctx: None,
            word_threshold: None,
            suppress_regex: None,
            speculative: false,
            fast_model: None,
            quality_model: None,
            speculative_window_ms: None,
            speculative_overlap_ms: None,
            correction_tolerance_wer: None,
            no_adaptive: false,
            always_correct: false,
        }
    }

    #[test]
    fn consuming_request_is_byte_identical_to_borrowed_request() {
        let mut args = minimal_args();
        args.input = Some(PathBuf::from("audio/naïve input.wav"));
        args.model = Some("models/large-v3-turbo".to_owned());
        args.language = Some("日本語".to_owned());
        args.db = PathBuf::from("state/telemetry.sqlite3");
        args.vad = true;
        args.vad_model = Some(PathBuf::from("models/silero-vad.bin"));
        args.no_stem = true;
        args.diarization_model = Some("pyannote/speaker-diarization".to_owned());
        args.gpu_device = Some("cuda:0".to_owned());
        args.hf_token = Some("token-with-control-\n-boundary".to_owned());
        args.transcript_path = Some(PathBuf::from("artifacts/transcript.json"));
        args.prompt = Some("Unicode prompt: déjà vu".to_owned());
        args.suppress_regex = Some("[♪♫]+".to_owned());
        args.speculative = true;
        args.fast_model = Some("tiny.en".to_owned());
        args.quality_model = Some("large-v3-turbo".to_owned());

        let retained = args.to_request().expect("retained request");
        let consumed = args.into_request().expect("consumed request");
        assert_eq!(
            serde_json::to_vec(&retained).expect("serialize retained request"),
            serde_json::to_vec(&consumed).expect("serialize consumed request")
        );
    }

    #[test]
    fn consuming_request_preserves_validation_errors() {
        let mut args = minimal_args();
        args.input = None;
        let retained = args
            .to_request()
            .expect_err("retained no-input error")
            .to_string();
        let consumed = args
            .into_request()
            .expect_err("consumed no-input error")
            .to_string();
        assert_eq!(retained, consumed);

        let mut args = minimal_args();
        args.stdin = true;
        let retained = args
            .to_request()
            .expect_err("retained conflicting-input error")
            .to_string();
        let consumed = args
            .into_request()
            .expect_err("consumed conflicting-input error")
            .to_string();
        assert_eq!(retained, consumed);
    }

    #[test]
    fn no_input_specified_returns_error() {
        let mut args = minimal_args();
        args.input = None;
        let err = args.to_request().expect_err("should fail with no input");
        let text = err.to_string();
        assert!(
            text.contains("specify one of"),
            "expected input mode error, got: {text}"
        );
    }

    #[test]
    fn mutually_exclusive_inputs_returns_error() {
        let mut args = minimal_args();
        args.stdin = true; // input + stdin = 2 modes
        let err = args.to_request().expect_err("should fail with two inputs");
        let text = err.to_string();
        assert!(
            text.contains("mutually exclusive"),
            "expected mutex error, got: {text}"
        );
    }

    #[test]
    fn file_input_produces_file_variant() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(matches!(request.input, InputSource::File { .. }));
    }

    #[test]
    fn stdin_input_produces_stdin_variant() {
        let mut args = minimal_args();
        args.input = None;
        args.stdin = true;
        let request = args.to_request().expect("should succeed");
        assert!(matches!(request.input, InputSource::Stdin { .. }));
    }

    #[test]
    fn mic_input_produces_microphone_variant() {
        let mut args = minimal_args();
        args.input = None;
        args.mic = true;
        args.mic_seconds = 30;
        args.mic_device = Some("hw:1".to_owned());
        let request = args.to_request().expect("should succeed");
        match &request.input {
            InputSource::Microphone {
                seconds, device, ..
            } => {
                assert_eq!(*seconds, 30);
                assert_eq!(device.as_deref(), Some("hw:1"));
            }
            other => panic!("expected Microphone, got: {other:?}"),
        }
    }

    #[test]
    fn timeout_converts_seconds_to_ms() {
        let mut args = minimal_args();
        args.timeout = Some(120);
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.timeout_ms, Some(120_000));
    }

    #[test]
    fn no_persist_flag_sets_persist_false() {
        let mut args = minimal_args();
        args.no_persist = true;
        let request = args.to_request().expect("should succeed");
        assert!(!request.persist);
    }

    #[test]
    fn flash_attention_flag_sets_some_true() {
        let mut args = minimal_args();
        args.flash_attention = true;
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.backend_params.flash_attention, Some(true));
    }

    #[test]
    fn flash_attention_off_sets_none() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.flash_attention.is_none());
    }

    #[test]
    fn vad_flag_produces_vad_params() {
        let mut args = minimal_args();
        args.vad = true;
        args.vad_threshold = Some(0.5);
        let request = args.to_request().expect("should succeed");
        let vad = request.backend_params.vad.expect("vad should be Some");
        assert_eq!(vad.threshold, Some(0.5));
    }

    #[test]
    fn no_vad_flag_means_none_vad_params() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.vad.is_none());
    }

    #[test]
    fn robot_summary_contains_expected_fields() {
        let args = minimal_args();
        let summary = args.robot_summary();
        assert_eq!(summary["backend"], "auto");
        assert_eq!(summary["translate"], false);
        assert_eq!(summary["persist"], true);
        assert_eq!(summary["diarization_engine"], "auto");
        assert_eq!(summary["speaker_count"]["mode"], "infer");
        assert_eq!(summary["speaker_hints_present"], false);
    }

    #[test]
    fn native_diarization_controls_build_a_typed_request() {
        let mut args = minimal_args();
        args.diarize = true;
        args.diarization_engine = DiarizationEngine::Acoustic;
        args.diarization_fallback = DiarizationFallbackPolicy::Error;
        args.enrollment_edge_guard_ms = 175;
        args.diarization_max_prototypes = 24;
        args.persist_speaker_profiles = true;

        let request = args.to_request().expect("valid diarization request");
        let diarization = request
            .backend_params
            .acoustic_diarization
            .expect("typed request");
        assert_eq!(diarization.engine, DiarizationEngine::Acoustic);
        assert_eq!(diarization.fallback, DiarizationFallbackPolicy::Error);
        assert_eq!(diarization.enrollment_edge_guard_ms, 175);
        assert_eq!(diarization.max_prototypes, 24);
        assert!(diarization.persist_profiles);
        assert!(diarization.known_intervals.is_empty());
    }

    #[test]
    fn native_diarization_controls_reject_no_diarize() {
        let mut args = minimal_args();
        args.no_diarize = true;
        args.diarization_engine = DiarizationEngine::Acoustic;
        let error = args
            .to_request()
            .expect_err("engine selection with --no-diarize must fail");
        assert!(error.to_string().contains("--no-diarize"));

        let mut args = minimal_args();
        args.no_diarize = true;
        args.persist_speaker_profiles = true;
        assert!(
            args.to_request()
                .expect_err("profile persistence with --no-diarize must fail")
                .to_string()
                .contains("--no-diarize")
        );

        let mut args = minimal_args();
        args.no_diarize = true;
        args.speaker_count_hard = Some(2);
        assert!(
            args.to_request()
                .expect_err("speaker-count search with --no-diarize must fail")
                .to_string()
                .contains("--no-diarize")
        );
    }

    #[test]
    fn native_diarization_is_enabled_by_default_and_can_be_disabled() {
        let default_request = minimal_args().to_request().expect("default request");
        assert!(default_request.diarize);
        assert!(
            default_request
                .backend_params
                .acoustic_diarization
                .is_some()
        );

        let mut disabled = minimal_args();
        disabled.no_diarize = true;
        let disabled_request = disabled.to_request().expect("disabled request");
        assert!(!disabled_request.diarize);
        assert!(
            disabled_request
                .backend_params
                .acoustic_diarization
                .is_none()
        );
    }

    #[test]
    fn speaker_hints_parser_accepts_v1_document_and_bare_array() {
        let interval = r#"{
            "speaker_ref":"near",
            "start_ms":100,
            "end_ms":900,
            "confidence":0.95,
            "policy":"hard_must_link"
        }"#;
        let bare = parse_speaker_hints(format!("[{interval}]").as_bytes()).expect("bare array");
        let document = parse_speaker_hints(
            format!("{{\"schema_version\":\"speaker-hints-v1\",\"known_intervals\":[{interval}]}}")
                .as_bytes(),
        )
        .expect("versioned document");
        assert_eq!(bare, document);
        assert_eq!(bare[0].speaker_ref, "near");
    }

    #[test]
    fn speaker_hints_parser_rejects_unknown_schema_and_malformed_json() {
        let wrong_schema = br#"{"schema_version":"speaker-hints-v2","known_intervals":[]}"#;
        assert!(
            parse_speaker_hints(wrong_schema)
                .expect_err("future schema must fail closed")
                .to_string()
                .contains("speaker-hints-v1")
        );
        assert!(parse_speaker_hints(b"not json").is_err());
        assert!(
            parse_speaker_hints(
                br#"{"schema_version":"speaker-hints-v1","known_intervals":[],"extra":true}"#,
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn speaker_hints_reader_rejects_symlinks_before_parsing() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary speaker-hints directory");
        let target = directory.path().join("target.json");
        let link = directory.path().join("link.json");
        std::fs::write(&target, b"[]").expect("write speaker-hints target");
        symlink(&target, &link).expect("create speaker-hints symlink");
        let error = read_speaker_hints(&link).expect_err("symlink must fail closed");
        assert!(error.to_string().contains("regular non-symlink"));
    }

    #[test]
    fn robot_summary_reports_hint_presence_without_disclosing_the_path() {
        let mut args = minimal_args();
        args.diarize = true;
        args.speaker_hints = Some(PathBuf::from(
            "/private/confidential-corpus/do-not-disclose.json",
        ));
        let summary = args.robot_summary();
        assert_eq!(summary["speaker_hints_present"], true);
        assert!(
            !serde_json::to_string(&summary)
                .expect("serialize summary")
                .contains("do-not-disclose")
        );
    }

    #[test]
    fn threading_params_forwarded_to_backend_params() {
        let mut args = minimal_args();
        args.threads = Some(8);
        args.processors = Some(2);
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.backend_params.threads, Some(8));
        assert_eq!(request.backend_params.processors, Some(2));
    }

    #[test]
    fn gpu_control_flags_forwarded() {
        let mut args = minimal_args();
        args.no_gpu = true;
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.no_gpu);
    }

    #[test]
    fn prompt_and_carry_initial_prompt_forwarded() {
        let mut args = minimal_args();
        args.prompt = Some("medical terms".to_owned());
        args.carry_initial_prompt = true;
        let request = args.to_request().expect("should succeed");
        assert_eq!(
            request.backend_params.prompt.as_deref(),
            Some("medical terms")
        );
        assert!(request.backend_params.carry_initial_prompt);
    }

    #[test]
    fn audio_windowing_params_forwarded() {
        let mut args = minimal_args();
        args.offset_ms = Some(5000);
        args.duration_ms = Some(30000);
        args.audio_ctx = Some(128);
        args.word_threshold = Some(0.25);
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.backend_params.offset_ms, Some(5000));
        assert_eq!(request.backend_params.duration_ms, Some(30000));
        assert_eq!(request.backend_params.audio_ctx, Some(128));
        assert_eq!(request.backend_params.word_threshold, Some(0.25));
    }

    #[test]
    fn decoding_control_flags_forwarded() {
        let mut args = minimal_args();
        args.no_fallback = true;
        args.suppress_nst = true;
        args.suppress_regex = Some(r"\[.*\]".to_owned());
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.no_fallback);
        assert!(request.backend_params.suppress_nst);
        assert_eq!(
            request.backend_params.suppress_regex.as_deref(),
            Some(r"\[.*\]")
        );
    }

    #[test]
    fn tiny_diarize_flag_forwarded() {
        let mut args = minimal_args();
        args.tiny_diarize = true;
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.tiny_diarize);
    }

    #[test]
    fn tiny_diarize_default_false() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(!request.backend_params.tiny_diarize);
    }

    #[test]
    fn normalize_segment_text_flag_enables_punctuation_config() {
        let mut args = minimal_args();
        args.normalize_segment_text = true;
        let request = args.to_request().expect("should succeed");
        let punctuation = request
            .backend_params
            .punctuation
            .expect("opt-in flag populates punctuation config");
        assert!(punctuation.enabled);
        assert!(punctuation.model.is_none());
    }

    #[test]
    fn segment_text_normalization_defaults_off() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(
            request.backend_params.punctuation.is_none(),
            "segment text must stay faithful to the decoded transcript by default"
        );
    }

    // --- Speaker count request ---

    #[test]
    fn speaker_count_infers_when_no_speaker_args() {
        let mut args = minimal_args();
        args.diarize = true;
        let request = args.to_request().expect("should succeed");
        assert_eq!(
            request
                .backend_params
                .acoustic_diarization
                .expect("--diarize creates native request")
                .speaker_count,
            SpeakerCountRequest::Infer
        );
    }

    #[test]
    fn speaker_count_built_from_explicit_hard_count() {
        let mut args = minimal_args();
        args.diarize = true;
        args.speaker_count_hard = Some(3);
        let summary = args.robot_summary();
        assert_eq!(summary["speaker_count"]["mode"], "hard_constraint");
        assert_eq!(summary["speaker_count"]["count"], 3);
        let request = args.to_request().expect("should succeed");
        let count = &request
            .backend_params
            .acoustic_diarization
            .expect("speaker count creates a diarization request")
            .speaker_count;
        assert_eq!(count, &SpeakerCountRequest::HardConstraint { count: 3 });
    }

    #[test]
    fn speaker_count_built_from_explicit_range() {
        let mut args = minimal_args();
        args.diarize = true;
        args.speaker_count_range = Some("2..8".to_owned());
        let request = args.to_request().expect("should succeed");
        let count = &request
            .backend_params
            .acoustic_diarization
            .expect("speaker range creates a diarization request")
            .speaker_count;
        assert_eq!(
            count,
            &SpeakerCountRequest::Range {
                minimum: 2,
                maximum: 8
            }
        );
    }

    #[test]
    fn speaker_count_point_prior_remains_soft() {
        let mut args = minimal_args();
        args.diarize = true;
        args.speaker_count_prior = Some("3".to_owned());
        let count = args
            .to_request()
            .expect("point prior")
            .backend_params
            .acoustic_diarization
            .expect("diarization request")
            .speaker_count;
        assert_eq!(
            count,
            SpeakerCountRequest::Prior {
                bins: vec![SpeakerCountPriorMass {
                    count: 3,
                    probability: 1.0,
                }]
            }
        );
    }

    #[test]
    fn speaker_count_distribution_prior_is_sorted_and_validated() {
        let mut args = minimal_args();
        args.diarize = true;
        args.speaker_count_prior = Some("3=0.75,2=0.25".to_owned());
        let count = args
            .to_request()
            .expect("distribution prior")
            .backend_params
            .acoustic_diarization
            .expect("diarization request")
            .speaker_count;
        assert_eq!(
            count,
            SpeakerCountRequest::Prior {
                bins: vec![
                    SpeakerCountPriorMass {
                        count: 2,
                        probability: 0.25,
                    },
                    SpeakerCountPriorMass {
                        count: 3,
                        probability: 0.75,
                    },
                ]
            }
        );
    }

    #[test]
    fn malformed_speaker_count_cli_values_fail_with_typed_messages() {
        let mut args = minimal_args();
        args.diarize = true;
        args.speaker_count_range = Some("4..2".to_owned());
        assert!(
            args.to_request()
                .expect_err("reversed range")
                .to_string()
                .contains("minimum <= maximum")
        );

        let mut args = minimal_args();
        args.diarize = true;
        args.speaker_count_prior = Some("2=0.8,3=0.8".to_owned());
        assert!(
            args.to_request()
                .expect_err("unnormalized prior")
                .to_string()
                .contains("sum to exactly 1")
        );
    }

    // --- Diarization config ---

    #[test]
    fn diarization_config_none_when_no_diarization_args() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.diarization_config.is_none());
    }

    #[test]
    fn diarization_config_built_from_no_stem() {
        let mut args = minimal_args();
        args.no_stem = true;
        let request = args.to_request().expect("should succeed");
        let dc = request
            .backend_params
            .diarization_config
            .expect("should be Some");
        assert!(dc.no_stem);
        assert!(dc.whisper_model.is_none());
        assert!(!dc.suppress_numerals);
    }

    #[test]
    fn diarization_config_includes_gpu_device_and_batch_size() {
        let mut args = minimal_args();
        args.no_stem = true;
        args.gpu_device = Some("0".to_owned());
        args.batch_size = Some(16);
        let request = args.to_request().expect("should succeed");
        let dc = request
            .backend_params
            .diarization_config
            .expect("should be Some");
        assert_eq!(dc.device.as_deref(), Some("0"));
        assert_eq!(dc.batch_size, Some(16));
    }

    #[test]
    fn diarization_config_from_model_and_suppress_numerals() {
        let mut args = minimal_args();
        args.diarization_model = Some("large-v3".to_owned());
        args.suppress_numerals = true;
        let request = args.to_request().expect("should succeed");
        let dc = request
            .backend_params
            .diarization_config
            .expect("should be Some");
        assert_eq!(dc.whisper_model.as_deref(), Some("large-v3"));
        assert!(dc.suppress_numerals);
    }

    // --- Decoding params ---

    #[test]
    fn decoding_params_none_when_no_decoding_args() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.decoding.is_none());
    }

    #[test]
    fn decoding_params_built_from_single_field() {
        let mut args = minimal_args();
        args.beam_size = Some(5);
        let request = args.to_request().expect("should succeed");
        let dp = request.backend_params.decoding.expect("should be Some");
        assert_eq!(dp.beam_size, Some(5));
        assert!(dp.best_of.is_none());
        assert!(dp.temperature.is_none());
    }

    #[test]
    fn decoding_params_built_from_all_fields() {
        let mut args = minimal_args();
        args.best_of = Some(3);
        args.beam_size = Some(5);
        args.max_context = Some(128);
        args.max_segment_length = Some(40);
        args.temperature = Some(0.8);
        args.temperature_increment = Some(0.2);
        args.entropy_threshold = Some(2.4);
        args.logprob_threshold = Some(-1.0);
        args.no_speech_threshold = Some(0.6);
        let request = args.to_request().expect("should succeed");
        let dp = request.backend_params.decoding.expect("should be Some");
        assert_eq!(dp.best_of, Some(3));
        assert_eq!(dp.beam_size, Some(5));
        assert_eq!(dp.max_context, Some(128));
        assert_eq!(dp.max_segment_length, Some(40));
        assert_eq!(dp.temperature, Some(0.8));
        assert_eq!(dp.temperature_increment, Some(0.2));
        assert_eq!(dp.entropy_threshold, Some(2.4));
        assert_eq!(dp.logprob_threshold, Some(-1.0));
        assert_eq!(dp.no_speech_threshold, Some(0.6));
    }

    // --- Output format combination ---

    #[test]
    fn output_formats_empty_by_default() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.output_formats.is_empty());
    }

    #[test]
    fn output_formats_collects_all_enabled_flags() {
        let mut args = minimal_args();
        args.output_txt = true;
        args.output_vtt = true;
        args.output_srt = true;
        args.output_csv = true;
        args.output_json_full = true;
        args.output_lrc = true;
        let request = args.to_request().expect("should succeed");
        let formats = &request.backend_params.output_formats;
        assert_eq!(formats.len(), 6);
        assert_eq!(formats[0], OutputFormat::Txt);
        assert_eq!(formats[1], OutputFormat::Vtt);
        assert_eq!(formats[2], OutputFormat::Srt);
        assert_eq!(formats[3], OutputFormat::Csv);
        assert_eq!(formats[4], OutputFormat::JsonFull);
        assert_eq!(formats[5], OutputFormat::Lrc);
    }

    #[test]
    fn output_formats_partial_selection() {
        let mut args = minimal_args();
        args.output_srt = true;
        args.output_lrc = true;
        let request = args.to_request().expect("should succeed");
        let formats = &request.backend_params.output_formats;
        assert_eq!(formats.len(), 2);
        assert_eq!(formats[0], OutputFormat::Srt);
        assert_eq!(formats[1], OutputFormat::Lrc);
    }

    #[test]
    fn default_args_leave_new_phase4_fields_at_defaults() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.threads.is_none());
        assert!(request.backend_params.processors.is_none());
        assert!(!request.backend_params.no_gpu);
        assert!(request.backend_params.prompt.is_none());
        assert!(!request.backend_params.carry_initial_prompt);
        assert!(!request.backend_params.no_fallback);
        assert!(!request.backend_params.suppress_nst);
        assert!(!request.backend_params.tiny_diarize);
        assert!(request.backend_params.offset_ms.is_none());
        assert!(request.backend_params.duration_ms.is_none());
        assert!(request.backend_params.audio_ctx.is_none());
        assert!(request.backend_params.word_threshold.is_none());
        assert!(request.backend_params.suppress_regex.is_none());
    }

    // --- Additional edge cases ---

    #[test]
    fn timeout_none_leaves_timeout_ms_none() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.timeout_ms.is_none());
    }

    #[test]
    fn timeout_zero_seconds_produces_zero_ms() {
        let mut args = minimal_args();
        args.timeout = Some(0);
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.timeout_ms, Some(0));
    }

    #[test]
    fn all_three_inputs_returns_error() {
        let mut args = minimal_args();
        args.stdin = true;
        args.mic = true;
        let err = args.to_request().expect_err("3 inputs should fail");
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn mic_with_ffmpeg_overrides_produces_microphone_variant() {
        let mut args = minimal_args();
        args.input = None;
        args.mic = true;
        args.mic_seconds = 60;
        args.mic_ffmpeg_format = Some("pulse".to_owned());
        args.mic_ffmpeg_source = Some("my-sink".to_owned());
        let request = args.to_request().expect("should succeed");
        match &request.input {
            InputSource::Microphone {
                seconds,
                ffmpeg_format,
                ffmpeg_source,
                ..
            } => {
                assert_eq!(*seconds, 60);
                assert_eq!(ffmpeg_format.as_deref(), Some("pulse"));
                assert_eq!(ffmpeg_source.as_deref(), Some("my-sink"));
            }
            other => panic!("expected Microphone, got: {other:?}"),
        }
    }

    #[test]
    fn translate_and_diarize_flags_forwarded() {
        let mut args = minimal_args();
        args.translate = true;
        args.diarize = true;
        let request = args.to_request().expect("should succeed");
        assert!(request.translate);
        assert!(request.diarize);
    }

    #[test]
    fn backend_kind_forwarded_to_request() {
        for kind in [
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
        ] {
            let mut args = minimal_args();
            args.backend = kind;
            let request = args.to_request().expect("should succeed");
            assert_eq!(request.backend, kind);
        }
    }

    #[test]
    fn model_and_language_forwarded() {
        let mut args = minimal_args();
        args.model = Some("large-v3".to_owned());
        args.language = Some("de".to_owned());
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.model.as_deref(), Some("large-v3"));
        assert_eq!(request.language.as_deref(), Some("de"));
    }

    #[test]
    fn robot_summary_reflects_no_persist() {
        let mut args = minimal_args();
        args.no_persist = true;
        let summary = args.robot_summary();
        assert_eq!(summary["persist"], false);
    }

    #[test]
    fn robot_summary_reflects_translate_and_diarize() {
        let mut args = minimal_args();
        args.translate = true;
        args.diarize = true;
        args.language = Some("fr".to_owned());
        let summary = args.robot_summary();
        assert_eq!(summary["translate"], true);
        assert_eq!(summary["diarize"], true);
        assert_eq!(summary["language"], "fr");
    }

    #[test]
    fn vad_flag_with_all_vad_params() {
        let mut args = minimal_args();
        args.vad = true;
        args.vad_model = Some(PathBuf::from("/models/vad.onnx"));
        args.vad_threshold = Some(0.6);
        args.vad_min_speech_ms = Some(200);
        args.vad_min_silence_ms = Some(100);
        args.vad_max_speech_s = Some(30.0);
        args.vad_speech_pad_ms = Some(50);
        args.vad_samples_overlap = Some(0.15);
        let request = args.to_request().expect("should succeed");
        let vad = request.backend_params.vad.expect("vad should be Some");
        assert_eq!(
            vad.model_path.as_deref(),
            Some(std::path::Path::new("/models/vad.onnx"))
        );
        assert_eq!(vad.threshold, Some(0.6));
        assert_eq!(vad.min_speech_duration_ms, Some(200));
        assert_eq!(vad.min_silence_duration_ms, Some(100));
        assert_eq!(vad.max_speech_duration_s, Some(30.0));
        assert_eq!(vad.speech_pad_ms, Some(50));
        assert_eq!(vad.samples_overlap, Some(0.15));
    }

    #[test]
    fn persist_true_by_default() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.persist);
    }

    #[test]
    fn gpu_device_forwarded_to_both_backend_and_diarization() {
        let mut args = minimal_args();
        args.gpu_device = Some("cuda:1".to_owned());
        args.no_stem = true; // triggers diarization_config creation
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.backend_params.gpu_device.as_deref(), Some("cuda:1"));
        let dc = request
            .backend_params
            .diarization_config
            .expect("diarization config");
        assert_eq!(dc.device.as_deref(), Some("cuda:1"));
    }

    #[test]
    fn timestamp_level_forwarded() {
        let mut args = minimal_args();
        args.timestamp_level = Some(TimestampLevel::Word);
        let request = args.to_request().expect("should succeed");
        assert_eq!(
            request.backend_params.timestamp_level,
            Some(TimestampLevel::Word)
        );
    }

    #[test]
    fn batch_size_forwarded_to_backend_params() {
        let mut args = minimal_args();
        args.batch_size = Some(32);
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.backend_params.batch_size, Some(32));
    }

    #[test]
    fn hf_token_forwarded_to_backend_params() {
        let mut args = minimal_args();
        args.hf_token = Some("hf_override_token".to_owned());
        let request = args.to_request().expect("should succeed");
        assert_eq!(
            request.backend_params.insanely_fast_hf_token.as_deref(),
            Some("hf_override_token")
        );
    }

    #[test]
    fn transcript_path_forwarded_to_backend_params() {
        let mut args = minimal_args();
        args.transcript_path = Some(PathBuf::from("artifacts/ifw.json"));
        let request = args.to_request().expect("should succeed");
        assert_eq!(
            request
                .backend_params
                .insanely_fast_transcript_path
                .as_deref(),
            Some(PathBuf::from("artifacts/ifw.json").as_path())
        );
    }

    #[test]
    fn boolean_inference_flags_default_false() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(!request.backend_params.no_timestamps);
        assert!(!request.backend_params.detect_language_only);
        assert!(!request.backend_params.split_on_word);
    }

    #[test]
    fn boolean_inference_flags_set_true() {
        let mut args = minimal_args();
        args.no_timestamps = true;
        args.detect_language_only = true;
        args.split_on_word = true;
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.no_timestamps);
        assert!(request.backend_params.detect_language_only);
        assert!(request.backend_params.split_on_word);
    }

    #[test]
    fn timeout_large_value_no_overflow() {
        let mut args = minimal_args();
        args.timeout = Some(86400); // 24 hours
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.timeout_ms, Some(86_400_000));
    }

    #[test]
    fn all_params_combined_to_request() {
        let mut args = minimal_args();
        args.backend = BackendKind::InsanelyFast;
        args.model = Some("large-v3-turbo".to_owned());
        args.language = Some("ja".to_owned());
        args.translate = true;
        args.diarize = true;
        args.timeout = Some(300);
        args.flash_attention = true;
        args.batch_size = Some(24);
        args.gpu_device = Some("cuda:0".to_owned());
        args.timestamp_level = Some(TimestampLevel::Word);
        args.speaker_count_hard = Some(3);
        args.hf_token = Some("hf_123".to_owned());
        args.transcript_path = Some(PathBuf::from("artifacts/ifw-out.json"));
        args.output_srt = true;
        args.output_vtt = true;
        args.no_stem = true;
        args.suppress_numerals = true;
        args.vad = true;
        args.vad_threshold = Some(0.5);
        args.best_of = Some(5);
        args.temperature = Some(0.0);
        args.threads = Some(4);
        args.no_gpu = true;
        args.prompt = Some("technical".to_owned());
        args.carry_initial_prompt = true;
        args.suppress_regex = Some(r"\[.*\]".to_owned());

        let request = args.to_request().expect("should succeed");

        assert_eq!(request.backend, BackendKind::InsanelyFast);
        assert_eq!(request.model.as_deref(), Some("large-v3-turbo"));
        assert_eq!(request.language.as_deref(), Some("ja"));
        assert!(request.translate);
        assert!(request.diarize);
        assert_eq!(request.timeout_ms, Some(300_000));
        assert_eq!(request.backend_params.flash_attention, Some(true));
        assert_eq!(request.backend_params.batch_size, Some(24));
        assert_eq!(request.backend_params.gpu_device.as_deref(), Some("cuda:0"));
        assert_eq!(
            request.backend_params.insanely_fast_hf_token.as_deref(),
            Some("hf_123")
        );
        assert_eq!(
            request
                .backend_params
                .insanely_fast_transcript_path
                .as_deref(),
            Some(PathBuf::from("artifacts/ifw-out.json").as_path())
        );
        assert_eq!(
            request.backend_params.timestamp_level,
            Some(TimestampLevel::Word)
        );
        assert!(matches!(
            request
                .backend_params
                .acoustic_diarization
                .as_ref()
                .map(|request| &request.speaker_count),
            Some(SpeakerCountRequest::HardConstraint { count: 3 })
        ));
        assert_eq!(request.backend_params.output_formats.len(), 2);
        assert!(request.backend_params.diarization_config.is_some());
        assert!(request.backend_params.vad.is_some());
        assert!(request.backend_params.decoding.is_some());
        assert_eq!(request.backend_params.threads, Some(4));
        assert!(request.backend_params.no_gpu);
        assert_eq!(request.backend_params.prompt.as_deref(), Some("technical"));
        assert!(request.backend_params.carry_initial_prompt);
        assert_eq!(
            request.backend_params.suppress_regex.as_deref(),
            Some(r"\[.*\]")
        );
    }

    #[test]
    fn stdin_and_mic_returns_error() {
        let mut args = minimal_args();
        args.input = None;
        args.stdin = true;
        args.mic = true;
        let err = args.to_request().expect_err("should fail");
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn db_path_forwarded() {
        let mut args = minimal_args();
        args.db = PathBuf::from("/custom/path/db.sqlite3");
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.db_path, PathBuf::from("/custom/path/db.sqlite3"));
    }

    #[test]
    fn robot_summary_includes_model_and_db() {
        let mut args = minimal_args();
        args.model = Some("large-v3".to_owned());
        args.db = PathBuf::from("/tmp/test.sqlite3");
        let summary = args.robot_summary();
        assert_eq!(summary["model"], "large-v3");
        assert_eq!(summary["db"], "/tmp/test.sqlite3");
    }

    #[test]
    fn vad_params_without_vad_flag_are_ignored() {
        let mut args = minimal_args();
        // Set VAD params but don't set --vad flag.
        args.vad_threshold = Some(0.5);
        args.vad_min_speech_ms = Some(200);
        let request = args.to_request().expect("should succeed");
        // Without --vad flag, vad params should be None.
        assert!(request.backend_params.vad.is_none());
    }

    #[test]
    fn exact_and_range_speaker_flags_are_rejected_as_ambiguous() {
        let mut args = minimal_args();
        args.diarize = true;
        args.speaker_count_hard = Some(4);
        args.speaker_count_range = Some("2..6".to_owned());
        let error = args
            .to_request()
            .expect_err("typed count request cannot represent contradictory modes");
        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn diarization_config_all_fields() {
        let mut args = minimal_args();
        args.no_stem = true;
        args.diarization_model = Some("medium".to_owned());
        args.suppress_numerals = true;
        args.gpu_device = Some("mps".to_owned());
        args.batch_size = Some(8);
        let request = args.to_request().expect("should succeed");
        let dc = request
            .backend_params
            .diarization_config
            .expect("should be Some");
        assert!(dc.no_stem);
        assert_eq!(dc.whisper_model.as_deref(), Some("medium"));
        assert!(dc.suppress_numerals);
        assert_eq!(dc.device.as_deref(), Some("mps"));
        assert_eq!(dc.batch_size, Some(8));
    }

    #[test]
    fn robot_summary_null_model_serializes_to_null() {
        let args = minimal_args();
        let summary = args.robot_summary();
        assert!(summary["model"].is_null());
    }

    #[test]
    fn input_and_mic_mutually_exclusive() {
        let mut args = minimal_args();
        // input is already Some
        args.mic = true;
        let err = args.to_request().expect_err("should fail");
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn batch_size_forwarded_to_diarization_config_when_present() {
        let mut args = minimal_args();
        args.batch_size = Some(16);
        args.no_stem = true; // triggers diarization_config
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.backend_params.batch_size, Some(16));
        let dc = request
            .backend_params
            .diarization_config
            .expect("should be Some");
        assert_eq!(dc.batch_size, Some(16));
    }

    #[test]
    fn mic_default_seconds_used_when_not_overridden() {
        let mut args = minimal_args();
        args.input = None;
        args.mic = true;
        // mic_seconds stays at default 15
        let request = args.to_request().expect("should succeed");
        match &request.input {
            InputSource::Microphone { seconds, .. } => {
                assert_eq!(*seconds, 15);
            }
            other => panic!("expected Microphone, got: {other:?}"),
        }
    }

    #[test]
    fn timeout_one_second_produces_1000_ms() {
        let mut args = minimal_args();
        args.timeout = Some(1);
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.timeout_ms, Some(1000));
    }

    #[test]
    fn stdin_hint_extension_is_none() {
        let mut args = minimal_args();
        args.input = None;
        args.stdin = true;
        let request = args.to_request().expect("should succeed");
        match &request.input {
            InputSource::Stdin { hint_extension } => {
                assert!(hint_extension.is_none(), "CLI stdin has no hint extension");
            }
            other => panic!("expected Stdin, got: {other:?}"),
        }
    }

    #[test]
    fn vad_flag_with_all_defaults_produces_all_none_params() {
        let mut args = minimal_args();
        args.vad = true;
        // All individual vad params remain at None default.
        let request = args.to_request().expect("should succeed");
        let vad = request
            .backend_params
            .vad
            .expect("should be Some when --vad set");
        assert!(vad.model_path.is_none());
        assert!(vad.threshold.is_none());
        assert!(vad.min_speech_duration_ms.is_none());
        assert!(vad.min_silence_duration_ms.is_none());
        assert!(vad.max_speech_duration_s.is_none());
        assert!(vad.speech_pad_ms.is_none());
        assert!(vad.samples_overlap.is_none());
    }

    #[test]
    fn input_and_stdin_mutually_exclusive() {
        let mut args = minimal_args();
        args.stdin = true; // input already Some from minimal_args
        let err = args.to_request().expect_err("should fail");
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn timeout_very_large_value_u64_max_division() {
        let mut args = minimal_args();
        args.timeout = Some(u64::MAX / 1000);
        let request = args.to_request().expect("should succeed");
        assert_eq!(request.timeout_ms, Some((u64::MAX / 1000) * 1000));
    }

    #[test]
    fn timeout_u64_max_saturates_to_u64_max_millis() {
        let mut args = minimal_args();
        args.timeout = Some(u64::MAX);

        let request = args.to_request().expect("should succeed");
        assert_eq!(request.timeout_ms, Some(u64::MAX));
    }

    #[test]
    fn robot_summary_default_args_has_all_expected_keys() {
        let args = minimal_args();
        let summary = args.robot_summary();
        let keys = [
            "backend",
            "model",
            "language",
            "translate",
            "diarize",
            "diarization_engine",
            "diarization_fallback",
            "speaker_count",
            "speaker_hints_present",
            "persist_speaker_profiles",
            "persist",
            "db",
            "speculative",
        ];
        for key in keys {
            assert!(
                summary.get(key).is_some(),
                "missing key `{key}` in robot_summary"
            );
        }
        // Default has no extra keys beyond what's defined.
        let obj = summary.as_object().expect("object");
        assert_eq!(
            obj.len(),
            keys.len(),
            "unexpected extra keys in robot_summary"
        );
    }

    #[test]
    fn robot_summary_backend_kind_serialized() {
        for (kind, expected) in [
            (BackendKind::WhisperCpp, "whisper_cpp"),
            (BackendKind::InsanelyFast, "insanely_fast"),
            (BackendKind::WhisperDiarization, "whisper_diarization"),
        ] {
            let mut args = minimal_args();
            args.backend = kind;
            let summary = args.robot_summary();
            assert_eq!(summary["backend"], expected);
        }
    }

    #[test]
    fn gpu_device_without_diarization_only_in_backend_params() {
        // gpu_device is forwarded to backend_params.gpu_device (line 559)
        // but NOT to diarization_config when no diarization flags are set.
        let mut args = minimal_args();
        args.gpu_device = Some("cuda:1".to_owned());
        let req = args.to_request().expect("valid");
        assert_eq!(
            req.backend_params.gpu_device.as_deref(),
            Some("cuda:1"),
            "gpu_device should be in backend_params"
        );
        assert!(
            req.backend_params.diarization_config.is_none(),
            "diarization_config should be None when no diarization flags set"
        );
    }

    #[test]
    fn timeout_seconds_to_millis_conversion() {
        // timeout field is in seconds; to_request converts to ms (line 592).
        let mut args = minimal_args();
        args.timeout = Some(120);
        let req = args.to_request().expect("valid");
        assert_eq!(req.timeout_ms, Some(120_000));

        // Zero timeout.
        args.timeout = Some(0);
        let req = args.to_request().expect("valid");
        assert_eq!(req.timeout_ms, Some(0));
    }

    #[test]
    fn max_context_negative_one_passes_through() {
        // max_context = -1 means unlimited (per whisper.cpp docs).
        let mut args = minimal_args();
        args.max_context = Some(-1);
        let req = args.to_request().expect("valid");
        let decoding = req
            .backend_params
            .decoding
            .expect("decoding should be Some");
        assert_eq!(decoding.max_context, Some(-1));
    }

    #[test]
    fn all_output_formats_combined() {
        // When all output format flags are set, all formats appear in the list.
        let mut args = minimal_args();
        args.output_txt = true;
        args.output_vtt = true;
        args.output_srt = true;
        args.output_csv = true;
        args.output_json_full = true;
        args.output_lrc = true;
        let req = args.to_request().expect("valid");
        assert_eq!(req.backend_params.output_formats.len(), 6);
        assert!(
            req.backend_params
                .output_formats
                .contains(&OutputFormat::Txt)
        );
        assert!(
            req.backend_params
                .output_formats
                .contains(&OutputFormat::Vtt)
        );
        assert!(
            req.backend_params
                .output_formats
                .contains(&OutputFormat::Srt)
        );
        assert!(
            req.backend_params
                .output_formats
                .contains(&OutputFormat::Csv)
        );
        assert!(
            req.backend_params
                .output_formats
                .contains(&OutputFormat::JsonFull)
        );
        assert!(
            req.backend_params
                .output_formats
                .contains(&OutputFormat::Lrc)
        );
    }

    #[test]
    fn robot_summary_no_persist_inverted() {
        // robot_summary reports persist as `!self.no_persist` (line 605).
        let mut args = minimal_args();
        args.no_persist = true;
        let summary = args.robot_summary();
        assert_eq!(summary["persist"], false);

        args.no_persist = false;
        let summary = args.robot_summary();
        assert_eq!(summary["persist"], true);
    }

    // ── bd-38c.6: ShutdownController tests ──

    #[test]
    fn shutdown_controller_is_not_shutting_down_initially() {
        ShutdownController::reset();
        assert!(
            !ShutdownController::is_shutting_down(),
            "should not be shutting down before trigger"
        );
    }

    #[test]
    fn shutdown_controller_trigger_sets_flag() {
        ShutdownController::reset();
        ShutdownController::trigger_shutdown();
        assert!(
            ShutdownController::is_shutting_down(),
            "should be shutting down after trigger"
        );
        ShutdownController::reset();
    }

    #[test]
    fn shutdown_controller_reset_clears_flag() {
        ShutdownController::trigger_shutdown();
        assert!(ShutdownController::is_shutting_down());
        ShutdownController::reset();
        assert!(
            !ShutdownController::is_shutting_down(),
            "reset should clear shutdown flag"
        );
    }

    #[test]
    fn shutdown_controller_signal_exit_code_is_130() {
        assert_eq!(
            ShutdownController::signal_exit_code(),
            130,
            "signal exit code should be 128 + SIGINT(2) = 130"
        );
    }

    #[test]
    fn shutdown_controller_trigger_is_idempotent() {
        ShutdownController::reset();
        ShutdownController::trigger_shutdown();
        ShutdownController::trigger_shutdown();
        ShutdownController::trigger_shutdown();
        assert!(ShutdownController::is_shutting_down());
        ShutdownController::reset();
    }

    // ── bd-2xe.4: ControlFrameKind / CLI enum tests ──

    #[test]
    fn control_frame_kind_handshake_variant_exists() {
        let kind = ControlFrameKind::Handshake;
        assert_eq!(kind, ControlFrameKind::Handshake);
    }

    #[test]
    fn control_frame_kind_eof_variant_exists() {
        let kind = ControlFrameKind::Eof;
        assert_eq!(kind, ControlFrameKind::Eof);
    }

    #[test]
    fn control_frame_kind_reset_variant_exists() {
        let kind = ControlFrameKind::Reset;
        assert_eq!(kind, ControlFrameKind::Reset);
    }

    #[test]
    fn control_frame_kind_all_variants_are_distinct() {
        assert_ne!(ControlFrameKind::Handshake, ControlFrameKind::Eof);
        assert_ne!(ControlFrameKind::Handshake, ControlFrameKind::Reset);
        assert_ne!(ControlFrameKind::Eof, ControlFrameKind::Reset);
    }

    #[test]
    fn cli_parse_tty_audio_send_control_handshake() {
        let cli =
            Cli::try_parse_from(["franken_whisper", "tty-audio", "send-control", "handshake"])
                .expect("should parse");
        match cli.command {
            Command::TtyAudio { command } => match command {
                TtyAudioCommand::SendControl { frame_type } => {
                    assert_eq!(frame_type, ControlFrameKind::Handshake);
                }
                other => panic!("expected SendControl, got: {other:?}"),
            },
            other => panic!("expected TtyAudio, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_tty_audio_send_control_eof() {
        let cli = Cli::try_parse_from(["franken_whisper", "tty-audio", "send-control", "eof"])
            .expect("should parse");
        match cli.command {
            Command::TtyAudio { command } => match command {
                TtyAudioCommand::SendControl { frame_type } => {
                    assert_eq!(frame_type, ControlFrameKind::Eof);
                }
                other => panic!("expected SendControl, got: {other:?}"),
            },
            other => panic!("expected TtyAudio, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_tty_audio_send_control_reset() {
        let cli = Cli::try_parse_from(["franken_whisper", "tty-audio", "send-control", "reset"])
            .expect("should parse");
        match cli.command {
            Command::TtyAudio { command } => match command {
                TtyAudioCommand::SendControl { frame_type } => {
                    assert_eq!(frame_type, ControlFrameKind::Reset);
                }
                other => panic!("expected SendControl, got: {other:?}"),
            },
            other => panic!("expected TtyAudio, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_tty_audio_retransmit_defaults() {
        let cli =
            Cli::try_parse_from(["franken_whisper", "tty-audio", "retransmit"]).expect("parse");
        match cli.command {
            Command::TtyAudio { command } => match command {
                TtyAudioCommand::Retransmit { recovery, rounds } => {
                    assert_eq!(recovery, TtyAudioRecoveryPolicy::SkipMissing);
                    assert_eq!(rounds, 1);
                }
                other => panic!("expected Retransmit, got: {other:?}"),
            },
            other => panic!("expected TtyAudio, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_tty_audio_retransmit_custom_options() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "tty-audio",
            "retransmit",
            "--recovery",
            "fail_closed",
            "--rounds",
            "3",
        ])
        .expect("parse");
        match cli.command {
            Command::TtyAudio { command } => match command {
                TtyAudioCommand::Retransmit { recovery, rounds } => {
                    assert_eq!(recovery, TtyAudioRecoveryPolicy::FailClosed);
                    assert_eq!(rounds, 3);
                }
                other => panic!("expected Retransmit, got: {other:?}"),
            },
            other => panic!("expected TtyAudio, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_tty_audio_send_control_invalid_frame_type_fails() {
        let result =
            Cli::try_parse_from(["franken_whisper", "tty-audio", "send-control", "invalid"]);
        assert!(
            result.is_err(),
            "invalid frame_type should fail CLI parsing"
        );
    }

    // --- bd-qlt.11: Speculative CLI flags ---

    #[test]
    fn speculative_default_is_false() {
        let args = minimal_args();
        assert!(!args.speculative);
        assert!(
            args.to_speculative_config()
                .expect("disabled speculative config is valid")
                .is_none()
        );
    }

    #[test]
    fn speculative_config_built_when_flag_set() {
        let mut args = minimal_args();
        args.speculative = true;
        let config = args
            .to_speculative_config()
            .expect("geometry should be valid")
            .expect("should build config");
        assert_eq!(config.window_size_ms, 3000);
        assert_eq!(config.overlap_ms, 500);
        assert!(config.adaptive);
        assert!(!config.tolerance.always_correct);
    }

    #[test]
    fn speculative_config_respects_custom_window() {
        let mut args = minimal_args();
        args.speculative = true;
        args.speculative_window_ms = Some(5000);
        args.speculative_overlap_ms = Some(1000);
        let config = args
            .to_speculative_config()
            .expect("geometry should be valid")
            .expect("should build config");
        assert_eq!(config.window_size_ms, 5000);
        assert_eq!(config.overlap_ms, 1000);
    }

    #[test]
    fn speculative_config_respects_model_names() {
        let mut args = minimal_args();
        args.speculative = true;
        args.fast_model = Some("whisper-tiny".to_owned());
        args.quality_model = Some("whisper-large".to_owned());
        let config = args
            .to_speculative_config()
            .expect("geometry should be valid")
            .expect("should build config");
        assert_eq!(config.fast_model_name, "whisper-tiny");
        assert_eq!(config.quality_model_name, "whisper-large");
    }

    #[test]
    fn speculative_config_no_adaptive_disables_adaptive() {
        let mut args = minimal_args();
        args.speculative = true;
        args.no_adaptive = true;
        let config = args
            .to_speculative_config()
            .expect("geometry should be valid")
            .expect("should build config");
        assert!(!config.adaptive);
    }

    #[test]
    fn speculative_config_always_correct_mode() {
        let mut args = minimal_args();
        args.speculative = true;
        args.always_correct = true;
        let config = args
            .to_speculative_config()
            .expect("geometry should be valid")
            .expect("should build config");
        assert!(config.tolerance.always_correct);
    }

    #[test]
    fn speculative_config_custom_wer_tolerance() {
        let mut args = minimal_args();
        args.speculative = true;
        args.correction_tolerance_wer = Some(0.25);
        let config = args
            .to_speculative_config()
            .expect("geometry should be valid")
            .expect("should build config");
        assert!((config.tolerance.max_wer - 0.25).abs() < 0.001);
    }

    #[test]
    fn robot_summary_includes_speculative() {
        let mut args = minimal_args();
        args.speculative = true;
        let summary = args.robot_summary();
        assert_eq!(summary["speculative"], true);
    }

    #[test]
    fn cli_parse_speculative_flag() {
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
                assert!(args.speculative);
            }
            other => panic!("expected Transcribe, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_speculative_with_models() {
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
        ])
        .expect("should parse");
        match cli.command {
            Command::Transcribe(args) => {
                assert!(args.speculative);
                assert_eq!(args.fast_model.as_deref(), Some("whisper-tiny"));
                assert_eq!(args.quality_model.as_deref(), Some("whisper-large"));
                assert_eq!(args.speculative_window_ms, Some(5000));
            }
            other => panic!("expected Transcribe, got: {other:?}"),
        }
    }

    #[test]
    fn speculative_config_default_model_names_are_auto_sentinels() {
        let mut args = minimal_args();
        args.speculative = true;
        let config = args
            .to_speculative_config()
            .expect("geometry should be valid")
            .expect("should build config");
        assert_eq!(config.fast_model_name, "auto-fast");
        assert_eq!(config.quality_model_name, "auto-quality");
        assert!(config.emit_events);
    }

    // ---------------------------------------------------------------
    // Speculative CLI integration: to_request() now succeeds with
    // --speculative and propagates a SpeculativeRequest through
    // BackendParams instead of bailing with FW-INVALID-REQUEST.
    // ---------------------------------------------------------------

    #[test]
    fn to_request_with_speculative_flag_populates_backend_params() {
        let mut args = minimal_args();
        args.speculative = true;
        args.fast_model = Some("whisper-tiny".to_owned());
        args.quality_model = Some("whisper-large".to_owned());
        args.speculative_window_ms = Some(2500);
        args.speculative_overlap_ms = Some(400);
        args.correction_tolerance_wer = Some(0.2);
        args.no_adaptive = false;
        args.always_correct = false;

        let request = args.to_request().expect("speculative request should build");
        let spec = request
            .backend_params
            .speculative
            .as_ref()
            .expect("backend_params.speculative should be populated when --speculative is set");
        assert_eq!(spec.fast_model_name, "whisper-tiny");
        assert_eq!(spec.quality_model_name, "whisper-large");
        assert_eq!(spec.window_size_ms, 2500);
        assert_eq!(spec.overlap_ms, 400);
        assert_eq!(spec.max_wer_tolerance, Some(0.2));
        assert!(spec.adaptive);
        assert!(!spec.always_correct);
    }

    #[test]
    fn to_request_without_speculative_flag_leaves_backend_params_none() {
        let args = minimal_args();
        let request = args
            .to_request()
            .expect("non-speculative request should build");
        assert!(
            request.backend_params.speculative.is_none(),
            "BackendParams.speculative must be None when --speculative is not set"
        );
    }

    #[test]
    fn to_request_rejects_speculative_overlap_without_forward_progress() {
        let mut args = minimal_args();
        args.speculative = true;
        args.speculative_window_ms = Some(1_000);
        args.speculative_overlap_ms = Some(5_000);

        let result = args.to_request();
        assert!(
            matches!(result, Err(FwError::InvalidRequest(message)) if message.contains("--speculative-overlap-ms")),
            "oversized overlap must be rejected instead of normalized to a 1 ms step"
        );
    }

    #[test]
    fn to_request_accepts_speculative_overlap_with_one_ms_step() {
        let mut args = minimal_args();
        args.speculative = true;
        args.speculative_window_ms = Some(1_000);
        args.speculative_overlap_ms = Some(999);

        let request = args.to_request().expect("positive step should be valid");
        let spec = request.backend_params.speculative.as_ref().expect("set");
        assert_eq!(spec.window_size_ms, 1_000);
        assert_eq!(spec.overlap_ms, 999);
    }

    #[test]
    fn to_request_rejects_zero_speculative_window() {
        let mut args = minimal_args();
        args.speculative = true;
        args.speculative_window_ms = Some(0);
        args.speculative_overlap_ms = Some(0);

        let result = args.to_request();
        assert!(
            matches!(result, Err(FwError::InvalidRequest(message)) if message.contains("--speculative-window-ms")),
            "zero-sized speculative window must be rejected"
        );
    }

    #[test]
    fn to_request_speculative_always_correct_propagates_to_request() {
        let mut args = minimal_args();
        args.speculative = true;
        args.always_correct = true;
        args.no_adaptive = true;
        let request = args.to_request().expect("should build");
        let spec = request.backend_params.speculative.as_ref().expect("set");
        assert!(spec.always_correct);
        assert!(!spec.adaptive);
    }

    #[test]
    fn to_speculative_request_matches_to_speculative_config_for_user_visible_knobs() {
        let mut args = minimal_args();
        args.speculative = true;
        args.fast_model = Some("alpha".to_owned());
        args.quality_model = Some("omega".to_owned());
        args.speculative_window_ms = Some(4_321);
        args.speculative_overlap_ms = Some(123);
        args.correction_tolerance_wer = Some(0.07);
        args.no_adaptive = false;
        args.always_correct = false;

        let request_form = args
            .to_speculative_request()
            .expect("request geometry")
            .expect("request form");
        let config_form = args
            .to_speculative_config()
            .expect("config geometry")
            .expect("config form");
        assert_eq!(request_form.fast_model_name, config_form.fast_model_name);
        assert_eq!(
            request_form.quality_model_name,
            config_form.quality_model_name
        );
        assert_eq!(request_form.window_size_ms, config_form.window_size_ms);
        assert_eq!(request_form.overlap_ms, config_form.overlap_ms);
        assert_eq!(
            request_form.max_wer_tolerance,
            Some(config_form.tolerance.max_wer)
        );
        assert_eq!(request_form.adaptive, config_form.adaptive);
        assert_eq!(
            request_form.always_correct,
            config_form.tolerance.always_correct
        );
    }

    #[test]
    fn runs_output_format_variants_are_distinct_and_parseable() {
        assert_ne!(RunsOutputFormat::Plain, RunsOutputFormat::Json);
        assert_ne!(RunsOutputFormat::Plain, RunsOutputFormat::Ndjson);
        assert_ne!(RunsOutputFormat::Json, RunsOutputFormat::Ndjson);

        let cli = Cli::try_parse_from(["franken_whisper", "runs", "--format", "json"])
            .expect("should parse");
        match cli.command {
            Command::Runs(args) => {
                assert_eq!(args.format, RunsOutputFormat::Json);
                assert_eq!(args.limit, 20);
                assert!(args.id.is_none());
            }
            other => panic!("expected Runs, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_tty_audio_decode_defaults_to_fail_closed() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "tty-audio",
            "decode",
            "--output",
            "out.raw",
        ])
        .expect("should parse");
        match cli.command {
            Command::TtyAudio { command } => match command {
                TtyAudioCommand::Decode { output, recovery } => {
                    assert_eq!(output, PathBuf::from("out.raw"));
                    assert_eq!(recovery, TtyAudioRecoveryPolicy::FailClosed);
                }
                other => panic!("expected Decode, got: {other:?}"),
            },
            other => panic!("expected TtyAudio, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_tty_audio_control_ack() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "tty-audio",
            "control",
            "ack",
            "--up-to-seq",
            "42",
        ])
        .expect("should parse");
        match cli.command {
            Command::TtyAudio { command } => match command {
                TtyAudioCommand::Control { command: ctrl } => match ctrl {
                    TtyAudioControlCommand::Ack { up_to_seq } => {
                        assert_eq!(up_to_seq, 42);
                    }
                    other => panic!("expected Ack, got: {other:?}"),
                },
                other => panic!("expected Control, got: {other:?}"),
            },
            other => panic!("expected TtyAudio, got: {other:?}"),
        }
    }

    #[test]
    fn cli_parse_tty_audio_control_retransmit_request_with_sequences() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "tty-audio",
            "control",
            "retransmit-request",
            "--sequences",
            "1,5,10",
        ])
        .expect("should parse");
        match cli.command {
            Command::TtyAudio { command } => match command {
                TtyAudioCommand::Control { command: ctrl } => match ctrl {
                    TtyAudioControlCommand::RetransmitRequest { sequences } => {
                        assert_eq!(sequences, vec![1, 5, 10]);
                    }
                    other => panic!("expected RetransmitRequest, got: {other:?}"),
                },
                other => panic!("expected Control, got: {other:?}"),
            },
            other => panic!("expected TtyAudio, got: {other:?}"),
        }
    }

    #[test]
    fn confidential_evaluation_cli_debug_redacts_every_path() {
        let args = ConfidentialEvaluationArgs {
            input_root: PathBuf::from("/PRIVATE/INPUT/ROOT"),
            manifest: PathBuf::from("/PRIVATE/MANIFEST.json"),
            output: PathBuf::from("/PRIVATE/OUTPUT.json"),
        };
        let debug = format!("{args:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("PRIVATE"));
        assert!(!debug.contains("MANIFEST"));
        assert!(!debug.contains("OUTPUT"));
    }

    #[test]
    fn public_corpus_cli_parses_build_and_redacts_paths() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "diarization-corpus",
            "build",
            "--input-root",
            "/EXTERNAL/PUBLIC",
            "--descriptor",
            "/EXTERNAL/PUBLIC/descriptor.json",
            "--output",
            "/EXTERNAL/OUTPUT/bundle.json",
            "--license-ack",
            "accept-ami-cc-by-4.0",
        ])
        .expect("public corpus command");
        let Command::DiarizationCorpus {
            command: PublicCorpusCommand::Build(args),
        } = cli.command
        else {
            panic!("expected public corpus build");
        };
        assert_eq!(args.license_ack, "accept-ami-cc-by-4.0");
        let debug = format!("{args:?}");
        assert!(!debug.contains("EXTERNAL"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn public_corpus_cli_parses_registry() {
        let cli = Cli::try_parse_from(["franken_whisper", "diarization-corpus", "registry"])
            .expect("public corpus registry");
        assert!(matches!(
            cli.command,
            Command::DiarizationCorpus {
                command: PublicCorpusCommand::Registry
            }
        ));
    }

    #[test]
    fn public_corpus_cli_parses_voxconverse_preparation_and_redacts_paths() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "diarization-corpus",
            "prepare-voxconverse",
            "--input-root",
            "/PRIVATE/VOX",
            "--development-audio-root",
            "/PRIVATE/VOX/dev-audio",
            "--test-audio-root",
            "/PRIVATE/VOX/test-audio",
            "--annotation-root",
            "/PRIVATE/VOX/labels",
            "--output",
            "/PRIVATE/VOX/descriptor.json",
            "--source-version",
            "voxconverse-fixture-v1",
            "--license-ack",
            "accept-voxconverse-cc-by-4.0-and-original-copyright",
        ])
        .expect("VoxConverse preparation command");
        let debug = format!("{cli:?}");
        assert!(!debug.contains("PRIVATE"));
        assert!(debug.contains("<redacted>"));
        let Command::DiarizationCorpus {
            command: PublicCorpusCommand::PrepareVoxconverse(args),
        } = cli.command
        else {
            panic!("expected VoxConverse preparation command");
        };
        assert_eq!(args.source_version, "voxconverse-fixture-v1");
        assert_eq!(
            args.license_ack,
            "accept-voxconverse-cc-by-4.0-and-original-copyright"
        );
    }

    #[test]
    fn public_corpus_cli_parses_ablation_and_redacts_every_path() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "diarization-corpus",
            "ablate",
            "--input-root",
            "/EXTERNAL/PUBLIC",
            "--descriptor",
            "/EXTERNAL/PUBLIC/descriptor.json",
            "--bundle-output",
            "/EXTERNAL/OUTPUT/bundle.json",
            "--output",
            "/EXTERNAL/OUTPUT/ablation.json",
            "--license-ack",
            "accept-ami-cc-by-4.0",
            "--maximum-recording-duration-ms",
            "300000",
            "--stage",
            "development",
        ])
        .expect("public corpus ablation command");
        let Command::DiarizationCorpus {
            command: PublicCorpusCommand::Ablate(args),
        } = cli.command
        else {
            panic!("expected public corpus ablation");
        };
        assert_eq!(args.license_ack, "accept-ami-cc-by-4.0");
        assert_eq!(args.maximum_recording_duration_ms, Some(300_000));
        assert_eq!(args.stage, PublicCorpusEvaluationStageArg::Development);
        assert!(args.locked_development_evidence.is_none());
        let debug = format!("{args:?}");
        assert!(!debug.contains("EXTERNAL"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn public_corpus_cli_parses_sidecar_study_and_redacts_every_path() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "diarization-corpus",
            "sidecar-study",
            "--input-root",
            "/SIDE_CAR_SECRET/INPUT",
            "--descriptor",
            "/SIDE_CAR_SECRET/INPUT/descriptor.json",
            "--bundle-output",
            "/SIDE_CAR_SECRET/OUTPUT/bundle.json",
            "--output",
            "/SIDE_CAR_SECRET/OUTPUT/study.json",
            "--license-ack",
            "accept-ami-cc-by-4.0",
            "--maximum-recording-duration-ms",
            "120000",
            "--stage",
            "certification",
            "--locked-development-evidence",
            "/SIDE_CAR_SECRET/LOCK/development.json",
        ])
        .expect("public corpus sidecar-study command");
        let debug = format!("{cli:?}");
        assert!(!debug.contains("SIDE_CAR_SECRET"));
        assert_eq!(debug.matches("<redacted>").count(), 5);

        let Command::DiarizationCorpus {
            command: PublicCorpusCommand::SidecarStudy(args),
        } = cli.command
        else {
            panic!("expected public corpus sidecar study");
        };
        assert_eq!(args.license_ack, "accept-ami-cc-by-4.0");
        assert_eq!(args.maximum_recording_duration_ms, Some(120_000));
        assert_eq!(args.stage, PublicCorpusEvaluationStageArg::Certification);
        assert!(args.locked_development_evidence.is_some());
    }

    #[test]
    fn public_corpus_cli_parses_model_comparison_and_redacts_every_path() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "diarization-corpus",
            "compare-models",
            "--input-root",
            "/MODEL_COMPARE_SECRET/INPUT",
            "--descriptor",
            "/MODEL_COMPARE_SECRET/INPUT/descriptor.json",
            "--bundle-output",
            "/MODEL_COMPARE_SECRET/OUTPUT/bundle.json",
            "--output",
            "/MODEL_COMPARE_SECRET/OUTPUT/evidence.json",
            "--license-ack",
            "accept-ami-cc-by-4.0",
        ])
        .expect("public corpus model-comparison command");
        let debug = format!("{cli:?}");
        assert!(!debug.contains("MODEL_COMPARE_SECRET"));
        assert_eq!(debug.matches("<redacted>").count(), 4);

        let Command::DiarizationCorpus {
            command: PublicCorpusCommand::CompareModels(args),
        } = cli.command
        else {
            panic!("expected public corpus model comparison");
        };
        assert_eq!(args.license_ack, "accept-ami-cc-by-4.0");
    }

    #[test]
    fn differential_oracle_cli_parses_registry() {
        let cli = Cli::try_parse_from(["franken_whisper", "diarization-oracle", "registry"])
            .expect("oracle registry");
        assert!(matches!(
            cli.command,
            Command::DiarizationOracle {
                command: DifferentialOracleCommand::Registry
            }
        ));
    }

    #[test]
    fn differential_oracle_cli_parses_run_and_redacts_every_path() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "diarization-oracle",
            "run",
            "--tool",
            "nemo-spectral",
            "--audio",
            "/PRIVATE/call.m4a",
            "--native",
            "/PRIVATE/native.json",
            "--reference",
            "/PRIVATE/reference.json",
            "--output",
            "/PRIVATE/report.json",
            "--timeout-seconds",
            "90",
        ])
        .expect("oracle run");
        let Command::DiarizationOracle {
            command: DifferentialOracleCommand::Run(args),
        } = cli.command
        else {
            panic!("expected differential oracle run");
        };
        assert_eq!(args.tool, DifferentialOracleToolArg::NemoSpectral);
        assert_eq!(args.timeout_seconds, 90);
        let debug = format!("{args:?}");
        assert!(!debug.contains("PRIVATE"));
        assert!(!debug.contains("call.m4a"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn sortformer_diarize_cli_is_explicit_and_redacts_every_path() {
        let cli = Cli::try_parse_from([
            "franken_whisper",
            "sortformer-diarize",
            "--input",
            "/PRIVATE/call.m4a",
            "--receipt",
            "/PRIVATE/conversion-receipt.json",
            "--package",
            "/PRIVATE/weights.safetensors",
        ])
        .expect("explicit native Sortformer command");
        let Command::SortformerDiarize(args) = cli.command else {
            panic!("expected native Sortformer command");
        };
        let debug = format!("{args:?}");
        assert_eq!(debug.matches("<redacted>").count(), 4);
        assert!(!debug.contains("PRIVATE"));
        assert!(!debug.contains("call.m4a"));
        assert!(!debug.contains("weights.safetensors"));

        let cached = Cli::try_parse_from([
            "franken_whisper",
            "sortformer-diarize",
            "--input",
            "/PRIVATE/call.m4a",
        ])
        .expect("cached native Sortformer command");
        let Command::SortformerDiarize(cached) = cached.command else {
            panic!("expected cached native Sortformer command");
        };
        assert!(cached.receipt.is_none());
        assert!(cached.package.is_none());
        assert!(cached.speaker_hints.is_none());
    }

    #[test]
    fn robot_listen_parses_defaults_and_full_flag_set() {
        // Defaults (bd-rt-listen-cmd-i48i consolidated flag list).
        let cli = Cli::try_parse_from(["fw", "robot", "listen"]).expect("bare listen");
        let Command::Robot {
            command: RobotCommand::Listen(args),
        } = cli.command
        else {
            panic!("expected robot listen");
        };
        assert!(matches!(args.source, ListenSourceArg::Mic));
        assert!(matches!(args.capture_backend, CaptureBackendArg::Auto));
        assert!(matches!(args.policy, ListenPolicyArg::Alignatt));
        assert_eq!(args.alignatt_holdback_ms, 200);
        assert_eq!(args.step_ms, 300);
        assert_eq!(args.max_buffer_sec, 12.0);
        assert_eq!(args.max_utterance_sec, 90.0);
        assert_eq!(args.stats_interval_sec, 30.0);
        assert_eq!(args.capture_buffer_sec, 30.0);
        assert_eq!(args.stdin_rate, 16_000);
        assert_eq!(args.stdin_channels, 1);
        assert!(matches!(args.stdin_format, StdinFormatArg::S16le));
        assert_eq!(args.vad_gate_db, 9.0);
        assert_eq!(args.vad_min_speech_ms, 250);
        assert_eq!(args.vad_endpoint_ms, 600);
        assert!(!args.no_partials && !args.no_vad && !args.no_context);
        assert!(!args.list_devices && !args.realtime_pace);
        assert!(args.language.is_none() && args.fast_model.is_none());

        // Full flag exercise.
        let cli = Cli::try_parse_from([
            "fw",
            "robot",
            "listen",
            "--source",
            "file-replay",
            "--input",
            "/tmp/a.wav",
            "--realtime-pace",
            "--fast-model",
            "tiny",
            "--language",
            "de",
            "--step-ms",
            "500",
            "--max-buffer-sec",
            "8",
            "--max-seconds",
            "60",
            "--max-utterance-sec",
            "30",
            "--no-partials",
            "--stats-interval-sec",
            "5",
            "--no-context",
            "--capture-buffer-sec",
            "10",
            "--vad-gate-db",
            "6",
            "--vad-min-speech-ms",
            "200",
            "--vad-endpoint-ms",
            "800",
        ])
        .expect("full listen flags");
        let Command::Robot {
            command: RobotCommand::Listen(args),
        } = cli.command
        else {
            panic!("expected robot listen");
        };
        assert!(matches!(args.source, ListenSourceArg::FileReplay));
        assert_eq!(
            args.input.as_deref(),
            Some(std::path::Path::new("/tmp/a.wav"))
        );
        assert!(args.realtime_pace && args.no_partials && args.no_context);
        assert_eq!(args.language.as_deref(), Some("de"));
        assert_eq!(args.step_ms, 500);
        assert_eq!(args.vad_endpoint_ms, 800);

        // stdin-pcm variants.
        let cli = Cli::try_parse_from([
            "fw",
            "robot",
            "listen",
            "--source",
            "stdin-pcm",
            "--stdin-rate",
            "48000",
            "--stdin-channels",
            "2",
            "--stdin-format",
            "f32le",
        ])
        .expect("stdin listen");
        let Command::Robot {
            command: RobotCommand::Listen(args),
        } = cli.command
        else {
            panic!("expected robot listen");
        };
        assert!(matches!(args.source, ListenSourceArg::StdinPcm));
        assert_eq!((args.stdin_rate, args.stdin_channels), (48_000, 2));
        assert!(matches!(args.stdin_format, StdinFormatArg::F32le));

        // Unknown policy value is a parse error (robot syntax contract:
        // the binary wrapper turns this into one JSON envelope + exit 2).
        assert!(Cli::try_parse_from(["fw", "robot", "listen", "--policy", "nonsense"]).is_err());
    }

    #[test]
    fn agent_discovery_commands_parse_with_stable_shapes() {
        let capabilities =
            Cli::try_parse_from(["fw", "capabilities", "--json"]).expect("capabilities command");
        assert!(matches!(
            capabilities.command,
            Command::Capabilities(CapabilitiesArgs { json: true })
        ));

        let models = Cli::try_parse_from(["fw", "models", "--json"]).expect("models command");
        assert!(matches!(
            models.command,
            Command::Models(ModelsArgs { json: true })
        ));

        let pull =
            Cli::try_parse_from(["fw", "pull", "sortformer", "--json"]).expect("pull command");
        assert!(matches!(
            pull.command,
            Command::Pull(PullArgs {
                model: PullModelArg::Sortformer,
                json: true,
            })
        ));

        for (arg, expected) in [
            ("tiny-en", PullModelArg::TinyEn),
            ("tiny", PullModelArg::Tiny),
        ] {
            let parsed = Cli::try_parse_from(["fw", "pull", arg, "--json"])
                .unwrap_or_else(|e| panic!("pull {arg} must parse: {e}"));
            match parsed.command {
                Command::Pull(args) => {
                    assert_eq!(args.model, expected, "pull {arg} target");
                    assert!(args.json);
                }
                other => panic!("pull {arg} parsed to {other:?}"),
            }
        }
        let pull_default = Cli::try_parse_from(["fw", "pull"]).expect("default pull command");
        assert!(matches!(
            pull_default.command,
            Command::Pull(PullArgs {
                model: PullModelArg::All,
                json: false,
            })
        ));

        let doctor =
            Cli::try_parse_from(["fw", "doctor", "--json", "--strict"]).expect("doctor command");
        assert!(matches!(
            doctor.command,
            Command::Doctor(DoctorArgs {
                json: true,
                strict: true,
                ..
            })
        ));

        let triage = Cli::try_parse_from(["fw", "agent", "triage", "--strict"])
            .expect("robot alias triage command");
        assert!(matches!(
            triage.command,
            Command::Robot {
                command: RobotCommand::Triage(HealthArgs { strict: true, .. })
            }
        ));

        let guide = Cli::try_parse_from(["fw", "robot-docs", "guide"]).expect("robot docs guide");
        assert!(matches!(
            guide.command,
            Command::RobotDocs {
                command: RobotDocsCommand::Guide
            }
        ));
    }

    #[test]
    fn underscore_value_aliases_remain_agent_friendly() {
        let mut args = minimal_args();
        args.backend = BackendKind::WhisperCpp;
        assert_eq!(args.backend, BackendKind::WhisperCpp);

        let parsed = Cli::try_parse_from([
            "fw",
            "transcribe",
            "--input",
            "test.wav",
            "--backend",
            "whisper_cpp",
            "--diarization-engine",
            "ecapa_fused",
        ])
        .expect("underscore aliases");
        let Command::Transcribe(parsed) = parsed.command else {
            panic!("expected transcribe command");
        };
        assert_eq!(parsed.backend, BackendKind::WhisperCpp);
        assert_eq!(parsed.diarization_engine, DiarizationEngine::EcapaFused);
    }

    #[test]
    fn version_report_discloses_model_distribution_boundary() {
        let error = Cli::try_parse_from(["fw", "--version"])
            .expect_err("version is represented as a clap display result");
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayVersion);
        let rendered = error.to_string();
        assert!(rendered.contains(env!("CARGO_PKG_VERSION")));
        assert!(rendered.contains("model weights: not bundled"));
        assert!(rendered.contains("hash-pinned GitHub release artifact"));
    }
}
