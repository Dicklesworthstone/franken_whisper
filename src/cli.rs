use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::json;

use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, BackendParams, DecodingParams, DiarizationConfig, InputSource, OutputFormat,
    SpeakerConstraints, TimestampLevel, TranscribeRequest, VadParams,
};
use crate::sync::ConflictPolicy;

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
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Transcribe(Box<TranscribeArgs>),
    Robot {
        #[command(subcommand)]
        command: RobotCommand,
    },
    Runs(RunsArgs),
    Sync {
        #[command(subcommand)]
        command: SyncCommand,
    },
    TtyAudio {
        #[command(subcommand)]
        command: TtyAudioCommand,
    },
    Tui,
    /// Download YouTube audio (videos / playlists / a URL file) and
    /// transcribe each into a markdown + JSON pair.
    Youtube(Box<YoutubeArgs>),
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

    /// Enable speaker diarization.
    #[arg(long)]
    pub diarize: bool,

    /// Maximum concurrent downloads.
    #[arg(long, default_value_t = 3)]
    pub concurrency: usize,

    /// Delete each audio file after its transcript is written.
    #[arg(long)]
    pub no_keep_audio: bool,

    /// Do not retry videos previously marked failed in the manifest.
    #[arg(long)]
    pub no_retry: bool,

    /// Stop the whole run on the first per-video failure.
    #[arg(long)]
    pub abort_on_error: bool,

    /// Emit the final run summary as JSON on stdout (for scripting).
    #[arg(long)]
    pub json_summary: bool,
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
            diarize: self.diarize,
            concurrency: self.concurrency,
            keep_audio: !self.no_keep_audio,
            retry_failed: !self.no_retry,
            abort_on_error: self.abort_on_error,
        })
    }
}

#[derive(Debug, Subcommand)]
pub enum RobotCommand {
    Run(Box<TranscribeArgs>),
    Schema,
    Backends,
    Health(HealthArgs),
    RoutingHistory(RoutingHistoryArgs),
}

#[derive(Debug, Args)]
pub struct HealthArgs {
    /// Path to frankensqlite database file.
    #[arg(long, default_value = ".franken_whisper/storage.sqlite3")]
    pub db: PathBuf,
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

    /// Request speaker diarization.
    #[arg(long)]
    pub diarize: bool,

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

    /// Exact number of speakers (insanely-fast diarization).
    #[arg(long)]
    pub num_speakers: Option<u32>,

    /// Minimum number of speakers (insanely-fast diarization).
    #[arg(long)]
    pub min_speakers: Option<u32>,

    /// Maximum number of speakers (insanely-fast diarization).
    #[arg(long)]
    pub max_speakers: Option<u32>,

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
    pub fn to_request(&self) -> FwResult<TranscribeRequest> {
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

        let input = if let Some(path) = &self.input {
            InputSource::File { path: path.clone() }
        } else if self.stdin {
            InputSource::Stdin {
                hint_extension: None,
            }
        } else {
            InputSource::Microphone {
                seconds: self.mic_seconds,
                device: self.mic_device.clone(),
                ffmpeg_format: self.mic_ffmpeg_format.clone(),
                ffmpeg_source: self.mic_ffmpeg_source.clone(),
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
                model_path: self.vad_model.clone(),
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

        // Speaker constraints — only build if any field is set.
        let speaker_constraints = if self.num_speakers.is_some()
            || self.min_speakers.is_some()
            || self.max_speakers.is_some()
        {
            Some(SpeakerConstraints {
                num_speakers: self.num_speakers,
                min_speakers: self.min_speakers,
                max_speakers: self.max_speakers,
            })
        } else {
            None
        };

        // Diarization config — only build if any diarization-specific flag is set.
        let diarization_config =
            if self.no_stem || self.diarization_model.is_some() || self.suppress_numerals {
                Some(DiarizationConfig {
                    no_stem: self.no_stem,
                    whisper_model: self.diarization_model.clone(),
                    suppress_numerals: self.suppress_numerals,
                    device: self.gpu_device.clone(),
                    batch_size: self.batch_size,
                })
            } else {
                None
            };

        let backend_params = BackendParams {
            output_formats,
            timestamp_level: self.timestamp_level,
            decoding,
            vad,
            speaker_constraints,
            diarization_config,
            gpu_device: self.gpu_device.clone(),
            flash_attention: if self.flash_attention {
                Some(true)
            } else {
                None
            },
            insanely_fast_hf_token: self.hf_token.clone(),
            insanely_fast_transcript_path: self.transcript_path.clone(),
            no_timestamps: self.no_timestamps,
            detect_language_only: self.detect_language_only,
            batch_size: self.batch_size,
            split_on_word: self.split_on_word,
            threads: self.threads,
            processors: self.processors,
            no_gpu: self.no_gpu,
            prompt: self.prompt.clone(),
            carry_initial_prompt: self.carry_initial_prompt,
            no_fallback: self.no_fallback,
            suppress_nst: self.suppress_nst,
            offset_ms: self.offset_ms,
            duration_ms: self.duration_ms,
            audio_ctx: self.audio_ctx,
            word_threshold: self.word_threshold,
            suppress_regex: self.suppress_regex.clone(),
            tiny_diarize: self.tiny_diarize,
            word_timestamps: None,
            insanely_fast_tuning: None,
            alignment: None,
            punctuation: None,
            source_separation: None,
            speculative: self.to_speculative_request(),
        };

        Ok(TranscribeRequest {
            input,
            backend: self.backend,
            model: self.model.clone(),
            language: self.language.clone(),
            translate: self.translate,
            diarize: self.diarize,
            persist: !self.no_persist,
            db_path: self.db.clone(),
            timeout_ms: self.timeout.map(|secs| secs.saturating_mul(1000)),
            backend_params,
        })
    }

    #[must_use]
    pub fn robot_summary(&self) -> serde_json::Value {
        json!({
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "translate": self.translate,
            "diarize": self.diarize,
            "persist": !self.no_persist,
            "db": self.db,
            "speculative": self.speculative,
        })
    }

    /// Build a `SpeculativeConfig` from CLI arguments.
    /// Returns `None` if `--speculative` is not set.
    #[must_use]
    pub fn to_speculative_config(&self) -> Option<crate::streaming::SpeculativeConfig> {
        if !self.speculative {
            return None;
        }

        let window_size_ms = self.speculative_window_ms.unwrap_or(3000);
        let raw_overlap_ms = self.speculative_overlap_ms.unwrap_or(500);
        let overlap_ms = raw_overlap_ms.min(window_size_ms.saturating_sub(1));

        Some(crate::streaming::SpeculativeConfig {
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
        })
    }

    /// Build a serde-friendly [`SpeculativeRequest`] for storage in
    /// `BackendParams.speculative`. Returns `None` if `--speculative` is not set.
    ///
    /// The orchestrator converts this into a
    /// [`crate::streaming::SpeculativeConfig`] at dispatch time, keeping
    /// `model.rs` free of any dependency on `streaming.rs`.
    #[must_use]
    pub fn to_speculative_request(&self) -> Option<crate::model::SpeculativeRequest> {
        if !self.speculative {
            return None;
        }
        let window_size_ms = self.speculative_window_ms.unwrap_or(3000);
        let raw_overlap_ms = self.speculative_overlap_ms.unwrap_or(500);
        let overlap_ms = raw_overlap_ms.min(window_size_ms.saturating_sub(1));
        Some(crate::model::SpeculativeRequest {
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
            max_wer_tolerance: self.correction_tolerance_wer,
            adaptive: !self.no_adaptive,
            always_correct: self.always_correct,
        })
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
            num_speakers: None,
            min_speakers: None,
            max_speakers: None,
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

    // --- Speaker constraints ---

    #[test]
    fn speaker_constraints_none_when_no_speaker_args() {
        let args = minimal_args();
        let request = args.to_request().expect("should succeed");
        assert!(request.backend_params.speaker_constraints.is_none());
    }

    #[test]
    fn speaker_constraints_built_from_num_speakers() {
        let mut args = minimal_args();
        args.num_speakers = Some(3);
        let request = args.to_request().expect("should succeed");
        let sc = request
            .backend_params
            .speaker_constraints
            .expect("should be Some");
        assert_eq!(sc.num_speakers, Some(3));
        assert!(sc.min_speakers.is_none());
        assert!(sc.max_speakers.is_none());
    }

    #[test]
    fn speaker_constraints_built_from_min_and_max() {
        let mut args = minimal_args();
        args.min_speakers = Some(2);
        args.max_speakers = Some(8);
        let request = args.to_request().expect("should succeed");
        let sc = request
            .backend_params
            .speaker_constraints
            .expect("should be Some");
        assert!(sc.num_speakers.is_none());
        assert_eq!(sc.min_speakers, Some(2));
        assert_eq!(sc.max_speakers, Some(8));
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
        args.num_speakers = Some(3);
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
        assert!(request.backend_params.speaker_constraints.is_some());
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
    fn speaker_constraints_all_fields() {
        let mut args = minimal_args();
        args.num_speakers = Some(4);
        args.min_speakers = Some(2);
        args.max_speakers = Some(6);
        let request = args.to_request().expect("should succeed");
        let sc = request
            .backend_params
            .speaker_constraints
            .expect("should be Some");
        assert_eq!(sc.num_speakers, Some(4));
        assert_eq!(sc.min_speakers, Some(2));
        assert_eq!(sc.max_speakers, Some(6));
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
        assert!(args.to_speculative_config().is_none());
    }

    #[test]
    fn speculative_config_built_when_flag_set() {
        let mut args = minimal_args();
        args.speculative = true;
        let config = args.to_speculative_config().expect("should build config");
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
        let config = args.to_speculative_config().expect("should build config");
        assert_eq!(config.window_size_ms, 5000);
        assert_eq!(config.overlap_ms, 1000);
    }

    #[test]
    fn speculative_config_respects_model_names() {
        let mut args = minimal_args();
        args.speculative = true;
        args.fast_model = Some("whisper-tiny".to_owned());
        args.quality_model = Some("whisper-large".to_owned());
        let config = args.to_speculative_config().expect("should build config");
        assert_eq!(config.fast_model_name, "whisper-tiny");
        assert_eq!(config.quality_model_name, "whisper-large");
    }

    #[test]
    fn speculative_config_no_adaptive_disables_adaptive() {
        let mut args = minimal_args();
        args.speculative = true;
        args.no_adaptive = true;
        let config = args.to_speculative_config().expect("should build config");
        assert!(!config.adaptive);
    }

    #[test]
    fn speculative_config_always_correct_mode() {
        let mut args = minimal_args();
        args.speculative = true;
        args.always_correct = true;
        let config = args.to_speculative_config().expect("should build config");
        assert!(config.tolerance.always_correct);
    }

    #[test]
    fn speculative_config_custom_wer_tolerance() {
        let mut args = minimal_args();
        args.speculative = true;
        args.correction_tolerance_wer = Some(0.25);
        let config = args.to_speculative_config().expect("should build config");
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
        let config = args.to_speculative_config().expect("should build config");
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
    fn to_request_speculative_overlap_is_clamped_below_window_size() {
        let mut args = minimal_args();
        args.speculative = true;
        // overlap > window: must clamp to window - 1 (matches to_speculative_config behavior).
        args.speculative_window_ms = Some(1_000);
        args.speculative_overlap_ms = Some(5_000);
        let request = args.to_request().expect("should build");
        let spec = request.backend_params.speculative.as_ref().expect("set");
        assert_eq!(spec.window_size_ms, 1_000);
        assert_eq!(spec.overlap_ms, 999);
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

        let request_form = args.to_speculative_request().expect("request form");
        let config_form = args.to_speculative_config().expect("config form");
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
}
