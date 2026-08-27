//! Cancel-correct YouTube ingestion pipeline + manifest state machine.
//!
//! Drives the end-to-end flow for the `youtube` subcommand: resolve inputs
//! (explicit URLs, a batch file, and/or playlist URLs) into a video list,
//! download best-audio for each via [`super::ytdlp`], transcribe with the
//! engine, and render markdown + JSON via [`super::render`]. A per-video
//! manifest makes the whole run idempotent and resumable.
//!
//! ## Concurrency & the asupersync boundary
//!
//! The actual transcription is cancel-correct *inside the engine*: the
//! orchestrator owns an asupersync runtime and threads a `CancellationToken`
//! (which honors the global Ctrl+C [`ShutdownController`]) through every
//! pipeline stage. This outer orchestration deliberately does **not** wrap a
//! second asupersync runtime around [`FrankenWhisperEngine::transcribe`]:
//! that call builds and `block_on`s its own runtime, so nesting one inside an
//! asupersync task would be unsound. This is exactly the sanctioned
//! "the dependency owns the runtime" boundary.
//!
//! Downloads (blocking `yt-dlp` subprocesses, where a thread pool is the
//! right tool and async buys nothing) run on a bounded worker pool feeding a
//! capacity-bounded channel; transcription consumes sequentially on the
//! caller thread (the engine already saturates the CPU via intra-op
//! parallelism). The channel bound keeps "downloaded but not yet
//! transcribed" audio — and therefore disk — bounded even for large
//! playlists. Cancellation is uniform: every loop checks the global shutdown
//! flag, in-flight `yt-dlp` children are killed via the cancellation token,
//! and the engine aborts its own work at the next checkpoint.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::cli::ShutdownController;
use crate::error::{FwError, FwResult};
use crate::model::{
    BackendKind, BackendParams, InputSource, TranscribeRequest, TranscriptionSegment,
};
use crate::orchestrator::{CancellationToken, FrankenWhisperEngine};

use super::naming::{self, OutputPaths};
use super::render::{self, RenderInput, RenderRun, RenderVideo, RenderWindowStats};
use super::ytdlp::{self, UrlKind, VideoMeta, VideoRef, YtdlpInfo};

/// Manifest file name written into the output directory.
const MANIFEST_NAME: &str = ".fw_youtube_manifest.json";

/// A video that has failed this many times is not retried again on a plain
/// re-run (it still counts as skipped). `--no-retry` skips any prior failure
/// regardless; deleting the manifest entry forces a fresh attempt.
const MAX_ATTEMPTS: u32 = 3;

/// Per-video processing state. Persisted in the manifest so a re-run resumes
/// exactly where a crash or cancellation left off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum VideoState {
    /// Discovered, not yet started.
    Pending,
    /// Audio downloaded to `audio_path`, not yet transcribed. `attempts`
    /// preserves the cross-run failure budget when cancellation retains a
    /// reusable download after a prior failed run.
    Downloaded {
        audio_path: String,
        #[serde(default)]
        attempts: u32,
    },
    /// Rendering completed and `--no-keep-audio` cleanup must settle before
    /// the terminal `Done` state is durable. A restart can retry deletion when
    /// `audio_path` still exists or treat `NotFound` as an already-settled
    /// cleanup without downloading or transcribing again.
    CleanupPending {
        title: String,
        audio_path: String,
        markdown_path: String,
        json_path: String,
        wall_ms: u64,
        rtf: Option<f64>,
        #[serde(default)]
        attempts: u32,
        #[serde(default)]
        last_error: Option<String>,
    },
    /// Fully processed; markdown + JSON written.
    Done {
        audio_path: Option<String>,
        markdown_path: String,
        json_path: String,
    },
    /// Failed after `attempts` tries; `error` is the last message.
    Failed { error: String, attempts: u32 },
}

/// One manifest entry: the discovered video plus its current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub id: String,
    pub title: String,
    pub url: String,
    #[serde(default)]
    pub state: Option<VideoState>,
}

/// The run manifest: per-video state, keyed by video id for deterministic
/// ordering and O(log n) lookup; `order` preserves discovery order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub order: Vec<String>,
    #[serde(default)]
    pub entries: BTreeMap<String, ManifestEntry>,
}

impl Manifest {
    fn load(path: &Path) -> FwResult<Self> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
                FwError::InvalidRequest(format!(
                    "corrupt manifest at {}: {e} (move it aside to start fresh)",
                    path.display()
                ))
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(FwError::Io(e)),
        }
    }

    /// Atomic write (tmp + rename in the same directory).
    fn save(&self, path: &Path) -> FwResult<()> {
        let body = serde_json::to_string_pretty(self).map_err(FwError::Json)?;
        let tmp = path.with_extension("json.tmp");
        let mut file = std::fs::File::create(&tmp).map_err(FwError::Io)?;
        file.write_all(body.as_bytes()).map_err(FwError::Io)?;
        file.sync_all().map_err(FwError::Io)?;
        drop(file);
        std::fs::rename(&tmp, path).map_err(FwError::Io)?;
        #[cfg(unix)]
        if let Some(parent) = path.parent() {
            std::fs::File::open(parent)
                .and_then(|directory| directory.sync_all())
                .map_err(FwError::Io)?;
        }
        Ok(())
    }

    fn upsert_discovered(&mut self, video: &VideoRef) {
        if !self.entries.contains_key(&video.id) {
            self.order.push(video.id.clone());
            self.entries.insert(
                video.id.clone(),
                ManifestEntry {
                    id: video.id.clone(),
                    title: video.title.clone(),
                    url: video.url.clone(),
                    state: Some(VideoState::Pending),
                },
            );
        }
    }

    fn set_state(&mut self, id: &str, state: VideoState) {
        if let Some(entry) = self.entries.get_mut(id) {
            entry.state = Some(state);
        }
    }

    fn attempts(&self, id: &str) -> u32 {
        match self.entries.get(id).and_then(|e| e.state.as_ref()) {
            Some(
                VideoState::Downloaded { attempts, .. }
                | VideoState::CleanupPending { attempts, .. }
                | VideoState::Failed { attempts, .. },
            ) => *attempts,
            _ => 0,
        }
    }
}

/// Output destination for youtube robot-mode NDJSON events (bd-27v1.1).
///
/// `Off` (the default) preserves the historical human / `--json-summary`
/// behavior with zero emission cost. `Stdout` streams one JSON object per
/// line through the shared locked-stdout robot path
/// ([`crate::robot::emit_event_value`]); `Capture` collects the identical
/// lines in memory so tests can assert on the full event stream without
/// touching process stdout.
#[derive(Clone, Debug, Default)]
pub enum YoutubeRobotEvents {
    /// Robot mode off: no event emission (default).
    #[default]
    Off,
    /// Stream NDJSON events to process stdout.
    Stdout,
    /// Collect NDJSON lines into a shared buffer (tests).
    Capture(std::sync::Arc<std::sync::Mutex<Vec<String>>>),
}

/// Emits sequenced `youtube.*` NDJSON events matching the robot stage-event
/// envelope: every line carries `event`, `schema_version`
/// ([`crate::robot::ROBOT_SCHEMA_VERSION`]), a stable per-run `run_id`, a
/// monotonic `seq`, and an RFC-3339 `ts`. Emission is thread-safe: download
/// workers emit `downloading` / `downloaded` concurrently while the
/// transcription consumer emits the rest, so sequence allocation and the
/// physical stdout/capture write share one emitter-local critical section.
#[derive(Debug)]
pub struct YoutubeEventEmitter {
    output: YoutubeRobotEvents,
    run_id: String,
    seq: std::sync::Mutex<u64>,
}

impl YoutubeEventEmitter {
    /// Build an emitter for the given destination with a fresh `yt-<uuid>`
    /// run id shared by every event of this run.
    pub fn new(output: YoutubeRobotEvents) -> Self {
        Self {
            output,
            run_id: format!("yt-{}", uuid::Uuid::new_v4()),
            seq: std::sync::Mutex::new(0),
        }
    }

    /// Stable run id shared by every event of this run.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Emit one `youtube.<event>` envelope, flattening `payload` fields into
    /// the envelope. In `Off` mode this returns immediately without building
    /// any JSON, so non-robot runs pay nothing.
    pub fn emit(&self, event: &str, payload: serde_json::Value) -> FwResult<()> {
        if matches!(self.output, YoutubeRobotEvents::Off) {
            return Ok(());
        }
        // Allocate the sequence number while holding the same emitter-local
        // critical section through the physical write. An atomic fetch before
        // a separate stdout/capture lock could publish seq=2 before seq=1.
        let mut seq = self
            .seq
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *seq += 1;
        let mut fields = match payload {
            serde_json::Value::Object(fields) => fields,
            _ => serde_json::Map::new(),
        };
        // Payload fields are flattened first; the authoritative envelope is
        // inserted last so callers cannot forge routing or ordering metadata.
        fields.insert(
            "event".to_owned(),
            serde_json::Value::String(format!("youtube.{event}")),
        );
        fields.insert(
            "schema_version".to_owned(),
            serde_json::Value::String(crate::robot::ROBOT_SCHEMA_VERSION.to_owned()),
        );
        fields.insert(
            "run_id".to_owned(),
            serde_json::Value::String(self.run_id.clone()),
        );
        fields.insert("seq".to_owned(), serde_json::Value::from(*seq));
        fields.insert(
            "ts".to_owned(),
            serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
        );
        let envelope = serde_json::Value::Object(fields);
        match &self.output {
            YoutubeRobotEvents::Off => Ok(()),
            YoutubeRobotEvents::Stdout => crate::robot::emit_event_value(&envelope),
            YoutubeRobotEvents::Capture(buffer) => {
                buffer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(serde_json::to_string(&envelope).map_err(FwError::Json)?);
                Ok(())
            }
        }
    }
}

/// Options controlling a YouTube ingestion run.
#[derive(Debug, Clone)]
pub struct YoutubeRunOptions {
    /// Explicit video / playlist URLs.
    pub urls: Vec<String>,
    /// Optional batch file (one URL per line; `#`/`;`/`]` comments, blanks ok).
    pub batch_file: Option<PathBuf>,
    /// Output directory (created if absent).
    pub output_dir: PathBuf,
    /// bd-lun9 batch-wave size: process downloads in waves of this many
    /// videos so untranscribed audio cannot pile up on disk while the
    /// sequential transcription consumer lags behind parallel download
    /// workers. 0 = single wave (all videos at once — historical behavior).
    pub batch_size: usize,

    /// Model spec forwarded to the engine.
    pub model: Option<String>,
    /// Language hint.
    pub language: Option<String>,
    /// Backend selection.
    pub backend: BackendKind,
    /// Enable diarization.
    pub diarize: bool,
    /// Max concurrent downloads.
    pub concurrency: usize,
    /// Keep the downloaded audio files after transcription.
    pub keep_audio: bool,
    /// Retry videos previously marked failed.
    pub retry_failed: bool,
    /// Stop scheduling later waves after the first observed per-video failure.
    /// Downloads already in flight in the current wave may still finish.
    pub abort_on_error: bool,
    /// Filename style for emitted artifacts (bd-tchp default: slug).
    pub naming_style: naming::NamingStyle,
    /// bd-27v1.1: destination for robot-mode NDJSON `youtube.*` events
    /// (`--robot` selects [`YoutubeRobotEvents::Stdout`]).
    pub robot_events: YoutubeRobotEvents,
}

/// Final outcome of a run, for the CLI to report / set an exit code.
#[derive(Debug, Clone, Default, Serialize)]
pub struct YoutubeRunSummary {
    pub done: Vec<String>,
    pub skipped: Vec<String>,
    pub failed: Vec<FailedVideo>,
    pub cancelled: bool,
}

/// A video that failed, with its last error message.
#[derive(Debug, Clone, Serialize)]
pub struct FailedVideo {
    pub id: String,
    pub title: String,
    pub error: String,
}

/// Parse a batch file: one URL per line, ignoring blank lines and comments
/// (`#`, `;`, or `]` leading char — matching yt-dlp's own batch semantics).
pub fn parse_batch_file(contents: &str) -> Vec<String> {
    contents
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with(['#', ';', ']']))
        .map(ToOwned::to_owned)
        .collect()
}

/// Resolve all inputs into a deduplicated, order-preserving list of videos.
fn resolve_videos(
    info: &YtdlpInfo,
    opts: &YoutubeRunOptions,
    token: &CancellationToken,
) -> FwResult<Vec<VideoRef>> {
    let mut raw_urls: Vec<String> = opts.urls.clone();
    if let Some(path) = &opts.batch_file {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            FwError::InvalidRequest(format!("read batch file {}: {e}", path.display()))
        })?;
        raw_urls.extend(parse_batch_file(&contents));
    }
    if raw_urls.is_empty() {
        return Err(FwError::InvalidRequest(
            "no inputs: pass URLs, --url, or --batch-file".to_owned(),
        ));
    }

    let mut videos: Vec<VideoRef> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for url in raw_urls {
        token.checkpoint()?;
        match ytdlp::classify_url(&url)? {
            UrlKind::Playlist => {
                for v in ytdlp::expand_playlist(info, &url, token)? {
                    if seen.insert(v.id.clone()) {
                        videos.push(v);
                    }
                }
            }
            UrlKind::Video | UrlKind::Ambiguous => {
                // Resolve the id by PARSING the URL (no network round-trip):
                // this is the #1 hotspot fix — dedup/naming get a stable id for
                // short/youtu.be/shorts/Ambiguous forms without a `yt-dlp -j`
                // call. The authoritative metadata fetch happens exactly once,
                // later, inside the download worker. Title/duration are filled
                // there (from the carried `VideoMeta`); we leave them empty/None
                // here.
                let video = match ytdlp::extract_video_id(&url) {
                    Some(id) => VideoRef {
                        id,
                        title: String::new(),
                        url: url.clone(),
                        duration_sec: None,
                    },
                    None => {
                        // classify_url accepted this as a Video/Ambiguous URL but
                        // the id is unrecoverable by parsing (should not happen —
                        // the two share the same URL grammar). Fall back to a
                        // single metadata fetch for THIS url only, preserving
                        // correctness at the cost of one round-trip.
                        let meta = ytdlp::fetch_metadata(info, &url, token)?;
                        VideoRef {
                            id: meta.id,
                            title: meta.title,
                            url: meta.webpage_url,
                            duration_sec: meta.duration_sec,
                        }
                    }
                };
                if seen.insert(video.id.clone()) {
                    videos.push(video);
                }
            }
        }
    }
    Ok(videos)
}

/// A unit of work handed from the download pool to the transcription consumer.
struct DownloadResult {
    video: VideoRef,
    /// `Ok((audio_path, meta))` on success, `Err(message)` on download failure.
    ///
    /// The worker fetches metadata exactly once (the single authoritative
    /// `yt-dlp -j` per video) and carries the resulting [`VideoMeta`] forward so
    /// the renderer never re-fetches it.
    outcome: Result<(PathBuf, VideoMeta), String>,
}

/// Download-stage work selected from manifest state plus on-disk evidence.
/// A reusable audio artifact still needs one metadata fetch for rendering, but
/// it must not invoke yt-dlp's download path again.
#[derive(Debug, Clone)]
enum DownloadWork {
    Fetch(VideoRef),
    Reuse {
        video: VideoRef,
        audio_path: PathBuf,
    },
}

impl DownloadWork {
    fn video(&self) -> &VideoRef {
        match self {
            Self::Fetch(video) | Self::Reuse { video, .. } => video,
        }
    }

    fn reuses_audio(&self) -> bool {
        matches!(self, Self::Reuse { .. })
    }
}

fn select_download_work(
    video: &VideoRef,
    state: Option<&VideoState>,
    audio_dir: &Path,
) -> DownloadWork {
    let recorded = match state {
        Some(VideoState::Downloaded { audio_path, .. }) => Some(PathBuf::from(audio_path)),
        _ => None,
    };
    let reusable = recorded
        .filter(|path| ytdlp::is_reusable_download_path(path, audio_dir, &video.id))
        .or_else(|| ytdlp::find_downloaded_by_id(audio_dir, &video.id));

    match reusable {
        Some(audio_path) => DownloadWork::Reuse {
            video: video.clone(),
            audio_path,
        },
        None => DownloadWork::Fetch(video.clone()),
    }
}

/// Run the full ingestion pipeline.
///
/// Probes `yt-dlp` once (the only environment-dependent step), then hands off
/// to [`run_with_info`], which is hermetic given a [`YtdlpInfo`] and therefore
/// unit-testable against the stub fixture without network access.
pub fn run(opts: &YoutubeRunOptions) -> FwResult<YoutubeRunSummary> {
    let info = ytdlp::probe()?;
    if info.stale {
        tracing::warn!(
            version = %info.version,
            "yt-dlp build is over 90 days old; YouTube may have changed — consider `yt-dlp -U`"
        );
    }
    run_with_info(opts, &info)
}

/// Run the pipeline against an already-probed `yt-dlp`.
///
/// Streams bd-27v1.1 robot events per the configured [`YoutubeRobotEvents`]
/// destination; see [`YoutubeEventEmitter`] for the envelope contract.
pub(crate) fn run_with_info(
    opts: &YoutubeRunOptions,
    info: &YtdlpInfo,
) -> FwResult<YoutubeRunSummary> {
    let token = CancellationToken::unbounded();
    let events = std::sync::Arc::new(YoutubeEventEmitter::new(opts.robot_events.clone()));
    // First event of every robot run: echoes the fully resolved request so an
    // agent can audit what the run was actually configured to do.
    events.emit(
        "run_start",
        serde_json::json!({
            "output_dir": opts.output_dir.display().to_string(),
            "n_urls": opts.urls.len(),
            "batch_file": opts.batch_file.as_ref().map(|p| p.display().to_string()),
            "concurrency": opts.concurrency.max(1),
            "batch_size": opts.batch_size,
            "backend": opts.backend.as_str(),
            "model": opts.model,
            "language": opts.language,
            "diarize": opts.diarize,
            "keep_audio": opts.keep_audio,
            "retry_failed": opts.retry_failed,
            "abort_on_error": opts.abort_on_error,
        }),
    )?;

    let outcome = run_with_info_body(opts, info, &token, &events);
    finalize_run(events.as_ref(), outcome)
}

fn run_with_info_body(
    opts: &YoutubeRunOptions,
    info: &YtdlpInfo,
    token: &CancellationToken,
    events: &std::sync::Arc<YoutubeEventEmitter>,
) -> FwResult<YoutubeRunSummary> {
    std::fs::create_dir_all(&opts.output_dir).map_err(FwError::Io)?;
    let audio_dir = opts.output_dir.join("audio");
    std::fs::create_dir_all(&audio_dir).map_err(FwError::Io)?;
    let manifest_path = opts.output_dir.join(MANIFEST_NAME);

    let mut manifest = Manifest::load(&manifest_path)?;
    let videos = resolve_videos(info, opts, token)?;
    for v in &videos {
        manifest.upsert_discovered(v);
    }
    manifest.save(&manifest_path)?;
    for v in &videos {
        events.emit(
            "discovered",
            serde_json::json!({ "id": v.id, "title": v.title, "url": v.url }),
        )?;
    }

    // Partition into work-to-do vs already-satisfied (idempotent resume).
    let mut summary = YoutubeRunSummary::default();
    let mut to_process: Vec<DownloadWork> = Vec::new();
    let mut abort_before_work = false;
    for v in &videos {
        let state = manifest
            .entries
            .get(&v.id)
            .and_then(|entry| entry.state.clone());
        let attempts = manifest.attempts(&v.id);
        match state.as_ref() {
            Some(VideoState::Done { .. }) => {
                tracing::info!(id = %v.id, "already done; skipping");
                summary.skipped.push(v.id.clone());
                events.emit(
                    "skipped",
                    serde_json::json!({ "id": v.id, "reason": "already_done" }),
                )?;
            }
            Some(VideoState::CleanupPending { .. }) => {
                tracing::info!(id = %v.id, "resuming pending post-render audio cleanup");
                let disposition = finish_pending_cleanup(
                    &mut manifest,
                    &manifest_path,
                    v,
                    events.as_ref(),
                    &mut summary,
                    std::fs::remove_file,
                )?;
                if disposition == RenderedVideoDisposition::CleanupFailed && opts.abort_on_error {
                    abort_before_work = true;
                    break;
                }
            }
            Some(_) if attempts > 0 && !opts.retry_failed => {
                tracing::info!(id = %v.id, "previously failed; --no-retry, skipping");
                summary.skipped.push(v.id.clone());
                events.emit(
                    "skipped",
                    serde_json::json!({ "id": v.id, "reason": "previously_failed_no_retry" }),
                )?;
            }
            Some(_) if attempts >= MAX_ATTEMPTS => {
                tracing::info!(
                    id = %v.id, attempts,
                    "exhausted retry budget; skipping (delete the manifest entry to force a retry)"
                );
                summary.skipped.push(v.id.clone());
                events.emit(
                    "skipped",
                    serde_json::json!({ "id": v.id, "reason": "retry_budget_exhausted" }),
                )?;
            }
            _ => to_process.push(select_download_work(v, state.as_ref(), &audio_dir)),
        }
    }

    if abort_before_work || to_process.is_empty() {
        return Ok(summary);
    }

    let engine = FrankenWhisperEngine::new()?;

    // ── bd-lun9 batch waves ─────────────────────────────────────────────
    // batch_size > 0 partitions work into waves so downloaded-but-untranscribed
    // audio cannot accumulate unbounded while the sequential transcription
    // consumer lags behind parallel download workers. 0 = single wave (history).
    let batches = partition_batches(to_process, opts.batch_size);
    'waves: for batch in batches {
        let mut abort_remaining = false;
        // ── Download stage: bounded worker pool feeding a bounded channel. ──
        // Capacity == concurrency keeps disk bounded (~concurrency in-flight +
        // queued). Each worker kills its yt-dlp child if the token fires.
        let concurrency = opts.concurrency.max(1);
        let (tx, rx) = sync_channel::<DownloadResult>(concurrency);
        let work = std::sync::Arc::new(std::sync::Mutex::new(batch.into_iter()));
        let audio_dir_arc = std::sync::Arc::new(audio_dir.clone());
        let info_arc = std::sync::Arc::new(info.clone());

        std::thread::scope(|scope| -> FwResult<()> {
            for _ in 0..concurrency {
                let tx = tx.clone();
                let work = std::sync::Arc::clone(&work);
                let audio_dir = std::sync::Arc::clone(&audio_dir_arc);
                let info = std::sync::Arc::clone(&info_arc);
                let events = std::sync::Arc::clone(events);
                scope.spawn(move || {
                    let dl_token = CancellationToken::unbounded();
                    loop {
                        if ShutdownController::is_shutting_down() {
                            break;
                        }
                        let next = {
                            let mut guard = work.lock().unwrap_or_else(|e| e.into_inner());
                            guard.next()
                        };
                        let Some(work_item) = next else { break };
                        let video = work_item.video().clone();
                        // Worker-side emission cannot propagate errors (this
                        // closure returns ()); a broken stdout pipe surfaces on
                        // the consumer side instead. Capture-mode serialization
                        // of plain values cannot fail in practice.
                        if !work_item.reuses_audio() {
                            let _ = events
                                .emit("downloading", serde_json::json!({ "id": video.id }));
                        }
                        let reused = work_item.reuses_audio();
                        let outcome = execute_download_work(
                            &info,
                            &work_item,
                            &audio_dir,
                            &dl_token,
                        );
                        if let Ok((audio_path, _meta)) = &outcome {
                            let bytes = std::fs::metadata(audio_path).ok().map(|m| m.len());
                            let _ = events.emit(
                                "downloaded",
                                serde_json::json!({
                                    "id": video.id,
                                    "audio_path": audio_path.display().to_string(),
                                    "bytes": bytes,
                                    "reused": reused,
                                }),
                            );
                        }
                        if tx.send(DownloadResult { video, outcome }).is_err() {
                            break; // consumer gone (cancel/abort)
                        }
                    }
                });
            }
            drop(tx); // close the channel once all workers finish

            // ── Transcription consumer: sequential, on this thread. ──
            for result in rx {
                let DownloadResult { video, outcome } = result;
                if token.checkpoint().is_err() {
                    // Cancelled while this download sat in the channel: persist its
                    // state so a resume reuses the audio rather than orphaning it.
                    if let Ok((audio_path, _meta)) = &outcome {
                        persist_cancelled_download(
                            &mut manifest,
                            &manifest_path,
                            &video.id,
                            audio_path,
                        )?;
                    }
                    summary.cancelled = true;
                    break;
                }
                let (audio_path, meta) = match outcome {
                    Ok(pair) => pair,
                    Err(error) => {
                        record_and_emit_failure(
                            &mut manifest,
                            &manifest_path,
                            &video,
                            &video.title,
                            &error,
                            events.as_ref(),
                            &mut summary,
                        )?;
                        if opts.abort_on_error {
                            abort_remaining = true;
                            break;
                        }
                        continue;
                    }
                };
                // ID-DIVERGENCE GUARD (manifest-key consistency, see the module
                // invariant): the manifest is keyed — on discovery AND on every
                // `set_state` — by `video.id`, which is derived **deterministically
                // from the input URL** (`extract_video_id`, or the rare fallback
                // fetch's id). Naming/output, by contrast, uses the authoritative
                // `meta.id` from yt-dlp. For valid YouTube URLs these are equal (the
                // `v=` param IS the video id and yt-dlp echoes it). They must stay
                // equal for resume to be correct: a re-run re-derives the SAME key
                // from the same URL and finds the `Done` entry, so a downloaded
                // video is never reprocessed. If yt-dlp ever canonicalizes the id to
                // something the URL parse can't reproduce, the output file (named
                // from `meta.id`) and the manifest key (the URL-derived id) would
                // disagree — resume still works (the key is URL-deterministic), but
                // the on-disk filename id would differ from the manifest key. Surface
                // that rare divergence rather than letting it pass silently.
                if meta.id != video.id {
                    tracing::warn!(
                        manifest_key = %video.id,
                        resolved_id = %meta.id,
                        url = %video.url,
                        "yt-dlp resolved a video id that differs from the URL-derived id; \
                         the output filename will use the resolved id while the manifest is \
                         keyed by the URL-derived id (resume stays correct)"
                    );
                }
                // We intentionally avoid a full-manifest `Downloaded` save on the
                // hot path. A crash after download is recovered by scanning the
                // controlled `<id>.*` audio directory on the next run; explicit
                // `Downloaded` state written by the cancellation path is also
                // honored. This keeps one terminal manifest rewrite per video
                // without sacrificing artifact reuse.
                events.emit("transcribing", serde_json::json!({ "id": video.id }))?;
                match transcribe_and_render(&engine, opts, &video, &meta, &audio_path) {
                    Ok((paths, wall_ms, rtf)) => {
                        let disposition = settle_rendered_video(
                            &mut manifest,
                            &manifest_path,
                            CompletedRender {
                                video: &video,
                                title: &meta.title,
                                audio_path: &audio_path,
                                paths: &paths,
                                wall_ms,
                                rtf,
                                keep_audio: opts.keep_audio,
                            },
                            events.as_ref(),
                            &mut summary,
                            std::fs::remove_file,
                        )?;
                        if disposition == RenderedVideoDisposition::CleanupFailed
                            && opts.abort_on_error
                        {
                            abort_remaining = true;
                            break;
                        }
                    }
                    Err(FwError::Cancelled(_)) => {
                        // Keep the current manifest state unchanged: it may be
                        // `Pending` or a prior `Failed` state whose attempt budget
                        // must survive cancellation. A resume finds the completed
                        // `<id>.*` audio artifact and reuses it before retrying
                        // transcription. Dropping this save also avoids a full
                        // O(N)-byte manifest rewrite on the cancel path.
                        summary.cancelled = true;
                        break;
                    }
                    Err(e) => {
                        let error = e.to_string();
                        record_and_emit_failure(
                            &mut manifest,
                            &manifest_path,
                            &video,
                            &meta.title,
                            &error,
                            events.as_ref(),
                            &mut summary,
                        )?;
                        if opts.abort_on_error {
                            abort_remaining = true;
                            break;
                        }
                    }
                }
            }
            Ok(())
        })?;
        // Cancellation and fail-fast abort both abandon later waves, but only
        // a real shutdown/cancellation marks the public summary cancelled.
        if summary.cancelled || abort_remaining {
            break 'waves;
        }
    }

    Ok(summary)
}

fn finalize_run(
    events: &YoutubeEventEmitter,
    outcome: FwResult<YoutubeRunSummary>,
) -> FwResult<YoutubeRunSummary> {
    match outcome {
        Ok(summary) => {
            emit_run_complete(events, &summary)?;
            Ok(summary)
        }
        Err(error) => {
            let summary = YoutubeRunSummary {
                cancelled: matches!(&error, FwError::Cancelled(_)),
                ..YoutubeRunSummary::default()
            };
            if let Err(emit_error) = emit_run_complete(events, &summary) {
                tracing::warn!(
                    error = %emit_error,
                    original_error = %error,
                    "failed to emit terminal youtube.run_complete after pipeline error"
                );
            }
            Err(error)
        }
    }
}

/// Terminal aggregate event (bd-27v1.1): exactly one per robot run, always
/// last; `cancelled` distinguishes actual shutdown/cancellation from a fully
/// completed or fail-fast run.
fn emit_run_complete(events: &YoutubeEventEmitter, summary: &YoutubeRunSummary) -> FwResult<()> {
    events.emit(
        "run_complete",
        serde_json::json!({
            "done": &summary.done,
            "skipped": &summary.skipped,
            "failed": &summary.failed,
            "cancelled": summary.cancelled,
        }),
    )
}

/// In-run retry budget for a single video's download (bd-lun9): attempts
/// beyond the first are spaced by full-jitter exponential backoff. The
/// cross-run budget lives in the manifest (`MAX_ATTEMPTS`); this one only
// bounds work inside a single run.
const DOWNLOAD_IN_RUN_RETRIES: u32 = 3;

/// Backoff base (`2^attempt * base`, capped) before full jitter.
const DOWNLOAD_BACKOFF_BASE_MS: u64 = 2_000;

/// Upper bound on any single backoff delay.
const DOWNLOAD_BACKOFF_CAP_MS: u64 = 30_000;

/// Transient-failure classification (bd-lun9): timeouts and I/O errors are
/// retryable by nature; a non-zero yt-dlp exit is retryable only when its
/// stderr smells like a transient upstream/network condition (rate limiting,
/// resolution blips, reset connections). Everything else — private videos,
/// geo blocks, bad URLs — is deterministic and must fail fast.
fn is_retryable_download_error(error: &FwError) -> bool {
    match error {
        FwError::CommandTimedOut { .. } | FwError::Io(_) => true,
        FwError::CommandFailed { stderr_suffix, .. } => {
            let s = stderr_suffix.to_ascii_lowercase();
            [
                "429",
                "503",
                "temporary failure",
                "connection reset",
                "connection timed out",
            ]
            .iter()
            .any(|needle| s.contains(needle))
        }
        // yt-dlp's signature mapper intentionally turns HTTP 429 into an
        // actionable InvalidRequest message. Preserve its transient nature
        // here instead of requiring the raw CommandFailed representation.
        FwError::InvalidRequest(message) => {
            let message = message.to_ascii_lowercase();
            message.contains("429") || message.contains("rate-limit")
        }
        _ => false,
    }
}

/// Pure full-jitter exponential delay computation (bd-lun9): base 2 s,
/// doubling per attempt, capped; SplitMix64 over nanos ^ attempt-hash for
/// the jitter — no rand dependency, the decoder's temperature-ladder idiom.
#[must_use]
fn backoff_delay_ms(attempt: u32) -> u64 {
    let exp_ms = DOWNLOAD_BACKOFF_BASE_MS
        .saturating_mul(1_u64 << attempt.min(4))
        .min(DOWNLOAD_BACKOFF_CAP_MS);
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0)
        ^ (u64::from(attempt) << 32);
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 27;
    (state >> 33) % (exp_ms + 1)
}

/// Cancellation-aware sleep in 100 ms slices, each followed by a token
/// checkpoint so Ctrl+C interrupts promptly mid-backoff. Returns `Err` if
/// cancelled while waiting.
fn sleep_cancellable(delay: std::time::Duration, token: &CancellationToken) -> FwResult<()> {
    let mut waited = std::time::Duration::from_millis(0);
    while waited < delay {
        token.checkpoint()?;
        let slice = std::time::Duration::from_millis(100);
        std::thread::sleep(slice);
        waited += slice;
    }
    Ok(())
}

/// Backoff then wait, cancellation-aware (bd-lun9).
fn jittered_backoff_sleep(attempt: u32, token: &CancellationToken) -> FwResult<()> {
    sleep_cancellable(
        std::time::Duration::from_millis(backoff_delay_ms(attempt)),
        token,
    )
}

/// Split the work list into bd-lun9 batch waves. `size == 0` yields a single
/// wave containing everything (historical behavior); otherwise consecutive
/// chunks of exactly `size` (the last may be short).
#[must_use]
fn partition_batches<T: Clone>(items: Vec<T>, size: usize) -> Vec<Vec<T>> {
    if size == 0 || items.len() <= size {
        return vec![items];
    }
    items.chunks(size).map(<[T]>::to_vec).collect()
}

fn execute_download_work(
    info: &YtdlpInfo,
    work: &DownloadWork,
    audio_dir: &Path,
    token: &CancellationToken,
) -> Result<(PathBuf, VideoMeta), String> {
    execute_download_work_with_metadata(info, work, audio_dir, token, ytdlp::fetch_metadata)
}

fn execute_download_work_with_metadata<F>(
    info: &YtdlpInfo,
    work: &DownloadWork,
    audio_dir: &Path,
    token: &CancellationToken,
    mut fetch_metadata: F,
) -> Result<(PathBuf, VideoMeta), String>
where
    F: FnMut(&YtdlpInfo, &str, &CancellationToken) -> FwResult<VideoMeta>,
{
    match work {
        DownloadWork::Fetch(video) => retry_download_stage(&video.id, token, || {
            download_one_attempt(info, video, audio_dir, token)
        }),
        DownloadWork::Reuse { video, audio_path } => {
            retry_download_stage(&video.id, token, || {
                fetch_metadata(info, &video.url, token)
                    .map(|meta| (audio_path.clone(), meta))
            })
        }
    }
}

/// Execute one download-stage operation with the same bounded transient retry
/// policy for both fresh downloads and metadata-only audio reuse.
fn retry_download_stage<T, F>(
    video_id: &str,
    token: &CancellationToken,
    mut operation: F,
) -> Result<T, String>
where
    F: FnMut() -> FwResult<T>,
{
    let mut attempt = 0_u32;
    loop {
        match operation() {
            Ok(ok) => return Ok(ok),
            Err(fw_err)
                if attempt < DOWNLOAD_IN_RUN_RETRIES && is_retryable_download_error(&fw_err) =>
            {
                tracing::warn!(
                    id = %video_id,
                    attempt = attempt + 1,
                    retries_left = DOWNLOAD_IN_RUN_RETRIES - attempt,
                    error = %fw_err,
                    "transient download-stage failure; backing off before retry"
                );
                jittered_backoff_sleep(attempt, token).map_err(|e| e.to_string())?;
                attempt += 1;
            }
            Err(fw_err) => return Err(fw_err.to_string()),
        }
    }
}

/// One physical metadata+download pass (the caller owns retries).
fn download_one_attempt(
    info: &YtdlpInfo,
    video: &VideoRef,
    audio_dir: &Path,
    token: &CancellationToken,
) -> FwResult<(PathBuf, VideoMeta)> {
    let t_meta = std::time::Instant::now();
    let meta = ytdlp::fetch_metadata(info, &video.url, token)?;
    crate::native_engine::perf_span("yt.dl_metadata", t_meta.elapsed().as_secs_f64() * 1e3, "");
    let t_dl = std::time::Instant::now();
    let path = ytdlp::download_audio(info, &meta, audio_dir, token)?;
    crate::native_engine::perf_span("yt.download", t_dl.elapsed().as_secs_f64() * 1e3, "");
    Ok((path, meta))
}

fn record_failure(
    manifest: &mut Manifest,
    manifest_path: &Path,
    video: &VideoRef,
    error: &str,
) -> FwResult<()> {
    let attempts = manifest.attempts(&video.id) + 1;
    manifest.set_state(
        &video.id,
        VideoState::Failed {
            error: error.to_owned(),
            attempts,
        },
    );
    manifest.save(manifest_path)?;
    tracing::warn!(id = %video.id, attempts, error, "video failed");
    Ok(())
}

fn persist_cancelled_download(
    manifest: &mut Manifest,
    manifest_path: &Path,
    video_id: &str,
    audio_path: &Path,
) -> FwResult<()> {
    let attempts = manifest.attempts(video_id);
    manifest.set_state(
        video_id,
        VideoState::Downloaded {
            audio_path: audio_path.display().to_string(),
            attempts,
        },
    );
    manifest.save(manifest_path)
}

fn record_and_emit_failure(
    manifest: &mut Manifest,
    manifest_path: &Path,
    video: &VideoRef,
    title: &str,
    error: &str,
    events: &YoutubeEventEmitter,
    summary: &mut YoutubeRunSummary,
) -> FwResult<()> {
    record_failure(manifest, manifest_path, video, error)?;
    let failure = FailedVideo {
        id: video.id.clone(),
        title: title.to_owned(),
        error: error.to_owned(),
    };
    events.emit(
        "failed",
        serde_json::json!({
            "id": &failure.id,
            "title": &failure.title,
            "error": &failure.error,
            "attempts": manifest.attempts(&failure.id),
        }),
    )?;
    summary.failed.push(failure);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderedVideoDisposition {
    Done,
    CleanupFailed,
}

struct CompletedRender<'a> {
    video: &'a VideoRef,
    title: &'a str,
    audio_path: &'a Path,
    paths: &'a OutputPaths,
    wall_ms: u64,
    rtf: Option<f64>,
    keep_audio: bool,
}

/// Commit the terminal state for a successfully rendered video.
///
/// Audio removal is part of the `--no-keep-audio` contract, so an I/O failure
/// other than `NotFound` is a per-video failure rather than a successful
/// `Done` transition. Before removal, the manifest durably records
/// [`VideoState::CleanupPending`], so a crash after deletion but before the
/// final save can settle from `NotFound` without redownloading. The removal
/// callback keeps this production transition deterministically testable on
/// platforms where permission failures cannot be induced reliably.
fn settle_rendered_video<R>(
    manifest: &mut Manifest,
    manifest_path: &Path,
    rendered: CompletedRender<'_>,
    events: &YoutubeEventEmitter,
    summary: &mut YoutubeRunSummary,
    remove_file: R,
) -> FwResult<RenderedVideoDisposition>
where
    R: FnOnce(&Path) -> std::io::Result<()>,
{
    let mut save_manifest = Manifest::save;
    if rendered.keep_audio {
        return persist_rendered_done_with(
            manifest,
            manifest_path,
            rendered.video,
            Some(rendered.audio_path.display().to_string()),
            &rendered.paths.md.display().to_string(),
            &rendered.paths.json.display().to_string(),
            rendered.wall_ms,
            rendered.rtf,
            events,
            summary,
            &mut save_manifest,
        );
    }

    persist_cleanup_intent_with(manifest, manifest_path, &rendered, &mut save_manifest)?;
    finish_pending_cleanup_with(
        manifest,
        manifest_path,
        rendered.video,
        events,
        summary,
        &mut save_manifest,
        remove_file,
    )
}

fn persist_cleanup_intent_with<S>(
    manifest: &mut Manifest,
    manifest_path: &Path,
    rendered: &CompletedRender<'_>,
    save_manifest: &mut S,
) -> FwResult<()>
where
    S: FnMut(&Manifest, &Path) -> FwResult<()>,
{
    let attempts = manifest.attempts(&rendered.video.id);
    manifest.set_state(
        &rendered.video.id,
        VideoState::CleanupPending {
            title: rendered.title.to_owned(),
            audio_path: rendered.audio_path.display().to_string(),
            markdown_path: rendered.paths.md.display().to_string(),
            json_path: rendered.paths.json.display().to_string(),
            wall_ms: rendered.wall_ms,
            rtf: rendered.rtf,
            attempts,
            last_error: None,
        },
    );
    save_manifest(manifest, manifest_path)
}

fn finish_pending_cleanup<R>(
    manifest: &mut Manifest,
    manifest_path: &Path,
    video: &VideoRef,
    events: &YoutubeEventEmitter,
    summary: &mut YoutubeRunSummary,
    remove_file: R,
) -> FwResult<RenderedVideoDisposition>
where
    R: FnOnce(&Path) -> std::io::Result<()>,
{
    let mut save_manifest = Manifest::save;
    finish_pending_cleanup_with(
        manifest,
        manifest_path,
        video,
        events,
        summary,
        &mut save_manifest,
        remove_file,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_pending_cleanup_with<R, S>(
    manifest: &mut Manifest,
    manifest_path: &Path,
    video: &VideoRef,
    events: &YoutubeEventEmitter,
    summary: &mut YoutubeRunSummary,
    save_manifest: &mut S,
    remove_file: R,
) -> FwResult<RenderedVideoDisposition>
where
    R: FnOnce(&Path) -> std::io::Result<()>,
    S: FnMut(&Manifest, &Path) -> FwResult<()>,
{
    let (title, audio_path, markdown_path, json_path, wall_ms, rtf) = match manifest
        .entries
        .get(&video.id)
        .and_then(|entry| entry.state.as_ref())
    {
        Some(VideoState::CleanupPending {
            title,
            audio_path,
            markdown_path,
            json_path,
            wall_ms,
            rtf,
            ..
        }) => (
            title.clone(),
            PathBuf::from(audio_path),
            markdown_path.clone(),
            json_path.clone(),
            *wall_ms,
            *rtf,
        ),
        _ => {
            return Err(FwError::ContractViolation(format!(
                "video {} has no pending cleanup state to settle",
                video.id
            )));
        }
    };

    let output_dir = manifest_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let audio_dir = output_dir.join("audio");
    if !ytdlp::is_owned_download_path(&audio_path, &audio_dir, &video.id) {
        return Err(FwError::ContractViolation(format!(
            "pending cleanup path {} is not owned by {} for video {}",
            audio_path.display(),
            audio_dir.display(),
            video.id
        )));
    }

    match remove_file(&audio_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            let error = format!(
                "delete downloaded audio {} after rendering: {error}",
                audio_path.display()
            );
            let attempts = manifest.attempts(&video.id).saturating_add(1);
            let Some(VideoState::CleanupPending {
                attempts: stored_attempts,
                last_error,
                ..
            }) = manifest
                .entries
                .get_mut(&video.id)
                .and_then(|entry| entry.state.as_mut())
            else {
                return Err(FwError::ContractViolation(format!(
                    "video {} lost its pending cleanup state",
                    video.id
                )));
            };
            *stored_attempts = attempts;
            *last_error = Some(error.clone());
            save_manifest(manifest, manifest_path)?;

            let failure = FailedVideo {
                id: video.id.clone(),
                title,
                error,
            };
            events.emit(
                "failed",
                serde_json::json!({
                    "id": &failure.id,
                    "title": &failure.title,
                    "error": &failure.error,
                    "attempts": attempts,
                }),
            )?;
            summary.failed.push(failure);
            return Ok(RenderedVideoDisposition::CleanupFailed);
        }
    }

    persist_rendered_done_with(
        manifest,
        manifest_path,
        video,
        None,
        &markdown_path,
        &json_path,
        wall_ms,
        rtf,
        events,
        summary,
        save_manifest,
    )
}

#[allow(clippy::too_many_arguments)]
fn persist_rendered_done_with<S>(
    manifest: &mut Manifest,
    manifest_path: &Path,
    video: &VideoRef,
    audio_path: Option<String>,
    markdown_path: &str,
    json_path: &str,
    wall_ms: u64,
    rtf: Option<f64>,
    events: &YoutubeEventEmitter,
    summary: &mut YoutubeRunSummary,
    save_manifest: &mut S,
) -> FwResult<RenderedVideoDisposition>
where
    S: FnMut(&Manifest, &Path) -> FwResult<()>,
{
    manifest.set_state(
        &video.id,
        VideoState::Done {
            audio_path,
            markdown_path: markdown_path.to_owned(),
            json_path: json_path.to_owned(),
        },
    );
    save_manifest(manifest, manifest_path)?;
    summary.done.push(video.id.clone());
    events.emit(
        "done",
        serde_json::json!({
            "id": video.id,
            "md_path": markdown_path,
            "json_path": json_path,
            "wall_ms": wall_ms,
            "rtf": rtf,
        }),
    )?;
    Ok(RenderedVideoDisposition::Done)
}

/// Transcribe a downloaded audio file and render markdown + JSON.
///
/// `meta` is the [`VideoMeta`] the download worker already fetched (the single
/// authoritative `yt-dlp -j` per video); the renderer never re-fetches it. The
/// `_video` reference is retained for symmetry/logging but its naming fields are
/// superseded by the richer `meta`.
fn transcribe_and_render(
    engine: &FrankenWhisperEngine,
    opts: &YoutubeRunOptions,
    _video: &VideoRef,
    meta: &VideoMeta,
    audio_path: &Path,
) -> FwResult<(OutputPaths, u64, Option<f64>)> {
    let started = chrono::Utc::now();
    let started_instant = Instant::now();

    let request = TranscribeRequest {
        input: InputSource::File {
            path: audio_path.to_path_buf(),
        },
        backend: opts.backend,
        model: opts.model.clone(),
        language: opts.language.clone(),
        translate: false,
        diarize: opts.diarize,
        persist: false,
        db_path: opts
            .output_dir
            .join(".franken_whisper")
            .join("storage.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    };

    let report = engine.transcribe(request)?;
    let wall_ms = started_instant.elapsed().as_millis() as u64;
    crate::native_engine::perf_span("yt.transcribe", wall_ms as f64, "");
    let segments: &[TranscriptionSegment] = &report.result.segments;
    let windows = project_native_windows(&report.result.raw_output)?;

    let rtf = meta
        .duration_sec
        .filter(|d| *d > 0.0)
        .map(|d| (wall_ms as f64 / 1000.0) / d);

    let (engine_label, backend_label) = engine_labels(&report);

    let base = naming::sanitize_base_with(
        opts.naming_style,
        &meta.title,
        meta.upload_date.as_deref(),
        &meta.id,
    );
    let paths = naming::output_paths(&opts.output_dir, &base);

    let input = RenderInput {
        video: RenderVideo {
            id: meta.id.clone(),
            title: meta.title.clone(),
            channel: meta.channel.clone(),
            uploader: meta.uploader.clone(),
            upload_date: meta.upload_date.clone(),
            duration_sec: meta.duration_sec,
            webpage_url: meta.webpage_url.clone(),
            description: meta.description.clone(),
        },
        run: RenderRun {
            model: opts
                .model
                .clone()
                .unwrap_or_else(|| report.result.backend.as_str().to_owned()),
            engine: engine_label,
            backend: backend_label,
            version_tag: Some(env!("CARGO_PKG_VERSION").to_owned()),
            started_rfc3339: started.to_rfc3339(),
            wall_ms,
            rtf,
        },
        segments,
        windows: &windows,
    };

    let t_render = std::time::Instant::now();
    render::write_atomic(&paths.md, &render::render_markdown(&input))?;
    let json = render::render_json(&input);
    render::write_atomic(
        &paths.json,
        &serde_json::to_string_pretty(&json).map_err(FwError::Json)?,
    )?;
    crate::native_engine::perf_span("yt.render", t_render.elapsed().as_secs_f64() * 1e3, "");
    Ok((paths, wall_ms, rtf))
}

/// Project the stable `native-v2` decode-window contract into the YouTube JSON
/// renderer. Subprocess backends do not expose this native evidence and yield
/// an empty list. Once an in-process producer declares `native-v2`, every
/// required field is checked so malformed producer output cannot be published
/// as an apparently complete sidecar.
fn project_native_windows(raw: &serde_json::Value) -> FwResult<Vec<RenderWindowStats>> {
    let schema_version = raw
        .get("schema_version")
        .and_then(serde_json::Value::as_str);
    let in_process = raw
        .get("in_process")
        .and_then(serde_json::Value::as_bool);
    match (schema_version, in_process) {
        (Some("native-v2"), Some(true)) => {}
        (Some("native-v2"), _) | (_, Some(true)) => {
            return Err(FwError::ContractViolation(
                "partial native raw-output declaration: native-v2 requires in_process=true and vice versa"
                    .to_owned(),
            ));
        }
        _ => return Ok(Vec::new()),
    }

    let windows = raw
        .get("windows")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            FwError::ContractViolation(
                "native-v2 raw output is missing the required windows array".to_owned(),
            )
        })?;

    windows
        .iter()
        .enumerate()
        .map(|(index, window)| {
            let field = |name: &str| {
                window.get(name).ok_or_else(|| {
                    FwError::ContractViolation(format!(
                        "native-v2 window {index} is missing required field {name}"
                    ))
                })
            };
            let finite = |name: &str| {
                let value = field(name)?.as_f64().ok_or_else(|| {
                    FwError::ContractViolation(format!(
                        "native-v2 window {index} field {name} is not numeric"
                    ))
                })?;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(FwError::ContractViolation(format!(
                        "native-v2 window {index} field {name} is not finite"
                    )))
                }
            };

            let window_offset_sec = finite("window_offset_sec")?;
            if window_offset_sec < 0.0 {
                return Err(FwError::ContractViolation(format!(
                    "native-v2 window {index} field window_offset_sec is negative"
                )));
            }
            let tokens = field("tokens")?.as_u64().ok_or_else(|| {
                FwError::ContractViolation(format!(
                    "native-v2 window {index} field tokens is not an unsigned integer"
                ))
            })?;
            let avg_logprob = finite("avg_logprob")?;
            let no_speech_prob = finite("no_speech_prob")?;
            if !(0.0..=1.0).contains(&no_speech_prob) {
                return Err(FwError::ContractViolation(format!(
                    "native-v2 window {index} field no_speech_prob is outside [0, 1]"
                )));
            }

            Ok(RenderWindowStats {
                window_offset_sec,
                tokens,
                avg_logprob,
                no_speech_prob,
            })
        })
        .collect()
}

/// Pull engine/backend labels out of the run report's raw output (best effort).
fn engine_labels(report: &crate::model::RunReport) -> (String, String) {
    let raw = &report.result.raw_output;
    let engine = raw.get("engine").and_then(|v| v.as_str()).map_or_else(
        || report.result.backend.as_str().to_owned(),
        ToOwned::to_owned,
    );
    let backend = raw
        .get("implementation")
        .and_then(|v| v.as_str())
        .unwrap_or("bridge")
        .to_owned();
    (engine, backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute path to the hermetic yt-dlp stub (emits 2 canned playlist
    /// entries: vid000000001 / vid000000002).
    fn stub_info() -> YtdlpInfo {
        YtdlpInfo {
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/youtube/ytdlp_stub.sh"),
            version: "2025.01.01".to_owned(),
            stale: false,
        }
    }

    fn opts_with_urls(urls: Vec<String>) -> YoutubeRunOptions {
        YoutubeRunOptions {
            urls,
            batch_file: None,
            output_dir: PathBuf::from("/tmp/unused"),
            model: None,
            language: None,
            backend: BackendKind::Auto,
            diarize: false,
            concurrency: 1,
            keep_audio: false,
            retry_failed: false,
            abort_on_error: false,
            batch_size: 0,
            naming_style: naming::NamingStyle::Slug,
            robot_events: YoutubeRobotEvents::Off,
        }
    }

    fn vr(id: &str) -> VideoRef {
        VideoRef {
            id: id.to_owned(),
            title: format!("Title {id}"),
            url: format!("https://www.youtube.com/watch?v={id}"),
            duration_sec: None,
        }
    }

    #[test]
    fn native_window_projection_preserves_exact_decoder_evidence() {
        let raw = serde_json::json!({
            "schema_version": "native-v2",
            "in_process": true,
            "windows": [{
                "window_offset_sec": 30.0,
                "tokens": 17,
                "avg_logprob": -0.42,
                "no_speech_prob": 0.125,
                "additive_future_field": "ignored"
            }]
        });

        let projected = project_native_windows(&raw).expect("valid native-v2 windows");
        assert_eq!(
            projected,
            vec![RenderWindowStats {
                window_offset_sec: 30.0,
                tokens: 17,
                avg_logprob: -0.42,
                no_speech_prob: 0.125,
            }]
        );
    }

    #[test]
    fn native_window_projection_is_empty_for_external_backends() {
        let raw = serde_json::json!({
            "engine": "whisper-cli",
            "segments": [],
        });
        assert!(
            project_native_windows(&raw)
                .expect("external output is not a native-v2 contract")
                .is_empty()
        );
    }

    #[test]
    fn native_window_projection_rejects_partial_native_declarations() {
        for raw in [
            serde_json::json!({
                "schema_version": "native-v2",
                "windows": [],
            }),
            serde_json::json!({
                "in_process": true,
                "windows": [],
            }),
            serde_json::json!({
                "schema_version": "native-v2",
                "in_process": false,
                "windows": [],
            }),
        ] {
            let error = project_native_windows(&raw)
                .expect_err("partial native declaration must fail closed");
            assert!(matches!(error, FwError::ContractViolation(_)));
        }
    }

    #[test]
    fn native_window_projection_rejects_missing_required_evidence() {
        let missing = serde_json::json!({
            "schema_version": "native-v2",
            "in_process": true,
            "windows": [{
                "window_offset_sec": 0.0,
                "tokens": 3,
                "avg_logprob": -0.5
            }]
        });
        let error = project_native_windows(&missing).expect_err("missing probability must fail");
        assert!(matches!(error, FwError::ContractViolation(_)));

        let out_of_range = serde_json::json!({
            "schema_version": "native-v2",
            "in_process": true,
            "windows": [{
                "window_offset_sec": 0.0,
                "tokens": 3,
                "avg_logprob": -0.5,
                "no_speech_prob": 1.5
            }]
        });
        let error = project_native_windows(&out_of_range)
            .expect_err("out-of-range probability must fail");
        assert!(matches!(error, FwError::ContractViolation(_)));
    }

    // ---- bd-lun9: batch waves + transient retry classification ------------

    #[test]
    fn partition_batches_zero_size_is_single_wave() {
        let items = vec![vr("a"), vr("b"), vr("c")];
        let waves = partition_batches(items, 0);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].len(), 3);
    }

    #[test]
    fn partition_batches_chunks_and_keeps_order() {
        let items = vec![vr("a"), vr("b"), vr("c"), vr("d"), vr("e")];
        let waves = partition_batches(items, 2);
        let ids: Vec<Vec<&str>> = waves
            .iter()
            .map(|w| w.iter().map(|v| v.id.as_str()).collect())
            .collect();
        assert_eq!(ids, [vec!["a", "b"], vec!["c", "d"], vec!["e"]]);
    }

    #[test]
    fn resume_selects_existing_recorded_or_orphaned_audio_without_fetch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");

        let recorded_video = vr("recorded001");
        let recorded_path = audio_dir.join("recorded001.opus");
        std::fs::write(&recorded_path, b"recorded sentinel").expect("recorded audio");
        let recorded_state = VideoState::Downloaded {
            audio_path: recorded_path.display().to_string(),
            attempts: 0,
        };
        let selected = select_download_work(
            &recorded_video,
            Some(&recorded_state),
            &audio_dir,
        );
        assert!(matches!(
            selected,
            DownloadWork::Reuse { ref audio_path, .. } if audio_path == &recorded_path
        ));

        let orphaned_video = vr("orphaned001");
        let orphaned_path = audio_dir.join("orphaned001.wav");
        std::fs::write(&orphaned_path, b"orphaned sentinel").expect("orphaned audio");
        let selected = select_download_work(
            &orphaned_video,
            Some(&VideoState::Pending),
            &audio_dir,
        );
        assert!(matches!(
            selected,
            DownloadWork::Reuse { ref audio_path, .. } if audio_path == &orphaned_path
        ));

        let missing_video = vr("missing00001");
        assert!(matches!(
            select_download_work(&missing_video, Some(&VideoState::Pending), &audio_dir),
            DownloadWork::Fetch(_)
        ));
    }

    #[test]
    fn resume_rejects_recorded_paths_outside_the_owned_audio_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let outside_path = dir.path().join("outside001.wav");
        std::fs::write(&outside_path, b"outside sentinel").expect("outside file");
        let video = vr("outside001");
        let state = VideoState::Downloaded {
            audio_path: outside_path.display().to_string(),
            attempts: 0,
        };

        assert!(matches!(
            select_download_work(&video, Some(&state), &audio_dir),
            DownloadWork::Fetch(_)
        ));
    }

    #[test]
    fn reuse_work_fetches_metadata_without_overwriting_audio() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let video = vr("reuse000001");
        let audio_path = audio_dir.join("reuse000001.wav");
        let sentinel: &[u8] = b"existing audio must remain byte exact";
        std::fs::write(&audio_path, sentinel).expect("seed audio");
        let work = DownloadWork::Reuse {
            video,
            audio_path: audio_path.clone(),
        };

        let (returned_path, meta) = execute_download_work(
            &stub_info(),
            &work,
            &audio_dir,
            &CancellationToken::unbounded(),
        )
        .expect("metadata-only reuse");
        assert_eq!(returned_path, audio_path);
        assert_eq!(meta.id, "reuse000001");
        assert_eq!(
            std::fs::read(&returned_path)
                .expect("read reused audio")
                .as_slice(),
            sentinel,
            "the download path must not run for reusable audio"
        );
    }

    #[test]
    fn reuse_work_retries_transient_metadata_without_overwriting_audio() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let video = vr("retryreuse1");
        let audio_path = audio_dir.join("retryreuse1.wav");
        let sentinel: &[u8] = b"retry metadata without downloading again";
        std::fs::write(&audio_path, sentinel).expect("seed audio");
        let work = DownloadWork::Reuse {
            video,
            audio_path: audio_path.clone(),
        };
        let mut metadata_calls = 0_u32;

        let (returned_path, meta) = execute_download_work_with_metadata(
            &stub_info(),
            &work,
            &audio_dir,
            &CancellationToken::unbounded(),
            |info, url, token| {
                metadata_calls += 1;
                if metadata_calls == 1 {
                    Err(FwError::InvalidRequest(
                        "YouTube rate-limited the downloader (HTTP 429). Wait and retry."
                            .to_owned(),
                    ))
                } else {
                    ytdlp::fetch_metadata(info, url, token)
                }
            },
        )
        .expect("transient metadata fetch should recover");

        assert_eq!(metadata_calls, 2, "one transient attempt plus one success");
        assert_eq!(returned_path, audio_path);
        assert_eq!(meta.id, "retryreuse1");
        assert_eq!(
            std::fs::read(&returned_path)
                .expect("read reused audio")
                .as_slice(),
            sentinel,
            "metadata retry must never invoke the audio download path"
        );
    }

    #[test]
    fn partition_batches_size_larger_than_work_is_one_wave() {
        let items = vec![vr("a"), vr("b")];
        let waves = partition_batches(items, 100);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].len(), 2);
    }

    #[test]
    fn transient_timeouts_are_retryable_but_private_videos_are_not() {
        assert!(is_retryable_download_error(&FwError::CommandTimedOut {
            command: "yt-dlp".to_owned(),
            timeout_ms: 300_000,
            stderr_suffix: String::new(),
        }));
        assert!(is_retryable_download_error(&FwError::Io(
            std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset")
        )));
        // Rate limiting is the canonical transient CommandFailed.
        assert!(is_retryable_download_error(&FwError::from_command_failure(
            "yt-dlp".to_owned(),
            1,
            "ERROR: HTTP Error 429: Too Many Requests".to_owned(),
        )));
        assert!(is_retryable_download_error(&FwError::InvalidRequest(
            "YouTube rate-limited the downloader (HTTP 429). Wait and retry.".to_owned(),
        )));
        // Deterministic content errors must fail fast.
        assert!(!is_retryable_download_error(
            &FwError::from_command_failure(
                "yt-dlp".to_owned(),
                1,
                "ERROR: Private video. Sign in".to_owned(),
            )
        ));
        assert!(!is_retryable_download_error(&FwError::InvalidRequest(
            "bad url".to_owned()
        )));
    }

    #[test]
    fn backoff_delay_never_exceeds_cap_even_at_huge_attempt() {
        for attempt in [0_u32, 1, 5, 30, 1000] {
            let d = backoff_delay_ms(attempt);
            assert!(d <= DOWNLOAD_BACKOFF_CAP_MS, "attempt {attempt} -> {d}ms");
        }
    }

    #[test]
    fn cancellable_sleep_surfaces_expired_deadline_as_cancelled() {
        use crate::orchestrator::CancellationToken as Tok;
        let token = Tok::with_deadline_from_now(std::time::Duration::from_millis(0));
        std::thread::sleep(std::time::Duration::from_millis(5));
        let err = sleep_cancellable(std::time::Duration::from_secs(60), &token).unwrap_err();
        assert!(matches!(err, FwError::Cancelled(_)));
    }

    // ---- resolve_videos: bug-hunt edge cases (no network for Video forms) --

    #[test]
    fn resolve_video_urls_are_local_and_dedup_by_id() {
        // Two URL forms for the SAME id (watch?v= and youtu.be/) must dedup to
        // one VideoRef — and never touch the network (Video/Ambiguous resolve
        // purely via extract_video_id).
        let token = CancellationToken::unbounded();
        let opts = opts_with_urls(vec![
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ".to_owned(),
            "https://youtu.be/dQw4w9WgXcQ".to_owned(), // same id, different form
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLx".to_owned(), // ambiguous, same id
            "https://youtu.be/SECOND00001".to_owned(),
        ]);
        let videos = resolve_videos(&stub_info(), &opts, &token).expect("resolve");
        let ids: Vec<&str> = videos.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(ids, vec!["dQw4w9WgXcQ", "SECOND00001"], "deduped by id");
    }

    #[test]
    fn resolve_empty_inputs_errors() {
        let token = CancellationToken::unbounded();
        let opts = opts_with_urls(vec![]);
        assert!(matches!(
            resolve_videos(&stub_info(), &opts, &token),
            Err(FwError::InvalidRequest(_))
        ));
    }

    #[test]
    fn resolve_playlist_expands_via_stub() {
        let token = CancellationToken::unbounded();
        let opts = opts_with_urls(vec![
            "https://www.youtube.com/playlist?list=PL123".to_owned(),
        ]);
        let videos = resolve_videos(&stub_info(), &opts, &token).expect("resolve");
        assert_eq!(videos.len(), 2);
        assert_eq!(videos[0].id, "vid000000001");
        assert_eq!(videos[1].id, "vid000000002");
    }

    #[test]
    fn resolve_mixed_playlist_and_videos_dedup_cross_source() {
        // A playlist (stub -> vid000000001, vid000000002) PLUS an explicit video
        // URL whose id collides with a playlist entry, PLUS a fresh video.
        // Order-preserving, first-seen-wins dedup across BOTH sources.
        let token = CancellationToken::unbounded();
        let opts = opts_with_urls(vec![
            "https://www.youtube.com/playlist?list=PL123".to_owned(),
            "https://youtu.be/vid000000002".to_owned(), // dup of a playlist entry
            "https://www.youtube.com/watch?v=FRESHvideo1".to_owned(),
        ]);
        let videos = resolve_videos(&stub_info(), &opts, &token).expect("resolve");
        let ids: Vec<&str> = videos.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(ids, vec!["vid000000001", "vid000000002", "FRESHvideo1"]);
    }

    #[test]
    fn resolve_duplicate_playlist_urls_dedup() {
        // The same playlist URL twice must not duplicate its entries.
        let token = CancellationToken::unbounded();
        let opts = opts_with_urls(vec![
            "https://www.youtube.com/playlist?list=PL123".to_owned(),
            "https://www.youtube.com/playlist?list=PL123".to_owned(),
        ]);
        let videos = resolve_videos(&stub_info(), &opts, &token).expect("resolve");
        assert_eq!(videos.len(), 2, "duplicate playlist expansion deduped");
    }

    /// SCALE MEASURE: resolve (dedup) + upsert + partition for K synthetic
    /// video URLs. Proves the path is linear (HashSet dedup + BTreeMap upsert,
    /// no accidental O(N²) Vec::contains scan) and reports timing. Uses local
    /// Video URLs so NO network/subprocess is involved — this isolates the
    /// pure-CPU resolve/dedup/manifest cost.
    #[test]
    fn resolve_and_upsert_scale_2000_is_linear() {
        const K: usize = 2000;
        let token = CancellationToken::unbounded();
        // K distinct video URLs + a full duplicate pass (4000 inputs, 2000 ids).
        let mut urls: Vec<String> = (0..K)
            .map(|i| format!("https://www.youtube.com/watch?v=vid{i:08}id"))
            .collect();
        urls.extend(urls.clone()); // duplicates to exercise dedup
        let opts = opts_with_urls(urls);

        let t_resolve = std::time::Instant::now();
        let videos = resolve_videos(&stub_info(), &opts, &token).expect("resolve");
        let resolve_elapsed = t_resolve.elapsed();
        assert_eq!(videos.len(), K, "dedup collapses the duplicate pass");

        // Upsert into the manifest (mirrors run()'s discovery loop).
        let mut manifest = Manifest::default();
        let t_upsert = std::time::Instant::now();
        for v in &videos {
            manifest.upsert_discovered(v);
        }
        let upsert_elapsed = t_upsert.elapsed();
        assert_eq!(manifest.order.len(), K);
        assert_eq!(manifest.entries.len(), K);

        eprintln!(
            "resolve+dedup scale: K={K} ids from {} inputs in {:?} ({:.2} us/id); \
             upsert {K} entries in {:?} ({:.2} us/entry)",
            2 * K,
            resolve_elapsed,
            resolve_elapsed.as_secs_f64() * 1e6 / (2 * K) as f64,
            upsert_elapsed,
            upsert_elapsed.as_secs_f64() * 1e6 / K as f64,
        );
    }

    /// REGRESSION GUARD for the manifest-key-consistency invariant (mission
    /// #1c). The manifest MUST be keyed by the URL-derived id (`extract_video_id`
    /// of the input URL) — NOT by yt-dlp's resolved `meta.id` — so that a re-run
    /// re-derives the SAME key from the same URL and finds the prior `Done`
    /// entry. If a future refactor switched the manifest key to `meta.id`, a
    /// re-run would re-derive the URL key, fail to match the `meta.id`-keyed
    /// entry, and reprocess the video forever. This test pins:
    /// (1) `resolve_videos` produces a `VideoRef.id` equal to `extract_video_id`,
    /// (2) `upsert_discovered` keys the manifest by exactly that id, and
    /// (3) the same URL re-resolves to the same key (idempotent discovery).
    #[test]
    fn manifest_key_is_url_derived_and_stable_across_reruns() {
        let token = CancellationToken::unbounded();
        // A spread of single-video URL forms; each must key by its URL-derived
        // id and re-derive identically on a second pass (the resume contract).
        let urls = vec![
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ".to_owned(),
            "https://youtu.be/SHORTID0001?t=42".to_owned(),
            "https://www.youtube.com/watch?v=WITHLIST001&list=PLabc".to_owned(),
            "https://www.youtube.com/shorts/SHORTSVID01".to_owned(),
        ];
        let opts = opts_with_urls(urls.clone());

        // First "run": resolve + upsert.
        let videos = resolve_videos(&stub_info(), &opts, &token).expect("resolve");
        let mut manifest = Manifest::default();
        for v in &videos {
            // Each VideoRef.id is exactly the URL-derived id (no network).
            assert_eq!(
                Some(v.id.clone()),
                ytdlp::extract_video_id(&v.url),
                "VideoRef id must equal extract_video_id(url) for {}",
                v.url
            );
            manifest.upsert_discovered(v);
        }
        // The manifest is keyed by the URL-derived ids, in discovery order.
        let keys_after_first: Vec<String> = manifest.order.clone();
        assert_eq!(
            keys_after_first,
            vec![
                "dQw4w9WgXcQ".to_owned(),
                "SHORTID0001".to_owned(),
                "WITHLIST001".to_owned(),
                "SHORTSVID01".to_owned(),
            ],
            "manifest keys must be the URL-derived ids"
        );

        // Mark all done (mirrors a completed run) and "re-run" discovery from
        // the SAME urls: upsert must re-derive the SAME keys and add nothing,
        // so the partition step would skip every already-done video.
        for k in &keys_after_first {
            manifest.set_state(
                k,
                VideoState::Done {
                    audio_path: None,
                    markdown_path: format!("{k}.md"),
                    json_path: format!("{k}.json"),
                },
            );
        }
        let videos2 = resolve_videos(&stub_info(), &opts, &token).expect("re-resolve");
        for v in &videos2 {
            manifest.upsert_discovered(v); // idempotent: no new keys
        }
        assert_eq!(
            manifest.order, keys_after_first,
            "re-run must re-derive identical keys (no duplicate/reprocess entries)"
        );
        // Every re-discovered video maps to a Done entry -> would be skipped.
        for v in &videos2 {
            assert!(
                matches!(
                    manifest.entries.get(&v.id).and_then(|e| e.state.as_ref()),
                    Some(VideoState::Done { .. })
                ),
                "re-discovered {} must already be Done (skipped on resume)",
                v.id
            );
        }
    }

    #[test]
    fn batch_file_strips_comments_and_blanks() {
        let body = "\
# a comment
https://youtu.be/aaaaaaaaaaa

  https://www.youtube.com/watch?v=bbbbbbbbbbb
; another comment
] yt-dlp-style comment
https://youtu.be/ccccccccccc
";
        let urls = parse_batch_file(body);
        assert_eq!(
            urls,
            vec![
                "https://youtu.be/aaaaaaaaaaa".to_owned(),
                "https://www.youtube.com/watch?v=bbbbbbbbbbb".to_owned(),
                "https://youtu.be/ccccccccccc".to_owned(),
            ]
        );
    }

    #[test]
    fn manifest_roundtrip_and_state_transitions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(MANIFEST_NAME);
        let mut m = Manifest::default();
        let v = VideoRef {
            id: "vid1".to_owned(),
            title: "Title".to_owned(),
            url: "https://youtu.be/vid1".to_owned(),
            duration_sec: Some(12.0),
        };
        m.upsert_discovered(&v);
        // Idempotent: a second upsert does not duplicate.
        m.upsert_discovered(&v);
        assert_eq!(m.order.len(), 1);
        m.set_state(
            "vid1",
            VideoState::Failed {
                error: "boom".to_owned(),
                attempts: 1,
            },
        );
        m.save(&path).expect("save");

        let reloaded = Manifest::load(&path).expect("load");
        assert_eq!(reloaded.attempts("vid1"), 1);
        assert!(matches!(
            reloaded.entries.get("vid1").and_then(|e| e.state.as_ref()),
            Some(VideoState::Failed { attempts: 1, .. })
        ));
    }

    #[test]
    fn cancelled_reusable_download_preserves_retry_budget_until_terminal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let audio_path = audio_dir.join("canceltry01.wav");
        std::fs::write(&audio_path, b"reusable cancelled download").expect("seed audio");

        let video = vr("canceltry01");
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.set_state(
            &video.id,
            VideoState::Failed {
                error: "second failure".to_owned(),
                attempts: MAX_ATTEMPTS - 1,
            },
        );
        manifest.save(&manifest_path).expect("seed failed manifest");

        persist_cancelled_download(&mut manifest, &manifest_path, &video.id, &audio_path)
            .expect("persist cancelled download");

        let mut retained = Manifest::load(&manifest_path).expect("reload retained download");
        assert_eq!(retained.attempts(&video.id), MAX_ATTEMPTS - 1);
        assert!(matches!(
            retained.entries[&video.id].state.as_ref(),
            Some(VideoState::Downloaded {
                audio_path: saved_path,
                attempts,
            }) if saved_path == &audio_path.display().to_string()
                && *attempts == MAX_ATTEMPTS - 1
        ));
        assert!(matches!(
            select_download_work(
                &video,
                retained.entries[&video.id].state.as_ref(),
                &audio_dir,
            ),
            DownloadWork::Reuse { audio_path: reused, .. } if reused == audio_path
        ));

        record_failure(&mut retained, &manifest_path, &video, "third failure")
            .expect("record next failure");
        assert_eq!(retained.attempts(&video.id), MAX_ATTEMPTS);
        persist_cancelled_download(&mut retained, &manifest_path, &video.id, &audio_path)
            .expect("retain terminal cancelled download");
        assert_eq!(retained.attempts(&video.id), MAX_ATTEMPTS);

        let (mut opts, buffer) = capturing_opts(vec![video.url.clone()], dir.path());
        opts.retry_failed = true;
        let summary = run_with_info(&opts, &stub_info()).expect("terminal budget run");
        assert_eq!(summary.skipped, vec![video.id.clone()]);
        assert!(summary.done.is_empty());
        assert!(summary.failed.is_empty());

        let persisted = Manifest::load(&manifest_path).expect("reload terminal manifest");
        assert!(matches!(
            persisted.entries[&video.id].state.as_ref(),
            Some(VideoState::Downloaded { attempts, .. }) if *attempts == MAX_ATTEMPTS
        ));
        let events = parse_events(&buffer);
        assert!(events.iter().any(|event| {
            event.get("event").and_then(serde_json::Value::as_str) == Some("youtube.skipped")
                && event.get("id").and_then(serde_json::Value::as_str) == Some(video.id.as_str())
                && event.get("reason").and_then(serde_json::Value::as_str)
                    == Some("retry_budget_exhausted")
        }));
        assert!(
            events.iter().all(
                |event| event.get("event").and_then(serde_json::Value::as_str)
                    != Some("youtube.downloading")
            ),
            "MAX_ATTEMPTS video must remain terminal"
        );
    }

    #[test]
    fn manifest_load_missing_is_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let m = Manifest::load(&dir.path().join("nope.json")).expect("load");
        assert!(m.order.is_empty());
    }

    #[test]
    fn manifest_load_corrupt_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad.json");
        std::fs::write(&path, b"{not json").expect("write");
        assert!(Manifest::load(&path).is_err());
    }

    /// Write-amplification guard (hotspot #4): the manifest is a
    /// full-rewrite-on-save BTreeMap, so each `save` writes O(N) bytes and the
    /// per-video save count drives total write volume. This drives K=200 video
    /// transitions through the manifest and measures cumulative bytes written
    /// under the OLD pattern (a `Downloaded` save *then* a `Done` save per
    /// video — 2 saves/video) vs the NEW pattern (a single terminal `Done` save
    /// per video). It asserts the new pattern roughly halves the bytes.
    ///
    /// O(N²) NOTE: even at 1 save/video the total is O(N²) bytes (each of N
    /// saves rewrites the whole ~O(N)-entry map). For realistic playlists
    /// (<500 videos, ~200 B/entry) that is tens of MB cumulative — modest. A
    /// future bead should only swap to an append-only journal (O(1) amortized
    /// per transition, compacted on load) if N ever exceeds ~2000.
    #[test]
    fn manifest_write_volume_halves_after_coalescing() {
        const K: usize = 200;

        // Build the discovered set once (mirrors the single bulk-init save).
        let mut manifest = Manifest::default();
        for i in 0..K {
            let v = VideoRef {
                id: format!("video{i:05}"),
                title: format!("Some Representative Video Title Number {i}"),
                url: format!("https://www.youtube.com/watch?v=video{i:05}"),
                duration_sec: Some(123.4),
            };
            manifest.upsert_discovered(&v);
        }

        let dir = tempfile::tempdir().expect("tempdir");

        // Helper: serialize the manifest exactly as `save` would and return the
        // byte length (the per-save write volume). Using the real serializer
        // keeps the measurement faithful to production `save`.
        let save_bytes = |m: &Manifest| -> usize {
            let path = dir.path().join("measure.json");
            m.save(&path).expect("save");
            std::fs::metadata(&path).expect("stat").len() as usize
        };

        // ── OLD pattern: bulk-init save + (Downloaded save, Done save)/video. ──
        let mut old_bytes = save_bytes(&manifest); // bulk-init
        let mut old_clone = manifest.clone();
        for i in 0..K {
            let id = format!("video{i:05}");
            old_clone.set_state(
                &id,
                VideoState::Downloaded {
                    audio_path: format!("audio/{id}.m4a"),
                    attempts: 0,
                },
            );
            old_bytes += save_bytes(&old_clone); // intermediate save (dropped now)
            old_clone.set_state(
                &id,
                VideoState::Done {
                    audio_path: None,
                    markdown_path: format!("out/{id}.md"),
                    json_path: format!("out/{id}.json"),
                },
            );
            old_bytes += save_bytes(&old_clone); // terminal save
        }

        // ── NEW pattern: bulk-init save + a single terminal Done save/video. ──
        let mut new_bytes = save_bytes(&manifest); // bulk-init
        let mut new_clone = manifest.clone();
        for i in 0..K {
            let id = format!("video{i:05}");
            new_clone.set_state(
                &id,
                VideoState::Done {
                    audio_path: None,
                    markdown_path: format!("out/{id}.md"),
                    json_path: format!("out/{id}.json"),
                },
            );
            new_bytes += save_bytes(&new_clone); // sole terminal save
        }

        let saved = old_bytes - new_bytes;
        let pct = (saved as f64 / old_bytes as f64) * 100.0;
        eprintln!(
            "manifest write volume @ K={K}: old={old_bytes} B, new={new_bytes} B, \
             saved={saved} B ({pct:.1}%)",
        );

        // Dropping one of two equal-cost O(N) saves per video, while the single
        // bulk-init save is shared, must cut total write volume by a bit under
        // half (the shared init save keeps it from hitting exactly 50%).
        assert!(
            new_bytes < old_bytes,
            "new pattern must write fewer bytes ({new_bytes} !< {old_bytes})"
        );
        assert!(
            pct > 45.0,
            "expected >45% byte reduction, got {pct:.1}% (old={old_bytes}, new={new_bytes})"
        );
    }
    // ---- bd-27v1.1: robot-mode NDJSON event stream ----------------------

    /// Robot-mode options wired to an in-memory capture buffer; returns the
    /// options plus the shared buffer the emitted lines land in.
    fn capturing_opts(
        urls: Vec<String>,
        output_dir: &Path,
    ) -> (
        YoutubeRunOptions,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut opts = opts_with_urls(urls);
        opts.output_dir = output_dir.to_path_buf();
        opts.robot_events = YoutubeRobotEvents::Capture(std::sync::Arc::clone(&buffer));
        (opts, buffer)
    }

    /// Parse captured NDJSON lines into JSON values.
    fn parse_events(buffer: &std::sync::Mutex<Vec<String>>) -> Vec<serde_json::Value> {
        buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|line| serde_json::from_str(line).expect("captured line must be JSON"))
            .collect()
    }

    #[test]
    fn cleanup_failure_keeps_durable_intent_and_emits_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let audio_path = audio_dir.join("cleanup0001.wav");
        let sentinel = b"retained audio must remain byte exact";
        std::fs::write(&audio_path, sentinel).expect("seed audio");

        let video = vr("cleanup0001");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.save(&manifest_path).expect("seed manifest");
        let paths = naming::output_paths(dir.path(), "rendered_cleanup0001");
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Capture(
            std::sync::Arc::clone(&buffer),
        ));
        let mut summary = YoutubeRunSummary::default();

        let disposition = settle_rendered_video(
            &mut manifest,
            &manifest_path,
            CompletedRender {
                video: &video,
                title: "Authoritative cleanup title",
                audio_path: &audio_path,
                paths: &paths,
                wall_ms: 42,
                rtf: Some(0.5),
                keep_audio: false,
            },
            &events,
            &mut summary,
            |_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "fixture denial",
                ))
            },
        )
        .expect("cleanup failure must become a per-video disposition");

        assert_eq!(disposition, RenderedVideoDisposition::CleanupFailed);
        assert_eq!(
            std::fs::read(&audio_path).expect("retained audio"),
            sentinel
        );
        assert!(summary.done.is_empty());
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].title, "Authoritative cleanup title");
        assert!(summary.failed[0].error.contains("fixture denial"));
        assert!(summary.failed[0].error.contains("cleanup0001.wav"));

        let persisted = Manifest::load(&manifest_path).expect("persisted manifest");
        let state = persisted.entries["cleanup0001"]
            .state
            .as_ref()
            .expect("durable cleanup state");
        assert!(matches!(
            state,
            VideoState::CleanupPending {
                audio_path: persisted_audio,
                markdown_path,
                json_path,
                attempts: 1,
                last_error: Some(last_error),
                ..
            } if persisted_audio == &audio_path.display().to_string()
                && markdown_path == &paths.md.display().to_string()
                && json_path == &paths.json.display().to_string()
                && last_error.contains("fixture denial")
        ));

        let parsed = parse_events(&buffer);
        assert_eq!(parsed.len(), 1, "cleanup failure emits one event");
        assert_eq!(parsed[0]["event"], "youtube.failed");
        assert!(parsed.iter().all(|event| event["event"] != "youtube.done"));
    }

    #[test]
    fn missing_audio_during_cleanup_is_already_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let video = vr("notfound001");
        let audio_path = dir.path().join("audio/notfound001.wav");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.save(&manifest_path).expect("seed manifest");
        let paths = naming::output_paths(dir.path(), "rendered_notfound001");
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Capture(
            std::sync::Arc::clone(&buffer),
        ));
        let mut summary = YoutubeRunSummary::default();

        let disposition = settle_rendered_video(
            &mut manifest,
            &manifest_path,
            CompletedRender {
                video: &video,
                title: "Already absent",
                audio_path: &audio_path,
                paths: &paths,
                wall_ms: 1,
                rtf: None,
                keep_audio: false,
            },
            &events,
            &mut summary,
            |_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "already absent",
                ))
            },
        )
        .expect("NotFound is successful cleanup");

        assert_eq!(disposition, RenderedVideoDisposition::Done);
        assert_eq!(summary.done, ["notfound001"]);
        assert!(summary.failed.is_empty());
        assert!(matches!(
            manifest.entries["notfound001"].state.as_ref(),
            Some(VideoState::Done {
                audio_path: None,
                ..
            })
        ));
        let parsed = parse_events(&buffer);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["event"], "youtube.done");
    }

    #[test]
    fn successful_cleanup_persists_intent_before_delete_and_done_after() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let video = vr("deleted0001");
        let audio_path = audio_dir.join("deleted0001.wav");
        std::fs::write(&audio_path, b"delete me").expect("seed audio");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.save(&manifest_path).expect("seed manifest");
        let paths = naming::output_paths(dir.path(), "rendered_deleted0001");
        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Off);
        let mut summary = YoutubeRunSummary::default();

        let disposition = settle_rendered_video(
            &mut manifest,
            &manifest_path,
            CompletedRender {
                video: &video,
                title: "Deleted audio",
                audio_path: &audio_path,
                paths: &paths,
                wall_ms: 1,
                rtf: None,
                keep_audio: false,
            },
            &events,
            &mut summary,
            |path| {
                let persisted = Manifest::load(&manifest_path)
                    .expect("cleanup intent must be durable before deletion");
                assert!(matches!(
                    persisted.entries["deleted0001"].state.as_ref(),
                    Some(VideoState::CleanupPending {
                        audio_path,
                        attempts: 0,
                        ..
                    }) if audio_path == &path.display().to_string()
                ));
                std::fs::remove_file(path)
            },
        )
        .expect("successful cleanup");

        assert_eq!(disposition, RenderedVideoDisposition::Done);
        assert!(!audio_path.exists());
        assert_eq!(summary.done, ["deleted0001"]);
        assert!(matches!(
            manifest.entries["deleted0001"].state.as_ref(),
            Some(VideoState::Done {
                audio_path: None,
                ..
            })
        ));
    }

    #[test]
    fn cleanup_intent_save_failure_leaves_audio_and_prior_durable_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let video = vr("intentfail01");
        let audio_path = audio_dir.join("intentfail01.wav");
        std::fs::write(&audio_path, b"must survive failed intent save").expect("seed audio");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.save(&manifest_path).expect("seed manifest");
        let paths = naming::output_paths(dir.path(), "rendered_intentfail01");
        let rendered = CompletedRender {
            video: &video,
            title: "Intent save failure",
            audio_path: &audio_path,
            paths: &paths,
            wall_ms: 3,
            rtf: Some(0.25),
            keep_audio: false,
        };
        let mut fail_save = |_: &Manifest, _: &Path| {
            Err(FwError::Storage("fixture intent save failure".to_owned()))
        };

        let error =
            persist_cleanup_intent_with(&mut manifest, &manifest_path, &rendered, &mut fail_save)
                .expect_err("intent persistence failure must stop before deletion");
        assert!(error.to_string().contains("fixture intent save failure"));
        assert_eq!(
            std::fs::read(&audio_path).expect("audio survives"),
            b"must survive failed intent save"
        );
        let durable = Manifest::load(&manifest_path).expect("reload prior manifest");
        assert!(matches!(
            durable.entries["intentfail01"].state.as_ref(),
            Some(VideoState::Pending)
        ));
    }

    #[test]
    fn pending_cleanup_rejects_unowned_manifest_path_before_removal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outside_path = dir.path().join("outside.wav");
        std::fs::write(&outside_path, b"must not be deleted").expect("seed outside file");
        let video = vr("unownedclean1");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.set_state(
            &video.id,
            VideoState::CleanupPending {
                title: "Unowned cleanup".to_owned(),
                audio_path: outside_path.display().to_string(),
                markdown_path: dir.path().join("rendered.md").display().to_string(),
                json_path: dir.path().join("rendered.json").display().to_string(),
                wall_ms: 1,
                rtf: None,
                attempts: 0,
                last_error: None,
            },
        );
        manifest.save(&manifest_path).expect("seed cleanup intent");
        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Off);
        let mut summary = YoutubeRunSummary::default();
        let remover_called = std::cell::Cell::new(false);

        let error = finish_pending_cleanup(
            &mut manifest,
            &manifest_path,
            &video,
            &events,
            &mut summary,
            |_| {
                remover_called.set(true);
                Ok(())
            },
        )
        .expect_err("unowned manifest path must fail closed");

        assert!(matches!(error, FwError::ContractViolation(_)));
        assert!(!remover_called.get());
        assert_eq!(
            std::fs::read(&outside_path).expect("outside file remains"),
            b"must not be deleted"
        );
        assert!(summary.done.is_empty());
        assert!(summary.failed.is_empty());
    }

    #[test]
    fn pending_cleanup_restart_deletes_present_audio_without_retranscribing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let video = vr("pendingclean01");
        let audio_path = audio_dir.join("pendingclean01.wav");
        std::fs::write(&audio_path, b"pending cleanup").expect("seed audio");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.set_state(
            &video.id,
            VideoState::Failed {
                error: "prior transcription failure".to_owned(),
                attempts: 2,
            },
        );
        manifest.save(&manifest_path).expect("seed manifest");
        let paths = naming::output_paths(dir.path(), "rendered_pendingclean01");
        let rendered = CompletedRender {
            video: &video,
            title: "Pending cleanup",
            audio_path: &audio_path,
            paths: &paths,
            wall_ms: 7,
            rtf: Some(0.75),
            keep_audio: false,
        };
        let mut save_manifest = Manifest::save;
        persist_cleanup_intent_with(&mut manifest, &manifest_path, &rendered, &mut save_manifest)
            .expect("persist cleanup intent");

        let (opts, buffer) = capturing_opts(vec![video.url.clone()], dir.path());
        let summary = run_with_info(&opts, &stub_info())
            .expect("restart should settle pending deletion without retrying ASR");

        assert!(!audio_path.exists());
        assert_eq!(summary.done, ["pendingclean01"]);
        assert!(summary.failed.is_empty());
        assert!(summary.skipped.is_empty());
        assert!(matches!(
            Manifest::load(&manifest_path).expect("reload done").entries["pendingclean01"]
                .state
                .as_ref(),
            Some(VideoState::Done {
                audio_path: None,
                markdown_path,
                json_path,
            }) if markdown_path == &paths.md.display().to_string()
                && json_path == &paths.json.display().to_string()
        ));
        let parsed = parse_events(&buffer);
        assert!(parsed.iter().any(|event| event["event"] == "youtube.done"));
        assert!(parsed.iter().all(|event| {
            event["event"] != "youtube.downloading" && event["event"] != "youtube.transcribing"
        }));
    }

    #[test]
    fn done_save_failure_recovers_when_deleted_audio_is_already_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let video = vr("donesavefail1");
        let audio_path = audio_dir.join("donesavefail1.wav");
        std::fs::write(&audio_path, b"delete before failed done save").expect("seed audio");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.save(&manifest_path).expect("seed manifest");
        let paths = naming::output_paths(dir.path(), "rendered_donesavefail1");
        let rendered = CompletedRender {
            video: &video,
            title: "Done save failure",
            audio_path: &audio_path,
            paths: &paths,
            wall_ms: 9,
            rtf: None,
            keep_audio: false,
        };
        let mut save_manifest = Manifest::save;
        persist_cleanup_intent_with(&mut manifest, &manifest_path, &rendered, &mut save_manifest)
            .expect("persist cleanup intent");

        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Off);
        let mut summary = YoutubeRunSummary::default();
        let mut fail_done_save =
            |_: &Manifest, _: &Path| Err(FwError::Storage("fixture done save failure".to_owned()));
        let error = finish_pending_cleanup_with(
            &mut manifest,
            &manifest_path,
            &video,
            &events,
            &mut summary,
            &mut fail_done_save,
            std::fs::remove_file,
        )
        .expect_err("final Done persistence must fail after deletion");
        assert!(error.to_string().contains("fixture done save failure"));
        assert!(!audio_path.exists());
        assert!(summary.done.is_empty());

        let mut restarted = Manifest::load(&manifest_path).expect("restart pending manifest");
        assert!(matches!(
            restarted.entries["donesavefail1"].state.as_ref(),
            Some(VideoState::CleanupPending { .. })
        ));
        let recovered = finish_pending_cleanup(
            &mut restarted,
            &manifest_path,
            &video,
            &events,
            &mut summary,
            std::fs::remove_file,
        )
        .expect("NotFound must settle the durable cleanup intent");
        assert_eq!(recovered, RenderedVideoDisposition::Done);
        assert_eq!(summary.done, ["donesavefail1"]);
        assert!(matches!(
            Manifest::load(&manifest_path)
                .expect("reload recovered Done")
                .entries["donesavefail1"]
                .state
                .as_ref(),
            Some(VideoState::Done {
                audio_path: None,
                ..
            })
        ));
    }

    #[test]
    fn keep_audio_never_invokes_remover_and_records_exact_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_dir = dir.path().join("audio");
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        let video = vr("keepaudio01");
        let audio_path = audio_dir.join("keepaudio01.wav");
        std::fs::write(&audio_path, b"keep me").expect("seed audio");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.save(&manifest_path).expect("seed manifest");
        let paths = naming::output_paths(dir.path(), "rendered_keepaudio01");
        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Off);
        let mut summary = YoutubeRunSummary::default();
        let remover_called = std::cell::Cell::new(false);

        let disposition = settle_rendered_video(
            &mut manifest,
            &manifest_path,
            CompletedRender {
                video: &video,
                title: "Kept audio",
                audio_path: &audio_path,
                paths: &paths,
                wall_ms: 1,
                rtf: None,
                keep_audio: true,
            },
            &events,
            &mut summary,
            |_| {
                remover_called.set(true);
                Ok(())
            },
        )
        .expect("keep-audio settlement");

        assert_eq!(disposition, RenderedVideoDisposition::Done);
        assert!(!remover_called.get());
        assert_eq!(std::fs::read(&audio_path).expect("kept audio"), b"keep me");
        assert!(matches!(
            manifest.entries["keepaudio01"].state.as_ref(),
            Some(VideoState::Done {
                audio_path: Some(path),
                ..
            }) if path == &audio_path.display().to_string()
        ));
    }

    #[test]
    fn downstream_failure_is_persisted_summarized_and_emitted_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let video = vr("failure0001");
        let mut manifest = Manifest::default();
        manifest.upsert_discovered(&video);
        manifest.save(&manifest_path).expect("seed manifest");
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Capture(
            std::sync::Arc::clone(&buffer),
        ));
        let mut summary = YoutubeRunSummary::default();

        record_and_emit_failure(
            &mut manifest,
            &manifest_path,
            &video,
            "Authoritative metadata title",
            "render fault",
            &events,
            &mut summary,
        )
        .expect("record downstream failure");

        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].title, "Authoritative metadata title");
        assert!(matches!(
            manifest.entries["failure0001"].state.as_ref(),
            Some(VideoState::Failed {
                error,
                attempts: 1,
            }) if error == "render fault"
        ));
        let persisted = Manifest::load(&manifest_path).expect("reload manifest");
        assert!(matches!(
            persisted.entries["failure0001"].state.as_ref(),
            Some(VideoState::Failed { attempts: 1, .. })
        ));

        let parsed = parse_events(&buffer);
        assert_eq!(parsed.len(), 1, "exactly one failure event");
        assert_eq!(parsed[0]["event"], "youtube.failed");
        assert_eq!(parsed[0]["title"], "Authoritative metadata title");
        assert_eq!(parsed[0]["error"], "render fault");
        assert_eq!(parsed[0]["attempts"], 1);
    }

    #[test]
    fn finalize_run_emits_exactly_one_cancelled_terminal_event() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Capture(
            std::sync::Arc::clone(&buffer),
        ));
        events
            .emit("run_start", serde_json::json!({ "n_urls": 1 }))
            .expect("start event");

        let error = finalize_run(
            &events,
            Err(FwError::Cancelled("test cancellation".to_owned())),
        )
        .expect_err("original cancellation must propagate");
        assert!(matches!(error, FwError::Cancelled(_)));

        let parsed = parse_events(&buffer);
        let terminals: Vec<_> = parsed
            .iter()
            .filter(|event| event["event"] == "youtube.run_complete")
            .collect();
        assert_eq!(terminals.len(), 1);
        assert_eq!(parsed.last().expect("last event")["event"], "youtube.run_complete");
        assert_eq!(terminals[0]["cancelled"], true);
    }

    #[test]
    fn run_emits_terminal_event_when_input_resolution_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (opts, buffer) = capturing_opts(vec!["not-a-youtube-url".to_owned()], dir.path());

        let error = run_with_info(&opts, &stub_info()).expect_err("invalid URL must fail");
        assert!(matches!(error, FwError::InvalidRequest(_)));
        let parsed = parse_events(&buffer);
        let names: Vec<_> = parsed
            .iter()
            .map(|event| event["event"].as_str().expect("event"))
            .collect();
        assert_eq!(names, ["youtube.run_start", "youtube.run_complete"]);
        assert_eq!(parsed[1]["cancelled"], false);
    }

    #[test]
    fn abort_on_error_stops_later_waves_without_claiming_cancellation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (mut opts, buffer) = capturing_opts(
            vec![
                "https://www.youtube.com/watch?v=ABORTFAIL01&fw_stub_fail=private".to_owned(),
                "https://www.youtube.com/watch?v=AFTERABORT1".to_owned(),
            ],
            dir.path(),
        );
        opts.abort_on_error = true;
        opts.concurrency = 1;
        opts.batch_size = 1;

        let summary = run_with_info(&opts, &stub_info()).expect("per-video failure summary");
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].id, "ABORTFAIL01");
        assert!(summary.done.is_empty());
        assert!(!summary.cancelled, "fail-fast is not Ctrl+C cancellation");

        let parsed = parse_events(&buffer);
        let terminal = parsed.last().expect("terminal event");
        assert_eq!(terminal["event"], "youtube.run_complete");
        let failed = terminal["failed"].as_array().expect("failed summaries");
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0]["id"], "ABORTFAIL01");
        assert_eq!(failed[0]["title"], "");
        assert!(
            failed[0]["error"]
                .as_str()
                .expect("failure error")
                .contains("private")
        );
        assert_eq!(terminal["cancelled"], false);
        let later_work_events = parsed
            .iter()
            .filter(|event| event["id"] == "AFTERABORT1")
            .filter(|event| event["event"] != "youtube.discovered")
            .count();
        assert_eq!(later_work_events, 0, "later waves must not start after abort");

        let manifest = Manifest::load(&dir.path().join(MANIFEST_NAME)).expect("manifest");
        assert!(matches!(
            manifest.entries["ABORTFAIL01"].state.as_ref(),
            Some(VideoState::Failed { attempts: 1, .. })
        ));
        assert!(matches!(
            manifest.entries["AFTERABORT1"].state.as_ref(),
            Some(VideoState::Pending)
        ));
    }

    #[test]
    fn emitter_envelope_carries_schema_seq_ts_and_stable_run_id() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events =
            YoutubeEventEmitter::new(YoutubeRobotEvents::Capture(std::sync::Arc::clone(&buffer)));
        events
            .emit("run_start", serde_json::json!({ "n_urls": 0 }))
            .expect("emit 1");
        events
            .emit("discovered", serde_json::json!({ "id": "x" }))
            .expect("emit 2");
        events
            .emit("run_complete", serde_json::json!({ "cancelled": false }))
            .expect("emit 3");

        let parsed = parse_events(&buffer);
        assert_eq!(parsed.len(), 3);
        let run_ids: Vec<&str> = parsed
            .iter()
            .map(|v| v["run_id"].as_str().expect("run_id"))
            .collect();
        assert!(run_ids.iter().all(|id| id.starts_with("yt-")));
        assert_eq!(run_ids[0], run_ids[1]);
        assert_eq!(run_ids[1], run_ids[2]);
        for (i, value) in parsed.iter().enumerate() {
            assert_eq!(
                value["schema_version"],
                serde_json::json!(crate::robot::ROBOT_SCHEMA_VERSION),
                "event {i} schema"
            );
            assert_eq!(value["seq"], serde_json::json!((i + 1) as u64));
            assert!(
                value["event"]
                    .as_str()
                    .expect("event")
                    .starts_with("youtube."),
                "event {i} must be namespaced"
            );
            let ts = value["ts"].as_str().expect("ts");
            chrono::DateTime::parse_from_rfc3339(ts)
                .unwrap_or_else(|e| panic!("ts must be RFC-3339 ({e}): {ts}"));
        }
        assert_eq!(parsed[0]["event"], "youtube.run_start");
        assert_eq!(parsed[2]["event"], "youtube.run_complete");
    }

    #[test]
    fn emitter_payload_cannot_forge_authoritative_envelope_fields() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events =
            YoutubeEventEmitter::new(YoutubeRobotEvents::Capture(std::sync::Arc::clone(&buffer)));
        let expected_run_id = events.run_id().to_owned();
        events
            .emit(
                "discovered",
                serde_json::json!({
                    "event": "forged.event",
                    "schema_version": "forged-schema",
                    "run_id": "forged-run",
                    "seq": 99,
                    "ts": "not-a-timestamp",
                    "id": "safe-id",
                }),
            )
            .expect("emit protected envelope");

        let parsed = parse_events(&buffer);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["event"], "youtube.discovered");
        assert_eq!(
            parsed[0]["schema_version"],
            crate::robot::ROBOT_SCHEMA_VERSION
        );
        assert_eq!(parsed[0]["run_id"], expected_run_id);
        assert_eq!(parsed[0]["seq"], 1);
        assert_eq!(parsed[0]["id"], "safe-id");
        chrono::DateTime::parse_from_rfc3339(parsed[0]["ts"].as_str().expect("ts"))
            .expect("authoritative RFC-3339 timestamp");
    }

    #[test]
    fn emitter_off_mode_is_a_no_op() {
        let events = YoutubeEventEmitter::new(YoutubeRobotEvents::Off);
        events
            .emit("downloading", serde_json::json!({ "id": "x" }))
            .expect("off-mode emit must succeed without side effects");
    }

    #[test]
    fn emitter_seq_is_unique_across_threads() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events = std::sync::Arc::new(YoutubeEventEmitter::new(YoutubeRobotEvents::Capture(
            std::sync::Arc::clone(&buffer),
        )));
        let workers = 8_u64;
        let per_worker = 25_u64;
        let mut handles = Vec::new();
        for worker in 0..workers {
            let events = std::sync::Arc::clone(&events);
            handles.push(std::thread::spawn(move || {
                for i in 0..per_worker {
                    events
                        .emit("tick", serde_json::json!({ "worker": worker, "i": i }))
                        .expect("emit");
                }
            }));
        }
        for handle in handles {
            handle.join().expect("worker thread");
        }
        let lines_len = buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        assert_eq!(lines_len, (workers * per_worker) as usize);
        let seqs: Vec<u64> = parse_events(&buffer)
            .into_iter()
            .map(|v| v["seq"].as_u64().expect("seq"))
            .collect();
        let expected: Vec<u64> = (1..=workers * per_worker).collect();
        assert_eq!(
            seqs, expected,
            "physical event order must be monotonically sequenced"
        );
    }

    #[test]
    fn robot_run_streams_start_discovered_skipped_and_terminal_complete() {
        // Both videos pre-Done in the manifest: the run emits the full
        // bookkeeping stream without ever reaching transcription.
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest_path = dir.path().join(MANIFEST_NAME);
        let mut manifest = Manifest::default();
        for vid in ["vid000000001", "vid000000002"] {
            manifest.order.push(vid.to_owned());
            manifest.entries.insert(
                vid.to_owned(),
                ManifestEntry {
                    id: vid.to_owned(),
                    title: format!("Title {vid}"),
                    url: format!("https://www.youtube.com/watch?v={vid}"),
                    state: Some(VideoState::Done {
                        audio_path: None,
                        markdown_path: format!("out/{vid}.md"),
                        json_path: format!("out/{vid}.json"),
                    }),
                },
            );
        }
        manifest.save(&manifest_path).expect("seed manifest");

        let (opts, buffer) = capturing_opts(
            vec!["https://www.youtube.com/playlist?list=PLstub".to_owned()],
            dir.path(),
        );
        let summary = run_with_info(&opts, &stub_info()).expect("run");

        assert_eq!(summary.skipped.len(), 2);
        assert!(summary.done.is_empty());
        assert!(summary.failed.is_empty());
        assert!(!summary.cancelled);

        let parsed = parse_events(&buffer);
        let names: Vec<&str> = parsed
            .iter()
            .map(|v| v["event"].as_str().expect("event"))
            .collect();
        assert_eq!(
            names,
            vec![
                "youtube.run_start",
                "youtube.discovered",
                "youtube.discovered",
                "youtube.skipped",
                "youtube.skipped",
                "youtube.run_complete",
            ],
            "full expected stream, got {names:?}"
        );
        assert_eq!(parsed[0]["output_dir"], dir.path().display().to_string());
        assert_eq!(parsed[0]["concurrency"], serde_json::json!(1));
        assert_eq!(parsed[3]["reason"], "already_done");
        assert_eq!(parsed[4]["reason"], "already_done");
        let complete = &parsed[5];
        assert_eq!(complete["done"], serde_json::json!([]));
        assert_eq!(
            complete["skipped"],
            serde_json::json!(["vid000000001", "vid000000002"])
        );
        assert_eq!(complete["failed"], serde_json::json!([]));
        assert_eq!(complete["cancelled"], serde_json::json!(false));
    }

    #[test]
    fn robot_run_streams_failed_event_on_deterministic_download_failure() {
        // fw_stub_fail=private is a deterministic failure: fail-fast (no retry
        // backoff), the engine is constructed but never transcribes, and the
        // stream still terminates with a run_complete aggregate.
        let dir = tempfile::tempdir().expect("tempdir");
        let (opts, buffer) = capturing_opts(
            vec!["https://www.youtube.com/watch?v=PRIVATE_ID&fw_stub_fail=private".to_owned()],
            dir.path(),
        );
        let summary = run_with_info(&opts, &stub_info()).expect("run");
        assert_eq!(summary.failed.len(), 1);
        assert_eq!(summary.failed[0].id, "PRIVATE_ID");
        assert!(!summary.cancelled);

        let parsed = parse_events(&buffer);
        eprintln!("captured youtube events: {parsed:?}");
        let names: Vec<&str> = parsed
            .iter()
            .map(|v| v["event"].as_str().expect("event"))
            .collect();
        assert_eq!(
            names,
            vec![
                "youtube.run_start",
                "youtube.discovered",
                "youtube.downloading",
                "youtube.failed",
                "youtube.run_complete",
            ],
            "failure stream, got {names:?}"
        );
        assert_eq!(parsed[3]["id"], "PRIVATE_ID");
        // classify_stderr maps the stub's "Private video" stderr to
        // FwError::InvalidRequest("video is private; ...") before the event
        // ever carries it, so match the mapped message, not the raw stderr.
        assert!(
            parsed[3]["error"]
                .as_str()
                .expect("error")
                .contains("private")
        );
        assert_eq!(parsed[3]["attempts"], serde_json::json!(1));
        let complete = &parsed[4];
        assert_eq!(complete["failed"].as_array().expect("failed").len(), 1);
        assert_eq!(complete["cancelled"], serde_json::json!(false));
    }
}
