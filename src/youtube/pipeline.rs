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
use super::render::{self, RenderInput, RenderRun, RenderVideo};
use super::ytdlp::{self, UrlKind, VideoMeta, VideoRef, YtdlpInfo};

/// Manifest file name written into the output directory.
const MANIFEST_NAME: &str = ".fw_youtube_manifest.json";

/// A video that has failed this many times is not retried again on a plain
/// re-run (it still counts as skipped). `--no-retry` skips any prior failure
/// regardless; deleting the manifest entry forces a fresh attempt.
const MAX_ATTEMPTS: u32 = 3;

/// Per-video processing state. Persisted in the manifest so a re-run resumes
/// exactly where a crash or cancellation left off.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum VideoState {
    /// Discovered, not yet started.
    Pending,
    /// Audio downloaded to `audio_path`, not yet transcribed.
    Downloaded { audio_path: String },
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
        std::fs::write(&tmp, body).map_err(FwError::Io)?;
        std::fs::rename(&tmp, path).map_err(FwError::Io)?;
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
            Some(VideoState::Failed { attempts, .. }) => *attempts,
            _ => 0,
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
    /// Abort the whole run on the first per-video failure.
    pub abort_on_error: bool,
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
                // Resolve to a concrete id via the canonical metadata fetch so
                // dedup and naming have a stable id even for short/youtu.be
                // forms. (Ambiguous watch?v=X&list=Y -> the single video, per
                // --no-playlist.)
                let meta = ytdlp::fetch_metadata(info, &url, token)?;
                if seen.insert(meta.id.clone()) {
                    videos.push(VideoRef {
                        id: meta.id.clone(),
                        title: meta.title.clone(),
                        url: meta.webpage_url.clone(),
                        duration_sec: meta.duration_sec,
                    });
                }
            }
        }
    }
    Ok(videos)
}

/// A unit of work handed from the download pool to the transcription consumer.
struct DownloadResult {
    video: VideoRef,
    /// `Ok(audio_path)` on success, `Err(message)` on download failure.
    outcome: Result<PathBuf, String>,
}

/// Run the full ingestion pipeline.
pub fn run(opts: &YoutubeRunOptions) -> FwResult<YoutubeRunSummary> {
    let token = CancellationToken::unbounded();
    let info = ytdlp::probe()?;
    if info.stale {
        tracing::warn!(
            version = %info.version,
            "yt-dlp build is over 90 days old; YouTube may have changed — consider `yt-dlp -U`"
        );
    }

    std::fs::create_dir_all(&opts.output_dir).map_err(FwError::Io)?;
    let audio_dir = opts.output_dir.join("audio");
    std::fs::create_dir_all(&audio_dir).map_err(FwError::Io)?;
    let manifest_path = opts.output_dir.join(MANIFEST_NAME);

    let mut manifest = Manifest::load(&manifest_path)?;
    let videos = resolve_videos(&info, opts, &token)?;
    for v in &videos {
        manifest.upsert_discovered(v);
    }
    manifest.save(&manifest_path)?;

    // Partition into work-to-do vs already-satisfied (idempotent resume).
    let mut summary = YoutubeRunSummary::default();
    let mut to_process: Vec<VideoRef> = Vec::new();
    for v in &videos {
        match manifest.entries.get(&v.id).and_then(|e| e.state.as_ref()) {
            Some(VideoState::Done { .. }) => {
                tracing::info!(id = %v.id, "already done; skipping");
                summary.skipped.push(v.id.clone());
            }
            Some(VideoState::Failed { .. }) if !opts.retry_failed => {
                tracing::info!(id = %v.id, "previously failed; --no-retry, skipping");
                summary.skipped.push(v.id.clone());
            }
            Some(VideoState::Failed { attempts, .. }) if *attempts >= MAX_ATTEMPTS => {
                tracing::info!(
                    id = %v.id, attempts,
                    "exhausted retry budget; skipping (delete the manifest entry to force a retry)"
                );
                summary.skipped.push(v.id.clone());
            }
            _ => to_process.push(v.clone()),
        }
    }

    if to_process.is_empty() {
        return Ok(summary);
    }

    let engine = FrankenWhisperEngine::new()?;

    // ── Download stage: bounded worker pool feeding a bounded channel. ──
    // Capacity == concurrency keeps disk bounded (~concurrency in-flight +
    // queued). Each worker kills its yt-dlp child if the token fires.
    let concurrency = opts.concurrency.max(1);
    let (tx, rx) = sync_channel::<DownloadResult>(concurrency);
    let work = std::sync::Arc::new(std::sync::Mutex::new(to_process.into_iter()));
    let audio_dir_arc = std::sync::Arc::new(audio_dir.clone());
    let info_arc = std::sync::Arc::new(info.clone());

    std::thread::scope(|scope| -> FwResult<()> {
        for _ in 0..concurrency {
            let tx = tx.clone();
            let work = std::sync::Arc::clone(&work);
            let audio_dir = std::sync::Arc::clone(&audio_dir_arc);
            let info = std::sync::Arc::clone(&info_arc);
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
                    let Some(video) = next else { break };
                    let outcome = download_one(&info, &video, &audio_dir, &dl_token);
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
                if let Ok(audio_path) = &outcome {
                    manifest.set_state(
                        &video.id,
                        VideoState::Downloaded {
                            audio_path: audio_path.display().to_string(),
                        },
                    );
                    manifest.save(&manifest_path)?;
                }
                summary.cancelled = true;
                break;
            }
            let audio_path = match outcome {
                Ok(p) => p,
                Err(error) => {
                    record_failure(&mut manifest, &manifest_path, &video, &error)?;
                    summary.failed.push(FailedVideo {
                        id: video.id.clone(),
                        title: video.title.clone(),
                        error,
                    });
                    if opts.abort_on_error {
                        summary.cancelled = true;
                        break;
                    }
                    continue;
                }
            };
            manifest.set_state(
                &video.id,
                VideoState::Downloaded {
                    audio_path: audio_path.display().to_string(),
                },
            );
            manifest.save(&manifest_path)?;

            match transcribe_and_render(&engine, &info_arc, opts, &video, &audio_path) {
                Ok(paths) => {
                    let audio_kept = if opts.keep_audio {
                        Some(audio_path.display().to_string())
                    } else {
                        let _ = std::fs::remove_file(&audio_path);
                        None
                    };
                    manifest.set_state(
                        &video.id,
                        VideoState::Done {
                            audio_path: audio_kept,
                            markdown_path: paths.md.display().to_string(),
                            json_path: paths.json.display().to_string(),
                        },
                    );
                    manifest.save(&manifest_path)?;
                    summary.done.push(video.id.clone());
                }
                Err(FwError::Cancelled(_)) => {
                    summary.cancelled = true;
                    manifest.save(&manifest_path)?;
                    break;
                }
                Err(e) => {
                    let error = e.to_string();
                    record_failure(&mut manifest, &manifest_path, &video, &error)?;
                    summary.failed.push(FailedVideo {
                        id: video.id.clone(),
                        title: video.title.clone(),
                        error,
                    });
                    if opts.abort_on_error {
                        summary.cancelled = true;
                        break;
                    }
                }
            }
        }
        Ok(())
    })?;

    Ok(summary)
}

/// Download a single video's audio (metadata fetch + best-audio download).
fn download_one(
    info: &YtdlpInfo,
    video: &VideoRef,
    audio_dir: &Path,
    token: &CancellationToken,
) -> Result<PathBuf, String> {
    let t_meta = std::time::Instant::now();
    let meta = ytdlp::fetch_metadata(info, &video.url, token).map_err(|e| e.to_string())?;
    crate::native_engine::perf_span("yt.dl_metadata", t_meta.elapsed().as_secs_f64() * 1e3, "");
    let t_dl = std::time::Instant::now();
    let path = ytdlp::download_audio(info, &meta, audio_dir, token).map_err(|e| e.to_string());
    crate::native_engine::perf_span("yt.download", t_dl.elapsed().as_secs_f64() * 1e3, "");
    path
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

/// Transcribe a downloaded audio file and render markdown + JSON.
fn transcribe_and_render(
    engine: &FrankenWhisperEngine,
    info: &YtdlpInfo,
    opts: &YoutubeRunOptions,
    video: &VideoRef,
    audio_path: &Path,
) -> FwResult<OutputPaths> {
    // Re-fetch metadata for the richest naming/JSON fields (cheap; also lets a
    // resume render without a re-download). On failure fall back to the
    // VideoRef we already have.
    let token = CancellationToken::unbounded();
    let t_rmeta = std::time::Instant::now();
    let meta_opt = ytdlp::fetch_metadata(info, &video.url, &token).ok();
    crate::native_engine::perf_span(
        "yt.render_metadata",
        t_rmeta.elapsed().as_secs_f64() * 1e3,
        "",
    );
    let meta = meta_opt.unwrap_or(VideoMeta {
        id: video.id.clone(),
        title: video.title.clone(),
        channel: None,
        uploader: None,
        upload_date: None,
        duration_sec: video.duration_sec,
        webpage_url: video.url.clone(),
        description: None,
        availability: None,
        live_status: None,
    });

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

    let rtf = meta
        .duration_sec
        .filter(|d| *d > 0.0)
        .map(|d| (wall_ms as f64 / 1000.0) / d);

    let (engine_label, backend_label) = engine_labels(&report);

    let base = naming::sanitize_base(&meta.title, meta.upload_date.as_deref(), &meta.id);
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
    };

    let t_render = std::time::Instant::now();
    render::write_atomic(&paths.md, &render::render_markdown(&input))?;
    let json = render::render_json(&input);
    render::write_atomic(
        &paths.json,
        &serde_json::to_string_pretty(&json).map_err(FwError::Json)?,
    )?;
    crate::native_engine::perf_span("yt.render", t_render.elapsed().as_secs_f64() * 1e3, "");
    Ok(paths)
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
}
