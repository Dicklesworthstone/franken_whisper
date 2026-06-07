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
                if let Ok((audio_path, _meta)) = &outcome {
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
            let (audio_path, meta) = match outcome {
                Ok(pair) => pair,
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
            // NB: we intentionally do NOT persist a `Downloaded` state here.
            // Write-amplification fix: the manifest is a full-rewrite-on-save
            // BTreeMap, so each save is O(N) bytes; saving here once per video
            // makes per-video work O(N²) total writes for an N-video playlist.
            // This intermediate save bought no resume benefit anyway — the
            // partition logic above re-enters `download_one` for any non-Done /
            // non-Failed entry (a `Downloaded` entry is re-downloaded on
            // resume, which the contract permits), so the only durable states
            // that matter are the terminal `Done`/`Failed` (and the rare
            // cancel-persist below). The single per-video save now happens at
            // the terminal transition.
            match transcribe_and_render(&engine, opts, &video, &meta, &audio_path) {
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
                    // No terminal state to persist (the entry is still
                    // `Pending`); a resume re-downloads + re-transcribes, which
                    // the contract permits. Dropping this save avoids a full
                    // O(N)-byte manifest rewrite on the cancel path.
                    summary.cancelled = true;
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

/// Download a single video's audio. This performs the **single** authoritative
/// `yt-dlp -j` metadata fetch for the video and returns the fetched
/// [`VideoMeta`] alongside the audio path, so the renderer never re-fetches it.
fn download_one(
    info: &YtdlpInfo,
    video: &VideoRef,
    audio_dir: &Path,
    token: &CancellationToken,
) -> Result<(PathBuf, VideoMeta), String> {
    let t_meta = std::time::Instant::now();
    let meta = ytdlp::fetch_metadata(info, &video.url, token).map_err(|e| e.to_string())?;
    crate::native_engine::perf_span("yt.dl_metadata", t_meta.elapsed().as_secs_f64() * 1e3, "");
    let t_dl = std::time::Instant::now();
    let path = ytdlp::download_audio(info, &meta, audio_dir, token).map_err(|e| e.to_string())?;
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
) -> FwResult<OutputPaths> {
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
}
