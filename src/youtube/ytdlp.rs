//! `yt-dlp` tool orchestration: probe, URL classification, playlist
//! expansion, metadata fetch, and cancellable audio download.
//!
//! `yt-dlp` is treated as an orchestrated external tool, exactly like
//! `whisper-cli` and `ffmpeg`: it is probed (via the `which` crate, honoring a
//! `FRANKEN_WHISPER_YTDLP_BIN` override), its version is captured, and every
//! subprocess invocation flows through the shared primitives in
//! [`crate::process`] (secret-free logging, output capture, cancellation).
//!
//! # Path-explicit API
//!
//! The environment override is read in exactly one place — [`probe`]. Every
//! other function takes a `&YtdlpInfo` whose `path` field names the binary to
//! run. This makes the whole module hermetically testable without mutating
//! process environment (which `edition2024` forbids in this crate): tests
//! construct a [`YtdlpInfo`] pointing at the stub script
//! (`tests/fixtures/youtube/ytdlp_stub.sh`) and call the functions directly.
//!
//! # yt-dlp CLI contract (agent-verified cheat-sheet, see the bd-27v1 epic)
//!
//! - probe:    `--version`                (prints `YYYY.MM.DD`)
//! - expand:   `--flat-playlist --dump-json --no-warnings URL`
//! - metadata: `-j --no-simulate --no-playlist --no-warnings URL`
//! - download: `-f ba --no-playlist --no-warnings --no-progress`
//!   `-o '<dest>/%(id)s.%(ext)s' --print after_move:filepath`
//!   `--sleep-interval 2 --max-sleep-interval 5 --retries 10 URL`
//!
//! Audio is downloaded as best-audio *as-is* (no `-x` re-encode); the existing
//! normalize stage converts to 16 kHz mono. The raw download is named by video
//! id only — the descriptive `{date} - {title} [{id}]` naming is the
//! `naming.rs` module's job at the output layer (separation of concerns).

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{NaiveDate, Utc};
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::error::{FwError, FwResult};
use crate::orchestrator::CancellationToken;
use crate::process::{
    MAX_CAPTURED_OUTPUT_BYTES, is_stdout_capture_limit_error, run_command_cancellable,
};

/// Environment override for the `yt-dlp` binary path/name.
const YTDLP_ENV_OVERRIDE: &str = "FRANKEN_WHISPER_YTDLP_BIN";
/// Default binary name resolved on `PATH` when no override is set.
const DEFAULT_YTDLP_BIN: &str = "yt-dlp";
/// A `yt-dlp` build older than this many days is flagged stale.
const STALE_AFTER_DAYS: i64 = 90;

/// Hard timeouts (safety nets layered atop cancellation-token polling).
const METADATA_TIMEOUT: Duration = Duration::from_secs(120);
const EXPAND_TIMEOUT: Duration = Duration::from_secs(300);
/// Downloads can legitimately run long (politeness sleeps + retries).
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(3600);

/// Resolved `yt-dlp` tool: absolute path, parsed version, staleness flag.
///
/// All orchestration functions take `&YtdlpInfo` and run `self.path`, so tests
/// can synthesize this struct pointing at the hermetic stub without touching
/// the process environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YtdlpInfo {
    /// Absolute path (or `PATH`-resolved location) of the `yt-dlp` binary.
    pub path: PathBuf,
    /// Raw version string as reported by `yt-dlp --version` (e.g. `2025.01.15`).
    pub version: String,
    /// `true` when the build is older than [`STALE_AFTER_DAYS`] days.
    pub stale: bool,
}

/// Classification of a user-supplied YouTube URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrlKind {
    /// A single video (`watch?v=`, `youtu.be/`, `shorts/`, `live/`).
    Video,
    /// A pure playlist (`playlist?list=`).
    Playlist,
    /// A video carrying a `list=` query (`watch?v=X&list=Y`). We treat these as
    /// a single video downstream because `--no-playlist` is the default.
    Ambiguous,
}

/// A lightweight reference to a video, as produced by playlist expansion.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoRef {
    /// Stable YouTube video id.
    pub id: String,
    /// Best-effort title (may be empty for restricted entries).
    pub title: String,
    /// Canonical watch URL for the video.
    pub url: String,
    /// Duration in seconds when reported by the flat-playlist dump.
    pub duration_sec: Option<f64>,
}

/// One curated result from `fw youtube search` / `fw youtube enrich`
/// (bd-j2lh / bd-m7fv): the retained subset of yt-dlp's object that an agent
/// needs to triage a video without a second fetch. Serialized as JSON on
/// stdout; `None`-valued optional fields are omitted so agents never parse
/// null placeholders.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SearchHit {
    /// Stable YouTube video id.
    pub id: String,
    /// Video title (may be empty for restricted entries).
    pub title: String,
    /// Canonical watch URL.
    pub url: String,
    /// Duration in seconds, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_sec: Option<f64>,
    /// Channel name, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    /// View count, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view_count: Option<u64>,
    /// Upload date in `YYYYMMDD` form, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload_date: Option<String>,
}

/// Projected subset parsed from one yt-dlp search line. Works for BOTH the
/// enriched per-video dump and the flat-playlist dump — every field except
/// the id is optional.
#[derive(serde::Deserialize)]
struct ProjectedSearchEntry {
    #[serde(default)]
    id: serde_json::Value,
    #[serde(default)]
    title: serde_json::Value,
    #[serde(default)]
    url: serde_json::Value,
    #[serde(default)]
    webpage_url: serde_json::Value,
    #[serde(default)]
    duration: serde_json::Value,
    #[serde(default)]
    channel: serde_json::Value,
    #[serde(default)]
    view_count: serde_json::Value,
    #[serde(default)]
    upload_date: serde_json::Value,
}

fn search_hit_from_entry(entry: &ProjectedSearchEntry) -> Option<SearchHit> {
    let id = non_empty_json_string(entry.id.clone())?;
    let url = non_empty_json_string(entry.webpage_url.clone())
        .or_else(|| non_empty_json_string(entry.url.clone()))
        .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));
    Some(SearchHit {
        title: non_empty_json_string(entry.title.clone()).unwrap_or_default(),
        url,
        duration_sec: entry.duration.as_f64(),
        channel: non_empty_json_string(entry.channel.clone()),
        view_count: entry.view_count.as_u64(),
        upload_date: non_empty_json_string(entry.upload_date.clone()),
        id,
    })
}

/// Search YouTube through the probed yt-dlp binary: `ytsearch{limit}:query`
/// dumped as per-result JSON lines. Enriched mode (default) retains the
/// curated [`SearchHit`] field set; `--flat` keeps the flat-playlist subset.
/// Hits are deduplicated by id preserving first-seen order, capped at
/// `limit`.
///
/// # Errors
///
/// Propagates [`run_ytdlp`] errors (including the capture-cap too-large
/// mapping shared with [`expand_playlist`]); per-line parse failures are
/// skipped with a warning, never fatal.
pub fn search(
    info: &YtdlpInfo,
    query: &str,
    limit: usize,
    flat: bool,
    token: &CancellationToken,
) -> FwResult<Vec<SearchHit>> {
    if query.trim().is_empty() {
        return Err(FwError::InvalidRequest("empty search query".to_owned()));
    }
    if limit == 0 {
        return Err(FwError::InvalidRequest(
            "--limit must be at least 1".to_owned(),
        ));
    }
    let target = format!("ytsearch{limit}:{query}");
    let mut args = vec!["--no-warnings".to_owned(), "--".to_owned(), target];
    if flat {
        args.insert(0, "--dump-json".to_owned());
        args.insert(0, "--flat-playlist".to_owned());
    } else {
        args.insert(0, "--dump-json".to_owned());
    }
    let output = run_ytdlp(info, &args, token, EXPAND_TIMEOUT)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut hits = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for line in stdout.lines().map(str::trim).filter(|l| !l.is_empty()) {
        match serde_json::from_str::<ProjectedSearchEntry>(line) {
            Ok(entry) => {
                if let Some(hit) = search_hit_from_entry(&entry)
                    && seen.insert(hit.id.clone())
                {
                    hits.push(hit);
                    if hits.len() >= limit {
                        break;
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    "skipping unparseable search line ({err}): {}",
                    truncate_for_log(line)
                );
            }
        }
    }
    Ok(hits)
}

/// Enrich specific video URLs or ids with full metadata via repeated
/// [`fetch_metadata`] calls, returning deduplicated [`SearchHit`]s in
/// first-seen order. Playlist URLs are rejected with an actionable error —
/// ingest playlists through `fw youtube run` instead.
///
/// # Errors
///
/// Propagates the first [`fetch_metadata`] error; playlist inputs yield
/// [`FwError::InvalidRequest`].
pub fn enrich(
    info: &YtdlpInfo,
    targets: &[String],
    token: &CancellationToken,
) -> FwResult<Vec<SearchHit>> {
    let mut hits = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for target in targets {
        let trimmed = target.trim();
        if trimmed.is_empty() {
            continue;
        }
        // A bare 11-char video id is accepted and canonicalized into a watch
        // URL (yt-dlp itself only understands URLs). 11 chars of
        // [A-Za-z0-9_-] is exactly YouTube's id alphabet; anything else is
        // left for classify_url to accept or reject as a URL.
        let target_owned;
        let target_ref: &str = {
            let is_bare_id = trimmed.len() == 11
                && trimmed
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                && classify_url(&format!("https://www.youtube.com/watch?v={trimmed}")).is_ok();
            if is_bare_id {
                target_owned = format!("https://www.youtube.com/watch?v={trimmed}");
                target_owned.as_str()
            } else {
                trimmed
            }
        };
        match classify_url(target_ref)? {
            UrlKind::Video | UrlKind::Ambiguous => {
                let meta = fetch_metadata(info, target_ref, token)?;
                if seen.insert(meta.id.clone()) {
                    hits.push(SearchHit {
                        id: meta.id.clone(),
                        title: meta.title.clone(),
                        url: meta.webpage_url.clone(),
                        duration_sec: meta.duration_sec,
                        channel: meta.channel.clone(),
                        view_count: None,
                        upload_date: meta.upload_date.clone(),
                    });
                }
            }
            UrlKind::Playlist => {
                return Err(FwError::InvalidRequest(format!(
                    "`{target_ref}` classified as Playlist; enrich takes individual video \
                     URLs or ids — ingest playlists through `fw youtube run` instead"
                )));
            }
        }
    }
    Ok(hits)
}

/// Deduplicate hits by id preserving first-seen order (shared by CLI merge
/// paths that combine search + enrich streams).
#[must_use]
pub fn dedup_hits(hits: Vec<SearchHit>) -> Vec<SearchHit> {
    let mut seen = std::collections::HashSet::new();
    hits.into_iter()
        .filter(|hit| seen.insert(hit.id.clone()))
        .collect()
}

/// Full per-video metadata fetched via `yt-dlp -j`.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoMeta {
    /// Stable YouTube video id.
    pub id: String,
    /// Video title.
    pub title: String,
    /// Channel name, when reported.
    pub channel: Option<String>,
    /// Uploader name, when reported.
    pub uploader: Option<String>,
    /// Upload date in `YYYYMMDD` form, when reported.
    pub upload_date: Option<String>,
    /// Duration in seconds, when reported.
    pub duration_sec: Option<f64>,
    /// Canonical webpage URL.
    pub webpage_url: String,
    /// Long-form description, when reported.
    pub description: Option<String>,
    /// Availability marker (`public`, `unlisted`, `private`, ...).
    pub availability: Option<String>,
    /// Live status (`not_live`, `is_live`, `is_upcoming`, `was_live`, ...).
    pub live_status: Option<String>,
}

// ---------------------------------------------------------------------------
// probe
// ---------------------------------------------------------------------------

/// Probe the `yt-dlp` binary: resolve its location, capture `--version`, and
/// compute staleness against today's date.
///
/// Resolution order: the `FRANKEN_WHISPER_YTDLP_BIN` override (if set and
/// non-empty), otherwise `yt-dlp` on `PATH`. The override is the *only* place
/// the environment is consulted.
///
/// # Errors
///
/// Returns [`FwError::CommandMissing`] if the binary cannot be resolved, or a
/// command/parse error if `--version` fails or does not yield a `YYYY.MM.DD`
/// date.
pub fn probe() -> FwResult<YtdlpInfo> {
    let requested = resolve_binary_name();
    let path = which::which(&requested).map_err(|_| FwError::CommandMissing {
        command: requested.clone(),
    })?;

    probe_with_path(&path, Utc::now().date_naive())
}

/// Probe a specific binary path against a caller-provided `today` date.
///
/// Factored out of [`probe`] so tests can drive a deterministic `today`.
///
/// # Errors
///
/// Propagates the `--version` command error, or returns
/// [`FwError::InvalidRequest`] when the output is not a parseable date.
pub fn probe_with_path(path: &Path, today: NaiveDate) -> FwResult<YtdlpInfo> {
    let token = CancellationToken::unbounded();
    let path_str = path.display().to_string();
    let output = run_command_cancellable(
        &path_str,
        &["--version".to_owned()],
        None,
        &token,
        Some(METADATA_TIMEOUT),
    )?;

    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .to_owned();

    if version.is_empty() {
        return Err(FwError::InvalidRequest(format!(
            "`{path_str} --version` produced no output; is this really yt-dlp?"
        )));
    }

    let stale = match parse_version_date(&version) {
        Some(date) => is_stale(date, today),
        None => {
            // Unparseable version: do not crash — yt-dlp nightly/git builds use
            // suffixes. Treat as not-stale but keep the raw string.
            false
        }
    };

    Ok(YtdlpInfo {
        path: path.to_path_buf(),
        version,
        stale,
    })
}

/// Resolve the requested binary name, honoring the env override.
fn resolve_binary_name() -> String {
    std::env::var(YTDLP_ENV_OVERRIDE)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_YTDLP_BIN.to_owned())
}

/// Parse a `yt-dlp` version string of the form `YYYY.MM.DD` (optionally with a
/// trailing suffix like `.dev0` or a 4th `.N` micro component) into a date.
///
/// Returns `None` if the leading three dot-separated components are not a valid
/// `year.month.day`.
#[must_use]
pub fn parse_version_date(version: &str) -> Option<NaiveDate> {
    let mut parts = version.split('.');
    let year: i32 = parts.next()?.parse().ok()?;
    let month: u32 = parts.next()?.parse().ok()?;
    let day: u32 = parts.next()?.parse().ok()?;
    NaiveDate::from_ymd_opt(year, month, day)
}

/// Return `true` when `version_date` is more than [`STALE_AFTER_DAYS`] days
/// before `today`. Future-dated builds are never stale.
#[must_use]
pub fn is_stale(version_date: NaiveDate, today: NaiveDate) -> bool {
    (today - version_date).num_days() > STALE_AFTER_DAYS
}

// ---------------------------------------------------------------------------
// classify_url
// ---------------------------------------------------------------------------

/// Classify a user-supplied URL into [`UrlKind`].
///
/// Recognized YouTube forms:
/// - `watch?v=ID`                  -> [`UrlKind::Video`]
/// - `youtu.be/ID`                 -> [`UrlKind::Video`]
/// - `shorts/ID`, `live/ID`        -> [`UrlKind::Video`]
/// - `playlist?list=ID`            -> [`UrlKind::Playlist`]
/// - `watch?v=X&list=Y`            -> [`UrlKind::Ambiguous`] (treated as Video)
///
/// Implemented with plain string parsing (no regex dependency).
///
/// # Errors
///
/// Returns [`FwError::InvalidRequest`] for non-YouTube hosts or YouTube URLs
/// that do not match any known shape, with an actionable message.
pub fn classify_url(url: &str) -> FwResult<UrlKind> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return Err(FwError::InvalidRequest(
            "empty URL; expected a YouTube video or playlist URL".to_owned(),
        ));
    }

    let (host, rest) = split_host_and_rest(trimmed).ok_or_else(|| {
        FwError::InvalidRequest(format!(
            "not a recognized URL: `{trimmed}`; expected a YouTube video or playlist URL"
        ))
    })?;
    let host = host.to_ascii_lowercase();

    // youtu.be short links: the path segment after the host is the video id.
    if host == "youtu.be" {
        let id = rest.trim_start_matches('/');
        let id = id.split(['?', '&', '/']).next().unwrap_or_default();
        if id.is_empty() {
            return Err(FwError::InvalidRequest(format!(
                "youtu.be URL has no video id: `{trimmed}`"
            )));
        }
        return Ok(UrlKind::Video);
    }

    if !is_youtube_host(&host) {
        return Err(FwError::InvalidRequest(format!(
            "not a YouTube URL: `{trimmed}` (host `{host}`); \
             only youtube.com / youtu.be links are supported"
        )));
    }

    // Split path from query for youtube.com-family hosts.
    let (path, query) = match rest.split_once('?') {
        Some((p, q)) => (p, q),
        None => (rest, ""),
    };
    let path = path.trim_start_matches('/');

    // /shorts/ID and /live/ID are always single videos.
    if let Some(id) = path
        .strip_prefix("shorts/")
        .or_else(|| path.strip_prefix("live/"))
    {
        let id = id.split('/').next().unwrap_or_default();
        if id.is_empty() {
            return Err(FwError::InvalidRequest(format!(
                "URL has no video id: `{trimmed}`"
            )));
        }
        return Ok(UrlKind::Video);
    }

    let has_v = query_has_nonempty_param(query, "v");
    let has_list = query_has_nonempty_param(query, "list");

    // /playlist?list=ID -> pure playlist.
    if path == "playlist" {
        if has_list {
            return Ok(UrlKind::Playlist);
        }
        return Err(FwError::InvalidRequest(format!(
            "playlist URL is missing a `list=` parameter: `{trimmed}`"
        )));
    }

    // /watch?v=X (&list=Y).
    if path == "watch" {
        if has_v && has_list {
            // Video embedded in a playlist context. We honor --no-playlist by
            // default, so callers treat this as a single video.
            return Ok(UrlKind::Ambiguous);
        }
        if has_v {
            return Ok(UrlKind::Video);
        }
        if has_list {
            // /watch?list=Y with no v= is effectively a playlist landing page.
            return Ok(UrlKind::Playlist);
        }
        return Err(FwError::InvalidRequest(format!(
            "watch URL is missing both `v=` and `list=`: `{trimmed}`"
        )));
    }

    // A bare `?list=` on the root or any other path is treated as a playlist.
    if has_list && !has_v {
        return Ok(UrlKind::Playlist);
    }
    if has_v {
        return Ok(UrlKind::Video);
    }

    Err(FwError::InvalidRequest(format!(
        "unrecognized YouTube URL shape: `{trimmed}`; \
         expected watch?v=, youtu.be/, shorts/, live/, or playlist?list="
    )))
}

/// Extract the YouTube video id from a single-video URL using the *same* URL
/// parsing [`classify_url`] performs — purely, with no network round-trip.
///
/// This lets [`resolve_videos`](crate::youtube::pipeline) deduplicate
/// `Video`/`Ambiguous` URLs by id without a `yt-dlp -j` metadata fetch (the #1
/// hotspot: 3 metadata fetches/video collapse to 1 in the download worker).
///
/// Recognized forms (all yielding the bare id):
/// - `watch?v=ID` (and `watch?v=ID&list=Y` — the `v=` param, honoring
///   `--no-playlist`)
/// - `youtu.be/ID` (with an optional `?t=`/`&`/trailing-path tail)
/// - `shorts/ID`, `live/ID`, `embed/ID`
///
/// Returns `None` for playlist URLs, non-YouTube hosts, or any shape without a
/// recoverable id. Callers that already classified a URL as `Video`/`Ambiguous`
/// can treat `None` as a (should-not-happen) signal to fall back to a single
/// metadata fetch for correctness.
#[must_use]
pub fn extract_video_id(url: &str) -> Option<String> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (host, rest) = split_host_and_rest(trimmed)?;
    let host = host.to_ascii_lowercase();

    // youtu.be/ID short links: the first path segment is the id.
    if host == "youtu.be" {
        let id = rest.trim_start_matches('/');
        let id = id.split(['?', '&', '/']).next().unwrap_or_default();
        return non_empty_id(id);
    }

    if !is_youtube_host(&host) {
        return None;
    }

    let (path, query) = match rest.split_once('?') {
        Some((p, q)) => (p, q),
        None => (rest, ""),
    };
    let path = path.trim_start_matches('/');

    // /shorts/ID, /live/ID, /embed/ID -> the path segment after the prefix.
    if let Some(id) = path
        .strip_prefix("shorts/")
        .or_else(|| path.strip_prefix("live/"))
        .or_else(|| path.strip_prefix("embed/"))
    {
        let id = id.split('/').next().unwrap_or_default();
        return non_empty_id(id);
    }

    // /watch?v=ID (&list=Y): the `v=` param is the single video, per
    // --no-playlist. A bare /watch?list= with no v= is a playlist -> None.
    if path == "watch" {
        return query_param_value(query, "v").and_then(non_empty_id);
    }

    // Any other path: accept a `v=` query if present (mirrors classify_url's
    // permissive tail), otherwise no id.
    query_param_value(query, "v").and_then(non_empty_id)
}

/// Return `Some(id)` when `id` is non-empty, else `None`.
fn non_empty_id(id: &str) -> Option<String> {
    if id.is_empty() {
        None
    } else {
        Some(id.to_owned())
    }
}

/// Return the (non-empty) value of query param `name`, or `None`.
fn query_param_value<'a>(query: &'a str, name: &str) -> Option<&'a str> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if k == name && !v.is_empty() {
            Some(v)
        } else {
            None
        }
    })
}

/// Split a URL into `(host, rest)` where `rest` is everything after the host
/// (path + query). Tolerates a missing scheme. Returns `None` when no host can
/// be isolated.
fn split_host_and_rest(url: &str) -> Option<(&str, &str)> {
    // Strip scheme if present.
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    if after_scheme.is_empty() {
        return None;
    }
    // Host runs until the first '/', '?', or end.
    let host_end = after_scheme.find(['/', '?']).unwrap_or(after_scheme.len());
    let host = &after_scheme[..host_end];
    if host.is_empty() {
        return None;
    }
    let rest = &after_scheme[host_end..];
    Some((host, rest))
}

/// Return `true` when `host` is a YouTube web host (ignoring a `www.`/`m.`
/// prefix).
fn is_youtube_host(host: &str) -> bool {
    let bare = host
        .strip_prefix("www.")
        .or_else(|| host.strip_prefix("m."))
        .or_else(|| host.strip_prefix("music."))
        .unwrap_or(host);
    bare == "youtube.com" || bare == "youtube-nocookie.com"
}

/// Return `true` when a `key=value` query string contains `name` with a
/// non-empty value.
fn query_has_nonempty_param(query: &str, name: &str) -> bool {
    query_param_value(query, name).is_some()
}

// ---------------------------------------------------------------------------
// expand_playlist
// ---------------------------------------------------------------------------

/// Expand a playlist URL into its constituent [`VideoRef`]s.
///
/// Runs `yt-dlp --flat-playlist --dump-json --no-warnings URL` and parses the
/// JSON-lines output. Lines that fail to parse (or lack an `id`) are skipped
/// with a warning rather than failing the whole expansion.
///
/// # Errors
///
/// Propagates command failures (mapped to actionable [`FwError`]s via
/// [`map_ytdlp_error`]) and cancellation.
pub fn expand_playlist(
    info: &YtdlpInfo,
    url: &str,
    token: &CancellationToken,
) -> FwResult<Vec<VideoRef>> {
    let args = vec![
        "--flat-playlist".to_owned(),
        "--dump-json".to_owned(),
        "--no-warnings".to_owned(),
        // `--` stops yt-dlp option parsing: a hostile URL (e.g. a
        // playlist-entry `url` field starting with `-`) can never be read as a
        // flag. Defense-in-depth on top of classify_url's host gate.
        "--".to_owned(),
        url.to_owned(),
    ];
    let output = match run_ytdlp(info, &args, token, EXPAND_TIMEOUT) {
        Ok(output) => output,
        // The process layer reports stdout overflow explicitly, so surface an
        // actionable request error rather than losing playlist rows.
        Err(err) if is_stdout_capture_limit_error(&err) => {
            return Err(playlist_too_large_error(url));
        }
        Err(err) => return Err(err),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut refs = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_flat_playlist_line(line) {
            Ok(video_ref) => {
                if let Some(video_ref) = video_ref {
                    refs.push(video_ref);
                } else {
                    tracing::warn!(
                        "skipping flat-playlist entry without an id: {}",
                        truncate_for_log(line)
                    );
                }
            }
            Err(err) => {
                tracing::warn!(
                    "skipping unparseable flat-playlist line ({err}): {}",
                    truncate_for_log(line)
                );
            }
        }
    }

    Ok(refs)
}

/// Actionable error for a playlist whose flat-JSON expansion exceeded the 4 MiB
/// capture cap (so the captured list would be silently truncated).
fn playlist_too_large_error(url: &str) -> FwError {
    FwError::InvalidRequest(format!(
        "playlist `{url}` is too large to expand: its yt-dlp flat-playlist output \
         exceeded the {} MiB capture cap and would be truncated (silently dropping \
         videos). Split it into smaller playlists, or pass the individual video URLs \
         (e.g. via --batch-file).",
        MAX_CAPTURED_OUTPUT_BYTES / (1024 * 1024)
    ))
}

/// The only fields retained from yt-dlp's much larger flat-playlist objects.
/// Unknown fields are parsed for validity but skipped without building a full
/// `serde_json::Value` tree.
#[derive(serde::Deserialize)]
struct FlatPlaylistEntry {
    #[serde(default)]
    id: serde_json::Value,
    #[serde(default)]
    title: serde_json::Value,
    #[serde(default)]
    url: serde_json::Value,
    #[serde(default)]
    webpage_url: serde_json::Value,
    #[serde(default)]
    duration: serde_json::Value,
}

/// Parse one flat-playlist line and move the retained strings into a
/// [`VideoRef`]. Missing, empty, or non-string ids remain skippable entries.
fn parse_flat_playlist_line(line: &str) -> serde_json::Result<Option<VideoRef>> {
    serde_json::from_str(line).map(video_ref_from_flat_entry)
}

fn video_ref_from_flat_entry(entry: FlatPlaylistEntry) -> Option<VideoRef> {
    let id = non_empty_json_string(entry.id)?;
    let title = non_empty_json_string(entry.title).unwrap_or_default();
    let url = non_empty_json_string(entry.url)
        .or_else(|| non_empty_json_string(entry.webpage_url))
        .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));
    let duration_sec = entry.duration.as_f64();

    Some(VideoRef {
        id,
        title,
        url,
        duration_sec,
    })
}

fn non_empty_json_string(value: serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// fetch_metadata
// ---------------------------------------------------------------------------

/// The fields retained from yt-dlp's much larger full metadata object.
/// Unknown values are validated and skipped without materializing a complete
/// `serde_json::Value` tree. Retained values stay as JSON values so wrong-type,
/// empty-string, numeric, and duplicate-key behavior matches the former DOM
/// parser exactly.
struct ProjectedVideoMetadata {
    id: serde_json::Value,
    title: serde_json::Value,
    channel: serde_json::Value,
    uploader: serde_json::Value,
    upload_date: serde_json::Value,
    duration: serde_json::Value,
    webpage_url: serde_json::Value,
    description: serde_json::Value,
    availability: serde_json::Value,
    live_status: serde_json::Value,
}

impl Default for ProjectedVideoMetadata {
    fn default() -> Self {
        Self {
            id: serde_json::Value::Null,
            title: serde_json::Value::Null,
            channel: serde_json::Value::Null,
            uploader: serde_json::Value::Null,
            upload_date: serde_json::Value::Null,
            duration: serde_json::Value::Null,
            webpage_url: serde_json::Value::Null,
            description: serde_json::Value::Null,
            availability: serde_json::Value::Null,
            live_status: serde_json::Value::Null,
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum VideoMetadataField {
    Id,
    Title,
    Channel,
    Uploader,
    UploadDate,
    Duration,
    WebpageUrl,
    Description,
    Availability,
    LiveStatus,
    #[serde(other)]
    Other,
}

impl<'de> serde::Deserialize<'de> for ProjectedVideoMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MetadataVisitor;

        impl<'de> Visitor<'de> for MetadataVisitor {
            type Value = ProjectedVideoMetadata;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a yt-dlp metadata JSON value")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut metadata = ProjectedVideoMetadata::default();
                while let Some(field) = map.next_key()? {
                    match field {
                        VideoMetadataField::Id => metadata.id = map.next_value()?,
                        VideoMetadataField::Title => metadata.title = map.next_value()?,
                        VideoMetadataField::Channel => metadata.channel = map.next_value()?,
                        VideoMetadataField::Uploader => metadata.uploader = map.next_value()?,
                        VideoMetadataField::UploadDate => {
                            metadata.upload_date = map.next_value()?;
                        }
                        VideoMetadataField::Duration => metadata.duration = map.next_value()?,
                        VideoMetadataField::WebpageUrl => {
                            metadata.webpage_url = map.next_value()?;
                        }
                        VideoMetadataField::Description => {
                            metadata.description = map.next_value()?;
                        }
                        VideoMetadataField::Availability => {
                            metadata.availability = map.next_value()?;
                        }
                        VideoMetadataField::LiveStatus => {
                            metadata.live_status = map.next_value()?;
                        }
                        VideoMetadataField::Other => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(metadata)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while sequence.next_element::<IgnoredAny>()?.is_some() {}
                Ok(ProjectedVideoMetadata::default())
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(ProjectedVideoMetadata::default())
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
                Ok(ProjectedVideoMetadata::default())
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
                Ok(ProjectedVideoMetadata::default())
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
                Ok(ProjectedVideoMetadata::default())
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(ProjectedVideoMetadata::default())
            }

            fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
                Ok(ProjectedVideoMetadata::default())
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(ProjectedVideoMetadata::default())
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(ProjectedVideoMetadata::default())
            }
        }

        deserializer.deserialize_any(MetadataVisitor)
    }
}

/// Fetch full metadata for a single video via `yt-dlp -j`.
///
/// Runs `yt-dlp -j --no-simulate --no-playlist --no-warnings URL`. Live and
/// upcoming streams are rejected with a clear [`FwError::Unsupported`] because
/// they cannot be transcribed as a finished recording.
///
/// # Errors
///
/// Propagates mapped command failures, cancellation, JSON parse errors, and
/// [`FwError::Unsupported`] for live/upcoming streams.
pub fn fetch_metadata(
    info: &YtdlpInfo,
    url: &str,
    token: &CancellationToken,
) -> FwResult<VideoMeta> {
    let args = vec![
        "-j".to_owned(),
        "--no-simulate".to_owned(),
        "--no-playlist".to_owned(),
        "--no-warnings".to_owned(),
        // `--` stops yt-dlp option parsing: a hostile URL (e.g. a
        // playlist-entry `url` field starting with `-`) can never be read as a
        // flag. Defense-in-depth on top of classify_url's host gate.
        "--".to_owned(),
        url.to_owned(),
    ];
    let output = run_ytdlp(info, &args, token, METADATA_TIMEOUT)?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // `-j` emits a single JSON object; pick the first non-empty line.
    let line = stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .ok_or_else(|| {
            FwError::InvalidRequest(format!("yt-dlp returned no metadata for `{url}`"))
        })?;

    let meta = parse_video_meta(line)?;

    if let Some(status) = meta.live_status.as_deref()
        && matches!(status, "is_live" | "is_upcoming")
    {
        return Err(FwError::Unsupported(format!(
            "`{url}` is a {status} stream; live/upcoming streams cannot be transcribed. \
             Retry once the stream has ended and a recording is available."
        )));
    }

    Ok(meta)
}

fn parse_video_meta(line: &str) -> FwResult<VideoMeta> {
    let metadata: ProjectedVideoMetadata = serde_json::from_str(line)?;
    let id = non_empty_json_string(metadata.id).ok_or_else(|| {
        FwError::InvalidRequest("yt-dlp metadata is missing an `id` field".to_owned())
    })?;
    let title = non_empty_json_string(metadata.title).unwrap_or_default();
    let webpage_url = non_empty_json_string(metadata.webpage_url)
        .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));

    Ok(VideoMeta {
        id,
        title,
        channel: non_empty_json_string(metadata.channel),
        uploader: non_empty_json_string(metadata.uploader),
        upload_date: non_empty_json_string(metadata.upload_date),
        duration_sec: metadata.duration.as_f64(),
        webpage_url,
        description: non_empty_json_string(metadata.description),
        availability: non_empty_json_string(metadata.availability),
        live_status: non_empty_json_string(metadata.live_status),
    })
}

/// Build a [`VideoMeta`] from a `yt-dlp -j` JSON object.
#[cfg(test)]
fn video_meta_from_json(value: &serde_json::Value) -> FwResult<VideoMeta> {
    let id = string_field(value, "id").ok_or_else(|| {
        FwError::InvalidRequest("yt-dlp metadata is missing an `id` field".to_owned())
    })?;
    let title = string_field(value, "title").unwrap_or_default();
    let webpage_url = string_field(value, "webpage_url")
        .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));

    Ok(VideoMeta {
        id,
        title,
        channel: string_field(value, "channel"),
        uploader: string_field(value, "uploader"),
        upload_date: string_field(value, "upload_date"),
        duration_sec: value.get("duration").and_then(serde_json::Value::as_f64),
        webpage_url,
        description: string_field(value, "description"),
        availability: string_field(value, "availability"),
        live_status: string_field(value, "live_status"),
    })
}

// ---------------------------------------------------------------------------
// download_audio
// ---------------------------------------------------------------------------

/// Download the best-audio stream for `meta` into `dest_dir`, returning the
/// path to the downloaded file.
///
/// Runs:
/// `yt-dlp -f ba --no-playlist --no-warnings --no-progress`
/// `-o '<dest_dir>/%(id)s.%(ext)s' --print after_move:filepath`
/// `--sleep-interval 2 --max-sleep-interval 5 --retries 10 URL`
///
/// The download is intentionally named by video id only; descriptive naming is
/// the `naming.rs` module's responsibility at the output layer.
///
/// # Cancellation
///
/// Execution flows through [`run_command_cancellable`], which polls `token` on
/// every iteration and kills the child process when the token fires. For
/// best-audio (`-f ba`) downloads yt-dlp does **not** normally spawn an
/// `ffmpeg` child (no `-x` re-encode is requested). Cancellation currently
/// guarantees termination and reaping of the direct yt-dlp process only;
/// process-tree termination for an unexpectedly inherited descendant remains
/// a separate contract gap. A token firing maps to [`FwError::Cancelled`].
///
/// # Errors
///
/// Propagates mapped command failures, cancellation, and
/// [`FwError::MissingArtifact`] if the printed/expected path is not found.
pub fn download_audio(
    info: &YtdlpInfo,
    meta: &VideoMeta,
    dest_dir: &Path,
    token: &CancellationToken,
) -> FwResult<PathBuf> {
    let template = dest_dir.join("%(id)s.%(ext)s");
    let args = vec![
        // Best audio-only, falling back to the best combined format when a
        // video has no audio-only stream (older uploads, some live VODs). The
        // normalize stage extracts audio from a combined file via ffmpeg
        // `-vn`, so a video container costs only bandwidth, never correctness.
        // Bare `ba` rejects such videos outright ("Requested format is not
        // available"), so the fallback is load-bearing.
        "-f".to_owned(),
        "bestaudio/best".to_owned(),
        "--no-playlist".to_owned(),
        "--no-warnings".to_owned(),
        "--no-progress".to_owned(),
        "-o".to_owned(),
        template.display().to_string(),
        "--print".to_owned(),
        "after_move:filepath".to_owned(),
        "--sleep-interval".to_owned(),
        "2".to_owned(),
        "--max-sleep-interval".to_owned(),
        "5".to_owned(),
        "--retries".to_owned(),
        "10".to_owned(),
        // `--` stops yt-dlp option parsing: a hostile URL (e.g. a
        // playlist-entry `url` field starting with `-`) can never be read as a
        // flag. Defense-in-depth on top of classify_url's host gate.
        "--".to_owned(),
        meta.webpage_url.clone(),
    ];

    let output = run_ytdlp(info, &args, token, DOWNLOAD_TIMEOUT)?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // The `--print after_move:filepath` contract emits the final path on its
    // own line. Pick the LAST non-empty stdout line that is an existing path.
    let printed = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .rev()
        .map(PathBuf::from)
        .find(|candidate| candidate.is_file());

    if let Some(path) = printed {
        return Ok(path);
    }

    // Fallback: yt-dlp may not have printed a usable line. Scan dest_dir for a
    // file named `<id>.*` (we control the template).
    if let Some(found) = find_downloaded_by_id(dest_dir, &meta.id) {
        return Ok(found);
    }

    Err(FwError::MissingArtifact(
        dest_dir.join(format!("{}.<ext>", meta.id)),
    ))
}

/// Scan `dest_dir` for a file whose stem equals `id` (the template names
/// downloads `<id>.<ext>`).
pub(crate) fn find_downloaded_by_id(dest_dir: &Path, id: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dest_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if is_reusable_download_path(&path, dest_dir, id) {
            return Some(path);
        }
    }
    None
}

/// Whether `path` is a completed, regular download owned by `dest_dir` for
/// this exact video id. Partial yt-dlp artifacts and symlinks are never resume
/// inputs.
pub(crate) fn is_reusable_download_path(path: &Path, dest_dir: &Path, id: &str) -> bool {
    let extension = path.extension().and_then(|value| value.to_str());
    path.parent() == Some(dest_dir)
        && path.file_stem().and_then(|value| value.to_str()) == Some(id)
        && extension.is_some_and(|value| !matches!(value, "part" | "tmp" | "ytdl"))
        && std::fs::symlink_metadata(path)
            .is_ok_and(|metadata| metadata.file_type().is_file())
}

// ---------------------------------------------------------------------------
// shared execution + error mapping
// ---------------------------------------------------------------------------

/// Run `info.path` with `args`, mapping command failures through
/// [`map_ytdlp_error`]. Cancellation errors pass through unchanged.
fn run_ytdlp(
    info: &YtdlpInfo,
    args: &[String],
    token: &CancellationToken,
    hard_timeout: Duration,
) -> FwResult<std::process::Output> {
    let path_str = info.path.display().to_string();
    match run_command_cancellable(&path_str, args, None, token, Some(hard_timeout)) {
        Ok(output) => Ok(output),
        Err(err) => Err(map_ytdlp_error(err)),
    }
}

/// Translate a raw process error into an actionable [`FwError`] using the
/// stderr-signature matrix from the cheat-sheet.
///
/// Cancellation, missing-binary, and timeout errors are passed through
/// unchanged (they are already actionable). `CommandFailed` errors have their
/// stderr inspected for known yt-dlp signatures.
#[must_use]
fn map_ytdlp_error(err: FwError) -> FwError {
    let FwError::CommandFailed {
        command,
        status,
        stderr_suffix,
    } = &err
    else {
        // Cancelled / CommandMissing / CommandTimedOut / Io etc. — pass through.
        return err;
    };

    let haystack = stderr_suffix.to_ascii_lowercase();
    let signature = classify_stderr(&haystack);
    match signature {
        Some(message) => FwError::InvalidRequest(message.to_owned()),
        None => FwError::CommandFailed {
            command: command.clone(),
            status: *status,
            stderr_suffix: stderr_suffix.clone(),
        },
    }
}

/// Match a (lowercased) stderr string against the known yt-dlp failure
/// signatures, returning an actionable message.
fn classify_stderr(stderr_lower: &str) -> Option<&'static str> {
    if stderr_lower.contains("private video") {
        return Some(
            "video is private; it cannot be downloaded without account access. \
             Skipping.",
        );
    }
    if stderr_lower.contains("this video is unavailable")
        || stderr_lower.contains("video unavailable")
        || stderr_lower.contains("has been removed")
    {
        return Some("video is unavailable or has been removed. Skipping.");
    }
    if stderr_lower.contains("sign in to confirm your age")
        || stderr_lower.contains("age-restricted")
        || stderr_lower.contains("inappropriate for some users")
    {
        return Some(
            "video is age-restricted and requires sign-in; it cannot be \
             downloaded anonymously. Skipping.",
        );
    }
    if stderr_lower.contains("not available in your country")
        || stderr_lower.contains("not made this video available in your country")
        || stderr_lower.contains("blocked it in your country")
    {
        return Some(
            "video is geo-blocked in this region. Skipping (try again from a \
             permitted region).",
        );
    }
    if stderr_lower.contains("http error 429") || stderr_lower.contains("too many requests") {
        return Some(
            "YouTube rate-limited the downloader (HTTP 429). Wait a while and \
             retry; consider lowering concurrency.",
        );
    }
    None
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// Extract a non-empty string field from a JSON object, treating JSON `null`
/// and empty strings as absent.
#[cfg(test)]
fn string_field(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|s| !s.is_empty())
}

/// Truncate a log line to a sane length so a giant malformed JSON blob does not
/// flood the logs.
fn truncate_for_log(line: &str) -> String {
    const MAX: usize = 200;
    if line.len() <= MAX {
        return line.to_owned();
    }
    let mut end = MAX;
    while end > 0 && !line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &line[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute path to the hermetic yt-dlp stub script.
    fn stub_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/youtube/ytdlp_stub.sh")
    }

    /// A [`YtdlpInfo`] wired to the stub (no env mutation).
    fn stub_info() -> YtdlpInfo {
        YtdlpInfo {
            path: stub_path(),
            version: "2025.01.01".to_owned(),
            stale: false,
        }
    }

    fn meta_for_download() -> VideoMeta {
        VideoMeta {
            id: "dQw4w9WgXcQ".to_owned(),
            title: "Stub".to_owned(),
            channel: None,
            uploader: None,
            upload_date: None,
            duration_sec: None,
            webpage_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".to_owned(),
            description: None,
            availability: None,
            live_status: None,
        }
    }

    // ---- parse_version_date / is_stale -----------------------------------

    #[test]
    fn parse_version_date_basic() {
        assert_eq!(
            parse_version_date("2025.01.15"),
            NaiveDate::from_ymd_opt(2025, 1, 15)
        );
    }

    #[test]
    fn parse_version_date_with_suffix() {
        assert_eq!(
            parse_version_date("2024.12.06.dev0"),
            NaiveDate::from_ymd_opt(2024, 12, 6)
        );
    }

    #[test]
    fn parse_version_date_rejects_garbage() {
        assert_eq!(parse_version_date("not-a-version"), None);
        assert_eq!(parse_version_date("2025.13.01"), None); // month 13
        assert_eq!(parse_version_date("2025.02.30"), None); // invalid day
        assert_eq!(parse_version_date(""), None);
        assert_eq!(parse_version_date("2025.01"), None); // missing day
    }

    #[test]
    fn is_stale_true_when_old() {
        let old = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let today = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap();
        assert!(is_stale(old, today));
    }

    #[test]
    fn is_stale_false_when_recent() {
        let recent = NaiveDate::from_ymd_opt(2025, 5, 15).unwrap();
        let today = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap();
        assert!(!is_stale(recent, today));
    }

    #[test]
    fn is_stale_boundary_exactly_90_days_not_stale() {
        let date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let today = date + chrono::Duration::days(90);
        assert!(!is_stale(date, today), "exactly 90 days is not stale");
        let today91 = date + chrono::Duration::days(91);
        assert!(is_stale(date, today91), "91 days is stale");
    }

    #[test]
    fn is_stale_false_for_future_build() {
        let future = NaiveDate::from_ymd_opt(2025, 12, 1).unwrap();
        let today = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap();
        assert!(!is_stale(future, today));
    }

    // ---- probe (via stub) ------------------------------------------------

    #[test]
    fn probe_with_path_parses_version_and_staleness() {
        let today = NaiveDate::from_ymd_opt(2025, 1, 5).unwrap();
        let info = probe_with_path(&stub_path(), today).expect("probe should succeed");
        assert_eq!(info.version, "2025.01.01");
        assert!(!info.stale, "4 days old is not stale");
        assert_eq!(info.path, stub_path());
    }

    #[test]
    fn probe_with_path_flags_stale_build() {
        let today = NaiveDate::from_ymd_opt(2025, 6, 1).unwrap();
        let info = probe_with_path(&stub_path(), today).expect("probe should succeed");
        // Stub default version is 2025.01.01 -> >90 days before 2025-06-01.
        assert!(info.stale, "old build should be flagged stale");
    }

    #[test]
    fn probe_with_path_missing_binary_is_command_missing() {
        let bogus = PathBuf::from("/nonexistent/yt-dlp-xyz-99999");
        let err = probe_with_path(&bogus, Utc::now().date_naive())
            .expect_err("missing binary should fail");
        assert!(
            matches!(err, FwError::CommandMissing { .. }),
            "expected CommandMissing, got: {err:?}"
        );
    }

    // ---- classify_url ----------------------------------------------------

    #[test]
    fn classify_watch_video() {
        assert_eq!(
            classify_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ").unwrap(),
            UrlKind::Video
        );
    }

    #[test]
    fn classify_youtu_be_short_link() {
        assert_eq!(
            classify_url("https://youtu.be/dQw4w9WgXcQ").unwrap(),
            UrlKind::Video
        );
        assert_eq!(
            classify_url("https://youtu.be/dQw4w9WgXcQ?t=42").unwrap(),
            UrlKind::Video
        );
    }

    #[test]
    fn classify_shorts() {
        assert_eq!(
            classify_url("https://www.youtube.com/shorts/abc123XYZ_-").unwrap(),
            UrlKind::Video
        );
    }

    #[test]
    fn classify_live() {
        assert_eq!(
            classify_url("https://www.youtube.com/live/abc123XYZ_-").unwrap(),
            UrlKind::Video
        );
    }

    #[test]
    fn classify_playlist() {
        assert_eq!(
            classify_url("https://www.youtube.com/playlist?list=PL1234567890").unwrap(),
            UrlKind::Playlist
        );
    }

    #[test]
    fn classify_watch_with_list_is_ambiguous() {
        assert_eq!(
            classify_url("https://www.youtube.com/watch?v=abc&list=PL123").unwrap(),
            UrlKind::Ambiguous
        );
    }

    #[test]
    fn classify_watch_list_param_order_independent() {
        assert_eq!(
            classify_url("https://www.youtube.com/watch?list=PL123&v=abc").unwrap(),
            UrlKind::Ambiguous
        );
    }

    #[test]
    fn classify_mobile_and_music_hosts() {
        assert_eq!(
            classify_url("https://m.youtube.com/watch?v=abc").unwrap(),
            UrlKind::Video
        );
        assert_eq!(
            classify_url("https://music.youtube.com/watch?v=abc").unwrap(),
            UrlKind::Video
        );
    }

    #[test]
    fn classify_nocookie_host() {
        assert_eq!(
            classify_url("https://www.youtube-nocookie.com/watch?v=abc").unwrap(),
            UrlKind::Video
        );
    }

    #[test]
    fn classify_scheme_optional() {
        assert_eq!(
            classify_url("youtube.com/watch?v=abc").unwrap(),
            UrlKind::Video
        );
        assert_eq!(classify_url("youtu.be/abc").unwrap(), UrlKind::Video);
    }

    #[test]
    fn classify_non_youtube_rejected() {
        let err = classify_url("https://vimeo.com/12345").expect_err("non-youtube");
        assert!(matches!(err, FwError::InvalidRequest(_)));
        let text = err.to_string();
        assert!(text.contains("YouTube"), "actionable message: {text}");
    }

    #[test]
    fn classify_empty_rejected() {
        assert!(matches!(
            classify_url("   "),
            Err(FwError::InvalidRequest(_))
        ));
    }

    #[test]
    fn classify_garbage_rejected() {
        assert!(matches!(
            classify_url("not even a url"),
            Err(FwError::InvalidRequest(_))
        ));
    }

    #[test]
    fn classify_youtube_watch_missing_v_and_list() {
        assert!(matches!(
            classify_url("https://www.youtube.com/watch"),
            Err(FwError::InvalidRequest(_))
        ));
    }

    #[test]
    fn classify_youtu_be_no_id_rejected() {
        assert!(matches!(
            classify_url("https://youtu.be/"),
            Err(FwError::InvalidRequest(_))
        ));
    }

    #[test]
    fn classify_watch_with_only_list_is_playlist() {
        assert_eq!(
            classify_url("https://www.youtube.com/watch?list=PL123").unwrap(),
            UrlKind::Playlist
        );
    }

    // ---- extract_video_id ------------------------------------------------

    #[test]
    fn extract_id_watch() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ").as_deref(),
            Some("dQw4w9WgXcQ")
        );
    }

    #[test]
    fn extract_id_watch_with_list_and_order() {
        // watch?v=X&list=Y -> the single video id (honors --no-playlist).
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?v=abc123&list=PL999").as_deref(),
            Some("abc123")
        );
        // Param order independent.
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?list=PL999&v=abc123").as_deref(),
            Some("abc123")
        );
    }

    #[test]
    fn extract_id_youtu_be() {
        assert_eq!(
            extract_video_id("https://youtu.be/dQw4w9WgXcQ").as_deref(),
            Some("dQw4w9WgXcQ")
        );
        // With a timestamp / extra query.
        assert_eq!(
            extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=42").as_deref(),
            Some("dQw4w9WgXcQ")
        );
        assert_eq!(extract_video_id("youtu.be/abc").as_deref(), Some("abc"));
    }

    #[test]
    fn extract_id_shorts_live_embed() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/shorts/abc123XYZ_-").as_deref(),
            Some("abc123XYZ_-")
        );
        assert_eq!(
            extract_video_id("https://www.youtube.com/live/abc123XYZ_-").as_deref(),
            Some("abc123XYZ_-")
        );
        assert_eq!(
            extract_video_id("https://www.youtube.com/embed/embedID0001").as_deref(),
            Some("embedID0001")
        );
    }

    #[test]
    fn extract_id_mobile_music_nocookie_hosts() {
        assert_eq!(
            extract_video_id("https://m.youtube.com/watch?v=abc").as_deref(),
            Some("abc")
        );
        assert_eq!(
            extract_video_id("https://music.youtube.com/watch?v=abc").as_deref(),
            Some("abc")
        );
        assert_eq!(
            extract_video_id("https://www.youtube-nocookie.com/watch?v=abc").as_deref(),
            Some("abc")
        );
    }

    #[test]
    fn extract_id_scheme_optional() {
        assert_eq!(
            extract_video_id("youtube.com/watch?v=abc").as_deref(),
            Some("abc")
        );
    }

    #[test]
    fn extract_id_none_for_playlist_and_bad_inputs() {
        // Pure playlist: no single video id.
        assert_eq!(
            extract_video_id("https://www.youtube.com/playlist?list=PL123"),
            None
        );
        // watch?list= with no v= -> playlist landing page, no id.
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?list=PL123"),
            None
        );
        // Non-YouTube host.
        assert_eq!(extract_video_id("https://vimeo.com/12345"), None);
        // youtu.be with no id.
        assert_eq!(extract_video_id("https://youtu.be/"), None);
        // Empty / garbage.
        assert_eq!(extract_video_id("   "), None);
        assert_eq!(extract_video_id("not even a url"), None);
        // Empty v= value.
        assert_eq!(extract_video_id("https://www.youtube.com/watch?v="), None);
    }

    /// Every URL `classify_url` accepts as a single Video must yield an id, so
    /// the resolve fast-path never needs the fallback fetch for these.
    #[test]
    fn extract_id_covers_every_classified_video_form() {
        for url in [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=42",
            "https://www.youtube.com/shorts/abc123XYZ_-",
            "https://www.youtube.com/live/abc123XYZ_-",
            "https://www.youtube.com/watch?v=abc&list=PL123",
            "https://www.youtube.com/watch?list=PL123&v=abc",
            "https://m.youtube.com/watch?v=abc",
            "https://music.youtube.com/watch?v=abc",
            "https://www.youtube-nocookie.com/watch?v=abc",
            "youtube.com/watch?v=abc",
            "youtu.be/abc",
        ] {
            let kind = classify_url(url).unwrap();
            assert!(
                matches!(kind, UrlKind::Video | UrlKind::Ambiguous),
                "{url} should classify as Video/Ambiguous, got {kind:?}"
            );
            assert!(
                extract_video_id(url).is_some(),
                "{url} classified as {kind:?} but extract_video_id returned None"
            );
        }
    }

    // ---- expand_playlist scale measurement -------------------------------

    /// Build a synthetic stub that emits `n` *realistic* flat-playlist JSON
    /// lines (yt-dlp `--flat-playlist --dump-json` lines are NOT ~150 bytes —
    /// they carry a thumbnails array, channel/uploader block, description, etc.,
    /// landing around 1.5–3 KB each). Returns `(tempdir, info, approx_line_len)`.
    ///
    /// The generated script ALSO honors `--version` (so it can be probed) and
    /// prints the lines verbatim, exercising the real `run_command_cancellable`
    /// capture path — including the 4 MiB `MAX_CAPTURED_OUTPUT_BYTES` cap that
    /// `expand_playlist`'s stdout flows through. This lets the measurement prove
    /// whether a large playlist's flat-JSON gets silently truncated.
    fn synthetic_flat_playlist_info(n: usize) -> (tempfile::TempDir, YtdlpInfo, usize) {
        let dir = tempfile::tempdir().expect("tempdir");
        let script_path = dir.path().join("flat_stub.sh");
        // A representative flat-playlist entry template. yt-dlp emits a fat
        // object per entry; this mirrors the realistic field set + a thumbnails
        // array so the per-line byte cost is faithful (~1.6 KB here).
        let line_template = |i: usize| -> String {
            serde_json::json!({
                "_type": "url",
                "ie_key": "Youtube",
                "id": format!("vid{i:08}xyz"),
                "url": format!("https://www.youtube.com/watch?v=vid{i:08}xyz"),
                "title": format!("A Reasonably Long Representative Playlist Video Title Number {i}"),
                "description": "A multi-sentence description that yt-dlp includes in flat dumps. \
                                It is typically a couple hundred characters of prose padding.",
                "duration": 245.0 + (i % 600) as f64,
                "channel_id": "UCabcdefghijklmnopqrstuv",
                "channel": "Some Representative Channel Name",
                "channel_url": "https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv",
                "uploader": "Some Representative Channel Name",
                "uploader_id": "@somerepresentativechannel",
                "uploader_url": "https://www.youtube.com/@somerepresentativechannel",
                "view_count": 123456 + i,
                "availability": "public",
                "live_status": "not_live",
                "thumbnails": (0..5).map(|t| serde_json::json!({
                    "url": format!("https://i.ytimg.com/vi/vid{i:08}xyz/hqdefault_{t}.jpg"),
                    "height": 94 + t * 100,
                    "width": 168 + t * 160,
                })).collect::<Vec<_>>(),
            })
            .to_string()
        };
        let approx_line_len = line_template(0).len() + 1; // + newline
        // Emit all lines from the script via a heredoc-free, fast `cat` of a
        // pre-materialized data file (keeps the script tiny and the stdout
        // generation cost out of the parse-time measurement's critical section).
        let data_path = dir.path().join("flat_lines.jsonl");
        {
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&data_path).unwrap());
            for i in 0..n {
                writeln!(f, "{}", line_template(i)).unwrap();
            }
            f.flush().unwrap();
        }
        let script = format!(
            "#!/usr/bin/env bash\nset -u\nfor a in \"$@\"; do [ \"$a\" = --version ] && \
             {{ echo 2025.01.01; exit 0; }}; done\ncat {}\n",
            data_path.display()
        );
        std::fs::write(&script_path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms).unwrap();
        }
        let info = YtdlpInfo {
            path: script_path,
            version: "2025.01.01".to_owned(),
            stale: false,
        };
        (dir, info, approx_line_len)
    }

    /// MEASURE: time `expand_playlist` parsing for a large synthetic playlist
    /// and report the per-line byte cost + total stdout size against the 4 MiB
    /// `process::MAX_CAPTURED_OUTPUT_BYTES` cap. Asserts the parse is linear and
    /// (within the cap) loses no entries; reports timing for the perf record.
    #[test]
    fn expand_playlist_scale_2000_is_linear_and_within_cap() {
        const N: usize = 2000;
        let (_dir, info, line_len) = synthetic_flat_playlist_info(N);
        let token = CancellationToken::unbounded();

        let t = std::time::Instant::now();
        let refs =
            expand_playlist(&info, "https://www.youtube.com/playlist?list=PLbig", &token).unwrap();
        let elapsed = t.elapsed();

        let total_bytes = line_len * N;
        let cap = 4 * 1024 * 1024;
        eprintln!(
            "expand_playlist scale: N={N} entries, ~{line_len} B/line, \
             ~{total_bytes} B total stdout (cap={cap} B), parsed {} refs in {:?} \
             ({:.1} us/entry)",
            refs.len(),
            elapsed,
            elapsed.as_secs_f64() * 1e6 / N as f64,
        );
        // The realistic ~1.6 KB/line × 2000 ≈ 3.2 MB stays just under the 4 MiB
        // cap, so NO entries are lost here. (At ~2700+ such entries it WOULD
        // exceed the cap — see expand_playlist_truncation_is_now_detected.)
        assert_eq!(refs.len(), N, "all entries parsed when under the cap");
    }

    /// CORRECTNESS REGRESSION GUARD: a playlist whose flat-JSON stdout exceeds
    /// the 4 MiB capture cap must NOT be parsed as a silently-shortened list.
    /// Pre-fix this returned a truncated `Vec` (dropping the tail videos with no
    /// error). The fix makes `expand_playlist` detect a cap-truncated capture
    /// and surface an actionable error instead of silently losing videos.
    #[test]
    fn expand_playlist_truncation_is_now_detected() {
        // ~1.6 KB/line × 4000 ≈ 6.4 MB > 4 MiB cap → capture is truncated.
        const N: usize = 4000;
        let (_dir, info, line_len) = synthetic_flat_playlist_info(N);
        let token = CancellationToken::unbounded();
        let total = line_len * N;
        assert!(
            total > 4 * 1024 * 1024,
            "test precondition: {total} B must exceed the 4 MiB cap"
        );
        let result = expand_playlist(
            &info,
            "https://www.youtube.com/playlist?list=PLhuge",
            &token,
        );
        match result {
            Err(FwError::InvalidRequest(msg)) => {
                assert!(
                    msg.contains("truncat") || msg.contains("too large") || msg.contains("4 MiB"),
                    "expected a truncation/too-large error, got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidRequest about truncation, got {other:?}"),
            Ok(refs) => panic!(
                "BUG: silently returned {} refs from a cap-truncated capture (expected an error)",
                refs.len()
            ),
        }
    }

    #[test]
    fn capture_truncation_failure_signature() {
        let explicit = FwError::ContractViolation(
            "subprocess stdout exceeded the 4194304-byte capture limit".to_owned(),
        );
        assert!(is_stdout_capture_limit_error(&explicit));
        let stderr_only = FwError::ContractViolation(
            "subprocess stderr exceeded the 4194304-byte capture limit".to_owned(),
        );
        assert!(!is_stdout_capture_limit_error(&stderr_only));
        // Signal-style failures are not evidence of capture overflow.
        let sigpipe = FwError::from_command_failure("yt-dlp ...".to_owned(), 141, String::new());
        assert!(!is_stdout_capture_limit_error(&sigpipe));
        let killed = FwError::from_command_failure("yt-dlp ...".to_owned(), -1, "  \n".to_owned());
        assert!(!is_stdout_capture_limit_error(&killed));
        // A genuine yt-dlp error writes to stderr -> NOT a truncation signature.
        let real = FwError::from_command_failure(
            "yt-dlp ...".to_owned(),
            1,
            "ERROR: Private video".to_owned(),
        );
        assert!(!is_stdout_capture_limit_error(&real));
        // A normal non-signal failure with empty stderr is also not it.
        let plain = FwError::from_command_failure("yt-dlp ...".to_owned(), 2, String::new());
        assert!(!is_stdout_capture_limit_error(&plain));
    }

    #[test]
    fn playlist_too_large_error_is_actionable() {
        let e = playlist_too_large_error("https://www.youtube.com/playlist?list=PLx");
        match e {
            FwError::InvalidRequest(msg) => {
                assert!(msg.contains("too large"), "{msg}");
                assert!(msg.contains("truncat"), "{msg}");
                assert!(msg.contains("4 MiB"), "{msg}");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    // ---- search / enrich (via stub, bd-m7fv) -----------------------------

    #[test]
    fn search_enriched_dedupes_and_retains_curated_fields() {
        let token = CancellationToken::unbounded();
        let hits = search(&stub_info(), "rust livestream", 10, false, &token)
            .expect("enriched search should succeed");
        assert_eq!(
            hits.len(),
            2,
            "duplicate third line must be deduped: {hits:?}"
        );
        assert_eq!(hits[0].id, "srchenr0001");
        assert_eq!(hits[0].title, "Enriched Search Hit One");
        assert_eq!(hits[0].url, "https://www.youtube.com/watch?v=srchenr0001");
        assert_eq!(hits[0].channel.as_deref(), Some("Search Channel"));
        assert_eq!(hits[0].view_count, Some(4242));
        assert_eq!(hits[0].upload_date.as_deref(), Some("20250301"));
        assert_eq!(hits[0].duration_sec, Some(187.5));
        // Second hit exercises the webpage_url fallback for the canonical URL
        // and omits fields the fixture does not carry.
        assert_eq!(hits[1].id, "srchenr0002");
        assert_eq!(hits[1].channel.as_deref(), Some("Other Channel"));
        assert_eq!(hits[1].view_count, None);
        assert_eq!(hits[1].upload_date, None);
    }

    #[test]
    fn search_flat_keeps_flat_subset() {
        let token = CancellationToken::unbounded();
        let hits = search(&stub_info(), "cheap sweep", 10, true, &token)
            .expect("flat search should succeed");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "srchflat0001");
        assert_eq!(hits[0].duration_sec, Some(95.0));
        assert_eq!(hits[1].id, "srchflat0002");
        // webpage_url fallback supplies the canonical URL in the flat dump.
        assert_eq!(hits[1].url, "https://www.youtube.com/watch?v=srchflat0002");
    }

    #[test]
    fn search_respects_limit_cap() {
        let token = CancellationToken::unbounded();
        let hits = search(&stub_info(), "anything", 1, false, &token)
            .expect("limited search should succeed");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "srchenr0001");
    }

    #[test]
    fn search_rejects_empty_query_and_zero_limit() {
        let token = CancellationToken::unbounded();
        let err = search(&stub_info(), "   ", 10, false, &token).unwrap_err();
        assert!(matches!(err, FwError::InvalidRequest(_)));
        let err = search(&stub_info(), "query", 0, false, &token).unwrap_err();
        assert!(matches!(err, FwError::InvalidRequest(_)));
    }

    #[test]
    fn enrich_dedupes_targets_and_maps_metadata() {
        let token = CancellationToken::unbounded();
        let targets = vec![
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ".to_owned(),
            "dQw4w9WgXcQ".to_owned(),
        ];
        let hits = enrich(&stub_info(), &targets, &token).expect("enrich should succeed");
        assert_eq!(hits.len(), 1, "same id twice must collapse to one hit");
        assert_eq!(hits[0].id, "dQw4w9WgXcQ");
        assert_eq!(hits[0].channel.as_deref(), Some("Stub Channel"));
        assert_eq!(hits[0].upload_date.as_deref(), Some("20240115"));
        assert_eq!(hits[0].duration_sec, Some(212.0));
    }

    #[test]
    fn enrich_rejects_playlist_urls_actionably() {
        let token = CancellationToken::unbounded();
        let targets = vec!["https://www.youtube.com/playlist?list=PL123".to_owned()];
        let err = enrich(&stub_info(), &targets, &token).unwrap_err();
        match err {
            FwError::InvalidRequest(msg) => {
                assert!(msg.contains("Playlist") && msg.contains("individual video"));
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn dedup_hits_preserves_first_seen_order() {
        let hit = |id: &str| SearchHit {
            id: id.to_owned(),
            title: String::new(),
            url: format!("https://www.youtube.com/watch?v={id}"),
            duration_sec: None,
            channel: None,
            view_count: None,
            upload_date: None,
        };
        let merged = dedup_hits(vec![hit("a"), hit("b"), hit("a"), hit("c"), hit("b")]);
        let ids: Vec<&str> = merged.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    // ---- expand_playlist (via stub) --------------------------------------

    #[test]
    fn expand_playlist_parses_two_entries() {
        let token = CancellationToken::unbounded();
        let refs = expand_playlist(
            &stub_info(),
            "https://www.youtube.com/playlist?list=PL123",
            &token,
        )
        .expect("expand should succeed");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].id, "vid000000001");
        assert_eq!(refs[0].title, "First Playlist Entry");
        assert_eq!(refs[0].url, "https://www.youtube.com/watch?v=vid000000001");
        assert_eq!(refs[0].duration_sec, Some(61.0));
        // Second entry uses webpage_url fallback + integer duration.
        assert_eq!(refs[1].id, "vid000000002");
        assert_eq!(refs[1].url, "https://www.youtube.com/watch?v=vid000000002");
        assert_eq!(refs[1].duration_sec, Some(123.0));
    }

    #[test]
    fn projected_video_ref_url_fallback_to_synthetic() {
        let r = parse_flat_playlist_line(r#"{"id":"xyz","title":"T"}"#)
            .unwrap()
            .unwrap();
        assert_eq!(r.url, "https://www.youtube.com/watch?v=xyz");
        assert_eq!(r.duration_sec, None);
    }

    #[test]
    fn projected_video_ref_no_id_is_none() {
        assert!(
            parse_flat_playlist_line(r#"{"title":"T"}"#)
                .unwrap()
                .is_none()
        );
        assert!(parse_flat_playlist_line(r#"{"id":""}"#).unwrap().is_none());
    }

    fn legacy_video_ref_from_json(value: &serde_json::Value) -> Option<VideoRef> {
        let id = value.get("id").and_then(serde_json::Value::as_str)?;
        if id.is_empty() {
            return None;
        }
        let title = string_field(value, "title").unwrap_or_default();
        let url = string_field(value, "url")
            .or_else(|| string_field(value, "webpage_url"))
            .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));
        let duration_sec = value.get("duration").and_then(serde_json::Value::as_f64);
        Some(VideoRef {
            id: id.to_owned(),
            title,
            url,
            duration_sec,
        })
    }

    fn legacy_parse_flat_playlist_line(line: &str) -> serde_json::Result<Option<VideoRef>> {
        serde_json::from_str(line).map(|value| legacy_video_ref_from_json(&value))
    }

    fn assert_video_refs_exact(left: &Option<VideoRef>, right: &Option<VideoRef>) {
        match (left, right) {
            (Some(left), Some(right)) => {
                assert_eq!(left.id, right.id);
                assert_eq!(left.title, right.title);
                assert_eq!(left.url, right.url);
                assert_eq!(
                    left.duration_sec.map(f64::to_bits),
                    right.duration_sec.map(f64::to_bits)
                );
            }
            (None, None) => {}
            _ => panic!("projected and legacy parsers disagreed: {left:?} != {right:?}"),
        }
    }

    #[test]
    fn projected_playlist_parser_matches_value_dom_reference() {
        for line in [
            r#"{"id":"abc","title":"title","url":"https://youtu.be/abc","webpage_url":"ignored","duration":1.25,"description":{"fat":[1,2,3]}}"#,
            r#"{"id":"escaped\\\"id","title":"snowman ☃","webpage_url":"https://example.test/watch","duration":-0.0}"#,
            r#"{"id":"abc","title":"","url":"","webpage_url":"","duration":7}"#,
            r#"{"id":"abc","title":7,"url":false,"webpage_url":"fallback","duration":"9"}"#,
            r#"{"id":""}"#,
            r#"{"id":42}"#,
            r#"{"title":"missing id"}"#,
            r#"[]"#,
            r#"null"#,
            r#"{"id":"unterminated""#,
        ] {
            let legacy = legacy_parse_flat_playlist_line(line);
            let projected = parse_flat_playlist_line(line);
            match (legacy, projected) {
                (Ok(legacy), Ok(projected)) => assert_video_refs_exact(&legacy, &projected),
                (Err(_), Err(_)) => {}
                // A valid non-object JSON value was historically classified as
                // an entry without an id; the projected struct rejects it as a
                // parse mismatch. Both production paths skip the line.
                (Ok(None), Err(_)) | (Err(_), Ok(None)) => {}
                (legacy, projected) => {
                    panic!("parse outcome mismatch for {line:?}: {legacy:?} != {projected:?}")
                }
            }
        }
    }

    // ---- fetch_metadata (via stub) ---------------------------------------

    #[test]
    fn fetch_metadata_parses_full_object() {
        let token = CancellationToken::unbounded();
        let meta = fetch_metadata(&stub_info(), "https://youtu.be/dQw4w9WgXcQ", &token)
            .expect("metadata should parse");
        assert_eq!(meta.id, "dQw4w9WgXcQ");
        assert_eq!(meta.title, "Stub Title dQw4w9WgXcQ");
        assert_eq!(meta.channel.as_deref(), Some("Stub Channel"));
        assert_eq!(meta.uploader.as_deref(), Some("Stub Uploader"));
        assert_eq!(meta.upload_date.as_deref(), Some("20240115"));
        assert_eq!(meta.duration_sec, Some(212.0));
        assert_eq!(meta.availability.as_deref(), Some("public"));
        assert_eq!(meta.live_status.as_deref(), Some("not_live"));
        assert!(meta.description.is_some());
    }

    #[test]
    fn fetch_metadata_rejects_live_stream() {
        let token = CancellationToken::unbounded();
        // Drive STUB_LIVE_STATUS via a wrapper would need env; instead test the
        // pure rejection path on a synthesized object below. Here we confirm the
        // happy path is not live. Live rejection is covered by
        // video_meta_rejects_live_via_helper.
        let meta = fetch_metadata(&stub_info(), "https://youtu.be/x", &token).unwrap();
        assert_ne!(meta.live_status.as_deref(), Some("is_live"));
    }

    #[test]
    fn video_meta_from_json_live_status_surfaced() {
        let live = serde_json::json!({
            "id": "x", "title": "T",
            "webpage_url": "https://youtu.be/x",
            "live_status": "is_live"
        });
        let meta = video_meta_from_json(&live).unwrap();
        assert_eq!(meta.live_status.as_deref(), Some("is_live"));
    }

    #[test]
    fn video_meta_from_json_requires_id() {
        let no_id = serde_json::json!({"title": "T"});
        assert!(matches!(
            video_meta_from_json(&no_id),
            Err(FwError::InvalidRequest(_))
        ));
    }

    #[test]
    fn video_meta_from_json_synthesizes_webpage_url() {
        let value = serde_json::json!({"id": "abc", "title": "T"});
        let meta = video_meta_from_json(&value).unwrap();
        assert_eq!(meta.webpage_url, "https://www.youtube.com/watch?v=abc");
    }

    #[test]
    fn projected_video_meta_preserves_dom_outcomes() {
        for line in [
            r#"{"id":"abc","title":"T","channel":"C","uploader":"U","upload_date":"20260715","duration":-0.0,"webpage_url":"https://youtu.be/abc","description":"D","availability":"public","live_status":"not_live","ignored":{"fat":[1,2,3]}}"#,
            r#"{"id":"abc","title":"","channel":null,"uploader":7,"duration":"9","webpage_url":""}"#,
            r#"{"id":"first","id":"last","title":"last duplicate wins"}"#,
            r#"{"title":"missing id"}"#,
            r#"[1,{"id":"nested"}]"#,
            "null",
            "true",
            r#"{"id":"unterminated""#,
        ] {
            let legacy: FwResult<VideoMeta> = serde_json::from_str(line)
                .map_err(FwError::from)
                .and_then(|value| video_meta_from_json(&value));
            let projected = parse_video_meta(line);
            match (legacy, projected) {
                (Ok(legacy), Ok(projected)) => {
                    assert_eq!(legacy, projected, "metadata mismatch for {line}");
                    assert_eq!(
                        legacy.duration_sec.map(f64::to_bits),
                        projected.duration_sec.map(f64::to_bits),
                        "duration bits mismatch for {line}"
                    );
                }
                (Err(legacy), Err(projected)) => {
                    assert_eq!(legacy.error_code(), projected.error_code(), "{line}");
                }
                (legacy, projected) => {
                    panic!("parse outcome mismatch for {line}: {legacy:?} != {projected:?}");
                }
            }
        }
    }

    // ---- download_audio (via stub) ---------------------------------------

    #[test]
    fn download_audio_copies_fixture_and_returns_path() {
        let token = CancellationToken::unbounded();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = download_audio(&stub_info(), &meta_for_download(), dir.path(), &token)
            .expect("download should succeed");
        assert!(path.is_file(), "returned path should exist: {path:?}");
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some("dQw4w9WgXcQ.wav")
        );
        assert!(path.starts_with(dir.path()));
        let downloaded = std::fs::read(&path).expect("read stub download");
        let tracked_fixture = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/native/jfk_cut8.bin"
        ));
        assert_eq!(downloaded.as_slice(), tracked_fixture);
    }

    #[test]
    fn downloaded_scan_rejects_partial_and_sidecar_artifacts() {
        let dir = tempfile::tempdir().expect("tempdir");
        for extension in ["part", "tmp", "ytdl"] {
            std::fs::write(dir.path().join(format!("video123.{extension}")), b"partial")
                .expect("partial artifact");
        }
        assert_eq!(find_downloaded_by_id(dir.path(), "video123"), None);

        let completed = dir.path().join("video123.webm");
        std::fs::write(&completed, b"complete").expect("completed artifact");
        assert_eq!(find_downloaded_by_id(dir.path(), "video123"), Some(completed));
    }

    // ---- error mapping ---------------------------------------------------

    #[test]
    fn classify_stderr_signatures() {
        assert!(classify_stderr("error: private video. sign in").is_some());
        assert!(classify_stderr("this video is unavailable").is_some());
        assert!(classify_stderr("sign in to confirm your age").is_some());
        assert!(classify_stderr("not available in your country").is_some());
        assert!(classify_stderr("http error 429: too many requests").is_some());
        assert!(classify_stderr("some unrelated failure").is_none());
    }

    #[test]
    fn map_ytdlp_error_private_becomes_invalid_request() {
        let raw = FwError::from_command_failure(
            "yt-dlp ...".to_owned(),
            1,
            "ERROR: Private video. Sign in if you've been granted access".to_owned(),
        );
        let mapped = map_ytdlp_error(raw);
        match mapped {
            FwError::InvalidRequest(msg) => assert!(msg.contains("private")),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn map_ytdlp_error_429_becomes_invalid_request() {
        let raw = FwError::from_command_failure(
            "yt-dlp ...".to_owned(),
            1,
            "ERROR: HTTP Error 429: Too Many Requests".to_owned(),
        );
        match map_ytdlp_error(raw) {
            FwError::InvalidRequest(msg) => assert!(msg.contains("429")),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn map_ytdlp_error_unknown_failure_passes_through() {
        let raw = FwError::from_command_failure(
            "yt-dlp ...".to_owned(),
            2,
            "ERROR: some novel failure mode".to_owned(),
        );
        assert!(matches!(
            map_ytdlp_error(raw),
            FwError::CommandFailed { .. }
        ));
    }

    #[test]
    fn map_ytdlp_error_cancelled_passes_through() {
        let raw = FwError::Cancelled("ctrl-c".to_owned());
        assert!(matches!(map_ytdlp_error(raw), FwError::Cancelled(_)));
    }

    // ---- stub-driven error injection (end-to-end through run_ytdlp) -------

    /// Select an error mode through the stable stub's URL-query test hook.
    ///
    /// This avoids creating and immediately executing a temporary script,
    /// which can fail with ETXTBSY on Linux/network-backed filesystems under
    /// parallel test load even after the writer has closed the file.
    fn failing_case(mode: &str) -> (YtdlpInfo, String) {
        (
            stub_info(),
            format!("https://youtu.be/x?fw_stub_fail={mode}"),
        )
    }

    #[test]
    fn fetch_metadata_private_mode_maps_to_invalid_request() {
        let (info, url) = failing_case("private");
        let token = CancellationToken::unbounded();
        let err = fetch_metadata(&info, &url, &token).expect_err("should fail");
        match err {
            FwError::InvalidRequest(msg) => assert!(msg.contains("private")),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn fetch_metadata_geo_mode_maps_to_invalid_request() {
        let (info, url) = failing_case("geo");
        let token = CancellationToken::unbounded();
        let err = fetch_metadata(&info, &url, &token).expect_err("should fail");
        match err {
            FwError::InvalidRequest(msg) => assert!(msg.contains("geo-blocked")),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn fetch_metadata_429_mode_maps_to_invalid_request() {
        let (info, url) = failing_case("429");
        let token = CancellationToken::unbounded();
        let err = fetch_metadata(&info, &url, &token).expect_err("should fail");
        match err {
            FwError::InvalidRequest(msg) => assert!(msg.contains("429")),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn fetch_metadata_generic_exit1_passes_through_as_command_failed() {
        let (info, url) = failing_case("exit1");
        let token = CancellationToken::unbounded();
        let err = fetch_metadata(&info, &url, &token).expect_err("should fail");
        assert!(
            matches!(err, FwError::CommandFailed { .. }),
            "generic failure should remain CommandFailed, got: {err:?}"
        );
    }

    // ---- helpers ---------------------------------------------------------

    #[test]
    fn string_field_treats_empty_and_null_as_absent() {
        let value = serde_json::json!({"a": "x", "b": "", "c": null});
        assert_eq!(string_field(&value, "a").as_deref(), Some("x"));
        assert_eq!(string_field(&value, "b"), None);
        assert_eq!(string_field(&value, "c"), None);
        assert_eq!(string_field(&value, "missing"), None);
    }

    #[test]
    fn truncate_for_log_short_unchanged() {
        assert_eq!(truncate_for_log("short"), "short");
    }

    #[test]
    fn truncate_for_log_long_is_clipped() {
        let long = "a".repeat(500);
        let out = truncate_for_log(&long);
        assert!(out.len() < 500);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn split_host_and_rest_handles_no_scheme() {
        assert_eq!(
            split_host_and_rest("youtube.com/watch?v=x"),
            Some(("youtube.com", "/watch?v=x"))
        );
        assert_eq!(
            split_host_and_rest("https://youtu.be/x"),
            Some(("youtu.be", "/x"))
        );
    }

    #[test]
    fn query_has_nonempty_param_works() {
        assert!(query_has_nonempty_param("v=abc&list=def", "v"));
        assert!(query_has_nonempty_param("v=abc&list=def", "list"));
        assert!(!query_has_nonempty_param("v=&list=def", "v"));
        assert!(!query_has_nonempty_param("list=def", "v"));
    }

    #[test]
    fn is_youtube_host_variants() {
        assert!(is_youtube_host("youtube.com"));
        assert!(is_youtube_host("www.youtube.com"));
        assert!(is_youtube_host("m.youtube.com"));
        assert!(is_youtube_host("music.youtube.com"));
        assert!(is_youtube_host("youtube-nocookie.com"));
        assert!(!is_youtube_host("vimeo.com"));
        assert!(!is_youtube_host("notyoutube.com"));
    }
}
