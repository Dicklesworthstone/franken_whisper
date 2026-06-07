//! Markdown + JSON renderers for transcribed YouTube videos.
//!
//! This module is the *output* end of the YouTube ingestion pipeline
//! (see [`crate::youtube`]). It consumes a single self-contained
//! [`RenderInput`] — the **integration contract** between this renderer and
//! `youtube/pipeline.rs` — and produces two artifacts:
//!
//! - a human-facing **Markdown** transcript ([`render_markdown`]), styled like
//!   the call-transcript format (H1 title, a metadata line, an honesty note,
//!   then timestamped paragraphs deep-linked into the video), and
//! - a machine-facing **JSON** document ([`render_json`]) following the epic
//!   schema (`video` / `run` / `utterances`).
//!
//! Both are written to disk via [`write_atomic`] (temp file in the same
//! directory + atomic rename, so a crash never leaves a half-written file in
//! place of a good one).
//!
//! # Integration contract
//!
//! The pipeline bead (`youtube/pipeline.rs`) owns the conversion from its own
//! download/metadata/transcription state into a [`RenderInput`]. This module
//! deliberately does **not** import from `ytdlp.rs` / `pipeline.rs`; it defines
//! its own borrow-friendly input structs so the two beads can land in parallel.
//! The only shared type pulled in from the wider crate is
//! [`TranscriptionSegment`](crate::model::TranscriptionSegment), which is the
//! native engine's segment shape and the thing the renderer actually consumes.
//!
//! Construct a [`RenderInput`] by filling [`RenderVideo`] (everything yt-dlp's
//! metadata fetch knows about the video) and [`RenderRun`] (everything the
//! transcription run knows about *how* it was produced), then borrow the
//! decoded `segments` slice. Nothing is consumed — the input borrows the
//! segments — so the pipeline can render Markdown and JSON from one input
//! without cloning.

use std::path::Path;

use serde_json::{Map, Value, json};

use crate::error::FwResult;
use crate::model::TranscriptionSegment;

/// Paragraphs break when the silent gap to the previous segment exceeds this
/// many seconds. Tuned for prose readability: ~2.5 s is a natural sentence /
/// breath boundary in speech without fragmenting normal pauses.
const PARAGRAPH_GAP_SEC: f64 = 2.5;

/// A paragraph is force-split once it grows past this many words, even with no
/// gap or speaker change. Without this cap, a long monologue with no pauses
/// renders as one unreadable wall of text; ~120 words is roughly a dense screen
/// paragraph.
const PARAGRAPH_WORD_CAP: usize = 120;

/// Maximum number of characters of the video description surfaced as a quoted
/// intro blockquote in the Markdown. The full description lives in the JSON;
/// the Markdown keeps only a short teaser so the document stays tight.
const DESCRIPTION_INTRO_CHARS: usize = 280;

/// Video-level metadata for rendering. Mirrors what yt-dlp's metadata fetch
/// surfaces; every optional field is omitted from the JSON when `None`.
///
/// Part of the [`RenderInput`] integration contract.
#[derive(Debug, Clone)]
pub struct RenderVideo {
    /// YouTube video id (the `v=` / `youtu.be/` slug). Used for deep links.
    pub id: String,
    /// Display title (H1 of the Markdown, `video.title` in JSON).
    pub title: String,
    /// Channel name, if known.
    pub channel: Option<String>,
    /// Uploader name, if known (often equal to `channel`; preserved distinctly).
    pub uploader: Option<String>,
    /// Upload date in yt-dlp's compact `YYYYMMDD` form, if known. Rendered as
    /// `YYYY-MM-DD` in Markdown; passed through verbatim in JSON.
    pub upload_date: Option<String>,
    /// Total video duration in seconds, if known.
    pub duration_sec: Option<f64>,
    /// Canonical watch URL (`https://www.youtube.com/watch?v=...`).
    pub webpage_url: String,
    /// Full description. Only the first [`DESCRIPTION_INTRO_CHARS`] chars are
    /// shown in Markdown (as a quoted intro); the whole thing is in JSON.
    pub description: Option<String>,
}

/// Run-level metadata: how the transcript was produced.
///
/// Part of the [`RenderInput`] integration contract.
#[derive(Debug, Clone)]
pub struct RenderRun {
    /// Model name/id (e.g. `large-v3`).
    pub model: String,
    /// Engine label (e.g. `native`, `whisper-cli`).
    pub engine: String,
    /// Backend label (e.g. `cpu`, `frankentorch`).
    pub backend: String,
    /// Release/version tag of franken_whisper, if available.
    pub version_tag: Option<String>,
    /// RFC 3339 timestamp of when the run started.
    pub started_rfc3339: String,
    /// Wall-clock duration of the run in milliseconds.
    pub wall_ms: u64,
    /// Real-time factor (wall / audio duration), if computable.
    pub rtf: Option<f64>,
}

/// The complete, self-contained input to the renderers.
///
/// This is the integration contract with `youtube/pipeline.rs`: the pipeline
/// assembles one of these and calls [`render_markdown`] / [`render_json`].
/// The `segments` field borrows the engine's decoded
/// [`TranscriptionSegment`]s; nothing is consumed.
#[derive(Debug)]
pub struct RenderInput<'a> {
    /// Video-level metadata.
    pub video: RenderVideo,
    /// Run-level metadata.
    pub run: RenderRun,
    /// The transcript segments, borrowed from the engine. These are the **raw**
    /// segments; the Markdown groups them into paragraphs, but the JSON
    /// `utterances` array is one entry per segment (count is preserved).
    pub segments: &'a [TranscriptionSegment],
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a timestamp for the Markdown body label.
///
/// `m:ss` below one hour (e.g. `1:23`), `h:mm:ss` at or after one hour
/// (e.g. `1:01:01`). Negative / non-finite inputs are clamped to zero.
fn format_timestamp_label(seconds: f64) -> String {
    let total = floor_secs(seconds);
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// Format a duration (H:MM:SS, hours unpadded) for the metadata line.
fn format_duration(seconds: f64) -> String {
    let total = floor_secs(seconds);
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{h}:{m:02}:{s:02}")
}

/// Floor a seconds value to a non-negative whole-second count.
fn floor_secs(seconds: f64) -> u64 {
    if seconds.is_finite() && seconds > 0.0 {
        seconds.floor() as u64
    } else {
        0
    }
}

/// Convert yt-dlp's `YYYYMMDD` to `YYYY-MM-DD`. Returns the input unchanged if
/// it is not exactly 8 ASCII digits.
fn format_upload_date(raw: &str) -> String {
    let bytes = raw.as_bytes();
    if bytes.len() == 8 && bytes.iter().all(u8::is_ascii_digit) {
        format!("{}-{}-{}", &raw[0..4], &raw[4..6], &raw[6..8])
    } else {
        raw.to_owned()
    }
}

/// Deep-link URL into the video at the given start second.
fn deep_link(id: &str, start_sec: f64) -> String {
    format!("https://youtu.be/{id}?t={}", floor_secs(start_sec))
}

/// Count whitespace-delimited words in `text`.
fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// The effective start of a segment (`start_sec`, defaulting to 0.0).
fn seg_start(seg: &TranscriptionSegment) -> f64 {
    seg.start_sec.unwrap_or(0.0)
}

/// The effective end of a segment (`end_sec`, falling back to `start_sec`,
/// then 0.0). Used only for gap computation between adjacent segments.
fn seg_end(seg: &TranscriptionSegment) -> f64 {
    seg.end_sec.or(seg.start_sec).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Paragraph grouping
// ---------------------------------------------------------------------------

/// A grouped paragraph: the segments that make it up, plus its lead-in
/// timestamp/speaker (taken from its first segment).
struct Paragraph<'a> {
    start_sec: f64,
    speaker: Option<&'a str>,
    segments: Vec<&'a TranscriptionSegment>,
}

/// Group raw segments into Markdown paragraphs.
///
/// A new paragraph begins when any of the following holds relative to the
/// segment being added:
/// - the silent gap from the previous segment's end exceeds
///   [`PARAGRAPH_GAP_SEC`],
/// - the speaker label changes (diarized inputs), or
/// - the current paragraph already exceeds [`PARAGRAPH_WORD_CAP`] words
///   (readability cap).
fn group_paragraphs(segments: &[TranscriptionSegment]) -> Vec<Paragraph<'_>> {
    let mut paragraphs: Vec<Paragraph<'_>> = Vec::new();
    let mut current: Option<Paragraph<'_>> = None;
    let mut current_words = 0usize;
    let mut prev_end: Option<f64> = None;

    for seg in segments {
        let speaker = seg.speaker.as_deref();
        let start = seg_start(seg);
        let seg_words = word_count(&seg.text);

        let gap_break = prev_end.is_some_and(|pe| start - pe > PARAGRAPH_GAP_SEC);
        let speaker_break = current.as_ref().is_some_and(|p| p.speaker != speaker);
        let word_break = current.is_some() && current_words >= PARAGRAPH_WORD_CAP;

        if current.is_none() || gap_break || speaker_break || word_break {
            if let Some(done) = current.take() {
                paragraphs.push(done);
            }
            current = Some(Paragraph {
                start_sec: start,
                speaker,
                segments: vec![seg],
            });
            current_words = seg_words;
        } else if let Some(p) = current.as_mut() {
            p.segments.push(seg);
            current_words += seg_words;
        }

        prev_end = Some(seg_end(seg));
    }

    if let Some(done) = current.take() {
        paragraphs.push(done);
    }
    paragraphs
}

/// Join a paragraph's segment texts into a single trimmed prose string,
/// collapsing inter-segment whitespace to a single space.
fn paragraph_text(p: &Paragraph<'_>) -> String {
    let mut out = String::new();
    for seg in &p.segments {
        let piece = seg.text.trim();
        if piece.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(piece);
    }
    out
}

// ---------------------------------------------------------------------------
// Markdown renderer
// ---------------------------------------------------------------------------

/// Render the Markdown transcript for a video.
///
/// Layout:
/// - `# <title>`
/// - metadata line (`**Channel:** … · **Uploaded:** … · **Duration:** …`)
/// - source line (watch URL + transcription provenance + RTF)
/// - an honesty note (machine transcription; approximate timestamps)
/// - optional quoted description intro (first
///   [`DESCRIPTION_INTRO_CHARS`] chars), if a description is present
/// - `---`
/// - timestamped paragraphs, each led by a deep-linked `**[m:ss](url)**` (plus
///   `SPEAKER_NN:` when diarized)
/// - `---`
/// - a footer provenance line
///
/// When there are no segments, the body is an honest "no speech detected"
/// note instead of paragraphs.
#[must_use]
pub fn render_markdown(input: &RenderInput<'_>) -> String {
    let v = &input.video;
    let r = &input.run;
    let mut out = String::new();

    // H1.
    out.push_str("# ");
    out.push_str(v.title.trim());
    out.push_str("\n\n");

    // Metadata line: channel · uploaded · duration.
    let mut meta_parts: Vec<String> = Vec::new();
    if let Some(channel) = v.channel.as_deref().filter(|c| !c.trim().is_empty()) {
        meta_parts.push(format!("**Channel:** {channel}"));
    }
    if let Some(date) = v.upload_date.as_deref().filter(|d| !d.trim().is_empty()) {
        meta_parts.push(format!("**Uploaded:** {}", format_upload_date(date)));
    }
    if let Some(dur) = v.duration_sec {
        meta_parts.push(format!("**Duration:** {}", format_duration(dur)));
    }
    if !meta_parts.is_empty() {
        out.push_str(&meta_parts.join(" · "));
        out.push('\n');
    }

    // Source / provenance line.
    let provider = transcribed_provider(r);
    let mut src_parts: Vec<String> = vec![format!(
        "**Source:** [{}]({})",
        display_url(&v.webpage_url),
        v.webpage_url
    )];
    src_parts.push(format!("**Transcribed:** {provider}"));
    if let Some(rtf) = r.rtf {
        src_parts.push(format!("RTF {}", format_rtf(rtf)));
    }
    out.push_str(&src_parts.join(" · "));
    out.push_str("\n\n");

    // Honesty note.
    out.push_str(
        "> Note: machine transcription; timestamps are approximate and deep-link into the video.\n\n",
    );

    // Optional description intro.
    if let Some(intro) = description_intro(v.description.as_deref()) {
        out.push_str("> ");
        out.push_str(&intro);
        out.push_str("\n\n");
    }

    out.push_str("---\n\n");

    // Body.
    if input.segments.is_empty() {
        out.push_str("_No speech detected in this video._\n");
    } else {
        let paragraphs = group_paragraphs(input.segments);
        let mut wrote_any = false;
        for p in &paragraphs {
            let text = paragraph_text(p);
            if text.is_empty() {
                continue;
            }
            let label = format_timestamp_label(p.start_sec);
            let link = deep_link(&v.id, p.start_sec);
            out.push_str(&format!("**[{label}]({link})**"));
            if let Some(spk) = p.speaker.filter(|s| !s.trim().is_empty()) {
                out.push(' ');
                out.push_str(spk);
                out.push(':');
            }
            out.push(' ');
            out.push_str(&text);
            out.push_str("\n\n");
            wrote_any = true;
        }
        if !wrote_any {
            out.push_str("_No speech detected in this video._\n\n");
        }
    }

    out.push_str("---\n\n");

    // Footer provenance.
    out.push('_');
    out.push_str(&footer_line(r));
    out.push_str("_\n");

    out
}

/// Provenance string for the source line:
/// `franken_whisper <version> (<engine>, <model>)` (version omitted if absent).
fn transcribed_provider(r: &RenderRun) -> String {
    match r.version_tag.as_deref().filter(|t| !t.trim().is_empty()) {
        Some(tag) => format!("franken_whisper {tag} ({}, {})", r.engine, r.model),
        None => format!("franken_whisper ({}, {})", r.engine, r.model),
    }
}

/// Footer line: full provenance including backend, wall time, and RTF.
fn footer_line(r: &RenderRun) -> String {
    let version = match r.version_tag.as_deref().filter(|t| !t.trim().is_empty()) {
        Some(tag) => format!("franken_whisper {tag}"),
        None => "franken_whisper".to_owned(),
    };
    let wall = format_wall(r.wall_ms);
    let mut s = format!(
        "Transcribed by {version} ({}, {}, {}) in {wall}",
        r.engine, r.model, r.backend
    );
    if let Some(rtf) = r.rtf {
        s.push_str(&format!(" — RTF {}", format_rtf(rtf)));
    }
    s
}

/// Wall-clock duration as a compact human string (e.g. `4.20s`, `1m03s`).
fn format_wall(wall_ms: u64) -> String {
    let total_secs = wall_ms as f64 / 1000.0;
    if total_secs < 60.0 {
        format!("{total_secs:.2}s")
    } else {
        let secs = wall_ms / 1000;
        let m = secs / 60;
        let s = secs % 60;
        format!("{m}m{s:02}s")
    }
}

/// RTF formatted to two decimals (e.g. `0.04`).
fn format_rtf(rtf: f64) -> String {
    format!("{rtf:.2}")
}

/// Strip the scheme from a URL for display text (keeps it tidy in the link).
fn display_url(url: &str) -> String {
    url.strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url)
        .to_owned()
}

/// Produce the quoted description intro (first [`DESCRIPTION_INTRO_CHARS`]
/// chars, newline-flattened, ellipsized if truncated). `None` when there is no
/// non-empty description.
fn description_intro(description: Option<&str>) -> Option<String> {
    let desc = description?.trim();
    if desc.is_empty() {
        return None;
    }
    // Flatten newlines/runs of whitespace so the blockquote stays one line.
    let flat: String = desc.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.is_empty() {
        return None;
    }
    let mut intro: String = flat
        .char_indices()
        .take_while(|&(i, _)| i < DESCRIPTION_INTRO_CHARS)
        .map(|(_, c)| c)
        .collect();
    if intro.len() < flat.len() {
        intro.push('…');
    }
    Some(intro)
}

// ---------------------------------------------------------------------------
// JSON renderer
// ---------------------------------------------------------------------------

/// Render the JSON document for a video, per the epic schema.
///
/// Shape:
/// ```text
/// {
///   "video": { "id", "title", "channel"?, "uploader"?, "upload_date"?,
///              "duration"?, "webpage_url", "description"? },
///   "run":   { "model", "engine", "backend", "version_tag"?, "started",
///              "wall_ms", "rtf"? },
///   "utterances": [ { "i", "start_sec", "end_sec", "text",
///                     "confidence", "speaker"? }, ... ]
/// }
/// ```
///
/// `utterances` is one entry **per raw segment** (count is preserved); it is
/// not the Markdown paragraph grouping. `None`-valued optional video/run fields
/// are omitted entirely rather than serialized as `null`. Per-utterance numeric
/// fields (`start_sec`, `end_sec`, `confidence`) are passed through verbatim,
/// including `null`, so the raw segment shape round-trips faithfully.
#[must_use]
pub fn render_json(input: &RenderInput<'_>) -> Value {
    let v = &input.video;
    let r = &input.run;

    let mut video = Map::new();
    video.insert("id".into(), json!(v.id));
    video.insert("title".into(), json!(v.title));
    insert_opt_str(&mut video, "channel", v.channel.as_deref());
    insert_opt_str(&mut video, "uploader", v.uploader.as_deref());
    insert_opt_str(&mut video, "upload_date", v.upload_date.as_deref());
    if let Some(d) = v.duration_sec {
        video.insert("duration".into(), json!(d));
    }
    video.insert("webpage_url".into(), json!(v.webpage_url));
    insert_opt_str(&mut video, "description", v.description.as_deref());

    let mut run = Map::new();
    run.insert("model".into(), json!(r.model));
    run.insert("engine".into(), json!(r.engine));
    run.insert("backend".into(), json!(r.backend));
    insert_opt_str(&mut run, "version_tag", r.version_tag.as_deref());
    run.insert("started".into(), json!(r.started_rfc3339));
    run.insert("wall_ms".into(), json!(r.wall_ms));
    if let Some(rtf) = r.rtf {
        run.insert("rtf".into(), json!(rtf));
    }

    let utterances: Vec<Value> = input
        .segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            let mut u = Map::new();
            u.insert("i".into(), json!(i));
            u.insert("start_sec".into(), json!(seg.start_sec));
            u.insert("end_sec".into(), json!(seg.end_sec));
            u.insert("text".into(), json!(seg.text));
            // Confidence is passed through verbatim, including null.
            u.insert("confidence".into(), json!(seg.confidence));
            insert_opt_str(&mut u, "speaker", seg.speaker.as_deref());
            Value::Object(u)
        })
        .collect();

    json!({
        "video": Value::Object(video),
        "run": Value::Object(run),
        "utterances": utterances,
    })
}

/// Insert a string field only when `Some` and non-empty, omitting it otherwise.
fn insert_opt_str(map: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(s) = value.filter(|s| !s.is_empty()) {
        map.insert(key.into(), json!(s));
    }
}

// ---------------------------------------------------------------------------
// Atomic write
// ---------------------------------------------------------------------------

/// Atomically write `contents` to `path`.
///
/// Writes to a uniquely-named temp file in the **same directory** as `path`
/// (so the final rename stays on one filesystem and is atomic), flushes and
/// fsyncs it, then renames it over `path`. On any failure the temp file is
/// cleaned up and the original file at `path`, if any, is left untouched.
///
/// # Errors
///
/// Returns [`FwError::Io`](crate::error::FwError::Io) if the temp file cannot be
/// created/written/synced or the rename fails.
pub fn write_atomic(path: impl AsRef<Path>, contents: &str) -> FwResult<()> {
    use std::io::Write;

    let path = path.as_ref();
    let dir = match path.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(d) => d.to_path_buf(),
        None => std::path::PathBuf::from("."),
    };

    // Same-dir temp file; tempfile cleans itself up on drop unless persisted.
    let mut tmp = tempfile::Builder::new()
        .prefix(".fw-render-")
        .suffix(".tmp")
        .tempfile_in(&dir)?;

    tmp.write_all(contents.as_bytes())?;
    tmp.flush()?;
    tmp.as_file().sync_all()?;

    // Atomic rename into place. On error the NamedTempFile is returned to us
    // (inside PersistError) and dropped here, removing the temp file; the
    // original file at `path` is never touched.
    tmp.persist(path)
        .map_err(|e| crate::error::FwError::Io(e.error))?;

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn seg(start: f64, end: f64, text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: Some(start),
            end_sec: Some(end),
            text: text.to_owned(),
            speaker: None,
            confidence: Some(0.9),
        }
    }

    fn seg_spk(start: f64, end: f64, text: &str, spk: &str, conf: f64) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec: Some(start),
            end_sec: Some(end),
            text: text.to_owned(),
            speaker: Some(spk.to_owned()),
            confidence: Some(conf),
        }
    }

    fn sample_video() -> RenderVideo {
        RenderVideo {
            id: "dQw4w9WgXcQ".to_owned(),
            title: "Sample Talk".to_owned(),
            channel: Some("Example Channel".to_owned()),
            uploader: Some("Example Channel".to_owned()),
            upload_date: Some("20240115".to_owned()),
            duration_sec: Some(3725.0),
            webpage_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".to_owned(),
            description: None,
        }
    }

    fn sample_run() -> RenderRun {
        RenderRun {
            model: "large-v3".to_owned(),
            engine: "native".to_owned(),
            backend: "cpu".to_owned(),
            version_tag: Some("v0.2.0".to_owned()),
            started_rfc3339: "2026-06-06T12:00:00Z".to_owned(),
            wall_ms: 4200,
            rtf: Some(0.04),
        }
    }

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/youtube")
    }

    /// Compare `actual` against the committed golden file `name`.
    ///
    /// Set `FW_UPDATE_GOLDEN=1` to (re)write goldens from current output.
    fn assert_golden(name: &str, actual: &str) {
        let path = fixture_dir().join(name);
        if std::env::var_os("FW_UPDATE_GOLDEN").is_some() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, actual).unwrap();
            return;
        }
        let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "missing golden {}: {e}. Re-run with FW_UPDATE_GOLDEN=1 to create it.",
                path.display()
            )
        });
        assert_eq!(
            actual, expected,
            "golden mismatch for {name}; re-run with FW_UPDATE_GOLDEN=1 if intentional"
        );
    }

    // --- formatting unit tests ------------------------------------------

    #[test]
    fn timestamp_label_minutes_under_hour() {
        assert_eq!(format_timestamp_label(0.0), "0:00");
        assert_eq!(format_timestamp_label(83.4), "1:23");
        assert_eq!(format_timestamp_label(599.9), "9:59");
        assert_eq!(format_timestamp_label(3599.0), "59:59");
    }

    #[test]
    fn timestamp_label_hours_at_rollover() {
        assert_eq!(format_timestamp_label(3600.0), "1:00:00");
        assert_eq!(format_timestamp_label(3661.0), "1:01:01");
        assert_eq!(format_timestamp_label(3725.7), "1:02:05");
    }

    #[test]
    fn upload_date_formatting() {
        assert_eq!(format_upload_date("20240115"), "2024-01-15");
        assert_eq!(format_upload_date("notadate"), "notadate");
        assert_eq!(format_upload_date("2024-01-15"), "2024-01-15");
    }

    #[test]
    fn deep_link_floors_start() {
        assert_eq!(deep_link("abc", 0.0), "https://youtu.be/abc?t=0");
        assert_eq!(deep_link("abc", 83.9), "https://youtu.be/abc?t=83");
        assert_eq!(deep_link("abc", -5.0), "https://youtu.be/abc?t=0");
    }

    // --- paragraph grouping ---------------------------------------------

    #[test]
    fn grouping_breaks_on_gap_over_threshold() {
        // Two segments < 2.5s apart stay together; a > 2.5s gap splits.
        let segs = vec![
            seg(0.0, 1.0, "one"),
            seg(2.0, 3.0, "two"),   // gap 1.0 -> same paragraph
            seg(6.0, 7.0, "three"), // gap 3.0 -> new paragraph
        ];
        let paras = group_paragraphs(&segs);
        assert_eq!(paras.len(), 2);
        assert_eq!(paras[0].segments.len(), 2);
        assert_eq!(paras[1].segments.len(), 1);
    }

    #[test]
    fn grouping_gap_exactly_at_threshold_does_not_break() {
        // Gap of exactly 2.5s is NOT > 2.5 -> stays together.
        let segs = vec![seg(0.0, 1.0, "a"), seg(3.5, 4.0, "b")];
        let paras = group_paragraphs(&segs);
        assert_eq!(paras.len(), 1);
    }

    #[test]
    fn grouping_breaks_on_speaker_change() {
        let segs = vec![
            seg_spk(0.0, 1.0, "hi", "SPEAKER_00", 0.9),
            seg_spk(1.2, 2.0, "there", "SPEAKER_00", 0.9),
            seg_spk(2.1, 3.0, "hello", "SPEAKER_01", 0.9),
        ];
        let paras = group_paragraphs(&segs);
        assert_eq!(paras.len(), 2);
        assert_eq!(paras[0].speaker, Some("SPEAKER_00"));
        assert_eq!(paras[1].speaker, Some("SPEAKER_01"));
    }

    #[test]
    fn grouping_breaks_on_word_cap() {
        // One long segment (>120 words) followed by another with a tiny gap:
        // the word cap forces a split even though the gap is small.
        let long = "word ".repeat(130);
        let segs = vec![seg(0.0, 10.0, long.trim()), seg(10.1, 11.0, "tail")];
        let paras = group_paragraphs(&segs);
        assert_eq!(paras.len(), 2, "word cap should force a paragraph split");
        assert_eq!(paras[1].segments.len(), 1);
    }

    // --- markdown golden tests ------------------------------------------

    #[test]
    fn golden_markdown_undiarized_three_paragraphs() {
        // Three paragraphs via two > 2.5s gaps.
        let segs = vec![
            seg(0.0, 2.0, "Welcome to the show."),
            seg(2.4, 4.0, "Today we cover renderers."),
            seg(20.0, 22.0, "First, the markdown format."),
            seg(22.5, 24.0, "It groups segments into paragraphs."),
            seg(83.0, 85.0, "Finally, the JSON schema."),
        ];
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
        };
        let md = render_markdown(&input);
        // Sanity: exactly three body paragraphs (lead-in markers).
        assert_eq!(md.matches("**[").count(), 3);
        assert_golden("markdown_undiarized.md", &md);
    }

    #[test]
    fn golden_markdown_diarized_two_speakers() {
        let segs = vec![
            seg_spk(0.0, 2.0, "Thanks for joining.", "SPEAKER_00", 0.95),
            seg_spk(2.3, 4.0, "How are you?", "SPEAKER_00", 0.91),
            seg_spk(4.5, 6.0, "Doing great, thanks.", "SPEAKER_01", 0.88),
            seg_spk(6.2, 8.0, "Glad to be here.", "SPEAKER_01", 0.9),
        ];
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
        };
        let md = render_markdown(&input);
        assert!(md.contains("SPEAKER_00:"));
        assert!(md.contains("SPEAKER_01:"));
        assert_golden("markdown_diarized.md", &md);
    }

    #[test]
    fn golden_markdown_hms_rollover() {
        // A segment at 3661s must render as 1:01:01.
        let segs = vec![
            seg(0.0, 2.0, "Intro near the start."),
            seg(3661.0, 3663.0, "Now well past the one hour mark."),
        ];
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
        };
        let md = render_markdown(&input);
        assert!(md.contains("**[1:01:01]"));
        assert!(md.contains("?t=3661"));
        assert_golden("markdown_hms_rollover.md", &md);
    }

    #[test]
    fn golden_markdown_empty_segments() {
        let segs: Vec<TranscriptionSegment> = Vec::new();
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
        };
        let md = render_markdown(&input);
        assert!(md.contains("No speech detected"));
        assert!(md.contains("# Sample Talk"));
        assert_golden("markdown_empty.md", &md);
    }

    #[test]
    fn golden_markdown_word_cap_split() {
        // One contiguous run of segments with no large gaps and one speaker,
        // long enough to exceed the 120-word cap and split into two paragraphs.
        let mut segs = Vec::new();
        let mut t = 0.0;
        for i in 0..20 {
            // ~12 words each => ~240 words total, no gap > 2.5s.
            let text =
                format!("segment number {i} has several words in it to build word count up now");
            segs.push(seg(t, t + 1.0, &text));
            t += 1.2;
        }
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
        };
        let md = render_markdown(&input);
        // No gaps and one speaker, so any split is purely the word cap.
        let paras = md.matches("**[").count();
        assert!(
            paras >= 2,
            "word cap should produce >=2 paragraphs, got {paras}"
        );
        assert_golden("markdown_word_cap.md", &md);
    }

    #[test]
    fn golden_markdown_with_description_intro() {
        let mut video = sample_video();
        video.description = Some(
            "This is the first line of the description.\n\nIt has multiple paragraphs and \
             links and timestamps that we deliberately truncate so the markdown header stays \
             tight and readable rather than dumping the entire video description verbatim into \
             the transcript document which would be far too long."
                .to_owned(),
        );
        let segs = vec![seg(0.0, 2.0, "Hello world.")];
        let input = RenderInput {
            video,
            run: sample_run(),
            segments: &segs,
        };
        let md = render_markdown(&input);
        assert!(md.contains('…'), "long description should be ellipsized");
        assert_golden("markdown_description.md", &md);
    }

    // --- JSON golden + schema tests -------------------------------------

    #[test]
    fn golden_json_diarized() {
        let segs = vec![
            seg_spk(0.0, 2.0, "Thanks for joining.", "SPEAKER_00", 0.95),
            seg_spk(2.3, 4.0, "How are you?", "SPEAKER_00", 0.91),
            seg_spk(4.5, 6.0, "Doing great, thanks.", "SPEAKER_01", 0.88),
        ];
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
        };
        let val = render_json(&input);
        let pretty = serde_json::to_string_pretty(&val).unwrap();
        assert_golden("json_diarized.json", &pretty);
    }

    #[test]
    fn json_omits_none_channel_and_optional_fields() {
        let mut video = sample_video();
        video.channel = None;
        video.uploader = None;
        video.upload_date = None;
        video.duration_sec = None;
        video.description = None;
        let mut run = sample_run();
        run.version_tag = None;
        run.rtf = None;
        let segs = vec![seg(0.0, 1.0, "hi")];
        let input = RenderInput {
            video,
            run,
            segments: &segs,
        };
        let val = render_json(&input);
        let v = val.get("video").unwrap().as_object().unwrap();
        assert!(!v.contains_key("channel"), "None channel must be omitted");
        assert!(!v.contains_key("uploader"));
        assert!(!v.contains_key("upload_date"));
        assert!(!v.contains_key("duration"));
        assert!(!v.contains_key("description"));
        // Required fields stay.
        assert!(v.contains_key("id"));
        assert!(v.contains_key("title"));
        assert!(v.contains_key("webpage_url"));
        let r = val.get("run").unwrap().as_object().unwrap();
        assert!(!r.contains_key("version_tag"));
        assert!(!r.contains_key("rtf"));
        assert!(r.contains_key("model"));
        assert!(r.contains_key("wall_ms"));
    }

    #[test]
    fn json_utterance_count_equals_segment_count() {
        let segs = vec![
            seg(0.0, 1.0, "a"),
            seg(1.0, 2.0, "b"),
            seg(10.0, 11.0, "c"),
            seg(11.0, 12.0, "d"),
        ];
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
        };
        let val = render_json(&input);
        let utts = val.get("utterances").unwrap().as_array().unwrap();
        assert_eq!(
            utts.len(),
            segs.len(),
            "utterances are raw segments, not paragraphs"
        );
        // Indices are sequential.
        for (i, u) in utts.iter().enumerate() {
            assert_eq!(u.get("i").unwrap().as_u64().unwrap() as usize, i);
        }
    }

    #[test]
    fn json_confidence_passthrough_and_speaker_omission() {
        let segs = vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "with conf".to_owned(),
                speaker: None,
                confidence: Some(0.731),
            },
            TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: "no conf".to_owned(),
                speaker: None,
                confidence: None,
            },
        ];
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
        };
        let val = render_json(&input);
        let utts = val.get("utterances").unwrap().as_array().unwrap();
        // Confidence passed through exactly.
        assert_eq!(utts[0].get("confidence").unwrap().as_f64().unwrap(), 0.731);
        // None confidence is present as null (explicit passthrough).
        assert!(utts[1].get("confidence").unwrap().is_null());
        // No speaker -> field omitted entirely.
        assert!(!utts[0].as_object().unwrap().contains_key("speaker"));
    }

    // --- atomic write ----------------------------------------------------

    #[test]
    fn write_atomic_succeeds_and_leaves_no_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.md");
        write_atomic(&target, "hello world").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "hello world");
        // No leftover .tmp files in the directory.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| {
                let n = e.file_name();
                let n = n.to_string_lossy();
                n.contains(".tmp") || n.starts_with(".fw-render-")
            })
            .collect();
        assert!(
            leftovers.is_empty(),
            "no temp files should remain: {leftovers:?}"
        );
    }

    #[test]
    fn write_atomic_overwrites_existing() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.json");
        std::fs::write(&target, "old contents").unwrap();
        write_atomic(&target, "new contents").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new contents");
    }

    #[test]
    fn write_atomic_failure_leaves_original_intact() {
        // Simulate failure via a missing parent directory: temp-file creation
        // fails, write returns an error, and no target is produced. An
        // unrelated pre-existing file is never disturbed.
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("keep.md");
        std::fs::write(&good, "ORIGINAL").unwrap();

        let bad_target = dir.path().join("does_not_exist_subdir").join("x.md");
        let err = write_atomic(&bad_target, "SHOULD NOT LAND");
        assert!(err.is_err(), "write into missing dir must fail");

        assert_eq!(std::fs::read_to_string(&good).unwrap(), "ORIGINAL");
        assert!(!bad_target.exists());
    }

    #[test]
    #[cfg(unix)]
    fn write_atomic_failure_preserves_same_path_original() {
        // Stronger atomicity check: an existing file at the exact target path
        // must survive a failed write to that same path. Force failure by
        // removing write permission on the directory after creating the
        // original, so temp-file creation fails.
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("locked.md");
        std::fs::write(&target, "ORIGINAL").unwrap();

        // Read+execute only: temp file creation will fail.
        let mut perms = std::fs::metadata(dir.path()).unwrap().permissions();
        perms.set_mode(0o500);
        std::fs::set_permissions(dir.path(), perms).unwrap();

        let res = write_atomic(&target, "REPLACEMENT");

        // Restore perms so tempdir cleanup works regardless of outcome.
        let mut restore = std::fs::metadata(dir.path()).unwrap().permissions();
        restore.set_mode(0o700);
        std::fs::set_permissions(dir.path(), restore).unwrap();

        assert!(res.is_err(), "write into read-only dir must fail");
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "ORIGINAL",
            "original must be intact after failed write"
        );
    }
}
