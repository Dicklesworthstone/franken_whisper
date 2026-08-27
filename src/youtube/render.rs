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

/// Decode-window evidence projected from an in-process native backend.
///
/// These fields intentionally mirror the stable `native-v2` raw-output
/// contract emitted by every native backend. Keeping them typed prevents the
/// YouTube JSON artifact from silently dropping or reshaping decoder evidence.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RenderWindowStats {
    /// Start offset of the decode window in seconds.
    pub window_offset_sec: f64,
    /// Number of decoded tokens in the window.
    pub tokens: u64,
    /// Mean token log probability for the window.
    pub avg_logprob: f64,
    /// Model probability that the window contains no speech.
    pub no_speech_prob: f64,
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
    /// Native decode-window evidence. External backends supply an empty slice.
    pub windows: &'a [RenderWindowStats],
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a timestamp for the Markdown body label.
///
/// `m:ss` below one hour (e.g. `1:23`), `h:mm:ss` at or after one hour
/// (e.g. `1:01:01`). Negative / non-finite inputs are clamped to zero.
#[cfg(test)]
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

/// Append a paragraph timestamp and YouTube deep link directly to `out`.
fn push_timestamp_link(out: &mut String, id: &str, start_sec: f64) {
    use std::fmt::Write as _;

    let total = floor_secs(start_sec);
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        write!(
            out,
            "**[{h}:{m:02}:{s:02}](https://youtu.be/{id}?t={total})**"
        )
        .expect("writing to a String cannot fail");
    } else {
        write!(out, "**[{m}:{s:02}](https://youtu.be/{id}?t={total})**")
            .expect("writing to a String cannot fail");
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

/// Sanitize a segment timestamp before JSON serialization.
///
/// Finite, real timestamps (including an exact `0.0`) pass through unchanged.
/// Non-finite values (`NaN`/`±Inf`) and subnormal/denormal magnitudes — e.g. the
/// `2.225e-308` a collapsed alignment can emit — are mapped to `None`, so the
/// JSON sidecar never carries a bogus denormal (or a `NaN` silently coerced to
/// `null` with the underlying corruption unrepaired).
fn sanitize_timestamp(value: Option<f64>) -> Option<f64> {
    let v = value?;
    if v == 0.0 || (v.is_finite() && v.abs() >= f64::MIN_POSITIVE.next_up()) {
        Some(v)
    } else {
        None
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

/// Append the title as a single normalized Markdown H1.
fn push_title_heading(out: &mut String, title: &str) {
    out.push_str("# ");
    let mut first = true;
    for word in title.split_whitespace() {
        if !first {
            out.push(' ');
        }
        out.push_str(word);
        first = false;
    }
    out.push_str("\n\n");
}

/// Append the optional video metadata as one Markdown line.
fn push_metadata_line(out: &mut String, video: &RenderVideo) {
    let mut has_part = false;

    if let Some(channel) = video
        .channel
        .as_deref()
        .filter(|channel| !channel.trim().is_empty())
    {
        out.push_str("**Channel:** ");
        out.push_str(channel);
        has_part = true;
    }
    if let Some(date) = video
        .upload_date
        .as_deref()
        .filter(|date| !date.trim().is_empty())
    {
        if has_part {
            out.push_str(" · ");
        }
        out.push_str("**Uploaded:** ");
        out.push_str(&format_upload_date(date));
        has_part = true;
    }
    if let Some(duration) = video.duration_sec {
        if has_part {
            out.push_str(" · ");
        }
        out.push_str("**Duration:** ");
        out.push_str(&format_duration(duration));
        has_part = true;
    }

    if has_part {
        out.push('\n');
    }
}

/// Append the source URL and transcription provenance as one Markdown line.
fn push_source_line(out: &mut String, video: &RenderVideo, run: &RenderRun) {
    let display_url = video
        .webpage_url
        .strip_prefix("https://")
        .or_else(|| video.webpage_url.strip_prefix("http://"))
        .unwrap_or(&video.webpage_url);

    out.push_str("**Source:** [");
    out.push_str(display_url);
    out.push_str("](");
    out.push_str(&video.webpage_url);
    out.push_str(") · **Transcribed:** franken_whisper");
    if let Some(tag) = run
        .version_tag
        .as_deref()
        .filter(|tag| !tag.trim().is_empty())
    {
        out.push(' ');
        out.push_str(tag);
    }
    out.push_str(" (");
    out.push_str(&run.engine);
    out.push_str(", ");
    out.push_str(&run.model);
    out.push(')');
    if let Some(rtf) = run.rtf {
        out.push_str(" · RTF ");
        out.push_str(&format_rtf(rtf));
    }
    out.push_str("\n\n");
}

/// Deep-link URL into the video at the given start second.
#[cfg(test)]
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
    /// Paragraphs are contiguous runs of the input, so each is represented as a
    /// borrowed slice of the original segments — zero per-paragraph allocation.
    segments: &'a [TranscriptionSegment],
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
    // Track the current paragraph as a half-open index range `[para_start, i)`
    // into `segments`; on each break we push the borrowed slice. This is
    // byte-identical to owning a `Vec<&Seg>` per paragraph (same contiguous
    // segment sequence) but allocates nothing per paragraph.
    let mut para_start: Option<usize> = None;
    let mut current_start_sec = 0.0_f64;
    let mut current_speaker: Option<&str> = None;
    let mut current_words = 0usize;
    let mut prev_end: Option<f64> = None;

    for (i, seg) in segments.iter().enumerate() {
        let speaker = seg.speaker.as_deref();
        let start = seg_start(seg);
        let seg_words = word_count(&seg.text);

        let gap_break = prev_end.is_some_and(|pe| start - pe > PARAGRAPH_GAP_SEC);
        let speaker_break = para_start.is_some() && current_speaker != speaker;
        let word_break = para_start.is_some() && current_words >= PARAGRAPH_WORD_CAP;

        if para_start.is_none() || gap_break || speaker_break || word_break {
            if let Some(s) = para_start.take() {
                paragraphs.push(Paragraph {
                    start_sec: current_start_sec,
                    speaker: current_speaker,
                    segments: &segments[s..i],
                });
            }
            para_start = Some(i);
            current_start_sec = start;
            current_speaker = speaker;
            current_words = seg_words;
        } else {
            current_words += seg_words;
        }

        prev_end = Some(seg_end(seg));
    }

    if let Some(s) = para_start.take() {
        paragraphs.push(Paragraph {
            start_sec: current_start_sec,
            speaker: current_speaker,
            segments: &segments[s..],
        });
    }
    paragraphs
}

/// Join a paragraph's segment texts into a single trimmed prose string,
/// collapsing inter-segment whitespace to a single space.
#[cfg(test)]
fn paragraph_text(p: &Paragraph<'_>) -> String {
    let mut out = String::new();
    for seg in p.segments {
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

/// Append a paragraph's trimmed segment texts directly to `out`.
///
/// Returns whether at least one non-empty segment was written. Whitespace
/// between non-empty segments is collapsed to one ASCII space, matching the
/// historical temporary-`String` path.
fn push_paragraph_text(out: &mut String, p: &Paragraph<'_>) -> bool {
    let mut wrote_text = false;
    for seg in p.segments {
        let piece = seg.text.trim();
        if piece.is_empty() {
            continue;
        }
        if wrote_text {
            out.push(' ');
        }
        out.push_str(piece);
        wrote_text = true;
    }
    wrote_text
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

    // H1. Collapse internal whitespace/newlines so a multi-line title cannot
    // break the heading into the body.
    push_title_heading(&mut out, &v.title);

    // Metadata line: channel · uploaded · duration.
    push_metadata_line(&mut out, v);

    // Source / provenance line.
    push_source_line(&mut out, v, r);

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
            let paragraph_start = out.len();
            push_timestamp_link(&mut out, &v.id, p.start_sec);
            if let Some(spk) = p.speaker.filter(|s| !s.trim().is_empty()) {
                out.push(' ');
                out.push_str(spk);
                out.push(':');
            }
            out.push(' ');
            if !push_paragraph_text(&mut out, p) {
                out.truncate(paragraph_start);
                continue;
            }
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

/// Produce the quoted description intro (first [`DESCRIPTION_INTRO_CHARS`]
/// chars, newline-flattened, ellipsized if truncated). `None` when there is no
/// non-empty description.
fn description_intro(description: Option<&str>) -> Option<String> {
    let desc = description?.trim();
    if desc.is_empty() {
        return None;
    }

    // Flatten only the prefix the Markdown can retain. YouTube descriptions
    // may be thousands of characters long, but the artifact needs at most the
    // first 280 normalized characters plus an ellipsis.
    let mut intro = String::with_capacity(DESCRIPTION_INTRO_CHARS + 3);
    let mut chars_written = 0usize;
    let mut first_word = true;
    let mut truncated = false;

    'words: for word in desc.split_whitespace() {
        if !first_word {
            if chars_written == DESCRIPTION_INTRO_CHARS {
                truncated = true;
                break;
            }
            intro.push(' ');
            chars_written += 1;
        }
        first_word = false;

        for ch in word.chars() {
            if chars_written == DESCRIPTION_INTRO_CHARS {
                truncated = true;
                break 'words;
            }
            intro.push(ch);
            chars_written += 1;
        }
    }

    if truncated {
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
///   "windows": [ { "window_offset_sec", "tokens", "avg_logprob",
///                  "no_speech_prob" }, ... ],
///   "utterances": [ { "i", "start_sec", "end_sec", "text",
///                     "confidence", "speaker"? }, ... ]
/// }
/// ```
///
/// `utterances` is one entry **per raw segment** (count is preserved); it is
/// not the Markdown paragraph grouping. `None`-valued optional video/run fields
/// are omitted entirely rather than serialized as `null`. `confidence` is passed
/// through verbatim (including `null`); `start_sec`/`end_sec` pass through
/// verbatim for finite, real values but are `null`-ed when non-finite or
/// subnormal/denormal (defense-in-depth so a collapsed alignment never emits a
/// bogus denormal timestamp into the artifact).
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
            u.insert("start_sec".into(), json!(sanitize_timestamp(seg.start_sec)));
            u.insert("end_sec".into(), json!(sanitize_timestamp(seg.end_sec)));
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
        "windows": input.windows,
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
/// fsyncs it, renames it over `path`, then fsyncs the parent directory. A
/// failure before rename leaves the original file untouched; a parent-sync
/// failure is reported after the replacement is atomically visible so callers
/// do not mistake an unconfirmed directory entry for a durable publication.
///
/// # Errors
///
/// Returns [`FwError::Io`](crate::error::FwError::Io) if the temp file cannot be
/// created/written/synced, the rename fails, or the parent-directory durability
/// barrier fails.
pub fn write_atomic(path: impl AsRef<Path>, contents: &str) -> FwResult<()> {
    write_atomic_with_parent_sync(path.as_ref(), contents, sync_parent_dir)
}

fn write_atomic_with_parent_sync(
    path: &Path,
    contents: &str,
    sync_parent: impl FnOnce(&Path) -> std::io::Result<()>,
) -> FwResult<()> {
    use std::io::Write;

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
    sync_parent(&dir)?;

    Ok(())
}

#[cfg(unix)]
fn sync_parent_dir(parent: &Path) -> std::io::Result<()> {
    std::fs::File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_dir(_parent: &Path) -> std::io::Result<()> {
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    #[derive(Debug)]
    struct PairedPerfStats {
        median: f64,
        p10: f64,
        p90: f64,
        wins: usize,
    }

    fn historical_description_intro(description: Option<&str>) -> Option<String> {
        let desc = description?.trim();
        if desc.is_empty() {
            return None;
        }
        let flat = desc.split_whitespace().collect::<Vec<_>>().join(" ");
        if flat.is_empty() {
            return None;
        }
        let mut intro = flat
            .chars()
            .take(DESCRIPTION_INTRO_CHARS)
            .collect::<String>();
        if flat.chars().count() > DESCRIPTION_INTRO_CHARS {
            intro.push('…');
        }
        Some(intro)
    }

    fn measure_description_intro<const HISTORICAL: bool>(
        description: &str,
        iterations: usize,
    ) -> Duration {
        let started = Instant::now();
        let mut checksum = 0usize;
        for _ in 0..iterations {
            let intro = if HISTORICAL {
                historical_description_intro(Some(black_box(description)))
            } else {
                description_intro(Some(black_box(description)))
            };
            checksum ^= intro.as_ref().map_or(0, String::len);
            black_box(&intro);
        }
        black_box(checksum);
        started.elapsed()
    }

    fn paired_description_ratios<const BASE_HISTORICAL: bool, const TEST_HISTORICAL: bool>(
        description: &str,
        iterations: usize,
        repetitions: usize,
    ) -> Vec<f64> {
        let mut ratios = Vec::with_capacity(repetitions);
        for repetition in 0..repetitions {
            let (base, test) = if repetition % 2 == 0 {
                (
                    measure_description_intro::<BASE_HISTORICAL>(description, iterations),
                    measure_description_intro::<TEST_HISTORICAL>(description, iterations),
                )
            } else {
                let test = measure_description_intro::<TEST_HISTORICAL>(description, iterations);
                let base = measure_description_intro::<BASE_HISTORICAL>(description, iterations);
                (base, test)
            };
            ratios.push(base.as_secs_f64() / test.as_secs_f64());
        }
        ratios
    }

    fn paired_perf_stats(ratios: &[f64]) -> PairedPerfStats {
        let mut sorted = ratios.to_vec();
        sorted.sort_by(f64::total_cmp);
        let last = sorted.len() - 1;
        PairedPerfStats {
            median: sorted[sorted.len() / 2],
            p10: sorted[last / 10],
            p90: sorted[last * 9 / 10],
            wins: ratios.iter().filter(|ratio| **ratio > 1.0).count(),
        }
    }

    fn format_ratios(ratios: &[f64]) -> String {
        ratios
            .iter()
            .map(|ratio| format!("{ratio:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn historical_push_title_heading(out: &mut String, title: &str) {
        out.push_str("# ");
        out.push_str(&title.split_whitespace().collect::<Vec<_>>().join(" "));
        out.push_str("\n\n");
    }

    fn historical_push_metadata_line(out: &mut String, video: &RenderVideo) {
        let mut parts = Vec::new();
        if let Some(channel) = video
            .channel
            .as_deref()
            .filter(|channel| !channel.trim().is_empty())
        {
            parts.push(format!("**Channel:** {channel}"));
        }
        if let Some(date) = video
            .upload_date
            .as_deref()
            .filter(|date| !date.trim().is_empty())
        {
            parts.push(format!("**Uploaded:** {}", format_upload_date(date)));
        }
        if let Some(duration) = video.duration_sec {
            parts.push(format!("**Duration:** {}", format_duration(duration)));
        }
        if !parts.is_empty() {
            out.push_str(&parts.join(" · "));
            out.push('\n');
        }
    }

    fn historical_push_source_line(out: &mut String, video: &RenderVideo, run: &RenderRun) {
        let provider = match run
            .version_tag
            .as_deref()
            .filter(|tag| !tag.trim().is_empty())
        {
            Some(tag) => format!("franken_whisper {tag} ({}, {})", run.engine, run.model),
            None => format!("franken_whisper ({}, {})", run.engine, run.model),
        };
        let display_url = video
            .webpage_url
            .strip_prefix("https://")
            .or_else(|| video.webpage_url.strip_prefix("http://"))
            .unwrap_or(&video.webpage_url)
            .to_owned();
        let mut parts = vec![format!(
            "**Source:** [{}]({})",
            display_url, video.webpage_url
        )];
        parts.push(format!("**Transcribed:** {provider}"));
        if let Some(rtf) = run.rtf {
            parts.push(format!("RTF {}", format_rtf(rtf)));
        }
        out.push_str(&parts.join(" · "));
        out.push_str("\n\n");
    }

    fn historical_push_timestamp_link(out: &mut String, id: &str, start_sec: f64) {
        let label = format_timestamp_label(start_sec);
        let link = deep_link(id, start_sec);
        out.push_str(&format!("**[{label}]({link})**"));
    }

    fn historical_push_paragraph_text(out: &mut String, paragraph: &Paragraph<'_>) -> bool {
        let text = paragraph_text(paragraph);
        if text.is_empty() {
            false
        } else {
            out.push_str(&text);
            true
        }
    }

    fn measure_title_heading<const HISTORICAL: bool>(title: &str, iterations: usize) -> Duration {
        let started = Instant::now();
        let mut out = String::with_capacity(128);
        let mut checksum = 0usize;
        for _ in 0..iterations {
            out.clear();
            if HISTORICAL {
                historical_push_title_heading(&mut out, black_box(title));
            } else {
                push_title_heading(&mut out, black_box(title));
            }
            checksum ^= out.len();
            black_box(out.as_str());
        }
        black_box(checksum);
        started.elapsed()
    }

    fn paired_title_ratios<const BASE_HISTORICAL: bool, const TEST_HISTORICAL: bool>(
        title: &str,
        iterations: usize,
        repetitions: usize,
    ) -> Vec<f64> {
        let mut ratios = Vec::with_capacity(repetitions);
        for repetition in 0..repetitions {
            let (base, test) = if repetition % 2 == 0 {
                (
                    measure_title_heading::<BASE_HISTORICAL>(title, iterations),
                    measure_title_heading::<TEST_HISTORICAL>(title, iterations),
                )
            } else {
                let test = measure_title_heading::<TEST_HISTORICAL>(title, iterations);
                let base = measure_title_heading::<BASE_HISTORICAL>(title, iterations);
                (base, test)
            };
            ratios.push(base.as_secs_f64() / test.as_secs_f64());
        }
        ratios
    }

    fn measure_metadata_line<const HISTORICAL: bool>(
        video: &RenderVideo,
        iterations: usize,
    ) -> Duration {
        let started = Instant::now();
        let mut out = String::with_capacity(160);
        let mut checksum = 0usize;
        for _ in 0..iterations {
            out.clear();
            if HISTORICAL {
                historical_push_metadata_line(&mut out, black_box(video));
            } else {
                push_metadata_line(&mut out, black_box(video));
            }
            checksum ^= out.len();
            black_box(out.as_str());
        }
        black_box(checksum);
        started.elapsed()
    }

    fn paired_metadata_ratios<const BASE_HISTORICAL: bool, const TEST_HISTORICAL: bool>(
        video: &RenderVideo,
        iterations: usize,
        repetitions: usize,
    ) -> Vec<f64> {
        let mut ratios = Vec::with_capacity(repetitions);
        for repetition in 0..repetitions {
            let (base, test) = if repetition % 2 == 0 {
                (
                    measure_metadata_line::<BASE_HISTORICAL>(video, iterations),
                    measure_metadata_line::<TEST_HISTORICAL>(video, iterations),
                )
            } else {
                let test = measure_metadata_line::<TEST_HISTORICAL>(video, iterations);
                let base = measure_metadata_line::<BASE_HISTORICAL>(video, iterations);
                (base, test)
            };
            ratios.push(base.as_secs_f64() / test.as_secs_f64());
        }
        ratios
    }

    fn measure_source_line<const HISTORICAL: bool>(
        video: &RenderVideo,
        run: &RenderRun,
        iterations: usize,
    ) -> Duration {
        let started = Instant::now();
        let mut out = String::with_capacity(192);
        let mut checksum = 0usize;
        for _ in 0..iterations {
            out.clear();
            if HISTORICAL {
                historical_push_source_line(&mut out, black_box(video), black_box(run));
            } else {
                push_source_line(&mut out, black_box(video), black_box(run));
            }
            checksum ^= out.len();
            black_box(out.as_str());
        }
        black_box(checksum);
        started.elapsed()
    }

    fn paired_source_ratios<const BASE_HISTORICAL: bool, const TEST_HISTORICAL: bool>(
        video: &RenderVideo,
        run: &RenderRun,
        iterations: usize,
        repetitions: usize,
    ) -> Vec<f64> {
        let mut ratios = Vec::with_capacity(repetitions);
        for repetition in 0..repetitions {
            let (base, test) = if repetition % 2 == 0 {
                (
                    measure_source_line::<BASE_HISTORICAL>(video, run, iterations),
                    measure_source_line::<TEST_HISTORICAL>(video, run, iterations),
                )
            } else {
                let test = measure_source_line::<TEST_HISTORICAL>(video, run, iterations);
                let base = measure_source_line::<BASE_HISTORICAL>(video, run, iterations);
                (base, test)
            };
            ratios.push(base.as_secs_f64() / test.as_secs_f64());
        }
        ratios
    }

    fn measure_timestamp_link<const HISTORICAL: bool>(
        id: &str,
        start_sec: f64,
        iterations: usize,
    ) -> Duration {
        let started = Instant::now();
        let mut out = String::with_capacity(96);
        let mut checksum = 0usize;
        for _ in 0..iterations {
            out.clear();
            if HISTORICAL {
                historical_push_timestamp_link(&mut out, black_box(id), black_box(start_sec));
            } else {
                push_timestamp_link(&mut out, black_box(id), black_box(start_sec));
            }
            checksum ^= out.len();
            black_box(out.as_str());
        }
        black_box(checksum);
        started.elapsed()
    }

    fn paired_timestamp_link_ratios<const BASE_HISTORICAL: bool, const TEST_HISTORICAL: bool>(
        id: &str,
        start_sec: f64,
        iterations: usize,
        repetitions: usize,
    ) -> Vec<f64> {
        let mut ratios = Vec::with_capacity(repetitions);
        for repetition in 0..repetitions {
            let (base, test) = if repetition % 2 == 0 {
                (
                    measure_timestamp_link::<BASE_HISTORICAL>(id, start_sec, iterations),
                    measure_timestamp_link::<TEST_HISTORICAL>(id, start_sec, iterations),
                )
            } else {
                let test = measure_timestamp_link::<TEST_HISTORICAL>(id, start_sec, iterations);
                let base = measure_timestamp_link::<BASE_HISTORICAL>(id, start_sec, iterations);
                (base, test)
            };
            ratios.push(base.as_secs_f64() / test.as_secs_f64());
        }
        ratios
    }

    fn measure_paragraph_text<const HISTORICAL: bool>(
        paragraph: &Paragraph<'_>,
        iterations: usize,
    ) -> Duration {
        let started = Instant::now();
        let mut out = String::with_capacity(512);
        let mut checksum = 0usize;
        for _ in 0..iterations {
            out.clear();
            let wrote = if HISTORICAL {
                historical_push_paragraph_text(&mut out, black_box(paragraph))
            } else {
                push_paragraph_text(&mut out, black_box(paragraph))
            };
            checksum ^= out.len() ^ usize::from(wrote);
            black_box(out.as_str());
        }
        black_box(checksum);
        started.elapsed()
    }

    fn paired_paragraph_text_ratios<const BASE_HISTORICAL: bool, const TEST_HISTORICAL: bool>(
        paragraph: &Paragraph<'_>,
        iterations: usize,
        repetitions: usize,
    ) -> Vec<f64> {
        let mut ratios = Vec::with_capacity(repetitions);
        for repetition in 0..repetitions {
            let (base, test) = if repetition % 2 == 0 {
                (
                    measure_paragraph_text::<BASE_HISTORICAL>(paragraph, iterations),
                    measure_paragraph_text::<TEST_HISTORICAL>(paragraph, iterations),
                )
            } else {
                let test = measure_paragraph_text::<TEST_HISTORICAL>(paragraph, iterations);
                let base = measure_paragraph_text::<BASE_HISTORICAL>(paragraph, iterations);
                (base, test)
            };
            ratios.push(base.as_secs_f64() / test.as_secs_f64());
        }
        ratios
    }

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
    fn direct_title_heading_matches_historical_semantics() {
        let cases = [
            "",
            " \n\t ",
            "one",
            "  two   words  ",
            "line one\nline two\tline three",
            "αβγ\u{2003}東京  emoji🙂\tcafé",
        ];

        for title in cases {
            let mut historical = "prefix\n".to_owned();
            historical_push_title_heading(&mut historical, title);
            let mut direct = "prefix\n".to_owned();
            push_title_heading(&mut direct, title);
            assert_eq!(direct, historical, "title={title:?}");
        }
    }

    #[test]
    #[ignore = "strict-remote release performance A/B"]
    fn direct_title_heading_perf() {
        const TARGET_ARM_SECS: f64 = 0.020;
        const WARMUP_REPETITIONS: usize = 3;
        const PAIRED_REPETITIONS: usize = 15;
        const NULL_MEDIAN_MIN: f64 = 0.97;
        const NULL_MEDIAN_MAX: f64 = 1.03;
        const MIN_CANDIDATE_MEDIAN: f64 = 1.10;
        const REQUIRED_WINS: usize = 13;

        let title =
            "  Rust Speech Systems:\tNative Whisper, Streaming,\nDiarization, and Fast Search  ";
        let mut historical = String::new();
        historical_push_title_heading(&mut historical, title);
        let mut direct = String::new();
        push_title_heading(&mut direct, title);
        assert_eq!(direct, historical, "timed fixture must remain byte exact");
        let output_sha256 = format!("{:x}", Sha256::digest(direct.as_bytes()));

        let calibration = measure_title_heading::<true>(title, 1);
        let iterations = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
        let iterations = iterations.clamp(256, 1_048_576);

        black_box(paired_title_ratios::<true, true>(
            title,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let null_ratios = paired_title_ratios::<true, true>(title, iterations, PAIRED_REPETITIONS);
        let null = paired_perf_stats(&null_ratios);

        black_box(paired_title_ratios::<true, false>(
            title,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let candidate_ratios =
            paired_title_ratios::<true, false>(title, iterations, PAIRED_REPETITIONS);
        let candidate = paired_perf_stats(&candidate_ratios);
        let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null.median);
        let keep_eligible = null_valid
            && candidate.median >= MIN_CANDIDATE_MEDIAN
            && candidate.p10 > null.p90
            && candidate.wins >= REQUIRED_WINS;

        eprintln!(
            "YOUTUBE_TITLE_CALIBRATION input_bytes={} output_bytes={} output_sha256={} baseline_ns={:.3} iterations={} target_arm_ms={:.1}",
            title.len(),
            direct.len(),
            output_sha256,
            calibration.as_secs_f64() * 1_000_000_000.0,
            iterations,
            TARGET_ARM_SECS * 1_000.0,
        );
        eprintln!(
            "YOUTUBE_TITLE_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
            format_ratios(&null_ratios),
            null.median,
            null.p10,
            null.p90,
            null.wins,
            PAIRED_REPETITIONS,
        );
        eprintln!(
            "YOUTUBE_TITLE_AB ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} null_valid={} keep_eligible={} min_median={MIN_CANDIDATE_MEDIAN:.2} required_wins={REQUIRED_WINS}",
            format_ratios(&candidate_ratios),
            candidate.median,
            candidate.p10,
            candidate.p90,
            candidate.wins,
            PAIRED_REPETITIONS,
            null_valid,
            keep_eligible,
        );
        assert!(
            keep_eligible,
            "candidate did not clear the declared keep gate"
        );
    }

    #[test]
    fn direct_metadata_line_matches_historical_semantics() {
        for mask in 0_u8..8 {
            let mut video = sample_video();
            video.channel = (mask & 1 != 0).then(|| "Example Channel".to_owned());
            video.upload_date = (mask & 2 != 0).then(|| "20240115".to_owned());
            video.duration_sec = (mask & 4 != 0).then_some(3725.75);

            let mut historical = "prefix\n".to_owned();
            historical_push_metadata_line(&mut historical, &video);
            let mut direct = "prefix\n".to_owned();
            push_metadata_line(&mut direct, &video);
            assert_eq!(direct, historical, "optional-field mask {mask:#05b}");
        }

        for (channel, date) in [
            (Some(""), Some("")),
            (Some(" \n\t "), Some(" \n\t ")),
            (Some("  preserved  "), Some("not-a-date")),
        ] {
            let mut video = sample_video();
            video.channel = channel.map(str::to_owned);
            video.upload_date = date.map(str::to_owned);
            video.duration_sec = None;

            let mut historical = String::new();
            historical_push_metadata_line(&mut historical, &video);
            let mut direct = String::new();
            push_metadata_line(&mut direct, &video);
            assert_eq!(direct, historical, "channel={channel:?}, date={date:?}");
        }
    }

    #[test]
    #[ignore = "strict-remote release performance A/B"]
    fn direct_metadata_line_perf() {
        const TARGET_ARM_SECS: f64 = 0.020;
        const WARMUP_REPETITIONS: usize = 3;
        const PAIRED_REPETITIONS: usize = 15;
        const NULL_MEDIAN_MIN: f64 = 0.97;
        const NULL_MEDIAN_MAX: f64 = 1.03;
        const MIN_CANDIDATE_MEDIAN: f64 = 1.10;
        const REQUIRED_WINS: usize = 13;

        let video = sample_video();
        let mut historical = String::new();
        historical_push_metadata_line(&mut historical, &video);
        let mut direct = String::new();
        push_metadata_line(&mut direct, &video);
        assert_eq!(direct, historical, "timed fixture must remain byte exact");
        let output_sha256 = format!("{:x}", Sha256::digest(direct.as_bytes()));

        let calibration = measure_metadata_line::<true>(&video, 1);
        let iterations = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
        let iterations = iterations.clamp(256, 1_048_576);

        black_box(paired_metadata_ratios::<true, true>(
            &video,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let null_ratios =
            paired_metadata_ratios::<true, true>(&video, iterations, PAIRED_REPETITIONS);
        let null = paired_perf_stats(&null_ratios);

        black_box(paired_metadata_ratios::<true, false>(
            &video,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let candidate_ratios =
            paired_metadata_ratios::<true, false>(&video, iterations, PAIRED_REPETITIONS);
        let candidate = paired_perf_stats(&candidate_ratios);
        let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null.median);
        let keep_eligible = null_valid
            && candidate.median >= MIN_CANDIDATE_MEDIAN
            && candidate.p10 > null.p90
            && candidate.wins >= REQUIRED_WINS;

        eprintln!(
            "YOUTUBE_METADATA_CALIBRATION output_bytes={} output_sha256={} baseline_ns={:.3} iterations={} target_arm_ms={:.1}",
            direct.len(),
            output_sha256,
            calibration.as_secs_f64() * 1_000_000_000.0,
            iterations,
            TARGET_ARM_SECS * 1_000.0,
        );
        eprintln!(
            "YOUTUBE_METADATA_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
            format_ratios(&null_ratios),
            null.median,
            null.p10,
            null.p90,
            null.wins,
            PAIRED_REPETITIONS,
        );
        eprintln!(
            "YOUTUBE_METADATA_AB ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} null_valid={} keep_eligible={} min_median={MIN_CANDIDATE_MEDIAN:.2} required_wins={REQUIRED_WINS}",
            format_ratios(&candidate_ratios),
            candidate.median,
            candidate.p10,
            candidate.p90,
            candidate.wins,
            PAIRED_REPETITIONS,
            null_valid,
            keep_eligible,
        );
        assert!(
            keep_eligible,
            "candidate did not clear the declared keep gate"
        );
    }

    #[test]
    fn direct_source_line_matches_historical_semantics() {
        let urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "www.youtube.com/watch?v=dQw4w9WgXcQ",
        ];
        let versions = [None, Some(""), Some(" \n\t "), Some("v0.2.0")];

        for url in urls {
            for version in versions {
                for rtf in [None, Some(0.04)] {
                    let mut video = sample_video();
                    video.webpage_url = url.to_owned();
                    let mut run = sample_run();
                    run.version_tag = version.map(str::to_owned);
                    run.rtf = rtf;

                    let mut historical = "prefix\n".to_owned();
                    historical_push_source_line(&mut historical, &video, &run);
                    let mut direct = "prefix\n".to_owned();
                    push_source_line(&mut direct, &video, &run);
                    assert_eq!(
                        direct, historical,
                        "url={url:?}, version={version:?}, rtf={rtf:?}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "strict-remote release performance A/B"]
    fn direct_source_line_perf() {
        const TARGET_ARM_SECS: f64 = 0.020;
        const WARMUP_REPETITIONS: usize = 3;
        const PAIRED_REPETITIONS: usize = 15;
        const NULL_MEDIAN_MIN: f64 = 0.97;
        const NULL_MEDIAN_MAX: f64 = 1.03;
        const MIN_CANDIDATE_MEDIAN: f64 = 1.10;
        const REQUIRED_WINS: usize = 13;

        let video = sample_video();
        let run = sample_run();
        let mut historical = String::new();
        historical_push_source_line(&mut historical, &video, &run);
        let mut direct = String::new();
        push_source_line(&mut direct, &video, &run);
        assert_eq!(direct, historical, "timed fixture must remain byte exact");
        let output_sha256 = format!("{:x}", Sha256::digest(direct.as_bytes()));

        let calibration = measure_source_line::<true>(&video, &run, 1);
        let iterations = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
        let iterations = iterations.clamp(256, 1_048_576);

        black_box(paired_source_ratios::<true, true>(
            &video,
            &run,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let null_ratios =
            paired_source_ratios::<true, true>(&video, &run, iterations, PAIRED_REPETITIONS);
        let null = paired_perf_stats(&null_ratios);

        black_box(paired_source_ratios::<true, false>(
            &video,
            &run,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let candidate_ratios =
            paired_source_ratios::<true, false>(&video, &run, iterations, PAIRED_REPETITIONS);
        let candidate = paired_perf_stats(&candidate_ratios);
        let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null.median);
        let keep_eligible = null_valid
            && candidate.median >= MIN_CANDIDATE_MEDIAN
            && candidate.p10 > null.p90
            && candidate.wins >= REQUIRED_WINS;

        eprintln!(
            "YOUTUBE_SOURCE_CALIBRATION output_bytes={} output_sha256={} baseline_ns={:.3} iterations={} target_arm_ms={:.1}",
            direct.len(),
            output_sha256,
            calibration.as_secs_f64() * 1_000_000_000.0,
            iterations,
            TARGET_ARM_SECS * 1_000.0,
        );
        eprintln!(
            "YOUTUBE_SOURCE_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
            format_ratios(&null_ratios),
            null.median,
            null.p10,
            null.p90,
            null.wins,
            PAIRED_REPETITIONS,
        );
        eprintln!(
            "YOUTUBE_SOURCE_AB ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} null_valid={} keep_eligible={} min_median={MIN_CANDIDATE_MEDIAN:.2} required_wins={REQUIRED_WINS}",
            format_ratios(&candidate_ratios),
            candidate.median,
            candidate.p10,
            candidate.p90,
            candidate.wins,
            PAIRED_REPETITIONS,
            null_valid,
            keep_eligible,
        );
        assert!(
            keep_eligible,
            "candidate did not clear the declared keep gate"
        );
    }

    #[test]
    fn direct_timestamp_link_matches_historical_semantics() {
        let ids = ["abc", "dQw4w9WgXcQ", "μ[]() ?"];
        let starts = [
            -5.0,
            -0.0,
            0.0,
            0.999,
            59.999,
            60.0,
            3599.999,
            3600.0,
            3661.9,
            86_400.0,
            4_294_967_295.75,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];

        for id in ids {
            for start_sec in starts {
                let mut historical = "prefix\n".to_owned();
                historical_push_timestamp_link(&mut historical, id, start_sec);
                let mut direct = "prefix\n".to_owned();
                push_timestamp_link(&mut direct, id, start_sec);
                assert_eq!(direct, historical, "id={id:?}, start_sec={start_sec:?}");
            }
        }
    }

    #[test]
    #[ignore = "strict-remote release performance A/B"]
    fn direct_timestamp_link_perf() {
        const TARGET_ARM_SECS: f64 = 0.020;
        const WARMUP_REPETITIONS: usize = 3;
        const PAIRED_REPETITIONS: usize = 15;
        const NULL_MEDIAN_MIN: f64 = 0.97;
        const NULL_MEDIAN_MAX: f64 = 1.03;
        const MIN_CANDIDATE_MEDIAN: f64 = 1.10;
        const REQUIRED_WINS: usize = 13;

        let id = "dQw4w9WgXcQ";
        let start_sec = 3725.75;
        let mut historical = String::new();
        historical_push_timestamp_link(&mut historical, id, start_sec);
        let mut direct = String::new();
        push_timestamp_link(&mut direct, id, start_sec);
        assert_eq!(direct, historical, "timed fixture must remain byte exact");
        let output_sha256 = format!("{:x}", Sha256::digest(direct.as_bytes()));

        let calibration = measure_timestamp_link::<true>(id, start_sec, 1);
        let iterations = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
        let iterations = iterations.clamp(256, 1_048_576);

        black_box(paired_timestamp_link_ratios::<true, true>(
            id,
            start_sec,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let null_ratios = paired_timestamp_link_ratios::<true, true>(
            id,
            start_sec,
            iterations,
            PAIRED_REPETITIONS,
        );
        let null = paired_perf_stats(&null_ratios);

        black_box(paired_timestamp_link_ratios::<true, false>(
            id,
            start_sec,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let candidate_ratios = paired_timestamp_link_ratios::<true, false>(
            id,
            start_sec,
            iterations,
            PAIRED_REPETITIONS,
        );
        let candidate = paired_perf_stats(&candidate_ratios);
        let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null.median);
        let keep_eligible = null_valid
            && candidate.median >= MIN_CANDIDATE_MEDIAN
            && candidate.p10 > null.p90
            && candidate.wins >= REQUIRED_WINS;

        eprintln!(
            "YOUTUBE_TIMESTAMP_LINK_CALIBRATION output_bytes={} output_sha256={} baseline_ns={:.3} iterations={} target_arm_ms={:.1}",
            direct.len(),
            output_sha256,
            calibration.as_secs_f64() * 1_000_000_000.0,
            iterations,
            TARGET_ARM_SECS * 1_000.0,
        );
        eprintln!(
            "YOUTUBE_TIMESTAMP_LINK_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
            format_ratios(&null_ratios),
            null.median,
            null.p10,
            null.p90,
            null.wins,
            PAIRED_REPETITIONS,
        );
        eprintln!(
            "YOUTUBE_TIMESTAMP_LINK_AB ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} null_valid={} keep_eligible={} min_median={MIN_CANDIDATE_MEDIAN:.2} required_wins={REQUIRED_WINS}",
            format_ratios(&candidate_ratios),
            candidate.median,
            candidate.p10,
            candidate.p90,
            candidate.wins,
            PAIRED_REPETITIONS,
            null_valid,
            keep_eligible,
        );
        assert!(
            keep_eligible,
            "candidate did not clear the declared keep gate"
        );
    }

    #[test]
    fn direct_paragraph_text_matches_historical_semantics() {
        let cases: Vec<Vec<TranscriptionSegment>> = vec![
            Vec::new(),
            vec![seg(0.0, 1.0, "")],
            vec![seg(0.0, 1.0, " \t\n ")],
            vec![seg(0.0, 1.0, "  one segment  ")],
            vec![
                seg(0.0, 1.0, "  leading and trailing  "),
                seg(1.0, 2.0, ""),
                seg(2.0, 3.0, "\tsecond\nline\t"),
            ],
            vec![
                seg(0.0, 1.0, " Καλημέρα "),
                seg(1.0, 2.0, "世界"),
                seg(2.0, 3.0, " emoji 🎙️ "),
            ],
        ];

        for segments in &cases {
            let paragraph = Paragraph {
                start_sec: 0.0,
                speaker: None,
                segments: segments.as_slice(),
            };
            let mut historical = "prefix: ".to_owned();
            let historical_wrote = historical_push_paragraph_text(&mut historical, &paragraph);
            let mut direct = "prefix: ".to_owned();
            let direct_wrote = push_paragraph_text(&mut direct, &paragraph);
            assert_eq!(direct_wrote, historical_wrote);
            assert_eq!(direct, historical);
        }
    }

    #[test]
    #[ignore = "strict-remote release performance A/B"]
    fn direct_paragraph_text_perf() {
        const TARGET_ARM_SECS: f64 = 0.020;
        const WARMUP_REPETITIONS: usize = 3;
        const PAIRED_REPETITIONS: usize = 15;
        const NULL_MEDIAN_MIN: f64 = 0.97;
        const NULL_MEDIAN_MAX: f64 = 1.03;
        const MIN_CANDIDATE_MEDIAN: f64 = 1.10;
        const REQUIRED_WINS: usize = 13;

        let segments = vec![
            seg(
                0.0,
                1.0,
                "  Native speech systems should spend their time decoding audio,  ",
            ),
            seg(
                1.0,
                2.0,
                "not repeatedly allocating intermediate transcript strings.",
            ),
            seg(2.0, 3.0, ""),
            seg(
                3.0,
                4.0,
                " Each segment is already owned by the transcription result, ",
            ),
            seg(
                4.0,
                5.0,
                "so the renderer can borrow, trim, and append it directly.",
            ),
            seg(5.0, 6.0, "   "),
            seg(
                6.0,
                7.0,
                "This fixture includes empty pieces and boundary whitespace",
            ),
            seg(
                7.0,
                8.0,
                "while preserving internal punctuation and Unicode: café 世界 🎙️.  ",
            ),
        ];
        let paragraph = Paragraph {
            start_sec: 0.0,
            speaker: None,
            segments: segments.as_slice(),
        };
        let mut historical = String::new();
        let historical_wrote = historical_push_paragraph_text(&mut historical, &paragraph);
        let mut direct = String::new();
        let direct_wrote = push_paragraph_text(&mut direct, &paragraph);
        assert_eq!(direct_wrote, historical_wrote);
        assert_eq!(direct, historical, "timed fixture must remain byte exact");
        let output_sha256 = format!("{:x}", Sha256::digest(direct.as_bytes()));

        let calibration = measure_paragraph_text::<true>(&paragraph, 1);
        let iterations = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
        let iterations = iterations.clamp(256, 1_048_576);

        black_box(paired_paragraph_text_ratios::<true, true>(
            &paragraph,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let null_ratios =
            paired_paragraph_text_ratios::<true, true>(&paragraph, iterations, PAIRED_REPETITIONS);
        let null = paired_perf_stats(&null_ratios);

        black_box(paired_paragraph_text_ratios::<true, false>(
            &paragraph,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let candidate_ratios =
            paired_paragraph_text_ratios::<true, false>(&paragraph, iterations, PAIRED_REPETITIONS);
        let candidate = paired_perf_stats(&candidate_ratios);
        let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null.median);
        let keep_eligible = null_valid
            && candidate.median >= MIN_CANDIDATE_MEDIAN
            && candidate.p10 > null.p90
            && candidate.wins >= REQUIRED_WINS;

        eprintln!(
            "YOUTUBE_PARAGRAPH_TEXT_CALIBRATION output_bytes={} output_sha256={} baseline_ns={:.3} iterations={} target_arm_ms={:.1}",
            direct.len(),
            output_sha256,
            calibration.as_secs_f64() * 1_000_000_000.0,
            iterations,
            TARGET_ARM_SECS * 1_000.0,
        );
        eprintln!(
            "YOUTUBE_PARAGRAPH_TEXT_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
            format_ratios(&null_ratios),
            null.median,
            null.p10,
            null.p90,
            null.wins,
            PAIRED_REPETITIONS,
        );
        eprintln!(
            "YOUTUBE_PARAGRAPH_TEXT_AB ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} null_valid={} keep_eligible={} min_median={MIN_CANDIDATE_MEDIAN:.2} required_wins={REQUIRED_WINS}",
            format_ratios(&candidate_ratios),
            candidate.median,
            candidate.p10,
            candidate.p90,
            candidate.wins,
            PAIRED_REPETITIONS,
            null_valid,
            keep_eligible,
        );
        assert!(
            keep_eligible,
            "candidate did not clear the declared keep gate"
        );
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
            windows: &[],
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
            windows: &[],
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
            windows: &[],
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
            windows: &[],
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
            windows: &[],
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
            windows: &[],
        };
        let md = render_markdown(&input);
        assert!(md.contains('…'), "long description should be ellipsized");
        assert_golden("markdown_description.md", &md);
    }

    #[test]
    fn bounded_description_intro_matches_historical_semantics() {
        let exactly_at_limit = "x".repeat(DESCRIPTION_INTRO_CHARS);
        let beyond_limit = "x".repeat(DESCRIPTION_INTRO_CHARS + 1);
        let separator_at_limit = format!("{}\tword", "x".repeat(DESCRIPTION_INTRO_CHARS));
        let unicode = "  αβγ\u{2003}東京\nemoji🙂  café\t".repeat(80);
        let cases = [
            None,
            Some(""),
            Some(" \n\t "),
            Some("one word"),
            Some(exactly_at_limit.as_str()),
            Some(beyond_limit.as_str()),
            Some(separator_at_limit.as_str()),
            Some(unicode.as_str()),
        ];

        for case in cases {
            assert_eq!(
                description_intro(case),
                historical_description_intro(case),
                "bounded normalization changed output for {case:?}"
            );
        }
    }

    #[test]
    #[ignore = "strict-remote release performance A/B"]
    fn bounded_description_intro_perf() {
        const TARGET_ARM_SECS: f64 = 0.020;
        const WARMUP_REPETITIONS: usize = 3;
        const PAIRED_REPETITIONS: usize = 15;
        const NULL_MEDIAN_MIN: f64 = 0.97;
        const NULL_MEDIAN_MAX: f64 = 1.03;
        const MIN_CANDIDATE_MEDIAN: f64 = 1.10;
        const REQUIRED_WINS: usize = 13;

        let phrase = " alpha\tβeta \n gamma delta🙂 epsilon ";
        let mut description = String::with_capacity(5_100);
        while description.len() < 5_000 {
            description.push_str(phrase);
        }

        let historical = historical_description_intro(Some(&description));
        let bounded = description_intro(Some(&description));
        assert_eq!(bounded, historical, "timed fixture must remain byte exact");
        let output = bounded.expect("non-empty description intro");
        let output_sha256 = format!("{:x}", Sha256::digest(output.as_bytes()));

        let calibration = measure_description_intro::<true>(&description, 1);
        let iterations = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
        let iterations = iterations.clamp(64, 131_072);

        black_box(paired_description_ratios::<true, true>(
            &description,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let null_ratios =
            paired_description_ratios::<true, true>(&description, iterations, PAIRED_REPETITIONS);
        let null = paired_perf_stats(&null_ratios);

        black_box(paired_description_ratios::<true, false>(
            &description,
            iterations,
            WARMUP_REPETITIONS,
        ));
        let candidate_ratios =
            paired_description_ratios::<true, false>(&description, iterations, PAIRED_REPETITIONS);
        let candidate = paired_perf_stats(&candidate_ratios);
        let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null.median);
        let keep_eligible = null_valid
            && candidate.median >= MIN_CANDIDATE_MEDIAN
            && candidate.p10 > null.p90
            && candidate.wins >= REQUIRED_WINS;

        eprintln!(
            "YOUTUBE_DESCRIPTION_CALIBRATION input_bytes={} output_bytes={} output_sha256={} baseline_ns={:.3} iterations={} target_arm_ms={:.1}",
            description.len(),
            output.len(),
            output_sha256,
            calibration.as_secs_f64() * 1_000_000_000.0,
            iterations,
            TARGET_ARM_SECS * 1_000.0,
        );
        eprintln!(
            "YOUTUBE_DESCRIPTION_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
            format_ratios(&null_ratios),
            null.median,
            null.p10,
            null.p90,
            null.wins,
            PAIRED_REPETITIONS,
        );
        eprintln!(
            "YOUTUBE_DESCRIPTION_AB ratios=[{}] median={:.6} p10={:.6} p90={:.6} wins={}/{} null_valid={} keep_eligible={} min_median={MIN_CANDIDATE_MEDIAN:.2} required_wins={REQUIRED_WINS}",
            format_ratios(&candidate_ratios),
            candidate.median,
            candidate.p10,
            candidate.p90,
            candidate.wins,
            PAIRED_REPETITIONS,
            null_valid,
            keep_eligible,
        );
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
            windows: &[],
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
            windows: &[],
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
            windows: &[],
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
    fn json_preserves_native_window_stats() {
        let segs = vec![seg(0.0, 1.0, "hello")];
        let windows = vec![RenderWindowStats {
            window_offset_sec: 30.0,
            tokens: 17,
            avg_logprob: -0.42,
            no_speech_prob: 0.125,
        }];
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
            windows: &windows,
        };

        let val = render_json(&input);
        assert_eq!(
            val["windows"],
            serde_json::json!([{
                "window_offset_sec": 30.0,
                "tokens": 17,
                "avg_logprob": -0.42,
                "no_speech_prob": 0.125,
            }])
        );
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
            windows: &[],
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

    #[test]
    fn json_sanitizes_nonfinite_and_denormal_timestamps() {
        // A collapsed alignment can stamp NaN or the 2.225e-308 denormal onto
        // segment offsets; render_json must never emit those as a bogus denormal,
        // while finite real offsets (including an exact 0.0) pass through unchanged.
        let segs = vec![
            TranscriptionSegment {
                start_sec: Some(f64::MIN_POSITIVE), // 2.225e-308 denormal marker
                end_sec: Some(f64::NAN),
                text: "degenerate".to_owned(),
                speaker: None,
                confidence: Some(0.5),
            },
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(4.4),
                text: "authoritative".to_owned(),
                speaker: None,
                confidence: Some(0.9),
            },
        ];
        let input = RenderInput {
            video: sample_video(),
            run: sample_run(),
            segments: &segs,
            windows: &[],
        };
        let val = render_json(&input);
        let utts = val.get("utterances").unwrap().as_array().unwrap();
        // Denormal start_sec and NaN end_sec are null-ed, never a bogus denormal.
        assert!(utts[0].get("start_sec").unwrap().is_null());
        assert!(utts[0].get("end_sec").unwrap().is_null());
        // Finite real offsets (including exact 0.0) pass through unchanged.
        assert_eq!(utts[1].get("start_sec").unwrap().as_f64().unwrap(), 0.0);
        assert_eq!(utts[1].get("end_sec").unwrap().as_f64().unwrap(), 4.4);
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
    fn write_atomic_propagates_parent_sync_failure_after_publication() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.md");
        let error = write_atomic_with_parent_sync(&target, "published", |parent| {
            assert_eq!(parent, dir.path());
            Err(std::io::Error::other("injected parent sync failure"))
        })
        .expect_err("parent sync failure must be reported");

        assert!(error.to_string().contains("injected parent sync failure"));
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "published",
            "publication must precede the durability barrier"
        );
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
        // Stronger atomicity check: an existing object at the exact target
        // path must survive a failed persist. A non-empty directory cannot be
        // replaced by a regular file, even when the test runs as root.
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("locked.md");
        std::fs::create_dir(&target).unwrap();
        let original = target.join("original.txt");
        std::fs::write(&original, "ORIGINAL").unwrap();

        let res = write_atomic(&target, "REPLACEMENT");

        assert!(res.is_err(), "replacing a non-empty directory must fail");
        assert_eq!(
            std::fs::read_to_string(&original).unwrap(),
            "ORIGINAL",
            "original target contents must survive a failed persist"
        );
    }
}
