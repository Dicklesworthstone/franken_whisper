//! Dual-filename sanitization for YouTube ingestion.
//!
//! YouTube titles are arbitrary user text: they routinely contain path
//! separators, shell-hostile glyphs, control characters, emoji, and
//! right-to-left scripts. Using them verbatim as filesystem names is unsafe
//! and, worse, non-deterministic across videos that share a title.
//!
//! The convention here is harvested from the user's
//! `bulk_youtube_transcript_optimizer`: build a descriptive, sanitized base
//! of the form
//!
//! ```text
//! {upload_date} - {title} [{id}]
//! ```
//!
//! The YouTube **video id** is appended as a stable, collision-proof suffix
//! (ids are unique per video and survive sanitization untouched, so two
//! distinct videos can never collide on disk). The *original* title is
//! preserved elsewhere — in the emitted JSON — and is never relied upon as a
//! filesystem name.
//!
//! # Unicode policy
//!
//! This module is intentionally **dependency-free** — it does not pull in a
//! Unicode normalization crate. The default [`sanitize_base`] keeps non-ASCII
//! characters: modern filesystems (APFS, ext4, NTFS, btrfs) store arbitrary
//! UTF-8 byte sequences just fine, so a Japanese, Arabic, or emoji title
//! lands on disk readable rather than mangled. The optimizer's historical
//! ASCII-only mode existed for legacy-tooling reasons; it is offered here as
//! the opt-in [`sanitize_base_ascii`] variant for `--restrict-filenames`-style
//! callers. The *pipeline* decides which to call; both are exposed.
//!
//! What is always removed, in **both** variants:
//! - control characters (anything below `0x20`),
//! - the path-hostile set `/ \ : * ? " < > |`,
//!
//! each replaced with `_`. Whitespace runs are collapsed to a single space and
//! the result is trimmed; leading dots and dashes are stripped (they create
//! hidden files on Unix and look like CLI flags).
//!
//! # Length budget
//!
//! The full base is kept at or below [`MAX_BASE_BYTES`] (180) **bytes** of
//! UTF-8. The title portion is truncated on a char boundary *before* the
//! ` [id]` suffix is appended, so the collision-proof suffix is always intact.
//! 180 bytes leaves comfortable headroom under the common 255-byte
//! per-component limit for the `.md` / `.json` extensions and any audio
//! container extension.

use std::path::{Path, PathBuf};

/// Maximum size, in UTF-8 bytes, of the returned base name (including the
/// ` [id]` suffix). 180 keeps us well under the typical 255-byte per-path-
/// component filesystem limit even after a multi-character extension is added.
pub const MAX_BASE_BYTES: usize = 180;

/// Fallback title used when the supplied title is empty or whitespace-only.
///
/// Using a literal (rather than collapsing to just the id, which would render
/// as the awkward `id [id]`) keeps every base human-scannable.
pub const FALLBACK_TITLE: &str = "untitled";

/// Characters that are hostile in a path component on at least one mainstream
/// filesystem (Windows forbids `\ / : * ? " < > |`; `/` is also the Unix
/// separator). Each is mapped to `_`.
const PATH_HOSTILE: &[char] = &['/', '\\', ':', '*', '?', '"', '<', '>', '|'];

/// Stems that are reserved device names on Windows (case-insensitive, with or
/// without an extension). We target Unix primarily, but guarding is cheap and
/// makes the output portable, so we prefix `_` when a stem matches.
const WINDOWS_RESERVED: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Build the sanitized, collision-proof base name, keeping non-ASCII text.
///
/// Produces `"{YYYY-MM-DD} - {title} [{id}]"`, where:
/// - `upload_date` is an optional `YYYYMMDD` string; when present and valid it
///   becomes a `YYYY-MM-DD - ` prefix, and is omitted cleanly otherwise
///   (`None`, wrong length, or non-digit content),
/// - `title` is folded (see module docs): path-hostile and control chars →
///   `_`, whitespace collapsed, trimmed, leading dots/dashes stripped; an
///   empty/whitespace-only title falls back to [`FALLBACK_TITLE`],
/// - `id` is appended verbatim as ` [id]` and is **never** truncated; YouTube
///   ids contain `-` and `_`, which survive untouched.
///
/// The whole base is held to [`MAX_BASE_BYTES`] by truncating the *title*
/// portion on a UTF-8 char boundary before the suffix is appended.
#[must_use]
pub fn sanitize_base(title: &str, upload_date: Option<&str>, id: &str) -> String {
    assemble(fold_title(title, false), upload_date, id)
}

/// ASCII-lossy variant of [`sanitize_base`] for `--restrict-filenames`-style
/// callers: every non-ASCII character is replaced with `_`, in addition to the
/// control/path-hostile folding the default applies.
///
/// All other behaviour — date prefix, id suffix, length budget, fallback,
/// reserved-name guard — is identical to [`sanitize_base`].
#[must_use]
pub fn sanitize_base_ascii(title: &str, upload_date: Option<&str>, id: &str) -> String {
    assemble(fold_title(title, true), upload_date, id)
}

/// Shared assembly: `[date prefix] + folded title (budgeted) + " [id]"`,
/// with the Windows-reserved-stem guard applied last.
fn assemble(folded_title: String, upload_date: Option<&str>, id: &str) -> String {
    let prefix = format_date_prefix(upload_date); // "YYYY-MM-DD - " or ""
    let suffix = format!(" [{id}]");

    // Bytes available for the title after reserving prefix + suffix. The
    // suffix is sacrosanct; the prefix is short and fixed, so the title is the
    // only thing we trim.
    let reserved = prefix.len() + suffix.len();
    let title_budget = MAX_BASE_BYTES.saturating_sub(reserved);

    let title = truncate_on_char_boundary(&folded_title, title_budget);
    // Re-trim: truncation can expose a trailing space.
    let title = title.trim_end();

    let mut base = String::with_capacity(prefix.len() + title.len() + suffix.len());
    base.push_str(&prefix);
    base.push_str(title);
    base.push_str(&suffix);

    guard_reserved_stem(base)
}

/// Fold a raw title into a filesystem-safe component.
///
/// `ascii_only` additionally maps every non-ASCII char to `_`. In both modes:
/// control chars (`< 0x20`) and the path-hostile set → `_`; whitespace runs
/// collapse to one space; leading dots, dashes and spaces are stripped; the
/// result is trimmed. An empty result falls back to [`FALLBACK_TITLE`].
fn fold_title(title: &str, ascii_only: bool) -> String {
    let mut out = String::with_capacity(title.len());
    let mut pending_space = false;

    for ch in title.chars() {
        if ch.is_whitespace() {
            // Defer emitting whitespace so runs collapse and leading
            // whitespace never lands.
            pending_space = !out.is_empty();
            continue;
        }

        if pending_space {
            out.push(' ');
            pending_space = false;
        }

        let mapped =
            if (ch as u32) < 0x20 || PATH_HOSTILE.contains(&ch) || (ascii_only && !ch.is_ascii()) {
                // Control char (whitespace handled above), path-hostile, or a
                // non-ASCII char under the restrict-ASCII variant.
                '_'
            } else {
                ch
            };
        out.push(mapped);
    }

    // Strip leading/trailing dots, dashes and spaces (hidden-file /
    // looks-like-a-flag hazards).
    let trimmed = out.trim_matches(|c| c == '.' || c == '-' || c == ' ');

    if trimmed.is_empty() {
        FALLBACK_TITLE.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Format an optional `YYYYMMDD` upload date into a `"YYYY-MM-DD - "` prefix.
///
/// Returns an empty string when the input is `None`, not exactly 8 chars, or
/// contains a non-digit. We validate shape (8 digits) but not calendar
/// plausibility — yt-dlp already emits a real date; the goal is only to avoid
/// emitting garbage prefixes.
fn format_date_prefix(upload_date: Option<&str>) -> String {
    let Some(date) = upload_date else {
        return String::new();
    };
    if date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit()) {
        format!("{}-{}-{} - ", &date[0..4], &date[4..6], &date[6..8])
    } else {
        String::new()
    }
}

/// Truncate `s` to at most `max_bytes` UTF-8 bytes, never splitting a char.
fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Find the last char boundary at or before max_bytes.
    let mut end = 0;
    for (idx, _) in s.char_indices() {
        if idx > max_bytes {
            break;
        }
        end = idx;
    }
    &s[..end]
}

/// Prefix `_` when the base's stem (text before the first `.`) is a
/// Windows-reserved device name, case-insensitively. Cheap portability guard.
fn guard_reserved_stem(base: String) -> String {
    let stem = base.split('.').next().unwrap_or(&base);
    let lower = stem.to_ascii_lowercase();
    if WINDOWS_RESERVED.contains(&lower.as_str()) {
        format!("_{base}")
    } else {
        base
    }
}

/// The trio of output paths derived from a sanitized base.
///
/// `md` and `json` sit directly under the output directory; audio downloads
/// live under an `audio/` subdirectory (the extension is filled in later by
/// the downloader, which only knows the container after yt-dlp runs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputPaths {
    /// `<out_dir>/<base>.md`
    pub md: PathBuf,
    /// `<out_dir>/<base>.json`
    pub json: PathBuf,
    /// `<out_dir>/audio/` — the directory the audio file is downloaded into.
    pub audio_dir: PathBuf,
}

/// Derive the [`OutputPaths`] for a sanitized `base` under `out_dir`.
///
/// `base` is expected to come from [`sanitize_base`] /
/// [`sanitize_base_ascii`]; this helper only joins paths and does not
/// re-sanitize.
#[must_use]
pub fn output_paths(out_dir: &Path, base: &str) -> OutputPaths {
    OutputPaths {
        md: out_dir.join(format!("{base}.md")),
        json: out_dir.join(format!("{base}.json")),
        audio_dir: out_dir.join("audio"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- date prefix handling -------------------------------------------

    #[test]
    fn date_present_well_formed() {
        let b = sanitize_base("Hello", Some("20240115"), "abc123");
        assert_eq!(b, "2024-01-15 - Hello [abc123]");
    }

    #[test]
    fn date_absent() {
        let b = sanitize_base("Hello", None, "abc123");
        assert_eq!(b, "Hello [abc123]");
    }

    #[test]
    fn date_malformed_too_short() {
        let b = sanitize_base("Hello", Some("2024011"), "abc123");
        assert_eq!(b, "Hello [abc123]");
    }

    #[test]
    fn date_malformed_non_digit() {
        let b = sanitize_base("Hello", Some("2024-1-15"), "abc123");
        assert_eq!(b, "Hello [abc123]");
    }

    #[test]
    fn date_malformed_empty_string() {
        let b = sanitize_base("Hello", Some(""), "abc123");
        assert_eq!(b, "Hello [abc123]");
    }

    // --- id robustness ---------------------------------------------------

    #[test]
    fn id_with_hyphen_and_underscore_survives() {
        // Real YouTube ids contain - and _; they must pass through verbatim.
        let b = sanitize_base("Talk", None, "dQ-w_4w9WgX");
        assert_eq!(b, "Talk [dQ-w_4w9WgX]");
        assert!(b.ends_with("[dQ-w_4w9WgX]"));
    }

    // --- path-hostile / control char folding ----------------------------

    #[test]
    fn every_path_hostile_char_folded() {
        let b = sanitize_base(r#"a/b\c:d*e?f"g<h>i|j"#, None, "x");
        assert_eq!(b, "a_b_c_d_e_f_g_h_i_j [x]");
    }

    #[test]
    fn control_chars_folded() {
        let title = "a\u{0001}b\u{0007}c\u{001f}d";
        let b = sanitize_base(title, None, "x");
        assert_eq!(b, "a_b_c_d [x]");
    }

    #[test]
    fn tab_and_newline_are_whitespace_collapsed_not_underscored() {
        let b = sanitize_base("a\t\n  b", None, "x");
        assert_eq!(b, "a b [x]");
    }

    // --- whitespace handling --------------------------------------------

    #[test]
    fn whitespace_runs_collapse_and_trim() {
        let b = sanitize_base("   hello     world   ", None, "x");
        assert_eq!(b, "hello world [x]");
    }

    // --- leading dot / dash stripping -----------------------------------

    #[test]
    fn leading_dot_stripped() {
        let b = sanitize_base(".hidden file", None, "x");
        assert_eq!(b, "hidden file [x]");
    }

    #[test]
    fn leading_dash_stripped() {
        let b = sanitize_base("--force looks like a flag", None, "x");
        assert_eq!(b, "force looks like a flag [x]");
    }

    #[test]
    fn mixed_leading_dots_dashes_spaces_stripped() {
        let b = sanitize_base(" .-. - real", None, "x");
        assert_eq!(b, "real [x]");
    }

    // --- empty / fallback ------------------------------------------------

    #[test]
    fn empty_title_falls_back() {
        let b = sanitize_base("", Some("20240115"), "vid");
        assert_eq!(b, "2024-01-15 - untitled [vid]");
    }

    #[test]
    fn whitespace_only_title_falls_back() {
        let b = sanitize_base("   \t\n  ", None, "vid");
        assert_eq!(b, "untitled [vid]");
    }

    #[test]
    fn all_dots_and_dashes_title_falls_back() {
        let b = sanitize_base("....----", None, "vid");
        assert_eq!(b, "untitled [vid]");
    }

    // --- unicode policy: default keeps, ascii folds ---------------------

    #[test]
    fn non_ascii_kept_in_default_variant() {
        let title = "日本語のタイトル";
        let b = sanitize_base(title, None, "x");
        assert_eq!(b, "日本語のタイトル [x]");
    }

    #[test]
    fn rtl_text_passthrough_default() {
        // Arabic (right-to-left) text must survive in the default variant.
        let title = "مرحبا بالعالم";
        let b = sanitize_base(title, None, "x");
        assert!(b.contains("مرحبا بالعالم"));
        assert!(b.ends_with("[x]"));
    }

    #[test]
    fn rtl_text_folded_in_ascii_variant() {
        let title = "مرحبا";
        let b = sanitize_base_ascii(title, None, "x");
        assert!(b.is_ascii());
        assert!(b.ends_with("[x]"));
    }

    #[test]
    fn ascii_variant_keeps_ascii_and_space() {
        let b = sanitize_base_ascii("café del mar", None, "x");
        // 'é' -> '_', rest ascii.
        assert_eq!(b, "caf_ del mar [x]");
    }

    #[test]
    fn ascii_variant_folds_emoji() {
        let b = sanitize_base_ascii("fun 🎉 times", None, "x");
        assert_eq!(b, "fun _ times [x]");
        assert!(b.is_ascii());
    }

    // --- emoji + char-boundary truncation -------------------------------

    #[test]
    fn emoji_title_kept_default() {
        let b = sanitize_base("party 🎉🎊 time", None, "x");
        assert_eq!(b, "party 🎉🎊 time [x]");
    }

    #[test]
    fn emoji_truncation_stays_on_char_boundary() {
        // 100 four-byte emoji = 400 bytes, forcing truncation. The result must
        // be valid UTF-8 (guaranteed by returning a String) and respect the
        // byte budget, with the id suffix fully intact.
        let title = "🎉".repeat(100);
        let b = sanitize_base(&title, None, "vid12345");
        assert!(b.len() <= MAX_BASE_BYTES, "len {} > budget", b.len());
        assert!(b.ends_with(" [vid12345]"));
        // No replacement char / partial sequence: every char is the full emoji.
        let stem = b.trim_end_matches(" [vid12345]");
        assert!(stem.chars().all(|c| c == '🎉'));
    }

    // --- long titles honor the byte budget, suffix intact ---------------

    #[test]
    fn three_hundred_char_title_budgeted_suffix_intact() {
        let title = "a".repeat(300);
        let b = sanitize_base(&title, Some("20240115"), "ID-with_chars");
        assert!(b.len() <= MAX_BASE_BYTES, "len {} > budget", b.len());
        assert!(b.starts_with("2024-01-15 - "));
        assert!(b.ends_with(" [ID-with_chars]"));
    }

    #[test]
    fn long_multibyte_title_budget_never_exceeded() {
        // 300 three-byte CJK chars = 900 bytes.
        let title = "語".repeat(300);
        let b = sanitize_base(&title, None, "xyz");
        assert!(b.len() <= MAX_BASE_BYTES, "len {} > budget", b.len());
        assert!(b.ends_with(" [xyz]"));
        // Valid UTF-8 by construction; ensure no truncation mid-char by
        // checking the stem is whole '語' chars.
        let stem = b.trim_end_matches(" [xyz]");
        assert!(stem.chars().all(|c| c == '語'));
    }

    #[test]
    fn truncation_does_not_leave_trailing_space() {
        // Construct a title that, after byte-budget truncation, would end on a
        // space unless we re-trim.
        let mut title = "a".repeat(MAX_BASE_BYTES - " [x]".len() - 1);
        title.push(' ');
        title.push('b');
        let b = sanitize_base(&title, None, "x");
        let stem = b.trim_end_matches(" [x]");
        assert!(!stem.ends_with(' '), "stem ended with space: {stem:?}");
    }

    // --- Windows reserved names -----------------------------------------

    #[test]
    fn reserved_name_guard_triggers_on_bare_stem() {
        // The guard fires when the assembled base's pre-dot stem is exactly a
        // reserved word, case-insensitively, with or without an extension.
        assert_eq!(guard_reserved_stem("nul".to_string()), "_nul");
        assert_eq!(guard_reserved_stem("NUL".to_string()), "_NUL");
        assert_eq!(guard_reserved_stem("con.md".to_string()), "_con.md");
        assert_eq!(guard_reserved_stem("com9".to_string()), "_com9");
        assert_eq!(guard_reserved_stem("lpt1.json".to_string()), "_lpt1.json");
    }

    #[test]
    fn reserved_name_end_to_end_when_stem_is_bare() {
        // A bare "CON" title with empty date and empty id assembles to "CON []",
        // whose pre-dot stem is "CON []" (not reserved). To get a reserved
        // stem end-to-end the base must literally be the device name, which
        // happens for the .md/.json *components* downstream — exercised via the
        // guard unit test above. Here we confirm the realistic title survives
        // unmolested rather than being wrongly prefixed.
        let b = sanitize_base("CON", None, "id");
        assert_eq!(b, "CON [id]");
    }

    #[test]
    fn non_reserved_stem_untouched() {
        assert_eq!(guard_reserved_stem("console".to_string()), "console");
        assert_eq!(guard_reserved_stem("Hello [x]".to_string()), "Hello [x]");
    }

    // --- output_paths ----------------------------------------------------

    #[test]
    fn output_paths_layout() {
        let out = Path::new("/tmp/out");
        let p = output_paths(out, "2024-01-15 - Talk [abc]");
        assert_eq!(p.md, Path::new("/tmp/out/2024-01-15 - Talk [abc].md"));
        assert_eq!(p.json, Path::new("/tmp/out/2024-01-15 - Talk [abc].json"));
        assert_eq!(p.audio_dir, Path::new("/tmp/out/audio"));
    }

    #[test]
    fn output_paths_relative_dir() {
        let p = output_paths(Path::new("results"), "v [id]");
        assert_eq!(p.md, Path::new("results/v [id].md"));
        assert_eq!(p.audio_dir, Path::new("results/audio"));
    }

    // --- end-to-end realistic case --------------------------------------

    #[test]
    fn realistic_messy_title() {
        let b = sanitize_base(
            r#"  How to: build a CLI / parser?? *FAST* | 2024  "#,
            Some("20240620"),
            "dQw4w9WgXcQ",
        );
        assert_eq!(
            b,
            "2024-06-20 - How to_ build a CLI _ parser__ _FAST_ _ 2024 [dQw4w9WgXcQ]"
        );
        assert!(b.len() <= MAX_BASE_BYTES);
    }
}
