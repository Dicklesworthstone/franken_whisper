#![forbid(unsafe_code)]

//! Dependency-free repository privacy gate.
//!
//! Compile with `rustc --edition=2024` and run with `--tracked` in release
//! automation or `--staged` before a commit. The path phase always completes
//! before the content phase; if a suspicious path exists, the tool exits
//! without opening any repository file. Findings contain path and reason code
//! only, never matched content.

use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

const MAX_TEXT_SCAN_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanMode {
    Tracked,
    Staged,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Finding {
    path: String,
    code: &'static str,
}

fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) | Err(()) => ExitCode::FAILURE,
    }
}

fn run(args: impl Iterator<Item = String>) -> Result<bool, ()> {
    let arguments = args.collect::<Vec<_>>();
    if arguments == ["--help"] || arguments == ["-h"] {
        println!("usage: check_repository_privacy (--tracked | --staged)");
        return Ok(true);
    }
    let mode = match arguments.as_slice() {
        [argument] if argument == "--tracked" => ScanMode::Tracked,
        [argument] if argument == "--staged" => ScanMode::Staged,
        _ => {
            eprintln!("{{\"event\":\"privacy_guard.error\",\"code\":\"FW-PRIVACY-USAGE\"}}");
            return Err(());
        }
    };
    let paths = git_paths(mode).map_err(|_| {
        eprintln!("{{\"event\":\"privacy_guard.error\",\"code\":\"FW-PRIVACY-GIT-QUERY\"}}");
    })?;

    let mut path_findings = paths
        .iter()
        .filter_map(|path| inspect_path(path))
        .collect::<Vec<_>>();
    path_findings.sort();
    path_findings.dedup();
    if !path_findings.is_empty() {
        emit_findings(&path_findings, "path");
        return Ok(false);
    }

    let mut content_findings = Vec::new();
    for path in &paths {
        if let Some(finding) = inspect_content(path, mode).map_err(|_| {
            eprintln!("{{\"event\":\"privacy_guard.error\",\"code\":\"FW-PRIVACY-READ\"}}");
        })? {
            content_findings.push(finding);
        }
    }
    content_findings.sort();
    content_findings.dedup();
    if !content_findings.is_empty() {
        emit_findings(&content_findings, "content");
        return Ok(false);
    }
    println!(
        "{{\"event\":\"privacy_guard.ok\",\"schema_version\":\"repository-privacy-guard-v1\",\"files_scanned\":{}}}",
        paths.len()
    );
    Ok(true)
}

fn git_paths(mode: ScanMode) -> io::Result<Vec<PathBuf>> {
    let output = match mode {
        ScanMode::Tracked => Command::new("git").args(["ls-files", "-z"]).output()?,
        ScanMode::Staged => Command::new("git")
            .args([
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--diff-filter=ACMR",
            ])
            .output()?,
    };
    if !output.status.success() {
        return Err(io::Error::other("git path query failed"));
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 repository path"))?;
    let mut paths = text
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn inspect_path(path: &Path) -> Option<Finding> {
    let normalized = normalized_path(path);
    let lower = normalized.to_ascii_lowercase();
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    let extension = file_name.rsplit_once('.').map(|(_, extension)| extension);
    if extension.is_some_and(is_media_extension) {
        return Some(finding(path, "FW-PRIVACY-MEDIA-PATH"));
    }
    if file_name.contains("transcript")
        && extension.is_some_and(|extension| {
            matches!(
                extension,
                "md" | "txt" | "json" | "jsonl" | "srt" | "vtt" | "csv" | "tsv"
            )
        })
    {
        return Some(finding(path, "FW-PRIVACY-TRANSCRIPT-PATH"));
    }
    if lower.starts_with("tests/artifacts/perf/")
        && (file_name.contains("transcript")
            || file_name.contains("sample")
            || file_name.contains("spans")
            || extension == Some("spans")
            || file_name.ends_with("_if.txt")
            || file_name.ends_with("_seq.txt"))
    {
        return Some(finding(path, "FW-PRIVACY-RAW-PERF-ARTIFACT"));
    }
    if lower
        .split('/')
        .any(|component| matches!(component, "downloads" | "confidential" | "private_corpus"))
    {
        return Some(finding(path, "FW-PRIVACY-PRIVATE-DIRECTORY"));
    }
    None
}

fn inspect_content(path: &Path, mode: ScanMode) -> io::Result<Option<Finding>> {
    if mode == ScanMode::Staged {
        return inspect_staged_content(path);
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() && is_risky_root(path) {
        return Ok(Some(finding(path, "FW-PRIVACY-RISKY-SYMLINK")));
    }
    if !metadata.is_file() {
        return Ok(None);
    }

    let mut prefix = [0_u8; 32];
    let mut file = File::open(path)?;
    let prefix_len = file.read(&mut prefix)?;
    if media_magic(&prefix[..prefix_len]) {
        return Ok(Some(finding(path, "FW-PRIVACY-MEDIA-CONTENT")));
    }
    if is_reviewed_source_or_report(path) {
        return Ok(None);
    }
    if metadata.len() > MAX_TEXT_SCAN_BYTES {
        if is_reviewed_binary(path) {
            return Ok(None);
        }
        let code = if is_risky_root(path) {
            "FW-PRIVACY-OVERSIZE-RISKY-ARTIFACT"
        } else {
            "FW-PRIVACY-UNREVIEWED-LARGE-BLOB"
        };
        return Ok(Some(finding(path, code)));
    }
    if !is_risky_root(path) {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    if bytes.contains(&0) {
        return Ok(None);
    }
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return Ok(None);
    };
    if looks_like_transcript(text) {
        Ok(Some(finding(path, "FW-PRIVACY-TRANSCRIPT-CONTENT")))
    } else {
        Ok(None)
    }
}

fn inspect_staged_content(path: &Path) -> io::Result<Option<Finding>> {
    let normalized = normalized_path(path);
    let stage = Command::new("git")
        .args(["ls-files", "--stage", "--"])
        .arg(&normalized)
        .output()?;
    if !stage.status.success() {
        return Err(io::Error::other("git staged-mode query failed"));
    }
    let stage_text = String::from_utf8(stage.stdout)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid staged mode"))?;
    if stage_text.starts_with("120000 ") && is_risky_root(path) {
        return Ok(Some(finding(path, "FW-PRIVACY-RISKY-SYMLINK")));
    }

    let object_spec = format!(":{normalized}");
    let size = Command::new("git")
        .args(["cat-file", "-s"])
        .arg(&object_spec)
        .output()?;
    if !size.status.success() {
        return Err(io::Error::other("git staged-size query failed"));
    }
    let size = String::from_utf8(size.stdout)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid staged size"))?;
    let size = size
        .trim()
        .parse::<u64>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid staged size"))?;
    let prefix = staged_prefix(&object_spec)?;
    if media_magic(&prefix) {
        return Ok(Some(finding(path, "FW-PRIVACY-MEDIA-CONTENT")));
    }
    if size > MAX_TEXT_SCAN_BYTES {
        if is_reviewed_binary(path) || is_reviewed_source_or_report(path) {
            return Ok(None);
        }
        let code = if is_risky_root(path) {
            "FW-PRIVACY-OVERSIZE-RISKY-ARTIFACT"
        } else {
            "FW-PRIVACY-UNREVIEWED-LARGE-BLOB"
        };
        return Ok(Some(finding(path, code)));
    }

    let output = Command::new("git").arg("show").arg(&object_spec).output()?;
    if !output.status.success() {
        return Err(io::Error::other("git staged-content query failed"));
    }
    if !is_risky_root(path) || is_reviewed_source_or_report(path) || output.stdout.contains(&0) {
        return Ok(None);
    }
    let Ok(text) = std::str::from_utf8(&output.stdout) else {
        return Ok(None);
    };
    if looks_like_transcript(text) {
        Ok(Some(finding(path, "FW-PRIVACY-TRANSCRIPT-CONTENT")))
    } else {
        Ok(None)
    }
}

fn staged_prefix(object_spec: &str) -> io::Result<Vec<u8>> {
    let mut child = Command::new("git")
        .arg("show")
        .arg(object_spec)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut prefix = [0_u8; 32];
    let mut read = 0;
    if let Some(stdout) = child.stdout.as_mut() {
        while read < prefix.len() {
            let count = stdout.read(&mut prefix[read..])?;
            if count == 0 {
                break;
            }
            read += count;
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    Ok(prefix[..read].to_vec())
}

fn normalized_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn finding(path: &Path, code: &'static str) -> Finding {
    Finding {
        path: normalized_path(path),
        code,
    }
}

fn is_media_extension(extension: &str) -> bool {
    matches!(
        extension,
        "wav"
            | "ulaw"
            | "mp3"
            | "flac"
            | "ogg"
            | "m4a"
            | "aac"
            | "aif"
            | "aiff"
            | "amr"
            | "caf"
            | "opus"
            | "wma"
            | "3gp"
            | "mp4"
            | "mov"
            | "webm"
            | "wv"
            | "oga"
            | "mka"
            | "mkv"
            | "adts"
            | "ac3"
            | "eac3"
            | "dts"
            | "ape"
            | "alac"
            | "au"
            | "snd"
            | "mp2"
            | "mpa"
            | "m4b"
            | "m4p"
            | "3g2"
            | "ra"
            | "rm"
            | "weba"
            | "raw"
            | "pcm"
    )
}

fn is_risky_root(path: &Path) -> bool {
    let lower = normalized_path(path).to_ascii_lowercase();
    lower.starts_with("tests/artifacts/")
        || lower.starts_with("artifacts/")
        || lower.starts_with("evaluation/")
        || lower.starts_with("evaluation_artifacts/")
        || lower.starts_with("corpus/")
        || lower.starts_with("data/")
}

fn is_reviewed_source_or_report(path: &Path) -> bool {
    let lower = normalized_path(path).to_ascii_lowercase();
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    if file_name.ends_with(".rs.txt") {
        return true;
    }
    let extension = file_name.rsplit_once('.').map(|(_, extension)| extension);
    if matches!(
        extension,
        Some("rs" | "py" | "sh" | "toml" | "patch" | "lock" | "yml" | "yaml")
    ) {
        return true;
    }
    extension == Some("md")
        && (matches!(file_name, "readme.md" | "results.md" | "hotspots.md")
            || file_name
                .strip_prefix("pass")
                .is_some_and(|suffix| suffix.ends_with(".md")))
}

fn is_reviewed_binary(path: &Path) -> bool {
    let lower = normalized_path(path).to_ascii_lowercase();
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    matches!(
        file_name.rsplit_once('.').map(|(_, extension)| extension),
        Some("png" | "jpg" | "jpeg" | "webp" | "pdf")
    )
}

fn media_magic(bytes: &[u8]) -> bool {
    bytes.starts_with(b"fLaC")
        || bytes.starts_with(b"OggS")
        || bytes.starts_with(b"ID3")
        || bytes.starts_with(b"wvpk")
        || bytes.starts_with(b"caff")
        || bytes.starts_with(b".snd")
        || bytes.starts_with(b"MAC ")
        || bytes.starts_with(b".RMF")
        || bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3])
        || (bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE")
        || (bytes.len() >= 12
            && bytes.starts_with(b"FORM")
            && (&bytes[8..12] == b"AIFF" || &bytes[8..12] == b"AIFC"))
        || (bytes.len() >= 8 && &bytes[4..8] == b"ftyp")
        || (bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0)
}

fn looks_like_transcript(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let alphabetic = text
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    if alphabetic >= 80
        && lower.contains("\"text\"")
        && (lower.contains("\"transcript\"")
            || (lower.contains("\"segments\"")
                && (lower.contains("\"start\"") || lower.contains("\"start_sec\""))))
    {
        return true;
    }

    let mut timestamp_lines = 0_usize;
    let mut speaker_lines = 0_usize;
    let mut prose_lines = 0_usize;
    let mut prose_words = 0_usize;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains("-->")
            && trimmed.matches(':').count() >= 2
            && trimmed.chars().any(|character| character.is_ascii_digit())
        {
            timestamp_lines += 1;
        }
        let lower_line = trimmed.to_ascii_lowercase();
        if (lower_line.starts_with("speaker_")
            || lower_line.starts_with("[speaker")
            || lower_line.starts_with("speaker "))
            && trimmed.split_whitespace().count() >= 4
        {
            speaker_lines += 1;
        }
        let words = trimmed
            .split_whitespace()
            .filter(|word| word.chars().any(char::is_alphabetic))
            .count();
        let code_markers = trimmed
            .chars()
            .filter(|character| "{}[]();=<>".contains(*character))
            .count();
        if words >= 12
            && trimmed
                .chars()
                .filter(|character| character.is_alphabetic())
                .count()
                >= 50
            && code_markers <= 2
        {
            prose_lines += 1;
            prose_words += words;
        }
    }
    timestamp_lines >= 2 || speaker_lines >= 3 || (prose_lines >= 4 && prose_words >= 80)
}

fn emit_findings(findings: &[Finding], phase: &str) {
    for finding in findings {
        println!(
            "{{\"event\":\"privacy_guard.finding\",\"schema_version\":\"repository-privacy-guard-v1\",\"phase\":\"{}\",\"code\":\"{}\",\"path\":\"{}\"}}",
            phase,
            finding.code,
            json_escape(&finding.path)
        );
    }
    println!(
        "{{\"event\":\"privacy_guard.failed\",\"schema_version\":\"repository-privacy-guard-v1\",\"phase\":\"{}\",\"finding_count\":{}}}",
        phase,
        findings.len()
    );
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\u{:04x}", u32::from(character));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{inspect_path, looks_like_transcript, media_magic};

    #[test]
    fn path_rules_are_case_insensitive_and_cover_raw_perf_shapes() {
        assert_eq!(
            inspect_path(Path::new("private/CALL.M4A"))
                .expect("audio finding")
                .code,
            "FW-PRIVACY-MEDIA-PATH"
        );
        assert_eq!(
            inspect_path(Path::new("private/CALL.WV"))
                .expect("WavPack finding")
                .code,
            "FW-PRIVACY-MEDIA-PATH"
        );
        assert_eq!(
            inspect_path(Path::new("private/CALL.PCM"))
                .expect("raw PCM finding")
                .code,
            "FW-PRIVACY-MEDIA-PATH"
        );
        assert_eq!(
            inspect_path(Path::new("notes/CustomerTranscript.MD"))
                .expect("transcript finding")
                .code,
            "FW-PRIVACY-TRANSCRIPT-PATH"
        );
        assert_eq!(
            inspect_path(Path::new("tests/artifacts/perf/run/innocent_name_seq.txt"))
                .expect("raw perf finding")
                .code,
            "FW-PRIVACY-RAW-PERF-ARTIFACT"
        );
        assert!(inspect_path(Path::new("docs/RESULTS.md")).is_none());
    }

    #[test]
    fn content_heuristics_detect_disguised_transcripts_without_values() {
        let srt = "1\n00:00:00,000 --> 00:00:01,000\nhello there\n\
                   2\n00:00:01,000 --> 00:00:02,000\ngoodbye\n";
        assert!(looks_like_transcript(srt));
        let json = r#"{"transcript":"a sufficiently long synthetic sentence used only by the guard unit test","segments":[{"start":0,"text":"another sufficiently long synthetic sentence used only by the guard unit test"}]}"#;
        assert!(looks_like_transcript(json));
        assert!(!looks_like_transcript("1.0 2.0 3.0\nmedian=4.0\np95=5.0"));
    }

    #[test]
    fn arbitrary_markdown_names_are_not_exempt_from_content_scanning() {
        assert!(!super::is_reviewed_source_or_report(Path::new(
            "tests/artifacts/perf/opaque.md"
        )));
        assert!(super::is_reviewed_source_or_report(Path::new(
            "tests/artifacts/perf/RESULTS.md"
        )));
        assert!(super::is_reviewed_source_or_report(Path::new(
            "tests/artifacts/perf/PASS-1.md"
        )));
    }

    #[test]
    fn media_magic_detects_common_renamed_containers() {
        assert!(media_magic(b"RIFF....WAVEfmt "));
        assert!(media_magic(b"....ftypM4A "));
        assert!(media_magic(b"fLaC"));
        assert!(media_magic(b"wvpk"));
        assert!(media_magic(&[0x1a, 0x45, 0xdf, 0xa3]));
        assert!(!media_magic(b"plain text"));
    }
}
