#![forbid(unsafe_code)]

//! Dependency-free repository privacy gate.
//!
//! Compile with `rustc --edition=2024` and run with `--tracked` in release
//! automation or `--staged` before a commit. The path phase always completes
//! before the content phase; if a suspicious path exists, the tool exits
//! without reading any candidate blob unless it is an exact, externally
//! hash-pinned reviewed legacy artifact. Both modes inspect immutable stage-zero
//! index blobs, never mutable worktree bytes. Findings contain path and reason
//! code only, never matched content. Media/model signatures are checked across
//! the repository; semantic transcript-content heuristics are intentionally
//! scoped to artifact/corpus roots where prose is not normal source material.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

const MAX_TEXT_SCAN_BYTES: u64 = 8 * 1024 * 1024;
const CONTENT_MAGIC_PREFIX_BYTES: usize = 8 * 1024;
const REVIEWED_LEGACY_ENV: &str = "FRANKEN_WHISPER_REVIEWED_LEGACY_ARTIFACTS";
const QUARANTINED_PERF_PATH: &str = "tests/artifacts/perf/20260606t2341z-scale-baseline";
const SCHEMA_VERSION: &str = "repository-privacy-guard-v2";

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexEntry {
    path: PathBuf,
    mode: String,
    oid: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReviewedSpec {
    mode: String,
    oid: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ContentInspection {
    findings: Vec<Finding>,
    blob_prefixes_inspected: usize,
    blobs_fully_inspected: usize,
    non_blob_entries_inspected: usize,
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
            emit_error("FW-PRIVACY-USAGE");
            return Err(());
        }
    };
    let repository_root = git_top_level().map_err(|_| {
        emit_error("FW-PRIVACY-GIT-QUERY");
    })?;
    env::set_current_dir(&repository_root).map_err(|_| {
        emit_error("FW-PRIVACY-REPO-ROOT");
    })?;
    let index_entries = git_index_entries().map_err(|_| {
        emit_error("FW-PRIVACY-GIT-INDEX");
    })?;
    let paths = git_paths(mode, &index_entries).map_err(|_| {
        emit_error("FW-PRIVACY-GIT-QUERY");
    })?;
    let reviewed_specs = reviewed_specs(mode).map_err(|_| {
        emit_error("FW-PRIVACY-AUTHORITY");
    })?;
    let (reviewed_paths, mut path_findings) =
        reviewed_paths(&paths, &index_entries, &reviewed_specs).map_err(|_| {
            emit_error("FW-PRIVACY-GIT-INDEX");
        })?;

    for path in &paths {
        if !reviewed_paths.contains(path) {
            path_findings.extend(inspect_path_findings(path));
        }
    }
    path_findings.sort();
    path_findings.dedup();
    if !path_findings.is_empty() {
        emit_findings(&path_findings, "path");
        return Ok(false);
    }

    let mut content_entries = Vec::new();
    for path in &paths {
        if reviewed_paths.contains(path) {
            continue;
        }
        let entry = index_entries.get(path).ok_or_else(|| {
            emit_error("FW-PRIVACY-GIT-INDEX");
        })?;
        content_entries.push(entry);
    }
    let files_content_scanned = content_entries.len();
    let mut inspection = inspect_contents(&content_entries).map_err(|_| {
        emit_error("FW-PRIVACY-READ");
    })?;
    inspection.findings.sort();
    inspection.findings.dedup();
    if !inspection.findings.is_empty() {
        emit_findings(&inspection.findings, "content");
        return Ok(false);
    }
    verify_repository_snapshot(mode, &index_entries, &paths).map_err(|_| {
        emit_error("FW-PRIVACY-SNAPSHOT-CHANGED");
    })?;
    let authority_fingerprint = authority_fingerprint(&reviewed_specs);
    println!(
        "{{\"event\":\"privacy_guard.ok\",\"schema_version\":\"{}\",\"semantic_text_scope\":\"artifact-and-corpus-roots\",\"index_entries_considered\":{},\"paths_inspected\":{},\"content_phase_entries\":{},\"blob_prefixes_inspected\":{},\"blobs_fully_inspected\":{},\"non_blob_entries_inspected\":{},\"authority_entries\":{},\"reviewed_exemptions\":{},\"authority_fingerprint_fnv1a64\":\"{}\"}}",
        SCHEMA_VERSION,
        paths.len(),
        paths.len().saturating_sub(reviewed_paths.len()),
        files_content_scanned,
        inspection.blob_prefixes_inspected,
        inspection.blobs_fully_inspected,
        inspection.non_blob_entries_inspected,
        reviewed_specs.len(),
        reviewed_paths.len(),
        authority_fingerprint
    );
    Ok(true)
}

fn verify_repository_snapshot(
    mode: ScanMode,
    expected_index: &BTreeMap<PathBuf, IndexEntry>,
    expected_paths: &[PathBuf],
) -> io::Result<()> {
    let current_index = git_index_entries()?;
    let current_paths = git_paths(mode, &current_index)?;
    let final_index = git_index_entries()?;
    if &current_index != expected_index
        || &final_index != expected_index
        || current_paths != expected_paths
    {
        return Err(io::Error::other("repository snapshot changed during scan"));
    }
    Ok(())
}

fn git_top_level() -> io::Result<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other("git repository root query failed"));
    }
    let root = String::from_utf8(output.stdout)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 repository root"))?;
    let root = root.trim_end_matches(['\r', '\n']);
    if root.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "empty repository root",
        ));
    }
    Ok(PathBuf::from(root))
}

fn git_index_entries() -> io::Result<BTreeMap<PathBuf, IndexEntry>> {
    let output = Command::new("git")
        .args(["ls-files", "--stage", "-z", "--full-name"])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other("git index query failed"));
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 index path"))?;
    parse_index_entries(&text)
}

fn parse_index_entries(text: &str) -> io::Result<BTreeMap<PathBuf, IndexEntry>> {
    let mut entries = BTreeMap::new();
    for record in text.split('\0').filter(|record| !record.is_empty()) {
        let (metadata, path) = record.split_once('\t').ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid git index record")
        })?;
        let mut fields = metadata.split_whitespace();
        let mode = fields
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing git index mode"))?;
        let oid = fields.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing git index object")
        })?;
        let stage = fields
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing git index stage"))?;
        if fields.next().is_some()
            || stage != "0"
            || !valid_index_mode(mode)
            || !valid_object_id(oid)
            || path.is_empty()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported git index record",
            ));
        }
        let path = PathBuf::from(path);
        let entry = IndexEntry {
            path: path.clone(),
            mode: mode.to_owned(),
            oid: oid.to_ascii_lowercase(),
        };
        if entries.insert(path, entry).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate git index path",
            ));
        }
    }
    Ok(entries)
}

fn git_paths(
    mode: ScanMode,
    index_entries: &BTreeMap<PathBuf, IndexEntry>,
) -> io::Result<Vec<PathBuf>> {
    let output = match mode {
        ScanMode::Tracked => return Ok(index_entries.keys().cloned().collect()),
        ScanMode::Staged => Command::new("git")
            .args([
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--diff-filter=ACMRT",
                "--no-renames",
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
    if paths.iter().any(|path| !index_entries.contains_key(path)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "staged path missing from stage-zero index",
        ));
    }
    Ok(paths)
}

fn reviewed_specs(mode: ScanMode) -> io::Result<BTreeMap<String, ReviewedSpec>> {
    if mode != ScanMode::Tracked {
        return Ok(BTreeMap::new());
    }
    let Some(value) = env::var_os(REVIEWED_LEGACY_ENV) else {
        return Ok(BTreeMap::new());
    };
    let value = value
        .into_string()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 reviewed authority"))?;
    parse_reviewed_specs(&value)
}

fn parse_reviewed_specs(value: &str) -> io::Result<BTreeMap<String, ReviewedSpec>> {
    let mut specs = BTreeMap::new();
    for line in value.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let mut fields = line.splitn(3, '|');
        let mode = fields.next().unwrap_or_default();
        let oid = fields.next().unwrap_or_default();
        let path = fields.next().unwrap_or_default();
        if mode != "100644"
            || !valid_object_id(oid)
            || !path.starts_with("tests/artifacts/perf/")
            || path
                .chars()
                .any(|character| matches!(character, '\\' | '|'))
            || path
                .split('/')
                .any(|component| component.is_empty() || matches!(component, "." | ".."))
            || inspect_path_findings(Path::new(path))
                .iter()
                .any(|finding| finding.code != "FW-PRIVACY-RAW-PERF-ARTIFACT")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid reviewed authority entry",
            ));
        }
        let spec = ReviewedSpec {
            mode: mode.to_owned(),
            oid: oid.to_ascii_lowercase(),
        };
        if specs.insert(path.to_owned(), spec).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate reviewed authority path",
            ));
        }
    }
    Ok(specs)
}

fn reviewed_paths(
    paths: &[PathBuf],
    index_entries: &BTreeMap<PathBuf, IndexEntry>,
    specs: &BTreeMap<String, ReviewedSpec>,
) -> io::Result<(BTreeSet<PathBuf>, Vec<Finding>)> {
    let mut reviewed = BTreeSet::new();
    let mut mismatches = Vec::new();
    for path in paths {
        let path_text = path
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 index path"))?;
        let Some(spec) = specs.get(path_text) else {
            continue;
        };
        let entry = index_entries
            .get(path)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing git index entry"))?;
        if reviewed_spec_matches(entry, spec) {
            reviewed.insert(path.clone());
        } else {
            mismatches.push(finding(path, "FW-PRIVACY-REVIEWED-BLOB-MISMATCH"));
        }
    }
    Ok((reviewed, mismatches))
}

fn reviewed_spec_matches(entry: &IndexEntry, spec: &ReviewedSpec) -> bool {
    entry.mode == "100644" && entry.mode == spec.mode && entry.oid == spec.oid
}

fn authority_fingerprint(specs: &BTreeMap<String, ReviewedSpec>) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for (path, spec) in specs {
        for byte in spec
            .mode
            .bytes()
            .chain([b'|'])
            .chain(spec.oid.bytes())
            .chain([b'|'])
            .chain(path.bytes())
            .chain([b'\n'])
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn valid_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_index_mode(value: &str) -> bool {
    matches!(value, "100644" | "100755" | "120000" | "160000")
}

fn inspect_path_findings(path: &Path) -> Vec<Finding> {
    let normalized = normalized_path(path);
    let lower = normalized.to_ascii_lowercase();
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    let extension = file_name.rsplit_once('.').map(|(_, extension)| extension);
    let mut findings = Vec::new();
    if lower == QUARANTINED_PERF_PATH
        || lower
            .strip_prefix(QUARANTINED_PERF_PATH)
            .is_some_and(|suffix| suffix.starts_with('/'))
    {
        findings.push(finding(path, "FW-PRIVACY-QUARANTINED-PATH"));
    }
    if extension.is_some_and(is_media_extension) {
        findings.push(finding(path, "FW-PRIVACY-MEDIA-PATH"));
    }
    if extension.is_some_and(is_model_artifact_extension)
        || is_model_artifact_name(file_name, extension)
    {
        findings.push(finding(path, "FW-PRIVACY-MODEL-PATH"));
    }
    if file_name.contains("transcript")
        && extension.is_some_and(|extension| {
            matches!(
                extension,
                "md" | "txt" | "json" | "jsonl" | "srt" | "vtt" | "csv" | "tsv"
            )
        })
    {
        findings.push(finding(path, "FW-PRIVACY-TRANSCRIPT-PATH"));
    }
    if lower.starts_with("tests/artifacts/perf/")
        && (file_name.contains("transcript")
            || file_name.contains("sample")
            || file_name.contains("spans")
            || extension == Some("spans")
            || file_name.ends_with("_if.txt")
            || file_name.ends_with("_seq.txt"))
    {
        findings.push(finding(path, "FW-PRIVACY-RAW-PERF-ARTIFACT"));
    }
    if lower.split('/').any(|component| {
        matches!(
            component,
            "downloads" | "confidential" | "private" | "private_corpus"
        )
    }) {
        findings.push(finding(path, "FW-PRIVACY-PRIVATE-DIRECTORY"));
    }
    findings
}

#[cfg(test)]
fn inspect_path(path: &Path) -> Option<Finding> {
    inspect_path_findings(path).into_iter().next()
}

fn inspect_contents(entries: &[&IndexEntry]) -> io::Result<ContentInspection> {
    let mut inspection = ContentInspection::default();
    let mut blobs = Vec::new();
    for entry in entries {
        match entry.mode.as_str() {
            "100644" | "100755" => blobs.push(*entry),
            "120000" if is_risky_root(&entry.path) => {
                inspection
                    .findings
                    .push(finding(&entry.path, "FW-PRIVACY-RISKY-SYMLINK"));
                inspection.non_blob_entries_inspected += 1;
            }
            // A non-risky symbolic link stores only its target path in Git. A
            // gitlink stores only a referenced commit ID; neither embeds target
            // content in the containing repository.
            "120000" | "160000" => inspection.non_blob_entries_inspected += 1,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported git index mode",
                ));
            }
        }
    }
    inspection.blob_prefixes_inspected = blobs.len();
    if blobs.is_empty() {
        return Ok(inspection);
    }

    let mut child = Command::new("git")
        .args(["cat-file", "--batch"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let Some(mut stdin) = child.stdin.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(io::Error::other("missing git batch input"));
    };
    let Some(stdout) = child.stdout.take() else {
        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        return Err(io::Error::other("missing git batch output"));
    };
    let mut reader = BufReader::new(stdout);

    let batch_result = (|| -> io::Result<()> {
        for entry in blobs {
            writeln!(stdin, "{}", entry.oid)?;
            stdin.flush()?;

            let mut header = String::new();
            if reader.read_line(&mut header)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "missing git batch header",
                ));
            }
            let mut fields = header.split_whitespace();
            let returned_oid = fields.next().unwrap_or_default();
            let object_type = fields.next().unwrap_or_default();
            let size = fields
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid blob size"))?;
            if fields.next().is_some()
                || !returned_oid.eq_ignore_ascii_case(&entry.oid)
                || object_type != "blob"
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid git batch header",
                ));
            }

            let capture_all = size <= MAX_TEXT_SCAN_BYTES && is_risky_root(&entry.path);
            if capture_all {
                inspection.blobs_fully_inspected += 1;
            }
            let capture_size = if capture_all {
                usize::try_from(size)
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "blob too large"))?
            } else {
                usize::try_from(size.min(CONTENT_MAGIC_PREFIX_BYTES as u64))
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "blob too large"))?
            };
            let mut bytes = vec![0_u8; capture_size];
            reader.read_exact(&mut bytes)?;
            let remaining = size.saturating_sub(capture_size as u64);
            let drained = io::copy(&mut reader.by_ref().take(remaining), &mut io::sink())?;
            if drained != remaining {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated git blob",
                ));
            }
            let mut terminator = [0_u8; 1];
            reader.read_exact(&mut terminator)?;
            if terminator != [b'\n'] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid git batch terminator",
                ));
            }
            if let Some(finding) = inspect_blob(entry, size, &bytes) {
                inspection.findings.push(finding);
            }
        }
        Ok(())
    })();

    drop(stdin);
    drop(reader);
    if let Err(error) = batch_result {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    if !child.wait()?.success() {
        return Err(io::Error::other("git batch read failed"));
    }
    Ok(inspection)
}

fn inspect_blob(entry: &IndexEntry, size: u64, bytes: &[u8]) -> Option<Finding> {
    let path = &entry.path;
    if media_magic(bytes) {
        return Some(finding(path, "FW-PRIVACY-MEDIA-CONTENT"));
    }
    if model_artifact_magic(bytes) {
        return Some(finding(path, "FW-PRIVACY-MODEL-CONTENT"));
    }
    if size > MAX_TEXT_SCAN_BYTES {
        let code = if is_risky_root(path) {
            "FW-PRIVACY-OVERSIZE-RISKY-ARTIFACT"
        } else {
            "FW-PRIVACY-UNREVIEWED-LARGE-BLOB"
        };
        return Some(finding(path, code));
    }
    if !is_risky_root(path) {
        return None;
    }
    if bytes.contains(&0) {
        return Some(finding(path, "FW-PRIVACY-UNREVIEWED-BINARY-ARTIFACT"));
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Some(finding(path, "FW-PRIVACY-UNREVIEWED-BINARY-ARTIFACT"));
    };
    if text
        .chars()
        .filter(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
        .count()
        > 2
    {
        return Some(finding(path, "FW-PRIVACY-UNREVIEWED-BINARY-ARTIFACT"));
    }
    looks_like_transcript(text).then(|| finding(path, "FW-PRIVACY-TRANSCRIPT-CONTENT"))
}

#[cfg(test)]
fn read_bounded_prefix(reader: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut prefix = vec![0_u8; CONTENT_MAGIC_PREFIX_BYTES];
    let mut read = 0;
    while read < prefix.len() {
        let count = reader.read(&mut prefix[read..])?;
        if count == 0 {
            break;
        }
        read += count;
    }
    prefix.truncate(read);
    Ok(prefix)
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
            | "awb"
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

fn is_model_artifact_extension(extension: &str) -> bool {
    matches!(
        extension,
        "nemo" | "pt" | "pth" | "ckpt" | "safetensors" | "onnx" | "npy" | "npz" | "gguf" | "ggml"
    )
}

fn is_model_artifact_name(file_name: &str, extension: Option<&str>) -> bool {
    extension == Some("bin")
        && (file_name.starts_with("ggml-") || file_name.starts_with("pytorch_model"))
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

fn media_magic(bytes: &[u8]) -> bool {
    bytes.starts_with(b"fLaC")
        || bytes.starts_with(b"OggS")
        || bytes.starts_with(b"ID3")
        || bytes.starts_with(b"#!AMR\n")
        || bytes.starts_with(b"#!AMR-WB\n")
        || bytes.starts_with(b"#!AMR_MC1.0\n")
        || bytes.starts_with(b"#!AMR-WB_MC1.0\n")
        || bytes.starts_with(b"wvpk")
        || bytes.starts_with(b"caff")
        || bytes.starts_with(b".snd")
        || bytes.starts_with(b"MAC ")
        || bytes.starts_with(b".RMF")
        || bytes.starts_with(b".ra\xfd")
        || bytes.starts_with(&[
            0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11, 0xa6, 0xd9, 0x00, 0xaa, 0x00, 0x62,
            0xce, 0x6c,
        ])
        || bytes.starts_with(&[0x0b, 0x77])
        || bytes.starts_with(&[0x7f, 0xfe, 0x80, 0x01])
        || bytes.starts_with(&[0xfe, 0x7f, 0x01, 0x80])
        || bytes.starts_with(&[0x1f, 0xff, 0xe8, 0x00])
        || bytes.starts_with(&[0xff, 0x1f, 0x00, 0xe8])
        || bytes.starts_with(&[0x64, 0x58, 0x20, 0x25])
        || bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3])
        || (bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE")
        || (bytes.len() >= 12
            && bytes.starts_with(b"FORM")
            && (&bytes[8..12] == b"AIFF" || &bytes[8..12] == b"AIFC"))
        || (bytes.len() >= 8 && &bytes[4..8] == b"ftyp")
        || (bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0)
}

fn model_artifact_magic(bytes: &[u8]) -> bool {
    if bytes.starts_with(b"\x93NUMPY")
        || bytes.starts_with(b"GGUF")
        || bytes.starts_with(b"ggml")
        || bytes.starts_with(b"lmgg")
        || bytes.starts_with(b"ggmf")
        || bytes.starts_with(b"fmgg")
        || bytes.starts_with(b"ggjt")
        || bytes.starts_with(b"tjgg")
    {
        return true;
    }
    if bytes.len() >= 265
        && &bytes[257..262] == b"ustar"
        && bytes[..100]
            .windows(b"model_config.yaml".len())
            .any(|window| window == b"model_config.yaml")
    {
        return true;
    }
    if bytes.starts_with(b"PK\x03\x04")
        && (bytes
            .windows(b"data.pkl".len())
            .any(|window| window == b"data.pkl")
            || bytes.windows(b".npy".len()).any(|window| window == b".npy"))
    {
        return true;
    }
    if bytes.first() == Some(&0x80)
        && bytes
            .windows(b"torch".len())
            .any(|window| window == b"torch")
        && bytes
            .windows(b"storage".len())
            .any(|window| window == b"storage")
    {
        return true;
    }
    if bytes.len() < 9 {
        return false;
    }
    let Ok(header_length_bytes) = <[u8; 8]>::try_from(&bytes[..8]) else {
        return false;
    };
    let Ok(header_length) = usize::try_from(u64::from_le_bytes(header_length_bytes)) else {
        return false;
    };
    let Some(header_end) = 8usize.checked_add(header_length) else {
        return false;
    };
    if header_length == 0 || header_length > 100 * 1024 * 1024 {
        return false;
    }
    if header_end > bytes.len() {
        return bytes.len() == CONTENT_MAGIC_PREFIX_BYTES && bytes[8] == b'{';
    }
    let header = &bytes[8..header_end];
    header.starts_with(b"{")
        && header
            .windows(b"\"dtype\"".len())
            .any(|window| window == b"\"dtype\"")
        && header
            .windows(b"\"data_offsets\"".len())
            .any(|window| window == b"\"data_offsets\"")
}

fn looks_like_transcript(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let alphabetic = text
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    let total_words = text
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .count();
    let total_code_markers = text
        .chars()
        .filter(|character| "{}[]();=<>".contains(*character))
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
    timestamp_lines >= 2
        || speaker_lines >= 3
        || (prose_lines >= 4 && prose_words >= 80)
        || (total_words >= 120 && alphabetic >= 600 && total_code_markers <= 8)
}

fn emit_error(code: &str) {
    eprintln!(
        "{{\"event\":\"privacy_guard.error\",\"schema_version\":\"{}\",\"code\":\"{}\"}}",
        SCHEMA_VERSION,
        json_escape(code)
    );
}

fn emit_findings(findings: &[Finding], phase: &str) {
    for finding in findings {
        println!(
            "{{\"event\":\"privacy_guard.finding\",\"schema_version\":\"{}\",\"phase\":\"{}\",\"code\":\"{}\",\"path\":\"{}\"}}",
            SCHEMA_VERSION,
            phase,
            finding.code,
            json_escape(&finding.path)
        );
    }
    println!(
        "{{\"event\":\"privacy_guard.failed\",\"schema_version\":\"{}\",\"phase\":\"{}\",\"finding_count\":{}}}",
        SCHEMA_VERSION,
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
    use std::io::{self, Read};
    use std::path::Path;

    use super::{
        IndexEntry, ReviewedSpec, inspect_blob, inspect_path, looks_like_transcript, media_magic,
        model_artifact_magic, parse_index_entries, parse_reviewed_specs, read_bounded_prefix,
        reviewed_spec_matches,
    };

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
            inspect_path(Path::new("private/CALL.AWB"))
                .expect("AMR-WB finding")
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
            inspect_path(Path::new("private/opaque.txt"))
                .expect("private directory finding")
                .code,
            "FW-PRIVACY-PRIVATE-DIRECTORY"
        );
        for path in [
            "models/ORACLE.NeMo",
            "models/weights.PT",
            "models/weights.PtH",
            "models/checkpoint.CKPT",
            "models/weights.SafeTensors",
            "models/graph.ONNX",
            "models/features.NPY",
            "models/features.NPZ",
            "models/weights.GGUF",
            "models/weights.GGML",
            "models/ggml-tiny.BIN",
            "models/PyTorch_Model-00001-of-00002.BIN",
        ] {
            assert_eq!(
                inspect_path(Path::new(path))
                    .expect("model-artifact finding")
                    .code,
                "FW-PRIVACY-MODEL-PATH"
            );
        }
        assert_eq!(
            inspect_path(Path::new("tests/artifacts/perf/run/innocent_name_seq.txt"))
                .expect("raw perf finding")
                .code,
            "FW-PRIVACY-RAW-PERF-ARTIFACT"
        );
        assert_eq!(
            inspect_path(Path::new(
                "tests/artifacts/perf/20260606T2341Z-scale-baseline/opaque.bin"
            ))
            .expect("quarantined directory finding")
            .code,
            "FW-PRIVACY-QUARANTINED-PATH"
        );
        assert_eq!(
            inspect_path(Path::new(
                "tests/artifacts/perf/20260606T2341Z-scale-baseline"
            ))
            .expect("exact quarantined namespace finding")
            .code,
            "FW-PRIVACY-QUARANTINED-PATH"
        );
        assert!(inspect_path(Path::new("docs/RESULTS.md")).is_none());
    }

    #[test]
    fn reviewed_authority_parser_is_exact_and_rejects_aliases() {
        let oid = "0123456789abcdef0123456789abcdef01234567";
        let path = "tests/artifacts/perf/synthetic-run/sample.txt";
        let authority = format!("100644|{oid}|{path}\n");
        let specs = parse_reviewed_specs(&authority).expect("valid reviewed authority");
        assert_eq!(specs.get(path).expect("reviewed path").oid, oid);

        for invalid in [
            format!("100755|{oid}|{path}"),
            format!("100644|{oid}|tests\\artifacts\\perf\\sample.txt"),
            format!("100644|{oid}|tests/artifacts/perf/../sample.txt"),
            format!("100644|{oid}|tests/artifacts/perf//sample.txt"),
            format!("100644|{oid}|tests/artifacts/perf/sample|copy.txt"),
            format!("100644|{oid}|tests/artifacts/perf/private-call.m4a"),
            format!("100644|{oid}|tests/artifacts/perf/confidential/sample.txt"),
            format!("100644|{oid}|tests/artifacts/perf/20260606T2341Z-scale-baseline/opaque.bin"),
            format!("100644|not-an-object|{path}"),
            format!("100644|{oid}|{path}\n100644|{oid}|{path}"),
        ] {
            assert!(
                parse_reviewed_specs(&invalid).is_err(),
                "invalid authority must fail closed"
            );
        }
    }

    #[test]
    fn reviewed_authority_requires_exact_mode_and_object_id() {
        let entry = IndexEntry {
            path: "tests/artifacts/perf/synthetic-run/sample.txt".into(),
            mode: "100644".to_owned(),
            oid: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        };
        let exact = ReviewedSpec {
            mode: entry.mode.clone(),
            oid: entry.oid.clone(),
        };
        assert!(reviewed_spec_matches(&entry, &exact));

        let wrong_mode = ReviewedSpec {
            mode: "100755".to_owned(),
            oid: entry.oid.clone(),
        };
        assert!(!reviewed_spec_matches(&entry, &wrong_mode));

        let wrong_oid = ReviewedSpec {
            mode: entry.mode.clone(),
            oid: "ffffffffffffffffffffffffffffffffffffffff".to_owned(),
        };
        assert!(!reviewed_spec_matches(&entry, &wrong_oid));
    }

    #[test]
    fn index_parser_accepts_only_unique_stage_zero_supported_entries() {
        let oid = "0123456789abcdef0123456789abcdef01234567";
        let record = format!("100644 {oid} 0\ttests/artifacts/perf/synthetic.txt\0");
        let entries = parse_index_entries(&record).expect("valid index entry");
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries
                .get(Path::new("tests/artifacts/perf/synthetic.txt"))
                .expect("parsed path")
                .oid,
            oid
        );

        for invalid in [
            format!("100644 {oid} 1\ttests/artifacts/perf/synthetic.txt\0"),
            format!("100999 {oid} 0\ttests/artifacts/perf/synthetic.txt\0"),
            format!("100644 not-an-object 0\ttests/artifacts/perf/synthetic.txt\0"),
            format!(
                "100644 {oid} 0\ttests/artifacts/perf/synthetic.txt\0\
                 100644 {oid} 0\ttests/artifacts/perf/synthetic.txt\0"
            ),
        ] {
            assert!(
                parse_index_entries(&invalid).is_err(),
                "invalid index data must fail closed"
            );
        }
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
    fn risky_plain_prose_is_scanned_and_report_filename_never_self_authorizes() {
        let prose = "This synthetic prose line has enough alphabetic words to model an ordinary spoken sentence without containing private material during reliable privacy gate testing.\n\
                     Another synthetic prose line has enough alphabetic words to model an ordinary spoken sentence without containing private material during reliable privacy gate testing.\n\
                     A third synthetic prose line has enough alphabetic words to model an ordinary spoken sentence without containing private material during reliable privacy gate testing.\n\
                     A fourth synthetic prose line has enough alphabetic words to model an ordinary spoken sentence without containing private material during reliable privacy gate testing.";
        assert!(looks_like_transcript(prose));
        let risky = IndexEntry {
            path: "tests/artifacts/perf/synthetic-run/RESULTS.md".into(),
            mode: "100644".to_owned(),
            oid: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        };
        assert_eq!(
            inspect_blob(&risky, prose.len() as u64, prose.as_bytes())
                .expect("risky prose finding")
                .code,
            "FW-PRIVACY-TRANSCRIPT-CONTENT"
        );
        let outside_risky = IndexEntry {
            path: "docs/RESULTS.md".into(),
            ..risky
        };
        assert!(inspect_blob(&outside_risky, prose.len() as u64, prose.as_bytes()).is_none());
    }

    #[test]
    fn risky_single_paragraph_and_binary_encodings_fail_closed() {
        let entry = IndexEntry {
            path: "tests/artifacts/perf/synthetic-run/opaque.txt".into(),
            mode: "100644".to_owned(),
            oid: "0123456789abcdef0123456789abcdef01234567".to_owned(),
        };
        let paragraph = std::iter::repeat_n("syntheticword", 130)
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            inspect_blob(&entry, paragraph.len() as u64, paragraph.as_bytes())
                .expect("single-paragraph transcript-like finding")
                .code,
            "FW-PRIVACY-TRANSCRIPT-CONTENT"
        );
        for bytes in [
            b"s\0y\0n\0t\0h\0e\0t\0i\0c\0".as_slice(),
            &[0xc3, 0x28, 0x80, 0x80],
        ] {
            assert_eq!(
                inspect_blob(&entry, bytes.len() as u64, bytes)
                    .expect("unreviewed binary finding")
                    .code,
                "FW-PRIVACY-UNREVIEWED-BINARY-ARTIFACT"
            );
        }
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

    #[test]
    fn media_magic_detects_banned_audio_headers_without_prefix_near_misses() {
        assert!(media_magic(b"#!AMR\nsynthetic"));
        assert!(media_magic(b"#!AMR-WB\nsynthetic"));
        assert!(media_magic(b"#!AMR_MC1.0\nsynthetic"));
        assert!(media_magic(b"#!AMR-WB_MC1.0\nsynthetic"));
        assert!(media_magic(&[0x0b, 0x77, 0x00, 0x00]));
        assert!(media_magic(&[
            0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11, 0xa6, 0xd9, 0x00, 0xaa, 0x00, 0x62,
            0xce, 0x6c,
        ]));
        assert!(media_magic(b".ra\xfdsynthetic"));
        for sync_word in [
            [0x7f, 0xfe, 0x80, 0x01],
            [0xfe, 0x7f, 0x01, 0x80],
            [0x1f, 0xff, 0xe8, 0x00],
            [0xff, 0x1f, 0x00, 0xe8],
            [0x64, 0x58, 0x20, 0x25],
        ] {
            assert!(media_magic(&sync_word));
        }

        assert!(!media_magic(b"#!AMR synthetic"));
        assert!(!media_magic(b"#!AMR_MC1.1\nsynthetic"));
        assert!(!media_magic(b"#!AMR-WB_MC1.1\nsynthetic"));
        assert!(!media_magic(&[0x0b, 0x76, 0x00, 0x00]));
        assert!(!media_magic(&[
            0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11, 0xa6, 0xd9, 0x00, 0xaa, 0x00, 0x62,
            0xce, 0x6d,
        ]));
        assert!(!media_magic(b".ra\xfcsynthetic"));
        assert!(!media_magic(&[0x7f, 0xfe, 0x80, 0x00]));
    }

    #[test]
    fn model_magic_detects_known_tensor_containers_without_generic_archive_false_positives() {
        assert!(model_artifact_magic(b"\x93NUMPY\x01\x00synthetic"));
        assert!(model_artifact_magic(b"GGUFsynthetic"));
        assert!(model_artifact_magic(b"ggmlsynthetic"));
        assert!(model_artifact_magic(
            b"PK\x03\x04synthetic-prefix/data.pkl-synthetic"
        ));
        assert!(model_artifact_magic(
            b"PK\x03\x04synthetic-prefix/array.npy-synthetic"
        ));
        assert!(model_artifact_magic(
            b"\x80\x04synthetic-torch-storage-payload"
        ));

        let mut nemo_header = vec![0_u8; 512];
        nemo_header[..17].copy_from_slice(b"model_config.yaml");
        nemo_header[257..262].copy_from_slice(b"ustar");
        assert!(model_artifact_magic(&nemo_header));

        let safetensors_header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(safetensors_header.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(safetensors_header);
        safetensors.extend_from_slice(&[0_u8; 4]);
        assert!(model_artifact_magic(&safetensors));

        let mut large_safetensors_prefix = vec![b' '; super::CONTENT_MAGIC_PREFIX_BYTES];
        large_safetensors_prefix[..8].copy_from_slice(&9_000_u64.to_le_bytes());
        large_safetensors_prefix[8] = b'{';
        assert!(model_artifact_magic(&large_safetensors_prefix));

        assert!(!model_artifact_magic(b"PK\x03\x04ordinary-archive"));
        assert!(!model_artifact_magic(b"\x80\x04ordinary-pickle"));
        assert!(!model_artifact_magic(
            b"\x40\x00\x00\x00\x00\x00\x00\x00{\"dtype\":\"F32\",\"data_offsets\":[0,4]}"
        ));
        assert!(!model_artifact_magic(b"plain text"));
    }

    #[test]
    fn bounded_prefix_reader_retries_short_reads() {
        struct ShortReader<'a> {
            bytes: &'a [u8],
            offset: usize,
        }

        impl Read for ShortReader<'_> {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                let remaining = &self.bytes[self.offset..];
                let count = remaining.len().min(output.len()).min(3);
                output[..count].copy_from_slice(&remaining[..count]);
                self.offset += count;
                Ok(count)
            }
        }

        let expected = b"model_config.yaml followed by enough bytes for a tar header";
        let mut reader = ShortReader {
            bytes: expected,
            offset: 0,
        };
        assert_eq!(read_bounded_prefix(&mut reader).expect("prefix"), expected);
    }
}
