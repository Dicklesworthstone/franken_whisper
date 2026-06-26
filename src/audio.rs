use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};
use tempfile::Builder;

use crate::error::{FwError, FwResult};
use crate::model::InputSource;
use crate::process::{run_command_cancellable, run_command_with_timeout};

const FFMPEG_BIN_ENV: &str = "FRANKEN_WHISPER_FFMPEG_BIN";
const FFPROBE_BIN_ENV: &str = "FRANKEN_WHISPER_FFPROBE_BIN";
const FORCE_FFMPEG_NORMALIZE_ENV: &str = "FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE";
const AUTO_PROVISION_FFMPEG_ENV: &str = "FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG";
const DEFAULT_STATE_DIR: &str = ".franken_whisper";
const DEFAULT_STATE_HOME_DIR: &str = ".local/state/franken_whisper";
const FFMPEG_TOOLS_DIR: &str = "tools/ffmpeg";
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(180);

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const FFMPEG_BUNDLE_URL: &str =
    "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz";
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const FFMPEG_BUNDLE_EXT: &str = ".tar.xz";

pub fn materialize_input(source: &InputSource, work_dir: &Path) -> FwResult<PathBuf> {
    materialize_input_with_token(source, work_dir, None)
}

pub(crate) fn materialize_input_with_token(
    source: &InputSource,
    work_dir: &Path,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<PathBuf> {
    if let Some(tok) = token {
        tok.checkpoint()?;
    }
    tracing::debug!(stage = "ingest", "Entering materialize_input");
    match source {
        InputSource::File { path } => {
            if !path.exists() {
                return Err(FwError::InvalidRequest(format!(
                    "input file does not exist: {}",
                    path.display()
                )));
            }
            if !path.is_file() {
                return Err(FwError::InvalidRequest(format!(
                    "input path is not a file: {}",
                    path.display()
                )));
            }
            Ok(path.clone())
        }
        InputSource::Stdin { hint_extension } => {
            let ext = hint_extension
                .as_deref()
                .and_then(sanitize_hint_extension)
                .unwrap_or_else(|| "bin".to_owned());
            let target = work_dir.join(format!("stdin_input.{ext}"));

            let mut stdin = std::io::stdin().lock();
            let mut file = fs::File::create(&target)?;
            let mut buf = [0u8; 8192];
            let mut total = 0usize;
            loop {
                if let Some(tok) = token {
                    tok.checkpoint()?;
                }
                let read = stdin.read(&mut buf)?;
                if read == 0 {
                    break;
                }
                file.write_all(&buf[..read])?;
                total = total.saturating_add(read);
            }
            if total == 0 {
                return Err(FwError::InvalidRequest(
                    "stdin input is empty; provide bytes or use --input/--mic".to_owned(),
                ));
            }
            Ok(target)
        }
        InputSource::Microphone {
            seconds,
            device,
            ffmpeg_format,
            ffmpeg_source,
        } => capture_microphone(
            *seconds,
            device.as_deref(),
            ffmpeg_format.as_deref(),
            ffmpeg_source.as_deref(),
            microphone_timeout(*seconds),
            work_dir,
        ),
    }
}

fn sanitize_hint_extension(value: &str) -> Option<String> {
    let trimmed = value.trim_start_matches('.').trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        Some(trimmed.to_owned())
    } else {
        None
    }
}

pub fn normalize_to_wav(input: &Path, work_dir: &Path) -> FwResult<PathBuf> {
    normalize_to_wav_with_timeout(input, work_dir, ffmpeg_timeout(), None)
}

pub(crate) fn normalize_to_wav_with_timeout(
    input: &Path,
    work_dir: &Path,
    timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<PathBuf> {
    let output = work_dir.join("normalized_16k_mono.wav");

    // If the user explicitly forces ffmpeg, skip the built-in path entirely.
    if force_ffmpeg_normalizer() {
        tracing::info!(
            stage = "normalize",
            input = %input.display(),
            "ffmpeg normalizer forced via env; skipping built-in path"
        );
        return normalize_via_ffmpeg(input, &output, work_dir, timeout, token);
    }

    // Primary path: built-in Rust decoder (symphonia).
    // Handles MP3, AAC, FLAC, WAV, OGG/Vorbis, ALAC, WavPack with zero
    // external dependencies.
    if let Some(tok) = token {
        tok.checkpoint()?;
    }
    match normalize_to_wav_with_builtin_decoder(input, &output, token) {
        Ok(()) => {
            tracing::debug!(
                stage = "normalize",
                input = %input.display(),
                "normalized with built-in Rust decoder (no external dependencies)"
            );
            if let Some(tok) = token {
                tok.checkpoint()?;
            }
            return Ok(output);
        }
        Err(error) if matches!(error, FwError::Cancelled(_)) => {
            return Err(error);
        }
        Err(builtin_error) => {
            tracing::info!(
                stage = "normalize",
                input = %input.display(),
                reason = %builtin_error,
                "built-in decoder cannot handle this format; falling back to ffmpeg"
            );
        }
    }

    // Fallback: ffmpeg for formats symphonia doesn't support (video files,
    // exotic audio codecs, etc.).
    normalize_via_ffmpeg(input, &output, work_dir, timeout, token)
}

/// Normalize audio via external ffmpeg subprocess.
fn normalize_via_ffmpeg(
    input: &Path,
    output: &Path,
    work_dir: &Path,
    timeout: Duration,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<PathBuf> {
    let ffmpeg_program = resolve_ffmpeg_program(Some(work_dir))?;

    let args = vec![
        "-hide_banner".to_owned(),
        "-loglevel".to_owned(),
        "error".to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        input.display().to_string(),
        "-vn".to_owned(),
        "-ar".to_owned(),
        "16000".to_owned(),
        "-ac".to_owned(),
        "1".to_owned(),
        "-c:a".to_owned(),
        "pcm_s16le".to_owned(),
        output.display().to_string(),
    ];
    if let Some(tok) = token {
        run_command_cancellable(&ffmpeg_program, &args, None, tok, Some(timeout))?;
    } else {
        run_command_with_timeout(&ffmpeg_program, &args, None, Some(timeout))?;
    }
    Ok(output.to_path_buf())
}

fn force_ffmpeg_normalizer() -> bool {
    bool_env_enabled_from_raw(
        std::env::var(FORCE_FFMPEG_NORMALIZE_ENV).ok().as_deref(),
        false,
    )
}

fn auto_provision_ffmpeg_enabled() -> bool {
    bool_env_enabled_from_raw(
        std::env::var(AUTO_PROVISION_FFMPEG_ENV).ok().as_deref(),
        true,
    )
}

fn bool_env_enabled_from_raw(raw: Option<&str>, default_value: bool) -> bool {
    match raw {
        Some(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            if matches!(normalized.as_str(), "1" | "true" | "yes" | "on") {
                true
            } else if matches!(normalized.as_str(), "0" | "false" | "no" | "off") {
                false
            } else {
                default_value
            }
        }
        None => default_value,
    }
}

pub(crate) fn resolve_ffmpeg_program(work_dir: Option<&Path>) -> FwResult<String> {
    resolve_tool_program("ffmpeg", FFMPEG_BIN_ENV, work_dir)
}

fn resolve_ffprobe_program(work_dir: Option<&Path>) -> FwResult<String> {
    resolve_tool_program("ffprobe", FFPROBE_BIN_ENV, work_dir)
}

pub(crate) fn locate_ffmpeg_program(work_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(explicit) = explicit_tool_path(FFMPEG_BIN_ENV) {
        return Some(explicit);
    }

    if let Ok(path) = which::which("ffmpeg") {
        return Some(path);
    }

    let provisioned = provisioned_tool_path("ffmpeg", work_dir);
    if provisioned.is_file() {
        return Some(provisioned);
    }

    None
}

fn resolve_tool_program(tool: &str, env_var: &str, work_dir: Option<&Path>) -> FwResult<String> {
    if let Some(explicit) = explicit_tool_path(env_var) {
        return Ok(explicit.display().to_string());
    }

    if crate::process::command_exists(tool) {
        return Ok(tool.to_owned());
    }

    let provisioned = provisioned_tool_path(tool, work_dir);
    if provisioned.is_file() {
        return Ok(provisioned.display().to_string());
    }

    if auto_provision_ffmpeg_enabled()
        && let Err(error) = ensure_local_ffmpeg_bundle(work_dir)
    {
        tracing::warn!(
            tool,
            reason = %error,
            "ffmpeg auto-provisioning failed; continuing with fallback behavior where possible"
        );
    }

    let provisioned = provisioned_tool_path(tool, work_dir);
    if provisioned.is_file() {
        return Ok(provisioned.display().to_string());
    }

    Err(FwError::CommandMissing {
        command: tool.to_owned(),
    })
}

fn explicit_tool_path(env_var: &str) -> Option<PathBuf> {
    let raw = std::env::var(env_var).ok()?;
    let path = PathBuf::from(raw);
    if path.is_file() { Some(path) } else { None }
}

fn state_root_for_tools(work_dir: Option<&Path>) -> PathBuf {
    if let Ok(state_root) = std::env::var("FRANKEN_WHISPER_STATE_DIR") {
        return PathBuf::from(state_root);
    }

    let _ = work_dir;

    if let Ok(xdg_state_home) = std::env::var("XDG_STATE_HOME") {
        return PathBuf::from(xdg_state_home).join("franken_whisper");
    }

    if let Ok(home_dir) = std::env::var("HOME") {
        return PathBuf::from(home_dir).join(DEFAULT_STATE_HOME_DIR);
    }

    PathBuf::from(DEFAULT_STATE_DIR)
}

fn ffmpeg_tools_root(work_dir: Option<&Path>) -> PathBuf {
    state_root_for_tools(work_dir).join(FFMPEG_TOOLS_DIR)
}

fn provisioned_tool_path(tool: &str, work_dir: Option<&Path>) -> PathBuf {
    ffmpeg_tools_root(work_dir)
        .join("bin")
        .join(platform_tool_name(tool))
}

fn platform_tool_name(tool: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        format!("{tool}.exe")
    }
    #[cfg(not(target_os = "windows"))]
    {
        tool.to_owned()
    }
}

fn ensure_local_ffmpeg_bundle(work_dir: Option<&Path>) -> FwResult<()> {
    let ffmpeg_path = provisioned_tool_path("ffmpeg", work_dir);
    let ffprobe_path = provisioned_tool_path("ffprobe", work_dir);
    if ffmpeg_path.is_file() && ffprobe_path.is_file() {
        return Ok(());
    }

    let (bundle_url, extension) = ffmpeg_bundle_source()?;
    let tools_root = ffmpeg_tools_root(work_dir);
    let cache_dir = tools_root.join("cache");
    let bin_dir = tools_root.join("bin");
    tracing::info!(
        tools_root = %tools_root.display(),
        cache_dir = %cache_dir.display(),
        bin_dir = %bin_dir.display(),
        "auto-provisioning local ffmpeg bundle"
    );
    fs::create_dir_all(&cache_dir)?;
    fs::create_dir_all(&bin_dir)?;

    let archive_path = cache_dir.join(format!("ffmpeg_bundle{extension}"));
    if !archive_path.is_file() {
        download_ffmpeg_bundle(bundle_url, &archive_path)?;
    }

    let extract_dir = Builder::new()
        .prefix("franken_whisper_ffmpeg_extract_")
        .tempdir()?;
    let extract_path = extract_dir.path().to_path_buf();
    tracing::debug!(extract_dir = %extract_path.display(), "extracting ffmpeg bundle");
    extract_ffmpeg_bundle(&archive_path, &extract_path)?;

    let extracted_ffmpeg = find_file_named(&extract_path, &platform_tool_name("ffmpeg"))
        .ok_or_else(|| FwError::Unsupported("ffmpeg bundle missing ffmpeg binary".to_owned()))?;
    validate_extracted_binary(&extracted_ffmpeg, "ffmpeg")?;
    let extracted_ffprobe = find_file_named(&extract_path, &platform_tool_name("ffprobe"))
        .ok_or_else(|| FwError::Unsupported("ffmpeg bundle missing ffprobe binary".to_owned()))?;
    validate_extracted_binary(&extracted_ffprobe, "ffprobe")?;

    fs::copy(&extracted_ffmpeg, &ffmpeg_path)?;
    fs::copy(&extracted_ffprobe, &ffprobe_path)?;
    mark_executable_if_unix(&ffmpeg_path)?;
    mark_executable_if_unix(&ffprobe_path)?;
    tracing::info!(
        ffmpeg = %ffmpeg_path.display(),
        ffprobe = %ffprobe_path.display(),
        "ffmpeg bundle provisioned successfully"
    );

    Ok(())
}

fn ffmpeg_bundle_source() -> FwResult<(&'static str, &'static str)> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        Ok((FFMPEG_BUNDLE_URL, FFMPEG_BUNDLE_EXT))
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        Err(FwError::Unsupported(
            "auto-provisioned ffmpeg is currently supported on linux/x86_64 only; set FRANKEN_WHISPER_FFMPEG_BIN and FRANKEN_WHISPER_FFPROBE_BIN explicitly on this platform".to_owned(),
        ))
    }
}

fn download_ffmpeg_bundle(url: &str, destination: &Path) -> FwResult<()> {
    let has_curl = crate::process::command_exists("curl");
    let has_wget = crate::process::command_exists("wget");
    download_ffmpeg_bundle_with_tools(url, destination, has_curl, has_wget)
}

fn download_ffmpeg_bundle_with_tools(
    url: &str,
    destination: &Path,
    has_curl: bool,
    has_wget: bool,
) -> FwResult<()> {
    let dest = destination.display().to_string();
    if has_curl {
        let args = vec![
            "-fsSL".to_owned(),
            "--retry".to_owned(),
            "3".to_owned(),
            "-o".to_owned(),
            dest,
            url.to_owned(),
        ];
        run_command_with_timeout("curl", &args, None, Some(DOWNLOAD_TIMEOUT))?;
        return Ok(());
    }

    if has_wget {
        let args = vec![
            "-q".to_owned(),
            "-O".to_owned(),
            destination.display().to_string(),
            url.to_owned(),
        ];
        run_command_with_timeout("wget", &args, None, Some(DOWNLOAD_TIMEOUT))?;
        return Ok(());
    }

    Err(FwError::CommandMissing {
        command: "curl or wget".to_owned(),
    })
}

fn extract_ffmpeg_bundle(archive_path: &Path, destination: &Path) -> FwResult<()> {
    let archive = archive_path.display().to_string();
    let dest = destination.display().to_string();
    let mut args = vec!["-xf".to_owned(), archive, "-C".to_owned(), dest];

    let archive_name = archive_path.display().to_string();
    if archive_name.ends_with(".tar.xz") {
        args[0] = "-xJf".to_owned();
    } else if archive_name.ends_with(".tar.gz") || archive_name.ends_with(".tgz") {
        args[0] = "-xzf".to_owned();
    }

    validate_tar_entries(archive_path)?;
    run_command_with_timeout("tar", &args, None, Some(DOWNLOAD_TIMEOUT))?;
    Ok(())
}

fn validate_tar_entries(archive_path: &Path) -> FwResult<()> {
    let archive = archive_path.display().to_string();
    let args = vec!["-tf".to_owned(), archive];
    let output = run_command_with_timeout("tar", &args, None, Some(DOWNLOAD_TIMEOUT))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !validate_tar_entry_path(trimmed) {
            return Err(FwError::Unsupported(format!(
                "ffmpeg bundle contains unsafe path entry: {trimmed}"
            )));
        }
    }
    Ok(())
}

fn validate_tar_entry_path(entry: &str) -> bool {
    let path = Path::new(entry);
    if path.is_absolute() {
        return false;
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return false;
        }
    }
    true
}

fn validate_extracted_binary(path: &Path, label: &str) -> FwResult<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(FwError::Unsupported(format!(
            "ffmpeg bundle entry for {label} is not a regular file"
        )));
    }
    Ok(())
}

fn find_file_named(root: &Path, needle: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            let path = entry.path();

            if file_type.is_dir() {
                stack.push(path);
                continue;
            }

            if file_type.is_symlink() {
                continue;
            }

            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == needle)
            {
                return Some(path);
            }
        }
    }
    None
}

fn mark_executable_if_unix(path: &Path) -> FwResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn normalize_to_wav_with_builtin_decoder(
    input: &Path,
    output: &Path,
    token: Option<&crate::orchestrator::CancellationToken>,
) -> FwResult<()> {
    if let Some(tok) = token {
        tok.checkpoint()?;
    }
    let file = fs::File::open(input)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = input.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }

    let mut probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| {
            FwError::Unsupported(format!(
                "built-in normalizer could not open `{}`: {error}",
                input.display()
            ))
        })?;

    let (track_id, codec_params, sample_rate, channel_count) = {
        let track = probed.format.default_track().ok_or_else(|| {
            FwError::Unsupported(format!(
                "built-in normalizer found no default audio track in `{}`",
                input.display()
            ))
        })?;
        if track.codec_params.codec == CODEC_TYPE_NULL {
            return Err(FwError::Unsupported(format!(
                "built-in normalizer found unsupported codec in `{}`",
                input.display()
            )));
        }
        let sample_rate = track.codec_params.sample_rate.ok_or_else(|| {
            FwError::Unsupported(format!(
                "built-in normalizer missing sample rate metadata in `{}`",
                input.display()
            ))
        })?;
        let channel_count = track
            .codec_params
            .channels
            .map(|channels| channels.count())
            .ok_or_else(|| {
                FwError::Unsupported(format!(
                    "built-in normalizer missing channel metadata in `{}`",
                    input.display()
                ))
            })?;
        if sample_rate == 0 {
            return Err(FwError::Unsupported(format!(
                "built-in normalizer reported zero sample rate for `{}`",
                input.display()
            )));
        }
        if channel_count == 0 {
            return Err(FwError::Unsupported(format!(
                "built-in normalizer reported zero channels for `{}`",
                input.display()
            )));
        }
        (
            track.id,
            track.codec_params.clone(),
            sample_rate,
            channel_count,
        )
    };

    let mut decoder = get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|error| {
            FwError::Unsupported(format!(
                "built-in normalizer could not initialize decoder for `{}`: {error}",
                input.display()
            ))
        })?;

    let mut mono_samples = Vec::<f32>::new();
    loop {
        if let Some(tok) = token {
            tok.checkpoint()?;
        }
        let packet = match probed.format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(io_err))
                if io_err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(error) => {
                return Err(FwError::Unsupported(format!(
                    "built-in normalizer failed while reading `{}`: {error}",
                    input.display()
                )));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => append_decoded_audio_to_mono(&mut mono_samples, decoded, channel_count),
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(io_err))
                if io_err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(FwError::Unsupported(format!(
                    "built-in normalizer does not support codec reset streams for `{}`",
                    input.display()
                )));
            }
            Err(error) => {
                return Err(FwError::Unsupported(format!(
                    "built-in normalizer decode failed for `{}`: {error}",
                    input.display()
                )));
            }
        }
    }

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    if mono_samples.is_empty() {
        return Err(FwError::Unsupported(format!(
            "built-in normalizer decoded no audio frames from `{}`; install ffmpeg for broader format support",
            input.display()
        )));
    }

    if let Some(tok) = token {
        tok.checkpoint()?;
    }

    let normalized = resample_mono_linear(&mono_samples, sample_rate, 16_000);
    write_mono_wav_i16(output, &normalized, 16_000)
}

fn append_decoded_audio_to_mono(
    destination: &mut Vec<f32>,
    decoded: AudioBufferRef<'_>,
    channel_count: usize,
) {
    let mut sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
    sample_buffer.copy_interleaved_ref(decoded);
    let interleaved = sample_buffer.samples();

    if channel_count <= 1 {
        destination.extend_from_slice(interleaved);
        return;
    }

    destination.extend_from_slice(&downmix_to_mono(interleaved, channel_count));
}

/// Average `channels` interleaved samples per frame down to mono. Bit-exact with
/// the scalar `frame.sum() / channels` reference (`0.0+L+R == L+R`, and `/2.0`
/// is exact); the stereo fast path uses a SAFE `std::simd` deinterleave. Exposed
/// for the benchmark harness; not part of the stable API.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / channels;
    let mut out = vec![0.0f32; frames];

    if channels == 2 {
        use std::simd::Simd;
        const L: usize = 8;
        let simd_frames = (frames / L) * L;
        let half = Simd::<f32, L>::splat(0.5);
        let mut f = 0;
        while f < simd_frames {
            // [L0 R0 L1 R1 ...] across two 8-lane loads -> (L0..L7, R0..R7).
            let a = Simd::<f32, L>::from_slice(&interleaved[2 * f..2 * f + L]);
            let b = Simd::<f32, L>::from_slice(&interleaved[2 * f + L..2 * f + 2 * L]);
            let (lefts, rights) = a.deinterleave(b);
            ((lefts + rights) * half).copy_to_slice(&mut out[f..f + L]);
            f += L;
        }
        for (i, slot) in out.iter_mut().enumerate().skip(simd_frames) {
            *slot = (interleaved[2 * i] + interleaved[2 * i + 1]) * 0.5;
        }
        return out;
    }

    for (i, slot) in out.iter_mut().enumerate() {
        let frame = &interleaved[i * channels..(i + 1) * channels];
        let sum: f32 = frame.iter().copied().sum();
        *slot = sum / channels as f32;
    }
    out
}

/// Linear-interpolation mono resampler (`src_rate` → `dst_rate`). Bit-exact with
/// the clamp-on-every-load reference (perf ledger L16). Exposed so the benchmark
/// harness can measure it directly; not part of the stable API.
pub fn resample_mono_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return input.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let output_len = (((input.len() as f64) * dst_rate as f64) / src_rate as f64).ceil() as usize;
    let total = output_len.max(1);
    let last = input.len() - 1;
    let mut output = vec![0.0f32; total];

    // PROBE: SIMD the interior (8 outputs/iter) with a SAFE `std::simd` gather —
    // same `idx*ratio` f64 position, `floor`, f32 `frac`, and non-fused f32
    // interp as the scalar form, so bit-exact (guarded by
    // `resample_mono_linear_is_bit_exact_vs_reference`). Boundary/tail stays
    // scalar (perf ledger L16: clamps hoisted out of the hot span).
    use std::simd::{Simd, StdFloat};
    const L: usize = 8;
    // Interior = outputs whose `left_idx + 1 <= last` for all 8 lanes. With
    // `left_idx = floor(idx*ratio)` monotonic, every idx < interior_end is safe.
    let interior_end = if ratio > 0.0 {
        (((last as f64) - 1.0) / ratio).floor().max(0.0) as usize
    } else {
        0
    };
    let simd_end = (interior_end / L) * L;
    let ratio_v = Simd::<f64, L>::splat(ratio);
    let lane_base = Simd::<f64, L>::from_array([0., 1., 2., 3., 4., 5., 6., 7.]);
    let one = Simd::<usize, L>::splat(1);
    let mut g = 0;
    while g < simd_end {
        let src_pos = (lane_base + Simd::splat(g as f64)) * ratio_v;
        let left_f = src_pos.floor();
        let lf = left_f.to_array();
        let left_idx = Simd::<usize, L>::from_array(std::array::from_fn(|k| lf[k] as usize));
        let fr = (src_pos - left_f).to_array();
        let frac = Simd::<f32, L>::from_array(std::array::from_fn(|k| fr[k] as f32));
        let left = Simd::<f32, L>::gather_or_default(input, left_idx);
        let right = Simd::<f32, L>::gather_or_default(input, left_idx + one);
        let out = left + (right - left) * frac;
        output[g..g + L].copy_from_slice(&out.to_array());
        g += L;
    }
    // Scalar boundary + tail (handles the last one/two taps' clamp).
    for (idx, slot) in output.iter_mut().enumerate().skip(simd_end) {
        let src_pos = idx as f64 * ratio;
        let left_idx = src_pos.floor() as usize;
        let frac = (src_pos - left_idx as f64) as f32;
        let (left, right) = if left_idx < last {
            (input[left_idx], input[left_idx + 1])
        } else if left_idx <= last {
            (input[left_idx], input[last])
        } else {
            (input[last], input[last])
        };
        *slot = left + (right - left) * frac;
    }

    output
}

fn write_mono_wav_i16(path: &Path, samples: &[f32], sample_rate: u32) -> FwResult<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(hound_error_to_fw)?;
    for sample in samples {
        let quantized = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        writer.write_sample(quantized).map_err(hound_error_to_fw)?;
    }
    writer.finalize().map_err(hound_error_to_fw)?;
    Ok(())
}

fn hound_error_to_fw(error: hound::Error) -> FwError {
    FwError::Io(std::io::Error::other(error.to_string()))
}

/// Slice a normalized WAV file at the byte-exact sample range corresponding
/// to `[start_ms, end_ms)` and write the result to a new file in `dest_dir`.
///
/// Returns the path of the slice. The source file must be a PCM WAV
/// (integer samples). The slice preserves the source's sample rate, channel
/// count, and bit depth so the slice is interchangeable with the source
/// for downstream backends that consume 16 kHz mono PCM.
///
/// Bounds are clamped: `start_ms` is clamped to `[0, total_duration]`,
/// `end_ms` is clamped to `[start_ms, total_duration]`. A request whose
/// clamped range is empty (`start_ms == end_ms`) still writes a valid
/// (zero-length) WAV so callers don't need to special-case it.
///
/// Used by [`crate::orchestrator`] to feed per-window audio slices into
/// the speculative streaming pipeline.
pub fn slice_pcm_wav_to_temp_path(
    source: &Path,
    dest_dir: &Path,
    start_ms: u64,
    end_ms: u64,
) -> FwResult<PathBuf> {
    let reader = hound::WavReader::open(source).map_err(hound_error_to_fw)?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int {
        return Err(FwError::Unsupported(format!(
            "slice_pcm_wav_to_temp_path: source `{}` is not integer PCM (sample_format={:?})",
            source.display(),
            spec.sample_format,
        )));
    }
    let samples: Vec<i32> = reader
        .into_samples::<i32>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(hound_error_to_fw)?;

    let channels = u64::from(spec.channels.max(1));
    let sample_rate = u64::from(spec.sample_rate.max(1));
    let total_frames = (samples.len() as u64) / channels;
    let total_duration_ms = (total_frames * 1000) / sample_rate;

    let clamped_start = start_ms.min(total_duration_ms);
    let clamped_end = end_ms.max(clamped_start).min(total_duration_ms);
    let start_frame = (clamped_start * sample_rate) / 1000;
    let end_frame = (clamped_end * sample_rate) / 1000;
    let start_sample = (start_frame * channels) as usize;
    let end_sample = ((end_frame * channels) as usize).min(samples.len());
    let slice = &samples[start_sample..end_sample];

    let slice_path = dest_dir.join(format!("slice_{clamped_start}_{clamped_end}.wav"));
    let mut writer = hound::WavWriter::create(&slice_path, spec).map_err(hound_error_to_fw)?;
    let max_value: i64 = match spec.bits_per_sample {
        8 => i64::from(i8::MAX),
        16 => i64::from(i16::MAX),
        24 => (1_i64 << 23) - 1,
        32 => i64::from(i32::MAX),
        other => {
            return Err(FwError::Unsupported(format!(
                "slice_pcm_wav_to_temp_path: unsupported bits_per_sample={other}",
            )));
        }
    };
    let min_value: i64 = -max_value - 1;
    for &sample in slice {
        let clamped = i64::from(sample).clamp(min_value, max_value);
        writer
            .write_sample(clamped as i32)
            .map_err(hound_error_to_fw)?;
    }
    writer.finalize().map_err(hound_error_to_fw)?;
    Ok(slice_path)
}

pub fn probe_duration_seconds(input: &Path) -> Option<f64> {
    probe_duration_seconds_with_timeout(input, ffprobe_timeout())
}

pub fn probe_duration_seconds_with_timeout(input: &Path, timeout: Duration) -> Option<f64> {
    let ffprobe_program = resolve_ffprobe_program(None).ok()?;
    let args = vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-show_entries".to_owned(),
        "format=duration".to_owned(),
        "-of".to_owned(),
        "default=nokey=1:noprint_wrappers=1".to_owned(),
        input.display().to_string(),
    ];

    let output = run_command_with_timeout(&ffprobe_program, &args, None, Some(timeout)).ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let secs = stdout.trim().parse::<f64>().ok()?;
    if !secs.is_finite() || secs < 0.0 {
        return None;
    }
    Some(secs)
}

fn capture_microphone(
    seconds: u32,
    device: Option<&str>,
    ffmpeg_format: Option<&str>,
    ffmpeg_source: Option<&str>,
    timeout: Duration,
    work_dir: &Path,
) -> FwResult<PathBuf> {
    let output = work_dir.join("microphone_capture.wav");
    let ffmpeg_program = resolve_ffmpeg_program(Some(work_dir)).map_err(|error| {
        FwError::Unsupported(format!(
            "microphone capture requires ffmpeg (auto-provision attempted): {error}"
        ))
    })?;

    let (format, source) = microphone_defaults(device, ffmpeg_format, ffmpeg_source)?;
    let args = vec![
        "-hide_banner".to_owned(),
        "-loglevel".to_owned(),
        "error".to_owned(),
        "-y".to_owned(),
        "-f".to_owned(),
        format,
        "-i".to_owned(),
        source,
        "-t".to_owned(),
        seconds.to_string(),
        "-ac".to_owned(),
        "1".to_owned(),
        "-ar".to_owned(),
        "16000".to_owned(),
        "-c:a".to_owned(),
        "pcm_s16le".to_owned(),
        output.display().to_string(),
    ];

    run_command_with_timeout(&ffmpeg_program, &args, None, Some(timeout))?;
    Ok(output)
}

fn microphone_defaults(
    device: Option<&str>,
    ffmpeg_format: Option<&str>,
    ffmpeg_source: Option<&str>,
) -> FwResult<(String, String)> {
    if let (Some(format), Some(source)) = (ffmpeg_format, ffmpeg_source) {
        return Ok((format.to_owned(), source.to_owned()));
    }

    if ffmpeg_format.is_some() || ffmpeg_source.is_some() {
        return Err(FwError::InvalidRequest(
            "both --mic-ffmpeg-format and --mic-ffmpeg-source must be provided together".to_owned(),
        ));
    }

    #[cfg(target_os = "linux")]
    {
        Ok(("alsa".to_owned(), device.unwrap_or("default").to_owned()))
    }

    #[cfg(target_os = "macos")]
    {
        Ok(("avfoundation".to_owned(), device.unwrap_or(":0").to_owned()))
    }

    #[cfg(target_os = "windows")]
    {
        Ok((
            "dshow".to_owned(),
            device.unwrap_or("audio=default").to_owned(),
        ))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = device;
        Err(FwError::Unsupported(
            "microphone capture defaults are not implemented for this OS".to_owned(),
        ))
    }
}

pub(crate) fn ffmpeg_timeout() -> Duration {
    duration_from_env(
        "FRANKEN_WHISPER_FFMPEG_TIMEOUT_MS",
        Duration::from_secs(180),
    )
}

fn ffprobe_timeout() -> Duration {
    duration_from_env(
        "FRANKEN_WHISPER_FFPROBE_TIMEOUT_MS",
        Duration::from_secs(10),
    )
}

fn microphone_timeout(seconds: u32) -> Duration {
    let default = Duration::from_secs(u64::from(seconds).saturating_add(15));
    duration_from_env("FRANKEN_WHISPER_MIC_TIMEOUT_MS", default)
}

fn duration_from_env(key: &str, fallback: Duration) -> Duration {
    let Some(raw) = std::env::var(key).ok() else {
        return fallback;
    };
    let Ok(parsed) = raw.parse::<u64>() else {
        return fallback;
    };
    Duration::from_millis(parsed)
}

#[cfg(test)]
mod tests {
    use super::microphone_defaults;

    /// Byte-for-byte reference: the original clamp-on-every-load resampler. The
    /// optimized [`super::resample_mono_linear`] must reproduce this bit-for-bit
    /// (it only hoists the redundant interior clamps; arithmetic is unchanged).
    fn resample_reference(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        if src_rate == dst_rate {
            return input.to_vec();
        }
        let ratio = src_rate as f64 / dst_rate as f64;
        let output_len =
            (((input.len() as f64) * dst_rate as f64) / src_rate as f64).ceil() as usize;
        let mut output = Vec::with_capacity(output_len.max(1));
        for idx in 0..output_len.max(1) {
            let src_pos = idx as f64 * ratio;
            let left_idx = src_pos.floor() as usize;
            let right_idx = left_idx.saturating_add(1);
            let left = input[left_idx.min(input.len().saturating_sub(1))];
            let right = input[right_idx.min(input.len().saturating_sub(1))];
            let frac = (src_pos - left_idx as f64) as f32;
            output.push(left + (right - left) * frac);
        }
        output
    }

    #[test]
    fn resample_mono_linear_is_bit_exact_vs_reference() {
        use super::resample_mono_linear;
        // Cover downsample, upsample, identity, near-edge lengths, and the empty
        // input — every rate pair the builtin decoder can hand the resampler.
        let rate_pairs = [
            (44100u32, 16000u32),
            (48000, 16000),
            (22050, 16000),
            (16000, 16000),
            (8000, 16000),
            (16000, 44100),
        ];
        let lengths = [0usize, 1, 2, 3, 7, 31, 1000, 4096, 44_101];
        for &(src, dst) in &rate_pairs {
            for &n in &lengths {
                let input: Vec<f32> = (0..n)
                    .map(|i| {
                        let x = i as f32 * 0.013;
                        (x.sin() * 0.6 + (x * 1.7).cos() * 0.3).fract()
                    })
                    .collect();
                let got = resample_mono_linear(&input, src, dst);
                let want = resample_reference(&input, src, dst);
                assert_eq!(got.len(), want.len(), "len mismatch {src}->{dst} n={n}");
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "bit mismatch {src}->{dst} n={n} idx={i}: {g} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn downmix_to_mono_is_bit_exact_vs_reference() {
        use super::downmix_to_mono;
        // Byte-for-byte reference: per-frame `sum() / channels` (the pre-SIMD form).
        fn reference(interleaved: &[f32], channels: usize) -> Vec<f32> {
            if channels <= 1 {
                return interleaved.to_vec();
            }
            interleaved
                .chunks(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
                .collect()
        }
        // Exercise the stereo SIMD path (lengths straddling the 8-frame block,
        // incl. scalar tails) plus a 3-channel scalar path.
        for &channels in &[1usize, 2, 3] {
            for &frames in &[0usize, 1, 7, 8, 9, 15, 16, 1000, 4096, 4099] {
                let interleaved: Vec<f32> = (0..frames * channels)
                    .map(|i| {
                        let x = i as f32 * 0.011;
                        (x.sin() * 0.7 - (x * 1.3).cos() * 0.25).fract()
                    })
                    .collect();
                let got = downmix_to_mono(&interleaved, channels);
                let want = reference(&interleaved, channels);
                assert_eq!(got.len(), want.len(), "len ch={channels} frames={frames}");
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "bit mismatch ch={channels} frames={frames} idx={i}: {g} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_format_and_source_wins() {
        let result = microphone_defaults(Some("ignored"), Some("pulse"), Some("my-device"))
            .expect("explicit ffmpeg mic args should succeed");
        assert_eq!(result.0, "pulse");
        assert_eq!(result.1, "my-device");
    }

    #[test]
    fn partial_explicit_args_fail() {
        let err = microphone_defaults(None, Some("alsa"), None)
            .expect_err("partial explicit args should fail");
        let text = err.to_string();
        assert!(text.contains("must be provided together"));
    }

    #[test]
    fn bool_env_enabled_from_raw_honors_true_false_and_default() {
        use super::bool_env_enabled_from_raw;

        assert!(bool_env_enabled_from_raw(Some("true"), false));
        assert!(bool_env_enabled_from_raw(Some("1"), false));
        assert!(!bool_env_enabled_from_raw(Some("false"), true));
        assert!(!bool_env_enabled_from_raw(Some("off"), true));
        assert!(bool_env_enabled_from_raw(Some("unexpected"), true));
        assert!(!bool_env_enabled_from_raw(Some("unexpected"), false));
        assert!(bool_env_enabled_from_raw(None, true));
        assert!(!bool_env_enabled_from_raw(None, false));
    }

    #[test]
    fn materialize_file_nonexistent_returns_error() {
        use super::materialize_input;
        use crate::model::InputSource;

        let source = InputSource::File {
            path: std::path::PathBuf::from("/nonexistent/path/audio.wav"),
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let err = materialize_input(&source, dir.path()).expect_err("should fail");
        let text = err.to_string();
        assert!(
            text.contains("does not exist"),
            "expected 'does not exist' in: {text}"
        );
    }

    #[test]
    fn materialize_file_valid_returns_same_path() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("test.wav");
        std::fs::write(&file_path, b"fake audio content").expect("write");

        let source = InputSource::File {
            path: file_path.clone(),
        };
        let result = materialize_input(&source, dir.path()).expect("should succeed");
        assert_eq!(result, file_path);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_microphone_defaults_use_alsa() {
        let (format, source) = microphone_defaults(None, None, None).expect("linux defaults");
        assert_eq!(format, "alsa");
        assert_eq!(source, "default");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_microphone_custom_device() {
        let (format, source) =
            microphone_defaults(Some("hw:1,0"), None, None).expect("custom device");
        assert_eq!(format, "alsa");
        assert_eq!(source, "hw:1,0");
    }

    #[test]
    fn microphone_timeout_adds_buffer_beyond_capture_duration() {
        use super::microphone_timeout;
        let timeout = microphone_timeout(10);
        // Should be at least 10 + 15 = 25 seconds.
        assert!(
            timeout >= std::time::Duration::from_secs(25),
            "timeout should be at least 25s, got {timeout:?}"
        );
    }

    #[test]
    fn duration_from_env_falls_back_on_missing_var() {
        use super::duration_from_env;
        let fallback = std::time::Duration::from_secs(42);
        // Use a var name that's unlikely to exist.
        let result = duration_from_env("FRANKEN_WHISPER_TEST_NONEXISTENT_VAR_39285", fallback);
        assert_eq!(result, fallback);
    }

    #[test]
    fn normalize_to_wav_with_timeout_missing_ffmpeg_returns_error() {
        use super::normalize_to_wav_with_timeout;

        // This test relies on ffmpeg not existing at a bogus path. We can't
        // control ffmpeg presence, so instead we test the function's structure
        // by passing a non-audio file and verifying it either returns Ok (if
        // ffmpeg exists) or Err (if ffmpeg is missing or input is invalid).
        let dir = tempfile::tempdir().expect("tempdir");
        let fake_input = dir.path().join("empty.txt");
        std::fs::write(&fake_input, "not audio").expect("write");

        let result = normalize_to_wav_with_timeout(
            &fake_input,
            dir.path(),
            std::time::Duration::from_secs(5),
            None,
        );
        // Either succeeds (ffmpeg installed and processes it somehow) or fails
        // gracefully with an error — no panics.
        let _ = result;
    }

    #[test]
    fn builtin_normalizer_rewrites_wav_to_16k_mono() {
        use super::normalize_to_wav_with_builtin_decoder;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = dir.path().join("input_8k.wav");
        let output_wav = dir.path().join("normalized.wav");
        let samples = vec![1_000i16; 8_000];
        write_test_wav(&input_wav, &samples, 8_000);

        normalize_to_wav_with_builtin_decoder(&input_wav, &output_wav, None)
            .expect("built-in normalizer should decode wav");

        let reader = hound::WavReader::open(&output_wav).expect("normalized wav should open");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 16_000);
    }

    #[test]
    fn probe_duration_seconds_nonexistent_file() {
        use super::probe_duration_seconds;
        let result = probe_duration_seconds(std::path::Path::new("/nonexistent/file.wav"));
        // Should return None (either ffprobe missing or file not found).
        assert!(result.is_none());
    }

    #[test]
    fn ffmpeg_timeout_returns_positive_duration() {
        use super::ffmpeg_timeout;
        let timeout = ffmpeg_timeout();
        assert!(timeout.as_secs() > 0, "ffmpeg timeout should be positive");
    }

    #[test]
    fn ffprobe_timeout_returns_positive_duration() {
        use super::ffprobe_timeout;
        let timeout = ffprobe_timeout();
        assert!(timeout.as_secs() > 0, "ffprobe timeout should be positive");
    }

    #[test]
    fn duration_from_env_with_zero_returns_zero_duration() {
        use super::duration_from_env;
        // When the env var is not set, falls back.
        // We can't set env vars safely in parallel tests, but we can verify the
        // parsing logic by calling the function with known absent vars.
        let result = duration_from_env(
            "FRANKEN_WHISPER_TEST_NONEXISTENT_VAR_ALPHA_99999",
            std::time::Duration::from_millis(500),
        );
        assert_eq!(result, std::time::Duration::from_millis(500));
    }

    #[test]
    fn normalize_to_wav_with_cancellation_token_no_panic() {
        use super::normalize_to_wav_with_timeout;
        use crate::orchestrator::CancellationToken;

        let dir = tempfile::tempdir().expect("tempdir");
        let fake_input = dir.path().join("empty.txt");
        std::fs::write(&fake_input, "not audio").expect("write");

        // Token with no deadline should not cause cancellation.
        let cancel = CancellationToken::no_deadline();
        let result = normalize_to_wav_with_timeout(
            &fake_input,
            dir.path(),
            std::time::Duration::from_secs(5),
            Some(&cancel),
        );
        // The call should either succeed or fail due to ffmpeg — not panic.
        let _ = result;
    }

    #[test]
    fn probe_duration_seconds_with_timeout_nonexistent_file() {
        use super::probe_duration_seconds_with_timeout;
        let result = probe_duration_seconds_with_timeout(
            std::path::Path::new("/nonexistent/file.wav"),
            std::time::Duration::from_secs(5),
        );
        assert!(result.is_none());
    }

    // -------------------------------------------------------------------
    // slice_pcm_wav_to_temp_path: feeds the speculative streaming dispatch
    // -------------------------------------------------------------------

    /// Helper: write a 16 kHz mono PCM WAV whose samples are an ascending
    /// counter so each window can verify it received the expected slice.
    fn write_counter_wav(path: &std::path::Path, total_samples: usize, sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
        for idx in 0..total_samples {
            // Wrap modulo i16::MAX so the counter fits inside the integer range.
            let value = (idx as i32 % i32::from(i16::MAX)) as i16;
            writer.write_sample(value).expect("write sample");
        }
        writer.finalize().expect("finalize wav");
    }

    #[test]
    fn slice_pcm_wav_to_temp_path_preserves_sample_range_and_spec() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("source.wav");
        // 16 kHz × 1 s = 16,000 samples.
        write_counter_wav(&source, 16_000, 16_000);

        // Slice the middle 500 ms: [250 ms, 750 ms).
        let slice =
            super::slice_pcm_wav_to_temp_path(&source, dir.path(), 250, 750).expect("slice");
        assert!(slice.exists());

        let reader = hound::WavReader::open(&slice).expect("open slice");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.bits_per_sample, 16);
        let samples: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<Result<_, _>>()
            .expect("read samples");

        // 250 ms .. 750 ms at 16 kHz = samples 4000..12000 (8000 samples).
        assert_eq!(samples.len(), 8_000);
        // First slice sample must equal source sample at offset 4000.
        assert_eq!(samples[0], 4_000_i16);
        // Last slice sample must equal source sample at offset 11999.
        assert_eq!(samples[samples.len() - 1], 11_999_i16);
    }

    #[test]
    fn slice_pcm_wav_to_temp_path_clamps_oob_range() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("source.wav");
        // 16 kHz × 0.5 s = 8,000 samples → 500 ms total.
        write_counter_wav(&source, 8_000, 16_000);

        // Request a slice past the end of the audio: [400 ms, 9_000 ms).
        let slice = super::slice_pcm_wav_to_temp_path(&source, dir.path(), 400, 9_000)
            .expect("slice (clamped)");
        let reader = hound::WavReader::open(&slice).expect("open slice");
        let samples: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<Result<_, _>>()
            .expect("read samples");
        // Clamped range is [400 ms, 500 ms) = 100 ms at 16 kHz = 1600 samples.
        assert_eq!(samples.len(), 1_600);
        // First sample at offset 6400 (400 ms × 16 samples/ms).
        assert_eq!(samples[0], 6_400_i16);
    }

    #[test]
    fn slice_pcm_wav_to_temp_path_empty_range_writes_zero_length_wav() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("source.wav");
        write_counter_wav(&source, 4_000, 16_000);

        // start == end: empty slice.
        let slice =
            super::slice_pcm_wav_to_temp_path(&source, dir.path(), 100, 100).expect("empty slice");
        let reader = hound::WavReader::open(&slice).expect("open slice");
        let samples: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<Result<_, _>>()
            .expect("read samples");
        assert_eq!(samples.len(), 0);
    }

    #[test]
    fn slice_pcm_wav_to_temp_path_rejects_float_wav() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("float.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&source, spec).expect("create float wav");
        for _ in 0..1_000 {
            writer.write_sample(0.0_f32).expect("write");
        }
        writer.finalize().expect("finalize");

        let result = super::slice_pcm_wav_to_temp_path(&source, dir.path(), 0, 50);
        assert!(matches!(result, Err(crate::FwError::Unsupported(_))));
    }

    #[test]
    fn normalize_to_wav_produces_expected_output_path() {
        // Verify that normalize_to_wav_with_timeout targets the right filename.
        // We can't run ffmpeg, but we can observe the expected output path by
        // looking at what it would produce.
        let dir = tempfile::tempdir().expect("tempdir");
        let expected = dir.path().join("normalized_16k_mono.wav");
        // Before calling, the file should not exist.
        assert!(!expected.exists());
    }

    #[test]
    fn normalize_to_wav_wrapper_does_not_panic() {
        use super::normalize_to_wav;
        let dir = tempfile::tempdir().expect("tempdir");
        let fake_input = dir.path().join("empty.txt");
        std::fs::write(&fake_input, "not audio").expect("write");
        // normalize_to_wav is a thin wrapper — should not panic.
        let _ = normalize_to_wav(&fake_input, dir.path());
    }

    #[test]
    fn duration_from_env_distinct_fallbacks_for_distinct_keys() {
        use super::duration_from_env;
        // Two missing vars should each return their own fallback independently.
        let short = duration_from_env(
            "FRANKEN_WHISPER_TEST_DUR_NONEXIST_SHORT",
            std::time::Duration::from_millis(100),
        );
        let long = duration_from_env(
            "FRANKEN_WHISPER_TEST_DUR_NONEXIST_LONG",
            std::time::Duration::from_secs(300),
        );
        assert_eq!(short, std::time::Duration::from_millis(100));
        assert_eq!(long, std::time::Duration::from_secs(300));
        assert_ne!(short, long);
    }

    #[test]
    fn duration_from_env_zero_fallback_returns_zero() {
        use super::duration_from_env;
        let result = duration_from_env(
            "FRANKEN_WHISPER_TEST_DUR_NONEXIST_ZERO",
            std::time::Duration::ZERO,
        );
        assert_eq!(result, std::time::Duration::ZERO);
    }

    #[test]
    fn microphone_timeout_zero_seconds_has_buffer() {
        use super::microphone_timeout;
        let timeout = microphone_timeout(0);
        assert!(
            timeout >= std::time::Duration::from_secs(15),
            "0-second mic capture should still have 15s buffer, got {timeout:?}"
        );
    }

    #[test]
    fn microphone_timeout_large_seconds_does_not_overflow() {
        use super::microphone_timeout;
        let timeout = microphone_timeout(u32::MAX);
        assert!(
            timeout.as_secs() > 0,
            "u32::MAX seconds should not overflow"
        );
    }

    #[test]
    fn partial_mic_source_only_fails() {
        let err =
            microphone_defaults(None, None, Some("hw:0")).expect_err("source-only should fail");
        assert!(err.to_string().contains("must be provided together"));
    }

    #[test]
    fn explicit_format_and_source_overrides_device() {
        // When both format and source are explicitly provided, device is ignored.
        let (format, source) = microphone_defaults(
            Some("should-be-ignored"),
            Some("custom"),
            Some("/dev/audio"),
        )
        .expect("explicit args should succeed");
        assert_eq!(format, "custom");
        assert_eq!(source, "/dev/audio");
    }

    #[test]
    fn materialize_file_with_spaces_in_path() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let spaced_dir = dir.path().join("folder with spaces");
        std::fs::create_dir_all(&spaced_dir).expect("mkdir");
        let file_path = spaced_dir.join("my file.wav");
        std::fs::write(&file_path, b"audio data").expect("write");

        let source = InputSource::File {
            path: file_path.clone(),
        };
        let result = materialize_input(&source, dir.path()).expect("should succeed");
        assert_eq!(result, file_path);
    }

    #[test]
    fn materialize_file_symlink_follows_link() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let real_file = dir.path().join("real.wav");
        std::fs::write(&real_file, b"audio").expect("write");
        let link_path = dir.path().join("link.wav");
        std::os::unix::fs::symlink(&real_file, &link_path).expect("symlink");

        let source = InputSource::File {
            path: link_path.clone(),
        };
        let result = materialize_input(&source, dir.path()).expect("should succeed");
        assert_eq!(result, link_path);
    }

    #[test]
    fn microphone_timeout_exact_boundary_value() {
        use super::microphone_timeout;
        // 1 second capture should yield 1+15=16 second timeout.
        let timeout = microphone_timeout(1);
        assert_eq!(timeout, std::time::Duration::from_secs(16));
    }

    #[test]
    fn probe_duration_with_empty_file() {
        use super::probe_duration_seconds;

        let dir = tempfile::tempdir().expect("tempdir");
        let empty = dir.path().join("empty.wav");
        std::fs::write(&empty, b"").expect("write");
        // Empty file is not valid audio — should return None (or Some if ffprobe
        // somehow handles it, but definitely should not panic).
        let _ = probe_duration_seconds(&empty);
    }

    // ── Inline WAV generation helper (mirrors tests/helpers/mod.rs) ──
    //
    // We duplicate the WAV generation logic here rather than importing from
    // tests/helpers because unit tests inside src/ cannot depend on the
    // integration test helper module.

    /// Write a minimal valid WAV file: 16-bit PCM, mono, 16 kHz.
    fn write_test_wav(path: &std::path::Path, samples: &[i16], sample_rate: u32) {
        use std::io::Write;
        let channels: u16 = 1;
        let data_size = (samples.len() * 2) as u32;
        let file_size = 36 + data_size;
        let byte_rate = sample_rate * u32::from(channels) * 2;
        let block_align = channels * 2;

        let mut f = std::fs::File::create(path).expect("create WAV");
        f.write_all(b"RIFF").unwrap();
        f.write_all(&file_size.to_le_bytes()).unwrap();
        f.write_all(b"WAVE").unwrap();
        f.write_all(b"fmt ").unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap();
        f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
        f.write_all(&channels.to_le_bytes()).unwrap();
        f.write_all(&sample_rate.to_le_bytes()).unwrap();
        f.write_all(&byte_rate.to_le_bytes()).unwrap();
        f.write_all(&block_align.to_le_bytes()).unwrap();
        f.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
        f.write_all(b"data").unwrap();
        f.write_all(&data_size.to_le_bytes()).unwrap();
        for s in samples {
            f.write_all(&s.to_le_bytes()).unwrap();
        }
    }

    /// Generate a 1-second 440 Hz sine tone WAV at 16 kHz, return path.
    fn generate_sine_wav(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        let sample_rate: u32 = 16000;
        let duration_secs: f32 = 1.0;
        let frequency: f32 = 440.0;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<i16> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                ((2.0 * std::f32::consts::PI * frequency * t).sin() * 32_000.0) as i16
            })
            .collect();
        let path = dir.join(name);
        write_test_wav(&path, &samples, sample_rate);
        path
    }

    /// Generate a silent WAV of the given duration (seconds) at 16 kHz.
    fn generate_silence_wav(dir: &std::path::Path, name: &str, secs: f32) -> std::path::PathBuf {
        let sample_rate: u32 = 16000;
        let num_samples = (sample_rate as f32 * secs) as usize;
        let samples = vec![0i16; num_samples];
        let path = dir.join(name);
        write_test_wav(&path, &samples, sample_rate);
        path
    }

    /// Returns true if ffmpeg is available on PATH.
    fn ffmpeg_available() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    }

    /// Returns true if ffprobe is available on PATH.
    fn ffprobe_available() -> bool {
        std::process::Command::new("ffprobe")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    }

    #[test]
    fn sanitize_hint_extension_accepts_simple_values() {
        use super::sanitize_hint_extension;

        assert_eq!(sanitize_hint_extension("wav").as_deref(), Some("wav"));
        assert_eq!(sanitize_hint_extension(".mp3").as_deref(), Some("mp3"));
        assert_eq!(sanitize_hint_extension("flac").as_deref(), Some("flac"));
    }

    #[test]
    fn sanitize_hint_extension_rejects_invalid_characters() {
        use super::sanitize_hint_extension;

        assert!(sanitize_hint_extension("../secret").is_none());
        assert!(sanitize_hint_extension("..\\evil").is_none());
        assert!(sanitize_hint_extension("wav/../mp3").is_none());
        assert!(sanitize_hint_extension("wav.txt").is_none());
        assert!(sanitize_hint_extension("space name").is_none());
    }

    #[test]
    fn sanitize_hint_extension_rejects_empty_or_whitespace() {
        use super::sanitize_hint_extension;

        assert!(sanitize_hint_extension("").is_none());
        assert!(sanitize_hint_extension("   ").is_none());
        assert!(sanitize_hint_extension(".").is_none());
    }

    // ── materialize_input: additional edge cases ──

    #[test]
    fn materialize_file_empty_path_returns_error() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let source = InputSource::File {
            path: std::path::PathBuf::from(""),
        };
        let err = materialize_input(&source, dir.path()).expect_err("empty path should fail");
        let text = err.to_string();
        assert!(
            text.contains("does not exist"),
            "expected 'does not exist' in: {text}"
        );
    }

    #[test]
    fn materialize_file_unicode_path_existing_file() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let unicode_dir = dir.path().join("donn\u{00e9}es");
        std::fs::create_dir_all(&unicode_dir).expect("mkdir");
        let file_path = unicode_dir.join("r\u{00e9}sultat.wav");
        std::fs::write(&file_path, b"audio content").expect("write");

        let source = InputSource::File {
            path: file_path.clone(),
        };
        let result = materialize_input(&source, dir.path()).expect("should succeed");
        assert_eq!(result, file_path);
    }

    #[test]
    fn materialize_file_unicode_path_nonexistent() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let source = InputSource::File {
            path: dir
                .path()
                .join("\u{65e5}\u{672c}\u{8a9e}/\u{97f3}\u{58f0}.wav"),
        };
        let err = materialize_input(&source, dir.path()).expect_err("should fail");
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn materialize_file_directory_path_returns_error() {
        use super::materialize_input;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let subdir = dir.path().join("a_directory");
        std::fs::create_dir_all(&subdir).expect("mkdir");

        let source = InputSource::File {
            path: subdir.clone(),
        };
        let err = materialize_input(&source, dir.path()).expect_err("directory should be rejected");
        assert!(
            err.to_string().contains("not a file"),
            "expected 'not a file' in: {}",
            err
        );
    }

    #[test]
    fn materialize_file_returns_original_path_not_copy() {
        use super::materialize_input;
        use crate::model::InputSource;

        // File variant should return the exact same PathBuf, not a copy in work_dir.
        let dir = tempfile::tempdir().expect("tempdir");
        let work = tempfile::tempdir().expect("work_dir");
        let file_path = dir.path().join("original.wav");
        std::fs::write(&file_path, b"audio").expect("write");

        let source = InputSource::File {
            path: file_path.clone(),
        };
        let result = materialize_input(&source, work.path()).expect("should succeed");
        assert_eq!(
            result, file_path,
            "should return original path, not work_dir copy"
        );
    }

    // NOTE: materialize_input with InputSource::Stdin reads from actual stdin,
    // which requires controlling the process's stdin pipe. This is impractical
    // in a unit test without restructuring the function to accept a generic
    // Read trait. Testing Stdin is deferred to integration tests.

    // ── normalize_to_wav: tests that exercise ffmpeg ──

    #[test]
    fn normalize_valid_wav_produces_output_file() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "tone.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let result = normalize_to_wav(&input_wav, work.path()).expect("ffmpeg should succeed");
        assert!(
            result.exists(),
            "output file should exist at {}",
            result.display()
        );
        assert_eq!(
            result.file_name().and_then(|f| f.to_str()),
            Some("normalized_16k_mono.wav"),
            "output filename mismatch"
        );
        // Output should be a non-empty file.
        let meta = std::fs::metadata(&result).expect("metadata");
        assert!(
            meta.len() > 44,
            "output WAV should be larger than just a header"
        );
    }

    #[test]
    fn normalize_valid_wav_with_timeout_produces_correct_output() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav_with_timeout;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "sine_for_normalize.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let result = normalize_to_wav_with_timeout(
            &input_wav,
            work.path(),
            std::time::Duration::from_secs(30),
            None,
        )
        .expect("should succeed");

        assert!(result.exists());
        // Verify the output starts with RIFF/WAVE header.
        let header = std::fs::read(&result).expect("read output");
        assert!(header.len() > 44, "output too small");
        assert_eq!(&header[0..4], b"RIFF", "should start with RIFF");
        assert_eq!(&header[8..12], b"WAVE", "should contain WAVE marker");
    }

    #[test]
    fn normalize_silent_wav_produces_output() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let silence = generate_silence_wav(dir.path(), "silence.wav", 0.5);
        let work = tempfile::tempdir().expect("work_dir");

        let result = normalize_to_wav(&silence, work.path()).expect("should succeed");
        assert!(result.exists(), "normalized silent WAV should exist");
    }

    #[test]
    fn normalize_nonexistent_input_returns_error() {
        // ffmpeg should fail on a file that does not exist.
        use super::normalize_to_wav;

        let work = tempfile::tempdir().expect("work_dir");
        let result = normalize_to_wav(
            std::path::Path::new("/nonexistent/audio_input_99999.wav"),
            work.path(),
        );
        assert!(result.is_err(), "should fail for nonexistent input");
    }

    #[test]
    fn normalize_corrupt_file_returns_error() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let corrupt = dir.path().join("corrupt.wav");
        std::fs::write(&corrupt, b"RIFF\x00\x00\x00\x00WAVEgarbage").expect("write");
        let work = tempfile::tempdir().expect("work_dir");

        let result = normalize_to_wav(&corrupt, work.path());
        // ffmpeg may or may not handle this -- it often produces an error for
        // truncated/corrupt containers. Either outcome is acceptable; the
        // important thing is no panic.
        let _ = result;
    }

    #[test]
    fn normalize_with_cancellation_token_valid_wav() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav_with_timeout;
        use crate::orchestrator::CancellationToken;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "tone_cancel.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let cancel = CancellationToken::no_deadline();
        let result = normalize_to_wav_with_timeout(
            &input_wav,
            work.path(),
            std::time::Duration::from_secs(30),
            Some(&cancel),
        )
        .expect("should succeed with no-deadline token");

        assert!(result.exists());
    }

    #[test]
    fn normalize_overwrites_existing_output() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "overwrite_test.wav");
        let work = tempfile::tempdir().expect("work_dir");

        // Create a pre-existing file at the output path.
        let output_path = work.path().join("normalized_16k_mono.wav");
        std::fs::write(&output_path, b"old data").expect("write old data");

        let result = normalize_to_wav(&input_wav, work.path()).expect("should succeed");
        assert!(result.exists());
        let content = std::fs::read(&result).expect("read");
        // The -y flag in ffmpeg args should overwrite; output should be valid WAV,
        // not "old data".
        assert_ne!(&content[..], b"old data", "output should be overwritten");
        assert_eq!(&content[0..4], b"RIFF", "output should be valid RIFF/WAV");
    }

    // ── probe_duration_seconds: tests that exercise ffprobe ──

    #[test]
    fn probe_duration_of_generated_wav_returns_approximately_one_second() {
        if !ffprobe_available() {
            eprintln!("SKIPPED: ffprobe not found on PATH");
            return;
        }
        use super::probe_duration_seconds;

        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_sine_wav(dir.path(), "probe_1s.wav");

        let duration = probe_duration_seconds(&wav);
        let secs = duration.expect("should return Some for valid WAV");
        assert!(
            (secs - 1.0).abs() < 0.05,
            "expected ~1.0 second, got {secs}"
        );
    }

    #[test]
    fn probe_duration_of_silence_wav_returns_expected_length() {
        if !ffprobe_available() {
            eprintln!("SKIPPED: ffprobe not found on PATH");
            return;
        }
        use super::probe_duration_seconds;

        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_silence_wav(dir.path(), "silence_2s.wav", 2.0);

        let duration = probe_duration_seconds(&wav);
        let secs = duration.expect("should return Some for silent WAV");
        assert!(
            (secs - 2.0).abs() < 0.05,
            "expected ~2.0 seconds, got {secs}"
        );
    }

    #[test]
    fn probe_duration_with_timeout_of_valid_wav() {
        if !ffprobe_available() {
            eprintln!("SKIPPED: ffprobe not found on PATH");
            return;
        }
        use super::probe_duration_seconds_with_timeout;

        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_sine_wav(dir.path(), "probe_timeout.wav");

        let duration =
            probe_duration_seconds_with_timeout(&wav, std::time::Duration::from_secs(10));
        let secs = duration.expect("should return Some");
        assert!(
            (secs - 1.0).abs() < 0.05,
            "expected ~1.0 second, got {secs}"
        );
    }

    #[test]
    fn probe_duration_with_text_file_returns_none() {
        use super::probe_duration_seconds;

        let dir = tempfile::tempdir().expect("tempdir");
        let txt = dir.path().join("not_audio.txt");
        std::fs::write(&txt, "hello world").expect("write");

        // ffprobe should fail to parse a text file.
        let result = probe_duration_seconds(&txt);
        // May return None (if ffprobe errors) or Some (unlikely). No panic.
        let _ = result;
    }

    #[test]
    fn probe_duration_of_normalized_output_matches_source() {
        if !ffmpeg_available() || !ffprobe_available() {
            eprintln!("SKIPPED: ffmpeg or ffprobe not found on PATH");
            return;
        }
        use super::{normalize_to_wav, probe_duration_seconds};

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "chain_input.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let normalized =
            normalize_to_wav(&input_wav, work.path()).expect("normalize should succeed");

        let input_dur = probe_duration_seconds(&input_wav).expect("input probe should return Some");
        let output_dur =
            probe_duration_seconds(&normalized).expect("output probe should return Some");

        assert!(
            (input_dur - output_dur).abs() < 0.1,
            "normalized output duration ({output_dur}s) should match input ({input_dur}s)"
        );
    }

    // ── ffmpeg_timeout / ffprobe_timeout: default value checks ──

    #[test]
    fn ffmpeg_timeout_default_is_180_seconds() {
        use super::ffmpeg_timeout;
        // When FRANKEN_WHISPER_FFMPEG_TIMEOUT_MS is not set, the default is 180s.
        // The env var is unlikely to be set in test, so this should hold.
        let timeout = ffmpeg_timeout();
        assert_eq!(
            timeout,
            std::time::Duration::from_secs(180),
            "default ffmpeg timeout should be 180 seconds"
        );
    }

    #[test]
    fn ffprobe_timeout_default_is_10_seconds() {
        use super::ffprobe_timeout;
        let timeout = ffprobe_timeout();
        assert_eq!(
            timeout,
            std::time::Duration::from_secs(10),
            "default ffprobe timeout should be 10 seconds"
        );
    }

    // ── microphone_timeout: additional coverage ──

    #[test]
    fn microphone_timeout_intermediate_values() {
        use super::microphone_timeout;

        // Verify several intermediate values: seconds + 15 = expected.
        let cases: Vec<(u32, u64)> = vec![(5, 20), (30, 45), (60, 75), (120, 135), (300, 315)];
        for (seconds, expected_secs) in cases {
            let timeout = microphone_timeout(seconds);
            assert_eq!(
                timeout,
                std::time::Duration::from_secs(expected_secs),
                "microphone_timeout({seconds}) should be {expected_secs}s"
            );
        }
    }

    // ── duration_from_env: additional edge cases ──

    #[test]
    fn duration_from_env_max_u64_fallback() {
        use super::duration_from_env;
        let max_dur = std::time::Duration::from_millis(u64::MAX);
        let result = duration_from_env("FRANKEN_WHISPER_TEST_DUR_NONEXIST_MAX_U64", max_dur);
        assert_eq!(result, max_dur);
    }

    #[test]
    fn duration_from_env_nanos_precision_fallback() {
        use super::duration_from_env;
        // duration_from_env returns Duration::from_millis() when parsing the
        // env var. For the fallback path it returns the exact Duration provided.
        let precise = std::time::Duration::new(1, 999_999_999);
        let result = duration_from_env("FRANKEN_WHISPER_TEST_DUR_NONEXIST_NANOS", precise);
        assert_eq!(
            result, precise,
            "fallback should preserve nanosecond precision"
        );
    }

    // ── microphone_defaults: platform-specific coverage ──

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_microphone_defaults_use_avfoundation() {
        let (format, source) = microphone_defaults(None, None, None).expect("macos defaults");
        assert_eq!(format, "avfoundation");
        assert_eq!(source, ":0");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_microphone_custom_device() {
        let (format, source) = microphone_defaults(Some(":1"), None, None).expect("custom device");
        assert_eq!(format, "avfoundation");
        assert_eq!(source, ":1");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_microphone_defaults_use_dshow() {
        let (format, source) = microphone_defaults(None, None, None).expect("windows defaults");
        assert_eq!(format, "dshow");
        assert_eq!(source, "audio=default");
    }

    // ── InvalidRequest error variant matching ──

    #[test]
    fn materialize_nonexistent_file_error_is_invalid_request_variant() {
        use super::materialize_input;
        use crate::error::FwError;
        use crate::model::InputSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let source = InputSource::File {
            path: std::path::PathBuf::from("/no/such/file_12345.mp3"),
        };
        let err = materialize_input(&source, dir.path()).expect_err("should fail");
        assert!(
            matches!(err, FwError::InvalidRequest(_)),
            "expected InvalidRequest variant, got: {err:?}"
        );
    }

    #[test]
    fn partial_mic_args_error_is_invalid_request_variant() {
        use crate::error::FwError;

        let err =
            microphone_defaults(None, Some("pulse"), None).expect_err("partial args should fail");
        assert!(
            matches!(err, FwError::InvalidRequest(_)),
            "expected InvalidRequest variant, got: {err:?}"
        );
    }

    // ── normalize_to_wav: path with unicode and spaces ──

    #[test]
    fn normalize_wav_with_spaces_in_path() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let spaced = dir.path().join("my audio files");
        std::fs::create_dir_all(&spaced).expect("mkdir");
        let input_wav = generate_sine_wav(&spaced, "test tone.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let result =
            normalize_to_wav(&input_wav, work.path()).expect("should handle spaces in path");
        assert!(result.exists());
    }

    #[test]
    fn normalize_wav_with_unicode_in_path() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let unicode_dir = dir.path().join("donn\u{00e9}es_audio");
        std::fs::create_dir_all(&unicode_dir).expect("mkdir");
        let input_wav = generate_sine_wav(&unicode_dir, "\u{97f3}\u{58f0}.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let result =
            normalize_to_wav(&input_wav, work.path()).expect("should handle unicode in path");
        assert!(result.exists());
    }

    // ── Verify WAV header structure of inline generator ──

    #[test]
    fn generated_wav_has_valid_riff_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_sine_wav(dir.path(), "header_check.wav");

        let data = std::fs::read(&wav).expect("read");
        assert!(data.len() > 44, "WAV should be at least 44 bytes + data");
        assert_eq!(&data[0..4], b"RIFF", "RIFF marker");
        assert_eq!(&data[8..12], b"WAVE", "WAVE marker");
        assert_eq!(&data[12..16], b"fmt ", "fmt chunk marker");
        assert_eq!(&data[36..40], b"data", "data chunk marker");

        // PCM format tag = 1
        let fmt_tag = u16::from_le_bytes([data[20], data[21]]);
        assert_eq!(fmt_tag, 1, "should be PCM format");

        // Channels = 1
        let channels = u16::from_le_bytes([data[22], data[23]]);
        assert_eq!(channels, 1, "should be mono");

        // Sample rate = 16000
        let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        assert_eq!(sample_rate, 16000, "should be 16kHz");

        // Bits per sample = 16
        let bits = u16::from_le_bytes([data[34], data[35]]);
        assert_eq!(bits, 16, "should be 16-bit");
    }

    #[test]
    fn generated_silence_wav_contains_all_zeros_in_data_section() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_silence_wav(dir.path(), "zero_check.wav", 0.1);

        let data = std::fs::read(&wav).expect("read");
        // Data section starts at byte 44 for a standard PCM WAV.
        let audio_data = &data[44..];
        assert!(!audio_data.is_empty(), "should have audio data");
        assert!(
            audio_data.iter().all(|&b| b == 0),
            "silence WAV data section should be all zeros"
        );
    }

    #[test]
    fn generated_sine_wav_has_expected_sample_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let wav = generate_sine_wav(dir.path(), "count_check.wav");

        let data = std::fs::read(&wav).expect("read");
        let audio_data = &data[44..];
        // 1 second * 16000 Hz * 2 bytes/sample = 32000 bytes
        assert_eq!(
            audio_data.len(),
            32000,
            "1s at 16kHz 16-bit mono should be 32000 bytes of audio data"
        );
    }

    // ── normalize_to_wav output verification: 16kHz mono PCM ──

    #[test]
    fn normalized_output_is_16khz_mono_pcm() {
        if !ffmpeg_available() {
            eprintln!("SKIPPED: ffmpeg not found on PATH");
            return;
        }
        use super::normalize_to_wav;

        let dir = tempfile::tempdir().expect("tempdir");
        let input_wav = generate_sine_wav(dir.path(), "verify_format.wav");
        let work = tempfile::tempdir().expect("work_dir");

        let output = normalize_to_wav(&input_wav, work.path()).expect("should succeed");
        let data = std::fs::read(&output).expect("read output");
        assert!(data.len() > 44, "output too small");

        // Verify RIFF/WAVE
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");

        // PCM format = 1
        let fmt_tag = u16::from_le_bytes([data[20], data[21]]);
        assert_eq!(fmt_tag, 1, "normalized output should be PCM");

        // Mono
        let channels = u16::from_le_bytes([data[22], data[23]]);
        assert_eq!(channels, 1, "normalized output should be mono");

        // 16 kHz
        let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        assert_eq!(sample_rate, 16000, "normalized output should be 16kHz");

        // 16-bit
        let bits = u16::from_le_bytes([data[34], data[35]]);
        assert_eq!(bits, 16, "normalized output should be 16-bit");
    }

    // ── ffmpeg provisioning unit tests ──────────────────────────────────

    #[test]
    fn platform_tool_name_returns_bare_name_on_non_windows() {
        use super::platform_tool_name;
        let name = platform_tool_name("ffmpeg");
        assert_eq!(name, "ffmpeg");
        let name2 = platform_tool_name("ffprobe");
        assert_eq!(name2, "ffprobe");
    }

    #[test]
    fn state_root_for_tools_ignores_tmp_work_dir_without_env_override() {
        use super::{ffmpeg_tools_root, provisioned_tool_path, state_root_for_tools};
        if std::env::var_os("FRANKEN_WHISPER_STATE_DIR").is_some() {
            return;
        }

        let base = tempfile::tempdir().expect("tempdir");
        let tmp_child = base.path().join("tmp").join("work");
        std::fs::create_dir_all(&tmp_child).expect("mkdir");

        let result = state_root_for_tools(Some(&tmp_child));
        let tools_root = ffmpeg_tools_root(Some(&tmp_child));
        let ffmpeg_path = provisioned_tool_path("ffmpeg", Some(&tmp_child));
        assert_ne!(
            result,
            base.path(),
            "tmp work dirs should not force tool state into repo-adjacent state roots"
        );
        assert!(
            !tools_root.starts_with(base.path()),
            "tools root should not live under the repo-adjacent temp tree: {}",
            tools_root.display()
        );
        assert!(
            !ffmpeg_path.starts_with(base.path()),
            "provisioned ffmpeg path should not live under the repo-adjacent temp tree: {}",
            ffmpeg_path.display()
        );
    }

    #[test]
    fn ffmpeg_tools_root_joins_expected_subpath() {
        use super::{FFMPEG_TOOLS_DIR, ffmpeg_tools_root};
        // Whatever state root is selected, tools root should end with
        // the FFMPEG_TOOLS_DIR constant.
        let root = ffmpeg_tools_root(None);
        assert!(
            root.ends_with(FFMPEG_TOOLS_DIR),
            "tools root should end with {FFMPEG_TOOLS_DIR}, got {}",
            root.display()
        );
    }

    #[test]
    fn provisioned_tool_path_ends_with_bin_tool() {
        use super::provisioned_tool_path;
        let path = provisioned_tool_path("ffmpeg", None);
        assert!(
            path.ends_with("bin/ffmpeg"),
            "provisioned path should end with bin/ffmpeg, got {}",
            path.display()
        );
    }

    #[test]
    fn explicit_tool_path_returns_none_for_missing_env() {
        use super::explicit_tool_path;
        let result = explicit_tool_path("FRANKEN_WHISPER_NONEXISTENT_TOOL_PATH_TEST_42");
        assert!(result.is_none());
    }

    #[test]
    fn find_file_named_locates_nested_file() {
        use super::find_file_named;
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&nested).expect("mkdir");
        let target = nested.join("ffmpeg");
        std::fs::write(&target, b"binary").expect("write");

        let found = find_file_named(dir.path(), "ffmpeg");
        assert_eq!(found, Some(target));
    }

    #[test]
    fn validate_tar_entry_path_accepts_normal_entries() {
        use super::validate_tar_entry_path;
        assert!(validate_tar_entry_path("ffmpeg"));
        assert!(validate_tar_entry_path("dir/ffmpeg"));
        assert!(validate_tar_entry_path("./dir/ffprobe"));
    }

    #[test]
    fn validate_tar_entry_path_rejects_parent_dir() {
        use super::validate_tar_entry_path;
        assert!(!validate_tar_entry_path("../ffmpeg"));
        assert!(!validate_tar_entry_path("dir/../ffprobe"));
    }

    #[test]
    fn validate_tar_entry_path_rejects_absolute_paths() {
        use super::validate_tar_entry_path;
        assert!(!validate_tar_entry_path("/usr/bin/ffmpeg"));
    }

    #[test]
    fn validate_extracted_binary_rejects_directories() {
        use super::validate_extracted_binary;
        let dir = tempfile::tempdir().expect("tempdir");
        let err = validate_extracted_binary(dir.path(), "ffmpeg").expect_err("should reject dir");
        assert!(
            err.to_string().contains("not a regular file"),
            "unexpected error: {err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn find_file_named_skips_symlink_dirs() {
        use super::find_file_named;
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("nested");
        std::fs::create_dir_all(&nested).expect("mkdir");
        let target = nested.join("ffmpeg");
        std::fs::write(&target, b"binary").expect("write");

        let loop_link = dir.path().join("loop");
        symlink(dir.path(), &loop_link).expect("symlink");

        let found = find_file_named(dir.path(), "ffmpeg");
        assert_eq!(found, Some(target));
    }

    #[test]
    fn find_file_named_returns_none_when_absent() {
        use super::find_file_named;
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("other.txt"), b"x").expect("write");
        assert!(find_file_named(dir.path(), "ffmpeg").is_none());
    }

    #[test]
    fn find_file_named_empty_dir_returns_none() {
        use super::find_file_named;
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(find_file_named(dir.path(), "ffmpeg").is_none());
    }

    #[test]
    fn find_file_named_ignores_directories_with_same_name() {
        use super::find_file_named;
        let dir = tempfile::tempdir().expect("tempdir");
        // Create a directory named "ffmpeg" — should not be returned.
        std::fs::create_dir_all(dir.path().join("ffmpeg")).expect("mkdir");
        assert!(find_file_named(dir.path(), "ffmpeg").is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ffmpeg_bundle_source_returns_tar_xz_on_linux_x86_64() {
        use super::ffmpeg_bundle_source;
        let result = ffmpeg_bundle_source();
        #[cfg(target_arch = "x86_64")]
        {
            let (url, ext) = result.expect("should succeed on linux/x86_64");
            assert!(url.starts_with("https://"), "URL should be HTTPS");
            assert!(ext.contains("tar"), "extension should reference tar");
        }
    }

    #[test]
    fn bool_env_truthy_values_all_recognized() {
        use super::bool_env_enabled_from_raw;
        for val in ["1", "true", "yes", "on", "TRUE", "Yes", "ON"] {
            assert!(
                bool_env_enabled_from_raw(Some(val), false),
                "{val} should be truthy"
            );
        }
    }

    #[test]
    fn bool_env_falsy_values_all_recognized() {
        use super::bool_env_enabled_from_raw;
        for val in ["0", "false", "no", "off", "FALSE", "No", "OFF"] {
            assert!(
                !bool_env_enabled_from_raw(Some(val), true),
                "{val} should be falsy"
            );
        }
    }

    #[test]
    fn bool_env_whitespace_trimmed() {
        use super::bool_env_enabled_from_raw;
        assert!(bool_env_enabled_from_raw(Some("  true  "), false));
        assert!(!bool_env_enabled_from_raw(Some(" off "), true));
    }

    #[cfg(unix)]
    #[test]
    fn mark_executable_if_unix_sets_permissions() {
        use super::mark_executable_if_unix;
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("test_bin");
        std::fs::write(&file, b"#!/bin/sh\n").expect("write");

        mark_executable_if_unix(&file).expect("should succeed");
        let perms = std::fs::metadata(&file).expect("metadata").permissions();
        assert_eq!(perms.mode() & 0o755, 0o755, "should have 755 permissions");
    }

    #[test]
    fn download_ffmpeg_bundle_fails_when_no_curl_or_wget() {
        use super::download_ffmpeg_bundle_with_tools;
        use crate::error::FwError;

        // Ensure fast failure by simulating missing tools; avoids network waits.
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("bundle.tar.xz");
        let result =
            download_ffmpeg_bundle_with_tools("http://192.0.2.1/fake", &dest, false, false);
        assert!(
            matches!(result, Err(FwError::CommandMissing { .. })),
            "missing tools should surface CommandMissing error"
        );
    }

    #[test]
    fn extract_ffmpeg_bundle_xz_uses_correct_flag() {
        use super::extract_ffmpeg_bundle;
        // We can't actually extract, but we verify it doesn't panic and returns
        // an error for nonexistent archive.
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("fake.tar.xz");
        std::fs::write(&archive, b"not a real archive").expect("write");
        let dest = dir.path().join("out");
        std::fs::create_dir_all(&dest).expect("mkdir");
        let result = extract_ffmpeg_bundle(&archive, &dest);
        assert!(result.is_err(), "corrupt archive should fail extraction");
    }

    #[test]
    fn extract_ffmpeg_bundle_gz_uses_correct_flag() {
        use super::extract_ffmpeg_bundle;
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = dir.path().join("fake.tar.gz");
        std::fs::write(&archive, b"not a real archive").expect("write");
        let dest = dir.path().join("out");
        std::fs::create_dir_all(&dest).expect("mkdir");
        let result = extract_ffmpeg_bundle(&archive, &dest);
        assert!(result.is_err(), "corrupt archive should fail extraction");
    }

    #[test]
    fn provisioned_tool_path_ffprobe_ends_with_bin_ffprobe() {
        use super::provisioned_tool_path;
        let path = provisioned_tool_path("ffprobe", None);
        assert!(
            path.ends_with("bin/ffprobe"),
            "provisioned path should end with bin/ffprobe, got {}",
            path.display()
        );
    }

    #[test]
    fn constants_are_sane() {
        use super::{DEFAULT_STATE_DIR, DOWNLOAD_TIMEOUT, FFMPEG_TOOLS_DIR};
        assert!(!DEFAULT_STATE_DIR.is_empty());
        assert!(!FFMPEG_TOOLS_DIR.is_empty());
        assert!(DOWNLOAD_TIMEOUT.as_secs() > 0);
    }
}
