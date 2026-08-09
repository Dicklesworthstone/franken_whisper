//! Explicit, hash-pinned distribution for native model packages.
//!
//! The binary contains only small immutable manifests. `fw pull whisper` and
//! `fw pull sortformer` stream release artifacts into a per-user cache.
//! Inference is offline and admits a cache entry only after its compiled size
//! and SHA-256 trust roots match.

#![forbid(unsafe_code)]

use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::bytes::Buf;
use asupersync::http::h1::HttpError;
use asupersync::http::{Body, Client, ClientError, Method};
use asupersync::runtime::RuntimeBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{FwError, FwResult};

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const BUILTIN_MANIFEST_JSON: &str = include_str!("../models/sortformer-manifest-v1.json");
pub const BUILTIN_WHISPER_MANIFEST_JSON: &str =
    include_str!("../models/whisper-manifest-v1.json");
pub const SORTFORMER_ARTIFACT_VERSION: &str = "sortformer-v2.1-f32-v1";
pub const SORTFORMER_DISTRIBUTION_POLICY: &str = "github_release_with_license_and_notice";
pub const SORTFORMER_MODEL_DIR_ENV: &str = "FRANKEN_WHISPER_MODEL_DIR";
pub const SORTFORMER_WEIGHTS_FILENAME: &str = "weights.safetensors";
pub const SORTFORMER_RECEIPT_FILENAME: &str = "conversion-receipt.json";
pub const SORTFORMER_LICENSE_FILENAME: &str = "NVIDIA-OPEN-MODEL-LICENSE.html";
pub const SORTFORMER_NOTICE_FILENAME: &str = "NOTICE.sortformer.txt";
pub const SORTFORMER_REQUIRED_NOTICE: &str =
    "Licensed by NVIDIA Corporation under the NVIDIA Open Model License";
pub const WHISPER_ARTIFACT_VERSION: &str = "whisper-large-v3-turbo-f16-v1";
pub const WHISPER_MODEL_ID: &str = "openai/whisper-large-v3-turbo";
pub const WHISPER_UPSTREAM_REPOSITORY: &str = "ggerganov/whisper.cpp";
pub const WHISPER_UPSTREAM_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
pub const WHISPER_WEIGHTS_FILENAME: &str = "ggml-large-v3-turbo.bin";
pub const WHISPER_WEIGHTS_BYTES: u64 = 1_624_555_275;
pub const WHISPER_WEIGHTS_SHA256: &str =
    "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69";
pub const WHISPER_DISTRIBUTION_POLICY: &str = "github_release_with_compiled_sha256";
pub const WHISPER_PREPARATION_RECIPE: &str =
    "franken-whisper-native-ggml-selection-v1-identity";
pub const WHISPER_LICENSE_URL: &str = "https://github.com/openai/whisper/blob/main/LICENSE";

const MAX_ARTIFACT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_URL_BYTES: usize = 4096;
const MAX_NAME_BYTES: usize = 255;
const HASH_BUFFER_BYTES: usize = 64 * 1024;
const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(50);
const INSTALL_LOCK_TIMEOUT: Duration = Duration::from_secs(30);
const INSTALL_LOCK_POLL_INTERVAL: Duration = Duration::from_millis(25);
const DOWNLOAD_REQUEST_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);
const EXPECTED_RELEASE_PREFIX: &str = "https://github.com/Dicklesworthstone/franken_whisper/releases/download/sortformer-v2.1-f32-v1/";
const EXPECTED_WHISPER_RELEASE_PREFIX: &str = "https://github.com/Dicklesworthstone/franken_whisper/releases/download/whisper-large-v3-turbo-f16-v1/";
static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub model_revision: String,
    pub artifact_version: String,
    pub conversion_recipe: String,
    pub distribution_policy: String,
    pub license_id: String,
    pub license_url: String,
    pub required_notice: String,
    pub speaker_lane_capacity: u16,
    pub files: Vec<RemoteFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhisperManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub upstream_repository: String,
    pub upstream_revision: String,
    pub artifact_version: String,
    pub preparation_recipe: String,
    pub distribution_policy: String,
    pub license_id: String,
    pub license_url: String,
    pub weight_precision: String,
    pub files: Vec<RemoteFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteFile {
    pub role: String,
    pub filename: String,
    pub size: u64,
    pub sha256: String,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedSortformerPackage {
    pub package_path: PathBuf,
    pub receipt_path: PathBuf,
    pub license_path: PathBuf,
    pub notice_path: PathBuf,
    pub artifact_version: String,
    pub package_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedWhisperPackage {
    pub weights_path: PathBuf,
    pub artifact_version: String,
    pub weights_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullOutcome {
    pub package: CachedSortformerPackage,
    pub from_cache: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperPullOutcome {
    pub package: CachedWhisperPackage,
    pub from_cache: bool,
}

pub fn builtin_manifest() -> FwResult<SortformerManifest> {
    parse_manifest(BUILTIN_MANIFEST_JSON.as_bytes())
}

pub fn builtin_whisper_manifest() -> FwResult<WhisperManifest> {
    let manifest: WhisperManifest = serde_json::from_slice(BUILTIN_WHISPER_MANIFEST_JSON.as_bytes())?;
    validate_whisper_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_whisper_manifest(manifest: &WhisperManifest) -> FwResult<()> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION
        || manifest.model_id != WHISPER_MODEL_ID
        || manifest.upstream_repository != WHISPER_UPSTREAM_REPOSITORY
        || manifest.upstream_revision != WHISPER_UPSTREAM_REVISION
        || manifest.artifact_version != WHISPER_ARTIFACT_VERSION
        || manifest.preparation_recipe != WHISPER_PREPARATION_RECIPE
        || manifest.distribution_policy != WHISPER_DISTRIBUTION_POLICY
        || manifest.license_id != "MIT"
        || manifest.license_url != WHISPER_LICENSE_URL
        || manifest.weight_precision != "f16"
        || manifest.files.len() != 1
    {
        return Err(whisper_manifest_error(
            "embedded Whisper manifest disagrees with the compiled runtime contract",
        ));
    }
    let file = &manifest.files[0];
    validate_filename(&file.filename)?;
    validate_hash(&file.sha256)?;
    validate_https_url(&file.url)?;
    if file.role != "weights"
        || file.filename != WHISPER_WEIGHTS_FILENAME
        || file.size != WHISPER_WEIGHTS_BYTES
        || file.sha256 != WHISPER_WEIGHTS_SHA256
        || file.url != format!("{EXPECTED_WHISPER_RELEASE_PREFIX}{WHISPER_WEIGHTS_FILENAME}")
        || file.size == 0
        || file.size > MAX_ARTIFACT_BYTES
    {
        return Err(whisper_manifest_error(
            "Whisper manifest file entry disagrees with its compiled trust root",
        ));
    }
    Ok(())
}

pub fn parse_manifest(bytes: &[u8]) -> FwResult<SortformerManifest> {
    let manifest: SortformerManifest = serde_json::from_slice(bytes)?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &SortformerManifest) -> FwResult<()> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION
        || manifest.model_id != crate::sortformer_conformance::SORTFORMER_MODEL_ID
        || manifest.model_revision != crate::sortformer_conformance::SORTFORMER_MODEL_REVISION
        || manifest.artifact_version != SORTFORMER_ARTIFACT_VERSION
        || manifest.conversion_recipe != "franken-whisper-native-sortformer-converter-v1-f32"
        || manifest.distribution_policy != SORTFORMER_DISTRIBUTION_POLICY
        || manifest.license_id != "NVIDIA Open Model License"
        || manifest.license_url != crate::sortformer_conformance::SORTFORMER_LICENSE_URL
        || manifest.required_notice != SORTFORMER_REQUIRED_NOTICE
        || manifest.speaker_lane_capacity != 4
    {
        return Err(manifest_error(
            "embedded Sortformer manifest disagrees with the compiled runtime contract",
        ));
    }
    if manifest.files.len() != 4 {
        return Err(manifest_error(
            "Sortformer manifest must contain exactly four required files",
        ));
    }

    let expected = [
        (
            "license",
            SORTFORMER_LICENSE_FILENAME,
            307_201,
            "ed91059bb4088adf04719686957b3e37cb372beead1cdbb115a930b17b85593c",
        ),
        (
            "notice",
            SORTFORMER_NOTICE_FILENAME,
            1_729,
            "2efddd2d681136ddede12497e67caf4cf2444f15de824ee671ebbc93fe276d13",
        ),
        (
            "conversion_receipt",
            SORTFORMER_RECEIPT_FILENAME,
            653_208,
            crate::sortformer_conformance::SORTFORMER_CONVERSION_RECEIPT_SHA256,
        ),
        (
            "weights",
            SORTFORMER_WEIGHTS_FILENAME,
            crate::sortformer_conformance::SORTFORMER_PACKAGE_BYTES,
            crate::sortformer_conformance::SORTFORMER_PACKAGE_SHA256,
        ),
    ];
    for (file, (role, filename, size, sha256)) in manifest.files.iter().zip(expected) {
        validate_filename(&file.filename)?;
        validate_hash(&file.sha256)?;
        validate_https_url(&file.url)?;
        if file.role != role
            || file.filename != filename
            || file.size != size
            || file.sha256 != sha256
            || file.url != format!("{EXPECTED_RELEASE_PREFIX}{filename}")
            || file.size == 0
            || file.size > MAX_ARTIFACT_BYTES
        {
            return Err(manifest_error(
                "Sortformer manifest file entry disagrees with its compiled trust root",
            ));
        }
    }
    Ok(())
}

fn manifest_error(message: impl Into<String>) -> FwError {
    FwError::ContractViolation(format!("sortformer.model_manifest: {}", message.into()))
}

fn whisper_manifest_error(message: impl Into<String>) -> FwError {
    FwError::ContractViolation(format!("whisper.model_manifest: {}", message.into()))
}

fn validate_filename(filename: &str) -> FwResult<()> {
    let valid_atom = !filename.is_empty()
        && filename.len() <= MAX_NAME_BYTES
        && filename
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    let mut components = Path::new(filename).components();
    let one_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if !valid_atom || !one_component || filename.ends_with('.') || is_windows_device_name(filename)
    {
        return Err(manifest_error(
            "artifact filename must be one portable non-reserved component",
        ));
    }
    Ok(())
}

fn is_windows_device_name(filename: &str) -> bool {
    let stem = filename.split('.').next().unwrap_or(filename);
    let upper = stem.to_ascii_uppercase();
    matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && matches!(upper.as_bytes()[3], b'1'..=b'9'))
}

fn validate_hash(hash: &str) -> FwResult<()> {
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(manifest_error(
            "artifact SHA-256 must be 64 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

fn validate_https_url(url: &str) -> FwResult<()> {
    let authority = url
        .strip_prefix("https://")
        .and_then(|rest| rest.split(['/', '?', '#']).next());
    if url.len() > MAX_URL_BYTES
        || authority.is_none_or(str::is_empty)
        || url
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
    {
        return Err(manifest_error("artifact URL must be a bounded HTTPS URL"));
    }
    Ok(())
}

#[must_use]
pub fn cache_root() -> Option<PathBuf> {
    if let Some(configured) = std::env::var_os(SORTFORMER_MODEL_DIR_ENV)
        && !configured.is_empty()
    {
        return Some(PathBuf::from(configured));
    }
    #[cfg(windows)]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return Some(PathBuf::from(local).join("franken_whisper").join("models"));
        }
        std::env::var_os("USERPROFILE").map(|profile| {
            PathBuf::from(profile)
                .join(".cache")
                .join("franken_whisper")
                .join("models")
        })
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(|home| {
            PathBuf::from(home)
                .join(".cache")
                .join("franken_whisper")
                .join("models")
        })
    }
}

pub fn sortformer_cache_dir() -> FwResult<PathBuf> {
    let root = cache_root().ok_or_else(|| {
        FwError::InvalidRequest(format!(
            "cannot resolve model cache; set {SORTFORMER_MODEL_DIR_ENV}"
        ))
    })?;
    if !root.is_absolute() {
        return Err(FwError::InvalidRequest(format!(
            "{SORTFORMER_MODEL_DIR_ENV} must be an absolute path"
        )));
    }
    Ok(root.join("sortformer").join(SORTFORMER_ARTIFACT_VERSION))
}

pub fn whisper_cache_dir() -> FwResult<PathBuf> {
    let root = cache_root().ok_or_else(|| {
        FwError::InvalidRequest(format!(
            "cannot resolve model cache; set {SORTFORMER_MODEL_DIR_ENV}"
        ))
    })?;
    if !root.is_absolute() {
        return Err(FwError::InvalidRequest(format!(
            "{SORTFORMER_MODEL_DIR_ENV} must be an absolute path"
        )));
    }
    Ok(root.join("whisper").join(WHISPER_ARTIFACT_VERSION))
}

pub fn resolve_cached_whisper() -> FwResult<CachedWhisperPackage> {
    resolve_cached_whisper_with_cancel(|| false)
}

pub fn resolve_cached_whisper_with_cancel<F>(
    is_cancelled: F,
) -> FwResult<CachedWhisperPackage>
where
    F: Fn() -> bool + Sync,
{
    let manifest = builtin_whisper_manifest()?;
    let directory = whisper_cache_dir()?;
    let remote = manifest
        .files
        .first()
        .ok_or_else(|| whisper_manifest_error("weights role is missing"))?;
    let weights_path = directory.join(&remote.filename);
    if !file_matches_with_cancel(&weights_path, remote, &is_cancelled)? {
        return Err(FwError::MissingArtifact(weights_path));
    }
    Ok(CachedWhisperPackage {
        weights_path,
        artifact_version: manifest.artifact_version,
        weights_sha256: WHISPER_WEIGHTS_SHA256.to_owned(),
    })
}

pub fn cached_whisper_readiness_with_cancel<F>(is_cancelled: F) -> FwResult<bool>
where
    F: Fn() -> bool + Sync,
{
    match resolve_cached_whisper_with_cancel(is_cancelled) {
        Ok(_) => Ok(true),
        Err(FwError::Cancelled(message)) => Err(FwError::Cancelled(message)),
        Err(_) => Ok(false),
    }
}

#[must_use]
pub fn cached_whisper_is_ready() -> bool {
    cached_whisper_readiness_with_cancel(|| false).unwrap_or(false)
}

pub fn resolve_cached_sortformer() -> FwResult<CachedSortformerPackage> {
    let manifest = builtin_manifest()?;
    let directory = sortformer_cache_dir()?;
    resolve_cached_in(&manifest, &directory)
}

/// Resolve and fully hash the cached package while honoring cooperative
/// cancellation between bounded reads.
pub fn resolve_cached_sortformer_with_cancel<F>(
    is_cancelled: F,
) -> FwResult<CachedSortformerPackage>
where
    F: Fn() -> bool + Sync,
{
    let manifest = builtin_manifest()?;
    let directory = sortformer_cache_dir()?;
    resolve_cached_in_with_cancel(&manifest, &directory, &is_cancelled)
}

fn resolve_cached_in(
    manifest: &SortformerManifest,
    directory: &Path,
) -> FwResult<CachedSortformerPackage> {
    resolve_cached_in_with_cancel(manifest, directory, &|| false)
}

fn resolve_cached_in_with_cancel<F>(
    manifest: &SortformerManifest,
    directory: &Path,
    is_cancelled: &F,
) -> FwResult<CachedSortformerPackage>
where
    F: Fn() -> bool + Sync,
{
    let mut package_path = None;
    let mut receipt_path = None;
    let mut license_path = None;
    let mut notice_path = None;
    for file in &manifest.files {
        let path = directory.join(&file.filename);
        if !file_matches_with_cancel(&path, file, is_cancelled)? {
            return Err(FwError::MissingArtifact(path));
        }
        match file.role.as_str() {
            "weights" => package_path = Some(path),
            "conversion_receipt" => receipt_path = Some(path),
            "license" => license_path = Some(path),
            "notice" => notice_path = Some(path),
            _ => return Err(manifest_error("manifest contains an unknown file role")),
        }
    }
    Ok(CachedSortformerPackage {
        package_path: package_path.ok_or_else(|| manifest_error("weights role is missing"))?,
        receipt_path: receipt_path.ok_or_else(|| manifest_error("receipt role is missing"))?,
        license_path: license_path.ok_or_else(|| manifest_error("license role is missing"))?,
        notice_path: notice_path.ok_or_else(|| manifest_error("notice role is missing"))?,
        artifact_version: manifest.artifact_version.clone(),
        package_sha256: crate::sortformer_conformance::SORTFORMER_PACKAGE_SHA256.to_owned(),
    })
}

#[must_use]
pub fn cached_sortformer_is_ready() -> bool {
    cached_sortformer_is_ready_with_cancel(|| false)
}

#[must_use]
pub fn cached_sortformer_is_ready_with_cancel<F>(is_cancelled: F) -> bool
where
    F: Fn() -> bool + Sync,
{
    cached_sortformer_readiness_with_cancel(is_cancelled).unwrap_or(false)
}

/// Return verified readiness while preserving cancellation as a distinct
/// outcome. Discovery callers that emit machine contracts must not translate
/// an interrupted 492 MB hash into a false "model missing" result.
pub fn cached_sortformer_readiness_with_cancel<F>(is_cancelled: F) -> FwResult<bool>
where
    F: Fn() -> bool + Sync,
{
    match resolve_cached_sortformer_with_cancel(is_cancelled) {
        Ok(_) => Ok(true),
        Err(FwError::Cancelled(message)) => Err(FwError::Cancelled(message)),
        Err(_) => Ok(false),
    }
}

pub fn pull_sortformer<F, P>(is_cancelled: F, mut progress: P) -> FwResult<PullOutcome>
where
    F: Fn() -> bool + Sync,
    P: FnMut(&str),
{
    cancellation_checkpoint(&is_cancelled)?;
    let manifest = builtin_manifest()?;
    let directory = sortformer_cache_dir()?;
    ensure_real_directory(&directory)?;
    match resolve_cached_in_with_cancel(&manifest, &directory, &is_cancelled) {
        Ok(package) => {
            progress("all Sortformer artifacts are already cached and verified");
            return Ok(PullOutcome {
                package,
                from_cache: true,
            });
        }
        Err(FwError::Cancelled(message)) => return Err(FwError::Cancelled(message)),
        Err(_) => {}
    }

    let runtime = RuntimeBuilder::new().build().map_err(|error| {
        FwError::BackendUnavailable(format!("cannot build model-download runtime: {error}"))
    })?;
    for remote in &manifest.files {
        cancellation_checkpoint(&is_cancelled)?;
        install_file(
            &runtime,
            remote,
            &directory.join(&remote.filename),
            &is_cancelled,
            &mut progress,
        )?;
    }
    cancellation_checkpoint(&is_cancelled)?;
    let package = resolve_cached_in_with_cancel(&manifest, &directory, &is_cancelled)?;
    Ok(PullOutcome {
        package,
        from_cache: false,
    })
}

pub fn pull_whisper<F, P>(is_cancelled: F, mut progress: P) -> FwResult<WhisperPullOutcome>
where
    F: Fn() -> bool + Sync,
    P: FnMut(&str),
{
    cancellation_checkpoint(&is_cancelled)?;
    let manifest = builtin_whisper_manifest()?;
    let directory = whisper_cache_dir()?;
    ensure_real_directory(&directory)?;
    match resolve_cached_whisper_with_cancel(&is_cancelled) {
        Ok(package) => {
            progress("native Whisper weights are already cached and verified");
            return Ok(WhisperPullOutcome {
                package,
                from_cache: true,
            });
        }
        Err(FwError::Cancelled(message)) => return Err(FwError::Cancelled(message)),
        Err(_) => {}
    }

    let runtime = RuntimeBuilder::new().build().map_err(|error| {
        FwError::BackendUnavailable(format!("cannot build model-download runtime: {error}"))
    })?;
    let remote = manifest
        .files
        .first()
        .ok_or_else(|| whisper_manifest_error("weights role is missing"))?;
    install_file(
        &runtime,
        remote,
        &directory.join(&remote.filename),
        &is_cancelled,
        &mut progress,
    )?;
    cancellation_checkpoint(&is_cancelled)?;
    let package = resolve_cached_whisper_with_cancel(&is_cancelled)?;
    Ok(WhisperPullOutcome {
        package,
        from_cache: false,
    })
}

fn cancellation_checkpoint<F>(is_cancelled: &F) -> FwResult<()>
where
    F: Fn() -> bool + Sync,
{
    if is_cancelled() {
        Err(FwError::Cancelled(
            "Sortformer model download interrupted".to_owned(),
        ))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
enum DownloadError {
    Request(ClientError),
    Body(HttpError),
    Status(u16),
    Io(std::io::Error),
    Cancelled,
    Timeout,
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(error) => write!(formatter, "request failed: {error}"),
            Self::Body(error) => write!(formatter, "response body failed: {error}"),
            Self::Status(status) => write!(formatter, "unexpected HTTP status {status}"),
            Self::Io(error) => write!(formatter, "I/O failed: {error}"),
            Self::Cancelled => formatter.write_str("download cancelled"),
            Self::Timeout => formatter.write_str("download deadline exceeded"),
        }
    }
}

fn streaming_client() -> Client {
    Client::builder()
        .max_body_size(MAX_HTTP_BODY_BYTES)
        .request_timeout(DOWNLOAD_REQUEST_TIMEOUT)
        .build()
}

async fn stream_url<F, S>(
    cx: &Cx,
    client: &Client,
    url: &str,
    is_cancelled: &F,
    mut sink: S,
) -> Result<u64, DownloadError>
where
    F: Fn() -> bool + Sync,
    S: FnMut(&[u8]) -> Result<(), std::io::Error>,
{
    if is_cancelled() {
        return Err(DownloadError::Cancelled);
    }
    let started = Instant::now();
    let mut request =
        Box::pin(client.request_streaming(cx, Method::Get, url, Vec::new(), Vec::new()));
    let mut request_cancellation_tick =
        asupersync::time::sleep(cx.now(), CANCELLATION_POLL_INTERVAL);
    let response = std::future::poll_fn(|task_cx| {
        if is_cancelled() || cx.checkpoint().is_err() {
            return Poll::Ready(Err(DownloadError::Cancelled));
        }
        if started.elapsed() >= DOWNLOAD_REQUEST_TIMEOUT {
            return Poll::Ready(Err(DownloadError::Timeout));
        }
        if let Poll::Ready(response) = request.as_mut().poll(task_cx) {
            return Poll::Ready(response.map_err(DownloadError::Request));
        }
        if Pin::new(&mut request_cancellation_tick)
            .poll(task_cx)
            .is_ready()
        {
            request_cancellation_tick.reset_after(cx.now(), CANCELLATION_POLL_INTERVAL);
            let _ = Pin::new(&mut request_cancellation_tick).poll(task_cx);
            if is_cancelled() || cx.checkpoint().is_err() {
                return Poll::Ready(Err(DownloadError::Cancelled));
            }
            if started.elapsed() >= DOWNLOAD_REQUEST_TIMEOUT {
                return Poll::Ready(Err(DownloadError::Timeout));
            }
        }
        Poll::Pending
    })
    .await?;
    if !(200..=299).contains(&response.head.status) {
        return Err(DownloadError::Status(response.head.status));
    }
    let mut total = 0_u64;
    let mut body = response.body;
    let mut cancellation_tick = asupersync::time::sleep(cx.now(), CANCELLATION_POLL_INTERVAL);
    while let Some(frame) = std::future::poll_fn(|task_cx| {
        if is_cancelled() || cx.checkpoint().is_err() {
            return Poll::Ready(Some(Err(DownloadError::Cancelled)));
        }
        if started.elapsed() >= DOWNLOAD_REQUEST_TIMEOUT {
            return Poll::Ready(Some(Err(DownloadError::Timeout)));
        }
        if let Poll::Ready(frame) = Pin::new(&mut body).poll_frame(task_cx) {
            return Poll::Ready(frame.map(|result| result.map_err(DownloadError::Body)));
        }
        if Pin::new(&mut cancellation_tick).poll(task_cx).is_ready() {
            cancellation_tick.reset_after(cx.now(), CANCELLATION_POLL_INTERVAL);
            let _ = Pin::new(&mut cancellation_tick).poll(task_cx);
            if is_cancelled() || cx.checkpoint().is_err() {
                return Poll::Ready(Some(Err(DownloadError::Cancelled)));
            }
            if started.elapsed() >= DOWNLOAD_REQUEST_TIMEOUT {
                return Poll::Ready(Some(Err(DownloadError::Timeout)));
            }
        }
        Poll::Pending
    })
    .await
    {
        if is_cancelled() {
            return Err(DownloadError::Cancelled);
        }
        let frame = frame?;
        if let Some(mut chunk) = frame.into_data() {
            while chunk.has_remaining() {
                let bytes = chunk.chunk();
                sink(bytes).map_err(DownloadError::Io)?;
                total = total.checked_add(bytes.len() as u64).ok_or_else(|| {
                    DownloadError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "download byte count overflowed u64",
                    ))
                })?;
                chunk.advance(bytes.len());
            }
        }
    }
    Ok(total)
}

fn install_file<F, P>(
    runtime: &asupersync::runtime::Runtime,
    remote: &RemoteFile,
    final_path: &Path,
    is_cancelled: &F,
    progress: &mut P,
) -> FwResult<()>
where
    F: Fn() -> bool + Sync,
    P: FnMut(&str),
{
    if file_matches_with_cancel(final_path, remote, is_cancelled)? {
        progress(&format!("verified cache hit: {}", remote.filename));
        return Ok(());
    }
    validate_install_target(final_path)?;
    let _lock = InstallLock::acquire(final_path, is_cancelled)?;
    if file_matches_with_cancel(final_path, remote, is_cancelled)? {
        progress(&format!("verified cache hit: {}", remote.filename));
        return Ok(());
    }
    validate_install_target(final_path)?;
    progress(&format!(
        "downloading {} ({:.1} MB)",
        remote.filename,
        remote.size as f64 / 1.0e6
    ));
    // Failed or cancelled staging files remain as operator-visible recovery
    // evidence. Only a fully verified descriptor is published under the
    // manifest filename, and publication never replaces an existing name.
    let (staging_path, staging_file) = create_staging_file(final_path)?;
    let download = runtime.block_on(download_remote_file(
        remote,
        &staging_path,
        staging_file,
        is_cancelled,
    ));
    download?;
    cancellation_checkpoint(is_cancelled)?;
    quarantine_invalid_existing(final_path)?;
    publish_staging_noreplace(&staging_path, final_path)?;
    sync_parent_dir(final_path)?;
    if !file_matches_with_cancel(final_path, remote, is_cancelled)? {
        return Err(manifest_error(
            "installed artifact failed its post-publication digest check",
        ));
    }
    progress(&format!("installed and verified: {}", remote.filename));
    Ok(())
}

async fn download_remote_file<F>(
    remote: &RemoteFile,
    staging_path: &Path,
    mut output: File,
    is_cancelled: &F,
) -> FwResult<()>
where
    F: Fn() -> bool + Sync,
{
    let client = streaming_client();
    let mut hash = Sha256::new();
    let mut received = 0_u64;
    let cx = Cx::current().ok_or_else(|| {
        FwError::BackendUnavailable("model-download runtime did not install a Cx".to_owned())
    })?;
    let result = stream_url(&cx, &client, &remote.url, is_cancelled, |chunk| {
        let next = received.checked_add(chunk.len() as u64).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "download byte count overflowed u64",
            )
        })?;
        if next > remote.size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "download exceeded manifest size",
            ));
        }
        output.write_all(chunk)?;
        hash.update(chunk);
        received = next;
        Ok(())
    })
    .await;
    match result {
        Err(DownloadError::Cancelled) => {
            return Err(FwError::Cancelled(
                "Sortformer model download interrupted".to_owned(),
            ));
        }
        Err(error) => {
            return Err(FwError::BackendUnavailable(format!(
                "failed to download {}: {error}",
                remote.filename
            )));
        }
        Ok(total) if total != remote.size => {
            return Err(manifest_error(format!(
                "downloaded {} bytes for {}, expected {}",
                total, remote.filename, remote.size
            )));
        }
        Ok(_) => {}
    }
    output.flush()?;
    output.sync_all()?;
    let digest: [u8; 32] = hash.finalize().into();
    if hex32(&digest) != remote.sha256 {
        return Err(manifest_error(format!(
            "downloaded {} failed SHA-256 verification",
            remote.filename
        )));
    }
    let path_metadata = std::fs::symlink_metadata(staging_path)?;
    let descriptor_metadata = output.metadata()?;
    if metadata_is_indirection(&path_metadata)
        || !path_metadata.is_file()
        || !descriptor_metadata.is_file()
        || metadata_is_indirection(&descriptor_metadata)
        || path_metadata.len() != remote.size
        || descriptor_metadata.len() != remote.size
    {
        return Err(manifest_error(
            "download staging file changed before publication",
        ));
    }
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::MetadataExt as _;

        if path_metadata.dev() != descriptor_metadata.dev()
            || path_metadata.ino() != descriptor_metadata.ino()
        {
            return Err(manifest_error(
                "download staging file identity changed before publication",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
fn file_matches(path: &Path, remote: &RemoteFile) -> bool {
    file_matches_with_cancel(path, remote, &|| false).unwrap_or(false)
}

fn file_matches_with_cancel<F>(path: &Path, remote: &RemoteFile, is_cancelled: &F) -> FwResult<bool>
where
    F: Fn() -> bool + Sync,
{
    cancellation_checkpoint(is_cancelled)?;
    let Ok(before) = std::fs::symlink_metadata(path) else {
        return Ok(false);
    };
    if metadata_is_indirection(&before) || !before.is_file() || before.len() != remote.size {
        return Ok(false);
    }
    let Some(file) = open_prechecked_cache_file(path, &before)? else {
        return Ok(false);
    };
    sha256_reader_with_cancel(file, is_cancelled).map(|digest| hex32(&digest) == remote.sha256)
}

fn open_prechecked_cache_file(path: &Path, before: &std::fs::Metadata) -> FwResult<Option<File>> {
    #[cfg(target_family = "unix")]
    let file = {
        use rustix::fs::{Mode, OFlags, open};

        match open(
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        ) {
            Ok(descriptor) => File::from(descriptor),
            Err(_) => return Ok(None),
        }
    };
    #[cfg(windows)]
    let file = {
        use std::os::windows::fs::OpenOptionsExt as _;

        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        match OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)
        {
            Ok(file) => file,
            Err(_) => return Ok(None),
        }
    };
    #[cfg(not(any(target_family = "unix", windows)))]
    let file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return Ok(None),
    };

    let after = file.metadata()?;
    if !after.is_file() || metadata_is_indirection(&after) || before.len() != after.len() {
        return Ok(None);
    }
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::MetadataExt as _;

        if before.dev() != after.dev() || before.ino() != after.ino() {
            return Ok(None);
        }
    }
    Ok(Some(file))
}

fn sha256_reader_with_cancel<F>(mut reader: impl Read, is_cancelled: &F) -> FwResult<[u8; 32]>
where
    F: Fn() -> bool + Sync,
{
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    loop {
        cancellation_checkpoint(is_cancelled)?;
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    cancellation_checkpoint(is_cancelled)?;
    Ok(hasher.finalize().into())
}

fn hex32(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(char::from_digit((byte >> 4) as u32, 16).unwrap_or('0'));
        output.push(char::from_digit((byte & 0x0f) as u32, 16).unwrap_or('0'));
    }
    output
}

struct InstallLock {
    _file: File,
}

impl InstallLock {
    fn acquire<F>(final_path: &Path, is_cancelled: &F) -> FwResult<Self>
    where
        F: Fn() -> bool + Sync,
    {
        let parent = final_path.parent().ok_or_else(|| {
            manifest_error("artifact install path does not have a parent directory")
        })?;
        let key = coordination_key(final_path)?;
        let path = parent.join(format!(".fw-pull-{key}.lock"));
        #[cfg(target_family = "unix")]
        let file = {
            use rustix::fs::{Mode, OFlags, open};

            let descriptor = open(
                &path,
                OFlags::RDWR
                    | OFlags::CREATE
                    | OFlags::CLOEXEC
                    | OFlags::NOFOLLOW
                    | OFlags::NONBLOCK,
                Mode::RUSR | Mode::WUSR,
            )
            .map_err(std::io::Error::from)?;
            File::from(descriptor)
        };
        #[cfg(windows)]
        let file = {
            use std::os::windows::fs::OpenOptionsExt as _;

            const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
                .open(&path)?
        };
        #[cfg(not(any(target_family = "unix", windows)))]
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let path_metadata = std::fs::symlink_metadata(&path)?;
        let descriptor_metadata = file.metadata()?;
        if metadata_is_indirection(&path_metadata)
            || !path_metadata.is_file()
            || !descriptor_metadata.is_file()
            || metadata_is_indirection(&descriptor_metadata)
        {
            return Err(manifest_error(
                "model install lock must be a regular non-symlink file",
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            if path_metadata.dev() != descriptor_metadata.dev()
                || path_metadata.ino() != descriptor_metadata.ino()
            {
                return Err(manifest_error(
                    "model install lock identity changed while opening it",
                ));
            }
        }
        let started = Instant::now();
        loop {
            cancellation_checkpoint(is_cancelled)?;
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock)
                    if started.elapsed() < INSTALL_LOCK_TIMEOUT =>
                {
                    std::thread::sleep(INSTALL_LOCK_POLL_INTERVAL);
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    return Err(FwError::StageTimeout {
                        stage: "sortformer_model_pull_lock".to_owned(),
                        budget_ms: INSTALL_LOCK_TIMEOUT.as_millis() as u64,
                    });
                }
                Err(std::fs::TryLockError::Error(error)) => return Err(FwError::Io(error)),
            }
        }
        Ok(Self { _file: file })
    }
}

fn coordination_key(final_path: &Path) -> FwResult<String> {
    let filename = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| manifest_error("artifact install filename must be UTF-8"))?;
    let digest: [u8; 32] = Sha256::digest(filename.as_bytes()).into();
    Ok(hex32(&digest)[..32].to_owned())
}

fn create_staging_file(final_path: &Path) -> FwResult<(PathBuf, File)> {
    let parent = final_path
        .parent()
        .ok_or_else(|| manifest_error("artifact install path does not have a parent directory"))?;
    let key = coordination_key(final_path)?;
    for _ in 0..64 {
        let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".fw-stage-{key}-{}-{nonce}.partial",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(manifest_error(
        "could not allocate a unique model download staging file",
    ))
}

fn publish_staging_noreplace(staging_path: &Path, final_path: &Path) -> FwResult<()> {
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use rustix::fs::{CWD, RenameFlags, renameat_with};

        renameat_with(CWD, staging_path, CWD, final_path, RenameFlags::NOREPLACE)
            .map_err(|error| std::io::Error::from(error).into())
    }
    #[cfg(windows)]
    {
        // `MoveFileExW` without REPLACE_EXISTING is no-clobbering. Rust's
        // Windows `rename` implementation preserves that behavior.
        std::fs::rename(staging_path, final_path).map_err(Into::into)
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    )))]
    {
        let _ = (staging_path, final_path);
        Err(manifest_error(
            "safe no-replace cache publication is unsupported on this platform",
        ))
    }
}

fn quarantine_invalid_existing(path: &Path) -> FwResult<()> {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return Ok(());
    };
    if metadata_is_indirection(&metadata) || !metadata.is_file() {
        return Err(manifest_error(
            "refusing to replace a non-regular or symlink model cache target",
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| manifest_error("model cache target does not have a parent"))?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| manifest_error("model cache target filename must be UTF-8"))?;
    for _ in 0..64 {
        let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let rejected = parent.join(format!(
            ".{filename}.rejected-{}-{nonce}",
            std::process::id()
        ));
        #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
        {
            use rustix::fs::{CWD, RenameFlags, renameat_with};

            match renameat_with(CWD, path, CWD, &rejected, RenameFlags::NOREPLACE) {
                Ok(()) => return Ok(()),
                Err(rustix::io::Errno::EXIST) => continue,
                Err(error) => return Err(std::io::Error::from(error).into()),
            }
        }
        #[cfg(windows)]
        {
            // `MoveFileExW` without REPLACE_EXISTING is no-clobbering. Rust's
            // Windows `rename` implementation preserves that behavior.
            match std::fs::rename(path, &rejected) {
                Ok(()) => return Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        #[cfg(not(any(
            target_os = "linux",
            target_os = "android",
            target_vendor = "apple",
            windows
        )))]
        {
            let _ = rejected;
            return Err(manifest_error(
                "safe no-replace cache quarantine is unsupported on this platform",
            ));
        }
    }
    Err(manifest_error(
        "could not allocate a quarantine name for an invalid cached artifact",
    ))
}

fn validate_install_target(path: &Path) -> FwResult<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata_is_indirection(&metadata) || !metadata.is_file() => Err(
            manifest_error("model cache target must be a regular non-symlink file"),
        ),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn ensure_real_directory(path: &Path) -> FwResult<()> {
    if !path.is_absolute() {
        return Err(manifest_error("model cache directory must be absolute"));
    }
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => {
                current.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(manifest_error(
                    "model cache directory must not contain parent traversal",
                ));
            }
            Component::Normal(name) => {
                current.push(name);
                let metadata = match std::fs::symlink_metadata(&current) {
                    Ok(metadata) => metadata,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        match std::fs::create_dir(&current) {
                            Ok(()) => {}
                            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                            Err(error) => return Err(error.into()),
                        }
                        std::fs::symlink_metadata(&current)?
                    }
                    Err(error) => return Err(error.into()),
                };
                if metadata_is_indirection(&metadata) || !metadata.is_dir() {
                    return Err(manifest_error(
                        "every model cache path component must be a real directory",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn metadata_is_indirection(metadata: &std::fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        return metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0;
    }
    #[cfg(not(windows))]
    false
}

#[cfg(unix)]
fn sync_parent_dir(path: &Path) -> FwResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| manifest_error("installed artifact does not have a parent directory"))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) -> FwResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_is_exact_and_release_bound() {
        let manifest = builtin_manifest().expect("embedded manifest");
        assert_eq!(manifest.files.len(), 4);
        assert_eq!(manifest.distribution_policy, SORTFORMER_DISTRIBUTION_POLICY);
        assert!(
            manifest
                .files
                .iter()
                .all(|file| file.url.starts_with(EXPECTED_RELEASE_PREFIX))
        );
        assert_eq!(
            manifest
                .files
                .iter()
                .find(|file| file.role == "weights")
                .map(|file| file.size),
            Some(491_570_584)
        );
    }

    #[test]
    fn committed_notice_matches_the_manifest() {
        let notice = include_bytes!("../NOTICE.sortformer.txt");
        let digest: [u8; 32] = Sha256::digest(notice).into();
        assert_eq!(notice.len(), 1_729);
        assert_eq!(
            hex32(&digest),
            "2efddd2d681136ddede12497e67caf4cf2444f15de824ee671ebbc93fe276d13"
        );
        assert!(String::from_utf8_lossy(notice).contains(SORTFORMER_REQUIRED_NOTICE));
    }

    #[test]
    fn manifest_rejects_path_traversal_and_mutable_urls() {
        assert!(validate_filename("../weights.safetensors").is_err());
        assert!(validate_filename("CON").is_err());
        assert!(validate_https_url("http://example.test/model").is_err());
        assert!(validate_https_url("https://example.test/model\nheader").is_err());
    }

    #[test]
    fn cache_match_requires_size_hash_regular_file() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("small.bin");
        std::fs::write(&path, b"model").expect("write fixture");
        let remote = RemoteFile {
            role: "fixture".to_owned(),
            filename: "small.bin".to_owned(),
            size: 5,
            sha256: format!("{:x}", Sha256::digest(b"model")),
            url: "https://example.test/small.bin".to_owned(),
        };
        assert!(file_matches(&path, &remote));
        std::fs::write(&path, b"other").expect("corrupt fixture");
        assert!(!file_matches(&path, &remote));
    }

    #[test]
    fn publication_never_opens_or_truncates_an_existing_target() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("weights.safetensors");
        std::fs::write(&path, b"operator-owned sentinel").expect("write sentinel");
        let (staging_path, mut staging_file) =
            create_staging_file(&path).expect("create isolated staging file");
        staging_file
            .write_all(b"new verified candidate")
            .expect("write staged candidate");
        staging_file.sync_all().expect("sync staged candidate");
        assert!(publish_staging_noreplace(&staging_path, &path).is_err());
        assert_eq!(
            std::fs::read(&path).expect("read sentinel"),
            b"operator-owned sentinel"
        );
        assert_eq!(
            std::fs::read(staging_path).expect("read retained staged candidate"),
            b"new verified candidate"
        );
    }

    #[test]
    fn invalid_cache_target_is_quarantined_without_data_loss() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("weights.safetensors");
        std::fs::write(&path, b"corrupt but recoverable").expect("write invalid cache entry");
        quarantine_invalid_existing(&path).expect("quarantine invalid cache entry");
        assert!(!path.exists());
        let quarantined = std::fs::read_dir(directory.path())
            .expect("read cache directory")
            .filter_map(Result::ok)
            .find(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".weights.safetensors.rejected-")
            })
            .expect("quarantined cache entry");
        assert_eq!(
            std::fs::read(quarantined.path()).expect("read quarantined bytes"),
            b"corrupt but recoverable"
        );
    }

    #[cfg(unix)]
    #[test]
    fn cache_match_rejects_symlinks_even_when_the_target_matches() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("tempdir");
        let target = directory.path().join("target.bin");
        let link = directory.path().join("small.bin");
        std::fs::write(&target, b"model").expect("write fixture");
        symlink(&target, &link).expect("symlink fixture");
        let remote = RemoteFile {
            role: "fixture".to_owned(),
            filename: "small.bin".to_owned(),
            size: 5,
            sha256: format!("{:x}", Sha256::digest(b"model")),
            url: "https://example.test/small.bin".to_owned(),
        };
        assert!(!file_matches(&link, &remote));
    }

    #[test]
    fn cancellation_fails_before_network_access() {
        let error = cancellation_checkpoint(&|| true).expect_err("cancelled pull");
        assert!(matches!(error, FwError::Cancelled(_)));
    }

    #[test]
    fn cancellation_is_not_collapsed_into_missing_cache_readiness() {
        let error = cached_sortformer_readiness_with_cancel(|| true)
            .expect_err("cancelled readiness hash must remain cancellation");
        assert!(matches!(error, FwError::Cancelled(_)));
    }

    #[test]
    fn hashing_large_artifacts_observes_cancellation_between_chunks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let polls = AtomicUsize::new(0);
        let bytes = vec![0x5a; HASH_BUFFER_BYTES * 4];
        let error = sha256_reader_with_cancel(std::io::Cursor::new(bytes), &|| {
            polls.fetch_add(1, Ordering::Relaxed) >= 2
        })
        .expect_err("hash cancellation");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(polls.load(Ordering::Relaxed) >= 3);
    }

    #[test]
    fn cache_directory_creation_rejects_relative_and_parent_traversal() {
        assert!(ensure_real_directory(Path::new("relative/cache")).is_err());
        assert!(ensure_real_directory(Path::new("/tmp/../cache")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn cache_directory_creation_rejects_an_intermediate_symlink() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temporary cache root");
        let real = root.path().join("real");
        let link = root.path().join("link");
        std::fs::create_dir(&real).expect("real directory");
        symlink(&real, &link).expect("intermediate symlink");

        let error = ensure_real_directory(&link.join("artifact"))
            .expect_err("an intermediate symlink must fail closed");
        assert!(error.to_string().contains("real directory"));
        assert!(!real.join("artifact").exists());
    }
}
