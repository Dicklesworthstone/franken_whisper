#![forbid(unsafe_code)]

use std::env;
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
use std::ffi::OsString;
use std::fs::File;
use std::io::Read as _;
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use franken_whisper::sortformer_conformance::SORTFORMER_PACKAGE_BYTES;
use franken_whisper::sortformer_f16_downcast::derive_sortformer_f16_artifact;
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
use uuid::Uuid;

const MAX_PARENT_RECEIPT_BYTES: u64 = 8 * 1024 * 1024;

struct Args {
    parent_receipt: PathBuf,
    parent_package: PathBuf,
    output_package: PathBuf,
    output_receipt: PathBuf,
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
struct OutputTarget {
    requested_parent: PathBuf,
    canonical_parent: PathBuf,
    directory: File,
    name: OsString,
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
struct OutputTarget;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (output_package, output_receipt) = validate_outputs(&args)?;
    let parent_receipt = read_bounded(
        &args.parent_receipt,
        MAX_PARENT_RECEIPT_BYTES,
        None,
        "parent receipt",
    )?;
    let parent_package = read_bounded(
        &args.parent_package,
        SORTFORMER_PACKAGE_BYTES,
        Some(SORTFORMER_PACKAGE_BYTES),
        "parent package",
    )?;
    let artifact = derive_sortformer_f16_artifact(parent_receipt, parent_package)
        .map_err(|error| error.to_string())?;
    let package_sha256 = artifact.receipt().package.sha256.clone();
    let package_bytes = artifact.receipt().package.bytes;
    let receipt_sha256 = artifact.receipt_sha256();
    let (derived_package, derived_receipt) = artifact.into_bytes();

    publish_artifact_pair(
        &output_package,
        &output_receipt,
        &derived_package,
        &derived_receipt,
    )?;

    println!("wrote derived Sortformer f16 artifact");
    println!("package bytes: {package_bytes}");
    println!("package sha256: {package_sha256}");
    println!("receipt sha256: {receipt_sha256}");
    Ok(())
}

fn publish_artifact_pair(
    output_package: &OutputTarget,
    output_receipt: &OutputTarget,
    package_bytes: &[u8],
    receipt_bytes: &[u8],
) -> Result<(), String> {
    publish_exact(output_package, package_bytes, "output package")?;
    publish_exact(output_receipt, receipt_bytes, "output receipt").map_err(|error| {
        format!(
            "{error}; artifact-pair completion is not confirmed: the output package was \
             durably committed, while receipt state is exactly as reported above; retry only \
             with identical bytes and paths"
        )
    })
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args_os();
    let _program = args.next();
    let values = args.collect::<Vec<_>>();
    if values.len() != 4 {
        return Err(
            "usage: sortformer-f16-downcast <parent-receipt> <parent-package> \
             <new-output-package> <new-output-receipt>"
                .to_owned(),
        );
    }
    Ok(Args {
        parent_receipt: PathBuf::from(values[0].clone()),
        parent_package: PathBuf::from(values[1].clone()),
        output_package: PathBuf::from(values[2].clone()),
        output_receipt: PathBuf::from(values[3].clone()),
    })
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn validate_outputs(args: &Args) -> Result<(OutputTarget, OutputTarget), String> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .map_err(|_| "repository boundary could not be resolved".to_owned())?;
    let package = bind_output_target(&args.output_package, &repository)?;
    let receipt = bind_output_target(&args.output_receipt, &repository)?;
    if same_output_target(&package, &receipt)? {
        return Err("output package and receipt paths must be distinct".to_owned());
    }
    Ok((package, receipt))
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn validate_outputs(args: &Args) -> Result<(OutputTarget, OutputTarget), String> {
    let _ = (&args.output_package, &args.output_receipt);
    Err("derived artifacts cannot be published safely on this platform".to_owned())
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn bind_output_target(path: &Path, repository: &Path) -> Result<OutputTarget, String> {
    use rustix::fs::{Mode, OFlags, open};

    let name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "output path must include a file name".to_owned())?;
    let name_text = name
        .to_str()
        .ok_or_else(|| "output file name must be lowercase ASCII".to_owned())?;
    if matches!(name_text, "." | "..")
        || name_text.is_empty()
        || name_text.bytes().any(|byte| {
            !byte.is_ascii_lowercase()
                && !byte.is_ascii_digit()
                && !matches!(byte, b'.' | b'_' | b'-')
        })
    {
        return Err(
            "output file name may contain only lowercase ASCII letters, digits, period, \
             underscore, and hyphen"
                .to_owned(),
        );
    }
    let requested_parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_owned();
    let directory = File::from(
        open(
            &requested_parent,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|_| "output parent directory could not be identity-bound".to_owned())?,
    );
    let canonical_parent = requested_parent
        .canonicalize()
        .map_err(|_| "output parent directory could not be resolved".to_owned())?;
    if canonical_parent.starts_with(repository) {
        return Err("derived model artifacts must be written outside the repository".to_owned());
    }
    let target = OutputTarget {
        requested_parent,
        canonical_parent,
        directory,
        name: name.to_owned(),
    };
    verify_output_parent_identity(&target)?;
    Ok(target)
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn same_output_target(left: &OutputTarget, right: &OutputTarget) -> Result<bool, String> {
    use std::os::unix::fs::MetadataExt as _;

    let left_parent = left
        .directory
        .metadata()
        .map_err(|_| "output package parent identity could not be read".to_owned())?;
    let right_parent = right
        .directory
        .metadata()
        .map_err(|_| "output receipt parent identity could not be read".to_owned())?;
    if left_parent.dev() == right_parent.dev()
        && left_parent.ino() == right_parent.ino()
        && left.name == right.name
    {
        return Ok(true);
    }
    match (
        existing_leaf_identity(left, "output package")?,
        existing_leaf_identity(right, "output receipt")?,
    ) {
        (Some(left), Some(right)) => Ok(left == right),
        _ => Ok(false),
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn existing_leaf_identity(
    target: &OutputTarget,
    label: &str,
) -> Result<Option<(u64, u64)>, String> {
    use rustix::fs::{AtFlags, statat};

    match statat(
        &target.directory,
        &target.name,
        AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(metadata) => {
            #[cfg(target_vendor = "apple")]
            let device = metadata.st_dev as u64;
            #[cfg(not(target_vendor = "apple"))]
            let device = metadata.st_dev;
            Ok(Some((device, metadata.st_ino)))
        }
        Err(error) if error == rustix::io::Errno::NOENT => Ok(None),
        Err(_) => Err(format!("{label} identity could not be inspected")),
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn verify_output_parent_identity(target: &OutputTarget) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt as _;

    let opened = target
        .directory
        .metadata()
        .map_err(|_| "validated output parent identity could not be read".to_owned())?;
    let canonical = target
        .canonical_parent
        .symlink_metadata()
        .map_err(|_| "output parent changed after validation".to_owned())?;
    let requested = target
        .requested_parent
        .symlink_metadata()
        .map_err(|_| "output parent changed after validation".to_owned())?;
    if !opened.is_dir()
        || !canonical.is_dir()
        || !requested.is_dir()
        || opened.dev() != canonical.dev()
        || opened.ino() != canonical.ino()
        || opened.dev() != requested.dev()
        || opened.ino() != requested.ino()
    {
        return Err("output parent changed after validation".to_owned());
    }
    if opened.uid() != rustix::process::geteuid().as_raw() || opened.mode() & 0o022 != 0 {
        return Err(
            "output parent must be owned by the effective user and not group/world writable"
                .to_owned(),
        );
    }
    Ok(())
}

fn read_bounded(
    path: &Path,
    max_bytes: u64,
    expected_bytes: Option<u64>,
    label: &str,
) -> Result<Vec<u8>, String> {
    let file = open_readonly_nonblocking(path, label)?;
    read_bounded_file(&file, max_bytes, expected_bytes, label)
}

fn read_bounded_file(
    file: &File,
    max_bytes: u64,
    expected_bytes: Option<u64>,
    label: &str,
) -> Result<Vec<u8>, String> {
    let metadata = file
        .metadata()
        .map_err(|_| format!("{label} could not be inspected"))?;
    if !metadata.is_file()
        || metadata.len() > max_bytes
        || expected_bytes.is_some_and(|expected| metadata.len() != expected)
    {
        return Err(format!("{label} size is outside the authenticated envelope"));
    }
    let read_limit = max_bytes
        .checked_add(1)
        .ok_or_else(|| format!("{label} size limit overflowed"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            usize::try_from(metadata.len())
                .map_err(|_| format!("{label} size does not fit this platform"))?,
        )
        .map_err(|_| format!("{label} allocation failed"))?;
    let mut reader = file.take(read_limit);
    reader
        .read_to_end(&mut bytes)
        .map_err(|_| format!("{label} could not be read"))?;
    let observed = u64::try_from(bytes.len())
        .map_err(|_| format!("{label} size does not fit u64"))?;
    if observed > max_bytes || expected_bytes.is_some_and(|expected| observed != expected) {
        return Err(format!("{label} size changed while it was read"));
    }
    Ok(bytes)
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn open_readonly_nonblocking(path: &Path, label: &str) -> Result<File, String> {
    use rustix::fs::{Mode, OFlags, open};

    let descriptor = open(
        path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|_| format!("{label} could not be opened as a regular file"))?;
    Ok(File::from(descriptor))
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn open_readonly_nonblocking(_path: &Path, label: &str) -> Result<File, String> {
    Err(format!(
        "{label} cannot be opened safely on this platform"
    ))
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn publish_exact(target: &OutputTarget, bytes: &[u8], label: &str) -> Result<(), String> {
    publish_exact_with_parent_sync(target, bytes, label, &File::sync_all)
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn publish_exact_with_parent_sync(
    target: &OutputTarget,
    bytes: &[u8],
    label: &str,
    sync_parent: &dyn Fn(&File) -> std::io::Result<()>,
) -> Result<(), String> {
    publish_exact_with_hooks(
        target,
        bytes,
        label,
        &write_all_staging,
        &File::sync_all,
        &File::sync_all,
        sync_parent,
        &|| Ok(()),
    )
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn write_all_staging(file: &mut File, bytes: &[u8]) -> std::io::Result<()> {
    file.write_all(bytes)
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn publish_exact_with_hooks(
    target: &OutputTarget,
    bytes: &[u8],
    label: &str,
    write_staging: &dyn Fn(&mut File, &[u8]) -> std::io::Result<()>,
    sync_staging: &dyn Fn(&File) -> std::io::Result<()>,
    sync_existing: &dyn Fn(&File) -> std::io::Result<()>,
    sync_parent: &dyn Fn(&File) -> std::io::Result<()>,
    before_rename: &dyn Fn() -> Result<(), String>,
) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt as _;

    use rustix::fs::{Mode, OFlags, RenameFlags, fchmod, openat, renameat_with};

    verify_output_parent_identity(target)?;
    let expected = u64::try_from(bytes.len())
        .map_err(|_| format!("{label} size does not fit u64"))?;
    if let Some(existing) = open_output_leaf(target, label, true)? {
        let existing_bytes = read_bounded_file(&existing, expected, Some(expected), label)?;
        verify_output_leaf_identity(target, &existing, label)?;
        if existing_bytes != bytes {
            return Err(format!(
                "refusing to overwrite an existing {label} with different bytes"
            ));
        }
        sync_existing(&existing)
            .map_err(|_| format!("existing {label} bytes could not be synchronized"))?;
        sync_parent(&target.directory)
            .map_err(|_| format!("{label} directory entry could not be synchronized"))?;
        verify_output_parent_identity(target)?;
        verify_output_leaf_identity(target, &existing, label)?;
        return Ok(());
    }

    let mut staged = None;
    for _ in 0..8 {
        let name = OsString::from(format!(
            ".sortformer-f16-stage-{}",
            Uuid::new_v4().simple()
        ));
        match openat(
            &target.directory,
            &name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::RUSR | Mode::WUSR,
        ) {
            Ok(descriptor) => {
                staged = Some((name, File::from(descriptor)));
                break;
            }
            Err(error) if error == rustix::io::Errno::EXIST => {}
            Err(_) => return Err(format!("{label} staging file could not be created")),
        }
    }
    let (staging_name, mut staging_file) = staged
        .ok_or_else(|| format!("{label} staging file could not be created after bounded retries"))?;
    if fchmod(&staging_file, Mode::RUSR | Mode::WUSR).is_err() {
        return Err(format!(
            "{label} staging permissions could not be restricted{}",
            staging_diagnostic_suffix(
                target,
                &staging_name,
                &staging_file,
                label,
                StagingContentState::CreatedEmpty,
            )
        ));
    }
    let staging_metadata = staging_file
        .metadata()
        .map_err(|_| {
            format!(
                "{label} staging identity could not be inspected{}",
                staging_diagnostic_suffix(
                    target,
                    &staging_name,
                    &staging_file,
                    label,
                    StagingContentState::CreatedEmpty,
                )
            )
        })?;
    if !staging_metadata.is_file()
        || staging_metadata.uid() != rustix::process::geteuid().as_raw()
        || staging_metadata.mode() & 0o7777 != 0o600
    {
        return Err(format!(
            "{label} staging file must remain a mode-0600 regular file owned by the \
             effective user{}",
            staging_diagnostic_suffix(
                target,
                &staging_name,
                &staging_file,
                label,
                StagingContentState::CreatedEmpty,
            )
        ));
    }
    if write_staging(&mut staging_file, bytes).is_err() {
        return Err(format!(
            "{label} staging bytes could not be written{}",
            staging_diagnostic_suffix(
                target,
                &staging_name,
                &staging_file,
                label,
                staging_write_failure_state(&staging_file, expected),
            )
        ));
    }
    if sync_staging(&staging_file).is_err() {
        return Err(format!(
            "{label} staging bytes could not be synchronized{}",
            staging_diagnostic_suffix(
                target,
                &staging_name,
                &staging_file,
                label,
                StagingContentState::FullButUnsynced { expected },
            )
        ));
    }
    let full_synced = || {
        staging_diagnostic_suffix(
            target,
            &staging_name,
            &staging_file,
            label,
            StagingContentState::FullSynced { expected },
        )
    };
    verify_output_parent_identity(target)
        .map_err(|error| format!("{error}{}", full_synced()))?;
    verify_named_leaf_identity(target, &staging_name, &staging_file, label)
        .map_err(|error| format!("{error}{}", full_synced()))?;
    before_rename().map_err(|error| format!("{error}{}", full_synced()))?;
    if let Err(rename_error) = renameat_with(
        &target.directory,
        &staging_name,
        &target.directory,
        &target.name,
        RenameFlags::NOREPLACE,
    ) {
        let final_state = output_leaf_state(target, &target.name, &staging_file, label).ok();
        let staging_state = output_leaf_state(target, &staging_name, &staging_file, label).ok();
        match classify_failed_rename(
            rename_error == rustix::io::Errno::EXIST,
            final_state,
            staging_state,
        ) {
            FailedRenameDisposition::PublishedUncertain => {
                return Err(committed_uncertain(
                    label,
                    "the final name acquired the staged inode despite a rename error",
                ));
            }
            FailedRenameDisposition::CollisionWithNamedStage => {
                return Err(format!(
                    "{label} could not be published without overwriting{}",
                    full_synced()
                ));
            }
            FailedRenameDisposition::FailedWithNamedStage => {
                return Err(format!(
                    "{label} no-clobber rename failed with {rename_error}{}",
                    full_synced()
                ));
            }
            FailedRenameDisposition::Ambiguous => {}
        }
        return Err(format!(
            "{label} publication state is uncertain after the no-clobber rename failed; \
             no final-commit or retained-staging-path claim is made; inspect output {} and \
             allocated staging name {}",
            target.name.to_string_lossy(),
            staging_name.to_string_lossy()
        ));
    }
    verify_output_parent_identity(target)
        .map_err(|_| committed_uncertain(label, "its parent identity could not be confirmed"))?;
    verify_output_leaf_identity(target, &staging_file, label)
        .map_err(|_| committed_uncertain(label, "its final identity could not be confirmed"))?;
    let published = open_output_leaf(target, label, false)
        .map_err(|_| committed_uncertain(label, "its final file could not be reopened"))?
        .ok_or_else(|| committed_uncertain(label, "its final file disappeared"))?;
    let published_bytes = read_bounded_file(&published, expected, Some(expected), label)
        .map_err(|_| committed_uncertain(label, "its final bytes could not be reread"))?;
    verify_output_leaf_identity(target, &published, label)
        .map_err(|_| committed_uncertain(label, "its reread identity could not be confirmed"))?;
    if published_bytes != bytes {
        return Err(committed_uncertain(
            label,
            "its final bytes changed during confirmation",
        ));
    }
    sync_parent(&target.directory)
        .map_err(|_| committed_uncertain(label, "its directory entry was not synchronized"))?;
    verify_output_parent_identity(target).map_err(|_| {
        committed_uncertain(label, "its parent identity changed after synchronization")
    })?;
    verify_output_leaf_identity(target, &published, label).map_err(|_| {
        committed_uncertain(label, "its final identity changed after synchronization")
    })?;
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn publish_exact(_target: &OutputTarget, _bytes: &[u8], label: &str) -> Result<(), String> {
    Err(format!("{label} cannot be published safely on this platform"))
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn open_output_leaf(
    target: &OutputTarget,
    label: &str,
    writable: bool,
) -> Result<Option<File>, String> {
    use std::os::unix::fs::MetadataExt as _;

    use rustix::fs::{Mode, OFlags, openat};

    verify_output_parent_identity(target)?;
    let access = if writable {
        OFlags::RDWR
    } else {
        OFlags::RDONLY
    };
    match openat(
        &target.directory,
        &target.name,
        access | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(descriptor) => {
            let file = File::from(descriptor);
            let metadata = file
                .metadata()
                .map_err(|_| format!("{label} could not be inspected"))?;
            if !metadata.is_file()
                || metadata.uid() != rustix::process::geteuid().as_raw()
                || metadata.mode() & 0o022 != 0
            {
                return Err(format!(
                    "{label} must be a regular file owned by the effective user and not \
                     group/world writable"
                ));
            }
            Ok(Some(file))
        }
        Err(error) if error == rustix::io::Errno::NOENT => Ok(None),
        Err(_) => Err(format!("{label} path could not be inspected safely")),
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn verify_output_leaf_identity(
    target: &OutputTarget,
    expected: &File,
    label: &str,
) -> Result<(), String> {
    verify_named_leaf_identity(target, &target.name, expected, label)
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputLeafState {
    Missing,
    Expected,
    Other,
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FailedRenameDisposition {
    PublishedUncertain,
    CollisionWithNamedStage,
    FailedWithNamedStage,
    Ambiguous,
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StagingContentState {
    CreatedEmpty,
    PartialWrite { observed: u64, expected: u64 },
    FullButUnsynced { expected: u64 },
    FullSynced { expected: u64 },
    Unknown { expected: u64 },
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn classify_failed_rename(
    destination_existed: bool,
    final_state: Option<OutputLeafState>,
    staging_state: Option<OutputLeafState>,
) -> FailedRenameDisposition {
    if final_state == Some(OutputLeafState::Expected) {
        return FailedRenameDisposition::PublishedUncertain;
    }
    if matches!(final_state, Some(OutputLeafState::Missing | OutputLeafState::Other))
        && staging_state == Some(OutputLeafState::Expected)
    {
        return if destination_existed {
            FailedRenameDisposition::CollisionWithNamedStage
        } else {
            FailedRenameDisposition::FailedWithNamedStage
        };
    }
    FailedRenameDisposition::Ambiguous
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn output_leaf_state(
    target: &OutputTarget,
    name: &std::ffi::OsStr,
    expected: &File,
    label: &str,
) -> Result<OutputLeafState, String> {
    use std::os::unix::fs::MetadataExt as _;

    use rustix::fs::{AtFlags, FileType, statat};

    let expected = expected
        .metadata()
        .map_err(|_| format!("{label} identity could not be read"))?;
    let current = match statat(&target.directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(current) => current,
        Err(error) if error == rustix::io::Errno::NOENT => {
            return Ok(OutputLeafState::Missing);
        }
        Err(_) => return Err(format!("{label} identity could not be verified")),
    };
    #[cfg(target_vendor = "apple")]
    let device_matches = expected.dev() == current.st_dev as u64;
    #[cfg(not(target_vendor = "apple"))]
    let device_matches = expected.dev() == current.st_dev;
    if expected.is_file()
        && FileType::from_raw_mode(current.st_mode) == FileType::RegularFile
        && device_matches
        && expected.ino() == current.st_ino
    {
        Ok(OutputLeafState::Expected)
    } else {
        Ok(OutputLeafState::Other)
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn verify_named_leaf_identity(
    target: &OutputTarget,
    name: &std::ffi::OsStr,
    expected: &File,
    label: &str,
) -> Result<(), String> {
    if output_leaf_state(target, name, expected, label)? != OutputLeafState::Expected {
        return Err(format!("{label} identity changed during publication"));
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn staging_write_failure_state(file: &File, expected: u64) -> StagingContentState {
    match file.metadata().map(|metadata| metadata.len()) {
        Ok(0) => StagingContentState::CreatedEmpty,
        Ok(observed) if observed < expected => {
            StagingContentState::PartialWrite { observed, expected }
        }
        _ => StagingContentState::Unknown { expected },
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn staging_diagnostic_suffix(
    target: &OutputTarget,
    name: &std::ffi::OsStr,
    file: &File,
    label: &str,
    content: StagingContentState,
) -> String {
    let content = match content {
        StagingContentState::CreatedEmpty => {
            "zero payload bytes were intentionally written".to_owned()
        }
        StagingContentState::PartialWrite { observed, expected } => format!(
            "only {observed} of {expected} payload bytes were observed; durability is unconfirmed"
        ),
        StagingContentState::FullButUnsynced { expected } => format!(
            "all {expected} payload bytes were written, but synchronization was not confirmed"
        ),
        StagingContentState::FullSynced { expected } => {
            format!("all {expected} payload bytes were synchronized before publication stopped")
        }
        StagingContentState::Unknown { expected } => format!(
            "the payload state is unknown after a failed write of {expected} expected bytes"
        ),
    };
    let path = match output_leaf_state(target, name, file, label) {
        Ok(OutputLeafState::Expected) => format!(
            "staging name {} resolved to the held inode at diagnosis time",
            name.to_string_lossy()
        ),
        Ok(OutputLeafState::Missing) => format!(
            "staging name {} was missing; no retained-path claim is made",
            name.to_string_lossy()
        ),
        Ok(OutputLeafState::Other) => format!(
            "staging name {} resolved to another inode; no retained-path claim is made",
            name.to_string_lossy()
        ),
        Err(_) => format!(
            "staging name {} could not be verified; no retained-path claim is made",
            name.to_string_lossy()
        ),
    };
    format!("; {content}; {path}")
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn committed_uncertain(label: &str, detail: &str) -> String {
    format!(
        "{label} was published but {detail}; durability/identity confirmation is uncertain; \
         retry only with identical bytes"
    )
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
mod tests {
    use super::*;

    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt as _;

        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
            .expect("set fixture permissions");
    }

    fn test_output_target(path: &Path) -> OutputTarget {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .canonicalize()
            .expect("canonical repository");
        bind_output_target(path, &repository).expect("bind test output target")
    }

    fn single_staging_path(directory: &Path) -> PathBuf {
        let stages = std::fs::read_dir(directory)
            .expect("read output directory")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".sortformer-f16-stage-")
            })
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        assert_eq!(stages.len(), 1);
        stages.into_iter().next().expect("one staging path")
    }

    #[test]
    fn failed_rename_classification_is_fail_closed() {
        assert_eq!(
            classify_failed_rename(true, Some(OutputLeafState::Expected), None),
            FailedRenameDisposition::PublishedUncertain
        );
        assert_eq!(
            classify_failed_rename(
                true,
                Some(OutputLeafState::Other),
                Some(OutputLeafState::Expected),
            ),
            FailedRenameDisposition::CollisionWithNamedStage
        );
        assert_eq!(
            classify_failed_rename(
                false,
                Some(OutputLeafState::Missing),
                Some(OutputLeafState::Expected),
            ),
            FailedRenameDisposition::FailedWithNamedStage
        );
        for (final_state, staging_state) in [
            (None, Some(OutputLeafState::Expected)),
            (Some(OutputLeafState::Other), None),
            (Some(OutputLeafState::Missing), Some(OutputLeafState::Other)),
        ] {
            assert_eq!(
                classify_failed_rename(true, final_state, staging_state),
                FailedRenameDisposition::Ambiguous
            );
        }
    }

    #[test]
    fn publication_is_idempotent_but_never_overwrites_different_bytes() {
        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        let target = test_output_target(&output);

        publish_exact(&target, b"first", "test artifact").expect("initial publication");
        publish_exact(&target, b"first", "test artifact").expect("idempotent recovery");
        let error = publish_exact(&target, b"second", "test artifact")
            .expect_err("different bytes must not overwrite");

        assert!(error.contains("refusing to overwrite"));
        assert_eq!(
            read_bounded(&output, 5, Some(5), "test artifact").expect("read published bytes"),
            b"first"
        );
    }

    #[test]
    fn retry_resynchronizes_an_identical_committed_output() {
        use std::io;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        let target = test_output_target(&output);
        let sync_calls = AtomicUsize::new(0);
        let sync_parent = |directory: &File| {
            if sync_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(io::Error::other("synthetic post-rename sync failure"))
            } else {
                directory.sync_all()
            }
        };

        let error = publish_exact_with_parent_sync(
            &target,
            b"durable after retry",
            "test artifact",
            &sync_parent,
        )
        .expect_err("first directory sync must fail after publication");
        assert!(error.contains("was published but its directory entry was not synchronized"));
        assert_eq!(
            read_bounded(&output, 19, Some(19), "test artifact")
                .expect("committed bytes remain visible"),
            b"durable after retry"
        );

        publish_exact_with_parent_sync(
            &target,
            b"durable after retry",
            "test artifact",
            &sync_parent,
        )
        .expect("retry must synchronize the existing identical output");
        assert_eq!(sync_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn arbitrary_identical_existing_file_must_sync_before_success() {
        use std::io;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        std::fs::write(&output, b"identical").expect("write existing output");
        set_mode(&output, 0o600);
        let target = test_output_target(&output);
        let file_sync_calls = AtomicUsize::new(0);
        let parent_sync_calls = AtomicUsize::new(0);

        let error = publish_exact_with_hooks(
            &target,
            b"identical",
            "test artifact",
            &write_all_staging,
            &File::sync_all,
            &|_| {
                file_sync_calls.fetch_add(1, Ordering::SeqCst);
                Err(io::Error::other("synthetic existing-file sync failure"))
            },
            &|_| {
                parent_sync_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            &|| Ok(()),
        )
        .expect_err("existing identical bytes must not bypass file synchronization");

        assert!(error.contains("existing test artifact bytes could not be synchronized"));
        assert_eq!(file_sync_calls.load(Ordering::SeqCst), 1);
        assert_eq!(parent_sync_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn failed_staging_write_reports_an_empty_created_file_exactly() {
        use std::io;

        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        let target = test_output_target(&output);
        let error = publish_exact_with_hooks(
            &target,
            b"payload",
            "test artifact",
            &|_, _| Err(io::Error::other("synthetic write failure")),
            &File::sync_all,
            &File::sync_all,
            &File::sync_all,
            &|| Ok(()),
        )
        .expect_err("failed write must report the created staging state");

        assert!(error.contains("zero payload bytes were intentionally written"));
        assert!(error.contains("resolved to the held inode at diagnosis time"));
        assert_eq!(
            std::fs::metadata(single_staging_path(directory.path()))
                .expect("inspect empty staging file")
                .len(),
            0
        );
    }

    #[test]
    fn failed_staging_write_reports_partial_bytes_without_durability_claims() {
        use std::io;
        use std::io::Write as _;

        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        let target = test_output_target(&output);
        let error = publish_exact_with_hooks(
            &target,
            b"payload",
            "test artifact",
            &|file, bytes| {
                file.write_all(&bytes[..3])?;
                Err(io::Error::other("synthetic partial write failure"))
            },
            &File::sync_all,
            &File::sync_all,
            &File::sync_all,
            &|| Ok(()),
        )
        .expect_err("partial write must not overclaim retained bytes");

        assert!(error.contains("only 3 of 7 payload bytes were observed"));
        assert!(error.contains("durability is unconfirmed"));
        assert_eq!(
            std::fs::read(single_staging_path(directory.path()))
                .expect("read partial staging file"),
            b"pay"
        );
    }

    #[test]
    fn failed_staging_sync_reports_full_but_unsynchronized_bytes() {
        use std::io;

        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        let target = test_output_target(&output);
        let error = publish_exact_with_hooks(
            &target,
            b"payload",
            "test artifact",
            &write_all_staging,
            &|_| Err(io::Error::other("synthetic staging sync failure")),
            &File::sync_all,
            &File::sync_all,
            &|| Ok(()),
        )
        .expect_err("failed sync must distinguish written from durable bytes");

        assert!(error.contains("all 7 payload bytes were written"));
        assert!(error.contains("synchronization was not confirmed"));
        assert_eq!(
            std::fs::read(single_staging_path(directory.path()))
                .expect("read unsynchronized staging file"),
            b"payload"
        );
    }

    #[test]
    fn pre_rename_failure_reports_the_complete_synchronized_stage() {
        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        let target = test_output_target(&output);
        let error = publish_exact_with_hooks(
            &target,
            b"payload",
            "test artifact",
            &write_all_staging,
            &File::sync_all,
            &File::sync_all,
            &File::sync_all,
            &|| Err("synthetic pre-rename stop".to_owned()),
        )
        .expect_err("pre-rename failure must name the synchronized stage");

        assert!(error.contains("all 7 payload bytes were synchronized"));
        assert!(error.contains("resolved to the held inode at diagnosis time"));
        assert_eq!(
            std::fs::read(single_staging_path(directory.path()))
                .expect("read synchronized staging file"),
            b"payload"
        );
    }

    #[test]
    fn group_writable_existing_output_is_rejected_before_comparison() {
        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        std::fs::write(&output, b"identical").expect("write existing output");
        set_mode(&output, 0o620);
        let target = test_output_target(&output);

        let error = publish_exact(&target, b"identical", "test artifact")
            .expect_err("group-writable output must not be trusted");
        assert!(error.contains("not group/world writable"));
    }

    #[test]
    fn parent_replacement_is_rejected_without_redirecting_publication() {
        let root = tempfile::tempdir().expect("create test root");
        let requested_parent = root.path().join("output");
        let moved_parent = root.path().join("moved-output");
        std::fs::create_dir(&requested_parent).expect("create original parent");
        set_mode(&requested_parent, 0o700);
        let output = requested_parent.join("artifact.bin");
        let target = test_output_target(&output);

        std::fs::rename(&requested_parent, &moved_parent).expect("move original parent");
        std::fs::create_dir(&requested_parent).expect("create replacement parent");
        set_mode(&requested_parent, 0o700);
        let error = publish_exact(&target, b"never redirected", "test artifact")
            .expect_err("replaced parent must be rejected");

        assert!(error.contains("output parent changed after validation"));
        assert!(!requested_parent.join("artifact.bin").exists());
        assert!(!moved_parent.join("artifact.bin").exists());
    }

    #[test]
    fn last_moment_collision_never_clobbers_and_retains_complete_staging_bytes() {
        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");
        let target = test_output_target(&output);
        let create_collision = || {
            std::fs::write(&output, b"competitor")
                .map_err(|error| format!("create synthetic collision: {error}"))
        };

        let error = publish_exact_with_hooks(
            &target,
            b"derived artifact",
            "test artifact",
            &write_all_staging,
            &File::sync_all,
            &File::sync_all,
            &File::sync_all,
            &create_collision,
        )
        .expect_err("no-clobber rename must reject a last-moment collision");

        assert!(error.contains("could not be published without overwriting"));
        assert_eq!(std::fs::read(&output).expect("read collision"), b"competitor");
        let stages = std::fs::read_dir(directory.path())
            .expect("read output directory")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".sortformer-f16-stage-")
            })
            .collect::<Vec<_>>();
        assert_eq!(stages.len(), 1);
        assert_eq!(
            std::fs::read(stages[0].path()).expect("read retained staging bytes"),
            b"derived artifact"
        );
    }

    #[test]
    fn receipt_failure_reports_the_durably_committed_partial_pair() {
        let directory = tempfile::tempdir().expect("create output directory");
        let package_path = directory.path().join("package.bin");
        let receipt_path = directory.path().join("receipt.json");
        std::fs::write(&receipt_path, b"incumbent").expect("write incumbent receipt");
        set_mode(&receipt_path, 0o600);
        let package = test_output_target(&package_path);
        let receipt = test_output_target(&receipt_path);

        let error = publish_artifact_pair(&package, &receipt, b"package", b"receipt")
            .expect_err("receipt collision must leave an explicit partial pair");

        assert!(error.contains("artifact-pair completion is not confirmed"));
        assert!(error.contains("output package was durably committed"));
        assert_eq!(
            std::fs::read(&package_path).expect("read committed package"),
            b"package"
        );
        assert_eq!(
            std::fs::read(&receipt_path).expect("read incumbent receipt"),
            b"incumbent"
        );
    }

    #[test]
    fn output_names_are_lowercase_ascii_and_existing_hardlinks_are_not_distinct() {
        let directory = tempfile::tempdir().expect("create output directory");
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .canonicalize()
            .expect("canonical repository");
        let invalid = bind_output_target(&directory.path().join("Artifact.bin"), &repository)
            .err()
            .expect("uppercase output name must be rejected");
        assert!(invalid.contains("lowercase ASCII"));

        let package = directory.path().join("package.bin");
        let receipt = directory.path().join("receipt.json");
        std::fs::write(&package, b"same inode").expect("write package output");
        std::fs::hard_link(&package, &receipt).expect("create hard-linked receipt output");
        let package = test_output_target(&package);
        let receipt = test_output_target(&receipt);
        assert!(
            same_output_target(&package, &receipt).expect("compare hard-linked output identities")
        );
    }

    #[test]
    fn output_leaf_symlink_is_rejected_without_touching_its_target() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("create output directory");
        let target_file = directory.path().join("target.bin");
        let output = directory.path().join("artifact.bin");
        std::fs::write(&target_file, b"target bytes").expect("write symlink target");
        symlink(&target_file, &output).expect("create output symlink");
        let target = test_output_target(&output);

        let error = publish_exact(&target, b"replacement", "test artifact")
            .expect_err("output symlink must be rejected");
        assert!(error.contains("path could not be inspected safely"));
        assert_eq!(
            std::fs::read(&target_file).expect("read symlink target"),
            b"target bytes"
        );
    }

    #[test]
    fn bounded_reader_rejects_symlink_inputs() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("create input directory");
        let target = directory.path().join("target.bin");
        let link = directory.path().join("link.bin");
        std::fs::write(&target, b"bytes").expect("write target");
        symlink(&target, &link).expect("create symlink");

        let error = read_bounded(&link, 5, Some(5), "test input")
            .expect_err("symlink input must be rejected");
        assert!(error.contains("could not be opened as a regular file"));
    }
}
