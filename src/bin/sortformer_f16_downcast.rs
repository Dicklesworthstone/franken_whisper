#![forbid(unsafe_code)]

use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use franken_whisper::sortformer_conformance::SORTFORMER_PACKAGE_BYTES;
use franken_whisper::sortformer_f16_downcast::derive_sortformer_f16_artifact;
use uuid::Uuid;

const MAX_PARENT_RECEIPT_BYTES: u64 = 8 * 1024 * 1024;

struct Args {
    parent_receipt: PathBuf,
    parent_package: PathBuf,
    output_package: PathBuf,
    output_receipt: PathBuf,
}

struct OutputTarget {
    requested_parent: PathBuf,
    canonical_parent: PathBuf,
    directory: File,
    name: OsString,
}

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

    publish_exact(&output_package, &derived_package, "output package")?;
    publish_exact(&output_receipt, &derived_receipt, "output receipt")?;

    println!("wrote derived Sortformer f16 artifact");
    println!("package bytes: {package_bytes}");
    println!("package sha256: {package_sha256}");
    println!("receipt sha256: {receipt_sha256}");
    Ok(())
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
fn validate_outputs(_args: &Args) -> Result<(OutputTarget, OutputTarget), String> {
    Err("derived artifacts cannot be published safely on this platform".to_owned())
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn bind_output_target(path: &Path, repository: &Path) -> Result<OutputTarget, String> {
    use rustix::fs::{Mode, OFlags, open};

    let name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "output path must include a file name".to_owned())?;
    if matches!(name.to_str(), Some("." | "..")) {
        return Err("output file name must not traverse directories".to_owned());
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
    use std::os::unix::ffi::OsStrExt as _;
    use std::os::unix::fs::MetadataExt as _;

    let left_parent = left
        .directory
        .metadata()
        .map_err(|_| "output package parent identity could not be read".to_owned())?;
    let right_parent = right
        .directory
        .metadata()
        .map_err(|_| "output receipt parent identity could not be read".to_owned())?;
    Ok(left_parent.dev() == right_parent.dev()
        && left_parent.ino() == right_parent.ino()
        && left
            .name
            .as_bytes()
            .eq_ignore_ascii_case(right.name.as_bytes()))
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
    let mut reader = file;
    reader
        .take(read_limit)
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
    publish_exact_with_parent_sync(target, bytes, label, &|directory| directory.sync_all())
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn publish_exact_with_parent_sync(
    target: &OutputTarget,
    bytes: &[u8],
    label: &str,
    sync_parent: &dyn Fn(&File) -> std::io::Result<()>,
) -> Result<(), String> {
    use rustix::fs::{Mode, OFlags, RenameFlags, openat, renameat_with};

    verify_output_parent_identity(target)?;
    let expected = u64::try_from(bytes.len())
        .map_err(|_| format!("{label} size does not fit u64"))?;
    if let Some(existing) = open_output_leaf(target, label)? {
        existing
            .sync_all()
            .map_err(|_| format!("existing {label} could not be synchronized"))?;
        let existing_bytes = read_bounded_file(&existing, expected, Some(expected), label)?;
        verify_output_leaf_identity(target, &existing, label)?;
        if existing_bytes != bytes {
            return Err(format!(
                "refusing to overwrite an existing {label} with different bytes"
            ));
        }
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
    if staging_file
        .write_all(bytes)
        .and_then(|()| staging_file.sync_all())
        .is_err()
    {
        return Err(format!(
            "{label} staging bytes could not be persisted{}",
            retained_staging_suffix(&staging_name)
        ));
    }
    verify_output_parent_identity(target)?;
    verify_named_leaf_identity(target, &staging_name, &staging_file, label)?;
    if let Err(error) = renameat_with(
        &target.directory,
        &staging_name,
        &target.directory,
        &target.name,
        RenameFlags::NOREPLACE,
    ) {
        let detail = if error == rustix::io::Errno::EXIST {
            "could not be published without overwriting"
        } else {
            "publication state is uncertain after the no-clobber rename failed"
        };
        return Err(format!(
            "{label} {detail}{}",
            retained_staging_suffix(&staging_name)
        ));
    }
    verify_output_parent_identity(target)
        .map_err(|_| format!("{label} was published but its parent identity changed"))?;
    verify_output_leaf_identity(target, &staging_file, label)?;
    let published = open_output_leaf(target, label)?
        .ok_or_else(|| format!("{label} disappeared after publication"))?;
    let published_bytes = read_bounded_file(&published, expected, Some(expected), label)?;
    verify_output_leaf_identity(target, &published, label)?;
    if published_bytes != bytes {
        return Err(format!("{label} changed after publication"));
    }
    sync_parent(&target.directory)
        .map_err(|_| format!("{label} directory entry could not be synchronized"))?;
    verify_output_parent_identity(target)?;
    verify_output_leaf_identity(target, &published, label)?;
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn publish_exact(_target: &OutputTarget, _bytes: &[u8], label: &str) -> Result<(), String> {
    Err(format!("{label} cannot be published safely on this platform"))
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn open_output_leaf(target: &OutputTarget, label: &str) -> Result<Option<File>, String> {
    use rustix::fs::{Mode, OFlags, openat};

    verify_output_parent_identity(target)?;
    match openat(
        &target.directory,
        &target.name,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(descriptor) => {
            let file = File::from(descriptor);
            if !file
                .metadata()
                .map_err(|_| format!("{label} could not be inspected"))?
                .is_file()
            {
                return Err(format!("{label} is not a regular file"));
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
fn verify_named_leaf_identity(
    target: &OutputTarget,
    name: &std::ffi::OsStr,
    expected: &File,
    label: &str,
) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt as _;

    use rustix::fs::{Mode, OFlags, openat};

    let current = File::from(
        openat(
            &target.directory,
            name,
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|_| format!("{label} identity could not be verified"))?,
    );
    let expected = expected
        .metadata()
        .map_err(|_| format!("{label} identity could not be read"))?;
    let current = current
        .metadata()
        .map_err(|_| format!("{label} identity could not be read"))?;
    if !expected.is_file()
        || !current.is_file()
        || expected.dev() != current.dev()
        || expected.ino() != current.ino()
    {
        return Err(format!("{label} identity changed during publication"));
    }
    Ok(())
}

fn retained_staging_suffix(name: &std::ffi::OsStr) -> String {
    format!(
        "; staging bytes retained in the output directory as {}",
        name.to_string_lossy()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_is_idempotent_but_never_overwrites_different_bytes() {
        let directory = tempfile::tempdir().expect("create output directory");
        let output = directory.path().join("artifact.bin");

        publish_exact(&output, b"first", "test artifact").expect("initial publication");
        publish_exact(&output, b"first", "test artifact").expect("idempotent recovery");
        let error = publish_exact(&output, b"second", "test artifact")
            .expect_err("different bytes must not overwrite");

        assert!(error.contains("refusing to overwrite"));
        assert_eq!(
            read_bounded(&output, 5, Some(5), "test artifact").expect("read published bytes"),
            b"first"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
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
