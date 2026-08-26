#![forbid(unsafe_code)]

use std::env;
use std::fs::File;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use franken_whisper::sortformer_conformance::SORTFORMER_PACKAGE_BYTES;
use franken_whisper::sortformer_f16_downcast::derive_sortformer_f16_artifact;
use tempfile::{Builder as TempFileBuilder, NamedTempFile};

const MAX_PARENT_RECEIPT_BYTES: u64 = 8 * 1024 * 1024;

struct Args {
    parent_receipt: PathBuf,
    parent_package: PathBuf,
    output_package: PathBuf,
    output_receipt: PathBuf,
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
    validate_outputs(&args)?;
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

    publish_exact(&args.output_package, &derived_package, "output package")?;
    publish_exact(&args.output_receipt, &derived_receipt, "output receipt")?;

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

fn validate_outputs(args: &Args) -> Result<(), String> {
    if args.output_package == args.output_receipt {
        return Err("output package and receipt paths must be distinct".to_owned());
    }
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .map_err(|_| "repository boundary could not be resolved".to_owned())?;
    for path in [&args.output_package, &args.output_receipt] {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err("output parent directory does not exist".to_owned());
        }
        let parent = parent
            .canonicalize()
            .map_err(|_| "output parent directory could not be resolved".to_owned())?;
        if parent.starts_with(&repository) {
            return Err("derived model artifacts must be written outside the repository".to_owned());
        }
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
    file.take(read_limit)
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

fn publish_exact(path: &Path, bytes: &[u8], label: &str) -> Result<(), String> {
    if path
        .try_exists()
        .map_err(|_| format!("{label} path could not be inspected"))?
    {
        let expected = u64::try_from(bytes.len())
            .map_err(|_| format!("{label} size does not fit u64"))?;
        let existing = read_bounded(path, expected, Some(expected), label)?;
        if existing == bytes {
            return Ok(());
        }
        return Err(format!(
            "refusing to overwrite an existing {label} with different bytes"
        ));
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut staged = TempFileBuilder::new()
        .prefix(".sortformer-f16-stage-")
        .tempfile_in(parent)
        .map_err(|_| format!("{label} staging file could not be created"))?;
    if staged
        .write_all(bytes)
        .and_then(|()| staged.as_file().sync_all())
        .is_err()
    {
        return Err(format!(
            "{label} staging bytes could not be persisted{}",
            retain_staging(staged)
        ));
    }
    match staged.persist_noclobber(path) {
        Ok(_) => {}
        Err(error) => {
            return Err(format!(
                "{label} could not be published without overwriting{}",
                retain_staging(error.file)
            ));
        }
    }
    sync_parent_directory(parent, label)?;

    let expected = u64::try_from(bytes.len())
        .map_err(|_| format!("{label} size does not fit u64"))?;
    let published = read_bounded(path, expected, Some(expected), label)?;
    if published != bytes {
        return Err(format!("{label} changed after publication"));
    }
    Ok(())
}

fn retain_staging(staged: NamedTempFile) -> String {
    match staged.keep() {
        Ok((_file, path)) => path.file_name().map_or_else(
            || "; staging bytes retained in the output directory".to_owned(),
            |name| {
                format!(
                    "; staging bytes retained in the output directory as {}",
                    name.to_string_lossy()
                )
            },
        ),
        Err(_) => "; staging-file retention failed".to_owned(),
    }
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path, label: &str) -> Result<(), String> {
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|_| format!("{label} directory entry could not be synchronized"))
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path, _label: &str) -> Result<(), String> {
    Ok(())
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
