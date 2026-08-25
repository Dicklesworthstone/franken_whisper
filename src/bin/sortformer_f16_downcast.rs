#![forbid(unsafe_code)]

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use franken_whisper::sortformer_f16_downcast::derive_sortformer_f16_artifact;

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
    let parent_receipt = fs::read(&args.parent_receipt)
        .map_err(|_| "parent receipt could not be read".to_owned())?;
    let parent_package = fs::read(&args.parent_package)
        .map_err(|_| "parent package could not be read".to_owned())?;
    let artifact = derive_sortformer_f16_artifact(parent_receipt, parent_package)
        .map_err(|error| error.to_string())?;
    let package_sha256 = artifact.receipt().package.sha256.clone();
    let package_bytes = artifact.receipt().package.bytes;
    let receipt_sha256 = artifact.receipt_sha256();
    let (derived_package, derived_receipt) = artifact.into_bytes();

    let mut package_output = create_new(&args.output_package, "output package")?;
    let mut receipt_output = create_new(&args.output_receipt, "output receipt")?;
    package_output
        .write_all(&derived_package)
        .and_then(|()| package_output.sync_all())
        .map_err(|_| "output package could not be published".to_owned())?;
    receipt_output
        .write_all(&derived_receipt)
        .and_then(|()| receipt_output.sync_all())
        .map_err(|_| "output receipt could not be published".to_owned())?;

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
    for path in [&args.output_package, &args.output_receipt] {
        if path
            .try_exists()
            .map_err(|_| "output path could not be inspected".to_owned())?
        {
            return Err("refusing to overwrite an existing output".to_owned());
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err("output parent directory does not exist".to_owned());
        }
    }
    Ok(())
}

fn create_new(path: &Path, label: &str) -> Result<std::fs::File, String> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| format!("{label} could not be created without overwriting"))
}
