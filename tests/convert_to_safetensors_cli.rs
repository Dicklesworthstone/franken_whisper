#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LEAK_MARKER: &str = "PRIVATE_LEAK_MARKER";

struct ConverterRun {
    output: Output,
    output_path: PathBuf,
}

fn write_python_stubs(root: &Path) -> PathBuf {
    let stubs = root.join("python-stubs");
    let safetensors = stubs.join("safetensors");
    fs::create_dir_all(&safetensors).expect("create Python stub packages");
    fs::write(
        stubs.join("sitecustomize.py"),
        r#"import os
import pathlib

_real_open = pathlib.Path.open

def _controlled_open(path, *args, **kwargs):
    if str(path) == os.environ.get("FW_TEST_UNREADABLE_INPUT"):
        raise PermissionError("PRIVATE_LEAK_MARKER unreadable checkpoint")
    return _real_open(path, *args, **kwargs)

pathlib.Path.open = _controlled_open
"#,
    )
    .expect("write sitecustomize stub");
    fs::write(stubs.join("numpy.py"), "__version__ = 'test-numpy'\n").expect("write numpy stub");
    fs::write(
        stubs.join("torch.py"),
        r#"import struct

__version__ = "test-torch"
float32 = "float32"

class FakeArray:
    def __init__(self, values):
        self.values = tuple(values)

    def astype(self, _dtype, copy=False):
        return self

    def tobytes(self, order="C"):
        return struct.pack("<" + "f" * len(self.values), *self.values)

class Tensor:
    def __init__(self, shape=(1,), values=(1.0,)):
        self.shape = tuple(shape)
        self.values = tuple(values)
        self.dtype = float32

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def to(self, _dtype):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return FakeArray(self.values)

def load(source, map_location=None, weights_only=None):
    assert map_location == "cpu"
    assert weights_only is True
    payload = source.read()
    if payload == b"MALFORMED":
        raise RuntimeError("PRIVATE_LEAK_MARKER malformed checkpoint")
    if payload == b"MISSING_KEY":
        return {"present": Tensor()}
    if payload == b"NOT_A_DICT":
        return [Tensor()]
    if payload == b"NESTED":
        return {"model": {
            "z.weight": Tensor((2,), (3.0, 4.0)),
            "a.weight": Tensor((1,), (1.5,)),
            "ignored": "metadata",
        }}
    return {"weight": Tensor()}
"#,
    )
    .expect("write torch stub");
    fs::write(
        safetensors.join("__init__.py"),
        "__version__ = 'test-safetensors'\n",
    )
    .expect("write safetensors package stub");
    fs::write(
        safetensors.join("torch.py"),
        r#"import json
import struct

from torch import Tensor

def load(package):
    header_length = struct.unpack("<Q", package[:8])[0]
    header = json.loads(package[8:8 + header_length])
    data = package[8 + header_length:]
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        count = (end - begin) // 4
        values = struct.unpack("<" + "f" * count, data[begin:end])
        tensors[name] = Tensor(entry["shape"], values)
    return tensors
"#,
    )
    .expect("write safetensors.torch stub");
    stubs
}

fn run_converter(
    root: &Path,
    checkpoint_bytes: &[u8],
    key: Option<&str>,
    unreadable: bool,
) -> ConverterRun {
    let stubs = write_python_stubs(root);
    let input = root.join("operator-private-input.ckpt");
    let output_path = root.join("new-output").join("weights.safetensors");
    fs::write(&input, checkpoint_bytes).expect("write checkpoint fixture");

    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("convert_generic_to_safetensors.py");
    let mut command = Command::new("python3");
    command
        .arg(script)
        .arg(&input)
        .arg(&output_path)
        .env("PYTHONPATH", stubs)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1");
    if let Some(key) = key {
        command.arg("--key").arg(key);
    }
    if unreadable {
        command.env("FW_TEST_UNREADABLE_INPUT", &input);
    }
    ConverterRun {
        output: command.output().expect("run generic converter"),
        output_path,
    }
}

fn assert_stable_failure(run: &ConverterRun, expected_stderr: &[u8]) {
    assert_eq!(run.output.status.code(), Some(2));
    assert_eq!(run.output.stdout, b"");
    assert_eq!(run.output.stderr, expected_stderr);
    let stderr = String::from_utf8_lossy(&run.output.stderr);
    assert!(!stderr.contains("Traceback"));
    assert!(!stderr.contains(LEAK_MARKER));
    assert!(!run.output_path.exists());
    assert!(!run.output_path.parent().expect("output parent").exists());
}

#[test]
fn generic_conversion_is_deterministic_and_create_new() {
    let first_root = tempfile::tempdir().expect("first converter harness");
    let first = run_converter(first_root.path(), b"NESTED", Some("model"), false);
    assert!(
        first.output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&first.output.stderr)
    );
    assert_eq!(first.output.stderr, b"");
    let first_bytes = fs::read(&first.output_path).expect("read first package");
    assert_eq!(
        fs::metadata(&first.output_path)
            .expect("first package metadata")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    let stdout = String::from_utf8(first.output.stdout).expect("UTF-8 success output");
    assert!(stdout.starts_with("wrote 2 tensors\ninput sha256: "));
    assert!(stdout.contains("\noutput sha256: "));
    assert!(!stdout.contains("operator-private-input"));

    let second_root = tempfile::tempdir().expect("second converter harness");
    let second = run_converter(second_root.path(), b"NESTED", Some("model"), false);
    assert!(
        second.output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&second.output.stderr)
    );
    assert_eq!(
        first_bytes,
        fs::read(&second.output_path).expect("read second package")
    );
    assert_eq!(stdout.as_bytes(), second.output.stdout);
}

#[test]
fn generic_direct_state_dict_converts_without_a_key() {
    let root = tempfile::tempdir().expect("direct converter harness");
    let run = run_converter(root.path(), b"VALID", None, false);
    assert!(
        run.output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run.output.stderr)
    );
    assert_eq!(run.output.stderr, b"");
    assert!(run.output_path.is_file());
    assert!(String::from_utf8_lossy(&run.output.stdout).starts_with("wrote 1 tensors\n"));
}

#[test]
fn generic_unreadable_input_is_a_stable_non_leaking_error() {
    let root = tempfile::tempdir().expect("unreadable converter harness");
    let run = run_converter(root.path(), b"VALID", None, true);
    assert_stable_failure(&run, b"error: checkpoint input could not be read\n");
}

#[test]
fn generic_malformed_checkpoint_is_a_stable_non_leaking_error() {
    let root = tempfile::tempdir().expect("malformed converter harness");
    let run = run_converter(root.path(), b"MALFORMED", None, false);
    assert_stable_failure(&run, b"error: checkpoint could not be loaded\n");
}

#[test]
fn generic_missing_requested_key_is_a_stable_non_leaking_error() {
    let root = tempfile::tempdir().expect("missing-key converter harness");
    let run = run_converter(root.path(), b"MISSING_KEY", Some(LEAK_MARKER), false);
    assert_stable_failure(&run, b"error: requested checkpoint key is missing\n");
}

#[test]
fn generic_converter_refuses_to_overwrite_an_output() {
    let root = tempfile::tempdir().expect("no-clobber converter harness");
    let stubs = write_python_stubs(root.path());
    let input = root.path().join("input.ckpt");
    let output_path = root.path().join("weights.safetensors");
    fs::write(&input, b"VALID").expect("write checkpoint fixture");
    fs::write(&output_path, b"EXISTING_BYTES").expect("write existing output fixture");
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("convert_generic_to_safetensors.py");
    let output = Command::new("python3")
        .arg(script)
        .arg(input)
        .arg(&output_path)
        .env("PYTHONPATH", stubs)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .expect("run no-clobber converter probe");
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(output.stdout, b"");
    assert_eq!(
        output.stderr,
        b"error: refusing to overwrite existing output\n"
    );
    assert_eq!(
        fs::read(output_path).expect("read preserved existing output"),
        b"EXISTING_BYTES"
    );
}
