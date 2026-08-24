#!/usr/bin/env python3
"""Export the pinned breizhn/DTLN speech-separation weights to path-free f32 artifacts.

OFFLINE TOOLING ONLY. The Rust runtime never imports Python and never reads
this HDF5 file. Run this script in the exact pinned oracle environment with
operator-supplied input files; it downloads nothing (an offline audit hook
denies sockets and child processes).

UPSTREAM IDENTITY (pinned):
    repository  breizhn/DTLN @ 1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc
                (MIT license; pretrained_model/ ships inside the licensed tree)
    checkpoint  pretrained_model/DTLN_norm_500h.h5
ARCHITECTURE CONTRACT (DTLN_model.py defaults at that revision):
    blockLen=512, block_shift=128, numUnits=128, numLayer=2, encoder_size=256,
    eps=1e-7, norm_stft=True (log(mag + 1e-7) before instant layer norm)
GATE CONTRACT: Keras LSTM columns are grouped [i, f, g, o] — identical to the
native engine's `LstmWeights` row grouping (src/native_engine/nn.rs), so no
permutation is applied or needed. Keras applies its fused bias ONCE per gate
(bias maps to b_ih; b_hh is zeros). Recurrent activation is Keras' DEFAULT
hard_sigmoid — clip(0.2*x + 0.5, 0, 1) — NOT the logistic sigmoid of the
upstream ONNX exports (which are therefore reference-only and deliberately
not consumed here). The Rust consumer must implement hard_sigmoid faithfully.
STFT SYNTHESIS: the analysis/synthesis bases are not stored in the model; the
consumer synthesizes them deterministically from blockLen/block_shift. This
receipt records those constants as the synthesis contract.
"""

import hashlib
import hmac
import json
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

RECEIPT_SCHEMA = "franken-whisper-dtln-conversion-receipt-v1"
ARTIFACT_SCHEMA = "franken-whisper-dtln-weights-f32-v1"

UPSTREAM_REPO = "breizhn/DTLN"
UPSTREAM_REVISION = "1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc"
CHECKPOINT_BYTES = 4_003_624
CHECKPOINT_SHA256 = "378c209ad3f4dedffb185b542169a51472361a23492c13b377feb2485a2a1bb3"
REFERENCE_ONLY_ONNX_SHA256 = {
    "model_1.onnx": "22b91cae3855e5a0620e66a917ca6c82c58db0e842c770f58d86751c5e8d4ae3",
    "model_2.onnx": "e20c92f9233fccf29cddf86970d0d0161a03aebccc26d6f4d5639c4d5ec2e639",
}

BLOCK_LEN = 512
BLOCK_SHIFT = 128
NUM_UNITS = 128
NUM_LAYER = 2
ENCODER_SIZE = 256

REQUIRED_PYTHON = (3, 13, 9)
REQUIRED_PACKAGES = {"h5py": "3.12.1", "numpy": "2.5.2"}

# Expected weight manifest, derived from the architecture contract (NOT from
# the observed file): export-name -> shape. Any drift fails closed.
EXPECTED_WEIGHTS: dict[str, tuple[int, ...]] = {
    # stage 1: STFT-magnitude mask kernel
    "s1.in_norm.gamma": (257,),
    "s1.in_norm.beta": (257,),
    "s1.lstm0.kernel": (257, 512),
    "s1.lstm0.recurrent": (128, 512),
    "s1.lstm0.bias": (512,),
    "s1.lstm1.kernel": (128, 512),
    "s1.lstm1.recurrent": (128, 512),
    "s1.lstm1.bias": (512,),
    "s1.mask.kernel": (128, 257),
    "s1.mask.bias": (257,),
    # stage 2: learned-domain mask kernel
    "s2.encoder": (1, 512, 256),
    "s2.in_norm.gamma": (256,),
    "s2.in_norm.beta": (256,),
    "s2.lstm0.kernel": (256, 512),
    "s2.lstm0.recurrent": (128, 512),
    "s2.lstm0.bias": (512,),
    "s2.lstm1.kernel": (128, 512),
    "s2.lstm1.recurrent": (128, 512),
    "s2.lstm1.bias": (512,),
    "s2.mask.kernel": (128, 256),
    "s2.mask.bias": (256,),
    "s2.decoder": (1, 256, 512),
}

H5_TO_EXPORT = {
    "instant_layer_normalization/instant_layer_normalization/gamma:0": "s1.in_norm.gamma",
    "instant_layer_normalization/instant_layer_normalization/beta:0": "s1.in_norm.beta",
    "lstm/lstm/lstm_cell/kernel:0": "s1.lstm0.kernel",
    "lstm/lstm/lstm_cell/recurrent_kernel:0": "s1.lstm0.recurrent",
    "lstm/lstm/lstm_cell/bias:0": "s1.lstm0.bias",
    "lstm_1/lstm_1/lstm_cell_1/kernel:0": "s1.lstm1.kernel",
    "lstm_1/lstm_1/lstm_cell_1/recurrent_kernel:0": "s1.lstm1.recurrent",
    "lstm_1/lstm_1/lstm_cell_1/bias:0": "s1.lstm1.bias",
    "dense/dense/kernel:0": "s1.mask.kernel",
    "dense/dense/bias:0": "s1.mask.bias",
    "conv1d/conv1d/kernel:0": "s2.encoder",
    "instant_layer_normalization_1/instant_layer_normalization_1/gamma:0": "s2.in_norm.gamma",
    "instant_layer_normalization_1/instant_layer_normalization_1/beta:0": "s2.in_norm.beta",
    "lstm_2/lstm_2/lstm_cell_2/kernel:0": "s2.lstm0.kernel",
    "lstm_2/lstm_2/lstm_cell_2/recurrent_kernel:0": "s2.lstm0.recurrent",
    "lstm_2/lstm_2/lstm_cell_2/bias:0": "s2.lstm0.bias",
    "lstm_3/lstm_3/lstm_cell_3/kernel:0": "s2.lstm1.kernel",
    "lstm_3/lstm_3/lstm_cell_3/recurrent_kernel:0": "s2.lstm1.recurrent",
    "lstm_3/lstm_3/lstm_cell_3/bias:0": "s2.lstm1.bias",
    "dense_1/dense_1/kernel:0": "s2.mask.kernel",
    "dense_1/dense_1/bias:0": "s2.mask.bias",
    "conv1d_1/conv1d_1/kernel:0": "s2.decoder",
}


def _install_offline_audit_hook() -> None:
    def _deny(event: str, args: list[Any]) -> None:
        if event in ("socket.connect", "subprocess.Popen", "os.system"):
            raise RuntimeError(f"offline audit hook denied {event}")

    sys.addaudithook(_deny)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stable_file_bytes(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    raw = path.read_bytes()
    if len(raw) != expected_bytes:
        raise RuntimeError(f"{path}: expected {expected_bytes} bytes, found {len(raw)}")
    digest = _sha256_bytes(raw)
    if not hmac.compare_digest(digest, expected_sha256):
        raise RuntimeError(f"{path}: sha256 {digest} != pinned {expected_sha256}")
    return raw


def _require_runtime() -> dict[str, Any]:
    if sys.version_info[:3] != REQUIRED_PYTHON:
        raise RuntimeError(
            f"pinned DTLN export requires Python {REQUIRED_PYTHON}, found {sys.version_info[:3]}"
        )
    observed = {}
    for module, version in REQUIRED_PACKAGES.items():
        imported = __import__(module)
        observed[module] = imported.__version__
        if observed[module] != version:
            raise RuntimeError(f"{module} {observed[module]} != pinned {version}")
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "python_interpreter_sha256": _sha256_bytes(executable.read_bytes()),
        "python_version": list(sys.version_info[:3]),
        "packages": observed,
    }


def _require_external_input(path: Path, label: str) -> Path:
    repo_root = Path(__file__).resolve(strict=True).parent.parent
    canonical = path.resolve(strict=True)
    if canonical == repo_root or repo_root in canonical.parents:
        raise RuntimeError(f"{label} must stay outside the repository")
    return canonical


def _validated_f32(name: str, dataset: h5py.Dataset) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(dataset[()], dtype=np.float32))
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name}: non-finite values")
    return array


def _prepare_output_directory(output_dir: Path) -> tuple[Path, int]:
    repo_root = Path(__file__).resolve(strict=True).parent.parent
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("output directory must not already exist")
    parent = output_dir.parent.resolve(strict=True)
    destination = parent / output_dir.name
    if destination == repo_root or repo_root in destination.parents:
        raise RuntimeError("DTLN artifacts must stay outside the repository")
    os.mkdir(destination, mode=0o700)
    descriptor = os.open(
        destination,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    os.fchmod(descriptor, 0o700)
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o700:
        os.close(descriptor)
        raise RuntimeError("output directory identity or mode changed")
    return destination, descriptor


def _publish_exclusive(directory_fd: int, name: str, value: bytes) -> None:
    fd = os.open(
        name,
        os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    with os.fdopen(fd, "w+b") as out:
        os.fchmod(out.fileno(), 0o600)
        created = os.fstat(out.fileno())
        if not stat.S_ISREG(created.st_mode):
            raise RuntimeError("published output is not a regular file")
        out.write(value)
        out.flush()
        os.fsync(out.fileno())
        out.seek(0)
        reread = out.read()
    published = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    identity = (created.st_dev, created.st_ino)
    if (
        identity != (published.st_dev, published.st_ino)
        or stat.S_IMODE(published.st_mode) != 0o600
        or not hmac.compare_digest(_sha256_bytes(reread), _sha256_bytes(value))
    ):
        raise RuntimeError("published output identity changed")


def write_safetensors_f32(tensors: dict[str, np.ndarray]) -> bytes:
    """Minimal deterministic safetensors writer (F32 little-endian, no metadata)."""
    header: dict[str, Any] = {}
    offset = 0
    blobs: list[bytes] = []
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name], dtype="<f4")
        blob = array.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    header_bytes = _canonical_json_bytes(header)
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(blobs)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        print("usage: export_dtln.py <DTLN_norm_500h.h5> <new_output_dir>")
        return 2
    _install_offline_audit_hook()
    checkpoint_path = _require_external_input(Path(sys.argv[1]), "checkpoint")
    runtime = _require_runtime()
    _stable_file_bytes(checkpoint_path, CHECKPOINT_BYTES, CHECKPOINT_SHA256)

    extracted: dict[str, np.ndarray] = {}
    seen_h5_keys: set[str] = set()
    with h5py.File(checkpoint_path, "r") as handle:
        # Keras 2 legacy layout: weight groups live directly under the file
        # root (verified against the pinned checkpoint), not under
        # "model_weights".
        for h5_key, export_name in H5_TO_EXPORT.items():
            node = handle
            for part in h5_key.split("/")[:-1]:
                node = node[part]
            dataset = node[h5_key.split("/")[-1]]
            extracted[export_name] = _validated_f32(export_name, dataset)
            seen_h5_keys.add(h5_key)
        walk_names: list[str] = []

        def _collect(node_name: str) -> None:
            walk_names.append(node_name)

        handle.visititems(lambda n, _obj: _collect(n))
        unexpected = [
            name for name in walk_names if name.endswith(":0") and name not in seen_h5_keys
        ]
        if unexpected:
            raise RuntimeError(f"unmapped weight tensors present: {sorted(unexpected)}")

    if set(extracted) != set(EXPECTED_WEIGHTS):
        missing = sorted(set(EXPECTED_WEIGHTS) - set(extracted))
        extra = sorted(set(extracted) - set(EXPECTED_WEIGHTS))
        raise RuntimeError(f"census drift: missing={missing} extra={extra}")
    for name, shape in EXPECTED_WEIGHTS.items():
        if tuple(extracted[name].shape) != shape:
            raise RuntimeError(f"{name}: shape {tuple(extracted[name].shape)} != contract {shape}")
    total_elements = int(sum(int(np.prod(v.shape)) for v in extracted.values()))

    artifact = write_safetensors_f32(extracted)
    tensor_manifest = [
        {
            "elements": int(np.prod(extracted[name].shape)),
            "name": name,
            "shape": list(extracted[name].shape),
            "sha256_f32le": _sha256_bytes(extracted[name].tobytes()),
        }
        for name in sorted(extracted)
    ]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "distribution_status": "operator_local",
        "distribution_note": (
            "artifacts stay operator-local by project provenance policy (size + "
            "reproducibility), not by license restriction: upstream is MIT with "
            "weights inside the licensed tree"
        ),
        "upstream": {
            "repo": UPSTREAM_REPO,
            "revision": UPSTREAM_REVISION,
            "license": "MIT",
            "weights_location": "in-tree pretrained_model/ (inside licensed repository)",
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "reference_only_onnx_sha256": REFERENCE_ONLY_ONNX_SHA256,
            "onnx_disposition": (
                "reference-only: keras2onnx replaces the trained hard_sigmoid "
                "recurrence with logistic sigmoid; h5 is the semantics-preserving source"
            ),
        },
        "architecture_contract": {
            "block_len": BLOCK_LEN,
            "block_shift": BLOCK_SHIFT,
            "num_units": NUM_UNITS,
            "num_layer": NUM_LAYER,
            "encoder_size": ENCODER_SIZE,
            "stft_norm_eps": 1e-7,
            "gate_order": ["i", "f", "g", "o"],
            "gate_order_note": "Keras column order equals native LstmWeights row order; no permutation",
            "recurrent_activation": "hard_sigmoid: clip(0.2*x + 0.5, 0, 1)",
            "bias_application": "single fused bias per gate -> b_ih; recurrent bias is zero",
            "kernel_layout": "[input, 4H] matches native matmul_bias w_t convention without transpose",
            "stft_synthesis": "analysis/synthesis bases synthesized from block_len/block_shift; not stored",
        },
        "runtime_identity": runtime,
        "exported_tensors": len(tensor_manifest),
        "exported_elements": total_elements,
        "tensor_manifest": tensor_manifest,
        "artifact": {
            "file": "dtln_weights_f32.safetensors",
            "bytes": len(artifact),
            "sha256": _sha256_bytes(artifact),
        },
        "runtime_parity": "pending bd-f2se: Rust forward vs upstream TFLite/onnxruntime oracle on fixed fixtures",
    }

    destination, directory_fd = _prepare_output_directory(Path(sys.argv[2]))
    try:
        _publish_exclusive(directory_fd, receipt["artifact"]["file"], artifact)
        _publish_exclusive(directory_fd, "receipt.json", _canonical_json_bytes(receipt))
    finally:
        os.close(directory_fd)
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "tensors": len(tensor_manifest),
                "elements": total_elements,
                "artifact_sha256": receipt["artifact"]["sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
