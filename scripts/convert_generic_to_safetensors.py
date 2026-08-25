#!/usr/bin/env python3
"""Convert an operator-supplied PyTorch state dict to deterministic safetensors.

This is the generic, unauthenticated conversion path. Frozen ECAPA and
Sortformer conversions remain in ``convert_to_safetensors.py`` because that
script's exact bytes are a compiled trust root for published receipts.

PyTorch checkpoints can contain executable pickle payloads. This tool therefore
uses ``weights_only=True`` and is offline, operator-invoked tooling; the Rust
runtime never imports PyTorch checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch as torch_types

EXIT_INVALID_INPUT = 2


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_INVALID_INPUT


def _load_checkpoint(path: Path, torch: object) -> tuple[object, str]:
    """Hash and decode one stable open checkpoint without reopening its path."""
    with path.open("rb") as source:
        initial = os.fstat(source.fileno())
        hasher = hashlib.sha256()
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
        after_hash = os.fstat(source.fileno())
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        ) != (
            after_hash.st_dev,
            after_hash.st_ino,
            after_hash.st_size,
            after_hash.st_mtime_ns,
        ):
            raise OSError("checkpoint identity changed")
        source.seek(0)
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        after_load = os.fstat(source.fileno())
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        ) != (
            after_load.st_dev,
            after_load.st_ino,
            after_load.st_size,
            after_load.st_mtime_ns,
        ):
            raise OSError("checkpoint identity changed")
    return checkpoint, hasher.hexdigest()


def _tensor_bytes(tensor: torch_types.Tensor) -> bytes:
    return tensor.numpy().astype("<f4", copy=False).tobytes(order="C")


def _build_safetensors(
    tensors: dict[str, torch_types.Tensor], metadata: dict[str, str], torch: object
) -> bytes:
    from safetensors.torch import load

    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        byte_length = tensor.numel() * 4
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + byte_length],
        }
        offset += byte_length

    header_json = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header_json += b" " * (-len(header_json) % 8)

    serialized = io.BytesIO()
    serialized.write(struct.pack("<Q", len(header_json)))
    serialized.write(header_json)
    for name in sorted(tensors):
        serialized.write(_tensor_bytes(tensors[name]))
    file_bytes = serialized.getvalue()
    serialized.close()

    package = load(file_bytes)
    if sorted(package) != sorted(tensors):
        raise RuntimeError("serialized tensor census changed")
    for name, expected in tensors.items():
        observed = package[name]
        if expected.dtype != torch.float32 or observed.dtype != torch.float32:
            raise RuntimeError("serialized tensor dtype changed")
        if tuple(observed.shape) != tuple(expected.shape):
            raise RuntimeError("serialized tensor shape changed")
        observed_digest = hashlib.sha256(_tensor_bytes(observed)).digest()
        expected_digest = hashlib.sha256(_tensor_bytes(expected)).digest()
        if not hmac.compare_digest(observed_digest, expected_digest):
            raise RuntimeError("serialized tensor payload changed")
    return file_bytes


def _publish_new_file(output: Path, file_bytes: bytes) -> None:
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        output,
        os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    with os.fdopen(descriptor, "w+b") as destination:
        os.fchmod(destination.fileno(), 0o600)
        destination.write(file_bytes)
        destination.flush()
        os.fsync(destination.fileno())
        destination.seek(0)
        if destination.read() != file_bytes:
            raise OSError("published bytes changed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a generic PyTorch state dict to deterministic f32 safetensors."
    )
    parser.add_argument("input", type=Path, help="input .ckpt / .pt file")
    parser.add_argument("output", type=Path, help="new output .safetensors file")
    parser.add_argument(
        "--key",
        default=None,
        help="unwrap the state dict from this top-level key",
    )
    args = parser.parse_args()

    try:
        input_is_file = args.input.is_file()
        output_exists = args.output.exists()
    except OSError:
        return _fail("checkpoint input could not be read")
    if not input_is_file:
        return _fail("checkpoint input could not be read")
    if output_exists:
        return _fail("refusing to overwrite existing output")

    try:
        import numpy
        import safetensors
        import torch
    except Exception:
        return _fail("required conversion dependencies are unavailable")

    try:
        checkpoint, input_sha = _load_checkpoint(args.input, torch)
    except OSError:
        return _fail("checkpoint input could not be read")
    except Exception:
        return _fail("checkpoint could not be loaded")

    if not isinstance(checkpoint, dict):
        return _fail("checkpoint state dict is invalid")
    if args.key is not None:
        if args.key not in checkpoint:
            return _fail("requested checkpoint key is missing")
        checkpoint = checkpoint[args.key]
    if not isinstance(checkpoint, dict):
        return _fail("checkpoint state dict is invalid")

    try:
        tensors: dict[str, torch_types.Tensor] = {}
        for name, value in checkpoint.items():
            if not isinstance(name, str):
                return _fail("checkpoint state dict is invalid")
            if not isinstance(value, torch.Tensor):
                continue
            tensors[name] = value.detach().to(torch.float32).contiguous()
        if not tensors:
            return _fail("checkpoint state dict is invalid")
        metadata = {
            "converter": "franken_whisper/scripts/convert_generic_to_safetensors.py",
            "profile": "generic",
            "source_sha256": input_sha,
            "exported_tensor_count": str(len(tensors)),
            "exported_dtype": "F32",
            "numpy_version": numpy.__version__,
            "torch_version": torch.__version__,
            "safetensors_version": safetensors.__version__,
        }
        package_bytes = _build_safetensors(tensors, metadata, torch)
    except Exception:
        return _fail("checkpoint state dict is invalid")

    try:
        _publish_new_file(args.output, package_bytes)
    except (OSError, ValueError):
        return _fail("safetensors output could not be published")

    print(f"wrote {len(tensors)} tensors")
    print(f"input sha256: {input_sha}")
    print(f"output sha256: {hashlib.sha256(package_bytes).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
