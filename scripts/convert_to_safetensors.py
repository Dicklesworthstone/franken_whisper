#!/usr/bin/env python3
"""Convert a PyTorch checkpoint (.ckpt / .pt state dict) to safetensors.

OFFLINE TOOLING ONLY. FrankenWhisper's Rust engine NEVER invokes this script and
NEVER unpickles anything — reading a PyTorch pickle can execute arbitrary code,
so unpickling is a deliberate, human-run, out-of-band step. The Rust loader
(src/native_engine/weights.rs) reads the resulting *.safetensors only.

Every retained tensor is written as contiguous f32. Metadata deliberately
excludes source paths and timestamps so the same pinned input and dependency
versions produce a byte-identical package.

The frozen ECAPA profile requires exactly:
    numpy==2.2.6
    torch==2.7.1
    safetensors==0.5.3

Usage:
    python3 convert_to_safetensors.py INPUT.ckpt OUTPUT.safetensors
        [--key KEY] [--profile PROFILE]

    --key KEY  if the checkpoint is a dict wrapping the state dict under a key
               (e.g. "state_dict" / "model"), unwrap that key first.
    --profile   "generic" (default), or "ecapa-tdnn-voxceleb-v1" for the exact
                frozen SpeechBrain checkpoint and 200-tensor inference census.

On success the output path and its sha256 are printed (pin this in
fetch_aux_models.sh).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ECAPA_PROFILE = "ecapa-tdnn-voxceleb-v1"
ECAPA_EXPORTER_VERSION = "franken-whisper-ecapa-export-v1"
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_MODEL_REVISION = "eac27266f68caa806381260bd44ace38b136c76a"
ECAPA_SOURCE_SHA256 = "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
ECAPA_SOURCE_BYTES = 83_316_686
ECAPA_SOURCE_TENSORS = 231
ECAPA_DROPPED_BATCH_COUNTERS = 31
ECAPA_EXPORTED_TENSORS = 200
REQUIRED_TORCH_VERSION = "2.7.1"
REQUIRED_SAFETENSORS_VERSION = "0.5.3"
REQUIRED_NUMPY_VERSION = "2.2.6"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_deterministic_safetensors(
    tensors: dict[str, "torch.Tensor"],
    output: Path,
    metadata: dict[str, str],
) -> None:
    """Write canonical F32 safetensors, then validate it with safetensors."""
    from safetensors import safe_open

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

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(struct.pack("<Q", len(header_json)))
        destination.write(header_json)
        for name in sorted(tensors):
            array = tensors[name].numpy().astype("<f4", copy=False)
            destination.write(array.tobytes(order="C"))

    # Independent format validation. The Rust runtime performs stricter
    # model-specific census, dtype, metadata, byte-size, and hash checks.
    with safe_open(output, framework="pt", device="cpu") as package:
        if sorted(package.keys()) != sorted(tensors):
            raise RuntimeError("written safetensors tensor census changed")
        if package.metadata() != metadata:
            raise RuntimeError("written safetensors metadata changed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .ckpt/.pt state dict to f32 safetensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="input .ckpt / .pt file")
    parser.add_argument("output", type=Path, help="output .safetensors file")
    parser.add_argument(
        "--key",
        default=None,
        help="unwrap the state dict from this top-level key (e.g. state_dict)",
    )
    parser.add_argument(
        "--profile",
        choices=("generic", ECAPA_PROFILE),
        default="generic",
        help="model-specific validation/export profile (default: generic)",
    )
    args = parser.parse_args()

    try:
        import numpy
        import safetensors
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"error: missing dependency ({exc}); pip install torch safetensors", file=sys.stderr)
        return 2

    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if args.output.exists():
        print(f"error: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2

    input_sha = _sha256(args.input)
    if args.profile == ECAPA_PROFILE:
        if args.key is not None:
            print(f"error: {ECAPA_PROFILE} does not accept --key", file=sys.stderr)
            return 2
        if input_sha != ECAPA_SOURCE_SHA256:
            print(
                f"error: {ECAPA_PROFILE} input sha256 mismatch "
                f"(got {input_sha}, want {ECAPA_SOURCE_SHA256})",
                file=sys.stderr,
            )
            return 2
        if args.input.stat().st_size != ECAPA_SOURCE_BYTES:
            print(
                f"error: {ECAPA_PROFILE} input size mismatch "
                f"(got {args.input.stat().st_size}, want {ECAPA_SOURCE_BYTES})",
                file=sys.stderr,
            )
            return 2
        torch_version = torch.__version__.split("+", 1)[0]
        if torch_version != REQUIRED_TORCH_VERSION:
            print(
                f"error: {ECAPA_PROFILE} requires torch=={REQUIRED_TORCH_VERSION} "
                f"(got {torch.__version__})",
                file=sys.stderr,
            )
            return 2
        if safetensors.__version__ != REQUIRED_SAFETENSORS_VERSION:
            print(
                f"error: {ECAPA_PROFILE} requires safetensors=="
                f"{REQUIRED_SAFETENSORS_VERSION} (got {safetensors.__version__})",
                file=sys.stderr,
            )
            return 2
        if numpy.__version__ != REQUIRED_NUMPY_VERSION:
            print(
                f"error: {ECAPA_PROFILE} requires numpy=={REQUIRED_NUMPY_VERSION} "
                f"(got {numpy.__version__})",
                file=sys.stderr,
            )
            return 2

    # weights_only=True refuses arbitrary pickled callables (defense in depth).
    obj = torch.load(args.input, map_location="cpu", weights_only=True)
    if args.key is not None:
        obj = obj[args.key]
    if not isinstance(obj, dict):
        print(f"error: checkpoint is not a state dict (got {type(obj).__name__})", file=sys.stderr)
        return 2

    tensors: dict[str, "torch.Tensor"] = {}
    dropped_batch_counters = 0
    for name, value in obj.items():
        if not isinstance(name, str):
            print(f"error: non-string state-dict key: {name!r}", file=sys.stderr)
            return 2
        if not isinstance(value, torch.Tensor):
            if args.profile == ECAPA_PROFILE:
                print(
                    f"error: {ECAPA_PROFILE} non-tensor entry: "
                    f"{name} ({type(value).__name__})",
                    file=sys.stderr,
                )
                return 2
            print(f"  skip non-tensor entry: {name} ({type(value).__name__})", file=sys.stderr)
            continue
        if args.profile == ECAPA_PROFILE and name.endswith(".num_batches_tracked"):
            if value.dtype != torch.int64 or value.numel() != 1:
                print(
                    f"error: invalid BatchNorm counter {name}: "
                    f"dtype={value.dtype}, elements={value.numel()}",
                    file=sys.stderr,
                )
                return 2
            dropped_batch_counters += 1
            continue
        if args.profile == ECAPA_PROFILE and value.dtype != torch.float32:
            print(
                f"error: retained ECAPA tensor {name} has dtype {value.dtype}, want torch.float32",
                file=sys.stderr,
            )
            return 2
        tensors[name] = value.detach().to(torch.float32).contiguous()

    if not tensors:
        print("error: no tensors found in checkpoint", file=sys.stderr)
        return 2

    if args.profile == ECAPA_PROFILE:
        if len(obj) != ECAPA_SOURCE_TENSORS:
            print(
                f"error: {ECAPA_PROFILE} source tensor count mismatch "
                f"(got {len(obj)}, want {ECAPA_SOURCE_TENSORS})",
                file=sys.stderr,
            )
            return 2
        if dropped_batch_counters != ECAPA_DROPPED_BATCH_COUNTERS:
            print(
                f"error: {ECAPA_PROFILE} BatchNorm counter count mismatch "
                f"(got {dropped_batch_counters}, want {ECAPA_DROPPED_BATCH_COUNTERS})",
                file=sys.stderr,
            )
            return 2
        if len(tensors) != ECAPA_EXPORTED_TENSORS:
            print(
                f"error: {ECAPA_PROFILE} exported tensor count mismatch "
                f"(got {len(tensors)}, want {ECAPA_EXPORTED_TENSORS})",
                file=sys.stderr,
            )
            return 2
        metadata = {
            "converter": "franken_whisper/scripts/convert_to_safetensors.py",
            "exporter_version": ECAPA_EXPORTER_VERSION,
            "profile": ECAPA_PROFILE,
            "source_model_id": ECAPA_MODEL_ID,
            "source_model_revision": ECAPA_MODEL_REVISION,
            "source_checkpoint_sha256": input_sha,
            "source_checkpoint_bytes": str(ECAPA_SOURCE_BYTES),
            "source_tensor_count": str(ECAPA_SOURCE_TENSORS),
            "dropped_batch_counter_count": str(ECAPA_DROPPED_BATCH_COUNTERS),
            "exported_tensor_count": str(ECAPA_EXPORTED_TENSORS),
            "exported_dtype": "F32",
            "numpy_version": REQUIRED_NUMPY_VERSION,
            "torch_version": REQUIRED_TORCH_VERSION,
            "safetensors_version": REQUIRED_SAFETENSORS_VERSION,
        }
    else:
        metadata = {
            "converter": "franken_whisper/scripts/convert_to_safetensors.py",
            "profile": "generic",
            "source_sha256": input_sha,
            "exported_tensor_count": str(len(tensors)),
            "exported_dtype": "F32",
        }

    _save_deterministic_safetensors(tensors, args.output, metadata)

    out_sha = _sha256(args.output)
    print(f"wrote {len(tensors)} tensors -> {args.output}")
    print(f"input  sha256: {input_sha}")
    print(f"output sha256: {out_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
