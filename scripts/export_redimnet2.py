#!/usr/bin/env python3
"""Export the pinned ReDimNet2-B2 oracle to path-free f32 artifacts.

OFFLINE TOOLING ONLY. The Rust runtime never imports Python and never loads the
pickle-backed upstream checkpoint. Run this script in the exact pinned oracle
environment after placing both the official v1.0.0 source tree and checkpoint
outside the FrankenWhisper repository.

The output directory must not exist and must be outside this repository. It is
created mode 0700; every artifact is created exclusively at mode 0600. The
synthetic truth pack contains no human speech, transcript, speaker identity, or
biometric sample. Receipts contain hashes and tensor shapes, never local paths.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import importlib.metadata
import io
import json
import os
import stat
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch as torch_types

EXPORTER_SCHEMA = "franken-whisper-redimnet2-export-v1"
RECEIPT_SCHEMA = "franken-whisper-redimnet2-conversion-receipt-v1"
TRUTH_SCHEMA = "franken-whisper-redimnet2-synthetic-truth-v1"
MODEL_ID = "PalabraAI/redimnet2:b2-vox2-lm"
MODEL_RELEASE = "v1.0.0"
MODEL_SOURCE_REVISION = "5294667e806ac3b0f27abc301a114ef132b64b42"
CHECKPOINT_BYTES = 15_897_450
CHECKPOINT_SHA256 = "0545a29679a87fe1c662d2bbd05e3b3fe0d1b392832729abaa135e4079a2f77a"
CONFIG_SHA256 = "63939097377bff85dc1553a54f6aa2dcacfea106881addde475fb4f64505dd1a"
SOURCE_TENSORS = 729
SOURCE_ELEMENTS = 3_918_862
DROPPED_BATCH_COUNTERS = 68
EXPORTED_TENSORS = 661
EXPORTED_ELEMENTS = 3_918_794
TOTAL_PARAMETERS = 3_677_760
TRAINABLE_PARAMETERS = 3_676_320
FROZEN_PARAMETERS = 1_440
SAMPLE_RATE_HZ = 16_000
SYNTHETIC_SAMPLES = 48_000
SYNTHETIC_FIXTURE_ID = "modular-noise-triangle-impulse-v1"
THREAD_COUNTS = (1, 8)
REPETITIONS_PER_THREAD_COUNT = 5

REQUIRED_PYTHON = (3, 12, 12)
REQUIRED_PACKAGES = {
    "numpy": "2.2.6",
    "safetensors": "0.5.3",
    "scipy": "1.15.3",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
}

SOURCE_FILES = {
    "README.md": "27d500b510a1cdc054a8ccbad484cf6062639fcb9d6b661714214fe766ed4e76",
    "hubconf.py": "d7e25603f67c329fe111b83c2a94fba8db48d4de1514e0014d2f54886f36a3cb",
    "requirements.txt": "053d043313aee66f53bdac14563a95e159174c27374388033e7ead4de6017d66",
    "redimnet2/__init__.py": "bd8dd7d51219ad582c20b1ac05ca0b6e7796e22b3eb32d2ab50ff2817340bdce",
    "redimnet2/redimnet2.py": "5013d784b5dce719572cf43d51c5f39ed61df89bdf623159fd15c74d5b55ba66",
    "redimnet2/layers/attention.py": "dcfb82870e66af8ca3c33dcf6d7d5c8aa5d6596d5f71f2543b8868308618f67f",
    "redimnet2/layers/blocks.py": "9edfd40b941e89e996090172bc0ecc0bccbf3ef71592e80e33cafd5b810c76a6",
    "redimnet2/layers/convnext.py": "beaf94aa2c3661951b0b1ef1e6d65f77901f35fa7e19bd30469eee716496c5e1",
    "redimnet2/layers/features.py": "5ef8dfb33c3330bddb3c65f693bfe0df597056a287ced99450550e4f5ba34979",
    "redimnet2/layers/features_tf.py": "bf5e3736c288b9582f2ff405ab6344237bc0b2ba647450f74b4002b3ce13d0a1",
    "redimnet2/layers/layernorm.py": "23900399027f6cae34b260e5778e076e0942ef886204b401221fe7c0c21bcabc",
    "redimnet2/layers/poolings.py": "f03849e5944e9fa6a7392b60cf3722e3e537a3e5cfd2988bd992c92ffc39a6a6",
    "redimnet2/layers/redim_structural.py": "93bae419bfdf11868c9e52dfb95578762b9c0a81e27a32ce930adcaefe457738",
    "redimnet2/layers/resblocks.py": "f79fbc2c7a6d88e9aa5efc4f5764ad137eafe1f5481b33ff7ed8811ee6a80778",
}

EXPECTED_TRUTH_SHAPES = {
    "waveform": (1, 48_000),
    "frontend": (1, 72, 299),
    "stem_1d": (1, 1_440, 296),
    "stage0_1d": (1, 1_440, 296),
    "stage3_1d": (1, 1_440, 296),
    "final_weighted_1d": (1, 1_440, 296),
    "backbone_2d": (1, 240, 6, 296),
    "attentive_pool": (1, 2_880),
    "pooled_bn": (1, 2_880),
    "raw_embedding": (1, 192),
    "l2_embedding": (1, 192),
}

# Frozen before native implementation. These are parity ceilings, not measured
# source floors and not permission to widen a future gate after seeing results.
PARITY_TOLERANCES = {
    "waveform": {"max_abs": 0.0, "relative_l2": 0.0},
    "frontend": {"max_abs": 5.0e-4, "relative_l2": 1.0e-5},
    "stem_1d": {"max_abs": 5.0e-4, "relative_l2": 5.0e-5},
    "stage0_1d": {"max_abs": 5.0e-4, "relative_l2": 5.0e-5},
    "stage3_1d": {"max_abs": 8.0e-4, "relative_l2": 8.0e-5},
    "final_weighted_1d": {"max_abs": 8.0e-4, "relative_l2": 8.0e-5},
    "backbone_2d": {"max_abs": 8.0e-4, "relative_l2": 8.0e-5},
    "attentive_pool": {"max_abs": 1.0e-3, "relative_l2": 1.0e-4},
    "pooled_bn": {"max_abs": 1.0e-3, "relative_l2": 1.0e-4},
    "raw_embedding": {"max_abs": 1.0e-3, "relative_l2": 1.0e-4},
    "l2_embedding": {"max_abs": 5.0e-5, "relative_l2": 5.0e-5},
}

TOLERANCE_CALIBRATION = {
    "frontend": {
        "initial_rejected_ceiling": {"max_abs": 2.0e-5, "relative_l2": 2.0e-6},
        "measured_source_floor": {
            "max_abs": 3.147125244140625e-4,
            "relative_l2": 5.86879853436786e-6,
        },
        "reason": "pre_native_cross_thread_source_floor_exceeded_provisional_ceiling",
    }
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_file_bytes(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    if path.is_symlink():
        raise RuntimeError("identity-bound input must not be a symlink")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
        raise RuntimeError("identity-bound input size or file type changed")
    value = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_mode,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_mode,
    )
    if before_identity != after_identity:
        raise RuntimeError("identity-bound input changed while it was read")
    if len(value) != expected_bytes or not hmac.compare_digest(
        _sha256_bytes(value), expected_sha256
    ):
        raise RuntimeError("identity-bound input digest changed")
    return value


def _verify_source_tree(source_root: Path) -> list[dict[str, Any]]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError("source root must be a direct directory")
    canonical_root = source_root.resolve(strict=True)
    if canonical_root != source_root.absolute():
        raise RuntimeError("source root must not contain indirect path components")
    manifest = []
    for relative, expected_sha256 in sorted(SOURCE_FILES.items()):
        source = canonical_root / relative
        if source.is_symlink() or source.resolve(strict=True) != source:
            raise RuntimeError("pinned source path is indirect")
        size = source.stat(follow_symlinks=False).st_size
        value = _stable_file_bytes(source, size, expected_sha256)
        manifest.append(
            {"bytes": len(value), "relative_path": relative, "sha256": expected_sha256}
        )
    return manifest


def _require_runtime() -> dict[str, str]:
    if sys.version_info[:3] != REQUIRED_PYTHON:
        raise RuntimeError("pinned ReDimNet2 export requires Python 3.12.12")
    observed = {
        package: importlib.metadata.version(package) for package in REQUIRED_PACKAGES
    }
    if observed != REQUIRED_PACKAGES:
        raise RuntimeError("pinned ReDimNet2 runtime package identity changed")
    return observed


def _verify_imported_source(source_root: Path) -> None:
    for relative in SOURCE_FILES:
        if not relative.endswith(".py") or relative == "hubconf.py":
            continue
        if relative.endswith("/__init__.py"):
            module_name = relative.removesuffix("/__init__.py").replace("/", ".")
        else:
            module_name = relative.removesuffix(".py").replace("/", ".")
        module = sys.modules.get(module_name)
        if module is None:
            raise RuntimeError(f"pinned source module was not imported: {module_name}")
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve(strict=True) != source_root / relative:
            raise RuntimeError(f"pinned source module resolved outside source root: {module_name}")


def _f32_tensor_sha256(tensor: torch_types.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validated_f32_tensor(name: str, tensor: torch_types.Tensor) -> torch_types.Tensor:
    import torch

    if tensor.dtype != torch.float32:
        raise RuntimeError(f"{name} is not f32")
    value = tensor.detach().cpu().contiguous()
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{name} contains a non-finite value")
    return value


def _build_deterministic_safetensors(
    tensors: dict[str, torch_types.Tensor],
) -> bytes:
    import torch
    from safetensors.torch import load

    header: dict[str, object] = {}
    offset = 0
    for name in sorted(tensors):
        tensor = _validated_f32_tensor(name, tensors[name])
        byte_length = tensor.numel() * 4
        header[name] = {
            "data_offsets": [offset, offset + byte_length],
            "dtype": "F32",
            "shape": list(tensor.shape),
        }
        offset += byte_length
    header_json = _canonical_json_bytes(header)
    header_json += b" " * (-len(header_json) % 8)

    serialized = io.BytesIO()
    serialized.write(struct.pack("<Q", len(header_json)))
    serialized.write(header_json)
    for name in sorted(tensors):
        array = tensors[name].numpy().astype("<f4", copy=False)
        serialized.write(array.tobytes(order="C"))
    value = serialized.getvalue()
    serialized.close()

    decoded = load(value)
    if sorted(decoded) != sorted(tensors):
        raise RuntimeError("serialized safetensors census changed")
    for name, expected in tensors.items():
        observed = decoded[name]
        if observed.dtype != torch.float32 or tuple(observed.shape) != tuple(expected.shape):
            raise RuntimeError("serialized safetensors type or shape changed")
        if not hmac.compare_digest(
            _f32_tensor_sha256(observed), _f32_tensor_sha256(expected)
        ):
            raise RuntimeError("serialized safetensors payload changed")
    return value


def _synthetic_waveform() -> torch_types.Tensor:
    import numpy
    import torch

    indices = numpy.arange(SYNTHETIC_SAMPLES, dtype=numpy.uint64)
    modular = ((indices * 48_271 + 1) % 2_147_483_647).astype(numpy.float64)
    modular = modular / 1_073_741_824.0 - 1.0
    triangle = numpy.abs((indices % 640).astype(numpy.int64) - 320).astype(
        numpy.float64
    )
    triangle = triangle / 320.0 - 0.5
    waveform = (0.07 * modular + 0.08 * triangle).astype(numpy.float32)
    waveform[::4_000] += numpy.float32(0.25)
    waveform = numpy.clip(waveform, -0.5, 0.5).astype(numpy.float32, copy=False)
    return torch.from_numpy(waveform).unsqueeze(0).contiguous()


def _capture_truth(
    model: torch_types.nn.Module, waveform: torch_types.Tensor
) -> dict[str, torch_types.Tensor]:
    import torch

    model.train(False)
    with torch.inference_mode():
        frontend = model.spec(waveform)
        backbone_input = frontend.unsqueeze(1)
        frames = backbone_input.shape[-1]
        retained_frames = (frames // model.backbone.time_stride) * model.backbone.time_stride
        backbone_input = backbone_input[..., :retained_frames]
        stem = model.backbone.stem(backbone_input)
        stage_outputs = []
        accumulated = [stem]
        for stage_index in range(model.backbone.num_stages):
            stage = model.backbone.run_stage(accumulated, stage_index)
            accumulated.append(stage)
            stage_outputs.append(stage)
        final_weighted = model.backbone.fin_wght1d(accumulated)
        backbone_2d = model.backbone.head(model.backbone.fin_to2d(final_weighted))
        pool_input = backbone_2d.reshape(
            backbone_2d.shape[0],
            backbone_2d.shape[1] * backbone_2d.shape[2],
            backbone_2d.shape[3],
        )
        attentive_pool = model.pool(pool_input)
        pooled_bn = model.bn(attentive_pool)
        raw_embedding = model.linear(pooled_bn)
        norm = torch.linalg.vector_norm(raw_embedding, ord=2, dim=1, keepdim=True)
        if not torch.isfinite(norm).all().item() or torch.any(norm <= 0).item():
            raise RuntimeError("raw embedding has no finite positive L2 norm")
        l2_embedding = raw_embedding / norm

    captured = {
        "attentive_pool": attentive_pool,
        "backbone_2d": backbone_2d,
        "final_weighted_1d": final_weighted,
        "frontend": frontend,
        "l2_embedding": l2_embedding,
        "pooled_bn": pooled_bn,
        "raw_embedding": raw_embedding,
        "stage0_1d": stage_outputs[0],
        "stage3_1d": stage_outputs[3],
        "stem_1d": stem,
        "waveform": waveform,
    }
    if set(captured) != set(EXPECTED_TRUTH_SHAPES):
        raise RuntimeError("synthetic truth seam census changed")
    validated = {}
    for name, expected_shape in EXPECTED_TRUTH_SHAPES.items():
        value = _validated_f32_tensor(name, captured[name])
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(f"synthetic truth shape changed for {name}")
        validated[name] = value
    return validated


def _drift_metrics(
    expected: torch_types.Tensor, observed: torch_types.Tensor
) -> dict[str, float]:
    import torch

    difference = observed.to(torch.float64) - expected.to(torch.float64)
    max_abs = float(torch.max(torch.abs(difference)).item())
    denominator = float(torch.linalg.vector_norm(expected.to(torch.float64)).item())
    numerator = float(torch.linalg.vector_norm(difference).item())
    relative_l2 = numerator / max(denominator, sys.float_info.min)
    if not (max_abs >= 0.0 and relative_l2 >= 0.0):
        raise RuntimeError("source nondeterminism floor is non-finite")
    return {"max_abs": max_abs, "relative_l2": relative_l2}


def _measure_source_floor(
    model: torch_types.nn.Module, waveform: torch_types.Tensor
) -> tuple[dict[str, torch_types.Tensor], list[dict[str, Any]], dict[str, dict[str, float]]]:
    import torch

    baseline = None
    runs = []
    maxima = {
        name: {"max_abs": 0.0, "relative_l2": 0.0}
        for name in EXPECTED_TRUTH_SHAPES
    }
    for thread_count in THREAD_COUNTS:
        torch.set_num_threads(thread_count)
        for repetition in range(REPETITIONS_PER_THREAD_COUNT):
            observed = _capture_truth(model, waveform)
            if baseline is None:
                baseline = observed
            seam_metrics = {}
            for name in sorted(observed):
                metrics = _drift_metrics(baseline[name], observed[name])
                seam_metrics[name] = metrics
                for metric, value in metrics.items():
                    maxima[name][metric] = max(maxima[name][metric], value)
            runs.append(
                {
                    "repetition": repetition,
                    "seams": seam_metrics,
                    "thread_count": thread_count,
                }
            )
    if baseline is None:
        raise RuntimeError("source nondeterminism floor produced no baseline")
    violations = {}
    for name, observed in maxima.items():
        tolerance = PARITY_TOLERANCES[name]
        if any(observed[metric] > tolerance[metric] for metric in tolerance):
            violations[name] = {"observed": observed, "tolerance": tolerance}
    if violations:
        raise RuntimeError(
            "source floor exceeds frozen parity tolerance: "
            + _canonical_json_bytes(violations).decode("ascii")
        )
    return baseline, runs, maxima


def _tensor_manifest(tensors: dict[str, torch_types.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "elements": tensor.numel(),
            "name": name,
            "sha256_f32le": _f32_tensor_sha256(tensor),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(tensors.items())
    ]


def _publish_exclusive(path: Path, value: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    if path.is_symlink():
        raise RuntimeError("published output unexpectedly became a symlink")
    observed = path.read_bytes()
    mode = path.stat(follow_symlinks=False).st_mode
    if stat.S_IMODE(mode) != 0o600 or not hmac.compare_digest(
        _sha256_bytes(observed), _sha256_bytes(value)
    ):
        raise RuntimeError("published output identity changed")


def _prepare_output_directory(output_dir: Path) -> Path:
    repo_root = Path(__file__).resolve(strict=True).parent.parent
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("output directory must not already exist")
    parent = output_dir.parent.resolve(strict=True)
    destination = parent / output_dir.name
    if destination == repo_root or repo_root in destination.parents:
        raise RuntimeError("ReDimNet2 artifacts must stay outside the repository")
    os.mkdir(destination, mode=0o700)
    os.chmod(destination, 0o700)
    return destination


def _require_external_input(path: Path, label: str) -> None:
    repo_root = Path(__file__).resolve(strict=True).parent.parent
    canonical = path.resolve(strict=True)
    if canonical == repo_root or repo_root in canonical.parents:
        raise RuntimeError(f"{label} must stay outside the repository")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    exporter_path = Path(__file__).resolve(strict=True)
    exporter_bytes = exporter_path.read_bytes()
    _require_external_input(args.checkpoint, "checkpoint")
    _require_external_input(args.source_root, "source root")
    source_manifest = _verify_source_tree(args.source_root)
    runtime = _require_runtime()
    checkpoint_bytes = _stable_file_bytes(
        args.checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256
    )

    import torch

    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model_config", "state_dict"}:
        raise RuntimeError("checkpoint top-level schema changed")
    config = checkpoint["model_config"]
    state = checkpoint["state_dict"]
    config_bytes = _canonical_json_bytes(config)
    if not hmac.compare_digest(_sha256_bytes(config_bytes), CONFIG_SHA256):
        raise RuntimeError("checkpoint model configuration changed")
    if not isinstance(state, dict) or len(state) != SOURCE_TENSORS:
        raise RuntimeError("checkpoint state tensor census changed")
    if sum(tensor.numel() for tensor in state.values()) != SOURCE_ELEMENTS:
        raise RuntimeError("checkpoint state element census changed")

    exported = {}
    dropped = []
    for name, tensor in sorted(state.items()):
        if name.endswith(".num_batches_tracked"):
            if tensor.dtype != torch.int64 or tensor.numel() != 1:
                raise RuntimeError("batch counter drop contract changed")
            dropped.append(name)
            continue
        exported[name] = _validated_f32_tensor(name, tensor)
    if (
        len(dropped) != DROPPED_BATCH_COUNTERS
        or len(exported) != EXPORTED_TENSORS
        or sum(tensor.numel() for tensor in exported.values()) != EXPORTED_ELEMENTS
    ):
        raise RuntimeError("exported tensor census changed")

    canonical_source_root = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(canonical_source_root))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from redimnet2.redimnet2 import ReDimNet2Wrap

            model = ReDimNet2Wrap(**config)
    finally:
        sys.path.pop(0)
    _verify_imported_source(canonical_source_root)
    load_result = model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError("checkpoint no longer loads strictly")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameters = total_parameters - trainable_parameters
    if (
        total_parameters != TOTAL_PARAMETERS
        or trainable_parameters != TRAINABLE_PARAMETERS
        or frozen_parameters != FROZEN_PARAMETERS
    ):
        raise RuntimeError("model parameter census changed")

    waveform = _synthetic_waveform()
    truth, floor_runs, floor_maxima = _measure_source_floor(model, waveform)
    package_bytes = _build_deterministic_safetensors(exported)
    truth_bytes = _build_deterministic_safetensors(truth)
    if exporter_path.read_bytes() != exporter_bytes:
        raise RuntimeError("exporter changed while conversion was running")
    if _verify_source_tree(args.source_root) != source_manifest:
        raise RuntimeError("source tree changed while conversion was running")

    receipt = {
        "checkpoint": {"bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA256},
        "config": {
            "canonical_json_sha256": CONFIG_SHA256,
            "value": config,
        },
        "conversion": {
            "dropped_batch_counter_names": dropped,
            "dropped_batch_counters": len(dropped),
            "exported_elements": EXPORTED_ELEMENTS,
            "exported_tensors": EXPORTED_TENSORS,
            "source_elements": SOURCE_ELEMENTS,
            "source_tensors": SOURCE_TENSORS,
            "tensor_manifest": _tensor_manifest(exported),
            "frozen_parameters": FROZEN_PARAMETERS,
            "total_parameters": TOTAL_PARAMETERS,
            "trainable_parameters": TRAINABLE_PARAMETERS,
        },
        "distribution": {
            "reason": "pinned_v1.0.0_tag_has_no_repository_license_file_and_model_weight_scope_is_not_explicit",
            "status": "operator_local_no_release",
        },
        "exporter": {
            "schema": EXPORTER_SCHEMA,
            "sha256": _sha256_bytes(exporter_bytes),
        },
        "model": {
            "id": MODEL_ID,
            "release": MODEL_RELEASE,
            "source_revision": MODEL_SOURCE_REVISION,
        },
        "network_access": False,
        "package": {
            "bytes": len(package_bytes),
            "format": "safetensors-f32-metadata-free",
            "sha256": _sha256_bytes(package_bytes),
        },
        "privacy": {
            "audio": False,
            "biometric_vectors_in_receipt": False,
            "human_speech": False,
            "local_paths": False,
            "speaker_identities": False,
            "transcripts": False,
        },
        "runtime": {"packages": runtime, "python": "3.12.12"},
        "schema": RECEIPT_SCHEMA,
        "source_files": source_manifest,
        "truth": {
            "fixture_id": SYNTHETIC_FIXTURE_ID,
            "package_bytes": len(truth_bytes),
            "package_sha256": _sha256_bytes(truth_bytes),
            "parity_tolerances": PARITY_TOLERANCES,
            "repetitions_per_thread_count": REPETITIONS_PER_THREAD_COUNT,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "schema": TRUTH_SCHEMA,
            "seam_manifest": _tensor_manifest(truth),
            "source_floor_maxima": floor_maxima,
            "source_floor_runs": floor_runs,
            "thread_counts": list(THREAD_COUNTS),
            "tolerance_calibration": TOLERANCE_CALIBRATION,
        },
    }
    receipt_bytes = _canonical_json_bytes(receipt) + b"\n"
    destination = _prepare_output_directory(args.output_dir)
    _publish_exclusive(destination / "redimnet2_b2_vox2_lm_f32.safetensors", package_bytes)
    _publish_exclusive(
        destination / "redimnet2_b2_vox2_lm_synthetic_truth.safetensors", truth_bytes
    )
    _publish_exclusive(destination / "conversion_receipt.json", receipt_bytes)
    directory_descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)

    print(
        json.dumps(
            {
                "package_bytes": len(package_bytes),
                "package_sha256": _sha256_bytes(package_bytes),
                "receipt_sha256": _sha256_bytes(receipt_bytes),
                "truth_bytes": len(truth_bytes),
                "truth_sha256": _sha256_bytes(truth_bytes),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
