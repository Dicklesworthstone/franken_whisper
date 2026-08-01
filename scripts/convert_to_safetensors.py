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
    Python 3.12.12
    numpy==2.2.6
    torch==2.7.1
    safetensors==0.5.3

Generating the full ECAPA oracle additionally requires exactly:
    torchaudio==2.7.1
    speechbrain==0.5.16

Usage:
    python3 convert_to_safetensors.py INPUT.ckpt OUTPUT.safetensors
        [--key KEY] [--profile PROFILE] [--full-oracle-output PATH]

    --key KEY  if the checkpoint is a dict wrapping the state dict under a key
               (e.g. "state_dict" / "model"), unwrap that key first.
    --profile   "generic" (default), or "ecapa-tdnn-voxceleb-v1" for the exact
                frozen SpeechBrain checkpoint and 200-tensor inference census.
    --full-oracle-output
                with the frozen ECAPA profile, also emit the transcript-free
                seven-stage public conformance oracle after exact hash checks.

On success the output path and its sha256 are printed (pin this in
fetch_aux_models.sh).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import io
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as numpy_types
    import torch as torch_types

ECAPA_PROFILE = "ecapa-tdnn-voxceleb-v1"
ECAPA_EXPORTER_VERSION = "franken-whisper-ecapa-export-v1"
ECAPA_FULL_ORACLE_EXPORTER_VERSION = "franken-whisper-ecapa-full-oracle-export-v1"
ECAPA_FULL_ORACLE_SCHEMA = "franken-whisper-ecapa-full-oracle-v1"
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_MODEL_REVISION = "eac27266f68caa806381260bd44ace38b136c76a"
ECAPA_TRAINING_CODE_REVISION = "aa0185408025e80f6c748d2c7af7fa96958c2231"
ECAPA_SOURCE_SHA256 = "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
ECAPA_SOURCE_BYTES = 83_316_686
ECAPA_SOURCE_TENSORS = 231
ECAPA_DROPPED_BATCH_COUNTERS = 31
ECAPA_EXPORTED_TENSORS = 200
ECAPA_PACKAGE_SHA256 = "9276a840c52cdd2e9afb73cd87a38e15749e12bf494d3ca47b5bc162f237cbcc"
ECAPA_CONTRACT_SHA256 = "9eb3e323aaa5550c87057996978d38ce57f9b280b829be6217440c8e63cef7a4"
ECAPA_GOLDEN_EVIDENCE_SHA256 = "073a910a2a8d171dca45e28940387ebfc0642e63224d62ebd62abe2b8efd9ac2"
ECAPA_FULL_ORACLE_SHA256 = "2c80806fbf68262ab1e0a1b52af18139f08272b7802fc3b0fd96011192dcf485"
ECAPA_FULL_ORACLE_BYTES = 2_160_320
ECAPA_FIXTURE_ID = "analytic-harmonic-chirp-impulse-v1"
ECAPA_FIXTURE_PCM_SHA256 = "acc240c07370020bbd1b3aaf9b8b81be43ef053b8da950969e86f62b6f1dba2f"
ECAPA_FIXTURE_SAMPLE_COUNT = 16_000
ECAPA_FULL_ORACLE_TENSORS = {
    "fbank_pre_normalization": (
        (1, 101, 80),
        "8fd529b6f2d3ec34d7b45bf39196ec8ebfb0c2b407d8b2e308717fe5bf8fcde8",
    ),
    "fbank_sentence_mean_normalized": (
        (1, 101, 80),
        "32afe9ace7c803c7e777e1d19ffe0630549f59da69c6593fe4aa4bff30cb5370",
    ),
    "initial_tdnn": (
        (1, 1_024, 101),
        "18274d7866b0181b17f9d3d58d0b585d9eb99ba7c9b8fabda6d3d7d23478d112",
    ),
    "first_se_res2": (
        (1, 1_024, 101),
        "b37629ffd2cca7c00533cd8f2baf23a22ce6b5b7348343c10b855cc37ef7bc24",
    ),
    "multi_feature_aggregation": (
        (1, 3_072, 101),
        "f8787f6f3fd0038d11feeb49b4e821993a9f4e890f518e03d890384e3ddbafb0",
    ),
    "attentive_pooling": (
        (1, 6_144, 1),
        "31261217b61f9519c6756330a8e9d6797626c49ece4fe2d4ee39f18c408e62b2",
    ),
    "embedding": (
        (1, 1, 192),
        "ff4b056c34a75e59ff51662faa22293cc7ef18785441d584b2b61dfd0b8cb5ae",
    ),
}
REQUIRED_TORCH_VERSION = "2.7.1"
REQUIRED_TORCHAUDIO_VERSION = "2.7.1"
REQUIRED_SAFETENSORS_VERSION = "0.5.3"
REQUIRED_NUMPY_VERSION = "2.2.6"
REQUIRED_SPEECHBRAIN_VERSION = "0.5.16"
REQUIRED_PYTHON_VERSION = (3, 12, 12)
REQUIRED_PYTHON_VERSION_TEXT = "3.12.12"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_exact_ecapa_source(path: Path) -> bytes:
    """Read the frozen checkpoint once into an exact, bounded owned buffer."""
    source_bytes = bytearray()
    maximum_bytes = ECAPA_SOURCE_BYTES + 1
    with path.open("rb") as source:
        while len(source_bytes) < maximum_bytes:
            chunk = source.read(min(1 << 20, maximum_bytes - len(source_bytes)))
            if not chunk:
                break
            source_bytes.extend(chunk)
    if len(source_bytes) != ECAPA_SOURCE_BYTES:
        raise RuntimeError(
            f"{ECAPA_PROFILE} input size mismatch "
            f"(got {len(source_bytes)}, want {ECAPA_SOURCE_BYTES})"
        )
    return bytes(source_bytes)


def _build_deterministic_safetensors(
    tensors: dict[str, torch_types.Tensor],
    metadata: dict[str, str],
) -> bytes:
    """Build canonical F32 safetensors and validate the complete byte stream."""
    import torch
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

    serialized = bytearray(struct.pack("<Q", len(header_json)))
    serialized.extend(header_json)
    for name in sorted(tensors):
        array = tensors[name].numpy().astype("<f4", copy=False)
        serialized.extend(array.tobytes(order="C"))
    file_bytes = bytes(serialized)

    # Exercise an independent safetensors parser before any output is
    # published. The Rust runtime repeats stricter model-specific checks.
    package = load(file_bytes)
    if sorted(package) != sorted(tensors):
        raise RuntimeError("serialized safetensors tensor census changed")
    for name, expected in tensors.items():
        observed = package[name]
        if expected.dtype != torch.float32 or observed.dtype != torch.float32:
            raise RuntimeError("serialized safetensors tensor dtype changed")
        if tuple(observed.shape) != tuple(expected.shape):
            raise RuntimeError("serialized safetensors tensor shape changed")
        if not hmac.compare_digest(
            _f32_tensor_sha256(observed),
            _f32_tensor_sha256(expected),
        ):
            raise RuntimeError("serialized safetensors tensor payload changed")
    try:
        decoded_header = json.loads(file_bytes[8 : 8 + len(header_json)])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("serialized safetensors header could not be decoded") from exc
    if decoded_header.get("__metadata__") != metadata:
        raise RuntimeError("serialized safetensors metadata changed")
    return file_bytes


def _publish_new_file(output: Path, file_bytes: bytes) -> None:
    """Atomically publish validated bytes without replacing an existing path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary_path = Path(temporary_name)
    primary_error = None
    published = False
    try:
        try:
            destination = os.fdopen(temporary_fd, "w+b")
        except BaseException:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
            raise
        with destination:
            written = destination.write(file_bytes)
            if written != len(file_bytes):
                raise OSError("short safetensors write")
            destination.flush()
            os.fsync(destination.fileno())
            destination.seek(0)
            written_hasher = hashlib.sha256()
            for chunk in iter(lambda: destination.read(1 << 20), b""):
                written_hasher.update(chunk)
            if not hmac.compare_digest(
                written_hasher.hexdigest(),
                hashlib.sha256(file_bytes).hexdigest(),
            ):
                raise OSError("written safetensors checksum changed")
        # A same-directory hard link installs the verified inode atomically and
        # fails rather than replacing a path created by another process.
        os.link(temporary_path, output)
        published = True
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            if primary_error is None and published:
                print(
                    "warning: output published but temporary-file cleanup failed",
                    file=sys.stderr,
                )
            elif primary_error is None:
                raise


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _f32_tensor_sha256(tensor: torch_types.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _analytic_ecapa_fixture(numpy: object) -> numpy_types.ndarray:
    values = []
    for index in range(ECAPA_FIXTURE_SAMPLE_COUNT):
        time = index / ECAPA_FIXTURE_SAMPLE_COUNT
        chirp_phase = 2.0 * math.pi * (120.0 * time + 180.0 * time * time)
        value = 0.22 * math.sin(2.0 * math.pi * 173.0 * time)
        value += 0.11 * math.sin(2.0 * math.pi * 347.0 * time)
        value += 0.07 * math.sin(chirp_phase)
        if index == 1_234:
            value += 0.5
        values.append(value)
    fixture = numpy.asarray(values, dtype=numpy.float32)
    fixture_bytes = fixture.astype("<f4", copy=False).tobytes(order="C")
    if not hmac.compare_digest(
        hashlib.sha256(fixture_bytes).hexdigest(),
        ECAPA_FIXTURE_PCM_SHA256,
    ):
        raise RuntimeError("analytic ECAPA fixture does not match its frozen identity")
    return fixture


def _build_ecapa_full_oracle(
    state_dict: dict[str, torch_types.Tensor],
    numpy: object,
    torch: object,
) -> tuple[dict[str, torch_types.Tensor], dict[str, str]]:
    try:
        import torchaudio
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization
    except ImportError as exc:
        raise RuntimeError(
            "full ECAPA oracle generation requires speechbrain and torchaudio"
        ) from exc

    try:
        speechbrain_version = importlib.metadata.version("speechbrain")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "full ECAPA oracle requires an installed speechbrain distribution"
        ) from exc
    torchaudio_version = _base_version(torchaudio.__version__)
    if speechbrain_version != REQUIRED_SPEECHBRAIN_VERSION:
        raise RuntimeError(
            f"full ECAPA oracle requires speechbrain=={REQUIRED_SPEECHBRAIN_VERSION} "
            f"(got {speechbrain_version})"
        )
    if torchaudio_version != REQUIRED_TORCHAUDIO_VERSION:
        raise RuntimeError(
            f"full ECAPA oracle requires torchaudio=={REQUIRED_TORCHAUDIO_VERSION} "
            f"(got {torchaudio.__version__})"
        )

    model = ECAPA_TDNN(
        input_size=80,
        channels=[1_024, 1_024, 1_024, 1_024, 3_072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
    )
    model.load_state_dict(state_dict, strict=True)
    model.train(False)

    captured: dict[str, torch_types.Tensor] = {}

    def capture(name: str):
        def hook(_module: object, _inputs: object, output: torch_types.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"ECAPA stage {name} did not return one tensor")
            captured[name] = (
                output.detach().cpu().to(torch.float32).contiguous().clone()
            )

        return hook

    handles = [
        model.blocks[0].register_forward_hook(capture("initial_tdnn")),
        model.blocks[1].register_forward_hook(capture("first_se_res2")),
        model.mfa.register_forward_hook(capture("multi_feature_aggregation")),
        model.asp.register_forward_hook(capture("attentive_pooling")),
    ]
    try:
        fixture = _analytic_ecapa_fixture(numpy)
        waveform = torch.from_numpy(fixture.copy()).reshape(1, -1)
        feature_extractor = Fbank(n_mels=80)
        normalizer = InputNormalization(norm_type="sentence", std_norm=False)
        feature_extractor.train(False)
        normalizer.train(False)
        with torch.no_grad():
            pre_normalization = feature_extractor(waveform)
            lengths = torch.ones(
                waveform.shape[0],
                dtype=waveform.dtype,
                device=waveform.device,
            )
            normalized = normalizer(pre_normalization.clone(), lengths)
            embedding = model(normalized, lengths=lengths)
    finally:
        for handle in handles:
            handle.remove()

    captured["fbank_pre_normalization"] = (
        pre_normalization.detach().cpu().to(torch.float32).contiguous()
    )
    captured["fbank_sentence_mean_normalized"] = (
        normalized.detach().cpu().to(torch.float32).contiguous()
    )
    captured["embedding"] = embedding.detach().cpu().to(torch.float32).contiguous()

    if set(captured) != set(ECAPA_FULL_ORACLE_TENSORS):
        raise RuntimeError("full ECAPA oracle stage census changed")
    for name, (expected_shape, expected_sha256) in ECAPA_FULL_ORACLE_TENSORS.items():
        tensor = captured[name]
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"full ECAPA oracle stage {name} shape changed: "
                f"got {tuple(tensor.shape)}, want {expected_shape}"
            )
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"full ECAPA oracle stage {name} is not finite F32")
        observed_sha256 = _f32_tensor_sha256(tensor)
        if not hmac.compare_digest(observed_sha256, expected_sha256):
            raise RuntimeError(
                f"full ECAPA oracle stage {name} hash mismatch: "
                f"got {observed_sha256}, want {expected_sha256}"
            )

    metadata = {
        "canonical_layout": "speechbrain_cpu_contiguous_c_order",
        "contract_sha256": ECAPA_CONTRACT_SHA256,
        "device": "cpu",
        "evaluation_mode": "true",
        "exported_dtype": "F32",
        "exported_tensor_count": str(len(ECAPA_FULL_ORACLE_TENSORS)),
        "exporter_version": ECAPA_FULL_ORACLE_EXPORTER_VERSION,
        "fixture_id": ECAPA_FIXTURE_ID,
        "fixture_pcm_sha256": ECAPA_FIXTURE_PCM_SHA256,
        "fixture_sample_count": str(ECAPA_FIXTURE_SAMPLE_COUNT),
        "generator": "franken_whisper/scripts/convert_to_safetensors.py",
        "golden_evidence_sha256": ECAPA_GOLDEN_EVIDENCE_SHA256,
        "numpy_version": REQUIRED_NUMPY_VERSION,
        "python_version": REQUIRED_PYTHON_VERSION_TEXT,
        "safetensors_version": REQUIRED_SAFETENSORS_VERSION,
        "schema_version": ECAPA_FULL_ORACLE_SCHEMA,
        "source_checkpoint_sha256": ECAPA_SOURCE_SHA256,
        "source_model_id": ECAPA_MODEL_ID,
        "source_model_revision": ECAPA_MODEL_REVISION,
        "source_weight_package_sha256": ECAPA_PACKAGE_SHA256,
        "speechbrain_version": speechbrain_version,
        "torch_version": REQUIRED_TORCH_VERSION,
        "torchaudio_version": torchaudio_version,
        "training_code_revision": ECAPA_TRAINING_CODE_REVISION,
    }
    return captured, metadata


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
    parser.add_argument(
        "--full-oracle-output",
        type=Path,
        default=None,
        help="also write the frozen public ECAPA full-stage oracle safetensors",
    )
    args = parser.parse_args()

    if (
        args.profile == ECAPA_PROFILE
        and sys.version_info[:3] != REQUIRED_PYTHON_VERSION
    ):
        print(
            f"error: {ECAPA_PROFILE} requires Python {REQUIRED_PYTHON_VERSION_TEXT} "
            f"(got {sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro})",
            file=sys.stderr,
        )
        return 2

    try:
        import numpy
        import safetensors
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(
            f"error: missing dependency ({exc}); pip install numpy torch safetensors",
            file=sys.stderr,
        )
        return 2

    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if args.output.exists():
        print(f"error: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    if args.full_oracle_output is not None:
        if args.profile != ECAPA_PROFILE:
            print("error: --full-oracle-output requires the frozen ECAPA profile", file=sys.stderr)
            return 2
        if args.full_oracle_output.exists():
            print(
                f"error: refusing to overwrite existing full oracle: {args.full_oracle_output}",
                file=sys.stderr,
            )
            return 2
        weight_output = args.output.resolve()
        oracle_output = args.full_oracle_output.resolve()
        if (
            weight_output == oracle_output
            or weight_output in oracle_output.parents
            or oracle_output in weight_output.parents
        ):
            print(
                "error: weight and full-oracle output paths must neither match nor contain one another",
                file=sys.stderr,
            )
            return 2

    ecapa_source_bytes = None
    if args.profile == ECAPA_PROFILE:
        try:
            ecapa_source_bytes = _read_exact_ecapa_source(args.input)
        except (OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        input_sha = hashlib.sha256(ecapa_source_bytes).hexdigest()
    else:
        input_sha = _sha256(args.input)
    if args.profile == ECAPA_PROFILE:
        if args.key is not None:
            print(f"error: {ECAPA_PROFILE} does not accept --key", file=sys.stderr)
            return 2
        if not hmac.compare_digest(input_sha, ECAPA_SOURCE_SHA256):
            print(
                f"error: {ECAPA_PROFILE} input sha256 mismatch "
                f"(got {input_sha}, want {ECAPA_SOURCE_SHA256})",
                file=sys.stderr,
            )
            return 2
        torch_version = _base_version(torch.__version__)
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
    load_source = (
        io.BytesIO(ecapa_source_bytes)
        if ecapa_source_bytes is not None
        else args.input
    )
    obj = torch.load(load_source, map_location="cpu", weights_only=True)
    load_source = None
    ecapa_source_bytes = None
    if args.key is not None:
        obj = obj[args.key]
    if not isinstance(obj, dict):
        print(f"error: checkpoint is not a state dict (got {type(obj).__name__})", file=sys.stderr)
        return 2

    tensors: dict[str, torch_types.Tensor] = {}
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
            "numpy_version": numpy.__version__,
            "torch_version": torch.__version__,
            "safetensors_version": safetensors.__version__,
        }

    oracle_tensors = None
    oracle_metadata = None
    try:
        if args.full_oracle_output is not None:
            oracle_tensors, oracle_metadata = _build_ecapa_full_oracle(obj, numpy, torch)
        weight_bytes = _build_deterministic_safetensors(tensors, metadata)
        out_sha = hashlib.sha256(weight_bytes).hexdigest()
        if args.profile == ECAPA_PROFILE and not hmac.compare_digest(
            out_sha,
            ECAPA_PACKAGE_SHA256,
        ):
            raise RuntimeError(
                "frozen ECAPA weight output sha256 mismatch "
                f"(got {out_sha}, want {ECAPA_PACKAGE_SHA256})"
            )
        oracle_bytes = None
        oracle_sha = None
        if args.full_oracle_output is not None:
            if oracle_tensors is None or oracle_metadata is None:
                raise RuntimeError("full ECAPA oracle was not constructed")
            oracle_bytes = _build_deterministic_safetensors(
                oracle_tensors,
                oracle_metadata,
            )
            oracle_sha = hashlib.sha256(oracle_bytes).hexdigest()
            if (
                len(oracle_bytes) != ECAPA_FULL_ORACLE_BYTES
                or not hmac.compare_digest(oracle_sha, ECAPA_FULL_ORACLE_SHA256)
            ):
                raise RuntimeError(
                    "full ECAPA oracle identity mismatch "
                    f"(got {len(oracle_bytes)} bytes and {oracle_sha})"
                )

        # All tensor, metadata, and frozen-identity checks complete before the
        # first final path is opened. Final paths remain exclusive-create.
        publication_paths = [args.output]
        if args.full_oracle_output is not None:
            publication_paths.append(args.full_oracle_output)
        for publication_path in publication_paths:
            publication_path.parent.mkdir(parents=True, exist_ok=True)
        _publish_new_file(args.output, weight_bytes)
        if args.full_oracle_output is not None and oracle_bytes is not None:
            _publish_new_file(args.full_oracle_output, oracle_bytes)
    except (OSError, RuntimeError, TypeError, ValueError, safetensors.SafetensorError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {len(tensors)} tensors -> {args.output}")
    print(f"input  sha256: {input_sha}")
    print(f"output sha256: {out_sha}")
    if args.full_oracle_output is not None:
        if oracle_tensors is None or oracle_bytes is None or oracle_sha is None:
            print("error: full ECAPA oracle publication state is invalid", file=sys.stderr)
            return 2
        print(
            f"wrote {len(oracle_tensors)} tensors -> {args.full_oracle_output}"
        )
        print(f"oracle bytes: {len(oracle_bytes)}")
        print(f"oracle sha256: {oracle_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
