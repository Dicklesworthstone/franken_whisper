#!/usr/bin/env python3
"""Convert a pinned PyTorch/NeMo checkpoint to deterministic safetensors.

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

The frozen Streaming Sortformer profile requires exactly the runtime recorded
in ``SORTFORMER_REQUIRED_PACKAGES`` below. It reads the two members of the
identity-bound ``.nemo`` archive from one owned, hash-verified byte buffer,
instantiates the pinned NeMo graph without ``restore_from`` temporary files,
and emits both a metadata-free package and a canonical conversion receipt.

Usage:
    python3 convert_to_safetensors.py INPUT.ckpt OUTPUT.safetensors
        [--key KEY] [--profile PROFILE] [--full-oracle-output PATH]
        [--receipt-output PATH]

    --key KEY  if the checkpoint is a dict wrapping the state dict under a key
               (e.g. "state_dict" / "model"), unwrap that key first.
    --profile   "generic" (default), or "ecapa-tdnn-voxceleb-v1" for the exact
                frozen SpeechBrain checkpoint and 200-tensor inference census.
    --full-oracle-output
                with the frozen ECAPA profile, also emit the transcript-free
                seven-stage public conformance oracle after exact hash checks.
    --receipt-output
                required by the frozen Streaming Sortformer profile; emit the
                canonical source-to-destination tensor receipt at this path.

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
import stat
import struct
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

SORTFORMER_PROFILE = "sortformer-streaming-4spk-v2.1"
SORTFORMER_RECEIPT_SCHEMA = "franken-whisper-sortformer-conversion-receipt-v1"
SORTFORMER_TENSOR_MANIFEST_SCHEMA = "franken-whisper-sortformer-tensor-manifest-v1"
SORTFORMER_CONVERTER_ID = "franken-whisper-native-sortformer-converter"
SORTFORMER_CONVERTER_VERSION = "1"
SORTFORMER_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
SORTFORMER_MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
SORTFORMER_NEMO_BYTES = 471_367_680
SORTFORMER_NEMO_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
SORTFORMER_CONFIG_BYTES = 3_567
SORTFORMER_CONFIG_SHA256 = "2865d469c4d2aac54aa5b8a956b2423c053806dd20d5bf5d08675942a1acface"
SORTFORMER_CHECKPOINT_BYTES = 471_352_898
SORTFORMER_CHECKPOINT_SHA256 = "eca9773c2dab91dd41fbaa4473cebb9d00811d67788ce2de609dadc6e499cdf4"
SORTFORMER_STATE_INVENTORY_SHA256 = "f4f219cf4ac6f755247b56d19e425db3d6a7c23c4509176549b363b63abdf532"
SORTFORMER_NEMO_SOURCE_REVISION = "40ace43c7cf151af78dc22027c02feeca7e06b6a"
SORTFORMER_EXTERNAL_CONTRACT_SHA256 = "7ac048e3372fe4c622840beddfbeef42944d961408360324cb7276a69c8542c5"
SORTFORMER_RUNTIME_FINGERPRINT_SHA256 = "3713fd3f024c1cef7d860706baf0dbaaf18058c03c26331da6254687693d564c"
SORTFORMER_ORACLE_ADAPTER_SHA256 = "8f376c979b7eaca41dc0a438d9aaa41c1c723052b97c45eb2acc59b6d6f00bde"
SORTFORMER_PARAMETER_TENSORS = 937
SORTFORMER_TRAINABLE_PARAMETERS = 117_693_960
SORTFORMER_STATE_TENSORS = 990
SORTFORMER_STATE_ELEMENTS = 117_744_681
SORTFORMER_STATE_F32_TENSORS = 973
SORTFORMER_STATE_F32_ELEMENTS = 117_744_664
SORTFORMER_STATE_F32_BYTES = 470_978_656
SORTFORMER_STATE_I64_TENSORS = 17
SORTFORMER_STATE_PAYLOAD_BYTES = 470_978_792
SORTFORMER_SOURCE_RECORDS = 992
SORTFORMER_EXPORTED_TENSORS = 974
SORTFORMER_DROPPED_TENSORS = 18
SORTFORMER_TENSOR_MANIFEST_SHA256 = (
    "2c32b0b9e48bb296e66615b038827d0fdde4b4fda2ce044a6c30cd317456c8d7"
)
SORTFORMER_PACKAGE_F32_ELEMENTS = 122_864_152
SORTFORMER_PACKAGE_PAYLOAD_BYTES = 491_456_608
SORTFORMER_PACKAGE_BYTES = 491_570_584
SORTFORMER_PACKAGE_SHA256 = (
    "487fa30cb0aa9799c77bd9985e6787962c3991fab8d4d576a4f1221d45298f6a"
)
SORTFORMER_POSITION_TENSOR = "encoder.pos_enc.pe"
SORTFORMER_DTYPE_SENTINEL = "preprocessor.dtype_sentinel_tensor"
SORTFORMER_SOURCE_LAYOUT = "pytorch_contiguous_row_major"
SORTFORMER_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SORTFORMER_REQUIRED_PACKAGES = {
    "nemo-toolkit": "3.1.0+40ace43c7c",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "numpy": "2.4.6",
    "safetensors": "0.8.0",
    "librosa": "0.11.0",
    "lhotse": "1.33.0",
    "soundfile": "0.14.0",
    "scipy": "1.18.0",
    "omegaconf": "2.3.0",
    "hydra-core": "1.3.2",
    "lightning": "2.4.0",
}
SORTFORMER_SOURCE_FILES = (
    ("nemo/collections/asr/data/audio_to_diar_label.py", "f9b0d23bd52da417ac18418ea1c83aa1119f59e6b37d3b2b3159c8cb2f036234"),
    ("nemo/collections/asr/models/sortformer_diar_models.py", "4978dba1a02b414893123f66905a1e523d5bb65766903269b325746c67f6920a"),
    ("nemo/collections/asr/modules/audio_preprocessing.py", "c061f521e14978d22ad57fa5ddf08f1103c2d1f1a4e01aca6698bfad007e8e7c"),
    ("nemo/collections/asr/modules/conformer_encoder.py", "a8b6f712cdf75a3be768848e8242ea9412ca7ff31ba2dda6b9602bcefc627cec"),
    ("nemo/collections/asr/modules/sortformer_modules.py", "3d136c245e3bf7a88c47fdd2eae1edb9189bbeddc3ff779cb5679a29d890b7eb"),
    ("nemo/collections/asr/modules/transformer/transformer_encoders.py", "a2859c86c8389f1954d5c8be04dc2bc422452517ef15e069cf42bfab5d304759"),
    ("nemo/collections/asr/modules/transformer/transformer_modules.py", "2564d95365cfafd486b1a3d10e2e2f438702907076f3716dd4c42d568b3bcc72"),
    ("nemo/collections/asr/parts/mixins/diarization.py", "5365e416ecab192cf59f1b9d6554ebce0ed3bdb2fee7575966ac1e3fca1a1408"),
    ("nemo/collections/asr/parts/preprocessing/features.py", "4290ed2d697362a68a6158fb8b7b8d1e2306b223b83172c63fc6b5d31b28ee69"),
    ("nemo/collections/asr/parts/preprocessing/segment.py", "a598d91b94110e0c12a1ba4a57894ce89109e597fa8e909cf7b5b6e7bb9369af"),
    ("nemo/collections/asr/parts/submodules/causal_convs.py", "7cf505c8caef44a37a7dec10b51eb2d60ec2f1efc3a2badc3c20c37e427cbd42"),
    ("nemo/collections/asr/parts/submodules/conformer_modules.py", "99bb846c51db028d6d30b3d844af22826068aeaa0e48eb586489a31a9cbacf9d"),
    ("nemo/collections/asr/parts/submodules/multi_head_attention.py", "4999fd0d679fd7315ba275f7311fe6608c48e492bd337f2e220c99b8b9729c69"),
    ("nemo/collections/asr/parts/submodules/subsampling.py", "4fbc689f3f66e4630b286196315a02b315ad53e8049c164fe40dd11168cf0834"),
    ("nemo/collections/asr/parts/utils/speaker_utils.py", "6c247bdda26fd010190e1c96f8399f77a5265a180086e134d9b167b3c8019dc0"),
    ("nemo/collections/asr/parts/utils/vad_utils.py", "7beb57efff5e08407f9f16afe9c0da7d0e2ddb9bd62e2a37424693e48c5f0437"),
    ("nemo/collections/common/parts/transformer_utils.py", "47f5e337230e7b4e176877f01c2ae85f75c024942dc567f27d8429c3e60e67c0"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_file_identity(path: Path) -> tuple[int, int, int, int, int, int, str]:
    """Hash one regular file and bind the digest to its stable open inode."""
    with path.open("rb") as source:
        initial = os.fstat(source.fileno())
        hasher = hashlib.sha256()
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
        final = os.fstat(source.fileno())
    path_final = path.stat()
    fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "st_mode",
    )
    initial_fields = tuple(getattr(initial, field) for field in fields)
    if (
        tuple(getattr(final, field) for field in fields) != initial_fields
        or tuple(getattr(path_final, field) for field in fields) != initial_fields
    ):
        raise RuntimeError("file identity changed while it was hashed")
    return (*initial_fields, hasher.hexdigest())


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


def _read_exact_member(archive: tarfile.TarFile, name: str, expected_bytes: int) -> bytes:
    members = [member for member in archive.getmembers() if member.name == name]
    if len(members) != 1:
        raise RuntimeError(f"{SORTFORMER_PROFILE} archive member census is invalid")
    member = members[0]
    if not member.isfile() or member.size != expected_bytes:
        raise RuntimeError(f"{SORTFORMER_PROFILE} archive member identity is invalid")
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"{SORTFORMER_PROFILE} archive member is unreadable")
    payload = source.read(expected_bytes + 1)
    if len(payload) != expected_bytes:
        raise RuntimeError(f"{SORTFORMER_PROFILE} archive member size changed")
    return payload


def _read_exact_sortformer_archive(path: Path) -> tuple[bytes, bytes]:
    """Authenticate one owned archive byte stream and retain only its members."""
    archive_buffer = io.BytesIO()
    with path.open("rb") as source:
        initial = os.fstat(source.fileno())
        if initial.st_size != SORTFORMER_NEMO_BYTES:
            raise RuntimeError(
                f"{SORTFORMER_PROFILE} input size mismatch "
                f"(got {initial.st_size}, want {SORTFORMER_NEMO_BYTES})"
            )
        archive_bytes = 0
        maximum_bytes = SORTFORMER_NEMO_BYTES + 1
        while archive_bytes < maximum_bytes:
            chunk = source.read(min(1 << 20, maximum_bytes - archive_bytes))
            if not chunk:
                break
            archive_buffer.write(chunk)
            archive_bytes += len(chunk)
        if archive_bytes != SORTFORMER_NEMO_BYTES:
            raise RuntimeError(f"{SORTFORMER_PROFILE} input size changed during read")
        archive_view = archive_buffer.getbuffer()
        try:
            if not hmac.compare_digest(
                hashlib.sha256(archive_view).hexdigest(), SORTFORMER_NEMO_SHA256
            ):
                raise RuntimeError(f"{SORTFORMER_PROFILE} input sha256 mismatch")
        finally:
            archive_view.release()
        final = os.fstat(source.fileno())
        if (
            final.st_dev != initial.st_dev
            or final.st_ino != initial.st_ino
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
        ):
            raise RuntimeError(f"{SORTFORMER_PROFILE} input inode changed during read")
    archive_buffer.seek(0)
    with tarfile.open(fileobj=archive_buffer, mode="r:*") as archive:
        if sorted(member.name for member in archive.getmembers()) != [
            "model_config.yaml",
            "model_weights.ckpt",
        ]:
            raise RuntimeError(f"{SORTFORMER_PROFILE} archive member set is not frozen")
        config_bytes = _read_exact_member(
            archive, "model_config.yaml", SORTFORMER_CONFIG_BYTES
        )
        checkpoint_bytes = _read_exact_member(
            archive, "model_weights.ckpt", SORTFORMER_CHECKPOINT_BYTES
        )
    archive_buffer.close()
    if not hmac.compare_digest(
        hashlib.sha256(config_bytes).hexdigest(), SORTFORMER_CONFIG_SHA256
    ):
        raise RuntimeError(f"{SORTFORMER_PROFILE} config sha256 mismatch")
    if not hmac.compare_digest(
        hashlib.sha256(checkpoint_bytes).hexdigest(), SORTFORMER_CHECKPOINT_SHA256
    ):
        raise RuntimeError(f"{SORTFORMER_PROFILE} checkpoint sha256 mismatch")
    return config_bytes, checkpoint_bytes


def _build_deterministic_safetensors(
    tensors: dict[str, torch_types.Tensor],
    metadata: dict[str, str] | None,
) -> bytes:
    """Build canonical F32 safetensors and validate the complete byte stream."""
    import torch
    from safetensors.torch import load

    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
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

    # BytesIO owns one mutable backing store and hands that store to the
    # immutable result on CPython, avoiding the prior bytearray-to-bytes peak.
    serialized = io.BytesIO()
    serialized.write(struct.pack("<Q", len(header_json)))
    serialized.write(header_json)
    for name in sorted(tensors):
        array = tensors[name].numpy().astype("<f4", copy=False)
        serialized.write(array.tobytes(order="C"))
    file_bytes = serialized.getvalue()
    serialized.close()

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
    if len(file_bytes) < 8:
        raise RuntimeError("serialized safetensors header is absent")
    header_bytes = struct.unpack("<Q", file_bytes[:8])[0]
    try:
        decoded_header = json.loads(file_bytes[8 : 8 + header_bytes])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("serialized safetensors header could not be decoded") from exc
    if metadata is None:
        if "__metadata__" in decoded_header:
            raise RuntimeError("serialized safetensors unexpectedly contains metadata")
    elif decoded_header.get("__metadata__") != metadata:
        raise RuntimeError("serialized safetensors metadata changed")
    return file_bytes


def _require_existing_output(output: Path, expected_bytes: bytes) -> None:
    """Accept only an owner-private, byte-identical retry artifact."""
    if output.is_symlink() or not output.is_file():
        raise OSError("existing output is not a regular file")
    identity = _stable_file_identity(output)
    if identity[2] != len(expected_bytes) or not hmac.compare_digest(
        identity[-1], hashlib.sha256(expected_bytes).hexdigest()
    ):
        raise OSError("existing output identity does not match this conversion")
    if not stat.S_ISREG(identity[-2]) or stat.S_IMODE(identity[-2]) != 0o600:
        raise OSError("existing output permissions are not owner-only mode 0600")


def _publish_new_file(output: Path, file_bytes: bytes) -> None:
    """Publish or reuse exact validated bytes at an owner-only final path."""
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            output,
            os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError:
        _require_existing_output(output, file_bytes)
        return

    with os.fdopen(descriptor, "w+b") as destination:
        os.fchmod(destination.fileno(), 0o600)
        view = memoryview(file_bytes)
        try:
            written = 0
            while written < len(view):
                count = destination.write(view[written:])
                if count is None or count <= 0:
                    raise OSError("short exclusive output write")
                written += count
        finally:
            view.release()
        destination.flush()
        os.fsync(destination.fileno())
        destination.seek(0)
        written_hasher = hashlib.sha256()
        for chunk in iter(lambda: destination.read(1 << 20), b""):
            written_hasher.update(chunk)
        if not hmac.compare_digest(
            written_hasher.hexdigest(), hashlib.sha256(file_bytes).hexdigest()
        ):
            raise OSError("written output checksum changed")
    _require_existing_output(output, file_bytes)


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _f32_tensor_sha256(tensor: torch_types.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sortformer_source_identities(
    source_root: Path,
) -> tuple[tuple[str, int, int, int, int, int, int, str], ...]:
    """Bind every selected NeMo source path to stable, expected bytes."""
    canonical_root = source_root.resolve(strict=True)
    identities = []
    for relative, expected_sha256 in SORTFORMER_SOURCE_FILES:
        source_path = canonical_root / relative
        if source_path.resolve(strict=True) != source_path:
            raise RuntimeError(f"{SORTFORMER_PROFILE} NeMo source path is indirect")
        identity = _stable_file_identity(source_path)
        if not hmac.compare_digest(identity[-1], expected_sha256):
            raise RuntimeError(f"{SORTFORMER_PROFILE} NeMo source identity mismatch")
        identities.append((relative, *identity))
    return tuple(identities)


def _require_sortformer_runtime() -> tuple[
    dict[str, str],
    Path,
    tuple[tuple[str, int, int, int, int, int, int, str], ...],
]:
    if sys.version_info[:3] != REQUIRED_PYTHON_VERSION:
        raise RuntimeError(
            f"{SORTFORMER_PROFILE} requires Python {REQUIRED_PYTHON_VERSION_TEXT}"
        )
    observed = {
        package: importlib.metadata.version(package)
        for package in SORTFORMER_REQUIRED_PACKAGES
    }
    if observed != SORTFORMER_REQUIRED_PACKAGES:
        raise RuntimeError(f"{SORTFORMER_PROFILE} runtime package identity mismatch")

    distribution = importlib.metadata.distribution("nemo-toolkit")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(f"{SORTFORMER_PROFILE} NeMo direct-url identity is absent")
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict) or any(
        vcs_info.get(field) != expected
        for field, expected in (
            ("commit_id", SORTFORMER_NEMO_SOURCE_REVISION),
            ("requested_revision", SORTFORMER_NEMO_SOURCE_REVISION),
            ("vcs", "git"),
        )
    ):
        raise RuntimeError(f"{SORTFORMER_PROFILE} NeMo source revision mismatch")

    import nemo

    nemo_root = Path(nemo.__file__).resolve().parent
    source_root = nemo_root.parent
    source_identities = _sortformer_source_identities(source_root)
    if [identity[0] for identity in source_identities] != sorted(
        identity[0] for identity in source_identities
    ):
        raise RuntimeError(f"{SORTFORMER_PROFILE} source identities are not canonical")

    runtime = {
        "python": REQUIRED_PYTHON_VERSION_TEXT,
        "nemo": observed["nemo-toolkit"],
        "torch": observed["torch"],
        "torchaudio": observed["torchaudio"],
        "numpy": observed["numpy"],
        "safetensors": observed["safetensors"],
        "librosa": observed["librosa"],
        "lhotse": observed["lhotse"],
        "soundfile": observed["soundfile"],
        "scipy": observed["scipy"],
        "omegaconf": observed["omegaconf"],
        "hydra_core": observed["hydra-core"],
        "lightning": observed["lightning"],
    }
    return runtime, source_root, source_identities


def _sortformer_tensor_bytes(
    tensor: torch_types.Tensor, numpy: Any, torch: Any
) -> tuple[str, bytes]:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise RuntimeError(
            f"{SORTFORMER_PROFILE} source tensor is not contiguous CPU storage"
        )
    detached = tensor.detach()
    if detached.dtype == torch.float32:
        if not torch.isfinite(detached).all().item():
            raise RuntimeError(f"{SORTFORMER_PROFILE} source tensor is non-finite")
        array = detached.numpy().astype("<f4", copy=False)
        return "f32", array.tobytes(order="C")
    if detached.dtype == torch.int64:
        array = detached.numpy().astype("<i8", copy=False)
        return "i64", array.tobytes(order="C")
    raise RuntimeError(f"{SORTFORMER_PROFILE} source tensor dtype is unsupported")


def _sortformer_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    disposition = record["disposition"]
    projected_disposition: dict[str, Any] = {"kind": disposition["kind"]}
    if disposition["kind"] == "exported":
        destination = disposition["destination"]
        projected_disposition["transform"] = disposition["transform"]
        projected_disposition["destination"] = {
            "name": destination["name"],
            "dtype": destination["dtype"],
            "shape": destination["shape"],
            "logical_layout": destination["logical_layout"],
            "elements": destination["elements"],
            "bytes": destination["bytes"],
        }
    return {
        "source_name": record["source_name"],
        "source_origin": record["source_origin"],
        "source_dtype": record["source_dtype"],
        "source_shape": record["source_shape"],
        "source_logical_layout": record["source_logical_layout"],
        "source_elements": record["source_elements"],
        "source_bytes": record["source_bytes"],
        "disposition": projected_disposition,
    }


def _sortformer_manifest_sha256(records: list[dict[str, Any]]) -> str:
    manifest = {
        "schema_version": SORTFORMER_TENSOR_MANIFEST_SCHEMA,
        "model_id": SORTFORMER_MODEL_ID,
        "model_revision": SORTFORMER_MODEL_REVISION,
        "nemo_sha256": SORTFORMER_NEMO_SHA256,
        "config_sha256": SORTFORMER_CONFIG_SHA256,
        "checkpoint_sha256": SORTFORMER_CHECKPOINT_SHA256,
        "records": [_sortformer_manifest_record(record) for record in records],
    }
    return hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()


def _sortformer_record(
    name: str,
    origin: str,
    tensor: torch_types.Tensor,
    numpy: Any,
    torch: Any,
) -> tuple[dict[str, Any], torch_types.Tensor | None]:
    dtype, raw = _sortformer_tensor_bytes(tensor, numpy, torch)
    shape = list(tensor.shape)
    elements = tensor.numel()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    record: dict[str, Any] = {
        "source_name": name,
        "source_origin": origin,
        "source_dtype": dtype,
        "source_shape": shape,
        "source_logical_layout": SORTFORMER_SOURCE_LAYOUT,
        "source_value_sha256": source_sha256,
        "source_elements": elements,
        "source_bytes": len(raw),
    }
    if dtype == "f32" and name != SORTFORMER_DTYPE_SENTINEL:
        record["disposition"] = {
            "kind": "exported",
            "transform": "identity_contiguous_f32",
            "destination": {
                "name": name,
                "dtype": "f32",
                "shape": shape,
                "logical_layout": SORTFORMER_SOURCE_LAYOUT,
                "value_sha256": source_sha256,
                "elements": elements,
                "bytes": len(raw),
            },
        }
        return record, tensor
    if dtype == "i64" and name.endswith(".num_batches_tracked"):
        if shape or elements != 1 or len(raw) != 8:
            raise RuntimeError(f"{SORTFORMER_PROFILE} training counter is invalid")
        record["disposition"] = {"kind": "dropped_train_only"}
        return record, None
    if (
        name == SORTFORMER_DTYPE_SENTINEL
        and dtype == "f32"
        and shape == [0]
        and elements == 0
        and source_sha256 == SORTFORMER_EMPTY_SHA256
    ):
        record["disposition"] = {"kind": "dropped_runtime_sentinel"}
        return record, None
    raise RuntimeError(f"{SORTFORMER_PROFILE} tensor disposition is unsupported")


def _build_sortformer_state(
    config_bytes: bytes,
    checkpoint_bytes: bytes,
    numpy: Any,
    torch: Any,
) -> tuple[dict[str, torch_types.Tensor], list[dict[str, Any]], str]:
    from nemo.collections.asr.models import SortformerEncLabelModel
    from omegaconf import OmegaConf

    config = OmegaConf.create(config_bytes.decode("utf-8"))
    model = SortformerEncLabelModel(cfg=config).float().cpu().eval()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_bytes = b""
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{SORTFORMER_PROFILE} checkpoint state_dict is absent")
    nested_state = checkpoint.get("state_dict")
    if isinstance(nested_state, dict):
        state_source = nested_state
    elif checkpoint and all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in checkpoint.items()
    ):
        state_source = checkpoint
    else:
        raise RuntimeError(f"{SORTFORMER_PROFILE} checkpoint state_dict is absent")
    model.load_state_dict(state_source, strict=True, assign=True)
    checkpoint = None
    state_source = None
    model.float().cpu().eval()

    parameter_tensors = list(model.parameters())
    if (
        len(parameter_tensors) != SORTFORMER_PARAMETER_TENSORS
        or sum(parameter.numel() for parameter in parameter_tensors)
        != SORTFORMER_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(f"{SORTFORMER_PROFILE} parameter census changed")

    state = model.state_dict()
    inventory = [
        {
            "name": name,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
        }
        for name, tensor in state.items()
    ]
    if not hmac.compare_digest(
        hashlib.sha256(_canonical_json_bytes(inventory)).hexdigest(),
        SORTFORMER_STATE_INVENTORY_SHA256,
    ):
        raise RuntimeError(f"{SORTFORMER_PROFILE} state inventory changed")

    state_names = set(state)
    nonpersistent = {
        name: tensor
        for name, tensor in model.named_buffers()
        if name not in state_names
    }
    if set(nonpersistent) != {
        SORTFORMER_POSITION_TENSOR,
        SORTFORMER_DTYPE_SENTINEL,
    }:
        raise RuntimeError(f"{SORTFORMER_PROFILE} non-persistent buffers changed")

    records = []
    tensors: dict[str, torch_types.Tensor] = {}
    for name, tensor in state.items():
        record, exported = _sortformer_record(
            name, "state_dict", tensor, numpy, torch
        )
        records.append(record)
        if exported is not None:
            tensors[name] = exported
    for name, tensor in nonpersistent.items():
        record, exported = _sortformer_record(
            name, "nonpersistent_buffer", tensor, numpy, torch
        )
        records.append(record)
        if exported is not None:
            tensors[name] = exported
    records.sort(key=lambda record: record["source_name"])

    state_f32 = [tensor for tensor in state.values() if tensor.dtype == torch.float32]
    state_i64 = [tensor for tensor in state.values() if tensor.dtype == torch.int64]
    if (
        len(state) != SORTFORMER_STATE_TENSORS
        or sum(tensor.numel() for tensor in state.values())
        != SORTFORMER_STATE_ELEMENTS
        or len(state_f32) != SORTFORMER_STATE_F32_TENSORS
        or sum(tensor.numel() for tensor in state_f32)
        != SORTFORMER_STATE_F32_ELEMENTS
        or sum(tensor.numel() * 4 for tensor in state_f32)
        != SORTFORMER_STATE_F32_BYTES
        or len(state_i64) != SORTFORMER_STATE_I64_TENSORS
        or sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        != SORTFORMER_STATE_PAYLOAD_BYTES
        or len(records) != SORTFORMER_SOURCE_RECORDS
        or len(tensors) != SORTFORMER_EXPORTED_TENSORS
        or sum(tensor.numel() for tensor in tensors.values())
        != SORTFORMER_PACKAGE_F32_ELEMENTS
    ):
        raise RuntimeError(f"{SORTFORMER_PROFILE} tensor census changed")
    dropped = sum(
        record["disposition"]["kind"] != "exported" for record in records
    )
    if dropped != SORTFORMER_DROPPED_TENSORS:
        raise RuntimeError(f"{SORTFORMER_PROFILE} dropped tensor census changed")
    return tensors, records, _sortformer_manifest_sha256(records)


def _build_sortformer_receipt(
    records: list[dict[str, Any]],
    manifest_sha256: str,
    package_bytes: bytes,
    runtime: dict[str, str],
    converter_sha256: str,
) -> dict[str, Any]:
    package_sha256 = hashlib.sha256(package_bytes).hexdigest()
    return {
        "schema_version": SORTFORMER_RECEIPT_SCHEMA,
        "model": {
            "model_id": SORTFORMER_MODEL_ID,
            "model_revision": SORTFORMER_MODEL_REVISION,
            "nemo_bytes": SORTFORMER_NEMO_BYTES,
            "nemo_sha256": SORTFORMER_NEMO_SHA256,
            "config_sha256": SORTFORMER_CONFIG_SHA256,
            "checkpoint_bytes": SORTFORMER_CHECKPOINT_BYTES,
            "checkpoint_sha256": SORTFORMER_CHECKPOINT_SHA256,
            "state_inventory_sha256": SORTFORMER_STATE_INVENTORY_SHA256,
            "tensor_manifest_sha256": manifest_sha256,
            "nemo_source_revision": SORTFORMER_NEMO_SOURCE_REVISION,
            "external_contract_sha256": SORTFORMER_EXTERNAL_CONTRACT_SHA256,
            "runtime_fingerprint_sha256": SORTFORMER_RUNTIME_FINGERPRINT_SHA256,
            "oracle_adapter_sha256": SORTFORMER_ORACLE_ADAPTER_SHA256,
            "trainable_parameters": SORTFORMER_TRAINABLE_PARAMETERS,
            "parameter_tensors": SORTFORMER_PARAMETER_TENSORS,
            "state_tensors": SORTFORMER_STATE_TENSORS,
            "state_elements": SORTFORMER_STATE_ELEMENTS,
            "state_f32_tensors": SORTFORMER_STATE_F32_TENSORS,
            "state_f32_elements": SORTFORMER_STATE_F32_ELEMENTS,
            "state_f32_bytes": SORTFORMER_STATE_F32_BYTES,
            "state_i64_tensors": SORTFORMER_STATE_I64_TENSORS,
            "state_payload_bytes": SORTFORMER_STATE_PAYLOAD_BYTES,
        },
        "execution": {
            "streaming_mode": True,
            "async_streaming": False,
            "encoder_attention_context": [-1, -1],
            "encoder_attention_style": "regular",
            "transformer_mask_future": False,
            "transformer_pre_ln": False,
            "drop_extra_pre_encoded": 0,
        },
        "source_files": [
            {"path": path, "sha256": sha256}
            for path, sha256 in SORTFORMER_SOURCE_FILES
        ],
        "converter": {
            "converter_id": SORTFORMER_CONVERTER_ID,
            "converter_version": SORTFORMER_CONVERTER_VERSION,
            "source_sha256": converter_sha256,
        },
        "runtime": runtime,
        "license": {
            "model_license_id": "NVIDIA Open Model License",
            "model_license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
            "model_license_snapshot_retrieved_date": "2026-08-06",
            "model_license_last_modified": "Mon, 03 Aug 2026 17:46:28 GMT",
            "model_license_etag": "4b001-658281e31650b",
            "model_license_payload_sha256": "13c9c998e24abd5211cff4b5c912902f566bd710294da98580be7b3376626f04",
            "model_weight_distribution_policy": "operator_local_no_git_no_release",
            "nemo_source_license_spdx": "Apache-2.0",
            "nemo_source_license_sha256": "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
            "embedded_notice_source_path": "nemo/collections/asr/parts/preprocessing/features.py",
            "embedded_notice_source_sha256": "4290ed2d697362a68a6158fb8b7b8d1e2306b223b83172c63fc6b5d31b28ee69",
            "embedded_notice_license_spdx": "MIT",
            "embedded_notice_attribution": "Ryan Leary",
            "embedded_notice_attribution_required": True,
        },
        "package": {
            "format": "safetensors",
            "sha256": package_sha256,
            "bytes": len(package_bytes),
            "payload_bytes": SORTFORMER_PACKAGE_PAYLOAD_BYTES,
            "f32_elements": SORTFORMER_PACKAGE_F32_ELEMENTS,
            "tensor_count": SORTFORMER_EXPORTED_TENSORS,
            "dtype": "f32",
            "byte_order": "little_endian",
            "tensor_order": "lexicographic_name_order",
            "logical_layout": SORTFORMER_SOURCE_LAYOUT,
            "metadata_policy": "absent",
        },
        "records": records,
    }


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


def _run_sortformer_export(
    args: argparse.Namespace,
    numpy: Any,
    safetensors: Any,
    torch: Any,
) -> int:
    try:
        converter_path = Path(__file__).resolve(strict=True)
        converter_identity = _stable_file_identity(converter_path)
        converter_sha256 = converter_identity[-1]
        runtime, source_root, source_identities = _require_sortformer_runtime()
        config_bytes, checkpoint_bytes = _read_exact_sortformer_archive(args.input)
        tensors, records, manifest_sha256 = _build_sortformer_state(
            config_bytes,
            checkpoint_bytes,
            numpy,
            torch,
        )
        config_bytes = b""
        checkpoint_bytes = b""
        if not hmac.compare_digest(
            manifest_sha256, SORTFORMER_TENSOR_MANIFEST_SHA256
        ):
            raise RuntimeError(f"{SORTFORMER_PROFILE} topology projection changed")
        package_bytes = _build_deterministic_safetensors(tensors, None)
        package_sha256 = hashlib.sha256(package_bytes).hexdigest()
        if len(package_bytes) != SORTFORMER_PACKAGE_BYTES or not hmac.compare_digest(
            package_sha256, SORTFORMER_PACKAGE_SHA256
        ):
            raise RuntimeError(f"{SORTFORMER_PROFILE} package identity changed")
        if _sortformer_source_identities(source_root) != source_identities:
            raise RuntimeError(f"{SORTFORMER_PROFILE} NeMo sources changed during export")
        if _stable_file_identity(converter_path) != converter_identity:
            raise RuntimeError(f"{SORTFORMER_PROFILE} converter changed during export")
        receipt = _build_sortformer_receipt(
            records,
            manifest_sha256,
            package_bytes,
            runtime,
            converter_sha256,
        )
        receipt_bytes = _canonical_json_bytes(receipt)

        # All source, runtime, tensor, and complete byte-stream checks finish
        # before either exclusive-create final path is opened. Publish the small
        # receipt first: an interrupted package write can be retried at a fresh
        # package path while reusing the exact content-addressed receipt.
        if _sortformer_source_identities(source_root) != source_identities:
            raise RuntimeError(f"{SORTFORMER_PROFILE} NeMo sources changed before publish")
        if _stable_file_identity(converter_path) != converter_identity:
            raise RuntimeError(f"{SORTFORMER_PROFILE} converter changed before publish")
        _publish_new_file(args.receipt_output, receipt_bytes)
        _publish_new_file(args.output, package_bytes)
        if _sortformer_source_identities(source_root) != source_identities:
            raise RuntimeError(f"{SORTFORMER_PROFILE} NeMo sources changed during publish")
        if _stable_file_identity(converter_path) != converter_identity:
            raise RuntimeError(f"{SORTFORMER_PROFILE} converter changed during publish")
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        tarfile.TarError,
        safetensors.SafetensorError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {len(tensors)} tensors -> {args.output}")
    print(f"input  sha256: {SORTFORMER_NEMO_SHA256}")
    print(f"output bytes: {len(package_bytes)}")
    print(f"output sha256: {receipt['package']['sha256']}")
    print(f"manifest sha256: {manifest_sha256}")
    print(f"converter sha256: {converter_sha256}")
    print(f"receipt bytes: {len(receipt_bytes)}")
    print(f"receipt sha256: {hashlib.sha256(receipt_bytes).hexdigest()}")
    return 0


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
        choices=("generic", ECAPA_PROFILE, SORTFORMER_PROFILE),
        default="generic",
        help="model-specific validation/export profile (default: generic)",
    )
    parser.add_argument(
        "--full-oracle-output",
        type=Path,
        default=None,
        help="also write the frozen public ECAPA full-stage oracle safetensors",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help="write the canonical frozen Sortformer conversion receipt",
    )
    args = parser.parse_args()

    if (
        args.profile in (ECAPA_PROFILE, SORTFORMER_PROFILE)
        and sys.version_info[:3] != REQUIRED_PYTHON_VERSION
    ):
        frozen_profile = (
            SORTFORMER_PROFILE
            if args.profile == SORTFORMER_PROFILE
            else ECAPA_PROFILE
        )
        print(
            f"error: {frozen_profile} requires Python {REQUIRED_PYTHON_VERSION_TEXT} "
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
    if args.profile != SORTFORMER_PROFILE and args.output.exists():
        print(f"error: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    if args.profile == SORTFORMER_PROFILE:
        if args.key is not None:
            print(f"error: {SORTFORMER_PROFILE} does not accept --key", file=sys.stderr)
            return 2
        if args.full_oracle_output is not None:
            print(
                f"error: {SORTFORMER_PROFILE} does not accept --full-oracle-output",
                file=sys.stderr,
            )
            return 2
        if args.receipt_output is None:
            print(
                f"error: {SORTFORMER_PROFILE} requires --receipt-output",
                file=sys.stderr,
            )
            return 2
        package_output = args.output.resolve()
        receipt_output = args.receipt_output.resolve()
        if (
            package_output == receipt_output
            or package_output in receipt_output.parents
            or receipt_output in package_output.parents
        ):
            print(
                "error: package and receipt output paths must neither match nor contain one another",
                file=sys.stderr,
            )
            return 2
    elif args.receipt_output is not None:
        print(
            f"error: --receipt-output requires the frozen {SORTFORMER_PROFILE} profile",
            file=sys.stderr,
        )
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

    if args.profile == SORTFORMER_PROFILE:
        return _run_sortformer_export(args, numpy, safetensors, torch)

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
