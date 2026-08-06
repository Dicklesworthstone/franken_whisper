#!/usr/bin/env python3
"""Export an identity-bound synthetic L1 frontend truth pack for Sortformer.

This is offline reference tooling. It never enters the Rust runtime and it
never accepts caller audio. The only admitted inputs are four deterministic,
non-human fixtures defined below. Model weights and generated artifacts stay
operator-local; the Rust verifier consumes their hashes and tensor payloads.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import io
import json
import os
import platform
import re
import struct
import sys
from pathlib import Path
from typing import Any


PROFILE = "sortformer-streaming-4spk-v2.1"
SCHEMA = "franken-whisper-sortformer-activation-receipt-v1"
FLOOR_SCHEMA = "franken-whisper-sortformer-oracle-floor-v1"
EXPORTER_ID = "franken-whisper-sortformer-activation-exporter"
EXPORTER_VERSION = "1"
FIXTURE_SET = "sortformer-synthetic-frontend-v1"
HELPER_SOURCE_SHA256 = (
    "6a946cc6647bf52244d0eaad89db834bdc52cc61fd08d9563632dd1f9d239c1e"
)
MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
MODEL_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
MODEL_BYTES = 471_367_680
CONFIG_SHA256 = "2865d469c4d2aac54aa5b8a956b2423c053806dd20d5bf5d08675942a1acface"
CHECKPOINT_SHA256 = (
    "eca9773c2dab91dd41fbaa4473cebb9d00811d67788ce2de609dadc6e499cdf4"
)
STATE_INVENTORY_SHA256 = (
    "f4f219cf4ac6f755247b56d19e425db3d6a7c23c4509176549b363b63abdf532"
)
NEMO_SOURCE_REVISION = "40ace43c7cf151af78dc22027c02feeca7e06b6a"
EXTERNAL_CONTRACT_SHA256 = (
    "7ac048e3372fe4c622840beddfbeef42944d961408360324cb7276a69c8542c5"
)
SAMPLE_RATE_HZ = 16_000
FFT_LENGTH = 512
WINDOW_SAMPLES = 400
HOP_SAMPLES = 160
MEL_BINS = 128
FLOOR_THREAD_COUNTS = (1, 8)
FLOOR_REPETITIONS = 5
TENSOR_LAYOUT = "pytorch_contiguous_row_major"
CONVERSION_RECEIPT_SHA256 = (
    "a1c6dce95ef4fd715965951bdaaa136e55e2219f93cf78122f8b462fbd07cbbe"
)
CONVERTED_PACKAGE_SHA256 = (
    "487fa30cb0aa9799c77bd9985e6787962c3991fab8d4d576a4f1221d45298f6a"
)


def _load_frozen_helper(script_path: Path) -> tuple[dict[str, Any], bytes]:
    helper_path = (script_path.parent / "convert_to_safetensors.py").resolve(strict=True)
    helper_bytes = helper_path.read_bytes()
    if not hmac.compare_digest(hashlib.sha256(helper_bytes).hexdigest(), HELPER_SOURCE_SHA256):
        raise RuntimeError(f"{PROFILE} conversion helper identity mismatch")
    namespace: dict[str, Any] = {
        "__file__": os.fspath(helper_path),
        "__name__": "franken_whisper_frozen_sortformer_conversion_helper",
    }
    exec(compile(helper_bytes, os.fspath(helper_path), "exec"), namespace)
    return namespace, helper_bytes


def _build_model(
    config_bytes: bytes,
    checkpoint_bytes: bytes,
    helper: dict[str, Any],
    torch: Any,
) -> Any:
    from nemo.collections.asr.models import SortformerEncLabelModel
    from omegaconf import OmegaConf

    config = OmegaConf.create(config_bytes.decode("utf-8"))
    model = SortformerEncLabelModel(cfg=config).float().cpu().eval()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{PROFILE} checkpoint state_dict is absent")
    nested_state = checkpoint.get("state_dict")
    if isinstance(nested_state, dict):
        state_source = nested_state
    elif checkpoint and all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in checkpoint.items()
    ):
        state_source = checkpoint
    else:
        raise RuntimeError(f"{PROFILE} checkpoint state_dict is absent")
    model.load_state_dict(state_source, strict=True, assign=True)
    model.float().cpu().eval()

    parameter_tensors = list(model.parameters())
    if len(parameter_tensors) != helper["SORTFORMER_PARAMETER_TENSORS"] or sum(
        parameter.numel() for parameter in parameter_tensors
    ) != helper["SORTFORMER_TRAINABLE_PARAMETERS"]:
        raise RuntimeError(f"{PROFILE} parameter census changed")
    inventory = [
        {
            "name": name,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
        }
        for name, tensor in model.state_dict().items()
    ]
    inventory_sha256 = hashlib.sha256(helper["_canonical_json_bytes"](inventory)).hexdigest()
    if not hmac.compare_digest(inventory_sha256, STATE_INVENTORY_SHA256):
        raise RuntimeError(f"{PROFILE} state inventory changed")
    return model


def _require_frontend_contract(model: Any, torch: Any) -> None:
    featurizer = model.preprocessor.featurizer
    observed = {
        "streaming_mode": bool(model.streaming_mode),
        "sample_rate": int(featurizer.sample_rate),
        "win_length": int(featurizer.win_length),
        "hop_length": int(featurizer.hop_length),
        "n_fft": int(featurizer.n_fft),
        "exact_pad": bool(featurizer.exact_pad),
        "preemph": float(featurizer.preemph),
        "nfilt": int(featurizer.nfilt),
        "log": bool(featurizer.log),
        "log_zero_guard_type": str(featurizer.log_zero_guard_type),
        "log_zero_guard_value": float(featurizer.log_zero_guard_value),
        "dither": float(featurizer.dither),
        "pad_to": int(featurizer.pad_to),
        "frame_splicing": int(featurizer.frame_splicing),
        "normalize": str(featurizer.normalize),
        "mag_power": float(featurizer.mag_power),
        "training": bool(featurizer.training),
        "window_shape": list(featurizer.window.shape),
        "mel_shape": list(featurizer.fb.shape),
    }
    expected = {
        "streaming_mode": True,
        "sample_rate": SAMPLE_RATE_HZ,
        "win_length": WINDOW_SAMPLES,
        "hop_length": HOP_SAMPLES,
        "n_fft": FFT_LENGTH,
        "exact_pad": False,
        "preemph": 0.97,
        "nfilt": MEL_BINS,
        "log": True,
        "log_zero_guard_type": "add",
        "log_zero_guard_value": 2**-24,
        "dither": 1e-5,
        "pad_to": 16,
        "frame_splicing": 1,
        "normalize": "NA",
        "mag_power": 2.0,
        "training": False,
        "window_shape": [WINDOW_SAMPLES],
        "mel_shape": [1, MEL_BINS, FFT_LENGTH // 2 + 1],
    }
    if observed != expected:
        raise RuntimeError(f"{PROFILE} effective frontend contract changed")
    if model.preprocessor.dtype_sentinel_tensor.dtype != torch.float32:
        raise RuntimeError(f"{PROFILE} frontend output dtype changed")


def _fixtures(torch: Any) -> list[tuple[str, str, Any]]:
    silence = torch.zeros(320, dtype=torch.int16)

    impulse = torch.zeros(480, dtype=torch.int16)
    impulse[0] = 16_384
    impulse[159] = -8_192
    impulse[320] = 24_576

    tone_cycle = [
        0,
        6_269,
        11_585,
        15_137,
        16_384,
        15_137,
        11_585,
        6_269,
        0,
        -6_269,
        -11_585,
        -15_137,
        -16_384,
        -15_137,
        -11_585,
        -6_269,
    ]
    tone = torch.tensor(
        [tone_cycle[index % len(tone_cycle)] for index in range(640)],
        dtype=torch.int16,
    )

    partial_tail = torch.tensor(
        [((index * 1_103 + 12_345) % 32_768 - 16_384) for index in range(321)],
        dtype=torch.int16,
    )
    return [
        ("silence_320", "all_zero_i16_v1", silence),
        ("impulse_480", "three_exact_impulses_i16_v1", impulse),
        ("tone_640", "exact_i16_cycle_v1", tone),
        ("partial_tail_321", "exact_integer_lcg_i16_v1", partial_tail),
    ]


def _frontend_trace(model: Any, pcm: Any, torch: Any) -> tuple[dict[str, Any], int, int]:
    featurizer = model.preprocessor.featurizer
    captured: dict[str, Any] = {}
    original_stft = featurizer.stft

    def capturing_stft(values: Any) -> Any:
        captured["preemphasis"] = values.detach().cpu().contiguous()
        spectrum = original_stft(values)
        captured["stft"] = spectrum.detach().cpu().contiguous()
        return spectrum

    featurizer.stft = capturing_stft
    try:
        waveform = pcm.reshape(1, -1).contiguous()
        lengths = torch.tensor([pcm.numel()], dtype=torch.long)
        processed, processed_lengths = model.process_signal(waveform, lengths)
    finally:
        featurizer.stft = original_stft

    if processed_lengths.shape != (1,):
        raise RuntimeError(f"{PROFILE} frontend length shape changed")
    valid_frames = int(processed_lengths.item())
    expected_valid_frames = pcm.numel() // HOP_SAMPLES
    if valid_frames != expected_valid_frames:
        raise RuntimeError(f"{PROFILE} frontend valid length changed")

    preemphasis = captured["preemphasis"]
    stft = captured["stft"]
    if preemphasis.shape != (1, pcm.numel()) or stft.ndim != 3:
        raise RuntimeError(f"{PROFILE} captured frontend shape changed")
    physical_frames = int(stft.shape[2])
    if physical_frames != valid_frames + 1:
        raise RuntimeError(f"{PROFILE} physical frontend length changed")

    padded_pcm = torch.nn.functional.pad(preemphasis, (FFT_LENGTH // 2, FFT_LENGTH // 2))
    frames = padded_pcm.unfold(1, FFT_LENGTH, HOP_SAMPLES)
    padded_window = torch.nn.functional.pad(
        featurizer.window.detach().cpu(),
        ((FFT_LENGTH - WINDOW_SAMPLES) // 2,) * 2,
    )
    windowed = (frames * padded_window).contiguous()
    if windowed.shape != (1, physical_frames, FFT_LENGTH):
        raise RuntimeError(f"{PROFILE} reconstructed windowed-frame shape changed")

    stft_real_imag = torch.view_as_real(stft).contiguous()
    magnitude = torch.sqrt(stft_real_imag.pow(2).sum(-1))
    power = magnitude.pow(2)
    mel_energy = torch.matmul(featurizer.fb.detach().cpu(), power)
    log_mel = torch.log(mel_energy + float(featurizer.log_zero_guard_value))
    observed_padded = processed.detach().cpu().contiguous()
    if observed_padded.shape[:2] != (1, MEL_BINS):
        raise RuntimeError(f"{PROFILE} padded frontend shape changed")
    log_mel_valid = log_mel[:, :, :valid_frames].contiguous()
    observed_valid = observed_padded[:, :, :valid_frames].contiguous()
    if not torch.equal(log_mel_valid, observed_valid):
        raise RuntimeError(f"{PROFILE} exported log-mel does not match model preprocessing")
    if torch.count_nonzero(observed_padded[:, :, valid_frames:]).item() != 0:
        raise RuntimeError(f"{PROFILE} padded frontend tail is not masked to zero")
    if not torch.isfinite(log_mel_valid).all().item():
        raise RuntimeError(f"{PROFILE} frontend produced non-finite activations")

    trace = {
        "decoded_pcm_f32": pcm.reshape(1, -1).contiguous(),
        "input_length_i64": torch.tensor([pcm.numel()], dtype=torch.int64),
        "preemphasis_f32": preemphasis,
        "windowed_frames_f32": windowed,
        "stft_complex_ri_f32": stft_real_imag,
        "power_f32": power.contiguous(),
        "mel_energy_f32": mel_energy.contiguous(),
        "log_mel_physical_f32": log_mel.contiguous(),
        "log_mel_f32": log_mel_valid,
        "frontend_padded_f32": observed_padded,
        "valid_length_i64": torch.tensor([valid_frames], dtype=torch.int64),
    }
    return trace, valid_frames, physical_frames


def _tensor_bytes(tensor: Any) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    if str(tensor.dtype) == "torch.float32":
        return array.astype("<f4", copy=False).tobytes(order="C")
    if str(tensor.dtype) == "torch.int64":
        return array.astype("<i8", copy=False).tobytes(order="C")
    raise RuntimeError(f"{PROFILE} activation tensor dtype is unsupported")


def _tensor_dtype(tensor: Any) -> tuple[str, str, int]:
    if str(tensor.dtype) == "torch.float32":
        return "f32", "F32", 4
    if str(tensor.dtype) == "torch.int64":
        return "i64", "I64", 8
    raise RuntimeError(f"{PROFILE} activation tensor dtype is unsupported")


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    raw = _tensor_bytes(tensor)
    dtype, _, _ = _tensor_dtype(tensor)
    return {
        "name": name,
        "dtype": dtype,
        "shape": list(tensor.shape),
        "logical_layout": TENSOR_LAYOUT,
        "elements": tensor.numel(),
        "bytes": len(raw),
        "value_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _f32_bits(value: float) -> str:
    return "0x" + struct.pack(">f", float(value)).hex()


def _f64_bits(value: float) -> str:
    return "0x" + struct.pack(">d", float(value)).hex()


def _pairwise_floor(values: list[Any], torch: Any) -> dict[str, Any]:
    if len(values) != len(FLOOR_THREAD_COUNTS) * FLOOR_REPETITIONS:
        raise RuntimeError(f"{PROFILE} oracle replay census changed")
    pair_count = 0
    compared_values = 0
    mismatch_count = 0
    maximum_absolute = 0.0
    absolute_sum = 0.0
    squared_difference_sum = 0.0
    squared_scale_sum = 0.0
    byte_exact = True
    for left_index, left in enumerate(values):
        for right in values[left_index + 1 :]:
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError(f"{PROFILE} oracle replay shape changed")
            pair_count += 1
            byte_exact = byte_exact and hmac.compare_digest(
                _tensor_bytes(left), _tensor_bytes(right)
            )
            left_f64 = left.to(torch.float64)
            right_f64 = right.to(torch.float64)
            difference = torch.abs(left_f64 - right_f64)
            compared_values += difference.numel()
            mismatch_count += int(torch.count_nonzero(difference).item())
            if difference.numel() != 0:
                maximum_absolute = max(maximum_absolute, float(torch.max(difference).item()))
                absolute_sum += float(torch.sum(difference).item())
                squared_difference_sum += float(torch.sum(difference * difference).item())
                scale = torch.maximum(torch.abs(left_f64), torch.abs(right_f64))
                squared_scale_sum += float(torch.sum(scale * scale).item())
    if pair_count != 45:
        raise RuntimeError(f"{PROFILE} oracle replay pair census changed")
    mean_absolute = absolute_sum / compared_values if compared_values else 0.0
    relative_l2 = (
        (squared_difference_sum / squared_scale_sum) ** 0.5
        if squared_scale_sum > 0.0
        else 0.0
    )
    return {
        "run_count": len(values),
        "pair_count": pair_count,
        "compared_values": compared_values,
        "mismatch_count": mismatch_count,
        "byte_exact": byte_exact,
        "max_abs_diff_f32_bits": _f32_bits(maximum_absolute),
        "mean_abs_diff_f64_bits": _f64_bits(mean_absolute),
        "relative_l2_f64_bits": _f64_bits(relative_l2),
    }


def _build_deterministic_safetensors(
    tensors: dict[str, Any], safetensors: Any, torch: Any
) -> bytes:
    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        dtype, safetensors_dtype, width = _tensor_dtype(tensor)
        if dtype == "f32" and not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"{PROFILE} activation tensor {name!r} is non-finite")
        raw = _tensor_bytes(tensor)
        if len(raw) != tensor.numel() * width:
            raise RuntimeError(f"{PROFILE} activation tensor byte count changed")
        header[name] = {
            "dtype": safetensors_dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(
        header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    package_bytes = b"".join(
        [struct.pack("<Q", len(header_bytes)), header_bytes, *payloads]
    )
    try:
        loaded = safetensors.torch.load(package_bytes)
    except Exception as error:
        raise RuntimeError(f"{PROFILE} activation package failed self-parse") from error
    if sorted(loaded) != sorted(tensors):
        raise RuntimeError(f"{PROFILE} activation package tensor census changed")
    for name, expected in tensors.items():
        observed = loaded[name]
        if observed.dtype != expected.dtype or tuple(observed.shape) != tuple(expected.shape):
            raise RuntimeError(f"{PROFILE} activation package tensor contract changed")
        if not hmac.compare_digest(_tensor_bytes(observed), _tensor_bytes(expected)):
            raise RuntimeError(f"{PROFILE} activation package tensor payload changed")
    return package_bytes


def _capture_floor(
    model: Any, torch: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    fixture_definitions = _fixtures(torch)
    baseline: dict[str, Any] = {
        "analysis_window_f32": model.preprocessor.featurizer.window.detach().cpu().contiguous(),
        "mel_filterbank_f32": model.preprocessor.featurizer.fb.detach().cpu().contiguous(),
    }
    fixture_receipts: list[dict[str, Any]] = []
    captures: dict[str, dict[str, list[Any]]] = {}

    for threads in FLOOR_THREAD_COUNTS:
        torch.set_num_threads(threads)
        for repetition in range(FLOOR_REPETITIONS):
            for fixture_name, generator, pcm_i16 in fixture_definitions:
                pcm = pcm_i16.to(torch.float32) / 32_768.0
                trace, valid_frames, physical_frames = _frontend_trace(model, pcm, torch)
                fixture_captures = captures.setdefault(fixture_name, {})
                for stage, tensor in trace.items():
                    fixture_captures.setdefault(stage, []).append(tensor)
                if threads == FLOOR_THREAD_COUNTS[0] and repetition == 0:
                    for stage, tensor in trace.items():
                        baseline[f"fixture.{fixture_name}.{stage}"] = tensor
                    pcm16_bytes = pcm_i16.numpy().astype("<i2", copy=False).tobytes(order="C")
                    generator_parameters = {
                        "generator": generator,
                        "sample_count": pcm_i16.numel(),
                        "pcm16_sha256": hashlib.sha256(pcm16_bytes).hexdigest(),
                    }
                    fixture_receipts.append(
                        {
                            "name": fixture_name,
                            "generator": generator,
                            "generator_parameters_sha256": hashlib.sha256(
                                json.dumps(
                                    generator_parameters,
                                    ensure_ascii=True,
                                    separators=(",", ":"),
                                    sort_keys=True,
                                ).encode("utf-8")
                            ).hexdigest(),
                            "sample_rate_hz": SAMPLE_RATE_HZ,
                            "channels": 1,
                            "sample_count": pcm_i16.numel(),
                            "valid_frames": valid_frames,
                            "physical_frames": physical_frames,
                            "pcm16_sha256": generator_parameters["pcm16_sha256"],
                            "decoded_f32_sha256": hashlib.sha256(
                                _tensor_bytes(trace["decoded_pcm_f32"])
                            ).hexdigest(),
                        }
                    )

    observations = []
    for fixture_name in sorted(captures):
        for stage in sorted(captures[fixture_name]):
            metric = _pairwise_floor(captures[fixture_name][stage], torch)
            metric.update({"fixture": fixture_name, "stage": stage})
            observations.append(metric)
    all_byte_exact = all(observation["byte_exact"] for observation in observations)
    mismatch_count = sum(observation["mismatch_count"] for observation in observations)

    floor = {
        "schema_version": FLOOR_SCHEMA,
        "baseline_threads": FLOOR_THREAD_COUNTS[0],
        "baseline_repetition": 0,
        "thread_counts": list(FLOOR_THREAD_COUNTS),
        "repetitions_per_thread": FLOOR_REPETITIONS,
        "all_byte_exact": all_byte_exact,
        "mismatch_count": mismatch_count,
        "comparison_rule": "exact_ieee_bits",
        "absolute_tolerance_f32_bits": "0x00000000",
        "relative_tolerance_f32_bits": "0x00000000",
        "margin_basis": "deterministic_synthetic_preprocessing_zero_floor_no_margin",
        "observations": observations,
    }
    return baseline, fixture_receipts, floor


def _execution_identity(
    torch: Any, numpy: Any, helper: dict[str, Any]
) -> dict[str, Any]:
    torch_configuration = torch.__config__.show()
    blas_match = re.search(r"(?:^|[, ])BLAS_INFO=([^,\n ]+)", torch_configuration)
    if blas_match is None:
        raise RuntimeError(f"{PROFILE} torch BLAS backend is unknown")
    numpy_configuration = json.dumps(numpy.__config__.CONFIG, sort_keys=True, default=str)
    return {
        "operating_system": platform.platform(),
        "machine_architecture": platform.machine(),
        "device": "cpu",
        "compute_dtype": "float32",
        "autocast": False,
        "quantization": "none",
        "deterministic_algorithms": True,
        "torch_intraop_thread_counts": list(FLOOR_THREAD_COUNTS),
        "torch_interop_threads": 1,
        "data_loader_workers": 0,
        "torch_blas_backend": blas_match.group(1).lower(),
        "torch_configuration_sha256": hashlib.sha256(torch_configuration.encode()).hexdigest(),
        "numpy_configuration_sha256": hashlib.sha256(numpy_configuration.encode()).hexdigest(),
        "python_executable_sha256": helper["_stable_file_identity"](
            Path(sys.executable).resolve(strict=True)
        )[-1],
    }


def _build_receipt(
    exporter_sha256: str,
    runtime: dict[str, str],
    source_identities: tuple[tuple[Any, ...], ...],
    execution: dict[str, Any],
    fixtures: list[dict[str, Any]],
    floor: dict[str, Any],
    tensors: dict[str, Any],
    package_bytes: bytes,
) -> dict[str, Any]:
    records = [_tensor_record(name, tensors[name]) for name in sorted(tensors)]
    payload_bytes = sum(record["bytes"] for record in records)
    return {
        "schema_version": SCHEMA,
        "canonical_json_version": "lexicographic-json-v1",
        "authority": "diagnostic_only",
        "equivalence_level": "partial_l1_synthetic_frontend",
        "fixture_set": FIXTURE_SET,
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "nemo_sha256": MODEL_SHA256,
            "nemo_bytes": MODEL_BYTES,
            "config_sha256": CONFIG_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "state_inventory_sha256": STATE_INVENTORY_SHA256,
            "nemo_source_revision": NEMO_SOURCE_REVISION,
            "external_contract_sha256": EXTERNAL_CONTRACT_SHA256,
            "conversion_receipt_sha256": CONVERSION_RECEIPT_SHA256,
            "converted_package_sha256": CONVERTED_PACKAGE_SHA256,
        },
        "exporter": {
            "exporter_id": EXPORTER_ID,
            "exporter_version": EXPORTER_VERSION,
            "source_sha256": exporter_sha256,
            "conversion_helper_sha256": HELPER_SOURCE_SHA256,
        },
        "runtime": runtime,
        "source_files": [
            {"path": identity[0], "sha256": identity[-1]}
            for identity in source_identities
        ],
        "execution": execution,
        "fixtures": fixtures,
        "oracle_floor": floor,
        "package": {
            "format": "safetensors",
            "dtype_set": sorted({record["dtype"] for record in records}),
            "byte_order": "little_endian",
            "tensor_order": "lexicographic_name_order",
            "logical_layout": TENSOR_LAYOUT,
            "metadata_policy": "absent",
            "tensor_count": len(records),
            "f32_elements": sum(
                record["elements"] for record in records if record["dtype"] == "f32"
            ),
            "i64_elements": sum(
                record["elements"] for record in records if record["dtype"] == "i64"
            ),
            "payload_bytes": payload_bytes,
            "bytes": len(package_bytes),
            "sha256": hashlib.sha256(package_bytes).hexdigest(),
        },
        "records": records,
    }


def run(arguments: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve(strict=True)
    helper, helper_bytes = _load_frozen_helper(script_path)
    exporter_identity = helper["_stable_file_identity"](script_path)
    exporter_sha256 = exporter_identity[-1]
    runtime, source_root, source_identities = helper["_require_sortformer_runtime"]()

    import numpy
    import safetensors
    import safetensors.torch
    import torch

    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config_bytes, checkpoint_bytes = helper["_read_exact_sortformer_archive"](arguments.model)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            model = _build_model(config_bytes, checkpoint_bytes, helper, torch)
    config_bytes = b""
    checkpoint_bytes = b""
    _require_frontend_contract(model, torch)
    tensors, fixtures, floor = _capture_floor(model, torch)
    if not floor["all_byte_exact"]:
        raise RuntimeError(f"{PROFILE} frontend oracle floor is nonzero")

    package_bytes = _build_deterministic_safetensors(tensors, safetensors, torch)
    receipt = _build_receipt(
        exporter_sha256,
        runtime,
        source_identities,
        _execution_identity(torch, numpy, helper),
        fixtures,
        floor,
        tensors,
        package_bytes,
    )
    receipt_bytes = helper["_canonical_json_bytes"](receipt)

    if helper["_sortformer_source_identities"](source_root) != source_identities:
        raise RuntimeError(f"{PROFILE} NeMo sources changed during activation export")
    if helper["_stable_file_identity"](script_path) != exporter_identity:
        raise RuntimeError(f"{PROFILE} activation exporter changed during execution")
    if not hmac.compare_digest(hashlib.sha256(helper_bytes).hexdigest(), HELPER_SOURCE_SHA256):
        raise RuntimeError(f"{PROFILE} conversion helper changed during execution")

    # The package is published first and the receipt is the completion signal.
    # An interrupted first write is recoverable because exact existing bytes
    # are accepted and neither destination is overwritten.
    helper["_publish_new_file"](arguments.package_output, package_bytes)
    helper["_publish_new_file"](arguments.receipt_output, receipt_bytes)
    if helper["_sortformer_source_identities"](source_root) != source_identities:
        raise RuntimeError(f"{PROFILE} NeMo sources changed during publication")
    if helper["_stable_file_identity"](script_path) != exporter_identity:
        raise RuntimeError(f"{PROFILE} activation exporter changed during publication")

    print(f"wrote {len(tensors)} synthetic activation tensors")
    print(f"package bytes: {len(package_bytes)}")
    print(f"package sha256: {receipt['package']['sha256']}")
    print(f"receipt bytes: {len(receipt_bytes)}")
    print(f"receipt sha256: {hashlib.sha256(receipt_bytes).hexdigest()}")
    print(f"exporter sha256: {exporter_sha256}")
    print(f"oracle floor byte exact: {str(floor['all_byte_exact']).lower()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the frozen synthetic Sortformer L1 frontend truth pack."
    )
    parser.add_argument("model", type=Path, help="pinned operator-local .nemo artifact")
    parser.add_argument("package_output", type=Path, help="new operator-local safetensors path")
    parser.add_argument("receipt_output", type=Path, help="new operator-local receipt path")
    arguments = parser.parse_args()
    try:
        return run(arguments)
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
