#!/usr/bin/env python3
"""Export identity-bound Sortformer activation truth packs.

This is offline reference tooling. It never enters the Rust runtime and it
admits either the four deterministic non-human fixtures defined below or an
exact, frozen VoxConverse public-corpus descriptor. Model weights, public voice
activations, and generated artifacts stay operator-local; the Rust verifier
consumes their independently pinned hashes and tensor payloads.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import io
import json
import math
import os
import platform
import re
import struct
import sys
import wave
from pathlib import Path
from typing import Any


PROFILE = "sortformer-streaming-4spk-v2.1"
SCHEMA = "franken-whisper-sortformer-activation-receipt-v1"
FLOOR_SCHEMA = "franken-whisper-sortformer-oracle-floor-v1"
EXPORTER_ID = "franken-whisper-sortformer-activation-exporter"
EXPORTER_VERSION = "2"
FIXTURE_SET = "sortformer-synthetic-frontend-v1"
PUBLIC_EXPORTER_VERSION = "3"
PUBLIC_FIXTURE_SET = "sortformer-voxconverse-recommended-streaming-seams-v2"
PUBLIC_SCHEMA = "franken-whisper-sortformer-public-activation-receipt-v2"
PUBLIC_FLOOR_SCHEMA = "franken-whisper-sortformer-public-oracle-floor-v1"
PUBLIC_DESCRIPTOR_SCHEMA = "public-diarization-corpus-input-v2"
PUBLIC_DESCRIPTOR_SHA256 = (
    "befd93742d6154175adceaf98c2e80db94ec9f144bc4a18669331f2a83a01ded"
)
PUBLIC_CORPUS_KEY = "voxconverse-v1"
PUBLIC_SOURCE_VERSION = (
    "voxconverse-v0.3-labels-24bf60be-dev-wav-md5-"
    "2a6e07e7473d9841abb132554a698a36-balanced4"
)
PUBLIC_AUTHORITATIVE_URL = "https://mm.kaist.ac.kr/datasets/voxconverse/"
PUBLIC_LICENSE_ID = "CC-BY-4.0-ORIGINAL-COPYRIGHT"
PUBLIC_LICENSE_ACKNOWLEDGEMENT_ID = (
    "accept-voxconverse-cc-by-4.0-and-original-copyright"
)
PUBLIC_FIXTURES = (
    {
        "name": "hiyis_exact_two_chunks",
        "recording_id": "hiyis",
        "start_sample": 0,
        "sample_count": 481_280,
        "expected_speaker_count": 1,
        "contains_overlap": False,
        "coverage": ["exact_two_chunks", "cache_fill", "cache_compression"],
    },
    {
        "name": "mevkw_complete_three_speakers",
        "recording_id": "mevkw",
        "start_sample": 0,
        "sample_count": 1_632_000,
        "expected_speaker_count": 3,
        "contains_overlap": True,
        "coverage": [
            "complete_recording",
            "overlap",
            "three_speakers",
            "multiple_cache_compressions",
        ],
    },
    {
        "name": "syiwe_complete_three_speakers",
        "recording_id": "syiwe",
        "start_sample": 0,
        "sample_count": 1_106_338,
        "expected_speaker_count": 3,
        "contains_overlap": False,
        "coverage": ["three_speakers", "multiple_chunks", "short_tail"],
    },
    {
        "name": "iqtde_complete_four_speakers",
        "recording_id": "iqtde",
        "start_sample": 0,
        "sample_count": 1_756_416,
        "expected_speaker_count": 4,
        "contains_overlap": False,
        "coverage": ["four_speakers", "multiple_cache_compressions", "partial_tail"],
    },
)
PUBLIC_PROBE_ELEMENTS = 4_096
PUBLIC_STREAMING_PROFILE = {
    "spkcache_len": 188,
    "fifo_len": 40,
    "chunk_len": 340,
    "spkcache_update_period": 300,
    "chunk_left_context": 1,
    "chunk_right_context": 40,
}
HELPER_SOURCE_SHA256 = (
    "3ce885d1dcb0aeeebf2bb73c165f501a1d240e01ad70354c65cf43d8a3c6d8ce"
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
    "407c642f3d51b399514f6a35227b1c80886387472a44fb78f01b824d26318fb0"
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


def _tensor_items(value: Any) -> list[tuple[str, Any]]:
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        return [("", value)]
    if isinstance(value, (tuple, list)):
        items = []
        for index, member in enumerate(value):
            if hasattr(member, "detach") and hasattr(member, "dtype"):
                items.append((f".{index}", member))
        return items
    return []


class _SeamCapture:
    """Forward hooks for the exact evaluated neural seams, scoped to one replay."""

    def __init__(self) -> None:
        self.step = -1
        self.tensors: dict[str, Any] = {}
        self.handles: list[Any] = []

    def add(self, name: str, tensor: Any) -> None:
        if name in self.tensors:
            raise RuntimeError(f"{PROFILE} duplicate public activation stage {name!r}")
        captured = tensor.detach().cpu().contiguous()
        if str(captured.dtype) == "torch.bool":
            captured = captured.long()
        if str(captured.dtype) not in {"torch.float32", "torch.int64"}:
            if captured.is_floating_point():
                captured = captured.float()
            else:
                captured = captured.long()
        self.tensors[name] = captured.contiguous()

    def _prefix(self, stage: str) -> str:
        if self.step < 0:
            raise RuntimeError(f"{PROFILE} neural hook fired outside a streaming step")
        return f"step.{self.step:03d}.{stage}"

    def _output_hook(self, stage: str, derive_sigmoid: bool = False):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            items = _tensor_items(output)
            for suffix, tensor in items:
                name = self._prefix(stage + suffix)
                self.add(name, tensor)
                if derive_sigmoid and suffix == "":
                    self.add(self._prefix("l5.probabilities"), tensor.sigmoid())

        return hook

    def _block_pre_hook(self, stage: str):
        def hook(_module: Any, inputs: Any, keyword_inputs: Any) -> None:
            input_items = _tensor_items(inputs)
            if not input_items:
                input_items = _tensor_items(keyword_inputs.get("x"))
            if not input_items:
                input_items = _tensor_items(keyword_inputs.get("encoder_query"))
            if not input_items:
                raise RuntimeError(f"{PROFILE} block seam {stage!r} lost its input boundary")
            self.add(self._prefix(f"{stage}.input"), input_items[0][1])

        return hook

    def _block_output_hook(self, stage: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            output_items = _tensor_items(output)
            if not output_items:
                raise RuntimeError(f"{PROFILE} block seam {stage!r} lost its output boundary")
            self.add(self._prefix(f"{stage}.output"), output_items[0][1])

        return hook

    def register(self, model: Any) -> None:
        conformer_internal = {
            "feed_forward1": "feed_forward1",
            "self_attn.linear_q": "attention_query",
            "self_attn.linear_k": "attention_key",
            "self_attn.linear_v": "attention_value",
            "self_attn.linear_out": "attention_output",
            "conv.depthwise_conv": "convolution_depthwise",
            "feed_forward2": "feed_forward2",
        }
        transformer_internal = {
            "first_sub_layer.query_net": "attention_query",
            "first_sub_layer.key_net": "attention_key",
            "first_sub_layer.value_net": "attention_value",
            "first_sub_layer.out_projection": "attention_output",
            "second_sub_layer.dense_in": "feed_forward_inner",
            "second_sub_layer.dense_out": "feed_forward_output",
        }
        for name, module in model.named_modules():
            if re.fullmatch(r"encoder\.pre_encode\.conv\.(0|2|3|5|6)", name):
                stage = "l2.subsampling." + name.rsplit(".", 1)[1]
                self.handles.append(module.register_forward_hook(self._output_hook(stage)))
                continue
            if name == "encoder.pre_encode.out":
                self.handles.append(
                    module.register_forward_hook(self._output_hook("l2.subsampling.projection"))
                )
                continue
            match = re.fullmatch(r"encoder\.layers\.(\d+)", name)
            if match is not None:
                stage = f"l3.fastconformer.block.{int(match.group(1)):02d}"
                self.handles.append(
                    module.register_forward_pre_hook(self._block_pre_hook(stage), with_kwargs=True)
                )
                self.handles.append(module.register_forward_hook(self._block_output_hook(stage)))
                continue
            match = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", name)
            if match is not None and match.group(2) in conformer_internal:
                stage = (
                    f"l3.fastconformer.block.{int(match.group(1)):02d}."
                    f"{conformer_internal[match.group(2)]}"
                )
                self.handles.append(module.register_forward_hook(self._output_hook(stage)))
                continue
            if name == "sortformer_modules.encoder_proj":
                self.handles.append(
                    module.register_forward_hook(self._output_hook("l4.encoder_projection"))
                )
                continue
            match = re.fullmatch(r"transformer_encoder\.layers\.(\d+)", name)
            if match is not None:
                stage = f"l4.transformer.block.{int(match.group(1)):02d}"
                self.handles.append(
                    module.register_forward_pre_hook(self._block_pre_hook(stage), with_kwargs=True)
                )
                self.handles.append(module.register_forward_hook(self._block_output_hook(stage)))
                continue
            match = re.fullmatch(r"transformer_encoder\.layers\.(\d+)\.(.+)", name)
            if match is not None and match.group(2) in transformer_internal:
                stage = (
                    f"l4.transformer.block.{int(match.group(1)):02d}."
                    f"{transformer_internal[match.group(2)]}"
                )
                self.handles.append(module.register_forward_hook(self._output_hook(stage)))
                continue
            if name == "sortformer_modules.first_hidden_to_hidden":
                self.handles.append(module.register_forward_hook(self._output_hook("l5.hidden")))
                continue
            if name == "sortformer_modules.single_hidden_to_spks":
                self.handles.append(
                    module.register_forward_hook(self._output_hook("l5.logits", derive_sigmoid=True))
                )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _state_options(streaming_state: Any) -> dict[str, bool]:
    return {
        name: getattr(streaming_state, name, None) is not None
        for name in (
            "spkcache",
            "spkcache_lengths",
            "spkcache_preds",
            "fifo",
            "fifo_lengths",
            "fifo_preds",
            "spk_perm",
            "mean_sil_emb",
            "n_sil_frames",
        )
    }


def _capture_state(prefix: str, streaming_state: Any, capture: _SeamCapture) -> None:
    for name, present in _state_options(streaming_state).items():
        if present:
            capture.add(f"{prefix}.{name}", getattr(streaming_state, name))


def _public_postprocessing(predictions: Any, torch: Any) -> dict[str, Any]:
    from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams, ts_vad_post_processing
    from omegaconf import OmegaConf

    activity = (predictions >= 0.5).to(torch.int64)
    speech = (activity.sum(dim=2) > 0).to(torch.int64)
    overlap = (activity.sum(dim=2) > 1).to(torch.int64)
    changes = torch.nonzero(
        torch.any(activity[:, 1:, :] != activity[:, :-1, :], dim=2), as_tuple=False
    ).to(torch.int64)
    turns = []
    parameters = OmegaConf.structured(PostProcessingParams())
    for speaker in range(predictions.shape[2]):
        segments = ts_vad_post_processing(
            predictions[0, :, speaker],
            cfg_vad_params=parameters,
            unit_10ms_frame_count=8,
            bypass_postprocessing=False,
        )
        for start, end in segments.tolist():
            turns.append([float(start), float(end), float(speaker)])
    turns.sort(key=lambda turn: (turn[0], turn[1], turn[2]))
    turns_tensor = (
        torch.tensor(turns, dtype=torch.float32)
        if turns
        else torch.empty((0, 3), dtype=torch.float32)
    )
    return {
        "l7.activity_i64": activity.contiguous(),
        "l7.speech_i64": speech.contiguous(),
        "l7.overlap_i64": overlap.contiguous(),
        "l7.change_indices_i64": changes.contiguous(),
        "l8.turns_f32": turns_tensor.contiguous(),
    }


def _public_neural_trace(
    model: Any, pcm: Any, torch: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    trace, valid_frames, physical_frames = _frontend_trace(model, pcm, torch)
    processed = trace["frontend_padded_f32"][:, :, :valid_frames].contiguous()
    processed_lengths = torch.tensor([valid_frames], dtype=torch.int64)
    capture = _SeamCapture()
    capture.register(model)
    transitions = []
    total_predictions = torch.zeros((1, 0, model.sortformer_modules.n_spk), dtype=torch.float32)
    streaming_state = model.sortformer_modules.init_streaming_state(
        batch_size=1, async_streaming=False, device=model.device
    )
    processed_offset = torch.zeros((1,), dtype=torch.long, device=model.device)
    try:
        with torch.no_grad():
            loader = model.sortformer_modules.streaming_feat_loader(
                feat_seq=processed,
                feat_seq_length=processed_lengths,
                feat_seq_offset=processed_offset,
            )
            for chunk_index, chunk, chunk_lengths, left_offset, right_offset in loader:
                capture.step = int(chunk_index)
                before_options = _state_options(streaming_state)
                before_cache_frames = int(streaming_state.spkcache.shape[1])
                before_total_frames = int(total_predictions.shape[1])
                _capture_state(f"step.{chunk_index:03d}.l6.before", streaming_state, capture)
                streaming_state, total_predictions = model.forward_streaming_step(
                    processed_signal=chunk,
                    processed_signal_length=chunk_lengths,
                    streaming_state=streaming_state,
                    total_preds=total_predictions,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
                _capture_state(f"step.{chunk_index:03d}.l6.after", streaming_state, capture)
                chunk_predictions = total_predictions[:, before_total_frames:, :]
                capture.add(f"step.{chunk_index:03d}.l5.stream_output", chunk_predictions)
                after_options = _state_options(streaming_state)
                after_cache_frames = int(streaming_state.spkcache.shape[1])
                transitions.append(
                    {
                        "step": int(chunk_index),
                        "left_offset": int(left_offset),
                        "right_offset": int(right_offset),
                        "input_feature_frames": int(chunk.shape[1]),
                        "valid_feature_frames": int(chunk_lengths.item()),
                        "output_frames": int(chunk_predictions.shape[1]),
                        "before_options": before_options,
                        "after_options": after_options,
                        "before_cache_frames": before_cache_frames,
                        "after_cache_frames": after_cache_frames,
                        "compression_transition": (
                            not before_options["spkcache_preds"]
                            and after_options["spkcache_preds"]
                        ),
                        "cache_compression": (
                            before_cache_frames + int(chunk_predictions.shape[1])
                            > model.sortformer_modules.spkcache_len
                        ),
                        "speaker_permutation_absent": not after_options["spk_perm"],
                    }
                )
    finally:
        capture.close()
    trace.update(capture.tensors)
    trace["l5.final_probabilities_f32"] = total_predictions.detach().cpu().contiguous()
    trace.update(_public_postprocessing(total_predictions.detach().cpu(), torch))
    return trace, transitions, valid_frames, physical_frames


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_public_fixture_inputs(
    public_root: Path, torch: Any
) -> tuple[list[tuple[dict[str, Any], Any]], list[dict[str, Any]], dict[str, Any]]:
    root = public_root.resolve(strict=True)
    descriptor_path = (root / "descriptor.json").resolve(strict=True)
    if descriptor_path.parent != root:
        raise RuntimeError(f"{PROFILE} public descriptor escaped its external root")
    descriptor_bytes = descriptor_path.read_bytes()
    if not hmac.compare_digest(_sha256_bytes(descriptor_bytes), PUBLIC_DESCRIPTOR_SHA256):
        raise RuntimeError(f"{PROFILE} public descriptor identity mismatch")
    descriptor = json.loads(descriptor_bytes)
    if (
        descriptor.get("schema_version") != PUBLIC_DESCRIPTOR_SCHEMA
        or descriptor.get("corpus_key") != PUBLIC_CORPUS_KEY
        or descriptor.get("source_version") != PUBLIC_SOURCE_VERSION
    ):
        raise RuntimeError(f"{PROFILE} public descriptor contract changed")
    recordings = descriptor.get("recordings")
    if not isinstance(recordings, list) or len(recordings) != 4:
        raise RuntimeError(f"{PROFILE} public recording census changed")
    by_id = {}
    for recording in recordings:
        if not isinstance(recording, dict) or not isinstance(recording.get("recording_id"), str):
            raise RuntimeError(f"{PROFILE} public recording entry is malformed")
        recording_id = recording["recording_id"]
        if recording_id in by_id:
            raise RuntimeError(f"{PROFILE} duplicate public recording identity")
        by_id[recording_id] = recording

    inputs = []
    receipts = []
    for fixture in PUBLIC_FIXTURES:
        recording = by_id.get(fixture["recording_id"])
        if recording is None:
            raise RuntimeError(f"{PROFILE} public fixture recording is absent")
        audio_relative = Path(recording["audio_path"])
        annotation_relative = Path(recording["annotation_path"])
        if audio_relative.is_absolute() or annotation_relative.is_absolute():
            raise RuntimeError(f"{PROFILE} public descriptor contains an absolute path")
        audio_path = (root / audio_relative).resolve(strict=True)
        annotation_path = (root / annotation_relative).resolve(strict=True)
        if root not in audio_path.parents or root not in annotation_path.parents:
            raise RuntimeError(f"{PROFILE} public input escaped its external root")
        audio_bytes = audio_path.read_bytes()
        annotation_bytes = annotation_path.read_bytes()
        if not hmac.compare_digest(_sha256_bytes(audio_bytes), recording["audio_sha256"]):
            raise RuntimeError(f"{PROFILE} public audio identity mismatch")
        if not hmac.compare_digest(
            _sha256_bytes(annotation_bytes), recording["annotation_sha256"]
        ):
            raise RuntimeError(f"{PROFILE} public annotation identity mismatch")
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            observed_wave = {
                "channels": wav.getnchannels(),
                "sample_width": wav.getsampwidth(),
                "sample_rate_hz": wav.getframerate(),
                "compression": wav.getcomptype(),
                "sample_count": wav.getnframes(),
            }
            if observed_wave != {
                "channels": 1,
                "sample_width": 2,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "compression": "NONE",
                "sample_count": wav.getnframes(),
            }:
                raise RuntimeError(f"{PROFILE} public WAV contract changed")
            pcm16_bytes = wav.readframes(wav.getnframes())
        if len(pcm16_bytes) != observed_wave["sample_count"] * 2:
            raise RuntimeError(f"{PROFILE} public WAV payload is truncated")
        start_sample = fixture["start_sample"]
        end_sample = start_sample + fixture["sample_count"]
        if end_sample > observed_wave["sample_count"]:
            raise RuntimeError(f"{PROFILE} public fixture exceeds its recording")
        clip_bytes = pcm16_bytes[start_sample * 2 : end_sample * 2]
        pcm_i16 = torch.frombuffer(bytearray(clip_bytes), dtype=torch.int16).clone()
        pcm = pcm_i16.to(torch.float32) / 32_768.0

        clip_start_seconds = start_sample / SAMPLE_RATE_HZ
        clip_end_seconds = end_sample / SAMPLE_RATE_HZ
        selected_intervals = []
        selected_speakers = set()
        for line in annotation_bytes.decode("utf-8").splitlines():
            fields = line.split()
            if len(fields) != 10 or fields[0] != "SPEAKER":
                raise RuntimeError(f"{PROFILE} public RTTM contract changed")
            start = float(fields[3])
            end = start + float(fields[4])
            if end <= clip_start_seconds or start >= clip_end_seconds:
                continue
            selected_intervals.append((max(start, clip_start_seconds), min(end, clip_end_seconds)))
            selected_speakers.add(fields[7])
        selected_intervals.sort()
        overlap = any(
            left[1] > right[0]
            for index, left in enumerate(selected_intervals)
            for right in selected_intervals[index + 1 :]
        )
        if len(selected_speakers) != fixture["expected_speaker_count"]:
            raise RuntimeError(f"{PROFILE} public fixture speaker census changed")
        if overlap != fixture["contains_overlap"]:
            raise RuntimeError(f"{PROFILE} public fixture overlap contract changed")

        fixture_receipt = {
            **fixture,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
            "audio_sha256": recording["audio_sha256"],
            "annotation_sha256": recording["annotation_sha256"],
            "full_recording_sample_count": observed_wave["sample_count"],
            "clip_pcm16_sha256": _sha256_bytes(clip_bytes),
            "clip_decoded_f32_sha256": _sha256_bytes(_tensor_bytes(pcm.reshape(1, -1))),
        }
        inputs.append((fixture_receipt, pcm))
        receipts.append(fixture_receipt)
    corpus = {
        "descriptor_schema": PUBLIC_DESCRIPTOR_SCHEMA,
        "descriptor_sha256": PUBLIC_DESCRIPTOR_SHA256,
        "corpus_key": PUBLIC_CORPUS_KEY,
        "source_version": PUBLIC_SOURCE_VERSION,
        "authoritative_url": PUBLIC_AUTHORITATIVE_URL,
        "license_id": PUBLIC_LICENSE_ID,
        "license_acknowledgement_id": PUBLIC_LICENSE_ACKNOWLEDGEMENT_ID,
        "retrieval_identity": PUBLIC_SOURCE_VERSION,
    }
    return inputs, receipts, corpus


def _empty_replay_metric(tensor: Any) -> dict[str, Any]:
    return {
        "dtype": _tensor_dtype(tensor)[0],
        "comparison_count": 0,
        "compared_values": 0,
        "mismatch_count": 0,
        "byte_exact": True,
        "full_value_byte_exact": True,
        "maximum_absolute": 0.0,
        "absolute_sum": 0.0,
        "squared_difference_sum": 0.0,
        "squared_scale_sum": 0.0,
    }


def _bounded_public_trace(
    trace: dict[str, Any], torch: Any
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    probes = {}
    contracts = {}
    for stage, tensor in trace.items():
        contiguous = tensor.detach().cpu().contiguous()
        full_raw = _tensor_bytes(contiguous)
        if contiguous.numel() <= PUBLIC_PROBE_ELEMENTS:
            probe = contiguous
            selection = "complete_tensor"
        else:
            flat = contiguous.reshape(-1)
            positions = (
                torch.arange(PUBLIC_PROBE_ELEMENTS, dtype=torch.int64)
                * (flat.numel() - 1)
                // (PUBLIC_PROBE_ELEMENTS - 1)
            )
            probe = torch.index_select(flat, 0, positions).contiguous()
            selection = "linear_index_endpoints_inclusive_v1"
        probes[stage] = probe
        contracts[stage] = {
            "dtype": _tensor_dtype(contiguous)[0],
            "full_shape": list(contiguous.shape),
            "full_elements": contiguous.numel(),
            "full_bytes": len(full_raw),
            "baseline_full_value_sha256": _sha256_bytes(full_raw),
            "probe_shape": list(probe.shape),
            "probe_elements": probe.numel(),
            "probe_selection": selection,
        }
    return probes, contracts


def _accumulate_replay_metric(metric: dict[str, Any], baseline: Any, replay: Any, torch: Any) -> None:
    if baseline.shape != replay.shape or baseline.dtype != replay.dtype:
        raise RuntimeError(f"{PROFILE} public oracle replay tensor contract changed")
    metric["comparison_count"] += 1
    metric["byte_exact"] = metric["byte_exact"] and hmac.compare_digest(
        _tensor_bytes(baseline), _tensor_bytes(replay)
    )
    left = baseline.to(torch.float64)
    right = replay.to(torch.float64)
    difference = torch.abs(left - right)
    metric["compared_values"] += difference.numel()
    metric["mismatch_count"] += int(torch.count_nonzero(difference).item())
    if difference.numel() != 0:
        metric["maximum_absolute"] = max(
            metric["maximum_absolute"], float(torch.max(difference).item())
        )
        metric["absolute_sum"] += float(torch.sum(difference).item())
        metric["squared_difference_sum"] += float(torch.sum(difference * difference).item())
        scale = torch.maximum(torch.abs(left), torch.abs(right))
        metric["squared_scale_sum"] += float(torch.sum(scale * scale).item())


def _margin_ceiling(value: float) -> float:
    if value == 0.0:
        return 0.0
    return 2.0 ** math.ceil(math.log2(value * 2.0))


def _finish_replay_metric(name: str, metric: dict[str, Any]) -> dict[str, Any]:
    compared = metric["compared_values"]
    mean_absolute = metric["absolute_sum"] / compared if compared else 0.0
    relative_l2 = (
        (metric["squared_difference_sum"] / metric["squared_scale_sum"]) ** 0.5
        if metric["squared_scale_sum"] > 0.0
        else 0.0
    )
    if metric["dtype"] == "i64" and not metric["byte_exact"]:
        raise RuntimeError(f"{PROFILE} public discrete oracle stage {name!r} is nondeterministic")
    return {
        "stage": name,
        "dtype": metric["dtype"],
        "run_count": len(FLOOR_THREAD_COUNTS) * FLOOR_REPETITIONS,
        "comparison_count": metric["comparison_count"],
        "compared_values": compared,
        "mismatch_count": metric["mismatch_count"],
        "byte_exact": metric["byte_exact"],
        "full_value_byte_exact": metric["full_value_byte_exact"],
        "max_abs_diff_f32_bits": _f32_bits(metric["maximum_absolute"]),
        "mean_abs_diff_f64_bits": _f64_bits(mean_absolute),
        "relative_l2_f64_bits": _f64_bits(relative_l2),
        "accepted_abs_tolerance_f32_bits": _f32_bits(_margin_ceiling(metric["maximum_absolute"])),
        "accepted_relative_l2_f64_bits": _f64_bits(_margin_ceiling(relative_l2)),
    }


def _capture_public_floor(
    model: Any, public_root: Path, torch: Any
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    inputs, fixtures, corpus = _read_public_fixture_inputs(public_root, torch)
    baseline: dict[str, Any] = {}
    baseline_transitions: dict[str, list[dict[str, Any]]] = {}
    seam_contracts: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for threads in FLOOR_THREAD_COUNTS:
        torch.set_num_threads(threads)
        for repetition in range(FLOOR_REPETITIONS):
            for fixture, pcm in inputs:
                trace, transitions, valid_frames, physical_frames = _public_neural_trace(
                    model, pcm, torch
                )
                trace, contracts = _bounded_public_trace(trace, torch)
                fixture["valid_frames"] = valid_frames
                fixture["physical_frames"] = physical_frames
                fixture["diarization_chunks"] = len(transitions)
                fixture_name = fixture["name"]
                qualified = {
                    f"fixture.{fixture_name}.{stage}": tensor for stage, tensor in trace.items()
                }
                qualified_contracts = {
                    f"fixture.{fixture_name}.{stage}": contract
                    for stage, contract in contracts.items()
                }
                if threads == FLOOR_THREAD_COUNTS[0] and repetition == 0:
                    baseline.update(qualified)
                    seam_contracts.update(qualified_contracts)
                    baseline_transitions[fixture_name] = transitions
                    for name, tensor in qualified.items():
                        metrics[name] = _empty_replay_metric(tensor)
                else:
                    if transitions != baseline_transitions[fixture_name]:
                        raise RuntimeError(f"{PROFILE} public cache transition replay changed")
                    expected_names = {
                        name for name in baseline if name.startswith(f"fixture.{fixture_name}.")
                    }
                    if set(qualified) != expected_names:
                        raise RuntimeError(f"{PROFILE} public activation stage census changed")
                    for name, tensor in qualified.items():
                        baseline_contract = seam_contracts[name]
                        replay_contract = qualified_contracts[name]
                        for field in (
                            "dtype",
                            "full_shape",
                            "full_elements",
                            "full_bytes",
                            "probe_shape",
                            "probe_elements",
                            "probe_selection",
                        ):
                            if replay_contract[field] != baseline_contract[field]:
                                raise RuntimeError(
                                    f"{PROFILE} public full seam contract changed"
                                )
                        metrics[name]["full_value_byte_exact"] = metrics[name][
                            "full_value_byte_exact"
                        ] and hmac.compare_digest(
                            replay_contract["baseline_full_value_sha256"],
                            baseline_contract["baseline_full_value_sha256"],
                        )
                        _accumulate_replay_metric(metrics[name], baseline[name], tensor, torch)
    observations = [_finish_replay_metric(name, metrics[name]) for name in sorted(metrics)]
    floor = {
        "schema_version": PUBLIC_FLOOR_SCHEMA,
        "baseline_threads": FLOOR_THREAD_COUNTS[0],
        "baseline_repetition": 0,
        "thread_counts": list(FLOOR_THREAD_COUNTS),
        "repetitions_per_thread": FLOOR_REPETITIONS,
        "comparison_rule": "fixed_baseline_against_each_replay",
        "margin_rule": "zero_if_exact_otherwise_smallest_power_of_two_at_least_twice_source_floor",
        "all_discrete_byte_exact": all(
            observation["byte_exact"]
            for observation in observations
            if observation["dtype"] == "i64"
        ),
        "observations": observations,
    }
    return baseline, fixtures, corpus, baseline_transitions, seam_contracts, floor


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


def _build_public_receipt(
    exporter_sha256: str,
    runtime: dict[str, str],
    source_identities: tuple[tuple[Any, ...], ...],
    execution: dict[str, Any],
    fixtures: list[dict[str, Any]],
    corpus: dict[str, Any],
    transitions: dict[str, list[dict[str, Any]]],
    seam_contracts: dict[str, dict[str, Any]],
    floor: dict[str, Any],
    tensors: dict[str, Any],
    package_bytes: bytes,
) -> dict[str, Any]:
    records = [_tensor_record(name, tensors[name]) for name in sorted(tensors)]
    payload_bytes = sum(record["bytes"] for record in records)
    execution = {
        **execution,
        "effective_frontend_dither": 0.0,
        "effective_frontend_pad_to": 0,
        "inference_mode": "eval_no_grad_synchronous_streaming",
        "activity_threshold_f32_bits": _f32_bits(0.5),
        "streaming_profile": PUBLIC_STREAMING_PROFILE,
        "postprocessing": {
            "onset": 0.5,
            "offset": 0.5,
            "pad_onset": 0.0,
            "pad_offset": 0.0,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
            "filter_speech_first": True,
        },
    }
    return {
        "schema_version": PUBLIC_SCHEMA,
        "canonical_json_version": "lexicographic-json-v1",
        "authority": "diagnostic_only",
        "equivalence_level": "l1_through_l8_public_source_truth_pack",
        "fixture_set": PUBLIC_FIXTURE_SET,
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
            "exporter_version": PUBLIC_EXPORTER_VERSION,
            "source_sha256": exporter_sha256,
            "conversion_helper_sha256": HELPER_SOURCE_SHA256,
        },
        "runtime": runtime,
        "source_files": [
            {"path": identity[0], "sha256": identity[-1]}
            for identity in source_identities
        ],
        "execution": execution,
        "corpus": corpus,
        "fixtures": fixtures,
        "streaming_transitions": transitions,
        "seam_contracts": [
            {"stage": stage, **seam_contracts[stage]} for stage in sorted(seam_contracts)
        ],
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
            "sha256": _sha256_bytes(package_bytes),
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
    if arguments.public_root is None:
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
        output_label = "synthetic"
    else:
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        for name, value in PUBLIC_STREAMING_PROFILE.items():
            setattr(model.sortformer_modules, name, value)
        tensors, fixtures, corpus, transitions, seam_contracts, floor = (
            _capture_public_floor(model, arguments.public_root, torch)
        )
        package_bytes = _build_deterministic_safetensors(tensors, safetensors, torch)
        receipt = _build_public_receipt(
            exporter_sha256,
            runtime,
            source_identities,
            _execution_identity(torch, numpy, helper),
            fixtures,
            corpus,
            transitions,
            seam_contracts,
            floor,
            tensors,
            package_bytes,
        )
        output_label = "public L1-L8"
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

    print(f"wrote {len(tensors)} {output_label} activation tensors")
    print(f"package bytes: {len(package_bytes)}")
    print(f"package sha256: {receipt['package']['sha256']}")
    print(f"receipt bytes: {len(receipt_bytes)}")
    print(f"receipt sha256: {hashlib.sha256(receipt_bytes).hexdigest()}")
    print(f"exporter sha256: {exporter_sha256}")
    if "all_byte_exact" in floor:
        print(f"oracle floor byte exact: {str(floor['all_byte_exact']).lower()}")
    else:
        print(
            "oracle discrete floor byte exact: "
            f"{str(floor['all_discrete_byte_exact']).lower()}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a frozen Sortformer synthetic L1 or public L1-L8 truth pack."
    )
    parser.add_argument("model", type=Path, help="pinned operator-local .nemo artifact")
    parser.add_argument("package_output", type=Path, help="new operator-local safetensors path")
    parser.add_argument("receipt_output", type=Path, help="new operator-local receipt path")
    parser.add_argument(
        "--public-root",
        type=Path,
        help="exact external VoxConverse root; enables the public L1-L8 profile",
    )
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
