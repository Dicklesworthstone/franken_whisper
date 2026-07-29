//! whisper.cpp ggml `.bin` model-file parser (f32 / f16).
//!
//! This module ports the *exact* on-disk format read by whisper.cpp's
//! `whisper_model_load()` (see `src/whisper.cpp` in the upstream repo). The
//! goal is byte-for-byte fidelity so the rest of the native engine inherits
//! whisper's weights, mel filterbank, and byte-level-BPE vocab without any
//! re-derivation.
//!
//! # File layout
//!
//! All scalars are little-endian. The file is a flat stream:
//!
//! 1. `magic` — one `u32`, must equal `0x6767_6d6c` (`"ggml"`).
//! 2. `hparams` — eleven consecutive `i32` (see [`WhisperHParams`]).
//! 3. mel filterbank — `n_mel` (`i32`), `n_fft` (`i32`), then
//!    `n_mel * n_fft` `f32` weights, row-major (`data[mel * n_fft + bin]`).
//! 4. vocab — `n_vocab_in_file` (`i32`), then that many entries each of
//!    `len` (`u32`) followed by `len` raw token bytes (byte-level BPE,
//!    already applied — decoding is concatenation).
//! 5. tensor directory — repeated until EOF: `n_dims` (`i32`),
//!    `name_len` (`i32`), `ttype` (`i32`), then `n_dims` dimension `i32`s in
//!    **ggml order** (fastest axis first / reversed row-major), then
//!    `name_len` name bytes, then the tensor payload (`n_elements * bpe`
//!    bytes, `bpe` = 4 for f32, 2 for f16). **There is no padding/alignment
//!    between entries** — whisper.cpp reads each tensor's bytes immediately
//!    after its name and loops; EOF is detected by the read of the next
//!    `n_dims`/`name_len`/`ttype` triple coming up short. We assert the
//!    parser consumes the file exactly to EOF.
//!
//! # ftype
//!
//! `hparams.ftype` selects the storage type of the "big" tensors: `0` = f32,
//! `1` = f16. Quantized formats (any other value) are rejected with a
//! structured [`FwError::Unsupported`] for now (bead bd-frp7 epic scope).
//! Note: each tensor entry *also* carries its own per-tensor `ttype`, so a
//! single file mixes f32 (e.g. biases, conv weights) and f16 (matmul
//! weights); we honour the per-tensor type when dequantizing.
//!
//! # Memory
//!
//! By default the whole file is read into a single `Vec<u8>` blob and tensor
//! entries index into it by `(byte_offset, byte_len)`. Files run up to ~3 GB.
//! bd-A14 (peak-RSS reduction): `FW_STREAM_LOAD=1` (unix) instead keeps only an
//! open handle and preads each tensor payload on demand, never allocating the
//! blob — see [`TensorSource`]. Byte-identical weights; gated default-off.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use ft_core::Float16;

use super::{GgmlDType, MelFilterbank, WhisperHParams};
use crate::error::{FwError, FwResult};

/// ggml file magic: ASCII `"ggml"` as a little-endian `u32`.
const GGML_MAGIC: u32 = 0x6767_6d6c;

/// Maximum number of tensor dimensions ggml encodes (matches upstream's
/// fixed `ne[4]`). Any `n_dims` outside `1..=GGML_MAX_DIMS` is malformed.
const GGML_MAX_DIMS: usize = 4;
/// whisper.cpp/ggml packs the quantization version into `hparams.ftype` as
/// `qnt_version * GGML_QNT_VERSION_FACTOR + base_ftype`; strip it to get the base.
const GGML_QNT_VERSION_FACTOR: i32 = 1000;

/// ggml `Q8_0` block: 32 quantized values per block.
const QK8_0: usize = 32;
/// `Q8_0` on-disk block size: one `f16` scale (2 bytes) + 32 `int8` quants.
const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;
/// ggml `Q5_0` block: 32 quantized values per block.
const QK5_0: usize = 32;
/// `Q5_0` on-disk block size: `f16` scale (2) + high-bit field (4) + nibbles (16).
const Q5_0_BLOCK_BYTES: usize = 2 + 4 + QK5_0 / 2;
/// ggml `Q4_0` block: 32 quantized values per block.
const QK4_0: usize = 32;
/// `Q4_0` on-disk block size: `f16` scale (2) + nibbles (16).
const Q4_0_BLOCK_BYTES: usize = 2 + QK4_0 / 2;
/// ggml `Q4_1` block: 32 values. `f16` scale + `f16` min + nibbles (16).
const QK4_1: usize = 32;
const Q4_1_BLOCK_BYTES: usize = 2 + 2 + QK4_1 / 2;
/// ggml `Q5_1` block: 32 values. `f16` scale + `f16` min + high-bits (4) + nibbles.
const QK5_1: usize = 32;
const Q5_1_BLOCK_BYTES: usize = 2 + 2 + 4 + QK5_1 / 2;

/// Read a little-endian `f16` at `raw[off..off+2]` as `f32`.
fn f16_at(raw: &[u8], off: usize) -> f32 {
    f32::from(Float16::from_bits(u16::from_le_bytes([
        raw[off],
        raw[off + 1],
    ])))
}

/// ggml k-quant super-block: 256 quantized values.
const QK_K: usize = 256;
/// `Q6_K` on-disk block size: low-nibbles (128) + high-2-bits (64) + int8
/// sub-scales (16) + `f16` super-scale (2).
const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
/// Packed 6-bit scale+min bytes per k-quant super-block (`Q4_K`/`Q5_K`).
const K_SCALE_SIZE: usize = 12;
/// `Q4_K` on-disk block size: `f16 d` (2) + `f16 dmin` (2) + packed 6-bit
/// scales (12) + 4-bit quants (128).
const Q4_K_BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
/// `Q5_K` on-disk block size: like `Q4_K` plus a 32-byte high-bit plane
/// (`f16 d` + `f16 dmin` + scales (12) + `qh` (32) + 4-bit quants (128)).
const Q5_K_BLOCK_BYTES: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
/// `Q3_K` on-disk block size: `hmask` (32) + 2-bit quants (64) + packed 6-bit
/// scales (12) + `f16 d` (2). No `dmin` — `Q3_K` has no per-block min.
const Q3_K_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2;
/// `Q2_K` on-disk block size: `scales[16]` (4-bit scale|4-bit min each) + 2-bit
/// quants (64) + `f16 d` (2) + `f16 dmin` (2).
const Q2_K_BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 + 2;

/// whisper.cpp `get_scale_min_k4`: unpack the 6-bit scale and 6-bit min for
/// sub-block `j` (0..8) from a k-quant super-block's packed 12-byte `scales`
/// array. Exact port (used by `Q4_K` and `Q5_K`).
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize a `Q4_K` payload to `f32`. Each 144-byte super-block is `[f16 d,
/// f16 dmin, scales[12], qs[128]]`; the 8 sub-scales/mins unpack via
/// `get_scale_min_k4` and each 32-value sub-block is `x = d*sc*nibble −
/// dmin*min`. Exact port of ggml `dequantize_row_q4_K`.
fn dequant_q4_k(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK_K;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q4_K_BLOCK_BYTES;
        let d = f16_at(raw, base);
        let dmin = f16_at(raw, base + 2);
        let scales = &raw[base + 4..base + 4 + K_SCALE_SIZE];
        let qs = &raw[base + 4 + K_SCALE_SIZE..base + Q4_K_BLOCK_BYTES];
        let mut y = b * QK_K;
        let mut q = 0; // running offset into qs (+32 per 64-value group)
        // Four 64-value groups; each consumes two 6-bit sub-scale/min pairs.
        for group in 0..4 {
            let is = group * 2;
            let (sc, m) = get_scale_min_k4(is, scales);
            let (d1, m1) = (d * f32::from(sc), dmin * f32::from(m));
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * f32::from(sc), dmin * f32::from(m));
            for l in 0..32 {
                out[y + l] = d1 * f32::from(qs[q + l] & 0x0F) - m1;
                out[y + l + 32] = d2 * f32::from(qs[q + l] >> 4) - m2;
            }
            y += 64;
            q += 32;
        }
    }
    out
}

/// Dequantize a `Q5_K` payload to `f32`. Each 176-byte super-block is `[f16 d,
/// f16 dmin, scales[12], qh[32], qs[128]]`; like `Q4_K` but each value gains a
/// 5th bit from the `qh` plane — `x = d*sc*((nibble)+(high?16:0)) − dmin*min`.
/// The high-bit selector shifts left 2 per 64-value group. Exact port of ggml
/// `dequantize_row_q5_K`.
fn dequant_q5_k(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK_K;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q5_K_BLOCK_BYTES;
        let d = f16_at(raw, base);
        let dmin = f16_at(raw, base + 2);
        let scales = &raw[base + 4..base + 4 + K_SCALE_SIZE];
        let qh = &raw[base + 4 + K_SCALE_SIZE..base + 4 + K_SCALE_SIZE + QK_K / 8];
        let ql = &raw[base + 4 + K_SCALE_SIZE + QK_K / 8..base + Q5_K_BLOCK_BYTES];
        let mut y = b * QK_K;
        // Four 64-value groups; the high-bit mask advances 2 bits per group.
        for group in 0..4 {
            let is = group * 2;
            let (u1, u2) = (1u8 << (2 * group), 1u8 << (2 * group + 1));
            let ql_off = group * 32;
            let (sc, m) = get_scale_min_k4(is, scales);
            let (d1, m1) = (d * f32::from(sc), dmin * f32::from(m));
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * f32::from(sc), dmin * f32::from(m));
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                out[y + l] = d1 * f32::from((ql[ql_off + l] & 0x0F) + hi1) - m1;
                out[y + l + 32] = d2 * f32::from((ql[ql_off + l] >> 4) + hi2) - m2;
            }
            y += 64;
        }
    }
    out
}

/// Dequantize a `Q3_K` payload to `f32`. Each 110-byte super-block is `[hmask[32],
/// qs[64] (2-bit), scales[12] (bit-shuffled 6-bit), f16 d]`; there is no per-block
/// min. The 16 signed 6-bit scales unpack via whisper.cpp's `aux[]` reshape, then
/// `x = d*(scale−32)*(2bit − (hmask-bit ? 0 : 4))`. Exact port of ggml
/// `dequantize_row_q3_K`.
fn dequant_q3_k(raw: &[u8], n_elements: usize) -> Vec<f32> {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let n_blocks = n_elements / QK_K;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q3_K_BLOCK_BYTES;
        let hmask = &raw[base..base + QK_K / 8];
        let qs = &raw[base + QK_K / 8..base + QK_K / 8 + QK_K / 4];
        let sc = &raw[base + QK_K / 8 + QK_K / 4..base + QK_K / 8 + QK_K / 4 + 12];
        let d_all = f16_at(raw, base + QK_K / 8 + QK_K / 4 + 12);

        // Reshape the 12 packed bytes into 16 signed 6-bit scales (each byte =
        // 4 low bits from aux[0..2] spliced with 2 bits from the third word).
        let a0 = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
        let a1 = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
        let tmp = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);
        let aux = [
            (a0 & KMASK2) | ((tmp & KMASK1) << 4),
            (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
            ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
            ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
        ];
        let mut scales = [0i8; 16];
        for (i, w) in aux.iter().enumerate() {
            let by = w.to_le_bytes();
            scales[i * 4] = by[0] as i8;
            scales[i * 4 + 1] = by[1] as i8;
            scales[i * 4 + 2] = by[2] as i8;
            scales[i * 4 + 3] = by[3] as i8;
        }

        let mut y = b * QK_K;
        let mut is = 0;
        // Two 128-value halves; qs advances 32 bytes between them.
        for nblk in 0..2 {
            let qoff = nblk * 32;
            for j in 0..4 {
                let shift = (j * 2) as u32;
                let m = 1u8 << (nblk * 4 + j); // high-bit selector, bit 0..7
                let dl = (i32::from(scales[is]) - 32) as f32 * d_all;
                is += 1;
                for l in 0..16 {
                    let q2 = i32::from((qs[qoff + l] >> shift) & 3);
                    let sub = if hmask[l] & m != 0 { 0 } else { 4 };
                    out[y] = dl * (q2 - sub) as f32;
                    y += 1;
                }
                let dl = (i32::from(scales[is]) - 32) as f32 * d_all;
                is += 1;
                for l in 0..16 {
                    let q2 = i32::from((qs[qoff + l + 16] >> shift) & 3);
                    let sub = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                    out[y] = dl * (q2 - sub) as f32;
                    y += 1;
                }
            }
        }
    }
    out
}

/// Dequantize a `Q2_K` payload to `f32`. Each 84-byte super-block is `[scales[16]
/// (low nibble = 4-bit scale, high nibble = 4-bit min), qs[64] (2-bit), f16 d,
/// f16 dmin]`; `x = d*(sc&0xF)*2bit − dmin*(sc>>4)`. Exact port of ggml
/// `dequantize_row_q2_K`.
fn dequant_q2_k(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK_K;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q2_K_BLOCK_BYTES;
        let scales = &raw[base..base + QK_K / 16];
        let qs = &raw[base + QK_K / 16..base + QK_K / 16 + QK_K / 4];
        let d = f16_at(raw, base + QK_K / 16 + QK_K / 4);
        let dmin = f16_at(raw, base + QK_K / 16 + QK_K / 4 + 2);
        let mut y = b * QK_K;
        let mut is = 0;
        // Two 128-value halves; qs advances 32 bytes between them.
        for nblk in 0..2 {
            let qoff = nblk * 32;
            for j in 0..4 {
                let shift = (j * 2) as u32;
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * f32::from(sc & 0x0F), dmin * f32::from(sc >> 4));
                for l in 0..16 {
                    out[y] = dl * f32::from((qs[qoff + l] >> shift) & 3) - ml;
                    y += 1;
                }
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * f32::from(sc & 0x0F), dmin * f32::from(sc >> 4));
                for l in 0..16 {
                    out[y] = dl * f32::from((qs[qoff + l + 16] >> shift) & 3) - ml;
                    y += 1;
                }
            }
        }
    }
    out
}

/// Dequantize a `Q6_K` payload to `f32`. Each 210-byte super-block is `[ql[128],
/// qh[64], int8 scales[16], f16 d]`; a 6-bit quant `(low-nibble | high-2-bits<<4)
/// − 32` is scaled by `d * scales[sub]`. Exact port of ggml `dequantize_row_q6_K`.
fn dequant_q6_k(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK_K;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q6_K_BLOCK_BYTES;
        let ql = &raw[base..base + QK_K / 2];
        let qh = &raw[base + QK_K / 2..base + QK_K / 2 + QK_K / 4];
        let sc_off = base + QK_K / 2 + QK_K / 4;
        let scales = &raw[sc_off..sc_off + QK_K / 16];
        let d = f16_at(raw, sc_off + QK_K / 16);
        let yb = b * QK_K;
        // Two 128-value chunks; ql +64, qh +32, scales +8 per chunk.
        for chunk in 0..2 {
            let (qlo, qho, sco, yo) = (chunk * 64, chunk * 32, chunk * 8, chunk * 128);
            for l in 0..32 {
                let is = l / 16;
                let q1 = (i32::from(ql[qlo + l] & 0x0F) | ((i32::from(qh[qho + l]) & 3) << 4)) - 32;
                let q2 = (i32::from(ql[qlo + l + 32] & 0x0F)
                    | ((i32::from(qh[qho + l] >> 2) & 3) << 4))
                    - 32;
                let q3 =
                    (i32::from(ql[qlo + l] >> 4) | ((i32::from(qh[qho + l] >> 4) & 3) << 4)) - 32;
                let q4 = (i32::from(ql[qlo + l + 32] >> 4)
                    | ((i32::from(qh[qho + l] >> 6) & 3) << 4))
                    - 32;
                let sc = |k: usize| f32::from(scales[sco + is + k] as i8);
                out[yb + yo + l] = d * sc(0) * q1 as f32;
                out[yb + yo + l + 32] = d * sc(2) * q2 as f32;
                out[yb + yo + l + 64] = d * sc(4) * q3 as f32;
                out[yb + yo + l + 96] = d * sc(6) * q4 as f32;
            }
        }
    }
    out
}

/// Dequantize a `Q4_1` payload: 20-byte block `[f16 d, f16 m, 16 nibbles]`;
/// `x = nibble * d + m` (ggml `dequantize_row_q4_1`).
fn dequant_q4_1(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK4_1;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q4_1_BLOCK_BYTES;
        let d = f16_at(raw, base);
        let m = f16_at(raw, base + 2);
        let qs = &raw[base + 4..base + 4 + QK4_1 / 2];
        for j in 0..QK4_1 / 2 {
            out[b * QK4_1 + j] = f32::from(qs[j] & 0x0F) * d + m;
            out[b * QK4_1 + j + QK4_1 / 2] = f32::from(qs[j] >> 4) * d + m;
        }
    }
    out
}

/// Dequantize a `Q5_1` payload: 24-byte block `[f16 d, f16 m, u32 qh, 16
/// nibbles]`; `x = ((nibble | hi<<4)) * d + m` (ggml `dequantize_row_q5_1`).
fn dequant_q5_1(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK5_1;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q5_1_BLOCK_BYTES;
        let d = f16_at(raw, base);
        let m = f16_at(raw, base + 2);
        let qh = u32::from_le_bytes([raw[base + 4], raw[base + 5], raw[base + 6], raw[base + 7]]);
        let qs = &raw[base + 8..base + 8 + QK5_1 / 2];
        for j in 0..QK5_1 / 2 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let x0 = u32::from(qs[j] & 0x0F) | xh0;
            let x1 = u32::from(qs[j] >> 4) | xh1;
            out[b * QK5_1 + j] = x0 as f32 * d + m;
            out[b * QK5_1 + j + QK5_1 / 2] = x1 as f32 * d + m;
        }
    }
    out
}

/// On-disk byte length of a tensor payload for `dtype` × `n_elements`. `Q8_0`
/// is block-based (34 bytes per 32 values) and requires a multiple-of-32 count.
fn ggml_byte_len(dtype: GgmlDType, n_elements: usize, name: &str) -> FwResult<usize> {
    let len = match dtype {
        GgmlDType::F32 => n_elements.checked_mul(4),
        GgmlDType::F16 => n_elements.checked_mul(2),
        GgmlDType::Q8_0 => {
            if !n_elements.is_multiple_of(QK8_0) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q8_0 element count {n_elements} is not a multiple of {QK8_0}"
                )));
            }
            (n_elements / QK8_0).checked_mul(Q8_0_BLOCK_BYTES)
        }
        GgmlDType::Q5_0 => {
            if !n_elements.is_multiple_of(QK5_0) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q5_0 element count {n_elements} is not a multiple of {QK5_0}"
                )));
            }
            (n_elements / QK5_0).checked_mul(Q5_0_BLOCK_BYTES)
        }
        GgmlDType::Q4_0 => {
            if !n_elements.is_multiple_of(QK4_0) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q4_0 element count {n_elements} is not a multiple of {QK4_0}"
                )));
            }
            (n_elements / QK4_0).checked_mul(Q4_0_BLOCK_BYTES)
        }
        GgmlDType::Q4_1 => {
            if !n_elements.is_multiple_of(QK4_1) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q4_1 element count {n_elements} is not a multiple of {QK4_1}"
                )));
            }
            (n_elements / QK4_1).checked_mul(Q4_1_BLOCK_BYTES)
        }
        GgmlDType::Q5_1 => {
            if !n_elements.is_multiple_of(QK5_1) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q5_1 element count {n_elements} is not a multiple of {QK5_1}"
                )));
            }
            (n_elements / QK5_1).checked_mul(Q5_1_BLOCK_BYTES)
        }
        GgmlDType::Q6_K => {
            if !n_elements.is_multiple_of(QK_K) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q6_k element count {n_elements} is not a multiple of {QK_K}"
                )));
            }
            (n_elements / QK_K).checked_mul(Q6_K_BLOCK_BYTES)
        }
        GgmlDType::Q4_K => {
            if !n_elements.is_multiple_of(QK_K) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q4_k element count {n_elements} is not a multiple of {QK_K}"
                )));
            }
            (n_elements / QK_K).checked_mul(Q4_K_BLOCK_BYTES)
        }
        GgmlDType::Q5_K => {
            if !n_elements.is_multiple_of(QK_K) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q5_k element count {n_elements} is not a multiple of {QK_K}"
                )));
            }
            (n_elements / QK_K).checked_mul(Q5_K_BLOCK_BYTES)
        }
        GgmlDType::Q3_K => {
            if !n_elements.is_multiple_of(QK_K) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q3_k element count {n_elements} is not a multiple of {QK_K}"
                )));
            }
            (n_elements / QK_K).checked_mul(Q3_K_BLOCK_BYTES)
        }
        GgmlDType::Q2_K => {
            if !n_elements.is_multiple_of(QK_K) {
                return Err(FwError::InvalidRequest(format!(
                    "tensor '{name}' q2_k element count {n_elements} is not a multiple of {QK_K}"
                )));
            }
            (n_elements / QK_K).checked_mul(Q2_K_BLOCK_BYTES)
        }
    };
    len.ok_or_else(|| FwError::InvalidRequest(format!("tensor '{name}' byte length overflow")))
}

/// Dequantize a `Q4_0` tensor payload to `f32`. Each 18-byte block is `[f16
/// scale, 16×(two 4-bit nibbles)]`; element `j` = `qs[j]` low nibble, element
/// `j+16` = high nibble, each `(nibble − 8) * scale`. Exact port of ggml
/// `dequantize_row_q4_0`.
fn dequant_q4_0(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK4_0;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q4_0_BLOCK_BYTES;
        let d = f32::from(Float16::from_bits(u16::from_le_bytes([
            raw[base],
            raw[base + 1],
        ])));
        let qs = &raw[base + 2..base + 2 + QK4_0 / 2];
        for j in 0..QK4_0 / 2 {
            let x0 = i32::from(qs[j] & 0x0F) - 8;
            let x1 = i32::from(qs[j] >> 4) - 8;
            out[b * QK4_0 + j] = x0 as f32 * d;
            out[b * QK4_0 + j + QK4_0 / 2] = x1 as f32 * d;
        }
    }
    out
}

/// Dequantize a `Q5_0` tensor payload to `f32`. Each 22-byte block is `[f16
/// scale, u32 high-bits, 16×(two 4-bit nibbles)]`; element `j` takes `qs[j]`'s
/// low nibble + high-bit `j`, element `j+16` the high nibble + high-bit `j+16`,
/// each mapped `(5-bit − 16) * scale`. Exact port of ggml `dequantize_row_q5_0`.
fn dequant_q5_0(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK5_0;
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let base = b * Q5_0_BLOCK_BYTES;
        let d = f32::from(Float16::from_bits(u16::from_le_bytes([
            raw[base],
            raw[base + 1],
        ])));
        let qh = u32::from_le_bytes([raw[base + 2], raw[base + 3], raw[base + 4], raw[base + 5]]);
        let qs = &raw[base + 6..base + 6 + QK5_0 / 2];
        for j in 0..QK5_0 / 2 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let x0 = (u32::from(qs[j] & 0x0F) | xh0) as i32 - 16;
            let x1 = (u32::from(qs[j] >> 4) | xh1) as i32 - 16;
            out[b * QK5_0 + j] = x0 as f32 * d;
            out[b * QK5_0 + j + QK5_0 / 2] = x1 as f32 * d;
        }
    }
    out
}

/// Dequantize a `Q8_0` tensor payload (`n_elements` values, `n_elements / 32`
/// blocks of `[f16 scale, 32×int8]`) to `f32`: `x = q * scale`. `raw.len()` must
/// equal `(n_elements / 32) * 34` (validated by the caller). Mirrors ggml's
/// `dequantize_row_q8_0`.
fn dequant_q8_0(raw: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / QK8_0;
    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let base = b * Q8_0_BLOCK_BYTES;
        let scale = f32::from(Float16::from_bits(u16::from_le_bytes([
            raw[base],
            raw[base + 1],
        ])));
        let qs = &raw[base + 2..base + 2 + QK8_0];
        for &q in qs {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

/// A single tensor's location and metadata within the model file.
///
/// `shape` is stored in **row-major (PyTorch) logical order**: the ggml file
/// stores dimensions reversed (fastest-moving axis `ne[0]` first), and we
/// reverse them on load so that, e.g., `decoder.token_embedding.weight`
/// reports `[n_vocab, n_state]` exactly as PyTorch would. The raw payload in
/// the blob is unchanged (still ggml/row-major-contiguous), so the flat
/// element order returned by [`GgmlModel::tensor_f32`] matches the reversed
/// shape directly.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Logical shape in row-major (PyTorch) order — the reverse of the
    /// `ne[]` order stored in the file.
    pub shape: Vec<usize>,
    /// Element storage type for this tensor (`F32` or `F16`).
    pub dtype: GgmlDType,
    /// Byte offset of the tensor payload within the model file.
    byte_offset: usize,
    /// Byte length of the tensor payload within the model file.
    byte_len: usize,
}

impl TensorEntry {
    /// Total number of elements (product of the logical shape).
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A fully parsed whisper.cpp ggml model file.
///
/// Holds the header hyper-parameters, embedded mel filterbank, byte-level
/// vocab, a name→[`TensorEntry`] directory, and the raw file bytes the
/// directory indexes into. Construct via [`GgmlModel::load`].
#[derive(Debug)]
pub struct GgmlModel {
    /// Header hyper-parameters (the eleven `i32`s after the magic).
    pub hparams: WhisperHParams,
    /// Mel filterbank embedded in the file (`n_mel x n_fft_bins`).
    pub filters: MelFilterbank,
    /// Vocab tokens as raw bytes, indexed by token id. Length is the vocab
    /// count stored *in the file* (which may be smaller than
    /// `hparams.n_vocab`; the gap is special/extra tokens synthesized by id —
    /// see [`GgmlModel::n_extra_tokens`]).
    pub vocab_tokens: Vec<Vec<u8>>,
    /// Tensor directory: tensor name → location/metadata in [`Self::source`].
    tensors: HashMap<String, TensorEntry>,
    /// Backing store for tensor payload bytes (see [`TensorSource`]).
    source: TensorSource,
}

/// Backing store for tensor payload bytes.
///
/// The default path holds the whole model file resident and every
/// [`GgmlModel::tensor_raw`] borrows a sub-slice (`Cow::Borrowed`, zero-copy).
/// The gated streaming path (`FW_STREAM_LOAD=1`, unix only) keeps only an open
/// file handle and preads each tensor payload on demand (`Cow::Owned`); it
/// never allocates the ~1.6 GB blob, cutting peak RSS (bd-A14). Weights are
/// byte-identical either way — the same file bytes reach the same dequant/quant
/// code; only where the bytes live (one resident blob vs. per-tensor pread
/// buffers) differs.
#[derive(Debug)]
enum TensorSource {
    /// Whole file resident in memory; payloads are borrowed slices.
    Resident(Vec<u8>),
    /// Open handle; payloads are pread on demand via positioned reads
    /// ([`read_exact_at`], `FileExt::read_at` on a shared handle — thread-safe
    /// with no cursor, so concurrent per-tensor loads compose).
    #[cfg(unix)]
    Streamed(std::fs::File),
}

/// Whether the gated streaming loader (`FW_STREAM_LOAD=1`) is enabled. Unix
/// only — the on-demand path relies on `FileExt::read_at` positioned reads
/// against a shared handle. Default-off; only the literal `1` enables it.
#[cfg(unix)]
fn stream_load_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FW_STREAM_LOAD").as_deref() == Ok("1"))
}

impl GgmlModel {
    /// Parse a ggml `.bin` model file from `path`.
    ///
    /// Reads the whole file into memory, validates the magic, parses the
    /// header / filterbank / vocab / tensor directory, and asserts the parse
    /// consumes the file exactly (no trailing bytes).
    ///
    /// # Errors
    ///
    /// - [`FwError::Io`] if the file cannot be read.
    /// - [`FwError::InvalidRequest`] for a bad magic, a truncated/malformed
    ///   structure, or trailing bytes after the tensor directory.
    /// - [`FwError::Unsupported`] for a quantized `ftype`.
    pub fn load(path: &Path) -> FwResult<Self> {
        // bd-A14: opt-in streaming loader preads each tensor on demand instead
        // of holding the whole ~1.6 GB file resident (peak-RSS win, default-off).
        #[cfg(unix)]
        if stream_load_enabled() {
            return Self::load_streamed(path);
        }
        let blob = read_blob_parallel(path)?;
        Self::parse(blob)
    }

    /// Parse an in-memory ggml blob (used by [`Self::load`] and tests).
    fn parse(blob: Vec<u8>) -> FwResult<Self> {
        let mut cur = Cursor::new(&blob);

        let magic = cur.read_u32()?;
        if magic != GGML_MAGIC {
            return Err(FwError::InvalidRequest(format!(
                "bad ggml magic: got {magic:#010x}, expected {GGML_MAGIC:#010x}"
            )));
        }

        let hparams = WhisperHParams {
            n_vocab: cur.read_i32()?,
            n_audio_ctx: cur.read_i32()?,
            n_audio_state: cur.read_i32()?,
            n_audio_head: cur.read_i32()?,
            n_audio_layer: cur.read_i32()?,
            n_text_ctx: cur.read_i32()?,
            n_text_state: cur.read_i32()?,
            n_text_head: cur.read_i32()?,
            n_text_layer: cur.read_i32()?,
            n_mels: cur.read_i32()?,
            ftype: cur.read_i32()?,
        };

        // ftype gates the "big tensor" storage type. whisper.cpp packs a
        // quantization VERSION into ftype as `version * GGML_QNT_VERSION_FACTOR +
        // base` (e.g. a q8_0 v2 model reports 2007, q5_0 v2 reports 2008), so
        // strip the version before mapping. Base WHISPER_FTYPE: 0=f32, 1=f16,
        // 7=q8_0, 8=q5_0 (quantized types dequantized to f32 on load). Any other
        // base is a format we don't decode yet; the per-tensor parse is the real
        // gate (it rejects unsupported GGML_TYPEs).
        let base_ftype = hparams.ftype.rem_euclid(GGML_QNT_VERSION_FACTOR);
        if !matches!(
            base_ftype,
            0 | 1 | 2 | 3 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14
        ) {
            return Err(FwError::Unsupported(format!(
                "quantized ggml ftype {} (base {base_ftype}) is not supported \
                 (only base 0=f32, 1=f16, 2=q4_0, 3=q4_1, 7=q8_0, 8=q5_0, 9=q5_1, \
                 10=q2_k, 11=q3_k, 12=q4_k, 13=q5_k, 14=q6_k)",
                hparams.ftype
            )));
        }

        // Mel filterbank.
        let n_mel = cur.read_i32()?;
        let n_fft = cur.read_i32()?;
        let n_mel = usize_from_i32(n_mel, "filters.n_mel")?;
        let n_fft_bins = usize_from_i32(n_fft, "filters.n_fft")?;
        let n_filter = n_mel
            .checked_mul(n_fft_bins)
            .ok_or_else(|| FwError::InvalidRequest("mel filterbank size overflow".to_owned()))?;
        // Clamp the capacity hint to what the remaining blob could actually
        // supply (each filter element is one 4-byte f32). A crafted header that
        // claims an absurd `n_mel * n_fft` must not force a multi-GB allocation
        // before the per-element reads reach EOF and error out.
        let filter_cap = n_filter.min(cur.remaining() / 4);
        let mut data = Vec::with_capacity(filter_cap);
        for _ in 0..n_filter {
            data.push(cur.read_f32()?);
        }
        let filters = MelFilterbank {
            n_mel,
            n_fft_bins,
            data,
        };

        // Vocab — raw byte-level BPE tokens.
        let n_vocab_file = cur.read_i32()?;
        let n_vocab_file = usize_from_i32(n_vocab_file, "file vocab count")?;
        // Clamp the capacity hint: every token costs at least its 4-byte u32
        // length prefix, so no more than `remaining / 4` tokens can possibly
        // follow. This bounds a crafted vocab count to the blob's real size.
        let vocab_cap = n_vocab_file.min(cur.remaining() / 4);
        let mut vocab_tokens = Vec::with_capacity(vocab_cap);
        for _ in 0..n_vocab_file {
            let len = cur.read_u32()? as usize;
            vocab_tokens.push(cur.read_bytes(len)?.to_vec());
        }

        // Tensor directory — loop until EOF.
        let mut tensors: HashMap<String, TensorEntry> = HashMap::new();
        loop {
            // whisper.cpp reads the next (n_dims, name_len, ttype) triple and
            // only *then* checks EOF; a clean end-of-directory is exactly the
            // point where there are no more bytes for that triple.
            if cur.at_end() {
                break;
            }
            let n_dims = cur.read_i32()?;
            let name_len = cur.read_i32()?;
            let ttype = cur.read_i32()?;

            let n_dims = usize_from_i32(n_dims, "tensor n_dims")?;
            if n_dims == 0 || n_dims > GGML_MAX_DIMS {
                return Err(FwError::InvalidRequest(format!(
                    "tensor n_dims {n_dims} out of range 1..={GGML_MAX_DIMS}"
                )));
            }
            let name_len = usize_from_i32(name_len, "tensor name length")?;

            let dtype = match ttype {
                0 => GgmlDType::F32,
                1 => GgmlDType::F16,
                2 => GgmlDType::Q4_0,
                3 => GgmlDType::Q4_1,
                6 => GgmlDType::Q5_0,
                7 => GgmlDType::Q5_1,
                8 => GgmlDType::Q8_0,
                10 => GgmlDType::Q2_K,
                11 => GgmlDType::Q3_K,
                12 => GgmlDType::Q4_K,
                13 => GgmlDType::Q5_K,
                14 => GgmlDType::Q6_K,
                other => {
                    return Err(FwError::Unsupported(format!(
                        "tensor element type {other} is not supported \
                         (only 0=f32, 1=f16, 2=q4_0, 3=q4_1, 6=q5_0, 7=q5_1, 8=q8_0, \
                         10=q2_k, 11=q3_k, 12=q4_k, 13=q5_k, 14=q6_k)"
                    )));
                }
            };

            // Dimensions are stored in ggml order (ne[0] = fastest axis).
            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ne.push(usize_from_i32(cur.read_i32()?, "tensor dimension")?);
            }

            let name_bytes = cur.read_bytes(name_len)?;
            let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| {
                FwError::InvalidRequest("tensor name is not valid UTF-8".to_owned())
            })?;

            // Reverse ggml dims → row-major (PyTorch) logical shape.
            let mut shape = ne;
            shape.reverse();

            let n_elements: usize = shape
                .iter()
                .copied()
                .try_fold(1usize, |acc, d| acc.checked_mul(d))
                .ok_or_else(|| {
                    FwError::InvalidRequest(format!("tensor '{name}' element count overflow"))
                })?;
            let byte_len = ggml_byte_len(dtype, n_elements, &name)?;

            let byte_offset = cur.pos();
            cur.skip(byte_len)?;

            tensors.insert(
                name,
                TensorEntry {
                    shape,
                    dtype,
                    byte_offset,
                    byte_len,
                },
            );
        }

        if !cur.at_end() {
            return Err(FwError::InvalidRequest(format!(
                "trailing bytes after tensor directory: {} byte(s) unconsumed",
                blob.len() - cur.pos()
            )));
        }

        Ok(Self {
            hparams,
            filters,
            vocab_tokens,
            tensors,
            source: TensorSource::Resident(blob),
        })
    }

    /// bd-A14 streaming loader (`FW_STREAM_LOAD=1`, unix): scan the directory
    /// from an open handle, **seeking over** each tensor payload instead of
    /// reading it, then retain the handle so payloads are pread on demand. This
    /// never allocates the whole-file blob, so peak RSS drops by the file size
    /// (~1.6 GB for large-v3-turbo).
    ///
    /// The scan reads only the small non-payload bytes (magic, hparams, mel
    /// filterbank, vocab, and each tensor's `n_dims`/`name_len`/`ttype`/dims/
    /// name); the large payloads are skipped with `seek`. The directory it
    /// builds is **byte-for-byte identical** to [`Self::parse`]'s (same offsets,
    /// lengths, dtypes, shapes) — asserted by `streamed_dir_matches_resident`.
    ///
    /// Errors mirror [`Self::parse`] (bad magic, malformed/truncated structure,
    /// unsupported dtype, trailing bytes).
    #[cfg(unix)]
    fn load_streamed(path: &Path) -> FwResult<Self> {
        let file = std::fs::File::open(path)?;
        let len = usize::try_from(file.metadata()?.len()).unwrap_or(usize::MAX);
        // The directory scan uses a SEPARATE buffered handle (dropped when the
        // scan ends); `file` is kept only for the on-demand payload preads.
        let scan = std::fs::File::open(path)?;
        let mut cur = StreamCursor::new(std::io::BufReader::with_capacity(1 << 20, scan), len);

        let magic = cur.read_u32()?;
        if magic != GGML_MAGIC {
            return Err(FwError::InvalidRequest(format!(
                "bad ggml magic: got {magic:#010x}, expected {GGML_MAGIC:#010x}"
            )));
        }

        let hparams = WhisperHParams {
            n_vocab: cur.read_i32()?,
            n_audio_ctx: cur.read_i32()?,
            n_audio_state: cur.read_i32()?,
            n_audio_head: cur.read_i32()?,
            n_audio_layer: cur.read_i32()?,
            n_text_ctx: cur.read_i32()?,
            n_text_state: cur.read_i32()?,
            n_text_head: cur.read_i32()?,
            n_text_layer: cur.read_i32()?,
            n_mels: cur.read_i32()?,
            ftype: cur.read_i32()?,
        };
        if hparams.ftype != 0 && hparams.ftype != 1 {
            return Err(FwError::Unsupported(format!(
                "quantized ggml ftype {} is not supported (only ftype 0=f32, 1=f16)",
                hparams.ftype
            )));
        }

        // Mel filterbank.
        let n_mel = usize_from_i32(cur.read_i32()?, "filters.n_mel")?;
        let n_fft_bins = usize_from_i32(cur.read_i32()?, "filters.n_fft")?;
        let n_filter = n_mel
            .checked_mul(n_fft_bins)
            .ok_or_else(|| FwError::InvalidRequest("mel filterbank size overflow".to_owned()))?;
        let filter_cap = n_filter.min(cur.remaining() / 4);
        let mut data = Vec::with_capacity(filter_cap);
        for _ in 0..n_filter {
            data.push(cur.read_f32()?);
        }
        let filters = MelFilterbank {
            n_mel,
            n_fft_bins,
            data,
        };

        // Vocab — raw byte-level BPE tokens.
        let n_vocab_file = usize_from_i32(cur.read_i32()?, "file vocab count")?;
        let vocab_cap = n_vocab_file.min(cur.remaining() / 4);
        let mut vocab_tokens = Vec::with_capacity(vocab_cap);
        for _ in 0..n_vocab_file {
            let len = cur.read_u32()? as usize;
            vocab_tokens.push(cur.read_vec(len)?);
        }

        // Tensor directory — loop until EOF, seeking over each payload.
        let mut tensors: HashMap<String, TensorEntry> = HashMap::new();
        loop {
            if cur.at_end() {
                break;
            }
            let n_dims = usize_from_i32(cur.read_i32()?, "tensor n_dims")?;
            let name_len = usize_from_i32(cur.read_i32()?, "tensor name length")?;
            let ttype = cur.read_i32()?;
            if n_dims == 0 || n_dims > GGML_MAX_DIMS {
                return Err(FwError::InvalidRequest(format!(
                    "tensor n_dims {n_dims} out of range 1..={GGML_MAX_DIMS}"
                )));
            }
            let dtype = match ttype {
                0 => GgmlDType::F32,
                1 => GgmlDType::F16,
                2 => GgmlDType::Q4_0,
                3 => GgmlDType::Q4_1,
                6 => GgmlDType::Q5_0,
                7 => GgmlDType::Q5_1,
                8 => GgmlDType::Q8_0,
                10 => GgmlDType::Q2_K,
                11 => GgmlDType::Q3_K,
                12 => GgmlDType::Q4_K,
                13 => GgmlDType::Q5_K,
                14 => GgmlDType::Q6_K,
                other => {
                    return Err(FwError::Unsupported(format!(
                        "tensor element type {other} is not supported \
                         (only 0=f32, 1=f16, 2=q4_0, 3=q4_1, 6=q5_0, 7=q5_1, 8=q8_0, \
                         10=q2_k, 11=q3_k, 12=q4_k, 13=q5_k, 14=q6_k)"
                    )));
                }
            };

            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ne.push(usize_from_i32(cur.read_i32()?, "tensor dimension")?);
            }

            let name_bytes = cur.read_vec(name_len)?;
            let name = String::from_utf8(name_bytes).map_err(|_| {
                FwError::InvalidRequest("tensor name is not valid UTF-8".to_owned())
            })?;

            let mut shape = ne;
            shape.reverse();

            let n_elements: usize = shape
                .iter()
                .copied()
                .try_fold(1usize, |acc, d| acc.checked_mul(d))
                .ok_or_else(|| {
                    FwError::InvalidRequest(format!("tensor '{name}' element count overflow"))
                })?;
            let byte_len = ggml_byte_len(dtype, n_elements, &name)?;

            let byte_offset = cur.pos();
            cur.skip(byte_len)?;

            tensors.insert(
                name,
                TensorEntry {
                    shape,
                    dtype,
                    byte_offset,
                    byte_len,
                },
            );
        }

        if !cur.at_end() {
            return Err(FwError::InvalidRequest(format!(
                "trailing bytes after tensor directory: {} byte(s) unconsumed",
                cur.remaining()
            )));
        }

        Ok(Self {
            hparams,
            filters,
            vocab_tokens,
            tensors,
            source: TensorSource::Streamed(file),
        })
    }

    /// Number of "extra"/special tokens synthesized by id, i.e. the gap
    /// between `hparams.n_vocab` and the vocab count stored in the file.
    ///
    /// whisper.cpp fills ids `[file_vocab, hparams.n_vocab)` with synthetic
    /// placeholder names (`[_EOT_]`, `[_SOT_]`, `[_LANG_xx]`, `[_TT_n]`,
    /// `[_extra_token_n]`, …). We don't need the *names* to decode real text
    /// (those ids never appear in transcribed text output), only to know how
    /// many ids exist beyond the file vocab; the tokenizer bead (bd-zpfy)
    /// derives the special-id *values* from `hparams`. tiny.en: file vocab
    /// 50257, `n_vocab` 51864 → 1607 extra. large-v3: 51866 vs 50257 → 1609.
    #[must_use]
    pub fn n_extra_tokens(&self) -> usize {
        (self.hparams.n_vocab.max(0) as usize).saturating_sub(self.vocab_tokens.len())
    }

    /// Tensor names in sorted order (stable iteration, e.g. for fixture dumps).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names.into_iter()
    }

    /// Look up a tensor entry by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// Decode a tensor to `(logical_shape, f32_values)`.
    ///
    /// f16 tensors are dequantized to f32 in pure safe Rust. The returned
    /// values are in the tensor's flat (row-major contiguous) order, matching
    /// the reversed logical shape.
    ///
    /// # Errors
    ///
    /// - [`FwError::InvalidRequest`] if `name` is unknown or the stored byte
    ///   length is inconsistent with the shape/dtype (corruption).
    /// Raw little-endian bytes of a tensor entry — the SINGLE byte-access choke
    /// point for every tensor accessor. On the resident source it borrows a blob
    /// sub-slice (`Cow::Borrowed`, zero-copy); on the gated streaming source
    /// (`FW_STREAM_LOAD`) it preads the payload into an owned buffer
    /// (`Cow::Owned`). `Cow` lets both share one call site with no caller churn.
    fn tensor_raw(&self, name: &str, entry: &TensorEntry) -> FwResult<Cow<'_, [u8]>> {
        match &self.source {
            TensorSource::Resident(blob) => blob
                .get(entry.byte_offset..entry.byte_offset + entry.byte_len)
                .map(Cow::Borrowed)
                .ok_or_else(|| {
                    FwError::InvalidRequest(format!("tensor '{name}' payload out of bounds"))
                }),
            #[cfg(unix)]
            TensorSource::Streamed(file) => {
                let mut buf = vec![0u8; entry.byte_len];
                read_exact_at(file, &mut buf, entry.byte_offset as u64).map_err(|e| {
                    FwError::InvalidRequest(format!("tensor '{name}' payload pread failed: {e}"))
                })?;
                Ok(Cow::Owned(buf))
            }
        }
    }

    pub fn tensor_f32(&self, name: &str) -> FwResult<(Vec<usize>, Vec<f32>)> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| FwError::InvalidRequest(format!("unknown tensor '{name}'")))?;
        let raw = self.tensor_raw(name, entry)?;

        let n_elements = entry.n_elements();
        let values = match entry.dtype {
            GgmlDType::F32 => {
                if raw.len() != n_elements * 4 {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' f32 byte length {} != {} elements * 4",
                        raw.len(),
                        n_elements
                    )));
                }
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            }
            GgmlDType::F16 => {
                if raw.len() != n_elements * 2 {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' f16 byte length {} != {} elements * 2",
                        raw.len(),
                        n_elements
                    )));
                }
                dequant_f16_parallel(&raw, n_elements)
            }
            GgmlDType::Q8_0 => {
                let expect = ggml_byte_len(GgmlDType::Q8_0, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q8_0 byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q8_0(&raw, n_elements)
            }
            GgmlDType::Q5_0 => {
                let expect = ggml_byte_len(GgmlDType::Q5_0, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q5_0 byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q5_0(&raw, n_elements)
            }
            GgmlDType::Q4_0 => {
                let expect = ggml_byte_len(GgmlDType::Q4_0, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q4_0 byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q4_0(&raw, n_elements)
            }
            GgmlDType::Q4_1 => {
                let expect = ggml_byte_len(GgmlDType::Q4_1, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q4_1 byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q4_1(&raw, n_elements)
            }
            GgmlDType::Q5_1 => {
                let expect = ggml_byte_len(GgmlDType::Q5_1, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q5_1 byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q5_1(&raw, n_elements)
            }
            GgmlDType::Q6_K => {
                let expect = ggml_byte_len(GgmlDType::Q6_K, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q6_k byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q6_k(&raw, n_elements)
            }
            GgmlDType::Q4_K => {
                let expect = ggml_byte_len(GgmlDType::Q4_K, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q4_k byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q4_k(&raw, n_elements)
            }
            GgmlDType::Q5_K => {
                let expect = ggml_byte_len(GgmlDType::Q5_K, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q5_k byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q5_k(&raw, n_elements)
            }
            GgmlDType::Q3_K => {
                let expect = ggml_byte_len(GgmlDType::Q3_K, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q3_k byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q3_k(&raw, n_elements)
            }
            GgmlDType::Q2_K => {
                let expect = ggml_byte_len(GgmlDType::Q2_K, n_elements, name)?;
                if raw.len() != expect {
                    return Err(FwError::InvalidRequest(format!(
                        "tensor '{name}' q2_k byte length {} != {expect}",
                        raw.len()
                    )));
                }
                dequant_q2_k(&raw, n_elements)
            }
        };

        Ok((entry.shape.clone(), values))
    }

    /// Borrow a tensor's **raw little-endian f16 bit patterns** as
    /// `(logical_shape, Vec<u16>)`, WITHOUT dequantizing to f32.
    ///
    /// This is the load-path accessor for the f16-resident decoder compute
    /// lever (`FRANKEN_WHISPER_NATIVE_F16_COMPUTE`): the GEMV kernel
    /// ([`super::nn::gemv_f16`]) dequantizes each weight to f32 on the fly while
    /// it multiplies, so keeping the weights as `u16` halves their resident
    /// footprint and weight-memory traffic. Each `u16` is the IEEE-754 half bit
    /// pattern in the file's native (little-endian) order; element order is the
    /// flat row-major contiguous order matching `shape` (same as
    /// [`Self::tensor_f32`]).
    ///
    /// # Errors
    ///
    /// - [`FwError::InvalidRequest`] if `name` is unknown, the stored byte
    ///   length is inconsistent with the shape (corruption), or the tensor is
    ///   stored as **f32** in the file (callers must keep f32-stored tensors on
    ///   the f32 path — there is nothing to dequantize).
    pub fn tensor_f16(&self, name: &str) -> FwResult<(Vec<usize>, Vec<u16>)> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| FwError::InvalidRequest(format!("unknown tensor '{name}'")))?;
        if entry.dtype != GgmlDType::F16 {
            return Err(FwError::InvalidRequest(format!(
                "tensor '{name}' is stored as f32, not f16; use tensor_f32 \
                 (f16-compute path applies only to f16-stored tensors)"
            )));
        }
        let raw = self.tensor_raw(name, entry)?;
        let n_elements = entry.n_elements();
        if raw.len() != n_elements * 2 {
            return Err(FwError::InvalidRequest(format!(
                "tensor '{name}' f16 byte length {} != {} elements * 2",
                raw.len(),
                n_elements
            )));
        }
        let bits: Vec<u16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        Ok((entry.shape.clone(), bits))
    }

    /// Like [`Self::tensor_f16`] but converts the raw f16 bytes DIRECTLY to
    /// `Vec<Float16>` in one PARALLEL pass — no intermediate `Vec<u16>` and no
    /// serial follow-up conversion.
    ///
    /// This is the f16-resident decoder load path. The big `[n_vocab, n_state]`
    /// token embedding (~133 MB for large-v3) dominated decoder load when it was
    /// copied twice serially (`tensor_f16` → `bits_to_halves`); a single
    /// threaded pass recovers idle memory bandwidth. Bit-identical: each
    /// `Float16` is `from_bits(le u16)` of the same byte pair in the same flat
    /// order. Errors identically to [`Self::tensor_f16`] (unknown / f32-stored /
    /// size-mismatched tensors).
    pub fn tensor_f16_halves(&self, name: &str) -> FwResult<(Vec<usize>, Vec<Float16>)> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| FwError::InvalidRequest(format!("unknown tensor '{name}'")))?;
        if entry.dtype != GgmlDType::F16 {
            return Err(FwError::InvalidRequest(format!(
                "tensor '{name}' is stored as f32, not f16; use tensor_f32 \
                 (f16-compute path applies only to f16-stored tensors)"
            )));
        }
        let raw = self.tensor_raw(name, entry)?;
        let n_elements = entry.n_elements();
        if raw.len() != n_elements * 2 {
            return Err(FwError::InvalidRequest(format!(
                "tensor '{name}' f16 byte length {} != {} elements * 2",
                raw.len(),
                n_elements
            )));
        }
        Ok((
            entry.shape.clone(),
            dequant_f16_to_halves_parallel(&raw, n_elements),
        ))
    }

    /// Borrow a tensor's raw little-endian f16 bytes (shape + `&[u8]`) with NO
    /// `Vec<u16>` copy — for a fused dequant-transpose that reads straight from
    /// the blob. Errors exactly like [`Self::tensor_f16`] (unknown / f32-stored /
    /// size-mismatched).
    pub fn tensor_f16_bytes(&self, name: &str) -> FwResult<(Vec<usize>, Cow<'_, [u8]>)> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| FwError::InvalidRequest(format!("unknown tensor '{name}'")))?;
        if entry.dtype != GgmlDType::F16 {
            return Err(FwError::InvalidRequest(format!(
                "tensor '{name}' is stored as f32, not f16"
            )));
        }
        let raw = self.tensor_raw(name, entry)?;
        let n_elements = entry.n_elements();
        if raw.len() != n_elements * 2 {
            return Err(FwError::InvalidRequest(format!(
                "tensor '{name}' f16 byte length {} != {} elements * 2",
                raw.len(),
                n_elements
            )));
        }
        Ok((entry.shape.clone(), raw))
    }
}

/// Read an entire file into one `Vec<u8>` using SEVERAL threads, each issuing
/// positioned `read_at` calls into a disjoint, contiguous band of the output
/// buffer.
///
/// The whisper model blob is up to ~1.5 GB and a single-threaded `std::fs::read`
/// is memory-bandwidth-bound — on a busy host it is the dominant cold-start cost
/// (the `parse` phase, measured ~1.36 s warm for large-v3-turbo). Splitting the
/// copy across bands recovers idle memory bandwidth. The bytes are identical to
/// `std::fs::read` (positioned reads of disjoint, exhaustively-filled ranges
/// covering `[0, len)`), so the parsed model is bit-identical. `read_at`
/// (`std::os::unix::fs::FileExt`) is SAFE Rust — no `unsafe`, unlike mmap (which
/// this `#![forbid(unsafe_code)]` crate cannot use).
#[cfg(unix)]
pub fn read_blob_parallel(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let len = usize::try_from(file.metadata()?.len()).unwrap_or(usize::MAX);
    let mut blob = vec![0u8; len];

    // Below this size the thread spawn/join costs more than the copy it saves.
    const MIN_PARALLEL: usize = 8 * 1024 * 1024;
    // Band count for the parallel blob memcpy. Default host∧32; `FW_BLOB_READ_WORKERS`
    // overrides it. The old host∧16 cap left aggregate memory bandwidth on the table:
    // on a 64-core box the 1.5 GB warm-cache turbo read (`model_parse` span, PERF_SPANS,
    // interleaved) measured 8→~195 ms, 16→~122 ms, 24→~114 ms, 32→~113 ms, 48→~112 ms —
    // so the knee is ~24-32 and 16→32 saves ~9 ms (−7.5%). Byte-identical either way
    // (disjoint bands cover exactly `[0, len)`). Clamped to `[1, host_parallelism]`;
    // on ≤16-core hosts the value is unchanged. `FW_BLOB_READ_WORKERS=16` reverts.
    let workers = std::env::var("FW_BLOB_READ_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|w| w.clamp(1, super::host_parallelism()))
        .unwrap_or_else(|| super::host_parallelism().min(32));
    if len < MIN_PARALLEL || workers < 2 {
        read_exact_at(&file, &mut blob, 0)?;
        return Ok(blob);
    }

    let band = len.div_ceil(workers);
    let file_ref = &file;
    let mut first_err: Option<std::io::Error> = None;
    std::thread::scope(|s| {
        let handles: Vec<_> = blob
            .chunks_mut(band)
            .enumerate()
            .map(|(i, chunk)| s.spawn(move || read_exact_at(file_ref, chunk, (i * band) as u64)))
            .collect();
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    first_err.get_or_insert(e);
                }
                Err(_) => {
                    first_err.get_or_insert_with(|| {
                        std::io::Error::other("model-blob reader thread panicked")
                    });
                }
            }
        }
    });
    match first_err {
        Some(e) => Err(e),
        None => Ok(blob),
    }
}

/// Non-unix fallback: positioned reads need `FileExt`, so just read serially.
#[cfg(not(unix))]
pub fn read_blob_parallel(path: &Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Fill `buf` completely from `file` starting at `offset`, looping over short
/// reads and retrying on `Interrupted`. Errors if EOF arrives before `buf` is
/// full (a truncated/raced model file).
#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut filled = 0usize;
    while filled < buf.len() {
        match file.read_at(&mut buf[filled..], offset + filled as u64) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "early EOF while reading model blob",
                ));
            }
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Convert a little-endian f16 byte stream to `Vec<Float16>` (the f16-resident
/// representation), splitting large tensors across threads. Each
/// `Float16::from_bits` is a pure bit reinterpret and chunk boundaries are
/// element-aligned, so the result is bit-identical to the serial loop regardless
/// of thread count. Mirrors [`dequant_f16_parallel`] but keeps f16 (half the
/// output bytes — a near-`memcpy`), so it scales to more workers.
fn dequant_f16_to_halves_parallel(raw: &[u8], n_elements: usize) -> Vec<Float16> {
    const PAR_THRESHOLD: usize = 1 << 20; // 1M elements: below this, serial wins.
    let serial = |bytes: &[u8], out: &mut [Float16]| {
        let (chunks, remainder) = bytes.as_chunks::<2>();
        debug_assert!(remainder.is_empty());
        for (c, o) in chunks.iter().zip(out.iter_mut()) {
            *o = Float16::from_bits(u16::from_le_bytes(*c));
        }
    };
    let mut values = vec![Float16::from_bits(0); n_elements];
    let workers = super::host_parallelism().min(16);
    if n_elements < PAR_THRESHOLD || workers < 2 {
        serial(raw, &mut values);
        return values;
    }
    let chunk = n_elements.div_ceil(workers);
    std::thread::scope(|s| {
        for (bytes, out) in raw.chunks(chunk * 2).zip(values.chunks_mut(chunk)) {
            s.spawn(move || serial(bytes, out));
        }
    });
    values
}

/// Dequantize a little-endian f16 byte stream to `f32`, splitting large
/// tensors across threads. The big matmul weights (e.g. large-v3-turbo's
/// 1.6 GB of f16) dominated model-load time when converted serially
/// (~2.4 s measured; see tests/artifacts/perf/20260605T0218Z hotspot #5).
/// Per-element conversion is pure and chunk boundaries are element-aligned,
/// so the output is bit-identical to the serial loop regardless of thread
/// count (isomorphism: same `f16_to_f32` on the same bytes in the same
/// positions).
fn dequant_f16_parallel(raw: &[u8], n_elements: usize) -> Vec<f32> {
    const PAR_THRESHOLD: usize = 1 << 20; // 1M elements: below this, serial wins.
    let serial = |bytes: &[u8], out: &mut [f32]| {
        let (chunks, remainder) = bytes.as_chunks::<2>();
        debug_assert!(remainder.is_empty());
        for (c, o) in chunks.iter().zip(out.iter_mut()) {
            *o = f16_to_f32(u16::from_le_bytes(*c));
        }
    };
    let mut values = vec![0.0f32; n_elements];
    let workers = super::host_parallelism().min(8);
    if n_elements < PAR_THRESHOLD || workers < 2 {
        serial(raw, &mut values);
        return values;
    }
    let chunk = n_elements.div_ceil(workers);
    std::thread::scope(|s| {
        for (bytes, out) in raw.chunks(chunk * 2).zip(values.chunks_mut(chunk)) {
            s.spawn(move || serial(bytes, out));
        }
    });
    values
}

/// Convert an IEEE-754 half-precision bit pattern to `f32`.
///
/// Delegates to `ft_core::Float16` (a re-export of the well-tested `half`
/// crate) per the bd-frp7 epic's "prefer half through ft-core" guidance.
/// Handles subnormals, infinities, and NaN correctly; see the unit tests for
/// the canonical bit-pattern matrix (`0x3C00`=1.0, `0x7C00`=+inf, …).
#[inline]
#[must_use]
fn f16_to_f32(bits: u16) -> f32 {
    Float16::from_bits(bits).to_f32()
}

/// Convert an `i32` count/dimension to `usize`, rejecting negatives.
fn usize_from_i32(value: i32, what: &str) -> FwResult<usize> {
    usize::try_from(value)
        .map_err(|_| FwError::InvalidRequest(format!("{what} is negative ({value})")))
}

/// Minimal little-endian byte cursor over a borrowed blob. Every read is
/// bounds-checked and surfaces a structured error on underflow instead of
/// panicking.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn at_end(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Bytes left to read from the current position. Used to clamp speculative
    /// `Vec::with_capacity` hints so a crafted header count cannot force a huge
    /// allocation before the per-element reads hit EOF.
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, len: usize) -> FwResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| FwError::InvalidRequest("read length overflow".to_owned()))?;
        let slice = self.buf.get(self.pos..end).ok_or_else(|| {
            FwError::InvalidRequest(format!(
                "unexpected end of file: needed {len} byte(s) at offset {}, have {}",
                self.pos,
                self.buf.len()
            ))
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> FwResult<()> {
        self.read_bytes(len).map(|_| ())
    }

    fn read_u32(&mut self) -> FwResult<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> FwResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_f32(&mut self) -> FwResult<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Little-endian streaming cursor over a seekable reader, mirroring [`Cursor`]'s
/// interface for the bd-A14 [`GgmlModel::load_streamed`] directory scan. Unlike
/// [`Cursor`] it never materializes the whole file: metadata is read through the
/// buffered reader and tensor payloads are [`Self::skip`]ped with a `seek`. Every
/// read is bounds-checked against the known file length so a crafted length
/// cannot force a huge allocation before hitting EOF (matching [`Cursor`]).
#[cfg(unix)]
struct StreamCursor<R: std::io::Read + std::io::Seek> {
    inner: R,
    pos: usize,
    len: usize,
}

#[cfg(unix)]
impl<R: std::io::Read + std::io::Seek> StreamCursor<R> {
    fn new(inner: R, len: usize) -> Self {
        Self { inner, pos: 0, len }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn at_end(&self) -> bool {
        self.pos >= self.len
    }

    fn remaining(&self) -> usize {
        self.len.saturating_sub(self.pos)
    }

    /// EOF-underflow error mirroring [`Cursor::read_bytes`]'s message shape.
    fn eof(&self, len: usize) -> FwError {
        FwError::InvalidRequest(format!(
            "unexpected end of file: needed {len} byte(s) at offset {}, have {}",
            self.pos, self.len
        ))
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> FwResult<()> {
        if buf.len() > self.remaining() {
            return Err(self.eof(buf.len()));
        }
        self.inner
            .read_exact(buf)
            .map_err(|_| self.eof(buf.len()))?;
        self.pos += buf.len();
        Ok(())
    }

    /// Read `len` bytes into a fresh `Vec`. The `remaining()` guard in
    /// [`Self::read_exact`] rejects an over-long `len` before allocating.
    fn read_vec(&mut self, len: usize) -> FwResult<Vec<u8>> {
        if len > self.remaining() {
            return Err(self.eof(len));
        }
        let mut v = vec![0u8; len];
        self.read_exact(&mut v)?;
        Ok(v)
    }

    /// Advance past `len` bytes without reading them (payload skip via `seek`).
    fn skip(&mut self, len: usize) -> FwResult<()> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| FwError::InvalidRequest("read length overflow".to_owned()))?;
        if end > self.len {
            return Err(self.eof(len));
        }
        self.inner
            .seek(std::io::SeekFrom::Current(len as i64))
            .map_err(|_| self.eof(len))?;
        self.pos = end;
        Ok(())
    }

    fn read_u32(&mut self) -> FwResult<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_i32(&mut self) -> FwResult<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_f32(&mut self) -> FwResult<f32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::find_model_file;

    #[test]
    fn q8_0_byte_len_and_dequant_math() {
        // Q8_0 layout: 34 bytes/block (f16 scale + 32 int8). 64 values = 2 blocks.
        assert_eq!(
            ggml_byte_len(GgmlDType::Q8_0, 64, "t").unwrap(),
            2 * Q8_0_BLOCK_BYTES
        );
        // Non-multiple-of-32 element counts are rejected.
        assert!(ggml_byte_len(GgmlDType::Q8_0, 40, "t").is_err());

        // One block: scale = 0.5 (f16), quants 0,1,2,-1,-128,127, rest 0 →
        // x = q * 0.5.
        let scale = Float16::from_f32(0.5);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        let mut qs = [0i8; QK8_0];
        qs[0] = 0;
        qs[1] = 1;
        qs[2] = 2;
        qs[3] = -1;
        qs[4] = -128;
        qs[5] = 127;
        raw.extend(qs.iter().map(|&q| q as u8));
        let out = dequant_q8_0(&raw, QK8_0);
        assert_eq!(out.len(), QK8_0);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.5);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], -0.5);
        assert_eq!(out[4], -64.0);
        assert_eq!(out[5], 63.5);
        for &v in &out[6..] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn q5_0_byte_len_and_dequant_math() {
        // Q5_0: 22 bytes/block (f16 scale + u32 high-bits + 16 nibble-bytes).
        assert_eq!(
            ggml_byte_len(GgmlDType::Q5_0, 64, "t").unwrap(),
            2 * Q5_0_BLOCK_BYTES
        );
        assert!(ggml_byte_len(GgmlDType::Q5_0, 48, "t").is_err());

        // One block, scale = 1.0. qh sets the 5th bit of elements 0 and 16.
        // qs[0] = 0x0F (elem0 low nibble 15, elem16 high nibble 0);
        // qh bit0 = 1 → elem0 5-bit = 31 → 31-16 = 15; qh bit16 = 1 → elem16 = 16 → 0.
        let scale = Float16::from_f32(1.0);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        let qh: u32 = (1 << 0) | (1 << 16);
        raw.extend_from_slice(&qh.to_le_bytes());
        let mut qs = [0u8; QK5_0 / 2];
        qs[0] = 0x0F; // elem0 nibble 15, elem16 nibble 0
        qs[1] = 0xF0; // elem1 nibble 0, elem17 nibble 15
        raw.extend_from_slice(&qs);
        let out = dequant_q5_0(&raw, QK5_0);
        assert_eq!(out.len(), QK5_0);
        assert_eq!(out[0], 15.0); // (15 | 16) - 16 = 15
        assert_eq!(out[16], 0.0); // (0 | 16) - 16 = 0
        assert_eq!(out[1], -16.0); // (0 | 0) - 16 = -16
        assert_eq!(out[17], -1.0); // (15 | 0) - 16 = -1
        // Untouched elements: nibble 0, no high bit → -16.
        assert_eq!(out[2], -16.0);
        assert_eq!(out[18], -16.0);
    }

    #[test]
    fn q4_0_byte_len_and_dequant_math() {
        // Q4_0: 18 bytes/block (f16 scale + 16 nibble-bytes). x = (nibble - 8)*d.
        assert_eq!(
            ggml_byte_len(GgmlDType::Q4_0, 64, "t").unwrap(),
            2 * Q4_0_BLOCK_BYTES
        );
        assert!(ggml_byte_len(GgmlDType::Q4_0, 33, "t").is_err());

        let scale = Float16::from_f32(2.0);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        let mut qs = [0u8; QK4_0 / 2];
        qs[0] = 0xF0; // elem0 nibble 0 -> (0-8)*2 = -16; elem16 nibble 15 -> (15-8)*2 = 14
        qs[1] = 0x08; // elem1 nibble 8 -> 0; elem17 nibble 0 -> -16
        raw.extend_from_slice(&qs);
        let out = dequant_q4_0(&raw, QK4_0);
        assert_eq!(out.len(), QK4_0);
        assert_eq!(out[0], -16.0);
        assert_eq!(out[16], 14.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[17], -16.0);
        assert_eq!(out[2], -16.0); // nibble 0 → -8*2
    }

    #[test]
    fn q4_1_and_q5_1_dequant_math() {
        // Q4_1: 20 bytes/block (f16 d, f16 m, 16 nibbles). x = nibble*d + m.
        assert_eq!(
            ggml_byte_len(GgmlDType::Q4_1, 32, "t").unwrap(),
            Q4_1_BLOCK_BYTES
        );
        let mut raw = Vec::new();
        raw.extend_from_slice(&Float16::from_f32(2.0).to_bits().to_le_bytes()); // d
        raw.extend_from_slice(&Float16::from_f32(-1.0).to_bits().to_le_bytes()); // m
        let mut qs = [0u8; QK4_1 / 2];
        qs[0] = 0x30; // elem0 nibble 0 -> 0*2-1 = -1; elem16 nibble 3 -> 3*2-1 = 5
        raw.extend_from_slice(&qs);
        let out = dequant_q4_1(&raw, QK4_1);
        assert_eq!(out[0], -1.0);
        assert_eq!(out[16], 5.0);

        // Q5_1: 24 bytes/block (f16 d, f16 m, u32 qh, 16 nibbles).
        assert_eq!(
            ggml_byte_len(GgmlDType::Q5_1, 32, "t").unwrap(),
            Q5_1_BLOCK_BYTES
        );
        let mut r2 = Vec::new();
        r2.extend_from_slice(&Float16::from_f32(1.0).to_bits().to_le_bytes()); // d
        r2.extend_from_slice(&Float16::from_f32(0.5).to_bits().to_le_bytes()); // m
        let qh: u32 = 1 << 0; // high bit of elem0
        r2.extend_from_slice(&qh.to_le_bytes());
        let mut qs2 = [0u8; QK5_1 / 2];
        qs2[0] = 0x0F; // elem0 nibble 15 + high bit -> 31; elem16 nibble 0 -> 0
        r2.extend_from_slice(&qs2);
        let out2 = dequant_q5_1(&r2, QK5_1);
        assert_eq!(out2[0], 31.0 * 1.0 + 0.5); // 31.5
        assert_eq!(out2[16], 0.0 * 1.0 + 0.5); // 0.5
    }

    #[test]
    fn q6_k_byte_len_and_dequant_math() {
        // Q6_K super-block: 210 bytes / 256 values = [ql[128], qh[64],
        // int8 scales[16], f16 d]. 256 values = 1 block.
        assert_eq!(
            ggml_byte_len(GgmlDType::Q6_K, 256, "t").unwrap(),
            Q6_K_BLOCK_BYTES
        );
        assert_eq!(
            ggml_byte_len(GgmlDType::Q6_K, 512, "t").unwrap(),
            2 * Q6_K_BLOCK_BYTES
        );
        // Non-multiple-of-256 element counts are rejected.
        assert!(ggml_byte_len(GgmlDType::Q6_K, 200, "t").is_err());

        // Hand-computed super-block. For chunk 0, l = 0 the four outputs land at
        // indices 0/32/64/96 and read scales[0]/[2]/[4]/[6], ql[0] (low→q1,
        // high→q3), ql[32] (low→q2, high→q4), and qh[0]'s four 2-bit fields.
        let mut raw = vec![0u8; Q6_K_BLOCK_BYTES];
        raw[0] = 0x21; // ql[0]:  low nibble 1, high nibble 2
        raw[32] = 0x53; // ql[32]: low nibble 3, high nibble 5
        // qh[0] at offset 128: bits[0:1]=0, [2:3]=1, [4:5]=2, [6:7]=3.
        raw[128] = 0b11_10_01_00;
        // int8 scales at offset 128+64 = 192.
        raw[192] = 1i8 as u8; // scales[0]
        raw[194] = 2i8 as u8; // scales[2]
        raw[196] = -1i8 as u8; // scales[4]
        raw[198] = 4i8 as u8; // scales[6]
        // f16 super-scale d = 0.5 at offset 192+16 = 208.
        raw[208..210].copy_from_slice(&Float16::from_f32(0.5).to_bits().to_le_bytes());

        let out = dequant_q6_k(&raw, QK_K);
        assert_eq!(out.len(), QK_K);
        // q1 = (1 | 0<<4) - 32 = -31 → 0.5 * 1  * -31 = -15.5
        assert_eq!(out[0], -15.5);
        // q2 = (3 | 1<<4) - 32 = -13 → 0.5 * 2  * -13 = -13.0
        assert_eq!(out[32], -13.0);
        // q3 = (2 | 2<<4) - 32 =   2 → 0.5 * -1 *   2 =  -1.0
        assert_eq!(out[64], -1.0);
        // q4 = (5 | 3<<4) - 32 =  21 → 0.5 * 4  *  21 =  42.0
        assert_eq!(out[96], 42.0);
    }

    #[test]
    fn q4_k_byte_len_and_dequant_math() {
        // Q4_K super-block: 144 bytes / 256 values = [f16 d, f16 dmin,
        // scales[12] (6-bit packed), qs[128] (4-bit)]. 256 values = 1 block.
        assert_eq!(
            ggml_byte_len(GgmlDType::Q4_K, 256, "t").unwrap(),
            Q4_K_BLOCK_BYTES
        );
        assert_eq!(
            ggml_byte_len(GgmlDType::Q4_K, 512, "t").unwrap(),
            2 * Q4_K_BLOCK_BYTES
        );
        assert!(ggml_byte_len(GgmlDType::Q4_K, 200, "t").is_err());

        // get_scale_min_k4 in isolation: j<4 reads scales[j] & scales[j+4] (low
        // 6 bits); j>=4 splices the high 2 bits of earlier bytes into bits 4-5.
        let sc = [2u8, 3, 0, 0, 1, 4, 0, 0, 0x25, 0x31, 0, 0];
        assert_eq!(get_scale_min_k4(0, &sc), (2, 1)); // scales[0]&63, scales[4]&63
        assert_eq!(get_scale_min_k4(1, &sc), (3, 4)); // scales[1]&63, scales[5]&63
        // j=4: d=(scales[8]&0xF)|((scales[0]>>6)<<4)=5|0, m=(scales[8]>>4)|((scales[4]>>6)<<4)=2|0
        assert_eq!(get_scale_min_k4(4, &sc), (5, 2));
        // j=5: d=(scales[9]&0xF)|((scales[1]>>6)<<4)=1|0, m=(scales[9]>>4)|((scales[5]>>6)<<4)=3|0
        assert_eq!(get_scale_min_k4(5, &sc), (1, 3));

        // Full super-block. d=1.0, dmin=0.5; the scales above; two 4-bit quants.
        let mut raw = vec![0u8; Q4_K_BLOCK_BYTES];
        raw[0..2].copy_from_slice(&Float16::from_f32(1.0).to_bits().to_le_bytes()); // d
        raw[2..4].copy_from_slice(&Float16::from_f32(0.5).to_bits().to_le_bytes()); // dmin
        raw[4..16].copy_from_slice(&sc); // scales[12]
        // qs[0] at offset 16: low nibble 5 → out[0], high nibble 0xA=10 → out[32].
        raw[16] = 0xA5;
        // qs[64] at offset 16+64=80 (group 2, is=4/5): low 4 → out[128], high 2 → out[160].
        raw[80] = 0x24;

        let out = dequant_q4_k(&raw, QK_K);
        assert_eq!(out.len(), QK_K);
        // Group 0: d1 = 1*2 = 2, m1 = 0.5*1 = 0.5; d2 = 1*3 = 3, m2 = 0.5*4 = 2.
        assert_eq!(out[0], 2.0 * 5.0 - 0.5); // 9.5
        assert_eq!(out[32], 3.0 * 10.0 - 2.0); // 28.0
        // Group 2: d1 = 1*5 = 5, m1 = 0.5*2 = 1; d2 = 1*1 = 1, m2 = 0.5*3 = 1.5.
        assert_eq!(out[128], 5.0 * 4.0 - 1.0); // 19.0
        assert_eq!(out[160], 1.0 * 2.0 - 1.5); // 0.5
    }

    #[test]
    fn q5_k_byte_len_and_dequant_math() {
        // Q5_K super-block: 176 bytes / 256 values = [f16 d, f16 dmin,
        // scales[12], qh[32] (high bit), qs[128] (low 4-bit)].
        assert_eq!(
            ggml_byte_len(GgmlDType::Q5_K, 256, "t").unwrap(),
            Q5_K_BLOCK_BYTES
        );
        assert_eq!(
            ggml_byte_len(GgmlDType::Q5_K, 512, "t").unwrap(),
            2 * Q5_K_BLOCK_BYTES
        );
        assert!(ggml_byte_len(GgmlDType::Q5_K, 200, "t").is_err());

        // Same scales as q4_k (get_scale_min_k4 already pinned there); this pins
        // the NEW 5th-bit plane: bit (2*group) sets the low half, (2*group+1) the
        // high half, so qh[l]=0x11 lights group 0's low half (bit0) and group 2's
        // low half (bit4).
        let sc = [2u8, 3, 0, 0, 1, 4, 0, 0, 0x25, 0x31, 0, 0];
        let mut raw = vec![0u8; Q5_K_BLOCK_BYTES];
        raw[0..2].copy_from_slice(&Float16::from_f32(1.0).to_bits().to_le_bytes()); // d
        raw[2..4].copy_from_slice(&Float16::from_f32(0.5).to_bits().to_le_bytes()); // dmin
        raw[4..16].copy_from_slice(&sc);
        raw[16] = 0x11; // qh[0]: bit0 (group0 low) + bit4 (group2 low)
        raw[48] = 0xA5; // ql[0]:  low nibble 5, high nibble 0xA=10
        raw[48 + 64] = 0x24; // ql[64] (group 2): low nibble 4, high nibble 2

        let out = dequant_q5_k(&raw, QK_K);
        assert_eq!(out.len(), QK_K);
        // Group 0: d1=2,m1=0.5,d2=3,m2=2. out[0]: nibble 5 + high(bit0=1)*16 = 21.
        assert_eq!(out[0], 2.0 * 21.0 - 0.5); // 41.5
        // out[32]: nibble 10 + high(bit1=0)*16 = 10.
        assert_eq!(out[32], 3.0 * 10.0 - 2.0); // 28.0
        // Group 2: d1=5,m1=1,d2=1,m2=1.5. out[128]: nibble 4 + high(bit4=1)*16 = 20.
        assert_eq!(out[128], 5.0 * 20.0 - 1.0); // 99.0
        // out[160]: nibble 2 + high(bit5=0)*16 = 2.
        assert_eq!(out[160], 1.0 * 2.0 - 1.5); // 0.5
    }

    #[test]
    fn q3_k_byte_len_and_dequant_math() {
        // Q3_K super-block: 110 bytes / 256 values = [hmask[32], qs[64] (2-bit),
        // scales[12] (bit-shuffled 6-bit), f16 d].
        assert_eq!(
            ggml_byte_len(GgmlDType::Q3_K, 256, "t").unwrap(),
            Q3_K_BLOCK_BYTES
        );
        assert_eq!(
            ggml_byte_len(GgmlDType::Q3_K, 512, "t").unwrap(),
            2 * Q3_K_BLOCK_BYTES
        );
        assert!(ggml_byte_len(GgmlDType::Q3_K, 200, "t").is_err());

        // Hand-computed block. d = 1.0. Scale reshape: scales[i] byte from aux[0]
        // = (sc[i] & 0x0f) | ((sc[8+i] & 3) << 4). Pick:
        //   scales[0] = 2 | (2<<4) = 34 → dl = 34-32 = 2
        //   scales[1] = 3 | (1<<4) = 19 → dl = 19-32 = -13
        //   scales[2] = 2 | (2<<4) = 34 → dl = 2
        let mut raw = vec![0u8; Q3_K_BLOCK_BYTES];
        // hmask[32] at offset 0.
        raw[0] = 1; // hmask[0]: bit0 set  → value 0 keeps its 2 bits (sub 0)
        raw[1] = 0; // hmask[1]: bit0 clear → value 1 gets sub 4
        raw[16] = 1; // hmask[16]: bit0 set → 2nd-inner value keeps its bits
        // qs[64] at offset 32 (2-bit quants).
        raw[32] = 3; // qs[0]  = 0b11
        raw[33] = 1; // qs[1]  = 0b01
        raw[48] = 2; // qs[16] = 0b10
        // scales[12] at offset 96.
        raw[96] = 2; // sc[0]
        raw[97] = 3; // sc[1]
        raw[98] = 2; // sc[2]
        raw[104] = 2; // sc[8]  → scales[0] high bits
        raw[105] = 1; // sc[9]  → scales[1] high bits
        raw[106] = 2; // sc[10] → scales[2] high bits
        // f16 d = 1.0 at offset 108.
        raw[108..110].copy_from_slice(&Float16::from_f32(1.0).to_bits().to_le_bytes());

        let out = dequant_q3_k(&raw, QK_K);
        assert_eq!(out.len(), QK_K);
        // is=0, shift=0, m=1: dl=2, qs[0]&3=3, hmask[0]&1 set → sub 0.
        assert_eq!(out[0], 2.0 * (3.0 - 0.0)); // 6.0
        // l=1: qs[1]&3=1, hmask[1]&1 clear → sub 4.
        assert_eq!(out[1], 2.0 * (1.0 - 4.0)); // -6.0
        // is=1 (2nd inner loop), dl=-13, qs[16]&3=2, hmask[16]&1 set → sub 0.
        assert_eq!(out[16], -13.0 * (2.0 - 0.0)); // -26.0
        // is=2 (j=1, shift=2, m=2): dl=2, qs[0]>>2&3=0, hmask[0]&2 clear → sub 4.
        assert_eq!(out[32], 2.0 * (0.0 - 4.0)); // -8.0
    }

    #[test]
    fn q2_k_byte_len_and_dequant_math() {
        // Q2_K super-block: 84 bytes / 256 values = [scales[16] (4-bit scale |
        // 4-bit min), qs[64] (2-bit), f16 d, f16 dmin].
        assert_eq!(
            ggml_byte_len(GgmlDType::Q2_K, 256, "t").unwrap(),
            Q2_K_BLOCK_BYTES
        );
        assert_eq!(
            ggml_byte_len(GgmlDType::Q2_K, 512, "t").unwrap(),
            2 * Q2_K_BLOCK_BYTES
        );
        assert!(ggml_byte_len(GgmlDType::Q2_K, 200, "t").is_err());

        // Hand-computed block. d=1.0, dmin=0.5. Each scales byte = scale | min<<4.
        let mut raw = vec![0u8; Q2_K_BLOCK_BYTES];
        raw[0] = 0x12; // scales[0]: scale 2, min 1  → dl=2, ml=0.5
        raw[1] = 0x43; // scales[1]: scale 3, min 4  → dl=3, ml=2
        raw[2] = 0x05; // scales[2]: scale 5, min 0  → dl=5, ml=0
        // qs[64] at offset 16.
        raw[16] = 0x0B; // qs[0]  = 0b1011 → shift0: 0b11=3, shift2: 0b10=2
        raw[32] = 0x02; // qs[16] = 0b10 = 2
        // f16 d = 1.0 at offset 80, dmin = 0.5 at offset 82.
        raw[80..82].copy_from_slice(&Float16::from_f32(1.0).to_bits().to_le_bytes());
        raw[82..84].copy_from_slice(&Float16::from_f32(0.5).to_bits().to_le_bytes());

        let out = dequant_q2_k(&raw, QK_K);
        assert_eq!(out.len(), QK_K);
        // is=0, shift=0: dl=2, ml=0.5, qs[0]&3=3 → 2*3 - 0.5.
        assert_eq!(out[0], 2.0 * 3.0 - 0.5); // 5.5
        // is=1 (2nd inner loop), dl=3, ml=2, qs[16]&3=2 → 3*2 - 2.
        assert_eq!(out[16], 3.0 * 2.0 - 2.0); // 4.0
        // is=2 (j=1, shift=2): dl=5, ml=0, qs[0]>>2&3=2 → 5*2 - 0.
        assert_eq!(out[32], 5.0 * 2.0 - 0.0); // 10.0
    }

    #[test]
    fn gated_q2_k_model_loads_and_dequants_close_to_f16() {
        // Requires ggml-tiny.en-q2_k.bin (`whisper-quantize <f16> <out> q2_k`).
        let (Some(q2_path), Some(f16_path)) =
            (find_model_file("tiny.en-q2_k"), find_model_file("tiny.en"))
        else {
            eprintln!("SKIP gated_q2_k: ggml-tiny.en-q2_k.bin or ggml-tiny.en.bin not found");
            return;
        };
        let q2 = GgmlModel::load(&q2_path).expect("load q2_k model");
        let f16 = GgmlModel::load(&f16_path).expect("load f16 model");
        assert_eq!(
            q2.hparams.ftype.rem_euclid(1000),
            10,
            "q2_k model base ftype must be 10"
        );
        let q2_name = q2
            .tensor_names()
            .find(|n| q2.tensor(n).is_some_and(|e| e.dtype == GgmlDType::Q2_K))
            .expect("q2_k model has at least one Q2_K tensor")
            .to_owned();
        let (qshape, qvals) = q2.tensor_f32(&q2_name).expect("q2_k dequant");
        let (fshape, fvals) = f16.tensor_f32(&q2_name).expect("f16 dequant");
        assert_eq!(qshape, fshape, "shape mismatch for '{q2_name}'");
        assert!(
            qvals.iter().all(|v| v.is_finite()),
            "q2_k dequant produced non-finite values"
        );
        let n = fvals.len() as f32;
        let mean_abs: f32 = fvals.iter().map(|v| v.abs()).sum::<f32>() / n;
        let mean_abs_diff: f32 = qvals
            .iter()
            .zip(&fvals)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n;
        // q2_k is the coarsest quant (2-bit): the loosest bound, ~40% of f16.
        // (The per-element dequant is byte-verified against an independent
        // reference implementation — this bound just documents 2-bit's lossiness.)
        assert!(
            mean_abs_diff < 0.40 * mean_abs,
            "q2_k vs f16 mean|Δ| {mean_abs_diff} exceeds 40% of mean|w| {mean_abs} for '{q2_name}'"
        );
    }

    #[test]
    fn gated_q3_k_model_loads_and_dequants_close_to_f16() {
        // Requires ggml-tiny.en-q3_k.bin (`whisper-quantize <f16> <out> q3_k`).
        let (Some(q3_path), Some(f16_path)) =
            (find_model_file("tiny.en-q3_k"), find_model_file("tiny.en"))
        else {
            eprintln!("SKIP gated_q3_k: ggml-tiny.en-q3_k.bin or ggml-tiny.en.bin not found");
            return;
        };
        let q3 = GgmlModel::load(&q3_path).expect("load q3_k model");
        let f16 = GgmlModel::load(&f16_path).expect("load f16 model");
        assert_eq!(
            q3.hparams.ftype.rem_euclid(1000),
            11,
            "q3_k model base ftype must be 11"
        );
        let q3_name = q3
            .tensor_names()
            .find(|n| q3.tensor(n).is_some_and(|e| e.dtype == GgmlDType::Q3_K))
            .expect("q3_k model has at least one Q3_K tensor")
            .to_owned();
        let (qshape, qvals) = q3.tensor_f32(&q3_name).expect("q3_k dequant");
        let (fshape, fvals) = f16.tensor_f32(&q3_name).expect("f16 dequant");
        assert_eq!(qshape, fshape, "shape mismatch for '{q3_name}'");
        assert!(
            qvals.iter().all(|v| v.is_finite()),
            "q3_k dequant produced non-finite values"
        );
        let n = fvals.len() as f32;
        let mean_abs: f32 = fvals.iter().map(|v| v.abs()).sum::<f32>() / n;
        let mean_abs_diff: f32 = qvals
            .iter()
            .zip(&fvals)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n;
        // q3_k is a coarse 3-bit k-quant: within ~25% of f16.
        assert!(
            mean_abs_diff < 0.25 * mean_abs,
            "q3_k vs f16 mean|Δ| {mean_abs_diff} exceeds 25% of mean|w| {mean_abs} for '{q3_name}'"
        );
    }

    #[test]
    fn gated_q5_k_model_loads_and_dequants_close_to_f16() {
        // Requires ggml-tiny.en-q5_k.bin (`whisper-quantize <f16> <out> q5_k`).
        let (Some(q5_path), Some(f16_path)) =
            (find_model_file("tiny.en-q5_k"), find_model_file("tiny.en"))
        else {
            eprintln!("SKIP gated_q5_k: ggml-tiny.en-q5_k.bin or ggml-tiny.en.bin not found");
            return;
        };
        let q5 = GgmlModel::load(&q5_path).expect("load q5_k model");
        let f16 = GgmlModel::load(&f16_path).expect("load f16 model");
        assert_eq!(
            q5.hparams.ftype.rem_euclid(1000),
            13,
            "q5_k model base ftype must be 13"
        );
        let q5_name = q5
            .tensor_names()
            .find(|n| q5.tensor(n).is_some_and(|e| e.dtype == GgmlDType::Q5_K))
            .expect("q5_k model has at least one Q5_K tensor")
            .to_owned();
        let (qshape, qvals) = q5.tensor_f32(&q5_name).expect("q5_k dequant");
        let (fshape, fvals) = f16.tensor_f32(&q5_name).expect("f16 dequant");
        assert_eq!(qshape, fshape, "shape mismatch for '{q5_name}'");
        assert!(
            qvals.iter().all(|v| v.is_finite()),
            "q5_k dequant produced non-finite values"
        );
        let n = fvals.len() as f32;
        let mean_abs: f32 = fvals.iter().map(|v| v.abs()).sum::<f32>() / n;
        let mean_abs_diff: f32 = qvals
            .iter()
            .zip(&fvals)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n;
        // q5_k is a 5-bit k-quant: within ~10% of f16 (between q6_k's 5% and q4_k's 15%).
        assert!(
            mean_abs_diff < 0.10 * mean_abs,
            "q5_k vs f16 mean|Δ| {mean_abs_diff} exceeds 10% of mean|w| {mean_abs} for '{q5_name}'"
        );
    }

    #[test]
    fn gated_q4_k_model_loads_and_dequants_close_to_f16() {
        // Requires ggml-tiny.en-q4_k.bin (`whisper-quantize <f16> <out> q4_k`).
        let (Some(q4_path), Some(f16_path)) =
            (find_model_file("tiny.en-q4_k"), find_model_file("tiny.en"))
        else {
            eprintln!("SKIP gated_q4_k: ggml-tiny.en-q4_k.bin or ggml-tiny.en.bin not found");
            return;
        };
        let q4 = GgmlModel::load(&q4_path).expect("load q4_k model");
        let f16 = GgmlModel::load(&f16_path).expect("load f16 model");
        assert_eq!(
            q4.hparams.ftype.rem_euclid(1000),
            12,
            "q4_k model base ftype must be 12"
        );
        let q4_name = q4
            .tensor_names()
            .find(|n| q4.tensor(n).is_some_and(|e| e.dtype == GgmlDType::Q4_K))
            .expect("q4_k model has at least one Q4_K tensor")
            .to_owned();
        let (qshape, qvals) = q4.tensor_f32(&q4_name).expect("q4_k dequant");
        let (fshape, fvals) = f16.tensor_f32(&q4_name).expect("f16 dequant");
        assert_eq!(qshape, fshape, "shape mismatch for '{q4_name}'");
        assert!(
            qvals.iter().all(|v| v.is_finite()),
            "q4_k dequant produced non-finite values"
        );
        let n = fvals.len() as f32;
        let mean_abs: f32 = fvals.iter().map(|v| v.abs()).sum::<f32>() / n;
        let mean_abs_diff: f32 = qvals
            .iter()
            .zip(&fvals)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n;
        // q4_k is a 4-bit k-quant (block scale+min, per-sub-block): within ~15%.
        assert!(
            mean_abs_diff < 0.15 * mean_abs,
            "q4_k vs f16 mean|Δ| {mean_abs_diff} exceeds 15% of mean|w| {mean_abs} for '{q4_name}'"
        );
    }

    #[test]
    fn gated_q6_k_model_loads_and_dequants_close_to_f16() {
        // Requires ggml-tiny.en-q6_k.bin (`whisper-quantize <f16> <out> q6_k`).
        // q6_k is a k-quant super-block format (256-value blocks).
        let (Some(q6_path), Some(f16_path)) =
            (find_model_file("tiny.en-q6_k"), find_model_file("tiny.en"))
        else {
            eprintln!("SKIP gated_q6_k: ggml-tiny.en-q6_k.bin or ggml-tiny.en.bin not found");
            return;
        };
        let q6 = GgmlModel::load(&q6_path).expect("load q6_k model");
        let f16 = GgmlModel::load(&f16_path).expect("load f16 model");
        assert_eq!(
            q6.hparams.ftype.rem_euclid(1000),
            14,
            "q6_k model base ftype must be 14"
        );
        let q6_name = q6
            .tensor_names()
            .find(|n| q6.tensor(n).is_some_and(|e| e.dtype == GgmlDType::Q6_K))
            .expect("q6_k model has at least one Q6_K tensor")
            .to_owned();
        let (qshape, qvals) = q6.tensor_f32(&q6_name).expect("q6_k dequant");
        let (fshape, fvals) = f16.tensor_f32(&q6_name).expect("f16 dequant");
        assert_eq!(qshape, fshape, "shape mismatch for '{q6_name}'");
        assert!(
            qvals.iter().all(|v| v.is_finite()),
            "q6_k dequant produced non-finite values"
        );
        let n = fvals.len() as f32;
        let mean_abs: f32 = fvals.iter().map(|v| v.abs()).sum::<f32>() / n;
        let mean_abs_diff: f32 = qvals
            .iter()
            .zip(&fvals)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n;
        // q6_k is high-precision (6-bit + per-16 sub-scales): within ~5% of f16.
        assert!(
            mean_abs_diff < 0.05 * mean_abs,
            "q6_k vs f16 mean|Δ| {mean_abs_diff} exceeds 5% of mean|w| {mean_abs} for '{q6_name}'"
        );
    }

    #[test]
    fn gated_q4_1_and_q5_1_models_load_and_dequant_close_to_f16() {
        for (name, dtype, tol) in [
            ("tiny.en-q4_1", GgmlDType::Q4_1, 0.15f32),
            ("tiny.en-q5_1", GgmlDType::Q5_1, 0.10f32),
        ] {
            let (Some(qpath), Some(f16_path)) = (find_model_file(name), find_model_file("tiny.en"))
            else {
                eprintln!("SKIP gated {name}: model not found");
                continue;
            };
            let q = GgmlModel::load(&qpath).unwrap_or_else(|e| panic!("load {name}: {e}"));
            let f16 = GgmlModel::load(&f16_path).expect("load f16");
            let tname = q
                .tensor_names()
                .find(|n| q.tensor(n).is_some_and(|e| e.dtype == dtype))
                .unwrap_or_else(|| panic!("{name} has no {dtype:?} tensor"))
                .to_owned();
            let (qs, qv) = q.tensor_f32(&tname).expect("q dequant");
            let (fs, fv) = f16.tensor_f32(&tname).expect("f16 dequant");
            assert_eq!(qs, fs, "shape mismatch {tname}");
            assert!(qv.iter().all(|v| v.is_finite()), "{name} non-finite");
            let n = fv.len() as f32;
            let ma: f32 = fv.iter().map(|v| v.abs()).sum::<f32>() / n;
            let md: f32 = qv.iter().zip(&fv).map(|(a, b)| (a - b).abs()).sum::<f32>() / n;
            assert!(
                md < tol * ma,
                "{name} mean|Δ| {md} exceeds {tol} of {ma} ({tname})"
            );
        }
    }

    #[test]
    fn gated_q4_0_model_loads_and_dequants_close_to_f16() {
        // Requires ggml-tiny.en-q4_0.bin (`whisper-quantize <f16> <out> q4_0`).
        let (Some(q4_path), Some(f16_path)) =
            (find_model_file("tiny.en-q4_0"), find_model_file("tiny.en"))
        else {
            eprintln!("SKIP gated_q4_0: ggml-tiny.en-q4_0.bin or ggml-tiny.en.bin not found");
            return;
        };
        let q4 = GgmlModel::load(&q4_path).expect("load q4_0 model");
        let f16 = GgmlModel::load(&f16_path).expect("load f16 model");
        assert_eq!(
            q4.hparams.ftype.rem_euclid(1000),
            2,
            "q4_0 model base ftype must be 2"
        );
        let q4_name = q4
            .tensor_names()
            .find(|n| q4.tensor(n).is_some_and(|e| e.dtype == GgmlDType::Q4_0))
            .expect("q4_0 model has at least one Q4_0 tensor")
            .to_owned();
        let (qshape, qvals) = q4.tensor_f32(&q4_name).expect("q4_0 dequant");
        let (fshape, fvals) = f16.tensor_f32(&q4_name).expect("f16 dequant");
        assert_eq!(qshape, fshape, "shape mismatch for '{q4_name}'");
        assert!(
            qvals.iter().all(|v| v.is_finite()),
            "q4_0 dequant produced non-finite values"
        );
        let n = fvals.len() as f32;
        let mean_abs: f32 = fvals.iter().map(|v| v.abs()).sum::<f32>() / n;
        let mean_abs_diff: f32 = qvals
            .iter()
            .zip(&fvals)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n;
        // 4-bit is the coarsest of the three; still tracks f16 within ~15%.
        assert!(
            mean_abs_diff < 0.15 * mean_abs,
            "q4_0 vs f16 mean|Δ| {mean_abs_diff} exceeds 15% of mean|w| {mean_abs} for '{q4_name}'"
        );
    }

    #[test]
    fn gated_q5_0_model_loads_and_dequants_close_to_f16() {
        // Requires ggml-tiny.en-q5_0.bin (`whisper-quantize <f16> <out> q5_0`).
        let (Some(q5_path), Some(f16_path)) =
            (find_model_file("tiny.en-q5_0"), find_model_file("tiny.en"))
        else {
            eprintln!("SKIP gated_q5_0: ggml-tiny.en-q5_0.bin or ggml-tiny.en.bin not found");
            return;
        };
        let q5 = GgmlModel::load(&q5_path).expect("load q5_0 model");
        let f16 = GgmlModel::load(&f16_path).expect("load f16 model");
        assert_eq!(
            q5.hparams.ftype.rem_euclid(1000),
            8,
            "q5_0 model base ftype must be 8"
        );
        let q5_name = q5
            .tensor_names()
            .find(|n| q5.tensor(n).is_some_and(|e| e.dtype == GgmlDType::Q5_0))
            .expect("q5_0 model has at least one Q5_0 tensor")
            .to_owned();
        let (qshape, qvals) = q5.tensor_f32(&q5_name).expect("q5_0 dequant");
        let (fshape, fvals) = f16.tensor_f32(&q5_name).expect("f16 dequant");
        assert_eq!(qshape, fshape, "shape mismatch for '{q5_name}'");
        assert!(
            qvals.iter().all(|v| v.is_finite()),
            "q5_0 dequant produced non-finite values"
        );
        let n = fvals.len() as f32;
        let mean_abs: f32 = fvals.iter().map(|v| v.abs()).sum::<f32>() / n;
        let mean_abs_diff: f32 = qvals
            .iter()
            .zip(&fvals)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n;
        // Q5_0 is coarser than Q8_0 (5-bit) but still tracks f16 within ~10%.
        assert!(
            mean_abs_diff < 0.10 * mean_abs,
            "q5_0 vs f16 mean|Δ| {mean_abs_diff} exceeds 10% of mean|w| {mean_abs} for '{q5_name}'"
        );
    }

    #[test]
    fn gated_q8_0_model_loads_and_dequants_close_to_f16() {
        // Requires a q8_0-quantized tiny.en next to the f16 one (produce it with
        // whisper.cpp's `whisper-quantize <f16> <out> q8_0`). Validates that a
        // real whisper.cpp q8_0 model parses AND that a Q8_0 tensor dequantizes
        // to values that track the f16 model's same weights (Q8_0 is int8 × f16
        // scale — high precision).
        let (Some(q8_path), Some(f16_path)) =
            (find_model_file("tiny.en-q8_0"), find_model_file("tiny.en"))
        else {
            eprintln!("SKIP gated_q8_0: ggml-tiny.en-q8_0.bin or ggml-tiny.en.bin not found");
            return;
        };
        let q8 = GgmlModel::load(&q8_path).expect("load q8_0 model");
        let f16 = GgmlModel::load(&f16_path).expect("load f16 model");
        // Same architecture; only storage differs. Model ftype 7 = q8_0.
        assert_eq!(q8.hparams.n_vocab, f16.hparams.n_vocab);
        assert_eq!(q8.hparams.n_text_layer, f16.hparams.n_text_layer);
        assert_eq!(
            q8.hparams.ftype.rem_euclid(1000),
            7,
            "q8_0 model base ftype must be 7 (whisper packs a version factor)"
        );

        // The quantization took effect: at least one tensor is Q8_0-stored.
        let q8_name = q8
            .tensor_names()
            .find(|n| q8.tensor(n).is_some_and(|e| e.dtype == GgmlDType::Q8_0))
            .expect("q8_0 model has at least one Q8_0 tensor")
            .to_owned();

        let (qshape, qvals) = q8.tensor_f32(&q8_name).expect("q8_0 dequant");
        let (fshape, fvals) = f16.tensor_f32(&q8_name).expect("f16 dequant");
        assert_eq!(qshape, fshape, "shape mismatch for '{q8_name}'");
        assert!(
            qvals.iter().all(|v| v.is_finite()),
            "q8_0 dequant produced non-finite values"
        );
        let n = fvals.len() as f32;
        let mean_abs: f32 = fvals.iter().map(|v| v.abs()).sum::<f32>() / n;
        let mean_abs_diff: f32 = qvals
            .iter()
            .zip(&fvals)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n;
        assert!(
            mean_abs_diff < 0.05 * mean_abs,
            "q8_0 vs f16 mean|Δ| {mean_abs_diff} exceeds 5% of mean|w| {mean_abs} for '{q8_name}'"
        );
    }

    /// `read_blob_parallel` (the multi-thread banded blob reader tuned by
    /// `FW_BLOB_READ_WORKERS`, `5fc0707`) must be byte-identical to a serial
    /// `std::fs::read` — the bands are disjoint and exhaustively cover `[0, len)`.
    /// Guards both the sub-`MIN_PARALLEL` serial path and the parallel banded path
    /// (a file just over 8 MiB) against an off-by-one band-offset regression that
    /// would silently corrupt the loaded model. Content varies with byte offset so
    /// any misplacement surfaces as a mismatch.
    #[test]
    fn read_blob_parallel_matches_serial_read() {
        // Must exceed `read_blob_parallel`'s (function-local) `MIN_PARALLEL = 8 MiB`
        // to exercise the parallel banded path.
        const OVER_MIN_PARALLEL: usize = 8 * 1024 * 1024 + 12_345;
        let dir = tempfile::tempdir().expect("tempdir");
        for (idx, &len) in [0usize, 1, 4096, OVER_MIN_PARALLEL].iter().enumerate() {
            let bytes: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(31) ^ (i >> 3)) as u8)
                .collect();
            let path = dir.path().join(format!("blob_{idx}.bin"));
            std::fs::write(&path, &bytes).expect("write");
            let got = read_blob_parallel(&path).expect("read_blob_parallel");
            assert_eq!(got.len(), len, "length mismatch at len={len}");
            assert_eq!(
                got, bytes,
                "content mismatch at len={len} (band-offset bug?)"
            );
            assert_eq!(
                got,
                std::fs::read(&path).expect("fs::read"),
                "parallel read differs from serial fs::read at len={len}"
            );
        }
    }

    /// Builder for a minimal but fully valid in-memory ggml blob, used for
    /// hermetic parser coverage that doesn't require a real model file.
    struct SyntheticModel {
        bytes: Vec<u8>,
    }

    impl SyntheticModel {
        /// A minimal model: tiny hparams, a 2x3 filterbank, a 3-token vocab,
        /// one 2x2 f32 tensor, and one 2x2 f16 tensor.
        fn minimal() -> Self {
            let mut b = SyntheticModel { bytes: Vec::new() };
            b.push_u32(GGML_MAGIC);
            // hparams: n_vocab .. ftype (11 i32). ftype=1 (f16).
            for v in [5i32, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1] {
                b.push_i32(v);
            }
            // filterbank 2x3.
            b.push_i32(2);
            b.push_i32(3);
            for v in [0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5] {
                b.push_f32(v);
            }
            // vocab: 3 tokens.
            b.push_i32(3);
            for tok in [b"!".as_slice(), b"\"", b"ab"] {
                b.push_u32(tok.len() as u32);
                b.bytes.extend_from_slice(tok);
            }
            // tensor 1: f32, ggml dims [3, 2] -> logical shape [2, 3].
            b.push_tensor(
                "w_f32",
                0,
                &[3, 2],
                &Payload::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            );
            // tensor 2: f16, ggml dims [2, 2] -> logical shape [2, 2].
            b.push_tensor(
                "w_f16",
                1,
                &[2, 2],
                &Payload::F16(vec![1.0, 0.0, -2.0, 0.5]),
            );
            b
        }

        fn push_u32(&mut self, v: u32) {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        fn push_i32(&mut self, v: i32) {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        fn push_f32(&mut self, v: f32) {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }

        /// `ne` is in ggml order (fastest axis first).
        fn push_tensor(&mut self, name: &str, ttype: i32, ne: &[i32], payload: &Payload) {
            self.push_i32(ne.len() as i32);
            self.push_i32(name.len() as i32);
            self.push_i32(ttype);
            for &d in ne {
                self.push_i32(d);
            }
            self.bytes.extend_from_slice(name.as_bytes());
            match payload {
                Payload::F32(vals) => {
                    for &v in vals {
                        self.push_f32(v);
                    }
                }
                Payload::F16(vals) => {
                    for &v in vals {
                        self.bytes
                            .extend_from_slice(&Float16::from_f32(v).to_bits().to_le_bytes());
                    }
                }
            }
        }
    }

    enum Payload {
        F32(Vec<f32>),
        F16(Vec<f32>),
    }

    // ── f16 dequant unit tests over canonical bit patterns ──

    #[test]
    fn f16_known_bit_patterns() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0xC000), -2.0);
        // Smallest positive subnormal: 2^-24.
        let subnormal = f16_to_f32(0x0001);
        assert!(
            (subnormal - 2f32.powi(-24)).abs() < 1e-30,
            "got {subnormal}"
        );
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    // ── synthetic-blob parser tests ──

    #[test]
    fn header_and_filterbank_roundtrip() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        assert_eq!(model.hparams.n_vocab, 5);
        assert_eq!(model.hparams.n_audio_state, 384);
        assert_eq!(model.hparams.ftype, 1);
        assert_eq!(model.filters.n_mel, 2);
        assert_eq!(model.filters.n_fft_bins, 3);
        assert_eq!(model.filters.data, vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn absurd_filterbank_count_errors_without_huge_allocation() {
        // A crafted header with a valid magic + hparams that then claims an
        // absurd filterbank size (n_mel * n_fft ≈ 1 billion floats ≈ 4 GiB)
        // but provides no actual filter data. The clamp must keep the
        // speculative `Vec::with_capacity` bounded by the (tiny) remaining
        // blob, so we get a clean truncation error instead of an OOM-scale
        // allocation.
        let mut b = SyntheticModel { bytes: Vec::new() };
        b.push_u32(GGML_MAGIC);
        for v in [5i32, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1] {
            b.push_i32(v);
        }
        // filterbank: claim 32768 x 32768 ≈ 1.07e9 elements, but append nothing.
        b.push_i32(32_768);
        b.push_i32(32_768);
        // No filter data follows: the very first read_f32 hits EOF.
        let blob_len = b.bytes.len();
        let err = GgmlModel::parse(b.bytes).expect_err("absurd filterbank must error");
        match err {
            FwError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("end of file") || msg.contains("overflow"),
                    "expected a truncation/overflow error, got: {msg}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
        // Sanity: the blob itself is tiny (header-only), proving the claimed
        // count vastly exceeded available bytes and the clamp was load-bearing.
        assert!(blob_len < 128, "header-only blob should be tiny");
    }

    #[test]
    fn absurd_vocab_count_errors_without_huge_allocation() {
        // Valid header + a real (small) filterbank, then an absurd vocab count
        // with no token data. The vocab capacity clamp bounds the allocation;
        // the first token-length read hits EOF for a clean error.
        let mut b = SyntheticModel { bytes: Vec::new() };
        b.push_u32(GGML_MAGIC);
        for v in [5i32, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1] {
            b.push_i32(v);
        }
        // filterbank 1x1 with one real float so we reach the vocab section.
        b.push_i32(1);
        b.push_i32(1);
        b.push_f32(0.0);
        // vocab: claim ~1 billion tokens, append nothing.
        b.push_i32(1_000_000_000);
        let err = GgmlModel::parse(b.bytes).expect_err("absurd vocab must error");
        match err {
            FwError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("end of file") || msg.contains("overflow"),
                    "expected a truncation/overflow error, got: {msg}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn vocab_bytes_preserved() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        assert_eq!(model.vocab_tokens.len(), 3);
        assert_eq!(model.vocab_tokens[0], b"!");
        assert_eq!(model.vocab_tokens[1], b"\"");
        assert_eq!(model.vocab_tokens[2], b"ab");
        // hparams n_vocab (5) > file vocab (3) => 2 extra/special tokens.
        assert_eq!(model.n_extra_tokens(), 2);
    }

    #[test]
    fn tensor_shape_is_reversed_from_ggml_order() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        // ggml ne = [3, 2] => logical row-major shape [2, 3].
        let entry = model.tensor("w_f32").expect("w_f32 present");
        assert_eq!(entry.shape, vec![2, 3]);
        assert_eq!(entry.dtype, GgmlDType::F32);
        assert_eq!(entry.n_elements(), 6);
    }

    #[test]
    fn tensor_f32_values_and_f16_dequant() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        let (shape, vals) = model.tensor_f32("w_f32").expect("decode w_f32");
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let (shape, vals) = model.tensor_f32("w_f16").expect("decode w_f16");
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(vals, vec![1.0, 0.0, -2.0, 0.5]);
    }

    #[test]
    fn tensor_f16_raw_bits_and_f32_rejected() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        // f16 tensor: raw u16 bit patterns, in flat row-major order. The
        // synthetic w_f16 holds [1.0, 0.0, -2.0, 0.5] (logical [2,2]).
        let (shape, bits) = model.tensor_f16("w_f16").expect("raw f16");
        assert_eq!(shape, vec![2, 2]);
        let want: Vec<u16> = [1.0f32, 0.0, -2.0, 0.5]
            .iter()
            .map(|&v| Float16::from_f32(v).to_bits())
            .collect();
        assert_eq!(bits, want, "raw f16 bit patterns must round-trip exactly");
        // Each raw bit pattern dequantizes to exactly the f32 path's value.
        let (_s, f32_vals) = model.tensor_f32("w_f16").expect("f32 f16");
        for (b, &f) in bits.iter().zip(&f32_vals) {
            assert_eq!(f16_to_f32(*b), f, "dequant of raw bits == tensor_f32 value");
        }
        // f32-stored tensors are rejected (nothing to dequantize).
        let err = model
            .tensor_f16("w_f32")
            .expect_err("f32 tensor must be rejected");
        assert!(matches!(err, FwError::InvalidRequest(_)), "got {err:?}");
        assert!(err.to_string().contains("f32"), "{err}");
    }

    #[test]
    fn tensor_names_are_sorted() {
        let model = GgmlModel::parse(SyntheticModel::minimal().bytes).expect("parse");
        let names: Vec<&str> = model.tensor_names().collect();
        assert_eq!(names, vec!["w_f16", "w_f32"]);
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        // Append a well-formed (n_dims, name_len, ttype) triple describing a
        // zero-length tensor whose payload is empty: the directory loop sees
        // a non-EOF position, parses the entry, and then finds extra bytes
        // left over (the name `"x"`'s tensor having zero elements still leaves
        // the appended junk unconsumed) — exercising the explicit
        // trailing-bytes guard rather than a mid-read truncation.
        let mut bytes = SyntheticModel::minimal().bytes;
        // A complete extra header for a 1-D f32 tensor with dim 0 and a 1-byte
        // name, i.e. nelements==0 so byte_len==0, leaving the appended junk.
        bytes.extend_from_slice(&1i32.to_le_bytes()); // n_dims
        bytes.extend_from_slice(&1i32.to_le_bytes()); // name_len
        bytes.extend_from_slice(&0i32.to_le_bytes()); // ttype f32
        bytes.extend_from_slice(&0i32.to_le_bytes()); // ne[0] = 0
        bytes.extend_from_slice(b"z"); // name
        // Now genuine trailing junk that no tensor entry will consume.
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        let err = GgmlModel::parse(bytes).expect_err("must reject trailing bytes");
        assert!(matches!(err, FwError::InvalidRequest(_)), "got {err:?}");
        // Either guard (explicit trailing-bytes check, or the short-read on the
        // next phantom header) is an acceptable rejection of a corrupt tail.
        let msg = err.to_string();
        assert!(
            msg.contains("trailing bytes") || msg.contains("unexpected end of file"),
            "{err}"
        );
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = SyntheticModel::minimal().bytes;
        bytes[0] = 0x00; // corrupt the magic
        let err = GgmlModel::parse(bytes).expect_err("must reject bad magic");
        assert!(matches!(err, FwError::InvalidRequest(_)), "got {err:?}");
        assert!(err.to_string().contains("magic"), "{err}");
    }

    #[test]
    fn unsupported_ftype_is_rejected() {
        let mut b = SyntheticModel { bytes: Vec::new() };
        b.push_u32(GGML_MAGIC);
        // ftype = 1015: quant version 1, base 15 (IQ2_XXS, a codebook quant we
        // don't decode). Exercises the version strip (1015 % 1000 == 15) — the
        // supported legacy + k-quants (base 0/1/2/3/7/8/9/10/11/12/13/14) pass.
        for v in [5i32, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1015] {
            b.push_i32(v);
        }
        let err = GgmlModel::parse(b.bytes).expect_err("must reject quantized ftype");
        match err {
            FwError::Unsupported(msg) => {
                assert!(
                    msg.contains("base 15"),
                    "base ftype should be listed: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_is_rejected() {
        let mut bytes = SyntheticModel::minimal().bytes;
        bytes.truncate(10); // cut off mid-hparams
        let err = GgmlModel::parse(bytes).expect_err("must reject truncation");
        assert!(matches!(err, FwError::InvalidRequest(_)), "got {err:?}");
    }

    // ── gated tests against the real tiny.en model ──

    /// bd-A14: the streaming loader (`load_streamed`) must build a directory
    /// byte-for-byte identical to the resident [`GgmlModel::parse`], and its
    /// on-demand preads must return the exact same payload bytes the resident
    /// blob slices do. This is the drift guard for the two parallel scanners.
    #[cfg(unix)]
    #[test]
    fn streamed_dir_matches_resident() {
        let Some(path) = find_model_file("tiny.en") else {
            eprintln!("SKIP streamed_dir_matches_resident: ggml-tiny.en.bin not found");
            return;
        };
        let resident = GgmlModel::parse(read_blob_parallel(&path).expect("read blob"))
            .expect("resident parse");
        let streamed = GgmlModel::load_streamed(&path).expect("streamed parse");

        // Header / filterbank / vocab identical.
        assert_eq!(resident.hparams, streamed.hparams, "hparams differ");
        assert_eq!(resident.filters.n_mel, streamed.filters.n_mel);
        assert_eq!(resident.filters.n_fft_bins, streamed.filters.n_fft_bins);
        assert_eq!(
            resident.filters.data, streamed.filters.data,
            "mel data differ"
        );
        assert_eq!(resident.vocab_tokens, streamed.vocab_tokens, "vocab differ");

        // Tensor directory identical (names, offsets, lengths, shapes, dtype).
        assert_eq!(
            resident.tensors.len(),
            streamed.tensors.len(),
            "tensor count differs"
        );
        for (name, r) in &resident.tensors {
            let s = streamed
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("streamed missing tensor '{name}'"));
            assert_eq!(r.byte_offset, s.byte_offset, "offset differs for '{name}'");
            assert_eq!(r.byte_len, s.byte_len, "len differs for '{name}'");
            assert_eq!(r.shape, s.shape, "shape differs for '{name}'");
            assert_eq!(
                format!("{:?}", r.dtype),
                format!("{:?}", s.dtype),
                "dtype differs for '{name}'"
            );
        }

        // Payload bytes identical for a representative sample: resident borrows
        // the blob, streamed preads — the bytes must match exactly.
        let mut names: Vec<&String> = resident.tensors.keys().collect();
        names.sort();
        for name in names.iter().take(8) {
            let rb = resident
                .tensor_raw(name, &resident.tensors[*name])
                .expect("resident raw");
            let sb = streamed
                .tensor_raw(name, &streamed.tensors[*name])
                .expect("streamed raw");
            assert_eq!(
                rb.as_ref(),
                sb.as_ref(),
                "payload bytes differ for '{name}'"
            );
        }
    }

    #[test]
    fn real_tiny_en_full_parse() {
        let Some(path) = find_model_file("tiny.en") else {
            eprintln!("SKIP real_tiny_en_full_parse: ggml-tiny.en.bin not found");
            return;
        };
        let model = GgmlModel::load(&path).expect("load tiny.en");

        // Exact hparams.
        assert_eq!(model.hparams.n_vocab, 51864);
        assert_eq!(model.hparams.n_audio_ctx, 1500);
        assert_eq!(model.hparams.n_audio_state, 384);
        assert_eq!(model.hparams.n_audio_head, 6);
        assert_eq!(model.hparams.n_audio_layer, 4);
        assert_eq!(model.hparams.n_text_ctx, 448);
        assert_eq!(model.hparams.n_text_state, 384);
        assert_eq!(model.hparams.n_text_layer, 4);
        assert_eq!(model.hparams.n_mels, 80);
        assert_eq!(model.hparams.ftype, 1);

        // Filterbank dims.
        assert_eq!(model.filters.n_mel, 80);
        assert_eq!(model.filters.n_fft_bins, 201);
        assert_eq!(model.filters.data.len(), 80 * 201);

        // File vocab (50257) < hparams n_vocab (51864): 1607 extra tokens.
        assert_eq!(model.vocab_tokens.len(), 50257);
        assert_eq!(model.n_extra_tokens(), 51864 - 50257);
        assert_eq!(model.vocab_tokens[0], b"!");
        assert_eq!(model.vocab_tokens[1], b"\"");
        assert_eq!(model.vocab_tokens[2], b"#");

        // Full-file consumption is enforced inside parse() (trailing-byte
        // check); reaching here proves the parser consumed exactly to EOF.

        // Known tensors and their logical (row-major) shapes.
        let conv1 = model.tensor("encoder.conv1.weight").expect("conv1");
        assert_eq!(conv1.shape, vec![384, 80, 3]);

        let tok_emb = model
            .tensor("decoder.token_embedding.weight")
            .expect("token_embedding");
        assert_eq!(tok_emb.shape, vec![51864, 384]);

        let pos_emb = model
            .tensor("encoder.positional_embedding")
            .expect("positional_embedding");
        assert_eq!(pos_emb.shape, vec![1500, 384]);

        // Spot-check that decoded data is finite.
        for name in [
            "encoder.conv1.weight",
            "encoder.positional_embedding",
            "decoder.token_embedding.weight",
        ] {
            let (_shape, vals) = model.tensor_f32(name).expect("decode");
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "tensor {name} has non-finite values"
            );
            assert!(!vals.is_empty());
        }
    }

    #[test]
    fn real_large_v3_turbo_hparams() {
        let Some(path) = find_model_file("large-v3-turbo") else {
            eprintln!("SKIP real_large_v3_turbo_hparams: ggml-large-v3-turbo.bin not found");
            return;
        };
        let model = GgmlModel::load(&path).expect("load large-v3-turbo");
        assert_eq!(model.hparams.n_vocab, 51866);
        assert_eq!(model.hparams.n_audio_ctx, 1500);
        assert_eq!(model.hparams.n_audio_state, 1280);
        assert_eq!(model.hparams.n_audio_head, 20);
        assert_eq!(model.hparams.n_audio_layer, 32);
        assert_eq!(model.hparams.n_text_ctx, 448);
        assert_eq!(model.hparams.n_text_state, 1280);
        assert_eq!(model.hparams.n_text_head, 20);
        assert_eq!(model.hparams.n_text_layer, 4);
        assert_eq!(model.hparams.n_mels, 128);
        assert_eq!(model.hparams.ftype, 1);
        assert_eq!(model.filters.n_mel, 128);
        assert_eq!(model.filters.n_fft_bins, 201);
        // File stores 50257 tokens; hparams 51866 => 1609 extra.
        assert_eq!(model.vocab_tokens.len(), 50257);
        assert_eq!(model.n_extra_tokens(), 51866 - 50257);
        assert_eq!(model.vocab_tokens[0], b"!");
        assert_eq!(model.vocab_tokens[1], b"\"");
        assert_eq!(model.vocab_tokens[2], b"#");
    }
}
