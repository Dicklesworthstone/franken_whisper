//! Neural source separation (bd-mmx3 / bd-f2se): DTLN-style dual-stage
//! spectral-mask separator consuming the pinned conversion artifact produced
//! by `scripts/export_dtln.py`.
//!
//! DATAFLOW (matches upstream breizhn/DTLN @ 1de1f15a, training graph):
//! rectangular framing (blockLen 512, blockShift 128 — `tf.signal.frame`
//! with NO window function), rfft per frame, stage 1 masks the magnitude in
//! the STFT domain (log(mag+1e-7) -> instant layer norm -> 2x LSTM(128) ->
//! dense+sigmoid), irfft of the masked spectrum, plain overlap-and-add;
//! stage 2 repeats the mask pattern in a learned domain: raw frame ->
//! Conv1D encoder (512->256, kernel 1 = dense matmul) -> ILN -> 2x LSTM ->
//! dense+sigmoid mask on encoded features -> decoder Conv1D (256->512) ->
//! overlap-and-add. No windowing anywhere; tail samples that do not fill a
//! final frame are dropped exactly like `tf.signal.frame`.
//!
//! MASKING DETAIL that matters for numerics: the instant layer norm feeds
//! ONLY the mask predictor; the multiply happens against the RAW magnitude /
//! raw encoded features (upstream `Multiply()([encoded_frames, mask_2])`).
//!
//! NUMERICS CONTRACT (from scripts/export_dtln.py receipt): Keras gate
//! columns are [i, f, g, o]; recurrent activation is hard_sigmoid
//! clip(0.2*x + 0.5, 0, 1); output/candidate activations are tanh; the fused
//! bias applies once (b_ih) with zero recurrent bias. Kernel matrices are
//! stored [input, 4H] and transposed once at load into the native
//! [`nn::LstmWeights`] row convention.
//!
//! The separator is deterministic, cancellation-checked between stages in
//! the pipeline caller, and model-gated: without the operator-provided
//! artifact everything here stays unused and the pipeline's separate stage
//! must report an explicit passthrough instead of claiming isolation.
const NUM_UNITS: usize = 128;

use crate::error::{FwError, FwResult};
use crate::native_engine::nn::{LstmState, LstmWeights};
use crate::native_engine::weights::SafetensorsFile;
use crate::native_engine::{Mat, nn};

pub const BLOCK_LEN: usize = 512;
pub const BLOCK_SHIFT: usize = 128;
pub const BINS: usize = BLOCK_LEN / 2 + 1;
const STFT_NORM_EPS: f32 = 1e-7;
const ENCODER_FEATURES: usize = 256;

/// Tensor names/shapes the pinned receipt guarantees, mirrored as compile-
/// time contract so loader drift fails closed here rather than at inference.
const EXPECTED: &[(&str, &[usize])] = &[
    ("s1.in_norm.gamma", &[257]),
    ("s1.in_norm.beta", &[257]),
    ("s1.lstm0.kernel", &[257, 512]),
    ("s1.lstm0.recurrent", &[128, 512]),
    ("s1.lstm0.bias", &[512]),
    ("s1.lstm1.kernel", &[128, 512]),
    ("s1.lstm1.recurrent", &[128, 512]),
    ("s1.lstm1.bias", &[512]),
    ("s1.mask.kernel", &[128, 257]),
    ("s1.mask.bias", &[257]),
    ("s2.encoder", &[1, 512, 256]),
    ("s2.in_norm.gamma", &[256]),
    ("s2.in_norm.beta", &[256]),
    ("s2.lstm0.kernel", &[256, 512]),
    ("s2.lstm0.recurrent", &[128, 512]),
    ("s2.lstm0.bias", &[512]),
    ("s2.lstm1.kernel", &[128, 512]),
    ("s2.lstm1.recurrent", &[128, 512]),
    ("s2.lstm1.bias", &[512]),
    ("s2.mask.kernel", &[128, 256]),
    ("s2.mask.bias", &[256]),
    ("s2.decoder", &[1, 256, 512]),
];

/// One DTLN separation kernel: feature norm + two LSTM layers + mask head.
struct MaskKernel {
    norm_gamma: Vec<f32>,
    norm_beta: Vec<f32>,
    lstm0: LstmWeights,
    lstm1: LstmWeights,
    mask_kernel: Mat, // [features_in, features_out]
    mask_bias: Vec<f32>,
}

#[derive(Default)]
struct MaskState {
    lstm0: LstmState,
    lstm1: LstmState,
}

impl MaskKernel {
    fn forward(&self, features: &[f32], state: &mut MaskState) -> FwResult<Vec<f32>> {
        let normalized = instant_layer_norm(features, &self.norm_gamma, &self.norm_beta);
        let input_mat = Mat::from_vec(1, normalized.len(), normalized);
        let hidden = self.lstm0.lstm_forward(&input_mat, &mut state.lstm0)?;
        let hidden = self.lstm1.lstm_forward(&hidden, &mut state.lstm1)?;
        nn::matmul_bias(&hidden, &self.mask_kernel, Some(&self.mask_bias))
            .map(|row| row.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
    }
}

/// Loaded DTLN model ready for batch separation.
pub struct DtlnSeparator {
    s1: MaskKernel,
    s2_encoder: Vec<f32>, // [512, 256] row-major after squeeze
    s2: MaskKernel,
    s2_decoder: Vec<f32>, // [256, 512] row-major after squeeze
}

/// Resolve the operator-provided converted weights path (model gating).
#[must_use]
pub fn weights_path_from_env() -> Option<std::path::PathBuf> {
    std::env::var_os("FRANKEN_WHISPER_DTLN_WEIGHTS")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
}

impl DtlnSeparator {
    /// Load and validate a converted artifact (safetensors F32 from
    /// `scripts/export_dtln.py`). Every name/shape is checked against the
    /// compile-time contract; finiteness is checked per tensor.
    pub fn from_safetensors(path: &std::path::Path) -> FwResult<Self> {
        let raw = std::fs::read(path)?;
        let tensors = parse_safetensors_f32(raw)?;
        let get = |name: &str| -> FwResult<&SafeTensor> {
            tensors
                .get(name)
                .ok_or_else(|| FwError::InvalidRequest(format!("dtln: missing tensor {name}")))
        };
        for (name, shape) in EXPECTED {
            let tensor = get(name)?;
            if tensor.shape != *shape {
                return Err(FwError::InvalidRequest(format!(
                    "dtln: {name} shape {:?} != contract {shape:?}",
                    tensor.shape
                )));
            }
            if tensor.data.iter().any(|v| !v.is_finite()) {
                return Err(FwError::InvalidRequest(format!("dtln: {name} non-finite")));
            }
        }
        // Keras stores kernels [input, 4H]; native LstmWeights wants rows 4H.
        let transpose = |name: &str| -> FwResult<Mat> {
            let t = get(name)?;
            let mut out = vec![0.0f32; t.data.len()];
            for r in 0..t.shape[0] {
                for c in 0..t.shape[1] {
                    out[c * t.shape[0] + r] = t.data[r * t.shape[1] + c];
                }
            }
            Ok(Mat::from_vec(t.shape[1], t.shape[0], out))
        };
        let zeros4 = || vec![0.0f32; 4 * NUM_UNITS];
        let lstm = |k: &str, r: &str, b: &str| -> FwResult<LstmWeights> {
            LstmWeights::new(transpose(k)?, transpose(r)?, get(b)?.data.clone(), zeros4())
        };
        let as_mat = |name: &str| -> FwResult<Mat> {
            let t = get(name)?;
            Ok(Mat::from_vec(t.shape[0], t.shape[1], t.data.clone()))
        };
        Ok(Self {
            s1: MaskKernel {
                norm_gamma: get("s1.in_norm.gamma")?.data.clone(),
                norm_beta: get("s1.in_norm.beta")?.data.clone(),
                lstm0: lstm("s1.lstm0.kernel", "s1.lstm0.recurrent", "s1.lstm0.bias")?,
                lstm1: lstm("s1.lstm1.kernel", "s1.lstm1.recurrent", "s1.lstm1.bias")?,
                mask_kernel: as_mat("s1.mask.kernel")?,
                mask_bias: get("s1.mask.bias")?.data.clone(),
            },
            s2_encoder: get("s2.encoder")?.data.clone(),
            s2: MaskKernel {
                norm_gamma: get("s2.in_norm.gamma")?.data.clone(),
                norm_beta: get("s2.in_norm.beta")?.data.clone(),
                lstm0: lstm("s2.lstm0.kernel", "s2.lstm0.recurrent", "s2.lstm0.bias")?,
                lstm1: lstm("s2.lstm1.kernel", "s2.lstm1.recurrent", "s2.lstm1.bias")?,
                mask_kernel: as_mat("s2.mask.kernel")?,
                mask_bias: get("s2.mask.bias")?.data.clone(),
            },
            s2_decoder: get("s2.decoder")?.data.clone(),
        })
    }

    /// Separate one utterance. Deterministic; resets recurrent state.
    pub fn separate(&mut self, samples: &[f32]) -> FwResult<Vec<f32>> {
        let mut state1 = MaskState::default();
        let mut state2 = MaskState::default();
        let frames_total = if samples.len() >= BLOCK_LEN {
            1 + (samples.len() - BLOCK_LEN) / BLOCK_SHIFT
        } else {
            0
        };
        // Overlap-add spans exactly (frames-1)*shift + block_len samples;
        // tail samples that never fill a frame are dropped, matching
        // tf.signal.frame's convention on the stage-2 input side.
        let output_len = if frames_total > 0 {
            (frames_total - 1) * BLOCK_SHIFT + BLOCK_LEN
        } else {
            0
        };
        let mut output = vec![0.0f32; output_len];
        let mut mag = vec![0.0f32; BINS];
        let mut phase = vec![0.0f32; BINS];
        for frame_index in 0..frames_total {
            let start = frame_index * BLOCK_SHIFT;
            let frame = &samples[start..start + BLOCK_LEN];
            rfft_mag_phase(frame, &mut mag, &mut phase);

            // Stage 1: mask in the STFT-magnitude domain.
            let log_mag: Vec<f32> = mag.iter().map(|&m| (m + STFT_NORM_EPS).ln()).collect();
            let mask1 = self.s1.forward(&log_mag, &mut state1)?;
            let masked_mag: Vec<f32> = mag.iter().zip(&mask1).map(|(&m, &k)| m * k).collect();
            let mut denoised = vec![0.0f32; BLOCK_LEN];
            irfft_mag_phase(&masked_mag, &phase, &mut denoised);

            // Stage 2 consumes the PRE-OLA frame directly — upstream trains
            // Conv1D on the ifftLayer output and overlap-adds only AFTER the
            // decoder (build_DTLN_model, lines ~346-355). Feeding it the
            // accumulated buffer instead double-counts overlapping frames.
            let encoded = matvec(&denoised, &self.s2_encoder, ENCODER_FEATURES);
            let mask2 = self.s2.forward(&encoded, &mut state2)?;
            let masked_enc: Vec<f32> = encoded.iter().zip(&mask2).map(|(&e, &k)| e * k).collect();
            let decoded = matvec(&masked_enc, &self.s2_decoder, BLOCK_LEN);
            for (j, slot) in decoded.iter().enumerate() {
                output[start + j] += slot;
            }
        }
        Ok(output)
    }
}

/// row-major [len, cols] times vector: out[j] = sum_i v[i] * w[i*cols + j].
fn matvec(v: &[f32], w_row_major: &[f32], cols: usize) -> Vec<f32> {
    debug_assert_eq!(w_row_major.len(), v.len() * cols);
    let mut out = vec![0.0f32; cols];
    for (i, &vi) in v.iter().enumerate() {
        let row = &w_row_major[i * cols..(i + 1) * cols];
        for (j, &wj) in row.iter().enumerate() {
            out[j] += vi * wj;
        }
    }
    out
}

fn instant_layer_norm(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let denom = (var + STFT_NORM_EPS).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| (v - mean) / denom * gamma[i] + beta[i])
        .collect()
}

// ---------------------------------------------------------------------------
// FFT: iterative radix-2 complex FFT sized to BLOCK_LEN (power of two), plus
// real-input mag/phase wrappers. Every butterfly stage reuses the same w^j
// chain restarted per block (standard Cooley-Tukey scheduling); verified
// against a naive O(n^2) DFT in tests.
// ---------------------------------------------------------------------------

fn fft_in_place(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = (i.reverse_bits()) >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let half = len / 2;
        for start in (0..n).step_by(len) {
            let (mut k_re, mut k_im) = (1.0f32, 0.0f32);
            for j in 0..half {
                let a = start + j;
                let b = a + half;
                let t_re = re[b] * k_re - im[b] * k_im;
                let t_im = re[b] * k_im + im[b] * k_re;
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
                let next_re = k_re * ang.cos() - k_im * ang.sin();
                k_im = k_re * ang.sin() + k_im * ang.cos();
                k_re = next_re;
            }
        }
        len *= 2;
    }
}

/// Real FFT of a BLOCK_LEN frame: magnitude and phase for BINS bins.
pub(crate) fn rfft_mag_phase(frame: &[f32], mag: &mut [f32], phase: &mut [f32]) {
    debug_assert_eq!(frame.len(), BLOCK_LEN);
    let mut re: Vec<f32> = frame.to_vec();
    let mut im = vec![0.0f32; BLOCK_LEN];
    fft_in_place(&mut re, &mut im);
    for k in 0..BINS {
        mag[k] = (re[k] * re[k] + im[k] * im[k]).sqrt();
        phase[k] = im[k].atan2(re[k]);
    }
}

/// Inverse real FFT from (possibly masked) magnitude + phase: rebuilds the
/// conjugate-symmetric spectrum, runs the transform, normalizes 1/N.
pub(crate) fn irfft_mag_phase(mag: &[f32], phase: &[f32], out: &mut [f32]) {
    debug_assert_eq!(mag.len(), BINS);
    let mut re = vec![0.0f32; BLOCK_LEN];
    let mut im = vec![0.0f32; BLOCK_LEN];
    for k in 0..BINS {
        let (c, s) = (phase[k].cos(), phase[k].sin());
        re[k] = mag[k] * c;
        im[k] = mag[k] * s;
    }
    // Conjugate-symmetric rebuild: X[N-k] = conj(X[k]); the Nyquist bin is
    // real by construction.
    for k in 1..BINS - 1 {
        re[BLOCK_LEN - k] = re[k];
        im[BLOCK_LEN - k] = -im[k];
    }
    // IDFT via forward FFT on the conjugated spectrum: for a conjugate-
    // symmetric spectrum X, x[n] = Re{FFT(conj(X))[n]}/N exactly.
    for v in &mut im {
        *v = -*v;
    }
    fft_in_place(&mut re, &mut im);
    let n_inv = 1.0f32 / BLOCK_LEN as f32;
    for (dst, &r) in out.iter_mut().zip(&re) {
        *dst = r * n_inv;
    }
}
struct SafeTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn parse_safetensors_f32(raw: Vec<u8>) -> FwResult<std::collections::BTreeMap<String, SafeTensor>> {
    let file = SafetensorsFile::from_owned_bytes(raw)?;
    let mut out = std::collections::BTreeMap::new();
    for name in file.names() {
        let dtype = file.dtype_name(name)?;
        if dtype != "F32" {
            return Err(FwError::InvalidRequest(format!(
                "safetensors: {name} dtype {dtype} != F32"
            )));
        }
        let (shape, data) = file.tensor_f32(name)?;
        out.insert(name.to_owned(), SafeTensor { shape, data });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0
        }
    }

    fn one_unit_lstm(
        input_weights: [f32; 4],
        recurrent_weights: [f32; 4],
        input_bias: [f32; 4],
    ) -> LstmWeights {
        LstmWeights::new(
            Mat::from_vec(4, 1, input_weights.to_vec()),
            Mat::from_vec(4, 1, recurrent_weights.to_vec()),
            input_bias.to_vec(),
            vec![0.0; 4],
        )
        .expect("one-unit LSTM fixture")
    }

    #[test]
    fn stacked_mask_lstm_layers_match_independent_state_reference() {
        let kernel = MaskKernel {
            norm_gamma: vec![0.0],
            norm_beta: vec![0.75],
            lstm0: one_unit_lstm(
                [0.3, -0.2, 0.4, 0.1],
                [0.15, 0.05, -0.1, 0.2],
                [0.2, 0.1, -0.05, 0.3],
            ),
            lstm1: one_unit_lstm(
                [-0.25, 0.35, 0.2, -0.15],
                [0.05, -0.2, 0.3, 0.1],
                [-0.1, 0.25, 0.15, 0.05],
            ),
            mask_kernel: Mat::from_vec(1, 1, vec![1.2]),
            mask_bias: vec![0.1],
        };
        let mut observed_state = MaskState::default();
        let mut expected_lstm0 = LstmState::default();
        let mut expected_lstm1 = LstmState::default();
        let frames = [[0.2_f32], [-0.6], [1.1]];
        let mut observed_sequence = Vec::new();

        for features in &frames {
            let observed = kernel
                .forward(features, &mut observed_state)
                .expect("stacked forward");

            let normalized = instant_layer_norm(features, &kernel.norm_gamma, &kernel.norm_beta);
            let input = Mat::from_vec(1, normalized.len(), normalized);
            let hidden0 = kernel
                .lstm0
                .lstm_forward(&input, &mut expected_lstm0)
                .expect("layer zero reference");
            let hidden1 = kernel
                .lstm1
                .lstm_forward(&hidden0, &mut expected_lstm1)
                .expect("layer one reference");
            let expected = nn::matmul_bias(&hidden1, &kernel.mask_kernel, Some(&kernel.mask_bias))
                .expect("reference mask projection")
                .data
                .iter()
                .map(|&value| 1.0 / (1.0 + (-value).exp()))
                .collect::<Vec<_>>();

            assert_eq!(observed, expected);
            observed_sequence.push(observed);
        }

        assert_eq!(observed_state.lstm0, expected_lstm0);
        assert_eq!(observed_state.lstm1, expected_lstm1);
        assert_ne!(
            observed_state.lstm0, observed_state.lstm1,
            "the two layer states must remain distinct for this asymmetric fixture"
        );

        let mut regressed_shared_state = LstmState::default();
        let regressed_sequence = frames
            .iter()
            .map(|features| {
                let normalized =
                    instant_layer_norm(features, &kernel.norm_gamma, &kernel.norm_beta);
                let input = Mat::from_vec(1, normalized.len(), normalized);
                let hidden0 = kernel
                    .lstm0
                    .lstm_forward(&input, &mut regressed_shared_state)
                    .expect("regressed layer zero");
                let hidden1 = kernel
                    .lstm1
                    .lstm_forward(&hidden0, &mut regressed_shared_state)
                    .expect("regressed layer one");
                nn::matmul_bias(&hidden1, &kernel.mask_kernel, Some(&kernel.mask_bias))
                    .expect("regressed mask projection")
                    .data
                    .iter()
                    .map(|&value| 1.0 / (1.0 + (-value).exp()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_ne!(
            observed_sequence, regressed_sequence,
            "the fixture must distinguish independent layer state from the prior shared-state recurrence"
        );
    }

    fn naive_dft_mag(frame: &[f32]) -> Vec<f64> {
        let n = frame.len();
        (0..=n / 2)
            .map(|k| {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (i, &x) in frame.iter().enumerate() {
                    let ang = -2.0 * std::f64::consts::PI * (i * k) as f64 / n as f64;
                    re += f64::from(x) * ang.cos();
                    im += f64::from(x) * ang.sin();
                }
                (re * re + im * im).sqrt()
            })
            .collect()
    }

    #[test]
    fn rfft_matches_naive_dft_on_seeded_frame() {
        let mut rng = Lcg(0x00_0B_D7);
        let frame: Vec<f32> = (0..BLOCK_LEN).map(|_| rng.next_f32()).collect();
        let mut mag = vec![0.0f32; BINS];
        let mut phase = vec![0.0f32; BINS];
        rfft_mag_phase(&frame, &mut mag, &mut phase);
        for (got, want) in mag.iter().zip(naive_dft_mag(&frame)) {
            assert!((f64::from(*got) - want).abs() < 1e-2, "mag {got} vs {want}");
        }
    }

    #[test]
    fn irfft_round_trips_within_tolerance() {
        let mut rng = Lcg(0xA11_CE0);
        let frame: Vec<f32> = (0..BLOCK_LEN).map(|_| rng.next_f32()).collect();
        let mut mag = vec![0.0f32; BINS];
        let mut phase = vec![0.0f32; BINS];
        rfft_mag_phase(&frame, &mut mag, &mut phase);
        let mut back = vec![0.0f32; BLOCK_LEN];
        irfft_mag_phase(&mag, &phase, &mut back);
        let max_err = frame
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 5e-3, "round-trip max err {max_err}");
    }

    fn contract_fixture_bytes(mask_bias: f32, encoder_identity: bool) -> Vec<u8> {
        let zero = |shape: &[usize]| vec![0.0f32; shape.iter().product::<usize>()];
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        for (name, shape) in EXPECTED {
            tensors.insert((*name).to_owned(), (shape.to_vec(), zero(shape)));
        }
        // Mask predictor driven purely by biases; norms collapse to constants
        // (gamma=0,beta=0) so prediction is input-independent — legitimate
        // because ILN feeds only the mask predictor, never the masked signal.
        tensors.get_mut("s1.mask.bias").unwrap().1 = vec![mask_bias; 257];
        tensors.get_mut("s2.mask.bias").unwrap().1 = vec![mask_bias; 256];
        if encoder_identity {
            // enc[j] = raw_frame[j] for j<256; decoder[j][j]=1 maps them back.
            for j in 0..ENCODER_FEATURES {
                tensors.get_mut("s2.encoder").unwrap().1[j * ENCODER_FEATURES + j] = 1.0;
                tensors.get_mut("s2.decoder").unwrap().1[j * BLOCK_LEN + j] = 1.0;
            }
        }
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        let mut body = Vec::new();
        for (name, (shape, data)) in &tensors {
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape,
                    "data_offsets": [offset, offset + data.len() * 4]}),
            );
            body.extend(data.iter().flat_map(|v| v.to_le_bytes()));
            offset += data.len() * 4;
        }
        let mut hb = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        while !hb.len().is_multiple_of(8) {
            hb.push(b' ');
        }
        let mut bytes = (hb.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&hb);
        bytes.extend_from_slice(&body);
        bytes
    }

    fn rewrite_contract_header(
        bytes: &[u8],
        mutate: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
    ) -> Vec<u8> {
        let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("fixture header prefix"));
        let header_len = usize::try_from(header_len).expect("fixture header length");
        let body = &bytes[8 + header_len..];
        let mut header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + header_len]).expect("fixture header JSON");
        mutate(header.as_object_mut().expect("fixture header object"));

        let mut header_bytes = serde_json::to_vec(&header).expect("rewritten fixture header");
        while !header_bytes.len().is_multiple_of(8) {
            header_bytes.push(b' ');
        }
        let mut rewritten = u64::try_from(header_bytes.len())
            .expect("rewritten fixture header length")
            .to_le_bytes()
            .to_vec();
        rewritten.extend_from_slice(&header_bytes);
        rewritten.extend_from_slice(body);
        rewritten
    }

    fn assert_invalid_safetensors(bytes: &[u8], expected_detail: &str) {
        let error = match parse_safetensors_f32(bytes.to_vec()) {
            Ok(_) => panic!("malformed safetensors metadata was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.error_code(), "FW-INVALID-REQUEST");
        assert!(
            error.to_string().contains(expected_detail),
            "expected `{expected_detail}` in `{error}`"
        );
    }

    fn write_and_load(bytes: &[u8], tag: &str) -> DtlnSeparator {
        let dir = std::env::temp_dir().join(format!("dtln_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let path = dir.join(format!("{tag}.safetensors"));
        std::fs::write(&path, bytes).expect("write");
        DtlnSeparator::from_safetensors(&path).expect("load")
    }

    #[test]
    fn loader_rejects_missing_and_mistyped_entries() {
        let good = contract_fixture_bytes(0.0, false);
        let parsed = parse_safetensors_f32(good.clone()).expect("parses");
        assert!(parsed.contains_key("s1.mask.bias"));
        let bad = String::from_utf8_lossy(&good).replacen("F32", "F16", 1);
        assert!(parse_safetensors_f32(bad.into_bytes()).is_err());
        assert!(parse_safetensors_f32(good[..good.len() - 9].to_vec()).is_err());
        // Shape-contract enforcement through the public loader.
        let dir = std::env::temp_dir().join(format!("dtln_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let path = dir.join("bad_shape.safetensors");
        // Corrupt one tensor's shape inside the header JSON (first match =
        // s1.lstm0.kernel's "[257,512]"), then persist BEFORE loading.
        let mutated = String::from_utf8_lossy(&good).replacen("[257,", "[256,", 1);
        std::fs::write(&path, mutated.as_bytes()).expect("write mutated");
        assert!(
            DtlnSeparator::from_safetensors(&path).is_err(),
            "shape contract must fail closed"
        );
    }

    #[test]
    fn loader_rejects_non_integer_shape_dimensions_instead_of_dropping_them() {
        let malformed = rewrite_contract_header(&contract_fixture_bytes(0.0, false), |header| {
            let tensor = header
                .get_mut("s1.mask.bias")
                .and_then(serde_json::Value::as_object_mut)
                .expect("mask bias tensor");
            tensor.insert("shape".to_owned(), serde_json::json!([257, "extra"]));
        });
        assert_invalid_safetensors(&malformed, "shape dimension");
    }

    #[test]
    fn loader_rejects_malformed_and_overflowing_data_offsets() {
        let fixture = contract_fixture_bytes(0.0, false);

        let wrong_arity = rewrite_contract_header(&fixture, |header| {
            let tensor = header
                .get_mut("s1.mask.bias")
                .and_then(serde_json::Value::as_object_mut)
                .expect("mask bias tensor");
            tensor.insert("data_offsets".to_owned(), serde_json::json!([0]));
        });
        assert_invalid_safetensors(&wrong_arity, "exactly 2 elements");

        let wrong_type = rewrite_contract_header(&fixture, |header| {
            let offsets = header
                .get_mut("s1.mask.bias")
                .and_then(|tensor| tensor.get_mut("data_offsets"))
                .and_then(serde_json::Value::as_array_mut)
                .expect("mask bias offsets");
            offsets[0] = serde_json::json!("not-an-offset");
        });
        assert_invalid_safetensors(&wrong_type, "data_offsets begin");

        let descending = rewrite_contract_header(&fixture, |header| {
            let tensor = header
                .get_mut("s1.mask.bias")
                .and_then(serde_json::Value::as_object_mut)
                .expect("mask bias tensor");
            tensor.insert("data_offsets".to_owned(), serde_json::json!([1, 0]));
        });
        assert_invalid_safetensors(&descending, "data_offsets begin");

        let overflowing = rewrite_contract_header(&fixture, |header| {
            let tensor = header
                .get_mut("s1.mask.bias")
                .and_then(serde_json::Value::as_object_mut)
                .expect("mask bias tensor");
            tensor.insert("data_offsets".to_owned(), serde_json::json!([0, u64::MAX]));
        });
        assert_invalid_safetensors(&overflowing, "offset");
    }

    #[test]
    fn loader_rejects_tensor_byte_count_that_disagrees_with_shape() {
        let malformed = rewrite_contract_header(&contract_fixture_bytes(0.0, false), |header| {
            let offsets = header
                .get_mut("s1.mask.bias")
                .and_then(|tensor| tensor.get_mut("data_offsets"))
                .and_then(serde_json::Value::as_array_mut)
                .expect("mask bias offsets");
            let start = offsets[0].clone();
            offsets[1] = start;
        });
        assert_invalid_safetensors(&malformed, "byte span");
    }

    #[test]
    fn synthetic_identity_mask_passes_first_block_through() {
        let mut sep = write_and_load(&contract_fixture_bytes(14.0, true), "identity");
        let input: Vec<f32> = (0..BLOCK_LEN * 6)
            .map(|i| ((i as f32) * 0.031).sin() * 0.5)
            .collect();
        let output = sep.separate(&input).expect("separate");
        assert_eq!(
            output.len(),
            (BLOCK_LEN * 6 - BLOCK_LEN) / BLOCK_SHIFT * BLOCK_LEN + BLOCK_LEN
        );
        // First hop: exactly one contributing frame, sigmoid(14) ~= 0.9999992.
        for (j, o) in output.iter().take(BLOCK_SHIFT).enumerate() {
            let delta = (o - input[j]).abs();
            assert!(delta < 1e-3, "pass-through sample {j}: delta {delta}");
        }
    }

    #[test]
    fn synthetic_zero_mask_attenuates_to_silence() {
        let mut sep = write_and_load(&contract_fixture_bytes(-14.0, false), "silence");
        let input: Vec<f32> = (0..BLOCK_LEN * 6)
            .map(|i| ((i as f32) * 0.05).sin() * 0.8)
            .collect();
        let output = sep.separate(&input).expect("separate");
        let rms =
            |x: &[f32]| (x.iter().map(|&v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
        let ratio = rms(&output) / rms(&input[..output.len()]).max(1e-12);
        assert!(ratio < 1e-3, "expected near-silence, rms ratio {ratio}");
    }

    /// Model-gated acceptance: speech+tone mixture through the REAL converted
    /// weights must attenuate the tone band by >= 10 dB. Skips when
    /// FRANKEN_WHISPER_DTLN_WEIGHTS is unset (model-gated convention).
    #[test]
    fn real_weights_attenuate_tone_band_by_at_least_10db() {
        let Some(path) = weights_path_from_env() else {
            eprintln!("SKIP: FRANKEN_WHISPER_DTLN_WEIGHTS not set (model-gated)");
            return;
        };
        let mut sep = DtlnSeparator::from_safetensors(&path).expect("real weights load");
        let fs = 16_000.0f32;
        let n = fs as usize;
        let mut mix = vec![0.0f32; n];
        for (i, slot) in mix.iter_mut().enumerate() {
            let t = i as f32 / fs;
            let env = 0.6 * (1.0 - (-(t * 2.0)).exp());
            let speech = 0.35 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.2 * env * (2.0 * std::f32::consts::PI * 600.0 * t).sin();
            let tone = 0.25 * (2.0 * std::f32::consts::PI * 4_000.0 * t).sin();
            *slot = speech + tone;
        }
        let out = sep.separate(&mix).expect("separate");
        let band_power = |x: &[f32]| -> f64 {
            let mut mag = vec![0.0f32; BINS];
            let mut ph = vec![0.0f32; BINS];
            let mut acc = 0.0f64;
            let mut frames = 0usize;
            for chunk in x.chunks(BLOCK_LEN).filter(|c| c.len() == BLOCK_LEN) {
                rfft_mag_phase(chunk, &mut mag, &mut ph);
                for (k, &m) in mag.iter().enumerate() {
                    let f = k as f32 * fs / BLOCK_LEN as f32;
                    if (3_800.0..=4_200.0).contains(&f) {
                        acc += f64::from(m * m);
                    }
                }
                frames += 1;
            }
            acc / frames.max(1) as f64
        };
        let before = band_power(&mix);
        let after = band_power(&out);
        let reduction_db = 10.0 * (before / after.max(1e-20)).log10();
        assert!(
            reduction_db >= 10.0,
            "tone-band attenuation {reduction_db:.2} dB < 10 dB gate"
        );
    }
}
