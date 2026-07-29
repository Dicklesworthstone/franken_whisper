//! Encoder frame-redundancy probe — the LAST unmeasured encoder redundancy axis
//! (land-or-dig, 2026-07-05).
//!
//! This session mapped encoder redundancy on two axes and found NONE:
//!   - DEPTH: layer-pruning is fatal at even 4/32 (project_encoder_flop_reduction_mapped).
//!   - SPECTRAL: weights are near-full-rank (low-rank dig, e2ee176).
//! The THIRD axis — SEQUENCE redundancy (are the 1500 encoder frames mutually
//! redundant, i.e. mergeable?) — is the precondition for ToMe / token-merging, the
//! biggest owner-gated encoder lever left, and has only ever been "reasoned a poor
//! bet", never MEASURED. This probe measures it directly on REAL audio:
//!   (a) adjacent-frame cosine similarity distribution cos(f[i], f[i+1]) — high ⇒
//!       neighbours mergeable (ToMe's exact signal),
//!   (b) effective rank of the [n_frames, n_state] hidden state via randomized range
//!       finding — how many dims the frames actually span (≪ n_frames ⇒ redundant),
//!   (c) a ToMe proxy: average each adjacent pair (2× merge) and report the
//!       reconstruction relerr — the error a 2× sequence reduction would inject.
//! Measured over the REAL-speech frames only (padding is trivially redundant and is
//! already handled by tail-truncation). Sample depth via `FW_ENCODER_LAYERS=N`.
//!
//! Verdict rule: encoder GEMM cost ∝ n_frames, so a viable merge (low relerr, high
//! adjacent-sim) at ratio R would cut ~R of the 70%-e2e encoder — but the transcript
//! tolerates ≪1% representation error (layer-prune fatal). High sim + low merge-error
//! ⇒ ToMe is a real owner lever (green light); low sim / high error ⇒ sequence
//! approximation is dead too, closing the encoder redundancy map.
//!
//! Needs FRANKEN_WHISPER_MODEL_DIR. Usage: `frame_redundancy_probe [wav] [repeat]`.
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK};
use franken_whisper::native_engine::nn;

/// Minimal PCM16 mono/stereo WAV → mono f32 in [-1,1] (copied from e2e_probe).
fn read_wav_mono16k(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    // locate 'data' chunk
    let mut i = 12;
    let (mut off, mut len) = (0usize, 0usize);
    let mut channels = 1u16;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        if id == b"fmt " {
            channels = u16::from_le_bytes([bytes[i + 10], bytes[i + 11]]);
        } else if id == b"data" {
            off = i + 8;
            len = sz.min(bytes.len() - (i + 8));
            break;
        }
        i += 8 + sz + (sz & 1);
    }
    let data = &bytes[off..off + len];
    let ch = channels.max(1) as usize;
    let n = len / 2;
    let mut samples = Vec::with_capacity(n / ch);
    let mut j = 0;
    while j + 2 * ch <= data.len() {
        let mut acc = 0i32;
        for c in 0..ch {
            acc += i16::from_le_bytes([data[j + 2 * c], data[j + 2 * c + 1]]) as i32;
        }
        samples.push((acc as f32 / ch as f32) / 32768.0);
        j += 2 * ch;
    }
    samples
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}
fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

/// Effective rank: captured energy of a rank-r randomized range of `m` [rows,cols].
fn captured_energy(m: &Mat, rows: usize, r: usize) -> f64 {
    let cols = m.cols;
    // Omega [rows? no — project columns]. Range of the row-space: sketch = M^T @ Omega.
    // We want how many dims the ROWS span → range of M (rows × cols), sketch M @ Omega[cols,r].
    let mut s = 0x9E37u64;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if (s >> 40) & 1 == 0 { 1.0f32 } else { -1.0f32 }
    };
    let omega = Mat::from_vec(cols, r, (0..cols * r).map(|_| rnd()).collect());
    let sub = Mat::from_vec(rows, cols, m.data[..rows * cols].to_vec());
    let y = nn::matmul(&sub, &omega).unwrap(); // [rows, r]
    // Gram-Schmidt on y columns → Q, captured = ||Q^T sub||/||sub||
    let mut qcols: Vec<Vec<f32>> = Vec::new();
    for j in 0..r {
        let mut v: Vec<f32> = (0..rows).map(|i| y.data[i * r + j]).collect();
        for q in &qcols {
            let d = dot(&v, q) as f32;
            for (vi, &qi) in v.iter_mut().zip(q) {
                *vi -= d * qi;
            }
        }
        let nm = norm(&v);
        if nm > 1e-5 {
            let inv = (1.0 / nm) as f32;
            for vi in v.iter_mut() {
                *vi *= inv;
            }
            qcols.push(v);
        }
    }
    let rk = qcols.len();
    let mut qt = vec![0.0f32; rk * rows];
    for (j, qc) in qcols.iter().enumerate() {
        for i in 0..rows {
            qt[j * rows + i] = qc[i];
        }
    }
    let proj = nn::matmul(&Mat::from_vec(rk, rows, qt), &sub).unwrap();
    let pf: f64 = proj
        .data
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let sf: f64 = sub
        .data
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    pf / sf
}

fn main() {
    let wav = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/native/jfk.wav".to_string());
    let repeat: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let path = find_model_file("large-v3-turbo").expect("set FRANKEN_WHISPER_MODEL_DIR");
    let model = GgmlModel::load(&path)
        .and_then(LoadedModel::from_ggml)
        .expect("load turbo");
    let base = read_wav_mono16k(&wav);
    let mut samples = Vec::new();
    for _ in 0..repeat {
        samples.extend_from_slice(&base);
    }
    let audio_sec = samples.len() as f32 / 16000.0;
    let full = mel::log_mel(&samples, &model.filters, 8).expect("log_mel");
    let window = mel::chunk_frames(&full, 0, FRAMES_PER_CHUNK);
    let depth = std::env::var("FW_ENCODER_LAYERS").unwrap_or_else(|_| "all(32)".into());
    let out = encoder::forward(&model.encoder, &window, 8, &(|| Ok(()))).expect("encoder");
    let n_ctx = out.rows;
    let real = ((audio_sec * 50.0).ceil() as usize).min(n_ctx); // ~50 enc frames/s
    println!(
        "=== encoder frame redundancy | wav={wav} x{repeat} ({audio_sec:.1}s) depth={depth} | n_ctx={n_ctx} real_frames={real} n_state={} ===",
        out.cols
    );

    // (a) adjacent-frame cosine similarity over real frames.
    let cols = out.cols;
    let mut sims: Vec<f64> = Vec::with_capacity(real - 1);
    for i in 0..real.saturating_sub(1) {
        let a = &out.data[i * cols..(i + 1) * cols];
        let b = &out.data[(i + 1) * cols..(i + 2) * cols];
        let na = norm(a);
        let nb = norm(b);
        if na > 0.0 && nb > 0.0 {
            sims.push(dot(a, b) / (na * nb));
        }
    }
    let mut sorted = sims.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mean = sims.iter().sum::<f64>() / sims.len() as f64;
    let median = sorted[sorted.len() / 2];
    let frac = |t: f64| sims.iter().filter(|&&s| s > t).count() as f64 / sims.len() as f64 * 100.0;
    println!(
        "  (a) adjacent cos-sim: mean={mean:.3} median={median:.3} | >0.8:{:.0}% >0.9:{:.0}% >0.95:{:.0}% >0.99:{:.0}%",
        frac(0.8),
        frac(0.9),
        frac(0.95),
        frac(0.99)
    );

    // (b) effective rank of the real-frame block (captured energy at r).
    for &r in &[32usize, 64, 128, 256] {
        if r < real {
            println!(
                "  (b) rank-{r:<4} captured energy = {:.2}%  (of {real} frames' span)",
                captured_energy(&out, real, r) * 100.0
            );
        }
    }

    // (c) ToMe 2× proxy: average adjacent pairs, reconstruction relerr over real frames.
    let mut err2 = 0.0f64;
    let mut sig2 = 0.0f64;
    let mut i = 0;
    while i + 1 < real {
        let a = &out.data[i * cols..(i + 1) * cols];
        let b = &out.data[(i + 1) * cols..(i + 2) * cols];
        for k in 0..cols {
            let avg = 0.5 * (a[k] + b[k]);
            let ea = a[k] as f64 - avg as f64;
            let eb = b[k] as f64 - avg as f64;
            err2 += ea * ea + eb * eb;
            sig2 += (a[k] as f64) * (a[k] as f64) + (b[k] as f64) * (b[k] as f64);
        }
        i += 2;
    }
    println!(
        "  (c) uniform 2× adjacent-merge reconstruction relerr = {:.2}%  (transcript tolerates <<1%)",
        (err2 / sig2).sqrt() * 100.0
    );
}
