//! Encoder self-attention SCORE-MATRIX rank probe — the precondition for
//! Nyström / linear (Performer/Linformer) attention (land-or-dig, 2026-07-05).
//!
//! Complements the frame-redundancy dig (54ed482, which showed FRAMES aren't
//! mergeable ⇒ ToMe dead). This measures a DIFFERENT approximation: is the softmax
//! ATTENTION MATRIX itself low-rank? If `softmax(QKᵀ)` [n,n] has effective rank r≪n,
//! then Nyström (r landmarks) / low-rank attention compute it in O(n·r) instead of
//! O(n²), a real lever for the ~10%-e2e encoder SDPA. Attention matrices ARE often
//! low-rank in practice (why Linformer/Nyströmformer work on some models) — but this
//! session's encoder has proven no-slack on every other axis, so MEASURE it on the
//! REAL model rather than assume.
//!
//! Method: run the real encoder to depth L (`FW_ENCODER_LAYERS=L`), take that residual
//! representation, apply layer L's real `attn_ln` + query/key projections (raw ggml
//! tensors), form per-head scores S_h = softmax(scale·Q_hK_hᵀ) over the REAL speech
//! frames, and measure captured energy at rank r via randomized range finding.
//! Caveat: the depth-L output carries an extra `ln_post`; attn_ln re-normalises it, so
//! the Q/K directions are a close proxy for layer L's true attention input — fine for
//! a low-rank-vs-full-rank verdict (the numerics need not be exact).
//!
//! Verdict: rank-r captures ≥99% at r≪n ⇒ Nyström/linear attention viable (owner GPU
//! signal). Near-full-rank ⇒ attention approximation dead too, completing the encoder
//! approximation map. Needs FRANKEN_WHISPER_MODEL_DIR.
//! Usage: `attn_lowrank_probe [wav] [repeat]` with FW_ENCODER_LAYERS=<L>.
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK};
use franken_whisper::native_engine::nn;

fn read_wav_mono16k(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    let mut i = 12;
    let (mut off, mut len, mut channels) = (0usize, 0usize, 1u16);
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        if id == b"fmt " { channels = u16::from_le_bytes([bytes[i + 10], bytes[i + 11]]); }
        else if id == b"data" { off = i + 8; len = sz.min(bytes.len() - (i + 8)); break; }
        i += 8 + sz + (sz & 1);
    }
    let data = &bytes[off..off + len];
    let ch = channels.max(1) as usize;
    let mut s = Vec::with_capacity(len / 2 / ch);
    let mut j = 0;
    while j + 2 * ch <= data.len() {
        let mut acc = 0i32;
        for c in 0..ch { acc += i16::from_le_bytes([data[j + 2 * c], data[j + 2 * c + 1]]) as i32; }
        s.push((acc as f32 / ch as f32) / 32768.0);
        j += 2 * ch;
    }
    s
}

fn tensor_wt(model: &GgmlModel, name: &str) -> Mat {
    // ggml [out,in] -> transposed [in,out] for nn::matmul (x[.,in] @ [in,out]).
    let (shape, data) = model.tensor_f32(name).expect("tensor");
    let (out_d, in_d) = (shape[0], shape[1]);
    let mut wt = vec![0.0f32; in_d * out_d];
    for o in 0..out_d { for i in 0..in_d { wt[i * out_d + o] = data[o * in_d + i]; } }
    Mat::from_vec(in_d, out_d, wt)
}
fn tensor_vec(model: &GgmlModel, name: &str) -> Vec<f32> { model.tensor_f32(name).expect("vec").1 }

/// Row-wise layer norm with affine, in place on a [rows,cols] flat buffer.
fn layer_norm(x: &mut [f32], rows: usize, cols: usize, w: &[f32], b: &[f32]) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (k, v) in row.iter_mut().enumerate() { *v = (*v - mean) * inv * w[k] + b[k]; }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 { a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum() }
fn nrm(a: &[f32]) -> f64 { dot(a, a).sqrt() }

/// Captured energy of a rank-r randomized range of a [n,n] row-major matrix `m`.
fn captured(m: &[f32], n: usize, r: usize) -> f64 {
    let mm = Mat::from_vec(n, n, m.to_vec());
    let mut s = 0x77u64;
    let mut rnd = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); if (s >> 40) & 1 == 0 { 1.0f32 } else { -1.0f32 } };
    let omega = Mat::from_vec(n, r, (0..n * r).map(|_| rnd()).collect());
    let y = nn::matmul(&mm, &omega).unwrap(); // [n,r]
    let mut qc: Vec<Vec<f32>> = Vec::new();
    for j in 0..r {
        let mut v: Vec<f32> = (0..n).map(|i| y.data[i * r + j]).collect();
        for q in &qc { let d = dot(&v, q) as f32; for (vi, &qi) in v.iter_mut().zip(q) { *vi -= d * qi; } }
        let nn_ = nrm(&v);
        if nn_ > 1e-5 { let inv = (1.0 / nn_) as f32; for vi in v.iter_mut() { *vi *= inv; } qc.push(v); }
    }
    let rk = qc.len();
    let mut qt = vec![0.0f32; rk * n];
    for (j, q) in qc.iter().enumerate() { for i in 0..n { qt[j * n + i] = q[i]; } }
    let proj = nn::matmul(&Mat::from_vec(rk, n, qt), &mm).unwrap();
    let pf: f64 = proj.data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let mf: f64 = m.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    pf / mf
}

fn main() {
    let wav = std::env::args().nth(1).unwrap_or_else(|| "tests/fixtures/native/jfk.wav".into());
    let repeat: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let l: usize = std::env::var("FW_ENCODER_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let layer = l.saturating_sub(1); // use the last-run layer's attn weights
    let path = find_model_file("large-v3-turbo").expect("set FRANKEN_WHISPER_MODEL_DIR");
    let model_g = GgmlModel::load(&path).expect("ggml");
    let model = LoadedModel::from_ggml(GgmlModel::load(&path).expect("ggml2")).expect("loaded");
    let base = read_wav_mono16k(&wav);
    let mut samples = Vec::new();
    for _ in 0..repeat { samples.extend_from_slice(&base); }
    let audio_sec = samples.len() as f32 / 16000.0;
    let full = mel::log_mel(&samples, &model.filters, 8).expect("mel");
    let window = mel::chunk_frames(&full, 0, FRAMES_PER_CHUNK);
    let out = encoder::forward(&model.encoder, &window, 8, &(|| Ok(()))).expect("enc");
    let (n_ctx, n_state) = (out.rows, out.cols);
    let n_head = model_g.hparams.n_audio_head as usize;
    let d_head = n_state / n_head;
    let real = ((audio_sec * 50.0).ceil() as usize).min(n_ctx);
    println!("=== encoder attn score-matrix rank | depth={l} attn-layer={layer} | real_frames={real} n_head={n_head} d_head={d_head} ===");

    // Project: h_norm = LN(out); Q = h_norm@Wq+bq; K = h_norm@Wk.
    let p = |s: &str| format!("encoder.blocks.{layer}.{s}");
    let ln_w = tensor_vec(&model_g, &p("attn_ln.weight"));
    let ln_b = tensor_vec(&model_g, &p("attn_ln.bias"));
    let wq = tensor_wt(&model_g, &p("attn.query.weight"));
    let bq = tensor_vec(&model_g, &p("attn.query.bias"));
    let wk = tensor_wt(&model_g, &p("attn.key.weight"));
    let mut hn = out.data.clone();
    layer_norm(&mut hn, n_ctx, n_state, &ln_w, &ln_b);
    let hn = Mat::from_vec(n_ctx, n_state, hn);
    let mut q = nn::matmul(&hn, &wq).unwrap();
    for r in 0..n_ctx { for k in 0..n_state { q.data[r * n_state + k] += bq[k]; } }
    let kk = nn::matmul(&hn, &wk).unwrap();
    let scale = (d_head as f32).powf(-0.5);

    // Per-head scores over real frames, measure rank.
    let heads = [0usize, n_head / 2, n_head - 1];
    for &h in &heads {
        let base = h * d_head;
        // scores[i,j] = softmax_j( scale * Q_h[i]·K_h[j] ) over real frames.
        let mut scores = vec![0.0f32; real * real];
        for i in 0..real {
            let qi = &q.data[i * n_state + base..i * n_state + base + d_head];
            let row = &mut scores[i * real..(i + 1) * real];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..real {
                let kj = &kk.data[j * n_state + base..j * n_state + base + d_head];
                let s: f32 = qi.iter().zip(kj).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                row[j] = s; if s > mx { mx = s; }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum; for v in row.iter_mut() { *v *= inv; }
        }
        let mut line = format!("  head {h:>2}: captured energy  ");
        for &r in &[8usize, 16, 32, 64, 128] {
            if r < real { line += &format!("r{r}={:.1}% ", captured(&scores, real, r) * 100.0); }
        }
        println!("{line}");
    }
    println!("  (rank r captures ≥99% at r≪{real} ⇒ Nyström/linear attention viable; near-full-rank ⇒ dead)");
}
