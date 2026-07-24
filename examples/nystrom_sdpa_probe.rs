//! Nyström low-rank encoder self-attention probe (land-or-dig, 2026-07-06).
//!
//! Executes the owner/prototype next-step from the 2026-07-05 low-rank GREEN LIGHT
//! (the encoder score matrix is rank ~32-64 of 1500). Nyströmformer approximates
//! `O = softmax(QKᵀ/√d)·V` in O(n·m·d) instead of O(n²·d) using m landmarks:
//!   Q̃,K̃ = segment-means of Q,K  [m,d]
//!   F = softmax(Q·K̃ᵀ·s) [n,m],  A = softmax(Q̃·K̃ᵀ·s) [m,m],  B = softmax(Q̃·Kᵀ·s) [m,n]
//!   O_nys = F · pinv(A) · (B · V)          (pinv via iterative Moore-Penrose)
//!
//! Two verdicts decide whether to build a gated FW_NYSTROM_ATTN engine path:
//!   (1) ACCURACY — per-head output relerr ‖O_nys−O_exact‖/‖O_exact‖ vs the EXACT
//!       softmax attention, on REAL turbo weights + REAL audio, by depth. The encoder
//!       is no-slack (layer-prune fatal, tolerates ≪1%), so >~0.1%/layer ⇒ transcript-
//!       fatal over 32 layers ⇒ REJECT (accuracy).
//!   (2) SPEED — single-head Nyström wall-clock vs the EXACT per-head matmul path. If
//!       Nyström isn't even faster than franken's own O(n²) per-head SDPA at the
//!       algorithm level, it can't beat the tuned fused `ft_kernel_cpu::sdpa_forward_f32`
//!       (2.35× over per-head) ⇒ REJECT (speed, the CountSketch/PQ failure mode).
//!
//! Needs FRANKEN_WHISPER_MODEL_DIR. `nystrom_sdpa_probe [wav] [repeat]`.
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK};
use franken_whisper::native_engine::nn;
use std::time::Instant;

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
    let (shape, data) = model.tensor_f32(name).expect("tensor");
    let (out_d, in_d) = (shape[0], shape[1]);
    let mut wt = vec![0.0f32; in_d * out_d];
    for o in 0..out_d { for i in 0..in_d { wt[i * out_d + o] = data[o * in_d + i]; } }
    Mat::from_vec(in_d, out_d, wt)
}
fn tensor_vec(model: &GgmlModel, name: &str) -> Vec<f32> { model.tensor_f32(name).expect("vec").1 }

fn layer_norm(x: &mut [f32], rows: usize, cols: usize, w: &[f32], b: &[f32]) {
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (k, v) in row.iter_mut().enumerate() { *v = (*v - mean) * inv * w[k] + b[k]; }
    }
}

/// Row-softmax of an [r,c] flat matrix in place (numerically stable).
fn softmax_rows(m: &mut [f32], r: usize, c: usize) {
    for i in 0..r {
        let row = &mut m[i * c..(i + 1) * c];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for v in row.iter_mut() { *v = (*v - mx).exp(); s += *v; }
        let inv = 1.0 / s;
        for v in row.iter_mut() { *v *= inv; }
    }
}

/// C = A[ra,k] @ B[k,cb] (naive, row-major) — small shapes, correctness-first.
fn mm(a: &[f32], b: &[f32], ra: usize, k: usize, cb: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; ra * cb];
    for i in 0..ra {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 { continue; }
            let brow = &b[p * cb..(p + 1) * cb];
            let crow = &mut c[i * cb..(i + 1) * cb];
            for j in 0..cb { crow[j] += av * brow[j]; }
        }
    }
    c
}
/// scores = Q[n,d] @ Kᵀ (K is [m,d]) * scale  -> [n,m]
fn qk_scaled(q: &[f32], kk: &[f32], n: usize, m: usize, d: usize, scale: f32) -> Vec<f32> {
    let mut s = vec![0.0f32; n * m];
    for i in 0..n {
        let qi = &q[i * d..(i + 1) * d];
        for j in 0..m {
            let kj = &kk[j * d..(j + 1) * d];
            let mut acc = 0.0f32;
            for t in 0..d { acc += qi[t] * kj[t]; }
            s[i * m + j] = acc * scale;
        }
    }
    s
}

/// Iterative Moore-Penrose pseudo-inverse of a small [m,m] matrix `a`
/// (Razavi et al.; Z_{k+1} = Z_k(2I − A Z_k)), 6 iterations from the standard init.
fn pinv(a: &[f32], m: usize) -> Vec<f32> {
    // init Z = Aᵀ / (‖A‖_1 · ‖A‖_∞)
    let mut n1 = 0.0f32; // max col abs-sum
    let mut ninf = 0.0f32; // max row abs-sum
    for j in 0..m { let mut s = 0.0f32; for i in 0..m { s += a[i * m + j].abs(); } n1 = n1.max(s); }
    for i in 0..m { let mut s = 0.0f32; for j in 0..m { s += a[i * m + j].abs(); } ninf = ninf.max(s); }
    let denom = (n1 * ninf).max(1e-12);
    let mut z = vec![0.0f32; m * m];
    for i in 0..m { for j in 0..m { z[i * m + j] = a[j * m + i] / denom; } }
    for _ in 0..6 {
        let az = mm(a, &z, m, m, m); // A Z
        // 2I - AZ
        let mut t = az;
        for i in 0..m * m { t[i] = -t[i]; }
        for i in 0..m { t[i * m + i] += 2.0; }
        z = mm(&z, &t, m, m, m); // Z (2I - AZ)
    }
    z
}

fn frob(a: &[f32]) -> f64 { a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt() }

fn main() {
    let wav = std::env::args().nth(1).unwrap_or_else(|| "tests/fixtures/native/jfk.wav".into());
    let repeat: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let path = find_model_file("large-v3-turbo").expect("set FRANKEN_WHISPER_MODEL_DIR");
    let model_g = GgmlModel::load(&path).expect("ggml");
    let model = LoadedModel::from_ggml(GgmlModel::load(&path).expect("ggml2")).expect("loaded");
    let base = read_wav_mono16k(&wav);
    let mut samples = Vec::new();
    for _ in 0..repeat { samples.extend_from_slice(&base); }
    let audio_sec = samples.len() as f32 / 16000.0;
    let full = mel::log_mel(&samples, &model.filters, 8).expect("mel");
    let window = mel::chunk_frames(&full, 0, FRAMES_PER_CHUNK);
    let n_head = model_g.hparams.n_audio_head as usize;

    // Depth via FW_ENCODER_LAYERS (OnceLock in encoder::forward). Invoke 3× (4/16/32).
    let l: usize = std::env::var("FW_ENCODER_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    println!("=== Nyström SDPA probe | {wav}x{repeat} ({audio_sec:.1}s) depth={l} n_head={n_head} ===");
    println!("relerr = ‖O_nys−O_exact‖/‖O_exact‖ (encoder tolerates ≪1%/layer); speed = exact_per_head ÷ nystrom");
    {
        let layer = l - 1;
        let out = encoder::forward(&model.encoder, &window, 8, &(|| Ok(()))).expect("enc");
        let (n_ctx, n_state) = (out.rows, out.cols);
        let d_head = n_state / n_head;
        let real = ((audio_sec * 50.0).ceil() as usize).min(n_ctx);
        let scale = (d_head as f32).powf(-0.5);
        let p = |s: &str| format!("encoder.blocks.{layer}.{s}");
        let ln_w = tensor_vec(&model_g, &p("attn_ln.weight"));
        let ln_b = tensor_vec(&model_g, &p("attn_ln.bias"));
        let wq = tensor_wt(&model_g, &p("attn.query.weight"));
        let bq = tensor_vec(&model_g, &p("attn.query.bias"));
        let wk = tensor_wt(&model_g, &p("attn.key.weight"));
        let wv = tensor_wt(&model_g, &p("attn.value.weight"));
        let bv = tensor_vec(&model_g, &p("attn.value.bias"));
        let mut hn = out.data.clone();
        layer_norm(&mut hn, n_ctx, n_state, &ln_w, &ln_b);
        let hn = Mat::from_vec(n_ctx, n_state, hn);
        let mut q = nn::matmul(&hn, &wq).unwrap();
        for r in 0..n_ctx { for k in 0..n_state { q.data[r * n_state + k] += bq[k]; } }
        let kk = nn::matmul(&hn, &wk).unwrap();
        let mut vv = nn::matmul(&hn, &wv).unwrap();
        for r in 0..n_ctx { for k in 0..n_state { vv.data[r * n_state + k] += bv[k]; } }

        for &h in &[0usize, n_head / 2, n_head - 1] {
            let b = h * d_head;
            let (n, d) = (real, d_head);
            // per-head contiguous Q,K,V [n,d]
            let mut qh = vec![0.0f32; n * d];
            let mut khd = vec![0.0f32; n * d];
            let mut vhd = vec![0.0f32; n * d];
            for i in 0..n {
                qh[i * d..(i + 1) * d].copy_from_slice(&q.data[i * n_state + b..i * n_state + b + d]);
                khd[i * d..(i + 1) * d].copy_from_slice(&kk.data[i * n_state + b..i * n_state + b + d]);
                vhd[i * d..(i + 1) * d].copy_from_slice(&vv.data[i * n_state + b..i * n_state + b + d]);
            }
            // EXACT: S=softmax(QKᵀ·scale)[n,n]; O=S@V  (+ time)
            let t0 = Instant::now();
            let mut sfull = qk_scaled(&qh, &khd, n, n, d, scale);
            softmax_rows(&mut sfull, n, n);
            let o_exact = mm(&sfull, &vhd, n, n, d);
            let t_exact = t0.elapsed().as_secs_f64() * 1e6;

            let mut line = format!("  L{l:>2} head {h:>2}: exact {t_exact:>8.0}us |");
            for &m in &[64usize, 128] {
                if m >= n { continue; }
                let t1 = Instant::now();
                // landmarks by segment-mean
                let seg = n / m;
                let mut qt = vec![0.0f32; m * d];
                let mut kt = vec![0.0f32; m * d];
                for g in 0..m {
                    let s0 = g * seg;
                    let s1 = if g == m - 1 { n } else { s0 + seg };
                    let cnt = (s1 - s0).max(1) as f32;
                    for i in s0..s1 { for t in 0..d { qt[g * d + t] += qh[i * d + t]; kt[g * d + t] += khd[i * d + t]; } }
                    for t in 0..d { qt[g * d + t] /= cnt; kt[g * d + t] /= cnt; }
                }
                let mut f = qk_scaled(&qh, &kt, n, m, d, scale); softmax_rows(&mut f, n, m); // [n,m]
                let mut a = qk_scaled(&qt, &kt, m, m, d, scale); softmax_rows(&mut a, m, m); // [m,m]
                let mut bmat = qk_scaled(&qt, &khd, m, n, d, scale); softmax_rows(&mut bmat, m, n); // [m,n]
                let ap = pinv(&a, m); // [m,m]
                let bv_ = mm(&bmat, &vhd, m, n, d); // [m,d]
                let apbv = mm(&ap, &bv_, m, m, d); // [m,d]
                let o_nys = mm(&f, &apbv, n, m, d); // [n,d]
                let t_nys = t1.elapsed().as_secs_f64() * 1e6;
                let mut diff = vec![0.0f32; n * d];
                for i in 0..n * d { diff[i] = o_nys[i] - o_exact[i]; }
                let relerr = frob(&diff) / frob(&o_exact).max(1e-12) * 100.0;
                line += &format!(" m{m}: {t_nys:>7.0}us {:>5.2}x relerr={relerr:>6.2}% |", t_exact / t_nys);
            }
            println!("{line}");
        }
    }
    println!("VIABLE iff (relerr ≪1%/layer, esp. at L32) AND (speed >1× vs exact per-head). Else owner-gated/reject.");
}
