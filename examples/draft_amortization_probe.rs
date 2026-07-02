//! Draft/speculative-decoding amortization probe (BlackThrush, 2026-07-02).
//!
//! The turbo per-token decode is int8-weight-BANDWIDTH-bound: every token streams
//! ~922 MB of weights from DRAM (32 layers × ~26.8 MB + 66 MB tied logits) at
//! ~95 GB/s effective, so ~9.5 ms/step. int4 (halve bytes) is a measured kernel
//! wash and per-row fc2 breaks turbo — the ONLY lever that beats the wall is
//! reading the weights ONCE for MORE THAN ONE token. That is exactly what a
//! speculative/draft decoder does: a cheap draft proposes K tokens, the target
//! (turbo) VERIFIES all K in ONE `tq == K` forward pass (weights read once), and
//! accepts the longest correct prefix. Accepted tokens are BYTE-EXACT with plain
//! greedy decoding (they are precisely what the target would have emitted).
//!
//! This probe measures the enabling quantity — the weight-read amortization
//! ratio `R(K) = K · t(tq=1) / t(tq=K)` at a fixed cache length. `R(K)` is draft
//! decoding's speedup CEILING for the decode slice (realized × the draft
//! acceptance rate). It answers the owner's build/skip question with a number.
//!
//! CAVEAT: the current `tq > 1` path is the f16 batched GEMV (`gemv_f16_batch`),
//! NOT int8 — so the batched pass reads 2 bytes/weight while the sequential path
//! reads int8 (1 byte). `R(K)` here is thus the win available with TODAY's
//! kernels; an int8 batched GEMV (`gemv_i8_batch`, not yet written) would read
//! 1 byte/weight once and raise the ceiling further. Reported alongside.
//!
//! Uses min-of-iters (best-case = least-contended run) so the ratio reflects CPU
//! capability, not the shared box's transient load. Needs `FRANKEN_WHISPER_MODEL_DIR`;
//! set `FW_PROBE_MODEL=large-v3-turbo` for the real 32-layer decode.
//!
//! Usage: `draft_amortization_probe [iters]`  (default 100).
use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::decoder::{self, DecoderState};
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK, N_SAMPLES_30S, SAMPLE_RATE};
use std::time::Instant;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let model_name = std::env::var("FW_PROBE_MODEL").unwrap_or_else(|_| "tiny.en".to_string());
    let path = find_model_file(&model_name)
        .expect("set FRANKEN_WHISPER_MODEL_DIR to the ggml models dir");
    let model = GgmlModel::load(&path)
        .and_then(LoadedModel::from_ggml)
        .expect("load model");

    // Synthetic 30 s audio → mel → encoder output; the decode cost is set by the
    // model shapes, not the audio, so synthetic input is representative.
    let sr = SAMPLE_RATE as f32;
    let audio: Vec<f32> = (0..N_SAMPLES_30S)
        .map(|i| {
            let t = i as f32 / sr;
            0.9 * (0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin())
        })
        .collect();
    let full = mel::log_mel(&audio, &model.filters, 8).expect("log_mel");
    let window = mel::chunk_frames(&full, 0, FRAMES_PER_CHUNK);
    let noop = || Ok(());
    let enc_out = encoder::forward(&model.encoder, &window, 8, &noop).expect("encoder");

    let w = &model.decoder;
    let sot = model.tokenizer.sot;
    let mut st = DecoderState::new(w, &enc_out).expect("decoder state");

    // Fixed cache length L for BOTH measurements (fair: same cross-attn / same
    // self-attn cache depth). One `tq == L` prefill call fills the cache to L.
    let prefill_len = 16usize;
    let prefill: Vec<i32> = (0..prefill_len)
        .map(|i| if i == 0 { sot } else { 1 + i as i32 })
        .collect();

    // Measure the cost of a single `tq == k` forward at cache length L, best-of-iters.
    let mut measure = |k: usize| -> f64 {
        let batch: Vec<i32> = (0..k).map(|i| 100 + i as i32).collect();
        // Warm.
        for _ in 0..3 {
            st.reset();
            decoder::forward_step(w, &mut st, &prefill, &noop).expect("prefill");
            std::hint::black_box(
                decoder::forward_step(w, &mut st, &batch, &noop).expect("step").len(),
            );
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            st.reset();
            decoder::forward_step(w, &mut st, &prefill, &noop).expect("prefill");
            let t = Instant::now();
            let logits = decoder::forward_step(w, &mut st, &batch, &noop).expect("step");
            best = best.min(t.elapsed().as_secs_f64());
            std::hint::black_box(logits.len());
        }
        best
    };

    let t1 = measure(1);
    println!(
        "draft_amortization_probe[{}]  cache_len={}  best-of-{}  (tq=1 baseline {:.3} ms)",
        model_name,
        prefill_len,
        iters,
        t1 * 1e3
    );
    println!("   K   t(tq=K) ms   per-token ms   amortization R(K)=K*t1/tK");
    for k in [1usize, 2, 4, 8] {
        let tk = if k == 1 { t1 } else { measure(k) };
        let per_tok = tk / k as f64;
        let r = k as f64 * t1 / tk;
        println!(
            "  {k:>2}   {:>9.3}   {:>10.3}   {:>6.2}x",
            tk * 1e3,
            per_tok * 1e3,
            r
        );
    }
    println!(
        "  R(K) = draft-decoding decode-slice speedup CEILING (× accept-rate). \
         tq>1 path is f16 today; an int8 batched GEMV would raise it."
    );
}
