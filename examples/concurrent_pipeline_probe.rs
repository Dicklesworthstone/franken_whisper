//! Window-pipelining viability probe (BlackThrush, 2026-07-02).
//!
//! The LAST byte-exact structural lever is cross-window pipelining: in
//! no_timestamps mode windows are independent, so decode-N can overlap
//! encode-(N+1). Memory ([[project_window_pipelining_lever]]) only ESTIMATES the
//! reclaim ("decode's idle-core fraction"); this MEASURES it. Key hypothesis: the
//! encoder is COMPUTE-bound (tiled f32 sgemm, ~10 GB/s, saturates FMA) and the
//! decode is BANDWIDTH-bound (int8 weight stream, ~97 GB/s, cores stall on
//! memory) — complementary profiles, so on this SMT2 box running them
//! concurrently on the shared 32-thread rayon pool could approach max(enc,dec)
//! rather than enc+dec (pipelining would then hide most of the decode).
//!
//! Sequential baseline: t(encode window) + t(decode N tokens), measured apart.
//! Concurrent: encode(window) on a scoped thread WHILE decode(N tokens) runs on
//! the main thread (both hit the global rayon pool — the real pipelining setup).
//! reclaim = seq_total - concurrent; complementarity = reclaim / min(enc,dec).
//! Needs FRANKEN_WHISPER_MODEL_DIR; FW_PROBE_MODEL=large-v3-turbo.
//! Usage: `concurrent_pipeline_probe [decode_tokens]` (default 200).
use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::decoder::{self, DecoderState};
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK, N_SAMPLES_30S, SAMPLE_RATE};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let n_tok: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let model_name = std::env::var("FW_PROBE_MODEL").unwrap_or_else(|_| "tiny.en".to_string());
    let path = find_model_file(&model_name).expect("set FRANKEN_WHISPER_MODEL_DIR");
    let model = GgmlModel::load(&path).and_then(LoadedModel::from_ggml).expect("load model");

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
    let enc_out = encoder::forward(&model.encoder, &window, 0, &noop).expect("encoder");

    let w = &model.decoder;
    let sot = model.tokenizer.sot;
    let seq: Vec<i32> = (0..n_tok).map(|i| if i == 0 { sot } else { 1 + (i as i32 % 50000) }).collect();
    let mut st = DecoderState::new(w, &enc_out).expect("decoder state");

    // Run one full decode pass (cache grows 0..n_tok), the real single-window shape.
    let decode_once = |st: &mut DecoderState| {
        st.reset();
        for &tok in &seq {
            black_box(decoder::forward_step(w, st, &[tok], &noop).expect("step").len());
        }
    };
    let encode_once = || black_box(encoder::forward(&model.encoder, &window, 0, &noop).expect("enc").rows);

    // Warm.
    for _ in 0..2 {
        encode_once();
        decode_once(&mut st);
    }

    // Sequential timings (best-of-5 each, min = least-contended).
    let mut t_enc = f64::INFINITY;
    let mut t_dec = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        encode_once();
        t_enc = t_enc.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        decode_once(&mut st);
        t_dec = t_dec.min(t.elapsed().as_secs_f64());
    }

    // Concurrent: encode on a scoped thread WHILE decode runs on main (both share
    // the global rayon pool — exactly what a pipelined transcribe would do).
    let mut t_conc = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        std::thread::scope(|s| {
            let enc = &model.encoder;
            let win = &window;
            let noop2 = || Ok(());
            s.spawn(move || black_box(encoder::forward(enc, win, 0, &noop2).expect("enc").rows));
            decode_once(&mut st);
        });
        t_conc = t_conc.min(t.elapsed().as_secs_f64());
    }

    let seq_total = t_enc + t_dec;
    let reclaim = seq_total - t_conc;
    let complementarity = reclaim / t_enc.min(t_dec);
    println!("concurrent_pipeline_probe[{model_name}]  decode_tokens={n_tok}  best-of-5 (min):");
    println!("  encode alone:      {:>7.1} ms", t_enc * 1e3);
    println!("  decode alone:      {:>7.1} ms  ({:.2} ms/tok)", t_dec * 1e3, t_dec * 1e3 / n_tok as f64);
    println!("  SEQUENTIAL total:  {:>7.1} ms", seq_total * 1e3);
    println!("  CONCURRENT total:  {:>7.1} ms", t_conc * 1e3);
    println!("  reclaim:           {:>7.1} ms  ({:.0}% of the smaller phase)", reclaim * 1e3, complementarity * 100.0);
    let max_phase = t_enc.max(t_dec);
    println!(
        "  concurrent vs max(enc,dec)={:.1} ms => {}",
        max_phase * 1e3,
        if t_conc <= max_phase * 1.08 { "~COMPLEMENTARY (pipelining hides the smaller phase — BUILD IT)" }
        else if reclaim > 0.15 * t_enc.min(t_dec) { "PARTIAL overlap (some reclaim)" }
        else { "CONTENDED (no meaningful reclaim — pipelining not worth it)" }
    );
}
