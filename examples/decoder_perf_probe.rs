//! Contention-immune per-token DECODE perf probe.
//!
//! Loads `tiny.en` once, builds the encoder output + `DecoderState` once (the
//! one-time setup), then loops a FIXED number of `forward_step`s so
//! `perf stat -e instructions:u` isolates the *per-token* decode instruction
//! count from the encoder/`DecoderState::new` setup that dominates (and
//! contaminates) the model-gated `decoder_token_step` Criterion bench. Mirrors
//! `mel_perf_probe`. Needs `FRANKEN_WHISPER_MODEL_DIR` pointing at the ggml models.
//!
//! Usage: `decoder_perf_probe [iters]`  (default 200; each iter = reset + 8 steps).
use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::decoder::{self, DecoderState};
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK, N_SAMPLES_30S, SAMPLE_RATE};

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let path =
        find_model_file("tiny.en").expect("set FRANKEN_WHISPER_MODEL_DIR to the ggml models dir");
    let model = GgmlModel::load(&path)
        .and_then(LoadedModel::from_ggml)
        .expect("load tiny.en");

    // Synthetic 30 s audio → mel window → encoder output. The per-token decode
    // cost is determined by the model shapes, not the audio content, so synthetic
    // input is representative (and keeps the probe self-contained, no jfk fixture).
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
    let seq: Vec<i32> = (0..8).map(|i| if i == 0 { sot } else { 1 + i }).collect();
    let mut st = DecoderState::new(w, &enc_out).expect("decoder state");

    // TIMED region: pure per-token decode (reset retains cross-K/V, regrows cache 0->8).
    let mut acc = 0usize;
    for _ in 0..iters {
        st.reset();
        for &tok in &seq {
            let logits = decoder::forward_step(w, &mut st, &[tok], &noop).expect("step");
            acc = acc.wrapping_add(logits.len());
        }
    }
    std::hint::black_box(acc);
    eprintln!(
        "decoder_perf_probe: iters={iters} steps={} acc={acc}",
        iters * seq.len()
    );
}
