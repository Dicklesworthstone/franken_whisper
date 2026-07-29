//! Speculative-decode ACCEPT-RATE probe (bd-wzgh viability).
//!
//! Measures the TRUE per-token accept rate for a draft/verify pair: greedily
//! decodes one window with the VERIFY model to get its committed tokens, then
//! TEACHER-FORCES the DRAFT on that exact token stream (feeding the verify tokens,
//! comparing the draft's argmax at each position to the verify model's actual next
//! token). That fraction is what bounds the spec-decode speedup — it is HIGHER than
//! the independent-transcript agreement (which diverges after the first mismatch;
//! teacher-forcing re-syncs every step, exactly like real spec-decode).
//!
//! The encoder is SHARED (distil-large-v3 and large-v3-turbo carry the same 32-layer
//! large-v3 encoder), so the verify model's encoder output feeds BOTH decoders'
//! cross-attention — the real spec-decode setup (encode once, draft+verify decode).
//!
//! Usage:
//!   FRANKEN_WHISPER_MODEL_DIR=<dir> \
//!   cargo run --release --example spec_accept_probe -- <verify.bin> <draft.bin> <clip16k.wav>

use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::decoder::{self, DecoderState};
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK};

const THREADS: usize = 16;

fn argmax(l: &[f32]) -> i32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as i32
}

fn read_wav_16k_mono(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("open wav");
    let spec = r.spec();
    assert_eq!(spec.sample_rate, 16_000, "probe expects 16 kHz wav");
    assert_eq!(spec.channels, 1, "probe expects mono wav");
    r.samples::<i16>()
        .map(|s| f32::from(s.expect("sample")) / 32768.0)
        .collect()
}

/// Greedy-decode ONE window; returns the committed text token ids (stops at eot).
fn greedy(
    model: &LoadedModel,
    enc: &franken_whisper::native_engine::Mat,
    prompt: &[i32],
) -> Vec<i32> {
    let noop = || Ok(());
    let mut st = DecoderState::new(&model.decoder, enc).expect("state");
    let mut logits =
        decoder::forward_step(&model.decoder, &mut st, prompt, &noop).expect("prefill");
    let eot = model.tokenizer.eot;
    let mut toks = Vec::new();
    for _ in 0..224 {
        let tok = argmax(&logits);
        if tok >= eot {
            break; // eot or any special token ends this greedy window
        }
        toks.push(tok);
        logits = decoder::forward_step(&model.decoder, &mut st, &[tok], &noop).expect("step");
    }
    toks
}

/// Teacher-force `draft` on `verify_toks`: at each position compare the draft's
/// argmax (given the verify prefix) to the verify model's actual next token.
fn teacher_forced_accept(
    draft: &LoadedModel,
    enc: &franken_whisper::native_engine::Mat,
    prompt: &[i32],
    verify_toks: &[i32],
) -> (usize, usize) {
    let noop = || Ok(());
    let mut st = DecoderState::new(&draft.decoder, enc).expect("draft state");
    // Prime with the same prompt; its last-position logits predict verify_toks[0].
    let mut logits =
        decoder::forward_step(&draft.decoder, &mut st, prompt, &noop).expect("draft prefill");
    let (mut accept, mut total) = (0usize, 0usize);
    for &vt in verify_toks {
        if argmax(&logits) == vt {
            accept += 1;
        }
        total += 1;
        logits = decoder::forward_step(&draft.decoder, &mut st, &[vt], &noop).expect("draft step");
    }
    (accept, total)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: spec_accept_probe <verify.bin> <draft.bin> <clip16k.wav>");
        std::process::exit(2);
    }
    let verify = GgmlModel::load(std::path::Path::new(&a[1]))
        .and_then(LoadedModel::from_ggml)
        .expect("load verify");
    let draft = GgmlModel::load(std::path::Path::new(&a[2]))
        .and_then(LoadedModel::from_ggml)
        .expect("load draft");

    let audio = read_wav_16k_mono(&a[3]);
    let full = mel::log_mel(&audio, &verify.filters, THREADS).expect("mel");
    let window = mel::chunk_frames(&full, 0, FRAMES_PER_CHUNK);
    // SHARED encoder output (verify's encoder == draft's; feeds both cross-attentions).
    let enc = encoder::forward(&verify.encoder, &window, THREADS, &|| Ok(())).expect("encoder");

    let prompt = verify.tokenizer.sot_sequence(Some("en"), false, false);
    let vtoks = greedy(&verify, &enc, &prompt);
    println!(
        "verify: {} committed tokens -> \"{}\"",
        vtoks.len(),
        verify.tokenizer.decode(&vtoks).trim()
    );
    if vtoks.is_empty() {
        println!("no tokens decoded; cannot measure accept rate");
        return;
    }
    let (acc, tot) = teacher_forced_accept(&draft, &enc, &prompt, &vtoks);
    println!(
        "draft teacher-forced accept (K=1): {}/{} = {:.1}%  (independent-transcript agreement is a LOWER bound)",
        acc,
        tot,
        100.0 * acc as f64 / tot as f64
    );
}
