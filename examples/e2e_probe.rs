//! Throwaway e2e probe (NOT committed): drive native_engine::transcribe_samples
//! directly on arbitrary audio/model/repeat-count to measure realistic e2e
//! wall-clock + RTF, including long-form (tiled audio) and word-timestamp paths.
//!
//! Usage: e2e_probe <model_short> <wav> <repeat> [wordts]
//!   model_short: tiny.en | large-v3-turbo
//!   repeat:      tile the audio N times to synthesize long-form
//!   wordts:      pass the literal "wordts" to enable DTW word timestamps

use franken_whisper::native_engine::decode::{DecodeParams, LoadedModel, transcribe_samples};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use std::time::Instant;

/// Minimal robust WAV reader: locate the `data` chunk, parse PCM16 mono/stereo,
/// downmix to mono, return f32 in [-1,1]. Assumes 16 kHz (whisper standard).
fn read_wav_mono16k(path: &str) -> (Vec<f32>, u32, u16) {
    let bytes = std::fs::read(path).expect("read wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut pos = 12;
    let mut channels = 1u16;
    let mut rate = 16000u32;
    let mut bits = 16u16;
    let mut data: &[u8] = &[];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
            rate = u32::from_le_bytes([
                bytes[body + 4],
                bytes[body + 5],
                bytes[body + 6],
                bytes[body + 7],
            ]);
            bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
        } else if id == b"data" {
            data = &bytes[body..(body + sz).min(bytes.len())];
        }
        pos = body + sz + (sz & 1);
    }
    assert_eq!(bits, 16, "expected PCM16");
    let n = data.len() / 2;
    let mut samples = Vec::with_capacity(n / channels as usize);
    let mut i = 0;
    while i + 2 * channels as usize <= data.len() {
        let mut acc = 0i32;
        for c in 0..channels as usize {
            let s = i16::from_le_bytes([data[i + 2 * c], data[i + 2 * c + 1]]);
            acc += s as i32;
        }
        samples.push((acc as f32 / channels as f32) / 32768.0);
        i += 2 * channels as usize;
    }
    (samples, rate, channels)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_short = args.get(1).map(String::as_str).unwrap_or("tiny.en");
    let wav = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tests/fixtures/native/jfk.wav".to_string());
    let repeat: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let wordts = args.get(4).map(|s| s == "wordts").unwrap_or(false);

    let path = find_model_file(model_short)
        .unwrap_or_else(|| panic!("model {model_short} not found in search dirs"));
    let t_load = Instant::now();
    let model = GgmlModel::load(&path)
        .and_then(LoadedModel::from_ggml)
        .expect("load model");
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let (base, rate, ch) = read_wav_mono16k(&wav);
    assert_eq!(rate, 16000, "probe assumes 16kHz input");
    let mut samples = Vec::with_capacity(base.len() * repeat);
    for _ in 0..repeat {
        samples.extend_from_slice(&base);
    }
    let audio_sec = samples.len() as f64 / 16000.0;

    let params = DecodeParams {
        language: Some("en".to_string()),
        translate: false,
        timestamps: std::env::var("PROBE_NO_TS").as_deref() != Ok("1"),
        n_threads: 0,
        max_text_ctx: None,
        word_timestamps: wordts,
        model_hint: Some(model_short.to_string()),
    };

    // warm (mmap/page-in) then timed
    let t = Instant::now();
    let out = transcribe_samples(&model, &samples, &params, &(|| Ok(()))).expect("transcribe");
    let dt = t.elapsed().as_secs_f64();

    // Per-sub-part decode attribution (only populated under
    // FRANKEN_WHISPER_PERF_SPANS=1; thread-local on this calling thread).
    if std::env::var("FRANKEN_WHISPER_PERF_SPANS").as_deref() == Ok("1") {
        use franken_whisper::native_engine::decoder::{SUB_LABELS, take_sub_ns};
        let ns = take_sub_ns();
        let total: u128 = ns.iter().sum();
        eprintln!("--- forward_step sub-part breakdown (sum over all tokens) ---");
        let mut idx: Vec<usize> = (0..ns.len()).collect();
        idx.sort_by(|&a, &b| ns[b].cmp(&ns[a]));
        for i in idx {
            let ms = ns[i] as f64 / 1e6;
            let pct = if total > 0 {
                ns[i] as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            eprintln!("  {:<18} {:>8.1} ms  {:>5.1}%", SUB_LABELS[i], ms, pct);
        }
        eprintln!(
            "  {:<18} {:>8.1} ms (forward_step total)",
            "SUM",
            total as f64 / 1e6
        );
    }

    // Layer-skip self-draft accept rate (only when FW_DRAFT_ACCEPT_LAYERS is set).
    if let Ok(k) = std::env::var("FW_DRAFT_ACCEPT_LAYERS") {
        let (m, tot) = franken_whisper::native_engine::decoder::drain_draft_accept();
        let pct = if tot > 0 { 100.0 * m as f64 / tot as f64 } else { 0.0 };
        eprintln!(
            "DRAFT_ACCEPT k={k} layers: {m}/{tot} decode steps matched full argmax = {pct:.1}% accept"
        );
    }

    let chars: usize = out.segments.iter().map(|s| s.text.len()).sum();
    if std::env::var("PROBE_DUMP_TEXT").as_deref() == Ok("1") {
        let full: String = out.segments.iter().map(|s| s.text.as_str()).collect();
        eprintln!("TRANSCRIPT>>>{full}<<<");
    }
    // Segment-level timestamp dump (PROBE_DUMP_SEGS=1): one line per segment as
    // `SEG i [t0 -> t1] text`, matching whisper.cpp's `[HH:MM:SS.mmm -->
    // HH:MM:SS.mmm]` layout for a franken-vs-whisper.cpp timestamp diff.
    if std::env::var("PROBE_DUMP_SEGS").as_deref() == Ok("1") {
        for (i, s) in out.segments.iter().enumerate() {
            eprintln!(
                "SEG {i} [{:.3} -> {:.3}] {}",
                s.start_sec.unwrap_or(f64::NAN),
                s.end_sec.unwrap_or(f64::NAN),
                s.text.trim()
            );
        }
    }
    // Word-level DTW timestamp dump (PROBE_DUMP_WORDS=1, needs the `wordts` arg):
    // one line per word as `WORD [t0 -> t1] text`, for a franken-vs-whisper.cpp
    // (`-dtw <model> -ml 1`) word-timestamp diff.
    if std::env::var("PROBE_DUMP_WORDS").as_deref() == Ok("1") {
        match out.word_timings.as_ref() {
            Some(wt) => {
                for w in wt.iter().flatten() {
                    eprintln!("WORD [{:.3} -> {:.3}] {}", w.start_sec, w.end_sec, w.text);
                }
            }
            None => eprintln!("WORD (none — word_timestamps not enabled; pass `wordts`)"),
        }
    }
    let rtf = dt / audio_sec;
    println!(
        "model={model_short} wav={wav} repeat={repeat} wordts={wordts} ch={ch} | audio={audio_sec:.1}s load={load_ms:.0}ms | transcribe={:.3}s RTF={rtf:.4} | segs={} chars={chars}",
        dt,
        out.segments.len()
    );
}
