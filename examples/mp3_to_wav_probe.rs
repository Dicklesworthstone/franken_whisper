//! Throwaway: decode any audio (mp3/…) to a normalized 16 kHz mono WAV via the
//! crate's own `audio::normalize_to_wav`, so e2e_probe (WAV-only) can drive the
//! NATIVE `transcribe_samples` on it. Usage: `mp3_to_wav_probe <in> <out_wav>`.
use franken_whisper::audio::normalize_to_wav;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    let work = out.parent().unwrap_or(Path::new("."));
    let wav = normalize_to_wav(input, work).expect("normalize_to_wav");
    // normalize_to_wav writes into work_dir with its own name; copy to `out`.
    if wav != out {
        std::fs::copy(&wav, out).expect("copy wav");
    }
    println!("{}", out.display());
}
