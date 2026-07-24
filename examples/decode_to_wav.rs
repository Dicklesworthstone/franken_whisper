//! Decode any symphonia-supported audio file (MP3/AAC/FLAC/OGG/WAV/…) to a
//! 16 kHz mono PCM WAV via fw's built-in normalizer — no ffmpeg/sox needed.
//!
//! Purpose: `whisper-cli` (the WER reference) cannot read `.mp3`, so the
//! discriminating real-speech clips under `sample_audio_files/` had no
//! whisper.cpp reference. This turns them into `.wav`s the reference can read,
//! unblocking corpus-WER evaluation of the owner-gated encoder-int8 / poly-exp
//! levers (see docs/PERF_FRONTIER.md "Corpus WER").
//!
//! Usage: `cargo run --release --example decode_to_wav -- <input> <output.wav>`

use franken_whisper::audio::normalize_to_wav;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: decode_to_wav <input-audio> <output.wav>");
        std::process::exit(2);
    }
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);

    // `normalize_to_wav` writes `normalized_16k_mono.wav` into a work dir, so
    // decode into the output's parent then rename to the requested name.
    let work_dir = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(work_dir)?;
    let produced = normalize_to_wav(input, work_dir)?;
    if produced != output {
        std::fs::rename(&produced, output)?;
    }
    println!("{}", output.display());
    Ok(())
}
