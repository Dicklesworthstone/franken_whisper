// Scratch probe (bd-3nw3): where does write_mono_wav_i16 time go?
// Measures quantize-math alone vs full hound-writer path on identical input.
use std::time::Instant;

const SAMPLES: usize = 30_000_000; // ~11.3 min of 44.1kHz mono

fn main() {
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut state = 0x1234_5678_9abc_def0_u64;
    for _ in 0..SAMPLES {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // [-1,1)
        samples.push((state % 20_000) as f32 / 10_000.0 - 1.0);
    }

    // 1) quantize math only
    let t = Instant::now();
    let mut acc: i64 = 0;
    for s in &samples {
        let s = if s.is_finite() { *s } else { 0.0 };
        let q = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        acc += i64::from(q);
    }
    let quant_only = t.elapsed();
    println!("quantize-only : {quant_only:?} (checksum {acc})");

    // 2) full current path (chunked hound writer)
    let tmp = std::env::temp_dir().join("bd3nw3_probe.wav");
    let t = Instant::now();
    write_wav(&tmp, &samples);
    let full = t.elapsed();
    println!("full writer   : {full:?} -> {}", tmp.display());

    // 3) quantize into Vec<i16> first, then hound-write from i16 slices
    let t = Instant::now();
    let quantized = prequantized(&samples);
    let t_q = t.elapsed();
    let tmp2 = std::env::temp_dir().join("bd3nw3_probe2.wav");
    let t = Instant::now();
    write_pre(&tmp2, &quantized);
    let w = t.elapsed();
    println!(
        "pre-quant     : {t_q:?} + write {w:?} (total {:?}) -> {}",
        t_q + w,
        tmp2.display()
    );

    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&tmp2);
}

fn prequantized(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| {
            let s = if s.is_finite() { *s } else { 0.0 };
            (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
        })
        .collect()
}

fn write_wav(path: &std::path::Path, samples: &[f32]) {
    const CHUNK: usize = 8192;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for chunk in samples.chunks(CHUNK) {
        let mut buffered = writer.get_i16_writer(chunk.len() as u32);
        for s in chunk {
            let s = if s.is_finite() { *s } else { 0.0 };
            buffered.write_sample((s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16);
        }
        buffered.flush().unwrap();
    }
    writer.finalize().unwrap();
}

fn write_pre(path: &std::path::Path, q: &[i16]) {
    const CHUNK: usize = 8192;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for chunk in q.chunks(CHUNK) {
        let mut buffered = writer.get_i16_writer(chunk.len() as u32);
        for &s in chunk {
            buffered.write_sample(s);
        }
        buffered.flush().unwrap();
    }
    writer.finalize().unwrap();
}
