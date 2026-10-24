// Probe (bd-3nw3): the f32 -> i16 quantize/write lever. Times the production
// chunked hound writer (arm A, a copy of `audio::write_mono_wav_i16`) against
// the pre-quantize-then-write candidate (arm B) on identical input, interleaved
// in one invocation with a second production arm (A') as the same-binary A/A
// null. Arm order rotates every round. Prints per-arm medians and bootstrap
// CI95s of the per-round ratios A'/A (null) and B/A (candidate).
//
//   cargo run --release --example bd3nw3_write_probe -- [rounds]
//
// Files go to `std::env::temp_dir()` (set TMPDIR to choose the disk).
use std::path::Path;
use std::time::Instant;

const SAMPLES: usize = 30_000_000; // ~11.3 min of 44.1kHz mono
const DEFAULT_ROUNDS: usize = 15;
const ARMS: [&str; 3] = ["A", "A'", "B"];

fn main() {
    let rounds = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&rounds| rounds >= 3)
        .unwrap_or(DEFAULT_ROUNDS);
    let samples = synthetic_samples();
    let path = std::env::temp_dir().join("bd3nw3_probe.wav");

    // Quantize math alone, for the stage-size context of the ledger row.
    let t = Instant::now();
    let mut checksum: i64 = 0;
    for sample in &samples {
        checksum += i64::from(quantize(*sample));
    }
    println!(
        "quantize-only: {:.1} ms (checksum {checksum})",
        t.elapsed().as_secs_f64() * 1e3
    );

    // One untimed run of each arm warms the page cache and the allocator.
    write_production(&path, &samples);
    write_prequantized(&path, &samples);

    let mut ms = vec![[0.0f64; 3]; rounds];
    let mut bytes = [0u64; 3];
    for (round, row) in ms.iter_mut().enumerate() {
        for offset in 0..3 {
            let arm = (round + offset) % 3;
            let t = Instant::now();
            if arm == 2 {
                write_prequantized(&path, &samples);
            } else {
                write_production(&path, &samples);
            }
            row[arm] = t.elapsed().as_secs_f64() * 1e3;
            bytes[arm] = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let _ = std::fs::remove_file(&path);
        }
    }
    assert!(
        bytes[0] == bytes[1] && bytes[1] == bytes[2],
        "arms wrote different byte counts: {bytes:?}"
    );

    for (arm, name) in ARMS.iter().enumerate() {
        let mut column: Vec<f64> = ms.iter().map(|row| row[arm]).collect();
        println!(
            "arm {name}: median {:.1} ms over {rounds} rounds ({} bytes written)",
            median(&mut column),
            bytes[arm]
        );
    }
    let null: Vec<f64> = ms.iter().map(|row| row[1] / row[0]).collect();
    let candidate: Vec<f64> = ms.iter().map(|row| row[2] / row[0]).collect();
    let (null_median, null_lo, null_hi) = bootstrap_median_ci95(&null);
    let (cand_median, cand_lo, cand_hi) = bootstrap_median_ci95(&candidate);
    println!(
        "same-invocation A/A null A'/A: median {null_median:.4}, bootstrap CI95 [{null_lo:.4}, {null_hi:.4}]"
    );
    println!("candidate B/A: median {cand_median:.4}, bootstrap CI95 [{cand_lo:.4}, {cand_hi:.4}]");
    let widest = (null_hi - 1.0).abs().max((1.0 - null_lo).abs());
    println!(
        "2x null margin: decidable iff the candidate CI95 lies outside [{:.4}, {:.4}]",
        1.0 - 2.0 * widest,
        1.0 + 2.0 * widest
    );
}

fn synthetic_samples() -> Vec<f32> {
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut state = 0x1234_5678_9abc_def0_u64;
    for _ in 0..SAMPLES {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // [-1,1)
        samples.push((state % 20_000) as f32 / 10_000.0 - 1.0);
    }
    samples
}

/// The production sample conversion (`audio::write_mono_wav_i16_with_checkpoint`).
#[allow(
    clippy::manual_clamp,
    reason = "mirrors production, which avoids a documented aarch64 nightly clamp miscompile"
)]
fn quantize(sample: f32) -> i16 {
    let s = if sample.is_finite() { sample } else { 0.0 };
    (s.max(-1.0).min(1.0) * f32::from(i16::MAX)).round() as i16
}

fn spec() -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

/// Arm A / A': the production writer, quantizing inside the chunked loop.
fn write_production(path: &Path, samples: &[f32]) {
    const CHUNK: usize = 8192;
    let mut writer = hound::WavWriter::create(path, spec()).expect("create wav");
    for chunk in samples.chunks(CHUNK) {
        let mut buffered = writer.get_i16_writer(chunk.len() as u32);
        for sample in chunk {
            buffered.write_sample(quantize(*sample));
        }
        buffered.flush().expect("flush wav chunk");
    }
    writer.finalize().expect("finalize wav");
}

/// Arm B: quantize everything into a `Vec<i16>` first, then write it.
fn write_prequantized(path: &Path, samples: &[f32]) {
    const CHUNK: usize = 8192;
    let quantized: Vec<i16> = samples.iter().map(|sample| quantize(*sample)).collect();
    let mut writer = hound::WavWriter::create(path, spec()).expect("create wav");
    for chunk in quantized.chunks(CHUNK) {
        let mut buffered = writer.get_i16_writer(chunk.len() as u32);
        for &sample in chunk {
            buffered.write_sample(sample);
        }
        buffered.flush().expect("flush wav chunk");
    }
    writer.finalize().expect("finalize wav");
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Median and percentile-bootstrap CI95 of the median (10,000 resamples,
/// fixed seed so a rerun on the same timings prints the same interval).
fn bootstrap_median_ci95(values: &[f64]) -> (f64, f64, f64) {
    const RESAMPLES: usize = 10_000;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut medians = Vec::with_capacity(RESAMPLES);
    let mut resample = vec![0.0; values.len()];
    for _ in 0..RESAMPLES {
        for slot in &mut resample {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *slot = values[(state % values.len() as u64) as usize];
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    let lo = medians[RESAMPLES * 25 / 1000];
    let hi = medians[RESAMPLES * 975 / 1000 - 1];
    (median(&mut values.to_vec()), lo, hi)
}
