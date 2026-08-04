use std::hint::black_box;
use std::io::Cursor;
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 16_000;
const SAMPLE_COUNT: usize = SAMPLE_RATE as usize * 30;
const PROFILE_REPS: usize = 9;
const PAIRED_REPS: usize = 15;
const WRITE_CHUNK_SAMPLES: usize = 8_192;
const TARGET_ARM_SECS: f64 = 0.050;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 40) as f32 / ((1u32 << 24) - 1) as f32;
        unit * 2.0 - 1.0
    }
}

#[inline(always)]
fn quantize(sample: f32) -> i16 {
    let sanitized = if sample.is_finite() { sample } else { 0.0 };
    (sanitized.max(-1.0).min(1.0) * f32::from(i16::MAX)).round() as i16
}

fn fixture() -> Vec<f32> {
    let mut lcg = Lcg(0x57a7_1600_0030);
    (0..SAMPLE_COUNT).map(|_| lcg.next_f32()).collect()
}

fn wav_spec() -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

fn quantize_only(samples: &[f32]) -> Vec<i16> {
    samples.iter().copied().map(quantize).collect()
}

fn historical_wav(samples: &[f32]) -> Vec<u8> {
    let mut output = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut output, wav_spec()).expect("create WAV");
        for &sample in samples {
            writer
                .write_sample(quantize(sample))
                .expect("write historical sample");
        }
        writer.finalize().expect("finalize historical WAV");
    }
    output.into_inner()
}

fn buffered_wav(samples: &[f32]) -> Vec<u8> {
    let mut output = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut output, wav_spec()).expect("create WAV");
        for chunk in samples.chunks(WRITE_CHUNK_SAMPLES) {
            let mut buffered = writer.get_i16_writer(chunk.len() as u32);
            for &sample in chunk {
                buffered.write_sample(quantize(sample));
            }
            buffered.flush().expect("flush buffered samples");
        }
        writer.finalize().expect("finalize buffered WAV");
    }
    output.into_inner()
}

fn timed<T>(operation: impl FnOnce() -> T) -> (Duration, T) {
    let started = Instant::now();
    let output = operation();
    (started.elapsed(), output)
}

fn median_ns(values: &[u128]) -> u128 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn measure_arm(
    samples: &[f32],
    implementation: fn(&[f32]) -> Vec<u8>,
    inner_steps: usize,
) -> Duration {
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..inner_steps {
        let output = implementation(black_box(samples));
        checksum ^= output.len();
        black_box(output);
    }
    black_box(checksum);
    started.elapsed()
}

fn paired_ratios(
    samples: &[f32],
    first: fn(&[f32]) -> Vec<u8>,
    second: fn(&[f32]) -> Vec<u8>,
    inner_steps: usize,
    repetitions: usize,
) -> Vec<f64> {
    let mut ratios = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let (first_elapsed, second_elapsed) = if repetition.is_multiple_of(2) {
            let first_before = measure_arm(samples, first, inner_steps);
            let second_before = measure_arm(samples, second, inner_steps);
            let second_after = measure_arm(samples, second, inner_steps);
            let first_after = measure_arm(samples, first, inner_steps);
            (first_before + first_after, second_before + second_after)
        } else {
            let second_before = measure_arm(samples, second, inner_steps);
            let first_before = measure_arm(samples, first, inner_steps);
            let first_after = measure_arm(samples, first, inner_steps);
            let second_after = measure_arm(samples, second, inner_steps);
            (first_before + first_after, second_before + second_after)
        };
        ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
    }
    ratios
}

fn assert_parity() {
    let edge_values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.25,
        -1.25,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.5,
        -0.5,
        f32::EPSILON,
        -f32::EPSILON,
    ];
    for &len in &[0usize, 1, 8_191, 8_192, 8_193, 16_391] {
        let samples: Vec<f32> = edge_values.iter().copied().cycle().take(len).collect();
        assert_eq!(
            buffered_wav(&samples),
            historical_wav(&samples),
            "WAV byte parity failed at len={len}"
        );
    }
}

fn main() {
    let samples = fixture();
    let reference = historical_wav(&samples);
    let candidate = buffered_wav(&samples);
    assert_eq!(candidate, reference, "30-second WAV byte parity");
    assert_parity();
    assert_eq!(reference.len(), 44 + samples.len() * 2);
    println!("WAV_PROFILE_FNV64={:016x}", fnv64(&reference));
    let executable = std::fs::read(std::env::current_exe().expect("benchmark path"))
        .expect("read benchmark binary");
    println!("WAV_BINARY_FNV64={:016x}", fnv64(&executable));

    for _ in 0..3 {
        black_box(quantize_only(black_box(&samples)));
        black_box(historical_wav(black_box(&samples)));
    }

    let mut quantize_ns = Vec::with_capacity(PROFILE_REPS);
    let mut historical_wav_ns = Vec::with_capacity(PROFILE_REPS);
    for _ in 0..PROFILE_REPS {
        let (quantize_elapsed, quantized) = timed(|| quantize_only(black_box(&samples)));
        black_box(quantized);
        quantize_ns.push(quantize_elapsed.as_nanos());

        let (wav_elapsed, wav) = timed(|| historical_wav(black_box(&samples)));
        black_box(wav);
        historical_wav_ns.push(wav_elapsed.as_nanos());
    }

    let quantize_median = median_ns(&quantize_ns);
    let wav_median = median_ns(&historical_wav_ns);
    let attributed_writer_share = 1.0 - quantize_median as f64 / wav_median as f64;
    println!("WAV_PROFILE_QUANTIZE_NS={quantize_ns:?}");
    println!("WAV_PROFILE_HISTORICAL_NS={historical_wav_ns:?}");
    println!(
        "WAV_PROFILE_MEDIANS quantize_ns={quantize_median} historical_wav_ns={wav_median} attributed_writer_share={attributed_writer_share:.6}"
    );

    let calibration = measure_arm(&samples, historical_wav, 1);
    let inner_steps = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
    let inner_steps = inner_steps.clamp(16, 256);

    black_box(paired_ratios(
        &samples,
        historical_wav,
        historical_wav,
        inner_steps,
        3,
    ));
    let null_ratios = paired_ratios(
        &samples,
        historical_wav,
        historical_wav,
        inner_steps,
        PAIRED_REPS,
    );
    black_box(paired_ratios(
        &samples,
        historical_wav,
        buffered_wav,
        inner_steps,
        3,
    ));
    let candidate_ratios = paired_ratios(
        &samples,
        historical_wav,
        buffered_wav,
        inner_steps,
        PAIRED_REPS,
    );
    let null_median = percentile(&null_ratios, 50);
    let null_p90 = percentile(&null_ratios, 90);
    let candidate_p10 = percentile(&candidate_ratios, 10);
    let candidate_median = percentile(&candidate_ratios, 50);
    let candidate_p90 = percentile(&candidate_ratios, 90);
    let wins = candidate_ratios
        .iter()
        .filter(|&&ratio| ratio > 1.0)
        .count();
    println!(
        "WAV_CALIBRATION historical_ns={} inner_steps={} target_arm_ms={:.1}",
        calibration.as_nanos(),
        inner_steps,
        TARGET_ARM_SECS * 1_000.0
    );
    println!("BASE_BASE_RATIOS={null_ratios:?}");
    println!("HISTORICAL_BUFFERED_RATIOS={candidate_ratios:?}");
    println!("NULL_MEDIAN={null_median:.6} NULL_P90={null_p90:.6}");
    println!(
        "CANDIDATE_P10={candidate_p10:.6} CANDIDATE_MEDIAN={candidate_median:.6} CANDIDATE_P90={candidate_p90:.6} WINS={wins}/{PAIRED_REPS}"
    );
}
