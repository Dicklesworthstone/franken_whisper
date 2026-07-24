use std::hint::black_box;
use std::time::{Duration, Instant};

const FRAME_SAMPLES: usize = 320;
const READ_BUFFER_BYTES: usize = 8192;

fn make_pcm(sample_count: usize) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(sample_count * 2);
    for index in 0..sample_count {
        let value = ((index.wrapping_mul(7919).wrapping_add(index / 17)) & 0xffff) as u16;
        pcm.extend_from_slice(&(value as i16).to_le_bytes());
    }
    pcm
}

fn frame_rms(samples: &[i16], frame_samples: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for chunk in samples.chunks(frame_samples.max(1)) {
        let sum_sq = chunk.iter().fold(0.0_f64, |acc, value| {
            let normalized = (*value as f64) / 32768.0;
            acc + (normalized * normalized)
        });
        out.push((sum_sq / chunk.len() as f64).sqrt() as f32);
    }
    out
}

fn decode_then_rms(pcm: &[u8], frame_samples: usize) -> Vec<f32> {
    let mut samples = Vec::with_capacity(pcm.len() / 2);
    let mut leftover = None;
    for buffer in pcm.chunks(READ_BUFFER_BYTES) {
        let mut start = 0usize;
        if let Some(previous) = leftover.take() {
            samples.push(i16::from_le_bytes([previous, buffer[0]]));
            start = 1;
        }
        let (pairs, remainder) = buffer[start..].as_chunks::<2>();
        for pair in pairs {
            samples.push(i16::from_le_bytes(*pair));
        }
        leftover = remainder.first().copied();
    }
    frame_rms(&samples, frame_samples)
}

fn decode_into_rms(pcm: &[u8], frame_samples: usize) -> Vec<f32> {
    let expected_samples = pcm.len() / 2;
    let mut rms = Vec::with_capacity(expected_samples.div_ceil(frame_samples));
    let mut frame_len = 0usize;
    let mut frame_sum_sq = 0.0_f64;
    let mut leftover = None;

    {
        let mut append_sample = |sample: i16| {
            let normalized = (sample as f64) / 32768.0;
            frame_sum_sq += normalized * normalized;
            frame_len += 1;
            if frame_len == frame_samples {
                rms.push((frame_sum_sq / frame_len as f64).sqrt() as f32);
                frame_len = 0;
                frame_sum_sq = 0.0;
            }
        };

        for buffer in pcm.chunks(READ_BUFFER_BYTES) {
            let mut start = 0usize;
            if let Some(previous) = leftover.take() {
                append_sample(i16::from_le_bytes([previous, buffer[0]]));
                start = 1;
            }
            let (pairs, remainder) = buffer[start..].as_chunks::<2>();
            for pair in pairs {
                append_sample(i16::from_le_bytes(*pair));
            }
            leftover = remainder.first().copied();
        }
    }

    if frame_len > 0 {
        rms.push((frame_sum_sq / frame_len as f64).sqrt() as f32);
    }
    rms
}

fn assert_same_bits(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert_eq!(left.to_bits(), right.to_bits(), "frame {index} changed");
    }
}

fn timed(iterations: usize, pcm: &[u8], implementation: fn(&[u8], usize) -> Vec<f32>) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        let output = implementation(black_box(pcm), FRAME_SAMPLES);
        black_box(output.len());
        black_box(output.last().map(|value| value.to_bits()));
    }
    started.elapsed()
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn median_ns(values: &[Duration]) -> u128 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2].as_nanos()
}

fn option_value<T: std::str::FromStr>(name: &str, default: T) -> T {
    let mut arguments = std::env::args();
    while let Some(argument) = arguments.next() {
        if argument == name {
            return arguments
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default);
        }
    }
    default
}

fn main() {
    for sample_count in [0, 1, 319, 320, 321, 8191, 480_000] {
        let pcm = make_pcm(sample_count);
        assert_same_bits(
            &decode_then_rms(&pcm, FRAME_SAMPLES),
            &decode_into_rms(&pcm, FRAME_SAMPLES),
        );
    }

    let samples = option_value("--sample-size", 10usize).max(10);
    let measurement_seconds = option_value("--measurement-time", 1.0f64).max(0.1);
    let pcm = make_pcm(30 * 16_000);
    let target = Duration::from_secs_f64(measurement_seconds / (samples as f64 * 4.0));

    let mut iterations = 1usize;
    loop {
        let duration = timed(iterations, &pcm, decode_then_rms);
        if duration >= target / 2 {
            let ns_per_iteration = duration.as_nanos().max(1) / iterations as u128;
            iterations = ((target.as_nanos() / ns_per_iteration) as usize).max(1);
            break;
        }
        iterations *= 2;
    }

    for _ in 0..3 {
        black_box(timed(iterations, &pcm, decode_then_rms));
        black_box(timed(iterations, &pcm, decode_into_rms));
    }

    let mut null_ratios = Vec::with_capacity(samples);
    let mut speedups = Vec::with_capacity(samples);
    let mut legacy_times = Vec::with_capacity(samples);
    let mut fused_times = Vec::with_capacity(samples);
    for sample in 0..samples {
        let null_a = timed(iterations, &pcm, decode_then_rms);
        let null_b = timed(iterations, &pcm, decode_then_rms);
        let (null_numerator, null_denominator) = if sample % 2 == 0 {
            (null_a, null_b)
        } else {
            (null_b, null_a)
        };
        null_ratios.push(null_numerator.as_secs_f64() / null_denominator.as_secs_f64());

        let (legacy, fused) = if sample % 2 == 0 {
            (
                timed(iterations, &pcm, decode_then_rms),
                timed(iterations, &pcm, decode_into_rms),
            )
        } else {
            let fused = timed(iterations, &pcm, decode_into_rms);
            let legacy = timed(iterations, &pcm, decode_then_rms);
            (legacy, fused)
        };
        legacy_times.push(legacy);
        fused_times.push(fused);
        speedups.push(legacy.as_secs_f64() / fused.as_secs_f64());
    }

    println!(
        "native_audio_pcm_rms samples_16k={} frames={} pairs={} iterations={} null_p10={:.6} null_median={:.6} null_p90={:.6} legacy_median_ns={} fused_median_ns={} speedup_p10={:.6} speedup_median={:.6} speedup_p90={:.6} wins={}/{}",
        pcm.len() / 2,
        (pcm.len() / 2).div_ceil(FRAME_SAMPLES),
        samples,
        iterations,
        percentile(&null_ratios, 10),
        percentile(&null_ratios, 50),
        percentile(&null_ratios, 90),
        median_ns(&legacy_times),
        median_ns(&fused_times),
        percentile(&speedups, 10),
        percentile(&speedups, 50),
        percentile(&speedups, 90),
        speedups.iter().filter(|ratio| **ratio > 1.0).count(),
        samples,
    );
    println!("null_ratios={null_ratios:?}");
    println!("speedups={speedups:?}");
}
