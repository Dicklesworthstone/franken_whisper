use std::hint::black_box;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

const PAIRS: usize = 21;
const TARGET_ARM: Duration = Duration::from_millis(25);

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    frame_bytes: usize,
    frame_count: usize,
    primary: bool,
}

const SHAPES: [Shape; 2] = [
    Shape {
        name: "20ms_160b",
        frame_bytes: 160,
        frame_count: 4_096,
        primary: true,
    },
    Shape {
        name: "200ms_1600b",
        frame_bytes: 1_600,
        frame_count: 1_024,
        primary: false,
    },
];

type VerifyFn = fn(&[Vec<u8>], &[String]) -> usize;

fn fixture(shape: Shape) -> (Vec<Vec<u8>>, Vec<String>) {
    let frames: Vec<Vec<u8>> = (0..shape.frame_count)
        .map(|frame| {
            (0..shape.frame_bytes)
                .map(|offset| {
                    (frame as u8)
                        .wrapping_mul(17)
                        .wrapping_add((offset as u8).wrapping_mul(29))
                })
                .collect()
        })
        .collect();
    let expected = frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let lower = format!("{:x}", Sha256::digest(frame));
            if index.is_multiple_of(2) {
                lower
            } else {
                lower.to_ascii_uppercase()
            }
        })
        .collect();
    (frames, expected)
}

fn historical_result(data: &[u8], expected: &str) -> Option<String> {
    let actual = format!("{:x}", Sha256::digest(data));
    if actual.eq_ignore_ascii_case(expected) {
        None
    } else {
        Some(actual)
    }
}

fn candidate_result(data: &[u8], expected: &str) -> Option<String> {
    let digest = Sha256::digest(data);
    if digest_matches_ascii_hex(&digest, expected.as_bytes()) {
        None
    } else {
        Some(format!("{digest:x}"))
    }
}

fn digest_matches_ascii_hex(digest: &[u8], expected: &[u8]) -> bool {
    expected.len() == digest.len() * 2
        && digest
            .iter()
            .zip(expected.chunks_exact(2))
            .all(|(&byte, pair)| {
                hex_nibble(byte >> 4).eq_ignore_ascii_case(&pair[0])
                    && hex_nibble(byte & 0x0f).eq_ignore_ascii_case(&pair[1])
            })
}

fn hex_nibble(nibble: u8) -> u8 {
    match nibble {
        0..=9 => b'0' + nibble,
        _ => b'a' + (nibble - 10),
    }
}

fn historical_verify_batch(frames: &[Vec<u8>], expected: &[String]) -> usize {
    frames
        .iter()
        .zip(expected)
        .filter(|(frame, expected)| {
            historical_result(black_box(frame.as_slice()), black_box(expected)).is_none()
        })
        .count()
}

fn candidate_verify_batch(frames: &[Vec<u8>], expected: &[String]) -> usize {
    frames
        .iter()
        .zip(expected)
        .filter(|(frame, expected)| {
            candidate_result(black_box(frame.as_slice()), black_box(expected)).is_none()
        })
        .count()
}

fn assert_exact_parity(frames: &[Vec<u8>], expected: &[String]) {
    assert_eq!(historical_verify_batch(frames, expected), frames.len());
    assert_eq!(candidate_verify_batch(frames, expected), frames.len());

    for &frame_index in &[0, frames.len() / 2, frames.len() - 1] {
        let data = &frames[frame_index];
        let lower = format!("{:x}", Sha256::digest(data));
        let mut cases = vec![
            lower.clone(),
            lower.to_ascii_uppercase(),
            String::new(),
            lower[..63].to_owned(),
            format!("{lower}0"),
            "g".repeat(64),
            "é".repeat(32),
        ];
        for index in 0..lower.len() {
            let mut mismatched = lower.clone().into_bytes();
            mismatched[index] = match mismatched[index] {
                b'0' => b'1',
                _ => b'0',
            };
            cases.push(String::from_utf8(mismatched).expect("ASCII digest"));
        }

        for expected in cases {
            assert_eq!(
                candidate_result(data, &expected),
                historical_result(data, &expected),
                "frame={frame_index}, expected={expected:?}"
            );
        }
    }
}

fn timed<T>(operation: impl FnOnce() -> T) -> (Duration, T) {
    let started = Instant::now();
    let output = operation();
    (started.elapsed(), output)
}

fn measure_arm(
    verify: VerifyFn,
    frames: &[Vec<u8>],
    expected: &[String],
    iterations: usize,
) -> Duration {
    let (elapsed, checksum) = timed(|| {
        let mut checksum = 0usize;
        for _ in 0..iterations {
            checksum = checksum.wrapping_add(verify(black_box(frames), black_box(expected)));
        }
        checksum
    });
    assert_eq!(checksum, frames.len() * iterations);
    black_box(checksum);
    elapsed
}

fn calibrate(frames: &[Vec<u8>], expected: &[String]) -> usize {
    let elapsed = measure_arm(historical_verify_batch, frames, expected, 1);
    let elapsed_ns = elapsed.as_nanos().max(1);
    let target_ns = TARGET_ARM.as_nanos();
    usize::try_from(target_ns.div_ceil(elapsed_ns))
        .unwrap_or(usize::MAX)
        .max(1)
}

fn paired_ratios(
    baseline: VerifyFn,
    contender: VerifyFn,
    frames: &[Vec<u8>],
    expected: &[String],
    iterations: usize,
) -> Vec<f64> {
    let mut ratios = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let (baseline_elapsed, contender_elapsed) = if pair.is_multiple_of(2) {
            (
                measure_arm(baseline, frames, expected, iterations),
                measure_arm(contender, frames, expected, iterations),
            )
        } else {
            let contender_elapsed = measure_arm(contender, frames, expected, iterations);
            let baseline_elapsed = measure_arm(baseline, frames, expected, iterations);
            (baseline_elapsed, contender_elapsed)
        };
        ratios.push(baseline_elapsed.as_secs_f64() / contender_elapsed.as_secs_f64());
    }
    ratios
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn fnv64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn main() {
    let executable = std::fs::read(std::env::current_exe().expect("benchmark path"))
        .expect("read benchmark binary");
    println!(
        "TTY_SHA_AB_BINARY_FNV64={:016x}",
        fnv64_update(0xcbf_29ce_4842_2325, &executable)
    );

    for shape in SHAPES {
        let (frames, expected) = fixture(shape);
        assert_exact_parity(&frames, &expected);
        let fixture_hash = frames.iter().fold(0xcbf_29ce_4842_2325, |hash, frame| {
            fnv64_update(hash, frame)
        });
        println!(
            "TTY_SHA_AB_FIXTURE shape={} bytes={} frames={} fnv64={fixture_hash:016x}",
            shape.name, shape.frame_bytes, shape.frame_count
        );

        for _ in 0..3 {
            black_box(historical_verify_batch(
                black_box(&frames),
                black_box(&expected),
            ));
            black_box(candidate_verify_batch(
                black_box(&frames),
                black_box(&expected),
            ));
        }

        let iterations = calibrate(&frames, &expected);
        let null = paired_ratios(
            historical_verify_batch,
            historical_verify_batch,
            &frames,
            &expected,
            iterations,
        );
        let candidate = paired_ratios(
            historical_verify_batch,
            candidate_verify_batch,
            &frames,
            &expected,
            iterations,
        );

        let null_median = percentile(&null, 50);
        let null_p90 = percentile(&null, 90);
        let candidate_p10 = percentile(&candidate, 10);
        let candidate_median = percentile(&candidate, 50);
        let candidate_p90 = percentile(&candidate, 90);
        let wins = candidate.iter().filter(|&&ratio| ratio > 1.0).count();

        println!(
            "TTY_SHA_AB_NULL shape={} iterations={iterations} ratios={null:?}",
            shape.name
        );
        println!(
            "TTY_SHA_AB_CANDIDATE shape={} ratios={candidate:?}",
            shape.name
        );
        println!(
            "TTY_SHA_AB_STATS shape={} null_median={null_median:.6} null_p90={null_p90:.6} candidate_p10={candidate_p10:.6} candidate_median={candidate_median:.6} candidate_p90={candidate_p90:.6} wins={wins}/{PAIRS}",
            shape.name
        );

        assert!(
            (0.97..=1.03).contains(&null_median),
            "{} null median {null_median:.6} outside [0.97, 1.03]",
            shape.name
        );
        if shape.primary {
            assert!(
                candidate_median >= 1.10,
                "{} candidate median {candidate_median:.6} below 1.10",
                shape.name
            );
            assert!(
                candidate_p10 > null_p90,
                "{} candidate p10 {candidate_p10:.6} did not clear null p90 {null_p90:.6}",
                shape.name
            );
            assert!(
                wins >= 18,
                "{} candidate won only {wins}/{PAIRS} pairs",
                shape.name
            );
        } else {
            assert!(
                candidate_median >= 0.98,
                "{} candidate median {candidate_median:.6} regressed",
                shape.name
            );
            assert!(
                candidate_p10 >= 0.95,
                "{} candidate p10 {candidate_p10:.6} below no-regression floor",
                shape.name
            );
        }
    }

    println!("TTY_SHA_AB_GATE=PASS");
}
