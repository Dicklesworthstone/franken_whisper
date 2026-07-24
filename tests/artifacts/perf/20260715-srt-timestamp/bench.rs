use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct InlineTimestamp(u64);

impl std::fmt::Display for InlineTimestamp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_ms = self.0;
        let h = total_ms / 3_600_000;
        let m = (total_ms % 3_600_000) / 60_000;
        let s = (total_ms % 60_000) / 1000;
        let ms = total_ms % 1000;
        write!(formatter, "{h:02}:{m:02}:{s:02},{ms:03}")
    }
}

fn allocating_timestamp(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn write_allocating(segments: &[(f64, f64, String)], output: &mut String) {
    output.clear();
    for (index, (start, end, text)) in segments.iter().enumerate() {
        writeln!(output, "{}", index + 1).unwrap();
        writeln!(
            output,
            "{} --> {}",
            allocating_timestamp(*start),
            allocating_timestamp(*end)
        )
        .unwrap();
        writeln!(output, "{text}\n").unwrap();
    }
}

fn write_inline(segments: &[(f64, f64, String)], output: &mut String) {
    output.clear();
    for (index, (start, end, text)) in segments.iter().enumerate() {
        writeln!(output, "{}", index + 1).unwrap();
        writeln!(
            output,
            "{} --> {}",
            InlineTimestamp((*start * 1000.0).round() as u64),
            InlineTimestamp((*end * 1000.0).round() as u64)
        )
        .unwrap();
        writeln!(output, "{text}\n").unwrap();
    }
}

fn elapsed(
    iterations: usize,
    segments: &[(f64, f64, String)],
    output: &mut String,
    implementation: fn(&[(f64, f64, String)], &mut String),
) -> Duration {
    let started = Instant::now();
    for _ in 0..iterations {
        implementation(black_box(segments), black_box(output));
        black_box(output.as_bytes());
    }
    started.elapsed()
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn median_duration(values: &[Duration]) -> Duration {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn option_value<T: std::str::FromStr>(name: &str, default: T) -> T {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next().and_then(|value| value.parse().ok()).unwrap_or(default);
        }
    }
    default
}

fn main() {
    let samples = option_value("--sample-size", 10usize).max(10);
    let measurement_seconds = option_value("--measurement-time", 1.0f64).max(0.1);
    let segments: Vec<(f64, f64, String)> = (0..256)
        .map(|index| {
            let start = index as f64 * 1.25;
            (
                start,
                start + 1.0,
                format!("segment {index:03}: transcript text with Unicode λ and 🎧"),
            )
        })
        .collect();

    let mut expected = String::new();
    let mut actual = String::new();
    write_allocating(&segments, &mut expected);
    write_inline(&segments, &mut actual);
    assert_eq!(actual, expected, "SRT bytes changed");

    let target = Duration::from_secs_f64(measurement_seconds / (samples as f64 * 4.0));
    let mut iterations = 1usize;
    loop {
        let duration = elapsed(iterations, &segments, &mut expected, write_allocating);
        if duration >= target / 2 || iterations >= 1 << 20 {
            let nanos_per_iteration = duration.as_nanos().max(1) / iterations as u128;
            iterations = ((target.as_nanos() / nanos_per_iteration) as usize).max(1);
            break;
        }
        iterations *= 2;
    }

    for _ in 0..3 {
        black_box(elapsed(iterations, &segments, &mut expected, write_allocating));
        black_box(elapsed(iterations, &segments, &mut actual, write_inline));
    }

    let mut null_ratios = Vec::with_capacity(samples);
    let mut speedups = Vec::with_capacity(samples);
    let mut allocating_times = Vec::with_capacity(samples);
    let mut inline_times = Vec::with_capacity(samples);
    for sample in 0..samples {
        let null_a = elapsed(iterations, &segments, &mut expected, write_allocating);
        let null_b = elapsed(iterations, &segments, &mut actual, write_allocating);
        let (numerator, denominator) = if sample % 2 == 0 {
            (null_a, null_b)
        } else {
            (null_b, null_a)
        };
        null_ratios.push(numerator.as_secs_f64() / denominator.as_secs_f64());

        let (allocating, inline) = if sample % 2 == 0 {
            (
                elapsed(iterations, &segments, &mut expected, write_allocating),
                elapsed(iterations, &segments, &mut actual, write_inline),
            )
        } else {
            let inline = elapsed(iterations, &segments, &mut actual, write_inline);
            let allocating = elapsed(iterations, &segments, &mut expected, write_allocating);
            (allocating, inline)
        };
        allocating_times.push(allocating);
        inline_times.push(inline);
        speedups.push(allocating.as_secs_f64() / inline.as_secs_f64());
    }

    println!(
        "srt_timestamp segments={} bytes={} samples={} iterations={} null_p10={:.6} null_median={:.6} null_p90={:.6} allocating_median_ns={} inline_median_ns={} speedup_p10={:.6} speedup_median={:.6} speedup_p90={:.6} wins={}/{}",
        segments.len(),
        expected.len(),
        samples,
        iterations,
        percentile(&null_ratios, 10),
        percentile(&null_ratios, 50),
        percentile(&null_ratios, 90),
        median_duration(&allocating_times).as_nanos(),
        median_duration(&inline_times).as_nanos(),
        percentile(&speedups, 10),
        percentile(&speedups, 50),
        percentile(&speedups, 90),
        speedups.iter().filter(|ratio| **ratio > 1.0).count(),
        samples,
    );
    println!("null_ratios={null_ratios:?}");
    println!("speedups={speedups:?}");
}
