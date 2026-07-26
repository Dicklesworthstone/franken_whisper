use std::hint::black_box;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

const PAIRS: usize = 41;
const MIN_OF: usize = 3;
const TARGET_NS: u128 = 2_000_000;
const BOOTSTRAP_RESAMPLES: usize = 20_000;

fn self_identity() -> String {
    let Ok(path) = std::env::current_exe() else {
        return "unavailable".to_owned();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return "unavailable".to_owned();
    };
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!(
        "{:x} ({} bytes) {}",
        hasher.finalize(),
        bytes.len(),
        path.display()
    )
}

#[derive(Debug, PartialEq)]
struct Segment {
    start_sec: Option<f64>,
    end_sec: Option<f64>,
    text: String,
    speaker: Option<String>,
}

fn parse_historical(content: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut block_lines: Vec<String> = Vec::new();

    for raw_line in content.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            if !block_lines.is_empty() {
                let block = block_lines.join("\n");
                if let Some(segment) = parse_block(block.trim()) {
                    segments.push(segment);
                }
                block_lines.clear();
            }
            continue;
        }
        block_lines.push(line.to_owned());
    }

    if !block_lines.is_empty() {
        let block = block_lines.join("\n");
        if let Some(segment) = parse_block(block.trim()) {
            segments.push(segment);
        }
    }

    segments
}

fn parse_borrowed(content: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut block_lines = Vec::new();

    for raw_line in content.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            if !block_lines.is_empty() {
                if let Some(segment) = parse_lines(&block_lines) {
                    segments.push(segment);
                }
                block_lines.clear();
            }
            continue;
        }
        block_lines.push(line);
    }

    if !block_lines.is_empty() {
        if let Some(segment) = parse_lines(&block_lines) {
            segments.push(segment);
        }
    }

    segments
}

fn parse_block(block: &str) -> Option<Segment> {
    let lines: Vec<&str> = block.lines().collect();
    parse_lines(&lines)
}

fn parse_lines(lines: &[&str]) -> Option<Segment> {
    if lines.is_empty() {
        return None;
    }

    let (start, end, text) =
        if let Some(timing_idx) = lines.iter().position(|line| line.contains("-->")) {
            if timing_idx + 1 >= lines.len() {
                return None;
            }

            let mut timing = lines[timing_idx].split("-->").map(str::trim);
            let start = timing.next().and_then(parse_time);
            let end = timing.next().and_then(parse_time);
            let text = lines[(timing_idx + 1)..].join(" ");
            (start, end, text)
        } else {
            if lines.len() < 3 {
                return None;
            }
            let index_line = lines[0].trim();
            if index_line.is_empty() || !index_line.chars().all(|ch| ch.is_ascii_digit()) {
                return None;
            }
            let text = lines[2..].join(" ");
            (None, None, text)
        };

    let text = text.trim().to_owned();
    if text.is_empty() {
        return None;
    }
    let (speaker, text) = extract_speaker_prefix(&text);
    Some(Segment {
        start_sec: start,
        end_sec: end,
        text,
        speaker,
    })
}

fn parse_time(value: &str) -> Option<f64> {
    let (hms, milliseconds) = if let Some(position) = value.rfind(',') {
        (&value[..position], &value[(position + 1)..])
    } else {
        let position = value.rfind('.')?;
        (&value[..position], &value[(position + 1)..])
    };
    let milliseconds = milliseconds.parse::<f64>().ok()?;
    let mut parts = hms.split(':');
    let hours = parts.next()?.parse::<f64>().ok()?;
    let minutes = parts.next()?.parse::<f64>().ok()?;
    let seconds = parts.next()?.parse::<f64>().ok()?;
    Some(hours * 3_600.0 + minutes * 60.0 + seconds + milliseconds / 1_000.0)
}

fn extract_speaker_prefix(text: &str) -> (Option<String>, String) {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix('[')
        && let Some((head, tail)) = rest.split_once(']')
    {
        let speaker = head.trim();
        let clean_tail = tail.trim_start_matches([':', '-', '|', ' ']).trim();
        if is_speaker_label(speaker) && !clean_tail.is_empty() {
            return (Some(speaker.to_owned()), clean_tail.to_owned());
        }
    }
    for separator in [":", "-", "|"] {
        let mut parts = trimmed.splitn(2, separator);
        let head = parts.next().unwrap_or_default().trim();
        let tail = parts.next().map(str::trim).unwrap_or_default();
        if is_speaker_label(head) && !tail.is_empty() {
            return (Some(head.to_owned()), tail.to_owned());
        }
    }
    (None, trimmed.to_owned())
}

fn is_speaker_label(label: &str) -> bool {
    let lowered = label.trim().to_ascii_lowercase();
    lowered.starts_with("speaker")
        || lowered.starts_with("spk")
        || lowered.starts_with("spkr")
        || (lowered.len() >= 2
            && lowered.starts_with('s')
            && lowered[1..]
                .chars()
                .all(|character| character.is_ascii_digit()))
}

fn corpus(block_count: usize) -> String {
    let mut content = String::with_capacity(block_count * 180);
    for index in 0..block_count {
        let start_seconds = index % 3_500;
        let end_seconds = start_seconds + 3;
        let separator = if index + 1 == block_count { "" } else { "\n\n" };
        let speaker = index % 16;
        content.push_str(&format!(
            "{}\n00:{:02}:{:02},{:03} --> 00:{:02}:{:02},{:03}\n[SPEAKER_{speaker:02}] Representative diarization text for segment {index}\nwith a second line to preserve realistic subtitle shape{separator}",
            index + 1,
            start_seconds / 60,
            start_seconds % 60,
            index % 1_000,
            end_seconds / 60,
            end_seconds % 60,
            (index * 7) % 1_000,
        ));
    }
    content
}

fn signature(segments: &[Segment]) -> u64 {
    segments.iter().fold(0_u64, |signature, segment| {
        signature
            .wrapping_mul(31)
            .wrapping_add(segment.start_sec.map(f64::to_bits).unwrap_or_default())
            .wrapping_add(segment.end_sec.map(f64::to_bits).unwrap_or_default())
            .wrapping_add(segment.text.len() as u64)
            .wrapping_add(segment.speaker.as_ref().map_or(0, |value| value.len()) as u64)
    })
}

#[derive(Clone, Copy)]
enum Arm {
    Historical,
    Borrowed,
}

fn timed(content: &str, arm: Arm, rounds: usize) -> (Duration, u64) {
    let started = Instant::now();
    let mut result = 0;
    for _ in 0..rounds {
        let segments = match arm {
            Arm::Historical => parse_historical(black_box(content)),
            Arm::Borrowed => parse_borrowed(black_box(content)),
        };
        result ^= signature(&segments);
    }
    (started.elapsed(), black_box(result))
}

fn calibrated_rounds(content: &str, arm: Arm) -> usize {
    let probe_rounds = 1;
    let probe_ns = timed(content, arm, probe_rounds).0.as_nanos().max(1);
    usize::try_from((TARGET_NS * probe_rounds as u128 / probe_ns).clamp(1, 1_000_000))
        .expect("bounded rounds")
}

fn min_ns_per_round(content: &str, arm: Arm, rounds: usize) -> f64 {
    (0..MIN_OF)
        .map(|_| timed(content, arm, rounds).0.as_nanos() as f64)
        .fold(f64::INFINITY, f64::min)
        / rounds as f64
}

fn paired_ratio(
    content: &str,
    second: Arm,
    historical_rounds: usize,
    second_rounds: usize,
    reverse: bool,
) -> f64 {
    let (historical_ns, second_ns) = if reverse {
        let second_ns = min_ns_per_round(content, second, second_rounds);
        let historical_ns = min_ns_per_round(content, Arm::Historical, historical_rounds);
        (historical_ns, second_ns)
    } else {
        (
            min_ns_per_round(content, Arm::Historical, historical_rounds),
            min_ns_per_round(content, second, second_rounds),
        )
    };
    historical_ns / second_ns.max(f64::MIN_POSITIVE)
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    let mut state = 0xa54f_f53a_5f1d_36f1_u64 ^ values.len() as u64;
    let mut sample = Vec::with_capacity(values.len());
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        sample.clear();
        for _ in 0..values.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sample.push(values[state as usize % values.len()]);
        }
        sample.sort_by(f64::total_cmp);
        medians.push(sample[sample.len() / 2]);
    }
    medians.sort_by(f64::total_cmp);
    (
        medians[BOOTSTRAP_RESAMPLES * 25 / 1_000],
        medians[BOOTSTRAP_RESAMPLES * 975 / 1_000],
    )
}

fn assert_exact(left: &[Segment], right: &[Segment]) {
    assert_eq!(left.len(), right.len(), "segment count mismatch");
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            left.start_sec.map(f64::to_bits),
            right.start_sec.map(f64::to_bits),
            "start mismatch at segment {index}"
        );
        assert_eq!(
            left.end_sec.map(f64::to_bits),
            right.end_sec.map(f64::to_bits),
            "end mismatch at segment {index}"
        );
        assert_eq!(left.text, right.text, "text mismatch at segment {index}");
        assert_eq!(
            left.speaker, right.speaker,
            "speaker mismatch at segment {index}"
        );
    }
}

fn parity_oracle() {
    for content in [
        "",
        "1\n00:00:01,000 --> 00:00:02,500\nhello",
        "1\r\n00:00:01.000 --> 00:00:02.500\r\n[SPEAKER_00]: hello\r\n\r\n   \r\n2\r\n00:00:03,000 --> 00:00:04,000\r\nworld\r\n",
        "00:00:01,000 --> 00:00:02,000\nindexless\nsecond line",
        "  7  \nnot a timestamp\n  text from the fallback path  ",
        "8\n00:bad --> 00:worse\nSPK-2 | malformed times still preserve text",
        "9\n00:00:01,000 --> 00:00:02,000\n[spkr_ø] unicode ☃ stays exact\nline with \\\"quotes\\\" and \\\\ slashes",
        "10\n00:00:01,000 --> 00:00:02,000\n   \n\n11\n00:00:03,000 --> 00:00:04,000\nS2 - trimmed speaker",
        "not-an-index\nline two\nline three",
        "12\n00:00:01,000 --> 00:00:02,000",
        "\n\t\n\r\n",
    ] {
        assert_exact(&parse_historical(content), &parse_borrowed(content));
    }
    let realistic = corpus(8_192);
    assert_exact(&parse_historical(&realistic), &parse_borrowed(&realistic));
}

fn format_ratios(ratios: &[f64]) -> String {
    ratios
        .iter()
        .map(|ratio| format!("{ratio:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    const BLOCKS: usize = 8_192;

    println!("bench_elf_sha256={}", self_identity());
    let content = corpus(BLOCKS);
    parity_oracle();
    let historical_rounds = calibrated_rounds(&content, Arm::Historical);
    let borrowed_rounds = calibrated_rounds(&content, Arm::Borrowed);

    for reverse in [false, true, false] {
        black_box(paired_ratio(
            &content,
            Arm::Historical,
            historical_rounds,
            historical_rounds,
            reverse,
        ));
        black_box(paired_ratio(
            &content,
            Arm::Borrowed,
            historical_rounds,
            borrowed_rounds,
            reverse,
        ));
    }

    let mut null_ratios = Vec::with_capacity(PAIRS);
    let mut candidate_ratios = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        null_ratios.push(paired_ratio(
            &content,
            Arm::Historical,
            historical_rounds,
            historical_rounds,
            pair % 2 != 0,
        ));
    }
    for pair in 0..PAIRS {
        candidate_ratios.push(paired_ratio(
            &content,
            Arm::Borrowed,
            historical_rounds,
            borrowed_rounds,
            pair % 2 != 0,
        ));
    }

    let (null_ci_low, null_ci_high) = bootstrap_median_ci(&null_ratios);
    let (candidate_ci_low, candidate_ci_high) = bootstrap_median_ci(&candidate_ratios);
    let null_half_width = (1.0 - null_ci_low).abs().max((null_ci_high - 1.0).abs());
    let required_speedup = 1.0 + 2.0 * null_half_width;
    let candidate_median = percentile(&candidate_ratios, 50);
    let candidate_wins = candidate_ratios
        .iter()
        .filter(|ratio| **ratio > 1.0)
        .count();
    let keep = candidate_median >= required_speedup;

    println!(
        "bench blocks={BLOCKS} historical_rounds={historical_rounds} borrowed_rounds={borrowed_rounds} pairs={PAIRS} min_of={MIN_OF}"
    );
    println!("null_ratios={}", format_ratios(&null_ratios));
    println!("candidate_ratios={}", format_ratios(&candidate_ratios));
    println!(
        "null_median={:.6} null_median_ci95=[{null_ci_low:.6},{null_ci_high:.6}] null_cv={:.6}",
        percentile(&null_ratios, 50),
        coefficient_of_variation(&null_ratios)
    );
    println!(
        "candidate_median={candidate_median:.6} candidate_median_ci95=[{candidate_ci_low:.6},{candidate_ci_high:.6}] candidate_cv={:.6} wins={candidate_wins}/{PAIRS}",
        coefficient_of_variation(&candidate_ratios)
    );
    println!(
        "gate=median_vs_null_ci95_2x_margin null_half_width={null_half_width:.6} required_speedup={required_speedup:.6} candidate_median={candidate_median:.6} cv_is_provenance_only=true verdict={}",
        if keep { "KEEP" } else { "REJECT" }
    );
}
