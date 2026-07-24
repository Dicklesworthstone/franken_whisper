use std::hint::black_box;
use std::time::{Duration, Instant};

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

fn paired_ratio(content: &str, second: Arm, rounds: usize, reverse: bool) -> f64 {
    let results = if reverse {
        [
            timed(content, second, rounds),
            timed(content, Arm::Historical, rounds),
            timed(content, Arm::Historical, rounds),
            timed(content, second, rounds),
        ]
    } else {
        [
            timed(content, Arm::Historical, rounds),
            timed(content, second, rounds),
            timed(content, second, rounds),
            timed(content, Arm::Historical, rounds),
        ]
    };
    let (historical_before, second_before, second_after, historical_after) = if reverse {
        (results[1], results[0], results[3], results[2])
    } else {
        (results[0], results[1], results[2], results[3])
    };
    for result in &results[1..] {
        assert_eq!(results[0].1, result.1, "timed signatures diverged");
    }
    let historical_ns =
        historical_before.0.as_nanos() as f64 + historical_after.0.as_nanos() as f64;
    let second_ns = second_before.0.as_nanos() as f64 + second_after.0.as_nanos() as f64;
    historical_ns / second_ns
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
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
    const ROUNDS: usize = 4;
    const PAIRS: usize = 21;

    let content = corpus(BLOCKS);
    parity_oracle();

    for reverse in [false, true, false] {
        black_box(paired_ratio(&content, Arm::Historical, 1, reverse));
        black_box(paired_ratio(&content, Arm::Borrowed, 1, reverse));
    }

    let mut null_ratios = Vec::with_capacity(PAIRS);
    let mut candidate_ratios = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        null_ratios.push(paired_ratio(
            &content,
            Arm::Historical,
            ROUNDS,
            pair % 2 != 0,
        ));
    }
    for pair in 0..PAIRS {
        candidate_ratios.push(paired_ratio(&content, Arm::Borrowed, ROUNDS, pair % 2 != 0));
    }

    let null_median = percentile(&null_ratios, 50);
    let null_p90 = percentile(&null_ratios, 90);
    let candidate_p10 = percentile(&candidate_ratios, 10);
    let candidate_median = percentile(&candidate_ratios, 50);
    let candidate_wins = candidate_ratios
        .iter()
        .filter(|ratio| **ratio > 1.0)
        .count();
    let keep = (0.97..=1.03).contains(&null_median)
        && candidate_median >= 1.10
        && candidate_p10 > null_p90.max(1.05)
        && candidate_wins >= 18;

    println!("bench blocks={BLOCKS} rounds={ROUNDS} pairs={PAIRS}");
    println!("null_ratios={}", format_ratios(&null_ratios));
    println!("candidate_ratios={}", format_ratios(&candidate_ratios));
    println!("null_median={null_median:.6} null_p90={null_p90:.6}");
    println!(
        "candidate_p10={candidate_p10:.6} candidate_median={candidate_median:.6} wins={candidate_wins}/{PAIRS}"
    );
    println!("decision={}", if keep { "KEEP" } else { "REJECT" });
}
