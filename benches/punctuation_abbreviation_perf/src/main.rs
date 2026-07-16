use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::{Duration, Instant};

const ABBREVIATIONS: &[&str] = &[
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "ave.", "blvd.", "vs.", "etc.",
    "approx.", "dept.", "est.", "govt.", "inc.", "ltd.", "no.", "vol.", "rev.", "gen.", "sgt.",
    "cpl.", "pvt.", "lt.", "capt.", "col.", "maj.", "cmdr.", "adm.", "hon.", "fig.", "eq.", "ref.",
    "sec.",
];

#[derive(Clone, Copy)]
enum Shape {
    Short,
    Medium,
    Stress,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Self::Short => "256x256B",
            Self::Medium => "16x4KiB",
            Self::Stress => "1x64KiB",
        }
    }

    fn dimensions(self) -> (usize, usize) {
        match self {
            Self::Short => (256, 256),
            Self::Medium => (16, 4 * 1024),
            Self::Stress => (1, 64 * 1024),
        }
    }
}

struct Fixture {
    shape: Shape,
    segments: Vec<String>,
    periods: Vec<Vec<usize>>,
}

fn historical(text: &str, period_byte_pos: usize) -> bool {
    let before = &text[..period_byte_pos + 1];
    let lower = before.to_ascii_lowercase();
    for abbr in ABBREVIATIONS {
        if lower.ends_with(abbr) {
            let prefix_len = before.len() - abbr.len();
            if prefix_len == 0 {
                return true;
            }
            if before
                .as_bytes()
                .get(prefix_len.wrapping_sub(1))
                .is_some_and(|b| b.is_ascii_whitespace())
            {
                return true;
            }
        }
    }
    false
}

fn candidate(text: &str, period_byte_pos: usize) -> bool {
    let before = &text[..period_byte_pos + 1];
    let before_bytes = before.as_bytes();
    for abbr in ABBREVIATIONS {
        let abbr_bytes = abbr.as_bytes();
        if before_bytes.len() < abbr_bytes.len() {
            continue;
        }
        let prefix_len = before_bytes.len() - abbr_bytes.len();
        if before_bytes[prefix_len..].eq_ignore_ascii_case(abbr_bytes)
            && (prefix_len == 0 || before_bytes[prefix_len - 1].is_ascii_whitespace())
        {
            return true;
        }
    }
    false
}

fn make_segment(target_bytes: usize, seed: usize) -> String {
    const PHRASES: &[&str] = &[
        "Dr. alpha beta gamma delta. ",
        "we met Prof. epsilon near café. ",
        "Eq. nine is useful in naïve tests. ",
        "Mr. zeta said wait... then continued. ",
        "the value is 3.14 and this ends. ",
        "Ref. eight covers emoji 🦀 safely. ",
    ];
    let mut text = String::with_capacity(target_bytes);
    let mut index = seed % PHRASES.len();
    while text.len() + PHRASES[index].len() <= target_bytes {
        text.push_str(PHRASES[index]);
        index = (index + 1) % PHRASES.len();
    }
    while text.len() < target_bytes {
        text.push('x');
    }
    text
}

fn fixture(shape: Shape) -> Fixture {
    let (segment_count, bytes_per_segment) = shape.dimensions();
    let segments: Vec<String> = (0..segment_count)
        .map(|index| make_segment(bytes_per_segment, index))
        .collect();
    let periods = segments
        .iter()
        .map(|text| text.match_indices('.').map(|(offset, _)| offset).collect())
        .collect();
    Fixture {
        shape,
        segments,
        periods,
    }
}

fn run_helper(fixture: &Fixture, rounds: u64, helper: fn(&str, usize) -> bool) -> u64 {
    let mut hits = 0_u64;
    for _ in 0..rounds {
        for (text, periods) in fixture.segments.iter().zip(&fixture.periods) {
            for &period in periods {
                hits = hits.wrapping_add(black_box(helper(black_box(text), period)) as u64);
            }
        }
    }
    black_box(hits)
}

fn run_lowercase(fixture: &Fixture, rounds: u64) -> usize {
    let mut bytes = 0_usize;
    for _ in 0..rounds {
        for (text, periods) in fixture.segments.iter().zip(&fixture.periods) {
            for &period in periods {
                bytes =
                    bytes.wrapping_add(black_box(text[..period + 1].to_ascii_lowercase()).len());
            }
        }
    }
    black_box(bytes)
}

fn elapsed_helper(fixture: &Fixture, rounds: u64, helper: fn(&str, usize) -> bool) -> Duration {
    let start = Instant::now();
    black_box(run_helper(fixture, rounds, helper));
    start.elapsed()
}

fn elapsed_lowercase(fixture: &Fixture, rounds: u64) -> Duration {
    let start = Instant::now();
    black_box(run_lowercase(fixture, rounds));
    start.elapsed()
}

fn calibrated_rounds(fixture: &Fixture, target: Duration) -> u64 {
    let mut rounds = 1_u64;
    loop {
        let elapsed = elapsed_helper(fixture, rounds, historical);
        if elapsed >= Duration::from_millis(5) {
            let scaled = (rounds as u128 * target.as_nanos() / elapsed.as_nanos().max(1)) as u64;
            return scaled.max(rounds).max(1);
        }
        rounds = rounds.saturating_mul(2);
        if rounds >= 1_048_576 {
            return rounds;
        }
    }
}

fn is_decimal_period(text: &str, period: usize) -> bool {
    period > 0
        && text
            .as_bytes()
            .get(period - 1)
            .is_some_and(|b| b.is_ascii_digit())
        && text
            .as_bytes()
            .get(period + 1)
            .is_some_and(|b| b.is_ascii_digit())
}

fn is_ellipsis_period(text: &str, period: usize) -> bool {
    let bytes = text.as_bytes();
    let mut start = period;
    while start > 0 && bytes.get(start - 1) == Some(&b'.') {
        start -= 1;
    }
    let mut end = period;
    while bytes.get(end + 1) == Some(&b'.') {
        end += 1;
    }
    end - start + 1 >= 3
}

fn apply_rule_four(text: &str, helper: fn(&str, usize) -> bool) -> String {
    let mut result = String::with_capacity(text.len());
    let mut capitalize_next = false;
    let mut byte_offset = 0_usize;
    for ch in text.chars() {
        if capitalize_next && ch.is_alphabetic() {
            result.extend(ch.to_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
        if ch == '?' || ch == '!' {
            capitalize_next = true;
        } else if ch == '.' {
            if !helper(text, byte_offset)
                && !is_decimal_period(text, byte_offset)
                && !is_ellipsis_period(text, byte_offset)
            {
                capitalize_next = true;
            }
        } else if !ch.is_whitespace() {
            capitalize_next = false;
        }
        byte_offset += ch.len_utf8();
    }
    result
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn verify_parity() -> (usize, usize, u64) {
    let mut cases = Vec::new();
    let prefixes = [
        "", " ", "\t", "\n", "\u{000b}", "\u{000c}", "\r", "x", "\u{2003}", "🦀",
    ];
    for abbr in ABBREVIATIONS {
        let letter_count = abbr.bytes().filter(u8::is_ascii_alphabetic).count();
        for mask in 0..(1_usize << letter_count) {
            let mut mixed = String::with_capacity(abbr.len());
            let mut letter_index = 0_usize;
            for byte in abbr.bytes() {
                if byte.is_ascii_alphabetic() {
                    let byte = if mask & (1 << letter_index) == 0 {
                        byte.to_ascii_lowercase()
                    } else {
                        byte.to_ascii_uppercase()
                    };
                    mixed.push(char::from(byte));
                    letter_index += 1;
                } else {
                    mixed.push(char::from(byte));
                }
            }
            for prefix in prefixes {
                cases.push(format!("{prefix}{mixed}"));
            }
        }
        cases.push(format!(" x{abbr}"));
        cases.push(format!(" {abbr}x."));
    }
    cases.extend([
        "done.".to_owned(),
        "3.14".to_owned(),
        "wait...".to_owned(),
        "café Dr.".to_owned(),
        "🦀\tPROF.".to_owned(),
        "\u{2003}Dr.".to_owned(),
        "hello world.".to_owned(),
    ]);

    let mut checked = 0_usize;
    for text in &cases {
        for (period, _) in text.match_indices('.') {
            assert_eq!(
                historical(text, period),
                candidate(text, period),
                "predicate mismatch for {text:?} at byte {period}"
            );
            checked += 1;
        }
    }

    let mut output_bytes = Vec::new();
    for shape in [Shape::Short, Shape::Medium, Shape::Stress] {
        let fixture = fixture(shape);
        for text in &fixture.segments {
            let historical_output = apply_rule_four(text, historical);
            let candidate_output = apply_rule_four(text, candidate);
            assert_eq!(historical_output, candidate_output);
            output_bytes.extend_from_slice(candidate_output.as_bytes());
            output_bytes.push(0);
        }
    }
    (checked, output_bytes.len(), fnv1a(&output_bytes))
}

fn sorted(mut samples: Vec<f64>) -> Vec<f64> {
    samples.sort_by(f64::total_cmp);
    samples
}

fn median(samples: &[f64]) -> f64 {
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2.0
    } else {
        samples[middle]
    }
}

fn percentile(samples: &[f64], percentile: f64) -> f64 {
    let index = ((samples.len() - 1) as f64 * percentile).round() as usize;
    samples[index]
}

fn cv(samples: &[f64]) -> f64 {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt() / mean
}

fn ratios(fixture: &Fixture, rounds: u64, candidate_arm: bool) -> Vec<f64> {
    let mut ratios = Vec::with_capacity(21);
    let other = if candidate_arm { candidate } else { historical };
    for _ in 0..3 {
        black_box(run_helper(fixture, rounds, historical));
        black_box(run_helper(fixture, rounds, other));
    }
    for sample in 0..21 {
        let elapsed_ns = |helper| elapsed_helper(fixture, rounds, helper).as_nanos() as f64;
        let (historical_ns, other_ns) = if sample % 2 == 0 {
            let historical_first = elapsed_ns(historical);
            let other_first = elapsed_ns(other);
            let other_second = elapsed_ns(other);
            let historical_second = elapsed_ns(historical);
            (
                historical_first + historical_second,
                other_first + other_second,
            )
        } else {
            let other_first = elapsed_ns(other);
            let historical_first = elapsed_ns(historical);
            let historical_second = elapsed_ns(historical);
            let other_second = elapsed_ns(other);
            (
                historical_first + historical_second,
                other_first + other_second,
            )
        };
        ratios.push(other_ns / historical_ns);
    }
    sorted(ratios)
}

fn print_samples(name: &str, samples: &[f64]) {
    let body = samples
        .iter()
        .map(|sample| format!("{sample:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{name}=[{body}] median={:.6} p10={:.6} p90={:.6} cv={:.4}%",
        median(samples),
        percentile(samples, 0.10),
        percentile(samples, 0.90),
        cv(samples) * 100.0
    );
}

fn profile() -> ExitCode {
    let (cases, output_bytes, output_hash) = verify_parity();
    let fixture = fixture(Shape::Medium);
    let rounds = calibrated_rounds(&fixture, Duration::from_millis(100));
    let historical_ns = elapsed_helper(&fixture, rounds, historical).as_nanos() as f64;
    let lowercase_ns = elapsed_lowercase(&fixture, rounds).as_nanos() as f64;
    let candidate_ns = elapsed_helper(&fixture, rounds, candidate).as_nanos() as f64;
    let period_count: usize = fixture.periods.iter().map(Vec::len).sum();
    println!(
        "profile shape={} segments={} periods_per_round={} rounds={rounds}",
        fixture.shape.name(),
        fixture.segments.len(),
        period_count
    );
    println!(
        "parity predicate_cases={cases} full_output_bytes={output_bytes} full_output_fnv64={output_hash:016x}"
    );
    println!("historical_ns={historical_ns:.0}");
    println!(
        "prefix_lowercase_ns={lowercase_ns:.0} share_of_historical={:.2}%",
        lowercase_ns / historical_ns * 100.0
    );
    println!(
        "allocation_free_design_ns={candidate_ns:.0} ratio={:.6}",
        candidate_ns / historical_ns
    );
    if lowercase_ns / historical_ns < 0.30 {
        eprintln!("profile gate failed: prefix lowercase is below 30% of historical helper time");
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}

fn measure() -> ExitCode {
    let (cases, output_bytes, output_hash) = verify_parity();
    println!(
        "parity predicate_cases={cases} full_output_bytes={output_bytes} full_output_fnv64={output_hash:016x}"
    );
    let mut passed = true;
    for shape in [Shape::Short, Shape::Medium, Shape::Stress] {
        let fixture = fixture(shape);
        let rounds = calibrated_rounds(&fixture, Duration::from_millis(160));
        let null = ratios(&fixture, rounds, false);
        let ab = ratios(&fixture, rounds, true);
        let period_count: usize = fixture.periods.iter().map(Vec::len).sum();
        println!(
            "measure shape={} segments={} periods_per_round={} rounds={rounds}",
            shape.name(),
            fixture.segments.len(),
            period_count
        );
        print_samples("historical_over_historical", &null);
        print_samples("candidate_over_historical", &ab);

        let null_ok = (0.97..=1.03).contains(&median(&null)) && cv(&null) <= 0.03;
        let separation_ok = percentile(&ab, 0.90) < percentile(&null, 0.10);
        let shape_ok = match shape {
            Shape::Short => median(&ab) <= 1.03,
            Shape::Medium => median(&ab) <= 0.80 && ab.iter().all(|ratio| *ratio < 1.0),
            Shape::Stress => true,
        };
        println!(
            "gates shape={} null_ok={null_ok} separation_ok={separation_ok} shape_ok={shape_ok}",
            shape.name()
        );
        passed &= null_ok && separation_ok && shape_ok;
    }
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("profile") => profile(),
        Some("measure") => measure(),
        _ => {
            eprintln!("usage: punctuation_abbreviation_perf <profile|measure>");
            ExitCode::from(64)
        }
    }
}
