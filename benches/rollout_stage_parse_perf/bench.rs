use std::hint::black_box;
use std::time::Instant;

const PROFILE_SAMPLES: usize = 11;
const PROFILE_TARGET_NS: u64 = 30_000_000;
const AB_PAIRS: usize = 21;
const AB_TARGET_NS: u64 = 40_000_000;

const CORPUS: &[&str] = &[
    "shadow",
    "validated",
    "fallback",
    "primary",
    "sole",
    "0",
    "1",
    "2",
    "3",
    "4",
    " SHADOW ",
    "\tValidated\n",
    " FALLBACK",
    "Primary ",
    "\u{2003}SoLe\u{2003}",
    "",
    "   ",
    "canary",
    "99",
    "shadowed",
    "primary-bridge",
    "\u{212a}hadow",
    "\u{017f}ole",
    "\u{0130}PRIMARY",
    "\u{ff33}\u{ff28}\u{ff21}\u{ff24}\u{ff2f}\u{ff37}",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Shadow,
    Validated,
    Fallback,
    Primary,
    Sole,
}

#[derive(Clone, Copy)]
enum Arm {
    Historical,
    Candidate,
}

struct Comparison {
    ratios: Vec<f64>,
    baseline_ns: Vec<f64>,
    candidate_ns: Vec<f64>,
}

#[inline(never)]
fn historical(value: &str) -> Option<Stage> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "shadow" | "0" => Some(Stage::Shadow),
        "validated" | "1" => Some(Stage::Validated),
        "fallback" | "2" => Some(Stage::Fallback),
        "primary" | "3" => Some(Stage::Primary),
        "sole" | "4" => Some(Stage::Sole),
        _ => None,
    }
}

#[inline(never)]
fn candidate(value: &str) -> Option<Stage> {
    let value = value.trim();
    match value.len() {
        1 => match value.as_bytes()[0] {
            b'0' => Some(Stage::Shadow),
            b'1' => Some(Stage::Validated),
            b'2' => Some(Stage::Fallback),
            b'3' => Some(Stage::Primary),
            b'4' => Some(Stage::Sole),
            _ => None,
        },
        4 if value.eq_ignore_ascii_case("sole") => Some(Stage::Sole),
        6 if value.eq_ignore_ascii_case("shadow") => Some(Stage::Shadow),
        7 if value.eq_ignore_ascii_case("primary") => Some(Stage::Primary),
        8 if value.eq_ignore_ascii_case("fallback") => Some(Stage::Fallback),
        9 if value.eq_ignore_ascii_case("validated") => Some(Stage::Validated),
        _ => None,
    }
}

#[inline(never)]
fn match_normalized(value: &str) -> Option<Stage> {
    match value {
        "shadow" | "0" => Some(Stage::Shadow),
        "validated" | "1" => Some(Stage::Validated),
        "fallback" | "2" => Some(Stage::Fallback),
        "primary" | "3" => Some(Stage::Primary),
        "sole" | "4" => Some(Stage::Sole),
        _ => None,
    }
}

fn checksum_stage(stage: Option<Stage>) -> usize {
    match stage {
        None => 0,
        Some(Stage::Shadow) => 1,
        Some(Stage::Validated) => 2,
        Some(Stage::Fallback) => 3,
        Some(Stage::Primary) => 4,
        Some(Stage::Sole) => 5,
    }
}

fn assert_one(value: &str, cases: &mut usize, checksum: &mut usize) {
    let baseline = historical(black_box(value));
    let optimized = candidate(black_box(value));
    assert_eq!(optimized, baseline, "parser parity for {value:?}");
    *cases += 1;
    *checksum = checksum
        .wrapping_mul(31)
        .wrapping_add(checksum_stage(baseline));
}

fn assert_oracle() -> (usize, usize) {
    let mut cases = 0usize;
    let mut checksum = 0usize;
    let whitespace = ["", " ", "\t", "\n", "\r\n", "\u{2003}"];

    for (name, expected) in [
        ("shadow", Stage::Shadow),
        ("validated", Stage::Validated),
        ("fallback", Stage::Fallback),
        ("primary", Stage::Primary),
        ("sole", Stage::Sole),
    ] {
        let letters = name.as_bytes();
        for mask in 0..(1usize << letters.len()) {
            let mut spelling = letters.to_vec();
            for (index, byte) in spelling.iter_mut().enumerate() {
                if mask & (1 << index) != 0 {
                    *byte = byte.to_ascii_uppercase();
                }
            }
            let spelling = String::from_utf8(spelling).expect("ASCII stage name");
            for prefix in whitespace {
                for suffix in whitespace {
                    let value = format!("{prefix}{spelling}{suffix}");
                    assert_eq!(historical(&value), Some(expected));
                    assert_one(&value, &mut cases, &mut checksum);
                }
            }
        }
    }

    for alias in ["0", "1", "2", "3", "4"] {
        for prefix in whitespace {
            for suffix in whitespace {
                let value = format!("{prefix}{alias}{suffix}");
                assert_one(&value, &mut cases, &mut checksum);
            }
        }
    }

    for value in CORPUS {
        assert_one(value, &mut cases, &mut checksum);
    }

    let alphabet = [
        'a',
        'A',
        's',
        'S',
        '0',
        '4',
        '9',
        '-',
        '_',
        ' ',
        '\t',
        '\n',
        '\u{00df}',
        '\u{0130}',
        '\u{017f}',
        '\u{2003}',
        '\u{212a}',
        '\u{1f642}',
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..10_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let length = (state as usize >> 8) % 14;
        let mut value = String::new();
        for _ in 0..length {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            value.push(alphabet[(state as usize >> 16) % alphabet.len()]);
        }
        assert_one(&value, &mut cases, &mut checksum);
    }

    (cases, checksum)
}

fn run_arm(arm: Arm, repeats: usize) -> usize {
    let mut checksum = 0usize;
    for repeat in 0..repeats {
        for (index, value) in CORPUS.iter().enumerate() {
            let stage = match arm {
                Arm::Historical => historical(black_box(value)),
                Arm::Candidate => candidate(black_box(value)),
            };
            checksum = checksum.wrapping_add(
                checksum_stage(black_box(stage))
                    .wrapping_mul(index + 1)
                    .wrapping_add(repeat & 1),
            );
        }
    }
    black_box(checksum)
}

fn run_normalize_only(repeats: usize) -> usize {
    let mut checksum = 0usize;
    for repeat in 0..repeats {
        for (index, value) in CORPUS.iter().enumerate() {
            let normalized = black_box(value.trim().to_ascii_lowercase());
            checksum = checksum.wrapping_add(
                black_box(normalized.len())
                    .wrapping_mul(index + 1)
                    .wrapping_add(repeat & 1),
            );
        }
    }
    black_box(checksum)
}

fn run_match_only(normalized: &[String], repeats: usize) -> usize {
    let mut checksum = 0usize;
    for repeat in 0..repeats {
        for (index, value) in normalized.iter().enumerate() {
            checksum = checksum.wrapping_add(
                checksum_stage(black_box(match_normalized(black_box(value))))
                    .wrapping_mul(index + 1)
                    .wrapping_add(repeat & 1),
            );
        }
    }
    black_box(checksum)
}

fn elapsed_ns(run: impl FnOnce() -> usize) -> f64 {
    let started = Instant::now();
    black_box(run());
    started.elapsed().as_nanos() as f64
}

fn repetitions_for(target_ns: u64) -> usize {
    let probe_repeats = 2_048usize;
    let elapsed = elapsed_ns(|| run_arm(Arm::Historical, probe_repeats));
    let scaled = ((target_ns as f64 / elapsed.max(1.0)) * probe_repeats as f64) as usize;
    scaled.clamp(probe_repeats, 50_000_000)
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    percentile(&sorted, 0.5)
}

fn cv_percent(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean * 100.0
}

fn compare(candidate_arm: Arm, repeats: usize) -> Comparison {
    let mut ratios = Vec::with_capacity(AB_PAIRS * 2);
    let mut baseline_ns = Vec::with_capacity(AB_PAIRS * 2);
    let mut candidate_ns = Vec::with_capacity(AB_PAIRS * 2);

    for pair in 0..AB_PAIRS {
        let (base_first, candidate_first, candidate_second, base_second) = if pair % 2 == 0 {
            let base_first = elapsed_ns(|| run_arm(Arm::Historical, repeats));
            let candidate_first = elapsed_ns(|| run_arm(candidate_arm, repeats));
            let candidate_second = elapsed_ns(|| run_arm(candidate_arm, repeats));
            let base_second = elapsed_ns(|| run_arm(Arm::Historical, repeats));
            (base_first, candidate_first, candidate_second, base_second)
        } else {
            let candidate_first = elapsed_ns(|| run_arm(candidate_arm, repeats));
            let base_first = elapsed_ns(|| run_arm(Arm::Historical, repeats));
            let base_second = elapsed_ns(|| run_arm(Arm::Historical, repeats));
            let candidate_second = elapsed_ns(|| run_arm(candidate_arm, repeats));
            (base_first, candidate_first, candidate_second, base_second)
        };
        ratios.push(candidate_first / base_first);
        ratios.push(candidate_second / base_second);
        baseline_ns.extend([base_first / repeats as f64, base_second / repeats as f64]);
        candidate_ns.extend([
            candidate_first / repeats as f64,
            candidate_second / repeats as f64,
        ]);
    }

    Comparison {
        ratios,
        baseline_ns,
        candidate_ns,
    }
}

fn print_comparison(label: &str, comparison: &Comparison) -> (f64, f64, f64, f64, usize) {
    let mut ratios = comparison.ratios.clone();
    ratios.sort_by(f64::total_cmp);
    let p10 = percentile(&ratios, 0.1);
    let med = percentile(&ratios, 0.5);
    let p90 = percentile(&ratios, 0.9);
    let cv = cv_percent(&ratios);
    let wins = ratios.iter().filter(|ratio| **ratio < 1.0).count();
    println!(
        "{label} p10={p10:.6}x median={med:.6}x p90={p90:.6}x cv={cv:.4}% wins={wins}/{} baseline_ns_per_corpus={:.3} candidate_ns_per_corpus={:.3}",
        ratios.len(),
        median(&comparison.baseline_ns),
        median(&comparison.candidate_ns),
    );
    println!("{label}_ratios={ratios:?}");
    (p10, med, p90, cv, wins)
}

fn profile() {
    let (oracle_cases, oracle_checksum) = assert_oracle();
    let normalized: Vec<String> = CORPUS
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .collect();
    let repeats = repetitions_for(PROFILE_TARGET_NS);
    let mut historical_ns = Vec::with_capacity(PROFILE_SAMPLES);
    let mut normalize_ns = Vec::with_capacity(PROFILE_SAMPLES);
    let mut match_ns = Vec::with_capacity(PROFILE_SAMPLES);

    for sample in 0..PROFILE_SAMPLES {
        if sample % 2 == 0 {
            historical_ns.push(elapsed_ns(|| run_arm(Arm::Historical, repeats)) / repeats as f64);
            normalize_ns.push(elapsed_ns(|| run_normalize_only(repeats)) / repeats as f64);
            match_ns.push(elapsed_ns(|| run_match_only(&normalized, repeats)) / repeats as f64);
        } else {
            match_ns.push(elapsed_ns(|| run_match_only(&normalized, repeats)) / repeats as f64);
            normalize_ns.push(elapsed_ns(|| run_normalize_only(repeats)) / repeats as f64);
            historical_ns.push(elapsed_ns(|| run_arm(Arm::Historical, repeats)) / repeats as f64);
        }
    }

    let historical_median = median(&historical_ns);
    let normalize_median = median(&normalize_ns);
    let match_median = median(&match_ns);
    println!(
        "profile oracle_cases={oracle_cases} oracle_checksum={oracle_checksum} corpus={} repeats={repeats} samples={PROFILE_SAMPLES}",
        CORPUS.len(),
    );
    println!(
        "profile historical_ns_per_corpus={historical_median:.3} normalize_ns_per_corpus={normalize_median:.3} match_ns_per_corpus={match_median:.3} normalize_over_historical={:.4}% match_over_historical={:.4}%",
        normalize_median / historical_median * 100.0,
        match_median / historical_median * 100.0,
    );
}

fn measure() {
    let (oracle_cases, oracle_checksum) = assert_oracle();
    let repeats = repetitions_for(AB_TARGET_NS);
    println!(
        "measure oracle_cases={oracle_cases} oracle_checksum={oracle_checksum} corpus={} repeats={repeats} pairs={AB_PAIRS} ratios_per_comparison={} gate=candidate_p90_lt_null_p10_and_42_of_42_wins",
        CORPUS.len(),
        AB_PAIRS * 2,
    );

    let null = compare(Arm::Historical, repeats);
    let candidate = compare(Arm::Candidate, repeats);
    let (null_p10, _, _, _, _) = print_comparison("null_candidate_over_baseline", &null);
    let (_, candidate_median, candidate_p90, _, candidate_wins) =
        print_comparison("allocation_free_over_historical", &candidate);
    let accepted =
        candidate_p90 < null_p10 && candidate_wins == AB_PAIRS * 2 && candidate_median < 1.0;
    println!("verdict={}", if accepted { "KEEP" } else { "REJECT" });
    if !accepted {
        std::process::exit(1);
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("profile") => profile(),
        Some("measure") | None => measure(),
        Some(mode) => {
            eprintln!("unknown mode {mode:?}; expected profile or measure");
            std::process::exit(2);
        }
    }
}
