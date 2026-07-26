use std::hint::black_box;
use std::time::Instant;

use sha2::{Digest, Sha256};

const SAMPLES: usize = 11;
const TARGET_SAMPLE_NS: u64 = 80_000_000;
const AB_PAIRS: usize = 41;
const AB_TARGET_NS: u64 = 2_000_000;
const MIN_OF: usize = 3;
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

struct BenchCase {
    program: &'static str,
    args: Vec<String>,
    rendered: Vec<String>,
}

#[derive(Clone, Copy)]
enum ProfileArm {
    Historical,
    CollectOnly,
    ScanOnly,
    JoinOnly,
}

fn is_sensitive_flag(flag: &str) -> bool {
    matches!(
        flag,
        "--hf-token"
            | "--hf_token"
            | "--api-key"
            | "--api_key"
            | "--access-token"
            | "--access_token"
            | "--auth-token"
            | "--auth_token"
            | "--password"
            | "--pass"
            | "--secret"
            | "--secret-key"
            | "--secret_key"
    )
}

fn historical_collect(program: &str, args: &[String]) -> Vec<String> {
    let mut rendered = Vec::with_capacity(args.len() + 1);
    rendered.push(program.to_owned());

    let mut redact_next = false;
    for arg in args {
        if redact_next {
            rendered.push("***".to_owned());
            redact_next = false;
            continue;
        }

        if let Some((flag, _value)) = arg.split_once('=')
            && is_sensitive_flag(flag)
        {
            rendered.push(format!("{flag}=***"));
            continue;
        }

        if is_sensitive_flag(arg) {
            rendered.push(arg.clone());
            redact_next = true;
            continue;
        }

        rendered.push(arg.clone());
    }

    rendered
}

fn historical(program: &str, args: &[String]) -> String {
    if args.is_empty() {
        return program.to_owned();
    }
    historical_collect(program, args).join(" ")
}

fn direct(program: &str, args: &[String]) -> String {
    if args.is_empty() {
        return program.to_owned();
    }

    let mut capacity = program.len().saturating_add(args.len());
    let mut redact_next = false;
    for arg in args {
        if redact_next {
            capacity = capacity.saturating_add(3);
            redact_next = false;
            continue;
        }

        if let Some((flag, _value)) = arg.split_once('=')
            && is_sensitive_flag(flag)
        {
            capacity = capacity.saturating_add(flag.len().saturating_add(4));
            continue;
        }

        capacity = capacity.saturating_add(arg.len());
        if is_sensitive_flag(arg) {
            redact_next = true;
        }
    }

    let mut rendered = String::with_capacity(capacity);
    rendered.push_str(program);

    let mut redact_next = false;
    for arg in args {
        rendered.push(' ');
        if redact_next {
            rendered.push_str("***");
            redact_next = false;
            continue;
        }

        if let Some((flag, _value)) = arg.split_once('=')
            && is_sensitive_flag(flag)
        {
            rendered.push_str(flag);
            rendered.push_str("=***");
            continue;
        }

        rendered.push_str(arg);
        if is_sensitive_flag(arg) {
            redact_next = true;
        }
    }

    rendered
}

fn scan_only(program: &str, args: &[String]) -> usize {
    if args.is_empty() {
        return program.len();
    }

    let mut output_len = program.len() + args.len();
    let mut redact_next = false;
    for arg in args {
        if redact_next {
            output_len += 3;
            redact_next = false;
            continue;
        }

        if let Some((flag, _value)) = arg.split_once('=')
            && is_sensitive_flag(flag)
        {
            output_len += flag.len() + 4;
            continue;
        }

        output_len += arg.len();
        if is_sensitive_flag(arg) {
            redact_next = true;
        }
    }
    output_len
}

fn make_cases() -> Vec<BenchCase> {
    let ffmpeg = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "input with spaces.flac",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
    ];
    let ytdlp = [
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--format",
        "bestaudio/best",
        "--output",
        "cache/%(id)s.%(ext)s",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--hf-token",
        "hf_secret_value",
        "--api-key=another-secret",
        "--socket-timeout",
        "30",
        "https://example.invalid/watch?v=unicode-λ",
        "--newline",
    ];
    let mut backend = Vec::with_capacity(64);
    for index in 0..64 {
        let arg = match index {
            7 => "--access_token".to_owned(),
            8 => "access-secret".to_owned(),
            31 => "--password=hunter2".to_owned(),
            47 => "--token-threshold".to_owned(),
            _ => format!("--option-{index}=value-{index}"),
        };
        backend.push(arg);
    }

    [
        ("ffmpeg", ffmpeg.iter().map(ToString::to_string).collect()),
        ("yt-dlp", ytdlp.iter().map(ToString::to_string).collect()),
        ("python3", backend),
    ]
    .into_iter()
    .map(|(program, args)| {
        let rendered = historical_collect(program, &args);
        BenchCase {
            program,
            args,
            rendered,
        }
    })
    .collect()
}

fn time_arm(cases: &[BenchCase], arm: ProfileArm, inner: usize) -> u64 {
    let started = Instant::now();
    for _ in 0..inner {
        for case in cases {
            match arm {
                ProfileArm::Historical => {
                    black_box(historical(black_box(case.program), black_box(&case.args)));
                }
                ProfileArm::CollectOnly => {
                    black_box(historical_collect(
                        black_box(case.program),
                        black_box(&case.args),
                    ));
                }
                ProfileArm::ScanOnly => {
                    black_box(scan_only(black_box(case.program), black_box(&case.args)));
                }
                ProfileArm::JoinOnly => {
                    black_box(black_box(&case.rendered).join(" "));
                }
            }
        }
    }
    started.elapsed().as_nanos() as u64
}

fn calibrate(cases: &[BenchCase], arm: ProfileArm) -> usize {
    let trial_inner = 512;
    let elapsed = time_arm(cases, arm, trial_inner).max(1);
    let scaled = (TARGET_SAMPLE_NS as u128 * trial_inner as u128 / elapsed as u128) as usize;
    scaled.clamp(1, 10_000_000)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn run_profile() {
    let cases = make_cases();
    for case in &cases {
        assert_eq!(
            historical(case.program, &case.args),
            case.rendered.join(" ")
        );
    }

    let historical_inner = calibrate(&cases, ProfileArm::Historical);
    let collect_inner = calibrate(&cases, ProfileArm::CollectOnly);
    let scan_inner = calibrate(&cases, ProfileArm::ScanOnly);
    let join_inner = calibrate(&cases, ProfileArm::JoinOnly);

    let mut historical_ns = Vec::with_capacity(SAMPLES);
    let mut collect_ns = Vec::with_capacity(SAMPLES);
    let mut scan_ns = Vec::with_capacity(SAMPLES);
    let mut join_ns = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        let order = match sample % 4 {
            0 => [
                ProfileArm::Historical,
                ProfileArm::CollectOnly,
                ProfileArm::ScanOnly,
                ProfileArm::JoinOnly,
            ],
            1 => [
                ProfileArm::JoinOnly,
                ProfileArm::Historical,
                ProfileArm::CollectOnly,
                ProfileArm::ScanOnly,
            ],
            2 => [
                ProfileArm::ScanOnly,
                ProfileArm::JoinOnly,
                ProfileArm::Historical,
                ProfileArm::CollectOnly,
            ],
            _ => [
                ProfileArm::CollectOnly,
                ProfileArm::ScanOnly,
                ProfileArm::JoinOnly,
                ProfileArm::Historical,
            ],
        };
        for arm in order {
            let inner = match arm {
                ProfileArm::Historical => historical_inner,
                ProfileArm::CollectOnly => collect_inner,
                ProfileArm::ScanOnly => scan_inner,
                ProfileArm::JoinOnly => join_inner,
            };
            let elapsed = time_arm(&cases, arm, inner) as f64 / inner as f64;
            match arm {
                ProfileArm::Historical => historical_ns.push(elapsed),
                ProfileArm::CollectOnly => collect_ns.push(elapsed),
                ProfileArm::ScanOnly => scan_ns.push(elapsed),
                ProfileArm::JoinOnly => join_ns.push(elapsed),
            }
        }
    }

    let historical = median(&mut historical_ns);
    let collect = median(&mut collect_ns);
    let scan = median(&mut scan_ns);
    let join = median(&mut join_ns);
    let owned_token_churn = 100.0 * (collect - scan).max(0.0) / historical;
    let join_floor = 100.0 * join / historical;
    println!(
        "PROFILE worker_pid={} samples={SAMPLES} target_sample_ns={TARGET_SAMPLE_NS}",
        std::process::id()
    );
    println!(
        "inners historical={historical_inner} collect={collect_inner} scan={scan_inner} join={join_inner}"
    );
    println!("historical_median_ns={historical:.3}");
    println!("collect_only_median_ns={collect:.3}");
    println!("scan_only_median_ns={scan:.3}");
    println!("join_only_median_ns={join:.3}");
    println!("historical_over_scan={:.6}", historical / scan);
    println!("historical_over_join_floor={:.6}", historical / join);
    println!("collect_over_scan={:.6}", collect / scan);
    println!("owned_token_churn_pct_of_historical={owned_token_churn:.4}");
    println!("join_floor_pct_of_historical={join_floor:.4}");
    println!("note=independently timed stages are not additive");
}

#[derive(Clone, Copy)]
enum RenderArm {
    Historical,
    Direct,
}

fn time_renderer(cases: &[BenchCase], arm: RenderArm, inner: usize) -> u64 {
    let started = Instant::now();
    for _ in 0..inner {
        for case in cases {
            let rendered = match arm {
                RenderArm::Historical => historical(black_box(case.program), black_box(&case.args)),
                RenderArm::Direct => direct(black_box(case.program), black_box(&case.args)),
            };
            black_box(rendered);
        }
    }
    started.elapsed().as_nanos() as u64
}

fn calibrate_renderer(cases: &[BenchCase], arm: RenderArm) -> usize {
    let trial_inner = 512;
    let elapsed = time_renderer(cases, arm, trial_inner).max(1);
    let scaled = (AB_TARGET_NS as u128 * trial_inner as u128 / elapsed as u128) as usize;
    scaled.clamp(1, 10_000_000)
}

fn min_ns_per_round(cases: &[BenchCase], arm: RenderArm, inner: usize) -> f64 {
    (0..MIN_OF)
        .map(|_| time_renderer(cases, arm, inner) as f64)
        .fold(f64::INFINITY, f64::min)
        / inner as f64
}

fn measure_ratios(
    cases: &[BenchCase],
    contender: RenderArm,
    historical_inner: usize,
    contender_inner: usize,
) -> Vec<f64> {
    let mut ratios = Vec::with_capacity(AB_PAIRS);
    for pair in 0..AB_PAIRS {
        let (historical_ns, contender_ns) = if pair % 2 == 0 {
            (
                min_ns_per_round(cases, RenderArm::Historical, historical_inner),
                min_ns_per_round(cases, contender, contender_inner),
            )
        } else {
            let contender_ns = min_ns_per_round(cases, contender, contender_inner);
            let historical_ns = min_ns_per_round(cases, RenderArm::Historical, historical_inner);
            (historical_ns, contender_ns)
        };
        ratios.push(historical_ns / contender_ns.max(f64::MIN_POSITIVE));
    }
    ratios
}

struct RatioStats {
    p10: f64,
    median: f64,
    p90: f64,
    cv: f64,
    wins: usize,
}

fn ratio_stats(ratios: &[f64]) -> RatioStats {
    let mut sorted = ratios.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let variance = ratios
        .iter()
        .map(|ratio| {
            let delta = ratio - mean;
            delta * delta
        })
        .sum::<f64>()
        / ratios.len() as f64;
    RatioStats {
        p10: sorted[(sorted.len() - 1) / 10],
        median: sorted[AB_PAIRS / 2],
        p90: sorted[(sorted.len() - 1) * 9 / 10],
        cv: variance.sqrt() / mean,
        wins: ratios.iter().filter(|ratio| **ratio > 1.0).count(),
    }
}

fn bootstrap_median_ci(ratios: &[f64]) -> (f64, f64) {
    let mut state = 0xbb67_ae85_84ca_a73b_u64 ^ ratios.len() as u64;
    let mut sample = Vec::with_capacity(ratios.len());
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        sample.clear();
        for _ in 0..ratios.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sample.push(ratios[state as usize % ratios.len()]);
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

fn print_ratios(label: &str, ratios: &[f64], stats: &RatioStats) {
    let values = ratios
        .iter()
        .map(|ratio| format!("{ratio:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{label}_ratios={values}");
    let (ci_low, ci_high) = bootstrap_median_ci(ratios);
    println!(
        "{label}_p10={:.6} {label}_median={:.6} {label}_p90={:.6} {label}_median_ci95=[{ci_low:.6},{ci_high:.6}] {label}_cv={:.6} {label}_wins={}/{}",
        stats.p10, stats.median, stats.p90, stats.cv, stats.wins, AB_PAIRS
    );
}

fn verify_parity(cases: &[BenchCase]) -> usize {
    const SENSITIVE_FLAGS: [&str; 13] = [
        "--hf-token",
        "--hf_token",
        "--api-key",
        "--api_key",
        "--access-token",
        "--access_token",
        "--auth-token",
        "--auth_token",
        "--password",
        "--pass",
        "--secret",
        "--secret-key",
        "--secret_key",
    ];

    let mut checked = 0;
    let mut verify = |program: &str, args: Vec<String>| {
        let old = historical(program, &args);
        let new = direct(program, &args);
        assert_eq!(old.as_bytes(), new.as_bytes());
        checked += 1;
    };

    for case in cases {
        verify(case.program, case.args.clone());
    }
    for flag in SENSITIVE_FLAGS {
        let secret = format!("UNIQUE_SECRET_{flag}");
        let separate = vec![flag.to_owned(), secret.clone()];
        let inline = vec![format!("{flag}={secret}")];
        assert!(!direct("prog", &separate).contains(&secret));
        assert!(!direct("prog", &inline).contains(&secret));
        verify("prog", separate);
        verify("prog", inline);
    }

    let fixtures = [
        ("", Vec::<String>::new()),
        ("prog", Vec::<String>::new()),
        ("", vec!["arg".to_owned()]),
        ("prog", vec![String::new(), String::new()]),
        ("prog", vec!["--secret".to_owned()]),
        (
            "prog",
            vec![
                "--api-key".to_owned(),
                "--password".to_owned(),
                "tail".to_owned(),
            ],
        ),
        (
            "prog",
            vec!["--secret".to_owned(), "--api-key=x".to_owned()],
        ),
        (
            "prog",
            vec![
                "--api-key=".to_owned(),
                "--api-key=one=two".to_owned(),
                "--API-KEY=value".to_owned(),
                "--token-threshold".to_owned(),
            ],
        ),
        ("pr\0g", vec!["spaces tabs\tlines\nλ\\quotes\"".to_owned()]),
        (
            "prog",
            vec!["normal".repeat(32_768), "--secret=x".repeat(32_768)],
        ),
    ];
    for (program, args) in fixtures {
        verify(program, args);
    }

    let many_args = (0..256)
        .map(|index| match index {
            63 => "--auth-token".to_owned(),
            64 => "many-arg-secret".to_owned(),
            127 => "--password=inline-secret".to_owned(),
            _ => format!("arg-{index}"),
        })
        .collect();
    verify("prog", many_args);
    checked
}

fn run_ab() {
    let cases = make_cases();
    let parity_cases = verify_parity(&cases);
    let historical_inner = calibrate_renderer(&cases, RenderArm::Historical);
    let direct_inner = calibrate_renderer(&cases, RenderArm::Direct);

    let null_ratios = measure_ratios(
        &cases,
        RenderArm::Historical,
        historical_inner,
        historical_inner,
    );
    let candidate_ratios =
        measure_ratios(&cases, RenderArm::Direct, historical_inner, direct_inner);
    let null = ratio_stats(&null_ratios);
    let candidate = ratio_stats(&candidate_ratios);
    let (null_ci_low, null_ci_high) = bootstrap_median_ci(&null_ratios);
    let null_half_width = (1.0 - null_ci_low).abs().max((null_ci_high - 1.0).abs());
    let required_speedup = 1.0 + 2.0 * null_half_width;
    let keep = candidate.median >= required_speedup;

    println!(
        "AB worker_pid={} pairs={AB_PAIRS} target_ns={AB_TARGET_NS} min_of={MIN_OF} parity_cases={parity_cases}",
        std::process::id()
    );
    println!("inners historical_calibrated={historical_inner} direct_calibrated={direct_inner}");
    print_ratios("null", &null_ratios, &null);
    print_ratios("candidate", &candidate_ratios, &candidate);
    println!(
        "gate=median_vs_null_ci95_2x_margin null_half_width={null_half_width:.6} required_speedup={required_speedup:.6} candidate_median={:.6} cv_is_provenance_only=true verdict={}",
        candidate.median,
        if keep { "KEEP" } else { "REJECT" }
    );
}

fn main() {
    println!("bench_elf_sha256={}", self_identity());
    if std::env::args().any(|arg| arg.eq("profile")) {
        run_profile();
    } else {
        run_ab();
    }
}
