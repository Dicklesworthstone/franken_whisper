use std::fs::{self, File};
use std::hint::black_box;
use std::os::unix::fs::symlink;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const PROFILE_SAMPLES: usize = 11;
const PAIRS: usize = 21;
const TARGET_ARM_NS: u128 = 30_000_000;

#[derive(Debug, Eq, PartialEq)]
enum Outcome {
    Path(PathBuf),
    Error(String),
}

struct Fixture {
    root: PathBuf,
    regular: PathBuf,
    directory: PathBuf,
    missing: PathBuf,
    file_symlink: PathBuf,
    directory_symlink: PathBuf,
    broken_symlink: PathBuf,
    fifo: PathBuf,
    socket: PathBuf,
    _listener: UnixListener,
}

type CheckFn = fn(&Path) -> Outcome;
type BoolFn = fn(&Path) -> bool;

fn missing_error(path: &Path) -> Outcome {
    Outcome::Error(format!("input file does not exist: {}", path.display()))
}

fn not_file_error(path: &Path) -> Outcome {
    Outcome::Error(format!("input path is not a file: {}", path.display()))
}

#[inline(never)]
fn historical(path: &Path) -> Outcome {
    if !path.exists() {
        return missing_error(path);
    }
    if !path.is_file() {
        return not_file_error(path);
    }
    Outcome::Path(path.to_path_buf())
}

#[inline(never)]
fn candidate(path: &Path) -> Outcome {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Outcome::Path(path.to_path_buf()),
        Ok(_) => not_file_error(path),
        Err(_) => missing_error(path),
    }
}

#[inline(never)]
fn exists_only(path: &Path) -> bool {
    path.exists()
}

#[inline(never)]
fn is_file_only(path: &Path) -> bool {
    path.is_file()
}

fn create_fixture() -> Fixture {
    let nonce = SystemTime::now() // ubs:ignore — benchmark fixture ID, not a security token
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let process_id = std::process::id(); // ubs:ignore — benchmark fixture ID, not a security token
    let root = Path::new("/tmp").join(format!(
        "franken-whisper-audio-input-metadata-{process_id}-{nonce}"
    ));
    fs::create_dir_all(&root).expect("create fixture root");

    let regular = root.join("audio.wav");
    File::create(&regular).expect("create regular fixture");
    let directory = root.join("directory");
    fs::create_dir(&directory).expect("create directory fixture");
    let missing = root.join("missing.wav");

    let file_symlink = root.join("audio-link.wav");
    symlink(&regular, &file_symlink).expect("create file symlink");
    let directory_symlink = root.join("directory-link");
    symlink(&directory, &directory_symlink).expect("create directory symlink");
    let broken_symlink = root.join("broken-link.wav");
    symlink(&missing, &broken_symlink).expect("create broken symlink");

    let fifo = root.join("audio-fifo");
    let fifo_status = Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .expect("execute mkfifo");
    assert!(fifo_status.success(), "mkfifo failed: {fifo_status}");

    let socket = root.join("audio-socket");
    let listener = UnixListener::bind(&socket).expect("create Unix socket fixture");

    Fixture {
        root,
        regular,
        directory,
        missing,
        file_symlink,
        directory_symlink,
        broken_symlink,
        fifo,
        socket,
        _listener: listener,
    }
}

fn assert_outcome(path: &Path, expected: Outcome) {
    let baseline = historical(path);
    let proposed = candidate(path);
    assert_eq!(baseline, expected, "historical outcome for {path:?}");
    assert_eq!(proposed, expected, "candidate outcome for {path:?}");
}

fn assert_oracle(fixture: &Fixture) -> usize {
    assert_outcome(&fixture.regular, Outcome::Path(fixture.regular.clone()));
    assert_outcome(&fixture.directory, not_file_error(&fixture.directory));
    assert_outcome(&fixture.missing, missing_error(&fixture.missing));
    assert_outcome(
        Path::new(""),
        Outcome::Error("input file does not exist: ".to_owned()),
    );
    assert_outcome(
        &fixture.file_symlink,
        Outcome::Path(fixture.file_symlink.clone()),
    );
    assert_outcome(
        &fixture.directory_symlink,
        not_file_error(&fixture.directory_symlink),
    );
    assert_outcome(
        &fixture.broken_symlink,
        missing_error(&fixture.broken_symlink),
    );
    assert_outcome(&fixture.fifo, not_file_error(&fixture.fifo));
    assert_outcome(&fixture.socket, not_file_error(&fixture.socket));
    assert_outcome(
        Path::new("/dev/null"),
        not_file_error(Path::new("/dev/null")),
    );
    assert_outcome(
        Path::new("/proc/self/exe"),
        Outcome::Path(PathBuf::from("/proc/self/exe")),
    );
    11
}

fn measure_check(path: &Path, implementation: CheckFn, repetitions: usize) -> Duration {
    let started = Instant::now();
    let mut checksum = 0usize;
    for index in 0..repetitions {
        match implementation(black_box(path)) {
            Outcome::Path(value) => {
                checksum ^= value.as_os_str().len().wrapping_add(index & 1);
            }
            Outcome::Error(value) => {
                checksum ^= value.len().wrapping_add(index & 1);
            }
        }
    }
    black_box(checksum);
    started.elapsed()
}

fn measure_bool(path: &Path, implementation: BoolFn, repetitions: usize) -> Duration {
    let started = Instant::now();
    let mut checksum = 0usize;
    for index in 0..repetitions {
        checksum ^= usize::from(implementation(black_box(path))).wrapping_add(index & 1);
    }
    black_box(checksum);
    started.elapsed()
}

fn calibrated_repetitions(path: &Path) -> usize {
    let probe_repetitions = 10_000usize;
    let probe_ns = measure_check(path, historical, probe_repetitions)
        .as_nanos()
        .max(1);
    let scaled = TARGET_ARM_NS
        .saturating_mul(probe_repetitions as u128)
        .checked_div(probe_ns)
        .unwrap_or(probe_repetitions as u128);
    usize::try_from(scaled.clamp(5_000, 1_000_000)).expect("bounded repetitions")
}

fn median_u128(values: &[u128]) -> u128 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
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

fn run_profile(fixture: &Fixture, oracle_cases: usize) {
    let repetitions = calibrated_repetitions(&fixture.regular);
    let mut exists_samples = Vec::with_capacity(PROFILE_SAMPLES);
    let mut is_file_samples = Vec::with_capacity(PROFILE_SAMPLES);
    let mut historical_samples = Vec::with_capacity(PROFILE_SAMPLES);
    let mut candidate_samples = Vec::with_capacity(PROFILE_SAMPLES);

    for sample in 0..PROFILE_SAMPLES {
        let mut record = |slot: usize| match slot {
            0 => exists_samples
                .push(measure_bool(&fixture.regular, exists_only, repetitions).as_nanos()),
            1 => is_file_samples
                .push(measure_bool(&fixture.regular, is_file_only, repetitions).as_nanos()),
            2 => historical_samples
                .push(measure_check(&fixture.regular, historical, repetitions).as_nanos()),
            3 => candidate_samples
                .push(measure_check(&fixture.regular, candidate, repetitions).as_nanos()),
            _ => unreachable!(),
        };
        for offset in 0..4 {
            record((sample + offset) % 4);
        }
    }

    let exists_ns = median_u128(&exists_samples) as f64 / repetitions as f64;
    let is_file_ns = median_u128(&is_file_samples) as f64 / repetitions as f64;
    let historical_ns = median_u128(&historical_samples) as f64 / repetitions as f64;
    let candidate_ns = median_u128(&candidate_samples) as f64 / repetitions as f64;
    println!(
        "profile oracle_cases={oracle_cases} samples={PROFILE_SAMPLES} repetitions={repetitions} fixture={} historical_metadata_probes=2 candidate_metadata_probes=1",
        fixture.root.display()
    );
    println!("exists_ns_per_call={exists_ns:.3}");
    println!("is_file_ns_per_call={is_file_ns:.3}");
    println!("historical_ns_per_call={historical_ns:.3}");
    println!("candidate_design_ns_per_call={candidate_ns:.3}");
    println!(
        "exists_over_historical={:.4}% is_file_over_historical={:.4}% candidate_design_over_historical={:.4}%",
        exists_ns / historical_ns * 100.0,
        is_file_ns / historical_ns * 100.0,
        candidate_ns / historical_ns * 100.0
    );
}

struct PairedResults {
    ratios: Vec<f64>,
    numerator_ns: Vec<u128>,
    denominator_ns: Vec<u128>,
}

fn paired_ratios(
    path: &Path,
    numerator: CheckFn,
    denominator: CheckFn,
    repetitions: usize,
) -> PairedResults {
    let mut ratios = Vec::with_capacity(PAIRS * 2);
    let mut numerator_ns = Vec::with_capacity(PAIRS * 2);
    let mut denominator_ns = Vec::with_capacity(PAIRS * 2);
    for pair in 0..PAIRS {
        let (first_n, first_d, second_d, second_n) = if pair % 2 == 0 {
            (
                measure_check(path, numerator, repetitions).as_nanos(),
                measure_check(path, denominator, repetitions).as_nanos(),
                measure_check(path, denominator, repetitions).as_nanos(),
                measure_check(path, numerator, repetitions).as_nanos(),
            )
        } else {
            let first_d = measure_check(path, denominator, repetitions).as_nanos();
            let first_n = measure_check(path, numerator, repetitions).as_nanos();
            let second_n = measure_check(path, numerator, repetitions).as_nanos();
            let second_d = measure_check(path, denominator, repetitions).as_nanos();
            (first_n, first_d, second_d, second_n)
        };
        numerator_ns.extend([first_n, second_n]);
        denominator_ns.extend([first_d, second_d]);
        ratios.extend([
            first_n as f64 / first_d.max(1) as f64,
            second_n as f64 / second_d.max(1) as f64,
        ]);
    }
    PairedResults {
        ratios,
        numerator_ns,
        denominator_ns,
    }
}

fn print_results(label: &str, results: &PairedResults, repetitions: usize) {
    let wins = results.ratios.iter().filter(|ratio| **ratio < 1.0).count();
    let numerator_ns = median_u128(&results.numerator_ns) as f64 / repetitions as f64;
    let denominator_ns = median_u128(&results.denominator_ns) as f64 / repetitions as f64;
    println!(
        "{label} p10={:.6}x median={:.6}x p90={:.6}x cv={:.4}% wins={wins}/{} numerator_ns_per_call={numerator_ns:.3} denominator_ns_per_call={denominator_ns:.3}",
        percentile(&results.ratios, 10),
        percentile(&results.ratios, 50),
        percentile(&results.ratios, 90),
        coefficient_of_variation(&results.ratios) * 100.0,
        results.ratios.len(),
    );
}

fn run_measure(fixture: &Fixture, oracle_cases: usize) {
    let repetitions = calibrated_repetitions(&fixture.regular);
    for _ in 0..5 {
        black_box(measure_check(&fixture.regular, historical, repetitions));
        black_box(measure_check(&fixture.regular, candidate, repetitions));
    }

    let null = paired_ratios(&fixture.regular, historical, historical, repetitions);
    let comparison = paired_ratios(&fixture.regular, candidate, historical, repetitions);
    println!(
        "measure oracle_cases={oracle_cases} repetitions={repetitions} pairs={PAIRS} ratios_per_comparison={} fixture={} ratio=candidate_over_historical",
        PAIRS * 2,
        fixture.root.display()
    );
    print_results("null", &null, repetitions);
    print_results("candidate", &comparison, repetitions);

    let null_median = percentile(&null.ratios, 50);
    let null_cv = coefficient_of_variation(&null.ratios);
    let candidate_median = percentile(&comparison.ratios, 50);
    let candidate_p90 = percentile(&comparison.ratios, 90);
    let candidate_wins = comparison
        .ratios
        .iter()
        .filter(|ratio| **ratio < 1.0)
        .count();
    let keep = (0.97..=1.03).contains(&null_median)
        && null_cv <= 0.03
        && candidate_p90 < percentile(&null.ratios, 10)
        && candidate_wins == comparison.ratios.len()
        && candidate_median < 0.80;
    println!(
        "gate=null_median_0.97_1.03+null_cv_le_3pct+candidate_p90_lt_null_p10+all_wins+candidate_median_lt_0.80 verdict={}",
        if keep { "KEEP" } else { "REJECT" }
    );
    if !keep {
        std::process::exit(2);
    }
}

fn main() {
    let fixture = create_fixture();
    let oracle_cases = assert_oracle(&fixture);
    match std::env::args().nth(1).as_deref() {
        Some("profile") => run_profile(&fixture, oracle_cases),
        Some("measure") => run_measure(&fixture, oracle_cases),
        other => panic!("expected profile or measure mode, got {other:?}"), // ubs:ignore — invalid benchmark CLI is a harness failure
    }
}
