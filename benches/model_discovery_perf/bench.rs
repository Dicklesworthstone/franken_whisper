use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::hint::black_box;
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::time::Instant;

const PREF: &[&str] = &[
    "large-v3-turbo",
    "large-v3",
    "large-v2",
    "large",
    "medium.en",
    "medium",
    "small.en",
    "small",
    "base.en",
    "base",
    "tiny.en",
    "tiny",
];

#[derive(Clone, Copy, Debug, Default)]
struct ScanStats {
    entries: usize,
    path_constructions: usize,
    metadata_probes: usize,
}

struct Fixture {
    root: PathBuf,
    crowded: PathBuf,
    custom: PathBuf,
    custom_tie: PathBuf,
    empty: PathBuf,
    first: PathBuf,
    later: PathBuf,
}

#[derive(Clone, Copy)]
enum Arm {
    Baseline,
    Candidate,
}

fn model_short_name(name: &OsStr) -> Option<&str> {
    name.to_str()?.strip_prefix("ggml-")?.strip_suffix(".bin")
}

#[inline(never)]
fn discover_baseline<const PROFILE: bool>(
    dirs: &[PathBuf],
    stats: &mut ScanStats,
) -> Option<PathBuf> {
    for dir in dirs {
        let Ok(read_dir) = fs::read_dir(dir) else {
            continue;
        };
        let mut found: Vec<(usize, String, PathBuf)> = Vec::new();
        for entry in read_dir.flatten() {
            if PROFILE {
                stats.entries += 1;
                stats.path_constructions += 1;
                stats.metadata_probes += 1;
            }
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(short) = path.file_name().and_then(model_short_name) {
                let rank = PREF.iter().position(|q| *q == short).unwrap_or(PREF.len());
                found.push((rank, short.to_owned(), path));
            }
        }
        if !found.is_empty() {
            found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            return Some(found.into_iter().next().expect("non-empty").2);
        }
    }
    None
}

#[inline(never)]
fn discover_candidate<const PROFILE: bool>(
    dirs: &[PathBuf],
    stats: &mut ScanStats,
) -> Option<PathBuf> {
    for dir in dirs {
        let Ok(read_dir) = fs::read_dir(dir) else {
            continue;
        };
        let mut found: Vec<(usize, String, PathBuf)> = Vec::new();
        for entry in read_dir.flatten() {
            if PROFILE {
                stats.entries += 1;
            }
            let file_name = entry.file_name();
            let Some(short) = model_short_name(&file_name) else {
                continue;
            };
            if PROFILE {
                stats.path_constructions += 1;
                stats.metadata_probes += 1;
            }
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let rank = PREF.iter().position(|q| *q == short).unwrap_or(PREF.len());
            found.push((rank, short.to_owned(), path));
        }
        if !found.is_empty() {
            found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            return Some(found.into_iter().next().expect("non-empty").2);
        }
    }
    None
}

fn touch(path: impl AsRef<Path>) {
    File::create(path).expect("create fixture file");
}

fn create_fixture() -> Fixture {
    let root = std::env::temp_dir().join(format!(
        "franken-whisper-model-discovery-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create fixture root");

    let crowded = root.join("crowded");
    let custom = root.join("custom");
    let custom_tie = root.join("custom-tie");
    let empty = root.join("empty");
    let first = root.join("first");
    let later = root.join("later");
    for dir in [&crowded, &custom, &custom_tie, &empty, &first, &later] {
        fs::create_dir_all(dir).expect("create fixture directory");
    }

    for index in 0..4096 {
        touch(crowded.join(format!("artifact-{index:04}.json")));
    }
    for name in ["tiny", "large-v3", "large-v3-turbo"] {
        touch(crowded.join(format!("ggml-{name}.bin")));
    }
    for name in [
        "ggml-near.bin.part",
        "prefix-ggml-base.bin",
        "ggml-base.gguf",
    ] {
        touch(crowded.join(name));
    }

    touch(custom.join("payload.dat"));
    touch(custom.join("ggml-zeta.bin"));
    symlink(custom.join("payload.dat"), custom.join("ggml-medium.bin"))
        .expect("create model symlink");
    symlink(
        custom.join("missing-target"),
        custom.join("ggml-large-v3.bin"),
    )
    .expect("create broken model symlink");
    fs::create_dir(custom.join("ggml-large-v3-turbo.bin")).expect("create model-shaped directory");

    touch(custom_tie.join("ggml-zeta.bin"));
    touch(custom_tie.join("ggml-alpha.bin"));

    touch(empty.join("ggml-base.bin.part"));
    touch(empty.join("not-a-model.bin"));
    touch(empty.join(OsString::from_vec(b"ggml-\xff.bin".to_vec())));

    touch(first.join("ggml-zeta.bin"));
    touch(later.join("ggml-large-v3-turbo.bin"));

    Fixture {
        root,
        crowded,
        custom,
        custom_tie,
        empty,
        first,
        later,
    }
}

fn selected_name(path: Option<&PathBuf>) -> Option<&OsStr> {
    path.and_then(|path| path.file_name())
}

fn assert_oracle(fixture: &Fixture) {
    let scenarios = [
        (
            "crowded",
            vec![fixture.crowded.clone()],
            Some("ggml-large-v3-turbo.bin"),
        ),
        (
            "symlink",
            vec![fixture.custom.clone()],
            Some("ggml-medium.bin"),
        ),
        (
            "custom-tie",
            vec![fixture.custom_tie.clone()],
            Some("ggml-alpha.bin"),
        ),
        ("empty", vec![fixture.empty.clone()], None),
        (
            "directory-precedence",
            vec![fixture.first.clone(), fixture.later.clone()],
            Some("ggml-zeta.bin"),
        ),
        (
            "fall-through",
            vec![fixture.empty.clone(), fixture.later.clone()],
            Some("ggml-large-v3-turbo.bin"),
        ),
    ];

    for (label, dirs, expected) in scenarios {
        let baseline = discover_baseline::<false>(&dirs, &mut ScanStats::default());
        let candidate = discover_candidate::<false>(&dirs, &mut ScanStats::default());
        assert_eq!(candidate, baseline, "selection parity for {label}");
        assert_eq!(
            selected_name(candidate.as_ref()),
            expected.map(OsStr::new),
            "golden selection for {label}"
        );
    }
}

fn run_arm(arm: Arm, dirs: &[PathBuf], scans: usize) -> f64 {
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..scans {
        let selected = match arm {
            Arm::Baseline => discover_baseline::<false>(dirs, &mut ScanStats::default()),
            Arm::Candidate => discover_candidate::<false>(dirs, &mut ScanStats::default()),
        };
        checksum ^= selected.as_ref().map_or(0, |path| path.as_os_str().len());
        black_box(&selected);
    }
    black_box(checksum);
    started.elapsed().as_secs_f64() * 1_000_000.0
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn summarize(label: &str, values: &mut [f64]) {
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    println!(
        "{label} p10={:.6} median={:.6} p90={:.6} cv={:.6} samples={}",
        percentile(values, 0.10),
        percentile(values, 0.50),
        percentile(values, 0.90),
        variance.sqrt() / mean,
        values.len()
    );
    println!("{label}_RATIOS {values:?}");
}

fn profile(fixture: &Fixture) {
    assert_oracle(fixture);
    let dirs = [fixture.crowded.clone()];
    let mut baseline_stats = ScanStats::default();
    let mut candidate_stats = ScanStats::default();
    let baseline = discover_baseline::<true>(&dirs, &mut baseline_stats);
    let candidate = discover_candidate::<true>(&dirs, &mut candidate_stats);
    assert_eq!(candidate, baseline);

    for _ in 0..4 {
        black_box(discover_baseline::<false>(&dirs, &mut ScanStats::default()));
        black_box(discover_candidate::<false>(
            &dirs,
            &mut ScanStats::default(),
        ));
    }
    let baseline_us = run_arm(Arm::Baseline, &dirs, 5);
    let candidate_us = run_arm(Arm::Candidate, &dirs, 5);
    let avoided = baseline_stats.metadata_probes - candidate_stats.metadata_probes;
    println!("PROFILE fixture={}", fixture.root.display());
    println!(
        "PROFILE baseline entries={} paths={} metadata_probes={} wall_us={baseline_us:.3}",
        baseline_stats.entries, baseline_stats.path_constructions, baseline_stats.metadata_probes
    );
    println!(
        "PROFILE candidate entries={} paths={} metadata_probes={} wall_us={candidate_us:.3}",
        candidate_stats.entries,
        candidate_stats.path_constructions,
        candidate_stats.metadata_probes
    );
    println!(
        "PROFILE avoided_metadata_probes={} avoided_share={:.6} candidate_over_baseline={:.6}",
        avoided,
        avoided as f64 / baseline_stats.metadata_probes as f64,
        candidate_us / baseline_us
    );
    println!("PARITY exact_selected_path=true scenarios=6");
}

fn measure(fixture: &Fixture) {
    assert_oracle(fixture);
    let dirs = [fixture.crowded.clone()];
    for _ in 0..8 {
        black_box(discover_baseline::<false>(&dirs, &mut ScanStats::default()));
        black_box(discover_candidate::<false>(
            &dirs,
            &mut ScanStats::default(),
        ));
    }

    let mut null_ratios = Vec::with_capacity(42);
    let mut candidate_ratios = Vec::with_capacity(42);
    let mut baseline_times = Vec::with_capacity(42);
    let mut candidate_times = Vec::with_capacity(42);
    for _ in 0..21 {
        let null_a1 = run_arm(Arm::Baseline, &dirs, 3);
        let null_b1 = run_arm(Arm::Baseline, &dirs, 3);
        let null_b2 = run_arm(Arm::Baseline, &dirs, 3);
        let null_a2 = run_arm(Arm::Baseline, &dirs, 3);
        null_ratios.push(null_b1 / null_a1);
        null_ratios.push(null_b2 / null_a2);

        let baseline_1 = run_arm(Arm::Baseline, &dirs, 3);
        let candidate_1 = run_arm(Arm::Candidate, &dirs, 3);
        let candidate_2 = run_arm(Arm::Candidate, &dirs, 3);
        let baseline_2 = run_arm(Arm::Baseline, &dirs, 3);
        candidate_ratios.push(candidate_1 / baseline_1);
        candidate_ratios.push(candidate_2 / baseline_2);
        baseline_times.extend([baseline_1, baseline_2]);
        candidate_times.extend([candidate_1, candidate_2]);
    }

    let wins = candidate_ratios
        .iter()
        .filter(|ratio| **ratio < 1.0)
        .count();
    summarize("NULL candidate_over_baseline", &mut null_ratios);
    summarize("AB candidate_over_baseline", &mut candidate_ratios);
    baseline_times.sort_by(f64::total_cmp);
    candidate_times.sort_by(f64::total_cmp);
    println!(
        "ARMS baseline_us={:.3} candidate_us={:.3} baseline_over_candidate={:.6} wins={wins}/{}",
        percentile(&baseline_times, 0.50),
        percentile(&candidate_times, 0.50),
        percentile(&baseline_times, 0.50) / percentile(&candidate_times, 0.50),
        candidate_ratios.len()
    );
    println!("PARITY exact_selected_path=true scenarios=6");
}

fn main() {
    let fixture = create_fixture();
    match std::env::args().nth(1).as_deref() {
        Some("profile") => profile(&fixture),
        Some("measure") | None => measure(&fixture),
        Some(mode) => {
            eprintln!("unknown mode: {mode}");
            std::process::exit(2);
        }
    }
}
