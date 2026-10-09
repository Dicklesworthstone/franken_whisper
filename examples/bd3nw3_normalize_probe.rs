// Probe (bd-3nw3): builtin symphonia normalize (arm A) vs the forced ffmpeg
// subprocess normalize (arm B) on the same compressed input, interleaved in
// one invocation with a second builtin arm (A') as the same-binary A/A null.
// The force flag is read from the environment per call, so the parent
// re-executes this binary once per timed run (`--one`) with the flag set for
// arm B only; each child times `normalize_to_wav` alone. Arm order rotates
// every round. Prints per-arm medians and bootstrap CI95s of the per-round
// ratios A'/A (null) and B/A (candidate).
//
//   cargo run --release --example bd3nw3_normalize_probe -- <input> [rounds]
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const DEFAULT_ROUNDS: usize = 9;
const ARMS: [&str; 3] = ["A", "A'", "B"];
const FORCE_FFMPEG: &str = "FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--one") {
        one(Path::new(&args[2]), Path::new(&args[3]));
        return;
    }
    let input = args
        .get(1)
        .expect("usage: bd3nw3_normalize_probe <input> [rounds]");
    let rounds = args
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&rounds| rounds >= 3)
        .unwrap_or(DEFAULT_ROUNDS);
    let work = std::env::temp_dir().join(format!("bd3nw3_work_{}", std::process::id()));
    std::fs::create_dir_all(&work).expect("work dir");
    let exe = std::env::current_exe().expect("probe executable path");

    let run = |arm: usize| -> (f64, u64) {
        let mut command = Command::new(&exe);
        command.arg("--one").arg(input).arg(&work);
        if arm == 2 {
            command.env(FORCE_FFMPEG, "1");
        } else {
            command.env_remove(FORCE_FFMPEG);
        }
        let output = command.output().expect("run probe child");
        assert!(output.status.success(), "probe child failed: {output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let field = |key: &str| -> f64 {
            stdout
                .split_whitespace()
                .find_map(|token| token.strip_prefix(key))
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("child output lacks {key}: {stdout}"))
        };
        (field("elapsed_ms="), field("out_bytes=") as u64)
    };

    // One untimed run of each variant warms the page cache.
    run(0);
    run(2);

    let mut ms = vec![[0.0f64; 3]; rounds];
    let mut bytes = [0u64; 3];
    for (round, row) in ms.iter_mut().enumerate() {
        for offset in 0..3 {
            let arm = (round + offset) % 3;
            let (elapsed, out_bytes) = run(arm);
            row[arm] = elapsed;
            bytes[arm] = out_bytes;
        }
    }
    let _ = std::fs::remove_dir(&work);

    for (arm, name) in ARMS.iter().enumerate() {
        let mut column: Vec<f64> = ms.iter().map(|row| row[arm]).collect();
        println!(
            "arm {name}: median {:.1} ms over {rounds} rounds ({} output bytes)",
            median(&mut column),
            bytes[arm]
        );
    }
    let null: Vec<f64> = ms.iter().map(|row| row[1] / row[0]).collect();
    let candidate: Vec<f64> = ms.iter().map(|row| row[2] / row[0]).collect();
    let (null_median, null_lo, null_hi) = bootstrap_median_ci95(&null);
    let (cand_median, cand_lo, cand_hi) = bootstrap_median_ci95(&candidate);
    println!(
        "same-invocation A/A null A'/A: median {null_median:.4}, bootstrap CI95 [{null_lo:.4}, {null_hi:.4}]"
    );
    println!("candidate B/A: median {cand_median:.4}, bootstrap CI95 [{cand_lo:.4}, {cand_hi:.4}]");
    let widest = (null_hi - 1.0).abs().max((1.0 - null_lo).abs());
    println!(
        "2x null margin: decidable iff the candidate CI95 lies outside [{:.4}, {:.4}]",
        1.0 - 2.0 * widest,
        1.0 + 2.0 * widest
    );
}

/// One timed normalization; the variant follows the inherited environment.
fn one(input: &Path, work: &Path) {
    let t = Instant::now();
    let out =
        franken_whisper::audio::normalize_to_wav(input, work).expect("normalize must succeed");
    let elapsed = t.elapsed();
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "elapsed_ms={:.3} out_bytes={size}",
        elapsed.as_secs_f64() * 1e3
    );
    // Remove the output so the next run starts from identical disk state.
    let _ = std::fs::remove_file(&out);
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Median and percentile-bootstrap CI95 of the median (10,000 resamples,
/// fixed seed so a rerun on the same timings prints the same interval).
fn bootstrap_median_ci95(values: &[f64]) -> (f64, f64, f64) {
    const RESAMPLES: usize = 10_000;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut medians = Vec::with_capacity(RESAMPLES);
    let mut resample = vec![0.0; values.len()];
    for _ in 0..RESAMPLES {
        for slot in &mut resample {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *slot = values[(state % values.len() as u64) as usize];
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    let lo = medians[RESAMPLES * 25 / 1000];
    let hi = medians[RESAMPLES * 975 / 1000 - 1];
    (median(&mut values.to_vec()), lo, hi)
}
