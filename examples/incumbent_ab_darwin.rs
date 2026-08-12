//! Darwin (Apple Silicon) incumbent A/B harness — the macOS counterpart of
//! `incumbent_ab.rs` (which is Linux-only: procfs/sysfs quiescence gates).
//!
//! Protocol `darwin-incumbent-ab-v1`:
//!   1. Fail-closed preflight against `docs/INCUMBENT_CONTRACT_DARWIN.json`
//!      (whisper-cli binary SHA-256; model SHA-256).
//!   2. Quiescence gates, sampled pre / between-pairs / post:
//!      - thermal: `pmset -g therm` must report `CPU_Speed_Limit = 100`
//!        (a thermally clamped host measures the ramp, not the code);
//!      - process census: `ps -Ao pid,pcpu,comm`; any non-harness process
//!        sustaining > 10% of one core in BOTH the pre and post censuses
//!        vetoes the run (kernel_task is kernel accounting, not a
//!        competitor, and is excluded — recorded in the report).
//!   3. Order-alternating comparison pairs (fw, wc) / (wc, fw), whole-job
//!      wall time, matched greedy (`-bs 1 -bo 1`; fw is greedy by default).
//!   4. Dual same-invocation A/A nulls (fw-vs-fw and wc-vs-wc, same
//!      alternating shape). Decidable ONLY when (a) both null medians lie in
//!      [0.98, 1.02], (b) the comparison bootstrap CI95 excludes 1.0, and
//!      (c) the comparison median clears the widest null-CI edge's distance
//!      from 1.0 by 2x.
//!   5. Transcript conformance: normalized word streams must differ by less
//!      than 10% or the row is refused regardless of timing.
//!
//! Result classes follow docs/PERF_LEDGER.md: a verdict from a loaded host is
//! UNDECIDABLE, never a loosened threshold.
//!
//! Usage:
//!   FW_BIN=/path/to/fw cargo run --release --example incumbent_ab_darwin -- \
//!       [pairs=5] [clip=tests/fixtures/native/jfk.wav]

use std::process::Command;
use std::time::Instant;

use sha2::{Digest, Sha256};

const CONTRACT: &str = "docs/INCUMBENT_CONTRACT_DARWIN.json";
const NULL_BAND: (f64, f64) = (0.98, 1.02);
const CENSUS_PCPU_VETO: f64 = 10.0; // percent of one core
const TRANSCRIPT_MAX_WORD_DIFF: f64 = 0.10;

fn sha256_file(path: &str) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut h = Sha256::new();
    h.update(&bytes);
    format!("{:x}", h.finalize())
}

fn run_out(cmd: &str, args: &[&str]) -> String {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawn {cmd}: {e}"));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `pmset -g therm` gate: CPU_Speed_Limit must be 100 (or the key absent —
/// some Macs omit it entirely when unclamped).
fn thermal_ok() -> (bool, String) {
    let out = run_out("pmset", &["-g", "therm"]);
    for line in out.lines() {
        if let Some(v) = line.trim().strip_prefix("CPU_Speed_Limit") {
            let val: f64 = v
                .trim_start_matches(['=', ' ', '\t'])
                .trim()
                .parse()
                .unwrap_or(0.0);
            return (val >= 100.0, out.trim().to_owned());
        }
    }
    (true, out.trim().to_owned())
}

/// Non-harness processes above the veto threshold: `(pid, pcpu, comm)`.
fn census(exclude: &[&str]) -> Vec<(u32, f64, String)> {
    let out = run_out("ps", &["-Ao", "pid,pcpu,comm"]);
    let mut hot = Vec::new();
    for line in out.lines().skip(1) {
        let mut it = line.split_whitespace();
        let (Some(pid), Some(pcpu)) = (it.next(), it.next()) else {
            continue;
        };
        let comm = it.collect::<Vec<_>>().join(" ");
        let pcpu: f64 = pcpu.parse().unwrap_or(0.0);
        if pcpu <= CENSUS_PCPU_VETO {
            continue;
        }
        let name = comm.rsplit('/').next().unwrap_or(&comm);
        // kernel_task is the kernel's own accounting, not a compute competitor.
        if name == "kernel_task" || exclude.iter().any(|e| name.contains(e)) {
            continue;
        }
        hot.push((pid.parse().unwrap_or(0), pcpu, name.to_owned()));
    }
    hot
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Deterministic bootstrap CI95 of the median (splitmix64; no wall-clock seed).
fn bootstrap_ci95(samples: &[f64]) -> (f64, f64) {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let n = samples.len();
    let mut medians: Vec<f64> = (0..10_000)
        .map(|_| {
            let resample: Vec<f64> = (0..n).map(|_| samples[(next() as usize) % n]).collect();
            median(resample)
        })
        .collect();
    medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (medians[249], medians[9749])
}

fn normalize_words(text: &str) -> Vec<String> {
    // whisper-cli stdout lines carry `[00:00:00.000 --> 00:00:05.000]` prefixes;
    // strip bracketed spans so timestamps don't count as transcript words.
    let mut stripped = String::with_capacity(text.len());
    let mut depth = 0usize;
    for c in text.chars() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            _ if depth == 0 => stripped.push(c),
            _ => {}
        }
    }
    stripped
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn word_diff_fraction(a: &[String], b: &[String]) -> f64 {
    // Levenshtein over words, normalized by the longer stream.
    let (n, m) = (a.len(), b.len());
    if n == 0 && m == 0 {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m] as f64 / n.max(m) as f64
}

struct Arm<'a> {
    label: &'a str,
    cmd: String,
    args: Vec<String>,
}

fn time_arm(arm: &Arm) -> (f64, String) {
    let t = Instant::now();
    let out = Command::new(&arm.cmd)
        .args(&arm.args)
        .output()
        .unwrap_or_else(|e| panic!("spawn {}: {e}", arm.cmd));
    let secs = t.elapsed().as_secs_f64();
    assert!(
        out.status.success(),
        "{} exited {:?}: {}",
        arm.label,
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    (secs, String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Alternating paired ratios `first/second` (per pair, order swaps).
fn paired_ratios(a: &Arm, b: &Arm, pairs: usize, label: &str) -> Vec<f64> {
    let mut ratios = Vec::with_capacity(pairs);
    for p in 0..pairs {
        let (ta, tb) = if p % 2 == 0 {
            let (ta, _) = time_arm(a);
            let (tb, _) = time_arm(b);
            (ta, tb)
        } else {
            let (tb, _) = time_arm(b);
            let (ta, _) = time_arm(a);
            (ta, tb)
        };
        let r = ta / tb;
        eprintln!(
            "  [{label}] pair {}: {}={ta:.3}s {}={tb:.3}s ratio={r:.4}",
            p + 1,
            a.label,
            b.label
        );
        ratios.push(r);
    }
    ratios
}

fn main() {
    let pairs: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let clip = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "tests/fixtures/native/jfk.wav".to_owned());

    // ── Preflight: contract pins ────────────────────────────────────────
    let contract: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(CONTRACT).expect("contract file"))
            .expect("contract json");
    let inc = &contract["incumbent"];
    let wc_path = std::env::var("WHISPER_CLI")
        .unwrap_or_else(|_| inc["default_path"].as_str().unwrap().to_owned());
    let wc_sha = sha256_file(&wc_path);
    let pinned_sha = inc["binary_sha256"].as_str().unwrap();
    assert_eq!(
        wc_sha, pinned_sha,
        "FAIL CLOSED: whisper-cli at {wc_path} (sha {wc_sha}) drifted from the pinned contract ({pinned_sha})"
    );
    let fw_bin = std::env::var("FW_BIN").expect("set FW_BIN to the fw release binary");
    let model = std::env::var("FW_AB_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.cache/franken_whisper/models/whisper/whisper-large-v3-turbo-f16-v1/ggml-large-v3-turbo.bin",
            std::env::var("HOME").unwrap()
        )
    });
    let model_sha = sha256_file(&model);
    let pinned_model = inc["model_sha256"]["ggml-large-v3-turbo.bin"]
        .as_str()
        .unwrap();
    assert_eq!(
        model_sha, pinned_model,
        "FAIL CLOSED: model sha drifted from the darwin contract"
    );
    eprintln!("preflight: whisper-cli + model match the darwin contract pins");

    // ── Quiescence gates (pre) ──────────────────────────────────────────
    let harness = ["fw", "whisper-cli", "incumbent_ab_darwin", "ps", "pmset"];
    let (t_ok, t_raw) = thermal_ok();
    let pre_census = census(&harness);
    let fw = Arm {
        label: "fw",
        cmd: fw_bin.clone(),
        args: [
            "transcribe",
            "--input",
            &clip,
            "--no-persist",
            "--no-diarize",
            "--json",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect(),
    };
    let wc = Arm {
        label: "wc",
        cmd: wc_path.clone(),
        args: ["-m", &model, "-f", &clip, "-bs", "1", "-bo", "1", "-np"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
    };

    // ── Transcript conformance (one run each, also warms both) ─────────
    let (_, fw_json) = time_arm(&fw);
    let fw_text: String = serde_json::from_str::<serde_json::Value>(&fw_json)
        .ok()
        .and_then(|v| v["result"]["transcript"].as_str().map(str::to_owned))
        .expect("fw --json transcript");
    let (_, wc_out) = time_arm(&wc);
    let (fw_words, wc_words) = (normalize_words(&fw_text), normalize_words(&wc_out));
    let word_diff = word_diff_fraction(&fw_words, &wc_words);
    eprintln!("transcript conformance: normalized word diff = {word_diff:.4}");

    // ── Measurement: comparison + dual A/A nulls, all interleaved ───────
    let comparison = paired_ratios(&wc, &fw, pairs, "wc/fw");
    let null_fw = paired_ratios(&fw, &fw, pairs, "fw A/A");
    let null_wc = paired_ratios(&wc, &wc, pairs, "wc A/A");
    let post_census = census(&harness);

    // ── Verdict ─────────────────────────────────────────────────────────
    let cmp_median = median(comparison.clone());
    let (ci_lo, ci_hi) = bootstrap_ci95(&comparison);
    let nfw_m = median(null_fw.clone());
    let nwc_m = median(null_wc.clone());
    let (nfw_lo, nfw_hi) = bootstrap_ci95(&null_fw);
    let (nwc_lo, nwc_hi) = bootstrap_ci95(&null_wc);
    let widest_null_edge = [nfw_lo, nfw_hi, nwc_lo, nwc_hi]
        .iter()
        .map(|e| (e - 1.0).abs())
        .fold(0.0f64, f64::max);

    let sustained: Vec<&(u32, f64, String)> = pre_census
        .iter()
        .filter(|(_, _, name)| post_census.iter().any(|(_, _, n2)| n2 == name))
        .collect();

    let mut reasons: Vec<String> = Vec::new();
    if !t_ok {
        reasons.push(format!("thermal clamp active: {t_raw}"));
    }
    if !sustained.is_empty() {
        reasons.push(format!(
            "sustained non-harness load (>{CENSUS_PCPU_VETO}% pcpu pre AND post): {:?}",
            sustained
                .iter()
                .map(|(_, p, n)| format!("{n}({p:.0}%)"))
                .collect::<Vec<_>>()
        ));
    }
    if !(NULL_BAND.0..=NULL_BAND.1).contains(&nfw_m) {
        reasons.push(format!("fw A/A null median {nfw_m:.4} outside [0.98,1.02]"));
    }
    if !(NULL_BAND.0..=NULL_BAND.1).contains(&nwc_m) {
        reasons.push(format!("wc A/A null median {nwc_m:.4} outside [0.98,1.02]"));
    }
    if word_diff > TRANSCRIPT_MAX_WORD_DIFF {
        reasons.push(format!(
            "transcript conformance failed: word diff {word_diff:.3} > {TRANSCRIPT_MAX_WORD_DIFF}"
        ));
    }
    let ci_excludes_one = ci_lo > 1.0 || ci_hi < 1.0;
    let clears_margin = (cmp_median - 1.0).abs() >= 2.0 * widest_null_edge;
    if reasons.is_empty() && !ci_excludes_one {
        reasons.push(format!(
            "comparison CI95 [{ci_lo:.4},{ci_hi:.4}] includes 1.0"
        ));
    }
    if reasons.is_empty() && !clears_margin {
        reasons.push(format!(
            "effect |{:.4}| does not clear 2x widest null edge ({widest_null_edge:.4})",
            cmp_median - 1.0
        ));
    }

    let verdict = if reasons.is_empty() {
        if cmp_median > 1.0 { "WIN" } else { "LOSS" }
    } else {
        "UNDECIDABLE"
    };
    let report = serde_json::json!({
        "protocol": "darwin-incumbent-ab-v1",
        "verdict": verdict,
        "reasons": reasons,
        "comparison": {
            "statistic": "median of per-pair wc_wall/fw_wall (order-alternating)",
            "median": cmp_median,
            "ci95": [ci_lo, ci_hi],
            "ratios": comparison,
        },
        "nulls": {
            "fw": {"median": nfw_m, "ci95": [nfw_lo, nfw_hi]},
            "wc": {"median": nwc_m, "ci95": [nwc_lo, nwc_hi]},
        },
        "transcript_word_diff": word_diff,
        "quiescence": {
            "thermal_ok": t_ok,
            "pre_census_hot": pre_census.iter().map(|(_, p, n)| format!("{n}:{p:.0}%")).collect::<Vec<_>>(),
            "post_census_hot": post_census.iter().map(|(_, p, n)| format!("{n}:{p:.0}%")).collect::<Vec<_>>(),
        },
        "pins": {
            "whisper_cli_sha256": wc_sha,
            "model_sha256": model_sha,
            "fw_bin": fw_bin,
            "clip": clip,
            "pairs": pairs,
        },
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
