//! Live-incumbent A/B: franken_whisper vs `whisper.cpp`, both driven from **one
//! invocation**, order-alternating, with an A/A null for **each** arm.
//!
//! ## Why this exists
//!
//! A self-speedup (our own code, before vs after) is maintenance. A competitive
//! claim requires a measured ratio against the actual legacy incumbent, produced
//! by a harness that runs the incumbent **side by side in the same invocation** —
//! otherwise the two engines are measured under different machine states and the
//! ratio inherits an uncontrolled between-session drift.
//!
//! This harness certifies live-incumbent transcription workloads with the
//! incumbent interleaved inside one invocation and with a null for each engine.
//!
//! ## What is measured, and why it is the fair quantity
//!
//! The default `transcribe_only` scope times **transcribe work, excluding
//! one-time model load**:
//!
//! - `whisper.cpp` self-reports `load time` and `total time`; its transcribe
//!   time is `total − load`.
//! - franken is timed in-process around `transcribe_samples`, with the model
//!   already resident.
//!
//! Comparing full process wall would instead compare `whisper-cli`'s thin
//! inference binary against franken's *orchestrator* (routing, storage,
//! normalization), which is not the quantity in question and would understate
//! franken. Excluding load on both sides is the matched comparison.
//!
//! Residual asymmetry, disclosed rather than hidden: `whisper.cpp` additionally
//! pays process spawn and stdout formatting inside its `total`, on the order of
//! milliseconds against a ~1.7 s measurement. It is not subtracted.
//!
//! Phase 2's `whole_job` scope removes that asymmetry. Each observation starts a
//! fresh process for each engine and parent-observed wall includes process
//! startup, model and audio I/O, inference, output serialization/formatting, and
//! teardown. The franken arm re-executes this same ELF in a hidden worker mode.
//!
//! ## Statistic and gate
//!
//! Per round, both engines run once, **alternating which goes first**, so any
//! monotonic machine drift lands on both arms equally. The statistic is the
//! **median of per-round ratios** (`wc_transcribe / fw_transcribe`).
//!
//! Two A/A nulls run in the same invocation and the same alternating shape:
//! franken against itself, and `whisper.cpp` against itself. A claim is
//! decidable only when the comparison median lies outside **both** null CI95s
//! with a 2× margin, and when the comparison medians from independently
//! sampled lighter- and heavier-load rounds differ by at most 0.1×. `cv` is
//! recorded as provenance and decides nothing. Whole-job runs also census
//! persistent non-harness processes between every arm: a process consuming more
//! than 0.1 CPU core makes the result undecidable, because steady cross-tool
//! load bias can survive both numerical controls.
//!
//! ## Usage
//!
//! ```text
//! incumbent_ab <model_short> <wav> [rounds]
//! FW_INCUMBENT_BIN=/path/to/whisper-cli   (default: legacy_whispercpp/.../whisper-cli)
//! FW_INCUMBENT_THREADS=27                 (screen the host and use its best setting)
//! FW_BENCH_SCOPE=whole_job                 (fresh-process end-to-end jobs)
//! FW_BENCH_THREADS=64                      (same requested count for both arms)
//! FW_WORKLOAD_NAME=meeting-text-only       (ledger-facing workload identity)
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use franken_whisper::conformance::word_error_rate;
use franken_whisper::native_engine::decode::{DecodeParams, LoadedModel, transcribe_samples};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MAX_LOAD_SPLIT_GAP: f64 = 0.1;
const MAX_CROSS_ENGINE_WER: f64 = 0.1;
const MAX_EXTERNAL_CPU_CORE_FRACTION: f64 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchScope {
    TranscribeOnly,
    WholeJob,
}

impl BenchScope {
    fn from_env() -> Self {
        match std::env::var("FW_BENCH_SCOPE").as_deref() {
            Ok("whole_job") => Self::WholeJob,
            Ok("transcribe_only") | Err(_) => Self::TranscribeOnly,
            Ok(other) => {
                panic!("unsupported FW_BENCH_SCOPE={other:?}; use transcribe_only or whole_job")
            }
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::TranscribeOnly => "transcribe_only",
            Self::WholeJob => "whole_job",
        }
    }
}

#[derive(Debug)]
struct Observation {
    measured_ms: f64,
    chars: usize,
    words: usize,
    segments: usize,
    transcript: String,
    transcript_sha256: String,
    actual_threads: usize,
}

#[derive(Clone, Debug)]
struct ProcessCpu {
    ticks: u64,
    command: String,
}

#[derive(Debug)]
struct ProcessSample {
    captured: Instant,
    by_pid: BTreeMap<u32, ProcessCpu>,
}

#[derive(Debug, Default)]
struct ExternalCpuActivity {
    max_core_fraction: f64,
    pid: u32,
    command: String,
    checkpoint: String,
    intervals: usize,
}

impl ExternalCpuActivity {
    fn observe(
        &mut self,
        previous: &ProcessSample,
        current: &ProcessSample,
        excluded_pids: &BTreeSet<u32>,
        clock_ticks_per_second: f64,
        checkpoint: &str,
    ) {
        self.intervals += 1;
        let elapsed = current
            .captured
            .duration_since(previous.captured)
            .as_secs_f64();
        if elapsed <= 0.0 || clock_ticks_per_second <= 0.0 {
            return;
        }
        for (pid, current_cpu) in &current.by_pid {
            if excluded_pids.contains(pid) {
                continue;
            }
            // A process first observed at this checkpoint started during the
            // interval (or raced the earlier `/proc` scan). Treating its prior
            // ticks as zero gives a conservative lower bound on interval CPU
            // use and prevents a competitor launched during the final arm from
            // escaping the gate.
            let previous_ticks = previous.by_pid.get(pid).map_or(0, |cpu| cpu.ticks);
            let delta_ticks = current_cpu.ticks.saturating_sub(previous_ticks);
            let core_fraction = delta_ticks as f64 / clock_ticks_per_second / elapsed;
            if core_fraction > self.max_core_fraction {
                self.max_core_fraction = core_fraction;
                self.pid = *pid;
                self.command.clone_from(&current_cpu.command);
                self.checkpoint.clear();
                self.checkpoint.push_str(checkpoint);
            }
        }
    }
}

/// Read a PCM16 WAV into mono f32. Mirrors `e2e_probe`'s reader.
fn read_wav_mono16k(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");
    let mut pos = 12;
    let mut channels = 1u16;
    let mut bits = 16u16;
    let mut data: &[u8] = &[];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
            bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
        } else if id == b"data" {
            data = &bytes[body..(body + sz).min(bytes.len())];
        }
        pos = body + sz + (sz & 1);
    }
    assert_eq!(bits, 16, "expected PCM16");
    let step = 2 * channels as usize;
    let mut samples = Vec::with_capacity(data.len() / step);
    let mut i = 0;
    while i + step <= data.len() {
        let mut acc = 0i32;
        for c in 0..channels as usize {
            acc += i16::from_le_bytes([data[i + 2 * c], data[i + 2 * c + 1]]) as i32;
        }
        samples.push((acc as f32 / channels as f32) / 32768.0);
        i += step;
    }
    samples
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(bytes) => sha256_bytes(&bytes),
        Err(_) => "unreadable".to_owned(),
    }
}

/// Self-reported identity of this harness binary (campaign harness contract).
fn executable_identity() -> String {
    match std::env::current_exe() {
        Ok(path) => format!("{} {}", sha256_file(&path), path.display()),
        Err(error) => format!("unavailable ({error})"),
    }
}

fn proc_status_value(field: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name == field).then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|cpuinfo| {
            cpuinfo.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name.trim() == "model name").then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn logical_threads() -> usize {
    std::fs::read_to_string("/proc/cpuinfo").map_or(0, |cpuinfo| {
        cpuinfo
            .lines()
            .filter(|line| line.starts_with("processor"))
            .count()
    })
}

fn physical_cores() -> usize {
    let Ok(cpu_dirs) = std::fs::read_dir("/sys/devices/system/cpu") else {
        return 0;
    };
    let mut cores = BTreeSet::new();
    for entry in cpu_dirs.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name
            .strip_prefix("cpu")
            .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()))
        {
            continue;
        }
        let topology = entry.path().join("topology");
        let package = std::fs::read_to_string(topology.join("physical_package_id"));
        let core = std::fs::read_to_string(topology.join("core_id"));
        if let (Ok(package), Ok(core)) = (package, core) {
            cores.insert((package.trim().to_owned(), core.trim().to_owned()));
        }
    }
    cores.len()
}

fn runtime_isa() -> Vec<&'static str> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2");
        }
        if std::is_x86_feature_detected!("avx") {
            features.push("avx");
        }
        if std::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::is_x86_feature_detected!("fma") {
            features.push("fma");
        }
        if std::is_x86_feature_detected!("f16c") {
            features.push("f16c");
        }
        if std::is_x86_feature_detected!("bmi1") {
            features.push("bmi1");
        }
        if std::is_x86_feature_detected!("bmi2") {
            features.push("bmi2");
        }
        if std::is_x86_feature_detected!("aes") {
            features.push("aes");
        }
    }
    features
}

fn host_provenance() -> Value {
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|_| "unavailable".to_owned());
    let cpuset_effective = std::fs::read_to_string("/sys/fs/cgroup/cpuset.cpus.effective")
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|_| "unavailable".to_owned());
    json!({
        "hostname": hostname,
        "cpu_model": cpu_model(),
        "physical_cores": physical_cores(),
        "logical_threads": logical_threads(),
        "available_threads": std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
        "affinity_cpus_allowed": proc_status_value("Cpus_allowed_list"),
        "cpuset_effective": cpuset_effective,
        "runtime_isa": runtime_isa(),
    })
}

fn parse_proc_stat(text: &str) -> Option<(u32, u64)> {
    // The command is parenthesized and may itself contain spaces or `)`, so the
    // final `) ` is the only safe boundary before the fixed-position fields.
    let command_end = text.rfind(") ")?;
    let fields: Vec<&str> = text[command_end + 2..].split_whitespace().collect();
    let parent_pid = fields.get(1)?.parse().ok()?;
    let user_ticks: u64 = fields.get(11)?.parse().ok()?;
    let system_ticks: u64 = fields.get(12)?.parse().ok()?;
    Some((parent_pid, user_ticks.saturating_add(system_ticks)))
}

fn process_sample() -> ProcessSample {
    let mut by_pid = BTreeMap::new();
    if let Ok(entries) = std::fs::read_dir("/proc") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(pid) = name.to_str().and_then(|value| value.parse::<u32>().ok()) else {
                continue;
            };
            let path = entry.path();
            let Some((_, ticks)) = std::fs::read_to_string(path.join("stat"))
                .ok()
                .and_then(|text| parse_proc_stat(&text))
            else {
                continue;
            };
            let command = std::fs::read_to_string(path.join("comm"))
                .map(|value| value.trim().to_owned())
                .unwrap_or_else(|_| "unavailable".to_owned());
            by_pid.insert(pid, ProcessCpu { ticks, command });
        }
    }
    ProcessSample {
        captured: Instant::now(),
        by_pid,
    }
}

fn process_ancestry() -> BTreeSet<u32> {
    let mut excluded = BTreeSet::new();
    let mut pid = std::process::id();
    for _ in 0..64 {
        if pid == 0 || !excluded.insert(pid) {
            break;
        }
        let Some((parent_pid, _)) = std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|text| parse_proc_stat(&text))
        else {
            break;
        };
        pid = parent_pid;
    }
    excluded
}

fn clock_ticks_per_second() -> f64 {
    Command::new("getconf")
        .arg("CLK_TCK")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|value| value.trim().parse().ok())
        .filter(|ticks: &f64| ticks.is_finite() && *ticks > 0.0)
        .unwrap_or(100.0)
}

fn run_then_checkpoint<F, G>(run: &F, checkpoint: &mut G, label: &str) -> Observation
where
    F: Fn() -> Observation,
    G: FnMut(&str),
{
    let observation = run();
    checkpoint(label);
    observation
}

/// Independent host-load covariate sampled before each measured round.
///
/// This must not be derived from either timed arm. Splitting on `fw_ms + wc_ms`
/// is endogenous because the grouping variable contains the numerator and
/// denominator of the comparison ratio, creating correlation by construction.
fn load_average_1m() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next()?.parse().ok())
        .unwrap_or(f64::NAN)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        f64::midpoint(sorted[mid - 1], sorted[mid])
    }
}

/// Deterministic bootstrap CI95 of the median (fixed LCG seed — no `rand`, and
/// reproducible across runs so a reviewer can re-derive the interval).
fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut medians = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut sample = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let idx = (state >> 33) as usize % values.len();
            sample.push(values[idx]);
        }
        medians.push(median(&sample));
    }
    medians.sort_by(f64::total_cmp);
    (medians[50], medians[1949])
}

fn cv(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    var.sqrt() / mean
}

/// Split ratios at the median of an independent covariate and return the
/// comparison median for the lighter and heavier halves plus their absolute
/// gap. For an odd number of rounds, the single middle-covariate round is
/// deliberately excluded.
fn split_by_covariate(covariate: &[f64], compare: &[f64]) -> Option<(f64, f64, f64)> {
    if covariate.len() != compare.len()
        || covariate.len() < 3
        || covariate.iter().any(|value| !value.is_finite())
        || compare.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let mut ranked: Vec<(f64, f64)> = covariate
        .iter()
        .zip(compare)
        .map(|(load, ratio)| (*load, *ratio))
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));

    let half = ranked.len() / 2;
    let light: Vec<f64> = ranked[..half].iter().map(|(_, ratio)| *ratio).collect();
    let heavy: Vec<f64> = ranked[ranked.len() - half..]
        .iter()
        .map(|(_, ratio)| *ratio)
        .collect();
    let light_median = median(&light);
    let heavy_median = median(&heavy);
    Some((
        light_median,
        heavy_median,
        (light_median - heavy_median).abs(),
    ))
}

fn incumbent_segments(stdout: &str, timestamps: bool) -> Vec<String> {
    if timestamps {
        stdout
            .lines()
            .filter_map(|line| {
                if !line.starts_with('[') {
                    return None;
                }
                let (_, segment) = line.split_once(']')?;
                let segment = segment.trim();
                (!segment.is_empty()).then(|| segment.to_owned())
            })
            .collect()
    } else {
        stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect()
    }
}

fn incumbent_actual_threads(text: &str) -> Option<usize> {
    text.lines().find_map(|line| {
        let (_, suffix) = line.split_once("n_threads")?;
        let (_, value) = suffix.split_once('=')?;
        value.split_whitespace().next()?.parse().ok()
    })
}

/// One `whisper.cpp` run.
///
/// `transcribe_only` measures the incumbent's self-reported `total − load`.
/// `whole_job` measures parent-observed process wall, including process startup,
/// model/audio I/O, inference, output formatting, and teardown.
fn run_incumbent(
    bin: &Path,
    model: &Path,
    wav: &str,
    threads: usize,
    timestamps: bool,
    scope: BenchScope,
) -> Observation {
    let mut args: Vec<String> = vec![
        "-m".into(),
        model.display().to_string(),
        "-f".into(),
        wav.into(),
        "-bs".into(),
        "1".into(),
        "-bo".into(),
        "1".into(),
        "-t".into(),
        threads.to_string(),
    ];
    // Match franken's mode on the incumbent side too: comparing franken's
    // no-timestamp decode against whisper.cpp's timestamped decode would charge
    // the incumbent for timestamp work franken is not doing.
    if !timestamps {
        args.push("-nt".into());
    }
    let started = Instant::now();
    let output = Command::new(bin)
        .args(&args)
        .output()
        .expect("run whisper-cli");
    let process_wall_ms = started.elapsed().as_secs_f64() * 1e3;
    assert!(
        output.status.success(),
        "whisper-cli exited with {}",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let text = String::from_utf8_lossy(&output.stderr).into_owned() + &stdout;

    let field = |needle: &str| -> f64 {
        text.lines()
            .find(|line| line.contains(needle))
            .and_then(|line| {
                line.rsplit('=')
                    .next()
                    .map(|tail| tail.trim().trim_end_matches(" ms").trim())
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse::<f64>().ok())
            })
            .unwrap_or(f64::NAN)
    };
    let total = field("total time");
    let load = field("load time");
    assert!(
        total.is_finite() && load.is_finite(),
        "could not parse whisper.cpp timings; is this whisper-cli?"
    );
    let segments = incumbent_segments(&stdout, timestamps);
    let chars = segments.iter().map(String::len).sum();
    let transcript = segments.join(" ");
    let words = transcript.split_whitespace().count();
    let transcript_sha256 = sha256_bytes(transcript.as_bytes());
    Observation {
        measured_ms: match scope {
            BenchScope::TranscribeOnly => total - load,
            BenchScope::WholeJob => process_wall_ms,
        },
        chars,
        words,
        segments: segments.len(),
        transcript,
        transcript_sha256,
        actual_threads: incumbent_actual_threads(&text).unwrap_or(0),
    }
}

/// One franken run: transcribe with the model already resident.
fn run_franken_resident(
    model: &LoadedModel,
    samples: &[f32],
    params: &DecodeParams,
) -> Observation {
    let started = Instant::now();
    let out = transcribe_samples(model, samples, params, &(|| Ok(()))).expect("fw transcribe");
    let elapsed = started.elapsed().as_secs_f64() * 1e3;
    let segments: Vec<String> = out
        .segments
        .iter()
        .map(|segment| segment.text.trim().to_owned())
        .collect();
    let chars = segments.iter().map(String::len).sum();
    let transcript = segments.join(" ");
    Observation {
        measured_ms: elapsed,
        chars,
        words: transcript.split_whitespace().count(),
        segments: segments.len(),
        transcript_sha256: sha256_bytes(transcript.as_bytes()),
        transcript,
        actual_threads: rayon::current_num_threads(),
    }
}

fn run_franken_whole(
    model: &Path,
    model_short: &str,
    wav: &str,
    threads: usize,
    timestamps: bool,
) -> Observation {
    let current_exe = std::env::current_exe().expect("resolve harness executable");
    let started = Instant::now();
    let output = Command::new(current_exe)
        .arg("--franken-worker")
        .arg(model)
        .arg(model_short)
        .arg(wav)
        .arg(if timestamps { "ts" } else { "no_ts" })
        .arg(threads.to_string())
        .env("RAYON_NUM_THREADS", threads.to_string())
        .env("FW_LOAD_WORKERS", threads.to_string())
        .output()
        .expect("run franken whole-job worker");
    let process_wall_ms = started.elapsed().as_secs_f64() * 1e3;
    assert!(
        output.status.success(),
        "franken worker exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let payload = stdout
        .lines()
        .find_map(|line| line.strip_prefix("FW_WORKER_RESULT "))
        .unwrap_or_else(|| panic!("franken worker emitted no result: {stdout}"));
    let value: Value = serde_json::from_str(payload).expect("parse franken worker result");
    let field_u64 = |name: &str| {
        value[name]
            .as_u64()
            .unwrap_or_else(|| panic!("franken worker result missing {name}")) as usize
    };
    let transcript = value["transcript"]
        .as_str()
        .expect("franken worker result missing transcript")
        .to_owned();
    Observation {
        measured_ms: process_wall_ms,
        chars: field_u64("chars"),
        words: field_u64("words"),
        segments: field_u64("segments"),
        transcript_sha256: value["transcript_sha256"]
            .as_str()
            .expect("franken worker result missing transcript_sha256")
            .to_owned(),
        transcript,
        actual_threads: field_u64("actual_threads"),
    }
}

fn franken_worker_main(args: &[String]) {
    let model_path = Path::new(args.get(2).expect("worker model path"));
    let model_short = args.get(3).expect("worker model short name");
    let wav = args.get(4).expect("worker wav");
    let timestamps = args.get(5).map(String::as_str) != Some("no_ts");
    let threads: usize = args
        .get(6)
        .and_then(|value| value.parse().ok())
        .expect("worker threads");

    let samples = read_wav_mono16k(wav);
    let model = GgmlModel::load(model_path)
        .and_then(LoadedModel::from_ggml)
        .expect("worker load model");
    let params = DecodeParams {
        language: Some("en".to_owned()),
        translate: false,
        timestamps,
        n_threads: threads,
        max_text_ctx: None,
        word_timestamps: false,
        model_hint: Some(model_short.to_owned()),
        ..DecodeParams::default()
    };
    let out =
        transcribe_samples(&model, &samples, &params, &(|| Ok(()))).expect("worker transcribe");
    let segments: Vec<String> = out
        .segments
        .iter()
        .map(|segment| segment.text.trim().to_owned())
        .collect();
    let chars = segments.iter().map(String::len).sum::<usize>();
    let transcript = segments.join(" ");
    let serialized_segments =
        serde_json::to_vec(&out.segments).expect("serialize worker transcription segments");
    println!(
        "FW_WORKER_RESULT {}",
        json!({
            "chars": chars,
            "words": transcript.split_whitespace().count(),
            "segments": segments.len(),
            "transcript_sha256": sha256_bytes(transcript.as_bytes()),
            "serialized_segments_sha256": sha256_bytes(&serialized_segments),
            "transcript": transcript,
            "actual_threads": rayon::current_num_threads(),
        })
    );
}

fn report(label: &str, ratios: &[f64]) -> (f64, f64, f64) {
    let med = median(ratios);
    let (lo, hi) = bootstrap_median_ci(ratios);
    println!(
        "{label} median={med:.6} ci95=[{lo:.6},{hi:.6}] cv={:.6} n={}",
        cv(ratios),
        ratios.len()
    );
    (med, lo, hi)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--franken-worker") {
        franken_worker_main(&args);
        return;
    }

    let model_short = args.get(1).map(String::as_str).unwrap_or("tiny.en");
    let wav = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tests/fixtures/native/jfk.wav".to_string());
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(11);
    // 4th positional arg "no_ts" certifies the no-timestamp cell instead. Both
    // engines switch together; a mismatched pair would charge one side for work
    // the other is not doing.
    let timestamps = args.get(4).map(String::as_str) != Some("no_ts");
    assert!(
        rounds >= 3 && rounds % 2 == 1,
        "rounds must be odd and >= 3"
    );
    let scope = BenchScope::from_env();
    let workload = std::env::var("FW_WORKLOAD_NAME").unwrap_or_else(|_| {
        format!(
            "{model_short}-{}",
            if timestamps {
                "segment-timestamps"
            } else {
                "text-only"
            }
        )
    });

    let incumbent = std::env::var("FW_INCUMBENT_BIN")
        .unwrap_or_else(|_| "legacy_whispercpp/whisper.cpp/build/bin/whisper-cli".to_string());
    let incumbent = PathBuf::from(incumbent);
    assert!(
        incumbent.is_file(),
        "incumbent binary not found at {} (set FW_INCUMBENT_BIN)",
        incumbent.display()
    );
    let matched_threads = std::env::var("FW_BENCH_THREADS").ok();
    let enforce_matched_threads = matched_threads.is_some() || scope == BenchScope::WholeJob;
    let threads: usize = matched_threads
        .clone()
        .or_else(|| std::env::var("FW_INCUMBENT_THREADS").ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    assert!(threads >= 1, "thread count must be positive");

    let model_path = find_model_file(model_short)
        .unwrap_or_else(|| panic!("model {model_short} not found in search dirs"));
    let samples = read_wav_mono16k(&wav);
    let params = DecodeParams {
        language: Some("en".to_string()),
        translate: false,
        timestamps,
        n_threads: matched_threads.as_ref().map_or(0, |_| threads),
        max_text_ctx: None,
        word_timestamps: false,
        model_hint: Some(model_short.to_string()),
        ..DecodeParams::default()
    };
    let resident_model = (scope == BenchScope::TranscribeOnly).then(|| {
        GgmlModel::load(&model_path)
            .and_then(LoadedModel::from_ggml)
            .expect("load model")
    });
    let run_franken = || match scope {
        BenchScope::TranscribeOnly => run_franken_resident(
            resident_model.as_ref().expect("resident model"),
            &samples,
            &params,
        ),
        BenchScope::WholeJob => {
            run_franken_whole(&model_path, model_short, &wav, threads, timestamps)
        }
    };
    let run_whisper = || run_incumbent(&incumbent, &model_path, &wav, threads, timestamps, scope);

    println!("harness_elf_sha256={}", executable_identity());
    println!(
        "incumbent_bin_sha256={} {}",
        sha256_file(&incumbent),
        incumbent.display()
    );
    println!(
        "model_sha256={} {}",
        sha256_file(&model_path),
        model_path.display()
    );
    println!("audio_sha256={} {}", sha256_file(Path::new(&wav)), wav);
    println!("INCUMBENT_AB_HOST {}", host_provenance());
    println!(
        "INCUMBENT_AB_CONFIG workload={workload:?} rounds={rounds} order=alternating \
         wav={wav} audio_sec={:.1} requested_threads={threads} timestamps={timestamps} \
         scope={} measured={}",
        samples.len() as f64 / 16000.0,
        scope.as_str(),
        match scope {
            BenchScope::TranscribeOnly => "transcribe_excluding_model_load",
            BenchScope::WholeJob =>
                "process_wall_including_startup_model_audio_io_inference_serialization_teardown",
        },
    );

    // Whole-job evidence additionally requires a quiet host. A steady competing
    // workload can bias the two tools differently while still passing both A/A
    // nulls and a load-split check. Sample every persistent process between arms
    // and gate on the busiest non-harness process's CPU use.
    let excluded_pids = process_ancestry();
    let clock_ticks_per_second = clock_ticks_per_second();
    let mut external_activity = ExternalCpuActivity::default();
    let mut last_process_sample = process_sample();
    let mut checkpoint = |label: &str| {
        let current = process_sample();
        external_activity.observe(
            &last_process_sample,
            &current,
            &excluded_pids,
            clock_ticks_per_second,
            label,
        );
        last_process_sample = current;
    };

    // Warm both engines once; neither warm-up is timed.
    let fw_warm = run_then_checkpoint(&run_franken, &mut checkpoint, "warm_franken");
    let wc_warm = run_then_checkpoint(&run_whisper, &mut checkpoint, "warm_incumbent");
    let wer = word_error_rate(&wc_warm.transcript, &fw_warm.transcript);
    let quality_clear = wer.wer <= MAX_CROSS_ENGINE_WER;
    let thread_clear = !enforce_matched_threads
        || (fw_warm.actual_threads == threads && wc_warm.actual_threads == threads);
    println!(
        "INCUMBENT_AB_COVERAGE fw_chars={} wc_chars={} fw_words={} wc_words={} \
         fw_segments={} wc_segments={} fw_text_sha256={} wc_text_sha256={} \
         wer={:.6} wer_edits={} wer_max={MAX_CROSS_ENGINE_WER:.6} quality_clear={quality_clear}",
        fw_warm.chars,
        wc_warm.chars,
        fw_warm.words,
        wc_warm.words,
        fw_warm.segments,
        wc_warm.segments,
        fw_warm.transcript_sha256,
        wc_warm.transcript_sha256,
        wer.wer,
        wer.edits,
    );
    println!(
        "INCUMBENT_AB_THREADS requested={threads} franken_actual={} \
         incumbent_actual={} franken_load_workers_config={} \
         affinity={} cpuset_effective={} enforce_matched={enforce_matched_threads} \
         thread_clear={thread_clear}",
        fw_warm.actual_threads,
        wc_warm.actual_threads,
        if scope == BenchScope::WholeJob {
            threads.to_string()
        } else {
            std::env::var("FW_LOAD_WORKERS").unwrap_or_else(|_| "default".to_owned())
        },
        proc_status_value("Cpus_allowed_list"),
        std::fs::read_to_string("/sys/fs/cgroup/cpuset.cpus.effective")
            .map(|value| value.trim().to_owned())
            .unwrap_or_else(|_| "unavailable".to_owned()),
    );
    let run_franken_checked = || {
        let observation = run_franken();
        assert_eq!(
            observation.transcript_sha256, fw_warm.transcript_sha256,
            "franken transcript changed within one invocation"
        );
        assert_eq!(
            observation.actual_threads, fw_warm.actual_threads,
            "franken actual thread count changed within one invocation"
        );
        observation
    };
    let run_whisper_checked = || {
        let observation = run_whisper();
        assert_eq!(
            observation.transcript_sha256, wc_warm.transcript_sha256,
            "whisper.cpp transcript changed within one invocation"
        );
        assert_eq!(
            observation.actual_threads, wc_warm.actual_threads,
            "whisper.cpp actual thread count changed within one invocation"
        );
        observation
    };

    let mut compare = Vec::with_capacity(rounds);
    let mut fw_null = Vec::with_capacity(rounds);
    let mut wc_null = Vec::with_capacity(rounds);
    let mut fw_ms = Vec::with_capacity(rounds);
    let mut wc_ms = Vec::with_capacity(rounds);
    let mut pre_round_load1 = Vec::with_capacity(rounds);

    for round in 0..rounds {
        pre_round_load1.push(load_average_1m());
        // Alternate which engine runs first so monotonic drift hits both equally.
        let (fw_a, wc_a) = if round % 2 == 0 {
            let fw = run_then_checkpoint(
                &run_franken_checked,
                &mut checkpoint,
                &format!("round_{round}_franken_a"),
            );
            let wc = run_then_checkpoint(
                &run_whisper_checked,
                &mut checkpoint,
                &format!("round_{round}_incumbent_a"),
            );
            (fw, wc)
        } else {
            let wc = run_then_checkpoint(
                &run_whisper_checked,
                &mut checkpoint,
                &format!("round_{round}_incumbent_a"),
            );
            let fw = run_then_checkpoint(
                &run_franken_checked,
                &mut checkpoint,
                &format!("round_{round}_franken_a"),
            );
            (fw, wc)
        };
        // Second observation of each engine, opposite order: pairs with the
        // first to form each arm's own A/A null inside this same invocation.
        let (fw_b, wc_b) = if round % 2 == 0 {
            let wc = run_then_checkpoint(
                &run_whisper_checked,
                &mut checkpoint,
                &format!("round_{round}_incumbent_b"),
            );
            let fw = run_then_checkpoint(
                &run_franken_checked,
                &mut checkpoint,
                &format!("round_{round}_franken_b"),
            );
            (fw, wc)
        } else {
            let fw = run_then_checkpoint(
                &run_franken_checked,
                &mut checkpoint,
                &format!("round_{round}_franken_b"),
            );
            let wc = run_then_checkpoint(
                &run_whisper_checked,
                &mut checkpoint,
                &format!("round_{round}_incumbent_b"),
            );
            (fw, wc)
        };

        compare.push(wc_a.measured_ms / fw_a.measured_ms);
        fw_null.push(fw_a.measured_ms / fw_b.measured_ms);
        wc_null.push(wc_a.measured_ms / wc_b.measured_ms);
        fw_ms.push(fw_a.measured_ms);
        wc_ms.push(wc_a.measured_ms);
    }
    drop(checkpoint);

    println!(
        "INCUMBENT_AB_TIMES fw_median_ms={:.3} wc_median_ms={:.3}",
        median(&fw_ms),
        median(&wc_ms)
    );
    // Raw per-round series. Interleaving cancels drift *over time*, but it does
    // NOT cancel one engine being more load-sensitive than the other — that bias
    // survives alternation and silently scales the ratio. Emitting the raw series
    // lets a reviewer regress ratio against absolute round cost (a proxy for
    // instantaneous load) and see whether the ratio moves with it.
    println!("INCUMBENT_AB_RAW fw_ms={fw_ms:?}");
    println!("INCUMBENT_AB_RAW wc_ms={wc_ms:?}");
    println!("INCUMBENT_AB_RAW compare={compare:?}");
    println!("INCUMBENT_AB_RAW null_fw={fw_null:?}");
    println!("INCUMBENT_AB_RAW null_wc={wc_null:?}");
    println!("INCUMBENT_AB_RAW pre_round_load1={pre_round_load1:?}");
    let external_host_clear = scope != BenchScope::WholeJob
        || external_activity.max_core_fraction <= MAX_EXTERNAL_CPU_CORE_FRACTION;
    println!(
        "INCUMBENT_AB_EXTERNAL_CPU max_core_fraction={:.6} max_allowed={:.6} \
         pid={} command={:?} checkpoint={:?} intervals={} \
         enforced={} external_host_clear={external_host_clear}",
        external_activity.max_core_fraction,
        MAX_EXTERNAL_CPU_CORE_FRACTION,
        external_activity.pid,
        external_activity.command,
        external_activity.checkpoint,
        external_activity.intervals,
        scope == BenchScope::WholeJob,
    );
    // This is part of the verdict, not commentary: differential load
    // sensitivity can survive order alternation and bias the cross-tool ratio.
    // The gate uses the independent pre-round load sample. Total arm cost is
    // still emitted as a diagnostic, but cannot decide its own ratio.
    let total_cost: Vec<f64> = fw_ms.iter().zip(&wc_ms).map(|(fw, wc)| fw + wc).collect();
    let (cost_light, cost_heavy, cost_gap) =
        split_by_covariate(&total_cost, &compare).expect("odd rounds >= 3 form a cost split");
    println!(
        "INCUMBENT_AB_COST_SPLIT diagnostic_only=true \
         lighter_rounds_median={cost_light:.6} heavier_rounds_median={cost_heavy:.6} \
         n_each={} gap={cost_gap:.6}",
        rounds / 2
    );
    let (light_median, heavy_median, load_split_gap) =
        split_by_covariate(&pre_round_load1, &compare)
            .expect("Linux pre-round load samples form a load split");
    println!(
        "INCUMBENT_AB_LOAD_SPLIT covariate=pre_round_load1 \
         lighter_rounds_median={light_median:.6} \
         heavier_rounds_median={heavy_median:.6} n_each={} gap={load_split_gap:.6} \
         max_gap={MAX_LOAD_SPLIT_GAP:.6}",
        rounds / 2
    );
    let (_, fw_lo, fw_hi) = report("INCUMBENT_AB_NULL_FW", &fw_null);
    let (_, wc_lo, wc_hi) = report("INCUMBENT_AB_NULL_WC", &wc_null);
    let (cmp_med, cmp_lo, cmp_hi) = report("INCUMBENT_AB_COMPARE", &compare);

    // Decidable only if the comparison clears BOTH nulls' worst edge by 2x.
    let fw_half = (fw_hi - 1.0).abs().max((1.0 - fw_lo).abs());
    let wc_half = (wc_hi - 1.0).abs().max((1.0 - wc_lo).abs());
    let required = 1.0 + 2.0 * fw_half.max(wc_half);
    let load_split_clear = load_split_gap <= MAX_LOAD_SPLIT_GAP;
    let verdict = if !quality_clear || !thread_clear || !load_split_clear || !external_host_clear {
        "UNDECIDABLE"
    } else if cmp_med > required && cmp_lo > 1.0 {
        "WIN"
    } else if cmp_med < 1.0 / required && cmp_hi < 1.0 {
        "LOSS"
    } else {
        "UNDECIDABLE"
    };
    println!(
        "INCUMBENT_AB_GATE method=median_vs_both_null_ci95_2x_margin \
         fw_null_half={fw_half:.6} wc_null_half={wc_half:.6} required={required:.6} \
         compare_median={cmp_med:.6} compare_ci95=[{cmp_lo:.6},{cmp_hi:.6}] \
         load_split_gap={load_split_gap:.6} load_split_max={MAX_LOAD_SPLIT_GAP:.6} \
         load_split_clear={load_split_clear} quality_clear={quality_clear} \
         thread_clear={thread_clear} external_host_clear={external_host_clear} \
         cv_is_provenance_only=true class=vs_incumbent verdict={verdict}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariate_split_excludes_middle_round_and_reports_absolute_gap() {
        let load = [1.0, 2.0, 3.0, 4.0, 5.0];
        let compare = [1.1, 1.2, 9.9, 1.3, 1.4];

        let (light, heavy, gap) =
            split_by_covariate(&load, &compare).expect("valid equal-length inputs");

        assert!((light - 1.15).abs() < f64::EPSILON);
        assert!((heavy - 1.35).abs() < f64::EPSILON);
        assert!((gap - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn covariate_split_rejects_mismatched_short_or_non_finite_inputs() {
        assert!(split_by_covariate(&[1.0, 2.0], &[1.0, 2.0]).is_none());
        assert!(split_by_covariate(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_none());
        assert!(split_by_covariate(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn proc_stat_parser_handles_spaces_and_parentheses_in_command() {
        let stat = "123 (bench worker (phase)) R 42 0 0 0 0 0 0 0 0 0 150 25";
        assert_eq!(parse_proc_stat(stat), Some((42, 175)));
        assert_eq!(parse_proc_stat("malformed"), None);
    }

    #[test]
    fn external_cpu_activity_reports_busiest_non_ancestor() {
        let start = Instant::now();
        let previous = ProcessSample {
            captured: start,
            by_pid: BTreeMap::from([
                (
                    7,
                    ProcessCpu {
                        ticks: 100,
                        command: "competitor".to_owned(),
                    },
                ),
                (
                    8,
                    ProcessCpu {
                        ticks: 100,
                        command: "ancestor".to_owned(),
                    },
                ),
            ]),
        };
        let current = ProcessSample {
            captured: start + std::time::Duration::from_secs(2),
            by_pid: BTreeMap::from([
                (
                    7,
                    ProcessCpu {
                        ticks: 170,
                        command: "competitor".to_owned(),
                    },
                ),
                (
                    8,
                    ProcessCpu {
                        ticks: 500,
                        command: "ancestor".to_owned(),
                    },
                ),
            ]),
        };
        let mut activity = ExternalCpuActivity::default();
        activity.observe(
            &previous,
            &current,
            &BTreeSet::from([8]),
            100.0,
            "after_franken",
        );

        assert!((activity.max_core_fraction - 0.35).abs() < f64::EPSILON);
        assert_eq!(activity.pid, 7);
        assert_eq!(activity.command, "competitor");
        assert_eq!(activity.checkpoint, "after_franken");
        assert_eq!(activity.intervals, 1);
    }

    #[test]
    fn incumbent_output_parser_extracts_transcript_and_actual_threads() {
        let output = "\
system_info: n_threads = 64 / 128 | AVX2 = 1\n\
[00:00:00.000 --> 00:00:01.000]  hello world\n\
diagnostic line\n\
[00:00:01.000 --> 00:00:02.000]  from whisper\n";

        assert_eq!(
            incumbent_segments(output, true),
            ["hello world".to_owned(), "from whisper".to_owned()]
        );
        assert_eq!(incumbent_actual_threads(output), Some(64));
        assert_eq!(
            incumbent_segments("\n hello world from whisper\n", false),
            ["hello world from whisper".to_owned()]
        );
    }
}
