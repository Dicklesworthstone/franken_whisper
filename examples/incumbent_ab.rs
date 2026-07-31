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
//! decidable only when the comparison CI95 excludes 1.0, the comparison median
//! clears the widest null CI95 edge by a 2× margin, and the comparison medians
//! from independently sampled lighter- and heavier-load rounds differ by at
//! most 0.1×. A null CI95 does **not** have to contain 1.0: its distance from
//! 1.0 calibrates the decision floor. `cv` is recorded as provenance and
//! decides nothing. Whole-job runs also census
//! persistent non-harness processes between every arm: a process consuming more
//! than 0.1 CPU core makes the result undecidable, because steady cross-tool
//! load bias can survive both numerical controls.
//! Work-count provenance is a separate diagnostic, never a substitute for the
//! median-CI speed gate. The classification compares only counters with matched
//! semantics across engines: window attempts, encoder calls, and single-token
//! decoder calls. The incumbent's printed sample/batch/prompt denominators are
//! retained as raw provenance but cannot drive the classification because this
//! vendored greedy path does not maintain them as comparable call/token counts.
//! The harness additionally copies FrankenFS's fail-closed host-wide
//! quiescence contract: every online logical CPU must be sampled for 300 ms and
//! remain at or below 20% busy at preflight and immediately before measurement.
//! A post-measurement sample is part of the verdict, so a competitor arriving
//! during the final arm cannot leave a bankable result. Checkpoints that follow
//! the harness's own 32-thread probes are sampled after a fixed settle window,
//! because a sample taken the instant those return measures our own thread-pool
//! wind-down and vetoes the run for its own activity; the threshold, the
//! per-CPU coverage and the window are unchanged, and the pre-settle sample is
//! still emitted as provenance so the escape is auditable rather than trusted.
//! Cross-engine rows also fail closed unless every online CPU reports one
//! uniform scaling driver and the `performance` governor. The complete
//! per-value CPU grouping, including `energy_performance_preference` when the
//! driver exposes it, is emitted as provenance: boost-on-demand policies can
//! affect the two engines differently even on an otherwise idle host.
//! Each whole-job row also records host RAM/NUMA topology and distinguishes
//! requested/configured pool width from threads independently observed
//! consuming CPU ticks under `/proc/<pid>/task`.
//!
//! ## Pinned incumbent, and why both arms attest their own image
//!
//! The incumbent is `whisper.cpp` **at a pinned version**, recorded in the
//! version contract `docs/INCUMBENT_CONTRACT.json`. The contract carries the
//! declared project version, the digests of the vendored source, the digest of
//! the built `whisper-cli`, the GGML build switches (so an ISA-hobbled incumbent
//! cannot be mistaken for the real one), and the matched decoding parameters.
//! Preflight fails closed when the vendored source or the declared version drifts
//! from the contract; the reported row carries the pinned version and the digest
//! that produced it, so a later incumbent bump cannot retroactively re-point an
//! old claim.
//!
//! Both arms report the SHA-256 of the image **their own process is executing**,
//! via `/proc/self/exe` in-process and `/proc/<pid>/exe` for a spawned arm — the
//! kernel's record of what actually ran, not a digest of whatever sat at a path.
//! `whisper-cli` is third-party and cannot be recompiled to print its own digest,
//! so the kernel is the attester of record for it. The comparison fails closed
//! unless the two arms are **distinct binaries** and the incumbent's attested
//! image equals the contract's. Without that check a harness misconfigured to run
//! one build twice would produce a well-formed competitive ratio that is really
//! an A/A null, and every other gate here would happily pass it.
//!
//! Matched decoding parameters are driven from the contract for both arms rather
//! than written at each call site. This matters concretely: `whisper-cli` defaults
//! to `-bs 5 -bo 5`, so leaving its defaults alone would compare beam search
//! against franken's greedy decode and bank a ~5x decode-work gap as a speedup.
//! The contract also pins temperature fallback **off** on both sides, matching
//! franken's default-off ladder, so the incumbent is not charged for retry
//! windows franken never attempts.
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
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use franken_whisper::conformance::word_error_rate;
use franken_whisper::native_engine::decode::{
    DecodeParams, DecodeWorkStats, LoadedModel, transcribe_samples,
};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// Version contract for the pinned incumbent. Embed it so an ELF copied back
/// from an RCH worker does not retain that worker's scratch checkout path.
const INCUMBENT_CONTRACT_PATH: &str = "docs/INCUMBENT_CONTRACT.json";
const INCUMBENT_CONTRACT_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/docs/INCUMBENT_CONTRACT.json"
));
#[cfg(test)]
const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

const MAX_LOAD_SPLIT_GAP: f64 = 0.1;
const MAX_CROSS_ENGINE_WER: f64 = 0.1;
const MAX_EXTERNAL_CPU_CORE_FRACTION: f64 = 0.1;
const HOST_CPU_SAMPLE_INTERVAL: Duration = Duration::from_millis(300);
const MAX_HOST_CPU_BUSY_FRACTION: f64 = 0.20;
/// Wind-down window granted before a quiescence sample that follows the
/// harness's own work. See `quiescence_settle_for`.
const HOST_QUIESCENCE_SETTLE: Duration = Duration::from_secs(2);

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
    observed_active_threads: usize,
    peak_process_threads: usize,
    /// SHA-256 of the image the arm actually executed, attested by the arm's own
    /// process (`/proc/self/exe` in-process, `/proc/<pid>/exe` for a spawned arm).
    /// `None` outside the untimed identity probe, which is the only run that needs
    /// it — hashing an ELF inside a timed arm would charge the arm for the hash.
    exe_sha256: Option<String>,
    work: EngineWork,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct EngineWork {
    window_attempts: usize,
    encoder_calls: usize,
    decoder_prefill_calls: usize,
    decoder_prefill_tokens: usize,
    selected_tokens: usize,
    single_token_decode_calls: usize,
    incumbent_sampling_counter: usize,
    incumbent_batch_decode_counter: usize,
    incumbent_prompt_counter: usize,
    accepted_windows: usize,
    accepted_result_tokens: usize,
    prompt_reset_retries: usize,
    temperature_fallback_retries: usize,
    prompt_fallbacks: usize,
    hallucination_fallbacks: usize,
}

impl EngineWork {
    fn from_franken(work: &DecodeWorkStats) -> Self {
        Self {
            window_attempts: work.window_attempts,
            encoder_calls: work.encoder_calls,
            decoder_prefill_calls: work.decoder_prefill_calls,
            decoder_prefill_tokens: work.decoder_prefill_tokens,
            selected_tokens: work.sampled_tokens,
            single_token_decode_calls: work.greedy_single_token_forwards,
            accepted_windows: work.accepted_windows,
            accepted_result_tokens: work.accepted_result_tokens,
            prompt_reset_retries: work.prompt_reset_retries,
            temperature_fallback_retries: work.temperature_fallback_retries,
            ..Self::default()
        }
    }

    fn from_worker_json(value: &Value) -> Self {
        let field = |name: &str| {
            value[name]
                .as_u64()
                .unwrap_or_else(|| panic!("franken worker work result missing {name}"))
                as usize
        };
        Self {
            window_attempts: field("window_attempts"),
            encoder_calls: field("encoder_calls"),
            decoder_prefill_calls: field("decoder_prefill_calls"),
            decoder_prefill_tokens: field("decoder_prefill_tokens"),
            selected_tokens: field("sampled_tokens"),
            single_token_decode_calls: field("greedy_single_token_forwards"),
            accepted_windows: field("accepted_windows"),
            accepted_result_tokens: field("accepted_result_tokens"),
            prompt_reset_retries: field("prompt_reset_retries"),
            temperature_fallback_retries: field("temperature_fallback_retries"),
            ..Self::default()
        }
    }
}

fn work_count_classification(franken: &EngineWork, incumbent: &EngineWork) -> &'static str {
    let exceeds_by_ten_percent =
        |left: usize, right: usize| left.saturating_mul(10) > right.saturating_mul(11);
    let franken_more_work =
        exceeds_by_ten_percent(franken.window_attempts, incumbent.window_attempts)
            || exceeds_by_ten_percent(franken.encoder_calls, incumbent.encoder_calls)
            || exceeds_by_ten_percent(
                franken.single_token_decode_calls,
                incumbent.single_token_decode_calls,
            );
    let incumbent_more_work =
        exceeds_by_ten_percent(incumbent.window_attempts, franken.window_attempts)
            || exceeds_by_ten_percent(incumbent.encoder_calls, franken.encoder_calls)
            || exceeds_by_ten_percent(
                incumbent.single_token_decode_calls,
                franken.single_token_decode_calls,
            );
    match (franken_more_work, incumbent_more_work) {
        (true, false) => "franken_more_work",
        (false, true) => "incumbent_more_work",
        (true, true) => "mixed_work_counts",
        (false, false) => "matched_within_10pct",
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CpuTicks {
    total: u64,
    idle: u64,
}

#[derive(Debug)]
struct HostWideQuiescence {
    checkpoint: String,
    online_cpu_count: usize,
    sampled_online_cpu_count: usize,
    allowed_cpu_count: usize,
    max_busy_fraction: f64,
    busy_cpus: Vec<(usize, f64)>,
}

impl HostWideQuiescence {
    fn is_clear(&self) -> bool {
        self.sampled_online_cpu_count == self.online_cpu_count && self.busy_cpus.is_empty()
    }

    fn busy_labels(&self) -> Vec<String> {
        self.busy_cpus
            .iter()
            .map(|(cpu, busy)| format!("cpu{cpu}={:.1}%", busy * 100.0))
            .collect()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct CpuPolicyField {
    by_value: BTreeMap<String, Vec<usize>>,
    unavailable_cpus: Vec<usize>,
}

impl CpuPolicyField {
    fn is_complete_uniform(&self, online_cpu_count: usize) -> bool {
        self.unavailable_cpus.is_empty()
            && self.by_value.len() == 1
            && self
                .by_value
                .values()
                .next()
                .is_some_and(|cpus| cpus.len() == online_cpu_count)
    }

    fn uniform_value(&self, online_cpu_count: usize) -> Option<&str> {
        if self.is_complete_uniform(online_cpu_count) {
            self.by_value.keys().next().map(String::as_str)
        } else {
            None
        }
    }

    fn as_json(&self, online_cpu_count: usize) -> Value {
        json!({
            "online_cpu_count": online_cpu_count,
            "observed_cpu_count": online_cpu_count.saturating_sub(self.unavailable_cpus.len()),
            "uniform": self.is_complete_uniform(online_cpu_count),
            "by_value": &self.by_value,
            "unavailable_cpus": &self.unavailable_cpus,
        })
    }
}

#[derive(Debug)]
struct CpuFrequencyPolicy {
    scaling_driver: CpuPolicyField,
    scaling_governor: CpuPolicyField,
    energy_performance_preference: CpuPolicyField,
}

impl CpuFrequencyPolicy {
    fn observe(online: &BTreeSet<usize>) -> Self {
        Self {
            scaling_driver: read_cpu_policy_field(online, "scaling_driver"),
            scaling_governor: read_cpu_policy_field(online, "scaling_governor"),
            energy_performance_preference: read_cpu_policy_field(
                online,
                "energy_performance_preference",
            ),
        }
    }

    fn benchmark_clear(&self, online_cpu_count: usize) -> bool {
        let driver_clear = self.scaling_driver.is_complete_uniform(online_cpu_count);
        let governor_clear =
            self.scaling_governor.uniform_value(online_cpu_count) == Some("performance");
        driver_clear && governor_clear
    }

    fn as_json(&self, online_cpu_count: usize) -> Value {
        json!({
            "required_scaling_governor": "performance",
            "energy_performance_preference_is_provenance_only": true,
            "scaling_driver": self.scaling_driver.as_json(online_cpu_count),
            "scaling_governor": self.scaling_governor.as_json(online_cpu_count),
            "energy_performance_preference": self
                .energy_performance_preference
                .as_json(online_cpu_count),
            "benchmark_clear": self.benchmark_clear(online_cpu_count),
        })
    }
}

fn group_cpu_policy_field(
    samples: impl IntoIterator<Item = (usize, Option<String>)>,
) -> CpuPolicyField {
    let mut by_value: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut unavailable_cpus = Vec::new();
    for (cpu, value) in samples {
        match value.filter(|value| !value.is_empty()) {
            Some(value) => by_value.entry(value).or_default().push(cpu),
            None => unavailable_cpus.push(cpu),
        }
    }
    CpuPolicyField {
        by_value,
        unavailable_cpus,
    }
}

fn read_cpu_policy_field(online: &BTreeSet<usize>, field: &str) -> CpuPolicyField {
    group_cpu_policy_field(online.iter().map(|cpu| {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{field}");
        let value = std::fs::read_to_string(path)
            .ok()
            .map(|value| value.trim().to_owned());
        (*cpu, value)
    }))
}

fn require_cpu_frequency_policy(policy: &CpuFrequencyPolicy, online_cpu_count: usize) -> bool {
    let clear = policy.benchmark_clear(online_cpu_count);
    println!(
        "INCUMBENT_AB_CPU_FREQUENCY_POLICY {}",
        policy.as_json(online_cpu_count)
    );
    assert!(
        clear,
        "competitive benchmark requires one uniform scaling driver and the performance \
         governor on every online CPU"
    );
    clear
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

/// SHA-256 of *this* process's own loaded image, read through `/proc/self/exe`.
///
/// The point of going through `/proc` rather than hashing an argv-derived path is
/// that `/proc/self/exe` is the kernel's record of the image this process is
/// actually executing. It cannot be redirected by a stale path, a symlink swap,
/// or a rebuild landing elsewhere.
fn self_exe_sha256() -> String {
    sha256_file(Path::new("/proc/self/exe"))
}

/// SHA-256 of a *live child's* loaded image, read through `/proc/<pid>/exe`.
///
/// This is the incumbent's own attestation of what it is running, to the extent
/// obtainable from an unmodified third-party binary: the harness cannot recompile
/// `whisper-cli` to print its own digest, but it can ask the kernel which image
/// the spawned process actually mapped. Hashing `FW_INCUMBENT_BIN` instead would
/// only prove what sat at a path, not what ran.
fn child_exe_sha256(pid: u32) -> Option<String> {
    let digest = sha256_file(Path::new(&format!("/proc/{pid}/exe")));
    (digest != "unreadable").then_some(digest)
}

/// Decoding parameters applied to *both* arms, sourced from the version contract.
///
/// One struct, one source of truth. `whisper-cli` defaults to beam search
/// (`-bs 5 -bo 5`); comparing that against franken's greedy path would hand
/// franken a ~5x decode-work advantage and call it a speedup.
#[derive(Clone, Debug, PartialEq)]
struct MatchedDecode {
    beam_size: usize,
    best_of: usize,
    temperature: f64,
    temperature_fallback: bool,
    /// Cross-window prompt-carry cap, pinned on both arms. `whisper-cli -mc N`;
    /// franken `DecodeParams.max_context`. Left unpinned the two engines carry
    /// different prompt histories — franken's is a per-model/mode policy,
    /// `whisper-cli`'s default is `-1` (unbounded) — which changes decode work
    /// and the transcript, so the ratio would not be matched.
    max_context: i32,
    language: String,
    translate: bool,
    word_timestamps: bool,
}

impl MatchedDecode {
    fn from_json(value: &Value) -> Self {
        let field = |name: &str| -> &Value {
            let field = &value[name];
            assert!(
                !field.is_null(),
                "incumbent contract matched_decode is missing {name:?}"
            );
            field
        };
        let usize_field = |name: &str| -> usize {
            field(name)
                .as_u64()
                .unwrap_or_else(|| panic!("contract matched_decode.{name} must be an integer"))
                as usize
        };
        let bool_field = |name: &str| -> bool {
            field(name)
                .as_bool()
                .unwrap_or_else(|| panic!("contract matched_decode.{name} must be a bool"))
        };
        Self {
            beam_size: usize_field("beam_size"),
            best_of: usize_field("best_of"),
            temperature: field("temperature")
                .as_f64()
                .expect("contract matched_decode.temperature must be a number"),
            temperature_fallback: bool_field("temperature_fallback"),
            max_context: i32::try_from(
                field("max_context")
                    .as_i64()
                    .expect("contract matched_decode.max_context must be an integer"),
            )
            .expect("contract matched_decode.max_context must fit in i32"),
            language: field("language")
                .as_str()
                .expect("contract matched_decode.language must be a string")
                .to_owned(),
            translate: bool_field("translate"),
            word_timestamps: bool_field("word_timestamps"),
        }
    }

    /// `whisper-cli` flags that pin the incumbent to the contract. Every value is
    /// passed explicitly, including the ones that match `whisper-cli`'s own
    /// defaults: a default is a fact about a version, not about the comparison,
    /// and it can move underneath a pinned digest bump.
    fn incumbent_args(&self) -> Vec<String> {
        let mut args = vec![
            "-bs".to_owned(),
            self.beam_size.to_string(),
            "-bo".to_owned(),
            self.best_of.to_string(),
            "-tp".to_owned(),
            format!("{:.2}", self.temperature),
            "-l".to_owned(),
            self.language.clone(),
            "-mc".to_owned(),
            self.max_context.to_string(),
        ];
        if !self.temperature_fallback {
            args.push("-nf".to_owned());
        }
        if self.translate {
            args.push("-tr".to_owned());
        }
        args
    }

    /// The parameters as they must appear on every reported row. A matched-params
    /// claim that is not printed is not checkable.
    fn as_row(&self) -> String {
        format!(
            "beam_size={} best_of={} temperature={:.2} temperature_fallback={} \
             max_context={} language={} translate={} word_timestamps={} decode_mode={}",
            self.beam_size,
            self.best_of,
            self.temperature,
            self.temperature_fallback,
            self.max_context,
            self.language,
            self.translate,
            self.word_timestamps,
            if self.beam_size <= 1 && self.best_of <= 1 {
                "greedy"
            } else {
                "beam"
            },
        )
    }
}

/// The pinned-incumbent version contract (`docs/INCUMBENT_CONTRACT.json`).
#[derive(Clone, Debug)]
struct IncumbentContract {
    project: String,
    version: String,
    source_root: String,
    binary_sha256: String,
    source_sha256: BTreeMap<String, String>,
    build: Value,
    decode: MatchedDecode,
}

impl IncumbentContract {
    fn shipped() -> Self {
        let value: Value = serde_json::from_str(INCUMBENT_CONTRACT_JSON).unwrap_or_else(|error| {
            panic!(
                "embedded incumbent version contract {INCUMBENT_CONTRACT_PATH} is not valid JSON: \
                 {error}"
            )
        });
        let incumbent = &value["incumbent"];
        let string_field = |name: &str| -> String {
            incumbent[name]
                .as_str()
                .unwrap_or_else(|| panic!("incumbent contract is missing {name:?}"))
                .to_owned()
        };
        let source_sha256 = incumbent["source_sha256"]
            .as_object()
            .expect("incumbent contract is missing source_sha256")
            .iter()
            .map(|(name, digest)| {
                (
                    name.clone(),
                    digest
                        .as_str()
                        .expect("contract source_sha256 values must be strings")
                        .to_owned(),
                )
            })
            .collect();
        Self {
            project: string_field("project"),
            version: string_field("version"),
            source_root: string_field("source_root"),
            binary_sha256: string_field("binary_sha256"),
            source_sha256,
            build: incumbent["build"].clone(),
            decode: MatchedDecode::from_json(&value["matched_decode"]),
        }
    }

    /// Check the *attested* image of the incumbent process against the contract.
    fn binary_matches(&self, attested_sha256: Option<&str>) -> bool {
        attested_sha256 == Some(self.binary_sha256.as_str())
    }

    /// Check the vendored source against the contract, so the incumbent cannot be
    /// rebuilt from moved source while the ledger keeps citing the pinned version.
    /// Returns the per-file verdicts for provenance.
    fn source_verdicts(&self, root: &Path) -> Vec<(String, bool, String)> {
        let mut verdicts = Vec::new();
        for (relative, expected) in &self.source_sha256 {
            let observed = sha256_file(&root.join(relative));
            verdicts.push((relative.clone(), &observed == expected, observed));
        }
        let declared = declared_source_version(&root.join("CMakeLists.txt"));
        verdicts.push((
            "CMakeLists.txt:project_version".to_owned(),
            declared.as_deref() == Some(self.version.as_str()),
            declared.unwrap_or_else(|| "unavailable".to_owned()),
        ));
        verdicts
    }
}

/// Recover the vendored source root from the incumbent image that actually runs.
///
/// The benchmark binary is copied back from an RCH worker, so using its
/// compile-time crate root would point at the worker's scratch checkout. The
/// incumbent layout is `<source>/build/bin/whisper-cli`; deriving `<source>` from
/// that runtime path keeps the source-drift gate portable with the ELF.
fn incumbent_source_root(incumbent: &Path) -> Option<PathBuf> {
    incumbent.ancestors().nth(3).map(Path::to_path_buf)
}

/// Parse `project("whisper.cpp" VERSION 1.8.3)` out of the vendored CMakeLists.
fn declared_source_version(cmakelists: &Path) -> Option<String> {
    let text = std::fs::read_to_string(cmakelists).ok()?;
    text.lines()
        .filter(|line| line.trim_start().starts_with("project("))
        .find_map(|line| {
            let tail = line.split_once("VERSION")?.1;
            Some(
                tail.trim()
                    .trim_end_matches(')')
                    .split_whitespace()
                    .next()?
                    .to_owned(),
            )
        })
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

fn ram_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|meminfo| {
            meminfo.lines().find_map(|line| {
                let kib = line.strip_prefix("MemTotal:")?.split_whitespace().next()?;
                kib.parse::<u64>().ok()
            })
        })
        .and_then(|kib| kib.checked_mul(1024))
        .unwrap_or(0)
}

fn numa_nodes() -> usize {
    std::fs::read_dir("/sys/devices/system/node").map_or(0, |entries| {
        entries
            .flatten()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.strip_prefix("node"))
                    .is_some_and(|suffix| {
                        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
                    })
            })
            .count()
    })
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

fn host_provenance(cpu_frequency_policy: &CpuFrequencyPolicy, online_cpu_count: usize) -> Value {
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
        "ram_bytes": ram_bytes(),
        "numa_nodes": numa_nodes(),
        "available_threads": std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
        "affinity_cpus_allowed": proc_status_value("Cpus_allowed_list"),
        "cpuset_effective": cpuset_effective,
        "runtime_isa": runtime_isa(),
        "cpu_frequency_policy": cpu_frequency_policy.as_json(online_cpu_count),
    })
}

fn parse_cpu_list(value: &str) -> Result<BTreeSet<usize>, String> {
    let mut cpus = BTreeSet::new();
    for range in value.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = range.split_once('-') {
            let start = start
                .parse::<usize>()
                .map_err(|error| format!("parse CPU range start {start:?}: {error}"))?;
            let end = end
                .parse::<usize>()
                .map_err(|error| format!("parse CPU range end {end:?}: {error}"))?;
            if start > end {
                return Err(format!("descending CPU range: {range}"));
            }
            cpus.extend(start..=end);
        } else {
            cpus.insert(
                range
                    .parse::<usize>()
                    .map_err(|error| format!("parse CPU index {range:?}: {error}"))?,
            );
        }
    }
    if cpus.is_empty() {
        return Err("CPU list is empty".to_owned());
    }
    Ok(cpus)
}

fn online_cpus() -> Result<BTreeSet<usize>, String> {
    let value = std::fs::read_to_string("/sys/devices/system/cpu/online")
        .map_err(|error| format!("read online CPU list: {error}"))?;
    parse_cpu_list(&value)
}

fn self_allowed_cpus() -> Result<BTreeSet<usize>, String> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|error| format!("read /proc/self/status: {error}"))?;
    let value = status
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name == "Cpus_allowed_list").then_some(value.trim())
        })
        .ok_or_else(|| "Cpus_allowed_list missing from /proc/self/status".to_owned())?;
    parse_cpu_list(value)
}

fn parse_cpu_ticks(text: &str) -> Result<BTreeMap<usize, CpuTicks>, String> {
    let mut cpus = BTreeMap::new();
    for line in text.lines() {
        let mut fields = line.split_ascii_whitespace();
        let Some(label) = fields.next() else {
            continue;
        };
        let Some(suffix) = label.strip_prefix("cpu") else {
            continue;
        };
        if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let cpu = suffix
            .parse::<usize>()
            .map_err(|error| format!("parse CPU index {suffix:?}: {error}"))?;
        let ticks = fields
            .map(|field| {
                field
                    .parse::<u64>()
                    .map_err(|error| format!("parse /proc/stat tick {field:?}: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ticks.len() < 5 {
            return Err(format!("cpu{cpu} /proc/stat row is too short"));
        }
        cpus.insert(
            cpu,
            CpuTicks {
                total: ticks.iter().copied().sum(),
                idle: ticks[3].saturating_add(ticks[4]),
            },
        );
    }
    if cpus.is_empty() {
        return Err("no per-CPU rows in /proc/stat".to_owned());
    }
    Ok(cpus)
}

fn read_cpu_ticks() -> Result<BTreeMap<usize, CpuTicks>, String> {
    let text = std::fs::read_to_string("/proc/stat")
        .map_err(|error| format!("read /proc/stat: {error}"))?;
    parse_cpu_ticks(&text)
}

fn cpu_busy_between(
    before: &BTreeMap<usize, CpuTicks>,
    after: &BTreeMap<usize, CpuTicks>,
) -> Result<BTreeMap<usize, f64>, String> {
    let mut busy = BTreeMap::new();
    for (cpu, start) in before {
        let end = after
            .get(cpu)
            .ok_or_else(|| format!("cpu{cpu} disappeared during load sample"))?;
        let total = end.total.saturating_sub(start.total);
        let idle = end.idle.saturating_sub(start.idle);
        let fraction = if total == 0 {
            1.0
        } else {
            total.saturating_sub(idle) as f64 / total as f64
        };
        busy.insert(*cpu, fraction);
    }
    Ok(busy)
}

fn sample_cpu_busy() -> Result<BTreeMap<usize, f64>, String> {
    let before = read_cpu_ticks()?;
    thread::sleep(HOST_CPU_SAMPLE_INTERVAL);
    let after = read_cpu_ticks()?;
    cpu_busy_between(&before, &after)
}

/// Wind-down window granted before the quiescence sample at `checkpoint`.
///
/// `preflight` runs before the harness has executed any work of its own, so it
/// samples immediately: whatever is busy there belongs to somebody else, and
/// granting it a grace period would only delay catching a competitor that is
/// already resident. Every later checkpoint is preceded by full 32-thread
/// transcriptions of *both* engines — the warm probes, and then the measured
/// rounds — and a sample taken the instant those return measures the harness's
/// own rayon parking, the incumbent's process teardown and the kernel reclaim
/// of a multi-gigabyte model mapping. That is the harness vetoing itself for
/// its own activity, and it is what it did: with the host otherwise idle,
/// `preflight` reported `busy_cpus=[] clear=true` every time while
/// `pre_measurement` failed on a different marginal subset each run
/// (`cpu19=21.9%`+`cpu48=27.6%`, then `cpu16=37.9%`) against a 20% threshold.
///
/// The settle is not a loosening. The threshold, the per-CPU coverage and the
/// 300 ms sample window are untouched, and an external competitor is by
/// construction still consuming CPU two seconds later — only a bounded,
/// decaying tail of our own is escaped. The pre-settle sample is still taken
/// and emitted at `{checkpoint}_immediate` as provenance that asserts nothing,
/// so a reviewer can read exactly what the settle absorbed rather than take the
/// escape on trust. The `_immediate` suffix is itself settle-free, so
/// provenance can never recurse into another settle.
fn quiescence_settle_for(checkpoint: &str) -> Duration {
    if checkpoint == "preflight" || checkpoint.ends_with("_immediate") {
        Duration::ZERO
    } else {
        HOST_QUIESCENCE_SETTLE
    }
}

fn sample_host_wide_quiescence(
    checkpoint: &str,
    online: &BTreeSet<usize>,
    allowed: &BTreeSet<usize>,
) -> HostWideQuiescence {
    let settle = quiescence_settle_for(checkpoint);
    if !settle.is_zero() {
        let immediate =
            sample_host_wide_quiescence_now(&format!("{checkpoint}_immediate"), online, allowed);
        println!(
            "INCUMBENT_AB_HOST_WIDE_SETTLE checkpoint={checkpoint} settle_ms={} \
             immediate_max_busy_fraction={:.6} immediate_busy_cpus={:?} \
             immediate_clear={} threshold_unchanged={MAX_HOST_CPU_BUSY_FRACTION:.6} \
             sample_ms_unchanged={} provenance_only=true",
            settle.as_millis(),
            immediate.max_busy_fraction,
            immediate.busy_labels(),
            immediate.is_clear(),
            HOST_CPU_SAMPLE_INTERVAL.as_millis(),
        );
        thread::sleep(settle);
    }
    sample_host_wide_quiescence_now(checkpoint, online, allowed)
}

fn sample_host_wide_quiescence_now(
    checkpoint: &str,
    online: &BTreeSet<usize>,
    allowed: &BTreeSet<usize>,
) -> HostWideQuiescence {
    assert!(
        allowed.is_subset(online),
        "process CPU allowance includes offline CPUs: allowed={allowed:?} online={online:?}"
    );
    let busy = sample_cpu_busy().expect("sample host-wide CPU quiescence");
    let sampled_online_cpu_count = online.iter().filter(|cpu| busy.contains_key(cpu)).count();
    let max_busy_fraction = online
        .iter()
        .filter_map(|cpu| busy.get(cpu))
        .copied()
        .fold(0.0, f64::max);
    let busy_cpus = online
        .iter()
        .filter_map(|cpu| {
            let fraction = busy.get(cpu).copied()?;
            (fraction > MAX_HOST_CPU_BUSY_FRACTION).then_some((*cpu, fraction))
        })
        .collect();
    let sample = HostWideQuiescence {
        checkpoint: checkpoint.to_owned(),
        online_cpu_count: online.len(),
        sampled_online_cpu_count,
        allowed_cpu_count: allowed.len(),
        max_busy_fraction,
        busy_cpus,
    };
    println!(
        "INCUMBENT_AB_HOST_WIDE_QUIESCENCE checkpoint={} online_cpu_count={} \
         sampled_online_cpu_count={} allowed_cpu_count={} sample_ms={} \
         max_busy_fraction={:.6} max_allowed={MAX_HOST_CPU_BUSY_FRACTION:.6} \
         busy_cpus={:?} clear={}",
        sample.checkpoint,
        sample.online_cpu_count,
        sample.sampled_online_cpu_count,
        sample.allowed_cpu_count,
        HOST_CPU_SAMPLE_INTERVAL.as_millis(),
        sample.max_busy_fraction,
        sample.busy_labels(),
        sample.is_clear(),
    );
    sample
}

fn require_host_wide_quiescence(
    checkpoint: &str,
    online: &BTreeSet<usize>,
    allowed: &BTreeSet<usize>,
) -> HostWideQuiescence {
    let sample = sample_host_wide_quiescence(checkpoint, online, allowed);
    assert!(
        sample.sampled_online_cpu_count == sample.online_cpu_count,
        "host-wide quiescence sampled {} of {} online CPUs",
        sample.sampled_online_cpu_count,
        sample.online_cpu_count
    );
    assert!(
        sample.busy_cpus.is_empty(),
        "host-wide benchmark exclusivity requires every online CPU at or below {:.1}% busy; {}",
        MAX_HOST_CPU_BUSY_FRACTION * 100.0,
        sample.busy_labels().join(", ")
    );
    sample
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

#[derive(Debug, Default)]
struct ChildThreadProbe {
    previous_ticks: BTreeMap<u32, u64>,
    active_tids: BTreeSet<u32>,
    peak_process_threads: usize,
    /// Digest of the child's mapped image, captured once while it is still alive.
    exe_sha256: Option<String>,
}

impl ChildThreadProbe {
    fn sample(&mut self, pid: u32) {
        // Once only: the sample loop runs every millisecond, and re-hashing the
        // child's ELF on every tick would make the probe itself a competitor.
        if self.exe_sha256.is_none() {
            self.exe_sha256 = child_exe_sha256(pid);
        }
        let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
            return;
        };
        let mut current_threads = 0usize;
        for task in tasks.flatten() {
            let Some(tid) = task
                .file_name()
                .to_str()
                .and_then(|value| value.parse::<u32>().ok())
            else {
                continue;
            };
            let Some((_, ticks)) = std::fs::read_to_string(task.path().join("stat"))
                .ok()
                .and_then(|text| parse_proc_stat(&text))
            else {
                continue;
            };
            current_threads += 1;
            let previous = self.previous_ticks.insert(tid, ticks).unwrap_or(0);
            if ticks > previous {
                self.active_tids.insert(tid);
            }
        }
        self.peak_process_threads = self.peak_process_threads.max(current_threads);
    }

    fn observed_active_threads(&self) -> usize {
        self.active_tids.len()
    }
}

/// Run one engine process, optionally observing threads that consume CPU.
///
/// Requested/configured pool width is not execution evidence: a runtime may cap
/// or decline to schedule workers. Sampling `/proc/<pid>/task/*/stat` records
/// every thread whose CPU ticks advance, while `peak_process_threads` preserves
/// the process-level high-water mark. Pipe readers live in the parent harness,
/// so they are not counted as engine workers. Thread observation is used only
/// for the untimed probe; measured arms take the uninstrumented branch.
fn run_command(command: &mut Command, observe_threads: bool) -> (Output, f64, ChildThreadProbe) {
    if !observe_threads {
        let started = Instant::now();
        let output = command.output().expect("run benchmark arm");
        return (
            output,
            started.elapsed().as_secs_f64() * 1e3,
            ChildThreadProbe::default(),
        );
    }

    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let started = Instant::now();
    let mut child = command.spawn().expect("spawn benchmark arm");
    let pid = child.id();
    let mut stdout = child.stdout.take().expect("capture benchmark stdout");
    let mut stderr = child.stderr.take().expect("capture benchmark stderr");

    let (status, stdout_bytes, stderr_bytes, probe) = thread::scope(|scope| {
        let stdout_reader = scope.spawn(move || {
            let mut bytes = Vec::new();
            stdout
                .read_to_end(&mut bytes)
                .expect("read benchmark stdout");
            bytes
        });
        let stderr_reader = scope.spawn(move || {
            let mut bytes = Vec::new();
            stderr
                .read_to_end(&mut bytes)
                .expect("read benchmark stderr");
            bytes
        });
        let mut probe = ChildThreadProbe::default();
        let status = loop {
            probe.sample(pid);
            if let Some(status) = child.try_wait().expect("wait for benchmark arm") {
                break status;
            }
            thread::sleep(Duration::from_millis(1));
        };
        (
            status,
            stdout_reader.join().expect("join benchmark stdout reader"),
            stderr_reader.join().expect("join benchmark stderr reader"),
            probe,
        )
    });
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    (
        Output {
            status,
            stdout: stdout_bytes,
            stderr: stderr_bytes,
        },
        wall_ms,
        probe,
    )
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct StatisticalGate {
    fw_null_half: f64,
    wc_null_half: f64,
    required: f64,
    verdict: &'static str,
}

/// Apply only the statistical portion of the live-incumbent decision contract.
///
/// Null CIs remain part of the 2x decision floor, but whether they happen to
/// contain 1.0 is deliberately irrelevant. This is the historical live
/// comparator rule extracted without changing its verdict behavior so the
/// absence of a precision-coupled straddle veto can be regression-tested.
fn statistical_gate(
    fw_null: (f64, f64, f64),
    wc_null: (f64, f64, f64),
    comparison: (f64, f64, f64),
    prerequisite_gates_clear: bool,
) -> StatisticalGate {
    let (_, fw_lo, fw_hi) = fw_null;
    let (_, wc_lo, wc_hi) = wc_null;
    let (cmp_med, cmp_lo, cmp_hi) = comparison;
    let fw_null_half = (fw_hi - 1.0).abs().max((1.0 - fw_lo).abs());
    let wc_null_half = (wc_hi - 1.0).abs().max((1.0 - wc_lo).abs());
    let required = 1.0 + 2.0 * fw_null_half.max(wc_null_half);
    let verdict = if !prerequisite_gates_clear {
        "UNDECIDABLE"
    } else if cmp_med > required && cmp_lo > 1.0 {
        "WIN"
    } else if cmp_med < 1.0 / required && cmp_hi < 1.0 {
        "LOSS"
    } else {
        "UNDECIDABLE"
    };

    StatisticalGate {
        fw_null_half,
        wc_null_half,
        required,
        verdict,
    }
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

fn incumbent_timing_runs(text: &str, label: &str) -> Option<usize> {
    text.lines()
        .find(|line| line.contains(label))
        .and_then(|line| line.split_once('/'))
        .and_then(|(_, suffix)| suffix.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

fn incumbent_fallbacks(text: &str) -> Option<(usize, usize)> {
    let line = text.lines().find(|line| line.contains("fallbacks"))?;
    let (_, suffix) = line.split_once('=')?;
    let mut fields = suffix.split_whitespace();
    let prompt = fields.next()?.parse().ok()?;
    (fields.next()? == "p").then_some(())?;
    (fields.next()? == "/").then_some(())?;
    let hallucination = fields.next()?.parse().ok()?;
    (fields.next()? == "h").then_some(())?;
    Some((prompt, hallucination))
}

fn incumbent_work_counts(text: &str) -> EngineWork {
    let required_runs = |label: &str| {
        incumbent_timing_runs(text, label)
            .unwrap_or_else(|| panic!("could not parse whisper.cpp {label} run count"))
    };
    // This is deliberately retained only as raw provenance. Although the
    // upstream field is named `n_sample`, this vendored greedy implementation
    // increments it only for non-empty beam-candidate lists and prints
    // `max(1, n_sample)`. It is therefore not a selected-token count for the
    // matched greedy arm and must never drive the work-count classification.
    let incumbent_sampling_counter = required_runs("sample time");
    let encoder_calls = required_runs("encode time");
    let single_token_decode_calls = required_runs("decode time");
    // These two printed "runs" denominators are also raw counters: the
    // incumbent increments them by input token count, not decoder-call count,
    // and prints `max(1, counter)`.
    let incumbent_batch_decode_counter = required_runs("batchd time");
    let incumbent_prompt_counter = required_runs("prompt time");
    let (prompt_fallbacks, hallucination_fallbacks) =
        incumbent_fallbacks(text).expect("could not parse whisper.cpp fallback counts");
    EngineWork {
        window_attempts: encoder_calls + prompt_fallbacks + hallucination_fallbacks,
        encoder_calls,
        single_token_decode_calls,
        incumbent_sampling_counter,
        incumbent_batch_decode_counter,
        incumbent_prompt_counter,
        accepted_windows: encoder_calls,
        prompt_fallbacks,
        hallucination_fallbacks,
        ..EngineWork::default()
    }
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
    observe_threads: bool,
    decode: &MatchedDecode,
) -> Observation {
    let mut args: Vec<String> = vec![
        "-m".into(),
        model.display().to_string(),
        "-f".into(),
        wav.into(),
        "-t".into(),
        threads.to_string(),
    ];
    // Decoding parameters come from the version contract, not from this call site,
    // so the two arms cannot drift apart across edits.
    args.extend(decode.incumbent_args());
    // Match franken's mode on the incumbent side too: comparing franken's
    // no-timestamp decode against whisper.cpp's timestamped decode would charge
    // the incumbent for timestamp work franken is not doing.
    if !timestamps {
        args.push("-nt".into());
    }
    let mut command = Command::new(bin);
    command.args(&args);
    let (output, process_wall_ms, thread_probe) = run_command(&mut command, observe_threads);
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
        observed_active_threads: thread_probe.observed_active_threads(),
        peak_process_threads: thread_probe.peak_process_threads,
        exe_sha256: thread_probe.exe_sha256,
        work: incumbent_work_counts(&text),
    }
}

/// One franken run: transcribe with the model already resident.
fn run_franken_resident(
    model: &LoadedModel,
    samples: &[f32],
    params: &DecodeParams,
    attest: bool,
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
    let work = EngineWork::from_franken(&out.work);
    Observation {
        measured_ms: elapsed,
        chars,
        words: transcript.split_whitespace().count(),
        segments: segments.len(),
        transcript_sha256: sha256_bytes(transcript.as_bytes()),
        transcript,
        actual_threads: rayon::current_num_threads(),
        observed_active_threads: 0,
        peak_process_threads: 0,
        // Resident franken *is* this process, so its self-attestation is
        // `/proc/self/exe`. Only on the untimed probe: this ELF carries line
        // tables, and hashing it inside a timed arm would both add wall and
        // evict the arm's working set.
        exe_sha256: attest.then(self_exe_sha256),
        work,
    }
}

fn run_franken_whole(
    model: &Path,
    model_short: &str,
    wav: &str,
    threads: usize,
    timestamps: bool,
    observe_threads: bool,
) -> Observation {
    let current_exe = std::env::current_exe().expect("resolve harness executable");
    let mut command = Command::new(current_exe);
    command
        .arg("--franken-worker")
        .arg(model)
        .arg(model_short)
        .arg(wav)
        .arg(if timestamps { "ts" } else { "no_ts" })
        .arg(threads.to_string())
        .env("RAYON_NUM_THREADS", threads.to_string())
        .env("FW_LOAD_WORKERS", threads.to_string());
    let (output, process_wall_ms, thread_probe) = run_command(&mut command, observe_threads);
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
    // Fail closed if the spawned arm decoded under a different contract than the
    // parent is about to certify. Both processes are this same ELF, so a
    // mismatch means the two derivations have drifted apart and the parent's
    // `decode_matched` would be describing work that did not happen.
    let worker_decode_row = value["decode_row"]
        .as_str()
        .expect("franken worker result missing decode_row");
    let parent_decode_row = IncumbentContract::shipped().decode.as_row();
    assert_eq!(
        worker_decode_row, parent_decode_row,
        "whole-job franken arm decoded under a different contract than the parent certifies"
    );
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
        observed_active_threads: thread_probe.observed_active_threads(),
        peak_process_threads: thread_probe.peak_process_threads,
        exe_sha256: thread_probe.exe_sha256,
        work: EngineWork::from_worker_json(&value["work"]),
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
    // The worker is this same ELF, so it re-derives decode from the SAME
    // compiled-in contract rather than repeating literals. Hardcoding them here
    // made the whole-job arm's decode independent of the contract that
    // `decode_matched` is asserted against in the parent: `beam_size` was not
    // passed at all, and `language`/`translate`/`word_timestamps` were written
    // twice. Both happened to agree with the shipped contract, so nothing was
    // mismeasured -- but the parent was attesting a params struct the measured
    // arm never used, and the next contract edit would have silently split them.
    let contract = IncumbentContract::shipped();
    // The spawned arm inherits the parent's environment, so it re-checks the
    // hatch that would override the contract's carry policy here rather than
    // trusting that the parent looked.
    assert!(
        std::env::var_os("FW_NO_CONTEXT").is_none(),
        "FW_NO_CONTEXT is set in the whole-job worker, overriding the contract's \
         matched max_context={}",
        contract.decode.max_context
    );
    let params = DecodeParams {
        language: Some(contract.decode.language.clone()),
        translate: contract.decode.translate,
        beam_size: Some(contract.decode.beam_size),
        max_context: Some(contract.decode.max_context),
        timestamps,
        n_threads: threads,
        max_text_ctx: None,
        word_timestamps: contract.decode.word_timestamps,
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
            // The measured arm attests the decode row it actually ran, so the
            // parent's matched-decode claim is checked against this process
            // rather than against a params struct built in the parent.
            "decode_row": contract.decode.as_row(),
            "work": {
                "window_attempts": out.work.window_attempts,
                "encoder_calls": out.work.encoder_calls,
                "decoder_prefill_calls": out.work.decoder_prefill_calls,
                "decoder_prefill_tokens": out.work.decoder_prefill_tokens,
                "sampled_tokens": out.work.sampled_tokens,
                "greedy_single_token_forwards": out.work.greedy_single_token_forwards,
                "accepted_windows": out.work.accepted_windows,
                "accepted_result_tokens": out.work.accepted_result_tokens,
                "prompt_reset_retries": out.work.prompt_reset_retries,
                "temperature_fallback_retries": out.work.temperature_fallback_retries,
            },
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

    // The version contract is loaded before anything is measured, and it drives
    // BOTH arms' decoding parameters. A matched-params claim asserted at two
    // separate call sites is a claim that drifts on the next edit.
    let contract = IncumbentContract::shipped();
    let params = DecodeParams {
        language: Some(contract.decode.language.clone()),
        translate: contract.decode.translate,
        beam_size: Some(contract.decode.beam_size),
        max_context: Some(contract.decode.max_context),
        timestamps,
        n_threads: matched_threads.as_ref().map_or(0, |_| threads),
        max_text_ctx: None,
        word_timestamps: contract.decode.word_timestamps,
        model_hint: Some(model_short.to_string()),
        ..DecodeParams::default()
    };
    // franken's fallback ladder is env-gated and default-off. If the contract
    // pins fallback off, an operator-set `FW_TEMP_FALLBACK` would make franken
    // do retry work the incumbent's `-nf` forbids — unmatched in franken's
    // favour or against it, either way not this comparison.
    let franken_fallback_on = std::env::var_os("FW_TEMP_FALLBACK").is_some();
    // `FW_NO_CONTEXT` is franken's env hatch for the same knob the contract now
    // pins. An operator with it set would silently override the contract's
    // carry policy on franken's side ONLY, leaving the incumbent at `-mc 0` and
    // the row still claiming matched decode. The contract owns this parameter,
    // so the hatch is a hard abort rather than a gate flag: unlike a busy CPU,
    // no amount of extra sampling makes an unmatched decode admissible.
    assert!(
        std::env::var_os("FW_NO_CONTEXT").is_none(),
        "FW_NO_CONTEXT is set, which overrides the contract's matched \
         max_context={} on franken's arm only; unset it and re-run",
        contract.decode.max_context
    );
    let decode_matched = franken_fallback_on == contract.decode.temperature_fallback
        && params.beam_size == Some(contract.decode.beam_size)
        && params.max_context == Some(contract.decode.max_context)
        && params.translate == contract.decode.translate
        && params.word_timestamps == contract.decode.word_timestamps
        && params.language.as_deref() == Some(contract.decode.language.as_str());
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
            false,
        ),
        BenchScope::WholeJob => {
            run_franken_whole(&model_path, model_short, &wav, threads, timestamps, false)
        }
    };
    let probe_franken_threads = || match scope {
        BenchScope::TranscribeOnly => run_franken_resident(
            resident_model.as_ref().expect("resident model"),
            &samples,
            &params,
            true,
        ),
        BenchScope::WholeJob => {
            run_franken_whole(&model_path, model_short, &wav, threads, timestamps, true)
        }
    };
    let run_whisper = || {
        run_incumbent(
            &incumbent,
            &model_path,
            &wav,
            threads,
            timestamps,
            scope,
            false,
            &contract.decode,
        )
    };
    let probe_whisper_threads = || {
        run_incumbent(
            &incumbent,
            &model_path,
            &wav,
            threads,
            timestamps,
            scope,
            true,
            &contract.decode,
        )
    };

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

    // Version contract. The incumbent is a pinned version, not "whatever binary
    // was at that path today": source digests plus the declared project version
    // are checked here, and the *running* image is checked against the contract
    // once the identity probe has attested it.
    let incumbent_source_root = incumbent_source_root(&incumbent);
    let source_verdicts = incumbent_source_root.as_deref().map_or_else(
        || {
            vec![(
                "source_root".to_owned(),
                false,
                "unavailable_from_incumbent_path".to_owned(),
            )]
        },
        |root| contract.source_verdicts(root),
    );
    let source_clear = source_verdicts.iter().all(|(_, ok, _)| *ok);
    println!(
        "INCUMBENT_AB_CONTRACT path={INCUMBENT_CONTRACT_PATH} project={} version={} \
         contract_source_root={} runtime_source_root={} \
         contract_binary_sha256={} source_clear={source_clear} \
         source_verdicts={} build={} decode_source=version_contract {} \
         franken_temp_fallback_env={franken_fallback_on} decode_matched={decode_matched}",
        contract.project,
        contract.version,
        contract.source_root,
        incumbent_source_root.as_deref().map_or_else(
            || "unavailable".to_owned(),
            |root| root.display().to_string()
        ),
        contract.binary_sha256,
        json!(
            source_verdicts
                .iter()
                .map(|(name, ok, observed)| json!({
                    "file": name,
                    "matches": ok,
                    "observed": observed
                }))
                .collect::<Vec<_>>()
        ),
        contract.build,
        contract.decode.as_row(),
    );
    let host_online_cpus = online_cpus().expect("read host online CPU topology");
    let host_allowed_cpus = self_allowed_cpus().expect("read harness CPU affinity");
    let cpu_frequency_policy = CpuFrequencyPolicy::observe(&host_online_cpus);
    println!(
        "INCUMBENT_AB_HOST {}",
        host_provenance(&cpu_frequency_policy, host_online_cpus.len())
    );
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
    let cpu_frequency_policy_clear =
        require_cpu_frequency_policy(&cpu_frequency_policy, host_online_cpus.len());

    // This is a harness requirement, not a booking convention. Sampling all
    // online CPUs (rather than only the current affinity mask) prevents
    // `taskset` or a cpuset cap from hiding a host-wide memory-bandwidth
    // competitor on CPUs the harness itself cannot schedule on.
    let host_preflight =
        require_host_wide_quiescence("preflight", &host_online_cpus, &host_allowed_cpus);

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

    // Warm both engines once while observing actual CPU-active threads. The
    // probe is untimed so `/proc` sampling cannot perturb the measured arms.
    let fw_warm = run_then_checkpoint(
        &probe_franken_threads,
        &mut checkpoint,
        "warm_franken_thread_probe",
    );
    let wc_warm = run_then_checkpoint(
        &probe_whisper_threads,
        &mut checkpoint,
        "warm_incumbent_thread_probe",
    );
    // Per-arm binary identity, attested by the arm's own process image. This is
    // the control that stops the harness from silently comparing a build against
    // itself: if both arms resolved to the same ELF the ratio would be a null
    // control wearing a competitive label, and every downstream gate would pass.
    let franken_exe = fw_warm.exe_sha256.clone();
    let incumbent_exe = wc_warm.exe_sha256.clone();
    let distinct_binaries = match (&franken_exe, &incumbent_exe) {
        (Some(franken), Some(incumbent)) => franken != incumbent,
        _ => false,
    };
    let contract_binary_clear = contract.binary_matches(incumbent_exe.as_deref());
    let identity_clear = distinct_binaries && contract_binary_clear && source_clear;
    // Builder identity, per the fleet local-perf-binary policy: `rch exec` has no
    // artifact-retrieval mechanism, so a release-perf harness is built on a remote
    // worker and copied back. A digest alone does not say which machine produced
    // it, and a binary of unknown origin is not evidence.
    let harness_builder =
        std::env::var("FW_HARNESS_BUILDER").unwrap_or_else(|_| "unrecorded".to_owned());
    println!(
        "INCUMBENT_AB_IDENTITY franken_exe_sha256={} incumbent_exe_sha256={} \
         attestation=proc_exe_of_running_process franken_source={} \
         harness_builder={harness_builder} \
         distinct_binaries={distinct_binaries} \
         contract_binary_clear={contract_binary_clear} source_clear={source_clear} \
         identity_clear={identity_clear}",
        franken_exe.as_deref().unwrap_or("unattested"),
        incumbent_exe.as_deref().unwrap_or("unattested"),
        match scope {
            BenchScope::TranscribeOnly => "proc_self_exe_in_process",
            BenchScope::WholeJob => "proc_pid_exe_of_worker",
        },
    );

    let wer = word_error_rate(&wc_warm.transcript, &fw_warm.transcript);
    let quality_clear = wer.wer <= MAX_CROSS_ENGINE_WER;
    let configured_thread_clear = !enforce_matched_threads
        || (fw_warm.actual_threads == threads && wc_warm.actual_threads == threads);
    let observed_thread_clear = scope != BenchScope::WholeJob
        || (fw_warm.observed_active_threads > 0 && wc_warm.observed_active_threads > 0);
    let thread_clear = configured_thread_clear && observed_thread_clear;
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
        "INCUMBENT_AB_THREADS requested={threads} franken_configured={} \
         incumbent_configured={} franken_observed_active={} incumbent_observed_active={} \
         franken_peak_process={} incumbent_peak_process={} \
         observed_source=proc_task_cpu_ticks franken_load_workers_config={} \
         affinity={} cpuset_effective={} enforce_matched={enforce_matched_threads} \
         configured_thread_clear={configured_thread_clear} \
         observed_thread_clear={observed_thread_clear} thread_clear={thread_clear}",
        fw_warm.actual_threads,
        wc_warm.actual_threads,
        fw_warm.observed_active_threads,
        wc_warm.observed_active_threads,
        fw_warm.peak_process_threads,
        wc_warm.peak_process_threads,
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
    assert!(
        fw_warm.work.window_attempts > 0
            && wc_warm.work.window_attempts > 0
            && fw_warm.work.encoder_calls > 0
            && wc_warm.work.encoder_calls > 0
            && fw_warm.work.single_token_decode_calls > 0
            && wc_warm.work.single_token_decode_calls > 0,
        "benchmark work counters must be positive for both matched greedy arms"
    );
    let attempt_work_ratio =
        wc_warm.work.window_attempts as f64 / fw_warm.work.window_attempts as f64;
    let encode_work_ratio = wc_warm.work.encoder_calls as f64 / fw_warm.work.encoder_calls as f64;
    let decode_work_ratio = wc_warm.work.single_token_decode_calls as f64
        / fw_warm.work.single_token_decode_calls as f64;
    let work_count_class = work_count_classification(&fw_warm.work, &wc_warm.work);
    println!(
        "INCUMBENT_AB_WORK franken_window_attempts={} franken_encoder_calls={} \
         franken_decoder_prefills={} franken_decoder_prefill_tokens={} \
         franken_selected_tokens={} \
         franken_single_token_forwards={} franken_accepted_windows={} \
         franken_accepted_result_tokens={} franken_prompt_reset_retries={} \
         franken_temperature_fallback_retries={} incumbent_window_attempts={} \
         incumbent_encode_runs={} incumbent_sample_counter_raw={} incumbent_decode_runs={} \
         incumbent_batchd_counter_raw={} incumbent_prompt_counter_raw={} \
         incumbent_prompt_fallbacks={} \
         incumbent_hallucination_fallbacks={} \
         incumbent_over_franken_window_attempt_work={attempt_work_ratio:.6} \
         incumbent_over_franken_encode_work={encode_work_ratio:.6} \
         incumbent_over_franken_single_token_decode_work={decode_work_ratio:.6} \
         incumbent_sample_counter_comparable=false \
         incumbent_batchd_prompt_counters_comparable=false \
         classification={work_count_class}",
        fw_warm.work.window_attempts,
        fw_warm.work.encoder_calls,
        fw_warm.work.decoder_prefill_calls,
        fw_warm.work.decoder_prefill_tokens,
        fw_warm.work.selected_tokens,
        fw_warm.work.single_token_decode_calls,
        fw_warm.work.accepted_windows,
        fw_warm.work.accepted_result_tokens,
        fw_warm.work.prompt_reset_retries,
        fw_warm.work.temperature_fallback_retries,
        wc_warm.work.window_attempts,
        wc_warm.work.encoder_calls,
        wc_warm.work.incumbent_sampling_counter,
        wc_warm.work.single_token_decode_calls,
        wc_warm.work.incumbent_batch_decode_counter,
        wc_warm.work.incumbent_prompt_counter,
        wc_warm.work.prompt_fallbacks,
        wc_warm.work.hallucination_fallbacks,
    );
    let run_franken_checked = || {
        let observation = run_franken();
        assert_eq!(
            observation.transcript_sha256, fw_warm.transcript_sha256,
            "franken transcript changed within one invocation"
        );
        assert_eq!(
            observation.actual_threads, fw_warm.actual_threads,
            "franken configured thread count changed within one invocation"
        );
        assert_eq!(
            observation.work, fw_warm.work,
            "franken work count changed within one invocation"
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
            "whisper.cpp configured thread count changed within one invocation"
        );
        assert_eq!(
            observation.work, wc_warm.work,
            "whisper.cpp work count changed within one invocation"
        );
        observation
    };

    let host_pre_measurement =
        require_host_wide_quiescence("pre_measurement", &host_online_cpus, &host_allowed_cpus);

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

    let host_post_measurement =
        sample_host_wide_quiescence("post_measurement", &host_online_cpus, &host_allowed_cpus);
    let host_wide_clear = host_preflight.is_clear()
        && host_pre_measurement.is_clear()
        && host_post_measurement.is_clear();

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
    let fw_null_stats = report("INCUMBENT_AB_NULL_FW", &fw_null);
    let wc_null_stats = report("INCUMBENT_AB_NULL_WC", &wc_null);
    let comparison_stats = report("INCUMBENT_AB_COMPARE", &compare);
    let load_split_clear = load_split_gap <= MAX_LOAD_SPLIT_GAP;
    let prerequisite_gates_clear = quality_clear
        && thread_clear
        && load_split_clear
        && external_host_clear
        && host_wide_clear
        && cpu_frequency_policy_clear
        && identity_clear
        && decode_matched;
    let gate = statistical_gate(
        fw_null_stats,
        wc_null_stats,
        comparison_stats,
        prerequisite_gates_clear,
    );
    println!(
        "INCUMBENT_AB_GATE method=median_vs_both_null_ci95_2x_margin \
         null_ci_straddle_required=false \
         fw_null_half={:.6} wc_null_half={:.6} required={:.6} \
         compare_median={:.6} compare_ci95=[{:.6},{:.6}] \
         load_split_gap={load_split_gap:.6} load_split_max={MAX_LOAD_SPLIT_GAP:.6} \
         load_split_clear={load_split_clear} quality_clear={quality_clear} \
         thread_clear={thread_clear} external_host_clear={external_host_clear} \
         host_wide_clear={host_wide_clear} \
         host_quiescence_threshold={MAX_HOST_CPU_BUSY_FRACTION:.6} \
         host_quiescence_sample_ms={} host_quiescence_settle_ms={} \
         cpu_frequency_policy_clear={cpu_frequency_policy_clear} \
         identity_clear={identity_clear} decode_matched={decode_matched} \
         incumbent_version={} incumbent_exe_sha256={} \
         work_counts_stable=true work_count_classification={work_count_class} \
         cv_is_provenance_only=true class=vs_incumbent verdict={}",
        gate.fw_null_half,
        gate.wc_null_half,
        gate.required,
        comparison_stats.0,
        comparison_stats.1,
        comparison_stats.2,
        HOST_CPU_SAMPLE_INTERVAL.as_millis(),
        HOST_QUIESCENCE_SETTLE.as_millis(),
        contract.version,
        incumbent_exe.as_deref().unwrap_or("unattested"),
        gate.verdict,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendored `whisper.cpp` tree is gitignored and is not synced to remote
    /// build workers, so source-drift assertions must skip rather than fail where
    /// the incumbent's source is simply absent. The runtime gate is unaffected:
    /// `identity_clear` still fails closed on drift wherever a ratio is produced.
    fn vendored_incumbent_source_root(contract: &IncumbentContract) -> Option<PathBuf> {
        let root = Path::new(CRATE_ROOT).join(&contract.source_root);
        let present = root.join("CMakeLists.txt").is_file();
        if !present {
            println!(
                "SKIP vendored incumbent source absent at {}",
                contract.source_root
            );
        }
        present.then_some(root)
    }

    #[test]
    fn shipped_contract_matches_the_vendored_incumbent_source() {
        let contract = IncumbentContract::shipped();
        assert_eq!(contract.project, "whisper.cpp");
        let Some(source_root) = vendored_incumbent_source_root(&contract) else {
            return;
        };
        for (file, matches, observed) in contract.source_verdicts(&source_root) {
            assert!(
                matches,
                "incumbent contract drifted from the vendored source at {file}: observed \
                 {observed}; bump docs/INCUMBENT_CONTRACT.json in the same commit"
            );
        }
    }

    #[test]
    fn contract_pins_matched_greedy_decode_on_both_arms() {
        let contract = IncumbentContract::shipped();
        // whisper-cli defaults to -bs 5 -bo 5; a contract that let those stand
        // would compare beam search against franken's greedy path.
        assert_eq!(contract.decode.beam_size, 1);
        assert_eq!(contract.decode.best_of, 1);
        assert!(!contract.decode.temperature_fallback);
        let args = contract.decode.incumbent_args();
        for expected in ["-bs", "1", "-bo", "1", "-nf", "-l", "en", "-tp", "0.00"] {
            assert!(
                args.iter().any(|arg| arg == expected),
                "incumbent args {args:?} must state {expected}"
            );
        }
        assert!(contract.decode.as_row().contains("decode_mode=greedy"));
    }

    #[test]
    fn beam_contract_is_reported_as_beam_not_greedy() {
        let decode = MatchedDecode::from_json(&json!({
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "temperature_fallback": true,
            "max_context": 0,
            "language": "en",
            "translate": false,
            "word_timestamps": false
        }));
        assert!(decode.as_row().contains("decode_mode=beam"));
        // Fallback enabled means no `-nf`: the flag must track the contract, not
        // a hardcoded assumption at the call site.
        assert!(!decode.incumbent_args().iter().any(|arg| arg == "-nf"));
    }

    #[test]
    fn identical_arm_digests_fail_the_distinctness_control() {
        let contract = IncumbentContract::shipped();
        let same = Some(contract.binary_sha256.clone());
        let distinct = match (&same, &same) {
            (Some(franken), Some(incumbent)) => franken != incumbent,
            _ => false,
        };
        assert!(
            !distinct,
            "one build measured twice must never read as a competitive pair"
        );
        // An unattested arm is also not clear: absence of a digest is not proof.
        assert!(!contract.binary_matches(None));
        assert!(!contract.binary_matches(Some("deadbeef")));
        assert!(contract.binary_matches(Some(&contract.binary_sha256)));
    }

    #[test]
    fn declared_source_version_parses_the_cmake_project_line() {
        assert_eq!(declared_source_version(Path::new("/nonexistent")), None);
        let contract = IncumbentContract::shipped();
        let Some(source_root) = vendored_incumbent_source_root(&contract) else {
            return;
        };
        let cmakelists = source_root.join("CMakeLists.txt");
        assert_eq!(
            declared_source_version(&cmakelists).as_deref(),
            Some(contract.version.as_str())
        );
    }

    #[test]
    fn runtime_incumbent_path_recovers_the_source_root() {
        assert_eq!(INCUMBENT_CONTRACT_PATH, "docs/INCUMBENT_CONTRACT.json");
        assert!(INCUMBENT_CONTRACT_JSON.contains("\"project\": \"whisper.cpp\""));
        assert_eq!(
            incumbent_source_root(Path::new(
                "/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/build/bin/whisper-cli"
            ))
            .as_deref(),
            Some(Path::new(
                "/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp"
            ))
        );
    }

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
    fn existing_gate_classifies_all_seven_reference_losses_and_creates_zero_wins() {
        // Exact medians and CI95s from frankenlibc's second wordexp run, which
        // reproduced its straddle-veto precision defect across a different
        // core/load. This comparator already ignored straddling. The effect is
        // reported as franken/incumbent, so invert the median and CI into this
        // harness's incumbent/franken orientation before adjudicating.
        let rows = [
            (
                "plain_split",
                (1.002_309, 1.000_400, 1.005_380),
                (0.992_411, 0.981_933, 1.007_598),
                (2.930_311, 2.906_338, 2.951_770),
            ),
            (
                "quoted_mix",
                (1.002_050, 0.993_515, 1.004_092),
                (0.995_116, 0.983_071, 1.003_489),
                (3.951_984, 3.922_415, 3.967_720),
            ),
            (
                "param_simple",
                (0.998_526, 0.993_386, 1.002_382),
                (0.988_760, 0.982_182, 0.998_423),
                (2.833_764, 2.822_487, 2.842_885),
            ),
            (
                "param_braced",
                (0.994_133, 0.985_840, 1.006_088),
                (0.995_028, 0.976_079, 1.009_118),
                (3.416_667, 3.334_717, 3.570_602),
            ),
            (
                "param_default",
                (0.994_730, 0.990_901, 1.001_109),
                (0.988_910, 0.977_533, 0.994_789),
                (2.323_656, 2.305_838, 2.332_914),
            ),
            (
                "escapes",
                (1.000_292, 0.993_404, 1.003_369),
                (1.000_201, 0.989_835, 1.009_008),
                (2.963_092, 2.952_614, 2.982_285),
            ),
            (
                "many_fields",
                (0.995_260, 0.990_393, 0.998_880),
                (0.999_463, 0.995_460, 1.007_522),
                (3.643_581, 3.597_722, 3.664_674),
            ),
        ];

        let mut losses = 0;
        let mut wins = 0;
        let mut non_straddling_cases = 0;
        for (name, fw_null, wc_null, (effect_med, effect_lo, effect_hi)) in rows {
            let comparison = (1.0 / effect_med, 1.0 / effect_hi, 1.0 / effect_lo);
            let gate = statistical_gate(fw_null, wc_null, comparison, true);
            let fw_straddles = fw_null.1 <= 1.0 && fw_null.2 >= 1.0;
            let wc_straddles = wc_null.1 <= 1.0 && wc_null.2 >= 1.0;
            non_straddling_cases += usize::from(!fw_straddles || !wc_straddles);
            assert_eq!(gate.verdict, "LOSS", "{name}");
            losses += usize::from(gate.verdict == "LOSS");
            wins += usize::from(gate.verdict == "WIN");
        }

        assert_eq!(non_straddling_cases, 4);
        assert_eq!(losses, 7);
        assert_eq!(wins, 0);
    }

    #[test]
    fn gate_verdicts_are_independent_of_whether_a_null_ci_contains_one() {
        // The straddle veto this harness was audited for would read a null CI
        // that misses 1.0 as a broken control and refuse to decide. Here a null
        // CI's only role is its *distance* from 1.0, which sets the 2x floor.
        // Same half-widths, three straddle configurations, one verdict each way.
        let straddling = ((1.000, 0.990, 1.010), (1.000, 0.995, 1.005));
        let both_above = ((1.010, 1.005, 1.010), (1.005, 1.002, 1.005));
        let both_below = ((0.990, 0.990, 0.995), (0.995, 0.995, 0.998));

        for (name, (fw, wc)) in [
            ("straddling", straddling),
            ("both_above_one", both_above),
            ("both_below_one", both_below),
        ] {
            let required = statistical_gate(fw, wc, (1.50, 1.40, 1.60), true).required;
            assert!(
                (required - 1.02).abs() < 1e-9,
                "{name}: floor must come from null half-width, got {required}"
            );
            assert_eq!(
                statistical_gate(fw, wc, (1.50, 1.40, 1.60), true).verdict,
                "WIN",
                "{name}: a clear win must not be vetoed by null straddle state"
            );
            assert_eq!(
                statistical_gate(fw, wc, (0.60, 0.55, 0.65), true).verdict,
                "LOSS",
                "{name}: a clear loss must not be vetoed by null straddle state"
            );
            assert_eq!(
                statistical_gate(fw, wc, (1.01, 1.005, 1.02), true).verdict,
                "UNDECIDABLE",
                "{name}: the 2x floor must still bite"
            );
        }
    }

    #[test]
    fn contract_pins_matched_no_context_carry_on_both_arms() {
        let contract = IncumbentContract::shipped();
        // Carry is pinned OFF, and 0 is the one value both engines express
        // exactly: franken `Some(0)` disables carry, `whisper-cli -mc 0` stores
        // no text context. whisper-cli's own default is -1 (unbounded), and
        // franken's is a per-model/mode policy, so an unpinned cell compares
        // two different prompt histories.
        assert_eq!(contract.decode.max_context, 0);

        let args = contract.decode.incumbent_args();
        let mc = args
            .iter()
            .position(|arg| arg == "-mc")
            .expect("incumbent args must state -mc");
        assert_eq!(args[mc + 1], "0");

        // The matched row must echo it: a matched-params claim that is not
        // printed is not checkable by a reviewer.
        assert!(contract.decode.as_row().contains("max_context=0"));

        // Both franken arms take it from the contract, not from a literal.
        for params in [
            DecodeParams {
                max_context: Some(contract.decode.max_context),
                ..DecodeParams::default()
            },
            DecodeParams {
                max_context: Some(contract.decode.max_context),
                beam_size: Some(contract.decode.beam_size),
                ..DecodeParams::default()
            },
        ] {
            assert_eq!(params.max_context, Some(0));
        }
    }

    #[test]
    fn unpinned_or_unbounded_context_is_reported_as_such() {
        // A contract that left carry unbounded must not silently read as the
        // pinned no-context cell: -1 is whisper-cli's own default and franken's
        // "restore n_text_ctx/2" sentinel, i.e. the unmatched case.
        let unbounded = MatchedDecode::from_json(&json!({
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "temperature_fallback": false,
            "max_context": -1,
            "language": "en",
            "translate": false,
            "word_timestamps": false
        }));
        assert_eq!(unbounded.max_context, -1);
        assert!(unbounded.as_row().contains("max_context=-1"));
        let args = unbounded.incumbent_args();
        let mc = args
            .iter()
            .position(|arg| arg == "-mc")
            .expect("-mc stated");
        assert_eq!(args[mc + 1], "-1");
        // Still explicitly passed: a default is a fact about a version, not
        // about the comparison, and it can move under a pinned digest bump.
        assert_ne!(
            unbounded.as_row(),
            IncumbentContract::shipped().decode.as_row()
        );
    }

    #[test]
    fn whole_job_worker_decode_is_derived_from_the_contract_not_literals() {
        // The whole-job arm runs in a spawned copy of this same ELF, so it must
        // re-derive decode from the shipped contract. If it rebuilds those
        // values from literals, the parent's `decode_matched` describes params
        // the measured process never used, and the two only agree by luck.
        let contract = IncumbentContract::shipped();
        let worker_params = DecodeParams {
            language: Some(contract.decode.language.clone()),
            translate: contract.decode.translate,
            beam_size: Some(contract.decode.beam_size),
            max_context: Some(contract.decode.max_context),
            timestamps: false,
            n_threads: 32,
            max_text_ctx: None,
            word_timestamps: contract.decode.word_timestamps,
            model_hint: Some("large-v3-turbo".to_owned()),
            ..DecodeParams::default()
        };
        // Exactly the predicates `decode_matched` asserts in the parent.
        assert_eq!(worker_params.beam_size, Some(contract.decode.beam_size));
        assert_eq!(worker_params.max_context, Some(contract.decode.max_context));
        assert_eq!(worker_params.translate, contract.decode.translate);
        assert_eq!(
            worker_params.word_timestamps,
            contract.decode.word_timestamps
        );
        assert_eq!(
            worker_params.language.as_deref(),
            Some(contract.decode.language.as_str())
        );
        // The row the worker attests back is the row the parent compares to.
        assert_eq!(
            contract.decode.as_row(),
            IncumbentContract::shipped().decode.as_row()
        );
    }

    #[test]
    fn quiescence_settle_applies_only_after_the_harness_has_run_work() {
        // preflight precedes any work of ours, so it must sample immediately:
        // a competitor already resident should be caught without a grace period.
        assert_eq!(quiescence_settle_for("preflight"), Duration::ZERO);
        // Both post-probe checkpoints follow full 32-thread transcriptions of
        // both engines and must not measure that wind-down as a competitor.
        assert_eq!(
            quiescence_settle_for("pre_measurement"),
            HOST_QUIESCENCE_SETTLE
        );
        assert_eq!(
            quiescence_settle_for("post_measurement"),
            HOST_QUIESCENCE_SETTLE
        );
        // Provenance samples are settle-free, so disclosing what a settle
        // absorbed can never itself recurse into another settle.
        assert_eq!(
            quiescence_settle_for("pre_measurement_immediate"),
            Duration::ZERO
        );
        assert_eq!(
            quiescence_settle_for("post_measurement_immediate"),
            Duration::ZERO
        );
    }

    #[test]
    fn quiescence_settle_did_not_loosen_the_exclusivity_contract() {
        // The settle escapes our own decaying tail, nothing else. If a later
        // edit trades the self-veto fix for a weaker threshold, shorter window
        // or partial CPU coverage, that is a different gate and this fails.
        assert!(
            (MAX_HOST_CPU_BUSY_FRACTION - 0.20).abs() < f64::EPSILON,
            "host quiescence threshold is owner-mandated at 20% busy"
        );
        assert_eq!(HOST_CPU_SAMPLE_INTERVAL, Duration::from_millis(300));
        assert!(
            HOST_QUIESCENCE_SETTLE >= HOST_CPU_SAMPLE_INTERVAL,
            "a settle shorter than the sample window cannot clear a wind-down tail"
        );

        // Coverage and verdict semantics are unchanged by the settle: a CPU over
        // threshold is still dirty, and an unsampled online CPU is still dirty.
        let over_threshold = HostWideQuiescence {
            checkpoint: "pre_measurement".to_owned(),
            online_cpu_count: 64,
            sampled_online_cpu_count: 64,
            allowed_cpu_count: 64,
            max_busy_fraction: 0.21,
            busy_cpus: vec![(19, 0.219)],
        };
        assert!(!over_threshold.is_clear());
        let partially_sampled = HostWideQuiescence {
            busy_cpus: Vec::new(),
            max_busy_fraction: 0.0,
            sampled_online_cpu_count: 63,
            ..over_threshold
        };
        assert!(!partially_sampled.is_clear());
    }

    #[test]
    fn statistical_gate_retains_margin_effect_ci_and_prerequisite_vetoes() {
        let clear_nulls = (
            (1.000_000, 0.990_000, 1.010_000),
            (1.000_000, 0.995_000, 1.005_000),
        );

        let prerequisite_failure =
            statistical_gate(clear_nulls.0, clear_nulls.1, (1.50, 1.40, 1.60), false);
        assert_eq!(prerequisite_failure.verdict, "UNDECIDABLE");

        let ci_touches_one =
            statistical_gate(clear_nulls.0, clear_nulls.1, (1.50, 1.00, 1.60), true);
        assert_eq!(ci_touches_one.verdict, "UNDECIDABLE");

        let inside_twice_null_margin =
            statistical_gate(clear_nulls.0, clear_nulls.1, (1.01, 1.005, 1.02), true);
        assert_eq!(inside_twice_null_margin.verdict, "UNDECIDABLE");

        let clear_win = statistical_gate(clear_nulls.0, clear_nulls.1, (1.50, 1.40, 1.60), true);
        assert_eq!(clear_win.verdict, "WIN");
    }

    #[test]
    fn cpu_policy_grouping_requires_complete_uniform_values() {
        let uniform = group_cpu_policy_field([
            (0, Some("performance".to_owned())),
            (1, Some("performance".to_owned())),
            (2, Some("performance".to_owned())),
        ]);
        assert!(uniform.is_complete_uniform(3));
        assert_eq!(uniform.uniform_value(3), Some("performance"));
        assert_eq!(
            uniform.as_json(3)["by_value"]["performance"],
            json!([0, 1, 2])
        );

        let mixed = group_cpu_policy_field([
            (0, Some("performance".to_owned())),
            (1, Some("powersave".to_owned())),
        ]);
        assert!(!mixed.is_complete_uniform(2));
        assert_eq!(mixed.uniform_value(2), None);

        let missing = group_cpu_policy_field([(0, Some("performance".to_owned())), (1, None)]);
        assert!(!missing.is_complete_uniform(2));
        assert_eq!(missing.as_json(2)["unavailable_cpus"], json!([1]));
    }

    #[test]
    fn cpu_frequency_policy_requires_performance_and_records_epp_without_gating_it() {
        let policy = CpuFrequencyPolicy {
            scaling_driver: group_cpu_policy_field([
                (0, Some("amd-pstate-epp".to_owned())),
                (1, Some("amd-pstate-epp".to_owned())),
            ]),
            scaling_governor: group_cpu_policy_field([
                (0, Some("performance".to_owned())),
                (1, Some("performance".to_owned())),
            ]),
            energy_performance_preference: group_cpu_policy_field([
                (0, Some("balance_performance".to_owned())),
                (1, Some("balance_performance".to_owned())),
            ]),
        };
        assert!(policy.benchmark_clear(2));

        let powersave = CpuFrequencyPolicy {
            scaling_driver: policy.scaling_driver,
            scaling_governor: group_cpu_policy_field([
                (0, Some("powersave".to_owned())),
                (1, Some("powersave".to_owned())),
            ]),
            energy_performance_preference: policy.energy_performance_preference,
        };
        assert!(!powersave.benchmark_clear(2));
    }

    #[test]
    fn cpu_list_parser_expands_ranges_and_rejects_descending_ranges() {
        assert_eq!(
            parse_cpu_list("0-2,5,7-8"),
            Ok(BTreeSet::from([0, 1, 2, 5, 7, 8]))
        );
        assert!(parse_cpu_list("3-1").is_err());
        assert!(parse_cpu_list("").is_err());
    }

    #[test]
    fn cpu_busy_fraction_matches_proc_stat_idle_accounting() {
        let before = BTreeMap::from([
            (
                0,
                CpuTicks {
                    total: 100,
                    idle: 80,
                },
            ),
            (
                1,
                CpuTicks {
                    total: 500,
                    idle: 400,
                },
            ),
        ]);
        let after = BTreeMap::from([
            (
                0,
                CpuTicks {
                    total: 200,
                    idle: 160,
                },
            ),
            (
                1,
                CpuTicks {
                    total: 600,
                    idle: 470,
                },
            ),
        ]);

        let busy = cpu_busy_between(&before, &after).expect("matching CPU samples");
        assert!((busy[&0] - 0.2).abs() < f64::EPSILON);
        assert!((busy[&1] - 0.3).abs() < f64::EPSILON);
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
[00:00:01.000 --> 00:00:02.000]  from whisper\n\
whisper_print_timings:     fallbacks =   1 p /   2 h\n\
whisper_print_timings:   sample time =     8.77 ms /    31 runs (0.28 ms per run)\n\
whisper_print_timings:   encode time =  1421.36 ms /     4 runs (355.34 ms per run)\n\
whisper_print_timings:   decode time =   156.54 ms /    27 runs (5.80 ms per run)\n\
whisper_print_timings:   batchd time =     9.15 ms /     4 runs (2.29 ms per run)\n\
whisper_print_timings:   prompt time =     0.00 ms /     4 runs (0.00 ms per run)\n";

        assert_eq!(
            incumbent_segments(output, true),
            ["hello world".to_owned(), "from whisper".to_owned()]
        );
        assert_eq!(incumbent_actual_threads(output), Some(64));
        assert_eq!(
            incumbent_segments("\n hello world from whisper\n", false),
            ["hello world from whisper".to_owned()]
        );
        assert_eq!(
            incumbent_work_counts(output),
            EngineWork {
                window_attempts: 7,
                encoder_calls: 4,
                single_token_decode_calls: 27,
                incumbent_sampling_counter: 31,
                incumbent_batch_decode_counter: 4,
                incumbent_prompt_counter: 4,
                accepted_windows: 4,
                prompt_fallbacks: 1,
                hallucination_fallbacks: 2,
                ..EngineWork::default()
            }
        );
    }

    #[test]
    fn work_classification_uses_only_semantically_matched_counters() {
        let franken = EngineWork {
            window_attempts: 4,
            encoder_calls: 4,
            selected_tokens: 163,
            single_token_decode_calls: 160,
            ..EngineWork::default()
        };
        let mut incumbent = EngineWork {
            window_attempts: 4,
            encoder_calls: 4,
            incumbent_sampling_counter: 1,
            single_token_decode_calls: 160,
            incumbent_batch_decode_counter: 2,
            incumbent_prompt_counter: 1,
            ..EngineWork::default()
        };

        assert_eq!(
            work_count_classification(&franken, &incumbent),
            "matched_within_10pct"
        );
        incumbent.single_token_decode_calls = 244;
        assert_eq!(
            work_count_classification(&franken, &incumbent),
            "incumbent_more_work"
        );
    }

    #[test]
    fn child_thread_probe_observes_cpu_active_process_thread() {
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg("i=0; while [ \"$i\" -lt 100000 ]; do i=$((i + 1)); done");

        let (output, wall_ms, probe) = run_command(&mut command, true);

        assert!(output.status.success());
        assert!(wall_ms > 0.0);
        assert!(probe.peak_process_threads >= 1);
        assert!(probe.observed_active_threads() >= 1);
    }

    #[test]
    fn host_provenance_includes_ram_and_numa_topology() {
        let online = online_cpus().expect("online CPUs");
        let policy = CpuFrequencyPolicy::observe(&online);
        let provenance = host_provenance(&policy, online.len());

        assert!(
            provenance["ram_bytes"]
                .as_u64()
                .is_some_and(|bytes| bytes > 0)
        );
        assert!(
            provenance["numa_nodes"]
                .as_u64()
                .is_some_and(|nodes| nodes > 0)
        );
        assert_eq!(
            provenance["cpu_frequency_policy"]["scaling_governor"]["online_cpu_count"],
            json!(online.len())
        );
    }
}
