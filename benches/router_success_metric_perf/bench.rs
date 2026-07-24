use std::collections::VecDeque;
use std::hint::black_box;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const PROFILE_SAMPLES: usize = 11;
const PAIRS: usize = 21;
const PROFILE_ARM_NS: u128 = 30_000_000;
const MEASURE_ARM_NS: u128 = 100_000_000;

#[derive(Clone, Copy, Debug)]
enum BackendKind {
    WhisperCpp,
    InsanelyFast,
    WhisperDiarization,
    Auto,
}

#[allow(dead_code)]
#[derive(Clone)]
struct RoutingOutcomeRecord {
    backend: BackendKind,
    success: bool,
    latency_ms: u64,
    error_message: Option<String>,
    recorded_at_rfc3339: String,
}

#[allow(dead_code)]
struct BackendMetrics {
    success_rate: f64,
    avg_latency_ms: f64,
    last_error: Option<String>,
    sample_count: usize,
    success_count: usize,
}

#[allow(dead_code)]
#[derive(Clone)]
struct CalibrationObservation {
    predicted_probability: f64,
    actual_outcome: f64,
    observed_at_rfc3339: String,
}

#[allow(dead_code)]
#[derive(Clone)]
struct CalibrationState {
    observations: VecDeque<CalibrationObservation>,
    window_size: usize,
}

#[allow(dead_code)]
#[derive(Clone)]
struct RoutingEvidenceLedgerEntry {
    decision_id: String,
    trace_id: String,
    timestamp_rfc3339: String,
    observed_state: String,
    chosen_action: String,
    recommended_order: Vec<String>,
    fallback_active: bool,
    fallback_reason: Option<String>,
    posterior_snapshot: Vec<f64>,
    calibration_score: f64,
    brier_score: Option<f64>,
    e_process: f64,
    ci_width: f64,
    adaptive_mode: bool,
    policy_id: String,
    loss_matrix_hash: String,
    availability: Vec<(String, bool)>,
    duration_bucket: String,
    diarize: bool,
    actual_outcome: Option<RoutingOutcomeRecord>,
}

#[allow(dead_code)]
#[derive(Clone)]
struct RoutingEvidenceLedger {
    entries: VecDeque<RoutingEvidenceLedgerEntry>,
    capacity: usize,
    total_recorded: u64,
}

#[allow(dead_code)]
#[derive(Clone)]
struct RouterState {
    histories: [VecDeque<RoutingOutcomeRecord>; 3],
    total_predictions: u64,
    correct_predictions: u64,
    calibration: CalibrationState,
    evidence_ledger: RoutingEvidenceLedger,
}

impl RouterState {
    fn slot(kind: BackendKind) -> Option<usize> {
        match kind {
            BackendKind::WhisperCpp => Some(0),
            BackendKind::InsanelyFast => Some(1),
            BackendKind::WhisperDiarization => Some(2),
            BackendKind::Auto => None,
        }
    }

    #[inline(never)]
    fn metrics_for(&self, kind: BackendKind) -> BackendMetrics {
        let Some(index) = Self::slot(kind) else {
            return BackendMetrics {
                success_rate: 0.0,
                avg_latency_ms: 0.0,
                last_error: None,
                sample_count: 0,
                success_count: 0,
            };
        };
        let history = &self.histories[index];
        let sample_count = history.len();
        if sample_count == 0 {
            return BackendMetrics {
                success_rate: 0.5,
                avg_latency_ms: 0.0,
                last_error: None,
                sample_count: 0,
                success_count: 0,
            };
        }

        let success_count = history.iter().filter(|record| record.success).count();
        let successful_latencies: Vec<f64> = history
            .iter()
            .filter(|record| record.success)
            .map(|record| record.latency_ms as f64)
            .collect();
        let avg_latency_ms = if successful_latencies.is_empty() {
            0.0
        } else {
            successful_latencies.iter().sum::<f64>() / successful_latencies.len() as f64
        };
        let last_error = history
            .iter()
            .rev()
            .find_map(|record| record.error_message.clone());

        BackendMetrics {
            success_rate: success_count as f64 / sample_count as f64,
            avg_latency_ms,
            last_error,
            sample_count,
            success_count,
        }
    }

    #[inline(never)]
    fn success_rate_for(&self, kind: BackendKind) -> f64 {
        let Some(index) = Self::slot(kind) else {
            return 0.0;
        };
        let history = &self.histories[index];
        if history.is_empty() {
            return 0.5;
        }
        history.iter().filter(|record| record.success).count() as f64 / history.len() as f64
    }
}

type MetricFn = fn(&Mutex<Option<RouterState>>, BackendKind) -> Option<f64>;

#[inline(never)]
fn snapshot(state: &Mutex<Option<RouterState>>) -> Option<RouterState> {
    state.lock().ok().and_then(|guard| guard.clone())
}

#[inline(never)]
fn historical(state: &Mutex<Option<RouterState>>, kind: BackendKind) -> Option<f64> {
    snapshot(state).map(|snapshot| snapshot.metrics_for(kind).success_rate)
}

#[inline(never)]
fn candidate(state: &Mutex<Option<RouterState>>, kind: BackendKind) -> Option<f64> {
    state
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|router| router.success_rate_for(kind)))
}

#[inline(never)]
fn lock_only(state: &Mutex<Option<RouterState>>, _: BackendKind) -> Option<f64> {
    state
        .lock()
        .ok()
        .map(|guard| if guard.is_some() { 1.0 } else { 0.0 })
}

#[inline(never)]
fn clone_only(state: &Mutex<Option<RouterState>>, _: BackendKind) -> Option<f64> {
    snapshot(state).map(|snapshot| snapshot.total_predictions as f64)
}

#[inline(never)]
fn borrowed_metrics(state: &Mutex<Option<RouterState>>, kind: BackendKind) -> Option<f64> {
    state.lock().ok().and_then(|guard| {
        guard
            .as_ref()
            .map(|router| router.metrics_for(kind).success_rate)
    })
}

fn outcome(slot: usize, index: usize) -> RoutingOutcomeRecord {
    RoutingOutcomeRecord {
        backend: match slot {
            0 => BackendKind::WhisperCpp,
            1 => BackendKind::InsanelyFast,
            _ => BackendKind::WhisperDiarization,
        },
        success: (index + slot) % 5 != 0,
        latency_ms: 120 + (index * 13 + slot * 17) as u64,
        error_message: ((index + slot) % 5 == 0)
            .then(|| format!("backend-{slot}-failure-{index}-{}", "x".repeat(48))),
        recorded_at_rfc3339: format!("2026-07-16T12:{slot:02}:{index:02}Z"),
    }
}

fn evidence_entry(index: usize) -> RoutingEvidenceLedgerEntry {
    RoutingEvidenceLedgerEntry {
        decision_id: format!("decision-{index:04}-{}", "d".repeat(32)),
        trace_id: format!("trace-{index:04}-{}", "t".repeat(32)),
        timestamp_rfc3339: format!("2026-07-16T12:{:02}:{:02}Z", index % 60, index % 60),
        observed_state: "all_backends_available".to_owned(),
        chosen_action: "whisper_cpp".to_owned(),
        recommended_order: vec![
            "whisper_cpp".to_owned(),
            "insanely_fast".to_owned(),
            "whisper_diarization".to_owned(),
        ],
        fallback_active: index % 11 == 0,
        fallback_reason: (index % 11 == 0).then(|| "calibration_guardrail".to_owned()),
        posterior_snapshot: vec![0.62, 0.27, 0.11],
        calibration_score: 0.91,
        brier_score: Some(0.08),
        e_process: 1.7,
        ci_width: 0.12,
        adaptive_mode: true,
        policy_id: "backend-selection-v1.0".to_owned(),
        loss_matrix_hash: format!("{:064x}", index + 1),
        availability: vec![
            ("whisper_cpp".to_owned(), true),
            ("insanely_fast".to_owned(), true),
            ("whisper_diarization".to_owned(), true),
        ],
        duration_bucket: "medium".to_owned(),
        diarize: index % 2 == 0,
        actual_outcome: Some(outcome(index % 3, index % 50)),
    }
}

fn router_state(history_len: usize, calibration_len: usize, evidence_len: usize) -> RouterState {
    let histories =
        std::array::from_fn(|slot| (0..history_len).map(|index| outcome(slot, index)).collect());
    let calibration = CalibrationState {
        observations: (0..calibration_len)
            .map(|index| CalibrationObservation {
                predicted_probability: 0.55 + (index % 10) as f64 / 100.0,
                actual_outcome: if index % 5 != 0 { 1.0 } else { 0.0 },
                observed_at_rfc3339: format!("2026-07-16T11:{:02}:{:02}Z", index, index),
            })
            .collect(),
        window_size: 50,
    };
    let evidence_ledger = RoutingEvidenceLedger {
        entries: (0..evidence_len).map(evidence_entry).collect(),
        capacity: 200,
        total_recorded: evidence_len as u64,
    };
    RouterState {
        histories,
        total_predictions: 400,
        correct_predictions: 364,
        calibration,
        evidence_ledger,
    }
}

fn assert_oracle() -> usize {
    let mut cases = 0;
    let absent = Mutex::new(None);
    assert_eq!(historical(&absent, BackendKind::WhisperCpp), None);
    assert_eq!(candidate(&absent, BackendKind::WhisperCpp), None);
    cases += 1;

    for state in [
        router_state(0, 0, 0),
        router_state(25, 25, 100),
        router_state(50, 50, 200),
    ] {
        let state = Mutex::new(Some(state));
        for kind in [
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
            BackendKind::Auto,
        ] {
            let old = historical(&state, kind).map(f64::to_bits);
            let new = candidate(&state, kind).map(f64::to_bits);
            assert_eq!(old, new, "success-rate parity for {kind:?}");
            cases += 1;
        }
    }
    cases
}

fn measure(
    state: &Mutex<Option<RouterState>>,
    implementation: MetricFn,
    repetitions: usize,
) -> Duration {
    let started = Instant::now();
    let mut checksum = 0_u64;
    for index in 0..repetitions {
        let value = implementation(black_box(state), BackendKind::WhisperCpp).unwrap_or(-1.0);
        checksum ^= value.to_bits().rotate_left((index & 63) as u32);
    }
    black_box(checksum);
    started.elapsed()
}

fn calibrated_repetitions(state: &Mutex<Option<RouterState>>, target_ns: u128) -> usize {
    let probe_repetitions = 25_usize;
    let probe_ns = measure(state, historical, probe_repetitions)
        .as_nanos()
        .max(1);
    let scaled = target_ns
        .saturating_mul(probe_repetitions as u128)
        .checked_div(probe_ns)
        .unwrap_or(probe_repetitions as u128);
    usize::try_from(scaled.clamp(25, 250_000)).expect("bounded repetitions")
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

fn run_profile(state: &Mutex<Option<RouterState>>, oracle_cases: usize) {
    let repetitions = calibrated_repetitions(state, PROFILE_ARM_NS);
    let implementations: [(&str, MetricFn); 5] = [
        ("lock_only", lock_only),
        ("snapshot_clone", clone_only),
        ("borrowed_full_metrics", borrowed_metrics),
        ("historical", historical),
        ("candidate_design", candidate),
    ];
    let mut samples = vec![Vec::with_capacity(PROFILE_SAMPLES); implementations.len()];
    for sample in 0..PROFILE_SAMPLES {
        for offset in 0..implementations.len() {
            let slot = (sample + offset) % implementations.len();
            samples[slot].push(measure(state, implementations[slot].1, repetitions).as_nanos());
        }
    }

    println!(
        "profile oracle_cases={oracle_cases} samples={PROFILE_SAMPLES} repetitions={repetitions} histories=3x50 calibration=50 evidence=200 lto=false"
    );
    let historical_ns = median_u128(&samples[3]) as f64 / repetitions as f64;
    for (slot, (label, _)) in implementations.iter().enumerate() {
        let ns = median_u128(&samples[slot]) as f64 / repetitions as f64;
        println!(
            "{label}_ns_per_call={ns:.3} share_of_historical={:.4}%",
            ns / historical_ns * 100.0
        );
    }
}

struct PairedResults {
    ratios: Vec<f64>,
    numerator_ns: Vec<u128>,
    denominator_ns: Vec<u128>,
}

fn paired_ratios(
    state: &Mutex<Option<RouterState>>,
    numerator: MetricFn,
    denominator: MetricFn,
    repetitions: usize,
) -> PairedResults {
    let mut ratios = Vec::with_capacity(PAIRS * 2);
    let mut numerator_ns = Vec::with_capacity(PAIRS * 2);
    let mut denominator_ns = Vec::with_capacity(PAIRS * 2);
    for pair in 0..PAIRS {
        let (first_n, first_d, second_d, second_n) = if pair % 2 == 0 {
            (
                measure(state, numerator, repetitions).as_nanos(),
                measure(state, denominator, repetitions).as_nanos(),
                measure(state, denominator, repetitions).as_nanos(),
                measure(state, numerator, repetitions).as_nanos(),
            )
        } else {
            let first_d = measure(state, denominator, repetitions).as_nanos();
            let first_n = measure(state, numerator, repetitions).as_nanos();
            let second_n = measure(state, numerator, repetitions).as_nanos();
            let second_d = measure(state, denominator, repetitions).as_nanos();
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

fn run_measure(state: &Mutex<Option<RouterState>>, oracle_cases: usize) {
    let repetitions = calibrated_repetitions(state, MEASURE_ARM_NS);
    for _ in 0..5 {
        black_box(measure(state, historical, repetitions));
        black_box(measure(state, candidate, repetitions));
    }

    let null = paired_ratios(state, historical, historical, repetitions);
    let comparison = paired_ratios(state, candidate, historical, repetitions);
    println!(
        "measure oracle_cases={oracle_cases} repetitions={repetitions} pairs={PAIRS} ratios_per_comparison={} histories=3x50 calibration=50 evidence=200 ratio=candidate_over_historical lto=false",
        PAIRS * 2
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
        && candidate_median < 0.25;
    println!(
        "gate=null_median_0.97_1.03+null_cv_le_3pct+candidate_p90_lt_null_p10+all_wins+candidate_median_lt_0.25 verdict={}",
        if keep { "KEEP" } else { "REJECT" }
    );
    if !keep {
        std::process::exit(2);
    }
}

fn main() {
    let oracle_cases = assert_oracle();
    let state = Mutex::new(Some(router_state(50, 50, 200)));
    match std::env::args().nth(1).as_deref() {
        Some("profile") => run_profile(&state, oracle_cases),
        Some("measure") => run_measure(&state, oracle_cases),
        other => panic!("expected profile or measure mode, got {other:?}"), // ubs:ignore — invalid benchmark CLI is a harness failure
    }
}
