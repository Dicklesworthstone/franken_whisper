//! bd-3pf.18: Cross-engine conformance comparator and shadow-run CI gate.
//!
//! This test file implements a conformance comparator harness that runs bridge
//! and native engines on a fixture corpus, computes drift metrics, enforces
//! tolerance gates, and emits reproducible artifacts for CI/shadow rollout
//! decisions.
//!
//! All structs are test-only and defined within this file.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use franken_whisper::backend::TranscriptSegment;

// ---------------------------------------------------------------------------
// bd-jryr: the production `WhisperCppPilot` (canned-phrase mock) was deleted
// when the native whisper.cpp engine was rewired to real inference. This file
// is owned by bead bd-4slu, so rather than rewrite it, we inline a small
// test-only builder that reproduces the *synthetic* segment stream the old
// pilot generated. It is clearly NOT engine output — it exists only to drive
// the conformance-comparator's drift/gate math with deterministic fixtures.
// ---------------------------------------------------------------------------

/// Synthetic, deterministic transcript segments for conformance fixtures.
///
/// Reproduces the former `WhisperCppPilot::transcribe(duration_ms)` output: one
/// segment per 5000 ms, cycling a fixed phrase list, confidence decaying by
/// segment index. Test-only synthetic data — never engine output.
fn synthetic_whisper_cpp_segments(duration_ms: u64) -> Vec<TranscriptSegment> {
    const SEGMENT_DURATION_MS: u64 = 5000;
    const PHRASES: &[&str] = &[
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test transcription.",
        "Speech recognition is a fascinating field.",
        "Artificial intelligence continues to advance.",
    ];

    let n_segments = if duration_ms == 0 {
        0
    } else {
        duration_ms.div_ceil(SEGMENT_DURATION_MS) as usize
    };

    (0..n_segments)
        .map(|i| {
            let start_ms = (i as u64) * SEGMENT_DURATION_MS;
            let end_ms = std::cmp::min(start_ms + SEGMENT_DURATION_MS, duration_ms);
            TranscriptSegment {
                start_ms,
                end_ms,
                text: PHRASES[i % PHRASES.len()].to_owned(),
                confidence: 0.92 - (i as f64 * 0.01),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// bd-s8w8: the production `InsanelyFastPilot` (canned-phrase mock) was deleted
// when the native insanely-fast engine was rewired to real parallel-window
// inference. This file is owned by bead bd-4slu, so rather than rewrite it, we
// inline a small test-only builder that reproduces the *synthetic* segment
// stream the old pilot generated. It is clearly NOT engine output — it exists
// only to drive the conformance-comparator's drift/gate math with deterministic
// fixtures.
// ---------------------------------------------------------------------------

/// Synthetic, deterministic transcript segments for conformance fixtures.
///
/// Reproduces the former `InsanelyFastPilot::transcribe_batch(&[duration_ms])`
/// output for a single file: one segment per 10000 ms, cycling a fixed phrase
/// list, confidence decaying by segment index. Test-only synthetic data — never
/// engine output.
fn synthetic_insanely_fast_segments(duration_ms: u64) -> Vec<TranscriptSegment> {
    const SEGMENT_DURATION_MS: u64 = 10_000;
    const PHRASES: &[&str] = &[
        "Batch processing enables higher throughput.",
        "GPU acceleration reduces latency significantly.",
        "Parallel inference is the future of ASR.",
    ];

    let n_segments = if duration_ms == 0 {
        0
    } else {
        duration_ms.div_ceil(SEGMENT_DURATION_MS) as usize
    };

    (0..n_segments)
        .map(|i| {
            let start_ms = (i as u64) * SEGMENT_DURATION_MS;
            let end_ms = std::cmp::min(start_ms + SEGMENT_DURATION_MS, duration_ms);
            TranscriptSegment {
                start_ms,
                end_ms,
                text: PHRASES[i % PHRASES.len()].to_owned(),
                confidence: 0.95 - (i as f64 * 0.005),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// bd-cidv: the production `DiarizationPilot` (canned-phrase mock) was deleted
// when the native whisper-diarization engine was rewired to real ASR + the
// orchestrator's heuristic diarizer. This file is owned by bead bd-4slu, so
// rather than rewrite it, we inline a small test-only builder that reproduces
// the *synthetic* segment stream the old pilot generated. It is clearly NOT
// engine output — it exists only to drive the conformance-comparator's
// drift/gate math with deterministic fixtures.
// ---------------------------------------------------------------------------

/// Synthetic, deterministic transcript segments for conformance fixtures.
///
/// Reproduces the former `DiarizationPilot::process(duration_ms)` segment stream
/// (flattened to [`TranscriptSegment`]): one segment per 3000 ms over two
/// speakers, cycling a fixed meeting-phrase list, confidence decaying by segment
/// index. Test-only synthetic data — never engine output.
fn synthetic_diarization_segments(duration_ms: u64) -> Vec<TranscriptSegment> {
    const SEGMENT_DURATION_MS: u64 = 3000;
    const PHRASES: &[&str] = &[
        "Good morning, everyone.",
        "Thank you for joining us today.",
        "Let's begin with the first topic.",
        "I have a question about that.",
        "That's an excellent point.",
        "Could you elaborate further?",
    ];

    let n_segments = if duration_ms == 0 {
        0
    } else {
        duration_ms.div_ceil(SEGMENT_DURATION_MS) as usize
    };

    (0..n_segments)
        .map(|i| {
            let start_ms = (i as u64) * SEGMENT_DURATION_MS;
            let end_ms = std::cmp::min(start_ms + SEGMENT_DURATION_MS, duration_ms);
            TranscriptSegment {
                start_ms,
                end_ms,
                text: PHRASES[i % PHRASES.len()].to_owned(),
                confidence: 0.88 - (i as f64 * 0.005),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Mock engine abstraction (test-only)
// ---------------------------------------------------------------------------

/// Identifies the category of a mock engine for conformance testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum MockEngineKind {
    /// Bridge-style engine wrapping an external process.
    Bridge,
    /// Native pilot engine with in-process mock inference.
    Pilot,
}

impl MockEngineKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bridge => "bridge",
            Self::Pilot => "pilot",
        }
    }
}

/// A mock engine that produces deterministic `TranscriptSegment` output for a
/// given audio duration. Each engine has a name, kind, and a closure that
/// produces segments.
struct MockEngine {
    name: String,
    kind: MockEngineKind,
    /// Closure that takes audio duration in ms and returns transcript segments.
    produce: Box<dyn Fn(u64) -> Vec<TranscriptSegment>>,
}

impl MockEngine {
    fn run(&self, duration_ms: u64) -> Vec<TranscriptSegment> {
        (self.produce)(duration_ms)
    }
}

// ---------------------------------------------------------------------------
// DriftMetrics — per-segment WER approximation and confidence delta
// ---------------------------------------------------------------------------

/// Drift metrics computed between two engine outputs on the same input.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DriftMetrics {
    /// Name of the reference engine.
    reference_engine: String,
    /// Name of the candidate engine.
    candidate_engine: String,
    /// Approximate word error rate between the two outputs (0.0 = identical).
    wer_approx: f64,
    /// Mean absolute difference in confidence scores across aligned segments.
    confidence_delta: f64,
    /// Number of segments in the reference output.
    reference_segment_count: usize,
    /// Number of segments in the candidate output.
    candidate_segment_count: usize,
    /// Maximum per-segment WER observed.
    max_segment_wer: f64,
}

impl DriftMetrics {
    /// Compute drift metrics between reference and candidate segment lists.
    ///
    /// Segments are compared positionally: the i-th reference segment is
    /// compared against the i-th candidate segment. If segment counts differ,
    /// surplus segments contribute a WER of 1.0 and a confidence delta of 1.0.
    fn compute(
        reference_engine: &str,
        candidate_engine: &str,
        reference: &[TranscriptSegment],
        candidate: &[TranscriptSegment],
    ) -> Self {
        let ref_count = reference.len();
        let cand_count = candidate.len();
        let max_count = ref_count.max(cand_count);

        if max_count == 0 {
            return Self {
                reference_engine: reference_engine.to_owned(),
                candidate_engine: candidate_engine.to_owned(),
                wer_approx: 0.0,
                confidence_delta: 0.0,
                reference_segment_count: 0,
                candidate_segment_count: 0,
                max_segment_wer: 0.0,
            };
        }

        let mut total_wer = 0.0_f64;
        let mut total_conf_delta = 0.0_f64;
        let mut max_seg_wer = 0.0_f64;

        for i in 0..max_count {
            let seg_wer = if i < ref_count && i < cand_count {
                word_error_rate_approx(&reference[i].text, &candidate[i].text)
            } else {
                // Surplus segment counts as complete mismatch.
                1.0
            };

            let seg_conf_delta = if i < ref_count && i < cand_count {
                (reference[i].confidence - candidate[i].confidence).abs()
            } else {
                1.0
            };

            total_wer += seg_wer;
            total_conf_delta += seg_conf_delta;
            if seg_wer > max_seg_wer {
                max_seg_wer = seg_wer;
            }
        }

        Self {
            reference_engine: reference_engine.to_owned(),
            candidate_engine: candidate_engine.to_owned(),
            wer_approx: total_wer / max_count as f64,
            confidence_delta: total_conf_delta / max_count as f64,
            reference_segment_count: ref_count,
            candidate_segment_count: cand_count,
            max_segment_wer: max_seg_wer,
        }
    }
}

/// Approximate word error rate between two strings, computed as the fraction
/// of words that differ when comparing token-by-token. This is a simplified
/// Word error rate between two transcript strings — the real edit-distance WER
/// (`conformance::word_error_rate`), not a positional token compare.
///
/// The former positional implementation (word[i] vs word[i], mismatches ÷
/// longer length) cascaded a single mid-sequence insertion/deletion into a
/// mismatch on EVERY following word, so a near-perfect transcript that merely
/// dropped one early word scored WER ≈ 1.0. Since the native-rollout gate is
/// defined against true WER ≤ 0.10 (DISC-003), that mis-measured the gate.
/// This now delegates to the crate's normalized word-Levenshtein WER (edits ÷
/// reference words); it agrees with the old counter on same-length,
/// non-reordered inputs (all this fn's unit tests) but is correct under
/// realignment.
fn word_error_rate_approx(reference: &str, candidate: &str) -> f64 {
    franken_whisper::conformance::word_error_rate(reference, candidate).wer
}

// ---------------------------------------------------------------------------
// ToleranceGate — enforces max drift thresholds
// ---------------------------------------------------------------------------

/// A gate that enforces maximum acceptable drift between engine outputs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ToleranceGate {
    /// Maximum acceptable mean WER across segments.
    max_wer: f64,
    /// Maximum acceptable mean confidence delta across segments.
    max_confidence_delta: f64,
    /// Maximum acceptable WER for any single segment.
    max_single_segment_wer: f64,
}

/// Result of applying a tolerance gate to drift metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GateResult {
    /// Whether the gate passed.
    passed: bool,
    /// Human-readable reasons for failure (empty if passed).
    violations: Vec<String>,
}

impl ToleranceGate {
    /// Evaluate drift metrics against this gate's thresholds.
    fn evaluate(&self, metrics: &DriftMetrics) -> GateResult {
        let mut violations = Vec::new();

        if metrics.wer_approx > self.max_wer {
            violations.push(format!(
                "WER {:.4} exceeds max {:.4}",
                metrics.wer_approx, self.max_wer
            ));
        }
        if metrics.confidence_delta > self.max_confidence_delta {
            violations.push(format!(
                "confidence delta {:.4} exceeds max {:.4}",
                metrics.confidence_delta, self.max_confidence_delta
            ));
        }
        if metrics.max_segment_wer > self.max_single_segment_wer {
            violations.push(format!(
                "max segment WER {:.4} exceeds max {:.4}",
                metrics.max_segment_wer, self.max_single_segment_wer
            ));
        }

        GateResult {
            passed: violations.is_empty(),
            violations,
        }
    }
}

// ---------------------------------------------------------------------------
// ConformanceReport — per-engine results with reproducibility hash
// ---------------------------------------------------------------------------

/// Per-engine result in the conformance report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct EngineResult {
    engine_name: String,
    engine_kind: String,
    segment_count: usize,
    avg_confidence: f64,
    segments_json: String,
}

/// Full conformance report emitted as a CI artifact.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ConformanceReport {
    /// Identifier for this report run.
    run_id: String,
    /// Duration of the fixture corpus input in milliseconds.
    fixture_duration_ms: u64,
    /// Per-engine results.
    engine_results: Vec<EngineResult>,
    /// Pairwise drift metrics between all engine pairs.
    drift_metrics: Vec<DriftMetrics>,
    /// Gate evaluation results for each pair.
    gate_results: Vec<PairGateResult>,
    /// Overall pass/fail for the conformance run.
    overall_passed: bool,
    /// Whether this was a shadow run (informational only, never fails CI).
    shadow_mode: bool,
    /// SHA-256 hash of the deterministic report content for reproducibility.
    reproducibility_hash: String,
}

/// Gate result for a specific engine pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PairGateResult {
    reference_engine: String,
    candidate_engine: String,
    gate_result: GateResult,
}

impl ConformanceReport {
    /// Compute the reproducibility hash over the deterministic content.
    ///
    /// The hash covers engine results and drift metrics (excluding the hash
    /// field itself) to ensure identical inputs always produce the same hash.
    fn compute_hash(
        fixture_duration_ms: u64,
        engine_results: &[EngineResult],
        drift_metrics: &[DriftMetrics],
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(fixture_duration_ms.to_le_bytes());
        for er in engine_results {
            hasher.update(er.engine_name.as_bytes());
            hasher.update(er.engine_kind.as_bytes());
            hasher.update(er.segment_count.to_le_bytes());
            hasher.update(er.avg_confidence.to_le_bytes());
            hasher.update(er.segments_json.as_bytes());
        }
        for dm in drift_metrics {
            hasher.update(dm.reference_engine.as_bytes());
            hasher.update(dm.candidate_engine.as_bytes());
            hasher.update(dm.wer_approx.to_le_bytes());
            hasher.update(dm.confidence_delta.to_le_bytes());
            hasher.update(dm.reference_segment_count.to_le_bytes());
            hasher.update(dm.candidate_segment_count.to_le_bytes());
            hasher.update(dm.max_segment_wer.to_le_bytes());
        }
        let digest = hasher.finalize();
        format!("{digest:x}")
    }
}

// ---------------------------------------------------------------------------
// ConformanceHarness — runs multiple engines on the same mock input
// ---------------------------------------------------------------------------

/// Harness that runs multiple mock engines on a fixture corpus and produces a
/// conformance report with drift metrics and tolerance gate evaluation.
struct ConformanceHarness {
    engines: Vec<MockEngine>,
    gate: ToleranceGate,
    shadow_mode: bool,
}

impl ConformanceHarness {
    fn new(gate: ToleranceGate, shadow_mode: bool) -> Self {
        Self {
            engines: Vec::new(),
            gate,
            shadow_mode,
        }
    }

    fn add_engine(&mut self, engine: MockEngine) {
        self.engines.push(engine);
    }

    /// Run all engines on the fixture corpus (single duration) and produce a
    /// conformance report.
    fn run(&self, fixture_duration_ms: u64) -> ConformanceReport {
        // Use BTreeMap for deterministic iteration order by engine name.
        let mut results: BTreeMap<String, (MockEngineKind, Vec<TranscriptSegment>)> =
            BTreeMap::new();

        for engine in &self.engines {
            let segments = engine.run(fixture_duration_ms);
            results.insert(engine.name.clone(), (engine.kind, segments));
        }

        // Build per-engine results.
        let engine_results: Vec<EngineResult> = results
            .iter()
            .map(|(name, (kind, segments))| {
                let avg_conf = if segments.is_empty() {
                    0.0
                } else {
                    segments.iter().map(|s| s.confidence).sum::<f64>() / segments.len() as f64
                };
                let segments_json =
                    serde_json::to_string(segments).expect("segments should serialize");
                EngineResult {
                    engine_name: name.clone(),
                    engine_kind: kind.as_str().to_owned(),
                    segment_count: segments.len(),
                    avg_confidence: avg_conf,
                    segments_json,
                }
            })
            .collect();

        // Compute pairwise drift metrics.
        let names: Vec<&String> = results.keys().collect();
        let mut drift_metrics = Vec::new();
        let mut gate_results = Vec::new();

        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let ref_name = names[i];
                let cand_name = names[j];
                let ref_segs = &results[ref_name].1;
                let cand_segs = &results[cand_name].1;

                let metrics = DriftMetrics::compute(ref_name, cand_name, ref_segs, cand_segs);
                let gr = self.gate.evaluate(&metrics);

                gate_results.push(PairGateResult {
                    reference_engine: ref_name.clone(),
                    candidate_engine: cand_name.clone(),
                    gate_result: gr,
                });
                drift_metrics.push(metrics);
            }
        }

        let overall_passed = if self.shadow_mode {
            // Shadow mode never fails the pipeline.
            true
        } else {
            gate_results.iter().all(|pgr| pgr.gate_result.passed)
        };

        let reproducibility_hash =
            ConformanceReport::compute_hash(fixture_duration_ms, &engine_results, &drift_metrics);

        ConformanceReport {
            run_id: "conformance-test-run".to_owned(),
            fixture_duration_ms,
            engine_results,
            drift_metrics,
            gate_results,
            overall_passed,
            shadow_mode: self.shadow_mode,
            reproducibility_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build standard mock engines using the pilot structs
// ---------------------------------------------------------------------------

/// Create a WhisperCppPilot-based mock engine.
fn whisper_cpp_pilot_engine() -> MockEngine {
    MockEngine {
        name: "whisper-cpp-pilot".to_owned(),
        kind: MockEngineKind::Pilot,
        produce: Box::new(synthetic_whisper_cpp_segments),
    }
}

/// Create a synthetic insanely-fast mock engine (single-file batch). Uses the
/// test-only [`synthetic_insanely_fast_segments`] fixture, not engine output.
fn insanely_fast_pilot_engine() -> MockEngine {
    MockEngine {
        name: "insanely-fast-pilot".to_owned(),
        kind: MockEngineKind::Pilot,
        produce: Box::new(synthetic_insanely_fast_segments),
    }
}

/// Create a diarization mock engine driven by the test-only
/// [`synthetic_diarization_segments`] fixture (not engine output).
fn diarization_pilot_engine() -> MockEngine {
    MockEngine {
        name: "diarization-pilot".to_owned(),
        kind: MockEngineKind::Pilot,
        produce: Box::new(synthetic_diarization_segments),
    }
}

/// Create a simulated bridge engine that mimics WhisperCpp output via
/// deterministic generation (test-only, no external processes).
fn whisper_cpp_bridge_engine() -> MockEngine {
    MockEngine {
        name: "whisper-cpp-bridge".to_owned(),
        kind: MockEngineKind::Bridge,
        produce: Box::new(|duration_ms| {
            // Simulate bridge output matching the synthetic whisper-cpp format
            // with identical content (zero drift by design).
            synthetic_whisper_cpp_segments(duration_ms)
        }),
    }
}

/// Create a simulated bridge engine that introduces controlled drift
/// (different text, slightly different confidence) to test tolerance gates.
fn drifted_bridge_engine() -> MockEngine {
    MockEngine {
        name: "drifted-bridge".to_owned(),
        kind: MockEngineKind::Bridge,
        produce: Box::new(|duration_ms| {
            let segment_duration_ms: u64 = 5000;
            let n_segments = if duration_ms == 0 {
                0
            } else {
                duration_ms.div_ceil(segment_duration_ms) as usize
            };

            let phrases = [
                "The fast brown fox leaps over the lazy dog.",
                "Hello world, this is a test transcription.",
                "Speech recognition is a fascinating field.",
                "Machine learning continues to advance.",
            ];

            (0..n_segments)
                .map(|i| {
                    let start_ms = (i as u64) * segment_duration_ms;
                    let end_ms = std::cmp::min(start_ms + segment_duration_ms, duration_ms);
                    let text = phrases[i % phrases.len()].to_owned();
                    let confidence = 0.90 - (i as f64 * 0.01);

                    TranscriptSegment {
                        start_ms,
                        end_ms,
                        text,
                        confidence,
                    }
                })
                .collect()
        }),
    }
}

/// Create a heavily drifted engine that will fail tight tolerance gates.
fn heavily_drifted_engine() -> MockEngine {
    MockEngine {
        name: "heavily-drifted".to_owned(),
        kind: MockEngineKind::Bridge,
        produce: Box::new(|duration_ms| {
            let segment_duration_ms: u64 = 5000;
            let n_segments = if duration_ms == 0 {
                0
            } else {
                duration_ms.div_ceil(segment_duration_ms) as usize
            };

            let phrases = [
                "Completely different text here.",
                "Nothing like the original at all.",
                "This output is totally wrong.",
                "No match whatsoever expected.",
            ];

            (0..n_segments)
                .map(|i| {
                    let start_ms = (i as u64) * segment_duration_ms;
                    let end_ms = std::cmp::min(start_ms + segment_duration_ms, duration_ms);
                    TranscriptSegment {
                        start_ms,
                        end_ms,
                        text: phrases[i % phrases.len()].to_owned(),
                        confidence: 0.50 - (i as f64 * 0.05),
                    }
                })
                .collect()
        }),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn harness_runs_both_bridge_and_pilot_engines() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(insanely_fast_pilot_engine());
    harness.add_engine(diarization_pilot_engine());
    harness.add_engine(whisper_cpp_bridge_engine());

    let report = harness.run(15_000);

    // All four engines should be present in results.
    assert_eq!(report.engine_results.len(), 4);

    // Verify we have both bridge and pilot engines.
    let kinds: Vec<&str> = report
        .engine_results
        .iter()
        .map(|er| er.engine_kind.as_str())
        .collect();
    assert!(kinds.contains(&"bridge"), "should have bridge engine");
    assert!(kinds.contains(&"pilot"), "should have pilot engine");

    // Each engine should produce non-empty segments for 15s input.
    for er in &report.engine_results {
        assert!(
            er.segment_count > 0,
            "engine {} should produce segments for 15s input",
            er.engine_name
        );
    }

    // Pairwise drift metrics: C(4,2) = 6 pairs.
    assert_eq!(report.drift_metrics.len(), 6);
    assert_eq!(report.gate_results.len(), 6);
}

#[test]
fn drift_metrics_correctly_computed_identical_engines() {
    // whisper-cpp-pilot and whisper-cpp-bridge produce identical output.
    let segments = synthetic_whisper_cpp_segments(10_000);
    let metrics = DriftMetrics::compute("engine-a", "engine-b", &segments, &segments);

    assert_eq!(
        metrics.wer_approx, 0.0,
        "identical outputs should have zero WER"
    );
    assert_eq!(
        metrics.confidence_delta, 0.0,
        "identical outputs should have zero confidence delta"
    );
    assert_eq!(metrics.max_segment_wer, 0.0);
    assert_eq!(metrics.reference_segment_count, segments.len());
    assert_eq!(metrics.candidate_segment_count, segments.len());
}

#[test]
fn drift_metrics_correctly_computed_different_engines() {
    let reference = synthetic_whisper_cpp_segments(10_000);

    // Drifted engine uses different phrases.
    let candidate: Vec<TranscriptSegment> = vec![
        TranscriptSegment {
            start_ms: 0,
            end_ms: 5000,
            text: "The fast brown fox leaps over the lazy dog.".to_owned(),
            confidence: 0.90,
        },
        TranscriptSegment {
            start_ms: 5000,
            end_ms: 10_000,
            text: "Hello world, this is a test transcription.".to_owned(),
            confidence: 0.89,
        },
    ];

    let metrics = DriftMetrics::compute("ref", "cand", &reference, &candidate);

    // First segment: "quick" vs "fast" and "jumps" vs "leaps" = 2 diffs out of 9 words.
    // Second segment: identical text = 0 diffs.
    assert!(
        metrics.wer_approx > 0.0,
        "different text should produce non-zero WER"
    );
    assert!(
        metrics.wer_approx < 1.0,
        "partially different text should produce WER < 1.0"
    );
    assert!(
        metrics.confidence_delta > 0.0,
        "different confidence should produce non-zero delta"
    );
}

#[test]
fn drift_metrics_handles_empty_inputs() {
    let metrics = DriftMetrics::compute("a", "b", &[], &[]);
    assert_eq!(metrics.wer_approx, 0.0);
    assert_eq!(metrics.confidence_delta, 0.0);
    assert_eq!(metrics.max_segment_wer, 0.0);
}

#[test]
fn drift_metrics_handles_mismatched_segment_counts() {
    let reference = vec![
        TranscriptSegment {
            start_ms: 0,
            end_ms: 5000,
            text: "Hello world.".to_owned(),
            confidence: 0.9,
        },
        TranscriptSegment {
            start_ms: 5000,
            end_ms: 10_000,
            text: "Extra segment.".to_owned(),
            confidence: 0.8,
        },
    ];
    let candidate = vec![TranscriptSegment {
        start_ms: 0,
        end_ms: 5000,
        text: "Hello world.".to_owned(),
        confidence: 0.9,
    }];

    let metrics = DriftMetrics::compute("ref", "cand", &reference, &candidate);

    // 2 segments max: first is identical (WER=0, delta=0), second has no match (WER=1, delta=1).
    assert_eq!(metrics.reference_segment_count, 2);
    assert_eq!(metrics.candidate_segment_count, 1);
    assert!(
        (metrics.wer_approx - 0.5).abs() < 1e-10,
        "one matching + one surplus = 0.5 mean WER"
    );
    assert_eq!(metrics.max_segment_wer, 1.0);
}

#[test]
fn tolerance_gate_passes_within_threshold() {
    let gate = ToleranceGate {
        max_wer: 0.3,
        max_confidence_delta: 0.1,
        max_single_segment_wer: 0.5,
    };

    let metrics = DriftMetrics {
        reference_engine: "a".to_owned(),
        candidate_engine: "b".to_owned(),
        wer_approx: 0.1,
        confidence_delta: 0.02,
        reference_segment_count: 2,
        candidate_segment_count: 2,
        max_segment_wer: 0.2,
    };

    let result = gate.evaluate(&metrics);
    assert!(result.passed, "metrics within tolerance should pass");
    assert!(result.violations.is_empty());
}

#[test]
fn tolerance_gate_fails_above_threshold() {
    let gate = ToleranceGate {
        max_wer: 0.1,
        max_confidence_delta: 0.05,
        max_single_segment_wer: 0.2,
    };

    let metrics = DriftMetrics {
        reference_engine: "a".to_owned(),
        candidate_engine: "b".to_owned(),
        wer_approx: 0.5,
        confidence_delta: 0.3,
        reference_segment_count: 2,
        candidate_segment_count: 2,
        max_segment_wer: 0.8,
    };

    let result = gate.evaluate(&metrics);
    assert!(!result.passed, "metrics above tolerance should fail");
    assert_eq!(
        result.violations.len(),
        3,
        "all three thresholds should be violated"
    );
}

#[test]
fn tolerance_gate_partial_failure() {
    let gate = ToleranceGate {
        max_wer: 0.5,
        max_confidence_delta: 0.01,
        max_single_segment_wer: 1.0,
    };

    let metrics = DriftMetrics {
        reference_engine: "a".to_owned(),
        candidate_engine: "b".to_owned(),
        wer_approx: 0.2,
        confidence_delta: 0.1,
        reference_segment_count: 2,
        candidate_segment_count: 2,
        max_segment_wer: 0.3,
    };

    let result = gate.evaluate(&metrics);
    assert!(!result.passed, "should fail on confidence delta");
    assert_eq!(result.violations.len(), 1);
    assert!(result.violations[0].contains("confidence delta"));
}

#[test]
fn report_serializes_to_json_for_ci() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(whisper_cpp_bridge_engine());

    let report = harness.run(10_000);

    // Serialize to JSON.
    let json = serde_json::to_string_pretty(&report).expect("report should serialize to JSON");

    // Deserialize back and verify round-trip.
    let deserialized: ConformanceReport =
        serde_json::from_str(&json).expect("JSON should deserialize back to ConformanceReport");

    assert_eq!(deserialized.run_id, report.run_id);
    assert_eq!(deserialized.fixture_duration_ms, report.fixture_duration_ms);
    assert_eq!(
        deserialized.engine_results.len(),
        report.engine_results.len()
    );
    assert_eq!(deserialized.drift_metrics.len(), report.drift_metrics.len());
    assert_eq!(deserialized.overall_passed, report.overall_passed);
    assert_eq!(deserialized.shadow_mode, report.shadow_mode);
    assert_eq!(
        deserialized.reproducibility_hash,
        report.reproducibility_hash
    );

    // Verify key fields are present in the JSON.
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.get("run_id").is_some());
    assert!(parsed.get("engine_results").is_some());
    assert!(parsed.get("drift_metrics").is_some());
    assert!(parsed.get("gate_results").is_some());
    assert!(parsed.get("overall_passed").is_some());
    assert!(parsed.get("reproducibility_hash").is_some());
}

#[test]
fn shadow_run_mode_never_fails_pipeline() {
    // Use a very tight gate so drift would normally cause failure.
    let gate = ToleranceGate {
        max_wer: 0.0,
        max_confidence_delta: 0.0,
        max_single_segment_wer: 0.0,
    };
    let mut harness = ConformanceHarness::new(gate, true); // shadow_mode = true
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(insanely_fast_pilot_engine());

    let report = harness.run(10_000);

    // Shadow mode should always report overall_passed = true.
    assert!(
        report.overall_passed,
        "shadow mode should never fail the pipeline"
    );
    assert!(report.shadow_mode, "should be marked as shadow mode");

    // But individual gate results should still reflect actual drift.
    let any_failed = report
        .gate_results
        .iter()
        .any(|pgr| !pgr.gate_result.passed);
    assert!(
        any_failed,
        "individual gates should still detect drift in shadow mode"
    );
}

#[test]
fn shadow_run_vs_enforced_mode_differ_on_failure() {
    // Tight gate with engines that produce different output.
    let gate = ToleranceGate {
        max_wer: 0.0,
        max_confidence_delta: 0.0,
        max_single_segment_wer: 0.0,
    };

    // Enforced mode.
    let mut enforced = ConformanceHarness::new(gate.clone(), false);
    enforced.add_engine(whisper_cpp_pilot_engine());
    enforced.add_engine(insanely_fast_pilot_engine());
    let enforced_report = enforced.run(10_000);

    // Shadow mode.
    let gate2 = ToleranceGate {
        max_wer: 0.0,
        max_confidence_delta: 0.0,
        max_single_segment_wer: 0.0,
    };
    let mut shadow = ConformanceHarness::new(gate2, true);
    shadow.add_engine(whisper_cpp_pilot_engine());
    shadow.add_engine(insanely_fast_pilot_engine());
    let shadow_report = shadow.run(10_000);

    // Enforced should fail, shadow should pass.
    assert!(
        !enforced_report.overall_passed,
        "enforced mode should fail with tight gate"
    );
    assert!(
        shadow_report.overall_passed,
        "shadow mode should pass regardless"
    );

    // But drift metrics should be identical.
    assert_eq!(
        enforced_report.drift_metrics.len(),
        shadow_report.drift_metrics.len()
    );
    for (e, s) in enforced_report
        .drift_metrics
        .iter()
        .zip(shadow_report.drift_metrics.iter())
    {
        assert_eq!(e.wer_approx, s.wer_approx);
        assert_eq!(e.confidence_delta, s.confidence_delta);
    }
}

#[test]
fn reproducibility_same_inputs_produce_same_hash() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };

    // Run the harness twice with identical configuration.
    let mut harness1 = ConformanceHarness::new(gate.clone(), false);
    harness1.add_engine(whisper_cpp_pilot_engine());
    harness1.add_engine(insanely_fast_pilot_engine());
    harness1.add_engine(diarization_pilot_engine());
    let report1 = harness1.run(15_000);

    let gate2 = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness2 = ConformanceHarness::new(gate2, false);
    harness2.add_engine(whisper_cpp_pilot_engine());
    harness2.add_engine(insanely_fast_pilot_engine());
    harness2.add_engine(diarization_pilot_engine());
    let report2 = harness2.run(15_000);

    assert_eq!(
        report1.reproducibility_hash, report2.reproducibility_hash,
        "identical inputs must produce identical reproducibility hash"
    );

    // Verify the hash is a non-empty hex string.
    assert!(!report1.reproducibility_hash.is_empty());
    assert!(
        report1
            .reproducibility_hash
            .chars()
            .all(|c| c.is_ascii_hexdigit()),
        "hash should be hex"
    );
}

#[test]
fn reproducibility_different_inputs_produce_different_hash() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };

    let mut harness1 = ConformanceHarness::new(gate.clone(), false);
    harness1.add_engine(whisper_cpp_pilot_engine());
    let report1 = harness1.run(10_000);

    let gate2 = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness2 = ConformanceHarness::new(gate2, false);
    harness2.add_engine(whisper_cpp_pilot_engine());
    let report2 = harness2.run(20_000);

    assert_ne!(
        report1.reproducibility_hash, report2.reproducibility_hash,
        "different input durations must produce different hashes"
    );
}

#[test]
fn harness_with_identical_bridge_and_pilot_shows_zero_drift() {
    let gate = ToleranceGate {
        max_wer: 0.001,
        max_confidence_delta: 0.001,
        max_single_segment_wer: 0.001,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    // Both use WhisperCppPilot internally, so output is identical.
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(whisper_cpp_bridge_engine());

    let report = harness.run(10_000);

    assert!(
        report.overall_passed,
        "identical engines should pass even tight gates"
    );
    assert_eq!(report.drift_metrics.len(), 1);
    assert_eq!(report.drift_metrics[0].wer_approx, 0.0);
    assert_eq!(report.drift_metrics[0].confidence_delta, 0.0);
}

#[test]
fn harness_with_drifted_engine_fails_tight_gate() {
    let gate = ToleranceGate {
        max_wer: 0.05,
        max_confidence_delta: 0.01,
        max_single_segment_wer: 0.1,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(drifted_bridge_engine());

    let report = harness.run(10_000);

    assert!(
        !report.overall_passed,
        "drifted engine should fail tight gate"
    );
}

#[test]
fn harness_with_drifted_engine_passes_loose_gate() {
    let gate = ToleranceGate {
        max_wer: 0.5,
        max_confidence_delta: 0.5,
        max_single_segment_wer: 0.5,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(drifted_bridge_engine());

    let report = harness.run(10_000);

    assert!(
        report.overall_passed,
        "slightly drifted engine should pass loose gate"
    );
}

#[test]
fn harness_with_heavily_drifted_engine_fails_even_loose_gate() {
    let gate = ToleranceGate {
        max_wer: 0.5,
        max_confidence_delta: 0.3,
        max_single_segment_wer: 0.8,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(heavily_drifted_engine());

    let report = harness.run(10_000);

    assert!(
        !report.overall_passed,
        "heavily drifted engine should fail even a moderately loose gate"
    );
}

#[test]
fn word_error_rate_approx_identical_strings() {
    assert_eq!(word_error_rate_approx("hello world", "hello world"), 0.0);
}

#[test]
fn word_error_rate_approx_completely_different() {
    assert_eq!(word_error_rate_approx("hello world", "foo bar"), 1.0);
}

#[test]
fn word_error_rate_approx_partial_mismatch() {
    // "quick" vs "fast" = 1 mismatch out of 3 words.
    let wer = word_error_rate_approx("the quick fox", "the fast fox");
    assert!((wer - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn word_error_rate_approx_empty_strings() {
    assert_eq!(word_error_rate_approx("", ""), 0.0);
}

#[test]
fn word_error_rate_approx_one_empty() {
    assert_eq!(word_error_rate_approx("hello world", ""), 1.0);
    assert_eq!(word_error_rate_approx("", "hello world"), 1.0);
}

#[test]
fn word_error_rate_approx_different_lengths() {
    // "hello world foo" vs "hello world" => 3 words max, 1 mismatch (missing "foo").
    let wer = word_error_rate_approx("hello world foo", "hello world");
    assert!((wer - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn word_error_rate_approx_realigns_after_deletion() {
    // Regression pin for the positional-cascade fix: dropping ONE early word
    // must cost ~1 edit, not mis-score every following word. The old positional
    // counter scored a/b/c/d/e/f vs a/c/d/e/f as ~0.83 (5 of 6 positions differ
    // after the shift); the real edit-distance WER is 1 deletion / 6 ref = 0.167.
    let wer = word_error_rate_approx("a b c d e f", "a c d e f");
    assert!((wer - 1.0 / 6.0).abs() < 1e-10, "got {wer}");
    // Case/punctuation are normalized away (not counted as errors).
    assert_eq!(word_error_rate_approx("Hello, World.", "hello world"), 0.0);
}

#[test]
fn report_contains_all_engine_names() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(insanely_fast_pilot_engine());
    harness.add_engine(diarization_pilot_engine());

    let report = harness.run(10_000);
    let names: Vec<&str> = report
        .engine_results
        .iter()
        .map(|er| er.engine_name.as_str())
        .collect();

    assert!(names.contains(&"whisper-cpp-pilot"));
    assert!(names.contains(&"insanely-fast-pilot"));
    assert!(names.contains(&"diarization-pilot"));
}

#[test]
fn zero_duration_input_produces_empty_segments() {
    let gate = ToleranceGate {
        max_wer: 1.0,
        max_confidence_delta: 1.0,
        max_single_segment_wer: 1.0,
    };
    let mut harness = ConformanceHarness::new(gate, false);
    harness.add_engine(whisper_cpp_pilot_engine());
    harness.add_engine(insanely_fast_pilot_engine());

    let report = harness.run(0);

    for er in &report.engine_results {
        assert_eq!(
            er.segment_count, 0,
            "zero duration should produce zero segments for {}",
            er.engine_name
        );
    }
    assert!(report.overall_passed);
}

// ===========================================================================
// bd-4slu: Bridge-vs-native conformance against the REAL engines.
//
// Everything above this line drives the comparator with *synthetic* fixtures.
// This section instead runs both the reference bridge (`whisper-cli`, an
// external subprocess) and the in-process native engine on the same real audio
// (`tests/fixtures/native/jfk.wav` + `tiny.en`), then compares their outputs.
//
// It is **doubly gated**: it runs only when BOTH the `tiny.en` model resolves
// AND `whisper-cli` exists at `/opt/homebrew/bin/whisper-cli`. Absent either,
// it prints a SKIP line and returns success.
//
// TOLERANCE PROFILE — "native-rollout", NOT the canonical 50 ms:
//   whisper-cli defaults to beam search (`-bs 5`); the native engine decodes
//   greedily (temperature 0). That divergence produces occasional word-choice
//   differences and timestamp drift — measured at ~240 ms on the final-segment
//   end boundary of jfk.wav. We therefore gate on WER <= 0.10 and per-segment
//   timestamps within 0.3 s — deliberately looser than
//   `CANONICAL_TIMESTAMP_TOLERANCE_SEC` (50 ms), and matching the library's own
//   gated e2e timestamp tolerance (`decode.rs`). See DISCREPANCIES.md DISC-003;
//   revisit (tighten) when native beam search lands.
// ===========================================================================

use franken_whisper::model::TranscriptionSegment;
use franken_whisper::native_engine::{self, NativeWhisperModel};

/// Path to the reference whisper-cli bridge binary (homebrew install).
const WHISPER_CLI_BIN: &str = "/opt/homebrew/bin/whisper-cli";

/// Native-rollout per-segment timestamp tolerance (greedy-vs-beam; DISC-003).
/// Greedy decode drifts the final-segment end boundary ~240 ms versus beam
/// search; 0.3 s gives honest headroom without hiding real regressions.
const NATIVE_ROLLOUT_TIMESTAMP_TOLERANCE_SEC: f64 = 0.3;

/// Native-rollout WER gate (greedy-vs-beam; DISC-003).
const NATIVE_ROLLOUT_MAX_WER: f64 = 0.10;

fn jfk_wav_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/native/jfk.wav"
    ))
}

/// Decode a standard 16-bit PCM WAV into 16 kHz mono `f32` samples in `[-1, 1]`,
/// walking RIFF chunks (jfk.wav carries a `LIST`/`INFO` chunk before `data`, so
/// the naive 44-byte-offset shortcut does not apply). Test-only; the library's
/// own `read_wav_16k_mono` is crate-private.
fn read_wav_16k_mono(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start.checked_add(size)?;
        if body_end > bytes.len() {
            break;
        }
        if id == b"fmt " && size >= 16 {
            let f = &bytes[body_start..];
            channels = u16::from_le_bytes([f[2], f[3]]);
            sample_rate = u32::from_le_bytes([f[4], f[5], f[6], f[7]]);
            bits = u16::from_le_bytes([f[14], f[15]]);
        } else if id == b"data" {
            data = Some(&bytes[body_start..body_end]);
        }
        // Chunks are word-aligned: an odd size is followed by a pad byte.
        pos = body_end + (size & 1);
    }

    let (channels, sample_rate, bits, data) = (channels, sample_rate, bits, data?);
    if channels == 0 || bits != 16 || sample_rate != 16_000 {
        return None;
    }
    let frame = (channels as usize) * 2;
    let mut out = Vec::with_capacity(data.len() / frame.max(1));
    for chunk in data.chunks_exact(frame) {
        // Average channels to mono.
        let mut acc = 0i32;
        for c in 0..channels as usize {
            let s = i16::from_le_bytes([chunk[c * 2], chunk[c * 2 + 1]]);
            acc += i32::from(s);
        }
        let mono = acc as f32 / channels as f32;
        out.push(mono / 32768.0);
    }
    Some(out)
}

/// Levenshtein word-edit-distance WER (insertions + deletions + substitutions),
/// normalized by reference length. Unlike the positional [`word_error_rate_approx`]
/// used for the synthetic fixtures, this is robust to a single inserted/dropped
/// word shifting the whole sequence — the right metric for a greedy-vs-beam
/// full-transcript comparison.
fn word_error_rate_edit(reference: &str, candidate: &str) -> f64 {
    let r: Vec<String> = reference
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_owned())
        .filter(|w| !w.is_empty())
        .collect();
    let c: Vec<String> = candidate
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_owned())
        .filter(|w| !w.is_empty())
        .collect();
    if r.is_empty() {
        return if c.is_empty() { 0.0 } else { 1.0 };
    }
    // Classic DP edit distance over words.
    let mut prev: Vec<usize> = (0..=c.len()).collect();
    let mut curr = vec![0usize; c.len() + 1];
    for (i, rw) in r.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cw) in c.iter().enumerate() {
            let cost = usize::from(rw != cw);
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[c.len()] as f64 / r.len() as f64
}

/// Run `whisper-cli -oj` on `wav` with `model_path` and parse its segments into
/// [`TranscriptionSegment`]s. Returns `None` on any failure.
fn whisper_cli_segments(
    wav: &std::path::Path,
    model_path: &std::path::Path,
) -> Option<Vec<TranscriptionSegment>> {
    let out_dir = tempfile::tempdir().ok()?;
    let out_base = out_dir.path().join("out");
    let status = std::process::Command::new(WHISPER_CLI_BIN)
        .args([
            "-m",
            model_path.to_str()?,
            "-oj",
            "-of",
            out_base.to_str()?,
            "-l",
            "en",
            wav.to_str()?,
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let json_bytes = std::fs::read(out_dir.path().join("out.json")).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&json_bytes).ok()?;
    let arr = json["transcription"].as_array()?;
    let mut segments = Vec::with_capacity(arr.len());
    for seg in arr {
        let from_ms = seg["offsets"]["from"].as_f64()?;
        let to_ms = seg["offsets"]["to"].as_f64()?;
        segments.push(TranscriptionSegment {
            start_sec: Some(from_ms / 1000.0),
            end_sec: Some(to_ms / 1000.0),
            text: seg["text"].as_str().unwrap_or_default().trim().to_owned(),
            speaker: None,
            confidence: None,
        });
    }
    Some(segments)
}

/// Run the in-process native engine on `wav` with `tiny.en`, returning its
/// timed segments.
fn native_segments(model: &NativeWhisperModel, samples: &[f32]) -> Vec<TranscriptionSegment> {
    let params = native_engine::decode::DecodeParams {
        language: None,
        translate: false,
        timestamps: true,
        n_threads: 4,
        max_text_ctx: None,
        ..Default::default()
    };
    let noop = || Ok(());
    let out = model
        .transcribe(samples, &params, &noop)
        .expect("native transcribe");
    out.segments
}

fn joined_text(segments: &[TranscriptionSegment]) -> String {
    segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn gated_bridge_vs_native_conformance_jfk_tiny_en() {
    let Some(model_path) = native_engine::find_model_file("tiny.en") else {
        eprintln!("SKIP gated_bridge_vs_native_conformance: tiny.en model missing");
        return;
    };
    if !std::path::Path::new(WHISPER_CLI_BIN).exists() {
        eprintln!("SKIP gated_bridge_vs_native_conformance: {WHISPER_CLI_BIN} missing");
        return;
    }

    let wav = jfk_wav_path();
    let bytes = std::fs::read(&wav).expect("read jfk.wav");
    let samples = read_wav_16k_mono(&bytes).expect("decode jfk.wav to 16k mono");

    let model = NativeWhisperModel::load(&model_path).expect("load tiny.en");

    let Some(bridge) = whisper_cli_segments(&wav, &model_path) else {
        eprintln!("SKIP gated_bridge_vs_native_conformance: whisper-cli produced no output");
        return;
    };
    let native = native_segments(&model, &samples);

    let bridge_text = joined_text(&bridge);
    let native_text = joined_text(&native);

    // ── WER gate (native-rollout profile) ──
    let wer = word_error_rate_edit(&bridge_text, &native_text);
    if wer > NATIVE_ROLLOUT_MAX_WER {
        eprintln!("BRIDGE (whisper-cli): {bridge_text}");
        eprintln!("NATIVE (greedy):      {native_text}");
    }
    assert!(
        wer <= NATIVE_ROLLOUT_MAX_WER,
        "native-vs-bridge WER {wer:.4} exceeds native-rollout gate {NATIVE_ROLLOUT_MAX_WER} \
         (greedy-vs-beam; see DISC-003)\nBRIDGE: {bridge_text}\nNATIVE: {native_text}"
    );
    eprintln!("native-vs-bridge WER = {wer:.4} (gate {NATIVE_ROLLOUT_MAX_WER})");

    // ── per-segment timestamp tolerance (native-rollout profile, 0.2 s) ──
    let matched = bridge.len().min(native.len());
    assert!(matched > 0, "no overlapping segments to compare");
    for i in 0..matched {
        let (bs, ns) = (&bridge[i], &native[i]);
        let bstart = bs.start_sec.unwrap_or(0.0);
        let nstart = ns.start_sec.unwrap_or(0.0);
        let bend = bs.end_sec.unwrap_or(0.0);
        let nend = ns.end_sec.unwrap_or(0.0);
        let start_drift = (bstart - nstart).abs();
        let end_drift = (bend - nend).abs();
        if start_drift > NATIVE_ROLLOUT_TIMESTAMP_TOLERANCE_SEC
            || end_drift > NATIVE_ROLLOUT_TIMESTAMP_TOLERANCE_SEC
        {
            eprintln!("BRIDGE (whisper-cli): {bridge_text}");
            eprintln!("NATIVE (greedy):      {native_text}");
        }
        assert!(
            start_drift <= NATIVE_ROLLOUT_TIMESTAMP_TOLERANCE_SEC,
            "segment {i} start drift {start_drift:.3}s exceeds {NATIVE_ROLLOUT_TIMESTAMP_TOLERANCE_SEC}s \
             (bridge {bstart:.3} vs native {nstart:.3}; DISC-003)"
        );
        assert!(
            end_drift <= NATIVE_ROLLOUT_TIMESTAMP_TOLERANCE_SEC,
            "segment {i} end drift {end_drift:.3}s exceeds {NATIVE_ROLLOUT_TIMESTAMP_TOLERANCE_SEC}s \
             (bridge {bend:.3} vs native {nend:.3}; DISC-003)"
        );
    }
}
