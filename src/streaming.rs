//! Speculative cancel-correct streaming pipeline.
//!
//! Orchestrates [`WindowManager`](crate::speculation::WindowManager),
//! [`CorrectionTracker`](crate::speculation::CorrectionTracker), and
//! [`ConcurrentTwoLaneExecutor`](crate::backend::ConcurrentTwoLaneExecutor)
//! to run fast + quality models in parallel with real-time correction.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::audio;
use crate::backend::{ConcurrentTwoLaneExecutor, QualitySelector, TranscriptSegment};
use crate::error::{FwError, FwResult};
use crate::model::{BackendKind, RunEvent, TranscriptionResult, TranscriptionSegment};
use crate::speculation::{
    CorrectionDecision, CorrectionTolerance, CorrectionTracker, PartialTranscript,
    SpeculationStats, SpeculationWindowController, WindowManager,
};

// ---------------------------------------------------------------------------
// bd-qlt.6: SpeculativeStreamingPipeline
// ---------------------------------------------------------------------------

/// Configuration for a speculative streaming run.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    pub window_size_ms: u64,
    pub overlap_ms: u64,
    pub fast_model_name: String,
    pub quality_model_name: String,
    pub tolerance: CorrectionTolerance,
    pub adaptive: bool,
    pub emit_events: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            window_size_ms: 3000,
            overlap_ms: 500,
            fast_model_name: "whisper-tiny".to_owned(),
            quality_model_name: "whisper-large".to_owned(),
            tolerance: CorrectionTolerance::default(),
            adaptive: true,
            emit_events: true,
        }
    }
}

/// Bridge a `TranscriptionSegment` (model) to a `TranscriptSegment` (backend).
fn to_backend_segment(s: &TranscriptionSegment) -> TranscriptSegment {
    TranscriptSegment {
        start_ms: s.start_sec.map(|v| (v * 1000.0) as u64).unwrap_or(0),
        end_ms: s.end_sec.map(|v| (v * 1000.0) as u64).unwrap_or(0),
        text: s.text.clone(),
        confidence: s.confidence.unwrap_or(0.0),
    }
}

fn bridge_and_store_segments(
    holder: &Mutex<Vec<TranscriptionSegment>>,
    original: Vec<TranscriptionSegment>,
) -> Vec<TranscriptSegment> {
    let bridged = original.iter().map(to_backend_segment).collect();
    *holder.lock().unwrap_or_else(|error| error.into_inner()) = original;
    bridged
}

fn take_stored_segments(holder: &Mutex<Vec<TranscriptionSegment>>) -> Vec<TranscriptionSegment> {
    let mut stored = holder.lock().unwrap_or_else(|error| error.into_inner());
    std::mem::take(&mut *stored)
}

/// The speculative streaming pipeline orchestrator.
///
/// Rather than holding engine references directly, the pipeline accepts
/// closures for fast and quality model inference. This keeps it testable
/// with deterministic mocks and ready for later engine-object integration.
pub struct SpeculativeStreamingPipeline {
    config: SpeculativeConfig,
    window_manager: WindowManager,
    correction_tracker: CorrectionTracker,
    adaptive_controller: Option<SpeculationWindowController>,
    next_seq: AtomicU64,
    events: Vec<RunEvent>,
    run_id: String,
}

impl SpeculativeStreamingPipeline {
    /// Create a new pipeline with the given configuration.
    #[must_use]
    pub fn new(config: SpeculativeConfig, run_id: String) -> Self {
        let window_manager = WindowManager::new(&run_id, config.window_size_ms, config.overlap_ms);
        let correction_tracker = CorrectionTracker::new(config.tolerance.clone());
        let adaptive_controller = config
            .adaptive
            .then(|| SpeculationWindowController::new(config.window_size_ms, 1000, 30_000, 500));
        Self {
            config,
            window_manager,
            correction_tracker,
            adaptive_controller,
            next_seq: AtomicU64::new(0),
            events: Vec::new(),
            run_id,
        }
    }

    fn next_seq(&self) -> u64 {
        self.next_seq.fetch_add(1, Ordering::Relaxed)
    }

    fn push_event(&mut self, code: &str, message: &str, payload: serde_json::Value) {
        self.events.push(RunEvent {
            seq: self.events.len() as u64,
            ts_rfc3339: chrono::Utc::now().to_rfc3339(),
            stage: "speculation".to_owned(),
            code: code.to_owned(),
            message: message.to_owned(),
            payload,
        });
    }

    fn apply_adaptive_window_update(&mut self, decision: &CorrectionDecision) {
        let Some(controller) = self.adaptive_controller.as_mut() else {
            return;
        };

        let drift = match decision {
            CorrectionDecision::Confirm { drift, .. } => drift,
            CorrectionDecision::Correct { correction } => &correction.drift,
        };

        controller.observe(decision, drift);
        let new_window_size = controller.apply();
        self.window_manager.set_window_size(new_window_size);
    }

    fn process_window_by_id<F, Q>(
        &mut self,
        window_id: u64,
        fast_fn: F,
        quality_fn: Q,
    ) -> FwResult<CorrectionDecision>
    where
        F: FnOnce() -> Vec<TranscriptionSegment> + Send + 'static,
        Q: FnOnce() -> Vec<TranscriptionSegment> + Send + 'static,
    {
        let seq = self.next_seq();
        let fast_model_name = self.config.fast_model_name.clone();
        let quality_model_name = self.config.quality_model_name.clone();

        let executor = ConcurrentTwoLaneExecutor::new(QualitySelector::SpeculativeCorrect);

        // Capture original model segments directly in bridge closures to avoid
        // precision-lossy round-trip (model → backend → model) that turns
        // `confidence: None` into `Some(0.0)` and truncates sub-ms timestamps.
        let fast_holder: Arc<Mutex<Vec<TranscriptionSegment>>> = Arc::new(Mutex::new(Vec::new()));
        let quality_holder: Arc<Mutex<Vec<TranscriptionSegment>>> =
            Arc::new(Mutex::new(Vec::new()));
        let fast_holder_bridge = fast_holder.clone();
        let quality_holder_bridge = quality_holder.clone();

        let fast_bridge = move || -> Vec<TranscriptSegment> {
            bridge_and_store_segments(fast_holder_bridge.as_ref(), fast_fn())
        };
        let quality_bridge = move || -> Vec<TranscriptSegment> {
            bridge_and_store_segments(quality_holder_bridge.as_ref(), quality_fn())
        };

        let result = executor.execute_with_early_emit(
            fast_bridge,
            quality_bridge,
            |_primary_result, _latency_ms| {},
            |_primary, _secondary, _p_lat, _q_lat| {},
        );

        let fast_segments = take_stored_segments(fast_holder.as_ref());

        // Register with tracker and window manager.
        let fast_ts = chrono::Utc::now().to_rfc3339();
        let partial = PartialTranscript::new(
            seq,
            window_id,
            fast_model_name,
            fast_segments.clone(),
            result.primary_latency_ms,
            fast_ts.clone(),
        );
        self.correction_tracker.register_partial(partial);

        let fast_partial_for_wm = PartialTranscript::new(
            seq,
            window_id,
            self.config.fast_model_name.clone(),
            fast_segments.clone(),
            result.primary_latency_ms,
            fast_ts.clone(),
        );
        self.window_manager
            .record_fast_result(window_id, fast_partial_for_wm);

        if self.config.emit_events {
            for segment in &fast_segments {
                let payload =
                    crate::robot::transcript_partial_value(&self.run_id, seq, &fast_ts, segment);
                self.push_event(
                    "transcript.partial",
                    "fast model emitted speculative partial segment",
                    payload,
                );
            }
        }

        // Use captured original model segments (no round-trip conversion).
        let quality_segments = take_stored_segments(quality_holder.as_ref());
        self.window_manager
            .record_quality_result(window_id, quality_segments.clone());

        // Submit to correction tracker.
        let decision = self.correction_tracker.submit_quality_result(
            window_id,
            &quality_model_name,
            quality_segments,
            result.secondary_latency_ms,
        )?;

        self.window_manager.resolve_window(window_id);

        if self.config.emit_events {
            match &decision {
                CorrectionDecision::Confirm { seq, drift } => {
                    let payload = crate::robot::transcript_confirm_value(
                        &self.run_id,
                        *seq,
                        window_id,
                        drift,
                        result.secondary_latency_ms,
                        &self.config.quality_model_name,
                    );
                    self.push_event(
                        "transcript.confirm",
                        "quality model confirmed speculative transcript",
                        payload,
                    );
                }
                CorrectionDecision::Correct { correction } => {
                    let retract_payload = crate::robot::transcript_retract_value(
                        &self.run_id,
                        correction.retracted_seq,
                        correction.window_id,
                        "quality_correction",
                        &correction.quality_model_id,
                    );
                    self.push_event(
                        "transcript.retract",
                        "quality model retracted speculative transcript",
                        retract_payload,
                    );

                    let correct_payload =
                        crate::robot::transcript_correct_value(&self.run_id, correction);
                    self.push_event(
                        "transcript.correct",
                        "quality model emitted correction transcript",
                        correct_payload,
                    );
                }
            }
        }

        self.apply_adaptive_window_update(&decision);

        Ok(decision)
    }

    /// Process a single window using provided model closures.
    ///
    /// Both closures are run in parallel on separate threads via
    /// [`ConcurrentTwoLaneExecutor`]. The fast model result is captured
    /// via the early-emit callback, then compared with the quality result
    /// to produce a [`CorrectionDecision`].
    pub fn process_window<F, Q>(
        &mut self,
        audio_hash: &str,
        audio_position_ms: u64,
        fast_fn: F,
        quality_fn: Q,
    ) -> FwResult<CorrectionDecision>
    where
        F: FnOnce() -> Vec<TranscriptionSegment> + Send + 'static,
        Q: FnOnce() -> Vec<TranscriptionSegment> + Send + 'static,
    {
        let window = self
            .window_manager
            .next_window(audio_position_ms, audio_hash);
        self.process_window_by_id(window.window_id, fast_fn, quality_fn)
    }

    fn process_duration_loop<C, M>(
        &mut self,
        total_duration_ms: u64,
        audio_hash_seed: &str,
        checkpoint: &mut C,
        model_runner: &mut M,
    ) -> FwResult<TranscriptionResult>
    where
        C: FnMut() -> FwResult<()>,
        M: FnMut(u64, u64) -> FwResult<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)>,
    {
        if total_duration_ms == 0 {
            checkpoint()?;
            let result = self.build_result();
            if self.config.emit_events {
                let stats = self.stats();
                self.push_event(
                    "transcript.speculation_stats",
                    "speculative pipeline aggregate statistics",
                    crate::robot::speculation_stats_value(&self.run_id, &stats),
                );
            }
            return Ok(result);
        }

        let mut position_ms = 0u64;
        while position_ms < total_duration_ms {
            checkpoint()?;

            let window_size_ms = self.window_manager.current_window_size().max(1);
            let step_ms = window_size_ms.saturating_sub(self.config.overlap_ms).max(1);

            let audio_hash = format!("{audio_hash_seed}:{position_ms}:{window_size_ms}");
            let Some(window) = self.window_manager.next_window_bounded(
                position_ms,
                total_duration_ms,
                &audio_hash,
            ) else {
                break;
            };

            let (fast_segments, quality_segments) = model_runner(window.start_ms, window.end_ms)?;
            self.process_window_by_id(
                window.window_id,
                move || fast_segments,
                move || quality_segments,
            )?;

            if window.end_ms >= total_duration_ms {
                break;
            }

            position_ms = position_ms.saturating_add(step_ms);
        }

        let result = self.build_result();
        if self.config.emit_events {
            let stats = self.stats();
            self.push_event(
                "transcript.speculation_stats",
                "speculative pipeline aggregate statistics",
                crate::robot::speculation_stats_value(&self.run_id, &stats),
            );
        }
        Ok(result)
    }

    /// Process an audio duration by repeatedly invoking model callbacks for each
    /// speculation window.
    pub fn process_duration_with_models<C, M>(
        &mut self,
        total_duration_ms: u64,
        audio_hash_seed: &str,
        mut checkpoint: C,
        mut model_runner: M,
    ) -> FwResult<TranscriptionResult>
    where
        C: FnMut() -> FwResult<()>,
        M: FnMut(u64, u64) -> FwResult<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)>,
    {
        self.process_duration_loop(
            total_duration_ms,
            audio_hash_seed,
            &mut checkpoint,
            &mut model_runner,
        )
    }

    /// Convenience wrapper with no cancellation hook.
    pub fn process_duration_with_models_no_checkpoint<M>(
        &mut self,
        total_duration_ms: u64,
        audio_hash_seed: &str,
        model_runner: M,
    ) -> FwResult<TranscriptionResult>
    where
        M: FnMut(u64, u64) -> FwResult<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)>,
    {
        self.process_duration_with_models(
            total_duration_ms,
            audio_hash_seed,
            || Ok(()),
            model_runner,
        )
    }

    /// Process an audio file by probing duration and invoking model callbacks
    /// for each bounded speculation window.
    pub fn process_file_with_models<C, M>(
        &mut self,
        audio_path: &Path,
        checkpoint: C,
        mut model_runner: M,
    ) -> FwResult<TranscriptionResult>
    where
        C: FnMut() -> FwResult<()>,
        M: FnMut(
            &Path,
            u64,
            u64,
        ) -> FwResult<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)>,
    {
        let duration_sec =
            audio::probe_duration_seconds_with_timeout(audio_path, Duration::from_secs(10))
                .ok_or_else(|| {
                    FwError::InvalidRequest(format!(
                        "failed to probe audio duration for {}",
                        audio_path.display()
                    ))
                })?;
        let total_duration_ms = (duration_sec * 1000.0).round() as u64;
        let hash_seed = audio_path.display().to_string();
        self.process_duration_with_models(
            total_duration_ms,
            &hash_seed,
            checkpoint,
            |start_ms, end_ms| model_runner(audio_path, start_ms, end_ms),
        )
    }

    /// Get merged, deduplicated transcript from all resolved windows.
    #[must_use]
    pub fn merged_transcript(&self) -> Vec<TranscriptionSegment> {
        self.window_manager.merge_segments()
    }

    /// Build a full `TranscriptionResult` from the pipeline state.
    #[must_use]
    pub fn build_result(&self) -> TranscriptionResult {
        let segments = self.merged_transcript();
        let transcript = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        TranscriptionResult {
            backend: BackendKind::Auto,
            transcript,
            language: Some("en".to_owned()),
            segments,
            acceleration: None,
            raw_output: serde_json::json!({}),
            artifact_paths: vec![],
        }
    }

    /// Get speculation statistics.
    #[must_use]
    pub fn stats(&self) -> SpeculationStats {
        let tracker_stats = self.correction_tracker.stats();
        SpeculationStats {
            windows_processed: tracker_stats.windows_processed,
            corrections_emitted: tracker_stats.corrections_emitted,
            confirmations_emitted: tracker_stats.confirmations_emitted,
            correction_rate: self.correction_tracker.correction_rate(),
            mean_fast_latency_ms: if tracker_stats.windows_processed > 0 {
                tracker_stats.total_fast_latency_ms as f64 / tracker_stats.windows_processed as f64
            } else {
                0.0
            },
            mean_quality_latency_ms: if tracker_stats.windows_processed > 0 {
                tracker_stats.total_quality_latency_ms as f64
                    / tracker_stats.windows_processed as f64
            } else {
                0.0
            },
            current_window_size_ms: self.window_manager.current_window_size(),
            mean_drift_wer: self.correction_tracker.mean_wer(),
        }
    }

    /// All events generated by the speculative pipeline.
    #[must_use]
    pub fn events(&self) -> &[RunEvent] {
        &self.events
    }

    /// Reference to the correction tracker.
    #[must_use]
    pub fn correction_tracker(&self) -> &CorrectionTracker {
        &self.correction_tracker
    }

    /// Reference to the window manager.
    #[must_use]
    pub fn window_manager(&self) -> &WindowManager {
        &self.window_manager
    }

    /// Run ID for this pipeline.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(
        text: &str,
        start: Option<f64>,
        end: Option<f64>,
        conf: Option<f64>,
    ) -> TranscriptionSegment {
        TranscriptionSegment {
            text: text.to_owned(),
            start_sec: start,
            end_sec: end,
            confidence: conf,
            speaker: None,
        }
    }

    fn historical_bridge_and_recover(
        original: Vec<TranscriptionSegment>,
    ) -> (Vec<TranscriptSegment>, Vec<TranscriptionSegment>) {
        let holder = Mutex::new(Vec::new());
        *holder.lock().unwrap_or_else(|error| error.into_inner()) = original.clone();
        let bridged = original.iter().map(to_backend_segment).collect();
        let recovered = holder
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        (bridged, recovered)
    }

    fn transferred_bridge_and_recover(
        original: Vec<TranscriptionSegment>,
    ) -> (Vec<TranscriptSegment>, Vec<TranscriptionSegment>) {
        let holder = Mutex::new(Vec::new());
        let bridged = bridge_and_store_segments(&holder, original);
        let recovered = take_stored_segments(&holder);
        debug_assert!(
            holder
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty()
        );
        (bridged, recovered)
    }

    #[test]
    fn to_backend_segment_converts_seconds_to_ms_and_defaults() {
        // Normal conversion: seconds → ms.
        let s = seg("hello", Some(1.5), Some(2.75), Some(0.9));
        let bs = to_backend_segment(&s);
        assert_eq!(bs.start_ms, 1500);
        assert_eq!(bs.end_ms, 2750);
        assert_eq!(bs.text, "hello");
        assert!((bs.confidence - 0.9).abs() < f64::EPSILON);

        // None fields → 0.
        let s2 = seg("world", None, None, None);
        let bs2 = to_backend_segment(&s2);
        assert_eq!(bs2.start_ms, 0);
        assert_eq!(bs2.end_ms, 0);
        assert!((bs2.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bridge_segment_ownership_transfer_matches_historical_bytes() {
        let cases = [
            Vec::new(),
            vec![TranscriptionSegment {
                start_sec: Some(0.125),
                end_sec: Some(1.875),
                text: " λ \\\"quoted\\\"\n🎧".to_owned(),
                speaker: Some("speaker-a".to_owned()),
                confidence: Some(0.975),
            }],
            vec![
                TranscriptionSegment {
                    start_sec: None,
                    end_sec: Some(-0.0),
                    text: String::new(),
                    speaker: None,
                    confidence: None,
                },
                TranscriptionSegment {
                    start_sec: Some(3.25),
                    end_sec: None,
                    text: "multi-byte Καλημέρα 世界".to_owned(),
                    speaker: Some(String::new()),
                    confidence: Some(-0.0),
                },
            ],
        ];

        for (index, segments) in cases.iter().enumerate() {
            let historical = historical_bridge_and_recover(segments.clone());
            let transferred = transferred_bridge_and_recover(segments.clone());
            assert_eq!(
                transferred.0, historical.0,
                "case {index} backend bridge parity"
            );
            assert_eq!(
                serde_json::to_vec(&transferred.1).expect("serialize transferred segments"),
                serde_json::to_vec(&historical.1).expect("serialize historical segments"),
                "case {index} original segment byte parity"
            );
        }
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn bridge_segment_ownership_transfer_perf() {
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 21;
        const CALIBRATION_ITERATIONS: usize = 64;
        const TARGET_ARM_NS: u128 = 30_000_000;

        fn fixture_segments(lane: &str) -> Vec<TranscriptionSegment> {
            (0..12)
                .map(|index| TranscriptionSegment {
                    start_sec: Some(f64::from(index) * 0.24),
                    end_sec: Some(f64::from(index) * 0.24 + 0.22),
                    text: format!(
                        " {lane} streaming segment {index:02}: ownership transfer preserves the original UTF-8 transcript λ 🎧"
                    ),
                    speaker: None,
                    confidence: Some(0.78 + f64::from(index) / 100.0),
                })
                .collect()
        }

        fn prepared_inputs(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> Vec<(Vec<TranscriptionSegment>, Vec<TranscriptionSegment>)> {
            (0..iterations)
                .map(|_| (fast.to_vec(), quality.to_vec()))
                .collect()
        }

        fn time_historical(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> u128 {
            let inputs = prepared_inputs(fast, quality, iterations);
            let started = Instant::now();
            for (fast_segments, quality_segments) in inputs {
                let fast_result = historical_bridge_and_recover(fast_segments);
                let quality_result = historical_bridge_and_recover(quality_segments);
                drop(black_box((fast_result, quality_result)));
            }
            started.elapsed().as_nanos()
        }

        fn time_transferred(
            fast: &[TranscriptionSegment],
            quality: &[TranscriptionSegment],
            iterations: usize,
        ) -> u128 {
            let inputs = prepared_inputs(fast, quality, iterations);
            let started = Instant::now();
            for (fast_segments, quality_segments) in inputs {
                let fast_result = transferred_bridge_and_recover(fast_segments);
                let quality_result = transferred_bridge_and_recover(quality_segments);
                drop(black_box((fast_result, quality_result)));
            }
            started.elapsed().as_nanos()
        }

        fn percentile(values: &[f64], percentile: usize) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[(sorted.len() - 1) * percentile / 100]
        }

        fn median_ns(values: &[u128]) -> u128 {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        }

        fn coefficient_of_variation(values: &[u128]) -> f64 {
            let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (values.len() - 1) as f64;
            variance.sqrt() / mean
        }

        let fast = fixture_segments("fast");
        let quality = fixture_segments("quality");
        let historical = (
            historical_bridge_and_recover(fast.clone()),
            historical_bridge_and_recover(quality.clone()),
        );
        let transferred = (
            transferred_bridge_and_recover(fast.clone()),
            transferred_bridge_and_recover(quality.clone()),
        );
        let historical_bytes = serde_json::to_vec(&historical).expect("serialize historical");
        let transferred_bytes = serde_json::to_vec(&transferred).expect("serialize transferred");
        assert_eq!(transferred_bytes, historical_bytes, "fixture byte parity");

        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "stream_bridge_transfer binary_sha256={:x} rows_per_lane={} output_bytes={} output_sha256={:x}",
            Sha256::digest(executable),
            fast.len(),
            historical_bytes.len(),
            Sha256::digest(&historical_bytes),
        );

        let calibration = time_historical(&fast, &quality, CALIBRATION_ITERATIONS);
        let iterations = ((TARGET_ARM_NS * CALIBRATION_ITERATIONS as u128) / calibration)
            .clamp(128, 4_096) as usize;
        eprintln!(
            "stream_bridge_transfer calibration_iterations={CALIBRATION_ITERATIONS} calibration_ns={calibration} iterations={iterations}"
        );

        for _ in 0..3 {
            black_box(time_historical(&fast, &quality, iterations));
            black_box(time_transferred(&fast, &quality, iterations));
        }

        let mut null_ratios = Vec::with_capacity(SAMPLES);
        let mut speedups = Vec::with_capacity(SAMPLES);
        let mut historical_times = Vec::with_capacity(SAMPLES);
        let mut transferred_times = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let null_first = time_historical(&fast, &quality, iterations);
            let null_second = time_historical(&fast, &quality, iterations);
            let (numerator, denominator) = if sample % 2 == 0 {
                (null_first, null_second)
            } else {
                (null_second, null_first)
            };
            null_ratios.push(numerator as f64 / denominator as f64);

            let (historical, transferred) = if sample % 2 == 0 {
                (
                    time_historical(&fast, &quality, iterations),
                    time_transferred(&fast, &quality, iterations),
                )
            } else {
                let transferred = time_transferred(&fast, &quality, iterations);
                let historical = time_historical(&fast, &quality, iterations);
                (historical, transferred)
            };
            historical_times.push(historical);
            transferred_times.push(transferred);
            speedups.push(historical as f64 / transferred as f64);
        }

        let null_p10 = percentile(&null_ratios, 10);
        let null_median = percentile(&null_ratios, 50);
        let null_p90 = percentile(&null_ratios, 90);
        let speedup_p10 = percentile(&speedups, 10);
        let speedup_median = percentile(&speedups, 50);
        let speedup_p90 = percentile(&speedups, 90);
        let wins = speedups.iter().filter(|ratio| **ratio > 1.0).count();
        eprintln!(
            "stream_bridge_transfer samples={SAMPLES} iterations={iterations} null_p10={null_p10:.6} null_median={null_median:.6} null_p90={null_p90:.6} historical_arm_median_ns={} transferred_arm_median_ns={} transferred_cv={:.4}% speedup_p10={speedup_p10:.6} speedup_median={speedup_median:.6} speedup_p90={speedup_p90:.6} wins={wins}/{SAMPLES}",
            median_ns(&historical_times),
            median_ns(&transferred_times),
            coefficient_of_variation(&transferred_times) * 100.0,
        );
        eprintln!(
            "stream_bridge_transfer null_ratios={null_ratios:?} speedups={speedups:?} historical_times_ns={historical_times:?} transferred_times_ns={transferred_times:?}"
        );

        assert!(
            (0.95..=1.05).contains(&null_median),
            "null median {null_median:.6} outside predeclared guard",
        );
        assert!(
            speedup_p10 > null_p90.max(1.10),
            "candidate p10 {speedup_p10:.6} did not clear max(null p90 {null_p90:.6}, 1.10)",
        );
        assert!(
            wins >= 18,
            "candidate won {wins}/{SAMPLES}; predeclared gate requires at least 18",
        );
    }

    #[test]
    fn poisoned_mutex_recovers_via_into_inner() {
        // Verify the pattern used in process_window_by_id: if one lane panics
        // while holding a Mutex, the other lane (and the caller) can still
        // recover the stored data via `unwrap_or_else(|e| e.into_inner())`.
        let holder: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));
        let h = holder.clone();

        // Poison the mutex by panicking while holding the lock.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = h.lock().unwrap();
            guard.push(42);
            panic!("intentional panic to poison the mutex");
        }));
        assert!(result.is_err(), "closure should have panicked");

        // The mutex is now poisoned — .lock() returns Err.
        assert!(holder.lock().is_err());

        // But unwrap_or_else(|e| e.into_inner()) recovers the data.
        let recovered = holder.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            *recovered,
            vec![42],
            "data should be recoverable from poisoned mutex"
        );
    }

    #[test]
    fn speculative_config_defaults() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.window_size_ms, 3000);
        assert_eq!(cfg.overlap_ms, 500);
        assert_eq!(cfg.fast_model_name, "whisper-tiny");
        assert_eq!(cfg.quality_model_name, "whisper-large");
        assert!(cfg.adaptive);
        assert!(cfg.emit_events);
        // Tolerance defaults.
        assert!((cfg.tolerance.max_wer - 0.1).abs() < f64::EPSILON);
        assert!(!cfg.tolerance.always_correct);
    }

    #[test]
    fn pipeline_fresh_build_result_is_empty() {
        let pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-run".to_owned());
        let result = pipeline.build_result();
        assert!(result.transcript.is_empty());
        assert!(result.segments.is_empty());
        assert_eq!(result.backend, BackendKind::Auto);
        assert_eq!(result.language, Some("en".to_owned()));
        assert_eq!(pipeline.run_id(), "test-run");
    }

    #[test]
    fn pipeline_process_window_identical_segments_confirms() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-confirm".to_owned(),
        );
        let segments = vec![seg("hello world", Some(0.0), Some(1.0), Some(0.95))];
        let fast_segs = segments.clone();
        let quality_segs = segments;

        let decision = pipeline
            .process_window("hash1", 0, move || fast_segs, move || quality_segs)
            .expect("process_window should succeed");

        assert!(matches!(decision, CorrectionDecision::Confirm { .. }));
        // Stats should reflect one window processed, zero corrections.
        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 1);
        assert_eq!(stats.corrections_emitted, 0);
        assert_eq!(stats.confirmations_emitted, 1);
        // Events should include transcript.partial and transcript.confirm.
        let events = pipeline.events();
        assert!(events.iter().any(|e| e.code == "transcript.partial"));
        assert!(events.iter().any(|e| e.code == "transcript.confirm"));
    }

    #[test]
    fn pipeline_zero_duration_returns_immediately() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-zero".to_owned());
        let result = pipeline
            .process_duration_with_models_no_checkpoint(0, "seed", |_start, _end| {
                panic!("model_runner should not be called for zero duration");
            })
            .expect("should succeed");
        assert!(result.transcript.is_empty());
        // Should have emitted speculation_stats event.
        assert!(
            pipeline
                .events()
                .iter()
                .any(|e| e.code == "transcript.speculation_stats")
        );
    }

    // ── Task #207 — streaming pass 2 edge-case tests ────────────────

    #[test]
    fn event_seq_numbers_are_contiguous_across_windows() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-seq".to_owned());
        // Process two windows.
        for i in 0..2 {
            let s = vec![seg("word", Some(0.0), Some(1.0), Some(0.9))];
            let f = s.clone();
            let q = s;
            let hash = format!("h{i}");
            pipeline
                .process_window(&hash, i * 1000, move || f, move || q)
                .unwrap();
        }
        let events = pipeline.events();
        assert!(
            events.len() >= 4,
            "expected at least 4 events, got {}",
            events.len()
        );
        // Verify seq values are 0, 1, 2, 3, ...
        for (idx, event) in events.iter().enumerate() {
            assert_eq!(event.seq, idx as u64, "event {idx} has wrong seq");
        }
    }

    #[test]
    fn emit_events_false_suppresses_all_events() {
        let config = SpeculativeConfig {
            emit_events: false,
            ..SpeculativeConfig::default()
        };

        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-no-events".to_owned());

        let s = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let f = s.clone();
        let q = s;
        pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        assert!(
            pipeline.events().is_empty(),
            "no events should be emitted when emit_events is false"
        );
    }

    #[test]
    fn correction_emits_retract_and_correct_events() {
        let tolerance = CorrectionTolerance {
            always_correct: true, // force correction
            ..CorrectionTolerance::default()
        };
        let config = SpeculativeConfig {
            tolerance,
            ..SpeculativeConfig::default()
        };

        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-correct".to_owned());

        let fast = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let f = fast;
        let q = quality;
        let decision = pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        let is_correct = matches!(decision, CorrectionDecision::Correct { .. });
        assert!(is_correct, "always_correct should force Correct decision");

        let events = pipeline.events();
        assert!(
            events.iter().any(|e| e.code == "transcript.retract"),
            "should emit transcript.retract"
        );
        assert!(
            events.iter().any(|e| e.code == "transcript.correct"),
            "should emit transcript.correct"
        );
    }

    #[test]
    fn build_result_joins_segments_with_space() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-join".to_owned());

        let result = pipeline
            .process_duration_with_models_no_checkpoint(
                5000, // 5 seconds → at least one window
                "seed",
                |_start, _end| {
                    let segs = vec![seg("hello world", Some(0.0), Some(1.0), Some(0.9))];
                    Ok((segs.clone(), segs))
                },
            )
            .unwrap();

        assert!(
            !result.transcript.is_empty(),
            "transcript should not be empty"
        );
        assert!(!result.segments.is_empty(), "segments should not be empty");
        assert_eq!(result.language, Some("en".to_owned()));
        assert_eq!(result.backend, BackendKind::Auto);
    }

    #[test]
    fn checkpoint_error_cancels_duration_loop() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-cancel".to_owned(),
        );

        let result = pipeline.process_duration_with_models(
            10_000,
            "seed",
            || Err(crate::error::FwError::Cancelled("cancelled".to_owned())),
            |_start, _end| {
                panic!("model_runner should not be called when checkpoint fails");
            },
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("cancelled"),
            "error should mention cancelled: {msg}"
        );
    }

    // ── Task #215 — streaming pass 3 edge-case tests ────────────────

    #[test]
    fn stats_zero_windows_all_fields_zero() {
        let pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-stats".to_owned(),
        );
        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 0);
        assert_eq!(stats.corrections_emitted, 0);
        assert_eq!(stats.confirmations_emitted, 0);
        assert!((stats.correction_rate - 0.0).abs() < 1e-9);
        assert!((stats.mean_fast_latency_ms - 0.0).abs() < 1e-9);
        assert!((stats.mean_quality_latency_ms - 0.0).abs() < 1e-9);
        assert!((stats.mean_drift_wer - 0.0).abs() < 1e-9);
        assert_eq!(stats.current_window_size_ms, 3000);
    }

    #[test]
    fn merged_transcript_and_accessor_consistency() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-accessor".to_owned(),
        );
        let s = vec![seg("hello world", Some(0.0), Some(1.0), Some(0.95))];
        let f = s.clone();
        let q = s;
        pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        // merged_transcript() directly returns segments from resolved windows.
        let merged = pipeline.merged_transcript();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "hello world");

        // correction_tracker() and window_manager() accessors are consistent.
        assert_eq!(pipeline.correction_tracker().stats().windows_processed, 1);
        assert_eq!(pipeline.window_manager().current_window_size(), 3000);
    }

    #[test]
    fn model_runner_error_propagates_through_duration_loop() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-model-err".to_owned(),
        );
        let result =
            pipeline.process_duration_with_models_no_checkpoint(5000, "seed", |_start, _end| {
                Err(FwError::InvalidRequest("model failure".to_owned()))
            });
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("model failure"),
            "error should propagate model failure: {msg}"
        );
    }

    #[test]
    fn emit_events_false_suppresses_duration_loop_stats() {
        let config = SpeculativeConfig {
            emit_events: false,
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-no-stats".to_owned());

        // Nonzero duration → runs the loop, end-of-loop stats guard tested.
        let result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "seed", |_start, _end| {
                let segs = vec![seg("hi", Some(0.0), Some(1.0), Some(0.8))];
                Ok((segs.clone(), segs))
            })
            .unwrap();
        assert!(!result.transcript.is_empty());
        assert!(
            pipeline.events().is_empty(),
            "emit_events=false should suppress all events including end-of-loop stats"
        );

        // Also test zero-duration path.
        let config2 = SpeculativeConfig {
            emit_events: false,
            ..SpeculativeConfig::default()
        };
        let mut pipeline2 =
            SpeculativeStreamingPipeline::new(config2, "test-no-stats-zero".to_owned());
        let _ = pipeline2
            .process_duration_with_models_no_checkpoint(0, "seed", |_s, _e| {
                panic!("should not be called");
            })
            .unwrap();
        assert!(
            pipeline2.events().is_empty(),
            "emit_events=false + zero duration should suppress early-return stats"
        );
    }

    #[test]
    fn to_backend_segment_sub_millisecond_truncation() {
        // Sub-ms value: 0.0009s → truncates to 0ms.
        let s1 = seg("a", Some(0.0009), Some(1.9999), Some(0.0));
        let bs1 = to_backend_segment(&s1);
        assert_eq!(bs1.start_ms, 0, "0.0009s should truncate to 0ms");
        assert_eq!(bs1.end_ms, 1999, "1.9999s should truncate to 1999ms");
        assert!(
            (bs1.confidence - 0.0).abs() < 1e-9,
            "confidence Some(0.0) → 0.0"
        );

        // Large value precision.
        let s2 = seg("b", Some(3599.999), None, None);
        let bs2 = to_backend_segment(&s2);
        assert_eq!(bs2.start_ms, 3_599_999);
        assert_eq!(bs2.end_ms, 0, "None end_sec → 0");
    }

    // ── Task #220 — streaming pass 4 edge-case tests ────────────────

    #[test]
    fn stats_after_one_window_has_finite_mean_latencies() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-mean-lat".to_owned(),
        );
        let s = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let f = s.clone();
        let q = s;
        pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 1);
        assert!(
            stats.mean_fast_latency_ms.is_finite(),
            "mean_fast_latency_ms must be finite, got {}",
            stats.mean_fast_latency_ms
        );
        assert!(
            stats.mean_quality_latency_ms.is_finite(),
            "mean_quality_latency_ms must be finite, got {}",
            stats.mean_quality_latency_ms
        );
        assert!(stats.mean_fast_latency_ms >= 0.0);
        assert!(stats.mean_quality_latency_ms >= 0.0);
    }

    #[test]
    fn zero_duration_checkpoint_error_propagates() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-zero-cancel".to_owned(),
        );

        let result = pipeline.process_duration_with_models(
            0,
            "seed",
            || Err(FwError::Cancelled("early abort".to_owned())),
            |_start, _end| panic!("model_runner must not be called"),
        );

        assert!(result.is_err(), "checkpoint error must propagate");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("early abort"),
            "error should carry checkpoint message: {msg}"
        );
        assert!(
            pipeline.events().is_empty(),
            "no events should be emitted before checkpoint fails"
        );
    }

    #[test]
    fn empty_fast_segments_emits_no_partial_events_but_still_decides() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-empty-fast".to_owned(),
        );

        let quality = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let decision = pipeline
            .process_window("h", 0, std::vec::Vec::new, move || quality)
            .unwrap();

        let events = pipeline.events();
        assert!(
            !events.iter().any(|e| e.code == "transcript.partial"),
            "no partial events when fast model returns empty segments"
        );
        let has_decision_event = events
            .iter()
            .any(|e| e.code == "transcript.confirm" || e.code == "transcript.correct");
        assert!(has_decision_event, "decision event must still be emitted");
        assert!(matches!(
            decision,
            CorrectionDecision::Confirm { .. } | CorrectionDecision::Correct { .. }
        ));
    }

    #[test]
    fn build_result_multi_segment_join_inserts_spaces_between_segments() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-multi-join".to_owned(),
        );

        let s1 = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let s2 = vec![seg("world", Some(1.0), Some(2.0), Some(0.9))];
        let f1 = s1.clone();
        let q1 = s1;
        let f2 = s2.clone();
        let q2 = s2;

        pipeline
            .process_window("h1", 0, move || f1, move || q1)
            .unwrap();
        pipeline
            .process_window("h2", 1000, move || f2, move || q2)
            .unwrap();

        let result = pipeline.build_result();
        assert_eq!(result.segments.len(), 2, "expected 2 merged segments");
        assert_eq!(
            result.transcript, "hello world",
            "segments must be joined with a single space"
        );
    }

    #[test]
    fn no_checkpoint_multiple_windows_all_processed() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-multi-nc".to_owned(),
        );

        let result = pipeline
            .process_duration_with_models_no_checkpoint(6000, "seed", |_start, _end| {
                let s = vec![seg("word", Some(0.0), Some(1.0), Some(0.9))];
                Ok((s.clone(), s))
            })
            .unwrap();

        let stats = pipeline.stats();
        assert!(
            stats.windows_processed >= 2,
            "6000ms with 3000ms windows should process >= 2 windows, got {}",
            stats.windows_processed
        );
        assert_eq!(result.backend, BackendKind::Auto);
        let stats_events: Vec<_> = pipeline
            .events()
            .iter()
            .filter(|e| e.code == "transcript.speculation_stats")
            .collect();
        assert_eq!(
            stats_events.len(),
            1,
            "exactly one end-of-loop stats event expected"
        );
    }

    // ── Task #225 — streaming pass 5 edge-case tests ────────────────

    #[test]
    fn stats_after_forced_correction_has_nonzero_rate_and_wer() {
        let config = SpeculativeConfig {
            tolerance: CorrectionTolerance {
                always_correct: true,
                ..CorrectionTolerance::default()
            },
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-rate".to_owned());

        let fast = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("world", Some(0.0), Some(1.0), Some(0.8))];
        pipeline
            .process_window("h", 0, move || fast, move || quality)
            .unwrap();

        let stats = pipeline.stats();
        assert_eq!(stats.corrections_emitted, 1);
        assert_eq!(stats.confirmations_emitted, 0);
        assert!(
            (stats.correction_rate - 1.0).abs() < 1e-9,
            "correction_rate should be 1.0, got {}",
            stats.correction_rate
        );
        assert!(
            stats.mean_drift_wer > 0.0,
            "mean_drift_wer should be positive after correction, got {}",
            stats.mean_drift_wer
        );
    }

    #[test]
    fn all_events_have_stage_speculation() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-stage".to_owned(),
        );

        let s = vec![seg("test", Some(0.0), Some(1.0), Some(0.9))];
        let f = s.clone();
        let q = s;
        pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        let events = pipeline.events();
        assert!(!events.is_empty());
        for event in events {
            assert_eq!(
                event.stage, "speculation",
                "event '{}' has wrong stage: '{}'",
                event.code, event.stage
            );
        }
    }

    #[test]
    fn checkpoint_error_on_second_iteration_stops_after_first_window() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-cancel2".to_owned(),
        );

        let mut call_count = 0u32;
        let result = pipeline.process_duration_with_models(
            10_000,
            "seed",
            || {
                call_count += 1;
                if call_count >= 2 {
                    Err(FwError::Cancelled("second abort".to_owned()))
                } else {
                    Ok(())
                }
            },
            |_start, _end| {
                let s = vec![seg("partial", Some(0.0), Some(1.0), Some(0.9))];
                Ok((s.clone(), s))
            },
        );

        assert!(
            result.is_err(),
            "should return error from second checkpoint"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("second abort"), "wrong error: {msg}");
        assert_eq!(
            pipeline.stats().windows_processed,
            1,
            "exactly one window should have been processed"
        );
    }

    #[test]
    fn overlap_exceeding_window_size_clamps_step_and_terminates() {
        let config = SpeculativeConfig {
            window_size_ms: 500,
            overlap_ms: 1000,
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-overlap".to_owned());

        let result = pipeline.process_duration_with_models_no_checkpoint(600, "seed", |_s, _e| {
            let s = vec![seg("x", Some(0.0), Some(0.5), Some(0.8))];
            Ok((s.clone(), s))
        });
        assert!(
            result.is_ok(),
            "should terminate even with oversized overlap"
        );
        assert!(pipeline.stats().windows_processed >= 1);
    }

    #[test]
    fn build_result_fixed_fields_are_constant_sentinels() {
        let pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-fixed".to_owned(),
        );
        let result = pipeline.build_result();

        assert!(
            result.acceleration.is_none(),
            "acceleration should always be None"
        );
        assert_eq!(
            result.raw_output,
            serde_json::json!({}),
            "raw_output should be empty JSON object"
        );
        assert!(
            result.artifact_paths.is_empty(),
            "artifact_paths should be empty"
        );
    }

    // ── Task #230 — streaming.rs pass 6 edge-case tests ────────────────

    #[test]
    fn to_backend_segment_some_zero_seconds_produces_zero_ms_distinct_from_none() {
        // Some(0.0) goes through the `(v * 1000.0) as u64` path (not `unwrap_or(0)`).
        let s = seg("zero", Some(0.0), Some(0.0), Some(0.0));
        let bs = to_backend_segment(&s);
        assert_eq!(bs.start_ms, 0, "Some(0.0) start should produce 0 ms");
        assert_eq!(bs.end_ms, 0, "Some(0.0) end should produce 0 ms");
        assert!((bs.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn duration_loop_non_multiple_total_produces_clamped_final_window() {
        // window=3000, overlap=500, total=4000
        // Step = 3000 - 500 = 2500
        // Window 1: (0, 3000); position advances to 2500
        // Window 2: (2500, 4000) — clamped by next_window_bounded
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: false,
                ..SpeculativeConfig::default()
            },
            "test-non-multiple".to_owned(),
        );
        let mut call_count = 0u64;
        let mut bounds = Vec::new();

        let _result = pipeline
            .process_duration_with_models_no_checkpoint(4000, "h", |start, end| {
                call_count += 1;
                bounds.push((start, end));
                Ok((vec![], vec![]))
            })
            .expect("should succeed");

        assert_eq!(call_count, 2, "should process exactly 2 windows");
        assert_eq!(bounds[0], (0, 3000), "first window: 0..3000");
        assert_eq!(
            bounds[1],
            (2500, 4000),
            "second window: 2500..4000 (clamped)"
        );
    }

    #[test]
    fn empty_quality_segments_with_nonempty_fast_triggers_decision() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: true,
                ..SpeculativeConfig::default()
            },
            "test-empty-quality".to_owned(),
        );

        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "h", |_start, _end| {
                // Fast returns one segment, quality returns empty.
                Ok((
                    vec![seg("hello world", Some(0.0), Some(1.0), Some(0.9))],
                    vec![],
                ))
            })
            .expect("should succeed");

        // There should be at least one decision event (confirm or correct).
        let events = pipeline.events();
        let decision_events: Vec<_> = events
            .iter()
            .filter(|e| {
                e.code == "transcript.confirm"
                    || e.code == "transcript.correct"
                    || e.code == "transcript.retract"
            })
            .collect();
        assert!(
            !decision_events.is_empty(),
            "empty quality with non-empty fast should still produce a decision event"
        );
    }

    #[test]
    fn stats_after_single_confirm_has_zero_correction_rate_and_one_confirm() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: false,
                ..SpeculativeConfig::default()
            },
            "test-confirm-stats".to_owned(),
        );

        // Both fast and quality return the same text → guaranteed confirm.
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "h", |_start, _end| {
                Ok((
                    vec![seg("same text", Some(0.0), Some(1.0), Some(0.9))],
                    vec![seg("same text", Some(0.0), Some(1.0), Some(0.9))],
                ))
            })
            .expect("should succeed");

        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 1, "exactly 1 window processed");
        assert_eq!(stats.confirmations_emitted, 1, "should be 1 confirmation");
        assert_eq!(stats.corrections_emitted, 0, "should be 0 corrections");
        assert!(
            (stats.correction_rate - 0.0).abs() < 1e-9,
            "correction rate should be 0.0, got {}",
            stats.correction_rate
        );
    }

    #[test]
    fn event_seq_is_positional_index_across_multiple_windows() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: true,
                ..SpeculativeConfig::default()
            },
            "test-seq-index".to_owned(),
        );

        // Process two windows (total 6000ms with default 3000ms window, 500ms overlap).
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(6000, "h", |_start, _end| {
                Ok((
                    vec![seg("hello", Some(0.0), Some(1.0), Some(0.8))],
                    vec![seg("hello", Some(0.0), Some(1.0), Some(0.8))],
                ))
            })
            .expect("should succeed");

        let events = pipeline.events();
        // Verify event seq values are strictly sequential (0, 1, 2, ...).
        for (i, event) in events.iter().enumerate() {
            assert_eq!(
                event.seq, i as u64,
                "event {i} should have seq={i}, got seq={}",
                event.seq
            );
        }
        // With 2+ windows and events, there should be more than 2 events.
        assert!(
            events.len() >= 4,
            "two windows should generate at least 4 events (2 partial + 2 decision), got {}",
            events.len()
        );
    }

    #[test]
    fn to_backend_segment_negative_seconds_saturates_to_zero() {
        // Negative f64 cast to u64 saturates to 0 in Rust (no panic).
        let s = seg("neg", Some(-5.0), Some(-0.001), Some(0.5));
        let bs = to_backend_segment(&s);
        assert_eq!(bs.start_ms, 0, "negative start should saturate to 0");
        assert_eq!(bs.end_ms, 0, "negative end should saturate to 0");
        assert_eq!(bs.text, "neg");
    }

    #[test]
    fn duration_loop_exact_multiple_breaks_on_boundary() {
        // window_size=3000, overlap=0 → step=3000.
        // total_duration=6000 → exactly 2 windows: [0,3000) and [3000,6000).
        // The second window.end_ms == total_duration_ms triggers the >= break.
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                window_size_ms: 3000,
                overlap_ms: 0,
                emit_events: false,
                ..SpeculativeConfig::default()
            },
            "test-exact-mult".to_owned(),
        );

        let mut call_count = 0u64;
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(6000, "h", |_start, _end| {
                call_count += 1;
                let s = vec![seg("w", Some(0.0), Some(1.0), Some(0.8))];
                Ok((s.clone(), s))
            })
            .expect("should succeed");

        assert_eq!(
            call_count, 2,
            "exactly 2 windows should be processed for 6000ms / 3000ms"
        );
        assert_eq!(pipeline.stats().windows_processed, 2);
    }

    #[test]
    fn correction_emits_two_events_while_confirm_emits_one() {
        // Corrections produce an extra "correction" event on top of the "partial" event.
        // First window: identical fast/quality → confirm (1 partial + 1 confirm = 2 events).
        // Second window: different fast/quality → correction (1 partial + 1 correction = 2 events).
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                window_size_ms: 3000,
                overlap_ms: 0,
                emit_events: true,
                tolerance: CorrectionTolerance {
                    max_wer: 0.0, // very strict — any difference triggers correction
                    max_confidence_delta: 0.0,
                    max_edit_distance: 0,
                    always_correct: false,
                },
                ..SpeculativeConfig::default()
            },
            "test-two-events".to_owned(),
        );

        let mut call_count = 0u64;
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(6000, "h", |_start, _end| {
                call_count += 1;
                if call_count == 1 {
                    // Identical → confirm
                    let s = vec![seg("same", Some(0.0), Some(1.0), Some(0.9))];
                    Ok((s.clone(), s))
                } else {
                    // Different → correction
                    Ok((
                        vec![seg("fast text", Some(0.0), Some(1.0), Some(0.5))],
                        vec![seg("quality text", Some(0.0), Some(1.0), Some(0.95))],
                    ))
                }
            })
            .expect("should succeed");

        let events = pipeline.events();
        // Verify at least some events were generated.
        assert!(
            events.len() >= 4,
            "2 windows with events should produce at least 4 events, got {}",
            events.len()
        );
        // Verify all seq values are sequential.
        for (i, event) in events.iter().enumerate() {
            assert_eq!(event.seq, i as u64, "seq should be positional");
        }
    }

    #[test]
    fn stats_mean_latencies_are_zero_when_zero_latency_windows() {
        // When model_runner returns instantly (0ms latency mock), means should be 0.0 not NaN.
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: false,
                ..SpeculativeConfig::default()
            },
            "test-zero-lat".to_owned(),
        );

        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "seed", |_start, _end| {
                let s = vec![seg("word", Some(0.0), Some(1.0), Some(0.9))];
                Ok((s.clone(), s))
            })
            .expect("should succeed");

        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 1, "should have 1 window");
        // Latencies come from the tracker, which sums up latencies from register_partial and
        // submit_quality_result. With the mock closures, these may be nonzero (measured internally).
        // But crucially the result is finite — no NaN.
        assert!(
            stats.mean_fast_latency_ms.is_finite(),
            "mean fast latency should be finite, got {}",
            stats.mean_fast_latency_ms
        );
        assert!(
            stats.mean_quality_latency_ms.is_finite(),
            "mean quality latency should be finite, got {}",
            stats.mean_quality_latency_ms
        );
    }

    #[test]
    fn to_backend_segment_fractional_millisecond_truncates_not_rounds() {
        // (0.0009 * 1000.0) as u64 = 0.9 → truncates to 0, not rounds to 1.
        let s = seg("sub", Some(0.0009), Some(1.9999), Some(0.5));
        let bs = to_backend_segment(&s);
        assert_eq!(bs.start_ms, 0, "0.9ms should truncate to 0");
        assert_eq!(bs.end_ms, 1999, "1999.9ms should truncate to 1999");
    }

    #[test]
    fn multi_segment_fast_model_emits_multiple_partial_events() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: true,
                ..SpeculativeConfig::default()
            },
            "test-multi-seg".to_owned(),
        );

        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "h", |_start, _end| {
                let segs = vec![
                    seg("hello", Some(0.0), Some(0.5), Some(0.9)),
                    seg("world", Some(0.5), Some(1.0), Some(0.9)),
                ];
                Ok((segs.clone(), segs))
            })
            .expect("should succeed");

        let partial_count = pipeline
            .events()
            .iter()
            .filter(|e| e.code == "transcript.partial")
            .count();
        assert_eq!(
            partial_count, 2,
            "two fast segments should emit 2 partial events, got {partial_count}"
        );
    }

    #[test]
    fn push_event_seq_matches_vec_index_with_multi_segment_window() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: true,
                ..SpeculativeConfig::default()
            },
            "test-seq-multi".to_owned(),
        );

        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "h", |_start, _end| {
                let segs = vec![
                    seg("a", Some(0.0), Some(0.5), Some(0.9)),
                    seg("b", Some(0.5), Some(1.0), Some(0.9)),
                ];
                Ok((segs.clone(), segs))
            })
            .expect("should succeed");

        // 2 partial events + 1 confirm event + 1 stats event = 4 events.
        let events = pipeline.events();
        assert_eq!(events.len(), 4, "should have exactly 4 events");
        assert_eq!(events[0].seq, 0);
        assert_eq!(events[0].code, "transcript.partial");
        assert_eq!(events[1].seq, 1);
        assert_eq!(events[1].code, "transcript.partial");
        assert_eq!(events[2].seq, 2);
        assert_eq!(events[2].code, "transcript.confirm");
        assert_eq!(events[3].seq, 3);
        assert_eq!(events[3].code, "transcript.speculation_stats");
    }

    #[test]
    fn correction_tracker_all_resolved_after_duration_loop() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: false,
                ..SpeculativeConfig::default()
            },
            "test-all-resolved".to_owned(),
        );

        let _result = pipeline
            .process_duration_with_models_no_checkpoint(6000, "h", |_start, _end| {
                let s = vec![seg("word", Some(0.0), Some(1.0), Some(0.9))];
                Ok((s.clone(), s))
            })
            .expect("should succeed");

        assert!(
            pipeline.correction_tracker().all_resolved(),
            "all partials should be resolved after the loop completes"
        );
    }

    #[test]
    fn window_manager_resolved_count_matches_stats_windows_processed() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: false,
                ..SpeculativeConfig::default()
            },
            "test-resolved-count".to_owned(),
        );

        let _result = pipeline
            .process_duration_with_models_no_checkpoint(6000, "h", |_start, _end| {
                let s = vec![seg("word", Some(0.0), Some(1.0), Some(0.9))];
                Ok((s.clone(), s))
            })
            .expect("should succeed");

        let resolved = pipeline.window_manager().windows_resolved();
        let processed = pipeline.stats().windows_processed;
        assert!(processed >= 2, "should process at least 2 windows");
        assert_eq!(
            resolved, processed as usize,
            "windows_resolved should match windows_processed"
        );
        assert_eq!(
            pipeline.window_manager().windows_pending(),
            0,
            "no windows should be pending after completion"
        );
    }

    #[test]
    fn process_file_with_models_returns_error_for_nonexistent_path() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-nonexistent".to_owned(),
        );

        let result = pipeline.process_file_with_models(
            std::path::Path::new("/nonexistent/audio_file_12345.wav"),
            || Ok(()),
            |_path, _s, _e| panic!("model runner should not be called"),
        );

        assert!(result.is_err(), "should fail for nonexistent file");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("failed to probe audio duration"),
            "error should mention probe failure, got: {err}"
        );
    }

    // -- bd-245: streaming.rs edge-case tests pass 9 --

    #[test]
    fn push_event_message_field_stored_correctly() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "test-msg".to_owned());

        let fast = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("hello", Some(0.0), Some(1.0), Some(0.95))];
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();

        let events = pipeline.events();
        // First event should be partial emission with a descriptive message.
        assert!(
            !events[0].message.is_empty(),
            "event message should not be empty"
        );
        assert_eq!(events[0].stage, "speculation");
    }

    #[test]
    fn duration_loop_single_window_exactly_fills_duration() {
        let cfg = SpeculativeConfig {
            window_size_ms: 3000,
            overlap_ms: 0,
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(cfg, "test-single".to_owned());

        let fast = vec![seg("one", Some(0.0), Some(3.0), Some(0.9))];
        let quality = vec![seg("one", Some(0.0), Some(3.0), Some(0.95))];
        let result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();

        let stats = pipeline.stats();
        assert_eq!(
            stats.windows_processed, 1,
            "exactly one window should be processed when duration == window_size"
        );
        assert!(!result.transcript.is_empty());
    }

    #[test]
    fn stats_event_payload_contains_run_id() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "my-run-42".to_owned());

        let fast = vec![seg("hi", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("hi", Some(0.0), Some(1.0), Some(0.95))];
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();

        let events = pipeline.events();
        let stats_event = events
            .iter()
            .find(|e| e.code == "transcript.speculation_stats")
            .expect("should have a stats event");
        assert_eq!(
            stats_event.payload["run_id"], "my-run-42",
            "stats payload should contain the run_id"
        );
    }

    #[test]
    fn pipeline_with_adaptive_false_still_processes() {
        let cfg = SpeculativeConfig {
            adaptive: false,
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(cfg, "test-no-adapt".to_owned());

        let fast = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("hello", Some(0.0), Some(1.0), Some(0.95))];
        let result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();

        assert!(!result.transcript.is_empty(), "should still produce output");
        assert!(
            pipeline.stats().windows_processed > 0,
            "should have processed windows"
        );
    }

    #[test]
    fn adaptive_pipeline_clamps_window_to_min_after_many_confirmations() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-adapt".to_owned(),
        );

        let fast = vec![seg("steady", Some(0.0), Some(1.0), Some(0.9))];
        let quality = fast.clone();

        for i in 0..20 {
            let decision = pipeline.process_window(
                &format!("hash-{i}"),
                i * 2500,
                {
                    let fast = fast.clone();
                    move || fast
                },
                {
                    let quality = quality.clone();
                    move || quality
                },
            );
            assert!(decision.is_ok(), "window {i} should process successfully");
        }

        assert_eq!(
            pipeline.window_manager().current_window_size(),
            1000,
            "adaptive controller should keep shrinking until it reaches the configured minimum"
        );
        assert_eq!(
            pipeline.stats().current_window_size_ms,
            1000,
            "reported stats should reflect the final adapted window size"
        );
    }

    #[test]
    fn all_event_stages_are_speculation() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-stages".to_owned(),
        );

        let fast = vec![seg("a", Some(0.0), Some(1.0), Some(0.8))];
        let quality = vec![seg("b", Some(0.0), Some(1.0), Some(0.9))];
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();

        for event in pipeline.events() {
            assert_eq!(
                event.stage, "speculation",
                "all events should have stage 'speculation', got: {}",
                event.stage
            );
        }
    }

    #[test]
    fn retract_event_payload_contains_run_id_and_quality_model() {
        let config = SpeculativeConfig {
            tolerance: CorrectionTolerance {
                always_correct: true,
                ..CorrectionTolerance::default()
            },
            quality_model_name: "quality-v1".to_owned(),
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "run-retract-test".to_owned());
        let fast = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("hello", Some(0.0), Some(1.0), Some(0.95))];
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();
        let retract = pipeline
            .events()
            .iter()
            .find(|e| e.code == "transcript.retract");
        assert!(retract.is_some(), "should have a retract event");
        let payload = &retract.unwrap().payload;
        assert_eq!(payload["run_id"], "run-retract-test");
        assert_eq!(payload["quality_model_id"], "quality-v1");
    }

    #[test]
    fn build_result_single_segment_no_extra_whitespace() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "single-seg".to_owned(),
        );
        let fast = vec![seg("  hello  ", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("  hello  ", Some(0.0), Some(1.0), Some(0.95))];
        let result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();
        // Single segment: join(" ") on a one-element vec yields the bare text.
        assert_eq!(result.transcript, "  hello  ");
        assert_eq!(result.segments.len(), 1);
    }

    #[test]
    fn correct_event_payload_contains_correction_struct() {
        let config = SpeculativeConfig {
            tolerance: CorrectionTolerance {
                always_correct: true,
                ..CorrectionTolerance::default()
            },
            ..SpeculativeConfig::default()
        };
        let mut pipeline =
            SpeculativeStreamingPipeline::new(config, "run-correct-payload".to_owned());
        let fast = vec![seg("abc", Some(0.0), Some(1.0), Some(0.5))];
        let quality = vec![seg("xyz", Some(0.0), Some(1.0), Some(0.9))];
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();
        let correct = pipeline
            .events()
            .iter()
            .find(|e| e.code == "transcript.correct");
        assert!(correct.is_some(), "should have a correct event");
        let payload = &correct.unwrap().payload;
        assert_eq!(payload["run_id"], "run-correct-payload");
        assert!(!payload["correction_id"].is_null());
    }

    #[test]
    fn duration_loop_zero_duration_emits_stats_event() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "zero-dur".to_owned());
        let result = pipeline
            .process_duration_with_models_no_checkpoint(0, "hash", |_s, _e| {
                panic!("model_runner should not be called for 0 duration");
            })
            .unwrap();
        assert!(result.transcript.is_empty());
        let stats_event = pipeline
            .events()
            .iter()
            .find(|e| e.code == "transcript.speculation_stats");
        assert!(
            stats_event.is_some(),
            "zero duration should still emit stats"
        );
        assert_eq!(stats_event.unwrap().payload["run_id"], "zero-dur");
    }

    #[test]
    fn next_seq_increments_independently_of_event_count() {
        let mut pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "seq-test".to_owned());
        // 3 fast segments per window → 3 partial events + 1 confirm event = 4 events/window.
        let fast = vec![
            seg("a", Some(0.0), Some(0.5), Some(0.9)),
            seg("b", Some(0.5), Some(1.0), Some(0.9)),
            seg("c", Some(1.0), Some(1.5), Some(0.9)),
        ];
        let quality = vec![
            seg("a", Some(0.0), Some(0.5), Some(0.95)),
            seg("b", Some(0.5), Some(1.0), Some(0.95)),
            seg("c", Some(1.0), Some(1.5), Some(0.95)),
        ];
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "hash", |_s, _e| {
                Ok((fast.clone(), quality.clone()))
            })
            .unwrap();
        // At least 1 window was processed.
        let stats = pipeline.stats();
        assert!(stats.windows_processed >= 1);
        // Events should be more than windows_processed (each window → multiple events).
        // Events include partials (3 per window) + confirm/correct (1 per window) + stats (1).
        assert!(
            pipeline.events().len() > stats.windows_processed as usize,
            "events {} should exceed windows_processed {}",
            pipeline.events().len(),
            stats.windows_processed
        );
    }

    // ── Task #265 — streaming.rs pass 10 edge-case tests ──────────────

    #[test]
    fn to_backend_segment_none_timestamps_and_confidence_use_defaults() {
        // None start/end → unwrap_or(0), None confidence → unwrap_or(0.0).
        let s = seg("none-vals", None, None, None);
        let bs = to_backend_segment(&s);
        assert_eq!(bs.start_ms, 0, "None start should default to 0");
        assert_eq!(bs.end_ms, 0, "None end should default to 0");
        assert!(
            (bs.confidence - 0.0).abs() < f64::EPSILON,
            "None confidence should default to 0.0"
        );
        assert_eq!(bs.text, "none-vals");
    }

    #[test]
    fn build_result_language_is_en_and_backend_is_auto() {
        let pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-lang-backend".to_owned(),
        );
        let result = pipeline.build_result();
        assert_eq!(
            result.language,
            Some("en".to_owned()),
            "language should be Some(\"en\")"
        );
        assert_eq!(result.backend, BackendKind::Auto, "backend should be Auto");
    }

    #[test]
    fn stats_on_empty_pipeline_returns_zero_latencies_not_nan() {
        let pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-empty-stats".to_owned(),
        );
        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 0);
        assert_eq!(stats.corrections_emitted, 0);
        assert_eq!(stats.confirmations_emitted, 0);
        assert!(
            (stats.mean_fast_latency_ms - 0.0).abs() < f64::EPSILON,
            "mean fast latency should be 0.0 for empty pipeline, got {}",
            stats.mean_fast_latency_ms
        );
        assert!(
            (stats.mean_quality_latency_ms - 0.0).abs() < f64::EPSILON,
            "mean quality latency should be 0.0 for empty pipeline, got {}",
            stats.mean_quality_latency_ms
        );
        assert!(stats.mean_fast_latency_ms.is_finite());
        assert!(stats.mean_quality_latency_ms.is_finite());
    }

    #[test]
    fn merged_transcript_returns_corrected_segment_fields_after_correction() {
        let config = SpeculativeConfig {
            window_size_ms: 3000,
            overlap_ms: 0,
            tolerance: CorrectionTolerance {
                always_correct: true,
                ..CorrectionTolerance::default()
            },
            emit_events: false,
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-merged".to_owned());

        let _result = pipeline
            .process_duration_with_models_no_checkpoint(3000, "h", |_start, _end| {
                Ok((
                    vec![seg("fast text", Some(0.0), Some(1.5), Some(0.5))],
                    vec![seg("quality text", Some(0.1), Some(1.6), Some(0.95))],
                ))
            })
            .unwrap();

        let merged = pipeline.merged_transcript();
        assert_eq!(merged.len(), 1, "should have exactly 1 merged segment");
        // After correction, the quality model's segment should be used.
        assert_eq!(
            merged[0].text, "quality text",
            "corrected text should come from quality model"
        );
        // Verify timing fields are present (from quality segments).
        assert!(merged[0].start_sec.is_some(), "start_sec should be present");
        assert!(merged[0].end_sec.is_some(), "end_sec should be present");
    }

    #[test]
    fn correction_rate_reflects_mixed_confirm_and_correct_across_windows() {
        // 3 windows: 2 confirm + 1 correct → correction_rate = 1/3 ≈ 0.333
        let config = SpeculativeConfig {
            window_size_ms: 3000,
            overlap_ms: 0,
            emit_events: false,
            tolerance: CorrectionTolerance {
                max_wer: 0.0,
                max_confidence_delta: 0.0,
                max_edit_distance: 0,
                always_correct: false,
            },
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-mixed-rate".to_owned());

        let mut call = 0u64;
        let _result = pipeline
            .process_duration_with_models_no_checkpoint(9000, "h", |_start, _end| {
                call += 1;
                if call == 2 {
                    // Window 2: different text → strict tolerance triggers correction.
                    Ok((
                        vec![seg("fast only", Some(0.0), Some(1.0), Some(0.5))],
                        vec![seg("quality only", Some(0.0), Some(1.0), Some(0.9))],
                    ))
                } else {
                    // Windows 1 and 3: identical → confirm.
                    let s = vec![seg("same", Some(0.0), Some(1.0), Some(0.9))];
                    Ok((s.clone(), s))
                }
            })
            .unwrap();

        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 3, "should process 3 windows");
        assert_eq!(stats.confirmations_emitted, 2, "2 confirms");
        assert_eq!(stats.corrections_emitted, 1, "1 correction");
        let expected_rate = 1.0 / 3.0;
        assert!(
            (stats.correction_rate - expected_rate).abs() < 1e-9,
            "correction_rate should be ~0.333, got {}",
            stats.correction_rate
        );
    }

    #[test]
    fn stats_current_window_size_ms_matches_config_default() {
        let config = SpeculativeConfig {
            window_size_ms: 7500,
            ..SpeculativeConfig::default()
        };
        let pipeline = SpeculativeStreamingPipeline::new(config, "test-ws".to_owned());
        let stats = pipeline.stats();
        assert_eq!(
            stats.current_window_size_ms, 7500,
            "current_window_size_ms should reflect configured value"
        );
    }

    #[test]
    fn stats_mean_drift_wer_exact_value_after_identical_windows() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                emit_events: false,
                ..SpeculativeConfig::default()
            },
            "test-wer-exact".to_owned(),
        );
        // Identical fast/quality → WER = 0.0.
        pipeline
            .process_duration_with_models_no_checkpoint(3000, "h", |_s, _e| {
                let s = vec![seg("identical text", Some(0.0), Some(1.0), Some(0.9))];
                Ok((s.clone(), s))
            })
            .unwrap();
        let stats = pipeline.stats();
        assert!(
            stats.mean_drift_wer.abs() < f64::EPSILON,
            "identical transcripts should yield mean_drift_wer 0.0, got {}",
            stats.mean_drift_wer
        );
    }

    #[test]
    fn process_window_with_speaker_field_preserved_in_merged_transcript() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-speaker".to_owned(),
        );
        let fast = vec![TranscriptionSegment {
            text: "hello".to_owned(),
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            confidence: Some(0.9),
            speaker: Some("SPEAKER_01".to_owned()),
        }];
        let quality = fast.clone();
        pipeline
            .process_window("h", 0, move || fast, move || quality)
            .unwrap();
        let merged = pipeline.merged_transcript();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].speaker.as_deref(), Some("SPEAKER_01"));
    }

    #[test]
    fn process_duration_with_models_large_duration_many_windows() {
        let mut pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig {
                window_size_ms: 1000,
                overlap_ms: 0,
                emit_events: false,
                ..SpeculativeConfig::default()
            },
            "test-many".to_owned(),
        );
        let result = pipeline
            .process_duration_with_models_no_checkpoint(10_000, "seed", |_s, _e| {
                let s = vec![seg("w", Some(0.0), Some(1.0), Some(0.8))];
                Ok((s.clone(), s))
            })
            .unwrap();
        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 10, "10000ms / 1000ms = 10 windows");
        assert_eq!(stats.confirmations_emitted, 10);
        assert_eq!(stats.corrections_emitted, 0);
        // All segments should appear in the result.
        assert!(!result.transcript.is_empty());
    }

    #[test]
    fn build_result_acceleration_field_is_none() {
        let pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "test-accel".to_owned(),
        );
        let result = pipeline.build_result();
        assert!(
            result.acceleration.is_none(),
            "acceleration should be None for speculative pipeline"
        );
    }

    // ── Task #298 — streaming.rs pass 9 edge-case tests ────────────

    #[test]
    fn merged_transcript_returns_empty_on_fresh_pipeline() {
        let pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "fresh-merge".to_owned(),
        );
        let merged = pipeline.merged_transcript();
        assert!(
            merged.is_empty(),
            "fresh pipeline merged_transcript() should be empty"
        );
    }

    #[test]
    fn correction_tracker_accessor_returns_zero_state_before_processing() {
        let pipeline =
            SpeculativeStreamingPipeline::new(SpeculativeConfig::default(), "ct-fresh".to_owned());
        let tracker = pipeline.correction_tracker();
        let stats = tracker.stats();
        assert_eq!(stats.windows_processed, 0);
        assert_eq!(stats.corrections_emitted, 0);
        assert_eq!(stats.confirmations_emitted, 0);
        assert!(
            tracker.all_resolved(),
            "empty tracker should be all_resolved"
        );
    }

    #[test]
    fn window_manager_accessor_reflects_config_window_size() {
        let config = SpeculativeConfig {
            window_size_ms: 7000,
            overlap_ms: 1000,
            ..SpeculativeConfig::default()
        };
        let pipeline = SpeculativeStreamingPipeline::new(config, "wm-cfg".to_owned());
        let wm = pipeline.window_manager();
        assert_eq!(
            wm.current_window_size(),
            7000,
            "window_manager should reflect configured window_size_ms"
        );
        assert_eq!(wm.windows_resolved(), 0);
        assert_eq!(wm.windows_pending(), 0);
    }

    #[test]
    fn events_accessor_returns_empty_before_any_processing() {
        let pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "events-fresh".to_owned(),
        );
        assert!(
            pipeline.events().is_empty(),
            "events() should be empty on fresh pipeline"
        );
    }

    #[test]
    fn stats_fresh_pipeline_returns_all_zeros() {
        let pipeline = SpeculativeStreamingPipeline::new(
            SpeculativeConfig::default(),
            "stats-fresh".to_owned(),
        );
        let stats = pipeline.stats();
        assert_eq!(stats.windows_processed, 0);
        assert_eq!(stats.corrections_emitted, 0);
        assert_eq!(stats.confirmations_emitted, 0);
        assert!((stats.correction_rate - 0.0).abs() < f64::EPSILON);
        assert!((stats.mean_fast_latency_ms - 0.0).abs() < f64::EPSILON);
        assert!((stats.mean_quality_latency_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.current_window_size_ms, 3000); // default
        assert!((stats.mean_drift_wer - 0.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // streaming edge-case tests pass 5
    // ------------------------------------------------------------------

    #[test]
    fn emit_events_false_suppresses_correction_events() {
        // Exercises the `if self.config.emit_events` guard on the
        // CorrectionDecision::Correct branch (lines 202, 219-241).
        // The existing test `emit_events_false_suppresses_all_events` only
        // tests the Confirm path (identical fast/quality). This test forces
        // a Correct decision with `always_correct: true` while emit_events
        // is false, verifying that transcript.retract and transcript.correct
        // events are NOT emitted.
        let config = SpeculativeConfig {
            emit_events: false,
            tolerance: CorrectionTolerance {
                always_correct: true,
                ..CorrectionTolerance::default()
            },
            ..SpeculativeConfig::default()
        };

        let mut pipeline =
            SpeculativeStreamingPipeline::new(config, "test-no-correct-events".to_owned());

        let fast = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("corrected hello", Some(0.0), Some(1.0), Some(0.95))];
        let f = fast;
        let q = quality;
        let decision = pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        assert!(
            matches!(decision, CorrectionDecision::Correct { .. }),
            "always_correct should force Correct decision"
        );
        assert!(
            pipeline.events().is_empty(),
            "no events (retract/correct) should be emitted when emit_events is false"
        );
    }

    #[test]
    fn to_backend_segment_nan_start_sec_saturates_to_zero() {
        // Exercises the NaN → u64 conversion at line 55:
        //   start_ms: s.start_sec.map(|v| (v * 1000.0) as u64).unwrap_or(0)
        // In Rust, `f64::NAN as u64` evaluates to 0 (NaN-to-integer saturation).
        let s = TranscriptionSegment {
            text: "nan test".to_owned(),
            start_sec: Some(f64::NAN),
            end_sec: Some(f64::INFINITY),
            confidence: Some(0.5),
            speaker: None,
        };
        let bs = to_backend_segment(&s);
        assert_eq!(bs.start_ms, 0, "NaN start_sec should saturate to 0");
        assert_eq!(
            bs.end_ms,
            u64::MAX,
            "INFINITY end_sec should saturate to u64::MAX"
        );
        assert_eq!(bs.text, "nan test");
    }

    #[test]
    fn process_duration_1ms_creates_single_window() {
        // Exercises the minimal non-zero duration boundary case where
        // total_duration_ms == 1. The window (0, 1) should be created,
        // processed, and the `window.end_ms >= total_duration_ms` break
        // at line 318 fires immediately after the first window.
        let config = SpeculativeConfig {
            window_size_ms: 3000,
            overlap_ms: 500,
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-1ms".to_owned());

        let result = pipeline
            .process_duration_with_models_no_checkpoint(1, "hash", |_start, _end| {
                Ok((
                    vec![seg("tiny", Some(0.0), Some(0.001), Some(0.9))],
                    vec![seg("tiny", Some(0.0), Some(0.001), Some(0.9))],
                ))
            })
            .expect("should succeed with 1ms duration");

        assert_eq!(
            pipeline.stats().windows_processed,
            1,
            "1ms duration should produce exactly 1 window"
        );
        assert!(
            result.transcript.contains("tiny"),
            "transcript should contain model output"
        );
    }

    #[test]
    fn push_event_populates_ts_rfc3339_field() {
        // Verifies that the `ts_rfc3339` field in `RunEvent` (line 99)
        // is populated with a non-empty, validly-formatted RFC 3339 string.
        // No other streaming test ever inspects the timestamp field.
        let config = SpeculativeConfig::default();
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-ts".to_owned());

        let fast = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let quality = vec![seg("hello", Some(0.0), Some(1.0), Some(0.9))];
        let f = fast;
        let q = quality;
        pipeline
            .process_window("h", 0, move || f, move || q)
            .unwrap();

        let events = pipeline.events();
        assert!(
            !events.is_empty(),
            "should have at least one event (confirm)"
        );
        for event in events {
            assert!(
                !event.ts_rfc3339.is_empty(),
                "ts_rfc3339 should not be empty"
            );
            // RFC 3339 timestamps should contain 'T' and '+' or 'Z'.
            assert!(
                event.ts_rfc3339.contains('T'),
                "ts_rfc3339 should be RFC 3339 format: {}",
                event.ts_rfc3339
            );
        }
    }

    #[test]
    fn process_duration_with_window_size_1_and_overlap_0_processes_all_ms() {
        // Exercises the loop with minimal window_size_ms (1) and overlap (0),
        // meaning step_ms = 1. With total_duration_ms = 5, the pipeline
        // should create exactly 5 windows at positions 0, 1, 2, 3, 4.
        let config = SpeculativeConfig {
            window_size_ms: 1,
            overlap_ms: 0,
            adaptive: false,
            ..SpeculativeConfig::default()
        };
        let mut pipeline = SpeculativeStreamingPipeline::new(config, "test-1ms-window".to_owned());

        let result = pipeline
            .process_duration_with_models_no_checkpoint(5, "hash", |start, end| {
                let text = format!("w{start}");
                Ok((
                    vec![seg(
                        &text,
                        Some(start as f64 / 1000.0),
                        Some(end as f64 / 1000.0),
                        Some(0.9),
                    )],
                    vec![seg(
                        &text,
                        Some(start as f64 / 1000.0),
                        Some(end as f64 / 1000.0),
                        Some(0.9),
                    )],
                ))
            })
            .expect("should succeed");

        assert_eq!(
            pipeline.stats().windows_processed,
            5,
            "5ms duration with 1ms windows and 0 overlap should produce 5 windows"
        );
        assert!(
            result.transcript.contains("w0"),
            "transcript should contain output from first window"
        );
    }
}
