#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use franken_whisper::model::*;
use serde_json::json;

/// Return path to the tests/fixtures directory.
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Return path to the tests/fixtures/golden directory.
pub fn golden_dir() -> PathBuf {
    fixtures_dir().join("golden")
}

/// Return path to the tests/mocks directory.
pub fn mocks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("mocks")
}

/// Create a test TranscribeRequest with sensible defaults.
pub fn create_test_request() -> TranscribeRequest {
    TranscribeRequest {
        input: InputSource::File {
            path: PathBuf::from("test.wav"),
        },
        backend: BackendKind::WhisperCpp,
        model: None,
        language: Some("en".to_owned()),
        translate: false,
        diarize: false,
        persist: false,
        db_path: PathBuf::from("/tmp/test.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    }
}

/// Create a test TranscriptionResult with populated segments.
pub fn create_test_result() -> TranscriptionResult {
    TranscriptionResult {
        backend: BackendKind::WhisperCpp,
        transcript: "Hello world. This is a test.".to_owned(),
        language: Some("en".to_owned()),
        segments: vec![
            TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(2.5),
                text: "Hello world.".to_owned(),
                speaker: None,
                confidence: Some(0.95),
            },
            TranscriptionSegment {
                start_sec: Some(2.5),
                end_sec: Some(6.0),
                text: "This is a test.".to_owned(),
                speaker: None,
                confidence: Some(0.88),
            },
        ],
        acceleration: None,
        diarization: None,
        raw_output: json!({"text": "Hello world. This is a test."}),
        artifact_paths: vec![],
    }
}

/// Create a full RunReport for testing.
pub fn create_test_report() -> RunReport {
    RunReport {
        run_id: "run-test-001".to_owned(),
        trace_id: "00000000000000000000000000000001".to_owned(),
        started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
        finished_at_rfc3339: "2026-01-01T00:01:00Z".to_owned(),
        input_path: "/tmp/test.wav".to_owned(),
        normalized_wav_path: "/tmp/normalized.wav".to_owned(),
        request: create_test_request(),
        result: create_test_result(),
        events: vec![
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ingest.ok".to_owned(),
                message: "input materialized".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 2,
                ts_rfc3339: "2026-01-01T00:00:30Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.ok".to_owned(),
                message: "transcription complete".to_owned(),
                payload: json!({"backend": "whisper_cpp"}),
            },
        ],
        warnings: vec![],
        evidence: vec![],
        replay: ReplayEnvelope::default(),
    }
}

/// Create a RunStore backed by a temporary database. Returns the store and
/// a TempDir guard (drop the guard to clean up).
pub fn create_test_db() -> (franken_whisper::storage::RunStore, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let db_path = tmp.path().join("test.sqlite3");
    let store = franken_whisper::storage::RunStore::open(&db_path).expect("failed to open test db");
    (store, tmp)
}

/// Assert that two segment lists match within a time tolerance.
pub fn assert_segments_match(
    actual: &[TranscriptionSegment],
    expected: &[TranscriptionSegment],
    tolerance_sec: f64,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "segment count mismatch: got {} expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if let (Some(a_start), Some(e_start)) = (a.start_sec, e.start_sec) {
            assert!(
                (a_start - e_start).abs() < tolerance_sec,
                "segment {i} start_sec: {a_start} vs {e_start} (tolerance {tolerance_sec})"
            );
        }
        if let (Some(a_end), Some(e_end)) = (a.end_sec, e.end_sec) {
            assert!(
                (a_end - e_end).abs() < tolerance_sec,
                "segment {i} end_sec: {a_end} vs {e_end} (tolerance {tolerance_sec})"
            );
        }
        assert_eq!(a.text.trim(), e.text.trim(), "segment {i} text mismatch");
    }
}

/// Assert that events have monotonically increasing seq numbers.
pub fn assert_events_ordered(events: &[RunEvent]) {
    for window in events.windows(2) {
        assert!(
            window[1].seq > window[0].seq,
            "events not monotonic: seq {} followed by {}",
            window[0].seq,
            window[1].seq
        );
    }
}

/// Build a map of environment variables pointing at mock backends.
///
/// Callers should pass this to `std::process::Command::envs()` when spawning
/// subprocesses that need to resolve backend binaries. This avoids mutating
/// the global process environment (which requires `unsafe` under Rust 2024
/// edition and is forbidden by the crate-level lint).
pub fn mock_backend_env() -> HashMap<String, String> {
    let mocks = mocks_dir();
    let mut env = HashMap::new();
    // These helpers prove the external bridge contract. A cached native model
    // must not silently bypass the mock executable and turn the test into a
    // host-dependent native inference run.
    env.insert(
        "FRANKEN_WHISPER_NATIVE_EXECUTION".to_owned(),
        "0".to_owned(),
    );
    env.insert(
        "FRANKEN_WHISPER_WHISPER_CPP_BIN".to_owned(),
        mocks.join("mock_whisper_cpp.sh").display().to_string(),
    );
    env.insert(
        "FRANKEN_WHISPER_INSANELY_FAST_BIN".to_owned(),
        mocks.join("mock_insanely_fast.sh").display().to_string(),
    );
    env.insert(
        "FRANKEN_WHISPER_PYTHON_BIN".to_owned(),
        "python3".to_owned(),
    );
    env
}

/// Generate a synthetic WAV file (16-bit PCM, mono, 16kHz) with a sine tone.
/// Returns the path to the generated file.
pub fn generate_test_wav(dir: &Path, name: &str, duration_secs: f32, frequency_hz: f32) -> PathBuf {
    let sample_rate: u32 = 16000;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let value = (2.0 * std::f32::consts::PI * frequency_hz * t).sin();
        let sample = (value * 32767.0) as i16;
        samples.push(sample);
    }

    let path = dir.join(name);
    write_wav_file(&path, &samples, sample_rate, 1);
    path
}

/// Generate a silent WAV file.
pub fn generate_silence_wav(dir: &Path, name: &str, duration_secs: f32) -> PathBuf {
    let sample_rate: u32 = 16000;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let samples = vec![0i16; num_samples];
    let path = dir.join(name);
    write_wav_file(&path, &samples, sample_rate, 1);
    path
}

fn write_wav_file(path: &Path, samples: &[i16], sample_rate: u32, channels: u16) {
    use std::io::Write;
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    let byte_rate = sample_rate * channels as u32 * 2;
    let block_align = channels * 2;

    let mut file = std::fs::File::create(path).expect("failed to create WAV file");
    // RIFF header
    file.write_all(b"RIFF").unwrap();
    file.write_all(&file_size.to_le_bytes()).unwrap();
    file.write_all(b"WAVE").unwrap();
    // fmt chunk
    file.write_all(b"fmt ").unwrap();
    file.write_all(&16u32.to_le_bytes()).unwrap(); // chunk size
    file.write_all(&1u16.to_le_bytes()).unwrap(); // PCM format
    file.write_all(&channels.to_le_bytes()).unwrap();
    file.write_all(&sample_rate.to_le_bytes()).unwrap();
    file.write_all(&byte_rate.to_le_bytes()).unwrap();
    file.write_all(&block_align.to_le_bytes()).unwrap();
    file.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    // data chunk
    file.write_all(b"data").unwrap();
    file.write_all(&data_size.to_le_bytes()).unwrap();
    for sample in samples {
        file.write_all(&sample.to_le_bytes()).unwrap();
    }
}
