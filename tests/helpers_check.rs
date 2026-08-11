mod helpers;

#[test]
fn helpers_compile_check() {
    let _fixtures = helpers::fixtures_dir();
    let _golden = helpers::golden_dir();
    let _mocks = helpers::mocks_dir();
    let _req = helpers::create_test_request();
    let _result = helpers::create_test_result();
    let _report = helpers::create_test_report();

    // Check that golden files exist
    assert!(
        helpers::golden_dir()
            .join("whisper_cpp_output.json")
            .exists()
    );
    assert!(
        helpers::golden_dir()
            .join("insanely_fast_output.json")
            .exists()
    );
    assert!(
        helpers::golden_dir()
            .join("diarization_output.txt")
            .exists()
    );
    assert!(
        helpers::golden_dir()
            .join("diarization_output.srt")
            .exists()
    );
    assert!(helpers::golden_dir().join("robot_events.ndjson").exists());
    assert!(helpers::golden_dir().join("tty_frames.ndjson").exists());

    // Check mock scripts exist and are files
    assert!(helpers::mocks_dir().join("mock_whisper_cpp.sh").exists());
    assert!(helpers::mocks_dir().join("mock_insanely_fast.sh").exists());
    assert!(helpers::mocks_dir().join("mock_diarization.py").exists());
}

#[test]
fn helpers_test_db_works() {
    let (_store, _tmp) = helpers::create_test_db();
}

#[test]
fn helpers_generate_wav_works() {
    let tmp = tempfile::tempdir().unwrap();
    let wav_path = helpers::generate_test_wav(tmp.path(), "tone.wav", 1.0, 440.0);
    assert!(wav_path.exists());
    let metadata = std::fs::metadata(&wav_path).unwrap();
    // 16kHz * 1s * 2 bytes/sample + 44 bytes header = 32044
    assert_eq!(metadata.len(), 32044);

    let silence_path = helpers::generate_silence_wav(tmp.path(), "silence.wav", 0.5);
    assert!(silence_path.exists());
}

#[test]
fn helpers_assert_segments_match_works() {
    use franken_whisper::model::TranscriptionSegment;

    let segs = vec![TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(2.5),
        text: "Hello world.".to_owned(),
        speaker: None,
        confidence: Some(0.95),
    }];
    helpers::assert_segments_match(&segs, &segs, 0.01);
}

#[test]
fn helpers_assert_events_ordered_works() {
    use franken_whisper::model::RunEvent;
    use serde_json::json;

    let events = vec![
        RunEvent {
            seq: 1,
            ts_rfc3339: "t1".into(),
            stage: "a".into(),
            code: "a".into(),
            message: "m".into(),
            payload: json!({}),
        },
        RunEvent {
            seq: 2,
            ts_rfc3339: "t2".into(),
            stage: "b".into(),
            code: "b".into(),
            message: "m".into(),
            payload: json!({}),
        },
    ];
    helpers::assert_events_ordered(&events);
}

#[test]
fn helpers_mock_backend_env_has_expected_keys() {
    let env = helpers::mock_backend_env();
    assert_eq!(
        env.get("FRANKEN_WHISPER_NATIVE_EXECUTION")
            .map(String::as_str),
        Some("0")
    );
    assert!(env.contains_key("FRANKEN_WHISPER_WHISPER_CPP_BIN"));
    assert!(env.contains_key("FRANKEN_WHISPER_INSANELY_FAST_BIN"));
    assert!(env.contains_key("FRANKEN_WHISPER_PYTHON_BIN"));
}

#[test]
fn golden_ndjson_lines_are_valid_json() {
    for filename in &["robot_events.ndjson", "tty_frames.ndjson"] {
        let path = helpers::golden_dir().join(filename);
        let content = std::fs::read_to_string(&path).unwrap();
        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            assert!(
                serde_json::from_str::<serde_json::Value>(line).is_ok(),
                "Invalid JSON in {} line {}: {}",
                filename,
                i + 1,
                line
            );
        }
    }
}
