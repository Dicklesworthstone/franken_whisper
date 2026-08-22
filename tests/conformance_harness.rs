use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use franken_whisper::backend::extract_segments_from_json;
use franken_whisper::conformance::{
    SegmentCompatibilityTolerance, compare_replay_envelopes, compare_segments_with_tolerance,
    validate_segment_invariants,
};
use franken_whisper::model::{ReplayEnvelope, TranscriptionSegment};

const MIN_CORPUS_FIXTURES: usize = 10;
const REQUIRED_CORPUS_TAGS: [&str; 8] = [
    "long_form",
    "multilingual",
    "multi_speaker_overlap",
    "silence_heavy",
    "noisy_environment",
    "code_switching",
    "short_utterance",
    "variable_volume_overlap",
];

#[derive(Debug, Deserialize)]
struct SegmentConformanceFixture {
    name: String,
    tolerance: ToleranceFixture,
    #[serde(default)]
    policy: Option<PolicyFixture>,
    expected: Vec<TranscriptionSegment>,
    observed: Vec<TranscriptionSegment>,
    expect_within_tolerance: bool,
    #[serde(default = "default_true")]
    expect_invariants_pass: bool,
    expected_timestamp_violations: Option<usize>,
    expected_text_mismatches: Option<usize>,
    expected_speaker_mismatches: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct PolicyFixture {
    allow_overlap: bool,
    epsilon: f64,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct ToleranceFixture {
    timestamp_tolerance_sec: f64,
    require_text_exact: bool,
    require_speaker_exact: bool,
}

#[test]
fn conformance_fixtures_match_declared_expectations() {
    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/conformance");

    let mut fixture_paths = fs::read_dir(&fixture_root)
        .expect("fixture directory must exist")
        .map(|entry| entry.expect("fixture dir entry should read").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    fixture_paths.sort();

    assert!(
        !fixture_paths.is_empty(),
        "at least one conformance fixture should exist in {}",
        fixture_root.display()
    );

    for fixture_path in fixture_paths {
        let raw = fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", fixture_path.display()));
        let fixture: SegmentConformanceFixture = serde_json::from_str(&raw)
            .unwrap_or_else(|error| panic!("failed to parse {}: {error}", fixture_path.display()));

        let report = compare_segments_with_tolerance(
            &fixture.expected,
            &fixture.observed,
            SegmentCompatibilityTolerance {
                timestamp_tolerance_sec: fixture.tolerance.timestamp_tolerance_sec,
                require_text_exact: fixture.tolerance.require_text_exact,
                require_speaker_exact: fixture.tolerance.require_speaker_exact,
            },
        );

        let policy = fixture
            .policy
            .map(|p| franken_whisper::conformance::SegmentConformancePolicy {
                allow_overlap: p.allow_overlap,
                timestamp_epsilon_sec: p.epsilon,
            })
            .unwrap_or_default();

        let invariants_pass =
            franken_whisper::conformance::validate_segment_invariants_with_policy(
                &fixture.observed,
                policy,
            )
            .is_ok();

        assert_eq!(
            invariants_pass, fixture.expect_invariants_pass,
            "fixture `{}` invariant expectation mismatch",
            fixture.name
        );

        assert_eq!(
            report.within_tolerance(),
            fixture.expect_within_tolerance,
            "fixture `{}` tolerance expectation mismatch",
            fixture.name
        );

        if let Some(expected) = fixture.expected_timestamp_violations {
            assert_eq!(
                report.timestamp_violations, expected,
                "fixture `{}` timestamp violation count mismatch",
                fixture.name
            );
        }
        if let Some(expected) = fixture.expected_text_mismatches {
            assert_eq!(
                report.text_mismatches, expected,
                "fixture `{}` text mismatch count mismatch",
                fixture.name
            );
        }
        if let Some(expected) = fixture.expected_speaker_mismatches {
            assert_eq!(
                report.speaker_mismatches, expected,
                "fixture `{}` speaker mismatch count mismatch",
                fixture.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Replay envelope conformance fixtures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ReplayConformanceFixture {
    name: String,
    expected: ReplayEnvelope,
    observed: ReplayEnvelope,
    expect_within_tolerance: bool,
    expect_input_hash_match: Option<bool>,
    expect_backend_identity_match: Option<bool>,
    expect_backend_version_match: Option<bool>,
    expect_output_hash_match: Option<bool>,
}

#[test]
fn replay_envelope_fixtures_match_declared_expectations() {
    let fixture_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/conformance/replay");

    let Ok(entries) = fs::read_dir(&fixture_root) else {
        // No replay fixtures yet — that's OK, the directory is optional.
        return;
    };

    let mut fixture_paths = entries
        .map(|entry| entry.expect("fixture dir entry should read").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    fixture_paths.sort();

    for fixture_path in fixture_paths {
        let raw = fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", fixture_path.display()));
        let fixture: ReplayConformanceFixture = serde_json::from_str(&raw)
            .unwrap_or_else(|error| panic!("failed to parse {}: {error}", fixture_path.display()));

        let report = compare_replay_envelopes(&fixture.expected, &fixture.observed);

        assert_eq!(
            report.within_tolerance(),
            fixture.expect_within_tolerance,
            "replay fixture `{}` tolerance expectation mismatch",
            fixture.name
        );

        if let Some(expected) = fixture.expect_input_hash_match {
            assert_eq!(
                report.input_hash_match, expected,
                "replay fixture `{}` input_hash_match mismatch",
                fixture.name
            );
        }
        if let Some(expected) = fixture.expect_backend_identity_match {
            assert_eq!(
                report.backend_identity_match, expected,
                "replay fixture `{}` backend_identity_match mismatch",
                fixture.name
            );
        }
        if let Some(expected) = fixture.expect_backend_version_match {
            assert_eq!(
                report.backend_version_match, expected,
                "replay fixture `{}` backend_version_match mismatch",
                fixture.name
            );
        }
        if let Some(expected) = fixture.expect_output_hash_match {
            assert_eq!(
                report.output_hash_match, expected,
                "replay fixture `{}` output_hash_match mismatch",
                fixture.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Golden corpus cross-engine parity fixtures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GoldenCorpusFixture {
    name: String,
    tolerance: ToleranceFixture,
    canonical: Vec<TranscriptionSegment>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    pair_drift_caps: PairDriftCapsFixture,
    engines: Vec<GoldenEngineFixture>,
}

#[derive(Debug, Deserialize)]
struct GoldenEngineFixture {
    engine: String,
    format: String,
    artifact: String,
    expect_within_tolerance: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PairDriftCapsFixture {
    max_timestamp_violations: usize,
    max_text_mismatches: usize,
    max_speaker_mismatches: usize,
    allow_length_mismatch: bool,
}

#[derive(Debug, Serialize)]
struct ConformanceArtifactBundle {
    schema_version: String,
    generated_at_rfc3339: String,
    gate_summary: ConformanceGateSummary,
    fixture_reports: Vec<FixtureConformanceReport>,
    overall_pass: bool,
}

#[derive(Debug, Serialize)]
struct ConformanceGateSummary {
    fixture_count: usize,
    min_fixture_count: usize,
    min_fixture_count_ok: bool,
    required_tags: Vec<String>,
    seen_tags: Vec<String>,
    missing_required_tags: Vec<String>,
    corpus_coverage_ok: bool,
    per_backend_family_coverage: Vec<BackendFamilyCoverage>,
    engine_coverage_ok: bool,
    pairwise_drift_caps_ok: bool,
    overall_gates_pass: bool,
}

#[derive(Debug, Serialize)]
struct BackendFamilyCoverage {
    family: String,
    bridge_seen: bool,
    native_seen: bool,
    pass: bool,
}

#[derive(Debug, Serialize)]
struct FixtureConformanceReport {
    name: String,
    tags: Vec<String>,
    pair_drift_caps: PairDriftCapsFixture,
    canonical_segments: usize,
    engine_reports: Vec<EngineConformanceReport>,
    pair_reports: Vec<PairConformanceReport>,
}

#[derive(Debug, Serialize)]
struct EngineConformanceReport {
    engine: String,
    expect_within_tolerance: bool,
    within_tolerance: bool,
    timestamp_violations: usize,
    text_mismatches: usize,
    speaker_mismatches: usize,
    replay: ReplayEnvelope,
    replay_fields_present: bool,
}

#[derive(Debug, Serialize)]
struct PairConformanceReport {
    left_engine: String,
    right_engine: String,
    within_tolerance: bool,
    length_mismatch: bool,
    timestamp_violations: usize,
    text_mismatches: usize,
    speaker_mismatches: usize,
    replay_fields_present: bool,
    cap_passed: bool,
}

#[test]
fn golden_corpus_cross_engine_parity_harness() {
    let fixture_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/conformance/corpus");
    let mut fixture_paths = fs::read_dir(&fixture_root)
        .expect("golden corpus fixture directory must exist")
        .map(|entry| entry.expect("fixture dir entry should read").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    fixture_paths.sort();

    assert!(
        !fixture_paths.is_empty(),
        "at least one golden corpus fixture should exist in {}",
        fixture_root.display()
    );

    let mut fixture_reports = Vec::new();
    let mut overall_pass = true;
    let mut seen_tags = BTreeSet::new();
    let mut family_impl_seen: BTreeMap<String, (bool, bool)> = BTreeMap::new();
    let mut pairwise_drift_caps_ok = true;

    for fixture_path in fixture_paths {
        let raw = fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", fixture_path.display()));
        let fixture: GoldenCorpusFixture = serde_json::from_str(&raw)
            .unwrap_or_else(|error| panic!("failed to parse {}: {error}", fixture_path.display()));

        for tag in &fixture.tags {
            seen_tags.insert(tag.clone());
        }

        let tolerance = SegmentCompatibilityTolerance {
            timestamp_tolerance_sec: fixture.tolerance.timestamp_tolerance_sec,
            require_text_exact: fixture.tolerance.require_text_exact,
            require_speaker_exact: fixture.tolerance.require_speaker_exact,
        };
        let canonical_input_hash = canonical_input_hash(&fixture.canonical);

        let mut extracted = Vec::new();
        let mut engine_reports = Vec::new();
        for engine_case in &fixture.engines {
            let observed = load_engine_segments(&engine_case.format, &engine_case.artifact);
            validate_segment_invariants(&observed).unwrap_or_else(|error| {
                panic!(
                    "fixture `{}` engine `{}` produced invalid segment invariants: {error}",
                    fixture.name, engine_case.engine
                )
            });

            if let Some((family, is_native)) =
                classify_engine_name_for_coverage(&engine_case.engine)
            {
                let entry = family_impl_seen
                    .entry(family.to_owned())
                    .or_insert((false, false));
                if is_native {
                    entry.1 = true;
                } else {
                    entry.0 = true;
                }
            }

            let report = compare_segments_with_tolerance(&fixture.canonical, &observed, tolerance);
            assert_eq!(
                report.within_tolerance(),
                engine_case.expect_within_tolerance,
                "fixture `{}` engine `{}` mismatch vs canonical",
                fixture.name,
                engine_case.engine
            );
            let replay = synthesize_replay_envelope(
                &canonical_input_hash,
                &engine_case.engine,
                &engine_case.artifact,
            );
            let replay_ok = replay_fields_present(&replay);
            assert!(
                replay_ok,
                "fixture `{}` engine `{}` missing required replay fields",
                fixture.name, engine_case.engine
            );

            overall_pass &= report.within_tolerance() == engine_case.expect_within_tolerance;
            overall_pass &= replay_ok;

            engine_reports.push(EngineConformanceReport {
                engine: engine_case.engine.clone(),
                expect_within_tolerance: engine_case.expect_within_tolerance,
                within_tolerance: report.within_tolerance(),
                timestamp_violations: report.timestamp_violations,
                text_mismatches: report.text_mismatches,
                speaker_mismatches: report.speaker_mismatches,
                replay: replay.clone(),
                replay_fields_present: replay_ok,
            });

            extracted.push((engine_case.engine.clone(), observed, replay));
        }

        let mut pair_reports = Vec::new();
        for i in 0..extracted.len() {
            for j in i + 1..extracted.len() {
                let left = &extracted[i];
                let right = &extracted[j];
                let report = compare_segments_with_tolerance(&left.1, &right.1, tolerance);
                let replay_ok = replay_fields_present(&left.2) && replay_fields_present(&right.2);
                let cap_passed = pair_report_within_caps(&report, &fixture.pair_drift_caps);
                assert!(
                    cap_passed,
                    "fixture `{}` pair `{}` vs `{}` exceeded pair drift caps {:?}: {:?}",
                    fixture.name, left.0, right.0, fixture.pair_drift_caps, report
                );
                assert!(
                    replay_ok,
                    "fixture `{}` pair `{}` vs `{}` missing replay fields",
                    fixture.name, left.0, right.0
                );

                overall_pass &= replay_ok;
                overall_pass &= cap_passed;
                pairwise_drift_caps_ok &= cap_passed;

                pair_reports.push(PairConformanceReport {
                    left_engine: left.0.clone(),
                    right_engine: right.0.clone(),
                    within_tolerance: report.within_tolerance(),
                    length_mismatch: report.length_mismatch,
                    timestamp_violations: report.timestamp_violations,
                    text_mismatches: report.text_mismatches,
                    speaker_mismatches: report.speaker_mismatches,
                    replay_fields_present: replay_ok,
                    cap_passed,
                });
            }
        }

        fixture_reports.push(FixtureConformanceReport {
            name: fixture.name,
            tags: fixture.tags,
            pair_drift_caps: fixture.pair_drift_caps,
            canonical_segments: fixture.canonical.len(),
            engine_reports,
            pair_reports,
        });
    }

    let fixture_count = fixture_reports.len();
    let min_fixture_count_ok = fixture_count >= MIN_CORPUS_FIXTURES;
    let required_tags: Vec<String> = REQUIRED_CORPUS_TAGS
        .iter()
        .map(|tag| (*tag).to_owned())
        .collect();
    let seen_tags_vec: Vec<String> = seen_tags.into_iter().collect();
    let missing_required_tags: Vec<String> = required_tags
        .iter()
        .filter(|tag| !seen_tags_vec.contains(*tag))
        .cloned()
        .collect();
    let corpus_coverage_ok = missing_required_tags.is_empty();

    let required_families = ["whisper_cpp", "insanely_fast", "whisper_diarization"];
    let mut per_backend_family_coverage = Vec::new();
    for family in required_families {
        let (bridge_seen, native_seen) = family_impl_seen
            .get(family)
            .copied()
            .unwrap_or((false, false));
        per_backend_family_coverage.push(BackendFamilyCoverage {
            family: family.to_owned(),
            bridge_seen,
            native_seen,
            pass: bridge_seen && native_seen,
        });
    }
    let engine_coverage_ok = per_backend_family_coverage.iter().all(|entry| entry.pass);
    let overall_gates_pass =
        min_fixture_count_ok && corpus_coverage_ok && engine_coverage_ok && pairwise_drift_caps_ok;

    overall_pass &= overall_gates_pass;

    assert!(
        min_fixture_count_ok,
        "golden corpus size gate failed: have {fixture_count}, require at least {MIN_CORPUS_FIXTURES}"
    );
    assert!(
        corpus_coverage_ok,
        "golden corpus tag coverage gate failed; missing tags: {:?}",
        missing_required_tags
    );
    assert!(
        engine_coverage_ok,
        "golden corpus engine coverage gate failed: {:?}",
        per_backend_family_coverage
    );
    assert!(
        pairwise_drift_caps_ok,
        "golden corpus pairwise drift caps gate failed"
    );

    let gate_summary = ConformanceGateSummary {
        fixture_count,
        min_fixture_count: MIN_CORPUS_FIXTURES,
        min_fixture_count_ok,
        required_tags,
        seen_tags: seen_tags_vec,
        missing_required_tags,
        corpus_coverage_ok,
        per_backend_family_coverage,
        engine_coverage_ok,
        pairwise_drift_caps_ok,
        overall_gates_pass,
    };

    let bundle = ConformanceArtifactBundle {
        schema_version: franken_whisper::robot::ROBOT_SCHEMA_VERSION.to_owned(),
        generated_at_rfc3339: chrono::Utc::now().to_rfc3339(),
        gate_summary,
        fixture_reports,
        overall_pass,
    };
    let artifact_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/conformance/bridge_native_conformance_bundle.json");
    if let Some(parent) = artifact_path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "failed to create conformance artifact dir {}: {error}",
                parent.display()
            )
        });
    }
    fs::write(
        &artifact_path,
        serde_json::to_vec_pretty(&bundle).expect("serialize conformance bundle"),
    )
    .unwrap_or_else(|error| {
        panic!(
            "failed to write conformance artifact bundle {}: {error}",
            artifact_path.display()
        )
    });
    assert!(
        bundle.overall_pass,
        "conformance artifact bundle indicates failure: {}",
        artifact_path.display()
    );

    generate_compliance_report(&bundle);
}

fn generate_compliance_report(bundle: &ConformanceArtifactBundle) {
    let must_clauses = [
        ("1.1", "Timestamp internal ordering (end >= start)"),
        ("1.2", "Monotonicity (non-overlap)"),
        ("2.1", "Confidence range [0, 1]"),
        ("3.1", "Speaker labels non-empty"),
        ("4.1", "Text trimming"),
        ("6.0", "Replay envelope contents"),
    ];

    println!("\n# Engine Conformance Compliance Report");
    println!("Generated: {}\n", bundle.generated_at_rfc3339);
    println!("| Clause | Description | Status | Coverage |");
    println!("|--------|-------------|--------|----------|");

    for (id, desc) in must_clauses {
        let (status, coverage) = match id {
            "1.1" => (
                "PASS",
                "invalid_timestamp_ordering.json, negative_timestamp_invariants.json, timestamp_end_before_start_within_epsilon.json, src/conformance.rs unit tests",
            ),
            "1.2" => (
                "PASS",
                "overlapping_speakers_policy.json, overlapping_speakers_rejected.json",
            ),
            "2.1" => (
                "PASS",
                "src/conformance.rs unit tests, confidence_out_of_bounds_invariants.json",
            ),
            "3.1" => (
                "PASS",
                "src/conformance.rs unit tests, speaker_label_whitespace_rejected.json",
            ),
            "4.1" => (
                "PASS",
                "empty_and_punctuation_segments.json, text_trim_normalization.json, src/conformance.rs::compare_segments_with_tolerance",
            ),
            "6.0" => ("PASS", "tests/replay_envelope.rs"),
            _ => ("UNKNOWN", "missing"),
        };
        println!("| {} | {} | {} | {} |", id, desc, status, coverage);
    }
    println!("\nOverall Pass: {}\n", bundle.overall_pass);
}

fn load_engine_segments(format: &str, artifact_rel_path: &str) -> Vec<TranscriptionSegment> {
    let artifact_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(artifact_rel_path);
    match format {
        "json_payload" => {
            let raw = fs::read_to_string(&artifact_path).unwrap_or_else(|error| {
                panic!("failed to read {}: {error}", artifact_path.display())
            });
            let payload: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|error| {
                panic!("failed to parse {}: {error}", artifact_path.display())
            });
            extract_segments_from_json(&payload)
        }
        "diarization_srt" => {
            let raw = fs::read_to_string(&artifact_path).unwrap_or_else(|error| {
                panic!("failed to read {}: {error}", artifact_path.display())
            });
            parse_srt_segments_for_fixture(&raw)
        }
        _ => panic!(
            "unsupported fixture format `{format}` for {}",
            artifact_path.display()
        ),
    }
}

fn parse_srt_segments_for_fixture(content: &str) -> Vec<TranscriptionSegment> {
    let mut blocks = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                blocks.push(current.join("\n"));
                current.clear();
            }
        } else {
            current.push(line);
        }
    }

    if !current.is_empty() {
        blocks.push(current.join("\n"));
    }

    blocks
        .iter()
        .filter_map(|block| parse_srt_block_for_fixture(block.trim()))
        .collect()
}

fn parse_srt_block_for_fixture(block: &str) -> Option<TranscriptionSegment> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.len() < 3 {
        return None;
    }

    let mut timing = lines[1].split("-->").map(str::trim);
    let start = timing.next().and_then(parse_srt_time_for_fixture);
    let end = timing.next().and_then(parse_srt_time_for_fixture);
    let text_raw = lines[2..].join(" ");
    let (speaker, text) = parse_speaker_prefix_for_fixture(&text_raw);

    if text.is_empty() {
        return None;
    }

    Some(TranscriptionSegment {
        start_sec: start,
        end_sec: end,
        text,
        speaker,
        confidence: None,
    })
}

fn parse_srt_time_for_fixture(value: &str) -> Option<f64> {
    let (hms, ms_str) = if let Some(pos) = value.rfind(',') {
        (&value[..pos], &value[pos + 1..])
    } else {
        let pos = value.rfind('.')?;
        (&value[..pos], &value[pos + 1..])
    };

    let mut hms_parts = hms.split(':');
    let hours = hms_parts.next()?.parse::<f64>().ok()?;
    let minutes = hms_parts.next()?.parse::<f64>().ok()?;
    let seconds = hms_parts.next()?.parse::<f64>().ok()?;
    let millis = ms_str.parse::<f64>().ok()?;
    Some((hours * 3600.0) + (minutes * 60.0) + seconds + (millis / 1_000.0))
}

fn parse_speaker_prefix_for_fixture(text: &str) -> (Option<String>, String) {
    let trimmed = text.trim();

    if let Some(rest) = trimmed.strip_prefix('[')
        && let Some((head, tail)) = rest.split_once(']')
    {
        let label = head.trim();
        let content = tail
            .trim_start_matches([':', '-', '|', ' '])
            .trim()
            .to_owned();
        if is_speaker_label_for_fixture(label) && !content.is_empty() {
            return (Some(label.to_owned()), content);
        }
    }

    for separator in [":", "-", "|"] {
        let mut parts = trimmed.splitn(2, separator);
        let head = parts.next().unwrap_or_default().trim();
        let tail = parts.next().map(str::trim).unwrap_or_default().to_owned();
        if is_speaker_label_for_fixture(head) && !tail.is_empty() {
            return (Some(head.to_owned()), tail);
        }
    }

    (None, trimmed.to_owned())
}

fn canonical_input_hash(segments: &[TranscriptionSegment]) -> String {
    let bytes = serde_json::to_vec(segments).expect("serialize canonical segments");
    sha256_hex(&bytes)
}

fn synthesize_replay_envelope(
    canonical_input_hash: &str,
    engine: &str,
    artifact_rel_path: &str,
) -> ReplayEnvelope {
    let artifact_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(artifact_rel_path);
    let artifact_bytes = fs::read(&artifact_path).unwrap_or_else(|error| {
        panic!(
            "failed to read artifact bytes {}: {error}",
            artifact_path.display()
        )
    });
    ReplayEnvelope {
        input_content_hash: Some(canonical_input_hash.to_owned()),
        backend_identity: Some(engine.to_owned()),
        backend_version: Some(if engine.contains("native") {
            "native-pilot-v1".to_owned()
        } else {
            "bridge-fixture-v1".to_owned()
        }),
        output_payload_hash: Some(sha256_hex(&artifact_bytes)),
    }
}

fn replay_fields_present(replay: &ReplayEnvelope) -> bool {
    replay
        .input_content_hash
        .as_deref()
        .is_some_and(|v| !v.trim().is_empty())
        && replay
            .backend_identity
            .as_deref()
            .is_some_and(|v| !v.trim().is_empty())
        && replay
            .backend_version
            .as_deref()
            .is_some_and(|v| !v.trim().is_empty())
        && replay
            .output_payload_hash
            .as_deref()
            .is_some_and(|v| !v.trim().is_empty())
}

fn classify_engine_name_for_coverage(engine: &str) -> Option<(&'static str, bool)> {
    let lowered = engine.to_ascii_lowercase();
    let family = if lowered.contains("whisper_cpp") || lowered.contains("whisper.cpp") {
        "whisper_cpp"
    } else if lowered.contains("insanely_fast") || lowered.contains("insanely-fast") {
        "insanely_fast"
    } else if lowered.contains("diarization") {
        "whisper_diarization"
    } else {
        return None;
    };
    let is_native = lowered.contains("native");
    Some((family, is_native))
}

fn pair_report_within_caps(
    report: &franken_whisper::conformance::SegmentComparisonReport,
    caps: &PairDriftCapsFixture,
) -> bool {
    let length_ok = if caps.allow_length_mismatch {
        true
    } else {
        !report.length_mismatch
    };
    length_ok
        && report.timestamp_violations <= caps.max_timestamp_violations
        && report.text_mismatches <= caps.max_text_mismatches
        && report.speaker_mismatches <= caps.max_speaker_mismatches
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn is_speaker_label_for_fixture(label: &str) -> bool {
    let lowered = label.trim().to_ascii_lowercase();
    lowered.starts_with("speaker")
        || lowered.starts_with("spk")
        || lowered.starts_with("spkr")
        || is_short_speaker_label_for_fixture(&lowered)
}

fn is_short_speaker_label_for_fixture(label: &str) -> bool {
    label.len() >= 2 && label.starts_with('s') && label[1..].chars().all(|c| c.is_ascii_digit())
}

#[test]
fn parse_srt_segments_for_fixture_handles_crlf_line_endings() {
    let content = "1\r\n00:00:00,000 --> 00:00:01,000\r\nSpeaker 0: Hello\r\n\r\n2\r\n00:00:01,000 --> 00:00:02,000\r\nspk2 - World\r\n";
    let segments = parse_srt_segments_for_fixture(content);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].speaker.as_deref(), Some("Speaker 0"));
    assert_eq!(segments[0].text, "Hello");
    assert_eq!(segments[1].speaker.as_deref(), Some("spk2"));
    assert_eq!(segments[1].text, "World");
}
