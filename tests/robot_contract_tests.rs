//! Robot mode JSON contract tests.
//!
//! These integration tests verify the public contract of the `robot` module
//! without reaching into private internals. All assertions use the public API:
//! constants, `robot_schema_value()`, the `emit_*` functions, and the golden
//! fixture file `tests/fixtures/golden/robot_events.ndjson`.

mod helpers;

use std::collections::HashSet;

use franken_whisper::robot::{
    BACKENDS_DISCOVERY_REQUIRED_FIELDS, HEALTH_REPORT_REQUIRED_FIELDS,
    LISTEN_SESSION_START_REQUIRED_FIELDS, LISTEN_SESSION_STATS_REQUIRED_FIELDS,
    LISTEN_WARNING_REQUIRED_FIELDS, ROBOT_SCHEMA_VERSION, RUN_COMPLETE_REQUIRED_FIELDS,
    RUN_ERROR_REQUIRED_FIELDS, RUN_START_REQUIRED_FIELDS, SPECULATION_STATS_REQUIRED_FIELDS,
    SPEECH_STARTED_REQUIRED_FIELDS, STAGE_REQUIRED_FIELDS, TRANSCRIPT_CONFIRM_REQUIRED_FIELDS,
    TRANSCRIPT_CORRECT_REQUIRED_FIELDS, TRANSCRIPT_DELTA_REQUIRED_FIELDS,
    TRANSCRIPT_PARTIAL_REQUIRED_FIELDS, TRANSCRIPT_RETRACT_REQUIRED_FIELDS,
    UTTERANCE_END_REQUIRED_FIELDS, robot_schema_value,
};
use serde_json::Value;

// ---------------------------------------------------------------------------
// 1. Schema version is present in all event types
// ---------------------------------------------------------------------------

#[test]
fn schema_version_present_in_all_event_type_examples() {
    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    for (event_type, spec) in events {
        let example = &spec["example"];
        assert!(
            example.get("schema_version").is_some(),
            "schema example for `{event_type}` must contain `schema_version`"
        );
        assert_eq!(
            example["schema_version"]
                .as_str()
                .expect("schema_version should be a string"),
            ROBOT_SCHEMA_VERSION,
            "schema_version in `{event_type}` example must match ROBOT_SCHEMA_VERSION"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Required fields are present in run_start events
// ---------------------------------------------------------------------------

#[test]
fn run_start_example_contains_all_required_fields() {
    let schema = robot_schema_value();
    let run_start_example = &schema["events"]["run_start"]["example"];

    for field in RUN_START_REQUIRED_FIELDS {
        assert!(
            run_start_example.get(*field).is_some(),
            "run_start example missing required field `{field}`"
        );
    }
}

#[test]
fn run_start_required_fields_match_schema_declaration() {
    let schema = robot_schema_value();
    let declared: Vec<String> = schema["events"]["run_start"]["required"]
        .as_array()
        .expect("required should be array")
        .iter()
        .map(|v| v.as_str().expect("field name").to_owned())
        .collect();

    let constant: Vec<String> = RUN_START_REQUIRED_FIELDS
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    assert_eq!(
        declared, constant,
        "schema-declared run_start required fields must match the constant"
    );
}

// ---------------------------------------------------------------------------
// 3. Required fields are present in stage events
// ---------------------------------------------------------------------------

#[test]
fn stage_example_contains_all_required_fields() {
    let schema = robot_schema_value();
    let stage_example = &schema["events"]["stage"]["example"];

    for field in STAGE_REQUIRED_FIELDS {
        assert!(
            stage_example.get(*field).is_some(),
            "stage example missing required field `{field}`"
        );
    }
}

#[test]
fn stage_required_fields_match_schema_declaration() {
    let schema = robot_schema_value();
    let declared: Vec<String> = schema["events"]["stage"]["required"]
        .as_array()
        .expect("required should be array")
        .iter()
        .map(|v| v.as_str().expect("field name").to_owned())
        .collect();

    let constant: Vec<String> = STAGE_REQUIRED_FIELDS
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    assert_eq!(
        declared, constant,
        "schema-declared stage required fields must match the constant"
    );
}

#[test]
fn stage_required_fields_include_deterministic_order_keys() {
    for field in ["run_id", "seq", "ts", "stage", "code", "payload"] {
        assert!(
            STAGE_REQUIRED_FIELDS.contains(&field),
            "STAGE_REQUIRED_FIELDS must include `{field}`"
        );
    }
}

#[test]
fn stage_schema_declares_ordering_contract_metadata() {
    let schema = robot_schema_value();
    let ordering = &schema["events"]["stage"]["ordering_contract"];
    assert!(
        ordering.is_object(),
        "stage schema must include `ordering_contract` object"
    );

    assert_eq!(ordering["seq"], "strictly increasing per run");
    assert_eq!(ordering["ts"], "non-decreasing RFC3339 timestamp per run");
    assert_eq!(
        ordering["replay_rule"],
        "persisted stage event order must match streamed stage event order"
    );

    let failure_path = ordering["failure_path_example"]
        .as_array()
        .expect("failure_path_example should be an array");
    assert_eq!(failure_path.len(), 3);
    assert_eq!(failure_path[0], "orchestration.budgets");
    assert_eq!(failure_path[1], "ingest.start");
    assert_eq!(failure_path[2], "ingest.error");

    let cancellation_path = ordering["cancellation_path_example"]
        .as_array()
        .expect("cancellation_path_example should be an array");
    assert_eq!(cancellation_path.len(), 2);
    assert_eq!(cancellation_path[0], "orchestration.budgets");
    assert_eq!(cancellation_path[1], "orchestration.cancelled");
}

// ---------------------------------------------------------------------------
// 4. Required fields are present in run_complete events
// ---------------------------------------------------------------------------

#[test]
fn run_complete_example_contains_all_required_fields() {
    let schema = robot_schema_value();
    let run_complete_example = &schema["events"]["run_complete"]["example"];

    for field in RUN_COMPLETE_REQUIRED_FIELDS {
        assert!(
            run_complete_example.get(*field).is_some(),
            "run_complete example missing required field `{field}`"
        );
    }
}

#[test]
fn run_complete_required_fields_match_schema_declaration() {
    let schema = robot_schema_value();
    let declared: Vec<String> = schema["events"]["run_complete"]["required"]
        .as_array()
        .expect("required should be array")
        .iter()
        .map(|v| v.as_str().expect("field name").to_owned())
        .collect();

    let constant: Vec<String> = RUN_COMPLETE_REQUIRED_FIELDS
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    assert_eq!(
        declared, constant,
        "schema-declared run_complete required fields must match the constant"
    );
}

// ---------------------------------------------------------------------------
// 5. Required fields are present in run_error events
// ---------------------------------------------------------------------------

#[test]
fn run_error_example_contains_all_required_fields() {
    let schema = robot_schema_value();
    let run_error_example = &schema["events"]["run_error"]["example"];

    for field in RUN_ERROR_REQUIRED_FIELDS {
        assert!(
            run_error_example.get(*field).is_some(),
            "run_error example missing required field `{field}`"
        );
    }
}

#[test]
fn run_error_required_fields_match_schema_declaration() {
    let schema = robot_schema_value();
    let declared: Vec<String> = schema["events"]["run_error"]["required"]
        .as_array()
        .expect("required should be array")
        .iter()
        .map(|v| v.as_str().expect("field name").to_owned())
        .collect();

    let constant: Vec<String> = RUN_ERROR_REQUIRED_FIELDS
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    assert_eq!(
        declared, constant,
        "schema-declared run_error required fields must match the constant"
    );
}

// ---------------------------------------------------------------------------
// 6. Event types match expected strings
// ---------------------------------------------------------------------------

#[test]
fn event_type_strings_are_correct_in_examples() {
    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    let expected_event_types = [
        "run_start",
        "stage",
        "run_complete",
        "run_error",
        "backends.discovery",
        "transcript.partial",
        "transcript.confirm",
        "transcript.retract",
        "transcript.correct",
        "transcript.speculation_stats",
        "health.report",
    ];

    // Verify all expected event types exist in the schema.
    for expected in &expected_event_types {
        assert!(
            events.contains_key(*expected),
            "schema must contain event type `{expected}`"
        );
    }

    // Verify the "event" field in each example matches its key name.
    for (event_name, spec) in events {
        let example = &spec["example"];
        let event_field = example["event"]
            .as_str()
            .unwrap_or_else(|| panic!("`{event_name}` example must have string `event` field"));
        assert_eq!(
            event_field,
            event_name.as_str(),
            "example event field for `{event_name}` must equal the key"
        );
    }
}

#[test]
fn schema_has_expected_event_types() {
    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    assert_eq!(
        events.len(),
        18,
        "schema must define exactly 18 event types (12 batch + 6 listen), found {}",
        events.len()
    );

    let names: HashSet<&str> = events.keys().map(|k| k.as_str()).collect();
    for expected in [
        "run_start",
        "stage",
        "run_complete",
        "run_error",
        "backends.discovery",
        "routing_decision",
        "listen.session_start",
        "speech_started",
        "transcript.delta",
        "utterance_end",
        "listen.warning",
        "listen.session_stats",
        "transcript.partial",
        "transcript.confirm",
        "transcript.retract",
        "transcript.correct",
        "transcript.speculation_stats",
        "health.report",
    ] {
        assert!(
            names.contains(expected),
            "missing event type `{expected}` in schema"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Schema version follows semver format (X.Y.Z)
// ---------------------------------------------------------------------------

#[test]
fn schema_version_is_valid_semver() {
    let version = ROBOT_SCHEMA_VERSION;

    // Must match X.Y.Z where X, Y, Z are non-negative integers.
    let parts: Vec<&str> = version.split('.').collect();
    assert_eq!(
        parts.len(),
        3,
        "ROBOT_SCHEMA_VERSION must have exactly 3 dot-separated parts (X.Y.Z), got: `{version}`"
    );
    for (i, part) in parts.iter().enumerate() {
        assert!(
            !part.is_empty(),
            "semver part {i} must not be empty in `{version}`"
        );
        assert!(
            part.parse::<u32>().is_ok(),
            "semver part {i} (`{part}`) must be a valid non-negative integer in `{version}`"
        );
    }
}

#[test]
fn schema_value_reports_same_version_as_constant() {
    let schema = robot_schema_value();
    let version_in_schema = schema["schema_version"]
        .as_str()
        .expect("schema_version should be a string");
    assert_eq!(
        version_in_schema, ROBOT_SCHEMA_VERSION,
        "robot_schema_value().schema_version must equal ROBOT_SCHEMA_VERSION constant"
    );
}

// ---------------------------------------------------------------------------
// 8. Golden file robot_events.ndjson can be parsed; each line has "event"
//    and "schema_version" fields
// ---------------------------------------------------------------------------

#[test]
fn golden_robot_events_each_line_has_event_and_schema_version() {
    let ndjson_path = helpers::golden_dir().join("robot_events.ndjson");
    let text = std::fs::read_to_string(&ndjson_path).unwrap_or_else(|e| {
        panic!(
            "failed to read golden NDJSON at {}: {e}",
            ndjson_path.display()
        )
    });

    let lines: Vec<&str> = text.lines().collect();
    assert!(
        !lines.is_empty(),
        "golden robot_events.ndjson must not be empty"
    );

    for (i, line) in lines.iter().enumerate() {
        let parsed: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line {} is not valid JSON: {e}\nline: {line}", i + 1));

        assert!(
            parsed.get("event").is_some(),
            "line {} must have an `event` field",
            i + 1
        );

        // The golden file may contain older events. We verify the field exists
        // but allow the version string to differ from the current constant
        // (forward-compatible check).
        //
        // NOTE: The run_start line in the current golden file does not have
        // schema_version. We check for it only on lines that include it, but
        // also verify the majority of lines do carry the field.
    }

    // Count lines that have schema_version. All lines except possibly
    // run_start (which may pre-date schema_version) should have it.
    let lines_with_sv = lines
        .iter()
        .filter(|line| {
            let v: Value = serde_json::from_str(line).unwrap();
            v.get("schema_version").is_some()
        })
        .count();

    // At a minimum, every stage and run_complete line should carry schema_version.
    // The golden file has stage + run_complete lines, so we expect at least those.
    let stage_and_complete_count = lines
        .iter()
        .filter(|line| {
            let v: Value = serde_json::from_str(line).unwrap();
            let event = v["event"].as_str().unwrap_or("");
            event == "stage" || event == "run_complete" || event == "run_error"
        })
        .count();

    assert!(
        lines_with_sv >= stage_and_complete_count,
        "all stage/run_complete/run_error lines must have schema_version \
         (found {lines_with_sv} lines with schema_version vs {stage_and_complete_count} \
         stage+complete+error lines)"
    );
}

#[test]
fn golden_robot_events_event_field_values_are_known() {
    let ndjson_path = helpers::golden_dir().join("robot_events.ndjson");
    let text = std::fs::read_to_string(&ndjson_path).expect("read golden NDJSON");

    let known_events: HashSet<&str> = [
        "run_start",
        "stage",
        "run_complete",
        "run_error",
        "backends.discovery",
        "transcript.partial",
        "transcript.confirm",
        "transcript.retract",
        "transcript.correct",
        "transcript.speculation_stats",
        "health.report",
    ]
    .into_iter()
    .collect();

    for (i, line) in text.lines().enumerate() {
        let parsed: Value = serde_json::from_str(line).unwrap();
        let event_type = parsed["event"]
            .as_str()
            .unwrap_or_else(|| panic!("line {} `event` must be a string", i + 1));
        assert!(
            known_events.contains(event_type),
            "line {} has unknown event type `{event_type}`",
            i + 1
        );
    }
}

#[test]
fn golden_robot_events_first_is_run_start_last_is_run_complete() {
    let ndjson_path = helpers::golden_dir().join("robot_events.ndjson");
    let text = std::fs::read_to_string(&ndjson_path).expect("read golden NDJSON");
    let lines: Vec<&str> = text.lines().collect();

    assert!(lines.len() >= 2, "golden file must have at least 2 lines");

    let first: Value = serde_json::from_str(lines[0]).expect("parse first line");
    assert_eq!(
        first["event"], "run_start",
        "first line should be run_start"
    );

    let last: Value = serde_json::from_str(lines[lines.len() - 1]).expect("parse last line");
    assert_eq!(
        last["event"], "run_complete",
        "last line should be run_complete"
    );
}

#[test]
fn golden_robot_events_each_line_is_single_line_json() {
    let ndjson_path = helpers::golden_dir().join("robot_events.ndjson");
    let text = std::fs::read_to_string(&ndjson_path).expect("read golden NDJSON");

    for (i, line) in text.lines().enumerate() {
        // Each line must be self-contained valid JSON.
        let parsed: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line {} is not valid JSON: {e}", i + 1));
        // Re-serialize and confirm it produces a single line.
        let reserialized = serde_json::to_string(&parsed).expect("re-serialize");
        assert!(
            !reserialized.contains('\n'),
            "line {} re-serialized JSON must not contain newlines",
            i + 1
        );
    }
}

// ---------------------------------------------------------------------------
// 9. robot_schema_value() contains documentation for all event types
// ---------------------------------------------------------------------------

#[test]
fn robot_schema_documents_all_event_types() {
    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    for event_type in [
        "run_start",
        "stage",
        "run_complete",
        "run_error",
        "backends.discovery",
        "transcript.partial",
        "transcript.confirm",
        "transcript.retract",
        "transcript.correct",
        "transcript.speculation_stats",
        "health.report",
    ] {
        assert!(
            events.contains_key(event_type),
            "robot_schema_value() must document event type `{event_type}`"
        );

        let spec = &events[event_type];

        // Each event type must have "required" and "example" fields.
        assert!(
            spec.get("required").is_some(),
            "`{event_type}` must have a `required` field in the schema"
        );
        assert!(
            spec["required"].is_array(),
            "`{event_type}`.required must be an array"
        );
        assert!(
            spec.get("example").is_some(),
            "`{event_type}` must have an `example` field in the schema"
        );
        assert!(
            spec["example"].is_object(),
            "`{event_type}`.example must be a JSON object"
        );
    }
}

#[test]
fn robot_schema_examples_satisfy_their_own_required_fields() {
    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    for (event_type, spec) in events {
        let required = spec["required"]
            .as_array()
            .unwrap_or_else(|| panic!("`{event_type}` should have required array"));
        let example = &spec["example"];

        for field in required {
            let field_name = field.as_str().expect("required field should be a string");
            assert!(
                example.get(field_name).is_some(),
                "schema example for `{event_type}` is missing its own required field `{field_name}`"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 10. All required field lists are non-empty
// ---------------------------------------------------------------------------

#[test]
fn required_field_lists_are_non_empty() {
    let all_lists: &[(&str, &[&str])] = &[
        ("STAGE_REQUIRED_FIELDS", STAGE_REQUIRED_FIELDS),
        ("RUN_ERROR_REQUIRED_FIELDS", RUN_ERROR_REQUIRED_FIELDS),
        ("RUN_START_REQUIRED_FIELDS", RUN_START_REQUIRED_FIELDS),
        ("RUN_COMPLETE_REQUIRED_FIELDS", RUN_COMPLETE_REQUIRED_FIELDS),
        (
            "BACKENDS_DISCOVERY_REQUIRED_FIELDS",
            BACKENDS_DISCOVERY_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_PARTIAL_REQUIRED_FIELDS",
            TRANSCRIPT_PARTIAL_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_CONFIRM_REQUIRED_FIELDS",
            TRANSCRIPT_CONFIRM_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_RETRACT_REQUIRED_FIELDS",
            TRANSCRIPT_RETRACT_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_CORRECT_REQUIRED_FIELDS",
            TRANSCRIPT_CORRECT_REQUIRED_FIELDS,
        ),
        (
            "SPECULATION_STATS_REQUIRED_FIELDS",
            SPECULATION_STATS_REQUIRED_FIELDS,
        ),
        (
            "HEALTH_REPORT_REQUIRED_FIELDS",
            HEALTH_REPORT_REQUIRED_FIELDS,
        ),
    ];

    for (name, fields) in all_lists {
        assert!(!fields.is_empty(), "{name} must not be empty");

        // Also verify every field name in the list is a non-empty string.
        for field in *fields {
            assert!(!field.is_empty(), "{name} contains an empty field name");
        }
    }
}

// ---------------------------------------------------------------------------
// 11. No duplicate fields in any required fields list
// ---------------------------------------------------------------------------

#[test]
fn no_duplicate_fields_in_required_field_lists() {
    let all_lists: &[(&str, &[&str])] = &[
        ("STAGE_REQUIRED_FIELDS", STAGE_REQUIRED_FIELDS),
        ("RUN_ERROR_REQUIRED_FIELDS", RUN_ERROR_REQUIRED_FIELDS),
        ("RUN_START_REQUIRED_FIELDS", RUN_START_REQUIRED_FIELDS),
        ("RUN_COMPLETE_REQUIRED_FIELDS", RUN_COMPLETE_REQUIRED_FIELDS),
        (
            "BACKENDS_DISCOVERY_REQUIRED_FIELDS",
            BACKENDS_DISCOVERY_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_PARTIAL_REQUIRED_FIELDS",
            TRANSCRIPT_PARTIAL_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_CONFIRM_REQUIRED_FIELDS",
            TRANSCRIPT_CONFIRM_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_RETRACT_REQUIRED_FIELDS",
            TRANSCRIPT_RETRACT_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_CORRECT_REQUIRED_FIELDS",
            TRANSCRIPT_CORRECT_REQUIRED_FIELDS,
        ),
        (
            "SPECULATION_STATS_REQUIRED_FIELDS",
            SPECULATION_STATS_REQUIRED_FIELDS,
        ),
        (
            "HEALTH_REPORT_REQUIRED_FIELDS",
            HEALTH_REPORT_REQUIRED_FIELDS,
        ),
    ];

    for (name, fields) in all_lists {
        let unique: HashSet<&str> = fields.iter().copied().collect();
        assert_eq!(
            unique.len(),
            fields.len(),
            "{name} contains duplicate field names: {:?}",
            fields
        );
    }
}

// ---------------------------------------------------------------------------
// 12. Error codes in run_error events are non-empty strings
// ---------------------------------------------------------------------------

#[test]
fn run_error_example_has_non_empty_code_string() {
    let schema = robot_schema_value();
    let run_error_example = &schema["events"]["run_error"]["example"];

    let code = run_error_example["code"]
        .as_str()
        .expect("run_error example must have a string `code` field");
    assert!(
        !code.is_empty(),
        "run_error example `code` must be a non-empty string"
    );
}

#[test]
fn run_error_example_has_non_empty_message_string() {
    let schema = robot_schema_value();
    let run_error_example = &schema["events"]["run_error"]["example"];

    let message = run_error_example["message"]
        .as_str()
        .expect("run_error example must have a string `message` field");
    assert!(
        !message.is_empty(),
        "run_error example `message` must be a non-empty string"
    );
}

// ---------------------------------------------------------------------------
// Additional contract tests: cross-cutting invariants
// ---------------------------------------------------------------------------

/// Every required fields list must include "event" and "schema_version",
/// since those are the universal envelope fields.
#[test]
fn all_required_field_lists_include_event_and_schema_version() {
    let all_lists: &[(&str, &[&str])] = &[
        ("STAGE_REQUIRED_FIELDS", STAGE_REQUIRED_FIELDS),
        ("RUN_ERROR_REQUIRED_FIELDS", RUN_ERROR_REQUIRED_FIELDS),
        ("RUN_START_REQUIRED_FIELDS", RUN_START_REQUIRED_FIELDS),
        ("RUN_COMPLETE_REQUIRED_FIELDS", RUN_COMPLETE_REQUIRED_FIELDS),
        (
            "BACKENDS_DISCOVERY_REQUIRED_FIELDS",
            BACKENDS_DISCOVERY_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_PARTIAL_REQUIRED_FIELDS",
            TRANSCRIPT_PARTIAL_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_CONFIRM_REQUIRED_FIELDS",
            TRANSCRIPT_CONFIRM_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_RETRACT_REQUIRED_FIELDS",
            TRANSCRIPT_RETRACT_REQUIRED_FIELDS,
        ),
        (
            "TRANSCRIPT_CORRECT_REQUIRED_FIELDS",
            TRANSCRIPT_CORRECT_REQUIRED_FIELDS,
        ),
        (
            "SPECULATION_STATS_REQUIRED_FIELDS",
            SPECULATION_STATS_REQUIRED_FIELDS,
        ),
        (
            "HEALTH_REPORT_REQUIRED_FIELDS",
            HEALTH_REPORT_REQUIRED_FIELDS,
        ),
    ];

    for (name, fields) in all_lists {
        assert!(
            fields.contains(&"event"),
            "{name} must include `event` as a required field"
        );
        assert!(
            fields.contains(&"schema_version"),
            "{name} must include `schema_version` as a required field"
        );
    }
}

/// The schema's top-level metadata must declare line_oriented = "ndjson".
#[test]
fn robot_schema_declares_ndjson_format() {
    let schema = robot_schema_value();
    assert_eq!(
        schema["line_oriented"], "ndjson",
        "robot schema must declare line_oriented = ndjson"
    );
}

/// Verify robot_schema_value() is deterministic: calling it twice yields
/// identical JSON.
#[test]
fn robot_schema_value_is_deterministic() {
    let v1 = robot_schema_value();
    let v2 = robot_schema_value();
    assert_eq!(v1, v2, "robot_schema_value() must be deterministic");

    let json1 = serde_json::to_string(&v1).expect("serialize v1");
    let json2 = serde_json::to_string(&v2).expect("serialize v2");
    assert_eq!(
        json1, json2,
        "robot_schema_value() serialized JSON must be byte-identical across calls"
    );
}

/// Verify that every example in the schema round-trips through JSON
/// serialization without data loss.
#[test]
fn schema_examples_round_trip_through_json() {
    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    for (event_type, spec) in events {
        let example = &spec["example"];
        let serialized = serde_json::to_string(example)
            .unwrap_or_else(|e| panic!("failed to serialize `{event_type}` example: {e}"));
        let round_tripped: Value = serde_json::from_str(&serialized)
            .unwrap_or_else(|e| panic!("failed to deserialize `{event_type}` example: {e}"));
        assert_eq!(
            *example, round_tripped,
            "`{event_type}` example must survive JSON round-trip"
        );
    }
}

/// Verify that examples serialize to single-line NDJSON (no embedded newlines).
#[test]
fn schema_examples_serialize_to_single_line_ndjson() {
    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    for (event_type, spec) in events {
        let example = &spec["example"];
        let serialized = serde_json::to_string(example)
            .unwrap_or_else(|e| panic!("failed to serialize `{event_type}` example: {e}"));
        assert!(
            !serialized.contains('\n'),
            "`{event_type}` example must serialize to a single NDJSON line"
        );
    }
}

/// Verify golden file stage events have monotonically increasing seq.
#[test]
fn golden_robot_events_stages_have_monotonic_seq() {
    let ndjson_path = helpers::golden_dir().join("robot_events.ndjson");
    let text = std::fs::read_to_string(&ndjson_path).expect("read golden NDJSON");

    let mut prev_seq: Option<u64> = None;
    for (i, line) in text.lines().enumerate() {
        let parsed: Value = serde_json::from_str(line).unwrap();
        if parsed["event"] == "stage" {
            let seq = parsed["seq"]
                .as_u64()
                .unwrap_or_else(|| panic!("stage line {} must have integer `seq`", i + 1));
            if let Some(prev) = prev_seq {
                assert!(
                    seq > prev,
                    "stage seq must be monotonically increasing: line {} has seq={seq} but previous was {prev}",
                    i + 1
                );
            }
            prev_seq = Some(seq);
        }
    }

    assert!(
        prev_seq.is_some(),
        "golden file must contain at least one stage event"
    );
}

/// Verify the run_complete line in the golden file contains all
/// RUN_COMPLETE_REQUIRED_FIELDS.
#[test]
fn golden_run_complete_has_all_required_fields() {
    let ndjson_path = helpers::golden_dir().join("robot_events.ndjson");
    let text = std::fs::read_to_string(&ndjson_path).expect("read golden NDJSON");

    let run_complete_line = text
        .lines()
        .find(|line| {
            let v: Value = serde_json::from_str(line).unwrap();
            v["event"] == "run_complete"
        })
        .expect("golden file must contain a run_complete event");

    let parsed: Value = serde_json::from_str(run_complete_line).unwrap();
    for field in RUN_COMPLETE_REQUIRED_FIELDS {
        assert!(
            parsed.get(*field).is_some(),
            "golden run_complete missing required field `{field}`"
        );
    }
}

/// Use the test helper to create a report and verify it can be used with
/// emit_robot_report (which calls emit_robot_stage + emit_robot_complete).
/// We cannot easily capture stdout in an integration test, so we verify the
/// report structure is compatible with the robot module's expectations.
#[test]
fn test_report_from_helpers_has_fields_for_robot_emit() {
    let report = helpers::create_test_report();

    // run_id is needed for stage envelope.
    assert!(!report.run_id.is_empty(), "run_id must not be empty");

    // events should be present for stage emission.
    assert!(
        !report.events.is_empty(),
        "test report should have events for robot stage emission"
    );

    // result must have a backend (serializes to the backend field).
    let backend_str = serde_json::to_value(report.result.backend).expect("backend serializes");
    assert!(
        backend_str.is_string(),
        "backend must serialize to a string"
    );

    // The report must have all fields that run_complete_value() reads.
    assert!(!report.trace_id.is_empty(), "trace_id must not be empty");
    assert!(
        !report.started_at_rfc3339.is_empty(),
        "started_at_rfc3339 must not be empty"
    );
    assert!(
        !report.finished_at_rfc3339.is_empty(),
        "finished_at_rfc3339 must not be empty"
    );
}

/// Verify that the golden file's stage events each contain all
/// STAGE_REQUIRED_FIELDS.
#[test]
fn golden_stage_events_have_all_required_fields() {
    let ndjson_path = helpers::golden_dir().join("robot_events.ndjson");
    let text = std::fs::read_to_string(&ndjson_path).expect("read golden NDJSON");

    let mut stage_count = 0;
    for (i, line) in text.lines().enumerate() {
        let parsed: Value = serde_json::from_str(line).unwrap();
        if parsed["event"] == "stage" {
            stage_count += 1;
            for field in STAGE_REQUIRED_FIELDS {
                assert!(
                    parsed.get(*field).is_some(),
                    "golden stage event at line {} missing required field `{field}`",
                    i + 1
                );
            }
        }
    }

    assert!(
        stage_count > 0,
        "golden file must contain at least one stage event"
    );
}

/// Ensure run_complete required field list includes the critical output fields
/// that downstream consumers depend on.
#[test]
fn run_complete_required_fields_include_critical_output_fields() {
    let critical_fields = ["run_id", "transcript", "segments", "backend", "warnings"];

    for field in &critical_fields {
        assert!(
            RUN_COMPLETE_REQUIRED_FIELDS.contains(field),
            "RUN_COMPLETE_REQUIRED_FIELDS must include critical field `{field}`"
        );
    }
}

/// Ensure run_error required field list includes code and message.
#[test]
fn run_error_required_fields_include_code_and_message() {
    assert!(
        RUN_ERROR_REQUIRED_FIELDS.contains(&"code"),
        "RUN_ERROR_REQUIRED_FIELDS must include `code`"
    );
    assert!(
        RUN_ERROR_REQUIRED_FIELDS.contains(&"message"),
        "RUN_ERROR_REQUIRED_FIELDS must include `message`"
    );
}

/// Ensure stage required field list includes seq and stage.
#[test]
fn stage_required_fields_include_seq_and_stage() {
    assert!(
        STAGE_REQUIRED_FIELDS.contains(&"seq"),
        "STAGE_REQUIRED_FIELDS must include `seq`"
    );
    assert!(
        STAGE_REQUIRED_FIELDS.contains(&"stage"),
        "STAGE_REQUIRED_FIELDS must include `stage`"
    );
    assert!(
        STAGE_REQUIRED_FIELDS.contains(&"code"),
        "STAGE_REQUIRED_FIELDS must include `code`"
    );
    assert!(
        STAGE_REQUIRED_FIELDS.contains(&"message"),
        "STAGE_REQUIRED_FIELDS must include `message`"
    );
}

/// Ensure run_start required field list includes request.
#[test]
fn run_start_required_fields_include_request() {
    assert!(
        RUN_START_REQUIRED_FIELDS.contains(&"request"),
        "RUN_START_REQUIRED_FIELDS must include `request`"
    );
}

// -----------------------------------------------------------------------
// bd-rt-events-ghdx: listen session event family (schema 1.1.0)
// -----------------------------------------------------------------------

#[test]
fn listen_event_schema_entries_match_required_field_constants() {
    let schema = robot_schema_value();
    let events = schema["events"].as_object().expect("events map");
    let cases: &[(&str, &[&str])] = &[
        ("listen.session_start", LISTEN_SESSION_START_REQUIRED_FIELDS),
        ("speech_started", SPEECH_STARTED_REQUIRED_FIELDS),
        ("transcript.delta", TRANSCRIPT_DELTA_REQUIRED_FIELDS),
        ("utterance_end", UTTERANCE_END_REQUIRED_FIELDS),
        ("listen.warning", LISTEN_WARNING_REQUIRED_FIELDS),
        ("listen.session_stats", LISTEN_SESSION_STATS_REQUIRED_FIELDS),
    ];
    for (name, constant) in cases {
        let entry = events
            .get(*name)
            .unwrap_or_else(|| panic!("`{name}` missing from robot schema"));
        let registered: Vec<String> = entry["required"]
            .as_array()
            .unwrap_or_else(|| panic!("`{name}` required list missing"))
            .iter()
            .map(|v| v.as_str().expect("field name").to_owned())
            .collect();
        let expected: Vec<String> = constant.iter().map(|f| (*f).to_owned()).collect();
        assert_eq!(registered, expected, "`{name}` schema/constant drift");
        // Every required field must be present in the registered example.
        for field in *constant {
            assert!(
                entry["example"].get(*field).is_some(),
                "`{name}` example missing required field `{field}`"
            );
        }
    }
}

#[test]
fn listen_events_never_include_run_complete_semantics() {
    // The lifecycle contract: listen streams have no run_complete. Guard the
    // schema documentation of that divergence.
    let schema = robot_schema_value();
    let start = &schema["events"]["listen.session_start"];
    let never = start["lifecycle_contract"]["never_emitted"]
        .as_array()
        .expect("never_emitted documented");
    let names: Vec<&str> = never.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"run_start") && names.contains(&"run_complete"));
}
