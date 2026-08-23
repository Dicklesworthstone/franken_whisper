# Conformance Test Coverage Matrix

> Generated: 2026-04-15
> Spec: `docs/engine_compatibility_spec.md`
> Harness: `tests/conformance_harness.rs`, `tests/conformance_comparator_tests.rs`

## Coverage Accounting Matrix

| Spec Section | MUST Clauses | SHOULD Clauses | Tested | Passing | Divergent | Score |
|-------------|:------------:|:--------------:|:------:|:-------:|:---------:|:-----:|
| 1. Segment Timestamp Invariants | 3 | 0 | 3 | 3 | 0 | 100% |
| 2. Confidence Calibration | 2 | 1 | 3 | 3 | 0 | 100% |
| 3. Speaker Labels | 2 | 1 | 3 | 3 | 0 | 100% |
| 4. Text Normalization | 2 | 1 | 3 | 3 | 0 | 100% |
| 5. Engine Trait Contract | 5 | 0 | 5 | 5 | 0 | 100% |
| 6. Replay Envelope | 4 | 0 | 4 | 4 | 0 | 100% |
| 7. Enforcement | 5 | 0 | 5 | 5 | 0 | 100% |
| 8. Runtime Lifecycle Contract | 8 | 2 | 10 | 10 | 0 | 100% |
| 9. Compatibility Envelope | 5 | 0 | 5 | 5 | 0 | 100% |
| **TOTAL** | **36** | **5** | **41** | **41** | **0** | **100%** |

## Test Inventory

### Unit Tests (`src/conformance.rs`)

| Category | Test Count | Coverage |
|----------|:----------:|:--------:|
| Timestamp invariants | 18 | Complete |
| Confidence bounds | 8 | Complete |
| Speaker label validation | 6 | Complete |
| Segment comparison | 14 | Complete |
| Replay envelope comparison | 8 | Complete |
| Rollout stage parsing | 12 | Complete |
| Shadow run comparison | 5 | Complete |
| Edge cases (epsilon, overflow) | 12 | Complete |
| **Total** | **83** | |

### Fixture-Driven Tests (`tests/conformance_harness.rs`)

| Test | Fixture Count | Purpose |
|------|:-------------:|---------|
| `conformance_fixtures_match_declared_expectations` | 37 | Segment comparison |
| `replay_envelope_fixtures_match_declared_expectations` | 7 | Replay drift detection |
| `golden_corpus_cross_engine_parity_harness` | 15 | Cross-engine parity |
| `parse_srt_segments_for_fixture_handles_crlf_line_endings` | 1 | SRT parser edge case |
| **Total** | **59** | |

### Integration Tests (`tests/conformance_comparator_tests.rs`)

| Category | Test Count |
|----------|:----------:|
| Comparator behavior | 25 |

## Fixture Coverage by Spec Section

### Section 1: Segment Timestamp Invariants

| Invariant | Fixture | Status |
|-----------|---------|:------:|
| 1.1 Internal ordering (end >= start) | `invalid_timestamp_ordering.json` | PASS |
| 1.2 Monotonicity (non-overlap) | `overlapping_speakers_rejected.json` | PASS |
| 1.2 Overlap allowed via policy | `overlapping_speakers_policy.json`, `overlap_allowed_policy.json` | PASS |
| 1.3 50ms tolerance | `timestamp_edge_at_tolerance.json`, `timestamp_epsilon_boundaries.json` | PASS |
| 1.3 Zero tolerance | `timestamp_zero_tolerance_pass.json`, `timestamp_zero_tolerance_fail.json` | PASS |
| Timestamp boundary violations | `timestamp_boundary_violation.json`, `timestamp_over_tolerance_fails.json` | PASS |
| End before start within epsilon | `timestamp_end_before_start_within_epsilon.json` | PASS |

### Section 2: Confidence Calibration

| Invariant | Fixture | Status |
|-----------|---------|:------:|
| 2.1 Range [0.0, 1.0] | `confidence_out_of_bounds_invariants.json` | PASS |
| 2.2 Semantics (drift ignored) | `confidence_drift_ignored.json` | PASS |
| 2.3 None confidence valid | Unit tests in `src/conformance.rs` | PASS |

### Section 3: Speaker Labels

| Invariant | Fixture | Status |
|-----------|---------|:------:|
| 3.1 Non-empty after trim | `speaker_label_whitespace_rejected.json` | PASS |
| 3.2 Recognized patterns | `speaker_label_crossengine.json` | PASS |
| 3.3 Cross-engine stability | `speaker_only_drift.json`, `speaker_exact_match_pass.json` | PASS |
| Unicode speaker labels | `unicode_speaker_labels.json` | PASS |

### Section 4: Text Normalization

| Invariant | Fixture | Status |
|-----------|---------|:------:|
| 4.1 Trimming | `text_trim_normalization.json` | PASS |
| 4.2 Cross-engine exact match | `text_case_mismatch_strict.json` | PASS |
| 4.2 Relaxed matching | `text_mismatch_allowed_relaxed.json` | PASS |
| 4.3 Empty segments | `empty_and_punctuation_segments.json`, `empty_text_invariants.json` | PASS |

### Section 5: Engine Trait Contract

| Method | Test Coverage | Status |
|--------|---------------|:------:|
| `name()` | `backend::tests::*_engine_metadata` | PASS |
| `kind()` | `backend::tests::*_engine_metadata` | PASS |
| `capabilities()` | `backend::tests::*_capabilities_*` | PASS |
| `is_available()` | `backend::tests::test_probe_system_health_*` | PASS |
| `run()` | Full conformance harness | PASS |

### Section 6: Replay Envelope

| Field | Fixture | Status |
|-------|---------|:------:|
| `input_content_hash` | `replay/input_hash_drift.json` | PASS |
| `backend_identity` | `replay/backend_identity_drift.json` | PASS |
| `backend_version` | `replay/backend_version_upgrade.json` | PASS |
| `output_payload_hash` | `replay/output_drift.json` | PASS |
| All fields identical | `replay/identical_envelopes.json` | PASS |
| All fields different | `replay/all_fields_different.json` | PASS |
| Missing fields | `replay/missing_fields.json` | PASS |

### Section 9: Compatibility Envelope (Release Gates)

| Gate | Fixture Coverage | Status |
|------|------------------|:------:|
| 9.1 Text parity | All corpus fixtures | PASS |
| 9.2 Timestamp tolerance | All corpus fixtures | PASS |
| 9.3 Speaker stability | Corpus fixtures with `pair_drift_caps` | PASS |
| 9.4 Confidence comparability | `confidence_*` fixtures | PASS |
| 9.5 Release-gate matrix | `golden_corpus_cross_engine_parity_harness` | PASS |

## Corpus Tag Coverage

Required tags (per `REQUIRED_CORPUS_TAGS`):

| Tag | Fixture Count | Status |
|-----|:-------------:|:------:|
| `long_form` | 1 | PASS |
| `multilingual` | 2 | PASS |
| `multi_speaker_overlap` | 1 | PASS |
| `silence_heavy` | 1 | PASS |
| `noisy_environment` | 2 | PASS |
| `code_switching` | 1 | PASS |
| `short_utterance` | 1 | PASS |
| `variable_volume_overlap` | 2 | PASS |

## Engine Family Coverage

| Engine Family | Bridge | Native | Status |
|---------------|:------:|:------:|:------:|
| whisper_cpp | YES | YES | PASS |
| insanely_fast | YES | YES | PASS |
| whisper_diarization | YES | YES | PASS |

## Known Divergences

See [DISCREPANCIES.md](../docs/planning/DISCREPANCIES.md) for intentional deviations:

| ID | Description | Resolution |
|----|-------------|:----------:|
| DISC-001 | SRT timestamp precision | ACCEPTED |
| DISC-002 | Speaker label remapping | ACCEPTED |

## Gaps and TODOs

- [x] ~~No fixture for `NaN`/`Infinity` timestamp edge cases~~ → `nonfinite_timestamp_invariants.json` (null proxy; actual NaN/Inf not expressible in JSON)
- [x] ~~No fixture for extremely long segments (> 10 minutes)~~ → `long_segment_duration.json` (10 min), `very_long_segment_15min.json` (15 min)
- [x] ~~Word-level timestamp conformance is minimal~~ → `word_level_boundary_cases.json` added (single-char words, punctuation segments)

## Commands

```bash
# Run full conformance suite
rch exec -- cargo test --test conformance_harness

# Run comparator tests
rch exec -- cargo test --test conformance_comparator_tests

# Run unit tests
rch exec -- cargo test conformance::tests

# Generate compliance report (JSON artifact)
rch exec -- cargo test golden_corpus_cross_engine_parity_harness -- --nocapture 2>&1 | grep -A1000 "ConformanceArtifactBundle"
```

## Maintenance

- **Fixture provenance**: Fixtures in `corpus/` generated via `scripts/gen_native_fixtures.py`
- **Update workflow**: Run `UPDATE_GOLDENS=1 cargo test` to regenerate golden files (if applicable)
- **Review cadence**: Regenerate and diff-review fixtures on each engine update
