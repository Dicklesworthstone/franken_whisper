# SYNC_STRATEGY.md

## Source of Truth

`frankensqlite` (`fsqlite`) database is canonical state.

JSONL is an adjunct audit/export medium for:
- git-friendly review,
- cross-machine transfer,
- recovery workflows.

## Scope

Tables:
- `runs`
- `segments`
- `events`
- internal incremental-export authority:
  - `sync_run_mutations` (one latest database sequence per run aggregate)
- derived native-diarization indexes:
  - `diarization_reports`
  - `diarization_turns`
  - `speaker_hints`
  - `speaker_profile_summaries`

JSONL channels:
- `runs.jsonl`
- `segments.jsonl`
- `events.jsonl`

The four diarization tables are deterministic indexes, not independent JSONL
channels. Their canonical typed request/result state lives in
`runs.request_json` and `runs.result_json`, and therefore in `runs.jsonl`.
Import rebuilds the indexes from that canonical state **inside the same
transaction** as the runs, segments, and events replay, before commit. The
rebuild also has its own rollback-safe savepoint. This avoids a second source
of truth and prevents a rejected report from committing canonical rows without
their corresponding indexes. A rebuild failure rolls back the whole import,
including any earlier inserts, replacements, deletions, and mutation-journal
changes.

## One-Way Operations Only

Allowed operations:
1. `db -> jsonl` export snapshot
2. `jsonl -> db` import replay

Disallowed:
- implicit two-way merge
- concurrent export+import against same DB

## Locking Model

Before sync, create lock file under `.franken_whisper/locks/sync.lock`.

Rules:
- lock acquisition is mandatory.
- stale lock detection based on timestamp + PID verification.
- sync aborts if lock is active and not stale.

## Snapshot Semantics

### Export (`db -> jsonl`)
1. Acquire lock.
2. Write export manifest (`manifest.json`) with:
- schema version
- created timestamp
- source DB path
- row counts
- checksum placeholders
3. Stream each table to temp `*.jsonl.tmp`.
4. Flush + fsync temp files.
5. Atomic rename `*.tmp -> *.jsonl`.
6. Update manifest checksums.
7. Release lock.

### Incremental Export (`db -> jsonl`, library API)

Schema v6 triggers replace the single `sync_run_mutations` row for a run when
its parent, segment, or event rows change. The resulting SQLite-assigned
sequence, rather than `finished_at` or another caller-owned timestamp, is the
incremental selection authority. A stable random `sync_database_id` in `_meta`
binds that database-local sequence to its lineage.

Within one read transaction, incremental export reads the sequence high-water
mark, selects every current run above the prior cursor, and exports each
selected run with its complete segment and event set. A concurrent commit after
that snapshot remains above the published cursor for the next export. The
manifest is published before `sync_cursor.json`; any failure retains the old
cursor and may repeat work, but cannot skip an unpublished delta. A legacy
timestamp-only cursor defaults to sequence zero and causes one safe full
re-export. More generally, any legacy cursor without a database identity
ignores its numeric sequence for that one upgrade export. A cursor with a
different database identity or a sequence outside the current database range
fails closed before snapshot publication.

Deleting a segment or event marks its surviving parent, whose complete child
set can be applied with `overwrite-strict`.

Deleting a parent run is published through an explicit, versioned channel
(`deleted-runs-v1`). The `runs` delete trigger refreshes the run's mutation
row, so within the same snapshot every run whose mutation row is above the
cursor but whose `runs` row is absent is written to `deleted_runs.jsonl` as
`{"run_id": ..., "mutation_seq": ...}`. The incremental manifest declares the
channel under `deleted_runs` (`format_version`, `file`, `count`, `sha256`). A
re-inserted run is a live run again and is exported normally, never as a
tombstone.

Import honors deletions only when the manifest declares the channel; absence
from a partial snapshot never implies deletion, and a stray undeclared
`deleted_runs.jsonl` is ignored. A declared channel fails closed on an unknown
format version, a non-canonical file name, a missing file, a checksum or count
mismatch, a malformed or duplicate record, or a run that the same archive both
carries and deletes. Inside the import transaction each listed run that exists
in the target is removed with its segments and events (derived diarization
rows are rebuilt from the surviving canonical runs). Under `skip`, an existing
target run is preserved and reported as a conflict instead. The import result
reports `runs_deleted`.

### Import (`jsonl -> db`)
1. Acquire lock.
2. Validate manifest, schema and export-format compatibility, channel checksums,
   and any declared deleted-run channel before opening the target database.
3. Open the database, ensure its schema exists, and begin a DB transaction.
4. Replay JSONL in deterministic order (`runs`, `segments`, `events`) and
   reconcile row counts and parent references.
5. Apply conflict policy by stable keys (`reject` default, `overwrite`/`overwrite-strict` opt-in).
6. Enforce overwrite safety constraints in current runtime:
   - `runs` parent-row replacement is allowed.
   - `overwrite` is fail-closed for child-row `UPDATE`/`DELETE` on `segments`/`events`.
   - `overwrite-strict` performs verified child-row replacement for imported runs:
     - conflicting child rows are replaced via delete+insert,
     - stale child rows not present in import are pruned.
7. Rebuild derived native-diarization indexes from validated canonical run JSON
   within a savepoint nested in the still-open import transaction.
8. Commit only after replay, conflict handling, and reconstruction succeed;
   otherwise roll back the import transaction.
9. Release lock on success or failure.

Schema initialization can leave an empty, initialized target database after a
semantic rejection. It must not leave partially imported runs or derived rows.
Checksum rejection occurs before target database creation. An empty snapshot
is valid; it initializes a fresh target and does not implicitly delete existing
runs from a populated target.

## Conflict Strategy

Default import mode: `reject` (`sync import-jsonl --conflict-policy reject`).

Rules:
- same primary key + same payload: no-op.
- same primary key + different payload: reject unless explicitly set to overwrite via
  `sync import-jsonl --conflict-policy overwrite` or strict overwrite via
  `sync import-jsonl --conflict-policy overwrite-strict`.
- `overwrite` does **not** imply unrestricted in-place mutation:
  - if resolving a conflict requires child-row `UPDATE` (`segments`/`events` same key, different payload), import fails closed.
  - if strict replacement requires deleting stale child rows not present in import, import fails closed.
- `overwrite-strict` is the explicit in-place strict replacement mode for imported runs.
- all conflicts logged to `sync_conflicts.jsonl`.

## Integrity

Integrity checks:
- per-file SHA-256 checksum in manifest.
- row-count reconciliation.
- referential checks: `segments.run_id` and `events.run_id` must reference existing `runs.id`.
- typed native-diarization request/result decoding and semantic validation during
  derived-index rebuild.

### Diarization Contract Reconstruction

Storage reconstruction recognizes these current report contracts:

| Contract | Producer family |
| --- | --- |
| `acoustic-diarization-v3` | Native acoustic and the compatible common report paths |
| `neural-diarization-common-v2` | Common neural diarization report paths |
| `sortformer-diarization-v1` | Native Sortformer activity reports, including the default auto path |

Recognition is only the first check. `RunStore` decodes the typed
`DiarizationReport` and uses the same `validate_against_request` boundary during
ordinary persistence and reconstruction. Validation binds the implementation,
feature schema, evidence mode, engine/fallback policy, speaker-count semantics,
turns, speaker references, enrollment digest, and provenance to the canonical
request. Unknown contracts and malformed or incompatible known contracts fail
closed with run-specific context. A report without an enabled canonical
diarization request is not accepted.

The compact request `diarize=true` with no explicit
`backend_params.acoustic_diarization` resolves to `DiarizationRequest::default()`
on both paths. Reconstruction must not reinterpret it as an explicit acoustic
request. Sortformer activity evidence remains Sortformer evidence; its valid
unresolved count outcome does not acquire a fabricated acoustic count estimate.
Native acoustic reports retain their required `speaker-count-estimate-v2`
payload. The shared validator, rather than schema relabeling or a generic
permissive deserializer, determines which combinations are valid.

Reconstruction preserves canonical run IDs and JSON payloads. The derived rows
retain the original contract marker, normalized-input and enrollment digests,
speaker references, timings, confidence, hint provenance, and privacy-safe
profile summaries. A newly initialized target has its own database-local sync
identity and mutation sequence; those are not copied from the source's local
incremental cursor.

Issue #19 reported a binary rejecting `sortformer-diarization-v1` while naming
only the acoustic and neural contracts. The Sortformer allowlist change is
present in source commit `248e6eff4e6899d57e25880a8d5aa939ad5400ed`. Source
support does not establish which revision an installed binary contains. Keep
the original snapshot intact and use a verified compatible importer; do not
edit its contract marker or disable checksums to force a restore.

### Recovery Regression Tests

`tests/recovery_smoke.rs` exercises the complete public
`persist_report -> sync::export -> sync::import` boundary:

- Default Sortformer and explicit acoustic snapshots restore into fresh
  databases. Tests compare every column in all three canonical and all four
  derived tables, validate the source request/report pair, compare restored
  typed reports, segments, events, and replay metadata, and require identical
  JSONL bytes on re-export. Re-import is checked for idempotence.
- Correctly checksummed malformed contracts are rejected against both fresh
  and populated databases. The matrix covers unknown versions, missing count
  state, incompatible feature/evidence schemas, invalid digests, invalid turn
  geometry/confidence, unknown report fields, disabled requests, and an
  incompatible explicit engine. Rollback checks include existing derived rows,
  the local mutation journal, and database identity metadata.
- Corruption of each checksummed channel is rejected before creating a fresh
  target and without modifying a populated target. Empty snapshots are valid
  and never imply deletion of existing runs.

The Sortformer fixture is a model-free canonical report, not a claim that
neural inference is exercised. The acoustic fixture runs the native acoustic
producer on deterministic synthetic PCM with a provenance-bearing enrollment
interval, so its estimate and enrollment digest come from the actual producer.
No model download or external transcription executable is required by these
recovery fixtures.

Run the recovery suite and the existing storage/sync unit suites in the normal
Rust workspace with its documented sibling dependencies:

```bash
cargo test --test recovery_smoke
cargo test --lib storage::tests
cargo test --lib sync::tests
cargo fmt --check
```

## Diarization Privacy Boundary

The persisted native-diarization contract includes speaker turns, opaque
within-run speaker references, hint provenance, fallback/diagnostic metadata,
and aggregate profile-quality summaries. It does not persist normalized PCM,
spectra, cepstra, pitch tracks, embeddings, covariance matrices, or reusable
speaker-profile vectors.

`persist_profiles` is default-off and is recorded as an explicit opt-in bit.
The v1 acoustic implementation still does not serialize reusable biometric
vectors when that bit is true; adding such a record requires a separately
versioned, encrypted, retention-bounded design and new privacy review.

## Versioning

Manifest carries:
- `schema_version`
- `export_format_version`

Import rules:
- exact major match required.
- minor mismatch allowed only with backward-compatible fields.

## Failure Recovery

Any failed sync operation must:
- preserve previous committed DB state,
- keep temp artifacts for forensic analysis,
- emit machine-readable error with operation stage.

See `RECOVERY_RUNBOOK.md` for exact procedures.
