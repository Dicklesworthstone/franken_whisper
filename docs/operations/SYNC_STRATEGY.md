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
Import rebuilds the indexes from that canonical state after the runs,
segments, and events transaction commits. This avoids a second source of truth
and keeps old snapshots forward-migratable.

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
set can be applied with `overwrite-strict`. The current JSONL format does not
carry deleted-run tombstones: deleting a parent advances the local high-water
mark, but an import into another database does not remove that run solely
because it disappeared from an incremental snapshot.

### Import (`jsonl -> db`)
1. Acquire lock.
2. Validate manifest + schema compatibility.
3. Begin DB transaction.
4. Replay JSONL in deterministic order (`runs`, `segments`, `events`).
5. Apply conflict policy by stable keys (`reject` default, `overwrite`/`overwrite-strict` opt-in).
6. Enforce overwrite safety constraints in current runtime:
   - `runs` parent-row replacement is allowed.
   - `overwrite` is fail-closed for child-row `UPDATE`/`DELETE` on `segments`/`events`.
   - `overwrite-strict` performs verified child-row replacement for imported runs:
     - conflicting child rows are replaced via delete+insert,
     - stale child rows not present in import are pruned.
7. Commit transaction.
8. Rebuild derived native-diarization indexes from canonical run JSON in a
   rollback-safe savepoint.
9. Release lock.

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
- typed native-diarization request/result decoding during derived-index rebuild.

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
