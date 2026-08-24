# Definition of Done

A change packet is done only when all sections below pass.

## 1. Scope Integrity

- Scope is captured in `TODO_IMPLEMENTATION_TRACKER.md` with granular checkboxes.
- All changed files map to an explicit tracker item.
- No hidden/untracked behavior changes remain.

## 2. Runtime/Orchestration Contract

- Stage order remains deterministic (`ingest -> normalize -> backend -> acceleration -> persist`).
- Stage budget policy is explicit and observable in run events.
- Timeout/cancel outcomes produce deterministic machine-readable stage/error codes.
- Conservative fallback behavior is preserved when adaptive logic is uncertain.

## 3. Robot Mode Contract

- Output remains line-oriented NDJSON.
- Envelope contract remains stable:
  - `run_start`
  - `stage`
  - `run_complete`
  - `run_error`
- No human decoration mixed into robot output payloads.

## 4. Storage + Sync Contract

- SQLite (`fsqlite`) remains source of truth.
- Sync is one-way, locked, and atomic.
- Manifest/version/checksum validation is enforced on import.
- Conflict policy is explicit (`reject` default, `overwrite` opt-in).
- Recovery artifacts (`sync_conflicts.jsonl`, JSONL snapshots) are deterministic.

## 5. Test Coverage Minimums

- Unit tests cover happy path, edge cases, and failure paths for changed modules.
- Added/updated tests validate new timeout/sync/backend behavior contracts.
- No failing tests in default or required feature configurations.

## 6. Mandatory Quality Gates

All must pass before handoff:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
cargo check --all-targets --features tui
cargo test --features fj-oracle --test conformance_oracle_tests
```

## 7. Documentation and Handoff

- Relevant docs updated (`README`, architecture/parity/spec docs as needed).
- Tracker reconciled with explicit done/pending/blocker states.
- Residual risks and concrete next steps documented.
