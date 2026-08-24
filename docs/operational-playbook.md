# franken_whisper Operational Playbook

This playbook defines the phase-gated execution method for this repo.

## Mission Gate

Every packet must preserve these non-negotiable contracts:
- Rust-only, memory-safe implementation (`#![forbid(unsafe_code)]`).
- SQLite is canonical state via `frankensqlite` (`fsqlite` only).
- JSONL is adjunct audit/recovery, never competing source of truth.
- Robot mode remains stable, line-oriented NDJSON.
- Adaptive logic ships only with deterministic conservative fallback.

## Phase Model

### Phase A: Contract Foundation
Entry criteria:
- AGENTS + README + spec docs reread.
- Tracker packet created with granular sub-tasks.

Exit criteria:
- Scope locked in `TODO_IMPLEMENTATION_TRACKER.md`.
- No ambiguous ownership for code/docs/tests.

### Phase B: Orchestration and Storage Integrity
Entry criteria:
- Pipeline stage map is explicit.
- Sync invariants identified.

Exit criteria:
- Stage budgets/timeouts are explicit and observable.
- Sync lock/atomicity/conflict handling validated.
- Error paths emit deterministic machine-readable stage codes.

### Phase C: Backend Parity Packeting
Entry criteria:
- Current adapter behavior audited.
- Legacy parity deltas enumerated.

Exit criteria:
- Adapter option forwarding and fallback ordering hardened.
- Diarization normalization edge cases covered by tests.
- Diagnostics expose availability prerequisites clearly.

### Phase D: Interface and Operator UX
Entry criteria:
- Robot contract validated for machine consumers.
- Human/TUI pathways checked for regressions.

Exit criteria:
- Robot schema stability maintained.
- TUI remains optional and non-disruptive.
- Operational docs updated for maintainers.

## Execution Loop (Per Packet)

1. Update tracker statuses before and after each material change.
2. Implement one leverage group at a time (storage, orchestration, backend, docs).
3. Add/adjust tests immediately after each behavior change.
4. Run mandatory gates:
   - `cargo fmt --check`
   - `cargo check --all-targets`
   - `cargo clippy --all-targets -- -D warnings`
   - `cargo test`
5. Run feature checks relevant to this repo:
   - `cargo check --all-targets --features tui`
   - `cargo check --all-targets --features gpu-frankentorch`
   - `cargo check --all-targets --features gpu-frankenjax`
6. Reconcile tracker and document residual risks explicitly.

## Escalation Rules

Escalate before merge when any of the following occurs:
- Sync invariants are violated or unverifiable.
- Robot schema needs a breaking change.
- Stage budget policy causes non-deterministic behavior.
- Required external backend prerequisites are unclear to operators.

## Live Session Operations (`fw robot listen`)

- **RAM:** the confirm lane keeps a second model resident beside the fast
  lane (tiny + large-v3-turbo). Size the host for both, or run
  `--quality-model none`.
- **Headless capture:** macOS TCC microphone prompts never render over SSH.
  Pre-authorize the terminal app before deployment, prove enumeration with
  `fw robot listen --list-devices` (metadata-only; never prompts), and use
  `--source stdin-pcm` when no mic is trustworthy. `silent_input` warnings
  are the TCC-denial signature (zeros, not errors).
- **Thread sizing:** decode throughput follows the global Rayon pool built
  once at first use (`ensure_default_rayon_pool`); set `RAYON_NUM_THREADS`
  explicitly for latency-oriented sizing on dedicated boxes.
- **Durability:** sessions persist at utterance granularity
  (backend `native-listen`, savepoint per closed utterance); a crashed
  session is a listen run without its session-end marker. `--no-persist`
  opts out entirely.

## Primary References

- `AGENTS.md`
- `PLAN_TO_PORT_WHISPER_STACK_TO_RUST.md`
- `PROPOSED_ARCHITECTURE.md`
- `FEATURE_PARITY.md`
- `SYNC_STRATEGY.md`
- `RECOVERY_RUNBOOK.md`
- `docs/realtime-streaming.md`
- `TODO_IMPLEMENTATION_TRACKER.md`
