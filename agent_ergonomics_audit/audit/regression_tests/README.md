# Regression test map

- CLI parsing, version disclosure, aliases: `src/cli.rs` unit tests.
- Capability, model, doctor, triage, and guide schemas: `src/robot.rs` unit tests.
- Both binary names, single-line JSON, syntax errors, and empty routing history: `tests/cli_integration.rs`.
- Installer parser and archive probes: recorded in post-pass simulations and re-run by release preparation.
- Exact sibling revisions: `scripts/prepare_release_siblings.sh --verify-only`.
- DSR five-target planning: external active config plus `dsr --json --dry-run build franken_whisper --allow-dirty`.
