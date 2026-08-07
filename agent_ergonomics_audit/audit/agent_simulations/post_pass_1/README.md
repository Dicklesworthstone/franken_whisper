# Post-pass agent simulations

All probes used synthetic placeholders or discovery-only commands. No private audio or transcript was opened.

## Binary orientation

- Both packaged command names reported version `0.5.0` and identical extended distribution text.
- `fw robot --help` returned ordinary human help on stdout, exit 0, with `triage` visible.
- `fw capabilities --json`, `fw models --json`, `fw doctor --json`, `fw robot triage`, and `fw robot schema` each returned exactly one JSON line.
- Captured stderr for `models`, `doctor`, and `triage` was exactly zero bytes after removing model-path logging.
- Discovery reported four model entries without local paths or network access. The acoustic baseline was labeled `available_uncertified_heuristic`; Sortformer was labeled evaluation-only and operator-local.
- Doctor reported `runtime_probe_performed: false`, `operationally_verified: false`, and static-preflight authority even when it found a local execution candidate.

## Error and empty-result behavior

- A synthetic misspelling, `fw robot run --inpt SENSITIVE_SENTINEL.m4a`, returned exit 2, one JSON line, empty stderr, `FW-INVALID-REQUEST`, and zero sentinel occurrences in either stream.
- Empty routing history returned one `routing_history.complete` line with `records: 0` and empty stderr.

## Installer behavior

- A checksum-authenticated local archive containing both binaries installed into a fresh external destination.
- `--verify` executed both version probes plus capabilities, schema, and detect-only doctor contracts; it explicitly did not claim model/runtime readiness.
- A second quiet install passed after macOS `TMPDIR` normalization and cleaned the installer-owned temporary root without a safety warning.
- An archive without a primary binary and an otherwise-good archive without a checksum both exited 1 before installing a binary.

## DSR packaging

- The first native macOS build failed because optional FrankenJAX manifests were not staged; the release sibling contract was expanded to exact clean FrankenJAX and FrankenTUI commits as well as FrankenSQLite and FrankenTorch.
- The repaired native macOS build succeeded and produced a `franken_whisper-0.5.0-darwin_arm64.tar.gz` containing two regular executable members: `franken_whisper` and `fw`.
- The extracted release `fw` executed and exposed all 15 non-help top-level commands.
- A DSR compatibility-alias scrape defect was traced to a shell assignment in the installer and removed by constructing the download asset name with `printf -v`.

## Proof boundary

These simulations prove binary packaging and agent-interface behavior. They do not certify transcription quality, diarization accuracy, Sortformer accuracy, or cross-platform artifacts not built in this pass.
