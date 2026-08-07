# Regression alerts

No scored surface regressed below its pre-pass score.

Watch items that remain outside this pass:

- a full five-target DSR build/release is still required before claiming cross-platform artifact readiness;
- Sortformer remains evaluation-only and operator-local;
- the built-in acoustic diarizer remains an uncertified heuristic baseline;
- `doctor.ready` is static preflight evidence and intentionally includes `operationally_verified: false`;
- the pinned FrankenTorch revision emits a new-nightly stable-feature warning during ordinary builds; strict Clippy outcome is recorded by the repository gate, not hidden here;
- human installer output is not a machine JSON contract; agent automation should use exit status and then interrogate `fw capabilities --json`.
