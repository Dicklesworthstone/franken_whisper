# Regression alerts

> Historical pass-1 watch list. Sortformer distribution is now an explicit,
> hash-pinned release-cache workflow; evaluation-only and Auto-certification
> requirements remain current until the separate public gate passes.

No scored surface regressed below its pre-pass score.

Watch items that remain outside this pass:

- a full five-target DSR build/release is still required before claiming cross-platform artifact readiness;
- Sortformer remains evaluation-only for accuracy/Auto routing, while its
  licensed package is installed explicitly from the hash-pinned model release;
- the built-in acoustic diarizer remains an uncertified heuristic baseline;
- `doctor.ready` is static preflight evidence and intentionally includes `operationally_verified: false`;
- the pinned FrankenTorch revision emits a new-nightly stable-feature warning during ordinary builds; strict Clippy outcome is recorded by the repository gate, not hidden here;
- human installer output is not a machine JSON contract; agent automation should use exit status and then interrogate `fw capabilities --json`.
