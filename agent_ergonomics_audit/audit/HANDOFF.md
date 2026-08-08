# Pass 1 handoff

> Historical handoff. The later Sortformer distribution tranche supersedes the
> operator-local packaging constraint below with a licensed, hash-pinned model
> release and explicit native pull command. It does not retroactively change
> the evidence observed during this pass or certify diarization accuracy.

Pass 1 implemented the agent-first command surface, deterministic local model
discovery, fail-closed dual-binary installer, and DSR-first release staging.
Two fresh-eyes reviews corrected robot help handling, zero-duration metrics,
readiness overclaims, path leakage, installer replacement and lock races, archive
name parsing, and release pinning.

Evidence completed in this pass:

- live process probes cover `--version`, root help, capabilities, models,
  doctor, robot triage, robot docs, invalid robot syntax, and empty routing
  history;
- robot probes produce one JSON line on stdout and no stderr, including argument
  failures, and do not echo a sentinel input path;
- model and doctor output is path-free and labels the acoustic diarizer as an
  uncertified heuristic and doctor readiness as static preflight only;
- installer probes cover authenticated dual-binary installation, verification,
  unknown options, missing values, and archive allowlisting;
- no confidential input was accessed or copied; ignored media and transcript
  patterns remain absent from the changed surface and will be checked again on
  the exact staged candidate;
- `cargo fmt --check` and `cargo check --all-targets` pass;
- UBS ran over the changed Rust surface; its heuristic test-token and panic
  findings did not identify a confirmed new production defect;
- one dirty-tree, proof-only DSR `darwin/arm64` artifact was built and inspected:
  run `78a8f7f1-a218-419f-a599-be58021ec325`, 22,421,849-byte archive,
  SHA-256 `bd2cfd662584009dc0108b1b806e0c2fdc708733813b3915b683c963982b78f9`;
  both executable members reported exactly version `0.5.0`, agent JSON stayed
  one-line/path-free, and the checksum-enforcing offline installer installed
  and verified both names from that archive;
- a DSR dry run resolved all five configured targets and source sync.

Evidence boundaries and next work:

- `cargo clippy --all-targets -- -D warnings` remains red on broad existing
  repository debt tracked by `bd-ii7l`; the gate was not weakened;
- the full tracked-tree privacy guard remains red on four historical raw
  performance artifacts introduced by commit `cf3ec1c4`; none is part of this
  pass, and the no-file-deletion rule prevents silently removing them;
- a successful DSR build does not execute the configured check suite, and a
  dirty-tree manifest records the current HEAD rather than a digest of edits;
- build all five configured DSR targets from a clean landed commit before
  claiming cross-platform release readiness;
- at the time of this historical pass, Sortformer was evaluation-only,
  operator-local, and excluded from release assets; later work preserves the
  evaluation-only accuracy boundary while distributing the licensed,
  hash-pinned package through an explicit model release outside Git;
- the acoustic diarizer remains an uncertified heuristic baseline;
- run a held-out, speaker-balanced corpus benchmark before making any
  diarization quality or promotion claim.
