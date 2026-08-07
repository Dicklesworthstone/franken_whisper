# Pre-pass simulations

Baseline source SHA: `4eb31dafbcfe45f714c3728a79af6464108da032`.

The current-source baseline binary exposed `transcribe`, `robot`, `runs`, `sync`, `tty-audio`, and human-oriented commands, but top-level help omitted useful descriptions for several core verbs. `--version`, `models --json`, `doctor --json`, `robot triage`, and `robot-docs guide` were unrecognized. `robot schema` emitted pretty multi-line JSON. A mistyped `robot run --inpt` produced Clap prose on stderr. Empty routing history produced no terminal record.

Installer simulations found two critical intent failures: `--offline` without a value reached a raw `set -u` error, while unknown `--ofline` was ignored and proceeded toward a real install attempt. The installed binary was left unchanged after that attempt was terminated. Release simulation found required sibling path dependencies absent in clean checkouts, advisory test gates, one-binary archives, and no active DSR repository configuration.

No confidential audio, transcript, speaker identity, path, embedding, or derived content was used in these simulations.
