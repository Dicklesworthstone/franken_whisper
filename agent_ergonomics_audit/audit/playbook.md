# Pass 1 playbook

> Historical playbook. Current model diagnostics remain local and path-free,
> while `fw pull sortformer` is the one explicit network-enabled provisioning
> command for the hash-pinned native Sortformer release package.

1. Make the binary self-describing: `fw --version`, `fw capabilities --json`, and `fw robot-docs guide` must work without a repository checkout.
2. Give agents one live entry point: `fw robot triage` returns readiness and the next exact command.
3. Keep model diagnostics local and path-free: `fw models --json` performs no network access and preserves Sortformer's operator-local boundary.
4. Make robot output machine-pure from argument parsing through terminal execution errors.
5. Publish one exact code per error variant plus the four process-exit classes.
6. Use one deterministic local model-selection policy for both availability and execution.
7. Treat installer parsing as a safety boundary; typos and missing values must fail before downloads or destination writes.
8. Authenticate archives, validate staged binary versions, and install both supported names.
9. Make DSR authoritative with pinned sibling commits and never ship through an advisory gate.
10. Pin every public behavior with unit or process-level regression coverage, then repeat two fresh-eyes passes.
