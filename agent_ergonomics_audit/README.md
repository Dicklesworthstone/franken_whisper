# Agent Ergonomics Audit Workspace

For tool: `franken_whisper`
Target: repository root (`franken_whisper`)

This is a measurement workspace produced by the
`agent-ergonomics-and-intuitiveness-maximization-for-cli-tools` skill.

## Layout

- `audit/manifest.json` — entry point (pass number, target SHA, artifact paths)
- `audit/surface_inventory.jsonl` — every agent surface discovered
- `audit/agent_surfaces.jsonl` — surfaces scored across 11 dimensions
- `audit/intent_inference_corpus.jsonl` — wrong-invocation corpus + outcomes
- `audit/recommendations.jsonl` — ranked recommendations
- `audit/applied_changes.jsonl` — what was applied + commit refs
- `audit/scorecard_pass_1.md` — human-readable scorecard
- `audit/heatmap.svg` — surfaces × dimensions heatmap
- `audit/playbook.md` — top-10 narrative
- `audit/uplift_diff.md` — pass-N vs pass-N-1 deltas
- `audit/regression_alerts.md` — surfaces that dropped scores
- `audit/regression_tests/` — golden/snapshot tests
- `audit/agent_simulations/` — fresh-agent canonical-task transcripts
- `audit/HANDOFF.md` — what's queued for next pass

## How to resume

This workspace lives **inside the target repo** at `agent_ergonomics_audit/` and is committed alongside the code on the target's current branch (typically `main`). The phase-loop scripts live in the **skill repo**, not in this workspace. From the skill repo's root, run:

1. `<SKILL>/scripts/discover-cli.sh "$REPO"` to confirm the binary still exists.
2. `<SKILL>/scripts/validate_pass.sh "$REPO/agent_ergonomics_audit"` to check artifact integrity.
3. Read `audit/HANDOFF.md` here in the workspace.
4. Pick a mode and send the resumed-pass kickoff prompt.
