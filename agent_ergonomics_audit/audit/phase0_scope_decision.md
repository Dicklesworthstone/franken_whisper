# Phase 0 scope decision

> Historical scope boundary for pass 1. The repository owner later explicitly
> authorized model release distribution, superseding only the operator-local
> packaging guardrail below. Privacy, accuracy-certification, and Auto-routing
> gates remain binding.

- Target: `/Users/jemanuel/projects/franken_whisper`
- Reference implementation: `/Users/jemanuel/projects/franken_ocr`
- Mode: `full`
- Primary agent profile: Codex CLI, with machine-readable behavior also checked for Claude Code and smaller agents
- Audit workspace: `/Users/jemanuel/projects/franken_whisper/agent_ergonomics_audit/` (in-tree; no sibling repository)
- Branch policy: remain on the current `main` branch; do not create a feature branch
- CASS mining: quick
- Triangulation: peer agents for comparison and fresh-eyes review

## Required outcomes

1. Make first-run installation, model discovery, health diagnosis, transcription, and diarization substantially easier for agents.
2. Reuse proven packaging and self-documentation patterns from `franken_ocr` where they fit the ASR domain.
3. Provide deterministic, schema-versioned machine output and copy-ready next commands.
4. Pin every new public contract with focused regression tests.

## Guardrails

- Preserve confidential-audio and transcript boundaries. No private media, transcript text, private paths, embeddings, or derived content may enter this repository or its audit artifacts.
- Preserve the authenticated Sortformer policy `operator_local_no_git_no_release`; this pass may improve guidance and diagnostics but may not publish or silently download restricted model artifacts.
- Preserve evaluation-only and rollout/certification boundaries. Packaging must not imply production accuracy or auto-route uncertified diarization engines.
- Do not weaken correctness, conformance, performance, privacy, or evidence gates.
- Do not delete files, use destructive Git operations, rewrite unrelated concurrent work, or create compatibility shims.
- Do not install a missing toolchain without explicit approval.
- Keep feature implementation out of scope except where a packaging or agent-discovery surface requires a small supporting read-only capability.

## Completion proof

- Pre/post agent simulations for canonical tasks.
- Focused contract tests for every changed public surface.
- Repository-mandated Rust quality gates, with scoped versus pre-existing failures reported separately.
- Privacy scan over every owned changed path.
- Beads state and dependency graph synchronized and truthful.
