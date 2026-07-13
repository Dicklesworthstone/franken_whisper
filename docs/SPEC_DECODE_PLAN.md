# Speculative Decode — implementation plan (the sole remaining perf lever)

> Actionable build plan for `bd-wzgh`. Rationale, viability data, and the "why this is the only
> lever left" are in `NEGATIVE_EVIDENCE.md` (CONSOLIDATED FRONTIER MAP + OWNER DECISION BRIEF).
> This file is the "how to build it." Author: GoldenOwl 2026-07-13.

## Why (one line)
Realistic long audio is **decode-dominated (62.5% of e2e)** and DRAM-weight-streaming-bound
(~342 MB/token re-streamed). Batch-verifying K drafted tokens in ONE all-layer pass amortizes that
stream K×. **Greedy-verify is BYTE-EXACT vs greedy** (accept only tokens the full model's argmax
agrees with), so this ships to the default quality path — no WER gate.

## ⚠ CORRECTION 2026-07-13 — the DRAFT side is BLOCKED (both cheap drafts measured DEAD); this feature is owner/infra-gated
Reading `[[project_draft_decoding_amortization]]` in full (should have been step 0) closes the draft side:
- **Layer-skip self-draft: MEASURED-DEAD, do-not-build.** The Whisper DECODER is only **4 layers** (turbo AND
  tiny.en — the "32" is the ENCODER). `FW_DRAFT_ACCEPT_LAYERS` probe: tiny.en k=1/2/3 = **1.7%/1.7%/62.9%**,
  turbo **0.0%/0.0%/10.7%** — all BELOW the ~47-82% break-even ⇒ a NET SLOWDOWN. Skipping k of 4 layers saves
  little and the early hidden state ≠ final argmax. `bd`-rejected (`5118b4a`/`aed4ae1`). Do NOT build a
  layer-skip drafter (my Phase-2/3 K=2/general-K plans below assumed a deep decoder — INVALID).
- **Prompt-lookup / n-gram: MEASURED-DEAD** (0.7-6.4% on real speech, `9d0d07b`/`0ddbd3b`) — ASR output is
  novel token-by-token.
- **⇒ the ONLY viable draft is a real, cheap, ACCURATE draft model, and NONE is on-box** (only turbo +
  tiny.en, different vocabs). So speculative decode is **BLOCKED on owner/infra** (supply a draft model).
- **EV is also modest + mode-gated:** decode is ~15% of e2e in TS mode and **~0 exposed in no_ts**
  (pipeline-hidden), so realized e2e ≈ **4-7%, timestamp-mode ONLY** (ceiling R(8)≈2.9-3.7× × accept ÷
  draft-cost). A SECONDARY lever, not the biggest.

## What IS landable now (the verify side — real, byte-exact, but inert without a draft)
The VERIFY primitives read the weights once for K tokens and are byte-exact — useful IF a draft model is ever
added: `gemv_i8_batch` (LANDED default-on), and **`logits_all` (Phase 1, LANDED `19a71ca`, 2.36× amortization
microbench, byte-exact)**. These are correct + tested but NOT on any production path (no viable draft to feed
them). **Do NOT build the draft/verify LOOP until a real draft model exists** — the loop without a working
draft is dead code.

## Phasing (each phase byte-exact, gated `FW_SPEC_DECODE` default-OFF, verified flag-on-transcript == greedy)

**Phase 1 — verify primitive (`logits_all`).** Add batched logits+argmax for all `tq` positions (today
`logits_last` does only the last). This is the amortization core: the `[K,1280]×[1280,51865]` GEMM reads the
133 MB tied-embedding ONCE for K positions vs K× separately. Test: byte-exact vs per-position `logits_last`.
Micro-bench: batched-K vs K-separate = the weight-read amortization (the measurable win of the whole scheme).

**Phase 2 — K=2 read-only draft (the tractable first complete version, AVOIDS the dual-cache trap).** For
K=2 the draft only needs to propose ONE extra token, so it can be a **read-only** early-exit forward (run
layers 0..k on the last committed token, NO cache append, argmax the partial logits). Then verify forwards
`[last_committed, draft]` batched (tq=2, all layers, `logits_all`) → argmax₀ (= true next tok t0), argmax₁
(= t1 iff draft==t0). Accept: if `draft==t0` emit {t0,t1} advance 2 (2 tokens from 1 verify pass); else emit
{t0} advance 1. **Cache:** verify appends positions for `[last_committed, draft]`; on reject, roll back the
appended position for `draft` (truncate each layer's `KvCache` len by 1 — the existing `KvCache` has a `len`
field; rollback = `len -= 1`, cheap). Read-only draft ⇒ NO dual cache ⇒ far simpler. Ceiling ~1.3–1.5×
decode (2-way amortization − draft overhead). **Verify:** `FW_SPEC_DECODE=1` transcript == default greedy,
byte-identical, on jfk×1/×3/×8 + track01 (5-window) both models both modes.

**Phase 3 — general K (dual cache, for the R(8)≈2.9× ceiling).** K>2 needs an autoregressive draft (K steps)
so the draft advances a SEPARATE k-layer draft cache (reset from the committed context each round; the main
cache stays the full-model one). Verify batches all K, accepts the matching prefix, truncates the main cache
to `past_len + accepted + 1`. This is the intricate part (dual-cache seeding/reset each round) — do it only
after Phase 2 proves the accept-rate + measures the real win, and tune K by the layer-skip accept curve.

**Phase 4 — tune + measure.** Sweep k (draft layers) and K by the accept curve; measure decode-loop span
(track01, decode-bound) flag-on vs off; the win is capped by the per-position logits (verify amortizes it,
but each position still contributes) — quantify the real e2e (~1.5–1.8× projected on realistic long audio).

## Guardrails
- Gated `FW_SPEC_DECODE` DEFAULT-OFF throughout; each phase's merge criterion is transcript BYTE-IDENTICAL to
  greedy (the verify makes it exact by construction; the test catches any cache-index/rollback bug).
- Bench COLD not warm (`project_draft_decoding_amortization`). Decode-bound clip (track01) is the workload;
  jfk is too short (encode-bound) to show the win.
- The draft's logits may be approximate/int8 (output stays exact) — exploit that for draft cheapness.
- Owner call to flip default-ON after Phase 4 (it's a decode-algorithm change; greedy-verify keeps it
  byte-exact, so no WER gate, but it's a behavioral/complexity change worth owner sign-off like FW_ENC_FREE_F32).
