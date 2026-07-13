# Speculative Decode — implementation plan (the sole remaining perf lever)

> Actionable build plan for `bd-wzgh`. Rationale, viability data, and the "why this is the only
> lever left" are in `NEGATIVE_EVIDENCE.md` (CONSOLIDATED FRONTIER MAP + OWNER DECISION BRIEF).
> This file is the "how to build it." Author: GoldenOwl 2026-07-13.

## Why (one line)
Realistic long audio is **decode-dominated (62.5% of e2e)** and DRAM-weight-streaming-bound
(~342 MB/token re-streamed). Batch-verifying K drafted tokens in ONE all-layer pass amortizes that
stream K×. **Greedy-verify is BYTE-EXACT vs greedy** (accept only tokens the full model's argmax
agrees with), so this ships to the default quality path — no WER gate.

## ✅ UPDATE 2026-07-13 (later session) — DRAFT UNBLOCKED (real model provisioned) + EV was UNDER-stated
Two claims in the ⚠ CORRECTION below are now SUPERSEDED by measurement (`NEGATIVE_EVIDENCE` ticks 13j–13l):

1. **The DRAFT blocker is RESOLVED.** `distil-large-v3` IS a real, cheap, accurate, VOCAB-COMPATIBLE draft and
   is now on-box (`fetch_test_models.sh --model distil-large-v3`, sha `2883a11b…`). Header-verified: n_vocab
   **51866 == turbo**, the **SAME 32-layer/n_state-1280 large-v3 encoder** (⇒ encoder computed ONCE, shared by
   draft+verify — no dual encode), and n_text_layer **2 (vs turbo's 4)**. Measured: distil decode **1.64×
   faster** than turbo (132 vs 217 ms/27 tok). So the draft is a SEPARATE cheap MODEL, not layer-skip.
   **⚠ BUT the accept rate is TOO LOW to win (MEASURED, `examples/spec_accept_probe.rs`, tick 13m):** teacher-
   forced K=1 accept = jfk 73.1%, jfk×3 81.2%, **track01 real conversational speech 54.5%**. At 54.5% a K=1 pass
   yields E[tok]=1.545 for cost ≈ draft(0.61×)+verify(~1.2×)=1.81 turbo-equiv ⇒ **~0.85× = NET SLOWDOWN**
   (general-K worse — rejected drafts waste K draft-decodes). distil's 2-layer decoder disagrees with turbo's
   4-layer too often on ambiguous speech (the earlier "89.5%" was WORD-LCS, far laxer than token-accept). **⇒
   spec-decode with the on-box distil draft does NOT win**; it needs a draft that is BOTH much cheaper AND ≥~85%
   token-agreeing — none is on-box. The `logits_all`/`gemv_i8_batch` verify primitives + the probe stay for a
   future better draft.

2. **EV was UNDER-stated: decode is NOT pipeline-hidden in no_ts.** The "~0 exposed in no_ts ⇒ 4-7% TS-only"
   claim was tiny.en-scoped/wrong for turbo. MEASURED (tick 13j, turbo/track01 no_ts): decode_loop is **64% of
   e2e** — pipelining hides the SMALLER phase, which on realistic long audio is the ENCODER (315 ms/win), NOT
   the decode (1014 ms/win). So spec-decode attacks a **64%-exposed** cost in the DEFAULT no_ts path, and being
   greedy-verify BYTE-EXACT it ships to the quality path. Realized e2e is a large fraction of that 64× accept —
   **the BIGGEST remaining lever, not a secondary one.**

**Architecture consequence:** with a real draft MODEL (own 2-layer decoder + own KV cache), the tractable build
is the **general-K dual-cache** shape (Phase 3), NOT the read-only layer-skip Phase 2. K=1 first increment:
distil autoregressively drafts 1 token from the committed prefix (its own cache), turbo verifies `[committed,
draft]` batched (`logits_all`, all 4 turbo layers) → accept iff turbo's argmax₀ == draft; on reject truncate the
draft's cache by 1. Gate `FW_SPEC_DECODE` default-OFF; merge criterion = transcript BYTE-IDENTICAL to greedy.
**Still the owner-ticketed multi-turn build (bd-wzgh) — NOT built autonomously — but no longer blocked.** The
next concrete measurement (before the full loop) is the TRUE teacher-forced accept rate (feed turbo's committed
tokens to distil, count distil-argmax == turbo-next); the 89.5% independent-transcript figure is a lower bound.

## ⚠ CORRECTION 2026-07-13 — the DRAFT side is BLOCKED (both cheap drafts measured DEAD); this feature is owner/infra-gated
> SUPERSEDED by the ✅ UPDATE above — a real draft model is now on-box; the layer-skip/prompt-lookup deadness
> stands but is moot. Kept for the record.

Reading `[[project_draft_decoding_amortization]]` in full (should have been step 0) closes the draft side:
- **Layer-skip self-draft: MEASURED-DEAD, do-not-build.** The Whisper DECODER is only **4 layers** (turbo AND
  tiny.en — the "32" is the ENCODER). `FW_DRAFT_ACCEPT_LAYERS` probe: tiny.en k=1/2/3 = **1.7%/1.7%/62.9%**,
  turbo **0.0%/0.0%/10.7%** — all BELOW the ~47-82% break-even ⇒ a NET SLOWDOWN. Skipping k of 4 layers saves
  little and the early hidden state ≠ final argmax. `bd`-rejected (`5118b4a`/`aed4ae1`). Do NOT build a
  layer-skip drafter (my Phase-2/3 K=2/general-K plans below assumed a deep decoder — INVALID).
- **Prompt-lookup / n-gram: MEASURED-DEAD** (0.7-6.4% on real speech, `9d0d07b`/`0ddbd3b`) — ASR output is
  novel token-by-token.
- **⇒ the ONLY viable draft is a real, cheap, ACCURATE draft model** — now RESOLVED (distil-large-v3, ✅ above).
- **EV** — SUPERSEDED: decode is 64%-exposed in no_ts (✅ above), not 4-7% TS-only.

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
