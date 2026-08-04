# PR (ready to merge): enable `FT_SDPA_POLY_EXP` by default for `large-v3-turbo`

**Status:** landed on the campaign branch as a single commit; this packet is the owner's
review/merge record. **One-line behavior change, gated to one model, output proven neutral.**

| | |
|---|---|
| franken_whisper commit | **`94714c1`** (`src/native_engine/{mod.rs,encoder.rs}`, +71/−10) |
| depends on frankentorch | `1fb80836` (`set_sdpa_poly_exp` setter) · `b13eeb36` (poly kernel) · `d336dc58` (ULP budget) — all already on frankentorch `main` |
| tracking | bd-bcm7 (closed) |
| kill-switch | `FW_SDPA_POLY_EXP=0` (disables even on turbo) |
| operator force | `FT_SDPA_POLY_EXP=1` (on for any model) |

## What it does

At encoder load, `Encoder::from_ggml` calls `configure_sdpa_poly_exp(hparams)`:

```rust
pub(crate) fn configure_sdpa_poly_exp(hparams: &WhisperHParams) {
    let killed = std::env::var("FW_SDPA_POLY_EXP").as_deref() == Ok("0");
    let forced = std::env::var("FT_SDPA_POLY_EXP").as_deref() == Ok("1");
    let want = forced || (is_large_v3_turbo(hparams) && !killed);
    ft_kernel_cpu::set_sdpa_poly_exp(want);   // 8-lane wide::f32x8 poly softmax in sdpa_forward_f32
}
```

Set **explicitly per load** (not just "turn on") so a `turbo → tiny.en` sequence in one process
cannot leak the ON state. `is_large_v3_turbo` was extracted from `calibrated_encoder_int8_model`
(DRY) and unit-tested (`is_large_v3_turbo_discriminates_models_for_poly_exp`: turbo→on, tiny.en→off,
unknown→off).

## Why it is safe to flip (the proof)

The poly softmax is **not bit-identical** by construction (a different transcendental + a lane-wise
sum reorder), so it ships behind a task-level gate, not an assert. On `large-v3-turbo` it is
**transcript-neutral**, measured on the shipped binary (`e2e_probe` sha256
`272102fd7cd643bf449eeed18002874cc98241f74290d2937a8d606a10b0c776`):

| check | result |
|---|---|
| **Transcript** (`PROBE_DUMP_TEXT`, jfk ×1/×3/×8, ON vs OFF) | **BYTE-IDENTICAL 3/3** |
| **WER** vs `whisper-cli -nt -t 32` (identical `jfk_x8.wav`) | ON **28.977%** = OFF 28.977% ⇒ **Δ 0.000** |
| **ULP budget** (asserted, frankentorch `d336dc58`) | `exp` ≤ 1 ULP; `O=P@V` vector rel **1.425e-6** |
| **E2E** (paired, order-alternated, 5 reps, 88 s audio) | 8.378 → **7.865 s = 1.0722×** (cv 0.8%, ON wins 5/5) |
| **Op-level** `attn_sdpa` span | **1.2465×** (cv 3.5%, 5/5) |
| vs whisper.cpp (both incl. load) | 9.57 → **9.06 s** vs 13.622 s ⇒ **1.42× → 1.50×** |

**`tiny.en` stays OFF** and is NOT in this PR's scope: it is uncertified — its long-form tail-drop
(`bd-r0qd`) makes a 1.4e-6 perturbation flip greedy argmax (track01 regressed 52.800 → 53.600). The
`is_large_v3_turbo` gate excludes it.

Full evidence: `docs/PROPOSAL_ft_sdpa_poly_exp_default_on.md`.

## Reviewer checklist

- [ ] frankentorch `main` includes `1fb80836` + `b13eeb36` + `d336dc58` (the setter, kernel, budget).
- [ ] Default `large-v3-turbo` transcripts change only to the one proven byte-identical above.
- [ ] `tiny.en` / unknown models unaffected (gate excludes them; unit-tested).
- [ ] Rollback is runtime: `FW_SDPA_POLY_EXP=0`. Code rollback: revert `94714c1` (isolated).

## Reproduce

```sh
export FRANKEN_WHISPER_MODEL_DIR=legacy_whispercpp/whisper.cpp/models
B=<release-perf e2e_probe>
# transcript (PROBE_DUMP_TEXT writes to STDERR):
PROBE_DUMP_TEXT=1 FW_SDPA_POLY_EXP=0 $B large-v3-turbo tests/fixtures/native/jfk.wav 8 2>&1 >/dev/null | grep TRANSCRIPT
PROBE_DUMP_TEXT=1                    $B large-v3-turbo tests/fixtures/native/jfk.wav 8 2>&1 >/dev/null | grep TRANSCRIPT   # default now ON
# build/test remote-only:
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo test -p franken_whisper --lib is_large_v3_turbo_discriminates
```
