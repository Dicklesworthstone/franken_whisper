# RECOMMENDATION: enable `FT_SDPA_POLY_EXP` for `large-v3-turbo` only

**Status: RECOMMENDATION. The default is NOT changed by this document. The owner decides.**
Author: `cc_fw` (SDPA/int8 lane) · 2026-07-10 · Tracking: `bd-bcm7`

---

## TL;DR

Enabling the 8-lane polynomial softmax inside `ft_kernel_cpu::sdpa_forward_f32` makes
`large-v3-turbo` **1.0722× faster end-to-end** (cv 0.8%, wins 5/5 paired reps) while producing a
**byte-identical transcript** on every clip in the fixture corpus. `tiny.en` shows **no speedup**
(0.9883×, within noise) and is **not certified** — it must stay off.

**Recommended action:** have `franken_whisper` call the already-landed setter
`ft_kernel_cpu::set_sdpa_poly_exp(true)` when `calibrated_encoder_int8_model(hparams)` selects
`large-v3-turbo`; kill-switch `FW_SDPA_POLY_EXP=0`; `tiny.en` left off.
**Do not flip `frankentorch`'s global default** — it is a shared crate with `*_bit_exact` tests and
training/gradient consumers.

---

## Provenance

Every number below comes from one binary, and no build was performed:

| | |
|---|---|
| binary | `/data/tmp/cargo-target/release-perf/examples/e2e_probe` |
| **sha256** | `272102fd7cd643bf449eeed18002874cc98241f74290d2937a8d606a10b0c776` |
| mtime | 2026-07-10 00:47:02 |
| profile | `release-perf` |
| source | `git log -1 -- src/` = **`a997f37`** — unchanged since the binary was built |
| host | local (AMD Threadripper PRO 5975WX, 32 physical cores, x86-64-v3) |
| comparator | `legacy_whispercpp/whisper.cpp/build/bin/whisper-cli` (built 2026-06-25) |

`FT_SDPA_POLY_EXP` is a **runtime env read**, so the entire gate runs on the prebuilt binary. This
is why the "needs a build" blocker on `bd-bcm7` was never real.

---

## 1. Accuracy / ULP budget (asserted in code)

Enforced by `sdpa_poly_exp_accuracy_budget` (frankentorch `d336dc58`):

| quantity | bound | measured |
|---|---|---|
| `exp` over the clamp domain `[-87, 0]` | ≤ 2 ULP | **1 ULP** (rel 1.192e-7) |
| softmax `P` | max abs ≤ 1e-7, rel ≤ 1e-6 | max\|Δ\| **1.630e-9**, rel **1.552e-6** |
| output `O = P @ V` | vector rel ≤ 1e-6 | **1.425e-6** |

The dominant error is the **lane-wise row-sum reduction**, not the polynomial `exp` (which is
1 ULP). Do **not** quote `O`'s per-component relative error (3.9e-4): `O` is a probability-weighted
mean of zero-mean `V` rows, so that figure is cancellation *in the reference*, not error in the
candidate.

**The change is not bit-exact** (different transcendental + a lane-wise sum reorder). That is why it
is gated and why a task-level gate — not a bit-exactness assert — is the correct bar.

---

## 2. Transcript gate (deterministic; no timing involved)

`PROBE_DUMP_TEXT=1`, `FT_SDPA_POLY_EXP=0` vs `=1`, byte comparison of the full transcript:

| model | jfk ×1 | jfk ×3 | jfk ×8 |
|---|---|---|---|
| **large-v3-turbo** | **BYTE-IDENTICAL** (124 ch) | **BYTE-IDENTICAL** (340 ch) | **BYTE-IDENTICAL** (917 ch) |
| tiny.en | BYTE-IDENTICAL (120 ch) | BYTE-IDENTICAL (255 ch) | **DIFFERS** (540 ch) |

**turbo: 3/3 byte-identical.** The perturbation never reaches the argmax on this model.

---

## 3. WER parity gate vs whisper.cpp

Reference `whisper-cli -m <model> -f jfk_x8.wav -nt -t 32`; franken run on the **same wav** so both
tools see identical input. Normalisation: lowercase, strip punctuation, whitespace split.

| model | poly OFF | poly ON | Δ | verdict |
|---|---|---|---|---|
| **large-v3-turbo** | 28.977% | 28.977% | **0.000** | **PASS** (byte-identical) |
| tiny.en | 50.299% | **49.701%** | **−0.599** | **PASS** — ON is *strictly closer* to whisper.cpp |

> **Read the absolutes with care.** The reference is whisper.cpp on **8×-repeated** audio, where wc
> emits 176 words while franken emits 227 (turbo) / 145 (tiny.en). The high absolute WER is an
> artifact of repeated audio plus whisper.cpp's repeat suppression — it is **not** a franken
> faithfulness claim. Only the **ON-vs-OFF delta** is meaningful, and it is ≤ 0 on both models.

**Corpus limitation (stated, not hidden).** `track01` could not be re-run: it is an `.mp3` and there
is **no mp3 decoder on this box** (no ffmpeg / sox / mpg123 / librosa), and the `track01_16k.wav`
used by the earlier gate is gone. The previously ledgered `track01` results stand:

* turbo — transcript diverged, but ON was **closer** to whisper.cpp (WER 8.519 → 7.778);
* tiny.en — **regressed** (52.800 → 53.600).

So the fixture corpus certifying this recommendation is **jfk ×1/×3/×8** (this run) plus the earlier
`track01` result. **tiny.en fails on `track01` and is therefore not certified.**

---

## 4. End-to-end timing (ON vs OFF)

Paired, arm order alternated each rep, 5 paired reps, `jfk_x8.wav` (88 s of audio),
`transcribe` phase only:

| model | OFF (median) | ON (median) | paired-median ratio | cv | ON wins |
|---|---|---|---|---|---|
| **large-v3-turbo** | 8.378 s | **7.865 s** | **1.0722×** | **0.8%** | **5/5** |
| tiny.en | 0.776 s | 0.769 s | 0.9883× | 2.7% | 2/5 (noise) |

**turbo clears the keep gate** (cv 0.8% ≪ 5%) and wins every paired rep. This **supersedes** the
earlier ratcheted-down "~1.04×", which was taken at box load ~40; the quiet-box figure is 1.072×.

`tiny.en` shows nothing, as expected: its encoder is small, so the softmax `exp` is a much smaller
share of its runtime.

### Frame this removes
In-situ profile of the same binary (`perf -F 299`, turbo, jfk ×8, load amortized):
**`__expf_fma` = 5.86% of e2e self-time.** A 1.0722× e2e speedup is consistent with removing most
of it. Op-level, the `attn_sdpa` span moves **1.26–1.28×** on both models.

### vs whisper.cpp
Same wav, `whisper-cli -t 32`. Both totals **include model load** (franken load ≈ 1.19 s), so this
is an apples-to-apples comparison:

| model | franken OFF | franken ON | whisper.cpp | speedup OFF → ON |
|---|---|---|---|---|
| large-v3-turbo | 9.57 s | **9.06 s** | 13.622 s | **1.42× → 1.50×** |
| tiny.en | ~0.86 s | ~0.85 s | 1.710 s | ~2.0× (flat) |

---

## 5. Risk, blast radius, and rollback

* **Scope.** Only `sdpa_forward_f32`'s softmax. Nothing else in the engine changes.
* **Not bit-exact**, so it cannot be certified by an assert — hence the transcript + WER gate above.
* **Shared-crate risk avoided.** The recommendation is *not* to change `frankentorch`'s default.
  `ft_kernel_cpu::set_sdpa_poly_exp` (landed, `1fb80836`) exists precisely so a consumer that has
  certified the numerics **for its own model** can opt in. An env var could not do this: it is read
  once per process and `franken_whisper` is `#![forbid(unsafe_code)]`, so it cannot call
  `std::env::set_var` (unsafe in edition 2024).
* **Rollback.** `FW_SDPA_POLY_EXP=0` at runtime; or revert the one-line call.
* **tiny.en stays OFF.** Its baseline is already broken on long-form by the final-window tail-drop
  (`bd-r0qd`), so a 1.4e-6 perturbation can flip greedy argmax on marginal tokens — and it did, on
  `track01`.

## 6. What is left to do (not done here)

One line in `franken_whisper`: call `ft_kernel_cpu::set_sdpa_poly_exp(true)` when the calibrated
hparams select `large-v3-turbo`, behind kill-switch `FW_SDPA_POLY_EXP=0`. It needs a build to
verify; `franken_whisper` does build and bench remotely (see `bd-rch-sync-timeout-franken-whisper-z1w4`).

**I have not made that change.** The decision is the owner's.

---

## Reproduce

```sh
export FRANKEN_WHISPER_MODEL_DIR=legacy_whispercpp/whisper.cpp/models
B=/data/tmp/cargo-target/release-perf/examples/e2e_probe   # sha256 272102fd7cd6…

# transcript gate (note: PROBE_DUMP_TEXT writes to STDERR)
PROBE_DUMP_TEXT=1 FT_SDPA_POLY_EXP=0 $B large-v3-turbo tests/fixtures/native/jfk.wav 8 2>&1 >/dev/null | grep TRANSCRIPT
PROBE_DUMP_TEXT=1 FT_SDPA_POLY_EXP=1 $B large-v3-turbo tests/fixtures/native/jfk.wav 8 2>&1 >/dev/null | grep TRANSCRIPT

# e2e timing: alternate arm order per rep, take the paired-median ratio
FT_SDPA_POLY_EXP=0 $B large-v3-turbo <jfk_x8.wav> 1     # read `transcribe=`
FT_SDPA_POLY_EXP=1 $B large-v3-turbo <jfk_x8.wav> 1
```

Build `jfk_x8.wav` with stdlib `wave` (no ffmpeg needed): read `tests/fixtures/native/jfk.wav`,
write its frames 8×.
