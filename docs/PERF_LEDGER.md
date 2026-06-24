# franken_whisper — Performance Lever Ledger

> Head-to-head, MEASURED optimization log for the native Rust engine. Owned by
> swarm agent **BlackThrush** (franken_whisper-cc). Every entry records a real
> criterion measurement; ~0-gain or regressing levers are REVERTED, not kept.

## Measurement protocol

- **Harness:** `benches/native_engine_bench.rs` (criterion).
- **Build/run:** per-crate via `rch exec` with
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cc` (never
  workspace-wide).
- **Baseline vs candidate:** criterion `--save-baseline` / `--baseline`. rch
  repo-convergence keeps a repo on one worker, so save/compare runs land on the
  same hardware; large effects (≫ worker variance) are the bar for keeping a
  lever.
- **Conformance gate:** every numeric kernel change ships with a **bit-exact
  parity test** against the pre-change reference, so a "win" cannot silently
  alter output. The mel output is conformance-checked against whisper.cpp's exact
  encoder input.
- **What the original is:** whisper.cpp's exact algorithms (this engine is a
  faithful Rust port). A kernel lever's "gain" is the measured speedup of the
  Rust port over its own faithful-port baseline while preserving bit-exact output
  — i.e. doing whisper.cpp's identical math, faster.

## Hermetic vs model-gated benches

| bench | hermetic? | status |
|---|---|---|
| `native_engine/mel/mel_30s` | yes | **measured** |
| `native_engine/f16_gemv/*` | yes | available |
| `encoder_window_{tiny,large}` | no (model+jfk.wav) | tiny unlocked locally; large needs `large-v3-turbo` |
| `decoder_token_step_{tiny,large}` | no | tiny unlocked locally |
| `logits_gemv_large` | no (large model) | blocked: model absent |
| `e2e_tiny_jfk` | no (model) | tiny unlocked locally |

> `tests/fixtures/native/jfk.wav` is gitignored; copied locally from
> `legacy_whispercpp/whisper.cpp/samples/jfk.wav` (mono 16 kHz, 11 s) to unlock
> the model-gated benches for measurement. The `large-v3-turbo` model is not
> present, so the large-shape levers remain blocked (concrete blocker).

---

## Levers

### L1 — log-mel FFT twiddle precompute (bit-exact)  — `src/native_engine/mel.rs`

**Hypothesis.** whisper.cpp's recursive `fft` recomputes `cos`/`sin` twiddles
per butterfly per frame, and the odd-`N` base case `dft(25)` (reached 16×/frame,
3000 frames) recomputes ~1250 f64 transcendentals per call — ~60 M `sin`/`cos`
per 30 s of audio. These are pure functions of `(k, j, n)` and can be precomputed
into f32 tables once, evaluated bit-for-bit identically thereafter.

**Change.** Precompute level twiddles `[400,200,100,50]` and the `n=25` DFT
`25×25` cos/sin table once (cached `OnceLock`, shared read-only across mel worker
threads); thread them through `fft`/`dft`. Arithmetic and accumulation order
unchanged → bit-exact.

**Conformance.** New test `fft_twiddle_table_is_bit_exact_vs_inline_reference`
asserts byte-for-byte `Vec<f32>` equality vs an inline-transcendental copy of the
original recursion across 10 transform widths × 64 random seeds.

**Measurement (worker vmi1149989, criterion; baseline + candidate on the SAME
worker via rch repo-convergence → valid A/B; baseline = pre-edit code):**

| bench | baseline (mel-pre) | candidate | change | speedup |
|---|---|---|---|---|
| `mel_30s` | 269.06 ms | 29.23 ms | **−89.1%** (p=0.00) | **≈9.2×** |

**Verdict: KEEP.** A 9.2× reduction on the always-on log-mel frontend, far above
any plausible worker variance, with **byte-identical output** (parity test green
— see below). The transcendental-elimination hypothesis is confirmed: the
`dft(25)` base case was the dominant cost.

**Honesty note — what "vs the original" means here.** This ratio is
franken_whisper's mel frontend vs **its own prior faithful-port baseline**, not a
direct timing of OpenAI Whisper's Python mel or whisper.cpp's C mel. The output
remains bit-exact to whisper.cpp's mel (the port's contract), so this is "do
whisper.cpp's identical math, 9.2× faster." A true head-to-head wall-clock vs the
C++/Python originals needs the original-vs-franken harness (bd-zk43 / bd-0hnz);
the large-shape kernels also need the `large-v3-turbo` model staged (bd-ms0x).

**Conformance gate:** `cargo test -p franken_whisper --lib native_engine::mel`
(incl. `fft_twiddle_table_is_bit_exact_vs_inline_reference`) + clippy
`-D warnings` — results recorded at commit time.

### L2 — log-mel FFT per-call allocation elimination (deferred)  — bd-02do

The recursive `fft` still `vec!`-allocates even/odd split + child-output buffers
at every recursion node (~60 allocs/frame × 3000 frames). Secondary to the
twiddle win (allocator churn ≈ single-digit ms vs the ~240 ms transcendental
cost just removed). Tracked in bd-02do as a follow-up via per-thread scratch
buffers. Not yet measured.
