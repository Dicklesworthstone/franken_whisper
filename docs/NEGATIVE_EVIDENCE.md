# Negative Evidence Ledger

This ledger records blocked, neutral, rejected, or non-comparable performance
evidence. It exists to prevent stale optimism from being reused as proof.

## 2026-06-26 - AGENT_NAME=IcyWren: log-mel stack FFT buffers REJECTED - heap Vec is not the bottleneck

### Land-or-dig scan

Repo-local `.scratch` and `.worktrees` directories are absent. Sibling worktrees
were ancestry-checked against current `origin/main` (`b1eb23b` at scan time).
The apparent detached-code wins were stale copies already present on `main`
after rebases:

- `franken_whisper-cod-b-fft-clean-daa0cf9` held the pre-rebase copy of the
  log-mel SIMD projection fusion, already landed on `main` as `b1eb23b`.
- `franken_whisper-cod-a-l14-validate` showed dirty Rayon-pool-cap code only
  because the worktree head was old; current `main` already contains
  `ensure_default_rayon_pool`, the L14 PERF ledger, and its OpenAI-ratio entry.
- `franken_whisper-cod-a-main-measure` was a stale docs-only OpenAI ratio entry
  already present in the current ledger.

With no missing measured win left to land, this pass dug one new preprocessing
lever.

### Candidate

New one-lever probe: in `src/native_engine/mel.rs::compute_8_columns`, replace
the per-8-frame heap allocations

```text
vec![FrameLanes::splat(0.0); N_FFT]
vec![FrameLanes::splat(0.0); 2 * N_FFT]
```

with fixed stack arrays. This targeted allocation/setup overhead after the
previous projection-fusion keep removed the larger scalar-spectrum transpose.
Arithmetic order, twiddle lookup, SIMD FFT, sparse projection, and output layout
were unchanged. Existing bit-exact coverage
`compute_8_columns_matches_scalar_columns_bit_exact` guards the path.

### Requested bench path status

The exact warm target dir remains blocked without destructive cleanup:

```text
AGENT_NAME=IcyWren
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b
rch exec -- cargo bench --profile release -p franken_whisper \
  --bench native_engine_bench -- mel_30s_realistic \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

Result: RCH fell back local and Cargo failed before benchmark execution with
`E0514` stale build-script artifacts compiled by rustc `beae78130` while the
current toolchain is `f20a92ec0`. The target dir was not cleaned because that
would be destructive. As in the prior keep, the comparable A/B used the
non-destructive sibling target dir
`/data/projects/.rch-targets/franken_whisper-cod-b-rustc-f20`.

### Measurements

Comparable same-host local fallback through `rch exec`, same worktree, same
generated lock state, same sibling target dir:

```text
AGENT_NAME=IcyWren
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b-rustc-f20
rch exec -- cargo bench --profile release -p franken_whisper \
  --bench native_engine_bench -- mel_30s_realistic \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

| Workload | Baseline median | Candidate median | Candidate/current ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `native_engine/mel/mel_30s_realistic` | 4.7133 ms | 4.7523 ms | 0.9918x | REJECT |

The candidate was also observed on remote worker `ovh-a` at `2.9013 ms`, but the
same-worker baseline could not be acquired immediately (`RCH_REQUIRE_REMOTE=1
RCH_WORKER=ovh-a RCH_WORKERS=ovh-a` reported no admissible worker), so that
remote number is routing evidence only and not used for the keep/reject call.

Fresh OpenAI Whisper preprocessing comparator used the exact Criterion fixture
audio (`synthetic_audio(N_SAMPLES_30S, 0xa11ce)` ported to Python), 15 warmups,
then 25 timed `whisper.audio.log_mel_spectrogram(samples, n_mels=80)` calls
with the first 5 timed calls discarded:

```text
OpenAI shape:         (80, 3000)
OpenAI steady median: 4.379693069 ms
OpenAI steady mean:   4.408710939 ms
baseline/OpenAI speed ratio:  4.379693069 / 4.713300000 = 0.9292x
candidate/OpenAI speed ratio: 4.379693069 / 4.752300000 = 0.9216x
```

### Decision

Rejected. Stack-resident FFT lane buffers do not beat the committed heap-buffer
path on the comparable run, and they slightly worsen the same OpenAI Whisper
preprocessing ratio. Source was manually restored before commit; no code is
retained.

## 2026-06-25 - AGENT_NAME=IcyWren chunk_frames row-copy KEEP, but still behind OpenAI view/copy

### Land-or-dig scan

Sibling worktree heads were checked with `git worktree list --porcelain` and
ancestor-tested against `main` `bfd3abf`. All measured-code worktrees were
already ancestors of `main`. The only non-ancestor was
`franken_whisper-cod-a-main-measure`; relative to current `main` it contains only
a stale `docs/NEGATIVE_EVIDENCE.md` OpenAI-ratio entry, while its underlying
mel-twiddle source win is already landed.

### Candidate

Preprocessing lane, avoiding the active `gemv_f16` work: `chunk_frames`, the
Rust equivalent of slicing one `[n_mels, n_frames]` Whisper encoder window from
the full row-major mel spectrogram. The old path prefilled the full destination
with `SILENCE_FLOOR`, then copied every in-range frame scalar-by-scalar. The
landed path computes the in-range row prefix once, appends each row with
`extend_from_slice`, and writes `SILENCE_FLOOR` only for the true tail padding.

Decision contract:

```text
keep: Criterion proves a current-franken `chunk_frames_80x3000_mid` win and
      `chunk_frames` matches the old scalar reference on offsets/tails.
reject: Criterion reports no significant current-franken improvement, or
        conformance/reference tests fail.
fallback: restore the scalar prefill-and-copy loop.
```

### RCH bench command status

Requested command shape:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --release -p franken_whisper \
    --bench native_engine_bench -- chunk_frames_80x3000_mid \
    --sample-size 20 --warm-up-time 0.1 --measurement-time 3 \
    --save-baseline chunk-pre
```

RCH selected worker `vmi1264463`, but remote sync timed out after 30 s and RCH
failed open locally. Cargo then rejected `bench --release` with `unexpected
argument '--release'`.

Supported release-profile scalar baseline:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --profile release -p franken_whisper \
    --bench native_engine_bench -- chunk_frames_80x3000_mid \
    --sample-size 20 --warm-up-time 0.1 --measurement-time 3 \
    --save-baseline chunk-pre
```

RCH had no admissible workers (`insufficient_slots=4,hard_preflight=1`) and
failed open locally. Result:

```text
native_engine/mel/chunk_frames_80x3000_mid
time: [150.92 us 156.47 us 163.44 us]
```

Supported release-profile candidate comparison:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --profile release -p franken_whisper \
    --bench native_engine_bench -- chunk_frames_80x3000_mid \
    --sample-size 20 --warm-up-time 0.1 --measurement-time 3 \
    --baseline chunk-pre
```

RCH again had no admissible workers and failed open locally. Result:

```text
native_engine/mel/chunk_frames_80x3000_mid
time:   [21.292 us 21.841 us 22.414 us]
change: [-86.742% -86.171% -85.625%] (p = 0.00 < 0.05)
Performance has improved.
```

Ratio vs scalar franken baseline: `156.47 / 21.841 = 7.164x`.

### OpenAI Whisper preprocessing comparator

Comparator command:

```text
uvx --from openai-whisper python
```

Timed `whisper.audio.log_mel_spectrogram(samples, n_mels=80, padding=N_SAMPLES)`
on the same deterministic 30 s, 16 kHz synthetic sine+dither shape, then sliced
`mel[:, 512:3512]`. OpenAI Whisper version was `20250625`; tensor shape was
`(80, 6000)`, stride `(6000, 1)`, contiguous `True`.

```text
view_runs_s:
  0.0000018580, 0.0000019509, 0.0000017321, 0.0000017246,
  0.0000017346, 0.0000019008, 0.0000016971, 0.0000016418,
  0.0000016177, 0.0000016381
view_median: 0.0000017283s

contiguous_runs_s:
  0.0000161745, 0.0000145278, 0.0000162915, 0.0000175538,
  0.0000149650, 0.0000178667, 0.0000143769, 0.0000148083,
  0.0000142467, 0.0000156478
contiguous_median: 0.0000153064s
```

Ratio convention: `OpenAI median / franken median`.

```text
franken row-copy candidate vs OpenAI strided view:   0.079131x
franken row-copy candidate vs OpenAI compact copy:   0.700810x
```

Caveat: OpenAI's actual slice can remain a strided PyTorch view, while the Rust
encoder owns a compact `Mel` buffer. The compact-copy comparator is therefore the
closer data-movement comparison, but the view timing is the honest OpenAI API
floor.

### Decision

Keep and land. This is a real measured Rust preprocessing-kernel win
(`7.164x`, Criterion `p=0.00`) and preserves the old scalar loop's output on
offset/tail cases. It does not close the OpenAI head-to-head for this operation:
franken remains slower than OpenAI's compact copy (`0.700810x`) and much slower
than OpenAI's strided view floor (`0.079131x`). Recorded here so the win is not
misrepresented as an OpenAI-beating slice path.

## 2026-06-25 - OWNER DECISION: native engine declared at its SAFE CEILING — optimization loop CLOSED

After a full kernel + load + timestamp-path audit (entries below), the engine has
no remaining in-policy speed lever. The owner was presented the only outstanding
lever — mmap model-load, which needs `#![forbid(unsafe_code)]` relaxed
(`memmap2::Mmap::map` is an `unsafe fn`) for a modest, mostly-warm gain (~7 ms tiny
/ ~150 ms warm-large; cold-load is disk-bound ≈0) — and **chose to KEEP
`#![forbid(unsafe_code)]`** and accept the safe ceiling.

**Ratified status (do not re-litigate):**
- Every hot kernel is at the AVX2 hardware ceiling on the Zen3 bench fleet and is
  brittle to source restructuring (mel/FFT/filterbank L1–L4, layer_norm L5,
  gelu/softmax L8, `gemv_f16`/`dot8`/`gemv_f16_batch` L9–L13 + two ceiling rejects
  this session, `conv1d` = im2col+sgemm). Decoder is algorithmically complete
  (KV-cache + precomputed cross-K/V). DTW post-proc is not the ts bottleneck.
- vs **OpenAI-Whisper**: franken wins **2.13–3.26×** everywhere. vs whisper.cpp:
  parity/win except tiny-cold-CLI (~190 ms, diffuse load+startup).
- The mmap load lever is **declined by owner policy**; do NOT re-attempt it,
  re-run the audit, or re-probe `dot8`/`gemv_f16` restructures (they regress).
- **⇒ The per-kernel optimization loop is closed.** Reopen only if the owner
  relaxes `forbid(unsafe_code)`, provides an AVX-512 (Zen4+) bench host + signs off
  on an `x86-64-v4` baseline, or stages the ggml models on the rch fleet.

## 2026-06-25 - BlackThrush: "safe streaming load" gives ~0 speed (parse is already zero-copy) + DTW post-proc not the bottleneck

Two last-mile levers checked and closed so no future turn re-spends on them.

**Safe streaming model load = ~0 speed (only memory).** `ggml.rs::parse` does NOT
copy the big tensors — for each it records `TensorEntry { byte_offset, byte_len }`,
`cur.skip(byte_len)`, and **keeps the whole `blob: Vec<u8>` in the returned
struct**; weights are later read as zero-copy *slices* into that blob. So
`fs::read`'s single pagecache→`blob` copy is the *only* copy at load. A **safe**
`BufReader` streaming refactor (no `unsafe`, no dep) would merely move that copy
per-tensor — **same total bytes copied, ~0 wall-clock change**, only lower peak
RSS (no whole-blob + typed buffers coexisting). The *speed* win (eliminate the
pagecache→blob copy: ~7 ms tiny / ~150 ms warm-large) requires **mmap**, whose
`memmap2::Mmap::map` is an `unsafe fn` → blocked by `#![forbid(unsafe_code)]`
(owner policy). ⇒ no safe load-speed lever exists; the module's "streaming loader"
doc-comment is aspirational (the code holds the full blob resident).

**DTW post-processing is not the ts-path bottleneck.** `dtw_path` is a scalar DP
over only `n_tokens × n_frames ≈ 28 × 1500 ≈ 42 k` cells (µs); `median_filter` /
`token_timestamps` are per-head/token over 1500 frames (small). The realistic
ts-path cost (504 ms) is dominated by the cross-attention **recording**, already
parallelized in L13. No DTW lever.

⇒ Combined with the prior entries, **every hot kernel AND the load/timestamp paths
are now audited at-ceiling or owner-policy-blocked.** The single remaining
speed lever in the whole engine — mmap load — needs `forbid(unsafe_code)` relaxed.

## 2026-06-25 - AGENT_NAME=IcyWren log-mel SIMD scratch reuse zero-gain rejected

### Land-or-dig scan

Repo-local `.scratch` and `.worktrees` directories were absent. Sibling worktree
heads were checked with `git worktree list --porcelain` and ancestor-tested
against current `main`. All measured-code worktrees were already ancestors of
`main`; the only non-ancestor was `franken_whisper-cod-a-main-measure`, a stale
docs-only OpenAI-ratio branch.

### Candidate

Preprocessing lane, avoiding the active `gemv_f16` work: `log_mel` /
frame-batched STFT. One-lever probe: reuse the top-level `compute_8_columns`
SIMD FFT input/output/transpose scratch buffers once per worker instead of
allocating those buffers for every 8-frame batch. Arithmetic order, FFT
twiddles, power projection, and output layout were unchanged.

Decision contract:

```text
keep: Criterion proves a current-franken `mel_30s_realistic` win and the
      OpenAI Whisper preprocessing ratio improves.
reject: Criterion reports no significant current-franken improvement or any
        comparator ratio regression.
fallback: keep the committed per-batch scratch allocation path.
```

### RCH bench command status

Requested command shape:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --release -p franken_whisper \
    --bench native_engine_bench -- mel_30s_realistic \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

Baseline attempt selected worker `vmi1227854`, but remote sync timed out after
30 s, then RCH failed open locally. Cargo then rejected `bench --release` with
`unexpected argument '--release'`. The candidate exact-form attempt later had no
admissible RCH workers and failed locally with the same Cargo argument error.

Supported release-profile baseline:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --profile release -p franken_whisper \
    --bench native_engine_bench -- mel_30s_realistic \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3 \
    --save-baseline logmel-pre
```

RCH selected worker `vmi1227854`, remote sync timed out, then failed open
locally. Result:

```text
native_engine/mel/mel_30s_realistic
time: [5.4693 ms 5.5973 ms 5.8447 ms]
```

Supported release-profile candidate comparison:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --profile release -p franken_whisper \
    --bench native_engine_bench -- mel_30s_realistic \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3 \
    --baseline logmel-pre
```

RCH selected worker `hz2`, remote sync timed out, then failed open locally.
Result:

```text
native_engine/mel/mel_30s_realistic
time:   [5.2833 ms 5.3899 ms 5.5499 ms]
change: [-9.8590% -5.0179% +0.6976%] (p = 0.11 > 0.05)
No change in performance detected.
```

### OpenAI Whisper preprocessing comparator

Comparator command:

```text
uvx --from openai-whisper python
```

Timed `whisper.audio.log_mel_spectrogram(samples, n_mels=80)` on the same
deterministic 30 s, 16 kHz synthetic sine+dither shape with
`torch.set_num_threads(8)`. Result:

```text
OpenAI Whisper runs_s:
  0.0058428091, 0.0052949998, 0.0055833368, 0.0055867631,
  0.0054258581, 0.0060877942, 0.0056735280, 0.0052744520,
  0.0052607148, 0.0054204389
OpenAI median: 0.005504597s

current franken Criterion midpoint:   0.005597300s
candidate Criterion midpoint:         0.005389900s
current vs OpenAI preprocessing:      0.983439x
candidate vs OpenAI preprocessing:    1.021280x
candidate vs current franken:         1.038479x midpoint only
```

Caveat: OpenAI returns `[80, 3000]`; franken's `log_mel` frontend computes the
Whisper-style 30 s audio plus the padded tail (`mel_30s_realistic` exercises the
same 3000 real FFT frames plus silence-tail handling). This is still useful as
a preprocessing head-to-head, but not proof that a 2% midpoint difference is
stable.

### Decision

Rejected as zero-gain. The OpenAI ratio was slightly favorable, but the
same-Criterion current-franken comparison did not prove a win (`p=0.11`, CI
crossed zero). Code was manually reverted; no source change was retained.

## 2026-06-25 - BlackThrush: BLOCKER — no benchable compute lever remains (AVX2 hardware ceiling + decoder algorithmically complete + models absent on rch fleet)

Closing the loop on three things probed this session, with hardware/code proof so
future turns don't re-spend on them:

**1. The "wider-SIMD / AVX-512" lever (left open in the `R-quad-dot8` entry) is
moot on the bench fleet — and there is no f32-SIMD headroom to take.** The rch
worker (`ovh-a`) is an **AMD Threadripper PRO 5975WX (Zen 3)**: `/proc/cpuinfo`
flags are `avx2 f16c fma` — **no `avx512*`**. So the committed 8-lane (`f32x8`,
256-bit) `dot8` already uses the **widest f32 SIMD this hardware has**; an
`x86-64-v4` build baseline would no-op (or `SIGILL`) here and cannot be measured.
The two rejected kernel attempts this session (`R-blocked-dequant` +1.1–2.1×,
`R-quad-dot8` +1.2–2.5×, both REGRESSIONS) confirm `gemv_f16`/`dot8` sits at the
**AVX2 hardware ceiling** (1280×1280 ≈ 184 µs ≈ 18 GB/s of f16 reads — compute-,
not bandwidth-bound) and is **brittle to any source restructuring** (hand-rolled
indexing defeats the `chunks_exact(8)`/`0..8` autovectorization idiom).

**2. The decoder is algorithmically complete — no recompute lever.** `decoder.rs`
runs an **incremental per-layer self-attention KV cache** (`KvCache`, one per
layer, causal append) and **precomputes cross-attention K/V once per window**;
`gemv_f16_batch` **dequants each weight row once and reuses it across all `tq`
prompt tokens** (convert hoisted out of the token loop). The classic whisper
decode wins (KV cache, cross-K/V caching, dequant amortization) are all present.

**3. The e2e-relevant benches cannot run on the rch fleet (measurement-infra
gap).** `benches/native_engine_bench.rs`'s `encoder_window_*`, `decoder_token_step_*`,
`logits_gemv_large`, `e2e_*` are model-gated; the worker prints
`SKIP … model {tiny.en|large-v3-turbo} missing`. The models exist **locally**
(`legacy_whispercpp/whisper.cpp/models/ggml-large-v3-turbo.bin`, jfk fixture) but
are gitignored / not synced to rch workers. So **only the hermetic kernel benches
(`mel`, `f16_gemv`, `layer_norm`) are measurable via `rch exec`, and all are
already optimized (L1–L5) or at ceiling (f16_gemv).**

**⇒ BLOCKER.** No measurable compute lever remains on the prescribed
`rch exec -- cargo bench` path. The only outstanding e2e gap is **tiny-cold-CLI vs
whisper.cpp (~190 ms, diffuse: ~80 load / ~84 startup+audio / ~26 transcribe)** —
not a kernel, dominated by eager model load (`ggml.rs` does `std::fs::read` —
whole-blob copy — then `parse(Vec<u8>)` copies every tensor out via a `Cursor`;
whisper.cpp `mmap`s and faults weights in lazily) and CLI startup. **vs
OpenAI-Whisper franken already wins 2.13–3.26× everywhere.** Concrete unblock
paths for a future turn, each needing a decision the kernel work can't make
unilaterally: (a) stage the ggml models + jfk.wav on the rch workers so the
model-gated/e2e benches run there; (b) owner sign-off on an `x86-64-v4` baseline
*and* a Zen4+/AVX-512 bench host to even test it; (c) an mmap cold-load path —
but this is **genuinely BLOCKED by `#![forbid(unsafe_code)]`**: `memmap2::Mmap::map`
is an `unsafe fn` (its contract is "the file must not be mutated while mapped"),
so the call site needs `unsafe`. (CORRECTION of an earlier line in this very entry
that called memmap2 a "safe-API dep" — that was wrong; the prior-session entry
near the bottom of this file had it right. Unblocking mmap needs lifting
`forbid(unsafe_code)` or an isolated helper crate — an owner policy call, not a
kernel lever.)

**Kernel audit completeness (this turn).** Confirmed the last un-audited hot
kernel, the encoder `conv1d` stem (`nn::conv1d`), is **already im2col + ft sgemm**
(parallel im2col gather, weight transpose is 0.07% of the conv matmul) — textbook
optimal, no lever. With mel/FFT/filterbank (L1–L4), layer_norm (L5), gelu/softmax
(L8), gemv_f16/dot8/gemv_f16_batch (L9–L13 + this session's two ceiling rejects),
KV-cache + cross-K/V (decoder), and now conv1d all verified optimized-or-at-ceiling,
**every hot kernel in the engine has been audited.** No benchable compute lever
remains.
## 2026-06-26 - AGENT_NAME=IcyWren: log-mel SIMD projection fusion KEEP — Rust preprocessing now edges steady OpenAI Whisper

Targeted the preprocessing lane while decoder GEMV work was occupied elsewhere:
`src/native_engine/mel.rs` already had cached Hann/twiddles, sparse mel
projection, and frame-batched `std::simd` FFT, but each 8-frame SIMD FFT batch
still transposed the full complex spectrum back into eight scalar buffers before
calling the scalar power+projection helper. The kept one-lever change fuses
SIMD power-spectrum calculation with sparse mel projection, while preserving each
lane's f64 accumulation order over filter-bin weights. The scalar transpose is
gone; arithmetic semantics are guarded by a new bit-exact test:
`native_engine::mel::tests::compute_8_columns_matches_scalar_columns_bit_exact`.

### Benchmark path caveats

The exact requested command form is not runnable in this repo:
`cargo bench --release` is rejected by Cargo (`unexpected argument '--release'`);
the supported equivalent used here is `cargo bench --profile release`. The exact
requested warm target dir, `/data/projects/.rch-targets/franken_whisper-cod-b`,
is also poisoned by old nightly build-script artifacts (`E0514`, compiled by
rustc `beae78130` but current toolchain is `f20a92ec0`). Per the no-destructive
rule, it was not cleaned. Measurements used non-destructive sibling target dir
`/data/projects/.rch-targets/franken_whisper-cod-b-rustc-f20`; current
`origin/main` also required a local `Cargo.lock` refresh because path crates
(`fsqlite`, `ftui`) have advanced, but that lockfile is not part of this commit.

### Measurements

Criterion, per-crate only:

```text
AGENT_NAME=IcyWren
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b-rustc-f20
rch exec -- cargo bench --profile release -p franken_whisper \
  --bench native_engine_bench -- mel_30s_realistic \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

Same disposable worktree and generated lock state, current `origin/main`
`2ee9485` baseline vs candidate:

| Workload | Baseline median | Candidate median | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `native_engine/mel/mel_30s_realistic` | 5.2211 ms | 4.7739 ms | **1.0937x** | KEEP |

Criterion reported `[-10.260% -9.2345% -8.0507%]`, `p = 0.00`, so this is a
statistically significant component win on the hermetic 30 s log-mel frontend.

OpenAI Whisper preprocessing comparator, same 30 s deterministic synthetic audio
shape `(80, 3000)`, `torch.set_num_threads(8)`, 15 warmups, then 25 timed
`whisper.audio.log_mel_spectrogram(samples, n_mels=80)` calls with the first 5
timed calls discarded:

```text
steady OpenAI median: 4.814779968 ms
steady OpenAI mean:   4.497219343 ms
candidate median:     4.773900000 ms
candidate/OpenAI speed ratio: 4.814779968 / 4.773900000 = 1.0086x
baseline/OpenAI speed ratio:  4.814779968 / 5.221100000 = 0.9222x
```

Scope: this is a preprocessing-component win, not an e2e claim. It is real
because the same log-mel fixture now moves from slower than OpenAI Whisper's
steady PyTorch preprocessing path to a small steady-state win, but e2e JFK
fixtures at 16 kHz remain dominated by encoder/decoder work and model/load
effects logged elsewhere.

## 2026-06-25 - BlackThrush: multi-accumulator `dot8` REJECTED — 2.5× REGRESSION; `dot8`'s idiom is load-bearing (DO NOT hand-restructure)

Applied the textbook FMA-latency lever (`/extreme-software-optimization`:
*independent accumulators to saturate the FMA units*) to `dot8`
(`src/native_engine/nn.rs`) — the single 8-lane accumulator is a loop-carried
dependency on one ymm, so four disjoint 8-lane accumulators over 32-element
chunks *should* hide FMA latency. Conformance held (tolerance reorder; 27/27 nn
tests green). **Perf went the wrong way, hard.** Criterion A/B vs the committed
`dot8` (`blk_pre` baseline), per-crate via `rch exec -- cargo bench -p
franken_whisper --bench native_engine_bench`, x86 fleet `ovh-a`:

| shape (`native_engine/f16_gemv`) | committed `dot8` | 4-accumulator | change |
| --- | ---: | ---: | --- |
| `dequant_1280x1280` | ~184 µs | 387 µs | **+122%** (p<0.05) → 2.2× slower |
| `dequant_384x384` | 68.3 µs | 157 µs | **+148%** (p<0.05) → 2.5× slower |

**Root cause + the durable lesson.** The committed `dot8` —
`for (ach,bch) in a.chunks_exact(8).zip(b.chunks_exact(8)) { for i in 0..8 {
acc[i] += ach[i]*bch[i] } }` — is a *specific idiom* the Rust/LLVM toolchain
pattern-matches into one tight `vfmadd` per 8 lanes. Indexing a wider chunk
(`ach[8+i]`, `ach[16+i]`, …) **breaks that pattern-match and scalarizes** the
inner loop → ~2.5× slower. This is the **second independent confirmation** this
session: the rejected blocked-dequant (entry below) hit the *identical* failure
(its hand-rolled `x[c+j+l]` fold also scalarized, also ~383 µs at 1280×1280).
Two different "optimizations", same ~383 µs scalarized floor vs the vectorized
~184 µs committed path.

⇒ **`dot8` is at the x86-64-v3 ceiling and its clean `chunks_exact(8)`/`0..8`
form is load-bearing. Do NOT hand-restructure it** (blocking, unrolling, manual
multi-accumulator) — every such attempt on this toolchain defeats autovectoriza-
tion and regresses ~2–2.5×. The single accumulator is *not* latency-bound in
practice (LLVM unrolls + the OOO engine overlaps with the dequant). Real headroom
here would need **wider SIMD** (AVX-512 via an `x86-64-v4` build baseline — a
deliberate min-CPU trade-off needing owner sign-off, cf. L7) or `std::simd`
`f32x8` written so LLVM keeps the FMA form; plain index-restructuring is a trap.
Code REVERTED (stash-preserved, non-destructive). **vs OpenAI-Whisper:** landing
this would have ~halved the dominant per-token decoder GEMV throughput, erasing
franken's large-v3-turbo lead — a net loss, not a gain.

## 2026-06-25 - BlackThrush: blocked-dequant `gemv_f16` REJECTED on x86 — 2.1× REGRESSION (in-code M4 claim did not reproduce)

An **uncommitted working-tree** variant of `gemv_f16`'s `row_dot` (`src/native_engine/nn.rs`)
replaced the committed "bulk-SIMD dequant whole row → clean `dot8`" with an
**interleaved blocked dequant**: dequant the f16 row in 256-element L1 chunks and
fold each chunk into 8-lane accumulators via a hand-rolled
`for l in 0..8 { acc[l] += scratch[j+l] * x[c+j+l] }` inner loop. Its in-code
comment claimed `Standalone (x86-64-v3): 1.18× ([1536,384]) … 1.65× ([1280,5120])
… 1.45× ([51865,1280])`, and the committed NEGATIVE_EVIDENCE ("large decoder
profiled") floated this same blocking as a deferred idea.

**It is a measured REGRESSION on the canonical x86 rch bench fleet** (worker
`ovh-a`, the standard PERF_LEDGER measurement environment). Criterion A/B,
committed baseline (`blk_pre`) vs the working-tree candidate, per-crate via
`rch exec -- cargo bench -p franken_whisper --bench native_engine_bench`:

| shape (`native_engine/f16_gemv`) | baseline | blocked candidate | change |
| --- | ---: | ---: | --- |
| `dequant_384x384` (tiny.en n_state) | 68.3 µs | 82.0 µs | **+19.9%** (p<0.05) |
| `dequant_1280x1280` (large attn Linear) | ~184 µs | 384.6 µs | **+109%** (p<0.05) |

**Root cause.** The committed path does one bulk `convert_to_f32_slice` (8-wide
`f16c`) into scratch, then a clean `chunks_exact(8)` `dot8` that LLVM
auto-vectorizes to tight `vfmadd` over the whole row. The blocked form's
`x[c + j + l]` indexed inner loop + 256-chunk structure **defeats that
auto-vectorization** (and on x86 the small-`inp` whole-row scratch already lives
in L1, so the blocking buys nothing while adding chunk/partial-convert overhead).
The standalone "win" was almost certainly an **M4/aarch64 (4-wide `fp16`)
artifact** — it does not hold on the x86-64-v3 fleet these benches actually run on,
which is precisely the architecture-dependent trap this ledger exists to catch.

**Action.** Code REVERTED (stash-preserved, non-destructive; main's working tree
restored to the committed bulk-dequant `dot8`). Had it been landed it would have
**~halved the dominant per-token decoder GEMV throughput** on x86, widening (not
closing) the franken-vs-whisper.cpp decoder gap and erasing franken's measured
large-v3-turbo e2e win (1.24×) and PyTorch-OpenAI-Whisper lead (2.13–3.26×). The
correct kernel on x86 remains the committed `bulk convert + dot8`. Conformance:
n/a (no source change landed; baseline is the already-green committed code).

## 2026-06-25 - BlackThrush: linear resampler — bit-exact +6% kernel (L16) but e2e-neutral; windowed variant REJECTED

Targeted the **one preprocessing kernel never touched** by L1–L16: the builtin
no-ffmpeg decoder's `resample_mono_linear` (`src/audio.rs`). Two bit-exact
restructures of the clamp-on-every-load loop were measured (standalone microbench,
`rustc -O -C target-cpu=x86-64-v3`, 30 s mono, best-of-60):

| variant | 44.1→16k | 48→16k | 22.05→16k | verdict |
| --- | ---: | ---: | ---: | --- |
| interior/tail split (L16) | **1.065×** | 1.061× | 1.061× | KEEP (bit-exact) |
| windowed-slice `&input[l..l+2]` | 0.981× | 0.978× | 0.972× | **REJECTED** (regression) |

Output is byte-identical to the original (`f32::to_bits()`, 6 rate pairs × 9
lengths, in-tree test green). The split is landed (L16); the windowed variant is a
measured loss and was discarded.

**Ratio "vs OpenAI-Whisper" — honest caveat / non-comparable.** There is **no clean
head-to-head**: OpenAI Whisper resamples via ffmpeg/torch (higher-quality sinc),
franken's builtin path is linear, and this path is **bypassed entirely** when
ffmpeg is present *and* early-returns when `src_rate == dst_rate` (every 16 kHz
input, incl. the jfk e2e fixture). So the +6% is **vs franken's own faithful
baseline** (same convention as L1's honesty note), and it is **e2e-neutral** — a
once-per-file cold-path cleanup, not a transcription-time gap-closer. Recorded so
this small win is not later misread as moving the head-to-head. The real remaining
head-to-head gap stays as previously logged: tiny-cold-CLI (~190 ms, diffuse) and
the diffuse large-decoder `gemv_f16` efficiency vs GGML's hand-tuned dot.

## 2026-06-25 - BlackThrush: large decoder profiled — diffuse gemv, no hotspot (offset by encoder win)

The large-v3-turbo decoder is franken's biggest *component* gap (~2.8 s vs whisper.cpp
~0.585 s ≈ 4.8×) but it's **offset by the 1.9× encoder win** → franken still wins the
large e2e (1.24×). Profiled it for an L9-style hotspot — found NONE:
`decoder_attrib` large (per-token): mlp_fc_gelu 27.9%, logits_gemv 17.5%, self_qkv
17.1%, cross_q/out + self_out 21%, cross_attn 11.3% → **~80% is `gemv_f16`**, all
mature (L11 rayon) with no single dominant spawn/dispatch hotspot. The 4.8× is
**diffuse f16-GEMV kernel efficiency vs GGML's hand-tuned dot**.

One plausible kernel lever (uncertain, NOT a hotspot): franken's `gemv_f16`
dequant-whole-row-to-scratch-then-`dot8` puts a 20 KB scratch (inp=5120 fc2) past L1
→ L2 traffic; a **blocked dequant** (256-elem chunks in an L1 scratch, then dot) would
cut that for large `inp`. Tiny `inp` (≤1280, ≤5 KB scratch) is already L1-resident, so
this only helps large. Deferred: it's a kernel rewrite that *extends* an already-won
e2e, not a gap-closer; franken's only e2e loss remains the tiny-cold-CLI 190 ms.

## 2026-06-25 - BlackThrush: ⇒ DOMINATION SCORECARD + last gap is a modest diffuse tiny-cold-CLI

After L9–L15, franken_whisper matches-or-beats whisper.cpp everywhere measured:

| dimension (jfk, 8t) | franken | whisper.cpp | result |
| --- | ---: | ---: | --- |
| tiny.en transcription | ~480 ms | ~454 ms | ~parity |
| large-v3-turbo transcription | 7.12 s | ~8.85 s | **franken 1.24× win** |
| large-v3-turbo cold-CLI (load+compute) | ~9.2 s | 9.75 s | **franken win** (after L15) |
| **tiny.en cold-CLI** | **710 ms** | 520 ms | franken 1.37× slower (only loss) |
| vs OpenAI Whisper (PyTorch) | — | — | **2.13–3.26× faster** |

**The one remaining loss — tiny.en cold-CLI (710 ms) — is modest AND diffuse**, NOT
the ~1 s startup I'd estimated. Profiled (perf spans + `/usr/bin/time`): model_parse
58 ms + model_weights 87 ms (load 146) + transcribe ~480 ms + startup/audio/output
~84 ms. The 190 ms gap vs whisper.cpp splits ~+80 load / ~+84 startup+audio / ~+26
transcribe — no single radical lever, all small. (whisper.cpp mmaps the model;
franken can't, `#![forbid(unsafe_code)]`, so its load is eager — but at tiny that's
only 146 ms.)

**Remaining levers (both low-priority — franken already dominates):**
1. **Parse streaming** (large): model_parse 1.28 s is the eager `fs::read` of 1.5 GB;
   overlapping read+convert (streaming loader) would cut total load ~2.07→~1.3 s and
   turn the *borderline* large cold-CLI win (9.2 vs 9.75) into a solid one. Substantial
   refactor; mmap is blocked, so streaming is the only path.
2. **Decode-termination nit:** franken appends one trailing token (" a.") on large
   (core text identical) — a `compute_logprobs`/EOT difference, cosmetic.

The substantive "dominate the original" mission is COMPLETE: franken beats whisper.cpp
on transcription at both sizes + cold-CLI large, and OpenAI Whisper PyTorch 2–3×.

## 2026-06-25 - AGENT_NAME=IcyWren cod-b attention Rayon dispatch recheck rejected

### Land-or-dig scan

Repo-local `.scratch` and `.worktrees` directories were absent. Sibling
worktrees were checked with `git worktree list --porcelain`; all measured-code
worktree heads were ancestors of current `main` except
`franken_whisper-cod-a-main-measure`, whose unique commit is stale docs-only
OpenAI-ratio evidence already superseded by current ledger entries. The
`franken_whisper-cod-b-land` release-profile win is already an ancestor of
current `main`.

### RCH bench command status

Requested command shape:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo bench --release -p franken_whisper \
    --bench native_engine_bench -- encoder_window_tiny \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

Result: blocked before benchmark execution. Cargo rejects `bench --release`
with `unexpected argument '--release'`. The supported release-profile form
below was then used, still via `rch exec`, still `-p franken_whisper`, still with
the warm cod-b target dir:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
  rch exec -- cargo bench --profile release -p franken_whisper \
    --bench native_engine_bench -- encoder_window_tiny \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

The detached bench worktree needed a local copy of the gitignored
`tests/fixtures/native/jfk.wav` fixture before model-backed Criterion benches
would run.

### Candidate

Alien-graveyard / extreme-optimization route: persistent worker-pool scheduling
for repeated parallel numeric kernels, with the high-level FrankenSuite rule to
reject mean-only or component-only claims. One-lever probe: replace
`nn::attention_raw`'s per-call `std::thread::scope` head-band dispatch with
Rayon `into_par_iter` over the same band starts. The intended win was removing
repeated thread-spawn overhead while preserving one private output buffer per
band and the existing disjoint merge.

Loss matrix:

```text
keep:   component win plus same-session loaded product median >= 1.03x faster,
        conformance unchanged, OpenAI loaded ratio not worse.
reject: product median < 1.03x faster, OpenAI ratio worsens, or any transcript
        drift.
fallback: retain the scoped-thread attention_raw implementation.
```

### Measurements

Criterion component bench, current `main` baseline:

```text
native_engine/encoder/encoder_window_tiny
time: [233.69 ms 250.04 ms 272.93 ms]
```

Criterion component bench, Rayon candidate:

```text
native_engine/encoder/encoder_window_tiny
time:   [209.39 ms 210.96 ms 212.41 ms]
change: [-21.061% -17.651% -13.812%] (p = 0.00 < 0.05)
component midpoint speedup: 250.04 / 210.96 = 1.185248x
```

Same-session loaded product A/B, `native_ab tiny.en 9 8`, run 0 discarded:

| Build | Runs after warmup (s) | Median | Mean | Ratio vs current franken | OpenAI loaded median | Ratio vs OpenAI loaded | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| current `fb7ed99` | 0.47129, 0.46955, 0.46733, 0.46477, 0.45449, 0.45827, 0.47474, 0.48605 | 0.468440 | 0.468311 | 1.000x | 0.423771608 | 0.904644x | Baseline loss |
| Rayon candidate | 0.54808, 0.57949, 0.51244, 0.50387, 0.79453, 0.54327, 0.51769, 0.51774 | 0.530505 | 0.564639 | 0.883008x | 0.423771608 | 0.798808x | REJECT |

Fresh OpenAI loaded API comparator:

```text
uvx --from openai-whisper python
torch.set_num_threads(8)
whisper.load_model("tiny.en", device="cpu")
one warmup, 9 timed transcribes over already-loaded audio

runs_s:
  [0.4351202598772943, 0.39924179087392986, 0.4191680650692433,
   0.4272299101576209, 0.4279980850405991, 0.4369896319694817,
   0.3955063030589372, 0.4070793210994452, 0.4237716079223901]
mean:   0.4191227750076602 s
median: 0.4237716079223901 s
artifact: /tmp/franken_whisper_cod_b_attention_raw_rayon_openai_loaded_8t.json
```

Behavior proof:

```text
diff -u \
  /tmp/franken_whisper_cod_b_fb7ed99_native_ab_8t_same_session.json \
  /tmp/franken_whisper_cod_b_attention_raw_rayon_native_ab_8t.json
result: no diff
```

Artifacts:

```text
/tmp/franken_whisper_cod_b_fb7ed99_native_ab_8t_same_session.times
/tmp/franken_whisper_cod_b_fb7ed99_native_ab_8t_same_session.json
/tmp/franken_whisper_cod_b_attention_raw_rayon_native_ab_8t.times
/tmp/franken_whisper_cod_b_attention_raw_rayon_native_ab_8t.json
/tmp/franken_whisper_cod_b_attention_raw_rayon_openai_loaded_8t.json
```

### Decision

Rejected and source reverted before commit. The candidate has a real isolated
encoder-window win, but at the loaded product surface it regresses franken from
0.468440 s to 0.530505 s median and worsens the loaded OpenAI Whisper ratio
from 0.904644x to 0.798808x. This is a component-only win that fails the
BOLD-VERIFY product gate.

## 2026-06-25 - BlackThrush: ⇒ LARGE-V3-TURBO head-to-head — franken WINS compute 1.24× (blocker resolved)

Resolved the documented blocker: downloaded `ggml-large-v3-turbo.bin` (1.5 GB f16,
HuggingFace) and added `bench_e2e_large_jfk`. First large head-to-head (jfk 11 s, 8t,
same ggml weights):

| stage (large-v3-turbo, jfk, 8t) | franken | whisper.cpp | result |
| --- | ---: | ---: | --- |
| **encoder (one window)** | **4.31 s** | 8.25 s | **franken 1.9× FASTER** |
| **transcription e2e (no-ts)** | **7.12 s** | ~8.85 s (9.75 total − 0.90 load) | **franken 1.24× FASTER** |
| model load | ~5.8 s (12.96 binary − 7.1 compute) | 0.90 s | **franken ~6× SLOWER** |
| cold binary (incl load) | 12.96 s | 9.75 s | whisper.cpp faster (franken load-bound) |

**franken WINS the large compute** — the encoder (matrixmultiply + rayon) is 1.9×
faster than GGML's encode at large shapes (the opposite of bd-4hc0's tiny finding;
matrixmultiply is competitive at large `[1500,1280]×[1280,5120]` sizes), and that
outweighs a slower decoder. So: **tiny.en ~parity, large-v3-turbo franken 1.24×
faster** — franken beats whisper.cpp on the high-quality model.

**Conformance:** franken core text **identical** to whisper.cpp — "And so, my fellow
Americans, ask not what your country can do for you, ask what you can do for your
country." franken appends one spurious trailing token (" a.") — a decode-termination
difference (29 vs 28 tokens), core 22 word-tokens match. Minor, worth a follow-up.

**NEW in-scope lever surfaced (biggest cold-start gap):** franken's model LOAD is
slower than whisper.cpp's (whisper.cpp load 0.90 s). **PROFILED via perf spans**
(large, cached file): `model_parse` (fs::read 1.5 GB + ggml parse) = **1.28 s**,
`model_weights` (`from_ggml`: per-weight `transpose_parallel` [out,in]→[in,out] +
f16 stage) = **1.97 s** → total **~3.25 s** (the earlier 5.8 s estimate was inflated
by startup/contention; the background SHA-256 hash overlaps and does NOT block the
wall). So the bottleneck is **`model_weights` (the transposes)**, and `EncoderWeights::
from_ggml` loads its layers in a **sequential** `for i in 0..n_layer` loop
(encoder.rs:252) — for large that's 32 layers serial.

**Lever (located, next dig):** parallelize `from_ggml` ACROSS layers (rayon over the
0..n_layer loop), with each layer's transpose run SERIAL (a `transpose_serial`
variant) to avoid the nested-pool oversubscription that rayon-layers × `thread::scope`-
transpose would cause. Est. `model_weights` ~1.97 s → ~0.6–1 s on 8 cores → total
load ~1.9–2.3 s, which would flip cold-CLI large from a loss (12.96 s vs 9.75 s) to a
WIN (~9 s). Bit-identical (the transpose is a pure permutation). SECONDARY priority:
this is a cold/one-time cost amortized in server mode; franken already wins the
*transcription* compute (large 1.24×, tiny ~parity), which is the primary head-to-head.

mmap (zero-copy parse) is BLOCKED by `#![forbid(unsafe_code)]` (memmap2's map is
`unsafe`), so the parse stays an eager `std::fs::read`.

## 2026-06-25 - AGENT_NAME=IcyWren attention Rayon head-band dispatch rejected

### Land-or-dig scan

Repo-local `.scratch` and `.worktrees` directories were absent. Sibling
worktrees were checked with `git worktree list --porcelain`; no measured code
win missing from current `main` was found to land.

### RCH bench command status

Requested command shape:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --release -p franken_whisper \
    --bench native_engine_bench -- e2e_tiny_jfk \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

Result: blocked before benchmark execution. Cargo rejects `bench --release` with
`unexpected argument '--release'`.

Supported release-profile follow-up:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --profile release -p franken_whisper \
    --bench native_engine_bench -- e2e_tiny_jfk \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

Result: command completed per-crate under the warm target dir, but the bench
harness skipped `e2e_tiny_jfk` and the other native benches because the model
directory was not visible to the harness environment.

### Candidate

Alien-graveyard / extreme-optimization route: persistent worker-pool scheduling
for repeated numeric kernels. One-lever probe: replace `attention_raw`'s
per-call `std::thread::scope` head-band dispatch with Rayon `into_par_iter`
over band starts. The intended win was removing repeated thread-spawn overhead
while preserving one private output buffer per band and the existing disjoint
merge, so output order and floating-point operation order per head stayed
unchanged.

### Measurement

Built locally after the RCH bench path could not produce a model-backed timing:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo +nightly-2026-06-09 build -p franken_whisper \
    --profile release --example native_ab --example decoder_attrib
result: pass
```

Loaded franken path:

```text
FRANKEN_WHISPER_MODEL_DIR=.../legacy_whispercpp/whisper.cpp/models \
  /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab tiny.en 9 8

candidate wall_ms:
  632.87, 570.70, 455.58, 473.64, 469.89, 486.16, 489.08, 482.24, 483.15
candidate median after run0: 0.482695s
candidate mean after run0:   0.488805s

current median after run0:   0.449450s
OpenAI Whisper median:       0.463101751s
candidate vs current:        0.931126x
candidate vs OpenAI Whisper: 0.959409x
current vs OpenAI Whisper:   1.030374x
```

Output parity: candidate `native_ab` JSON matched the current franken baseline
exactly.

Decoder attribution also moved the wrong way:

```text
current decoder_attrib tiny.en 160:
  total attributed: 8.2432 ms/token
  mlp_fc_gelu_proj: 2.0315 ms/token
  logits_gemv:      1.7171 ms/token
  self_qkv_proj:    1.3783 ms/token
  cross_attn:       1.1091 ms/token

candidate decoder_attrib tiny.en 160:
  total attributed: 8.9484 ms/token
  mlp_fc_gelu_proj: 2.8995 ms/token
  cross_attn:       1.5470 ms/token
  logits_gemv:      1.3459 ms/token
  self_qkv_proj:    1.2252 ms/token
```

### Decision

Rejected. The candidate regressed the loaded tiny.en path versus both current
franken and OpenAI Whisper. No code was retained; the working tree is back to
the scoped-thread `attention_raw` implementation.

## 2026-06-25 - AGENT_NAME=IcyWren stack-scratch GEMV probe rejected

### Land-or-dig scan

Repo-local `.scratch` and `.worktrees` directories were absent. Sibling worktrees
were checked with `git worktree list --porcelain`; all measured-code worktrees
were ancestors of current `main` except `franken_whisper-cod-a-main-measure`,
which is the stale docs-only OpenAI-ratio branch already superseded by main's
current ledger entries. Nothing missing from current `main` was landable.

### RCH bench command status

Requested command shape:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --release -p franken_whisper \
    --bench native_engine_bench -- e2e_tiny_jfk \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

Result: blocked before benchmark execution. Cargo rejects `bench --release` with
`unexpected argument '--release'`.

Supported release-profile follow-up:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --profile release -p franken_whisper \
    --bench native_engine_bench -- e2e_tiny_jfk \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 3
```

Result: blocked before benchmark execution by path-dependency version skew in
the synced `frankensqlite` workspace: `fsqlite` requested
`fsqlite-parser ^0.1.12`, while the path crate reported `0.1.11`.

### Candidate

Alien-graveyard / extreme-optimization route: cache-aware numeric-kernel
scratch placement. The current attribution showed GEMV-heavy costs dominating
the loaded tiny.en decoder step:

```text
current decoder_attrib tiny.en 160:
  total attributed: 8.2432 ms/token
  mlp_fc_gelu_proj: 2.0315 ms/token
  logits_gemv:      1.7171 ms/token
  self_qkv_proj:    1.3783 ms/token
  cross_attn:       1.1091 ms/token
```

One-lever probe: replace per-call/per-worker heap scratch `Vec<f32>` in
`nn::gemv_f16` and `nn::gemv_f16_batch` with stack-resident scratch arrays for
the common Whisper widths 384, 1280, and 1536, falling back to `Vec` for other
widths. This was intended to remove allocator traffic without changing
dequantization, dot-product order, output-row order, or logits filtering.

### Measurement

Built locally after RCH bench blockage:

```text
AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo +nightly-2026-06-09 build -p franken_whisper \
    --profile release --example native_ab --example decoder_attrib
result: pass
```

Loaded franken path:

```text
FRANKEN_WHISPER_MODEL_DIR=legacy_whispercpp/whisper.cpp/models \
  /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
  tiny.en 9 8
```

Run 0 discarded:

| Build | Runs after warmup (s) | Median | Mean | Ratio vs current franken | OpenAI loaded median | Ratio vs OpenAI loaded | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| current `a3f5cee` | 0.45257, 0.43648, 0.42870, 0.44397, 0.44733, 0.46900, 0.46109, 0.45157 | 0.449450 | 0.448839 | 1.000x | 0.463102 | 1.030x | Baseline |
| stack scratch candidate | 0.60755, 0.68237, 0.53276, 0.52593, 0.52014, 0.48834, 0.51435, 0.61754 | 0.529345 | 0.561123 | 0.849x | 0.463102 | 0.875x | REJECT |

OpenAI loaded-array comparator:

```text
uvx --from openai-whisper python
torch.set_num_threads(8)
whisper.load_model("tiny.en", device="cpu")
one warmup, 9 timed transcribes over an already-loaded NumPy WAV array
median: 0.4631017509382218 s
artifact: /tmp/franken_whisper_cod_a_openai_loaded_array_8t_20260625e.json
```

Behavior proof:

```text
diff -u \
  /tmp/franken_whisper_cod_a_current_a3f5cee_native_ab_8t_20260625e.json \
  /tmp/franken_whisper_cod_a_candidate_stack_scratch_native_ab_8t_20260625e.json
result: no diff
```

Candidate attribution also regressed rather than helping:

```text
candidate decoder_attrib tiny.en 160:
  total attributed: 12.6505 ms/token
  mlp_fc_gelu_proj: 3.3216 ms/token
  logits_gemv:      2.6671 ms/token
  cross_attn:       2.4900 ms/token
```

Verdict: rejected and reverted. Stack arrays for the common f16 GEMV widths
appear to hurt this hot loop, likely via stack-frame pressure / poorer codegen
rather than allocator savings. Do not retry this family without assembly or
perf-counter evidence showing allocation traffic is actually material.

## 2026-06-25 - AGENT_NAME=IcyWren OpenAI Whisper after Rayon pool cap

### Worktree scan

No measured win was found in a detached worktree that was missing from current
`main`. The earlier greedy-logprob allocation lever (`3cbd80e`) was explicitly
reverted by `6d0d5be` as a measured near-zero path, so it was not re-landed.
Current head for this pass: `a9ecb3b`.

### Kept in-crate lever

Lever: initialize Rayon's global pool to `native_engine::default_threads()` when
`RAYON_NUM_THREADS` is unset. On this 64-way host, Rayon's default pool was too
wide for the native engine's tuned 8-16-worker kernels. The new default is 16
threads, while an explicit `RAYON_NUM_THREADS` override remains honored.

Loaded-model A/B, `examples/native_ab tiny.en 9 <threads>`, run 0 discarded:

| Threads | Baseline median | Candidate median | Franken speedup | OpenAI loaded median | Candidate ratio vs OpenAI loaded | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 0.603520 s | 0.540470 s | 1.117x | 0.582495 s | 1.078x | Beats OpenAI loaded API on this 4t run |
| 8 | 0.624235 s | 0.535540 s | 1.166x | 0.420035 s | 0.784x | Still slower than OpenAI loaded API at 8t |

Artifacts:

```text
/tmp/franken_whisper_cod_b_a9ecb3b_native_ab_4t.times
/tmp/franken_whisper_cod_b_a9ecb3b_native_ab_8t.times
/tmp/franken_whisper_cod_b_rayoncap_candidate_native_ab_4t.times
/tmp/franken_whisper_cod_b_rayoncap_candidate_native_ab_8t.times
/tmp/franken_whisper_cod_b_a9ecb3b_openai_loaded.json
```

Behavior proof:

```text
diff -u \
  /tmp/franken_whisper_cod_b_a9ecb3b_native_ab_8t.json \
  /tmp/franken_whisper_cod_b_rayoncap_candidate_native_ab_8t.json
result: no diff

diff -u \
  /tmp/franken_whisper_cod_b_a9ecb3b_native_ab_4t.json \
  /tmp/franken_whisper_cod_b_rayoncap_candidate_native_ab_4t.json
result: no diff
```

Decoder attribution:

```text
baseline:
  wall total=2612.8ms, per-step=13.064ms
  top costs: mlp_fc_gelu_proj 3.8018ms/tok, logits_gemv 2.6819ms/tok,
    cross_attn 2.4902ms/tok

candidate:
  wall total=2375.6ms, per-step=11.878ms
  top costs: mlp_fc_gelu_proj 3.8129ms/tok, logits_gemv 1.9428ms/tok,
    cross_attn 2.1612ms/tok
```

### Fresh OpenAI Whisper CLI comparator

`speed_ratio = OpenAI Whisper wall time / franken_whisper wall time`.

| Build | Franken mean | Franken median | OpenAI CLI mean | OpenAI CLI median | Mean ratio | Median ratio | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline `a9ecb3b` | 0.935723 s | 0.933698 s | 2.994287 s | 3.023920 s | 3.200x | 3.239x | Routing baseline |
| Rayon-cap candidate | 0.783912 s | 0.774636 s | 3.318963 s | 3.259243 s | 4.234x | 4.207x | Kept product CLI win |

Artifacts:

```text
/tmp/franken_whisper_cod_b_a9ecb3b_openai_cli.json
/tmp/franken_whisper_cod_b_rayoncap_candidate_openai_cli.json
```

### Validation

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  cargo build --profile release -p franken_whisper \
  --bin franken_whisper --example native_ab --example decoder_attrib
result: pass

CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  cargo test -p franken_whisper --test conformance_comparator_tests
result: pass; 26 passed / 0 failed
```

Additional quality gates are recorded in the commit/session closeout. Known
boundary: this is a real loaded-model franken win and a stronger OpenAI CLI win,
but it does not beat the loaded OpenAI API at 8 threads.

## 2026-06-25 - BlackThrush: ⇒ AFTER L9–L13, NEAR PARITY (gap 1.37×→~1.08×)

Fresh back-to-back re-measure after the L9–L13 decoder optimizations (same host,
same 8 threads, transcription-only):

| Workload (tiny.en jfk, 8t, transcription-only) | franken | whisper.cpp | ratio |
| --- | ---: | ---: | ---: |
| apples-to-apples (both no word timestamps) | **486 ms** | ~452 ms (522 total − 70 load) | **~0.93× (franken 1.07× slower — NEAR PARITY)** |
| franken realistic (WITH DTW word timestamps) | 504 ms | n/a (whisper.cpp `dtw=0`) | franken adds word ts for +18 ms |

**Re-verified later same day @ HEAD (peer agent's L14 rayon-pool cap merged, higher
host contention):** franken no-ts **489 ms** vs whisper.cpp ~499 ms transcription
(569 total − 70 load) = **~parity / franken ~1.02× faster under load**. franken's
number is *stable* across measurements (~486–489 ms) while whisper.cpp swings with
contention (452–499 ms) — so the gap is now **~parity (1.0–1.07×, contention-
dependent)**, robustly holding. The decoder optimization (L9–L13) + the peer's L14
took franken from 1.37× slower to dead-even with whisper.cpp on tiny.en CPU.

**From 1.37× slower → ~1.07× (near parity) via 5 in-scope, conformance-green decoder
wins** (L9 mlp spawn threshold, L10 m=1 gemv, L11 rayon gemv_f16, L12 rayon
cross-attn no-ts, L13 rayon cross-attn record/ts). All the missing pieces were
whisper.cpp/GGML techniques: persistent thread pool (vs per-call `thread::scope`
spawn) + dedicated gemv (vs sgemm packing at m=1). franken's mel + encoder already
won; this closed the decoder. Outright win still needs bd-4hc0 (encoder
`matrixmultiply→gemm`, ~2× the encoder; out-of-scope `ft-kernel-cpu`). See
`docs/PERF_LEDGER.md` L9–L13 for the per-lever evidence.

## 2026-06-25 - BlackThrush: whisper.cpp comparator BUILT + first head-to-head (bd-zk43)

The prior blocker ("`whisper-cli` and ggml models absent") is **RESOLVED**: I built
`whisper-cli` from `legacy_whispercpp/whisper.cpp` (cmake Release, gcc, ggml 0.9.6)
and `ggml-tiny.en.bin` is present. This is the first **engine-level** franken vs
**whisper.cpp** head-to-head (same `ggml-tiny.en` weights, same `jfk.wav`, same 8
threads, same contended host, transcription-only / model pre-loaded).

| Workload (tiny.en, jfk 11s, 8t, transcription-only) | franken | whisper.cpp | ratio |
| --- | ---: | ---: | ---: |
| full pipeline (franken w/ DTW timestamps; whisper.cpp `dtw=0`) | 614 ms | ~448 ms | **0.73× (franken 1.37× SLOWER)** |
| apples-to-apples (both no word timestamps) | 596 ms | ~448 ms | **0.75× (franken 1.33× SLOWER)** |

**The README's "tiny.en 475ms vs whisper.cpp 1105ms (2.33×)" is STALE/FALSE.**
whisper.cpp is actually ~448ms transcription (514ms incl. 66ms load) — not 1105ms.
franken is ~1.33× **slower** than whisper.cpp here. (NB franken *does* still win
~2.13–3.26× vs **OpenAI Whisper PyTorch** — a much slower baseline — per the
2026-06-24 entries; the two "originals" are not the same bar.)

**Stage breakdown — the gap is the DECODER, not the encoder:**

| stage | franken | whisper.cpp | note |
| --- | ---: | ---: | --- |
| mel | ~4 ms | ~8 ms | franken WINS (L1/L3/L4 levers) |
| encoder | ~204 ms | ~242 ms | franken WINS slightly |
| decoder (+sample) | **~388 ms** | **~198 ms** | **whisper.cpp ~2× faster** |
| DTW word ts | ~18 ms | n/a (dtw=0) | franken-only feature |

This **overturns the session's prior focus**: bd-4hc0 (encoder GEMM 3.75×) would
*extend* an already-winning encoder, but the real loss vs whisper.cpp is the
**decoder (~2×), which is in-scope franken code** (gemv_f16 + the per-token loop).
The earlier "decoder gemv is mature" conclusion compared franken-to-franken; vs
GGML there is ~2× headroom. NEXT: profile the franken decoder per-token to locate
the 2× (logits gemv vs GGML f16 matmul? per-token overhead/allocations? flash
attn? whisper.cpp uses `flash attn = 1`). Also flagged: franken **binary**
wall-clock (1.733 s hyperfine) vs whisper-cli total (~514 ms) implies ~1.1 s
franken startup/model-load overhead worth a separate look (CLI/short-clip UX).

Reproduce:

```text
# build whisper-cli
cmake -S legacy_whispercpp/whisper.cpp -B legacy_whispercpp/whisper.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build legacy_whispercpp/whisper.cpp/build -j 64 --target whisper-cli
# whisper.cpp (reports load/mel/encode/decode/total)
legacy_whispercpp/whisper.cpp/build/bin/whisper-cli \
  -m legacy_whispercpp/whisper.cpp/models/ggml-tiny.en.bin \
  -f legacy_whispercpp/whisper.cpp/samples/jfk.wav -t 8
# franken (engine-level, pre-loaded; v3 build, n_threads=8 in bench)
FRANKEN_WHISPER_MODEL_DIR=.../legacy_whispercpp/whisper.cpp/models \
  RCH_MIN_LOCAL_TIME_MS=99999999 CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cc \
  cargo bench -p franken_whisper --bench native_engine_bench -- e2e_tiny_jfk
```

## 2026-06-24 - franken_whisper-cod-b kickoff

### Scope

- Goal: dominate the original OpenAI Whisper / `whisper.cpp` lineage on realistic
  workloads with measured head-to-head ratios.
- Required comparator: current `franken_whisper` native or pipeline path vs the
  original implementation on the same worker, same model, same audio, same
  correctness gate.
- Required build lane: crate-scoped `rch` benchmark with
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b`.

### Fresh evidence collected

| Workload | Franken path | Original path | Ratio vs original | Verdict |
| --- | --- | --- | ---: | --- |
| Native engine criterion bench | `native_engine_bench` via `rch` | none | N/A | Completed native-only; useful routing, not head-to-head |
| Exact requested bench command | `cargo bench --release -p franken_whisper --bench native_engine_bench` | none | N/A | Blocked: Cargo rejects `--release` for `bench` |
| Head-to-head `whisper.cpp` run | none | `whisper-cli` | N/A | Blocked: `whisper-cli` not installed on this host |
| Model-gated native E2E | `tiny.en` / `large-v3-turbo` | `ggml-*.bin` | N/A | Blocked: no local ggml model files found in default search dirs |

Command evidence:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo bench --profile release -p franken_whisper \
  --bench native_engine_bench -- --sample-size 10 --warm-up-time 0.1 \
  --measurement-time 0.1

worker: vmi1152480
result: pass, remote elapsed 538.3 s

native_engine/mel/mel_30s:
  time [163.94 ms 165.97 ms 167.16 ms]

native_engine/f16_gemv/f16_gemv_dequant_1280x1280:
  time [639.33 us 735.46 us 828.75 us]
  throughput [1.9769 Gelem/s 2.2277 Gelem/s 2.5627 Gelem/s]

native_engine/f16_gemv/f16_gemv_dequant_384x384:
  time [134.92 us 144.90 us 159.33 us]
  throughput [925.49 Melem/s 1.0177 Gelem/s 1.0929 Gelem/s]

skips:
  encoder_window_tiny: model tiny.en or jfk.wav missing
  encoder_window_large: model large-v3-turbo or jfk.wav missing
  decoder_token_step_tiny: model tiny.en or jfk.wav missing
  decoder_token_step_large: model large-v3-turbo or jfk.wav missing
  logits_gemv_large: model large-v3-turbo missing
  e2e_tiny_jfk: model tiny.en missing
```

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo bench --release -p franken_whisper \
  --bench native_engine_bench -- --sample-size 10 --warm-up-time 0.1 \
  --measurement-time 0.1

worker: ovh-a
result: error: unexpected argument '--release' found
```

Comparator availability:

```text
which whisper-cli -> not found
find ~/.cache/franken_whisper ~/models /data ... ggml*.bin -> no results
```

Coordination availability:

```text
MCP Agent Mail macro_start_session -> HTTP request failed at
http://127.0.0.1:8765/mcp/
curl http://127.0.0.1:8765/health -> connection refused
```

### Historical evidence, not fresh proof

These artifacts are useful for routing but are not current head-to-head proof:

- `README.md` claims `tiny.en` 11 s clip: native CPU 475 ms vs `whisper.cpp`
  1,105 ms, or 2.33x faster; `large-v3-turbo` 9.73 s vs 9.59 s, parity.
- `tests/artifacts/perf/20260606T2341Z-scale-baseline/PASS4_RESULTS.md`
  records an internal native IF vs sequential tiny.en scale result:
  262.9 s vs pass-1 604 s on a 2 h call, or 2.30x faster. This is not an
  original-implementation comparison.
- The same PASS4 artifact rejects work stealing, lower worker thread minima,
  and model-size-aware worker policy as no-ship levers.

### Current blockers to real ratios

1. Install or point `FRANKEN_WHISPER_WHISPER_CPP_BIN` at a valid `whisper-cli`.
2. Provide the same ggml model files to both native and original paths, at
   minimum `ggml-tiny.en.bin` and `ggml-large-v3-turbo.bin`.
3. Use Cargo's supported release-profile spelling for benches:
   `cargo bench --profile release ...` or document a wrapper that accepts
   `cargo bench --release`.
4. Run on one worker with interleaved A/B ordering and conformance gates:
   WER, segment timestamps, replay envelope, and exact model identity.

### First radical lever queue

These are candidates only. None may ship without a fresh profile, comparator,
green conformance, and a before/after ratio in this ledger.

| Rank | Lever | Graveyard / artifact mapping | Proof gate |
| ---: | --- | --- | --- |
| 1 | Native-vs-`whisper.cpp` head-to-head harness for 11 s, 10 min, 2 h, noisy, multilingual, and long-form YouTube audio | Evidence ledger + safe data-plane fallback; benchmark honesty gate | Same-worker ratio, WER 0.0000 or bounded drift, model identity hash |
| 2 | Decoder/logits f16-resident GEMV layout, SIMD/auto-vec, cache-sized bands, branchless hot loops | Vectorized execution, Swiss-table-style metadata thinking, cache-oblivious layout | `native_engine_bench` logits/decoder wins and E2E no regression |
| 3 | Built-in normalizer vs ffmpeg batch dominance: arena/buffer reuse, SIMD resample/mix, cache-friendly WAV writes | Cache-oblivious data movement, arena allocation, staged early exit | `normalize_bench` plus real MP3/FLAC/video A/B against ffmpeg |
| 4 | TTY audio throughput: reusable buffers, compression policy, FEC control frames, branchless decode checks | Adaptive controller with deterministic fallback, evidence ledger | `tty_bench`, decode SNR/integrity conformance, no protocol drift |
| 5 | Long-form pipeline scheduler: batch windows with bounded seams, no-regret/conformal guard for fallback | Learning-augmented policy with reject option and conservative fallback | Real 2 h audio, word-diff bound, p95 wall and RSS ratio |

### Filed perf-lever beads

- `bd-zk43`: P0 native vs `whisper.cpp` real-workload head-to-head harness,
  dependent on the broader ledger-harness bead `bd-0hnz`.
- `bd-n0m3`: P1 decoder/logits f16 GEMV layout and SIMD lever.
- `bd-3nw3`: P1 built-in audio normalization vs ffmpeg batch dominance.
- `bd-cy9u`: P1 streaming TTY transport throughput and control-frame FEC.
- `bd-3vhz`: P1 long-form scheduler and speculative window policy.

The P1 levers depend on `bd-zk43` so product-speed claims cannot bypass the
original-comparator evidence gate.

### Validation after this entry

```text
git diff --check -> pass
cargo fmt --check -> pass
br dep cycles -> pass, no dependency cycles detected
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo test -p franken_whisper \
  --test conformance_comparator_tests
result: local fallback, pass, 26 passed / 0 failed
ubs docs/NEGATIVE_EVIDENCE.md .beads/issues.jsonl -> not applicable; no
supported languages detected
```

## 2026-06-24 - franken_whisper-cod-a kickoff

### Ratio convention

`speed_ratio = original_wall_time / franken_wall_time`.

- `> 1.0`: franken is faster.
- `= 1.0`: parity.
- `< 1.0`: franken is slower.

### Existing original-vs-franken ratios normalized

These rows were mined from the checked-in performance artifacts and README
claims. They are useful routing evidence, but must be refreshed before any new
dominance claim ships.

| Workload | Franken path | Original path | Ratio vs original | Verdict |
| --- | --- | --- | ---: | --- |
| 11 s JFK, `tiny.en`, CPU | native release-perf, 475 ms | `whisper.cpp` CPU, 1,105 ms | 2.33x | Historical win; refresh required |
| 11 s JFK, `large-v3-turbo`, CPU | native release-perf, 9.731 s | `whisper.cpp` CPU, 9.585 s | 0.985x | Historical wall-time loss/parity; native had lower user CPU |
| 11 s JFK, `large-v3-turbo`, CPU vs GPU control | native CPU, 9.731 s | `whisper.cpp` Metal, 2.169 s | 0.223x | Historical loss vs GPU control; not a CPU claim |
| 2 h tiny workload policy comparison | native IF default, 262.9 s | native sequential pass, 604 s | N/A | Internal franken comparison; not original-vs-franken |
| YouTube metadata optimization | reduced metadata probes | prior franken path | N/A | Internal I/O win; not original-vs-franken |

### Fresh validation commands

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --release --bench normalize_bench -- \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1

result: blocked by Cargo CLI shape
rch route: local fallback, no admissible workers
error: unexpected argument '--release' found
```

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo bench --bench normalize_bench -- \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1

result: pass
rch route: local fallback, no admissible workers
bench profile: optimized
cold build: 4m 29s

normalize/whisper_cpp/segments/1:    [1.2256 us 1.2692 us 1.3816 us]
normalize/whisper_cpp/segments/10:   [7.4960 us 8.1008 us 8.7399 us]
normalize/whisper_cpp/segments/100:  [92.081 us 96.753 us 102.31 us]
normalize/whisper_cpp/segments/500:  [392.94 us 407.52 us 423.29 us]
normalize/insanely_fast/chunks/500:  [653.37 us 732.28 us 825.94 us]
normalize/insanely_fast_batch/100x20:[1.1473 ms 1.2984 ms 1.4611 ms]
to_transcription_result/segments/100:[3.6775 us 3.8964 us 4.3087 us]

verdict: build/bench gate only. No original comparator, no product-speed claim.
```

### Abandoned or scoped-out historical levers

| Lever | Observed result | Verdict |
| --- | --- | --- |
| Fused QKV projection | About 16% slower in prior artifacts | Rejected |
| Per-head batched matmul | About 4% slower in prior artifacts | Rejected |
| Residual-add fusion | Slower in prior artifacts | Rejected |
| Encoder scratch/output reuse | Neutral in prior artifacts | Rejected as no-ship |
| Encoder f16 panels | Overhead/regression in prior artifacts | Rejected |
| Cross-attention f16 K/V | Rejected in prior artifacts | Rejected |
| Wider cross-attention head workers | Neutral in prior artifacts | Rejected |
| Per-token linear widening | About 29% slower in prior artifacts | Rejected |
| Work stealing / model-size-aware worker policy | No reliable long-form win in prior artifacts | Rejected |

### Perf-lever beads filed by cod-a

- `bd-0hnz`: P0 original-vs-franken benchmark ledger harness.
- `bd-1bjy`: P0 shipped release-profile native speed gap.
- `bd-vsg6`: P1 native word timestamp DTW vs `whisper.cpp`.
- `bd-z4o7`: P1 real codec normalization vs ffmpeg plus `whisper.cpp`.
- `bd-9sc3`: P1 diarization and noisy multi-speaker workloads.
- `bd-kdg7`: P2 live/speculative streaming TTFT vs original streaming baselines.

`bd-0hnz` blocks the downstream cod-a perf levers so optimization work cannot
bypass the original-comparator evidence gate.

## 2026-06-24 - franken_whisper-cod-a OpenAI Whisper head-to-head

### Fresh OpenAI Whisper comparator ratio

| Workload | Franken path | Original path | Ratio vs original | Conformance | Verdict |
| --- | --- | --- | ---: | --- | --- |
| 11 s JFK, `tiny.en`, CPU, 8 threads | `franken_whisper` native `whisper.cpp-native`, release-perf, ggml `tiny.en` | OpenAI Whisper `openai-whisper==20250625`, PyTorch CPU, `tiny.en` | 2.13x | Normalized word tokens identical; raw punctuation differs by comma/leading space | Fresh measured win |

Command evidence:

```text
git SHA: 2ef3fa8
build:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
    rch exec -- cargo build -p franken_whisper --profile release-perf \
    --bin franken_whisper
  result: pass; rch local fallback; release-perf build 7m23s

franken command:
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
  FRANKEN_WHISPER_NATIVE_EXECUTION=1 \
  FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole \
  /data/projects/.rch-targets/franken_whisper-cod-a/release-perf/franken_whisper \
    transcribe --input /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    --backend whisper-cpp --model tiny.en --language en --threads 8 \
    --no-persist --json >/dev/null

OpenAI Whisper command:
  out=/tmp/franken_whisper_cod_a_openai_run_$(date +%s%N); mkdir -p "$out"; \
  PATH=/home/ubuntu/.local/state/franken_whisper/tools/ffmpeg/bin:$PATH \
  uvx --from openai-whisper whisper \
    /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    --model tiny.en --language en --device cpu --fp16 False --threads 8 \
    --output_format json --output_dir "$out" --verbose False >/dev/null

hyperfine:
  --warmup 1 --runs 5
  franken mean: 1.733 s +/- 0.685 s [user 3.869 s, sys 1.470 s]
  OpenAI mean: 3.698 s +/- 0.653 s [user 12.682 s, sys 0.766 s]
  speed_ratio = 3.698 / 1.733 = 2.13x

conformance:
  franken transcript:
    And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
  OpenAI Whisper transcript:
    " And so, my fellow Americans ask not what your country can do for you ask what you can do for your country."
  normalized lowercase alnum tokens: identical, 22/22 tokens.
```

Notes:

- This entry is a fresh product-level comparator measurement against OpenAI
  Whisper, not evidence for the in-progress `src/native_engine/mel.rs` twiddle
  precompute lever.
- The uncommitted mel twiddle lever and `docs/PERF_LEDGER.md` were reserved by
  `BlackThrush` during this run, so cod-a did not edit or land that code.

## 2026-06-24 - franken_whisper-cod-a OpenAI Whisper after mel twiddle

### Fresh current-main OpenAI Whisper comparator ratio

| Workload | Franken path | Original path | Ratio vs original | Conformance | Verdict |
| --- | --- | --- | ---: | --- | --- |
| 11 s JFK, `tiny.en`, CPU, 8 threads | `franken_whisper` native `whisper.cpp-native`, `release-perf`, ggml `tiny.en`, commit `b0577d9` | OpenAI Whisper `openai-whisper==20250625`, PyTorch CPU, `tiny.en` | 3.26x | Normalized word tokens identical, 22/22; raw punctuation differs by comma/leading space | Fresh measured current-main win |

Command evidence:

```text
git SHA: b0577d9
worktree: /data/projects/franken_whisper-cod-a-main-measure

build:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
    rch exec -- cargo build -p franken_whisper --profile release-perf \
    --bin franken_whisper
  result: pass; rch remote vmi1264463; release-perf build 14m39s

bench:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
    rch exec -- cargo bench --profile release-perf -p franken_whisper \
    --bench native_engine_bench -- --sample-size 10 --warm-up-time 0.1 \
    native_engine/mel/mel_30s
  result: pass; rch remote hz2
  native_engine/mel/mel_30s: [38.150 ms 40.770 ms 43.015 ms]
  note: `cargo bench --release` remains invalid on this Cargo; it exits with
    `unexpected argument '--release'`, so `--profile release-perf` is the
    package-scoped equivalent used here.

franken command:
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
  FRANKEN_WHISPER_NATIVE_EXECUTION=1 \
  FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole \
  /data/projects/.rch-targets/franken_whisper-cod-a/release-perf/franken_whisper \
    transcribe --input /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    --backend whisper-cpp --model tiny.en --language en --threads 8 \
    --no-persist --json >/dev/null

OpenAI Whisper command:
  out=/tmp/franken_whisper_cod_a_openai_run_$(date +%s%N); mkdir -p "$out"; \
  PATH=/home/ubuntu/.local/state/franken_whisper/tools/ffmpeg/bin:$PATH \
  uvx --from openai-whisper whisper \
    /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    --model tiny.en --language en --device cpu --fp16 False --threads 8 \
    --output_format json --output_dir "$out" --verbose False >/dev/null

hyperfine:
  --warmup 1 --runs 5
  export: /tmp/franken_whisper_cod_a_openai_jfk_tiny_after_mel_hyperfine.json
  franken mean: 0.907338 s +/- 0.015135 s [user 3.958501 s, sys 1.232468 s]
  OpenAI mean: 2.957021 s +/- 0.057506 s [user 11.522156 s, sys 0.685133 s]
  speed_ratio = 2.957021 / 0.907338 = 3.26x

conformance:
  franken transcript:
    And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
  OpenAI Whisper transcript:
    " And so, my fellow Americans ask not what your country can do for you ask what you can do for your country."
  normalized lowercase alnum tokens: identical, 22/22 tokens.
```

Notes:

- This lands the current-main product-level ratio after the `src/native_engine/mel.rs`
  twiddle precompute commit (`656f55c`) and its clippy follow-up (`b0577d9`).
- Agent Mail writes were unavailable during this run because its SQLite database
  reported corruption and refused writes; no MCP file reservation could be
  created for this ledger-only edit.

### Rule for future entries

Every future entry must include: command, worker/host, git SHA, model SHA or
path, workload, original time, franken time, ratio, conformance result, verdict,
and whether the code was kept, reverted, or only routed into a bead.

## 2026-06-24 - franken_whisper-cod-b OpenAI loaded-model API check

### Ratio convention

`speed_ratio = original_wall_time / franken_wall_time`.

- `> 1.0`: franken is faster.
- `= 1.0`: parity.
- `< 1.0`: franken is slower.

### Fresh loaded-model comparator ratio

This entry uses the OpenAI Whisper Python API with the model loaded before the
timed section. That is a different workload from the CLI-startup comparator
above, where each OpenAI run pays Python process and model-load cost.

| Workload | Franken path | Original path | Ratio vs original | Conformance | Verdict |
| --- | --- | --- | ---: | --- | --- |
| 11 s JFK, `tiny.en`, CPU, model-reuse/API comparator | `franken_whisper` native `whisper.cpp-native`, shipped `release` profile, ggml `tiny.en`, CLI one-shot | OpenAI Whisper Python API, model loaded once before timing, PyTorch CPU, `tiny.en` | 0.465x | Normalized word tokens identical; franken adds final punctuation | Fresh measured loss for loaded-model API workload |

Command evidence:

```text
git SHA: 656f55c
build:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
    rch exec -- cargo build --profile release -p franken_whisper \
    --bin franken_whisper
  result: pass; rch worker ovh-a; release build 3m30s; artifact retrieved to
  /data/projects/.rch-targets/franken_whisper-cod-b/release/franken_whisper

OpenAI Whisper API command:
  uv tool run --from openai-whisper python - <<'PY'
  import json, time, wave
  import numpy as np
  import whisper
  path = 'tests/fixtures/native/jfk.wav'
  with wave.open(path, 'rb') as w:
      raw = w.readframes(w.getnframes())
  audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
  model = whisper.load_model('tiny.en', device='cpu')
  model.transcribe(audio, language='en', fp16=False, verbose=False,
                   temperature=0.0, condition_on_previous_text=False)
  runs = []
  texts = []
  for _ in range(5):
      start = time.perf_counter()
      result = model.transcribe(audio, language='en', fp16=False,
                                verbose=False, temperature=0.0,
                                condition_on_previous_text=False)
      runs.append(time.perf_counter() - start)
      texts.append(result.get('text', '').strip())
  print(json.dumps({'runs_s': runs, 'texts': texts}, indent=2))
  PY

OpenAI API result:
  runs_s: [0.4218380448874086, 0.49884854396805167,
           0.4356057639233768, 0.43359226104803383,
           0.4581331869121641]
  median: 0.4356057639233768 s
  transcript:
    And so my fellow Americans ask not what your country can do for you ask what you can do for your country

franken command:
  RUST_LOG=error \
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
  FRANKEN_WHISPER_NATIVE_EXECUTION=1 \
  FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole \
  FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG=0 \
  /data/projects/.rch-targets/franken_whisper-cod-b/release/franken_whisper \
    transcribe --input /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    --backend whisper-cpp --model tiny.en --language en --temperature 0.0 \
    --beam-size 1 --best-of 1 --no-persist --json

franken hyperfine:
  --warmup 1 --runs 5
  mean: 0.9460924366400001 s +/- 0.01580093129598703 s
  median: 0.93704627044 s
  min/max: 0.93067355044 s / 0.9646279524400001 s
  raw runs: [0.9646279524400001, 0.9364485274400001,
             0.9616658824400001, 0.93067355044, 0.93704627044]
  user/system: 5.5155 s / 1.1490 s

franken run metadata:
  backend_identity: whisper.cpp-native
  backend_version: native-pilot-v1/0.2.0
  input_content_hash: d16054d2df9adaae9c6228d86113f256a4b43d448d1f6b9107b75e2136a934a0
  output_payload_hash: 7fdbdde9a772bc5bda7d9933f1e475224837e8081f9505290f871316d87fd486
  backend.ok elapsed_ms: 880
  latency summary: service_total_ms=905, queue_total_ms=1
  transcript:
    And so my fellow Americans ask not what your country can do for you ask what you can do for your country.

ratio:
  speed_ratio = 0.4356057639233768 / 0.93704627044 = 0.464871x
  inverse = 2.151134x slower than the loaded OpenAI API comparator
```

Notes:

- The landed mel twiddle commit `656f55c` improves the stale pre-`656f55c`
  release binary check from `0.327x` to `0.465x` on this loaded-model API
  comparator, but this remains a loss.
- This does not negate the previous CLI-startup comparator win; it narrows the
  claim boundary. `franken_whisper` currently wins when the original pays CLI
  startup/model-load cost, but loses to a reusable in-process OpenAI Whisper
  Python model on this short `tiny.en` fixture.
- No code lever was attempted in this session because the only fresh measured
  win found in the checkout (`656f55c`, mel twiddle precompute) was already on
  `main` and `master`.

Validation after this entry:

```text
git diff --check -- docs/NEGATIVE_EVIDENCE.md -> pass
ubs docs/NEGATIVE_EVIDENCE.md -> not applicable; no supported Markdown scanner
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo test -p franken_whisper --test conformance_comparator_tests
result: local fallback, pass, 26 passed / 0 failed
```

## 2026-06-24 - franken_whisper-cod-b release opt-level throughput lever

### Lever

Change the shipped `release` profile from `opt-level = "z"` to `opt-level = 3`.
This is a codegen/profile lever, not an algorithmic change. It follows the
one-lever rule from the optimization loop: improve the current shipped binary
without changing the native ASR data path or transcript semantics.

Loss matrix:

| Action | State | Loss |
| --- | --- | --- |
| Keep `opt-level = 3` | Real JFK/native workload improves and conformance stays green | Lower latency, larger binary |
| Keep `opt-level = 3` | No material gain or conformance drift | Reject/revert |
| Keep `opt-level = "z"` | Size remains smaller but short-clip native latency stays slower | User-visible latency loss |

Fallback trigger: revert this profile change if package-scoped conformance fails
or if same-target repeat measurements show less than 3% median improvement on
the JFK native release workload.

### Fresh measurements

Comparator remains the loaded OpenAI Whisper Python API from the previous
section:

```text
OpenAI Whisper API median: 0.4356057639233768 s
```

| Workload | Franken path | Median wall time | Ratio vs previous shipped release | Ratio vs OpenAI loaded API | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| 11 s JFK, `tiny.en`, CPU, one-shot CLI | shipped `release`, `opt-level = "z"`, git `656f55c`/`a79a2ae` baseline | 0.93704627044 s | 1.000000x | 0.464871x | Baseline loss |
| 11 s JFK, `tiny.en`, CPU, one-shot CLI | shipped `release`, candidate `opt-level = 3` | 0.8180240627 s | 1.145500x | 0.532510x | Kept franken-side win; still OpenAI loss |
| 11 s JFK, `tiny.en`, CPU, one-shot CLI | existing `release-perf` profile probe | 0.7972604734200001 s | 1.175338x | 0.546379x | Routing/profiling probe only; not shipped |

Ratios:

```text
candidate speedup vs previous shipped release:
  0.93704627044 / 0.8180240627 = 1.145500x

candidate ratio vs loaded OpenAI Whisper API:
  0.4356057639233768 / 0.8180240627 = 0.532510x

inverse:
  0.8180240627 / 0.4356057639233768 = 1.877900x slower than loaded OpenAI API
```

Command evidence:

```text
base build:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
    rch exec -- cargo build --profile release -p franken_whisper \
    --bin franken_whisper
  result: pass; rch worker ovh-a; release build 3m30s

candidate build:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
    rch exec -- cargo build --profile release -p franken_whisper \
    --bin franken_whisper
  result: remote compile succeeded on vmi1152480 in 7m57s; artifact retrieval
  returned RCH-E309/exit 102 after the binary had been retrieved locally.
  smoke/bench binary:
    /data/projects/.rch-targets/franken_whisper-cod-b/release/franken_whisper
    size: 16346328 bytes
    mtime: 2026-06-24 18:54:41 -0400

candidate validation rebuild:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
    rch exec -- cargo build --profile release -p franken_whisper \
    --bin franken_whisper
  result: pass; rch worker ovh-a; release build 4m45s

candidate command:
  RUST_LOG=error \
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
  FRANKEN_WHISPER_NATIVE_EXECUTION=1 \
  FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole \
  FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG=0 \
  /data/projects/.rch-targets/franken_whisper-cod-b/release/franken_whisper \
    transcribe --input /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    --backend whisper-cpp --model tiny.en --language en --temperature 0.0 \
    --beam-size 1 --best-of 1 --no-persist --json

candidate hyperfine:
  --warmup 1 --runs 5
  mean: 0.8182363673000002 s +/- 0.004627252252512483 s
  median: 0.8180240627 s
  min/max: 0.8124394947 s / 0.8253633857 s
  raw runs: [0.8182149747, 0.8180240627, 0.8124394947,
             0.8171399187, 0.8253633857]
  user/system: 3.88823802 s / 1.26704392 s

release-perf probe:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
    rch exec -- cargo build --profile release-perf -p franken_whisper \
    --bin franken_whisper
  result: local fallback, pass, 5m30s
  median: 0.7972604734200001 s
  mean: 0.79741639482 s +/- 0.012041208883753702 s
  raw runs: [0.80969776242, 0.7924667834200001, 0.8075918634200001,
             0.78006509142, 0.7972604734200001]
```

Conformance evidence:

```text
candidate transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country.

replay hashes unchanged from the previous shipped-release entry:
  input_content_hash: d16054d2df9adaae9c6228d86113f256a4b43d448d1f6b9107b75e2136a934a0
  output_payload_hash: 7fdbdde9a772bc5bda7d9933f1e475224837e8081f9505290f871316d87fd486
```

Validation after this entry:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo fmt --check -p franken_whisper
result: pass

CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo check -p franken_whisper --all-targets
result: pass; rch worker hz2

CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo clippy -p franken_whisper --all-targets -- -D warnings
result: pass; rch worker hz2

CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo test -p franken_whisper --test conformance_comparator_tests
result: pass; rch worker vmi1153651; 26 passed / 0 failed

CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
  rch exec -- cargo build --profile release -p franken_whisper \
  --bin franken_whisper
result: pass; rch worker ovh-a
```

## 2026-06-25 - IcyWren canonical WAV reuse probe

### Lever

Probe the already-normalized audio fast path: if the input is 16 kHz mono PCM16
WAV, reuse the input path instead of decoding/resampling/re-emitting a temporary
`normalized_16k_mono.wav`.

This follows the one-lever rule: only the normalization bypass is evaluated, with
the native backend, model, decoding parameters, and release profile held fixed.

Fallback trigger: reject unless same-lane five-run median improves by at least
3% and conformance remains green.

### Fresh measurements

```text
git SHA: c09920e
agent: IcyWren
target dir: /data/projects/.rch-targets/franken_whisper-cod-a
model dir: /data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models
fixture: /data/projects/franken_whisper/tests/fixtures/native/jfk.wav
fixture format: RIFF WAV, PCM16, mono, 16000 Hz

worktree scan:
  no in-repo .scratch or .worktrees directory found
  detached franken_whisper-cod-a-* worktrees were older/equivalent to main; no
  unlanded measured win found there

build:
  AGENT_NAME=IcyWren \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
    rch exec -- cargo build -p franken_whisper --profile release \
    --bin franken_whisper
  result: pass; rch local fallback; per-crate release build 2m20s
```

Current-main franken baseline from the same session, before the fast-path binary
rebuild:

```text
artifact: /tmp/franken_whisper_cod_a_current_openai_cli_hyperfine.json
franken mean: 0.7803093495200001 s
franken median: 0.7786617565200001 s
OpenAI CLI mean: 2.71806207152 s
OpenAI CLI median: 2.6709804735200002 s
```

Candidate run:

```text
artifact: /tmp/franken_whisper_cod_a_reuse_wav_openai_cli_hyperfine.json
hyperfine: --warmup 1 --runs 5

franken mean: 0.7749115707800002 s +/- 0.01518011373626203 s
franken median: 0.7709063643800002 s
franken min/max: 0.75680692938 s / 0.7983650903800001 s
franken raw runs:
  [0.7983650903800001, 0.7709063643800002, 0.7779107963800002,
   0.75680692938, 0.7705686733800001]
franken user/system: 3.5846550600000002 s / 1.2107641199999999 s

OpenAI CLI mean: 2.9077083927800005 s +/- 0.026109604708964593 s
OpenAI CLI median: 2.90920979738 s
OpenAI CLI min/max: 2.86921734038 s / 2.94291941938 s
OpenAI CLI user/system: 11.72460946 s / 0.67641032 s

same-run CLI-startup ratio:
  mean: 2.9077083927800005 / 0.7749115707800002 = 3.752310x
  median: 2.90920979738 / 0.7709063643800002 = 3.773752x

candidate speedup vs current-main franken baseline:
  mean: 0.7803093495200001 / 0.7749115707800002 = 1.006966x
  median: 0.7786617565200001 / 0.7709063643800002 = 1.010060x
```

Loaded-model OpenAI API comparator remains a loss:

```text
OpenAI API comparator artifact: inline Python run in this session
OpenAI API runs_s:
  [0.4458767760079354, 0.3950597520451993, 0.4111597209703177,
   0.41976338904350996, 0.34919446893036366]
OpenAI API mean: 0.4042108213994652 s
OpenAI API median: 0.4111597209703177 s

candidate ratio vs loaded OpenAI API:
  mean: 0.4042108213994652 / 0.7749115707800002 = 0.521622x
  median: 0.4111597209703177 / 0.7709063643800002 = 0.533346x
```

Conformance / metadata:

```text
artifact: /tmp/franken_whisper_cod_a_reuse_wav_candidate.json
normalized_wav_path:
  /data/projects/franken_whisper/tests/fixtures/native/jfk.wav
candidate transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
OpenAI API transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country
normalized lowercase alnum tokens: identical, 22/22 tokens
replay input_content_hash:
  59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e
replay output_payload_hash:
  79b7c1d5acb4a89217485af436be5bdb1b222aea11b85b778b51dd8a76ddb229
normalize stage service_ms: 20
backend stage service_ms: 711
```

### Verdict

Rejected as a product-speed lever. The fast path fires and preserves transcript
semantics, but the measured median e2e gain is only 1.006966x to 1.010060x
against current main, below the 3% keep threshold and inside host noise. The
same-run OpenAI CLI-startup path remains a fresh 3.77x win, while the stricter
loaded-model API comparator remains a 0.533x loss.

Graveyard / artifact mapping used for the decision:

- `alien_cs_graveyard.md` section 0.10: require statistical honesty, raw
  distributions, and a practical effect threshold before keeping a lever.
- `alien_cs_graveyard.md` section 15.7: zygote/COW model preloading remains a
  future architecture-sized lever for the loaded-model API gap, not a safe
  same-turn patch.
- `alien-artifact-coding`: explicit loss matrix, confidence threshold, and
  conservative fallback trigger.

Validation after this entry:

```text
git diff --check -- docs/NEGATIVE_EVIDENCE.md src/audio.rs .beads/issues.jsonl
result: pass

rustfmt --edition 2024 --check src/audio.rs
result: pass

cargo fmt --check
result: blocked by pre-existing unrelated formatting drift in
  src/native_engine/mel.rs
  src/native_engine/nn.rs

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo check -p franken_whisper --all-targets
result: pass; rch remote hz2

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo clippy -p franken_whisper --all-targets -- -D warnings
result: pass; rch remote hz2

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo test -p franken_whisper --test conformance_comparator_tests
result: pass; local fallback; 26 passed / 0 failed

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  rch exec -- cargo test -p franken_whisper audio::tests::normalize
result: pass; rch remote vmi1149989; 14 passed / 0 failed

UBS docs/NEGATIVE_EVIDENCE.md src/audio.rs .beads/issues.jsonl
result: non-zero due existing broad scanner findings in src/audio.rs; no new
  docs finding. The Rust compile/clippy/test gates above are green.

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
  rch exec -- cargo bench --profile release -p franken_whisper \
  --bench native_engine_bench -- --sample-size 10 --warm-up-time 0.1 \
  --measurement-time 0.1 native_engine/e2e/e2e_tiny_jfk
result: cancelled as non-decisive infrastructure-stale RCH probe after
  build 29902804231389480 sat active on vmi1153651 for >22m with hook
  heartbeats fresh but progress stale for ~10m. Product-level hyperfine/API
  measurements above are the deciding evidence for this lever.
```

Cod-b restart confirmation:

```text
target dir: /data/projects/.rch-targets/franken_whisper-cod-b
baseline artifact: /tmp/franken_whisper_bold_baseline_openai_cli.json
candidate artifact: /tmp/franken_whisper_bold_candidate_openai_cli.json

baseline franken mean/median: 0.76559047432 s / 0.76678070072 s
baseline OpenAI CLI mean/median: 2.91310054872 s / 2.90075806972 s
baseline OpenAI CLI-startup ratio:
  mean: 3.805038x
  median: 3.783035x

candidate franken mean/median: 0.7569894835400002 s / 0.7626821207400001 s
candidate OpenAI CLI mean/median: 2.6945086197399997 s / 2.69349702974 s
candidate OpenAI CLI-startup ratio:
  mean: 3.559506x
  median: 3.531612x

candidate speedup vs cod-b current-main franken baseline:
  mean: 1.011362x
  median: 1.005374x

build notes:
  stale remote build 29902804231389477 on vmi1167313 was cancelled.
  stale remote retry 29902804231389526 on vmi1264463 was cancelled.
  direct per-crate local build with CARGO_TARGET_DIR cod-b passed in 2m10s.

verdict:
  source fast path reverted. The cod-b restart confirms the same conclusion:
  this is a ~0-gain/no-ship product-speed lever, below the 3% keep threshold.
```

## 2026-06-25 - IcyWren current-main OpenAI Whisper bold-verify

### Worktree scan

No measured win was found in a detached worktree that is missing from current
`main`.

```text
current main: fb2ca46
AGENT_NAME: IcyWren

/data/projects/franken_whisper-cod-a-clean:
  HEAD 2ef3fa8 docs: record bold-verify validation evidence
  ancestor of main: yes

/data/projects/franken_whisper-cod-a-ledger:
  HEAD df99f60 Record OpenAI Whisper head-to-head ratio
  ancestor of main: yes

/data/projects/franken_whisper-cod-a-main-measure:
  HEAD 766f5f1 Record OpenAI ratio after mel twiddle
  ancestor of main: no
  verdict: stale measurement branch, not a landable win. The same ratio was
    already recorded on main; its diff would remove current docs/benches and
    roll back PERF_LEDGER content.

/data/projects/franken_whisper-cod-a-push:
  HEAD 5a42ed4 Record OpenAI ratio after mel twiddle
  ancestor of main: yes

/data/projects/franken_whisper-cod-b-land:
  HEAD 866760c perf: speed up shipped release profile
  ancestor of main: yes
```

### Fresh CLI-startup comparator

`speed_ratio = OpenAI Whisper wall time / franken_whisper wall time`.

| Workload | Franken path | Original path | Mean ratio | Median ratio | Conformance | Verdict |
| --- | --- | --- | ---: | ---: | --- | --- |
| 11 s JFK, `tiny.en`, CPU, 8 threads, one-shot CLI | `franken_whisper` native `whisper.cpp-native`, shipped `release`, ggml `tiny.en`, commit `fb2ca46` | OpenAI Whisper `openai-whisper==20250625`, PyTorch CPU, `tiny.en`, CLI startup each run | 4.142x | 4.161x | Normalized word tokens match the prior JFK comparator; punctuation differs | Fresh measured current-main win |

Command evidence:

```text
build:
  AGENT_NAME=IcyWren \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
    rch exec -- cargo build -p franken_whisper --profile release \
    --bin franken_whisper --example native_ab
  result: cancelled stale RCH build 29902804231389571 after hook progress went
    stale on vmi1227854; no benchmark result used from that build.

fallback build:
  AGENT_NAME=IcyWren \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
    cargo build -p franken_whisper --profile release \
    --bin franken_whisper --example native_ab
  result: pass; local per-crate fallback; 7m03s

hyperfine:
  artifact: /tmp/franken_whisper_cod_a_current_main_openai_cli_20260625.json
  --warmup 1 --runs 5

franken command:
  RUST_LOG=error \
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
  FRANKEN_WHISPER_NATIVE_EXECUTION=1 \
  FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE=sole \
  FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG=0 \
  /data/projects/.rch-targets/franken_whisper-cod-a/release/franken_whisper \
    transcribe --input /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    --backend whisper-cpp --model tiny.en --language en --temperature 0.0 \
    --beam-size 1 --best-of 1 --threads 8 --no-persist --json >/dev/null

OpenAI Whisper command:
  out=/tmp/franken_whisper_cod_a_openai_cli_$(date +%s%N); mkdir -p "$out"; \
  PATH=/home/ubuntu/.local/state/franken_whisper/tools/ffmpeg/bin:$PATH \
  uvx --from openai-whisper whisper \
    /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    --model tiny.en --language en --device cpu --fp16 False --threads 8 \
    --output_format json --output_dir "$out" --verbose False >/dev/null

franken mean/median:
  0.7824998565600001 s / 0.77482173956 s
OpenAI CLI mean/median:
  3.2414650167600003 s / 3.22429294756 s
ratio mean/median:
  4.142448065115336x / 4.161335159995623x
```

### New lever dug and rejected: loaded-model thread-count surface

The radical lever from `alien_cs_graveyard.md` section 15.7 is zygote/COW or
daemon-style model preloading: remove CLI startup/model-load from the hot path
and hold initialized model state across requests. This pass tested the nearest
existing in-crate surface for that idea, `examples/native_ab.rs`, which loads a
model once and times repeated in-process `transcribe_samples` calls. The example
now accepts an optional `threads` argument so the strict loaded-model comparator
can test the same thread count as the product CLI without adding a new harness.

OpenAI loaded API commands loaded `tiny.en` once, performed one warmup
transcription, then timed five transcriptions with `torch.set_num_threads(N)`.
Franken loaded timings used seven `native_ab` runs and discarded run 0 as the
warmup-equivalent pass.

| Workload | Franken loaded median | OpenAI loaded median | Ratio vs OpenAI | Verdict |
| --- | ---: | ---: | ---: | --- |
| 11 s JFK, `tiny.en`, loaded model, 4 threads | 0.558630 s | 0.5165675168391317 s | 0.925x | Rejected, franken slower |
| 11 s JFK, `tiny.en`, loaded model, 8 threads | 0.575680 s | 0.4237874080426991 s | 0.736x | Rejected, franken slower |

Franken artifacts:

```text
4-thread:
  command:
    FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
      /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
      tiny.en 7 4
  times artifact:
    /tmp/franken_whisper_cod_a_native_ab_4t_serial_20260625.times
  json artifact:
    /tmp/franken_whisper_cod_a_native_ab_4t_serial_20260625.json
  runs_s:
    [0.602880, 0.557460, 0.569590, 0.599170, 0.526460, 0.532780, 0.559800]
  median_after_run0:
    0.558630 s

8-thread:
  command:
    FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
      /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
      tiny.en 7 8
  times artifact:
    /tmp/franken_whisper_cod_a_native_ab_8t_serial_20260625.times
  json artifact:
    /tmp/franken_whisper_cod_a_native_ab_8t_serial_20260625.json
  runs_s:
    [0.568100, 0.595000, 0.539010, 0.549250, 0.622530, 0.737010, 0.556360]
  median_after_run0:
    0.575680 s
```

OpenAI loaded artifacts:

```text
4-thread artifact:
  /tmp/franken_whisper_cod_a_openai_loaded_api_4t_20260625.json
4-thread runs_s:
  [0.5165675168391317, 0.5101817869581282, 0.5235818079672754,
   0.5159314800985157, 0.5502593470737338]
4-thread median:
  0.5165675168391317 s

8-thread artifact:
  /tmp/franken_whisper_cod_a_openai_loaded_api_8t_20260625.json
8-thread runs_s:
  [0.4237874080426991, 0.4046412599273026, 0.41719948407262564,
   0.4865700639784336, 0.4542978859972209]
8-thread median:
  0.4237874080426991 s
```

Conformance:

```text
franken transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
OpenAI loaded API transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country
normalized lowercase alnum tokens:
  identical, 22/22 tokens
```

Decision:

```text
keep:
  examples/native_ab.rs optional [threads] argument, because it improves the
  bd-0hnz loaded-model comparator harness and makes the tested state explicit.
  The example was also cleaned up to return Result instead of panicking on
  expected harness setup failures.

reject as performance lever:
  "use 8 threads for loaded native tiny.en JFK" is slower than 4 threads in
  this surface (0.575680 s vs 0.558630 s median after warmup) and loses to the
  loaded OpenAI API. No product default changed.

route:
  Current main still has a strong OpenAI CLI-startup win. The loaded-model gap
  remains an architecture-sized zygote/daemon or external FrankenTorch GEMM
  problem, not a same-turn franken_whisper hot-path patch.
```

Graveyard / artifact mapping used for this decision:

- `alien_cs_graveyard.md` section 0.2: opportunity gate rejects low-value
  tweaks without practical effect.
- `alien_cs_graveyard.md` section 0.3: behavior is transcript-equivalent modulo
  punctuation; no product inference path was changed.
- `alien_cs_graveyard.md` section 15.7: zygote/COW model preloading is the right
  architecture family for loaded-model API parity, but this same-turn test of
  the nearest existing surface does not produce a win.
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md`
  benchmarking rules: no single-run or mean-only claims; this entry records raw
  distributions and medians.

Validation:

```text
git diff --check -- docs/NEGATIVE_EVIDENCE.md examples/native_ab.rs
result: pass

rustfmt --edition 2024 --check examples/native_ab.rs
result: pass

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo fmt --check
result: blocked by pre-existing unrelated formatting drift in
  src/native_engine/mel.rs
  src/native_engine/nn.rs

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo build -p franken_whisper --profile release --example native_ab
result: pass; local per-crate fallback; 55.64s after final harness cleanup

final smoke:
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
    /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
    tiny.en 2 4
  result: pass; transcript matches the JFK fixture; artifact:
    /tmp/franken_whisper_cod_a_native_ab_final_smoke_20260625.json

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo check -p franken_whisper --all-targets
result: pass; 5.05s after final harness cleanup

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo clippy -p franken_whisper --all-targets -- -D warnings
result: pass; 0.39s after final harness cleanup

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo test -p franken_whisper --examples
result: pass; 3 example targets, 0 tests each

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo test -p franken_whisper --test conformance_comparator_tests
result: pass; 26 passed / 0 failed

ubs docs/NEGATIVE_EVIDENCE.md examples/native_ab.rs
result: pass; 0 critical / 0 warnings
```

## 2026-06-25 - franken_whisper-cod-b cross-attention tiny serial cutoff rejection

AGENT_NAME: IcyWren

Decision source:

- No unlanded measured worktree win was found in `.scratch/.worktrees` or the
  sibling `franken_whisper-cod-*` worktrees; current `main` already contains the
  prior OpenAI Whisper evidence and PERF_LEDGER closeout.
- New lever tested from the alien-graveyard / alien-artifact /
  extreme-optimization pass: raise decoder cross-attention head-parallel cutoff
  from `1 << 13` to `1 << 14` so tiny's `tq=1, tk≈1500, n_head=6` path stays
  serial instead of barely paying per-token `thread::scope` overhead.
- Math/output contract: scheduling-only change; per-head outputs still scatter
  into disjoint output bands, so the candidate should preserve token output.

Fresh current-main baseline:

```text
git SHA: fb2ca46
target dir: /data/projects/.rch-targets/franken_whisper-cod-b
artifact: /tmp/franken_whisper_bold_fb2ca46_openai_cli.json

franken mean/median: 0.7901267332600002 s / 0.7786809866600001 s
OpenAI CLI mean/median: 3.28170112066 s / 3.14340868066 s
ratio vs OpenAI Whisper CLI:
  mean: 4.153386x
  median: 4.036838x

baseline perf spans:
  model_parse: 68.45 ms
  model_weights: 114.50 ms
  mel: 6.49 ms
  encoder_window: 256.41 ms
  cross_kv: 42.23 ms
  decoder_prefill: 10.19 ms
  decode_loop: 265.37 ms
  backend_run: 766.41 ms
```

Candidate measurement:

```text
candidate source delta:
  src/native_engine/decoder.rs cross_attention PAR_THRESHOLD
  1 << 13 -> 1 << 14

build:
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-b \
    cargo build --profile release -p franken_whisper --bin franken_whisper
  result: pass; direct local package-scoped build; 2m39s

artifact: /tmp/franken_whisper_bold_cross_threshold_openai_cli.json
span artifact: /tmp/franken_whisper_bold_cross_threshold_span.json

franken mean/median: 0.8191999434200001 s / 0.8188211554200001 s
OpenAI CLI mean/median: 3.08047327522 s / 3.0076341314199997 s
ratio vs OpenAI Whisper CLI:
  mean: 3.760344x
  median: 3.673127x

candidate franken speedup vs current-main baseline:
  mean: 0.964510x
  median: 0.950978x

candidate perf spans:
  model_parse: 62.31 ms
  model_weights: 100.60 ms
  mel: 3.74 ms
  encoder_window: 215.93 ms
  cross_kv: 43.60 ms
  decoder_prefill: 10.48 ms
  decode_loop: 306.57 ms
  backend_run: 747.50 ms
```

Verdict:

- Rejected and source reverted. The lever makes the measured franken path slower
  by both mean and median, and the span run shows the intended target
  (`decode_loop`) regressed from 265.37 ms to 306.57 ms.
- Current-main remains a measured win versus OpenAI Whisper CLI on this
  one-shot tiny JFK workload, but this cutoff change is a no-ship lever.

## 2026-06-25 - IcyWren transcript-only loaded-model probe rejection

AGENT_NAME: IcyWren

Worktree scan:

```text
current main: 358ffa5
no in-repo .scratch/.worktrees directory found

/data/projects/franken_whisper-cod-a-clean:
  HEAD 2ef3fa8; ancestor of main: yes
/data/projects/franken_whisper-cod-a-ledger:
  HEAD df99f60; ancestor of main: yes
/data/projects/franken_whisper-cod-a-main-measure:
  HEAD 766f5f1; ancestor of main: no
  verdict: stale measurement branch. Its diff would delete current
    NEGATIVE_EVIDENCE / PERF_LEDGER content and does not contain a landable
    measured win absent from main.
/data/projects/franken_whisper-cod-a-push:
  HEAD 5a42ed4; ancestor of main: yes
/data/projects/franken_whisper-cod-b-land:
  HEAD 866760c; ancestor of main: yes
```

New lever tested:

- Probe family: transcript-only loaded-model mode, mapped to
  `alien_cs_graveyard.md` section 15.7 (preloaded model / no startup cost)
  and section 0.2/0.3 (opportunity and isomorphism gates).
- Temporary harness delta: add an optional `timestamps|no-timestamps` argument
  to `examples/native_ab.rs` so the existing loaded-model harness could run
  `DecodeParams { timestamps: false, .. }`.
- Product-surface analogue: `--no-timestamps` / transcript-only requests.
- Behavior contract: compare normalized word tokens only; timestamp spans are
  intentionally not part of this transcript-only workload.

Baseline and candidate commands:

```text
build:
  AGENT_NAME=IcyWren \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
    cargo build -p franken_whisper --profile release --example native_ab
  result: pass; package-scoped; 1m02s

timestamped baseline:
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
    /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
    tiny.en 7 4 timestamps
  times artifact:
    /tmp/franken_whisper_cod_a_native_ab_timestamps_4t_20260625b.times
  json artifact:
    /tmp/franken_whisper_cod_a_native_ab_timestamps_4t_20260625b.json

transcript-only candidate:
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
    /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
    tiny.en 7 4 no-timestamps
  times artifact:
    /tmp/franken_whisper_cod_a_native_ab_no_timestamps_4t_20260625b.times
  json artifact:
    /tmp/franken_whisper_cod_a_native_ab_no_timestamps_4t_20260625b.json

transcript-only repeat:
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
    /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
    tiny.en 9 4 no-timestamps
  times artifact:
    /tmp/franken_whisper_cod_a_native_ab_no_timestamps_4t_repeat_20260625b.times
  json artifact:
    /tmp/franken_whisper_cod_a_native_ab_no_timestamps_4t_repeat_20260625b.json

OpenAI loaded API:
  uvx --from openai-whisper python
  torch.set_num_threads(4)
  whisper.load_model("tiny.en", device="cpu")
  one warmup transcribe, then 5 timed transcribes
  artifact:
    /tmp/franken_whisper_cod_a_openai_loaded_api_4t_20260625b.json
```

Measured distributions:

```text
timestamped baseline runs_s:
  [0.543480, 0.521640, 0.504980, 0.510860, 0.505800, 0.508430, 0.505710]
timestamped median_after_run0:
  0.507115 s

transcript-only candidate runs_s:
  [0.516090, 0.499610, 0.487720, 0.496420, 0.491980, 0.489780, 0.484760]
transcript-only candidate median_after_run0:
  0.490880 s

transcript-only repeat runs_s:
  [0.526350, 0.496650, 0.507200, 0.492290, 0.490020, 0.502070, 0.498060,
   0.476220, 0.485440]
transcript-only repeat median_after_run0:
  0.494470 s

OpenAI loaded 4-thread runs_s:
  [0.51801612903364, 0.5054755490273237, 0.4925481260288507,
   0.5291743979323655, 0.49697991902939975]
OpenAI loaded 4-thread median:
  0.5054755490273237 s

candidate speedup vs timestamped baseline:
  first pass median: 1.033073x
  repeat median:     1.025573x

candidate ratio vs OpenAI loaded API:
  first pass median: 1.029733x
  repeat median:     1.022257x
```

Conformance:

```text
timestamped franken transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
transcript-only franken transcript:
  And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
OpenAI loaded API transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country
normalized lowercase alnum tokens:
  identical, 22/22 tokens
```

Verdict:

- Rejected as a product speed lever. The first pass barely clears a 3% local
  timestamped-baseline gain, but the repeat falls to 2.56%, and the fresh
  loaded-OpenAI ratio is only 1.02-1.03x. That is too close to host noise for a
  BOLD-VERIFY keep.
- Temporary source delta fully reverted before commit; `git diff --
  examples/native_ab.rs` is empty.
- The measured loaded-model gap remains routed to architecture-sized
  preloaded-service/zygote work or the external FrankenTorch GEMM bead
  (`bd-4hc0`), not a no-timestamps default change.

Validation:

```text
git diff --check -- docs/NEGATIVE_EVIDENCE.md examples/native_ab.rs
result: pass

rustfmt --edition 2024 --check examples/native_ab.rs
result: pass

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo check -p franken_whisper --all-targets
result: pass; 1m02s; existing dead_code warnings in src/native_engine/decode.rs
  for process_logits and argmax

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo test -p franken_whisper --test conformance_comparator_tests
result: pass; 26 passed / 0 failed

ubs docs/NEGATIVE_EVIDENCE.md examples/native_ab.rs
result: pass; Rust scanner ran on examples/native_ab.rs; 0 critical /
  0 warnings
```

## 2026-06-25 - IcyWren greedy logprob allocation probe rejection and revert

AGENT_NAME: IcyWren

Scope:

- Bead: `bd-0hnz`, BOLD-VERIFY head-to-head evidence loop.
- Commit measured then reverted: `3cbd80e42346017ec8246b2bfcf048ee9ef9ccfe`
  (`perf(native): avoid greedy logprob allocation`).
- Candidate family: greedy decode logit-filter specialization. The intended
  lever was to avoid allocating the full `logprobs` vector during greedy
  sampling and keep the full log-probability path only for tests.
- Alien mapping: `alien_cs_graveyard.md` section 0.1/0.2/0.10
  (profile-first, opportunity gate, statistical benchmarking) plus
  `alien-artifact-coding` algebraic-preservation check. The force-timestamp
  decision appears algebraically equivalent because the global logsumexp term
  cancels in `logsumexp(timestamp logits) > max(text logits)`, but the measured
  runtime did not clear the keep gate.

Worktree scan:

```text
current main during probe advanced from 15b03e7 to 3cbd80e while measurement
was in flight; 3cbd80e was already pushed to origin/main and origin/master.

/data/projects/franken_whisper-cod-a-clean:
  HEAD 2ef3fa8; ancestor of main: yes
/data/projects/franken_whisper-cod-a-ledger:
  HEAD df99f60; ancestor of main: yes
/data/projects/franken_whisper-cod-a-main-measure:
  HEAD 766f5f1; divergent stale measurement branch; its diff would delete
  current ledger/code and is not a clean landable win
/data/projects/franken_whisper-cod-a-push:
  HEAD 5a42ed4; ancestor of main: yes
/data/projects/franken_whisper-cod-b-land:
  HEAD 866760c; ancestor of main: yes
```

Baseline and candidate commands:

```text
clean baseline worktree:
  git worktree add --detach \
    /data/projects/franken_whisper-cod-a-baseline-15b03e7 15b03e7
  cp /data/projects/franken_whisper/tests/fixtures/native/jfk.wav \
    /data/projects/franken_whisper-cod-a-baseline-15b03e7/tests/fixtures/native/jfk.wav

baseline build:
  AGENT_NAME=IcyWren \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
    cargo build -p franken_whisper --profile release --example native_ab
  workdir: /data/projects/franken_whisper-cod-a-baseline-15b03e7
  result: pass; package-scoped; 1m04s

baseline run:
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
    /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
    tiny.en 11 4
  times artifact:
    /tmp/franken_whisper_cod_a_baseline_15b03e7_greedy_4t_20260625c.times
  json artifact:
    /tmp/franken_whisper_cod_a_baseline_15b03e7_greedy_4t_20260625c.json

candidate build:
  AGENT_NAME=IcyWren \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  RUSTFLAGS='--cfg franken_whisper_cod_a_candidate' \
    cargo build -p franken_whisper --profile release --example native_ab
  workdir: /data/projects/franken_whisper
  result: pass; package-scoped; 4m48s

candidate run:
  FRANKEN_WHISPER_MODEL_DIR=/data/projects/franken_whisper/legacy_whispercpp/whisper.cpp/models \
    /data/projects/.rch-targets/franken_whisper-cod-a/release/examples/native_ab \
    tiny.en 11 4
  times artifact:
    /tmp/franken_whisper_cod_a_candidate_greedy_logits_4t_20260625c.times
  json artifact:
    /tmp/franken_whisper_cod_a_candidate_greedy_logits_4t_20260625c.json
```

OpenAI Whisper comparator:

```text
command:
  uvx --from openai-whisper python
  torch.set_num_threads(4)
  load jfk.wav as mono 16 kHz NumPy array via Python wave module
  whisper.load_model("tiny.en", device="cpu")
  one warmup transcribe, then 7 timed transcribes

note:
  The file-path OpenAI run failed before timing because ffmpeg is not installed
  on PATH in this environment. Passing the already-normalized WAV samples as a
  NumPy array avoids ffmpeg and measures the loaded-model inference surface.

artifact:
  /tmp/franken_whisper_cod_a_openai_loaded_api_4t_greedy_probe_20260625c.json
```

Measured distributions:

```text
clean baseline runs_s:
  [0.543580, 0.517000, 0.517820, 0.516430, 0.513130, 0.524670,
   0.520970, 0.510800, 0.524850, 0.519030, 0.522010]
clean baseline median_after_run0:
  0.518425 s

candidate runs_s:
  [0.782370, 0.534470, 0.536800, 0.541730, 0.527400, 0.535830,
   0.525170, 0.530490, 0.535410, 0.526360, 0.526620]
candidate median_after_run0:
  0.532480 s

OpenAI loaded-array runs_s:
  [0.49046763498336077, 0.48980318498797715, 0.5516256908886135,
   0.5295493719168007, 0.5154292068909854, 0.574532015947625,
   0.6386740889865905]
OpenAI loaded-array median:
  0.5295493719168007 s

candidate speedup vs clean baseline:
  0.9736046424278846x

clean baseline ratio vs OpenAI loaded-array:
  1.0214580159459916x

candidate ratio vs OpenAI loaded-array:
  0.9944962663701936x
```

Conformance:

```text
clean baseline franken transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
candidate franken transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
OpenAI loaded-array transcript:
  And so my fellow Americans ask not what your country can do for you ask what you can do for your country
normalized lowercase alnum tokens:
  identical, 22/22 tokens
```

Verdict:

- Rejected and reverted. The candidate is a measured regression versus clean
  `15b03e7` (`0.9736x`) and effectively tied/slower versus the fresh loaded
  OpenAI Whisper comparator (`0.9945x`).
- This is below the BOLD keep threshold and below the normal noise-safe
  threshold for an in-crate product lever.
- Revert action: `git revert --no-commit 3cbd80e42346017ec8246b2bfcf048ee9ef9ccfe`,
  followed by this corrected negative-evidence ledger entry.

Validation after revert:

```text
git diff --check -- docs/NEGATIVE_EVIDENCE.md src/native_engine/decode.rs
result: pass

rustfmt --edition 2024 --check src/native_engine/decode.rs
result: pass

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo check -p franken_whisper --all-targets
result: pass

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo clippy -p franken_whisper --all-targets -- -D warnings
result: pass

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo test -p franken_whisper --lib native_engine::decode::tests
result: pass; 37 passed / 0 failed

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo test -p franken_whisper --test conformance_comparator_tests
result: pass; 26 passed / 0 failed

AGENT_NAME=IcyWren \
CARGO_TARGET_DIR=/data/projects/.rch-targets/franken_whisper-cod-a \
  cargo test -p franken_whisper
result: fail; 3223 passed / 1 failed.
failing test:
  orchestrator::tests::stage_budget_timeout_maps_to_timeout_error_code
isolated rerun:
  cargo test -p franken_whisper \
    orchestrator::tests::stage_budget_timeout_maps_to_timeout_error_code
  result: same failure. This is outside the reverted decode/logit path.

cargo fmt -p franken_whisper --check
result: fail on pre-existing formatting drift in src/native_engine/mel.rs and
  src/native_engine/nn.rs, outside this revert.

ubs docs/NEGATIVE_EVIDENCE.md src/native_engine/decode.rs
result: fail; existing UBS rust security heuristics flag non-secret byte/string
  comparisons and tokenizer decode calls in decode.rs as critical.

ubs --skip-rust=8 docs/NEGATIVE_EVIDENCE.md src/native_engine/decode.rs
result: pass; 0 critical, warnings only.
```
