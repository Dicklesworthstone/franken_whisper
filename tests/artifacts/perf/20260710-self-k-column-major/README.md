# Packed self-K column layout — PARKED, not rejected

Status: **analysis-complete, unmeasured candidate**. The patch was not kept in
the working tree because strict RCH could not finish syncing this repository to
the selected remote worker. No local Cargo fallback was allowed. This artifact
must not be cited as a performance rejection.

## Profile routing

Workload: full timestamped `large-v3-turbo` transcription of the dense
124.5-second track01 fixture, `RAYON_NUM_THREADS=8`, existing symbolized
`release-perf` `e2e_probe`. The transcription took 23.329 seconds (RTF 0.1874,
12 segments, 1,337 characters). Flat sampling captured 32K transcribe-window
samples with **zero lost samples**.

The executable was built at source `91b44b1`. The requested in-crate mel,
tokenizer, decoder, and `nn.rs` paths are unchanged through the profiled HEAD;
the sibling `frankentorch` revision advanced, so sibling-frame magnitudes are
routing context rather than fresh comparator claims.

External sgemm frames are intentionally omitted below.

| self | frame | routing |
|---:|---|---|
| 21.67% | `nn::dot_maddubs_i7_m2n4` | cc-owned int8 |
| 14.34% | `nn::matmul_bias_i7_quantized` closure | cc-owned int8 |
| 13.08% | `ft_kernel_cpu::sdpa_forward_f32` | cc-owned SDPA |
| 7.53% | `__expf_fma` | cc-owned SDPA |
| 6.03% | `nn::gemv_i8` closure | cc-owned int8 |
| 4.63% | `encoder::matmul_bias_i8` closure | cc-owned int8 |
| 1.68% | `nn::gemv_i8w_f32a_blocked` | cc-owned int8 |
| 1.39% | `nn::quantize_act_i7_gelu` closure | cc-owned int8 |
| 1.07% | `nn::gemv_i8` | cc-owned int8 |
| 0.78% | `nn::norm_rows_into` | prior decoder-fused-LN row |
| 0.74% | `nn::maddubs_i7_headmajor_block` | cc-owned int8 |
| 0.69% | `__memset_avx2_unaligned_erms` | mixed call sites; not KV-attributed |
| 0.65% | `nn::quantize_act_i7` closure | cc-owned int8 |
| 0.39% | `__memmove_avx_unaligned_erms` | mixed call sites |
| 0.29% | encoder quantization closure | cc-owned int8 |
| 0.20% | unresolved kernel address | outside crate |
| 0.19% | unresolved kernel address | outside crate |
| 0.17% | `encoder::forward_time_major` | outside decoder lane |
| **0.17%** | **`nn::attention_with_cache`** | **top open requested family** |
| 0.14% | `DecoderState::new` closure 4 | cross-K/V f16 conversion |
| 0.11% | `nn::softmax_rows` | decoder attention |

The full-process capture contained 38,308 samples; the table uses the exact
transcription time slice `2428346.252,2428369.586`.

## Integrity correction

The historical byte-exact self-K score-loop rejection used
`examples/self_attn_scores_probe.rs`. That file contains private replicas named
`scores_scalar` and `scores_swap`; it never calls production
`attention_with_cache` or `attention_decode_step`, and it recorded no production
function self-time. Under the active ledger-integrity rule that REJECT is
invalid.

Fresh `perf annotate` on the production symbol positively establishes the
mechanism. `attention_with_cache` has 0.17% full-transcription self-time; within
that symbol the scalar score chain's two sampled instructions account for
52.87% local period (`vmulss` 40.71% + `vaddss` 12.16%), or approximately 0.09%
of full transcription. The already-vectorized score-times-V AXPY accounts for
the next sampled vector additions. This is a real path, not a replica.

## Candidate

The rejected replica swapped loops over the existing token-major K cache and
therefore read one key every 5,120 bytes. The parked candidate changes the
primitive instead:

- retain the historical `[token, state]` key cache for prefill and parity;
- mirror K as `[state, capacity_tokens]` for single-token decode;
- append each new key to both layouts;
- compute scores d-outer/j-inner over contiguous key columns;
- preserve, for every `scores[j]`, the exact d-ascending sequence of
  `score += qd * (key * scale)` operations.

Independent scores may therefore be vectorized across `j` without changing a
floating-point reduction. The candidate also measures its real costs: the
extra append scatter, a second K allocation, and the packed score kernel. It is
structurally different from the strided loop-swap replica.

The patch adds an explicit candidate constructor while leaving `KvCache::new`
on the historical layout. The Criterion group:

- calls real `attention_with_cache` for both arms;
- asserts output bits are identical before timing;
- preloads both caches to 223 of 224 turbo text tokens;
- alternates arm order for 25 paired repetitions;
- reports the paired-ratio CV and win count;
- consumes each arm's full output slice through `black_box`;
- registers both real arms for profiler reachability.

No timing result exists. Even an infinitely fast score chain can remove at most
about 0.09% of this workload, before charging packed-cache append and memory
costs, so promotion would require unusually strong evidence on a decoder-heavy
workload. That bound does not reject the primitive.

## Strict-RCH blocker

Required command:

```text
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench \
  --profile release-perf --bench native_engine_bench -- \
  native_engine/self_attn_k_layout --noplot
```

RCH selected healthy worker `vmi1264463`, prepared 26 dependency roots, and
then failed closed:

```text
sync_to_remote: timed out after 30000ms
[RCH] local (remote execution failed)
[RCH] remote required; refusing local fallback (remote execution failed)
```

No local Cargo or rustc process was spawned. The same repository-sync blocker
reproduced independently in the concurrent f16-prefill lane.

## Reproduction artifacts

```text
candidate patch:
  tests/artifacts/perf/20260710-self-k-column-major/candidate.patch

flat profile:
  /tmp/fw-cod-fw-integrity-track01-flat-20260710.data
  sha256 15a513d12bef45766eca5d13c9ef61bf15d7b7089524e0f46fa17bb408db8341

DWARF callgraph profile:
  /tmp/fw-cod-fw-integrity-track01-callgraph-20260710.data
  sha256 0f6ddb17e673eb86c47fee7df83431e6837de4e60148d436ac3d3de6a73d2a41

profiled executable:
  /data/tmp/cargo-target/release-perf/examples/e2e_probe
  Build ID acd75e8eb9b593d129a8563461349529921d46ef
  sha256 272102fd7cd643bf449eeed18002874cc98241f74290d2937a8d606a10b0c776

audio:
  /tmp/fw-cod-fw-track01-20260710/track01.wav
  sha256 a21dcd888ae070381189e869e54de39c66fc65f1b9ad50a54a8cf14369930e9e
```

Retry only when strict RCH can complete project sync. Apply the patch, run the
single-invocation paired benchmark above, then profile the retrieved benchmark
binary and record non-zero self-time for
`attention_decode_step_column_keys`. Without that self-time, neither WIN nor
REJECT is admissible.
