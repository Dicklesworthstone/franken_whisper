# Negative Evidence Ledger

This ledger records blocked, neutral, rejected, or non-comparable performance
evidence. It exists to prevent stale optimism from being reused as proof.

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
