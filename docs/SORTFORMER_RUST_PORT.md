# Native Streaming Sortformer Rust Port Contract

Status: Phase -1 truth pack for `bd-y4ip.9`  
Fit verdict: **CONDITIONAL GO**  
Promotion authority: none; this document does not enable a runtime route

## 1. Decision and no-claim boundary

The full end-to-end NVIDIA Streaming Sortformer v2.1 graph is the primary
learned-model candidate for a native Rust port. It directly owns speech
segmentation, overlap, speaker change, speaker activity, and speaker-count
behavior up to four speakers. A speaker encoder such as ReDimNet or ECAPA does
not replace those stages and remains a compact hybrid or low-resource ablation.

The existing native acoustic diarizer remains the production incumbent,
low-memory baseline, deterministic capacity fallback, and regression oracle.
The operator-installed NeMo adapter remains an external reference only. No
native Sortformer route, automatic routing change, accuracy claim, or speed
claim is authorized until the ordered parity and public evaluation gates below
pass.

The port is a conditional fit rather than an unconditional fit because:

- the 117.7-million-parameter graph is a tier-2 port with a substantial
  FastConformer front end and stateful streaming overlay;
- the custom speaker-cache compression, top-k selection, and FIFO update
  semantics still require seam fixtures before implementation;
- the custom NVIDIA Open Model License requires an explicit distribution
  policy; and
- the current one-record nondeterminism pilot is not a complete oracle floor.

No existing native diarization code should be deleted to make room for this
route. A failed parity, resource, capacity, or public-accuracy gate leaves the
current production behavior unchanged.

## 2. Immutable upstream and runtime pins

| Item | Frozen identity |
|---|---|
| Model | `nvidia/diar_streaming_sortformer_4spk-v2.1` |
| Hugging Face revision | `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d` |
| `.nemo` bytes | `471367680` |
| `.nemo` SHA-256 | `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8` |
| `model_config.yaml` SHA-256 | `2865d469c4d2aac54aa5b8a956b2423c053806dd20d5bf5d08675942a1acface` |
| `model_weights.ckpt` bytes | `471352898` |
| `model_weights.ckpt` SHA-256 | `eca9773c2dab91dd41fbaa4473cebb9d00811d67788ce2de609dadc6e499cdf4` |
| Canonical 990-entry state inventory SHA-256 | `f4f219cf4ac6f755247b56d19e425db3d6a7c23c4509176549b363b63abdf532` |
| NeMo source revision | `40ace43c7cf151af78dc22027c02feeca7e06b6a` |
| Python | `3.12.12` |
| NeMo package | `3.1.0+40ace43c7c` |
| PyTorch / torchaudio | `2.7.1` / `2.7.1` |
| NumPy | `2.4.6` |
| External model-contract SHA-256 | `7ac048e3372fe4c622840beddfbeef42944d961408360324cb7276a69c8542c5` |
| Qualified runtime-fingerprint SHA-256 | `3713fd3f024c1cef7d860706baf0dbaaf18058c03c26331da6254687693d564c` |

The archive configuration says `nemo_version: 2.6.0rc0`; that string is not
the executable reference. The accepted oracle behavior is the identity-bound
NeMo source revision and package/runtime set above. A future source or package
change creates a new evidence row and cannot silently inherit these tolerances.

The relevant installed source files at the pinned revision were independently
hashed:

| Source file | SHA-256 |
|---|---|
| `models/sortformer_diar_models.py` | `4978dba1a02b414893123f66905a1e523d5bb65766903269b325746c67f6920a` |
| `modules/sortformer_modules.py` | `3d136c245e3bf7a88c47fdd2eae1edb9189bbeddc3ff779cb5679a29d890b7eb` |
| `modules/conformer_encoder.py` | `a8b6f712cdf75a3be768848e8242ea9412ca7ff31ba2dda6b9602bcefc627cec` |
| `modules/transformer/transformer_encoders.py` | `a2859c86c8389f1954d5c8be04dc2bc422452517ef15e069cf42bfab5d304759` |
| `modules/transformer/transformer_modules.py` | `2564d95365cfafd486b1a3d10e2e2f438702907076f3716dd4c42d568b3bcc72` |
| `modules/audio_preprocessing.py` | `c061f521e14978d22ad57fa5ddf08f1103c2d1f1a4e01aca6698bfad007e8e7c` |
| `parts/preprocessing/features.py` | `4290ed2d697362a68a6158fb8b7b8d1e2306b223b83172c63fc6b5d31b28ee69` |
| `parts/submodules/conformer_modules.py` | `99bb846c51db028d6d30b3d844af22826068aeaa0e48eb586489a31a9cbacf9d` |
| `parts/submodules/multi_head_attention.py` | `4999fd0d679fd7315ba275f7311fe6608c48e492bd337f2e220c99b8b9729c69` |
| `parts/submodules/subsampling.py` | `4fbc689f3f66e4630b286196315a02b315ad53e8049c164fe40dd11168cf0834` |
| `parts/submodules/causal_convs.py` | `7cf505c8caef44a37a7dec10b51eb2d60ec2f1efc3a2badc3c20c37e427cbd42` |
| `parts/utils/vad_utils.py` | `7beb57efff5e08407f9f16afe9c0da7d0e2ddb9bd62e2a37424693e48c5f0437` |
| `parts/mixins/diarization.py` | `5365e416ecab192cf59f1b9d6554ebce0ed3bdb2fee7575966ac1e3fca1a1408` |
| `modules/transformer/transformer_utils.py` | `0af8d73b225bc3ebd58541a1e1cece468e79aac42590836803a8cbc765da7691` |

These hashes identify the source semantics inspected for the port. They do not
prove that arbitrary Python code or model bytes are trustworthy.

The current oracle hard-pins only Python, NeMo, PyTorch, torchaudio, and NumPy.
The observed transitive runtime also includes librosa 0.11.0, Lhotse 1.33.0,
SoundFile 0.14.0, SciPy 1.18.0, OmegaConf 2.3.0, Hydra 1.3.2, and Lightning
2.4.0. `bd-y4ip.10` must either bind all packages that influence an accepted
seam or eliminate them from seam authority by exporting the exact intermediate
tensors. A repeated version document alone does not close that gap.

## 3. License and artifact-distribution boundary

The model is under the custom
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
not the repository's Rust source license. The model card and license must be
reviewed again before distributing any derived weights.

The initial Rust route therefore uses this conservative policy:

- no original or converted weights are committed to Git, embedded in the
  binary, attached to a release, or copied into test fixtures;
- the operator obtains the model under the upstream license and performs a
  local, identity-bound conversion;
- the converter emits a non-executable safetensors package plus a canonical
  conversion receipt outside the repository;
- runtime loading never deserializes the pickle-based `.ckpt` from the `.nemo`
  archive; and
- any future downloader or redistributor requires an explicit license and
  Notice review as its own gate.

This is a repository policy, not legal advice and not a claim that every future
form of redistribution is permitted.

## 4. Model and state census

Pinned live-model introspection produced:

| Census | Value |
|---|---:|
| Trainable parameters | 117,693,960 |
| Parameter tensors | 937 |
| State tensors | 990 |
| State elements | 117,744,681 |
| Float32 state tensors | 973 |
| Int64 state tensors | 17 |
| Float32 state elements | 117,744,664 |
| Float32 state bytes | 470,978,656 |
| Total state payload bytes | 470,978,792 |
| Conformer layers | 17 |
| Transformer blocks | 18 |
| Linear modules | 266 |
| Layer-normalization modules | 121 |
| Dropout modules | 124 |
| Conv1d modules | 34 |
| Conv2d modules | 5 |
| BatchNorm1d modules | 17 |

Approximate element ownership from the same introspection is 109,565,969
encoder elements, 8,007,552 transformer elements, 137,864 Sortformer-module
elements, and 33,296 preprocessor elements. The f32 weights alone are about
449 MiB. Peak process RSS and activation high-water marks are not inferred
from that number; they remain measurement gates.

The encoder also regenerates a non-persistent positional buffer
`encoder.pos_enc.pe` with shape `[1, 9999, 512]`: 5,119,488 f32 values, about
19.5 MiB. It is absent from `state_dict`. The converted package must either
export its exact values or bind and test a deterministic Rust regeneration
algorithm; silently omitting it from the census is not acceptable.

The conversion receipt must enumerate every tensor and record, at minimum:

- source key and destination key;
- source and destination dtype;
- source and destination shape and logical layout;
- transpose, reshape, split, concatenate, or cast transform;
- source-value and destination-value SHA-256;
- byte length and element count; and
- the pinned model, source, config, converter, and receipt-schema identities.

The loader must reject missing tensors, unexpected tensors, duplicate names,
wrong shapes, unsupported dtypes, non-finite materialization, checksum drift,
and unknown package-schema versions. A matching total byte count is not
sufficient.

## 5. Frozen forward graph

### 5.1 Input and frontend

- mono 16 kHz finite PCM;
- signed 16-bit WAV decoding maps each sample to `sample / 32768.0`;
- pre-emphasis preserves the first sample and then uses
  `x[t] - 0.97 * x[t - 1]`;
- constant center padding for STFT;
- 400-sample non-periodic Hann window centered in a 512-point FFT, 160-sample
  hop, 257 one-sided bins, and squared magnitude;
- 128 log-mel features, frame splicing 1, no feature normalization;
- the exact stored Slaney mel buffer with shape `[1, 128, 257]`;
- natural logarithm after adding `2^-24`;
- archive dither value `1e-5`, but diarization evaluation sets dither to zero
  and `pad_to` to zero; and
- batch size one, CPU float32, no autocast, no quantization.

The first oracle seam is the exact framed log-mel tensor and its valid length.
The effective valid-frame count is `floor(samples / 160)`: centered STFT
creates one extra frame, but NeMo marks it invalid and the model crops to the
declared length. The existing Whisper frontend is not reusable as-is because
its window periodicity, padding, transform length, logarithm, clamping, and
normalization semantics differ. Every frontend fact must still be captured by
numeric fixtures rather than accepted from source inspection alone.

### 5.2 FastConformer encoder

- depthwise-striding subsampling by 8 with 256 convolution channels;
- 17 layers, model width 512, eight attention heads;
- feed-forward expansion factor 4;
- relative-position attention with x-scaling and untied biases;
- convolution kernel 9 with batch normalization; and
- evaluation-mode dropout and stochastic-depth behavior.

Subsampling is three symmetric stride-2 stages: a `1 -> 256` 3-by-3 Conv2d,
then two grouped depthwise 3-by-3 stages with pointwise `256 -> 256`
convolutions and ReLU after each stage. The result flattens `256 * 16 = 4096`
coordinates before the `4096 -> 512` affine.

Each Conformer block has two half-residual feed-forward modules,
Transformer-XL relative attention, and a Swish convolution module. Despite its
class name, the kernel-9 `CausalConv1D` receives integer padding 4 and is
symmetric four-left/four-right in this graph. Relative attention uses learned
`pos_bias_u` and `pos_bias_v`, a bias-free positional affine, relative shift,
and division by `sqrt(64)`. The pinned graph does not use PyTorch SDPA.

The port must reproduce the NeMo ordering of feed-forward, attention,
convolution, residual, scaling, and normalization operations. Existing Whisper
attention helpers are not accepted as evidence of equivalent conventions.

### 5.3 Projection, transformer, and speaker head

- learned projection from 512 FastConformer coordinates to width 192;
- 18 transformer blocks at width 192, inner width 768, eight heads;
- ReLU feed-forward activation;
- post-layer-normalization inside each block, with no final transformer layer
  normalization operation; and
- four sigmoid speaker-activity outputs.

Transformer attention uses eight 24-dimensional heads, divides both Q and K
by `sqrt(sqrt(24))`, and adds `-10000` for padded positions rather than
negative infinity. The inference head is exactly ReLU, evaluation-identity
dropout, `192 -> 192`, ReLU, evaluation-identity dropout, `192 -> 4`, then
sigmoid. The checkpoint's `384 -> 4` `hidden_to_spks` tensor is dead in the
accepted inference path. It remains in the complete conversion receipt but
must not be wired into the Rust graph by name intuition.

### 5.4 Streaming state machine

The accepted high-latency profile is the canonical contract already frozen in
`src/differential_oracle.rs`:

| Quantity | Output-frame count |
|---|---:|
| Chunk | 340 |
| Left context | 1 |
| Right context | 40 |
| FIFO capacity | 40 |
| Configured cache update | 300 |
| Speaker cache | 188 |
| Silence cache frames per speaker | 3 |

The nominal input-buffer latency is 30.4 seconds and the output stride is 80
ms. The pinned validator warns about the configured 300-frame update being
shorter than the chunk but does not rewrite it. FIFO movement is:

```text
min(max(configured_update, chunk - fifo_capacity + current_fifo),
    current_fifo + chunk)
```

This moves 300 frames for the first full chunk with an empty FIFO, leaves 40
queued, immediately compresses the 300-frame cache to 188, and moves 340 frames
in steady state. An interior chunk includes 381 pre-encoding frames; with a
full cache and FIFO, the recurrent encoder sequence can reach
`188 + 40 + 381 = 609` frames. Tail behavior is a required seam, not an
extrapolation.

The Rust state must explicitly own speaker-cache embeddings and predictions,
their lengths, FIFO embeddings and predictions, FIFO lengths, mean-silence
embedding, silence-frame count, and the optional permutation field. Cache
compression uses prediction-derived scores, speaker-specific score boosts,
reserved silence frames, top-k selection, chronological reordering, disabled
placeholder handling, and gathering. Top-k ties and sort stability are oracle
questions until frozen fixtures answer them.

The state type includes an optional speaker-permutation field. Speaker
permutation is training-only in the pinned path. Evaluation passes
`permute_spk = false`, so the accepted inference state has no active
`spk_perm`. The Rust reference must reproduce that absence; later training or
adaptation support would be a different contract.

### 5.5 Post-processing and capacity

- at most four contiguous arrival-ordered labels `speaker_0` through
  `speaker_3`;
- each 80 ms probability is repeated eight times onto a 10 ms grid;
- onset and offset are 0.5; onset uses strict `>` and offset strict `<`, so an
  exact equality preserves the current state;
- onset/offset padding and minimum on/off durations are zero;
- an activity lane still open at the end uses the final 10 ms sample index as
  its endpoint, creating a one-frame-short source convention before adapter
  clipping and validation;
- the adapter relabels non-empty lanes by first onset and clips a terminal end
  that exceeds document duration by at most the accepted 79 ms tolerance;
- accepted turns are aligned to 80 ms, except the final end may equal document
  duration;
- overlap is represented by concurrent labeled turns, not an overlap flag; and
- speech, overlap, and speaker-change stages are derived exactly from the final
  turns.

A request or reference known to require more than four speakers is
capacity-ineligible for this model. Product routing must select the existing
native path rather than clamp, drop, or merge speakers to satisfy the model.
The model itself cannot determine that an otherwise unconstrained recording
actually contains five or more speakers: it can only activate zero through
four lanes. Unknown-capacity production inputs therefore require a separate
capacity sentinel or a candid capped-output status before this route can be
certified.

Known timestamp intervals do not change the neural forward pass during parity
work. A later product adapter may use them as hard or soft constraints when
mapping the four opaque arrival-ordered streams to caller-provided speaker
references. Contradictory hard intervals must fail closed; they must not tune
or mutate model weights during a call.

## 6. Operator inventory and FrankenTorch boundary

The port uses one Sortformer-specific CPU facade over FrankenTorch. It may
reuse general kernels only when the exact shape, layout, dtype, reduction, and
mask contract is proven at a seam.

| Operator family | Initial route | Required proof |
|---|---|---|
| Dense affine / batched GEMM | explicit CPU-only FrankenTorch matmul | scalar or oracle tensor differential |
| Conv2d and depthwise Conv2d | FrankenTorch kernels where layouts match | impulse, padding, tail, and random tensor cases |
| Conv1d / depthwise Conv1d | safe reference lowering first | exact padding, groups, dilation, and tail cases |
| Batch normalization | evaluation-mode FrankenTorch apply | epsilon, running-stat, affine, and layout parity |
| Layer normalization | FrankenTorch layer norm | axis, epsilon, and accumulation parity |
| Relative-position attention | safe model-specific composition first | Q/K/V, position bias, mask, softmax, and output seams |
| Transformer attention | safe model-specific composition first | per-block Q/K/V and output seams |
| ReLU / Swish / sigmoid | safe reference, then general kernel | special values and full-tensor drift |
| Softmax and masks | FrankenTorch only after mask equivalence | fully masked, tail, and long-context cases |
| Top-k, stable sort, gather, optional permutation | safe Rust model logic | tie, placeholder, disabled, and eval-absence fixtures |
| Cache/FIFO mutation | safe Rust bounded state machine | empty, first-full, steady, tail, and cancellation fixtures |
| Log-mel frontend | exact Sortformer-specific reference | frame count plus per-stage tensor parity |

No new tensor runtime is justified. No model-specific fused kernel is justified
before the f32 whole-forward gate passes and profiling identifies a measured
hot path. Quantization and fusion belong to `bd-y4ip.12`, not the reference
implementation.

## 7. Oracle floor pilot

The current external adapter emits only the final discrete stage document. It
does not expose probabilities or intermediate activations and is therefore not
an L1-L6 parity oracle. `bd-y4ip.10` needs a separate identity-bound,
public-input activation exporter whose tensor names, shapes, dtypes, and bytes
are included in the conversion and evidence contract.

The pinned f32 CPU model was run twice with one PyTorch intra-op thread and
twice with eight intra-op threads on one public 53.603313-second input whose
PCM-file SHA-256 is
`c8bc396e0d6a257b45c5d200f91eeff73ee3f5b21ee0bb0bfc3417356b86fe22`.
The probability tensor shape was `[1, 670, 4]`.

- repeated runs within each thread regime were bit-identical;
- changing from one to eight threads changed 763 probability elements;
- cross-thread maximum absolute difference was
  `7.152557373046875e-07`;
- cross-thread mean absolute difference was
  `2.7044704253853524e-09`; and
- all four derived segment outputs had SHA-256
  `17bd517a696ff2301713b47fcacceb49a653f4dca3885ea4f63eeaeaa45ad5a2`.

This is a pilot, not the oracle floor used for acceptance. `bd-y4ip.10` must
repeat the measurement over predeclared public inputs that cover silence,
short tails, exact chunks, multiple chunks, overlap, two through four speakers,
and cache compression. Tolerances are selected after those results and must be
at least as strict as the observed source variability permits.

## 8. Ordered equivalence ladder

Each level is prerequisite evidence for the next. A later level cannot repair
or excuse a failed earlier level.

| Level | Required equivalence |
|---|---|
| L0 Artifact | Exact model/config/source identity and complete conversion receipt |
| L1 Frontend | Frame count, valid lengths, window, spectrum, mel, and log-mel tensors |
| L2 Subsampling | Every convolution stage and final factor-8 encoder input |
| L3 FastConformer | Input/output and selected internals for all 17 blocks |
| L4 Projection and transformer | Projection plus all 18 transformer blocks |
| L5 Head | Four sigmoid activity probabilities before streaming mutation |
| L6 Streaming state | Every cache/FIFO tensor, length, selection, and proof that permutation remains absent in eval |
| L7 Discrete activity | Thresholded speaker activity under the frozen output geometry |
| L8 Final document | Arrival labels, turns, speech, overlap, and changes pass the strict existing validator |
| L9 Public task behavior | DER/JER/count/overlap/calibration on frozen development and sealed test rows |
| L10 Resources | Same-invocation CPU, RTF, peak RSS, allocation, and cancellation gates |

Numeric reports must include maximum absolute error, mean absolute error, a
scale-aware relative statistic, mismatch counts over the declared tolerance,
and the source-oracle floor. Discrete reports must be permutation-aware where
speaker labels are arbitrary. Whole-model equality without seam evidence is
insufficient.

## 9. Open-question register

| ID | Question | State | Gate |
|---|---|---|---|
| OQ-01 | Does evaluation inject configured dither? | Resolved: no | L1 |
| OQ-02 | Exact STFT padding, mel floor, and length rounding | Open for numeric fixture | L1 |
| OQ-03 | Depthwise subsampling padding and layout at tails | Open for numeric fixture | L2 |
| OQ-04 | Exact relative-position attention equations, masks, and scaling | Open for numeric fixture | L3 |
| OQ-05 | Exact inference head branch | Resolved from pinned source; numeric fixture required | L5 |
| OQ-06 | Top-k tie behavior and chronological reordering | Open for numeric fixture | L6 |
| OQ-07 | First, steady, and partial-tail cache mutation | Partially source-resolved; fixtures required | L6 |
| OQ-08 | Speaker permutation during accepted inference | Resolved: disabled and absent in eval | L6 |
| OQ-09 | Converted package tensor map and transforms | Open until conversion receipt | L0 |
| OQ-10 | Cross-input and cross-thread oracle variability | Pilot only | L1-L8 |
| OQ-11 | Model bytes in repository or releases | Resolved: forbidden for initial route | L0 |
| OQ-12 | Known requirement above four speakers | Resolved: deterministic native fallback | L8-L9 |
| OQ-13 | Known timestamp intervals during parity | Resolved: post-forward mapping only | L9 |
| OQ-14 | Other operating systems and CPU feature tiers | Open; require separate runtime rows | L10 |
| OQ-15 | Frontend/postprocessing transitive runtime identity | Open: pin it or remove it from seam authority | L0-L1 |
| OQ-16 | Unknown recording actually contains more than four speakers | Open: capacity sentinel or capped status | L9 |
| OQ-17 | Activity still open at the final 10 ms sample versus strict 80 ms output validation | Open: tail fixtures and one canonical rule | L7-L8 |

OQ-02 through OQ-07, OQ-09, OQ-10, and OQ-15 can invalidate graph correctness
and block the f32 implementation gate. OQ-17 blocks final discrete parity.
OQ-14 does not block same-host parity but blocks a broad cross-platform support
claim.

## 10. Implementation slices

### Slice A: truth and conversion (`bd-y4ip.10`)

1. Extend the public-input oracle floor.
2. Define the canonical conversion-receipt schema.
3. Convert the pinned checkpoint outside Git into safetensors.
4. Audit the exact tensor census and hashes.
5. Capture all L1-L8 oracle seam fixtures outside Git when they contain large
   activations; commit only compact public-safe fixtures that satisfy repository
   policy.
6. Add tamper and identity-drift tests.

### Slice B: safe f32 engine (`bd-y4ip.11`)

Add a genuinely separate `sortformer_inference` module. It should load only
the identity-bound non-executable package, use one explicit CPU-only
FrankenTorch facade, checkpoint cancellation between bounded chunks and every
layer, and initially return `DifferentialStageDocument`. The strict Sortformer
validator in `src/differential_oracle.rs` is the first output boundary.

This slice must not change `DiarizationEngine`, CLI parsing, automatic routing,
the production report contract, or comparison protocol v2. It remains
evaluation-only until f32 parity is complete.

### Slice C: optimization (`bd-y4ip.12`)

Only after f32 parity:

1. profile the whole forward pass;
2. apply a stagewise quantization ladder and retain drift reports;
3. establish reference-kernel equivalence for each candidate hot path;
4. fuse one measured bottleneck at a time; and
5. compare the live incumbent and candidate in the same invocation with
   matched audio, normalization, threads, diarization parameters, scoring,
   decoding, and transcript equivalence.

A self-speedup, unmatched thread setting, changed scoring rule, or external
oracle timing row is not a native performance win.

### Slice D: product integration

After accuracy and resource certification, add a model-neutral native report
kind, typed Sortformer provenance, resolver/cache behavior, count and capacity
semantics, waveform preservation, CLI and robot parsing, persistence, and a
versioned comparison protocol. `Auto` remains unchanged until that work passes
its own frozen gate.

## 11. Privacy, safety, and failure behavior

- No confidential audio or transcript is part of conversion, parity, or
  committed test evidence.
- Model packages and large activation fixtures remain outside the repository.
- Errors and evidence retain hashes and typed failure codes, not local paths,
  audio samples, transcripts, speaker names, or embeddings.
- The Rust loader is safe code and rejects executable checkpoint formats.
- Every allocation derives from checked tensor dimensions and declared bounds.
- Cancellation is checked during hashing, loading, frontend work, every layer,
  every chunk, cache mutation, post-processing, and validation.
- Timeout or cancellation must not leave an inference worker mutating shared
  state after the caller returns.
- Unsupported capacity, missing model, identity drift, malformed output, or
  resource breach invokes a typed skip or the declared native fallback; it
  never silently degrades into a success row.

## 12. Current proof state

Completed here:

- full-model fit-screen and architectural selection;
- immutable upstream, artifact, config, source, and runtime pins;
- license and model-byte distribution policy;
- parameter, tensor, and operator census;
- explicit streaming and capacity contract;
- one-host nondeterminism pilot; and
- dependency-correct implementation and proof ladder.

Not completed here:

- conversion receipt or converted safetensors package;
- complete multi-input oracle floor;
- L1-L8 seam fixtures;
- any Rust model forward pass;
- f32, quantized, or fused parity;
- authoritative peak RSS, RTF, or cancellation evidence;
- frozen public multi-record accuracy certification; or
- production routing.

Those omissions are downstream gates, not evidence that they passed.
