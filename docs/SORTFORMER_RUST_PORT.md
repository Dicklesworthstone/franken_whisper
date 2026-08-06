# Native Streaming Sortformer Rust Port Contract

Status: Phase -1 fit screen and converted-weight artifact admission complete;
external-oracle provenance plus L1-L8 parity evidence remain open under
`bd-y4ip.10`
Fit verdict: **CONDITIONAL GO**
Promotion authority: none; this document does not enable a runtime route

## 1. Decision and no-claim boundary

The full end-to-end NVIDIA Streaming Sortformer v2.1 graph is the primary
learned-model candidate for a native Rust port. It directly owns speech
segmentation, overlap, speaker change, speaker activity, and speaker-count
behavior up to four speakers. A speaker encoder such as ReDimNet or ECAPA does
not replace those stages and remains a compact hybrid or low-resource ablation.

The existing native acoustic diarizer remains available as a deterministic
native-acoustic comparator, low-memory fallback candidate, and regression
control; it is not the learned-model graph oracle. The pinned NeMo graph plus a
future source/runtime/exporter-bound activation adapter will become the L1-L6
external graph oracle under `bd-y4ip.10`; the current final-output adapter is
only an identity-checked discrete-output diagnostic oracle. No native Sortformer route,
automatic routing change, accuracy claim, or speed claim is authorized until
the ordered parity and public evaluation gates below pass. The currently
resolved production behavior remains unchanged.

The port is a conditional fit rather than an unconditional fit because:

- the 117.7-million-parameter graph is a tier-2 port with a substantial
  FastConformer front end and stateful streaming overlay;
- the custom speaker-cache compression, top-k selection, and FIFO update
  semantics still require seam fixtures before implementation;
- the custom NVIDIA Open Model License requires an explicit distribution
  policy; and
- the current one-record exploratory nondeterminism probe is not an accepted
  oracle floor.

No existing native diarization code should be deleted to make room for this
route. A failed parity, resource, capacity, or public-accuracy gate leaves the
current production behavior unchanged.

## 2. Primary upstream and runtime pins

| Item | Frozen identity |
|---|---|
| Model | `nvidia/diar_streaming_sortformer_4spk-v2.1` |
| Hugging Face revision | `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d` |
| `.nemo` bytes | `471367680` |
| `.nemo` SHA-256 | `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8` |
| `model_config.yaml` SHA-256 | `2865d469c4d2aac54aa5b8a956b2423c053806dd20d5bf5d08675942a1acface` |
| `model_weights.ckpt` bytes | `471352898` |
| `model_weights.ckpt` SHA-256 | `eca9773c2dab91dd41fbaa4473cebb9d00811d67788ce2de609dadc6e499cdf4` |
| Pinned externally-derived 990-entry state inventory SHA-256 | `f4f219cf4ac6f755247b56d19e425db3d6a7c23c4509176549b363b63abdf532` |
| Canonical 992-record topology-projection SHA-256 | `2c32b0b9e48bb296e66615b038827d0fdde4b4fda2ce044a6c30cd317456c8d7` |
| Reviewed converter source SHA-256 | `6a946cc6647bf52244d0eaad89db834bdc52cc61fd08d9563632dd1f9d239c1e` |
| Canonical conversion receipt SHA-256 | `a1c6dce95ef4fd715965951bdaaa136e55e2219f93cf78122f8b462fbd07cbbe` |
| Converted package bytes | `491570584` |
| Converted package SHA-256 | `487fa30cb0aa9799c77bd9985e6787962c3991fab8d4d576a4f1221d45298f6a` |
| NeMo source revision | `40ace43c7cf151af78dc22027c02feeca7e06b6a` |
| Python | `3.12.12` |
| NeMo package | `3.1.0+40ace43c7c` |
| PyTorch / torchaudio | `2.7.1` / `2.7.1` |
| NumPy | `2.4.6` |
| safetensors | `0.8.0` |
| Sortformer oracle adapter SHA-256 | `8f376c979b7eaca41dc0a438d9aaa41c1c723052b97c45eb2acc59b6d6f00bde` |
| External model-contract SHA-256 | `7ac048e3372fe4c622840beddfbeef42944d961408360324cb7276a69c8542c5` |
| Qualified runtime-fingerprint SHA-256 | `3713fd3f024c1cef7d860706baf0dbaaf18058c03c26331da6254687693d564c` |

### 2.1 Converted-weight admission evidence

On 2026-08-05, the frozen converter completed both an initial publication and
an exact-artifact retry on Darwin 25.2.0 arm64. Each exited zero with 974
tensors and the converter, topology-projection, receipt, and package identities
listed above; the resulting directory was mode `0700` and both files were mode
`0600`. The final focused Rust verifier run passed 31 tests with only the
deliberately operator-local test ignored. That exact ignored test was then run
locally against the admitted receipt/package pair and passed 1/1. The loader
rejects directories and symlinks before opening; on Linux, Android, and Apple
targets its post-precheck open is nonblocking, does not follow the final
symlink, rechecks the regular-file type, and binds the opened device/inode to
the precheck. This proves L0 converted-weight admission only; it is not a
model-forward or diarization accuracy result.

The archive configuration says `nemo_version: 2.6.0rc0`; that string is not
the executable reference. The intended oracle behavior is the pinned NeMo
source revision and package/runtime set above. It becomes accepted seam
authority only after `bd-y4ip.10` authenticates the executed source/runtime and
exporter. A future source or package change creates a new evidence row and
cannot silently inherit these tolerances.

The relevant installed source files at the pinned revision were independently
hashed:

| Source file | SHA-256 |
|---|---|
| `nemo/collections/asr/data/audio_to_diar_label.py` | `f9b0d23bd52da417ac18418ea1c83aa1119f59e6b37d3b2b3159c8cb2f036234` |
| `nemo/collections/asr/models/sortformer_diar_models.py` | `4978dba1a02b414893123f66905a1e523d5bb65766903269b325746c67f6920a` |
| `nemo/collections/asr/modules/sortformer_modules.py` | `3d136c245e3bf7a88c47fdd2eae1edb9189bbeddc3ff779cb5679a29d890b7eb` |
| `nemo/collections/asr/modules/conformer_encoder.py` | `a8b6f712cdf75a3be768848e8242ea9412ca7ff31ba2dda6b9602bcefc627cec` |
| `nemo/collections/asr/modules/transformer/transformer_encoders.py` | `a2859c86c8389f1954d5c8be04dc2bc422452517ef15e069cf42bfab5d304759` |
| `nemo/collections/asr/modules/transformer/transformer_modules.py` | `2564d95365cfafd486b1a3d10e2e2f438702907076f3716dd4c42d568b3bcc72` |
| `nemo/collections/asr/modules/audio_preprocessing.py` | `c061f521e14978d22ad57fa5ddf08f1103c2d1f1a4e01aca6698bfad007e8e7c` |
| `nemo/collections/asr/parts/preprocessing/features.py` | `4290ed2d697362a68a6158fb8b7b8d1e2306b223b83172c63fc6b5d31b28ee69` |
| `nemo/collections/asr/parts/preprocessing/segment.py` | `a598d91b94110e0c12a1ba4a57894ce89109e597fa8e909cf7b5b6e7bb9369af` |
| `nemo/collections/asr/parts/submodules/conformer_modules.py` | `99bb846c51db028d6d30b3d844af22826068aeaa0e48eb586489a31a9cbacf9d` |
| `nemo/collections/asr/parts/submodules/multi_head_attention.py` | `4999fd0d679fd7315ba275f7311fe6608c48e492bd337f2e220c99b8b9729c69` |
| `nemo/collections/asr/parts/submodules/subsampling.py` | `4fbc689f3f66e4630b286196315a02b315ad53e8049c164fe40dd11168cf0834` |
| `nemo/collections/asr/parts/submodules/causal_convs.py` | `7cf505c8caef44a37a7dec10b51eb2d60ec2f1efc3a2badc3c20c37e427cbd42` |
| `nemo/collections/asr/parts/utils/speaker_utils.py` | `6c247bdda26fd010190e1c96f8399f77a5265a180086e134d9b167b3c8019dc0` |
| `nemo/collections/asr/parts/utils/vad_utils.py` | `7beb57efff5e08407f9f16afe9c0da7d0e2ddb9bd62e2a37424693e48c5f0437` |
| `nemo/collections/asr/parts/mixins/diarization.py` | `5365e416ecab192cf59f1b9d6554ebce0ed3bdb2fee7575966ac1e3fca1a1408` |
| `nemo/collections/common/parts/transformer_utils.py` | `47f5e337230e7b4e176877f01c2ae85f75c024942dc567f27d8429c3e60e67c0` |

These hashes identify the source semantics inspected for the port. They do not
prove that arbitrary Python code or model bytes are trustworthy.

The receipt verifier pins Python, NeMo, PyTorch, torchaudio, NumPy,
safetensors, librosa 0.11.0, Lhotse 1.33.0, SoundFile 0.14.0, SciPy 1.18.0,
OmegaConf 2.3.0, Hydra 1.3.2, and Lightning 2.4.0. That list is not yet an
audited transitive closure. `bd-y4ip.10` must bind every package and executed
source that influences an accepted seam or eliminate it from seam authority by
exporting the exact decoded PCM plus exact intermediate tensors. A repeated
version document alone does not close that gap. Every accepted future oracle
row must also authenticate the adapter executable against the pinned digest
above before its self-reported identity is trusted, authenticate the installed
NeMo source bytes rather than only package metadata, and remove or contain its
executable/audio/model hash-then-open windows. The converter's model input is
now one owned verified byte stream, but the external final-output adapter has a
separate lifecycle. Those are `bd-y4ip.10`/`bd-y4ip.7` gates, not evidence
supplied by this document.

## 3. License and artifact-distribution boundary

The model is under the custom
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
not the repository's Rust source license. The model card and license must be
reviewed again before distributing any derived weights.

The reference snapshot retrieved on 2026-08-06 had raw-payload SHA-256
`13c9c998e24abd5211cff4b5c912902f566bd710294da98580be7b3376626f04`,
HTTP `Last-Modified: Mon, 03 Aug 2026 17:46:28 GMT`, and ETag
`4b001-658281e31650b`. The pinned NeMo source tree is Apache-2.0; its `LICENSE`
payload has SHA-256
`43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1`.
The pinned `nemo/collections/asr/parts/preprocessing/features.py` also contains a Ryan Leary MIT
notice, which must be preserved in any distribution that carries derived
source. These identities establish an auditable review input, not legal
approval.

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

### 3.1 Initial hardware and memory tier

The f32 reference target is the current 64-bit little-endian
`aarch64-apple-darwin` development host: Apple M4 Pro, 64 GiB physical memory,
and Rust `1.99.0-nightly (9f36de775 2026-07-19)`. The reference code must not
require Apple-only instructions, but passing on this host authorizes no x86-64,
other ARM, operating-system, or lower-memory support claim.

This is a desktop/server memory-tier port, not an embedded or low-memory route.
The immutable f32 tensor payload is about 468.7 MiB after adding the positional
buffer; the safetensors container, decoded tensors, activations, streaming
state, and allocator overhead make that number an invalid RSS estimate. Until
L10 measures child-only peak RSS, evaluation requires at least 4 GiB of
available memory and treats that floor as provisional rather than certified.
Product routing and any low-memory fallback choice remain owned by the later
integration gate.

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

There are two named non-persistent buffers outside `state_dict`:
`preprocessor.dtype_sentinel_tensor`, an empty f32 tensor, and
`encoder.pos_enc.pe` with shape `[1, 9999, 512]`: 5,119,488 f32 values, about
19.5 MiB. The converted package must export the exact positional values or
bind and test a deterministic Rust regeneration algorithm; silently omitting
them from the audit is not acceptable.

All 17 int64 state tensors are
`encoder.layers.{0..16}.conv.batch_norm.num_batches_tracked`. They are
training-only counters. The initial package census is therefore 992 audited
source records: export the 973 f32 state tensors plus the positional buffer as
974 destination tensors, and explicitly receipt the 17 counters plus the empty
dtype sentinel as 18 drops. Before the safetensors header, the expected
destination tensor payload is 122,864,152 f32 elements or 491,456,608 bytes.

The conversion receipt must enumerate every tensor and record, at minimum:

- source key and destination key;
- source and destination dtype;
- source and destination shape and logical layout;
- the typed conversion transform;
- source-value and destination-value SHA-256;
- byte length and element count; and
- the pinned model, source, config, converter, and receipt-schema identities.

The loader must reject missing tensors, unexpected tensors, duplicate names,
duplicate or unknown tensor-entry fields, wrong shapes, unsupported dtypes,
non-finite materialization, checksum drift, and unknown package-schema
versions. A matching total byte count is not sufficient.

Version 1 permits only a byte-preserving
`identity_contiguous_f32` export: the destination name, shape, layout, value
hash, element count, and byte count must equal the f32 source record. The 17
training-only i64 counters and the empty runtime dtype sentinel are the only
legal drops. Transpose, reshape, split, concatenate, cast, quantization, and
prepacking require a new receipt schema and new parity evidence.

The trust chain is deliberately one-way. The Rust binary compiles the reviewed
converter-source, canonical topology-projection, receipt, package-length, and
package SHA-256 roots. Its public loader accepts only receipt and package paths;
there is no caller-supplied digest parameter. The topology digest is recomputed
from all structural receipt fields, so a renamed or reshaped f32 tensor cannot
hide behind matching aggregate counts. The authenticated receipt binds every
source and destination value hash. The loader independently proves each
exported destination's exact name, shape, dtype, raw payload, finite values,
compact lexicographic layout, metadata absence, and complete census; it does
not re-derive dropped source hashes from the executable checkpoint.

The frozen offline profile in `scripts/convert_to_safetensors.py` produced this
package from one owned, hash-verified copy of the exact `.nemo` byte stream. It
verified both archive members, installed `direct_url.json` revision metadata,
17 selected source files before and after export, the listed package-version
tuple, the original insertion-order state inventory, every tensor's contiguous
CPU representation, the frozen topology-projection and package identities, and
all aggregate censuses.
It also binds and rechecks its own source inode and digest, instantiates the
pinned graph without `restore_from` temporary extraction, publishes owner-only
mode-0600 outputs, and permits retries only by reusing exact artifacts. The
licensed model, converted package, and receipt remain operator-local and
outside Git.

## 5. Frozen forward graph

### 5.1 Input and frontend

- mono 16 kHz finite PCM bounded to `[-1, 1]`;
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

The converted package must retain `preprocessor.featurizer.window` with shape
`[400]` and raw little-endian f32 SHA-256
`c427e2029118cf789649e5a4d439b6115d0dd0cbf95dcd22f65e3c848add8c5b`, plus
`preprocessor.featurizer.fb` with shape `[1, 128, 257]` and SHA-256
`bce5ec5f194a5913f6508cee5a85512e7bad2352db8fc28f5c6ff75af8b09137`.
Rust loads and verifies these stored buffers rather than regenerating them.

The first oracle seam is the exact framed log-mel tensor and its valid length.
The effective valid-frame count is `floor(samples / 160)`: centered STFT
creates one extra frame, but NeMo marks it invalid and the model crops to the
declared length. The existing Whisper frontend is not reusable as-is because
its window periodicity, padding, transform length, logarithm, clamping, and
normalization semantics differ. Every frontend fact must still be captured by
numeric fixtures rather than accepted from source inspection alone.

The initial Rust L1 implementation is deliberately whole-file and bounded to
115,200,000 samples (two hours at 16 kHz), which includes the 90-minute target
class. It materializes the complete `[128, T]` log-mel tensor, although
pre-emphasis is computed on demand rather than stored in a second waveform
buffer. L10 must measure peak RSS on a 90-minute public or synthetic input.
True incremental pre-emphasis, STFT overlap continuity, and concatenation parity
are separate seams; this bound certifies no live-streaming frontend.

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

The archive encoder uses regular attention with `att_context_size = [-1, -1]`.
Evaluation never enters the training-only causal-context mutation, so
FastConformer's `_create_masks` returns no attention mask. It also returns an
inverted length mask (`true` means padding), but the relative-position
self-attention branch does not consume that pad mask; the convolution module
does. Attention is therefore entirely unmasked and bidirectional within the
concatenated cache, FIFO, and chunk sequence. The accepted batch-one
synchronous path normally has physical length equal to declared length, so it
does not introduce a padded tail at that seam.

The port must reproduce the NeMo ordering of feed-forward, attention,
convolution, residual, scaling, and normalization operations. Existing Whisper
attention helpers are not accepted as evidence of equivalent conventions.
FastConformer scales its input by exactly `sqrt(512)`. Every PyTorch LayerNorm
uses epsilon `1e-5`. Each convolution BatchNorm1d uses stored evaluation running
statistics and affine parameters with epsilon `1e-5`; training momentum is not
an inference input. Dropout is identity in evaluation, and stochastic depth is
disabled (`stochastic_depth_drop_prob = 0`).

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

`mask_future` is absent and therefore false, so the transformer's diagonal
future mask is absent. `form_attention_mask` returns valid key positions as
`[batch, 1, 1, length]`, with zero for valid keys and `-10000` for padded keys,
then broadcasts them across heads and queries when adding to attention logits;
it adds no causal triangle and no separate padded-query mask. The archive
sets `pre_ln = false`, which is why the otherwise named
`pre_ln_final_layer_norm` path does not instantiate a final layer norm.

### 5.4 Streaming state machine

The archive sets `streaming_mode = true`; `async_streaming` is absent, and the
source defaults it to false. The accepted path is therefore synchronous
stateful chunk emulation. It reads and preprocesses the entire waveform first,
skips global peak-amplitude normalization because streaming mode is set, and
crops features to the processed signal length. It is not evidence of a true
incremental-waveform frontend; live audio continuity requires separate frontend
and state seams.

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

For a chunk beginning at feature index `start`, the feature loader uses
`left = min(8, start)`, `end = min(start + 340*8, feature_length)`, and
`right = min(40*8, feature_length - end)`, then transposes and pre-encodes.
After factor-8 subsampling it uses `round(left/8)` left frames and
`ceil(right/8)` right frames. `drop_extra_pre_encoded` is zero.

The nominal input-buffer latency is 30.4 seconds and the output stride is 80
ms. The pinned validator warns about the configured 300-frame update being
shorter than the chunk but does not rewrite it. FIFO movement is:

```text
min(max(configured_update, chunk_len - fifo_capacity + current_fifo),
    current_fifo + chunk_len)
```

This moves 300 frames for the first full chunk with an empty FIFO, leaves 40
queued, immediately compresses the 300-frame cache to 188, and moves 340 frames
in steady state. Rust must evaluate the partial-tail subtraction with checked
signed arithmetic rather than permitting an unsigned underflow. An interior
chunk includes 381 pre-encoding frames; with a
full cache and FIFO, the recurrent encoder sequence can reach
`188 + 40 + 381 = 609` frames. Tail behavior is a required seam, not an
extrapolation.

Initial speaker-cache and FIFO embeddings have shape `[batch, 0, 512]`.
`spkcache_preds` and `fifo_preds` are initially `None`, as are
`spkcache_lengths`, `fifo_lengths`, and `spk_perm`; these Options must not be
substituted with empty tensors without a seam proving equivalence. `fifo_preds`
becomes a tensor at the first update, while `spkcache_preds` is synthesized only
at the first over-capacity cache compression. Mean silence is a zero
`[batch, 512]` tensor and the silence count is zero-valued i64 `[batch]`.
Lengths otherwise derive from physical shapes. Rust may own checked scalar
lengths as an implementation invariant, but parity must not invent oracle
length tensors.

Cache compression has 44 non-silence slots per speaker
(`188/4 - 3`). Its strong, weak, and minimum-positive counts are respectively
`floor(44*0.75) = 33`, `floor(44*1.5) = 66`, and
`floor(44*0.5) = 22`. A frame is silent when `sum_s p_s < 0.2`; the running
silence embedding is
`(old_mean*n_old + sum(new_silent_embeddings)) / (n_old + new_count)`, with the
denominator clamped to at least one.

For speaker lane `j`, the base score is:

```text
ln(max(p_j, 0.25)) - ln(max(1 - p_j, 0.25))
  + sum_k ln(max(1 - p_k, 0.25)) - ln(0.5)
```

`p_j <= 0.5` forces negative infinity. Once a lane has at least 22 positive
scores, its active non-positive scores also become negative infinity. There is
no evaluation-time noise or speaker permutation. Frame rows at indices 188 and
later receive `+0.05`. Per lane, `torch.topk(sorted=false)` first selects 33
rows for a `-2*ln(0.5)` boost, then selects 66 rows from the already boosted
scores for a further `-ln(0.5)` boost. Three positive-infinity silence
placeholder rows are appended per speaker.

The boosted matrix is flattened in speaker-major order and a global
`topk(188, sorted=false)` is taken. Winners at negative infinity receive the
sentinel index 99,999; indices are then numerically sorted, reduced modulo the
augmented score-row count (the physical rows plus three placeholders), and any
remainder at or beyond the original physical row count is disabled together
with the final three silence rows. Gathering uses row zero for disabled entries,
then replaces their embeddings with mean silence and predictions with zero. The resulting cache is
grouped by speaker and chronological only within each lane, not globally; one
physical frame may be selected once for each speaker lane. Exact top-k ties
remain an oracle-fixture question.

The output mask order is also fixed. FastConformer returns embeddings and
lengths; `encoder_mask = arange < length` supplies transformer key validity;
the sigmoid head runs; predictions are multiplied by
`encoder_mask.unsqueeze(-1)` to zero padded query rows; and
`apply_mask_to_preds` applies the FastConformer lengths again before central
chunk slicing and cache/FIFO mutation. A final crop to `ceil(signal_length/8)`
exists only in the distributed cross-rank padding branch. The accepted
batch-one, non-distributed path does not take that branch because whole-file
features were already cropped.

The NeMo state type also includes an optional speaker-permutation field.
Speaker permutation is training-only in the pinned path. Evaluation passes
`permute_spk = false`, so the accepted inference state has no active
`spk_perm`. The Rust evaluation state may omit non-evaluation support but must
prove that the oracle field stays absent; later training or adaptation support
would be a different contract.

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

A caller-provided pre-inference requirement above four speakers is
capacity-ineligible for this model. Before any product integration, such a
request must remain on the currently resolved route; the future integration
layer may choose a non-Sortformer path only after that path passes a separate
greater-than-four-speaker accuracy, capacity, privacy, and resource gate;
otherwise it preserves the current route or returns typed capacity-ineligible.
It must never clamp, drop, or merge speakers. A reference annotation loaded or
discovered for scoring only marks that completed row capacity-ineligible and
must never retroactively select a route.
The model itself cannot determine that an otherwise unconstrained recording
actually contains five or more speakers: it can only activate zero through
four lanes. Unknown-capacity production inputs therefore require a separate
capacity sentinel or a candid capped-output status before this route can be
certified.

Known timestamp intervals do not change the neural forward pass during parity
work. A later product adapter may use them as hard or soft constraints when
mapping the four opaque arrival-ordered streams to caller-provided speaker
references. Contradictory hard intervals must fail closed; they must not tune
or mutate model weights during a call. Soft intervals cannot override hard
evidence, and the output contract must distinguish an anonymous lane from a
lane identified through accepted reference evidence.

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
| Log-mel frontend | pinned-buffer, source-derived candidate | frame count plus per-stage tensor parity required at L1 |

No new tensor runtime is justified. No model-specific fused kernel is justified
before the f32 whole-forward gate passes and profiling identifies a measured
hot path. Quantization and fusion belong to `bd-y4ip.12`, not the reference
implementation.

## 7. Oracle floor pilot

The current external adapter emits only the final discrete stage document. It
does not expose probabilities or intermediate activations and is therefore not
an L1-L6 parity oracle. `bd-y4ip.10` needs a separate identity-bound activation
exporter whose tensor names, shapes, dtypes, and byte hashes are included in the
evidence contract. Exact activation bytes derived from real human speech stay
external and ephemeral; Git retains only identities and aggregate drift. Exact
committed seam values are limited to deterministic synthetic non-human inputs.

The immediate next gate is L1, not a whole-graph comparison. First freeze the
separate activation-exporter source and executable digests. Then capture exact
decoded PCM and valid lengths plus pre-emphasis, STFT, power, mel, and log-mel
tensors for deterministic synthetic silence, impulse, tone, and partial-tail
fixtures. After those exact synthetic seams agree, repeat each declared public
real-voice input five times at one thread and five times at eight threads. Keep
real-voice values outside Git and retain only identities, shapes, hashes, and
aggregate drift used to set the predeclared source-variability floor.

An exploratory, separately run probe reported two f32 CPU runs with one PyTorch
intra-op thread and two with eight intra-op threads on one public
53.603313-second input whose PCM-file SHA-256 is
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

The probe executable and dataset record identity were not bound to the current
adapter contract, and that adapter cannot emit probabilities. These figures are
therefore exploratory and non-authoritative, not an oracle floor or accepted
evidence row. `bd-y4ip.10` must bind the exact probe/exporter digest plus public
corpus name, version, record identifier, license, and retrieval identity, then
repeat the measurement over predeclared inputs that cover silence, short tails,
exact chunks, multiple chunks, overlap, two through four speakers, and cache
compression. Each tolerance must be the smallest predeclared bound that covers
the measured source-variability floor plus a stated numerical margin; it may
never be tighter than the measured variability or selected after seeing Rust
results.

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
| OQ-02 | Exact STFT padding, mel floor, and length rounding | Source-resolved; numeric fixture pending | L1 |
| OQ-03 | Depthwise subsampling padding and layout at tails | Source-resolved; numeric tail fixture pending | L2 |
| OQ-04 | Exact relative-position attention equations, masks, and scaling | Source-resolved; numeric fixture pending | L3 |
| OQ-05 | Exact inference head branch | Resolved from pinned source; numeric fixture required | L5 |
| OQ-06 | Top-k tie behavior and chronological reordering | Lane grouping/order source-resolved; exact tie mapping pending | L6 |
| OQ-07 | First, steady, and partial-tail cache mutation | Partially source-resolved; fixtures required | L6 |
| OQ-08 | Speaker permutation during accepted inference | Resolved: disabled and absent in eval | L6 |
| OQ-09 | Converted package tensor map and transforms | Resolved: exact 992-record manifest, receipt, and 974-tensor package admitted | L0 |
| OQ-10 | Cross-input and cross-thread oracle variability | Pilot only | L1-L8 |
| OQ-11 | Model bytes in repository or releases | Resolved: forbidden for initial route | L0 |
| OQ-12 | Known requirement above four speakers | Model ineligibility resolved; product fallback pending `bd-y4ip.14` | L8-L10 |
| OQ-13 | Known timestamp intervals during parity | Resolved: post-forward mapping only | L9 |
| OQ-14 | Other operating systems and CPU feature tiers | Open; require separate runtime rows | L10 |
| OQ-15 | Frontend/postprocessing transitive runtime identity | Open: pin it or remove it from seam authority | L0-L1 |
| OQ-16 | Unknown recording actually contains more than four speakers | Open: capacity sentinel or capped status | L9 |
| OQ-17 | Activity still open at the final 10 ms sample versus strict 80 ms output validation | Open: tail fixtures and one canonical rule | L7-L8 |

The fit screen has no remaining upstream semantic question that invalidates its
conditional-GO result. The numeric/tie fixtures in OQ-02 through OQ-07 and the
remaining oracle work in OQ-10 and OQ-15 are owned by `bd-y4ip.10` and block
`bd-y4ip.11` parity completion. OQ-09 is resolved. OQ-17 blocks final discrete
parity. OQ-14 does not block same-host parity but blocks a broad cross-platform
support claim.

## 10. Implementation slices

### Slice A: truth and conversion (`bd-y4ip.10`)

1. Extend the public-input oracle floor. **Open.**
2. Define the canonical conversion-receipt schema. **Complete for L0 v1.**
3. Convert the pinned checkpoint outside Git into safetensors. **Complete for
   the operator-local identity-bound, verifier-admitted package; no weights
   entered Git.**
4. Audit the exact tensor census and hashes. **Complete for all 992 source
   records, 974 exports, and 18 typed drops.**
5. Capture all L1-L8 real-voice oracle activations outside Git and retain only
   their identities and aggregate drift; commit exact activation values only
   for deterministic synthetic non-human fixtures. **Open.**
6. Add tamper and identity-drift tests. **L0 artifact tests complete; L1-L8
   seam/exporter tests remain open.**

### Slice B: safe f32 engine (`bd-y4ip.11`)

Add a genuinely separate `sortformer_inference` module. It should load only
the identity-bound non-executable package, use one explicit CPU-only
FrankenTorch facade, checkpoint cancellation between bounded chunks and every
layer, and initially return `DifferentialStageDocument`. The strict Sortformer
validator in `src/differential_oracle.rs` is the first output boundary.

This slice must not change `DiarizationEngine`, CLI parsing, automatic routing,
the production report contract, or comparison protocol v2. It remains
evaluation-only until f32 parity is complete.

### Slice C: evaluation worker (`bd-y4ip.13`)

After whole-forward f32 parity, wire a native Sortformer worker only into the
existing evaluation/comparison lane. It must emit typed provenance and
capacity status and must not alter resolver defaults, production routing, or
the comparison protocol's authority. Public accuracy and resource rows remain
required before product integration.

### Slice D: optimization (`bd-y4ip.12`)

Only after f32 parity:

1. profile the whole forward pass;
2. apply a stagewise quantization ladder and retain drift reports;
3. establish reference-kernel equivalence for each candidate hot path;
4. fuse one measured bottleneck at a time; and
5. compare the live production baseline and candidate in the same invocation with
   matched audio, normalization, threads, diarization parameters, scoring,
   decoding, and transcript equivalence.

A self-speedup, unmatched thread setting, changed scoring rule, or external
oracle timing row is not a native performance win.

### Slice E: product integration (`bd-y4ip.14`)

After accuracy and resource certification, add a model-neutral native report
kind, typed Sortformer provenance, resolver/cache behavior, count and capacity
semantics, waveform preservation, CLI and robot parsing, persistence, and a
versioned comparison protocol, including an explicit capacity sentinel or a
candid four-lane-capped status. This slice also owns typed hard/soft known-
interval mapping, contradiction tests, immutable per-call weights, and explicit
anonymous-versus-identified output semantics. Identified output and any `Auto`
promotion require a mapping-specific frozen gate with predeclared public and
adversarial inputs, mapping accuracy, calibration, coverage, and false-
identification evidence; failure preserves anonymous output. `Auto` remains
unchanged until that work passes its own frozen gate.

## 11. Privacy, safety, and failure behavior

- No confidential audio or transcript is part of conversion, parity, or
  committed test evidence.
- Model packages remain operator-local and outside the repository. All
  real-voice activation or embedding values are ephemeral; Git may retain their
  hashes and aggregate drift only. Exact committed values use synthetic
  non-human inputs.
- Retained Git, robot, and oracle evidence uses hashes and typed failure codes,
  not local paths, audio samples, transcripts, speaker names, or embeddings.
  Operator-local conversion commands and terminal diagnostics may contain local
  model/output paths and must not be copied into committed evidence.
- The Rust loader is safe code and rejects executable checkpoint formats.
- Every future model allocation must derive from checked tensor dimensions and
  declared bounds.
- The completed loader/frontend uses fallible bounded allocation and checks
  cancellation before and after large reservations, during bounded hashing and
  loading, during chunked zero-fill, and throughout frontend work. The future
  engine must also check every layer, every chunk, cache mutation,
  post-processing, and validation.
- Timeout or cancellation must not leave an inference worker mutating shared
  state after the caller returns.
- Unsupported capacity, missing model, identity drift, malformed output, or a
  resource breach must produce a typed skip; a fallback may be invoked only
  after `bd-y4ip.14` separately certifies its applicable capacity, accuracy,
  privacy, and resource envelope. Neither path may silently degrade into a
  success row.

## 12. Current proof state

Completed here:

- full-model fit-screen and architectural selection;
- primary upstream, artifact, config, source, and runtime pins, with the
  transitive-closure gap explicitly open;
- license and model-byte distribution policy;
- aggregate parameter/export/drop/operator/state census plus the pinned
  externally-derived insertion-order inventory digest;
- a reviewed frozen offline converter, canonical 992-record topology projection,
  653,202-byte receipt, and 491,570,584-byte metadata-free package, all with
  compiled digests while licensed bytes remain operator-local;
- explicit streaming and capacity contract;
- one-host exploratory nondeterminism probe, explicitly non-authoritative;
- a cycle-free, dependency-wired implementation and proof ladder;
- a safe conversion-receipt/package verifier with compiled trust roots, exact
  topology recomputation, strict safetensors parsing, fallible allocation,
  regular-file and path-swap defenses, synthetic tamper tests, and
  operator-local real-package admission proof; and
- a pinned-buffer, source-derived bounded whole-file Rust log-mel frontend
  candidate with fallible allocation and synthetic mathematical/unit tests.

Not completed here:

- complete multi-input oracle floor;
- L1-L8 seam fixtures;
- any Rust neural model forward pass beyond the log-mel frontend;
- f32, quantized, or fused parity;
- authoritative peak RSS, RTF, or cancellation evidence;
- frozen public multi-record accuracy certification; or
- production routing.

Those omissions are downstream gates, not evidence that they passed.
