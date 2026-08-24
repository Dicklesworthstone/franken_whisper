# Native Diarization Contract v3

Status: implementation contract for `bd-odj7`
Contract identifier: `acoustic-diarization-v3`

## 1. Purpose and authority

The native diarization stack answers “who spoke when?” from the normalized
waveform. It does not infer a name, gender, or legal identity. Its speaker
references are opaque within-run cluster identifiers or references provided
explicitly by the caller.

This contract governs the native acoustic path, the explicit `ecapa` and
`ecapa-fused` paths, and external implementations after output normalization.
An implementation cannot call itself acoustic if it only examines timestamps,
text, word counts, or segment position. ECAPA similarity is likewise acoustic
speaker evidence, not biometric identification.

The native engine contract remains authoritative for ASR text and timestamps.
This document adds the permutation-invariant speaker, confidence, supervision,
privacy, and bounded-resource contract that the ASR contract lacks.

### 1.1 Runtime identity matrix

These identities are distinct and must not be collapsed in reports, JSON, or
evaluation evidence:

| Requested path | CLI spelling | JSON/library spelling | Report implementation | Report contract | `speaker_evidence_mode` |
|---|---|---|---|---|---|
| Native acoustic v2 features | `acoustic` | `acoustic` | `native-acoustic-v2` | `acoustic-diarization-v3` | `acoustic_v2` |
| ECAPA identity only | `ecapa` | `ecapa` | `native-ecapa-only-v1` | `neural-diarization-common-v2` | `ecapa_only` |
| ECAPA plus bounded channel evidence | `ecapa-fused` | `ecapa_fused` | `native-ecapa-fused-v1` | `neural-diarization-common-v2` | `ecapa_with_acoustic_channel` |
| Unavailable ECAPA with `unknown` fallback | n/a (result state) | n/a | `native-ecapa-unavailable-v1` | `neural-diarization-common-v2` | `none` |
| Normalized external result | `external` | `external` | `external-backend` | `acoustic-diarization-v3` | `external` |

The ECAPA provider identity is
`ecapa-tdnn-voxceleb-cosine-v6-development`; the current native probabilistic
clustering identity is
`acoustic-clustering-probabilistic-v20-channel-evidence-bound-fused-consensus-development`.
The nested wire schemas remain `neural-speaker-representation-summary-v1` and
`diarization-operational-partition-v2`. Their names are schema identities, not
an accepted `neural` CLI or JSON engine value.

## 2. Canonical data flow

```text
normalized 16 kHz mono PCM with finite samples in the closed [-1.0, 1.0] range
    -> VAD and speech-quality mask
    -> 25 ms frames / 10 ms hop
    -> separate voice and channel features
    -> multiscale acoustic change scores
    -> microturns and tracklets
    -> acoustic-v2 sufficient statistics OR ECAPA discovery/validation embeddings
    -> bounded channel evidence only for acoustic and ecapa-fused identity scoring
    -> robust within-call speaker profiles
    -> constrained deterministic clustering
    -> temporal smoothing and unknown rejection
    -> independent diarized-turn timeline
    -> DTW word-boundary transcript projection
```

`normalized_input_sha256` binds every result and hint range to the exact PCM
used by ASR. Existing RMS/VAD summaries may seed analysis but cannot substitute
for waveform samples.

## 3. State, actions, and loss

The decision state consists of:

- microturn voice and channel sufficient statistics plus quality masks;
- hard and soft known-speaker intervals;
- current robust profiles and channel subprofiles;
- one typed speaker-count request: inference, caller prior, range, or hard
  search constraint; the current probabilistic count/assignment identity is v20
  and remains development-uncertified;
- prior temporal assignment;
- algorithm, feature-schema, weights, and calibration identities;
- remaining time, memory, and prototype budgets.

Available actions are:

- add or suppress a change boundary;
- merge compatible adjacent microturns into a tracklet;
- merge two compatible prototypes;
- assign a turn to a profile;
- create a new unanchored profile;
- return `UNKNOWN`;
- mark `overlap_suspected`;
- invoke a named deterministic fallback.

The initial loss hypothesis is a ledgered, calibratable policy rather than an
accuracy claim:

| Outcome | Initial loss |
|---|---:|
| Contradict a valid hard hint | forbidden |
| Wrong-speaker assignment or false speaker merge | 4.0 |
| False split of one speaker | 2.0 |
| Unknown assignment | 1.0 |
| Unsupported short speaker switch | 0.5 |

Changing these weights changes the policy identity and requires retained
evidence.

Speaker-change selection has a separate v2 loss contract. Its states are
`NoBoundary`, `Defer`, `EmitBoundary`, and `ConservativeFallback`. False splits
cost `1.0`; missed changes cost `9.0` because a later clustering stage can
reversibly merge an over-segmented tracklet, while an omitted boundary mixes
two acoustic regimes before clustering. The corresponding Bayes action
threshold is `1 / (1 + 9) = 0.10`. Timing, hint contradiction, latency, and
fallback costs are also frozen in `AcousticChangeCalibration`; changing any
value changes `acoustic_change_calibration_sha256`.

## 4. Acoustic evidence

The default `acoustic-feature-v2` schema maintains two views:

- **Voice:** twelve energy-centered filterbank cepstra, selected first and
  second differences, nullable fundamental frequency with a separate validity
  mask and uncertainty, periodicity, harmonic-to-noise ratio, three
  vocal-tract/formant proxies, voiced fraction, and temporal modulation.
- **Channel:** RMS and dynamics, spectral centroid/bandwidth/rolloff/flatness,
  effective band limit, high-frequency attenuation, spectral tilt, band
  ratios, crest/clipping, noise floor, stationary coloration, and conservative
  temporal-smearing, muffling, or distortion summaries.

Voice evidence is primary. Channel evidence is capped and may create multiple
channel subprofiles beneath one voice profile. Pitch is never sufficient by
itself and must never be translated into a gender label. Missing or uncertain
pitch is an invalid coordinate, never a physical zero.

### 4.1 Feature schema and aggregation

The library exposes a declarative `AcousticFeatureSchema` for v1 and v2. Every
family declares its coordinate range, units, voice/channel ownership, validity
rule, and normalization rule. The default is v2. The old eight-voice/eight-
channel v1 representation is reachable only through the explicit
`AcousticFeatureSchemaVersion::V1` segmentation entry point; it is not selected
by compatibility inference.

V2 owns 28 voice coordinates and 14 channel coordinates. Every tracklet,
enrollment observation, profile, prototype, and cluster carries the active
channel-coordinate prefix explicitly. V1 therefore compares exactly its first
eight channel coordinates, while a no-channel evaluation owns zero; neither
choice may be inferred from whichever voice coordinates happen to be valid.

V2 identity aggregation admits only voiced frames that are neither low-energy,
clipped, nor transient. It retains at most 64 deterministic highest-quality
subwindows per tracklet, then uses per-coordinate medians and MAD-derived
variance. A coordinate with no supporting subwindow remains invalid through
enrollment, prototype construction, clustering, and assignment distance.
Channel statistics separately admit usable non-clipped speech.

Per-recording normalization gives every provisional tracklet equal weight. It
uses the median and the larger of scaled MAD, scaled IQR, and a conservative
floor for each supported coordinate. This prevents a long or dominant speaker
from defining the center by frame count and erasing a short minority speaker.
The normalization never imputes a missing coordinate.

The frozen representation ablation surface is `full_v2`, `no_pitch`,
`no_channel`, `no_deltas`, `no_modulation`, and `v1`. These are experiment
configurations, not adaptive runtime selections. Each retained result records
the ablation ID, feature schema and hash, complete diarization request and
hash, and a configuration hash binding all of them to the runner version. The
speaker-count complexity penalty uses supported voice coordinates instead of
charging reduced representations for absent v2 dimensions.

`fw diarization-corpus ablate` executes all six representations and four
speaker-change detectors with the same frozen scorer: 250 ms speaker-boundary
collar, 250 ms primary change collar, excluded overlap, and ten calibration
bins. Change diagnostics additionally retain aggregate precision, recall, F1,
and p50/p90/p95 absolute timing error at 100, 250, and 500 ms collars, plus a
predeclared 19-point threshold sweep from 0.05 through 0.95. It uses reference
speech regions only as oracle VAD, subtracts ignored scoring regions, and never
supplies the reference speaker count. Source WAV/RTTM
files remain beneath an external input root. The validated reference bundle
and aggregate-only ablation evidence must be new JSON files in a separate
external directory; no source media or per-recording hypotheses are copied.
Evidence parsing recomputes schema, scorer, request, configuration,
development and held-out gates, and self hashes before accepting an artifact.
The artifact also carries a deterministic accuracy hash computed after
normalizing wall time, RTF, and process-RSS observations away, so identical
accuracy results can be distinguished from expected host-performance drift.

The representation decision is predeclared before observing corpus output.
On the frozen development split, full v2 must reduce micro-DER by at least 5%
relative to v1, must not increase macro-JER, and must not reduce change-F1.
On the frozen test split, full v2 must not increase either micro-DER or
macro-JER. Both decisions and their component deltas are part of the
self-hashed aggregate artifact; a run that merely completes is not a
successful representation result.

The v7 runner has an authority-bearing two-stage interface. `--stage
development` evaluates only the development split and cannot issue a held-out
verdict. `--stage certification` evaluates only the test split and requires
`--locked-development-evidence`; it verifies the exact development result
hash, accuracy hash, bundle, descriptor, duration protocol, calibration
identity, calibration-fit identity, detector-selection policy, and selected
detector and clustering mode before reading test audio. The v7 aggregate adds
selective coverage/risk, duration-weighted assignment calibration, typed
fallback counts, signed count error, and overlap TP/FP/FN. Overlap DER
exclusion does not remove intervals from the independent overlap detector
score. Test recordings observed during an earlier experiment are no longer
unseen and cannot be reused to mint a new promotion claim.

### 4.2 Frozen AMI representation result

The first non-synthetic run was frozen before its metrics were inspected. It
used the public AMI scenario split, oracle VAD but not oracle speaker count,
two recordings per development/test split, and deterministic 300-second
prefixes (600 seconds of scored audio per split). The path-free bundle hash is
`0d5219be241a3560cb55e3d8d9f63cd8d78ded4e46774d490583f34265eacd59`;
the performance-independent accuracy hash is
`9a0adb04b474d12a58f54cba63040b88b6f8139b73944c599603be95593c03b7`;
and the self-hashed aggregate result is
`de956df2fd96c690e866947fa44051ece31ea4e670933f4bf6058d7164b88696`.
The evidence JSON remains outside the checkout.

Full v2 reduced development micro-DER from `0.70067` to `0.53905` (23.07%
relative), reduced macro-JER from `0.90291` to `0.85863`, and raised
change-F1 from `0.09924` to `0.11429`, so the predeclared development gate
passed. On the frozen test slice, micro-DER fell from `0.69968` to `0.48714`
and macro-JER from `0.90965` to `0.85830`, so the held-out non-regression gate
also passed. Full-v2 RTF was `0.1069` on development and `0.0935` on test with
a sampled peak process RSS of 137,363,456 bytes on the measurement host.

The ablation is deliberately treated as diagnostic rather than uniformly
positive. Removing deltas made development micro-DER substantially worse
(`0.62330`), and removing modulation slightly worsened DER/JER and materially
worsened change-F1 (`0.09249`). Removing pitch or channel coordinates improved
DER/JER but reduced development change-F1. Most importantly, no variant found
the exact speaker count on this small slice; full v2 had mean absolute count
error `2.5`. The representation is therefore promoted over v1, while
pitch/channel fusion, boundary calibration, and count selection remain open
accuracy work. Because this test slice has now been observed, it must not be
reused as unseen promotion evidence after tuning; a new frozen test subset is
required.

### 4.3 Change posterior v2 development status

`acoustic-change-posterior-v2` replaces the raw Euclidean threshold with
bounded diagonal sufficient statistics and explicit detector ablations:
variance-aware GLR posterior, terminal Page-Hinkley/CUSUM approximation,
two-regime diagonal Bayesian/BIC approximation, and `FixedSafeV1`. All five
temporal scales remain inside a 401-frame ring. Voice and channel evidence are
fused separately; silence, legal word geometry, and optional TinyDiarize
support are typed bonuses rather than timestamp overrides. A coarse boundary
is refined inside a hash-bound ±300 ms neighborhood using spectral flux,
voicing and pitch discontinuity, energy valleys, and legal timestamp support.

Weak posterior excursions use bounded hysteresis: at most one candidate is
emitted during a 100-frame active interval and the detector re-arms only after
20 frames below half the action threshold. Evidence at or above 0.50 uses the
short 20-frame peak lane so rapid, unambiguous alternation is not swallowed.
Ill-conditioned covariance or insufficient voiced support invokes the frozen
fixed-safe score and records the fallback reason. The fixed-safe detector keeps
its original 20-frame suppression and remains the production default.

Development comparisons use the aggregate-only
`public-diarization-acoustic-ablation-v3` artifact. A 120-second-per-recording
AMI development diagnostic found that the Bayesian candidate reached change
F1 `0.13333`, Brier `0.21668`, and mean absolute boundary error `0.1975` s,
while the fixed-safe operating point matched no reference changes on that
slice. The candidate nevertheless failed promotion: ECE was `0.19496` against
the `0.10` gate, micro-DER regressed from `0.33854` to `0.51338`, and macro-JER
regressed from `0.52042` to `0.69470`. A two-second weak-peak suppression
experiment reduced recall to `0.08333` and was rejected. These are development
diagnostics, not certification.

Accordingly, normal segmentation and diarization entry points remain on
`FixedSafeV1`; posterior candidates are available only through explicit
ablation APIs. No held-out certification has been run for v2, and no
development result may be described as a promotion. The next accuracy work
must improve reversible clustering and boundary fusion before rerunning the
hash-locked development gate.

Privacy-safe diagnostics expose only schema and schema hash, aggregate
frame/missingness counts, supported dimension counts, bounded retained state,
and fallback/calibration state. They never expose feature values, audio,
transcript text, paths, or recording identifiers.

### 4.3.1 Experimental multiscale sidecar v4

`acoustic-multiscale-sidecar-v4` is an evaluation-only surface. Its
configuration defaults to `Off`, has independent per-axis mode orders and a
SHA-256, and is not a member of the six frozen `AcousticFeatureAblation`
variants. No
normal transcription, segmentation, clustering, robot, or persistence path
constructs this study. Running its arithmetic therefore cannot change the
acoustic-v2 schema hash, feature dimensions, report bytes, or default result.
This v4 identity supersedes the pre-evidence v3 identity. V3 had already
corrected v2 for complete observation configuration, independent trajectory
availability, stationary non-wrapping geometry, and near-constant admission.
V4 additionally validates submitted PCM for every configuration, including
trajectory-only and fully disabled studies, and freezes trajectory/scattering
precision, cast, operation-accounting, and scratch-buffer formulas in the
configuration digest. Neither v2- nor v3-labeled prototype observations are
schema-compatible with v4.

The current prototype implements four independently selectable, bounded
kernel families:

| Family | Input and support | Output | Owner |
|---|---|---|---|
| Frame Haar or D4/db2 analysis | One exact 400-sample normalized 16 kHz frame through the configuration-bound runner; at most four levels | Per-level local detail-energy fraction, log mean-square energy, normalized entropy, coefficient flatness, crest factor, adjacent-detail change, and Parseval residual | `MixedAuxiliary` |
| Modulation regression | A 64-frame ring at the acoustic-v2 10 ms cadence | Normalized regression power at 1.5625, 3.125, 6.25, and 12.5 Hz for the voice temporal-modulation, channel-level, and channel-coloration trajectories | `Voice`, `Channel`, and `Channel` respectively |
| Masked stationary trajectory Haar or D4/db2 analysis | One shared 64-frame ring of voiced cepstral-envelope magnitude, frame-local voiced occupancy, and low/mid/high band-energy fractions; at most four levels | Per-family and per-level valid support, mean absolute detail, RMS detail, separately available normalized entropy, and separately available adjacent-detail change | `Voice` for envelope magnitude, `MixedAuxiliary` for voiced occupancy, and `Channel` for every band fraction |
| Fixed scattering summaries | The same masked, normalized 64-frame trajectories with non-wrapping undecimated Haar supports 2, 4, and 8 | Independently selected first-order mean modulus and/or second-order mean modulus for ordered scale pairs `(2,4)`, `(2,8)`, and `(4,8)` | Same per-trajectory ownership as the masked trajectory input |

The standalone wavelet kernel also accepts bounded conformance fixtures from
the basis-specific minimum through 400 samples, and the public standalone
modulation sidecar can emit an unbound summary. Evaluation evidence must instead
use `AcousticSidecarStudy::observe_normalized_16khz_frame`; that executor binds
the 400-sample support, 16 kHz sample rate, 160-sample hop, selected bases,
level counts, scattering selection, and numerical contract to one configuration
digest. A study observation carries both the complete configuration and the
raw 32-byte digest in private, getter-only bindings, so a trajectory-only or
scattering-only result cannot be misleadingly described as merely `Off` by
its frame-wavelet axis or silently relabeled under another configuration.
Direct standalone or otherwise unbound kernel results do not carry this
binding and are therefore conformance diagnostics, not evaluation evidence.

The executor checks all 400 submitted samples are finite and within inclusive
`[-1, 1]` before any configured family runs. It applies that domain check even
when the frame-wavelet axis is disabled, so a trajectory-only, modulation-only,
scattering-only, or fully disabled observation cannot bless invalid PCM.
Physical sample rate, hop cadence, and content provenance remain caller
preconditions: the in-memory executor cannot verify an external sample rate,
prove that successive arrays advance by exactly 160 samples, or infer that a
separately supplied `AcousticFrameFeatures` value was derived from the same PCM
array. A valid evaluator must construct both from one normalized stream,
preserve contiguous frame indices, and bind its extractor revision separately.
The sidecar digest identifies configuration, schema, and sidecar arithmetic;
it does not hash every upstream acoustic-v2 threshold or FFT-bin decision.

Wavelet input is mean-centered and unit-energy normalized. Its centered RMS
must first exceed `8 * f32::EPSILON * max(1, abs(input_mean))`; otherwise a
constant or representationally near-constant frame is explicitly unavailable
instead of having quantization residue amplified to unit energy. DC-offset and
positive-gain invariance applies only while the centered input remains above
that gate and every transformed PCM sample remains within the submitted
`[-1, 1]` domain. Coefficient flatness adds a configuration-hashed,
scale-relative power stabilizer equal to the level's mean detail power times `f32::EPSILON`
to every coefficient power and to the arithmetic-mean denominator. This keeps
the ratio gain-invariant and bounded while preventing `f32` re-quantization of
mathematically zero coefficients after a DC shift from dominating the
geometric mean. A detail level at or below `PCM_EPSILON²` retains its measured
local energy fraction, uses `POWER_EPSILON` for log energy, and reports zero
distribution-shape statistics, including flatness. An odd intermediate width
duplicates its final sample, which is right half-sample symmetric extension,
and the analysis filters then use periodic support. Haar uses
`[1/sqrt(2), 1/sqrt(2)]` and `[1/sqrt(2), -1/sqrt(2)]`. D4 uses the frozen
four-tap analysis coefficients with forward taps starting at `2 * output_index`
and approximation-then-detail output order. Every level checks a Parseval
residual computed from raw energies, independently of reported-fraction
clamping, and rejects a relative error above `2e-5`. Silence or near-constant
input returns an explicit zero-level result rather than invented coefficients. Raw-PCM wavelets
combine vocal source, room, device, codec, and
playback effects, so they may never enter a reusable voice profile.

Modulation outputs are point-frequency regressions, not frequency bands. Each
family removes its valid-sample mean, residualizes the sine/cosine basis
against the intercept, solves the checked two-coordinate least-squares system,
and reports explained-energy fraction. A summary exists only after 64
contiguous frames; a duplicate or gap is rejected without state mutation. A
family requires at least 32 valid observations, centered RMS above
`8 * f32::EPSILON * max(1, abs(valid_mean))`, and a full-rank residualized
sine/cosine Gram system at every selected
frequency. Failure of any condition makes that complete family unavailable
with zero output rather than inventing measured absence. Invalid observations
are omitted, never replaced with zeros. Voice validity is a broader
sidecar-specific temporal-modulation mask:
voiced, non-low-energy, non-clipped, non-transient, and with a positive frame
index. Unlike the acoustic-v2 identity mask, it intentionally does not require
reliable pitch or RMS at or above -50 dBFS; public coverage gates and, where
Voice can be compared with auxiliary owners, the owner-separated same-speaker
auxiliary-dominance gates decide whether that broader support is useful. Channel validity
requires a non-low-energy, non-clipped observation. The fixed complex-step
constants, derived twiddle table, oldest-to-newest ring order, and compile-time
recurrence are configuration-hashed and avoid target-varying runtime
transcendental calls. The summary retains the valid counts, so an evaluator can
measure coverage and reject a candidate that works only on easy speech.

The trajectory ring has canonical family order
`voiced_cepstral_envelope_magnitude`, `voiced_occupancy`, `low_band_fraction`,
`mid_band_fraction`, `high_band_fraction`, and canonical oldest-to-newest time
order. Cepstral-envelope magnitude is the frame-local RMS across all 12
gain-centered acoustic-v2 cepstral-envelope coordinates. It is voice-owned and
valid on a voiced, non-low-energy, non-clipped, non-transient frame; it has no
predecessor-frame or positive-index dependency. Gain centering removes level,
not room, device, or codec coloration, so `Voice` remains a provisional study
axis subject to the lane-appropriate same-speaker auxiliary-dominance gates,
rather than evidence of reusable identity.
Voiced occupancy is the frame-local binary `quality.voiced` indicator, not the
history-bearing `voiced_fraction` IIR; it is valid whenever the frame is not
clipped and remains mixed auxiliary activity evidence rather than reusable
speaker identity. Each band fraction is valid only on a non-low-energy,
non-clipped frame and remains channel-owned. With the current 40 Hz FFT-bin
centers, acoustic-v2 derives the fractions over half-open bin-index ranges
whose centers are 0-440 Hz, 480-1,960 Hz, and 2,000-8,000 Hz respectively.
Those truncated extractor cut points are upstream provenance and are not
silently promoted into a sidecar identity.

No trajectory result exists before 64 contiguous frames, and a duplicate or
gap is rejected without advancing state. A family requires at least 32 of 64
valid observations and centered RMS above the declared gate; an otherwise supported
constant or near-constant family is explicitly unavailable. Specifically, its
centered RMS must exceed `8 * f32::EPSILON * max(1, abs(valid_mean))` before
unit-energy normalization. That representability gate prevents one-ULP input
jitter from becoming full-scale evidence; offset and gain invariance apply
only while both compared trajectories remain above the gate. Its valid
observations are mean-centered and jointly unit-energy normalized. Invalid
observations are omitted from both moments and coefficients; they are never
zero-imputed. The trajectory transform is an undecimated stationary cascade:
level `j` applies forward taps at every retained position with dyadic dilation
`2^j`, then passes the approximation path to the next level. It uses only
non-wrapping valid support, so the newest frame is never treated as adjacent
to the oldest and a one-frame sliding window does not re-anchor a decimation
lattice. A coefficient exists only when every input in that filter support is
valid, and that mask is propagated through the approximation path. A reported
level requires at least two valid detail coefficients. Normalized entropy and
adjacent-detail change have independent availability flags; both are
unavailable at or below a unit-normalized detail RMS floor of
`8 * f32::EPSILON`, preventing transform roundoff from becoming full-scale
shape evidence. Adjacent-detail change also retains its valid-pair count, so a
missing adjacent pair cannot masquerade as measured zero change. Adjacent
change is linear over neighboring retained coefficients and never wraps the
final coefficient back to the first.

The scattering candidate uses fixed, zero-learned, undecimated non-wrapping
Haar high-pass filters: the first half of each support is positive, the second
half negative, and the complete filter has unit L2 norm. First order averages
the valid modulus response at each support. Second order filters a first-order
modulus path only at a larger support and averages the resulting valid modulus.
One output requires at least eight valid positions. Non-wrapping
support prevents a smooth trend or single regime boundary from acquiring an
artificial reverse transition at the window seam. Focused analytic ramp and
scalar differential tests use `2e-6`; the promotion tolerance remains to be
frozen by `.15.3`. A constant or near-constant input trajectory is rejected
before normalization rather than reported as a bank of zeros. `FirstOrder`,
`SecondOrder`, and `FirstAndSecondOrder` are distinct
hashed selections. `SecondOrder` computes
only prerequisite first-order supports 2 and 4, then deliberately leaves
first-order output fields unavailable and zero. The combined selection also
computes and reports support 8. Selected evidence therefore cannot be confused
with hidden intermediate work.

Cancellation is checked before a study frame, before wavelet levels, before
each modulation family, before each selected frequency, before each trajectory
family and stationary-wavelet level, and before each scattering scale and
scale pair. The modulation and trajectory states are cloned together and
committed only after every enabled family succeeds. A cancelled frame can
therefore be retried without a hidden advance in either ring, including
cancellation after the modulation projections have completed but before
trajectory analysis begins.
All trajectory/scattering moment, energy, filter-response, and aggregate
accumulation is `f64`; normalized samples, coefficients/moduli, and public
summary values are rounded once when stored as `f32`. Adjacent trajectory-detail
change widens both stored `f32` coefficients before forming the difference and
absolute value in `f64`. Trajectory-wavelet accounting records one validity
visit per inspected support tap and two filter terms per tap only when the
complete low/high support is valid. Scattering records one validity visit per
inspected support tap and one filter term per tap only for a fully valid
support. These precision, cast, and counting rules, the three
trajectory-wavelet scratch value/mask pairs, four first-order scattering pairs,
and one additional second-order pair are configuration-hashed.

Diagnostics name filter-tap terms, validity-mask visits, valid
sample-frequency visits, exact buffer/table payload bytes, and target-specific
in-struct bytes. On direct five-family, fully valid conformance arrays, four
stationary trajectory levels use 4,600 filter-tap terms and 2,300 validity
visits for Haar, or 7,120 and 3,560 for D4. Full-valid first-order scattering
uses 4,130 filter terms and visits; second-order uses 7,450; combined selection
uses 9,730. The trajectory-wavelet scratch payload is 960 bytes. Scattering scratch is 1,280
bytes for first order alone and 1,600 bytes whenever second order is selected
on the declared Rust representation. A visit
count is not a scalar FLOP count, and none of these fields is a stack or RSS
bound. Wall time, RTF, and sampled RSS belong in the outer public evaluator so
host-dependent measurements cannot contaminate deterministic accuracy hashes.

All frame-wavelet, modulation, trajectory-wavelet, and scattering results remain
signal-derived. They intentionally have no serialization implementation, and
their custom `Debug` output omits feature values. Source-derived, public-corpus,
or per-recording sidecar observations and feature values must not be logged,
written to SQLite/JSONL, placed in a speaker-profile store, or retained in
repository evidence. Deterministically generated synthetic conformance values
and goldens may remain in unit tests. The public sidecar-study artifact retains
only aggregate, path-free and transcript-free metrics plus schema/configuration
hashes, operation counts, performance observations, and a self-hash. It never
serializes an `AcousticSidecarStudyObservation` or any constituent feature
value.

The sidecar kernels and evaluator are not an accuracy result or promotion.
Focused synthetic/unit checks cover bounded arithmetic, fixed transform goldens
plus in-tree scalar differential references, masked band-energy stationary
trajectory wavelets, fixed first/second-order scattering summaries, a
Voice-owned voiced-envelope magnitude trajectory, frame-local occupancy,
cancellation rollback, missingness boundaries for both scattering orders,
affine and explicit trajectory/scattering one-frame-translation metamorphic
checks, fixed-state accounting, configuration separation, and default-path
isolation.
It still does not include multi-coordinate cepstral trajectory candidates—the
current RMS magnitude intentionally collapses coefficient sign and ordering—or
any retained real-corpus development, RTF/RSS, held-out, or adoption result.
Nothing in this section establishes that any new candidate improves
diarization. `bd-odj7.13.15` therefore remains in progress.

The evaluator uses the separate
`public-diarization-acoustic-sidecar-study-v3` schema and
`public-diarization-acoustic-sidecar-study-runner-v3`; it does not modify public
acoustic ablation v8. The protocol fixes oracle VAD (`oracle_vad=true`) and
leaves speaker count inferred (`oracle_speaker_count=false` with an `Infer`
request). Its DER/JER and speaker-count evidence therefore evaluates
segmentation and clustering conditioned on reference speech regions, not
end-to-end VAD accuracy. Before reading development metrics it freezes this
exact lane order:

1. `full_v2_baseline`
2. `frame_haar_l4`
3. `frame_d4_l4`
4. `modulation`
5. `frame_haar_l4_and_modulation`
6. `frame_d4_l4_and_modulation`
7. `trajectory_haar_l4`
8. `trajectory_d4_l4`
9. `scattering_first_order`
10. `scattering_second_order`
11. `scattering_first_and_second_order`
12. `all_haar_l4`
13. `all_d4_l4`

The baseline is explicitly unfused. For each candidate, the v3 artifact retains
aggregate boundary metrics; conditional same/different-speaker metrics;
comparable-frame and retained-pair score coverage; Channel and MixedAuxiliary
dominance opportunities, counts, and rates split by same/different-speaker
class; separate boundary and lagged-pair calibrations and hashes; operation
counts; RTF/RSS; and deterministic/self hashes. Candidate-pipeline DER/JER and
speaker-count metrics are admitted only when `fusion_executed=true`;
unavailable fusion cannot masquerade as a measured zero or a baseline-equivalent
result. Development separately fits adjacent-frame boundary probability and
`P(different speaker | selected comparable reference-labeled frozen-lag pair)`.
The `acoustic-sidecar-boundary-fusion-v2` configuration identity binds the
selected change-detector mode in addition to sidecar configuration,
calibration, and selector constants; detector modes with different fallback
or peak-selection behavior therefore cannot share a fusion hash.
The v2 selected-sequence digest binds the normalized-PCM identity, retained
count, every selected key and frame/lag coordinate, and its same/different
reference label. Aggregate class totals therefore cannot conceal a label swap
between two retained positions, and all evaluated lanes must emit the same
digest. A candidate is accuracy-eligible when its only failures, if any, are
`PerformanceRegression` or the derived `NotSelectedByRanking`. Eligible
candidates rank by minimum candidate-pipeline micro-DER and then frozen lane
order. The unique top-ranked candidate must pass the complete live gate;
failure does not promote a runner-up. An advancing winner's disposition is
`advance_to_certification`; all other candidates are `rejected` and the
unfused lane remains `baseline`. Held-out audio stays sealed unless the exact
development artifact authorizes that one candidate.
Certification evaluates the unfused baseline plus only the locked candidate
and may mark that candidate `adopted` only after held-out non-regression. If no
candidate passes development, the correct outcome is a retained aggregate
negative result and no feature adoption. Implementing this contract does not
claim that any lane has advanced or been adopted; that authority requires a
real public-corpus artifact.

The frozen v3 selection policy requires at least 1% relative development
micro-DER improvement and certification micro-DER non-regression. Both stages
cap absolute macro-JER regression at 0.01 and require boundary-F1 and
speaker-count non-regression. Comparable-frame coverage must be at least 0.25.
Overall pair-score coverage and the same- and different-speaker class-specific
coverages—scored retained pairs divided by their retained bottom-k
denominators—must each be at least 0.25. At least 100 scored same-speaker and
100 scored different-speaker pairs are required, spanning at least five scored
recordings overall and five recordings in each class. Pair ROC AUC must be at
least 0.55, Brier at most 0.25, and ECE at most 0.10. Every expected
Voice-versus-auxiliary owner comparison requires at least 100 same-speaker
opportunities and a dominance rate at most 0.50: Modulation expects Channel;
combined frame/modulation, trajectory, scattering, and all-feature lanes expect
both Channel and MixedAuxiliary. Pure frame-wavelet lanes have MixedAuxiliary
but no Voice owner and therefore have no meaningful dominance comparison;
their pair discrimination/calibration gates still apply. Different-speaker
dominance remains diagnostic only. Separately, at least five recordings must
contribute paired DER and its 95% bootstrap upper delta bound must be
nonpositive. Relative RTF and RSS regression must each be at most 0.25.

The 2,000-replicate paired bootstrap uses the versioned
`public-sidecar-paired-bootstrap-splitmix64-v2` procedure. Its canonical seed
binds only the frozen seed policy, sampler, lane, split, and replicate count;
the DER/JER stream identity is bound separately. It deliberately does not bind
raw descriptor bytes or row cardinality, so descriptor whitespace, canonical
record ordering, held-out metadata, or identity-matched rows with no paired
metric cannot reroll an unchanged development sample. One SHA-256 digest
initializes each replicate stream; versioned SplitMix64 draws replace a digest
per draw, and cancellation is checked throughout pairing, observed-mean
accumulation, and resampling.

The eligible pair universe comprises 25/50/100/200-frame-lag pairs whose two
frames each have a unique reference-speaker label; unknown, ignored, and
overlapping labels are excluded. Eligibility and bottom-k admission occur
independently of feature availability. Each recording retains at most 4,096
pairs ordered by a score- and label-independent SHA-256 key over the
selection-key-v3 identity, normalized duration-clipped PCM digest, left/right
frame indices, and lag. The key excludes pair-scorer identity, feature
availability, probability, and same/different label; scorer, target,
population, and selected-sequence-digest identities are separately
protocol-bound. A v2 digest over each recording's normalized PCM identity,
ordered selected keys and coordinates, and per-key same/different reference
label must match across every evaluated lane. Evidence retains
aggregate eligible/retained/scored counts, retained same/different counts,
overall and per-class score coverage and recording support, maximum retained
counts and capacities, reliability-bin sufficient statistics, and a 100-bin
score-order histogram whose ten fine bins per reliability bin carry linked
class counts and probability, squared-probability, and squared-error sums. It
retains no selected members, frame indices, labels, keys, or probabilities.
The producer bins exact `f32` scores after intersecting them with the scorer's
`[f32::EPSILON, 1 - f32::EPSILON]` clamp. Verification checks linked counts and
first/second-moment feasibility against the exact closed `f32` support that
maps to each bin. Those checks reject direct aggregate contradictions, but the
omitted individual probabilities mean that sufficient statistics cannot prove
every member's bin assignment or independently replay score ordering and
binned ROC AUC.
Allocator capacities are target diagnostics and are normalized out of the
deterministic accuracy hash.

The verifier recomputes boundary precision/recall/F1; boundary and pair
Brier/ECE from retained reliability sums; pair class means and 100-bin ROC AUC;
fine-to-coarse score-distribution consistency; overall and per-class coverage;
dominance ratios; promotion gates; ranking; dispositions; protocol and
configuration hashes; the deterministic-accuracy hash; and the self-hash. It
validates retained calibration identities, parameter bounds and hashes,
development fit-count provenance, cross-lane selected-pair identity, bootstrap
seed/count/interval shape, and bounded sampler accounting. Because raw fit
histograms, selected pair members, audio-derived contrasts, per-recording
scoring rows, and per-recording bootstrap deltas are absent, it cannot
independently establish every selected probability's histogram membership,
refit either calibration, regenerate bottom-k membership, replay pipeline
DER/JER, or reproduce bootstrap interval values. Verification
therefore establishes artifact-internal consistency against the frozen
protocol, not corpus replay.

Reference labels for calibration and pair diagnostics use one monotonic,
overlap-aware sweep over canonically ordered turns, ignored intervals, and
change points. Active turns are retired by end-time heap, so a long annotation
cannot force a complete turn/change scan for every 10 ms frame. This cursor is
evaluation-local and retains no label or timestamp in the aggregate artifact.

RTF and RSS are host-dependent operational gate inputs, not accuracy evidence:
they are excluded from the deterministic accuracy hash but retained in the
self-hashed result. The deterministic projection clears result/accuracy
self-hash fields and, for certification, the locked development result hash
while retaining the locked development accuracy hash. It zeros sidecar and
candidate-pipeline wall time, RTF, and RSS; retained pair/signal allocator
capacities; and target-sized retained-state bytes. It removes
`PerformanceRegression` and `NotSelectedByRanking`, recomputes gate pass state,
and reruns development selection or certification adoption. Other deterministic
accuracy, calibration, configuration, and decision evidence remains bound.
The RTF comparison uses the live unfused baseline and the candidate
under the same request in one invocation. RSS is the Linux process high-water
mark when available and otherwise a sampled process RSS, so it is a coarse
process-level observation rather than an isolated per-lane footprint. Passing
these bounds cannot establish a speed or memory win, and no such claim exists
without a separately retained public probe.

### 4.4 Historical probabilistic clustering evidence

The measurements in this subsection are retained historical v2 development
evidence. They used
`public-diarization-acoustic-ablation-v7`,
`public-diarization-acoustic-ablation-runner-v7`, and
the historical `diarization-scorer-v3`, not the current scorer-v5 authority.
The study evaluated two public AMI development recordings,
each clipped to a deterministic 120-second prefix, with oracle VAD and without
oracle speaker count. The public bundle hash is
`34f405b6220d479f4d0d86937de77d51375ed39120abfd3a2f38e775a24e874e`;
the candidate calibration hash is
`fc286e7aec51d4b2362e3162bcd8a77451ef610a9e31f1295ba469856f4025ca`;
the performance-independent accuracy hash is
`4a0e62a073067c2d9c5f45378600844e240d9a0447954219ec6f28dd8d203f34`;
and the self-hashed result is
`8aef28a314c500feb33ff96afe233067fb03a6b92d093171967136e2ca8aac55`.
The aggregate artifacts remain outside the checkout.

Against fixed-safe clustering, the probabilistic candidate reduced micro-DER
from `0.27723` to `0.24649` (11.09% relative), reduced speaker confusion by
3.99 seconds, reduced mean absolute count error from `2.0` to `1.5`, increased
selective coverage from `0.87323` to `0.90952`, and slightly reduced selective
risk from `0.16741` to `0.16683`. All requested probabilistic runs completed
without fallback and mean five-view count stability was `1.0`. Wall time for
240 seconds of audio was `33.702` seconds (RTF `0.14043`) versus `33.757`
seconds (RTF `0.14065`) for fixed-safe; sampled peak RSS was 136,609,792 versus
136,265,728 bytes.

The candidate did not pass. Macro-JER regressed from `0.51001` to `0.58986`,
assignment ECE was `0.21716` against the `0.10` limit, and both modes had
overlap F1 `0.0`. An intermediate monotone square-root confidence experiment
reduced ECE to `0.13968` but still missed the limit and was not selected.
Accordingly, `selected_clustering_mode` remains `fixed_safe_v1`, no candidate
lock was minted, and no held-out recording was read. These results establish
real public development evidence and performance observations, not production
promotion or held-out certification.

The subsequent historical
`acoustic-clustering-probabilistic-v5-development` count candidate added a
separately versioned `speaker-count-estimate-v2` report. It did not inherit the
v2 evaluation authority, and neither historical study confers authority on the
current
`acoustic-clustering-probabilistic-v20-channel-evidence-bound-fused-consensus-development`
identity. The current v20 identity remains `DevelopmentUncertified`; this
subsection contains no retained promotion or production-accuracy evidence for
it. The default acoustic assignment path therefore remains `fixed_safe_v1`;
the explicit ECAPA development modes exercise v20 only because the caller opts
into that uncertified path. Native fixed-safe runs still emit the
count-estimate object, but with
`fixed_safe_uncalibrated`, no concrete bins or selected count, and all
probability mass assigned to `unresolved`.

## 5. Known intervals

`speaker-hints-v1` carries:

- `speaker_ref`: non-empty opaque reference;
- finite `start_ms < end_ms` within the normalized audio;
- confidence in `[0, 1]`;
- `hard_must_link` or `soft_enrollment`;
- optional provenance metadata;
- ordered document hash; request order is semantic because `hint_index` is
  positional.

Hard intervals with different references may not overlap after sample
quantization. Hard intervals are immutable assignments, but enrollment still
removes boundary guards, non-speech, and low-quality frames. An interval with
no usable speech fails rather than creating an empty trusted profile.

Soft hints contribute capped pseudo-counts and priors. Contradiction checks use
the selected identity representation: acoustic-v2 for the acoustic engine and
ECAPA embeddings for either ECAPA engine. Channel and classical voice
coordinates cannot veto ECAPA-only enrollment. Provenance is audit metadata
and cannot increase confidence by itself.

A request with nonempty `known_intervals` rejects both external execution and
`fallback=external` before diarization. External labels cannot enforce the
immutable hint identities, so silently discarding that constraint is forbidden.

Profile training is independently audited from attribution. A hard interval
remains an immutable assignment even when its acoustic observation is
quarantined from profile training. Training uses robust global and
leave-one-out distance checks, a nearest-peer contamination check, and a
low-voiced-coverage downweight. One speaker may retain at most four voice
subprofiles and multiple channel subprofiles, which preserves repeated vocal
or recording modes without allowing unbounded profile growth.

Within-call metric adaptation is conservative and reversible. It requires at
least two enrolled speakers, two observations per speaker, and six total
observations. Per-coordinate weights are bounded to `[0.9375, 1.0625]`; unmet
support returns exact unit weights with a typed fallback. Durable profile
summaries expose only counts and policy outcomes, never reusable voice or
channel vectors.

The native acoustic contract accepts at most 1,024 known intervals per
request, 256 bytes per speaker reference, and 4,096 bytes per provenance
value. These limits are validated before the hard-hint overlap check.

## 6. Output and confidence

The diarized-turn timeline is the acoustic source of truth. Turns use
deterministic `(start_ms, end_ms, speaker_ref)` order, each individual speaker's
turns do not overlap, and simultaneous turns for distinct speakers are legal
only when every participating turn is labeled and carries explicit overlap
evidence. Each turn contains:

- finite monotonic start and end;
- optional speaker ID, where absence means unknown;
- independent speaker-assignment and change confidence;
- source implementation and feature-schema identities;
- anchored/inferred source;
- overlap suspicion and fallback status.

ASR segment `confidence` remains ASR confidence. Speaker confidence is a
separate field. Transcript projection may split only at legal DTW word
boundaries and cannot invent, drop, duplicate, or reorder text.

**Projection fusion (`projection-fusion-v1`, bd-d4py).** Projection runs in
two passes. The primary pass keeps the historical conservative gates
(70% duration dominance for non-word segments, the 0.30 turn-confidence gate
for words). A second fusion pass then attributes segments the primary pass
left `null` using the same turn evidence: a segment overlapping any labeled
turn takes the max-overlap labeled turn, and a timed segment in a turn gap
takes the nearest labeled turn within 2 s (hard-hint turns are never
extrapolated into gaps: a `hard_must_link` interval asserts identity only for
its own audio). Word-granularity speaker changes
that land mid-clause are re-anchored to the nearest sentence-final
punctuation boundary within ±4 words, using the transcript's own punctuation
as the boundary oracle (quantized diarizer boundaries — e.g. Sortformer's
80 ms lanes — otherwise misattribute the first/last word of each turn).
Fusion rewrites only the projected per-segment speaker labels: turn
timelines, text bytes, timing, and ASR confidence are untouched, untimed
segments stay `UNKNOWN`, and the report additionally carries the merged
`speaker_segments` view (consecutive same-speaker runs with joined text and
duration-weighted turn confidence).

### 6.1 Speaker-count and evidence result

`SpeakerCountRequest` has exactly one mode:

- `Infer` selects only speakers that pass acoustic evidence gates;
- `Prior` carries sorted, unique probability mass over positive counts as soft
  evidence pooled into at most 15% of concrete posterior mass; counts outside
  its support remain eligible, and five-view acoustic agreement attenuates the
  prior to 7.5% at unanimity;
- `Range` is a soft uniform preference inside the interval, not a hard search
  bound;
- `HardConstraint` searches for exactly the requested count but does not assert
  that the count exists in the recording.

Every mode keeps `UNKNOWN` legal. Only a `hard_must_link` interval can prohibit
`UNKNOWN` for its own tracklet. Count cardinality never suppresses rejection,
and no assignment may be changed merely to make a missing label appear.

An inferred speaker must have retained occupancy and quality evidence:
independent voiced support, recurrence or repeated tracklets, assignment
confidence, profile reliability, and separation from already-supported
speakers. A hard-hinted speaker is supported by the hard attribution even when
its sample is quarantined from profile training. Soft-enrollment observations
cannot validate themselves; an uncorroborated soft name remains an audited
hint and does not become a fabricated profile or output label.

`speaker_count` in the stable report contains the original typed request,
`resolved`, `satisfied`, `unsatisfied`, or `unresolved` status, the
evidence-supported count, active references, dominant and unknown voiced
shares, structured reasons, and a per-speaker evidence summary. `hint_evidence`
separately records a privacy-safe disposition and count audit for every known
interval. The report does not expose acoustic feature values.

Adjacent assignments merge into a turn only when their hard-hint and overlap
provenance agree. This prevents a long merged turn from laundering a weak or
unsupported assignment into the confidence of a neighboring tracklet.

### 6.2 Canonical DTW projection timeline (`bd-2noc.1`, `bd-2noc.2`)

`group_tokens_into_words` quantizes token boundaries to the alignment grid and
may legitimately emit a terminal `[t, t]` observation. That decoder
observation is not itself a projection interval. The native backend now owns
one conversion from raw observations into `dtw-projection-v2` canonical units
before diarization or persistence can consume them.

A canonical unit has:

- finite `f64` seconds;
- half-open `[start_sec, end_sec)` semantics;
- non-negative, strictly positive duration;
- monotonic, non-overlapping order;
- a minimum word-aligned duration of 1 ms, because acoustic boundary hints are
  integer milliseconds;
- source segment and word indices;
- boundary provenance and explicit clamp/expansion status.

The adjacency epsilon is 1 microsecond. It exists only to absorb
floating-point noise around a boundary that is intended to be identical. It
is not the 50 ms cross-engine comparison budget. A sub-epsilon overlap is
canonicalized to exact adjacency. Material overlap, reversed time,
non-finite time, negative time, unpaired parent timestamps, materially
overlapping parents, and extra timing vectors fail with stable
`FW-DTW-PROJECTION-*` reasons.

For valid DTW observations, the adapter reserves 1 ms for each remaining word,
clamps observations to the enclosing decoder segment, and expands quantized
zero-width units within that reservation. This repairs a terminal `[t, t]`
without inventing, dropping, duplicating, or reordering text. Punctuation-only
content remains legal.

Missing DTW words for a segment use deterministic linear interpolation within
that parent segment. If a parent is too short to represent every word as a
distinct millisecond interval, the adapter emits one parent-segment unit. Both
cases retain fallback provenance and set `word_aligned_safe=false`; acoustic
projection then uses conservative duration dominance rather than claiming
legal DTW word boundaries. `no_timestamps` also disables that claim.

The privacy-safe `projection_timeline` raw-output object records the schema,
units, interval semantics, tolerances, input/output counts, provenance counts,
adjustment counts, and `word_aligned_safe`. That object contains no transcript
text, recording path, or speaker identity. The orchestrator trusts the typed
`projection_timeline.word_aligned_safe` field rather than inferring legality
from the older descriptive `word_timestamps` string.

`fallback_reasons` is an ordered list of stable reason strings. It distinguishes
missing decoder word timestamps, insufficient parent duration for distinct
millisecond word units, and timestamp suppression requested by the caller.
This required field and the durable summary contract are the reason the schema
identity is v2; consumers must reject v1 rather than silently assuming the new
fallback semantics.

The simulated proportional CTC alignment stage does not rewrite a timeline
carrying that proof. Real decoder attention-DTW offsets are authoritative, and
mixing proportional corrections with per-segment fallbacks can otherwise
reintroduce overlap at the end of the recording. The align stage records a
deterministic preservation note and leaves canonical segment bytes unchanged.

SQLite `result_json` is the authoritative durable copy of the typed
diarization report and the projection timeline. `RunStore` exposes both after
restart, and JSONL export/import carries the same canonical payload into a
fresh database before rebuilding its normalized diarization indexes. Robot
`run_complete` deliberately emits speaker turns and projected segments but not
the backend raw-output object, which can contain internal model paths; durable
projection provenance is retrieved through the stored-run surface instead.
That surface exposes only the privacy-safe `projection_timeline` sub-object,
not the rest of backend raw output or its internal model paths.

The fixed-safe speaker assignment combines best-versus-second margin with
profile reliability and reports `heuristic_uncalibrated`. The probabilistic
development candidate reports a separately versioned normalized local-emission
score, not a calibrated temporal posterior. The local score, not the reported
mapping, controls post-Viterbi abstention so a confidence transform cannot
silently expand coverage. This abstention can replace a Viterbi label with
`UNKNOWN`; it does not rerun the dynamic program and therefore makes no claim
that the returned sequence is the final path-optimal temporal assignment.
Pre-gating low local-emission states before Viterbi, or choosing the next
viable state after a local rejection, are separate v5 development evaluation
candidates. They cannot replace the conservative post-Viterbi abstention until
frozen public comparative DER, JER, and selective-risk calibration establishes
the changed loss tradeoff.
Speaker-change
candidates separately carry the versioned v2 posterior, component evidence,
supporting-scale mask, refinement offset, detector identity, fallback reason,
and calibration hash. The retained scoring surface reports Brier score,
expected calibration error, reliability bins, threshold sweeps, and coverage
when ground truth is available, so returning unknown or emitting no boundaries
cannot appear successful.

The probabilistic temporal path uses duration-aware continuity: switching cost
depends on current run length, next-tracklet duration, inter-tracklet gap,
boundary confidence, and whether `UNKNOWN` is involved. Short unsupported
fragments receive a penalty, while a strong acoustic boundary or real gap earns
bounded credit. The fixed-safe path retains its original constants, and every
candidate parameter is part of the speaker-pair calibration hash.

Overlap is an independent acoustic claim. The tracklet aggregates a bounded
dual-periodicity probability only from non-clipped, non-transient frames. A
probabilistic assignment emits a second speaker only when both independent
speaker likelihoods exceed their floor and their ratio is sufficiently close;
otherwise `overlap_suspected` remains diagnostic. A supported secondary
assignment projects to two simultaneous turns. A secondary label without
overlap evidence fails closed.

`speaker_queries` is a bounded active-agent surface for unknown, low-confidence,
or overlap-ambiguous spans. Adjacent requests are merged and capped at 32.
Each query ID hashes the input hash, interval, reason, candidates, and policy.
Queries contain no audio, transcript text, feature values, or path. When an
agent supplies a known interval in response, its provenance changes the
canonical hint hash but never its acoustic weight.

## 7. Determinism and resource limits

- Frame cadence is 16 kHz, 400 samples, 160-sample hop for feature schemas v1
  and v2.
- Cancellation is checked at least every 32 frames and within clustering and
  smoothing. Projection is bounded linear work inside the independently
  budgeted Diarize stage; persistence and cleanup retain their existing
  pipeline cancellation boundaries.
- Whole-call raw frame matrices and full raw-audio wavelet transforms are
  forbidden. `AcousticFeatureStream` accepts arbitrary sample chunks, retains
  only one fixed frame buffer plus DSP state, and is byte-for-byte equivalent
  to batch extraction across chunk boundaries. A late invalid chunk may return
  an error after earlier frames have already been emitted; callers that require
  atomic publication must buffer or transact their own output sink.
- `AcousticSegmentationStream` adds the fixed 401-frame change ring and emits
  compact tracklets. Its retained working state is duration-independent;
  returned tracklets are deliberately output-proportional. Normal runtime
  retains only that ring and compact tracklets. The
  public evaluator may opt into a duration-proportional score stream, but its
  existing 256 MiB source-audio cap gives that diagnostic allocation a fixed
  upper bound; it is immediately reduced to aggregate metrics and never enters
  the stable report.
- The native acoustic implementation defaults to at most 512 global prototypes. Anchored prototypes
  are never discarded. Cap pressure is reported.
- Exact constrained agglomeration is bounded by the prototype cap, followed by
  linear-in-turns temporal refinement.
- Stable labels place anchored references first, then unanchored speakers by
  earliest reliable occurrence and a total-order comparison of the compact
  feature vector. Generated `SPEAKER_NN` labels never collide with an opaque
  reference supplied by the caller.
- Identical input, request, implementation, and feature schema produce
  byte-identical typed output in the deterministic implementation.

## 8. Conservative fallback

Insufficient or out-of-calibration evidence remains unknown. Hard-hinted
intervals may remain labeled while all other speech stays unknown. The engine
does not invent speakers merely to satisfy a range minimum or exact count; it
reports `unsatisfied_constraints`, reports `speaker_count_unresolved`, or
returns a hard error according to typed fallback policy.

The bounded development speaker-count design, currently identified by the v20
probabilistic clustering identity above, uses five deterministic semantic
views: full evidence, no pitch, no dynamics, no formants, and no channel. It
retains their complete bounded merge-risk curves, combines them with a
symmetrized degree-bounded normalized-affinity eigengap proposal, applies hard
constraint-graph lower bounds, and linearly pools at most 15% caller-prior mass
into the acoustic count distribution before checking the selected count
against effective post-assignment occupancy. A clipped prior or range receives
only the fraction of that weight retained inside the feasible count domain.
Five-view acoustic agreement linearly attenuates the mix further to 7.5% at
unanimity. The ordinary inferred-count search ceiling is eight, but distinct
hard anchors may raise both the lower bound and ceiling, up to the global bound
of 64, rather than creating an inverted search interval. The bounded pool can
move probability but cannot erase acoustically supported counts, acquire the
unbounded leverage of a near-zero log prior, or veto unanimous evidence through
the unresolved-mass threshold. The public estimate carries ordered concrete
count bins plus separate unresolved
mass, entropy, stability, six typed lane summaries, and content-free
calibration/evidence hashes. It also reports content-free resource accounting:
prototype and retained-edge counts, affinity comparisons, estimated peak
algorithm buffers, stability replicates, eigensolver iterations and sparse
matrix-vector terms, and the final residual when available. These values and
the solver's diagonal shift are bound into the evidence/calibration
fingerprints. Lane agreement is only a development score input; the estimate
is explicitly `development_uncertified`, not described as a calibrated
posterior.

Selection requires the concrete MAP action to dominate unresolved mass, at
least three of five feature views to support it, co-association consensus, and
matching supported occupancy. Any failure retains an unresolved estimate and
invokes the typed fixed-safe assignment fallback. No-voice, insufficient
prototype, invalid-affinity, non-convergence, contradictory-constraint, and
resource-limited states remain non-authoritative. Public evidence aggregates
stability for every requested probabilistic run, including runs that fell
back, so fallback cannot inflate the mean by disappearing from its
denominator.

The six-dimensional temporal/lexical heuristic is not an acoustic fallback.
An explicit acoustic request cannot silently invoke external or lexical
behavior. Soft count priors and ranges are consumed by the native acoustic and
explicit `ecapa` and `ecapa-fused` engines and rejected by external engines
instead of being silently hardened into backend min/max controls. Every
fallback names its source and reason.

For either explicit ECAPA request, fallback depends on whether inference
completed but the resulting speaker evidence was insufficient, or whether the
representation could not be resolved, loaded, or inferred at all:

| Policy | ECAPA evidence insufficient | ECAPA package/load/inference unavailable |
|---|---|---|
| `unknown` | Retain the native result, hard-hint assignments, and UNKNOWN assignments | Emit hard hints where possible and UNKNOWN elsewhere under `native-ecapa-unavailable-v1` |
| `acoustic` | Rerun the common stack with acoustic-v2 speaker identity and retain ECAPA provenance | Rerun acoustic identity and attach an unavailable-ECAPA summary |
| `external` | Use valid external labels when present; otherwise retain the insufficient ECAPA report | Use valid external labels when present; otherwise return an error |
| `error` | Return an inner-diarizer error | Return an error |

This asymmetry is intentional: an absent external result cannot be reported as
successful external fallback. A completed but insufficient ECAPA run remains
inspectable, whereas an unavailable representation cannot supply a native
ECAPA result.

## 9. Scoring

The retained evaluation authority is `diarization-scorer-v5`. Low-level
metric helpers are not sufficient evidence by themselves: a retained verdict
must be produced by `score_diarization_documents` from these exact versioned
documents:

| Document | Schema identity |
|---|---|
| Reference | `diarization-reference-v2` |
| Hypothesis | `diarization-hypothesis-v2` |
| Configuration | `diarization-scorer-config-v2` |
| Result | `diarization-score-result-v2` |
| Corpus manifest | `diarization-corpus-manifest-v1` |
| Leakage audit | `diarization-leakage-audit-v1` |

All time geometry in those documents uses integer milliseconds. Documents
must use canonical interval and identifier order; corpus recording/split keys
are strictly increasing so duplicate entries within one split fail closed.
Unknown fields, unsupported versions, zero-duration calls, out-of-bounds
intervals, missing reference labels, confidence outside `[0, 1]`, confidence
attached to an unknown label, ambiguous overlapping hints, and non-canonical
ordering fail closed with `diarization.scorer.*` reason codes.

The reference and hypothesis identify the same opaque recording and exact
duration. They contain timed speaker turns but deliberately contain no
filesystem path, media URI, transcript text, or free-form provenance field.
Reference v2 may additionally contain aligned word intervals whose `word_id`
is an opaque non-lexical annotation identity. The interval and reference
speaker are sufficient to score speaker attribution; lexical tokens are
forbidden from retained evaluation inputs. Reference turns are labeled;
hypothesis turns may use an absent label for an honest unknown. Hypothesis v2
may carry the complete bounded `speaker-count-estimate-v2`. Assignment
confidence, count uncertainty, and overlap suspicion remain separate
observations.

### 9.1 Frozen metric policy

DER is computed after maximum-overlap one-to-one speaker permutation:

```text
DER = (missed speaker-time + false-alarm speaker-time
       + speaker-confusion time) / reference speaker-time
```

The report keeps all three components separate. Overlapping reference speakers
contribute speaker-time independently. JER is the mean per-reference-speaker
Jaccard error after the same mapping. Overlap F1 is computed directly as
`2 TP / (2 TP + FP + FN)`, so a corpus with reference overlap and no predicted
overlap has defined F1 zero even though precision alone is undefined.

`speaker_boundary_collar_ms` is the half-width removed around each reference
speaker change for DER/JER and associated attribution metrics. It does not
erase the change from the change-point task. `overlap_policy` explicitly
includes or excludes reference-overlap regions; no default hidden in a corpus
adapter may change that policy. Corpus-defined ignored regions are also
removed and contribute to the reported ignored duration.

The same result separately reports:

- union speech miss, false alarm, and speech-activity error rate;
- one-to-one change precision, recall, F1, and mean absolute timing error under
  the named `change_boundary_collar_ms`;
- conditional speaker-attribution accuracy and raw speaker-count error;
- proper multiclass count-posterior Brier score, finite negative log
  likelihood plus an explicit zero-reference-probability flag, concrete
  top-k coverage, a deterministic credible set that may retain unresolved
  mass, entropy, and calibration authority; a reference count outside the
  concrete posterior support still contributes its missing target-class term
  to the Brier score rather than disappearing from the outcome space;
- scored occupancy per anonymized hypothesis label, effective speaker count,
  phantom-label count, dominant-label share, UNKNOWN share, recurrence, and
  per-reference recall-collapse diagnostics;
- transcript-free aligned-word speaker counts and word diarization error rate
  (WDER), with the same ignored-region, collar, and overlap policy as the
  duration score;
- duration-weighted overlap precision, recall, and F1;
- hard/soft hint adherence, contradiction, unknown duration, and hard
  violation duration;
- known-speaker coverage and selective risk, so abstaining everywhere cannot
  look accurate;
- duration-weighted Brier score and fixed-bin expected calibration error,
  always accompanied by confidence coverage;
- wall time, audio duration, real-time factor, and peak RSS, kept separate from
  accuracy.

Reference, hypothesis, configuration, and result hashes use canonical compact
JSON and SHA-256. `result_sha256` hashes the complete result with that one field
temporarily empty, making tampering independently detectable. Repeated scoring
of identical canonical inputs must serialize byte-for-byte identically.

Speaker-count exactness is not a collapse test. A hypothesis can emit exactly
the reference number of label names while assigning nearly all speech to one
label. Scorer v4 therefore independently declares dominant collapse when a
multi-speaker reference crosses the configured labeled-share threshold, and
reference collapse when any mapped reference speaker falls below its
configured attribution recall. Labels below
`minimum_effective_occupancy_ms` do not count as effective or phantom
speakers. UNKNOWN share is measured over hypothesis speaker-time
(`UNKNOWN / (labeled + UNKNOWN)`), so false alarms cannot make it exceed one.
All thresholds are integer millionths or milliseconds in the self-hashed
configuration.

Reference labels that occur only inside excluded scoring regions do not enter
the reference-collapse count or minority-recall diagnostic. Hypothesis labels
seen only there remain visible with zero scored occupancy, but are neither
effective nor phantom speakers.

### 9.2 Corpus manifests and leakage

`diarization-corpus-manifest-v1` is path-free and contains only opaque corpus,
license, recording, source-call, speaker, derivation, augmentation, and
enrollment identities plus `train|development|test`. Identifier validation
rejects path separators, traversal, control characters, media extensions,
transcript markers, and common private-download markers.

The deterministic leakage audit checks every pair of different splits for:

- duplicate recordings;
- clips sharing an origin call;
- known speakers appearing across splits;
- shared derived or mixture ancestors;
- augmentations of the same source;
- shared or cross-linked enrollment recordings.

Findings expose only validated opaque IDs and machine-stable categories. A
passing audit and its self-hash are mandatory inputs to any corpus tuning or
held-out comparison.

Synthetic audio proves arithmetic and invariants only. Corpus accuracy requires
retained real multi-speaker ground truth, license provenance, exact scorer
configuration, and output hashes.

## 10. Evidence and rollout

Internal decision evidence can contain:

- normalized input, hint document, algorithm, feature, weights, calibration,
  parameter, and loss-matrix hashes;
- change-score components and selected boundaries;
- profile quality summaries and anchor use;
- selected-solution clustering merge trace, speaker-count candidates, cap
  pressure, and temporal refinement settings;
- assignment margins, confidence, unknown/overlap decisions, and fallback;
- stage duration, RTF, allocations or peak memory, cancellation, and errors.

Retained public evaluation evidence is stricter: it contains only path-free
aggregate counts, metrics, reliability bins, threshold/collar summaries,
configuration hashes, stage locks, and performance totals. It never retains a
per-recording boundary, score, feature, identifier, filename, transcript, or
audio excerpt. A separately retained path-free public bundle is record-level
scorer input and may contain validated opaque public recording/speaker IDs and
reference timestamps; it contains no source path, filename, audio, transcript,
or acoustic feature.

Focused unit/e2e proof, corpus DER/JER proof, broad Cargo proof, and performance
certification are separate authority states. Rollout follows
Shadow -> Validated -> Fallback -> Primary -> Sole. A hard-hint violation,
privacy regression, non-deterministic replay, contract error, or supported
quality/performance regression triggers rollback.

The runtime meaning of those stages is fail closed:

| Stage | `auto` behavior |
|---|---|
| Shadow | Acoustic output is not exposed |
| Validated | Contract proof may accumulate, but acoustic output is not exposed |
| Fallback | Use verified external speaker evidence when present; otherwise acoustic |
| Primary | Prefer acoustic |
| Sole | Admit only acoustic |

An invalid rollout environment value resolves to Shadow and is reported as an
invalid configuration. An explicit acoustic request is a caller decision and
bypasses only the `auto` admission gate; it does not bypass validation,
resource, privacy, or fallback rules.

## 11. Privacy and corpus handling

Schema v5 may persist turns, hint audit rows, typed speaker-count outcomes,
per-hint dispositions, and privacy-safe profile summaries inside SQLite. It
does not persist raw PCM, frame features, Fourier spectra, cepstra, reusable
speaker vectors, the CLI hint-document source path, or corpus metadata.
`persist_profiles` records explicit consent but does not expand the v5 storage
surface; reusable vectors require a separately reviewed schema and retention
policy.

Canonical result JSON, robot output, and persisted typed reports may expose
only content-free ECAPA provenance: provider version, the public expected and
loaded package digests, the stable verified-package load source
(`package_verified`),
availability status, aggregate embedded/zero-padded/skipped tracklet counts,
and stable reason codes. Cache warmth is deliberately excluded from the typed
report so identical input, request, and model bytes do not produce different
authoritative output. A separate external cache probe verifies miss, hit, and
corrupt-package invalidation behavior without changing report bytes.
They must never expose a model path, embedding coordinates, PCM, filterbank or
other feature values, tensor values, or per-tracklet neural payloads.

The privacy-safe `diarization-operational-partition-v2` summary may retain its
method, selected count, confidence, calibration digest, and authority. Its
confidence is operational evidence, not automatically a calibrated posterior:
`FixedSafeUncalibrated` and `DevelopmentUncertified` authority remain explicit
non-certification states.

Private evaluation material must be read in place and must never be copied into
the repository, fixtures, remote build inputs, JSONL snapshots, Beads, logs, or
documentation. Its filenames, transcript text, hashes, durations, and derived
metrics are not admissible public evidence. Only hermetic synthetic fixtures
may be retained here until a redistributable, provenance-cleared corpus is
approved.

### 11.1 Local confidential evaluator

`confidential-diarization-evaluation-manifest-v2` is deliberately different
from the path-free public corpus manifest. It is a local input document that
contains absolute audio, reference, and hypothesis paths and therefore must
remain outside the checkout. Its Rust representation supports deserialization
only: it has no `Debug` or serialization surface.

The `diarization-eval` command:

- discovers the canonical project root from the current checkout;
- requires a canonical absolute input root disjoint from that project;
- resolves every manifest source through symlinks and requires it to remain
  beneath the input root;
- requires a new absolute `.json` output whose canonical parent is outside the
  project;
- caps manifest and scoring-document reads and streams audio hashing with
  bounded memory;
- maps every path, parse, and I/O failure to a stable
  `confidential_evaluation.*` error that never contains a source basename or
  source value;
- checkpoints cancellation before the manifest, every recording, every
  streaming audio-hash block, and the final write, leaving no aggregate when
  cancelled;
- writes only
  `confidential-diarization-evaluation-aggregate-v2`.

The aggregate contains micro/macro accuracy, change, raw count error,
count-posterior proper scores and coverage, occupancy-collapse totals,
transcript-free word-attribution totals, overlap, calibration, and optional
performance summaries plus opaque content/config fingerprints. Posterior
unavailability, unresolved selections, and zero reference probability remain
separate counts. It contains no per-recording row, path, filename, transcript,
timestamp, speaker/recording identity, feature vector, or excerpt. Repeated
evaluation of identical content is byte-stable apart from the caller-chosen
external filename, which is never serialized.

### 11.2 Public and user-licensed corpus adapter

`diarization-corpus registry` emits
`public-diarization-corpus-registry-v1`, the frozen acquisition and conversion
contract. The initial registry spans:

| Key | Conditions | License |
|---|---|---|
| `ami-scenario-v1` | English meetings, close/far microphones, overlap | `CC-BY-4.0` |
| `aishell-4-openslr111-v1` | Mandarin meetings, arrays, noise, short turns, overlap | `CC-BY-SA-4.0` |
| `voxconverse-v1` | in-the-wild backgrounds, same-gender and overlapping speech | `CC-BY-4.0` with original-video copyright notice |
| `callhome-american-english-2e-v1` | dyadic 8 kHz telephone/channel mismatch | operator-held LDC user agreement |

The registry records an authoritative source URL, license URL and exact
acknowledgement ID, expected external layout, conversion contract, integrity
policy, condition tags, and split policy. A registry entry documents what the
operator must acquire; it does not download data, accept a license, or confer
rights. The LDC entry is usable only by an operator who already has lawful
access.

`diarization-corpus build` consumes
`public-diarization-corpus-input-v2` from an absolute root outside the checkout.
Every selected input is a relative path under that canonical root. Symlink
escapes, traversal, absolute descriptor paths, wrong SHA-256 values, unexpected
WAV sample rate/channel count, invalid selected channels, malformed RTTM, and
unmapped speakers fail closed. Out-of-bounds turns also fail closed except for
one corpus-specific conversion rule: an official VoxConverse turn that begins
before WAV EOF and ends no more than 100 ms after EOF is clipped to EOF, with
the clipped-turn count and clipped milliseconds retained in path-free evidence.
The pinned archive audit admitted 61 test turns (zero development turns), with
a maximum 92 ms overrun; counterexamples at 101 ms or beginning at EOF remain
rejected. This is a frozen source-normalization rule, not a scorer-tolerance
change. RTTM is the deliberately small interchange surface: exactly ten
`SPEAKER` fields, plain decimal seconds, one selected recording/channel, and an
explicit source-label to path-free speaker-ID map. Concurrent different-speaker
turns are preserved and marked as overlap. Ignored regions remain explicit
scorer inputs.

Each recording may bind an optional external
`public-diarization-word-annotation-v1` document by relative path and exact
SHA-256. It contains only the recording identity and canonically ordered
opaque word IDs, integer-millisecond intervals, and reference speaker IDs.
The adapter validates every word against active reference speech, caps
per-recording and corpus totals, and never imports lexical text.

The generated `public-diarization-corpus-bundle-v3` contains the path-free
manifest, canonical reference documents, media/annotation/reference SHA-256
values, optional word-annotation SHA-256 values and counts, checked WAV
geometry, annotation-tail normalization counts, and a passing self-hashed
leakage audit. It never contains local
paths, URIs, transcripts, or media bytes. The output is created once in a
directory outside both the checkout and input root; source media is never
copied. Entire output file names use lowercase ASCII letters, digits, period,
underscore, and hyphen and end in the exact suffix `.json`; uppercase names are
rejected, and a handle-relative directory preflight rejects any existing
ASCII-case-fold sibling. This makes ordinary case-sensitive and
case-insensitive collisions fail consistently; exact-name no-clobber
publication remains the atomic creation guard, and concurrent mutation by the
effective user remains outside the threat boundary. On Linux, Android, and
Apple platforms, complete JSON is serialized into a
private staging file relative to an identity-bound directory handle, fsynced,
and atomically published with no-clobber rename semantics. The terminal parent
must not be a symlink, must be owned by the effective user, and must not be
group/world writable. Requested and canonical terminal-directory identities
are checked throughout; the effective user is trusted because it can also alter
a final artifact after publication. Ancestor rename authority, ACLs, privileged
mount changes, and arbitrary mount aliases of strict checkout/input descendants
are outside this boundary, so callers must use a privately controlled output
path and filesystem. Other platforms fail closed with
`public_corpus.output_platform` before corpus materialization.
Before the no-clobber rename commit point, the run creates no final-name
artifact and never modifies an already existing final-name entry. After
permission verification, cancellation or later failure truncates and
synchronizes the private staging inode to a zero-length mode-0600 marker; this
is not a secure block erase. The writer explicitly applies
mode 0600 after creation and verifies that the inode is a regular file owned by
the effective user with exactly those access bits before serializing payload
bytes. POSIX/NFSv4 ACLs and mount-level permission synthesis are outside this
mode-bit check; the caller must select an output filesystem that does not grant
broader access through those mechanisms. A failure while establishing or
verifying permissions leaves an empty marker without a mode guarantee. Failure to complete a payload scrub
reports `public_corpus.output_cleanup_uncertain`. After that commit point, a
failed identity or directory-sync
confirmation reports `public_corpus.output_commit_uncertain`; the final-name
artifact may already exist and is never truncated. An ambiguous rename result
is also commit-uncertain and conservatively preserves the held inode; only an
authoritative no-clobber conflict permits pre-commit scrubbing. A
bundle/evidence pair is staged before either publication, but the two
independent final-name renames are not a cross-file transaction. Successful
evidence publication is therefore the explicit completion signal. If the
bundle commit succeeds and evidence publication does not, a retry may resume
only when the existing bundle remains an owner-only regular file whose inode
identity and complete pretty-JSON bytes exactly match the newly recomputed,
canonically verified bundle. The retry opens that bundle read-only and creates
only the still-absent evidence destination through the same no-clobber commit.
A byte mismatch, identity change, permission change, non-regular destination,
or existing evidence file fails closed; recovery never deletes, truncates, or
overwrites either final destination.
The path-bearing descriptor type is deserialization-only and has no `Debug` or
serialization implementation.

The current ablation evidence is
`public-diarization-acoustic-ablation-v8` with runner v8. Every split reports
the full count confusion matrix, exact/error quantiles, reference-count and
duration strata, posterior calibration summaries, collapse/occupancy
diagnostics, and optional micro/macro WDER. These additions do not promote a
candidate: the historical v7 development result in section 4.4 remains the
last retained verdict until a hash-locked v8 development run passes.

`diarization-corpus sidecar-study` is a sibling evaluation command, not an
extension of ablation v8 or a normal diarization mode. It consumes the same
absolute external input root and path-bearing descriptor, creates a new
path-free bundle plus a new aggregate-only sidecar artifact outside both the
checkout and input root, and refuses to overwrite either output. The request
type has no `Debug` or serialization implementation. CLI `Debug` output
redacts the input root, descriptor, bundle output, evidence output, and optional
locked-development path. Successful stdout is the exact pretty-JSON byte
sequence retained in the evidence file, including its terminal newline; the
artifact's `result_sha256` remains the canonical self-hash with that field
cleared, not a hash of the pretty file bytes.

Both stages parse and hash the complete descriptor and audit its path-free
cross-split metadata. `--stage development` then opens only development WAV,
RTTM, and optional word-annotation bytes and must not receive a lock.
`--stage certification` requires `--locked-development-evidence`; it first
verifies that artifact's result and deterministic-accuracy hashes, protocol,
candidate order/configuration hashes, separately fitted boundary and lagged-pair
calibrations and their provenance, selected lane, recomputed gates, ranking,
and disposition. It then hashes the current descriptor, requires the descriptor
and protocol identities to match the development lock, and only afterward opens
held-out WAV or annotation bytes. A failed development selection therefore
cannot unlock certification. The materialized bundles are split-specific, so
their bundle hashes differ across stages; the common descriptor and protocol
hashes form the cross-stage binding. The aggregate study evidence from either
stage retains no source path, filename, recording/speaker identifier,
timestamped observation, raw feature value, audio, or transcript. The separate
path-free public bundle does retain the selected split's validated public
reference rows, including recording/speaker identifiers and timestamps needed
by the scorer; it retains no source paths, filenames, audio, transcript text,
or sidecar feature values. Cancellation is checked throughout preparation and
evaluation, and evidence is written only after canonical verification.
None of these CLI or artifact paths construct a sidecar during transcription or
change the default-off acoustic-v2 path.

`diarization-corpus compare-models` is a separate development-only diagnostic.
One invocation runs the same validated mono signed-PCM 16 kHz WAV bytes and the
same frozen scorer over `native_acoustic`, `native_ecapa`,
`native_ecapa_fused`, the release-bound `native_sortformer`, and the pinned
operator-installed `external_sortformer` oracle. The headline uses inferred
speaker count; a reference count above the Sortformer four-speaker contract is
declared capacity-ineligible and is never passed to either Sortformer lane.
Consecutive sorted observations follow a ten-row balanced Williams schedule,
and `order_balance_complete` is true only after a complete schedule. Every
declared lane is retained as completed, skipped, or failed with a stable
reason. Protocol v7 binds each complete effective native request, the ordered
payload-free outcome taxonomy, the external Sortformer adapter version and
executable SHA-256, and the protocol body to a pinned canonical SHA-256
identity; changing any of them requires a new version-and-digest pair.
The command requires eight native Rayon workers to match the pinned Sortformer
intra-op thread count and applies the same frozen 1800-second whole-attempt
limit to every lane. Each lane/observation runs in a fresh bounded worker
process. Its request binds the executable, source WAV, normalized PCM,
reference, scorer, protocol, and applicable model artifacts. Cancellation and
timeout terminate the worker's complete process group, including a nested
external adapter, and a live recursive-descendant probe measures the process-
tree cancellation path. A group-signal error is not itself success: it is
accepted only if direct-root reap and the subsequent absence probe certify that
the complete group is gone.
Before reading that request, the process-group root authenticates an inherited
kernel-pipe capability against its direct parent and starts a liveness watcher.
The parent retains the only write end. If the parent crashes or is killed, EOF
makes the worker kill its complete group, including nested adapters; platforms
without this capability fail before an observed worker is launched.

The aggregate comparison evidence is `diagnostic_only`,
`development_uncertified`, forbids a superiority claim, and records
`production_route_changed=false`. Absolute DER/JER values use the project
scorer and must not be compared directly with published md-eval numbers without
an explicit scorer-equivalence probe. Timing is retained for deployment
observation only. Parent-observed time for every lane includes process launch,
bounded IPC, identity validation, audio decode, model load, inference, output
validation, scoring, post-run identity validation, and resource-probe parsing.
Those fresh-process scopes are cross-lane comparable. The artifact reports an
approximate sampled maximum of the concurrent RSS sum across the complete
worker process group, including nested subprocess adapters. Sample starts are
separated by at least 50 ms and each probe is bounded; probe work makes the
actual start-to-start cadence longer, so this is neither an exact high-water
mark nor an exactly-50-ms sampler. Cancellation, timeout, and output-limit
checks precede observation. The retained cancellation probe exercises this same
observer path. On Unix, cleanup reaps the root and confirms that the owned
process group disappeared; cleanup failure overrides the nominal lane result.
A fast exit without a sample is explicitly unavailable, while repeated loss of
a still-live group fails the resource probe. Platform scans check cancellation
and the enclosing attempt deadline. A matched live Linux group member cannot be
silently omitted: it must expose a valid RSS field. A complete zero-only scan is
missing rather than a measured zero, and repeated zero-only scans fail closed.
Recursive-process cancellation
latency is retained separately; unavailable platform probes remain explicitly
unavailable rather than becoming zero. The comparison command downloads no
model. `native_sortformer` uses only the release-bound cache installed by the
explicit `fw pull sortformer` command; unavailable ECAPA, native Sortformer,
or external-adapter components produce typed skips.
The command does not alter transcription routing or the default-off acoustic
sidecar.

`verify_public_model_comparison_bundle_identity_pair` is deliberately an
artifact-identity verifier. It validates each artifact structurally and checks
the shared corpus, source, descriptor, bundle, split, and recording-count tuple.
Aggregate-only evidence omits the per-record normalized-input and outcome rows,
so this verifier cannot recompute the observation-set commitment or aggregate
metrics. Derivation proof requires source reconstruction and a fresh comparison
run.

ReDimNet2-B2 is a compact evaluation-only representation provider, not a
segmentation, overlap, speaker-count, or clustering authority. Its pinned
upstream identity is PalabraAI/redimnet2 v1.0.0 at peeled revision
`5294667e806ac3b0f27abc301a114ef132b64b42`, with checkpoint size 15,897,450
bytes and SHA-256
`0545a29679a87fe1c662d2bbd05e3b3fe0d1b392832729abaa135e4079a2f77a`.
The checkpoint configuration, not paper prose, is authoritative: 72 mel bands,
six reshape stages, a 1,440-channel weighted 1D path, attentive-statistics
pooling, and a raw 192-dimensional embedding that is not L2-normalized. The
model contains 3,677,760 parameters: 3,676,320 trainable plus the frozen
1,440-element first-stage weighting tensor.

`scripts/export_redimnet2.py` is the only accepted raw-checkpoint boundary. Its
current exporter and conversion-receipt schemas are v2; the synthetic truth
tensor contract remains v1. It requires the exact Python/package tuple and
14-file source manifest. Before
import, it requires the executable `redimnet2` package-file census to equal the
manifest exactly, rejecting unmanifested bytecode, native extensions, regular
files, special entries, and symlinks; it disables bytecode writes, invalidates
import caches, and rejects preloaded ReDimNet modules. It loads the
checkpoint with `weights_only=True` from an already hash-verified owned byte
buffer, strictly loads the upstream graph, drops exactly 68 scalar int64 batch
counters, and retains 661 finite contiguous f32 tensors containing 3,918,794
elements. The receipt binds the interpreter binary, Python implementation and
cache tag, package RECORD/METADATA digests, and a path-free file-set commitment
after verifying every hashed RECORD entry. It also binds Torch
Git/build/configuration identity. A Python audit hook denies socket and
child-process events; the exporter verifies source-module provenance and
source/exporter stability before
publication. It creates a new external mode-0700 directory and three mode-0600
files through a stable directory handle using no-follow and exclusive creation,
then verifies inode identity and fsyncs the directory; repository-contained
input or output paths fail closed. Neither the Rust runtime nor this exporter
downloads a model.

The canonical converted package is 15,745,544 bytes with SHA-256
`d41a729f5ef008d70c6d6bf4ab7ca27e299a478ff665665a4e31afff7f46ddeb`.
The synthetic oracle package is 8,828,392 bytes with SHA-256
`21042537873c3dacafafd134d7c9e296318458f55f1a429c00bc9542f95f3238`.
It binds waveform, frontend, stem, stage 0, stage 3, final weighted 1D,
backbone 2D, attentive pooling, pooled batch normalization, raw embedding, and
consumer-normalized embedding seams. Five source replays at each of one and
eight PyTorch threads were deterministic within a thread count. The cross-
thread source floor peaked at max-absolute `3.147125244140625e-4` in the
frontend and relative-L2 `1.4254854456195043e-5` in the final weighted path;
the receipt retains every seam floor, ceiling, absolute headroom, floor
multiplier, and pre-native rounding rationale. The manual terminal seam is
accepted only when it is bit-exact to the authoritative upstream `forward`.
Known upstream deprecation warnings and the one pinned initialization message
are captured and retained only as path-free code, count, byte, and message
digests; unknown warnings or console output fail. The
initial frontend ceiling of `2e-5`/`2e-6` was rejected before native work and
is preserved in the receipt rather than erased. Two independent final hardened
exports were byte-identical, including the path-free receipt at SHA-256
`e4e5aab1838dd386895425acc11e3405191e30ce2111c313c2734bfc2bccd77e`.

The v1.0.0 tag contains no top-level repository license file even though a
later main revision has an MIT file and individual pinned sources carry MIT or
Apache notices. Model-weight redistribution scope is therefore unresolved.
The receipt fixes distribution status to `operator_local_no_release`; no
converted package, checkpoint, truth tensors, feature values, local paths,
audio, transcripts, or biometric vectors may enter Git, a public release, or
runtime evidence. This licensing boundary does not prevent local parity and
comparison work, but it does prevent artifact publication or a distributable
product claim.

**License-scope investigation (bd-y4ip.15, 2026-08-23, read-only evidence
pass).** Facts, each retrieved from the upstream repository
(`github.com/PalabraAI/redimnet2`) on this date:

- The `v1.0.0` tag tree (`README.md`, `assets/`, `hubconf.py`, `redimnet2/`,
  `requirements.txt`) contains no license file. Its release body reads only
  "pre-trained redimnet2 weights vox2-dev (ptn and lm)" (published
  2026-03-04T14:25:10Z; it remains the repository's ONLY release).
- Release asset `b2-vox2-lm.pt` — the pinned checkpoint (15,897,450 bytes,
  SHA-256 `0545a29679a87fe1c662d2bbd05e3b3fe0d1b392832729abaa135e4079a2f77a`)
  — was uploaded 2026-03-04T14:25:03Z and never re-published afterward;
  no asset postdates the license addition.
- Upstream added `## License / MIT` to the README on 2026-07-06
  ("Update public ReDimNet2 release docs and hub loading", commit
  `608196213116`) while that same README links the v1.0.0 checkpoint assets
  as its official download path, then added the top-level MIT LICENSE file on
  2026-07-09 (commit `2a8d15f65b1d`, "Add MIT License to the project").
- The repository description states it contains "the official implementation
  and pretrained weights". No issue or discussion mentions licensing.
  There is no official HuggingFace mirror (third-party mirrors already label
  conversions MIT; noted, not authoritative).

Assessment: this demonstrates clear maintainer INTENT that the project —
including its published checkpoints — is MIT, but it is not an authoritative
statement that the MIT grant extends retroactively to the March 2026 release
assets, which were distributed before any license declaration existed.
Circumstantial intent does not meet the bead's bar of "authoritative upstream
clarification or another legally sufficient artifact."

Disposition: distribution status stays `operator_local_no_release`. A
maintainer inquiry has been drafted (retained in the tracker comment on
bd-y4ip.15) for the operator to send from an operator identity; upon a
sufficient reply or equivalent artifact, update THIS section first, then the
receipt status, then bd-y4ip.3.

The AMI adapter enforces the corpus site's scenario-only training,
development, and unseen-test meeting-family split. Other corpora use an
external descriptor whose exact bytes are frozen by SHA-256, followed by the
same cross-split speaker/origin/derivation/augmentation/enrollment audit. Any
tuning or comparison must name the bundle hash before looking at held-out
results. Changing a source file, mapping, ignored region, or split produces a
different bundle identity.

CI uses only generated WAV/RTTM fixtures to prove adapter arithmetic, malformed
input handling, overlap, ignored regions, channel/sample-rate checking,
license acknowledgement, cancellation, and byte-stable replay. Full accuracy
certification points the same command at externally acquired data and retains
the resulting bundle and scorer outputs outside the repository.

### 11.3 Adversarial and metamorphic acoustic recipes

`src/adversarial_corpus.rs` is the public-safe failure-reproduction substrate.
It generates finite in-memory PCM from `adversarial-synthetic-call-v1` recipes;
it does not read a path or serialize the resulting samples. Synthetic profiles
contain only oscillator, amplitude, stationary coloration, and stereo-position
parameters. Turns contain only a numeric profile index, integer time range,
gain, pitch movement, and a playback condition. They do not contain words,
names, demographic labels, recordings, or biometric templates.

The v1 challenge registry contains one deterministic seed for each required
regime:

| Source or transform family | Metamorphic contract |
|---|---|
| Gain/distance imbalance | Speaker labels remain permutation-equivalent |
| Stationary EQ/muffling and band limitation | Labels remain stable; quality may degrade |
| Resampling and quantization | Timing is unchanged; consistency error is measured |
| Clipping, noise, reverb, and interruptions | Labels remain stable or become explicitly uncertain; no invented identity |
| Leading/trailing silence | Every reference boundary shifts by the exact leading duration |
| Rapid turns and long turns | Source geometry is authoritative |
| Similar pitch and within-speaker voice-state shifts | Pitch alone may neither merge nor split an identity |
| Within-speaker channel shifts and loudspeaker playback | Channel evidence may create a subprofile, not a new voice |
| Stereo channel swap | Speaker output is channel-permutation invariant |
| Controlled overlap | Overlap evidence increases without fabricating a third identity |

Every `adversarial-transform-plan-v1` binds the exact input PCM hash and
contains at most 64 bounded integer-parameter steps. Source authority is either
synthetic or public-licensed; the latter requires a lowercase SHA-256 of an
external acknowledgement or license record, never its path or text. Execution
checkpoints cancellation between steps, rejects non-finite or malformed PCM,
caps allocations, and emits `adversarial-transform-evidence-v1`. Its graph
records only the plan hash, per-step recipe hash, input/output audio hashes,
and expected relationship. It contains no audio, path, filename, transcript,
embedding, speaker name, or per-frame feature.

Pipeline harnesses provide aggregate fingerprints for input, normalization,
speech mask, feature extraction, change detection, clustering, projection,
and scoring. Comparison returns the first differing or missing stage. A stable
regression classification is an uppercase error code plus that stage.
Deterministic delta minimization removes transform subsequences only when the
caller-supplied evaluator reproduces the exact same classification twice;
disagreement fails as a non-deterministic classifier. The result retains
original step indices and an evaluation count. The minimized artifact is a
recipe, not an accuracy certificate.

An identity-preserving recipe does not promise that an imperfect candidate
will pass. Its purpose is to turn violations into small, reproducible public
regressions. Promotion still requires the frozen scorer, public corpus gates,
unseen held-out evidence, and the rollout authority described above.

### 11.4 Stage-aware external differential oracles

`src/differential_oracle.rs` is an explicit developer-only bridge to
operator-installed diarization systems. It does not add a Cargo dependency,
does not run during transcription, and does not make any external system part
of the shipped native decision path. Its stable registry spans three
architecturally different families:

- cascaded pipelines: pyannote and NeMo spectral clustering;
- Bayesian HMM refinement: VBx;
- end-to-end/attractor systems: EEND, DiaPer, and Sortformer.

Each registry entry names a dedicated
`FRANKEN_WHISPER_*_ORACLE_BIN` override and a default adapter executable. The
operator supplies that adapter. A version probe receives
`--franken-whisper-diarization-oracle-version --protocol
franken-whisper-diarization-oracle-protocol-v2` and must emit one strict
`franken-whisper-diarization-oracle-version-v2` JSON object on stdout. A run
receives `--franken-whisper-diarization-oracle-run`, the same protocol flag,
an external `--audio` path, and a lowercase SHA-256 `--recording-key`. It must
emit one strict `franken-whisper-diarization-stage-document-v1` object on
stdout. Arguments are never retained, and neither stdout nor stderr content is
copied into the report.

The Sortformer adapter has an additional fail-closed model contract. It is
exactly `nvidia/diar_streaming_sortformer_4spk-v2.1` at repository revision
`fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`, with the operator-installed
artifact independently hashed locally. Hugging Face LFS metadata pins the
471,367,680-byte `.nemo` artifact to SHA-256
`8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`.
The locally computed hash must equal that exact value; size plus an arbitrary
well-formed digest is not accepted. The frozen input is mono 16 kHz PCM16 WAV.
Every PCM sample is decoded during validation, the sample-derived duration must
be within 79 ms of the stage-document duration, and the audio hash is checked
before and after the adapter run. The adapter executable is resolved once and
that exact absolute executable is used for both probes.

The model emits four arrival-ordered activity slots on 80 ms frames. The
high-latency streaming profile fixes chunk/right-context/FIFO/configured-cache-update/cache
lengths to 340/40/40/300/188 frames (30.4 seconds nominal input-buffer latency).
At the pinned NeMo revision, parameter validation only warns that the configured
300-frame cache-update period is shorter than the 340-frame chunk; it does not
mutate the setting. FIFO updates use
`min(max(configured, chunk - fifo_capacity + current_fifo), current_fifo + chunk)`.
For full chunks starting with an empty 40-frame FIFO this moves 300 frames first,
leaves 40 frames queued, and then moves 340 frames in steady state; tail behavior
depends on the current chunk and FIFO lengths. The remaining profile fixes batch
size one, inferred count up to four, and untuned onset/offset 0.5 with zero padding
and minimum-duration filtering. The canonical adapter labels are
`speaker_0` through `speaker_3`; arrival order is determined by each label's
minimum onset and tied first onsets are allowed. Turns start and end on 80 ms
frames, except that an end may equal the document duration. NeMo expands each
80 ms prediction into eight 10 ms VAD frames; if activity remains live at EOF,
its binarizer reports the final repeated subframe and therefore an end with a
70 ms residue. The adapter may canonicalize only that residue to the physical
document duration, and only when the gap is at most 79 ms. All starts and every
other non-document end remain strict 80 ms grid points. Sortformer overlap is
represented only by concurrent labeled turns, never by
`overlap_suspected`. Speech activity, overlap, and speaker-change boundaries
must exactly equal the O(n log n) event-sweep derivation from the final turns.

The accepted `franken-whisper-sortformer-oracle-v3` adapter must verify the
installed `nemo-toolkit` distribution's `direct_url.json` commit and requested
revision rather than repeating the expected tool revision as an unchecked
constant. It also fails closed on drift from the qualified Python 3.12.12,
NeMo `3.1.0+40ace43c7c`, PyTorch 2.7.1, torchaudio 2.7.1, and NumPy 2.4.6
environment. BLAS and CPU-feature fields are derived from installed runtime
configuration, and the adapter rechecks the raw 340/1/40/40/300/188 streaming
profile plus the derived first/steady FIFO-pop schedule after NeMo's parameter
validator runs. The host binds the
resulting version document and exact adapter executable hash into evidence.
This operator override is a declared trust boundary, not remote attestation.
The host validates the version schema and internal hashes and allowlists the
reviewed executable digest, but it cannot remotely attest that the external
runtime honored its source-level checks. The same-invocation comparison
protocol separately binds the exact adapter version and executable digest, so
changing operator code cannot silently reuse an older protocol identity.
Acceptance of a self-consistent version document alone must not be inflated
into certification of its implementation.

The immutable native conversion receipt records the v2 adapter digest that was
used when that package was produced. That historical conversion provenance is
deliberately distinct from the current v3 runtime comparison adapter identity:
revising output canonicalization must not rewrite or invalidate an already
published conversion receipt, while protocol v7 still prevents a runtime
adapter revision from silently reusing prior comparison evidence.

The frozen execution profile is CPU-only float32, with autocast disabled,
quantization disabled, deterministic algorithms enabled, batch size one, zero
data-loader workers, eight PyTorch intra-op threads, and one inter-op thread.
Changing any of those values requires a different contract hash and evidence
row. The adapter also computes a path-free
`sortformer-runtime-fingerprint-v1` object containing the schema, Python, NeMo,
PyTorch, torchaudio and NumPy versions, BLAS backend, operating system, machine
architecture, CPU feature tier, device, dtype, autocast, quantization, thread
counts, data-loader worker count, and deterministic-algorithm state. The v2
version response carries that strict deny-unknown-fields object plus its
SHA-256. The host validates the frozen profile and normalized path-free tokens,
recomputes the digest, and retains only the digest in the report, so benchmark
rows with silent runtime drift cannot be treated as matched.

The v2 version document adds `model_contract_sha256`,
`model_artifact_sha256`, `model_artifact_bytes`, and
the structured `runtime_fingerprint` plus `runtime_fingerprint_sha256`. All are
mandatory for a successfully validated Sortformer probe. The retained report separately records
`expected_model_contract_sha256` even when no adapter is available and records
the observed contract/artifact/runtime attestations only after a valid version
probe. Non-Sortformer tools must omit all model-contract fields. The tool and
adapter versions are exact pins, so a model, NeMo source, runtime profile, or
adapter revision change is a new evidence row rather than an in-place
substitution. A non-PCM input, changed input bytes, mismatched contract,
malformed/nonfinite or cross-stage-inconsistent output, a fifth output slot, or
noncanonical label fails closed with a typed skip. A reference with five or
more labeled speakers is retained as `reference_model_capacity_exceeded`; it
is never removed from the declared comparison population or collapsed to four
speakers.

Contract and report hashes use `lexicographic-canonical-json-v1`: recursively
sort every object by key, emit compact JSON with no insignificant whitespace,
preserve array order, and use JSON scalar rendering. This is a deliberately
small cross-language encoding contract for the ASCII keys and integer/Boolean
model contract; it is not claimed to implement RFC 8785. The encoding version
is itself included in the hashed Sortformer contract, and the runtime
fingerprint uses the same encoding. Cancellation terminates the directly
invoked adapter and zero data-loader workers are part of the profile; full
descendant-process teardown and timing comparability remain qualification gates
for the same-invocation benchmark rather than claims made by this seam.

The canonical stage document has a duration and optional outputs for:

1. speech activity intervals;
2. opaque, non-lexical word timing IDs;
3. speaker-change boundaries;
4. opaque segment-to-cluster assignments;
5. overlap intervals;
6. final diarization turns.

Word identities must use bounded `w-` hexadecimal tokens and cluster segment
identities bounded `seg-` hexadecimal tokens. There is no field for transcript
text, a media path, a model path, a raw embedding vector, or a speaker name.
Intervals and counts are bounded; activity/overlap intervals must be ordered
and non-overlapping; confidences must be finite and in `[0, 1]`.

The comparator reports all six stages in that order. Activity and overlap use
exact integer-millisecond intersection-over-union. Word timing joins opaque
IDs and measures two-boundary collar recall. Change points use the frozen
one-to-one matcher. Cluster comparison uses contingency counts and pairwise
co-assignment, making it label-permutation invariant in linear-logarithmic
time rather than materializing all segment pairs. It also measures shared
segment coverage and requires matching geometry for each shared opaque segment
identity, so an adapter cannot appear equivalent by omitting or moving
anchors. Confidence availability and mean absolute confidence delta are
reported separately, but do not determine equivalence because independently
implemented tools need not calibrate confidence to the same scale. Final turns
use the frozen Hungarian speaker mapping and retain only label-free DER/JER
components. Missing stages remain explicitly missing; they are not converted
into errors or fabricated values. `earliest_divergence` is the first present
stage whose frozen diagnostic threshold is exceeded.

Parsing caps and comparison caps are intentionally distinct. A stage document
may be retained for diagnosis at the general safety limits, but the current
quadratic change-point/final-turn scorers refuse to compare any single native,
oracle, or reference document with more than 2,048 change points, 2,048 final
turns, or 32 distinct speaker labels (counting UNKNOWN as one label). This
prevents a schema-valid diagnostic from turning dynamic programming, atomic
interval construction, or Hungarian assignment into an unbounded CPU or memory
request. Larger rows require a future sweep/banded scorer with cancellation
inside the algorithm; they are not silently sampled or truncated.

An optional third stage document can help interpret a disagreement. Its only
categories are `reference_favors_native`, `reference_favors_oracle`,
`reference_tied`, and inconclusive/unavailable states. These are diagnostics,
not correctness certificates. Every report hard-codes:

```json
{"authority":"diagnostic_only","native_incorrectness_claim_permitted":false}
```

Missing binaries, nonzero exits, timeouts, incompatible versions, invalid
JSON, invalid geometry, and tool/recording identity mismatches create a clean
`skipped` report with a stable reason and failure stage. Cancellation remains
cancellation and kills the child rather than being misreported as a tool
result. Safe partial provenance is retained when available: tool family,
validated tool/adapter versions, executable hash, version/run stdout hashes,
audio hash, and input-document hashes. Paths, stderr, output content, labels,
word IDs, and local recording identities are not retained. Every report
self-verifies its authority, state invariants, stage ordering, configuration
hash, provenance hashes, and result hash before being written with
create-new semantics outside the checkout.

The CLI exposes only explicit development commands:

```bash
franken_whisper diarization-oracle registry
franken_whisper diarization-oracle run \
  --tool pyannote \
  --audio /absolute/external/audio \
  --native /absolute/external/native-stage.json \
  --reference /absolute/external/reference-stage.json \
  --output /absolute/external/differential-report.json
```

All input documents, media, and output reports must be absolute external
files. A missing `--reference` is valid and produces
`inconclusive_no_reference` for genuine disagreements.

### 11.5 Optional ECAPA model and numerical conformance boundary

`src/ecapa_conformance.rs` freezes the model/evidence contract and
`src/ecapa_inference.rs` implements its bounded safe-Rust ECAPA-TDNN forward
path. The orchestrator routes both explicit ECAPA engines through
`diarize_ecapa_pcm` and the common segmentation, constraints, count, temporal
UNKNOWN/overlap, label, and projection contracts. `ecapa` uses ECAPA
coordinates for speaker identity without acoustic channel evidence in pair
scoring. `ecapa-fused` authorizes separately bounded acoustic channel evidence
and five-lane coassociation consensus on that ECAPA identity path. It reports the
typed `EcapaFusedConsensus` operational method only when at least one selected
consensus merge joins a compatible ECAPA pair with valid channel dimensions. With missing
or mutually constrained channel evidence it underclaims generic
`ProbabilisticConsensus` provenance and the supported-profile redecode route is
ineligible. It can never claim the ECAPA-only `EcapaSpherical` method. The
evaluation-only supported-profile redecode route accepts only the exact
ECAPA-only/spherical or channel-proven fused/consensus pairs, and only when
every acoustic tracklet has a neural representation. Partial neural coverage
therefore leaves missing speech UNKNOWN except for immutable hard attribution
and makes the redecode candidate a deterministic no-op. Neither mode enters
`auto`, changes the acoustic default, downloads a model, or parses a framework
checkpoint at runtime. Public-corpus accuracy and calibration remain
development-uncertified.

The ECAPA development decision policy places its equal-loss different-speaker
boundary at cosine distance `0.80`. Robust final-assignment and held-out
validation separation begin at that same `0.80` boundary. Lane consensus and
temporal recurrence may require stricter evidence; they may never introduce a
hidden, weaker `0.70` separation gate. This bound is part of
`acoustic-clustering-probabilistic-v20-channel-evidence-bound-fused-consensus-development`; it
is a versioned conservative decision policy, not an accuracy-certification
claim.

The source is
[`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/eac27266f68caa806381260bd44ace38b136c76a)
at immutable revision `eac27266f68caa806381260bd44ace38b136c76a`,
under Apache-2.0. The `embedding_model.ckpt` identity is:

- 83,316,686 source bytes;
- SHA-256
  `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2`;
- 231 source entries: 200 inference `f32` tensors and 31 scalar `i64`
  BatchNorm `num_batches_tracked` counters;
- hyperparameter SHA-256
  `ecd11c44202b32edb72709dd1013a16f2f060ebee3438ae8a9f9fecb0666ecd2`;
- training-code revision
  `aa0185408025e80f6c748d2c7af7fa96958c2231`.

The model card says the model was trained on VoxCeleb1 and VoxCeleb2. That is
speaker-verification training over English web video, not calibration for
meeting diarization. Telephone, far-field, playback, muffling, accent,
language, overlap, and domain shifts require held-out validation. An embedding
is acoustic similarity evidence, never a gender, name, or person-identity
claim.

The versioned export protocol is
`franken-whisper-ecapa-export-v1`, implemented by the
`ecapa-tdnn-voxceleb-v1` profile in
`scripts/convert_to_safetensors.py`. Source checkpoint loading is a
development-only, out-of-process trust boundary. The profile accepts only the
checkpoint identity above and requires Python 3.12.12, NumPy 2.2.6, Torch 2.7.1,
and safetensors 0.5.3. It then does the following:

1. requires the exact 83,316,686-byte source hash and 231-entry census;
2. rejects every non-tensor or unexpected dtype, dropping only the 31 named
   `num_batches_tracked` counters;
3. preserves unfused BatchNorm parameters and PyTorch logical row-major layout;
4. materializes contiguous IEEE-754 `f32` values in little-endian order;
5. emits canonical, lexicographically ordered safetensors header and payload
   data with path- and time-free provenance metadata; and
6. parses the complete in-memory byte stream with the official safetensors
   reader, then writes, fsyncs, and rehashes a same-directory temporary file
   before atomically hard-linking that verified inode into an exclusively
   created final path.

Two isolated executions produce the same 83,246,544-byte package, SHA-256
`9276a840c52cdd2e9afb73cd87a38e15749e12bf494d3ca47b5bc162f237cbcc`.
The contained tensor payload is 83,223,808 bytes. Neither source nor exported
weights belong in Git, and the converted artifact is not yet published.
`scripts/fetch_aux_models.sh` therefore pins the immutable source URL, source
hash, conversion command, and output hash without pretending that a download
is available.

Runtime resolution requires the exact filename
`ecapa_tdnn_voxceleb.safetensors` under `$FRANKEN_WHISPER_MODEL_DIR/aux/` or,
when that variable is unset, `~/.cache/franken_whisper/models/aux/`. Its
required SHA-256 is
`9276a840c52cdd2e9afb73cd87a38e15749e12bf494d3ca47b5bc162f237cbcc`.
The fetch script does not install it; it prints the pinned local conversion
procedure and expected digest.

The shipped Rust verifier first streams the complete package through a bounded,
cancel-aware exact-size and SHA-256 check. It then passes that same
authenticated owned byte buffer—without reopening the path—to
`native_engine::weights::SafetensorsFile` and `WeightsManifest` to require the
exact 200 names and shapes, require every dtype to be `F32`, and compare the
complete deterministic metadata object. Structural, mapping, dtype, metadata,
truncation, corruption, and cancellation failures report stable `ecapa.*`
reasons without printing paths, tensor contents, or source bytes. There is no
second model-package format or sidecar manifest.

`ecapa_frontend_conformance` is the independent bounded scalar oracle and
accepts at most 16,000 samples (one second). `ecapa_frontend_runtime` is the
current product path: it accepts 8,000 through 48,000 samples (one half-second
through three seconds), uses the safe-Rust fixed FFT shared with the native
Whisper frontend, and checks cancellation while processing frames. Both use
the same 400-sample periodic Hamming window, 160-sample hop, centered zero
padding, 400-point one-sided squared-magnitude spectrum, 80 SpeechBrain
symmetric triangular HTK mel filters over 0–8 kHz, `amin=1e-10`, 80 dB
clipping, and per-utterance feature-mean subtraction without
standard-deviation normalization. The model boundary admits 51 through 301
normalized feature frames.

Resampling and downmixing must already have occurred at the normalized-audio
boundary. Callers must apply `validate_ecapa_input_format` while sample-rate
and channel metadata are still available; neither raw-slice frontend can
detect mislabeled 8 kHz or interleaved PCM, and neither guesses their format.

The raw 192-value model output is the golden embedding stage. `EcapaModel`
returns an L2-unit-normalized vector and rejects non-finite, wrong-shaped, or
norm-below-`1e-6` output. The common diarizer consumes that normalized
representation now. For an admitted tracklet of at least two seconds,
`diarize_ecapa_pcm` uses disjoint first and last windows for discovery and
held-out validation, with each window capped at three seconds. Tracklets from
one half-second up to two seconds use a discovery window only. Shorter admitted
tracklets are centered and zero-padded to one half-second. This is current
runtime integration, not evidence that either ECAPA mode has passed its public
accuracy-promotion gates.
The public analytic fixture combines 173 Hz and 347 Hz harmonics, a chirp, and
one impulse; it contains no speech or identity evidence.
`franken-whisper-ecapa-golden-v1` binds full-array hashes and selected values
for the two frontend stages, initial TDNN, first SE-Res2 block, multi-feature
aggregation, attentive pooling, and raw embedding.

`franken-whisper-ecapa-full-oracle-v1` is the corresponding transcript-free
seven-tensor safetensors capture. Its exact identity is 2,160,320 bytes and
SHA-256
`2c80806fbf68262ab1e0a1b52af18139f08272b7802fc3b0fd96011192dcf485`.
The payload contains 539,616 `f32` values: 16,160 frontend values and 523,456
neural-stage values. Its deterministic metadata binds the golden-evidence and
contract identities, analytic fixture, model and training-code revisions,
export schema, Python 3.12.12, NumPy 2.2.6, Torch and Torchaudio 2.7.1,
Safetensors 0.5.3, and SpeechBrain 0.5.16. Oracle generation passes explicit
all-valid lengths through SpeechBrain normalization and ECAPA inference, clones
the raw filterbank before sentence normalization, and snapshots every hooked
stage so later in-place operations cannot mutate evidence.

The offline exporter constructs and independently parses both safetensors byte
streams before exclusively creating either output. The Rust oracle verifier
then applies the same bounded, cancel-aware exact-size and SHA-256 check as the
weight verifier, parses that authenticated owned buffer without reopening the
path, and requires the exact names, shapes, `F32` dtypes, metadata, and
per-tensor payload hashes. Neither artifact is vendored or tracked in Git.

The network convolution boundary is distinct from the frontend boundary.
Every ECAPA convolution uses SpeechBrain's same-length reflection padding over
the dilation-expanded effective kernel; it does not use the frontend's centered
zero padding. TDNN order is convolution, ReLU, then evaluation-mode BatchNorm.
Res2Net chunk zero is the identity, chunk one is convolved directly, and each
later chunk is added to the preceding block output before convolution.
Attention is normalized over time independently for every channel, and both
global-context and attentive standard deviations clamp variance at `1e-12`.
Inference accepts 51 through 301 frames (one half-second through three seconds
at the frozen hop) and features with absolute value at most 160. Longer
tracklets are deterministically windowed by `diarize_ecapa_pcm` under the
discovery/held-out policy above; the neural kernel never allocates or runs in
proportion to a complete recording.

The forward path preplans a checked conservative numeric-buffer ceiling before
copying input. At 301 frames, the ECAPA-owned `f32` activation and kernel-band
payload ceiling is 7,944,704 bytes. The plan adds an 8,388,608-byte reserve for
the reviewed FrankenTorch/matrixmultiply packing buffers, yielding a combined
16,333,312-byte ceiling and a 20 MiB default caller limit. Allocator metadata,
stack use, resident weights, and test-only golden captures are explicitly
outside that number. Every in-scope production-forward ECAPA-owned heap `f32`
scratch allocation first acquires a safe RAII logical-byte lease. The lease
fails closed if live buffers would exceed the admitted owned bound and
decrements on every success/error/cancellation drop path. Successful inference
requires the live count to return to zero and then records the observed logical
peak in the versioned trace. The external model test exercises this meter at
both the oracle's 101 frames and the admitted 301-frame maximum; test-only stage
captures remain deliberately unmetered.

Folded evaluation BatchNorm makes the resident model payload 83,070,208 bytes.
The separately named 204,065,488-byte load accounting is exactly the retained
83,246,544-byte package plus that resident payload plus the largest
37,748,736-byte decoded source tensor. It bounds those logical payloads, not
allocator capacity/metadata, JSON maps, names, shapes, reader buffers, stack,
or process RSS. Compute loops and finite-value scans checkpoint in bounded row,
channel, or value chunks. Public library callers can supply the same callback
while loading and inferring; the no-callback convenience still honors the
process Ctrl-C token. The kernel entry point is explicitly FrankenTorch CPU f32
and cannot auto-dispatch to Metal. Timing and maximum-attention-sum diagnostics
are observational, content-redacted, and nondeterministic; the latter is a
low-bandwidth signal-derived aggregate rather than source content. Exact
repeatability is asserted only within the same process/build/host kernel path;
portable cross-backend and cross-host conformance is tolerance-based.

The packing proof was reviewed against FrankenTorch revision
`523aaf827faf538aa541126ee222fcd7af348410`. Diagnostics expose that evidence
identity as `scratch_proof_reviewed_frankentorch_revision`; the field records
the source revision against which the proof was reviewed, not an attestation of
the mutable sibling checkout compiled into the running binary. The repository
intentionally consumes FrankenTorch as a sibling path dependency rather than
pretending Cargo pins that checkout. Changing that checkout or source topology
requires renewing the proof and updating the field. Builds also reject
matrixmultiply's `MATMUL_SGEMM_NC`, `MATMUL_SGEMM_KC`, and
`MATMUL_SGEMM_MC` compile-time overrides; they would otherwise invalidate the
published packing reserve.

Each golden `reference_sha256` hashes the CPU-contiguous C-order tensor in its
declared shape after encoding every value as little-endian IEEE-754 `f32`.
The network-stage shapes are channel-first `[1, channels, time]`; a native
time-major matrix must therefore be logically transposed before comparison.
The hash authenticates the exact SpeechBrain/PyTorch oracle capture and its
layout; it is not an exact-byte requirement for output from a distinct numeric
backend. A supplied full oracle capture must match this hash before its values
can be used for tolerance-based native comparison.

Declared maximum absolute/relative tolerances are respectively `0.05/0.005`
for pre-normalization filterbanks, `0.08/0.005` for normalized filterbanks,
`0.002/0.002` for initial TDNN, SE-Res2, and aggregation, `0.001/0.002` for
pooling, and `0.02/0.002` for the raw embedding. The scalar Rust frontend is
held to a tighter `0.001` absolute error on the frozen selected points.
The authenticated external conformance test compares the complete Rust frontend
arrays with the two oracle frontend tensors. It then feeds the oracle-normalized
filterbank into the native forward path, isolating backend arithmetic while it
compares every value in all five neural-stage tensors. A second full neural pass
feeds the Rust frontend output into the same network and compares all 523,456
neural-stage values again, detecting error amplification across the composed
boundary. Thus all 539,616 reference elements are checked and every neural
value is also checked through composition, together with attention
normalization, output unit norm, fixed-build repeatability, and observed scratch
accounting. The weight package and full oracle remain external test inputs
supplied through
`FRANKEN_WHISPER_ECAPA_TEST_WEIGHTS` and
`FRANKEN_WHISPER_ECAPA_TEST_ORACLE`; ordinary `cargo test` deliberately skips
this large public-artifact proof rather than silently substituting a fixture.
Non-finite values, native shape drift, oracle evidence hash/version/shape
drift, or a tolerance failure are hard conformance failures. They may not be
waived by downstream DER, silently widened, or converted into an
acoustic-engine success. A tolerance change requires a new schema/version,
regenerated public evidence, and an explicit discrepancy record.

### 11.6 Repository and release guard

Audio/video extensions and transcript sidecars are ignored broadly, including
case variants and text/JSON/subtitle forms. Raw decoder spans and transcript-
shaped performance text are also ignored under `tests/artifacts/perf`.

`scripts/check_repository_privacy.rs` is a standalone standard-library-only
gate. It scans tracked or staged path names first. If any prohibited path is
present, it emits only path/reason NDJSON and exits before reading file
contents. Only after the path phase is clean does it inspect magic bytes and
transcript-shaped content in risky artifact roots. Recognized container magic
therefore catches several renamed media formats, but this is defense in depth:
headerless raw audio and transcript-shaped content outside the enumerated risky
roots are not proven absent by content inspection alone. Path/extension rules
remain the primary boundary. The gate never prints matched content.

Both the automatic tag workflow and the distribution workflow compile this
gate directly with `rustc`. Distribution builds remain allowed to proceed
after advisory test failures, but never after a privacy failure. A known
legacy raw-performance artifact set intentionally keeps this release gate red
until owner-authorized working-tree removal and a separately authorized public
history rewrite are complete.
