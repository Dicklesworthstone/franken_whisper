//! Reproducible adapters for public or user-licensed diarization corpora.
//!
//! This module deliberately separates path-bearing local preparation inputs
//! from the path-free corpus, reference, leakage, and integrity evidence that
//! can be retained externally. It never copies source media and refuses to
//! write generated annotations inside the project checkout.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diarization::{
    DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION, DIARIZATION_REFERENCE_SCHEMA_VERSION,
    DiarizationCorpusManifest, DiarizationLeakageAudit, DiarizationReferenceDocument,
    EvaluationRegion, EvaluationSplit, EvaluationTurn, audit_diarization_manifest,
    parse_diarization_corpus_manifest, parse_diarization_reference, verify_leakage_audit_hash,
};
use crate::error::{FwError, FwResult};

/// Schema identity for the path-bearing, external-only adapter input.
pub const PUBLIC_CORPUS_INPUT_SCHEMA_VERSION: &str = "public-diarization-corpus-input-v1";
/// Schema identity for the path-free generated bundle.
pub const PUBLIC_CORPUS_BUNDLE_SCHEMA_VERSION: &str = "public-diarization-corpus-bundle-v1";
/// Frozen implementation identity for this adapter.
pub const PUBLIC_CORPUS_ADAPTER_VERSION: &str = "public-diarization-corpus-adapter-v1";
/// Schema identity for the built-in public-corpus registry.
pub const PUBLIC_CORPUS_REGISTRY_SCHEMA_VERSION: &str = "public-diarization-corpus-registry-v1";

const MAX_DESCRIPTOR_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ANNOTATION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RECORDINGS: usize = 100_000;
const MAX_TURNS_PER_RECORDING: usize = 1_000_000;
const MAX_TOTAL_TURNS: usize = 2_000_000;
const HASH_HEX_LEN: usize = 64;

/// How a registry entry freezes its train/development/test assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicCorpusSplitPolicy {
    /// The official AMI scenario-only family split is checked in code.
    AmiScenarioOfficialV1,
    /// The external descriptor is frozen by its SHA-256 and then leakage-audited.
    ExternalDescriptorV1,
}

/// One built-in corpus source and its reproducibility/licensing contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusRegistryEntry {
    pub corpus_key: String,
    pub description: String,
    pub authoritative_url: String,
    pub license_id: String,
    pub license_url: String,
    /// Exact CLI value required by `--license-ack`.
    pub license_acknowledgement_id: String,
    pub split_policy: PublicCorpusSplitPolicy,
    pub expected_local_layout: String,
    pub conversion_contract: String,
    pub upstream_integrity_note: String,
    pub condition_tags: Vec<String>,
}

/// Complete built-in registry emitted by robot-safe CLI output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusRegistry {
    pub schema_version: String,
    pub adapter_version: String,
    pub entries: Vec<PublicCorpusRegistryEntry>,
}

/// Path-free integrity and media-layout evidence for one recording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusRecordingEvidence {
    pub recording_id: String,
    pub split: EvaluationSplit,
    pub audio_sha256: String,
    pub annotation_sha256: String,
    pub reference_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    pub selected_channel: u16,
    pub turn_count: usize,
    pub overlap_turn_count: usize,
    pub ignored_region_count: usize,
}

/// Generated public-corpus evidence. This contains no paths, URIs, or text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusBundle {
    pub schema_version: String,
    pub adapter_version: String,
    pub corpus_key: String,
    pub source_version: String,
    pub license_id: String,
    pub license_acknowledgement_id: String,
    pub descriptor_sha256: String,
    pub manifest: DiarizationCorpusManifest,
    pub leakage_audit: DiarizationLeakageAudit,
    pub references: Vec<DiarizationReferenceDocument>,
    pub recordings: Vec<PublicCorpusRecordingEvidence>,
    /// Hash of the complete bundle with this field temporarily empty.
    pub bundle_sha256: String,
}

/// External-only local descriptor. It intentionally has neither `Debug` nor
/// `Serialize`, preventing accidental logging or retention of source paths.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicCorpusInput {
    schema_version: String,
    corpus_key: String,
    source_version: String,
    recordings: Vec<PublicCorpusInputRecording>,
}

/// One path-bearing external source row.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicCorpusInputRecording {
    recording_id: String,
    split: EvaluationSplit,
    origin_recording_id: String,
    audio_path: PathBuf,
    audio_sha256: String,
    expected_sample_rate_hz: u32,
    expected_channel_count: u16,
    selected_channel: u16,
    annotation_path: PathBuf,
    annotation_sha256: String,
    annotation_recording_id: String,
    annotation_channel: String,
    speaker_map: BTreeMap<String, String>,
    #[serde(default)]
    ignored_regions: Vec<EvaluationRegion>,
    #[serde(default)]
    derived_from_recording_ids: Vec<String>,
    #[serde(default)]
    augmentation_group_id: Option<String>,
    #[serde(default)]
    enrollment_recording_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WaveMetadata {
    sample_rate_hz: u32,
    channel_count: u16,
    duration_ms: u64,
}

/// Return the frozen registry of supported corpus sources.
#[must_use]
pub fn public_corpus_registry() -> PublicCorpusRegistry {
    let mut entries = vec![
        PublicCorpusRegistryEntry {
            corpus_key: "aishell-4-openslr111-v1".to_owned(),
            description:
                "Mandarin multi-channel meetings with short turns, noise, and overlap".to_owned(),
            authoritative_url: "https://www.openslr.org/111/".to_owned(),
            license_id: "CC-BY-SA-4.0".to_owned(),
            license_url:
                "https://creativecommons.org/licenses/by-sa/4.0/legalcode".to_owned(),
            license_acknowledgement_id: "accept-aishell-4-cc-by-sa-4.0".to_owned(),
            split_policy: PublicCorpusSplitPolicy::ExternalDescriptorV1,
            expected_local_layout:
                "<external-root>/audio/**/*.wav and <external-root>/annotations/**/*.rttm"
                    .to_owned(),
            conversion_contract:
                "Select one immutable WAV channel per recording and convert official speaker activity to ten-field RTTM without changing time geometry"
                    .to_owned(),
            upstream_integrity_note:
                "OpenSLR publishes archive names and sizes; this adapter requires SHA-256 for every selected WAV and RTTM after extraction"
                    .to_owned(),
            condition_tags: vec![
                "far-field".to_owned(),
                "meeting".to_owned(),
                "multichannel".to_owned(),
                "overlap".to_owned(),
                "short-turn".to_owned(),
            ],
        },
        PublicCorpusRegistryEntry {
            corpus_key: "ami-scenario-v1".to_owned(),
            description:
                "English meetings with synchronized close-talk and far-field microphones"
                    .to_owned(),
            authoritative_url: "https://groups.inf.ed.ac.uk/ami/corpus/".to_owned(),
            license_id: "CC-BY-4.0".to_owned(),
            license_url: "https://creativecommons.org/licenses/by/4.0/legalcode".to_owned(),
            license_acknowledgement_id: "accept-ami-cc-by-4.0".to_owned(),
            split_policy: PublicCorpusSplitPolicy::AmiScenarioOfficialV1,
            expected_local_layout:
                "<external-root>/audio/<meeting>.wav and <external-root>/annotations/<meeting>.rttm"
                    .to_owned(),
            conversion_contract:
                "Use one official scenario meeting and one named microphone view per recording; convert NXT speaker segments to ten-field RTTM and preserve the official scenario-only family split"
                    .to_owned(),
            upstream_integrity_note:
                "The adapter requires SHA-256 for every selected WAV and converted RTTM because the corpus site does not publish a complete SHA-256 manifest"
                    .to_owned(),
            condition_tags: vec![
                "close-talk".to_owned(),
                "far-field".to_owned(),
                "meeting".to_owned(),
                "overlap".to_owned(),
                "same-speaker-multi-device".to_owned(),
            ],
        },
        PublicCorpusRegistryEntry {
            corpus_key: "callhome-american-english-2e-v1".to_owned(),
            description:
                "Licensed English two-channel 8 kHz conversational telephone speech".to_owned(),
            authoritative_url: "https://catalog.ldc.upenn.edu/LDC2026S08".to_owned(),
            license_id: "LDC-USER-AGREEMENT".to_owned(),
            license_url: "https://catalog.ldc.upenn.edu/LDC2026S08".to_owned(),
            license_acknowledgement_id: "accept-ldc2026s08-user-agreement".to_owned(),
            split_policy: PublicCorpusSplitPolicy::ExternalDescriptorV1,
            expected_local_layout:
                "<external-root>/audio/**/*.wav and <external-root>/annotations/**/*.rttm"
                    .to_owned(),
            conversion_contract:
                "Under an operator-held LDC license, decode selected 8 kHz FLAC channels to immutable PCM WAV and convert licensed speaker turns to ten-field RTTM; no LDC material may enter Git"
                    .to_owned(),
            upstream_integrity_note:
                "LDC access is user-licensed; record SHA-256 for every locally decoded WAV and RTTM and retain the descriptor outside the checkout"
                    .to_owned(),
            condition_tags: vec![
                "channel-mismatch".to_owned(),
                "dyadic".to_owned(),
                "telephone".to_owned(),
                "two-channel".to_owned(),
            ],
        },
        PublicCorpusRegistryEntry {
            corpus_key: "voxconverse-v1".to_owned(),
            description:
                "In-the-wild multi-speaker clips with overlap and challenging backgrounds"
                    .to_owned(),
            authoritative_url: "https://mm.kaist.ac.kr/datasets/voxconverse/".to_owned(),
            license_id: "CC-BY-4.0-ORIGINAL-COPYRIGHT".to_owned(),
            license_url: "https://mm.kaist.ac.kr/datasets/voxconverse/".to_owned(),
            license_acknowledgement_id:
                "accept-voxconverse-cc-by-4.0-and-original-copyright".to_owned(),
            split_policy: PublicCorpusSplitPolicy::ExternalDescriptorV1,
            expected_local_layout:
                "<external-root>/audio/{dev,test}/*.wav and <external-root>/annotations/{dev,test}/*.rttm"
                    .to_owned(),
            conversion_contract:
                "Use the upstream WAV and RTTM pairing without transcript material; keep development and test identities disjoint and freeze every selected file by SHA-256"
                    .to_owned(),
            upstream_integrity_note:
                "The corpus page publishes archive MD5 values; this adapter additionally requires SHA-256 for every selected extracted WAV and RTTM"
                    .to_owned(),
            condition_tags: vec![
                "background-noise".to_owned(),
                "in-the-wild".to_owned(),
                "overlap".to_owned(),
                "same-gender".to_owned(),
                "short-turn".to_owned(),
            ],
        },
    ];
    entries.sort_by(|left, right| left.corpus_key.cmp(&right.corpus_key));
    for entry in &mut entries {
        entry.condition_tags.sort();
    }
    PublicCorpusRegistry {
        schema_version: PUBLIC_CORPUS_REGISTRY_SCHEMA_VERSION.to_owned(),
        adapter_version: PUBLIC_CORPUS_ADAPTER_VERSION.to_owned(),
        entries,
    }
}

/// Parse and fully validate one generated bundle.
pub fn parse_public_corpus_bundle(bytes: &[u8]) -> FwResult<PublicCorpusBundle> {
    let bundle = serde_json::from_slice(bytes).map_err(|_| {
        public_corpus_error(
            "bundle_json",
            "bundle must be valid public-corpus JSON without trailing data",
        )
    })?;
    verify_public_corpus_bundle(&bundle)?;
    Ok(bundle)
}

/// Verify schemas, hashes, scorer documents, ordering, and leakage evidence.
pub fn verify_public_corpus_bundle(bundle: &PublicCorpusBundle) -> FwResult<()> {
    if bundle.schema_version != PUBLIC_CORPUS_BUNDLE_SCHEMA_VERSION {
        return Err(public_corpus_error(
            "bundle_schema_version",
            "unsupported public-corpus bundle schema version",
        ));
    }
    if bundle.adapter_version != PUBLIC_CORPUS_ADAPTER_VERSION {
        return Err(public_corpus_error(
            "adapter_version",
            "unsupported public-corpus adapter version",
        ));
    }
    let registry = public_corpus_registry();
    let entry = registry
        .entries
        .iter()
        .find(|candidate| candidate.corpus_key == bundle.corpus_key)
        .ok_or_else(|| {
            public_corpus_error(
                "corpus_key",
                "bundle corpus key is not in the frozen registry",
            )
        })?;
    if bundle.license_id != entry.license_id
        || bundle.license_acknowledgement_id != entry.license_acknowledgement_id
    {
        return Err(public_corpus_error(
            "license_contract",
            "bundle license identity does not match the frozen registry",
        ));
    }
    for (field, value) in [
        ("descriptor_sha256", &bundle.descriptor_sha256),
        ("bundle_sha256", &bundle.bundle_sha256),
    ] {
        if !is_sha256_hex(value) {
            return Err(public_corpus_error(
                "hash_format",
                &format!("{field} must be 64 lowercase hexadecimal characters"),
            ));
        }
    }
    validate_public_id(&bundle.source_version, "source_version")?;

    let manifest_bytes = serde_json::to_vec(&bundle.manifest)?;
    parse_diarization_corpus_manifest(&manifest_bytes)?;
    if bundle.manifest.corpus_id != bundle.corpus_key
        || bundle.manifest.license_id != bundle.license_id
    {
        return Err(public_corpus_error(
            "manifest_identity",
            "embedded manifest identity differs from the bundle",
        ));
    }
    verify_leakage_audit_hash(&bundle.leakage_audit)?;
    if !bundle.leakage_audit.passed {
        return Err(public_corpus_error(
            "leakage_audit",
            "generated public-corpus bundle must have a passing leakage audit",
        ));
    }
    let regenerated_audit = audit_diarization_manifest(&bundle.manifest)?;
    if regenerated_audit != bundle.leakage_audit {
        return Err(public_corpus_error(
            "leakage_audit_mismatch",
            "embedded leakage audit does not match the embedded manifest",
        ));
    }
    if bundle.references.len() != bundle.recordings.len()
        || bundle.references.len() != bundle.manifest.recordings.len()
    {
        return Err(public_corpus_error(
            "recording_cardinality",
            "reference, evidence, and manifest recording counts differ",
        ));
    }
    if !bundle
        .references
        .windows(2)
        .all(|window| window[0].recording_id < window[1].recording_id)
        || !bundle
            .recordings
            .windows(2)
            .all(|window| window[0].recording_id < window[1].recording_id)
    {
        return Err(public_corpus_error(
            "recording_order",
            "bundle recordings must be strictly ordered by recording_id",
        ));
    }
    for ((reference, evidence), manifest_recording) in bundle
        .references
        .iter()
        .zip(&bundle.recordings)
        .zip(&bundle.manifest.recordings)
    {
        parse_diarization_reference(&serde_json::to_vec(reference)?)?;
        if reference.recording_id != evidence.recording_id
            || reference.recording_id != manifest_recording.recording_id
            || evidence.split != manifest_recording.split
        {
            return Err(public_corpus_error(
                "recording_alignment",
                "reference, evidence, and manifest recording identities differ",
            ));
        }
        if evidence.reference_sha256 != canonical_sha256(reference)?
            || evidence.turn_count != reference.turns.len()
            || evidence.overlap_turn_count
                != reference
                    .turns
                    .iter()
                    .filter(|turn| turn.overlap_suspected)
                    .count()
            || evidence.ignored_region_count != reference.ignored_regions.len()
        {
            return Err(public_corpus_error(
                "recording_evidence",
                "recording evidence does not match the embedded reference",
            ));
        }
        if !is_sha256_hex(&evidence.audio_sha256)
            || !is_sha256_hex(&evidence.annotation_sha256)
            || !is_sha256_hex(&evidence.reference_sha256)
        {
            return Err(public_corpus_error(
                "recording_hash_format",
                "recording evidence hashes must be lowercase SHA-256",
            ));
        }
        if evidence.sample_rate_hz == 0
            || evidence.channel_count == 0
            || evidence.selected_channel == 0
            || evidence.selected_channel > evidence.channel_count
        {
            return Err(public_corpus_error(
                "recording_audio_contract",
                "recording evidence has invalid sample-rate or channel geometry",
            ));
        }
    }
    let mut unhashed = bundle.clone();
    let expected = unhashed.bundle_sha256.clone();
    unhashed.bundle_sha256.clear();
    if canonical_sha256(&unhashed)? != expected {
        return Err(public_corpus_error(
            "bundle_hash_mismatch",
            "bundle_sha256 does not match canonical bundle content",
        ));
    }
    Ok(())
}

/// Build one path-free bundle from external WAV and RTTM inputs.
///
/// The output is opened with `create_new`, and all source/output roots must be
/// absolute, canonical, and disjoint from the project checkout.
pub fn build_public_corpus_bundle(
    project_root: &Path,
    input_root: &Path,
    descriptor_path: &Path,
    output_path: &Path,
    license_acknowledgement_id: &str,
) -> FwResult<PublicCorpusBundle> {
    build_public_corpus_bundle_with_cancel(
        project_root,
        input_root,
        descriptor_path,
        output_path,
        license_acknowledgement_id,
        || false,
    )
}

/// Cancellation-aware form of [`build_public_corpus_bundle`].
pub fn build_public_corpus_bundle_with_cancel(
    project_root: &Path,
    input_root: &Path,
    descriptor_path: &Path,
    output_path: &Path,
    license_acknowledgement_id: &str,
    mut is_cancelled: impl FnMut() -> bool,
) -> FwResult<PublicCorpusBundle> {
    checkpoint_cancelled(&mut is_cancelled)?;
    let canonical_project = canonical_directory(project_root, "project_root")?;
    let canonical_input = canonical_directory(input_root, "input_root")?;
    if paths_overlap(&canonical_project, &canonical_input) {
        return Err(public_corpus_error(
            "input_root_overlap",
            "input root must be disjoint from the project checkout",
        ));
    }
    let canonical_descriptor =
        canonical_input_file(&canonical_input, descriptor_path, "descriptor")?;
    let output_parent = validate_new_output(&canonical_project, &canonical_input, output_path)?;
    checkpoint_cancelled(&mut is_cancelled)?;

    let descriptor_bytes = read_bounded(&canonical_descriptor, MAX_DESCRIPTOR_BYTES, "descriptor")?;
    let descriptor_sha256 = format!("{:x}", Sha256::digest(&descriptor_bytes));
    let mut descriptor: PublicCorpusInput =
        serde_json::from_slice(&descriptor_bytes).map_err(|_| {
            public_corpus_error(
                "descriptor_json",
                "descriptor must be valid public-corpus input JSON without trailing data",
            )
        })?;
    if descriptor.schema_version != PUBLIC_CORPUS_INPUT_SCHEMA_VERSION {
        return Err(public_corpus_error(
            "descriptor_schema_version",
            "unsupported public-corpus input schema version",
        ));
    }
    validate_public_id(&descriptor.corpus_key, "corpus_key")?;
    validate_public_id(&descriptor.source_version, "source_version")?;
    if descriptor.recordings.is_empty() || descriptor.recordings.len() > MAX_RECORDINGS {
        return Err(public_corpus_error(
            "recording_count",
            "descriptor recording count is outside the supported range",
        ));
    }
    let registry = public_corpus_registry();
    let registry_entry = registry
        .entries
        .iter()
        .find(|entry| entry.corpus_key == descriptor.corpus_key)
        .ok_or_else(|| {
            public_corpus_error(
                "corpus_key",
                "descriptor corpus key is not in the frozen registry",
            )
        })?;
    if license_acknowledgement_id != registry_entry.license_acknowledgement_id {
        return Err(public_corpus_error(
            "license_acknowledgement",
            "the exact registry license acknowledgement is required",
        ));
    }
    descriptor
        .recordings
        .sort_by(|left, right| left.recording_id.cmp(&right.recording_id));

    let mut recording_ids = BTreeSet::new();
    let mut references = Vec::with_capacity(descriptor.recordings.len());
    let mut manifest_recordings = Vec::with_capacity(descriptor.recordings.len());
    let mut evidence = Vec::with_capacity(descriptor.recordings.len());
    let mut total_turn_count = 0_usize;
    for recording in descriptor.recordings {
        checkpoint_cancelled(&mut is_cancelled)?;
        validate_public_id(&recording.recording_id, "recording_id")?;
        if !recording_ids.insert(recording.recording_id.clone()) {
            return Err(public_corpus_error(
                "duplicate_recording",
                "descriptor recording IDs must be unique",
            ));
        }
        validate_public_id(&recording.origin_recording_id, "origin_recording_id")?;
        validate_split(
            registry_entry.split_policy,
            &recording.recording_id,
            recording.split,
        )?;
        validate_sha256(&recording.audio_sha256, "audio_sha256")?;
        validate_sha256(&recording.annotation_sha256, "annotation_sha256")?;
        if recording.expected_sample_rate_hz == 0
            || recording.expected_channel_count == 0
            || recording.selected_channel == 0
            || recording.selected_channel > recording.expected_channel_count
        {
            return Err(public_corpus_error(
                "audio_contract",
                "expected sample-rate and channel geometry is invalid",
            ));
        }
        validate_public_id(
            &recording.annotation_recording_id,
            "annotation_recording_id",
        )?;
        validate_rttm_channel(&recording.annotation_channel)?;
        validate_speaker_map(&recording.speaker_map)?;
        let audio_path = canonical_relative_file(&canonical_input, &recording.audio_path, "audio")?;
        let annotation_path =
            canonical_relative_file(&canonical_input, &recording.annotation_path, "annotation")?;
        let (actual_audio_sha256, wave) = hash_and_inspect_wave(&audio_path, &mut is_cancelled)?;
        if actual_audio_sha256 != recording.audio_sha256 {
            return Err(public_corpus_error(
                "audio_checksum_mismatch",
                "audio SHA-256 does not match the descriptor",
            ));
        }
        if wave.sample_rate_hz != recording.expected_sample_rate_hz
            || wave.channel_count != recording.expected_channel_count
        {
            return Err(public_corpus_error(
                "audio_metadata_mismatch",
                "WAV sample rate or channel count does not match the descriptor",
            ));
        }
        let annotation_bytes = read_bounded(&annotation_path, MAX_ANNOTATION_BYTES, "annotation")?;
        let actual_annotation_sha256 = format!("{:x}", Sha256::digest(&annotation_bytes));
        if actual_annotation_sha256 != recording.annotation_sha256 {
            return Err(public_corpus_error(
                "annotation_checksum_mismatch",
                "annotation SHA-256 does not match the descriptor",
            ));
        }
        let turns = parse_rttm(
            &annotation_bytes,
            &recording.annotation_recording_id,
            &recording.annotation_channel,
            &recording.speaker_map,
            wave.duration_ms,
        )?;
        total_turn_count = total_turn_count
            .checked_add(turns.len())
            .filter(|count| *count <= MAX_TOTAL_TURNS)
            .ok_or_else(|| {
                public_corpus_error(
                    "total_turn_count",
                    "corpus turn count exceeds the supported memory-safety limit",
                )
            })?;
        let mut ignored_regions = recording.ignored_regions;
        ignored_regions.sort_by(|left, right| {
            (left.start_ms, left.end_ms, left.reason_code.as_str()).cmp(&(
                right.start_ms,
                right.end_ms,
                right.reason_code.as_str(),
            ))
        });
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: recording.recording_id.clone(),
            duration_ms: wave.duration_ms,
            turns,
            ignored_regions,
            speaker_hints: Vec::new(),
        };
        parse_diarization_reference(&serde_json::to_vec(&reference)?)?;
        let reference_sha256 = canonical_sha256(&reference)?;

        let speaker_refs = reference
            .turns
            .iter()
            .filter_map(|turn| turn.speaker.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut derived_from_recording_ids = recording.derived_from_recording_ids;
        let mut enrollment_recording_ids = recording.enrollment_recording_ids;
        derived_from_recording_ids.sort();
        enrollment_recording_ids.sort();
        manifest_recordings.push(crate::diarization::CorpusRecordingManifest {
            recording_id: recording.recording_id.clone(),
            split: recording.split,
            origin_recording_id: recording.origin_recording_id,
            speaker_refs,
            derived_from_recording_ids,
            augmentation_group_id: recording.augmentation_group_id,
            enrollment_recording_ids,
        });
        evidence.push(PublicCorpusRecordingEvidence {
            recording_id: recording.recording_id,
            split: recording.split,
            audio_sha256: actual_audio_sha256,
            annotation_sha256: actual_annotation_sha256,
            reference_sha256,
            sample_rate_hz: wave.sample_rate_hz,
            channel_count: wave.channel_count,
            selected_channel: recording.selected_channel,
            turn_count: reference.turns.len(),
            overlap_turn_count: reference
                .turns
                .iter()
                .filter(|turn| turn.overlap_suspected)
                .count(),
            ignored_region_count: reference.ignored_regions.len(),
        });
        references.push(reference);
    }

    manifest_recordings.sort_by(|left, right| {
        (left.recording_id.as_str(), left.split).cmp(&(right.recording_id.as_str(), right.split))
    });
    references.sort_by(|left, right| left.recording_id.cmp(&right.recording_id));
    evidence.sort_by(|left, right| left.recording_id.cmp(&right.recording_id));
    let manifest = DiarizationCorpusManifest {
        schema_version: DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION.to_owned(),
        corpus_id: descriptor.corpus_key.clone(),
        license_id: registry_entry.license_id.clone(),
        recordings: manifest_recordings,
    };
    parse_diarization_corpus_manifest(&serde_json::to_vec(&manifest)?)?;
    let leakage_audit = audit_diarization_manifest(&manifest)?;
    if !leakage_audit.passed {
        return Err(public_corpus_error(
            "split_leakage",
            "descriptor violates the frozen cross-split leakage contract",
        ));
    }
    let mut bundle = PublicCorpusBundle {
        schema_version: PUBLIC_CORPUS_BUNDLE_SCHEMA_VERSION.to_owned(),
        adapter_version: PUBLIC_CORPUS_ADAPTER_VERSION.to_owned(),
        corpus_key: descriptor.corpus_key,
        source_version: descriptor.source_version,
        license_id: registry_entry.license_id.clone(),
        license_acknowledgement_id: registry_entry.license_acknowledgement_id.clone(),
        descriptor_sha256,
        manifest,
        leakage_audit,
        references,
        recordings: evidence,
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = canonical_sha256(&bundle)?;
    verify_public_corpus_bundle(&bundle)?;
    checkpoint_cancelled(&mut is_cancelled)?;
    write_new_bundle(output_path, &output_parent, &bundle)?;
    Ok(bundle)
}

fn parse_rttm(
    bytes: &[u8],
    recording_id: &str,
    channel: &str,
    speaker_map: &BTreeMap<String, String>,
    duration_ms: u64,
) -> FwResult<Vec<EvaluationTurn>> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| public_corpus_error("rttm_utf8", "RTTM annotation must be valid UTF-8"))?;
    let mut turns = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = trimmed.split_ascii_whitespace().collect::<Vec<_>>();
        if fields.len() != 10 || fields[0] != "SPEAKER" {
            return Err(public_corpus_error(
                "rttm_shape",
                &format!(
                    "RTTM line {} must contain exactly ten SPEAKER fields",
                    line_index + 1
                ),
            ));
        }
        if fields[1] != recording_id || fields[2] != channel {
            continue;
        }
        let start_ms = parse_rttm_milliseconds(fields[3], line_index)?;
        let duration = parse_rttm_milliseconds(fields[4], line_index)?;
        if duration == 0 {
            return Err(public_corpus_error(
                "rttm_duration",
                &format!("RTTM line {} has zero duration", line_index + 1),
            ));
        }
        let end_ms = start_ms.checked_add(duration).ok_or_else(|| {
            public_corpus_error(
                "rttm_time_overflow",
                &format!("RTTM line {} exceeds supported time range", line_index + 1),
            )
        })?;
        if end_ms > duration_ms {
            return Err(public_corpus_error(
                "rttm_bounds",
                &format!("RTTM line {} exceeds WAV duration", line_index + 1),
            ));
        }
        let speaker = speaker_map.get(fields[7]).ok_or_else(|| {
            public_corpus_error(
                "rttm_speaker_map",
                &format!(
                    "RTTM line {} speaker is absent from speaker_map",
                    line_index + 1
                ),
            )
        })?;
        turns.push(EvaluationTurn {
            start_ms,
            end_ms,
            speaker: Some(speaker.clone()),
            speaker_confidence: None,
            overlap_suspected: false,
        });
        if turns.len() > MAX_TURNS_PER_RECORDING {
            return Err(public_corpus_error(
                "rttm_turn_count",
                "RTTM turn count exceeds the supported limit",
            ));
        }
    }
    if turns.is_empty() {
        return Err(public_corpus_error(
            "rttm_no_matching_turns",
            "RTTM contains no turns for the selected recording and channel",
        ));
    }
    turns.sort_by(|left, right| {
        (
            left.start_ms,
            left.end_ms,
            left.speaker.as_deref().unwrap_or_default(),
        )
            .cmp(&(
                right.start_ms,
                right.end_ms,
                right.speaker.as_deref().unwrap_or_default(),
            ))
    });
    mark_overlapping_turns(&mut turns);
    Ok(turns)
}

fn parse_rttm_milliseconds(value: &str, line_index: usize) -> FwResult<u64> {
    if value.is_empty() || value.starts_with('-') || value.starts_with('+') {
        return Err(public_corpus_error(
            "rttm_time",
            &format!(
                "RTTM line {} time must be a non-negative decimal",
                line_index + 1
            ),
        ));
    }
    let mut parts = value.split('.');
    let whole = parts.next().unwrap_or_default();
    let fraction = parts.next();
    if parts.next().is_some()
        || whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.is_some_and(|digits| {
            digits.is_empty()
                || digits.len() > 9
                || !digits.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return Err(public_corpus_error(
            "rttm_time",
            &format!(
                "RTTM line {} time must be a plain decimal with at most nine fractional digits",
                line_index + 1
            ),
        ));
    }
    let whole_seconds = whole.parse::<u64>().map_err(|_| {
        public_corpus_error(
            "rttm_time",
            &format!("RTTM line {} time is out of range", line_index + 1),
        )
    })?;
    let mut milliseconds = whole_seconds.checked_mul(1_000).ok_or_else(|| {
        public_corpus_error(
            "rttm_time",
            &format!("RTTM line {} time is out of range", line_index + 1),
        )
    })?;
    if let Some(digits) = fraction {
        let bytes = digits.as_bytes();
        let hundreds = u64::from(bytes.first().copied().unwrap_or(b'0') - b'0');
        let tens = u64::from(bytes.get(1).copied().unwrap_or(b'0') - b'0');
        let ones = u64::from(bytes.get(2).copied().unwrap_or(b'0') - b'0');
        milliseconds = milliseconds
            .checked_add(hundreds * 100 + tens * 10 + ones)
            .ok_or_else(|| {
                public_corpus_error(
                    "rttm_time",
                    &format!("RTTM line {} time is out of range", line_index + 1),
                )
            })?;
        if bytes.get(3).is_some_and(|digit| *digit >= b'5') {
            milliseconds = milliseconds.checked_add(1).ok_or_else(|| {
                public_corpus_error(
                    "rttm_time",
                    &format!("RTTM line {} time is out of range", line_index + 1),
                )
            })?;
        }
    }
    Ok(milliseconds)
}

fn mark_overlapping_turns(turns: &mut [EvaluationTurn]) {
    let mut overlaps = vec![false; turns.len()];
    {
        let mut maximum_end_by_speaker = BTreeMap::<Option<&str>, u64>::new();
        let mut ranked_maximum_ends = BTreeSet::<(u64, Option<&str>)>::new();
        for (index, turn) in turns.iter().enumerate() {
            let speaker = turn.speaker.as_deref();
            if ranked_maximum_ends
                .iter()
                .rev()
                .find(|(_, candidate)| *candidate != speaker)
                .is_some_and(|(end_ms, _)| *end_ms > turn.start_ms)
            {
                overlaps[index] = true;
            }
            let prior_end = maximum_end_by_speaker.get(&speaker).copied();
            if prior_end.is_none_or(|end_ms| turn.end_ms > end_ms) {
                if let Some(end_ms) = prior_end {
                    ranked_maximum_ends.remove(&(end_ms, speaker));
                }
                maximum_end_by_speaker.insert(speaker, turn.end_ms);
                ranked_maximum_ends.insert((turn.end_ms, speaker));
            }
        }
    }
    {
        let mut minimum_start_by_speaker = BTreeMap::<Option<&str>, u64>::new();
        let mut ranked_minimum_starts = BTreeSet::<(u64, Option<&str>)>::new();
        for (index, turn) in turns.iter().enumerate().rev() {
            let speaker = turn.speaker.as_deref();
            if ranked_minimum_starts
                .iter()
                .find(|(_, candidate)| *candidate != speaker)
                .is_some_and(|(start_ms, _)| *start_ms < turn.end_ms)
            {
                overlaps[index] = true;
            }
            let prior_start = minimum_start_by_speaker.get(&speaker).copied();
            if prior_start.is_none_or(|start_ms| turn.start_ms < start_ms) {
                if let Some(start_ms) = prior_start {
                    ranked_minimum_starts.remove(&(start_ms, speaker));
                }
                minimum_start_by_speaker.insert(speaker, turn.start_ms);
                ranked_minimum_starts.insert((turn.start_ms, speaker));
            }
        }
    }
    for (turn, overlap) in turns.iter_mut().zip(overlaps) {
        turn.overlap_suspected = overlap;
    }
}

fn hash_and_inspect_wave(
    path: &Path,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<(String, WaveMetadata)> {
    let mut file = File::open(path)
        .map_err(|_| public_corpus_error("audio_read", "audio input could not be opened"))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        checkpoint_cancelled(is_cancelled)?;
        let read = file
            .read(&mut buffer)
            .map_err(|_| public_corpus_error("audio_read", "audio input could not be read"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|_| public_corpus_error("audio_read", "audio input could not be rewound"))?;
    let reader = hound::WavReader::new(file).map_err(|_| {
        public_corpus_error(
            "wave_parse",
            "audio input must be a readable finite PCM or IEEE-float WAV",
        )
    })?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err(public_corpus_error(
            "wave_metadata",
            "WAV sample rate and channel count must be non-zero",
        ));
    }
    let frames = u64::from(reader.duration());
    let duration_ms = frames
        .checked_mul(1_000)
        .and_then(|scaled| scaled.checked_add(u64::from(spec.sample_rate) - 1))
        .map(|scaled| scaled / u64::from(spec.sample_rate))
        .ok_or_else(|| {
            public_corpus_error("wave_duration", "WAV duration exceeds the supported range")
        })?;
    if duration_ms == 0 {
        return Err(public_corpus_error(
            "wave_duration",
            "WAV must contain at least one millisecond of audio",
        ));
    }
    Ok((
        format!("{:x}", hasher.finalize()),
        WaveMetadata {
            sample_rate_hz: spec.sample_rate,
            channel_count: spec.channels,
            duration_ms,
        },
    ))
}

fn validate_split(
    policy: PublicCorpusSplitPolicy,
    recording_id: &str,
    actual: EvaluationSplit,
) -> FwResult<()> {
    if policy == PublicCorpusSplitPolicy::ExternalDescriptorV1 {
        return Ok(());
    }
    let meeting = recording_id
        .strip_prefix("ami-")
        .or_else(|| recording_id.strip_prefix("AMI-"))
        .ok_or_else(|| {
            public_corpus_error(
                "ami_recording_id",
                "AMI recording IDs must start with the ami- namespace",
            )
        })?;
    let family = meeting.get(..6).ok_or_else(|| {
        public_corpus_error(
            "ami_recording_id",
            "AMI recording ID does not contain an ASCII scenario meeting family",
        )
    })?;
    let expected = if AMI_SCENARIO_TRAIN.contains(&family) {
        EvaluationSplit::Train
    } else if AMI_SCENARIO_DEVELOPMENT.contains(&family) {
        EvaluationSplit::Development
    } else if AMI_SCENARIO_TEST.contains(&family) {
        EvaluationSplit::Test
    } else {
        return Err(public_corpus_error(
            "ami_split_unknown",
            "AMI recording is outside the frozen scenario-only family split",
        ));
    };
    if actual == expected {
        Ok(())
    } else {
        Err(public_corpus_error(
            "ami_split_mismatch",
            "AMI recording split differs from the frozen official scenario-only split",
        ))
    }
}

const AMI_SCENARIO_TRAIN: [&str; 25] = [
    "ES2002", "ES2005", "ES2006", "ES2007", "ES2008", "ES2009", "ES2010", "ES2012", "ES2013",
    "ES2015", "ES2016", "IS1000", "IS1001", "IS1002", "IS1003", "IS1004", "IS1005", "IS1006",
    "IS1007", "TS3005", "TS3008", "TS3009", "TS3010", "TS3011", "TS3012",
];
const AMI_SCENARIO_DEVELOPMENT: [&str; 5] = ["ES2003", "ES2011", "IS1008", "TS3004", "TS3006"];
const AMI_SCENARIO_TEST: [&str; 5] = ["ES2004", "ES2014", "IS1009", "TS3003", "TS3007"];

fn validate_speaker_map(speaker_map: &BTreeMap<String, String>) -> FwResult<()> {
    if speaker_map.is_empty() {
        return Err(public_corpus_error(
            "speaker_map",
            "speaker_map must contain at least one source-to-opaque identity",
        ));
    }
    let mut targets = BTreeSet::new();
    for (source, target) in speaker_map {
        if source.is_empty()
            || source.len() > 160
            || source.trim() != source
            || source.chars().any(char::is_control)
            || source.bytes().any(|byte| byte.is_ascii_whitespace())
        {
            return Err(public_corpus_error(
                "speaker_map_source",
                "speaker_map source labels must be bounded non-whitespace tokens",
            ));
        }
        validate_public_id(target, "speaker_map target")?;
        if !targets.insert(target) {
            return Err(public_corpus_error(
                "speaker_map_target",
                "speaker_map target identities must be unique",
            ));
        }
    }
    Ok(())
}

fn validate_rttm_channel(channel: &str) -> FwResult<()> {
    if channel.is_empty()
        || channel.len() > 32
        || channel.trim() != channel
        || channel
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control())
    {
        Err(public_corpus_error(
            "annotation_channel",
            "RTTM channel must be one bounded non-whitespace token",
        ))
    } else {
        Ok(())
    }
}

fn validate_public_id(value: &str, field: &str) -> FwResult<()> {
    if value.is_empty()
        || value.len() > 160
        || value.trim() != value
        || value.chars().any(char::is_control)
        || value.contains('/')
        || value.contains('\\')
        || value.contains("..")
    {
        return Err(public_corpus_error(
            "opaque_id",
            &format!("{field} must be a bounded path-free opaque identifier"),
        ));
    }
    let lower = value.to_ascii_lowercase();
    for forbidden in [
        "downloads",
        "transcript",
        ".m4a",
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".aac",
        ".wma",
        ".mp4",
        ".srt",
        ".md",
    ] {
        if lower.contains(forbidden) {
            return Err(public_corpus_error(
                "opaque_id_sensitive",
                &format!("{field} contains a forbidden path or media marker"),
            ));
        }
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> FwResult<()> {
    if is_sha256_hex(value) {
        Ok(())
    } else {
        Err(public_corpus_error(
            "hash_format",
            &format!("{field} must be 64 lowercase hexadecimal characters"),
        ))
    }
}

fn canonical_directory(path: &Path, field: &str) -> FwResult<PathBuf> {
    if !path.is_absolute() {
        return Err(public_corpus_error(
            "absolute_path",
            &format!("{field} must be absolute"),
        ));
    }
    let canonical = path.canonicalize().map_err(|_| {
        public_corpus_error(
            "directory",
            &format!("{field} must be an existing readable directory"),
        )
    })?;
    if !canonical.is_dir() {
        return Err(public_corpus_error(
            "directory",
            &format!("{field} must resolve to a directory"),
        ));
    }
    Ok(canonical)
}

fn canonical_input_file(root: &Path, path: &Path, field: &str) -> FwResult<PathBuf> {
    if !path.is_absolute() {
        return Err(public_corpus_error(
            "absolute_path",
            &format!("{field} must be absolute"),
        ));
    }
    let canonical = path.canonicalize().map_err(|_| {
        public_corpus_error(
            "input_file",
            &format!("{field} must resolve to a readable file"),
        )
    })?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err(public_corpus_error(
            "input_escape",
            &format!("{field} must resolve beneath input_root"),
        ));
    }
    Ok(canonical)
}

fn canonical_relative_file(root: &Path, relative: &Path, field: &str) -> FwResult<PathBuf> {
    if relative.is_absolute()
        || relative.as_os_str().is_empty()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(public_corpus_error(
            "relative_path",
            &format!("{field} path must be a non-empty relative path without traversal"),
        ));
    }
    let canonical = root.join(relative).canonicalize().map_err(|_| {
        public_corpus_error(
            "input_file",
            &format!("{field} input must resolve to a readable file"),
        )
    })?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err(public_corpus_error(
            "input_escape",
            &format!("{field} input must resolve beneath input_root"),
        ));
    }
    Ok(canonical)
}

fn validate_new_output(project: &Path, input: &Path, output: &Path) -> FwResult<PathBuf> {
    if !output.is_absolute() || output.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return Err(public_corpus_error(
            "output_path",
            "output must be an absolute path with a .json extension",
        ));
    }
    if output.exists() {
        return Err(public_corpus_error(
            "output_exists",
            "output must not already exist",
        ));
    }
    let parent = output.parent().ok_or_else(|| {
        public_corpus_error(
            "output_parent",
            "output must have an existing parent directory",
        )
    })?;
    let canonical_parent = parent.canonicalize().map_err(|_| {
        public_corpus_error(
            "output_parent",
            "output parent must be an existing directory",
        )
    })?;
    if !canonical_parent.is_dir()
        || paths_overlap(project, &canonical_parent)
        || paths_overlap(input, &canonical_parent)
    {
        return Err(public_corpus_error(
            "output_overlap",
            "output parent must be disjoint from the project and input roots",
        ));
    }
    Ok(canonical_parent)
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn read_bounded(path: &Path, limit: u64, field: &str) -> FwResult<Vec<u8>> {
    let file = File::open(path).map_err(|_| {
        public_corpus_error("input_read", &format!("{field} input could not be opened"))
    })?;
    let metadata = file.metadata().map_err(|_| {
        public_corpus_error("input_read", &format!("{field} metadata could not be read"))
    })?;
    if metadata.len() > limit {
        return Err(public_corpus_error(
            "input_size",
            &format!("{field} input exceeds its safety limit"),
        ));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        public_corpus_error(
            "input_size",
            &format!("{field} input length is unsupported on this platform"),
        )
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(limit + 1).read_to_end(&mut bytes).map_err(|_| {
        public_corpus_error("input_read", &format!("{field} input could not be read"))
    })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(public_corpus_error(
            "input_size",
            &format!("{field} input exceeds its safety limit"),
        ));
    }
    Ok(bytes)
}

fn write_new_bundle(
    output_path: &Path,
    canonical_parent: &Path,
    bundle: &PublicCorpusBundle,
) -> FwResult<()> {
    let output_name = output_path
        .file_name()
        .ok_or_else(|| public_corpus_error("output_path", "output must include a file name"))?;
    let canonical_target = canonical_parent.join(output_name);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&canonical_target)
        .map_err(|_| {
            public_corpus_error(
                "output_create",
                "new public-corpus output could not be created",
            )
        })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, bundle).map_err(|_| {
        public_corpus_error(
            "output_write",
            "public-corpus output could not be serialized",
        )
    })?;
    writer.write_all(b"\n").map_err(|_| {
        public_corpus_error("output_write", "public-corpus output could not be written")
    })?;
    writer.flush().map_err(|_| {
        public_corpus_error("output_write", "public-corpus output could not be flushed")
    })?;
    writer.get_ref().sync_all().map_err(|_| {
        public_corpus_error(
            "output_write",
            "public-corpus output could not be durably synchronized",
        )
    })
}

fn canonical_sha256<T: Serialize>(value: &T) -> FwResult<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == HASH_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn checkpoint_cancelled(is_cancelled: &mut impl FnMut() -> bool) -> FwResult<()> {
    if is_cancelled() {
        Err(FwError::Cancelled(
            "public_corpus.cancelled: public corpus preparation cancelled".to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn public_corpus_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("public_corpus.{code}: {message}"))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::json;
    use sha2::Digest as _;
    use tempfile::tempdir;

    use super::{
        PUBLIC_CORPUS_INPUT_SCHEMA_VERSION, build_public_corpus_bundle,
        build_public_corpus_bundle_with_cancel, parse_public_corpus_bundle, public_corpus_registry,
        validate_split,
    };
    use crate::FwResult;
    use crate::diarization::EvaluationSplit;

    fn write_wave(path: &Path, sample_rate: u32, channels: u16, frames: u32) {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("WAV");
        for _ in 0..frames * u32::from(channels) {
            writer.write_sample(0_i16).expect("sample");
        }
        writer.finalize().expect("finalize WAV");
    }

    fn sha256(path: &Path) -> String {
        format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(path).expect("fixture"))
        )
    }

    fn descriptor(
        corpus_key: &str,
        recording_id: &str,
        split: &str,
        audio_sha256: &str,
        annotation_sha256: &str,
        sample_rate: u32,
        channels: u16,
    ) -> serde_json::Value {
        json!({
            "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
            "corpus_key": corpus_key,
            "source_version": "fixture-v1",
            "recordings": [{
                "recording_id": recording_id,
                "split": split,
                "origin_recording_id": format!("{recording_id}-origin"),
                "audio_path": "audio.wav",
                "audio_sha256": audio_sha256,
                "expected_sample_rate_hz": sample_rate,
                "expected_channel_count": channels,
                "selected_channel": 1,
                "annotation_path": "annotation.rttm",
                "annotation_sha256": annotation_sha256,
                "annotation_recording_id": "source-call",
                "annotation_channel": "1",
                "speaker_map": {
                    "source-a": format!("{recording_id}-speaker-a"),
                    "source-b": format!("{recording_id}-speaker-b")
                },
                "ignored_regions": [{
                    "start_ms": 900,
                    "end_ms": 950,
                    "reason_code": "annotation_uncertain"
                }]
            }]
        })
    }

    struct Fixture {
        project: tempfile::TempDir,
        input: tempfile::TempDir,
        output: tempfile::TempDir,
        descriptor_path: PathBuf,
        output_path: PathBuf,
    }

    impl Fixture {
        fn new(corpus_key: &str, recording_id: &str, split: &str) -> Self {
            let project = tempdir().expect("project");
            let input = tempdir().expect("input");
            let output = tempdir().expect("output");
            write_wave(&input.path().join("audio.wav"), 8_000, 2, 8_000);
            std::fs::write(
                input.path().join("annotation.rttm"),
                concat!(
                    "SPEAKER source-call 1 0.000 0.600 <NA> <NA> source-a <NA> <NA>\n",
                    "SPEAKER source-call 1 0.400 0.500 <NA> <NA> source-b <NA> <NA>\n",
                ),
            )
            .expect("RTTM");
            let audio_hash = sha256(&input.path().join("audio.wav"));
            let annotation_hash = sha256(&input.path().join("annotation.rttm"));
            let descriptor_path = input.path().join("descriptor.json");
            std::fs::write(
                &descriptor_path,
                serde_json::to_vec_pretty(&descriptor(
                    corpus_key,
                    recording_id,
                    split,
                    &audio_hash,
                    &annotation_hash,
                    8_000,
                    2,
                ))
                .expect("descriptor JSON"),
            )
            .expect("descriptor");
            let output_path = output.path().join("bundle.json");
            Self {
                project,
                input,
                output,
                descriptor_path,
                output_path,
            }
        }

        fn build(&self, acknowledgement: &str) -> FwResult<super::PublicCorpusBundle> {
            build_public_corpus_bundle(
                self.project.path(),
                self.input.path(),
                &self.descriptor_path,
                &self.output_path,
                acknowledgement,
            )
        }
    }

    #[test]
    fn registry_is_sorted_complete_and_path_free() {
        let registry = public_corpus_registry();
        assert_eq!(registry.entries.len(), 4);
        assert!(
            registry
                .entries
                .windows(2)
                .all(|window| window[0].corpus_key < window[1].corpus_key)
        );
        for entry in &registry.entries {
            assert!(entry.authoritative_url.starts_with("https://"));
            assert!(entry.license_url.starts_with("https://"));
            assert!(!entry.license_acknowledgement_id.is_empty());
            assert!(!entry.condition_tags.is_empty());
            assert!(
                entry
                    .condition_tags
                    .windows(2)
                    .all(|window| window[0] < window[1])
            );
        }
    }

    #[test]
    fn build_requires_exact_license_acknowledgement() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let error = fixture.build("yes").expect_err("missing acknowledgement");
        assert!(error.to_string().contains("license_acknowledgement"));
        assert!(!fixture.output_path.exists());
    }

    #[test]
    fn checksum_mismatch_fails_before_output() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture.descriptor_path).expect("descriptor"))
                .expect("JSON");
        value["recordings"][0]["audio_sha256"] = json!("0".repeat(64));
        std::fs::write(
            &fixture.descriptor_path,
            serde_json::to_vec_pretty(&value).expect("JSON"),
        )
        .expect("descriptor");
        let error = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect_err("checksum mismatch");
        assert!(error.to_string().contains("audio_checksum_mismatch"));
        assert!(!fixture.output_path.exists());
    }

    #[test]
    fn malformed_rttm_fails_with_stable_code() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        std::fs::write(
            fixture.input.path().join("annotation.rttm"),
            "SPEAKER too few fields\n",
        )
        .expect("malformed RTTM");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture.descriptor_path).expect("descriptor"))
                .expect("JSON");
        value["recordings"][0]["annotation_sha256"] =
            json!(sha256(&fixture.input.path().join("annotation.rttm")));
        std::fs::write(
            &fixture.descriptor_path,
            serde_json::to_vec_pretty(&value).expect("JSON"),
        )
        .expect("descriptor");
        let error = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect_err("malformed RTTM");
        assert!(error.to_string().contains("rttm_shape"));
    }

    #[test]
    fn wave_channel_and_sample_rate_contracts_are_checked() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture.descriptor_path).expect("descriptor"))
                .expect("JSON");
        value["recordings"][0]["expected_sample_rate_hz"] = json!(16_000);
        std::fs::write(
            &fixture.descriptor_path,
            serde_json::to_vec_pretty(&value).expect("JSON"),
        )
        .expect("descriptor");
        let error = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect_err("sample-rate mismatch");
        assert!(error.to_string().contains("audio_metadata_mismatch"));
    }

    #[test]
    fn overlap_ignored_regions_and_determinism_survive_round_trip() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let bundle = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect("bundle");
        assert_eq!(bundle.references[0].turns.len(), 2);
        assert!(
            bundle.references[0]
                .turns
                .iter()
                .all(|turn| turn.overlap_suspected)
        );
        assert_eq!(bundle.references[0].ignored_regions.len(), 1);
        let retained = std::fs::read(&fixture.output_path).expect("retained bundle");
        assert_eq!(
            parse_public_corpus_bundle(&retained).expect("parse bundle"),
            bundle
        );
        let hypothesis = crate::diarization::DiarizationHypothesisDocument {
            schema_version: crate::diarization::DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: bundle.references[0].recording_id.clone(),
            duration_ms: bundle.references[0].duration_ms,
            turns: bundle.references[0].turns.clone(),
            performance: None,
        };
        let score = crate::diarization::score_diarization_documents(
            &bundle.references[0],
            &hypothesis,
            &crate::diarization::DiarizationScorerConfig::default(),
        )
        .expect("generated reference must run through the frozen scorer");
        assert_eq!(score.diarization.der, Some(0.0));
        assert_eq!(score.diarization.jer, Some(0.0));

        let second_output = fixture.output.path().join("bundle-second.json");
        let second = build_public_corpus_bundle(
            fixture.project.path(),
            fixture.input.path(),
            &fixture.descriptor_path,
            &second_output,
            "accept-aishell-4-cc-by-sa-4.0",
        )
        .expect("second bundle");
        assert_eq!(second, bundle);
        assert_eq!(
            std::fs::read_to_string(&second_output).expect("second output"),
            std::fs::read_to_string(&fixture.output_path).expect("first output")
        );
    }

    #[test]
    fn official_ami_split_is_enforced() {
        validate_split(
            super::PublicCorpusSplitPolicy::AmiScenarioOfficialV1,
            "ami-ES2003a-array",
            EvaluationSplit::Development,
        )
        .expect("official dev split");
        let error = validate_split(
            super::PublicCorpusSplitPolicy::AmiScenarioOfficialV1,
            "ami-ES2003a-array",
            EvaluationSplit::Test,
        )
        .expect_err("wrong split");
        assert!(error.to_string().contains("ami_split_mismatch"));
    }

    #[test]
    fn malformed_unicode_ami_id_returns_error_instead_of_panicking() {
        let error = validate_split(
            super::PublicCorpusSplitPolicy::AmiScenarioOfficialV1,
            "ami-aééé",
            EvaluationSplit::Development,
        )
        .expect_err("non-ASCII family");
        assert!(error.to_string().contains("ami_recording_id"));
    }

    #[test]
    fn cancellation_leaves_no_output() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let mut checks = 0_u8;
        let error = build_public_corpus_bundle_with_cancel(
            fixture.project.path(),
            fixture.input.path(),
            &fixture.descriptor_path,
            &fixture.output_path,
            "accept-aishell-4-cc-by-sa-4.0",
            || {
                checks = checks.saturating_add(1);
                checks >= 2
            },
        )
        .expect_err("cancelled");
        assert!(matches!(error, crate::FwError::Cancelled(_)));
        assert!(!fixture.output_path.exists());
    }

    #[test]
    fn output_must_remain_outside_project_and_inputs() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let unsafe_output = fixture.project.path().join("bundle.json");
        let error = build_public_corpus_bundle(
            fixture.project.path(),
            fixture.input.path(),
            &fixture.descriptor_path,
            &unsafe_output,
            "accept-aishell-4-cc-by-sa-4.0",
        )
        .expect_err("project output");
        assert!(error.to_string().contains("output_overlap"));
    }

    #[test]
    fn rttm_time_rounding_is_decimal_and_deterministic() {
        assert_eq!(
            super::parse_rttm_milliseconds("1.2344", 0).expect("time"),
            1_234
        );
        assert_eq!(
            super::parse_rttm_milliseconds("1.2345", 0).expect("time"),
            1_235
        );
        assert!(super::parse_rttm_milliseconds("1e3", 0).is_err());
    }

    #[test]
    fn overlap_marking_distinguishes_same_and_different_speakers() {
        let mut turns = vec![
            crate::diarization::EvaluationTurn::labeled(0, 100, "speaker-a"),
            crate::diarization::EvaluationTurn::labeled(10, 90, "speaker-a"),
            crate::diarization::EvaluationTurn::labeled(95, 110, "speaker-b"),
            crate::diarization::EvaluationTurn::labeled(200, 300, "speaker-c"),
        ];
        super::mark_overlapping_turns(&mut turns);
        assert!(turns[0].overlap_suspected);
        assert!(!turns[1].overlap_suspected);
        assert!(turns[2].overlap_suspected);
        assert!(!turns[3].overlap_suspected);
    }
}
