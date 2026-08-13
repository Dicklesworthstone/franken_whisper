#![feature(portable_simd)]
// `deny` (not `forbid`) so explicitly scoped native-engine kernels can use
// runtime-gated SIMD and fully-overwritten preallocated output buffers. Each
// exception carries a local safety argument; unannotated unsafe code is still
// rejected. The performance evidence lives in docs/NEGATIVE_EVIDENCE.md.
#![deny(unsafe_code)]
#![allow(clippy::needless_raw_string_hashes)]

pub mod accelerate;
pub mod adversarial_corpus;
pub mod audio;
pub mod backend;
pub mod cli;
pub mod confidential_evaluation;
pub mod conformance;
pub mod denoise;
pub mod diarization;
pub mod diarization_projection;
pub mod differential_oracle;
pub mod ecapa_conformance;
pub mod ecapa_inference;
pub mod error;
pub mod export;
pub mod logging;
pub mod model;
pub mod model_distribution;
pub mod native_engine;
pub mod orchestrator;
pub mod process;
pub mod public_corpus;
pub mod replay_pack;
pub mod robot;
pub mod sortformer_conformance;
pub mod sortformer_identity;
pub mod sortformer_inference;
pub mod speculation;
pub mod storage;
pub mod streaming;
pub mod sync;
pub mod tty_audio;
pub mod tui;
pub mod youtube;

pub use error::{FwError, FwResult};
pub use model::{BackendKind, RunReport, TranscribeRequest, TranscriptionResult};
pub use orchestrator::{FrankenWhisperEngine, PipelineBuilder, PipelineConfig, PipelineStage};
