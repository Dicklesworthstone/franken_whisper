#![feature(portable_simd)]
// `deny` (not `forbid`) so the SINGLE audited f16c GEMV dot in
// `native_engine::nn` (gated by `#[allow(unsafe_code)]` + a runtime
// `is_x86_feature_detected!` check, with a safe two-pass fallback) can use the
// `vcvtph2ps`+`fmadd` intrinsics GGML uses — a measured 2.5–5× on the dominant
// decoder GEMV (owner-approved 2026-06-25; see docs/NEGATIVE_EVIDENCE.md). Every
// other `unsafe` in the crate is still rejected at compile time.
#![deny(unsafe_code)]
#![allow(clippy::needless_raw_string_hashes)]

pub mod accelerate;
pub mod audio;
pub mod backend;
pub mod cli;
pub mod conformance;
pub mod error;
pub mod export;
pub mod logging;
pub mod model;
pub mod native_engine;
pub mod orchestrator;
pub mod process;
pub mod replay_pack;
pub mod robot;
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
