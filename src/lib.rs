#![feature(portable_simd)]
#![forbid(unsafe_code)]
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
