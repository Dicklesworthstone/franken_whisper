//! YouTube ingestion pipeline: yt-dlp orchestration → native transcription
//! → markdown + JSON outputs.
//!
//! `yt-dlp` is an orchestrated external tool (probed like `whisper-cli` and
//! `ffmpeg`; override via `FRANKEN_WHISPER_YTDLP_BIN`). Audio is downloaded
//! as best-audio without re-encoding — the existing normalize stage handles
//! 16 kHz mono conversion. See the bd-epic bead for the full design record.
//!
//! Module map:
//! - [`ytdlp`]    — tool probe, URL classification, playlist expansion,
//!   metadata fetch, cancellable audio download
//! - [`naming`]   — dual-filename sanitization (`{date} - {title} [{id}]`)
//! - [`pipeline`] — asupersync cancel-correct pipeline + manifest state machine
//! - [`render`]   — markdown + JSON renderers

pub mod naming;
pub mod pipeline;
pub mod render;
pub mod ytdlp;
