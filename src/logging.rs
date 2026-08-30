//! Structured logging configuration for franken_whisper.
//!
//! Initializes a `tracing` subscriber with:
//! - `RUST_LOG` environment filter support
//! - Default level: INFO
//! - JSON output when `RUST_LOG_FORMAT=json`
//! - Human-readable output otherwise

use tracing_subscriber::EnvFilter;

/// Initialize the global tracing subscriber.
///
/// Call this once at program startup (main.rs).
/// Safe to call multiple times (subsequent calls are no-ops).
pub fn init() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("franken_whisper=info"));

    let is_json = std::env::var("RUST_LOG_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false);

    if is_json {
        let _ = subscriber.json().try_init();
    } else {
        let _ = subscriber.try_init();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_does_not_panic() {
        // Calling init() should not panic even if called multiple times
        init();
        init();
    }

    #[test]
    fn env_filter_accepts_global_level() {
        // `RUST_LOG=warn` is a valid operator choice and intentionally does
        // not contain the crate name. The previous assertion depended on the
        // developer's ambient environment and failed whenever a global level
        // was configured.
        let filter = EnvFilter::try_new("warn").expect("global level filter");
        assert!(format!("{filter:?}").to_ascii_lowercase().contains("warn"));
    }

    #[test]
    fn default_filter_parses_info_level() {
        // Validates the exact filter string used in production init().
        let filter = EnvFilter::new("franken_whisper=info");
        let dbg = format!("{filter:?}");
        assert!(
            dbg.contains("franken_whisper"),
            "default filter should target franken_whisper crate: {dbg}"
        );
    }

    #[test]
    fn env_filter_with_multiple_targets_does_not_panic() {
        // Edge case: a user could set RUST_LOG to a multi-target directive.
        // Verify that EnvFilter accepts compound directives without panicking.
        let filter = EnvFilter::new("franken_whisper=trace,hyper=warn,tower=off");
        let dbg = format!("{filter:?}");
        assert!(dbg.contains("franken_whisper"));
    }
}
