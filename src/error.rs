use std::path::PathBuf;

use thiserror::Error;

pub type FwResult<T> = Result<T, FwError>;

/// Stable process exit-code contract exposed by `capabilities --json`.
pub const PROCESS_EXIT_CODES: &[(i32, &str, &str)] = &[
    (0, "success", "requested operation completed"),
    (1, "runtime_error", "request parsed but execution failed"),
    (2, "usage_error", "command-line syntax or value was invalid"),
    (130, "interrupted", "operation was cancelled by SIGINT"),
];

/// Stable, variant-specific error catalog. `FwError::error_code()` is the
/// executable mapping; this catalog gives agents recovery semantics without
/// making them scrape prose from README files.
pub const ERROR_CODE_CATALOG: &[(&str, &str)] = &[
    ("FW-IO", "inspect filesystem permissions and retry"),
    (
        "FW-JSON",
        "validate the referenced JSON document against the declared schema",
    ),
    (
        "FW-CMD-MISSING",
        "install the named optional command or select a native backend",
    ),
    (
        "FW-CMD-FAILED",
        "inspect the sanitized subprocess failure and backend configuration",
    ),
    (
        "FW-CMD-TIMEOUT",
        "increase the command budget or choose a faster backend",
    ),
    (
        "FW-BACKEND-UNAVAILABLE",
        "run `fw robot triage` and follow its recommended command",
    ),
    (
        "FW-INVALID-REQUEST",
        "correct the request; use the command-specific `--help`",
    ),
    (
        "FW-CONTRACT-VIOLATION",
        "do not retry blindly; preserve evidence and report the producer bug",
    ),
    (
        "FW-STORAGE",
        "inspect the state path and run `fw doctor --json`",
    ),
    (
        "FW-UNSUPPORTED",
        "select a supported platform, backend, or output mode",
    ),
    (
        "FW-MISSING-ARTIFACT",
        "provision the exact named artifact and verify its digest",
    ),
    (
        "FW-CANCELLED",
        "retry only if cancellation was not intentional",
    ),
    (
        "FW-STAGE-TIMEOUT",
        "increase the stage budget or reduce the workload",
    ),
];

#[derive(Debug, Error)]
pub enum FwError {
    #[error("i/o failure: {0}")]
    Io(#[from] std::io::Error),

    #[error("json failure: {0}")]
    Json(#[from] serde_json::Error),

    #[error("missing command `{command}` on PATH")]
    CommandMissing { command: String },

    #[error("command failed: `{command}` (status: {status}){stderr_suffix}")]
    CommandFailed {
        command: String,
        status: i32,
        stderr_suffix: String,
    },

    #[error("command timed out after {timeout_ms}ms: `{command}`{stderr_suffix}")]
    CommandTimedOut {
        command: String,
        timeout_ms: u64,
        stderr_suffix: String,
    },

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// An in-process producer returned data that violates a FrankenWhisper
    /// contract. This is deliberately distinct from caller input errors: a
    /// fallback must never conceal an internal postcondition failure.
    #[error("internal contract violation: {0}")]
    ContractViolation(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("missing expected artifact at `{0}`")]
    MissingArtifact(PathBuf),

    #[error("pipeline cancelled: {0}")]
    Cancelled(String),

    #[error("stage `{stage}` exceeded budget of {budget_ms}ms")]
    StageTimeout { stage: String, budget_ms: u64 },
}

impl FwError {
    #[must_use]
    pub fn from_command_failure(command: String, status: i32, stderr: String) -> Self {
        let trimmed = stderr.trim();
        let stderr_suffix = if trimmed.is_empty() {
            String::new()
        } else {
            format!("; stderr: {trimmed}")
        };
        Self::CommandFailed {
            command,
            status,
            stderr_suffix,
        }
    }

    #[must_use]
    pub fn from_command_timeout(command: String, timeout_ms: u64, stderr: String) -> Self {
        let trimmed = stderr.trim();
        let stderr_suffix = if trimmed.is_empty() {
            String::new()
        } else {
            format!("; stderr: {trimmed}")
        };
        Self::CommandTimedOut {
            command,
            timeout_ms,
            stderr_suffix,
        }
    }

    #[must_use]
    pub const fn robot_error_code(&self) -> &'static str {
        match self {
            Self::CommandTimedOut { .. } | Self::StageTimeout { .. } => "FW-ROBOT-TIMEOUT",
            Self::BackendUnavailable(_) => "FW-ROBOT-BACKEND",
            Self::InvalidRequest(_) => "FW-ROBOT-REQUEST",
            Self::Storage(_) => "FW-ROBOT-STORAGE",
            Self::Cancelled(_) => "FW-ROBOT-CANCELLED",
            _ => "FW-ROBOT-EXEC",
        }
    }

    /// Stable, unique, machine-readable error code for every variant.
    /// Unlike `robot_error_code()` which groups variants, this gives each
    /// variant its own distinct code string.
    #[must_use]
    pub const fn error_code(&self) -> &'static str {
        match self {
            Self::Io(_) => "FW-IO",
            Self::Json(_) => "FW-JSON",
            Self::CommandMissing { .. } => "FW-CMD-MISSING",
            Self::CommandFailed { .. } => "FW-CMD-FAILED",
            Self::CommandTimedOut { .. } => "FW-CMD-TIMEOUT",
            Self::BackendUnavailable(_) => "FW-BACKEND-UNAVAILABLE",
            Self::InvalidRequest(_) => "FW-INVALID-REQUEST",
            Self::ContractViolation(_) => "FW-CONTRACT-VIOLATION",
            Self::Storage(_) => "FW-STORAGE",
            Self::Unsupported(_) => "FW-UNSUPPORTED",
            Self::MissingArtifact(_) => "FW-MISSING-ARTIFACT",
            Self::Cancelled(_) => "FW-CANCELLED",
            Self::StageTimeout { .. } => "FW-STAGE-TIMEOUT",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FwError;

    #[test]
    fn timeout_errors_map_to_timeout_robot_code() {
        let stage_timeout = FwError::StageTimeout {
            stage: "backend".to_owned(),
            budget_ms: 10,
        };
        assert_eq!(stage_timeout.robot_error_code(), "FW-ROBOT-TIMEOUT");

        let command_timeout = FwError::from_command_timeout(
            "ffmpeg -i in.wav out.wav".to_owned(),
            1000,
            String::new(),
        );
        assert_eq!(command_timeout.robot_error_code(), "FW-ROBOT-TIMEOUT");
    }

    #[test]
    fn robot_error_code_mapping_regression_matrix() {
        let matrix = vec![
            (
                FwError::Io(std::io::Error::other("disk read failed")),
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::BackendUnavailable("missing backend".to_owned()),
                "FW-ROBOT-BACKEND",
            ),
            (
                FwError::InvalidRequest("bad request".to_owned()),
                "FW-ROBOT-REQUEST",
            ),
            (
                FwError::ContractViolation("bad producer output".to_owned()),
                "FW-ROBOT-EXEC",
            ),
            (FwError::Storage("db fail".to_owned()), "FW-ROBOT-STORAGE"),
            (
                FwError::Cancelled("cancelled".to_owned()),
                "FW-ROBOT-CANCELLED",
            ),
            (
                FwError::StageTimeout {
                    stage: "normalize".to_owned(),
                    budget_ms: 42,
                },
                "FW-ROBOT-TIMEOUT",
            ),
            (
                FwError::from_command_timeout(
                    "ffmpeg -i in.wav out.wav".to_owned(),
                    5_000,
                    String::new(),
                ),
                "FW-ROBOT-TIMEOUT",
            ),
            (
                FwError::CommandFailed {
                    command: "ffmpeg -i in.wav out.wav".to_owned(),
                    status: 1,
                    stderr_suffix: "; stderr: boom".to_owned(),
                },
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::CommandMissing {
                    command: "ffmpeg".to_owned(),
                },
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::MissingArtifact(std::path::PathBuf::from("missing.json")),
                "FW-ROBOT-EXEC",
            ),
            (
                FwError::Unsupported("not supported".to_owned()),
                "FW-ROBOT-EXEC",
            ),
        ];

        for (error, expected_code) in matrix {
            assert_eq!(error.robot_error_code(), expected_code);
        }
    }

    #[test]
    fn from_command_failure_with_empty_stderr() {
        let err = FwError::from_command_failure("cmd".to_owned(), 1, String::new());
        let text = err.to_string();
        assert!(text.contains("cmd"));
        assert!(text.contains("status: 1"));
        // No stderr suffix when stderr is empty.
        assert!(!text.contains("stderr"));
    }

    #[test]
    fn from_command_failure_with_nonempty_stderr() {
        let err = FwError::from_command_failure("prog arg".to_owned(), 2, "  oh no  \n".to_owned());
        let text = err.to_string();
        assert!(text.contains("prog arg"));
        assert!(text.contains("status: 2"));
        assert!(text.contains("stderr: oh no"), "should trim stderr: {text}");
    }

    #[test]
    fn from_command_timeout_with_empty_stderr() {
        let err = FwError::from_command_timeout("slow".to_owned(), 5000, String::new());
        let text = err.to_string();
        assert!(text.contains("5000ms"));
        assert!(!text.contains("stderr"));
    }

    #[test]
    fn from_command_timeout_with_nonempty_stderr() {
        let err =
            FwError::from_command_timeout("slow".to_owned(), 1000, "  partial output  ".to_owned());
        let text = err.to_string();
        assert!(text.contains("1000ms"));
        assert!(
            text.contains("stderr: partial output"),
            "should trim stderr: {text}"
        );
    }

    #[test]
    fn from_command_failure_multiline_stderr_is_trimmed() {
        let stderr = "  line one\nline two\n  line three  \n".to_owned();
        let err = FwError::from_command_failure("cmd".to_owned(), 1, stderr);
        let text = err.to_string();
        // Trim only strips leading/trailing whitespace, not internal newlines.
        assert!(
            text.contains("line one\nline two\n  line three"),
            "multiline stderr should preserve internal newlines: {text}"
        );
    }

    #[test]
    fn from_command_failure_whitespace_only_stderr_treated_as_empty() {
        let err = FwError::from_command_failure("cmd".to_owned(), 1, "   \n\t  ".to_owned());
        let text = err.to_string();
        assert!(
            !text.contains("stderr"),
            "whitespace-only stderr should be omitted: {text}"
        );
    }

    #[test]
    fn from_command_timeout_multiline_stderr_is_trimmed() {
        let stderr = "\nerror detail\nmore detail\n".to_owned();
        let err = FwError::from_command_timeout("slow".to_owned(), 5000, stderr);
        let text = err.to_string();
        assert!(
            text.contains("error detail\nmore detail"),
            "multiline stderr preserved: {text}"
        );
    }

    #[test]
    fn missing_artifact_displays_path() {
        let err = FwError::MissingArtifact(std::path::PathBuf::from("/tmp/output/result.json"));
        let text = err.to_string();
        assert!(
            text.contains("/tmp/output/result.json"),
            "should include full path: {text}"
        );
    }

    #[test]
    fn stage_timeout_displays_stage_and_budget() {
        let err = FwError::StageTimeout {
            stage: "normalize".to_owned(),
            budget_ms: 180_000,
        };
        let text = err.to_string();
        assert!(text.contains("normalize"), "should mention stage: {text}");
        assert!(text.contains("180000"), "should mention budget: {text}");
    }

    #[test]
    fn command_missing_displays_command_name() {
        let err = FwError::CommandMissing {
            command: "whisper-cli".to_owned(),
        };
        let text = err.to_string();
        assert!(
            text.contains("whisper-cli"),
            "should mention command: {text}"
        );
    }

    #[test]
    fn cancelled_error_displays_reason() {
        let err = FwError::Cancelled("pipeline deadline exceeded at ingest stage".to_owned());
        let text = err.to_string();
        assert!(
            text.contains("pipeline deadline exceeded"),
            "should include reason: {text}"
        );
    }

    #[test]
    fn display_messages_for_all_variants() {
        let cases: Vec<(FwError, &str)> = vec![
            (
                FwError::Io(std::io::Error::other("disk fail")),
                "i/o failure",
            ),
            (
                FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
                "json failure",
            ),
            (
                FwError::CommandMissing {
                    command: "ffmpeg".to_owned(),
                },
                "missing command",
            ),
            (
                FwError::CommandFailed {
                    command: "cmd".to_owned(),
                    status: 1,
                    stderr_suffix: String::new(),
                },
                "command failed",
            ),
            (
                FwError::CommandTimedOut {
                    command: "slow".to_owned(),
                    timeout_ms: 5000,
                    stderr_suffix: String::new(),
                },
                "command timed out",
            ),
            (
                FwError::BackendUnavailable("gone".to_owned()),
                "backend unavailable",
            ),
            (FwError::InvalidRequest("bad".to_owned()), "invalid request"),
            (
                FwError::ContractViolation("bad output".to_owned()),
                "internal contract violation",
            ),
            (FwError::Storage("db".to_owned()), "storage error"),
            (FwError::Unsupported("nope".to_owned()), "unsupported"),
            (
                FwError::MissingArtifact(std::path::PathBuf::from("out.json")),
                "missing expected artifact",
            ),
            (FwError::Cancelled("done".to_owned()), "pipeline cancelled"),
            (
                FwError::StageTimeout {
                    stage: "be".to_owned(),
                    budget_ms: 10,
                },
                "exceeded budget",
            ),
        ];

        // Verify we cover all 13 variants.
        assert_eq!(cases.len(), 13, "test should cover every FwError variant");

        for (error, expected_substring) in cases {
            let text = error.to_string();
            assert!(
                text.contains(expected_substring),
                "expected `{expected_substring}` in: {text}"
            );
        }
    }

    #[test]
    fn io_error_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let fw_err: FwError = io_err.into();
        assert!(matches!(fw_err, FwError::Io(_)));
        let text = fw_err.to_string();
        assert!(text.contains("file not found"), "got: {text}");
    }

    #[test]
    fn json_error_from_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let fw_err: FwError = json_err.into();
        assert!(matches!(fw_err, FwError::Json(_)));
        let text = fw_err.to_string();
        assert!(
            text.contains("json failure"),
            "should start with 'json failure': {text}"
        );
    }

    #[test]
    fn stage_timeout_zero_budget() {
        let err = FwError::StageTimeout {
            stage: "ingest".to_owned(),
            budget_ms: 0,
        };
        let text = err.to_string();
        assert!(text.contains("0ms"), "should show 0ms budget: {text}");
        assert!(text.contains("ingest"), "should show stage: {text}");
    }

    #[test]
    fn backend_unavailable_display() {
        let err = FwError::BackendUnavailable(
            "whisper-cli not found; insanely-fast-whisper not found".to_owned(),
        );
        let text = err.to_string();
        assert!(
            text.contains("whisper-cli not found"),
            "should contain detail: {text}"
        );
    }

    #[test]
    fn invalid_request_display() {
        let err = FwError::InvalidRequest("multiple inputs specified".to_owned());
        let text = err.to_string();
        assert!(
            text.contains("multiple inputs"),
            "should contain detail: {text}"
        );
    }

    #[test]
    fn storage_error_display() {
        let err = FwError::Storage("database locked".to_owned());
        let text = err.to_string();
        assert!(
            text.contains("database locked"),
            "should contain detail: {text}"
        );
    }

    #[test]
    fn unsupported_display() {
        let err = FwError::Unsupported("streaming not implemented".to_owned());
        let text = err.to_string();
        assert!(
            text.contains("streaming not implemented"),
            "should contain detail: {text}"
        );
    }

    #[test]
    fn command_timed_out_large_timeout() {
        let err = FwError::CommandTimedOut {
            command: "whisper-cli --model large-v3".to_owned(),
            timeout_ms: 3_600_000,
            stderr_suffix: String::new(),
        };
        let text = err.to_string();
        assert!(
            text.contains("3600000ms"),
            "should show large timeout: {text}"
        );
    }

    #[test]
    fn from_command_timeout_whitespace_only_stderr_treated_as_empty() {
        let err = FwError::from_command_timeout("slow".to_owned(), 5000, "   \n\t  \n".to_owned());
        let text = err.to_string();
        assert!(
            !text.contains("stderr"),
            "whitespace-only stderr should be omitted in timeout too: {text}"
        );
    }

    #[test]
    fn from_command_failure_zero_and_negative_status() {
        let zero = FwError::from_command_failure("cmd".to_owned(), 0, String::new());
        assert!(
            zero.to_string().contains("status: 0"),
            "zero status displayed"
        );

        let neg = FwError::from_command_failure("cmd".to_owned(), -9, "killed".to_owned());
        let text = neg.to_string();
        assert!(text.contains("status: -9"), "negative status: {text}");
        assert!(text.contains("stderr: killed"), "stderr present: {text}");
    }

    #[test]
    fn fw_error_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<FwError>();
        assert_sync::<FwError>();
    }

    #[test]
    fn missing_artifact_unicode_path() {
        let err = FwError::MissingArtifact(std::path::PathBuf::from("/tmp/données/résultat.json"));
        let text = err.to_string();
        assert!(
            text.contains("résultat.json"),
            "unicode path preserved: {text}"
        );
    }

    #[test]
    fn robot_error_code_returns_only_known_codes() {
        let known_codes = [
            "FW-ROBOT-EXEC",
            "FW-ROBOT-TIMEOUT",
            "FW-ROBOT-BACKEND",
            "FW-ROBOT-REQUEST",
            "FW-ROBOT-STORAGE",
            "FW-ROBOT-CANCELLED",
        ];

        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
            FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            FwError::CommandMissing {
                command: "x".to_owned(),
            },
            FwError::CommandFailed {
                command: "x".to_owned(),
                status: 1,
                stderr_suffix: String::new(),
            },
            FwError::CommandTimedOut {
                command: "x".to_owned(),
                timeout_ms: 1,
                stderr_suffix: String::new(),
            },
            FwError::BackendUnavailable("x".to_owned()),
            FwError::InvalidRequest("x".to_owned()),
            FwError::ContractViolation("x".to_owned()),
            FwError::Storage("x".to_owned()),
            FwError::Unsupported("x".to_owned()),
            FwError::MissingArtifact(std::path::PathBuf::from("x")),
            FwError::Cancelled("x".to_owned()),
            FwError::StageTimeout {
                stage: "x".to_owned(),
                budget_ms: 1,
            },
        ];

        for error in &all_errors {
            let code = error.robot_error_code();
            assert!(
                known_codes.contains(&code),
                "unexpected robot_error_code `{code}` for {:?}",
                error
            );
        }
    }

    // ── error_code() tests ──

    /// Construct every FwError variant and verify error_code() returns a
    /// non-empty string starting with "FW-".
    #[test]
    fn test_every_variant_has_error_code() {
        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
            FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            FwError::CommandMissing {
                command: "x".to_owned(),
            },
            FwError::CommandFailed {
                command: "x".to_owned(),
                status: 1,
                stderr_suffix: String::new(),
            },
            FwError::CommandTimedOut {
                command: "x".to_owned(),
                timeout_ms: 1,
                stderr_suffix: String::new(),
            },
            FwError::BackendUnavailable("x".to_owned()),
            FwError::InvalidRequest("x".to_owned()),
            FwError::ContractViolation("x".to_owned()),
            FwError::Storage("x".to_owned()),
            FwError::Unsupported("x".to_owned()),
            FwError::MissingArtifact(std::path::PathBuf::from("x")),
            FwError::Cancelled("x".to_owned()),
            FwError::StageTimeout {
                stage: "x".to_owned(),
                budget_ms: 1,
            },
        ];

        // Ensure we cover all 13 variants.
        assert_eq!(
            all_errors.len(),
            13,
            "test should cover every FwError variant"
        );

        for error in &all_errors {
            let code = error.error_code();
            assert!(
                !code.is_empty(),
                "error_code() must not be empty for {:?}",
                error
            );
            assert!(
                code.starts_with("FW-"),
                "error_code() must start with FW- but got `{code}` for {:?}",
                error
            );
        }
    }

    /// Verify no two variants share the same error_code().
    #[test]
    fn test_error_codes_are_unique() {
        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
            FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            FwError::CommandMissing {
                command: "x".to_owned(),
            },
            FwError::CommandFailed {
                command: "x".to_owned(),
                status: 1,
                stderr_suffix: String::new(),
            },
            FwError::CommandTimedOut {
                command: "x".to_owned(),
                timeout_ms: 1,
                stderr_suffix: String::new(),
            },
            FwError::BackendUnavailable("x".to_owned()),
            FwError::InvalidRequest("x".to_owned()),
            FwError::ContractViolation("x".to_owned()),
            FwError::Storage("x".to_owned()),
            FwError::Unsupported("x".to_owned()),
            FwError::MissingArtifact(std::path::PathBuf::from("x")),
            FwError::Cancelled("x".to_owned()),
            FwError::StageTimeout {
                stage: "x".to_owned(),
                budget_ms: 1,
            },
        ];

        let codes: Vec<&str> = all_errors.iter().map(|e| e.error_code()).collect();
        let mut seen = std::collections::HashSet::new();
        for code in &codes {
            assert!(seen.insert(code), "duplicate error_code detected: `{code}`");
        }
    }

    /// Verify all error codes match the pattern FW-[A-Z-]+.
    #[test]
    fn test_error_code_format() {
        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
            FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            FwError::CommandMissing {
                command: "x".to_owned(),
            },
            FwError::CommandFailed {
                command: "x".to_owned(),
                status: 1,
                stderr_suffix: String::new(),
            },
            FwError::CommandTimedOut {
                command: "x".to_owned(),
                timeout_ms: 1,
                stderr_suffix: String::new(),
            },
            FwError::BackendUnavailable("x".to_owned()),
            FwError::InvalidRequest("x".to_owned()),
            FwError::ContractViolation("x".to_owned()),
            FwError::Storage("x".to_owned()),
            FwError::Unsupported("x".to_owned()),
            FwError::MissingArtifact(std::path::PathBuf::from("x")),
            FwError::Cancelled("x".to_owned()),
            FwError::StageTimeout {
                stage: "x".to_owned(),
                budget_ms: 1,
            },
        ];

        for error in &all_errors {
            let code = error.error_code();
            assert!(
                code.starts_with("FW-"),
                "code must start with FW-: `{code}`"
            );
            let suffix = &code[3..]; // everything after "FW-"
            assert!(
                !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_uppercase() || c == '-'),
                "code suffix must match [A-Z-]+ but got `{suffix}` in `{code}`"
            );
        }
    }

    #[test]
    fn error_code_exact_mapping_regression_matrix() {
        let matrix: Vec<(FwError, &str)> = vec![
            (FwError::Io(std::io::Error::other("test")), "FW-IO"),
            (
                FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
                "FW-JSON",
            ),
            (
                FwError::CommandMissing {
                    command: "x".to_owned(),
                },
                "FW-CMD-MISSING",
            ),
            (
                FwError::CommandFailed {
                    command: "x".to_owned(),
                    status: 1,
                    stderr_suffix: String::new(),
                },
                "FW-CMD-FAILED",
            ),
            (
                FwError::CommandTimedOut {
                    command: "x".to_owned(),
                    timeout_ms: 1,
                    stderr_suffix: String::new(),
                },
                "FW-CMD-TIMEOUT",
            ),
            (
                FwError::BackendUnavailable("x".to_owned()),
                "FW-BACKEND-UNAVAILABLE",
            ),
            (
                FwError::InvalidRequest("x".to_owned()),
                "FW-INVALID-REQUEST",
            ),
            (
                FwError::ContractViolation("x".to_owned()),
                "FW-CONTRACT-VIOLATION",
            ),
            (FwError::Storage("x".to_owned()), "FW-STORAGE"),
            (FwError::Unsupported("x".to_owned()), "FW-UNSUPPORTED"),
            (
                FwError::MissingArtifact(std::path::PathBuf::from("x")),
                "FW-MISSING-ARTIFACT",
            ),
            (FwError::Cancelled("x".to_owned()), "FW-CANCELLED"),
            (
                FwError::StageTimeout {
                    stage: "x".to_owned(),
                    budget_ms: 1,
                },
                "FW-STAGE-TIMEOUT",
            ),
        ];

        assert_eq!(matrix.len(), 13);
        for (error, expected_code) in matrix {
            assert_eq!(
                error.error_code(),
                expected_code,
                "wrong error_code for {:?}",
                error
            );
        }
    }

    #[test]
    fn std_error_source_chains_for_io_and_json() {
        use std::error::Error;

        let io_err = FwError::Io(std::io::Error::other("disk fail"));
        assert!(io_err.source().is_some(), "Io variant should chain source");

        let json_err = FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err());
        assert!(
            json_err.source().is_some(),
            "Json variant should chain source"
        );

        // Non-From variants should have no source
        let plain = FwError::Storage("db fail".to_owned());
        assert!(
            plain.source().is_none(),
            "Storage variant has no chained source"
        );
    }

    #[test]
    fn debug_format_contains_variant_name() {
        let cases: Vec<(FwError, &str)> = vec![
            (FwError::Storage("x".to_owned()), "Storage"),
            (FwError::Cancelled("x".to_owned()), "Cancelled"),
            (
                FwError::CommandMissing {
                    command: "x".to_owned(),
                },
                "CommandMissing",
            ),
            (
                FwError::MissingArtifact(std::path::PathBuf::from("x")),
                "MissingArtifact",
            ),
        ];

        for (error, expected_variant) in cases {
            let debug = format!("{error:?}");
            assert!(
                debug.contains(expected_variant),
                "Debug should contain `{expected_variant}`: {debug}"
            );
        }
    }

    #[test]
    fn error_code_and_robot_code_namespace_disjoint() {
        let all_errors: Vec<FwError> = vec![
            FwError::Io(std::io::Error::other("test")),
            FwError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            FwError::CommandMissing {
                command: "x".to_owned(),
            },
            FwError::CommandFailed {
                command: "x".to_owned(),
                status: 1,
                stderr_suffix: String::new(),
            },
            FwError::CommandTimedOut {
                command: "x".to_owned(),
                timeout_ms: 1,
                stderr_suffix: String::new(),
            },
            FwError::BackendUnavailable("x".to_owned()),
            FwError::InvalidRequest("x".to_owned()),
            FwError::ContractViolation("x".to_owned()),
            FwError::Storage("x".to_owned()),
            FwError::Unsupported("x".to_owned()),
            FwError::MissingArtifact(std::path::PathBuf::from("x")),
            FwError::Cancelled("x".to_owned()),
            FwError::StageTimeout {
                stage: "x".to_owned(),
                budget_ms: 1,
            },
        ];

        let error_codes: std::collections::HashSet<&str> =
            all_errors.iter().map(|e| e.error_code()).collect();
        let robot_codes: std::collections::HashSet<&str> =
            all_errors.iter().map(|e| e.robot_error_code()).collect();

        // No error_code() should appear in the robot_error_code() set
        for code in &error_codes {
            assert!(
                !robot_codes.contains(code),
                "error_code `{code}` collides with a robot_error_code"
            );
        }
    }

    #[test]
    fn from_command_failure_with_special_chars_in_stderr() {
        let stderr = "error: \"invalid\" path <&'\0>".to_owned();
        let err = FwError::from_command_failure("cmd".to_owned(), 1, stderr);
        let text = err.to_string();
        assert!(
            text.contains("stderr: error: \"invalid\" path"),
            "special chars preserved: {text}"
        );
    }

    #[test]
    fn from_command_failure_empty_command_string_still_displays() {
        let err = FwError::from_command_failure(String::new(), 127, "not found".to_owned());
        let text = err.to_string();
        assert!(text.contains("status: 127"), "status present: {text}");
        assert!(text.contains("stderr: not found"), "stderr present: {text}");
        assert_eq!(err.error_code(), "FW-CMD-FAILED");
    }

    #[test]
    fn from_command_timeout_zero_timeout_displays() {
        let err = FwError::from_command_timeout("fast".to_owned(), 0, String::new());
        let text = err.to_string();
        assert!(text.contains("0ms"), "zero timeout shown: {text}");
        assert!(text.contains("fast"), "command shown: {text}");
        assert_eq!(err.error_code(), "FW-CMD-TIMEOUT");
    }

    #[test]
    fn missing_artifact_empty_path() {
        let err = FwError::MissingArtifact(std::path::PathBuf::new());
        let text = err.to_string();
        assert!(
            text.contains("missing expected artifact"),
            "prefix present: {text}"
        );
        assert_eq!(err.error_code(), "FW-MISSING-ARTIFACT");
    }

    #[test]
    fn stage_timeout_empty_stage_string() {
        let err = FwError::StageTimeout {
            stage: String::new(),
            budget_ms: u64::MAX,
        };
        let text = err.to_string();
        assert!(
            text.contains(&u64::MAX.to_string()),
            "large budget displayed: {text}"
        );
        assert_eq!(err.robot_error_code(), "FW-ROBOT-TIMEOUT");
    }

    #[test]
    fn source_returns_none_for_all_struct_variants() {
        use std::error::Error;

        let variants: Vec<FwError> = vec![
            FwError::CommandMissing {
                command: "x".to_owned(),
            },
            FwError::CommandFailed {
                command: "x".to_owned(),
                status: 1,
                stderr_suffix: String::new(),
            },
            FwError::CommandTimedOut {
                command: "x".to_owned(),
                timeout_ms: 1,
                stderr_suffix: String::new(),
            },
            FwError::BackendUnavailable("x".to_owned()),
            FwError::InvalidRequest("x".to_owned()),
            FwError::ContractViolation("x".to_owned()),
            FwError::Storage("x".to_owned()),
            FwError::Unsupported("x".to_owned()),
            FwError::MissingArtifact(std::path::PathBuf::from("x")),
            FwError::Cancelled("x".to_owned()),
            FwError::StageTimeout {
                stage: "x".to_owned(),
                budget_ms: 1,
            },
        ];

        for variant in &variants {
            assert!(
                variant.source().is_none(),
                "{:?} should have no chained source",
                variant
            );
        }
    }
}
