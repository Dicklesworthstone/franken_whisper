use std::io::Read;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{FwError, FwResult};

#[must_use]
pub fn command_exists(program: &str) -> bool {
    which::which(program).is_ok()
}

pub fn run_command(program: &str, args: &[String], cwd: Option<&Path>) -> FwResult<Output> {
    run_command_with_timeout(program, args, cwd, None)
}

fn render_command_for_log(program: &str, args: &[String]) -> String {
    if args.is_empty() {
        return program.to_owned();
    }

    let mut capacity = program.len().saturating_add(args.len());
    let mut redact_next = false;
    for arg in args {
        if redact_next {
            capacity = capacity.saturating_add(3);
            redact_next = false;
            continue;
        }

        if let Some((flag, _value)) = arg.split_once('=')
            && is_sensitive_flag(flag)
        {
            capacity = capacity.saturating_add(flag.len().saturating_add(4));
            continue;
        }

        capacity = capacity.saturating_add(arg.len());
        if is_sensitive_flag(arg) {
            redact_next = true;
        }
    }

    let mut rendered = String::with_capacity(capacity);
    rendered.push_str(program);
    let mut redact_next = false;
    for arg in args {
        rendered.push(' ');
        if redact_next {
            rendered.push_str("***");
            redact_next = false;
            continue;
        }

        if let Some((flag, _value)) = arg.split_once('=')
            && is_sensitive_flag(flag)
        {
            rendered.push_str(flag);
            rendered.push_str("=***");
            continue;
        }

        rendered.push_str(arg);
        if is_sensitive_flag(arg) {
            redact_next = true;
        }
    }

    rendered
}

fn is_sensitive_flag(flag: &str) -> bool {
    matches!(
        flag,
        "--hf-token"
            | "--hf_token"
            | "--api-key"
            | "--api_key"
            | "--access-token"
            | "--access_token"
            | "--auth-token"
            | "--auth_token"
            | "--password"
            | "--pass"
            | "--secret"
            | "--secret-key"
            | "--secret_key"
    )
}

pub fn run_command_with_timeout(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    timeout: Option<Duration>,
) -> FwResult<Output> {
    if !command_exists(program) {
        return Err(FwError::CommandMissing {
            command: program.to_owned(),
        });
    }

    let rendered = render_command_for_log(program, args);
    let mut command = Command::new(program);
    command.args(args);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }

    if let Some(limit) = timeout {
        let mut child = command.spawn()?;
        let started_at = Instant::now();

        let stdout_pipe = child.stdout.take().ok_or_else(|| {
            FwError::Io(std::io::Error::other(format!(
                "failed to capture stdout for `{rendered}`"
            )))
        })?;
        let stderr_pipe = child.stderr.take().ok_or_else(|| {
            FwError::Io(std::io::Error::other(format!(
                "failed to capture stderr for `{rendered}`"
            )))
        })?;

        let stdout_rx = spawn_pipe_reader(stdout_pipe);
        let stderr_rx = spawn_pipe_reader(stderr_pipe);

        loop {
            if let Some(status) = child.try_wait()? {
                let stdout = recv_pipe_output(stdout_rx)?;
                let stderr = recv_pipe_output(stderr_rx)?;
                return validate_captured_command_output(&rendered, status, stdout, stderr);
            }

            if started_at.elapsed() >= limit {
                let _ = child.kill();
                let _ = child.wait();
                let _ = recv_pipe_output_after_termination(stdout_rx);
                let stderr = recv_pipe_output_after_termination(stderr_rx)
                    .map(|capture| capture.bytes)
                    .unwrap_or_default();
                let stderr_str = String::from_utf8_lossy(&stderr).into_owned();
                return Err(FwError::from_command_timeout(
                    rendered,
                    saturating_duration_ms(limit),
                    stderr_str,
                ));
            }

            thread::sleep(Duration::from_millis(20));
        }
    }

    let output = command.output()?;
    validate_command_output(&rendered, output)
}

// The early sleeps total 50ms, preserving the original polling phase and
// steady-state ceiling after accelerating short-lived subprocesses.
const CANCELLABLE_POLL_CEILING_MS: u64 = 50;
const CANCELLABLE_EARLY_POLL_MS: [u64; 6] = [1, 2, 4, 8, 16, 19];

fn cancellable_poll_delay(iteration: usize) -> Duration {
    Duration::from_millis(
        CANCELLABLE_EARLY_POLL_MS
            .get(iteration)
            .copied()
            .unwrap_or(CANCELLABLE_POLL_CEILING_MS),
    )
}

/// Run a subprocess with cancellation-aware polling.
///
/// Instead of a fixed timeout, this variant polls `token.checkpoint()` after
/// front-loaded sleeps that rejoin the original 50ms cadence at 50ms. If the
/// checkpoint returns `Err(Cancelled)`, the child process is killed immediately
/// and the error is propagated. An optional hard timeout is still respected as
/// a safety net.
pub(crate) fn run_command_cancellable(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    token: &crate::orchestrator::CancellationToken,
    hard_timeout: Option<Duration>,
) -> FwResult<Output> {
    run_command_cancellable_with_probe(program, args, cwd, token, hard_timeout, None)
}

/// Run a subprocess while honoring both the structured pipeline token and an
/// optional caller-owned cancellation predicate.
pub(crate) fn run_command_cancellable_with_probe(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    token: &crate::orchestrator::CancellationToken,
    hard_timeout: Option<Duration>,
    additional_cancel: Option<&(dyn Fn() -> bool + Sync)>,
) -> FwResult<Output> {
    token.checkpoint()?;
    if additional_cancel.is_some_and(|probe| probe()) {
        return Err(FwError::Cancelled(
            "subprocess cancelled by caller predicate".to_owned(),
        ));
    }
    if !command_exists(program) {
        return Err(FwError::CommandMissing {
            command: program.to_owned(),
        });
    }

    let rendered = render_command_for_log(program, args);
    let mut command = Command::new(program);
    command.args(args);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }

    let mut child = command.spawn()?;
    let started_at = Instant::now();
    let mut poll_iteration = 0usize;

    let stdout_pipe = child.stdout.take().ok_or_else(|| {
        FwError::Io(std::io::Error::other(format!(
            "failed to capture stdout for `{rendered}`"
        )))
    })?;
    let stderr_pipe = child.stderr.take().ok_or_else(|| {
        FwError::Io(std::io::Error::other(format!(
            "failed to capture stderr for `{rendered}`"
        )))
    })?;

    let stdout_rx = spawn_pipe_reader(stdout_pipe);
    let stderr_rx = spawn_pipe_reader(stderr_pipe);

    loop {
        if let Some(status) = child.try_wait()? {
            let stdout = recv_pipe_output(stdout_rx)?;
            let stderr = recv_pipe_output(stderr_rx)?;
            return validate_captured_command_output(&rendered, status, stdout, stderr);
        }

        // Check pipeline deadline via cancellation token.
        if let Err(err) = token.checkpoint() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = recv_pipe_output_after_termination(stdout_rx);
            let _ = recv_pipe_output_after_termination(stderr_rx);
            return Err(err);
        }
        if additional_cancel.is_some_and(|probe| probe()) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = recv_pipe_output_after_termination(stdout_rx);
            let _ = recv_pipe_output_after_termination(stderr_rx);
            return Err(FwError::Cancelled(
                "subprocess cancelled by caller predicate".to_owned(),
            ));
        }

        // Hard timeout safety net.
        if let Some(limit) = hard_timeout
            && started_at.elapsed() >= limit
        {
            let _ = child.kill();
            let _ = child.wait();
            let _ = recv_pipe_output_after_termination(stdout_rx);
            let stderr = recv_pipe_output_after_termination(stderr_rx)
                .map(|capture| capture.bytes)
                .unwrap_or_default();
            let stderr_str = String::from_utf8_lossy(&stderr).into_owned();
            return Err(FwError::from_command_timeout(
                rendered,
                saturating_duration_ms(limit),
                stderr_str,
            ));
        }

        thread::sleep(cancellable_poll_delay(poll_iteration));
        poll_iteration = poll_iteration.saturating_add(1);
    }
}

fn validate_command_output(rendered: &str, output: Output) -> FwResult<Output> {
    if output.status.success() {
        return Ok(output);
    }

    let status = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    Err(FwError::from_command_failure(
        rendered.to_owned(),
        status,
        stderr,
    ))
}

fn saturating_duration_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

const MAX_CAPTURED_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const PIPE_CAPTURE_COMPLETION_GRACE: Duration = Duration::from_secs(1);
const PIPE_CAPTURE_TERMINATION_GRACE: Duration = Duration::from_millis(100);

#[derive(Debug)]
struct PipeCapture {
    bytes: Vec<u8>,
    limit_exceeded: bool,
}

fn spawn_pipe_reader<R>(pipe: R) -> std::sync::mpsc::Receiver<std::io::Result<PipeCapture>>
where
    R: Read + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(read_pipe_with_limit(pipe));
    });
    rx
}

fn read_pipe_with_limit<R: Read>(mut pipe: R) -> std::io::Result<PipeCapture> {
    let mut buf = [0u8; 8192];
    let mut bytes = Vec::new();
    let mut limit_exceeded = false;

    loop {
        let read = pipe.read(&mut buf)?;
        if read == 0 {
            break;
        }
        let remaining = MAX_CAPTURED_OUTPUT_BYTES.saturating_sub(bytes.len());
        let retained = remaining.min(read);
        bytes.extend_from_slice(&buf[..retained]);
        if retained < read {
            limit_exceeded = true;
        }
    }

    Ok(PipeCapture {
        bytes,
        limit_exceeded,
    })
}

fn recv_pipe_output(
    rx: std::sync::mpsc::Receiver<std::io::Result<PipeCapture>>,
) -> FwResult<PipeCapture> {
    recv_pipe_output_with_timeout(rx, PIPE_CAPTURE_COMPLETION_GRACE)
}

fn recv_pipe_output_after_termination(
    rx: std::sync::mpsc::Receiver<std::io::Result<PipeCapture>>,
) -> FwResult<PipeCapture> {
    recv_pipe_output_with_timeout(rx, PIPE_CAPTURE_TERMINATION_GRACE)
}

fn recv_pipe_output_with_timeout(
    rx: std::sync::mpsc::Receiver<std::io::Result<PipeCapture>>,
    timeout: Duration,
) -> FwResult<PipeCapture> {
    match rx.recv_timeout(timeout) {
        Ok(result) => result.map_err(FwError::Io),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(FwError::ContractViolation(
            "subprocess output pipe remained open after the child terminated".to_owned(),
        )),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(FwError::Io(
            std::io::Error::other("subprocess pipe reader terminated"),
        )),
    }
}

fn validate_captured_command_output(
    rendered: &str,
    status: std::process::ExitStatus,
    stdout: PipeCapture,
    stderr: PipeCapture,
) -> FwResult<Output> {
    if stdout.limit_exceeded || stderr.limit_exceeded {
        let stream = match (stdout.limit_exceeded, stderr.limit_exceeded) {
            (true, true) => "stdout and stderr",
            (true, false) => "stdout",
            (false, true) => "stderr",
            (false, false) => unreachable!("validated output has no exceeded stream"),
        };
        return Err(FwError::ContractViolation(format!(
            "subprocess {stream} exceeded the {MAX_CAPTURED_OUTPUT_BYTES}-byte capture limit"
        )));
    }
    validate_command_output(
        rendered,
        Output {
            status,
            stdout: stdout.bytes,
            stderr: stderr.bytes,
        },
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::orchestrator::CancellationToken;

    use super::{
        cancellable_poll_delay, render_command_for_log, run_command_cancellable,
        run_command_cancellable_with_probe,
    };

    #[test]
    fn cancellable_poll_schedule_rejoins_fixed_cadence() {
        let delays: [u128; 8] =
            std::array::from_fn(|iteration| cancellable_poll_delay(iteration).as_millis());

        assert_eq!(delays, [1, 2, 4, 8, 16, 19, 50, 50]);
        assert_eq!(delays[..6].iter().sum::<u128>(), 50);
    }

    #[test]
    fn cancellable_completes_fast_command() {
        // A command that exits immediately should succeed with a far-future deadline.
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_secs(60));
        let result =
            run_command_cancellable("true", &[], None, &cancel, Some(Duration::from_secs(10)));
        assert!(result.is_ok(), "true should succeed: {result:?}");
    }

    #[test]
    fn cancellable_kills_on_expired_deadline() {
        // Create a token whose deadline is already in the past.
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        // Tiny sleep to ensure we're past the deadline.
        std::thread::sleep(Duration::from_millis(10));

        let result = run_command_cancellable(
            "sleep",
            &["60".to_owned()],
            None,
            &cancel,
            Some(Duration::from_secs(120)),
        );

        assert!(result.is_err(), "should be cancelled");
        let err = result.unwrap_err();
        assert!(
            matches!(err, crate::error::FwError::Cancelled(_)),
            "expected Cancelled error, got: {err:?}"
        );
    }

    #[test]
    fn cancellable_kills_on_additional_caller_predicate() {
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_secs(60));
        let polls = std::sync::atomic::AtomicUsize::new(0);
        let caller_cancelled = || polls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 2;
        let result = run_command_cancellable_with_probe(
            "sleep",
            &["60".to_owned()],
            None,
            &cancel,
            Some(Duration::from_secs(120)),
            Some(&caller_cancelled),
        );
        assert!(
            matches!(&result, Err(crate::error::FwError::Cancelled(_))),
            "caller predicate must cancel and reap the subprocess: {result:?}"
        );
    }

    #[test]
    fn cancellable_hard_timeout_takes_effect() {
        // Token with no deadline (far future), but hard timeout is tiny.
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_secs(600));
        let result = run_command_cancellable(
            "sleep",
            &["60".to_owned()],
            None,
            &cancel,
            Some(Duration::from_millis(100)),
        );

        assert!(result.is_err(), "should hit hard timeout");
        // Should NOT be Cancelled — should be a CommandTimeout.
        let err = result.unwrap_err();
        assert!(
            !matches!(err, crate::error::FwError::Cancelled(_)),
            "expected timeout error, not Cancelled: {err:?}"
        );
    }

    #[test]
    fn cancellable_no_deadline_still_works() {
        // Token with no deadline at all — should complete normally for fast commands.
        let cancel = CancellationToken::no_deadline();
        let result = run_command_cancellable("true", &[], None, &cancel, None);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // run_command / run_command_with_timeout / validate_command_output tests
    // -----------------------------------------------------------------------

    use super::{run_command, run_command_with_timeout, saturating_duration_ms};

    #[test]
    fn run_command_succeeds_for_true() {
        let output = run_command("true", &[], None).expect("true should succeed");
        assert!(output.status.success());
    }

    #[test]
    fn recv_pipe_output_reports_disconnect() {
        use super::{PipeCapture, recv_pipe_output};
        let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<PipeCapture>>();
        drop(tx);
        assert!(recv_pipe_output(rx).is_err());
    }

    #[test]
    fn recv_pipe_output_is_bounded_when_pipe_stays_open() {
        use super::{PipeCapture, recv_pipe_output_with_timeout};
        let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<PipeCapture>>();
        let started_at = std::time::Instant::now();
        let result = recv_pipe_output_with_timeout(rx, Duration::from_millis(5));
        assert!(matches!(
            result,
            Err(crate::error::FwError::ContractViolation(_))
        ));
        assert!(started_at.elapsed() < Duration::from_secs(1));
        drop(tx);
    }

    #[test]
    fn pipe_capture_drains_and_reports_output_over_limit() {
        use super::{MAX_CAPTURED_OUTPUT_BYTES, read_pipe_with_limit};
        let input = vec![b'x'; MAX_CAPTURED_OUTPUT_BYTES + 8_193];
        let capture = read_pipe_with_limit(std::io::Cursor::new(input)).expect("capture pipe");
        assert_eq!(capture.bytes.len(), MAX_CAPTURED_OUTPUT_BYTES);
        assert!(capture.limit_exceeded);
    }

    #[test]
    fn run_command_missing_program_returns_command_missing() {
        let err = run_command("nonexistent_binary_xyz_12345", &[], None)
            .expect_err("nonexistent binary should fail");
        assert!(
            matches!(err, crate::error::FwError::CommandMissing { .. }),
            "expected CommandMissing, got: {err:?}"
        );
    }

    #[test]
    fn run_command_nonzero_exit_returns_command_failed() {
        let err = run_command("false", &[], None).expect_err("false should fail");
        let text = err.to_string();
        assert!(
            text.contains("command failed") || text.contains("status"),
            "expected command failure message, got: {text}"
        );
    }

    #[test]
    fn run_command_with_timeout_succeeds_when_fast() {
        let output = run_command_with_timeout("true", &[], None, Some(Duration::from_secs(5)))
            .expect("true should succeed within timeout");
        assert!(output.status.success());
    }

    #[test]
    fn run_command_with_timeout_kills_slow_command() {
        let err = run_command_with_timeout(
            "sleep",
            &["60".to_owned()],
            None,
            Some(Duration::from_millis(100)),
        )
        .expect_err("should timeout");
        let text = err.to_string();
        assert!(
            text.contains("timed out") || text.contains("timeout"),
            "expected timeout message, got: {text}"
        );
    }

    #[test]
    fn run_command_captures_stderr() {
        // `ls` on a nonexistent path writes to stderr and exits non-zero.
        let err = run_command("ls", &["/nonexistent_path_xyz_99999".to_owned()], None)
            .expect_err("ls on nonexistent should fail");
        let text = err.to_string();
        assert!(
            text.contains("nonexistent_path") || text.contains("No such file"),
            "expected stderr content, got: {text}"
        );
    }

    #[test]
    fn run_command_with_cwd() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = run_command("pwd", &[], Some(dir.path())).expect("pwd should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(dir.path().to_str().unwrap()),
            "expected cwd in stdout, got: {stdout}"
        );
    }

    #[test]
    fn render_command_for_log_redacts_sensitive_flags() {
        let args = vec![
            "--hf-token".to_owned(),
            "hf_secret_123".to_owned(),
            "--api-key=secret_api_key".to_owned(),
            "--token-threshold".to_owned(),
            "0.1".to_owned(),
            "positional".to_owned(),
        ];
        let rendered = render_command_for_log("prog", &args);
        assert!(rendered.contains("--hf-token ***"));
        assert!(rendered.contains("--api-key=***"));
        assert!(rendered.contains("--token-threshold 0.1"));
        assert!(rendered.contains("positional"));
        assert!(
            !rendered.contains("hf_secret_123"),
            "hf token should be redacted"
        );
        assert!(
            !rendered.contains("secret_api_key"),
            "api key should be redacted"
        );
        assert_eq!(
            rendered,
            "prog --hf-token *** --api-key=*** --token-threshold 0.1 positional"
        );
        assert_eq!(render_command_for_log("prog", &[]), "prog");
    }

    #[test]
    fn saturating_duration_ms_normal_case() {
        assert_eq!(saturating_duration_ms(Duration::from_secs(5)), 5000);
        assert_eq!(saturating_duration_ms(Duration::from_millis(1234)), 1234);
    }

    #[test]
    fn saturating_duration_ms_max_does_not_panic() {
        let result = saturating_duration_ms(Duration::from_secs(u64::MAX));
        assert_eq!(result, u64::MAX);
    }

    // -----------------------------------------------------------------------
    // command_exists tests
    // -----------------------------------------------------------------------

    use super::command_exists;

    #[test]
    fn command_exists_true_for_known_binary() {
        // `ls` and `true` exist on all Unix-like systems.
        assert!(command_exists("ls"), "ls should exist");
        assert!(command_exists("true"), "true should exist");
    }

    #[test]
    fn command_exists_false_for_absent_binary() {
        assert!(
            !command_exists("definitely_not_a_real_binary_abc_xyz_99999"),
            "absent binary should not exist"
        );
    }

    // -----------------------------------------------------------------------
    // validate_command_output tests
    // -----------------------------------------------------------------------

    use super::validate_command_output;
    use std::os::unix::process::ExitStatusExt;
    use std::process::ExitStatus;

    fn fake_output(code: i32, stderr: &str) -> std::process::Output {
        std::process::Output {
            status: ExitStatus::from_raw(code << 8), // raw wait status: exit code in upper byte
            stdout: Vec::new(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn validate_command_output_success_returns_ok() {
        let output = fake_output(0, "");
        let result = validate_command_output("test-cmd", output);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_command_output_nonzero_exit_returns_error() {
        let output = fake_output(1, "something went wrong");
        let result = validate_command_output("test-cmd", output);
        assert!(result.is_err());
        let text = result.unwrap_err().to_string();
        assert!(
            text.contains("something went wrong"),
            "error should contain stderr, got: {text}"
        );
    }

    #[test]
    fn validate_command_output_preserves_exit_code_in_error() {
        let output = fake_output(42, "exit code 42");
        let err = validate_command_output("my-tool --flag", output).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("42"),
            "error should mention exit code 42, got: {text}"
        );
    }

    #[test]
    fn validate_command_output_empty_stderr_still_fails_on_nonzero() {
        let output = fake_output(2, "");
        let result = validate_command_output("cmd", output);
        assert!(
            result.is_err(),
            "non-zero exit with empty stderr should still fail"
        );
    }

    // ── Additional edge case tests ──

    #[test]
    fn run_command_with_timeout_none_behaves_like_run_command() {
        let output = run_command_with_timeout("true", &[], None, None).expect("should succeed");
        assert!(output.status.success());
    }

    #[test]
    fn run_command_with_args() {
        let output = run_command("echo", &["hello".to_owned(), "world".to_owned()], None)
            .expect("echo should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("hello world"),
            "expected 'hello world', got: {stdout}"
        );
    }

    #[test]
    fn cancellable_missing_program_returns_command_missing() {
        let cancel = CancellationToken::no_deadline();
        let err = run_command_cancellable("nonexistent_binary_xyz_99999", &[], None, &cancel, None)
            .expect_err("should fail");
        assert!(
            matches!(err, crate::error::FwError::CommandMissing { .. }),
            "expected CommandMissing, got: {err:?}"
        );
    }

    #[test]
    fn cancellable_captures_output_from_successful_command() {
        let cancel = CancellationToken::no_deadline();
        let output =
            run_command_cancellable("echo", &["test_output".to_owned()], None, &cancel, None)
                .expect("echo should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("test_output"),
            "should capture stdout: {stdout}"
        );
    }

    #[test]
    fn cancellable_nonzero_exit_returns_error() {
        let cancel = CancellationToken::no_deadline();
        let err = run_command_cancellable("false", &[], None, &cancel, None)
            .expect_err("false should fail");
        assert!(
            !matches!(err, crate::error::FwError::Cancelled(_)),
            "should not be cancelled, should be command failure: {err:?}"
        );
    }

    #[test]
    fn saturating_duration_ms_zero() {
        assert_eq!(saturating_duration_ms(Duration::ZERO), 0);
    }

    #[test]
    fn saturating_duration_ms_subsecond() {
        assert_eq!(saturating_duration_ms(Duration::from_millis(500)), 500);
        assert_eq!(saturating_duration_ms(Duration::from_millis(1)), 1);
    }

    #[test]
    fn validate_command_output_includes_command_name_in_error() {
        let output = fake_output(1, "boom");
        let err = validate_command_output("my-special-cmd --flag", output).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("my-special-cmd"),
            "error should mention command: {text}"
        );
    }

    #[test]
    fn cancellable_with_cwd_succeeds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cancel = CancellationToken::no_deadline();
        let output = run_command_cancellable("pwd", &[], Some(dir.path()), &cancel, None)
            .expect("pwd should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(dir.path().to_str().unwrap()),
            "expected cwd in stdout, got: {stdout}"
        );
    }

    #[test]
    fn run_command_empty_args_succeeds() {
        let output = run_command("true", &[], None).expect("true with no args");
        assert!(output.status.success());
    }

    #[test]
    fn cancellable_with_hard_timeout_none_and_no_deadline() {
        // Both safety nets disabled — should still work for fast commands.
        let cancel = CancellationToken::no_deadline();
        let output = run_command_cancellable("echo", &["ok".to_owned()], None, &cancel, None)
            .expect("should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("ok"));
    }

    #[test]
    fn run_command_preserves_large_stdout_payload() {
        let output = run_command(
            "sh",
            &["-c".to_owned(), "yes x | head -c 200000".to_owned()],
            None,
        )
        .expect("large stdout command should succeed");
        assert_eq!(
            output.stdout.len(),
            200_000,
            "stdout should be fully captured after process exit"
        );
    }

    #[test]
    fn run_command_preserves_large_stderr_payload_on_failure() {
        let err = run_command(
            "sh",
            &[
                "-c".to_owned(),
                "yes e | head -c 200000 >&2; exit 7".to_owned(),
            ],
            None,
        )
        .expect_err("command should fail with large stderr output");
        let text = err.to_string();
        assert!(
            text.len() > 100_000,
            "large stderr payload should remain materially intact"
        );
        assert!(
            text.contains("status: 7"),
            "exit status should be preserved"
        );
    }

    #[test]
    fn validate_command_output_signal_terminated_uses_negative_one() {
        // When a process is killed by a signal, exit code may not be available.
        // On Unix, from_raw(9) represents SIGKILL (signal 9, no exit code).
        let output = std::process::Output {
            status: ExitStatus::from_raw(9), // signal 9 (SIGKILL), no exit code
            stdout: Vec::new(),
            stderr: b"killed".to_vec(),
        };
        let result = validate_command_output("signaled-cmd", output);
        assert!(result.is_err(), "signal-killed process should fail");
        let text = result.unwrap_err().to_string();
        // The code falls back to -1 when .code() returns None.
        assert!(
            text.contains("-1") || text.contains("killed"),
            "should mention -1 or killed: {text}"
        );
    }

    #[test]
    fn run_command_with_timeout_missing_program_returns_command_missing() {
        let err = run_command_with_timeout(
            "nonexistent_xyz_99",
            &[],
            None,
            Some(Duration::from_secs(5)),
        )
        .expect_err("should fail");
        assert!(matches!(err, crate::error::FwError::CommandMissing { .. }));
    }
}
