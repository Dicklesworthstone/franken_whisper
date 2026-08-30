use std::io::{Read, Seek, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{FwError, FwResult};

#[cfg(not(windows))]
type ManagedChild = std::process::Child;
#[cfg(windows)]
type ManagedChild = Box<dyn process_wrap::std::ChildWrapper>;

#[cfg(not(windows))]
fn spawn_managed_child(mut command: Command) -> std::io::Result<ManagedChild> {
    command.spawn()
}

#[cfg(windows)]
fn spawn_managed_child(command: Command) -> std::io::Result<ManagedChild> {
    use process_wrap::std::{CommandWrap, JobObject};

    let mut command = CommandWrap::from(command);
    command.wrap(JobObject);
    let mut spawned_pid = None;
    let spawn = command.spawn_with(|command| {
        let child = command.spawn()?;
        spawned_pid = Some(child.id());
        Ok(child)
    });
    match spawn {
        Ok(child) => Ok(child),
        Err(error) => {
            if let Some(pid) = spawned_pid {
                terminate_failed_windows_job_assignment(pid).map_err(|cleanup| {
                    std::io::Error::other(format!(
                        "Windows Job Object setup failed ({error}); suspended-root cleanup also failed ({cleanup})"
                    ))
                })?;
            }
            Err(error)
        }
    }
}

#[cfg(windows)]
fn terminate_failed_windows_job_assignment(pid: u32) -> std::io::Result<()> {
    const CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);

    let pid = pid.to_string();
    let mut cleanup = Command::new("taskkill")
        .args(["/PID", pid.as_str(), "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let started_at = Instant::now();
    let status = loop {
        match cleanup.try_wait()? {
            Some(status) => break status,
            None if started_at.elapsed() < CLEANUP_TIMEOUT => {
                thread::sleep(Duration::from_millis(10));
            }
            None => {
                let _ = cleanup.kill();
                let _ = cleanup.wait();
                return Err(std::io::Error::other(
                    "taskkill exceeded the suspended-root cleanup deadline",
                ));
            }
        }
    };
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(
            "taskkill rejected suspended-root cleanup",
        ))
    }
}

#[cfg(unix)]
static PROCESS_TREE_EXTERNALLY_OWNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(unix)]
const PARENT_LIVENESS_FD_ENV: &str = "FRANKEN_WHISPER_PARENT_LIVENESS_FD";
#[cfg(unix)]
const PARENT_LIVENESS_PID_ENV: &str = "FRANKEN_WHISPER_PARENT_LIVENESS_PID";

#[cfg(unix)]
struct PreparedParentLivenessLease {
    reader: std::os::fd::OwnedFd,
    writer: std::os::fd::OwnedFd,
}

#[cfg(unix)]
impl PreparedParentLivenessLease {
    fn activate(self) -> std::os::fd::OwnedFd {
        let Self { reader, writer } = self;
        drop(reader);
        writer
    }
}

#[cfg(unix)]
fn prepare_parent_liveness_lease(command: &mut Command) -> FwResult<PreparedParentLivenessLease> {
    use std::os::fd::AsRawFd as _;

    let (reader, writer) = rustix::pipe::pipe().map_err(std::io::Error::from)?;
    if reader.as_raw_fd() <= 2 {
        return Err(FwError::Unsupported(
            "parent-liveness authority requires an inherited descriptor above standard I/O"
                .to_owned(),
        ));
    }
    rustix::io::fcntl_setfd(&reader, rustix::io::FdFlags::empty()).map_err(std::io::Error::from)?;
    rustix::io::fcntl_setfd(&writer, rustix::io::FdFlags::CLOEXEC).map_err(std::io::Error::from)?;
    command.env(PARENT_LIVENESS_FD_ENV, reader.as_raw_fd().to_string());
    command.env(PARENT_LIVENESS_PID_ENV, std::process::id().to_string());
    Ok(PreparedParentLivenessLease { reader, writer })
}

#[cfg(not(unix))]
fn prepare_parent_liveness_lease(_command: &mut Command) -> FwResult<()> {
    Err(FwError::Unsupported(
        "parent-liveness authority is unsupported on this platform".to_owned(),
    ))
}

#[cfg(unix)]
fn inherited_parent_liveness_reader() -> FwResult<std::fs::File> {
    use std::os::unix::fs::FileTypeExt as _;

    let fd_text = std::env::var(PARENT_LIVENESS_FD_ENV).map_err(|_| {
        FwError::ContractViolation(
            "an externally owned subprocess tree requires an inherited parent-liveness descriptor"
                .to_owned(),
        )
    })?;
    let fd = fd_text
        .parse::<i32>()
        .ok()
        .filter(|fd| *fd > 2)
        .ok_or_else(|| {
            FwError::ContractViolation(
                "the inherited parent-liveness descriptor is outside the accepted range".to_owned(),
            )
        })?;
    let expected_parent = std::env::var(PARENT_LIVENESS_PID_ENV)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .and_then(rustix::process::Pid::from_raw)
        .ok_or_else(|| {
            FwError::ContractViolation(
                "the inherited parent-liveness authority has an invalid parent identity".to_owned(),
            )
        })?;
    if rustix::process::getppid() != Some(expected_parent) {
        return Err(FwError::ContractViolation(
            "the inherited parent-liveness authority does not belong to the direct parent"
                .to_owned(),
        ));
    }

    let mut last_error = None;
    for root in ["/dev/fd", "/proc/self/fd"] {
        match std::fs::File::open(Path::new(root).join(fd.to_string())) {
            Ok(file) => {
                let metadata = file.metadata().map_err(FwError::Io)?;
                if !metadata.file_type().is_fifo() {
                    return Err(FwError::ContractViolation(
                        "the inherited parent-liveness descriptor is not a kernel pipe".to_owned(),
                    ));
                }
                return Ok(file);
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(FwError::ContractViolation(format!(
        "the inherited parent-liveness descriptor could not be opened: {}",
        last_error
            .map(|error| error.kind().to_string())
            .unwrap_or_else(|| "unavailable".to_owned())
    )))
}

#[cfg(unix)]
fn start_parent_liveness_watcher(mut reader: std::fs::File) -> FwResult<()> {
    thread::Builder::new()
        .name("fw-parent-liveness".to_owned())
        .spawn(move || {
            let mut byte = [0u8; 1];
            loop {
                match reader.read(&mut byte) {
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Ok(0) | Ok(_) | Err(_) => {
                        let process_group = rustix::process::getpgrp();
                        let _ = rustix::process::kill_process_group(
                            process_group,
                            rustix::process::Signal::KILL,
                        );
                        std::process::exit(125);
                    }
                }
            }
        })
        .map(drop)
        .map_err(|_| {
            FwError::ContractViolation(
                "the parent-liveness watcher could not be started".to_owned(),
            )
        })
}

/// Authenticate that this process belongs to a bounded parent-owned Unix
/// process group and start the inherited parent-liveness watcher. Nested
/// subprocesses then inherit that group instead of escaping into a new one;
/// their direct child is still reaped locally, while the outer owner remains
/// authoritative for recursive termination. Parent death closes the sole
/// lease writer and makes the group root terminate the complete group.
#[cfg(unix)]
pub(crate) fn mark_process_tree_externally_owned() -> FwResult<()> {
    if rustix::process::getpgrp() != rustix::process::getpid() {
        return Err(FwError::ContractViolation(
            "an externally owned subprocess tree must enter through a fresh process-group root"
                .to_owned(),
        ));
    }
    let reader = inherited_parent_liveness_reader()?;
    if PROCESS_TREE_EXTERNALLY_OWNED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        return Err(FwError::ContractViolation(
            "the externally owned subprocess tree was already initialized".to_owned(),
        ));
    }
    if let Err(error) = start_parent_liveness_watcher(reader) {
        PROCESS_TREE_EXTERNALLY_OWNED.store(false, std::sync::atomic::Ordering::Release);
        return Err(error);
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn mark_process_tree_externally_owned() -> FwResult<()> {
    Err(FwError::Unsupported(
        "parent-liveness authority is unsupported on this platform".to_owned(),
    ))
}

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
            if let Some((flag, _value)) = arg.split_once('=')
                && is_sensitive_flag(flag)
            {
                capacity = capacity.saturating_add(flag.len().saturating_add(4));
                redact_next = false;
                continue;
            }
            if is_sensitive_flag(arg) {
                capacity = capacity.saturating_add(arg.len());
                continue;
            }
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
            if let Some((flag, _value)) = arg.split_once('=')
                && is_sensitive_flag(flag)
            {
                rendered.push_str(flag);
                rendered.push_str("=***");
                redact_next = false;
                continue;
            }
            if is_sensitive_flag(arg) {
                rendered.push_str(arg);
                continue;
            }
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
            | "--prompt"
            | "--secret"
            | "--secret-key"
            | "--secret_key"
    )
}

fn sensitive_arg_values(args: &[String]) -> Vec<String> {
    let mut values = Vec::new();
    let mut capture_next = false;

    for arg in args {
        if capture_next {
            if let Some((flag, value)) = arg.split_once('=')
                && is_sensitive_flag(flag)
            {
                if !value.is_empty() {
                    values.push(value.to_owned());
                }
                capture_next = false;
                continue;
            }
            if is_sensitive_flag(arg) {
                continue;
            }
            if !arg.is_empty() {
                values.push(arg.clone());
            }
            capture_next = false;
            continue;
        }

        if let Some((flag, value)) = arg.split_once('=')
            && is_sensitive_flag(flag)
        {
            if !value.is_empty() {
                values.push(value.to_owned());
            }
            continue;
        }

        capture_next = is_sensitive_flag(arg);
    }

    sort_sensitive_values(&mut values);
    values
}

fn sort_sensitive_values(values: &mut Vec<String>) {
    // Replace longer values first so an overlapping short secret cannot leave
    // the suffix of a longer secret visible. The lexical tie-breaker makes the
    // order deterministic and groups duplicates for `dedup`.
    values
        .sort_unstable_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));
    values.dedup();
}

const SENSITIVE_INHERITED_ENVIRONMENT: [&str; 5] = [
    "FRANKEN_WHISPER_HF_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "FW_INITIAL_PROMPT",
];

fn redact_sensitive_text(text: &str, sensitive_values: &[String]) -> String {
    let replacement: String = std::iter::repeat_n(redaction_marker(sensitive_values), 3).collect();
    let mut redacted = text.to_owned();
    for value in sensitive_values {
        redacted = redacted.replace(value.as_str(), &replacement);
    }
    redacted
}

fn redaction_marker(sensitive_values: &[String]) -> char {
    const PREFERRED: [char; 5] = ['*', '#', '•', '█', '�'];
    let used: std::collections::HashSet<char> = sensitive_values
        .iter()
        .flat_map(|value| value.chars())
        .collect();
    PREFERRED
        .into_iter()
        .find(|candidate| !used.contains(candidate))
        .or_else(|| ('\u{e000}'..='\u{f8ff}').find(|candidate| !used.contains(candidate)))
        // NUL cannot be passed to a spawned process argument, so even a
        // pathological value containing every visible fallback cannot equal
        // this final separator.
        .unwrap_or('\0')
}

fn sensitive_byte_mask(bytes: &[u8], sensitive_values: &[String]) -> Vec<bool> {
    let mut mask = vec![false; bytes.len()];
    for value in sensitive_values {
        let needle = value.as_bytes();
        if needle.is_empty() {
            continue;
        }
        mark_sensitive_matches(bytes, needle, &mut mask);
    }
    mask
}

fn mark_sensitive_matches(bytes: &[u8], needle: &[u8], mask: &mut [bool]) {
    debug_assert!(!needle.is_empty());
    debug_assert_eq!(bytes.len(), mask.len());

    let mut prefix = vec![0usize; needle.len()];
    for index in 1..needle.len() {
        let mut matched = prefix[index - 1];
        while matched > 0 && needle[index] != needle[matched] {
            matched = prefix[matched - 1];
        }
        if needle[index] == needle[matched] {
            matched += 1;
        }
        prefix[index] = matched;
    }

    let mut matched = 0usize;
    let mut pending_range: Option<(usize, usize)> = None;
    for (index, byte) in bytes.iter().enumerate() {
        while matched > 0 && *byte != needle[matched] {
            matched = prefix[matched - 1];
        }
        if *byte == needle[matched] {
            matched += 1;
        }
        if matched == needle.len() {
            let end = index + 1;
            let start = end - needle.len();
            pending_range = match pending_range.take() {
                Some((range_start, range_end)) if start <= range_end => {
                    Some((range_start, range_end.max(end)))
                }
                Some((range_start, range_end)) => {
                    mask[range_start..range_end].fill(true);
                    Some((start, end))
                }
                None => Some((start, end)),
            };
            matched = prefix[matched - 1];
        }
    }
    if let Some((start, end)) = pending_range {
        mask[start..end].fill(true);
    }
}

fn render_redacted_bytes(bytes: &[u8], mask: &[bool], start: usize, marker: char) -> String {
    debug_assert_eq!(bytes.len(), mask.len());
    let mut rendered = Vec::with_capacity(bytes.len().saturating_sub(start));
    let mut marker_buf = [0u8; 4];
    let marker = marker.encode_utf8(&mut marker_buf).as_bytes();
    let mut index = start.min(bytes.len());
    while index < bytes.len() {
        if mask[index] {
            for _ in 0..3 {
                rendered.extend_from_slice(marker);
            }
            index += 1;
            while index < bytes.len() && mask[index] {
                index += 1;
            }
        } else {
            rendered.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&rendered).into_owned()
}

fn captured_stderr_for_error(stderr: &[u8], sensitive_values: &[String]) -> String {
    let mask = sensitive_byte_mask(stderr, sensitive_values);
    let rendered = render_redacted_bytes(stderr, &mask, 0, redaction_marker(sensitive_values));
    redact_sensitive_text(&rendered, sensitive_values)
}

fn command_error_diagnostics(program: &str, args: &[String]) -> (String, Vec<String>) {
    command_error_diagnostics_with_environment(program, args, |name| std::env::var(name).ok())
}

fn command_error_diagnostics_with_environment<F>(
    program: &str,
    args: &[String],
    mut environment_value: F,
) -> (String, Vec<String>)
where
    F: FnMut(&str) -> Option<String>,
{
    let mut sensitive_values = sensitive_arg_values(args);
    sensitive_values.extend(
        SENSITIVE_INHERITED_ENVIRONMENT
            .iter()
            .filter_map(|name| environment_value(name))
            .filter(|value| !value.is_empty()),
    );
    sort_sensitive_values(&mut sensitive_values);
    let rendered = render_command_for_log(program, args);
    let rendered = redact_sensitive_text(&rendered, &sensitive_values);
    (rendered, sensitive_values)
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

    let (rendered, sensitive_values) = command_error_diagnostics(program, args);
    let mut command = Command::new(program);
    command.args(args);
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }
    command.stdin(Stdio::null());

    if bounded_process_tree_unsupported() {
        return Err(FwError::Unsupported(
            "bounded subprocess trees are unsupported on this platform".to_owned(),
        ));
    }
    let prepared_capture = prepare_bounded_output_capture(&mut command)?;
    let owns_process_group = configure_descendant_process_tree(&mut command);
    let mut child = spawn_managed_child(command)?;
    let started_at = Instant::now();
    let (stdout_reader, stderr_reader) =
        match start_bounded_output_capture(&mut child, prepared_capture, &rendered) {
            Ok(readers) => readers,
            Err(error) => {
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                return Err(merge_process_tree_cleanup_result(error, cleanup));
            }
        };

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Clean the platform ownership boundary even after the root
                // exits successfully, so in-bound descendants cannot retain
                // inherited pipes or continue operator-local work.
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                let stdout_result = stdout_reader.finish();
                let stderr_result = stderr_reader.finish();
                cleanup?;
                let stdout = stdout_result?;
                let stderr = stderr_result?;
                return validate_captured_command_output(
                    &rendered,
                    status,
                    stdout,
                    stderr,
                    &sensitive_values,
                );
            }
            Ok(None) => {}
            Err(error) => {
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                let _ = stdout_reader.finish();
                let _ = stderr_reader.finish();
                return Err(merge_process_tree_cleanup_result(
                    FwError::Io(error),
                    cleanup,
                ));
            }
        }

        match bounded_output_limit_stream(&stdout_reader, &stderr_reader) {
            Ok(Some(stream)) => {
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                let _ = stdout_reader.finish();
                let _ = stderr_reader.finish();
                return Err(merge_process_tree_cleanup_result(
                    capture_limit_error(stream),
                    cleanup,
                ));
            }
            Ok(None) => {}
            Err(error) => {
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                let _ = stdout_reader.finish();
                let _ = stderr_reader.finish();
                return Err(merge_process_tree_cleanup_result(error, cleanup));
            }
        }

        if let Some(limit) = timeout
            && started_at.elapsed() >= limit
        {
            let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
            let _ = stdout_reader.finish();
            let stderr = stderr_reader
                .finish()
                .map(|capture| capture.bytes)
                .unwrap_or_default();
            let stderr_str = captured_stderr_for_error(&stderr, &sensitive_values);
            return Err(merge_process_tree_cleanup_result(
                FwError::from_command_timeout(
                    rendered,
                    saturating_duration_ms(limit),
                    stderr_str,
                ),
                cleanup,
            ));
        }

        thread::sleep(Duration::from_millis(20));
    }
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
    run_command_cancellable_with_optional_input(
        program,
        args,
        cwd,
        token,
        hard_timeout,
        additional_cancel,
        None,
        None,
    )
}

/// Run a cancellable subprocess with one bounded stdin document. The payload
/// is staged in an anonymous temporary file before launch, so a child that
/// stops reading cannot leave a blocked writer thread behind after timeout or
/// cancellation.
#[cfg(test)]
pub(crate) fn run_command_cancellable_with_input_and_probe(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    token: &crate::orchestrator::CancellationToken,
    hard_timeout: Option<Duration>,
    additional_cancel: Option<&(dyn Fn() -> bool + Sync)>,
    stdin_payload: &[u8],
) -> FwResult<Output> {
    run_command_cancellable_with_optional_input(
        program,
        args,
        cwd,
        token,
        hard_timeout,
        additional_cancel,
        Some(stdin_payload),
        None,
    )
}

/// Run a cancellable subprocess with bounded stdin while observing the complete
/// child process group at the normal polling cadence.
///
/// The observer receives the root child PID. On Unix this API fails closed if
/// the caller is already inside an externally owned process group; otherwise
/// the child PID is also the fresh process-group identifier created with
/// `process_group(0)`. The child also inherits the read end of a liveness pipe
/// whose sole writer remains in this direct parent. Returning an error
/// terminates and reaps the entire tree.
pub(crate) fn run_command_cancellable_with_input_probe_and_observer(
    program: &str,
    args: &[String],
    token: &crate::orchestrator::CancellationToken,
    hard_timeout: Option<Duration>,
    additional_cancel: Option<&(dyn Fn() -> bool + Sync)>,
    stdin_payload: &[u8],
    observer: &mut dyn FnMut(u32) -> FwResult<()>,
) -> FwResult<Output> {
    run_command_cancellable_with_optional_input(
        program,
        args,
        None,
        token,
        hard_timeout,
        additional_cancel,
        Some(stdin_payload),
        Some(observer),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_command_cancellable_with_optional_input(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    token: &crate::orchestrator::CancellationToken,
    hard_timeout: Option<Duration>,
    additional_cancel: Option<&(dyn Fn() -> bool + Sync)>,
    stdin_payload: Option<&[u8]>,
    mut observer: Option<&mut dyn FnMut(u32) -> FwResult<()>>,
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

    let (rendered, sensitive_values) = command_error_diagnostics(program, args);
    let mut command = Command::new(program);
    command.args(args);
    if bounded_process_tree_unsupported() {
        return Err(FwError::Unsupported(
            "bounded subprocess trees are unsupported on this platform".to_owned(),
        ));
    }
    if let Some(payload) = stdin_payload {
        let mut stdin_file = tempfile::tempfile()?;
        stdin_file.write_all(payload)?;
        stdin_file.flush()?;
        stdin_file.seek(std::io::SeekFrom::Start(0))?;
        command.stdin(Stdio::from(stdin_file));
    } else {
        command.stdin(Stdio::null());
    }
    let prepared_capture = prepare_bounded_output_capture(&mut command)?;
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }

    #[cfg(unix)]
    let prepared_parent_liveness_lease = observer
        .is_some()
        .then(|| prepare_parent_liveness_lease(&mut command))
        .transpose()?;
    #[cfg(not(unix))]
    if observer.is_some() {
        prepare_parent_liveness_lease(&mut command)?;
    }

    let owns_process_group = configure_descendant_process_tree(&mut command);
    #[cfg(unix)]
    if observer.is_some() && !owns_process_group {
        return Err(FwError::Unsupported(
            "process-tree observation requires a fresh caller-owned process group".to_owned(),
        ));
    }
    let mut child = spawn_managed_child(command)?;
    #[cfg(unix)]
    let _parent_liveness_lease =
        prepared_parent_liveness_lease.map(PreparedParentLivenessLease::activate);
    let started_at = Instant::now();
    let mut poll_iteration = 0usize;
    let (stdout_reader, stderr_reader) =
        match start_bounded_output_capture(&mut child, prepared_capture, &rendered) {
            Ok(readers) => readers,
            Err(error) => {
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                return Err(merge_process_tree_cleanup_result(error, cleanup));
            }
        };
    loop {
        if let Err(err) = token.checkpoint() {
            let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
            let _ = stdout_reader.finish();
            let _ = stderr_reader.finish();
            return Err(merge_process_tree_cleanup_result(err, cleanup));
        }
        if additional_cancel.is_some_and(|probe| probe()) {
            let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
            let _ = stdout_reader.finish();
            let _ = stderr_reader.finish();
            return Err(merge_process_tree_cleanup_result(
                FwError::Cancelled("subprocess cancelled by caller predicate".to_owned()),
                cleanup,
            ));
        }

        // Hard timeout safety net.
        if let Some(limit) = hard_timeout
            && started_at.elapsed() >= limit
        {
            let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
            let _ = stdout_reader.finish();
            let stderr = stderr_reader
                .finish()
                .map(|capture| capture.bytes)
                .unwrap_or_default();
            let stderr_str = captured_stderr_for_error(&stderr, &sensitive_values);
            return Err(merge_process_tree_cleanup_result(
                FwError::from_command_timeout(rendered, saturating_duration_ms(limit), stderr_str),
                cleanup,
            ));
        }

        match bounded_output_limit_stream(&stdout_reader, &stderr_reader) {
            Ok(Some(stream)) => {
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                let _ = stdout_reader.finish();
                let _ = stderr_reader.finish();
                return Err(merge_process_tree_cleanup_result(
                    capture_limit_error(stream),
                    cleanup,
                ));
            }
            Ok(None) => {}
            Err(error) => {
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                let _ = stdout_reader.finish();
                let _ = stderr_reader.finish();
                return Err(merge_process_tree_cleanup_result(error, cleanup));
            }
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                // Close inherited stdin/stdout/stderr in any descendants before
                // joining I/O helpers; otherwise a successful root could leave
                // the bounded caller blocked forever.
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                let stdout_result = stdout_reader.finish();
                let stderr_result = stderr_reader.finish();
                cleanup?;
                let stdout = stdout_result?;
                let stderr = stderr_result?;
                return validate_captured_command_output(
                    &rendered,
                    status,
                    stdout,
                    stderr,
                    &sensitive_values,
                );
            }
            Ok(None) => {}
            Err(error) => {
                let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
                let _ = stdout_reader.finish();
                let _ = stderr_reader.finish();
                return Err(merge_process_tree_cleanup_result(
                    FwError::Io(error),
                    cleanup,
                ));
            }
        }

        if let Some(observer) = observer.as_deref_mut()
            && let Err(error) = observer(child.id())
        {
            let cleanup = terminate_descendant_process_tree(&mut child, owns_process_group);
            let _ = stdout_reader.finish();
            let _ = stderr_reader.finish();
            return Err(merge_process_tree_cleanup_result(error, cleanup));
        }

        thread::sleep(cancellable_poll_delay(poll_iteration));
        poll_iteration = poll_iteration.saturating_add(1);
    }
}

/// Put each bounded child at the root of a process tree that cancellation can
/// terminate as one unit. On Unix, `process_group(0)` creates a group whose id
/// is the child pid. Windows tree ownership is established later by
/// `spawn_managed_child`, which assigns the suspended root to a Job Object
/// before allowing it to run.
fn configure_descendant_process_tree(command: &mut Command) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;

        let externally_owned =
            PROCESS_TREE_EXTERNALLY_OWNED.load(std::sync::atomic::Ordering::Acquire);
        if !externally_owned {
            command.process_group(0);
        }
        !externally_owned
    }
    #[cfg(not(unix))]
    {
        let _ = command;
        false
    }
}

const fn bounded_process_tree_unsupported() -> bool {
    !cfg!(any(unix, windows))
}

#[cfg(not(windows))]
const PROCESS_TREE_CLEANUP_TIMEOUT: Duration = Duration::from_millis(500);

fn process_tree_cleanup_error(detail: &str) -> FwError {
    FwError::ContractViolation(format!(
        "bounded subprocess cleanup could not certify the complete process tree: {detail}"
    ))
}

fn combine_process_tree_cleanup_error(primary: FwError, cleanup: FwError) -> FwError {
    FwError::ContractViolation(format!(
        "{cleanup}; the original subprocess outcome was: {primary}"
    ))
}

fn merge_process_tree_cleanup_result(primary: FwError, cleanup: FwResult<()>) -> FwError {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup) => combine_process_tree_cleanup_error(primary, cleanup),
    }
}

#[cfg(not(windows))]
fn wait_for_child_reap(child: &mut ManagedChild, deadline: Instant) -> FwResult<()> {
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
            Ok(None) => {
                return Err(process_tree_cleanup_error(
                    "the root child did not terminate before the cleanup deadline",
                ));
            }
            Err(_) => {
                return Err(process_tree_cleanup_error(
                    "the root child could not be monitored while being reaped",
                ));
            }
        }
    }
}

#[cfg(unix)]
fn terminate_descendant_process_tree(
    child: &mut ManagedChild,
    owns_process_group: bool,
) -> FwResult<()> {
    let deadline = Instant::now() + PROCESS_TREE_CLEANUP_TIMEOUT;
    let process_group = owns_process_group.then(|| rustix::process::Pid::from_child(child));
    let mut group_signal_error = None;
    if let Some(process_group) = process_group
        && let Err(error) =
            rustix::process::kill_process_group(process_group, rustix::process::Signal::KILL)
        && error != rustix::io::Errno::SRCH
    {
        group_signal_error = Some(error);
    }

    match child.try_wait() {
        Ok(Some(_)) => {}
        Ok(None) => {
            if let Err(error) = child.kill()
                && error.kind() != std::io::ErrorKind::InvalidInput
            {
                return Err(process_tree_cleanup_error(
                    "the root child rejected direct termination",
                ));
            }
        }
        Err(_) => {
            return Err(process_tree_cleanup_error(
                "the root child could not be inspected before termination",
            ));
        }
    }
    wait_for_child_reap(child, deadline)?;

    if let Some(process_group) = process_group {
        loop {
            match rustix::process::test_kill_process_group(process_group) {
                Err(error) if error == rustix::io::Errno::SRCH => break,
                Err(error)
                    if error == rustix::io::Errno::INTR && Instant::now() < deadline =>
                {
                    // Signal probes can be interrupted on Unix. That says
                    // nothing about whether the group survived; retry within
                    // the same bounded certification window.
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => {
                    return Err(process_tree_cleanup_error(&format!(
                        "the owned Unix process group could not be inspected after termination ({error})"
                    )));
                }
                Ok(()) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(()) => {
                    if let Some(error) = group_signal_error {
                        return Err(process_tree_cleanup_error(&format!(
                            "the owned Unix process group rejected termination ({error}) and remained alive"
                        )));
                    }
                    return Err(process_tree_cleanup_error(
                        "the owned Unix process group remained alive after termination",
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(windows)]
fn terminate_descendant_process_tree(
    child: &mut ManagedChild,
    _owns_process_group: bool,
) -> FwResult<()> {
    child.kill().map_err(|_| {
        process_tree_cleanup_error("the owned Windows Job Object rejected termination")
    })
}

#[cfg(not(any(unix, windows)))]
fn terminate_descendant_process_tree(
    child: &mut ManagedChild,
    _owns_process_group: bool,
) -> FwResult<()> {
    child
        .kill()
        .map_err(|_| process_tree_cleanup_error("the root child rejected direct termination"))?;
    wait_for_child_reap(child, Instant::now() + PROCESS_TREE_CLEANUP_TIMEOUT)
}

fn validate_command_output(
    rendered: &str,
    output: Output,
    sensitive_values: &[String],
) -> FwResult<Output> {
    if output.status.success() {
        return Ok(output);
    }

    let status = output.status.code().unwrap_or(-1);
    let stderr = captured_stderr_for_error(&output.stderr, sensitive_values);
    Err(FwError::from_command_failure(
        rendered.to_owned(),
        status,
        stderr,
    ))
}

fn saturating_duration_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

pub(crate) const MAX_CAPTURED_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
const PIPE_CAPTURE_STOP_DRAIN_GRACE: Duration = Duration::from_millis(100);

#[derive(Debug)]
struct PipeCapture {
    bytes: Vec<u8>,
    limit_exceeded: bool,
    drain_incomplete: bool,
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
struct BoundedOutputReader {
    receiver: std::sync::mpsc::Receiver<std::io::Result<PipeCapture>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    limit_exceeded: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: thread::JoinHandle<()>,
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
impl BoundedOutputReader {
    fn finish(self) -> FwResult<PipeCapture> {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        let result = recv_pipe_output(self.receiver);
        if self.handle.join().is_err() {
            return Err(FwError::Io(std::io::Error::other(
                "subprocess pipe reader panicked",
            )));
        }
        result
    }
}

fn bounded_output_limit_stream(
    stdout: &BoundedOutputReader,
    stderr: &BoundedOutputReader,
) -> FwResult<Option<&'static str>> {
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        let stdout_exceeded = stdout
            .limit_exceeded
            .load(std::sync::atomic::Ordering::Acquire);
        let stderr_exceeded = stderr
            .limit_exceeded
            .load(std::sync::atomic::Ordering::Acquire);
        Ok(match (stdout_exceeded, stderr_exceeded) {
            (true, true) => Some("stdout and stderr"),
            (true, false) => Some("stdout"),
            (false, true) => Some("stderr"),
            (false, false) => None,
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    {
        let stdout_exceeded = regular_file_exceeds_capture_limit(&stdout.file)?;
        let stderr_exceeded = regular_file_exceeds_capture_limit(&stderr.file)?;
        Ok(match (stdout_exceeded, stderr_exceeded) {
            (true, true) => Some("stdout and stderr"),
            (true, false) => Some("stdout"),
            (false, true) => Some("stderr"),
            (false, false) => None,
        })
    }
}

#[cfg(any(
    test,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
fn regular_file_exceeds_capture_limit(file: &std::fs::File) -> FwResult<bool> {
    Ok(file.metadata()?.len() > MAX_CAPTURED_OUTPUT_BYTES as u64)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
struct BoundedOutputReader {
    file: std::fs::File,
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
impl BoundedOutputReader {
    fn finish(self) -> FwResult<PipeCapture> {
        read_bounded_output_file(self.file)
    }
}

#[cfg(any(
    test,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
fn read_bounded_output_file(mut file: std::fs::File) -> FwResult<PipeCapture> {
    if regular_file_exceeds_capture_limit(&file)? {
        return Ok(PipeCapture {
            bytes: Vec::new(),
            limit_exceeded: true,
            drain_incomplete: false,
        });
    }
    file.seek(std::io::SeekFrom::Start(0))?;
    read_pipe_with_limit(file).map_err(FwError::Io)
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
struct PreparedBoundedOutputCapture;

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
struct PreparedBoundedOutputCapture {
    stdout: std::fs::File,
    stderr: std::fs::File,
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn prepare_bounded_output_capture(command: &mut Command) -> FwResult<PreparedBoundedOutputCapture> {
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    Ok(PreparedBoundedOutputCapture)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn prepare_bounded_output_capture(command: &mut Command) -> FwResult<PreparedBoundedOutputCapture> {
    let stdout = tempfile::tempfile()?;
    let stderr = tempfile::tempfile()?;
    command.stdout(Stdio::from(stdout.try_clone()?));
    command.stderr(Stdio::from(stderr.try_clone()?));
    Ok(PreparedBoundedOutputCapture { stdout, stderr })
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn start_bounded_output_capture(
    child: &mut ManagedChild,
    _prepared: PreparedBoundedOutputCapture,
    rendered: &str,
) -> FwResult<(BoundedOutputReader, BoundedOutputReader)> {
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
    Ok((
        spawn_pipe_reader(stdout_pipe),
        spawn_pipe_reader(stderr_pipe),
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn start_bounded_output_capture(
    _child: &mut ManagedChild,
    prepared: PreparedBoundedOutputCapture,
    _rendered: &str,
) -> FwResult<(BoundedOutputReader, BoundedOutputReader)> {
    Ok((
        BoundedOutputReader {
            file: prepared.stdout,
        },
        BoundedOutputReader {
            file: prepared.stderr,
        },
    ))
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn spawn_pipe_reader<R>(pipe: R) -> BoundedOutputReader
where
    R: Read + Send + std::os::fd::AsFd + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let limit_exceeded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader_stop = std::sync::Arc::clone(&stop);
    let reader_limit_exceeded = std::sync::Arc::clone(&limit_exceeded);
    let handle = thread::spawn(move || {
        let result = set_pipe_nonblocking(&pipe).and_then(|()| {
            read_pipe_with_limit_until_stopped(pipe, &reader_stop, &reader_limit_exceeded)
        });
        let _ = tx.send(result);
    });
    BoundedOutputReader {
        receiver: rx,
        stop,
        limit_exceeded,
        handle,
    }
}

#[cfg(any(
    test,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
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
        drain_incomplete: false,
    })
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn set_pipe_nonblocking<Fd: std::os::fd::AsFd>(pipe: &Fd) -> std::io::Result<()> {
    let flags = rustix::fs::fcntl_getfl(pipe).map_err(errno_to_io_error)?;
    rustix::fs::fcntl_setfl(pipe, flags | rustix::fs::OFlags::NONBLOCK).map_err(errno_to_io_error)
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn errno_to_io_error(error: rustix::io::Errno) -> std::io::Error {
    std::io::Error::from_raw_os_error(error.raw_os_error())
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn read_pipe_with_limit_until_stopped<R: Read>(
    pipe: R,
    stop: &std::sync::atomic::AtomicBool,
    limit_exceeded: &std::sync::atomic::AtomicBool,
) -> std::io::Result<PipeCapture> {
    read_pipe_with_limit_until_stopped_with_grace(
        pipe,
        stop,
        limit_exceeded,
        PIPE_CAPTURE_STOP_DRAIN_GRACE,
    )
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn read_pipe_with_limit_until_stopped_with_grace<R: Read>(
    mut pipe: R,
    stop: &std::sync::atomic::AtomicBool,
    limit_exceeded_signal: &std::sync::atomic::AtomicBool,
    stop_drain_grace: Duration,
) -> std::io::Result<PipeCapture> {
    let mut buf = [0u8; 8192];
    let mut bytes = Vec::new();
    let mut limit_exceeded = false;
    let mut drain_incomplete = false;
    let mut stop_observed_at = None;

    loop {
        match pipe.read(&mut buf) {
            Ok(0) => break,
            Ok(read) => {
                let remaining = MAX_CAPTURED_OUTPUT_BYTES.saturating_sub(bytes.len());
                let retained = remaining.min(read);
                bytes.extend_from_slice(&buf[..retained]);
                if retained < read {
                    limit_exceeded = true;
                    limit_exceeded_signal.store(true, std::sync::atomic::Ordering::Release);
                }
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    let observed_at = stop_observed_at.get_or_insert_with(Instant::now);
                    if observed_at.elapsed() >= stop_drain_grace {
                        drain_incomplete = true;
                        break;
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    let observed_at = stop_observed_at.get_or_insert_with(Instant::now);
                    if observed_at.elapsed() >= stop_drain_grace {
                        drain_incomplete = true;
                        break;
                    }
                }
                thread::sleep(Duration::from_millis(1));
            }
            Err(error) => return Err(error),
        }
    }

    Ok(PipeCapture {
        bytes,
        limit_exceeded,
        drain_incomplete,
    })
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn recv_pipe_output(
    rx: std::sync::mpsc::Receiver<std::io::Result<PipeCapture>>,
) -> FwResult<PipeCapture> {
    rx.recv()
        .map_err(|_| FwError::Io(std::io::Error::other("subprocess pipe reader terminated")))?
        .map_err(FwError::Io)
}

fn validate_captured_command_output(
    rendered: &str,
    status: std::process::ExitStatus,
    stdout: PipeCapture,
    stderr: PipeCapture,
    sensitive_values: &[String],
) -> FwResult<Output> {
    if stdout.drain_incomplete || stderr.drain_incomplete {
        return Err(FwError::ContractViolation(
            "subprocess output capture did not drain completely after the child terminated"
                .to_owned(),
        ));
    }
    if stdout.limit_exceeded || stderr.limit_exceeded {
        let stream = match (stdout.limit_exceeded, stderr.limit_exceeded) {
            (true, true) => "stdout and stderr",
            (true, false) => "stdout",
            (false, true) => "stderr",
            (false, false) => unreachable!("validated output has no exceeded stream"),
        };
        return Err(capture_limit_error(stream));
    }
    validate_command_output(
        rendered,
        Output {
            status,
            stdout: stdout.bytes,
            stderr: stderr.bytes,
        },
        sensitive_values,
    )
}

fn capture_limit_error(stream: &str) -> FwError {
    FwError::ContractViolation(capture_limit_message(stream))
}

fn capture_limit_message(stream: &str) -> String {
    format!("subprocess {stream} exceeded the {MAX_CAPTURED_OUTPUT_BYTES}-byte capture limit")
}

pub(crate) fn is_stdout_capture_limit_error(error: &FwError) -> bool {
    let FwError::ContractViolation(message) = error else {
        return false;
    };
    message == &capture_limit_message("stdout")
        || message == &capture_limit_message("stdout and stderr")
}

// ---------------------------------------------------------------------------
// bd-rt-ffmpeg-pipe-7dbu: incremental-stdout subprocess plumbing
//
// Everything above buffers a child's output to completion; the live listen
// path needs to READ WHILE THE CHILD RUNS (unbounded ffmpeg device capture).
// StreamingChild hands the caller the live stdout pipe, drains stderr on a
// dedicated thread into a bounded tail (the pipe-deadlock discipline), and
// guarantees the child is killed and reaped on drop.
// ---------------------------------------------------------------------------

/// Public stderr-tail bound for error reporting (last bytes win). The drainer
/// retains up to one maximum-secret-length of private overlap so redaction can
/// recognize a value that crosses this nominal boundary; callers still receive
/// at most this many raw-context bytes before replacement markers are rendered.
const STREAMING_STDERR_TAIL_BYTES: usize = 4096;

fn streaming_stderr_tail_capacity(sensitive_values: &[String]) -> usize {
    let overlap = sensitive_values
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(0)
        .saturating_sub(1);
    STREAMING_STDERR_TAIL_BYTES.saturating_add(overlap)
}

fn append_streaming_stderr_tail(tail: &mut Vec<u8>, chunk: &[u8], capacity: usize) {
    tail.extend_from_slice(chunk);
    if tail.len() > capacity {
        let excess = tail.len() - capacity;
        tail.drain(..excess);
    }
}

#[cfg(not(windows))]
fn drain_streaming_stderr<R: Read>(
    mut pipe: R,
    tail: &std::sync::Mutex<Vec<u8>>,
    capacity: usize,
) {
    let mut buf = [0u8; 1024];
    loop {
        match pipe.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let mut tail = tail
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                append_streaming_stderr_tail(&mut tail, &buf[..n], capacity);
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                tracing::debug!(%error, "streaming stderr drain stopped after read failure");
                break;
            }
        }
    }
}

fn sanitized_streaming_stderr_tail(tail: &[u8], sensitive_values: &[String]) -> String {
    let mask = sensitive_byte_mask(tail, sensitive_values);
    let start = tail.len().saturating_sub(STREAMING_STDERR_TAIL_BYTES);
    let rendered = render_redacted_bytes(tail, &mask, start, redaction_marker(sensitive_values));
    let rendered = redact_sensitive_text(&rendered, sensitive_values);
    bounded_text_tail(&rendered, STREAMING_STDERR_TAIL_BYTES)
}

fn bounded_text_tail(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_owned();
    }

    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_owned()
}

/// A spawned child whose stdout is consumed incrementally by the caller.
///
/// stderr is drained continuously (a child that logs megabytes can never
/// wedge us) with only the tail retained; [`StreamingChild::kill`] terminates
/// and reaps the complete owned process group (idempotent), and dropping the
/// handle does the same — no zombies or pipe-holding descendants, matching
/// `run_command_cancellable`'s ownership semantics.
#[cfg(not(windows))]
impl std::fmt::Debug for StreamingChild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingChild")
            .field("command", &self.rendered_command)
            .field("reaped", &self.reaped)
            .finish_non_exhaustive()
    }
}

#[cfg(not(windows))]
pub struct StreamingChild {
    child: ManagedChild,
    stdout: Option<std::process::ChildStdout>,
    stderr_tail: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    stderr_join: Option<thread::JoinHandle<()>>,
    rendered_command: String,
    sensitive_values: Vec<String>,
    owns_process_group: bool,
    reaped: bool,
}

#[cfg(windows)]
pub struct StreamingChild {
    _never_constructed: std::convert::Infallible,
}

#[cfg(not(windows))]
impl StreamingChild {
    /// Take the live stdout pipe (once). The caller owns read pacing;
    /// blocking reads unblock promptly after [`Self::kill`] closes the pipe.
    pub fn take_stdout(&mut self) -> Option<std::process::ChildStdout> {
        self.stdout.take()
    }

    /// The redacted command line (for logs / error context).
    #[must_use]
    pub fn rendered_command(&self) -> &str {
        &self.rendered_command
    }

    /// Non-blocking exit probe.
    pub fn try_wait(&mut self) -> FwResult<Option<std::process::ExitStatus>> {
        self.child.try_wait().map_err(FwError::Io)
    }

    /// Terminate and reap the complete owned process group. Idempotent;
    /// returns the root exit status when cleanup ran on this call.
    pub fn kill(&mut self) -> FwResult<Option<std::process::ExitStatus>> {
        if self.reaped {
            return Ok(None);
        }
        if let Err(error) =
            terminate_descendant_process_tree(&mut self.child, self.owns_process_group)
        {
            // A descendant may still own stderr. Detach instead of converting
            // the cleanup failure into an unbounded join; Drop will retry the
            // process-group cleanup once more.
            let _ = self.stderr_join.take();
            return Err(error);
        }
        let status = self.child.try_wait().map_err(FwError::Io)?.ok_or_else(|| {
            process_tree_cleanup_error(
                "the streaming root was not reaped after process-tree termination",
            )
        })?;
        self.reaped = true;
        if let Some(join) = self.stderr_join.take() {
            join.join().map_err(|_| {
                process_tree_cleanup_error("the streaming stderr drainer panicked during cleanup")
            })?;
        }
        tracing::debug!(command = %self.rendered_command, ?status, "streaming process tree terminated and reaped");
        Ok(Some(status))
    }

    /// Finish a producer after its stdout reaches EOF. Natural nonzero exits
    /// become `FW-CMD-FAILED`; a root that closes stdout but does not exit
    /// within `max_wait` is terminated as a complete tree and becomes
    /// `FW-CMD-TIMEOUT`. In every branch, cleanup failure is preserved rather
    /// than being replaced by a successful end-of-stream result.
    pub(crate) fn finish_after_stdout_eof(&mut self, max_wait: Duration) -> FwResult<()> {
        let deadline = Instant::now() + max_wait;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    let cleanup = self.kill().map(drop);
                    let stderr = self.stderr_tail();
                    let outcome = if status.success() {
                        Ok(())
                    } else {
                        Err(FwError::from_command_failure(
                            self.rendered_command.clone(),
                            status.code().unwrap_or(-1),
                            stderr,
                        ))
                    };
                    return match outcome {
                        Ok(()) => cleanup,
                        Err(primary) => Err(merge_process_tree_cleanup_result(primary, cleanup)),
                    };
                }
                Ok(None) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(None) => {
                    let cleanup = self.kill().map(drop);
                    let stderr = self.stderr_tail();
                    return Err(merge_process_tree_cleanup_result(
                        FwError::from_command_timeout(
                            self.rendered_command.clone(),
                            saturating_duration_ms(max_wait),
                            stderr,
                        ),
                        cleanup,
                    ));
                }
                Err(error) => {
                    let cleanup = self.kill().map(drop);
                    return Err(merge_process_tree_cleanup_result(
                        FwError::Io(error),
                        cleanup,
                    ));
                }
            }
        }
    }

    /// The retained, argv-secret-redacted stderr tail (lossy UTF-8), for
    /// FW-CMD-FAILED messages.
    #[must_use]
    pub fn stderr_tail(&self) -> String {
        let tail = self
            .stderr_tail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        sanitized_streaming_stderr_tail(&tail, &self.sensitive_values)
    }
}

#[cfg(not(windows))]
impl Drop for StreamingChild {
    fn drop(&mut self) {
        if let Err(error) = self.kill() {
            tracing::warn!(command = %self.rendered_command, %error, "streaming process-tree cleanup failed during drop");
        }
    }
}

/// Spawn `program args...` with a live stdout pipe and continuously-drained
/// bounded stderr. See [`StreamingChild`].
///
/// Windows: not yet supported — the Job-Object child wrapper does not expose
/// pipe handles the way `std::process::Child` does; live ffmpeg capture on
/// Windows lands with that plumbing (the cpal WASAPI source is the primary
/// Windows path regardless).
#[cfg(windows)]
pub fn spawn_streaming_stdout(_program: &str, _args: &[String]) -> FwResult<StreamingChild> {
    Err(FwError::Unsupported(
        "streaming subprocess capture is not yet supported on Windows; use the cpal capture backend"
            .to_owned(),
    ))
}

#[cfg(not(windows))]
pub fn spawn_streaming_stdout(program: &str, args: &[String]) -> FwResult<StreamingChild> {
    let (rendered_command, sensitive_values) = command_error_diagnostics(program, args);
    let stderr_tail_capacity = streaming_stderr_tail_capacity(&sensitive_values);
    tracing::debug!(command = %rendered_command, "spawning streaming-stdout child");
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if bounded_process_tree_unsupported() {
        return Err(FwError::Unsupported(
            "bounded streaming subprocess trees are unsupported on this platform".to_owned(),
        ));
    }
    let owns_process_group = configure_descendant_process_tree(&mut command);
    if !owns_process_group {
        return Err(FwError::Unsupported(
            "streaming subprocess capture requires a fresh caller-owned process group".to_owned(),
        ));
    }
    let mut child = spawn_managed_child(command).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            FwError::CommandMissing {
                command: program.to_owned(),
            }
        } else {
            FwError::Io(error)
        }
    })?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stderr_tail = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let stderr_join = stderr.map(|pipe| {
        let tail = std::sync::Arc::clone(&stderr_tail);
        thread::spawn(move || drain_streaming_stderr(pipe, &tail, stderr_tail_capacity))
    });
    Ok(StreamingChild {
        child,
        stdout,
        stderr_tail,
        stderr_join,
        rendered_command,
        sensitive_values,
        owns_process_group,
        reaped: false,
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::orchestrator::CancellationToken;

    #[cfg(unix)]
    use super::run_command_cancellable_with_input_probe_and_observer;
    use super::{
        cancellable_poll_delay, command_error_diagnostics_with_environment, render_command_for_log,
        run_command_cancellable, run_command_cancellable_with_input_and_probe,
        run_command_cancellable_with_probe, sensitive_arg_values,
    };

    struct PlatformCommand {
        program: &'static str,
        args: Vec<String>,
    }

    #[cfg(unix)]
    fn platform_command(unix_script: &str, _windows_script: &str) -> PlatformCommand {
        PlatformCommand {
            program: "sh",
            args: vec!["-c".to_owned(), unix_script.to_owned()],
        }
    }

    #[cfg(windows)]
    fn platform_command(_unix_script: &str, windows_script: &str) -> PlatformCommand {
        PlatformCommand {
            program: "powershell.exe",
            args: vec![
                "-NoLogo".to_owned(),
                "-NoProfile".to_owned(),
                "-NonInteractive".to_owned(),
                "-Command".to_owned(),
                windows_script.to_owned(),
            ],
        }
    }

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
        let command = platform_command("exit 0", "exit 0");
        let result = run_command_cancellable(
            command.program,
            &command.args,
            None,
            &cancel,
            Some(Duration::from_secs(10)),
        );
        assert!(result.is_ok(), "success fixture should succeed: {result:?}");
    }

    #[test]
    fn cancellable_stdin_payload_reaches_the_child_exactly() {
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_secs(60));
        let payload = b"strict worker request\n";
        let command = platform_command(
            "cat",
            "[Console]::OpenStandardInput().CopyTo([Console]::OpenStandardOutput())",
        );
        let output = run_command_cancellable_with_input_and_probe(
            command.program,
            &command.args,
            None,
            &cancel,
            Some(Duration::from_secs(10)),
            None,
            payload,
        )
        .expect("stdin fixture must receive the bounded payload");
        assert_eq!(output.stdout, payload);
    }

    #[cfg(unix)]
    #[test]
    fn cancellable_process_observer_receives_the_stable_group_root() {
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_secs(60));
        let mut observed = Vec::new();
        let mut observer = |root_pid| {
            observed.push(root_pid);
            Ok(())
        };
        let output = run_command_cancellable_with_input_probe_and_observer(
            "sh",
            &["-c".to_owned(), "sleep 0.15".to_owned()],
            &cancel,
            Some(Duration::from_secs(10)),
            None,
            &[],
            &mut observer,
        )
        .expect("observed command must complete");
        assert!(output.status.success());
        assert!(!observed.is_empty());
        assert!(observed[0] > 0);
        assert!(observed.iter().all(|pid| *pid == observed[0]));
    }

    #[cfg(unix)]
    #[test]
    fn process_observer_failure_terminates_the_complete_group() {
        let directory = tempfile::tempdir().expect("temporary pid directory");
        let pid_path = directory.path().join("observer-descendant.pid");
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_secs(60));
        let mut polls = 0usize;
        let mut observer = |_root_pid| {
            polls = polls.saturating_add(1);
            if polls >= 2 && pid_path.is_file() {
                Err(crate::error::FwError::ContractViolation(
                    "observer fixture failure".to_owned(),
                ))
            } else {
                Ok(())
            }
        };
        let result = run_command_cancellable_with_input_probe_and_observer(
            "sh",
            &[
                "-c".to_owned(),
                "sleep 60 & child=$!; printf '%s' \"$child\" > \"$1\"; wait".to_owned(),
                "fw-process-observer-test".to_owned(),
                pid_path.to_string_lossy().into_owned(),
            ],
            &cancel,
            Some(Duration::from_secs(10)),
            None,
            &[],
            &mut observer,
        );
        assert!(
            matches!(result, Err(crate::error::FwError::ContractViolation(_))),
            "observer failure must escape after reaping the process tree: {result:?}"
        );

        let descendant_pid: i32 = std::fs::read_to_string(&pid_path)
            .expect("descendant pid fixture")
            .parse()
            .expect("numeric descendant pid");
        let descendant_pid =
            rustix::process::Pid::from_raw(descendant_pid).expect("positive descendant process id");
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while rustix::process::test_kill_process(descendant_pid).is_ok()
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            rustix::process::test_kill_process(descendant_pid).is_err(),
            "observer failure left a descendant process alive"
        );
    }

    #[test]
    fn cancellable_kills_on_expired_deadline() {
        // Create a token whose deadline is already in the past.
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_millis(0));
        // Tiny sleep to ensure we're past the deadline.
        std::thread::sleep(Duration::from_millis(10));

        let command = platform_command("sleep 60", "Start-Sleep -Seconds 60");
        let result = run_command_cancellable(
            command.program,
            &command.args,
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
        let command = platform_command("sleep 60", "Start-Sleep -Seconds 60");
        let result = run_command_cancellable_with_probe(
            command.program,
            &command.args,
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
        let command = platform_command("sleep 60", "Start-Sleep -Seconds 60");
        let result = run_command_cancellable(
            command.program,
            &command.args,
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
        let command = platform_command("exit 0", "exit 0");
        let result = run_command_cancellable(command.program, &command.args, None, &cancel, None);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // run_command / run_command_with_timeout / validate_command_output tests
    // -----------------------------------------------------------------------

    use super::{run_command, run_command_with_timeout, saturating_duration_ms};

    #[test]
    fn run_command_succeeds_for_platform_fixture() {
        let command = platform_command("exit 0", "exit 0");
        let output = run_command(command.program, &command.args, None)
            .expect("platform success fixture should succeed");
        assert!(output.status.success());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn recv_pipe_output_reports_disconnect() {
        use super::{PipeCapture, recv_pipe_output};
        let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<PipeCapture>>();
        drop(tx);
        assert!(recv_pipe_output(rx).is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn pipe_reader_finish_joins_when_writer_stays_open() {
        use std::io::Write;

        let (reader, mut writer) = std::os::unix::net::UnixStream::pair().expect("socket pair");
        writer.write_all(b"complete output").expect("write fixture");
        let started_at = std::time::Instant::now();
        let capture = super::spawn_pipe_reader(reader)
            .finish()
            .expect("finish reader with open writer");
        assert_eq!(capture.bytes, b"complete output");
        assert!(!capture.limit_exceeded);
        assert!(capture.drain_incomplete);
        assert!(started_at.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn regular_file_capture_finishes_while_inherited_writer_stays_open() {
        use std::io::Write as _;

        let reader = tempfile::tempfile().expect("temporary capture file");
        let mut inherited_writer = reader.try_clone().expect("inherited writer fixture");
        inherited_writer
            .write_all(b"complete output")
            .expect("write fixture");
        inherited_writer.flush().expect("flush fixture");
        let started_at = std::time::Instant::now();
        let capture = super::read_bounded_output_file(reader)
            .expect("regular-file capture with open inherited writer");
        assert_eq!(capture.bytes, b"complete output");
        assert!(!capture.limit_exceeded);
        assert!(!capture.drain_incomplete);
        assert!(started_at.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn regular_file_capture_limit_is_detected_without_waiting_for_writer_close() {
        let file = tempfile::tempfile().expect("temporary capture file");
        file.set_len(super::MAX_CAPTURED_OUTPUT_BYTES as u64 + 1)
            .expect("extend capture fixture");
        assert!(
            super::regular_file_exceeds_capture_limit(&file).expect("inspect regular-file capture")
        );
        let capture = super::read_bounded_output_file(file).expect("bounded regular-file capture");
        assert!(capture.bytes.is_empty());
        assert!(capture.limit_exceeded);
        assert!(!capture.drain_incomplete);
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn stopped_reader_reports_incomplete_continuous_drain() {
        let stop = std::sync::atomic::AtomicBool::new(true);
        let limit_exceeded = std::sync::atomic::AtomicBool::new(false);
        let capture = super::read_pipe_with_limit_until_stopped_with_grace(
            std::io::repeat(b'x'),
            &stop,
            &limit_exceeded,
            Duration::ZERO,
        )
        .expect("bounded continuous drain");
        assert!(capture.drain_incomplete);
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn nonblocking_reader_publishes_the_live_capture_limit() {
        let stop = std::sync::atomic::AtomicBool::new(false);
        let limit_exceeded = std::sync::atomic::AtomicBool::new(false);
        let input = vec![b'x'; super::MAX_CAPTURED_OUTPUT_BYTES + 1];
        let capture = super::read_pipe_with_limit_until_stopped_with_grace(
            std::io::Cursor::new(input),
            &stop,
            &limit_exceeded,
            Duration::from_secs(1),
        )
        .expect("bounded nonblocking-reader fixture");
        assert!(capture.limit_exceeded);
        assert!(limit_exceeded.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn pipe_capture_drains_and_reports_output_over_limit() {
        use super::{MAX_CAPTURED_OUTPUT_BYTES, read_pipe_with_limit};
        let input = vec![b'x'; MAX_CAPTURED_OUTPUT_BYTES + 8_193];
        let capture = read_pipe_with_limit(std::io::Cursor::new(input)).expect("capture pipe");
        assert_eq!(capture.bytes.len(), MAX_CAPTURED_OUTPUT_BYTES);
        assert!(capture.limit_exceeded);
        assert!(!capture.drain_incomplete);
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
        let command = platform_command("exit 7", "exit 7");
        let err = run_command(command.program, &command.args, None)
            .expect_err("nonzero fixture should fail");
        let text = err.to_string();
        assert!(
            text.contains("command failed") || text.contains("status"),
            "expected command failure message, got: {text}"
        );
    }

    #[test]
    fn run_command_with_timeout_succeeds_when_fast() {
        let command = platform_command("exit 0", "exit 0");
        let output = run_command_with_timeout(
            command.program,
            &command.args,
            None,
            Some(Duration::from_secs(5)),
        )
        .expect("success fixture should complete within timeout");
        assert!(output.status.success());
    }

    #[test]
    fn run_command_with_timeout_kills_slow_command() {
        let command = platform_command("sleep 60", "Start-Sleep -Seconds 60");
        let err = run_command_with_timeout(
            command.program,
            &command.args,
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

    #[cfg(unix)]
    #[test]
    fn run_command_stops_a_live_output_flood_at_the_capture_limit() {
        let started_at = std::time::Instant::now();
        let error = run_command_with_timeout("yes", &[], None, Some(Duration::from_secs(10)))
            .expect_err("unbounded stdout must trip the live capture limit");
        assert!(super::is_stdout_capture_limit_error(&error));
        assert!(
            started_at.elapsed() < Duration::from_secs(5),
            "capture limiting must terminate output floods before the hard timeout"
        );
    }

    #[cfg(unix)]
    #[test]
    fn run_command_without_timeout_rejects_finite_output_over_capture_limit() {
        let args = vec![
            "-c".to_owned(),
            format!("head -c {} /dev/zero", super::MAX_CAPTURED_OUTPUT_BYTES + 1),
        ];
        let error = run_command("sh", &args, None)
            .expect_err("timeout-free capture must enforce the output bound");
        assert!(super::is_stdout_capture_limit_error(&error));
    }

    #[cfg(unix)]
    #[test]
    fn timeout_terminates_the_complete_descendant_process_group() {
        let directory = tempfile::tempdir().expect("temporary pid directory");
        let pid_path = directory.path().join("descendant.pid");
        let args = vec![
            "-c".to_owned(),
            "sleep 60 & child=$!; printf '%s' \"$child\" > \"$1\"; wait".to_owned(),
            "fw-process-tree-test".to_owned(),
            pid_path.to_string_lossy().into_owned(),
        ];
        let result = run_command_with_timeout("sh", &args, None, Some(Duration::from_millis(250)));
        assert!(result.is_err(), "forking fixture must hit the timeout");

        let descendant_pid: i32 = std::fs::read_to_string(&pid_path)
            .expect("descendant pid fixture")
            .parse()
            .expect("numeric descendant pid");
        let descendant_pid =
            rustix::process::Pid::from_raw(descendant_pid).expect("positive descendant process id");
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while rustix::process::test_kill_process(descendant_pid).is_ok()
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            rustix::process::test_kill_process(descendant_pid).is_err(),
            "timeout left a descendant process alive"
        );
    }

    #[cfg(windows)]
    fn windows_descendant_fixture(pid_path: &std::path::Path, root_tail: &str) -> Vec<String> {
        let escaped_pid_path = pid_path.to_string_lossy().replace('\'', "''");
        let script = format!(
            "$child = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-NonInteractive', '-Command', 'Start-Sleep -Seconds 60') -PassThru; $identity = '{{0}},{{1}}' -f $child.Id,$child.StartTime.ToFileTimeUtc(); Set-Content -LiteralPath '{escaped_pid_path}' -NoNewline -Value $identity; {root_tail}"
        );
        vec![
            "-NoProfile".to_owned(),
            "-NonInteractive".to_owned(),
            "-Command".to_owned(),
            script,
        ]
    }

    #[cfg(windows)]
    fn windows_descendant_identity(pid_path: &std::path::Path) -> (u32, i64) {
        let identity = std::fs::read_to_string(pid_path).expect("descendant identity fixture");
        let (pid, start_time) = identity
            .split_once(',')
            .expect("pid and start-time identity");
        (
            pid.parse().expect("numeric descendant pid"),
            start_time.parse().expect("numeric descendant start time"),
        )
    }

    #[cfg(windows)]
    fn windows_process_identity_is_alive(pid: u32, start_time: i64) -> bool {
        let script = format!(
            "$process = Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($process -and $process.StartTime.ToFileTimeUtc() -eq {start_time}) {{ exit 0 }} else {{ exit 1 }}"
        );
        std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", script.as_str()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[cfg(windows)]
    fn assert_windows_descendant_reaped(pid_path: &std::path::Path) {
        let (pid, start_time) = windows_descendant_identity(pid_path);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while windows_process_identity_is_alive(pid, start_time)
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !windows_process_identity_is_alive(pid, start_time),
            "Windows Job Object left the exact descendant process alive"
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_process_identity_probe_rejects_a_reused_pid() {
        let pid = std::process::id();
        let query =
            format!("[Console]::Out.Write((Get-Process -Id {pid}).StartTime.ToFileTimeUtc())");
        let output = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", query.as_str()])
            .output()
            .expect("query current Windows process identity");
        assert!(output.status.success());
        let start_time: i64 = String::from_utf8(output.stdout)
            .expect("PowerShell identity is UTF-8")
            .parse()
            .expect("PowerShell identity is a FileTime integer");
        assert!(
            windows_process_identity_is_alive(pid, start_time),
            "the exact live process identity must match"
        );
        assert!(
            !windows_process_identity_is_alive(pid, start_time.saturating_sub(1)),
            "an existing PID with a different creation time must not match"
        );
    }

    #[cfg(windows)]
    #[test]
    fn successful_windows_root_cannot_leave_a_descendant_process_alive() {
        let directory = tempfile::tempdir().expect("temporary pid directory");
        let pid_path = directory.path().join("windows-success-descendant.pid");
        let args = windows_descendant_fixture(&pid_path, "exit 0");
        let output =
            run_command_with_timeout("powershell.exe", &args, None, Some(Duration::from_secs(20)))
                .expect("successful root must retain Job Object cleanup authority");
        assert!(output.status.success());
        assert_windows_descendant_reaped(&pid_path);
    }

    #[cfg(windows)]
    #[test]
    fn failed_windows_job_assignment_cleanup_fails_closed() {
        let error = super::terminate_failed_windows_job_assignment(u32::MAX)
            .expect_err("an impossible process id must not receive cleanup authority");
        assert!(
            error.to_string().contains("taskkill"),
            "cleanup failure must remain explicit: {error}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_timeout_terminates_the_complete_descendant_process_tree() {
        let directory = tempfile::tempdir().expect("temporary pid directory");
        let pid_path = directory.path().join("windows-timeout-descendant.pid");
        let args = windows_descendant_fixture(&pid_path, "Wait-Process -Id $child.Id");
        let error =
            run_command_with_timeout("powershell.exe", &args, None, Some(Duration::from_secs(5)))
                .expect_err("long-running Windows fixture must time out");
        assert!(error.to_string().contains("timed out"));
        assert_windows_descendant_reaped(&pid_path);
    }

    #[cfg(windows)]
    #[test]
    fn windows_output_cap_terminates_the_complete_descendant_process_tree() {
        let directory = tempfile::tempdir().expect("temporary pid directory");
        let pid_path = directory.path().join("windows-output-cap-descendant.pid");
        let args = windows_descendant_fixture(
            &pid_path,
            "$chunk = 'x' * 65536; while ($true) { [Console]::Out.Write($chunk) }",
        );
        let error =
            run_command_with_timeout("powershell.exe", &args, None, Some(Duration::from_secs(20)))
                .expect_err("unbounded Windows stdout must trip the capture cap");
        assert!(super::is_stdout_capture_limit_error(&error));
        assert_windows_descendant_reaped(&pid_path);
    }

    #[cfg(windows)]
    #[test]
    fn cancellation_terminates_the_complete_windows_descendant_process_tree() {
        let directory = tempfile::tempdir().expect("temporary pid directory");
        let pid_path = directory.path().join("windows-descendant.pid");
        let args = windows_descendant_fixture(&pid_path, "Wait-Process -Id $child.Id");
        let cancel = CancellationToken::with_deadline_from_now(Duration::from_secs(30));
        let descendant_started = || pid_path.is_file();
        let result = run_command_cancellable_with_probe(
            "powershell.exe",
            &args,
            None,
            &cancel,
            Some(Duration::from_secs(20)),
            Some(&descendant_started),
        );
        assert!(
            matches!(result, Err(crate::error::FwError::Cancelled(_))),
            "fixture cancellation must escape after reaping the process tree: {result:?}"
        );
        assert_windows_descendant_reaped(&pid_path);
    }

    #[cfg(unix)]
    #[test]
    fn successful_root_cannot_leave_a_descendant_process_alive() {
        let output = run_command(
            "sh",
            &[
                "-c".to_owned(),
                "sleep 60 </dev/null >/dev/null 2>&1 & child=$!; printf '%s' \"$child\"; exit 0"
                    .to_owned(),
            ],
            None,
        )
        .expect("successful root command");
        let descendant_pid: i32 = String::from_utf8(output.stdout)
            .expect("UTF-8 descendant pid")
            .parse()
            .expect("numeric descendant pid");
        let descendant_pid =
            rustix::process::Pid::from_raw(descendant_pid).expect("positive descendant pid");
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while rustix::process::test_kill_process(descendant_pid).is_ok()
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            rustix::process::test_kill_process(descendant_pid).is_err(),
            "successful timeout-free command left a descendant process alive"
        );
    }

    #[test]
    fn run_command_captures_stderr() {
        let command = platform_command(
            "printf '%s' 'nonexistent_path_xyz_99999' >&2; exit 7",
            "[Console]::Error.Write('nonexistent_path_xyz_99999'); exit 7",
        );
        let err = run_command(command.program, &command.args, None)
            .expect_err("stderr fixture should fail");
        let text = err.to_string();
        assert!(
            text.contains("nonexistent_path_xyz_99999"),
            "expected stderr content, got: {text}"
        );
    }

    #[test]
    fn run_command_with_cwd() {
        // Behavior contract, not path identity: RCH workers expose the same
        // checkout/temp trees under different path spellings (/data vs /Users
        // aliases), so comparing reported cwd strings is environment-sensitive.
        // A child that creates a file relative to its cwd proves the requested
        // directory took effect under every spelling, including root.
        let dir = tempfile::tempdir().expect("tempdir");
        let marker = "fw-cwd-marker.txt";
        let command = platform_command(
            "printf '' > fw-cwd-marker.txt",
            "[IO.File]::WriteAllBytes('fw-cwd-marker.txt', [byte[]]@())",
        );
        run_command(command.program, &command.args, Some(dir.path()))
            .expect("cwd fixture should succeed");
        assert!(
            dir.path().join(marker).is_file(),
            "child must resolve relative paths against the requested cwd"
        );
    }

    #[test]
    fn render_command_for_log_redacts_sensitive_flags() {
        let args = vec![
            "--hf-token".to_owned(),
            "hf_secret_123".to_owned(),
            "--api-key=secret_api_key".to_owned(),
            "--prompt=private patient context".to_owned(),
            "--token-threshold".to_owned(),
            "0.1".to_owned(),
            "positional".to_owned(),
        ];
        let rendered = render_command_for_log("prog", &args);
        assert!(rendered.contains("--hf-token ***"));
        assert!(rendered.contains("--api-key=***"));
        assert!(rendered.contains("--prompt=***"));
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
        assert!(
            !rendered.contains("private patient context"),
            "prompt text should be redacted"
        );
        assert_eq!(
            rendered,
            "prog --hf-token *** --api-key=*** --prompt=*** --token-threshold 0.1 positional"
        );
        assert_eq!(render_command_for_log("prog", &[]), "prog");
    }

    #[test]
    fn sensitive_arg_values_are_longest_first_unique_and_nonempty() {
        let args = vec![
            "--hf-token".to_owned(),
            "token".to_owned(),
            "--api-key=token-suffix".to_owned(),
            "--auth-token=token".to_owned(),
            "--password=".to_owned(),
            "--secret".to_owned(),
        ];

        assert_eq!(
            sensitive_arg_values(&args),
            vec!["token-suffix".to_owned(), "token".to_owned()]
        );
    }

    #[test]
    fn adjacent_sensitive_flags_rearm_redaction_for_their_own_values() {
        let split_secret = "adjacent_secret_123";
        let equal_secret = "equal_secret_456";
        let args = vec![
            "--hf-token".to_owned(),
            "--api-key".to_owned(),
            split_secret.to_owned(),
            "--secret".to_owned(),
            format!("--auth-token={equal_secret}"),
        ];

        let (rendered, sensitive_values) =
            command_error_diagnostics_with_environment("prog", &args, |_| None);
        assert_eq!(
            sensitive_values,
            vec![split_secret.to_owned(), equal_secret.to_owned()]
        );
        assert_eq!(
            rendered,
            "prog --hf-token --api-key *** --secret --auth-token=***"
        );
        assert!(!rendered.contains(split_secret));
        assert!(!rendered.contains(equal_secret));
    }

    #[test]
    fn command_diagnostics_redact_repeated_sensitive_value_everywhere() {
        let secret = "repeated_secret_123";
        let args = vec![
            "--hf-token".to_owned(),
            secret.to_owned(),
            "--label".to_owned(),
            secret.to_owned(),
        ];

        let (rendered, sensitive_values) =
            command_error_diagnostics_with_environment("prog", &args, |_| None);
        assert_eq!(sensitive_values, vec![secret.to_owned()]);
        assert!(
            !rendered.contains(secret),
            "repeated secret leaked: {rendered}"
        );
        assert!(
            rendered.contains("--label ***"),
            "nonsensitive occurrence was not scrubbed: {rendered}"
        );
    }

    #[test]
    fn command_diagnostics_do_not_reemit_asterisk_only_secret() {
        let secret = "***";
        let args = vec!["--secret".to_owned(), secret.to_owned()];

        let (rendered, _) =
            command_error_diagnostics_with_environment("prog", &args, |_| None);
        assert!(
            !rendered.contains(secret),
            "mask re-emitted the secret: {rendered}"
        );
        assert!(
            rendered.contains("--secret ###"),
            "alternate marker was not selected: {rendered}"
        );
    }

    #[test]
    fn command_diagnostics_include_known_inherited_secrets() {
        let secret = "inherited_hf_secret_987";
        let args = vec!["--label".to_owned(), secret.to_owned()];
        let (rendered, sensitive_values) = command_error_diagnostics_with_environment(
            "prog",
            &args,
            |name| (name == "HF_TOKEN").then(|| secret.to_owned()),
        );

        assert!(sensitive_values.iter().any(|value| value == secret));
        assert!(!rendered.contains(secret));
        let stderr = super::captured_stderr_for_error(secret.as_bytes(), &sensitive_values);
        assert!(!stderr.contains(secret));
    }

    #[cfg(unix)]
    #[test]
    fn failing_child_stderr_redacts_inherited_hf_token() {
        const PROBE_ENV: &str = "__FW_INHERITED_SECRET_PROBE";
        const SECRET: &str = "inherited_hf_token_live_654321";
        if std::env::var_os(PROBE_ENV).is_some() {
            let args = vec![
                "-c".to_owned(),
                "printf 'inherited-context:%s\n' \"$HF_TOKEN\" >&2; exit 9".to_owned(),
            ];
            let error = run_command("sh", &args, None).expect_err("fixture must fail");
            let text = error.to_string();
            assert!(text.contains("inherited-context:"));
            assert!(!text.contains(SECRET), "inherited HF token leaked: {text}");
            return;
        }

        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .arg("process::tests::failing_child_stderr_redacts_inherited_hf_token")
            .arg("--exact")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(PROBE_ENV, "1")
            .env("HF_TOKEN", SECRET)
            .output()
            .expect("spawn inherited-secret probe");
        assert!(
            output.status.success(),
            "inherited-secret probe failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn captured_stderr_scrubs_secret_reconstructed_by_lossy_utf8() {
        let secret = "A�B";
        let sensitive_values = vec![secret.to_owned()];

        let rendered = super::captured_stderr_for_error(b"A\xffB", &sensitive_values);
        assert_eq!(rendered, "***");
        assert!(
            !rendered.contains(secret),
            "lossy UTF-8 reconstructed the secret: {rendered}"
        );
    }

    #[test]
    fn captured_stderr_redacts_overlapping_repetitive_matches() {
        let sensitive_values = vec!["aaa".to_owned()];

        let rendered = super::captured_stderr_for_error(b"aaaaab", &sensitive_values);
        assert_eq!(rendered, "***b");
    }

    #[cfg(unix)]
    #[test]
    fn failing_child_stderr_redacts_split_and_equal_argv_secrets() {
        let hf_secret = "hf_secret_echo_123";
        let api_secret = "api_secret_echo_456";
        let args = vec![
            "-c".to_owned(),
            "printf 'benign-context:%s:%s:%s\\n' \"$1\" \"$2\" \"$3\" >&2; exit 9".to_owned(),
            "fw-secret-probe".to_owned(),
            "--hf-token".to_owned(),
            hf_secret.to_owned(),
            format!("--api-key={api_secret}"),
        ];

        let error = run_command("sh", &args, None).expect_err("fixture must fail");
        let text = error.to_string();
        assert!(
            text.contains("benign-context"),
            "benign stderr was lost: {text}"
        );
        assert!(
            text.contains("--hf-token:***:--api-key=***"),
            "sensitive argv values were not replaced in context: {text}"
        );
        assert!(!text.contains(hf_secret), "HF token leaked: {text}");
        assert!(!text.contains(api_secret), "API key leaked: {text}");
    }

    #[cfg(unix)]
    #[test]
    fn failing_child_stderr_redacts_whisper_prompt_text() {
        let prompt = "patient Jane Doe; access phrase blue orchard";
        let args = vec![
            "-c".to_owned(),
            "printf 'whisper-context:%s:%s\n' \"$1\" \"$2\" >&2; exit 9".to_owned(),
            "fw-prompt-probe".to_owned(),
            "--prompt".to_owned(),
            prompt.to_owned(),
        ];

        let error = run_command("sh", &args, None).expect_err("fixture must fail");
        let text = error.to_string();
        assert!(
            text.contains("whisper-context:--prompt:***"),
            "benign prompt context was lost: {text}"
        );
        assert!(!text.contains(prompt), "Whisper prompt leaked: {text}");
    }

    #[cfg(unix)]
    #[test]
    fn bounded_timeout_stderr_redacts_argv_secret() {
        let secret = "timeout_secret_echo_789";
        let args = vec![
            "-c".to_owned(),
            "printf 'timeout-context:%s\\n' \"$2\" >&2; sleep 30".to_owned(),
            "fw-timeout-probe".to_owned(),
            "--hf-token".to_owned(),
            secret.to_owned(),
        ];

        let error = run_command_with_timeout("sh", &args, None, Some(Duration::from_secs(1)))
            .expect_err("fixture must time out");
        let text = error.to_string();
        assert!(
            text.contains("timeout-context:***"),
            "context was lost: {text}"
        );
        assert!(!text.contains(secret), "timeout leaked argv secret: {text}");
    }

    #[cfg(unix)]
    #[test]
    fn bounded_nonzero_stderr_redacts_argv_secret() {
        let secret = "bounded_failure_secret_901";
        let args = vec![
            "-c".to_owned(),
            "printf 'bounded-failure:%s\\n' \"$2\" >&2; exit 7".to_owned(),
            "fw-bounded-failure-probe".to_owned(),
            "--api-key".to_owned(),
            secret.to_owned(),
        ];

        let error = run_command_with_timeout("sh", &args, None, Some(Duration::from_secs(5)))
            .expect_err("fixture must fail");
        let text = error.to_string();
        assert!(text.contains("bounded-failure"), "context was lost: {text}");
        assert!(
            !text.contains(secret),
            "bounded failure leaked secret: {text}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn cancellable_timeout_stderr_redacts_argv_secret() {
        let secret = "cancellable_secret_echo_012";
        let args = vec![
            "-c".to_owned(),
            "printf 'cancellable-context:%s\\n' \"$2\" >&2; sleep 30".to_owned(),
            "fw-cancellable-probe".to_owned(),
            "--auth-token".to_owned(),
            secret.to_owned(),
        ];
        let token = CancellationToken::no_deadline();

        let error =
            run_command_cancellable("sh", &args, None, &token, Some(Duration::from_secs(1)))
                .expect_err("fixture must time out");
        let text = error.to_string();
        assert!(
            text.contains("cancellable-context:***"),
            "context was lost: {text}"
        );
        assert!(
            !text.contains(secret),
            "cancellable timeout leaked argv secret: {text}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn cancellable_nonzero_stderr_redacts_argv_secret() {
        let secret = "cancellable_failure_secret_234";
        let args = vec![
            "-c".to_owned(),
            "printf 'cancellable-failure:%s\\n' \"$2\" >&2; exit 8".to_owned(),
            "fw-cancellable-failure-probe".to_owned(),
            "--auth-token".to_owned(),
            secret.to_owned(),
        ];
        let token = CancellationToken::no_deadline();

        let error =
            run_command_cancellable("sh", &args, None, &token, Some(Duration::from_secs(5)))
                .expect_err("fixture must fail");
        let text = error.to_string();
        assert!(
            text.contains("cancellable-failure"),
            "context was lost: {text}"
        );
        assert!(
            !text.contains(secret),
            "cancellable failure leaked secret: {text}"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn streaming_stderr_tail_redacts_argv_secret() {
        let secret = "streaming_secret_echo_345";
        let args = vec![
            "-c".to_owned(),
            "printf 'stream-context:%s\\n' \"$2\" >&2".to_owned(),
            "fw-stream-probe".to_owned(),
            "--access-token".to_owned(),
            secret.to_owned(),
        ];
        let mut child = super::spawn_streaming_stdout("sh", &args).expect("spawn fixture");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if child.try_wait().expect("probe child").is_some() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "streaming fixture did not exit"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        child.kill().expect("join stderr drainer");

        let tail = child.stderr_tail();
        assert!(
            tail.contains("stream-context:***"),
            "context was lost: {tail}"
        );
        assert!(
            !tail.contains(secret),
            "streaming tail leaked argv secret: {tail}"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn streaming_stderr_drain_retries_interrupted_reads() {
        struct InterruptedOnce {
            interrupted: bool,
            bytes: std::io::Cursor<Vec<u8>>,
        }

        impl std::io::Read for InterruptedOnce {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if !self.interrupted {
                    self.interrupted = true;
                    return Err(std::io::Error::from(std::io::ErrorKind::Interrupted));
                }
                std::io::Read::read(&mut self.bytes, buf)
            }
        }

        let tail = std::sync::Mutex::new(Vec::new());
        super::drain_streaming_stderr(
            InterruptedOnce {
                interrupted: false,
                bytes: std::io::Cursor::new(b"retained after interrupt".to_vec()),
            },
            &tail,
            super::STREAMING_STDERR_TAIL_BYTES,
        );
        let tail = tail
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(tail, b"retained after interrupt");
    }

    #[cfg(unix)]
    #[test]
    fn streaming_child_kill_terminates_descendant_holding_pipes() {
        use std::io::BufRead as _;

        let args = vec![
            "-c".to_owned(),
            "sleep 30 & printf '%s\\n' \"$!\"; wait".to_owned(),
        ];
        let mut child = super::spawn_streaming_stdout("sh", &args).expect("spawn process tree");
        let stdout = child.take_stdout().expect("streaming stdout");
        let mut stdout = std::io::BufReader::new(stdout);
        let mut descendant = String::new();
        stdout
            .read_line(&mut descendant)
            .expect("read descendant pid");
        let descendant = descendant
            .trim()
            .parse::<i32>()
            .ok()
            .and_then(rustix::process::Pid::from_raw)
            .expect("valid descendant pid");
        drop(stdout);

        let started = std::time::Instant::now();
        child.kill().expect("terminate complete process group");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "process-tree cleanup exceeded its bound: {:?}",
            started.elapsed()
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while rustix::process::test_kill_process(descendant).is_ok()
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            rustix::process::test_kill_process(descendant).is_err(),
            "streaming descendant survived root cleanup"
        );
    }

    #[test]
    fn streaming_stderr_tail_redacts_secret_across_nominal_boundary() {
        let secret = "boundary_secret_567";
        let sensitive_values = vec![secret.to_owned()];
        let capacity = super::streaming_stderr_tail_capacity(&sensitive_values);
        let suffix_len = super::STREAMING_STDERR_TAIL_BYTES + 1 - secret.len();
        let mut raw = vec![b'p'; 100];
        raw.extend_from_slice(secret.as_bytes());
        raw.extend(std::iter::repeat_n(b'z', suffix_len));

        let mut retained = Vec::new();
        for chunk in raw.chunks(257) {
            super::append_streaming_stderr_tail(&mut retained, chunk, capacity);
        }
        let tail = super::sanitized_streaming_stderr_tail(&retained, &sensitive_values);

        assert!(tail.len() <= super::STREAMING_STDERR_TAIL_BYTES);
        assert!(
            tail.starts_with('*'),
            "boundary secret prefix was not masked"
        );
        assert!(
            !tail.contains(secret) && !tail.contains(&secret[1..]),
            "boundary-straddling secret leaked: {tail}"
        );
        assert!(tail.ends_with("zzzz"), "benign tail context was lost");
    }

    #[test]
    fn streaming_stderr_tail_remains_bounded_when_markers_expand() {
        let sensitive_values = vec!["*".to_owned(), "#".to_owned()];
        let capacity = super::streaming_stderr_tail_capacity(&sensitive_values);
        let raw: Vec<u8> = (0..super::STREAMING_STDERR_TAIL_BYTES + 100)
            .map(|index| match index % 3 {
                0 => b'*',
                1 => b'#',
                _ => b'a',
            })
            .collect();
        let mut retained = Vec::new();
        for chunk in raw.chunks(257) {
            super::append_streaming_stderr_tail(&mut retained, chunk, capacity);
        }

        let tail = super::sanitized_streaming_stderr_tail(&retained, &sensitive_values);
        assert!(
            tail.len() <= super::STREAMING_STDERR_TAIL_BYTES,
            "expanded marker tail exceeded the public bound: {}",
            tail.len()
        );
        assert!(
            !tail.contains('*') && !tail.contains('#'),
            "one-byte secret leaked: {tail}"
        );
        assert!(tail.contains('a'), "benign tail context was lost");
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
    fn command_exists_true_for_platform_fixture() {
        let command = platform_command("exit 0", "exit 0");
        assert!(
            command_exists(command.program),
            "platform test shell should exist"
        );
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
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;
    #[cfg(windows)]
    use std::os::windows::process::ExitStatusExt;
    use std::process::ExitStatus;

    #[cfg(unix)]
    fn fake_exit_status(code: i32) -> ExitStatus {
        ExitStatus::from_raw(code << 8)
    }

    #[cfg(windows)]
    fn fake_exit_status(code: i32) -> ExitStatus {
        ExitStatus::from_raw(code as u32)
    }

    fn fake_output(code: i32, stderr: &str) -> std::process::Output {
        std::process::Output {
            status: fake_exit_status(code),
            stdout: Vec::new(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn validate_command_output_redacts_overlapping_argv_secrets() {
        let args = vec![
            "--hf-token".to_owned(),
            "fwsecret".to_owned(),
            "--api-key=fwsecret-long".to_owned(),
        ];
        let sensitive_values = sensitive_arg_values(&args);
        let output = fake_output(
            1,
            "benign failure: --api-key=fwsecret-long and --hf-token fwsecret",
        );

        let error = validate_command_output(
            "test-cmd --hf-token *** --api-key=***",
            output,
            &sensitive_values,
        )
        .expect_err("fixture must fail");
        let text = error.to_string();
        assert!(
            text.contains("benign failure"),
            "benign context was lost: {text}"
        );
        assert!(
            text.contains("--api-key=*** and --hf-token ***"),
            "redacted context is incomplete: {text}"
        );
        assert!(!text.contains("fwsecret"), "secret leaked: {text}");
        assert!(
            !text.contains("-long"),
            "longer overlapping secret was only partially redacted: {text}"
        );
    }

    #[test]
    fn validate_command_output_success_returns_ok() {
        let output = fake_output(0, "");
        let result = validate_command_output("test-cmd", output, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_command_output_nonzero_exit_returns_error() {
        let output = fake_output(1, "something went wrong");
        let result = validate_command_output("test-cmd", output, &[]);
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
        let err = validate_command_output("my-tool --flag", output, &[]).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("42"),
            "error should mention exit code 42, got: {text}"
        );
    }

    #[test]
    fn validate_command_output_empty_stderr_still_fails_on_nonzero() {
        let output = fake_output(2, "");
        let result = validate_command_output("cmd", output, &[]);
        assert!(
            result.is_err(),
            "non-zero exit with empty stderr should still fail"
        );
    }

    // ── Additional edge case tests ──

    #[test]
    fn run_command_with_timeout_none_behaves_like_run_command() {
        let command = platform_command("exit 0", "exit 0");
        let output = run_command_with_timeout(command.program, &command.args, None, None)
            .expect("success fixture should succeed");
        assert!(output.status.success());
    }

    #[test]
    fn run_command_with_args() {
        let command = platform_command(
            "printf '%s' 'hello world'",
            "[Console]::Out.Write('hello world')",
        );
        let output = run_command(command.program, &command.args, None)
            .expect("stdout fixture should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("hello world"),
            "expected 'hello world', got: {stdout}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn run_command_with_timeout_does_not_inherit_parent_stdin() {
        const PROBE_ENV: &str = "__FW_NULL_STDIN_PROBE";
        if std::env::var_os(PROBE_ENV).is_some() {
            let args = vec![
                "-c".to_owned(),
                "if IFS= read -r value; then printf 'unexpected:%s' \"$value\" >&2; exit 9; fi"
                    .to_owned(),
            ];
            run_command_with_timeout(
                "sh",
                &args,
                None,
                Some(Duration::from_secs(2)),
            )
            .expect("bounded subprocess stdin must be EOF");
            return;
        }

        let mut child = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .arg("process::tests::run_command_with_timeout_does_not_inherit_parent_stdin")
            .arg("--exact")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(PROBE_ENV, "1")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn stdin probe");
        let mut stdin = child.stdin.take().expect("probe stdin");
        std::io::Write::write_all(&mut stdin, b"parent-only-sentinel\n")
            .expect("write parent sentinel");
        drop(stdin);
        let output = child.wait_with_output().expect("wait for stdin probe");
        assert!(
            output.status.success(),
            "stdin probe failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
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
        let command = platform_command(
            "printf '%s' 'test_output'",
            "[Console]::Out.Write('test_output')",
        );
        let output = run_command_cancellable(command.program, &command.args, None, &cancel, None)
            .expect("stdout fixture should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("test_output"),
            "should capture stdout: {stdout}"
        );
    }

    #[test]
    fn cancellable_nonzero_exit_returns_error() {
        let cancel = CancellationToken::no_deadline();
        let command = platform_command("exit 7", "exit 7");
        let err = run_command_cancellable(command.program, &command.args, None, &cancel, None)
            .expect_err("nonzero fixture should fail");
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
        let err = validate_command_output("my-special-cmd --flag", output, &[]).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("my-special-cmd"),
            "error should mention command: {text}"
        );
    }

    #[test]
    fn cancellable_with_cwd_succeeds() {
        // Same behavior contract as run_command_with_cwd: prove the requested
        // cwd took effect via a relative-path side effect instead of comparing
        // reported path strings, which differ across RCH /data vs /Users
        // aliases.
        let dir = tempfile::tempdir().expect("tempdir");
        let marker = "fw-cwd-cancellable-marker.txt";
        let cancel = CancellationToken::no_deadline();
        let command = platform_command(
            "printf '' > fw-cwd-cancellable-marker.txt",
            "[IO.File]::WriteAllBytes('fw-cwd-cancellable-marker.txt', [byte[]]@())",
        );
        run_command_cancellable(
            command.program,
            &command.args,
            Some(dir.path()),
            &cancel,
            None,
        )
        .expect("cancellable cwd fixture should succeed");
        assert!(
            dir.path().join(marker).is_file(),
            "cancelled-capable child must resolve relative paths against the requested cwd"
        );
    }

    #[test]
    fn run_command_success_fixture_succeeds() {
        let command = platform_command("exit 0", "exit 0");
        let output = run_command(command.program, &command.args, None)
            .expect("platform success fixture should succeed");
        assert!(output.status.success());
    }

    #[test]
    fn cancellable_with_hard_timeout_none_and_no_deadline() {
        // Both safety nets disabled — should still work for fast commands.
        let cancel = CancellationToken::no_deadline();
        let command = platform_command("printf '%s' ok", "[Console]::Out.Write('ok')");
        let output = run_command_cancellable(command.program, &command.args, None, &cancel, None)
            .expect("should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("ok"));
    }

    #[test]
    fn run_command_preserves_large_stdout_payload() {
        let command = platform_command(
            "yes x | head -c 200000",
            "[Console]::Out.Write('x' * 200000)",
        );
        let output = run_command(command.program, &command.args, None)
            .expect("large stdout command should succeed");
        assert_eq!(
            output.stdout.len(),
            200_000,
            "stdout should be fully captured after process exit"
        );
    }

    #[test]
    fn run_command_preserves_large_stderr_payload_on_failure() {
        let command = platform_command(
            "yes e | head -c 200000 >&2; exit 7",
            "[Console]::Error.Write('e' * 200000); exit 7",
        );
        let err = run_command(command.program, &command.args, None)
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

    #[cfg(unix)]
    #[test]
    fn validate_command_output_signal_terminated_uses_negative_one() {
        // When a process is killed by a signal, exit code may not be available.
        // On Unix, from_raw(9) represents SIGKILL (signal 9, no exit code).
        let output = std::process::Output {
            status: ExitStatus::from_raw(9), // signal 9 (SIGKILL), no exit code
            stdout: Vec::new(),
            stderr: b"killed".to_vec(),
        };
        let result = validate_command_output("signaled-cmd", output, &[]);
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
