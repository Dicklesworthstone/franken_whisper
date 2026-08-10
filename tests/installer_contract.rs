#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;
use std::process::Command;

fn write_fake_fw(root: &Path, pull_status: i32, doctor_ready: bool) -> std::path::PathBuf {
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).expect("create fake bin directory");
    let binary = bin_dir.join("fw");
    let script = format!(
        r#"#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FW_INSTALLER_CALL_LOG"
printf '%s\n' "$HOME" >> "$FW_INSTALLER_HOME_LOG"
if [[ "$1 $2" == "pull all" ]]; then
    exit {pull_status}
fi
if [[ "$1 $2" == "doctor --json" ]]; then
    printf '%s\n' '{{"ready":{doctor_ready}}}'
    exit 0
fi
exit 64
"#
    );
    fs::write(&binary, script).expect("write fake fw");
    let mut permissions = fs::metadata(&binary)
        .expect("fake fw metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&binary, permissions).expect("make fake fw executable");
    bin_dir
}

fn source_and_run(root: &Path, body: &str) -> std::process::Output {
    let install_script = Path::new(env!("CARGO_MANIFEST_DIR")).join("install.sh");
    Command::new("bash")
        .arg("-c")
        .arg(format!(
            "source \"{}\" --no-pull\n{}",
            install_script.display(),
            body
        ))
        .env("HOME", root)
        .env("FW_INSTALLER_CALL_LOG", root.join("calls.log"))
        .env("FW_INSTALLER_HOME_LOG", root.join("homes.log"))
        .env("FW_INSTALL_PROMPT_TIMEOUT", "1")
        .output()
        .expect("run installer function harness")
}

#[test]
fn installer_script_is_valid_bash() {
    let install_script = Path::new(env!("CARGO_MANIFEST_DIR")).join("install.sh");
    let status = Command::new("bash")
        .arg("-n")
        .arg(install_script)
        .status()
        .expect("run bash syntax check");
    assert!(status.success());
}

#[test]
fn installer_accepts_the_exact_dsr_release_archive_members() {
    let root = tempfile::tempdir().expect("temporary archive harness");
    let stage = root.path().join("stage");
    fs::create_dir(&stage).expect("create archive stage");
    for member in [
        "franken_whisper",
        "fw",
        "README.md",
        "LICENSE",
        "NOTICE.sortformer.txt",
        "THIRD_PARTY_NOTICES.md",
        "AGENTS.md",
    ] {
        fs::write(stage.join(member), member).expect("write archive member");
    }
    let archive = root
        .path()
        .join("franken_whisper-0.7.0-darwin_arm64.tar.gz");
    let status = Command::new("tar")
        .args(["-czf"])
        .arg(&archive)
        .arg("-C")
        .arg(&stage)
        .args([
            "franken_whisper",
            "fw",
            "README.md",
            "LICENSE",
            "NOTICE.sortformer.txt",
            "THIRD_PARTY_NOTICES.md",
            "AGENTS.md",
        ])
        .status()
        .expect("create release archive");
    assert!(status.success());

    let output = source_and_run(
        root.path(),
        &format!("validate_archive_members \"{}\" tar.gz", archive.display()),
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn release_checksum_resolution_prefers_dsr_sidecar_and_binds_the_archive_name() {
    let root = tempfile::tempdir().expect("temporary checksum harness");
    let good_sha = "a".repeat(64);
    let output = source_and_run(
        root.path(),
        &format!(
            r#"
TMP="{}"
VERSION="v0.7.0"
download_file() {{
    local url="$1" dest="$2"
    case "$url" in
        *.tar.gz.sha256)
            printf '%s  unrelated.tar.gz\n' "{}" > "$dest"
            printf '%s  franken_whisper-0.7.0-darwin_arm64.tar.gz\n' "{}" >> "$dest"
            return 0
            ;;
        *) return 1 ;;
    esac
}}
release_archive_checksum \
    "https://example.invalid/franken_whisper-0.7.0-darwin_arm64.tar.gz" \
    "franken_whisper-0.7.0-darwin_arm64.tar.gz"
"#,
            root.path().display(),
            "b".repeat(64),
            good_sha,
        ),
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        format!("{good_sha}\n")
    );
}

#[test]
fn release_checksum_resolution_accepts_dsr_combined_manifest_fallback() {
    let root = tempfile::tempdir().expect("temporary checksum harness");
    let good_sha = "c".repeat(64);
    let output = source_and_run(
        root.path(),
        &format!(
            r#"
TMP="{}"
VERSION="v0.7.0"
download_file() {{
    local url="$1" dest="$2"
    case "$url" in
        */SHA256SUMS)
            printf '%s  franken_whisper-0.7.0-linux_amd64.tar.gz\n' "{}" > "$dest"
            return 0
            ;;
        *) return 1 ;;
    esac
}}
release_archive_checksum \
    "https://example.invalid/franken_whisper-0.7.0-linux_amd64.tar.gz" \
    "franken_whisper-0.7.0-linux_amd64.tar.gz"
"#,
            root.path().display(),
            good_sha,
        ),
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        format!("{good_sha}\n")
    );
}

#[test]
fn headless_install_provisions_both_models_and_verifies_doctor() {
    let root = tempfile::tempdir().expect("temporary installer harness");
    let bin_dir = write_fake_fw(root.path(), 0, true);
    let output = source_and_run(
        root.path(),
        &format!(
            "DEST=\"{}\"\nNO_PULL=0\nQUIET=1\nOFFLINE_TARBALL=\"\"\nprovision_default_models",
            bin_dir.display()
        ),
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let calls = fs::read_to_string(root.path().join("calls.log")).expect("read call log");
    assert_eq!(calls, "pull all\ndoctor --json\n");
}

#[test]
fn nonquiet_headless_install_takes_the_documented_yes_default() {
    let root = tempfile::tempdir().expect("temporary installer harness");
    let bin_dir = write_fake_fw(root.path(), 0, true);
    let output = source_and_run(
        root.path(),
        &format!(
            "DEST=\"{}\"\nNO_PULL=0\nQUIET=0\nOFFLINE_TARBALL=\"\"\nprovision_default_models",
            bin_dir.display()
        ),
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let calls = fs::read_to_string(root.path().join("calls.log")).expect("read call log");
    assert_eq!(calls, "pull all\ndoctor --json\n");
}

#[test]
fn model_pull_failure_fails_the_install() {
    let root = tempfile::tempdir().expect("temporary installer harness");
    let bin_dir = write_fake_fw(root.path(), 42, false);
    let output = source_and_run(
        root.path(),
        &format!(
            "DEST=\"{}\"\nNO_PULL=0\nQUIET=1\nOFFLINE_TARBALL=\"\"\nprovision_default_models",
            bin_dir.display()
        ),
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Native model provisioning failed"));
}

#[test]
fn post_pull_doctor_failure_fails_closed() {
    let root = tempfile::tempdir().expect("temporary installer harness");
    let bin_dir = write_fake_fw(root.path(), 0, false);
    let output = source_and_run(
        root.path(),
        &format!(
            "DEST=\"{}\"\nNO_PULL=0\nQUIET=1\nOFFLINE_TARBALL=\"\"\nprovision_default_models",
            bin_dir.display()
        ),
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("did not satisfy the compiled trust roots"));
    let calls = fs::read_to_string(root.path().join("calls.log")).expect("read call log");
    assert_eq!(calls, "pull all\ndoctor --json\n");
}

#[test]
fn offline_install_accepts_only_a_preseeded_verified_cache() {
    let ready_root = tempfile::tempdir().expect("temporary ready offline harness");
    let ready_bin = write_fake_fw(ready_root.path(), 0, true);
    let ready = source_and_run(
        ready_root.path(),
        &format!(
            "DEST=\"{}\"\nNO_PULL=0\nQUIET=1\nOFFLINE_TARBALL=\"archive.tar.gz\"\nprovision_default_models",
            ready_bin.display()
        ),
    );
    assert!(ready.status.success());
    let calls =
        fs::read_to_string(ready_root.path().join("calls.log")).expect("read ready call log");
    assert_eq!(calls, "doctor --json\n");

    let missing_root = tempfile::tempdir().expect("temporary missing offline harness");
    let missing_bin = write_fake_fw(missing_root.path(), 0, false);
    let missing = source_and_run(
        missing_root.path(),
        &format!(
            "DEST=\"{}\"\nNO_PULL=0\nQUIET=1\nOFFLINE_TARBALL=\"archive.tar.gz\"\nprovision_default_models",
            missing_bin.display()
        ),
    );
    assert!(!missing.status.success());
    assert!(
        String::from_utf8_lossy(&missing.stderr)
            .contains("requires both model packages to be preseeded")
    );
    let calls =
        fs::read_to_string(missing_root.path().join("calls.log")).expect("read missing call log");
    assert_eq!(calls, "doctor --json\n");
}

#[test]
fn no_pull_is_an_explicit_network_free_escape() {
    let root = tempfile::tempdir().expect("temporary installer harness");
    let bin_dir = write_fake_fw(root.path(), 0, true);
    let output = source_and_run(
        root.path(),
        &format!(
            "DEST=\"{}\"\nNO_PULL=1\nQUIET=1\nOFFLINE_TARBALL=\"\"\nprovision_default_models",
            bin_dir.display()
        ),
    );
    assert!(output.status.success());
    assert!(!root.path().join("calls.log").exists());
}

#[test]
fn sudo_install_provisions_the_invoking_users_cache() {
    let root = tempfile::tempdir().expect("temporary sudo installer harness");
    let bin_dir = write_fake_fw(root.path(), 0, true);
    let sudo_log = root.path().join("sudo.log");
    let output = source_and_run(
        root.path(),
        &format!(
            r#"
id() {{
    if [[ "$1" == "-u" && $# -eq 1 ]]; then printf '0\n'; return 0; fi
    if [[ "$1" == "-u" && "$2" == "installer-user" ]]; then printf '501\n'; return 0; fi
    command id "$@"
}}
sudo() {{
    [[ "$1" == "-H" ]] && shift
    [[ "$1" == "-u" ]] || return 97
    local user="$2"
    shift 2
    if [[ "$1" == "sh" ]]; then
        printf '/Users/installer-user'
        return 0
    fi
    printf '%s|%s|%s\n' "$user" "/Users/installer-user" "$*" >> "{}"
    HOME=/Users/installer-user "$@"
}}
export SUDO_USER=installer-user
DEST="{}"
NO_PULL=0
QUIET=1
OFFLINE_TARBALL=""
provision_default_models
"#,
            sudo_log.display(),
            bin_dir.display(),
        ),
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let calls = fs::read_to_string(root.path().join("calls.log")).expect("read call log");
    assert_eq!(calls, "pull all\ndoctor --json\n");
    let sudo_calls = fs::read_to_string(sudo_log).expect("read sudo call log");
    assert!(
        sudo_calls
            .lines()
            .all(|line| line.starts_with("installer-user|/Users/installer-user|"))
    );
    let homes = fs::read_to_string(root.path().join("homes.log")).expect("read HOME log");
    assert_eq!(homes, "/Users/installer-user\n/Users/installer-user\n");
}
