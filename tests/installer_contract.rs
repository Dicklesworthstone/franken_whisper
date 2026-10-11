#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Link the tracked fake `fw` into `root/bin` and record its outcome beside it.
///
/// The test never writes a script it then execs (bd-wijg): on Linux, execve
/// fails with ETXTBSY while any process holds the file open for writing, and a
/// child that another test thread forks while `fs::write` has it open keeps
/// that descriptor until the child execs. Only the outcome DATA file is
/// written; the executable is a symlink to a tracked script nothing writes.
fn write_fake_fw(root: &Path, pull_status: i32, doctor_ready: bool) -> std::path::PathBuf {
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).expect("create fake bin directory");
    let tracked = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/installer/fake_fw.sh");
    let mode = fs::metadata(&tracked)
        .expect("tracked fake fw metadata")
        .permissions()
        .mode();
    assert_eq!(mode & 0o111, 0o111, "tracked fake fw must be executable");
    fs::write(
        bin_dir.join("fake_fw_outcome"),
        format!("{pull_status} {doctor_ready}\n"),
    )
    .expect("write fake fw outcome");
    std::os::unix::fs::symlink(&tracked, bin_dir.join("fw")).expect("link fake fw");
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
    for shell in installer_test_shells() {
        let status = Command::new(&shell)
            .arg("-n")
            .arg(&install_script)
            .status()
            .expect("run bash syntax check");
        assert!(status.success(), "syntax check with {}", shell.display());
    }
}

/// Exercise stock macOS Bash as well as a newer Bash selected through PATH.
/// FW_INSTALLER_TEST_BASH can add another interpreter, e.g. a Linux Bash 3.2
/// build, without changing which shell the other installer tests use.
fn installer_test_shells() -> Vec<PathBuf> {
    let mut shells = vec![PathBuf::from("bash")];
    for path in ["/bin/bash", "/opt/homebrew/bin/bash", "/usr/local/bin/bash"] {
        if Path::new(path).exists() {
            shells.push(PathBuf::from(path));
        }
    }
    if let Some(shell) = std::env::var_os("FW_INSTALLER_TEST_BASH") {
        assert!(
            !shell.is_empty(),
            "FW_INSTALLER_TEST_BASH must name a shell"
        );
        shells.push(PathBuf::from(shell));
    }
    shells
}

/// Like `source_and_run`, but with an explicit shell and no inherited proxy
/// settings. `/bin/bash` is what a stock macOS `curl | bash` gets: bash 3.2,
/// where `set -u` treats an empty array as unbound.
fn source_and_run_with_shell(
    shell: &Path,
    root: &Path,
    body: &str,
    extra_env: &[(&str, &str)],
) -> std::process::Output {
    let install_script = Path::new(env!("CARGO_MANIFEST_DIR")).join("install.sh");
    let mut command = Command::new(shell);
    command
        .args(["--noprofile", "--norc"])
        .arg("-c")
        .arg(format!("source \"$1\" --no-pull\n{body}"))
        .arg("installer-contract")
        .arg(install_script)
        .env("HOME", root)
        .env("FW_INSTALL_PROMPT_TIMEOUT", "1");
    for name in [
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
        "BASH_ENV",
        "VERSION",
        "OWNER",
        "REPO",
        "DEST",
        "FW_INSTALL_DIR",
        "ARTIFACT_URL",
        "CHECKSUM",
    ] {
        command.env_remove(name);
    }
    for (name, value) in extra_env {
        command.env(name, value);
    }
    command.output().expect("run installer harness")
}

#[test]
fn every_proxy_args_expansion_is_safe_when_the_list_is_empty() {
    // On bash < 4.4 a bare "${PROXY_ARGS[@]}" with no proxy configured is an
    // unbound-variable error under `set -u`; inside an `if` it ended the whole
    // installer with exit status 0 before anything was downloaded.
    let script = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("install.sh"))
        .expect("read install.sh");
    // The comment explaining the guard is not a sixth curl invocation.
    let script = script
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let guarded = r#"${PROXY_ARGS[@]+"${PROXY_ARGS[@]}"}"#;
    let guarded_uses = script.matches(guarded).count();
    assert_eq!(
        guarded_uses, 5,
        "expected all five curl calls to use {guarded}"
    );
    assert_eq!(
        script.matches("PROXY_ARGS[@]").count(),
        2 * guarded_uses,
        "every PROXY_ARGS expansion must use {guarded}"
    );
}

const PROXY_CONFIGURATIONS: [Option<&str>; 5] = [
    None,
    Some(""),
    Some("http://proxy.example:3128"),
    Some("http://user:p a'$`*?[x]@proxy.example:3128"),
    Some(r#"http://user name:p$a;ss*?[ab]|\"'@proxy.example:3128"#),
];

fn assert_curl_args(root: &Path, call: &str, proxy: Option<&str>, suffix: &[&str]) {
    // NUL separation distinguishes zero arguments from one empty argument and
    // preserves spaces, quotes and delimiters occurring inside a proxy value.
    let args = fs::read_to_string(root.join(format!("{call}.argv"))).expect("curl was called");
    assert!(args.ends_with('\0'), "unterminated argv log for {call}");
    let actual: Vec<&str> = args.split_terminator('\0').collect();
    let mut expected = vec!["-fsSL"];
    if let Some(proxy) = proxy.filter(|value| !value.is_empty()) {
        expected.extend(["--proxy", proxy]);
    }
    expected.extend_from_slice(suffix);
    assert_eq!(actual, expected, "curl argv for {call}, proxy {proxy:?}");
}

#[test]
fn network_check_runs_with_and_without_a_proxy_on_the_system_shell() {
    let body = r#"
curl() {
    printf '%s\0' "$@" >"$HOME/network.argv"
    return "$FW_TEST_CURL_STATUS"
}
DEST="$HOME/install directory"
preflight_checks
printf 'reached-after-preflight\n'
# Preserve an explicit later failure status through the installer's EXIT trap.
exit "$FW_TEST_INSTALL_STATUS"
"#;
    for shell in installer_test_shells() {
        for proxy in PROXY_CONFIGURATIONS {
            for (curl_status, install_status) in [("0", 0), ("7", 0), ("0", 23), ("7", 23)] {
                let root = tempfile::tempdir().expect("temporary proxy harness");
                let mut extra: Vec<(&str, &str)> =
                    proxy.map(|url| ("HTTPS_PROXY", url)).into_iter().collect();
                extra.push(("FW_TEST_CURL_STATUS", curl_status));
                let exit_status = install_status.to_string();
                extra.push(("FW_TEST_INSTALL_STATUS", &exit_status));
                let output = source_and_run_with_shell(&shell, root.path(), body, &extra);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert_eq!(
                    output.status.code(),
                    Some(install_status),
                    "{} with proxy {proxy:?}, curl {curl_status}, install {install_status}: {stderr}",
                    shell.display()
                );
                // Exit 0 alone is insufficient: Bash 3.2 used to exit 0
                // before curl or any subsequent installer step ran.
                assert_eq!(output.stdout, b"reached-after-preflight\n");
                assert!(root.path().join("install directory").is_dir());
                assert_eq!(
                    stderr.contains("Could not reach github.com"),
                    curl_status != "0",
                    "stderr: {stderr}"
                );
                assert_curl_args(
                    root.path(),
                    "network",
                    proxy,
                    &[
                        "--connect-timeout",
                        "3",
                        "--max-time",
                        "5",
                        "-o",
                        "/dev/null",
                        "https://github.com",
                    ],
                );
            }
        }
    }
}

#[test]
fn release_and_rustup_curl_calls_preserve_proxy_arguments() {
    let body = r#"
QUIET=1
MAX_RETRIES=1
curl() {
    case "${@: -1}" in
        */releases\?per_page=50)
            printf '%s\0' "$@" >"$HOME/api.argv"
            return 22
            ;;
        */releases/latest)
            printf '%s\0' "$@" >"$HOME/redirect.argv"
            printf 'https://github.com/Dicklesworthstone/franken_whisper/releases/tag/v0.10.0\n'
            ;;
        https://example.invalid/release.tar.gz)
            printf '%s\0' "$@" >"$HOME/download.argv"
            while [ "$#" -gt 1 ]; do
                if [ "$1" = -o ]; then
                    printf 'archive payload\n' >"$2"
                    return 0
                fi
                shift
            done
            return 99
            ;;
        https://sh.rustup.rs)
            printf '%s\0' "$@" >"$HOME/rustup.argv"
            printf 'fake rustup payload\n'
            ;;
        *) return 99 ;;
    esac
}
resolve_version
[ "$VERSION" = v0.10.0 ]
download_file https://example.invalid/release.tar.gz "$HOME/release with spaces.tar.gz"
# Simulate cargo becoming available after the bootstrap. Consume its input
# without executing downloaded code or installing anything on the test host.
command() {
    if [ "$#" -eq 2 ] && [ "$1" = -v ] && [ "$2" = cargo ]; then
        [ -s "$HOME/rustup-input" ]
    else
        builtin command "$@"
    fi
}
sh() { cat >"$HOME/rustup-input"; }
ensure_rust
printf 'reached-after-curl-paths\n'
"#;
    for shell in installer_test_shells() {
        for proxy in PROXY_CONFIGURATIONS {
            let root = tempfile::tempdir().expect("temporary curl harness");
            let extra: Vec<(&str, &str)> =
                proxy.map(|url| ("HTTPS_PROXY", url)).into_iter().collect();
            let output = source_and_run_with_shell(&shell, root.path(), body, &extra);
            assert_eq!(
                output.status.code(),
                Some(0),
                "{} with proxy {proxy:?}: {}",
                shell.display(),
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(output.stdout, b"reached-after-curl-paths\n");
            assert_curl_args(
                root.path(),
                "api",
                proxy,
                &[
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    "30",
                    "-H",
                    "Accept: application/vnd.github.v3+json",
                    "https://api.github.com/repos/Dicklesworthstone/franken_whisper/releases?per_page=50",
                ],
            );
            assert_curl_args(
                root.path(),
                "redirect",
                proxy,
                &[
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{url_effective}",
                    "https://github.com/Dicklesworthstone/franken_whisper/releases/latest",
                ],
            );
            let partial = root.path().join("release with spaces.tar.gz.part");
            assert_curl_args(
                root.path(),
                "download",
                proxy,
                &[
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "120",
                    "--retry",
                    "2",
                    "-o",
                    partial.to_str().expect("UTF-8 temporary path"),
                    "https://example.invalid/release.tar.gz",
                ],
            );
            assert_curl_args(root.path(), "rustup", proxy, &["https://sh.rustup.rs"]);
            assert_eq!(
                fs::read(root.path().join("release with spaces.tar.gz"))
                    .expect("published download"),
                b"archive payload\n"
            );
            assert!(
                !partial.exists(),
                "successful download must publish the partial"
            );
            assert_eq!(
                fs::read(root.path().join("rustup-input")).expect("bootstrap received input"),
                b"fake rustup payload\n"
            );
        }
    }
}

#[test]
fn installer_exit_status_reflects_the_download_outcome_after_preflight() {
    let body = r#"
VERSION=v0.10.0
DEST="$HOME/install directory"
LOCK_DIR="$HOME/installer.lock.d"
INSTALLER_TEMP_ROOT="$HOME"
INSTALLER_TEMP_TEMPLATE="$HOME/franken-whisper-install.XXXXXX"
detect_platform() { PLATFORM=darwin_arm64; }
curl() {
    printf 'network\n' >>"$HOME/stages.log"
    return "$FW_TEST_CURL_STATUS"
}
# Keep the real main/preflight/EXIT-trap control flow, but isolate installation
# side effects. The download stage must be reached before any success verdict.
download_release() {
    printf 'download\n' >>"$HOME/stages.log"
    return "$FW_TEST_DOWNLOAD_STATUS"
}
maybe_add_path() { :; }
print_summary() { printf 'install-summary\n'; }
main
printf 'reached-after-main\n'
"#;
    for shell in installer_test_shells() {
        for proxy in PROXY_CONFIGURATIONS {
            for curl_status in ["0", "7"] {
                for download_status in ["0", "22"] {
                    let root = tempfile::tempdir().expect("temporary main harness");
                    let mut extra: Vec<(&str, &str)> =
                        proxy.map(|url| ("HTTPS_PROXY", url)).into_iter().collect();
                    extra.extend([
                        ("FW_TEST_CURL_STATUS", curl_status),
                        ("FW_TEST_DOWNLOAD_STATUS", download_status),
                    ]);
                    let output = source_and_run_with_shell(&shell, root.path(), body, &extra);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let succeeded = download_status == "0";
                    assert_eq!(
                        output.status.code(),
                        Some(if succeeded { 0 } else { 1 }),
                        "{} with proxy {proxy:?}, curl {curl_status}, download {download_status}: {stderr}",
                        shell.display()
                    );
                    assert_eq!(
                        fs::read(root.path().join("stages.log"))
                            .expect("installer reached download"),
                        b"network\ndownload\n"
                    );
                    assert!(
                        !root.path().join("installer.lock.d").exists(),
                        "EXIT cleanup must release the installer lock"
                    );
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    assert_eq!(stdout.contains("install-summary"), succeeded, "{stdout}");
                    assert_eq!(stdout.contains("reached-after-main"), succeeded, "{stdout}");
                    assert_eq!(
                        stderr.contains("Binary download or verification failed"),
                        !succeeded,
                        "{stderr}"
                    );
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn non_runnable_linux_upgrade_preserves_both_commands_and_reports_alsa_prerequisite() {
    let root = tempfile::tempdir().expect("create installer validation workspace");
    let destination = root.path().join("installed");
    fs::create_dir(&destination).expect("create old installation");
    fs::write(destination.join("fw"), b"original short command").expect("write old alias");
    fs::write(
        destination.join("franken_whisper"),
        b"original long command",
    )
    .expect("write old primary");
    let invalid = root.path().join("invalid-executable");
    fs::write(&invalid, b"\x7fELF\x00\x00\x00\x00").expect("write non-runnable executable");
    fs::set_permissions(&invalid, fs::Permissions::from_mode(0o700)).expect("set executable mode");
    let output = source_and_run(
        root.path(),
        &format!(
            "DEST='{}'\nVERSION=v0.10.0\ninstall_binary_pair '{}' '{}'",
            destination.display(),
            invalid.display(),
            invalid.display()
        ),
    );
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("libasound.so.2"), "stderr: {error}");
    assert!(
        error.contains("existing installation was not replaced"),
        "stderr: {error}"
    );
    assert_eq!(
        fs::read(destination.join("fw")).expect("read original alias"),
        b"original short command"
    );
    assert_eq!(
        fs::read(destination.join("franken_whisper")).expect("read original primary"),
        b"original long command"
    );
}

fn resolve_release_version(api_response: &str, redirect: &str) -> std::process::Output {
    source_and_run(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        &format!(
            r#"
VERSION=""
QUIET=1
MAX_RETRIES=1
curl() {{
    case "${{@: -1}}" in
        */releases\?per_page=50)
            cat <<'FW_RELEASES'
{api_response}
FW_RELEASES
            ;;
        */releases/latest) printf '%s\n' '{redirect}' ;;
        *) return 99 ;;
    esac
}}
resolve_version
printf 'resolved=%s\n' "$VERSION"
"#
        ),
    )
}

#[test]
fn latest_version_skips_prerelease_and_draft_records_after_their_tags() {
    let output = resolve_release_version(
        r#"[
  {
    "tag_name": "v9.0.0",
    "draft": false,
    "prerelease": true
  },
  {
    "tag_name": "v8.0.0",
    "draft": true,
    "prerelease": false
  },
  {
    "tag_name": "v0.10.0",
    "draft": false,
    "prerelease": false
  },
  {
    "tag_name": "v0.9.3",
    "draft": false,
    "prerelease": false
  }
]"#,
        "https://example.invalid/tag/v99.0.0",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"resolved=v0.10.0\n");
}

#[test]
fn latest_version_skips_model_head_and_requires_both_stable_flags() {
    let output = resolve_release_version(
        r#"[
  {
    "tag_name": "whisper-tiny-f16-v1",
    "draft": false,
    "prerelease": false
  },
  {
    "tag_name": "v9.0.0"
  },
  {
    "draft": false,
    "prerelease": false,
    "tag_name": "v0.10.0-rc.1"
  },
  {
    "draft": false,
    "prerelease": false,
    "tag_name": "v0.9.3",
    "assets": [
      {
        "tag_name": "v99.0.0",
        "draft": true,
        "prerelease": true
      }
    ]
  }
]"#,
        "https://example.invalid/tag/v99.0.0",
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"resolved=v0.9.3\n");
}

#[test]
fn latest_version_fails_without_a_qualifying_stable_tag() {
    let api_response = r#"[
  {
    "tag_name": "v9.0.0",
    "draft": false,
    "prerelease": true
  },
  {
    "tag_name": "v8.0.0",
    "draft": true,
    "prerelease": false
  },
  {
    "tag_name": "sortformer-f16-v1",
    "draft": false,
    "prerelease": false
  }
]"#;
    for redirect in [
        "https://example.invalid/tag/whisper-tiny-f16-v1",
        "https://example.invalid/tag/v0.10.0-rc.1",
    ] {
        let output = resolve_release_version(api_response, redirect);
        assert!(
            !output.status.success(),
            "redirect must not qualify: {redirect}"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("Could not resolve the latest release"),
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn public_site_examples_match_the_cli_contract() {
    let site = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("site")
            .join("index.html"),
    )
    .expect("read public site");

    assert!(site.contains("fw robot run --input meeting.m4a"));
    assert!(site.contains("select(.event == \"run_complete\") | .segments[]"));
    assert!(site.contains("fw transcribe --input recording.mp3 --json"));
    assert!(site.contains("fw pull all --json"));
    assert!(site.contains("fw doctor --json"));
    assert!(!site.contains("fw transcribe meeting.m4a --diarize --robot"));
    assert!(!site.contains("fw transcribe recording.mp3 --diarize"));
}

#[test]
fn playground_uses_current_wasm_bindgen_initialization_contract() {
    let app = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("site")
            .join("app.js"),
    )
    .expect("read playground app");
    let worker = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("site")
            .join("engine-worker.js"),
    )
    .expect("read playground worker");

    assert!(worker.contains("engine.default({ module_or_path: wasmUrl })"));
    assert!(!worker.contains("await engine.default();"));
    assert!(app.contains("const DENOISER_STALL_MS = 180_000"));
    assert!(app.contains("state.activeStage.startsWith(\"audio:denoise\")"));
    assert!(app.contains("stopUnresponsiveDenoiser()"));
    assert!(app.contains("Reload cached models"));
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
        "CHANGELOG.md",
    ] {
        fs::write(stage.join(member), member).expect("write archive member");
    }
    let archive = root
        .path()
        .join("franken_whisper-0.7.2-darwin_arm64.tar.gz");
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
            "CHANGELOG.md",
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
fn installer_rejects_duplicate_or_nested_changelog_members() {
    for members in [
        vec!["franken_whisper", "fw", "CHANGELOG.md", "CHANGELOG.md"],
        vec!["franken_whisper", "fw", "docs/CHANGELOG.md"],
    ] {
        let root = tempfile::tempdir().expect("temporary changelog archive harness");
        let archive = root.path().join("release.tar.gz");
        write_flat_targz(&archive, &members);
        let output = source_and_run(
            root.path(),
            &format!("validate_archive_members \"{}\" tar.gz", archive.display()),
        );
        assert!(
            !output.status.success(),
            "members must be rejected: {members:?}"
        );
    }
}

#[test]
fn installer_rejects_symlink_changelog_member() {
    let root = tempfile::tempdir().expect("temporary symlink archive harness");
    let stage = root.path().join("stage");
    fs::create_dir(&stage).expect("create symlink archive stage");
    for member in ["franken_whisper", "fw"] {
        fs::write(stage.join(member), member).expect("write regular archive member");
    }
    std::os::unix::fs::symlink("/etc/passwd", stage.join("CHANGELOG.md"))
        .expect("create rejected metadata symlink");
    let archive = root.path().join("release.tar.gz");
    let status = Command::new("tar")
        .arg("-czf")
        .arg(&archive)
        .arg("-C")
        .arg(&stage)
        .args(["franken_whisper", "fw", "CHANGELOG.md"])
        .status()
        .expect("create symlink archive");
    assert!(status.success());
    let output = source_and_run(
        root.path(),
        &format!("validate_archive_members \"{}\" tar.gz", archive.display()),
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("non-regular entry"));
}

/// Build a flat ustar tar.gz containing exactly `members`, each a regular
/// file. Hand-rolled instead of shelling out to `tar` because macOS bsdtar
/// special-cases AppleDouble `._*` names at archive-creation time (it folds
/// them into metadata or drops them), which would make these fixtures
/// platform-dependent; GNU tar on the Linux release path stores them
/// literally, and that is the case the installer must survive.
fn write_flat_targz(path: &Path, members: &[&str]) {
    use std::io::Write as _;
    let file = fs::File::create(path).expect("create archive file");
    let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    for member in members {
        let data = member.as_bytes();
        let mut header = [0u8; 512];
        header[..member.len()].copy_from_slice(member.as_bytes());
        header[100..107].copy_from_slice(b"0000644");
        header[108..115].copy_from_slice(b"0000000");
        header[116..123].copy_from_slice(b"0000000");
        header[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
        header[136..147].copy_from_slice(b"00000000000");
        header[156] = b'0';
        header[257..262].copy_from_slice(b"ustar");
        header[263..265].copy_from_slice(b"00");
        for byte in &mut header[148..156] {
            *byte = b' ';
        }
        let checksum: u32 = header.iter().map(|&byte| u32::from(byte)).sum();
        header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
        gz.write_all(&header).expect("write member header");
        gz.write_all(data).expect("write member data");
        gz.write_all(&vec![0u8; (512 - data.len() % 512) % 512])
            .expect("write member padding");
    }
    gz.write_all(&[0u8; 1024]).expect("write archive trailer");
    gz.finish().expect("finish gzip stream");
}

/// Issue #1 regression: the v0.7.2 linux_amd64 tarball was packaged on macOS
/// without `COPYFILE_DISABLE` and carried AppleDouble `._*` shadows of every
/// member, which the member allowlist rejected and fresh installs failed.
/// Sidecars of allowlisted members (plus `.DS_Store`) must now validate as
/// ignorable no-ops, while genuinely unexpected members stay rejected.
#[test]
fn installer_ignores_macos_appledouble_sidecar_members() {
    let root = tempfile::tempdir().expect("temporary archive harness");
    let archive = root.path().join("franken_whisper-0.7.2-linux_amd64.tar.gz");
    // Each AppleDouble sidecar immediately precedes its base member — the
    // order macOS tar actually emits (and the pairing bsdtar folds silently
    // when it reads the archive back; a trailing unpaired `._*` entry makes
    // bsdtar error out, which is not the shape under test).
    write_flat_targz(
        &archive,
        &[
            "._franken_whisper",
            "franken_whisper",
            "._fw",
            "fw",
            "._README.md",
            "README.md",
            "._LICENSE",
            "LICENSE",
            "._NOTICE.sortformer.txt",
            "NOTICE.sortformer.txt",
            "._THIRD_PARTY_NOTICES.md",
            "THIRD_PARTY_NOTICES.md",
            "._AGENTS.md",
            "AGENTS.md",
            "._CHANGELOG.md",
            "CHANGELOG.md",
            "._.DS_Store",
            ".DS_Store",
        ],
    );

    let output = source_and_run(
        root.path(),
        &format!("validate_archive_members \"{}\" tar.gz", archive.display()),
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Extraction must drop the sidecars: only allowlisted payload may land
    // in the extract directory (bsdtar would otherwise replay `._fw` as
    // extended attributes onto the real binary).
    let extract_root = root.path().join("tmp-extract");
    fs::create_dir(&extract_root).expect("create extract TMP");
    let extract = source_and_run(
        root.path(),
        &format!(
            r#"
TMP="{}"
install_binary_pair() {{ :; }}
extract_and_install "{}" tar.gz
ls -A "$TMP/extract"
"#,
            extract_root.display(),
            archive.display()
        ),
    );
    assert!(
        extract.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&extract.stderr)
    );
    let listed = String::from_utf8_lossy(&extract.stdout);
    assert!(
        !listed.contains("._") && !listed.contains(".DS_Store"),
        "sidecars must not be extracted, got: {listed}"
    );
}

/// A `._` member that does not shadow an allowlisted name is still an
/// unexpected member, as is any other stray file. macOS bsdtar hides some
/// unpaired `._*` names from `tar -tzf` before the validator ever sees them,
/// so the AppleDouble stray is only asserted where the system tar lists it
/// (GNU tar does; that is the platform the linux_amd64 archive targets).
#[test]
fn installer_still_rejects_unexpected_members() {
    for stray in ["._payload", "extra.txt"] {
        let root = tempfile::tempdir().expect("temporary archive harness");
        // The stray leads so bsdtar (which pairs a `._*` entry with the
        // member that follows it and errors on a trailing unpaired one) can
        // still list the archive; GNU tar lists every member either way.
        let archive = root.path().join("franken_whisper-0.7.2-linux_amd64.tar.gz");
        write_flat_targz(&archive, &[stray, "franken_whisper", "fw"]);

        let listing = Command::new("tar")
            .arg("-tzf")
            .arg(&archive)
            .output()
            .expect("list archive with system tar");
        assert!(listing.status.success());
        if !String::from_utf8_lossy(&listing.stdout)
            .lines()
            .any(|line| line == stray)
        {
            continue;
        }

        let output = source_and_run(
            root.path(),
            &format!("validate_archive_members \"{}\" tar.gz", archive.display()),
        );
        assert!(
            !output.status.success(),
            "stray member {stray} must be rejected"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("unexpected member"),
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
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
fn insufficient_model_cache_space_fails_before_download() {
    let root = tempfile::tempdir().expect("temporary installer harness");
    let bin_dir = write_fake_fw(root.path(), 0, false);
    let output = source_and_run(
        root.path(),
        &format!(
            r#"
available_kb_for_path() {{ printf '1\n'; }}
DEST="{}"
NO_PULL=0
QUIET=1
OFFLINE_TARBALL=""
provision_default_models
"#,
            bin_dir.display(),
        ),
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Insufficient free space"));
    let calls = fs::read_to_string(root.path().join("calls.log")).expect("read call log");
    assert_eq!(calls, "doctor --json\n");
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
