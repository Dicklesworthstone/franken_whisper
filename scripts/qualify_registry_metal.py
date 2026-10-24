#!/usr/bin/env python3
"""Qualify native Metal on an already-qualified registry .crate (Python 3.11+).

Requires Apple Silicon, a usable GPU, local Cargo and the committed toolchain.
Never publishes, uses sibling sources, changes a candidate, or treats a missing
GPU as success. Fault injection models unsubmitted commands, not real GPU resets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib
from typing import Any

import verify_registry_archives as archives

REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
PHASES = ("registry_artifacts", "registry_resolution", "whisper_tests",
          "oracle_conformance", "package_verification", "packaged_artifact_validation")
TESTS = {
    "registry_metal_production": {"production_gemm_matches_reference_without_fault_injection"},
    "registry_metal_faults": {
        "transient_retries_preserve_gemm_output",
        "exhausted_attempts_preserve_output_and_recovery_restores_retries",
        "failed_buffers_release_queue_capacity",
        "injected_faults_are_thread_local",
        "fused_encoder_retries_without_returning_partial_activations",
    },
}
FAULT_OVERRIDE = "profile.release.package.ft-kernel-metal.debug-assertions=true"
RUNTIME_PACKAGES = {"frankentorch-core", "frankentorch-kernel-cpu", "ft-kernel-metal",
                    "frankenjax-core", "frankenjax-lax"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(repo: Path, revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"],
                                   stderr=subprocess.PIPE)


def require_host() -> None:
    if sys.platform != "darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise archives.AuditError("native Metal qualification requires an Apple Silicon host")


def require_receipt(report: dict[str, Any], revision: str) -> None:
    if (report.get("status") != "registry_candidate_verified"
            or report.get("published") is not False
            or report.get("source_revision") != revision
            or any(report.get(p) != "pass" for p in PHASES)
            or type(report.get("packaged_oracle_tests_passed")) is not int
            or report["packaged_oracle_tests_passed"] < 1):
        raise archives.AuditError("candidate lacks complete same-revision package/oracle qualification")


def require_tests(data: bytes, expected: set[str]) -> int:
    """Require every named test to start and pass exactly once in one suite."""
    started: set[str] = set()
    passed: set[str] = set()
    suite_started = suite_finished = False
    for line in data.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict) or suite_finished:
            raise archives.AuditError("invalid or trailing native test events")
        kind, state = event.get("type"), event.get("event")
        if kind == "suite" and state == "started" and not suite_started:
            if type(event.get("test_count")) is not int or event["test_count"] != len(expected):
                raise archives.AuditError("native test inventory is incomplete")
            suite_started = True
        elif kind == "test" and suite_started:
            name = event.get("name")
            if name not in expected:
                raise archives.AuditError(f"unexpected native test: {name}")
            if state == "started" and name not in started:
                started.add(name)
            elif state == "ok" and name in started and name not in passed:
                passed.add(name)
            else:
                raise archives.AuditError(f"failed, skipped or duplicate native test: {name}")
        elif kind == "suite" and state == "ok" and suite_started:
            counts = {"passed": len(expected), "failed": 0, "ignored": 0, "filtered_out": 0}
            if (started != expected or passed != expected
                    or any(type(event.get(k)) is not int or event[k] != v for k, v in counts.items())):
                raise archives.AuditError("native suite did not execute every required test")
            suite_finished = True
        else:
            raise archives.AuditError("unexpected/failed native test suite")
    if not suite_finished:
        raise archives.AuditError("missing native test completion")
    return len(passed)


def check_lock(data: bytes, root: dict[str, Any], audited: list[dict[str, Any]]) -> None:
    expected = {p["name"]: (p["version"], p["archive_sha256"]) for p in audited}
    if not RUNTIME_PACKAGES <= expected.keys():
        raise archives.AuditError("archive audit omits a required implementation")
    local = 0
    found = set()
    identities = set()
    for package in tomllib.loads(data.decode())["package"]:
        name, version = package["name"], package["version"]
        identity = (name, version, package.get("source"))
        if identity in identities:
            raise archives.AuditError("duplicate locked package")
        identities.add(identity)
        if identity == (root["name"], root["version"], None):
            local += 1
            continue
        if (package.get("source") != REGISTRY
                or not archives.HEX64.fullmatch(package.get("checksum", ""))):
            raise archives.AuditError(f"non-registry or unchecksummed input: {name}")
        if name in expected:
            if (version, package["checksum"]) != expected[name]:
                raise archives.AuditError(f"locked artifact differs from audited registry package: {name}")
            found.add(name)
    if local != 1 or not RUNTIME_PACKAGES <= found:
        raise archives.AuditError("packaged registry closure is incomplete")


def check_metadata(data: bytes, root: dict[str, Any]) -> None:
    metadata = json.loads(data)
    root_id = metadata["resolve"]["root"]
    packages = metadata["packages"]
    local = [p for p in packages if p.get("source") is None]
    if (len(local) != 1 or local[0]["id"] != root_id
            or any(local[0][k] != root[k] for k in ("name", "version"))):
        raise archives.AuditError("metadata has the wrong local release root")
    if any(p["id"] != root_id and p.get("source") != REGISTRY for p in packages):
        raise archives.AuditError("Cargo resolved a non-registry dependency")
    metal = [p for p in packages if p["name"] == "ft-kernel-metal"]
    node = next(n for n in metadata["resolve"]["nodes"] if n["id"] == root_id)
    if (len(metal) != 1 or metal[0].get("source") != REGISTRY
            or not any(d["name"] == "ft_kernel_metal" and d["pkg"] == metal[0]["id"]
                       for d in node["deps"])):
        raise archives.AuditError("native probes are not linked to the registry Metal dependency")


def clean_environment(output: Path) -> dict[str, str]:
    excluded = {"RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC", "RUSTDOC", "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER", "RUSTC_BOOTSTRAP", "RUSTUP_TOOLCHAIN", "RUST_MIN_STACK"}
    prefixes = ("CARGO_", "RUST_TEST_", "FT_METAL_", "DYLD_")
    env = {k: v for k, v in os.environ.items() if k not in excluded and not k.startswith(prefixes)}
    env["CARGO_HOME"] = str(output / "cargo-home")
    return env


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise archives.AuditError(f"symlink appeared in native source: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path.read_bytes())
    return result


def qualify(candidate: Path, output: Path, repo: Path, revision: str,
            cargo: str, report: dict[str, Any]) -> None:
    receipt = json.loads((candidate / "registry-release-receipt.json").read_text())
    require_receipt(receipt, revision)
    source_manifest = (candidate / "source/Cargo.toml").read_bytes()
    if sha256(source_manifest) != receipt.get("manifest_sha256"):
        raise archives.AuditError("qualified manifest changed")
    root = tomllib.loads(source_manifest.decode())["package"]
    if (root["name"] != "franken_whisper"
            or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", root["version"])):
        raise archives.AuditError("invalid release package identity")
    crate = candidate / "target/package" / f"{root['name']}-{root['version']}.crate"
    if crate.stat().st_size > archives.MAX_DOWNLOAD:
        raise archives.AuditError("qualified crate exceeds archive size bound")
    data = crate.read_bytes()
    if sha256(data) != receipt.get("crate_sha256"):
        raise archives.AuditError("qualified crate changed")
    files = archives.archive_files(data, root["name"], root["version"])
    manifest = tomllib.loads(files["Cargo.toml"][1].decode())
    archives.registry_dependencies(manifest)
    if any(manifest["package"][k] != root[k] for k in ("name", "version")):
        raise archives.AuditError("packaged identity mismatch")
    if sha256(files["Cargo.lock"][1]) != receipt.get("packaged_lock_sha256"):
        raise archives.AuditError("qualified packaged lock changed")
    # These must be genuine committed tests in the shipped archive, not an
    # unchecked test overlay or the developer's mutable sibling source.
    required = [f"tests/{target}.rs" for target in TESTS]
    required += [".cargo/config.toml", "rust-toolchain.toml"]
    for path in required:
        if path not in files or files[path][1] != git_blob(repo, revision, path):
            raise archives.AuditError(f"missing or altered committed native input: {path}")
    if any(path.startswith(".cargo/") and path != ".cargo/config.toml" for path in files):
        raise archives.AuditError("unexpected packaged Cargo configuration")
    specs_path = output / "source-contract.json"
    specs_path.write_bytes(git_blob(repo, revision, "scripts/registry-source-contract.json"))
    specs = archives.load_contract(specs_path)
    audit = archives.audit(specs)
    (output / "registry-archive-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["artifact_integrity"] != "pass":
        raise archives.AuditError("registry dependencies are missing or differ from the source contract")
    check_lock(files["Cargo.lock"][1], root, audit["verified"])
    report.update(source_revision=revision, crate_sha256=sha256(data),
                  packaged_lock_sha256=sha256(files["Cargo.lock"][1]),
                  audited_packages=audit["verified"], registry_artifacts="pass")
    source = output / "source"
    source.mkdir()
    for path, (mode, contents) in files.items():
        destination = source / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        destination.chmod(0o755 if mode & 0o111 else 0o644)
    before = inventory(source)
    env = clean_environment(output)

    def run(label: str, arguments: list[str], lane: str = "production") -> bytes:
        command = [cargo, *arguments]
        report["commands"].append(command)
        env["CARGO_TARGET_DIR"] = str(output / (lane + "-target"))
        with (output / f"{label}.stdout").open("wb") as stdout, (output / f"{label}.stderr").open("wb") as stderr:
            subprocess.run(command, cwd=source, env=env, stdout=stdout, stderr=stderr,
                           timeout=1800, check=True)
        if inventory(source) != before or sha256(crate.read_bytes()) != report["crate_sha256"]:
            raise archives.AuditError("native qualification changed its source or package inputs")
        return (output / f"{label}.stdout").read_bytes()

    run("cargo-version", ["--version", "--verbose"])
    run("fetch", ["fetch", "--locked"])
    check_metadata(run("metadata", ["metadata", "--frozen", "--all-features", "--format-version", "1"]), root)
    # No source replacement and no registry updates once the packaged lock is fetched.
    for target, names in TESTS.items():
        fault = target == "registry_metal_faults"
        args = ["test", "--frozen", "--release", "--all-features", "--test", target]
        if fault:
            args += ["--config", FAULT_OVERRIDE]
        args += ["--", "--ignored", "--test-threads=1", "-Z", "unstable-options", "--format=json"]
        count = require_tests(run(target, args, "fault" if fault else "production"), names)
        report["fault_tests_passed" if fault else "production_tests_passed"] = count
    report["status"] = "native_metal_verified"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory outside all checkouts")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--cargo", default="cargo", help="local Cargo executable, not a remote command wrapper")
    args = parser.parse_args(argv)
    output, candidate, repo = args.output.resolve(), args.candidate.resolve(), args.repo.resolve()
    report: dict[str, Any] = {
        "schema": 1, "status": "not_run", "published": False, "commands": [],
        "production_tests_passed": 0, "fault_tests_passed": 0, "registry_artifacts": "not_run",
        "host": {"system": sys.platform, "machine": platform.machine()},
        "fault_model": "unsubmitted_command_buffers_not_real_gpu_reset",
        "fault_build_override": FAULT_OVERRIDE,
    }
    owns_output = False
    try:
        require_host()
        if not archives.HEX40.fullmatch(args.expected_revision):
            raise archives.AuditError("expected revision must be a full commit SHA")
        if output.exists() or any(output == p or p in output.parents for p in (repo, candidate)):
            raise archives.AuditError("native evidence needs a new directory outside the checkout/candidate")
        for parent in (output, *output.parents):
            if any((parent / ".cargo" / n).exists() for n in ("config", "config.toml")):
                raise archives.AuditError(f"native build would inherit Cargo configuration: {parent}")
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        qualify(candidate, output, repo, args.expected_revision, args.cargo, report)
    except (archives.AuditError, OSError, ValueError, KeyError, TypeError, StopIteration,
            subprocess.SubprocessError) as exc:
        report.update(status="failed", error=str(exc))
        if owns_output:
            (output / "native-metal-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    (output / "native-metal-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
