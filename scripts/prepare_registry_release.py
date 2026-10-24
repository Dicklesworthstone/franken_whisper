#!/usr/bin/env python3
"""Stage and qualify a registry-only release without modifying development inputs.

Python 3.11+ and Git are required. The default mode also needs network access
and the configured Cargo runner. --prepare-only is NOT a release validation.
No mode publishes crates, changes the checkout, or supplies fallback sources.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib
from typing import Any

import verify_registry_archives as archives

ALIASES = {
    "ft-core": "frankentorch-core",
    "ft-kernel-cpu": "frankentorch-kernel-cpu",
    "ft-kernel-metal": "ft-kernel-metal",
    "fj-core": "frankenjax-core",
    "fj-lax": "frankenjax-lax",
}
REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
KINDS = ("dependencies", "dev-dependencies", "build-dependencies")
# These are release semantics, not dependency/source overrides. Keep the exact
# reviewed configuration (including its comments) in the candidate and package.
# A changed build baseline needs a deliberate review, not silent sanitization.
BUILD_CONFIG = {
    "target": {'cfg(target_arch = "x86_64")': {
        "rustflags": ["-Z", "threads=4", "-C", "target-cpu=x86-64-v3"]}},
    "env": {"RUST_MIN_STACK": "67108864"},
    "build": {"rustflags": ["-Z", "threads=4"]},
}


def check_build_config(data: bytes) -> None:
    if tomllib.loads(data.decode("utf-8")) != BUILD_CONFIG:
        raise archives.AuditError("unreviewed Cargo build configuration or source override")


def dependencies(manifest: dict[str, Any]):
    for section in (manifest, *manifest.get("target", {}).values()):
        for kind in KINDS:
            yield section.get(kind, {})


def inline(value: Any) -> str:
    """Render the limited TOML value types allowed in dependency declarations."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(inline(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {inline(v)}" for k, v in value.items()) + " }"
    raise archives.AuditError(f"unsupported dependency value {value!r}")


def registry_manifest(text: str, specs: list[dict[str, Any]]) -> str:
    """Change only dependency locations/identities; prove all other TOML survives."""
    original = tomllib.loads(text)
    expected = copy.deepcopy(original)
    versions = {s["name"]: s["version"] for s in specs}
    if not set(ALIASES.values()) <= versions.keys():
        raise archives.AuditError("source contract omits a required kernel/oracle package")
    features = original.get("features", {})
    if not {"dep:fj-core", "dep:fj-lax"} <= set(features.get("fj-oracle", [])):
        raise archives.AuditError("fj-oracle must retain both independent oracle dependencies")
    seen = set()

    def rewrite(alias: str, value: Any) -> Any:
        if not isinstance(value, dict):
            if alias in ALIASES:
                raise archives.AuditError(f"expected explicit package declaration for {alias}")
            return value
        value = dict(value)
        if {"git", "workspace", "registry", "registry-index"} & value.keys():
            raise archives.AuditError(f"unsupported source override for {alias}")
        if alias in ALIASES:
            name = ALIASES[alias]
            if value.get("package", alias) not in (alias, name):
                raise archives.AuditError(f"unexpected package identity for {alias}")
            if alias.startswith("fj-") and value.get("optional") is not True:
                raise archives.AuditError(f"{alias} must remain optional")
            value.update(package=name, version="=" + versions[name])
            seen.add(alias)
        value.pop("path", None)
        if not isinstance(value.get("version"), str) or not value["version"]:
            raise archives.AuditError(f"missing registry version for {alias}")
        return value

    for section in dependencies(expected):
        for alias, value in section.items():
            section[alias] = rewrite(alias, value)
    if seen != ALIASES.keys():
        raise archives.AuditError(f"missing dependency aliases: {sorted(ALIASES.keys() - seen)}")
    archives.registry_dependencies(expected)

    # Preserve the entire source manifest (comments, profiles, features, targets).
    # Unknown/multiline layouts fail the semantic equality assertion below rather
    # than silently dropping an entry or requiring a new TOML writer dependency.
    output, in_dependencies = [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("["):
            in_dependencies = bool(re.fullmatch(
                r"\[(?:.*\.)?(?:build-|dev-)?dependencies\]\s*(?:#.*)?", line.strip()))
        match = re.fullmatch(r"(\s*)([\w-]+)\s*=\s*(\{.*\})(\s*(?:#.*)?)(\n?)", line)
        if in_dependencies and match:
            indent, alias, value, comment, newline = match.groups()
            dep = tomllib.loads("value = " + value)["value"]
            line = f"{indent}{alias} = {inline(rewrite(alias, dep))}{comment}{newline}"
        output.append(line)
    rendered = "".join(output)
    if tomllib.loads(rendered) != expected:
        raise archives.AuditError("unsupported manifest layout; registry rewrite would be incomplete")
    return rendered


def check_lock(data: bytes, root: dict[str, str], specs: list[dict[str, Any]]) -> None:
    lock = tomllib.loads(data.decode())
    found: dict[str, set[str]] = {}
    root_count = 0
    for package in lock.get("package", []):
        name, version = package["name"], package["version"]
        if name == root["name"] and version == root["version"] and "source" not in package:
            root_count += 1
            continue
        if package.get("source") != REGISTRY:
            raise archives.AuditError(f"non-crates.io lock input: {name} {version}")
        if not archives.HEX64.fullmatch(package.get("checksum", "")):
            raise archives.AuditError(f"missing registry checksum for {name}")
        found.setdefault(name, set()).add(version)
    if root_count != 1:
        raise archives.AuditError("lockfile must contain exactly one local release root")
    for spec in specs:
        # The published helper is independently audited but is not a transitive
        # runtime dependency of Whisper (dependency dev-dependencies do not flow).
        if spec["name"] in ALIASES.values() and found.get(spec["name"]) != {spec["version"]}:
            raise archives.AuditError(f"missing/mismatched locked implementation: {spec['name']}")


def check_metadata(metadata: dict[str, Any], root: dict[str, str]) -> None:
    root_id = metadata.get("resolve", {}).get("root")
    roots = [p for p in metadata.get("packages", []) if p["id"] == root_id]
    if (len(roots) != 1 or roots[0].get("source") is not None
            or any(roots[0].get(k) != root[k] for k in ("name", "version"))):
        raise archives.AuditError("Cargo metadata does not describe the staged release root")
    for package in metadata["packages"]:
        if package["id"] != root_id and package.get("source") != REGISTRY:
            raise archives.AuditError(f"Cargo resolved a non-registry dependency: {package['name']}")


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def snapshot(root: Path, output: Path) -> tuple[Path, str]:
    root, output = root.resolve(), output.resolve()
    if Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve() != root:
        raise archives.AuditError("--repo must name the repository root")
    if root == output or root in output.parents:
        raise archives.AuditError("candidate must be outside the development checkout")
    # Cargo reads ancestor configuration independently of CARGO_HOME.
    for parent in (output, *output.parents):
        if any((parent / ".cargo" / name).exists() for name in ("config", "config.toml")):
            raise archives.AuditError(f"candidate would inherit Cargo configuration from {parent}")
    if git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise archives.AuditError("tracked worktree changes must be committed before qualification")
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    output.mkdir(parents=True, exist_ok=False)
    staged = output / "source"
    staged.mkdir()
    # Read immutable Git blobs rather than copying the mutable worktree. This
    # also defeats assume-unchanged/skip-worktree and checkout content filters.
    for raw in git(root, "ls-tree", "-rz", "--full-tree", revision).split(b"\0"):
        if not raw:
            continue
        header, raw_path = raw.split(b"\t", 1)
        mode, kind, oid = header.split()
        path = Path(os.fsdecode(raw_path))
        if path.parts[0] in (".git", "target") or path == Path("Cargo.lock"):
            continue
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise archives.AuditError(f"unsupported tracked symlink/submodule: {path}")
        data = git(root, "cat-file", "blob", oid.decode("ascii"))
        if archives.git_object("blob", data).hex() != oid.decode("ascii"):
            raise archives.AuditError(f"Git blob identity mismatch: {path}")
        if path.parts[0] == ".cargo":
            if path != Path(".cargo/config.toml"):
                raise archives.AuditError(f"unreviewed Cargo configuration file: {path}")
            check_build_config(data)
        destination = staged / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        destination.chmod(0o755 if mode == b"100755" else 0o644)
    if (git(root, "rev-parse", "HEAD").decode().strip() != revision
            or git(root, "status", "--porcelain", "--untracked-files=no").strip()):
        raise archives.AuditError("source changed during candidate preparation")
    return staged, revision


def cargo_environment(output: Path) -> dict[str, str]:
    excluded = {"RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC", "RUSTDOC", "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER", "RUSTC_BOOTSTRAP", "RUSTUP_TOOLCHAIN",
                "RUST_MIN_STACK", "RUST_TEST_THREADS"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("CARGO_") and k not in excluded}
    # Keep HOME for rustup, SSH and the remote runner; Cargo's own user config,
    # credentials, index, cache and target directory are isolated explicitly.
    env["CARGO_HOME"] = str(output / "cargo-home")
    env["CARGO_TARGET_DIR"] = str(output / "target")
    return env


def input_inventory(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


def dependency_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare effective dependency options across Cargo's manifest normalization."""
    result = {}
    for target, section in [("", manifest), *manifest.get("target", {}).items()]:
        for kind in KINDS:
            for alias, value in section.get(kind, {}).items():
                dep = {"version": value} if isinstance(value, str) else dict(value)
                dep.setdefault("package", alias)
                dep.setdefault("optional", False)
                dep.setdefault("default-features", True)
                dep["features"] = sorted(dep.get("features", []))
                result[f"{target}/{kind}/{alias}"] = dep
    return result


def check_audited_checksums(lock: bytes, verified: list[dict[str, str]],
                            specs: list[dict[str, Any]]) -> None:
    checksums = {(p["name"], p["version"]): p["archive_sha256"] for p in verified}
    required = {(s["name"], s["version"]) for s in specs}
    if set(checksums) != required or len(verified) != len(required):
        raise archives.AuditError("archive audit did not verify the complete source contract")
    for package in tomllib.loads(lock.decode())["package"]:
        key = package["name"], package["version"]
        if key in checksums and package.get("checksum") != checksums[key]:
            raise archives.AuditError(f"locked checksum differs from audited archive: {key}")


def check_package_inputs(files: dict[str, tuple[int, bytes]], inputs: dict[str, str],
                         manifest: dict[str, Any]) -> None:
    """Bind distributed code, config and oracle to the inputs actually tested."""
    required = {p for p in inputs if p.startswith(("src/", "models/")) and p != "src/.DS_Store"}
    required.add("tests/conformance_oracle_tests.rs")
    required.update(p for p in (".cargo/config.toml", "rust-toolchain.toml") if p in inputs)
    missing = required - files.keys()
    if missing:
        raise archives.AuditError(f"package omits required build/conformance inputs: {sorted(missing)}")
    generated = {"Cargo.toml", "Cargo.lock", ".cargo_vcs_info.json"}
    for path, (_, data) in files.items():
        if path in generated:
            continue
        original = "Cargo.toml" if path == "Cargo.toml.orig" else path
        if inputs.get(original) != hashlib.sha256(data).hexdigest():
            raise archives.AuditError(f"package contains untested or changed input: {path}")
    if "Cargo.toml.orig" not in files:
        raise archives.AuditError("package omits its original manifest")
    packaged = tomllib.loads(files["Cargo.toml"][1].decode())
    archives.registry_dependencies(packaged)
    if packaged.get("features", {}) != manifest.get("features", {}):
        raise archives.AuditError("packaging changed the supported feature graph")
    if dependency_contract(packaged) != dependency_contract(manifest):
        raise archives.AuditError("packaging changed the tested dependency contract")
    if "profile" in manifest and packaged.get("profile") != manifest["profile"]:
        raise archives.AuditError("packaging changed the release profiles")
    if ".cargo/config.toml" in files:
        check_build_config(files[".cargo/config.toml"][1])


def extract_package(files: dict[str, tuple[int, bytes]], destination: Path) -> None:
    # archive_files already rejected traversal, duplicate names, symlinks and
    # oversized archives. Validate collisions before creating anything.
    for name in files:
        if any(parent.as_posix() in files for parent in Path(name).parents):
            raise archives.AuditError(f"package file/directory collision: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name, (mode, data) in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(0o755 if mode & 0o111 else 0o644)


def check_oracle_execution(stdout: bytes) -> int:
    """A zero-test or all-ignored libtest exit 0 is not conformance evidence."""
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    started = [e for e in events if e.get("type") == "suite" and e.get("event") == "started"]
    finished = [e for e in events if e.get("type") == "suite" and e.get("event") in ("ok", "failed")]
    passed = [e for e in events if e.get("type") == "test" and e.get("event") == "ok"]
    if (len(started) != 1 or len(finished) != 1 or not passed
            or len({e.get("name") for e in passed}) != len(passed)
            or started[0].get("test_count") != len(passed)
            or finished[0].get("event") != "ok"
            or finished[0].get("passed") != len(passed)
            or any(finished[0].get(k) != 0 for k in ("failed", "ignored", "measured", "filtered_out"))):
        raise archives.AuditError("oracle must execute a nonempty, unfiltered suite with no ignored or failed tests")
    return len(passed)


def qualify(staged: Path, output: Path, specs: list[dict[str, Any]],
            runner: list[str], report: dict[str, Any]) -> None:
    inputs = input_inventory(staged)
    if "tests/conformance_oracle_tests.rs" not in inputs:
        raise archives.AuditError("candidate has no independent oracle test source")
    artifact_report = archives.audit(specs)
    (output / "registry-archive-audit.json").write_text(json.dumps(artifact_report, indent=2) + "\n")
    if artifact_report["artifact_integrity"] != "pass":
        raise archives.AuditError("registry archives are not the required published implementations; see audit receipt")
    report["registry_artifacts"] = "pass"
    manifest = tomllib.loads((staged / "Cargo.toml").read_text())
    root = manifest["package"]
    env = cargo_environment(output)

    def run(label: str, *arguments: str, cwd: Path = staged,
            build_env: dict[str, str] | None = None) -> bytes:
        command = [*runner, *arguments]
        report["commands"].append(command)
        invocation = dict(label=label, command=command, cwd=str(cwd), status="running")
        report.setdefault("invocations", []).append(invocation)
        with (output / f"{label}.stdout").open("wb") as stdout, (output / f"{label}.stderr").open("wb") as stderr:
            try:
                subprocess.run(command, cwd=cwd, env=build_env or env, stdout=stdout, stderr=stderr, check=True)
            except (OSError, subprocess.CalledProcessError):
                invocation["status"] = "failed"
                raise
        invocation["status"] = "pass"
        return (output / f"{label}.stdout").read_bytes()

    run("cargo-version", "--version", "--verbose")
    # Resolve every feature and every platform; do not filter out the Metal edge.
    lock_path = staged / "Cargo.lock"
    lock_option = ["--locked"] if lock_path.exists() else []
    report["replaying_registry_lock"] = bool(lock_option)
    metadata = json.loads(run("metadata", "metadata", "--format-version", "1", "--all-features", *lock_option))
    check_metadata(metadata, root)
    lock = lock_path.read_bytes()
    check_lock(lock, root, specs)
    check_audited_checksums(lock, artifact_report["verified"], specs)
    report["registry_resolution"] = "pass"
    report["candidate_lock_sha256"] = hashlib.sha256(lock).hexdigest()
    run("tests-default", "test", "--locked", "--lib", "--tests")
    run("tests-oracle", "test", "--locked", "--features", "fj-oracle", "--test", "conformance_oracle_tests")
    report["oracle_conformance"] = "pass"
    run("tests-all-features", "test", "--locked", "--all-features", "--lib", "--tests")
    report["whisper_tests"] = "pass"
    # Verification intentionally stays enabled. --no-verify is never used.
    run("package", "package", "--locked", "--all-features")
    if lock_path.read_bytes() != lock:
        raise archives.AuditError("locked validation changed the candidate lockfile")
    name, version = root["name"], root["version"]
    crate_path = output / "target" / "package" / f"{name}-{version}.crate"
    crate_data = crate_path.read_bytes()
    files = archives.archive_files(crate_data, name, version)
    check_package_inputs(files, inputs, manifest)
    packaged = tomllib.loads(files["Cargo.toml"][1].decode())
    archives.registry_dependencies(packaged)
    if any(packaged.get("package", {}).get(k) != root[k] for k in ("name", "version")):
        raise archives.AuditError("packaged root identity differs from the candidate")
    if packaged.get("features", {}).get("fj-oracle") != root_manifest_features(staged).get("fj-oracle"):
        raise archives.AuditError("packaging changed/removed the oracle feature")
    check_lock(files["Cargo.lock"][1], root, specs)
    check_audited_checksums(files["Cargo.lock"][1], artifact_report["verified"], specs)
    report["package_verification"] = "pass"

    # Test what will be distributed, not a sibling-populated source checkout or
    # Cargo's internal verification directory. Use a second empty Cargo home and
    # target directory. Fetch under the packaged lock, then disallow resolution
    # and network access for every build/conformance invocation.
    extracted = output / "packaged-source"
    extract_package(files, extracted)
    packaged_env = cargo_environment(output / "packaged-validation")
    run("packaged-fetch", "fetch", "--locked", cwd=extracted, build_env=packaged_env)
    metadata = json.loads(run("packaged-metadata", "metadata", "--frozen", "--all-features",
                              "--format-version", "1", cwd=extracted, build_env=packaged_env))
    check_metadata(metadata, root)
    for label, features in (("default", []), ("all-features", ["--all-features"])):
        run(f"packaged-build-{label}", "build", "--frozen", "--release", "--lib", "--bins",
            *features, cwd=extracted, build_env=packaged_env)
    oracle_output = run("packaged-oracle", "test", "--frozen", "--release", "--features", "fj-oracle",
                        "--test", "conformance_oracle_tests", "--", "-Z", "unstable-options",
                        "--format=json", cwd=extracted, build_env=packaged_env)
    report["packaged_oracle_tests_passed"] = check_oracle_execution(oracle_output)
    for path, (_, data) in files.items():
        current = extracted / path
        if current.is_symlink() or not current.is_file() or current.read_bytes() != data:
            raise archives.AuditError(f"packaged validation changed an input: {path}")
    for path, digest in inputs.items():
        if path == "Cargo.lock":
            continue
        current = staged / path
        if current.is_symlink() or not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != digest:
            raise archives.AuditError(f"candidate validation changed an input: {path}")
    if crate_path.read_bytes() != crate_data:
        raise archives.AuditError("package archive changed during validation")
    report.update(status="registry_candidate_verified", package_verification="pass",
                  packaged_artifact_validation="pass", packaged_source=str(extracted),
                  crate=str(crate_path), crate_sha256=hashlib.sha256(crate_data).hexdigest(),
                  packaged_lock_sha256=hashlib.sha256(files["Cargo.lock"][1]).hexdigest())


def root_manifest_features(staged: Path) -> dict[str, Any]:
    return tomllib.loads((staged / "Cargo.toml").read_text()).get("features", {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the source checkout")
    parser.add_argument("--registry-lock", type=Path, help="replay a previously qualified registry Cargo.lock with --locked")
    parser.add_argument("--expected-revision", help="require this exact committed source revision (for CI handoffs)")
    parser.add_argument("--release-tag", help="require v<package.version>; never changes the package version")
    parser.add_argument("--prepare-only", action="store_true", help="stage inputs only; does NOT establish registry/build readiness")
    parser.add_argument("--cargo-command", default="rch exec -- cargo", help="Cargo runner, split without invoking a shell")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    report: dict[str, Any] = {"schema": 1, "status": "not_prepared", "published": False,
        "registry_artifacts": "not_run", "registry_resolution": "not_run",
        "whisper_tests": "not_run", "oracle_conformance": "not_run", "package_verification": "not_run",
        "packaged_artifact_validation": "not_run",
        "metal_gpu_conformance": "not_certified_by_this_gate", "commands": []}
    owns_output = False
    try:
        if output.exists():
            raise archives.AuditError("output directory already exists; refusing to overwrite evidence")
        staged, revision = snapshot(args.repo, output)
        owns_output = True
        report["source_revision"] = revision
        if args.expected_revision is not None and args.expected_revision != revision:
            raise archives.AuditError("source revision differs from the requested release input")
        specs = archives.load_contract(staged / "scripts" / "registry-source-contract.json")
        manifest_path = staged / "Cargo.toml"
        version = tomllib.loads(manifest_path.read_text())["package"]["version"]
        if args.release_tag is not None:
            report["release_tag"] = args.release_tag
            if args.release_tag != "v" + version:
                raise archives.AuditError(f"release tag {args.release_tag!r} does not match package version {version}")
        manifest_path.write_text(registry_manifest(manifest_path.read_text(), specs))
        if args.registry_lock:
            lock = args.registry_lock.read_bytes()
            check_lock(lock, tomllib.loads(manifest_path.read_text())["package"], specs)
            (staged / "Cargo.lock").write_bytes(lock)
        report.update(status="prepared_only", source_revision=revision,
                      manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        config = staged / ".cargo/config.toml"
        report["build_config_sha256"] = (
            hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None)
        if not args.prepare_only:
            runner = shlex.split(args.cargo_command)
            if not runner:
                raise archives.AuditError("Cargo runner cannot be empty")
            qualify(staged, output, specs, runner, report)
    except (archives.AuditError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        report.update(status="failed", error=str(exc))
        if owns_output:
            (output / "registry-release-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    (output / "registry-release-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
