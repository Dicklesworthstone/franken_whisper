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
import shutil
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
    for raw in git(root, "ls-files", "-z").split(b"\0"):
        if not raw:
            continue
        path = Path(os.fsdecode(raw))
        if path.parts[0] in (".git", ".cargo", "target") or path == Path("Cargo.lock"):
            continue
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise archives.AuditError(f"unsupported tracked symlink/submodule: {path}")
        destination = staged / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if (git(root, "rev-parse", "HEAD").decode().strip() != revision
            or git(root, "status", "--porcelain", "--untracked-files=no").strip()):
        raise archives.AuditError("source changed during candidate preparation")
    return staged, revision


def cargo_environment(output: Path) -> dict[str, str]:
    excluded = {"RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC", "RUSTDOC", "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER", "RUSTC_BOOTSTRAP", "RUSTUP_TOOLCHAIN"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("CARGO_") and k not in excluded}
    # Keep HOME for rustup, SSH and the remote runner; Cargo's own user config,
    # credentials, index, cache and target directory are isolated explicitly.
    env["CARGO_HOME"] = str(output / "cargo-home")
    env["CARGO_TARGET_DIR"] = str(output / "target")
    return env


def qualify(staged: Path, output: Path, specs: list[dict[str, Any]],
            runner: list[str], report: dict[str, Any]) -> None:
    artifact_report = archives.audit(specs)
    (output / "registry-archive-audit.json").write_text(json.dumps(artifact_report, indent=2) + "\n")
    if artifact_report["artifact_integrity"] != "pass":
        raise archives.AuditError("registry archives are not the required published implementations; see audit receipt")
    report["registry_artifacts"] = "pass"
    root = tomllib.loads((staged / "Cargo.toml").read_text())["package"]
    env = cargo_environment(output)

    def run(label: str, *arguments: str) -> bytes:
        command = [*runner, *arguments]
        report["commands"].append(command)
        with (output / f"{label}.stdout").open("wb") as stdout, (output / f"{label}.stderr").open("wb") as stderr:
            subprocess.run(command, cwd=staged, env=env, stdout=stdout, stderr=stderr, check=True)
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
    packaged = tomllib.loads(files["Cargo.toml"][1].decode())
    archives.registry_dependencies(packaged)
    if any(packaged.get("package", {}).get(k) != root[k] for k in ("name", "version")):
        raise archives.AuditError("packaged root identity differs from the candidate")
    if packaged.get("features", {}).get("fj-oracle") != root_manifest_features(staged).get("fj-oracle"):
        raise archives.AuditError("packaging changed/removed the oracle feature")
    check_lock(files["Cargo.lock"][1], root, specs)
    report.update(status="registry_candidate_verified", package_verification="pass",
                  crate=str(crate_path), crate_sha256=hashlib.sha256(crate_data).hexdigest(),
                  packaged_lock_sha256=hashlib.sha256(files["Cargo.lock"][1]).hexdigest())


def root_manifest_features(staged: Path) -> dict[str, Any]:
    return tomllib.loads((staged / "Cargo.toml").read_text()).get("features", {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the source checkout")
    parser.add_argument("--registry-lock", type=Path, help="replay a previously qualified registry Cargo.lock with --locked")
    parser.add_argument("--prepare-only", action="store_true", help="stage inputs only; does NOT establish registry/build readiness")
    parser.add_argument("--cargo-command", default="rch exec -- cargo", help="Cargo runner, split without invoking a shell")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    report: dict[str, Any] = {"schema": 1, "status": "not_prepared", "published": False,
        "registry_artifacts": "not_run", "registry_resolution": "not_run",
        "whisper_tests": "not_run", "oracle_conformance": "not_run", "package_verification": "not_run",
        "metal_gpu_conformance": "not_certified_by_this_gate", "commands": []}
    owns_output = False
    try:
        if output.exists():
            raise archives.AuditError("output directory already exists; refusing to overwrite evidence")
        staged, revision = snapshot(args.repo, output)
        owns_output = True
        specs = archives.load_contract(staged / "scripts" / "registry-source-contract.json")
        manifest_path = staged / "Cargo.toml"
        manifest_path.write_text(registry_manifest(manifest_path.read_text(), specs))
        if args.registry_lock:
            lock = args.registry_lock.read_bytes()
            check_lock(lock, tomllib.loads(manifest_path.read_text())["package"], specs)
            (staged / "Cargo.lock").write_bytes(lock)
        report.update(status="prepared_only", source_revision=revision,
                      manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
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
