#!/usr/bin/env python3
"""Build unpublished Torch/JAX packages together, without changing their checkout.

Cargo 1.90+ stages selected workspace packages for package verification before
publication. These local artifacts are NOT evidence of crates.io availability.
Python 3.11+, Git and the committed Whisper Rust toolchain are required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shlex
import subprocess
import sys
import tomllib
from typing import Any

import verify_registry_archives as archives
import registry_test_outputs as test_outputs

FAMILIES = {
    "torch": ("frankentorch", {
        "frankentorch-core": "crates/ft-core",
        "frankentorch-kernel-cpu": "crates/ft-kernel-cpu",
        "ft-kernel-metal": "crates/ft-kernel-metal",
    }),
    "jax": ("frankenjax", {
        "frankenjax-test-utils": "crates/fj-test-utils",
        "frankenjax-core": "crates/fj-core",
        "frankenjax-lax": "crates/fj-lax",
    }),
}
REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
KINDS = ("dependencies", "build-dependencies", "dev-dependencies")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def value(item: Any) -> str:
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (str, int)):
        return json.dumps(item, ensure_ascii=False)
    if isinstance(item, list):
        return "[" + ", ".join(value(v) for v in item) + "]"
    if isinstance(item, dict):
        return "{ " + ", ".join(f"{json.dumps(k)} = {value(v)}" for k, v in item.items()) + " }"
    raise archives.AuditError(f"unsupported workspace TOML value: {item!r}")


def render(document: dict[str, Any]) -> str:
    text = "\n".join(f"{json.dumps(k)} = {value(v)}" for k, v in document.items()) + "\n"
    if tomllib.loads(text) != document:
        raise archives.AuditError("workspace TOML did not round-trip")
    return text


def dependency_entries(manifest: dict[str, Any]):
    for target, section in [("", manifest), *manifest.get("target", {}).items()]:
        for kind in KINDS:
            for alias, raw in section.get(kind, {}).items():
                yield (target, kind, alias), ({"version": raw} if isinstance(raw, str) else dict(raw))


def effective_dependencies(manifest: dict[str, Any], workspace: dict[str, Any]) -> dict:
    result = {}
    for key, dep in dependency_entries(manifest):
        if dep.pop("workspace", False):
            inherited = workspace["dependencies"][key[2]]
            inherited = {"version": inherited} if isinstance(inherited, str) else dict(inherited)
            features = inherited.get("features", []) + dep.get("features", [])
            inherited.update(dep)
            inherited["features"] = features
            dep = inherited
        if set(dep) - {"version", "package", "path", "features", "optional", "default-features"}:
            raise archives.AuditError(f"unsupported dependency/source options: {key}")
        if not isinstance(dep.get("version"), str) or not dep["version"]:
            raise archives.AuditError(f"dependency lacks a publication version: {key}")
        dep.pop("path", None)
        dep.setdefault("package", key[2])
        dep.setdefault("optional", False)
        dep.setdefault("default-features", True)
        dep["features"] = sorted(set(dep.get("features", [])))
        result[key] = dep
    return result


def publication_order(manifests: dict[str, dict], workspace: dict, paths: dict[str, str],
                      specs: dict[str, dict]) -> list[str]:
    """Include dev-dependency edges: the oracle test helper must ship first."""
    edges = {name: set() for name in manifests}
    for name, manifest in manifests.items():
        effective = effective_dependencies(manifest, workspace)
        for key, raw in dependency_entries(manifest):
            inherited = raw.get("workspace") is True
            dep = workspace["dependencies"][key[2]] if inherited else raw
            dep = {"version": dep} if isinstance(dep, str) else dep
            normalized = effective[key]
            dependency = normalized["package"]
            if "path" in dep:
                base = "" if inherited else paths[name]
                resolved = posixpath.normpath(posixpath.join(base, dep["path"]))
                if dependency not in paths or resolved != paths[dependency]:
                    raise archives.AuditError(f"{name}: dependency outside the publication set: {key}")
            if dependency in specs:
                if normalized["version"] != "=" + specs[dependency]["version"]:
                    raise archives.AuditError(f"{name}: internal dependency must use the exact prepared version: {dependency}")
                edges[name].add(dependency)
    order = []
    while len(order) < len(edges):
        ready = sorted(n for n, deps in edges.items() if n not in order and deps <= set(order))
        if not ready:
            raise archives.AuditError("publication dependencies contain a cycle")
        order.extend(ready)
    return order


def inventory(root: Path) -> dict[str, tuple[str, int]]:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise archives.AuditError(f"unexpected staged symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = (sha256(path.read_bytes()), path.stat().st_mode & 0o111)
    return result


def workspace_test_files(specs: dict[str, dict]) -> dict[str, str]:
    """Collect reviewed, repository-relative test data outside member crates.

    These are workspace test inputs, not package payload or registry substitutes.
    Pin the existing Git blobs: never regenerate a golden from the code under test.
    """
    files: dict[str, str] = {}
    for name, spec in specs.items():
        declared = spec.get("workspace_test_files", {})
        if not isinstance(declared, dict):
            raise archives.AuditError(f"{name}: workspace_test_files must map paths to Git blobs")
        for path, oid in declared.items():
            if not isinstance(path, str):
                raise archives.AuditError(f"{name}: invalid workspace test input path")
            relative = PurePosixPath(path)
            # Only explicit data files under artifacts are supported. No glob
            # pathspecs, source/configuration injection, or paths outside the repo.
            if (relative.is_absolute() or relative.as_posix() != path
                    or not path.startswith("artifacts/") or ".." in relative.parts
                    or any(c in path for c in "\\:*?[]")
                    or any(ord(c) < 32 or ord(c) == 127 for c in path)
                    or any(p.startswith(".") for p in relative.parts)
                    or relative.suffix not in (".txt", ".golden", ".json", ".csv", ".bin")):
                raise archives.AuditError(f"{name}: unsafe workspace test input path: {path!r}")
            if not isinstance(oid, str) or not archives.HEX40.fullmatch(oid):
                raise archives.AuditError(f"{name}: invalid workspace test input blob: {path}")
            if path in files and files[path] != oid:
                raise archives.AuditError(f"conflicting workspace test input blobs: {path}")
            files[path] = oid
    return dict(sorted(files.items()))


def prepare(repo: Path, output: Path, family: str, specs: list[dict], toolchain: bytes,
            expected_revision: str, without_overrides: bool, report: dict) -> tuple[Path, dict]:
    if not archives.HEX40.fullmatch(expected_revision):
        raise archives.AuditError("expected revision must be a full commit SHA")
    if git(repo, "rev-parse", "HEAD").decode().strip() != expected_revision:
        raise archives.AuditError("checkout differs from the expected revision")
    if git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise archives.AuditError("commit tracked changes before preparing dependencies")
    repository, paths = FAMILIES[family]
    canonical = "https://github.com/Dicklesworthstone/" + repository
    selected = {s["name"]: s for s in specs if s["repository"] == canonical}
    if set(selected) != set(paths):
        raise archives.AuditError("source contract must cover the complete publication family")
    test_files = workspace_test_files(selected)
    root = tomllib.loads(git(repo, "show", f"{expected_revision}:Cargo.toml").decode())
    if "workspace" not in root or set(root) - {"workspace", "profile", "patch", "replace"}:
        raise archives.AuditError("expected a virtual publication workspace")
    overrides = {k: root[k] for k in ("patch", "replace") if root.get(k)}
    if overrides and not without_overrides:
        raise archives.AuditError("workspace overrides do not ship; explicitly select --without-workspace-overrides to test the unpatched closure")
    channel = tomllib.loads(toolchain.decode())["toolchain"]["channel"]
    if not isinstance(channel, str) or not channel.startswith("nightly-"):
        raise archives.AuditError("use Whisper's committed dated nightly toolchain")
    staged = output / "source"
    staged.mkdir()
    # Keep paths, licenses, tests, fixtures and benches inside each selected crate.
    # Do not copy arbitrary worktree files, Cargo caches, root patches or sibling repos.
    # include_str! in existing unit tests also reaches repository-level golden
    # files. Stage only the explicitly reviewed blobs at their original paths.
    selected_files = [*paths.values(), "LICENSE", "README.md", *test_files]
    test_inputs = {}
    for entry in git(repo, "ls-tree", "-rz", "--full-tree", expected_revision, "--", *selected_files).split(b"\0"):
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = header.split()
        relative = PurePosixPath(os.fsdecode(raw_path))
        if (mode not in (b"100644", b"100755") or kind != b"blob"
                or relative.is_absolute() or ".." in relative.parts or ".cargo" in relative.parts):
            raise archives.AuditError(f"unsupported publication input: {relative}")
        if relative.parts[0] == "artifacts":
            expected_blob = test_files.get(relative.as_posix())
            if expected_blob != oid.decode() or mode != b"100644":
                raise archives.AuditError(f"missing or changed reviewed workspace test input: {relative}")
        content = git(repo, "cat-file", "blob", oid.decode())
        if archives.git_object("blob", content).hex() != oid.decode():
            raise archives.AuditError(f"Git blob mismatch: {relative}")
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        destination.chmod(0o755 if mode == b"100755" else 0o644)
        if relative.as_posix() in test_files:
            test_inputs[relative.as_posix()] = {"git_blob": oid.decode(), "sha256": sha256(content)}
    missing = test_files.keys() - test_inputs.keys()
    if missing:
        raise archives.AuditError(f"missing committed workspace test inputs: {sorted(missing)}")
    report["workspace_test_inputs"] = test_inputs
    workspace = root["workspace"]
    manifests = {}
    for name, path in paths.items():
        manifest = tomllib.loads((staged / path / "Cargo.toml").read_text())
        package = manifest["package"]
        for field in ("name", "version", "repository"):
            actual = package.get(field)
            if actual == {"workspace": True}:
                actual = workspace["package"][field]
            if actual != selected[name][field]:
                raise archives.AuditError(f"prepared package identity mismatch: {name}.{field}")
        tree = git(repo, "rev-parse", f"{expected_revision}:{path}/src").decode().strip()
        if tree != selected[name]["src_tree"]:
            raise archives.AuditError(f"unreviewed source tree: {name}")
        manifests[name] = manifest
    order = publication_order(manifests, workspace, paths, selected)
    reduced = copy.deepcopy(root)
    for key in ("patch", "replace"):
        reduced.pop(key, None)
    reduced["workspace"].update(members=list(paths.values()), **{"default-members": list(paths.values())})
    reduced["workspace"].pop("exclude", None)
    for alias, dep in list(reduced["workspace"].get("dependencies", {}).items()):
        if isinstance(dep, dict) and "path" in dep and dep["path"] not in paths.values():
            del reduced["workspace"]["dependencies"][alias]
    # All inherited options used by these crates must remain identical.
    for manifest in manifests.values():
        if effective_dependencies(manifest, workspace) != effective_dependencies(manifest, reduced["workspace"]):
            raise archives.AuditError("workspace reduction changed dependency options")
    (staged / "Cargo.toml").write_text(render(reduced))
    (staged / "rust-toolchain.toml").write_bytes(toolchain)
    report.update(source_revision=expected_revision, repository=canonical,
                  publication_order=order, omitted_workspace_overrides=overrides,
                  toolchain=channel, toolchain_sha256=sha256(toolchain),
                  source_inputs=inventory(staged), status="prepared_only")
    if git(repo, "rev-parse", "HEAD").decode().strip() != expected_revision:
        raise archives.AuditError("checkout changed during preparation")
    return staged, reduced["workspace"]


def validate_metadata(metadata: dict, stage: Path, specs: dict[str, dict], paths: dict[str, str]) -> None:
    locals_seen = {}
    for package in metadata["packages"]:
        if package.get("source") is None:
            name = package["name"]
            if (name not in specs or name in locals_seen or package["version"] != specs[name]["version"]
                    or Path(package["manifest_path"]).resolve() != (stage / paths[name] / "Cargo.toml").resolve()):
                raise archives.AuditError("Cargo resolved an unexpected local package")
            locals_seen[name] = package["id"]
        elif package["source"] != REGISTRY:
            raise archives.AuditError("Cargo resolved a non-crates.io external dependency")
        elif package["name"] in specs:
            raise archives.AuditError("Cargo resolved a second registry copy of a staged package")
    if set(locals_seen) != set(specs) or set(metadata["workspace_members"]) != set(locals_seen.values()):
        raise archives.AuditError("Cargo workspace does not contain exactly the publication set")


def check_lock(data: bytes, local: dict[str, dict], bundled: dict[str, dict]) -> None:
    seen = set()
    for package in tomllib.loads(data.decode()).get("package", []):
        name, version = package["name"], package["version"]
        if package.get("source") is None:
            if name not in local or version != local[name]["version"] or name in seen:
                raise archives.AuditError("unexpected local lockfile package")
            seen.add(name)
        else:
            if name in local:
                raise archives.AuditError("lockfile contains a second copy of a local package")
            if package["source"] != REGISTRY or not archives.HEX64.fullmatch(package.get("checksum", "")):
                raise archives.AuditError("external lock input is not checksummed crates.io source")
            if name in bundled and (version != bundled[name]["version"] or package["checksum"] != bundled[name]["archive_sha256"]):
                raise archives.AuditError(f"packaged lock uses a different bundled implementation: {name}")
    if seen != set(local):
        raise archives.AuditError("lockfile omits a local publication package")


def qualify(stage: Path, output: Path, family: str, specs: list[dict], workspace: dict,
            runner: list[str], jobs: int, timeout: int, report: dict) -> None:
    if runner not in (["cargo"], ["rch", "exec", "--", "cargo"]):
        raise archives.AuditError("runner must be cargo or rch exec -- cargo; source overrides are not accepted")
    if any(key.startswith(("FT_", "FJ_", "FW_", "FRANKEN_")) for key in os.environ):
        raise archives.AuditError("remove inherited kernel/behavior overrides before qualification")
    paths = FAMILIES[family][1]
    selected = {s["name"]: s for s in specs if s["name"] in paths}
    excluded = {"RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC", "RUSTDOC", "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER", "RUSTUP_TOOLCHAIN", "RUSTC_BOOTSTRAP"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("CARGO_") and k not in excluded}
    env.update(CARGO_HOME=str(output / "cargo-home"), CARGO_TARGET_DIR=str(output / "target"))
    baseline = inventory(stage)

    def run(label: str, *args: str) -> bytes:
        command = [*runner, *args]
        report["commands"].append(command)
        with (output / f"{label}.stdout").open("wb") as stdout, (output / f"{label}.stderr").open("wb") as stderr:
            subprocess.run(command, cwd=stage, env=env, stdout=stdout, stderr=stderr, check=True, timeout=timeout)
        return (output / f"{label}.stdout").read_bytes()

    run("cargo-version", "--version", "--verbose")
    lock_path = stage / "Cargo.lock"
    lock_args = ["--locked"] if lock_path.exists() else []
    metadata = json.loads(run("metadata", "metadata", "--format-version", "1", "--all-features", *lock_args))
    validate_metadata(metadata, stage, selected, paths)
    lock = lock_path.read_bytes()
    check_lock(lock, selected, {})
    after_resolution = inventory(stage)
    after_resolution.pop("Cargo.lock", None)
    before_resolution = dict(baseline)
    before_resolution.pop("Cargo.lock", None)
    if before_resolution != after_resolution:
        raise archives.AuditError("dependency resolution changed source inputs")
    baseline = inventory(stage)
    report.update(resolution="pass", lock_sha256=sha256(lock), replaying_lock=bool(lock_args))
    expected_outputs = test_outputs.output_paths(selected)
    output_roots = test_outputs.output_roots(selected)
    test_outputs.require_fresh_outputs(stage, baseline, expected_outputs, output_roots)
    report["unit_tests"] = "running"
    try:
        run("unit-tests", "test", "--workspace", "--all-features", "--locked", "--jobs", str(jobs))
    except (OSError, subprocess.SubprocessError):
        report["unit_tests"] = "failed"
        raise
    report["unit_tests"] = "pass"
    after_tests = inventory(stage)
    test_outputs.collect_outputs(
        stage, output, baseline, after_tests, expected_outputs, report, output_roots
    )
    # Freeze the evidence too: no additions or mutations are allowed while
    # packaging, including to the six reviewed test-generated JSON files.
    baseline = after_tests
    # One invocation is crucial: Cargo stages unpublished workspace dependencies
    # for one another. Serial per-package invocations recreate the publication deadlock.
    report["package_verification"] = "running"
    try:
        run("package", "package", "--workspace", "--all-features", "--locked", "--jobs", str(jobs))
    except (OSError, subprocess.SubprocessError):
        report["package_verification"] = "failed"
        raise
    if inventory(stage) != baseline:
        raise archives.AuditError("tests or packaging changed the locked source inputs")
    records, packaged_files = {}, {}
    for name, spec in selected.items():
        path = output / "target/package" / f"{name}-{spec['version']}.crate"
        data = path.read_bytes()
        # This checksum is local, not a claim about the live crates.io index.
        record = archives.verify_archive(spec, {"cksum": sha256(data)}, data)
        files = archives.archive_files(data, name, spec["version"])
        manifest = tomllib.loads(files["Cargo.toml"][1].decode())
        original = tomllib.loads((stage / paths[name] / "Cargo.toml").read_text())
        if effective_dependencies(manifest, {}) != effective_dependencies(original, workspace):
            raise archives.AuditError(f"packaging changed dependencies/features: {name}")
        for feature, members in original.get("features", {}).items():
            if set(manifest.get("features", {}).get(feature, [])) != set(members):
                raise archives.AuditError(f"packaging changed feature {name}/{feature}")
        lib_name = original.get("lib", {}).get("name", name.replace("-", "_"))
        if manifest.get("lib", {}).get("name", name.replace("-", "_")) != lib_name:
            raise archives.AuditError(f"packaging changed Rust library identity: {name}")
        record["artifact"] = str(path.relative_to(output))
        records[name], packaged_files[name] = record, files
    for name, files in packaged_files.items():
        check_lock(files["Cargo.lock"][1], {name: selected[name]}, records)
        locked = {p["name"] for p in tomllib.loads(files["Cargo.lock"][1].decode())["package"]}
        manifest = tomllib.loads(files["Cargo.toml"][1].decode())
        for dep in effective_dependencies(manifest, {}).values():
            if dep["package"] in selected and dep["package"] not in locked:
                raise archives.AuditError(f"packaged lock omits a bundled dependency: {name} -> {dep['package']}")
        records[name]["packaged_lock_sha256"] = sha256(files["Cargo.lock"][1])
    report.update(status="dependency_packages_verified", package_verification="pass", artifacts=records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("registry-source-contract.json"))
    parser.add_argument("--toolchain-file", type=Path, default=Path(__file__).resolve().parents[1] / "rust-toolchain.toml")
    parser.add_argument("--without-workspace-overrides", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lock", type=Path, help="replay this family's previously generated lock")
    parser.add_argument("--cargo-command", default="rch exec -- cargo")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600, help="timeout per Cargo command in seconds")
    args = parser.parse_args(argv)
    report = {"schema": 1, "status": "not_prepared", "published": False,
              "registry_availability": "not_checked", "native_metal": "not_certified",
              "resolution": "not_run", "unit_tests": "not_run", "test_input_integrity": "not_run",
              "package_verification": "not_run", "commands": []}
    owns_output = False
    output, repo = args.output.resolve(), args.repo.resolve()
    try:
        if args.jobs < 1 or args.timeout < 1:
            raise archives.AuditError("jobs and timeout must be positive")
        if output == repo or repo in output.parents:
            raise archives.AuditError("output must be outside the source checkout")
        for parent in (output, *output.parents):
            if any((parent / ".cargo" / name).exists() for name in ("config", "config.toml")):
                raise archives.AuditError("output would inherit ancestor Cargo configuration")
        specs = archives.load_contract(args.contract)
        toolchain = args.toolchain_file.read_bytes()
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        stage, workspace = prepare(repo, output, args.family, specs, toolchain,
                                   args.expected_revision, args.without_workspace_overrides, report)
        report["contract_sha256"] = sha256(args.contract.read_bytes())
        if args.lock:
            selected = {s["name"]: s for s in specs if s["name"] in FAMILIES[args.family][1]}
            check_lock(args.lock.read_bytes(), selected, {})
            (stage / "Cargo.lock").write_bytes(args.lock.read_bytes())
        if not args.prepare_only:
            qualify(stage, output, args.family, specs, workspace, shlex.split(args.cargo_command), args.jobs, args.timeout, report)
    except (archives.AuditError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report.update(status="failed", error=str(exc))
    if owns_output:
        (output / "dependency-package-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
