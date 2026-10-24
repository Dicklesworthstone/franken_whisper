#!/usr/bin/env python3
"""Check immutable registry archives against reviewed source trees (Python 3.11+).

This is an artifact-integrity gate, NOT a build/conformance certification. It
uses neither sibling checkouts nor Cargo's cache, patches, or source replacement.
No archive is extracted, no dependency is published, and no baseline is updated.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tomllib
from typing import Any, Callable
import urllib.error
import urllib.request

MAX_DOWNLOAD = 64 * 1024 * 1024
MAX_UNPACKED = 256 * 1024 * 1024
MAX_MEMBERS = 100_000
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class AuditError(ValueError):
    """A release input could not be established as the reviewed artifact."""


def index_path(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise AuditError(f"invalid crate name: {name!r}")
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def git_object(kind: str, data: bytes) -> bytes:
    # SHA-1 here is Git object identity, not archive transport authentication.
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).digest()


def source_tree(files: dict[str, tuple[int, bytes]], prefix: str = "src/") -> str:
    """Reconstruct Git's tree ID, including paths, file modes and all bytes."""
    tree: dict[str, Any] = {}
    for path, value in files.items():
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix):].split("/")
        node = tree
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise AuditError(f"file/directory collision: {path}")
            node = child
        if parts[-1] in node:
            raise AuditError(f"duplicate source path: {path}")
        node[parts[-1]] = value
    if not tree:
        raise AuditError(f"archive has no {prefix} source files")

    def digest(node: dict[str, Any]) -> bytes:
        body = bytearray()
        for name, value in sorted(
            node.items(), key=lambda kv: (kv[0] + ("/" if isinstance(kv[1], dict) else "")).encode()
        ):
            if isinstance(value, dict):
                mode, oid = "40000", digest(value)
            else:
                permissions, data = value
                mode = "100755" if permissions & 0o111 else "100644"
                oid = git_object("blob", data)
            body.extend(mode.encode() + b" " + name.encode() + b"\0" + oid)
        return git_object("tree", bytes(body))

    return digest(tree).hex()


def archive_files(data: bytes, name: str, version: str) -> dict[str, tuple[int, bytes]]:
    """Read bounded regular files only; reject ambiguous/unsafe member names."""
    files: dict[str, tuple[int, bytes]] = {}
    seen: set[str] = set()
    total = 0
    root = f"{name}-{version}"
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for count, member in enumerate(archive, 1):
                if count > MAX_MEMBERS:
                    raise AuditError("archive has too many members")
                path = member.name.rstrip("/") if member.isdir() else member.name
                parts = path.split("/")
                if (not parts or parts[0] != root or "\\" in path
                        or any(p in ("", ".", "..") for p in parts)):
                    raise AuditError(f"unsafe archive member: {member.name!r}")
                if path in seen:
                    raise AuditError(f"duplicate archive member: {path}")
                seen.add(path)
                if member.isdir():
                    continue
                if not member.isfile() or len(parts) < 2:
                    raise AuditError(f"non-regular archive member: {path}")
                total += member.size
                if member.size < 0 or total > MAX_UNPACKED:
                    raise AuditError("archive exceeds unpacked size limit")
                stream = archive.extractfile(member)
                if stream is None:
                    raise AuditError(f"unreadable archive member: {path}")
                content = stream.read(member.size + 1)
                if len(content) != member.size:
                    raise AuditError(f"truncated archive member: {path}")
                files["/".join(parts[1:])] = (member.mode, content)
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise AuditError(f"invalid crate archive: {exc}") from exc
    return files


def registry_dependencies(manifest: dict[str, Any]) -> None:
    """Cargo's normalized published manifest must not depend on local inputs."""
    if manifest.get("patch") or manifest.get("replace"):
        raise AuditError("published manifest contains dependency overrides")
    sections = [manifest, *manifest.get("target", {}).values()]
    for section in sections:
        for kind in ("dependencies", "build-dependencies", "dev-dependencies"):
            for alias, dep in section.get(kind, {}).items():
                if not isinstance(dep, dict):
                    continue
                forbidden = {"path", "git", "workspace"} & dep.keys()
                if forbidden or dep.get("registry") or dep.get("registry-index"):
                    raise AuditError(f"non-registry dependency {kind}.{alias}: {dep}")


def index_record(body: bytes, name: str, version: str) -> dict[str, Any]:
    try:
        records = [json.loads(line) for line in body.splitlines() if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise AuditError(f"invalid index response for {name}") from exc
    matches = [r for r in records if isinstance(r, dict)
               and r.get("name") == name and r.get("vers") == version]
    if len(matches) != 1:
        raise AuditError(f"{name} {version} is missing or ambiguous in crates.io; publication required")
    record = matches[0]
    if record.get("yanked") is not False:
        raise AuditError(f"{name} {version} is yanked or has no valid yank status")
    if not isinstance(record.get("cksum"), str) or not HEX64.fullmatch(record["cksum"]):
        raise AuditError(f"{name} {version} has no valid registry checksum")
    return record


def verify_archive(spec: dict[str, Any], record: dict[str, Any], data: bytes) -> dict[str, str]:
    name, version = spec["name"], spec["version"]
    checksum = hashlib.sha256(data).hexdigest()
    if checksum != record["cksum"]:
        raise AuditError(f"{name} {version}: downloaded archive checksum differs from crates.io index")
    files = archive_files(data, name, version)
    if "Cargo.toml" not in files:
        raise AuditError(f"{name} {version}: archive has no Cargo.toml")
    try:
        manifest = tomllib.loads(files["Cargo.toml"][1].decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise AuditError(f"{name} {version}: invalid packaged Cargo.toml") from exc
    package = manifest.get("package", {})
    if package.get("name") != name or package.get("version") != version:
        raise AuditError(f"{name} {version}: packaged identity mismatch")
    if package.get("repository", "").removesuffix(".git").rstrip("/") != spec["repository"]:
        raise AuditError(f"{name} {version}: packaged repository mismatch")
    registry_dependencies(manifest)
    actual = source_tree(files)
    if actual != spec["src_tree"]:
        raise AuditError(f"{name} {version}: source mismatch; expected {spec['src_tree']}, got {actual}")
    # src/ alone cannot attest to a newly introduced build script or an external
    # library entry point. Any such input must have its own reviewed blob ID.
    extras = spec.get("extra_files", {})
    required = set(extras)
    build = package.get("build")
    if isinstance(build, str):
        required.add(build)
    elif build is not False and "build.rs" in files:
        required.add("build.rs")
    lib_path = manifest.get("lib", {}).get("path", "src/lib.rs")
    if not isinstance(lib_path, str) or ".." in lib_path.split("/"):
        raise AuditError(f"{name}: invalid library entry point")
    if not lib_path.startswith("src/"):
        required.add(lib_path)
    if lib_path not in files:
        raise AuditError(f"{name}: missing library entry point {lib_path}")
    for path in required:
        if path not in extras:
            raise AuditError(f"{name}: unreviewed build input {path}")
        if path not in files or git_object("blob", files[path][1]).hex() != extras[path]:
            raise AuditError(f"{name}: missing or mismatched extra input {path}")
    return {"name": name, "version": version, "archive_sha256": checksum, "src_tree": actual}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "franken-whisper-release-audit/1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(MAX_DOWNLOAD + 1)
            if len(data) > MAX_DOWNLOAD:
                raise AuditError(f"download exceeds size limit: {url}")
            return data
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise AuditError(f"registry artifact missing (not published): {url}") from exc
        raise AuditError(f"registry HTTP error {exc.code}: {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AuditError(f"registry unavailable; release readiness NOT established: {exc}") from exc


def load_contract(path: Path) -> list[dict[str, Any]]:
    try:
        contract = json.loads(path.read_text())
        specs = contract["packages"]
        if contract["schema"] != 1 or not isinstance(specs, list) or not specs:
            raise ValueError("expected schema 1 and a nonempty package list")
        names = set()
        for spec in specs:
            index_path(spec["name"])
            if spec["name"] in names:
                raise ValueError("duplicate package name")
            names.add(spec["name"])
            if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", spec["version"]):
                raise ValueError("expected exact version")
            for field in ("src_tree", "source_revision"):
                if not HEX40.fullmatch(spec[field]):
                    raise ValueError(f"invalid {field}")
            if not spec["repository"].startswith("https://github.com/"):
                raise ValueError("expected canonical GitHub repository")
            for blob in spec.get("extra_files", {}).values():
                if not HEX40.fullmatch(blob):
                    raise ValueError("invalid extra input blob")
        return specs
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AuditError(f"invalid source contract {path}: {exc}") from exc


def audit(specs: list[dict[str, Any]], get: Callable[[str], bytes] = fetch) -> dict[str, Any]:
    results, errors = [], []
    for spec in specs:
        name, version = spec["name"], spec["version"]
        try:
            record = index_record(get(f"https://index.crates.io/{index_path(name)}"), name, version)
            data = get(f"https://static.crates.io/crates/{name}/{name}-{version}.crate")
            results.append(verify_archive(spec, record, data))
        except AuditError as exc:
            errors.append(f"{name} {version}: {exc}")
    return {"schema": 1, "artifact_integrity": "pass" if not errors else "fail",
            "build_and_conformance": "not_run", "verified": results, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path,
                        default=Path(__file__).with_name("registry-source-contract.json"))
    args = parser.parse_args()
    try:
        report = audit(load_contract(args.contract))
    except AuditError as exc:
        print(f"registry archive audit: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["artifact_integrity"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
