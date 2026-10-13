#!/usr/bin/env bash
# Provision or verify the exact sibling commits required by Cargo path
# dependencies. Existing sibling directories are never reset, cleaned, or
# overwritten: a mismatch fails closed with an actionable error.
set -euo pipefail

# v0.11.0 pins: frankensqlite v0.4.12, frankentui v0.9.0 and frankentts
# v0.1.11 are their release tags. frankentorch and frankenjax have no release
# tags: frankentorch is main as of 2026-10-10 (the Metal command-buffer fixes
# from c5359dc5 that the encoder needs), frankenjax is unchanged from v0.10.0.
FRANKENSQLITE_REVISION="26aa19afcbd5afdb01658a10bb33417ad0b513be"
FRANKENTORCH_REVISION="39b90cf7a90d7ecb8ae6f1637a6e287855182aeb"
FRANKENTUI_REVISION="befbd2471a1ec4a6b8ce0a0df32116c317cca9db"
FRANKENTTS_REVISION="47b2dd1ca79101653fd58b29e0f1b3d137a14454"
FRANKENJAX_REVISION="ec74b7b24a7935683f06fbd88129ce3a7cf9c7fc"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode="checkout"
if [ "${1:-}" = "--verify-only" ]; then
    mode="verify"
    shift
fi
if [ "$#" -gt 1 ]; then
    echo "usage: $0 [--verify-only] [parent-directory]" >&2
    exit 2
fi
parent=${1:-$(dirname "$repo_root")}

require_sibling() {
    local name="$1" remote="$2" revision="$3"
    local target="$parent/$name"
    if [ ! -e "$target" ]; then
        if [ "$mode" = "verify" ]; then
            echo "missing required sibling checkout: $target" >&2
            exit 1
        fi
        git clone --filter=blob:none --no-checkout "$remote" "$target"
        git -C "$target" fetch --depth 1 origin "$revision"
        git -C "$target" checkout --detach "$revision"
    fi

    [ -d "$target/.git" ] || {
        echo "required sibling path is not a git checkout: $target" >&2
        exit 1
    }
    local actual
    actual=$(git -C "$target" rev-parse HEAD)
    [ "$actual" = "$revision" ] || {
        echo "$name is at $actual but release builds require $revision" >&2
        echo "Use a separate clean parent directory; this script will not alter an existing checkout." >&2
        exit 1
    }
    if [ -n "$(git -C "$target" status --porcelain=v1)" ]; then
        echo "$name has tracked or untracked changes; release sibling inputs must be clean" >&2
        exit 1
    fi
    printf '%s %s\n' "$name" "$revision"
}

command -v git >/dev/null 2>&1 || {
    echo "git is required to prepare release sibling crates" >&2
    exit 1
}
if [ "$mode" = "checkout" ]; then
    mkdir -p "$parent"
elif [ ! -d "$parent" ]; then
    echo "sibling parent directory does not exist: $parent" >&2
    exit 1
fi

require_sibling \
    "frankensqlite" \
    "https://github.com/Dicklesworthstone/frankensqlite.git" \
    "$FRANKENSQLITE_REVISION"
require_sibling \
    "frankentorch" \
    "https://github.com/Dicklesworthstone/frankentorch.git" \
    "$FRANKENTORCH_REVISION"
require_sibling \
    "frankentui" \
    "https://github.com/Dicklesworthstone/frankentui.git" \
    "$FRANKENTUI_REVISION"
require_sibling \
    "frankentts" \
    "https://github.com/Dicklesworthstone/franken_tts.git" \
    "$FRANKENTTS_REVISION"
require_sibling \
    "frankenjax" \
    "https://github.com/Dicklesworthstone/frankenjax.git" \
    "$FRANKENJAX_REVISION"
