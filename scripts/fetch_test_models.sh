#!/usr/bin/env bash
#
# fetch_test_models.sh — provision the whisper ggml model used by FrankenWhisper's
# gated native-engine conformance / e2e test suites (bd-4slu).
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS FETCHES
# ─────────────────────────────────────────────────────────────────────────────
# A single model file:
#
#     ggml-tiny.en.bin  (~74 MB)
#       from https://huggingface.co/ggerganov/whisper.cpp
#       sha256 921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f
#
# It is installed into:
#
#     ${FRANKEN_WHISPER_TEST_MODEL_DIR:-$HOME/.cache/franken_whisper/test-models}
#
# which is exactly one of the directories the Rust resolver searches for a
# short model name (`find_model_file("tiny.en")` in src/native_engine/mod.rs).
# With the file present, the otherwise-skipped native-engine tests run for real:
#
#     cargo test --test native_engine_e2e
#     cargo test --test conformance_comparator_tests
#
# The audio fixture (tests/fixtures/native/jfk.wav) ships IN the repo, so there
# is nothing to download for it — this script only verifies it is present.
#
# FrankenWhisper itself NEVER downloads models at runtime (data never leaves the
# machine); model provisioning is this explicit, user-invoked, opt-in step.
#
# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
#   scripts/fetch_test_models.sh [--model NAME] [--force] [--dest DIR] [-h|--help]
#
#   --model NAME   which model to fetch (default: tiny.en, the conformance model).
#                  Also: large-v3-turbo (~1.6 GB) — for PERF benchmarking only
#                  (enc-int8 / SDPA-poly-exp / load-cap sweeps need a larger
#                  n_state than tiny.en; NOT required for any test).
#   --force        re-download even if a valid copy is already present
#   --dest DIR     override the install dir (default below)
#   -h | --help    show this help and exit
#
#   Env:
#     FRANKEN_WHISPER_TEST_MODEL_DIR   default install dir override
#
# The script is idempotent: a present file whose sha256 already matches the pin
# is left untouched (and reported OK); only a missing or corrupt file triggers a
# download. Re-running it is always safe.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────

DEST="${FRANKEN_WHISPER_TEST_MODEL_DIR:-$HOME/.cache/franken_whisper/test-models}"
FORCE=0
MODEL_NAME="tiny.en"   # default: the conformance/e2e model (--model overrides)

# Model registry: short name -> "FILE|URL|SHA256". tiny.en is the default the gated
# native-engine test suites need; larger entries are for PERF benchmarking (they are
# NOT needed for tests) — e.g. enc-int8 / SDPA-poly-exp / load-cap sweeps that are only
# measurable at a larger n_state than tiny.en's 384. All from ggerganov/whisper.cpp; the
# sha256s are pinned from verified downloads (turbo: size 1624555275, transcribes jfk).
model_spec() {
  case "$1" in
    tiny.en)
      echo "ggml-tiny.en.bin|https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin|921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f" ;;
    large-v3-turbo|turbo)
      echo "ggml-large-v3-turbo.bin|https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin|1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69" ;;
    *)
      die "unknown model '$1' (known: tiny.en [default], large-v3-turbo)" ;;
  esac
}

# Populated from the registry once MODEL_NAME is known (see resolve_model).
MODEL_FILE=""
MODEL_URL=""
MODEL_SHA256=""
resolve_model() {
  local spec; spec="$(model_spec "$MODEL_NAME")"
  MODEL_FILE="${spec%%|*}"; spec="${spec#*|}"
  MODEL_URL="${spec%%|*}"
  MODEL_SHA256="${spec##*|}"
}

# In-repo audio fixture: present in the checkout, never downloaded.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
JFK_WAV="${REPO_ROOT}/tests/fixtures/native/jfk.wav"

# ── helpers ──────────────────────────────────────────────────────────────────

log()  { printf '[fetch-test-models] %s\n' "$*" >&2; }
die()  { printf '[fetch-test-models] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

have() { command -v "$1" >/dev/null 2>&1; }

# Compute sha256 of a file with whichever tool is available.
file_sha256() {
  local f="$1"
  if have sha256sum; then
    sha256sum "$f" | awk '{print $1}'
  elif have shasum; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    die "no sha256sum or shasum on PATH; cannot verify downloads"
  fi
}

# Download URL -> path using curl or wget.
download() {
  local url="$1" out="$2"
  log "downloading: $url"
  if have curl; then
    curl -fSL --retry 3 -o "$out" "$url"
  elif have wget; then
    wget -O "$out" "$url"
  else
    die "no curl or wget on PATH; cannot download"
  fi
}

# True (0) when dest exists and already matches the pinned sha256.
is_valid() {
  local path="$1"
  [ -f "$path" ] || return 1
  local got
  got="$(file_sha256 "$path")"
  [ "$got" = "$MODEL_SHA256" ]
}

# ── audio fixture check (no download) ────────────────────────────────────────

verify_fixture() {
  if [ -f "$JFK_WAV" ]; then
    log "audio fixture present: $JFK_WAV"
  else
    die "audio fixture missing: $JFK_WAV (it should ship in the repo checkout)"
  fi
}

# ── model fetch ──────────────────────────────────────────────────────────────

fetch_model() {
  local dest_path="$DEST/$MODEL_FILE"

  if [ "$FORCE" -ne 1 ] && is_valid "$dest_path"; then
    log "already present and valid (sha256 OK): $dest_path"
    return 0
  fi

  if [ -f "$dest_path" ] && [ "$FORCE" -ne 1 ]; then
    log "present but sha256 mismatch — re-downloading: $dest_path"
  fi

  mkdir -p "$DEST"
  local tmp
  tmp="$(mktemp "${DEST}/.${MODEL_FILE}.XXXXXX")"
  # shellcheck disable=SC2064
  trap "rm -f '$tmp'" EXIT
  download "$MODEL_URL" "$tmp"

  local got
  got="$(file_sha256 "$tmp")"
  [ "$got" = "$MODEL_SHA256" ] \
    || die "sha256 mismatch for $MODEL_FILE (got $got, want $MODEL_SHA256)"
  log "sha256 OK: $MODEL_FILE"

  mv -f "$tmp" "$dest_path"
  trap - EXIT
  log "installed: $dest_path"
}

# ── argument parsing ─────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --force)   FORCE=1; shift ;;
    --dest)    [ $# -ge 2 ] || die "--dest needs an argument"; DEST="${2%/}"; shift 2 ;;
    --model)   [ $# -ge 2 ] || die "--model needs an argument"; MODEL_NAME="$2"; shift 2 ;;
    *)         die "unknown argument: $1 (try --help)" ;;
  esac
done

resolve_model
log "install dir: $DEST"
log "model: $MODEL_NAME ($MODEL_FILE)"
verify_fixture
fetch_model
log "done. Native-engine gated tests will now run (model + jfk.wav present)."
