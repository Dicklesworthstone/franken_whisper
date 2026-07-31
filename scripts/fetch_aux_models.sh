#!/usr/bin/env bash
#
# fetch_aux_models.sh — provision auxiliary (non-whisper) neural model weights
# for FrankenWhisper's Epic B pipeline stages (neural diarization / separation).
#
# ─────────────────────────────────────────────────────────────────────────────
# STATUS / HONESTY (2026-07-31)
# ─────────────────────────────────────────────────────────────────────────────
# The first aux model we need is the SpeechBrain ECAPA-TDNN VoxCeleb speaker
# embedding network (consumed by bd-ohex). Its upstream repo
#
#     https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
#
# ships the weights ONLY as a PyTorch pickle checkpoint (`embedding_model.ckpt`,
# ~83 MB), NOT as safetensors. As of this writing there is NO trustworthy
# community-published safetensors conversion of this model (searched HF + web;
# the repo contents are: classifier.ckpt, embedding_model.ckpt,
# mean_var_norm_emb.ckpt, hyperparams.yaml, label_encoder.txt, config.json,
# example wavs). FrankenWhisper's Rust loader reads SAFETENSORS ONLY (it never
# unpickles, by design — arbitrary-code-execution risk), so this script does
# NOT yet install a usable safetensors file.
#
# The repository now freezes a deterministic local conversion profile and the
# exact converted package SHA-256, but the converted artifact is not published.
# Until it is hosted, `ecapa` prints pinned download, verification, and
# conversion instructions instead of installing anything. This is intentional:
# the runtime never unpickles the source checkpoint.
#
# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
#   scripts/fetch_aux_models.sh [--allow-unpinned] [--dest DIR] [MODEL ...]
#
#   MODELs (default: all known): ecapa, separation
#   --dest DIR         override the install dir (default below)
#   --allow-unpinned   install files whose sha256 is still TBD (UNSAFE; dev only)
#   -h | --help        show this help
#
# Install dir (mirrors the Rust resolver's aux search path):
#   ${FRANKEN_WHISPER_MODEL_DIR:-$HOME/.cache/franken_whisper/models}/aux/
#
# FrankenWhisper itself NEVER downloads models — provisioning is this explicit,
# user-invoked step (data never leaves the machine otherwise).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR="${FRANKEN_WHISPER_MODEL_DIR:-$HOME/.cache/franken_whisper/models}"
DEST="${DEFAULT_MODEL_DIR%/}/aux"
ALLOW_UNPINNED=0

# Upstream sources. ECAPA weights are pickle-only upstream; the safetensors URL
# is the converted artifact we publish (TBD).
ECAPA_REVISION="eac27266f68caa806381260bd44ace38b136c76a"
ECAPA_CKPT_URL="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/$ECAPA_REVISION/embedding_model.ckpt?download=true"
ECAPA_CKPT_SHA256="0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
ECAPA_SAFETENSORS_FILE="ecapa_tdnn_voxceleb.safetensors"
# Published converted artifact URL — fill in once hosted:
ECAPA_SAFETENSORS_URL="TBD"

SEPARATION_SAFETENSORS_FILE="separation_masknet.safetensors"
SEPARATION_SAFETENSORS_URL="TBD"

# ── sha256 pin table ─────────────────────────────────────────────────────────
# One entry per installable artifact (the safetensors files we publish). "TBD"
# means unpinned: the script REFUSES to install it unless --allow-unpinned.
# Format: FILENAME<space>SHA256HEX
sha256_for() {
  case "$1" in
    "$ECAPA_SAFETENSORS_FILE")      echo "9276a840c52cdd2e9afb73cd87a38e15749e12bf494d3ca47b5bc162f237cbcc" ;;
    "$SEPARATION_SAFETENSORS_FILE") echo "TBD" ;;
    *) echo "" ;;
  esac
}

# ── helpers ──────────────────────────────────────────────────────────────────

log()  { printf '[fetch-aux] %s\n' "$*" >&2; }
die()  { printf '[fetch-aux] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'
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

# Install a published safetensors artifact: download to a temp file, verify the
# pinned sha256 (unless --allow-unpinned), then move into place.
install_pinned() {
  local file="$1" url="$2"
  local expected dest_path tmp
  expected="$(sha256_for "$file")"
  dest_path="$DEST/$file"

  if [ "$url" = "TBD" ]; then
    die "no published download URL for '$file' yet (see the conversion instructions printed above)"
  fi

  if [ -f "$dest_path" ]; then
    log "already present: $dest_path"
    if [ "$expected" != "TBD" ]; then
      local have_sum
      have_sum="$(file_sha256 "$dest_path")"
      [ "$have_sum" = "$expected" ] || die "existing $dest_path sha256 mismatch (got $have_sum, want $expected)"
      log "sha256 OK: $file"
    fi
    return 0
  fi

  if [ "$expected" = "TBD" ] && [ "$ALLOW_UNPINNED" -ne 1 ]; then
    die "refusing to install unpinned '$file' (sha256 is TBD); re-run with --allow-unpinned to override (UNSAFE)"
  fi

  mkdir -p "$DEST"
  tmp="$(mktemp "${DEST}/.${file}.XXXXXX")"
  # shellcheck disable=SC2064
  trap "rm -f '$tmp'" EXIT
  download "$url" "$tmp"

  if [ "$expected" != "TBD" ]; then
    local got
    got="$(file_sha256 "$tmp")"
    [ "$got" = "$expected" ] || die "sha256 mismatch for $file (got $got, want $expected)"
    log "sha256 OK: $file"
  else
    log "WARNING: installing UNPINNED $file (--allow-unpinned); integrity NOT verified"
  fi

  mv -f "$tmp" "$dest_path"
  trap - EXIT
  log "installed: $dest_path"
}

# ── per-model actions ────────────────────────────────────────────────────────

fetch_ecapa() {
  if [ "$ECAPA_SAFETENSORS_URL" != "TBD" ]; then
    install_pinned "$ECAPA_SAFETENSORS_FILE" "$ECAPA_SAFETENSORS_URL"
    return 0
  fi

  cat >&2 <<EOF
[fetch-aux] ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) has NO published
[fetch-aux] safetensors artifact yet, and FrankenWhisper will not unpickle the
[fetch-aux] upstream .ckpt (security). Nothing was downloaded for 'ecapa'.
[fetch-aux]
[fetch-aux] To produce the weights yourself (offline, one time):
[fetch-aux]   1. Download the upstream pickle checkpoint:
[fetch-aux]        curl -fSL -o embedding_model.ckpt \\
[fetch-aux]          "$ECAPA_CKPT_URL"
[fetch-aux]      Verify its SHA-256 before unpickling:
[fetch-aux]        printf '%s  %s\n' "$ECAPA_CKPT_SHA256" embedding_model.ckpt | shasum -a 256 -c -
[fetch-aux]   2. Convert it with the exact deterministic profile:
[fetch-aux]        uv run --isolated --python 3.13 \\
[fetch-aux]          --with numpy==2.2.6 --with torch==2.7.1 --with safetensors==0.5.3 \\
[fetch-aux]          python scripts/convert_to_safetensors.py \\
[fetch-aux]          embedding_model.ckpt \\
[fetch-aux]          "$DEST/$ECAPA_SAFETENSORS_FILE" \\
[fetch-aux]          --profile ecapa-tdnn-voxceleb-v1
[fetch-aux]   3. Require output SHA-256:
[fetch-aux]        9276a840c52cdd2e9afb73cd87a38e15749e12bf494d3ca47b5bc162f237cbcc
[fetch-aux]
[fetch-aux] The converted file lands at:
[fetch-aux]   $DEST/$ECAPA_SAFETENSORS_FILE
EOF
}

fetch_separation() {
  if [ "$SEPARATION_SAFETENSORS_URL" != "TBD" ]; then
    install_pinned "$SEPARATION_SAFETENSORS_FILE" "$SEPARATION_SAFETENSORS_URL"
    return 0
  fi
  log "separation mask model (bd-mmx3) is a PLACEHOLDER: no source pinned yet. Nothing fetched."
}

# ── argument parsing ─────────────────────────────────────────────────────────

MODELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --allow-unpinned) ALLOW_UNPINNED=1; shift ;;
    --dest) [ $# -ge 2 ] || die "--dest needs an argument"; DEST="${2%/}"; shift 2 ;;
    ecapa|separation) MODELS+=("$1"); shift ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

if [ "${#MODELS[@]}" -eq 0 ]; then
  MODELS=(ecapa separation)
fi

log "install dir: $DEST"
for model in "${MODELS[@]}"; do
  case "$model" in
    ecapa)      fetch_ecapa ;;
    separation) fetch_separation ;;
    *)          die "unknown model: $model" ;;
  esac
done

log "done."
