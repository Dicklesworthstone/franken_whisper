#!/usr/bin/env bash
# Builds fw-wasm into site/pkg for local serving or Pages deploy (bd-m2jm, W3).
#
# SERIAL build only, deliberately: no +atomics, no shared memory, no COOP/COEP
# requirement, runs everywhere including iOS WebKit. Threads are a separate,
# measured lever (see frankentts site/build.sh for the dual-build recipe and
# the iPhone shared-memory-grow kill that motivates it); the tiny.en engine
# must first prove itself serially.
#
# +simd128: std::simd in nn.rs lowers to real wasm SIMD (baseline in every
# supported browser: Chrome 91+, Firefox 89+, Safari 16.4+).
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-target}"

RUSTFLAGS="-C target-feature=+simd128 -C link-arg=--max-memory=4294967296" \
  cargo build --manifest-path fw-wasm/Cargo.toml \
  --target wasm32-unknown-unknown --release

wasm-bindgen "$TARGET_DIR/wasm32-unknown-unknown/release/fw_wasm.wasm" \
  --out-dir site/pkg --target web

# Size gate: the engine (without weights) must stay shippable. Raise only
# with a ledger entry, never to land a feature.
GZ=$(gzip -c site/pkg/fw_wasm_bg.wasm | wc -c | tr -d ' ')
echo "fw_wasm_bg.wasm gzip: ${GZ} bytes"
if [ "${GZ}" -gt 3145728 ]; then
  echo "SIZE GATE FAILED: gzip ${GZ} > 3 MiB budget" >&2
  exit 1
fi
