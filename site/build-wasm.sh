#!/usr/bin/env bash
# Builds the fw-wasm engine into site/pkg (serial) and site/pkg-threaded
# (shared memory + rayon Web Workers), for local serving or Pages deploy.
#
# Deliberately NOT wasm-pack: wasm-pack injects its own RUSTFLAGS, which
# clobbers the target-feature list. If `+atomics` does not reach std, LLD
# emits no `__wasm_init_tls` and wasm-bindgen's threading transform fails. We
# drive cargo + wasm-bindgen directly so the flags we write are the flags
# that link. (Recipe proven on franken_ocr's and frankentts's lanes.)
#
# WHY TWO MODULES AND NOT ONE: sharedness is a LINK-TIME property. A module
# whose memory is shared cannot instantiate without crossOriginIsolated, and
# iOS Safari kills tabs that GROW a shared memory past ~2 GB — fatal for a
# 2.4 GB engine. So iOS/WebKit and non-isolated contexts get the serial
# module; isolated desktop Blink gets the threaded one. engine-worker.js
# picks at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
SERIAL_WASM="$TARGET_DIR/wasm32-unknown-unknown/release/fw_wasm.wasm"
THREADED_TARGET_DIR="${TARGET_DIR}-threaded"
THREADED_WASM="$THREADED_TARGET_DIR/wasm32-unknown-unknown/release/fw_wasm.wasm"

# ── lane 1: serial (iOS/WebKit and any non-isolated context) ────────────────
echo "== cargo build (serial, +simd128) =="
RUSTFLAGS="-C target-feature=+simd128 -C link-arg=--max-memory=4294967296" \
  cargo build --manifest-path fw-wasm/Cargo.toml \
  --target wasm32-unknown-unknown --release

echo "== wasm-bindgen -> site/pkg =="
wasm-bindgen "$SERIAL_WASM" --out-dir site/pkg --target web
rm -f site/pkg/.gitignore

# ── lane 2: threaded ────────────────────────────────────────────────────────
# `-Z build-std=std,panic_abort` rebuilds std WITH `+atomics`; the prebuilt
# wasm32-unknown-unknown std is compiled without it, and mixing the two is
# exactly how `__wasm_init_tls` goes missing. Nightly-only (we already pin
# one). One `-C link-arg=` per export: rustc's link-arg takes exactly ONE
# argument, so packing several `--export=`s into one flag silently drops all
# but the first.
THREADED_RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals"
THREADED_RUSTFLAGS="$THREADED_RUSTFLAGS -C link-arg=--shared-memory"
THREADED_RUSTFLAGS="$THREADED_RUSTFLAGS -C link-arg=--import-memory"
THREADED_RUSTFLAGS="$THREADED_RUSTFLAGS -C link-arg=--max-memory=4294967296"
for sym in __heap_base __tls_base __tls_size __tls_align __wasm_init_tls; do
  THREADED_RUSTFLAGS="$THREADED_RUSTFLAGS -C link-arg=--export=$sym"
done

echo "== cargo build (threaded, +simd128,+atomics; -Z build-std) =="
RUSTFLAGS="$THREADED_RUSTFLAGS" \
  cargo build --manifest-path fw-wasm/Cargo.toml --features threads \
  --target wasm32-unknown-unknown --release \
  --target-dir "$THREADED_TARGET_DIR" \
  -Z build-std=std,panic_abort

# Fail LOUDLY here rather than let wasm-bindgen emit its famously unhelpful
# "failed to find `__wasm_init_tls`". `node` reads the export section itself,
# so this is the module's own answer, not a flag we hoped took effect.
echo "== threaded link check (the five TLS exports) =="
node -e '
const fs = require("fs");
const mod = new WebAssembly.Module(fs.readFileSync(process.argv[1]));
const have = new Set(WebAssembly.Module.exports(mod).map((e) => e.name));
const want = ["__heap_base", "__tls_base", "__tls_size", "__tls_align", "__wasm_init_tls"];
const missing = want.filter((s) => !have.has(s));
if (missing.length) {
  console.error(`threaded link is missing exports: ${missing.join(", ")} — refusing to ship it`);
  process.exit(1);
}
console.log(`exports present: ${want.join(" ")}`);
' "$THREADED_WASM"

echo "== wasm-bindgen -> site/pkg-threaded =="
wasm-bindgen "$THREADED_WASM" --out-dir site/pkg-threaded --target web
rm -f site/pkg-threaded/.gitignore

echo "== shipped sizes =="
for dir in site/pkg site/pkg-threaded; do
  raw=$(wc -c <"$dir/fw_wasm_bg.wasm")
  gz=$(gzip -c "$dir/fw_wasm_bg.wasm" | wc -c | tr -d ' ')
  echo "$dir: raw $raw bytes, gzip $gz bytes"
  # Size gate: raise only with a ledger entry, never to land a feature.
  if [ "$gz" -gt 3145728 ]; then
    echo "SIZE GATE FAILED: $dir gzip ${gz} > 3 MiB budget" >&2
    exit 1
  fi
done
