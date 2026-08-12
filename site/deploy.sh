#!/usr/bin/env bash
# Deploy site/ to Cloudflare Pages with per-deploy cache stamping.
#
# Every file that CONTAINS a versioned URL must be stamped, not just the entry
# points — engine-worker.js imports ./pkg/fw_wasm.js?v=@SITEV@ and would
# otherwise ship the literal placeholder. The stamp defeats CDN zone TTL
# overrides that ignore origin no-cache headers (a measured 4-hour-stale-script
# incident on a sibling project).
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f pkg/fw_wasm_bg.wasm ]; then
  echo "site/pkg is missing — run site/build-wasm.sh first" >&2
  exit 1
fi
# Freshness gate: refuse to ship a wasm artifact older than the crate source.
newest_src=$(find ../fw-wasm/src ../src -name '*.rs' -newer pkg/fw_wasm_bg.wasm | head -1)
if [ -n "$newest_src" ]; then
  echo "STALE ARTIFACT: $newest_src is newer than site/pkg/fw_wasm_bg.wasm — run site/build-wasm.sh" >&2
  exit 1
fi

VERSION="$(git rev-parse --short HEAD)-$(date +%s)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# demo.html/demo.js/fw-worker.js are the superseded first-cut demo (kept in
# the repo for history, replaced by the playground on the index page).
rsync -a --exclude build-wasm.sh --exclude deploy.sh --exclude .wrangler \
  --exclude demo.js --exclude fw-worker.js . "$STAGE/"

for f in "$STAGE"/index.html "$STAGE"/app.js "$STAGE"/chrome.js \
         "$STAGE"/engine-worker.js "$STAGE"/loader.js "$STAGE"/demo.html; do
  [ -f "$f" ] && perl -pi -e "s/\@SITEV\@/$VERSION/g" "$f"
done

(cd "$STAGE" && wrangler pages deploy . --project-name franken-whisper --branch main --commit-dirty=true)
echo "deployed with stamp $VERSION"
