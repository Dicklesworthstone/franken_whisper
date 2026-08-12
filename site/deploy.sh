#!/usr/bin/env bash
# Deploy the franken_whisper site to Cloudflare Pages with cache stamping.
#
# Every module URL carries ?v=@SITEV@ so the whole script graph rotates
# together on deploy — CDN zone cache TTL overrides can serve stale JS for
# hours otherwise (learned on frankentts: fresh HTML paired with stale JS).
set -euo pipefail
cd "$(dirname "$0")"

VERSION="$(git rev-parse --short HEAD)-$(date +%s)"
STAGE="$(mktemp -d /tmp/franken-whisper-site-deploy.XXXXXX)"
rsync -a --exclude deploy.sh --exclude .wrangler ./ "$STAGE/"

for f in "$STAGE"/index.html "$STAGE"/demo.html; do
  [ -f "$f" ] && perl -pi -e "s/\@SITEV\@/$VERSION/g" "$f"
done

# The demo requires the wasm package. Refuse to deploy a demo page that would
# 404 its own engine (pkg/ is gitignored build output — run site/build-wasm.sh).
if [ -f "$STAGE/demo.html" ] && [ ! -f "$STAGE/pkg/fw_wasm_bg.wasm" ]; then
  echo "demo.html present but site/pkg/fw_wasm_bg.wasm missing — run site/build-wasm.sh first" >&2
  exit 1
fi

(cd "$STAGE" && wrangler pages deploy . --project-name franken-whisper --branch main --commit-dirty=true)
echo "deployed version $VERSION"
