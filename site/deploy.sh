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

for f in "$STAGE"/index.html; do
  perl -pi -e "s/\@SITEV\@/$VERSION/g" "$f"
done

(cd "$STAGE" && wrangler pages deploy . --project-name franken-whisper --branch main --commit-dirty=true)
echo "deployed version $VERSION"
