#!/usr/bin/env bash
# Upload the converted Sortformer package to Hugging Face so the site's
# model proxy serves it from the HF CDN instead of GitHub releases (which
# 5xx under concurrent downloads). One-time; needs YOUR HF account:
#
#   uvx --from huggingface_hub hf auth login    # paste a WRITE token
#   ./scripts/upload-sortformer-to-hf.sh
#
# The proxy (site/functions/model/[[path]].js) already prefers
# https://huggingface.co/Dicklesworthstone/franken-whisper-sortformer and
# falls back to the GitHub release, so it flips over automatically the
# moment this upload lands. The site verifies sha256 pins client-side
# regardless of which upstream served the bytes.
set -euo pipefail

REPO="${1:-Dicklesworthstone/franken-whisper-sortformer}"
SRC="${FW_SF_DIR:-$HOME/.cache/franken_whisper/models/sortformer/sortformer-v2.1-f32-v1}"

for f in weights.safetensors conversion-receipt.json NOTICE.sortformer.txt NVIDIA-OPEN-MODEL-LICENSE.html; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f (run: fw models pull sortformer)" >&2; exit 1; }
done

# Pin check before anything leaves this machine.
want="487fa30cb0aa9799c77bd9985e6787962c3991fab8d4d576a4f1221d45298f6a"
got=$(shasum -a 256 "$SRC/weights.safetensors" | cut -d' ' -f1)
[ "$got" = "$want" ] || { echo "weights.safetensors sha mismatch: $got" >&2; exit 1; }

uvx --from huggingface_hub hf repo create "$REPO" --repo-type model 2>/dev/null || true
uvx --from huggingface_hub hf upload "$REPO" "$SRC/weights.safetensors" weights.safetensors
uvx --from huggingface_hub hf upload "$REPO" "$SRC/conversion-receipt.json" conversion-receipt.json
uvx --from huggingface_hub hf upload "$REPO" "$SRC/NOTICE.sortformer.txt" NOTICE.sortformer.txt
uvx --from huggingface_hub hf upload "$REPO" "$SRC/NVIDIA-OPEN-MODEL-LICENSE.html" NVIDIA-OPEN-MODEL-LICENSE.html

echo "uploaded. Verify through the site proxy:"
echo "  curl -sI https://franken-whisper.pages.dev/model/sortformer/weights.safetensors | head -3"
