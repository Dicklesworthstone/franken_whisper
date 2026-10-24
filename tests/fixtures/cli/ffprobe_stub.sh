#!/bin/bash
set -euo pipefail
input=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -*)
      shift
      ;;
    *)
      input="$1"
      shift
      ;;
  esac
done
if [[ -n "${FRANKEN_WHISPER_TEST_FFPROBE_MARKER:-}" ]]; then
  printf '%s\n' "$input" >> "${FRANKEN_WHISPER_TEST_FFPROBE_MARKER}"
fi
printf '0.25\n'
