#!/bin/bash
set -euo pipefail
out_prefix=""
split_on_word=0
max_segment_length=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    -sow)
      split_on_word=1
      shift
      ;;
    -ml)
      max_segment_length="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" || "${split_on_word}" != "1" || "${max_segment_length}" != "100" ]]; then
  echo "missing split-on-word zero-duration regression contract" >&2
  exit 42
fi
printf '%s\n' '{"result":{"language":"en"},"transcription":[{"offsets":{"from":0,"to":1000},"text":" first","confidence":0.91},{"offsets":{"from":1000,"to":1000},"text":" zero-width","confidence":0.72},{"offsets":{"from":1500,"to":2000},"text":" third","confidence":0.83}]}' > "${out_prefix}.json"
