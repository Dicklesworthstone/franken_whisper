#!/bin/bash
set -euo pipefail
out_prefix=""
tiny_diarize=0
json_output=0
max_context=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    -tdrz)
      tiny_diarize=1
      shift
      ;;
    --tdrz)
      echo "unsupported TinyDiarize flag spelling" >&2
      exit 41
      ;;
    -oj)
      json_output=1
      shift
      ;;
    -mc)
      max_context="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" || "${out_prefix##*/}" != "whispercpp_output" || "${tiny_diarize}" != "1" || "${json_output}" != "1" || "${max_context}" != "0" ]]; then
  echo "missing TinyDiarize JSON artifact contract" >&2
  exit 42
fi
printf '%s\n' '{"result":{"language":"en"},"transcription":[{"timestamps":{"from":"00:00:00,000","to":"00:00:00,500"},"offsets":{"from":0,"to":500},"text":" synthetic speaker one","speaker_turn_next":true},{"timestamps":{"from":"00:00:00,500","to":"00:00:01,000"},"offsets":{"from":500,"to":1000},"text":" synthetic speaker two","speaker_turn_next":false}]}' > "${out_prefix}.json"
