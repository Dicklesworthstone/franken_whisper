#!/bin/bash
set -euo pipefail
out_prefix=""
input=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    -f)
      input="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" || -z "${input}" ]]; then
  exit 0
fi
echo call >> "${FW_TEST_STUB_CALL_LOG}"
calls="$(wc -l < "${FW_TEST_STUB_CALL_LOG}")"
if [[ -n "${FW_TEST_STUB_BLOCK_ONLY_CALL:-}" ]]; then
  if [[ "${calls}" -eq "${FW_TEST_STUB_BLOCK_ONLY_CALL}" ]]; then
    sleep 60
  fi
elif [[ "${calls}" -ge 2 ]]; then
  sleep 60
fi
printf '{"text":"clip","language":"en","segments":[{"start":0.0,"end":0.1,"text":"clip","confidence":0.8}]}\n' > "${out_prefix}.json"
