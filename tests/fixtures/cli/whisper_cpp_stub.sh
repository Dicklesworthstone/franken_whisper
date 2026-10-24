#!/bin/bash
set -euo pipefail
out_prefix=""
model=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    -m)
      model="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" ]]; then
  echo "missing -of output prefix" >&2
  exit 2
fi
if [[ -n "${FW_TEST_STUB_LOG:-}" ]]; then
  printf '%s\t%s\n' "${model}" "${out_prefix}" >> "${FW_TEST_STUB_LOG}"
fi
printf '%s\n' '{"text":"stub transcript","language":"en","segments":[{"start":0.0,"end":0.5,"text":"stub transcript","speaker":"SPEAKER_00","confidence":0.9}]}' > "${out_prefix}.json"
