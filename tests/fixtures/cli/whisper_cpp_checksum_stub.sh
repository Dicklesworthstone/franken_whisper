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
  echo "missing -of or -f" >&2
  exit 2
fi
sum="$(cksum < "${input}")"
sum="${sum%% *}"
printf '{"text":"clip %s","language":"en","segments":[{"start":0.0,"end":0.1,"text":"clip","confidence":0.8},{"start":0.1,"end":0.2,"text":"%s","confidence":0.6}]}\n' "${sum}" "${sum}" > "${out_prefix}.json"
