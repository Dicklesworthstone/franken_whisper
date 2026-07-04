#!/usr/bin/env bash
# whisper_cpp_ab.sh — authoritative same-worker A/B: franken_whisper native vs
# the original whisper.cpp `whisper-cli` (bd-zk43).
#
# Methodology (contention-robust, apples-to-apples):
#   * interleaved best-of-N per (model,clip): min over N SUCCESSFUL runs = the
#     least-contended sample, and alternating native/whisper.cpp equalizes
#     shared-host load across both.
#   * GREEDY decode on BOTH: native is greedy; whisper.cpp is forced `-bs 1 -bo 1`
#     (its default is beam-5/best-of-5, which would do ~5x the decode work — an
#     unfair comparison). no-timestamps on both (`--json` native / `-nt` wc).
#   * matched thread count (THREADS) so the compute budget is identical.
#   * FAILED runs (crash / missing model / no timing) are skipped, counted, and
#     never folded into the min; if a tool fails every run the ratio is suppressed
#     (a failed comparator must not masquerade as a fast one).
#
# Env:
#   NATIVE   franken_whisper binary        (default: release build in cargo target)
#   WC       whisper-cli binary            (default: legacy_whispercpp build)
#   THREADS  threads for both tools        (default: 16)
#   N        best-of-N samples             (default: 5)
#
# Usage:  whisper_cpp_ab.sh "MODEL|CLIP|LABEL" ["MODEL2|CLIP2|LABEL2" ...]
# Record every row in docs/NEGATIVE_EVIDENCE.md with git SHA / worker / load / SHAs.
set -u
NATIVE=${NATIVE:-target/release/franken_whisper}
WC=${WC:-legacy_whispercpp/whisper.cpp/build/bin/whisper-cli}
N=${N:-5}
THREADS=${THREADS:-16}
TMP=$(mktemp -d) || { echo "mktemp failed" >&2; exit 1; }
cleanup() { [ -n "${TMP:-}" ] && [ -d "$TMP" ] && rm -rf -- "$TMP"; }
trap cleanup EXIT

min() { python3 -c "import sys;print(min(float(sys.argv[1]),float(sys.argv[2])))" "$1" "$2"; }

# Each returns 0 + prints the elapsed seconds on success; non-zero + no output on
# failure (non-zero tool exit, or /usr/bin/time produced no parseable `real` line).
run_native() {
  local t
  RAYON_NUM_THREADS=$THREADS /usr/bin/time -p "$NATIVE" transcribe --input "$2" --model "$1" --json \
    >"$TMP/nat.json" 2>"$TMP/nat.t" || return 1
  t=$(grep -m1 '^real' "$TMP/nat.t" | awk '{print $2}')
  [ -n "$t" ] || return 1
  printf '%s' "$t"
}
run_wc() {
  local t
  /usr/bin/time -p "$WC" -m "$1" -f "$2" -t "$THREADS" -bs 1 -bo 1 -np -nt \
    >"$TMP/wc.txt" 2>"$TMP/wc.t" || return 1
  t=$(grep -m1 '^real' "$TMP/wc.t" | awk '{print $2}')
  [ -n "$t" ] || return 1
  printf '%s' "$t"
}

ab() {
  local model=$1 clip=$2 label=$3 i nt wt nb=99999 wb=99999 nok=0 wok=0
  for i in $(seq 1 "$N"); do
    if nt=$(run_native "$model" "$clip"); then nb=$(min "$nb" "$nt"); nok=$((nok + 1)); fi
    if wt=$(run_wc "$model" "$clip"); then wb=$(min "$wb" "$wt"); wok=$((wok + 1)); fi
  done
  if [ "$nok" -eq 0 ] || [ "$wok" -eq 0 ]; then
    printf '%-30s FAILED  native_ok=%s/%s wc_ok=%s/%s — ratio suppressed (check model/binary paths)\n' \
      "$label" "$nok" "$N" "$wok" "$N"
    return
  fi
  local ratio; ratio=$(python3 -c "print(f'{$nb/$wb:.2f}')")
  printf '%-30s native %7ss | whisper.cpp %7ss | native/wc = %sx | greedy t=%s best-of-%s (ok nat %s/%s wc %s/%s)\n' \
    "$label" "$nb" "$wb" "$ratio" "$THREADS" "$N" "$nok" "$N" "$wok" "$N"
}

echo "== franken_whisper native vs whisper.cpp  (best-of-$N, GREEDY both, threads=$THREADS) =="
echo "== $(date -u '+%Y-%m-%dT%H:%M:%SZ')  host=$(hostname)  load:$(uptime | grep -oE 'load average.*') =="
for spec in "$@"; do
  IFS='|' read -r model clip label <<< "$spec"
  ab "$model" "$clip" "$label"
done
