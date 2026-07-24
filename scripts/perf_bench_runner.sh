#!/usr/bin/env bash
# Cargo target runner used by strict-RCH performance benches. It records the
# benchmark on the selected worker, then emits the worker identity, exact binary
# hash, and flat self-time table into the same RCH invocation's stdout.

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: perf_bench_runner.sh <benchmark-binary> [args...]" >&2
    exit 64
fi

if ! command -v perf >/dev/null 2>&1; then
    echo "perf_bench_runner: remote worker has no perf; benchmark not executed" >&2
    exit 69
fi

perf_cmd=(perf)
if [[ -r /proc/sys/kernel/perf_event_paranoid ]] &&
   (( $(< /proc/sys/kernel/perf_event_paranoid) > 2 )); then
    if command -v sudo >/dev/null 2>&1 && sudo -n perf --version >/dev/null 2>&1; then
        perf_cmd=(sudo -n perf)
    else
        echo "perf_bench_runner: perf_event_paranoid blocks profiling and sudo -n perf is unavailable; benchmark not executed" >&2
        exit 77
    fi
fi

binary=$1
shift
perf_data="/tmp/fw-cod-fw-$(basename "$binary")-$$.data"

echo "PERF_BENCH_WORKER_BEGIN"
hostname
sha256sum -- "$binary"
echo "PERF_BENCH_WORKER_END"

"${perf_cmd[@]}" record -m 1 -F 499 -e cycles:u -o "$perf_data" -- "$binary" "$@"
status=$?

echo "PERF_BENCH_SELF_TIME_BEGIN"
"${perf_cmd[@]}" report \
    -i "$perf_data" \
    --stdio \
    --no-children \
    --show-nr-samples \
    --percent-limit 0.01 \
    --sort symbol
report_status=$?
echo "PERF_BENCH_SELF_TIME_END"
echo "PERF_BENCH_DATA=$perf_data"

if [[ $status -ne 0 ]]; then
    exit "$status"
fi
exit "$report_status"
