#!/bin/bash
set -euo pipefail
input=""
output=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-i" && $# -ge 2 ]]; then
    input="$2"
    shift 2
    continue
  fi
  output="$1"
  shift
done
if [[ -z "$output" ]]; then
  echo "stub ffmpeg missing required output arg" >&2
  exit 2
fi
if [[ -n "${FRANKEN_WHISPER_TEST_FFMPEG_MARKER:-}" ]]; then
  printf '%s\n' "$input -> $output" >> "${FRANKEN_WHISPER_TEST_FFMPEG_MARKER}"
fi
if [[ -n "${FRANKEN_WHISPER_TEST_FFMPEG_DELAY_SECONDS:-}" ]]; then
  /bin/sleep "${FRANKEN_WHISPER_TEST_FFMPEG_DELAY_SECONDS}"
fi
if [[ -n "${FRANKEN_WHISPER_TEST_FFMPEG_RELEASE_FILE:-}" ]]; then
  # Hold the conversion until the test creates this file (at most 60 s).
  for ((tick = 0; tick < 6000; tick++)); do
    [[ -e "${FRANKEN_WHISPER_TEST_FFMPEG_RELEASE_FILE}" ]] && break
    /bin/sleep 0.01
  done
fi
if [[ -n "$input" && -f "$input" ]]; then
  /bin/cp "$input" "$output"
else
  : > "$output"
fi
