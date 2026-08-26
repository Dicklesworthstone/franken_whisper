#!/usr/bin/env bash
# Hermetic yt-dlp stub for franken_whisper youtube tests.
#
# Honors the EXACT flag shapes that src/youtube/ytdlp.rs emits, so the real
# orchestration paths (probe / expand_playlist / fetch_metadata / download_audio)
# can be exercised with NO network access. Wire it in by pointing
# FRANKEN_WHISPER_YTDLP_BIN (probe) or by constructing a YtdlpInfo whose `path`
# is this script (every other fn takes &YtdlpInfo and never reads the env).
#
# Modes (selected by inspecting the argument vector):
#   --version                         -> prints a canned YYYY.MM.DD date line
#   --flat-playlist --dump-json --simulate URL
#                                     -> prints 2 canned flat-playlist JSON lines
#   -j --simulate ... (no --flat-playlist)
#                                     -> prints one canned full-metadata JSON object
#   -f bestaudio/best ... -o TEMPLATE --print after_move:filepath URL
#                                     -> copies $STUB_FIXTURE_WAV (default: the
#                                        tracked repo jfk_cut8.bin WAV) to
#                                        <dest_dir>/<id>.wav and
#                                        prints that path on its own stdout line
#
# Error injection via STUB_FAIL_MODE or an `fw_stub_fail=<mode>` URL query:
#   private -> stderr "ERROR: Private video. Sign in if you've been granted..." exit 1
#   geo     -> stderr "ERROR: ... This video is not available in your country." exit 1
#   429     -> stderr "ERROR: HTTP Error 429: Too Many Requests"                exit 1
#   exit1   -> generic stderr + exit 1
#
# Download-shape injection via URL query:
#   fw_stub_ext=flv -> materialize and print the exact-id artifact with a .flv suffix
#
# Override knobs (env):
#   STUB_VERSION        version string printed for --version   (default 2025.01.01)
#   STUB_FIXTURE_WAV    source wav copied on download   (default tracked jfk_cut8.bin)
#   STUB_VIDEO_ID       id used for the download output name    (default dQw4w9WgXcQ)
#   STUB_LIVE_STATUS    live_status field in -j metadata        (default not_live)

set -u

# Profiling aid (measurement-only): when STUB_CALL_LOG is set, append one line
# per invocation recording the mode + URL so a profiler can count yt-dlp calls.
if [ -n "${STUB_CALL_LOG:-}" ]; then
  printf '%s\n' "$*" >> "$STUB_CALL_LOG"
fi

# ---- error injection ------------------------------------------------------
# The URL-query form keeps parallel Rust tests hermetic without creating and
# immediately executing temporary wrapper scripts. Some Linux/network-backed
# filesystems can reject that pattern transiently with ETXTBSY even after the
# writer has closed the file.
FAIL_MODE="${STUB_FAIL_MODE:-}"
OUTPUT_EXT="wav"
for arg in "$@"; do
  case "$arg" in
    *fw_stub_fail=private*) [ -n "$FAIL_MODE" ] || FAIL_MODE="private" ;;
    *fw_stub_fail=geo*)     [ -n "$FAIL_MODE" ] || FAIL_MODE="geo" ;;
    *fw_stub_fail=429*)     [ -n "$FAIL_MODE" ] || FAIL_MODE="429" ;;
    *fw_stub_fail=exit1*)   [ -n "$FAIL_MODE" ] || FAIL_MODE="exit1" ;;
    *fw_stub_ext=flv*)      OUTPUT_EXT="flv" ;;
  esac
done

case "$FAIL_MODE" in
  private)
    echo "ERROR: [youtube] PRIVATE_ID: Private video. Sign in if you've been granted access to this video" >&2
    exit 1
    ;;
  geo)
    echo "ERROR: [youtube] GEO_ID: The uploader has not made this video available in your country" >&2
    exit 1
    ;;
  429)
    echo "ERROR: [youtube] RATE_ID: HTTP Error 429: Too Many Requests" >&2
    exit 1
    ;;
  exit1)
    echo "ERROR: something went wrong" >&2
    exit 1
    ;;
esac

# ---- mode detection -------------------------------------------------------
WANT_VERSION=0
WANT_FLAT=0
WANT_J=0
WANT_SIMULATE=0
WANT_NO_SIMULATE=0
WANT_DOWNLOAD=0
WANT_SEARCH=0
DEST_TEMPLATE=""
NEXT_IS_OUTPUT=0
OPTIONS_DONE=0

LAST_URL=""
for arg in "$@"; do
  if [ "$NEXT_IS_OUTPUT" -eq 1 ]; then
    DEST_TEMPLATE="$arg"
    NEXT_IS_OUTPUT=0
    continue
  fi
  if [ "$OPTIONS_DONE" -eq 1 ]; then
    case "$arg" in
      ytsearch*) WANT_SEARCH=1 ;;
      *)         LAST_URL="$arg" ;;
    esac
    continue
  fi
  case "$arg" in
    --)               OPTIONS_DONE=1 ;;
    --version)        WANT_VERSION=1 ;;
    --flat-playlist)  WANT_FLAT=1 ;;
    -j)               WANT_J=1 ;;
    --simulate)       WANT_SIMULATE=1 ;;
    --no-simulate)    WANT_NO_SIMULATE=1 ;;
    ytsearch*)        WANT_SEARCH=1 ;;
    -f)               WANT_DOWNLOAD=1 ;;  # format selection only appears on downloads
    -o)               NEXT_IS_OUTPUT=1 ;;
    http*)            LAST_URL="$arg" ;;  # remember the URL for id derivation
  esac
done

require_simulation() {
  if [ "$WANT_SIMULATE" -ne 1 ] || [ "$WANT_NO_SIMULATE" -ne 0 ]; then
    echo "ERROR: metadata-only operation must use --simulate and must not use --no-simulate" >&2
    exit 2
  fi
}

# Derive a stable video id from a watch URL so multi-URL runs stay distinct
# and self-consistent. Falls back to STUB_VIDEO_ID for non-watch inputs.
url_to_id() {
  local u="$1" id=""
  u="${u%%#*}"
  case "$u" in
    *watch?v=*)   id="${u#*watch?v=}"; id="${id%%&*}" ;;
    *youtu.be/*)  id="${u#*youtu.be/}"; id="${id%%[?&]*}" ;;
    */shorts/*)   id="${u#*/shorts/}"; id="${id%%[?&]*}" ;;
    *) id="" ;;
  esac
  [ -n "$id" ] && printf '%s' "$id" || printf '%s' "${STUB_VIDEO_ID:-dQw4w9WgXcQ}"
}

# ---- --version ------------------------------------------------------------
if [ "$WANT_VERSION" -eq 1 ]; then
  echo "${STUB_VERSION:-2025.01.01}"
  exit 0
fi

# ---- catalog search (bd-m7fv) ---------------------------------------------
if [ "$WANT_SEARCH" -eq 1 ]; then
  require_simulation
  if [ "$WANT_FLAT" -eq 1 ]; then
    echo '{"id":"srchflat0001","title":"Flat Search Hit One","url":"https://www.youtube.com/watch?v=srchflat0001","duration":95.0}'
    echo '{"id":"srchflat0002","title":"Flat Search Hit Two","webpage_url":"https://www.youtube.com/watch?v=srchflat0002","duration":150}'
  else
    # Third line repeats the first id: exercises the caller's dedup-by-id.
    cat <<'EOF'
{"id":"srchenr0001","title":"Enriched Search Hit One","channel":"Search Channel","view_count":4242,"upload_date":"20250301","duration":187.5,"url":"https://www.youtube.com/watch?v=srchenr0001"}
{"id":"srchenr0002","title":"Enriched Search Hit Two","channel":"Other Channel","duration":99.0,"webpage_url":"https://www.youtube.com/watch?v=srchenr0002"}
{"id":"srchenr0001","title":"Enriched Search Hit One (repeat)","url":"https://www.youtube.com/watch?v=srchenr0001"}
EOF
  fi
  exit 0
fi

# ---- playlist expansion ---------------------------------------------------
if [ "$WANT_FLAT" -eq 1 ]; then
  require_simulation
  echo '{"id":"vid000000001","title":"First Playlist Entry","url":"https://www.youtube.com/watch?v=vid000000001","duration":61.0}'
  # Second line intentionally uses webpage_url instead of url (fallback path)
  # and an integer duration to exercise numeric coercion.
  echo '{"id":"vid000000002","title":"Second Playlist Entry","webpage_url":"https://www.youtube.com/watch?v=vid000000002","duration":123}'
  exit 0
fi

# ---- full metadata --------------------------------------------------------
if [ "$WANT_J" -eq 1 ] && [ "$WANT_DOWNLOAD" -eq 0 ]; then
  require_simulation
  MID="$(url_to_id "$LAST_URL")"
  cat <<EOF
{"id":"$MID","title":"Stub Title $MID","channel":"Stub Channel","uploader":"Stub Uploader","upload_date":"20240115","duration":212.0,"webpage_url":"https://www.youtube.com/watch?v=$MID","description":"A canned description.","availability":"public","live_status":"${STUB_LIVE_STATUS:-not_live}"}
EOF
  exit 0
fi

# ---- download -------------------------------------------------------------
if [ "$WANT_DOWNLOAD" -eq 1 ]; then
  VIDEO_ID="$(url_to_id "$LAST_URL")"

  # Resolve the destination directory from the -o template
  # (e.g. /tmp/xxx/%(id)s.%(ext)s -> /tmp/xxx) and materialize <id>.wav there.
  if [ -z "$DEST_TEMPLATE" ]; then
    echo "ERROR: stub download invoked without -o template" >&2
    exit 1
  fi
  DEST_DIR="$(dirname "$DEST_TEMPLATE")"
  mkdir -p "$DEST_DIR"

  SRC_WAV="${STUB_FIXTURE_WAV:-}"
  if [ -z "$SRC_WAV" ]; then
    # The .bin suffix bypasses the fixture-size ignore rule; the bytes are a
    # valid RIFF/WAVE file tracked in Git and therefore present on RCH workers.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC_WAV="$SCRIPT_DIR/../native/jfk_cut8.bin"
  fi
  if [ ! -f "$SRC_WAV" ]; then
    echo "ERROR: stub fixture wav not found at $SRC_WAV" >&2
    exit 1
  fi

  OUT_PATH="$DEST_DIR/$VIDEO_ID.$OUTPUT_EXT"
  cp "$SRC_WAV" "$OUT_PATH"

  # Emulate yt-dlp's `--print after_move:filepath`: print the final path on
  # its own line. A leading noise line ensures the parser picks the LAST
  # non-empty line that is an existing path.
  echo "[download] Destination: $OUT_PATH"
  echo "$OUT_PATH"
  exit 0
fi

echo "ERROR: stub invoked with unrecognized argument shape: $*" >&2
exit 2
