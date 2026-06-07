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
#   --flat-playlist --dump-json URL   -> prints 2 canned flat-playlist JSON lines
#   -j ... (no --flat-playlist)       -> prints one canned full-metadata JSON object
#   -f ba ... -o TEMPLATE --print after_move:filepath URL
#                                     -> copies $STUB_FIXTURE_WAV (default: the
#                                        repo jfk.wav) to <dest_dir>/<id>.wav and
#                                        prints that path on its own stdout line
#
# Error injection via STUB_FAIL_MODE:
#   private -> stderr "ERROR: Private video. Sign in if you've been granted..." exit 1
#   geo     -> stderr "ERROR: ... This video is not available in your country." exit 1
#   429     -> stderr "ERROR: HTTP Error 429: Too Many Requests"                exit 1
#   exit1   -> generic stderr + exit 1
#
# Override knobs (env):
#   STUB_VERSION        version string printed for --version   (default 2025.01.01)
#   STUB_FIXTURE_WAV    source wav copied on download          (default repo jfk.wav)
#   STUB_VIDEO_ID       id used for the download output name    (default dQw4w9WgXcQ)
#   STUB_LIVE_STATUS    live_status field in -j metadata        (default not_live)

set -u

# ---- error injection ------------------------------------------------------
case "${STUB_FAIL_MODE:-}" in
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
WANT_DOWNLOAD=0
DEST_TEMPLATE=""
NEXT_IS_OUTPUT=0

LAST_URL=""
for arg in "$@"; do
  if [ "$NEXT_IS_OUTPUT" -eq 1 ]; then
    DEST_TEMPLATE="$arg"
    NEXT_IS_OUTPUT=0
    continue
  fi
  case "$arg" in
    --version)        WANT_VERSION=1 ;;
    --flat-playlist)  WANT_FLAT=1 ;;
    -j)               WANT_J=1 ;;
    -f)               WANT_DOWNLOAD=1 ;;  # `-f ba` only appears on the download path
    -o)               NEXT_IS_OUTPUT=1 ;;
    http*)            LAST_URL="$arg" ;;  # remember the URL for id derivation
  esac
done

# Derive a stable video id from a watch URL so multi-URL runs stay distinct
# and self-consistent. Falls back to STUB_VIDEO_ID for non-watch inputs.
url_to_id() {
  local u="$1" id=""
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

# ---- playlist expansion ---------------------------------------------------
if [ "$WANT_FLAT" -eq 1 ]; then
  echo '{"id":"vid000000001","title":"First Playlist Entry","url":"https://www.youtube.com/watch?v=vid000000001","duration":61.0}'
  # Second line intentionally uses webpage_url instead of url (fallback path)
  # and an integer duration to exercise numeric coercion.
  echo '{"id":"vid000000002","title":"Second Playlist Entry","webpage_url":"https://www.youtube.com/watch?v=vid000000002","duration":123}'
  exit 0
fi

# ---- full metadata --------------------------------------------------------
if [ "$WANT_J" -eq 1 ] && [ "$WANT_DOWNLOAD" -eq 0 ]; then
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
    # Default: jfk.wav relative to this script (tests/fixtures/native/jfk.wav).
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SRC_WAV="$SCRIPT_DIR/../native/jfk.wav"
  fi
  if [ ! -f "$SRC_WAV" ]; then
    echo "ERROR: stub fixture wav not found at $SRC_WAV" >&2
    exit 1
  fi

  OUT_PATH="$DEST_DIR/$VIDEO_ID.wav"
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
