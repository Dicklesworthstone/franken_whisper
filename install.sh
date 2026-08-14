#!/usr/bin/env bash
#
# franken_whisper installer — cross-platform binary installer
#
# Agent-first Rust ASR orchestrator with ffmpeg normalization and
# frankensqlite-backed persistence.
#
# One-liner install (with cache buster):
#   curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh?$(date +%s)" | bash
#
# Or without cache buster:
#   curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh | bash
#
# Options:
#   --version vX.Y.Z   Install specific version (default: latest)
#   --dest DIR         Install to DIR (default: ~/.local/bin)
#   --system           Install to /usr/local/bin (requires sudo)
#   --easy-mode        Auto-update PATH in shell rc files
#   --verify           Run self-test after install
#   --no-pull          Skip automatic native Whisper + diarization model download
#   --force            Reinstall even if the same version is already present
#   --artifact-url URL Use a custom release artifact URL
#   --checksum SHA     Provide expected SHA256 checksum
#   --offline TARBALL  Install from a local archive (airgap); verifies a
#                      sibling .sha256 if present
#   --from-source      Build from source instead of downloading a binary
#   --no-verify        Skip checksum verification (testing only)
#   --quiet            Suppress non-error output
#   --no-gum           Disable gum formatting even if available
#   --uninstall        Remove franken_whisper and clean up
#   --help             Show this help
#
# Environment:
#   HTTP_PROXY / HTTPS_PROXY   Honored on every download
#   FW_INSTALL_DIR             Override default install directory
#   FW_INSTALL_PROMPT_TIMEOUT  Model prompt timeout in seconds (default: 120)
#   FRANKEN_WHISPER_MODEL_DIR  Override the model cache root
#   VERSION                    Override version to install
#
# Platforms (prebuilt binaries — 5 release targets):
#   Linux x86_64          franken_whisper-X.Y.Z-linux_amd64.tar.gz
#   Linux aarch64         franken_whisper-X.Y.Z-linux_arm64.tar.gz
#   macOS x86_64 (Intel)  franken_whisper-X.Y.Z-darwin_amd64.tar.gz
#   macOS aarch64 (M-series) franken_whisper-X.Y.Z-darwin_arm64.tar.gz
#   Windows x64           franken_whisper-X.Y.Z-windows_amd64.zip
#
# Windows users: this bash installer covers linux + darwin (and WSL). On
# native Windows, download the windows_amd64.zip from the releases page and
# unzip franken_whisper.exe onto your PATH manually.
#
set -euo pipefail
umask 022
shopt -s lastpipe 2>/dev/null || true

# ============================================================================
# Configuration
# ============================================================================
VERSION="${VERSION:-}"
OWNER="${OWNER:-Dicklesworthstone}"
REPO="${REPO:-franken_whisper}"
BINARY_NAME="franken_whisper"
ALIAS_NAME="fw"
DEST_DEFAULT="$HOME/.local/bin"
DEST="${DEST:-$DEST_DEFAULT}"
DEST_EXPLICIT=0
EASY=0
QUIET=0
VERIFY=0
NO_PULL=0
FROM_SOURCE=0
UNINSTALL=0
FORCE_INSTALL=0
NO_CHECKSUM=0
CHECKSUM="${CHECKSUM:-}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
OFFLINE_TARBALL=""
LOCK_FILE="/tmp/franken-whisper-install.lock"
INSTALLER_TEMP_ROOT="${TMPDIR:-/tmp}"
while [ "$INSTALLER_TEMP_ROOT" != "/" ] && [[ "$INSTALLER_TEMP_ROOT" == */ ]]; do
    INSTALLER_TEMP_ROOT="${INSTALLER_TEMP_ROOT%/}"
done
if [ "$INSTALLER_TEMP_ROOT" = "/" ]; then
    INSTALLER_TEMP_TEMPLATE="/franken-whisper-install.XXXXXX"
else
    INSTALLER_TEMP_TEMPLATE="$INSTALLER_TEMP_ROOT/franken-whisper-install.XXXXXX"
fi
SYSTEM=0
NO_GUM=0
MAX_RETRIES=3
DOWNLOAD_TIMEOUT=120
# The two compiled model packages occupy 2,117,087,997 bytes. Reserve modest
# filesystem/metadata headroom before beginning a fresh default pull.
MIN_MODEL_CACHE_KB=2250000
# shellcheck disable=SC2034  # informational metadata, not read by the script
INSTALLER_VERSION="2.0.0"

# The running binary is authoritative for its version. The marker remains a
# fallback for older/manual installations that cannot self-report.
VERSION_MARKER_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/franken_whisper"
VERSION_MARKER="$VERSION_MARKER_DIR/.installed-version"

# Proxy args applied to EVERY curl invocation. Empty array expands to nothing.
PROXY_ARGS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

GUM_AVAILABLE=false

# ============================================================================
# Gum detection (no auto-install — keep installer lean)
# ============================================================================
check_gum() {
    [[ "$NO_GUM" -eq 1 ]] && return 1
    if command -v gum &>/dev/null && [ -t 1 ]; then
        GUM_AVAILABLE=true
        return 0
    fi
    return 1
}

# ============================================================================
# Proxy
# ============================================================================
setup_proxy() {
    PROXY_ARGS=()
    if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
        PROXY_ARGS=(--proxy "${HTTPS_PROXY:-$https_proxy}")
    elif [[ -n "${HTTP_PROXY:-${http_proxy:-}}" ]]; then
        PROXY_ARGS=(--proxy "${HTTP_PROXY:-$http_proxy}")
    fi
}

# ============================================================================
# Styled output
# ============================================================================
print_banner() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style \
            --border double \
            --border-foreground 208 \
            --padding "0 2" \
            --margin "1 0" \
            --bold \
            "$(gum style --foreground 208 'franken_whisper installer')" \
            "$(gum style --foreground 245 'Agent-first Rust ASR orchestration')"
    else
        echo ""
        echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${BLUE}║${NC}  ${BOLD}${GREEN}franken_whisper installer${NC}                           ${BOLD}${BLUE}║${NC}"
        echo -e "${BOLD}${BLUE}║${NC}  ${DIM}Agent-first Rust ASR orchestration${NC}                  ${BOLD}${BLUE}║${NC}"
        echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
}

# Draw a box around text with automatic width calculation.
# Usage: draw_box "ansi_color" "line1" "line2" ...
draw_box() {
    local color="$1"; shift
    local lines=("$@")
    local max_width=0 esc stripped len
    esc=$(printf '\033')
    local strip_ansi_sed="s/${esc}\\[[0-9;]*m//g"

    for line in "${lines[@]}"; do
        stripped=$(printf '%b' "$line" | LC_ALL=C sed "$strip_ansi_sed")
        len=${#stripped}
        [ "$len" -gt "$max_width" ] && max_width=$len
    done

    local inner_width=$((max_width + 4))
    local border=""
    for ((i=0; i<inner_width; i++)); do border+="═"; done

    printf "\033[%sm╔%s╗\033[0m\n" "$color" "$border"
    for line in "${lines[@]}"; do
        stripped=$(printf '%b' "$line" | LC_ALL=C sed "$strip_ansi_sed")
        len=${#stripped}
        local padding=$((max_width - len)) pad_str=""
        for ((i=0; i<padding; i++)); do pad_str+=" "; done
        printf "\033[%sm║\033[0m  %b%s  \033[%sm║\033[0m\n" "$color" "$line" "$pad_str" "$color"
    done
    printf "\033[%sm╚%s╝\033[0m\n" "$color" "$border"
}

log_info() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 39 "→ $1" >&2
    else
        echo -e "${BLUE}→${NC} $1" >&2
    fi
}

log_warn() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 214 "⚠ $1" >&2
    else
        echo -e "${YELLOW}⚠${NC} $1" >&2
    fi
}

log_error() {
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 196 "✗ $1" >&2
    else
        echo -e "${RED}✗${NC} $1" >&2
    fi
}

log_step() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 208 "→ $1" >&2
    else
        echo -e "${BLUE}→${NC} $1" >&2
    fi
}

log_success() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 82 "✓ $1" >&2
    else
        echo -e "${GREEN}✓${NC} $1" >&2
    fi
}

log_debug() {
    [[ "${DEBUG:-0}" -eq 1 ]] || return 0
    echo -e "${CYAN}[fw:debug]${NC} $1" >&2
}

# Run a command behind a gum spinner (or a plain step line as fallback).
run_with_spinner() {
    local title="$1"; shift
    if [[ "$GUM_AVAILABLE" == "true" ]] && [ "$QUIET" -eq 0 ]; then
        gum spin --spinner dot --title "$title" -- "$@"
    else
        log_step "$title"
        "$@"
    fi
}

die() {
    log_error "$@"
    exit 1
}

# ============================================================================
# Usage
# ============================================================================
usage() {
    cat <<'EOF'
franken_whisper installer — install the agent-first ASR orchestration CLI

Usage:
  curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_whisper/main/install.sh | bash
  curl -fsSL .../install.sh | bash -s -- [OPTIONS]

Options:
  --version vX.Y.Z   Install specific version (default: latest)
  --dest DIR         Install to DIR (default: ~/.local/bin)
  --system           Install to /usr/local/bin (requires sudo)
  --easy-mode        Auto-update PATH in shell rc files
  --verify           Run self-test after install
  --no-pull          Skip automatic native Whisper + diarization model download
  --force            Reinstall even if the same version is already present
  --artifact-url URL Use a custom release artifact URL
  --checksum SHA     Provide expected SHA256 checksum
  --offline TARBALL  Install from a local archive (airgap); verifies a
                     sibling <TARBALL>.sha256 if present
  --from-source      Build from source instead of downloading a binary
  --no-verify        Skip checksum verification (testing only)
  --quiet            Suppress non-error output
  --no-gum           Disable gum formatting even if available
  --uninstall        Remove franken_whisper and clean up
  --help             Show this help

Environment Variables:
  HTTP_PROXY / HTTPS_PROXY   Honored on every download
  FW_INSTALL_DIR             Override default install directory
  FW_INSTALL_PROMPT_TIMEOUT  Model prompt timeout in seconds (default: 120)
  FRANKEN_WHISPER_MODEL_DIR  Override the model cache root
  VERSION                    Override version to install

Platforms (prebuilt binaries):
  Linux x86_64             franken_whisper-X.Y.Z-linux_amd64.tar.gz
  Linux aarch64            franken_whisper-X.Y.Z-linux_arm64.tar.gz
  macOS x86_64 (Intel)     franken_whisper-X.Y.Z-darwin_amd64.tar.gz
  macOS aarch64 (M-series) franken_whisper-X.Y.Z-darwin_arm64.tar.gz
  Windows x64              franken_whisper-X.Y.Z-windows_amd64.zip

Windows note:
  This bash installer covers linux + darwin (and WSL). On native Windows,
  download the windows_amd64.zip from the releases page and unzip
  franken_whisper.exe onto your PATH manually.

Examples:
  # Default install (latest release)
  curl -fsSL .../install.sh | bash

  # System install with PATH auto-update
  curl -fsSL .../install.sh | sudo bash -s -- --system --easy-mode

  # Specific version
  curl -fsSL .../install.sh | bash -s -- --version v0.9.0

  # Airgapped install from a local archive
  bash install.sh --offline ./franken_whisper-0.9.0-linux_amd64.tar.gz

  # Uninstall
  curl -fsSL .../install.sh | bash -s -- --uninstall
EOF
    exit 0
}

# ============================================================================
# Argument Parsing
# ============================================================================
require_option_value() {
    local option="$1"
    if [ "$#" -lt 2 ] || [ -z "${2:-}" ] || [[ "${2:-}" == --* ]]; then
        die "$option requires a non-empty value"
    fi
}

require_assignment_value() {
    local option="$1" value="$2"
    [ -n "$value" ] || die "$option requires a non-empty value"
}

# shellcheck disable=SC2034  # SYSTEM records --system intent for clarity; DEST is what's actually used
while [ $# -gt 0 ]; do
    case "$1" in
        --version) require_option_value "$1" "${2:-}"; VERSION="$2"; shift 2;;
        --version=*) VERSION="${1#*=}"; require_assignment_value "--version" "$VERSION"; shift;;
        --dest) require_option_value "$1" "${2:-}"; DEST="$2"; DEST_EXPLICIT=1; shift 2;;
        --dest=*) DEST="${1#*=}"; require_assignment_value "--dest" "$DEST"; DEST_EXPLICIT=1; shift;;
        --system)
            SYSTEM=1; DEST="/usr/local/bin"
            # Keep the version marker beside the system binary (not under the
            # invoking user's HOME) so a later `sudo ... --uninstall` finds and
            # removes it regardless of which user's HOME sudo preserves.
            VERSION_MARKER_DIR="/usr/local/share/franken_whisper"
            VERSION_MARKER="$VERSION_MARKER_DIR/.installed-version"
            shift;;
        --easy-mode) EASY=1; shift;;
        --verify) VERIFY=1; shift;;
        --no-pull) NO_PULL=1; shift;;
        --force) FORCE_INSTALL=1; shift;;
        --artifact-url) require_option_value "$1" "${2:-}"; ARTIFACT_URL="$2"; shift 2;;
        --artifact-url=*) ARTIFACT_URL="${1#*=}"; require_assignment_value "--artifact-url" "$ARTIFACT_URL"; shift;;
        --checksum) require_option_value "$1" "${2:-}"; CHECKSUM="$2"; shift 2;;
        --checksum=*) CHECKSUM="${1#*=}"; require_assignment_value "--checksum" "$CHECKSUM"; shift;;
        --offline) require_option_value "$1" "${2:-}"; OFFLINE_TARBALL="$2"; shift 2;;
        --offline=*) OFFLINE_TARBALL="${1#*=}"; require_assignment_value "--offline" "$OFFLINE_TARBALL"; shift;;
        --from-source) FROM_SOURCE=1; shift;;
        --no-verify) NO_CHECKSUM=1; shift;;
        --quiet|-q) QUIET=1; shift;;
        --no-gum) NO_GUM=1; shift;;
        --uninstall) UNINSTALL=1; shift;;
        -h|--help) usage;;
        --*) die "Unknown option: $1 (run with --help for supported options)";;
        *) die "Unexpected positional argument: $1 (run with --help for usage)";;
    esac
done

# Environment variable overrides
if [ -n "${FW_INSTALL_DIR:-}" ]; then
    [ "$SYSTEM" -eq 0 ] || die "FW_INSTALL_DIR cannot be combined with --system"
    DEST="$FW_INSTALL_DIR"
    DEST_EXPLICIT=1
fi

[ "$SYSTEM" -eq 0 ] || [ "$DEST_EXPLICIT" -eq 0 ] || die "--system cannot be combined with --dest"
[ -z "$OFFLINE_TARBALL" ] || [ "$FROM_SOURCE" -eq 0 ] || die "--offline cannot be combined with --from-source"
[ -z "$OFFLINE_TARBALL" ] || [ -z "$ARTIFACT_URL" ] || die "--offline cannot be combined with --artifact-url"
[ "$FROM_SOURCE" -eq 0 ] || [ -z "$ARTIFACT_URL" ] || die "--from-source cannot be combined with --artifact-url"
if [ -n "$VERSION" ] && [[ ! "$VERSION" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]]; then
    die "Invalid --version '$VERSION' (expected vX.Y.Z or X.Y.Z)"
fi
if [ -n "$VERSION" ] && [[ "$VERSION" != v* ]]; then
    VERSION="v$VERSION"
fi
if [ -n "$CHECKSUM" ] && [[ ! "${CHECKSUM%% *}" =~ ^[0-9A-Fa-f]{64}$ ]]; then
    die "Invalid --checksum (expected exactly 64 hexadecimal characters)"
fi
if [ -n "$ARTIFACT_URL" ] && [[ ! "$ARTIFACT_URL" =~ ^https:// ]]; then
    die "--artifact-url must use HTTPS; use --offline for local archives"
fi
[ -z "$ARTIFACT_URL" ] || [ -n "$VERSION" ] || die "--artifact-url requires --version so the staged binary can be authenticated"
if [ -n "$ARTIFACT_URL" ] && [ -z "$CHECKSUM" ] && [ "$NO_CHECKSUM" -eq 0 ]; then
    die "--artifact-url requires --checksum (or explicit testing-only --no-verify)"
fi
if [ "$UNINSTALL" -eq 1 ] && { [ -n "$VERSION" ] || [ -n "$OFFLINE_TARBALL" ] || [ "$FROM_SOURCE" -eq 1 ] || [ -n "$ARTIFACT_URL" ] || [ -n "$CHECKSUM" ] || [ "$VERIFY" -eq 1 ]; }; then
    die "--uninstall cannot be combined with install-source, version, checksum, or verification options"
fi

check_gum || true
setup_proxy

# ============================================================================
# Uninstall
# ============================================================================
do_uninstall() {
    print_banner
    log_step "Uninstalling franken_whisper..."

    if [ -f "$DEST/$BINARY_NAME" ]; then
        rm -f "$DEST/$BINARY_NAME"
        log_success "Removed $DEST/$BINARY_NAME"
    else
        log_warn "Binary not found at $DEST/$BINARY_NAME"
    fi

    if [ -f "$DEST/$ALIAS_NAME" ]; then
        rm -f "$DEST/$ALIAS_NAME"
        log_success "Removed $DEST/$ALIAS_NAME"
    fi

    if [ -f "$VERSION_MARKER" ]; then
        rm -f "$VERSION_MARKER"
        rmdir "$VERSION_MARKER_DIR" 2>/dev/null || true
        log_step "Removed version marker"
    fi

    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [ -f "$rc" ] && grep -q "# franken_whisper installer" "$rc" 2>/dev/null; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' '/# franken_whisper installer/d' "$rc" 2>/dev/null || true
            else
                sed -i '/# franken_whisper installer/d' "$rc" 2>/dev/null || true
            fi
            log_step "Cleaned $rc"
        fi
    done

    log_success "franken_whisper uninstalled"
    exit 0
}

[ "$UNINSTALL" -eq 1 ] && do_uninstall

# ============================================================================
# Platform Detection
# ============================================================================
# Sets PLATFORM (release asset suffix) and IS_WSL. Unsupported targets fail
# with an explicit message; source builds require an explicit --from-source.
PLATFORM=""
IS_WSL=0
detect_platform() {
    local os arch
    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *) die "Unsupported OS: $(uname -s)" ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) die "Unsupported architecture: $(uname -m). Use --from-source only after confirming sibling-crate support." ;;
    esac

    # WSL: behaves like linux for our purposes, but warn the user.
    if [ "$os" = "linux" ] && grep -qi microsoft /proc/version 2>/dev/null; then
        # shellcheck disable=SC2034  # IS_WSL records detection state for future use/debugging
        IS_WSL=1
        log_warn "WSL detected — installing the linux_${arch} binary (some audio/tty features may need extra setup)"
    fi

    # All four POSIX targets ship a prebuilt binary:
    #   linux_amd64, linux_arm64, darwin_amd64, darwin_arm64
    # (windows_amd64 ships a .zip but native Windows isn't covered by this
    # bash installer — see the --help Windows note.)
    PLATFORM="${os}_${arch}"
    if [ "$os" = "windows" ]; then
        die "Native Windows is not supported by this bash installer. Download franken_whisper-<ver>-windows_amd64.zip from the releases page instead."
    fi
}

# ============================================================================
# Version Resolution
# ============================================================================
resolve_version() {
    if [ -n "$VERSION" ]; then return 0; fi

    log_step "Resolving latest version..."
    local latest_url="https://api.github.com/repos/${OWNER}/${REPO}/releases/latest"
    local tag="" attempts=0

    while [ $attempts -lt $MAX_RETRIES ] && [ -z "$tag" ]; do
        attempts=$((attempts + 1))
        if command -v curl &>/dev/null; then
            tag=$(curl -fsSL "${PROXY_ARGS[@]}" \
                --connect-timeout 10 --max-time 30 \
                -H "Accept: application/vnd.github.v3+json" \
                "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || echo "")
        elif command -v wget &>/dev/null; then
            tag=$(wget -qO- --timeout=30 "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || echo "")
        fi
        [ -z "$tag" ] && [ $attempts -lt $MAX_RETRIES ] && sleep 2
    done

    if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]]; then
        VERSION="$tag"
        log_success "Latest version: $VERSION"
        return 0
    fi

    # Fallback: parse the latest-release redirect.
    log_step "Trying redirect-based version resolution..."
    local redirect_url="https://github.com/${OWNER}/${REPO}/releases/latest"
    if command -v curl &>/dev/null; then
        tag=$(curl -fsSL "${PROXY_ARGS[@]}" -o /dev/null -w '%{url_effective}' "$redirect_url" 2>/dev/null | sed -E 's|.*/tag/||' || echo "")
    fi

    if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]] && [[ "$tag" != *"/"* ]]; then
        VERSION="$tag"
        log_success "Latest version (via redirect): $VERSION"
        return 0
    fi

    die "Could not resolve the latest release. Retry with --version vX.Y.Z, --offline ARCHIVE, or explicitly choose --from-source."
}

# ============================================================================
# Locking
# ============================================================================
LOCK_DIR="${LOCK_FILE}.d"
LOCKED=0

acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        LOCKED=1
        echo $$ > "$LOCK_DIR/pid"
        return 0
    fi

    if [ -f "$LOCK_DIR/pid" ]; then
        local old_pid
        old_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
        if [[ ! "$old_pid" =~ ^[0-9]+$ ]]; then
            log_warn "Removing invalid installer lock metadata"
            rm -f "$LOCK_DIR/pid"
            rmdir "$LOCK_DIR" 2>/dev/null || true
            if mkdir "$LOCK_DIR" 2>/dev/null; then
                LOCKED=1; echo $$ > "$LOCK_DIR/pid"; return 0
            fi
        elif ! kill -0 "$old_pid" 2>/dev/null; then
            log_warn "Removing stale lock metadata (PID $old_pid not running)"
            rm -f "$LOCK_DIR/pid"
            rmdir "$LOCK_DIR" 2>/dev/null || true
            if mkdir "$LOCK_DIR" 2>/dev/null; then
                LOCKED=1; echo $$ > "$LOCK_DIR/pid"; return 0
            fi
        fi
    fi

    if [ "$LOCKED" -eq 0 ]; then
        die "Another installation is running. Inspect PID metadata at $LOCK_DIR/pid before retrying."
    fi
}

# ============================================================================
# Cleanup
# ============================================================================
TMP=""
cleanup() {
    if [ -n "$TMP" ]; then
        local temp_parent temp_name
        temp_parent=$(dirname "$TMP")
        temp_name=$(basename "$TMP")
        if [ "$temp_parent" = "$INSTALLER_TEMP_ROOT" ] && [[ "$temp_name" == franken-whisper-install.* ]]; then
            # `TMP` is a freshly-created installer-owned directory. Delete its
            # children bottom-up, then remove the now-empty root. Never apply a
            # recursive removal command to an unvalidated path.
            find "$TMP" -depth -mindepth 1 -delete 2>/dev/null || \
                log_warn "Temporary installer files remain at $TMP"
            rmdir "$TMP" 2>/dev/null || true
        else
            log_warn "Refusing to clean unexpected temporary path: $TMP"
        fi
    fi
    if [ "$LOCKED" -eq 1 ]; then
        rm -f "$LOCK_DIR/pid" 2>/dev/null || true
        rmdir "$LOCK_DIR" 2>/dev/null || \
            log_warn "Installer lock directory was not empty and was retained: $LOCK_DIR"
    fi
}
trap cleanup EXIT

# ============================================================================
# Preflight checks
# ============================================================================
check_disk_space() {
    # ~50MB headroom for the archive + extracted binary. Source builds need
    # far more (5 repo clones + crate cache + release target: several GB) —
    # build_from_source emits its own explicit disk warning before cloning.
    local min_kb=51200
    # Walk up to the nearest existing ancestor so df has a real path to stat.
    local path="$DEST"
    while [ -n "$path" ] && [ ! -d "$path" ]; do
        local parent; parent=$(dirname "$path")
        [ "$parent" = "$path" ] && break
        path="$parent"
    done
    [ -d "$path" ] || path="/"
    if command -v df >/dev/null 2>&1; then
        local avail_kb
        avail_kb=$(df -Pk "$path" 2>/dev/null | awk 'NR==2 {print $4}' || true)
        if [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$min_kb" ]; then
            die "Insufficient disk space in $path (need at least 50MB)"
        fi
    fi
}

available_kb_for_path() {
    local path="$1"
    while [ -n "$path" ] && [ ! -d "$path" ]; do
        local parent; parent=$(dirname "$path")
        [ "$parent" = "$path" ] && break
        path="$parent"
    done
    [ -d "$path" ] || path="/"
    command -v df >/dev/null 2>&1 || return 1
    df -Pk "$path" 2>/dev/null | awk 'NR==2 {print $4}'
}

check_write_permissions() {
    if [ ! -d "$DEST" ]; then
        if ! mkdir -p "$DEST" 2>/dev/null; then
            log_error "Cannot create $DEST (insufficient permissions)"
            die "Try running with sudo or choose a writable --dest"
        fi
    fi
    if [ ! -w "$DEST" ]; then
        log_error "No write permission to $DEST"
        die "Try running with sudo or choose a writable --dest"
    fi
}

check_existing_install() {
    if [ -x "$DEST/$BINARY_NAME" ]; then
        local current
        current=$(read_installed_version)
        log_info "Existing franken_whisper detected (version: ${current})"
    fi
}

check_network() {
    # Offline / source / custom-artifact paths don't need github reachable.
    [ -n "$OFFLINE_TARBALL" ] && return 0
    [ "$FROM_SOURCE" -eq 1 ] && return 0
    command -v curl >/dev/null 2>&1 || { log_warn "curl not found; skipping network check"; return 0; }
    if ! curl -fsSL "${PROXY_ARGS[@]}" --connect-timeout 3 --max-time 5 -o /dev/null "https://github.com" 2>/dev/null; then
        log_warn "Could not reach github.com; download may fail"
    fi
}

preflight_checks() {
    log_info "Running preflight checks"
    check_disk_space
    check_write_permissions
    check_existing_install
    check_network
}

# ============================================================================
# Installed-version detection
# ============================================================================
binary_reported_version() {
    local binary="$1" reported
    [ -x "$binary" ] || return 1
    reported=$("$binary" --version 2>/dev/null | head -1 || true)
    if [[ "$reported" =~ ([0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?) ]]; then
        printf 'v%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

read_installed_version() {
    local reported
    if reported=$(binary_reported_version "$DEST/$BINARY_NAME"); then
        printf '%s\n' "$reported"
        return 0
    fi
    if [ -f "$VERSION_MARKER" ]; then
        cat "$VERSION_MARKER" 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

write_installed_version() {
    mkdir -p "$VERSION_MARKER_DIR" 2>/dev/null || true
    printf '%s\n' "${1:-unknown}" > "$VERSION_MARKER" 2>/dev/null || true
}

# Already-installed short-circuit. The binary self-report wins, with the marker
# retained only for legacy/manual installations. Honors --force.
already_installed() {
    [ "$FORCE_INSTALL" -eq 1 ] && return 1
    [ -n "$OFFLINE_TARBALL" ] && return 1   # offline: caller knows what they want
    [ -z "$VERSION" ] && return 1
    [ -x "$DEST/$BINARY_NAME" ] || return 1
    [ -x "$DEST/$ALIAS_NAME" ] || return 1
    local installed
    installed=$(read_installed_version)
    [ "$installed" = "$VERSION" ]
}

# ============================================================================
# PATH modification
# ============================================================================
maybe_add_path() {
    case ":$PATH:" in
        *:"$DEST":*) return 0;;
        *)
            if [ "$EASY" -eq 1 ]; then
                local updated=0
                for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
                    if [ -f "$rc" ] && [ -w "$rc" ]; then
                        if ! grep -qF "$DEST" "$rc" 2>/dev/null; then
                            echo "" >> "$rc"
                            echo "export PATH=\"$DEST:\$PATH\"  # franken_whisper installer" >> "$rc"
                        fi
                        updated=1
                    fi
                done
                if [ "$updated" -eq 1 ]; then
                    log_warn "PATH updated; restart shell or run: export PATH=\"$DEST:\$PATH\""
                else
                    log_warn "Add $DEST to PATH to use franken_whisper"
                fi
            else
                log_warn "Add $DEST to PATH to use franken_whisper"
            fi
        ;;
    esac
}

# ============================================================================
# Download with retry (proxy-aware)
# ============================================================================
download_file() {
    local url="$1" dest="$2" attempt=0
    local partial="${dest}.part"

    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        log_debug "Download attempt $attempt for $url"
        if command -v curl &>/dev/null; then
            if curl -fsSL "${PROXY_ARGS[@]}" --connect-timeout 30 --max-time "$DOWNLOAD_TIMEOUT" \
                --retry 2 -o "$partial" "$url"; then
                mv -f "$partial" "$dest"; return 0
            fi
        elif command -v wget &>/dev/null; then
            if wget --quiet --timeout="$DOWNLOAD_TIMEOUT" -O "$partial" "$url"; then
                mv -f "$partial" "$dest"; return 0
            fi
        else
            die "Neither curl nor wget found"
        fi
        [ $attempt -lt $MAX_RETRIES ] && { log_warn "Download failed, retrying in 3s..."; sleep 3; }
    done
    rm -f "$partial" 2>/dev/null || true
    return 1
}

# ============================================================================
# Checksum verification (dual tool: sha256sum / shasum)
# ============================================================================
sha256_of() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo ""
    fi
}

# Resolve the release archive checksum from the artifacts DSR actually
# publishes. Prefer the archive-bound sidecar, then the canonical combined
# SHA256SUMS manifest; retain the older checksums-sha256.txt name only as a
# final compatibility source for historical releases.
release_archive_checksum() {
    local archive_url="$1" archive_name="$2"
    local release_base="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}"
    local candidate_url candidate_path expected
    local -a candidate_urls=(
        "${archive_url}.sha256"
        "${release_base}/SHA256SUMS"
        "${release_base}/checksums-sha256.txt"
    )
    local -a candidate_paths=(
        "$TMP/${archive_name}.sha256"
        "$TMP/SHA256SUMS"
        "$TMP/checksums-sha256.txt"
    )
    local index
    for index in "${!candidate_urls[@]}"; do
        candidate_url="${candidate_urls[$index]}"
        candidate_path="${candidate_paths[$index]}"
        if download_file "$candidate_url" "$candidate_path"; then
            expected=$(awk -v name="$archive_name" \
                '$2 == name || $2 == "*" name { print $1; exit }' \
                "$candidate_path" 2>/dev/null || true)
            if [[ "$expected" =~ ^[0-9A-Fa-f]{64}$ ]]; then
                printf '%s\n' "$expected"
                return 0
            fi
        fi
    done
    return 1
}

# ============================================================================
# Validated binary-pair install
# ============================================================================
validate_binary_for_install() {
    local src="$1" label="$2"
    local reported expected
    if ! reported=$(binary_reported_version "$src"); then
        log_error "$label failed --version validation"
        return 1
    fi
    if [ -n "$VERSION" ]; then
        expected="$VERSION"
        [[ "$expected" == v* ]] || expected="v$expected"
        if [ "$reported" != "$expected" ]; then
            log_error "$label reports $reported but the requested release is $expected"
            return 1
        fi
    fi
}

install_binary_atomic() {
    local src="$1" dest="$2"
    local tmp_dest="${dest}.tmp.$$"
    install -m 0755 "$src" "$tmp_dest"
    if ! mv -f "$tmp_dest" "$dest"; then
        rm -f "$tmp_dest" 2>/dev/null || true
        die "Failed to move binary into place"
    fi
}

# Validate both command names before replacing either one. Install the alias
# first and the authoritative long name last, so an unlikely second rename
# failure never makes the primary command advertise an unvalidated release.
install_binary_pair() {
    local long_src="$1" alias_src="$2"
    local long_report alias_report
    validate_binary_for_install "$long_src" "$BINARY_NAME binary" || \
        die "Release archive failed binary validation; existing installation was not replaced"
    validate_binary_for_install "$alias_src" "$ALIAS_NAME binary" || \
        die "Release archive failed alias validation; existing installation was not replaced"
    long_report=$(binary_reported_version "$long_src")
    alias_report=$(binary_reported_version "$alias_src")
    [ "$long_report" = "$alias_report" ] || \
        die "Release archive contains mismatched binary versions; existing installation was not replaced"
    install_binary_atomic "$alias_src" "$DEST/$ALIAS_NAME"
    install_binary_atomic "$long_src" "$DEST/$BINARY_NAME"
}

# macOS filesystem sidecars are inert packaging noise, not payload: Finder
# ".DS_Store" entries and AppleDouble "._<member>" resource-fork shadows of an
# allowlisted member (as produced by macOS tar without COPYFILE_DISABLE=1).
# They are admitted as no-op members so such an archive still installs, but
# they are excluded from extraction and never installed.
is_macos_sidecar_member() {
    local name="$1" shadowed
    case "$name" in
        ".DS_Store") return 0 ;;
        ._?*) ;;
        *) return 1 ;;
    esac
    shadowed="${name#._}"
    case "$shadowed" in
        "$BINARY_NAME"|"$ALIAS_NAME"|"README.md"|"LICENSE"|"NOTICE.sortformer.txt"|"THIRD_PARTY_NOTICES.md"|"AGENTS.md"|".DS_Store")
            return 0 ;;
        *) return 1 ;;
    esac
}

# Admit only flat release archives with the exact supported member names. Tar
# entry types are checked before extraction; both formats are checked again for
# regular, non-symlink binaries before installation.
validate_archive_members() {
    local archive="$1" archive_ext="$2" members member normalized
    local binary_count=0 alias_count=0 readme_count=0 license_count=0 sortformer_notice_count=0 third_party_count=0 agents_count=0
    if [ "$archive_ext" = "zip" ]; then
        members=$(unzip -Z1 "$archive") || return 1
    else
        members=$(tar -tzf "$archive") || return 1
        if tar -tvzf "$archive" | awk '{print substr($1, 1, 1)}' | grep -qv '^-'; then
            log_error "Archive contains a non-regular entry"
            return 1
        fi
    fi
    while IFS= read -r member; do
        normalized="${member#./}"
        case "$normalized" in
            "$BINARY_NAME") binary_count=$((binary_count + 1)) ;;
            "$ALIAS_NAME") alias_count=$((alias_count + 1)) ;;
            "README.md") readme_count=$((readme_count + 1)) ;;
            "LICENSE") license_count=$((license_count + 1)) ;;
            "NOTICE.sortformer.txt") sortformer_notice_count=$((sortformer_notice_count + 1)) ;;
            "THIRD_PARTY_NOTICES.md") third_party_count=$((third_party_count + 1)) ;;
            "AGENTS.md") agents_count=$((agents_count + 1)) ;;
            *)
                if is_macos_sidecar_member "$normalized"; then
                    continue
                fi
                log_error "Archive contains an unexpected member"
                return 1
                ;;
        esac
    done <<< "$members"
    [ "$binary_count" -eq 1 ] && [ "$alias_count" -eq 1 ] && \
        [ "$readme_count" -le 1 ] && [ "$license_count" -le 1 ] && \
        [ "$sortformer_notice_count" -le 1 ] && \
        [ "$third_party_count" -le 1 ] && [ "$agents_count" -le 1 ] || {
        log_error "Archive must contain exactly one $BINARY_NAME, exactly one $ALIAS_NAME, and no duplicate allowlisted members"
        return 1
    }
}

# Extract an archive into TMP and install both supported binary names.
extract_and_install() {
    local archive="$1" archive_ext="$2"

    log_step "Extracting..."
    if [[ "$archive_ext" == "zip" ]]; then
        command -v unzip &>/dev/null || die "unzip required for .zip archives"
    else
        command -v tar &>/dev/null || die "tar required for .tar.gz archives"
    fi
    validate_archive_members "$archive" "$archive_ext" || return 1
    # macOS sidecar members pass validation as no-ops but are excluded here so
    # nothing beyond the allowlisted payload lands in the extract directory
    # (and so bsdtar never replays an AppleDouble "._fw" as extended
    # attributes onto the real binary).
    if [[ "$archive_ext" == "zip" ]]; then
        unzip -o "$archive" -x '._*' '.DS_Store' -d "$TMP/extract" >/dev/null 2>&1 || return 1
    else
        mkdir -p "$TMP/extract"
        tar --exclude '._*' --exclude '.DS_Store' -xzf "$archive" -C "$TMP/extract" 2>/dev/null || return 1
    fi

    local bin="$TMP/extract/$BINARY_NAME"
    if [ ! -f "$bin" ] || [ -L "$bin" ]; then
        log_error "Binary not found after extraction"
        return 1
    fi

    local alias_bin="$TMP/extract/$ALIAS_NAME"
    if [ ! -f "$alias_bin" ] || [ -L "$alias_bin" ]; then
        log_error "Short alias not found as a regular file after extraction"
        return 1
    fi
    chmod +x "$bin" "$alias_bin"
    install_binary_pair "$bin" "$alias_bin"
    return 0
}

# ============================================================================
# Offline install (airgap) — install from a local archive
# ============================================================================
install_offline() {
    [ -f "$OFFLINE_TARBALL" ] || die "Offline archive not found: $OFFLINE_TARBALL"
    log_step "Installing from local archive: $OFFLINE_TARBALL"

    local archive_ext="tar.gz"
    [[ "$OFFLINE_TARBALL" == *.zip ]] && archive_ext="zip"

    # Verify against a sibling .sha256 if one exists (or an explicit --checksum).
    local expected=""
    if [ "$NO_CHECKSUM" -eq 0 ]; then
        if [ -n "$CHECKSUM" ]; then
            expected="${CHECKSUM%% *}"
        elif [ -f "${OFFLINE_TARBALL}.sha256" ]; then
            expected=$(awk '{print $1}' "${OFFLINE_TARBALL}.sha256" 2>/dev/null | head -1)
        fi
        if [ -n "$expected" ]; then
            log_step "Verifying checksum..."
            local actual; actual=$(sha256_of "$OFFLINE_TARBALL")
            expected=$(printf '%s' "$expected" | tr 'A-F' 'a-f')
            actual=$(printf '%s' "$actual" | tr 'A-F' 'a-f')
            if [ -z "$actual" ]; then
                die "No SHA256 tool found; use --no-verify only in a controlled test environment"
            elif [ "$expected" != "$actual" ]; then
                die "Checksum mismatch! expected=$expected got=$actual"
            else
                log_success "Checksum verified"
            fi
        else
            die "No checksum available for offline archive; provide --checksum, a sibling .sha256, or explicit testing-only --no-verify"
        fi
    else
        log_warn "Checksum verification disabled (--no-verify)"
    fi

    extract_and_install "$OFFLINE_TARBALL" "$archive_ext" || die "Failed to install from offline archive"
    write_installed_version "${VERSION:-offline}"
    log_success "Installed to $DEST/$BINARY_NAME and $DEST/$ALIAS_NAME (offline)"
}

# ============================================================================
# Build from source
# ============================================================================
# HONESTY NOTE on source builds:
#   franken_whisper depends on SIBLING path crates that live OUTSIDE its repo:
#     - ../frankensqlite/crates/{fsqlite,fsqlite-types}   (required)
#     - ../frankentorch/crates/{ft-kernel-cpu,ft-core,
#                                ft-kernel-metal}          (required/target-gated)
#     - ../frankentui/crates/ftui                         (optional feature,
#                                                           manifest required)
#   Cargo.lock does not record commits for path dependencies. Release source
#   builds therefore use scripts/prepare_release_siblings.sh from the selected
#   franken_whisper revision to provision exact clean sibling commits. Existing
#   mismatched sibling directories fail closed and are never reset or cleaned.
ensure_rust() {
    command -v cargo >/dev/null 2>&1 && return 0
    log_step "Installing Rust via rustup..."
    log_warn "cargo not found: about to install Rust via the official rustup.rs"
    log_warn "bootstrap (a second remote script, auto-accepted). Ctrl-C now to"
    log_warn "abort and install Rust yourself if you prefer."
    # The grace period only makes sense when the warning was visible.
    [ "$QUIET" -eq 1 ] || sleep 3
    curl -fsSL "${PROXY_ARGS[@]}" https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
    # shellcheck disable=SC1091  # rustup-generated env file not present at lint time
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    command -v cargo >/dev/null 2>&1
}

build_from_source() {
    log_step "Building from source..."
    log_warn "Source builds clone four repositories and compile a release"
    log_warn "binary: expect several GB of disk use (crate cache + target dir)"
    log_warn "and multiple minutes of compile time."
    ensure_rust || die "Rust (cargo) is required for source builds and could not be installed"

    local build_root="$TMP/src"
    mkdir -p "$build_root"
    local fw_dir="$build_root/franken_whisper"

    log_step "Cloning franken_whisper..."
    git clone --quiet "https://github.com/${OWNER}/${REPO}.git" "$fw_dir" || die "Failed to clone franken_whisper"

    # Check out the requested tag (default branch only for an explicit
    # unversioned --from-source request).
    if [ -n "$VERSION" ]; then
        if ! (cd "$fw_dir" && git checkout --quiet "$VERSION" 2>/dev/null); then
            die "Could not check out requested source tag $VERSION"
        fi
    fi

    local sibling_preparer="$fw_dir/scripts/prepare_release_siblings.sh"
    [ -f "$sibling_preparer" ] || die "Selected source revision lacks a pinned sibling preparation contract"
    log_step "Provisioning exact sibling path-dependency revisions..."
    bash "$sibling_preparer" "$build_root" || die "Failed to provision pinned sibling dependencies"

    log_step "Building with cargo (default features; this may take several minutes)..."
    local target_dir="$TMP/target"
    if ! (cd "$fw_dir" && CARGO_TARGET_DIR="$target_dir" cargo build --locked --release --bins); then
        log_error "Source build failed."
        log_error "The pinned sibling contract was satisfied, but Cargo still failed."
        log_error "Prefer a prebuilt release binary, or reproduce with the same pinned sibling checkouts:"
        log_error "  git clone https://github.com/${OWNER}/${REPO}.git && cd ${REPO} && git checkout ${VERSION:-main}"
        log_error "  (clone frankensqlite + frankentorch as siblings, then: cargo build --release)"
        die "Build from source failed"
    fi

    local bin="$target_dir/release/$BINARY_NAME"
    if [ ! -x "$bin" ]; then
        bin=$(find "$target_dir" -name "$BINARY_NAME" -type f -perm -111 2>/dev/null | head -1)
    fi
    [ -x "$bin" ] || die "Binary not found after build"

    local alias_bin="$target_dir/release/$ALIAS_NAME"
    [ -x "$alias_bin" ] || die "Short alias binary not found after build"

    install_binary_pair "$bin" "$alias_bin"
    write_installed_version "${VERSION:-source}"
    log_success "Installed to $DEST/$BINARY_NAME and $DEST/$ALIAS_NAME (source build)"
}

# ============================================================================
# Download release binary
# ============================================================================
download_release() {
    local platform="$1"
    local archive_ext="tar.gz"
    [[ "$platform" == windows_* ]] && archive_ext="zip"

    local archive_name url
    if [ -n "$ARTIFACT_URL" ]; then
        url="$ARTIFACT_URL"
        printf -v archive_name '%s' "${ARTIFACT_URL##*/}"
        [[ "$archive_name" == *.zip ]] && archive_ext="zip"
    else
        # Assets use the version WITHOUT the leading 'v' (e.g. 0.7.0).
        local ver_no_v="${VERSION#v}"
        local name="$BINARY_NAME" version="$ver_no_v"
        local os="${platform%_*}" arch="${platform#*_}" ext="$archive_ext"
        archive_name="${name}-${version}-${os}_${arch}.${ext}"
        url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${archive_name}"
    fi

    log_step "Downloading $archive_name..."
    download_file "$url" "$TMP/$archive_name" || return 1
    [ -f "$TMP/$archive_name" ] || return 1

    # Checksum verification against the DSR per-asset sidecar or SHA256SUMS.
    if [ "$NO_CHECKSUM" -eq 1 ]; then
        log_warn "Checksum verification disabled (--no-verify)"
    else
        local expected=""
        if [ -n "$CHECKSUM" ]; then
            expected="${CHECKSUM%% *}"
        else
            expected=$(release_archive_checksum "$url" "$archive_name" || true)
        fi

        if [ -n "$expected" ]; then
            log_step "Verifying checksum..."
            local actual; actual=$(sha256_of "$TMP/$archive_name")
            expected=$(printf '%s' "$expected" | tr 'A-F' 'a-f')
            actual=$(printf '%s' "$actual" | tr 'A-F' 'a-f')
            if [ -z "$actual" ]; then
                log_error "No SHA256 tool found (sha256sum/shasum)"
                return 1
            elif [ "$expected" != "$actual" ]; then
                log_error "Checksum mismatch!"
                log_error "  Expected: $expected"
                log_error "  Got:      $actual"
                rm -f "$TMP/$archive_name"
                return 1
            else
                log_success "Checksum verified: ${actual:0:16}..."
            fi
        else
            log_error "Checksum not available for $archive_name"
            return 1
        fi
    fi

    extract_and_install "$TMP/$archive_name" "$archive_ext" || return 1
    write_installed_version "$VERSION"
    log_success "Installed to $DEST/$BINARY_NAME and $DEST/$ALIAS_NAME"
    return 0
}

# ============================================================================
# Self-test
# ============================================================================
# Validate the installed binary, stable agent-discovery surfaces, and both
# hash-pinned model packages required by the default all-native pipeline.
run_self_test() {
    log_step "Verifying installed agent contracts..."
    local long_version alias_version capabilities schema doctor
    long_version=$(binary_reported_version "$DEST/$BINARY_NAME") || die "Installed franken_whisper cannot report its version"
    alias_version=$(binary_reported_version "$DEST/$ALIAS_NAME") || die "Installed fw alias cannot report its version"
    [ "$long_version" = "$alias_version" ] || die "Installed binary names report different versions"
    capabilities=$("$DEST/$ALIAS_NAME" capabilities --json) || die "Installed fw capabilities probe failed"
    schema=$("$DEST/$ALIAS_NAME" robot schema) || die "Installed fw robot schema probe failed"
    doctor=$("$DEST/$ALIAS_NAME" doctor --json) || die "Installed fw doctor probe failed"
    [[ "$capabilities" == *'"schema_version":"franken-whisper-capabilities-v1"'* ]] || die "Capabilities probe returned an unexpected schema"
    [[ "$schema" == *'"schema_version"'* ]] || die "Robot schema probe returned an unexpected payload"
    [[ "$doctor" == *'"schema_version":"franken-whisper-doctor-v1"'* ]] || die "Doctor probe returned an unexpected schema"
    [[ "$doctor" == *'"ready":true'* ]] || die "Native default model verification failed; run: fw pull all"
    log_success "Installation contracts and both native default model packages verified"
}

# Return success only when a human operator can answer a prompt. `/dev/tty`
# keeps the prompt usable under `curl ... | bash`, where stdin contains the
# installer rather than terminal input.
interactive_tty() {
    [ -t 0 ] && return 0
    ( : </dev/tty >/dev/tty ) 2>/dev/null
}

confirm_default_model_pull() {
    # Headless and quiet installs take the documented default: provision both
    # packages. Interactive installs make the 2.12 GB transfer visible while
    # retaining Yes as the automatic default.
    if [ "$QUIET" -eq 1 ] || ! interactive_tty; then
        return 0
    fi

    local prompt_timeout="${FW_INSTALL_PROMPT_TIMEOUT:-120}"
    if [[ ! "$prompt_timeout" =~ ^[0-9]+$ ]] || [ "$prompt_timeout" -eq 0 ]; then
        prompt_timeout=120
    fi
    local answer=""
    local read_status=0
    printf 'Download and verify the native Whisper + diarization models now (about 2.12 GB)? (Y/n): ' >/dev/tty
    IFS= read -r -t "$prompt_timeout" answer </dev/tty || read_status=$?
    if [ "$read_status" -ne 0 ]; then
        printf '\n' >/dev/tty
        log_info "No reply within ${prompt_timeout}s; taking the documented default (Yes)."
        return 0
    fi
    case "$answer" in
        n|N|no|No|NO) return 1 ;;
        *) return 0 ;;
    esac
}

provision_default_models() {
    if [ "$NO_PULL" -eq 1 ]; then
        log_warn "Skipping native default model provisioning (--no-pull)"
        return 0
    fi

    local bin="$DEST/$ALIAS_NAME"
    local cache_home="$HOME"
    local pull_user=""
    local -a pull_command=("$bin" pull all)
    local -a doctor_command=("$bin" doctor --json)

    # A system install commonly runs under sudo. Pulling as root would populate
    # root's private cache, leaving the invoking user unable to use the model.
    # Drop only model provisioning back to that validated user, retain a
    # target-user HOME, and forward only the model/proxy variables the pull
    # contract documents.
    if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
        if ! command -v sudo >/dev/null 2>&1 || ! id -u "$SUDO_USER" >/dev/null 2>&1; then
            die "Cannot safely identify the invoking sudo user for native model provisioning"
        fi
        cache_home=$(sudo -H -u "$SUDO_USER" sh -c 'printf "%s" "$HOME"') || {
            die "Cannot resolve the invoking sudo user's home for native model provisioning"
        }
        if [ -z "$cache_home" ] || [[ "$cache_home" != /* ]]; then
            die "The invoking sudo user's home is not an absolute path"
        fi
        pull_user="$SUDO_USER"
        local -a forwarded_env=()
        local env_name
        for env_name in \
            FRANKEN_WHISPER_MODEL_DIR \
            HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY \
            http_proxy https_proxy all_proxy no_proxy
        do
            if [ "${!env_name+x}" = "x" ]; then
                forwarded_env+=("$env_name=${!env_name}")
            fi
        done
        local -a user_prefix=(sudo -H -u "$pull_user")
        if [ "${#forwarded_env[@]}" -gt 0 ]; then
            user_prefix+=(env "${forwarded_env[@]}")
        fi
        pull_command=("${user_prefix[@]}" "$bin" pull all)
        doctor_command=("${user_prefix[@]}" "$bin" doctor --json)
    fi

    local cache_root="${FRANKEN_WHISPER_MODEL_DIR:-$cache_home/.cache/franken_whisper/models}"
    if [[ "$cache_root" != /* ]]; then
        die "FRANKEN_WHISPER_MODEL_DIR must be absolute"
    fi

    # An offline archive never authorizes a network operation. Accept an
    # already-preseeded, verified cache; otherwise require the caller to make
    # the missing-model choice explicit with --no-pull.
    if [ -n "$OFFLINE_TARBALL" ]; then
        local offline_doctor
        offline_doctor=$("${doctor_command[@]}") || die "Could not inspect the offline model cache"
        if [[ "$offline_doctor" != *'"ready":true'* ]]; then
            die "Offline install requires both model packages to be preseeded and verified, or explicit --no-pull"
        fi
        log_success "Verified preseeded native model packages under $cache_root"
        return 0
    fi

    if ! confirm_default_model_pull; then
        NO_PULL=1
        log_warn "Skipped native model provisioning at the operator's request"
        log_info "Provision both default models later with: fw pull all"
        return 0
    fi

    local available_model_kb=""
    available_model_kb=$(available_kb_for_path "$cache_root" || true)
    if [[ "$available_model_kb" =~ ^[0-9]+$ ]] && [ "$available_model_kb" -lt "$MIN_MODEL_CACHE_KB" ]; then
        # A complete cache needs no additional capacity. Ask the binary to hash
        # both compiled trust roots before rejecting a low-free-space host.
        local cached_doctor
        cached_doctor=$("${doctor_command[@]}") || die "Could not verify the existing native model cache"
        if [[ "$cached_doctor" == *'"ready":true'* ]]; then
            log_success "Verified existing native Whisper and Sortformer models under $cache_root"
            return 0
        fi
        die "Insufficient free space for both native model packages under $cache_root (need at least 2.25 GB available)"
    elif [ -z "$available_model_kb" ]; then
        log_warn "Could not measure free space for the model cache; the verified pull will fail closed on an I/O error"
    fi

    log_step "Provisioning native Whisper and Sortformer models (about 2.12 GB)..."
    if [ -n "$pull_user" ]; then
        log_info "The model pull will run as the invoking user: $pull_user"
    fi
    if [ "$QUIET" -eq 1 ]; then
        "${pull_command[@]}" >/dev/null 2>&1 || die "Native model provisioning failed; retry with: fw pull all"
    else
        "${pull_command[@]}" || die "Native model provisioning failed; retry with: fw pull all"
    fi
    local doctor
    doctor=$("${doctor_command[@]}") || die "Native model verification failed after download"
    [[ "$doctor" == *'"ready":true'* ]] || die "Native model packages did not satisfy the compiled trust roots"
    log_success "Verified native Whisper and Sortformer models under $cache_root"
}

# ============================================================================
# Shell completions
# ============================================================================
# NOTE: franken_whisper v0.7.2 has NO `completions` subcommand (verified:
# `franken_whisper completions` => "unrecognized subcommand"). There is
# nothing to generate, so completion installation is deliberately skipped.

# ============================================================================
# AI agent hooks / skills
# ============================================================================
# NOTE: franken_whisper is a plain ASR CLI, not a guardrail/hook tool. It has
# no PreToolUse/BeforeTool semantics and ships no agent skill, so agent
# auto-configuration is deliberately omitted (unlike dcg/rch which gate or
# offload agent tool calls).

# ============================================================================
# Summary
# ============================================================================
print_summary() {
    [ "$QUIET" -eq 1 ] && return 0
    local installed_version path_status
    installed_version=$(read_installed_version)

    if [[ ":$PATH:" == *":$DEST:"* ]]; then
        path_status="on PATH"
    else
        path_status="NOT on PATH"
    fi

    echo ""
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style \
            --border rounded --border-foreground 82 --padding "1 2" --margin "1 0" \
            "$(gum style --foreground 82 --bold '✓ franken_whisper installed!')" \
            "" \
            "$(gum style --foreground 245 "Version:  $installed_version")" \
            "$(gum style --foreground 245 "Location: $DEST/$BINARY_NAME")" \
            "$(gum style --foreground 245 "PATH:     $path_status")"
    else
        draw_box "0;32" \
            "${GREEN}✓ franken_whisper installed!${NC}" \
            "" \
            "Version:  $installed_version" \
            "Location: $DEST/$BINARY_NAME" \
            "PATH:     $path_status"
    fi
    echo ""

    if [[ ":$PATH:" != *":$DEST:"* ]]; then
        log_warn "Add to PATH: export PATH=\"$DEST:\$PATH\"   (or re-run with --easy-mode)"
        echo ""
    fi

    echo "  Quick start:"
    echo "    fw robot triage"
    echo "    fw models --json"
    echo "    fw doctor --json"
    echo "    fw robot run --input audio.mp3 --backend auto"
    echo "    fw --help"
    if [ "$NO_PULL" -eq 0 ]; then
        echo ""
        echo "  Model policy:"
        echo "    The installer provisioned or verified the native Whisper and"
        echo "    Sortformer packages. Inspect readiness with 'fw models --json'."
        echo ""
    fi
    echo "  Uninstall:"
    echo "    curl -fsSL https://raw.githubusercontent.com/${OWNER}/${REPO}/main/install.sh | bash -s -- --uninstall"
    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    acquire_lock
    print_banner

    TMP=$(mktemp -d "$INSTALLER_TEMP_TEMPLATE")

    # Offline / airgap path: no platform/version resolution needed.
    if [ -n "$OFFLINE_TARBALL" ]; then
        log_step "Install directory: $DEST"
        check_write_permissions
        install_offline
        maybe_add_path
        provision_default_models
        [ "$VERIFY" -eq 1 ] && run_self_test
        print_summary
        return 0
    fi

    detect_platform
    log_step "Platform: $PLATFORM"
    log_step "Install directory: $DEST"

    if [ "$FROM_SOURCE" -eq 0 ]; then
        resolve_version
    fi

    preflight_checks

    # Already-installed short-circuit (binary self-report, marker fallback).
    if already_installed; then
        log_success "franken_whisper $VERSION is already installed at $DEST/$BINARY_NAME"
        log_info "Use --force to reinstall the binary"
        maybe_add_path
        provision_default_models
        [ "$VERIFY" -eq 1 ] && run_self_test
        print_summary
        return 0
    fi

    if [ "$FROM_SOURCE" -eq 0 ] && [ -n "$VERSION" ]; then
        if download_release "$PLATFORM"; then
            :
        else
            die "Binary download or verification failed. Retry, use --offline ARCHIVE, or explicitly choose --from-source."
        fi
    else
        build_from_source
    fi

    maybe_add_path
    provision_default_models
    [ "$VERIFY" -eq 1 ] && run_self_test
    print_summary
}

if [[ "${BASH_SOURCE[0]:-}" == "${0:-}" ]] || [[ -z "${BASH_SOURCE[0]:-}" ]]; then
    main "$@"
fi
