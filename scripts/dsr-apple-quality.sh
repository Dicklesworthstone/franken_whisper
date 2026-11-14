#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root/ios"

# Sandbox microphone access and Hardened Runtime audio input are separate
# capabilities. Missing the former suppresses the permission prompt (#20).
# This source check does not prove the signed app's entitlements or TCC UI.
python3 - <<'PY'
from pathlib import Path
import plistlib

with Path("FrankenWhisper-Mac.entitlements").open("rb") as source:
    entitlements = plistlib.load(source)
for key in (
    "com.apple.security.app-sandbox",
    "com.apple.security.device.microphone",
    "com.apple.security.device.audio-input",
):
    if entitlements.get(key) is not True:
        raise SystemExit(f"Mac microphone packaging: {key} must be true")

with Path("Sources/Info.plist").open("rb") as source:
    description = plistlib.load(source).get("NSMicrophoneUsageDescription")
if not isinstance(description, str) or not description.strip():
    raise SystemExit("Microphone packaging: NSMicrophoneUsageDescription must be a nonempty string")
print("Microphone source entitlements and usage description passed")
PY

build_root="${FRANKEN_APPLE_BUILD_ROOT:-${DSR_QUALITY_RUN_DIR:-$repo_root/ios/build/dsr-apple-quality}}"
mkdir -p "$build_root"
sbh check --need 20G "$build_root"
command -v xcodegen >/dev/null
xcodegen generate --spec project.yml
git diff --exit-code -- FrankenWhisper.xcodeproj Sources/Info.plist
git ls-files -z -- '*.swift' | xargs -0 xcrun swiftc -parse
plutil -lint Sources/Info.plist
plutil -lint Sources/PrivacyInfo.xcprivacy
/Users/jemanuel/.local/bin/ensure-simulator-audio-safe prepare
xcodebuild -project FrankenWhisper.xcodeproj -scheme FrankenWhisper \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath "$build_root/derived-data" \
  CODE_SIGNING_ALLOWED=NO build
xcodebuild -project FrankenWhisper.xcodeproj -scheme FrankenWhisper \
  -destination 'platform=macOS,variant=Mac Catalyst' \
  -derivedDataPath "$build_root/derived-data" \
  CODE_SIGNING_ALLOWED=NO test -only-testing:FrankenWhisperTests

simulator_id="${FW_IOS_SIMULATOR_ID:-}"
if [[ -z "$simulator_id" ]]; then
  simulator_id="$({ xcrun simctl list devices available || true; } | awk -F '[()]' '
    /iPhone/ && /\(Booted\)$/ { print $2; found = 1; exit }
    /iPhone/ && fallback == "" { fallback = $2 }
    END { if (!found) print fallback }
  ')"
fi
if [[ -z "$simulator_id" ]]; then
  echo "No available iPhone Simulator for FrankenWhisper UI tests" >&2
  exit 1
fi

# This lane may boot a shut-down device. Re-prove both audio safety layers
# immediately before the simulator action and preserve existing windows.
/Users/jemanuel/.local/bin/ensure-simulator-audio-safe prepare
xcodebuild -project FrankenWhisper.xcodeproj -scheme FrankenWhisper \
  -destination "platform=iOS Simulator,id=$simulator_id" \
  -derivedDataPath "$build_root/derived-data" \
  CODE_SIGNING_ALLOWED=NO test \
  -only-testing:FrankenWhisperUITests/FrankenWhisperAppearanceUITests \
  -only-testing:FrankenWhisperUITests/SubtitleBurnInUITests/testTranslateToEnglishTaskIsDiscoverableAndOptIn \
  -only-testing:FrankenWhisperUITests/SubtitleBurnInUITests/testAdvancedPromptAndBeamControlsAreDiscoverableWithoutAudio
