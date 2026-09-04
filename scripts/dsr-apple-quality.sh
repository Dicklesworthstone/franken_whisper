#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root/ios"

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
