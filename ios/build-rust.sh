#!/usr/bin/env bash
# Builds the Rust engine for iOS device + simulator and assembles FwCore.xcframework.
# Run before the first Xcode build and after any Rust change:  ios/build-rust.sh
#
# fw-ios is a STANDALONE crate (not a workspace member, same as fw-wasm), so the
# build runs from fw-ios/ and its default target dir is fw-ios/target.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-fw-ios/target}"
PROFILE="${FW_IOS_PROFILE:-release}"

for target in aarch64-apple-ios aarch64-apple-ios-sim; do
  rustup target list --installed | grep -q "$target" || rustup target add "$target"
  (cd fw-ios && nice -n 19 cargo build --"$PROFILE" --locked -j 4 --target "$target")
done

HEADERS=$(mktemp -d /tmp/fw-ios-headers.XXXXXX)
trap 'rm -rf "$HEADERS"' EXIT
cp fw-ios/include/fw_ios.h "$HEADERS/"
cat > "$HEADERS/module.modulemap" <<'EOF'
module FwCore {
    header "fw_ios.h"
    export *
}
EOF

FRAMEWORK=ios/FwCore.xcframework
rm -rf "$FRAMEWORK"
xcodebuild -create-xcframework \
  -library "$TARGET_DIR/aarch64-apple-ios/$PROFILE/libfw_ios.a" -headers "$HEADERS" \
  -library "$TARGET_DIR/aarch64-apple-ios-sim/$PROFILE/libfw_ios.a" -headers "$HEADERS" \
  -output "$FRAMEWORK"

du -sh "$FRAMEWORK"
echo "built $FRAMEWORK"
echo "next: (cd ios && xcodegen generate) if project.yml changed, then build in Xcode"
