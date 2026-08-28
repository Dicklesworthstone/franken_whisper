#!/usr/bin/env bash
# Builds the Rust engine for iOS device + simulator + Mac Catalyst and assembles
# FwCore.xcframework.
# Run before the first Xcode build and after any Rust change:  ios/build-rust.sh
#
# fw-ios is a STANDALONE crate (not a workspace member, same as fw-wasm), so the
# build runs from fw-ios/ and its default target dir is fw-ios/target.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-fw-ios/target}"
PROFILE="${FW_IOS_PROFILE:-release}"
APPLE_RUST_TOOLCHAIN="${APPLE_RUST_TOOLCHAIN:-nightly-2026-08-25-aarch64-apple-darwin}"
APPLE_CARGO="${APPLE_CARGO:-$(rustup which --toolchain "$APPLE_RUST_TOOLCHAIN" cargo)}"

for target in \
  aarch64-apple-ios \
  aarch64-apple-ios-sim \
  aarch64-apple-ios-macabi \
  x86_64-apple-ios-macabi
do
  rustup target list --toolchain "$APPLE_RUST_TOOLCHAIN" --installed | grep -qx "$target" || \
    rustup target add --toolchain "$APPLE_RUST_TOOLCHAIN" "$target"
  (
    cd fw-ios
    RUSTUP_TOOLCHAIN="$APPLE_RUST_TOOLCHAIN" RCH_CARGO_WRAPPER_BYPASS=1 \
      nice -n 19 "$APPLE_CARGO" build --"$PROFILE" --locked -j 4 --target "$target"
  )
done

HEADERS=$(mktemp -d /tmp/fw-ios-headers.XXXXXX)
cp fw-ios/include/fw_ios.h fw-ios/include/module.modulemap "$HEADERS/"

CATALYST_ROOT=$(mktemp -d /tmp/fw-ios-maccatalyst.XXXXXX)
CATALYST_LIB="$CATALYST_ROOT/libfw_ios.a"
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-macabi/$PROFILE/libfw_ios.a" \
  "$TARGET_DIR/x86_64-apple-ios-macabi/$PROFILE/libfw_ios.a" \
  -output "$CATALYST_LIB"

FRAMEWORK=ios/FwCore.xcframework
OUTPUT_ROOT=$(mktemp -d /tmp/fw-xcframework.XXXXXX)
STAGED_FRAMEWORK="$OUTPUT_ROOT/FwCore.xcframework"
xcodebuild -create-xcframework \
  -library "$TARGET_DIR/aarch64-apple-ios/$PROFILE/libfw_ios.a" -headers "$HEADERS" \
  -library "$TARGET_DIR/aarch64-apple-ios-sim/$PROFILE/libfw_ios.a" -headers "$HEADERS" \
  -library "$CATALYST_LIB" -headers "$HEADERS" \
  -output "$STAGED_FRAMEWORK"

if [[ -e "$FRAMEWORK" ]]; then
  mv "$FRAMEWORK" "$FRAMEWORK.previous-$(date +%Y%m%d-%H%M%S)"
fi
mv "$STAGED_FRAMEWORK" "$FRAMEWORK"

du -sh "$FRAMEWORK"
echo "built $FRAMEWORK"
echo "next: (cd ios && xcodegen generate) if project.yml changed, then build in Xcode"
