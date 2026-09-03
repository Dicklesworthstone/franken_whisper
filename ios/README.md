# FrankenWhisper for iOS

The browser playground as a native SwiftUI app: download the models once, then
record or import audio or video and get a speaker-attributed transcript entirely
on device — the same pure-Rust engine (`src/native_engine`) the CLI and the
website run, compiled for the phone through the `fw-ios` boundary crate.

Videos from Photos, Files, and the Share sheet keep their pixels on the device.
AVFoundation extracts a local audio track for the Rust engine; the original
video is retained only in the app's purgeable cache. Decoder-provided DTW word
alignment powers an optional Subtitle Studio with a live preview, four polished
starting looks, font/size/text/highlight/backdrop controls, and word-level
karaoke glow/pop animation. The burned H.264/QuickTime export can be shared or
saved to Photos. FrankenWhisper refuses to fabricate karaoke timing when a
decoder result does not contain real alignment data.

## Building

```bash
# 1. Build the Rust engine for device + simulator and assemble FwCore.xcframework
ios/build-rust.sh

# 2. Generate the Xcode project (only needed after project.yml changes)
cd ios && xcodegen generate

# 3. Build/run
xcodebuild -project FrankenWhisper.xcodeproj -scheme FrankenWhisper \
  -destination "generic/platform=iOS Simulator" CODE_SIGNING_ALLOWED=NO build
# or open FrankenWhisper.xcodeproj in Xcode and run on a device/simulator.
```

Requirements: Xcode with the iOS platform installed, `xcodegen` (brew), and the
`aarch64-apple-ios` / `aarch64-apple-ios-sim` Rust targets (the script adds
them). Simulator builds are arm64-only (Apple Silicon hosts); an Intel host
would need the `x86_64-apple-ios` Rust target added to `build-rust.sh`.

## App Store archive

```bash
xcodebuild -project ios/FrankenWhisper.xcodeproj -scheme FrankenWhisper \
  -configuration Release -destination "generic/platform=iOS" \
  -archivePath "$PWD/ios/build/FrankenWhisper.xcarchive" archive

xcodebuild -exportArchive \
  -archivePath "$PWD/ios/build/FrankenWhisper.xcarchive" \
  -exportPath "$PWD/ios/build/export" \
  -exportOptionsPlist ios/AppStoreExportOptions.plist
```

The export configuration preserves the checked-in marketing version and build
number instead of letting App Store Connect renumber the binary.

## Live dictation keyboard (1.1)

The containing app now has a low-latency dictation lane backed by the pinned
74 MB multilingual Whisper tiny package. It keeps an explicitly activated,
visible microphone session alive for at most one hour, splits speech at
natural pauses, and decodes each bounded phrase through the same local Rust
engine. A bundled custom keyboard reads only append-only committed text from an
App Group and inserts it into the current text field. While listening, the app
also publishes a normalized level plus 14 real frequency bands so the keyboard
can render the captured signal without exposing raw microphone samples.

iOS does not allow keyboard extensions to use the microphone. The first Start
therefore opens FrankenWhisper for one visible activation; after the user
returns, subsequent Start/Finish commands stay in the keyboard for the life of
that one-hour session. Between utterances the microphone remains visibly
active but incoming samples are discarded. The keyboard requires Full Access
because Apple gates App Group access behind that switch. It uses the expanded
sandbox only for this local command/transcript handoff: the extension contains
no network client and never receives audio or model bytes. Background audio
mode exists solely to keep that explicit recording session alive across apps.

## How it hangs together

- **`fw-ios/`** (repo root) is the boundary crate: a `staticlib` that mounts
  the parent's engine sources by `#[path]` (the fw-wasm recipe) and exposes
  the C ABI documented in `fw-ios/include/fw_ios.h` — engine open/close,
  Sortformer + denoiser load, PCM/file staging, the fused run, progress and
  live-transcript callbacks, cooperative cancel.
- **Models are downloaded, never bundled** (`Sources/ModelStore.swift`):
  Whisper large-v3-turbo **q8_0** (874 MB, the browser lane's
  transcript-identical quantization), multilingual Whisper **tiny f16** (74 MB,
  realtime/keyboard speed lane), the Sortformer diarizer package (492 MB +
  receipt), and the FastEnhancer denoiser (838 KB) — resumable
  32 MiB ranged downloads, SHA-256 pinned to `site/model-manifest.js`,
  stored in Application Support, excluded from iCloud backup.
- **Memory**: the q8_0 weights stay block-resident (the `target_os = "ios"`
  arms of the wasm q8 lane in `src/native_engine/{encoder,decoder}.rs`), and
  `FW_STREAM_LOAD=1` (set in `FrankenWhisperApp.init`) preads tensors at load
  instead of holding the file as a blob. Working set is roughly 0.9 GB
  (whisper) + 0.5 GB (Sortformer); the entitlement requests the increased
  memory limit, which needs that capability on your signing profile for
  device runs.
- **Two accuracy classes**: ordinary transcription never changes model or
  options; it remains on large-v3-turbo with optional denoising, timestamps,
  and Sortformer. Realtime/keyboard work uses a separate resident tiny handle,
  greedy text-only decode, and no diarization. `FW_IOS_LIVE_FAST_MODEL=0` is
  the diagnostic kill switch that routes live work back through the established
  large-model path.
- **Live transcript**: the engine's per-window segment feed and span
  heartbeat (`src/native_engine/plat.rs`, `ios_hooks`) stream through the
  C callbacks into the UI, so text appears window by window and the progress
  bar counts real `encoder_window` events.
- **Speaker names, like the website**: an optional pre-run names field feeds
  Whisper's decoding prompt (so names come out spelled right) and then labels
  the detected `SPEAKER_NN` lanes in order of first appearance; after a run,
  each voice can be renamed in place. Names flow into every export.
- **Transcribe or translate**: the batch workspace exposes the native Whisper
  task directly. Source-language transcription remains the default; Translate
  to English is an explicit opt-in, runs through the same local Rust decoder,
  and is labeled in the result and human-readable exports.
- **Exports**: a styled self-contained HTML page and GitHub-flavored Markdown
  (both matching the browser demo's exports), plus plain text; SRT and JSON
  live behind the "More" menu. Video results additionally offer an entirely
  local caption burn-in with real word-timed karaoke animation.
- **Capture** is an `AVAudioEngine` tap converted to 16 kHz mono f32
  (`Sources/AudioRecorder.swift`); imported audio decodes in Rust via the same
  symphonia path the browser uses, while imported video audio is extracted by
  AVFoundation into a cache-scoped M4A before entering that same engine.

## Notes

- `FwCore.xcframework` and `FrankenWhisper.xcodeproj` are generated. The
  `project.yml`, Swift sources, privacy manifest, app-icon catalog, entitlement,
  and App Store export options are the reviewable release sources.
- On-device speed on A17/A18-class hardware is unmeasured; the app shows the
  measured wall time and RTF after each run and claims nothing else.
- The Sortformer diarizer is development-uncertified upstream and capped at
  four anonymous lanes; the app repeats that caveat in the result footer and
  never presents a lane count as a true speaker count.
- Nothing leaves the device after the one-time model download: no accounts,
  no telemetry, no cloud.
