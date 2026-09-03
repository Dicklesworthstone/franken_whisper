# FrankenWhisper iPhone, iPad, and Mac Excellence Plan

Status: active

Source fence when opened: `b46e551a5e83e2102393b285e090f079d36fe206`

This plan preserves the existing full-accuracy transcription path while making
Live and Keyboard modes markedly faster, clearer, and more native. The
cross-product requirement ledger lives in the active Codex workspace's
`FRANKENSUITE_APP_EXCELLENCE_MASTER_TODO.md`.

## Product outcome

FrankenWhisper becomes a premium private acoustic instrument: high-accuracy
record/file transcription remains intact, while Live and Keyboard workflows
use a deliberately low-latency path and a concise real spectrum/level display.
The app gets an event-driven Electro-Acoustic Observatory, an adaptive iPad
workspace, and an optimized Mac Catalyst transcription studio.

## Execution checklist

Current source tranche (2026-08-28): the real-data Electro-Acoustic
Observatory, focused iPhone destinations, reactive monster, adaptive desktop
workspace, focused Mac commands/import/drop, Catalyst Rust-slice script, App
Group extension, privacy-safe widget, batch and live-dictation Live
Activity/Dynamic Island, App Intents, deep links, and audio/video share staging
are implemented in source. Photos/Files video import, local audio extraction,
real DTW word-aligned karaoke preview, customizable local subtitle burn-in,
share, Photos save, and an explicit native Translate to English batch task are
also implemented. The pre-extension simulator build was green. YAML,
plist, privacy-manifest, and diff hygiene checks are green. Regeneration,
extension compilation, universal framework completion, Mac launch, and device
acceptance remain open.

- [ ] Inventory existing engine spans, partial-segment callbacks, audio levels, spectrum values, diarization events, and fusion events.
- [ ] Define a stable Swift run-state model that maps only to those real events.
- [ ] Keep full-accuracy transcription settings and quality unchanged.
- [ ] Tune Live/Keyboard mode independently for startup and incremental latency.
- [ ] Keep realtime model warm automatically after download while foreground-ready.
- [ ] Separate microphone acquisition, recording, decoding, insertion, completion, cancellation, and error states.
- [ ] Keep keyboard copy concise and allow sessions longer than fifteen minutes within an explicit safe limit.
- [ ] Preserve dictated text when host insertion fails and provide a direct retry/recovery path.
- [ ] Build shared semantic theme, panels, controls, telemetry, machine disclosure, and adaptive workspace primitives.
- [ ] Restructure into Record, Live, Keyboard, and Library destinations.
- [ ] Build the Electro-Acoustic Observatory from actual FFT, window, segment, diarization, and fusion data.
- [ ] Keep partial transcript legible while the hero view animates.
- [ ] Make speaker lanes accessible without relying on color alone.
- [ ] Add selectable transcript actions, speaker rename, find, copy, share, and export.
- [ ] Add private local history with retention, deletion, redaction, and optional Spotlight indexing.
- [ ] Add responsive iPad sidebar/workspace/inspector layout.
- [ ] Add Mac Catalyst support and Catalyst-compatible Rust library slice.
- [ ] Exclude the iOS keyboard extension from Mac while preserving app functionality.
- [ ] Optimize for the Mac idiom with source list, transcript workspace, inspector, menus, shortcuts, drag/drop, exports, resizable windows, and multiple sessions.
- [ ] Extend the App Group staging schema without breaking installed keyboards.
- [ ] Add widgets, Live Activity/Dynamic Island, Transcribe Audio/Open Live Dictation App Intents, control, audio/video share extension, quick actions, and Handoff.
- [ ] Keep large models out of widget, share, and keyboard extension processes.
- [ ] Add accessibility, Reduce Motion, Reduce Transparency, Low Power, thermal throttling, and bounded animation cadence.
- [ ] Add coherent dark/tinted app-icon and widget marks.
- [ ] Regenerate Xcode, run focused Rust/Swift tests, and build app, keyboard, simulator, iPad, and Mac Catalyst targets.
- [ ] Install on the connected iPhone and exercise full accuracy, Live, Keyboard in Notes, long session, cancellation, result recovery, widget, activity, intent, and share-extension routes.
- [ ] Reconcile App Store copy so low-latency and full-accuracy modes are represented truthfully.

## Acceptance boundary

The work is complete only when the keyboard gives immediate causal feedback,
handoff/insertion is recoverable and as native as iOS permits, the processing
view is driven by real acoustic/model data, the Mac target works as a desktop
studio, and installed-device acceptance covers both Live and full-accuracy
paths.
