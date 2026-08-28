import SwiftUI

@main
struct FrankenWhisperApp: App {
    init() {
        // Engine levers are process-wide OnceLock reads on the Rust side, so they
        // must be set before the FIRST engine call (fw-ios/include/fw_ios.h).
        //
        // FW_STREAM_LOAD: pread each tensor from the 874 MB q8_0 file instead of
        // holding it as one resident blob during load — the difference between a
        // load-time peak near 2 GB and one near the resident working set.
        setenv("FW_STREAM_LOAD", "1", 1)
        // FW_LOAD_WORKERS: cap concurrent tensor loaders; each in-flight tensor is
        // an owned pread buffer, so a small cap trades a slower load for a lower
        // peak. Two is plenty against phone NAND.
        setenv("FW_LOAD_WORKERS", "2", 1)
        // Four rayon workers: the performance cores, leaving efficiency cores for
        // UI and the audio session instead of oversubscribing all six.
        #if targetEnvironment(macCatalyst)
        let cores = ProcessInfo.processInfo.activeProcessorCount
        setenv("RAYON_NUM_THREADS", String(max(2, min(10, cores - 2))), 1)
        #else
        setenv("RAYON_NUM_THREADS", "4", 1)
        #endif

        // A finished transcript remains available to the keyboard only for the
        // lifetime of this app session. Do not resurrect dictated text from a
        // prior launch merely because the App Group defaults are persistent.
        DictationBridge.write(.empty)
    }

    var body: some Scene {
        WindowGroup {
            LabView()
                .preferredColorScheme(.dark)
#if targetEnvironment(macCatalyst)
                .frame(minWidth: 760, minHeight: 620)
#endif
        }
#if targetEnvironment(macCatalyst)
        .defaultSize(width: 1220, height: 840)
        .windowResizability(.contentMinSize)
#endif
        .commands { WhisperCommands() }
    }
}

struct WhisperCommandActions {
    let importFile: () -> Void
    let toggleRecording: () -> Void
    let transcribe: () -> Void
    let stop: () -> Void
    let canRecord: Bool
    let canTranscribe: Bool
    let canStop: Bool
}

private struct WhisperCommandKey: FocusedValueKey {
    typealias Value = WhisperCommandActions
}

extension FocusedValues {
    var whisperCommands: WhisperCommandActions? {
        get { self[WhisperCommandKey.self] }
        set { self[WhisperCommandKey.self] = newValue }
    }
}

private struct WhisperCommands: Commands {
    @FocusedValue(\.whisperCommands) private var actions

    var body: some Commands {
        CommandMenu("Transcription") {
            Button("Open Audio…") { actions?.importFile() }
                .keyboardShortcut("o", modifiers: [.command])

            Divider()

            Button("Start or Stop Recording") { actions?.toggleRecording() }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(actions?.canRecord != true)

            Button("Transcribe") { actions?.transcribe() }
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(actions?.canTranscribe != true)

            Divider()

            Button("Stop Transcription") { actions?.stop() }
                .keyboardShortcut(.escape, modifiers: [])
                .disabled(actions?.canStop != true)
        }
    }
}
