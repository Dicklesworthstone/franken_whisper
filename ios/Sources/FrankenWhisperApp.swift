import SwiftUI
import UIKit

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
        // Keep an explicit launch override for physical-device A/Bs. OnceLock
        // reads this value on first engine use, so overwriting a supplied value
        // here would silently compare the same arm twice.
        if ProcessInfo.processInfo.environment["RAYON_NUM_THREADS"] == nil {
            #if targetEnvironment(macCatalyst)
            let cores = ProcessInfo.processInfo.activeProcessorCount
            setenv("RAYON_NUM_THREADS", String(max(2, min(10, cores - 2))), 1)
            #else
            setenv("RAYON_NUM_THREADS", "4", 1)
            #endif
        }
        if ProcessInfo.processInfo.environment["FW_IOS_PROFILE"] == "1" {
            // Repeated identical fixture runs must perform physical inference;
            // the production cache remains enabled outside this hidden lane.
            setenv("FW_TRANSCRIPT_CACHE", "0", 1)
        }

        // A finished transcript remains available to the keyboard only for the
        // lifetime of this app session. Do not resurrect dictated text from a
        // prior launch merely because the App Group defaults are persistent.
        DictationBridge.write(.empty)
    }

    var body: some Scene {
        WindowGroup {
            LabView()
                .preferredColorScheme(.dark)
                .background(CatalystWindowFreedom())
#if targetEnvironment(macCatalyst)
                .frame(minWidth: 480, minHeight: 420)
#endif
        }
#if targetEnvironment(macCatalyst)
        .defaultSize(width: 1220, height: 840)
        .windowResizability(.contentMinSize)
#endif
        .commands { WhisperCommands() }
    }
}

private struct CatalystWindowFreedom: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> Controller { Controller() }
    func updateUIViewController(_ controller: Controller, context: Context) { controller.configure() }

    final class Controller: UIViewController {
        override func viewDidAppear(_ animated: Bool) {
            super.viewDidAppear(animated)
            configure()
        }

        override func viewDidLayoutSubviews() {
            super.viewDidLayoutSubviews()
            configure()
        }

        func configure() {
#if targetEnvironment(macCatalyst)
            guard let restrictions = view.window?.windowScene?.sizeRestrictions else { return }
            restrictions.minimumSize = CGSize(width: 480, height: 420)
            restrictions.maximumSize = CGSize(width: 10_000, height: 10_000)
#endif
        }
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
