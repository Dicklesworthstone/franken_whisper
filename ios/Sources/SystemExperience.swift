import AppIntents
import Foundation
import WidgetKit

#if canImport(ActivityKit) && !targetEnvironment(macCatalyst)
import ActivityKit

@MainActor
final class WhisperActivityController {
    static let shared = WhisperActivityController()
    private var activity: Activity<FrankenWhisperRunActivityAttributes>?
    private var startedAt: Date?

    private init() {}

    func transition(
        to runState: LabModel.RunState,
        elapsed: TimeInterval,
        emittedSegments: Int
    ) {
        switch runState {
        case .idle:
            end(status: .cancelled, headline: "Transcription stopped", detail: "Input remains on this device")
        case .staging:
            begin()
        case .running(let done, let total, let stage):
            let status: FrankenWhisperRunContentState.Status
            if stage.contains("speaker") { status = .speakers }
            else if stage.contains("fus") { status = .fusing }
            else { status = .decoding }
            let state = FrankenWhisperRunContentState(
                stage: stage.capitalized,
                detail: detail(stage: stage, windowsDone: done, windowsTotal: total, segments: emittedSegments),
                windowsDone: done,
                windowsTotal: total,
                emittedSegments: emittedSegments,
                elapsedSeconds: max(0, Int(elapsed)),
                status: status
            )
            if let activity {
                Task { await activity.update(ActivityContent(state: state, staleDate: nil)) }
            }
            publish(.working, headline: state.stage, detail: state.detail)
        case .done:
            end(status: .complete, headline: "Transcript assembled", detail: "Ready to read and export")
        case .failed:
            end(status: .failed, headline: "Signal interrupted", detail: "Open FrankenWhisper to retry")
        }
    }

    func transitionLive(
        to state: LabModel.LiveDictationState,
        sessionMinutesRemaining: Int,
        queuedPhrases: Int,
        hasText: Bool
    ) {
        switch state {
        case .idle:
            end(
                status: hasText ? .complete : .cancelled,
                headline: hasText ? "Dictation delivered" : "Dictation session ended",
                detail: hasText ? "Your private text is ready in the keyboard" : "No audio or text was retained"
            )
        case .starting(let stage):
            beginLive(stage: stage)
        case .armed:
            updateLive(
                status: .armed,
                headline: "Keyboard ready",
                detail: "Private mic session armed · \(sessionMinutesRemaining) min remaining",
                queuedPhrases: queuedPhrases
            )
        case .listening:
            updateLive(
                status: .listening,
                headline: "Listening locally",
                detail: queuedPhrases > 0
                    ? "\(queuedPhrases) phrase\(queuedPhrases == 1 ? "" : "s") decoding on this device"
                    : "Speak naturally · pause briefly to commit a phrase",
                queuedPhrases: queuedPhrases
            )
        case .finishing:
            updateLive(
                status: .fusing,
                headline: "Finishing the phrase",
                detail: "Decoding and preparing private keyboard insertion",
                queuedPhrases: queuedPhrases
            )
        case .failed:
            end(status: .failed, headline: "Dictation interrupted", detail: "Open FrankenWhisper to retry")
        }
    }

    private func begin() {
        guard activity == nil, startedAt == nil else { return }
        let start = Date.now
        startedAt = start
        let attributes = FrankenWhisperRunActivityAttributes(runID: UUID(), startedAt: start)
        let state = FrankenWhisperRunContentState(
            stage: "Preparing the signal",
            detail: "Reading audio into private on-device memory",
            windowsDone: 0,
            windowsTotal: 0,
            emittedSegments: 0,
            elapsedSeconds: 0,
            status: .preparing
        )
        if ActivityAuthorizationInfo().areActivitiesEnabled {
            activity = try? Activity.request(
                attributes: attributes,
                content: ActivityContent(state: state, staleDate: nil),
                pushType: nil
            )
        }
        publish(.working, headline: state.stage, detail: state.detail)
    }

    private func beginLive(stage: String) {
        if activity == nil, startedAt == nil {
            let start = Date.now
            startedAt = start
            let attributes = FrankenWhisperRunActivityAttributes(runID: UUID(), startedAt: start)
            let state = FrankenWhisperRunContentState(
                stage: "Activating the microphone",
                detail: stage,
                windowsDone: 0,
                windowsTotal: 0,
                emittedSegments: 0,
                elapsedSeconds: 0,
                status: .activating
            )
            if ActivityAuthorizationInfo().areActivitiesEnabled {
                activity = try? Activity.request(
                    attributes: attributes,
                    content: ActivityContent(state: state, staleDate: nil),
                    pushType: nil
                )
            }
        }
        updateLive(status: .activating, headline: "Activating the microphone", detail: stage, queuedPhrases: 0)
    }

    private func updateLive(
        status: FrankenWhisperRunContentState.Status,
        headline: String,
        detail: String,
        queuedPhrases: Int
    ) {
        let state = FrankenWhisperRunContentState(
            stage: headline,
            detail: detail,
            windowsDone: 0,
            windowsTotal: 0,
            emittedSegments: queuedPhrases,
            elapsedSeconds: max(0, Int(Date().timeIntervalSince(startedAt ?? .now))),
            status: status
        )
        if let activity {
            Task { await activity.update(ActivityContent(state: state, staleDate: nil)) }
        }
        publish(.working, headline: headline, detail: detail)
    }

    private func end(status: FrankenWhisperRunContentState.Status, headline: String, detail: String) {
        guard activity != nil || startedAt != nil else { return }
        let current = activity
        let start = current?.attributes.startedAt ?? startedAt ?? .now
        activity = nil
        startedAt = nil
        let state = FrankenWhisperRunContentState(
            stage: headline,
            detail: detail,
            windowsDone: 0,
            windowsTotal: 0,
            emittedSegments: 0,
            elapsedSeconds: max(0, Int(Date().timeIntervalSince(start))),
            status: status
        )
        let dismissal: ActivityUIDismissalPolicy = status == .complete ? .after(.now + 45) : .immediate
        if let current {
            Task { await current.end(ActivityContent(state: state, staleDate: nil), dismissalPolicy: dismissal) }
        }
        let readiness: FrankenWhisperWidgetSnapshot.Readiness
        switch status {
        case .complete: readiness = .complete
        case .failed: readiness = .needsAttention
        default: readiness = .ready
        }
        publish(readiness, headline: headline, detail: detail)
    }

    private func detail(stage: String, windowsDone: Int, windowsTotal: Int, segments: Int) -> String {
        if stage.contains("speaker") { return "Assigning voices to \(segments) emitted phrases" }
        if stage.contains("fus") { return "Aligning transcript and speaker timelines" }
        return "\(windowsDone) of \(windowsTotal) real audio windows · \(segments) phrases"
    }

    private func publish(
        _ readiness: FrankenWhisperWidgetSnapshot.Readiness,
        headline: String,
        detail: String
    ) {
        FrankenWhisperSharedStore.save(
            FrankenWhisperWidgetSnapshot(
                readiness: readiness,
                headline: headline,
                detail: detail,
                updatedAt: .now
            )
        )
        WidgetCenter.shared.reloadTimelines(ofKind: "FrankenWhisperObservatoryWidget")
    }
}
#else
@MainActor
final class WhisperActivityController {
    static let shared = WhisperActivityController()
    private init() {}
    func transition(to runState: LabModel.RunState, elapsed: TimeInterval, emittedSegments: Int) {}
    func transitionLive(
        to state: LabModel.LiveDictationState,
        sessionMinutesRemaining: Int,
        queuedPhrases: Int,
        hasText: Bool
    ) {}
}
#endif

struct NewTranscriptionIntent: AppIntent {
    static let title: LocalizedStringResource = "New Transcription"
    static let description = IntentDescription("Open FrankenWhisper to record or import audio privately.")
    static let openAppWhenRun = true
    @MainActor func perform() async throws -> some IntentResult {
        FrankenWhisperSharedStore.request(.transcribe)
        return .result()
    }
}

struct OpenLiveDictationIntent: AppIntent {
    static let title: LocalizedStringResource = "Open Live Dictation"
    static let description = IntentDescription(
        "Open the low-latency dictation observatory. Microphone capture starts only after you tap Record."
    )
    static let openAppWhenRun = true
    @MainActor func perform() async throws -> some IntentResult {
        FrankenWhisperSharedStore.request(.live)
        return .result()
    }
}

struct FrankenWhisperShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: NewTranscriptionIntent(),
            phrases: ["Transcribe audio with \(.applicationName)"],
            shortTitle: "New Transcription",
            systemImageName: "waveform"
        )
        AppShortcut(
            intent: OpenLiveDictationIntent(),
            phrases: ["Open live dictation in \(.applicationName)"],
            shortTitle: "Live Dictation",
            systemImageName: "waveform.badge.mic"
        )
    }
}
