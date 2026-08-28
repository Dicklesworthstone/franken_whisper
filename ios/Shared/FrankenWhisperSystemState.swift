import Foundation

#if !targetEnvironment(macCatalyst)
import ActivityKit
#endif

struct FrankenWhisperRunContentState: Codable, Hashable {
    enum Status: String, Codable, Hashable {
        case preparing
        case activating
        case listening
        case armed
        case decoding
        case speakers
        case fusing
        case complete
        case cancelled
        case failed
    }

    var stage: String
    var detail: String
    var windowsDone: Int
    var windowsTotal: Int
    var emittedSegments: Int
    var elapsedSeconds: Int
    var status: Status
}

#if !targetEnvironment(macCatalyst)
struct FrankenWhisperRunActivityAttributes: ActivityAttributes {
    typealias ContentState = FrankenWhisperRunContentState
    var runID: UUID
    var startedAt: Date
}
#else
struct FrankenWhisperRunActivityAttributes {
    typealias ContentState = FrankenWhisperRunContentState
    var runID: UUID
    var startedAt: Date
}
#endif

struct FrankenWhisperWidgetSnapshot: Codable, Hashable {
    enum Readiness: String, Codable {
        case modelRequired
        case waking
        case ready
        case working
        case complete
        case needsAttention
    }

    var readiness: Readiness
    var headline: String
    var detail: String
    var updatedAt: Date

    static let placeholder = FrankenWhisperWidgetSnapshot(
        readiness: .ready,
        headline: "Observatory ready",
        detail: "Private transcription on this device",
        updatedAt: .now
    )
}

enum FrankenWhisperSharedStore {
    static let suiteName = "group.com.frankenwhisper.dictation"
    private static let widgetKey = "widget.snapshot.v1"
    private static let stagedMediaKey = "share.staged-media.v1"
    private static let requestedActionKey = "intent.requested-action.v1"

    enum RequestedAction: String {
        case transcribe
        case live
    }

    static func loadWidgetSnapshot() -> FrankenWhisperWidgetSnapshot {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let data = defaults.data(forKey: widgetKey),
              let snapshot = try? JSONDecoder().decode(FrankenWhisperWidgetSnapshot.self, from: data)
        else { return .placeholder }
        return snapshot
    }

    static func save(_ snapshot: FrankenWhisperWidgetSnapshot) {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let data = try? JSONEncoder().encode(snapshot)
        else { return }
        defaults.set(data, forKey: widgetKey)
    }

    static func stageMedia(from source: URL, preferredExtension: String? = nil) throws -> URL {
        guard let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: suiteName)
        else { throw CocoaError(.fileNoSuchFile) }
        let directory = container.appendingPathComponent("Incoming", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let ext = preferredExtension ?? (source.pathExtension.isEmpty ? "audio" : source.pathExtension)
        let destination = directory.appendingPathComponent("\(UUID().uuidString).\(ext)")
        try FileManager.default.copyItem(at: source, to: destination)
        UserDefaults(suiteName: suiteName)?.set(destination.lastPathComponent, forKey: stagedMediaKey)
        return destination
    }

    static func consumeStagedMediaURL() -> URL? {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let name = defaults.string(forKey: stagedMediaKey),
              let container = FileManager.default.containerURL(
                forSecurityApplicationGroupIdentifier: suiteName)
        else { return nil }
        defaults.removeObject(forKey: stagedMediaKey)
        return container.appendingPathComponent("Incoming", isDirectory: true).appendingPathComponent(name)
    }

    static func request(_ action: RequestedAction) {
        UserDefaults(suiteName: suiteName)?.set(action.rawValue, forKey: requestedActionKey)
    }

    static func consumeRequestedAction() -> RequestedAction? {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let rawValue = defaults.string(forKey: requestedActionKey)
        else { return nil }
        defaults.removeObject(forKey: requestedActionKey)
        return RequestedAction(rawValue: rawValue)
    }
}
