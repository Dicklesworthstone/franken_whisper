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
        headline: "Open the Observatory",
        detail: "Transcribe privately on this device",
        updatedAt: .now
    )
}

enum FrankenWhisperSharedStore {
    static let suiteName = "group.com.frankenwhisper.dictation"
    private static let widgetKey = "widget.snapshot.v1"
    private static let stagedMediaKey = "share.staged-media.v1"
    private static let requestedActionKey = "intent.requested-action.v1"

    private static func incomingDirectory(in container: URL) -> URL {
        container.appendingPathComponent("Incoming", isDirectory: true)
    }

    /// Staged names are generated as `<UUID>.<extension>`. Revalidate that
    /// grammar before turning cross-process preference data back into a path:
    /// a merely path-component-safe value such as `.` resolves to the Incoming
    /// directory itself.
    static func isValidStagedMediaName(_ name: String) -> Bool {
        guard name == (name as NSString).lastPathComponent else { return false }
        let nameURL = URL(fileURLWithPath: name)
        let stem = nameURL.deletingPathExtension().lastPathComponent
        let ext = nameURL.pathExtension
        return UUID(uuidString: stem) != nil
            && !ext.isEmpty
            && ext.count <= 16
            && ext.allSatisfy { $0.isLetter || $0.isNumber }
    }

    private static func stagedMediaURL(named name: String, in container: URL) -> URL? {
        guard isValidStagedMediaName(name) else { return nil }
        return incomingDirectory(in: container).appendingPathComponent(name)
    }

    private static func isRegularStagedFile(_ url: URL) -> Bool {
        guard let values = try? url.resourceValues(
            forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        else { return false }
        return values.isRegularFile == true && values.isSymbolicLink != true
    }

    enum RequestedAction: String {
        case transcribe
        case live
    }

    static func loadWidgetSnapshot() -> FrankenWhisperWidgetSnapshot {
        guard let defaults = UserDefaults(suiteName: suiteName) else { return .placeholder }
        defaults.synchronize()
        guard let data = defaults.data(forKey: widgetKey),
              let snapshot = try? JSONDecoder().decode(FrankenWhisperWidgetSnapshot.self, from: data)
        else { return .placeholder }
        return snapshot
    }

    static func save(_ snapshot: FrankenWhisperWidgetSnapshot) {
        guard let defaults = UserDefaults(suiteName: suiteName),
              let data = try? JSONEncoder().encode(snapshot)
        else { return }
        defaults.set(data, forKey: widgetKey)
        defaults.synchronize()
    }

    static func stageMedia(from source: URL, preferredExtension: String? = nil) throws -> URL {
        guard let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: suiteName),
              let defaults = UserDefaults(suiteName: suiteName)
        else { throw CocoaError(.fileNoSuchFile) }
        let directory = incomingDirectory(in: container)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let rawExtension = preferredExtension
            ?? (source.pathExtension.isEmpty ? "audio" : source.pathExtension)
        let cleanedExtension = rawExtension.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined()
        let ext = String(cleanedExtension.prefix(16)).isEmpty
            ? "audio"
            : String(cleanedExtension.prefix(16))
        let destination = directory.appendingPathComponent("\(UUID().uuidString).\(ext)")
        defaults.synchronize()
        let previousName = defaults.string(forKey: stagedMediaKey)
        try FileManager.default.copyItem(at: source, to: destination)
        guard isRegularStagedFile(destination) else {
            try? FileManager.default.removeItem(at: destination)
            throw CocoaError(.fileReadInvalidFileName)
        }
        defaults.set(destination.lastPathComponent, forKey: stagedMediaKey)
        defaults.synchronize()

        // A completed but never-consumed earlier share should not accumulate
        // private media forever. Only unlink a name matching this store's
        // exact UUID grammar; malformed preference data is never a path.
        if let previousName,
           previousName != destination.lastPathComponent,
           let previousURL = stagedMediaURL(named: previousName, in: container),
           isRegularStagedFile(previousURL) {
            try? FileManager.default.removeItem(at: previousURL)
        }
        return destination
    }

    static func consumeStagedMediaURL() -> URL? {
        guard let defaults = UserDefaults(suiteName: suiteName) else { return nil }
        defaults.synchronize()
        guard let name = defaults.string(forKey: stagedMediaKey) else { return nil }
        guard let container = FileManager.default.containerURL(
                  forSecurityApplicationGroupIdentifier: suiteName),
              let stagedURL = stagedMediaURL(named: name, in: container)
        else {
            defaults.removeObject(forKey: stagedMediaKey)
            defaults.synchronize()
            return nil
        }
        defaults.removeObject(forKey: stagedMediaKey)
        defaults.synchronize()
        guard isRegularStagedFile(stagedURL) else { return nil }
        return stagedURL
    }

    /// Revoke a share-extension handoff that the user cancelled before opening
    /// the app. Only remove the pointer when it still names this exact file; a
    /// newer share must not be invalidated by an older controller finishing.
    static func discardStagedMedia(at url: URL) {
        guard let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: suiteName),
              let validatedURL = stagedMediaURL(named: url.lastPathComponent, in: container),
              validatedURL.standardizedFileURL == url.standardizedFileURL
        else { return }
        if let defaults = UserDefaults(suiteName: suiteName) {
            defaults.synchronize()
            if defaults.string(forKey: stagedMediaKey) == validatedURL.lastPathComponent {
                defaults.removeObject(forKey: stagedMediaKey)
                defaults.synchronize()
            }
        }
        if isRegularStagedFile(validatedURL) {
            try? FileManager.default.removeItem(at: validatedURL)
        }
    }

    static func request(_ action: RequestedAction) {
        guard let defaults = UserDefaults(suiteName: suiteName) else { return }
        defaults.set(action.rawValue, forKey: requestedActionKey)
        defaults.synchronize()
    }

    static func consumeRequestedAction() -> RequestedAction? {
        guard let defaults = UserDefaults(suiteName: suiteName) else { return nil }
        defaults.synchronize()
        guard let rawValue = defaults.string(forKey: requestedActionKey)
        else { return nil }
        defaults.removeObject(forKey: requestedActionKey)
        defaults.synchronize()
        return RequestedAction(rawValue: rawValue)
    }
}
