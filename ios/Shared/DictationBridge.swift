import Foundation

/// The tiny snapshot handoff between the containing app and its keyboard. The
/// app is the sole snapshot writer; the keyboard writes only the separate
/// command value below and never sees microphone samples or model files.
/// Apple requires the keyboard's Full Access switch for this shared container
/// even though no data is sent over a network. One encoded value makes each
/// cross-process snapshot atomic.
struct DictationSnapshot: Codable, Equatable {
    enum State: String, Codable {
        case idle
        /// The containing app has a time-bounded background microphone
        /// session alive. The keyboard can start locally without opening it.
        case armed
        case listening
        case finishing
        case failed
    }

    var sessionID: String
    var state: State
    var text: String
    var revision: Int
    var message: String?
    /// Normalized microphone energy and frequency bands. These are derived
    /// locally from the current capture window and exist only to animate the
    /// keyboard while it is actively listening.
    var level: Float?
    var spectrum: [Float]?
    var expiresAt: TimeInterval?
    var updatedAt: TimeInterval

    static let empty = DictationSnapshot(
        sessionID: "", state: .idle, text: "", revision: 0,
        message: nil, level: nil, spectrum: nil, expiresAt: nil, updatedAt: 0)
}

/// A tiny control lane in the opposite direction. Once the containing app has
/// visibly activated a time-bounded microphone session, the keyboard can start
/// and stop utterances without another app switch. Initial microphone
/// activation still happens in the containing app because iOS never grants a
/// custom keyboard microphone access.
struct DictationCommand: Codable, Equatable {
    enum Action: String, Codable {
        case start
        case stop
        case endSession = "end_session"
    }

    var id: String
    var action: Action
    var createdAt: TimeInterval
}

enum DictationBridge {
    static let appGroup = "group.com.frankenwhisper.dictation"
    private static let snapshotKey = "dictation.snapshot.v1"
    private static let commandKey = "dictation.command.v1"

    static func read() -> DictationSnapshot {
        guard let defaults = UserDefaults(suiteName: appGroup) else { return .empty }
        defaults.synchronize()
        guard let data = defaults.data(forKey: snapshotKey),
              let snapshot = try? JSONDecoder().decode(DictationSnapshot.self, from: data)
        else { return .empty }
        return snapshot
    }

    static func write(_ snapshot: DictationSnapshot) {
        guard let defaults = UserDefaults(suiteName: appGroup),
              let data = try? JSONEncoder().encode(snapshot)
        else { return }
        defaults.set(data, forKey: snapshotKey)
        defaults.synchronize()
    }

    static func readCommand() -> DictationCommand? {
        guard let defaults = UserDefaults(suiteName: appGroup) else { return nil }
        // Match snapshot reads: synchronize before fetching so the containing
        // app observes the extension's newest command, not a value captured
        // before cross-process preferences were refreshed.
        defaults.synchronize()
        guard let data = defaults.data(forKey: commandKey) else { return nil }
        return try? JSONDecoder().decode(DictationCommand.self, from: data)
    }

    static func writeCommand(_ action: DictationCommand.Action) {
        guard let defaults = UserDefaults(suiteName: appGroup),
              let data = try? JSONEncoder().encode(
                  DictationCommand(
                      id: UUID().uuidString,
                      action: action,
                      createdAt: Date().timeIntervalSince1970))
        else { return }
        defaults.set(data, forKey: commandKey)
        defaults.synchronize()
    }
}
