import Foundation

/// The tiny, append-only handoff between the containing app and its keyboard.
/// The app is the sole writer; the keyboard has read-only App Group access and
/// never sees microphone samples or model files. Apple requires the keyboard's
/// Full Access switch for this shared container even though no data is sent
/// over a network. One encoded value makes each cross-process snapshot atomic.
struct DictationSnapshot: Codable, Equatable {
    enum State: String, Codable {
        case idle
        case listening
        case finishing
        case failed
    }

    var sessionID: String
    var state: State
    var text: String
    var revision: Int
    var message: String?
    var updatedAt: TimeInterval

    static let empty = DictationSnapshot(
        sessionID: "", state: .idle, text: "", revision: 0,
        message: nil, updatedAt: 0)
}

enum DictationBridge {
    static let appGroup = "group.com.frankenwhisper.dictation"
    private static let snapshotKey = "dictation.snapshot.v1"

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
}
