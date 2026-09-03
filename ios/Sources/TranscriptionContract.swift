import Foundation

// Values shared by the Swift UI, transcript exporters, tests, and the
// `fw_run_prepared` JSON boundary documented in fw_ios.h.

struct TranscriptSegment: Codable, Identifiable, Hashable {
    var startSec: Double?
    var endSec: Double?
    var text: String
    var speaker: String?
    var confidence: Double?
    var id: String { "\(startSec ?? -1)-\(endSec ?? -1)-\(text)" }
}

struct SpeakerRun: Codable, Identifiable, Hashable {
    var startSec: Double?
    var endSec: Double?
    var speaker: String?
    var text: String
    var segmentCount: Int
    var speakerConfidence: Double?
    var id: String { "\(startSec ?? -1)-\(speaker ?? "?")-\(segmentCount)" }
}

struct WordTiming: Codable, Hashable, SubtitleTimingSource {
    var text: String
    var startSec: Double
    var endSec: Double
}

struct Transcription: Codable {
    var language: String?
    var segments: [TranscriptSegment]
    var turns: [Turn]
    var speakerSegments: [SpeakerRun]
    var words: [[WordTiming]]?
    var droppedWindows: Int
    var audioSec: Double
    var skippedLeadingSec: Double
    /// Set when diarization was requested but failed after a successful
    /// decode. The transcript survives without speakers and this explains why.
    var diarizationError: String?

    struct Turn: Codable, Hashable {
        var startMs: UInt64
        var endMs: UInt64
        var speakerRef: String?
    }

    var transcript: String {
        segments.map { $0.text.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}

struct RunOptions {
    var language: String?
    var initialPrompt: String?
    var translate = false
    var diarize = false
    /// `nil` preserves the native engine's normal timestamped transcript.
    /// Live keyboard dictation sets this to false because it only needs text.
    var timestamps: Bool?
    var wordTimestamps = false

    var json: String {
        var object: [String: Any] = [
            "translate": translate,
            "diarize": diarize,
            "word_timestamps": wordTimestamps
        ]
        if let language, !language.isEmpty, language != "auto" {
            object["language"] = language
        }
        if let initialPrompt, !initialPrompt.trimmingCharacters(in: .whitespaces).isEmpty {
            object["initial_prompt"] = initialPrompt
        }
        if let timestamps {
            object["timestamps"] = timestamps
        }
        let data = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
        return String(bytes: data, encoding: .utf8) ?? "{}"
    }
}
