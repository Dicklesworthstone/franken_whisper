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

enum DecodeMode: String, CaseIterable, Identifiable {
    case fast
    case careful

    var id: Self { self }
    var label: String { self == .fast ? "Fast" : "Careful" }
    var beamSize: Int? { self == .careful ? 5 : nil }
}

enum TranscriptionPrompt {
    static let maxContextCharacters = 800

    static func combined(speakerNames: [String], context: String) -> String? {
        var parts: [String] = []
        if !speakerNames.isEmpty {
            parts.append("Speakers: \(speakerNames.joined(separator: ", ")).")
        }

        let trimmedContext = context.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedContext.isEmpty {
            parts.append("Context: \(trimmedContext)")
        }

        guard !parts.isEmpty else { return nil }
        return String(parts.joined(separator: " ").prefix(maxContextCharacters))
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
    /// `nil` keeps the engine's byte-identical greedy default. The careful
    /// UI mode sends five, matching whisper.cpp's quality-oriented default.
    var beamSize: Int?

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
        if let beamSize {
            object["beam_size"] = beamSize
        }
        let data = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
        return String(bytes: data, encoding: .utf8) ?? "{}"
    }
}
