// Transcript export: plain text and SRT, the two formats people actually
// paste and import. Built from the same segments the UI renders, so what you
// share is exactly what you saw.

import CoreTransferable
import Foundation
import UniformTypeIdentifiers

enum TranscriptFormat: String, CaseIterable, Identifiable {
    case text = "TXT"
    case srt = "SRT"
    var id: String { rawValue }

    var fileExtension: String {
        switch self {
        case .text: "txt"
        case .srt: "srt"
        }
    }
}

enum TranscriptExport {
    static func text(from result: Transcription) -> String {
        // Speaker-attributed runs when diarized (one paragraph per speaker
        // run), plain segment text otherwise.
        if !result.speakerSegments.isEmpty {
            return result.speakerSegments.map { run in
                let speaker = run.speaker.map { "[\($0)] " } ?? ""
                return speaker + run.text
            }
            .joined(separator: "\n\n")
        }
        return result.transcript
    }

    static func srt(from result: Transcription) -> String {
        var out = ""
        var index = 1
        for segment in result.segments {
            guard let start = segment.startSec, let end = segment.endSec else { continue }
            let text = segment.text.trimmingCharacters(in: .whitespaces)
            if text.isEmpty { continue }
            let speaker = segment.speaker.map { "[\($0)] " } ?? ""
            out += "\(index)\n\(timestamp(start)) --> \(timestamp(end))\n\(speaker)\(text)\n\n"
            index += 1
        }
        return out
    }

    private static func timestamp(_ seconds: Double) -> String {
        let clamped = max(0, seconds)
        let hours = Int(clamped) / 3600
        let minutes = (Int(clamped) % 3600) / 60
        let secs = Int(clamped) % 60
        let millis = Int((clamped - clamped.rounded(.down)) * 1000)
        return String(format: "%02d:%02d:%02d,%03d", hours, minutes, secs, millis)
    }
}

/// A shareable transcript file for ShareLink.
struct TranscriptFile: Transferable {
    let content: String
    let baseName: String
    let format: TranscriptFormat

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .plainText) { file in
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(file.baseName)-\(UUID().uuidString.prefix(6))")
                .appendingPathExtension(file.format.fileExtension)
            try file.content.data(using: .utf8)?.write(to: url)
            return SentTransferredFile(url)
        }
    }
}
