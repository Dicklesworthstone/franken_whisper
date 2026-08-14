// Transcript export. The headline formats mirror the website's exports —
// a self-contained dark HTML page and GitHub-flavored Markdown, both built
// with the user's speaker names — plus plain text. SRT and JSON are niche
// interchange formats and live behind the "More formats" menu.
//
// Everything is built from the same segments the UI renders, so what you
// share is exactly what you saw.

import CoreTransferable
import Foundation
import UniformTypeIdentifiers

enum TranscriptFormat: String, CaseIterable, Identifiable {
    case html = "HTML"
    case markdown = "MD"
    case text = "TXT"
    case srt = "SRT"
    case json = "JSON"
    var id: String { rawValue }

    /// The segmented picker's formats; SRT/JSON ride in the "More" menu.
    static let primary: [TranscriptFormat] = [.html, .markdown, .text]
    static let niche: [TranscriptFormat] = [.srt, .json]

    var fileExtension: String {
        switch self {
        case .html: "html"
        case .markdown: "md"
        case .text: "txt"
        case .srt: "srt"
        case .json: "json"
        }
    }

    var menuLabel: String {
        switch self {
        case .html: "Styled web page (HTML)"
        case .markdown: "Markdown (GitHub-flavored)"
        case .text: "Plain text"
        case .srt: "Subtitles (SRT)"
        case .json: "Structured data (JSON)"
        }
    }
}

/// Everything an export needs beyond the result itself: where the audio came
/// from, how long the run took, and the user's speaker-name assignments
/// (raw `SPEAKER_NN` lane → display name).
struct ExportContext {
    var sourceName: String
    var wallSeconds: Double
    var names: [String: String]

    func display(_ speaker: String?) -> String {
        guard let speaker else { return "UNKNOWN" }
        return names[speaker] ?? speaker
    }
}

enum TranscriptExport {
    // The website's lane palette (site/app.js SPK_COLORS), keyed by lane
    // number so the HTML export matches what the browser demo produces.
    private static let speakerHexes = ["#34d399", "#fbbf24", "#93c5fd", "#f9a8d4"]

    private static func hex(for speaker: String?) -> String {
        guard let speaker,
            let lane = speaker.split(separator: "_").last.flatMap({ Int($0) })
        else { return "#94a3b8" }
        return speakerHexes[lane % speakerHexes.count]
    }

    static func content(
        _ format: TranscriptFormat, from result: Transcription, context: ExportContext
    ) -> String {
        switch format {
        case .html: html(from: result, context: context)
        case .markdown: markdown(from: result, context: context)
        case .text: text(from: result, context: context)
        case .srt: srt(from: result, context: context)
        case .json: json(from: result)
        }
    }

    // ── Plain text ─────────────────────────────────────────────────────────

    static func text(from result: Transcription, context: ExportContext) -> String {
        if !result.speakerSegments.isEmpty {
            return result.speakerSegments.map { run in
                "[\(context.display(run.speaker))] \(run.text)"
            }
            .joined(separator: "\n\n")
        }
        return result.transcript
    }

    // ── GitHub-flavored Markdown (mirrors site/app.js exportMd) ────────────

    static func markdown(from result: Transcription, context: ExportContext) -> String {
        var lines: [String] = [
            "# Transcript — \(context.sourceName)",
            "",
            "- Engine: franken_whisper on iPhone (Whisper large-v3-turbo q8_0 + Sortformer diarization)",
            String(
                format: "- Audio: %.1f s · transcribed%@ in %.1f s on device",
                result.audioSec,
                result.speakerSegments.isEmpty ? "" : " + diarized",
                context.wallSeconds),
        ]
        if !context.names.isEmpty {
            let assigned = context.names.sorted { $0.key < $1.key }
                .map { "\($0.key) = \($0.value)" }
                .joined(separator: ", ")
            lines.append(
                "- Speaker names were assigned to detected voices in order of first appearance (\(assigned))"
            )
        }
        if result.droppedWindows > 0 {
            lines.append("- **Warning: \(result.droppedWindows) window(s) dropped without output**")
        }
        if let reason = result.diarizationError {
            lines.append("- **Warning: diarization unavailable — \(reason)**")
        }
        lines.append("")

        if !result.speakerSegments.isEmpty {
            for run in result.speakerSegments {
                lines.append(
                    "**[\(span(run.startSec, run.endSec))] \(context.display(run.speaker)):** "
                        + run.text.trimmingCharacters(in: .whitespaces))
                lines.append("")
            }
        } else {
            for segment in result.segments {
                let text = segment.text.trimmingCharacters(in: .whitespaces)
                if text.isEmpty { continue }
                lines.append("**[\(span(segment.startSec, segment.endSec))]** \(text)")
                lines.append("")
            }
        }
        return lines.joined(separator: "\n")
    }

    // ── Self-contained HTML (mirrors site/app.js exportHtml) ───────────────

    static func html(from result: Transcription, context: ExportContext) -> String {
        let rows: String
        if !result.speakerSegments.isEmpty {
            rows = result.speakerSegments.map { run in
                let color = hex(for: run.speaker)
                return "      <div class=\"seg\"><span class=\"t\">\(span(run.startSec, run.endSec))</span> "
                    + "<span class=\"spk\" style=\"color:\(color);border-color:\(color)66\">"
                    + "\(escape(context.display(run.speaker)))</span> "
                    + escape(run.text.trimmingCharacters(in: .whitespaces)) + "</div>"
            }
            .joined(separator: "\n")
        } else {
            rows = result.segments.compactMap { segment in
                let text = segment.text.trimmingCharacters(in: .whitespaces)
                if text.isEmpty { return nil }
                return "      <div class=\"seg\"><span class=\"t\">\(span(segment.startSec, segment.endSec))</span> "
                    + escape(text) + "</div>"
            }
            .joined(separator: "\n")
        }
        let warning =
            result.droppedWindows > 0
            ? "<div class=\"warn\">⚠ \(result.droppedWindows) window(s) dropped without output — a real content gap</div>"
            : ""
        let meta = String(
            format: "franken_whisper on iPhone (large-v3-turbo q8_0 + Sortformer) · %.1f s of audio · processed on device in %.1f s",
            result.audioSec, context.wallSeconds)
        return """
            <!doctype html>
            <html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Transcript — \(escape(context.sourceName))</title>
            <style>
              body{margin:0;background:#060b09;color:#d6e5dd;font:16px/1.75 ui-sans-serif,system-ui,sans-serif;padding:3rem 1.5rem}
              main{max-width:46rem;margin:0 auto}
              h1{font-size:1.3rem;color:#a7f3d0}
              .meta{color:#7d8f86;font-size:.85rem;margin-bottom:2rem}
              .warn{color:#fbbf24;font-size:.85rem;margin-bottom:1rem}
              .seg{margin:.6rem 0}
              .t{color:#5f7a6e;font-family:ui-monospace,monospace;font-size:.75rem;margin-right:.5rem}
              .spk{font-family:ui-monospace,monospace;font-size:.72rem;font-weight:800;border:1px solid;border-radius:999px;padding:.08rem .5rem;margin-right:.5rem}
            </style></head><body><main>
              <h1>Transcript — \(escape(context.sourceName))</h1>
              <div class="meta">\(meta)</div>
              \(warning)
            \(rows)
            </main></body></html>
            """
    }

    // ── SRT ────────────────────────────────────────────────────────────────

    static func srt(from result: Transcription, context: ExportContext) -> String {
        var out = ""
        var index = 1
        for segment in result.segments {
            guard let start = segment.startSec, let end = segment.endSec else { continue }
            let text = segment.text.trimmingCharacters(in: .whitespaces)
            if text.isEmpty { continue }
            let speaker = segment.speaker.map { "[\(context.display($0))] " } ?? ""
            out += "\(index)\n\(timestamp(start)) --> \(timestamp(end))\n\(speaker)\(text)\n\n"
            index += 1
        }
        return out
    }

    // ── JSON ───────────────────────────────────────────────────────────────

    /// The full structured result — segments, speaker runs, turns, and the
    /// DTW word timings when they were requested — in the engine's own
    /// snake_case key style. This is the only export that carries `words`.
    static func json(from result: Transcription) -> String {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(result) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    // ── Formatting helpers ─────────────────────────────────────────────────

    private static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }

    private static func clock(_ seconds: Double?) -> String {
        guard let seconds else { return "—" }
        let clamped = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", clamped / 60, clamped % 60)
    }

    private static func span(_ start: Double?, _ end: Double?) -> String {
        "\(clock(start))–\(clock(end))"
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
