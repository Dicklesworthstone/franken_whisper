import AVFoundation
import CoreImage
import Foundation
import UIKit

enum SubtitleFontChoice: String, CaseIterable, Identifiable {
    case bold = "Bold"
    case rounded = "Rounded"
    case serif = "Serif"
    case mono = "Mono"

    var id: Self { self }

    func uiFont(size: CGFloat) -> UIFont {
        switch self {
        case .bold:
            UIFont.systemFont(ofSize: size, weight: .heavy)
        case .rounded:
            UIFont(
                descriptor: UIFont.systemFont(ofSize: size, weight: .heavy)
                    .fontDescriptor.withDesign(.rounded)
                    ?? UIFont.systemFont(ofSize: size, weight: .heavy).fontDescriptor,
                size: size
            )
        case .serif:
            UIFont(
                descriptor: UIFont.systemFont(ofSize: size, weight: .bold)
                    .fontDescriptor.withDesign(.serif)
                    ?? UIFont.systemFont(ofSize: size, weight: .bold).fontDescriptor,
                size: size
            )
        case .mono:
            UIFont.monospacedSystemFont(ofSize: size, weight: .bold)
        }
    }
}

struct SubtitleRenderStyle {
    var font: SubtitleFontChoice
    var fontSize1080: CGFloat
    var textColor: UIColor
    /// The selected karaoke color drives the active-word pill; its foreground
    /// is picked automatically for readable contrast.
    var highlightColor: UIColor
    var backgroundOpacity: CGFloat
}

enum SubtitleExportError: LocalizedError {
    case missingVideo
    case invalidVideoSize
    case cannotCreateExporter
    case exportFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingVideo: "The source video track is unavailable."
        case .invalidVideoSize: "The source video has an invalid display size."
        case .cannotCreateExporter: "This device cannot export that video format."
        case .exportFailed(let reason): "Video export failed: \(reason)"
        }
    }
}

enum SubtitleVideoExporter {
    /// `withTaskCancellationHandler` invokes `onCancel` from an arbitrary
    /// executor. AVAssetExportSession's cancellation entry point is designed
    /// for exactly that cross-thread signal, but the Objective-C class has no
    /// Sendable annotation. Keep the unchecked boundary tiny and immutable.
    private final class ExportCancellation: @unchecked Sendable {
        private let exporter: AVAssetExportSession

        init(_ exporter: AVAssetExportSession) {
            self.exporter = exporter
        }

        func cancel() {
            exporter.cancelExport()
        }
    }

    private struct MeasuredWord {
        let word: SubtitleTimelineWord
        let size: CGSize
    }

    /// Bridge one callback-based export without losing cancellation in the
    /// registration gap. Factored as a seam so the pre-cancel schedule is
    /// deterministic in unit tests without encoding a real movie.
    static func awaitExportCompletion(
        start: (@escaping () -> Void) -> Void,
        cancel: @escaping @Sendable () -> Void
    ) async throws {
        try Task.checkCancellation()
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                start { continuation.resume() }
                // Cancellation can land after the preflight but before the
                // callback source is fully registered. Recheck on the far side
                // so one of the two cancellation paths always reaches it.
                if Task.isCancelled {
                    cancel()
                }
            }
        } onCancel: {
            cancel()
        }
        try Task.checkCancellation()
    }

    static func export(
        videoURL: URL,
        cues: [SubtitleCue],
        style: SubtitleRenderStyle,
        progress: @MainActor @escaping (Double) -> Void
    ) async throws -> URL {
        let asset = AVURLAsset(url: videoURL)
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        guard let sourceVideo = videoTracks.first else { throw SubtitleExportError.missingVideo }

        let naturalSize = try await sourceVideo.load(.naturalSize)
        let preferredTransform = try await sourceVideo.load(.preferredTransform)
        let transformedRect = CGRect(origin: .zero, size: naturalSize).applying(preferredTransform)
        let renderSize = CGSize(width: abs(transformedRect.width), height: abs(transformedRect.height))
        guard renderSize.width.isFinite,
              renderSize.height.isFinite,
              renderSize.width > 0,
              renderSize.height > 0,
              renderSize.width <= 16_384,
              renderSize.height <= 16_384
        else {
            throw SubtitleExportError.invalidVideoSize
        }

        // Render against the source frame's actual composition timestamp. The
        // previous Core Animation graph expressed every cue as a normalized
        // fraction of the asset duration. Besides crashing the Simulator's GL
        // compiler for realistic transcripts, that introduced a second clock
        // between decoder timings and video presentation timestamps. This
        // handler is called for each real frame and selects the DTW word using
        // request.compositionTime directly.
        let frameRenderer = SubtitleFrameRenderer(cues: cues, style: style)
        let compositionExtent = CGRect(origin: .zero, size: renderSize)
        let videoComposition = AVVideoComposition(
            asset: asset,
            applyingCIFiltersWithHandler: { request in
                // CI filter graphs are allowed to advertise an infinite
                // working extent. The output contract is the finite,
                // orientation-correct track size validated above.
                let source = request.sourceImage.cropped(to: compositionExtent)
                guard let overlay = frameRenderer.overlay(
                    at: request.compositionTime.seconds,
                    renderSize: renderSize
                ) else {
                    request.finish(with: source, context: nil)
                    return
                }
                request.finish(
                    with: overlay.composited(over: source).cropped(to: compositionExtent),
                    context: nil
                )
            }
        )

        guard let exporter = AVAssetExportSession(
            // Export the original asset directly so every playable auxiliary
            // audio/edit-list timing detail survives unchanged.
            asset: asset,
            presetName: AVAssetExportPresetHighestQuality
        ) else { throw SubtitleExportError.cannotCreateExporter }

        let fileType: AVFileType = exporter.supportedFileTypes.contains(.mp4) ? .mp4 : .mov
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankenwhisper-captioned-\(UUID().uuidString)")
            .appendingPathExtension(fileType == .mp4 ? "mp4" : "mov")
        exporter.outputURL = output
        exporter.outputFileType = fileType
        exporter.videoComposition = videoComposition
        exporter.shouldOptimizeForNetworkUse = false
        let cancellation = ExportCancellation(exporter)

        let progressTask = Task { @MainActor in
            while !Task.isCancelled,
                  exporter.status == .waiting || exporter.status == .exporting
            {
                progress(Double(exporter.progress))
                do {
                    try await Task.sleep(for: .milliseconds(120))
                } catch {
                    return
                }
            }
        }
        defer { progressTask.cancel() }
        do {
            try await awaitExportCompletion(
                start: { completion in
                    exporter.exportAsynchronously(completionHandler: completion)
                },
                cancel: { cancellation.cancel() }
            )
        } catch {
            try? FileManager.default.removeItem(at: output)
            throw error
        }

        switch exporter.status {
        case .completed:
            await MainActor.run { progress(1) }
            return output
        case .cancelled:
            try? FileManager.default.removeItem(at: output)
            throw CancellationError()
        case .failed:
            try? FileManager.default.removeItem(at: output)
            throw SubtitleExportError.exportFailed(
                exporter.error?.localizedDescription ?? "unknown AVFoundation error"
            )
        default:
            try? FileManager.default.removeItem(at: output)
            throw SubtitleExportError.exportFailed("export ended unexpectedly")
        }
    }

    /// Produces one transparent caption image for each visible cue/word state.
    /// AVFoundation can request frames concurrently, so the cache is bounded
    /// and synchronized by NSCache instead of retaining a full-frame image for
    /// every word in a long movie.
    private final class SubtitleFrameRenderer: @unchecked Sendable {
        private let cues: [SubtitleCue]
        private let style: SubtitleRenderStyle
        private let cache = NSCache<NSString, CIImage>()

        init(cues: [SubtitleCue], style: SubtitleRenderStyle) {
            self.cues = cues
            self.style = style
            cache.totalCostLimit = 64 * 1_024 * 1_024
        }

        func overlay(at seconds: Double, renderSize: CGSize) -> CIImage? {
            guard seconds.isFinite,
                  renderSize.width.isFinite,
                  renderSize.height.isFinite,
                  renderSize.width > 0,
                  renderSize.height > 0,
                  renderSize.width <= 16_384,
                  renderSize.height <= 16_384,
                  let cue = activeCue(at: seconds)
            else { return nil }

            let activeWordID = cue.words.first {
                seconds >= $0.startSec && seconds < $0.endSec
            }?.id ?? -1
            let width = Int(renderSize.width.rounded())
            let height = Int(renderSize.height.rounded())
            let key = "\(width)x\(height)-\(cue.id)-\(activeWordID)" as NSString
            if let cached = cache.object(forKey: key) {
                return cached
            }

            guard let image = render(
                cue: cue,
                activeWordID: activeWordID,
                renderSize: CGSize(width: width, height: height)
            ) else { return nil }
            let overlay = CIImage(cgImage: image)
            cache.setObject(overlay, forKey: key, cost: width * height * 4)
            return overlay
        }

        private func activeCue(at seconds: Double) -> SubtitleCue? {
            SubtitleTimeline.activeCue(in: cues, at: seconds)
        }

        private func render(
            cue: SubtitleCue,
            activeWordID: Int,
            renderSize: CGSize
        ) -> CGImage? {
            let scale = min(renderSize.width, renderSize.height) / 1080
            let font = style.font.uiFont(size: max(18, style.fontSize1080 * scale))
            let speakerAccent = cue.speaker.map {
                SubtitleSpeakerPalette.uiColor(for: $0.laneID)
            } ?? style.highlightColor
            let maxTextWidth = renderSize.width * 0.84
            let horizontalPadding = font.pointSize * 0.50
            let verticalPadding = font.pointSize * 0.30
            let lineGap = font.pointSize * 0.16
            let spacing = max(
                (" " as NSString).size(withAttributes: [.font: font]).width,
                font.pointSize * 0.38
            )
            let lines = SubtitleVideoExporter.measureLines(
                words: cue.words,
                font: font,
                maximumWidth: maxTextWidth,
                spacing: spacing
            )
            guard !lines.isEmpty else { return nil }

            let speakerFont = style.font.uiFont(size: max(14, font.pointSize * 0.48))
            let speakerLabel = cue.speaker?.displayName.flatMap { rawName -> String? in
                let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !name.isEmpty else { return nil }
                return SubtitleVideoExporter.fittedLabel(
                    name,
                    font: speakerFont,
                    maximumWidth: renderSize.width * 0.38
                )
            }
            let speakerLabelSize = speakerLabel.map {
                ($0 as NSString).size(withAttributes: [.font: speakerFont])
            }
            let speakerLabelHorizontalPadding = speakerFont.pointSize * 0.76
            let speakerLabelWidth = speakerLabelSize.map {
                $0.width + speakerLabelHorizontalPadding * 2
            } ?? 0
            let speakerLabelHeight = speakerLabelSize.map {
                max($0.height * 1.34, speakerFont.lineHeight * 1.25)
            } ?? 0

            let lineHeight = font.lineHeight * 1.16
            let contentHeight = CGFloat(lines.count) * lineHeight
                + CGFloat(max(0, lines.count - 1)) * lineGap
            let boxWidth = min(
                renderSize.width * 0.92,
                max(
                    (lines.map {
                        SubtitleVideoExporter.lineWidth($0, spacing: spacing)
                    }.max() ?? 0) + horizontalPadding * 2,
                    speakerLabelWidth + horizontalPadding
                )
            )
            let boxHeight = contentHeight + verticalPadding * 2
            let boxX = (renderSize.width - boxWidth) / 2
            let bottomSafeZone = max(renderSize.height * 0.11, font.pointSize * 0.75)
            let boxY = renderSize.height - bottomSafeZone - boxHeight
            let boxFrame = CGRect(x: boxX, y: boxY, width: boxWidth, height: boxHeight)

            let format = UIGraphicsImageRendererFormat()
            format.scale = 1
            format.opaque = false
            let renderer = UIGraphicsImageRenderer(size: renderSize, format: format)
            let rendered = renderer.image { context in
                let cg = context.cgContext
                let boxPath = UIBezierPath(
                    roundedRect: boxFrame,
                    cornerRadius: font.pointSize * 0.34
                )
                if style.backgroundOpacity > 0.001 || cue.speaker != nil {
                    cg.saveGState()
                    cg.setShadow(
                        offset: CGSize(width: 0, height: font.pointSize * 0.04),
                        blur: font.pointSize * 0.18,
                        color: UIColor.black.withAlphaComponent(0.45).cgColor
                    )
                    UIColor.black.withAlphaComponent(style.backgroundOpacity).setFill()
                    boxPath.fill()
                    cg.restoreGState()
                    if cue.speaker != nil {
                        speakerAccent.withAlphaComponent(0.72).setStroke()
                        boxPath.lineWidth = max(1.5, font.pointSize * 0.035)
                        boxPath.stroke()
                    }
                }

                if let speakerLabel, speakerLabelHeight > 0 {
                    let badgeFrame = CGRect(
                        x: boxX + horizontalPadding * 0.55,
                        y: max(0, boxY - speakerLabelHeight - font.pointSize * 0.12),
                        width: min(speakerLabelWidth, boxWidth - horizontalPadding),
                        height: speakerLabelHeight
                    )
                    drawPill(
                        in: badgeFrame,
                        color: speakerAccent,
                        fontSize: speakerFont.pointSize,
                        context: cg
                    )
                    drawText(
                        speakerLabel,
                        font: speakerFont,
                        color: SubtitleVideoExporter.contrastingTextColor(for: speakerAccent),
                        frame: badgeFrame,
                        strokeWidth: 0
                    )
                }

                for (lineIndex, line) in lines.enumerated() {
                    let width = SubtitleVideoExporter.lineWidth(line, spacing: spacing)
                    var x = (renderSize.width - width) / 2
                    let y = boxY + verticalPadding
                        + CGFloat(lineIndex) * (lineHeight + lineGap)

                    for measured in line {
                        let frame = CGRect(
                            x: x,
                            y: y,
                            width: ceil(measured.size.width) + 3,
                            height: ceil(lineHeight) + 3
                        )
                        drawText(
                            measured.word.text,
                            font: font,
                            color: style.textColor,
                            frame: frame,
                            strokeWidth: -4
                        )

                        if measured.word.id == activeWordID {
                            let pillInset = font.pointSize * 0.13
                            let pillFrame = frame.insetBy(
                                dx: -pillInset,
                                dy: -pillInset * 0.45
                            )
                            drawPill(
                                in: pillFrame,
                                color: speakerAccent,
                                fontSize: font.pointSize,
                                context: cg
                            )
                            drawText(
                                measured.word.text,
                                font: font,
                                color: SubtitleVideoExporter.contrastingTextColor(
                                    for: speakerAccent
                                ),
                                frame: frame,
                                strokeWidth: 0
                            )
                        }
                        x += measured.size.width + spacing
                    }
                }
            }
            return rendered.cgImage
        }

        private func drawText(
            _ text: String,
            font: UIFont,
            color: UIColor,
            frame: CGRect,
            strokeWidth: CGFloat
        ) {
            var attributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: color,
            ]
            if strokeWidth != 0 {
                attributes[.strokeColor] = UIColor.black.withAlphaComponent(0.94)
                attributes[.strokeWidth] = strokeWidth
            }
            let string = text as NSString
            let measured = string.size(withAttributes: attributes)
            string.draw(
                at: CGPoint(
                    x: max(frame.minX + 1, frame.midX - measured.width / 2),
                    y: max(frame.minY, frame.midY - measured.height / 2)
                ),
                withAttributes: attributes
            )
        }

        private func drawPill(
            in frame: CGRect,
            color: UIColor,
            fontSize: CGFloat,
            context: CGContext
        ) {
            let path = UIBezierPath(
                roundedRect: frame,
                cornerRadius: min(frame.height / 2, fontSize * 0.26)
            )
            context.saveGState()
            context.setShadow(
                offset: .zero,
                blur: fontSize * 0.16,
                color: color.withAlphaComponent(0.55).cgColor
            )
            color.setFill()
            path.fill()
            context.restoreGState()

            guard let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: [
                    SubtitleVideoExporter.mix(color, with: .white, amount: 0.24).cgColor,
                    color.cgColor,
                ] as CFArray,
                locations: [0, 1]
            ) else { return }
            context.saveGState()
            context.addPath(path.cgPath)
            context.clip()
            context.drawLinearGradient(
                gradient,
                start: CGPoint(x: frame.minX, y: frame.maxY),
                end: CGPoint(x: frame.maxX, y: frame.minY),
                options: []
            )
            context.restoreGState()
        }
    }

    private static func measureLines(
        words: [SubtitleTimelineWord],
        font: UIFont,
        maximumWidth: CGFloat,
        spacing: CGFloat
    ) -> [[MeasuredWord]] {
        var lines: [[MeasuredWord]] = []
        var current: [MeasuredWord] = []
        var currentWidth: CGFloat = 0

        for word in words {
            let size = (word.text as NSString).size(withAttributes: [.font: font])
            let addition = (current.isEmpty ? 0 : spacing) + size.width
            if !current.isEmpty, currentWidth + addition > maximumWidth {
                lines.append(current)
                current = []
                currentWidth = 0
            }
            current.append(MeasuredWord(word: word, size: size))
            currentWidth += (current.count == 1 ? 0 : spacing) + size.width
        }
        if !current.isEmpty { lines.append(current) }
        return lines
    }

    private static func lineWidth(_ line: [MeasuredWord], spacing: CGFloat) -> CGFloat {
        line.reduce(0) { $0 + $1.size.width }
            + CGFloat(max(0, line.count - 1)) * spacing
    }

    private static func fittedLabel(
        _ label: String,
        font: UIFont,
        maximumWidth: CGFloat
    ) -> String {
        guard maximumWidth > 0 else { return "" }
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        if (label as NSString).size(withAttributes: attributes).width <= maximumWidth {
            return label
        }
        var candidate = label
        while candidate.count > 1 {
            candidate.removeLast()
            let truncated = candidate.trimmingCharacters(in: .whitespaces) + "\u{2026}"
            if (truncated as NSString).size(withAttributes: attributes).width <= maximumWidth {
                return truncated
            }
        }
        return "\u{2026}"
    }

    private static func mix(_ color: UIColor, with other: UIColor, amount: CGFloat) -> UIColor {
        var r1: CGFloat = 0
        var g1: CGFloat = 0
        var b1: CGFloat = 0
        var a1: CGFloat = 0
        var r2: CGFloat = 0
        var g2: CGFloat = 0
        var b2: CGFloat = 0
        var a2: CGFloat = 0
        guard color.getRed(&r1, green: &g1, blue: &b1, alpha: &a1),
              other.getRed(&r2, green: &g2, blue: &b2, alpha: &a2)
        else { return color }
        let t = max(0, min(1, amount))
        return UIColor(
            red: r1 + (r2 - r1) * t,
            green: g1 + (g2 - g1) * t,
            blue: b1 + (b2 - b1) * t,
            alpha: a1 + (a2 - a1) * t
        )
    }

    private static func contrastingTextColor(for color: UIColor) -> UIColor {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        guard color.getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            return .black
        }
        let luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return luminance > 0.58 ? UIColor.black.withAlphaComponent(0.90) : .white
    }

}
