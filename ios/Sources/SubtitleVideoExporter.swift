import AVFoundation
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

@MainActor
enum SubtitleVideoExporter {
    private struct MeasuredWord {
        let word: SubtitleTimelineWord
        let size: CGSize
    }

    static func export(
        videoURL: URL,
        cues: [SubtitleCue],
        style: SubtitleRenderStyle,
        progress: @escaping (Double) -> Void
    ) async throws -> URL {
        let asset = AVURLAsset(url: videoURL)
        let duration = try await asset.load(.duration)
        let totalSeconds = duration.seconds
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        guard let sourceVideo = videoTracks.first else { throw SubtitleExportError.missingVideo }

        let naturalSize = try await sourceVideo.load(.naturalSize)
        let preferredTransform = try await sourceVideo.load(.preferredTransform)
        let transformedRect = CGRect(origin: .zero, size: naturalSize).applying(preferredTransform)
        let renderSize = CGSize(width: abs(transformedRect.width), height: abs(transformedRect.height))
        guard renderSize.width > 0, renderSize.height > 0 else {
            throw SubtitleExportError.invalidVideoSize
        }

        let instruction = AVMutableVideoCompositionInstruction()
        instruction.timeRange = CMTimeRange(start: .zero, duration: duration)
        let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: sourceVideo)
        let normalizedTransform = preferredTransform.concatenating(
            CGAffineTransform(
                translationX: -transformedRect.minX,
                y: -transformedRect.minY
            )
        )
        layerInstruction.setTransform(normalizedTransform, at: .zero)
        instruction.layerInstructions = [layerInstruction]

        let videoComposition = AVMutableVideoComposition()
        videoComposition.instructions = [instruction]
        videoComposition.renderSize = renderSize
        let nominalRate = try await sourceVideo.load(.nominalFrameRate)
        let framesPerSecond = max(24, min(60, Int32(nominalRate.rounded())))
        videoComposition.frameDuration = CMTime(value: 1, timescale: framesPerSecond)

        let parent = CALayer()
        parent.frame = CGRect(origin: .zero, size: renderSize)
        let videoLayer = CALayer()
        videoLayer.frame = parent.bounds
        let overlay = CALayer()
        overlay.frame = parent.bounds
        parent.addSublayer(videoLayer)
        parent.addSublayer(overlay)
        addCaptionLayers(
            cues: cues,
            style: style,
            renderSize: renderSize,
            duration: totalSeconds,
            to: overlay
        )
        videoComposition.animationTool = AVVideoCompositionCoreAnimationTool(
            postProcessingAsVideoLayer: videoLayer,
            in: parent
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

        let progressTask = Task { @MainActor in
            while exporter.status == .waiting || exporter.status == .exporting {
                progress(Double(exporter.progress))
                try? await Task.sleep(for: .milliseconds(120))
            }
        }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            exporter.exportAsynchronously {
                continuation.resume()
            }
        }
        progressTask.cancel()

        switch exporter.status {
        case .completed:
            progress(1)
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

    private static func addCaptionLayers(
        cues: [SubtitleCue],
        style: SubtitleRenderStyle,
        renderSize: CGSize,
        duration: Double,
        to overlay: CALayer
    ) {
        guard duration.isFinite, duration > 0 else { return }
        let scale = min(renderSize.width, renderSize.height) / 1080
        let font = style.font.uiFont(size: max(18, style.fontSize1080 * scale))
        let maxTextWidth = renderSize.width * 0.84
        let horizontalPadding = font.pointSize * 0.50
        let verticalPadding = font.pointSize * 0.30
        let lineGap = font.pointSize * 0.16
        // Active pills extend beyond each glyph run; a normal typographic space
        // is too tight and visually joins adjacent words ("PRIVATEAI"). Keep an
        // intentional social-caption rhythm with enough air around every pill.
        let spacing = max(
            (" " as NSString).size(withAttributes: [.font: font]).width,
            font.pointSize * 0.38
        )

        for cue in cues where cue.endSec > cue.startSec {
            let lines = measureLines(
                words: cue.words,
                font: font,
                maximumWidth: maxTextWidth,
                spacing: spacing
            )
            guard !lines.isEmpty else { continue }

            let lineHeight = font.lineHeight * 1.16
            let contentHeight = CGFloat(lines.count) * lineHeight
                + CGFloat(max(0, lines.count - 1)) * lineGap
            let boxWidth = min(
                renderSize.width * 0.92,
                (lines.map { lineWidth($0, spacing: spacing) }.max() ?? 0)
                    + horizontalPadding * 2
            )
            let boxHeight = contentHeight + verticalPadding * 2
            let boxX = (renderSize.width - boxWidth) / 2
            // Social-video safe zone: comfortably above transport controls,
            // profile captions, and the home indicator.
            let boxY = max(renderSize.height * 0.11, font.pointSize * 0.75)

            let container = CALayer()
            container.frame = CGRect(origin: .zero, size: renderSize)
            container.opacity = 0
            container.add(
                visibilityAnimation(start: cue.startSec, end: cue.endSec, duration: duration),
                forKey: "cueVisibility"
            )
            container.add(
                cueEntranceAnimation(start: cue.startSec, duration: duration),
                forKey: "cueEntrance"
            )

            if style.backgroundOpacity > 0.001 {
                let background = CALayer()
                background.frame = CGRect(x: boxX, y: boxY, width: boxWidth, height: boxHeight)
                background.backgroundColor = UIColor.black
                    .withAlphaComponent(style.backgroundOpacity).cgColor
                background.cornerRadius = font.pointSize * 0.34
                background.shadowColor = UIColor.black.cgColor
                background.shadowOpacity = 0.45
                background.shadowRadius = font.pointSize * 0.18
                background.shadowOffset = CGSize(width: 0, height: -font.pointSize * 0.04)
                container.addSublayer(background)
            }

            for (lineIndex, line) in lines.enumerated() {
                let width = lineWidth(line, spacing: spacing)
                var x = (renderSize.width - width) / 2
                let y = boxY + verticalPadding
                    + CGFloat(lines.count - lineIndex - 1) * (lineHeight + lineGap)

                for measured in line {
                    let frame = CGRect(
                        x: x,
                        y: y,
                        width: ceil(measured.size.width) + 3,
                        height: ceil(lineHeight) + 3
                    )
                    container.addSublayer(
                        textLayer(
                            text: measured.word.text,
                            font: font,
                            color: style.textColor,
                            frame: frame,
                            strokeWidth: -4.0
                        )
                    )

                    let pillInset = font.pointSize * 0.13
                    let pillFrame = frame.insetBy(dx: -pillInset, dy: -pillInset * 0.45)
                    let pill = karaokePill(
                        frame: pillFrame,
                        color: style.highlightColor,
                        fontSize: font.pointSize
                    )
                    pill.opacity = 0
                    pill.add(
                        visibilityAnimation(
                            start: measured.word.startSec,
                            end: measured.word.endSec,
                            duration: duration
                        ),
                        forKey: "wordVisibility"
                    )
                    pill.add(
                        wordPopAnimation(start: measured.word.startSec, duration: duration),
                        forKey: "wordPop"
                    )
                    container.addSublayer(pill)

                    let highlight = textLayer(
                        text: measured.word.text,
                        font: font,
                        color: contrastingTextColor(for: style.highlightColor),
                        frame: frame,
                        strokeWidth: 0
                    )
                    highlight.opacity = 0
                    highlight.add(
                        visibilityAnimation(
                            start: measured.word.startSec,
                            end: measured.word.endSec,
                            duration: duration
                        ),
                        forKey: "wordVisibility"
                    )
                    highlight.add(
                        wordPopAnimation(start: measured.word.startSec, duration: duration),
                        forKey: "wordPop"
                    )
                    container.addSublayer(highlight)
                    x += measured.size.width + spacing
                }
            }
            overlay.addSublayer(container)
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

    private static func textLayer(
        text: String,
        font: UIFont,
        color: UIColor,
        frame: CGRect,
        strokeWidth: CGFloat
    ) -> CALayer {
        // CATextLayer's attributed UIKit font payload is not reliably preserved
        // by AVVideoCompositionCoreAnimationTool (it can produce perfect pills
        // with completely missing glyphs). Rasterize each short word once at 3x
        // and animate the resulting pixel layer instead. This is deterministic
        // across iOS and Catalyst and still keeps every per-frame operation on
        // Core Animation rather than redrawing text thirty times a second.
        let format = UIGraphicsImageRendererFormat()
        format.scale = 3
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: frame.size, format: format)
        let image = renderer.image { _ in
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
            let origin = CGPoint(
                x: max(1, (frame.width - measured.width) / 2),
                y: max(0, (frame.height - measured.height) / 2)
            )
            string.draw(at: origin, withAttributes: attributes)
        }

        let layer = CALayer()
        layer.frame = frame
        layer.contents = image.cgImage
        layer.contentsScale = image.scale
        layer.contentsGravity = .resizeAspect
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.60
        layer.shadowRadius = font.pointSize * 0.06
        layer.shadowOffset = CGSize(width: 0, height: -font.pointSize * 0.025)
        return layer
    }

    private static func karaokePill(
        frame: CGRect,
        color: UIColor,
        fontSize: CGFloat
    ) -> CAGradientLayer {
        let pill = CAGradientLayer()
        pill.frame = frame
        pill.cornerRadius = min(frame.height / 2, fontSize * 0.26)
        pill.colors = [
            mix(color, with: .white, amount: 0.24).cgColor,
            color.cgColor,
        ]
        pill.startPoint = CGPoint(x: 0.15, y: 1)
        pill.endPoint = CGPoint(x: 0.85, y: 0)
        pill.shadowColor = color.cgColor
        pill.shadowOpacity = 0.55
        pill.shadowRadius = fontSize * 0.16
        pill.shadowOffset = .zero
        return pill
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

    private static func visibilityAnimation(
        start: Double,
        end: Double,
        duration: Double
    ) -> CAKeyframeAnimation {
        let start = max(0, min(duration, start)) / duration
        let end = max(start, min(duration, end) / duration)
        let animation = CAKeyframeAnimation(keyPath: "opacity")
        if start <= 0, end >= 1 {
            animation.keyTimes = [0, 1]
            animation.values = [1, 1]
        } else if start <= 0 {
            animation.keyTimes = [0, NSNumber(value: end), 1]
            animation.values = [1, 0, 0]
        } else if end >= 1 {
            animation.keyTimes = [0, NSNumber(value: start), 1]
            animation.values = [0, 1, 1]
        } else {
            animation.keyTimes = [0, NSNumber(value: start), NSNumber(value: end), 1]
            animation.values = [0, 1, 0, 0]
        }
        animation.calculationMode = .discrete
        animation.beginTime = AVCoreAnimationBeginTimeAtZero
        animation.duration = duration
        animation.isRemovedOnCompletion = false
        animation.fillMode = .both
        return animation
    }

    private static func cueEntranceAnimation(start: Double, duration: Double) -> CAKeyframeAnimation {
        normalizedAnimation(
            keyPath: "transform.scale",
            points: [
                (0, 0.88),
                (start, 0.88),
                (start + 0.10, 1.06),
                (start + 0.19, 1.00),
                (duration, 1.00),
            ],
            duration: duration
        )
    }

    private static func wordPopAnimation(start: Double, duration: Double) -> CAKeyframeAnimation {
        normalizedAnimation(
            keyPath: "transform.scale",
            points: [
                (0, 0.82),
                (start, 0.82),
                (start + 0.06, 1.15),
                (start + 0.13, 1.00),
                (duration, 1.00),
            ],
            duration: duration
        )
    }

    private static func normalizedAnimation(
        keyPath: String,
        points: [(Double, CGFloat)],
        duration: Double
    ) -> CAKeyframeAnimation {
        let animation = CAKeyframeAnimation(keyPath: keyPath)
        var times: [NSNumber] = []
        var values: [CGFloat] = []
        var last = -Double.infinity
        for (rawTime, value) in points {
            let time = max(0, min(duration, rawTime))
            guard time > last else { continue }
            times.append(NSNumber(value: time / duration))
            values.append(value)
            last = time
        }
        animation.keyTimes = times
        animation.values = values
        animation.timingFunctions = values.dropFirst().map { _ in
            CAMediaTimingFunction(name: .easeOut)
        }
        animation.beginTime = AVCoreAnimationBeginTimeAtZero
        animation.duration = duration
        animation.isRemovedOnCompletion = false
        animation.fillMode = .both
        return animation
    }
}
