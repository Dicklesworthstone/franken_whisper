import AVFoundation
import CoreGraphics
import CoreVideo
import UIKit
import XCTest

final class SubtitleVideoExporterTests: XCTestCase {
    @MainActor
    func testProductionRendererCreatesLegibleKaraokeFrames() async throws {
#if targetEnvironment(simulator)
        throw XCTSkip(
            "VideoToolbox encoding is unavailable in this simulator runtime; "
                + "the identical renderer is exercised by the Catalyst and device destinations."
        )
#else
        let source = try await makeVerticalFixture()
        let cues = [
            cue(
                words: ["PRIVATE", "AI", "RIGHT", "ON", "YOUR", "PHONE"],
                start: 0.25,
                wordDuration: 0.27,
                speaker: SubtitleSpeaker(laneID: "SPEAKER_00", displayName: nil)
            ),
            cue(
                words: ["FAST", "LOCAL", "SPECTACULAR"],
                start: 2.15,
                wordDuration: 0.48,
                speaker: SubtitleSpeaker(laneID: "SPEAKER_01", displayName: "Sarah")
            )
        ]
        let style = SubtitleRenderStyle(
            font: .rounded,
            fontSize1080: 64,
            textColor: .white,
            highlightColor: UIColor(red: 0.12, green: 0.96, blue: 0.64, alpha: 1),
            backgroundOpacity: 0.62
        )

        let output = try await SubtitleVideoExporter.export(
            videoURL: source,
            cues: cues,
            style: style,
            progress: { _ in }
        )

        let rendered = AVURLAsset(url: output)
        let duration = try await rendered.load(.duration).seconds
        let tracks = try await rendered.loadTracks(withMediaType: .video)
        let audioTracks = try await rendered.loadTracks(withMediaType: .audio)
        XCTAssertEqual(tracks.count, 1)
        XCTAssertEqual(audioTracks.count, 1, "Burn-in must preserve the source video's soundtrack")
        XCTAssertEqual(duration, fixtureDuration, accuracy: 0.08)
        XCTAssertGreaterThan(try output.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0, 10_000)

        let artifactDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "FrankenWhisperSubtitleVisualQA-\(UUID().uuidString)",
                isDirectory: true
            )
        try FileManager.default.createDirectory(
            at: artifactDirectory,
            withIntermediateDirectories: true
        )
        let copiedVideo = artifactDirectory.appendingPathComponent("karaoke-render.mp4")
        try FileManager.default.copyItem(at: output, to: copiedVideo)

        let generator = AVAssetImageGenerator(asset: rendered)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        for (index, seconds) in [0.08, 0.31, 0.72, 1.28, 2.22, 2.72, 3.35].enumerated() {
            let image = try generator.copyCGImage(
                at: CMTime(seconds: seconds, preferredTimescale: 600),
                actualTime: nil
            )
            let url = artifactDirectory.appendingPathComponent(
                String(format: "frame-%02d-%.2f.png", index, seconds)
            )
            guard let data = UIImage(cgImage: image).pngData() else {
                return XCTFail("Could not encode visual-QA frame")
            }
            try data.write(to: url, options: .atomic)
        }

        print("FW_SUBTITLE_VISUAL_QA=\(artifactDirectory.path)")
#endif
    }

    /// Explicitly gated visual QA for an operator-supplied real video. It is
    /// excluded from normal test runs so a minute-long export never slows the
    /// default suite. The CLI report provides genuine segment and diarization
    /// timing; only within-segment word positions are interpolated because the
    /// desktop CLI report does not expose the app's DTW word arrays.
    @MainActor
    func testOperatorSampleVideoRendersAnonymousSpeakersByColorOnly() async throws {
#if targetEnvironment(simulator)
        throw XCTSkip("VideoToolbox encoding is unavailable in this simulator runtime")
#else
        let environment = ProcessInfo.processInfo.environment
        guard let videoPath = environment["FW_SUBTITLE_SAMPLE_VIDEO"],
              let reportPath = environment["FW_SUBTITLE_SAMPLE_REPORT"],
              let outputRoot = environment["FW_SUBTITLE_VISUAL_QA_OUTPUT"]
        else {
            throw XCTSkip("Set the three FW_SUBTITLE_* variables to run real-video visual QA")
        }

        let reportData = try Data(contentsOf: URL(fileURLWithPath: reportPath))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let report = try decoder.decode(SampleCLIEnvelope.self, from: reportData)
        let timeline = interpolatedTimeline(from: report.result.segments)
        let cues = SubtitleTimeline.makeCues(
            from: timeline.words,
            segmentSpeakers: timeline.speakers
        )
        XCTAssertGreaterThanOrEqual(Set(cues.compactMap { $0.speaker?.laneID }).count, 2)
        XCTAssertTrue(
            cues.allSatisfy { $0.speaker?.displayName == nil },
            "Raw anonymous lanes must remain color-only"
        )

        let style = SubtitleRenderStyle(
            font: .rounded,
            fontSize1080: 64,
            textColor: .white,
            highlightColor: UIColor(red: 0.12, green: 0.96, blue: 0.64, alpha: 1),
            backgroundOpacity: 0.62
        )
        let output = try await SubtitleVideoExporter.export(
            videoURL: URL(fileURLWithPath: videoPath),
            cues: cues,
            style: style,
            progress: { _ in }
        )

        let artifactDirectory = URL(fileURLWithPath: outputRoot, isDirectory: true)
            .appendingPathComponent("sample-burn-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: artifactDirectory,
            withIntermediateDirectories: true
        )
        let copiedVideo = artifactDirectory.appendingPathComponent("speaker-colors-no-labels.mp4")
        try FileManager.default.copyItem(at: output, to: copiedVideo)

        let rendered = AVURLAsset(url: output)
        let generator = AVAssetImageGenerator(asset: rendered)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let sampleCues = evenlySampledCues(cues, count: 10)
        for (index, cue) in sampleCues.enumerated() {
            let seconds = cue.startSec + (cue.endSec - cue.startSec) * 0.45
            let image = try generator.copyCGImage(
                at: CMTime(seconds: seconds, preferredTimescale: 600),
                actualTime: nil
            )
            let url = artifactDirectory.appendingPathComponent(
                String(format: "frame-%02d-%.2f.png", index, seconds)
            )
            guard let data = UIImage(cgImage: image).pngData() else {
                return XCTFail("Could not encode real-video visual-QA frame")
            }
            try data.write(to: url, options: .atomic)
        }

        print("FW_SUBTITLE_SAMPLE_VISUAL_QA=\(artifactDirectory.path)")
#endif
    }

    private let fixtureDuration = 4.2

    private struct SampleCLIEnvelope: Decodable {
        var result: SampleTranscriptionResult
    }

    private struct SampleTranscriptionResult: Decodable {
        var segments: [SampleSegment]
    }

    private struct SampleSegment: Decodable {
        var startSec: Double
        var endSec: Double
        var text: String
        var speaker: String?
    }

    private struct SampleTiming: SubtitleTimingSource {
        var text: String
        var startSec: Double
        var endSec: Double
    }

    private func interpolatedTimeline(
        from segments: [SampleSegment]
    ) -> (words: [[SampleTiming]], speakers: [String?]) {
        var nestedWords: [[SampleTiming]] = []
        var speakers: [String?] = []
        for segment in segments where segment.endSec > segment.startSec {
            let tokens = segment.text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard !tokens.isEmpty else { continue }
            let weights = tokens.map { max(1, $0.count) }
            let totalWeight = max(1, weights.reduce(0, +))
            let duration = segment.endSec - segment.startSec
            var consumedWeight = 0
            let words = zip(tokens, weights).map { token, weight in
                let start = segment.startSec
                    + duration * Double(consumedWeight) / Double(totalWeight)
                consumedWeight += weight
                let end = segment.startSec
                    + duration * Double(consumedWeight) / Double(totalWeight)
                return SampleTiming(text: token, startSec: start, endSec: end)
            }
            nestedWords.append(words)
            speakers.append(segment.speaker)
        }
        return (nestedWords, speakers)
    }

    private func evenlySampledCues(_ cues: [SubtitleCue], count: Int) -> [SubtitleCue] {
        guard cues.count > count, count > 1 else { return cues }
        return (0..<count).map { index in
            let position = Double(index) * Double(cues.count - 1) / Double(count - 1)
            return cues[Int(position.rounded())]
        }
    }

    private func cue(
        words: [String],
        start: Double,
        wordDuration: Double,
        speaker: SubtitleSpeaker? = nil
    ) -> SubtitleCue {
        let timelineWords = words.enumerated().map { index, text in
            let wordStart = start + Double(index) * wordDuration
            return SubtitleTimelineWord(
                id: index,
                text: text,
                startSec: wordStart,
                endSec: wordStart + wordDuration * 0.88,
                speaker: speaker
            )
        }
        return SubtitleCue(
            id: Int(start * 100),
            words: timelineWords
        )
    }

    private func makeVerticalFixture() async throws -> URL {
        let width = 540
        let height = 960
        let framesPerSecond: Int32 = 30
        let totalFrames = Int(fixtureDuration * Double(framesPerSecond))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankenwhisper-source-\(UUID().uuidString).mov")
        // Motion JPEG has a deterministic software encoder in the simulator;
        // the production exporter still performs the real H.264/HEVC render.
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.jpeg,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoQualityKey: 0.72,
                ],
            ]
        )
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
        )
        XCTAssertTrue(writer.canAdd(input))
        writer.add(input)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime: .zero)

        for frameNumber in 0..<totalFrames {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(2))
            }
            guard let pool = adaptor.pixelBufferPool else {
                throw CocoaError(.coderInvalidValue)
            }
            var maybeBuffer: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &maybeBuffer) == kCVReturnSuccess,
                  let buffer = maybeBuffer
            else { throw CocoaError(.coderInvalidValue) }
            try drawFixtureFrame(
                buffer,
                frameNumber: frameNumber,
                totalFrames: totalFrames,
                width: width,
                height: height
            )
            let time = CMTime(value: CMTimeValue(frameNumber), timescale: framesPerSecond)
            XCTAssertTrue(adaptor.append(buffer, withPresentationTime: time))
        }

        input.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw writer.error ?? CocoaError(.fileWriteUnknown)
        }
        return try await addAudioTrack(to: url)
    }

    private func addAudioTrack(to videoURL: URL) async throws -> URL {
        let sampleRate = 44_100.0
        let audioURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankenwhisper-audio-\(UUID().uuidString).m4a")
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        ) else { throw CocoaError(.coderInvalidValue) }

        do {
            let file = try AVAudioFile(
                forWriting: audioURL,
                settings: [
                    AVFormatIDKey: kAudioFormatMPEG4AAC,
                    AVSampleRateKey: sampleRate,
                    AVNumberOfChannelsKey: 1,
                    AVEncoderBitRateKey: 96_000,
                ]
            )
            let frameCount = AVAudioFrameCount(fixtureDuration * sampleRate)
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: frameCount
            ), let samples = buffer.floatChannelData?[0]
            else { throw CocoaError(.coderInvalidValue) }
            buffer.frameLength = frameCount
            for index in 0..<Int(frameCount) {
                // A quiet two-tone bed makes accidental soundtrack loss
                // machine-detectable without making the QA clip obnoxious.
                let seconds = Double(index) / sampleRate
                samples[index] = Float(
                    0.025 * sin(2 * .pi * 220 * seconds)
                        + 0.012 * sin(2 * .pi * 440 * seconds)
                )
            }
            try file.write(from: buffer)
        }

        let videoAsset = AVURLAsset(url: videoURL)
        let audioAsset = AVURLAsset(url: audioURL)
        let composition = AVMutableComposition()
        guard let sourceVideo = try await videoAsset.loadTracks(withMediaType: .video).first,
              let sourceAudio = try await audioAsset.loadTracks(withMediaType: .audio).first,
              let videoTrack = composition.addMutableTrack(
                  withMediaType: .video,
                  preferredTrackID: kCMPersistentTrackID_Invalid
              ),
              let audioTrack = composition.addMutableTrack(
                  withMediaType: .audio,
                  preferredTrackID: kCMPersistentTrackID_Invalid
              )
        else { throw CocoaError(.coderInvalidValue) }

        let duration = CMTime(seconds: fixtureDuration, preferredTimescale: 600)
        try videoTrack.insertTimeRange(
            CMTimeRange(start: .zero, duration: duration),
            of: sourceVideo,
            at: .zero
        )
        try audioTrack.insertTimeRange(
            CMTimeRange(start: .zero, duration: duration),
            of: sourceAudio,
            at: .zero
        )
        videoTrack.preferredTransform = try await sourceVideo.load(.preferredTransform)

        guard let exporter = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetPassthrough
        ) else { throw CocoaError(.coderInvalidValue) }
        let muxedURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("frankenwhisper-av-fixture-\(UUID().uuidString).mov")
        exporter.outputURL = muxedURL
        exporter.outputFileType = .mov
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            exporter.exportAsynchronously { continuation.resume() }
        }
        guard exporter.status == .completed else {
            throw exporter.error ?? CocoaError(.fileWriteUnknown)
        }
        return muxedURL
    }

    private func drawFixtureFrame(
        _ buffer: CVPixelBuffer,
        frameNumber: Int,
        totalFrames: Int,
        width: Int,
        height: Int
    ) throws {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer),
              let context = CGContext(
                  data: baseAddress,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                  space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
                      | CGImageAlphaInfo.premultipliedFirst.rawValue
              )
        else { throw CocoaError(.coderInvalidValue) }

        let progress = CGFloat(frameNumber) / CGFloat(max(1, totalFrames - 1))
        let colors = [
            UIColor(red: 0.015, green: 0.055, blue: 0.085, alpha: 1).cgColor,
            UIColor(red: 0.035, green: 0.16, blue: 0.14, alpha: 1).cgColor,
            UIColor(red: 0.12, green: 0.025, blue: 0.19, alpha: 1).cgColor,
        ] as CFArray
        let gradient = CGGradient(
            colorsSpace: CGColorSpaceCreateDeviceRGB(),
            colors: colors,
            locations: [0, 0.52, 1]
        )!
        context.drawLinearGradient(
            gradient,
            start: CGPoint(x: 0, y: 0),
            end: CGPoint(x: CGFloat(width), y: CGFloat(height)),
            options: []
        )

        context.setBlendMode(.screen)
        for index in 0..<5 {
            let phase = progress * .pi * 2 + CGFloat(index) * 1.1
            let center = CGPoint(
                x: CGFloat(width) * (0.18 + 0.16 * CGFloat(index)) + sin(phase) * 38,
                y: CGFloat(height) * (0.35 + 0.07 * cos(phase * 0.7))
            )
            context.setFillColor(
                UIColor(
                    red: 0.08 + CGFloat(index) * 0.035,
                    green: 0.62,
                    blue: 0.48 + CGFloat(index) * 0.07,
                    alpha: 0.18
                ).cgColor
            )
            context.fillEllipse(in: CGRect(x: center.x - 95, y: center.y - 95, width: 190, height: 190))
        }
        context.setBlendMode(.normal)

        // A deliberately busy lower-third makes caption contrast and the social
        // safe-zone placement easy to judge without depending on UIKit's current
        // drawing context inside the simulator test runner.
        context.setFillColor(UIColor.black.withAlphaComponent(0.26).cgColor)
        context.fill(CGRect(x: 34, y: 720, width: 472, height: 126))
        context.setStrokeColor(
            UIColor(red: 0.20, green: 0.95, blue: 0.70, alpha: 0.72).cgColor
        )
        context.setLineWidth(4)
        context.stroke(CGRect(x: 34, y: 720, width: 472, height: 126))
        for index in 0..<8 {
            let barHeight = 18 + abs(sin(progress * 10 + CGFloat(index))) * 70
            context.fill(
                CGRect(
                    x: 62 + CGFloat(index) * 52,
                    y: 742,
                    width: 22,
                    height: barHeight
                )
            )
        }
    }
}
