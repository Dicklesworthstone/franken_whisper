import AVFoundation
import CoreTransferable
import Foundation
import UniformTypeIdentifiers

struct VideoInput: Equatable, Sendable {
    let videoURL: URL
    let audioURL: URL
    let name: String
    let duration: Double
    let displayWidth: Double
    let displayHeight: Double
    /// The audio track's position in the original video timeline. Audio is
    /// extracted to start at zero for transcription; burn-in adds this value
    /// back so captions remain aligned with edited or delayed-audio videos.
    let audioTimelineOffset: Double

    var aspectRatio: Double {
        guard displayWidth > 0, displayHeight > 0 else { return 16.0 / 9.0 }
        return displayWidth / displayHeight
    }
}

struct PickedVideo: Transferable, Sendable {
    let localURL: URL
    let originalName: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let destination = try VideoImportService.copyIntoWorkspace(received.file)
            return Self(
                localURL: destination,
                originalName: received.file.lastPathComponent
            )
        }
    }
}

enum VideoImportError: LocalizedError {
    case notPlayable
    case missingVideo
    case missingAudio
    case invalidDuration
    case exportUnavailable
    case exportFailed(String)

    var errorDescription: String? {
        switch self {
        case .notPlayable:
            "That video is not playable on this device."
        case .missingVideo:
            "The selected item does not contain a video track."
        case .missingAudio:
            "That video has no playable audio track to transcribe."
        case .invalidDuration:
            "The video duration could not be read."
        case .exportUnavailable:
            "iOS could not create a local audio track from that video."
        case .exportFailed(let reason):
            "The video's audio could not be prepared: \(reason)"
        }
    }
}

enum VideoImportService {
    private static let directoryName = "VideoImports"

    static func prepareExternalVideo(_ source: URL) async throws -> VideoInput {
        let managed = try await Task.detached(priority: .userInitiated) {
            try copyIntoWorkspace(source)
        }.value
        do {
            return try await prepareManagedVideo(managed, displayName: source.lastPathComponent)
        } catch {
            try? FileManager.default.removeItem(at: managed)
            throw error
        }
    }

    static func preparePickedVideo(_ picked: PickedVideo) async throws -> VideoInput {
        do {
            return try await prepareManagedVideo(
                picked.localURL,
                displayName: picked.originalName
            )
        } catch {
            // FileRepresentation has already copied this item into our cache.
            // If validation or audio extraction fails, no later owner exists to
            // reclaim that private movie.
            try? FileManager.default.removeItem(at: picked.localURL)
            throw error
        }
    }

    static func discard(_ input: VideoInput) {
        let directory = workspaceDirectory()
        for url in [input.videoURL, input.audioURL] where url.deletingLastPathComponent() == directory {
            try? FileManager.default.removeItem(at: url)
        }
    }

    fileprivate static func copyIntoWorkspace(_ source: URL) throws -> URL {
        let directory = workspaceDirectory()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let ext = source.pathExtension.isEmpty ? "mov" : source.pathExtension.lowercased()
        let destination = directory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(ext)
        try FileManager.default.copyItem(at: source, to: destination)
        return destination
    }

    private static func prepareManagedVideo(
        _ videoURL: URL,
        displayName: String
    ) async throws -> VideoInput {
        let asset = AVURLAsset(url: videoURL)
        guard try await asset.load(.isPlayable) else { throw VideoImportError.notPlayable }

        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = videoTracks.first else { throw VideoImportError.missingVideo }
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard let audioTrack = audioTracks.first else { throw VideoImportError.missingAudio }
        let audioTimeRange = try await audioTrack.load(.timeRange)
        guard audioTimeRange.duration.isNumeric, audioTimeRange.duration.seconds > 0 else {
            throw VideoImportError.missingAudio
        }

        let duration = try await asset.load(.duration)
        let seconds = duration.seconds
        guard seconds.isFinite, seconds > 0 else { throw VideoImportError.invalidDuration }

        let naturalSize = try await videoTrack.load(.naturalSize)
        let transform = try await videoTrack.load(.preferredTransform)
        let displayRect = CGRect(origin: .zero, size: naturalSize).applying(transform)
        let displayWidth = abs(displayRect.width)
        let displayHeight = abs(displayRect.height)

        let audioURL = workspaceDirectory()
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("m4a")
        do {
            try await extractAudio(
                from: asset,
                track: audioTrack,
                sourceRange: audioTimeRange,
                to: audioURL
            )
        } catch {
            // AVAssetExportSession may leave a partial container behind after a
            // cancellation or codec failure; never let repeated imports leak it.
            try? FileManager.default.removeItem(at: audioURL)
            throw error
        }

        return VideoInput(
            videoURL: videoURL,
            audioURL: audioURL,
            name: displayName,
            duration: seconds,
            displayWidth: displayWidth,
            displayHeight: displayHeight,
            // Preserve negative edit-list starts as well as delayed audio. The
            // timeline shifter clips captions at video time zero; clamping here
            // would instead move every caption in an early-starting track late.
            audioTimelineOffset: audioTimeRange.start.seconds.isFinite
                ? audioTimeRange.start.seconds
                : 0
        )
    }

    private static func extractAudio(
        from asset: AVAsset,
        track: AVAssetTrack,
        sourceRange: CMTimeRange,
        to destination: URL
    ) async throws {
        let composition = AVMutableComposition()
        guard let compositionTrack = composition.addMutableTrack(
            withMediaType: .audio,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else { throw VideoImportError.exportUnavailable }

        do {
            try compositionTrack.insertTimeRange(
                sourceRange,
                of: track,
                at: .zero
            )
        } catch {
            throw VideoImportError.exportFailed(error.localizedDescription)
        }

        guard let exporter = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetAppleM4A
        ) else { throw VideoImportError.exportUnavailable }
        exporter.outputURL = destination
        exporter.outputFileType = .m4a
        exporter.shouldOptimizeForNetworkUse = false

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            exporter.exportAsynchronously {
                continuation.resume()
            }
        }
        switch exporter.status {
        case .completed:
            return
        case .cancelled:
            throw CancellationError()
        case .failed:
            throw VideoImportError.exportFailed(
                exporter.error?.localizedDescription ?? "unknown AVFoundation error"
            )
        default:
            throw VideoImportError.exportFailed("audio export ended unexpectedly")
        }
    }

    private static func workspaceDirectory() -> URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(directoryName, isDirectory: true)
    }
}
