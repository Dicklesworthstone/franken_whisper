// Model download, verification, and storage.
//
// The manifest constants mirror site/model-manifest.js and the compiled pins
// in src/sortformer_conformance.rs / src/denoise.rs — those are the source of
// truth and must move together if a model release is ever re-pinned. Files
// download in 32 MiB HTTP ranges (resumable from the bytes already on disk),
// hash incrementally as they arrive, and are refused on any digest mismatch.
// Each file lists upstream hosts in preference order (Hugging Face first —
// its CDN absorbs load that made GitHub releases 5xx under concurrency).
// Storage is Application Support, excluded from iCloud backup.

import CryptoKit
import Foundation

/// Rejects lifecycle messages that arrive at an engine actor after a newer
/// user intent. Unstructured task creation order is not actor mailbox order.
/// This small value type lives with the testable storage primitives so the
/// exact production ordering rule is exercised without loading native models.
struct EngineLifecycleFence {
    private(set) var latestToken: UInt64 = 0

    mutating func accept(_ token: UInt64) -> Bool {
        guard token >= latestToken else { return false }
        latestToken = token
        return true
    }
}

struct ModelFile {
    let label: String
    /// Where the app keeps the file, relative to the model directory.
    let relativePath: String
    let bytes: Int64
    let sha256: String
    /// Full download URLs in preference order.
    let urls: [String]
}

enum ModelManifest {
    private static let hfWhisperCpp = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
    private static let hfOrg =
        "https://huggingface.co/Dicklesworthstone/franken-whisper-models/resolve/main/"
    private static let ghSortformer =
        "https://github.com/Dicklesworthstone/franken_whisper/releases/download/sortformer-v2.1-f32-v1/"
    private static let ghTiny =
        "https://github.com/Dicklesworthstone/franken_whisper/releases/download/whisper-tiny-f16-v1/"

    static let whisper = ModelFile(
        label: "Whisper large-v3-turbo q8_0",
        relativePath: "whisper/ggml-large-v3-turbo-q8_0.bin",
        bytes: 874_188_075,
        sha256: "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
        urls: [
            hfWhisperCpp + "ggml-large-v3-turbo-q8_0.bin",
            hfOrg + "ggml-large-v3-turbo-q8_0.bin",
        ])

    // The keyboard/realtime lane deliberately uses the repo's pinned tiny
    // packages rather than paying large-v3-turbo's quality-oriented latency.
    // The multilingual tiny package preserves the app's auto-detect contract
    // without a model swap when the user changes languages.
    static let tiny = ModelFile(
        label: "Whisper tiny multilingual realtime model",
        relativePath: "whisper/ggml-tiny.bin",
        bytes: 77_691_713,
        sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        urls: [ghTiny + "ggml-tiny.bin"])

    static let sortformerWeights = ModelFile(
        label: "Sortformer diarizer",
        relativePath: "sortformer/weights.safetensors",
        bytes: 491_570_584,
        sha256: "487fa30cb0aa9799c77bd9985e6787962c3991fab8d4d576a4f1221d45298f6a",
        urls: [
            hfOrg + "weights.safetensors",
            ghSortformer + "weights.safetensors",
        ])

    static let sortformerReceipt = ModelFile(
        label: "Sortformer conversion receipt",
        relativePath: "sortformer/conversion-receipt.json",
        bytes: 653_208,
        sha256: "407c642f3d51b399514f6a35227b1c80886387472a44fb78f01b824d26318fb0",
        urls: [
            hfOrg + "conversion-receipt.json",
            ghSortformer + "conversion-receipt.json",
        ])

    static let denoiser = ModelFile(
        label: "FastEnhancer-S denoiser",
        relativePath: "denoiser/fastenhancer-s-48k-denoise.safetensors",
        bytes: 838_440,
        sha256: "28c1807fd9113e4ca09d3aacb2ecb07a742917321bfaced8b92598daffbd098b",
        urls: [
            hfOrg + "fastenhancer-s-48k-denoise.safetensors",
        ])

    static let files: [ModelFile] = [
        tiny, whisper, sortformerWeights, sortformerReceipt, denoiser,
    ]
    static let totalBytes = files.reduce(Int64(0)) { $0 + $1.bytes }
    static let chunkBytes: Int64 = 32 * 1024 * 1024
}

enum DownloadPhase: Equatable {
    case idle
    case downloading(label: String, done: Int64, total: Int64, eta: String)
    case verifying(label: String)
    case ready
    case failed(String)
}

@MainActor
@Observable
final class ModelStore {
    var phase: DownloadPhase = .idle
    var cachedBytes: Int64 = 0

    private var task: Task<Void, Never>?
    private var isClearing = false

    let modelDirectory: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("franken_whisper/models", isDirectory: true)
    }()

    func url(for file: ModelFile) -> URL {
        modelDirectory.appendingPathComponent(file.relativePath)
    }

    init() {
        refreshCachedBytes()
        // A receipt is published only after the full digest succeeds. Size
        // alone is insufficient: the app can be killed after the last range
        // lands but before verification finishes. Receipts retain fast launch
        // without trusting that crash-window file on the next run.
        if isComplete { phase = .ready }
    }

    var isComplete: Bool {
        ModelManifest.files.allSatisfy(hasValidVerificationReceipt)
    }

    private func sizeOnDisk(of file: ModelFile) -> Int64 {
        (try? FileManager.default.attributesOfItem(atPath: url(for: file).path)[.size] as? Int64)
            .flatMap { $0 } ?? 0
    }

    private func verificationReceiptURL(for file: ModelFile) -> URL {
        url(for: file).appendingPathExtension("verified")
    }

    private func hasValidVerificationReceipt(for file: ModelFile) -> Bool {
        guard sizeOnDisk(of: file) == file.bytes,
              let value = try? String(
                  contentsOf: verificationReceiptURL(for: file), encoding: .utf8)
        else { return false }
        return value == "\(file.sha256)\n"
    }

    private func markVerified(_ file: ModelFile) throws {
        try Data("\(file.sha256)\n".utf8)
            .write(to: verificationReceiptURL(for: file), options: .atomic)
    }

    private func removeVerificationReceipt(for file: ModelFile) {
        try? FileManager.default.removeItem(at: verificationReceiptURL(for: file))
    }

    func refreshCachedBytes() {
        cachedBytes = ModelManifest.files.reduce(Int64(0)) { $0 + sizeOnDisk(of: $1) }
    }

    func startDownload() {
        guard task == nil, !isClearing else { return }
        task = Task { [weak self] in
            await self?.run()
            self?.task = nil
        }
    }

    func cancelDownload() {
        task?.cancel()
    }

    func clear() async {
        guard !isClearing else { return }
        isClearing = true
        defer { isClearing = false }

        // Keep the single-flight handle installed until cancellation has
        // actually unwound. Clearing it eagerly allowed Retry/Clear to race an
        // older writer against the same multi-gigabyte destination files.
        if let activeTask = task {
            activeTask.cancel()
            await activeTask.value
        }
        do {
            if FileManager.default.fileExists(atPath: modelDirectory.path) {
                try FileManager.default.removeItem(at: modelDirectory)
            }
            refreshCachedBytes()
            phase = .idle
        } catch {
            // A failed removal is not a successful clear. Preserve the real
            // byte count and surface the filesystem error so the UI never
            // claims that multi-gigabyte model files disappeared when they did
            // not.
            refreshCachedBytes()
            phase = .failed("Could not clear downloaded models: \(error.localizedDescription)")
        }
    }

    private func run() async {
        // Partial ranges are durable resume state too. Keep the storage meter
        // honest after cancellation or a terminal host failure, not only after
        // a whole model happens to finish.
        defer { refreshCachedBytes() }
        do {
            try FileManager.default.createDirectory(
                at: modelDirectory, withIntermediateDirectories: true)
            var directory = modelDirectory
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try? directory.setResourceValues(values)

            var doneBytes: Int64 = 0
            for file in ModelManifest.files {
                try await ensure(file: file, alreadyDone: doneBytes)
                doneBytes += file.bytes
                refreshCachedBytes()
            }
            phase = .ready
        } catch is CancellationError {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func ensure(file: ModelFile, alreadyDone: Int64) async throws {
        let destination = url(for: file)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        if hasValidVerificationReceipt(for: file) { return }

        if sizeOnDisk(of: file) == file.bytes {
            // A legacy cache or interrupted final verification has no durable
            // receipt. Authenticate it once; subsequent launches stay fast.
            phase = .verifying(label: file.label)
            if try await digest(of: destination) == file.sha256 {
                try markVerified(file)
                return
            }
            try FileManager.default.removeItem(at: destination)
            removeVerificationReceipt(for: file)
        }

        // Try each host in order; a host failure mid-file keeps the bytes
        // already written, so the next host resumes from that offset.
        var lastError: Error = EngineError.invalid("\(file.label): no hosts")
        for url in file.urls {
            do {
                try await download(file: file, from: url, to: destination, alreadyDone: alreadyDone)
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // URLSession commonly reports a cancelled async request as
                // URLError.cancelled rather than CancellationError. Preserve
                // the task-level meaning so Cancel returns to idle instead of
                // rotating hosts and eventually presenting a false failure.
                try Task.checkCancellation()
                lastError = error
            }
        }
        throw lastError
    }

    private func download(
        file: ModelFile, from source: String, to destination: URL, alreadyDone: Int64
    ) async throws {
        var offset = sizeOnDisk(of: file)
        // Any write invalidates prior authentication. Usually no receipt is
        // present here, but removing it defensively keeps future refactors from
        // ever pairing a newly-written file with stale trust metadata.
        removeVerificationReceipt(for: file)
        if offset > file.bytes {
            try FileManager.default.removeItem(at: destination)
            offset = 0
        }
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
            offset = 0
        }

        let sink = try FileHandle(forWritingTo: destination)
        defer { try? sink.close() }
        try sink.seekToEnd()

        // A resume cannot reuse a streamed hash for the prefix; hash the whole
        // file after. A from-zero download hashes as it goes.
        var live: SHA256? = offset == 0 ? SHA256() : nil
        let started = Date()
        let startedOffset = offset

        while offset < file.bytes {
            try Task.checkCancellation()
            let end = min(offset + ModelManifest.chunkBytes, file.bytes) - 1
            guard let url = URL(string: source) else {
                throw EngineError.invalid("\(file.label): bad URL")
            }
            var request = URLRequest(url: url)
            request.setValue("bytes=\(offset)-\(end)", forHTTPHeaderField: "Range")
            let (data, response) = try await URLSession.shared.data(for: request)
            let received = try Self.validatedResponseLength(
                response: response,
                dataCount: data.count,
                requestedRange: offset...end,
                fileBytes: file.bytes,
                label: file.label)
            try sink.write(contentsOf: data)
            live?.update(data: data)
            offset += received

            let elapsed = Date().timeIntervalSince(started)
            let rate = elapsed > 1 ? Double(offset - startedOffset) / elapsed : 0
            let remaining =
                rate > 0 ? Double(ModelManifest.totalBytes - alreadyDone - offset) / rate : 0
            let eta = rate > 0 ? Self.formatEta(seconds: remaining) : "estimating…"
            phase = .downloading(
                label: file.label, done: alreadyDone + offset, total: ModelManifest.totalBytes,
                eta: eta)
        }
        try sink.close()

        phase = .verifying(label: file.label)
        let digestHex: String
        if let live {
            digestHex = live.finalize().map { String(format: "%02x", $0) }.joined()
        } else {
            digestHex = try await digest(of: destination)
        }
        guard digestHex == file.sha256 else {
            try? FileManager.default.removeItem(at: destination)
            removeVerificationReceipt(for: file)
            throw EngineError.invalid("\(file.label): digest mismatch; cleared for retry")
        }
        try markVerified(file)
    }

    /// Validate that a range response contains exactly the bytes it claims.
    /// A zero-length 206 previously left `offset` unchanged and spun forever;
    /// an unchecked Content-Range could append the wrong region and corrupt a
    /// resumable download before the final digest eventually rejected it.
    nonisolated static func validatedResponseLength(
        response: URLResponse,
        dataCount: Int,
        requestedRange: ClosedRange<Int64>,
        fileBytes: Int64,
        label: String
    ) throws -> Int64 {
        guard let http = response as? HTTPURLResponse, dataCount > 0 else {
            throw EngineError.invalid("\(label): empty or non-HTTP range response")
        }

        let received = Int64(dataCount)
        if http.statusCode == 200 {
            guard requestedRange.lowerBound == 0, received == fileBytes else {
                throw EngineError.invalid("\(label): server ignored the requested range")
            }
            return received
        }

        guard http.statusCode == 206,
              let header = http.value(forHTTPHeaderField: "Content-Range")
        else {
            throw EngineError.invalid("\(label): range fetch failed")
        }
        let components = header.split(separator: " ", maxSplits: 1)
        guard components.count == 2,
              components[0].lowercased() == "bytes"
        else {
            throw EngineError.invalid("\(label): malformed Content-Range")
        }
        let rangeAndTotal = components[1].split(separator: "/", maxSplits: 1)
        let bounds = rangeAndTotal.first?.split(separator: "-", maxSplits: 1) ?? []
        guard rangeAndTotal.count == 2,
              bounds.count == 2,
              let responseStart = Int64(bounds[0]),
              let responseEnd = Int64(bounds[1]),
              let responseTotal = Int64(rangeAndTotal[1]),
              responseStart == requestedRange.lowerBound,
              responseEnd >= responseStart,
              responseEnd <= requestedRange.upperBound,
              responseTotal == fileBytes,
              received == responseEnd - responseStart + 1
        else {
            throw EngineError.invalid("\(label): Content-Range does not match the requested bytes")
        }
        return received
    }

    /// Streaming SHA-256 of a file, 8 MiB at a time, off the main actor.
    private nonisolated func digest(of url: URL) async throws -> String {
        let digestTask = Task.detached(priority: .utility) {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hasher = SHA256()
            while true {
                try Task.checkCancellation()
                let data = try handle.read(upToCount: 8 * 1024 * 1024) ?? Data()
                if data.isEmpty { break }
                hasher.update(data: data)
            }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }
        // Detached work does not inherit cancellation from its waiter. Bridge
        // it explicitly so Cancel/Clear cannot appear frozen while a stale
        // multi-gigabyte verification continues reading the whole file.
        return try await withTaskCancellationHandler {
            try await digestTask.value
        } onCancel: {
            digestTask.cancel()
        }
    }

    private static func formatEta(seconds: Double) -> String {
        let total = Int(seconds.rounded())
        let minutes = total / 60
        return minutes > 0 ? "~\(minutes)m \(total % 60)s left" : "~\(total)s left"
    }
}
