// The Rust engine behind an actor: one handle, strictly serialized access,
// exactly the concurrency contract fw-ios/include/fw_ios.h documents.
//
// Progress plumbing: the engine's per-window heartbeat and live-transcript
// feed arrive through two process-wide C callbacks. The trampolines capture
// nothing (a C function pointer in Swift cannot); the opaque ctx is a single
// process-lifetime HookBox whose closures the app swaps per run.

import FwCore
import Foundation

enum EngineError: LocalizedError {
    case native(code: Int32, message: String)
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .native(_, let message): message
        case .invalid(let message): message
        }
    }

    /// Code 6 is the header's stable "cancelled via fw_request_cancel".
    var isCancellation: Bool {
        if case .native(let code, _) = self { return code == 6 }
        return false
    }

    static func lastFromNative(code: Int32) -> EngineError {
        .native(code: code, message: String(cString: fw_last_error_message()))
    }
}

/// Process-lifetime box the C trampolines bounce through. The closures must
/// be fast and non-blocking: they run on the engine's decode thread.
final class EngineHooks: @unchecked Sendable {
    static let shared = EngineHooks()
    private let lock = NSLock()
    private var spanHandler: (@Sendable (String, Double) -> Void)?
    private var segmentsHandler: (@Sendable (String) -> Void)?

    func set(
        span: (@Sendable (String, Double) -> Void)?,
        segments: (@Sendable (String) -> Void)?
    ) {
        lock.lock()
        spanHandler = span
        segmentsHandler = segments
        lock.unlock()
    }

    fileprivate func fire(span: String, value: Double) {
        lock.lock()
        let handler = spanHandler
        lock.unlock()
        handler?(span, value)
    }

    fileprivate func fire(segmentsJSON: String) {
        lock.lock()
        let handler = segmentsHandler
        lock.unlock()
        handler?(segmentsJSON)
    }

    /// Installed once per process, before the first engine call.
    fileprivate static let install: Void = {
        let ctx = Unmanaged.passUnretained(EngineHooks.shared).toOpaque()
        fw_set_progress_callback(
            { ctx, span, value in
                guard let ctx, let span else { return }
                Unmanaged<EngineHooks>.fromOpaque(ctx).takeUnretainedValue()
                    .fire(span: String(cString: span), value: value)
            }, ctx)
        fw_set_segments_callback(
            { ctx, json in
                guard let ctx, let json else { return }
                Unmanaged<EngineHooks>.fromOpaque(ctx).takeUnretainedValue()
                    .fire(segmentsJSON: String(cString: json))
            }, ctx)
    }()
}

struct StageInfo: Decodable {
    var audioSec: Double
    var skippedLeadingSec: Double
    var denoised: Bool
}

// ── The actor ──────────────────────────────────────────────────────────────

/// All engine access lives here. The Rust handle is not thread-safe; the
/// actor's serialization is the whole safety argument, so no engine call may
/// leave this type. Load and run are long BLOCKING calls that park one
/// cooperative-pool thread — the same accepted tradeoff as FrankenTTS.
actor Engine {
    private var handle: OpaquePointer?
    private var loadedModel: URL?
    private var lifecycleFence = EngineLifecycleFence()

    static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    var isLoaded: Bool { handle != nil }
    var hasDiarizer: Bool { fw_engine_has_sortformer(handle) == 1 }
    var hasDenoiser: Bool { fw_engine_has_denoiser(handle) == 1 }

    /// Hydrates the whisper model from its ggml file. Multi-second on a
    /// phone; the caller watches EngineHooks for "whisper:*" stage markers.
    func load(modelPath: URL, lifecycleToken: UInt64) throws {
        guard lifecycleFence.accept(lifecycleToken) else {
            throw EngineError.invalid("engine lifecycle operation was superseded")
        }
        let requestedModel = modelPath.standardizedFileURL
        if handle != nil, loadedModel == requestedModel { return }
        // The physical-device profiler and realtime lane can request different
        // model files. A non-nil handle is idempotent only for the same file.
        if let handle { fw_engine_close(handle) }
        handle = nil
        loadedModel = nil
        _ = EngineHooks.install
        guard let opened = fw_engine_open(modelPath.path) else {
            throw EngineError.lastFromNative(code: 3)
        }
        handle = opened
        loadedModel = requestedModel
    }

    /// Authenticate + load the Sortformer diarizer (receipt + package paths).
    func loadSortformer(receipt: URL, package: URL) throws {
        guard let handle else { throw EngineError.invalid("engine not loaded") }
        let code = fw_engine_load_sortformer(handle, receipt.path, package.path)
        guard code == 0 else { throw EngineError.lastFromNative(code: code) }
    }

    func loadDenoiser(at path: URL) throws {
        guard let handle else { throw EngineError.invalid("engine not loaded") }
        let code = fw_engine_load_denoiser(handle, path.path)
        guard code == 0 else { throw EngineError.lastFromNative(code: code) }
    }

    /// Drops everything (model, diarizer, denoiser, staged PCM), freeing
    /// ~1.5 GB. Safe to call at any idle moment; the next run reloads.
    func unload(lifecycleToken: UInt64) {
        // A delayed memory-pressure task must not close a model claimed by a
        // newer foreground assembly that happened to reach this actor first.
        guard lifecycleFence.accept(lifecycleToken) else { return }
        if let handle {
            fw_engine_close(handle)
        }
        handle = nil
        loadedModel = nil
    }

    /// Stage microphone PCM (16 kHz mono, [-1, 1]) for the next run.
    func stage(pcm samples: [Float], denoise: Bool) throws -> StageInfo {
        guard let handle else { throw EngineError.invalid("engine not loaded") }
        var out: UnsafeMutablePointer<CChar>?
        let code = samples.withUnsafeBufferPointer { buffer in
            fw_stage_pcm(handle, buffer.baseAddress, buffer.count, denoise, &out)
        }
        return try Self.takeJSON(StageInfo.self, code: code, out: out)
    }

    /// Decode + stage an imported audio file (mp3/m4a/wav bytes).
    func stage(fileData: Data, ext: String, denoise: Bool) throws -> StageInfo {
        guard let handle else { throw EngineError.invalid("engine not loaded") }
        var out: UnsafeMutablePointer<CChar>?
        let code = fileData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            fw_stage_audio_file(
                handle, raw.bindMemory(to: UInt8.self).baseAddress, raw.count,
                ext.lowercased(), denoise, &out)
        }
        return try Self.takeJSON(StageInfo.self, code: code, out: out)
    }

    /// Run the fused whisper (+ optional Sortformer + fusion) pipeline over
    /// the staged PCM. Minutes for long audio; live windows stream through
    /// EngineHooks while this call is still running.
    func run(options: RunOptions) throws -> Transcription {
        guard let handle else { throw EngineError.invalid("engine not loaded") }
        var out: UnsafeMutablePointer<CChar>?
        let code = fw_run_prepared(handle, options.json, &out)
        return try Self.takeJSON(Transcription.self, code: code, out: out)
    }

    /// Cooperative cancel of the in-flight run: process-wide flag, checked at
    /// the engine's checkpoints. Deliberately nonisolated — the whole point
    /// is to interrupt a call currently holding the actor.
    nonisolated func requestCancel() {
        fw_request_cancel()
    }

    nonisolated func resetCancel() {
        fw_reset_cancel()
    }

    private static func takeJSON<T: Decodable>(
        _ type: T.Type, code: Int32, out: UnsafeMutablePointer<CChar>?
    ) throws -> T {
        guard code == 0, let out else {
            if let out { fw_string_free(out) }
            throw EngineError.lastFromNative(code: code)
        }
        defer { fw_string_free(out) }
        let json = String(cString: out)
        do {
            return try decoder().decode(T.self, from: Data(json.utf8))
        } catch {
            throw EngineError.invalid("engine returned unparseable JSON: \(error)")
        }
    }
}
