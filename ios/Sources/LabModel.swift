// All app state and orchestration: model download → engine assembly → audio
// input (mic or file) → the run (with live windows) → results and export.
//
// The engine's long calls are blocking actor methods; every one runs inside
// a Task so the UI stays responsive, and stale publishes are guarded by a
// generation counter (a cancelled run must never overwrite a newer one).

import AVFoundation
import CryptoKit
import Foundation
import SwiftUI
import UIKit
import UniformTypeIdentifiers

private final class WhisperProfileSpanCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String: [Double]] = [:]

    func reset() {
        lock.lock()
        values.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    func record(span: String, value: Double) {
        lock.lock()
        values[span, default: []].append(value)
        lock.unlock()
    }

    func snapshot() -> [String: [Double]] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

/// Device-local batch-transcription forecast. The first run uses a coarse hardware
/// tier lookup; completed runs teach it real-time factor, decode-window cadence, and
/// post-decode tail cost separately for diarized and plain transcripts. No estimate is
/// exposed until a real 30-second inference window completes.
private struct WhisperAdaptiveETA {
    private static let version = "v1"

    private var audioSeconds = 0.0
    private var windowsTotal = 1
    private var includesDiarization = false
    private var decodeStartedElapsed: TimeInterval?
    private var diarizationStartedElapsed: TimeInterval?
    private var fuseStartedElapsed: TimeInterval?
    private var hasMeasuredWindow = false
    private(set) var predictedFinishElapsed: TimeInterval?

    mutating func reset(includesDiarization: Bool) {
        audioSeconds = 0
        windowsTotal = 1
        self.includesDiarization = includesDiarization
        decodeStartedElapsed = nil
        diarizationStartedElapsed = nil
        fuseStartedElapsed = nil
        hasMeasuredWindow = false
        predictedFinishElapsed = nil
    }

    mutating func beginDecodedAudio(
        seconds: Double,
        windows: Int,
        elapsed: TimeInterval
    ) {
        audioSeconds = max(0.1, seconds)
        windowsTotal = max(1, windows)
        decodeStartedElapsed = elapsed
    }

    mutating func observe(state: LabModel.RunState, elapsed: TimeInterval) {
        switch state {
        case .running(let done, let total, let stage) where stage == "decoding" && done > 0:
            hasMeasuredWindow = true
            windowsTotal = max(1, total)
            let decodeStart = decodeStartedElapsed ?? 0
            let measuredWindowSeconds = max(0.05, elapsed - decodeStart) / Double(done)
            let remainingWindows = Double(max(0, total - done))
            let tail = includesDiarization
                ? learnedTailSecondsPerAudioSecond * audioSeconds + learnedFuseSeconds
                : learnedFuseSeconds
            let measuredCandidate = elapsed + measuredWindowSeconds * remainingWindows + tail
            let priorFromRTF = learnedRealTimeFactor * audioSeconds
            let priorFromWindows = decodeStart
                + learnedSecondsPerWindow * Double(total)
                + tail
            let priorCandidate = (priorFromRTF + priorFromWindows) * 0.5
            let confidence = min(0.90, 0.48 + 0.42 * Double(done) / Double(max(1, total)))
            smooth(
                toward: max(
                    elapsed + 0.5,
                    measuredCandidate * confidence + priorCandidate * (1 - confidence)
                )
            )

        case .running(_, _, let stage) where stage == "labeling speakers":
            guard hasMeasuredWindow else { return }
            if diarizationStartedElapsed == nil { diarizationStartedElapsed = elapsed }
            let stageStart = diarizationStartedElapsed ?? elapsed
            smooth(
                toward: stageStart
                    + learnedTailSecondsPerAudioSecond * audioSeconds
                    + learnedFuseSeconds
            )

        case .running(_, _, let stage) where stage == "fusing":
            guard hasMeasuredWindow else { return }
            if fuseStartedElapsed == nil { fuseStartedElapsed = elapsed }
            smooth(toward: (fuseStartedElapsed ?? elapsed) + learnedFuseSeconds)

        case .idle, .done, .failed:
            predictedFinishElapsed = nil
        case .staging, .running:
            break
        }
    }

    mutating func finish(elapsed: TimeInterval) {
        guard audioSeconds > 0, elapsed.isFinite, elapsed > 0 else {
            predictedFinishElapsed = nil
            return
        }
        Self.update(key: realTimeFactorKey, sample: elapsed / audioSeconds)
        if let decodeStartedElapsed {
            let decodeEnd = diarizationStartedElapsed ?? fuseStartedElapsed ?? elapsed
            Self.update(
                key: secondsPerWindowKey,
                sample: max(0.01, decodeEnd - decodeStartedElapsed) / Double(windowsTotal)
            )
        }
        if let diarizationStartedElapsed {
            let tailEnd = fuseStartedElapsed ?? elapsed
            Self.update(
                key: tailSecondsPerAudioSecondKey,
                sample: max(0.01, tailEnd - diarizationStartedElapsed) / audioSeconds
            )
        }
        if let fuseStartedElapsed {
            Self.update(
                key: fuseSecondsKey,
                sample: max(0.01, elapsed - fuseStartedElapsed)
            )
        }
        predictedFinishElapsed = nil
    }

    private mutating func smooth(toward candidate: TimeInterval) {
        if let old = predictedFinishElapsed {
            let smoothed = old * 0.68 + candidate * 0.32
            predictedFinishElapsed = min(smoothed, old + 3.0)
        } else {
            predictedFinishElapsed = candidate
        }
    }

    private var lane: String { includesDiarization ? "diarized" : "plain" }
    private var realTimeFactorKey: String { "whisper.eta.\(Self.version).\(lane).rtf" }
    private var secondsPerWindowKey: String { "whisper.eta.\(Self.version).\(lane).secondsPerWindow" }
    private var tailSecondsPerAudioSecondKey: String {
        "whisper.eta.\(Self.version).\(lane).tailSecondsPerAudioSecond"
    }
    private var fuseSecondsKey: String { "whisper.eta.\(Self.version).\(lane).fuseSeconds" }

    private var learnedRealTimeFactor: Double {
        let learned = UserDefaults.standard.double(forKey: realTimeFactorKey)
        if learned > 0 { return learned }
        let gib = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let plain: Double = gib >= 8 ? 0.65 : (gib >= 6 ? 0.82 : 1.15)
        return plain + (includesDiarization ? 0.22 : 0)
    }

    private var learnedTailSecondsPerAudioSecond: Double {
        let learned = UserDefaults.standard.double(forKey: tailSecondsPerAudioSecondKey)
        return learned > 0 ? learned : 0.18
    }

    private var learnedSecondsPerWindow: Double {
        let learned = UserDefaults.standard.double(forKey: secondsPerWindowKey)
        if learned > 0 { return learned }
        let gib = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        return gib >= 8 ? 16 : (gib >= 6 ? 21 : 29)
    }

    private var learnedFuseSeconds: Double {
        let learned = UserDefaults.standard.double(forKey: fuseSecondsKey)
        return learned > 0 ? learned : 0.8
    }

    private static func update(key: String, sample: Double) {
        guard sample.isFinite, sample > 0 else { return }
        let defaults = UserDefaults.standard
        let old = defaults.double(forKey: key)
        defaults.set(old > 0 ? old * 0.72 + sample * 0.28 : sample, forKey: key)
    }
}

@MainActor
@Observable
final class LabModel {
    // ── Collaborators ──────────────────────────────────────────────────────
    let store = ModelStore()
    let recorder = AudioRecorder()
    let history = TranscriptHistoryStore()
    private let engine = Engine()
    /// A separate resident tiny model for realtime/keyboard work. Keeping it
    /// independent prevents the full-accuracy large-v3-turbo lane from being
    /// weakened or unloaded when the low-latency lane is used.
    private let liveEngine = Engine()

    // ── Engine state ───────────────────────────────────────────────────────
    enum EngineState: Equatable {
        case notLoaded
        case loading(stage: String)
        case ready
        case failed(String)
    }
    var engineState: EngineState = .notLoaded
    var liveEngineState: EngineState = .notLoaded
    var diarizerLoaded = false
    var denoiserLoaded = false
    private(set) var enginePausedForMemoryPressure = false

    // ── Input ──────────────────────────────────────────────────────────────
    enum Input: Equatable {
        case none
        case recording(pcm: [Float])
        case file(data: Data, ext: String, name: String)
        case video(media: VideoInput, audioData: Data)

        var seconds: Double? {
            switch self {
            case .none: nil
            case .recording(let pcm): Double(pcm.count) / 16_000.0
            case .file: nil  // unknown until staged
            case .video(let media, _): media.duration
            }
        }

        var isVideo: Bool {
            if case .video = self { return true }
            return false
        }
    }
    var input: Input = .none
    var isImporting = false
    var inputName: String {
        switch input {
        case .none: ""
        case .recording: "microphone capture"
        case .file(_, _, let name): name
        case .video(let media, _): media.name
        }
    }

    var videoInput: VideoInput? {
        guard case .video(let media, _) = input else { return nil }
        return media
    }

    // ── Options ────────────────────────────────────────────────────────────
    var diarize = true
    var denoise = true
    var wordTimestamps = false
    /// Batch-mode task selection. Live keyboard dictation deliberately stays
    /// source-language transcription for predictable latency and insertion.
    var translateToEnglish = false
    var language = "auto"
    /// The website's speaker-names field: comma- or newline-separated names
    /// and titles ("Jeff Emanuel (host), Dr. Sarah Chen (guest)"). Feeds
    /// Whisper's decoding prompt so the names come out spelled right, then
    /// maps onto detected SPEAKER_NN lanes in order of first appearance.
    var speakerNamesRaw = ""
    static let languages: [(code: String, label: String)] = [
        ("auto", "Auto-detect"), ("en", "English"), ("es", "Spanish"), ("fr", "French"),
        ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"),
        ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"), ("ru", "Russian"),
        ("uk", "Ukrainian"), ("pl", "Polish"), ("hi", "Hindi"), ("ar", "Arabic"),
    ]

    // ── Run state ──────────────────────────────────────────────────────────
    enum RunState: Equatable {
        case idle
        case staging
        case running(windowsDone: Int, windowsTotal: Int, stage: String)
        case done
        case failed(String)
    }
    var runState: RunState = .idle {
        didSet {
            let now = Date()
            runActivity.transition(
                to: runState,
                elapsed: elapsed(at: now),
                emittedSegments: liveSegments.count
            )
            eta.observe(state: runState, elapsed: elapsed(at: now))
            estimatedFinishElapsed = eta.predictedFinishElapsed
        }
    }
    /// Raw `SPEAKER_NN` lane → the user's display name, seeded from the
    /// pre-run field when the result lands and editable afterwards through
    /// the transcript card's rename rows.
    var speakerNameMap: [String: String] = [:]
    var liveSegments: [TranscriptSegment] = []
    /// Live segments arrive in the trimmed timebase (fw_ios.h); add this to
    /// their times for display. The final result already has it added back.
    var liveOffsetSec: Double = 0
    var result: Transcription?
    /// Captures the task that produced `result`; the mutable option may change
    /// while somebody is reading or exporting the finished transcript.
    var resultWasTranslated = false
    var wallSeconds: Double = 0
    var runAudioSeconds: Double = 0
    var estimatedFinishElapsed: TimeInterval?
    var lastError: String?

    // ── Live dictation ────────────────────────────────────────────────────
    enum LiveDictationState: Equatable {
        case idle
        case starting(String)
        /// The microphone session is alive locally, but samples are discarded
        /// until the keyboard sends its next Start command.
        case armed
        case listening
        case finishing
        case failed(String)
    }
    var liveDictationState: LiveDictationState = .idle {
        didSet { publishLiveActivityState() }
    }
    var liveDictationText = ""
    var liveLastPhrase = ""
    /// Mutable AlignAtt tail. It is visible in the containing app but never
    /// crosses the App Group or enters the host text field until Rust commits
    /// it, preserving the append-only keyboard contract.
    var livePartialText = ""
    var liveCommittedDeltaCount = 0
    var livePolicyHoldingBack = false
    var liveIsPreviewDecoding = false
    var liveQueuedUtterances = 0
    private var liveDetector = LiveUtteranceDetector()
    private var liveCaptureTask: Task<Void, Never>?
    private var liveDecodeTask: Task<Void, Never>?
    private var liveStartTask: Task<Void, Never>?
    private struct PendingLiveUtterance {
        var pcm: [Float]
        var committedSamples: Int
    }
    private var liveQueue: [PendingLiveUtterance] = []
    private var liveUtteranceCounter = 0
    private var activeLiveUtteranceID: Int?
    private var activeLiveCommittedSamples = 0
    private var activeLiveLastScheduledSamples = 0
    private var liveSessionID = ""
    private var liveServiceID = ""
    private var liveRevision = 0
    private var liveStopRequested = false
    private var liveEndSessionRequested = false
    private var liveSessionExpiresAt: Date?
    private var liveMeterLevel: Float = 0
    private var liveSpectrum = [Float](repeating: 0, count: LiveSpectrumAnalyzer.bandCount)
    private var lastLiveMeterPublishedAt: TimeInterval = 0
    var keyboardHandoffVisible = false
    private var keyboardStartPending = false
    private var lastHandledDictationCommandID = ""
    private var liveFinishBackgroundTask: UIBackgroundTaskIdentifier = .invalid

    /// One explicit activation buys a useful native-feeling work session.
    /// It still expires automatically so the microphone is never left alive
    /// indefinitely after the user forgets about it.
    private static let liveSessionDuration: TimeInterval = 60 * 60

    /// Mega-kernel discipline: the product split has an emergency kill switch.
    /// Set FW_IOS_LIVE_FAST_MODEL=0 before first engine use to route live work
    /// through the established large model while investigating regressions.
    private static let usesFastLiveModel: Bool = {
#if targetEnvironment(macCatalyst)
        // The custom system keyboard is an iOS feature. Loading its separate tiny
        // latency model on macOS delayed the batch engine and consumed memory for a
        // workflow the platform cannot expose.
        false
#else
        ProcessInfo.processInfo.environment["FW_IOS_LIVE_FAST_MODEL"] != "0"
#endif
    }()

    private var generation = 0
    private var liveEngineGeneration = 0
    private var runTask: Task<Void, Never>?
    private let runActivity = WhisperActivityController.shared
    private var eta = WhisperAdaptiveETA()
    private(set) var runStarted: Date?
    /// Guards the async permission → start window: a second Record tap while
    /// the system prompt is up must not install a second tap (ObjC crash).
    private var micRequestInFlight = false

    /// Hidden physical-device lane for the exact tiny-model realtime route.
    /// The fixture is copied into this app's private Documents container by
    /// the profiling host; no benchmark asset or control enters the product UI.
    func runProfilingBenchmarkIfRequested() async {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FW_IOS_PROFILE"] == "1" else { return }
        liveEngineGeneration += 1
        let lifecycleToken = UInt64(liveEngineGeneration)

        let requestedRuns = Int(environment["FW_IOS_PROFILE_RUNS"] ?? "20") ?? 20
        let runs = max(1, min(100, requestedRuns))
        let fixtureName = URL(
            fileURLWithPath: environment["FW_IOS_PROFILE_AUDIO"] ?? "fw-ios-profile.wav"
        ).lastPathComponent
        let documents = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        )[0]
        let fixtureURL = documents.appendingPathComponent(fixtureName)
        let requestedModelName = environment["FW_IOS_PROFILE_MODEL"]
            .map { URL(fileURLWithPath: $0).lastPathComponent }
        let profileModelURL = requestedModelName.map {
            documents.appendingPathComponent($0)
        } ?? store.url(for: ModelManifest.tiny)
        let profileModelLabel = environment["FW_IOS_PROFILE_MODEL_LABEL"]
            ?? (requestedModelName == nil ? "tiny-multilingual-f16" : "custom")
        let profileModelSHA256 = environment["FW_IOS_PROFILE_MODEL_SHA256"]
            ?? (requestedModelName == nil ? ModelManifest.tiny.sha256 : "unreported")
        let profileModelSHA256Source = environment["FW_IOS_PROFILE_MODEL_SHA256"] == nil
            ? (requestedModelName == nil ? "bundled-manifest" : "unreported")
            : "host-provided"
        let profileDenoise = environment["FW_IOS_PROFILE_DENOISE"] == "1"
        let requestedDenoiserName = environment["FW_IOS_PROFILE_DENOISER"]
            .map { URL(fileURLWithPath: $0).lastPathComponent }
        let profileDenoiserURL = requestedDenoiserName.map {
            documents.appendingPathComponent($0)
        } ?? store.url(for: ModelManifest.denoiser)
        let profileDenoiserSHA256 = environment["FW_IOS_PROFILE_DENOISER_SHA256"]
            ?? (requestedDenoiserName == nil ? ModelManifest.denoiser.sha256 : "unreported")
        let profileDenoiserSHA256Source = environment["FW_IOS_PROFILE_DENOISER_SHA256"] == nil
            ? (requestedDenoiserName == nil ? "bundled-manifest" : "unreported")
            : "host-provided"
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let receiptURL = documents.appendingPathComponent(
            "fw-ios-profile-\(stamp).jsonl")
        var receiptLines: [String] = []
        let spans = WhisperProfileSpanCollector()

        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            UIApplication.shared.isIdleTimerDisabled = false
            EngineHooks.shared.set(span: nil, segments: nil)
        }

        func appendReceipt(_ object: [String: Any]) throws {
            let data = try JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
            guard let line = String(data: data, encoding: .utf8) else {
                throw EngineError.invalid("profiling receipt was not UTF-8")
            }
            receiptLines.append(line)
            try Data((receiptLines.joined(separator: "\n") + "\n").utf8)
                .write(to: receiptURL, options: .atomic)
            print("FW_IOS_PROFILE \(line)")
        }

        do {
            guard FileManager.default.fileExists(atPath: profileModelURL.path) else {
                throw EngineError.invalid(
                    "profiling model is missing: \(profileModelURL.lastPathComponent)")
            }
            if profileDenoise,
               !FileManager.default.fileExists(atPath: profileDenoiserURL.path)
            {
                throw EngineError.invalid(
                    "profiling denoiser is missing: \(profileDenoiserURL.lastPathComponent)")
            }
            let fixtureData = try Data(contentsOf: fixtureURL, options: .mappedIfSafe)
            let pcm = try Self.decodeProfilePCM16MonoWav(fixtureData)
            let profileModelAttributes = try FileManager.default.attributesOfItem(
                atPath: profileModelURL.path)
            let profileModelBytes = profileModelAttributes[.size] as? NSNumber
            let profileDenoiserBytes: Int64
            if profileDenoise {
                let attributes = try FileManager.default.attributesOfItem(
                    atPath: profileDenoiserURL.path)
                profileDenoiserBytes = (attributes[.size] as? NSNumber)?.int64Value ?? -1
            } else {
                profileDenoiserBytes = requestedDenoiserName == nil
                    ? ModelManifest.denoiser.bytes : -1
            }
            let fixtureDigest = SHA256.hash(data: fixtureData)
                .map { String(format: "%02x", $0) }.joined()
            let audioSeconds = Double(pcm.count) / 16_000.0

            try appendReceipt([
                "event": "run_start",
                "schema_version": 1,
                "runs": runs,
                "lane": "realtime_model_quality_ladder",
                "fixture": fixtureName,
                "fixture_sha256": fixtureDigest,
                "fixture_bytes": fixtureData.count,
                "audio_seconds": audioSeconds,
                "model": [
                    "label": profileModelLabel,
                    "filename": profileModelURL.lastPathComponent,
                    "bytes": profileModelBytes?.int64Value ?? -1,
                    "sha256": profileModelSHA256,
                    "sha256_source": profileModelSHA256Source,
                ] as [String: Any],
                "denoiser": [
                    "enabled": profileDenoise,
                    "filename": profileDenoiserURL.lastPathComponent,
                    "bytes": profileDenoiserBytes,
                    "sha256": profileDenoiserSHA256,
                    "sha256_source": profileDenoiserSHA256Source,
                ] as [String: Any],
                "rayon_threads": environment["RAYON_NUM_THREADS"] ?? "unset",
                "arm_dotprod": environment["FW_ARM_DOTPROD"] ?? "unset",
                "transcript_cache": environment["FW_TRANSCRIPT_CACHE"] ?? "unset",
                "performance_switches": [
                    "batch_gemv_cap": environment["FW_BATCH_GEMV_CAP"] ?? "unset",
                    "f16_batch_twopass": environment["FW_F16_BATCH_TWOPASS"] ?? "unset",
                    "f16_compute": environment["FRANKEN_WHISPER_NATIVE_F16_COMPUTE"] ?? "unset",
                    "mid_gemv_cap": environment["FW_MID_GEMV_CAP"] ?? "unset",
                    "enc_int8_attn_in": environment["FW_ENC_INT8_ATTN_IN"] ?? "unset",
                    "enc_attn_out_i8i32": environment["FW_ENC_ATTN_OUT_I8I32"] ?? "unset",
                    "enc_int8_fc1": environment["FW_ENC_INT8_FC1"] ?? "unset",
                    "enhance_accelerate": environment["FTTS_ENHANCE_ACCELERATE"] ?? "unset",
                    "enhance_gru_accelerate": environment["FTTS_ENHANCE_GRU_ACCELERATE"] ?? "unset",
                    "enhance_conv_accelerate": environment["FTTS_ENHANCE_CONV_ACCELERATE"] ?? "unset",
                    "enhance_concat_accelerate": environment["FTTS_ENHANCE_CONCAT_ACCELERATE"] ?? "unset",
                    "enhance_split_concat_accelerate": environment["FTTS_ENHANCE_SPLIT_CONCAT_ACCELERATE"] ?? "unset",
                ] as [String: Any],
                "device_model": UIDevice.current.model,
                "system_version": UIDevice.current.systemVersion,
                "active_processors": ProcessInfo.processInfo.activeProcessorCount,
                "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
                "receipt_path": receiptURL.path,
            ])

            EngineHooks.shared.set(
                span: { span, value in spans.record(span: span, value: value) },
                segments: nil
            )
            let loadStartedUptime = ProcessInfo.processInfo.systemUptime
            liveEngine.resetCancel()
            try await liveEngine.load(
                modelPath: profileModelURL,
                lifecycleToken: lifecycleToken
            )
            if profileDenoise {
                try await liveEngine.loadDenoiser(at: profileDenoiserURL)
            }
            try appendReceipt([
                "event": "engine_loaded",
                "load_ms": (ProcessInfo.processInfo.systemUptime - loadStartedUptime) * 1_000,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            ])

            let options = RunOptions(
                language: nil,
                initialPrompt: nil,
                translate: false,
                diarize: false,
                timestamps: false,
                wordTimestamps: false
            )
            var firstTranscriptDigest: String?
            var allTranscriptsIdentical = true
            for index in 0..<runs {
                spans.reset()
                liveEngine.resetCancel()
                let wallStartedUptime = ProcessInfo.processInfo.systemUptime
                let stageStartedUptime = ProcessInfo.processInfo.systemUptime
                let stage = try await liveEngine.stage(pcm: pcm, denoise: profileDenoise)
                let stageMs =
                    (ProcessInfo.processInfo.systemUptime - stageStartedUptime) * 1_000
                let runStartedUptime = ProcessInfo.processInfo.systemUptime
                let transcription = try await liveEngine.run(options: options)
                let runMs = (ProcessInfo.processInfo.systemUptime - runStartedUptime) * 1_000
                let wallMs = (ProcessInfo.processInfo.systemUptime - wallStartedUptime) * 1_000
                let transcriptData = Data(transcription.transcript.utf8)
                let transcriptDigest = SHA256.hash(data: transcriptData)
                    .map { String(format: "%02x", $0) }.joined()
                if firstTranscriptDigest == nil { firstTranscriptDigest = transcriptDigest }
                let matchesFirst = transcriptDigest == firstTranscriptDigest
                allTranscriptsIdentical = allTranscriptsIdentical && matchesFirst
                try appendReceipt([
                    "event": "sample",
                    "index": index,
                    "wall_ms": wallMs,
                    "stage_ms": stageMs,
                    "run_ms": runMs,
                    "audio_seconds": stage.audioSec,
                    "realtime_speed": stage.audioSec / max(wallMs / 1_000, 0.000_001),
                    "transcript": transcription.transcript,
                    "transcript_sha256": transcriptDigest,
                    "transcript_bytes": transcriptData.count,
                    "matches_first_transcript": matchesFirst,
                    "segments": transcription.segments.count,
                    "dropped_windows": transcription.droppedWindows,
                    "spans_ms": spans.snapshot(),
                    "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
                ])
            }
            try appendReceipt([
                "event": "run_complete",
                "completed_runs": runs,
                "all_transcripts_identical": allTranscriptsIdentical,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            ])
        } catch {
            try? appendReceipt([
                "event": "run_error",
                "message": error.localizedDescription,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
            ])
        }
    }

    private static func decodeProfilePCM16MonoWav(_ data: Data) throws -> [Float] {
        let bytes = [UInt8](data)
        guard bytes.count >= 44,
              String(decoding: bytes[0..<4], as: UTF8.self) == "RIFF",
              String(decoding: bytes[8..<12], as: UTF8.self) == "WAVE"
        else { throw EngineError.invalid("profiling fixture is not a RIFF/WAVE file") }

        func u16(_ offset: Int) -> UInt16 {
            UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
        }
        func u32(_ offset: Int) -> UInt32 {
            UInt32(bytes[offset])
                | (UInt32(bytes[offset + 1]) << 8)
                | (UInt32(bytes[offset + 2]) << 16)
                | (UInt32(bytes[offset + 3]) << 24)
        }

        var format: (audio: UInt16, channels: UInt16, rate: UInt32, bits: UInt16)?
        var sampleRange: Range<Int>?
        var offset = 12
        while offset + 8 <= bytes.count {
            let id = String(decoding: bytes[offset..<(offset + 4)], as: UTF8.self)
            let length = Int(u32(offset + 4))
            let start = offset + 8
            let end = start + length
            guard end <= bytes.count else {
                throw EngineError.invalid("profiling WAV contains a truncated chunk")
            }
            if id == "fmt ", length >= 16 {
                format = (u16(start), u16(start + 2), u32(start + 4), u16(start + 14))
            } else if id == "data" {
                sampleRange = start..<end
            }
            offset = end + (length & 1)
        }
        guard let format, format.audio == 1, format.channels == 1,
              format.rate == 16_000, format.bits == 16, let sampleRange
        else {
            throw EngineError.invalid("profiling WAV must be mono PCM16 at 16 kHz")
        }
        var pcm: [Float] = []
        pcm.reserveCapacity(sampleRange.count / 2)
        var cursor = sampleRange.lowerBound
        while cursor + 1 < sampleRange.upperBound {
            let raw = UInt16(bytes[cursor]) | (UInt16(bytes[cursor + 1]) << 8)
            pcm.append(Float(Int16(bitPattern: raw)) / 32_768.0)
            cursor += 2
        }
        return pcm
    }

    var isBusy: Bool {
        if runTask != nil { return true }
        if keyboardStartPending || micRequestInFlight { return true }
        if isLiveDictationActive { return true }
        if recorder.isRecording { return true }
        if isImporting { return true }
        if case .running = runState { return true }
        if case .staging = runState { return true }
        if case .loading = engineState { return true }
        if case .loading = liveEngineState { return true }
        return false
    }

    var isLiveDictationActive: Bool {
        switch liveDictationState {
        case .idle, .armed, .failed: false
        case .starting, .listening, .finishing: true
        }
    }

    /// Input pickers and external URL/drop/share handoffs all converge here.
    /// The external paths can bypass disabled SwiftUI controls, so the model
    /// itself must prevent a new source from replacing the media that an
    /// active run or microphone transaction is still consuming.
    var canAcceptInput: Bool {
        runTask == nil && !isImporting && !recorder.isRecording && !isLiveDictationActive
            && !keyboardStartPending && !micRequestInFlight
    }

    var hasArmedDictationSession: Bool {
        if case .armed = liveDictationState { return true }
        return recorder.isRecording && liveSessionExpiresAt != nil
    }

    var liveSessionMinutesRemaining: Int {
        guard let liveSessionExpiresAt else { return 0 }
        return max(0, Int(ceil(liveSessionExpiresAt.timeIntervalSinceNow / 60)))
    }

    // ── Engine assembly ────────────────────────────────────────────────────

    /// Hydrate the latency lane first. Once it is ready, hydrate the heavier
    /// batch lane in the background unless a keyboard handoff is waiting to
    /// start immediately. This makes a cold keyboard launch pay ~78 MB of
    /// model load instead of the full 874 MB + auxiliary models.
    func prepareEngines() {
        if Self.usesFastLiveModel {
            assembleLiveEngine()
        } else {
            assembleEngine()
        }
    }

    func assembleLiveEngine() {
        guard Self.usesFastLiveModel else {
            if engineState == .ready {
                liveEngineState = .ready
                beginKeyboardDictationIfReady()
            } else {
                assembleEngine()
            }
            return
        }
        // Resetting the native cancellation flag is process-wide. Never do it
        // from a background hydration while the batch engine owns that flag.
        guard runTask == nil, !recorder.isRecording, !micRequestInFlight,
              store.phase == .ready,
              liveEngineState == .notLoaded || isFailed(liveEngineState)
        else { return }
        if case .loading = engineState { return }

        liveEngineGeneration += 1
        let gen = liveEngineGeneration
        liveEngineState = .loading(stage: "waking the realtime model")
        let modelURL = store.url(for: ModelManifest.tiny)

        Task { [liveEngine] in
            do {
                liveEngine.resetCancel()
                try await liveEngine.load(
                    modelPath: modelURL,
                    lifecycleToken: UInt64(gen)
                )
                guard self.liveEngineGeneration == gen else { return }
                self.liveEngineState = .ready
                if self.keyboardStartPending {
                    self.beginKeyboardDictationIfReady()
                } else {
                    self.assembleEngine()
                }
            } catch {
                guard self.liveEngineGeneration == gen else { return }
                self.liveEngineState = .failed(error.localizedDescription)
                if self.keyboardStartPending {
                    self.failLiveDictation(
                        "The realtime speech engine could not start: \(error.localizedDescription)")
                } else {
                    // `prepareEngines` deliberately starts with the tiny lane.
                    // A corrupt/unloadable realtime model must not strand the
                    // independent full-quality transcription lane forever.
                    self.assembleEngine()
                }
            }
        }
    }

    /// Load whisper (+ Sortformer + denoiser) from the verified cache. The
    /// stage markers stream through EngineHooks into `engineState`.
    func assembleEngine() {
        guard runTask == nil, !recorder.isRecording, !isLiveDictationActive,
              liveDecodeTask == nil, store.phase == .ready,
              engineState == .notLoaded || isFailed(engineState)
        else {
            return
        }
        if case .loading = liveEngineState { return }
        enginePausedForMemoryPressure = false
        generation += 1
        let gen = generation
        engineState = .loading(stage: "waking the machine")
        installHooks(gen: gen, forLoad: true)
        let whisper = store.url(for: ModelManifest.whisper)
        let receipt = store.url(for: ModelManifest.sortformerReceipt)
        let package = store.url(for: ModelManifest.sortformerWeights)
        let denoiser = store.url(for: ModelManifest.denoiser)
        Task { [engine] in
            do {
                // Cancellation is process-wide and sticky; a run cancelled
                // earlier must not fail the Sortformer load's checkpoints.
                engine.resetCancel()
                try await engine.load(
                    modelPath: whisper,
                    lifecycleToken: UInt64(gen)
                )
                // The diarizer and denoiser are enhancements: a failure there
                // degrades (no speakers / no clean-up) instead of blocking
                // transcription. Their absence is visible in the UI.
                var diarizerOK = true
                var denoiserOK = true
                do { try await engine.loadSortformer(receipt: receipt, package: package) } catch {
                    diarizerOK = false
                }
                do { try await engine.loadDenoiser(at: denoiser) } catch { denoiserOK = false }
                guard self.generation == gen else { return }
                self.diarizerLoaded = diarizerOK
                self.denoiserLoaded = denoiserOK
                self.engineState = .ready
                if Self.usesFastLiveModel {
                    if self.liveEngineState == .ready {
                        self.beginKeyboardDictationIfReady()
                    } else {
                        self.assembleLiveEngine()
                    }
                } else {
                    self.liveEngineState = .ready
                    self.beginKeyboardDictationIfReady()
                }
            } catch {
                guard self.generation == gen else { return }
                self.engineState = .failed(error.localizedDescription)
                if Self.usesFastLiveModel {
                    // The independent tiny lane can still provide keyboard
                    // dictation even if the heavier batch model failed.
                    if self.liveEngineState == .ready {
                        self.beginKeyboardDictationIfReady()
                    } else {
                        self.assembleLiveEngine()
                    }
                } else if self.keyboardStartPending {
                    self.failLiveDictation(
                        "The local speech engine could not start: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Frees ~1.5 GB when the system is under pressure and no run is active.
    func unloadEngineForMemoryPressure() {
        guard !isBusy, !recorder.isRecording else { return }
        generation += 1
        liveEngineGeneration += 1
        let batchLifecycleToken = UInt64(generation)
        let liveLifecycleToken = UInt64(liveEngineGeneration)
        Task { [engine, liveEngine] in
            await engine.unload(lifecycleToken: batchLifecycleToken)
            await liveEngine.unload(lifecycleToken: liveLifecycleToken)
        }
        engineState = .notLoaded
        liveEngineState = .notLoaded
        diarizerLoaded = false
        denoiserLoaded = false
        enginePausedForMemoryPressure = true
    }

    /// Unload the resident engine before removing its verified model files.
    /// This keeps the Clear action available even though normal assembly is
    /// automatic now.
    func clearModels() {
        guard !isBusy else { return }
        generation += 1
        liveEngineGeneration += 1
        let gen = generation
        let liveGen = liveEngineGeneration
        engineState = .loading(stage: "clearing the machine")
        liveEngineState = .loading(stage: "clearing the realtime model")
        Task { [engine, liveEngine] in
            await engine.unload(lifecycleToken: UInt64(gen))
            await liveEngine.unload(lifecycleToken: UInt64(liveGen))
            guard self.generation == gen else { return }
            await self.store.clear()
            self.diarizerLoaded = false
            self.denoiserLoaded = false
            self.enginePausedForMemoryPressure = false
            self.engineState = .notLoaded
            self.liveEngineState = .notLoaded
            if self.keyboardStartPending {
                self.failLiveDictation(
                    "The local models were cleared. Download them again before using keyboard dictation.")
            }
        }
    }

    // ── Input handling ─────────────────────────────────────────────────────

    func toggleRecording() {
        guard !isLiveDictationActive else { return }
        if recorder.isRecording {
            let pcm = recorder.stop()
            if pcm.count >= 8_000 {  // half a second minimum
                let peak = pcm.reduce(Float(0)) { max($0, abs($1)) }
                guard peak > 0.005 else {
                    lastError = String(
                        format: "We could barely hear that recording (peak %.3f). Check the microphone level and try again.",
                        peak)
                    return
                }
                replaceInput(with: .recording(pcm: pcm))
                result = nil
                runState = .idle
            } else {
                lastError = "Recording too short — hold the button and speak."
            }
        } else {
            guard !isBusy else { return }
            micRequestInFlight = true
            result = nil
            // Ask for the microphone explicitly: a denied permission would
            // otherwise record silence and "transcribe" nothing, with no clue
            // why. The prompt only appears the first time.
            Task {
                defer { self.micRequestInFlight = false }
                guard await AVAudioApplication.requestRecordPermission() else {
                    self.lastError =
                        "Microphone access is off. Enable it in Settings › FrankenWhisper."
                    return
                }
                guard !self.recorder.isRecording else { return }
                do { try self.recorder.start() } catch {
                    self.lastError = "Microphone unavailable: \(error.localizedDescription)"
                }
            }
        }
    }

    // ── Live dictation ────────────────────────────────────────────────────

    /// Handles the public custom URL used by the system keyboard's mic key.
    /// The foreground transition is intentional: iOS requires the containing
    /// app to own microphone activation. Once the indicator says listening,
    /// the user swipes back and capture continues under the audio background
    /// mode while all inference stays on device.
    func handleKeyboardURL(_ url: URL) {
        guard url.scheme?.lowercased() == "frankenwhisper",
              url.host?.lowercased() == "dictate"
        else { return }

        keyboardHandoffVisible = true
        // Reopening the same URL while this service is already starting or
        // active is idempotent. Reinitializing the transaction here could
        // mark it failed while its original permission task still proceeds.
        if isLiveDictationActive { return }
        keyboardStartPending = true
        lastError = nil

        // The keyboard arrives through an external URL, so it can bypass all
        // of LabView's disabled controls. Never let it overlap the batch
        // engine (both lanes use EngineHooks.shared), an in-flight import, or
        // an ordinary recording. Most importantly, rejecting the handoff must
        // not stop and discard that unrelated recording.
        guard runTask == nil, !isImporting, !micRequestInFlight,
              !recorder.isRecording || hasArmedDictationSession
        else {
            rejectLiveDictation(
                "Finish the current recording, import, or transcription before starting keyboard dictation.")
            return
        }

        guard store.phase == .ready else {
            failLiveDictation(
                "Download and verify the local models in FrankenWhisper before using its keyboard.")
            return
        }

        // Both native handles share process-wide progress and cancellation
        // plumbing. Queue the latency lane behind an already-running batch
        // hydration instead of running the two engines concurrently.
        if case .loading = engineState {
            liveDictationState = .starting("Finishing local model setup…")
            publishLiveSnapshot(state: .idle, message: "Finishing local model setup…")
            return
        }

        if liveEngineState == .ready {
            beginKeyboardDictationIfReady()
        } else {
            assembleLiveEngine()
        }
    }

    func dismissKeyboardHandoff() {
        keyboardStartPending = false
        if case .starting = liveDictationState {
            liveStartTask?.cancel()
            // The task's defer owns these handles. Clearing them before the
            // cancelled permission/start transaction unwinds would admit a
            // second recorder start against the same AVAudioEngine tap.
            liveDictationState = .idle
            publishLiveSnapshot(state: .idle, message: nil)
        }
        keyboardHandoffVisible = false
    }

    func retryKeyboardDictation() {
        guard runTask == nil, !isImporting, !micRequestInFlight,
              !recorder.isRecording || hasArmedDictationSession
        else {
            rejectLiveDictation(
                "Finish the current recording, import, or transcription before starting keyboard dictation.")
            return
        }
        keyboardStartPending = true
        liveDictationState = .idle
        if case .loading = engineState {
            liveDictationState = .starting("Finishing local model setup…")
            publishLiveSnapshot(state: .idle, message: "Finishing local model setup…")
            return
        }
        if liveEngineState == .ready {
            beginKeyboardDictationIfReady()
        } else {
            assembleLiveEngine()
        }
    }

    private func beginKeyboardDictationIfReady() {
        guard keyboardStartPending, liveEngineState == .ready else { return }
        keyboardStartPending = false
        startLiveDictation()
    }

    /// Refuse a new live transaction without disturbing an unrelated batch
    /// run, file import, or ordinary microphone recording already in flight.
    private func rejectLiveDictation(_ message: String) {
        keyboardStartPending = false
        liveDictationState = .failed(message)
        publishLiveSnapshot(state: .failed, message: message)
    }

    private func failLiveDictation(_ message: String) {
        keyboardStartPending = false
        liveCaptureTask?.cancel()
        liveCaptureTask = nil
        if recorder.isRecording { _ = recorder.stop() }
        UIApplication.shared.isIdleTimerDisabled = false
        liveSessionExpiresAt = nil
        liveEndSessionRequested = false
        liveQueue.removeAll(keepingCapacity: true)
        activeLiveUtteranceID = nil
        activeLiveCommittedSamples = 0
        activeLiveLastScheduledSamples = 0
        livePartialText = ""
        livePolicyHoldingBack = false
        liveIsPreviewDecoding = false
        liveQueuedUtterances = 0
        liveSessionID = UUID().uuidString
        liveRevision = 0
        liveDictationState = .failed(message)
        publishLiveSnapshot(state: .failed, message: message)
        endLiveFinishBackgroundTask()
    }

    /// Start one explicit, user-visible microphone session and immediately
    /// begin its first utterance. The audio engine remains alive for one hour
    /// after Finish, allowing subsequent keyboard starts to remain in
    /// the caller's app. While armed, captured samples are discarded locally.
    func startLiveDictation() {
        guard runTask == nil, !isImporting, !micRequestInFlight,
              !recorder.isRecording || hasArmedDictationSession
        else {
            rejectLiveDictation(
                "Finish the current recording, import, or transcription before starting keyboard dictation.")
            return
        }

        // The default iPhone path owns a separate warm tiny engine. A
        // background large-model hydration must not delay an already-ready
        // realtime lane; only queue when Live still depends on that load.
        if case .loading = engineState,
           !Self.usesFastLiveModel || liveEngineState != .ready
        {
            keyboardStartPending = true
            liveDictationState = .starting("Finishing local model setup…")
            publishLiveSnapshot(state: .idle, message: "Finishing local model setup…")
            return
        }

        guard liveEngineState == .ready else {
            failLiveDictation(
                "The realtime speech engine is not ready yet. Please try again when it finishes loading.")
            return
        }

        // A capture watcher can outlive a failed/stopped recorder by one
        // scheduler turn. Clear that stale handle so a retry cannot silently
        // bounce off the start guard forever.
        if !recorder.isRecording, liveCaptureTask != nil {
            liveCaptureTask?.cancel()
            liveCaptureTask = nil
        }

        if recorder.isRecording, hasArmedDictationSession {
            beginArmedUtterance()
            return
        }

        guard !isLiveDictationActive, !recorder.isRecording else { return }
        guard liveDecodeTask == nil else {
            failLiveDictation("FrankenWhisper is still finishing the previous phrase. Try again in a moment.")
            return
        }
        guard liveStartTask == nil, !micRequestInFlight else { return }

        micRequestInFlight = true
        liveDictationState = .starting("Checking microphone access…")
        liveStartTask = Task {
            defer {
                self.micRequestInFlight = false
                self.liveStartTask = nil
            }

            switch AVAudioApplication.shared.recordPermission {
            case .granted:
                break
            case .denied:
                self.failLiveDictation(
                    "Microphone access is off. Enable it in Settings › Privacy & Security › Microphone.")
                return
            case .undetermined:
                self.liveDictationState = .starting("Allow microphone access to begin…")
                guard await AVAudioApplication.requestRecordPermission() else {
                    self.failLiveDictation(
                        "Microphone access is off. Enable it in Settings › Privacy & Security › Microphone.")
                    return
                }
            @unknown default:
                self.failLiveDictation("iOS could not determine microphone permission. Please try again.")
                return
            }

            guard !Task.isCancelled else { return }
            self.liveDictationState = .starting("Starting the microphone…")
            do {
                try self.recorder.start()
            } catch {
                self.failLiveDictation("Microphone unavailable: \(error.localizedDescription)")
                return
            }

            guard !Task.isCancelled else {
                _ = self.recorder.stop()
                return
            }

            self.liveServiceID = UUID().uuidString
            self.liveSessionExpiresAt = Date().addingTimeInterval(Self.liveSessionDuration)
            self.liveEndSessionRequested = false
            self.lastHandledDictationCommandID = DictationBridge.readCommand()?.id ?? ""
            EngineHooks.shared.set(span: nil, segments: nil)
            self.beginArmedUtterance()

            let serviceID = self.liveServiceID
            self.liveCaptureTask = Task { [weak self] in
                guard let self else { return }
                defer {
                    if self.liveServiceID == serviceID {
                        self.liveCaptureTask = nil
                    }
                }
                while !Task.isCancelled, self.recorder.isRecording {
                    try? await Task.sleep(for: .milliseconds(100))
                    guard !Task.isCancelled else { break }
                    if let command = DictationBridge.readCommand(),
                       command.id != self.lastHandledDictationCommandID
                    {
                        self.lastHandledDictationCommandID = command.id
                        switch command.action {
                        case .start:
                            self.beginArmedUtterance()
                        case .stop:
                            self.stopLiveDictation()
                        case .endSession:
                            self.endLiveDictationSession()
                        }
                    }
                    let fresh = self.recorder.takeAvailable()
                    if self.liveDictationState == .listening {
                        self.updateLiveMeter(fresh)
                        self.consumeLiveSamples(fresh)
                    }
                    if let expires = self.liveSessionExpiresAt, Date() >= expires {
                        self.liveEndSessionRequested = true
                        if self.liveDictationState == .listening {
                            self.stopLiveDictation()
                        } else if self.liveDecodeTask == nil, self.liveQueue.isEmpty {
                            self.finishLiveService()
                        }
                    }
                }
            }
        }
    }

    /// Begin a fresh insertion transaction while reusing the already-active
    /// microphone and tiny model. This is the normal keyboard Start path after
    /// the one-time foreground activation.
    private func beginArmedUtterance() {
        guard recorder.isRecording else { return }
        switch liveDictationState {
        case .armed, .starting:
            break
        case .idle, .listening, .finishing, .failed:
            return
        }
        liveDetector.reset()
        _ = recorder.takeAvailable() // discard standby audio captured before the tap
        liveQueue.removeAll(keepingCapacity: true)
        liveDictationText = ""
        liveLastPhrase = ""
        livePartialText = ""
        liveCommittedDeltaCount = 0
        livePolicyHoldingBack = false
        liveIsPreviewDecoding = false
        activeLiveUtteranceID = nil
        activeLiveCommittedSamples = 0
        activeLiveLastScheduledSamples = 0
        liveQueuedUtterances = 0
        liveSessionID = UUID().uuidString
        liveRevision = 0
        liveStopRequested = false
        liveMeterLevel = 0
        liveSpectrum = [Float](repeating: 0, count: LiveSpectrumAnalyzer.bandCount)
        lastLiveMeterPublishedAt = 0
        liveDictationState = .listening
        UIApplication.shared.isIdleTimerDisabled = true
        publishLiveSnapshot(state: .listening, message: nil)
    }

    /// Finish the current insertion transaction. The time-bounded microphone
    /// service stays armed, so the next keyboard Start does not leave the app.
    func stopLiveDictation() {
        guard liveDictationState == .listening else { return }
        liveStopRequested = true
        liveDictationState = .finishing
        liveMeterLevel = 0
        liveSpectrum = [Float](repeating: 0, count: LiveSpectrumAnalyzer.bandCount)
        publishLiveSnapshot(state: .finishing, message: nil)
        if liveFinishBackgroundTask == .invalid {
            liveFinishBackgroundTask = UIApplication.shared.beginBackgroundTask(
                withName: "Finish local dictation"
            ) { [weak self] in
                Task { @MainActor in self?.endLiveFinishBackgroundTask() }
            }
        }
        let tail = recorder.takeAvailable()
        consumeLiveSamples(tail)
        if let final = liveDetector.flush() {
            enqueueLiveUtterance(final)
        }
        finishLiveDictationIfReady()
    }

    func endLiveDictationSession() {
        liveEndSessionRequested = true
        if liveDictationState == .listening {
            stopLiveDictation()
        } else if liveDecodeTask == nil, liveQueue.isEmpty {
            finishLiveService()
        }
    }

    func clearLiveDictationText() {
        guard !isLiveDictationActive else { return }
        liveDictationText = ""
        liveLastPhrase = ""
        livePartialText = ""
        liveCommittedDeltaCount = 0
        livePolicyHoldingBack = false
        liveRevision += 1
        publishLiveSnapshot(
            state: hasArmedDictationSession ? .armed : .idle,
            message: armedSessionMessage)
    }

    private func consumeLiveSamples(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        let wasInSpeech = liveDetector.isInSpeech
        let completed = liveDetector.push(samples)
        if !wasInSpeech, (liveDetector.isInSpeech || !completed.isEmpty) {
            beginLiveUtteranceTracking()
        }
        for utterance in completed {
            enqueueLiveUtterance(utterance)
        }
        if liveDetector.isInSpeech, activeLiveUtteranceID == nil {
            beginLiveUtteranceTracking()
        }
        scheduleLivePreviewIfNeeded()
    }

    private func beginLiveUtteranceTracking() {
        liveUtteranceCounter += 1
        activeLiveUtteranceID = liveUtteranceCounter
        activeLiveCommittedSamples = 0
        activeLiveLastScheduledSamples = 0
        livePartialText = ""
        livePolicyHoldingBack = false
    }

    private func enqueueLiveUtterance(_ pcm: [Float]) {
        guard pcm.count >= 8_000 else { return }
        if activeLiveUtteranceID == nil {
            beginLiveUtteranceTracking()
        }
        liveQueue.append(
            PendingLiveUtterance(
                pcm: pcm,
                committedSamples: min(activeLiveCommittedSamples, pcm.count)))
        activeLiveUtteranceID = nil
        activeLiveCommittedSamples = 0
        activeLiveLastScheduledSamples = 0
        livePartialText = ""
        livePolicyHoldingBack = false
        liveQueuedUtterances = liveQueue.count + (liveDecodeTask == nil ? 0 : 1)
        processNextLiveUtterance()
    }

    private func scheduleLivePreviewIfNeeded() {
        guard liveDecodeTask == nil, liveQueue.isEmpty,
              let utteranceID = activeLiveUtteranceID,
              let fullPCM = liveDetector.activeUtterance
        else { return }
        let start = min(activeLiveCommittedSamples, fullPCM.count)
        let available = fullPCM.count - start
        let newlyCaptured = fullPCM.count - activeLiveLastScheduledSamples
        guard available >= 16_000, newlyCaptured >= 4_800 else { return }

        let pcm = Array(fullPCM[start...])
        activeLiveLastScheduledSamples = fullPCM.count
        liveIsPreviewDecoding = true
        let prompt = String(liveDictationText.suffix(240))
        let language = self.language == "auto" ? nil : self.language
        let options = LiveDecodeOptions(
            language: language,
            initialPrompt: prompt.isEmpty ? nil : prompt,
            endOfUtterance: false)
        let inferenceEngine = Self.usesFastLiveModel ? liveEngine : engine

        liveDecodeTask = Task { [inferenceEngine] in
            do {
                inferenceEngine.resetCancel()
                let decision = try await inferenceEngine.liveDecode(pcm: pcm, options: options)
                if self.activeLiveUtteranceID == utteranceID,
                   self.activeLiveCommittedSamples == start
                {
                    self.applyLiveDecision(decision, baseSample: start, sliceSamples: pcm.count)
                }
            } catch {
                if let engineError = error as? EngineError, engineError.isCancellation {
                    // App teardown may cancel a preview. No committed text is lost.
                } else if self.activeLiveUtteranceID == utteranceID {
                    // A preview is garnish. Keep recording and let the full-context
                    // endpoint decode decide the durable text.
                    self.livePartialText = ""
                    self.livePolicyHoldingBack = true
                }
            }
            self.liveDecodeTask = nil
            self.liveIsPreviewDecoding = false
            if !self.liveQueue.isEmpty {
                self.processNextLiveUtterance()
            } else {
                self.scheduleLivePreviewIfNeeded()
                self.finishLiveDictationIfReady()
            }
        }
    }

    private func processNextLiveUtterance() {
        guard liveDecodeTask == nil, !liveQueue.isEmpty else { return }
        let utterance = liveQueue.removeFirst()
        liveQueuedUtterances = liveQueue.count + 1
        let start = min(utterance.committedSamples, utterance.pcm.count)
        // AlignAtt never commits the final segment before endpoint close, but
        // keep the host robust if a future policy legitimately advances to
        // the exact PCM boundary. There is nothing left for the endpoint ABI
        // to decode, and that ABI intentionally rejects an empty slice.
        guard start < utterance.pcm.count else {
            liveQueuedUtterances = liveQueue.count
            if !liveQueue.isEmpty {
                processNextLiveUtterance()
            } else {
                finishLiveDictationIfReady()
            }
            return
        }
        let pcm = Array(utterance.pcm[start...])
        let language = self.language == "auto" ? nil : self.language
        let prompt = String(liveDictationText.suffix(240))
        let options = LiveDecodeOptions(
            language: language,
            initialPrompt: prompt.isEmpty ? nil : prompt,
            endOfUtterance: true)

        let inferenceEngine = Self.usesFastLiveModel ? liveEngine : engine
        liveDecodeTask = Task { [inferenceEngine] in
            do {
                inferenceEngine.resetCancel()
                let decision = try await inferenceEngine.liveDecode(pcm: pcm, options: options)
                self.applyLiveDecision(decision, baseSample: start, sliceSamples: pcm.count)
                self.livePartialText = ""
                self.livePolicyHoldingBack = false
            } catch {
                if let engineError = error as? EngineError, engineError.isCancellation {
                    // Cancellation is used only for app teardown; keep any text
                    // already committed and close the session honestly.
                } else {
                    self.liveStopRequested = true
                    self.liveQueue.removeAll(keepingCapacity: true)
                    self.failLiveDictation("Local dictation failed: \(error.localizedDescription)")
                }
            }
            self.liveDecodeTask = nil
            self.liveQueuedUtterances = self.liveQueue.count
            if !self.liveQueue.isEmpty {
                self.processNextLiveUtterance()
            } else {
                self.finishLiveDictationIfReady()
            }
        }
    }

    private func applyLiveDecision(
        _ decision: LiveDecodeResult,
        baseSample: Int,
        sliceSamples: Int
    ) {
        livePartialText = decision.partialTail?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        livePolicyHoldingBack = decision.holdback
        let commit = decision.commitText.trimmingCharacters(in: .whitespacesAndNewlines)
        if let through = decision.commitThroughSec, !decision.endOfUtterance {
            let advance = Int((through * 16_000).rounded())
            activeLiveCommittedSamples = min(baseSample + max(0, advance), baseSample + sliceSamples)
        }
        guard !commit.isEmpty else { return }
        let separator = liveDictationText.isEmpty ? "" : " "
        liveDictationText += separator + commit
        liveLastPhrase = commit
        liveCommittedDeltaCount += 1
        liveRevision += 1
        let state: DictationSnapshot.State = liveStopRequested ? .finishing : .listening
        publishLiveSnapshot(state: state, message: nil)
    }

    private func finishLiveDictationIfReady() {
        guard liveStopRequested, liveDecodeTask == nil, liveQueue.isEmpty else { return }
        UIApplication.shared.isIdleTimerDisabled = false
        liveQueuedUtterances = 0
        if case .failed = liveDictationState {
            endLiveFinishBackgroundTask()
            return
        }
        if liveEndSessionRequested || !recorder.isRecording {
            finishLiveService()
        } else {
            liveDictationState = .armed
            publishLiveSnapshot(state: .armed, message: armedSessionMessage)
        }
        endLiveFinishBackgroundTask()
    }

    private var armedSessionMessage: String? {
        guard hasArmedDictationSession else { return nil }
        return nil
    }

    private func updateLiveMeter(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        let frame = LiveSpectrumAnalyzer.analyze(samples)
        liveMeterLevel = liveMeterLevel * 0.45 + frame.level * 0.55
        if liveSpectrum.count != frame.bands.count {
            liveSpectrum = frame.bands
        } else {
            for index in frame.bands.indices {
                liveSpectrum[index] = liveSpectrum[index] * 0.35 + frame.bands[index] * 0.65
            }
        }

        // The keyboard polls at 4 Hz. Publishing faster only burns energy in
        // UserDefaults synchronization without making the animation smoother.
        let now = Date().timeIntervalSince1970
        guard now - lastLiveMeterPublishedAt >= 0.22 else { return }
        lastLiveMeterPublishedAt = now
        publishLiveSnapshot(state: .listening, message: nil)
    }

    private func finishLiveService() {
        liveCaptureTask?.cancel()
        liveCaptureTask = nil
        if recorder.isRecording { _ = recorder.stop() }
        liveServiceID = ""
        liveSessionExpiresAt = nil
        liveEndSessionRequested = false
        liveStopRequested = false
        activeLiveUtteranceID = nil
        activeLiveCommittedSamples = 0
        activeLiveLastScheduledSamples = 0
        livePartialText = ""
        livePolicyHoldingBack = false
        liveIsPreviewDecoding = false
        liveMeterLevel = 0
        liveSpectrum = [Float](repeating: 0, count: LiveSpectrumAnalyzer.bandCount)
        UIApplication.shared.isIdleTimerDisabled = false
        if case .failed = liveDictationState {
            endLiveFinishBackgroundTask()
            return
        }
        liveDictationState = .idle
        publishLiveSnapshot(state: .idle, message: nil)
        endLiveFinishBackgroundTask()
        if engineState == .notLoaded { assembleEngine() }
    }

    private func endLiveFinishBackgroundTask() {
        guard liveFinishBackgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(liveFinishBackgroundTask)
        liveFinishBackgroundTask = .invalid
    }

    private func publishLiveSnapshot(state: DictationSnapshot.State, message: String?) {
        publishLiveActivityState()
        DictationBridge.write(
            DictationSnapshot(
                sessionID: liveSessionID,
                state: state,
                text: liveDictationText,
                revision: liveRevision,
                message: message,
                level: state == .listening ? liveMeterLevel : 0,
                spectrum: state == .listening ? liveSpectrum : nil,
                expiresAt: liveSessionExpiresAt?.timeIntervalSince1970,
                updatedAt: Date().timeIntervalSince1970))
    }

    private func publishLiveActivityState() {
        // A rejected external keyboard handoff must not replace or terminate
        // the Live Activity for a batch transcription already in progress.
        guard runTask == nil else { return }
        runActivity.transitionLive(
            to: liveDictationState,
            sessionMinutesRemaining: liveSessionMinutesRemaining,
            queuedPhrases: liveQueuedUtterances,
            hasText: !liveDictationText.isEmpty
        )
    }

    @discardableResult
    func acceptFile(url: URL, removeSourceAfterImport: Bool = false) -> Bool {
        guard canAcceptInput else {
            if removeSourceAfterImport {
                try? FileManager.default.removeItem(at: url)
            }
            lastError = "Finish the current recording, import, or transcription before replacing its input."
            return false
        }
        let ext = url.pathExtension.lowercased()
        if let type = UTType(filenameExtension: ext), type.conforms(to: .movie) {
            acceptVideo(
                url: url,
                alreadyManaged: false,
                removeSourceAfterImport: removeSourceAfterImport
            )
            return true
        }
        let scoped = url.startAccessingSecurityScopedResource()
        isImporting = true
        Task {
            defer {
                if scoped { url.stopAccessingSecurityScopedResource() }
                if removeSourceAfterImport {
                    try? FileManager.default.removeItem(at: url)
                }
                self.isImporting = false
            }
            do {
                let data = try await Task.detached(priority: .userInitiated) {
                    try Data(contentsOf: url, options: .mappedIfSafe)
                }.value
                replaceInput(with: .file(
                    data: data, ext: url.pathExtension.lowercased(), name: url.lastPathComponent)
                )
                result = nil
                runState = .idle
            } catch {
                lastError = "Could not read \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
        return true
    }

    func acceptPickedVideo(_ picked: PickedVideo) {
        guard canAcceptInput else {
            // The transferable has already created an app-owned cache copy. A
            // picker completion racing another operation must not orphan it or
            // replace the source that operation is still consuming.
            try? FileManager.default.removeItem(at: picked.localURL)
            lastError =
                "Finish the current recording, import, or transcription before replacing its input."
            return
        }
        acceptVideo(url: picked.localURL, alreadyManaged: true, picked: picked)
    }

    private func acceptVideo(
        url: URL,
        alreadyManaged: Bool,
        picked: PickedVideo? = nil,
        removeSourceAfterImport: Bool = false
    ) {
        guard canAcceptInput else {
            if removeSourceAfterImport || alreadyManaged {
                try? FileManager.default.removeItem(at: url)
            }
            return
        }
        let scoped = alreadyManaged ? false : url.startAccessingSecurityScopedResource()
        isImporting = true
        Task {
            var preparedMedia: VideoInput?
            defer {
                if scoped { url.stopAccessingSecurityScopedResource() }
                if removeSourceAfterImport {
                    try? FileManager.default.removeItem(at: url)
                }
                self.isImporting = false
            }
            do {
                let media: VideoInput
                if let picked {
                    media = try await VideoImportService.preparePickedVideo(picked)
                } else {
                    media = try await VideoImportService.prepareExternalVideo(url)
                }
                preparedMedia = media
                let audioData = try await Task.detached(priority: .userInitiated) {
                    try Data(contentsOf: media.audioURL, options: .mappedIfSafe)
                }.value
                replaceInput(with: .video(media: media, audioData: audioData))
                result = nil
                runState = .idle
                // Karaoke export needs real decoder alignment, so video input
                // makes the word-timing requirement visible immediately.
                wordTimestamps = true
            } catch {
                if let preparedMedia {
                    VideoImportService.discard(preparedMedia)
                }
                lastError = "Could not prepare that video: \(error.localizedDescription)"
            }
        }
    }

    private func replaceInput(with newInput: Input) {
        if case .video(let oldMedia, _) = input {
            let replacingSameVideo: Bool
            if case .video(let newMedia, _) = newInput {
                replacingSameVideo = oldMedia.videoURL == newMedia.videoURL
            } else {
                replacingSameVideo = false
            }
            if !replacingSameVideo { VideoImportService.discard(oldMedia) }
        }
        input = newInput
    }

    func reportFileImportError(_ error: Error) {
        lastError = "Could not import that media file: \(error.localizedDescription)"
    }

    func reportVideoPickerError(_ error: Error) {
        lastError = "Could not open that video from Photos: \(error.localizedDescription)"
    }

    // ── The run ────────────────────────────────────────────────────────────

    func transcribe() {
        guard engineState == .ready, input != .none, !isBusy else { return }
        generation += 1
        let gen = generation
        liveSegments = []
        liveOffsetSec = 0
        result = nil
        resultWasTranslated = false
        lastError = nil
        wallSeconds = 0
        runAudioSeconds = input.seconds ?? 0
        estimatedFinishElapsed = nil
        let usesDiarization = diarize && diarizerLoaded
        eta.reset(includesDiarization: usesDiarization)
        runStarted = Date()
        runState = .staging
        UIApplication.shared.isIdleTimerDisabled = true
        installHooks(gen: gen, forLoad: false)

        let input = self.input
        // Names feed Whisper's decoding prompt (the CLI's --prompt) so names
        // and titles come out spelled right; the same list later maps onto
        // detected speakers in order of first appearance — exactly the
        // website's behavior.
        let names = Self.parseSpeakerNames(speakerNamesRaw)
        let translateToEnglish = self.translateToEnglish
        let options = RunOptions(
            language: language == "auto" ? nil : language,
            initialPrompt: names.isEmpty ? nil : "Speakers: \(names.joined(separator: ", ")).",
            translate: translateToEnglish,
            diarize: usesDiarization,
            wordTimestamps: wordTimestamps || input.isVideo)
        let denoise = self.denoise && denoiserLoaded

        runTask = Task { [engine] in
            defer {
                Task { @MainActor in
                    self.runTask = nil
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            }
            do {
                engine.resetCancel()
                let stage: StageInfo
                switch input {
                case .recording(let pcm):
                    stage = try await engine.stage(pcm: pcm, denoise: denoise)
                case .file(let data, let ext, _):
                    stage = try await engine.stage(fileData: data, ext: ext, denoise: denoise)
                case .video(_, let audioData):
                    stage = try await engine.stage(
                        fileData: audioData,
                        ext: "m4a",
                        denoise: denoise
                    )
                case .none:
                    // Unreachable (guarded at entry), but never strand the
                    // UI in .staging if it somehow happens.
                    self.runState = .idle
                    return
                }
                guard self.generation == gen else { return }
                self.liveOffsetSec = stage.skippedLeadingSec
                self.runAudioSeconds = stage.audioSec
                let windows = max(1, Int((stage.audioSec / 30.0).rounded(.up)))
                self.eta.beginDecodedAudio(
                    seconds: stage.audioSec,
                    windows: windows,
                    elapsed: self.elapsed(at: Date())
                )
                self.runState = .running(windowsDone: 0, windowsTotal: windows, stage: "decoding")
                let result = try await engine.run(options: options)
                guard self.generation == gen else { return }
                self.result = result
                self.resultWasTranslated = translateToEnglish
                self.speakerNameMap = Self.assignNames(names, to: result)
                self.wallSeconds = Date().timeIntervalSince(self.runStarted ?? Date())
                self.eta.finish(elapsed: self.wallSeconds)
                self.estimatedFinishElapsed = nil
                self.saveCurrentResultToHistory(result)
                self.runState = .done
            } catch {
                guard self.generation == gen else { return }
                if let engineError = error as? EngineError, engineError.isCancellation {
                    self.runState = .idle
                } else {
                    self.runState = .failed(error.localizedDescription)
                }
            }
        }
    }

    func cancelRun() {
        engine.requestCancel()
    }

    func elapsed(at date: Date) -> TimeInterval {
        max(0, date.timeIntervalSince(runStarted ?? date))
    }

    // ── Speaker naming ─────────────────────────────────────────────────────

    /// Comma- or newline-separated, trimmed, empties dropped (site parity).
    static func parseSpeakerNames(_ raw: String) -> [String] {
        raw.split(whereSeparator: { $0 == "," || $0.isNewline })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// Assign names to detected `SPEAKER_NN` lanes in order of first
    /// appearance in the merged transcript; lanes beyond the provided names
    /// keep their raw label (site parity).
    static func assignNames(_ names: [String], to result: Transcription) -> [String: String] {
        var map: [String: String] = [:]
        guard !names.isEmpty else { return map }
        for run in result.speakerSegments {
            guard let lane = run.speaker, map[lane] == nil else { continue }
            if map.count < names.count {
                map[lane] = names[map.count]
            }
        }
        return map
    }

    /// The distinct lanes of the current result, in first-appearance order —
    /// the rows of the transcript card's rename editor.
    var detectedSpeakers: [String] {
        guard let result else { return [] }
        var seen: [String] = []
        for run in result.speakerSegments {
            if let lane = run.speaker, !seen.contains(lane) {
                seen.append(lane)
            }
        }
        return seen
    }

    func displaySpeaker(_ lane: String?) -> String {
        guard let lane else { return "UNKNOWN" }
        let name = speakerNameMap[lane]?.trimmingCharacters(in: .whitespaces) ?? ""
        return name.isEmpty ? lane : name
    }

    /// Stores the RAW text: trimming here would fight the keyboard (the
    /// binding round-trips per keystroke, so a trailing space typed
    /// mid-"John Smith" would vanish instantly). Display and export trim.
    func setSpeakerName(_ name: String, for lane: String) {
        if name.isEmpty {
            speakerNameMap.removeValue(forKey: lane)
        } else {
            speakerNameMap[lane] = name
        }
    }

    /// Everything an export needs beyond the result itself. Names are
    /// trimmed here so in-progress whitespace never reaches a document.
    var exportContext: ExportContext {
        var trimmed: [String: String] = [:]
        for (lane, name) in speakerNameMap {
            let clean = name.trimmingCharacters(in: .whitespaces)
            if !clean.isEmpty { trimmed[lane] = clean }
        }
        return ExportContext(
            sourceName: inputName.isEmpty ? "recording" : inputName,
            wallSeconds: wallSeconds,
            names: trimmed,
            translatedToEnglish: resultWasTranslated)
    }

    private func saveCurrentResultToHistory(_ result: Transcription) {
        guard !result.segments.isEmpty else { return }
        let context = exportContext
        _ = try? history.record(
            TranscriptHistoryResult(
                markdown: TranscriptExport.markdown(from: result, context: context),
                sourceName: context.sourceName,
                language: result.language ?? (context.translatedToEnglish ? "en" : "unknown"),
                audioSeconds: result.audioSec,
                processingSeconds: context.wallSeconds,
                translatedToEnglish: context.translatedToEnglish
            )
        )
    }

    // ── Hook plumbing ──────────────────────────────────────────────────────

    /// Route the engine's heartbeat into observable state. Handlers run on
    /// the decode thread; they hop to the main actor and drop stale updates.
    private func installHooks(gen: Int, forLoad: Bool) {
        EngineHooks.shared.set(
            span: { [weak self] span, value in
                Task { @MainActor in
                    guard let self, self.generation == gen else { return }
                    if forLoad {
                        if span.hasPrefix("whisper:") || span.hasPrefix("sortformer:") {
                            self.engineState = .loading(stage: Self.humanStage(span, value))
                        }
                        return
                    }
                    if span == "encoder_window",
                        case .running(let done, let total, _) = self.runState
                    {
                        self.runState = .running(
                            windowsDone: min(done + 1, total), windowsTotal: total,
                            stage: "decoding")
                    } else if span.hasPrefix("audio:denoise"), case .staging = self.runState {
                        // value is the 0...1 completed fraction.
                    } else if span == "sortformer:diarize" || span == "sortformer_tick",
                        case .running(let done, let total, _) = self.runState
                    {
                        self.runState = .running(
                            windowsDone: done, windowsTotal: total, stage: "labeling speakers")
                    } else if span == "fuse:project",
                        case .running(let done, let total, _) = self.runState
                    {
                        self.runState = .running(
                            windowsDone: done, windowsTotal: total, stage: "fusing")
                    }
                }
            },
            segments: { [weak self] json in
                // One JSON array of segments per finished decode window.
                guard let data = json.data(using: .utf8),
                    let batch = try? Engine.decoder().decode([TranscriptSegment].self, from: data)
                else { return }
                Task { @MainActor in
                    guard let self, self.generation == gen else { return }
                    self.liveSegments.append(contentsOf: batch)
                    self.runActivity.transition(
                        to: self.runState,
                        elapsed: self.elapsed(at: Date()),
                        emittedSegments: self.liveSegments.count
                    )
                }
            })
    }

    private static func humanStage(_ span: String, _ value: Double) -> String {
        switch span {
        case "whisper:scan": "reading the model file"
        case "whisper:weights": "hydrating whisper weights"
        case "whisper:ready": "whisper ready"
        case "sortformer:verify": "verifying the diarizer package"
        case "sortformer:weights": "hydrating the diarizer"
        case "sortformer:ready": "diarizer ready"
        default: span
        }
    }

    private func isFailed(_ state: EngineState) -> Bool {
        if case .failed = state { return true }
        return false
    }
}
