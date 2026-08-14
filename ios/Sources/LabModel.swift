// All app state and orchestration: model download → engine assembly → audio
// input (mic or file) → the run (with live windows) → results and export.
//
// The engine's long calls are blocking actor methods; every one runs inside
// a Task so the UI stays responsive, and stale publishes are guarded by a
// generation counter (a cancelled run must never overwrite a newer one).

import AVFoundation
import Foundation
import SwiftUI

@MainActor
@Observable
final class LabModel {
    // ── Collaborators ──────────────────────────────────────────────────────
    let store = ModelStore()
    let recorder = AudioRecorder()
    private let engine = Engine()

    // ── Engine state ───────────────────────────────────────────────────────
    enum EngineState: Equatable {
        case notLoaded
        case loading(stage: String)
        case ready
        case failed(String)
    }
    var engineState: EngineState = .notLoaded
    var diarizerLoaded = false
    var denoiserLoaded = false

    // ── Input ──────────────────────────────────────────────────────────────
    enum Input: Equatable {
        case none
        case recording(pcm: [Float])
        case file(data: Data, ext: String, name: String)

        var seconds: Double? {
            switch self {
            case .none: nil
            case .recording(let pcm): Double(pcm.count) / 16_000.0
            case .file: nil  // unknown until staged
            }
        }
    }
    var input: Input = .none
    var isImporting = false
    var inputName: String {
        switch input {
        case .none: ""
        case .recording: "microphone capture"
        case .file(_, _, let name): name
        }
    }

    // ── Options ────────────────────────────────────────────────────────────
    var diarize = true
    var denoise = true
    var wordTimestamps = false
    var language = "auto"
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
    var runState: RunState = .idle
    var liveSegments: [TranscriptSegment] = []
    /// Live segments arrive in the trimmed timebase (fw_ios.h); add this to
    /// their times for display. The final result already has it added back.
    var liveOffsetSec: Double = 0
    var result: Transcription?
    var wallSeconds: Double = 0
    var lastError: String?

    private var generation = 0
    private var runTask: Task<Void, Never>?
    private(set) var runStarted: Date?
    /// Guards the async permission → start window: a second Record tap while
    /// the system prompt is up must not install a second tap (ObjC crash).
    private var micRequestInFlight = false

    var isBusy: Bool {
        if isImporting { return true }
        if case .running = runState { return true }
        if case .staging = runState { return true }
        if case .loading = engineState { return true }
        return false
    }

    // ── Engine assembly ────────────────────────────────────────────────────

    /// Load whisper (+ Sortformer + denoiser) from the verified cache. The
    /// stage markers stream through EngineHooks into `engineState`.
    func assembleEngine() {
        guard store.phase == .ready, engineState == .notLoaded || isFailed(engineState) else {
            return
        }
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
                try await engine.load(modelPath: whisper)
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
            } catch {
                guard self.generation == gen else { return }
                self.engineState = .failed(error.localizedDescription)
            }
        }
    }

    /// Frees ~1.5 GB when the system is under pressure and no run is active.
    func unloadEngineForMemoryPressure() {
        guard !isBusy else { return }
        generation += 1
        Task { [engine] in await engine.unload() }
        engineState = .notLoaded
        diarizerLoaded = false
        denoiserLoaded = false
    }

    // ── Input handling ─────────────────────────────────────────────────────

    func toggleRecording() {
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
                input = .recording(pcm: pcm)
                result = nil
                runState = .idle
            } else {
                lastError = "Recording too short — hold the button and speak."
            }
        } else {
            guard !micRequestInFlight else { return }
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

    func acceptFile(url: URL) {
        guard !isImporting else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        isImporting = true
        Task {
            defer {
                if scoped { url.stopAccessingSecurityScopedResource() }
                self.isImporting = false
            }
            do {
                let data = try await Task.detached(priority: .userInitiated) {
                    try Data(contentsOf: url, options: .mappedIfSafe)
                }.value
                input = .file(
                    data: data, ext: url.pathExtension.lowercased(), name: url.lastPathComponent)
                result = nil
                runState = .idle
            } catch {
                lastError = "Could not read \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    func reportFileImportError(_ error: Error) {
        lastError = "Could not import that audio file: \(error.localizedDescription)"
    }

    // ── The run ────────────────────────────────────────────────────────────

    func transcribe() {
        guard engineState == .ready, input != .none, runTask == nil else { return }
        generation += 1
        let gen = generation
        runState = .staging
        liveSegments = []
        liveOffsetSec = 0
        result = nil
        lastError = nil
        runStarted = Date()
        UIApplication.shared.isIdleTimerDisabled = true
        installHooks(gen: gen, forLoad: false)

        let input = self.input
        let options = RunOptions(
            language: language == "auto" ? nil : language,
            initialPrompt: nil,
            translate: false,
            diarize: diarize && diarizerLoaded,
            wordTimestamps: wordTimestamps)
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
                case .none:
                    // Unreachable (guarded at entry), but never strand the
                    // UI in .staging if it somehow happens.
                    self.runState = .idle
                    return
                }
                guard self.generation == gen else { return }
                self.liveOffsetSec = stage.skippedLeadingSec
                let windows = max(1, Int((stage.audioSec / 30.0).rounded(.up)))
                self.runState = .running(windowsDone: 0, windowsTotal: windows, stage: "decoding")
                let result = try await engine.run(options: options)
                guard self.generation == gen else { return }
                self.result = result
                self.wallSeconds = Date().timeIntervalSince(self.runStarted ?? Date())
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
