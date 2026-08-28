// All app state and orchestration: model download → engine assembly → audio
// input (mic or file) → the run (with live windows) → results and export.
//
// The engine's long calls are blocking actor methods; every one runs inside
// a Task so the UI stays responsive, and stale publishes are guarded by a
// generation counter (a cancelled run must never overwrite a newer one).

import AVFoundation
import Foundation
import SwiftUI
import UIKit

@MainActor
@Observable
final class LabModel {
    // ── Collaborators ──────────────────────────────────────────────────────
    let store = ModelStore()
    let recorder = AudioRecorder()
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
    var runState: RunState = .idle
    /// Raw `SPEAKER_NN` lane → the user's display name, seeded from the
    /// pre-run field when the result lands and editable afterwards through
    /// the transcript card's rename rows.
    var speakerNameMap: [String: String] = [:]
    var liveSegments: [TranscriptSegment] = []
    /// Live segments arrive in the trimmed timebase (fw_ios.h); add this to
    /// their times for display. The final result already has it added back.
    var liveOffsetSec: Double = 0
    var result: Transcription?
    var wallSeconds: Double = 0
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
    var liveDictationState: LiveDictationState = .idle
    var liveDictationText = ""
    var liveLastPhrase = ""
    var liveQueuedUtterances = 0
    private var liveDetector = LiveUtteranceDetector()
    private var liveCaptureTask: Task<Void, Never>?
    private var liveDecodeTask: Task<Void, Never>?
    private var liveStartTask: Task<Void, Never>?
    private var liveQueue: [[Float]] = []
    private var liveSessionID = ""
    private var liveServiceID = ""
    private var liveRevision = 0
    private var liveStopRequested = false
    private var liveEndSessionRequested = false
    private var liveSessionExpiresAt: Date?
    var keyboardHandoffVisible = false
    private var keyboardStartPending = false
    private var lastHandledDictationCommandID = ""
    private var liveFinishBackgroundTask: UIBackgroundTaskIdentifier = .invalid

    /// One explicit activation buys a useful native-feeling window. Keeping a
    /// microphone alive indefinitely would waste battery and surprise users.
    private static let liveSessionDuration: TimeInterval = 15 * 60

    /// Mega-kernel discipline: the product split has an emergency kill switch.
    /// Set FW_IOS_LIVE_FAST_MODEL=0 before first engine use to route live work
    /// through the established large model while investigating regressions.
    private static let usesFastLiveModel =
        ProcessInfo.processInfo.environment["FW_IOS_LIVE_FAST_MODEL"] != "0"

    private var generation = 0
    private var liveEngineGeneration = 0
    private var runTask: Task<Void, Never>?
    private(set) var runStarted: Date?
    /// Guards the async permission → start window: a second Record tap while
    /// the system prompt is up must not install a second tap (ObjC crash).
    private var micRequestInFlight = false

    var isBusy: Bool {
        if isLiveDictationActive { return true }
        if recorder.isRecording { return true }
        if isImporting { return true }
        if case .running = runState { return true }
        if case .staging = runState { return true }
        if case .loading = engineState { return true }
        return false
    }

    var isLiveDictationActive: Bool {
        switch liveDictationState {
        case .idle, .armed, .failed: false
        case .starting, .listening, .finishing: true
        }
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
        guard store.phase == .ready,
              liveEngineState == .notLoaded || isFailed(liveEngineState)
        else { return }

        liveEngineGeneration += 1
        let gen = liveEngineGeneration
        liveEngineState = .loading(stage: "waking the realtime model")
        let modelURL = store.url(for: ModelManifest.tiny)

        Task { [liveEngine] in
            do {
                liveEngine.resetCancel()
                try await liveEngine.load(modelPath: modelURL)
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
                }
            }
        }
    }

    /// Load whisper (+ Sortformer + denoiser) from the verified cache. The
    /// stage markers stream through EngineHooks into `engineState`.
    func assembleEngine() {
        guard store.phase == .ready, engineState == .notLoaded || isFailed(engineState) else {
            return
        }
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
                if !Self.usesFastLiveModel {
                    self.liveEngineState = .ready
                    self.beginKeyboardDictationIfReady()
                }
            } catch {
                guard self.generation == gen else { return }
                self.engineState = .failed(error.localizedDescription)
                if self.keyboardStartPending {
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
        Task { [engine, liveEngine] in
            await engine.unload()
            await liveEngine.unload()
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
        engineState = .loading(stage: "clearing the machine")
        liveEngineState = .loading(stage: "clearing the realtime model")
        Task { [engine, liveEngine] in
            await engine.unload()
            await liveEngine.unload()
            guard self.generation == gen else { return }
            self.store.clear()
            self.diarizerLoaded = false
            self.denoiserLoaded = false
            self.enginePausedForMemoryPressure = false
            self.engineState = .notLoaded
            self.liveEngineState = .notLoaded
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
        keyboardStartPending = true
        lastError = nil

        guard store.phase == .ready else {
            failLiveDictation(
                "Download and verify the local models in FrankenWhisper before using its keyboard.")
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
            liveStartTask = nil
            micRequestInFlight = false
            liveDictationState = .idle
            publishLiveSnapshot(state: .idle, message: nil)
        }
        keyboardHandoffVisible = false
    }

    func retryKeyboardDictation() {
        keyboardStartPending = true
        liveDictationState = .idle
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

    private func failLiveDictation(_ message: String) {
        keyboardStartPending = false
        liveCaptureTask?.cancel()
        liveCaptureTask = nil
        if recorder.isRecording { _ = recorder.stop() }
        UIApplication.shared.isIdleTimerDisabled = false
        liveSessionExpiresAt = nil
        liveEndSessionRequested = false
        liveSessionID = UUID().uuidString
        liveRevision = 0
        liveDictationState = .failed(message)
        publishLiveSnapshot(state: .failed, message: message)
    }

    /// Start one explicit, user-visible microphone session and immediately
    /// begin its first utterance. The audio engine remains alive for fifteen
    /// minutes after Finish, allowing subsequent keyboard starts to remain in
    /// the caller's app. While armed, captured samples are discarded locally.
    func startLiveDictation() {
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
                        for utterance in self.liveDetector.push(fresh) {
                            self.enqueueLiveUtterance(utterance)
                        }
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
        liveQueuedUtterances = 0
        liveSessionID = UUID().uuidString
        liveRevision = 0
        liveStopRequested = false
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
        publishLiveSnapshot(state: .finishing, message: nil)
        if liveFinishBackgroundTask == .invalid {
            liveFinishBackgroundTask = UIApplication.shared.beginBackgroundTask(
                withName: "Finish local dictation"
            ) { [weak self] in
                Task { @MainActor in self?.endLiveFinishBackgroundTask() }
            }
        }
        let tail = recorder.takeAvailable()
        for utterance in liveDetector.push(tail) {
            enqueueLiveUtterance(utterance)
        }
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
        liveRevision += 1
        publishLiveSnapshot(
            state: hasArmedDictationSession ? .armed : .idle,
            message: armedSessionMessage)
    }

    private func enqueueLiveUtterance(_ pcm: [Float]) {
        guard pcm.count >= 8_000 else { return }
        liveQueue.append(pcm)
        liveQueuedUtterances = liveQueue.count + (liveDecodeTask == nil ? 0 : 1)
        processNextLiveUtterance()
    }

    private func processNextLiveUtterance() {
        guard liveDecodeTask == nil, !liveQueue.isEmpty else { return }
        let pcm = liveQueue.removeFirst()
        liveQueuedUtterances = liveQueue.count + 1
        let language = self.language == "auto" ? nil : self.language
        let prompt = String(liveDictationText.suffix(240))
        let options = RunOptions(
            language: language,
            initialPrompt: prompt.isEmpty ? nil : prompt,
            translate: false,
            diarize: false,
            timestamps: false,
            wordTimestamps: false)

        let inferenceEngine = Self.usesFastLiveModel ? liveEngine : engine
        liveDecodeTask = Task { [inferenceEngine] in
            do {
                inferenceEngine.resetCancel()
                _ = try await inferenceEngine.stage(pcm: pcm, denoise: false)
                let transcription = try await inferenceEngine.run(options: options)
                let phrase = transcription.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
                if !phrase.isEmpty {
                    self.appendLivePhrase(phrase)
                }
            } catch {
                if let engineError = error as? EngineError, engineError.isCancellation {
                    // Cancellation is used only for app teardown; keep any text
                    // already committed and close the session honestly.
                } else {
                    self.liveStopRequested = true
                    self.liveQueue.removeAll(keepingCapacity: true)
                    self.failLiveDictation("Local dictation failed: \(error.localizedDescription)")
                    self.endLiveFinishBackgroundTask()
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

    private func appendLivePhrase(_ phrase: String) {
        let separator = liveDictationText.isEmpty ? "" : " "
        liveDictationText += separator + phrase
        liveLastPhrase = phrase
        liveRevision += 1
        let state: DictationSnapshot.State = liveStopRequested ? .finishing : .listening
        publishLiveSnapshot(state: state, message: nil)
    }

    private func finishLiveDictationIfReady() {
        guard liveStopRequested, liveDecodeTask == nil, liveQueue.isEmpty else { return }
        UIApplication.shared.isIdleTimerDisabled = false
        liveQueuedUtterances = 0
        if case .failed = liveDictationState {
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
        return "Ready for another dictation · about \(liveSessionMinutesRemaining) min left"
    }

    private func finishLiveService() {
        liveCaptureTask?.cancel()
        liveCaptureTask = nil
        if recorder.isRecording { _ = recorder.stop() }
        liveServiceID = ""
        liveSessionExpiresAt = nil
        liveEndSessionRequested = false
        liveStopRequested = false
        UIApplication.shared.isIdleTimerDisabled = false
        if case .failed = liveDictationState { return }
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
        DictationBridge.write(
            DictationSnapshot(
                sessionID: liveSessionID,
                state: state,
                text: liveDictationText,
                revision: liveRevision,
                message: message,
                updatedAt: Date().timeIntervalSince1970))
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
        guard engineState == .ready, input != .none, runTask == nil,
              !isLiveDictationActive else { return }
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
        // Names feed Whisper's decoding prompt (the CLI's --prompt) so names
        // and titles come out spelled right; the same list later maps onto
        // detected speakers in order of first appearance — exactly the
        // website's behavior.
        let names = Self.parseSpeakerNames(speakerNamesRaw)
        let options = RunOptions(
            language: language == "auto" ? nil : language,
            initialPrompt: names.isEmpty ? nil : "Speakers: \(names.joined(separator: ", ")).",
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
                self.speakerNameMap = Self.assignNames(names, to: result)
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
            names: trimmed)
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
