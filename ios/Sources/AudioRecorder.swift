// Microphone capture: AVAudioEngine tap → mono Float32 at 16 kHz, the exact
// PCM whisper consumes. The recording lives only in memory and is handed to
// the engine, matching the app's microphone permission copy.
//
// Threading: the AVAudioEngine tap fires on a realtime audio thread, which
// must never block on the main thread. Conversion and accumulation therefore
// happen inside a small lock-protected sink the tap owns; the UI observes
// only the main-actor fields. (Same shape as the FrankenTTS enrollment
// recorder, retargeted from 24 kHz to 16 kHz.)

import AVFoundation
import Foundation

/// Owned by the audio tap: converts to 16 kHz mono and accumulates under a lock.
private final class CaptureSink: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    private var recentPeak: Float = 0
    let converter: AVAudioConverter
    let format: AVAudioFormat

    init?(from input: AVAudioFormat) {
        guard
            let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: AudioRecorder.targetRate,
                channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: input, to: target)
        else { return nil }
        self.converter = converter
        self.format = target
    }

    func consume(_ buffer: AVAudioPCMBuffer) {
        let ratio = format.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
            return
        }
        var served = false
        _ = converter.convert(to: out, error: nil) { _, status in
            if served {
                status.pointee = .noDataNow
                return nil
            }
            served = true
            status.pointee = .haveData
            return buffer
        }
        guard let channel = out.floatChannelData?[0], out.frameLength > 0 else { return }
        let chunk = UnsafeBufferPointer(start: channel, count: Int(out.frameLength))
        var peak: Float = 0
        for value in chunk { peak = max(peak, abs(value)) }
        lock.lock()
        samples.append(contentsOf: chunk)
        recentPeak = peak
        lock.unlock()
    }

    /// Most recent chunk's peak, for the live level meter.
    func levelPeek() -> Float {
        lock.lock()
        defer { lock.unlock() }
        return recentPeak
    }

    func drain() -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let out = samples
        samples = []
        return out
    }

    /// Take the samples captured since the last call while leaving the audio
    /// engine running. Live dictation polls this; ordinary recording never
    /// calls it and therefore still receives the entire capture from `stop()`.
    func drainAvailable() -> [Float] {
        drain()
    }
}

@MainActor
@Observable
final class AudioRecorder {
    /// nonisolated: the capture sink reads this off the main actor (it is an
    /// immutable constant, so isolation buys nothing but a Swift 6 error).
    nonisolated static let targetRate: Double = 16_000

    var isRecording = false
    var seconds: Double = 0
    /// 0...1 input level for the meter, updated a few times a second.
    var level: Float = 0

    private let audioEngine = AVAudioEngine()
    private var sink: CaptureSink?
    private var timer: Timer?

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        // .default keeps the system input chain, including automatic gain
        // control. .measurement disables it and hands over raw low-gain
        // samples — near-silent transcription input on a phone held at arm's
        // length. Lesson inherited from FrankenTTS, learned on a real device.
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)

        let input = audioEngine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard let sink = CaptureSink(from: inputFormat) else {
            throw EngineError.invalid("cannot build the 16 kHz capture pipeline")
        }
        self.sink = sink
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            sink.consume(buffer)
        }
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            // Never leave the tap installed on failure: a second installTap
            // on the same bus raises an ObjC exception (a crash, not a throw).
            input.removeTap(onBus: 0)
            self.sink = nil
            try? session.setActive(false, options: [.notifyOthersOnDeactivation])
            throw error
        }
        isRecording = true
        seconds = 0
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.seconds += 0.25
                self.level = min(1, (self.sink?.levelPeek() ?? 0) * 1.6)
            }
        }
    }

    /// Stops and returns the captured 16 kHz mono PCM.
    func stop() -> [Float] {
        timer?.invalidate()
        timer = nil
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        isRecording = false
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        let samples = sink?.drain() ?? []
        sink = nil
        level = 0
        return samples
    }

    /// Drain newly captured PCM without stopping the realtime tap.
    func takeAvailable() -> [Float] {
        sink?.drainAvailable() ?? []
    }
}
