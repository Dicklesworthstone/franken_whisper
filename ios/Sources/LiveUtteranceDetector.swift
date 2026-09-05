import Foundation

/// A small causal endpoint detector for the iOS host-pushed audio lane.
/// It does no recognition: it only turns continuous 16 kHz PCM into bounded
/// utterances that the same Rust engine used by batch transcription decodes.
/// Apple-specific capture stays host-owned; Rust now shares the consequential
/// AlignAtt commit/preview policy with the CLI live driver.
struct LiveUtteranceDetector {
    private static let sampleRate = 16_000
    private static let frameSamples = 320       // 20 ms
    private static let preRollSamples = 4_800   // 300 ms
    private static let minSpeechFrames = 13     // 260 ms
    private static let endpointFrames = 35      // 700 ms
    private static let maxUtteranceSamples = 28 * sampleRate

    private var pending: [Float] = []
    private var preRoll: [Float] = []
    private var utterance: [Float] = []
    private var consecutiveSpeech = 0
    private var consecutiveSilence = 0
    private var noiseRMS: Float = 0.004
    private(set) var isInSpeech = false

    /// Snapshot the active utterance without changing VAD state. This is the
    /// host-owned rolling audio that Rust's true-live AlignAtt step decodes;
    /// callers must treat it as immutable and advance their own committed
    /// sample cursor only from `commit_through_sec`.
    var activeUtterance: [Float]? {
        guard isInSpeech else { return nil }
        if pending.isEmpty { return utterance }
        return utterance + pending
    }

    mutating func reset() {
        self = LiveUtteranceDetector()
    }

    mutating func push(_ samples: [Float]) -> [[Float]] {
        guard !samples.isEmpty else { return [] }
        pending.append(contentsOf: samples)
        var completed: [[Float]] = []
        var consumedSamples = 0

        while pending.count - consumedSamples >= Self.frameSamples {
            let frameEnd = consumedSamples + Self.frameSamples
            let frame = Array(pending[consumedSamples..<frameEnd])
            consumedSamples = frameEnd
            if let closed = consume(frame) { completed.append(closed) }
        }
        if consumedSamples > 0 {
            // Remove once per callback. Removing every 20 ms frame shifted the
            // entire remainder each time and made large catch-up batches
            // quadratic on the latency-sensitive dictation path.
            pending.removeFirst(consumedSamples)
        }
        return completed
    }

    mutating func flush() -> [Float]? {
        if !pending.isEmpty {
            let tail = pending
            pending.removeAll(keepingCapacity: true)
            if isInSpeech {
                utterance.append(contentsOf: tail)
            } else {
                preRoll.append(contentsOf: tail)
            }
        }
        guard isInSpeech, utterance.count >= Self.sampleRate / 2 else {
            reset()
            return nil
        }
        let out = utterance
        reset()
        return out
    }

    private mutating func consume(_ frame: [Float]) -> [Float]? {
        let meanSquare = frame.reduce(Float(0)) { $0 + $1 * $1 } / Float(frame.count)
        let rms = sqrt(meanSquare)
        let gate = max(0.009, noiseRMS * 3.2)
        let voiced = rms >= gate

        if !isInSpeech {
            if !voiced {
                noiseRMS = noiseRMS * 0.985 + rms * 0.015
                consecutiveSpeech = 0
            } else {
                consecutiveSpeech += 1
            }
            preRoll.append(contentsOf: frame)
            if preRoll.count > Self.preRollSamples {
                preRoll.removeFirst(preRoll.count - Self.preRollSamples)
            }
            if consecutiveSpeech >= Self.minSpeechFrames {
                isInSpeech = true
                utterance = preRoll
                preRoll.removeAll(keepingCapacity: true)
                consecutiveSilence = 0
            }
            return nil
        }

        utterance.append(contentsOf: frame)
        consecutiveSilence = voiced ? 0 : consecutiveSilence + 1
        let endpoint = consecutiveSilence >= Self.endpointFrames
        let reachedCap = utterance.count >= Self.maxUtteranceSamples
        guard endpoint || reachedCap else { return nil }

        let keepTail = endpoint ? 8 * Self.frameSamples : 0 // 160 ms of closing silence
        let trim = max(0, consecutiveSilence * Self.frameSamples - keepTail)
        let end = max(0, utterance.count - trim)
        let out = Array(utterance.prefix(end))
        isInSpeech = false
        utterance.removeAll(keepingCapacity: true)
        consecutiveSpeech = 0
        consecutiveSilence = 0
        return out.count >= Self.sampleRate / 2 ? out : nil
    }
}
