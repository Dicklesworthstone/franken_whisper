import Foundation

struct LiveSpectrumFrame: Equatable {
    var level: Float
    var bands: [Float]
}

/// A deliberately small display-only analyzer. It samples a Hann-windowed
/// microphone frame at logarithmically spaced speech frequencies. The result
/// is an honest view of the captured signal; it is never fed into inference.
enum LiveSpectrumAnalyzer {
    static let bandCount = 14
    private static let sampleRate: Float = 16_000

    static func analyze(_ samples: [Float]) -> LiveSpectrumFrame {
        guard samples.count >= 64 else {
            return LiveSpectrumFrame(level: 0, bands: Array(repeating: 0, count: bandCount))
        }

        let count = min(1_024, samples.count)
        let start = samples.count - count
        var sumSquares: Float = 0
        for index in start..<samples.count {
            let value = samples[index]
            sumSquares += value * value
        }
        let rms = sqrt(sumSquares / Float(count))
        let decibels = 20 * log10(max(rms, 0.000_001))
        let level = min(1, max(0, (decibels + 55) / 43))

        guard level > 0.005 else {
            return LiveSpectrumFrame(level: 0, bands: Array(repeating: 0, count: bandCount))
        }

        let low: Float = 90
        let high: Float = 6_000
        var magnitudes = [Float](repeating: 0, count: bandCount)
        for band in 0..<bandCount {
            let fraction = Float(band) / Float(bandCount - 1)
            let frequency = low * pow(high / low, fraction)
            let angularStep = 2 * Float.pi * frequency / sampleRate
            var real: Float = 0
            var imaginary: Float = 0
            for offset in 0..<count {
                let window = 0.5 - 0.5 * cos(2 * Float.pi * Float(offset) / Float(count - 1))
                let phase = angularStep * Float(offset)
                let sample = samples[start + offset] * window
                real += sample * cos(phase)
                imaginary -= sample * sin(phase)
            }
            magnitudes[band] = sqrt(real * real + imaginary * imaginary)
        }

        let strongest = magnitudes.max() ?? 0
        guard strongest > 0 else {
            return LiveSpectrumFrame(level: level, bands: Array(repeating: 0, count: bandCount))
        }
        let bands = magnitudes.map { magnitude in
            let shape = pow(min(1, magnitude / strongest), 0.55)
            return min(1, max(0, shape * (0.18 + 0.82 * level)))
        }
        return LiveSpectrumFrame(level: level, bands: bands)
    }
}
