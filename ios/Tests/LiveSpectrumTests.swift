import XCTest

final class LiveSpectrumTests: XCTestCase {
    func testSilenceProducesEmptySpectrum() {
        let frame = LiveSpectrumAnalyzer.analyze([Float](repeating: 0, count: 1_024))
        XCTAssertEqual(frame.level, 0)
        XCTAssertEqual(frame.bands, [Float](repeating: 0, count: LiveSpectrumAnalyzer.bandCount))
    }

    func testSpectrumIsBounded() {
        let samples = (0..<1_024).map { index in
            Float(sin(2 * Double.pi * 440 * Double(index) / 16_000)) * 0.25
        }
        let frame = LiveSpectrumAnalyzer.analyze(samples)
        XCTAssertGreaterThan(frame.level, 0)
        XCTAssertEqual(frame.bands.count, LiveSpectrumAnalyzer.bandCount)
        XCTAssertTrue(frame.bands.allSatisfy { (0...1).contains($0) })
    }

    func testToneEnergyPeaksNearItsBand() {
        let samples = (0..<1_024).map { index in
            Float(sin(2 * Double.pi * 440 * Double(index) / 16_000)) * 0.25
        }
        let frame = LiveSpectrumAnalyzer.analyze(samples)
        let peak = frame.bands.enumerated().max(by: { $0.element < $1.element })?.offset
        XCTAssertNotNil(peak)
        XCTAssertTrue((4...7).contains(peak!))
    }
}
