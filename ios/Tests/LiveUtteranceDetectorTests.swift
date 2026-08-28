import XCTest

final class LiveUtteranceDetectorTests: XCTestCase {
    private let frame = 320

    func testSilenceDoesNotProduceAnUtterance() {
        var detector = LiveUtteranceDetector()

        XCTAssertTrue(detector.push(samples(amplitude: 0, frames: 100)).isEmpty)
        XCTAssertNil(detector.flush())
    }

    func testNaturalPauseCommitsOnePhrase() {
        var detector = LiveUtteranceDetector()
        let speech = samples(amplitude: 0.04, frames: 55)
        let pause = samples(amplitude: 0, frames: 40)

        let completed = detector.push(speech + pause)

        XCTAssertEqual(completed.count, 1)
        XCTAssertGreaterThan(completed[0].count, 16_000)
        XCTAssertFalse(detector.isInSpeech)
    }

    func testTwoSpeechBurstsProduceTwoPhrases() {
        var detector = LiveUtteranceDetector()
        let speech = samples(amplitude: 0.04, frames: 45)
        let pause = samples(amplitude: 0, frames: 40)

        let completed = detector.push(speech + pause + speech + pause)

        XCTAssertEqual(completed.count, 2)
    }

    func testFlushReturnsActiveHalfSecondPhrase() {
        var detector = LiveUtteranceDetector()
        XCTAssertTrue(detector.push(samples(amplitude: 0.04, frames: 30)).isEmpty)

        let final = detector.flush()

        XCTAssertNotNil(final)
        XCTAssertGreaterThanOrEqual(final?.count ?? 0, 8_000)
        XCTAssertFalse(detector.isInSpeech)
    }

    private func samples(amplitude: Float, frames: Int) -> [Float] {
        Array(repeating: amplitude, count: frame * frames)
    }
}
