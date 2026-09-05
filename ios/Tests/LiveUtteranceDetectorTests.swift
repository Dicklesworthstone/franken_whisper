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

    func testLargeBatchMatchesIrregularCallbackChunks() {
        let input =
            samples(amplitude: 0.002, frames: 9)
            + samples(amplitude: 0.04, frames: 80)
            + samples(amplitude: 0, frames: 42)
            + samples(amplitude: 0.05, frames: 55)
            + samples(amplitude: 0, frames: 41)

        var batched = LiveUtteranceDetector()
        var chunked = LiveUtteranceDetector()
        let batchedOutput = batched.push(input)

        var chunkedOutput: [[Float]] = []
        var cursor = 0
        let callbackSizes = [137, 911, 43, 2_047, 319, 5_003]
        var callbackIndex = 0
        while cursor < input.count {
            let end = min(input.count, cursor + callbackSizes[callbackIndex % callbackSizes.count])
            chunkedOutput.append(contentsOf: chunked.push(Array(input[cursor..<end])))
            cursor = end
            callbackIndex += 1
        }

        XCTAssertEqual(batchedOutput, chunkedOutput)
        XCTAssertEqual(batched.flush(), chunked.flush())
        XCTAssertEqual(batched.isInSpeech, chunked.isInSpeech)
    }

    func testActiveUtteranceSnapshotIsNonMutatingAndIncludesPendingTail() {
        var detector = LiveUtteranceDetector()
        let speech = samples(amplitude: 0.04, frames: 20)
        let pendingTail = Array(repeating: Float(0.05), count: 137)

        XCTAssertTrue(detector.push(speech + pendingTail).isEmpty)
        let first = detector.activeUtterance
        let second = detector.activeUtterance

        XCTAssertNotNil(first)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first?.count, 20 * frame + pendingTail.count)
        XCTAssertTrue(detector.isInSpeech)
    }

    func testActiveUtteranceDisappearsAfterEndpoint() {
        var detector = LiveUtteranceDetector()
        let completed = detector.push(
            samples(amplitude: 0.04, frames: 45)
                + samples(amplitude: 0, frames: 40))

        XCTAssertEqual(completed.count, 1)
        XCTAssertNil(detector.activeUtterance)
    }

    func testLiveDecodeResultDecodesStableAndMutableFields() throws {
        let json = #"""
        {
          "language": "en",
          "commit_text": "stable words",
          "commit_through_sec": 1.4,
          "partial_tail": " changing edge",
          "commit_tokens": 2,
          "commit_confidence": 0.91,
          "holdback": true,
          "end_of_utterance": false
        }
        """#.data(using: .utf8)!

        let result = try Engine.decoder().decode(LiveDecodeResult.self, from: json)

        XCTAssertEqual(result.language, "en")
        XCTAssertEqual(result.commitText, "stable words")
        XCTAssertEqual(result.commitThroughSec, 1.4)
        XCTAssertEqual(result.partialTail, " changing edge")
        XCTAssertEqual(result.commitTokens, 2)
        XCTAssertEqual(result.commitConfidence, 0.91)
        XCTAssertTrue(result.holdback)
        XCTAssertFalse(result.endOfUtterance)
    }

    private func samples(amplitude: Float, frames: Int) -> [Float] {
        Array(repeating: amplitude, count: frame * frames)
    }
}
