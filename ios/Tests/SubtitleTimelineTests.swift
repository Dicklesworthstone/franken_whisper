import XCTest

final class SubtitleTimelineTests: XCTestCase {
    private struct Timing: SubtitleTimingSource {
        var text: String
        var startSec: Double
        var endSec: Double
    }

    func testPunctuationUsesThePreviousWordsRealTiming() {
        let cues = SubtitleTimeline.makeCues(from: [[
            Timing(text: " Hello", startSec: 0.10, endSec: 0.42),
            Timing(text: ",", startSec: 0.42, endSec: 0.46),
            Timing(text: " world", startSec: 0.47, endSec: 0.91),
            Timing(text: "!", startSec: 0.91, endSec: 0.96),
        ]])

        XCTAssertEqual(cues.count, 1)
        XCTAssertEqual(cues[0].words.map(\.text), ["Hello,", "world!"])
        XCTAssertEqual(cues[0].words[0].startSec, 0.10)
        XCTAssertEqual(cues[0].words[0].endSec, 0.46)
        XCTAssertEqual(cues[0].words[1].endSec, 0.96)
    }

    func testLongSpeechIsSplitIntoReadableBoundedCues() {
        let timings = (0..<16).map { index in
            Timing(
                text: "word\(index)",
                startSec: Double(index) * 0.34,
                endSec: Double(index) * 0.34 + 0.28
            )
        }

        let cues = SubtitleTimeline.makeCues(from: [timings])

        XCTAssertEqual(cues.flatMap(\.words).count, timings.count)
        XCTAssertTrue(cues.allSatisfy { $0.words.count <= SubtitleTimeline.maximumWordsPerCue })
        XCTAssertTrue(cues.allSatisfy {
            $0.endSec - $0.startSec <= SubtitleTimeline.maximumCueDuration
        })
    }

    func testInvalidDecoderTimingIsRejectedRatherThanFabricated() {
        let cues = SubtitleTimeline.makeCues(from: [[
            Timing(text: "valid", startSec: 0, endSec: 0.4),
            Timing(text: "backwards", startSec: 1.0, endSec: 0.8),
            Timing(text: "unknown", startSec: .nan, endSec: 1.2),
        ]])

        XCTAssertEqual(cues.flatMap(\.words).map(\.text), ["valid"])
    }

    func testAudioTrackOffsetIsRestoredForVideoBurnIn() {
        let cues = SubtitleTimeline.makeCues(from: [[
            Timing(text: "delayed", startSec: 0.20, endSec: 0.62),
        ]])

        let shifted = SubtitleTimeline.offset(cues, by: 1.75)

        XCTAssertEqual(shifted[0].words[0].startSec, 1.95, accuracy: 0.000_1)
        XCTAssertEqual(shifted[0].words[0].endSec, 2.37, accuracy: 0.000_1)
    }

    func testSpeakerChangesSplitCuesAndOnlyExplicitNamesAreDisplayed() {
        let words = [
            [
                Timing(text: "first", startSec: 0, endSec: 0.35),
                Timing(text: "voice", startSec: 0.4, endSec: 0.75)
            ],
            [
                Timing(text: "second", startSec: 0.8, endSec: 1.15),
                Timing(text: "voice", startSec: 1.2, endSec: 1.55)
            ]
        ]

        let cues = SubtitleTimeline.makeCues(
            from: words,
            segmentSpeakers: ["SPEAKER_00", "SPEAKER_01"],
            speakerNames: ["SPEAKER_01": "Sarah"]
        )

        XCTAssertEqual(cues.count, 2)
        XCTAssertEqual(cues[0].speaker?.laneID, "SPEAKER_00")
        XCTAssertNil(cues[0].speaker?.displayName, "Anonymous lanes are color-only")
        XCTAssertEqual(cues[1].speaker?.laneID, "SPEAKER_01")
        XCTAssertEqual(cues[1].speaker?.displayName, "Sarah")
    }

    func testPunctuationDoesNotCrossAChangeOfSpeaker() {
        let words = [
            [Timing(text: "hello", startSec: 0, endSec: 0.35)],
            [
                Timing(text: "—", startSec: 0.4, endSec: 0.45),
                Timing(text: "goodbye", startSec: 0.46, endSec: 0.9)
            ]
        ]

        let cues = SubtitleTimeline.makeCues(
            from: words,
            segmentSpeakers: ["SPEAKER_00", "SPEAKER_01"]
        )

        XCTAssertEqual(cues.count, 2)
        XCTAssertEqual(cues[0].text, "hello")
        XCTAssertEqual(cues[1].text, "— goodbye")
        XCTAssertEqual(cues[1].speaker?.laneID, "SPEAKER_01")
    }

    func testSpeakerSpanFallbackPrefersGreatestOverlapThenConfidence() {
        let words = [[Timing(text: "hello", startSec: 0.4, endSec: 0.8)]]
        let cues = SubtitleTimeline.makeCues(
            from: words,
            segmentSpeakers: [],
            speakerSpans: [
                SubtitleSpeakerSpan(
                    startSec: 0.0,
                    endSec: 0.6,
                    laneID: "SPEAKER_00",
                    confidence: 0.99
                ),
                SubtitleSpeakerSpan(
                    startSec: 0.45,
                    endSec: 0.9,
                    laneID: "SPEAKER_01",
                    confidence: 0.50
                )
            ]
        )

        XCTAssertEqual(cues.first?.speaker?.laneID, "SPEAKER_01")
    }
}
