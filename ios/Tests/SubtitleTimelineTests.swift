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

    func testNegativeAudioTrackOffsetClipsOnlyAtTheVideoBoundary() {
        let cues = SubtitleTimeline.makeCues(from: [[
            Timing(text: "before", startSec: 0.10, endSec: 0.35),
            Timing(text: "visible", startSec: 0.40, endSec: 0.80),
        ]])

        let shifted = SubtitleTimeline.offset(cues, by: -0.25)
        let words = shifted.flatMap(\.words)

        XCTAssertEqual(words.map(\.text), ["before", "visible"])
        XCTAssertEqual(words[0].startSec, 0, accuracy: 0.000_1)
        XCTAssertEqual(words[0].endSec, 0.10, accuracy: 0.000_1)
        XCTAssertEqual(words[1].startSec, 0.15, accuracy: 0.000_1)
        XCTAssertEqual(words[1].endSec, 0.55, accuracy: 0.000_1)
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

    func testSpeakerSpanFallbackTreatsFloatingPointTieAsAConfidenceTie() {
        let cues = SubtitleTimeline.makeCues(
            from: [[Timing(text: "hello", startSec: 0, endSec: 5)]],
            segmentSpeakers: [],
            speakerSpans: [
                SubtitleSpeakerSpan(
                    startSec: 0.5,
                    endSec: 1.2,
                    laneID: "SPEAKER_00",
                    confidence: 0.95
                ),
                SubtitleSpeakerSpan(
                    startSec: 3.8,
                    endSec: 4.5,
                    laneID: "SPEAKER_01",
                    confidence: 0.20
                ),
            ]
        )

        XCTAssertEqual(cues.first?.speaker?.laneID, "SPEAKER_00")
    }

    func testPathologicalDTWSilenceIsClampedToRealSpeechTurns() {
        let cues = SubtitleTimeline.makeCues(
            from: [[
                Timing(text: "The", startSec: 0, endSec: 7.84),
                Timing(text: "twelve", startSec: 7.84, endSec: 8.18),
                Timing(text: "auto", startSec: 14.82, endSec: 16.0),
                Timing(text: "healthy", startSec: 19.92, endSec: 21.0),
            ]],
            segmentSpeakers: [],
            speechSpans: [
                SubtitleSpeechSpan(startSec: 7.60, endSec: 8.96),
                SubtitleSpeechSpan(startSec: 13.60, endSec: 14.88),
                SubtitleSpeechSpan(startSec: 19.60, endSec: 20.96),
            ]
        )
        let words = cues.flatMap(\.words)

        XCTAssertEqual(words[0].startSec, 7.60, accuracy: 0.000_1)
        XCTAssertEqual(words[0].endSec, 7.84, accuracy: 0.000_1)
        XCTAssertEqual(words[1].startSec, 7.84, accuracy: 0.000_1)
        XCTAssertEqual(words[1].endSec, 8.18, accuracy: 0.000_1)
        XCTAssertEqual(words[2].startSec, 14.76, accuracy: 0.000_1)
        XCTAssertEqual(words[2].endSec, 14.88, accuracy: 0.000_1)
        XCTAssertEqual(words[3].startSec, 19.92, accuracy: 0.000_1)
        XCTAssertEqual(words[3].endSec, 21.0, accuracy: 0.000_1)
    }

    func testOverlappingSpeakerTurnsRemainOneContinuousSpeechRegion() throws {
        let cues = SubtitleTimeline.makeCues(
            from: [[Timing(text: "overlap", startSec: 0, endSec: 5.0)]],
            segmentSpeakers: [],
            speechSpans: [
                SubtitleSpeechSpan(startSec: 0.70, endSec: 2.05),
                SubtitleSpeechSpan(startSec: 1.55, endSec: 2.90),
            ]
        )

        let word = try XCTUnwrap(cues.first?.words.first)
        XCTAssertEqual(word.startSec, 0.70, accuracy: 0.000_1)
        XCTAssertEqual(word.endSec, 2.90, accuracy: 0.000_1)
    }

    func testSeparatedSpeechTurnsNeverBridgeTheSilentGap() throws {
        let cues = SubtitleTimeline.makeCues(
            from: [[Timing(text: "boundary", startSec: 0, endSec: 5)]],
            segmentSpeakers: [],
            speechSpans: [
                SubtitleSpeechSpan(startSec: 0.50, endSec: 1.20),
                SubtitleSpeechSpan(startSec: 3.80, endSec: 4.50),
            ]
        )

        let word = try XCTUnwrap(cues.first?.words.first)
        XCTAssertEqual(word.startSec, 0.50, accuracy: 0.000_1)
        XCTAssertEqual(word.endSec, 1.20, accuracy: 0.000_1)
    }
}
