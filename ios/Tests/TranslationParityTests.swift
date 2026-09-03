import Foundation
import XCTest

final class TranslationParityTests: XCTestCase {
    func testRunOptionsCarriesTranslateTaskAcrossTheSwiftBoundary() throws {
        let options = RunOptions(
            language: "es",
            initialPrompt: nil,
            translate: true,
            diarize: false,
            wordTimestamps: false
        )
        let data = try XCTUnwrap(options.json.data(using: .utf8))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        XCTAssertEqual(object["language"] as? String, "es")
        XCTAssertEqual(object["translate"] as? Bool, true)
    }

    func testDefaultTaskRemainsSourceLanguageTranscription() throws {
        let data = try XCTUnwrap(RunOptions(language: nil, initialPrompt: nil).json.data(using: .utf8))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        XCTAssertEqual(object["translate"] as? Bool, false)
    }

    func testHumanReadableExportsLabelTranslatedOutput() {
        let result = Transcription(
            language: "es",
            segments: [
                TranscriptSegment(
                    startSec: 0,
                    endSec: 1,
                    text: "Hello from Madrid.",
                    speaker: nil,
                    confidence: 0.98
                )
            ],
            turns: [],
            speakerSegments: [],
            words: nil,
            droppedWindows: 0,
            audioSec: 1,
            skippedLeadingSec: 0,
            diarizationError: nil
        )
        let context = ExportContext(
            sourceName: "spanish.wav",
            wallSeconds: 0.5,
            names: [:],
            translatedToEnglish: true
        )

        XCTAssertTrue(TranscriptExport.markdown(from: result, context: context).contains("translated to English"))
        XCTAssertTrue(TranscriptExport.html(from: result, context: context).contains("translated to English"))

        let transcriptionContext = ExportContext(
            sourceName: "spanish.wav",
            wallSeconds: 0.5,
            names: [:],
            translatedToEnglish: false
        )
        XCTAssertTrue(
            TranscriptExport.markdown(from: result, context: transcriptionContext)
                .contains("· transcribed in")
        )
        XCTAssertTrue(
            TranscriptExport.html(from: result, context: transcriptionContext)
                .contains("· transcribed locally")
        )
    }
}
