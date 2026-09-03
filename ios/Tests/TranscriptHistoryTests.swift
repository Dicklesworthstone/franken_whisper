import Foundation
import XCTest
@testable import FrankenWhisper

final class TranscriptHistoryTests: XCTestCase {
    func testHistoryPersistsTranscriptWithoutSourceMediaOrPromptMetadata() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = TranscriptHistoryStore(directory: directory)
        let transcript = "# Transcript — interview.m4a\n\n**[0:00–0:02] Jeff:** hello"

        let entry = try store.record(
            TranscriptHistoryResult(markdown: transcript, sourceName: " interview.m4a ",
                                    language: "en", audioSeconds: 2, processingSeconds: 1.25,
                                    translatedToEnglish: false)
        )

        XCTAssertEqual(entry.sourceName, "interview.m4a")
        XCTAssertEqual(store.text(for: entry), transcript)
        XCTAssertEqual(store.storageBytes, Data(transcript.utf8).count)
        let manifest = try String(contentsOf: directory.appendingPathComponent("history.json"),
                                  encoding: .utf8)
        XCTAssertTrue(manifest.contains("frankenwhisper.transcript-history.v1"))
        XCTAssertFalse(manifest.contains(transcript))
        for forbidden in ["audioData", "videoURL", "initialPrompt", "words", "speakerNameMap"] {
            XCTAssertFalse(manifest.localizedCaseInsensitiveContains(forbidden), forbidden)
        }
        let restored = TranscriptHistoryStore(directory: directory)
        XCTAssertEqual(restored.entries, [entry])
        XCTAssertEqual(restored.text(for: entry), transcript)
    }

    func testHistoryRejectsEmptyOrNonFiniteResults() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = TranscriptHistoryStore(directory: directory)
        XCTAssertThrowsError(try store.record(result(markdown: "", audioSeconds: 1)))
        XCTAssertThrowsError(try store.record(result(markdown: "transcript", audioSeconds: .infinity)))
        XCTAssertTrue(store.entries.isEmpty)
    }

    func testHistoryPrunesByCountAndAge() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let now = Date()
        let store = TranscriptHistoryStore(directory: directory, now: now)
        for index in 0..<(TranscriptHistoryStore.maximumEntries + 3) {
            try store.record(result(markdown: "transcript \(index)", audioSeconds: 1), createdAt: now)
        }
        XCTAssertEqual(store.entries.count, TranscriptHistoryStore.maximumEntries)
        let retainedURLs = try store.entries.map { try XCTUnwrap(store.fileURL(for: $0)) }
        let expired = TranscriptHistoryStore(
            directory: directory,
            now: now.addingTimeInterval(TranscriptHistoryStore.maximumAge + 1)
        )
        XCTAssertTrue(expired.entries.isEmpty)
        XCTAssertEqual(expired.storageBytes, 0)
        XCTAssertTrue(retainedURLs.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) })
    }

    func testMalformedManifestDoesNotDeleteUnclaimedTranscript() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = TranscriptHistoryStore(directory: directory)
        let entry = try store.record(result(markdown: "transcript", audioSeconds: 1))
        let transcriptURL = try XCTUnwrap(store.fileURL(for: entry))
        try Data("{not-json".utf8).write(to: directory.appendingPathComponent("history.json"),
                                         options: .atomic)
        let recovered = TranscriptHistoryStore(directory: directory)
        XCTAssertTrue(recovered.entries.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: transcriptURL.path))
    }

    func testDeleteAndClearRemoveOnlyOwnedHistoryFiles() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let unrelated = directory.appendingPathComponent("keep-me.md")
        try Data("owner data".utf8).write(to: unrelated)
        let store = TranscriptHistoryStore(directory: directory)
        let first = try store.record(result(markdown: "first", audioSeconds: 1))
        let second = try store.record(result(markdown: "second", audioSeconds: 1))
        let firstURL = try XCTUnwrap(store.fileURL(for: first))
        let secondURL = try XCTUnwrap(store.fileURL(for: second))
        store.delete(first)
        XCTAssertFalse(FileManager.default.fileExists(atPath: firstURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: secondURL.path))
        store.deleteAll()
        XCTAssertFalse(FileManager.default.fileExists(atPath: secondURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: unrelated.path))
        XCTAssertTrue(store.entries.isEmpty)
    }

    private func result(markdown: String, audioSeconds: Double) -> TranscriptHistoryResult {
        TranscriptHistoryResult(markdown: markdown, sourceName: "recording", language: "en",
                                audioSeconds: audioSeconds, processingSeconds: 1,
                                translatedToEnglish: false)
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FrankenWhisperHistoryTests-" + UUID().uuidString,
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
