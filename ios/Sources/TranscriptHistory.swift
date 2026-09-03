import Foundation
import Observation

struct TranscriptHistoryEntry: Codable, Equatable, Identifiable {
    let id: UUID
    let createdAt: Date
    let sourceName: String
    let language: String
    let characterCount: Int
    let audioSeconds: Double
    let processingSeconds: Double
    let translatedToEnglish: Bool
    let byteCount: Int
    let fileName: String

    private enum CodingKeys: String, CodingKey {
        case id, createdAtMilliseconds, sourceName, language, characterCount
        case audioSeconds, processingSeconds, translatedToEnglish, byteCount, fileName
    }

    init(id: UUID, createdAt: Date, sourceName: String, language: String,
         characterCount: Int, audioSeconds: Double, processingSeconds: Double,
         translatedToEnglish: Bool, byteCount: Int, fileName: String) {
        self.id = id
        self.createdAt = createdAt
        self.sourceName = sourceName
        self.language = language
        self.characterCount = characterCount
        self.audioSeconds = audioSeconds
        self.processingSeconds = processingSeconds
        self.translatedToEnglish = translatedToEnglish
        self.byteCount = byteCount
        self.fileName = fileName
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(UUID.self, forKey: .id)
        let milliseconds = try values.decode(Int64.self, forKey: .createdAtMilliseconds)
        createdAt = Date(timeIntervalSince1970: Double(milliseconds) / 1_000)
        sourceName = try values.decode(String.self, forKey: .sourceName)
        language = try values.decode(String.self, forKey: .language)
        characterCount = try values.decode(Int.self, forKey: .characterCount)
        audioSeconds = try values.decode(Double.self, forKey: .audioSeconds)
        processingSeconds = try values.decode(Double.self, forKey: .processingSeconds)
        translatedToEnglish = try values.decode(Bool.self, forKey: .translatedToEnglish)
        byteCount = try values.decode(Int.self, forKey: .byteCount)
        fileName = try values.decode(String.self, forKey: .fileName)
    }

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(id, forKey: .id)
        try values.encode(Self.milliseconds(createdAt), forKey: .createdAtMilliseconds)
        try values.encode(sourceName, forKey: .sourceName)
        try values.encode(language, forKey: .language)
        try values.encode(characterCount, forKey: .characterCount)
        try values.encode(audioSeconds, forKey: .audioSeconds)
        try values.encode(processingSeconds, forKey: .processingSeconds)
        try values.encode(translatedToEnglish, forKey: .translatedToEnglish)
        try values.encode(byteCount, forKey: .byteCount)
        try values.encode(fileName, forKey: .fileName)
    }

    static func normalized(_ date: Date) -> Date? {
        let scaled = date.timeIntervalSince1970 * 1_000
        guard scaled.isFinite, let value = Int64(exactly: scaled.rounded()) else { return nil }
        return Date(timeIntervalSince1970: Double(value) / 1_000)
    }

    private static func milliseconds(_ date: Date) throws -> Int64 {
        let scaled = date.timeIntervalSince1970 * 1_000
        guard scaled.isFinite, let value = Int64(exactly: scaled.rounded()) else {
            throw TranscriptHistoryError.invalidResult
        }
        return value
    }
}

struct TranscriptHistoryResult {
    let markdown: String
    let sourceName: String
    let language: String
    let audioSeconds: Double
    let processingSeconds: Double
    let translatedToEnglish: Bool
}

@Observable
final class TranscriptHistoryStore {
    static let maximumEntries = 20
    static let maximumAge: TimeInterval = 14 * 24 * 60 * 60
    static let maximumStoredBytes = 8 * 1_024 * 1_024

    private static let manifestSchema = "frankenwhisper.transcript-history.v1"
    private static let manifestName = "history.json"
    private(set) var entries: [TranscriptHistoryEntry] = []
    private(set) var storageBytes = 0
    private let directory: URL
    private let fileManager: FileManager
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(directory requestedDirectory: URL? = nil, now: Date = .now,
         fileManager: FileManager = .default) {
        self.fileManager = fileManager
        directory = requestedDirectory ?? Self.defaultDirectory(fileManager: fileManager)
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        prepareDirectory()
        reload(now: now)
    }

    @discardableResult
    func record(_ result: TranscriptHistoryResult,
                createdAt: Date = .now) throws -> TranscriptHistoryEntry {
        let data = Data(result.markdown.utf8)
        guard let stableCreatedAt = TranscriptHistoryEntry.normalized(createdAt),
              !result.markdown.isEmpty, data.count <= Self.maximumStoredBytes,
              result.audioSeconds.isFinite, result.audioSeconds >= 0,
              result.processingSeconds.isFinite, result.processingSeconds >= 0 else {
            throw TranscriptHistoryError.invalidResult
        }

        let id = UUID()
        let fileName = "\(id.uuidString.lowercased()).md"
        let url = directory.appendingPathComponent(fileName, isDirectory: false)
        try data.write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        excludeFromBackup(url)
        let entry = TranscriptHistoryEntry(
            id: id, createdAt: stableCreatedAt,
            sourceName: Self.boundedLabel(result.sourceName, fallback: "Recording"),
            language: Self.boundedLabel(result.language, fallback: "Unknown language"),
            characterCount: result.markdown.count, audioSeconds: result.audioSeconds,
            processingSeconds: result.processingSeconds,
            translatedToEnglish: result.translatedToEnglish,
            byteCount: data.count, fileName: fileName
        )
        let previousEntries = entries
        entries.insert(entry, at: 0)
        let removed = prune(now: stableCreatedAt, deleteRemoved: false)
        do {
            try persistManifest()
            for removedEntry in removed { removeDocument(for: removedEntry) }
        } catch {
            entries = previousEntries
            try? fileManager.removeItem(at: url)
            recalculateStorage()
            throw error
        }
        return entry
    }

    func fileURL(for entry: TranscriptHistoryEntry) -> URL? {
        guard entries.contains(where: { $0.id == entry.id && $0.fileName == entry.fileName }),
              Self.isOwnedFileName(entry.fileName, id: entry.id) else { return nil }
        let url = directory.appendingPathComponent(entry.fileName, isDirectory: false)
        guard fileManager.fileExists(atPath: url.path) else { return nil }
        return url
    }

    func text(for entry: TranscriptHistoryEntry) -> String? {
        guard let url = fileURL(for: entry), let data = try? Data(contentsOf: url),
              data.count == entry.byteCount, data.count <= Self.maximumStoredBytes else { return nil }
        return String(data: data, encoding: .utf8)
    }

    func delete(_ entry: TranscriptHistoryEntry) {
        guard let index = entries.firstIndex(where: { $0.id == entry.id }) else { return }
        let removed = entries.remove(at: index)
        recalculateStorage()
        do {
            try persistManifest()
            removeDocument(for: removed)
        } catch {
            entries.insert(removed, at: index)
            recalculateStorage()
        }
    }

    func deleteAll() {
        let removed = entries
        entries.removeAll(keepingCapacity: false)
        recalculateStorage()
        do {
            try persistManifest()
            for entry in removed { removeDocument(for: entry) }
        } catch {
            entries = removed
            recalculateStorage()
        }
    }

    private func prepareDirectory() {
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        excludeFromBackup(directory)
    }

    private func excludeFromBackup(_ url: URL) {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableURL = url
        try? mutableURL.setResourceValues(values)
    }

    private func reload(now: Date) {
        let manifestURL = directory.appendingPathComponent(Self.manifestName)
        guard let data = try? Data(contentsOf: manifestURL), data.count <= 512_000,
              let manifest = try? decoder.decode(Manifest.self, from: data),
              manifest.schema == Self.manifestSchema else {
            entries = []
            storageBytes = 0
            return
        }
        var seenIDs = Set<UUID>()
        entries = manifest.entries.filter { entry in
            guard seenIDs.insert(entry.id).inserted, Self.isValidMetadata(entry),
                  Self.isOwnedFileName(entry.fileName, id: entry.id) else { return false }
            let url = directory.appendingPathComponent(entry.fileName, isDirectory: false)
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true, values.fileSize == entry.byteCount else { return false }
            return true
        }
        let removed = prune(now: now, deleteRemoved: false)
        do {
            try persistManifest()
            for entry in removed { removeDocument(for: entry) }
        } catch {
            // The previous manifest and every referenced document remain intact.
        }
    }

    @discardableResult
    private func prune(now: Date, deleteRemoved: Bool = true) -> [TranscriptHistoryEntry] {
        entries.sort {
            if $0.createdAt == $1.createdAt { return $0.id.uuidString < $1.id.uuidString }
            return $0.createdAt > $1.createdAt
        }
        var kept: [TranscriptHistoryEntry] = []
        var removed: [TranscriptHistoryEntry] = []
        var bytes = 0
        for entry in entries {
            let fits = kept.count < Self.maximumEntries
                && now.timeIntervalSince(entry.createdAt) <= Self.maximumAge
                && entry.createdAt.timeIntervalSince(now) <= 60
                && bytes <= Self.maximumStoredBytes - entry.byteCount
            if fits {
                kept.append(entry)
                bytes += entry.byteCount
            } else {
                removed.append(entry)
            }
        }
        entries = kept
        storageBytes = bytes
        if deleteRemoved { for entry in removed { removeDocument(for: entry) } }
        return removed
    }

    private func removeDocument(for entry: TranscriptHistoryEntry) {
        guard Self.isOwnedFileName(entry.fileName, id: entry.id) else { return }
        try? fileManager.removeItem(at: directory.appendingPathComponent(entry.fileName))
    }

    private func recalculateStorage() {
        storageBytes = entries.reduce(0) { $0 + $1.byteCount }
    }

    private func persistManifest() throws {
        let data = try encoder.encode(Manifest(schema: Self.manifestSchema, entries: entries))
        try data.write(to: directory.appendingPathComponent(Self.manifestName),
                       options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }

    private static func boundedLabel(_ value: String, fallback: String) -> String {
        let bounded = String(value.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160))
        return bounded.isEmpty ? fallback : bounded
    }

    private static func isValidMetadata(_ entry: TranscriptHistoryEntry) -> Bool {
        !entry.sourceName.isEmpty && entry.sourceName.count <= 160
            && !entry.language.isEmpty && entry.language.count <= 160
            && entry.characterCount > 0
            && entry.audioSeconds.isFinite && entry.audioSeconds >= 0
            && entry.processingSeconds.isFinite && entry.processingSeconds >= 0
            && entry.byteCount > 0 && entry.byteCount <= maximumStoredBytes
    }

    private static func isOwnedFileName(_ fileName: String, id: UUID) -> Bool {
        let url = URL(fileURLWithPath: fileName)
        return url.lastPathComponent == fileName
            && url.deletingPathExtension().lastPathComponent == id.uuidString.lowercased()
            && url.pathExtension.lowercased() == "md"
    }

    private static func defaultDirectory(fileManager: FileManager) -> URL {
        let root = (try? fileManager.url(for: .applicationSupportDirectory,
                                         in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fileManager.temporaryDirectory
        return root.appendingPathComponent("FrankenWhisper", isDirectory: true)
            .appendingPathComponent("Transcript History", isDirectory: true)
    }

    private struct Manifest: Codable {
        let schema: String
        let entries: [TranscriptHistoryEntry]
    }
}

enum TranscriptHistoryError: LocalizedError {
    case invalidResult
    var errorDescription: String? { "The transcript is not valid for local history." }
}
