import Observation
import SwiftUI
import UIKit

struct TranscriptHistorySheet: View {
    @Bindable var history: TranscriptHistoryStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                LaboratoryBackground()
                ScrollView {
                    TranscriptHistoryLibrary(history: history)
                        .padding(16)
                }
                .scrollIndicators(.hidden)
            }
            .navigationTitle("Recent transcripts")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .accessibilityIdentifier("transcript-history-sheet")
    }
}

struct TranscriptHistoryLibrary: View {
    @Bindable var history: TranscriptHistoryStore
    @State private var confirmClear = false
    @State private var copiedID: UUID?

    var body: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    LabLabel(text: "Private library")
                    Spacer()
                    if !history.entries.isEmpty {
                        Button("Clear All", role: .destructive) { confirmClear = true }
                            .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                    }
                }
                privacyNote
                if history.entries.isEmpty {
                    ContentUnavailableView(
                        "No recent transcripts",
                        systemImage: "quote.bubble",
                        description: Text("Finished batch transcriptions appear here automatically.")
                    )
                    .foregroundStyle(Lab.textSecondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 24)
                } else {
                    LazyVStack(spacing: 12) {
                        ForEach(history.entries) { entry in historyCard(entry) }
                    }
                }
            }
        }
        .alert("Clear recent transcripts?", isPresented: $confirmClear) {
            Button("Clear All", role: .destructive) { history.deleteAll() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "This permanently removes every locally saved transcript. "
                    + "Your models and source recordings stay intact."
            )
        }
        .accessibilityIdentifier("transcript-history-library")
    }

    private var privacyNote: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "lock.shield.fill").foregroundStyle(Lab.emerald)
            VStack(alignment: .leading, spacing: 5) {
                Text(
                    "Only the Markdown transcript and minimal run facts are saved. Source audio, "
                        + "video, model prompts, and word-level timing are never copied into history. "
                        + "The newest 20 transcripts are kept for up to 14 days."
                )
                .font(.system(size: Lab.typeSize(11), weight: .medium))
                .foregroundStyle(Lab.textSecondary)
                Text(storageSummary)
                    .font(.system(size: Lab.typeSize(9), weight: .bold, design: .monospaced))
                    .foregroundStyle(Lab.violet)
            }
        }
        .padding(12)
        .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
    }

    private var storageSummary: String {
        let size = ByteCountFormatter.string(fromByteCount: Int64(history.storageBytes), countStyle: .file)
        let noun = history.entries.count == 1 ? "transcript" : "transcripts"
        return "\(history.entries.count) \(noun) · \(size) on this device"
    }

    private func historyCard(_ entry: TranscriptHistoryEntry) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            historyHeader(entry)
            historyPreview(entry)
            historyActions(entry)
            Text(entrySummary(entry))
                .font(.system(size: Lab.typeSize(9), design: .monospaced))
                .foregroundStyle(Lab.textSecondary)
        }
        .padding(12)
        .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Lab.stroke, lineWidth: 1))
        .accessibilityIdentifier("history-entry-\(entry.id.uuidString)")
    }

    private func historyHeader(_ entry: TranscriptHistoryEntry) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text(entry.sourceName)
                    .font(.system(size: Lab.typeSize(15), weight: .bold))
                    .foregroundStyle(Lab.textPrimary)
                    .lineLimit(1)
                Text(entry.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.system(size: Lab.typeSize(10), design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
            }
            Spacer()
            Text(entry.translatedToEnglish ? "EN TRANSLATION" : entry.language.uppercased())
                .font(.system(size: Lab.typeSize(9), weight: .bold, design: .monospaced))
                .foregroundStyle(Lab.cyan)
                .lineLimit(1)
        }
    }

    @ViewBuilder
    private func historyPreview(_ entry: TranscriptHistoryEntry) -> some View {
        if let preview = history.text(for: entry) {
            Text(preview)
                .font(.system(size: Lab.typeSize(11), design: .monospaced))
                .foregroundStyle(Lab.textPrimary)
                .lineLimit(5)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            StatusLine(kind: .warn, text: "The saved transcript is unavailable.")
        }
    }

    private func historyActions(_ entry: TranscriptHistoryEntry) -> some View {
        HStack(spacing: 8) {
            if let url = history.fileURL(for: entry) {
                ShareLink(item: url) { Label("Share", systemImage: "square.and.arrow.up") }
                    .buttonStyle(GhostButtonStyle())
            }
            Button {
                UIPasteboard.general.string = history.text(for: entry)
                copiedID = entry.id
            } label: {
                Label(copiedID == entry.id ? "Copied" : "Copy",
                      systemImage: copiedID == entry.id ? "checkmark" : "doc.on.doc")
            }
            .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
            .disabled(history.text(for: entry) == nil)
            Spacer()
            Button(role: .destructive) { history.delete(entry) } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(GhostButtonStyle(tint: Lab.danger))
            .accessibilityLabel("Delete \(entry.sourceName) transcript")
        }
    }

    private func entrySummary(_ entry: TranscriptHistoryEntry) -> String {
        ["\(entry.characterCount) characters",
         String(format: "%.1fs audio", entry.audioSeconds),
         String(format: "%.1fs processing", entry.processingSeconds)]
            .joined(separator: " · ")
    }
}
