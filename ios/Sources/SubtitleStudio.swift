import AVKit
import CoreTransferable
import Photos
import SwiftUI
import UniformTypeIdentifiers
import UIKit

private extension SubtitleFontChoice {
    func swiftUIFont(size: CGFloat) -> Font {
        switch self {
        case .bold: .system(size: size, weight: .heavy)
        case .rounded: .system(size: size, weight: .heavy, design: .rounded)
        case .serif: .system(size: size, weight: .bold, design: .serif)
        case .mono: .system(size: size, weight: .bold, design: .monospaced)
        }
    }
}

private enum SubtitleStylePreset: String, CaseIterable, Identifiable {
    case monsterMint = "Monster Mint"
    case electricIce = "Electric Ice"
    case viralHeat = "Viral Heat"
    case cinema = "Cinema"

    var id: Self { self }

    var font: SubtitleFontChoice {
        switch self {
        case .monsterMint: .rounded
        case .electricIce, .viralHeat: .bold
        case .cinema: .serif
        }
    }

    var size: Double {
        switch self {
        case .monsterMint: 64
        case .electricIce: 62
        case .viralHeat: 70
        case .cinema: 56
        }
    }

    var highlight: Color {
        switch self {
        case .monsterMint: Color(red: 0.20, green: 0.96, blue: 0.68)
        case .electricIce: Color(red: 0.16, green: 0.78, blue: 1.00)
        case .viralHeat: Color(red: 1.00, green: 0.76, blue: 0.08)
        case .cinema: Color(red: 0.82, green: 0.50, blue: 1.00)
        }
    }

    var backdrop: Double {
        switch self {
        case .monsterMint: 0.58
        case .electricIce: 0.50
        case .viralHeat: 0.68
        case .cinema: 0.38
        }
    }
}

struct CaptionedVideoFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .mpeg4Movie) { file in
            SentTransferredFile(file.url)
        }
        .exportingCondition { $0.url.pathExtension.lowercased() == "mp4" }
        FileRepresentation(exportedContentType: .quickTimeMovie) { file in
            SentTransferredFile(file.url)
        }
        .exportingCondition { $0.url.pathExtension.lowercased() == "mov" }
    }
}

/// A compact wrapping layout keeps the editor preview faithful to the burn-in
/// renderer without allowing the active pill to shove neighboring words around.
private struct KaraokeWordWrap: Layout {
    var horizontalSpacing: CGFloat = 2
    var verticalSpacing: CGFloat = 3

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        arrangement(for: subviews, width: proposal.width ?? .greatestFiniteMagnitude).size
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        let result = arrangement(for: subviews, width: bounds.width)
        for (index, origin) in result.origins.enumerated() {
            subviews[index].place(
                at: CGPoint(x: bounds.minX + origin.x, y: bounds.minY + origin.y),
                anchor: .topLeading,
                proposal: .unspecified
            )
        }
    }

    private func arrangement(for subviews: Subviews, width: CGFloat) -> (
        size: CGSize,
        origins: [CGPoint]
    ) {
        var origins: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        var usedWidth: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width {
                x = 0
                y += lineHeight + verticalSpacing
                lineHeight = 0
            }
            origins.append(CGPoint(x: x, y: y))
            usedWidth = max(usedWidth, x + size.width)
            lineHeight = max(lineHeight, size.height)
            x += size.width + horizontalSpacing
        }
        return (CGSize(width: min(width, usedWidth), height: y + lineHeight), origins)
    }
}

@MainActor
struct SubtitleStudio: View {
    let video: VideoInput
    let result: Transcription

    @Environment(\.dismiss) private var dismiss
    @State private var player: AVPlayer
    @State private var fontChoice = SubtitleFontChoice.rounded
    @State private var fontSize: Double = 64
    @State private var textColor = Color.white
    @State private var highlightColor = Color(red: 0.20, green: 0.96, blue: 0.68)
    @State private var backgroundOpacity: Double = 0.58
    @State private var selectedPreset: SubtitleStylePreset? = .monsterMint
    @State private var exportProgress = 0.0
    @State private var isExporting = false
    @State private var exportedURL: URL?
    @State private var message: String?

    private let cues: [SubtitleCue]

    init(
        video: VideoInput,
        result: Transcription,
        speakerNames: [String: String] = [:]
    ) {
        self.video = video
        self.result = result
        cues = SubtitleTimeline.offset(
            SubtitleTimeline.makeCues(
                from: result.words,
                segmentSpeakers: result.segments.map(\.speaker),
                speakerSpans: result.speakerSegments.compactMap { run in
                    guard let start = run.startSec,
                          let end = run.endSec,
                          let lane = run.speaker
                    else { return nil }
                    return SubtitleSpeakerSpan(
                        startSec: start,
                        endSec: end,
                        laneID: lane,
                        confidence: run.speakerConfidence ?? 0
                    )
                },
                speakerNames: speakerNames
            ),
            by: video.audioTimelineOffset
        )
        _player = State(initialValue: AVPlayer(url: video.videoURL))
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    preview
                    disclosure
                    controls
                    exportControls
                }
                .padding()
                .frame(maxWidth: 760)
                .frame(maxWidth: .infinity)
            }
            .background(LaboratoryBackground())
            .navigationTitle("Subtitle Studio")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .disabled(isExporting)
                }
            }
        }
        .interactiveDismissDisabled(isExporting)
        .tint(Lab.emerald)
        .onDisappear {
            player.pause()
            discardExportedVideo()
        }
    }

    private var preview: some View {
        ZStack(alignment: .bottom) {
            VideoPlayer(player: player)
                .aspectRatio(video.aspectRatio, contentMode: .fit)
                .frame(maxHeight: 470)
                .background(.black)

            TimelineView(.animation(minimumInterval: 1.0 / 24.0, paused: false)) { _ in
                if let cue = activeCue(at: player.currentTime().seconds) {
                    previewCaption(cue, at: player.currentTime().seconds)
                        .padding(.horizontal, 18)
                        .padding(.bottom, 24)
                        .transition(.opacity)
                }
            }
            .allowsHitTesting(false)
        }
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Lab.emerald.opacity(0.32), lineWidth: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Captioned video preview")
    }

    private var disclosure: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("REAL WORD TIMING", systemImage: "waveform.and.magnifyingglass")
                .font(.system(size: Lab.typeSize(11), weight: .black, design: .monospaced))
                .foregroundStyle(Lab.emerald)
            Text(
                "The moving highlight follows \(cues.flatMap(\.words).count) DTW-aligned words "
                    + "from the local Rust model. The video, audio, transcript, and rendered export stay on this device."
            )
            .font(.system(size: Lab.typeSize(12)))
            .foregroundStyle(Lab.textSecondary)
            if cues.contains(where: { $0.speaker != nil }) {
                Text(
                    "Diarized voices keep distinct colors throughout the video. "
                        + "A speaker label appears only when you assigned that voice a name."
                )
                .font(.system(size: Lab.typeSize(12)))
                .foregroundStyle(Lab.textSecondary)
            }
        }
        .padding(14)
        .background(Lab.panel.opacity(0.92), in: RoundedRectangle(cornerRadius: 14))
    }

    private var controls: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 16) {
                LabLabel(text: "Caption Design")

                VStack(alignment: .leading, spacing: 8) {
                    Text("STARTING LOOK")
                        .font(.system(size: Lab.typeSize(9), weight: .black, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(SubtitleStylePreset.allCases) { preset in
                                Button {
                                    apply(preset)
                                } label: {
                                    HStack(spacing: 7) {
                                        Circle()
                                            .fill(
                                                LinearGradient(
                                                    colors: [preset.highlight.opacity(0.72), preset.highlight],
                                                    startPoint: .bottomLeading,
                                                    endPoint: .topTrailing
                                                )
                                            )
                                            .frame(width: 10, height: 10)
                                            .shadow(color: preset.highlight.opacity(0.65), radius: 3)
                                        Text(preset.rawValue)
                                            .lineLimit(1)
                                    }
                                    .font(.system(size: Lab.typeSize(11), weight: .bold))
                                    .foregroundStyle(
                                        selectedPreset == preset ? Color.black : Lab.textPrimary
                                    )
                                    .padding(.horizontal, 11)
                                    .padding(.vertical, 8)
                                    .background(
                                        selectedPreset == preset ? preset.highlight : Lab.panel,
                                        in: Capsule(style: .continuous)
                                    )
                                    .overlay {
                                        Capsule(style: .continuous)
                                            .stroke(preset.highlight.opacity(0.50), lineWidth: 1)
                                    }
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }

                HStack {
                    Text("Typeface")
                    Spacer()
                    Picker("Typeface", selection: resetting($fontChoice)) {
                        ForEach(SubtitleFontChoice.allCases) { choice in
                            Text(choice.rawValue).tag(choice)
                        }
                    }
                    .pickerStyle(.menu)
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Size")
                        Spacer()
                        Text("\(Int(fontSize))")
                            .monospacedDigit()
                            .foregroundStyle(Lab.textSecondary)
                    }
                    Slider(value: resetting($fontSize), in: 38...92, step: 1)
                }

                ColorPicker("Text color", selection: resetting($textColor), supportsOpacity: false)
                ColorPicker(
                    cues.contains(where: { $0.speaker != nil })
                        ? "Fallback karaoke highlight"
                        : "Karaoke highlight",
                    selection: resetting($highlightColor),
                    supportsOpacity: false
                )

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Backdrop")
                        Spacer()
                        Text("\(Int(backgroundOpacity * 100))%")
                            .monospacedDigit()
                            .foregroundStyle(Lab.textSecondary)
                    }
                    Slider(value: resetting($backgroundOpacity), in: 0...0.85, step: 0.05)
                }
            }
            .font(.system(size: Lab.typeSize(13)))
        }
    }

    @ViewBuilder
    private var exportControls: some View {
        if cues.isEmpty {
            StatusLine(
                kind: .err,
                text: "This transcript has no word-alignment data. Transcribe the video again to create karaoke captions."
            )
        } else {
            Button {
                burnInSubtitles()
            } label: {
                Label(
                    exportedURL == nil ? "Burn subtitles into video" : "Burn again",
                    systemImage: "captions.bubble.fill"
                )
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(isExporting)

            if isExporting {
                VStack(alignment: .leading, spacing: 7) {
                    ProgressView(value: exportProgress)
                        .tint(Lab.emerald)
                    Text("Rendering locally · \(Int(exportProgress * 100))%")
                        .font(.system(size: Lab.typeSize(11), design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                }
            }

            if let exportedURL {
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 10) { finishedActions(exportedURL) }
                    VStack(alignment: .leading, spacing: 10) { finishedActions(exportedURL) }
                }
            }

            if let message {
                StatusLine(kind: exportedURL == nil ? .err : .ok, text: message)
            }
        }
    }

    @ViewBuilder
    private func finishedActions(_ url: URL) -> some View {
        ShareLink(
            item: CaptionedVideoFile(url: url),
            preview: SharePreview("FrankenWhisper captioned video")
        ) {
            Label("Share video", systemImage: "square.and.arrow.up")
        }
        .buttonStyle(GhostButtonStyle(tint: Lab.emerald))

        Button {
            saveToPhotos(url)
        } label: {
            Label("Save to Photos", systemImage: "photo.badge.arrow.down")
        }
        .buttonStyle(GhostButtonStyle())
    }

    private func previewCaption(_ cue: SubtitleCue, at seconds: Double) -> some View {
        let accent = cue.speaker.map { Lab.speakerColor($0.laneID) } ?? highlightColor
        return VStack(spacing: 5) {
            if let name = cue.speaker?.displayName {
                Text(name)
                    .font(.system(size: Lab.typeSize(11), weight: .black, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                    .foregroundStyle(contrastingColor(for: accent))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(accent, in: Capsule(style: .continuous))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            KaraokeWordWrap(horizontalSpacing: 2, verticalSpacing: 3) {
                ForEach(cue.words) { word in
                    let active = seconds >= word.startSec && seconds < word.endSec
                    Text(word.text)
                        .font(fontChoice.swiftUIFont(size: CGFloat(min(36, fontSize * 0.56))))
                        .foregroundStyle(active ? contrastingColor(for: accent) : textColor)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 3)
                        .background {
                            Capsule(style: .continuous)
                                .fill(
                                    active
                                        ? AnyShapeStyle(
                                            LinearGradient(
                                                colors: [accent.opacity(0.78), accent],
                                                startPoint: .bottomLeading,
                                                endPoint: .topTrailing
                                            )
                                        )
                                        : AnyShapeStyle(Color.clear)
                                )
                                .shadow(
                                    color: active ? accent.opacity(0.62) : .clear,
                                    radius: active ? 6 : 0
                                )
                        }
                        .scaleEffect(active ? 1.08 : 1)
                        .animation(.spring(response: 0.20, dampingFraction: 0.62), value: active)
                        .shadow(color: .black.opacity(0.96), radius: 2, x: 0, y: 1)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.black.opacity(backgroundOpacity), in: RoundedRectangle(cornerRadius: 10))
        .overlay {
            if cue.speaker != nil {
                RoundedRectangle(cornerRadius: 10)
                    .stroke(accent.opacity(0.62), lineWidth: 1.5)
            }
        }
    }

    private func contrastingColor(for color: Color) -> Color {
        let uiColor = UIColor(color)
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        guard uiColor.getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            return .black
        }
        let luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return luminance > 0.58 ? .black.opacity(0.90) : .white
    }

    private func activeCue(at seconds: Double) -> SubtitleCue? {
        guard seconds.isFinite else { return nil }
        return cues.first { seconds >= $0.startSec && seconds < $0.endSec }
    }

    private func resetting<Value>(_ binding: Binding<Value>) -> Binding<Value> {
        Binding(
            get: { binding.wrappedValue },
            set: { value in
                binding.wrappedValue = value
                selectedPreset = nil
                discardExportedVideo()
                message = nil
                exportProgress = 0
            }
        )
    }

    private func apply(_ preset: SubtitleStylePreset) {
        selectedPreset = preset
        fontChoice = preset.font
        fontSize = preset.size
        textColor = .white
        highlightColor = preset.highlight
        backgroundOpacity = preset.backdrop
        discardExportedVideo()
        message = nil
        exportProgress = 0
    }

    private func burnInSubtitles() {
        player.pause()
        discardExportedVideo()
        message = nil
        exportProgress = 0
        isExporting = true
        let style = SubtitleRenderStyle(
            font: fontChoice,
            fontSize1080: CGFloat(fontSize),
            textColor: UIColor(textColor),
            highlightColor: UIColor(highlightColor),
            backgroundOpacity: CGFloat(backgroundOpacity)
        )

        Task {
            do {
                let url = try await SubtitleVideoExporter.export(
                    videoURL: video.videoURL,
                    cues: cues,
                    style: style
                ) { progress in
                    exportProgress = progress
                }
                exportedURL = url
                exportProgress = 1
                message = "Captioned video ready — still entirely local."
            } catch {
                message = "Could not render the captioned video: \(error.localizedDescription)"
            }
            isExporting = false
        }
    }

    private func discardExportedVideo() {
        guard let exportedURL else { return }
        try? FileManager.default.removeItem(at: exportedURL)
        self.exportedURL = nil
    }

    private func saveToPhotos(_ url: URL) {
        Task {
            let authorization = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard authorization == .authorized || authorization == .limited else {
                message = "Photos access is off. You can still use Share video."
                return
            }
            do {
                try await withCheckedThrowingContinuation {
                    (continuation: CheckedContinuation<Void, Error>) in
                    PHPhotoLibrary.shared().performChanges {
                        PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
                    } completionHandler: { success, error in
                        if success {
                            continuation.resume()
                        } else {
                            continuation.resume(
                                throwing: error ?? CocoaError(.fileWriteUnknown)
                            )
                        }
                    }
                }
                message = "Saved to Photos."
            } catch {
                message = "Could not save to Photos: \(error.localizedDescription)"
            }
        }
    }
}
