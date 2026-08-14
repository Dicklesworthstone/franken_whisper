// The whole main screen: the website playground's three numbered stations as
// a native scrolling page — 01 The Specimen (models + engine), 02 The Signal
// (microphone or file), 03 The Transcript (live windows, speakers, export).

import SwiftUI
import UniformTypeIdentifiers

struct LabView: View {
    @State private var model = LabModel()
    @State private var showDownloadConsent = false
    @State private var showFileImporter = false
    @State private var exportFormat: TranscriptFormat = .text
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            Lab.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    header
                    specimenCard
                    signalCard
                    transcriptCard
                    footer
                }
                .padding(16)
                .frame(maxWidth: 700)
                .frame(maxWidth: .infinity)
            }
        }
        .tint(Lab.emerald)
        .alert(
            "Something snapped", isPresented: .init(
                get: { model.lastError != nil },
                set: { if !$0 { model.lastError = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.lastError ?? "")
        }
        .confirmationDialog(
            "Download the models?", isPresented: $showDownloadConsent, titleVisibility: .visible
        ) {
            Button("Download \(Self.gigabytes(ModelManifest.totalBytes))") {
                model.store.startDownload()
            }
            Button("Not now", role: .cancel) {}
        } message: {
            Text(
                "Whisper large-v3-turbo (q8), the Sortformer speaker diarizer, and the "
                    + "FastEnhancer denoiser — downloaded once over your connection, verified "
                    + "by SHA-256, and stored on this device. Everything afterwards runs offline.")
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio, .mpeg4Audio, .mp3, .wav],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                model.acceptFile(url: url)
            }
        }
        .onReceive(
            NotificationCenter.default.publisher(
                for: UIApplication.didReceiveMemoryWarningNotification)
        ) { _ in
            model.unloadEngineForMemoryPressure()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background {
                model.unloadEngineForMemoryPressure()
            }
        }
    }

    // ── Header ─────────────────────────────────────────────────────────────

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Bolt()
                Text("FRANKENWHISPER")
                    .font(.system(size: 22, weight: .black, design: .monospaced))
                    .kerning(2)
                    .foregroundStyle(Lab.textPrimary)
                Bolt()
            }
            Text("IT_HEARS // speech → text with speakers, entirely on this phone")
                .font(.system(size: 11, design: .monospaced))
                .kerning(1)
                .foregroundStyle(Lab.textSecondary)
        }
        .padding(.top, 8)
    }

    // ── 01 The Specimen ────────────────────────────────────────────────────

    private var specimenCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "01 · The Specimen")

                switch model.store.phase {
                case .idle:
                    Text(
                        "The machine needs its brain: \(Self.gigabytes(ModelManifest.totalBytes)) "
                            + "of verified model weights, downloaded once."
                    )
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
                    Button("Fetch the models") { showDownloadConsent = true }
                        .buttonStyle(PrimaryButtonStyle())

                case .downloading(let label, let done, let total, let eta):
                    LabProgressBar(fraction: Double(done) / Double(max(1, total)))
                    StatusLine(
                        kind: .neutral,
                        text:
                            "\(label) — \(Self.gigabytes(done)) / \(Self.gigabytes(total)) · \(eta)"
                    )
                    Button("Pause") { model.store.cancelDownload() }
                        .buttonStyle(GhostButtonStyle())

                case .verifying(let label):
                    LabProgressBar(fraction: 1)
                    StatusLine(kind: .neutral, text: "verifying \(label) (SHA-256)…")

                case .failed(let reason):
                    StatusLine(kind: .err, text: reason)
                    Button("Retry") { model.store.startDownload() }
                        .buttonStyle(PrimaryButtonStyle())

                case .ready:
                    engineRows
                }
            }
        }
    }

    @ViewBuilder private var engineRows: some View {
        switch model.engineState {
        case .notLoaded:
            StatusLine(
                kind: .ok,
                text: "weights cached · \(Self.gigabytes(model.store.cachedBytes)) on device")
            Button("Assemble the machine") { model.assembleEngine() }
                .buttonStyle(PrimaryButtonStyle())
            Button("Clear downloaded models") { model.store.clear() }
                .buttonStyle(GhostButtonStyle(tint: Lab.danger))

        case .loading(let stage):
            ProgressView()
                .tint(Lab.emerald)
            StatusLine(kind: .neutral, text: stage + "…")
            Text("First assembly takes a minute; the weights stream in tensor by tensor.")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Lab.textSecondary)

        case .ready:
            StatusLine(kind: .ok, text: "engine alive · whisper large-v3-turbo q8_0")
            StatusLine(
                kind: model.diarizerLoaded ? .ok : .warn,
                text: model.diarizerLoaded
                    ? "sortformer diarizer alive · 4 anonymous speaker lanes"
                    : "diarizer unavailable — transcripts will have no speakers")
            StatusLine(
                kind: model.denoiserLoaded ? .ok : .warn,
                text: model.denoiserLoaded
                    ? "fastenhancer denoiser alive"
                    : "denoiser unavailable — audio goes in raw")

        case .failed(let reason):
            StatusLine(kind: .err, text: reason)
            Button("Try again") { model.assembleEngine() }
                .buttonStyle(PrimaryButtonStyle())
        }
    }

    // ── 02 The Signal ──────────────────────────────────────────────────────

    private var signalCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "02 · The Signal")

                if model.recorder.isRecording {
                    LevelMeter(level: model.recorder.level)
                    StatusLine(
                        kind: .warn,
                        text: String(format: "recording · %.0f s", model.recorder.seconds))
                }

                HStack(spacing: 10) {
                    Button(model.recorder.isRecording ? "Stop" : "Record") {
                        model.toggleRecording()
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    .disabled(model.engineState != .ready || model.isBusy && !model.recorder.isRecording)

                    Button("Import audio") { showFileImporter = true }
                        .buttonStyle(GhostButtonStyle())
                        .disabled(model.recorder.isRecording || model.isBusy)
                }

                if model.input != .none {
                    StatusLine(
                        kind: .ok,
                        text: model.input.seconds.map {
                            String(format: "%@ · %.1f s", model.inputName, $0)
                        } ?? model.inputName)
                }

                optionsRows

                if case .running(let done, let total, let stage) = model.runState {
                    LabProgressBar(fraction: Double(done) / Double(max(1, total)))
                    StatusLine(kind: .neutral, text: "\(stage) · window \(done)/\(total)")
                    Button("Abort") { model.cancelRun() }
                        .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                } else if case .staging = model.runState {
                    StatusLine(kind: .neutral, text: "preparing audio…")
                } else {
                    Button("Transcribe") { model.transcribe() }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(
                            model.engineState != .ready || model.input == .none || model.isBusy
                                || model.recorder.isRecording)
                }

                if case .failed(let reason) = model.runState {
                    StatusLine(kind: .err, text: reason)
                }
            }
        }
    }

    private var optionsRows: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: $model.diarize) {
                optionLabel("Who spoke when (Sortformer)")
            }
            .disabled(!model.diarizerLoaded)
            Toggle(isOn: $model.denoise) {
                optionLabel("Denoise first (FastEnhancer)")
            }
            .disabled(!model.denoiserLoaded)
            Toggle(isOn: $model.wordTimestamps) {
                optionLabel("Word-level timestamps (DTW)")
            }
            HStack {
                optionLabel("Language")
                Spacer()
                Picker("Language", selection: $model.language) {
                    ForEach(LabModel.languages, id: \.code) { language in
                        Text(language.label).tag(language.code)
                    }
                }
                .pickerStyle(.menu)
            }
        }
        .toggleStyle(SwitchToggleStyle(tint: Lab.emerald))
        .disabled(model.isBusy)
    }

    private func optionLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 12, design: .monospaced))
            .foregroundStyle(Lab.textSecondary)
    }

    // ── 03 The Transcript ──────────────────────────────────────────────────

    private var transcriptCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "03 · The Transcript")

                if let result = model.result {
                    resultView(result)
                } else if case .running = model.runState, !model.liveSegments.isEmpty {
                    liveView
                } else {
                    Text("The transcript materializes here, window by window, as the model listens.")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                }
            }
        }
    }

    /// Segments streamed live while the decode is still running.
    private var liveView: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(model.liveSegments) { segment in
                segmentRow(segment)
            }
            HStack(spacing: 6) {
                ProgressView().tint(Lab.emerald).scaleEffect(0.7)
                Text("still listening…")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
            }
        }
    }

    private func resultView(_ result: Transcription) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            if !result.speakerSegments.isEmpty {
                ForEach(result.speakerSegments) { run in
                    speakerRow(run)
                }
            } else {
                ForEach(result.segments) { segment in
                    segmentRow(segment)
                }
            }

            if result.droppedWindows > 0 {
                StatusLine(
                    kind: .warn,
                    text: "\(result.droppedWindows) window(s) dropped without transcript — "
                        + "a real content gap, not silence")
            }

            resultMeta(result)

            HStack(spacing: 10) {
                Picker("Format", selection: $exportFormat) {
                    ForEach(TranscriptFormat.allCases) { format in
                        Text(format.rawValue).tag(format)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 140)

                ShareLink(
                    item: TranscriptFile(
                        content: exportFormat == .text
                            ? TranscriptExport.text(from: result)
                            : TranscriptExport.srt(from: result),
                        baseName: "frankenwhisper-transcript",
                        format: exportFormat),
                    preview: SharePreview("Transcript")
                ) {
                    Text("Share")
                }
                .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
            }
        }
    }

    private func speakerRow(_ run: SpeakerRun) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Text(run.speaker ?? "UNKNOWN")
                    .font(.system(size: 10, weight: .black, design: .monospaced))
                    .kerning(1.2)
                    .foregroundStyle(Lab.speakerColor(run.speaker))
                if let start = run.startSec {
                    Text(Self.clock(start))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                }
            }
            Text(run.text)
                .font(.system(size: 14))
                .foregroundStyle(Lab.textPrimary)
                .textSelection(.enabled)
        }
    }

    private func segmentRow(_ segment: TranscriptSegment) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(segment.startSec.map(Self.clock) ?? "—")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Lab.textSecondary)
                .frame(width: 44, alignment: .trailing)
            Text(segment.text.trimmingCharacters(in: .whitespaces))
                .font(.system(size: 14))
                .foregroundStyle(Lab.textPrimary)
                .textSelection(.enabled)
        }
    }

    private func resultMeta(_ result: Transcription) -> some View {
        let rtf = result.audioSec > 0 ? model.wallSeconds / result.audioSec : 0
        return VStack(alignment: .leading, spacing: 4) {
            StatusLine(
                kind: .neutral,
                text: String(
                    format: "%.1f s of audio · %.1f s on device · RTF %.2f · language %@",
                    result.audioSec, model.wallSeconds, rtf, result.language ?? "?"))
            if !result.turns.isEmpty {
                StatusLine(
                    kind: .neutral,
                    text: "\(Set(result.turns.compactMap(\.speakerRef)).count) speaker lane(s) active "
                        + "· 4-lane capacity, true count unknown")
            }
        }
    }

    // ── Footer ─────────────────────────────────────────────────────────────

    private var footer: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Nothing leaves this device. No accounts, no telemetry, no cloud.")
            Text("franken_whisper — the same pure-Rust engine as the CLI and the browser demo.")
        }
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(Lab.textSecondary.opacity(0.7))
        .padding(.bottom, 24)
    }

    // ── Formatting ─────────────────────────────────────────────────────────

    private static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
    }

    private static func clock(_ seconds: Double) -> String {
        let clamped = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", clamped / 60, clamped % 60)
    }
}
