// The whole main screen: the website playground's three numbered stations as
// a native scrolling page — 01 The Specimen (models + engine), 02 The Signal
// (microphone or file), 03 The Transcript (live windows, speakers, export).

import SwiftUI
import UniformTypeIdentifiers

struct LabView: View {
    @State private var model = LabModel()
    @State private var showDownloadConsent = false
    @State private var showClearConfirmation = false
    @State private var showFileImporter = false
    @State private var exportFormat: TranscriptFormat = .html
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
            .scrollIndicators(.hidden)
            .scrollDismissesKeyboard(.interactively)
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
        .confirmationDialog(
            "Clear downloaded models?", isPresented: $showClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear \(Self.gigabytes(ModelManifest.totalBytes))", role: .destructive) {
                model.store.clear()
            }
            Button("Keep models", role: .cancel) {}
        } message: {
            Text(
                "This removes the verified Whisper, speaker, and denoiser models. "
                    + "You will need to download them again before transcribing.")
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio, .mpeg4Audio, .mp3, .wav],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                model.acceptFile(url: url)
            case .failure(let error):
                model.reportFileImportError(error)
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
        .sensoryFeedback(.success, trigger: model.result?.transcript)
        .sensoryFeedback(.impact(weight: .medium), trigger: model.recorder.isRecording)
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
        .accessibilityElement(children: .combine)
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
            if Self.lowMemoryDevice {
                StatusLine(
                    kind: .warn,
                    text: "This device reports under 6 GB of memory. The on-device engine may be unloaded under pressure."
                )
            }
            Button("Clear downloaded models") { showClearConfirmation = true }
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
                        text: "recording · \(Self.clock(model.recorder.seconds)) · tap Stop when finished")
                    Text(
                        "Speak naturally and keep the level meter moving. Audio stays in memory on this phone."
                    )
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
                }

                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 10) { inputButtons }
                    VStack(alignment: .leading, spacing: 10) { inputButtons }
                }

                if model.input != .none {
                    StatusLine(
                        kind: .ok,
                        text: model.input.seconds.map {
                            String(format: "%@ · %.1f s", model.inputName, $0)
                        } ?? model.inputName)
                }

                if model.isImporting {
                    HStack(spacing: 8) {
                        ProgressView().tint(Lab.emerald)
                        StatusLine(kind: .neutral, text: "reading audio file…")
                    }
                }

                optionsRows

                if case .running(let done, let total, let stage) = model.runState {
                    LabProgressBar(fraction: Double(done) / Double(max(1, total)))
                    TimelineView(.periodic(from: .now, by: 1)) { context in
                        StatusLine(
                            kind: .neutral,
                            text: "\(stage) · window \(done)/\(total) · \(Self.clock(model.elapsed(at: context.date))) elapsed"
                        )
                    }
                    Button("Abort") { model.cancelRun() }
                        .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                } else if case .staging = model.runState {
                    TimelineView(.periodic(from: .now, by: 1)) { context in
                        StatusLine(
                            kind: .neutral,
                            text: "preparing audio · \(Self.clock(model.elapsed(at: context.date))) elapsed"
                        )
                    }
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
            .animation(.snappy(duration: 0.3), value: model.runState)
            .animation(.snappy(duration: 0.3), value: model.recorder.isRecording)
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

            // The website's speaker-names field: feeds the decoding prompt so
            // names come out spelled right, then labels the detected voices
            // in speaking order. Editable per lane after the run too.
            TextField(
                "Speaker names — e.g. Jeff Emanuel (host), Dr. Sarah Chen (guest)",
                text: $model.speakerNamesRaw,
                axis: .vertical
            )
            .labTextField()
            .textInputAutocapitalization(.words)
            .autocorrectionDisabled()
            .submitLabel(.done)
            Text(
                model.diarize && model.diarizerLoaded
                    ? "Names help spelling and label the detected voices in speaking order — they assign labels, they don't identify anyone."
                    : "Names help the model spell them correctly in the transcript."
            )
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(Lab.textSecondary.opacity(0.8))
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

    /// Segments streamed live while the decode is still running. Indexed IDs:
    /// repeated identical phrases can produce identical (time, text) tuples,
    /// and live times need the trimmed-leading-silence offset added back
    /// (the final result already has it; fw_ios.h documents the split).
    private var liveView: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(model.liveSegments.enumerated()), id: \.offset) { _, segment in
                segmentRow(segment, timeOffset: model.liveOffsetSec)
            }
            HStack(spacing: 6) {
                ProgressView().tint(Lab.emerald).scaleEffect(0.7)
                Text("still listening…")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
            }
        }
        .animation(.easeOut(duration: 0.25), value: model.liveSegments)
    }

    private func resultView(_ result: Transcription) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            if result.segments.isEmpty {
                StatusLine(
                    kind: .warn,
                    text: "No speech was detected. Try a louder recording or import a clearer clip."
                )
            } else if !result.speakerSegments.isEmpty {
                ForEach(Array(result.speakerSegments.enumerated()), id: \.offset) { _, run in
                    speakerRow(run)
                }
            } else {
                ForEach(Array(result.segments.enumerated()), id: \.offset) { _, segment in
                    segmentRow(segment)
                }
            }

            if !model.detectedSpeakers.isEmpty {
                speakerNameEditor
            }

            if let reason = result.diarizationError {
                StatusLine(
                    kind: .warn,
                    text: "speakers unavailable for this run — \(reason)")
            }

            if result.droppedWindows > 0 {
                StatusLine(
                    kind: .warn,
                    text: "\(result.droppedWindows) window(s) dropped without transcript — "
                        + "a real content gap, not silence")
            }

            resultMeta(result)

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { exportControls(result) }
                VStack(alignment: .leading, spacing: 10) { exportControls(result) }
            }
        }
    }

    /// A styled web page and GitHub Markdown are the headline exports (they
    /// carry the speaker names and the lab look); SRT and JSON are niche
    /// interchange formats and live behind "More".
    @ViewBuilder private func exportControls(_ result: Transcription) -> some View {
        Picker("Format", selection: $exportFormat) {
            ForEach(TranscriptFormat.primary) { format in
                Text(format.rawValue).tag(format)
            }
        }
        .pickerStyle(.segmented)
        .frame(width: 190)

        ShareLink(
            item: TranscriptFile(
                content: TranscriptExport.content(
                    exportFormat, from: result, context: model.exportContext),
                baseName: "frankenwhisper-transcript",
                format: exportFormat),
            preview: SharePreview("Transcript (\(exportFormat.rawValue))")
        ) {
            Label("Share", systemImage: "square.and.arrow.up")
        }
        .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
        .disabled(result.segments.isEmpty)

        Menu {
            ForEach(TranscriptFormat.niche) { format in
                ShareLink(
                    item: TranscriptFile(
                        content: TranscriptExport.content(
                            format, from: result, context: model.exportContext),
                        baseName: "frankenwhisper-transcript",
                        format: format),
                    preview: SharePreview("Transcript (\(format.rawValue))")
                ) {
                    Label(format.menuLabel, systemImage: "doc.text")
                }
            }
        } label: {
            Label("More", systemImage: "ellipsis.circle")
        }
        .buttonStyle(GhostButtonStyle())
        .disabled(result.segments.isEmpty)
    }

    /// One TextField per detected lane, in speaking order — rename the
    /// voices after the fact without re-running anything.
    private var speakerNameEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("NAME THE VOICES")
                .font(.system(size: 10, weight: .black, design: .monospaced))
                .kerning(1.5)
                .foregroundStyle(Lab.textSecondary)
            ForEach(model.detectedSpeakers, id: \.self) { lane in
                HStack(spacing: 8) {
                    Circle()
                        .fill(Lab.speakerColor(lane))
                        .frame(width: 8, height: 8)
                    Text(lane)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                        .frame(width: 90, alignment: .leading)
                    TextField(
                        "name this voice",
                        text: Binding(
                            get: { model.speakerNameMap[lane] ?? "" },
                            set: { model.setSpeakerName($0, for: lane) })
                    )
                    .labTextField()
                    .textInputAutocapitalization(.words)
                    .autocorrectionDisabled()
                    .submitLabel(.done)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func speakerRow(_ run: SpeakerRun) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Text(model.displaySpeaker(run.speaker))
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

    private func segmentRow(_ segment: TranscriptSegment, timeOffset: Double = 0) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(segment.startSec.map { Self.clock($0 + timeOffset) } ?? "—")
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
                // Turns' speaker_ref can be absent (anonymous lanes); the
                // projected speaker runs always carry the display labels.
                let lanes = Set(result.turns.compactMap(\.speakerRef))
                let speakers =
                    lanes.isEmpty
                    ? Set(result.speakerSegments.compactMap(\.speaker)) : lanes
                StatusLine(
                    kind: .neutral,
                    text: "\(speakers.count) speaker lane(s) active · labels separate voices, not identities")
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

    @ViewBuilder private var inputButtons: some View {
        HStack(spacing: 16) {
            RecordButton(isRecording: model.recorder.isRecording) {
                model.toggleRecording()
            }
            .disabled(
                model.engineState != .ready
                    || (model.isBusy && !model.recorder.isRecording))
            .opacity(model.engineState == .ready ? 1 : 0.35)

            Text(model.recorder.isRecording ? "listening…" : "record")
                .font(.system(size: 11, weight: .black, design: .monospaced))
                .kerning(1.5)
                .textCase(.uppercase)
                .foregroundStyle(model.recorder.isRecording ? Lab.danger : Lab.textSecondary)
        }

        Button {
            showFileImporter = true
        } label: {
            Label("Import audio", systemImage: "waveform.badge.plus")
        }
        .buttonStyle(GhostButtonStyle())
        .disabled(model.recorder.isRecording || model.isBusy || model.engineState != .ready)
    }

    // ── Formatting ─────────────────────────────────────────────────────────

    private static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
    }

    private static func clock(_ seconds: Double) -> String {
        let clamped = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", clamped / 60, clamped % 60)
    }

    private static var lowMemoryDevice: Bool {
        ProcessInfo.processInfo.physicalMemory < 6 * 1024 * 1024 * 1024
    }
}
