// The whole main screen: the website playground plus the systemwide local
// dictation lane — 01 models, 02 batch input, 03 live dictation, 04 result.

import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

private enum LabTextEntry: Hashable {
    case speakerNames
    case speakerLane(String)
}

private struct LabTextEntryFramePreferenceKey: PreferenceKey {
    static let defaultValue: [LabTextEntry: CGRect] = [:]

    static func reduce(
        value: inout [LabTextEntry: CGRect],
        nextValue: () -> [LabTextEntry: CGRect]
    ) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}

private extension View {
    func reportLabTextEntryFrame(_ entry: LabTextEntry) -> some View {
        background {
            GeometryReader { proxy in
                Color.clear.preference(
                    key: LabTextEntryFramePreferenceKey.self,
                    value: [entry: proxy.frame(in: .named("lab-text-entry-space"))]
                )
            }
        }
    }
}

struct LabView: View {
    @AppStorage(LabAppearance.storageKey) private var appearance = LabAppearance.dark.rawValue
    private enum Destination: String, CaseIterable, Identifiable {
        case transcribe = "Transcribe"
        case live = "Live"
        case result = "Result"
        case models = "Models"

        var id: Self { self }
    }

    @State private var model = LabModel()
    @State private var destination: Destination = .transcribe
    @State private var showDownloadConsent = false
    @State private var showClearConfirmation = false
    @State private var showFileImporter = false
    @State private var pickedVideoItem: PhotosPickerItem?
    @State private var showSubtitleStudio = false
    @State private var showHistory = false
    @State private var exportFormat: TranscriptFormat = .html
    @State private var textEntryFrames: [LabTextEntry: CGRect] = [:]
    @FocusState private var focusedTextEntry: LabTextEntry?
    @Environment(\.scenePhase) private var scenePhase

    init() {
        let requested = ProcessInfo.processInfo.environment["FW_INITIAL_DESTINATION"]
        _destination = State(
            initialValue: Destination(rawValue: requested ?? "") ?? .transcribe
        )
    }

    private var profilingRequested: Bool {
        ProcessInfo.processInfo.environment["FW_IOS_PROFILE"] == "1"
    }

    var body: some View {
        commandModifiers(
            continuityModifiers(
                lifecycleModifiers(
                    importModifiers(
                        presentationModifiers(laboratoryCanvas)
                    )
                )
            )
        )
    }

    private var laboratoryCanvas: some View {
        ZStack {
            LaboratoryBackground()
            GeometryReader { geometry in
                if usesDashboardLayout(width: geometry.size.width) {
                    HStack(alignment: .top, spacing: 24) {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 22) {
                                header
                                specimenCard
#if !targetEnvironment(macCatalyst)
                                dictationCard
#endif
                                footer
                            }
                            .padding(.vertical, 2)
                        }
                        .scrollIndicators(.hidden)
                        .defaultScrollAnchor(.top)
                        .frame(width: min(410, geometry.size.width * 0.38))

                        ScrollView {
                            VStack(alignment: .leading, spacing: 22) {
                                signalCard
                                transcriptCard
                            }
                            .padding(.vertical, 2)
                        }
                        .scrollIndicators(.hidden)
                        .defaultScrollAnchor(.top)
                        .frame(maxWidth: .infinity)
                    }
                    .padding(24)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 22) {
                            header
                            destinationPicker
                            compactWorkspace
                            footer
                        }
                        .padding(16)
                        .frame(maxWidth: 720)
                        .frame(maxWidth: .infinity)
                    }
                    .scrollIndicators(.hidden)
                    .scrollDismissesKeyboard(.interactively)
                }
            }
            if model.keyboardHandoffVisible {
                keyboardHandoffOverlay
                    .transition(.opacity.combined(with: .scale(scale: 0.97)))
            }
        }
    }

    private func presentationModifiers<Content: View>(_ content: Content) -> some View {
        content
        .coordinateSpace(name: "lab-text-entry-space")
        .onPreferenceChange(LabTextEntryFramePreferenceKey.self) { frames in
            textEntryFrames = frames
        }
        .simultaneousGesture(
            SpatialTapGesture(coordinateSpace: .named("lab-text-entry-space"))
                .onEnded { tap in
                    guard focusedTextEntry != nil else { return }
                    let tappedAField = textEntryFrames.values.contains { frame in
                        frame.contains(tap.location)
                    }
                    if !tappedAField { focusedTextEntry = nil }
                }
        )
        .tint(Lab.emerald)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    focusedTextEntry = nil
                }
                .font(.system(size: Lab.typeSize(13), weight: .semibold))
            }
        }
        .preferredColorScheme((LabAppearance(rawValue: appearance) ?? .dark).colorScheme)
        .alert(
            "Something snapped", isPresented: .init(
                get: { model.lastError != nil },
                set: { if !$0 { model.lastError = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.lastError ?? "")
        }
        .sheet(isPresented: $showSubtitleStudio) {
            if let video = model.videoInput, let result = model.result {
                SubtitleStudio(
                    video: video,
                    result: result,
                    speakerNames: model.speakerNameMap
                )
            }
        }
        .sheet(isPresented: $showHistory) {
            TranscriptHistorySheet(history: model.history)
        }
        .onChange(of: showSubtitleStudio) { _, isPresented in
            // The studio's AVPlayer/exporter still owns the current movie.
            // A queued share can safely replace it only after the sheet closes.
            if !isPresented { consumeStagedMedia() }
        }
    }

    private func importModifiers<Content: View>(_ content: Content) -> some View {
        content
        .confirmationDialog(
            "Download the models?", isPresented: $showDownloadConsent, titleVisibility: .visible
        ) {
            Button("Download \(Self.gigabytes(ModelManifest.totalBytes))") {
                model.store.startDownload()
            }
            Button("Not now", role: .cancel) {}
        } message: {
            Text(
                "Whisper large-v3-turbo (q8), a 74 MB multilingual tiny model for realtime "
                    + "dictation, the Sortformer speaker diarizer, and the "
                    + "FastEnhancer denoiser — downloaded once over your connection, verified "
                    + "by SHA-256, and stored on this device. Everything afterwards runs offline.")
        }
        .confirmationDialog(
            "Clear downloaded models?", isPresented: $showClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear \(Self.gigabytes(ModelManifest.totalBytes))", role: .destructive) {
                model.clearModels()
            }
            Button("Keep models", role: .cancel) {}
        } message: {
            Text(
                "This removes the verified Whisper, speaker, and denoiser models. "
                    + "You will need to download them again before transcribing.")
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio, .movie, .mpeg4Audio, .mp3, .wav],
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
        .onChange(of: pickedVideoItem) { _, item in
            guard let item else { return }
            Task {
                defer { pickedVideoItem = nil }
                do {
                    guard let picked = try await item.loadTransferable(type: PickedVideo.self)
                    else {
                        throw CocoaError(.fileReadUnknown)
                    }
                    model.acceptPickedVideo(picked)
                } catch {
                    model.reportVideoPickerError(error)
                }
            }
        }
    }

    private func lifecycleModifiers<Content: View>(_ content: Content) -> some View {
        content
        .onReceive(
            NotificationCenter.default.publisher(
                for: UIApplication.didReceiveMemoryWarningNotification)
        ) { _ in
            model.unloadEngineForMemoryPressure()
        }
        .onAppear {
            if !profilingRequested {
                model.prepareEngines()
                consumeRequestedAction()
                consumeStagedMedia()
            }
        }
        .task {
            if profilingRequested { await model.runProfilingBenchmarkIfRequested() }
        }
        .onOpenURL { url in
            if url.scheme?.lowercased() == "frankenwhisper",
               url.host?.lowercased() == "cancel-run"
            {
                model.cancelRun()
                return
            }
            if url.scheme?.lowercased() == "frankenwhisper",
               url.host?.lowercased() == "end-live"
            {
                model.endLiveDictationSession()
                return
            }
            if url.scheme?.lowercased() == "frankenwhisper",
               url.host?.lowercased() == "new"
            {
                destination = .transcribe
                consumeStagedMedia()
                return
            }
            if url.scheme?.lowercased() == "frankenwhisper",
               url.host?.lowercased() == "dictate"
            {
                destination = .live
            }
            withAnimation(.easeOut(duration: 0.18)) {
                model.handleKeyboardURL(url)
            }
        }
    }

    private func continuityModifiers<Content: View>(_ content: Content) -> some View {
        content
        .onChange(of: model.store.phase) { _, phase in
            if phase == .ready, scenePhase == .active, !profilingRequested {
                model.prepareEngines()
            }
        }
        .onChange(of: model.runState) { _, state in
            if case .done = state {
                withAnimation(.snappy) { destination = .result }
            }
        }
        .onChange(of: model.canAcceptInput) { _, canAccept in
            // A share received during a run stays securely staged rather than
            // replacing that run's source. Consume it as soon as every input
            // consumer has released the old media.
            if canAccept { consumeStagedMedia() }
        }
        .onChange(of: scenePhase) { _, phase in
            // Keep the large local model resident across an ordinary app
            // switch. Unloading here made every keyboard handoff pay the full
            // hydration cost. Real memory warnings still unload it above.
            if phase == .active, !profilingRequested {
                model.prepareEngines()
                consumeRequestedAction()
                consumeStagedMedia()
            }
        }
        .sensoryFeedback(.success, trigger: model.result?.transcript)
        .sensoryFeedback(.success, trigger: model.liveLastPhrase)
        .sensoryFeedback(.impact(weight: .medium), trigger: model.recorder.isRecording)
        .userActivity("com.frankenwhisper.workspace") { activity in
            activity.title = "FrankenWhisper \(destination.rawValue)"
            activity.isEligibleForHandoff = true
            activity.userInfo = ["route": destination.rawValue]
        }
        .onContinueUserActivity("com.frankenwhisper.workspace") { activity in
            guard let rawValue = activity.userInfo?["route"] as? String,
                  let restored = Destination(rawValue: rawValue)
            else { return }
            destination = restored
        }
    }

    private func commandModifiers<Content: View>(_ content: Content) -> some View {
        content
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            return model.acceptFile(url: url)
        }
        .focusedSceneValue(
            \.whisperCommands,
            WhisperCommandActions(
                importFile: { showFileImporter = true },
                toggleRecording: { model.toggleRecording() },
                transcribe: {
                    focusedTextEntry = nil
                    model.transcribe()
                },
                stop: { model.cancelRun() },
                canRecord: model.recorder.isRecording
                    || (!model.isBusy && !model.isLiveDictationActive),
                canTranscribe: model.engineState == .ready && model.input != .none && !model.isBusy,
                canStop: {
                    if case .running = model.runState { return true }
                    if case .staging = model.runState { return true }
                    return false
                }()
            )
        )
    }

    private func usesDashboardLayout(width: CGFloat) -> Bool {
#if targetEnvironment(macCatalyst)
        return width >= 700
#else
        return UIDevice.current.userInterfaceIdiom == .pad ? width >= 700 : width >= 940
#endif
    }

    private var destinationPicker: some View {
        Picker("Workspace", selection: $destination) {
            ForEach(availableDestinations) { destination in
                Text(destination.rawValue).tag(destination)
            }
        }
        .pickerStyle(.segmented)
        .accessibilityLabel("FrankenWhisper workspace")
    }

    private var availableDestinations: [Destination] {
#if targetEnvironment(macCatalyst)
        [.transcribe, .result, .models]
#else
        Destination.allCases
#endif
    }

    @ViewBuilder
    private var compactWorkspace: some View {
        switch destination {
        case .transcribe:
            if model.store.phase != .ready { specimenCard }
            signalCard
        case .live:
#if targetEnvironment(macCatalyst)
            signalCard
#else
            dictationCard
#endif
        case .result:
            transcriptCard
        case .models:
            specimenCard
        }
    }

    private func consumeStagedMedia() {
        guard model.canAcceptInput, !showSubtitleStudio else { return }
        guard let staged = FrankenWhisperSharedStore.consumeStagedMediaURL() else { return }
        // The share extension copied this into the App Group solely for the
        // handoff. LabModel owns the asynchronous read/copy lifecycle and must
        // remove that private staging file afterward, including on failure.
        model.acceptFile(url: staged, removeSourceAfterImport: true)
    }

    private func consumeRequestedAction() {
        switch FrankenWhisperSharedStore.consumeRequestedAction() {
        case .transcribe:
            destination = .transcribe
        case .live:
            destination = .live
        case .none:
            break
        }
    }

    // ── 03 Live Dictation ─────────────────────────────────────────────────

    private var keyboardHandoffOverlay: some View {
        ZStack {
            Lab.background.opacity(0.96).ignoresSafeArea()
            VStack(spacing: 18) {
                Text("FRANKENWHISPER KEYBOARD")
                    .font(.system(size: Lab.typeSize(12), weight: .bold, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Lab.emerald)

                switch model.liveDictationState {
                case .starting(let stage):
                    ProgressView().tint(Lab.emerald).scaleEffect(1.4)
                    Text(stage)
                        .font(.title2.bold())
                        .multilineTextAlignment(.center)
                case .listening:
                    Image(systemName: "waveform.circle.fill")
                        .font(.system(size: Lab.typeSize(58)))
                        .foregroundStyle(Lab.emerald)
                        .symbolEffect(.pulse)
                    Text("Listening")
                        .font(.title2.bold())
                        .multilineTextAlignment(.center)
                    LevelMeter(level: model.recorder.level)
                        .frame(maxWidth: 280)
                    Text("Return to your app and speak.")
                    .font(.system(size: Lab.typeSize(14)))
                    .multilineTextAlignment(.center)
                    .foregroundStyle(Lab.textSecondary)
                case .armed:
                    Image(systemName: "keyboard.badge.ellipsis")
                        .font(.system(size: Lab.typeSize(52)))
                        .foregroundStyle(Lab.emerald)
                    Text("Keyboard ready · \(model.liveSessionMinutesRemaining)m")
                        .font(.title2.bold())
                case .finishing:
                    ProgressView().tint(Lab.emerald).scaleEffect(1.4)
                    Text("Finishing on device…")
                        .font(.title2.bold())
                case .failed(let reason):
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: Lab.typeSize(44)))
                        .foregroundStyle(Lab.danger)
                    Text(reason)
                        .font(.system(size: Lab.typeSize(14)))
                        .multilineTextAlignment(.center)
                        .foregroundStyle(Lab.textSecondary)
                case .idle:
                    ProgressView().tint(Lab.emerald).scaleEffect(1.4)
                    Text(engineHandoffStatus)
                        .font(.title2.bold())
                        .multilineTextAlignment(.center)
                }

                Text("ON-DEVICE · NOTHING UPLOADED")
                    .font(.system(size: Lab.typeSize(11), design: .monospaced))
                    .multilineTextAlignment(.center)
                    .foregroundStyle(Lab.textSecondary.opacity(0.8))

                if model.liveDictationState == .listening {
                    Button("Stop dictation") { model.stopLiveDictation() }
                        .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                } else if case .failed = model.liveDictationState {
                    Button("Try again") { model.retryKeyboardDictation() }
                        .buttonStyle(PrimaryButtonStyle())
                    Button("Back to FrankenWhisper") {
                        withAnimation(.easeOut(duration: 0.18)) {
                            model.dismissKeyboardHandoff()
                        }
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                } else if model.liveDictationState != .finishing {
                    Button("Cancel") {
                        withAnimation(.easeOut(duration: 0.18)) {
                            model.dismissKeyboardHandoff()
                        }
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.textSecondary))
                }
            }
            .padding(28)
            .frame(maxWidth: 420)
        }
        .zIndex(10)
    }

    private var engineHandoffStatus: String {
        switch model.liveEngineState {
        case .notLoaded: "Preparing the realtime speech engine…"
        case .loading(let stage): stage
        case .ready: "Preparing microphone access…"
        case .failed(let reason): reason
        }
    }

    private var dictationCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "03 · Live Dictation")
                Text(
                    "Fast on-device dictation. Activate once, then Start and Finish stay in the keyboard "
                        + "for an hour. Full transcription below keeps the larger accuracy model."
                )
                .font(.system(size: Lab.typeSize(12), design: .monospaced))
                .foregroundStyle(Lab.textSecondary)

                switch model.liveDictationState {
                case .idle:
                    Button("Enable 1-hour keyboard session") { model.startLiveDictation() }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(model.liveEngineState != .ready || model.isBusy)
                case .starting(let stage):
                    HStack(spacing: 8) {
                        ProgressView().tint(Lab.emerald)
                        StatusLine(kind: .neutral, text: stage)
                    }
                case .listening:
                    LevelMeter(level: model.recorder.level)
                    StatusLine(
                        kind: .ok,
                        text: model.liveQueuedUtterances > 0
                            ? "listening · \(model.liveQueuedUtterances) phrase(s) decoding locally"
                            : "listening · pause briefly to commit a phrase")
                    Button("Stop dictation") { model.stopLiveDictation() }
                        .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                case .armed:
                    StatusLine(
                        kind: .ok,
                        text: "keyboard ready · \(model.liveSessionMinutesRemaining) min left")
                    HStack(spacing: 10) {
                        Button("Start dictating") { model.startLiveDictation() }
                            .buttonStyle(PrimaryButtonStyle())
                        Button("End session") { model.endLiveDictationSession() }
                            .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                    }
                case .finishing:
                    HStack(spacing: 8) {
                        ProgressView().tint(Lab.emerald)
                        StatusLine(
                            kind: .neutral,
                            text: "finishing \(max(1, model.liveQueuedUtterances)) phrase(s) on device…")
                    }
                case .failed(let reason):
                    StatusLine(kind: .err, text: reason)
                    Button("Try live dictation again") { model.startLiveDictation() }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(model.liveEngineState != .ready)
                }

                if !model.liveDictationText.isEmpty {
                    Text(model.liveDictationText)
                        .font(.system(size: Lab.typeSize(14)))
                        .foregroundStyle(Lab.textPrimary)
                        .textSelection(.enabled)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 8))
                    HStack(spacing: 10) {
                        Button {
                            UIPasteboard.general.string = model.liveDictationText
                        } label: {
                            Label("Copy", systemImage: "doc.on.doc")
                        }
                        .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                        Button("Clear") { model.clearLiveDictationText() }
                            .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                            .disabled(model.isLiveDictationActive)
                    }
                }

                Text(
                    "SETUP  Add FrankenWhisper in iOS Keyboard settings and enable Full Access for its local handoff. Nothing is uploaded."
                )
                .font(.system(size: Lab.typeSize(10), design: .monospaced))
                .foregroundStyle(Lab.textSecondary.opacity(0.8))

                Button {
                    guard let settings = URL(string: UIApplication.openSettingsURLString) else {
                        return
                    }
                    UIApplication.shared.open(settings)
                } label: {
                    Label("Open FrankenWhisper settings", systemImage: "gearshape")
                }
                .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
            }
            .animation(.snappy(duration: 0.25), value: model.liveDictationState)
        }
    }

    // ── Header ─────────────────────────────────────────────────────────────

    private var header: some View {
        HStack(spacing: 13) {
            MonsterStatusMark(mood: monsterMood, instrument: .hearing, accent: Lab.cyan)
                .frame(width: 54, height: 54)
            VStack(alignment: .leading, spacing: 5) {
                FrankenWordmark(
                    productInitial: "W",
                    productRemainder: "HISPER",
                    fullName: "FrankenWhisper"
                )
                Text("IT_HEARS // private speech observatory")
                    .font(.system(size: Lab.typeSize(11), design: .monospaced))
                    .kerning(1)
                    .foregroundStyle(Lab.textSecondary)
            }
            Spacer(minLength: 0)
            Button { showHistory = true } label: {
                ZStack(alignment: .topTrailing) {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.system(size: Lab.typeSize(15), weight: .bold))
                        .frame(width: 44, height: 44)
                        .background(Lab.panelStrong, in: Circle())
                        .overlay(Circle().stroke(Lab.stroke))
                    if !model.history.entries.isEmpty {
                        Text("\(model.history.entries.count)")
                            .font(.system(size: Lab.typeSize(8), weight: .black, design: .monospaced))
                            .foregroundStyle(Lab.background)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Lab.cyan, in: Capsule())
                    }
                }
            }
            .buttonStyle(.plain)
            .foregroundStyle(Lab.cyan)
            .accessibilityIdentifier("transcript-history-button")
            .accessibilityLabel("Recent transcripts")
            .accessibilityValue(
                model.history.entries.isEmpty
                    ? "Empty"
                    : "\(model.history.entries.count) saved transcript\(model.history.entries.count == 1 ? "" : "s")"
            )
            LabAppearanceButton(selection: $appearance)
        }
        .padding(.top, 8)
        .accessibilityElement(children: .contain)
    }

    private var monsterMood: MonsterMood {
        if model.lastError != nil { return .error }
        if case .failed = model.engineState { return .error }
        if case .failed = model.liveEngineState { return .error }
        if model.isLiveDictationActive || model.recorder.isRecording { return .working }
        if case .running = model.runState { return .working }
        if case .staging = model.runState { return .working }
        if case .loading = model.engineState { return .waking }
        if case .loading = model.liveEngineState { return .waking }
        if model.result != nil { return .success }
        return .idle
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
                    .font(.system(size: Lab.typeSize(13), design: .monospaced))
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
            if model.enginePausedForMemoryPressure {
                StatusLine(
                    kind: .warn,
                    text: "engine paused after memory pressure to protect this app")
                Button("Reload engine") { model.assembleEngine() }
                    .buttonStyle(PrimaryButtonStyle())
            } else {
                ProgressView()
                    .tint(Lab.emerald)
                StatusLine(kind: .neutral, text: "starting the engine automatically…")
            }
            if Self.lowMemoryDevice {
                StatusLine(
                    kind: .warn,
                    text: "This device reports under 6 GB of memory. The on-device engine may be unloaded under pressure."
                )
            }

        case .loading(let stage):
            ProgressView()
                .tint(Lab.emerald)
            StatusLine(kind: .neutral, text: stage + "…")
            Text("First assembly takes a minute; the weights stream in tensor by tensor.")
                .font(.system(size: Lab.typeSize(11), design: .monospaced))
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

        if !model.isBusy {
            Button("Clear downloaded models") { showClearConfirmation = true }
                .buttonStyle(GhostButtonStyle(tint: Lab.danger))
        }
    }

    // ── 02 The Signal ──────────────────────────────────────────────────────

    private var signalCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "02 · The Signal")

                if model.recorder.isRecording && !model.hasArmedDictationSession {
                    LevelMeter(level: model.recorder.level)
                    StatusLine(
                        kind: .warn,
                        text: "recording · \(Self.clock(model.recorder.seconds)) · tap Stop when finished")
                    Text(
                        "Speak naturally and keep the level meter moving. Audio stays in memory on this device."
                    )
                    .font(.system(size: Lab.typeSize(11), design: .monospaced))
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
                        StatusLine(kind: .neutral, text: "preparing private local media…")
                    }
                }

                optionsRows

                if case .running = model.runState {
                    WhisperObservatory(
                        state: model.runState,
                        segments: model.liveSegments,
                        started: model.runStarted,
                        estimatedFinishElapsed: model.estimatedFinishElapsed,
                        cancel: model.cancelRun
                    )
                } else if case .staging = model.runState {
                    WhisperObservatory(
                        state: model.runState,
                        segments: model.liveSegments,
                        started: model.runStarted,
                        estimatedFinishElapsed: model.estimatedFinishElapsed,
                        cancel: model.cancelRun
                    )
                } else {
                    Button("Transcribe") { model.transcribe() }
                        .buttonStyle(PrimaryButtonStyle())
                        .accessibilityIdentifier("fw.transcribe")
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
            .disabled(model.input.isVideo)
            if model.input.isVideo {
                StatusLine(
                    kind: .ok,
                    text: "word timing is automatic for video karaoke captions"
                )
            }
            VStack(alignment: .leading, spacing: 5) {
                optionLabel("Speech task")
                Picker("Speech task", selection: $model.translateToEnglish) {
                    Text("Transcribe").tag(false)
                    Text("Translate to English").tag(true)
                }
                .pickerStyle(.segmented)
                .accessibilityIdentifier("fw.translationTask")
                Text(
                    model.translateToEnglish
                        ? "Use the selected or auto-detected source language and produce English text on device."
                        : "Keep the transcript in the language that was spoken."
                )
                .font(.system(size: Lab.typeSize(10), design: .monospaced))
                .foregroundStyle(Lab.textSecondary.opacity(0.8))
            }
            ViewThatFits(in: .horizontal) {
                HStack {
                    optionLabel("Language")
                    Spacer()
                    languagePicker
                }
                VStack(alignment: .leading, spacing: 2) {
                    optionLabel("Language")
                    languagePicker
                }
            }

            // The website's speaker-names field: feeds the decoding prompt so
            // names come out spelled right, then labels the detected voices
            // in speaking order. Editable per lane after the run too.
            TextField(
                "Speaker names — e.g. Jeff Emanuel (host), Dr. Sarah Chen (guest)",
                text: $model.speakerNamesRaw,
                axis: .vertical
            )
            .focused($focusedTextEntry, equals: .speakerNames)
            .labTextField()
            .textInputAutocapitalization(.words)
            .autocorrectionDisabled()
            .submitLabel(.done)
            .reportLabTextEntryFrame(.speakerNames)
            Text(
                model.diarize && model.diarizerLoaded
                    ? "Names help spelling and label the detected voices in speaking order — they assign labels, they don't identify anyone."
                    : "Names help the model spell them correctly in the transcript."
            )
            .font(.system(size: Lab.typeSize(10), design: .monospaced))
            .foregroundStyle(Lab.textSecondary.opacity(0.8))
        }
        .toggleStyle(SwitchToggleStyle(tint: Lab.emerald))
        .disabled(model.isBusy)
    }

    private func optionLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: Lab.typeSize(12), design: .monospaced))
            .foregroundStyle(Lab.textSecondary)
    }

    private var languagePicker: some View {
        Picker("Language", selection: $model.language) {
            ForEach(LabModel.languages, id: \.code) { language in
                Text(language.label).tag(language.code)
            }
        }
        .pickerStyle(.menu)
        // Menu-style pickers can report an implausibly narrow ideal width at
        // accessibility sizes. Preserve the selected language as one compact
        // control; ViewThatFits moves it below the label when necessary.
        .fixedSize(horizontal: true, vertical: true)
    }

    // ── 03 The Transcript ──────────────────────────────────────────────────

    private var transcriptCard: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                LabLabel(text: "04 · The Transcript")

                if let result = model.result {
                    resultView(result)
                } else if case .running = model.runState, !model.liveSegments.isEmpty {
                    liveView
                } else {
                    Text("The transcript materializes here, window by window, as the model listens.")
                        .font(.system(size: Lab.typeSize(12), design: .monospaced))
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
                    .font(.system(size: Lab.typeSize(11), design: .monospaced))
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

            if let video = model.videoInput {
                videoSubtitleControls(video: video, result: result)
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { exportControls(result) }
                VStack(alignment: .leading, spacing: 10) { exportControls(result) }
            }
        }
    }

    private func videoSubtitleControls(video: VideoInput, result: Transcription) -> some View {
        let wordCount = SubtitleTimeline.makeCues(from: result.words).flatMap(\.words).count
        return VStack(alignment: .leading, spacing: 8) {
            Button {
                showSubtitleStudio = true
            } label: {
                Label("Style & burn karaoke subtitles", systemImage: "captions.bubble.fill")
            }
            .buttonStyle(PrimaryButtonStyle())
            .accessibilityIdentifier("fw.subtitleStudio")
            .disabled(wordCount == 0)

            Text(
                wordCount > 0
                    ? "\(wordCount) decoder-aligned words are ready to preview, style, render, save, or share."
                    : "No word alignment was returned, so FrankenWhisper will not fabricate karaoke timing."
            )
            .font(.system(size: Lab.typeSize(10), design: .monospaced))
            .foregroundStyle(wordCount > 0 ? Lab.textSecondary : Lab.danger)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Caption controls for \(video.name)")
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
                .font(.system(size: Lab.typeSize(10), weight: .black, design: .monospaced))
                .kerning(1.5)
                .foregroundStyle(Lab.textSecondary)
            ForEach(model.detectedSpeakers, id: \.self) { lane in
                HStack(spacing: 8) {
                    Circle()
                        .fill(Lab.speakerColor(lane))
                        .frame(width: 8, height: 8)
                    Text(lane)
                        .font(.system(size: Lab.typeSize(10), design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                        .frame(width: 90, alignment: .leading)
                    TextField(
                        "name this voice",
                        text: Binding(
                            get: { model.speakerNameMap[lane] ?? "" },
                            set: { model.setSpeakerName($0, for: lane) })
                    )
                    .focused($focusedTextEntry, equals: .speakerLane(lane))
                    .labTextField()
                    .textInputAutocapitalization(.words)
                    .autocorrectionDisabled()
                    .submitLabel(.done)
                    .reportLabTextEntryFrame(.speakerLane(lane))
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func speakerRow(_ run: SpeakerRun) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Text(model.displaySpeaker(run.speaker))
                    .font(.system(size: Lab.typeSize(10), weight: .black, design: .monospaced))
                    .kerning(1.2)
                    .foregroundStyle(Lab.speakerColor(run.speaker))
                if let start = run.startSec {
                    Text(Self.clock(start))
                        .font(.system(size: Lab.typeSize(10), design: .monospaced))
                        .foregroundStyle(Lab.textSecondary)
                }
            }
            Text(run.text)
                .font(.system(size: Lab.typeSize(14)))
                .foregroundStyle(Lab.textPrimary)
                .textSelection(.enabled)
        }
    }

    private func segmentRow(_ segment: TranscriptSegment, timeOffset: Double = 0) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(segment.startSec.map { Self.clock($0 + timeOffset) } ?? "—")
                .font(.system(size: Lab.typeSize(10), design: .monospaced))
                .foregroundStyle(Lab.textSecondary)
                .frame(width: 44, alignment: .trailing)
            Text(segment.text.trimmingCharacters(in: .whitespaces))
                .font(.system(size: Lab.typeSize(14)))
                .foregroundStyle(Lab.textPrimary)
                .textSelection(.enabled)
        }
    }

    private func resultMeta(_ result: Transcription) -> some View {
        let rtf = result.audioSec > 0 ? model.wallSeconds / result.audioSec : 0
        let languageSummary = model.resultWasTranslated
            ? "translated to English · source \(result.language ?? "?")"
            : "language \(result.language ?? "?")"
        return VStack(alignment: .leading, spacing: 4) {
            StatusLine(
                kind: .neutral,
                text: String(
                    format: "%.1f s of audio · %.1f s on device · RTF %.2f · %@",
                    result.audioSec, model.wallSeconds, rtf, languageSummary))
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
        VStack(spacing: 8) {
            Text("Nothing leaves this device. No accounts, no telemetry, no cloud.")
#if targetEnvironment(macCatalyst)
            Text("Audio and transcripts remain local to this Mac.")
#else
            Text("The optional keyboard makes no network requests and only reads locally committed dictation text.")
#endif
            Text("franken_whisper — the same pure-Rust engine as the CLI and the browser demo.")
            Text(
                "If you like this free app, please show your appreciation by trying out my paid skills site at [JeffreysSkills.md](https://jeffreys-skills.md)."
            )
            .font(.system(size: Lab.typeSize(10), design: .monospaced))
            .foregroundStyle(Lab.textSecondary.opacity(0.72))
            .tint(Lab.emerald.opacity(0.8))
            .multilineTextAlignment(.center)
            .frame(maxWidth: 320)
        }
        .font(.system(size: Lab.typeSize(10), design: .monospaced))
        .foregroundStyle(Lab.textSecondary.opacity(0.7))
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
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
                .font(.system(size: Lab.typeSize(11), weight: .black, design: .monospaced))
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

        PhotosPicker(selection: $pickedVideoItem, matching: .videos) {
            Label("Video", systemImage: "photo.on.rectangle.angled")
        }
        .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
        .accessibilityIdentifier("fw.videoPicker")
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
