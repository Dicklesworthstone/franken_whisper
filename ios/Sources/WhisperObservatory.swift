import SwiftUI

/// A cinematic, truthful view into the native transcription pipeline.
/// Animation supplies atmosphere only; every count and stage comes from the
/// Rust callbacks already used by `LabModel`.
struct WhisperObservatory: View {
    let state: LabModel.RunState
    let segments: [TranscriptSegment]
    let started: Date?
    let cancel: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.isLuminanceReduced) private var luminanceReduced

    private var run: (done: Int, total: Int, stage: String)? {
        guard case .running(let done, let total, let stage) = state else { return nil }
        return (done, total, stage)
    }

    private var stage: String {
        switch state {
        case .staging: "Preparing the signal"
        case .running(_, _, let stage): stage.capitalized
        case .done: "Transcript complete"
        case .failed: "The circuit opened"
        case .idle: "Observatory standing by"
        }
    }

    private var detail: String {
        switch state {
        case .staging:
            "Decoding audio, trimming silence, and applying the selected on-device conditioning."
        case .running(let done, let total, let stage) where stage == "decoding":
            "The encoder has completed \(done) of \(total) real 30-second inference windows."
        case .running(_, _, let stage) where stage == "labeling speakers":
            "Sortformer is mapping acoustic turns into anonymous speaker lanes."
        case .running where run?.stage == "fusing":
            "Timestamped words and speaker turns are being projected onto one transcript."
        case .running(let done, let total, _):
            "Native pipeline activity · \(done) of \(total) windows complete."
        case .done:
            "All requested local stages finished and the final transcript was published."
        case .failed(let reason): reason
        case .idle: "Choose or record audio to energize the machine."
        }
    }

    private var active: Bool {
        if case .staging = state { return true }
        if case .running = state { return true }
        return false
    }

    private var accent: Color {
        switch state {
        case .failed: Lab.danger
        case .done: Lab.emerald
        case .running(_, _, let stage) where stage == "labeling speakers": Lab.violet
        case .running(_, _, let stage) where stage == "fusing": Lab.amber
        default: Lab.cyan
        }
    }

    private var uniqueSpeakers: Int {
        Set(segments.compactMap(\.speaker)).count
    }

    var body: some View {
        TimelineView(
            .animation(
                minimumInterval: animationConstrained ? 1.0 / 12.0 : 1.0 / 24.0,
                paused: !active || reduceMotion
            )
        ) { timeline in
            VStack(alignment: .leading, spacing: 16) {
                header(at: timeline.date)

                ZStack {
                    chamberCanvas(time: timeline.date.timeIntervalSinceReferenceDate)
                    chamberLabels
                }
                .frame(height: 238)
                .accessibilityHidden(true)

                metrics(at: timeline.date)

                if let latest = segments.last?.text.trimmingCharacters(in: .whitespaces),
                   !latest.isEmpty
                {
                    HStack(alignment: .firstTextBaseline, spacing: 9) {
                        Image(systemName: "quote.bubble.fill")
                            .foregroundStyle(accent)
                        Text(latest)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(Lab.textPrimary)
                            .lineLimit(2)
                            .contentTransition(.opacity)
                    }
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                if active {
                    Button(role: .cancel, action: cancel) {
                        Label("Abort run", systemImage: "stop.fill")
                    }
                    .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                }
            }
            .animation(.snappy(duration: 0.35), value: stage)
            .animation(.easeOut(duration: 0.25), value: segments.count)
        }
        .padding(18)
        .background {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(
                    reduceTransparency
                        ? Color(red: 0.015, green: 0.055, blue: 0.05)
                        : Color.black.opacity(0.44)
                )
                .overlay {
                    LinearGradient(
                        colors: [accent.opacity(0.12), .clear, Lab.violet.opacity(0.06)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                }
        }
        .overlay {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .strokeBorder(
                    LinearGradient(
                        colors: [accent.opacity(0.55), Color.white.opacity(0.04), Lab.violet.opacity(0.2)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 1
                )
        }
        .shadow(color: accent.opacity(active ? 0.2 : 0.06), radius: 30, y: 14)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Transcription observatory")
        .accessibilityValue("\(stage). \(detail)")
    }

    private var animationConstrained: Bool {
        reduceMotion
            || luminanceReduced
            || ProcessInfo.processInfo.isLowPowerModeEnabled
            || ProcessInfo.processInfo.thermalState.rawValue
                >= ProcessInfo.ThermalState.serious.rawValue
    }

    private func header(at date: Date) -> some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle().fill(accent.opacity(0.14))
                Circle().stroke(accent.opacity(0.6), lineWidth: 1)
                Image(systemName: stageSymbol)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(accent)
                    .symbolEffect(.pulse, isActive: active && !reduceMotion)
            }
            .frame(width: 44, height: 44)

            VStack(alignment: .leading, spacing: 4) {
                Text(stage.uppercased())
                    .font(.system(size: 13, weight: .black, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Lab.textPrimary)
                Text(detail)
                    .font(.system(size: 12))
                    .foregroundStyle(Lab.textSecondary)
            }

            Spacer(minLength: 8)
            if active {
                Text(Self.clock(elapsed(at: date)))
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Lab.textSecondary)
                    .monospacedDigit()
            }
        }
    }

    private var stageSymbol: String {
        switch state {
        case .staging: "waveform.path.ecg"
        case .running(_, _, let stage) where stage == "labeling speakers": "person.2.wave.2"
        case .running(_, _, let stage) where stage == "fusing": "point.3.connected.trianglepath.dotted"
        case .running: "ear.badge.waveform"
        case .done: "checkmark.seal.fill"
        case .failed: "bolt.slash.fill"
        case .idle: "waveform.badge.magnifyingglass"
        }
    }

    private var chamberLabels: some View {
        VStack {
            HStack {
                instrumentLabel("AUDIO")
                Spacer()
                instrumentLabel("NEURAL FIELD")
                Spacer()
                instrumentLabel("WORDS")
            }
            Spacer()
            HStack {
                instrumentLabel(run.map { "WINDOW \($0.done)/\($0.total)" } ?? "SIGNAL")
                Spacer()
                instrumentLabel("\(segments.count) SEGMENTS")
            }
        }
        .padding(12)
    }

    private func instrumentLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .bold, design: .monospaced))
            .tracking(1.2)
            .foregroundStyle(Lab.textSecondary.opacity(0.8))
    }

    private func metrics(at date: Date) -> some View {
        HStack(spacing: 9) {
            metric("WINDOWS", run.map { "\($0.done)/\($0.total)" } ?? "—")
            metric("SEGMENTS", segments.count.formatted())
            metric("SPEAKERS", uniqueSpeakers == 0 ? "—" : uniqueSpeakers.formatted())
            metric("ELAPSED", Self.clock(elapsed(at: date)))
        }
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: 8, weight: .bold, design: .monospaced))
                .tracking(1)
                .foregroundStyle(Lab.textSecondary.opacity(0.75))
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundStyle(Lab.textPrimary)
                .monospacedDigit()
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 10))
    }

    private func chamberCanvas(time: TimeInterval) -> some View {
        Canvas(opaque: false, colorMode: .linear, rendersAsynchronously: true) { context, size in
            let rect = CGRect(origin: .zero, size: size).insetBy(dx: 1, dy: 1)
            let midY = size.height * 0.5

            context.fill(
                Path(roundedRect: rect, cornerRadius: 18),
                with: .linearGradient(
                    Gradient(colors: [Color.black.opacity(0.52), accent.opacity(0.045)]),
                    startPoint: .zero,
                    endPoint: CGPoint(x: size.width, y: size.height)
                )
            )

            drawWaveguide(context: &context, size: size, midY: midY, time: time)
            drawNeuralField(context: &context, size: size, time: time)
            drawWindows(context: &context, size: size)
        }
    }

    private func drawWaveguide(
        context: inout GraphicsContext,
        size: CGSize,
        midY: CGFloat,
        time: TimeInterval
    ) {
        var wave = Path()
        let sampleCount = 72
        for index in 0...sampleCount {
            let fraction = CGFloat(index) / CGFloat(sampleCount)
            let envelope = sin(.pi * fraction)
            let realEnergy = CGFloat(min(1, Double(segments.count) / 12.0))
            let amplitude = 7 + 18 * max(0.18, realEnergy)
            let phase = CGFloat(time) * (reduceMotion ? 0 : 2.2)
            let y = midY + sin(fraction * 10 * .pi + phase) * amplitude * envelope
            let point = CGPoint(x: 16 + fraction * (size.width - 32), y: y)
            index == 0 ? wave.move(to: point) : wave.addLine(to: point)
        }
        context.stroke(
            wave,
            with: .linearGradient(
                Gradient(colors: [Lab.cyan.opacity(0.28), accent, Lab.violet.opacity(0.45)]),
                startPoint: CGPoint(x: 0, y: midY),
                endPoint: CGPoint(x: size.width, y: midY)
            ),
            lineWidth: 2
        )
    }

    private func drawNeuralField(
        context: inout GraphicsContext,
        size: CGSize,
        time: TimeInterval
    ) {
        let center = CGPoint(x: size.width * 0.55, y: size.height * 0.48)
        let count = max(7, min(22, segments.count + 7))
        var points: [CGPoint] = []
        for index in 0..<count {
            let angle = CGFloat(index) * 2.399963 + CGFloat(reduceMotion ? 0 : time * 0.06)
            let radius = 18 + CGFloat(index % 6) * 8
            points.append(
                CGPoint(
                    x: center.x + cos(angle) * radius,
                    y: center.y + sin(angle) * radius * 0.72
                )
            )
        }
        for index in points.indices where index > 0 {
            var connection = Path()
            connection.move(to: points[index])
            connection.addLine(to: points[(index * 5) % index])
            context.stroke(connection, with: .color(accent.opacity(0.12)), lineWidth: 0.7)
        }
        for (index, point) in points.enumerated() {
            let energized = index < min(points.count, segments.count + 2)
            let radius: CGFloat = energized ? 3.2 : 2
            context.fill(
                Path(ellipseIn: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)),
                with: .color((energized ? accent : Lab.textSecondary).opacity(energized ? 0.9 : 0.25))
            )
        }
    }

    private func drawWindows(context: inout GraphicsContext, size: CGSize) {
        guard let run else { return }
        let visible = min(18, max(1, run.total))
        let width = min(15, (size.width - 36) / CGFloat(visible) - 3)
        let startX = size.width * 0.5 - CGFloat(visible) * (width + 3) * 0.5
        for index in 0..<visible {
            let firstRepresented = max(0, run.total - visible)
            let complete = firstRepresented + index < run.done
            let frame = CGRect(
                x: startX + CGFloat(index) * (width + 3),
                y: size.height - 42,
                width: width,
                height: 14
            )
            context.fill(
                Path(roundedRect: frame, cornerRadius: 3),
                with: .color(complete ? accent.opacity(0.9) : Color.white.opacity(0.06))
            )
        }
    }

    private func elapsed(at date: Date) -> TimeInterval {
        max(0, date.timeIntervalSince(started ?? date))
    }

    private static func clock(_ seconds: TimeInterval) -> String {
        let whole = max(0, Int(seconds.rounded(.down)))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}
