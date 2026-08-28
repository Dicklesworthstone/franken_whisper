// The monster's palette: the FrankenSuite dark-emerald laboratory, in native
// clothes. Shared bones with the FrankenTTS/FrankenOCR apps so the family
// reads as one bench of instruments.

import SwiftUI

enum Lab {
    static let background = Color(red: 0.004, green: 0.024, blue: 0.019)
    static let backgroundRaised = Color(red: 0.014, green: 0.065, blue: 0.052)
    static let panel = Color.black.opacity(0.46)
    static let stroke = Color.white.opacity(0.075)
    static let emerald = Color(red: 0.204, green: 0.827, blue: 0.6) // #34d399
    static let emeraldDeep = Color(red: 0.0, green: 0.259, blue: 0.145) // #004225
    static let cyan = Color(red: 0.22, green: 0.84, blue: 0.96)
    static let textPrimary = Color(red: 0.886, green: 0.91, blue: 0.941)
    static let textSecondary = Color(red: 0.58, green: 0.639, blue: 0.722)
    static let amber = Color(red: 0.984, green: 0.749, blue: 0.141) // warnings only
    static let violet = Color(red: 0.655, green: 0.545, blue: 0.98) // structure/metadata
    static let danger = Color(red: 0.973, green: 0.443, blue: 0.443)

    /// Deterministic per-speaker tint: the same speaker always gets the same
    /// color within a transcript, like the CLI's TUI.
    static func speakerColor(_ speaker: String?) -> Color {
        guard let speaker, !speaker.isEmpty else { return textSecondary }
        let palette: [Color] = [emerald, violet, amber, Color(red: 0.38, green: 0.72, blue: 0.96),
                                Color(red: 0.96, green: 0.55, blue: 0.73)]
        var hash: UInt32 = 2_166_136_261
        for byte in speaker.utf8 {
            hash = (hash ^ UInt32(byte)) &* 16_777_619
        }
        return palette[Int(hash % UInt32(palette.count))]
    }
}

/// Spatial laboratory atmosphere shared by compact and desktop layouts. The
/// grid is intentionally quiet: text and controls retain the visual hierarchy.
struct LaboratoryBackground: View {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        ZStack {
            Lab.background
            RadialGradient(
                colors: [Lab.emerald.opacity(0.16), .clear],
                center: .topLeading,
                startRadius: 0,
                endRadius: 520
            )
            RadialGradient(
                colors: [Lab.violet.opacity(0.11), .clear],
                center: .bottomTrailing,
                startRadius: 0,
                endRadius: 640
            )
            if !reduceTransparency {
                Canvas { context, size in
                    let spacing: CGFloat = 44
                    var path = Path()
                    for x in stride(from: CGFloat.zero, through: size.width, by: spacing) {
                        path.move(to: CGPoint(x: x, y: 0))
                        path.addLine(to: CGPoint(x: x, y: size.height))
                    }
                    for y in stride(from: CGFloat.zero, through: size.height, by: spacing) {
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: size.width, y: y))
                    }
                    context.stroke(path, with: .color(Lab.emerald.opacity(0.035)), lineWidth: 0.5)
                }
            }
        }
        .ignoresSafeArea()
        .accessibilityHidden(true)
    }
}

/// Section label in the site's uppercase-tracked style.
struct LabLabel: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 11, weight: .black, design: .monospaced))
            .kerning(2.5)
            .foregroundStyle(Lab.emerald)
    }
}

/// One stitched bolt stud, the theme's signature.
struct Bolt: View {
    var body: some View {
        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [Color(white: 0.35), Color(white: 0.05)],
                        center: .topLeading, startRadius: 1, endRadius: 8))
            Rectangle().fill(Color(white: 0.15)).frame(width: 1.2, height: 7).rotationEffect(.degrees(45))
            Rectangle().fill(Color(white: 0.15)).frame(width: 1.2, height: 7).rotationEffect(.degrees(-45))
        }
        .frame(width: 13, height: 13)
        .overlay(Circle().stroke(Color.white.opacity(0.15), lineWidth: 0.8))
        .shadow(color: Lab.emerald.opacity(0.35), radius: 4)
        .accessibilityHidden(true)
    }
}

/// The laboratory panel: dark card, hairline border, bolts on two corners.
struct LabPanel<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(Lab.panel)
                    .overlay {
                        LinearGradient(
                            colors: [Lab.emerald.opacity(0.045), .clear, Lab.violet.opacity(0.035)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                    }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(
                        LinearGradient(
                            colors: [Lab.emerald.opacity(0.2), Lab.stroke, Lab.violet.opacity(0.12)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 1
                    )
            )
            .shadow(color: Color.black.opacity(0.28), radius: 20, y: 10)
            .overlay(alignment: .topLeading) { Bolt().offset(x: -5, y: -5) }
            .overlay(alignment: .bottomTrailing) { Bolt().offset(x: 5, y: 5) }
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .black, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.vertical, 11)
            .frame(minHeight: 44)
            .background(
                LinearGradient(
                    colors: [Lab.emeraldDeep, Lab.emerald.opacity(0.8)],
                    startPoint: .topLeading, endPoint: .bottomTrailing),
                in: Capsule())
            .opacity(isEnabled ? (configuration.isPressed ? 0.75 : 1) : 0.35)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

struct GhostButtonStyle: ButtonStyle {
    var tint: Color = Lab.textSecondary
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .black, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(tint)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .frame(minHeight: 44)
            .background(tint.opacity(configuration.isPressed ? 0.14 : 0.04), in: Capsule())
            .overlay(Capsule().stroke(tint.opacity(0.35), lineWidth: 1))
            .opacity(isEnabled ? 1 : 0.35)
    }
}

/// Thin emerald progress bar with the lab's hairline framing.
struct LabProgressBar: View {
    /// 0...1; values outside are clamped.
    let fraction: Double
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.06))
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [Lab.emeraldDeep, Lab.emerald],
                            startPoint: .leading, endPoint: .trailing))
                    .frame(width: max(6, geo.size.width * min(1, max(0, fraction))))
            }
        }
        .frame(height: 6)
        .animation(.easeOut(duration: 0.25), value: fraction)
        .accessibilityElement()
        .accessibilityLabel("Progress")
        .accessibilityValue("\(Int(min(1, max(0, fraction)) * 100)) percent")
    }
}

/// One diagnostic line: colored state dot + monospaced message.
struct StatusLine: View {
    enum Kind { case neutral, ok, warn, err }
    let kind: Kind
    let text: String
    private var color: Color {
        switch kind {
        case .neutral: Lab.textSecondary
        case .ok: Lab.emerald
        case .warn: Lab.amber
        case .err: Lab.danger
        }
    }
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(text)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(color)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Live input level meter for the recorder (0...1).
struct LevelMeter: View {
    let level: Float
    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.06))
                Capsule()
                    .fill(Lab.emerald.opacity(0.85))
                    .frame(width: max(3, geo.size.width * CGFloat(min(1, max(0, level)))))
            }
        }
        .frame(height: 4)
        .animation(.linear(duration: 0.1), value: level)
        .accessibilityElement()
        .accessibilityLabel("Microphone level")
        .accessibilityValue("\(Int(min(1, max(0, level)) * 100)) percent")
    }
}

/// Dark inset text field matching the site's `.text-input`: hairline border,
/// monospaced, emerald caret.
struct LabTextFieldModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(.system(size: 13, design: .monospaced))
            .foregroundStyle(Lab.textPrimary)
            .tint(Lab.emerald)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.1), lineWidth: 1))
    }
}

extension View {
    func labTextField() -> some View { modifier(LabTextFieldModifier()) }
}

/// The expanding ring behind an active recording.
private struct PulseRing: View {
    @State private var expanded = false
    var body: some View {
        Circle()
            .stroke(Lab.danger.opacity(0.55), lineWidth: 2)
            .scaleEffect(expanded ? 1.45 : 1)
            .opacity(expanded ? 0 : 0.9)
            .onAppear {
                withAnimation(.easeOut(duration: 1.2).repeatForever(autoreverses: false)) {
                    expanded = true
                }
            }
    }
}

/// A proper record control: a big round mic button that flips to a stop
/// square and pulses while capture is live.
struct RecordButton: View {
    let isRecording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                if isRecording { PulseRing() }
                Circle()
                    .fill(isRecording ? Lab.danger.opacity(0.16) : Lab.emeraldDeep.opacity(0.55))
                Circle()
                    .stroke(isRecording ? Lab.danger : Lab.emerald, lineWidth: 2)
                Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundStyle(isRecording ? Lab.danger : Lab.emerald)
                    .contentTransition(.symbolEffect(.replace))
            }
            .frame(width: 64, height: 64)
            .shadow(
                color: (isRecording ? Lab.danger : Lab.emerald).opacity(0.35), radius: 10)
        }
        .buttonStyle(.plain)
        .animation(.snappy, value: isRecording)
        .accessibilityLabel(isRecording ? "Stop recording" : "Start recording")
    }
}
