// The monster's palette: the FrankenSuite dark-emerald laboratory, in native
// clothes. Shared bones with the FrankenTTS/FrankenOCR apps so the family
// reads as one bench of instruments.

import SwiftUI

enum Lab {
    static let background = Color(red: 0.008, green: 0.039, blue: 0.024) // #020a06 family
    static let panel = Color.black.opacity(0.4)
    static let stroke = Color.white.opacity(0.06)
    static let emerald = Color(red: 0.204, green: 0.827, blue: 0.6) // #34d399
    static let emeraldDeep = Color(red: 0.0, green: 0.259, blue: 0.145) // #004225
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
    }
}

/// The laboratory panel: dark card, hairline border, bolts on two corners.
struct LabPanel<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Lab.panel, in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(Lab.stroke, lineWidth: 1))
            .overlay(alignment: .topLeading) { Bolt().offset(x: -5, y: -5) }
            .overlay(alignment: .bottomTrailing) { Bolt().offset(x: 5, y: 5) }
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .black, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.vertical, 11)
            .background(
                LinearGradient(
                    colors: [Lab.emeraldDeep, Lab.emerald.opacity(0.8)],
                    startPoint: .topLeading, endPoint: .bottomTrailing),
                in: Capsule())
            .opacity(configuration.isPressed ? 0.75 : 1)
    }
}

struct GhostButtonStyle: ButtonStyle {
    var tint: Color = Lab.textSecondary
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .black, design: .monospaced))
            .kerning(1.2)
            .textCase(.uppercase)
            .foregroundStyle(tint)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Color.white.opacity(0.02), in: Capsule())
            .overlay(Capsule().stroke(Color.white.opacity(0.1), lineWidth: 1))
            .opacity(configuration.isPressed ? 0.7 : 1)
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
    }
}
