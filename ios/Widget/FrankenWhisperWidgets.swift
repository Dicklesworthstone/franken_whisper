import ActivityKit
import SwiftUI
import WidgetKit

private let signalGreen = Color(red: 0.20, green: 0.83, blue: 0.60)
private let signalCyan = Color(red: 0.25, green: 0.82, blue: 0.96)
private let signalInk = Color(red: 0.002, green: 0.025, blue: 0.025)

struct WhisperTimelineEntry: TimelineEntry {
    let date: Date
    let snapshot: FrankenWhisperWidgetSnapshot
}

struct WhisperTimelineProvider: TimelineProvider {
    func placeholder(in context: Context) -> WhisperTimelineEntry {
        WhisperTimelineEntry(date: .now, snapshot: .placeholder)
    }
    func getSnapshot(in context: Context, completion: @escaping (WhisperTimelineEntry) -> Void) {
        completion(WhisperTimelineEntry(date: .now, snapshot: FrankenWhisperSharedStore.loadWidgetSnapshot()))
    }
    func getTimeline(in context: Context, completion: @escaping (Timeline<WhisperTimelineEntry>) -> Void) {
        let entry = WhisperTimelineEntry(date: .now, snapshot: FrankenWhisperSharedStore.loadWidgetSnapshot())
        completion(Timeline(entries: [entry], policy: .after(.now.addingTimeInterval(15 * 60))))
    }
}

struct FrankenWhisperObservatoryWidget: Widget {
    let kind = "FrankenWhisperObservatoryWidget"
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: WhisperTimelineProvider()) { entry in
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "waveform.badge.mic")
                        .font(.title3.bold()).foregroundStyle(signalCyan)
                    Spacer()
                    Text("LOCAL_SIGNAL")
                        .font(.system(size: 8, weight: .black, design: .monospaced))
                        .tracking(1.3).foregroundStyle(signalGreen)
                }
                Spacer(minLength: 0)
                Text(entry.snapshot.headline).font(.headline).foregroundStyle(.white).lineLimit(2)
                Text(entry.snapshot.detail).font(.caption).foregroundStyle(.white.opacity(0.68)).lineLimit(2)
            }
            .containerBackground(for: .widget) {
                LinearGradient(
                    colors: [signalInk, Color.black, signalCyan.opacity(0.14)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }
            .widgetURL(URL(string: "frankenwhisper://new"))
        }
        .configurationDisplayName("Speech Observatory")
        .description("See private transcription status and open FrankenWhisper.")
        .supportedFamilies([.systemSmall, .systemMedium])
    }
}

struct FrankenWhisperLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: FrankenWhisperRunActivityAttributes.self) { context in
            WhisperLockView(context: context)
                .activityBackgroundTint(signalInk)
                .activitySystemActionForegroundColor(signalCyan)
                .widgetURL(URL(string: "frankenwhisper://new"))
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: icon(context.state.status)).font(.title2.bold()).foregroundStyle(signalCyan)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Text(timerInterval: context.attributes.startedAt...Date.distantFuture, countsDown: false)
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                DynamicIslandExpandedRegion(.center) {
                    Text(context.state.stage).font(.headline).lineLimit(1)
                }
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .firstTextBaseline) {
                            Text(context.state.detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                            Spacer(minLength: 8)
                            if context.state.status != .complete && context.state.status != .failed {
                                Link("Stop", destination: stopURL(context.state))
                                    .font(.caption.weight(.semibold)).foregroundStyle(.red)
                            }
                        }
                        WindowRail(done: context.state.windowsDone, total: context.state.windowsTotal)
                    }
                }
            } compactLeading: {
                Image(systemName: icon(context.state.status)).foregroundStyle(signalCyan)
            } compactTrailing: {
                compactTrailing(context.state)
            } minimal: {
                Image(systemName: icon(context.state.status)).foregroundStyle(signalCyan)
            }
            .widgetURL(URL(string: "frankenwhisper://new"))
            .keylineTint(signalCyan)
        }
    }
}

private struct WhisperLockView: View {
    let context: ActivityViewContext<FrankenWhisperRunActivityAttributes>
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon(context.state.status))
                .font(.title2.bold()).foregroundStyle(signalCyan)
                .frame(width: 44, height: 44).background(signalCyan.opacity(0.12), in: Circle())
            VStack(alignment: .leading, spacing: 3) {
                Text(context.state.stage).font(.headline).lineLimit(1)
                Text(context.state.detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                WindowRail(done: context.state.windowsDone, total: context.state.windowsTotal)
            }
            Spacer(minLength: 4)
            Text(timerInterval: context.attributes.startedAt...Date.distantFuture, countsDown: false)
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
        .padding(15)
    }
}

private struct WindowRail: View {
    let done: Int
    let total: Int
    var body: some View {
        if total > 0 {
            ProgressView(value: Double(done), total: Double(total)).tint(signalCyan)
        } else {
            HStack(spacing: 3) {
                ForEach(0..<8, id: \.self) { index in
                    Capsule().fill(index.isMultiple(of: 3) ? signalCyan : signalGreen.opacity(0.25)).frame(height: 3)
                }
            }
        }
    }
}

private func icon(_ status: FrankenWhisperRunContentState.Status) -> String {
    switch status {
    case .preparing: "waveform.circle"
    case .activating: "mic.badge.plus"
    case .listening: "mic.fill"
    case .armed: "keyboard.badge.ellipsis"
    case .decoding: "waveform.path.ecg"
    case .speakers: "person.2.wave.2"
    case .fusing: "point.3.connected.trianglepath.dotted"
    case .complete: "checkmark.seal.fill"
    case .cancelled: "stop.circle"
    case .failed: "exclamationmark.triangle.fill"
    }
}

@ViewBuilder
private func compactTrailing(_ state: FrankenWhisperRunContentState) -> some View {
    switch state.status {
    case .listening, .fusing where state.windowsTotal == 0:
        if state.emittedSegments > 0 {
            Text(state.emittedSegments, format: .number)
                .font(.caption2.monospacedDigit()).foregroundStyle(signalGreen)
        } else {
            Image(systemName: "waveform").foregroundStyle(signalGreen)
        }
    case .armed:
        Text("READY").font(.system(size: 8, weight: .black, design: .monospaced)).foregroundStyle(signalGreen)
    default:
        Text("\(state.windowsDone)/\(state.windowsTotal)")
            .font(.caption2.monospacedDigit()).foregroundStyle(signalGreen)
    }
}

private func stopURL(_ state: FrankenWhisperRunContentState) -> URL {
    let host: String
    switch state.status {
    case .listening, .armed, .activating:
        host = "end-live"
    case .fusing where state.windowsTotal == 0:
        host = "end-live"
    default:
        host = "cancel-run"
    }
    return URL(string: "frankenwhisper://\(host)")!
}

@main
struct FrankenWhisperWidgetBundle: WidgetBundle {
    var body: some Widget {
        FrankenWhisperObservatoryWidget()
        FrankenWhisperLiveActivity()
    }
}
