import Foundation

protocol SubtitleTimingSource {
    var text: String { get }
    var startSec: Double { get }
    var endSec: Double { get }
}

/// One decoder-timed word in the caption timeline. These values come directly
/// from FrankenWhisper's DTW alignment; the subtitle feature never estimates
/// word boundaries from character counts or segment duration.
struct SubtitleTimelineWord: Identifiable, Hashable {
    let id: Int
    var text: String
    var startSec: Double
    var endSec: Double
}

struct SubtitleCue: Identifiable, Hashable {
    let id: Int
    var words: [SubtitleTimelineWord]

    var startSec: Double { words.first?.startSec ?? 0 }
    var endSec: Double { words.last?.endSec ?? startSec }
    var text: String { words.map(\.text).joined(separator: " ") }
}

enum SubtitleTimeline {
    static let maximumWordsPerCue = 7
    static let maximumCueDuration = 3.2

    static func makeCues<T: SubtitleTimingSource>(from nestedWords: [[T]]?) -> [SubtitleCue] {
        guard let nestedWords else { return [] }

        var flattened: [SubtitleTimelineWord] = []
        flattened.reserveCapacity(nestedWords.reduce(0) { $0 + $1.count })

        for timing in nestedWords.joined() {
            let text = timing.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty,
                  timing.startSec.isFinite,
                  timing.endSec.isFinite,
                  timing.startSec >= 0,
                  timing.endSec > timing.startSec
            else { continue }

            if isStandalonePunctuation(text), !flattened.isEmpty {
                flattened[flattened.count - 1].text += text
                flattened[flattened.count - 1].endSec = max(
                    flattened[flattened.count - 1].endSec,
                    timing.endSec
                )
            } else {
                flattened.append(
                    SubtitleTimelineWord(
                        id: flattened.count,
                        text: text,
                        startSec: timing.startSec,
                        endSec: timing.endSec
                    )
                )
            }
        }

        var cues: [SubtitleCue] = []
        var current: [SubtitleTimelineWord] = []

        func flush() {
            guard !current.isEmpty else { return }
            cues.append(SubtitleCue(id: cues.count, words: current))
            current.removeAll(keepingCapacity: true)
        }

        for word in flattened {
            if let first = current.first,
               current.count >= maximumWordsPerCue
                || word.endSec - first.startSec > maximumCueDuration
            {
                flush()
            }
            current.append(word)
            if current.count >= 3, endsSentence(word.text) {
                flush()
            }
        }
        flush()
        return cues
    }

    /// Word alignment is relative to the extracted audio file. Restore the
    /// audio track's original position before drawing over the source video.
    static func offset(_ cues: [SubtitleCue], by seconds: Double) -> [SubtitleCue] {
        guard seconds.isFinite, seconds != 0 else { return cues }
        return cues.compactMap { cue in
            let words = cue.words.compactMap { word -> SubtitleTimelineWord? in
                let shiftedEnd = word.endSec + seconds
                guard shiftedEnd > 0 else { return nil }
                let shiftedStart = max(0, word.startSec + seconds)
                guard shiftedEnd > shiftedStart else { return nil }
                return SubtitleTimelineWord(
                    id: word.id,
                    text: word.text,
                    startSec: shiftedStart,
                    endSec: shiftedEnd
                )
            }
            guard !words.isEmpty else { return nil }
            return SubtitleCue(id: cue.id, words: words)
        }
    }

    private static func isStandalonePunctuation(_ text: String) -> Bool {
        text.unicodeScalars.allSatisfy {
            CharacterSet.punctuationCharacters.contains($0)
        }
    }

    private static func endsSentence(_ text: String) -> Bool {
        guard let last = text.last else { return false }
        return ".!?\u{2026}".contains(last)
    }
}
