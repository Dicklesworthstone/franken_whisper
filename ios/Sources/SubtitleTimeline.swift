import Foundation
import UIKit

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
    /// The diarizer's stable anonymous lane is retained only as a color key.
    /// `displayName` is deliberately nil until the user explicitly names it;
    /// raw `SPEAKER_NN` labels must never leak into a burned-in caption.
    var speaker: SubtitleSpeaker?
}

struct SubtitleSpeaker: Hashable {
    var laneID: String
    var displayName: String?
}

struct SubtitleSpeakerSpan: Hashable {
    var startSec: Double
    var endSec: Double
    var laneID: String
    var confidence: Double
}

/// A diarizer-observed interval containing actual speech. Decoder DTW can
/// occasionally pin a boundary word to the edge of a 30-second Whisper window,
/// making that one word swallow several seconds of leading or trailing silence.
/// These spans are used only to trim those pathological silent tails; normal
/// word alignment remains the decoder's unmodified timing.
struct SubtitleSpeechSpan: Hashable {
    var startSec: Double
    var endSec: Double
}

enum SubtitleSpeakerPalette {
    static func uiColor(for speaker: String?) -> UIColor {
        guard let speaker, !speaker.isEmpty else {
            return UIColor(red: 0.58, green: 0.639, blue: 0.722, alpha: 1)
        }
        let palette: [UIColor] = [
            UIColor(red: 0.204, green: 0.827, blue: 0.6, alpha: 1),
            UIColor(red: 0.984, green: 0.749, blue: 0.141, alpha: 1),
            UIColor(red: 0.38, green: 0.72, blue: 0.96, alpha: 1),
            UIColor(red: 0.96, green: 0.55, blue: 0.73, alpha: 1),
            UIColor(red: 0.655, green: 0.545, blue: 0.98, alpha: 1)
        ]
        if let lane = speaker.split(separator: "_").last.flatMap({ Int($0) }) {
            return palette[lane % palette.count]
        }
        var hash: UInt32 = 2_166_136_261
        for byte in speaker.utf8 {
            hash = (hash ^ UInt32(byte)) &* 16_777_619
        }
        return palette[Int(hash % UInt32(palette.count))]
    }
}

struct SubtitleCue: Identifiable, Hashable {
    let id: Int
    var words: [SubtitleTimelineWord]

    var startSec: Double { words.first?.startSec ?? 0 }
    var endSec: Double { words.last?.endSec ?? startSec }
    var text: String { words.map(\.text).joined(separator: " ") }
    var speaker: SubtitleSpeaker? { words.first?.speaker }
}

enum SubtitleTimeline {
    static let maximumWordsPerCue = 7
    static let maximumCueDuration = 3.2

    static func makeCues<T: SubtitleTimingSource>(from nestedWords: [[T]]?) -> [SubtitleCue] {
        makeCues(
            from: nestedWords,
            segmentSpeakers: [],
            speakerSpans: [],
            speakerNames: [:]
        )
    }

    /// Decoder word groups normally correspond 1:1 with transcript segments,
    /// which is the most precise speaker attribution available. Projected
    /// spans are a timestamp-overlap fallback for older result shapes.
    static func makeCues<T: SubtitleTimingSource>(
        from nestedWords: [[T]]?,
        segmentSpeakers: [String?],
        speakerSpans: [SubtitleSpeakerSpan] = [],
        speechSpans: [SubtitleSpeechSpan] = [],
        speakerNames: [String: String] = [:]
    ) -> [SubtitleCue] {
        let flattened = trimPathologicalSilentTails(
            in: flattenedWords(
                from: nestedWords,
                segmentSpeakers: segmentSpeakers,
                speakerSpans: speakerSpans,
                speakerNames: speakerNames
            ),
            speechSpans: speechSpans
        )

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
                || word.speaker?.laneID != first.speaker?.laneID {
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

    private static func flattenedWords<T: SubtitleTimingSource>(
        from nestedWords: [[T]]?,
        segmentSpeakers: [String?],
        speakerSpans: [SubtitleSpeakerSpan],
        speakerNames: [String: String]
    ) -> [SubtitleTimelineWord] {
        guard let nestedWords else { return [] }
        var flattened: [SubtitleTimelineWord] = []
        flattened.reserveCapacity(nestedWords.reduce(0) { $0 + $1.count })

        for (segmentIndex, timings) in nestedWords.enumerated() {
            for timing in timings {
                let text = timing.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty,
                      timing.startSec.isFinite,
                      timing.endSec.isFinite,
                      timing.startSec >= 0,
                      timing.endSec > timing.startSec
                else { continue }

                let segmentLane = segmentIndex < segmentSpeakers.count
                    ? segmentSpeakers[segmentIndex]
                    : nil
                let speaker = subtitleSpeaker(
                    lane: segmentLane ?? overlappingLane(for: timing, in: speakerSpans),
                    names: speakerNames
                )
                if isStandalonePunctuation(text),
                   !flattened.isEmpty,
                   flattened[flattened.count - 1].speaker?.laneID == speaker?.laneID {
                    flattened[flattened.count - 1].text += text
                    flattened[flattened.count - 1].endSec = max(
                        flattened[flattened.count - 1].endSec,
                        timing.endSec
                    )
                    continue
                }
                flattened.append(
                    SubtitleTimelineWord(
                        id: flattened.count,
                        text: text,
                        startSec: timing.startSec,
                        endSec: timing.endSec,
                        speaker: speaker
                    )
                )
            }
        }
        return flattened
    }

    /// Keep genuine DTW timing authoritative unless a word spends both a
    /// substantial absolute duration and a substantial fraction of its span
    /// outside speech detected by Sortformer. The best-overlap turn is used
    /// rather than the union of turns, so a single malformed word cannot bridge
    /// multiple seconds of silence between two separate utterances.
    private static func trimPathologicalSilentTails(
        in words: [SubtitleTimelineWord],
        speechSpans: [SubtitleSpeechSpan]
    ) -> [SubtitleTimelineWord] {
        guard !speechSpans.isEmpty else { return words }
        let validSpans = speechSpans.filter {
            $0.startSec.isFinite
                && $0.endSec.isFinite
                && $0.startSec >= 0
                && $0.endSec > $0.startSec
        }
        guard !validSpans.isEmpty else { return words }

        return words.map { word in
            let duration = word.endSec - word.startSec
            guard duration > 0 else { return word }

            let candidate = validSpans.compactMap { span -> (SubtitleSpeechSpan, Double)? in
                let overlap = min(word.endSec, span.endSec) - max(word.startSec, span.startSec)
                return overlap > 0 ? (span, overlap) : nil
            }
            .max { lhs, rhs in lhs.1 < rhs.1 }

            guard let (span, overlap) = candidate else { return word }
            let outsideSpeech = duration - overlap
            guard outsideSpeech >= 0.45,
                  outsideSpeech / duration >= 0.35
            else { return word }

            var correctedStart = max(word.startSec, span.startSec)
            var correctedEnd = min(word.endSec, span.endSec)
            guard correctedEnd > correctedStart else { return word }

            // An 80 ms diarizer boundary can leave a clipped word too brief to
            // read. Grow only inside the same proven speech turn, never back
            // into the silence that triggered the correction.
            let minimumReadableDuration = min(0.12, span.endSec - span.startSec)
            if correctedEnd - correctedStart < minimumReadableDuration {
                correctedStart = max(span.startSec, correctedEnd - minimumReadableDuration)
                if correctedEnd - correctedStart < minimumReadableDuration {
                    correctedEnd = min(span.endSec, correctedStart + minimumReadableDuration)
                }
            }

            var corrected = word
            corrected.startSec = correctedStart
            corrected.endSec = correctedEnd
            return corrected
        }
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
                    endSec: shiftedEnd,
                    speaker: word.speaker
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

    private static func subtitleSpeaker(
        lane: String?,
        names: [String: String]
    ) -> SubtitleSpeaker? {
        guard let lane, !lane.isEmpty else { return nil }
        let cleanName = names[lane]?.trimmingCharacters(in: .whitespacesAndNewlines)
        return SubtitleSpeaker(
            laneID: lane,
            displayName: cleanName.flatMap { $0.isEmpty ? nil : $0 }
        )
    }

    /// Prefer the run with the greatest actual overlap. This handles the
    /// occasional adjacent projected runs whose boundary regions overlap;
    /// midpoint-only attribution can otherwise choose the wrong voice.
    private static func overlappingLane<T: SubtitleTimingSource>(
        for timing: T,
        in spans: [SubtitleSpeakerSpan]
    ) -> String? {
        spans.compactMap { span -> OverlapCandidate? in
            guard span.startSec.isFinite,
                  span.endSec.isFinite,
                  span.endSec > span.startSec,
                  !span.laneID.isEmpty
            else { return nil }
            let overlap = min(timing.endSec, span.endSec) - max(timing.startSec, span.startSec)
            guard overlap > 0 else { return nil }
            return OverlapCandidate(
                lane: span.laneID,
                duration: overlap,
                confidence: span.confidence
            )
        }
        .max {
            if $0.duration == $1.duration { return $0.confidence < $1.confidence }
            return $0.duration < $1.duration
        }?
        .lane
    }

    private struct OverlapCandidate {
        var lane: String
        var duration: Double
        var confidence: Double
    }
}
