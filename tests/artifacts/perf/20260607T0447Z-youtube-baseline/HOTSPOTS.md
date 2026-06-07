# Hotspot Table — youtube orchestration layer (run 20260607-youtube-baseline)

Host: Apple M4 Pro, macOS 26.2, release-perf. Scenario: `franken_whisper youtube`
on real YouTube (jNQXAC9IVRw "Me at the zoo", 19s) — single video, watch URL,
tiny.en native, total wall 12.46s. Stub run (2 videos) used to COUNT yt-dlp
invocations (instant stub isolates orchestration from network variance).

| Rank | Location | Metric | Value | % of wall | Category | Evidence |
|------|----------|--------|-------|-----------|----------|----------|
| 1 | redundant metadata fetches (pipeline.rs:238 resolve + :442 download + :477 render) | yt-dlp `-j` calls/video | **3** (2 redundant) | ~27% (~3.3s/video) | network/redundant | stub call-log: 6 metadata calls for 2 videos; spans yt.dl_metadata 1635ms + yt.render_metadata 1700ms |
| 2 | yt.download (ytdlp.rs download_audio) | wall | 5691ms | 46% | network (irreducible) | spans_real |
| 3 | resolve_videos serial fetch (pipeline.rs:233-242) | wall | ~1.6s × N, SERIAL pre-pool | (in #1) | network/serialization | for-loop over URLs fetches before the concurrent download pool starts |
| 4 | manifest.save (pipeline.rs, 6 sites) rewrites WHOLE manifest each call | write volume | O(N²) cumulative bytes | small abs | I/O | 6 save sites; each serializes the full BTreeMap |
| 5 | yt.transcribe (engine) | wall | 577ms | 5% | CPU (already optimized r1-3) | spans_real |
| 6 | yt.render (md+json) | wall | 11.8ms | <0.1% | CPU | spans_real — negligible |

# Hypothesis Ledger

| hypothesis | verdict | evidence |
|---|---|---|
| metadata fetched redundantly per video | SUPPORTS (headline) | stub: 3 `-j` calls/video; eliminable to 1 → ~3.3s/video on real network |
| resolve stage serializes a network fetch per Video URL before any download overlap | SUPPORTS | pipeline.rs:233-242 for-loop calls fetch_metadata synchronously; for watch-URL batches this is N serial round-trips with zero overlap |
| download dominates and is irreducible | SUPPORTS | 5.7s/video network-bound; concurrency pool already overlaps across videos |
| transcription is a youtube-layer hotspot | REJECTS | 577ms, 5% — engine already at its floor (rounds 1-3) |
| render (md/json/atomic writes) is hot | REJECTS | 11.8ms |
| manifest I/O is a wall-clock hotspot | REJECTS for small N; SUPPORTS as O(N²) write-volume risk for large playlists | full-map rewrite × 6 saves/video |

# The lever (for extreme-software-optimization)
Collapse 3 metadata fetches → 1: fetch VideoMeta ONCE in the download worker,
thread it through to render (carry it on the channel alongside the audio path),
and in resolve_videos DON'T fetch for Video/Ambiguous URLs — extract the id
locally (the URL parsing already exists in classify_url) for dedup, deferring
the single authoritative fetch to the worker. Saves ~2 network round-trips/video
(~3.3s/video here; ~2N round-trips across an N-video playlist) AND removes the
serial pre-pool fetch so download concurrency starts immediately.
Golden gate: identical md/json output (same metadata, just fetched once).
