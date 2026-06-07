# Optimization Loop Results — youtube orchestration (2026-06-07)

Baseline: HOTSPOTS.md in this dir. Engine internals already at floor (rounds 1-3).

| Pass | Lever | Outcome | Evidence |
|------|-------|---------|----------|
| 1 | Collapse 3 metadata fetches/video → 1 | **~3.3s/video saved** (2 network round-trips eliminated) + serial pre-pool fetch removed so download concurrency starts immediately | stub call-log: `-j` calls 6→2 for 2 videos; md/json byte-identical except run-time fields |
| 2 | Coalesce manifest saves | −49.9% cumulative write bytes (24.07→12.07 MB @ N=200) | new byte-count test; resume correctness preserved |
| 3 | Resolve/expand at scale | ZERO-CHANGE on hot paths (all linear) + **fixed a real bug**: playlists > ~1400 entries silently truncated at process.rs's 4 MB capture cap → now an actionable error | scale tests + truncation-detection tests |
| 4 | Bug-hunt after optimization | No regression; locked the manifest-key invariant (URL-derived id, not yt-dlp's meta.id) with a divergence guard + regression test | adversarial audit |
| 5 | Landing | metadata fetches 3→1/video confirmed (total yt-dlp calls **9→5** for 2 videos); full gates green | this run |

## Net result
A 2-video watch-URL run dropped from **9 yt-dlp invocations to 5** (1 version + 2 metadata + 2 downloads). Each eliminated metadata fetch is a ~1.6 s real-network round-trip, so an N-video playlist saves ~2N redundant network calls and begins downloading immediately instead of after N serial resolve fetches. Two real correctness bugs fixed en route (playlist truncation at the capture cap; manifest-key invariant hardening). Manifest write volume halved.

## Follow-ups noted
- process.rs streaming/uncapped capture for very large playlist expansions (root-cause fix for the 4 MB cap; today's fix is an honest error).
- Manifest append-only journal if N>2000 ever matters (current full-rewrite is O(N²) bytes but modest at realistic scale).
