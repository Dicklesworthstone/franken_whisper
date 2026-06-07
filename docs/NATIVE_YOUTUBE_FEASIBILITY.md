# Native YouTube extraction — feasibility & resume notes

**Status (2026-06-07): PAUSED, blocked on FrankenEngine.** The YouTube path in
`fw` currently orchestrates the external `yt-dlp` binary (`src/youtube/`). The
goal of a fully native, memory-safe Rust port (no Python `yt-dlp`) is sound and
the architecture below is decided — but it is gated on the
[FrankenEngine](https://github.com/Dicklesworthstone/franken_engine) JavaScript
engine maturing first. A one-afternoon spike proved the dependency isn't ready,
so we paused before sinking weeks into a foundation that can't yet run.

## Decided architecture (for when we resume)

Native, memory-safe Rust, behind a `native-ytdl` cargo feature (planned;
default OFF so `fw`'s normal build stays lean). The existing `yt-dlp`
orchestration stays as the `FRANKEN_WHISPER_YTDL=ytdlp` fallback (also planned —
today the code always uses the `yt-dlp` orchestration; only the
`FRANKEN_WHISPER_YTDLP_BIN` *binary-path* override exists).

- **Networking:** `asupersync` HTTP (no reqwest/hyper).
- **JavaScript (cipher + BotGuard):** **FrankenEngine** (`frankenengine-engine`,
  `HybridRouter::eval`), `default-features = false` — required today because its
  default `asupersync-integration` feature fails to compile against
  `franken-decision` 0.3.2 (the same `update_posterior` `E0053` drift
  `franken_whisper` itself migrated; see the gap report §6). We want only the JS
  engine anyway, not the decision/evidence control plane. Not boa, not V8.
- **Extraction logic:** port yt-dlp's `youtube` extractor + the InnerTube
  protocol to Rust. The *volatile* knowledge (sig-function-extraction regexes,
  client versions/keys) can optionally be synced from yt-dlp's source so we ride
  the community's fixes without running their Python — data in, native execution.
- **Self-healing cipher:** fetch the *live* `base.js` from YouTube at runtime and
  run it through FrankenEngine, so a weekly cipher *value/rotation* change just
  works — we never hardcode the cipher transform. (A *structural* base.js change
  still needs an update to the extraction pattern that locates the function in
  base.js, which is the volatile "knowledge" we sync per the bullet above.)

## Why it's paused — two findings

1. **FrankenEngine can't run the cipher yet.** Its public `HybridRouter::eval`
   runs literals, function-expression calls, and object-method dispatch, but is
   **missing Array methods (`reverse`/`splice`/`slice`/`join`), `String.split`,
   `String.fromCharCode`, and working loops** (a 5-iteration `for` exhausts the
   100 000-instruction budget). The YouTube signature cipher is
   `split("")` → a fixed sequence of helper transforms (`reverse`/`splice`/swap)
   → `join("")` — i.e. exactly the missing Array/String builtins — so it cannot
   run today. (Explicit `for`-loops, also broken here, dominate the `n`-transform
   and BotGuard.) BotGuard (for PO tokens) is far harder JS than the cipher.
   - The full gap analysis, reproducible harness, prioritized work items, and a
     verified acceptance gate live in the FrankenEngine repo at
     **`YOUTUBE_CIPHER_JS_GAP_REPORT.md`** (handed to the FrankenEngine agents).

2. **YouTube moved past the cipher (2025 SABR + PO tokens).** On a test video,
   `yt-dlp`'s own verbose log shows `bestaudio` resolving to **no plain GET-able
   URL** (`yt-dlp -f bestaudio --print url` printed empty), with "YouTube is
   forcing SABR streaming" and "Detected experiment to bind GVS PO Token" —
   consistent with the broad 2025 SABR rollout (yt-dlp issue #12482). `yt-dlp`
   still *downloads* such videos because it implements SABR + PO-token handling
   internally — which is precisely why it remains the reliable fallback. A native
   port that stops at "InnerTube + sig/n cipher → GET url" would *not* download
   them; the full path *also* needs **SABR/UMP streaming** and **PO-token
   (BotGuard) attestation** — the hardest, most volatile code in the extraction
   world. The user chose the full-native scope anyway (in-ethos), which makes a
   capable JS engine (for BotGuard) the linchpin — hence FrankenEngine first.

## Resume condition

Resume the native port when FrankenEngine passes the acceptance gate in its
`YOUTUBE_CIPHER_JS_GAP_REPORT.md` (the §5 `decipherSig("0123456789") ==
"31204576"` milestone, then the live-base.js stretch). At that point: build the
InnerTube client on `asupersync`, wire FrankenEngine for sig/n, then tackle the
SABR + PO-token layer (the genuinely hard part) as its own epic.

## Until then

Keep building the YouTube *UX* on the working `yt-dlp` path — `fw youtube
search`/`enrich` (agent curation), batched backoff+jitter downloads, slug
naming, `fw doctor`, capabilities/robot-docs, ergonomics. None of that is wasted:
it's the same CLI surface the native engine will slot under later.
