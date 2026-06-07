# Native YouTube extraction — feasibility & resume notes

**Status (2026-06-07): PAUSED, but the cipher is NOT the blocker.** The YouTube
path in `fw` currently orchestrates the external `yt-dlp` binary (`src/youtube/`).
The goal of a fully native, memory-safe Rust port (no Python `yt-dlp`) is sound
and the architecture below is decided. A spike against
[FrankenEngine](https://github.com/Dicklesworthstone/franken_engine) `main`
@ `8ca80bfc` proved its JS engine **runs the YouTube signature cipher correctly
today** (acceptance gate `decipherSig("0123456789") == "31204576"` passes). The
genuinely-blocked part is the **media download** layer (YouTube's 2025 SABR
streaming + PO-token attestation); the rest — InnerTube metadata/search/playlist
+ the sig/n cipher — is buildable natively now.

> **Correction:** an earlier draft of this memo said FrankenEngine "can't run the
> cipher." That was tested against a stale `v0.1.0` checkout (the clone was
> detached on the release tag, 228 commits behind `main`). Re-tested on `main`,
> the cipher works. See the corrected gap report:
> `docs/FRANKENENGINE_YOUTUBE_CIPHER_JS_GAP_REPORT.md`.

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

## Where the real blocker is — findings (corrected against `main`)

1. **FrankenEngine runs the cipher today (on `main`).** Its public
   `HybridRouter::eval` handles every cipher primitive — Array
   `reverse`/`splice`/`slice`/`join`, `String.split`/`fromCharCode`/`charCodeAt`,
   `for` loops, bitwise — plus `RegExp`, `String.replace`, `Array.map`+closures,
   `JSON`, `Math.imul`. The acceptance gate passes. So the sig/n cipher layer of
   the native port is **unblocked**. The only FrankenEngine gaps left are three
   *BotGuard*-specific features — **typed arrays, the `Function` constructor, and
   `try/catch`** — which block PO-token attestation but not the cipher. (Details +
   reproducible harness: `docs/FRANKENENGINE_YOUTUBE_CIPHER_JS_GAP_REPORT.md`.)

2. **YouTube moved past the cipher (2025 SABR + PO tokens) — this is the real
   blocker.** On a test video,
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

The cipher prerequisite is already met (FrankenEngine `main` passes the
acceptance gate), so the native port can resume *partially* whenever we want:
build the InnerTube client on `asupersync` + metadata/search/playlist + wire
FrankenEngine for sig/n. The part that stays blocked is **media download**:
- **SABR/UMP streaming** — a protocol to implement in Rust (JS-engine-independent).
- **PO-token (BotGuard)** — needs FrankenEngine to land the three remaining
  features (typed arrays, `Function` constructor, `try/catch`); track that in the
  gap report.

So: the metadata/cipher layer is a "resume now" decision; the download layer is
gated on (SABR implementation) + (BotGuard → 3 FrankenEngine features). Until
both are done, downloads fall back to `yt-dlp`.

## Until then

Keep building the YouTube *UX* on the working `yt-dlp` path — `fw youtube
search`/`enrich` (agent curation), batched backoff+jitter downloads, slug
naming, `fw doctor`, capabilities/robot-docs, ergonomics. None of that is wasted:
it's the same CLI surface the native engine will slot under later.
