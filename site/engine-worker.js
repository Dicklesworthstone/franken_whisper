// franken_whisper engine worker: the whole pipeline lives here — verified
// model downloads (loader.js → OPFS), the wasm engine (whisper large-v3-turbo
// + Sortformer diarization, one code path with the native CLI), and the
// fused transcribe+diarize call.
//
// Module workers silently DROP messages posted before `onmessage` exists, and
// any top-level `await import()` opens that window. So: install a buffering
// handler FIRST, then import, then drain.
const pending = [];
let ready = false;

// A wasm trap surfaces as an opaque `RuntimeError: unreachable`. The engine's
// panic hook banks the real message in `globalThis.__fwLastPanic` first; an
// allocation failure aborts without any message, which on this engine means
// the 4 GB wasm ceiling — say so instead of shrugging.
function describeError(err) {
  const raw = String(err?.message ?? err);
  if (!/unreachable/i.test(raw)) return raw;
  const panic = globalThis.__fwLastPanic;
  delete globalThis.__fwLastPanic;
  if (panic) return `engine panic: ${panic}`;
  return (
    "the engine ran out of memory (wasm's 4 GB ceiling) or hit an internal trap. " +
    "Try a shorter recording; details, if any, are in the browser console."
  );
}

self.onmessage = (e) => {
  if (!ready) pending.push(e);
  else route(e).catch((err) => post("error", { message: describeError(err) }));
};

function post(type, data) {
  self.postMessage({ type, ...(data ?? {}) });
}

let wasm = null;
let loaderMod = null;
let manifest = null;
let whisperHandle = null; // FileSystemSyncAccessHandle over the ggml file
let whisperReady = false;
let sortformerReady = false;

// Feed the engine's host-fed clock before every re-entry (std::time::Instant
// is a trap on wasm32; plat.rs reads this).
function feedClock() {
  wasm.set_now_micros(Math.trunc(performance.now() * 1000));
}

// The engine's streamed ggml loader pulls tensor payloads through this hook:
// a SYNCHRONOUS positioned read against the OPFS access handle. This is why
// the 1.5 GB model never exists as one blob on either side of the boundary.
globalThis.__fwModelReadAt = (offset, len) => {
  if (!whisperHandle) throw new Error("whisper OPFS handle not open");
  const buf = new Uint8Array(len);
  const got = whisperHandle.read(buf, { at: offset });
  if (got !== len) {
    throw new Error(`OPFS read at ${offset}: wanted ${len}, got ${got}`);
  }
  return buf;
};

// Engine stage callbacks (named stages, e.g. "whisper:weights",
// "sortformer:diarize") — relayed so the page can narrate long operations.
globalThis.__fwStage = (name) => post("stage", { stage: name });

// Live transcript: each finished 30-second window's segments stream to the
// page as they land, so words appear while the run is still going.
globalThis.__fwSegments = (json) => {
  try {
    post("partial-segments", { segments: JSON.parse(json) });
  } catch {
    /* a malformed partial is display-only; the final result is authoritative */
  }
};

// Engine perf spans double as the progress heartbeat: "encoder_window" fires
// once per 30-second window (the page counts them against the window total),
// "sortformer_tick" is the diarizer's throttled liveness signal. Forward the
// two the page uses, rate-limited so a fast run cannot flood the channel.
let lastTickForward = 0;
globalThis.__fwSpan = (name, ms) => {
  if (name === "encoder_window") {
    post("span", { span: name, ms });
  } else if (name === "sortformer_tick") {
    const now = performance.now();
    if (now - lastTickForward > 250) {
      lastTickForward = now;
      post("span", { span: name, ms });
    }
  }
};

async function ensureWithProgress(modelId) {
  return loaderMod.ensureModel(modelId, (p) => post("model-progress", { model: modelId, ...p }));
}

async function loadWhisper() {
  if (whisperReady) return;
  const dir = await ensureWithProgress("whisper");
  const file = await dir.getFileHandle(manifest.MODELS.whisper.weights.name);
  whisperHandle = await file.createSyncAccessHandle();
  let info;
  try {
    feedClock();
    info = JSON.parse(wasm.load_whisper_streamed(manifest.MODELS.whisper.weights.bytes));
  } finally {
    // Close on BOTH paths: a leaked sync handle makes every retry fail with
    // "access handle already open" until the worker restarts.
    whisperHandle.close();
    whisperHandle = null;
  }
  whisperReady = true;
  post("whisper-ready", { info });
}

let denoiserReady = false;
async function loadDenoiser() {
  if (denoiserReady) return;
  const dir = await ensureWithProgress("denoiser");
  const file = await (await dir.getFileHandle("fastenhancer-s-48k-denoise.safetensors")).getFile();
  feedClock();
  wasm.load_denoiser(new Uint8Array(await file.arrayBuffer()));
  denoiserReady = true;
}

async function loadSortformer() {
  if (sortformerReady) return;
  const dir = await ensureWithProgress("sortformer");
  const weightsFile = await (await dir.getFileHandle("weights.safetensors")).getFile();
  const receiptFile = await (await dir.getFileHandle("conversion-receipt.json")).getFile();
  const weights = new Uint8Array(await weightsFile.arrayBuffer());
  const receipt = new Uint8Array(await receiptFile.arrayBuffer());
  feedClock();
  const info = JSON.parse(wasm.load_sortformer(receipt, weights));
  sortformerReady = true;
  post("sortformer-ready", { info });
}

async function route(e) {
  const m = e.data;
  switch (m.type) {
    case "load-models": {
      // The pool arms FIRST: the engine's model load touches rayon's global
      // pool, which can only initialize once — arming later fails with
      // "already initialized". Model loading itself never runs on the pool
      // (tensor reads go through this worker's thread-confined OPFS handle;
      // the engine keeps its load on the calling thread on wasm), so the
      // parked workers only ever serve compute.
      await armThreadPool();
      await loadWhisper();
      await loadSortformer();
      await loadDenoiser();
      post("ready", { version: wasm.version(), lane, threads });
      break;
    }
    case "transcribe": {
      if (!whisperReady || !sortformerReady) {
        throw new Error("models not loaded yet");
      }
      feedClock();
      const t0 = performance.now();
      // Two stages so the page can build its progress model (window total,
      // remaining-time estimate) from the decoded duration before the long
      // stage starts.
      const meta = JSON.parse(
        wasm.decode_audio(new Uint8Array(m.audio), m.ext ?? "", m.denoise !== false),
      );
      post("audio-meta", {
        audio_sec: meta.audio_sec,
        skipped_leading_sec: meta.skipped_leading_sec ?? 0,
        denoised: meta.denoised === true,
      });
      feedClock();
      const result = JSON.parse(wasm.run_prepared(m.prompt ?? undefined, m.language ?? undefined));
      feedClock();
      post("result", { result, wall_ms: Math.round(performance.now() - t0) });
      break;
    }
    case "clear-cache": {
      await loaderMod.clearModels();
      whisperReady = false;
      sortformerReady = false;
      post("cache-cleared", {});
      break;
    }
    default:
      throw new Error(`unknown message type: ${m.type}`);
  }
}

// ---- lane selection --------------------------------------------------------
//
// `pkg-threaded` is linked `--shared-memory --import-memory` with atomics and
// carries wasm-bindgen-rayon's worker pool. Sharedness is a LINK-TIME
// property, so this is a choice of module, not a flag. The choice is an
// ALLOW-LIST, not feature detection: SharedArrayBuffer also exists on
// iOS/WebKit under COOP/COEP, but growing a shared memory toward 2 GB kills
// the tab there, and this engine holds 2.4 GB of weights. Blink only, and
// only when actually cross-origin isolated.
function blinkAllowsThreads() {
  if (typeof SharedArrayBuffer !== "function") return false;
  if (self.crossOriginIsolated !== true) return false;
  const ua = self.navigator?.userAgent ?? "";
  // Every iOS browser is WebKit under the skin, whatever the brand in the UA.
  if (/iPhone|iPad|iPod|CriOS|FxiOS|EdgiOS/.test(ua)) return false;
  return /Chrom(e|ium)\/\d/.test(ua);
}

// The rayon workers spawn from a blob URL of wasm-bindgen-rayon's helper
// script. A `worker-src` CSP without `blob:` kills that — and it would kill
// it AFTER the 2 GB models are staged, when falling back is ruinous. Prove
// the blob-module-worker path with a byte-sized canary at boot, before a
// single weight byte is downloaded.
async function blobModuleWorkerWorks() {
  let url;
  try {
    url = URL.createObjectURL(new Blob(["self.postMessage('ok');"], { type: "text/javascript" }));
    const w = new Worker(url, { type: "module" });
    try {
      await new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error("canary timeout")), 3000);
        w.onmessage = () => {
          clearTimeout(t);
          resolve();
        };
        w.onerror = (e) => {
          clearTimeout(t);
          reject(e);
        };
      });
      return true;
    } finally {
      w.terminate();
    }
  } catch {
    return false;
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
}

let lane = "serial";
let threads = 1;

function targetThreads() {
  return Math.max(1, Math.min((self.navigator?.hardwareConcurrency ?? 4) - 1, 8));
}

// Called from route("load-models") AFTER both models hydrate: a rayon worker
// parked in `Atomics.wait` blocks a shared-memory `grow`, and everything up
// to that point is nothing but grows. Arming after hydration costs nothing.
async function armThreadPool() {
  if (lane !== "threaded" || threads > 1) return;
  post("stage", { stage: "threads:arming" });
  await wasm.initThreadPool(targetThreads());
  threads = wasm.thread_count();
  post("stage", { stage: `threads:armed (${threads})` });
}

// Import AFTER the buffering handler is installed.
(async () => {
  try {
    const wantThreads = blinkAllowsThreads() && (await blobModuleWorkerWorks());
    const pkgDir = wantThreads ? "./pkg-threaded" : "./pkg";
    const [engine, loader, mf] = await Promise.all([
      import(`${pkgDir}/fw_wasm.js?v=@SITEV@`),
      import("./loader.js?v=@SITEV@"),
      import("./model-manifest.js?v=@SITEV@"),
    ]);
    await engine.default(); // instantiate the wasm module
    // The build flags are a claim; the instantiated memory is the receipt. A
    // "threaded" module whose memory is a plain ArrayBuffer would run, run
    // single-threaded, and never say so.
    if (wantThreads) {
      if (!(engine.wasm_memory().buffer instanceof SharedArrayBuffer)) {
        throw new Error("pkg-threaded instantiated with a non-shared memory");
      }
      lane = "threaded";
    }
    wasm = engine;
    loaderMod = loader;
    manifest = mf;
    ready = true;
    post("booted", { version: wasm.version(), lane });
    for (const queued of pending.splice(0)) {
      route(queued).catch((err) => post("error", { message: describeError(err) }));
    }
  } catch (err) {
    post("error", { message: `engine failed to boot: ${String(err?.message ?? err)}` });
  }
})();
