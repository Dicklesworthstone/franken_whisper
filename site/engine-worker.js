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
self.onmessage = (e) => {
  if (!ready) pending.push(e);
  else route(e).catch((err) => post("error", { message: String(err?.message ?? err) }));
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

async function ensureWithProgress(modelId) {
  return loaderMod.ensureModel(modelId, (p) => post("model-progress", { model: modelId, ...p }));
}

async function loadWhisper() {
  if (whisperReady) return;
  const dir = await ensureWithProgress("whisper");
  const file = await dir.getFileHandle(manifest.MODELS.whisper.weights.name);
  whisperHandle = await file.createSyncAccessHandle();
  feedClock();
  const info = JSON.parse(wasm.load_whisper_streamed(manifest.MODELS.whisper.weights.bytes));
  // Tensors are hydrated (f16-resident); the sync handle is no longer needed.
  whisperHandle.close();
  whisperHandle = null;
  whisperReady = true;
  post("whisper-ready", { info });
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
      // Whisper first (the big one — its progress dominates), then the
      // diarizer; the page shows one combined progress bar.
      await loadWhisper();
      await loadSortformer();
      post("ready", { version: wasm.version() });
      break;
    }
    case "transcribe": {
      if (!whisperReady || !sortformerReady) {
        throw new Error("models not loaded yet");
      }
      feedClock();
      const t0 = performance.now();
      const result = JSON.parse(
        wasm.transcribe_and_diarize(new Uint8Array(m.audio), m.ext ?? ""),
      );
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

// Import AFTER the buffering handler is installed.
(async () => {
  try {
    const [engine, loader, mf] = await Promise.all([
      import("./pkg/fw_wasm.js?v=@SITEV@"),
      import("./loader.js?v=@SITEV@"),
      import("./model-manifest.js?v=@SITEV@"),
    ]);
    await engine.default(); // instantiate the wasm module
    wasm = engine;
    loaderMod = loader;
    manifest = mf;
    ready = true;
    post("booted", { version: wasm.version() });
    for (const queued of pending.splice(0)) {
      route(queued).catch((err) => post("error", { message: String(err?.message ?? err) }));
    }
  } catch (err) {
    post("error", { message: `engine failed to boot: ${String(err?.message ?? err)}` });
  }
})();
