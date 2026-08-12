// franken_whisper wasm engine worker (bd-m2jm, W3).
//
// Module workers silently DROP messages posted before `onmessage` exists, and
// any top-level `await import()` opens that window. So: install a buffering
// handler FIRST, then import, then drain.
const pending = [];
let ready = false;
self.onmessage = (e) => {
  if (!ready) pending.push(e);
  else route(e);
};

const MODEL = {
  url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
  file: "ggml-tiny.en.bin",
  bytes: 77704715,
  sha256: "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
};

let wasm = null;

// Feed the engine's host-fed clock before every re-entry so perf spans read
// real durations (std::time::Instant is a trap on wasm32).
function feedClock() {
  wasm.set_now_micros(Math.trunc(performance.now() * 1000));
}

function post(type, data) {
  self.postMessage({ type, ...data });
}

async function sha256Hex(buf) {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// tiny.en (78 MB): fetch → OPFS (one writable held open — WebKit copies the
// whole file per createWritable open) → SHA-256 pin verify → bytes.
// A cached file re-verifies by size + full digest (~0.3 s for 78 MB; fine at
// this size — the endpoint-hash fast path only earns its complexity at GBs).
async function modelBytes() {
  const root = await navigator.storage.getDirectory();
  try {
    const handle = await root.getFileHandle(MODEL.file);
    const file = await handle.getFile();
    if (file.size === MODEL.bytes) {
      const buf = await file.arrayBuffer();
      if ((await sha256Hex(buf)) === MODEL.sha256) {
        post("model-progress", { loaded: MODEL.bytes, total: MODEL.bytes, cached: true });
        return new Uint8Array(buf);
      }
    }
    await root.removeEntry(MODEL.file); // wrong size or digest: refetch
  } catch {
    /* not cached yet */
  }

  const resp = await fetch(MODEL.url);
  if (!resp.ok) throw new Error(`model fetch failed: HTTP ${resp.status}`);
  const declared = Number(resp.headers.get("content-length") || MODEL.bytes);
  if (declared > MODEL.bytes * 1.05) {
    throw new Error(`model fetch size ${declared} exceeds pinned ${MODEL.bytes}`);
  }
  const handle = await root.getFileHandle(MODEL.file, { create: true });
  const writable = await handle.createWritable(); // held open for the whole download
  const reader = resp.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    await writable.write(value);
    loaded += value.byteLength;
    if (loaded > MODEL.bytes) {
      await writable.abort();
      throw new Error("model stream exceeded pinned byte length");
    }
    post("model-progress", { loaded, total: MODEL.bytes, cached: false });
  }
  await writable.close();

  const buf = await (await handle.getFile()).arrayBuffer();
  const digest = await sha256Hex(buf);
  if (buf.byteLength !== MODEL.bytes || digest !== MODEL.sha256) {
    await root.removeEntry(MODEL.file);
    throw new Error(`model verification failed: got ${digest.slice(0, 12)}…, want ${MODEL.sha256.slice(0, 12)}…`);
  }
  return new Uint8Array(buf);
}

async function route(e) {
  const { type, id } = e.data;
  try {
    if (type === "init") {
      const bytes = await modelBytes();
      feedClock();
      const t0 = performance.now();
      const info = JSON.parse(wasm.load_model(bytes));
      post("ready", { id, info, load_ms: Math.round(performance.now() - t0), version: wasm.version() });
    } else if (type === "transcribe") {
      feedClock();
      const t0 = performance.now();
      const result = JSON.parse(
        wasm.transcribe_audio(new Uint8Array(e.data.audio), e.data.ext, true),
      );
      feedClock();
      post("result", { id, result, wall_ms: Math.round(performance.now() - t0) });
    } else {
      throw new Error(`unknown message type: ${type}`);
    }
  } catch (err) {
    post("error", { id, message: String(err?.message ?? err) });
  }
}

// Import AFTER the buffering handler is installed.
(async () => {
  try {
    const mod = await import("./pkg/fw_wasm.js");
    await mod.default(); // instantiate
    wasm = mod;
    ready = true;
    post("booted", { version: wasm.version() });
    for (const e of pending.splice(0)) route(e);
  } catch (err) {
    post("error", { id: null, message: `engine failed to boot: ${String(err?.message ?? err)}` });
  }
})();
