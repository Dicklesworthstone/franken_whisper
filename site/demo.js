// franken_whisper in-browser demo: pick an mp3/m4a/wav, transcribe with the
// wasm engine (tiny.en, serial build), export .md or styled .html.
const $ = (id) => document.getElementById(id);

const state = {
  worker: null,
  ready: false,
  busy: false,
  fileName: null,
  result: null,
  wallMs: 0,
};

function setStatus(text, kind = "info") {
  const el = $("demo-status");
  el.textContent = text;
  el.dataset.kind = kind;
}

function fmtTime(sec) {
  if (sec == null) return "--:--";
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}.${String(Math.floor((s % 1) * 10))}`;
}

function renderSegments(result) {
  const box = $("demo-output");
  box.innerHTML = "";
  for (const seg of result.segments) {
    const row = document.createElement("div");
    row.className = "demo-seg";
    const t = document.createElement("span");
    t.className = "demo-seg-time";
    t.textContent = `[${fmtTime(seg.start_sec)} → ${fmtTime(seg.end_sec)}]`;
    const txt = document.createElement("span");
    txt.textContent = seg.text;
    row.append(t, txt);
    box.appendChild(row);
  }
  if (!result.segments.length) {
    box.textContent = "(no speech detected)";
  }
}

// ---- exports -------------------------------------------------------------

function markdownExport() {
  const { result, fileName, wallMs } = state;
  const lines = [
    `# Transcript — ${fileName}`,
    "",
    `- Engine: franken_whisper wasm (tiny.en, serial build)`,
    `- Audio: ${result.audio_sec.toFixed(1)} s · transcribed in ${(wallMs / 1000).toFixed(1)} s in this browser`,
    result.dropped_windows > 0 ? `- **Warning: ${result.dropped_windows} window(s) dropped without output**` : null,
    "",
  ].filter((l) => l !== null);
  for (const seg of result.segments) {
    lines.push(`**[${fmtTime(seg.start_sec)} → ${fmtTime(seg.end_sec)}]** ${seg.text.trim()}`);
    lines.push("");
  }
  return lines.join("\n");
}

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function htmlExport() {
  const { result, fileName, wallMs } = state;
  const rows = result.segments
    .map(
      (seg) =>
        `      <div class="seg"><span class="t">[${fmtTime(seg.start_sec)} → ${fmtTime(seg.end_sec)}]</span> ${esc(seg.text.trim())}</div>`,
    )
    .join("\n");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transcript — ${esc(fileName)}</title>
<style>
  body{margin:0;background:#04100a;color:#d1fae5;font:16px/1.7 ui-sans-serif,system-ui,sans-serif;padding:3rem 1.5rem}
  main{max-width:46rem;margin:0 auto}
  h1{font-size:1.3rem;color:#a7f3d0}
  .meta{color:#64748b;font-size:.85rem;margin-bottom:2rem}
  .seg{margin:.6rem 0}
  .t{color:#10b981;font-family:ui-monospace,monospace;font-size:.8rem;margin-right:.5rem}
</style></head><body><main>
  <h1>Transcript — ${esc(fileName)}</h1>
  <div class="meta">franken_whisper wasm (tiny.en) · ${result.audio_sec.toFixed(1)} s of audio · transcribed in ${(wallMs / 1000).toFixed(1)} s in the browser</div>
${rows}
</main></body></html>
`;
}

function download(name, text, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ---- wiring --------------------------------------------------------------

function bootWorker() {
  const worker = new Worker("fw-worker.js", { type: "module" });
  state.worker = worker;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "booted") {
      setStatus(`engine booted (fw-wasm ${m.version}) — fetching tiny.en weights…`);
      worker.postMessage({ type: "init", id: 0 });
    } else if (m.type === "model-progress") {
      const pct = Math.floor((m.loaded / m.total) * 100);
      setStatus(m.cached ? "verified cached weights" : `downloading tiny.en: ${pct}% of 78 MB`);
    } else if (m.type === "ready") {
      state.ready = true;
      setStatus(`ready — model loaded in ${(m.load_ms / 1000).toFixed(1)} s. Pick an audio file.`, "ok");
      $("demo-file").disabled = false;
    } else if (m.type === "result") {
      state.busy = false;
      state.result = m.result;
      state.wallMs = m.wall_ms;
      const rt = m.result.audio_sec > 0 ? (m.wall_ms / 1000 / m.result.audio_sec).toFixed(2) : "?";
      setStatus(
        `done: ${m.result.audio_sec.toFixed(1)} s of audio in ${(m.wall_ms / 1000).toFixed(1)} s (${rt}x realtime, measured on this device)`,
        "ok",
      );
      renderSegments(m.result);
      $("demo-export-md").disabled = false;
      $("demo-export-html").disabled = false;
      $("demo-file").disabled = false;
    } else if (m.type === "error") {
      state.busy = false;
      setStatus(`error: ${m.message}`, "err");
      $("demo-file").disabled = !state.ready;
    }
  };
  worker.onerror = (e) => setStatus(`worker error: ${e.message}`, "err");
}

function init() {
  const start = $("demo-start");
  start.addEventListener("click", () => {
    start.disabled = true;
    setStatus("booting wasm engine…");
    bootWorker();
  });

  $("demo-file").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file || state.busy) return;
    state.busy = true;
    state.fileName = file.name;
    $("demo-file").disabled = true;
    $("demo-export-md").disabled = true;
    $("demo-export-html").disabled = true;
    setStatus(`transcribing ${file.name}… (single-threaded wasm; long files take a while)`);
    $("demo-output").textContent = "";
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    const buf = await file.arrayBuffer();
    state.worker.postMessage({ type: "transcribe", id: 1, audio: buf, ext }, [buf]);
  });

  $("demo-export-md").addEventListener("click", () => {
    download(`${state.fileName}.transcript.md`, markdownExport(), "text/markdown");
  });
  $("demo-export-html").addEventListener("click", () => {
    download(`${state.fileName}.transcript.html`, htmlExport(), "text/html");
  });
}

init();
