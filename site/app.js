// franken_whisper playground driver: consent → verified downloads → fused
// transcribe+diarize → speaker-attributed transcript + exports.
import { MODELS, totalBytes } from "./model-manifest.js?v=@SITEV@";

const $ = (id) => document.getElementById(id);

const COMBINED_TOTAL = totalBytes("whisper") + totalBytes("sortformer");

const state = {
  worker: null,
  booted: false,
  modelsReady: false,
  busy: false,
  file: null,
  result: null,
  wallMs: 0,
  progressBase: { whisper: 0, sortformer: 0 },
};

const STAGE_LABELS = {
  "whisper:scan": "scanning the model's tensor directory…",
  "whisper:weights": "hydrating Whisper weights (f16-resident)…",
  "whisper:ready": "Whisper ready.",
  "sortformer:verify": "re-verifying the Sortformer package against its conversion receipt…",
  "sortformer:weights": "building the Sortformer graph…",
  "sortformer:ready": "Sortformer ready.",
  "audio:decode": "decoding audio (Symphonia) and resampling to 16 kHz…",
  "whisper:decode": "transcribing (large-v3-turbo, single wasm thread — this is the long part)…",
  "sortformer:diarize": "diarizing (Sortformer, 80 ms frames)…",
  "fuse:project": "fusing speakers onto the transcript…",
  done: "finishing…",
};

function setStatus(text, kind = "") {
  const el = $("status");
  el.textContent = text;
  el.className = `status ${kind}`;
}

function fmtBytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  return `${n} B`;
}

function fmtTime(sec) {
  if (sec == null || !Number.isFinite(sec)) return "--:--";
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function speakerClass(speaker) {
  const m = /^SPEAKER_(\d+)$/.exec(speaker ?? "");
  return m ? `spk-${Number(m[1]) % 4}` : "spk-x";
}

// ---- worker wiring ---------------------------------------------------------

function bootWorker() {
  const worker = new Worker("./engine-worker.js?v=@SITEV@", { type: "module" });
  state.worker = worker;
  worker.onmessage = (e) => handle(e.data);
  worker.onerror = (e) => setStatus(`worker error: ${e.message}`, "err");
}

function handle(m) {
  switch (m.type) {
    case "booted": {
      state.booted = true;
      setStatus(`engine booted (fw-wasm ${m.version}). Load the models to begin.`);
      $("load-models").disabled = false;
      break;
    }
    case "model-progress": {
      const bar = $("progress-bar");
      const text = $("progress-text");
      $("progress-wrap").hidden = false;
      // Combined progress across both models: whisper first, then sortformer.
      const base = m.model === "sortformer" ? totalBytes("whisper") : 0;
      const done = base + (m.loaded ?? 0);
      const pct = Math.min(100, Math.floor((done / COMBINED_TOTAL) * 100));
      bar.style.width = `${pct}%`;
      const phase = m.phase === "rehash" ? "resuming (re-hashing banked bytes)" : m.phase;
      text.textContent = `${m.file}: ${fmtBytes(m.loaded)} of ${fmtBytes(m.total)} (${phase}) — ${pct}% overall`;
      break;
    }
    case "stage": {
      const label = STAGE_LABELS[m.stage] ?? m.stage;
      setStatus(label);
      if (state.busy) {
        $("stage-line").hidden = false;
        $("stage-line").textContent = label;
      }
      break;
    }
    case "whisper-ready": {
      setStatus("Whisper loaded. Fetching the diarizer…");
      break;
    }
    case "sortformer-ready": {
      setStatus("Sortformer loaded.");
      break;
    }
    case "ready": {
      state.modelsReady = true;
      $("progress-wrap").hidden = true;
      $("load-models").disabled = true;
      $("load-models").querySelector("span").textContent = "Models loaded ✓";
      setStatus("Both models ready. Pick a recording.", "ok");
      maybeEnableRun();
      break;
    }
    case "result": {
      state.busy = false;
      state.result = m.result;
      state.wallMs = m.wall_ms;
      $("stage-line").hidden = true;
      renderResult();
      maybeEnableRun();
      break;
    }
    case "cache-cleared": {
      state.modelsReady = false;
      $("load-models").disabled = false;
      $("load-models").querySelector("span").textContent = "Load both models (2.0 GB)";
      setStatus("Cache cleared. Models will re-download on next load.");
      maybeEnableRun();
      break;
    }
    case "error": {
      state.busy = false;
      $("stage-line").hidden = true;
      setStatus(`error: ${m.message}`, "err");
      maybeEnableRun();
      break;
    }
    default:
      break;
  }
}

// ---- rendering -------------------------------------------------------------

function renderResult() {
  const r = state.result;
  const rt = r.audio_sec > 0 ? (state.wallMs / 1000 / r.audio_sec).toFixed(2) : "?";
  const speakers = new Set(
    r.speaker_segments.map((s) => s.speaker).filter((s) => s != null),
  );
  $("result-placeholder").hidden = true;
  $("result-wrap").hidden = false;
  $("result-meta").textContent =
    `${fmtTime(r.audio_sec)} of audio · ${(state.wallMs / 1000).toFixed(1)} s wall ` +
    `(${rt}× realtime on this machine) · ${speakers.size} speaker(s) · ` +
    `${r.turns.length} turns` +
    (r.dropped_windows > 0 ? ` · ⚠ ${r.dropped_windows} dropped window(s)` : "");

  const box = $("output");
  box.textContent = "";
  for (const seg of r.speaker_segments) {
    const row = document.createElement("div");
    row.className = "seg";
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = `${fmtTime(seg.start_sec)}–${fmtTime(seg.end_sec)}`;
    const spk = document.createElement("span");
    spk.className = `spk ${speakerClass(seg.speaker)}`;
    spk.textContent = seg.speaker ?? "UNKNOWN";
    const txt = document.createElement("span");
    txt.textContent = seg.text;
    row.append(t, spk, txt);
    box.appendChild(row);
  }
  if (!r.speaker_segments.length) box.textContent = "(no speech detected)";
  $("download-md").hidden = false;
  $("download-html").hidden = false;
  setStatus("Done.", "ok");
}

// ---- exports ---------------------------------------------------------------

function exportMd() {
  const r = state.result;
  const lines = [
    `# Transcript — ${state.file.name}`,
    "",
    `- Engine: franken_whisper wasm (Whisper large-v3-turbo + Sortformer diarization)`,
    `- Audio: ${r.audio_sec.toFixed(1)} s · transcribed + diarized in ${(state.wallMs / 1000).toFixed(1)} s in this browser`,
    r.dropped_windows > 0
      ? `- **Warning: ${r.dropped_windows} window(s) dropped without output**`
      : null,
    "",
  ].filter((l) => l !== null);
  for (const seg of r.speaker_segments) {
    lines.push(
      `**[${fmtTime(seg.start_sec)}–${fmtTime(seg.end_sec)}] ${seg.speaker ?? "UNKNOWN"}:** ${seg.text.trim()}`,
    );
    lines.push("");
  }
  download(`${state.file.name}.transcript.md`, lines.join("\n"), "text/markdown");
}

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const SPK_COLORS = ["#34d399", "#fbbf24", "#93c5fd", "#f9a8d4"];

function exportHtml() {
  const r = state.result;
  const rows = r.speaker_segments
    .map((seg) => {
      const m = /^SPEAKER_(\d+)$/.exec(seg.speaker ?? "");
      const color = m ? SPK_COLORS[Number(m[1]) % 4] : "#94a3b8";
      return `      <div class="seg"><span class="t">${fmtTime(seg.start_sec)}–${fmtTime(seg.end_sec)}</span> <span class="spk" style="color:${color};border-color:${color}66">${esc(seg.speaker ?? "UNKNOWN")}</span> ${esc(seg.text.trim())}</div>`;
    })
    .join("\n");
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transcript — ${esc(state.file.name)}</title>
<style>
  body{margin:0;background:#060b09;color:#d6e5dd;font:16px/1.75 ui-sans-serif,system-ui,sans-serif;padding:3rem 1.5rem}
  main{max-width:46rem;margin:0 auto}
  h1{font-size:1.3rem;color:#a7f3d0}
  .meta{color:#7d8f86;font-size:.85rem;margin-bottom:2rem}
  .seg{margin:.6rem 0}
  .t{color:#5f7a6e;font-family:ui-monospace,monospace;font-size:.75rem;margin-right:.5rem}
  .spk{font-family:ui-monospace,monospace;font-size:.72rem;font-weight:800;border:1px solid;border-radius:999px;padding:.08rem .5rem;margin-right:.5rem}
</style></head><body><main>
  <h1>Transcript — ${esc(state.file.name)}</h1>
  <div class="meta">franken_whisper wasm (large-v3-turbo + Sortformer) · ${r.audio_sec.toFixed(1)} s of audio · processed in ${(state.wallMs / 1000).toFixed(1)} s in the browser</div>
${rows}
</main></body></html>
`;
  download(`${state.file.name}.transcript.html`, html, "text/html");
}

function download(name, text, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ---- input wiring ----------------------------------------------------------

function maybeEnableRun() {
  $("run").disabled = !(state.modelsReady && state.file && !state.busy);
}

function acceptFile(file) {
  if (!file) return;
  state.file = file;
  $("drop-hint").textContent = `${file.name} (${fmtBytes(file.size)}) — ready.`;
  maybeEnableRun();
}

function init() {
  $("license").textContent =
    `${MODELS.whisper.label}: ${MODELS.whisper.license} · ${MODELS.sortformer.label}: ${MODELS.sortformer.license}`;

  $("load-models").addEventListener("click", () => {
    $("consent").hidden = false;
    $("load-models").disabled = true;
  });
  $("consent-yes").addEventListener("click", () => {
    $("consent").hidden = true;
    setStatus("downloading models (verified, resumable — a reload picks up where it left off)…");
    state.worker.postMessage({ type: "load-models" });
  });
  $("consent-no").addEventListener("click", () => {
    $("consent").hidden = true;
    $("load-models").disabled = false;
  });
  $("clear-cache").addEventListener("click", () => {
    state.worker.postMessage({ type: "clear-cache" });
  });

  const drop = $("drop");
  const input = $("file-input");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") input.click();
  });
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("hover");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("hover");
    acceptFile(e.dataTransfer?.files?.[0]);
  });
  input.addEventListener("change", () => acceptFile(input.files?.[0]));

  $("run").addEventListener("click", async () => {
    if (!state.file || state.busy) return;
    state.busy = true;
    maybeEnableRun();
    $("download-md").hidden = true;
    $("download-html").hidden = true;
    setStatus("running…");
    const ext = (state.file.name.split(".").pop() || "").toLowerCase();
    const buf = await state.file.arrayBuffer();
    state.worker.postMessage({ type: "transcribe", audio: buf, ext }, [buf]);
  });

  $("download-md").addEventListener("click", exportMd);
  $("download-html").addEventListener("click", exportHtml);

  bootWorker();
}

init();
