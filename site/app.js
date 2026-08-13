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
  nameMap: new Map(),
  lane: "serial",
  // Live-run progress model (see the "run progress" section below).
  run: null,
  // Download-rate estimator (EMA of bytes/second).
  dl: { lastAt: 0, lastLoaded: 0, rate: 0, lastFile: null },
};

// Calibration learned from completed runs on THIS machine, persisted so the
// second run's estimate is good from the first second. Values are seconds of
// wall clock per second of audio for the two long stages. Keyed per engine
// lane: the threaded and serial builds differ ~5x, and a browser update can
// flip which lane a machine gets.
function calKey() {
  return `fw-run-calibration-v1-${state.lane ?? "serial"}`;
}
function loadCalibration() {
  try {
    const c = JSON.parse(localStorage.getItem(calKey()) ?? "null");
    if (c && Number.isFinite(c.whisperRate) && Number.isFinite(c.sortformerRate)) return c;
  } catch {
    /* fall through */
  }
  // First-run defaults measured on an M-series desktop: threaded lane 1.56x
  // realtime overall, serial 8.35x; the diarizer measured ~0.11x of audio.
  return state.lane === "threaded"
    ? { whisperRate: 1.6, sortformerRate: 0.1 }
    : { whisperRate: 8.0, sortformerRate: 0.12 };
}
function saveCalibration(cal) {
  try {
    localStorage.setItem(calKey(), JSON.stringify(cal));
  } catch {
    /* storage may be denied; the defaults still work */
  }
}

const STAGE_LABELS = {
  "whisper:scan": "scanning the model's tensor directory…",
  "whisper:weights": "hydrating Whisper weights (f16-resident)…",
  "whisper:ready": "Whisper ready.",
  "sortformer:verify": "re-verifying the Sortformer package against its conversion receipt…",
  "sortformer:weights": "building the Sortformer graph…",
  "sortformer:ready": "Sortformer ready.",
  "audio:decode": "decoding audio (Symphonia) and resampling to 16 kHz…",
  "whisper:decode": "transcribing with large-v3-turbo (the long part)…",
  "sortformer:diarize": "diarizing (Sortformer, 80 ms frames)…",
  "fuse:project": "fusing speakers onto the transcript…",
  "threads:arming": "starting the worker pool…",
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

// ---- speaker naming --------------------------------------------------------

// Parse the optional "names/titles" field: comma- or newline-separated,
// trimmed, empties dropped.
function parseSpeakerNames(raw) {
  return (raw ?? "")
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

// Assign the user's names to detected SPEAKER_NN labels in order of first
// appearance in the merged transcript. Returns a Map from raw label to
// display name; labels beyond the provided names keep their raw form.
function buildSpeakerNameMap(result, names) {
  const map = new Map();
  if (!names.length) return map;
  for (const seg of result.speaker_segments) {
    const label = seg.speaker;
    if (label == null || map.has(label)) continue;
    if (map.size < names.length) map.set(label, names[map.size]);
  }
  return map;
}

function displaySpeaker(label, nameMap) {
  if (label == null) return "UNKNOWN";
  return nameMap.get(label) ?? label;
}

// ---- run progress ----------------------------------------------------------
//
// The engine emits one "encoder_window" span per 30-second window, in order,
// and whisper's decode is by far the dominant stage. The model:
//   whisper remaining  = windows_left × observed mean seconds per window
//                        (calibrated rate until the first window lands)
//   sortformer         = audio_sec × calibrated rate (ticks prove liveness)
//   bar percent        = elapsed / (elapsed + remaining estimate)
// Every number shown is measured on this machine; the first-run defaults say
// "about" and correct themselves as windows complete.

function fmtDur(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "…";
  const s = Math.round(sec);
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
}

function beginRun(audioSec) {
  const cal = loadCalibration();
  state.run = {
    audioSec,
    cal,
    startedAt: performance.now(),
    phase: "whisper",
    windowsTotal: Math.max(1, Math.ceil(audioSec / 30)),
    windowsDone: 0,
    lastWindowAt: performance.now(),
    windowSecs: [],
    sortStartedAt: null,
  };
  // The live panel takes over the transcript column (feedback belongs where
  // the result will land, next to the button that started it — not in the
  // model-download area at the top of the page).
  $("result-placeholder").hidden = true;
  $("result-wrap").hidden = true;
  const live = $("live-transcript");
  live.textContent = "";
  live.hidden = false;
  appendListeningCue();
  $("run-panel").hidden = false;
  $("run-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  updateRunProgress();
}

// The pulsing cue at the bottom of the live stream: proof of life between
// windows, replaced each time real words arrive.
function appendListeningCue() {
  const cue = document.createElement("div");
  cue.className = "seg";
  cue.id = "live-cue";
  const span = document.createElement("span");
  span.className = "listening";
  span.textContent = "decoding";
  cue.appendChild(span);
  $("live-transcript").appendChild(cue);
}

function appendLiveSegments(segments) {
  const live = $("live-transcript");
  if (live.hidden) return;
  document.getElementById("live-cue")?.remove();
  const offset = state.run?.offsetSec ?? 0;
  for (const seg of segments) {
    const row = document.createElement("div");
    row.className = "seg";
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = `${fmtTime((seg.start_sec ?? 0) + offset)}–${fmtTime((seg.end_sec ?? 0) + offset)}`;
    const txt = document.createElement("span");
    txt.textContent = seg.text;
    row.append(t, txt);
    live.appendChild(row);
  }
  appendListeningCue();
  live.scrollTop = live.scrollHeight;
}

const PHASE_LABELS = {
  whisper: "Transcribing with large-v3-turbo",
  sortformer: "Identifying the speakers (Sortformer)",
  fuse: "Attributing lines to speakers",
};

function runEstimates(run) {
  const now = performance.now();
  const elapsed = (now - run.startedAt) / 1000;
  // Seconds of audio each window covers (the last window is usually short).
  const secPerWindow = run.audioSec / run.windowsTotal;
  const meanWindow =
    run.windowSecs.length > 0
      ? run.windowSecs.reduce((a, b) => a + b, 0) / run.windowSecs.length
      : run.cal.whisperRate * secPerWindow;
  const sortTotal = run.cal.sortformerRate * run.audioSec;
  let remaining;
  if (run.phase === "whisper") {
    const windowsLeft = Math.max(0, run.windowsTotal - run.windowsDone);
    // The in-flight window's spent time is not subtracted; the estimate
    // stays a little conservative rather than optimistic.
    remaining = windowsLeft * meanWindow + sortTotal;
  } else if (run.phase === "sortformer") {
    const sortElapsed = (now - (run.sortStartedAt ?? now)) / 1000;
    remaining = Math.max(0, sortTotal - sortElapsed);
  } else {
    remaining = 0;
  }
  return { elapsed, remaining };
}

function updateRunProgress() {
  const run = state.run;
  if (!run) return;
  const { elapsed, remaining } = runEstimates(run);
  const pct = Math.min(99, Math.floor((elapsed / Math.max(1e-6, elapsed + remaining)) * 100));
  $("run-pct").textContent = `${pct}%`;
  $("run-bar-fill").style.width = `${pct}%`;
  $("run-stage").textContent = PHASE_LABELS[run.phase] ?? run.phase;
  let timing;
  let short;
  if (run.phase === "whisper") {
    const inWindow = Math.min(run.windowsDone + 1, run.windowsTotal);
    const basis = run.windowSecs.length > 0 ? "measured on this run" : "first estimate";
    timing =
      `window ${inWindow} of ${run.windowsTotal} · elapsed ${fmtDur(elapsed)} · ` +
      `about ${fmtDur(remaining)} remaining (${basis})`;
    short = `${pct}% · window ${inWindow}/${run.windowsTotal} · ~${fmtDur(remaining)} left`;
  } else if (run.phase === "sortformer") {
    timing = `elapsed ${fmtDur(elapsed)} · about ${fmtDur(remaining)} remaining`;
    short = `${pct}% · diarizing · ~${fmtDur(remaining)} left`;
  } else {
    timing = `elapsed ${fmtDur(elapsed)}`;
    short = `${pct}% · finishing`;
  }
  $("run-timing").textContent = timing;
  // Compact echo directly under the Run button, so the click gets feedback
  // exactly where it happened.
  $("stage-line").hidden = false;
  $("stage-line").textContent = short;
}

// Refresh the countdown once a second even when no engine event arrives.
setInterval(() => {
  if (state.busy && state.run) updateRunProgress();
}, 1000);

function finishRunCalibration() {
  const run = state.run;
  if (!run) return;
  const cal = { ...run.cal };
  if (run.windowSecs.length > 0) {
    const meanWindow = run.windowSecs.reduce((a, b) => a + b, 0) / run.windowSecs.length;
    cal.whisperRate = meanWindow / (run.audioSec / run.windowsTotal);
  }
  if (run.sortStartedAt != null) {
    const end = run.sortEndedAt ?? performance.now();
    const sortSecs = (end - run.sortStartedAt) / 1000;
    cal.sortformerRate = Math.max(0.01, sortSecs / run.audioSec);
  }
  saveCalibration(cal);
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
      state.lane = m.lane ?? "serial";
      setStatus(`engine booted (fw-wasm ${m.version}, ${state.lane} lane). Load the models to begin.`);
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
      // Live rate + remaining-time estimate (EMA over deltas, per file).
      let eta = "";
      if (m.phase === "download") {
        const now = performance.now();
        const d = state.dl;
        if (d.lastFile !== m.file) {
          d.lastFile = m.file;
          d.lastAt = now;
          d.lastLoaded = m.loaded;
          d.rate = 0;
        } else if (now > d.lastAt) {
          const inst = ((m.loaded - d.lastLoaded) * 1000) / (now - d.lastAt);
          d.rate = d.rate === 0 ? inst : d.rate * 0.8 + inst * 0.2;
          d.lastAt = now;
          d.lastLoaded = m.loaded;
        }
        if (d.rate > 1) {
          const remainingBytes = COMBINED_TOTAL - done;
          eta = ` · ${fmtBytes(d.rate)}/s · about ${fmtDur(remainingBytes / d.rate)} left`;
        }
      }
      text.textContent =
        `${m.file}: ${fmtBytes(m.loaded)} of ${fmtBytes(m.total)} (${phase}); ${pct}% overall${eta}`;
      break;
    }
    case "stage": {
      const label = STAGE_LABELS[m.stage] ?? m.stage;
      setStatus(label);
      if (state.run) {
        if (m.stage === "sortformer:diarize") {
          state.run.phase = "sortformer";
          state.run.sortStartedAt = performance.now();
          const cue = document.querySelector("#live-cue .listening");
          if (cue) cue.textContent = "identifying speakers";
        } else if (m.stage === "fuse:project") {
          state.run.sortEndedAt = performance.now();
          state.run.phase = "fuse";
        }
        updateRunProgress();
      } else if (state.busy) {
        $("stage-line").hidden = false;
        $("stage-line").textContent = label;
      }
      break;
    }
    case "audio-meta": {
      beginRun(m.audio_sec);
      state.run.offsetSec = m.skipped_leading_sec ?? 0;
      const skipped =
        state.run.offsetSec > 0.5 ? `skipped ${fmtDur(state.run.offsetSec)} of leading silence; ` : "";
      setStatus(
        `decoded ${fmtDur(m.audio_sec)} of audio; ${skipped}transcription starts now ` +
          `(${state.run.windowsTotal} window${state.run.windowsTotal === 1 ? "" : "s"} of up to 30 s each)`,
      );
      break;
    }
    case "partial-segments": {
      appendLiveSegments(m.segments ?? []);
      break;
    }
    case "span": {
      const run = state.run;
      if (!run) break;
      if (m.span === "encoder_window") {
        const now = performance.now();
        run.windowSecs.push((now - run.lastWindowAt) / 1000);
        run.lastWindowAt = now;
        run.windowsDone += 1;
      }
      // sortformer_tick needs no bookkeeping: the 1-second repaint shows the
      // countdown, and the tick's arrival proves the stage is alive.
      updateRunProgress();
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
      state.lane = m.lane ?? "serial";
      $("progress-wrap").hidden = true;
      $("load-models").disabled = true;
      $("load-models").querySelector("span").textContent = "Models loaded ✓";
      const laneNote =
        m.lane === "threaded"
          ? `threaded engine, ${m.threads} workers`
          : "serial engine (this browser gets the single-thread build)";
      setStatus(`Both models ready (${laneNote}). Pick a recording.`, "ok");
      maybeEnableRun();
      break;
    }
    case "result": {
      state.busy = false;
      state.result = m.result;
      state.wallMs = m.wall_ms;
      state.nameMap = buildSpeakerNameMap(m.result, parseSpeakerNames($("speaker-names").value));
      finishRunCalibration();
      state.run = null;
      $("run-panel").hidden = true;
      $("progress-wrap").hidden = true;
      $("stage-line").hidden = true;
      renderResult();
      maybeEnableRun();
      break;
    }
    case "cache-cleared": {
      state.modelsReady = false;
      $("load-models").disabled = false;
      $("load-models").querySelector("span").textContent = "Load both models (1.3 GB)";
      setStatus("Cache cleared. Models will re-download on next load.");
      maybeEnableRun();
      break;
    }
    case "error": {
      state.busy = false;
      state.run = null;
      $("run-panel").hidden = true;
      $("progress-wrap").hidden = true;
      $("stage-line").hidden = true;
      setStatus(`error: ${m.message}`, "err");
      // A failed model load must leave a way forward: re-enable the load
      // button as a retry (already-verified files are skipped, a partial
      // download resumes from its last byte).
      if (!state.modelsReady) {
        const btn = $("load-models");
        btn.disabled = false;
        btn.querySelector("span").textContent = "Retry loading models";
      }
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
  const named = state.nameMap.size > 0;
  $("result-meta").textContent =
    `${fmtTime(r.audio_sec)} of audio · ${(state.wallMs / 1000).toFixed(1)} s wall ` +
    `(${rt}× realtime on this machine) · ${speakers.size} speaker(s) · ` +
    `${r.turns.length} turns` +
    (named ? " · names assigned by first appearance" : "") +
    ((r.skipped_leading_sec ?? 0) > 0.5
      ? ` · skipped ${fmtDur(r.skipped_leading_sec)} of leading silence`
      : "") +
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
    spk.textContent = displaySpeaker(seg.speaker, state.nameMap);
    if (seg.speaker != null && state.nameMap.has(seg.speaker)) {
      spk.title = `detected as ${seg.speaker}`;
    }
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
    state.nameMap.size > 0
      ? `- Speaker names were assigned to detected voices in order of first appearance (${[...state.nameMap.entries()].map(([k, v]) => `${k} = ${v}`).join(", ")})`
      : null,
    r.dropped_windows > 0
      ? `- **Warning: ${r.dropped_windows} window(s) dropped without output**`
      : null,
    "",
  ].filter((l) => l !== null);
  for (const seg of r.speaker_segments) {
    lines.push(
      `**[${fmtTime(seg.start_sec)}–${fmtTime(seg.end_sec)}] ${displaySpeaker(seg.speaker, state.nameMap)}:** ${seg.text.trim()}`,
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
      return `      <div class="seg"><span class="t">${fmtTime(seg.start_sec)}–${fmtTime(seg.end_sec)}</span> <span class="spk" style="color:${color};border-color:${color}66">${esc(displaySpeaker(seg.speaker, state.nameMap))}</span> ${esc(seg.text.trim())}</div>`;
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
  $("drop-hint").textContent = `${file.name} (${fmtBytes(file.size)}), ready.`;
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
    setStatus("downloading models (verified and resumable; a reload picks up where it left off)…");
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

  $("sample").addEventListener("click", async () => {
    try {
      const resp = await fetch("./assets/samples/jfk-1961.mp3");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      acceptFile(new File([blob], "jfk-1961.mp3", { type: "audio/mpeg" }));
      setStatus("Sample loaded: JFK inaugural excerpt, 11 seconds, one speaker.", "ok");
    } catch (e) {
      setStatus(`could not fetch the sample: ${e.message ?? e}`, "err");
    }
  });

  $("run").addEventListener("click", async () => {
    if (!state.file || state.busy) return;
    state.busy = true;
    maybeEnableRun();
    $("download-md").hidden = true;
    $("download-html").hidden = true;
    setStatus("running…");
    const ext = (state.file.name.split(".").pop() || "").toLowerCase();
    // The names field feeds Whisper's decoding prompt (the CLI's --prompt)
    // so names and titles come out spelled right; the same list later maps
    // onto detected speakers in order of first appearance.
    const names = parseSpeakerNames($("speaker-names").value);
    const prompt = names.length ? `Speakers: ${names.join(", ")}.` : undefined;
    const language = $("language").value === "auto" ? undefined : $("language").value;
    const buf = await state.file.arrayBuffer();
    state.worker.postMessage({ type: "transcribe", audio: buf, ext, prompt, language }, [buf]);
  });

  $("download-md").addEventListener("click", exportMd);
  $("download-html").addEventListener("click", exportHtml);

  bootWorker();
}

init();
