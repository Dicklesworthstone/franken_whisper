// Production click-path harness: drives the REAL playground page (index.html
// + app.js + engine-worker.js + loader.js + OPFS staging + lane pick) in
// headless Chrome via CDP, exactly as a visitor uses it: click Load models,
// accept the consent, wait for ready, click the sample-clip button, click
// Transcribe + diarize, then watch #status / #progress-text until a result
// or error. Prints a JSON verdict; exits 0 only on a rendered transcript.
//
//   node site/harness/prod.mjs [base-url] [deadline-seconds]
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = process.argv[2] ?? "http://127.0.0.1:8791";
const DEADLINE_S = Number(process.argv[3] ?? 480);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9700 + Math.floor(Math.random() * 200);

const chrome = spawn(
  CHROME,
  [
    "--headless=new",
    "--disable-gpu",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${mkdtempSync(join(tmpdir(), "fw-prod-"))}`,
    "about:blank",
  ],
  { stdio: "ignore" },
);
const cleanup = () => {
  try {
    chrome.kill();
  } catch {
    /* gone */
  }
};
process.on("exit", cleanup);

async function target() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch {
      /* booting */
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error("chrome debugger never came up");
}

const ws = new WebSocket(await target());
await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = rej;
});
let seq = 0;
const waiting = new Map();
const consoleLines = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) {
    waiting.get(m.id)(m);
    waiting.delete(m.id);
  } else if (m.method === "Runtime.consoleAPICalled") {
    const text = (m.params.args ?? []).map((a) => a.value ?? a.description ?? "").join(" ");
    consoleLines.push(`[${m.params.type}] ${text}`.slice(0, 300));
  } else if (m.method === "Runtime.exceptionThrown") {
    consoleLines.push(`[exception] ${m.params.exceptionDetails?.text ?? ""}`.slice(0, 300));
  }
};
function cdp(method, params = {}) {
  const id = ++seq;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((r) => waiting.set(id, r));
}
async function evalJs(expression) {
  const r = await cdp("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  return r.result?.result?.value;
}
async function status() {
  return evalJs(
    "JSON.stringify({status: document.getElementById('status')?.textContent, " +
      "progress: document.getElementById('progress-text')?.textContent, " +
      "meta: document.getElementById('result-meta')?.textContent, " +
      "runDisabled: document.getElementById('run')?.disabled})",
  ).then((s) => JSON.parse(s ?? "{}"));
}

await cdp("Runtime.enable");
await cdp("Page.enable");
await cdp("Page.navigate", { url: `${BASE}/` });
await new Promise((r) => setTimeout(r, 3000));

const clicks = [
  ["#load-models", "document.getElementById('load-models').click()"],
  ["#consent-yes", "document.getElementById('consent-yes').click()"],
];
for (const [name, js] of clicks) {
  await evalJs(js);
  console.error(`[prod] clicked ${name}`);
  await new Promise((r) => setTimeout(r, 800));
}

// Wait for "ready" (models loaded + pool armed).
const t0 = Date.now();
let lastLine = "";
for (;;) {
  await new Promise((r) => setTimeout(r, 2000));
  const s = await status();
  const line = `${s.status ?? ""} | ${s.progress ?? ""}`;
  if (line !== lastLine) {
    lastLine = line;
    console.error(`[prod] ${line.slice(0, 160)}`);
  }
  if (/Both models ready/.test(s.status ?? "")) break;
  if (/^error:/.test(s.status ?? "")) {
    console.log(JSON.stringify({ ok: false, phase: "load", status: s.status, console: consoleLines.slice(-8) }));
    process.exit(1);
  }
  if ((Date.now() - t0) / 1000 > DEADLINE_S) {
    console.log(JSON.stringify({ ok: false, phase: "load-timeout", status: s, console: consoleLines.slice(-8) }));
    process.exit(1);
  }
}
console.error("[prod] models ready; loading sample clip");
await evalJs("document.getElementById('sample').click()");
await new Promise((r) => setTimeout(r, 1500));
await evalJs("document.getElementById('run').click()");
console.error("[prod] clicked run");

for (;;) {
  await new Promise((r) => setTimeout(r, 2000));
  const s = await status();
  const line = `${s.status ?? ""} | ${s.progress ?? ""}`;
  if (line !== lastLine) {
    lastLine = line;
    console.error(`[prod] ${line.slice(0, 160)}`);
  }
  if (/^Done\./.test(s.status ?? "") && s.meta) {
    const text = await evalJs("document.getElementById('output')?.textContent");
    console.log(JSON.stringify({ ok: true, meta: s.meta, text: (text ?? "").slice(0, 300) }));
    process.exit(0);
  }
  if (/^error:/.test(s.status ?? "")) {
    console.log(JSON.stringify({ ok: false, phase: "run", status: s.status, console: consoleLines.slice(-8) }));
    process.exit(1);
  }
  if ((Date.now() - t0) / 1000 > DEADLINE_S) {
    console.log(JSON.stringify({ ok: false, phase: "run-timeout", lastState: s, console: consoleLines.slice(-8) }));
    process.exit(1);
  }
}
