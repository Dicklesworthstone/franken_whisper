// CDP driver for the harness pages: launches headless Chrome, navigates,
// polls document.title until HARNESS-OK / HARNESS-FAIL (or the deadline),
// prints the #result text, exits 0/1. No dependencies — Node's built-in
// WebSocket client speaks CDP directly.
//
//   node site/harness/drive.mjs <url> [timeout-seconds]
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const URL_ARG = process.argv[2];
const DEADLINE_S = Number(process.argv[3] ?? 300);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9222 + Math.floor(Math.random() * 500);

const chrome = spawn(
  CHROME,
  [
    "--headless=new",
    "--disable-gpu",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${mkdtempSync(join(tmpdir(), "fw-harness-"))}`,
    "about:blank",
  ],
  { stdio: "ignore" },
);
const cleanup = () => {
  try {
    chrome.kill();
  } catch {
    /* already gone */
  }
};
process.on("exit", cleanup);

async function cdpTarget() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error("Chrome debugger never came up");
}

const ws = new WebSocket(await cdpTarget());
await new Promise((resolve, reject) => {
  ws.onopen = resolve;
  ws.onerror = reject;
});
let seq = 0;
const waiting = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) {
    waiting.get(m.id)(m);
    waiting.delete(m.id);
  }
};
function cdp(method, params = {}) {
  const id = ++seq;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve) => waiting.set(id, resolve));
}
async function evalJs(expression) {
  const r = await cdp("Runtime.evaluate", { expression, returnByValue: true });
  return r.result?.result?.value;
}

await cdp("Page.enable");
await cdp("Page.navigate", { url: URL_ARG });

const t0 = Date.now();
let lastStages = "";
for (;;) {
  await new Promise((r) => setTimeout(r, 2000));
  const title = await evalJs("document.title");
  const text = (await evalJs("document.getElementById('result')?.textContent")) ?? "";
  if (text !== lastStages) {
    lastStages = text;
    const lines = text.trim().split("\n");
    console.error(`[harness] ${lines[lines.length - 1]}`);
  }
  if (title === "HARNESS-OK" || title === "HARNESS-FAIL") {
    console.log(text);
    cleanup();
    process.exit(title === "HARNESS-OK" ? 0 : 1);
  }
  if ((Date.now() - t0) / 1000 > DEADLINE_S) {
    console.log(`DEADLINE after ${DEADLINE_S}s; last state:\n${text}`);
    cleanup();
    process.exit(1);
  }
}
