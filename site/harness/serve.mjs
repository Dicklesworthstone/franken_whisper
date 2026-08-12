// Local harness server: serves site/ with the COOP/COEP headers the threaded
// lane needs (python -m http.server sends neither, so crossOriginIsolated is
// false and the shared-memory module refuses to instantiate). Also exposes
// the local model files under /local/* so the harness can exercise the REAL
// turbo + Sortformer artifacts without a 2 GB download.
//
//   node site/harness/serve.mjs [port]
//
// Env (all optional, defaults match this repo's dev machine):
//   FW_TURBO=…/ggml-large-v3-turbo.bin
//   FW_SF_DIR=…/sortformer-v2.1-f32-v1
//   FW_CLIP=…/meeting.mp3
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { join, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = resolve(fileURLToPath(new URL("..", import.meta.url)));
const PORT = Number(process.argv[2] ?? 8791);
const TURBO = process.env.FW_TURBO ?? "/Users/jemanuel/models/whisper/ggml-large-v3-turbo.bin";
const SF_DIR =
  process.env.FW_SF_DIR ??
  "/Users/jemanuel/.cache/franken_whisper/models/sortformer/sortformer-v2.1-f32-v1";
const CLIP = process.env.FW_CLIP ?? join(SITE, "assets/samples/jfk-1961.mp3");

const LOCAL = {
  "/local/turbo.bin": TURBO,
  "/local/sortformer.safetensors": join(SF_DIR, "weights.safetensors"),
  "/local/receipt.json": join(SF_DIR, "conversion-receipt.json"),
  "/local/clip.bin": CLIP,
  // The production model proxy (functions/model/[[path]].js on Pages),
  // mapped to the same local artifacts so the REAL page + loader + OPFS
  // staging path runs against localhost.
  "/model/whisper/ggml-large-v3-turbo.bin": TURBO,
  "/model/sortformer/weights.safetensors": join(SF_DIR, "weights.safetensors"),
  "/model/sortformer/conversion-receipt.json": join(SF_DIR, "conversion-receipt.json"),
};

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".mp3": "audio/mpeg",
  ".woff2": "font/woff2",
};

createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cache-Control", "no-store");
  try {
    const local = LOCAL[url.pathname];
    const path = local ?? join(SITE, url.pathname === "/" ? "index.html" : url.pathname.slice(1));
    if (!local && !resolve(path).startsWith(SITE)) throw new Error("outside site/");
    const info = await stat(path);
    res.setHeader("Content-Type", MIME[extname(path)] ?? "application/octet-stream");
    res.setHeader("Content-Length", info.size);
    createReadStream(path).pipe(res);
  } catch {
    res.statusCode = 404;
    res.end("not found");
  }
}).listen(PORT, "127.0.0.1", () => {
  console.log(`harness server: http://127.0.0.1:${PORT}/ (COOP/COEP on)`);
});
