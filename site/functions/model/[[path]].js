// Same-origin model proxy (Cloudflare Pages Function).
//
// Upstreams send no CORS headers (GitHub) or would widen the page's CSP
// (Hugging Face), so this function forwards /model/<model>/<file> to pinned
// upstream URLs and the page keeps `connect-src 'self'`. It is a model
// mirror, not an open proxy: only the allow-listed files forward, and only
// the Range request header crosses.
//
// Each file lists upstreams in preference order. Hugging Face first: its CDN
// absorbs load that made GitHub releases return 5xx under concurrent
// downloads. The GitHub release stays as the fallback (and as the primary
// for the Sortformer package until its Hugging Face upload lands — see
// scripts/upload-sortformer-to-hf.sh).
const HF_SORTFORMER = "https://huggingface.co/Dicklesworthstone/franken-whisper-sortformer/resolve/main/";
const GH_SORTFORMER =
  "https://github.com/Dicklesworthstone/franken_whisper/releases/download/sortformer-v2.1-f32-v1/";

const FILES = {
  "whisper/ggml-large-v3-turbo.bin": [
    // The canonical upstream our manifest pins (same sha256, verified):
    // whisper.cpp's own model repo on Hugging Face.
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    "https://github.com/Dicklesworthstone/franken_whisper/releases/download/whisper-large-v3-turbo-f16-v1/ggml-large-v3-turbo.bin",
  ],
  "sortformer/weights.safetensors": [
    `${HF_SORTFORMER}weights.safetensors`,
    `${GH_SORTFORMER}weights.safetensors`,
  ],
  "sortformer/conversion-receipt.json": [
    `${HF_SORTFORMER}conversion-receipt.json`,
    `${GH_SORTFORMER}conversion-receipt.json`,
  ],
};

export async function onRequest({ request, params }) {
  const parts = Array.isArray(params.path) ? params.path : [params.path];
  if (parts.length !== 2) return new Response("not found", { status: 404 });
  const upstreams = FILES[parts.join("/")];
  if (!upstreams) return new Response("not found", { status: 404 });

  const headers = request.headers.has("range")
    ? { range: request.headers.get("range") }
    : {};
  let last = null;
  for (const url of upstreams) {
    const resp = await fetch(new Request(url, { method: "GET", headers, redirect: "follow" }));
    if (resp.ok || resp.status === 206) {
      const out = new Headers();
      for (const name of ["content-length", "content-range", "accept-ranges", "etag"]) {
        const value = resp.headers.get(name);
        if (value) out.set(name, value);
      }
      out.set("content-type", "application/octet-stream");
      // Immutable pinned artifacts: cache aggressively (the loader verifies
      // every byte against sha256 pins regardless of where they came from).
      out.set("cache-control", "public, max-age=31536000, immutable");
      return new Response(resp.body, { status: resp.status, headers: out });
    }
    last = resp.status;
    // Drain the failed body so the subrequest slot frees promptly.
    try {
      await resp.body?.cancel();
    } catch {
      /* already closed */
    }
  }
  return new Response(`every upstream failed (last: ${last})`, { status: 502 });
}
