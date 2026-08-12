// Verified model loader: /model/* (same-origin proxy) → OPFS → verified files.
//
// OPFS (not the Cache API) because the whisper engine reads the 1.5 GB ggml
// file through a SYNCHRONOUS positioned reader (FileSystemSyncAccessHandle in
// the worker) so tensors hydrate one at a time — the file is never resident
// as one blob on either side of the boundary.
//
// Discipline per file (size + full SHA-256 against the pinned manifest):
// * Cold: streamed fetch → ONE writable held open (WebKit copies the whole
//   file per createWritable open) → incremental SHA-256 (WebCrypto cannot
//   stream) → full-pin check → verified marker records head/tail digests.
// * Warm: size + head/tail endpoint digests against the marker written by a
//   past FULL verification (reads ~16 MB instead of 2 GB). `?fullverify`
//   forces the full digest on every cached asset.
// * Resume: an unverified partial file re-hashes its existing bytes, then
//   Range-fetches the remainder — an interrupted 1.5 GB pull loses nothing.
//
// Runs inside the engine worker (OPFS sync handles are worker-only anyway).

import { Sha256 } from "./sha256.js?v=@SITEV@";
import { CHUNK_BYTES, ENDPOINT_BYTES, MODELS } from "./model-manifest.js?v=@SITEV@";

const FULL_VERIFY = new URLSearchParams(globalThis.location?.search ?? "").has("fullverify");

function assetUrl(modelId, name) {
  return `/model/${modelId}/${name}`;
}

function markerName(name) {
  return `${name}.verified.json`;
}

async function opfsRoot() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle("fw-models", { create: true });
}

async function readMarker(dir, name) {
  try {
    const handle = await dir.getFileHandle(markerName(name));
    return JSON.parse(await (await handle.getFile()).text());
  } catch {
    return null;
  }
}

async function writeMarker(dir, name, marker) {
  const handle = await dir.getFileHandle(markerName(name), { create: true });
  const writable = await handle.createWritable();
  await writable.write(JSON.stringify(marker));
  await writable.close();
}

function yieldToEventLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function digestFileRange(file, start, end) {
  const hasher = new Sha256();
  let pos = start;
  while (pos < end) {
    const next = Math.min(pos + CHUNK_BYTES, end);
    hasher.update(new Uint8Array(await file.slice(pos, next).arrayBuffer()));
    pos = next;
    await yieldToEventLoop();
  }
  return hasher.hex();
}

/** Cheap warm-start proof: exact size + head/tail digests recorded by a past
 * full verification. Falls back to (and `?fullverify` forces) the full pin. */
async function cachedFileValid(dir, spec) {
  let file;
  try {
    const handle = await dir.getFileHandle(spec.name);
    file = await handle.getFile();
  } catch {
    return false;
  }
  if (file.size !== spec.bytes) return false;
  const marker = await readMarker(dir, spec.name);
  if (!marker || marker.sha256 !== spec.sha256) return false;
  if (!FULL_VERIFY && marker.head && marker.tail && spec.bytes >= 2 * ENDPOINT_BYTES) {
    const head = await digestFileRange(file, 0, ENDPOINT_BYTES);
    const tail = await digestFileRange(file, spec.bytes - ENDPOINT_BYTES, spec.bytes);
    return head === marker.head && tail === marker.tail;
  }
  return (await digestFileRange(file, 0, spec.bytes)) === spec.sha256;
}

/** Stream (or resume) one file into OPFS with incremental verification. */
async function downloadVerified(dir, modelId, spec, onProgress) {
  const handle = await dir.getFileHandle(spec.name, { create: true });
  let existing = (await handle.getFile()).size;
  if (existing > spec.bytes) {
    // Corrupt overshoot: start over.
    await dir.removeEntry(spec.name);
    return downloadVerified(dir, modelId, spec, onProgress);
  }

  // Re-hash whatever a previous attempt already banked, so the incremental
  // digest covers the WHOLE file even across resumes.
  let hasher = new Sha256();
  if (existing > 0) {
    const file = await handle.getFile();
    let pos = 0;
    while (pos < existing) {
      const next = Math.min(pos + CHUNK_BYTES, existing);
      hasher.update(new Uint8Array(await file.slice(pos, next).arrayBuffer()));
      pos = next;
      onProgress?.({ file: spec.name, loaded: pos, total: spec.bytes, phase: "rehash" });
      await yieldToEventLoop();
    }
  }

  if (existing < spec.bytes) {
    const headers = existing > 0 ? { range: `bytes=${existing}-` } : {};
    const resp = await fetch(assetUrl(modelId, spec.name), { headers });
    if (!resp.ok && resp.status !== 206) {
      throw new Error(`${spec.name}: HTTP ${resp.status}`);
    }
    if (existing > 0 && resp.status !== 206) {
      // Server ignored the range: restart the hash and overwrite from zero.
      existing = 0;
      hasher = new Sha256();
    }
    const writable = await handle.createWritable({ keepExistingData: existing > 0 });
    if (existing > 0) await writable.seek(existing);
    const reader = resp.body.getReader();
    let loaded = existing;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (loaded + value.byteLength > spec.bytes) {
          throw new Error(`${spec.name}: stream exceeded pinned ${spec.bytes} bytes`);
        }
        hasher.update(value);
        await writable.write(value);
        loaded += value.byteLength;
        onProgress?.({ file: spec.name, loaded, total: spec.bytes, phase: "download" });
      }
      await writable.close();
    } catch (err) {
      try {
        await writable.abort();
      } catch {
        /* already errored */
      }
      throw err;
    }
    if (loaded !== spec.bytes) {
      throw new Error(`${spec.name}: got ${loaded} of ${spec.bytes} bytes (will resume)`);
    }
  }

  const digest = hasher.hex();
  if (digest !== spec.sha256) {
    await dir.removeEntry(spec.name);
    throw new Error(
      `${spec.name}: digest ${digest.slice(0, 12)}… does not match pin ${spec.sha256.slice(0, 12)}…`,
    );
  }
  // Record endpoint digests for cheap warm-start proofs.
  const file = await handle.getFile();
  const marker = { sha256: spec.sha256, bytes: spec.bytes };
  if (spec.bytes >= 2 * ENDPOINT_BYTES) {
    marker.head = await digestFileRange(file, 0, ENDPOINT_BYTES);
    marker.tail = await digestFileRange(file, spec.bytes - ENDPOINT_BYTES, spec.bytes);
  }
  await writeMarker(dir, spec.name, marker);
}

/** Ensure every file of `modelId` is present and verified in OPFS.
 * Returns the OPFS directory handle holding the files. */
export async function ensureModel(modelId, onProgress) {
  const model = MODELS[modelId];
  if (!model) throw new Error(`unknown model: ${modelId}`);
  const dir = await opfsRoot();
  const files = [model.weights, ...model.sidecars];
  for (const spec of files) {
    if (await cachedFileValid(dir, spec)) {
      onProgress?.({ file: spec.name, loaded: spec.bytes, total: spec.bytes, phase: "cached" });
      continue;
    }
    await downloadVerified(dir, modelId, spec, onProgress);
    onProgress?.({ file: spec.name, loaded: spec.bytes, total: spec.bytes, phase: "verified" });
  }
  return dir;
}

/** Drop every cached model file (the playground's "clear cache" button). */
export async function clearModels() {
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry("fw-models", { recursive: true });
  } catch {
    /* nothing cached */
  }
}
