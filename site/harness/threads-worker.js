// Harness worker: the threaded lane end to end with the REAL models, served
// same-origin by serve.mjs. No OPFS (the model arrives as one ArrayBuffer and
// feeds the streamed loader through the same __fwModelReadAt hook the
// production worker registers over its OPFS handle).
const post = (m) => self.postMessage(m);
const stage = (s) => post({ stage: s });

try {
  stage("boot");
  if (self.crossOriginIsolated !== true) throw new Error("not crossOriginIsolated");
  const pkg = await import("../pkg-threaded/fw_wasm.js");
  await pkg.default();
  if (!(pkg.wasm_memory().buffer instanceof SharedArrayBuffer)) {
    throw new Error("threaded module got a non-shared memory");
  }
  globalThis.__fwStage = (s) => stage(`engine:${s}`);

  // Arm FIRST: the engine's model load initializes rayon's global pool
  // implicitly (rayon::join in the weight build), and a global pool can only
  // be built once — arming later hits "already initialized". The
  // parked-worker/grow hazard is what this harness exists to test.
  stage("arm thread pool");
  await pkg
    .initThreadPool(Math.max(1, Math.min((navigator.hardwareConcurrency ?? 4) - 1, 8)))
    .catch((e) => {
      throw new Error(`initThreadPool: ${e?.message ?? e} :: ${String(e?.stack ?? "").slice(0, 300)}`);
    });

  stage("fetch turbo (1.5 GB, local)");
  const turbo = new Uint8Array(await (await fetch("/local/turbo.bin")).arrayBuffer());
  globalThis.__fwModelReadAt = (offset, len) => turbo.subarray(offset, offset + len);
  stage("load whisper (streamed)");
  pkg.set_now_micros(Math.trunc(performance.now() * 1000));
  pkg.load_whisper_streamed(turbo.byteLength);

  stage("fetch + load sortformer");
  const receipt = new Uint8Array(await (await fetch("/local/receipt.json")).arrayBuffer());
  const weights = new Uint8Array(await (await fetch("/local/sortformer.safetensors")).arrayBuffer());
  pkg.load_sortformer(receipt, weights);

  const threads = pkg.thread_count();

  stage(`transcribe+diarize (threads=${threads})`);
  const clip = new Uint8Array(await (await fetch("/local/clip.bin")).arrayBuffer());
  pkg.set_now_micros(Math.trunc(performance.now() * 1000));
  const t0 = performance.now();
  const result = JSON.parse(pkg.transcribe_and_diarize(clip, "mp3", undefined));
  const wallSec = (performance.now() - t0) / 1000;

  post({
    ok: threads > 1 && result.segments.length > 0,
    threads,
    wall_sec: Number(wallSec.toFixed(1)),
    audio_sec: Number(result.audio_sec.toFixed(1)),
    rt_ratio: Number((wallSec / result.audio_sec).toFixed(2)),
    dropped_windows: result.dropped_windows,
    speakers: [...new Set(result.speaker_segments.map((s) => s.speaker))],
    text: result.speaker_segments.map((s) => s.text.trim()).join(" ").slice(0, 400),
  });
} catch (e) {
  post({ ok: false, error: String(e?.message ?? e) });
}
