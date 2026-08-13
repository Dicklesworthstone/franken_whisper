// Pinned model files for the browser playground. Mirrors the CLI's own pins:
// models/whisper-manifest-v1.json for the whisper weights (the same bytes
// `fw` verifies on pull) and the compiled Sortformer receipt/package digests
// in src/sortformer_conformance.rs. The loader refuses a mismatched digest
// and the engine re-verifies the Sortformer package against its conversion
// receipt in wasm, so a truncated or tampered download can never hydrate.
//
// Assets are fetched same-origin from /model/<model>/<file>, which the Pages
// Function (functions/model/[[path]].js) proxies to the GitHub release —
// release assets send no CORS headers, so the browser cannot fetch them
// cross-origin.
export const MODELS = {
  whisper: {
    label: "Whisper large-v3-turbo, q8_0 (speech → text)",
    license: "OpenAI Whisper - MIT",
    // The q8_0 lane (bd-3be3 Phase 1): 46% smaller download and less than
    // half the resident memory of the f16 file, transcript-gated IDENTICAL
    // to the f16 lane (jfk + meeting + multi-window fixtures; the blocks
    // stay resident and dequantize per row in-kernel).
    weights: {
      name: "ggml-large-v3-turbo-q8_0.bin",
      bytes: 874188075,
      sha256: "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
      // Precomputed SHA-256 of the first/last ENDPOINT_BYTES, so a warm start
      // proves the cached file with a ~16 MB read instead of a full-file
      // hash. The loader still full-hashes on first download and under
      // `?fullverify`.
      head: "6ed55aa6d0ab4f46c1708213e06d536e7e47a59c8cef868bbce705e57b0c15cf",
      tail: "0f9a1f8417f5602ace4c9cf46c1181798409be0b81406f2576baad519364cfe0",
    },
    sidecars: [],
  },
  sortformer: {
    label: "NVIDIA Streaming Sortformer (who spoke when)",
    license: "NVIDIA Open Model License",
    weights: {
      name: "weights.safetensors",
      bytes: 491570584,
      sha256: "487fa30cb0aa9799c77bd9985e6787962c3991fab8d4d576a4f1221d45298f6a",
      head: "0e7073139cfb591596c1ee93f12db45ca4f3dd37dbf33119d7b0e4b9b10b2fb7",
      tail: "1b81ffb92d6a0a57c69a37b7cd083527b8ed2685f1339c8db7b7296f0e98689a",
    },
    sidecars: [
      {
        name: "conversion-receipt.json",
        bytes: 653208,
        sha256: "407c642f3d51b399514f6a35227b1c80886387472a44fb78f01b824d26318fb0",
      },
    ],
  },
};

// Chunked-download + verification tuning (see loader.js).
// FastEnhancer-S speech denoiser (MIT), the default clean-up stage before
// transcription; tiny, so it loads with the models in a blink.
MODELS.denoiser = {
  label: "FastEnhancer-S denoiser (audio clean-up)",
  license: "FastEnhancer - MIT",
  weights: {
    name: "fastenhancer-s-48k-denoise.safetensors",
    bytes: 838440,
    sha256: "28c1807fd9113e4ca09d3aacb2ecb07a742917321bfaced8b92598daffbd098b",
  },
  sidecars: [],
};

export const CHUNK_BYTES = 32 * 1024 * 1024;
export const ENDPOINT_BYTES = 8 * 1024 * 1024;

export function totalBytes(model) {
  const m = MODELS[model];
  return m.weights.bytes + m.sidecars.reduce((sum, f) => sum + f.bytes, 0);
}
