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
    label: "Whisper large-v3-turbo (speech → text)",
    license: "OpenAI Whisper - MIT",
    weights: {
      name: "ggml-large-v3-turbo.bin",
      bytes: 1624555275,
      sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
      // Precomputed SHA-256 of the first/last ENDPOINT_BYTES, so a warm start
      // proves the cached file with a ~16 MB read instead of a full-file
      // hash. The loader still full-hashes on first download and under
      // `?fullverify`.
      head: "d996847e6b5c831f4f51aac00469be269d0e98af082057b4ba63914ddb45c188",
      tail: "1cbb0a5b9f9402c6188fc89a41f18420d325acbe809010fba8408cde634a005b",
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
export const CHUNK_BYTES = 32 * 1024 * 1024;
export const ENDPOINT_BYTES = 8 * 1024 * 1024;

export function totalBytes(model) {
  const m = MODELS[model];
  return m.weights.bytes + m.sidecars.reduce((sum, f) => sum + f.bytes, 0);
}
