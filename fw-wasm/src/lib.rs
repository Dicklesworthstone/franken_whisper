// Same nightly feature set as the parent crate: nn.rs uses std::simd, which
// is portable by design (simd128 on wasm32, NEON/AVX natively) — the same
// SIMD source compiles for the browser.
#![feature(portable_simd)]

//! franken_whisper browser bindings (bd-m2jm).
//!
//! One code path: the modules below are the parent crate's OWN sources,
//! mounted by `#[path]` — not copies. The only wasm-specific code is the
//! `plat` seam inside `native_engine` (host-fed clock, serial thread scope)
//! and the thin `wasm_api` surface at the bottom of this file. The `model`
//! and `model_distribution` shims exist because the parent's versions drag
//! CLI/filesystem dependencies the browser can never use; both are minimal
//! and documented against their canonical definitions.

// The parent crate's error module, verbatim (thiserror + std only).
#[path = "../../src/error.rs"]
pub mod error;

// Minimal shims for the two parent modules `native_engine` references that
// are NOT wasm-portable in their canonical form (clap/uuid/fs machinery).
pub mod model;
pub mod model_distribution;

// The engine itself: the exact sources the native binary ships.
#[path = "../../src/native_engine/mod.rs"]
pub mod native_engine;

// The diarization stack, same deal — the parent's OWN sources: Sortformer
// conformance (receipt/package authentication), Sortformer inference
// (frontend + FastConformer + transformer + turn extraction), and the
// portable projection-fusion module the CLI's orchestrator also calls.
#[path = "../../src/diarization_projection.rs"]
pub mod diarization_projection;
#[path = "../../src/sortformer_conformance.rs"]
pub mod sortformer_conformance;
#[path = "../../src/sortformer_inference.rs"]
pub mod sortformer_inference;

// bytes → 16 kHz mono f32 (symphonia; wasm-only).
pub mod audio_decode;

#[cfg(target_arch = "wasm32")]
mod wasm_api {
    use std::cell::RefCell;

    use wasm_bindgen::prelude::*;

    use crate::native_engine::decode::{self, DecodeParams, LoadedModel};
    use crate::native_engine::ggml::{GgmlModel, HostReader};
    use crate::sortformer_inference::{SortformerPcm, SortformerSession};

    thread_local! {
        // Single-threaded wasm: the loaded models live in the worker that
        // called the load entry points; everything below runs on that worker.
        static MODEL: RefCell<Option<LoadedModel>> = const { RefCell::new(None) };
        static DIARIZER: RefCell<Option<SortformerSession>> = const { RefCell::new(None) };
    }

    // Host-supplied positioned read against the whisper model file staged in
    // OPFS: the host registers `globalThis.__fwModelReadAt(offset, len) ->
    // Uint8Array` (a FileSystemSyncAccessHandle read in the worker; an fs
    // pread in the Node smoke harness). This is what lets a 1.5 GB ggml file
    // load tensor-by-tensor instead of as one resident blob, and it is why
    // `load_whisper_streamed` must run in a worker (sync reads).
    #[wasm_bindgen(inline_js = "export function fw_model_read_at(offset, len) {
        const f = globalThis.__fwModelReadAt;
        if (!f) throw new Error('host model reader not registered (globalThis.__fwModelReadAt)');
        return f(offset, len);
    }
    export function fw_stage(name) {
        const f = globalThis.__fwStage;
        if (f) f(name);
    }")]
    extern "C" {
        #[wasm_bindgen(catch)]
        fn fw_model_read_at(offset: f64, len: f64) -> Result<js_sys::Uint8Array, JsValue>;
        /// Optional host progress hook: names the stage currently running.
        fn fw_stage(name: &str);
    }

    fn js_err(e: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&e.to_string())
    }

    /// Doctrine: a panic in a tab must never surface as an opaque
    /// `RuntimeError: unreachable`. Installed at module start.
    #[wasm_bindgen(start)]
    pub fn start() {
        std::panic::set_hook(Box::new(|info| {
            fw_console_error(&info.to_string());
        }));
    }

    // console.error without a web-sys dependency. Also banks the message in
    // `globalThis.__fwLastPanic` so the host can attach the REAL reason to
    // the opaque `RuntimeError: unreachable` the trap surfaces as. (Alloc
    // failures abort without running the hook at all — the host treats a
    // trap with no banked message as out-of-memory.)
    #[wasm_bindgen(inline_js = "export function fw_console_error(m) {
        globalThis.__fwLastPanic = m;
        console.error('fw-wasm panic:', m);
    }")]
    extern "C" {
        fn fw_console_error(m: &str);
    }

    /// Host-fed monotonic clock (microseconds). The embedding calls this with
    /// `Math.trunc(performance.now() * 1000)` whenever it re-enters wasm so
    /// the engine's perf spans read real durations instead of zero.
    #[wasm_bindgen]
    pub fn set_now_micros(micros: f64) {
        crate::native_engine::plat::set_now_micros(micros as u64);
    }

    /// Introspection: crate version, so the page can display what is running.
    #[wasm_bindgen]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    fn model_info_json(loaded: &LoadedModel) -> String {
        let hp = &loaded.hparams;
        serde_json::json!({
            "n_vocab": hp.n_vocab,
            "n_audio_layer": hp.n_audio_layer,
            "n_text_layer": hp.n_text_layer,
            "n_mels": hp.n_mels,
            "multilingual": hp.n_vocab >= 51_865,
        })
        .to_string()
    }

    /// Parse ggml model bytes held fully in memory and hold the loaded model
    /// in this worker. Suitable for small models (tiny/base); large models
    /// should use [`load_whisper_streamed`]. Returns hparams JSON.
    ///
    /// # Errors
    ///
    /// Any parse/load failure, as a string naming the failing stage.
    #[wasm_bindgen]
    pub fn load_model(bytes: Vec<u8>) -> Result<String, JsValue> {
        fw_stage("whisper:parse");
        let ggml = GgmlModel::from_bytes(bytes).map_err(js_err)?;
        fw_stage("whisper:weights");
        let loaded = LoadedModel::from_ggml(ggml).map_err(js_err)?;
        let info = model_info_json(&loaded);
        MODEL.with(|m| *m.borrow_mut() = Some(loaded));
        fw_stage("whisper:ready");
        Ok(info)
    }

    /// Load the whisper model through the host's positioned reader
    /// (`globalThis.__fwModelReadAt`), never holding the file as one blob:
    /// the ~large-v3-turbo path. `total_len` is the exact file byte length.
    /// Weights that are f16 in the file stay f16-resident (encoder AND
    /// decoder), which is what fits turbo in the wasm32 address space.
    ///
    /// # Errors
    ///
    /// Reader failures and the same parse/load classes as [`load_model`].
    #[wasm_bindgen]
    pub fn load_whisper_streamed(total_len: f64) -> Result<String, JsValue> {
        fw_stage("whisper:scan");
        let reader = HostReader::new(total_len as u64, |offset, buf| {
            let chunk = fw_model_read_at(offset as f64, buf.len() as f64).map_err(|e| {
                std::io::Error::other(format!(
                    "host read at {offset} (+{}) failed: {e:?}",
                    buf.len()
                ))
            })?;
            if chunk.length() as usize != buf.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "host read at {offset}: wanted {} bytes, got {}",
                        buf.len(),
                        chunk.length()
                    ),
                ));
            }
            chunk.copy_to(buf);
            Ok(())
        });
        let ggml = GgmlModel::load_from_host_reader(reader).map_err(js_err)?;
        fw_stage("whisper:weights");
        let loaded = LoadedModel::from_ggml(ggml).map_err(js_err)?;
        let info = model_info_json(&loaded);
        MODEL.with(|m| *m.borrow_mut() = Some(loaded));
        fw_stage("whisper:ready");
        Ok(info)
    }

    /// Authenticate and load the Sortformer diarization package (conversion
    /// receipt JSON + weights.safetensors bytes) — the same verification
    /// chain the CLI runs (trust-root receipt digest, package sha256 pins,
    /// full tensor census). Returns `{"speaker_lanes": 4}` JSON on success.
    ///
    /// # Errors
    ///
    /// Verification or graph-load failures, naming the failing stage.
    #[wasm_bindgen]
    pub fn load_sortformer(receipt: Vec<u8>, package: Vec<u8>) -> Result<String, JsValue> {
        fw_stage("sortformer:verify");
        let package = crate::sortformer_conformance::load_verified_sortformer_package_from_bytes(
            receipt,
            package,
            &|| Ok(()),
        )
        .map_err(js_err)?;
        fw_stage("sortformer:weights");
        let session = SortformerSession::from_verified_package(&package).map_err(js_err)?;
        DIARIZER.with(|d| *d.borrow_mut() = Some(session));
        fw_stage("sortformer:ready");
        Ok(serde_json::json!({
            "speaker_lanes": crate::sortformer_inference::SORTFORMER_SPEAKER_LANES,
        })
        .to_string())
    }

    /// Decode `audio` (mp3/m4a/wav bytes; `ext` is a lowercase extension
    /// hint) and transcribe it with the loaded model. Returns JSON:
    /// `{language, segments: [{start_sec, end_sec, text, speaker, confidence}],
    ///   dropped_windows}`.
    ///
    /// # Errors
    ///
    /// "no model loaded", audio decode failures, or decode-loop errors —
    /// each as a string naming the failing stage.
    #[wasm_bindgen]
    pub fn transcribe_audio(
        audio: Vec<u8>,
        ext: &str,
        timestamps: bool,
    ) -> Result<String, JsValue> {
        fw_stage("audio:decode");
        let samples = crate::audio_decode::decode_to_16k_mono(audio, ext).map_err(js_err)?;
        MODEL.with(|slot| {
            let slot = slot.borrow();
            let model = slot
                .as_ref()
                .ok_or_else(|| js_err("no model loaded: call load_model first"))?;
            let params = DecodeParams {
                timestamps,
                n_threads: 1,
                ..DecodeParams::default()
            };
            fw_stage("whisper:decode");
            let out =
                decode::transcribe_samples(model, &samples, &params, &|| Ok(())).map_err(js_err)?;
            let result = serde_json::json!({
                "language": out.language,
                "segments": out.segments,
                "dropped_windows": out.dropped_windows.len(),
                "audio_sec": samples.len() as f64 / 16_000.0,
            });
            Ok(result.to_string())
        })
    }

    /// The fused pipeline, mirroring the CLI's `--diarize` flow: whisper
    /// transcription, native Sortformer diarization, the CLI's own
    /// turn-mapping + projection-fusion-v1 (same relocated code, not a
    /// mirror), and merged speaker-attributed runs. Returns JSON:
    /// `{language, segments, turns, speaker_segments, dropped_windows,
    ///   audio_sec}` where `segments[].speaker` is filled by the projection
    /// and `speaker_segments` are the merged per-speaker runs.
    ///
    /// `initial_prompt` is the CLI's `--prompt`: optional context text (for
    /// the playground, the speakers' names and titles) that biases decoding
    /// so names and jargon come out spelled right. `None`/empty is a no-op,
    /// byte-identical to no prompt.
    ///
    /// Projection runs at segment granularity (`word_aligned = false`); the
    /// CLI's word-aligned snap path needs word timestamps, which stay off in
    /// the browser build for now.
    ///
    /// # Errors
    ///
    /// "no model loaded" / "no diarizer loaded", audio decode failures, or
    /// any engine error — each naming the failing stage.
    #[wasm_bindgen]
    pub fn transcribe_and_diarize(
        audio: Vec<u8>,
        ext: &str,
        initial_prompt: Option<String>,
    ) -> Result<String, JsValue> {
        fw_stage("audio:decode");
        let samples = crate::audio_decode::decode_to_16k_mono(audio, ext).map_err(js_err)?;
        let audio_sec = samples.len() as f64 / 16_000.0;

        let out = MODEL.with(|slot| {
            let slot = slot.borrow();
            let model = slot
                .as_ref()
                .ok_or_else(|| js_err("no model loaded: call load_whisper_streamed first"))?;
            let params = DecodeParams {
                timestamps: true,
                n_threads: 1,
                initial_prompt: initial_prompt.filter(|p| !p.trim().is_empty()),
                ..DecodeParams::default()
            };
            fw_stage("whisper:decode");
            decode::transcribe_samples(model, &samples, &params, &|| Ok(())).map_err(js_err)
        })?;

        let diarization = DIARIZER.with(|slot| {
            let slot = slot.borrow();
            let session = slot
                .as_ref()
                .ok_or_else(|| js_err("no diarizer loaded: call load_sortformer first"))?;
            fw_stage("sortformer:diarize");
            session
                .diarize(SortformerPcm::mono_16khz(&samples))
                .map_err(js_err)
        })?;

        fw_stage("fuse:project");
        let (turns, _labels) =
            crate::diarization_projection::sortformer_output_to_diarization_turns(
                &diarization,
                &|| Ok(()),
            )
            .map_err(js_err)?;
        let projection = crate::diarization_projection::project_diarization_onto_segments(
            &out.segments,
            &turns,
            false,
        )
        .map_err(js_err)?;
        let speaker_segments = crate::diarization_projection::build_speaker_attributed_segments(
            &projection.segments,
            &turns,
        );

        fw_stage("done");
        Ok(serde_json::json!({
            "language": out.language,
            "segments": projection.segments,
            "turns": turns,
            "speaker_segments": speaker_segments,
            "mixed_speaker_segment_indices": projection.mixed_speaker_segment_indices,
            "overlap_suspected_segment_indices": projection.overlap_suspected_segment_indices,
            "dropped_windows": out.dropped_windows.len(),
            "audio_sec": audio_sec,
        })
        .to_string())
    }
}
