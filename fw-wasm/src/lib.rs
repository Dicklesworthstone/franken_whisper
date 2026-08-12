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

// bytes → 16 kHz mono f32 (symphonia; wasm-only).
pub mod audio_decode;

#[cfg(target_arch = "wasm32")]
mod wasm_api {
    use std::cell::RefCell;

    use wasm_bindgen::prelude::*;

    use crate::native_engine::decode::{self, DecodeParams, LoadedModel};
    use crate::native_engine::ggml::GgmlModel;

    thread_local! {
        // Single-threaded wasm: the loaded model lives in the worker that
        // called `load_model`. All entry points below must run on that worker.
        static MODEL: RefCell<Option<LoadedModel>> = const { RefCell::new(None) };
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

    // console.error without a web-sys dependency.
    #[wasm_bindgen(
        inline_js = "export function fw_console_error(m) { console.error('fw-wasm panic:', m); }"
    )]
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

    /// Parse ggml model bytes (e.g. tiny.en fetched into OPFS) and hold the
    /// loaded model in this worker. Returns hparams JSON on success.
    ///
    /// # Errors
    ///
    /// Any parse/load failure, as a string naming the failing stage.
    #[wasm_bindgen]
    pub fn load_model(bytes: Vec<u8>) -> Result<String, JsValue> {
        let ggml = GgmlModel::from_bytes(bytes).map_err(js_err)?;
        let loaded = LoadedModel::from_ggml(ggml).map_err(js_err)?;
        let hp = &loaded.hparams;
        let info = serde_json::json!({
            "n_vocab": hp.n_vocab,
            "n_audio_layer": hp.n_audio_layer,
            "n_text_layer": hp.n_text_layer,
            "n_mels": hp.n_mels,
            "multilingual": hp.n_vocab >= 51_865,
        })
        .to_string();
        MODEL.with(|m| *m.borrow_mut() = Some(loaded));
        Ok(info)
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
}
