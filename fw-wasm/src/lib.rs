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
#[path = "../../src/sortformer_f16_contract.rs"]
pub mod sortformer_f16_contract;
#[path = "../../src/sortformer_inference.rs"]
pub mod sortformer_inference;

// bytes → 16 kHz mono f32 (symphonia; wasm-only).
pub mod audio_decode;

// The neural denoise stage, the parent's own module (FastEnhancer-S via
// ftts-kernels; length-preserving 16 kHz in/out).
#[path = "../../src/denoise.rs"]
pub mod denoise;

/// `initThreadPool(n)` — the rayon worker-pool constructor, exported ONLY by
/// the threaded lane. The host MUST await it before any engine call in that
/// lane; the serial lane has no such export, which is how the page tells the
/// lanes apart.
#[cfg(all(target_arch = "wasm32", feature = "threads"))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(target_arch = "wasm32")]
mod wasm_api {
    use std::cell::RefCell;

    use wasm_bindgen::prelude::*;

    use crate::native_engine::decode::{self, DecodeParams, LoadedModel};
    use crate::native_engine::ggml::{GgmlModel, HostReader};
    use crate::sortformer_inference::{SortformerPcm, SortformerSession};

    const WASM_DIARIZATION_PROJECTION_SCHEMA_VERSION: &str =
        "wasm-diarization-projection-v1";

    thread_local! {
        // The loaded models live in the worker that called the load entry
        // points; every entry point below runs on that worker.
        static MODEL: RefCell<Option<LoadedModel>> = const { RefCell::new(None) };
        static DIARIZER: RefCell<Option<SortformerSession>> = const { RefCell::new(None) };
        // Decoded 16 kHz mono PCM staged by `decode_audio` (with the trimmed
        // leading-silence offset in seconds), consumed by `run_prepared` —
        // split so the page learns the duration (and the window count its
        // progress bar needs) before the long stage starts.
        static PCM: RefCell<Option<(Vec<f32>, f64)>> = const { RefCell::new(None) };
        static DENOISER: RefCell<Option<crate::denoise::Denoiser>> = const { RefCell::new(None) };
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
    }
    export function fw_span(name, ms) {
        const f = globalThis.__fwSpan;
        if (f) f(name, ms);
    }
    export function fw_segments(json) {
        const f = globalThis.__fwSegments;
        if (f) f(json);
    }")]
    extern "C" {
        #[wasm_bindgen(catch)]
        fn fw_model_read_at(offset: f64, len: f64) -> Result<js_sys::Uint8Array, JsValue>;
        /// Optional host progress hook: names the stage currently running.
        fn fw_stage(name: &str);
        /// Optional host span hook: one engine perf span (name, milliseconds).
        fn fw_span(name: &str, ms: f64);
        /// Optional host live-transcript hook: one window's segments as JSON.
        fn fw_segments(json: &str);
    }

    fn js_err(e: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&e.to_string())
    }

    /// Doctrine: a panic in a tab must never surface as an opaque
    /// `RuntimeError: unreachable`. Installed at module start, along with the
    /// span hook that forwards the engine's perf spans to the host — the
    /// page's progress bar counts `encoder_window` events through it.
    #[wasm_bindgen(start)]
    pub fn start() {
        std::panic::set_hook(Box::new(|info| {
            fw_console_error(&info.to_string());
        }));
        crate::native_engine::plat::set_span_hook(Box::new(|span, ms| fw_span(span, ms)));
        crate::native_engine::plat::set_segment_hook(Box::new(|json| fw_segments(json)));
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

    /// The instantiated linear memory, so the host can verify the threaded
    /// lane really got a `SharedArrayBuffer`-backed memory (the build flags
    /// are a claim; the instantiated memory is the receipt).
    #[wasm_bindgen]
    pub fn wasm_memory() -> JsValue {
        wasm_bindgen::memory()
    }

    /// Rayon's actual worker count right now — a threaded module whose pool
    /// never armed is visible as `1`, never silently serial.
    #[wasm_bindgen]
    pub fn thread_count() -> usize {
        #[cfg(feature = "threads")]
        {
            rayon::current_num_threads()
        }
        #[cfg(not(feature = "threads"))]
        {
            1
        }
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

    // The fused pipeline mirrors the CLI's `--diarize` flow: whisper
    // transcription, native Sortformer diarization, the CLI's own
    // turn-mapping + projection-fusion-v1 (the same relocated code, not a
    // mirror), then merged speaker-attributed runs. The result JSON is
    // `{language, segments, turns, speaker_segments, projection_timeline,
    // dropped_windows, audio_sec, processed_audio_sec}`; `initial_prompt` is
    // the CLI's `--prompt` (None/empty is
    // byte-identical to no prompt). The fixed large-v3-turbo route records
    // decoder attention and uses the same canonical DTW word normalizer as the
    // native backend. Missing/too-short parent timings retain the same
    // canonical fallback units and disable only the word-aligned projection
    // policy; a wholly absent timing set preserves the prior decoder-segment
    // path. Exposed as two stages so the page can build a progress model
    // between them, plus a single-shot compatibility wrapper.

    /// Hydrate the FastEnhancer denoiser from its pinned artifact bytes
    /// (838 KB; the page verifies the sha256 pin before the bytes get here,
    /// and hydration re-validates the tensor geometry).
    ///
    /// # Errors
    ///
    /// Malformed artifact bytes.
    #[wasm_bindgen]
    pub fn load_denoiser(bytes: Vec<u8>) -> Result<(), JsValue> {
        let denoiser = crate::denoise::Denoiser::from_bytes(bytes).map_err(js_err)?;
        DENOISER.with(|slot| *slot.borrow_mut() = Some(denoiser));
        Ok(())
    }

    /// Stage one of the split pipeline: decode `audio` (mp3/m4a/wav bytes) to
    /// 16 kHz mono PCM and hold it for [`run_prepared`]. Returns
    /// `{"audio_sec": …}` so the page can size its progress model (whisper
    /// processes 30-second windows sequentially, so `ceil(audio_sec / 30)` is
    /// the window total its bar counts against) before the long stage starts.
    ///
    /// # Errors
    ///
    /// Audio decode failures, each naming the failing stage.
    #[wasm_bindgen]
    pub fn decode_audio(audio: Vec<u8>, ext: &str, denoise: bool) -> Result<String, JsValue> {
        fw_stage("audio:decode");
        let mut samples = crate::audio_decode::decode_to_16k_mono(audio, ext).map_err(js_err)?;
        // Default neural denoise stage (bd-z6kz): before silence detection,
        // so the trim sees the cleaned signal. Length-preserving; timestamps
        // are untouched. The page's checkbox (default on) is the off switch.
        let denoised = denoise
            && DENOISER.with(|slot| {
                if let Some(denoiser) = slot.borrow().as_ref() {
                    samples = denoiser.denoise_16k_with_progress(&samples, |completed, total| {
                        fw_stage(&format!("audio:denoise:{completed}/{total}"));
                    });
                    true
                } else {
                    false
                }
            });

        // Trim leading silence before the model hears anything (whisper
        // hallucinates on a silent first window); every emitted timestamp
        // gets the offset added back, so output times stay file-truthful.
        let skip = crate::audio_decode::leading_silence_samples(&samples);
        let offset_sec = skip as f64 / 16_000.0;
        if skip > 0 {
            samples.drain(..skip);
        }
        let audio_sec = samples.len() as f64 / 16_000.0;
        PCM.with(|slot| *slot.borrow_mut() = Some((samples, offset_sec)));
        Ok(serde_json::json!({
            "audio_sec": audio_sec,
            "skipped_leading_sec": offset_sec,
            "denoised": denoised,
        })
        .to_string())
    }

    /// Stage two: run the fused whisper + Sortformer + fusion pipeline over
    /// the PCM staged by [`decode_audio`] (consumed by this call).
    ///
    /// # Errors
    ///
    /// "no audio prepared" / "no model loaded" / "no diarizer loaded", or any
    /// engine error, each naming the failing stage.
    #[wasm_bindgen]
    pub fn run_prepared(
        initial_prompt: Option<String>,
        language: Option<String>,
    ) -> Result<String, JsValue> {
        let (samples, offset_sec) = PCM
            .with(|slot| slot.borrow_mut().take())
            .ok_or_else(|| js_err("no audio prepared: call decode_audio first"))?;
        run_fused(&samples, offset_sec, initial_prompt, language)
    }

    /// Compatibility wrapper over [`decode_audio`] + [`run_prepared`] (the
    /// Node harnesses call this single-shot form).
    ///
    /// # Errors
    ///
    /// Same classes as the two stages it wraps.
    #[wasm_bindgen]
    pub fn transcribe_and_diarize(
        audio: Vec<u8>,
        ext: &str,
        initial_prompt: Option<String>,
        language: Option<String>,
        denoise: Option<bool>,
    ) -> Result<String, JsValue> {
        decode_audio(audio, ext, denoise.unwrap_or(true))?;
        run_prepared(initial_prompt, language)
    }

    fn run_fused(
        samples: &[f32],
        offset_sec: f64,
        initial_prompt: Option<String>,
        language: Option<String>,
    ) -> Result<String, JsValue> {
        let processed_audio_sec = samples.len() as f64 / 16_000.0;

        let out = MODEL.with(|slot| {
            let slot = slot.borrow();
            let model = slot
                .as_ref()
                .ok_or_else(|| js_err("no model loaded: call load_whisper_streamed first"))?;
            // `language`: the CLI's --language. None = auto-detect, which can
            // misfire when the first window is music/noise (a real user got a
            // Japanese guess on an English recording); the page exposes a
            // selector so the user can pin it.
            // The 32-layer large-v1/v2/v3 family is ambiguous from hparams
            // alone. The byte loaders do not receive a trusted model name, so
            // keep that generic family on the conservative segment path
            // instead of silently applying large-v3 alignment heads. The
            // deployed large-v3-turbo route is uniquely identified by its
            // four decoder layers and retains DTW.
            let word_timestamps =
                !(model.hparams.n_text_layer == 32 && model.hparams.n_text_state == 1_280);
            let params = DecodeParams {
                timestamps: true,
                word_timestamps,
                n_threads: 1,
                initial_prompt: initial_prompt.filter(|p| !p.trim().is_empty()),
                language: language.filter(|l| !l.trim().is_empty() && l != "auto"),
                ..DecodeParams::default()
            };
            fw_stage("whisper:decode");
            decode::transcribe_samples(model, samples, &params, &|| Ok(())).map_err(js_err)
        })?;

        let diarization = DIARIZER.with(|slot| {
            let slot = slot.borrow();
            let session = slot
                .as_ref()
                .ok_or_else(|| js_err("no diarizer loaded: call load_sortformer first"))?;
            fw_stage("sortformer:diarize");
            // The diarizer has no window structure to count, so its heartbeat
            // is the cancellation checkpoint: one throttled span per 256
            // calls tells the page the stage is alive and moving.
            let ticks = std::sync::atomic::AtomicU64::new(0);
            let checkpoint = move || {
                let n = ticks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n % 256 == 0 {
                    crate::native_engine::plat::emit_span("sortformer_tick", n as f64);
                }
                Ok(())
            };
            session
                .diarize_with_checkpoint(SortformerPcm::mono_16khz(samples), &checkpoint)
                .map_err(js_err)
        })?;

        fw_stage("fuse:project");
        let (turns, _labels) =
            crate::diarization_projection::sortformer_output_to_diarization_turns(
                &diarization,
                &|| Ok(()),
            )
            .map_err(js_err)?;
        let (projection_segments, word_aligned, projection_granularity) = match out
            .word_timings
            .as_ref()
            .filter(|timings| timings.iter().any(|segment| !segment.is_empty()))
        {
            Some(word_timings) => {
                let normalized =
                    crate::diarization_projection::normalize_dtw_projection_units(
                        &out.segments,
                        word_timings,
                        || Ok(()),
                    )
                    .map_err(js_err)?;
                (
                    normalized.segments,
                    normalized.word_aligned_safe,
                    "canonical_dtw_units",
                )
            }
            None => (out.segments.clone(), false, "decoder_segments"),
        };
        let mut projection = crate::diarization_projection::project_diarization_onto_segments(
            &projection_segments,
            &turns,
            word_aligned,
        )
        .map_err(js_err)?;
        let mut turns = turns;
        let mut speaker_segments = crate::diarization_projection::build_speaker_attributed_segments(
            &projection.segments,
            &turns,
        );

        // Add the trimmed leading-silence offset back to every emitted time,
        // so output timestamps stay truthful to the ORIGINAL file.
        if offset_sec > 0.0 {
            let offset_ms = (offset_sec * 1_000.0).round() as u64;
            for seg in &mut projection.segments {
                seg.start_sec = seg.start_sec.map(|s| s + offset_sec);
                seg.end_sec = seg.end_sec.map(|s| s + offset_sec);
            }
            for turn in &mut turns {
                turn.start_ms += offset_ms;
                turn.end_ms += offset_ms;
            }
            for run in &mut speaker_segments {
                run.start_sec = run.start_sec.map(|s| s + offset_sec);
                run.end_sec = run.end_sec.map(|s| s + offset_sec);
            }
        }

        fw_stage("done");
        Ok(serde_json::json!({
            "language": out.language,
            "segments": projection.segments,
            "turns": turns,
            "speaker_segments": speaker_segments,
            "mixed_speaker_segment_indices": projection.mixed_speaker_segment_indices,
            "overlap_suspected_segment_indices": projection.overlap_suspected_segment_indices,
            "projection_timeline": {
                "schema_version": WASM_DIARIZATION_PROJECTION_SCHEMA_VERSION,
                "granularity": projection_granularity,
                "word_aligned_safe": word_aligned,
            },
            "dropped_windows": out.dropped_windows.len(),
            "audio_sec": processed_audio_sec + offset_sec,
            "processed_audio_sec": processed_audio_sec,
            "skipped_leading_sec": offset_sec,
        })
        .to_string())
    }
}
