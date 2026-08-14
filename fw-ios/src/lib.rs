// Same nightly feature set as the parent crate: nn.rs uses std::simd, which
// is portable by design (NEON on aarch64-apple-ios, AVX natively) — the same
// SIMD source compiles for the phone.
#![feature(portable_simd)]

//! fw-ios — the C ABI the FrankenWhisper iPhone app links against (bd-n6wl).
//!
//! One code path: the modules below are the parent crate's OWN sources,
//! mounted by `#[path]` — not copies — exactly like `fw-wasm/`. The design
//! rules are inherited from franken_ocr's `focr-ios` and frankentts'
//! `ftts-ffi`:
//!
//! * **Errors are values, never panics.** Every fallible edge returns a
//!   status or NULL and leaves a human-readable reason in the calling
//!   thread's error slot. The release profile is `panic = "abort"`, so an
//!   escaping panic would kill the app; every entry point catches unwinds
//!   anyway so debug builds stay sound.
//! * **This crate adds zero numerics.** It stages PCM, calls the same
//!   `transcribe_samples` / `SortformerSession::diarize` /
//!   projection-fusion entry points the CLI and the browser call, and
//!   serializes the result. There is exactly one inference implementation.
//! * **No environment reads at this boundary.** Session-wide engine levers
//!   (`FW_*`) are set by the app via `setenv` before the first engine call,
//!   mirroring the CLI's `OnceLock` contract; nothing here mutates env.
//! * **The whisper model loads from a PATH, not from bytes**, so the app can
//!   enable `FW_STREAM_LOAD` and hold per-tensor pread buffers instead of a
//!   resident 874 MB blob during load.
//!
//! Ownership and threading are specified in `include/fw_ios.h`, which is the
//! document the Swift side actually reads. Keep the two in lockstep.

// The parent crate's error module, verbatim (thiserror + std only).
#[path = "../../src/error.rs"]
pub mod error;

// The same minimal shims fw-wasm uses (mounted from there, not copied): the
// canonical `model` / `model_distribution` modules drag clap/uuid/fs
// machinery an app bundle can never use.
#[path = "../../fw-wasm/src/model.rs"]
pub mod model;
#[path = "../../fw-wasm/src/model_distribution.rs"]
pub mod model_distribution;

// The engine itself: the exact sources the native binary ships.
#[path = "../../src/native_engine/mod.rs"]
pub mod native_engine;

// The diarization stack, same deal — the parent's OWN sources.
#[path = "../../src/diarization_projection.rs"]
pub mod diarization_projection;
#[path = "../../src/sortformer_conformance.rs"]
pub mod sortformer_conformance;
#[path = "../../src/sortformer_inference.rs"]
pub mod sortformer_inference;

// bytes → 16 kHz mono f32 (symphonia; compiled for wasm32 and iOS only —
// its module-level cfg lives in the mounted file).
#[path = "../../fw-wasm/src/audio_decode.rs"]
pub mod audio_decode;

// The neural denoise stage, the parent's own module (FastEnhancer-S via
// ftts-kernels; length-preserving 16 kHz in/out).
#[path = "../../src/denoise.rs"]
pub mod denoise;

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::error::{FwError, FwResult};
use crate::native_engine::decode::{self, DecodeParams, LoadedModel};
use crate::native_engine::ggml::GgmlModel;
use crate::sortformer_inference::{SortformerPcm, SortformerSession};

// ── Exit codes (documented in fw_ios.h; keep the two in lockstep) ──────────

const EXIT_OK: i32 = 0;
const EXIT_GENERIC: i32 = 1;
const EXIT_USAGE: i32 = 2;
const EXIT_MODEL_IO: i32 = 3;
const EXIT_INPUT_DECODE: i32 = 4;
const EXIT_UNSUPPORTED: i32 = 5;
const EXIT_CANCELLED: i32 = 6;

fn code_for(err: &FwError) -> i32 {
    match err {
        FwError::Cancelled(_) => EXIT_CANCELLED,
        FwError::Unsupported(_) => EXIT_UNSUPPORTED,
        FwError::InvalidRequest(_) => EXIT_INPUT_DECODE,
        FwError::Io(_) => EXIT_MODEL_IO,
        _ => EXIT_GENERIC,
    }
}

// ── Error slot ─────────────────────────────────────────────────────────────

thread_local! {
    /// The calling thread's last error. Thread-local so two threads reporting
    /// failures cannot overwrite each other's message, and so no lock is
    /// taken on the error path.
    static LAST_ERROR: RefCell<CString> =
        RefCell::new(CString::new("").expect("empty string has no interior NUL"));
}

fn set_error(message: impl Into<String>) {
    let message = message.into();
    // A NUL inside the message would truncate it silently at the boundary;
    // replace rather than drop the diagnostic entirely.
    let sanitized = message.replace('\0', "?");
    let value = CString::new(sanitized)
        .unwrap_or_else(|_| CString::new("error message contained a NUL").expect("literal"));
    LAST_ERROR.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = value;
        }
    });
}

fn clear_error() {
    LAST_ERROR.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = CString::new("").expect("literal");
        }
    });
}

/// Record `err` (with the failing stage's name) and return its stable exit
/// code, so the Swift side can branch on "cancelled" vs "bad audio" without
/// string matching.
fn fail(stage: &str, err: &FwError) -> i32 {
    set_error(format!("{stage}: {err}"));
    code_for(err)
}

/// Run `body`, converting an unwind into a recorded error plus `fallback`.
fn guarded<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            set_error(format!("internal panic: {message}"));
            fallback
        }
    }
}

// ── Pointer helpers (the audited island) ───────────────────────────────────
//
// Everything that dereferences a caller-supplied pointer goes through here,
// so the `unsafe` surface is a handful of small functions with one SAFETY
// note each rather than a scattering across every entry point.
#[allow(unsafe_code)]
mod ptr_island {
    use super::{CStr, c_char};

    /// Borrow a caller-supplied NUL-terminated UTF-8 string.
    ///
    /// # Safety
    /// `s` must be NULL or a valid pointer to a NUL-terminated string that
    /// stays alive and unmodified for the duration of the call.
    pub(super) unsafe fn opt_str<'a>(s: *const c_char) -> Option<&'a str> {
        if s.is_null() {
            return None;
        }
        // SAFETY: non-null by the check above; the caller's contract
        // (documented in fw_ios.h) guarantees NUL termination and validity.
        unsafe { CStr::from_ptr(s) }.to_str().ok()
    }

    /// Borrow a caller-supplied byte buffer.
    ///
    /// # Safety
    /// `ptr` must be NULL, or point to `len` initialized bytes that stay
    /// alive and unmodified for the duration of the call. A zero `len`
    /// yields an empty slice without dereferencing `ptr`.
    pub(super) unsafe fn opt_bytes<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
        if ptr.is_null() {
            return None;
        }
        if len == 0 {
            return Some(&[]);
        }
        // SAFETY: non-null and len > 0 by the checks above; the caller's
        // contract guarantees `len` initialized bytes valid for the call.
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Borrow a caller-supplied f32 buffer.
    ///
    /// # Safety
    /// Same contract as [`opt_bytes`], element type `f32`.
    pub(super) unsafe fn opt_f32s<'a>(ptr: *const f32, len: usize) -> Option<&'a [f32]> {
        if ptr.is_null() {
            return None;
        }
        if len == 0 {
            return Some(&[]);
        }
        // SAFETY: non-null and len > 0 by the checks above; the caller's
        // contract guarantees `len` initialized floats valid for the call.
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Write `value` through a caller-supplied out-parameter, if non-NULL.
    ///
    /// # Safety
    /// `out` must be NULL or a valid, aligned, writable pointer to a `T`.
    pub(super) unsafe fn write_out<T>(out: *mut T, value: T) -> bool {
        if out.is_null() {
            return false;
        }
        // SAFETY: non-null by the check above; the caller's contract
        // guarantees it is aligned and writable.
        unsafe { out.write(value) };
        true
    }
}

/// Move a Rust `String` across the boundary as a caller-owned `char *`.
/// Returns NULL (and records an error) if the string contains a NUL.
fn into_owned_c_string(value: String) -> *mut c_char {
    match CString::new(value) {
        Ok(c) => c.into_raw(),
        Err(_) => {
            set_error("result contained an interior NUL and could not be returned");
            std::ptr::null_mut()
        }
    }
}

/// Hand a JSON result back through `out_json`, reclaiming it if the slot is
/// NULL (nobody would free it otherwise).
#[allow(unsafe_code)]
fn deliver_json(entry: &str, json: String, out_json: *mut *mut c_char) -> i32 {
    let ptr = into_owned_c_string(json);
    if ptr.is_null() {
        return EXIT_GENERIC;
    }
    // SAFETY: `out_json` is the caller's out-parameter, documented as NULL or
    // a valid writable `char *` slot.
    if !unsafe { ptr_island::write_out(out_json, ptr) } {
        // SAFETY: `ptr` is the CString we just produced above.
        drop(unsafe { CString::from_raw(ptr) });
        set_error(format!("{entry}: out_json was NULL"));
        return EXIT_USAGE;
    }
    EXIT_OK
}

// ── Cancellation ───────────────────────────────────────────────────────────

static CANCEL: AtomicBool = AtomicBool::new(false);

fn checkpoint() -> FwResult<()> {
    if CANCEL.load(Ordering::Relaxed) {
        Err(FwError::Cancelled("cancelled by host".to_string()))
    } else {
        Ok(())
    }
}

/// See `fw_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn fw_request_cancel() {
    CANCEL.store(true, Ordering::Relaxed);
}

/// See `fw_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn fw_reset_cancel() {
    CANCEL.store(false, Ordering::Relaxed);
}

// ── Diagnostics ────────────────────────────────────────────────────────────

/// See `fw_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn fw_last_error_message() -> *const c_char {
    // Returning a pointer into thread-local storage is sound for the
    // documented lifetime ("valid until the next call that replaces it on
    // this thread"): the CString is owned by this thread's slot and is only
    // replaced by another call on this same thread.
    LAST_ERROR.with(|slot| match slot.try_borrow() {
        Ok(slot) => slot.as_ptr(),
        Err(_) => c"".as_ptr(),
    })
}

/// See `fw_ios.h`.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub extern "C" fn fw_version() -> *const c_char {
    static VERSION: OnceLock<CString> = OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(env!("CARGO_PKG_VERSION")).unwrap_or_else(|_| c"0".to_owned()))
        .as_ptr()
}

// ── Progress + live-transcript callbacks ───────────────────────────────────
//
// The engine's per-window heartbeat (`plat::emit_span`, counted by the app's
// progress bar) and per-window segment feed (`plat::emit_partial_segments`,
// the live transcript) forward into C callbacks installed here. On non-iOS
// targets those plat hooks do not exist, so installation is a no-op there —
// the entry points still exist so the host `cargo check` and any future
// Catalyst lane keep one surface.

/// See `fw_ios.h`.
pub type FwProgressFn = extern "C" fn(ctx: *mut c_void, span: *const c_char, value: f64);
/// See `fw_ios.h`.
pub type FwSegmentsFn = extern "C" fn(ctx: *mut c_void, segments_json: *const c_char);

/// A C callback plus its opaque context, parked in a global so the engine's
/// hooks can reach it from whichever thread runs the decode loop.
#[derive(Clone, Copy)]
struct CallbackTarget<F: Copy> {
    func: F,
    ctx: *mut c_void,
}

// SAFETY: `ctx` is never dereferenced by Rust — it is an opaque token handed
// straight back to `func`. The header makes thread-safety the callback's
// obligation ("invoked from the decode thread; must be thread-safe") and
// makes lifetime the installer's obligation ("clear it before releasing
// whatever ctx points at"). Under those documented conditions, moving the
// pair between threads is sound.
#[allow(unsafe_code)]
unsafe impl<F: Copy> Send for CallbackTarget<F> {}
// SAFETY: as above; the target is immutable once installed, so shared
// references carry no additional obligation beyond `Send`.
#[allow(unsafe_code)]
unsafe impl<F: Copy> Sync for CallbackTarget<F> {}

fn progress_slot() -> &'static Mutex<Option<CallbackTarget<FwProgressFn>>> {
    static SLOT: OnceLock<Mutex<Option<CallbackTarget<FwProgressFn>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

fn segments_slot() -> &'static Mutex<Option<CallbackTarget<FwSegmentsFn>>> {
    static SLOT: OnceLock<Mutex<Option<CallbackTarget<FwSegmentsFn>>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Install the plat hooks exactly once; they forward through the slots above
/// so callback replacement never races the decode loop.
#[cfg(target_os = "ios")]
fn ensure_hooks_installed() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        crate::native_engine::plat::set_span_hook(Box::new(|span, ms| {
            // `try_lock`, not `lock`: a progress hook must never be able to
            // block — or deadlock — the run it is reporting on.
            let Ok(slot) = progress_slot().try_lock() else {
                return;
            };
            let Some(target) = *slot else { return };
            let Ok(span) = CString::new(span) else {
                return;
            };
            (target.func)(target.ctx, span.as_ptr(), ms);
        }));
        crate::native_engine::plat::set_segment_hook(Box::new(|json| {
            let Ok(slot) = segments_slot().try_lock() else {
                return;
            };
            let Some(target) = *slot else { return };
            let Ok(json) = CString::new(json) else {
                return;
            };
            (target.func)(target.ctx, json.as_ptr());
        }));
    });
}

#[cfg(not(target_os = "ios"))]
fn ensure_hooks_installed() {}

/// Forward a stage marker to the app's progress callback (the FFI-level
/// analogue of the browser's `fw_stage`; spans emitted by the engine itself
/// arrive through the same callback via the plat hook).
fn emit_stage(name: &str, value: f64) {
    let Ok(slot) = progress_slot().try_lock() else {
        return;
    };
    let Some(target) = *slot else { return };
    let Ok(name) = CString::new(name) else {
        return;
    };
    (target.func)(target.ctx, name.as_ptr(), value);
}

/// See `fw_ios.h`.
///
/// # Safety
/// `ctx` is never dereferenced here — it is an opaque token handed back to
/// `func` — but the caller must keep whatever it points at alive for as long
/// as the callback is installed, and must clear the callback (`func = NULL`)
/// before releasing it. `func` may be invoked from any thread.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_set_progress_callback(func: Option<FwProgressFn>, ctx: *mut c_void) {
    guarded((), || {
        ensure_hooks_installed();
        if let Ok(mut slot) = progress_slot().lock() {
            *slot = func.map(|func| CallbackTarget { func, ctx });
        }
    });
}

/// See `fw_ios.h`.
///
/// # Safety
/// Same contract as [`fw_set_progress_callback`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_set_segments_callback(func: Option<FwSegmentsFn>, ctx: *mut c_void) {
    guarded((), || {
        ensure_hooks_installed();
        if let Ok(mut slot) = segments_slot().lock() {
            *slot = func.map(|func| CallbackTarget { func, ctx });
        }
    });
}

// ── Engine ─────────────────────────────────────────────────────────────────

/// Opaque engine handle: the loaded whisper model, the optional Sortformer
/// diarizer and FastEnhancer denoiser, the PCM staged between `fw_stage_*`
/// and `fw_run_prepared`, and the info JSON whose pointer was handed out for
/// it (valid exactly as long as the engine is).
pub struct FwEngine {
    model: LoadedModel,
    diarizer: Option<SortformerSession>,
    denoiser: Option<crate::denoise::Denoiser>,
    /// Decoded 16 kHz mono PCM staged by `fw_stage_audio_file` /
    /// `fw_stage_pcm` (with the trimmed leading-silence offset in seconds),
    /// consumed by `fw_run_prepared` — split so the app learns the duration
    /// (and the window count its progress bar needs) before the long stage.
    staged: Option<(Vec<f32>, f64)>,
    info: CString,
}

fn model_info_json(model: &LoadedModel) -> String {
    let hp = &model.hparams;
    serde_json::json!({
        "crate_version": env!("CARGO_PKG_VERSION"),
        "n_vocab": hp.n_vocab,
        "n_audio_layer": hp.n_audio_layer,
        "n_text_layer": hp.n_text_layer,
        "n_mels": hp.n_mels,
        "multilingual": hp.n_vocab >= 51_865,
        "default_threads": crate::native_engine::default_threads(),
    })
    .to_string()
}

/// See `fw_ios.h`.
///
/// # Safety
/// `model_path` must be NULL or point to a NUL-terminated UTF-8 string that
/// stays valid for the duration of the call. A non-NULL return is a handle
/// the caller owns and must release exactly once with [`fw_engine_close`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_engine_open(model_path: *const c_char) -> *mut FwEngine {
    guarded(std::ptr::null_mut(), || {
        clear_error();
        // SAFETY: documented as NULL or a valid NUL-terminated string.
        let Some(path) = (unsafe { ptr_island::opt_str(model_path) }) else {
            set_error("fw_engine_open: model_path was NULL or not valid UTF-8");
            return std::ptr::null_mut();
        };
        emit_stage("whisper:scan", 0.0);
        let ggml = match GgmlModel::load(Path::new(path)) {
            Ok(ggml) => ggml,
            Err(err) => {
                fail("fw_engine_open(parse)", &err);
                return std::ptr::null_mut();
            }
        };
        emit_stage("whisper:weights", 0.0);
        let model = match LoadedModel::from_ggml(ggml) {
            Ok(model) => model,
            Err(err) => {
                fail("fw_engine_open(weights)", &err);
                return std::ptr::null_mut();
            }
        };
        emit_stage("whisper:ready", 0.0);
        let info = model_info_json(&model);
        let info = CString::new(info).unwrap_or_else(|_| c"{}".to_owned());
        Box::into_raw(Box::new(FwEngine {
            model,
            diarizer: None,
            denoiser: None,
            staged: None,
            info,
        }))
    })
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL, or a handle returned by [`fw_engine_open`] that has
/// not already been closed. Closing the same handle twice is undefined;
/// closing NULL is an explicit no-op.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_engine_close(engine: *mut FwEngine) {
    if engine.is_null() {
        return;
    }
    guarded((), || {
        // SAFETY: non-null by the check above, and documented as a pointer
        // previously returned by `fw_engine_open` and not yet closed. Taking
        // ownership back into a Box drops the model and its caches.
        drop(unsafe { Box::from_raw(engine) });
    });
}

/// Borrow an engine handle.
///
/// # Safety
/// `engine` must be NULL or a live handle from `fw_engine_open`; access must
/// be serialized by the caller (see fw_ios.h).
#[allow(unsafe_code)]
unsafe fn engine_mut<'a>(engine: *mut FwEngine) -> Option<&'a mut FwEngine> {
    if engine.is_null() {
        return None;
    }
    // SAFETY: non-null by the check above; the caller's contract guarantees
    // the handle is live and access to it is serialized.
    Some(unsafe { &mut *engine })
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle from [`fw_engine_open`]. The
/// returned pointer is owned by the engine and valid only until it is closed.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_engine_info_json(engine: *const FwEngine) -> *const c_char {
    if engine.is_null() {
        return c"{}".as_ptr();
    }
    // SAFETY: non-null by the check above; documented as a live handle.
    unsafe { &*engine }.info.as_ptr()
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle; `receipt_path` / `package_path`
/// must be NULL or valid NUL-terminated UTF-8 strings for the call.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_engine_load_sortformer(
    engine: *mut FwEngine,
    receipt_path: *const c_char,
    package_path: *const c_char,
) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        // SAFETY: documented as NULL or a live engine handle.
        let Some(engine) = (unsafe { engine_mut(engine) }) else {
            set_error("fw_engine_load_sortformer: engine was NULL");
            return EXIT_USAGE;
        };
        // SAFETY: documented as NULL or valid NUL-terminated strings.
        let (Some(receipt_path), Some(package_path)) =
            (unsafe { ptr_island::opt_str(receipt_path) }, unsafe {
                ptr_island::opt_str(package_path)
            })
        else {
            set_error("fw_engine_load_sortformer: a path was NULL or not valid UTF-8");
            return EXIT_USAGE;
        };
        emit_stage("sortformer:verify", 0.0);
        let read = |p: &str| -> FwResult<Vec<u8>> { Ok(std::fs::read(p)?) };
        let receipt = match read(receipt_path) {
            Ok(bytes) => bytes,
            Err(err) => return fail("fw_engine_load_sortformer(receipt)", &err),
        };
        let package = match read(package_path) {
            Ok(bytes) => bytes,
            Err(err) => return fail("fw_engine_load_sortformer(package)", &err),
        };
        let package =
            match crate::sortformer_conformance::load_verified_sortformer_package_from_bytes(
                receipt,
                package,
                &checkpoint,
            ) {
                Ok(package) => package,
                Err(err) => return fail("fw_engine_load_sortformer(verify)", &err),
            };
        emit_stage("sortformer:weights", 0.0);
        match SortformerSession::from_verified_package(&package) {
            Ok(session) => {
                engine.diarizer = Some(session);
                emit_stage("sortformer:ready", 0.0);
                EXIT_OK
            }
            Err(err) => fail("fw_engine_load_sortformer(graph)", &err),
        }
    })
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle; `artifact_path` must be NULL or a
/// valid NUL-terminated UTF-8 string for the call.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_engine_load_denoiser(
    engine: *mut FwEngine,
    artifact_path: *const c_char,
) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        // SAFETY: documented as NULL or a live engine handle.
        let Some(engine) = (unsafe { engine_mut(engine) }) else {
            set_error("fw_engine_load_denoiser: engine was NULL");
            return EXIT_USAGE;
        };
        // SAFETY: documented as NULL or a valid NUL-terminated string.
        let Some(path) = (unsafe { ptr_island::opt_str(artifact_path) }) else {
            set_error("fw_engine_load_denoiser: artifact_path was NULL or not valid UTF-8");
            return EXIT_USAGE;
        };
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) => return fail("fw_engine_load_denoiser(read)", &FwError::Io(err)),
        };
        match crate::denoise::Denoiser::from_bytes(bytes) {
            Ok(denoiser) => {
                engine.denoiser = Some(denoiser);
                EXIT_OK
            }
            Err(err) => fail("fw_engine_load_denoiser(hydrate)", &err),
        }
    })
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle from [`fw_engine_open`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_engine_has_sortformer(engine: *const FwEngine) -> i32 {
    if engine.is_null() {
        return 0;
    }
    // SAFETY: non-null by the check above; documented as a live handle.
    i32::from(unsafe { &*engine }.diarizer.is_some())
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle from [`fw_engine_open`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_engine_has_denoiser(engine: *const FwEngine) -> i32 {
    if engine.is_null() {
        return 0;
    }
    // SAFETY: non-null by the check above; documented as a live handle.
    i32::from(unsafe { &*engine }.denoiser.is_some())
}

// ── Staging ────────────────────────────────────────────────────────────────

/// Denoise (optional) + leading-silence trim + stage. The staged tuple is
/// `(pcm, skipped_leading_sec)`; every emitted timestamp gets the offset
/// added back so output times stay truthful to the ORIGINAL audio.
fn stage_samples(engine: &mut FwEngine, mut samples: Vec<f32>, denoise: bool) -> String {
    let denoised = denoise
        && if let Some(denoiser) = engine.denoiser.as_ref() {
            samples = denoiser.denoise_16k_with_progress(&samples, |completed, total| {
                emit_stage("audio:denoise", completed as f64 / (total.max(1)) as f64);
            });
            true
        } else {
            false
        };

    // Trim leading silence before the model hears anything (whisper
    // hallucinates on a silent first window). Compiled only where the
    // audio_decode module exists; elsewhere (host checks) trim nothing.
    #[cfg(target_os = "ios")]
    let skip = crate::audio_decode::leading_silence_samples(&samples);
    #[cfg(not(target_os = "ios"))]
    let skip = 0usize;
    let offset_sec = skip as f64 / 16_000.0;
    if skip > 0 {
        samples.drain(..skip);
    }
    let audio_sec = samples.len() as f64 / 16_000.0;
    engine.staged = Some((samples, offset_sec));
    serde_json::json!({
        "audio_sec": audio_sec,
        "skipped_leading_sec": offset_sec,
        "denoised": denoised,
    })
    .to_string()
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle (serialized access); `bytes` must
/// be NULL or `len` initialized bytes valid for the call; `ext` NULL or a
/// valid NUL-terminated string; `out_json` NULL or a writable `char *` slot
/// (on success it receives a caller-owned string for [`fw_string_free`]).
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_stage_audio_file(
    engine: *mut FwEngine,
    bytes: *const u8,
    len: usize,
    ext: *const c_char,
    denoise: bool,
    out_json: *mut *mut c_char,
) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        // SAFETY: documented as NULL or a live engine handle.
        let Some(engine) = (unsafe { engine_mut(engine) }) else {
            set_error("fw_stage_audio_file: engine was NULL");
            return EXIT_USAGE;
        };
        // SAFETY: documented as NULL or `len` initialized bytes.
        let Some(bytes) = (unsafe { ptr_island::opt_bytes(bytes, len) }) else {
            set_error("fw_stage_audio_file: bytes was NULL");
            return EXIT_USAGE;
        };
        if bytes.is_empty() {
            set_error("fw_stage_audio_file: bytes was empty");
            return EXIT_INPUT_DECODE;
        }
        // SAFETY: documented as NULL or a valid NUL-terminated string.
        let ext = unsafe { ptr_island::opt_str(ext) }.unwrap_or("");
        #[cfg(target_os = "ios")]
        {
            emit_stage("audio:decode", 0.0);
            let samples = match crate::audio_decode::decode_to_16k_mono(bytes.to_vec(), ext) {
                Ok(samples) => samples,
                Err(err) => return fail("fw_stage_audio_file(decode)", &err),
            };
            let json = stage_samples(engine, samples, denoise);
            deliver_json("fw_stage_audio_file", json, out_json)
        }
        #[cfg(not(target_os = "ios"))]
        {
            let _ = (engine, ext, denoise, out_json);
            fail(
                "fw_stage_audio_file",
                &FwError::Unsupported(
                    "in-crate audio decode is compiled for iOS targets only".to_string(),
                ),
            )
        }
    })
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle (serialized access); `pcm` must be
/// NULL or `len` initialized f32 samples (16 kHz mono, [-1, 1]) valid for the
/// call; `out_json` NULL or a writable `char *` slot (caller-owned result).
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_stage_pcm(
    engine: *mut FwEngine,
    pcm: *const f32,
    len: usize,
    denoise: bool,
    out_json: *mut *mut c_char,
) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        // SAFETY: documented as NULL or a live engine handle.
        let Some(engine) = (unsafe { engine_mut(engine) }) else {
            set_error("fw_stage_pcm: engine was NULL");
            return EXIT_USAGE;
        };
        // SAFETY: documented as NULL or `len` initialized f32 samples.
        let Some(pcm) = (unsafe { ptr_island::opt_f32s(pcm, len) }) else {
            set_error("fw_stage_pcm: pcm was NULL");
            return EXIT_USAGE;
        };
        if pcm.is_empty() {
            set_error("fw_stage_pcm: pcm was empty");
            return EXIT_INPUT_DECODE;
        }
        let json = stage_samples(engine, pcm.to_vec(), denoise);
        deliver_json("fw_stage_pcm", json, out_json)
    })
}

// ── The run ────────────────────────────────────────────────────────────────

/// Options accepted by [`fw_run_prepared`]; every field optional, unknown
/// fields rejected so a typo'd key fails loudly instead of silently running
/// with defaults.
#[derive(serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct RunOptions {
    /// The CLI's `--language`. None/empty/"auto" = auto-detect.
    language: Option<String>,
    /// The CLI's `--prompt`. None/empty is byte-identical to no prompt.
    initial_prompt: Option<String>,
    /// Translate to English instead of transcribing.
    translate: bool,
    /// Run the loaded Sortformer diarizer and project turns onto segments.
    /// Errors if no diarizer was loaded.
    diarize: bool,
    /// Emit timestamp tokens / timed segments. Default true.
    timestamps: Option<bool>,
    /// Real DTW word-level timestamps (adds `words` to the result).
    word_timestamps: bool,
    /// Beam width, clamped [1, 8] by the engine. None/1 = greedy.
    beam_size: Option<usize>,
    /// Thread-count hint; defaults to the engine's host policy.
    n_threads: Option<usize>,
}

fn run_fused(engine: &mut FwEngine, opts: &RunOptions) -> FwResult<String> {
    let (samples, offset_sec) = engine.staged.take().ok_or_else(|| {
        FwError::InvalidRequest("no audio prepared: call fw_stage_* first".to_string())
    })?;
    let audio_sec = samples.len() as f64 / 16_000.0;

    let params = DecodeParams {
        timestamps: opts.timestamps.unwrap_or(true),
        n_threads: opts
            .n_threads
            .unwrap_or_else(crate::native_engine::default_threads),
        translate: opts.translate,
        word_timestamps: opts.word_timestamps,
        beam_size: opts.beam_size,
        initial_prompt: opts.initial_prompt.clone().filter(|p| !p.trim().is_empty()),
        language: opts
            .language
            .clone()
            .filter(|l| !l.trim().is_empty() && l != "auto"),
        ..DecodeParams::default()
    };
    emit_stage("whisper:decode", 0.0);
    let out = decode::transcribe_samples(&engine.model, &samples, &params, &checkpoint)?;

    let mut segments = out.segments;
    let mut word_timings = out.word_timings;
    let mut turns = Vec::new();
    let mut speaker_segments = Vec::new();
    let mut mixed_indices = Vec::new();
    let mut overlap_indices = Vec::new();

    if opts.diarize {
        let session = engine.diarizer.as_ref().ok_or_else(|| {
            FwError::InvalidRequest(
                "diarize requested but no diarizer loaded: call fw_engine_load_sortformer first"
                    .to_string(),
            )
        })?;
        emit_stage("sortformer:diarize", 0.0);
        // The diarizer has no window structure to count, so its heartbeat is
        // the cancellation checkpoint: one throttled span per 256 calls tells
        // the app the stage is alive and moving.
        let ticks = std::sync::atomic::AtomicU64::new(0);
        let diar_checkpoint = move || {
            let n = ticks.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 256 == 0 {
                emit_stage("sortformer_tick", n as f64);
            }
            checkpoint()
        };
        let diarization = session
            .diarize_with_checkpoint(SortformerPcm::mono_16khz(&samples), &diar_checkpoint)?;

        emit_stage("fuse:project", 0.0);
        let (mapped_turns, _labels) =
            crate::diarization_projection::sortformer_output_to_diarization_turns(
                &diarization,
                &checkpoint,
            )?;
        // Projection runs at segment granularity (`word_aligned = false`),
        // matching the browser lane; word timestamps, when requested, ride
        // alongside in `words` without changing speaker attribution.
        let projection = crate::diarization_projection::project_diarization_onto_segments(
            &segments,
            &mapped_turns,
            false,
        )?;
        segments = projection.segments;
        mixed_indices = projection.mixed_speaker_segment_indices;
        overlap_indices = projection.overlap_suspected_segment_indices;
        turns = mapped_turns;
        speaker_segments =
            crate::diarization_projection::build_speaker_attributed_segments(&segments, &turns);
    }

    // Add the trimmed leading-silence offset back to every emitted time, so
    // output timestamps stay truthful to the ORIGINAL audio.
    if offset_sec > 0.0 {
        let offset_ms = (offset_sec * 1_000.0).round() as u64;
        for seg in &mut segments {
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
        if let Some(words) = word_timings.as_mut() {
            for segment_words in words.iter_mut() {
                for word in segment_words.iter_mut() {
                    word.start_sec += offset_sec;
                    word.end_sec += offset_sec;
                }
            }
        }
    }

    // WordTiming has no serde derives in the engine; serialize by hand.
    let words_json = word_timings.map(|per_segment| {
        per_segment
            .into_iter()
            .map(|words| {
                words
                    .into_iter()
                    .map(|w| {
                        serde_json::json!({
                            "text": w.text,
                            "start_sec": w.start_sec,
                            "end_sec": w.end_sec,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });

    emit_stage("done", 0.0);
    Ok(serde_json::json!({
        "language": out.language,
        "segments": segments,
        "turns": turns,
        "speaker_segments": speaker_segments,
        "mixed_speaker_segment_indices": mixed_indices,
        "overlap_suspected_segment_indices": overlap_indices,
        "words": words_json,
        "dropped_windows": out.dropped_windows.len(),
        "audio_sec": audio_sec,
        "skipped_leading_sec": offset_sec,
    })
    .to_string())
}

/// See `fw_ios.h`.
///
/// # Safety
/// `engine` must be NULL or a live handle (serialized access);
/// `options_json` NULL (all defaults) or a valid NUL-terminated UTF-8 JSON
/// object; `out_json` NULL or a writable `char *` slot, which on success
/// receives a caller-owned string to release with [`fw_string_free`].
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_run_prepared(
    engine: *mut FwEngine,
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    guarded(EXIT_GENERIC, || {
        clear_error();
        // SAFETY: documented as NULL or a live engine handle.
        let Some(engine) = (unsafe { engine_mut(engine) }) else {
            set_error("fw_run_prepared: engine was NULL");
            return EXIT_USAGE;
        };
        // SAFETY: documented as NULL or a valid NUL-terminated string.
        let opts = match unsafe { ptr_island::opt_str(options_json) } {
            None => RunOptions::default(),
            Some(raw) if raw.trim().is_empty() => RunOptions::default(),
            Some(raw) => match serde_json::from_str::<RunOptions>(raw) {
                Ok(opts) => opts,
                Err(err) => {
                    set_error(format!("fw_run_prepared: bad options_json: {err}"));
                    return EXIT_USAGE;
                }
            },
        };
        match run_fused(engine, &opts) {
            Ok(json) => deliver_json("fw_run_prepared", json, out_json),
            Err(err) => fail("fw_run_prepared", &err),
        }
    })
}

// ── Memory ─────────────────────────────────────────────────────────────────

/// See `fw_ios.h`.
///
/// # Safety
/// `s` must be NULL, or a pointer this library returned through a `char **`
/// out-parameter and that has not already been freed.
#[unsafe(no_mangle)]
#[allow(unsafe_code)]
pub unsafe extern "C" fn fw_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: documented as a pointer previously returned by this library
    // through a `char **` out-parameter and not yet freed.
    drop(unsafe { CString::from_raw(s) });
}
