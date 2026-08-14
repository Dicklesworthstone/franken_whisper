//! Host-side smoke harness for the fw-ios C ABI (bd-n6wl).
//!
//! `cargo test -p fw-ios` cannot exist (the `#[path]`-mounted engine sources
//! carry `#[cfg(test)]` modules that reference parent-only modules), so this
//! example is the executable proof that the boundary behaves: it drives the
//! exact `extern "C"` surface the Swift app calls, on the host, against the
//! locally cached tiny.en model and the repo's jfk.wav fixture.
//!
//!     cargo run --release --example smoke
//!
//! Model-gated stages print SKIP (and still exit 0) when an artifact is
//! absent; contract stages (NULL safety, error slot, options validation,
//! fail-fast ordering, cancellation) always run. Any violated expectation
//! panics, so a non-zero exit is a real regression.
//!
//! Host caveats this harness respects: `fw_stage_audio_file` is documented
//! as iOS-only (expects code 5 here), and the live-segment / span hooks are
//! `target_os = "ios"` plat seams, so only the FFI-level stage markers are
//! asserted on the progress callback and the segments callback is not.
#![allow(unsafe_code)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use fw_ios::{
    FwEngine, fw_engine_close, fw_engine_has_denoiser, fw_engine_has_sortformer,
    fw_engine_info_json, fw_engine_load_sortformer, fw_engine_open, fw_last_error_message,
    fw_request_cancel, fw_reset_cancel, fw_run_prepared, fw_set_progress_callback,
    fw_stage_audio_file, fw_stage_pcm, fw_string_free, fw_version,
};

static PROGRESS_CALLS: AtomicUsize = AtomicUsize::new(0);

extern "C" fn count_progress(_ctx: *mut c_void, _span: *const c_char, _value: f64) {
    PROGRESS_CALLS.fetch_add(1, Ordering::Relaxed);
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(fw_last_error_message()) }
        .to_string_lossy()
        .into_owned()
}

/// Take a `char **` result, free it, return the parsed JSON.
fn take_json(code: i32, out: *mut c_char, context: &str) -> serde_json::Value {
    assert_eq!(code, 0, "{context}: code {code}, error: {}", last_error());
    assert!(!out.is_null(), "{context}: success but NULL out_json");
    let text = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();
    unsafe { fw_string_free(out) };
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{context}: unparseable JSON ({e})"))
}

/// Minimal RIFF reader for the fixture: 16-bit PCM, 16 kHz, mono → f32.
fn read_wav_16k_mono(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read jfk.wav");
    assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");
    let mut at = 12usize;
    let mut data: Option<&[u8]> = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size =
            u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("chunk size")) as usize;
        let body = &bytes[at + 8..(at + 8 + size).min(bytes.len())];
        if id == b"fmt " {
            let channels = u16::from_le_bytes(body[2..4].try_into().expect("channels"));
            let rate = u32::from_le_bytes(body[4..8].try_into().expect("rate"));
            let bits = u16::from_le_bytes(body[14..16].try_into().expect("bits"));
            assert_eq!(
                (channels, rate, bits),
                (1, 16_000, 16),
                "fixture format drifted"
            );
        } else if id == b"data" {
            data = Some(body);
        }
        at += 8 + size + (size & 1);
    }
    let data = data.expect("no data chunk");
    data.chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
        .collect()
}

fn stage(engine: *mut FwEngine, pcm: &[f32]) -> serde_json::Value {
    let mut out: *mut c_char = std::ptr::null_mut();
    let code = unsafe { fw_stage_pcm(engine, pcm.as_ptr(), pcm.len(), false, &raw mut out) };
    take_json(code, out, "fw_stage_pcm")
}

fn run(engine: *mut FwEngine, options: &str) -> (i32, Option<serde_json::Value>) {
    let options = CString::new(options).expect("options");
    let mut out: *mut c_char = std::ptr::null_mut();
    let code = unsafe { fw_run_prepared(engine, options.as_ptr(), &raw mut out) };
    if code == 0 {
        (0, Some(take_json(code, out, "fw_run_prepared")))
    } else {
        assert!(out.is_null(), "failure must not hand out a result");
        (code, None)
    }
}

fn main() {
    // ── Contract stages (always run) ───────────────────────────────────────
    let version = unsafe { CStr::from_ptr(fw_version()) }.to_string_lossy();
    assert!(!version.is_empty());
    println!("fw-ios {version}");

    // NULL safety: every entry point must refuse, not crash.
    unsafe {
        assert!(fw_engine_open(std::ptr::null()).is_null());
        assert!(
            last_error().contains("fw_engine_open"),
            "error slot names the stage"
        );
        fw_engine_close(std::ptr::null_mut());
        assert_eq!(fw_engine_has_sortformer(std::ptr::null()), 0);
        assert_eq!(fw_engine_has_denoiser(std::ptr::null()), 0);
        let info = CStr::from_ptr(fw_engine_info_json(std::ptr::null()));
        assert_eq!(info.to_str().expect("utf8"), "{}");
        assert_eq!(
            fw_stage_pcm(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                false,
                std::ptr::null_mut()
            ),
            2
        );
        assert_eq!(
            fw_run_prepared(std::ptr::null_mut(), std::ptr::null(), std::ptr::null_mut()),
            2
        );
        assert_eq!(
            fw_engine_load_sortformer(std::ptr::null_mut(), std::ptr::null(), std::ptr::null()),
            2
        );
        fw_string_free(std::ptr::null_mut());
        fw_reset_cancel();
    }
    println!("PASS null-safety + error slot");

    unsafe { fw_set_progress_callback(Some(count_progress), std::ptr::null_mut()) };

    // ── Model-gated stages ─────────────────────────────────────────────────
    let home = std::env::var("HOME").expect("HOME");
    let model = PathBuf::from(&home).join(".cache/franken_whisper/models/ggml-tiny.en.bin");
    let wav = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/native/jfk.wav");
    if !model.is_file() || !wav.is_file() {
        println!("SKIP model-gated stages (tiny.en or jfk.wav missing)");
        return;
    }

    let model_c = CString::new(model.to_string_lossy().into_owned()).expect("path");
    let engine = unsafe { fw_engine_open(model_c.as_ptr()) };
    assert!(!engine.is_null(), "open tiny.en: {}", last_error());
    let info: serde_json::Value = serde_json::from_str(
        unsafe { CStr::from_ptr(fw_engine_info_json(engine)) }
            .to_str()
            .expect("utf8"),
    )
    .expect("info json");
    assert_eq!(
        info["multilingual"],
        serde_json::Value::Bool(false),
        "tiny.en is English-only"
    );
    assert!(
        PROGRESS_CALLS.load(Ordering::Relaxed) >= 3,
        "open must emit whisper:scan/weights/ready stage markers"
    );
    println!("PASS open + info: {info}");

    // File staging is an iOS-only lane; the host must refuse with code 5.
    {
        let bytes = [0u8; 4];
        let mut out: *mut c_char = std::ptr::null_mut();
        let code = unsafe {
            fw_stage_audio_file(
                engine,
                bytes.as_ptr(),
                bytes.len(),
                c"wav".as_ptr(),
                false,
                &raw mut out,
            )
        };
        assert_eq!(code, 5, "host fw_stage_audio_file must be unsupported");
        assert!(out.is_null());
    }

    // Run before staging: named refusal, code 4.
    let (code, _) = run(engine, "");
    assert_eq!(code, 4, "run without staged audio must be 4");
    assert!(last_error().contains("no audio prepared"));

    // Options validation: malformed JSON and unknown keys are usage errors.
    let pcm = read_wav_16k_mono(&wav);
    let staged = stage(engine, &pcm);
    println!("PASS stage: {staged}");
    let (code, _) = run(engine, "{not json");
    assert_eq!(code, 2, "malformed options must be 2");
    let (code, _) = run(engine, r#"{"beam_widthhh": 3}"#);
    assert_eq!(code, 2, "unknown option keys must be rejected");

    // Fail-fast ordering: diarize without a diarizer refuses BEFORE the
    // decode and must NOT consume the staged PCM…
    let (code, _) = run(engine, r#"{"diarize": true}"#);
    assert_eq!(code, 4, "diarize without diarizer must be 4");
    assert!(last_error().contains("no diarizer loaded"));
    // …so this plain run works without re-staging and yields the transcript.
    let (code, result) = run(engine, "{}");
    assert_eq!(code, 0, "plain run: {}", last_error());
    let result = result.expect("result json");
    let transcript = result["segments"]
        .as_array()
        .expect("segments array")
        .iter()
        .map(|s| s["text"].as_str().unwrap_or(""))
        .collect::<String>()
        .to_lowercase();
    assert!(
        transcript.contains("country"),
        "jfk transcript drifted: {transcript:?}"
    );
    assert!(result["diarization_error"].is_null());
    assert!(result["words"].is_null(), "words absent unless requested");
    println!("PASS transcribe: {transcript:?}");

    // Cancellation: sticky flag aborts the next run with code 6.
    stage(engine, &pcm);
    fw_request_cancel();
    let (code, _) = run(engine, "{}");
    assert_eq!(
        code,
        6,
        "cancelled run must be 6, got error: {}",
        last_error()
    );
    fw_reset_cancel();

    // ── Diarizer-gated stage ───────────────────────────────────────────────
    let sf = PathBuf::from(&home)
        .join(".cache/franken_whisper/models/sortformer/sortformer-v2.1-f32-v1");
    let receipt = sf.join("conversion-receipt.json");
    let package = sf.join("weights.safetensors");
    if receipt.is_file() && package.is_file() {
        let receipt_c = CString::new(receipt.to_string_lossy().into_owned()).expect("path");
        let package_c = CString::new(package.to_string_lossy().into_owned()).expect("path");
        let code =
            unsafe { fw_engine_load_sortformer(engine, receipt_c.as_ptr(), package_c.as_ptr()) };
        assert_eq!(code, 0, "sortformer load: {}", last_error());
        assert_eq!(unsafe { fw_engine_has_sortformer(engine) }, 1);

        stage(engine, &pcm);
        let (code, result) = run(engine, r#"{"diarize": true, "word_timestamps": true}"#);
        assert_eq!(code, 0, "diarized run: {}", last_error());
        let result = result.expect("result json");
        assert!(
            result["diarization_error"].is_null(),
            "clean fixture must diarize"
        );
        let turns = result["turns"].as_array().expect("turns").len();
        let runs = result["speaker_segments"]
            .as_array()
            .expect("speaker runs")
            .len();
        let words: usize = result["words"]
            .as_array()
            .expect("words present when requested")
            .iter()
            .map(|w| w.as_array().map_or(0, Vec::len))
            .sum();
        assert!(turns >= 1, "one speaker, at least one turn");
        assert!(runs >= 1, "at least one attributed run");
        assert!(words >= 10, "jfk has ~22 timed words, got {words}");
        println!("PASS diarize+words: {turns} turn(s), {runs} run(s), {words} word(s)");
    } else {
        println!("SKIP diarizer stage (sortformer package missing)");
    }

    unsafe { fw_engine_close(engine) };
    println!("SMOKE OK");
}
