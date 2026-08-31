#![allow(unsafe_code)]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

static PROGRESS_CALLS: AtomicUsize = AtomicUsize::new(0);
static SEGMENT_CALLS: AtomicUsize = AtomicUsize::new(0);

extern "C" fn self_clearing_progress(_ctx: *mut c_void, _span: *const c_char, _value: f64) {
    PROGRESS_CALLS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: clearing has no context and is explicitly reentrant by contract.
    unsafe { fw_ios::fw_set_progress_callback(None, std::ptr::null_mut()) };
}

extern "C" fn self_clearing_segments(_ctx: *mut c_void, _json: *const c_char) {
    SEGMENT_CALLS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: clearing has no context and is explicitly reentrant by contract.
    unsafe { fw_ios::fw_set_segments_callback(None, std::ptr::null_mut()) };
}

extern "C" fn recursively_emitting_progress(_ctx: *mut c_void, _span: *const c_char, _value: f64) {
    PROGRESS_CALLS.fetch_add(1, Ordering::Relaxed);
    fw_ios::callback_test_emit_progress();
}

extern "C" fn recursively_emitting_segments(_ctx: *mut c_void, _json: *const c_char) {
    SEGMENT_CALLS.fetch_add(1, Ordering::Relaxed);
    fw_ios::callback_test_emit_segments();
}

#[test]
fn callback_registry_is_reentrant_and_recursive_delivery_fails_closed() {
    PROGRESS_CALLS.store(0, Ordering::Relaxed);
    SEGMENT_CALLS.store(0, Ordering::Relaxed);
    let (finished_tx, finished_rx) = mpsc::sync_channel(1);

    std::thread::spawn(move || {
        for _ in 0..10 {
            // SAFETY: both callbacks own no context and clear themselves before return.
            unsafe {
                fw_ios::fw_set_progress_callback(
                    Some(self_clearing_progress),
                    std::ptr::null_mut(),
                );
                fw_ios::fw_set_segments_callback(
                    Some(self_clearing_segments),
                    std::ptr::null_mut(),
                );
            }
            fw_ios::callback_test_emit_progress();
            fw_ios::callback_test_emit_segments();
            // A second emission must observe the cleared slots.
            fw_ios::callback_test_emit_progress();
            fw_ios::callback_test_emit_segments();
        }

        for _ in 0..10 {
            // SAFETY: both callbacks own no context; nested delivery is deliberately hostile.
            unsafe {
                fw_ios::fw_set_progress_callback(
                    Some(recursively_emitting_progress),
                    std::ptr::null_mut(),
                );
                fw_ios::fw_set_segments_callback(
                    Some(recursively_emitting_segments),
                    std::ptr::null_mut(),
                );
            }
            fw_ios::callback_test_emit_progress();
            fw_ios::callback_test_emit_segments();
            // SAFETY: quiesce the context-free callbacks before the next round.
            unsafe {
                fw_ios::fw_set_progress_callback(None, std::ptr::null_mut());
                fw_ios::fw_set_segments_callback(None, std::ptr::null_mut());
            }
        }

        finished_tx
            .send(())
            .expect("test receiver must remain alive");
    });

    finished_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("callback registry operation deadlocked");
    assert_eq!(PROGRESS_CALLS.load(Ordering::Relaxed), 20);
    assert_eq!(SEGMENT_CALLS.load(Ordering::Relaxed), 20);
}
