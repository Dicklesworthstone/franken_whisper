//! Platform seams for wasm32 portability (bd-m2jm, W1).
//!
//! On native targets every item is a pure re-export of the `std` original, so
//! the native build is byte-for-byte the same code it was before this module
//! existed. On `wasm32` the same names resolve to browser-safe shims:
//!
//! - [`Instant`]: `std::time::Instant::now()` is an opaque trap on
//!   `wasm32-unknown-unknown`. The shim reads a host-fed monotonic clock that
//!   the JS embedding advances via [`set_now_micros`] (`performance.now()`
//!   in the worker's message loop). If the host never feeds it, all spans
//!   read as zero — timing degrades, correctness does not.
//! - [`scope`] / [`available_parallelism`]: browser wasm has no blocking
//!   threads on the main path, so `scope` runs every spawned closure
//!   serially, in spawn order, on the calling "thread". All existing
//!   `scope(|s| { s.spawn(..); .. })` callers already partition work into
//!   independent chunks whose results are order-insensitive joins, so the
//!   serial execution is behaviorally identical (and byte-identical for the
//!   integer/float kernels, which never rely on cross-chunk reduction order
//!   within a scope). `available_parallelism` reports 1.

#[cfg(not(target_arch = "wasm32"))]
pub use std::thread::{Scope, ScopedJoinHandle, available_parallelism, scope};
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
pub use wasm_impl::{
    Instant, Scope, ScopedJoinHandle, available_parallelism, emit_partial_segments, emit_span,
    scope, segment_hook_active, set_now_micros, set_segment_hook, set_span_hook,
};

#[cfg(target_arch = "wasm32")]
mod wasm_impl {
    use std::marker::PhantomData;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static NOW_MICROS: AtomicU64 = AtomicU64::new(0);

    /// Host-fed monotonic clock. The JS embedding calls this with
    /// `performance.now() * 1000.0` (truncated) whenever it re-enters wasm.
    pub fn set_now_micros(micros: u64) {
        // Monotonic guard: never let the clock run backwards.
        NOW_MICROS.fetch_max(micros, Ordering::Relaxed);
    }

    /// Browser-safe stand-in for `std::time::Instant` backed by the host-fed
    /// clock above. Only the methods this crate actually uses are provided.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Instant(u64);

    impl Instant {
        pub fn now() -> Self {
            Instant(NOW_MICROS.load(Ordering::Relaxed))
        }
        pub fn elapsed(&self) -> Duration {
            Duration::from_micros(NOW_MICROS.load(Ordering::Relaxed).saturating_sub(self.0))
        }
        pub fn duration_since(&self, earlier: Instant) -> Duration {
            Duration::from_micros(self.0.saturating_sub(earlier.0))
        }
    }

    /// Serial drop-in for `std::thread::Scope` (the non-atomics lane —
    /// iOS/WebKit gets this build): `spawn` runs the closure immediately on
    /// the caller and stores the result in the handle.
    #[cfg(not(target_feature = "atomics"))]
    pub struct Scope<'scope, 'env: 'scope> {
        _marker: PhantomData<(&'scope mut &'scope (), &'env mut &'env ())>,
    }

    /// Serial drop-in for `std::thread::ScopedJoinHandle`.
    #[cfg(not(target_feature = "atomics"))]
    pub struct ScopedJoinHandle<'scope, T> {
        result: T,
        _marker: PhantomData<&'scope ()>,
    }

    #[cfg(not(target_feature = "atomics"))]
    impl<'scope, T> ScopedJoinHandle<'scope, T> {
        #[allow(clippy::missing_errors_doc)]
        pub fn join(self) -> std::thread::Result<T> {
            Ok(self.result)
        }
    }

    #[cfg(not(target_feature = "atomics"))]
    impl<'scope, 'env> Scope<'scope, 'env> {
        pub fn spawn<F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
        where
            F: FnOnce() -> T + Send + 'scope,
            T: Send + 'scope,
        {
            ScopedJoinHandle {
                result: f(),
                _marker: PhantomData,
            }
        }
    }

    /// Serial drop-in for `std::thread::scope`.
    #[cfg(not(target_feature = "atomics"))]
    pub fn scope<'env, F, T>(f: F) -> T
    where
        F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
    {
        f(&Scope {
            _marker: PhantomData,
        })
    }

    /// One logical core: the serial path.
    #[cfg(not(target_feature = "atomics"))]
    #[allow(clippy::missing_errors_doc, clippy::unnecessary_wraps)]
    pub fn available_parallelism() -> std::io::Result<std::num::NonZeroUsize> {
        Ok(std::num::NonZeroUsize::MIN)
    }

    /// Threaded lane (`+atomics` build, desktop browsers with
    /// crossOriginIsolated): `scope` runs over rayon's Web-Worker-backed
    /// global pool (wasm-bindgen-rayon; the host arms it with
    /// `initThreadPool` after the models hydrate).
    ///
    /// Deferred-batch semantics: `spawn` BANKS the closure (type-erased) and
    /// returns a handle; the first `join` — or scope exit, for handles never
    /// joined — executes every banked closure in parallel via
    /// `rayon::in_place_scope` and fills the result slots. For fork-join
    /// usage (spawn everything, then join everything), which is the only
    /// pattern this engine's scopes use, the observable behavior matches
    /// `std::thread::scope`: every spawned closure has completed by the time
    /// its `join` returns, and all of them by the time `scope` returns. What
    /// this deliberately does NOT support is a spawn that must run
    /// CONCURRENTLY with the scope's own body (producer/consumer through a
    /// channel) — the engine has no such scope, and one would deadlock
    /// loudly, not corrupt data.
    ///
    /// Result types must be `'static` (they cross the type-erasure boundary
    /// as `Box<dyn Any>`); every engine scope returns owned buffers, which
    /// already are.
    #[cfg(target_feature = "atomics")]
    type BankedJob<'env> = Box<dyn FnOnce() + Send + 'env>;

    #[cfg(target_feature = "atomics")]
    type SharedBatch<'env> = std::sync::Arc<std::sync::Mutex<Vec<BankedJob<'env>>>>;

    #[cfg(target_feature = "atomics")]
    type ResultSlot = std::sync::Arc<std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>>>;

    /// Deferred-batch drop-in for `std::thread::Scope` (threaded lane).
    #[cfg(target_feature = "atomics")]
    pub struct Scope<'env> {
        batch: SharedBatch<'env>,
    }

    /// Handle to a banked closure's future result (threaded lane). Carries a
    /// clone of the scope's batch so `join` can trigger execution without a
    /// back-reference; its `'env` parameter keeps it from outliving the data
    /// the banked closures borrow.
    #[cfg(target_feature = "atomics")]
    pub struct ScopedJoinHandle<'env, T> {
        slot: ResultSlot,
        batch: SharedBatch<'env>,
        _t: PhantomData<T>,
    }

    #[cfg(target_feature = "atomics")]
    fn run_batch(batch: &SharedBatch<'_>) {
        let jobs = std::mem::take(
            &mut *batch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        if jobs.is_empty() {
            return;
        }
        rayon::in_place_scope(|ray| {
            for job in jobs {
                ray.spawn(move |_| job());
            }
        });
    }

    #[cfg(target_feature = "atomics")]
    impl<'env, T: std::any::Any + Send> ScopedJoinHandle<'env, T> {
        /// Ensure the banked batch has run, then take this handle's result.
        /// If a concurrent `run_batch` (a handle joined from a pool task)
        /// drained this handle's job but has not finished it, cooperate with
        /// the pool until the slot fills instead of panicking.
        #[allow(clippy::missing_errors_doc)]
        pub fn join(self) -> std::thread::Result<T> {
            let take = || {
                self.slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
            };
            let mut boxed = take();
            if boxed.is_none() {
                run_batch(&self.batch);
                boxed = take();
            }
            while boxed.is_none() {
                std::thread::yield_now();
                boxed = take();
            }
            let boxed = boxed.expect("checked Some above");
            Ok(*boxed
                .downcast::<T>()
                .expect("banked scope job produced a mismatched type"))
        }
    }

    #[cfg(target_feature = "atomics")]
    impl<'env> Scope<'env> {
        pub fn spawn<F, T>(&self, f: F) -> ScopedJoinHandle<'env, T>
        where
            F: FnOnce() -> T + Send + 'env,
            T: std::any::Any + Send,
        {
            let slot: ResultSlot = std::sync::Arc::new(std::sync::Mutex::new(None));
            let job_slot = std::sync::Arc::clone(&slot);
            self.batch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(Box::new(move || {
                    let value: Box<dyn std::any::Any + Send> = Box::new(f());
                    *job_slot
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(value);
                }));
            ScopedJoinHandle {
                slot,
                batch: std::sync::Arc::clone(&self.batch),
                _t: PhantomData,
            }
        }
    }

    /// Threaded drop-in for `std::thread::scope` (see [`Scope`]).
    #[cfg(target_feature = "atomics")]
    pub fn scope<'env, F, T>(f: F) -> T
    where
        F: FnOnce(&Scope<'env>) -> T,
    {
        let scope = Scope {
            batch: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let result = f(&scope);
        // Handles never joined still complete before scope returns, matching
        // std::thread::scope.
        run_batch(&scope.batch);
        result
    }

    /// The armed pool size (1 until the host calls `initThreadPool`).
    #[cfg(target_feature = "atomics")]
    #[allow(clippy::missing_errors_doc)]
    pub fn available_parallelism() -> std::io::Result<std::num::NonZeroUsize> {
        std::num::NonZeroUsize::new(rayon::current_num_threads())
            .ok_or_else(|| std::io::Error::other("rayon pool reported zero threads"))
    }

    /// Host span hook: the wasm embedding registers a callback and every
    /// [`super::super::perf_span`] measurement is forwarded to it — the
    /// engine's progress heartbeat for the page (window counts, tick counts).
    /// Unregistered = the forward is a cheap lock + None check.
    static SPAN_HOOK: std::sync::Mutex<Option<Box<dyn Fn(&str, f64) + Send + Sync>>> =
        std::sync::Mutex::new(None);

    /// Register the host span hook (replaces any previous hook).
    pub fn set_span_hook(hook: Box<dyn Fn(&str, f64) + Send + Sync>) {
        *SPAN_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook);
    }

    /// Forward one span to the registered hook, if any.
    pub fn emit_span(span: &str, ms: f64) {
        if let Some(hook) = SPAN_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            hook(span, ms);
        }
    }

    /// Host partial-transcript hook: the decode loop feeds each finished
    /// window's segments (as JSON) to the page while the run is still going,
    /// so the transcript renders live instead of appearing all at once.
    static SEGMENT_HOOK: std::sync::Mutex<Option<Box<dyn Fn(&str) + Send + Sync>>> =
        std::sync::Mutex::new(None);

    /// Register the host partial-transcript hook (replaces any previous).
    pub fn set_segment_hook(hook: Box<dyn Fn(&str) + Send + Sync>) {
        *SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook);
    }

    /// Whether a hook is registered (decode-loop serialization guard).
    pub fn segment_hook_active() -> bool {
        SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }

    /// Whether a hook is registered (decode-loop serialization guard).
    pub fn segment_hook_active() -> bool {
        SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }

    /// Forward one window's segments (JSON array) to the registered hook.
    pub fn emit_partial_segments(json: &str) {
        if let Some(hook) = SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            hook(json);
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), target_os = "ios"))]
pub use ios_hooks::{
    emit_partial_segments, emit_span, segment_hook_active, set_segment_hook, set_span_hook,
};

/// Desktop twin of the wasm/iOS live-transcript hook registry
/// (bd-rt-segment-hook-vo6p): the per-window partial-segment feed was
/// compiled away on macOS/Linux, which is why a long batch transcription
/// emitted nothing until the very end. Same shape and semantics as the
/// wasm/iOS modules above; the confirm lane and batch progressive output
/// are the desktop consumers. The hook fires on the decode thread, after
/// timestamp rules, before the next window's encode; segment times are
/// window-relative and the caller re-bases them. A panicking hook
/// propagates (it runs on the caller's own decode call stack — a hook
/// panic is a caller bug, not engine state corruption; the transcription
/// cache is only written after a fully successful decode).
#[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios")))]
mod desktop_hooks {
    type SegmentHook = Box<dyn Fn(&str) + Send + Sync>;

    static SEGMENT_HOOK: std::sync::Mutex<Option<SegmentHook>> = std::sync::Mutex::new(None);

    /// Register the process-wide partial-transcript hook (replaces any
    /// previous). Prefer the scoped per-call hook on
    /// `transcribe_samples_with_window_hook` for new desktop callers —
    /// this global exists for wasm/iOS surface parity and single-consumer
    /// processes.
    pub fn set_segment_hook(hook: SegmentHook) {
        *SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook);
    }

    /// Whether a hook is registered (lets the decode loop skip JSON
    /// serialization entirely when nobody is listening — the
    /// zero-cost-when-off contract).
    pub fn segment_hook_active() -> bool {
        SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }

    /// Forward one window's segments (JSON array) to the registered hook.
    pub fn emit_partial_segments(json: &str) {
        if let Some(hook) = SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            hook(json);
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios")))]
pub use desktop_hooks::{emit_partial_segments, segment_hook_active, set_segment_hook};

/// iOS twin of the wasm host hooks (fw-ios, the phone app's FFI crate): the
/// engine's progress heartbeat (`emit_span`) and live-transcript feed
/// (`emit_partial_segments`) forward to callbacks the Swift side installs
/// through the C ABI. Same shape as the wasm hooks above; a separate module
/// (rather than widening `wasm_impl`) so the wasm lane's clock/scope shims
/// never leak into a native build. Unregistered hooks cost one lock + None
/// check, and desktop builds compile none of this — every call site is gated
/// `any(target_arch = "wasm32", target_os = "ios")`, keeping Linux/macOS
/// byte-identical.
#[cfg(all(not(target_arch = "wasm32"), target_os = "ios"))]
mod ios_hooks {
    /// Host span callback: (span name, milliseconds-or-count value).
    pub type SpanHook = Box<dyn Fn(&str, f64) + Send + Sync>;
    /// Host partial-transcript callback: one window's segments as JSON.
    pub type SegmentHook = Box<dyn Fn(&str) + Send + Sync>;

    static SPAN_HOOK: std::sync::Mutex<Option<SpanHook>> = std::sync::Mutex::new(None);

    /// Register the host span hook (replaces any previous hook).
    pub fn set_span_hook(hook: SpanHook) {
        *SPAN_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook);
    }

    /// Forward one span to the registered hook, if any.
    pub fn emit_span(span: &str, ms: f64) {
        if let Some(hook) = SPAN_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            hook(span, ms);
        }
    }

    static SEGMENT_HOOK: std::sync::Mutex<Option<SegmentHook>> = std::sync::Mutex::new(None);

    /// Register the host partial-transcript hook (replaces any previous).
    pub fn set_segment_hook(hook: SegmentHook) {
        *SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hook);
    }

    /// Forward one window's segments (JSON array) to the registered hook.
    pub fn emit_partial_segments(json: &str) {
        if let Some(hook) = SEGMENT_HOOK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            hook(json);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    /// The seam must be a pure re-export on native: `plat::Instant` and
    /// `plat::scope` ARE the std items, not lookalikes.
    #[test]
    fn native_seam_is_std() {
        let t: super::Instant = std::time::Instant::now();
        let _ = t.elapsed();
        let sum: i64 = super::scope(|s| {
            let a = s.spawn(|| 1i64);
            let b = s.spawn(|| 2i64);
            a.join().unwrap() + b.join().unwrap()
        });
        assert_eq!(sum, 3);
        assert!(super::available_parallelism().unwrap().get() >= 1);
    }
}
