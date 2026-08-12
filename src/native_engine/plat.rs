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
    Instant, Scope, ScopedJoinHandle, available_parallelism, scope, set_now_micros,
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

    /// Serial drop-in for `std::thread::Scope`: `spawn` runs the closure
    /// immediately on the caller and stores the result in the handle.
    pub struct Scope<'scope, 'env: 'scope> {
        _marker: PhantomData<(&'scope mut &'scope (), &'env mut &'env ())>,
    }

    /// Serial drop-in for `std::thread::ScopedJoinHandle`.
    pub struct ScopedJoinHandle<'scope, T> {
        result: T,
        _marker: PhantomData<&'scope ()>,
    }

    impl<'scope, T> ScopedJoinHandle<'scope, T> {
        #[allow(clippy::missing_errors_doc)]
        pub fn join(self) -> std::thread::Result<T> {
            Ok(self.result)
        }
    }

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
    pub fn scope<'env, F, T>(f: F) -> T
    where
        F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
    {
        f(&Scope {
            _marker: PhantomData,
        })
    }

    /// One logical core: the serial path.
    #[allow(clippy::missing_errors_doc, clippy::unnecessary_wraps)]
    pub fn available_parallelism() -> std::io::Result<std::num::NonZeroUsize> {
        Ok(std::num::NonZeroUsize::MIN)
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
