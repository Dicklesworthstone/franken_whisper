#![feature(portable_simd)]
// `deny` (not `forbid`) so explicitly scoped native-engine kernels can use
// runtime-gated SIMD and fully-overwritten preallocated output buffers. Each
// exception carries a local safety argument; unannotated unsafe code is still
// rejected. The performance evidence lives in docs/NEGATIVE_EVIDENCE.md.
#![deny(unsafe_code)]
#![allow(clippy::needless_raw_string_hashes)]

pub mod accelerate;
pub mod adversarial_corpus;
pub mod audio;
pub mod backend;
pub mod capture;
pub mod cli;
pub mod confidential_evaluation;
pub mod conformance;
pub mod denoise;
pub mod diarization;
pub mod diarization_projection;
pub mod differential_oracle;
pub mod ecapa_conformance;
pub mod ecapa_inference;
pub mod error;
pub mod export;
pub mod listen;
pub mod live_policy;
pub mod logging;
pub mod model;
pub mod model_distribution;
pub mod native_engine;
pub mod orchestrator;
pub mod process;
pub mod public_corpus;
pub mod replay_pack;
pub mod robot;
pub mod separate;
pub mod sortformer_conformance;
pub mod sortformer_f16_contract;
pub mod sortformer_f16_downcast;
pub mod sortformer_identity;
pub mod sortformer_inference;
pub mod speculation;
pub mod storage;
pub mod streaming;
pub mod sync;
pub mod tty_audio;
pub mod tui;
pub mod youtube;

pub use error::{FwError, FwResult};
pub use model::{BackendKind, RunReport, TranscribeRequest, TranscriptionResult};
pub use orchestrator::{FrankenWhisperEngine, PipelineBuilder, PipelineConfig, PipelineStage};

/// Carry an existing caller context across an owned runtime entry or spawn.
/// Capture before entering the runtime: its root context would otherwise hide
/// the caller's restrictions, budget, and cancellation. Install only for each
/// poll and destruction so neither a suspended future nor a worker migration
/// retains a TLS guard, and cancellation cleanup cannot regain worker authority.
/// With no caller, the runtime continues to supply its own context.
fn with_caller_cx<F: std::future::Future>(
    future: F,
) -> impl std::future::Future<Output = F::Output> {
    CallerCxFuture {
        caller: asupersync::Cx::current(),
        future: Some(Box::pin(future)),
    }
}

struct CallerCxFuture<F> {
    caller: Option<asupersync::Cx>,
    // The allocation keeps F pinned while the owner moves between workers.
    // Taking the option ensures F is destroyed before the context guard ends.
    future: Option<std::pin::Pin<Box<F>>>,
}

impl<F: std::future::Future> std::future::Future for CallerCxFuture<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        task_cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        let _guard = asupersync::Cx::set_current(this.caller.clone());
        let outcome = this
            .future
            .as_mut()
            .expect("caller context future polled after completion")
            .as_mut()
            .poll(task_cx);
        if outcome.is_ready() {
            drop(this.future.take());
        }
        outcome
    }
}

impl<F> Drop for CallerCxFuture<F> {
    fn drop(&mut self) {
        let _guard = asupersync::Cx::set_current(self.caller.clone());
        drop(self.future.take());
    }
}

#[cfg(test)]
mod runtime_context_tests {
    use std::future::Future;
    use std::marker::PhantomPinned;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};

    use asupersync::{Budget, Cx, cx::cap, runtime::RuntimeBuilder, runtime::SpawnError};

    #[derive(Clone, Copy)]
    enum PollBehavior {
        Ready,
        Pending,
        Panic,
    }

    struct DropObservedFuture {
        behavior: PollBehavior,
        dropped_contexts: Arc<Mutex<Vec<Option<Cx>>>>,
        _pinned: PhantomPinned,
    }

    impl Future for DropObservedFuture {
        type Output = usize;

        fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<usize> {
            match self.as_ref().get_ref().behavior {
                PollBehavior::Ready => Poll::Ready(42),
                PollBehavior::Pending => Poll::Pending,
                PollBehavior::Panic => panic!("drop context unwind control"),
            }
        }
    }

    impl Drop for DropObservedFuture {
        fn drop(&mut self) {
            self.dropped_contexts.lock().unwrap().push(Cx::current());
        }
    }

    fn assert_same_context(actual: &Cx, expected: &Cx) {
        assert_eq!(actual.task_id(), expected.task_id());
        assert_eq!(actual.region_id(), expected.region_id());
        assert_eq!(actual.budget(), expected.budget());
        assert_eq!(actual.capabilities(), expected.capabilities());
        assert_eq!(
            actual.cancelled_by(asupersync::types::CancelKind::User),
            expected.cancelled_by(asupersync::types::CancelKind::User)
        );
    }

    fn assert_drop_keeps_caller(behavior: PollBehavior, poll: bool) {
        let runtime = RuntimeBuilder::new().worker_threads(1).build().unwrap();
        let parent = runtime.request_cx_with_budget(Budget::INFINITE.with_cost_quota(17));
        let dropped_contexts = Arc::new(Mutex::new(Vec::new()));
        let (future, expected) = {
            let _restricted = parent.restrict::<cap::None>().set_current_restricted();
            let expected = Cx::current().unwrap();
            let future = super::with_caller_cx(DropObservedFuture {
                behavior,
                dropped_contexts: Arc::clone(&dropped_contexts),
                _pinned: PhantomPinned,
            });
            (future, expected)
        };
        let other = runtime.request_cx_with_budget(Budget::INFINITE);
        let _other = Cx::set_current(Some(other.clone()));
        parent.cancel_with(
            asupersync::types::CancelKind::User,
            Some("drop caller cancelled"),
        );

        let observed_in_poll = Arc::clone(&dropped_contexts);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut future = Box::pin(future);
            if poll {
                let outcome = future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()));
                match behavior {
                    PollBehavior::Ready => {
                        assert_eq!(outcome, Poll::Ready(42));
                        assert_eq!(observed_in_poll.lock().unwrap().len(), 1);
                    }
                    PollBehavior::Pending => {
                        assert!(outcome.is_pending());
                        assert!(observed_in_poll.lock().unwrap().is_empty());
                    }
                    PollBehavior::Panic => unreachable!("poll must unwind"),
                }
            }
            // For Pending this is task cancellation; without a poll it models
            // dropping a queued task. Panic exercises automatic unwind cleanup.
            drop(future);
        }));
        assert_eq!(
            result.is_err(),
            poll && matches!(behavior, PollBehavior::Panic)
        );
        assert_same_context(&Cx::current().unwrap(), &other);
        assert!(Cx::current().unwrap().checkpoint().is_ok());
        let contexts = dropped_contexts.lock().unwrap();
        assert_eq!(contexts.len(), 1, "the inner future must drop exactly once");
        let actual = contexts[0].as_ref().expect("destructor retains caller");
        assert_same_context(actual, &expected);
        assert!(!actual.capabilities().spawn);
        assert!(actual.checkpoint().is_err());
    }

    #[test]
    fn runtime_context_drop_after_ready_retains_caller() {
        assert_drop_keeps_caller(PollBehavior::Ready, true);
    }

    #[test]
    fn runtime_context_drop_during_poll_unwind_retains_caller() {
        assert_drop_keeps_caller(PollBehavior::Panic, true);
    }

    #[test]
    fn runtime_context_drop_pending_cancelled_future_retains_caller() {
        assert_drop_keeps_caller(PollBehavior::Pending, true);
    }

    #[test]
    fn runtime_context_drop_before_first_poll_retains_caller() {
        assert_drop_keeps_caller(PollBehavior::Pending, false);
    }

    #[test]
    fn runtime_context_drop_without_caller_keeps_worker_context() {
        assert!(Cx::current().is_none());
        let runtime = RuntimeBuilder::new().worker_threads(1).build().unwrap();
        let worker = runtime.request_cx_with_budget(Budget::INFINITE);
        for behavior in [PollBehavior::Ready, PollBehavior::Pending] {
            let dropped_contexts = Arc::new(Mutex::new(Vec::new()));
            let mut future = Box::pin(super::with_caller_cx(DropObservedFuture {
                behavior,
                dropped_contexts: Arc::clone(&dropped_contexts),
                _pinned: PhantomPinned,
            }));
            let _worker = Cx::set_current(Some(worker.clone()));
            let outcome = future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()));
            assert_eq!(outcome.is_ready(), matches!(behavior, PollBehavior::Ready));
            drop(future);
            let contexts = dropped_contexts.lock().unwrap();
            assert_eq!(contexts.len(), 1);
            assert_same_context(contexts[0].as_ref().unwrap(), &worker);
            assert_same_context(&Cx::current().unwrap(), &worker);
        }
        assert!(Cx::current().is_none());
    }

    #[test]
    fn runtime_context_preserves_restriction_budget_cancel_and_poll_restoration() {
        let runtime = RuntimeBuilder::new().worker_threads(1).build().unwrap();
        let parent = runtime.request_cx_with_budget(Budget::INFINITE.with_cost_quota(17));
        let _parent = Cx::set_current(Some(parent.clone()));
        let mut polls = 0;
        let future = {
            let _restricted = parent.restrict::<cap::None>().set_current_restricted();
            super::with_caller_cx(std::future::poll_fn(|_| {
                polls += 1;
                let cx = Cx::current().expect("held caller context");
                assert_eq!(cx.task_id(), parent.task_id());
                assert_eq!(cx.region_id(), parent.region_id());
                assert_eq!(cx.budget(), parent.budget());
                assert!(!cx.capabilities().spawn);
                assert!(cx.io().is_none());
                assert!(cx.timer_driver().is_none());
                assert!(matches!(
                    cx.spawn_blocking(|_| 42),
                    Err(SpawnError::RuntimeUnavailable)
                ));
                if polls == 1 {
                    assert!(cx.checkpoint().is_ok());
                    Poll::Pending
                } else {
                    assert!(cx.checkpoint().is_err());
                    assert!(cx.cancelled_by(asupersync::types::CancelKind::User));
                    Poll::Ready(42)
                }
            }))
        };
        let other = runtime.request_cx_with_budget(Budget::INFINITE);
        let _other = Cx::set_current(Some(other.clone()));
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert_eq!(Cx::current().unwrap().capabilities(), other.capabilities());
        assert_eq!(Cx::current().unwrap().budget(), other.budget());
        parent.cancel_with(
            asupersync::types::CancelKind::User,
            Some("caller cancelled"),
        );
        assert_eq!(future.as_mut().poll(&mut context), Poll::Ready(42));
        assert_eq!(Cx::current().unwrap().capabilities(), other.capabilities());
        assert!(Cx::current().unwrap().checkpoint().is_ok());

        let panicking = {
            let _restricted = parent.restrict::<cap::None>().set_current_restricted();
            super::with_caller_cx(async {
                assert!(!Cx::current().unwrap().capabilities().spawn);
                panic!("poll unwind control");
            })
        };
        let mut panicking = std::pin::pin!(panicking);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = panicking.as_mut().poll(&mut context);
            }))
            .is_err()
        );
        assert_eq!(Cx::current().unwrap().capabilities(), other.capabilities());
    }

    #[test]
    fn runtime_context_without_caller_keeps_native_runtime_authority() {
        assert!(Cx::current().is_none());
        let runtime = RuntimeBuilder::new()
            .worker_threads(1)
            .blocking_threads(1, 1)
            .build()
            .unwrap();
        let future = super::with_caller_cx(async {
            let cx = Cx::current().expect("native runtime context");
            assert!(cx.timer_driver().is_some());
            let mut work = cx
                .spawn_blocking(|_| 42)
                .expect("native blocking admission");
            asupersync::time::timeout(cx.now(), std::time::Duration::from_secs(5), work.join(&cx))
                .await
                .expect("bounded blocking completion")
                .expect("blocking result")
        });
        assert_eq!(runtime.block_on(future), 42);
        assert!(Cx::current().is_none());
    }

    #[test]
    fn runtime_context_survives_native_spawn_and_reschedule() {
        let runtime = RuntimeBuilder::new().worker_threads(1).build().unwrap();
        let parent = runtime.request_cx_with_budget(Budget::INFINITE.with_cost_quota(17));
        let expected = parent.clone();
        let dropped_contexts = Arc::new(Mutex::new(Vec::new()));
        let drop_observer = DropObservedFuture {
            behavior: PollBehavior::Ready,
            dropped_contexts: Arc::clone(&dropped_contexts),
            _pinned: PhantomPinned,
        };
        let mut polls = 0;
        let (future, expected_drop) = {
            let _restricted = parent.restrict::<cap::None>().set_current_restricted();
            let expected_drop = Cx::current().unwrap();
            let future = super::with_caller_cx(std::future::poll_fn(move |context| {
                let _ = &drop_observer;
                polls += 1;
                let cx = Cx::current().expect("spawn retained caller context");
                assert_eq!(cx.task_id(), expected.task_id());
                assert_eq!(cx.budget(), expected.budget());
                assert!(!cx.capabilities().spawn);
                assert!(matches!(
                    cx.spawn_blocking(|_| 42),
                    Err(SpawnError::RuntimeUnavailable)
                ));
                if polls == 1 {
                    context.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(42)
                }
            }));
            (future, expected_drop)
        };
        let task = runtime.handle().spawn(future);
        let answer = runtime.block_on(async {
            let cx = Cx::current().unwrap();
            asupersync::time::timeout(cx.now(), std::time::Duration::from_secs(5), task)
                .await
                .expect("bounded native task completion")
        });
        assert_eq!(answer, 42);
        assert!(Cx::current().is_none());
        let contexts = dropped_contexts.lock().unwrap();
        assert_eq!(contexts.len(), 1);
        assert_same_context(contexts[0].as_ref().unwrap(), &expected_drop);
    }
}
