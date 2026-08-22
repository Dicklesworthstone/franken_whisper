//! Continuous audio capture sources for the real-time listen driver.
//!
//! This module is the hard prerequisite of the live-mic epic
//! (bd-rt-listen-epic-polh / bd-rt-capture-trait-lckq): until now the only
//! microphone path was batch capture (`audio::capture_microphone` spawns
//! `ffmpeg -t N` into a file). Everything here is *continuous*: a source
//! delivers interleaved `f32` samples for as long as the session runs.
//!
//! Design (locked in the bead):
//! - **Pull model.** The driver loop calls [`CaptureSource::read`] with a
//!   deadline; sources push into an internal SPSC ring from their producer
//!   thread/callback. This keeps the driver single-threaded and testable.
//! - **The cpal callback never blocks and never allocates.** It only writes
//!   into a preallocated lock-free ring ([`ringbuf`]) and bumps atomics.
//! - **Overruns are visible, never silent.** When the ring is full, incoming
//!   audio is dropped *frame-aligned* and counted; the driver surfaces the
//!   cumulative count in `session_stats.capture_overruns`.
//!   Deviation from the bead's "drop-oldest" wording, recorded here on
//!   purpose: a lock-free SPSC producer cannot evict already-queued samples,
//!   so overflow drops the *incoming* (newest) chunk instead. With the
//!   default ~30 s ring this only happens when the consumer stalls for tens
//!   of seconds, at which point which end is dropped is immaterial — the
//!   contract that matters (audio loss is counted and observable) holds.
//! - **Device-native format.** The cpal source captures at the device's own
//!   sample rate/channel count (macOS devices are typically 44.1/48 kHz);
//!   conversion to 16 kHz mono is the streaming resampler's job
//!   (bd-rt-resampler-pbk9), *not* the capture layer's.
//! - **`cpal::Stream` is `!Send`** (macOS), so the stream lives on a
//!   dedicated audio thread; the [`CpalCaptureSource`] handle owns only the
//!   ring consumer plus a stop channel and is therefore `Send`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ringbuf::HeapRb;
use ringbuf::traits::{Consumer as _, Observer as _, Split as _};

use crate::error::{FwError, FwResult};

/// Poll granularity for consumer-side waits. The driver step cadence is
/// hundreds of milliseconds; a 1 ms poll costs nothing measurable and avoids
/// signalling machinery inside the audio callback.
const READ_POLL: Duration = Duration::from_millis(1);

/// Result of one [`CaptureSource::read`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRead {
    /// Frames (samples per channel) written into the caller's buffer.
    /// `0` on timeout is normal, not an error.
    pub frames: usize,
    /// The source has ended (fixture exhausted, stream stopped or errored)
    /// AND no further buffered audio remains.
    pub ended: bool,
    /// Cumulative frames dropped to ring overruns since the source opened.
    pub overrun_frames_dropped: u64,
}

/// A continuous, pull-based audio source delivering interleaved `f32`
/// samples in the source's native sample rate and channel count.
pub trait CaptureSource: Send {
    /// Native sample rate in Hz.
    fn sample_rate(&self) -> u32;
    /// Native channel count (samples are interleaved).
    fn channels(&self) -> u16;
    /// Block up to `max_wait` for ANY data; return what is available.
    ///
    /// Fills `out` with interleaved samples in whole frames only. Returns
    /// `frames: 0` on timeout (not an error). `out.len()` must be at least
    /// one frame (`channels()` samples).
    fn read(&mut self, out: &mut [f32], max_wait: Duration) -> FwResult<CaptureRead>;
    /// Stop the source. Idempotent; `read` after `stop` drains any buffered
    /// audio then reports `ended: true`.
    fn stop(&mut self);
}

// ---------------------------------------------------------------------------
// Frame-aligned ring feeding (extracted so overrun accounting is unit-testable
// without audio hardware).
// ---------------------------------------------------------------------------

/// Push `samples` (interleaved, a whole number of frames) into `producer`,
/// dropping frame-aligned tail data when the ring lacks space. Returns the
/// number of *frames* dropped; the caller accumulates into its counter.
///
/// Frame alignment is preserved by construction: pushes only ever write a
/// multiple of `channels` samples, so the consumer can pop in channel
/// multiples without ever splitting a frame.
fn feed_ring<P: ringbuf::traits::Producer<Item = f32> + ringbuf::traits::Observer>(
    producer: &mut P,
    samples: &[f32],
    channels: usize,
) -> u64 {
    debug_assert!(channels > 0);
    debug_assert_eq!(samples.len() % channels, 0);
    let vacant_frames = producer.vacant_len() / channels;
    let want_frames = samples.len() / channels;
    let write_frames = want_frames.min(vacant_frames);
    let wrote = producer.push_slice(&samples[..write_frames * channels]);
    // `push_slice` on an SPSC ring writes everything that fits; we sized the
    // slice to the vacancy we observed, and the consumer only ever *grows*
    // vacancy concurrently, so the whole slice lands.
    debug_assert_eq!(wrote, write_frames * channels);
    (want_frames - write_frames) as u64
}

/// Rate-limited overrun logging: warn on the first overrun and again each
/// time the cumulative count grows past 10x the last logged value.
fn maybe_log_overruns(total: u64, last_logged: &AtomicU64, context: &str) {
    if total == 0 {
        return;
    }
    let prev = last_logged.load(Ordering::Relaxed);
    if (prev == 0 || total >= prev.saturating_mul(10))
        && last_logged
            .compare_exchange(prev, total, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        tracing::warn!(
            cumulative_frames_dropped = total,
            context,
            "capture ring overrun: audio frames dropped (consumer too slow)"
        );
    }
}

// ---------------------------------------------------------------------------
// cpal hardware source
// ---------------------------------------------------------------------------

/// Shared state between the audio-side producers and the `Send` handle.
struct SharedState {
    overrun_frames: AtomicU64,
    overrun_logged: AtomicU64,
    /// Stream stopped or errored; set by the error callback or teardown.
    ended: AtomicBool,
    /// First stream error message, if any (consumer-side reads only).
    error: std::sync::Mutex<Option<String>>,
}

impl SharedState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            overrun_frames: AtomicU64::new(0),
            overrun_logged: AtomicU64::new(0),
            ended: AtomicBool::new(false),
            error: std::sync::Mutex::new(None),
        })
    }

    fn record_error(&self, message: String) {
        if let Ok(mut slot) = self.error.lock() {
            slot.get_or_insert(message);
        }
        self.ended.store(true, Ordering::Release);
    }

    fn take_error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|mut slot| slot.take())
    }
}

/// Live microphone capture via cpal (CoreAudio / ALSA / WASAPI), in the
/// device's native configuration.
///
/// The `cpal::Stream` is owned by a dedicated audio thread (it is `!Send` on
/// macOS); this handle holds the ring consumer and a stop channel and is
/// `Send`, satisfying [`CaptureSource`].
pub struct CpalCaptureSource {
    consumer: ringbuf::HeapCons<f32>,
    shared: Arc<SharedState>,
    stop_tx: Option<mpsc::Sender<()>>,
    join: Option<std::thread::JoinHandle<()>>,
    sample_rate: u32,
    channels: u16,
}

/// What the audio thread reports back after attempting stream setup.
enum CpalSetup {
    Ready {
        sample_rate: u32,
        channels: u16,
        format: &'static str,
    },
    Failed(String),
}

impl CpalCaptureSource {
    /// Open the default (or named) input device with its native config and a
    /// ring holding `ring_capacity_sec` seconds of audio (driver default:
    /// 30 s via `--capture-buffer-sec`).
    pub fn open(device_name: Option<&str>, ring_capacity_sec: f64) -> FwResult<Self> {
        use cpal::traits::{DeviceTrait as _, HostTrait as _, StreamTrait as _};

        let shared = SharedState::new();
        let (setup_tx, setup_rx) = mpsc::channel::<CpalSetup>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        // The ring is created on the audio thread once the device config is
        // known (capacity depends on the native rate), and the consumer half
        // is shipped back through this channel.
        let (cons_tx, cons_rx) = mpsc::channel::<ringbuf::HeapCons<f32>>();

        let device_name_owned = device_name.map(str::to_owned);
        let shared_audio = Arc::clone(&shared);
        let join = std::thread::Builder::new()
            .name("fw-capture-cpal".into())
            .spawn(move || {
                let host = cpal::default_host();
                let device = match &device_name_owned {
                    None => host.default_input_device(),
                    Some(name) => host.input_devices().ok().and_then(|mut devices| {
                        devices.find(|d| d.name().is_ok_and(|n| n == *name))
                    }),
                };
                let Some(device) = device else {
                    let requested = device_name_owned.as_deref().unwrap_or("<default>");
                    let _ = setup_tx.send(CpalSetup::Failed(format!(
                        "input device not found: {requested}"
                    )));
                    return;
                };
                let supported = match device.default_input_config() {
                    Ok(cfg) => cfg,
                    Err(err) => {
                        let _ = setup_tx
                            .send(CpalSetup::Failed(format!("no usable input config: {err}")));
                        return;
                    }
                };
                let sample_rate = supported.sample_rate().0;
                let channels = supported.channels();
                let sample_format = supported.sample_format();
                let config: cpal::StreamConfig = supported.into();

                let capacity_samples =
                    ((f64::from(sample_rate) * ring_capacity_sec * f64::from(channels)).ceil()
                        as usize)
                        .max(usize::from(channels) * 1024);
                let ring = HeapRb::<f32>::new(capacity_samples);
                let (mut producer, consumer) = ring.split();
                if cons_tx.send(consumer).is_err() {
                    return; // handle construction aborted
                }

                let ch = usize::from(channels);
                let shared_data = Arc::clone(&shared_audio);
                let shared_err = Arc::clone(&shared_audio);
                let err_cb = move |err: cpal::StreamError| {
                    shared_err.record_error(format!("capture stream error: {err}"));
                };

                // One conversion scratch buffer, preallocated at roughly the
                // largest callback size we expect; grown only if a platform
                // hands us more (grow happens outside steady state).
                let mut scratch: Vec<f32> = Vec::with_capacity(capacity_samples.min(65_536));
                macro_rules! build_stream {
                    ($ty:ty, $to_f32:expr) => {
                        device.build_input_stream(
                            &config,
                            move |data: &[$ty], _info: &cpal::InputCallbackInfo| {
                                scratch.clear();
                                if scratch.capacity() < data.len() {
                                    scratch.reserve(data.len() - scratch.capacity());
                                }
                                #[allow(clippy::redundant_closure_call)]
                                scratch.extend(data.iter().map(|&s| ($to_f32)(s)));
                                let aligned = scratch.len() - (scratch.len() % ch);
                                let dropped = feed_ring(&mut producer, &scratch[..aligned], ch);
                                if dropped > 0 {
                                    shared_data
                                        .overrun_frames
                                        .fetch_add(dropped, Ordering::Relaxed);
                                }
                            },
                            err_cb,
                            None,
                        )
                    };
                }
                let stream = match sample_format {
                    cpal::SampleFormat::F32 => build_stream!(f32, |s: f32| s),
                    cpal::SampleFormat::I16 => {
                        build_stream!(i16, |s: i16| f32::from(s) / 32_768.0)
                    }
                    cpal::SampleFormat::U16 => {
                        build_stream!(u16, |s: u16| (f32::from(s) - 32_768.0) / 32_768.0)
                    }
                    other => {
                        let _ = setup_tx.send(CpalSetup::Failed(format!(
                            "unsupported input sample format: {other:?}"
                        )));
                        return;
                    }
                };
                let stream = match stream {
                    Ok(s) => s,
                    Err(err) => {
                        let _ = setup_tx.send(CpalSetup::Failed(format!(
                            "failed to build input stream: {err}"
                        )));
                        return;
                    }
                };
                if let Err(err) = stream.play() {
                    let _ = setup_tx.send(CpalSetup::Failed(format!(
                        "failed to start input stream: {err}"
                    )));
                    return;
                }
                let _ = setup_tx.send(CpalSetup::Ready {
                    sample_rate,
                    channels,
                    format: match sample_format {
                        cpal::SampleFormat::F32 => "f32",
                        cpal::SampleFormat::I16 => "i16",
                        _ => "u16",
                    },
                });

                // Hold the stream until stop is requested (or the handle is
                // dropped, which disconnects the channel).
                let _keep_alive = stream;
                loop {
                    match stop_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                }
                shared_audio.ended.store(true, Ordering::Release);
                tracing::debug!("cpal capture stream torn down");
            })
            .map_err(FwError::Io)?;

        // Wait (bounded) for setup verdict.
        let setup = setup_rx
            .recv_timeout(Duration::from_secs(10))
            .map_err(|_| {
                FwError::BackendUnavailable(
                    "audio capture thread did not report stream setup within 10s".into(),
                )
            })?;
        match setup {
            CpalSetup::Failed(message) => {
                let _ = join.join();
                Err(FwError::BackendUnavailable(format!(
                    "microphone capture unavailable: {message}"
                )))
            }
            CpalSetup::Ready {
                sample_rate,
                channels,
                format,
            } => {
                let consumer = cons_rx.recv().map_err(|_| {
                    FwError::ContractViolation(
                        "capture thread reported Ready but sent no ring consumer".into(),
                    )
                })?;
                tracing::info!(
                    device = device_name.unwrap_or("<default>"),
                    sample_rate,
                    channels,
                    format,
                    ring_capacity_sec,
                    "cpal capture stream open"
                );
                Ok(Self {
                    consumer,
                    shared,
                    stop_tx: Some(stop_tx),
                    join: Some(join),
                    sample_rate,
                    channels,
                })
            }
        }
    }
}

impl CaptureSource for CpalCaptureSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn read(&mut self, out: &mut [f32], max_wait: Duration) -> FwResult<CaptureRead> {
        let ch = usize::from(self.channels).max(1);
        if out.len() < ch {
            return Err(FwError::InvalidRequest(format!(
                "capture read buffer holds {} samples; need at least one frame ({ch})",
                out.len()
            )));
        }
        let usable = out.len() - (out.len() % ch);
        let deadline = Instant::now() + max_wait;
        let mut written = 0usize;
        loop {
            written += self.consumer.pop_slice(&mut out[written..usable]);
            // Return as soon as we have data (whole frames are guaranteed by
            // the producer's alignment), or on deadline, or at stream end.
            let ended_flag = self.shared.ended.load(Ordering::Acquire);
            if written > 0 || Instant::now() >= deadline || ended_flag {
                if let Some(message) = self.shared.take_error() {
                    return Err(FwError::Io(std::io::Error::other(message)));
                }
                let total_overruns = self.shared.overrun_frames.load(Ordering::Relaxed);
                maybe_log_overruns(total_overruns, &self.shared.overrun_logged, "cpal");
                debug_assert_eq!(written % ch, 0);
                return Ok(CaptureRead {
                    frames: written / ch,
                    ended: ended_flag && self.consumer.is_empty(),
                    overrun_frames_dropped: total_overruns,
                });
            }
            std::thread::sleep(READ_POLL.min(deadline.saturating_duration_since(Instant::now())));
        }
    }

    fn stop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        self.shared.ended.store(true, Ordering::Release);
    }
}

impl Drop for CpalCaptureSource {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// Fixture source (deterministic replay for tests / `--source file-replay`)
// ---------------------------------------------------------------------------

/// Replays a PCM buffer either as fast as the consumer pulls (unpaced,
/// deterministic e2e) or paced to real time against an absolute schedule
/// (latency-realistic runs; the schedule is anchored to the first `read`, so
/// wall-clock jitter never accumulates as drift).
pub struct FixtureCaptureSource {
    data: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    /// Next sample index to deliver.
    pos: usize,
    /// Max frames per read in unpaced mode (chunk-size invariance knob).
    chunk_frames: usize,
    /// Real-time pacing schedule anchor; `None` = unpaced.
    pace_start: Option<Instant>,
    paced: bool,
    stopped: bool,
}

impl FixtureCaptureSource {
    /// Unpaced replay delivering at most `chunk_frames` frames per read.
    pub fn new_unpaced(
        data: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        chunk_frames: usize,
    ) -> FwResult<Self> {
        Self::validate(&data, sample_rate, channels)?;
        Ok(Self {
            data,
            sample_rate,
            channels,
            pos: 0,
            chunk_frames: chunk_frames.max(1),
            pace_start: None,
            paced: false,
            stopped: false,
        })
    }

    /// Real-time paced replay: frames become available only once their
    /// schedule position (`frame_index / sample_rate` seconds after the
    /// first read) has passed.
    pub fn new_paced(data: Vec<f32>, sample_rate: u32, channels: u16) -> FwResult<Self> {
        Self::validate(&data, sample_rate, channels)?;
        Ok(Self {
            data,
            sample_rate,
            channels,
            pos: 0,
            chunk_frames: usize::MAX,
            pace_start: None,
            paced: true,
            stopped: false,
        })
    }

    fn validate(data: &[f32], sample_rate: u32, channels: u16) -> FwResult<()> {
        if sample_rate == 0 || channels == 0 {
            return Err(FwError::InvalidRequest(
                "fixture capture requires nonzero sample rate and channels".into(),
            ));
        }
        if !data.len().is_multiple_of(usize::from(channels)) {
            return Err(FwError::InvalidRequest(format!(
                "fixture PCM length {} is not a whole number of {}-channel frames",
                data.len(),
                channels
            )));
        }
        Ok(())
    }

    fn total_frames(&self) -> usize {
        self.data.len() / usize::from(self.channels)
    }

    fn delivered_frames(&self) -> usize {
        self.pos / usize::from(self.channels)
    }
}

impl CaptureSource for FixtureCaptureSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn read(&mut self, out: &mut [f32], max_wait: Duration) -> FwResult<CaptureRead> {
        let ch = usize::from(self.channels);
        if out.len() < ch {
            return Err(FwError::InvalidRequest(format!(
                "capture read buffer holds {} samples; need at least one frame ({ch})",
                out.len()
            )));
        }
        if self.stopped || self.pos >= self.data.len() {
            return Ok(CaptureRead {
                frames: 0,
                ended: true,
                overrun_frames_dropped: 0,
            });
        }
        let out_frames = out.len() / ch;
        let remaining_frames = self.total_frames() - self.delivered_frames();
        let mut allowed = out_frames.min(remaining_frames).min(self.chunk_frames);
        if self.paced {
            let start = *self.pace_start.get_or_insert_with(Instant::now);
            let deadline = Instant::now() + max_wait;
            loop {
                let elapsed = start.elapsed().as_secs_f64();
                let due =
                    ((elapsed * f64::from(self.sample_rate)) as usize).min(self.total_frames());
                let available = due.saturating_sub(self.delivered_frames());
                if available > 0 {
                    allowed = allowed.min(available);
                    break;
                }
                let now = Instant::now();
                if now >= deadline {
                    return Ok(CaptureRead {
                        frames: 0,
                        ended: false,
                        overrun_frames_dropped: 0,
                    });
                }
                std::thread::sleep(READ_POLL.min(deadline - now));
            }
        }
        let samples = allowed * ch;
        out[..samples].copy_from_slice(&self.data[self.pos..self.pos + samples]);
        self.pos += samples;
        Ok(CaptureRead {
            frames: allowed,
            ended: self.pos >= self.data.len(),
            overrun_frames_dropped: 0,
        })
    }

    fn stop(&mut self) {
        self.stopped = true;
    }
}

// ---------------------------------------------------------------------------
// Scriptable mock (test instrument shared by driver/endpoint unit tests)
// ---------------------------------------------------------------------------

/// One scripted behavior step for [`MockCaptureSource`].
#[cfg(test)]
#[derive(Debug, Clone)]
pub enum MockStep {
    /// Deliver these interleaved samples (whole frames).
    Deliver(Vec<f32>),
    /// Report a timeout (no data) for one `read` call.
    Timeout,
    /// Report this many cumulative overrun frames from here on.
    Overruns(u64),
    /// Fail the `read` call with an I/O error carrying this message.
    Fail(String),
    /// End of stream.
    End,
}

/// Deterministic scripted source for unit-testing consumers of
/// [`CaptureSource`] without hardware (bd-rt-capture-trait polish item 4).
#[cfg(test)]
pub struct MockCaptureSource {
    pub script: std::collections::VecDeque<MockStep>,
    pub sample_rate: u32,
    pub channels: u16,
    overruns: u64,
    ended: bool,
}

#[cfg(test)]
impl MockCaptureSource {
    pub fn new(sample_rate: u32, channels: u16, script: Vec<MockStep>) -> Self {
        Self {
            script: script.into(),
            sample_rate,
            channels,
            overruns: 0,
            ended: false,
        }
    }
}

#[cfg(test)]
impl CaptureSource for MockCaptureSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn read(&mut self, out: &mut [f32], _max_wait: Duration) -> FwResult<CaptureRead> {
        loop {
            if self.ended {
                return Ok(CaptureRead {
                    frames: 0,
                    ended: true,
                    overrun_frames_dropped: self.overruns,
                });
            }
            match self.script.pop_front() {
                None | Some(MockStep::End) => {
                    self.ended = true;
                }
                Some(MockStep::Timeout) => {
                    return Ok(CaptureRead {
                        frames: 0,
                        ended: false,
                        overrun_frames_dropped: self.overruns,
                    });
                }
                Some(MockStep::Overruns(count)) => {
                    self.overruns = count;
                }
                Some(MockStep::Fail(message)) => {
                    return Err(FwError::Io(std::io::Error::other(message)));
                }
                Some(MockStep::Deliver(samples)) => {
                    let n = samples
                        .len()
                        .min(out.len() - out.len() % usize::from(self.channels));
                    out[..n].copy_from_slice(&samples[..n]);
                    // Anything the caller's buffer cannot hold is pushed back
                    // for the next read so no scripted audio is lost.
                    if n < samples.len() {
                        self.script
                            .push_front(MockStep::Deliver(samples[n..].to_vec()));
                    }
                    return Ok(CaptureRead {
                        frames: n / usize::from(self.channels),
                        ended: false,
                        overrun_frames_dropped: self.overruns,
                    });
                }
            }
        }
    }

    fn stop(&mut self) {
        self.ended = true;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ringbuf::traits::Producer as _;

    fn stereo_frames(n: usize) -> Vec<f32> {
        // Distinct values per sample so misalignment is detectable.
        (0..n * 2).map(|i| i as f32).collect()
    }

    // ---- feed_ring: overrun accounting + frame alignment ----

    #[test]
    fn feed_ring_counts_dropped_frames_and_preserves_alignment() {
        // Ring holds 4 stereo frames (8 samples).
        let ring = HeapRb::<f32>::new(8);
        let (mut producer, mut consumer) = ring.split();

        // Push 6 frames into space for 4: exactly 2 frames dropped.
        let dropped = feed_ring(&mut producer, &stereo_frames(6), 2);
        assert_eq!(dropped, 2);

        // Everything readable is whole frames, in order, from the FRONT of
        // the pushed data (newest dropped).
        let mut out = [0f32; 8];
        let popped = consumer.pop_slice(&mut out);
        assert_eq!(popped, 8);
        assert_eq!(&out[..], &stereo_frames(4)[..]);
    }

    #[test]
    fn feed_ring_partial_vacancy_never_splits_a_frame() {
        let ring = HeapRb::<f32>::new(8);
        let (mut producer, mut consumer) = ring.split();
        // Occupy 3 samples -> vacancy 5 samples = 2 whole stereo frames.
        assert_eq!(producer.push_slice(&[9.0, 9.0, 9.0]), 3);
        let mut drain = [0f32; 3];
        // Push 3 frames into 5-sample vacancy: 2 fit, 1 dropped.
        let dropped = feed_ring(&mut producer, &stereo_frames(3), 2);
        assert_eq!(dropped, 1);
        assert_eq!(consumer.pop_slice(&mut drain), 3);
        let mut out = [0f32; 8];
        let popped = consumer.pop_slice(&mut out);
        assert_eq!(popped, 4); // exactly 2 whole frames
        assert_eq!(&out[..4], &stereo_frames(2)[..]);
    }

    #[test]
    fn feed_ring_zero_drop_when_space_available() {
        let ring = HeapRb::<f32>::new(64);
        let (mut producer, _consumer) = ring.split();
        assert_eq!(feed_ring(&mut producer, &stereo_frames(10), 2), 0);
    }

    // ---- FixtureCaptureSource: unpaced ----

    #[test]
    fn fixture_unpaced_respects_chunk_size_and_reports_end() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect(); // 100 mono frames
        let mut src = FixtureCaptureSource::new_unpaced(data.clone(), 16_000, 1, 30).unwrap();
        let mut out = [0f32; 64];
        let mut got: Vec<f32> = Vec::new();
        let mut reads = 0;
        loop {
            let r = src.read(&mut out, Duration::from_millis(1)).unwrap();
            got.extend_from_slice(&out[..r.frames]);
            reads += 1;
            assert!(r.frames <= 30, "chunk cap violated: {}", r.frames);
            if r.ended {
                break;
            }
        }
        assert_eq!(got, data);
        assert_eq!(reads, 4); // 30 + 30 + 30 + 10
        // Reads after exhaustion: 0 frames, still ended.
        let r = src.read(&mut out, Duration::from_millis(1)).unwrap();
        assert_eq!((r.frames, r.ended), (0, true));
    }

    #[test]
    fn fixture_rejects_ragged_frames_and_zero_rate() {
        assert!(FixtureCaptureSource::new_unpaced(vec![0.0; 3], 16_000, 2, 8).is_err());
        assert!(FixtureCaptureSource::new_unpaced(vec![0.0; 4], 0, 2, 8).is_err());
    }

    #[test]
    fn fixture_stop_ends_stream_without_hanging() {
        let mut src = FixtureCaptureSource::new_unpaced(vec![0.0; 1000], 16_000, 1, 100).unwrap();
        src.stop();
        let mut out = [0f32; 16];
        let r = src.read(&mut out, Duration::from_millis(1)).unwrap();
        assert_eq!((r.frames, r.ended), (0, true));
    }

    // ---- FixtureCaptureSource: paced (virtual schedule) ----

    #[test]
    fn fixture_paced_delivers_on_schedule_not_all_at_once() {
        // 0.2 s of audio at 1 kHz = 200 frames.
        let mut src = FixtureCaptureSource::new_paced(vec![0.5; 200], 1_000, 1).unwrap();
        let mut out = [0f32; 400];

        // Immediately after start, only a few frames are due.
        let first = src.read(&mut out, Duration::from_millis(30)).unwrap();
        assert!(
            first.frames < 100,
            "paced source delivered {} frames instantly; pacing broken",
            first.frames
        );

        // Drain the rest; total must be exact and take roughly the fixture
        // duration (generous bounds: scheduling, not precision, is under test).
        let started = Instant::now();
        let mut total = first.frames;
        while total < 200 {
            let r = src.read(&mut out, Duration::from_millis(50)).unwrap();
            total += r.frames;
            if r.ended {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "paced drain did not complete in time (got {total}/200)"
            );
        }
        assert_eq!(total, 200);
    }

    #[test]
    fn fixture_paced_timeout_returns_zero_frames_not_error() {
        // 1 frame per second: nothing is due within a 5 ms wait after start.
        let mut src = FixtureCaptureSource::new_paced(vec![0.1; 4], 1, 1).unwrap();
        let mut out = [0f32; 4];
        // Anchor the schedule.
        let _ = src.read(&mut out, Duration::from_millis(1)).unwrap();
        let r = src.read(&mut out, Duration::from_millis(5)).unwrap();
        assert_eq!(r.frames, 0);
        assert!(!r.ended);
    }

    // ---- CaptureRead buffer contract ----

    #[test]
    fn read_rejects_buffer_smaller_than_one_frame() {
        let mut src = FixtureCaptureSource::new_unpaced(vec![0.0; 8], 16_000, 2, 4).unwrap();
        let mut tiny = [0f32; 1];
        let err = src.read(&mut tiny, Duration::from_millis(1)).unwrap_err();
        assert_eq!(err.error_code(), "FW-INVALID-REQUEST");
    }

    // ---- MockCaptureSource ----

    #[test]
    fn mock_source_replays_script_in_order() {
        let mut src = MockCaptureSource::new(
            16_000,
            1,
            vec![
                MockStep::Deliver(vec![1.0, 2.0]),
                MockStep::Timeout,
                MockStep::Overruns(7),
                MockStep::Deliver(vec![3.0]),
                MockStep::End,
            ],
        );
        let mut out = [0f32; 8];

        let r = src.read(&mut out, Duration::ZERO).unwrap();
        assert_eq!((r.frames, r.ended), (2, false));
        assert_eq!(&out[..2], &[1.0, 2.0]);

        let r = src.read(&mut out, Duration::ZERO).unwrap();
        assert_eq!((r.frames, r.ended), (0, false));

        let r = src.read(&mut out, Duration::ZERO).unwrap();
        assert_eq!((r.frames, r.overrun_frames_dropped), (1, 7));
        assert_eq!(out[0], 3.0);

        let r = src.read(&mut out, Duration::ZERO).unwrap();
        assert!(r.ended);
    }

    #[test]
    fn mock_source_error_step_surfaces_as_io_error() {
        let mut src =
            MockCaptureSource::new(16_000, 1, vec![MockStep::Fail("device unplugged".into())]);
        let mut out = [0f32; 4];
        let err = src.read(&mut out, Duration::ZERO).unwrap_err();
        assert_eq!(err.error_code(), "FW-IO");
        assert!(err.to_string().contains("device unplugged"));
    }

    #[test]
    fn mock_source_requeues_undelivered_tail() {
        let mut src = MockCaptureSource::new(
            16_000,
            1,
            vec![MockStep::Deliver((0..10).map(|i| i as f32).collect())],
        );
        let mut small = [0f32; 4];
        let r = src.read(&mut small, Duration::ZERO).unwrap();
        assert_eq!(r.frames, 4);
        let r = src.read(&mut small, Duration::ZERO).unwrap();
        assert_eq!(r.frames, 4);
        assert_eq!(&small[..4], &[4.0, 5.0, 6.0, 7.0]);
    }

    // ---- overrun log rate limiting ----

    #[test]
    fn overrun_logging_is_rate_limited_to_decade_growth() {
        let last = AtomicU64::new(0);
        // First overrun logs (updates the marker)...
        maybe_log_overruns(3, &last, "test");
        assert_eq!(last.load(Ordering::Relaxed), 3);
        // ...small growth does not...
        maybe_log_overruns(25, &last, "test");
        assert_eq!(last.load(Ordering::Relaxed), 3);
        // ...10x growth does.
        maybe_log_overruns(30, &last, "test");
        assert_eq!(last.load(Ordering::Relaxed), 30);
    }

    // ---- live hardware smoke (needs a real input device; run manually) ----

    #[test]
    #[ignore = "requires audio hardware + mic permission; run manually"]
    fn cpal_source_captures_live_audio_smoke() {
        let mut src = CpalCaptureSource::open(None, 5.0).expect("open default input");
        assert!(src.sample_rate() > 0);
        assert!(src.channels() > 0);
        let mut out = vec![0f32; 48_000];
        let mut total = 0usize;
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline && total < 8_000 {
            let r = src
                .read(&mut out, Duration::from_millis(200))
                .expect("read");
            total += r.frames;
        }
        src.stop();
        assert!(total > 0, "no audio frames captured from live device");
    }
}
